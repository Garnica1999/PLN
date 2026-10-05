"""18_h1_ruido_m4.py
Fecha: 2026-10-03
Descripcion: Experimento H1 (entrenamiento robusto al ruido del target silver
M4) + protocolo de evaluacion nuevo. El experimento 17 midio ~31.8 % de
issues cleanlab OOF sobre el train canonico del clasificador M4 (16 clases).

Contenido:
  - Protocolo (--protocolo, idempotente): split temporal por ano
    (train <=2011 / val 2012-2017 / test >=2018, ajustado por minimo de 1500
    docs si hiciera falta) y holdout bloqueado (~15 % del train canonico,
    estratificado por clase, seed 42; clases con <10 ejemplos en train dejan
    TODOS sus docs en train). El holdout queda SELLADO: ningun variante lo
    evalua ni entrena con el.
  - H1 (--h1): tres variantes reentrenadas en el train reducido (train
    canonico - holdout) con la configuracion v1 exacta (TF-IDF de lemas
    enmascarados min_df=3, max_features=100000, ngram (1,2), sublinear_tf;
    LogReg con hiperparametros leidos del joblib v1):
      V1r = baseline reentrenado en train reducido.
      V2  = descarte de documentos flagged por cleanlab (find_label_issues
            recalculado sobre las probabilidades OOF del experimento 17,
            intersectadas con el train reducido).
      V3  = pesos por confianza de etiqueta
            (cleanlab.rank.get_label_quality_scores) como sample_weight.
    Seleccion por macro-F1 de espacio fijo en val canonica; test canonico
    ("development test reutilizado") y test temporal se leen UNA sola vez
    para el ganador por val y para V1r como comparador. IC bootstrap 1000/42.

Estado del arte aplicado:
  - Aprendizaje robusto a ruido de etiquetas: Northcutt et al. (2021,
    "Pervasive Label Errors"), cleanlab 2.9.0 con find_label_issues
    (filter_by=prune_by_noise_rate, defaults) y quality scores como pesos.
  - Validacion temporal estricta (particiones cronologicas obligatorias,
    hallazgo E2/B) + holdout bloqueado para la evaluacion final.
  - IC bootstrap percentil al 95 % (Efron y Tibshirani, 1993).

Uso:
    /home/carlos/venv/bin/python scripts/18_h1_ruido_m4.py --todo
    /home/carlos/venv/bin/python scripts/18_h1_ruido_m4.py --protocolo
    /home/carlos/venv/bin/python scripts/18_h1_ruido_m4.py --h1 --jobs 3
    /home/carlos/venv/bin/python scripts/18_h1_ruido_m4.py --smoke --todo
"""

import argparse
import gc
import importlib.util
import json
import logging
import os
import random
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from cleanlab.filter import find_label_issues
from cleanlab.rank import get_label_quality_scores
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# ===== SECCION 1: configuracion =====

RAIZ = Path(__file__).resolve().parent.parent

# Espacio fijo de 16 clases del baseline v1 (orden alfabetico, identico a
# model.classes_ del joblib logreg_enmascarado). Nunca se fusionan clases.
CLASES_M4 = [
    "derechos_colectivos_ambiente",
    "derechos_fundamentales",
    "derechos_sociales_economicos_culturales",
    "elecciones",
    "habitantes_territorio",
    "organismos_control",
    "organizacion_estado",
    "organizacion_territorial",
    "participacion_democratica",
    "principios_fundamentales",
    "proteccion_aplicacion_derechos",
    "rama_ejecutiva",
    "rama_judicial",
    "rama_legislativa",
    "reforma_constitucion",
    "regimen_economico",
]

# Variantes de H1 en orden de preferencia para desempate (mas simple primero).
VARIANTES = ("v1r", "v2", "v3")

# Bloques de la matriz TF-IDF (train reducido + 4 conjuntos de evaluacion).
BLOQUES_MATRIZ = ("train_reducido", "val", "test", "temporal_val", "temporal_test")
# Bloques sobre los que cada variante predice probabilidades.
BLOQUES_PRED = ("val", "test", "temporal_val", "temporal_test")

CONFIG = {
    "semilla": 42,
    "min_docs_temporal": 1500,
    "corte_train_temporal": 2011,
    "corte_val_temporal": 2017,
    "fraccion_holdout": 0.15,
    "umbral_excepcion_holdout": 10,
    "umbral_colapso_estratos": 5,
    "n_bootstrap": 1000,
    "jobs": 3,
    "hilos_por_variante": 8,
    "limite_smoke_total": 500,
    "limites_smoke": {"train": 350, "val": 75, "test": 75},
    "dir_smoke": "/tmp/opencode/h1_smoke",
    "rutas_datos": {
        "lemas": "data/lemas_por_doc.parquet",
        "etiquetas": "data/etiquetas_areas.parquet",
        "splits": "data/splits_m4.parquet",
        "oof_probs": "resultados/m4_oof_probs.npz",
        "modelo_v1": "modelos/m4_baseline/logreg_enmascarado.joblib",
        "baseline_json": "resultados/m4_baseline.json",
        "ruido_json": "resultados/m4_ruido_etiquetas.json",
    },
    "rutas_protocolo": {
        "temporal": "data/splits_m4_temporal.parquet",
        "holdout": "data/holdout_m4_bloqueado.parquet",
        "protocolo_json": "resultados/protocolo_evaluacion_m4.json",
    },
    "rutas_h1": {
        "matriz": "resultados/m4_h1_matriz.npz",
        "preds": "resultados/m4_h1_{variante}_preds.npz",
        "json": "resultados/m4_h1_ruido_v1.json",
        "figura": "figuras/fig_h1_ruido_m4.png",
        "log": "experimentos/18_h1_ruido_m4.log",
    },
}

# Referencia v1 congelada (sin reentrenar): test canonico del joblib v1.
# Se contrasta contra los JSON canonicos cuando estan disponibles.
REFERENCIA_V1 = {
    "macro_f1_observado": 0.2870081504940973,
    "macro_f1_espacio_fijo": 0.2690701410882162,
    "fuente": "resultados/m4_baseline.json + resultados/m4_ruido_etiquetas.json",
}

# Premisas de diseno congeladas en el JSON de H1 (trazabilidad).
PREMISAS = [
    (
        "Holdout bloqueado SELLADO: data/holdout_m4_bloqueado.parquet se construye desde el "
        "train canonico (data/splits_m4.parquet) y NINGUN variante de H1 lo evalua ni entrena "
        "con el; solo se usa para excluir sus ids del train reducido. La evaluacion final sobre "
        "el holdout queda reservada y fuera del alcance de este script."
    ),
    (
        "Test canonico reutilizado de desarrollo: el baseline v1 ya reporto su macro-F1 sobre "
        "el test canonico (0.287008 observado / 0.269070 espacio fijo); por eso la lectura de "
        "test de H1 se marca 'development_test_reutilizado' y se limita UNA sola vez al ganador "
        "por val y a V1r como comparador."
    ),
    (
        "Test temporal (split por ano, docs 2018-2021) con UNA sola lectura para el ganador por "
        "val y V1r; el resto de variantes solo se evalua en val canonica (seleccion)."
    ),
    (
        "El split temporal es una particion alternativa del MISMO corpus, no un holdout "
        "cronologico limpio: una fraccion de los docs de test temporal pertenece al train "
        "canonico y fue vista en entrenamiento. El solapamiento exacto con el train reducido se "
        "registra en conteos.temporal_test_en_train_reducido y conteos.temporal_val_en_train_reducido; "
        "una lectura temporal estrictamente out-of-sample exigiria reentrenar solo con docs "
        "pre-2018, fuera del alcance de H1."
    ),
    (
        "Vectorizador TF-IDF ajustado SOLO sobre el train reducido (sin holdout), replicando la "
        "configuracion exacta del escenario enmascarado v1 de 06_baseline_clasico.py: lemas "
        "enmascarados, min_df=3, max_features=100000, ngram_range=(1,2), sublinear_tf=True. "
        "El split canonico de train/val/test no se rehace ni se modifica."
    ),
    (
        "V2/V3 usan las probabilidades OOF del experimento 17 (resultados/m4_oof_probs.npz, "
        "out-of-sample sobre el train canonico) intersectadas por id con el train reducido; "
        "los flags de cleanlab se recalculan sobre ese subconjunto (no se reutilizan del JSON "
        "17) y, si faltan clases en el subconjunto, se restringe a las presentes y se "
        "renormalizan las filas (patron del experimento 17)."
    ),
    (
        "V3 usa get_label_quality_scores de cleanlab (metodo por defecto self_confidence) como "
        "sample_weight de LogisticRegression; V2 descarta los documentos flagged por "
        "find_label_issues (defaults: filter_by=prune_by_noise_rate, frac_noise=1.0)."
    ),
    (
        "Metricas en espacio fijo de 16 clases (zero_division=0); macro_f1_espacio_fijo es el "
        "criterio de seleccion en val canonica. IC bootstrap percentil 95 % con 1000 "
        "remuestreos de casos y semilla 42."
    ),
    (
        "Referencia v1 incluida sin reentrenar: valores leidos de los JSON canonicos "
        "(m4_baseline.json y m4_ruido_etiquetas.json); el joblib v1 solo se usa para leer sus "
        "hiperparametros, nunca para predecir en H1."
    ),
    (
        "La matriz intermedia resultados/m4_h1_matriz.npz se construye una sola vez en el "
        "orquestador (del/gc antes de lanzar subprocesos) y se reutiliza salvo --forzar; cada "
        "variante corre en un subproceso con OMP/OPENBLAS/MKL/NUMEXPR=8 hilos y es resumible "
        "via resultados/m4_h1_{variante}_preds.npz."
    ),
    (
        "Modo smoke: subconjunto estratificado de ~500 docs para matriz y modelado (un solo "
        "ajuste por variante, equivalente a 1 fold), protocolo calculado sobre el corpus "
        "completo con las tablas ligeras de etiquetas/splits y salidas SOLO en "
        "/tmp/opencode/h1_smoke/."
    ),
]

LOGGER = logging.getLogger("h1_ruido_m4_18")

# Caches de modulos con nombre que empieza por digito (importlib por ruta).
_MODULO_06 = None
_MODULO_17 = None


# ===== SECCION 2: utilidades (logger, md5, versiones, modulos, guardas) =====


def configurar_logger(ruta_log: str, solo_consola: bool = False) -> logging.Logger:
    """Configura el logger del experimento (archivo append + consola).

    Args:
        ruta_log: Ruta del archivo de log (se crea el directorio padre).
        solo_consola: True para subprocesos (la salida se canaliza al log del
            orquestador con prefijo, evitando escrituras concurrentes).

    Returns:
        Logger configurado.
    """
    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers.clear()
    formato = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    if not solo_consola:
        Path(ruta_log).parent.mkdir(parents=True, exist_ok=True)
        manejador_archivo = logging.FileHandler(ruta_log, mode="a", encoding="utf-8")
        manejador_archivo.setFormatter(formato)
        LOGGER.addHandler(manejador_archivo)
    manejador_consola = logging.StreamHandler()
    manejador_consola.setFormatter(formato)
    LOGGER.addHandler(manejador_consola)
    LOGGER.propagate = False
    return LOGGER


def _cargar_modulo(nombre: str, ruta: Path):
    """Importa un script por ruta (su nombre empieza por digito).

    Args:
        nombre: Nombre interno para sys.modules.
        ruta: Ruta del script a importar.

    Returns:
        Modulo ejecutado.

    Raises:
        ImportError: Si la ruta no puede cargarse.
    """
    spec = importlib.util.spec_from_file_location(nombre, ruta)
    if spec is None or spec.loader is None:
        raise ImportError(f"no se pudo cargar {ruta}")
    modulo = importlib.util.module_from_spec(spec)
    sys.modules[nombre] = modulo
    spec.loader.exec_module(modulo)
    return modulo


def modulo_17():
    """Carga (y cachea) scripts/17_auditoria_etiquetas_m4.py.

    Returns:
        Modulo 17 ejecutado.
    """
    global _MODULO_17
    if _MODULO_17 is None:
        _MODULO_17 = _cargar_modulo(
            "auditoria_m4_17", RAIZ / "scripts" / "17_auditoria_etiquetas_m4.py"
        )
    return _MODULO_17


def modulo_06():
    """Carga scripts/06_baseline_clasico.py reutilizando el cargador de 17.

    Returns:
        Modulo 06 ejecutado.
    """
    global _MODULO_06
    if _MODULO_06 is None:
        _MODULO_06 = modulo_17().cargar_modulo_baseline()
    return _MODULO_06


def rutas_activas(smoke: bool) -> dict:
    """Construye el mapa de rutas de entrada/salida segun el modo smoke.

    Args:
        smoke: True para usar salidas exclusivamente en /tmp/opencode/h1_smoke.

    Returns:
        Dict combinado de rutas de datos, protocolo y H1.
    """
    rutas = dict(CONFIG["rutas_datos"])
    if smoke:
        base = Path(CONFIG["dir_smoke"])
        rutas.update(
            {
                "temporal": str(base / "splits_m4_temporal.parquet"),
                "holdout": str(base / "holdout_m4_bloqueado.parquet"),
                "protocolo_json": str(base / "protocolo_evaluacion_m4.json"),
                "matriz": str(base / "m4_h1_matriz.npz"),
                "preds": str(base / "m4_h1_{variante}_preds.npz"),
                "json": str(base / "m4_h1_ruido_v1.json"),
                "figura": str(base / "fig_h1_ruido_m4.png"),
                "log": str(base / "18_h1_ruido_m4.log"),
            }
        )
    else:
        rutas.update(CONFIG["rutas_protocolo"])
        rutas.update(CONFIG["rutas_h1"])
    return rutas


def verificar_entradas(rutas: dict, requiere_protocolo: bool, logger: logging.Logger) -> None:
    """Aborta si falta alguna entrada canonica (solo lectura).

    Args:
        rutas: Dict de rutas activas.
        requiere_protocolo: True si se exigen los artefactos del protocolo.
        logger: Logger activo.

    Raises:
        FileNotFoundError: Si falta alguna entrada requerida.
    """
    claves = ["lemas", "etiquetas", "splits", "oof_probs", "modelo_v1"]
    if requiere_protocolo:
        claves += ["temporal", "holdout", "protocolo_json"]
    faltantes = [rutas[clave] for clave in claves if not Path(rutas[clave]).exists()]
    if faltantes:
        raise FileNotFoundError(
            "faltan entradas requeridas (ejecute --protocolo primero si aplica): "
            + ", ".join(faltantes)
        )
    logger.info("entradas verificadas: %d archivos", len(claves))


def md5_archivo(ruta: str) -> str | None:
    """Delega el md5 por bloques en el modulo 17 (reutilizacion).

    Args:
        ruta: Ruta del archivo.

    Returns:
        Hexdigest md5 o None si no existe.
    """
    return modulo_17().md5_archivo(ruta)


def versiones_entorno() -> dict:
    """Delega el inventario de versiones en el modulo 17 (reutilizacion).

    Returns:
        Dict con python, plataforma y versiones de librerias clave.
    """
    return modulo_17().versiones_entorno()


def escribir_parquet(ruta: str, tabla: pa.Table, permitir: bool, descripcion: str) -> None:
    """Escribe un parquet nuevo (o lo reescribe solo si esta permitido).

    Args:
        ruta: Ruta de salida.
        tabla: Tabla pyarrow a escribir.
        permitir: True para permitir reescritura (smoke en scratch).
        descripcion: Etiqueta para el log.

    Raises:
        FileExistsError: Si el archivo existe y no se permite reescribir.
    """
    destino = Path(ruta)
    if destino.exists() and not permitir:
        raise FileExistsError(f"{descripcion} ya existe (no se sobrescribe): {destino}")
    destino.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(tabla, destino)
    LOGGER.info("%s guardado: %s (md5 %s)", descripcion, destino, md5_archivo(ruta))


def escribir_json(ruta: str, contenido: dict, permitir: bool, descripcion: str) -> None:
    """Escribe un JSON nuevo (o lo reescribe solo si esta permitido).

    Args:
        ruta: Ruta de salida.
        contenido: Objeto serializable.
        permitir: True para permitir reescritura (smoke/--forzar).
        descripcion: Etiqueta para el log.

    Raises:
        FileExistsError: Si el archivo existe y no se permite reescribir.
    """
    destino = Path(ruta)
    if destino.exists() and not permitir:
        raise FileExistsError(f"{descripcion} ya existe (no se sobrescribe): {destino}")
    destino.parent.mkdir(parents=True, exist_ok=True)
    with open(destino, "w", encoding="utf-8") as archivo:
        json.dump(contenido, archivo, ensure_ascii=False, indent=2, default=str)
    LOGGER.info("%s guardado: %s (md5 %s)", descripcion, destino, md5_archivo(ruta))


# ===== SECCION 3: protocolo (split temporal + holdout bloqueado) =====


def cargar_etiquetas_splits(logger: logging.Logger) -> pd.DataFrame:
    """Carga etiquetas + split canonico (tablas ligeras, sin lemas).

    Args:
        logger: Logger activo.

    Returns:
        DataFrame con id, area_primaria, ano y particion (solo docs etiquetados
        con split canonico; sin_area_definida excluido).

    Raises:
        RuntimeError: Si el merge no cubre exactamente el split canonico.
    """
    etiquetas = pq.read_table(CONFIG["rutas_datos"]["etiquetas"]).to_pandas()
    splits = pq.read_table(CONFIG["rutas_datos"]["splits"]).to_pandas()
    df = etiquetas.merge(splits, on="id", how="inner")
    df = df[df["area_primaria"] != "sin_area_definida"].reset_index(drop=True)
    if len(df) != len(splits):
        raise RuntimeError(
            f"el merge cubre {len(df)} ids pero el split canonico tiene {len(splits)}"
        )
    if df["ano"].isna().any():
        raise RuntimeError("hay documentos sin ano: el split temporal no es construible")
    logger.info(
        "etiquetas+splits: %d docs (%s), anos %d-%d",
        len(df),
        df["particion"].value_counts().to_dict(),
        int(df["ano"].min()),
        int(df["ano"].max()),
    )
    return df[["id", "area_primaria", "ano", "particion"]]


def elegir_limites_temporales(
    conteos_ano: pd.Series, minimo: int, logger: logging.Logger
) -> tuple[tuple[int, int], tuple[int, int, int]]:
    """Elige cortes temporales factibles mas cercanos a la propuesta.

    La propuesta es train <=2011 / val 2012-2017 / test >=2018. Si algun
    conjunto queda por debajo del minimo, se explora la rejilla de pares de
    cortes (corte_train < corte_val) y se elige el par factible con distancia
    L1 minima a la propuesta (desempate: mayor minimo de docs, luego cortes
    menores). Determinista.

    Args:
        conteos_ano: Serie ano -> numero de docs.
        minimo: Minimo de documentos exigido a cada conjunto.
        logger: Logger activo.

    Returns:
        ((corte_train, corte_val), (n_train, n_val, n_test)).

    Raises:
        RuntimeError: Si ningun par de cortes cumple el minimo.
    """
    anios = sorted(int(a) for a in conteos_ano.index)
    if len(anios) < 3:
        raise RuntimeError("se necesitan al menos 3 anos distintos para el split temporal")
    acumulado = {int(a): int(conteos_ano[a]) for a in conteos_ano.index}
    propuesta = (CONFIG["corte_train_temporal"], CONFIG["corte_val_temporal"])
    mejor_clave = None
    mejor = None
    for i in range(len(anios) - 2):
        corte_train = anios[i]
        n_train = sum(acumulado[a] for a in anios if a <= corte_train)
        for j in range(i + 1, len(anios) - 1):
            corte_val = anios[j]
            n_val = sum(acumulado[a] for a in anios if corte_train < a <= corte_val)
            n_test = sum(acumulado[a] for a in anios if a > corte_val)
            if min(n_train, n_val, n_test) < minimo:
                continue
            clave = (
                abs(corte_train - propuesta[0]) + abs(corte_val - propuesta[1]),
                -min(n_train, n_val, n_test),
                corte_train,
                corte_val,
            )
            if mejor_clave is None or clave < mejor_clave:
                mejor_clave = clave
                mejor = (corte_train, corte_val, n_train, n_val, n_test)
    if mejor is None:
        raise RuntimeError(
            f"ningun par de cortes temporales cumple el minimo de {minimo} docs por conjunto"
        )
    corte_train, corte_val, n_train, n_val, n_test = mejor
    if (corte_train, corte_val) != propuesta:
        logger.warning(
            "limites temporales ajustados de %s a (%d, %d) para cumplir minimo=%d",
            propuesta,
            corte_train,
            corte_val,
            minimo,
        )
    return (corte_train, corte_val), (n_train, n_val, n_test)


def _conteos_por_clase(df: pd.DataFrame, columna: str, valores: list[str]) -> dict:
    """Cuenta documentos por clase dentro de cada valor de particion.

    Args:
        df: DataFrame con area_primaria y la columna de particion.
        columna: Nombre de la columna de particion.
        valores: Valores de particion a resumir.

    Returns:
        Dict particion -> {clase: n}.
    """
    salida = {}
    for valor in valores:
        conteo = df.loc[df[columna] == valor, "area_primaria"].value_counts()
        salida[valor] = {str(clase): int(n) for clase, n in conteo.items()}
    return salida


def construir_split_temporal(
    df: pd.DataFrame, smoke: bool, logger: logging.Logger
) -> tuple[pd.DataFrame, dict]:
    """Construye el split temporal por ano con la regla documentada.

    Args:
        df: DataFrame con id, area_primaria y ano.
        smoke: True relaja el minimo por conjunto a 1 doc (subconjunto).
        logger: Logger activo.

    Returns:
        (DataFrame con id y particion_temporal, metadatos de la regla).
    """
    df = df.reset_index(drop=True)
    conteos_ano = df["ano"].value_counts().sort_index()
    minimo = 1 if smoke else CONFIG["min_docs_temporal"]
    (corte_train, corte_val), (n_train, n_val, n_test) = elegir_limites_temporales(
        conteos_ano, minimo, logger
    )
    particion = np.where(
        df["ano"] <= corte_train, "train", np.where(df["ano"] <= corte_val, "val", "test")
    )
    df_temporal = df.assign(particion_temporal=particion)
    metadatos = {
        "regla_final": (
            f"train: ano<={corte_train}; val: {corte_train + 1}-{corte_val}; "
            f"test: ano>={corte_val + 1}"
        ),
        "limites_propuestos": {
            "corte_train": CONFIG["corte_train_temporal"],
            "corte_val": CONFIG["corte_val_temporal"],
        },
        "limites_finales": {"corte_train": corte_train, "corte_val": corte_val},
        "ajustado_por_minimo": (corte_train, corte_val)
        != (CONFIG["corte_train_temporal"], CONFIG["corte_val_temporal"]),
        "minimo_docs_por_conjunto": minimo,
        "conteos": {"train": n_train, "val": n_val, "test": n_test},
        "conteos_por_clase": _conteos_por_clase(
            df_temporal, "particion_temporal", ["train", "val", "test"]
        ),
    }
    logger.info(
        "split temporal: %s (train %d, val %d, test %d)",
        metadatos["regla_final"],
        n_train,
        n_val,
        n_test,
    )
    return df_temporal[["id", "particion_temporal"]], metadatos


def construir_holdout(
    df_train: pd.DataFrame, smoke: bool, logger: logging.Logger
) -> tuple[pd.DataFrame, dict]:
    """Construye el holdout bloqueado desde el train canonico.

    Regla: ~15 % estratificado por clase (seed 42). Clases con n < 10 en
    train dejan TODOS sus ejemplos en train; para la estratificacion las
    clases con n < 5 se colapsan en un unico estrato (salvaguarda; con los
    datos actuales no se activa porque las clases <10 ya quedan fuera).

    Args:
        df_train: Subconjunto de train canonico (id, area_primaria).
        smoke: True para permitir holdout vacio (subconjunto muy pequeno).
        logger: Logger activo.

    Returns:
        (DataFrame con id del holdout, metadatos de la regla y conteos).
    """
    conteo = df_train["area_primaria"].value_counts()
    excluidas = sorted(conteo[conteo < CONFIG["umbral_excepcion_holdout"]].index)
    elegibles = df_train[~df_train["area_primaria"].isin(excluidas)].copy()
    if elegibles.empty:
        if not smoke:
            raise RuntimeError("no hay clases elegibles para el holdout")
        logger.warning("smoke: sin clases elegibles para holdout (holdout vacio)")
        ids_holdout = pd.Series([], dtype=df_train["id"].dtype)
        colapsados: list[str] = []
    else:
        estratos = elegibles["area_primaria"].where(
            elegibles["area_primaria"].map(conteo) >= CONFIG["umbral_colapso_estratos"],
            "estratos_colapsados",
        )
        colapsados = sorted(estratos[estratos == "estratos_colapsados"].unique().tolist())
        if colapsados:
            logger.warning("estratos colapsados usados en el holdout: %s", colapsados)
        _, ids_holdout = train_test_split(
            elegibles["id"],
            test_size=CONFIG["fraccion_holdout"],
            random_state=CONFIG["semilla"],
            stratify=estratos,
        )
    holdout = pd.DataFrame({"id": ids_holdout.values})
    n_reducido = len(df_train) - len(holdout)
    metadatos = {
        "fraccion_objetivo": CONFIG["fraccion_holdout"],
        "semilla": CONFIG["semilla"],
        "umbral_excepcion_train": CONFIG["umbral_excepcion_holdout"],
        "umbral_colapso_estratos": CONFIG["umbral_colapso_estratos"],
        "clases_excluidas_del_holdout": {str(c): int(conteo[c]) for c in excluidas},
        "estratos_colapsados": colapsados,
        "n_train_canonico": len(df_train),
        "n_holdout": len(holdout),
        "n_train_reducido": int(n_reducido),
        "sellado": (
            "solo lectura: ningun variante de H1 evalua ni entrena con el holdout; "
            "queda reservado para la evaluacion final"
        ),
        "conteos_por_clase": {
            "train_canonico": {
                str(c): int(n) for c, n in df_train["area_primaria"].value_counts().items()
            },
            "holdout_bloqueado": {
                str(c): int(n)
                for c, n in holdout.merge(df_train, on="id")["area_primaria"]
                .value_counts()
                .items()
            },
        },
    }
    metadatos["conteos_por_clase"]["train_reducido"] = {
        clase: metadatos["conteos_por_clase"]["train_canonico"].get(clase, 0)
        - metadatos["conteos_por_clase"]["holdout_bloqueado"].get(clase, 0)
        for clase in metadatos["conteos_por_clase"]["train_canonico"]
    }
    logger.info(
        "holdout bloqueado: %d docs de %d train canonico (train reducido %d); excluidas <10: %s",
        len(holdout),
        len(df_train),
        n_reducido,
        {c: int(conteo[c]) for c in excluidas},
    )
    return holdout, metadatos


def ejecutar_protocolo(args, logger: logging.Logger) -> tuple[dict, float]:
    """Ejecuta el protocolo de evaluacion (idempotente fuera de smoke).

    Si los tres artefactos ya existen (modo real), verifica md5 y reutiliza;
    si el conjunto esta incompleto aborta sin sobrescribir. En smoke se
    regeneran siempre en /tmp/opencode/h1_smoke (scratch).

    Args:
        args: Argumentos CLI (smoke).
        logger: Logger activo.

    Returns:
        (JSON del protocolo, segundos totales).

    Raises:
        FileExistsError: Si el protocolo esta incompleto en modo real.
        RuntimeError: Si los md5 no coinciden con los registrados.
    """
    t0 = time.perf_counter()
    rutas = rutas_activas(args.smoke)
    rutas_parquet = (rutas["temporal"], rutas["holdout"])
    destino_json = rutas["protocolo_json"]
    if not args.smoke:
        existentes = [r for r in (*rutas_parquet, destino_json) if Path(r).exists()]
        if len(existentes) == 3:
            with open(destino_json, encoding="utf-8") as archivo:
                protocolo = json.load(archivo)
            for clave, ruta in (
                ("splits_m4_temporal", rutas["temporal"]),
                ("holdout_m4_bloqueado", rutas["holdout"]),
            ):
                actual = md5_archivo(ruta)
                esperado = protocolo.get("md5", {}).get(clave)
                if actual != esperado:
                    raise RuntimeError(
                        f"md5 de {ruta} ({actual}) no coincide con el protocolo ({esperado})"
                    )
            logger.info(
                "protocolo existente verificado por md5 y reutilizado: %s [%.1f s]",
                destino_json,
                time.perf_counter() - t0,
            )
            return protocolo, time.perf_counter() - t0
        if existentes:
            raise FileExistsError(
                "artefactos de protocolo incompletos (no se sobrescribe nada): "
                + ", ".join(existentes)
            )

    df = cargar_etiquetas_splits(logger)
    temporal, meta_temporal = construir_split_temporal(df, args.smoke, logger)
    n_antes = len(df)
    df = df.merge(temporal, on="id", how="inner")
    if len(df) != n_antes:
        raise RuntimeError(
            f"el split temporal cubre {len(df)} de {n_antes} docs: datos desalineados"
        )
    df_train = df[df["particion"] == "train"]
    holdout, meta_holdout = construir_holdout(df_train, args.smoke, logger)

    escribir_parquet(
        rutas["temporal"],
        pa.Table.from_pandas(temporal),
        permitir=args.smoke,
        descripcion="split temporal",
    )
    escribir_parquet(
        rutas["holdout"],
        pa.Table.from_pandas(holdout),
        permitir=args.smoke,
        descripcion="holdout bloqueado",
    )
    protocolo = {
        "fecha": pd.Timestamp.now().isoformat(),
        "script": "scripts/18_h1_ruido_m4.py",
        "smoke": bool(args.smoke),
        "versiones": versiones_entorno(),
        "md5_canonicos": {
            "splits_m4": md5_archivo(CONFIG["rutas_datos"]["splits"]),
            "etiquetas_areas": md5_archivo(CONFIG["rutas_datos"]["etiquetas"]),
        },
        "regla_temporal": meta_temporal,
        "regla_holdout": meta_holdout,
        "md5": {
            "splits_m4_temporal": md5_archivo(rutas["temporal"]),
            "holdout_m4_bloqueado": md5_archivo(rutas["holdout"]),
        },
        "archivos": {
            "splits_m4_temporal": rutas["temporal"],
            "holdout_m4_bloqueado": rutas["holdout"],
        },
        "nota_sellado": (
            "el holdout es de solo lectura y queda reservado para la evaluacion final; "
            "H1 solo lo usa para excluir sus ids del train reducido"
        ),
    }
    escribir_json(
        destino_json,
        protocolo,
        permitir=args.smoke,
        descripcion="protocolo de evaluacion",
    )
    logger.info(
        "protocolo generado en %.1f s (temporal %d/%d/%d; holdout %d)",
        time.perf_counter() - t0,
        meta_temporal["conteos"]["train"],
        meta_temporal["conteos"]["val"],
        meta_temporal["conteos"]["test"],
        meta_holdout["n_holdout"],
    )
    return protocolo, time.perf_counter() - t0


# ===== SECCION 4: datos y matriz TF-IDF de H1 =====


def cargar_datos_smoke(logger: logging.Logger) -> pd.DataFrame:
    """Carga un subconjunto estratificado de ~500 docs para el smoke.

    Lee las tablas completas, submuestrea ANTES de normalizar/enmascarar
    (mismo flujo que cargar_datos de 06 sobre el subconjunto) y conserva las
    particiones canonicas de cada documento: NO se rehacen splits.

    Args:
        logger: Logger activo.

    Returns:
        DataFrame submuestreado con lemas_norm, lemas_mask y particion.
    """
    base = modulo_06()
    etiquetas = pq.read_table(CONFIG["rutas_datos"]["etiquetas"]).to_pandas()
    splits = pq.read_table(CONFIG["rutas_datos"]["splits"]).to_pandas()
    lemas = pq.read_table(CONFIG["rutas_datos"]["lemas"]).to_pandas()
    df = lemas.merge(etiquetas[["id", "area_primaria", "ano"]], on="id")
    df = df[df["area_primaria"] != "sin_area_definida"].reset_index(drop=True)
    df = df.merge(splits, on="id", how="inner")
    partes = []
    for particion in ("train", "val", "test"):
        sub = df[df["particion"] == particion].reset_index(drop=True)
        partes.append(
            base.submuestrear_estratificado(
                sub, CONFIG["limites_smoke"][particion], CONFIG["semilla"]
            )
        )
    sub = pd.concat(partes, ignore_index=True)
    sub["lemas_norm"] = sub["lemas_texto"].map(base.normalizar_lemas)
    sub["lemas_mask"] = sub["lemas_norm"].map(base.enmascarar_citas_lemas)
    logger.info(
        "smoke: %d docs (%s), clases presentes %d",
        len(sub),
        sub["particion"].value_counts().to_dict(),
        sub["area_primaria"].nunique(),
    )
    return sub


def cargar_datos_h1(smoke: bool, logger: logging.Logger) -> pd.DataFrame:
    """Carga datos + splits canonicos + split temporal para H1.

    Args:
        smoke: True para usar el subconjunto de ~500 docs.
        logger: Logger activo.

    Returns:
        DataFrame con lemas_norm, lemas_mask, area_primaria, particion y
        particion_temporal.

    Raises:
        RuntimeError: Si el espacio de clases de los datos no es el fijo de 16
            (solo en modo completo) o si el merge no cubre los splits.
    """
    rutas = rutas_activas(smoke)
    if smoke:
        # El loader smoke ya trae la particion canonica de cada documento.
        df = cargar_datos_smoke(logger)
    else:
        df = modulo_06().cargar_datos()
        splits = pq.read_table(CONFIG["rutas_datos"]["splits"]).to_pandas()
        df = df.merge(splits, on="id", how="inner")
        if len(df) != len(splits):
            raise RuntimeError(
                f"los datos cubren {len(df)} ids pero los splits canonicos tienen {len(splits)}"
            )
        clases_datos = sorted(df["area_primaria"].unique().tolist())
        if clases_datos != CLASES_M4:
            raise RuntimeError(
                "el espacio de clases no coincide con el espacio fijo v1: "
                f"faltan={sorted(set(CLASES_M4) - set(clases_datos))} "
                f"sobran={sorted(set(clases_datos) - set(CLASES_M4))}"
            )
    n_antes = len(df)
    temporal = pq.read_table(rutas["temporal"]).to_pandas()
    df = df.merge(temporal, on="id", how="inner")
    if len(df) != n_antes:
        raise RuntimeError(
            f"el split temporal cubre {len(df)} de {n_antes} docs: datos desalineados"
        )
    logger.info(
        "datos H1: %d docs (particion %s; temporal %s)",
        len(df),
        df["particion"].value_counts().to_dict(),
        df["particion_temporal"].value_counts().to_dict(),
    )
    return df


def construir_bloques_matriz(
    df: pd.DataFrame, smoke: bool, logger: logging.Logger
) -> tuple[dict, dict]:
    """Vectoriza el train reducido y los cuatro conjuntos de evaluacion.

    El TF-IDF se ajusta SOLO sobre el train reducido (train canonico menos
    holdout) replicando la configuracion v1 (min_df/max_features/ngram de
    CONFIG de 06, sublinear_tf=True, lemas enmascarados).

    Args:
        df: DataFrame de cargar_datos_h1.
        smoke: True para no exigir cobertura total de clases.
        logger: Logger activo.

    Returns:
        (bloques {nombre: {X, ids, y}}, metadatos con conteos y params).

    Raises:
        RuntimeError: Si el holdout no es subconjunto del train canonico.
    """
    base = modulo_06()
    rutas = rutas_activas(smoke)
    holdout = pq.read_table(rutas["holdout"]).to_pandas()
    ids_holdout = {str(v) for v in holdout["id"]}
    ids_str = df["id"].astype(str)
    ids_train = set(ids_str[df["particion"] == "train"])
    ids_df = set(ids_str)
    # En smoke el holdout completo (calculado sobre el corpus) solo intersecta
    # parcialmente el subconjunto; se valida la interseccion presente.
    ids_holdout_presentes = ids_holdout & ids_df
    if not ids_holdout_presentes <= ids_train:
        raise RuntimeError(
            "el holdout contiene ids fuera del train canonico: "
            f"{len(ids_holdout_presentes - ids_train)} ids"
        )
    subconjuntos = {
        "train_reducido": df[(df["particion"] == "train") & (~ids_str.isin(ids_holdout))],
        "val": df[df["particion"] == "val"],
        "test": df[df["particion"] == "test"],
        "temporal_val": df[df["particion_temporal"] == "val"],
        "temporal_test": df[df["particion_temporal"] == "test"],
    }
    if subconjuntos["train_reducido"].empty:
        raise RuntimeError("el train reducido quedo vacio")
    vectorizador = TfidfVectorizer(
        min_df=base.CONFIG["min_df"],
        max_features=base.CONFIG["max_features"],
        ngram_range=tuple(base.CONFIG["ngram"]),
        sublinear_tf=True,
    )
    X_train = vectorizador.fit_transform(subconjuntos["train_reducido"]["lemas_mask"])
    bloques = {}
    for nombre in BLOQUES_MATRIZ:
        sub = subconjuntos[nombre]
        X = X_train if nombre == "train_reducido" else vectorizador.transform(sub["lemas_mask"])
        bloques[nombre] = {
            "X": X,
            "ids": np.asarray([str(v) for v in sub["id"]]),
            "y": np.asarray([str(v) for v in sub["area_primaria"]]),
        }
    params_v1 = base.cargar_hiperparametros_v1(rutas["modelo_v1"])
    ids_reducido = set(subconjuntos["train_reducido"]["id"].astype(str))
    conteos = {
        "train_canonico": len(df[df["particion"] == "train"]),
        "holdout_bloqueado": len(ids_holdout_presentes),
        "train_reducido": len(subconjuntos["train_reducido"]),
        "val_canonica": len(subconjuntos["val"]),
        "test_canonico": len(subconjuntos["test"]),
        "temporal_train": len(df[df["particion_temporal"] == "train"]),
        "temporal_val": len(subconjuntos["temporal_val"]),
        "temporal_test": len(subconjuntos["temporal_test"]),
        "temporal_val_en_train_reducido": len(
            set(subconjuntos["temporal_val"]["id"].astype(str)) & ids_reducido
        ),
        "temporal_test_en_train_reducido": len(
            set(subconjuntos["temporal_test"]["id"].astype(str)) & ids_reducido
        ),
    }
    metadatos = {
        "conteos": conteos,
        "params_v1": params_v1,
        "vec_params": {
            "min_df": int(vectorizador.min_df),
            "max_features": int(vectorizador.max_features),
            "ngram_range": [int(v) for v in vectorizador.ngram_range],
            "sublinear_tf": bool(vectorizador.sublinear_tf),
        },
        "n_features": len(vectorizador.get_feature_names_out()),
        "nnz_train_reducido": int(X_train.nnz),
    }
    logger.info(
        "matriz TF-IDF: %d features, train reducido %d docs (%d nnz); bloques %s",
        metadatos["n_features"],
        conteos["train_reducido"],
        metadatos["nnz_train_reducido"],
        {k: int(v["X"].shape[0]) for k, v in bloques.items()},
    )
    return bloques, metadatos


def guardar_matriz_h1(
    ruta: str, bloques: dict, metadatos: dict, logger: logging.Logger
) -> None:
    """Persiste la matriz TF-IDF de H1 (CSR float64) + metadatos en .npz.

    Args:
        ruta: Ruta de salida .npz.
        bloques: Dict de construir_bloques_matriz.
        metadatos: Metadatos (conteos, params, vec_params).
        logger: Logger activo.
    """
    destino = Path(ruta)
    destino.parent.mkdir(parents=True, exist_ok=True)
    arreglos = {
        "clases": np.asarray(CLASES_M4),
        "params_v1_json": np.asarray(json.dumps(metadatos["params_v1"])),
        "vec_params_json": np.asarray(json.dumps(metadatos["vec_params"])),
        "conteos_json": np.asarray(json.dumps(metadatos["conteos"])),
    }
    for nombre, bloque in bloques.items():
        X = bloque["X"]
        arreglos[f"{nombre}_data"] = X.data
        arreglos[f"{nombre}_indices"] = X.indices
        arreglos[f"{nombre}_indptr"] = X.indptr
        arreglos[f"{nombre}_shape"] = np.asarray(X.shape, dtype=np.int64)
        arreglos[f"{nombre}_ids"] = bloque["ids"]
        arreglos[f"{nombre}_labels"] = bloque["y"]
    np.savez(destino, **arreglos)
    logger.info(
        "matriz H1 guardada: %s (%.1f MB)",
        destino,
        destino.stat().st_size / 1e6,
    )


def cargar_metadatos_matriz(ruta: str) -> dict:
    """Lee solo los metadatos de la matriz H1 (sin reconstruir CSR).

    Args:
        ruta: Ruta del .npz.

    Returns:
        Dict con conteos, params_v1, vec_params, n_features.
    """
    with np.load(ruta, allow_pickle=False) as datos:
        return {
            "conteos": json.loads(str(datos["conteos_json"])),
            "params_v1": json.loads(str(datos["params_v1_json"])),
            "vec_params": json.loads(str(datos["vec_params_json"])),
            "n_features": int(datos["train_reducido_shape"][1]),
        }


def cargar_matriz_h1(ruta: str) -> dict:
    """Carga la matriz H1 completa y reconstruye los CSR por bloque.

    Args:
        ruta: Ruta del .npz.

    Returns:
        Dict con clases, params_v1, vec_params, conteos y bloques {X, ids, y}.
    """
    with np.load(ruta, allow_pickle=False) as datos:
        crudos = {clave: datos[clave] for clave in datos.files}
    salida = {
        "clases": [str(c) for c in crudos["clases"]],
        "params_v1": json.loads(str(crudos["params_v1_json"])),
        "vec_params": json.loads(str(crudos["vec_params_json"])),
        "conteos": json.loads(str(crudos["conteos_json"])),
    }
    for nombre in BLOQUES_MATRIZ:
        X = csr_matrix(
            (
                crudos[f"{nombre}_data"],
                crudos[f"{nombre}_indices"],
                crudos[f"{nombre}_indptr"],
            ),
            shape=tuple(int(v) for v in crudos[f"{nombre}_shape"]),
        )
        salida[nombre] = {
            "X": X,
            "ids": np.asarray([str(v) for v in crudos[f"{nombre}_ids"]]),
            "y": np.asarray([str(v) for v in crudos[f"{nombre}_labels"]]),
        }
    return salida


def preparar_matriz_h1(smoke: bool, forzar: bool, logger: logging.Logger) -> dict:
    """Construye o reutiliza la matriz H1 y devuelve sus metadatos.

    Memoria consciente: el orquestador libera los bloques (del/gc) antes de
    lanzar los subprocesos; cada variante carga su propia copia.

    Args:
        smoke: True para reconstruir siempre en scratch.
        forzar: True para reconstruir la matriz existente.
        logger: Logger activo.

    Returns:
        Metadatos de la matriz (conteos, params, vec_params, n_features).
    """
    rutas = rutas_activas(smoke)
    ruta = rutas["matriz"]
    if Path(ruta).exists() and not (forzar or smoke):
        metadatos = cargar_metadatos_matriz(ruta)
        logger.info("matriz H1 reutilizada: %s (%s)", ruta, metadatos["conteos"])
        return metadatos
    df = cargar_datos_h1(smoke, logger)
    bloques, metadatos = construir_bloques_matriz(df, smoke, logger)
    guardar_matriz_h1(ruta, bloques, metadatos, logger)
    del bloques, df
    gc.collect()
    return metadatos


# ===== SECCION 5: variantes H1 (cleanlab + entrenamiento + predicciones) =====


def cargar_oof_reducido(
    ruta_oof: str, ids_train: np.ndarray, y_train: np.ndarray, logger: logging.Logger
) -> np.ndarray:
    """Carga las probabilidades OOF del experimento 17 para el train reducido.

    Intersecta por id (string) las probabilidades out-of-sample del train
    canonico con el train reducido y verifica que las etiquetas coincidan.

    Args:
        ruta_oof: Ruta de resultados/m4_oof_probs.npz.
        ids_train: Ids del train reducido (strings, orden de la matriz).
        y_train: Etiquetas del train reducido.
        logger: Logger activo.

    Returns:
        Matriz (n_train_reducido, 16) de probabilidades OOF.

    Raises:
        RuntimeError: Si falta algun id o las etiquetas no coinciden.
    """
    with np.load(ruta_oof, allow_pickle=False) as datos:
        ids_oof = np.asarray([str(v) for v in datos["ids"]])
        probs_oof = np.asarray(datos["probs"], dtype=np.float64)
        y_oof = np.asarray([str(v) for v in datos["y_true"]])
    mapa = {valor: i for i, valor in enumerate(ids_oof)}
    faltantes = [valor for valor in ids_train if valor not in mapa]
    if faltantes:
        raise RuntimeError(
            f"{len(faltantes)} ids del train reducido no estan en las probs OOF: {faltantes[:5]}"
        )
    indices = np.asarray([mapa[valor] for valor in ids_train], dtype=np.int64)
    probs = probs_oof[indices]
    y_alineadas = y_oof[indices]
    if not np.array_equal(y_alineadas, y_train):
        raise RuntimeError("las etiquetas OOF no coinciden con las del train reducido")
    logger.info(
        "OOF reducido: %d docs alineados por id (%s)",
        len(ids_train),
        ruta_oof,
    )
    return probs


def preparar_probs_cleanlab(
    y: np.ndarray, probs: np.ndarray, logger: logging.Logger, nombre: str
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Prepara etiquetas/probabilidades para cleanlab (patron del modulo 17).

    Si faltan clases en el subconjunto se restringe a las presentes y se
    renormalizan las filas (cleanlab exige etiquetas 0..K-1 y filas que sumen 1).

    Args:
        y: Etiquetas observadas (strings).
        probs: Probabilidades alineadas al espacio fijo de 16.
        logger: Logger activo.
        nombre: Etiqueta de la variante para el log.

    Returns:
        (y_int, probs_uso, clases_uso).

    Raises:
        RuntimeError: Si hay menos de 2 clases presentes (cleanlab no aplica).
    """
    presentes = sorted(set(y.tolist()))
    if len(presentes) < 2:
        raise RuntimeError(f"{nombre}: cleanlab requiere al menos 2 clases presentes")
    clases_uso = list(CLASES_M4)
    if presentes != clases_uso:
        columnas = [clases_uso.index(clase) for clase in presentes]
        probs = probs[:, columnas]
        sumas = probs.sum(axis=1, keepdims=True)
        probs = probs / np.where(sumas > 0, sumas, 1.0)
        clases_uso = presentes
        logger.warning(
            "%s: cleanlab restringido a %d clases presentes (faltan %d)",
            nombre,
            len(presentes),
            len(CLASES_M4) - len(presentes),
        )
    indice = {clase: j for j, clase in enumerate(clases_uso)}
    y_int = np.asarray([indice[clase] for clase in y], dtype=np.int64)
    return y_int, probs, clases_uso


def ajustar_variante(
    nombre: str,
    X: csr_matrix,
    y: np.ndarray,
    probs_oof: np.ndarray | None,
    params: dict,
    logger: logging.Logger,
) -> tuple[LogisticRegression, dict]:
    """Ajusta la variante pedida con los hiperparametros v1.

    Args:
        nombre: v1r (baseline reentrenado), v2 (descarte flagged) o v3 (pesos).
        X: Matriz TF-IDF del train reducido.
        y: Etiquetas del train reducido.
        probs_oof: Probabilidades OOF del train reducido (v2/v3).
        params: Hiperparametros v1 de LogisticRegression.
        logger: Logger activo.

    Returns:
        (modelo ajustado, metadatos de la variante).

    Raises:
        ValueError: Si la variante es desconocida o probs_oof falta en v2/v3.
        RuntimeError: Si el descarte de V2 deja el train vacio.
    """
    if nombre == "v1r":
        modelo = LogisticRegression(**params)
        modelo.fit(X, y)
        metadatos = {
            "modo": "baseline reentrenado en train reducido",
            "n_entrenados": int(X.shape[0]),
            "n_descartados": 0,
            "tasa_descartados": 0.0,
        }
    elif nombre == "v2":
        if probs_oof is None:
            raise ValueError("v2 requiere las probabilidades OOF")
        y_int, probs_uso, clases_uso = preparar_probs_cleanlab(y, probs_oof, logger, "v2")
        flags = find_label_issues(labels=y_int, pred_probs=probs_uso)
        X_uso = X[~flags]
        y_uso = y[~flags]
        if X_uso.shape[0] == 0:
            raise RuntimeError("v2: el descarte de flagged dejo el train vacio")
        modelo = LogisticRegression(**params)
        modelo.fit(X_uso, y_uso)
        metadatos = {
            "modo": "descarte de documentos flagged por cleanlab find_label_issues",
            "n_entrenados": int(X_uso.shape[0]),
            "n_descartados": int(flags.sum()),
            "tasa_descartados": float(flags.mean()),
            "clases_cleanlab": clases_uso,
        }
    elif nombre == "v3":
        if probs_oof is None:
            raise ValueError("v3 requiere las probabilidades OOF")
        y_int, probs_uso, clases_uso = preparar_probs_cleanlab(y, probs_oof, logger, "v3")
        pesos = get_label_quality_scores(labels=y_int, pred_probs=probs_uso)
        modelo = LogisticRegression(**params)
        modelo.fit(X, y, sample_weight=pesos)
        metadatos = {
            "modo": "pesos por confianza de etiqueta (sample_weight)",
            "n_entrenados": int(X.shape[0]),
            "n_descartados": 0,
            "tasa_descartados": 0.0,
            "pesos": {
                "min": float(pesos.min()),
                "media": float(pesos.mean()),
                "max": float(pesos.max()),
            },
            "clases_cleanlab": clases_uso,
        }
    else:
        raise ValueError(f"variante desconocida: {nombre}")
    logger.info(
        "%s: %s | n=%d descartados=%d iters=%d",
        nombre,
        metadatos["modo"],
        metadatos["n_entrenados"],
        metadatos["n_descartados"],
        int(modelo.n_iter_.max()),
    )
    return modelo, metadatos


def guardar_preds_variante(
    ruta: str,
    nombre: str,
    matriz: dict,
    preds: dict,
    metadatos: dict,
    permitir: bool,
    logger: logging.Logger,
) -> None:
    """Persiste ids + probabilidades val/test/temporal de una variante.

    Args:
        ruta: Ruta de salida .npz.
        nombre: Nombre de la variante.
        matriz: Matriz H1 cargada (para ids y etiquetas de cada bloque).
        preds: Dict bloque -> matriz de probabilidades (n, 16).
        metadatos: Metadatos de la variante.
        permitir: True para permitir reescritura (smoke/--forzar).
        logger: Logger activo.

    Raises:
        FileExistsError: Si el archivo existe y no se permite reescribir.
    """
    destino = Path(ruta)
    if destino.exists() and not permitir:
        raise FileExistsError(f"preds existentes (no se sobrescribe): {destino}")
    destino.parent.mkdir(parents=True, exist_ok=True)
    arreglos = {
        "variante": np.asarray(nombre),
        "clases": np.asarray(CLASES_M4),
        "metadatos_json": np.asarray(json.dumps(metadatos, ensure_ascii=False)),
    }
    for bloque in BLOQUES_PRED:
        arreglos[f"ids_{bloque}"] = matriz[bloque]["ids"]
        arreglos[f"y_{bloque}"] = matriz[bloque]["y"]
        arreglos[f"probs_{bloque}"] = np.asarray(preds[bloque], dtype=np.float64)
    np.savez(destino, **arreglos)
    logger.info("preds %s guardadas: %s (%.1f MB)", nombre, destino, destino.stat().st_size / 1e6)


def ejecutar_variante(nombre: str, args, logger: logging.Logger, rutas: dict) -> None:
    """Cuerpo del subproceso --solo-variante: ajusta y predice una variante.

    Args:
        nombre: v1r, v2 o v3.
        args: Argumentos CLI (forzar, smoke).
        logger: Logger activo (solo consola; el orquestador canaliza).
        rutas: Dict de rutas activas.

    Raises:
        FileNotFoundError: Si falta la matriz H1.
    """
    ruta_preds = rutas["preds"].format(variante=nombre)
    permitir = args.smoke or args.forzar
    if Path(ruta_preds).exists() and not permitir:
        logger.info("variante %s: preds existentes, se reutilizan (%s)", nombre, ruta_preds)
        return
    t0 = time.perf_counter()
    tiempos: dict = {}
    ruta_matriz = Path(rutas["matriz"])
    if not ruta_matriz.exists():
        raise FileNotFoundError(f"falta la matriz H1: {ruta_matriz}")
    t = time.perf_counter()
    matriz = cargar_matriz_h1(rutas["matriz"])
    tiempos["carga_matriz_seg"] = time.perf_counter() - t
    params = matriz["params_v1"]
    X = matriz["train_reducido"]["X"]
    y = matriz["train_reducido"]["y"]
    ids = matriz["train_reducido"]["ids"]
    probs_oof = None
    if nombre in ("v2", "v3"):
        t = time.perf_counter()
        probs_oof = cargar_oof_reducido(rutas["oof_probs"], ids, y, logger)
        tiempos["carga_oof_seg"] = time.perf_counter() - t
    t = time.perf_counter()
    modelo, metadatos = ajustar_variante(nombre, X, y, probs_oof, params, logger)
    tiempos["fit_seg"] = time.perf_counter() - t
    t = time.perf_counter()
    preds = {
        bloque: modulo_17().expandir_probs(modelo, matriz[bloque]["X"], CLASES_M4)
        for bloque in BLOQUES_PRED
    }
    tiempos["pred_seg"] = time.perf_counter() - t
    metadatos.update(
        {
            "variante": nombre,
            "smoke": bool(args.smoke),
            "fecha": pd.Timestamp.now().isoformat(),
            "n_train_reducido": len(y),
            "n_iter": int(modelo.n_iter_.max()),
            "params_v1": params,
            "tiempos_seg": tiempos,
            "tiempo_total_seg": time.perf_counter() - t0,
        }
    )
    guardar_preds_variante(ruta_preds, nombre, matriz, preds, metadatos, permitir, logger)
    logger.info(
        "variante %s terminada en %.1f s (fit %.1f s, pred %.1f s)",
        nombre,
        time.perf_counter() - t0,
        tiempos["fit_seg"],
        tiempos["pred_seg"],
    )
    del matriz, modelo, preds, X
    gc.collect()


# ===== SECCION 6: orquestacion de variantes (subprocesos en paralelo) =====


def _relay_salida(proceso: subprocess.Popen, nombre: str, logger: logging.Logger) -> None:
    """Canaliza la salida del subproceso de variante al log del orquestador.

    Args:
        proceso: Subproceso Popen con stdout=PIPE.
        nombre: Nombre de la variante.
        logger: Logger activo.
    """
    if proceso.stdout is None:
        return
    for linea in proceso.stdout:
        linea = linea.rstrip()
        if linea:
            logger.info("[%s] %s", nombre, linea)


def lanzar_variantes(args, logger: logging.Logger, rutas: dict) -> dict:
    """Lanza hasta --jobs subprocesos de variante en paralelo (resumible).

    Args:
        args: Argumentos CLI (jobs, smoke, forzar).
        logger: Logger activo.
        rutas: Dict de rutas activas.

    Returns:
        Dict variante -> codigo de retorno (las preds existentes se saltan
        salvo --forzar o smoke).

    Raises:
        RuntimeError: Si alguna variante termina con codigo distinto de 0.
    """
    pendientes = []
    for nombre in VARIANTES:
        salida = Path(rutas["preds"].format(variante=nombre))
        if salida.exists() and not (args.forzar or args.smoke):
            logger.info("variante %s ya existe, se reutiliza (%s)", nombre, salida)
        else:
            pendientes.append(nombre)
    entorno = os.environ.copy()
    for variable in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        entorno[variable] = str(CONFIG["hilos_por_variante"])
    activos: dict = {}
    codigos: dict = {}
    try:
        while pendientes or activos:
            while pendientes and len(activos) < max(1, args.jobs):
                nombre = pendientes.pop(0)
                comando = [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--solo-variante",
                    nombre,
                    "--jobs",
                    str(args.jobs),
                ]
                if args.smoke:
                    comando.append("--smoke")
                if args.forzar:
                    comando.append("--forzar")
                proceso = subprocess.Popen(
                    comando,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    cwd=str(RAIZ),
                    env=entorno,
                )
                hilo = threading.Thread(
                    target=_relay_salida, args=(proceso, nombre, logger), daemon=True
                )
                hilo.start()
                activos[nombre] = (proceso, hilo)
                logger.info("variante %s lanzada (pid %d)", nombre, proceso.pid)
            time.sleep(1.0)
            for nombre in list(activos):
                proceso, hilo = activos[nombre]
                if proceso.poll() is not None:
                    hilo.join(timeout=10)
                    codigos[nombre] = int(proceso.returncode)
                    if proceso.returncode != 0:
                        logger.error("variante %s termino con rc=%d", nombre, proceso.returncode)
                    else:
                        logger.info("variante %s completada", nombre)
                    del activos[nombre]
    except KeyboardInterrupt:
        logger.warning("interrupcion: terminando %d variantes activas", len(activos))
        for proceso, hilo in activos.values():
            proceso.terminate()
        for proceso, hilo in activos.values():
            proceso.wait(timeout=30)
            hilo.join(timeout=5)
        raise
    fallidas = {nombre: rc for nombre, rc in codigos.items() if rc != 0}
    if fallidas:
        raise RuntimeError(f"variantes fallidas: {fallidas}")
    return codigos


# ===== SECCION 7: metricas, decision y figura =====


def cargar_preds_variante(ruta: str) -> dict:
    """Carga las predicciones persistidas de una variante.

    Args:
        ruta: Ruta del .npz.

    Returns:
        Dict con variante, metadatos y bloques {ids, y, probs}.
    """
    with np.load(ruta, allow_pickle=False) as datos:
        crudos = {clave: datos[clave] for clave in datos.files}
    salida = {
        "variante": str(crudos["variante"]),
        "metadatos": json.loads(str(crudos["metadatos_json"])),
    }
    for bloque in BLOQUES_PRED:
        salida[bloque] = {
            "ids": np.asarray([str(v) for v in crudos[f"ids_{bloque}"]]),
            "y": np.asarray([str(v) for v in crudos[f"y_{bloque}"]]),
            "probs": np.asarray(crudos[f"probs_{bloque}"], dtype=np.float64),
        }
    return salida


def evaluar_bloque(
    y: np.ndarray, probs: np.ndarray, etiqueta: str, logger: logging.Logger
) -> dict:
    """Metricas completas + IC bootstrap de un bloque de evaluacion.

    Args:
        y: Etiquetas reales (strings).
        probs: Probabilidades (n, 16) alineadas al espacio fijo.
        etiqueta: Nombre del bloque para el log.
        logger: Logger activo.

    Returns:
        Dict de evaluar_predicciones (17) + bootstrap_macro_f1_espacio_fijo.
    """
    reutil = modulo_17()
    prediccion = np.asarray(CLASES_M4)[np.argmax(probs, axis=1)]
    metricas = reutil.evaluar_predicciones(y, prediccion, CLASES_M4)
    t0 = time.perf_counter()
    metricas["bootstrap_macro_f1_espacio_fijo"] = reutil.bootstrap_macro_f1(
        y,
        prediccion,
        CLASES_M4,
        CONFIG["n_bootstrap"],
        CONFIG["semilla"],
        logger,
        f"bootstrap {etiqueta}",
    )
    logger.info(
        "%s: macro-F1 fijo=%.4f obs=%.4f acc=%.4f IC95=[%.4f, %.4f] [%.1f s]",
        etiqueta,
        metricas["macro_f1_espacio_fijo"],
        metricas["macro_f1"],
        metricas["accuracy"],
        metricas["bootstrap_macro_f1_espacio_fijo"]["ic95_inf"],
        metricas["bootstrap_macro_f1_espacio_fijo"]["ic95_sup"],
        time.perf_counter() - t0,
    )
    return metricas


def seleccionar_ganador(val_metricas: dict) -> str:
    """Selecciona el ganador por macro-F1 de espacio fijo en val canonica.

    Args:
        val_metricas: Dict variante -> metricas de val.

    Returns:
        Nombre de la variante ganadora (desempate por orden de VARIANTES).
    """
    ganador = VARIANTES[0]
    for nombre in VARIANTES[1:]:
        if (
            val_metricas[nombre]["macro_f1_espacio_fijo"]
            > val_metricas[ganador]["macro_f1_espacio_fijo"]
        ):
            ganador = nombre
    return ganador


def cargar_referencia_v1(rutas: dict, logger: logging.Logger) -> dict:
    """Lee la referencia v1 (test canonico) de los JSON canonicos.

    Args:
        rutas: Dict de rutas activas.
        logger: Logger activo.

    Returns:
        Dict con macro_f1_observado, macro_f1_espacio_fijo y fuente.
    """
    referencia = dict(REFERENCIA_V1)
    try:
        with open(rutas["baseline_json"], encoding="utf-8") as archivo:
            baseline = json.load(archivo)
        referencia["macro_f1_observado"] = float(baseline["enmascarado"]["logreg"]["macro_f1"])
    except (FileNotFoundError, KeyError, TypeError) as error:
        logger.warning("no se pudo leer la referencia observada de v1: %s", error)
    try:
        with open(rutas["ruido_json"], encoding="utf-8") as archivo:
            ruido = json.load(archivo)
        referencia["macro_f1_espacio_fijo"] = float(
            ruido["resumen"]["v1_test_macro_f1_espacio_fijo"]
        )
    except (FileNotFoundError, KeyError, TypeError) as error:
        logger.warning("no se pudo leer la referencia de espacio fijo de v1: %s", error)
    logger.info(
        "referencia v1 (test canonico, sin reentrenar): obs=%.4f fijo=%.4f",
        referencia["macro_f1_observado"],
        referencia["macro_f1_espacio_fijo"],
    )
    return referencia


def construir_decision(
    val_metricas: dict, variantes_resultado: dict, ganador: str, referencia: dict, lecturas: list
) -> dict:
    """Construye el bloque de decision del JSON de H1.

    Args:
        val_metricas: Dict variante -> metricas de val canonica.
        variantes_resultado: Dict variante -> resultado (con test/temporal).
        ganador: Variante ganadora por val.
        referencia: Referencia v1 del test canonico.
        lecturas: Variantes que leyeron test canonico/temporal.

    Returns:
        Dict de decision.
    """
    val_v1r = val_metricas["v1r"]["macro_f1_espacio_fijo"]
    val_ganador = val_metricas[ganador]["macro_f1_espacio_fijo"]
    test_ganador = variantes_resultado[ganador]["test_dev"]["macro_f1_espacio_fijo"]
    test_v1r = variantes_resultado["v1r"]["test_dev"]["macro_f1_espacio_fijo"]
    temporal_ganador = variantes_resultado[ganador]["temporal_test"]["macro_f1_espacio_fijo"]
    temporal_v1r = variantes_resultado["v1r"]["temporal_test"]["macro_f1_espacio_fijo"]
    return {
        "ganador_por_val": ganador,
        "criterio": "mayor macro_f1_espacio_fijo en val canonica (desempate: orden v1r, v2, v3)",
        "val_macro_f1_espacio_fijo": {
            nombre: val_metricas[nombre]["macro_f1_espacio_fijo"] for nombre in VARIANTES
        },
        "delta_val_ganador_vs_v1r": float(val_ganador - val_v1r),
        "test_dev": {
            "ganador": float(test_ganador),
            "v1r": float(test_v1r),
            "referencia_v1": float(referencia["macro_f1_espacio_fijo"]),
            "delta_ganador_vs_v1r": float(test_ganador - test_v1r),
            "gana_a_referencia_v1": bool(test_ganador > referencia["macro_f1_espacio_fijo"]),
            "development_test_reutilizado": True,
        },
        "temporal_test": {
            "ganador": float(temporal_ganador),
            "v1r": float(temporal_v1r),
            "delta_ganador_vs_v1r": float(temporal_ganador - temporal_v1r),
            "development_test_reutilizado": True,
        },
        "lecturas_test_canonico": lecturas,
        "lecturas_test_temporal": lecturas,
        "holdout_bloqueado": (
            "no evaluado en H1; reservado para la evaluacion final del proyecto"
        ),
    }


def figura_h1(
    val_metricas: dict,
    ganador: str,
    ruta: str,
    permitir: bool,
    logger: logging.Logger,
) -> None:
    """Genera la figura de H1: macro-F1 val por variante + F1 por clase.

    Args:
        val_metricas: Dict variante -> metricas de val canonica.
        ganador: Variante ganadora por val.
        ruta: Ruta de salida .png.
        permitir: True para permitir reescritura (smoke/--forzar).
        logger: Logger activo.

    Raises:
        FileExistsError: Si la figura existe y no se permite reescribir.
    """
    destino = Path(ruta)
    if destino.exists() and not permitir:
        raise FileExistsError(f"figura existente (no se sobrescribe): {destino}")
    destino.parent.mkdir(parents=True, exist_ok=True)
    figura, ejes = plt.subplots(1, 2, figsize=(15, 6.5))
    eje = ejes[0]
    valores = [val_metricas[nombre]["macro_f1_espacio_fijo"] for nombre in VARIANTES]
    colores = ["#c44e52" if nombre == ganador else "#4c72b0" for nombre in VARIANTES]
    eje.bar(VARIANTES, valores, color=colores)
    for posicion, valor in enumerate(valores):
        eje.text(posicion, valor + 0.004, f"{valor:.4f}", ha="center", fontsize=9)
    eje.set_title("Macro-F1 val canonica (espacio fijo 16)")
    eje.set_ylabel("macro-F1")
    eje.set_ylim(0, max(valores + [0.05]) * 1.25)
    eje.grid(axis="y", alpha=0.3)
    eje = ejes[1]
    posiciones = np.arange(len(CLASES_M4))
    ancho = 0.26
    for indice, nombre in enumerate(VARIANTES):
        f1_clase = [val_metricas[nombre]["por_clase"][clase]["f1"] for clase in CLASES_M4]
        eje.barh(posiciones + (indice - 1) * ancho, f1_clase, height=ancho, label=nombre)
    eje.set_yticks(posiciones)
    eje.set_yticklabels(CLASES_M4, fontsize=8)
    eje.invert_yaxis()
    eje.set_xlabel("F1 val")
    eje.set_title("F1 por clase (val canonica)")
    eje.legend(fontsize=9)
    eje.grid(axis="x", alpha=0.3)
    figura.suptitle(
        f"H1 robustez al ruido del target M4 | ganador val: {ganador} | holdout sellado",
        fontsize=12,
    )
    figura.tight_layout()
    figura.savefig(destino, dpi=150, bbox_inches="tight")
    plt.close(figura)
    logger.info("figura H1 guardada: %s", destino)


# ===== SECCION 8: ejecucion de H1 y salida =====


def ejecutar_h1(args, logger: logging.Logger) -> tuple[dict, float]:
    """Ejecuta H1 completo: matriz, variantes, metricas, decision y JSON.

    Args:
        args: Argumentos CLI.
        logger: Logger activo.

    Returns:
        (JSON de H1, segundos totales).

    Raises:
        FileExistsError: Si el JSON o la figura ya existen sin --forzar.
        RuntimeError: Si algun archivo canonico cambio durante la corrida.
    """
    t0 = time.perf_counter()
    rutas = rutas_activas(args.smoke)
    permitir = args.smoke or args.forzar
    for clave in ("json", "figura"):
        if Path(rutas[clave]).exists() and not permitir:
            raise FileExistsError(
                f"salida de H1 ya existente (no se sobrescribe; use --forzar): {rutas[clave]}"
            )
    verificar_entradas(rutas, requiere_protocolo=True, logger=logger)
    with open(rutas["protocolo_json"], encoding="utf-8") as archivo:
        protocolo = json.load(archivo)
    claves_md5 = ["lemas", "etiquetas", "splits", "oof_probs", "modelo_v1"]
    md5_inicial = {clave: md5_archivo(rutas[clave]) for clave in claves_md5}

    logger.info("--- H1: matriz TF-IDF (train reducido) ---")
    t = time.perf_counter()
    metadatos_matriz = preparar_matriz_h1(args.smoke, args.forzar, logger)
    tiempos = {"matriz_seg": time.perf_counter() - t}

    logger.info("--- H1: variantes en paralelo (jobs=%d) ---", args.jobs)
    t = time.perf_counter()
    lanzar_variantes(args, logger, rutas)
    tiempos["variantes_seg"] = time.perf_counter() - t

    logger.info("--- H1: metricas y decision ---")
    t = time.perf_counter()
    val_metricas: dict = {}
    variantes_resultado: dict = {}
    for nombre in VARIANTES:
        datos = cargar_preds_variante(rutas["preds"].format(variante=nombre))
        val_metricas[nombre] = evaluar_bloque(
            datos["val"]["y"], datos["val"]["probs"], f"val {nombre}", logger
        )
        variantes_resultado[nombre] = {
            "metadatos": datos["metadatos"],
            "val": val_metricas[nombre],
        }
    ganador = seleccionar_ganador(val_metricas)
    lecturas = list(dict.fromkeys(["v1r", ganador]))
    logger.info("ganador por val: %s (lecturas de test: %s)", ganador, lecturas)
    for nombre in lecturas:
        datos = cargar_preds_variante(rutas["preds"].format(variante=nombre))
        test = evaluar_bloque(
            datos["test"]["y"], datos["test"]["probs"], f"test dev {nombre}", logger
        )
        test["development_test_reutilizado"] = True
        temporal = evaluar_bloque(
            datos["temporal_test"]["y"],
            datos["temporal_test"]["probs"],
            f"temporal test {nombre}",
            logger,
        )
        temporal["development_test_reutilizado"] = True
        variantes_resultado[nombre]["test_dev"] = test
        variantes_resultado[nombre]["temporal_test"] = temporal
    referencia = cargar_referencia_v1(rutas, logger)
    decision = construir_decision(val_metricas, variantes_resultado, ganador, referencia, lecturas)
    figura_h1(val_metricas, ganador, rutas["figura"], permitir, logger)
    tiempos["metricas_seg"] = time.perf_counter() - t

    resultado = {
        "fecha": pd.Timestamp.now().isoformat(),
        "script": "scripts/18_h1_ruido_m4.py",
        "smoke": bool(args.smoke),
        "versiones": versiones_entorno(),
        "parametros_cli": {
            "jobs": args.jobs,
            "smoke": args.smoke,
            "forzar": args.forzar,
        },
        "premisas": PREMISAS,
        "clases_espacio_fijo": CLASES_M4,
        "md5_entradas": {clave: md5_inicial[clave] for clave in claves_md5},
        "protocolo": {
            "regla_temporal": protocolo["regla_temporal"],
            "regla_holdout": protocolo["regla_holdout"],
            "md5": protocolo["md5"],
            "archivos": protocolo["archivos"],
            "nota_sellado": protocolo["nota_sellado"],
        },
        "conteos": metadatos_matriz["conteos"],
        "matriz": {
            "n_features": metadatos_matriz["n_features"],
            "params_v1": metadatos_matriz["params_v1"],
            "vec_params": metadatos_matriz["vec_params"],
            "ruta": rutas["matriz"],
        },
        "referencia_v1": referencia,
        "variantes": variantes_resultado,
        "decision": decision,
        "tiempos_seg": tiempos,
        "artefactos": {
            "matriz": rutas["matriz"],
            "preds": {
                nombre: rutas["preds"].format(variante=nombre) for nombre in VARIANTES
            },
            "json": rutas["json"],
            "figura": rutas["figura"],
            "log": rutas["log"],
            "protocolo_json": rutas["protocolo_json"],
        },
    }
    tiempos["total_seg"] = time.perf_counter() - t0
    escribir_json(rutas["json"], resultado, permitir=permitir, descripcion="resultado H1")
    for clave in claves_md5:
        if md5_archivo(rutas[clave]) != md5_inicial[clave]:
            raise RuntimeError(f"archivo canonico modificado durante la corrida: {clave}")
    logger.info("integridad canonica verificada (md5 sin cambios)")
    imprimir_resumen(resultado, logger)
    return resultado, time.perf_counter() - t0


def imprimir_resumen(resultado: dict, logger: logging.Logger) -> None:
    """Imprime el resumen final en consola (maximo 10 lineas).

    Args:
        resultado: JSON de H1.
        logger: Logger activo.
    """
    decision = resultado["decision"]
    val = decision["val_macro_f1_espacio_fijo"]
    logger.info(
        "RESUMEN H1 | ganador val=%s (%.4f) | val v1r=%.4f v2=%.4f v3=%.4f",
        decision["ganador_por_val"],
        val[decision["ganador_por_val"]],
        val["v1r"],
        val["v2"],
        val["v3"],
    )
    logger.info(
        "test dev (reutilizado): ganador %.4f | v1r %.4f | referencia v1 %.4f",
        decision["test_dev"]["ganador"],
        decision["test_dev"]["v1r"],
        decision["test_dev"]["referencia_v1"],
    )
    logger.info(
        "test temporal: ganador %.4f | v1r %.4f",
        decision["temporal_test"]["ganador"],
        decision["temporal_test"]["v1r"],
    )
    logger.info(
        "holdout bloqueado: %d docs (sellado, no evaluado)",
        resultado["conteos"]["holdout_bloqueado"],
    )
    logger.info(
        "salidas: %s | %s",
        resultado["artefactos"]["json"],
        resultado["artefactos"]["figura"],
    )
    logger.info("tiempo total H1: %.1f s", resultado["tiempos_seg"]["total_seg"])


# ===== SECCION 9: CLI y main =====


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Define y parsea los argumentos de linea de comandos.

    Args:
        argv: Lista de argumentos (None usa sys.argv).

    Returns:
        Namespace con protocolo, h1, todo, jobs, solo_variante, forzar, smoke.
    """
    parser = argparse.ArgumentParser(
        description=(
            "H1: entrenamiento robusto al ruido del target silver M4 + protocolo "
            "de evaluacion (split temporal + holdout bloqueado)."
        )
    )
    parser.add_argument("--protocolo", action="store_true", help="solo protocolo (split temporal + holdout)")
    parser.add_argument("--h1", action="store_true", help="solo H1 (requiere protocolo previo)")
    parser.add_argument("--todo", action="store_true", help="protocolo + H1 (default)")
    parser.add_argument("--jobs", type=int, default=CONFIG["jobs"], help="maximo de variantes en paralelo (default 3)")
    parser.add_argument(
        "--solo-variante",
        choices=list(VARIANTES),
        default=None,
        dest="solo_variante",
        help="interno: ejecuta solo esa variante (subproceso)",
    )
    parser.add_argument("--forzar", action="store_true", help="rehace matriz/preds/JSON existentes")
    parser.add_argument("--smoke", action="store_true", help="subconjunto ~500 docs; salidas solo en /tmp/opencode/h1_smoke")
    return parser.parse_args(argv)


def resolver_modo(args: argparse.Namespace) -> str:
    """Determina el modo de ejecucion a partir de los flags.

    Args:
        args: Argumentos CLI.

    Returns:
        "protocolo", "h1" o "todo" (default todo).
    """
    if args.todo or not any((args.protocolo, args.h1, args.todo)):
        return "todo"
    if args.protocolo and args.h1:
        return "todo"
    return "protocolo" if args.protocolo else "h1"


def main(argv: list[str] | None = None) -> None:
    """Orquesta protocolo y H1 segun los flags.

    Args:
        argv: Argumentos CLI (None usa sys.argv).
    """
    args = parse_args(argv)
    if args.jobs < 1:
        raise ValueError("--jobs debe ser >= 1")
    os.chdir(RAIZ)
    rutas = rutas_activas(args.smoke)
    if args.smoke:
        Path(CONFIG["dir_smoke"]).mkdir(parents=True, exist_ok=True)
    if args.solo_variante is not None:
        logger = configurar_logger(rutas["log"], solo_consola=True)
        random.seed(CONFIG["semilla"])
        np.random.seed(CONFIG["semilla"])
        ejecutar_variante(args.solo_variante, args, logger, rutas)
        return
    logger = configurar_logger(rutas["log"])
    random.seed(CONFIG["semilla"])
    np.random.seed(CONFIG["semilla"])
    modo = resolver_modo(args)
    logger.info(
        "=== INICIO H1 ruido M4 | modo=%s smoke=%s jobs=%d forzar=%s ===",
        modo,
        args.smoke,
        args.jobs,
        args.forzar,
    )
    t0 = time.perf_counter()
    tiempos = {}
    if modo in ("protocolo", "todo"):
        logger.info("--- protocolo de evaluacion ---")
        _, tiempos["protocolo_seg"] = ejecutar_protocolo(args, logger)
    if modo in ("h1", "todo"):
        _, tiempos["h1_seg"] = ejecutar_h1(args, logger)
    logger.info(
        "=== FIN H1 ruido M4 en %.1f s | tiempos: %s ===",
        time.perf_counter() - t0,
        tiempos,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception:  # noqa: BLE001 - protocolo: traza completa y salida 1 ante cualquier fallo
        traceback.print_exc()
        sys.exit(1)
