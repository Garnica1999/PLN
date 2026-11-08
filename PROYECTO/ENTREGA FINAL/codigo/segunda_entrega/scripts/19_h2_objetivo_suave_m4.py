"""19_h2_objetivo_suave_m4.py
Fecha: 2026-10-03
Descripcion: Experimento H2 (objetivo suave por ambiguedad multi-tema del
target silver M4). H1 (experimento 18) fue NEGATIVO: descartar documentos
flagged o pesarlos por confianza no mejoro al baseline v1r. H2 prueba una
hipotesis distinta: parte del techo del clasificador se explica por la
AMBIGUEDAD multi-tema, porque data/etiquetas_areas.parquet registra en
top3_json la distribucion completa de bloques citados por documento y el
entrenamiento v1 solo usa el bloque mayoritario (area_primaria). Variantes:

  - W1 = v1r reutilizado de H1 (baseline de referencia, NO es un modelo nuevo).
  - W2 = expansion ponderada (doc, bloque) con peso n_i / suma(n) sobre los
    bloques del top3 dentro del espacio fijo de 16 y LogReg multinomial v1 con
    sample_weight; equivale exactamente a entropia cruzada contra el objetivo
    suave q(x) (ver premisas).
  - W3 = 16 clasificadores binarios LogReg v1 (OneVsRestClassifier, n_jobs=4)
    sobre membresia en top3; prediccion single-label por argmax de las
    probabilidades por bloque.

Protocolo vigente (creado por el experimento 18): train reducido = train
canonico - holdout SELLADO (data/holdout_m4_bloqueado.parquet, 2.481 docs;
jamas se entrena ni evalua con el); seleccion en val canonica con macro-F1 de
espacio fijo de 16; test canonico UNA lectura solo si hay ganador con IC95 del
delta pareado vs W1 con inferior > 0 (marcado development_test_reutilizado);
temporal solo lectura para el ganador; IC bootstrap 1000/42. W2/W3 corren en
subprocesos (--jobs 2) con hilos limitados por entorno y son resumibles.

Estado del arte aplicado:
  - Objetivos suaves / label smoothing (Szegedy et al., 2016) y soft targets
    de destilacion (Hinton et al., 2015): la expansion ponderada reproduce el
    gradiente de la entropia cruzada contra la distribucion q(x) del doc.
  - Multietiqueta OvR para ambiguedad multi-tema (Tsoumakas y Katakis, 2007);
    relacion con la variante B de la Fase U (alli con umbrales, aqui argmax).
  - IC bootstrap percentil 95 % y comparacion pareada (Efron y Tibshirani,
    1993); validacion temporal estricta (particion cronologica obligatoria).

Uso:
    /home/carlos/venv/bin/python scripts/19_h2_objetivo_suave_m4.py --todo
    /home/carlos/venv/bin/python scripts/19_h2_objetivo_suave_m4.py --h2
    /home/carlos/venv/bin/python scripts/19_h2_objetivo_suave_m4.py --smoke --todo
    /home/carlos/venv/bin/python scripts/19_h2_objetivo_suave_m4.py --solo-variante w2
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
import pyarrow.parquet as pq
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

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

# Variantes nuevas de H2 (W1 es la referencia reutilizada).
VARIANTES = ("w2", "w3")
# Orden de presentacion/evaluacion (incluye la referencia).
VARIANTES_TODAS = ("w1", "w2", "w3")

# Bloques de la matriz TF-IDF (train reducido + 4 conjuntos de evaluacion).
BLOQUES_MATRIZ = ("train_reducido", "val", "test", "temporal_val", "temporal_test")
# Bloques sobre los que cada variante predice probabilidades.
BLOQUES_PRED = ("val", "test", "temporal_val", "temporal_test")

# Parametros del vectorizador v1 esperados en la matriz reutilizada.
VEC_PARAMS_V1 = {
    "min_df": 3,
    "max_features": 100000,
    "ngram_range": [1, 2],
    "sublinear_tf": True,
}

CONFIG = {
    "semilla": 42,
    "n_bootstrap": 1000,
    "jobs": 2,
    "hilos_por_variante": 8,
    "n_jobs_ovr": 4,
    "limite_smoke_total": 500,
    "dir_smoke": "/tmp/opencode/h2_smoke",
    "rutas_datos": {
        "lemas": "data/lemas_por_doc.parquet",
        "etiquetas": "data/etiquetas_areas.parquet",
        "splits": "data/splits_m4.parquet",
        "temporal": "data/splits_m4_temporal.parquet",
        "holdout": "data/holdout_m4_bloqueado.parquet",
        "modelo_v1": "modelos/m4_baseline/logreg_enmascarado.joblib",
        "protocolo_json": "resultados/protocolo_evaluacion_m4.json",
    },
    "rutas_h2": {
        "matriz": "resultados/m4_h1_matriz.npz",
        "matriz_reconstruida": "resultados/m4_h2_matriz.npz",
        "w1_preds": "resultados/m4_h1_v1r_preds.npz",
        "w1_preds_respaldo": "resultados/m4_h2_w1_preds.npz",
        "preds": "resultados/m4_h2_{variante}_preds.npz",
        "json": "resultados/m4_h2_objetivo_suave_v1.json",
        "figura": "figuras/fig_h2_objetivo_suave.png",
        "log": "experimentos/19_h2_objetivo_suave.log",
        "h1_json": "resultados/m4_h1_ruido_v1.json",
    },
}

# Premisas de diseno congeladas en el JSON de H2 (trazabilidad).
PREMISAS = [
    (
        "H2 (ambiguedad multi-tema): el entrenamiento v1 usa solo el bloque mayoritario "
        "(area_primaria) del etiquetado silver por citas, pero top3_json registra la "
        "distribucion completa de bloques citados. La hipotesis es que parte del techo del "
        "macro-F1 se explica por esa ambiguedad multi-tema y que entrenar con la distribucion "
        "completa (objetivo suave) mejora la prediccion del bloque dominante."
    ),
    (
        "Premisa formal de W2: expandir cada documento a filas (doc, bloque) con peso "
        "q_i = n_i / suma(n_j) (bloques con n>0 dentro de las 16 clases) y entrenar la "
        "perdida multinomial ponderada es EQUIVALENTE a entrenar entropia cruzada contra el "
        "objetivo suave q(x): sum_i q_i * CE(onehot(b_i), p(x)) = CE(q(x), p(x)), porque el "
        "gradiente de la suma ponderada de filas duplicadas coincide con p(x) - q(x). Es el "
        "mismo principio de label smoothing (Szegedy et al., 2016) y soft targets (Hinton et "
        "al., 2015), aplicado aqui con la distribucion empirica de citas de cada documento."
    ),
    (
        "W2: el denominador de los pesos suma SOLO los bloques conservados (n>0 y dentro del "
        "espacio fijo de 16); la masa de bloques fuera del espacio (p.ej. deberes) se "
        "descarta y se cuenta en los metadatos. Los documentos sin ningun bloque en el "
        "espacio se excluyen del train de W2 (se cuentan); en W3 son filas negativas "
        "legitimas (sin bloque citado en el espacio)."
    ),
    (
        "W3: 16 clasificadores binarios LogReg v1 (solver lbfgs) sobre membresia en top3 "
        "(positivo si n>=1 y el bloque pertenece a las 16 clases), agregados con "
        "OneVsRestClassifier(n_jobs=4) y convertidos a single-label por argmax de las "
        "probabilidades por bloque (no calibradas y que no suman 1). Relacion con la variante "
        "B de la Fase U: alli la conversion usaba umbrales por clase optimizados en "
        "validacion; aqui es argmax. NO es un experimento nuevo de la Fase U ni modifica sus "
        "artefactos."
    ),
    (
        "Holdout bloqueado SELLADO: data/holdout_m4_bloqueado.parquet (2.481 docs) se "
        "construyo desde el train canonico y NINGUNA variante de H2 lo evalua ni entrena con "
        "el; solo se usa para excluir sus ids del train reducido. La evaluacion final sobre el "
        "holdout queda reservada y fuera del alcance de este script."
    ),
    (
        "Protocolo de lectura: seleccion por macro-F1 de espacio fijo de 16 en val canonica; "
        "el test canonico se lee UNA sola vez y solo si hay ganador con IC95 del delta "
        "pareado vs W1 (inferior > 0), marcado development_test_reutilizado; W1 se incluye "
        "como comparador reutilizando sus predicciones ya persistidas de H1. El test temporal "
        "se lee solo para el ganador y W1. Si ninguna variante gana no hay lecturas de test."
    ),
    (
        "Matriz de features: se reutiliza resultados/m4_h1_matriz.npz (TF-IDF v1 ajustado "
        "SOLO sobre el train reducido de 14.076 docs: lemas enmascarados, min_df=3, "
        "max_features=100000, ngram (1,2), sublinear_tf=True). Antes de reutilizarla se "
        "verifica su integridad (estructura, espacio de clases, exclusion del holdout, "
        "parametros del vectorizador) y la frescura de las entradas contra los md5 "
        "registrados en el JSON de H1 y en el protocolo. Si falta o no cuadra se reconstruye "
        "con la MISMA configuracion en resultados/m4_h2_matriz.npz y el motivo queda "
        "documentado; no se reentrenan vectorizadores distintos."
    ),
    (
        "W1 no es un modelo nuevo: reutiliza resultados/m4_h1_v1r_preds.npz validando ids y "
        "etiquetas contra la matriz activa. Si falta o no es consistente se reentrena la "
        "configuracion v1r (params v1 leidos del joblib) sobre el train reducido y se guarda "
        "en resultados/m4_h2_w1_preds.npz, documentado en el JSON."
    ),
    (
        "Los targets de W2/W3 se derivan SOLO de los top3 del train reducido (etiquetado "
        "silver por citas a la Constitucion); no se usa informacion de val/test/holdout en "
        "entrenamiento. Las features son exclusivamente el texto enmascarado de lemas (sin "
        "leakage)."
    ),
    (
        "Metricas: macro-F1 de espacio fijo de 16 (zero_division=0) como criterio de "
        "seleccion; IC bootstrap percentil 95 % con 1000 remuestreos de casos y semilla 42 "
        "por variante; comparaciones pareadas W2 vs W1 y W3 vs W1 remuestreando los MISMOS "
        "indices para ambas variantes (delta e IC95 sobre macro-F1 fijo16 y observado). "
        "Ganador: variante con delta puntual > 0 e IC95 inferior > 0; si ambas cumplen gana "
        "la de mayor delta (desempate: mayor macro-F1 fijo16 en val, luego orden w2, w3)."
    ),
    (
        "Paralelizacion y resumibilidad: W2 y W3 corren como subprocesos "
        "(--solo-variante) con --jobs 2 y OMP/OPENBLAS/MKL/NUMEXPR=8 hilos por variante; W3 "
        "usa ademas n_jobs=4 interno. Cada variante guarda resultados/m4_h2_{variante}_preds.npz "
        "y se salta si existe salvo --forzar. Tiempos impresos por seccion."
    ),
    (
        "Modo smoke: subconjunto estratificado de ~500 docs (350/75/75) reutilizando "
        "cargar_datos_smoke del experimento 18, matriz scratch y W1 reentrenado en el "
        "subconjunto; salidas SOLO en /tmp/opencode/h2_smoke/. Los resultados smoke no "
        "deciden gates ni se mezclan con las salidas reales."
    ),
    (
        "El split temporal es una particion alternativa del MISMO corpus, no un holdout "
        "cronologico limpio: una fraccion de sus docs pertenece al train canonico y fue vista "
        "en entrenamiento (conteos temporal_*_en_train_reducido en los metadatos de la "
        "matriz); una lectura estrictamente out-of-sample exigiria reentrenar solo con docs "
        "pre-2018, fuera del alcance de H2."
    ),
]

LOGGER = logging.getLogger("h2_objetivo_suave_19")

# Caches de modulos con nombre que empieza por digito (importlib por ruta).
_MODULO_06 = None
_MODULO_17 = None
_MODULO_18 = None


# ===== SECCION 2: utilidades (logger, md5, versiones, modulos, escritura) =====


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


def modulo_18():
    """Carga (y cachea) scripts/18_h1_ruido_m4.py.

    Returns:
        Modulo 18 ejecutado (utilidades, protocolo, matriz y evaluacion).
    """
    global _MODULO_18
    if _MODULO_18 is None:
        _MODULO_18 = _cargar_modulo(
            "h1_ruido_m4_18_h2", RAIZ / "scripts" / "18_h1_ruido_m4.py"
        )
    return _MODULO_18


def modulo_17():
    """Carga scripts/17_auditoria_etiquetas_m4.py via el cache del modulo 18.

    Returns:
        Modulo 17 ejecutado.
    """
    global _MODULO_17
    if _MODULO_17 is None:
        _MODULO_17 = modulo_18().modulo_17()
    return _MODULO_17


def modulo_06():
    """Carga scripts/06_baseline_clasico.py reutilizando el cargador de 17.

    Returns:
        Modulo 06 ejecutado (CONFIG, cargar_datos, evaluar_single_label, etc.).
    """
    global _MODULO_06
    if _MODULO_06 is None:
        _MODULO_06 = modulo_17().cargar_modulo_baseline()
    return _MODULO_06


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


def rutas_activas(smoke: bool) -> dict:
    """Construye el mapa de rutas de entrada/salida segun el modo smoke.

    En smoke TODAS las salidas van a /tmp/opencode/h2_smoke/; las entradas
    canonicas (datos, protocolo, referencia H1) se leen siempre de sus rutas
    reales de solo lectura.

    Args:
        smoke: True para usar salidas exclusivamente en /tmp/opencode/h2_smoke.

    Returns:
        Dict combinado de rutas de datos y de salidas H2.
    """
    rutas = dict(CONFIG["rutas_datos"])
    if smoke:
        base = Path(CONFIG["dir_smoke"])
        rutas.update(
            {
                "matriz": str(base / "m4_h2_matriz.npz"),
                "matriz_reconstruida": str(base / "m4_h2_matriz.npz"),
                "w1_preds": str(base / "m4_h2_w1_preds.npz"),
                "w1_preds_respaldo": str(base / "m4_h2_w1_preds_respaldo.npz"),
                "preds": str(base / "m4_h2_{variante}_preds.npz"),
                "json": str(base / "m4_h2_objetivo_suave_v1.json"),
                "figura": str(base / "fig_h2_objetivo_suave.png"),
                "log": str(base / "19_h2_objetivo_suave.log"),
                "h1_json": CONFIG["rutas_h2"]["h1_json"],
            }
        )
    else:
        rutas.update(CONFIG["rutas_h2"])
    return rutas


def verificar_entradas(rutas: dict, logger: logging.Logger) -> None:
    """Aborta si falta alguna entrada canonica (solo lectura).

    Args:
        rutas: Dict de rutas activas.
        logger: Logger activo.

    Raises:
        FileNotFoundError: Si falta alguna entrada requerida.
    """
    claves = [
        "lemas",
        "etiquetas",
        "splits",
        "temporal",
        "holdout",
        "modelo_v1",
        "protocolo_json",
    ]
    faltantes = [rutas[clave] for clave in claves if not Path(rutas[clave]).exists()]
    if faltantes:
        raise FileNotFoundError(
            "faltan entradas requeridas (ejecute primero el protocolo del experimento 18): "
            + ", ".join(faltantes)
        )
    if not Path(rutas["h1_json"]).exists():
        logger.warning(
            "sin referencia H1 (%s): no se verificara frescura de la matriz ni se "
            "reportara la referencia de test",
            rutas["h1_json"],
        )
    logger.info("entradas verificadas: %d archivos canonicos", len(claves))


def verificar_protocolo(rutas: dict, logger: logging.Logger) -> dict:
    """Verifica por md5 los artefactos del protocolo vigente (solo lectura).

    Args:
        rutas: Dict de rutas activas.
        logger: Logger activo.

    Returns:
        JSON del protocolo leido.

    Raises:
        RuntimeError: Si los md5 no coinciden con los registrados.
    """
    with open(rutas["protocolo_json"], encoding="utf-8") as archivo:
        protocolo = json.load(archivo)
    registrados = protocolo.get("md5", {})
    for clave, ruta in (
        ("splits_m4_temporal", rutas["temporal"]),
        ("holdout_m4_bloqueado", rutas["holdout"]),
    ):
        actual = md5_archivo(ruta)
        if actual != registrados.get(clave):
            raise RuntimeError(
                f"md5 de {ruta} ({actual}) no coincide con el protocolo ({registrados.get(clave)})"
            )
    logger.info(
        "protocolo verificado por md5: holdout sellado %d docs, train reducido %d docs",
        protocolo["regla_holdout"]["n_holdout"],
        protocolo["regla_holdout"]["n_train_reducido"],
    )
    return protocolo


# ===== SECCION 3: matriz H2 (reutilizacion de la matriz H1 o reconstruccion) =====


def cargar_ids_labels_matriz(ruta: str) -> dict:
    """Lee SOLO ids y etiquetas de los bloques de prediccion de la matriz.

    Evita reconstruir los CSR (memoria consciente): el orquestador solo
    necesita ids/etiquetas para validar la reutilizacion de W1.

    Args:
        ruta: Ruta del .npz de la matriz.

    Returns:
        Dict bloque -> {ids, y} para los bloques de BLOQUES_PRED.
    """
    with np.load(ruta, allow_pickle=False) as datos:
        salida = {}
        for bloque in BLOQUES_PRED:
            salida[bloque] = {
                "ids": np.asarray([str(v) for v in datos[f"{bloque}_ids"]]),
                "y": np.asarray([str(v) for v in datos[f"{bloque}_labels"]]),
            }
    return salida


def verificar_matriz(
    ruta: str, rutas: dict, logger: logging.Logger, referencia_h1: bool
) -> tuple[bool, list[str]]:
    """Verifica integridad y frescura de una matriz TF-IDF de H1/H2.

    Comprueba estructura (claves), coherencia shapes/ids/etiquetas, espacio de
    clases fijo, parametros del vectorizador v1, conteos y exclusion total del
    holdout del train reducido. Si `referencia_h1` es True, compara ademas los
    md5 actuales de las entradas canonicas contra los registrados en el JSON
    de H1 y en el protocolo (matriz potencialmente obsoleta si no coinciden).

    Args:
        ruta: Ruta del .npz.
        rutas: Dict de rutas activas.
        logger: Logger activo.
        referencia_h1: True para verificar frescura contra el registro de H1.

    Returns:
        (es_valida, motivos) con la lista de motivos de descarte.
    """
    motivos: list[str] = []
    if not Path(ruta).exists():
        return False, ["no existe"]
    try:
        with np.load(ruta, allow_pickle=False) as datos:
            claves = set(datos.files)
            requeridas = {"clases", "params_v1_json", "vec_params_json", "conteos_json"}
            for bloque in BLOQUES_MATRIZ:
                requeridas.update(
                    f"{bloque}_{sufijo}"
                    for sufijo in ("data", "indices", "indptr", "shape", "ids", "labels")
                )
            faltan = sorted(requeridas - claves)
            if faltan:
                motivos.append(f"faltan claves: {faltan[:6]}")
            else:
                clases = [str(c) for c in datos["clases"]]
                if clases != CLASES_M4:
                    motivos.append("el espacio de clases no coincide con el fijo de 16")
                vec = json.loads(str(datos["vec_params_json"]))
                if vec != VEC_PARAMS_V1:
                    motivos.append(f"vec_params distintos de v1: {vec}")
                conteos = json.loads(str(datos["conteos_json"]))
                for bloque in BLOQUES_MATRIZ:
                    shape = tuple(int(v) for v in datos[f"{bloque}_shape"])
                    ids = [str(v) for v in datos[f"{bloque}_ids"]]
                    etiquetas = [str(v) for v in datos[f"{bloque}_labels"]]
                    if shape[0] != len(ids) or len(ids) != len(etiquetas):
                        motivos.append(f"{bloque}: shape/ids/etiquetas incoherentes")
                    if not set(etiquetas) <= set(CLASES_M4):
                        motivos.append(f"{bloque}: etiquetas fuera del espacio fijo")
                    if len(set(ids)) != len(ids):
                        motivos.append(f"{bloque}: ids duplicados")
                if "train_reducido" in conteos:
                    n_train = len(datos["train_reducido_ids"])
                    if int(conteos["train_reducido"]) != n_train:
                        motivos.append(
                            "conteos.train_reducido no coincide con las filas de la matriz"
                        )
                ids_holdout = set(
                    pq.read_table(rutas["holdout"]).to_pandas()["id"].astype(str)
                )
                ids_train = {str(v) for v in datos["train_reducido_ids"]}
                interseccion = ids_holdout & ids_train
                if interseccion:
                    motivos.append(
                        f"el holdout aparece en el train reducido: {len(interseccion)} ids"
                    )
    except Exception as error:  # noqa: BLE001 - cualquier fallo de lectura invalida la matriz
        motivos.append(f"error leyendo la matriz: {error}")
    if not motivos and referencia_h1:
        if Path(rutas["h1_json"]).exists():
            with open(rutas["h1_json"], encoding="utf-8") as archivo:
                h1 = json.load(archivo)
            registrados = h1.get("md5_entradas", {})
            for clave in ("lemas", "etiquetas", "splits", "modelo_v1"):
                if md5_archivo(rutas[clave]) != registrados.get(clave):
                    motivos.append(
                        f"md5 de {clave} distinto al registrado en H1 (matriz potencialmente obsoleta)"
                    )
        with open(rutas["protocolo_json"], encoding="utf-8") as archivo:
            protocolo = json.load(archivo)
        registrados = protocolo.get("md5", {})
        if md5_archivo(rutas["temporal"]) != registrados.get("splits_m4_temporal"):
            motivos.append("md5 del split temporal distinto al del protocolo")
        if md5_archivo(rutas["holdout"]) != registrados.get("holdout_m4_bloqueado"):
            motivos.append("md5 del holdout distinto al del protocolo")
    valida = not motivos
    if valida:
        logger.info(
            "matriz verificada: %s (%s)",
            ruta,
            "referencia H1" if referencia_h1 else "reconstruida",
        )
    else:
        logger.warning("matriz descartada: %s | motivos: %s", ruta, "; ".join(motivos))
    return valida, motivos


def elegir_matriz_activa(rutas: dict, logger: logging.Logger) -> tuple[str | None, dict]:
    """Elige la matriz activa: H1 reutilizada o reconstruida de H2.

    Args:
        rutas: Dict de rutas activas.
        logger: Logger activo.

    Returns:
        (ruta activa o None, info con origen y motivos de descarte).
    """
    ok_h1, motivos_h1 = verificar_matriz(rutas["matriz"], rutas, logger, referencia_h1=True)
    if ok_h1:
        return rutas["matriz"], {
            "origen": "reutilizada_h1",
            "md5": md5_archivo(rutas["matriz"]),
            "motivos_descarte_h1": None,
            "motivos_descarte_reconstruida": None,
        }
    ok_re, motivos_re = verificar_matriz(
        rutas["matriz_reconstruida"], rutas, logger, referencia_h1=False
    )
    if ok_re:
        return rutas["matriz_reconstruida"], {
            "origen": "reconstruida_h2",
            "md5": md5_archivo(rutas["matriz_reconstruida"]),
            "motivos_descarte_h1": motivos_h1,
            "motivos_descarte_reconstruida": None,
        }
    return None, {
        "origen": None,
        "motivos_descarte_h1": motivos_h1,
        "motivos_descarte_reconstruida": motivos_re,
    }


def resolver_ruta_matriz(smoke: bool, rutas: dict, logger: logging.Logger) -> tuple[str, dict]:
    """Resuelve la matriz activa para orquestador y subprocesos.

    Args:
        smoke: True si es corrida smoke (matriz scratch).
        rutas: Dict de rutas activas.
        logger: Logger activo.

    Returns:
        (ruta de la matriz, info con origen/md5/motivos).

    Raises:
        FileNotFoundError: Si no hay matriz smoke ni matriz valida reutilizable.
    """
    if smoke:
        ruta = rutas["matriz"]
        if not Path(ruta).exists():
            raise FileNotFoundError(f"falta la matriz smoke: {ruta}")
        return ruta, {
            "origen": "smoke_construida",
            "md5": md5_archivo(ruta),
            "motivos_descarte_h1": None,
            "motivos_descarte_reconstruida": None,
        }
    ruta, info = elegir_matriz_activa(rutas, logger)
    if ruta is None:
        raise FileNotFoundError(
            "sin matriz valida (H1: "
            f"{info['motivos_descarte_h1']}; reconstruida: {info['motivos_descarte_reconstruida']})"
        )
    info["md5"] = md5_archivo(ruta)
    return ruta, info


def construir_bloques_smoke(rutas: dict, logger: logging.Logger) -> tuple[dict, dict]:
    """Construye la matriz TF-IDF scratch del smoke (~500 docs).

    Reutiliza cargar_datos_smoke del modulo 18 (subconjunto estratificado
    350/75/75 con particiones canonicas) y ajusta el TF-IDF v1 SOLO sobre su
    train reducido. Los bloques temporales se derivan del split temporal real
    (solo lectura) intersectado con el subconjunto.

    Args:
        rutas: Dict de rutas activas (smoke).
        logger: Logger activo.

    Returns:
        (bloques {nombre: {X, ids, y}}, metadatos con conteos y params).

    Raises:
        RuntimeError: Si el holdout cae fuera del train o algun bloque queda vacio.
    """
    base = modulo_06()
    mod18 = modulo_18()
    df = mod18.cargar_datos_smoke(logger)
    temporal = pq.read_table(CONFIG["rutas_datos"]["temporal"]).to_pandas()
    holdout = pq.read_table(CONFIG["rutas_datos"]["holdout"]).to_pandas()
    n_antes = len(df)
    df = df.merge(temporal, on="id", how="inner")
    if len(df) != n_antes:
        raise RuntimeError(
            f"el split temporal cubre {len(df)} de {n_antes} docs del smoke: desalineados"
        )
    ids_str = df["id"].astype(str)
    ids_holdout = set(holdout["id"].astype(str))
    ids_train = set(ids_str[df["particion"] == "train"])
    ids_holdout_presentes = ids_holdout & set(ids_str)
    if not ids_holdout_presentes <= ids_train:
        raise RuntimeError("el holdout contiene ids fuera del train canonico del smoke")
    subconjuntos = {
        "train_reducido": df[(df["particion"] == "train") & (~ids_str.isin(ids_holdout))],
        "val": df[df["particion"] == "val"],
        "test": df[df["particion"] == "test"],
        "temporal_val": df[df["particion_temporal"] == "val"],
        "temporal_test": df[df["particion_temporal"] == "test"],
    }
    for nombre, sub in subconjuntos.items():
        if sub.empty:
            raise RuntimeError(f"bloque smoke vacio: {nombre}")
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
        "matriz smoke: %d features, train reducido %d docs; bloques %s",
        metadatos["n_features"],
        conteos["train_reducido"],
        {k: int(v["X"].shape[0]) for k, v in bloques.items()},
    )
    return bloques, metadatos


def reconstruir_matriz_full(rutas: dict, logger: logging.Logger) -> tuple[dict, dict]:
    """Reconstruye la matriz TF-IDF v1 sobre el train reducido (fallback).

    Reutiliza integramente el pipeline del experimento 18: cargar_datos_h1
    (lemas + etiquetas + splits + split temporal), construir_bloques_matriz
    (TF-IDF v1 sobre train reducido sin holdout) y sus metadatos.

    Args:
        rutas: Dict de rutas activas.
        logger: Logger activo.

    Returns:
        (bloques, metadatos) en el formato estandar de la matriz H1.

    Raises:
        FileNotFoundError: Si falta data/lemas_por_doc.parquet.
    """
    if not Path(rutas["lemas"]).exists():
        raise FileNotFoundError(f"falta {rutas['lemas']} para reconstruir la matriz")
    mod18 = modulo_18()
    logger.warning("reconstruyendo matriz TF-IDF v1 (fallback): esto puede tardar varios minutos")
    df = mod18.cargar_datos_h1(False, logger)
    bloques, metadatos = mod18.construir_bloques_matriz(df, False, logger)
    return bloques, metadatos


def preparar_matriz_h2(args, rutas: dict, logger: logging.Logger) -> dict:
    """Construye o reutiliza la matriz activa de H2 y devuelve sus metadatos.

    Memoria consciente: los bloques se liberan (del/gc) antes de lanzar los
    subprocesos; cada variante carga su propia copia.

    Args:
        args: Argumentos CLI (smoke, forzar).
        rutas: Dict de rutas activas.
        logger: Logger activo.

    Returns:
        Dict con ruta, origen, md5, motivos, conteos, params y tiempos.

    Raises:
        FileExistsError: Si hay que reconstruir y la salida existe sin --forzar.
    """
    t0 = time.perf_counter()
    mod18 = modulo_18()
    if args.smoke:
        destino = rutas["matriz"]
        bloques, metadatos = construir_bloques_smoke(rutas, logger)
        mod18.guardar_matriz_h1(destino, bloques, metadatos, logger)
        del bloques
        gc.collect()
        return {
            "ruta": destino,
            "origen": "smoke_construida",
            "md5": md5_archivo(destino),
            "motivos_descarte_h1": None,
            "motivos_descarte_reconstruida": None,
            "conteos": metadatos["conteos"],
            "params_v1": metadatos["params_v1"],
            "vec_params": metadatos["vec_params"],
            "n_features": metadatos["n_features"],
            "tiempos_seg": {"total": time.perf_counter() - t0},
        }
    ruta, info = elegir_matriz_activa(rutas, logger)
    if ruta is not None:
        metadatos = mod18.cargar_metadatos_matriz(ruta)
        logger.info(
            "matriz H2 activa: %s (origen %s) [%.1f s]",
            ruta,
            info["origen"],
            time.perf_counter() - t0,
        )
        return {
            "ruta": ruta,
            "origen": info["origen"],
            "md5": info["md5"],
            "motivos_descarte_h1": info["motivos_descarte_h1"],
            "motivos_descarte_reconstruida": info["motivos_descarte_reconstruida"],
            "conteos": metadatos["conteos"],
            "params_v1": metadatos["params_v1"],
            "vec_params": metadatos["vec_params"],
            "n_features": metadatos["n_features"],
            "tiempos_seg": {"total": time.perf_counter() - t0},
        }
    destino = rutas["matriz_reconstruida"]
    if Path(destino).exists() and not args.forzar:
        raise FileExistsError(
            f"matriz reconstruida existente e invalida (no se sobrescribe; use --forzar): {destino}"
        )
    logger.warning("H1 descartada (%s); se reconstruye la matriz", info["motivos_descarte_h1"])
    t = time.perf_counter()
    bloques, metadatos = reconstruir_matriz_full(rutas, logger)
    mod18.guardar_matriz_h1(destino, bloques, metadatos, logger)
    del bloques
    gc.collect()
    logger.info("matriz reconstruida en %.1f s: %s", time.perf_counter() - t, destino)
    return {
        "ruta": destino,
        "origen": "reconstruida_h2",
        "md5": md5_archivo(destino),
        "motivos_descarte_h1": info["motivos_descarte_h1"],
        "motivos_descarte_reconstruida": info["motivos_descarte_reconstruida"],
        "conteos": metadatos["conteos"],
        "params_v1": metadatos["params_v1"],
        "vec_params": metadatos["vec_params"],
        "n_features": metadatos["n_features"],
        "tiempos_seg": {"total": time.perf_counter() - t0},
    }


# ===== SECCION 4: W1 (baseline reutilizado o reentrenado de respaldo) =====


def validar_w1(datos: dict, ids_labels: dict) -> list[str]:
    """Valida que las predicciones W1 sean reutilizables con la matriz activa.

    Args:
        datos: Salida de cargar_preds_variante del modulo 18.
        ids_labels: Ids/etiquetas de la matriz activa por bloque.

    Returns:
        Lista de motivos de rechazo (vacia si es reutilizable).
    """
    motivos: list[str] = []
    if datos["variante"] not in ("v1r", "w1"):
        motivos.append(f"variante inesperada: {datos['variante']}")
    for bloque in BLOQUES_PRED:
        esperado = ids_labels[bloque]
        if not np.array_equal(datos[bloque]["ids"], esperado["ids"]):
            motivos.append(f"{bloque}: ids no coinciden con la matriz activa")
        if not np.array_equal(datos[bloque]["y"], esperado["y"]):
            motivos.append(f"{bloque}: etiquetas no coinciden con la matriz activa")
        probs = datos[bloque]["probs"]
        if probs.shape != (len(esperado["ids"]), len(CLASES_M4)):
            motivos.append(f"{bloque}: shape de probabilidades inesperado {probs.shape}")
        elif not np.isfinite(probs).all():
            motivos.append(f"{bloque}: probabilidades no finitas")
    return motivos


def _entrenar_w1(
    ruta_salida: str, matriz: dict, permitir: bool, smoke: bool, logger: logging.Logger
) -> dict:
    """Entrena el baseline v1r (params v1) sobre el train reducido y guarda.

    Args:
        ruta_salida: Ruta del .npz de preds W1.
        matriz: Matriz cargada (formato del modulo 18).
        permitir: True para permitir reescritura (smoke/--forzar).
        smoke: True si es corrida smoke.
        logger: Logger activo.

    Returns:
        Dict con origen, ruta, md5, datos y metadatos.

    Raises:
        FileExistsError: Si la salida existe y no se permite reescribir.
    """
    mod18 = modulo_18()
    if Path(ruta_salida).exists() and not permitir:
        raise FileExistsError(f"preds W1 de respaldo existentes (no se sobrescribe): {ruta_salida}")
    params = matriz["params_v1"]
    t = time.perf_counter()
    modelo = LogisticRegression(**params)
    modelo.fit(matriz["train_reducido"]["X"], matriz["train_reducido"]["y"])
    fit_seg = time.perf_counter() - t
    t = time.perf_counter()
    preds = {
        bloque: modulo_17().expandir_probs(modelo, matriz[bloque]["X"], CLASES_M4)
        for bloque in BLOQUES_PRED
    }
    pred_seg = time.perf_counter() - t
    metadatos = {
        "modo": (
            "baseline v1r reentrenado en el train reducido del smoke"
            if smoke
            else "baseline v1r reentrenado como respaldo (preds de H1 no reutilizables)"
        ),
        "variante": "w1",
        "smoke": bool(smoke),
        "fecha": pd.Timestamp.now().isoformat(),
        "n_entrenados": int(matriz["train_reducido"]["X"].shape[0]),
        "n_iter": int(modelo.n_iter_.max()),
        "params_v1": params,
        "tiempos_seg": {"fit_seg": fit_seg, "pred_seg": pred_seg},
    }
    mod18.guardar_preds_variante(ruta_salida, "w1", matriz, preds, metadatos, permitir, logger)
    datos = mod18.cargar_preds_variante(ruta_salida)
    del modelo, preds
    gc.collect()
    logger.info(
        "W1 reentrenado (%s): fit %.1f s, pred %.1f s, n=%d",
        ruta_salida,
        fit_seg,
        pred_seg,
        metadatos["n_entrenados"],
    )
    return {
        "origen": "reentrenado_smoke" if smoke else "reentrenado_respaldo",
        "ruta": ruta_salida,
        "md5": md5_archivo(ruta_salida),
        "datos": datos,
        "metadatos": metadatos,
        "validacion": [],
        "tiempos_seg": {"fit_seg": fit_seg, "pred_seg": pred_seg},
    }


def preparar_w1(args, rutas: dict, logger: logging.Logger, info_matriz: dict) -> dict:
    """Prepara W1: reutiliza las preds v1r de H1 o reentrena de respaldo.

    En modo completo valida ids/etiquetas/shapes contra la matriz activa; en
    smoke reentrena siempre (la matriz smoke es distinta). No es un modelo
    nuevo: la reutilizacion no reentrena nada.

    Args:
        args: Argumentos CLI (smoke, forzar).
        rutas: Dict de rutas activas.
        logger: Logger activo.
        info_matriz: Info de la matriz activa (ruta).

    Returns:
        Dict con origen, ruta, md5, datos (bloques ids/y/probs) y metadatos.

    Raises:
        FileExistsError: Si el respaldo existe y no se permite reescribir.
    """
    t0 = time.perf_counter()
    mod18 = modulo_18()
    if not args.smoke and Path(rutas["w1_preds"]).exists():
        ids_labels = cargar_ids_labels_matriz(info_matriz["ruta"])
        datos = mod18.cargar_preds_variante(rutas["w1_preds"])
        motivos = validar_w1(datos, ids_labels)
        if not motivos:
            logger.info(
                "W1 reutilizado de H1: %s (variante %s) [%.1f s]",
                rutas["w1_preds"],
                datos["variante"],
                time.perf_counter() - t0,
            )
            return {
                "origen": "reutilizado_h1",
                "ruta": rutas["w1_preds"],
                "md5": md5_archivo(rutas["w1_preds"]),
                "datos": datos,
                "metadatos": datos["metadatos"],
                "validacion": [],
                "tiempos_seg": {"validacion_seg": time.perf_counter() - t0},
            }
        logger.warning(
            "W1 de H1 no reutilizable (%s); se reentrena respaldo", "; ".join(motivos)
        )
    ruta_salida = rutas["w1_preds"] if args.smoke else rutas["w1_preds_respaldo"]
    permitir = args.smoke or args.forzar
    matriz = mod18.cargar_matriz_h1(info_matriz["ruta"])
    resultado = _entrenar_w1(ruta_salida, matriz, permitir, args.smoke, logger)
    del matriz
    gc.collect()
    return resultado


# ===== SECCION 5: variantes W2/W3 (targets top3 + entrenamiento + preds) =====


def cargar_top3(ids: np.ndarray, logger: logging.Logger) -> tuple[pd.DataFrame, dict]:
    """Carga el top3_json de los documentos pedidos (solo train reducido).

    Args:
        ids: Ids del train reducido (strings, orden de la matriz).
        logger: Logger activo.

    Returns:
        (DataFrame con id + top3_json alineado a ids, dict id -> [(bloque, n)]).

    Raises:
        RuntimeError: Si falta algun id en etiquetas_areas.parquet.
    """
    if len(ids) == 0:
        raise RuntimeError("cargar_top3: la lista de ids del train reducido esta vacia")
    etiquetas = pq.read_table(CONFIG["rutas_datos"]["etiquetas"]).to_pandas()
    crudos = {
        str(valor_id): texto
        for valor_id, texto in zip(etiquetas["id"], etiquetas["top3_json"])
    }
    faltantes = [valor for valor in ids if valor not in crudos]
    if faltantes:
        raise RuntimeError(
            f"{len(faltantes)} ids del train reducido sin top3_json: {faltantes[:5]}"
        )
    mapa = {
        valor: [
            (str(item["bloque"]), int(item["n"])) for item in json.loads(crudos[valor])
        ]
        for valor in ids
    }
    df_top3 = pd.DataFrame(
        {"id": np.asarray(ids), "top3_json": [crudos[valor] for valor in ids]}
    )
    n_items = sum(len(v) for v in mapa.values())
    logger.info("top3 cargado: %d docs, %d items (bloques por doc medio %.2f)", len(ids), n_items, n_items / len(ids))
    return df_top3, mapa


def expandir_entrenamiento_ponderado(
    ids: np.ndarray, top3: dict, logger: logging.Logger
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Expande (doc, bloque) con pesos n_i / suma(n) sobre el espacio de 16.

    La masa de bloques fuera del espacio fijo se descarta y se cuenta; los
    documentos sin ningun bloque en el espacio quedan fuera de la expansion.

    Args:
        ids: Ids del train reducido (orden de la matriz).
        top3: Dict id -> [(bloque, n)].
        logger: Logger activo.

    Returns:
        (filas, etiquetas, pesos, detalle) donde filas indexa la matriz.
    """
    indice_clase = {clase: j for j, clase in enumerate(CLASES_M4)}
    filas: list[int] = []
    etiquetas: list[str] = []
    pesos: list[float] = []
    docs_sin_bloque = 0
    items_fuera = 0
    items_no_positivos = 0
    for i, doc in enumerate(ids):
        conservados: list[tuple[str, int]] = []
        for bloque, n in top3.get(doc, []):
            if n <= 0:
                items_no_positivos += 1
            elif bloque not in indice_clase:
                items_fuera += 1
            else:
                conservados.append((bloque, n))
        if not conservados:
            docs_sin_bloque += 1
            continue
        suma = float(sum(n for _, n in conservados))
        for bloque, n in conservados:
            filas.append(i)
            etiquetas.append(bloque)
            pesos.append(n / suma)
    detalle = {
        "n_filas": len(filas),
        "n_docs_expandidos": len(set(filas)),
        "docs_sin_bloque_espacio": docs_sin_bloque,
        "items_fuera_espacio": items_fuera,
        "items_no_positivos": items_no_positivos,
    }
    logger.info(
        "expansion W2: %d filas de %d docs (sin bloque en espacio: %d; items fuera: %d)",
        detalle["n_filas"],
        len(ids),
        docs_sin_bloque,
        items_fuera,
    )
    return (
        np.asarray(filas, dtype=np.int64),
        np.asarray(etiquetas),
        np.asarray(pesos, dtype=np.float64),
        detalle,
    )


def ajustar_w2(
    X, ids: np.ndarray, top3: dict, params: dict, logger: logging.Logger
) -> tuple[LogisticRegression, dict]:
    """Ajusta W2: LogReg multinomial v1 sobre la expansion ponderada.

    Premisa: la perdida multinomial ponderada sobre filas duplicadas
    (doc, bloque) con peso q_i es identica a la entropia cruzada contra el
    objetivo suave q(x) del documento (ver premisas del modulo).

    Args:
        X: Matriz TF-IDF del train reducido (CSR).
        ids: Ids del train reducido.
        top3: Dict id -> [(bloque, n)].
        params: Hiperparametros v1 de LogisticRegression.
        logger: Logger activo.

    Returns:
        (modelo ajustado, metadatos de la variante).

    Raises:
        RuntimeError: Si la expansion queda vacia.
    """
    filas, etiquetas, pesos, detalle = expandir_entrenamiento_ponderado(ids, top3, logger)
    if len(filas) == 0:
        raise RuntimeError("W2: la expansion ponderada quedo vacia")
    X_exp = X[filas]
    t = time.perf_counter()
    modelo = LogisticRegression(**params)
    modelo.fit(X_exp, etiquetas, sample_weight=pesos)
    fit_seg = time.perf_counter() - t
    metadatos = {
        "modo": (
            "expansion ponderada (doc, bloque) con sample_weight; equivale a entropia "
            "cruzada contra el objetivo suave q(x) del top3"
        ),
        "n_entrenados": int(X_exp.shape[0]),
        "n_docs_expandidos": detalle["n_docs_expandidos"],
        "docs_sin_bloque_espacio": detalle["docs_sin_bloque_espacio"],
        "items_fuera_espacio": detalle["items_fuera_espacio"],
        "items_no_positivos": detalle["items_no_positivos"],
        "pesos": {
            "min": float(pesos.min()),
            "media": float(pesos.mean()),
            "max": float(pesos.max()),
        },
        "n_iter": int(modelo.n_iter_.max()),
        "fit_seg": fit_seg,
    }
    del X_exp
    gc.collect()
    logger.info(
        "W2: %d filas expandidas de %d docs | iters=%d fit=%.1f s",
        metadatos["n_entrenados"],
        detalle["n_docs_expandidos"],
        metadatos["n_iter"],
        fit_seg,
    )
    return modelo, metadatos


def ajustar_w3(
    X, df_top3: pd.DataFrame, params: dict, logger: logging.Logger
) -> tuple[OneVsRestClassifier, dict, list[int]]:
    """Ajusta W3: 16 LogReg binarios v1 (OvR) sobre membresia en top3.

    Reutiliza cargar_targets_multietiqueta del modulo 06 (positivo si n>=1 y
    el bloque pertenece al espacio fijo). Las clases sin positivos en train se
    excluyen del ajuste (sus probabilidades quedan en 0 al predecir).

    Args:
        X: Matriz TF-IDF del train reducido (CSR).
        df_top3: DataFrame id + top3_json alineado al train reducido.
        params: Hiperparametros v1 de LogisticRegression.
        logger: Logger activo.

    Returns:
        (modelo OvR, metadatos, indices de clases presentes).

    Raises:
        RuntimeError: Si hay menos de 2 clases con positivos en train.
    """
    base = modulo_06()
    Y = base.cargar_targets_multietiqueta(df_top3, CLASES_M4)
    positivos = Y.sum(axis=0)
    presentes = [j for j in range(len(CLASES_M4)) if int(positivos[j]) > 0]
    ausentes = [CLASES_M4[j] for j in range(len(CLASES_M4)) if int(positivos[j]) == 0]
    if len(presentes) < 2:
        raise RuntimeError(f"W3: solo {len(presentes)} clases con positivos en train")
    Y_pres = Y[:, presentes]
    t = time.perf_counter()
    modelo = OneVsRestClassifier(
        LogisticRegression(**params), n_jobs=CONFIG["n_jobs_ovr"]
    )
    modelo.fit(X, Y_pres)
    fit_seg = time.perf_counter() - t
    metadatos = {
        "modo": (
            "OvR multietiqueta (membresia en top3) con conversion single-label por "
            "argmax de las probabilidades por bloque"
        ),
        "n_entrenados": int(X.shape[0]),
        "n_positivos_por_clase": {
            clase: int(positivos[j]) for j, clase in enumerate(CLASES_M4)
        },
        "clases_sin_positivos_train": ausentes,
        "n_docs_sin_positivos": int((Y.sum(axis=1) == 0).sum()),
        "n_jobs_ovr": int(CONFIG["n_jobs_ovr"]),
        "relacion_fase_u": (
            "replica la variante B de la Fase U (OvR multietiqueta) con argmax en vez de "
            "umbrales optimizados; no es un experimento nuevo de la Fase U"
        ),
        "n_iter_max": int(max(estimador.n_iter_.max() for estimador in modelo.estimators_)),
        "fit_seg": fit_seg,
    }
    logger.info(
        "W3: %d binarios (presentes %d, sin positivos %d) | iters_max=%d fit=%.1f s",
        len(presentes),
        len(presentes),
        len(ausentes),
        metadatos["n_iter_max"],
        fit_seg,
    )
    return modelo, metadatos, presentes


def expandir_probs_ovr(modelo: OneVsRestClassifier, X, presentes: list[int]) -> np.ndarray:
    """Expande las probabilidades OvR al espacio fijo de 16 clases (ausentes 0).

    Args:
        modelo: OneVsRestClassifier ajustado sobre las clases presentes.
        X: Matriz de features a predecir.
        presentes: Indices (en el espacio fijo) de las clases ajustadas.

    Returns:
        Matriz (n, 16) con las probabilidades por bloque.

    Raises:
        RuntimeError: Si el modelo devuelve un numero de columnas inesperado.
    """
    probs = modelo.predict_proba(X)
    if probs.shape[1] != len(presentes):
        raise RuntimeError(
            f"W3: predict_proba devolvio {probs.shape[1]} columnas, se esperaban {len(presentes)}"
        )
    salida = np.zeros((X.shape[0], len(CLASES_M4)), dtype=np.float64)
    for j, columna in enumerate(presentes):
        salida[:, columna] = probs[:, j]
    return salida


def ejecutar_variante(nombre: str, args, logger: logging.Logger, rutas: dict) -> None:
    """Cuerpo del subproceso --solo-variante: ajusta y predice W2 o W3.

    Args:
        nombre: w2 (expansion ponderada) o w3 (OvR multietiqueta).
        args: Argumentos CLI (forzar, smoke).
        logger: Logger activo (solo consola; el orquestador canaliza).
        rutas: Dict de rutas activas.

    Raises:
        FileNotFoundError: Si falta la matriz activa.
        RuntimeError: Si el espacio de clases de la matriz no es el fijo.
    """
    ruta_preds = rutas["preds"].format(variante=nombre)
    permitir = args.smoke or args.forzar
    if Path(ruta_preds).exists() and not permitir:
        logger.info("variante %s: preds existentes, se reutilizan (%s)", nombre, ruta_preds)
        return
    t0 = time.perf_counter()
    tiempos: dict = {}
    mod18 = modulo_18()
    ruta_matriz, info_matriz = resolver_ruta_matriz(args.smoke, rutas, logger)
    t = time.perf_counter()
    matriz = mod18.cargar_matriz_h1(ruta_matriz)
    tiempos["carga_matriz_seg"] = time.perf_counter() - t
    clases = [str(c) for c in matriz["clases"]]
    if clases != CLASES_M4:
        raise RuntimeError(f"espacio de clases de la matriz distinto del fijo: {clases}")
    params = matriz["params_v1"]
    X_train = matriz["train_reducido"]["X"]
    ids_train = matriz["train_reducido"]["ids"]
    t = time.perf_counter()
    df_top3, top3 = cargar_top3(ids_train, logger)
    tiempos["carga_top3_seg"] = time.perf_counter() - t
    t = time.perf_counter()
    if nombre == "w2":
        modelo, metadatos = ajustar_w2(X_train, ids_train, top3, params, logger)
    else:
        modelo, metadatos, presentes = ajustar_w3(X_train, df_top3, params, logger)
    tiempos["fit_seg"] = time.perf_counter() - t
    # Liberar la matriz de train antes de predecir (memoria consciente).
    matriz["train_reducido"]["X"] = None
    del X_train
    gc.collect()
    t = time.perf_counter()
    if nombre == "w2":
        preds = {
            bloque: modulo_17().expandir_probs(modelo, matriz[bloque]["X"], CLASES_M4)
            for bloque in BLOQUES_PRED
        }
    else:
        preds = {
            bloque: expandir_probs_ovr(modelo, matriz[bloque]["X"], presentes)
            for bloque in BLOQUES_PRED
        }
    tiempos["pred_seg"] = time.perf_counter() - t
    metadatos.update(
        {
            "variante": nombre,
            "smoke": bool(args.smoke),
            "fecha": pd.Timestamp.now().isoformat(),
            "n_train_reducido": len(ids_train),
            "params_v1": params,
            "ruta_matriz": ruta_matriz,
            "origen_matriz": info_matriz.get("origen"),
            "tiempos_seg": tiempos,
            "tiempo_total_seg": time.perf_counter() - t0,
        }
    )
    mod18.guardar_preds_variante(ruta_preds, nombre, matriz, preds, metadatos, permitir, logger)
    logger.info(
        "variante %s terminada en %.1f s (fit %.1f s, pred %.1f s)",
        nombre,
        time.perf_counter() - t0,
        tiempos["fit_seg"],
        tiempos["pred_seg"],
    )
    del matriz, modelo, preds
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


# ===== SECCION 7: metricas, comparaciones pareadas, decision y figura =====


def bootstrap_pareado_fijo16(
    y_true,
    pred_a,
    pred_b,
    n_boot: int,
    semilla: int,
    logger: logging.Logger,
    nombre: str,
) -> dict:
    """IC95 bootstrap pareado de la diferencia de macro-F1 (fijo16 y observado).

    Remuestrea documentos con reemplazo (los MISMOS indices para A y B) y
    calcula macro-F1(a) - macro-F1(b) en espacio fijo de 16 (criterio de
    seleccion) y en el protocolo observado. p-valor bilateral aproximado por
    proporcion de deltas a cada lado de 0 (2*min(cola)). Reutiliza
    evaluar_single_label del modulo 06.

    Args:
        y_true: Etiquetas reales de validacion.
        pred_a: Predicciones de la variante candidata.
        pred_b: Predicciones de W1 (referencia).
        n_boot: Numero de remuestreos.
        semilla: Semilla del generador.
        logger: Logger activo.
        nombre: Etiqueta de la comparacion para el log.

    Returns:
        Dict con delta e IC95 en espacio fijo y observado, p-valor y tiempos.
    """
    base = modulo_06()
    y_true = np.asarray(y_true)
    pred_a = np.asarray(pred_a)
    pred_b = np.asarray(pred_b)
    n = len(y_true)
    rng = np.random.default_rng(semilla)
    t0 = time.perf_counter()
    deltas_fijo = np.empty(n_boot, dtype=np.float64)
    deltas_obs = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        met_a = base.evaluar_single_label(y_true[idx], pred_a[idx], CLASES_M4)
        met_b = base.evaluar_single_label(y_true[idx], pred_b[idx], CLASES_M4)
        deltas_fijo[b] = met_a["macro_f1_espacio_fijo"] - met_b["macro_f1_espacio_fijo"]
        deltas_obs[b] = met_a["macro_f1"] - met_b["macro_f1"]
    punto_fijo = base.evaluar_single_label(y_true, pred_a, CLASES_M4)[
        "macro_f1_espacio_fijo"
    ] - base.evaluar_single_label(y_true, pred_b, CLASES_M4)["macro_f1_espacio_fijo"]
    punto_obs = base.evaluar_single_label(y_true, pred_a, CLASES_M4)["macro_f1"] - base.evaluar_single_label(
        y_true, pred_b, CLASES_M4
    )["macro_f1"]

    def _resumen(deltas: np.ndarray, punto: float) -> dict:
        inferior, superior = np.percentile(deltas, [2.5, 97.5])
        return {
            "punto": float(punto),
            "media_bootstrap": float(deltas.mean()),
            "ic95_inf": float(inferior),
            "ic95_sup": float(superior),
            "p_valor_bilateral": float(
                2.0 * min(float(np.mean(deltas <= 0.0)), float(np.mean(deltas >= 0.0)))
            ),
        }

    resultado = {
        "delta_macro_f1_espacio_fijo": _resumen(deltas_fijo, float(punto_fijo)),
        "delta_macro_f1_observado": _resumen(deltas_obs, float(punto_obs)),
        "n_bootstrap": int(n_boot),
        "semilla": int(semilla),
        "tiempo_seg": time.perf_counter() - t0,
    }
    resumen = resultado["delta_macro_f1_espacio_fijo"]
    logger.info(
        "%s: delta fijo16=%+.4f IC95=[%+.4f, %+.4f] p=%.3f (obs %+.4f) [%.1f s]",
        nombre,
        resumen["punto"],
        resumen["ic95_inf"],
        resumen["ic95_sup"],
        resumen["p_valor_bilateral"],
        resultado["delta_macro_f1_observado"]["punto"],
        resultado["tiempo_seg"],
    )
    return resultado


def seleccionar_ganador(val_metricas: dict, comparaciones: dict) -> str | None:
    """Selecciona la variante ganadora por IC pareado vs W1 en val.

    Criterio: delta puntual > 0 e IC95 inferior > 0 (macro-F1 fijo16). Si
    ambas cumplen, gana la de mayor delta (desempate: mayor macro-F1 fijo16 en
    val, luego orden w2, w3).

    Args:
        val_metricas: Dict variante -> metricas de val canonica.
        comparaciones: Dict nombre_vs_w1 -> bootstrap pareado.

    Returns:
        Nombre de la variante ganadora o None si ninguna supera a W1.
    """
    cumplen = []
    for nombre in VARIANTES:
        comp = comparaciones[f"{nombre}_vs_w1"]["delta_macro_f1_espacio_fijo"]
        if comp["punto"] > 0 and comp["ic95_inf"] > 0:
            cumplen.append(nombre)
    if not cumplen:
        return None

    def clave(nombre: str) -> tuple:
        comp = comparaciones[f"{nombre}_vs_w1"]["delta_macro_f1_espacio_fijo"]
        return (
            comp["punto"],
            val_metricas[nombre]["macro_f1_espacio_fijo"],
            -VARIANTES.index(nombre),
        )

    return max(cumplen, key=clave)


def cargar_referencia_h1(rutas: dict, logger: logging.Logger) -> dict:
    """Lee la referencia de H1 (negativo) desde su JSON canonico.

    Args:
        rutas: Dict de rutas activas.
        logger: Logger activo.

    Returns:
        Dict con disponibilidad, val/test de v1r y resultado de H1.
    """
    salida = {"disponible": False}
    try:
        with open(rutas["h1_json"], encoding="utf-8") as archivo:
            h1 = json.load(archivo)
        salida = {
            "disponible": True,
            "fecha": h1.get("fecha"),
            "val_v1r_macro_f1_espacio_fijo": h1["decision"]["val_macro_f1_espacio_fijo"]["v1r"],
            "test_v1r_macro_f1_espacio_fijo": h1["decision"]["test_dev"]["v1r"],
            "resultado_h1": "negativo (v1r gano en val; sin adopcion)",
        }
        logger.info(
            "referencia H1: val v1r=%.4f | test v1r=%.4f (development_test_reutilizado)",
            salida["val_v1r_macro_f1_espacio_fijo"],
            salida["test_v1r_macro_f1_espacio_fijo"],
        )
    except (FileNotFoundError, KeyError, TypeError) as error:
        logger.warning("sin referencia H1 (%s)", error)
    return salida


def construir_decision(
    val_metricas: dict,
    comparaciones: dict,
    ganador: str | None,
    variantes_resultado: dict,
    referencia_h1: dict,
) -> dict:
    """Construye el bloque de decision del JSON de H2.

    Args:
        val_metricas: Dict variante -> metricas de val canonica.
        comparaciones: Dict nombre_vs_w1 -> bootstrap pareado.
        ganador: Variante ganadora por val (o None).
        variantes_resultado: Dict variante -> resultado (con test/temporal).
        referencia_h1: Referencia de H1.

    Returns:
        Dict de decision con lecturas y metricas de test/temporal.
    """
    candidatos = []
    for nombre in VARIANTES:
        comp = comparaciones[f"{nombre}_vs_w1"]["delta_macro_f1_espacio_fijo"]
        if comp["punto"] > 0 and comp["ic95_inf"] > 0:
            candidatos.append(nombre)
    decision = {
        "ganador_por_val": ganador,
        "resultado": "positivo" if ganador else "negativo",
        "criterio": (
            "delta pareado (macro-F1 fijo16) de la variante vs W1 en val canonica con "
            "IC95 inferior > 0; si ambas cumplen, mayor delta (desempate: mayor macro-F1 "
            "fijo16 en val, luego orden w2, w3)"
        ),
        "candidatos_con_ic95_positivo": candidatos,
        "val_macro_f1_espacio_fijo": {
            nombre: val_metricas[nombre]["macro_f1_espacio_fijo"] for nombre in VARIANTES_TODAS
        },
        "val_macro_f1_observado": {
            nombre: val_metricas[nombre]["macro_f1"] for nombre in VARIANTES_TODAS
        },
        "delta_val_w2_vs_w1": comparaciones["w2_vs_w1"]["delta_macro_f1_espacio_fijo"]["punto"],
        "delta_val_w3_vs_w1": comparaciones["w3_vs_w1"]["delta_macro_f1_espacio_fijo"]["punto"],
        "lecturas_test_canonico": [ganador] if ganador else [],
        "comparador_test_canonico": "w1 (preds persistidas de H1; development_test_reutilizado)",
        "referencia_h1": referencia_h1,
        "holdout_bloqueado": (
            "no evaluado en H2; reservado para la evaluacion final del proyecto"
        ),
    }
    if ganador is None:
        decision["test_dev"] = None
        decision["temporal_test"] = None
        decision["nota_negativa"] = (
            "ninguna variante supera a W1 con IC95 del delta pareado > 0 en val; "
            "sin lecturas de test canonico ni temporal"
        )
    else:
        test_ganador = variantes_resultado[ganador]["test_dev"]
        test_w1 = variantes_resultado["w1"]["test_dev"]
        decision["test_dev"] = {
            "ganador": ganador,
            "macro_f1_espacio_fijo_ganador": test_ganador["macro_f1_espacio_fijo"],
            "macro_f1_espacio_fijo_w1": test_w1["macro_f1_espacio_fijo"],
            "delta_ganador_vs_w1": float(
                test_ganador["macro_f1_espacio_fijo"] - test_w1["macro_f1_espacio_fijo"]
            ),
            "macro_f1_observado_ganador": test_ganador["macro_f1"],
            "referencia_h1_v1r_test": referencia_h1.get("test_v1r_macro_f1_espacio_fijo"),
            "development_test_reutilizado": True,
        }
        temporal_ganador = variantes_resultado[ganador]["temporal_test"]
        temporal_w1 = variantes_resultado["w1"]["temporal_test"]
        decision["temporal_test"] = {
            "ganador": ganador,
            "macro_f1_espacio_fijo_ganador": temporal_ganador["macro_f1_espacio_fijo"],
            "macro_f1_espacio_fijo_w1": temporal_w1["macro_f1_espacio_fijo"],
            "delta_ganador_vs_w1": float(
                temporal_ganador["macro_f1_espacio_fijo"]
                - temporal_w1["macro_f1_espacio_fijo"]
            ),
            "macro_f1_observado_ganador": temporal_ganador["macro_f1"],
            "development_test_reutilizado": True,
        }
    return decision


def figura_h2(
    val_metricas: dict,
    comparaciones: dict,
    ganador: str | None,
    ruta: str,
    permitir: bool,
    logger: logging.Logger,
) -> None:
    """Genera la figura de H2: macro-F1 val por variante y delta pareado vs W1.

    Args:
        val_metricas: Dict variante -> metricas de val canonica.
        comparaciones: Dict nombre_vs_w1 -> bootstrap pareado.
        ganador: Variante ganadora por val (o None).
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
    figura, ejes = plt.subplots(1, 2, figsize=(14, 6.2))
    eje = ejes[0]
    valores = [val_metricas[nombre]["macro_f1_espacio_fijo"] for nombre in VARIANTES_TODAS]
    colores = ["#c44e52" if nombre == ganador else "#4c72b0" for nombre in VARIANTES_TODAS]
    eje.bar(VARIANTES_TODAS, valores, color=colores)
    for posicion, valor in enumerate(valores):
        eje.text(posicion, valor + 0.004, f"{valor:.4f}", ha="center", fontsize=9)
    eje.set_title("Macro-F1 val canonica (espacio fijo 16)")
    eje.set_ylabel("macro-F1")
    eje.set_ylim(0, max(valores + [0.05]) * 1.25)
    eje.grid(axis="y", alpha=0.3)
    eje = ejes[1]
    deltas = []
    inferiores = []
    superiores = []
    for nombre in VARIANTES:
        comp = comparaciones[f"{nombre}_vs_w1"]["delta_macro_f1_espacio_fijo"]
        deltas.append(comp["punto"])
        inferiores.append(comp["punto"] - comp["ic95_inf"])
        superiores.append(comp["ic95_sup"] - comp["punto"])
    colores = [
        "#55a868"
        if comparaciones[f"{nombre}_vs_w1"]["delta_macro_f1_espacio_fijo"]["ic95_inf"] > 0
        else "#8c8c8c"
        for nombre in VARIANTES
    ]
    eje.bar(VARIANTES, deltas, yerr=[inferiores, superiores], capsize=6, color=colores)
    eje.axhline(0.0, color="black", linewidth=1)
    for posicion, valor in enumerate(deltas):
        eje.text(posicion, valor, f"{valor:+.4f}", ha="center", va="bottom", fontsize=9)
    eje.set_title("Delta val vs W1 (fijo16) con IC95 pareado")
    eje.set_ylabel("macro-F1 variante - W1")
    eje.grid(axis="y", alpha=0.3)
    figura.suptitle(
        "H2 objetivo suave M4 | ganador val: "
        f"{ganador if ganador else 'ninguno (negativo)'} | holdout sellado",
        fontsize=12,
    )
    figura.tight_layout()
    figura.savefig(destino, dpi=150, bbox_inches="tight")
    plt.close(figura)
    logger.info("figura H2 guardada: %s", destino)


# ===== SECCION 8: ejecucion de H2 y salida =====


def ejecutar_h2(args, logger: logging.Logger, modo: str) -> tuple[dict, float]:
    """Ejecuta H2 completo: matriz, W1, W2/W3, metricas, decision y JSON.

    Args:
        args: Argumentos CLI.
        logger: Logger activo.
        modo: todo (con verificacion md5 del protocolo) o h2.

    Returns:
        (JSON de H2, segundos totales).

    Raises:
        FileExistsError: Si el JSON o la figura ya existen sin --forzar.
        RuntimeError: Si algun archivo canonico cambia durante la corrida o
            las variantes no son consistentes entre si.
    """
    t0 = time.perf_counter()
    rutas = rutas_activas(args.smoke)
    permitir = args.smoke or args.forzar
    for clave in ("json", "figura"):
        if Path(rutas[clave]).exists() and not permitir:
            raise FileExistsError(
                f"salida de H2 ya existente (no se sobrescribe; use --forzar): {rutas[clave]}"
            )
    verificar_entradas(rutas, logger)
    with open(rutas["protocolo_json"], encoding="utf-8") as archivo:
        protocolo = json.load(archivo)
    claves_md5 = ["lemas", "etiquetas", "splits", "modelo_v1", "temporal", "holdout"]
    md5_inicial = {clave: md5_archivo(rutas[clave]) for clave in claves_md5}

    logger.info("--- H2: matriz de features (reutilizacion de la matriz H1) ---")
    t = time.perf_counter()
    info_matriz = preparar_matriz_h2(args, rutas, logger)
    tiempos = {"matriz_seg": time.perf_counter() - t}

    logger.info("--- H2: W1 (baseline v1r reutilizado) ---")
    t = time.perf_counter()
    info_w1 = preparar_w1(args, rutas, logger, info_matriz)
    tiempos["w1_seg"] = time.perf_counter() - t

    logger.info("--- H2: variantes W2/W3 en paralelo (jobs=%d) ---", args.jobs)
    t = time.perf_counter()
    lanzar_variantes(args, logger, rutas)
    tiempos["variantes_seg"] = time.perf_counter() - t

    logger.info("--- H2: metricas val, comparaciones pareadas y decision ---")
    t = time.perf_counter()
    mod18 = modulo_18()
    preds_datos = {"w1": info_w1["datos"]}
    for nombre in VARIANTES:
        preds_datos[nombre] = mod18.cargar_preds_variante(
            rutas["preds"].format(variante=nombre)
        )
    ids_ref = preds_datos["w1"]["val"]["ids"]
    y_ref = preds_datos["w1"]["val"]["y"]
    for nombre in VARIANTES:
        datos = preds_datos[nombre]["val"]
        if not np.array_equal(datos["ids"], ids_ref) or not np.array_equal(datos["y"], y_ref):
            raise RuntimeError(
                f"la variante {nombre} no comparte ids/etiquetas de val con W1: inconsistente"
            )
    val_metricas: dict = {}
    pred_val: dict = {}
    for nombre in VARIANTES_TODAS:
        datos = preds_datos[nombre]["val"]
        val_metricas[nombre] = mod18.evaluar_bloque(
            datos["y"], datos["probs"], f"val {nombre}", logger
        )
        pred_val[nombre] = np.asarray(CLASES_M4)[np.argmax(datos["probs"], axis=1)]
    comparaciones = {}
    for nombre in VARIANTES:
        comparaciones[f"{nombre}_vs_w1"] = bootstrap_pareado_fijo16(
            y_ref,
            pred_val[nombre],
            pred_val["w1"],
            CONFIG["n_bootstrap"],
            CONFIG["semilla"],
            logger,
            f"pareado {nombre} vs w1",
        )
    ganador = seleccionar_ganador(val_metricas, comparaciones)
    logger.info(
        "ganador por val: %s (candidatos con IC>0: %s)",
        ganador if ganador else "ninguno",
        [
            nombre
            for nombre in VARIANTES
            if comparaciones[f"{nombre}_vs_w1"]["delta_macro_f1_espacio_fijo"]["ic95_inf"] > 0
            and comparaciones[f"{nombre}_vs_w1"]["delta_macro_f1_espacio_fijo"]["punto"] > 0
        ],
    )
    variantes_resultado: dict = {}
    for nombre in VARIANTES_TODAS:
        variantes_resultado[nombre] = {"val": val_metricas[nombre]}
    variantes_resultado["w1"]["origen"] = info_w1["origen"]
    variantes_resultado["w1"]["ruta_preds"] = info_w1["ruta"]
    variantes_resultado["w1"]["metadatos"] = info_w1["metadatos"]
    for nombre in VARIANTES:
        variantes_resultado[nombre]["ruta_preds"] = rutas["preds"].format(variante=nombre)
        variantes_resultado[nombre]["metadatos"] = preds_datos[nombre]["metadatos"]
    if ganador is not None:
        logger.info(
            "lectura de test canonico (UNA vez, development_test_reutilizado): %s (comparador w1)",
            ganador,
        )
        for nombre in dict.fromkeys([ganador, "w1"]):
            datos = preds_datos[nombre]
            test = mod18.evaluar_bloque(
                datos["test"]["y"], datos["test"]["probs"], f"test dev {nombre}", logger
            )
            test["development_test_reutilizado"] = True
            temporal = mod18.evaluar_bloque(
                datos["temporal_test"]["y"],
                datos["temporal_test"]["probs"],
                f"temporal test {nombre}",
                logger,
            )
            temporal["development_test_reutilizado"] = True
            variantes_resultado[nombre]["test_dev"] = test
            variantes_resultado[nombre]["temporal_test"] = temporal
    else:
        logger.info("resultado negativo: sin lecturas de test canonico ni temporal")
    referencia_h1 = cargar_referencia_h1(rutas, logger)
    decision = construir_decision(
        val_metricas, comparaciones, ganador, variantes_resultado, referencia_h1
    )
    figura_h2(val_metricas, comparaciones, ganador, rutas["figura"], permitir, logger)
    tiempos["metricas_seg"] = time.perf_counter() - t

    resultado = {
        "fecha": pd.Timestamp.now().isoformat(),
        "script": "scripts/19_h2_objetivo_suave_m4.py",
        "smoke": bool(args.smoke),
        "modo": modo,
        "versiones": versiones_entorno(),
        "parametros_cli": {
            "jobs": args.jobs,
            "smoke": args.smoke,
            "forzar": args.forzar,
            "modo": modo,
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
        "conteos": info_matriz["conteos"],
        "matriz": {
            "ruta": info_matriz["ruta"],
            "origen": info_matriz["origen"],
            "md5": info_matriz["md5"],
            "motivos_descarte_h1": info_matriz["motivos_descarte_h1"],
            "motivos_descarte_reconstruida": info_matriz["motivos_descarte_reconstruida"],
            "n_features": info_matriz["n_features"],
            "params_v1": info_matriz["params_v1"],
            "vec_params": info_matriz["vec_params"],
        },
        "w1": {
            "origen": info_w1["origen"],
            "ruta": info_w1["ruta"],
            "md5": info_w1["md5"],
            "validacion": info_w1["validacion"],
            "metadatos": info_w1["metadatos"],
        },
        "variantes": variantes_resultado,
        "comparaciones_pareadas_val": comparaciones,
        "decision": decision,
        "tiempos_seg": tiempos,
        "artefactos": {
            "matriz": info_matriz["ruta"],
            "w1_preds": info_w1["ruta"],
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
    resultado["md5_salidas"] = {
        "preds_w2": md5_archivo(rutas["preds"].format(variante="w2")),
        "preds_w3": md5_archivo(rutas["preds"].format(variante="w3")),
        "figura": md5_archivo(rutas["figura"]),
    }
    escribir_json(rutas["json"], resultado, permitir=permitir, descripcion="resultado H2")
    for clave in claves_md5:
        if md5_archivo(rutas[clave]) != md5_inicial[clave]:
            raise RuntimeError(f"archivo canonico modificado durante la corrida: {clave}")
    if info_matriz["origen"] == "reutilizada_h1" and md5_archivo(info_matriz["ruta"]) != info_matriz["md5"]:
        raise RuntimeError("la matriz H1 reutilizada cambio durante la corrida")
    if info_w1["origen"] == "reutilizado_h1" and md5_archivo(info_w1["ruta"]) != info_w1["md5"]:
        raise RuntimeError("las preds v1r reutilizadas cambiaron durante la corrida")
    logger.info("integridad canonica verificada (md5 sin cambios)")
    imprimir_resumen(resultado, logger)
    return resultado, time.perf_counter() - t0


def imprimir_resumen(resultado: dict, logger: logging.Logger) -> None:
    """Imprime el resumen final en consola (maximo 10 lineas).

    Args:
        resultado: JSON de H2.
        logger: Logger activo.
    """
    decision = resultado["decision"]
    val = decision["val_macro_f1_espacio_fijo"]
    logger.info(
        "RESUMEN H2 | ganador val=%s | val fijo16: w1=%.4f w2=%.4f w3=%.4f",
        decision["ganador_por_val"] if decision["ganador_por_val"] else "ninguno",
        val["w1"],
        val["w2"],
        val["w3"],
    )
    for nombre in VARIANTES:
        comp = resultado["comparaciones_pareadas_val"][f"{nombre}_vs_w1"][
            "delta_macro_f1_espacio_fijo"
        ]
        logger.info(
            "delta val %s vs w1: %+.4f IC95=[%+.4f, %+.4f] p=%.3f",
            nombre,
            comp["punto"],
            comp["ic95_inf"],
            comp["ic95_sup"],
            comp["p_valor_bilateral"],
        )
    if decision["test_dev"] is None:
        logger.info("resultado negativo: sin lecturas de test (holdout sellado intacto)")
    else:
        test = decision["test_dev"]
        temporal = decision["temporal_test"]
        logger.info(
            "test dev (reutilizado): %s %.4f | w1 %.4f | referencia H1 v1r %.4f",
            test["ganador"],
            test["macro_f1_espacio_fijo_ganador"],
            test["macro_f1_espacio_fijo_w1"],
            test["referencia_h1_v1r_test"] if test["referencia_h1_v1r_test"] else float("nan"),
        )
        logger.info(
            "test temporal: %s %.4f | w1 %.4f",
            temporal["ganador"],
            temporal["macro_f1_espacio_fijo_ganador"],
            temporal["macro_f1_espacio_fijo_w1"],
        )
    logger.info(
        "matriz: %s (%s) | W1: %s",
        resultado["matriz"]["ruta"],
        resultado["matriz"]["origen"],
        resultado["w1"]["origen"],
    )
    logger.info(
        "salidas: %s | %s",
        resultado["artefactos"]["json"],
        resultado["artefactos"]["figura"],
    )
    logger.info("tiempo total H2: %.1f s", resultado["tiempos_seg"]["total_seg"])


# ===== SECCION 9: CLI y main =====


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Define y parsea los argumentos de linea de comandos.

    Args:
        argv: Lista de argumentos (None usa sys.argv).

    Returns:
        Namespace con todo, h2, jobs, solo_variante, forzar y smoke.
    """
    parser = argparse.ArgumentParser(
        description=(
            "H2: objetivo suave por ambiguedad multi-tema (top3_json) sobre el "
            "protocolo de evaluacion vigente del experimento 18."
        )
    )
    parser.add_argument(
        "--todo",
        action="store_true",
        help="H2 completo con verificacion md5 del protocolo (default)",
    )
    parser.add_argument(
        "--h2",
        action="store_true",
        help="solo H2 (omite la verificacion md5 del protocolo; exige que los archivos existan)",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=CONFIG["jobs"],
        help="maximo de variantes en paralelo (default 2)",
    )
    parser.add_argument(
        "--solo-variante",
        choices=list(VARIANTES),
        default=None,
        dest="solo_variante",
        help="interno: ejecuta solo esa variante (subproceso)",
    )
    parser.add_argument("--forzar", action="store_true", help="rehace preds/JSON/figura existentes")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="subconjunto ~500 docs; salidas solo en /tmp/opencode/h2_smoke",
    )
    return parser.parse_args(argv)


def resolver_modo(args: argparse.Namespace) -> str:
    """Determina el modo de ejecucion a partir de los flags.

    Args:
        args: Argumentos CLI.

    Returns:
        "h2" si solo se pidio --h2; "todo" en cualquier otro caso.
    """
    if args.h2 and not args.todo:
        return "h2"
    return "todo"


def main(argv: list[str] | None = None) -> None:
    """Orquesta la ejecucion de H2 segun los flags.

    Args:
        argv: Argumentos CLI (None usa sys.argv).

    Raises:
        ValueError: Si --jobs es menor que 1.
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
        "=== INICIO H2 objetivo suave M4 | modo=%s smoke=%s jobs=%d forzar=%s ===",
        modo,
        args.smoke,
        args.jobs,
        args.forzar,
    )
    if modo == "todo":
        verificar_protocolo(rutas, logger)
    else:
        logger.info(
            "modo --h2: se omite la verificacion md5 del protocolo (solo se exige que exista)"
        )
    _, total_seg = ejecutar_h2(args, logger, modo)
    logger.info("=== FIN H2 objetivo suave M4 en %.1f s ===", total_seg)


if __name__ == "__main__":
    try:
        main()
    except Exception:  # noqa: BLE001 - protocolo: traza completa y salida 1 ante cualquier fallo
        traceback.print_exc()
        sys.exit(1)
