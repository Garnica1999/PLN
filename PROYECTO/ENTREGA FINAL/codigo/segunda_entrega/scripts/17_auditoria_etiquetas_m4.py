"""17_auditoria_etiquetas_m4.py
Fecha: 2026-10-03
Descripcion: Auditoria de ruido/calidad de las etiquetas SILVER de M4
(area tematica derivada del bloque mayoritario de citas NER de la
Constitucion) y establecimiento de un baseline honesto. NO es un
experimento de mejora: es diagnostico.

Componentes:
    1. Modo val (rapido, sin reentrenar): reproduce las metricas del
       joblib v1 (LogReg enmascarado, macro-F1 test 0.287 observado /
       0.269 espacio fijo), audita las etiquetas con cleanlab sobre val
       y test, y calcula IC bootstrap al 95 % del macro-F1 de espacio
       fijo. OJO: el joblib v1 se reentreno en train+val en su corrida
       original (ver scripts/06_baseline_clasico.py), por lo que val es
       in-sample (memorizado) y solo test es out-of-sample para ese
       artefacto; ambas auditorias se reportan con esa advertencia.
    2. Modo oof (5-fold sobre train, 16.557 docs, reanudable): la
       matriz TF-IDF se ajusta UNA vez sobre todo train (transform no
       supervisado, premisa documentada) y cada fold corre como
       subproceso independiente con limites de hilos. Concatena las
       probabilidades out-of-sample y audita el ruido del train completo
       con cleanlab. Incluye baseline de clase mayoritaria en val/test.

Artefactos NUEVOS (nunca sobrescribe archivos existentes fuera de
--smoke): resultados/m4_ruido_etiquetas.json, resultados/m4_oof_probs.npz,
resultados/m4_oof_matrix.npz, resultados/m4_oof_fold{k}.npz,
figuras/fig_ruido_etiquetas_m4.png y
experimentos/17_auditoria_etiquetas_m4.log.

Estado del arte aplicado:
    - Confident learning para auditoria de ruido de etiquetas
      (Northcutt et al., 2021, "Pervasive Label Errors in Test Sets
      Destabilize Machine Learning Benchmarks"; cleanlab 2.9.0).
    - Probabilidades out-of-sample via K-fold OOF (protocolo estandar
      de auditoria; evita la sobreconfianza in-sample).
    - IC bootstrap percentil al 95 % (Efron y Tibshirani, 1993) con
      1000 remuestreos de casos y semilla 42.
    - Hiperparametros y vectorizacion EXACTOS del baseline v1 (TF-IDF
      de lemas enmascarado, min_df=3, max_features=100000, ngram
      (1, 2), sublinear_tf=True; C=53.88, lbfgs, max_iter=600).

Uso:
    /home/carlos/venv/bin/python scripts/17_auditoria_etiquetas_m4.py --modo val
    /home/carlos/venv/bin/python scripts/17_auditoria_etiquetas_m4.py --modo todo --jobs 4
    /home/carlos/venv/bin/python scripts/17_auditoria_etiquetas_m4.py --modo oof --smoke
"""

import argparse
import gc
import hashlib
import importlib.util
import json
import logging
import os
import platform
import random
import subprocess
import sys
import threading
import time
import traceback
from collections import Counter
from pathlib import Path

import cleanlab
import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import scipy
import sklearn
from cleanlab.count import compute_confident_joint
from cleanlab.filter import find_label_issues
from scipy.sparse import csr_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold

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

CONFIG = {
    "semilla": 42,
    "n_folds": 5,
    "n_folds_smoke": 2,
    "jobs": 4,
    "hilos_por_fold": 6,
    "umbral_clases_pequenas": 5,
    "limite_smoke": 400,
    "n_bootstrap": 1000,
    "top_ejemplos": 20,
    "tolerancia_reproduccion": 1e-6,
    "dir_smoke": "/tmp/opencode",
    "rutas_datos": {
        "lemas": "data/lemas_por_doc.parquet",
        "etiquetas": "data/etiquetas_areas.parquet",
        "splits": "data/splits_m4.parquet",
    },
    "rutas_modelo": {
        "modelo": "modelos/m4_baseline/logreg_enmascarado.joblib",
        "vectorizer": "modelos/m4_baseline/tfidf_enmascarado.joblib",
        "referencia": "resultados/m4_baseline.json",
    },
    "rutas_salida": {
        "json": "resultados/m4_ruido_etiquetas.json",
        "oof_probs": "resultados/m4_oof_probs.npz",
        "oof_matrix": "resultados/m4_oof_matrix.npz",
        "fold": "resultados/m4_oof_fold{k}.npz",
        "figura": "figuras/fig_ruido_etiquetas_m4.png",
        "log": "experimentos/17_auditoria_etiquetas_m4.log",
    },
    "rutas_salida_smoke": {
        "json": "/tmp/opencode/m4_ruido_etiquetas_smoke.json",
        "oof_probs": "/tmp/opencode/m4_oof_probs_smoke.npz",
        "oof_matrix": "/tmp/opencode/m4_oof_matrix_smoke.npz",
        "fold": "/tmp/opencode/m4_oof_fold{k}_smoke.npz",
        "figura": "/tmp/opencode/fig_ruido_etiquetas_m4_smoke.png",
        "log": "/tmp/opencode/17_auditoria_etiquetas_m4_smoke.log",
    },
}

# Premisas de diseno que se congelan en el JSON de salida (trazabilidad).
PREMISAS = [
    (
        "El joblib v1 (modelos/m4_baseline/logreg_enmascarado.joblib) se reentreno en "
        "train+val en su corrida original (06_baseline_clasico.py, entrenar_logreg_optuna): "
        "sus predicciones en val son in-sample (memorizadas) y cleanlab sobre val NO estima "
        "ruido real. La estimacion honesta de ruido sale del modo oof (out-of-sample sobre "
        "train); en el modo val solo cleanlab sobre test es out-of-sample para ese artefacto."
    ),
    (
        "El vectorizer del modo oof se ajusta UNA vez sobre todo el train canonico por "
        "eficiencia (reutiliza la configuracion exacta del escenario enmascarado v1). Es un "
        "transform no supervisado (no usa etiquetas), por lo que no hay leakage de etiquetas; "
        "el fold de validacion de cada split queda dentro del ajuste de vocabulario/IDF "
        "(premisa transductiva documentada, no hay fuga de la variable objetivo)."
    ),
    (
        "Las etiquetas silver provienen del bloque mayoritario de citas NER de la "
        "Constitucion (data/etiquetas_areas.parquet, columna area_primaria, via cita CP): la "
        "auditoria mide el ruido respecto a esa senal automatica, no contra ground truth humano."
    ),
    (
        "Espacio fijo de 16 clases en orden alfabetico (identico a model.classes_ de v1); en "
        "OOF, una clase ausente del train de un fold produce columna de probabilidad 0 "
        "(nunca predicha) y queda registrada en el JSON."
    ),
    (
        "Splits canonicos intocados (data/splits_m4.parquet): train 16.557 / val 3.549 / "
        "test 3.548. val y test son de solo lectura y no se usan para tomar decisiones de "
        "modelado (test se reporta una sola vez como diagnostico congelado)."
    ),
    (
        "Los folds OOF respetan las etiquetas: clases con n < 5 reparten sus ejemplos entre "
        "folds distintos por round-robin determinista (ids ordenados numericamente) y el "
        "resto usa StratifiedKFold(semilla=42). Las etiquetas NUNCA se fusionan."
    ),
    (
        "IC bootstrap percentil al 95 % con 1000 remuestreos de casos y semilla 42 sobre el "
        "macro-F1 de espacio fijo (16 clases, zero_division=0)."
    ),
    (
        "cleanlab 2.9.0: find_label_issues con defaults (filter_by=prune_by_noise_rate, "
        "frac_noise=1.0) y confident joint calibrado; el mapeo de columnas usa el orden "
        "alfabetico de clases (np.unique), consistente con CLASES_M4."
    ),
]

LOGGER = logging.getLogger("auditoria_etiquetas_m4_17")

# Cache del modulo 06 (su nombre empieza por digito: importlib por ruta).
_MODULO_BASELINE = None


# ===== SECCION 2: utilidades (logger, md5, versiones, modulo baseline) =====


def configurar_logger(ruta_log: str, solo_consola: bool = False) -> logging.Logger:
    """Configura el logger de la auditoria (archivo en modo append + consola).

    Args:
        ruta_log: Ruta del archivo de log (se crea el directorio padre).
        solo_consola: True para subprocesos de fold (la salida se canaliza
            al log del orquestador con prefijo, evitando escrituras
            concurrentes al mismo archivo).

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


def cargar_modulo_baseline():
    """Importa scripts/06_baseline_clasico.py por ruta (cacheado).

    El nombre del modulo empieza por digito, por lo que no puede importarse
    con la sintaxis normal; se usa importlib.util.spec_from_file_location
    para reutilizar sus funciones sin duplicarlas.

    Returns:
        Modulo 06_baseline_clasico ya ejecutado (CONFIG, funciones).
    """
    global _MODULO_BASELINE
    if _MODULO_BASELINE is None:
        ruta = RAIZ / "scripts" / "06_baseline_clasico.py"
        spec = importlib.util.spec_from_file_location("baseline_m4_06", ruta)
        if spec is None or spec.loader is None:
            raise ImportError(f"no se pudo cargar {ruta}")
        modulo = importlib.util.module_from_spec(spec)
        sys.modules["baseline_m4_06"] = modulo
        spec.loader.exec_module(modulo)
        _MODULO_BASELINE = modulo
    return _MODULO_BASELINE


def md5_archivo(ruta: str) -> str | None:
    """Calcula el md5 de un archivo por bloques (integridad, no seguridad).

    Args:
        ruta: Ruta del archivo.

    Returns:
        Hexdigest md5, o None si el archivo no existe.
    """
    archivo = Path(ruta)
    if not archivo.exists():
        return None
    digest = hashlib.md5()
    with open(archivo, "rb") as f:
        for bloque in iter(lambda: f.read(8 * 1024 * 1024), b""):
            digest.update(bloque)
    return digest.hexdigest()


def versiones_entorno() -> dict:
    """Recolecta versiones de interprete y librerias clave.

    Returns:
        Dict con python, plataforma y versiones de sklearn/scipy/numpy/
        pandas/joblib/cleanlab/matplotlib.
    """
    return {
        "python": platform.python_version(),
        "plataforma": platform.platform(),
        "sklearn": sklearn.__version__,
        "scipy": scipy.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "joblib": joblib.__version__,
        "cleanlab": cleanlab.__version__,
        "matplotlib": matplotlib.__version__,
    }


def rutas_activas(smoke: bool) -> dict:
    """Construye el mapa de rutas de entrada/salida segun el modo smoke.

    Args:
        smoke: True para usar salidas exclusivamente en /tmp/opencode.

    Returns:
        Dict combinado de rutas de datos, modelo y salidas.
    """
    rutas = {}
    rutas.update(CONFIG["rutas_datos"])
    rutas.update(CONFIG["rutas_modelo"])
    rutas.update(CONFIG["rutas_salida_smoke"] if smoke else CONFIG["rutas_salida"])
    return rutas


def verificar_salidas_nuevas(modo: str, rutas: dict, smoke: bool, forzar: bool) -> None:
    """Aborta si una salida final del modo ya existe (no se sobrescribe nada).

    Args:
        modo: val, oof o todo.
        rutas: Dict de rutas activas.
        smoke: True si es corrida smoke (salidas de scratch, se permite reescribir).
        forzar: True si se pidio rehacer la corrida (permite reescribir salidas).

    Raises:
        FileExistsError: Si existe alguna salida final del modo pedido.
    """
    if smoke or forzar:
        return
    claves = ["json", "figura"]
    if modo in ("oof", "todo"):
        claves.append("oof_probs")
    existentes = [rutas[clave] for clave in claves if Path(rutas[clave]).exists()]
    if existentes:
        raise FileExistsError(
            "salidas ya existentes (esta auditoria nunca las sobrescribe; use --forzar "
            "para rehacer): " + ", ".join(existentes)
        )


# ===== SECCION 3: datos, splits y folds =====


def cargar_datos_completo(logger: logging.Logger) -> pd.DataFrame:
    """Carga lemas + etiquetas + splits canonicos reutilizando el modulo 06.

    Args:
        logger: Logger activo.

    Returns:
        DataFrame con lemas_norm, lemas_mask, area_primaria y particion.

    Raises:
        RuntimeError: Si los splits no cubren todos los documentos o si el
            espacio de clases de los datos no coincide con CLASES_M4.
    """
    base = cargar_modulo_baseline()
    t0 = time.perf_counter()
    df = base.cargar_datos()
    splits = pq.read_table(CONFIG["rutas_datos"]["splits"]).to_pandas()
    df = df.merge(splits, on="id", how="inner")
    logger.info("datos completos: %d filas (%d s)", len(df), time.perf_counter() - t0)
    if len(df) != len(splits):
        raise RuntimeError(
            f"los splits cubren {len(splits)} ids pero solo {len(df)} se unieron: "
            "datos y splits estan desalineados"
        )
    clases_datos = sorted(df["area_primaria"].unique().tolist())
    if clases_datos != CLASES_M4:
        raise RuntimeError(
            "el espacio de clases de los datos no coincide con el espacio fijo v1: "
            f"faltan={sorted(set(CLASES_M4) - set(clases_datos))} "
            f"sobran={sorted(set(clases_datos) - set(CLASES_M4))}"
        )
    logger.info("particiones: %s", df["particion"].value_counts().to_dict())
    return df


def cargar_datos_smoke(limite: int, semilla: int, logger: logging.Logger) -> pd.DataFrame:
    """Carga un subconjunto estratificado por particion para el smoke.

    Reproduce el flujo de cargar_datos (mismo merge, mismo filtro de
    sin_area_definida y mismas funciones de normalizacion/enmascarado)
    pero preprocesa SOLO el subconjunto para que el smoke sea rapido. No
    aplica la agrupacion de clases raras (<3 docs) porque el corpus
    actual no tiene ninguna (se verifica y se advierte si cambiara).

    Args:
        limite: Maximo de documentos por particion.
        semilla: Semilla del submuestreo estratificado.
        logger: Logger activo.

    Returns:
        DataFrame submuestreado con las particiones canonicas originales.
    """
    base = cargar_modulo_baseline()
    etiquetas = pq.read_table(CONFIG["rutas_datos"]["etiquetas"]).to_pandas()
    conteo = etiquetas.loc[
        etiquetas["area_primaria"] != "sin_area_definida", "area_primaria"
    ].value_counts()
    raras = conteo[conteo < 3]
    if len(raras):
        logger.warning(
            "clases con <3 docs en el corpus completo (%s): el smoke podria diferir "
            "del flujo completo (cargar_datos las agrupa en otras_areas)",
            raras.to_dict(),
        )
    lemas = pq.read_table(CONFIG["rutas_datos"]["lemas"]).to_pandas()
    splits = pq.read_table(CONFIG["rutas_datos"]["splits"]).to_pandas()
    df = lemas.merge(etiquetas[["id", "area_primaria"]], on="id")
    df = df[df["area_primaria"] != "sin_area_definida"].reset_index(drop=True)
    df = df.merge(splits, on="id", how="inner")
    partes = []
    for particion in ("train", "val", "test"):
        sub = df[df["particion"] == particion].reset_index(drop=True)
        partes.append(base.submuestrear_estratificado(sub, limite, semilla))
    sub = pd.concat(partes, ignore_index=True)
    sub["lemas_norm"] = sub["lemas_texto"].map(base.normalizar_lemas)
    sub["lemas_mask"] = sub["lemas_norm"].map(base.enmascarar_citas_lemas)
    logger.info("smoke: %d docs (limite %d por particion)", len(sub), limite)
    return sub


def asignar_folds(
    y: np.ndarray,
    ids: np.ndarray,
    n_folds: int,
    semilla: int,
    umbral_clases_pequenas: int,
    logger: logging.Logger,
) -> np.ndarray:
    """Asigna cada documento de train a un fold de forma determinista.

    Clases con n < umbral se reparten por round-robin sobre los ids
    ordenados numericamente (garantiza presencia en folds distintos); el
    resto usa StratifiedKFold(seed). Las etiquetas no se fusionan.

    Args:
        y: Etiquetas de train.
        ids: Identificadores de train (strings numericos).
        n_folds: Numero de folds.
        semilla: Semilla de StratifiedKFold.
        umbral_clases_pequenas: Umbral de "clase pequena".
        logger: Logger activo.

    Returns:
        Array int con el fold (0..n_folds-1) de cada documento.
    """
    y = np.asarray([str(v) for v in y])
    ids = np.asarray([str(v) for v in ids])
    fold = np.full(len(y), -1, dtype=np.int64)
    conteo = Counter(y.tolist())
    pequenas = sorted([clase for clase, n in conteo.items() if n < umbral_clases_pequenas])
    for clase in pequenas:
        idx = np.where(y == clase)[0]
        orden = idx[np.argsort([int(v) for v in ids[idx]], kind="stable")]
        for posicion, i in enumerate(orden):
            fold[i] = posicion % n_folds
    resto = np.where(fold == -1)[0]
    if len(resto):
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=semilla)
        for f, (_, va) in enumerate(skf.split(resto, y[resto])):
            fold[resto[va]] = f
    if (fold < 0).any():
        raise RuntimeError("quedaron documentos sin fold asignado")
    resumen = {k: int((fold == k).sum()) for k in range(n_folds)}
    logger.info("folds: n=%d k=%d tamanos=%s clases_pequenas(round-robin)=%s", len(y), n_folds, resumen, pequenas)
    return fold


def guardar_matriz_oof(
    ruta: str,
    X,
    ids: np.ndarray,
    y: np.ndarray,
    folds: np.ndarray,
    params: dict,
    vec_params: dict,
    logger: logging.Logger,
) -> None:
    """Persiste la matriz TF-IDF de train + metadatos en un .npz.

    Args:
        ruta: Ruta de salida .npz.
        X: Matriz CSR TF-IDF de train completo.
        ids: Ids de train (mismo orden que X).
        y: Etiquetas de train.
        folds: Fold asignado a cada documento.
        params: Hiperparametros v1 del LogReg.
        vec_params: Parametros del vectorizer ajustado.
        logger: Logger activo.
    """
    destino = Path(ruta)
    destino.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        destino,
        data=X.data,
        indices=X.indices,
        indptr=X.indptr,
        shape=np.asarray(X.shape, dtype=np.int64),
        ids=np.asarray([str(v) for v in ids]),
        labels=np.asarray([str(v) for v in y]),
        folds=folds.astype(np.int64),
        clases=np.asarray(CLASES_M4),
        params_v1_json=np.asarray(json.dumps(params)),
        vec_params_json=np.asarray(json.dumps(vec_params)),
    )
    logger.info(
        "matriz OOF guardada: %s (%d docs x %d features, %d nnz, %.1f MB)",
        destino,
        X.shape[0],
        X.shape[1],
        X.nnz,
        destino.stat().st_size / 1e6,
    )


def cargar_matriz_oof(ruta: str) -> dict:
    """Carga el .npz de la matriz OOF y devuelve sus arrays.

    Args:
        ruta: Ruta del .npz.

    Returns:
        Dict clave -> array (sin pickle).
    """
    with np.load(ruta, allow_pickle=False) as datos:
        return {clave: datos[clave] for clave in datos.files}


# ===== SECCION 4: modo val (joblib v1, cleanlab, bootstrap) =====


def metricas_por_clase(y_true: np.ndarray, y_pred: np.ndarray, clases: list[str]) -> dict:
    """Calcula precision/recall/F1/soporte por clase sobre el espacio fijo.

    Args:
        y_true: Etiquetas reales.
        y_pred: Etiquetas predichas.
        clases: Clases del espacio fijo.

    Returns:
        Dict clase -> {precision, recall, f1, soporte}.
    """
    precision, recall, f1, soporte = precision_recall_fscore_support(
        y_true, y_pred, labels=clases, zero_division=0
    )
    return {
        clase: {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "soporte": int(soporte[i]),
        }
        for i, clase in enumerate(clases)
    }


def evaluar_predicciones(y_true, y_pred, clases: list[str]) -> dict:
    """Metricas completas (v1 comparables + desglose por clase).

    Args:
        y_true: Etiquetas reales.
        y_pred: Etiquetas predichas.
        clases: Clases del espacio fijo.

    Returns:
        Dict con macro_f1 (observado), macro_f1_espacio_fijo, accuracy,
        n y por_clase.
    """
    base = cargar_modulo_baseline()
    metricas = base.evaluar_single_label(y_true, y_pred, clases)
    metricas["n"] = len(y_true)
    metricas["por_clase"] = metricas_por_clase(y_true, y_pred, clases)
    return metricas


def auditar_cleanlab(
    y,
    probs: np.ndarray,
    ids,
    clases: list[str],
    top_n: int,
    logger: logging.Logger,
    nombre: str,
) -> dict:
    """Audita etiquetas con cleanlab (find_label_issues + confident joint).

    Args:
        y: Etiquetas observadas.
        probs: Probabilidades predichas (n, 16) alineadas a clases.
        ids: Identificadores de los documentos.
        clases: Clases del espacio fijo.
        top_n: Numero de ejemplos flagged a listar (menor self-confidence).
        logger: Logger activo.
        nombre: Etiqueta de la auditoria para el log.

    Returns:
        Dict con n_issues, tasa_issues, desglose por clase,
        self_confidence_media, confident_joint (resumida y completa) y
        top_ejemplos flagged.
    """
    t0 = time.perf_counter()
    y = np.asarray([str(v) for v in y])
    ids = np.asarray([str(v) for v in ids])
    probs = np.asarray(probs, dtype=np.float64)
    presentes = sorted(set(y.tolist()))
    faltan: list[str] = []
    if presentes != list(clases):
        faltan = sorted(set(clases) - set(presentes))
        logger.warning("%s: clases sin ejemplos en y (%s); cleanlab se restringe a las presentes", nombre, faltan)
        probs = probs[:, [clases.index(c) for c in presentes]]
        clases_uso = presentes
        # cleanlab exige filas de probabilidades que sumen 1: al descartar
        # columnas de clases ausentes hay que renormalizar cada fila.
        sumas = probs.sum(axis=1, keepdims=True)
        probs = probs / np.where(sumas > 0, sumas, 1.0)
    else:
        clases_uso = list(clases)
    indice = {clase: j for j, clase in enumerate(clases_uso)}
    # cleanlab exige etiquetas enteras 0..K-1 alineadas con las columnas.
    y_int = np.asarray([indice[clase] for clase in y], dtype=np.int64)
    issues = find_label_issues(labels=y_int, pred_probs=probs)
    self_conf = probs[np.arange(len(y)), y_int]
    joint = compute_confident_joint(labels=y_int, pred_probs=probs, calibrate=True)
    por_clase = {}
    for clase in clases_uso:
        mascara = y == clase
        n_clase = int(mascara.sum())
        n_issues = int(issues[mascara].sum())
        por_clase[clase] = {
            "soporte": n_clase,
            "issues": n_issues,
            "tasa": (n_issues / n_clase) if n_clase else None,
            "self_confidence_media": float(self_conf[mascara].mean()) if n_clase else None,
        }
    top_ejemplos = []
    idx_flag = np.where(issues)[0]
    orden = idx_flag[np.argsort(self_conf[idx_flag], kind="stable")]
    for i in orden[:top_n]:
        j_y = indice[y[i]]
        fila = probs[i].copy()
        fila[j_y] = -1.0
        j_segunda = int(np.argmax(fila))
        top_ejemplos.append(
            {
                "id": ids[i],
                "etiqueta": y[i],
                "prob_etiqueta": float(probs[i, j_y]),
                "segunda_clase": clases_uso[j_segunda],
                "prob_segunda": float(probs[i, j_segunda]),
            }
        )
    diagonal = float(np.trace(joint))
    total = float(joint.sum())
    resumen_joint = {
        "suma_diagonal": diagonal,
        "suma_fuera_diagonal": total - diagonal,
        "fraccion_diagonal": (diagonal / total) if total else None,
    }
    logger.info(
        "%s: %d issues de %d docs (%.2f %%), self-confidence media %.3f [%.1f s]",
        nombre,
        int(issues.sum()),
        len(y),
        100.0 * float(issues.mean()),
        float(self_conf.mean()),
        time.perf_counter() - t0,
    )
    return {
        "n_docs": len(y),
        "n_issues": int(issues.sum()),
        "tasa_issues": float(issues.mean()),
        "clases_auditadas": list(clases_uso),
        "clases_excluidas": faltan,
        "tasa_issues_por_clase": por_clase,
        "self_confidence_media": float(self_conf.mean()),
        "confident_joint_resumen": resumen_joint,
        "confident_joint": joint.tolist(),
        "top_ejemplos": top_ejemplos,
        "tiempo_seg": time.perf_counter() - t0,
    }


def bootstrap_macro_f1(
    y_true,
    y_pred,
    clases: list[str],
    n_boot: int,
    semilla: int,
    logger: logging.Logger,
    nombre: str,
) -> dict:
    """IC bootstrap percentil del macro-F1 de espacio fijo (remuestreo de casos).

    Args:
        y_true: Etiquetas reales.
        y_pred: Etiquetas predichas.
        clases: Clases del espacio fijo.
        n_boot: Numero de remuestreos.
        semilla: Semilla del generador.
        logger: Logger activo.
        nombre: Etiqueta de la serie para el log.

    Returns:
        Dict con punto, media, desviacion e IC95 (percentiles 2.5/97.5).
    """
    base = cargar_modulo_baseline()
    t0 = time.perf_counter()
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)
    rng = np.random.default_rng(semilla)
    valores = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        valores[b] = base.evaluar_single_label(
            y_true[idx], y_pred[idx], clases
        )["macro_f1_espacio_fijo"]
    punto = base.evaluar_single_label(y_true, y_pred, clases)["macro_f1_espacio_fijo"]
    resultado = {
        "macro_f1_espacio_fijo": float(punto),
        "media_bootstrap": float(valores.mean()),
        "desviacion_bootstrap": float(valores.std(ddof=1)),
        "ic95_inf": float(np.percentile(valores, 2.5)),
        "ic95_sup": float(np.percentile(valores, 97.5)),
        "n_boot": int(n_boot),
        "semilla": int(semilla),
    }
    logger.info(
        "%s: macro-F1 fijo=%.4f IC95=[%.4f, %.4f] (%d remuestreos, %.1f s)",
        nombre,
        resultado["macro_f1_espacio_fijo"],
        resultado["ic95_inf"],
        resultado["ic95_sup"],
        n_boot,
        time.perf_counter() - t0,
    )
    return resultado


def ejecutar_modo_val(df: pd.DataFrame, args, logger: logging.Logger, rutas: dict) -> tuple[dict, float]:
    """Ejecuta la auditoria rapida sobre val/test con el joblib v1.

    Args:
        df: DataFrame con particiones.
        args: Argumentos CLI.
        logger: Logger activo.
        rutas: Dict de rutas activas.

    Returns:
        (resultado del modo, segundos totales).
    """
    t0 = time.perf_counter()
    base = cargar_modulo_baseline()
    tiempos: dict = {}
    t = time.perf_counter()
    modelo = joblib.load(rutas["modelo"])
    vectorizer = joblib.load(rutas["vectorizer"])
    params = base.cargar_hiperparametros_v1(rutas["modelo"])
    tiempos["carga_modelo_seg"] = time.perf_counter() - t
    if list(modelo.classes_) != CLASES_M4:
        logger.warning("clases del joblib v1 distintas del espacio fijo: %s", list(modelo.classes_))
    logger.info("v1 cargado: params=%s", params)

    va = df[df["particion"] == "val"]
    te = df[df["particion"] == "test"]
    t = time.perf_counter()
    X_val = vectorizer.transform(va["lemas_mask"])
    X_test = vectorizer.transform(te["lemas_mask"])
    tiempos["transform_seg"] = time.perf_counter() - t
    t = time.perf_counter()
    proba_val = modelo.predict_proba(X_val)
    proba_test = modelo.predict_proba(X_test)
    pred_val = modelo.predict(X_val)
    pred_test = modelo.predict(X_test)
    tiempos["prediccion_seg"] = time.perf_counter() - t
    y_val = va["area_primaria"].values
    y_test = te["area_primaria"].values
    ids_val = va["id"].values
    ids_test = te["id"].values
    logger.info("val n=%d | test n=%d", len(y_val), len(y_test))

    metricas = {
        "val": evaluar_predicciones(y_val, pred_val, CLASES_M4),
        "test": evaluar_predicciones(y_test, pred_test, CLASES_M4),
    }
    logger.info(
        "metricas v1: val macro-F1=%.4f (fijo %.4f, acc %.4f) | test macro-F1=%.4f (fijo %.4f, acc %.4f)",
        metricas["val"]["macro_f1"],
        metricas["val"]["macro_f1_espacio_fijo"],
        metricas["val"]["accuracy"],
        metricas["test"]["macro_f1"],
        metricas["test"]["macro_f1_espacio_fijo"],
        metricas["test"]["accuracy"],
    )

    with open(rutas["referencia"], encoding="utf-8") as archivo:
        referencia = json.load(archivo)
    ref_logreg = referencia.get("enmascarado", {}).get("logreg", {})
    comparacion = {
        "ruta": rutas["referencia"],
        "test_macro_f1_referencia": ref_logreg.get("macro_f1"),
        "test_macro_f1_reproducido": metricas["test"]["macro_f1"],
        "test_accuracy_referencia": ref_logreg.get("accuracy"),
        "test_accuracy_reproducido": metricas["test"]["accuracy"],
    }
    if ref_logreg.get("macro_f1") is not None:
        comparacion["delta_test_macro_f1"] = abs(
            metricas["test"]["macro_f1"] - float(ref_logreg["macro_f1"])
        )
        comparacion["delta_test_accuracy"] = abs(
            metricas["test"]["accuracy"] - float(ref_logreg["accuracy"])
        )
        comparacion["dentro_tolerancia_1e-6"] = (
            comparacion["delta_test_macro_f1"] <= CONFIG["tolerancia_reproduccion"]
            and comparacion["delta_test_accuracy"] <= CONFIG["tolerancia_reproduccion"]
        )
        if not comparacion["dentro_tolerancia_1e-6"]:
            logger.warning(
                "reproduccion v1 fuera de tolerancia: delta macro-F1=%.2e delta acc=%.2e",
                comparacion["delta_test_macro_f1"],
                comparacion["delta_test_accuracy"],
            )
    if args.smoke:
        comparacion["nota_smoke"] = (
            "subconjunto de 400 docs por particion: la comparacion contra "
            "m4_baseline.json no es interpretable en smoke"
        )

    t = time.perf_counter()
    audit_val = auditar_cleanlab(
        y_val, proba_val, ids_val, CLASES_M4, CONFIG["top_ejemplos"], logger,
        "cleanlab val (in-sample v1)",
    )
    audit_test = auditar_cleanlab(
        y_test, proba_test, ids_test, CLASES_M4, CONFIG["top_ejemplos"], logger,
        "cleanlab test (out-of-sample v1)",
    )
    tiempos["cleanlab_seg"] = time.perf_counter() - t
    t = time.perf_counter()
    boot_val = bootstrap_macro_f1(y_val, pred_val, CLASES_M4, CONFIG["n_bootstrap"], CONFIG["semilla"], logger, "bootstrap val")
    boot_test = bootstrap_macro_f1(y_test, pred_test, CLASES_M4, CONFIG["n_bootstrap"], CONFIG["semilla"], logger, "bootstrap test")
    tiempos["bootstrap_seg"] = time.perf_counter() - t

    resultado = {
        "tiempos_seg": tiempos,
        "n_val": len(y_val),
        "n_test": len(y_test),
        "parametros_v1": params,
        "metricas_v1": metricas,
        "comparacion_m4_baseline": comparacion,
        "advertencia_val_in_sample": (
            "El joblib v1 se reentreno en train+val (06_baseline_clasico.py): sus "
            "predicciones en val estan memorizadas (macro-F1=1.0) y cleanlab sobre val "
            "NO mide ruido real. La medicion valida en este modo es "
            "cleanlab_test_out_of_sample (v1 nunca vio test); la medicion valida sobre "
            "train sale del modo oof."
        ),
        "cleanlab_val_in_sample": audit_val,
        "cleanlab_test_out_of_sample": audit_test,
        "bootstrap_macro_f1_espacio_fijo": {"val": boot_val, "test": boot_test},
    }
    return resultado, time.perf_counter() - t0


# ===== SECCION 5: modo oof (5 folds, cleanlab sobre train, mayoria) =====


def expandir_probs(modelo, X, clases: list[str]) -> np.ndarray:
    """Expande predict_proba al espacio fijo de 16 clases (ausentes -> 0).

    Args:
        modelo: LogReg ajustado.
        X: Matriz de features a predecir.
        clases: Clases del espacio fijo en orden canonico.

    Returns:
        Matriz (n, 16) con las probabilidades alineadas al espacio fijo.

    Raises:
        RuntimeError: Si el modelo contiene una clase fuera del espacio fijo.
    """
    probs = modelo.predict_proba(X)
    indice = {str(clase): j for j, clase in enumerate(clases)}
    salida = np.zeros((X.shape[0], len(clases)), dtype=np.float64)
    for j, clase in enumerate(modelo.classes_):
        clave = str(clase)
        if clave not in indice:
            raise RuntimeError(f"clase del modelo fuera del espacio fijo: {clave}")
        salida[:, indice[clave]] = probs[:, j]
    return salida


def ejecutar_fold(k: int, args, logger: logging.Logger, rutas: dict) -> None:
    """Entrena y predice un fold OOF (cuerpo del subproceso --solo-fold k).

    Args:
        k: Indice del fold.
        args: Argumentos CLI.
        logger: Logger activo.
        rutas: Dict de rutas activas.

    Raises:
        FileNotFoundError: Si falta la matriz OOF.
        ValueError: Si el fold no tiene documentos de validacion.
    """
    ruta_salida = Path(rutas["fold"].format(k=k))
    if ruta_salida.exists() and not args.forzar:
        logger.info("fold %d: salida existente, se omite (%s)", k, ruta_salida)
        return
    t0 = time.perf_counter()
    datos = cargar_matriz_oof(rutas["oof_matrix"])
    X = csr_matrix(
        (datos["data"], datos["indices"], datos["indptr"]),
        shape=tuple(int(v) for v in datos["shape"]),
    )
    folds = datos["folds"]
    y = np.asarray([str(v) for v in datos["labels"]])
    ids = np.asarray([str(v) for v in datos["ids"]])
    clases = [str(c) for c in datos["clases"]]
    params = json.loads(str(datos["params_v1_json"]))
    idx_train = np.where(folds != k)[0]
    idx_val = np.where(folds == k)[0]
    if len(idx_val) == 0:
        raise ValueError(f"fold {k} sin documentos de validacion")
    X_train = X[idx_train]
    X_val = X[idx_val]
    del X
    gc.collect()
    modelo = LogisticRegression(**params)
    t_fit = time.perf_counter()
    modelo.fit(X_train, y[idx_train])
    tiempo_fit = time.perf_counter() - t_fit
    t_pred = time.perf_counter()
    probs = expandir_probs(modelo, X_val, clases)
    tiempo_pred = time.perf_counter() - t_pred
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        ruta_salida,
        ids=ids[idx_val],
        probs=probs.astype(np.float32),
        y_true=y[idx_val],
        clases=np.asarray(clases),
        n_iter=np.int64(modelo.n_iter_.max()),
        tiempo_seg=np.float64(time.perf_counter() - t0),
    )
    del X_train, X_val
    gc.collect()
    logger.info(
        "fold %d: n_train=%d n_val=%d iters=%d fit=%.1fs pred=%.1fs total=%.1fs -> %s",
        k,
        len(idx_train),
        len(idx_val),
        int(modelo.n_iter_.max()),
        tiempo_fit,
        tiempo_pred,
        time.perf_counter() - t0,
        ruta_salida,
    )


def _relay_salida(proceso: subprocess.Popen, k: int, logger: logging.Logger) -> None:
    """Canaliza la salida del subproceso de fold al log del orquestador.

    Args:
        proceso: Subproceso Popen con stdout=PIPE.
        k: Indice del fold.
        logger: Logger activo.
    """
    if proceso.stdout is None:
        return
    for linea in proceso.stdout:
        linea = linea.rstrip()
        if linea:
            logger.info("[fold %d] %s", k, linea)


def lanzar_folds(n_folds: int, args, logger: logging.Logger, rutas: dict) -> dict:
    """Lanza hasta --jobs subprocesos de fold en paralelo (reanudable).

    Args:
        n_folds: Numero de folds.
        args: Argumentos CLI.
        logger: Logger activo.
        rutas: Dict de rutas activas.

    Returns:
        Dict fold -> codigo de retorno (los folds ya existentes se saltan
        salvo --forzar).

    Raises:
        RuntimeError: Si algun fold termina con codigo distinto de 0.
    """
    pendientes = []
    for k in range(n_folds):
        salida = Path(rutas["fold"].format(k=k))
        if salida.exists() and not args.forzar:
            logger.info("fold %d ya existe, se reutiliza (%s)", k, salida)
        else:
            pendientes.append(k)
    entorno = os.environ.copy()
    for variable in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        entorno[variable] = str(CONFIG["hilos_por_fold"])
    activos: dict = {}
    codigos: dict = {}
    try:
        while pendientes or activos:
            while pendientes and len(activos) < max(1, args.jobs):
                k = pendientes.pop(0)
                comando = [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--modo",
                    "oof",
                    "--solo-fold",
                    str(k),
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
                    target=_relay_salida, args=(proceso, k, logger), daemon=True
                )
                hilo.start()
                activos[k] = (proceso, hilo)
                logger.info("fold %d lanzado (pid %d)", k, proceso.pid)
            time.sleep(1.0)
            for k in list(activos):
                proceso, hilo = activos[k]
                if proceso.poll() is not None:
                    hilo.join(timeout=10)
                    codigos[k] = int(proceso.returncode)
                    if proceso.returncode != 0:
                        logger.error("fold %d termino con rc=%d", k, proceso.returncode)
                    else:
                        logger.info("fold %d completado", k)
                    del activos[k]
    except KeyboardInterrupt:
        logger.warning("interrupcion: terminando %d folds activos", len(activos))
        for proceso, hilo in activos.values():
            proceso.terminate()
        for proceso, hilo in activos.values():
            proceso.wait(timeout=30)
            hilo.join(timeout=5)
        raise
    fallidos = {k: rc for k, rc in codigos.items() if rc != 0}
    if fallidos:
        raise RuntimeError(f"folds fallidos: {fallidos}")
    return codigos


def concatenar_oof(n_folds: int, rutas: dict, logger: logging.Logger) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Concatena las predicciones OOF de todos los folds ordenadas por id.

    Args:
        n_folds: Numero de folds.
        rutas: Dict de rutas activas.
        logger: Logger activo.

    Returns:
        (ids, probs, y_true, info por fold) ordenados por id.

    Raises:
        FileNotFoundError: Si falta la salida de algun fold.
    """
    ids, probs, y_true = [], [], []
    info = {}
    for k in range(n_folds):
        ruta = Path(rutas["fold"].format(k=k))
        if not ruta.exists():
            raise FileNotFoundError(f"falta la salida del fold {k}: {ruta}")
        with np.load(ruta, allow_pickle=False) as datos:
            ids.append(np.asarray([str(v) for v in datos["ids"]]))
            probs.append(datos["probs"].astype(np.float64))
            y_true.append(np.asarray([str(v) for v in datos["y_true"]]))
            info[str(k)] = {
                "n_val": int(datos["ids"].shape[0]),
                "n_iter": int(datos["n_iter"]),
                "tiempo_seg": float(datos["tiempo_seg"]),
            }
    ids = np.concatenate(ids)
    probs = np.concatenate(probs, axis=0)
    y_true = np.concatenate(y_true)
    orden = np.argsort(ids, kind="stable")
    logger.info("OOF concatenado: %d docs en %d folds", len(ids), n_folds)
    return ids[orden], probs[orden], y_true[orden], info


def guardar_oof_probs(
    ruta: str, ids, probs, y_true, smoke: bool, forzar: bool, logger: logging.Logger
) -> None:
    """Persiste ids + matriz de probabilidades OOF alineada al espacio fijo.

    Args:
        ruta: Ruta de salida .npz.
        ids: Ids ordenados.
        probs: Probabilidades OOF (n, 16).
        y_true: Etiquetas de train.
        smoke: True si es corrida smoke (permite reescribir en /tmp).
        forzar: True si se pidio rehacer la corrida (permite reescribir).
        logger: Logger activo.

    Raises:
        FileExistsError: Si la salida existe y no es smoke ni forzar.
    """
    destino = Path(ruta)
    if destino.exists() and not (smoke or forzar):
        raise FileExistsError(f"salida existente (no se sobrescribe): {destino}")
    destino.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        destino,
        ids=np.asarray([str(v) for v in ids]),
        probs=np.asarray(probs, dtype=np.float64),
        y_true=np.asarray([str(v) for v in y_true]),
        clases=np.asarray(CLASES_M4),
    )
    logger.info("probabilidades OOF guardadas: %s (%.1f MB)", destino, destino.stat().st_size / 1e6)


def ejecutar_modo_oof(df: pd.DataFrame, args, logger: logging.Logger, rutas: dict) -> tuple[dict, float]:
    """Ejecuta la auditoria OOF completa (matriz, folds, cleanlab, mayoria).

    Args:
        df: DataFrame con particiones.
        args: Argumentos CLI.
        logger: Logger activo.
        rutas: Dict de rutas activas.

    Returns:
        (resultado del modo, segundos totales).
    """
    t0 = time.perf_counter()
    base = cargar_modulo_baseline()
    params = base.cargar_hiperparametros_v1(rutas["modelo"])
    n_folds = CONFIG["n_folds_smoke"] if args.smoke else CONFIG["n_folds"]
    tiempos: dict = {}
    ruta_matriz = Path(rutas["oof_matrix"])
    if ruta_matriz.exists() and not args.forzar:
        datos_matriz = cargar_matriz_oof(ruta_matriz)
        folds = datos_matriz["folds"]
        y_train = datos_matriz["labels"]
        ids_train = datos_matriz["ids"]
        vec_params = json.loads(str(datos_matriz["vec_params_json"]))
        logger.info("matriz OOF reutilizada: %s (%d docs)", ruta_matriz, len(y_train))
    else:
        t = time.perf_counter()
        datos_vec = base.vectorizar(df, "lemas_mask")
        tiempos["vectorizacion_seg"] = time.perf_counter() - t
        X = datos_vec["X_train"]
        y_train = datos_vec["y_train"]
        ids_train = df.loc[df["particion"] == "train", "id"].values
        vec = datos_vec["vec"]
        vec_params = {
            "min_df": int(vec.min_df),
            "max_features": int(vec.max_features),
            "ngram_range": [int(v) for v in vec.ngram_range],
            "sublinear_tf": bool(vec.sublinear_tf),
        }
        del datos_vec["X_val"], datos_vec["X_test"]
        gc.collect()
        folds = asignar_folds(
            y_train,
            ids_train,
            n_folds,
            CONFIG["semilla"],
            CONFIG["umbral_clases_pequenas"],
            logger,
        )
        t = time.perf_counter()
        guardar_matriz_oof(ruta_matriz, X, ids_train, y_train, folds, params, vec_params, logger)
        tiempos["guardado_matriz_seg"] = time.perf_counter() - t
        del X, datos_vec
        gc.collect()
    if len(folds) != len(y_train):
        raise RuntimeError("la matriz OOF reutilizada tiene folds y etiquetas desalineados")

    # Memoria consciente: las features ya estan vectorizadas/guardadas y las
    # columnas de texto pesadas no se necesitan durante los folds; se liberan
    # antes de lanzar los subprocesos (la maquina tiene historial de OOM).
    y_val = df.loc[df["particion"] == "val", "area_primaria"].values
    y_test = df.loc[df["particion"] == "test", "area_primaria"].values
    df.drop(
        columns=["lemas_texto", "lemas_norm", "lemas_mask", "top3_json"],
        inplace=True,
        errors="ignore",
    )
    gc.collect()

    t = time.perf_counter()
    lanzar_folds(n_folds, args, logger, rutas)
    tiempos["folds_seg"] = time.perf_counter() - t
    t = time.perf_counter()
    ids_oof, probs_oof, y_oof, info_folds = concatenar_oof(n_folds, rutas, logger)
    tiempos["concatenacion_seg"] = time.perf_counter() - t
    if len(ids_oof) != len(y_train) or len(set(ids_oof.tolist())) != len(ids_oof):
        raise RuntimeError("la cobertura OOF no es exacta: hay ids repetidos o faltantes")
    n_train_total = len(y_train)
    for info in info_folds.values():
        info["n_train"] = n_train_total - info["n_val"]

    t = np.asarray(CLASES_M4)[np.argmax(probs_oof, axis=1)]
    metricas_oof = evaluar_predicciones(y_oof, t, CLASES_M4)
    logger.info(
        "OOF: macro-F1=%.4f (fijo %.4f, acc %.4f)",
        metricas_oof["macro_f1"],
        metricas_oof["macro_f1_espacio_fijo"],
        metricas_oof["accuracy"],
    )
    t = time.perf_counter()
    audit_oof = auditar_cleanlab(
        y_oof, probs_oof, ids_oof, CLASES_M4, CONFIG["top_ejemplos"], logger,
        "cleanlab OOF train (out-of-sample)",
    )
    tiempos["cleanlab_seg"] = time.perf_counter() - t
    guardar_oof_probs(rutas["oof_probs"], ids_oof, probs_oof, y_oof, args.smoke, args.forzar, logger)

    mayoria = str(pd.Series(y_train).value_counts().idxmax())
    pred_mayoria_val = np.full(len(y_val), mayoria)
    pred_mayoria_test = np.full(len(y_test), mayoria)
    baseline_mayoria = {
        "clase": mayoria,
        "val": evaluar_predicciones(y_val, pred_mayoria_val, CLASES_M4),
        "test": evaluar_predicciones(y_test, pred_mayoria_test, CLASES_M4),
    }
    logger.info(
        "baseline mayoria (%s): val macro-F1 fijo=%.4f | test macro-F1 fijo=%.4f",
        mayoria,
        baseline_mayoria["val"]["macro_f1_espacio_fijo"],
        baseline_mayoria["test"]["macro_f1_espacio_fijo"],
    )
    resultado = {
        "n_folds": int(n_folds),
        "semilla_folds": CONFIG["semilla"],
        "parametros_v1": params,
        "parametros_vectorizer": vec_params,
        "folds": info_folds,
        "tiempos_seg": tiempos,
        "metricas_oof": metricas_oof,
        "cleanlab_oof": audit_oof,
        "baseline_mayoria": baseline_mayoria,
        "artefactos": {
            "matriz": rutas["oof_matrix"],
            "probs": rutas["oof_probs"],
            "md5_folds": {
                str(k): md5_archivo(rutas["fold"].format(k=k)) for k in range(n_folds)
            },
        },
    }
    return resultado, time.perf_counter() - t0


# ===== SECCION 6: figura y resumen =====


def figura_ruido(
    resultado: dict, ruta: str, smoke: bool, forzar: bool, logger: logging.Logger
) -> None:
    """Grafica la tasa de issues por clase frente al soporte.

    Fuente primaria: OOF train si existe (estimacion valida); si no, test
    out-of-sample del joblib v1. La serie de val v1 (in-sample,
    contaminada) se superpone como diagnostico de memorizacion.

    Args:
        resultado: Resultado consolidado de la auditoria.
        ruta: Ruta de salida PNG.
        smoke: True si es corrida smoke (permite reescribir).
        forzar: True si se pidio rehacer la corrida (permite reescribir).
        logger: Logger activo.

    Raises:
        FileExistsError: Si la figura existe y no es smoke ni forzar.
    """
    destino = Path(ruta)
    if destino.exists() and not (smoke or forzar):
        raise FileExistsError(f"salida existente (no se sobrescribe): {destino}")
    matplotlib.use("Agg")
    series = []
    modo_oof = resultado.get("modo_oof")
    modo_val = resultado.get("modo_val")
    if modo_oof:
        series.append(
            ("OOF train (out-of-sample, fuente valida)", modo_oof["cleanlab_oof"], "#1f77b4", "o", True)
        )
    if modo_val:
        if not modo_oof:
            series.append(
                ("test v1 (out-of-sample del joblib v1)", modo_val["cleanlab_test_out_of_sample"], "#1f77b4", "o", True)
            )
        series.append(
            ("val v1 (in-sample, contaminado)", modo_val["cleanlab_val_in_sample"], "#d62728", "x", False)
        )
    if not series:
        logger.warning("figura: sin auditorias disponibles, no se genera")
        return
    figura, eje = plt.subplots(figsize=(11, 7))
    for etiqueta, auditoria, color, marcador, anotar in series:
        por_clase = auditoria["tasa_issues_por_clase"]
        x = [v["soporte"] for v in por_clase.values()]
        y = [(v["tasa"] if v["tasa"] is not None else 0.0) for v in por_clase.values()]
        eje.scatter(x, y, s=50, c=color, marker=marcador, label=etiqueta, alpha=0.85)
        if anotar:
            desplazamientos = [(3, 3), (3, -9), (-14, 6), (-14, -10)]
            for pos, (clase, valores) in enumerate(por_clase.items()):
                if valores["tasa"] is None:
                    continue
                dx, dy = desplazamientos[pos % len(desplazamientos)]
                eje.annotate(
                    clase.replace("_", " ")[:24],
                    (valores["soporte"], valores["tasa"]),
                    fontsize=6.5,
                    xytext=(dx, dy),
                    textcoords="offset points",
                )
    principal = series[0][1]
    eje.axhline(
        principal["tasa_issues"],
        color=series[0][2],
        linestyle="--",
        linewidth=1.0,
        alpha=0.6,
        label=f"tasa global {principal['tasa_issues']:.3f}",
    )
    eje.set_xscale("log")
    eje.set_xlabel("soporte de la clase (n docs, escala log)")
    eje.set_ylabel("tasa de issues cleanlab")
    eje.set_title(
        "Auditoria de ruido de etiquetas silver M4\n"
        f"fuente: {series[0][0]} | {principal['n_issues']} issues / {principal['n_docs']} docs"
    )
    eje.grid(alpha=0.25)
    eje.legend(fontsize=8, loc="best")
    Path(ruta).parent.mkdir(parents=True, exist_ok=True)
    figura.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(figura)
    logger.info("figura guardada: %s", ruta)


def construir_resumen(resultado: dict) -> dict:
    """Extrae las cifras clave de la auditoria para el JSON y la consola.

    Args:
        resultado: Resultado consolidado.

    Returns:
        Dict con las metricas titulares disponibles.
    """
    resumen = {}
    if "modo_val" in resultado:
        modo_val = resultado["modo_val"]
        resumen["v1_val_macro_f1_observado"] = modo_val["metricas_v1"]["val"]["macro_f1"]
        resumen["v1_test_macro_f1_observado"] = modo_val["metricas_v1"]["test"]["macro_f1"]
        resumen["v1_test_macro_f1_espacio_fijo"] = modo_val["metricas_v1"]["test"]["macro_f1_espacio_fijo"]
        resumen["v1_test_accuracy"] = modo_val["metricas_v1"]["test"]["accuracy"]
        resumen["cleanlab_test_tasa_issues"] = modo_val["cleanlab_test_out_of_sample"]["tasa_issues"]
        resumen["ic95_test_macro_f1_espacio_fijo"] = [
            modo_val["bootstrap_macro_f1_espacio_fijo"]["test"]["ic95_inf"],
            modo_val["bootstrap_macro_f1_espacio_fijo"]["test"]["ic95_sup"],
        ]
    if "modo_oof" in resultado:
        modo_oof = resultado["modo_oof"]
        resumen["oof_macro_f1_observado"] = modo_oof["metricas_oof"]["macro_f1"]
        resumen["oof_macro_f1_espacio_fijo"] = modo_oof["metricas_oof"]["macro_f1_espacio_fijo"]
        resumen["oof_tasa_issues"] = modo_oof["cleanlab_oof"]["tasa_issues"]
        resumen["baseline_mayoria_test_macro_f1_espacio_fijo"] = modo_oof["baseline_mayoria"]["test"]["macro_f1_espacio_fijo"]
    return resumen


def imprimir_resumen(resultado: dict, logger: logging.Logger) -> None:
    """Imprime el resumen final en consola (maximo 10 lineas).

    Args:
        resultado: Resultado consolidado.
        logger: Logger activo.
    """
    resumen = resultado.get("resumen", {})
    logger.info("RESUMEN AUDITORIA ETIQUETAS M4 (silver)")
    if "modo_val" in resultado:
        comparacion = resultado["modo_val"]["comparacion_m4_baseline"]
        logger.info(
            "v1 test: macro-F1 obs=%.4f (ref %.4f) fijo=%.4f acc=%.4f | val=%.4f (in-sample)",
            resumen["v1_test_macro_f1_observado"],
            comparacion.get("test_macro_f1_referencia", float("nan")),
            resumen["v1_test_macro_f1_espacio_fijo"],
            resumen["v1_test_accuracy"],
            resumen["v1_val_macro_f1_observado"],
        )
        logger.info(
            "cleanlab test (out-of-sample v1): %.2f %% issues | IC95 macro-F1 fijo [%.4f, %.4f]",
            100.0 * resumen["cleanlab_test_tasa_issues"],
            resumen["ic95_test_macro_f1_espacio_fijo"][0],
            resumen["ic95_test_macro_f1_espacio_fijo"][1],
        )
    if "modo_oof" in resultado:
        logger.info(
            "OOF train: macro-F1 obs=%.4f fijo=%.4f | cleanlab %.2f %% issues",
            resumen["oof_macro_f1_observado"],
            resumen["oof_macro_f1_espacio_fijo"],
            100.0 * resumen["oof_tasa_issues"],
        )
        logger.info(
            "baseline mayoria test: macro-F1 fijo=%.4f (clase %s)",
            resumen["baseline_mayoria_test_macro_f1_espacio_fijo"],
            resultado["modo_oof"]["baseline_mayoria"]["clase"],
        )
    logger.info("JSON: %s | figura: %s", resultado["salidas"]["json"], resultado["salidas"]["figura"])


# ===== SECCION 7: CLI y main =====


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Define y parsea los argumentos de linea de comandos.

    Args:
        argv: Lista de argumentos (None usa sys.argv).

    Returns:
        Namespace con modo, jobs, solo_fold, smoke y forzar.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Auditoria de ruido de etiquetas silver M4: cleanlab, OOF 5-fold "
            "y IC bootstrap. Diagnostico, no mejora."
        )
    )
    parser.add_argument("--modo", choices=["val", "oof", "todo"], default="val", help="val: joblib v1 sin reentrenar; oof: 5-fold sobre train; todo: ambos")
    parser.add_argument("--jobs", type=int, default=CONFIG["jobs"], help="maximo de subprocesos de fold en paralelo (default 4)")
    parser.add_argument("--solo-fold", type=int, default=None, dest="solo_fold", help="interno: ejecuta solo el fold k (subproceso)")
    parser.add_argument("--smoke", action="store_true", help="subconjunto 400 docs / 2 folds; salidas solo en /tmp/opencode")
    parser.add_argument("--forzar", action="store_true", help="rehace la matriz OOF y los folds existentes")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Orquesta la auditoria segun el modo pedido.

    Args:
        argv: Argumentos CLI (None usa sys.argv).
    """
    args = parse_args(argv)
    os.chdir(RAIZ)
    rutas = rutas_activas(args.smoke)
    if args.solo_fold is not None:
        logger = configurar_logger(rutas["log"], solo_consola=True)
        ejecutar_fold(args.solo_fold, args, logger, rutas)
        return
    logger = configurar_logger(rutas["log"])
    random.seed(CONFIG["semilla"])
    np.random.seed(CONFIG["semilla"])
    logger.info(
        "=== INICIO auditoria etiquetas M4 | modo=%s smoke=%s jobs=%d forzar=%s ===",
        args.modo,
        args.smoke,
        args.jobs,
        args.forzar,
    )
    if args.smoke:
        Path(CONFIG["dir_smoke"]).mkdir(parents=True, exist_ok=True)
    verificar_salidas_nuevas(args.modo, rutas, args.smoke, args.forzar)
    t0 = time.perf_counter()
    if args.smoke:
        df = cargar_datos_smoke(CONFIG["limite_smoke"], CONFIG["semilla"], logger)
    else:
        df = cargar_datos_completo(logger)
    resultado = {
        "fecha": pd.Timestamp.now().isoformat(),
        "script": "scripts/17_auditoria_etiquetas_m4.py",
        "modo": args.modo,
        "smoke": bool(args.smoke),
        "versiones": versiones_entorno(),
        "md5_entradas": {
            "lemas": md5_archivo(rutas["lemas"]),
            "etiquetas": md5_archivo(rutas["etiquetas"]),
            "splits": md5_archivo(rutas["splits"]),
            "modelo_v1": md5_archivo(rutas["modelo"]),
            "vectorizer_v1": md5_archivo(rutas["vectorizer"]),
            "m4_baseline_json": md5_archivo(rutas["referencia"]),
        },
        "premisas": PREMISAS,
        "clases_espacio_fijo": CLASES_M4,
        "parametros_cli": {"modo": args.modo, "jobs": args.jobs, "smoke": args.smoke, "forzar": args.forzar},
    }
    tiempos = {}
    if args.modo in ("val", "todo"):
        logger.info("--- modo val: joblib v1 sin reentrenar ---")
        resultado["modo_val"], tiempos["modo_val_seg"] = ejecutar_modo_val(df, args, logger, rutas)
    if args.modo in ("oof", "todo"):
        logger.info("--- modo oof: 5-fold sobre train (n_folds=%d) ---", CONFIG["n_folds_smoke"] if args.smoke else CONFIG["n_folds"])
        resultado["modo_oof"], tiempos["modo_oof_seg"] = ejecutar_modo_oof(df, args, logger, rutas)
    del df
    gc.collect()
    figura_ruido(resultado, rutas["figura"], args.smoke, args.forzar, logger)
    resultado["tiempos_seg"] = tiempos
    resultado["tiempos_seg"]["total_seg"] = time.perf_counter() - t0
    resultado["salidas"] = {"json": rutas["json"], "figura": rutas["figura"]}
    resultado["resumen"] = construir_resumen(resultado)
    destino_json = Path(rutas["json"])
    if destino_json.exists() and not (args.smoke or args.forzar):
        raise FileExistsError(f"salida existente (no se sobrescribe): {destino_json}")
    destino_json.parent.mkdir(parents=True, exist_ok=True)
    with open(destino_json, "w", encoding="utf-8") as archivo:
        json.dump(resultado, archivo, ensure_ascii=False, indent=2, default=str)
    logger.info("JSON guardado: %s (md5 %s)", destino_json, md5_archivo(rutas["json"]))
    imprimir_resumen(resultado, logger)
    logger.info("=== FIN auditoria etiquetas M4 en %.1f s ===", time.perf_counter() - t0)


if __name__ == "__main__":
    try:
        main()
    except Exception:  # noqa: BLE001 - protocolo: traza completa y salida 1 ante cualquier fallo
        traceback.print_exc()
        sys.exit(1)
