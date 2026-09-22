"""Baseline clasico M4: clasificacion de area tematica constitucional.

Pipeline v1 (Fase L1, ADR-2/6/8):
    1. Carga lemas por documento + etiquetas de area (via Constitucion).
    2. Split estratificado 70/15/15 UNA VEZ (seed 42) guardado en
       data/splits_m4.parquet.
    3. Escenario A (completo): TF-IDF sobre lemas + LinearSVC / LogReg.
       Escenario B (enmascarado): citas normativas enmascaradas con [CITA]
       a nivel de lemas (aproximacion documentada) + mismos modelos.
    4. Metricas macro-F1/accuracy/F1 por clase en test (tabla 2x2 modelo x
       escenario), matriz de confusion, SHAP del mejor lineal.
    5. Guardado de metricas, figuras y modelo.

Ciclo v2 - Fase U (2026-09-11): tres variantes sobre las MISMAS features
ganadoras v1 (TF-IDF de lemas enmascarado, hiperparametros leidos del joblib
v1 con get_params) y los MISMOS splits canonicos:
    A. Jerarquica (cascada): LogReg grueso sobre 6 macro-grupos de la CP +
       un LogReg fino por macro-grupo (entrenado solo con docs del grupo);
       prediccion = clase fina del grupo predicho.
    B. Multietiqueta: OneVsRest LogReg (16 binarios) sobre los bloques con
       n>=1 de top3_json, con umbrales por clase optimizados por coordinate
       ascent sobre macro-F1 multietiqueta de VALIDACION.
    C. Calibracion del LogReg plano v1: base reentrenada SOLO con train
       usando los hiperparametros v1 (el joblib v1 reentreno en train+val en
       su corrida final y contaminaria la calibracion) + sesgos aditivos por
       clase (16) optimizados por coordinate ascent sobre macro-F1 de
       VALIDACION.
    Seleccion: mejor macro-F1 single-label de validacion; test solo al
    final. Salida: resultados/m4_jerarquico.json (m4_baseline.json se
    mantiene INTOCABLE). Sin leakage: los targets multietiqueta provienen
    de citas, las features SOLO del texto enmascarado de lemas.

Ciclo v2 - Fase W(b) (2026-09-11): EuroVoc-es como senal auxiliar, sobre las
MISMAS features/splits y el mismo protocolo (seleccion por validacion, test
una sola vez). El mapeo clase->categoria y el lexico espanol se construyen con
scripts/15_eurovoc_lexico.py (SPARQL oficial de Publications Office; cobertura
parcial anticipada por E1):
    D. Jerarquica con taxonomia EuroVoc: misma cascada que A pero los
       macro-grupos (8) provienen del mapeo EuroVoc de las 16 clases
       (config/eurovoc_bloques.json). Es reestructuracion del target, no
       feature de entrada: sin leakage.
    E. Features auxiliares por lexico EuroVoc: para cada documento, 16
       columnas = proporcion de aciertos de terminos EuroVoc de la categoria
       de cada clase (lemas del cierre skos:broader+ de sus conceptos raiz),
       concatenadas (hstack) al TF-IDF v1; LogReg reentrenado con los
       hiperparametros v1. Las features salen SOLO del texto enmascarado.
    Salida NUEVA: resultados/m4_eurovoc.json (m4_jerarquico.json queda
    INTOCABLE como referencia de C test=0.2980). Recomendacion explicita de
    adopcion por variante: solo se adopta si gana en validacion y su test no
    cae por debajo de C.

Ciclo v3 - Sprint A5 (2026-09-11): modo --fase-a5 (flag NUEVO, default OFF),
sobre las MISMAS features/splits canonicos y el mismo protocolo (seleccion
por validacion, test una sola vez):
    1. Reproduccion de la C calibrada de la Fase U desde su artefacto
       (modelos/m4_jerarquico/ganadora_C_calibracion.joblib): sanity
       obligatorio test=0.2980095633575616 +- 1e-6.
    2. BETO chunk-mean (Fase V) sobre la cache de chunks: inferencia val/test
       del checkpoint beto_completo reutilizando las funciones de
       scripts/07_transformer.py (forward_doc_chunk_mean, ChunksDataset,
       collate_chunks) y softmax fp32 sobre los logits. Sanity obligatorio
       test=0.2433442600823086 +- 0.001; variante secundaria enmascarada
       (0.23632401040291953 +- 0.001).
    3. Ensamble C+BETO: p_ens = w*p_C + (1-w)*p_BETO, w en {0.5,0.6,0.7,0.8}
       elegido por macro-F1 de VALIDACION; test del w elegido + bootstrap
       pareado (1000, seed 42) de la diferencia de macro-F1 test vs C.
    4. Class weights: LogReg class_weight='balanced' y 1/sqrt(frecuencia)
       entrenados en train + la misma calibracion de sesgos por clase en
       validacion (procedimiento de la Fase U).
    Salida NUEVA: resultados/m4_clasificador_c3.json (version "c3-a5"); los
    JSON y modelos de ciclos previos quedan INTOCABLES (md5 en el propio
    JSON). Smoke: --fase-a5 --limite N escribe en /tmp/opencode/.

Uso:
    /home/carlos/venv/bin/python scripts/06_baseline_clasico.py
    /home/carlos/venv/bin/python scripts/06_baseline_clasico.py --fase-u
    /home/carlos/venv/bin/python scripts/06_baseline_clasico.py --limite 1500
    /home/carlos/venv/bin/python scripts/06_baseline_clasico.py --fase-a5
    /home/carlos/venv/bin/python scripts/06_baseline_clasico.py --fase-a5 --limite 800
"""

import argparse
import hashlib
import importlib
import json
import logging
import re
import sys
import time
import traceback
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.sparse import csr_matrix, hstack, vstack
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

# ===== SECCION 1: configuracion =====
CONFIG = {
    "semilla": 42,
    "min_df": 3,
    "max_features": 100000,  # reducido 200k->100k por OOM con job externo en RAM
    "ngram": (1, 2),
    "optuna_trials": 15,  # reducido 25->15 por OOM con job externo en RAM
    "shap_muestra": 500,
    "ruta_lemas": "data/lemas_por_doc.parquet",
    "ruta_etiquetas": "data/etiquetas_areas.parquet",
    "ruta_splits": "data/splits_m4.parquet",
    "dir_modelo": "modelos/m4_baseline",
    "ruta_metricas": "resultados/m4_baseline.json",
    "ruta_fig_confusion": "figuras/fig_matriz_confusion_m4.png",
    "ruta_fig_shap": "figuras/shap_lemas_m4.png",
    "ruta_log": "experimentos/m4_baseline.log",
    # --- ciclo v2 - Fase U/W(b) ---
    "dir_modelo_v1": "modelos/m4_baseline",
    "ruta_metricas_v1": "resultados/m4_baseline.json",
    "ruta_json_fase_u": "resultados/m4_eurovoc.json",
    "ruta_json_jerarquico_ref": "resultados/m4_jerarquico.json",
    "ruta_json_smoke": "/tmp/opencode/m4_eurovoc_smoke.json",
    "ruta_log_fase_u": "experimentos/m4_eurovoc.log",
    "ruta_log_smoke": "/tmp/opencode/m4_eurovoc_smoke.log",
    "dir_modelo_fase_u": "modelos/m4_eurovoc",
    "ruta_fig_variantes": "figuras/fig_m4_eurovoc_variantes.png",
    "ruta_eurovoc_bloques": "config/eurovoc_bloques.json",
    "ruta_eurovoc_terminos": "config/eurovoc_terminos_es.json",
    # --- ciclo v3 - Sprint A5 (clasificador C + BETO chunk-mean + class weights) ---
    "ruta_artefacto_c": "modelos/m4_jerarquico/ganadora_C_calibracion.joblib",
    "dir_beto_chunks": "modelos/m4_transformer_chunks",
    "dir_cache_chunks": "data/cache_chunks_beto",
    "ruta_json_c3": "resultados/m4_clasificador_c3.json",
    "ruta_json_c3_smoke": "/tmp/opencode/m4_clasificador_c3_smoke.json",
    "ruta_log_c3": "experimentos/m4_clasificador_c3.log",
    "ruta_log_c3_smoke": "/tmp/opencode/m4_clasificador_c3_smoke.log",
    "ruta_fig_c3": "figuras/c3_a5_clasificador.png",
    "pesos_ensamble": [0.5, 0.6, 0.7, 0.8],
    "bootstrap_n": 1000,
    "bootstrap_seed": 42,
    "tolerancia_sanity_beto": 0.001,
    "sanity_n_repro_07": 20,
    "modos_class_weight": ["balanced", "sqrt_inv"],
}

LOGGER = logging.getLogger("m4_baseline")

# Lemas que aparecen como nucleo de citas normativas tras lematizacion.
LEMAS_CITA = {
    # Con tilde (forma dominante de spaCy: articulo 91.784 vs articulo 1.215)
    "artículo",
    "artículos",
    # Sin tilde (variantes residuales)
    "articulo",
    "articulos",
    "ley",
    "leyes",
    "decreto",
    "decretos",
    "acto",
    "legislativo",
    "sentencia",
    "sentencias",
    "auto",
    "numeral",
    "numerales",
    "inciso",
    "incisos",
    "literal",
    "literales",
    "parágrafo",
    "paragrafo",
    "constitución",
    "constitucion",
    "política",  # "politica constitucional" no es cita; se excluye abajo
}
# "politica" se retira: no es nucleo de cita (falso positivo del reviewer).
LEMAS_CITA.discard("política")
# Token [CITA] que reemplaza las secuencias enmascaradas.
TOKEN_CITA = "CITA"
# Numeros romanos REALES (regex estricta; evita falsos como civil/mi/mil).
RE_ROMANO = re.compile(
    r"^(?=.{1,7}$)m{0,4}(cm|cd|d?c{0,3})(xc|xl|l?x{0,3})(ix|iv|v?i{0,3})$",
    re.IGNORECASE,
)
# Numeros arabigos aislados (nucleo de toda cita).
RE_ARABIGO = re.compile(r"^\d{1,4}$")


def es_numero_cita(token: str) -> bool:
    """Determina si un token de lemas puede ser el numero de una cita.

    Args:
        token: Token (posible arabigo o romano).

    Returns:
        True si es arabigo de 1-4 digitos o romano real (regex estricta
        que evita falsos como civil/mi/mil).
    """
    return bool(RE_ARABIGO.match(token) or RE_ROMANO.match(token))
# Prefijos de sentencia citada ya lematizados (t, c, su + numero).
RE_SENT_CITADA = re.compile(r"^(?:t|c|su)$", re.IGNORECASE)


def configurar_logging(ruta_log: str | None = None) -> logging.Logger:
    """Configura el logger del experimento (archivo + consola).

    Args:
        ruta_log: Ruta del archivo de log; si es None se usa la del baseline
            v1. La Fase U usa su propio log para no tocar el de v1.

    Returns:
        Logger configurado.
    """
    ruta = ruta_log or CONFIG["ruta_log"]
    Path(ruta).parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(ruta, mode="w"),
            logging.StreamHandler(),
        ],
        force=True,
    )
    return LOGGER


def normalizar_lemas(texto: str) -> str:
    """Normaliza lemas multi-palabra a un solo token con guion bajo.

    Args:
        texto: Secuencia de lemas separados por espacio.

    Returns:
        Secuencia con lemas multi-palabra unidos por "_".
    """
    return " ".join(tok.replace(" ", "_") for tok in texto.split())


def enmascarar_citas_lemas(texto: str) -> str:
    """Enmascara secuencias de lemas tipicas de citas normativas.

    Correcciones tras auditoria L2 (2026-09-09, ml-reviewer):
    - La lematizacion (is_alpha) elimino los arabigos, pero los ROMANOS
      sobreviven ("ley ii", "articulo xxix"): se enmascaran nucleo+romano.
    - Se anaden nucleos con tilde ("articulo") y plurales, que spaCy emite.
    - Romano con regex estricta (evita falsos civil/mi/mil).
    - Ademas del patron nucleo+numero, se enmascara romano suelto tras
      nucleo aunque haya palabras intermedias no relevantes ("numeral 3 del
      literal b" -> una sola CITA por segmento).

    Args:
        texto: Secuencia de lemas normalizados.

    Returns:
        Secuencia con las citas reemplazadas por CITA.
    """
    tokens = texto.split()
    salida: list[str] = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        base = tok.split("_")[0]
        if base in LEMAS_CITA:
            # Consumir el nucleo y todo numero/anexo contiguo.
            j = i + 1
            consumio_numero = False
            while j < len(tokens):
                sig = tokens[j]
                if es_numero_cita(sig):
                    consumio_numero = True
                    j += 1
                elif consumio_numero and sig in {"de", "del", "y", "la", "el", "las", "los"}:
                    # Conectores dentro de series de citas ("articulo 11 y 13").
                    j += 1
                    if j < len(tokens) and es_numero_cita(tokens[j]):
                        continue
                    j -= 1
                    break
                else:
                    break
            if consumio_numero:
                salida.append(TOKEN_CITA)
                i = j
            else:
                salida.append(tok)
                i += 1
        elif RE_SENT_CITADA.match(base) and i + 1 < len(tokens) and es_numero_cita(tokens[i + 1]):
            # Sentencia citada: t/c/su + numero (+ de + ano).
            j = i + 2
            if j < len(tokens) and tokens[j] == "de" and j + 1 < len(tokens) and es_numero_cita(tokens[j + 1]):
                j += 2
            salida.append(TOKEN_CITA)
            i = j
        else:
            salida.append(tok)
            i += 1
    return " ".join(salida)


def cargar_datos() -> pd.DataFrame:
    """Carga y une lemas + etiquetas, excluyendo sin_area_definida.

    Returns:
        DataFrame con id, lemas, area y particiones train/val/test.
    """
    lemas = pq.read_table(CONFIG["ruta_lemas"]).to_pandas()
    etiquetas = pq.read_table(CONFIG["ruta_etiquetas"]).to_pandas()
    columnas_etiquetas = ["id", "area_primaria", "tipo", "ano", "top3_json"]
    df = lemas.merge(etiquetas[columnas_etiquetas], on="id")
    n_antes = len(df)
    df = df[df["area_primaria"] != "sin_area_definida"].reset_index(drop=True)
    # Clases con <3 docs no admiten split estratificado 70/15/15 (elecciones
    # tiene 1 doc): se agrupan en otras_areas (documentado en metricas).
    conteo = df["area_primaria"].value_counts()
    raras = conteo[conteo < 3].index.tolist()
    if raras:
        df.loc[df["area_primaria"].isin(raras), "area_primaria"] = "otras_areas"
        LOGGER.info("clases raras (<3 docs) agrupadas en otras_areas: %s", raras)
    df["lemas_norm"] = df["lemas_texto"].map(normalizar_lemas)
    df["lemas_mask"] = df["lemas_norm"].map(enmascarar_citas_lemas)
    LOGGER.info(
        "datos: %d filas (%d excluidas sin_area), %d clases",
        len(df),
        n_antes - len(df),
        df["area_primaria"].nunique(),
    )
    return df


def crear_splits(df: pd.DataFrame) -> pd.DataFrame:
    """Crea el split estratificado 70/15/15 UNA VEZ y lo persiste.

    Args:
        df: DataFrame con lemas y area.

    Returns:
        DataFrame con columna particion (train/val/test).
    """
    ruta = Path(CONFIG["ruta_splits"])
    if ruta.exists():
        splits = pq.read_table(ruta).to_pandas()
        df = df.merge(splits, on="id")
        LOGGER.info("splits existentes reutilizados: %s", splits["particion"].value_counts().to_dict())
        return df
    _, resto = train_test_split(
        df["id"],
        test_size=0.30,
        random_state=CONFIG["semilla"],
        stratify=df["area_primaria"],
    )
    # Clases con 1 doc en resto rompen el split estratificado val/test:
    # se asignan manualmente a val y se estratifica el resto.
    conteo_resto = df.loc[resto.index, "area_primaria"].value_counts()
    unicas = conteo_resto[conteo_resto == 1].index
    ids_unicas = resto[df.loc[resto.index, "area_primaria"].isin(unicas)]
    resto_est = resto[~resto.isin(ids_unicas)]
    if len(ids_unicas):
        LOGGER.info(
            "clases con 1 doc en resto asignadas a val: %s",
            df.loc[ids_unicas.index, "area_primaria"].tolist(),
        )
    val, test = train_test_split(
        resto_est,
        test_size=0.50,
        random_state=CONFIG["semilla"],
        stratify=df.loc[resto_est.index, "area_primaria"],
    )
    val = pd.Index(list(val) + list(ids_unicas))
    df["particion"] = "train"
    df.loc[df["id"].isin(val), "particion"] = "val"
    df.loc[df["id"].isin(test), "particion"] = "test"
    pq.write_table(
        pa.Table.from_pandas(df[["id", "particion"]]),
        ruta,
    )
    LOGGER.info("splits creados: %s", df["particion"].value_counts().to_dict())
    return df


def vectorizar(df: pd.DataFrame, columna: str) -> dict:
    """Vectoriza un escenario con TF-IDF sobre la columna dada.

    Args:
        df: DataFrame con particiones.
        columna: lemas_norm (completo) o lemas_mask (enmascarado).

    Returns:
        Dict con matrices X_train/X_val/X_test, vectorizador y clases.
    """
    vec = TfidfVectorizer(
        min_df=CONFIG["min_df"],
        max_features=CONFIG["max_features"],
        ngram_range=CONFIG["ngram"],
        sublinear_tf=True,
    )
    tr = df[df["particion"] == "train"]
    va = df[df["particion"] == "val"]
    te = df[df["particion"] == "test"]
    X_train = vec.fit_transform(tr[columna])
    X_val = vec.transform(va[columna])
    X_test = vec.transform(te[columna])
    LOGGER.info(
        "vectorizado %s: %d features (train %d, val %d, test %d)",
        columna,
        len(vec.get_feature_names_out()),
        X_train.shape[0],
        X_val.shape[0],
        X_test.shape[0],
    )
    return {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "vec": vec,
        "y_train": tr["area_primaria"].values,
        "y_val": va["area_primaria"].values,
        "y_test": te["area_primaria"].values,
    }


def entrenar_linearsvc(datos: dict) -> Pipeline:
    """Entrena LinearSVC sobre TF-IDF (baseline rapido).

    Args:
        datos: Dict de vectorizar().

    Returns:
        Pipeline ajustado.
    """
    modelo = LinearSVC(C=1.0, random_state=CONFIG["semilla"], dual="auto")
    modelo.fit(datos["X_train"], datos["y_train"])
    return modelo


def entrenar_logreg_optuna(datos: dict) -> LogisticRegression:
    """Optimiza LogisticRegression multinomial con Optuna (macro-F1 en val).

    Nota de diseño (2026-09-08): LightGBM sobre TF-IDF disperso de 100k
    features provoco muertes silenciosas por presion de RAM en esta
    maquina compartida (3 corridas) y es suboptimo para texto (los modelos
    lineales dominan en clasificacion de texto disperso). Se sustituye
    por LogisticRegression multinomial optimizada con Optuna: misma
    familia de busqueda de hiperparametros, huella de RAM minima.

    Args:
        datos: Dict de vectorizar().

    Returns:
        Mejor LogisticRegression reentrenada en train+val.
    """
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objetivo(trial: optuna.Trial) -> float:
        params = {
            "C": trial.suggest_float("C", 1e-2, 1e2, log=True),
            "penalty": trial.suggest_categorical("penalty", ["l2"]),
            "solver": "lbfgs",
            "max_iter": 600,
            "random_state": CONFIG["semilla"],
        }
        modelo = LogisticRegression(**params)
        modelo.fit(datos["X_train"], datos["y_train"])
        y_pred = modelo.predict(datos["X_val"])
        return f1_score(datos["y_val"], y_pred, average="macro")

    estudio = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=CONFIG["semilla"]))
    estudio.optimize(objetivo, n_trials=CONFIG["optuna_trials"], show_progress_bar=False)
    LOGGER.info("optuna: mejor macro-F1 val=%.4f params=%s", estudio.best_value, estudio.best_params)

    # Reentrenar la mejor configuracion en train+val para evaluar en test.
    X_tv = vstack([datos["X_train"], datos["X_val"]]).tocsr()
    y_tv = np.concatenate([datos["y_train"], datos["y_val"]])
    modelo = LogisticRegression(
        C=estudio.best_params["C"],
        penalty="l2",
        solver="lbfgs",
        max_iter=600,
        random_state=CONFIG["semilla"],
    )
    modelo.fit(X_tv, y_tv)
    return modelo


def evaluar(modelo, datos: dict, es_logreg: bool = False) -> dict:
    """Evalua un modelo en test y devuelve metricas + predicciones.

    Args:
        modelo: Modelo ajustado (LinearSVC o LGBM).
        datos: Dict de vectorizar().
        es_logreg: True si el modelo es LogisticRegression (no necesario: predict directo).

    Returns:
        Dict con macro_f1, accuracy, f1_por_clase, y_pred.
    """
    y_pred = modelo.predict(datos["X_test"])
    return {
        "macro_f1": float(f1_score(datos["y_test"], y_pred, average="macro")),
        "accuracy": float(accuracy_score(datos["y_test"], y_pred)),
        "f1_por_clase": classification_report(
            datos["y_test"], y_pred, output_dict=True, zero_division=0
        ),
        "y_pred": y_pred,
    }


def figura_confusion(y_true, y_pred, ruta: str) -> None:
    """Genera la matriz de confusion de las 8 clases mas frecuentes.

    Args:
        y_true: Etiquetas reales.
        y_pred: Predicciones.
        ruta: Ruta del PNG a guardar.
    """
    top = pd.Series(y_true).value_counts().head(8).index.tolist()
    mask = pd.Series(y_true).isin(top)
    cm = confusion_matrix(
        pd.Series(y_true)[mask], pd.Series(y_pred)[mask], labels=top, normalize="true"
    )
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(top)))
    ax.set_xticklabels(top, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top, fontsize=8)
    for i in range(len(top)):
        for j in range(len(top)):
            ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im)
    ax.set_title("Matriz de confusion M4 (top-8 clases, normalizada)")
    ax.set_xlabel("Prediccion")
    ax.set_ylabel("Real")
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


def shap_lineal(modelo, datos: dict, ruta: str) -> dict:
    """Importancia de lemas del modelo lineal (pesos absolutos medios).

    Nota: para LinearSVC se usa la magnitud media de los coeficientes por
    clase (proxy SHAP lineal exacto para modelos lineales).

    Args:
        modelo: LinearSVC ajustado.
        datos: Dict de vectorizar().
        ruta: Ruta del PNG.

    Returns:
        Dict con top-20 lemas por importancia global.
    """
    nombres = datos["vec"].get_feature_names_out()
    importancias = np.abs(modelo.coef_).mean(axis=0)
    orden = np.argsort(importancias)[::-1][:20]
    top = {str(nombres[i]): float(importancias[i]) for i in orden}
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(
        [str(nombres[i]) for i in orden][::-1],
        importancias[orden][::-1],
        color="#2F75B5",
    )
    ax.set_title("Top-20 lemas por importancia (LinearSVC, |coef| medio)")
    ax.set_xlabel("Importancia media")
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)
    return top


def ejecutar_baseline_v1(df: pd.DataFrame) -> None:
    """Ejecuta el pipeline completo del baseline clasico M4 (v1).

    Args:
        df: DataFrame ya cargado con lemas, etiquetas y particiones.
    """
    t0 = time.time()
    LOGGER.info("inicio baseline M4")

    resultados: dict = {"fecha": pd.Timestamp.now().isoformat(), "config": {k: v for k, v in CONFIG.items() if k != "ruta_log"}}
    mejor = {"macro_f1": 0.0}

    for escenario, columna in [("completo", "lemas_norm"), ("enmascarado", "lemas_mask")]:
        datos = vectorizar(df, columna)
        # LinearSVC.
        svc = entrenar_linearsvc(datos)
        m_svc = evaluar(svc, datos)
        LOGGER.info("[%s] LinearSVC macro-F1 test=%.4f acc=%.4f", escenario, m_svc["macro_f1"], m_svc["accuracy"])
        # LogisticRegression + Optuna (sustituye LightGBM: ver docstring).
        logreg = entrenar_logreg_optuna(datos)
        m_lgr = evaluar(logreg, datos)
        LOGGER.info("[%s] LogReg macro-F1 test=%.4f acc=%.4f", escenario, m_lgr["macro_f1"], m_lgr["accuracy"])

        resultados[escenario] = {
            "linearsvc": {"macro_f1": m_svc["macro_f1"], "accuracy": m_svc["accuracy"]},
            "logreg": {"macro_f1": m_lgr["macro_f1"], "accuracy": m_lgr["accuracy"]},
        }
        # Rastrear el mejor modelo global (para figuras/guardado).
        if m_svc["macro_f1"] > mejor["macro_f1"]:
            mejor = {
                "macro_f1": m_svc["macro_f1"],
                "modelo": svc,
                "datos": datos,
                "y_pred": m_svc["y_pred"],
                "nombre": f"linearsvc_{escenario}",
            }
        if m_lgr["macro_f1"] > mejor["macro_f1"]:
            mejor = {
                "macro_f1": m_lgr["macro_f1"],
                "modelo": logreg,
                "datos": datos,
                "y_pred": m_lgr["y_pred"],
                "nombre": f"logreg_{escenario}",
            }

    # Gap de leakage (ADR-2): diferencia entre escenarios por modelo.
    resultados["gap_leakage"] = {
        "linearsvc": resultados["completo"]["linearsvc"]["macro_f1"]
        - resultados["enmascarado"]["linearsvc"]["macro_f1"],
        "logreg": resultados["completo"]["logreg"]["macro_f1"]
        - resultados["enmascarado"]["logreg"]["macro_f1"],
    }
    LOGGER.info("gap leakage: %s", resultados["gap_leakage"])

    # Figuras y guardado del mejor.
    datos_mejor = mejor["datos"]
    figura_confusion(datos_mejor["y_test"], mejor["y_pred"], CONFIG["ruta_fig_confusion"])
    # Importancias para el mejor modelo SEA CUAL SEA (ambos son lineales:
    # LinearSVC y LogisticRegression exponen .coef_).
    resultados["shap_top20"] = shap_lineal(mejor["modelo"], datos_mejor, CONFIG["ruta_fig_shap"])
    resultados["nota_shap"] = (
        "Importancias |coef| medio del mejor modelo lineal (equivalente SHAP "
        "lineal); generado por el propio script (trazable)."
    )
    Path(CONFIG["dir_modelo"]).mkdir(parents=True, exist_ok=True)
    joblib.dump(mejor["modelo"], f"{CONFIG['dir_modelo']}/{mejor['nombre']}.joblib")
    joblib.dump(datos_mejor["vec"], f"{CONFIG['dir_modelo']}/tfidf_{mejor['nombre'].split('_', 1)[1]}.joblib")

    resultados["mejor_modelo"] = {"nombre": mejor["nombre"], "macro_f1": mejor["macro_f1"]}
    resultados["tiempo_total_seg"] = time.time() - t0
    Path(CONFIG["ruta_metricas"]).parent.mkdir(parents=True, exist_ok=True)
    with open(CONFIG["ruta_metricas"], "w", encoding="utf-8") as f:
        json.dump(resultados, f, ensure_ascii=False, indent=2, default=str)
    LOGGER.info("fin en %.1f s; mejor=%s macro-F1=%.4f", resultados["tiempo_total_seg"], mejor["nombre"], mejor["macro_f1"])


# ===== SECCION 7: ciclo v2 - Fase U (jerarquico + multietiqueta + calibracion) =====

# Macro-grupos de las 16 clases segun la estructura de la Constitucion
# (agrupacion aprobada en el plan v2, Fase U). La union debe ser exactamente
# el conjunto de 16 clases; validar_macro_grupos lo verifica en runtime.
MACRO_GRUPOS = {
    "derechos": [
        "derechos_fundamentales",
        "derechos_sociales_economicos_culturales",
        "derechos_colectivos_ambiente",
        "proteccion_aplicacion_derechos",
    ],
    "principios": ["principios_fundamentales"],
    "ramas_poder_estado": [
        "rama_legislativa",
        "rama_ejecutiva",
        "rama_judicial",
        "organizacion_estado",
        "organismos_control",
    ],
    "territorio_y_participacion": [
        "organizacion_territorial",
        "habitantes_territorio",
        "participacion_democratica",
        "elecciones",
    ],
    "regimen_economico": ["regimen_economico"],
    "reforma_constitucion": ["reforma_constitucion"],
}

# Rejillas de la optimizacion por coordinate ascent (siempre en validacion).
GRID_UMBRALES = np.round(np.arange(0.05, 0.96, 0.025), 4)
GRID_SESGOS = np.round(np.arange(-2.0, 2.01, 0.05), 4)
MAX_RONDAS_COORDENADA = 3

# Macro-F1 test del baseline v1 (m4_baseline.json): sanity check de artefactos.
MACRO_F1_TEST_V1 = 0.2870081504940973
# Macro-F1 test de la ganadora C de la Fase U (m4_jerarquico.json): referencia
# de comparacion y adopcion de la Fase W(b).
MACRO_F1_TEST_C_FASE_U = 0.2980095633575616
# Macro-F1 test de BETO chunk-mean de la Fase V (m4_transformer_chunks.json):
# sanity obligatorio de la inferencia sobre la cache de chunks en A5.
MACRO_F1_TEST_BETO_COMPLETO_FASE_V = 0.2433442600823086
MACRO_F1_TEST_BETO_ENMASCARADO_FASE_V = 0.23632401040291953
# Tolerancia de los sanity checks de BETO (banda +-0.001 del plan A5).
TOLERANCIA_SANITY_BETO = 0.001
# JSON y modelos de ciclos previos que A5 NO debe modificar (md5 al final).
RUTAS_INTOCABLES_A5 = (
    "resultados/m4_baseline.json",
    "resultados/m4_jerarquico.json",
    "resultados/m4_eurovoc.json",
    "resultados/m4_transformer.json",
    "resultados/m4_transformer_chunks.json",
)
# Cota de n-gramas de los terminos EuroVoc (el lexico la respeta al construirse).
EUROVOC_MAX_NGRAMA = 6


class ModeloConstanteBinaria:
    """Clasificador binario trivial para clases sin ambas etiquetas en train.

    Caso de borde del smoke: si una clase rara no tiene positivos (o no tiene
    negativos) en el train submuestreado, LogisticRegression no puede
    ajustarse; se usa una probabilidad constante en su lugar.

    Args:
        proba_positiva: Probabilidad constante de la clase positiva.
    """

    def __init__(self, proba_positiva: float) -> None:
        self.proba_positiva = float(proba_positiva)

    def predict_proba(self, X) -> np.ndarray:
        """Devuelve la probabilidad constante para cualquier matriz X.

        Args:
            X: Matriz de features (solo se usa su numero de filas).

        Returns:
            Matriz (n, 2) con columnas [1-p, p].
        """
        p = np.full(X.shape[0], self.proba_positiva)
        return np.column_stack([1.0 - p, p])


def construir_clase_a_macro(macro_grupos: dict[str, list[str]]) -> dict[str, str]:
    """Invierte un mapa macro-grupo -> clases en clase -> macro-grupo.

    Args:
        macro_grupos: Dict macro-grupo -> lista de clases.

    Returns:
        Dict clase -> macro-grupo.
    """
    return {clase: grupo for grupo, clases in macro_grupos.items() for clase in clases}


def validar_macro_grupos(clases: list[str], macro_grupos: dict[str, list[str]]) -> None:
    """Valida que los macro-grupos cubran exactamente las clases del dataset.

    Args:
        clases: Clases canonicas observadas en los datos.
        macro_grupos: Dict macro-grupo -> lista de clases a validar (A o D).

    Raises:
        ValueError: Si hay clases repetidas o la cobertura no es exacta.
    """
    cubiertas = [clase for grupo in macro_grupos.values() for clase in grupo]
    if len(cubiertas) != len(set(cubiertas)):
        raise ValueError("hay clases repetidas en los macro-grupos")
    if set(cubiertas) != set(clases):
        raise ValueError(
            f"macro-grupos no cubren las clases: faltan={sorted(set(clases) - set(cubiertas))} "
            f"sobran={sorted(set(cubiertas) - set(clases))}"
        )


def cargar_hiperparametros_v1(ruta_modelo: str) -> dict:
    """Lee los hiperparametros del LogReg v1 desde el joblib (get_params).

    Evita re-ejecutar Optuna: el C optimo y demas parametros v1 se reutilizan
    tal cual para que las variantes de la Fase U sean comparables con el
    baseline. Se conserva una lista blanca de claves para no arrastrar
    parametros deprecados del objeto guardado. Nota sklearn 1.8:
    penalty='l2' esta deprecado y es el comportamiento por defecto (equivalente
    a l1_ratio=0), por lo que se omite para evitar FutureWarning sin cambiar la
    regularizacion.

    Args:
        ruta_modelo: Ruta del joblib logreg_enmascarado de v1.

    Returns:
        Dict de parametros compatible con LogisticRegression.
    """
    modelo = joblib.load(ruta_modelo)
    params = modelo.get_params()
    claves = ["C", "solver", "max_iter", "tol", "fit_intercept", "random_state"]
    params_uso = {clave: params[clave] for clave in claves}
    if params.get("penalty") not in (None, "l2", "deprecated"):
        params_uso["penalty"] = params["penalty"]
    return params_uso


def entrenar_logreg_base_train(X_train, y_train, params_v1: dict, nombre: str) -> LogisticRegression:
    """Entrena un LogReg con los hiperparametros v1 SOLO con train.

    El joblib v1 se reentreno en train+val en su corrida final (ver
    entrenar_logreg_optuna), por lo que su macro-F1 de validacion es 1.0
    (memoriza val) y no sirve para calibrar sin leakage. Para que la
    calibracion de la variante C sea valida se reentrena la misma
    configuracion (features y hiperparametros v1) usando solo train.

    Args:
        X_train: Matriz de features de train.
        y_train: Etiquetas finas de train.
        params_v1: Hiperparametros LogReg v1.
        nombre: Nombre de la variante para el log.

    Returns:
        LogisticRegression ajustado en train.
    """
    modelo = LogisticRegression(**params_v1)
    modelo.fit(X_train, y_train)
    LOGGER.info(
        "%s: base entrenada en train (%d docs, %d iters)",
        nombre,
        X_train.shape[0],
        int(modelo.n_iter_.max()),
    )
    return modelo


def submuestrear_estratificado(df: pd.DataFrame, limite: int, semilla: int) -> pd.DataFrame:
    """Submuestra estratificada por clase para el smoke (splits canonicos).

    Reparte el limite de forma proporcional al tamano de cada clase con un
    minimo de 2 documentos por clase (si existen) y conserva la particion
    canonica de cada documento: NO se rehacen splits. El objetivo es que las
    variantes jerarquica y multietiqueta tengan senal en train/val/test.

    Args:
        df: DataFrame con area_primaria y particion.
        limite: Numero maximo de documentos de la submuestra.
        semilla: Semilla del muestreo.

    Returns:
        DataFrame submuestreado con las particiones originales.
    """
    n_clases = df["area_primaria"].nunique()
    if limite >= len(df):
        return df
    if limite < 2 * n_clases:
        return df.sample(n=limite, random_state=semilla).reset_index(drop=True)
    partes = []
    for _, grupo in df.groupby("area_primaria", sort=True):
        cuota = min(len(grupo), max(2, round(limite * len(grupo) / len(df))))
        partes.append(grupo.sample(n=cuota, random_state=semilla))
    sub = pd.concat(partes)
    while len(sub) > limite:
        mayor = sub["area_primaria"].value_counts().idxmax()
        idx = sub[sub["area_primaria"] == mayor].sample(n=1, random_state=semilla).index
        sub = sub.drop(idx)
    LOGGER.info("smoke: submuestra %d -> %d docs", len(df), len(sub))
    return sub.reset_index(drop=True)


def cargar_targets_multietiqueta(df: pd.DataFrame, clases: list[str]) -> np.ndarray:
    """Binariza los bloques con n>=1 de top3_json (targets multietiqueta).

    Solo se consideran las 16 clases canonicas del clasificador; los bloques
    fuera de ese espacio (p.ej. 'deberes') se ignoran y se cuentan en el log
    para trazabilidad. Los targets provienen de citas; las features del
    modelo son exclusivamente el texto enmascarado de lemas (sin leakage).

    Args:
        df: DataFrame con la columna top3_json.
        clases: Clases canonicas en orden fijo.

    Returns:
        Matriz binaria uint8 (n_docs, n_clases).
    """
    indice = {clase: j for j, clase in enumerate(clases)}
    Y = np.zeros((len(df), len(clases)), dtype=np.uint8)
    ignorados = 0
    for i, texto in enumerate(df["top3_json"].values):
        for item in json.loads(texto):
            if item["n"] >= 1:
                j = indice.get(item["bloque"])
                if j is None:
                    ignorados += 1
                else:
                    Y[i, j] = 1
    LOGGER.info("multietiqueta: %d etiquetas fuera del espacio de 16 clases", ignorados)
    return Y


def _macro_f1(y_true, y_pred) -> float:
    """Macro-F1 con zero_division=0 (protocolo v1, comparable).

    Args:
        y_true: Etiquetas reales.
        y_pred: Etiquetas predichas.

    Returns:
        Macro-F1 flotante.
    """
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def evaluar_single_label(y_true, y_pred, clases: list[str]) -> dict:
    """Metricas single-label comparables con el baseline v1.

    `macro_f1` replica el protocolo v1 (f1_score sin `labels`, es decir,
    sobre la union de etiquetas observadas en y_true/y_pred). `macro_f1_espacio_fijo`
    calcula ademas el macro-F1 sobre el espacio fijo de clases recibido.

    Args:
        y_true: Etiquetas reales.
        y_pred: Etiquetas predichas.
        clases: Clases del espacio fijo de evaluacion.

    Returns:
        Dict con macro_f1, macro_f1_espacio_fijo y accuracy.
    """
    return {
        "macro_f1": _macro_f1(y_true, y_pred),
        "macro_f1_espacio_fijo": float(
            f1_score(y_true, y_pred, average="macro", labels=clases, zero_division=0)
        ),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }


def cargar_referencia_c() -> dict:
    """Lee la C ganadora de la Fase U congelada en m4_jerarquico.json.

    Si el archivo no existe se usa la constante documentada del test (0.2980)
    y val None, para no bloquear una corrida en otro checkout.

    Returns:
        Dict con fecha, val y test de la C de referencia.
    """
    ruta = Path(CONFIG["ruta_json_jerarquico_ref"])
    if not ruta.exists():
        LOGGER.warning("sin %s: referencia C usa solo la constante de test", ruta)
        return {"fecha": None, "val": None, "test": MACRO_F1_TEST_C_FASE_U}
    with open(ruta, encoding="utf-8") as archivo:
        referencia = json.load(archivo)
    variante_c = referencia["variante_c_calibracion"]
    return {
        "fecha": referencia.get("fecha"),
        "val": float(variante_c["val_con_sesgos"]["macro_f1"]),
        "test": float(variante_c["test_con_sesgos"]["macro_f1"]),
    }


def cargar_mapeo_eurovoc() -> dict:
    """Carga el mapeo EuroVoc (clases -> categorias + macro-grupos de la D).

    Returns:
        Contenido de config/eurovoc_bloques.json.

    Raises:
        FileNotFoundError: Si falta el archivo (ejecutar 15_eurovoc_lexico.py).
    """
    ruta = Path(CONFIG["ruta_eurovoc_bloques"])
    if not ruta.exists():
        raise FileNotFoundError(
            f"falta {ruta}; ejecutar scripts/15_eurovoc_lexico.py antes de la Fase W(b)"
        )
    with open(ruta, encoding="utf-8") as archivo:
        return json.load(archivo)


def cargar_terminos_eurovoc() -> dict | None:
    """Carga el lexico lematizado por clase de la variante E.

    La variante E es opcional: si el archivo no existe se omite con aviso y la
    corrida sigue con A/B/C/D (la factibilidad del lexico ya quedo evaluada en
    config/eurovoc_bloques.json).

    Returns:
        Contenido de config/eurovoc_terminos_es.json o None si no existe.
    """
    ruta = Path(CONFIG["ruta_eurovoc_terminos"])
    if not ruta.exists():
        LOGGER.warning("sin %s: variante E se omite", ruta)
        return None
    with open(ruta, encoding="utf-8") as archivo:
        return json.load(archivo)


def construir_vectorizador_eurovoc(
    terminos_por_clase: dict,
    clases: list[str],
) -> tuple[CountVectorizer, np.ndarray, dict]:
    """Prepara el vectorizador de n-gramas y la matriz termino -> clase.

    El vocabulario es FIJO (lexico externo construido por 15_eurovoc_lexico.py);
    CountVectorizer solo cuenta ocurrencias de esos n-gramas y no aprende nada
    dependiente del corpus, por lo que no hay leakage. La matriz de agregacion
    suma los terminos compartidos entre clases.

    Args:
        terminos_por_clase: Dict clase -> {"terminos": [...]} del lexico.
        clases: Clases canonicas en orden de columnas.

    Returns:
        Tupla (vectorizador, matriz binaria termino x clase, meta con
        n_terminos y max_tokens).

    Raises:
        ValueError: Si el lexico esta vacio.
    """
    vocab = sorted(
        {termino for clase in clases for termino in terminos_por_clase[clase]["terminos"]}
    )
    if not vocab:
        raise ValueError("lexico EuroVoc vacio: no se puede construir la variante E")
    max_tokens = min(max(len(termino.split()) for termino in vocab), EUROVOC_MAX_NGRAMA)
    vectorizador = CountVectorizer(
        vocabulary=vocab,
        tokenizer=str.split,
        token_pattern=None,
        lowercase=True,
        ngram_range=(1, max_tokens),
        dtype=np.float32,
    )
    indice = {termino: j for j, termino in enumerate(vocab)}
    matriz_clase = np.zeros((len(vocab), len(clases)), dtype=np.float32)
    for j, clase in enumerate(clases):
        for termino in terminos_por_clase[clase]["terminos"]:
            fila = indice.get(termino)
            if fila is not None:
                matriz_clase[fila, j] = 1.0
    meta = {
        "n_terminos": len(vocab),
        "max_tokens": max_tokens,
        "n_columnas": len(clases),
        "clases": list(clases),
    }
    return vectorizador, matriz_clase, meta


def convertir_features_eurovoc(
    vectorizador: CountVectorizer,
    matriz_clase: np.ndarray,
    textos,
) -> np.ndarray:
    """Convierte textos en features normalizadas de conteo EuroVoc por clase.

    Normalizacion: por documento, la proporcion de aciertos atribuida a cada
    clase (suma 1 si hay al menos un acierto; fila de ceros si no hay ninguno).
    Esto mantiene la escala comparable al TF-IDF L2-normalizado v1.

    Args:
        vectorizador: CountVectorizer de vocabulario fijo ya ajustado.
        matriz_clase: Matriz binaria termino x clase.
        textos: Secuencia de textos de lemas enmascarados.

    Returns:
        Matriz densa (n_docs, n_clases) float32.
    """
    conteos = vectorizador.transform(textos)
    aciertos = np.asarray(conteos @ matriz_clase, dtype=np.float32)
    totales = aciertos.sum(axis=1, keepdims=True)
    return aciertos / np.maximum(totales, 1.0)


def entrenar_cascada(
    X_train,
    y_train,
    params_v1: dict,
    macro_grupos: dict[str, list[str]],
    nombre: str,
) -> dict:
    """Entrena una cascada jerarquica: clasificador grueso + finos por grupo.

    El clasificador grueso se ajusta con los macro-grupos recibidos. Cada
    clasificador fino se ajusta SOLO con los documentos de train de su
    macro-grupo. Los grupos con una sola clase (o una sola presente en train,
    caso borde del smoke) no requieren fino: se resuelven con un fallback
    determinista. La variante A usa los 6 grupos ad-hoc de la Fase U y la
    variante D los 8 grupos derivados de EuroVoc; el codigo es el mismo.

    Args:
        X_train: Matriz de features de train.
        y_train: Etiquetas finas de train.
        params_v1: Hiperparametros LogReg v1 (comparabilidad).
        macro_grupos: Dict macro-grupo -> clases que lo componen.
        nombre: Nombre de la variante para el log.

    Returns:
        Dict con coarse, finos, fallbacks, grupos y clase_a_macro.
    """
    clase_a_macro = construir_clase_a_macro(macro_grupos)
    macro_train = np.array([clase_a_macro[y] for y in y_train])
    coarse = LogisticRegression(**params_v1)
    coarse.fit(X_train, macro_train)
    LOGGER.info("%s: grueso entrenado (%d macro-grupos)", nombre, len(coarse.classes_))
    finos, fallbacks = {}, {}
    for grupo, clases_grupo in macro_grupos.items():
        mascara = np.isin(y_train, clases_grupo)
        presentes = sorted(set(y_train[mascara]))
        if len(presentes) <= 1:
            fallbacks[grupo] = presentes[0] if presentes else clases_grupo[0]
            continue
        modelo = LogisticRegression(**params_v1)
        modelo.fit(X_train[mascara], y_train[mascara])
        finos[grupo] = modelo
        LOGGER.info(
            "%s: fino %s (%d docs train, %d clases)", nombre, grupo, int(mascara.sum()), len(presentes)
        )
    return {
        "coarse": coarse,
        "finos": finos,
        "fallbacks": fallbacks,
        "grupos": macro_grupos,
        "clase_a_macro": clase_a_macro,
    }


def predecir_cascada(modelo: dict, X) -> tuple[np.ndarray, np.ndarray]:
    """Predice la clase fina y el macro-grupo con una cascada jerarquica.

    Args:
        modelo: Dict devuelto por entrenar_cascada.
        X: Matriz de features.

    Returns:
        Tupla (y_pred_fina, grupos_pred) alineada con las filas de X.
    """
    indice_grupos = modelo["coarse"].predict_proba(X).argmax(axis=1)
    grupos_pred = modelo["coarse"].classes_[indice_grupos]
    y_pred = np.empty(X.shape[0], dtype=object)
    for grupo in modelo["grupos"]:
        mascara = grupos_pred == grupo
        if not mascara.any():
            continue
        if grupo in modelo["finos"]:
            y_pred[mascara] = modelo["finos"][grupo].predict(X[mascara])
        else:
            y_pred[mascara] = modelo["fallbacks"][grupo]
    return y_pred, grupos_pred


def entrenar_variante_b(X_train, Y_train, params_v1: dict) -> list:
    """Entrena 16 clasificadores binarios OneVsRest para multietiqueta.

    Args:
        X_train: Matriz de features de train.
        Y_train: Matriz binaria (n, 16) de targets multietiqueta.
        params_v1: Hiperparametros LogReg v1 (comparabilidad).

    Returns:
        Lista de 16 clasificadores binarios ajustados.
    """
    modelos = []
    for j in range(Y_train.shape[1]):
        y_j = Y_train[:, j]
        if len(np.unique(y_j)) < 2:
            modelos.append(ModeloConstanteBinaria(float(y_j.mean())))
            continue
        modelo = LogisticRegression(**params_v1)
        modelo.fit(X_train, y_j)
        modelos.append(modelo)
        if (j + 1) % 4 == 0:
            LOGGER.info("variante B: %d/%d binarios entrenados", j + 1, Y_train.shape[1])
    return modelos


def predecir_proba_multietiqueta(modelos: list, X) -> np.ndarray:
    """Concatena la probabilidad positiva de los 16 clasificadores binarios.

    Args:
        modelos: Lista de clasificadores binarios.
        X: Matriz de features.

    Returns:
        Matriz (n, 16) de probabilidades de la clase positiva.
    """
    return np.column_stack([modelo.predict_proba(X)[:, 1] for modelo in modelos])


def optimizar_umbrales_multietiqueta(P_val, Y_val, max_rondas: int) -> tuple[np.ndarray, float]:
    """Optimiza un umbral por clase en validacion (coordinate ascent).

    Metrica objetivo: macro-F1 multietiqueta sobre las 16 columnas binarias.
    Solo usa validacion; test nunca participa.

    Args:
        P_val: Probabilidades (n, 16) en validacion.
        Y_val: Etiquetas binarias (n, 16) en validacion.
        max_rondas: Maximo de pasadas completas sobre las clases.

    Returns:
        Tupla (umbrales, mejor_macro_f1_val).
    """
    umbrales = np.full(P_val.shape[1], 0.5)
    mejor = float(f1_score(Y_val, (P_val >= umbrales).astype(np.uint8), average="macro", zero_division=0))
    for ronda in range(max_rondas):
        mejoro = False
        for j in range(P_val.shape[1]):
            mejor_j, umbral_j = mejor, umbrales[j]
            for t in GRID_UMBRALES:
                candidato = umbrales.copy()
                candidato[j] = t
                score = float(
                    f1_score(Y_val, (P_val >= candidato).astype(np.uint8), average="macro", zero_division=0)
                )
                if score > mejor_j + 1e-12:
                    mejor_j, umbral_j = score, t
            if mejor_j > mejor + 1e-12:
                mejor, umbrales[j], mejoro = mejor_j, umbral_j, True
        LOGGER.info("variante B: ronda %d umbrales -> macro-F1 val=%.4f", ronda + 1, mejor)
        if not mejoro:
            break
    return umbrales, mejor


def conversion_single_label(P, umbrales: np.ndarray, clases: list[str]) -> np.ndarray:
    """Convierte probabilidades multietiqueta en una etiqueta unica.

    Decodificacion: argmax(P - umbral) por documento, es decir, la clase con
    mayor margen sobre su propio umbral. Si todas las clases quedan bajo su
    umbral, el argmax del margen sigue devolviendo una clase (documentado)
    para no introducir una etiqueta "ninguna" fuera del espacio de clases.

    Args:
        P: Probabilidades (n, 16).
        umbrales: Umbral por clase, en el orden de las columnas.
        clases: Nombres de clase en el orden de las columnas.

    Returns:
        Vector de etiquetas unicas.
    """
    return np.asarray(clases)[np.argmax(P - umbrales, axis=1)]


def evaluar_variante_b(y_primaria, Y_true, P, umbrales: np.ndarray, clases: list[str]) -> dict:
    """Metricas de la variante B: multietiqueta + conversion single-label.

    Args:
        y_primaria: Etiqueta primaria real (para la conversion single-label).
        Y_true: Matriz binaria real (n, 16).
        P: Probabilidades (n, 16).
        umbrales: Umbral por clase.
        clases: Clases canonicas.

    Returns:
        Dict con macro/micro-F1 multietiqueta y la conversion single-label.
    """
    Y_pred = (P >= umbrales).astype(np.uint8)
    single = conversion_single_label(P, umbrales, clases)
    return {
        "macro_f1_multietiqueta": float(f1_score(Y_true, Y_pred, average="macro", zero_division=0)),
        "micro_f1_multietiqueta": float(f1_score(Y_true, Y_pred, average="micro", zero_division=0)),
        "single_label_convertida": evaluar_single_label(y_primaria, single, clases),
    }


def optimizar_sesgos_logits(logits_val, y_val, clases: list[str], max_rondas: int) -> tuple[np.ndarray, dict]:
    """Optimiza un sesgo aditivo por clase en validacion (coordinate ascent).

    Decision final: argmax(logits + sesgo). La metrica (macro-F1 single-label)
    y los sesgos se calculan SIEMPRE en validacion; test queda intacto.

    Args:
        logits_val: Matriz (n, 16) de decision_function en validacion.
        y_val: Etiquetas finas de validacion.
        clases: Clases en el orden de las columnas de logits_val.
        max_rondas: Maximo de pasadas completas sobre las clases.

    Returns:
        Tupla (sesgos, detalle) con macro-F1 val antes y despues.
    """
    clases_arr = np.asarray(clases)
    sesgos = np.zeros(logits_val.shape[1])
    antes = _macro_f1(y_val, clases_arr[np.argmax(logits_val, axis=1)])
    mejor = antes
    for ronda in range(max_rondas):
        mejoro = False
        for j in range(logits_val.shape[1]):
            mejor_j, sesgo_j = mejor, sesgos[j]
            for delta in GRID_SESGOS:
                candidato = sesgos.copy()
                candidato[j] = delta
                score = _macro_f1(y_val, clases_arr[np.argmax(logits_val + candidato, axis=1)])
                if score > mejor_j + 1e-12:
                    mejor_j, sesgo_j = score, delta
            if mejor_j > mejor + 1e-12:
                mejor, sesgos[j], mejoro = mejor_j, sesgo_j, True
        LOGGER.info("variante C: ronda %d sesgos -> macro-F1 val=%.4f", ronda + 1, mejor)
        if not mejoro:
            break
    return sesgos, {"macro_f1_val_antes": antes, "macro_f1_val_despues": mejor}


def figura_variantes_fase_u(valores_val: dict, valores_test: dict, ruta: str) -> None:
    """Genera la figura comparativa de macro-F1 val/test de la Fase U.

    Args:
        valores_val: Dict variante -> macro-F1 de validacion.
        valores_test: Dict variante -> macro-F1 de test.
        ruta: Ruta del PNG a guardar.
    """
    nombres = list(valores_val)
    x = np.arange(len(nombres))
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - 0.2, [valores_val[n] for n in nombres], width=0.4, label="val", color="#2F75B5")
    ax.bar(x + 0.2, [valores_test[n] for n in nombres], width=0.4, label="test", color="#ED7D31")
    ax.axhline(0.32, color="#C00000", linestyle="--", linewidth=1.2, label="meta test 0.32")
    ax.set_xticks(x)
    ax.set_xticklabels(nombres, rotation=20, ha="right")
    ax.set_ylabel("macro-F1")
    ax.set_title("Fase W(b): macro-F1 val/test por variante")
    ax.legend()
    fig.tight_layout()
    Path(ruta).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


def md5_archivo(ruta: str) -> str:
    """Calcula el md5 de un archivo por bloques (sin cargarlo en RAM).

    Args:
        ruta: Ruta del archivo.

    Returns:
        Digest md5 hexadecimal.
    """
    hash_md5 = hashlib.md5()
    with open(ruta, "rb") as archivo:
        for bloque in iter(lambda: archivo.read(1024 * 1024), b""):
            hash_md5.update(bloque)
    return hash_md5.hexdigest()


def ejecutar_fase_u(df: pd.DataFrame, ruta_json: str, limite: int | None) -> dict:
    """Ejecuta las variantes A/B/C/D/E y guarda su JSON.

    Protocolo: modelos con train; umbrales, sesgos y seleccion con
    validacion; test solo al final (y sin participar en ninguna decision).
    No toca m4_baseline.json, m4_jerarquico.json, ni los splits, ni los
    modelos v1. D usa los macro-grupos EuroVoc (reestructuracion del target)
    y E concatena el lexico EuroVoc al TF-IDF v1 (features solo del texto).

    Args:
        df: DataFrame con lemas, etiquetas, top3_json y particiones.
        ruta_json: Ruta de salida (resultados/ o /tmp para el smoke).
        limite: Limite de documentos del smoke (None = run completo).

    Returns:
        Dict de resultados serializado a JSON.

    Raises:
        FileNotFoundError: Si faltan los artefactos v1 o el mapeo EuroVoc.
    """
    t0 = time.time()
    smoke = limite is not None
    clases = sorted(df["area_primaria"].unique())
    validar_macro_grupos(clases, MACRO_GRUPOS)
    mapeo_eurovoc = cargar_mapeo_eurovoc()
    macro_grupos_eurovoc = {
        grupo: datos["clases"] for grupo, datos in mapeo_eurovoc["macro_grupos"].items()
    }
    validar_macro_grupos(clases, macro_grupos_eurovoc)
    clase_a_macro_a = construir_clase_a_macro(MACRO_GRUPOS)
    clase_a_macro_d = construir_clase_a_macro(macro_grupos_eurovoc)
    terminos_eurovoc = cargar_terminos_eurovoc()

    ruta_vec = f"{CONFIG['dir_modelo_v1']}/tfidf_enmascarado.joblib"
    ruta_mod = f"{CONFIG['dir_modelo_v1']}/logreg_enmascarado.joblib"
    if not Path(ruta_vec).exists() or not Path(ruta_mod).exists():
        raise FileNotFoundError(f"faltan artefactos v1: {ruta_vec} / {ruta_mod}")
    vec = joblib.load(ruta_vec)
    modelo_v1 = joblib.load(ruta_mod)
    params_v1 = cargar_hiperparametros_v1(ruta_mod)
    if sorted(modelo_v1.classes_) != clases:
        raise ValueError("las clases del modelo v1 no coinciden con las del dataset")
    LOGGER.info("fase U: docs=%d clases=%d smoke=%s limite=%s", len(df), len(clases), smoke, limite)
    LOGGER.info("hiperparametros v1 reutilizados: %s", params_v1)

    tr = df[df["particion"] == "train"]
    va = df[df["particion"] == "val"]
    te = df[df["particion"] == "test"]
    X_train = vec.transform(tr["lemas_mask"])
    X_val = vec.transform(va["lemas_mask"])
    X_test = vec.transform(te["lemas_mask"])
    y_train = tr["area_primaria"].values
    y_val = va["area_primaria"].values
    y_test = te["area_primaria"].values
    LOGGER.info("features v1: train=%s val=%s test=%s", X_train.shape, X_val.shape, X_test.shape)

    resultados: dict = {
        "fecha": pd.Timestamp.now().isoformat(),
        "fase": "ciclo v2 - Fase W(b) EuroVoc-es (variantes A/B/C/D/E)",
        "modo": f"smoke_limite_{limite}" if smoke else "completo",
        "config": {k: v for k, v in CONFIG.items() if k != "ruta_log"},
        "hiperparametros_v1_usados": params_v1,
        "macro_grupos": MACRO_GRUPOS,
        "macro_grupos_eurovoc": macro_grupos_eurovoc,
        "mapeo_eurovoc": mapeo_eurovoc,
        "nota_targets_multietiqueta": (
            "targets = bloques con n>=1 de top3_json binarizados sobre las 16 clases "
            "canonicas; el bloque 'deberes' (fuera del espacio de clases) se ignora "
            "y se cuenta en el log. Features SOLO del texto enmascarado de lemas."
        ),
    }

    # Referencia v1: recalculo del joblib guardado (sanity check de test).
    # OJO: v1 reentreno su modelo final en train+val, por lo que su val esta
    # contaminado (macro-F1 1.0) y NO se usa para calibrar ni seleccionar.
    pred_v1_val = modelo_v1.predict(X_val)
    pred_v1_test = modelo_v1.predict(X_test)
    resultados["referencia_v1_logreg_enmascarado"] = {
        "val": evaluar_single_label(y_val, pred_v1_val, clases),
        "test": evaluar_single_label(y_test, pred_v1_test, clases),
        "nota": (
            "recalculo del joblib v1 (modelos/m4_baseline/logreg_enmascarado). "
            "v1 reentreno en train+val: su val=1.0 esta contaminado y no se usa "
            "para calibrar/seleccionar; solo su test es comparable (esperado "
            "0.2870081505 en el run completo)."
        ),
    }
    if not smoke:
        delta_v1 = abs(
            resultados["referencia_v1_logreg_enmascarado"]["test"]["macro_f1"] - MACRO_F1_TEST_V1
        )
        if delta_v1 > 1e-6:
            LOGGER.warning(
                "sanity check v1: macro-F1 test=%.6f difiere de m4_baseline.json (%.6f)",
                resultados["referencia_v1_logreg_enmascarado"]["test"]["macro_f1"],
                MACRO_F1_TEST_V1,
            )
        else:
            LOGGER.info(
                "sanity check v1 OK: macro-F1 test=%.6f reproduce m4_baseline.json",
                resultados["referencia_v1_logreg_enmascarado"]["test"]["macro_f1"],
            )

    # --- Variante A: jerarquica (cascada) con macro-grupos ad-hoc de la CP ---
    t_a = time.time()
    modelo_a = entrenar_cascada(X_train, y_train, params_v1, MACRO_GRUPOS, "variante A")
    pred_a_val, grupos_a_val = predecir_cascada(modelo_a, X_val)
    pred_a_test, grupos_a_test = predecir_cascada(modelo_a, X_test)
    y_macro_val = np.array([clase_a_macro_a[y] for y in y_val])
    y_macro_test = np.array([clase_a_macro_a[y] for y in y_test])
    grupos_orden = sorted(MACRO_GRUPOS)
    resultados["variante_a_jerarquica"] = {
        "val": evaluar_single_label(y_val, pred_a_val, clases),
        "test": evaluar_single_label(y_test, pred_a_test, clases),
        "grueso_val": evaluar_single_label(y_macro_val, grupos_a_val, grupos_orden),
        "grueso_test": evaluar_single_label(y_macro_test, grupos_a_test, grupos_orden),
        "tiempo_seg": time.time() - t_a,
    }
    LOGGER.info(
        "variante A: macro-F1 val=%.4f test=%.4f (grueso val=%.4f test=%.4f)",
        resultados["variante_a_jerarquica"]["val"]["macro_f1"],
        resultados["variante_a_jerarquica"]["test"]["macro_f1"],
        resultados["variante_a_jerarquica"]["grueso_val"]["macro_f1"],
        resultados["variante_a_jerarquica"]["grueso_test"]["macro_f1"],
    )

    # --- Variante B: multietiqueta OneVsRest + umbrales en validacion ---
    t_b = time.time()
    Y_train = cargar_targets_multietiqueta(tr, clases)
    Y_val = cargar_targets_multietiqueta(va, clases)
    Y_test = cargar_targets_multietiqueta(te, clases)
    modelos_b = entrenar_variante_b(X_train, Y_train, params_v1)
    P_val = predecir_proba_multietiqueta(modelos_b, X_val)
    P_test = predecir_proba_multietiqueta(modelos_b, X_test)
    umbrales, macro_val_b_multi = optimizar_umbrales_multietiqueta(P_val, Y_val, MAX_RONDAS_COORDENADA)
    iteraciones_b = {
        clase: int(modelo.n_iter_[0]) if hasattr(modelo, "n_iter_") else -1
        for clase, modelo in zip(clases, modelos_b)
    }
    resultados["variante_b_multietiqueta"] = {
        "val": evaluar_variante_b(y_val, Y_val, P_val, umbrales, clases),
        "test": evaluar_variante_b(y_test, Y_test, P_test, umbrales, clases),
        "macro_f1_multietiqueta_val_optimizado": float(macro_val_b_multi),
        "umbrales_por_clase": {clase: float(u) for clase, u in zip(clases, umbrales)},
        "iteraciones_por_clase": iteraciones_b,
        "tiempo_seg": time.time() - t_b,
    }
    LOGGER.info(
        "variante B: macro-F1 multi val=%.4f test=%.4f | single-label test=%.4f",
        resultados["variante_b_multietiqueta"]["val"]["macro_f1_multietiqueta"],
        resultados["variante_b_multietiqueta"]["test"]["macro_f1_multietiqueta"],
        resultados["variante_b_multietiqueta"]["test"]["single_label_convertida"]["macro_f1"],
    )

    # --- Variante C: calibracion del LogReg plano v1 (sesgos en validacion) ---
    t_c = time.time()
    modelo_c = entrenar_logreg_base_train(X_train, y_train, params_v1, "variante C")
    clases_c = np.asarray(modelo_c.classes_)
    logits_val = modelo_c.decision_function(X_val)
    logits_test = modelo_c.decision_function(X_test)
    pred_c_val_sin = modelo_c.predict(X_val)
    pred_c_test_sin = modelo_c.predict(X_test)
    sesgos, detalle_c = optimizar_sesgos_logits(logits_val, y_val, modelo_c.classes_, MAX_RONDAS_COORDENADA)
    pred_c_val = clases_c[np.argmax(logits_val + sesgos, axis=1)]
    pred_c_test = clases_c[np.argmax(logits_test + sesgos, axis=1)]
    resultados["variante_c_calibracion"] = {
        "val_sin_sesgos": evaluar_single_label(y_val, pred_c_val_sin, clases),
        "val_con_sesgos": evaluar_single_label(y_val, pred_c_val, clases),
        "test_sin_sesgos": evaluar_single_label(y_test, pred_c_test_sin, clases),
        "test_con_sesgos": evaluar_single_label(y_test, pred_c_test, clases),
        "sesgos_por_clase": {clase: float(b) for clase, b in zip(modelo_c.classes_, sesgos)},
        "detalle_optimizacion": detalle_c,
        "nota": (
            "base reentrenada SOLO con train usando hiperparametros v1; el joblib v1 "
            "incluye val en su entrenamiento final y contaminaria la calibracion."
        ),
        "tiempo_seg": time.time() - t_c,
    }
    LOGGER.info(
        "variante C: macro-F1 val %.4f -> %.4f | test sin/con sesgos %.4f/%.4f",
        resultados["variante_c_calibracion"]["val_sin_sesgos"]["macro_f1"],
        resultados["variante_c_calibracion"]["val_con_sesgos"]["macro_f1"],
        resultados["variante_c_calibracion"]["test_sin_sesgos"]["macro_f1"],
        resultados["variante_c_calibracion"]["test_con_sesgos"]["macro_f1"],
    )

    # --- Variante D: jerarquica con taxonomia EuroVoc (macro-grupos del mapeo) ---
    t_d = time.time()
    modelo_d = entrenar_cascada(
        X_train, y_train, params_v1, macro_grupos_eurovoc, "variante D"
    )
    pred_d_val, grupos_d_val = predecir_cascada(modelo_d, X_val)
    pred_d_test, grupos_d_test = predecir_cascada(modelo_d, X_test)
    y_macro_d_val = np.array([clase_a_macro_d[y] for y in y_val])
    y_macro_d_test = np.array([clase_a_macro_d[y] for y in y_test])
    grupos_d_orden = sorted(macro_grupos_eurovoc)
    resultados["variante_d_jerarquica_eurovoc"] = {
        "val": evaluar_single_label(y_val, pred_d_val, clases),
        "test": evaluar_single_label(y_test, pred_d_test, clases),
        "grueso_val": evaluar_single_label(y_macro_d_val, grupos_d_val, grupos_d_orden),
        "grueso_test": evaluar_single_label(y_macro_d_test, grupos_d_test, grupos_d_orden),
        "macro_grupos_usados": macro_grupos_eurovoc,
        "nota": (
            "cascada identica a A pero con los macro-grupos derivados del mapeo "
            "EuroVoc (config/eurovoc_bloques.json); es reestructuracion del target, "
            "sin features de entrada y sin leakage."
        ),
        "tiempo_seg": time.time() - t_d,
    }
    LOGGER.info(
        "variante D: macro-F1 val=%.4f test=%.4f (grueso val=%.4f test=%.4f)",
        resultados["variante_d_jerarquica_eurovoc"]["val"]["macro_f1"],
        resultados["variante_d_jerarquica_eurovoc"]["test"]["macro_f1"],
        resultados["variante_d_jerarquica_eurovoc"]["grueso_val"]["macro_f1"],
        resultados["variante_d_jerarquica_eurovoc"]["grueso_test"]["macro_f1"],
    )

    # --- Variante E: features auxiliares por lexico EuroVoc (si hay lexico) ---
    t_e = time.time()
    if terminos_eurovoc is None:
        resultados["variante_e_eurovoc_features"] = {
            "omitida": True,
            "motivo": f"no existe {CONFIG['ruta_eurovoc_terminos']}",
            "tiempo_seg": time.time() - t_e,
        }
        LOGGER.warning("variante E omitida: sin lexico EuroVoc")
    else:
        vec_eurovoc, matriz_clase, meta_e = construir_vectorizador_eurovoc(
            terminos_eurovoc["terminos_por_clase"], clases
        )
        # Vocabulario fijo: fit solo valida el vocabulario del lexico y no
        # aprende estadisticos del corpus (sin leakage).
        vec_eurovoc.fit(tr["lemas_mask"])
        F_train = convertir_features_eurovoc(vec_eurovoc, matriz_clase, tr["lemas_mask"])
        F_val = convertir_features_eurovoc(vec_eurovoc, matriz_clase, va["lemas_mask"])
        F_test = convertir_features_eurovoc(vec_eurovoc, matriz_clase, te["lemas_mask"])
        X_train_e = hstack([X_train, csr_matrix(F_train)]).tocsr()
        X_val_e = hstack([X_val, csr_matrix(F_val)]).tocsr()
        X_test_e = hstack([X_test, csr_matrix(F_test)]).tocsr()
        modelo_e = entrenar_logreg_base_train(X_train_e, y_train, params_v1, "variante E")
        pred_e_val = modelo_e.predict(X_val_e)
        pred_e_test = modelo_e.predict(X_test_e)
        meta_e["cobertura_docs_con_acierto"] = {
            "train": float(np.mean(F_train.sum(axis=1) > 0)),
            "val": float(np.mean(F_val.sum(axis=1) > 0)),
            "test": float(np.mean(F_test.sum(axis=1) > 0)),
        }
        resultados["variante_e_eurovoc_features"] = {
            "val": evaluar_single_label(y_val, pred_e_val, clases),
            "test": evaluar_single_label(y_test, pred_e_test, clases),
            "lexico": meta_e,
            "nota": (
                "TF-IDF v1 enmascarado + 16 columnas de proporcion de aciertos del "
                "lexico EuroVoc de cada clase (hstack); LogReg reentrenado con "
                "hiperparametros v1. La normalizacion por documento (suma 1 cuando "
                "hay aciertos) mantiene la escala comparable al TF-IDF L2 v1. Sin "
                "leakage: el lexico es externo y las features salen solo del texto "
                "enmascarado de lemas."
            ),
            "tiempo_seg": time.time() - t_e,
        }
        LOGGER.info(
            "variante E: macro-F1 val=%.4f test=%.4f | %d terminos, cobertura val=%.3f",
            resultados["variante_e_eurovoc_features"]["val"]["macro_f1"],
            resultados["variante_e_eurovoc_features"]["test"]["macro_f1"],
            meta_e["n_terminos"],
            meta_e["cobertura_docs_con_acierto"]["val"],
        )

    # --- Seleccion por macro-F1 single-label de VALIDACION ---
    val_por_variante = {
        "A_jerarquica": resultados["variante_a_jerarquica"]["val"]["macro_f1"],
        "B_multietiqueta": resultados["variante_b_multietiqueta"]["val"]["single_label_convertida"]["macro_f1"],
        "C_calibracion": resultados["variante_c_calibracion"]["val_con_sesgos"]["macro_f1"],
        "D_jerarquica_eurovoc": resultados["variante_d_jerarquica_eurovoc"]["val"]["macro_f1"],
    }
    test_por_variante = {
        "A_jerarquica": resultados["variante_a_jerarquica"]["test"]["macro_f1"],
        "B_multietiqueta": resultados["variante_b_multietiqueta"]["test"]["single_label_convertida"]["macro_f1"],
        "C_calibracion": resultados["variante_c_calibracion"]["test_con_sesgos"]["macro_f1"],
        "D_jerarquica_eurovoc": resultados["variante_d_jerarquica_eurovoc"]["test"]["macro_f1"],
    }
    if not resultados["variante_e_eurovoc_features"].get("omitida", False):
        val_por_variante["E_eurovoc_features"] = resultados["variante_e_eurovoc_features"]["val"]["macro_f1"]
        test_por_variante["E_eurovoc_features"] = resultados["variante_e_eurovoc_features"]["test"]["macro_f1"]
    ganadora = max(val_por_variante, key=val_por_variante.get)
    resultados["seleccion"] = {
        "ganadora": ganadora,
        "macro_f1_val_por_variante": {k: float(v) for k, v in val_por_variante.items()},
        "macro_f1_test_por_variante": {k: float(v) for k, v in test_por_variante.items()},
        "macro_f1_test_ganadora": float(test_por_variante[ganadora]),
        "meta_macro_f1_test_0_32_cumplida": bool(test_por_variante[ganadora] >= 0.32),
        "nota_seleccion": (
            "Seleccion por macro-F1 single-label de VALIDACION (mismo protocolo v1, "
            "comparable entre variantes). Para B se usa su conversion a etiqueta unica; "
            "su metrica nativa (macro-F1 multietiqueta) se reporta aparte y NO se usa "
            "para seleccionar. Test no participa en ninguna decision."
        ),
    }

    # --- Comparacion contra la C de la Fase U (referencia congelada) ---
    referencia_c = cargar_referencia_c()
    deltas_val = (
        {k: float(v - referencia_c["val"]) for k, v in val_por_variante.items()}
        if referencia_c["val"] is not None
        else None
    )
    resultados["comparacion_vs_c_referencia"] = {
        "referencia": CONFIG["ruta_json_jerarquico_ref"],
        "fecha_referencia": referencia_c.get("fecha"),
        "C_val": referencia_c["val"],
        "C_test": referencia_c["test"],
        "delta_val_por_variante": deltas_val,
        "delta_test_por_variante": {
            k: float(v - referencia_c["test"]) for k, v in test_por_variante.items()
        },
        "nota": (
            "C de referencia = ganadora de la Fase U (m4_jerarquico.json, test 0.2980). "
            "Los deltas se calculan contra ese valor congelado, no contra la C "
            "recalculada en esta corrida (que debe reproducirlo)."
        ),
    }
    resultados["recomendacion_adopcion"] = {
        "regla": (
            "adoptar solo si la variante gana en macro-F1 de VALIDACION y su test no "
            "cae por debajo del test de la C de referencia (0.2980)."
        ),
        "por_variante": {},
    }
    for variante, val_variante in val_por_variante.items():
        test_variante = test_por_variante[variante]
        adoptar = bool(variante == ganadora and test_variante >= referencia_c["test"] - 1e-12)
        if adoptar and variante == "C_calibracion":
            motivo = "C conserva el mejor val/test del run; D/E no la superan."
        elif adoptar:
            motivo = "gana en validacion y no empeora el test de la C de referencia."
        else:
            motivo = "no gana en validacion o su test queda por debajo de la C de referencia."
        resultados["recomendacion_adopcion"]["por_variante"][variante] = {
            "adoptar": adoptar,
            "macro_f1_val": float(val_variante),
            "macro_f1_test": float(test_variante),
            "motivo": motivo,
        }
    resultados["nota_adopcion"] = (
        "La senal EuroVoc (variantes D y E) es experimental: se registra gane o "
        "pierda. Si no gana en validacion y no mejora el test de C, queda como "
        "resultado NEGATIVO documentado (PLAN_v2, Fase W(b)); m4_jerarquico.json y "
        "m4_baseline.json permanecen INTOCABLES."
    )
    LOGGER.info(
        "resumen macro-F1 val/test: %s",
        ", ".join(f"{k}={v:.4f}/{test_por_variante[k]:.4f}" for k, v in val_por_variante.items()),
    )

    # --- Figuras y guardado del artefacto ganador (solo run completo) ---
    if not smoke:
        figura_variantes_fase_u(val_por_variante, test_por_variante, CONFIG["ruta_fig_variantes"])
        artefactos = {
            "A_jerarquica": {
                "coarse": modelo_a["coarse"],
                "finos": modelo_a["finos"],
                "fallbacks": modelo_a["fallbacks"],
                "grupos": modelo_a["grupos"],
            },
            "B_multietiqueta": {"modelos": modelos_b, "umbrales": umbrales},
            "C_calibracion": {"sesgos": sesgos, "modelo_base": modelo_c},
            "D_jerarquica_eurovoc": {
                "coarse": modelo_d["coarse"],
                "finos": modelo_d["finos"],
                "fallbacks": modelo_d["fallbacks"],
                "grupos": modelo_d["grupos"],
                "macro_grupos_eurovoc": mapeo_eurovoc["macro_grupos"],
            },
        }
        if not resultados["variante_e_eurovoc_features"].get("omitida", False):
            artefactos["E_eurovoc_features"] = {
                "modelo_base": modelo_e,
                "ruta_terminos": CONFIG["ruta_eurovoc_terminos"],
                "construccion_features": "vectorizador EuroVoc + TF-IDF v1 (hstack)",
            }
        Path(CONFIG["dir_modelo_fase_u"]).mkdir(parents=True, exist_ok=True)
        ruta_artefacto = f"{CONFIG['dir_modelo_fase_u']}/ganadora_{ganadora}.joblib"
        joblib.dump(
            {
                "variante": ganadora,
                "clases": clases,
                "macro_grupos": MACRO_GRUPOS,
                "macro_grupos_eurovoc": macro_grupos_eurovoc,
                "vectorizador": ruta_vec,
                "artefacto": artefactos[ganadora],
            },
            ruta_artefacto,
        )
        resultados["seleccion"]["ruta_artefacto_ganadora"] = ruta_artefacto

    resultados["md5_artefactos_v1"] = {
        CONFIG["ruta_metricas_v1"]: md5_archivo(CONFIG["ruta_metricas_v1"]),
        f"{CONFIG['dir_modelo_v1']}/logreg_enmascarado.joblib": md5_archivo(ruta_mod),
        f"{CONFIG['dir_modelo_v1']}/tfidf_enmascarado.joblib": md5_archivo(ruta_vec),
    }
    if Path(CONFIG["ruta_json_jerarquico_ref"]).exists():
        resultados["md5_artefactos_v1"][CONFIG["ruta_json_jerarquico_ref"]] = md5_archivo(
            CONFIG["ruta_json_jerarquico_ref"]
        )
    if Path(CONFIG["ruta_eurovoc_terminos"]).exists():
        resultados["md5_config_eurovoc"] = {
            CONFIG["ruta_eurovoc_bloques"]: md5_archivo(CONFIG["ruta_eurovoc_bloques"]),
            CONFIG["ruta_eurovoc_terminos"]: md5_archivo(CONFIG["ruta_eurovoc_terminos"]),
        }
    resultados["tiempo_total_seg"] = time.time() - t0
    Path(ruta_json).parent.mkdir(parents=True, exist_ok=True)
    with open(ruta_json, "w", encoding="utf-8") as f:
        json.dump(resultados, f, ensure_ascii=False, indent=2, default=str)
    LOGGER.info(
        "fase W(b) fin en %.1f s; ganadora=%s macro-F1 test=%.4f meta_0.32=%s",
        resultados["tiempo_total_seg"],
        ganadora,
        test_por_variante[ganadora],
        resultados["seleccion"]["meta_macro_f1_test_0_32_cumplida"],
    )
    return resultados


# ===== SECCION 8: ciclo v3 - Sprint A5 (clasificador C + BETO chunk-mean) =====
# Protocolo (PLAN_v3, Sprint A5; ADR-16): VALIDACION para toda seleccion (w del
# ensamble, sesgos de calibracion); TEST una sola vez al final; bootstrap
# pareado (1000, seed 42) de la diferencia de macro-F1 test contra la C de la
# Fase U. Salida NUEVA resultados/m4_clasificador_c3.json (version "c3-a5");
# ningun JSON/modelo de fases previas se escribe desde aqui.


def _softmax_filas(logits: np.ndarray) -> np.ndarray:
    """Softmax numericamente estable por filas.

    Args:
        logits: Matriz (n, n_clases) de scores.

    Returns:
        Matriz (n, n_clases) de probabilidades que suma 1 por fila.
    """
    z = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(z)
    return exp / exp.sum(axis=1, keepdims=True)


def cargar_artefacto_c() -> dict:
    """Carga y valida la C calibrada de la Fase U (joblib congelado).

    Returns:
        Dict con modelo_base, sesgos, clases, ruta y md5.

    Raises:
        FileNotFoundError: Si falta el artefacto.
        ValueError: Si la variante o la alineacion sesgos/clases no son las
            esperadas.
    """
    ruta = CONFIG["ruta_artefacto_c"]
    if not Path(ruta).exists():
        raise FileNotFoundError(f"falta el artefacto C: {ruta}")
    artefacto = joblib.load(ruta)
    if artefacto.get("variante") != "C_calibracion":
        raise ValueError(f"{ruta} no es C_calibracion: {artefacto.get('variante')}")
    datos = artefacto.get("artefacto") or {}
    if "modelo_base" not in datos or "sesgos" not in datos:
        raise ValueError(f"artefacto C incompleto en {ruta}: claves {sorted(datos)}")
    modelo = datos["modelo_base"]
    sesgos = np.asarray(datos["sesgos"], dtype=float)
    clases = [str(c) for c in artefacto["clases"]]
    if len(sesgos) != len(clases) or [str(c) for c in modelo.classes_] != clases:
        raise ValueError("sesgos/clases del artefacto C no estan alineados")
    LOGGER.info("artefacto C cargado: %d clases", len(clases))
    return {"modelo": modelo, "sesgos": sesgos, "clases": clases, "ruta": ruta, "md5": md5_archivo(ruta)}


def probabilidades_c(artefacto_c: dict, X) -> np.ndarray:
    """Probabilidades calibradas de C: softmax(logits + sesgos por clase).

    Args:
        artefacto_c: Dict de cargar_artefacto_c.
        X: Matriz dispersa de features TF-IDF enmascaradas.

    Returns:
        Matriz (n, n_clases) de probabilidades en el orden de las clases del
        artefacto.
    """
    logits = artefacto_c["modelo"].decision_function(X)
    return _softmax_filas(logits + artefacto_c["sesgos"])


def importar_modulo_07():
    """Importa scripts/07_transformer.py para reutilizar su inferencia chunk-mean.

    La importacion es perezosa (solo en --fase-a5) para que el baseline v1 y
    las fases U/W(b) no dependan de torch/transformers. Se agrega el directorio
    del script a sys.path para que el paquete lib.chunks_beto sea importable.

    Returns:
        Modulo 07_transformer.
    """
    directorio_scripts = str(Path(__file__).resolve().parent)
    if directorio_scripts not in sys.path:
        sys.path.insert(0, directorio_scripts)
    return importlib.import_module("07_transformer")


def cargar_modelo_beto(ruta_modelo: str, dispositivo) -> object:
    """Carga un checkpoint BETO chunk-mean y lo deja en modo evaluacion.

    Args:
        ruta_modelo: Directorio del checkpoint (beto_completo o
            beto_enmascarado).
        dispositivo: cuda/cpu.

    Returns:
        BertForSequenceClassification en eval().

    Raises:
        FileNotFoundError: Si el directorio no existe.
    """
    if not Path(ruta_modelo).exists():
        raise FileNotFoundError(f"falta el checkpoint BETO: {ruta_modelo}")
    from transformers import AutoModelForSequenceClassification

    modelo = AutoModelForSequenceClassification.from_pretrained(ruta_modelo).to(dispositivo)
    modelo.eval()
    return modelo


def inferir_proba_beto(modelo, ids, cache: dict, dispositivo, modulo07) -> np.ndarray:
    """Infiere probabilidades softmax por documento con BETO chunk-mean.

    Replica el procedimiento de evaluacion de 07 (ChunksDataset + collate +
    forward_doc_chunk_mean con autocast fp16 y micro_chunks_eval) y agrega un
    softmax fp32 sobre los logits. El orden de salida es el de `ids`.

    Args:
        modelo: BETO chunk-mean en eval().
        ids: Secuencia de ids de documentos (claves de la cache).
        cache: Dict {id: chunks} de cargar_cache_chunks (07).
        dispositivo: cuda/cpu.
        modulo07: Modulo 07_transformer.

    Returns:
        Matriz (n_docs, n_clases) de probabilidades float32.
    """
    import torch

    ids_serie = pd.Series([int(i) for i in ids])
    etiquetas_falsas = pd.Series(np.zeros(len(ids_serie), dtype=int))
    dataset = modulo07.ChunksDataset(ids_serie, etiquetas_falsas, {0: 0}, cache=cache)
    cargador = modulo07.DataLoader(
        dataset,
        batch_size=modulo07.CONFIG["batch_eval"],
        shuffle=False,
        num_workers=0,
        collate_fn=modulo07.collate_chunks,
        pin_memory=False,
    )
    modelo.eval()
    partes = []
    with torch.no_grad():
        for lote in cargador:
            with torch.autocast(dispositivo.type, dtype=torch.float16, enabled=modulo07.CONFIG["amp"]):
                logits, _ = modulo07.forward_doc_chunk_mean(
                    modelo, lote, dispositivo, micro=modulo07.CONFIG["micro_chunks_eval"]
                )
            partes.append(torch.softmax(logits.float(), dim=-1).cpu().numpy())
    return np.concatenate(partes, axis=0)


def sanity_beto_reproduce_07(modelo, ids, cache: dict, dispositivo, modulo07, n: int) -> dict:
    """Compara la inferencia propia contra `predecir_chunk_mean` de 07.

    Verificacion de reproducibilidad del procedimiento (mismos chunks, mismos
    lotes): obligatoria y barata. No sustituye al sanity de macro-F1 test.

    Args:
        modelo: BETO chunk-mean en eval().
        ids: Secuencia de ids (se usan los primeros n).
        cache: Dict {id: chunks}.
        dispositivo: cuda/cpu.
        modulo07: Modulo 07_transformer.
        n: Numero de documentos a comparar.

    Returns:
        Dict con n_docs, n_coincidencias y coincide (bool).
    """
    muestra = [int(i) for i in ids][:n]
    ids_serie = pd.Series(muestra)
    etiquetas_falsas = pd.Series(np.zeros(len(muestra), dtype=int))
    dataset = modulo07.ChunksDataset(ids_serie, etiquetas_falsas, {0: 0}, cache=cache)
    cargador = modulo07.DataLoader(
        dataset,
        batch_size=modulo07.CONFIG["batch_eval"],
        shuffle=False,
        num_workers=0,
        collate_fn=modulo07.collate_chunks,
        pin_memory=False,
    )
    pred_07, _ = modulo07.predecir_chunk_mean(modelo, cargador, dispositivo)
    probs = inferir_proba_beto(modelo, muestra, cache, dispositivo, modulo07)
    pred_propio = np.argmax(probs, axis=1).tolist()
    coincidencias = int(sum(a == b for a, b in zip(pred_07, pred_propio)))
    resultado = {
        "n_docs": len(muestra),
        "n_coincidencias": coincidencias,
        "coincide": bool(coincidencias == len(muestra)),
    }
    LOGGER.info("sanity BETO vs 07: %d/%d predicciones coinciden", coincidencias, len(muestra))
    return resultado


def bootstrap_pareado_macro_f1(
    y_true,
    y_pred_a,
    y_pred_b,
    n_bootstrap: int,
    seed: int,
) -> dict:
    """IC 95% bootstrap pareado de la diferencia de macro-F1 entre dos modelos.

    Remuestrea documentos con reemplazo (mismo indice para A y B) y calcula
    macro-F1(a) - macro-F1(b). p-valor bilateral aproximado por proporcion de
    deltas a cada lado de 0 (2*min(cola)).

    Args:
        y_true: Etiquetas reales.
        y_pred_a: Predicciones del modelo A (candidato).
        y_pred_b: Predicciones del modelo B (referencia C).
        n_bootstrap: Numero de remuestreos.
        seed: Semilla del generador.

    Returns:
        Dict con delta_observado, ic95, p_valor_bilateral, n_bootstrap y seed.
    """
    y_true = np.asarray(y_true)
    pred_a = np.asarray(y_pred_a)
    pred_b = np.asarray(y_pred_b)
    rng = np.random.default_rng(seed)
    n_docs = len(y_true)
    deltas = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n_docs, size=n_docs)
        deltas[i] = _macro_f1(y_true[idx], pred_a[idx]) - _macro_f1(y_true[idx], pred_b[idx])
    inferior, superior = np.percentile(deltas, [2.5, 97.5])
    p_valor = float(2.0 * min(float(np.mean(deltas <= 0.0)), float(np.mean(deltas >= 0.0))))
    return {
        "delta_observado": float(_macro_f1(y_true, pred_a) - _macro_f1(y_true, pred_b)),
        "ic95": [float(inferior), float(superior)],
        "p_valor_bilateral": p_valor,
        "n_bootstrap": int(n_bootstrap),
        "seed": int(seed),
    }


def pesos_sqrt_inv(y_train) -> dict:
    """Pesos de clase proporcionales a 1/sqrt(frecuencia) en train.

    Args:
        y_train: Etiquetas de entrenamiento.

    Returns:
        Dict clase -> peso float.
    """
    clases, conteos = np.unique(y_train, return_counts=True)
    return {str(clase): float(1.0 / np.sqrt(conteo)) for clase, conteo in zip(clases, conteos)}


def entrenar_logreg_pesos(X_train, y_train, params_v1: dict, modo: str, nombre: str) -> LogisticRegression:
    """Entrena un LogReg con pesos de clase (balanced o 1/sqrt(frecuencia)).

    Args:
        X_train: Matriz de features de train.
        y_train: Etiquetas de train.
        params_v1: Hiperparametros v1 (comparabilidad con C).
        modo: "balanced" o "sqrt_inv".
        nombre: Nombre de la variante para el log.

    Returns:
        LogisticRegression ajustado en train.

    Raises:
        ValueError: Si el modo no es soportado.
    """
    if modo == "balanced":
        pesos = "balanced"
    elif modo == "sqrt_inv":
        pesos = pesos_sqrt_inv(y_train)
    else:
        raise ValueError(f"modo de class_weight no soportado: {modo}")
    modelo = LogisticRegression(**params_v1, class_weight=pesos)
    modelo.fit(X_train, y_train)
    LOGGER.info(
        "%s: base entrenada en train con class_weight=%s (%d iters)",
        nombre, modo, int(modelo.n_iter_.max()),
    )
    return modelo


def evaluar_logreg_calibrado(modelo, X_val, X_test, y_val, y_test, nombre: str) -> dict:
    """Evalua un LogReg con la calibracion C (sesgos por clase en validacion).

    Args:
        modelo: LogisticRegression ajustado en train.
        X_val: Features de validacion.
        X_test: Features de test.
        y_val: Etiquetas de validacion (para optimizar sesgos).
        y_test: Etiquetas de test.
        nombre: Nombre de la variante para el log.

    Returns:
        Dict con metricas val/test sin/con sesgos, sesgos y predicciones de
        test/val (estas ultimas se usan para bootstrap y matriz de confusion).
    """
    clases_modelo = np.asarray(modelo.classes_)
    logits_val = modelo.decision_function(X_val)
    logits_test = modelo.decision_function(X_test)
    sesgos, detalle = optimizar_sesgos_logits(logits_val, y_val, modelo.classes_, MAX_RONDAS_COORDENADA)
    pred_val_sin = modelo.predict(X_val)
    pred_test_sin = modelo.predict(X_test)
    pred_val = clases_modelo[np.argmax(logits_val + sesgos, axis=1)]
    pred_test = clases_modelo[np.argmax(logits_test + sesgos, axis=1)]
    resultado = {
        "modo": nombre,
        "n_iter": int(modelo.n_iter_.max()),
        "val_sin_sesgos": evaluar_single_label(y_val, pred_val_sin, list(clases_modelo)),
        "val_con_sesgos": evaluar_single_label(y_val, pred_val, list(clases_modelo)),
        "test_sin_sesgos": evaluar_single_label(y_test, pred_test_sin, list(clases_modelo)),
        "test_con_sesgos": evaluar_single_label(y_test, pred_test, list(clases_modelo)),
        "sesgos_por_clase": {str(c): float(b) for c, b in zip(clases_modelo, sesgos)},
        "detalle_optimizacion": detalle,
        "y_pred_val": pred_val,
        "y_pred_test": pred_test,
    }
    LOGGER.info(
        "%s: macro-F1 val %.4f -> %.4f | test %.4f -> %.4f",
        nombre,
        resultado["val_sin_sesgos"]["macro_f1"],
        resultado["val_con_sesgos"]["macro_f1"],
        resultado["test_sin_sesgos"]["macro_f1"],
        resultado["test_con_sesgos"]["macro_f1"],
    )
    return resultado


def evaluar_ensamble(
    p_c_val,
    p_beto_val,
    p_c_test,
    p_beto_test,
    y_val,
    y_test,
    clases: list[str],
    pesos: list[float],
    n_bootstrap: int,
    seed_bootstrap: int,
) -> dict:
    """Barre los pesos del ensamble C+BETO y elige por macro-F1 de validacion.

    Args:
        p_c_val: Probabilidades calibradas de C en validacion.
        p_beto_val: Probabilidades de BETO en validacion.
        p_c_test: Probabilidades calibradas de C en test.
        p_beto_test: Probabilidades de BETO en test.
        y_val: Etiquetas de validacion.
        y_test: Etiquetas de test.
        clases: Clases en el orden de las columnas.
        pesos: Rejilla de w (peso de C; BETO usa 1-w).
        n_bootstrap: Remuestreos del IC pareado vs C.
        seed_bootstrap: Semilla del bootstrap.

    Returns:
        Dict con metricas por w, w elegido por validacion, test del elegido,
        predicciones de test del elegido y bootstrap pareado vs C.
    """
    clases_arr = np.asarray(clases)
    por_w = {}
    for w in pesos:
        pred_val = clases_arr[np.argmax(w * p_c_val + (1.0 - w) * p_beto_val, axis=1)]
        pred_test = clases_arr[np.argmax(w * p_c_test + (1.0 - w) * p_beto_test, axis=1)]
        por_w[f"{w:.2f}"] = {
            "w": float(w),
            "val": evaluar_single_label(y_val, pred_val, list(clases_arr)),
            "test": evaluar_single_label(y_test, pred_test, list(clases_arr)),
            "y_pred_val": pred_val,
            "y_pred_test": pred_test,
        }
    w_elegido = max(por_w, key=lambda clave: por_w[clave]["val"]["macro_f1"])
    elegido = por_w[w_elegido]
    pred_c_test = clases_arr[np.argmax(p_c_test, axis=1)]
    bootstrap = bootstrap_pareado_macro_f1(
        y_test, elegido["y_pred_test"], pred_c_test, n_bootstrap, seed_bootstrap
    )
    LOGGER.info(
        "ensamble: w=%.2f elegido por val (macro-F1 val=%.4f test=%.4f; delta test vs C=%+.4f)",
        elegido["w"],
        elegido["val"]["macro_f1"],
        elegido["test"]["macro_f1"],
        bootstrap["delta_observado"],
    )
    return {
        "pesos_evaluados": [float(w) for w in pesos],
        "metricas_por_w": {
            clave: {"w": datos["w"], "val": datos["val"], "test": datos["test"]}
            for clave, datos in por_w.items()
        },
        "w_elegido_val": elegido["w"],
        "val_w_elegido": elegido["val"],
        "test_w_elegido": elegido["test"],
        "bootstrap_pareado_vs_c": bootstrap,
        "y_pred_test_w_elegido": elegido["y_pred_test"],
    }


def matriz_confusion_serializable(y_true, y_pred, clases: list[str]) -> dict:
    """Matriz de confusion (conteos) serializable a JSON.

    Args:
        y_true: Etiquetas reales.
        y_pred: Predicciones.
        clases: Orden fijo de filas/columnas.

    Returns:
        Dict con labels y matriz de enteros.
    """
    matriz = confusion_matrix(y_true, y_pred, labels=list(clases))
    return {"labels": [str(c) for c in clases], "matriz": matriz.astype(int).tolist()}


def figura_c3_a5(valores_val: dict, valores_test: dict, ruta: str) -> None:
    """Figura comparativa de macro-F1 val/test del Sprint A5.

    Args:
        valores_val: Dict variante -> macro-F1 de validacion.
        valores_test: Dict variante -> macro-F1 de test.
        ruta: Ruta del PNG a guardar.
    """
    nombres = list(valores_val)
    x = np.arange(len(nombres))
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.bar(x - 0.2, [valores_val[n] for n in nombres], width=0.4, label="val", color="#2F75B5")
    ax.bar(x + 0.2, [valores_test[n] for n in nombres], width=0.4, label="test", color="#ED7D31")
    ax.axhline(0.32, color="#C00000", linestyle="--", linewidth=1.2, label="meta test 0.32")
    ax.axhline(
        MACRO_F1_TEST_C_FASE_U, color="#70AD47", linestyle=":", linewidth=1.2, label="C test 0.2980"
    )
    ax.set_xticks(x)
    ax.set_xticklabels(nombres, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("macro-F1")
    ax.set_title("ciclo v3 - Sprint A5: macro-F1 val/test (v1, C, BETO, ensamble, class weights)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    Path(ruta).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


def _sanity_beto(resultados_beto: dict, referencia: float, etiqueta: str, smoke: bool) -> bool:
    """Verifica la banda del sanity de macro-F1 test de un escenario BETO.

    Args:
        resultados_beto: Bloque del escenario con "test" -> "macro_f1".
        referencia: Macro-F1 test congelado de la Fase V.
        etiqueta: Nombre del escenario para los mensajes.
        smoke: True si la corrida es un smoke (no aplica la banda completa).

    Returns:
        True si el valor cae dentro de la tolerancia (o si es smoke).

    Raises:
        RuntimeError: En run completo si el valor queda fuera de la banda.
    """
    valor = float(resultados_beto["test"]["macro_f1"])
    delta = abs(valor - referencia)
    dentro = delta <= CONFIG["tolerancia_sanity_beto"]
    if smoke:
        LOGGER.info(
            "sanity BETO %s (smoke): test=%.4f ref=%.4f delta=%.4f (banda no aplica con subset)",
            etiqueta, valor, referencia, delta,
        )
        return True
    if not dentro:
        raise RuntimeError(
            f"sanity BETO {etiqueta} FALLA: macro-F1 test={valor:.6f} fuera de "
            f"{referencia:.6f} +- {CONFIG['tolerancia_sanity_beto']} (delta={delta:.6f}); "
            "investigar antes de continuar (cache de chunks / checkpoint / mapeo de clases)"
        )
    LOGGER.info("sanity BETO %s OK: test=%.6f reproduce la Fase V", etiqueta, valor)
    return True


def ejecutar_fase_a5(df: pd.DataFrame, clases_dataset: list[str], ruta_json: str, limite: int | None) -> dict:
    """Ejecuta el Sprint A5: repro C, ensamble C+BETO y class weights.

    Protocolo: los modelos base se entrenan/leen de train (C y v1 del artefacto;
    class weights se ajustan en train); sesgos, w del ensamble y la seleccion se
    deciden en validacion; test se evalua al final y solo alimenta los IC
    pareados. La clase de BETO se reconstruye con el mismo criterio que 07
    (sorted de las clases del dataset completo, `clases_dataset`).

    Args:
        df: DataFrame con lemas_mask, area_primaria y particion (en smoke,
            submuestreado con particiones canonicas).
        clases_dataset: Clases canonicas del dataset completo (espacio de
            evaluacion y orden de BETO).
        ruta_json: Ruta de salida (resultados/ o /tmp/opencode en smoke).
        limite: Limite del smoke (None = run completo).

    Returns:
        Dict de resultados serializado a JSON.

    Raises:
        FileNotFoundError: Si faltan artefactos (C, v1, cache BETO).
        ValueError: Si las clases del artefacto C no coinciden con el dataset.
        RuntimeError: Si falla un sanity obligatorio (C, BETO) en run completo.
    """
    import torch

    t0 = time.time()
    smoke = limite is not None
    LOGGER.info("fase A5: docs=%d clases=%d smoke=%s limite=%s", len(df), len(clases_dataset), smoke, limite)

    tr = df[df["particion"] == "train"]
    va = df[df["particion"] == "val"]
    te = df[df["particion"] == "test"]
    y_train = tr["area_primaria"].values
    y_val = va["area_primaria"].values
    y_test = te["area_primaria"].values

    ruta_vec = f"{CONFIG['dir_modelo_v1']}/tfidf_enmascarado.joblib"
    ruta_mod = f"{CONFIG['dir_modelo_v1']}/logreg_enmascarado.joblib"
    if not Path(ruta_vec).exists() or not Path(ruta_mod).exists():
        raise FileNotFoundError(f"faltan artefactos v1: {ruta_vec} / {ruta_mod}")
    vec = joblib.load(ruta_vec)
    modelo_v1 = joblib.load(ruta_mod)
    params_v1 = cargar_hiperparametros_v1(ruta_mod)
    artefacto_c = cargar_artefacto_c()
    if artefacto_c["clases"] != [str(c) for c in clases_dataset]:
        raise ValueError("las clases del artefacto C no coinciden con el dataset")
    LOGGER.info("hiperparametros v1 reutilizados: %s", params_v1)

    resultados: dict = {
        "version": "c3-a5",
        "fecha": pd.Timestamp.now().isoformat(),
        "fase": "ciclo v3 - Sprint A5 (clasificador C + BETO chunk-mean + class weights)",
        "modo": f"smoke_limite_{limite}" if smoke else "completo",
        "splits": {"train": len(tr), "val": len(va), "test": len(te)},
        "clases": [str(c) for c in clases_dataset],
        "config_a5": {
            clave: CONFIG[clave]
            for clave in (
                "ruta_artefacto_c", "dir_beto_chunks", "dir_cache_chunks",
                "pesos_ensamble", "bootstrap_n", "bootstrap_seed",
                "tolerancia_sanity_beto", "sanity_n_repro_07", "modos_class_weight",
            )
        },
        "protocolo": (
            "Seleccion (w del ensamble, sesgos) SOLO con validacion; test una vez al "
            "final; bootstrap pareado (1000, seed 42) de macro-F1 test vs C; ningun "
            "artefacto de fases previas se modifica (md5 al final)."
        ),
    }

    # --- 1. Referencia v1 y reproduccion de C desde el artefacto ---
    t_c = time.time()
    X_val = vec.transform(va["lemas_mask"])
    X_test = vec.transform(te["lemas_mask"])
    pred_v1_val = modelo_v1.predict(X_val)
    pred_v1_test = modelo_v1.predict(X_test)
    resultados["referencia_v1_logreg_enmascarado"] = {
        "val": evaluar_single_label(y_val, pred_v1_val, artefacto_c["clases"]),
        "test": evaluar_single_label(y_test, pred_v1_test, artefacto_c["clases"]),
        "nota": (
            "v1 reentreno en train+val: su val esta contaminado (1.0) y no se usa "
            "para calibrar/seleccionar; test esperado 0.2870081505 (run completo)."
        ),
    }
    p_c_val = probabilidades_c(artefacto_c, X_val)
    p_c_test = probabilidades_c(artefacto_c, X_test)
    clases_arr = np.asarray(artefacto_c["clases"])
    pred_c_val = clases_arr[np.argmax(p_c_val, axis=1)]
    pred_c_test = clases_arr[np.argmax(p_c_test, axis=1)]
    c_val = evaluar_single_label(y_val, pred_c_val, artefacto_c["clases"])
    c_test = evaluar_single_label(y_test, pred_c_test, artefacto_c["clases"])
    delta_c = abs(float(c_test["macro_f1"]) - MACRO_F1_TEST_C_FASE_U)
    if not smoke and delta_c > 1e-6:
        raise RuntimeError(
            f"sanity C FALLA: macro-F1 test={c_test['macro_f1']:.16f} != "
            f"{MACRO_F1_TEST_C_FASE_U:.16f} (delta={delta_c:.2e}); revisar el artefacto C"
        )
    resultados["reproduccion_c"] = {
        "val": c_val,
        "test": c_test,
        "sesgos_por_clase": {
            clase: float(sesgo) for clase, sesgo in zip(artefacto_c["clases"], artefacto_c["sesgos"])
        },
        "sanity_test_ok": bool(delta_c <= 1e-6),
        "delta_vs_referencia_test": float(delta_c),
        "ruta_artefacto": artefacto_c["ruta"],
        "md5_artefacto": artefacto_c["md5"],
        "tiempo_seg": time.time() - t_c,
    }
    LOGGER.info(
        "repro C: val=%.6f test=%.6f (delta vs ref=%.2e)",
        c_val["macro_f1"], c_test["macro_f1"], delta_c,
    )

    # --- 2. BETO chunk-mean sobre la cache (escenarios completo y enmascarado) ---
    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dispositivo.type != "cuda":
        raise RuntimeError("la fase A5 requiere GPU para la inferencia BETO (07 aborta sin CUDA)")
    modulo07 = importar_modulo_07()
    bloques_beto = {}
    for escenario, columna, referencia, ruta_checkpoint in (
        ("completo", "texto", MACRO_F1_TEST_BETO_COMPLETO_FASE_V, f"{CONFIG['dir_beto_chunks']}/beto_completo"),
        ("enmascarado", "texto_mask", MACRO_F1_TEST_BETO_ENMASCARADO_FASE_V, f"{CONFIG['dir_beto_chunks']}/beto_enmascarado"),
    ):
        t_beto = time.time()
        cache = modulo07.cargar_cache_chunks(columna, CONFIG["dir_cache_chunks"])
        modelo_beto = cargar_modelo_beto(ruta_checkpoint, dispositivo)
        p_val = inferir_proba_beto(modelo_beto, va["id"], cache, dispositivo, modulo07)
        p_test = inferir_proba_beto(modelo_beto, te["id"], cache, dispositivo, modulo07)
        pred_val = np.asarray(clases_dataset)[np.argmax(p_val, axis=1)]
        pred_test = np.asarray(clases_dataset)[np.argmax(p_test, axis=1)]
        bloque = {
            "val": evaluar_single_label(y_val, pred_val, list(clases_dataset)),
            "test": evaluar_single_label(y_test, pred_test, list(clases_dataset)),
            "ruta_modelo": ruta_checkpoint,
            "tiempo_seg": time.time() - t_beto,
        }
        _sanity_beto(bloque, referencia, escenario, smoke)
        bloque["sanity_test_banda_ok"] = bool(
            abs(float(bloque["test"]["macro_f1"]) - referencia) <= CONFIG["tolerancia_sanity_beto"]
        )
        if escenario == "completo":
            bloque["sanity_reproduce_07"] = sanity_beto_reproduce_07(
                modelo_beto, te["id"], cache, dispositivo, modulo07, CONFIG["sanity_n_repro_07"]
            )
        bloques_beto[escenario] = {"bloque": bloque, "p_val": p_val, "p_test": p_test}
        del modelo_beto
        if dispositivo.type == "cuda":
            torch.cuda.empty_cache()
        LOGGER.info(
            "BETO %s: val=%.4f test=%.4f (%.1f s)",
            escenario, bloque["val"]["macro_f1"], bloque["test"]["macro_f1"], bloque["tiempo_seg"],
        )
    resultados["beto_completo"] = bloques_beto["completo"]["bloque"]
    resultados["beto_enmascarado"] = bloques_beto["enmascarado"]["bloque"]
    resultados["beto_enmascarado"]["nota"] = (
        "variante secundaria (escenario enmascarado); el ensamble se calcula con "
        "beto_completo, el mejor checkpoint del escenario sin enmascarar."
    )

    # --- 3. Ensamble C + BETO completo (w elegido en validacion) ---
    t_ens = time.time()
    ensamble = evaluar_ensamble(
        p_c_val,
        bloques_beto["completo"]["p_val"],
        p_c_test,
        bloques_beto["completo"]["p_test"],
        y_val,
        y_test,
        list(clases_dataset),
        [float(w) for w in CONFIG["pesos_ensamble"]],
        int(CONFIG["bootstrap_n"]),
        int(CONFIG["bootstrap_seed"]),
    )
    pred_ens_test = ensamble.pop("y_pred_test_w_elegido")
    ensamble["tiempo_seg"] = time.time() - t_ens
    resultados["ensamble"] = ensamble

    # --- 4. Class weights (balanced y 1/sqrt(frecuencia)) + calibracion C ---
    t_cw = time.time()
    X_train = vec.transform(tr["lemas_mask"])
    modelos_pesos = {}
    for modo in CONFIG["modos_class_weight"]:
        modelo_pesos = entrenar_logreg_pesos(X_train, y_train, params_v1, modo, f"class_weight_{modo}")
        evaluacion = evaluar_logreg_calibrado(
            modelo_pesos, X_val, X_test, y_val, y_test, f"class_weight_{modo}"
        )
        pred_test_pesos = evaluacion.pop("y_pred_test")
        evaluacion.pop("y_pred_val")
        evaluacion["bootstrap_pareado_vs_c"] = bootstrap_pareado_macro_f1(
            y_test, pred_test_pesos, pred_c_test, int(CONFIG["bootstrap_n"]), int(CONFIG["bootstrap_seed"])
        )
        modelos_pesos[modo] = {"evaluacion": evaluacion, "y_pred_test": pred_test_pesos}
    resultados["class_weights"] = {modo: datos["evaluacion"] for modo, datos in modelos_pesos.items()}
    resultados["class_weights"]["tiempo_seg"] = time.time() - t_cw

    # --- 5. Seleccion por validacion, recomendacion y matrices de confusion ---
    candidatos_val = {
        "C_calibrada": float(c_val["macro_f1"]),
        "ensamble_C_BETO": float(ensamble["val_w_elegido"]["macro_f1"]),
    }
    candidatos_test = {
        "C_calibrada": float(c_test["macro_f1"]),
        "ensamble_C_BETO": float(ensamble["test_w_elegido"]["macro_f1"]),
    }
    candidatos_bootstrap = {
        "C_calibrada": None,
        "ensamble_C_BETO": ensamble["bootstrap_pareado_vs_c"],
    }
    for modo, datos in modelos_pesos.items():
        nombre = f"class_weight_{modo}"
        candidatos_val[nombre] = float(datos["evaluacion"]["val_con_sesgos"]["macro_f1"])
        candidatos_test[nombre] = float(datos["evaluacion"]["test_con_sesgos"]["macro_f1"])
        candidatos_bootstrap[nombre] = datos["evaluacion"]["bootstrap_pareado_vs_c"]
    ganador_val = max(candidatos_val, key=candidatos_val.get)
    resultados["seleccion"] = {
        "ganador_val": ganador_val,
        "macro_f1_val_por_candidato": candidatos_val,
        "macro_f1_test_por_candidato": candidatos_test,
        "nota": "seleccion por macro-F1 de VALIDACION; test no participa en la eleccion.",
    }
    recomendacion = {
        "regla": (
            "adoptar un candidato solo si (a) supera en macro-F1 de validacion a C, "
            "(b) su test no cae por debajo del test de C y (c) el IC 95% pareado de "
            "la diferencia test (candidato - C) queda enteramente > 0."
        ),
        "ensamble": {"adoptar": False, "motivo": ""},
        "class_weights": {"adoptar": False, "motivo": ""},
        "por_candidato": {},
    }
    for nombre, valor_val in candidatos_val.items():
        bootstrap = candidatos_bootstrap[nombre]
        valor_test = candidatos_test[nombre]
        if nombre == "C_calibrada":
            adoptar = True
            motivo = "referencia vigente (Fase U); se conserva salvo que un candidato la supere con evidencia."
        else:
            gana_val = valor_val > candidatos_val["C_calibrada"] + 1e-12
            no_empeora_test = valor_test >= candidatos_test["C_calibrada"] - 1e-12
            if bootstrap is None:
                significativo = False
                detalle_ic = "sin bootstrap"
            else:
                significativo = bootstrap["ic95"][0] > 0.0
                detalle_ic = f"ic95_inferior={bootstrap['ic95'][0]:+.4f}"
            adoptar = bool(gana_val and no_empeora_test and significativo)
            motivo = f"gana_val={gana_val}, no_empeora_test={no_empeora_test}, {detalle_ic}"
        recomendacion["por_candidato"][nombre] = {
            "adoptar": bool(adoptar),
            "macro_f1_val": float(valor_val),
            "macro_f1_test": float(valor_test),
            "delta_test_vs_c": float(valor_test - candidatos_test["C_calibrada"]),
            "motivo": motivo,
        }
    ensamble_adoptar = recomendacion["por_candidato"]["ensamble_C_BETO"]["adoptar"]
    pesos_adoptar = any(
        recomendacion["por_candidato"][f"class_weight_{modo}"]["adoptar"]
        for modo in CONFIG["modos_class_weight"]
    )
    recomendacion["ensamble"] = {
        "adoptar": bool(ensamble_adoptar),
        "motivo": (
            "supera a C con IC pareado positivo en test"
            if ensamble_adoptar
            else "no supera a C en validacion o su IC pareado de test cruza 0; no adoptar en el producto"
        ),
    }
    recomendacion["class_weights"] = {
        "adoptar": bool(pesos_adoptar),
        "motivo": (
            "algun modo supera a C con IC pareado positivo en test"
            if pesos_adoptar
            else "ningun modo supera a C con evidencia (IC pareado); no adoptar en el producto"
        ),
    }
    recomendacion["global"] = (
        "adoptar el candidato ganador con evidencia"
        if (ensamble_adoptar or pesos_adoptar)
        else "mantener C calibrada de la Fase U; los candidatos A5 no superan el ruido bootstrap"
    )
    recomendacion["premisas_adr16"] = [
        "resultado NUEVO en resultados/m4_clasificador_c3.json (version c3-a5); los JSON/modelos de ciclos previos quedan intactos",
        "toda seleccion en validacion; test una sola vez al final",
        "adopcion solo con IC 95% bootstrap pareado favorable (1000 remuestreos, seed 42)",
        "si no hay evidencia, se documenta el negativo con igual rigor y no entra al producto",
    ]
    resultados["recomendacion"] = recomendacion
    resultados["matrices_confusion_test"] = {
        "C_calibrada": matriz_confusion_serializable(y_test, pred_c_test, list(clases_dataset)),
        "ensamble_C_BETO": matriz_confusion_serializable(y_test, pred_ens_test, list(clases_dataset)),
    }
    for modo, datos in modelos_pesos.items():
        resultados["matrices_confusion_test"][f"class_weight_{modo}"] = matriz_confusion_serializable(
            y_test, datos["y_pred_test"], list(clases_dataset)
        )
    LOGGER.info(
        "seleccion por val: ganador=%s (val=%.4f test=%.4f); recomendacion global=%s",
        ganador_val,
        candidatos_val[ganador_val],
        candidatos_test[ganador_val],
        recomendacion["global"],
    )

    # --- 6. Figuras (solo run completo), md5 de intocables y guardado ---
    if not smoke:
        valores_val = {
            "v1_logreg": float(resultados["referencia_v1_logreg_enmascarado"]["val"]["macro_f1"]),
            "C_calibrada": float(c_val["macro_f1"]),
            "BETO_completo": float(resultados["beto_completo"]["val"]["macro_f1"]),
            "BETO_enmascarado": float(resultados["beto_enmascarado"]["val"]["macro_f1"]),
        }
        valores_test = {
            "v1_logreg": float(resultados["referencia_v1_logreg_enmascarado"]["test"]["macro_f1"]),
            "C_calibrada": float(c_test["macro_f1"]),
            "BETO_completo": float(resultados["beto_completo"]["test"]["macro_f1"]),
            "BETO_enmascarado": float(resultados["beto_enmascarado"]["test"]["macro_f1"]),
        }
        for clave, datos in ensamble["metricas_por_w"].items():
            valores_val[f"ens_w{clave}"] = float(datos["val"]["macro_f1"])
            valores_test[f"ens_w{clave}"] = float(datos["test"]["macro_f1"])
        for modo, datos in modelos_pesos.items():
            valores_val[f"cw_{modo}"] = float(datos["evaluacion"]["val_con_sesgos"]["macro_f1"])
            valores_test[f"cw_{modo}"] = float(datos["evaluacion"]["test_con_sesgos"]["macro_f1"])
        figura_c3_a5(valores_val, valores_test, CONFIG["ruta_fig_c3"])
        resultados["figura"] = CONFIG["ruta_fig_c3"]
    resultados["md5_intocables"] = {ruta: md5_archivo(ruta) for ruta in RUTAS_INTOCABLES_A5 if Path(ruta).exists()}
    resultados["md5_artefactos_reutilizados"] = {
        ruta_vec: md5_archivo(ruta_vec),
        ruta_mod: md5_archivo(ruta_mod),
        CONFIG["ruta_artefacto_c"]: artefacto_c["md5"],
    }
    resultados["tiempo_total_seg"] = time.time() - t0
    Path(ruta_json).parent.mkdir(parents=True, exist_ok=True)
    with open(ruta_json, "w", encoding="utf-8") as f:
        json.dump(resultados, f, ensure_ascii=False, indent=2, default=str)
    LOGGER.info(
        "fase A5 fin en %.1f s; ganador val=%s; guardado %s",
        resultados["tiempo_total_seg"], ganador_val, ruta_json,
    )
    return resultados


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parsea los argumentos CLI del script.

    Args:
        argv: Lista de argumentos (None = sys.argv[1:]).

    Returns:
        Namespace con fase_u, fase_a5 y limite.
    """
    parser = argparse.ArgumentParser(
        description="Baseline clasico M4, Fase U/W(b) EuroVoc (ciclo v2) y Sprint A5 (ciclo v3)."
    )
    parser.add_argument(
        "--fase-u",
        action="store_true",
        help=(
            "ejecuta las fases U/W(b) (variantes A/B/C/D/E) y guarda "
            "resultados/m4_eurovoc.json (m4_jerarquico.json intacto)"
        ),
    )
    parser.add_argument(
        "--fase-a5",
        action="store_true",
        help=(
            "ciclo v3, Sprint A5: repro de C + ensamble C+BETO chunk-mean + class "
            "weights; guarda resultados/m4_clasificador_c3.json (default OFF)"
        ),
    )
    parser.add_argument(
        "--limite",
        type=int,
        default=None,
        help=(
            "smoke: submuestra estratificada de N docs; con --fase-a5 la salida va a "
            "/tmp/opencode/m4_clasificador_c3_smoke.json y sin ella a "
            "/tmp/opencode/m4_eurovoc_smoke.json"
        ),
    )
    args = parser.parse_args(argv)
    if args.limite is not None and args.limite < 50:
        parser.error("--limite debe ser >= 50 para que las variantes tengan senal")
    if args.fase_u and args.fase_a5:
        parser.error("--fase-u y --fase-a5 son mutuamente excluyentes")
    return args


def main(argv: list[str] | None = None) -> None:
    """Punto de entrada del script.

    Despacho: baseline v1 (sin flags), fases U/W(b) (--fase-u o --limite solo) o
    Sprint A5 (--fase-a5, con --limite opcional para smoke).

    Args:
        argv: Argumentos CLI (None = sys.argv[1:]).
    """
    args = parse_args(argv)
    smoke = args.limite is not None
    if args.fase_a5:
        configurar_logging(CONFIG["ruta_log_c3_smoke"] if smoke else CONFIG["ruta_log_c3"])
    elif smoke:
        configurar_logging(CONFIG["ruta_log_smoke"])
    elif args.fase_u:
        configurar_logging(CONFIG["ruta_log_fase_u"])
    else:
        configurar_logging()
    LOGGER.info(
        "inicio proceso: fase_u=%s fase_a5=%s limite=%s", args.fase_u, args.fase_a5, args.limite
    )
    df = cargar_datos()
    df = crear_splits(df)
    if args.fase_a5:
        # El mapeo de clases de BETO usa el dataset COMPLETO (protocolo de 07);
        # se captura antes del submuestreo del smoke.
        clases_dataset = sorted(df["area_primaria"].unique())
        if smoke:
            df = submuestrear_estratificado(df, args.limite, CONFIG["semilla"])
            ejecutar_fase_a5(df, clases_dataset, CONFIG["ruta_json_c3_smoke"], args.limite)
        else:
            ejecutar_fase_a5(df, clases_dataset, CONFIG["ruta_json_c3"], None)
    elif smoke:
        df = submuestrear_estratificado(df, args.limite, CONFIG["semilla"])
        ejecutar_fase_u(df, CONFIG["ruta_json_smoke"], args.limite)
    elif args.fase_u:
        ejecutar_fase_u(df, CONFIG["ruta_json_fase_u"], None)
    else:
        ejecutar_baseline_v1(df)
    LOGGER.info("proceso finalizado")


if __name__ == "__main__":
    try:
        main()
    except Exception:  # noqa: BLE001 - protocolo: traza completa y salida 1 ante cualquier fallo
        traceback.print_exc()
        sys.exit(1)
