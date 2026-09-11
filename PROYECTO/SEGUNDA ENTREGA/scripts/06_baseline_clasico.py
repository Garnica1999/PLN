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

Uso:
    /home/carlos/venv/bin/python scripts/06_baseline_clasico.py
    /home/carlos/venv/bin/python scripts/06_baseline_clasico.py --fase-u
    /home/carlos/venv/bin/python scripts/06_baseline_clasico.py --limite 1500
"""

import argparse
import hashlib
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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parsea los argumentos CLI del script.

    Args:
        argv: Lista de argumentos (None = sys.argv[1:]).

    Returns:
        Namespace con fase_u y limite.
    """
    parser = argparse.ArgumentParser(
        description="Baseline clasico M4, Fase U y Fase W(b) EuroVoc (ciclo v2)."
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
        "--limite",
        type=int,
        default=None,
        help="smoke: submuestra estratificada de N docs, salida en /tmp/opencode/m4_eurovoc_smoke.json",
    )
    args = parser.parse_args(argv)
    if args.limite is not None and args.limite < 50:
        parser.error("--limite debe ser >= 50 para que las variantes tengan senal")
    return args


def main(argv: list[str] | None = None) -> None:
    """Punto de entrada: baseline v1 (sin flags) o fases U/W(b) (--fase-u/--limite).

    Args:
        argv: Argumentos CLI (None = sys.argv[1:]).
    """
    args = parse_args(argv)
    smoke = args.limite is not None
    if smoke:
        configurar_logging(CONFIG["ruta_log_smoke"])
    elif args.fase_u:
        configurar_logging(CONFIG["ruta_log_fase_u"])
    else:
        configurar_logging()
    LOGGER.info("inicio proceso: fase_u=%s limite=%s", args.fase_u, args.limite)
    df = cargar_datos()
    df = crear_splits(df)
    if smoke:
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
