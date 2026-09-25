"""P0.3 - OOF corregido de los modelos base sobre el desarrollo (D-016).

Pregunta que responde
---------------------
¿Cual es el rendimiento out-of-fold, con el protocolo corregido (grupos =
near-duplicado UNION pseudo-autor), de los modelos base sobre el conjunto de
desarrollo, y cual es la referencia de V03 en ese mismo conjunto?

Metodo
------
1. Se carga la particion congelada de P0.2 y se restringe el trabajo al
   desarrollo (nunca se toca el holdout).
2. Para cada modelo base se ajustan vectorizador y modelo DENTRO de cada
   pliegue y se guardan las probabilidades out-of-fold:
   - M1: TF-IDF word(1,2) min_df=2 sublinear + LR C=1.0 balanced.
   - M2: union word(1,2) + char_wb(3,5) min_df=2 sublinear + LR C=2.0 balanced
     (el modelo de V03).
   - M3: misma union + LinearSVC C=1.0 balanced (probabilidad por sigmoide del
     decision_function).
   - M4: TF-IDF char_wb(3,5) min_df=2 sublinear + LR C=1.0 balanced.
3. Se reportan: corrected_cv (protocolo v2) de cada modelo y, para M2, tambien
   el historical_cv sobre el mismo desarrollo (grupos = solo near-duplicado) y
   su F1 con los umbrales congelados de V03 (0.44/0.51/0.46).

Salidas
-------
``features/v2_oof_base_corrected.npz`` (ids, y_true, probabilidades, pliegues),
``reports/baseline_v03_dev_metrics.json`` y filas ``V2-BASE-01..04`` en
``experiments.csv``.

Ejecucion
---------
    /home/carlos/venv/bin/python experiments/14_oof_corregido_v2.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import expit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.svm import LinearSVC

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src import data as data_module
from src.evaluation import metricas_multietiqueta
from src.splits import (
    ETIQUETAS,
    RANDOM_STATE,
    cargar_grupos_near_duplicado,
    combinaciones,
    grupos_fusionados,
    pliegues_corregidos,
)

IDS_PATH = PROJECT_ROOT / "data" / "processed" / "holdout_v2_ids.csv"
OOF_PATH = PROJECT_ROOT / "features" / "v2_oof_base_corrected.npz"
REPORTE_JSON = PROJECT_ROOT / "reports" / "baseline_v03_dev_metrics.json"
EXPERIMENTS_CSV = PROJECT_ROOT / "experiments.csv"
UMBRALES_V03 = {"HS": 0.44, "TR": 0.51, "AG": 0.46}

PARAMS_WORD = {"analyzer": "word", "ngram_range": (1, 2), "min_df": 2, "sublinear_tf": True}
PARAMS_CHAR = {"analyzer": "char_wb", "ngram_range": (3, 5), "min_df": 2, "sublinear_tf": True}
LR_C1 = {"C": 1.0, "class_weight": "balanced", "max_iter": 1000, "solver": "liblinear", "random_state": RANDOM_STATE}
LR_C2 = {"C": 2.0, "class_weight": "balanced", "max_iter": 1000, "solver": "liblinear", "random_state": RANDOM_STATE}
SVC_PARAMS = {"C": 1.0, "class_weight": "balanced", "max_iter": 5000, "random_state": RANDOM_STATE}

MODELOS = {
    "M1_word_lr": {"tipo": "lr", "params": LR_C1, "representacion": "word"},
    "M2_union_lr_c2": {"tipo": "lr", "params": LR_C2, "representacion": "union"},
    "M3_union_svc": {"tipo": "svc", "params": SVC_PARAMS, "representacion": "union"},
    "M4_char_lr": {"tipo": "lr", "params": LR_C1, "representacion": "char"},
}


def log(message: str) -> None:
    """Imprime un mensaje de progreso con prefijo de fase.

    Args:
        message: texto a imprimir.
    """
    print(f"[P0.3] {message}", flush=True)


def construir_matrices(textos_entrenamiento: pd.Series, textos_validacion: pd.Series, representacion: str):
    """Ajusta los vectorizadores en el pliegue y transforma validacion.

    Args:
        textos_entrenamiento: textos del pliegue de entrenamiento.
        textos_validacion: textos del pliegue de validacion.
        representacion: ``word``, ``char`` o ``union``.

    Returns:
        Tupla ``(matriz_entrenamiento, matriz_validacion)``.
    """
    if representacion == "word":
        vectorizador = TfidfVectorizer(**PARAMS_WORD)
        return (
            vectorizador.fit_transform(textos_entrenamiento),
            vectorizador.transform(textos_validacion),
        )
    if representacion == "char":
        vectorizador = TfidfVectorizer(**PARAMS_CHAR)
        return (
            vectorizador.fit_transform(textos_entrenamiento),
            vectorizador.transform(textos_validacion),
        )
    word = TfidfVectorizer(**PARAMS_WORD)
    char = TfidfVectorizer(**PARAMS_CHAR)
    matriz_entrenamiento = sparse.hstack(
        [
            word.fit_transform(textos_entrenamiento),
            char.fit_transform(textos_entrenamiento),
        ]
    ).tocsr()
    matriz_validacion = sparse.hstack(
        [word.transform(textos_validacion), char.transform(textos_validacion)]
    ).tocsr()
    return matriz_entrenamiento, matriz_validacion


def evaluar_modelo(
    train_dev: pd.DataFrame, pliegues: list[tuple[np.ndarray, np.ndarray]], especificacion: dict
) -> tuple[np.ndarray, list[dict]]:
    """Calcula las probabilidades out-of-fold de un modelo base.

    Args:
        train_dev: DataFrame del conjunto de desarrollo.
        pliegues: lista de tuplas ``(entrenamiento, validacion)``.
        especificacion: diccionario con ``tipo``, ``params`` y ``representacion``.

    Returns:
        Tupla ``(probabilidades, metricas_por_pliegue)``.
    """
    probabilidades = np.zeros((len(train_dev), len(ETIQUETAS)), dtype=float)
    y_true = train_dev[list(ETIQUETAS)].to_numpy()
    metricas_pliegues = []
    for numero, (idx_entrenamiento, idx_validacion) in enumerate(pliegues, start=1):
        matriz_entrenamiento, matriz_validacion = construir_matrices(
            train_dev["text"].iloc[idx_entrenamiento],
            train_dev["text"].iloc[idx_validacion],
            especificacion["representacion"],
        )
        for indice, etiqueta in enumerate(ETIQUETAS):
            if especificacion["tipo"] == "lr":
                modelo = LogisticRegression(**especificacion["params"])
                modelo.fit(matriz_entrenamiento, train_dev[etiqueta].iloc[idx_entrenamiento])
                puntajes = modelo.predict_proba(matriz_validacion)[:, 1]
            else:
                modelo = LinearSVC(**especificacion["params"])
                modelo.fit(matriz_entrenamiento, train_dev[etiqueta].iloc[idx_entrenamiento])
                puntajes = expit(modelo.decision_function(matriz_validacion))
            probabilidades[idx_validacion, indice] = puntajes
        metrica = metricas_multietiqueta(
            y_true[idx_validacion], (probabilidades[idx_validacion] >= 0.5).astype(int)
        )
        metrica["pliegue"] = numero
        metricas_pliegues.append(metrica)
        log(f"  pliegue {numero}: F1 media por columna {metrica['f1_media_columnas']:.4f}")
    return probabilidades, metricas_pliegues


def evaluar_protocolo_historico(
    train_dev: pd.DataFrame, grupos_near_dup_dev: np.ndarray
) -> dict:
    """Reevalua M2 con el protocolo historico (solo near-duplicado) sobre el desarrollo.

    Args:
        train_dev: DataFrame del conjunto de desarrollo.
        grupos_near_dup_dev: grupos de near-duplicado del desarrollo.

    Returns:
        Diccionario con el resumen out-of-fold y el detalle por pliegue.
    """
    divisor = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    pliegues = list(
        divisor.split(train_dev, combinaciones(train_dev), groups=grupos_near_dup_dev)
    )
    probabilidades, _ = evaluar_modelo(
        train_dev, pliegues, MODELOS["M2_union_lr_c2"]
    )
    y_true = train_dev[list(ETIQUETAS)].to_numpy()
    resumen = metricas_multietiqueta(y_true, (probabilidades >= 0.5).astype(int))
    resumen_umbrales = metricas_multietiqueta(
        y_true,
        np.column_stack(
            [
                (probabilidades[:, indice] >= UMBRALES_V03[etiqueta]).astype(int)
                for indice, etiqueta in enumerate(ETIQUETAS)
            ]
        ),
    )
    return {"umbral_0_5": resumen, "umbrales_v03": resumen_umbrales}


def main() -> None:
    """Ejecuta P0.3 y guarda los artefactos."""
    inicio = time.perf_counter()
    train = data_module.read_train()
    particion = pd.read_csv(IDS_PATH)
    ids_dev = particion.loc[particion["particion"] == "dev", "id"].astype(int).tolist()
    train_dev = train[train["id"].astype(int).isin(ids_dev)].reset_index(drop=True)
    log(f"desarrollo: {len(train_dev)} filas (holdout reservado)")

    grupos_full = grupos_fusionados(train)
    mapa_grupo = dict(zip(train["id"].astype(int), grupos_full.tolist()))
    grupos_dev = np.array([mapa_grupo[i] for i in train_dev["id"].astype(int)])
    grupos_near_dup = cargar_grupos_near_duplicado()
    grupos_near_dup_dev = train_dev["id"].astype(int).map(grupos_near_dup).to_numpy()

    pliegues = pliegues_corregidos(train_dev, grupos_dev)
    y_true = train_dev[list(ETIQUETAS)].to_numpy()
    resultados = {}
    probabilidades_por_modelo = {}
    for nombre, especificacion in MODELOS.items():
        log(f"modelo {nombre}")
        probabilidades, metricas_pliegues = evaluar_modelo(train_dev, pliegues, especificacion)
        probabilidades_por_modelo[nombre] = probabilidades
        resumen = metricas_multietiqueta(y_true, (probabilidades >= 0.5).astype(int))
        resumen["std_pliegues"] = float(
            np.std([fila["f1_media_columnas"] for fila in metricas_pliegues])
        )
        resultados[nombre] = {"resumen": resumen, "pliegues": metricas_pliegues}
        log(f"  corrected_cv = {resumen['f1_media_columnas']:.4f} (std {resumen['std_pliegues']:.4f})")

    # V03 con sus umbrales congelados sobre el OOF corregido.
    proba_v03 = probabilidades_por_modelo["M2_union_lr_c2"]
    prediccion_umbrales = np.column_stack(
        [
            (proba_v03[:, indice] >= UMBRALES_V03[etiqueta]).astype(int)
            for indice, etiqueta in enumerate(ETIQUETAS)
        ]
    )
    resumen_v03_umbrales = metricas_multietiqueta(y_true, prediccion_umbrales)

    log("protocolo historico (solo near-duplicado) sobre el desarrollo")
    historico = evaluar_protocolo_historico(train_dev, grupos_near_dup_dev)

    OOF_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OOF_PATH,
        ids=train_dev["id"].astype(int).to_numpy(),
        y_true=y_true,
        proba=np.stack([probabilidades_por_modelo[nombre] for nombre in MODELOS], axis=1),
        modelos=np.array(list(MODELOS)),
        etiquetas=np.array(list(ETIQUETAS)),
        grupos=grupos_dev,
    )
    reporte = {
        "fecha": "2026-09-25",
        "protocolo": "v2 (D-016): grupos = near-duplicado UNION pseudo-autor",
        "n_dev": len(train_dev),
        "umbrales_v03": UMBRALES_V03,
        "modelos": resultados,
        "v03_con_umbrales_corrected": resumen_v03_umbrales,
        "v03_protocolo_historico_sobre_dev": historico,
        "referencias_train_completo": {
            "v03_historical_cv": 0.7926588861781015,
            "v03_corrected_cv_pa01": 0.7838,
        },
        "duracion_segundos": round(time.perf_counter() - inicio, 2),
        "artefacto_oof": str(OOF_PATH.relative_to(PROJECT_ROOT)),
    }
    REPORTE_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORTE_JSON.write_text(json.dumps(reporte, indent=2, ensure_ascii=False), encoding="utf-8")

    filas = []
    for nombre, resultado in resultados.items():
        resumen = resultado["resumen"]
        filas.append(
            {
                "experiment_id": f"V2-BASE-{nombre}",
                "competition": "KAGGLE_TOXIC_2026_PUJ",
                "question": "Rendimiento out-of-fold corregido del modelo base sobre el desarrollo",
                "hypothesis": "El protocolo corregido (near-dup + pseudo-autor) da la referencia honesta",
                "preprocessing": "texto crudo, lowercase",
                "feature_groups": "WORD_NGRAM" if nombre.startswith("M1") else "CHAR_NGRAM" if nombre.startswith("M4") else "WORD_NGRAM+CHAR_NGRAM",
                "representation": MODELOS[nombre]["representacion"],
                "model": nombre,
                "hyperparameters": json.dumps(MODELOS[nombre]["params"], ensure_ascii=False),
                "validation": "StratifiedGroupKFold(5) sobre desarrollo, grupos fusionados",
                "f1": round(resumen["f1_media_columnas"], 4),
                "macro_f1": round(resumen["f1_macro6"], 4),
                "weighted_f1": "",
                "precision": "",
                "recall": "",
                "training_time": round(reporte["duracion_segundos"], 2),
                "dimension": "",
                "result": json.dumps(
                    {
                        "corrected_cv": resumen["f1_media_columnas"],
                        "std_pliegues": resumen["std_pliegues"],
                        "f1_por_etiqueta": resumen["f1_por_etiqueta"],
                    },
                    ensure_ascii=False,
                ),
                "interpretation": "Referencia v2 sobre el desarrollo",
                "decision": "Baseline del protocolo v2",
                "date": "2026-09-25",
                "script": "experiments/14_oof_corregido_v2.py",
                "notes": "Holdout no utilizado",
            }
        )
    existente = pd.read_csv(EXPERIMENTS_CSV) if EXPERIMENTS_CSV.exists() else pd.DataFrame()
    ids_nuevos = {fila["experiment_id"] for fila in filas}
    if len(existente):
        existente = existente[~existente["experiment_id"].isin(ids_nuevos)]
        existente.to_csv(EXPERIMENTS_CSV, index=False)
    pd.DataFrame(filas).to_csv(EXPERIMENTS_CSV, mode="a", header=True, index=False)
    log("filas V2-BASE-* registradas")
    log(f"V03 corrected con umbrales: {resumen_v03_umbrales['f1_media_columnas']:.4f} "
        f"| V03 historico sobre dev: {historico['umbral_0_5']['f1_media_columnas']:.4f}")
    log(f"artefactos: {OOF_PATH.name}, {REPORTE_JSON.name}")


if __name__ == "__main__":
    main()
