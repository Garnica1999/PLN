"""Control de robustez por pseudo-autor (cierre de la brecha de D-004).

Pregunta que responde
---------------------
¿Cambia la estimacion out-of-fold del modelo final si la validacion cruzada
agrupa ademas por pseudo-autor (primera mencion ``@usuario``) y no solo por
near-duplicados? Si un mismo usuario concentra varios tweets con etiquetas
correlacionadas, el split agrupado por near-duplicado podria seguir siendo
optimista.

Metodo
------
1. Se cargan train, los grupos de near-duplicado y se calcula el pseudo-autor
   como la primera mencion de cada tweet.
2. Se fusionan ambos criterios en un unico grupo (union-find): dos tweets caen
   en el mismo grupo si son near-duplicados o comparten pseudo-autor.
3. Se reevalua el modelo final (MOD-03b_BEST: union TF-IDF word(1,2) +
   char_wb(3,5) + LogisticRegression C=2.0 balanced x3) con
   ``StratifiedGroupKFold(5)`` sobre los grupos fusionados, con la misma
   estratificacion por combinacion HS x TR x AG y la misma semilla.
4. Se compara la F1 media por columna contra la publicada (0.7926588861781015)
   y se reporta la diferencia.

Salidas
-------
``reports/control_pseudo_autores.json`` y una fila ``PA-01`` en
``experiments.csv``.

Ejecucion
---------
    /home/carlos/venv/bin/python experiments/12_control_pseudo_autores.py
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src import data as data_module
from src.evaluation import metricas_multietiqueta

ETIQUETAS = ("HS", "TR", "AG")
RANDOM_STATE = 42
N_SPLITS = 5
F1_PUBLICADA = 0.7926588861781015
MENTION_PATTERN = re.compile(r"@(\w+)")
GRUPOS_PATH = PROJECT_ROOT / "data" / "processed" / "grupos_near_duplicados.csv"
REPORTE_JSON = PROJECT_ROOT / "reports" / "control_pseudo_autores.json"
EXPERIMENTS_CSV = PROJECT_ROOT / "experiments.csv"

PARAMS_WORD = {
    "analyzer": "word",
    "ngram_range": (1, 2),
    "min_df": 2,
    "sublinear_tf": True,
}
PARAMS_CHAR = {
    "analyzer": "char_wb",
    "ngram_range": (3, 5),
    "min_df": 2,
    "sublinear_tf": True,
}
PARAMS_MODELO = {
    "C": 2.0,
    "class_weight": "balanced",
    "max_iter": 1000,
    "solver": "liblinear",
    "random_state": RANDOM_STATE,
}


def log(message: str) -> None:
    """Imprime un mensaje de progreso con prefijo de fase.

    Args:
        message: texto a imprimir.
    """
    print(f"[PA] {message}", flush=True)


def grupos_fusionados(train: pd.DataFrame) -> tuple[np.ndarray, dict]:
    """Fusiona near-duplicados y pseudo-autores en una sola particion.

    Args:
        train: DataFrame de entrenamiento con ``id`` y ``text``.

    Returns:
        Tupla ``(asignacion, resumen)`` donde ``asignacion`` es el arreglo de
        grupos alineado con ``train`` y ``resumen`` describe la fusion.
    """
    near_dup = pd.read_csv(GRUPOS_PATH).set_index("id")["grupo"]
    ids = train["id"].astype(int).to_numpy()
    autores = [
        match.group(1).lower() if (match := MENTION_PATTERN.search(text)) else ""
        for text in train["text"]
    ]
    padre = list(range(len(ids)))

    def encontrar(nodo: int) -> int:
        """Devuelve la raiz del nodo con compresion de caminos."""
        while padre[nodo] != nodo:
            padre[nodo] = padre[padre[nodo]]
            nodo = padre[nodo]
        return nodo

    def unir(a: int, b: int) -> None:
        """Une dos nodos en la misma componente."""
        raiz_a, raiz_b = encontrar(a), encontrar(b)
        if raiz_a != raiz_b:
            padre[max(raiz_a, raiz_b)] = min(raiz_a, raiz_b)

    # Fusiona por near-duplicado (usando el grupo de Fase 0).
    representante: dict[int, int] = {}
    for posicion, identificador in enumerate(ids):
        grupo = int(near_dup.get(identificador, identificador))
        if grupo in representante:
            unir(posicion, representante[grupo])
        else:
            representante[grupo] = posicion
    # Fusiona por pseudo-autor (solo menciones no vacias).
    por_autor: dict[str, int] = {}
    autores_con_grupo = 0
    tweets_con_autor = 0
    for posicion, autor in enumerate(autores):
        if not autor:
            continue
        tweets_con_autor += 1
        if autor in por_autor:
            unir(posicion, por_autor[autor])
        else:
            por_autor[autor] = posicion
    autores_con_grupo = sum(
        1 for autor, primera in por_autor.items() if any(
            autores[i] == autor for i in range(len(autores)) if i != primera
        )
    )
    asignacion = np.array([encontrar(nodo) for nodo in range(len(ids))])
    resumen = {
        "n_tweets": len(ids),
        "tweets_con_mencion": tweets_con_autor,
        "pseudo_autores_distintos": len(por_autor),
        "pseudo_autores_con_mas_de_un_tweet": autores_con_grupo,
        "n_grupos_fusionados": len(set(asignacion.tolist())),
    }
    return asignacion, resumen


def evaluar(train: pd.DataFrame, grupos: np.ndarray) -> dict:
    """Evalua el modelo final con la particion fusionada.

    Args:
        train: DataFrame de entrenamiento con etiquetas.
        grupos: asignacion de grupo por fila.

    Returns:
        Diccionario con metricas out-of-fold y detalle por pliegue.
    """
    combinacion = (
        train["HS"].astype(str) + train["TR"].astype(str) + train["AG"].astype(str)
    )
    y_true = train[list(ETIQUETAS)].to_numpy()
    oof = np.zeros_like(y_true)
    divisor = StratifiedGroupKFold(
        n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE
    )
    pliegues = []
    for numero, (idx_entrenamiento, idx_validacion) in enumerate(
        divisor.split(train, combinacion, groups=grupos), start=1
    ):
        grupos_entrenamiento = set(grupos[idx_entrenamiento].tolist())
        grupos_validacion = set(grupos[idx_validacion].tolist())
        if grupos_entrenamiento & grupos_validacion:
            raise RuntimeError("Hay grupos compartidos entre train y validacion")
        word = TfidfVectorizer(**PARAMS_WORD)
        char = TfidfVectorizer(**PARAMS_CHAR)
        matriz_entrenamiento = sparse.hstack(
            [
                word.fit_transform(train["text"].iloc[idx_entrenamiento]),
                char.fit_transform(train["text"].iloc[idx_entrenamiento]),
            ]
        ).tocsr()
        matriz_validacion = sparse.hstack(
            [
                word.transform(train["text"].iloc[idx_validacion]),
                char.transform(train["text"].iloc[idx_validacion]),
            ]
        ).tocsr()
        for indice, etiqueta in enumerate(ETIQUETAS):
            modelo = LogisticRegression(**PARAMS_MODELO)
            modelo.fit(matriz_entrenamiento, train[etiqueta].iloc[idx_entrenamiento])
            oof[idx_validacion, indice] = (
                modelo.predict_proba(matriz_validacion)[:, 1] >= 0.5
            ).astype(int)
        metrica = metricas_multietiqueta(y_true[idx_validacion], oof[idx_validacion])
        metrica["pliegue"] = numero
        pliegues.append(metrica)
        log(f"pliegue {numero}: F1 media por columna {metrica['f1_media_columnas']:.4f}")
    global_ = metricas_multietiqueta(y_true, oof)
    global_["std_pliegues"] = float(
        np.std([fila["f1_media_columnas"] for fila in pliegues])
    )
    return {"global": global_, "pliegues": pliegues}


def registrar(resultado: dict, resumen_grupos: dict, segundos: float) -> None:
    """Anade la fila ``PA-01`` a ``experiments.csv``.

    Args:
        resultado: salida de :func:`evaluar`.
        resumen_grupos: resumen de la fusion de grupos.
        segundos: duracion de la evaluacion.
    """
    f1 = float(resultado["global"]["f1_media_columnas"])
    fila = {
        "experiment_id": "PA-01",
        "competition": "KAGGLE_TOXIC_2026_PUJ",
        "question": "Cambia el OOF si la validacion agrupa ademas por pseudo-autor?",
        "hypothesis": "El control por pseudo-autor reduce el optimismo del split por near-duplicados",
        "preprocessing": "texto crudo, lowercase",
        "feature_groups": "WORD_NGRAM+CHAR_NGRAM",
        "representation": "union TF-IDF word(1,2) + char_wb(3,5), min_df=2, sublinear_tf",
        "model": "LogisticRegression(class_weight=balanced, C=2.0, liblinear) x3",
        "hyperparameters": json.dumps(PARAMS_MODELO, ensure_ascii=False),
        "validation": "StratifiedGroupKFold(5) con grupos fusionados (near-duplicado + pseudo-autor)",
        "f1": round(f1, 4),
        "macro_f1": round(float(resultado["global"]["f1_macro6"]), 4),
        "weighted_f1": "",
        "precision": "",
        "recall": "",
        "training_time": round(segundos, 2),
        "dimension": "",
        "result": json.dumps(
            {
                "f1_media_columnas": f1,
                "f1_publicada_solo_near_dup": F1_PUBLICADA,
                "diferencia": round(f1 - F1_PUBLICADA, 6),
                "std_pliegues": resultado["global"]["std_pliegues"],
                "resumen_grupos": resumen_grupos,
            },
            ensure_ascii=False,
        ),
        "interpretation": (
            "Si la diferencia es pequena, el optimismo del split por near-duplicados es acotado "
            "y el protocolo de validacion del proyecto queda respaldado"
        ),
        "decision": "Cierre de la brecha de D-004 detectada en la auditoria final",
        "date": "2026-09-23",
        "script": "experiments/12_control_pseudo_autores.py",
        "notes": "Umbral fijo 0.5; sin post-proceso",
    }
    existente = pd.read_csv(EXPERIMENTS_CSV) if EXPERIMENTS_CSV.exists() else pd.DataFrame()
    if "PA-01" in existente.get("experiment_id", pd.Series(dtype=str)).tolist():
        existente = existente[existente["experiment_id"] != "PA-01"]
        existente.to_csv(EXPERIMENTS_CSV, index=False)
    pd.DataFrame([fila]).to_csv(EXPERIMENTS_CSV, mode="a", header=not EXPERIMENTS_CSV.exists(), index=False)


def main() -> None:
    """Ejecuta el control de robustez por pseudo-autor."""
    inicio = time.perf_counter()
    train = data_module.read_train()
    asignacion, resumen = grupos_fusionados(train)
    log(f"grupos fusionados: {json.dumps(resumen, ensure_ascii=False)}")
    resultado = evaluar(train, asignacion)
    segundos = time.perf_counter() - inicio
    diferencia = float(resultado["global"]["f1_media_columnas"]) - F1_PUBLICADA
    salida = {
        "fecha": "2026-09-23",
        "resumen_grupos": resumen,
        "f1_con_grupos_fusionados": float(resultado["global"]["f1_media_columnas"]),
        "f1_solo_near_duplicados": F1_PUBLICADA,
        "diferencia": diferencia,
        "metricas_globales": resultado["global"],
        "pliegues": resultado["pliegues"],
        "duracion_segundos": round(segundos, 2),
    }
    REPORTE_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORTE_JSON.write_text(json.dumps(salida, indent=2, ensure_ascii=False), encoding="utf-8")
    registrar(resultado, resumen, segundos)
    log(f"F1 fusionado {resultado['global']['f1_media_columnas']:.4f} vs publicado {F1_PUBLICADA:.4f} "
        f"(diferencia {diferencia:+.6f})")
    log(f"reporte: {REPORTE_JSON}")


if __name__ == "__main__":
    main()
