"""FASE 24 — Diagnóstico de la matriz de test cacheada (control de integridad tipo D-021).

Motivo
    La reconstrucción del v12 desde la matriz cacheada (`meta_features_test_v11.npy`) reprodujo la
    submission enviada solo en el **93,88%** de las filas, cuando el mismo control con la matriz de la
    v11 dio el 100,0000%. Antes de aceptar cualquier candidato hay que localizar la discrepancia.

Protocolo
    1. Reconstruir el lado test **fresco** con el mismo código del pipeline (13 fuentes) más el bloque
       NB-SVM de char+menciones y el bloque supervisado.
    2. Comparar bloque a bloque contra la matriz cacheada: diferencia máxima, correlación media y
       búsqueda de la mejor coincidencia (para detectar permutaciones entre bloques del mismo tipo).
    3. Informe con el bloque responsable y su magnitud.

Salida: ``experiments/phase24_diagnostico_matriz_test.json``.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config as cfg  # noqa: E402
from src import data  # noqa: E402
from src.config import set_seed  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))

RESULTS_PATH = cfg.EXPERIMENTS / "phase24_diagnostico_matriz_test.json"


def bloques_cache() -> list[tuple[str, int, int]]:
    """Bloques de la matriz cacheada: (nombre, inicio, fin)."""
    return [("fuentes_pipeline", 0, 72), ("nb_char_mencion", 72, 78), ("densas", 78, 113),
            ("estructuradas", 113, 200), ("supervisadas", 200, 273)]


def bloques_fresco() -> list[tuple[str, int, int]]:
    """Bloques de la reconstrucción fresca (mismo esquema)."""
    return bloques_cache()


def main() -> int:
    """Ejecuta el diagnóstico comparando la caché con una reconstrucción fresca."""
    set_seed()
    started = time.time()
    payload: dict = {"fase": "diagnóstico de la matriz de test", "fecha": "2026-09-25"}

    from src import advanced_features as af, pipeline

    train = data.load_train()
    test = data.load_test()
    y = train[cfg.TARGET].astype(str).to_numpy()
    cache = np.load(cfg.FEATURES / "meta_features_test_v11.npy")
    dense_test = pd.read_parquet(cfg.FEATURES / "dense_test.parquet")
    dense_train = pd.read_parquet(cfg.FEATURES / "dense_train.parquet")
    dense_columns = pipeline.DENSE_FEATURES

    print("[1/4] representaciones y 13 fuentes del pipeline...", flush=True)
    representaciones = pipeline.build_representations(train, test)
    scores, detalle = pipeline.fit_base_models(representaciones, y,
                                               dense_train[dense_columns].to_numpy(float),
                                               dense_test[dense_columns].to_numpy(float),
                                               verbose=False)
    del detalle
    print(f"      fuentes: {[int(matriz.shape[1]) for matriz in scores]}", flush=True)

    print("[2/4] NB-SVM de char+menciones...", flush=True)
    from advanced_06_submission_v10 import (CHAR_CONFIG, MENTION_CONFIG, nb_log_count_ratio,
                                            tokens_of)
    char = TfidfVectorizer(**CHAR_CONFIG)
    mencion = TfidfVectorizer(**MENTION_CONFIG)
    x_train = sp.hstack([char.fit_transform(train[cfg.TEXT_COL]),
                         mencion.fit_transform(tokens_of(train[cfg.TEXT_COL], r"@\w+"))]).tocsr()
    x_test = sp.hstack([char.transform(test[cfg.TEXT_COL]),
                        mencion.transform(tokens_of(test[cfg.TEXT_COL], r"@\w+"))]).tocsr()
    nb_test = np.zeros((len(test), len(cfg.CLASSES)), dtype=np.float32)
    for posicion, etiqueta in enumerate(cfg.CLASSES):
        binario = (y == etiqueta).astype(int)
        ratio = nb_log_count_ratio(x_train, binario)
        modelo = LinearSVC(C=0.2, max_iter=3000)
        modelo.fit(x_train.multiply(ratio).tocsr(), binario)
        nb_test[:, posicion] = modelo.decision_function(x_test.multiply(ratio).tocsr())
    print("[3/4] bloques supervisado y estructurado...", flush=True)
    supervised_test = pd.concat([
        af.supervised_entity_features(train[cfg.TEXT_COL], y, test[cfg.TEXT_COL], "mention"),
        af.supervised_entity_features(train[cfg.TEXT_COL], y, test[cfg.TEXT_COL], "hashtag"),
        af.supervised_entity_features(train[cfg.TEXT_COL], y, test[cfg.TEXT_COL], "rt_source"),
        af.lexical_rarity_features(train[cfg.TEXT_COL], test[cfg.TEXT_COL]),
        af.class_lexicon_scores(train[cfg.TEXT_COL], y, test[cfg.TEXT_COL]),
    ], axis=1).to_numpy(float)
    structured_test = pd.read_parquet(cfg.FEATURES / "age_structured_features_test.parquet") \
        .to_numpy(float)
    fresco = np.concatenate(scores + [nb_test, dense_test[dense_columns].to_numpy(float),
                                      structured_test, supervised_test], axis=1)
    payload["formas"] = {"cache": list(cache.shape), "fresco": list(fresco.shape)}

    print("[4/4] comparación bloque a bloque...", flush=True)
    comparaciones = []
    for (nombre, inicio, fin) in bloques_cache():
        izquierda = cache[:, inicio:fin]
        derecha = fresco[:, inicio:fin]
        diferencia = float(np.abs(izquierda - derecha).max())
        correlacion = float(np.mean([np.corrcoef(izquierda[:, columna], derecha[:, columna])[0, 1]
                                     for columna in range(fin - inicio)]))
        comparaciones.append({"bloque": nombre, "columnas": fin - inicio,
                              "max_diferencia_absoluta": round(diferencia, 6),
                              "correlacion_media_columnas": round(correlacion, 6),
                              "coincide_exacto": bool(diferencia == 0.0)})
        print(f"  {nombre:18s} max_diff {diferencia:.6f} | corr {correlacion:.6f}", flush=True)
    payload["comparacion_bloques"] = comparaciones

    # Detección de permutaciones dentro del bloque de fuentes (sub-bloques conocidos).
    sub_bloques = [(0, 6), (6, 12), (12, 18), (18, 24), (24, 30), (30, 36), (36, 37), (37, 43),
                   (43, 49), (49, 55), (55, 61), (61, 67), (67, 72)]
    permutaciones = []
    for (inicio, fin) in sub_bloques:
        mejor, mejor_correlacion = None, -2.0
        for (otro_inicio, otro_fin) in sub_bloques:
            if otro_fin - otro_inicio != fin - inicio:
                continue
            correlacion = float(np.mean([
                np.corrcoef(cache[:, inicio + columna], fresco[:, otro_inicio + columna])[0, 1]
                for columna in range(fin - inicio)]))
            if correlacion > mejor_correlacion:
                mejor, mejor_correlacion = (otro_inicio, otro_fin), correlacion
        permutaciones.append({"sub_bloque_cache": [inicio, fin],
                              "mejor_coincidencia_fresco": list(mejor),
                              "correlacion": round(mejor_correlacion, 4),
                              "es_su_posicion": bool(mejor == (inicio, fin))})
    payload["permutaciones_fuentes"] = permutaciones
    for fila in permutaciones:
        print(f"  cache[{fila['sub_bloque_cache'][0]}:{fila['sub_bloque_cache'][1]}] <-> "
              f"fresco[{fila['mejor_coincidencia_fresco'][0]}:{fila['mejor_coincidencia_fresco'][1]}] "
              f"corr {fila['correlacion']} {'OK' if fila['es_su_posicion'] else 'PERMUTADO'}",
              flush=True)

    payload["segundos"] = round(time.time() - started, 1)
    RESULTS_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str),
                            encoding="utf-8")
    print(f"\nresultado: {RESULTS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
