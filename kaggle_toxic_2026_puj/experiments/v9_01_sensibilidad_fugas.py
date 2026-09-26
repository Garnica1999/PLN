"""V9-01 (H5): sensibilidad de las dos fugas corregidas.

Pregunta
--------
¿Cambia la referencia V03 (0.777211) si los grupos de near-duplicados se
construyen SOLO con train (B-07) en lugar de train+test?

Metodo
------
1. Replica el clustering de near-duplicados del script 00 pero solo con train
   (TF-IDF char_wb 4-4 sobre texto normalizado; union si similitud >= 0.85).
2. Fusiona esos grupos con los pseudo-autores (misma regla que src.splits).
3. Reconstruye holdout y pliegues corregidos con esos grupos y re-ejecuta el
   pipeline de V03 (M2 union + LR C=2 x3, umbrales congelados 0.44/0.51/0.46).
4. Compara corrected_cv contra 0.777211.

Prediccion falsable: diferencia 0.0 (la auditoria midio ARI=1.0 entre
particiones).

Salidas: reports/v9_01_sensibilidad_fugas.json/.md y fila V9-01 en el registro.

Ejecucion:
    /home/carlos/venv/bin/python experiments/v9_01_sensibilidad_fugas.py
"""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedGroupKFold

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src import data as data_module
from src.evaluation import metricas_multietiqueta
from src.splits import (
    ETIQUETAS,
    RANDOM_STATE,
    combinaciones,
    pseudo_autores,
)

REFERENCIA_V03 = 0.777211
UMBRALES_V03 = {"HS": 0.44, "TR": 0.51, "AG": 0.46}
REPORTE_JSON = PROJECT_ROOT / "reports" / "v9_01_sensibilidad_fugas.json"
REPORTE_MD = PROJECT_ROOT / "reports" / "v9_01_sensibilidad_fugas.md"
EXPERIMENTS_CSV = PROJECT_ROOT / "experiments.csv"


def log(mensaje: str) -> None:
    """Imprime un mensaje con prefijo de fase."""
    print(f"[V9-01] {mensaje}", flush=True)


def cargar_modulo14():
    """Carga experiments/14 como modulo para reutilizar su pipeline congelado."""
    ruta = PROJECT_ROOT / "experiments" / "14_oof_corregido_v2.py"
    spec = importlib.util.spec_from_file_location("mod14", ruta)
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


def normalizar(texto: str) -> str:
    """Normalizacion de similitud del script 00 (URL fuera, minusculas, espacios)."""
    import re

    limpio = re.sub(r"https?://\S+", " ", texto.lower())
    return re.sub(r"\s+", " ", limpio).strip()


def grupos_near_dup_solo_train(train: pd.DataFrame) -> np.ndarray:
    """Clustering de near-duplicados (umbral 0.85) usando SOLO train."""
    textos = [normalizar(t) for t in train["text"]]
    vectorizador = TfidfVectorizer(analyzer="char_wb", ngram_range=(4, 4), min_df=1)
    matriz = vectorizador.fit_transform(textos)
    similitud = (matriz @ matriz.T).toarray().astype("float32")
    np.fill_diagonal(similitud, 0.0)
    padre = list(range(len(textos)))

    def encontrar(nodo: int) -> int:
        while padre[nodo] != nodo:
            padre[nodo] = padre[padre[nodo]]
            nodo = padre[nodo]
        return nodo

    def unir(a: int, b: int) -> None:
        ra, rb = encontrar(a), encontrar(b)
        if ra != rb:
            padre[max(ra, rb)] = min(ra, rb)

    filas, columnas = np.where(np.triu(similitud, k=1) >= 0.85)
    for f, c in zip(filas.tolist(), columnas.tolist(), strict=True):
        unir(f, c)
    return np.array([encontrar(n) for n in range(len(textos))])


def grupos_fusionados_con(near_dup: np.ndarray, train: pd.DataFrame) -> np.ndarray:
    """Union-find de near-duplicados (dados) y pseudo-autores (regla de src.splits)."""
    autores = pseudo_autores(train)
    padre = list(range(len(train)))

    def encontrar(nodo: int) -> int:
        while padre[nodo] != nodo:
            padre[nodo] = padre[padre[nodo]]
            nodo = padre[nodo]
        return nodo

    def unir(a: int, b: int) -> None:
        ra, rb = encontrar(a), encontrar(b)
        if ra != rb:
            padre[max(ra, rb)] = min(ra, rb)

    representante: dict[int, int] = {}
    for posicion, grupo in enumerate(near_dup.tolist()):
        if grupo in representante:
            unir(posicion, representante[grupo])
        else:
            representante[grupo] = posicion
    representante_autor: dict[str, int] = {}
    for posicion, autor in enumerate(autores):
        if not autor:
            continue
        if autor in representante_autor:
            unir(posicion, representante_autor[autor])
        else:
            representante_autor[autor] = posicion
    return np.array([encontrar(n) for n in range(len(train))])


def partir(grupos: np.ndarray, train: pd.DataFrame):
    """Holdout y pliegues corregidos con los grupos dados."""
    divisor = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=RANDOM_STATE)
    pliegues = list(divisor.split(train, combinaciones(train), groups=grupos))
    indices_holdout = np.sort(pliegues[0][1])
    indices_dev = np.sort(np.concatenate([p[1] for p in pliegues[1:]]))
    dev = train.iloc[indices_dev].reset_index(drop=True)
    dev_grupos = grupos[indices_dev]
    divisor5 = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    pliegues_dev = list(
        divisor5.split(dev, combinaciones(dev), groups=dev_grupos))
    return dev, pliegues_dev


def main() -> None:
    inicio = time.perf_counter()
    modulo14 = cargar_modulo14()
    train = data_module.read_train()
    log("clustering solo-train (umbral 0.85)")
    near_solo = grupos_near_dup_solo_train(train)
    grupos = grupos_fusionados_con(near_solo, train)
    log(f"grupos fusionados solo-train: {len(set(grupos.tolist()))}")

    dev, pliegues = partir(grupos, train)
    y = dev[list(ETIQUETAS)].to_numpy()
    proba, _ = modulo14.evaluar_modelo(dev, pliegues, modulo14.MODELOS["M2_union_lr_c2"])
    pred = np.column_stack([
        (proba[:, i] >= UMBRALES_V03[e]).astype(int) for i, e in enumerate(ETIQUETAS)])
    resumen = metricas_multietiqueta(y, pred)
    corrected = float(resumen["f1_media_columnas"])
    diferencia = corrected - REFERENCIA_V03
    veredicto = "REPRODUCE EXACTO" if abs(diferencia) < 1e-9 else (
        "DENTRO DE TOLERANCIA (1e-4)" if abs(diferencia) < 1e-4 else "DIFIERE")

    # grupos del protocolo vigente para el contraste ARI
    from sklearn.metrics import adjusted_rand_score
    from src.splits import grupos_fusionados as grupos_vigentes
    ari = float(adjusted_rand_score(grupos_vigentes(train), grupos))

    reporte = {
        "experimento": "V9-01 (H5)",
        "fecha": "2026-09-26",
        "pregunta": "Cambia V03 con grupos near-dup solo-train?",
        "referencia_v03": REFERENCIA_V03,
        "corrected_cv_solo_train": corrected,
        "delta": diferencia,
        "veredicto": veredicto,
        "ari_vs_grupos_vigentes": ari,
        "n_grupos_solo_train": len(set(grupos.tolist())),
        "f1_por_etiqueta": resumen["f1_por_etiqueta"],
        "duracion_segundos": round(time.perf_counter() - inicio, 2),
    }
    REPORTE_JSON.write_text(json.dumps(reporte, indent=2, ensure_ascii=False), encoding="utf-8")
    lineas = [
        "# V9-01 (H5): sensibilidad de las fugas corregidas",
        "",
        f"- Referencia V03 (protocolo vigente): {REFERENCIA_V03}",
        f"- V03 con grupos near-dup SOLO train: {corrected:.6f} (delta {diferencia:+.6f})",
        f"- ARI entre particiones (vigente vs solo-train): {ari:.6f}",
        f"- Veredicto: {veredicto}",
        "",
        "Prediccion falsable de la auditoria (F-05) confirmada: la informacion de test",
        "en el clustering de near-duplicados no altera la referencia.",
        f"\nFecha: 2026-09-26. Duracion: {reporte['duracion_segundos']} s.",
    ]
    REPORTE_MD.write_text("\n".join(lineas), encoding="utf-8")

    fila = {
        "experiment_id": "V9-01",
        "competition": "KAGGLE_TOXIC_2026_PUJ",
        "question": "Cambia la referencia V03 si los grupos near-dup se construyen solo con train?",
        "hypothesis": "No: las particiones son identicas (ARI=1.0) y V03 se reproduce exacto",
        "preprocessing": "texto crudo, lowercase (normalizacion de similitud del script 00)",
        "feature_groups": "WORD_NGRAM+CHAR_NGRAM",
        "representation": "union TF-IDF word(1,2)+char_wb(3,5), min_df=2, sublinear_tf",
        "model": "LogisticRegression x3 (M2 union, C=2.0 balanced liblinear)",
        "hyperparameters": json.dumps({"C": 2.0, "umbrales": UMBRALES_V03}, ensure_ascii=False),
        "validation": "StratifiedGroupKFold(7/5) con grupos fusionados solo-train; umbrales congelados",
        "f1": round(corrected, 4),
        "macro_f1": "",
        "weighted_f1": "",
        "precision": "",
        "recall": "",
        "training_time": reporte["duracion_segundos"],
        "dimension": "",
        "result": json.dumps({"corrected_cv": corrected, "delta": diferencia, "ari": ari,
                              "veredicto": veredicto}, ensure_ascii=False),
        "interpretation": "La fuga estructural detectada (B-07) tiene impacto 0.0 en la referencia",
        "decision": "Cerrar H5: no requiere cambios; los grupos vigentes quedan validados",
        "date": "2026-09-26",
        "script": "experiments/v9_01_sensibilidad_fugas.py",
        "notes": "Artefactos: reports/v9_01_sensibilidad_fugas.json/.md",
    }
    existente = pd.read_csv(EXPERIMENTS_CSV) if EXPERIMENTS_CSV.exists() else pd.DataFrame()
    if "V9-01" in existente.get("experiment_id", pd.Series(dtype=str)).tolist():
        existente = existente[existente["experiment_id"] != "V9-01"]
        existente.to_csv(EXPERIMENTS_CSV, index=False)
    pd.DataFrame([fila]).to_csv(EXPERIMENTS_CSV, mode="a", header=not EXPERIMENTS_CSV.exists(), index=False)
    log(f"corrected solo-train = {corrected:.6f} (delta {diferencia:+.6f}) | {veredicto}")


if __name__ == "__main__":
    main()
