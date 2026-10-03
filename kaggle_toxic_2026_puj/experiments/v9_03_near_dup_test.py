"""V9-03 (H2): propagacion de etiquetas train->test por near-duplicados.

Pregunta
--------
El 3 % de los pares train-test del EDA tienen similitud >= 0.85 (copias). ¿Vale
la pena sustituir la prediccion de esos tuits de test por la etiqueta de su copia
de train?

Metodo
------
1. Se recomputan los pares near-duplicados train-test con el metodo del script 00
   (TF-IDF char_wb 4-4 sobre texto normalizado) y se listan con su similitud.
2. Para cada par se compara la etiqueta de train de la copia con la prediccion de
   V03 en el tuit de test (diagnostico de acuerdo por etiqueta).
3. Sonda v18: V03 con las filas de test del nivel A (similitud >= 0.85)
   sustituidas por las etiquetas de su copia de train.
4. Se cuantifican filas afectadas y celdas que cambian.

Salidas: reports/v9_03_near_dup_test.json/.md/.csv,
submissions/toxic_submission_v18_near_dup_propagacion.csv y fila V9-03.

Ejecucion:
    /home/carlos/venv/bin/python experiments/v9_03_near_dup_test.py
"""

from __future__ import annotations

import csv
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src import data as data_module

ETIQUETAS = ("HS", "TR", "AG")
UMBRAL_NIVEL_A = 0.85
UMBRAL_NIVEL_B = 0.80
REPORTE_JSON = PROJECT_ROOT / "reports" / "v9_03_near_dup_test.json"
REPORTE_MD = PROJECT_ROOT / "reports" / "v9_03_near_dup_test.md"
REPORTE_CSV = PROJECT_ROOT / "reports" / "v9_03_near_dup_test.csv"
SUBMISSION = PROJECT_ROOT / "submissions" / "toxic_submission_v18_near_dup_propagacion.csv"
EXPERIMENTS_CSV = PROJECT_ROOT / "experiments.csv"


def log(mensaje: str) -> None:
    """Imprime un mensaje con prefijo de fase."""
    print(f"[V9-03] {mensaje}", flush=True)


def normalizar(texto: str) -> str:
    """Normalizacion de similitud del script 00."""
    limpio = re.sub(r"https?://\S+", " ", texto.lower())
    return re.sub(r"\s+", " ", limpio).strip()


def main() -> None:  # noqa: PLR0915
    inicio = time.perf_counter()
    train = data_module.read_train()
    test = data_module.read_test()
    train["id"] = train["id"].astype(int)
    test["id"] = test["id"].astype(int)
    v03 = pd.read_csv(PROJECT_ROOT / "submissions" / "toxic_submission_v03_lr_union_thr.csv")
    v03 = v03.set_index("id")

    textos = [normalizar(t) for t in train["text"]] + [normalizar(t) for t in test["text"]]
    vectorizador = TfidfVectorizer(analyzer="char_wb", ngram_range=(4, 4), min_df=1)
    matriz = vectorizador.fit_transform(textos)
    n_train = len(train)
    similitud = (matriz[:n_train] @ matriz[n_train:].T).toarray()

    filas_idx, cols_idx = np.where(similitud >= UMBRAL_NIVEL_B)
    pares = []
    for i, j in zip(filas_idx.tolist(), cols_idx.tolist(), strict=True):
        sim = float(similitud[i, j])
        if sim < UMBRAL_NIVEL_B:
            continue
        id_train = int(train["id"].iloc[i])
        id_test = int(test["id"].iloc[j])
        pred = {e: int(v03.loc[id_test, e]) for e in ETIQUETAS}
        etiqueta = {e: int(train[e].iloc[i]) for e in ETIQUETAS}
        acuerdos = [int(etiqueta[e] == pred[e]) for e in ETIQUETAS]
        pares.append({
            "id_train": id_train, "id_test": id_test, "similitud": round(sim, 4),
            "nivel": "A" if sim >= UMBRAL_NIVEL_A else "B",
            **{f"y_train_{e}": etiqueta[e] for e in ETIQUETAS},
            **{f"pred_v03_{e}": pred[e] for e in ETIQUETAS},
            "celdas_acuerdo": int(sum(acuerdos)),
        })
    pares.sort(key=lambda p: -p["similitud"])

    filas_a = [p for p in pares if p["nivel"] == "A"]
    filas_b = [p for p in pares if p["nivel"] == "B"]
    ids_test_a = sorted({p["id_test"] for p in filas_a})
    log(f"pares >= {UMBRAL_NIVEL_A}: {len(filas_a)} (nivel A); "
        f"pares {UMBRAL_NIVEL_B}-{UMBRAL_NIVEL_A}: {len(filas_b)} (nivel B); "
        f"tuits de test afectados (A): {len(ids_test_a)}")
    celdas_acuerdo = sum(p["celdas_acuerdo"] for p in filas_a)
    log(f"acuerdo train-copia vs V03 en nivel A: {celdas_acuerdo}/"
        f"{3 * len(filas_a)} celdas")

    # Sonda v18: propagacion nivel A
    marco = v03.copy()
    propagadas = 0
    for p in filas_a:
        fila = marco.loc[p["id_test"]]
        for e in ETIQUETAS:
            etiqueta = p[f"y_train_{e}"]
            if int(fila[e]) != etiqueta:
                propagadas += 1
            fila[e] = etiqueta
            marco.loc[p["id_test"], e] = etiqueta
    marco = marco.reset_index()
    marco.to_csv(SUBMISSION, index=False, lineterminator="\n")
    celdas_cambian = int((marco[list(ETIQUETAS)].to_numpy()
                          != v03.reset_index()[list(ETIQUETAS)].to_numpy()).sum())
    integridad = {
        "filas": len(marco), "columnas": list(marco.columns),
        "ids_iguales": bool((marco["id"].astype(int)
                             == v03.reset_index()["id"].astype(int)).all()),
        "binario": bool(set(marco[list(ETIQUETAS)].to_numpy().ravel()) <= {0, 1}),
    }

    # Diagnostico global (todos los pares)
    total_celdas = 3 * len(pares)
    acuerdo_global = sum(p["celdas_acuerdo"] for p in pares) / max(total_celdas, 1)
    reporte = {
        "experimento": "V9-03 (H2)",
        "fecha": "2026-09-26",
        "umbrales": {"nivel_A": UMBRAL_NIVEL_A, "nivel_B": UMBRAL_NIVEL_B},
        "n_pares_nivel_A": len(filas_a),
        "n_pares_nivel_B": len(filas_b),
        "tuits_test_afectados_A": len(ids_test_a),
        "acuerdo_global_celdas": round(float(acuerdo_global), 4),
        "acuerdo_nivel_A_celdas": round(celdas_acuerdo / max(3 * len(filas_a), 1), 4),
        "celdas_que_cambian_vs_v03": celdas_cambian,
        "submission_sonda": str(SUBMISSION.relative_to(PROJECT_ROOT)),
        "integridad_submission": integridad,
        "pares": pares,
        "duracion_segundos": round(time.perf_counter() - inicio, 2),
    }
    REPORTE_JSON.write_text(json.dumps(reporte, indent=2, ensure_ascii=False), encoding="utf-8")
    with REPORTE_CSV.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(pares[0].keys()) if pares else [])
        if pares:
            writer.writeheader()
            writer.writerows(pares)
    lineas = [
        "# V9-03 (H2): propagacion train->test por near-duplicados",
        "",
        f"- Pares nivel A (>= {UMBRAL_NIVEL_A}): {len(filas_a)}; tuits de test afectados: "
        f"{len(ids_test_a)}",
        f"- Pares nivel B ({UMBRAL_NIVEL_B}-{UMBRAL_NIVEL_A}): {len(filas_b)}",
        f"- Acuerdo celdas nivel A (etiqueta de la copia vs V03): "
        f"{reporte['acuerdo_nivel_A_celdas']:.2%}",
        f"- Celdas que cambian en la sonda v18: {celdas_cambian}",
        "",
        "## Pares nivel A",
        "",
        "| id_train | id_test | sim | y_train | V03 | acuerdo |",
        "|---|---|---|---|---|---|",
    ]
    for p in filas_a:
        y = "/".join(str(p[f"y_train_{e}"]) for e in ETIQUETAS)
        v = "/".join(str(p[f"pred_v03_{e}"]) for e in ETIQUETAS)
        lineas.append(f"| {p['id_train']} | {p['id_test']} | {p['similitud']:.3f} | "
                      f"{y} | {v} | {p['celdas_acuerdo']}/3 |")
    lineas += [
        "",
        f"Submission sonda: `{reporte['submission_sonda']}` (integridad: {integridad}).",
        "Lectura: el efecto maximo posible esta acotado por el numero de celdas que cambian;",
        "la decision de usar la copia como etiqueta depende del acuerdo observado.",
        f"\nFecha: 2026-09-26. Duracion: {reporte['duracion_segundos']} s.",
    ]
    REPORTE_MD.write_text("\n".join(lineas), encoding="utf-8")

    fila_reg = {
        "experiment_id": "V9-03",
        "competition": "KAGGLE_TOXIC_2026_PUJ",
        "question": "Mejora propagar la etiqueta de train a los tuits de test con copia casi identica?",
        "hypothesis": "Las copias train-test (>=0.85) permiten sustituir la prediccion por la etiqueta de la copia",
        "preprocessing": "texto normalizado (URL fuera, minusculas)",
        "feature_groups": "SIMILITUD_CHAR",
        "representation": "TF-IDF char_wb(4,4) de la copia (metodo del script 00)",
        "model": "capa de decision sobre V03 (sin entrenamiento)",
        "hyperparameters": json.dumps({"nivel_A": UMBRAL_NIVEL_A, "nivel_B": UMBRAL_NIVEL_B},
                                      ensure_ascii=False),
        "validation": "diagnostico de acuerdo por par (etiqueta de la copia vs prediccion V03)",
        "f1": "",
        "macro_f1": "", "weighted_f1": "", "precision": "", "recall": "",
        "training_time": reporte["duracion_segundos"],
        "dimension": "",
        "result": json.dumps({"n_pares_A": len(filas_a), "n_pares_B": len(filas_b),
                              "tuits_afectados": len(ids_test_a),
                              "acuerdo_A": reporte["acuerdo_nivel_A_celdas"],
                              "celdas_que_cambian": celdas_cambian}, ensure_ascii=False),
        "interpretation": "Efecto maximo acotado a pocas celdas; el diagnostico decide si conviene",
        "decision": "Sonda v18 generada y pre-registrada; no adoptada ni enviada",
        "date": "2026-09-26",
        "script": "experiments/v9_03_near_dup_test.py",
        "notes": "Artefactos: reports/v9_03_near_dup_test.json/.md/.csv; submissions/toxic_submission_v18_near_dup_propagacion.csv",
    }
    existente = pd.read_csv(EXPERIMENTS_CSV) if EXPERIMENTS_CSV.exists() else pd.DataFrame()
    if "V9-03" in existente.get("experiment_id", pd.Series(dtype=str)).tolist():
        existente = existente[existente["experiment_id"] != "V9-03"]
        existente.to_csv(EXPERIMENTS_CSV, index=False)
    pd.DataFrame([fila_reg]).to_csv(EXPERIMENTS_CSV, mode="a",
                                    header=not EXPERIMENTS_CSV.exists(), index=False)
    log(f"sonda v18 escrita ({celdas_cambian} celdas cambian frente a V03)")


if __name__ == "__main__":
    main()
