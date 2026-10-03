"""V9-05 (H3): representacion supervisada de bajo rango (direcciones discriminantes).

Pregunta
--------
¿Mejora una compresion supervisada fold-local del TF-IDF (direcciones
discriminantes) frente a la compresion no supervisada (SVD) del mismo rango?

Metodo
------
1. Por pliegue: TF-IDF union ajustado en el entrenamiento.
2. Direcciones discriminantes por etiqueta: d_i = media_pos - media_neg
   (normalizada); proyecciones p_i = X . d_i y sus productos cruzados -> 6
   features; LR C=1 balanced.
3. Control simetrico: TruncatedSVD(6) ajustado en el entrenamiento -> LR.
4. Umbrales congelados de V03; protocolo corregido.

Salidas: reports/v9_05_proyecciones_supervisadas.json/.md y fila V9-05.

Ejecucion:
    /home/carlos/venv/bin/python experiments/v9_05_proyecciones_supervisadas.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src import data as data_module
from src.splits import ETIQUETAS, grupos_fusionados, holdout_v2, pliegues_corregidos

REFERENCIA_V03 = 0.7772114525165721
UMBRALES_V03 = {"HS": 0.44, "TR": 0.51, "AG": 0.46}
PARAMS_WORD = {"analyzer": "word", "ngram_range": (1, 2), "min_df": 2, "sublinear_tf": True}
PARAMS_CHAR = {"analyzer": "char_wb", "ngram_range": (3, 5), "min_df": 2, "sublinear_tf": True}
REPORTE_JSON = PROJECT_ROOT / "reports" / "v9_05_proyecciones_supervisadas.json"
REPORTE_MD = PROJECT_ROOT / "reports" / "v9_05_proyecciones_supervisadas.md"
EXPERIMENTS_CSV = PROJECT_ROOT / "experiments.csv"


def log(mensaje: str) -> None:
    """Imprime un mensaje con prefijo de fase."""
    print(f"[V9-05] {mensaje}", flush=True)


def evaluar(y: np.ndarray, proba: np.ndarray) -> float:
    """F1 media column-wise con umbrales congelados."""
    valores = []
    for i, etiqueta in enumerate(ETIQUETAS):
        pred = (proba[:, i] >= UMBRALES_V03[etiqueta]).astype(int)
        tp = float(np.sum((y[:, i] == 1) & (pred == 1)))
        fp = float(np.sum((y[:, i] == 0) & (pred == 1)))
        fn = float(np.sum((y[:, i] == 1) & (pred == 0)))
        den = 2 * tp + fp + fn
        valores.append(2 * tp / den if den > 0 else 0.0)
    return float(np.mean(valores))


def main() -> None:  # noqa: PLR0915
    inicio = time.perf_counter()
    train = data_module.read_train()
    grupos = grupos_fusionados(train)
    indices_dev, _ = holdout_v2(train, grupos)
    dev = train.iloc[indices_dev].reset_index(drop=True)
    pliegues = pliegues_corregidos(dev, grupos[indices_dev])
    y = dev[list(ETIQUETAS)].to_numpy()

    proba_sdr = np.zeros_like(y, dtype=float)
    proba_svd = np.zeros_like(y, dtype=float)
    for k, (tr, va) in enumerate(pliegues):
        word = TfidfVectorizer(**PARAMS_WORD)
        char = TfidfVectorizer(**PARAMS_CHAR)
        x_tr = sparse.hstack([word.fit_transform(dev["text"].iloc[tr]),
                              char.fit_transform(dev["text"].iloc[tr])]).tocsr()
        x_va = sparse.hstack([word.transform(dev["text"].iloc[va]),
                              char.transform(dev["text"].iloc[va])]).tocsr()
        # Direcciones discriminantes fold-local
        direcciones = []
        for i in range(3):
            positivos = np.asarray(x_tr[y[tr, i] == 1].sum(axis=0)).ravel()
            negativos = np.asarray(x_tr[y[tr, i] == 0].sum(axis=0)).ravel()
            n_pos = max(int((y[tr, i] == 1).sum()), 1)
            n_neg = max(int((y[tr, i] == 0).sum()), 1)
            direccion = positivos / n_pos - negativos / n_neg
            norma = np.linalg.norm(direccion) + 1e-12
            direcciones.append(direccion / norma)
        proyecciones_tr = np.column_stack([x_tr @ d for d in direcciones])
        proyecciones_va = np.column_stack([x_va @ d for d in direcciones])
        def ampliar(p):
            return np.column_stack([p, p[:, 0] * p[:, 1], p[:, 0] * p[:, 2], p[:, 1] * p[:, 2]])
        sdr_tr, sdr_va = ampliar(proyecciones_tr), ampliar(proyecciones_va)
        # Control SVD no supervisada del mismo rango
        svd = TruncatedSVD(n_components=6, random_state=42)
        svd_tr = svd.fit_transform(x_tr)
        svd_va = svd.transform(x_va)
        for i in range(3):
            modelo = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000,
                                        random_state=42)
            modelo.fit(sdr_tr, y[tr, i])
            proba_sdr[va, i] = modelo.predict_proba(sdr_va)[:, 1]
            modelo = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000,
                                        random_state=42)
            modelo.fit(svd_tr, y[tr, i])
            proba_svd[va, i] = modelo.predict_proba(svd_va)[:, 1]
        log(f"  pliegue {k + 1} listo")

    f1_sdr = evaluar(y, proba_sdr)
    f1_svd = evaluar(y, proba_svd)
    veredicto = ("SDR SUPERA A SVD" if f1_sdr > f1_svd + 1e-9 else
                 ("SVD SUPERA A SDR" if f1_svd > f1_sdr + 1e-9 else "EMPATE"))
    reporte = {
        "experimento": "V9-05 (H3)",
        "fecha": "2026-09-26",
        "sdr_6_features_umbrales_v03": f1_sdr,
        "svd_6_componentes_umbrales_v03": f1_svd,
        "referencia_v03": REFERENCIA_V03,
        "delta_sdr_vs_svd": round(f1_sdr - f1_svd, 6),
        "delta_mejor_vs_v03": round(max(f1_sdr, f1_svd) - REFERENCIA_V03, 6),
        "veredicto": veredicto,
        "duracion_segundos": round(time.perf_counter() - inicio, 2),
    }
    REPORTE_JSON.write_text(json.dumps(reporte, indent=2, ensure_ascii=False), encoding="utf-8")
    lineas = [
        "# V9-05 (H3): proyecciones supervisadas de bajo rango vs SVD",
        "",
        "| compresion (6 dims, fold-local) | F1 con umbrales V03 |",
        "|---|---|",
        f"| Direcciones discriminantes (SDR) | {f1_sdr:.4f} |",
        f"| SVD no supervisada (control) | {f1_svd:.4f} |",
        f"| V03 (24646 features) | {REFERENCIA_V03:.6f} |",
        "",
        f"Veredicto: **{veredicto}** (delta SDR-SVD {f1_sdr - f1_svd:+.4f}; mejor vs V03 "
        f"{max(f1_sdr, f1_svd) - REFERENCIA_V03:+.4f}).",
        f"\nFecha: 2026-09-26. Duracion: {reporte['duracion_segundos']} s.",
    ]
    REPORTE_MD.write_text("\n".join(lineas), encoding="utf-8")

    fila_reg = {
        "experiment_id": "V9-05",
        "competition": "KAGGLE_TOXIC_2026_PUJ",
        "question": "Mejora una compresion supervisada fold-local del TF-IDF frente a SVD del mismo rango?",
        "hypothesis": "Las direcciones discriminantes concentran la senal supervisada que SVD ignora",
        "preprocessing": "texto crudo (pipeline congelado de V03)",
        "feature_groups": "WORD_NGRAM+CHAR_NGRAM proyectado a 6 dims",
        "representation": "direcciones discriminantes (media_pos - media_neg) vs SVD(6)",
        "model": "LogisticRegression C=1.0 balanced",
        "hyperparameters": json.dumps({"dims": 6, "C": 1.0}, ensure_ascii=False),
        "validation": "protocolo corregido 5 pliegues; umbrales congelados V03",
        "f1": round(max(f1_sdr, f1_svd), 4),
        "macro_f1": "", "weighted_f1": "", "precision": "", "recall": "",
        "training_time": reporte["duracion_segundos"],
        "dimension": "6",
        "result": json.dumps({"sdr": f1_sdr, "svd": f1_svd,
                              "delta": f1_sdr - f1_svd, "veredicto": veredicto},
                             ensure_ascii=False),
        "interpretation": "Test directo de compresion supervisada vs no supervisada a rango igual",
        "decision": "Sin adopcion (resultado registrado)",
        "date": "2026-09-26",
        "script": "experiments/v9_05_proyecciones_supervisadas.py",
        "notes": "Artefactos: reports/v9_05_proyecciones_supervisadas.json/.md",
    }
    existente = pd.read_csv(EXPERIMENTS_CSV) if EXPERIMENTS_CSV.exists() else pd.DataFrame()
    if "V9-05" in existente.get("experiment_id", pd.Series(dtype=str)).tolist():
        existente = existente[existente["experiment_id"] != "V9-05"]
        existente.to_csv(EXPERIMENTS_CSV, index=False)
    pd.DataFrame([fila_reg]).to_csv(EXPERIMENTS_CSV, mode="a",
                                    header=not EXPERIMENTS_CSV.exists(), index=False)
    log(f"SDR {f1_sdr:.4f} vs SVD {f1_svd:.4f} vs V03 {REFERENCIA_V03:.6f} -> {veredicto}")


if __name__ == "__main__":
    main()
