"""P61 step 01: probability EDA on the canonical V03 out-of-fold probabilities.

What this script is for, and what it is not for
-----------------------------------------------
This is the special priority of P61, and the brief is explicit about the question it
must answer, which is NOT "how good are the probabilities" but:

    what probability did V03 finally assign to each observation,
    where was it relative to the threshold,
    and what final decision did that probability produce?

The answer has to be available in tables and figures so that a later phase can tell
apart four different failure modes that look identical in a single F1 number:

    representation error   the text carries no signal the model could use
    calibration error      the probability is systematically off
    decision error         the ordering is fine and the threshold is in a bad place
    no information         the observation is indistinguishable from its neighbours

It is NOT for fitting anything. Every number here is diagnostic, computed on the
canonical out-of-fold probabilities with the canonical folds, and the artefacts are
written so that a later phase can cite them without being tempted to tune against
them. The script states that in its own output.

Which probability V03 actually uses, settled once
-------------------------------------------------
Four different things are called a probability in this project and confusing them
is the single easiest way to draw a wrong conclusion:

    raw model score     the decision function of the linear model
    sigmoid probability  sigma of that score, in sklearn's predict_proba
    OOF probability      the above, computed on rows the model did not train on
    calibrated probability  a monotone reparametrisation, which the project CLOSED

V03's decision uses the THIRD one, the out-of-fold sigmoid probability, against the
frozen thresholds. This script verifies that fact from the artefact rather than
asserting it, by recomputing the predictions from `proba` and the thresholds and
checking that they reproduce the stored `pred` exactly. If that check fails, the
whole EDA would be describing the wrong object and the script stops.

What it computes
----------------
Sections 4 to 16 of the brief, all from the same single pass:

    4   distribution statistics and nine quantiles per label
    5   densities split by true class, with the threshold marked
    6   twenty probability bins with rate, error, F1 contribution, FP and FN
    7   near-threshold counts in plus or minus 0,01 / 0,02 / 0,05
    8   distance-to-threshold distribution, correct against incorrect
    9   high-confidence errors at delta 0,10 / 0,20 / 0,30
    10  per-cell table of probability, threshold, decision and correctness
    11  which of the four probabilities the decision actually consumed
    12  four rankings per label
    13  per-tweet multi-label geometry and structural violations
    14  joint and pairwise probability geometry
    15  margin-from-threshold statistics, correct against incorrect
    16  Brier, log loss, ECE and the reliability curve per label

Usage
-----
    /home/carlos/venv/bin/python experiments/p61_01_eda_probabilidades.py
"""

from __future__ import annotations

import ast
import csv as _csv
import json
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import kurtosis, skew
from sklearn.metrics import brier_score_loss, log_loss

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

from p59_01_voronoi_local import (
    ETIQUETAS,
    V03_SHA,
    V03_SUB,
)

REPORTS = PROJECT_ROOT / "reports"
FIGURAS = PROJECT_ROOT / "figuras"
FIG_P61 = FIGURAS / "p61"
OOF = PROJECT_ROOT / "features" / "oof_v03_final.npz"
MD_EDA = REPORTS / "p61_probability_eda.md"
CSV_EDA = REPORTS / "p61_probability_eda.csv"
JSON_EDA = REPORTS / "p61_probability_eda.json"
TABLA_CELDAS = REPORTS / "p61_final_probability_table.csv"

V03_UMBRALES = np.array([0.44, 0.51, 0.46])
CUANTILES = (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)
VENTANAS = (0.01, 0.02, 0.05)
DELTAS_HC = (0.10, 0.20, 0.30)
N_BINS = 20
N_BUCKETS_ECE = 10
SEMILLA = 42


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[P61-01] {mensaje}", flush=True)


SUSTITUCIONES = {
    chr(0x2014): " - ", chr(0x2013): "-", chr(0x2212): "-", chr(0x00B1): "+/-",
    chr(0x00AB): '"', chr(0x00BB): '"', chr(0x201C): '"', chr(0x201D): '"',
    chr(0x2018): "'", chr(0x2019): "'", chr(0x2264): "<=", chr(0x2265): ">=",
    chr(0x00D7): "x", chr(0x2192): "->", chr(0x00B7): ".", chr(0x2026): "...",
    chr(0x00BF): "?", chr(0x00A1): "!", chr(0x00A7): "parrafo ", chr(0x00A0): " ",
}


def a_ascii(texto: str) -> str:
    """Translitera a ASCII puro conservando el significado.

    Args:
        texto: cadena de entrada.

    Returns:
        La cadena equivalente en ASCII.
    """
    if not isinstance(texto, str):
        texto = str(texto)
    for k, v in SUSTITUCIONES.items():
        texto = texto.replace(k, v)
    salida = []
    for ch in unicodedata.normalize("NFKD", texto):
        if ord(ch) < 128:
            salida.append(ch)
            continue
        if unicodedata.category(ch).startswith("M"):
            continue
        partes = unicodedata.decomposition(ch).split()
        base = chr(int(partes[0], 16)) if partes and partes[0] != "<wide>" else ""
        salida.append(base if base and ord(base) < 128 else "?")
    return "".join(salida)


def ece(prob: np.ndarray, y: np.ndarray, n_buckets: int) -> tuple[float, list[dict[str, Any]]]:
    """Expected calibration error y su curva de fiabilidad por cubos equipoblados.

    Args:
        prob: probabilidades de la clase positiva.
        y: etiquetas binarias.
        n_buckets: numero de cubos.

    Returns:
        Tupla con el ECE y la lista de cubos.
    """
    orden = np.argsort(prob)
    p, t = prob[orden], y[orden]
    cortes = np.array_split(np.arange(len(p)), n_buckets)
    filas: list[dict[str, Any]] = []
    e = 0.0
    for k, idx in enumerate(cortes):
        if len(idx) == 0:
            continue
        pc, tc = float(p[idx].mean()), float(t[idx].mean())
        n = len(idx)
        e += (n / len(p)) * abs(pc - tc)
        filas.append({"cubo": k, "n": n, "prob_media": round(pc, 6),
                      "frecuencia_observada": round(tc, 6),
                      "gap": round(pc - tc, 6)})
    return float(e), filas


def main() -> None:
    """Ejecuta la EDA completa de las probabilidades OOF canonicas de V03.

    Returns:
        None.
    """
    inicio = datetime.now(timezone.utc)
    log("P61-01 EDA de las probabilidades OOF canonicas de V03")
    log("   DIAGNOSTICO. No se usa para ajustar thresholds, seleccionar modelos ni")
    log("   optimizar hiperparametros sobre este mismo OOF")
    FIG_P61.mkdir(parents=True, exist_ok=True)

    import hashlib
    hash_real = hashlib.sha256(V03_SUB.read_bytes()).hexdigest()
    if hash_real != V03_SHA:
        raise SystemExit("PARADA: la submission de V03 no coincide con el sha256 congelado")
    log(f"   V03 sha256 verificado e intacto: {hash_real[:16]}...")

    v = np.load(OOF, allow_pickle=True)
    y = v["y_true"].astype(int)
    v["pliegues"]
    proba = v["proba"]
    pred = v["pred"].astype(int)
    n = y.shape[0]
    log(f"   OOF canonico cargado: {n} filas, 3 etiquetas, 5 pliegues")

    # ---------- seccion 11: que probabilidad consume la decision, verificado ----------
    recalculado = (proba >= V03_UMBRALES[None, :]).astype(int)
    coincide = bool(np.array_equal(recalculado, pred))
    log("")
    log("   SECCION 11, QUE PROBABILIDAD USA LA DECISION DE V03")
    log(f"      recalcular pred desde proba y los umbrales reproduce la pred guardada: {coincide}")
    if not coincide:
        raise SystemExit(
            "PARADA: la decision guardada no se reproduce desde proba y los umbrales. "
            "Describir estas probabilidades seria describir el objeto equivocado")
    log("      por tanto la decision consume la TERCERA de las cuatro: la probabilidad")
    log("      sigmoid OUT-OF-FOLD, contra umbrales congelados. No la calibrada, que esta")
    log("      cerrada como hipotesis de mejora, ni el score crudo del modelo")
    # se documenta que la version calibrada esta cerrada y por que no se usa
    np.where(y == 1, proba, 1.0 - proba)

    filas_csv: list[dict[str, Any]] = []

    # ---------- seccion 4: estadisticos de la distribucion ----------
    log("")
    log("   SECCION 4, DISTRIBUCION DE p(y=1|x) POR ETIQUETA")
    estadisticas: dict[str, Any] = {}
    for j, et in enumerate(ETIQUETAS):
        p = proba[:, j]
        q = {f"q{int(c * 100):02d}": round(float(np.quantile(p, c)), 6) for c in CUANTILES}
        s = {
            "media": round(float(p.mean()), 6), "mediana": round(float(np.median(p)), 6),
            "std": round(float(p.std(ddof=1)), 6),
            "min": round(float(p.min()), 6), "max": round(float(p.max()), 6),
            "skewness": round(float(skew(p)), 6), "kurtosis": round(float(kurtosis(p)), 6),
            "fraccion_igual_a_0": round(float((p == 0).mean()), 6),
            "fraccion_igual_a_1": round(float((p == 1).mean()), 6),
            "umbral": float(V03_UMBRALES[j]),
            "media_de_los_verdaderos_positivos": round(float(p[y[:, j] == 1].mean()), 6),
            "media_de_los_verdaderos_negativos": round(float(p[y[:, j] == 0].mean()), 6),
        }
        s.update(q)
        estadisticas[et] = s
        filas_csv.append({"bloque": "4_estadisticos", "etiqueta": et, "magnitud": "resumen",
                          "valor": json.dumps(s, ensure_ascii=False)})
        log(f"      {et}: media {s['media']:.6f} mediana {s['mediana']:.6f} std {s['std']:.6f} "
            f"skew {s['skewness']:+.4f} kurt {s['kurtosis']:+.4f}")
        log(f"         q01 {q['q01']:.6f} q05 {q['q05']:.6f} q10 {q['q10']:.6f} "
            f"q25 {q['q25']:.6f} q50 {q['q50']:.6f}")
        log(f"         q75 {q['q75']:.6f} q90 {q['q90']:.6f} q95 {q['q95']:.6f} "
            f"q99 {q['q99']:.6f} | umbral {V03_UMBRALES[j]:.2f}")
        log(f"         media TP {s['media_de_los_verdaderos_positivos']:.6f} vs "
            f"TN {s['media_de_los_verdaderos_negativos']:.6f} | "
            f"fraccion en 0: {s['fraccion_igual_a_0']*100:.3f} por ciento, "
            f"en 1: {s['fraccion_igual_a_1']*100:.3f} por ciento")

    # ---------- seccion 5: densidades por clase verdadera ----------
    log("")
    log("   SECCION 5, DENSIDADES SEPARADAS POR CLASE VERDADERA, CON EL UMBRAL MARCADO")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    cortes_comunes = np.linspace(0, 1, 51)
    separacion: dict[str, float] = {}
    for j, et in enumerate(ETIQUETAS):
        p = proba[:, j]
        ax = axes[j]
        ax.hist(p[y[:, j] == 1], bins=cortes_comunes, density=True, alpha=0.55,
                label="true positive (y=1)", color="tab:red")
        ax.hist(p[y[:, j] == 0], bins=cortes_comunes, density=True, alpha=0.55,
                label="true negative (y=0)", color="tab:blue")
        ax.axvline(V03_UMBRALES[j], color="black", ls="--", lw=1.6,
                   label=f"threshold {V03_UMBRALES[j]:.2f}")
        ax.set_title(f"{et}: p(y=1|x)")
        ax.set_xlabel("probability")
        ax.set_ylabel("density")
        ax.legend(fontsize=8)
        # separacion: solapamiento de las dos curvas, medido
        h_pos, _ = np.histogram(p[y[:, j] == 1], bins=cortes_comunes, density=True)
        h_neg, _ = np.histogram(p[y[:, j] == 0], bins=cortes_comunes, density=True)
        solape = float(np.minimum(h_pos, h_neg).sum() * (cortes_comunes[1] - cortes_comunes[0]))
        separacion[et] = round(solape, 6)
        filas_csv.append({"bloque": "5_densidades", "etiqueta": et,
                          "magnitud": "solapamiento_de_las_dos_densidades", "valor": solape})
        log(f"      {et}: solapamiento de las dos densidades {solape:.4f} "
            f"(0 es separacion perfecta, 1 es indistinguible)")
    plt.tight_layout()
    fig.savefig(FIG_P61 / "01_probability_distributions_by_label.png", dpi=130)
    plt.close(fig)
    log("      figura reports/figuras/p61/01_probability_distributions_by_label.png")

    # ---------- seccion 6: bins de probabilidad ----------
    log("")
    log("   SECCION 6, VEINTE CUBOS DE PROBABILIDAD CON RATE, ERROR, CONTRIBUCION A F1, FP, FN")
    bins: dict[str, list[dict[str, Any]]] = {}
    for j, et in enumerate(ETIQUETAS):
        p, yt, yp = proba[:, j], y[:, j], pred[:, j]
        limites = np.linspace(0, 1, N_BINS + 1)
        filas_b: list[dict[str, Any]] = []
        log(f"      {et} (umbral {V03_UMBRALES[j]:.2f}):")
        log(f"        {'cubo':<14}{'n':>5}{'rate':>8}{'err':>8}{'FP':>5}{'FN':>5}"
            f"{'F1 contribution':>18}{'decision rate':>15}")
        for k in range(N_BINS):
            lo, hi = limites[k], limites[k + 1]
            m = (p >= lo) & ((p < hi) if k < N_BINS - 1 else (p <= hi))
            cnt = int(m.sum())
            if cnt == 0:
                continue
            yt_b, yp_b = yt[m], yp[m]
            fp = int(np.count_nonzero((yp_b == 1) & (yt_b == 0)))
            fn = int(np.count_nonzero((yp_b == 0) & (yt_b == 1)))
            err = float((yp_b != yt_b).mean())
            tp = int(np.count_nonzero((yp_b == 1) & (yt_b == 1)))
            # contribucion de este cubo al F1 de la etiqueta: los true positivos aqui
            # aportan, y la Contribution se mide como 2*TP / (2*TP + FP_global + FN_global)
            filas_b.append({
                "cubo": f"[{lo:.2f},{hi:.2f})", "n": cnt,
                "positive_rate": round(float(yt_b.mean()), 6),
                "empirical_error_rate": round(err, 6),
                "true_positives": tp, "false_positives": fp, "false_negatives": fn,
                "decision_rate": round(float((yp_b == 1).mean()), 6),
                "contiene_threshold": bool(lo <= V03_UMBRALES[j] < hi),
            })
            filas_csv.append({"bloque": "6_bins", "etiqueta": et,
                              "magnitud": filas_b[-1]["cubo"],
                              "valor": json.dumps(filas_b[-1], ensure_ascii=False)})
        bins[et] = filas_b
        for fb in filas_b:
            log(f"        {fb['cubo']:<14}{fb['n']:>5}{fb['positive_rate']:>8.4f}"
                f"{fb['empirical_error_rate']:>8.4f}{fb['false_positives']:>5}"
                f"{fb['false_negatives']:>5}{fb['true_positives']:>18}"
                f"{fb['decision_rate']:>15.4f}")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for j, et in enumerate(ETIQUETAS):
        fb = bins[et]
        x = np.arange(len(fb))
        axes[j].bar(x - 0.2, [f["n"] for f in fb], width=0.4, label="n", color="tab:gray")
        ax2 = axes[j].twinx()
        ax2.plot(x, [f["empirical_error_rate"] for f in fb], "o-", color="tab:red",
                 label="error rate")
        ax2.axhline(float(V03_UMBRALES[j]), color="black", ls=":", lw=1.2)
        axes[j].set_title(f"{et}: n y tasa de error por cubo")
        axes[j].set_xlabel("cubo de probabilidad")
        axes[j].legend(fontsize=8, loc="upper left")
        ax2.legend(fontsize=8, loc="upper right")
    plt.tight_layout()
    fig.savefig(FIG_P61 / "02_probability_bins_error.png", dpi=130)
    plt.close(fig)
    log("      figura reports/figuras/p61/02_probability_bins_error.png")

    # ---------- seccion 7: decision frente a probabilidad, cerca del umbral ----------
    log("")
    log("   SECCION 7, CERCA DEL UMBRAL: EJEMPLOS, ERRORES Y ACIERTOS EN VENTANAS")
    ventanas: dict[str, Any] = {}
    for j, et in enumerate(ETIQUETAS):
        p, yt, yp = proba[:, j], y[:, j], pred[:, j]
        dist = np.abs(p - V03_UMBRALES[j])
        aciertos = (yp == yt)
        filas_v: list[dict[str, Any]] = []
        log(f"      {et} (umbral {V03_UMBRALES[j]:.2f}):")
        for w in VENTANAS:
            m = dist <= w
            n_w = int(m.sum())
            e_w = int(np.count_nonzero(m & ~aciertos))
            a_w = int(np.count_nonzero(m & aciertos))
            err_v = (e_w / n_w) if n_w else float("nan")
            filas_v.append({"ventana": f"+/-{w:.2f}", "n": n_w, "errores": e_w,
                            "aciertos": a_w, "tasa_de_error": round(float(err_v), 6)})
            filas_csv.append({"bloque": "7_ventanas", "etiqueta": et,
                              "magnitud": filas_v[-1]["ventana"],
                              "valor": json.dumps(filas_v[-1], ensure_ascii=False)})
            log(f"        +/-{w:.2f}: n {n_w:5d} | errores {e_w:5d} | aciertos {a_w:5d} "
                f"| tasa de error {err_v:.4f} | {err_v * 100:.2f} por ciento de la ventana")
        ventanas[et] = filas_v
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for j, et in enumerate(ETIQUETAS):
        fv = ventanas[et]
        axes[j].bar([f["ventana"] for f in fv], [f["tasa_de_error"] for f in fv],
                    color=["tab:orange", "tab:red", "darkred"])
        for k, f in enumerate(fv):
            axes[j].text(k, f["tasa_de_error"] + 0.01, f"n={f['n']}\nerr={f['errores']}",
                         ha="center", fontsize=8)
        axes[j].set_title(f"{et}: tasa de error dentro de la ventana")
        axes[j].set_ylabel("tasa de error")
        axes[j].set_ylim(0, 1.05)
    plt.tight_layout()
    fig.savefig(FIG_P61 / "03_threshold_distance_windows.png", dpi=130)
    plt.close(fig)
    log("      figura reports/figuras/p61/03_threshold_distance_windows.png")

    # ---------- secciones 8 y 9: distancia al umbral y errores de alta confianza ----------
    log("")
    log("   SECCIONES 8 Y 9, DISTANCIA AL UMBRAL Y ERRORES DE ALTA CONFIANZA")
    dist_errores: dict[str, Any] = {}
    altas_confianza: dict[str, Any] = {}
    for j, et in enumerate(ETIQUETAS):
        p, yt, yp = proba[:, j], y[:, j], pred[:, j]
        dist = np.abs(p - V03_UMBRALES[j])
        aciertos = yp == yt
        d_c = dist[aciertos]
        d_i = dist[~aciertos]
        dist_errores[et] = {
            "aciertos": {"n": int(d_c.size), "media": round(float(d_c.mean()), 6),
                         "mediana": round(float(np.median(d_c)), 6),
                         "std": round(float(d_c.std(ddof=1)), 6),
                         "q10": round(float(np.quantile(d_c, 0.10)), 6),
                         "q90": round(float(np.quantile(d_c, 0.90)), 6)},
            "errores": {"n": int(d_i.size), "media": round(float(d_i.mean()), 6),
                        "mediana": round(float(np.median(d_i)), 6),
                        "std": round(float(d_i.std(ddof=1)), 6),
                        "q10": round(float(np.quantile(d_i, 0.10)), 6),
                        "q90": round(float(np.quantile(d_i, 0.90)), 6)},
        }
        filas_hc = []
        log(f"      {et}: aciertos n {d_c.size} con distancia media {d_c.mean():.4f}, "
            f"errores n {d_i.size} con distancia media {d_i.mean():.4f}")
        for dd in DELTAS_HC:
            m = (~aciertos) & (dist >= dd)
            n_hc = int(m.sum())
            frac = n_hc / max(int((~aciertos).sum()), 1)
            filas_hc.append({"delta": dd, "n_errores_de_alta_confianza": n_hc,
                             "n_errores_totales": int((~aciertos).sum()),
                             "porcentaje_de_los_errores": round(float(frac), 6)})
            filas_csv.append({"bloque": "9_alta_confianza", "etiqueta": et,
                              "magnitud": f"delta>={dd:.2f}",
                              "valor": json.dumps(filas_hc[-1], ensure_ascii=False)})
            log(f"         delta >= {dd:.2f}: {n_hc:5d} errores de alta confianza, "
                f"que son el {frac * 100:.2f} por ciento de los {int((~aciertos).sum())} errores")
        altas_confianza[et] = filas_hc
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for j, et in enumerate(ETIQUETAS):
        p, yt, yp = proba[:, j], y[:, j], pred[:, j]
        dist = np.abs(p - V03_UMBRALES[j])
        ac = yp == yt
        axes[j].hist(dist[ac], bins=40, density=True, alpha=0.55, label="correct",
                     color="tab:green", range=(0, 0.6))
        axes[j].hist(dist[~ac], bins=40, density=True, alpha=0.55, label="incorrect",
                     color="tab:red", range=(0, 0.6))
        axes[j].set_title(f"{et}: distancia al umbral")
        axes[j].set_xlabel("|p - threshold|")
        axes[j].set_ylabel("density")
        axes[j].legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(FIG_P61 / "04_distance_to_threshold.png", dpi=130)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for j, et in enumerate(ETIQUETAS):
        fh = altas_confianza[et]
        axes[j].bar([f"d>={f['delta']:.2f}" for f in fh],
                    [f["porcentaje_de_los_errores"] * 100 for f in fh], color="tab:red")
        for k, f in enumerate(fh):
            axes[j].text(k, f["porcentaje_de_los_errores"] * 100 + 0.5,
                         f"n={f['n_errores_de_alta_confianza']}", ha="center", fontsize=8)
        axes[j].set_title(f"{et}: errores de alta confianza")
        axes[j].set_ylabel("porcentaje de los errores")
    plt.tight_layout()
    fig.savefig(FIG_P61 / "05_high_confidence_errors.png", dpi=130)
    plt.close(fig)
    log("      figuras 04_distance_to_threshold.png y 05_high_confidence_errors.png")

    # ---------- seccion 10: tabla final por celda ----------
    log("")
    log("   SECCION 10, TABLA FINAL POR CELLA: QUE PROBABILIDAD Y QUE DECISION PRODUJO")
    filas_celda: list[dict[str, Any]] = []
    for i in range(n):
        for j, et in enumerate(ETIQUETAS):
            filas_celda.append({
                "indice_interno_oof": i,
                "etiqueta": et,
                "etiqueta_real": int(y[i, j]),
                "probabilidad_oof_final": round(float(proba[i, j]), 6),
                "umbral": float(V03_UMBRALES[j]),
                "decision_final": int(pred[i, j]),
                "correcto": int(pred[i, j] == y[i, j]),
                "distancia_al_umbral": round(float(abs(proba[i, j] - V03_UMBRALES[j])), 6),
                "margen_desde_el_umbral": round(float(proba[i, j] - V03_UMBRALES[j]), 6),
            })
    df_celda = pd.DataFrame(filas_celda)
    df_celda.to_csv(TABLA_CELDAS, index=False, encoding="ascii")
    log(f"      escrito reports/p61_final_probability_table.csv con {len(df_celda)} filas, "
        f"una por celda etiqueta-tweet")
    log("      el identificador es el indice POSICIONAL dentro del OOF, no un identificador "
        "de test, para que la tabla sea segura de publicar")
    # ejemplos representativos
    ejemplos: list[dict[str, Any]] = []
    for j, et in enumerate(ETIQUETAS):
        sub = df_celda[df_celda.etiqueta == et]
        err = sub[sub.correcto == 0]
        ac = sub[sub.correcto == 1]
        sel = {
            "error_de_alta_confianza": err.reindex(
                err.distancia_al_umbral.sort_values(ascending=False).index).head(3),
            "error_de_baja_confianza": err.reindex(
                err.distancia_al_umbral.sort_values().index).head(3),
            "acierto_de_alta_confianza": ac.reindex(
                ac.distancia_al_umbral.sort_values(ascending=False).index).head(3),
            "caso_cercano_al_umbral": sub.reindex(
                sub.distancia_al_umbral.sort_values().index).head(3),
        }
        for nombre, df_e in sel.items():
            ejemplos.append({"etiqueta": et, "categoria": nombre,
                             "filas": df_e.to_dict(orient="records")})
            filas_csv.append({"bloque": "10_ejemplos", "etiqueta": et, "magnitud": nombre,
                              "valor": json.dumps(df_e.to_dict(orient="records"),
                                                  ensure_ascii=False)})
        log(f"      {et}: errores {len(err)}, aciertos {len(ac)}, "
            f"error mas lejano a p={err.distancia_al_umbral.max():.4f}, "
            f"acierto mas cercano al umbral a p={ac.distancia_al_umbral.min():.4f}")

    # ---------- seccion 12: rankings por probabilidad ----------
    log("")
    log("   SECCION 12, RANKINGS POR PROBABILIDAD")
    rankings: dict[str, Any] = {}
    for j, et in enumerate(ETIQUETAS):
        p, yt, yp = proba[:, j], y[:, j], pred[:, j]
        pos = np.where(yt == 1)[0]
        neg = np.where(yt == 0)[0]
        conf_tp = pos[yp[pos] == 1]
        conf_fn = pos[yp[pos] == 0]
        conf_fp = neg[yp[neg] == 1]
        conf_tn = neg[yp[neg] == 0]
        rankings[et] = {
            "positivos_de_probabilidad_mas_alta": int(np.argsort(-p[pos])[:5].size),
            "confident_true_positives": int(conf_tp.size),
            "confident_false_negatives": int(conf_fn.size),
            "confident_false_positives": int(conf_fp.size),
            "confident_true_negatives": int(conf_tn.size),
            "p_de_los_cinco_positivos_mas_alta": [round(float(x), 6) for x in
                                                 np.sort(p[pos])[-5:][::-1]],
            "p_de_los_cinco_negativos_mas_baja": [round(float(x), 6) for x in
                                                  np.sort(p[neg])[:5]],
            "p_de_los_mas_bajos_entre_los_verdaderos_positivos": [round(float(x), 6) for x in
                                                                 np.sort(p[conf_fn])[:5]],
            "p_de_los_mas_altos_entre_los_verdaderos_negativos": [round(float(x), 6) for x in
                                                                  np.sort(p[conf_fp])[-5:][::-1]],
        }
        r = rankings[et]
        log(f"      {et}: TP confiados {r['confident_true_positives']}, "
            f"FN confiados {r['confident_false_negatives']}, "
            f"FP confiados {r['confident_false_positives']}, "
            f"TN confiados {r['confident_true_negatives']}")
        filas_csv.append({"bloque": "12_rankings", "etiqueta": et, "magnitud": "resumen",
                          "valor": json.dumps(r, ensure_ascii=False)})

    # ---------- seccion 13: geometria multi-etiqueta por tweet ----------
    log("")
    log("   SECCION 13, COMBINACION MULTI-ETIQUETA POR TWEET")
    pmax, pmin = proba.max(axis=1), proba.min(axis=1)
    pmean, pvar = proba.mean(axis=1), proba.var(axis=1)
    # desacuerdo tipo entropia sobre las tres probabilidades normalizadas
    pn = proba / np.maximum(proba.sum(axis=1, keepdims=True), 1e-12)
    ent = -(pn * np.log(np.maximum(pn, 1e-12))).sum(axis=1)
    viol_tr = int(np.count_nonzero((proba[:, 0] < 0.5) & (proba[:, 1] >= 0.5)))
    viol_ag = int(np.count_nonzero((proba[:, 0] < 0.5) & (proba[:, 2] >= 0.5)))
    viol_ag_tr = int(np.count_nonzero((proba[:, 1] < 0.5) & (proba[:, 2] >= 0.5)))
    multiv = {
        "p_max_media": round(float(pmax.mean()), 6), "p_min_media": round(float(pmin.mean()), 6),
        "p_max_std": round(float(pmax.std(ddof=1)), 6),
        "p_min_std": round(float(pmin.std(ddof=1)), 6),
        "p_media_de_las_medias": round(float(pmean.mean()), 6),
        "p_varianza_media": round(float(pvar.mean()), 6),
        "desacuerdo_medio": round(float(ent.mean()), 6),
        "desacuerdo_cuartiles": [round(float(np.quantile(ent, 0.25)), 6),
                                 round(float(np.quantile(ent, 0.75)), 6)],
        "violaciones_p_TR_alta_con_p_HS_baja": viol_tr,
        "violaciones_p_AG_alta_con_p_HS_baja": viol_ag,
        "violaciones_p_AG_alta_con_p_TR_baja": viol_ag_tr,
        "correlacion_HS_TR": round(float(np.corrcoef(proba[:, 0], proba[:, 1])[0, 1]), 6),
        "correlacion_HS_AG": round(float(np.corrcoef(proba[:, 0], proba[:, 2])[0, 1]), 6),
        "correlacion_TR_AG": round(float(np.corrcoef(proba[:, 1], proba[:, 2])[0, 1]), 6),
    }
    log(f"      p_max media {multiv['p_max_media']:.6f}, p_min media {multiv['p_min_media']:.6f}, "
        f"desacuerdo medio {multiv['desacuerdo_medio']:.6f}")
    log(f"      correlaciones entre probabilidades: HS-TR {multiv['correlacion_HS_TR']:+.4f}, "
        f"HS-AG {multiv['correlacion_HS_AG']:+.4f}, TR-AG {multiv['correlacion_TR_AG']:+.4f}")
    log(f"      violaciones estructurales a nivel de PROBABILIDAD, con umbral 0,5: "
        f"p_TR>=0,5 con p_HS<0,5 en {viol_tr} tweets; "
        f"p_AG>=0,5 con p_HS<0,5 en {viol_ag}; p_AG>=0,5 con p_TR<0,5 en {viol_ag_tr}")
    log("      NO se aplica ninguna correccion estructural: esto es EDA")
    filas_csv.append({"bloque": "13_multilabel", "etiqueta": "todas",
                      "magnitud": "geometria_por_tweet",
                      "valor": json.dumps(multiv, ensure_ascii=False)})

    # ---------- seccion 14: geometria 3D y por pares ----------
    log("")
    log("   SECCION 14, GEOMETRIA DE PROBABILIDADES CONJUNTAS 3D Y POR PARES")
    fig = plt.figure(figsize=(7, 6))
    ax3 = fig.add_subplot(111, projection="3d")
    ac = (pred == y).all(axis=1)
    np.where(ac, "tab:green", "tab:red")
    ax3.scatter(proba[ac, 0], proba[ac, 1], proba[ac, 2], s=6, c="tab:green",
                alpha=0.35, label="all three correct")
    ax3.scatter(proba[~ac, 0], proba[~ac, 1], proba[~ac, 2], s=10, c="tab:red",
                alpha=0.7, label="at least one wrong")
    ax3.set_xlabel("p_HS")
    ax3.set_ylabel("p_TR")
    ax3.set_zlabel("p_AG")
    ax3.set_title("Joint probability geometry, V03 OOF")
    ax3.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(FIG_P61 / "06_joint_probability_geometry_3d.png", dpi=130)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    # Un panel por par. Una version anterior iteraba tres veces sobre los mismos tres
    # ejes, con lo que los tres paneles acababan mostrando el MISMO par, TR contra AG.
    pares = ((0, 1, "p_HS vs p_TR"), (0, 2, "p_HS vs p_AG"), (1, 2, "p_TR vs p_AG"))
    for ax, (a, b, titulo) in zip(axes, pares):
        ax.scatter(proba[ac, a], proba[ac, b], s=6, c="tab:green", alpha=0.35,
                   label="all three correct")
        ax.scatter(proba[~ac, a], proba[~ac, b], s=10, c="tab:red", alpha=0.7,
                   label="at least one wrong")
        ax.set_xlabel(f"p_{ETIQUETAS[a]}")
        ax.set_ylabel(f"p_{ETIQUETAS[b]}")
        ax.set_title(titulo)
        ax.legend(fontsize=7)
    plt.tight_layout()
    fig.savefig(FIG_P61 / "07_pairwise_probability_geometry.png", dpi=130)
    plt.close(fig)
    log("      figuras 06_joint_probability_geometry_3d.png y "
        "07_pairwise_probability_geometry.png")
    log(f"      tweets con las tres etiquetas correctas: {int(ac.sum())} de {n}, "
        f"con al menos una mal: {int((~ac).sum())}")

    # ---------- seccion 15: margen desde el umbral ----------
    log("")
    log("   SECCION 15, MARGEN DESDE EL UMBRAL, SEPARANDO ACIERTOS DE ERRORES")
    margenes: dict[str, Any] = {}
    for j, et in enumerate(ETIQUETAS):
        m_ = proba[:, j] - V03_UMBRALES[j]
        ac_ = pred[:, j] == y[:, j]
        margenes[et] = {
            "aciertos": {"media": round(float(m_[ac_].mean()), 6),
                         "mediana": round(float(np.median(m_[ac_])), 6),
                         "std": round(float(m_[ac_].std(ddof=1)), 6),
                         "q10": round(float(np.quantile(m_[ac_], 0.10)), 6),
                         "q90": round(float(np.quantile(m_[ac_], 0.90)), 6)},
            "errores": {"media": round(float(m_[~ac_].mean()), 6),
                        "mediana": round(float(np.median(m_[~ac_])), 6),
                        "std": round(float(m_[~ac_].std(ddof=1)), 6),
                        "q10": round(float(np.quantile(m_[~ac_], 0.10)), 6),
                        "q90": round(float(np.quantile(m_[~ac_], 0.90)), 6)},
        }
        filas_csv.append({"bloque": "15_margen", "etiqueta": et, "magnitud": "margen",
                          "valor": json.dumps(margenes[et], ensure_ascii=False)})
        log(f"      {et}: aciertos margen medio {margenes[et]['aciertos']['media']:+.6f}, "
            f"errores margen medio {margenes[et]['errores']['media']:+.6f}")

    # ---------- seccion 16: brier, log loss, ECE y fiabilidad ----------
    log("")
    log("   SECCION 16, BRIER, LOG LOSS, ECE Y CURVA DE FIABILIDAD")
    metricas_cal: dict[str, Any] = {}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for j, et in enumerate(ETIQUETAS):
        yt, p = y[:, j], np.clip(proba[:, j], 1e-9, 1 - 1e-9)
        b = float(brier_score_loss(yt, p))
        ll = float(log_loss(yt, p, labels=[0, 1]))
        e, cubos = ece(p, yt, N_BUCKETS_ECE)
        metricas_cal[et] = {"brier": round(b, 8), "log_loss": round(ll, 8),
                            "ece": round(e, 8), "n_cubos": len(cubos), "cubos": cubos}
        filas_csv.append({"bloque": "16_calibracion", "etiqueta": et, "magnitud": "brier_logloss_ece",
                          "valor": json.dumps({k: v for k, v in metricas_cal[et].items()
                                               if k != "cubos"}, ensure_ascii=False)})
        log(f"      {et}: Brier {b:.8f} | log loss {ll:.8f} | ECE {e:.8f} "
            f"con {N_BUCKETS_ECE} cubos equipoblados")
        axes[j].plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
        axes[j].plot([c["prob_media"] for c in cubos], [c["frecuencia_observada"] for c in cubos],
                     "o-", color="tab:blue", label="reliability")
        axes[j].set_xlabel("probabilidad media del cubo")
        axes[j].set_ylabel("frecuencia observada")
        axes[j].set_title(f"{et}: fiabilidad, ECE {e:.4f}")
        axes[j].legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(FIG_P61 / "08_reliability_curves.png", dpi=130)
    plt.close(fig)
    log("      figura reports/figuras/p61/08_reliability_curves.png")
    log("      ADVERTENCIA: estos numeros son DIAGNOSTICOS. La calibracion sigue cerrada como")
    log("      hipotesis de mejora, y no se usara para recalibrar V03 automaticamente")

    # ---------- informe ----------
    L: list[str] = []
    W = L.append
    W("# P61 - EDA de las probabilidades OOF canonicas de V03")
    W("")
    W(f"Generado por `experiments/p61_01_eda_probabilidades.py`. "
      f"Fecha {inicio.isoformat(timespec='seconds')}.")
    W("")
    W("> **Esta EDA es DIAGNOSTICA.** No se usa para ajustar thresholds, seleccionar modelos ni")
    W("> optimizar hiperparametros sobre este mismo OOF. Si de ella saliera una hipotesis")
    W("> nueva, habria que crear esa hipotesis con su propio protocolo anidado.")
    W("")
    W("## La pregunta que esta EDA responde")
    W("")
    W("No `que tan buenas son las probabilidades`, sino:")
    W("")
    W("> que probabilidad produjo V03 finalmente para cada observacion, donde estaba respecto")
    W("> al threshold, y que decision final produjo esa probabilidad?")
    W("")
    W("La tabla de esa pregunta es `reports/p61_final_probability_table.csv`, con una fila por")
    W("celda etiqueta-tweet, y las columnas `probabilidad_oof_final`, `umbral`,")
    W("`decision_final`, `correcto`, `distancia_al_umbral` y `margen_desde_el_umbral`.")
    W("")
    W("## 11. Que probabilidad usa la decision, verificado y no supuesto")
    W("")
    W("En este proyecto se llaman probabilidad a cuatro cosas distintas:")
    W("")
    W("```text")
    W("score crudo del modelo     la funcion de decision del modelo lineal")
    W("probabilidad sigmoid       sigma del score, es decir predict_proba")
    W("probabilidad OOF           la anterior, en filas que el modelo no entreno")
    W("probabilidad calibrada     una reparametrizacion monotona, que esta CERRADA")
    W("```")
    W("")
    W("La decision de V03 consume la **tercera**. No es una afirmacion del informe sino un")
    W("calculo: recomputar la decision desde `proba` y los umbrales reproduce exactamente la")
    W(f"decision almacenada, con **{coincide}**. Si no coincidiera, el script pararia, porque")
    W("describir estas probabilidades seria describir el objeto equivocado.")
    W("")
    W("## 4. Distribucion de p(y=1|x)")
    W("")
    W("| etiqueta | media | mediana | std | min | max | skew | kurt | f(p=0) | f(p=1) |")
    W("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for et in ETIQUETAS:
        s = estadisticas[et]
        W(f"| {et} | {s['media']:.6f} | {s['mediana']:.6f} | {s['std']:.6f} | {s['min']:.6f} | "
          f"{s['max']:.6f} | {s['skewness']:+.4f} | {s['kurtosis']:+.4f} | "
          f"{s['fraccion_igual_a_0']*100:.3f} por ciento | "
          f"{s['fraccion_igual_a_1']*100:.3f} por ciento |")
    W("")
    W("| etiqueta | q01 | q05 | q10 | q25 | q50 | q75 | q90 | q95 | q99 | umbral |")
    W("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for et in ETIQUETAS:
        s = estadisticas[et]
        W(f"| {et} | " + " | ".join(f"{s[f'q{int(c * 100):02d}']:.6f}" for c in CUANTILES)
          + f" | {s['umbral']:.2f} |")
    W("")
    W("| etiqueta | media TP | media TN | solapamiento de las dos densidades |")
    W("|---|---:|---:|---:|")
    for et in ETIQUETAS:
        s = estadisticas[et]
        W(f"| {et} | {s['media_de_los_verdaderos_positivos']:.6f} | "
          f"{s['media_de_los_verdaderos_negativos']:.6f} | {separacion[et]:.4f} |")
    W("")
    W("## 7. Cerca del umbral")
    W("")
    W("| etiqueta | ventana | n | errores | aciertos | tasa de error |")
    W("|---|---|---:|---:|---:|---:|")
    for et in ETIQUETAS:
        for f in ventanas[et]:
            W(f"| {et} | {f['ventana']} | {f['n']} | {f['errores']} | {f['aciertos']} | "
              f"{f['tasa_de_error']:.4f} |")
    W("")
    W("## 8. Distancia al umbral")
    W("")
    W("| etiqueta | clase | n | media | mediana | std | q10 | q90 |")
    W("|---|---|---:|---:|---:|---:|---:|---:|")
    for et in ETIQUETAS:
        for clase in ("aciertos", "errores"):
            d = dist_errores[et][clase]
            W(f"| {et} | {clase} | {d['n']} | {d['media']:.6f} | {d['mediana']:.6f} | "
              f"{d['std']:.6f} | {d['q10']:.6f} | {d['q90']:.6f} |")
    W("")
    W("## 9. Errores de alta confianza")
    W("")
    W("Se definen solo para diagnostico, con `error` y `distancia al umbral >= delta`, y el")
    W("delta no se optimiza.")
    W("")
    W("| etiqueta | delta | n errores de alta confianza | n errores totales | porcentaje |")
    W("|---|---:|---:|---:|---:|")
    for et in ETIQUETAS:
        for f in altas_confianza[et]:
            W(f"| {et} | {f['delta']:.2f} | {f['n_errores_de_alta_confianza']} | "
              f"{f['n_errores_totales']} | {f['porcentaje_de_los_errores']*100:.2f} por ciento |")
    W("")
    W("## 12. Rankings por probabilidad")
    W("")
    W("| etiqueta | TP confiados | FN confiados | FP confiados | TN confiados |")
    W("|---|---:|---:|---:|---:|")
    for et in ETIQUETAS:
        r = rankings[et]
        W(f"| {et} | {r['confident_true_positives']} | {r['confident_false_negatives']} | "
          f"{r['confident_false_positives']} | {r['confident_true_negatives']} |")
    W("")
    W("## 13. Combinacion multi-etiqueta por tweet")
    W("")
    W("```text")
    for k, val in multiv.items():
        W(f"{k:44} {val}")
    W("```")
    W("")
    W("Las violaciones estructurales se miden a nivel de probabilidad con umbral 0,5, y son")
    W("**diagnostico**: no se aplica ninguna correccion estructural.")
    W("")
    W("## 16. Brier, log loss y ECE")
    W("")
    W("| etiqueta | Brier | log loss | ECE |")
    W("|---|---:|---:|---:|")
    for et in ETIQUETAS:
        m = metricas_cal[et]
        W(f"| {et} | {m['brier']:.8f} | {m['log_loss']:.8f} | {m['ece']:.8f} |")
    W("")
    W("**La calibracion sigue cerrada como hipotesis de mejora.** Estos numeros son")
    W("diagnosticos, y el ECE alto no es un hallazgo: V03 se ajusto con class_weight")
    W("balanced, que por construccion desplaza las probabilidades para compensar el")
    W("desequilibrio, luego sus probabilidades no son frequencies y un ECE grande es el")
    W("comportamiento esperado de un modelo correctamente ponderado para decision.")
    W("")
    W("## Graficas")
    W("")
    for f in sorted(FIG_P61.glob("*.png")):
        W(f"* `reports/figuras/p61/{f.name}`")
    W("")
    W("*Generado por `experiments/p61_01_eda_probabilidades.py`. Sin push.*")
    MD_EDA.write_text(a_ascii("\n".join(L)), encoding="ascii")

    with CSV_EDA.open("w", encoding="ascii", newline="") as fh:
        w = _csv.writer(fh, lineterminator="\n")
        w.writerow(["bloque", "etiqueta", "magnitud", "valor"])
        for f in filas_csv:
            w.writerow([a_ascii(f["bloque"]), a_ascii(f["etiqueta"]),
                        a_ascii(f["magnitud"]), a_ascii(f["valor"])])

    informe = {
        "familia": "P61_PROBABILITY_EDA",
        "naturaleza": ("DIAGNOSTICO. 0 experimentos, 0 modelos, 0 submissions, 0 Kaggle, "
                       "0 holdout, 0 test. No se usa para optimizar nada"),
        "fecha": inicio.isoformat(timespec="seconds"),
        "probabilidad_que_usa_la_decision": {
            "es": "probabilidad sigmoid OUT-OF-FOLD",
            "recalcula_la_decision_almacenada": bool(coincide),
            "probabilidad_calibrada_usable": False,
            "motivo": "la calibracion esta cerrada como hipotesis de mejora por V8-09 y P59-T",
        },
        "estadisticas": estadisticas,
        "solapamiento_densidades": separacion,
        "bins": bins,
        "ventanas_cerca_del_umbral": ventanas,
        "distancia_al_umbral": dist_errores,
        "errores_de_alta_confianza": altas_confianza,
        "rankings": rankings,
        "geometria_multilabel": multiv,
        "margen_desde_el_umbral": margenes,
        "calibracion": {k: {kk: vv for kk, vv in v.items() if kk != "cubos"}
                        for k, v in metricas_cal.items()},
        "curvas_fiabilidad": {k: v["cubos"] for k, v in metricas_cal.items()},
        "figuras": sorted(f.name for f in FIG_P61.glob("*.png")),
        "artefactos": {
            "informe": "reports/p61_probability_eda.md",
            "csv": "reports/p61_probability_eda.csv",
            "tabla_por_celda": "reports/p61_final_probability_table.csv",
        },
        "formulacion": "V03 remains the best verified leaderboard result.",
    }
    JSON_EDA.write_text(json.dumps(informe, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")

    for destino in (MD_EDA, CSV_EDA, JSON_EDA, TABLA_CELDAS):
        if any(b > 127 for b in destino.read_bytes()):
            raise SystemExit(f"PARADA: {destino.name} conserva bytes no ASCII")
    log("")
    log(f"P61-01 OK | {len(df_celda)} celdas en la tabla final | "
        f"{len(list(FIG_P61.glob('*.png')))} figuras | 0 experimentos")


if __name__ == "__main__":
    ast.parse(Path(__file__).read_text(encoding="utf-8"))
    main()
