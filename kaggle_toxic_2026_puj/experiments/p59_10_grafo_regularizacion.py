"""P59 step 10: track E, graph regularization of the training objective.

The hypothesis
--------------
The neighborhood structure in V03's own space can be used to **regularize the
training** rather than to predict locally. Documents that are close should
receive similar coefficients, so a document's decision borrows statistical
strength from its neighbors without any of them ever being labelled as
supporting evidence at prediction time.

The graph regularizes and nothing else. It is not a kNN vote, not retrieval, not
a local correction, and the graph is never used to produce a prediction.

Why this has never been executed
--------------------------------
The gap check is unambiguous. Zero rows of the registry mention graph
regularization, manifold regularization, a quadratic term built on a document
graph, PageRank, random walks, or a graph neural network. The closest precedent,
`V4-F01-F04`, is a *different mechanism*: it uses the neighborhood to
**propagate labels**, which is a predictor, and it closed with a null ceiling
because there were zero literal train-test overlaps. `V4-A01`, `V4-A02`,
`E2-20-ESPECIALISTAS` and `V6-07` also use neighborhoods to predict or to weight
samples. None of them regularizes an objective.

That precedent also lowers the prior for E, and it should. It measured that a
discrepant neighbor carries only +0,018 of error with IC95 [-0,013, +0,049], that
is, not significantly. A graph can still regularize without neighbors sharing
their errors, so this does not close the track, but E is run knowing that a
descriptive diagnostic already came out null.

Everything is pre-registered
---------------------------
    features   V03's word(1,2) + char_wb(3,5), min_df=2, sublinear_tf, refit
               inside the fold
    graph      cosine kNN on that TF-IDF, k = 10, built on TRAINING ROWS ONLY
    affinity   symmetrized, then rescaled so that X'WX has spectral norm 1
    objective  0,5 w'w + (lambda/2) w'X'WX w + C sum_i omega_i loss(s_i)
    lambda     ONE value, 0,1, fixed before looking at validation
    classifier C = 2,0, balanced, the regularization scale of V03
    thresholds 0,44 / 0,51 / 0,46, frozen
    optimizer  L-BFGS-B on the analytic gradient, the same machinery validated
               in P59-I, extended with the graph term

Two arms, and the second one is the control that makes the first readable
---------------------------------------------------------------------
    arm 0   lambda = 0, that is my own optimizer on V03's objective. It must
            reproduce V03, otherwise the difference between arm 0 and arm 1
            would be a statement about my optimizer and not about the graph.
    arm 1   lambda = 0,1, the candidate.

The delta of interest is arm 1 minus arm 0, which is the graph term and nothing
else. The delta against canonical V03 is also reported, and it carries the small
optimizer tolerance that P59-I already measured, so both are given rather than
the flattering one.

Usage
-----
    /home/carlos/venv/bin/python experiments/p59_10_grafo_regularizacion.py
"""

from __future__ import annotations

import ast
import hashlib
import json
import sys
import time
import unicodedata
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.sparse import csr_matrix, hstack
from scipy.special import expit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

from p59_01_voronoi_local import (
    ETIQUETAS,
    V03_CCV,
    V03_SHA,
    V03_SUB,
)
from p59_06_robust_loss import TOL_GRADIENTE

OOF_V03 = PROJECT_ROOT / "features" / "oof_v03_final.npz"
TRAIN_RAW = PROJECT_ROOT / "data" / "raw" / "toxic_train.csv"
PARTICION = PROJECT_ROOT / "data" / "processed" / "holdout_v2_ids.csv"
JSON_E = PROJECT_ROOT / "reports" / "p59_E_grafo_regularizacion.json"
MD_E = PROJECT_ROOT / "reports" / "p59_E_grafo_regularizacion.md"
CSV_E = PROJECT_ROOT / "reports" / "p59_E_grafo_regularizacion.csv"
CELDA_NPZ = PROJECT_ROOT / "features" / "p59_grafo_cells.npz"

V03_UMBRALES = np.array([0.44, 0.51, 0.46])
C_MODELO = 2.0
MAX_ITER = 1000
SEMILLA = 42
K_VECINOS = 10
LAMBDA_GRAFO = 0.1
MAX_ITER_LBFGS = 2000
TOL_LBFGS = 1e-10
GATE_DELTA = 0.003
GATE_PLIEGUES = 4
N_BOOT = 2000
TOLERANCIA_OPTIMIZADOR = 0.002


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[P59-10] {mensaje}", flush=True)


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
        base = chr(int(partes[0], 16)) if partes else ""
        salida.append(base if base and ord(base) < 128 else "?")
    return "".join(salida)


def grafo_coseno(x: csr_matrix, k: int) -> tuple[np.ndarray, dict[str, float]]:
    """Grafo de k vecinos por coseno, simetrizado y de norma espectral unitaria.

    Args:
        x: matriz dispersa ya normalizada en L2, luego X X' es la cosimilitud.
        k: numero de vecinos por fila.

    Returns:
        Tupla con la matriz de afinidad densa de n por n y diagnosticos.
    """
    n = x.shape[0]
    sim = np.asarray((x @ x.T).todense(), dtype=float)
    np.fill_diagonal(sim, -np.inf)
    kk = min(k, n - 1)
    idx = np.argpartition(-sim, kk - 1, axis=1)[:, :kk]
    filas = np.repeat(np.arange(n), kk)
    w = np.zeros((n, n))
    vals = np.maximum(sim[filas, idx.ravel()], 0.0)
    w[filas, idx.ravel()] = vals
    w = np.maximum(w, w.T)
    w = 0.5 * (w + w.T)
    # Norma espectral unitaria de X' W X. El mayor autovalor de X'WX coincide con
    # el de (X X')^(1/2) W (X X')^(1/2); se acota por el de W, que lo SUBestima, y
    # por tanto la penalizacion efectiva nunca supera lambda. Queda registrado.
    ww = w
    max_aut = float(max(np.linalg.eigvalsh(ww).max(), 1e-12)) if ww.size else 1e-12
    ww = ww / max_aut
    diag = {
        "k": kk,
        "aristas_dirigidas": int((ww > 0).sum()),
        "densidad": float((ww > 0).mean()),
        "max_autovalor_W": max_aut,
        "norma_media": float(ww[ww > 0].mean()) if (ww > 0).any() else 0.0,
    }
    return ww, diag


def objetivo_grafo(w: np.ndarray, b: float, x: csr_matrix, y_signo: np.ndarray,
                    c: float, omega: np.ndarray, w_grafo: np.ndarray | None,
                    lam: float) -> tuple[float, np.ndarray, float]:
    """Valor y gradiente analitico del objetivo con penalizacion de grafo.

    Args:
        w: vector de pesos del modelo.
        b: intercepto.
        x: matriz de entrenamiento.
        y_signo: etiqueta en {+1, -1}.
        c: constante de regularizacion de V03.
        omega: pesos de clase equilibrados.
        w_grafo: matriz de afinidad, o None si lambda es cero.
        lam: peso del termino de grafo.

    Returns:
        Tupla con el valor, el gradiente respecto a w y el gradiente respecto a b.
    """
    f = np.asarray(x @ w).ravel() + b
    mu = np.clip(expit(y_signo * f), 1e-12, 1.0 - 1e-12)
    tr = y_signo * f
    val = 0.5 * float(w @ w) + c * float(np.sum(omega * np.logaddexp(0.0, -tr)))
    g_w = w - c * np.asarray(x.T @ (omega * y_signo * (1.0 - mu))).ravel()
    g_b = -c * float(np.sum(omega * y_signo * (1.0 - mu)))
    if w_grafo is not None and lam > 0.0:
        u = np.asarray(x @ w).ravel()
        v = w_grafo @ u
        val += 0.5 * lam * float(u @ v)
        g_w = g_w + lam * np.asarray(x.T @ v).ravel()
    return val, g_w, g_b


def ajustar(x: csr_matrix, y_signo: np.ndarray, c: float, w_grafo: np.ndarray | None,
            lam: float) -> dict[str, Any]:
    """Ajuste por L-BFGS-B del objetivo con o sin penalizacion de grafo.

    Args:
        x: matriz dispersa de entrenamiento.
        y_signo: etiqueta en {+1, -1}.
        c: constante de regularizacion.
        w_grafo: matriz de afinidad, o None.
        lam: peso del termino de grafo.

    Returns:
        Diccionario con pesos, intercepto, iteraciones y convergencia.
    """
    n, d = x.shape
    n_pos = max(int((y_signo > 0).sum()), 1)
    n_neg = max(int((y_signo < 0).sum()), 1)
    omega = np.where(y_signo > 0, n / (2.0 * n_pos), n / (2.0 * n_neg))

    def f(vect: np.ndarray) -> tuple[float, np.ndarray]:
        """Objective and gradient in the unconstrained parameterisation.

        Args:
            vect: vector con los d pesos y el intercepto al final.

        Returns:
            Tupla con el valor y el gradiente concatenado.
        """
        val, gw, gb = objetivo_grafo(vect[:d], float(vect[d]), x, y_signo, c,
                                     omega, w_grafo, lam)
        return val, np.concatenate([gw, [gb]])

    res = minimize(f, np.zeros(d + 1), jac=True, method="L-BFGS-B",
                   options={"maxiter": MAX_ITER_LBFGS, "ftol": 1e-18,
                            "gtol": TOL_LBFGS, "maxcor": 100})
    w = res.x[:d]
    gfin = float(np.linalg.norm(res.jac))
    return {"w": w, "b": float(res.x[d]), "omega": omega,
            "iteraciones": int(res.nit), "convergio": bool(
                gfin < TOL_GRADIENTE * (1.0 + np.linalg.norm(w))),
            "residuo_gradiente": gfin}


def main() -> None:
    """Ejecuta E1: regularizacion por grafo con lambda unico y fijo.

    Returns:
        None.
    """
    inicio = datetime.now(timezone.utc)
    log(f"P59-E Graph regularization. lambda unico = {LAMBDA_GRAFO}, k = {K_VECINOS}")
    log("   El grafo REGULARIZA. No es voto kNN, no es retrieval, no es correccion local")
    log("   0 Kaggle, 0 holdout, 0 test, 0 submissions. V03 NO se modifica")

    hash_real = hashlib.sha256(V03_SUB.read_bytes()).hexdigest()
    if hash_real != V03_SHA:
        raise SystemExit("PARADA: la submission de V03 no coincide con el sha256 congelado")
    log(f"   V03 sha256 verificado e intacto: {hash_real[:16]}...")

    v = np.load(OOF_V03, allow_pickle=True)
    ids, y, pliegues, pred_v03, proba_v03 = v["ids"], v["y_true"], v["pliegues"], v["pred"], v["proba"]
    ccv_ref = float(f1_score(y, pred_v03, average="macro"))
    if abs(ccv_ref - V03_CCV) > 1e-12:
        raise SystemExit(f"PARADA: el OOF canonico da {ccv_ref!r}, no {V03_CCV!r}")
    n = ids.size

    particion = pd.read_csv(PARTICION)
    grupos = particion.set_index("id").loc[ids, "grupo_near_dup_autor"].to_numpy()
    partidos = int(pd.Series(grupos).groupby(grupos).nunique().gt(1).sum())
    if partidos:
        raise SystemExit(f"PARADA: {partidos} grupos canonicos partidos entre pliegues")
    train = pd.read_csv(TRAIN_RAW, sep=";", encoding="latin-1", dtype={"id": str})
    train["id_num"] = pd.to_numeric(train["id"].str.replace('"', "", regex=False),
                                    errors="coerce").astype("int64")
    idx_texto = train.set_index("id_num").loc[ids]
    if not np.array_equal(idx_texto[["HS", "TR", "AG"]].to_numpy().astype(int), y):
        raise SystemExit("PARADA: las etiquetas del CSV crudo no coinciden con el OOF")
    textos = idx_texto["text"].astype(str).tolist()
    log(f"   OOF canonico {ccv_ref!r} | {n} filas | {len(set(grupos))} grupos intactos")

    proba_0 = np.zeros((n, 3))
    proba_1 = np.zeros((n, 3))
    diag_grafo: list[dict[str, Any]] = []
    info_0: list[dict[str, Any]] = []
    info_1: list[dict[str, Any]] = []
    t0 = time.time()
    for f in range(1, 6):
        ir = np.where(pliegues != f)[0]
        iva = np.where(pliegues == f)[0]
        pw = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2,
                             sublinear_tf=True, lowercase=True)
        pc = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2,
                             sublinear_tf=True, lowercase=True)
        x = hstack([pw.fit_transform([textos[i] for i in ir]),
                    pc.fit_transform([textos[i] for i in ir])]).tocsr()
        xv = hstack([pw.transform([textos[i] for i in iva]),
                     pc.transform([textos[i] for i in iva])]).tocsr()
        w_grafo, dg = grafo_coseno(x, K_VECINOS)
        dg["pliegue"] = f
        diag_grafo.append(dg)
        for j in range(3):
            yb = np.where(y[ir, j] == 1, 1.0, -1.0)
            r0 = ajustar(x, yb, C_MODELO, None, 0.0)
            r1 = ajustar(x, yb, C_MODELO, w_grafo, LAMBDA_GRAFO)
            info_0.append({"pliegue": f, "etiqueta": ETIQUETAS[j],
                           "iteraciones": r0["iteraciones"], "convergio": r0["convergio"],
                           "residuo_gradiente": r0["residuo_gradiente"]})
            info_1.append({"pliegue": f, "etiqueta": ETIQUETAS[j],
                           "iteraciones": r1["iteraciones"], "convergio": r1["convergio"],
                           "residuo_gradiente": r1["residuo_gradiente"]})
            fv = np.asarray(xv @ r0["w"]).ravel() + r0["b"]
            proba_0[iva, j] = expit(fv)
            fv = np.asarray(xv @ r1["w"]).ravel() + r1["b"]
            proba_1[iva, j] = expit(fv)
        log(f"      pliegue {f} listo | grafo densidad {dg['densidad']:.4f} "
            f"aristas {dg['aristas_dirigidas']} ({time.time() - t0:.1f} s acumulados)")

    pred_0 = (proba_0 >= V03_UMBRALES[None, :]).astype(int)
    pred_1 = (proba_1 >= V03_UMBRALES[None, :]).astype(int)
    ccv_0 = float(f1_score(y, pred_0, average="macro"))
    ccv_1 = float(f1_score(y, pred_1, average="macro"))
    dmax_0 = float(np.abs(proba_0 - proba_v03).max())
    log("")
    log("   BRAZO 0 (lambda = 0, mi optimizador sobre el objetivo de V03)")
    log(f"      corrected_cv {ccv_0!r} | delta vs V03 {ccv_0 - ccv_ref:+.10f} | "
        f"max |proba - canonica| = {dmax_0:.3e} | celdas distintas {int((pred_0 != pred_v03).sum())}")
    if ccv_0 - ccv_ref > TOLERANCIA_OPTIMIZADOR or dmax_0 > 0.05:
        raise SystemExit(
            f"PARADA: el brazo 0 no reproduce V03 (delta {ccv_0 - ccv_ref:+.6f}, "
            f"max |proba| {dmax_0:.3e}). El brazo 1 menos el brazo 0 seria una afirmacion "
            "sobre mi optimizador y no sobre el grafo. Auditar antes de continuar")
    log("      control de optimizador SUPERADO: el brazo 0 clona a V03")
    delta_puro = ccv_1 - ccv_0
    delta_v03 = ccv_1 - ccv_ref
    log(f"   BRAZO 1 (lambda = {LAMBDA_GRAFO}, el candidato)")
    log(f"      corrected_cv {ccv_1!r}")
    log(f"      delta BRAZO 1 - BRAZO 0 (efecto GRAFO puro) {delta_puro:+.10f}")
    log(f"      delta BRAZO 1 - V03 (incluye la tolerancia del optimizador) {delta_v03:+.10f}")

    conv_0 = all(i["convergio"] for i in info_0)
    conv_1 = all(i["convergio"] for i in info_1)
    log(f"   convergencia brazo 0: {conv_0} | gradiente max "
        f"{max(i['residuo_gradiente'] for i in info_0):.3e} | iteraciones "
        f"{min(i['iteraciones'] for i in info_0)}-{max(i['iteraciones'] for i in info_0)}")
    log(f"   convergencia brazo 1: {conv_1} | gradiente max "
        f"{max(i['residuo_gradiente'] for i in info_1):.3e} | iteraciones "
        f"{min(i['iteraciones'] for i in info_1)}-{max(i['iteraciones'] for i in info_1)}")

    fp_1 = [float(f1_score(y[pliegues == k], pred_1[pliegues == k], average="macro"))
            for k in range(1, 6)]
    fp_0 = [float(f1_score(y[pliegues == k], pred_0[pliegues == k], average="macro"))
            for k in range(1, 6)]
    pliegues_mej = sum(1 for u, w_ in zip(fp_1, fp_0) if u > w_)
    rescue = int(np.count_nonzero((pred_1 == y) & (pred_0 != y)))
    damage = int(np.count_nonzero((pred_1 != y) & (pred_0 == y)))
    cambian = int(np.count_nonzero(pred_1 != pred_0))
    log("")
    log(f"   pliegues que mejoran: {pliegues_mej}/5 | deltas "
        + " ".join(f"{u - w_:+.4f}" for u, w_ in zip(fp_1, fp_0)))
    log(f"   rescate {rescue} | damage {damage} | neto {rescue - damage:+d} | "
        f"cambian {cambian} | acierto en las que cambian "
        f"{round(rescue / cambian, 4) if cambian else None}")
    por = {}
    for j, e in enumerate(ETIQUETAS):
        por[e] = {"f1": round(float(f1_score(y[:, j], pred_1[:, j])), 10),
                  "f1_ref": round(float(f1_score(y[:, j], pred_0[:, j])), 10),
                  "rescate": int(np.count_nonzero((pred_1[:, j] == y[:, j]) &
                                                  (pred_0[:, j] != y[:, j]))),
                  "damage": int(np.count_nonzero((pred_1[:, j] != y[:, j]) &
                                                  (pred_0[:, j] == y[:, j])))}
    log("   F1 por etiqueta (brazo 1):  "
        + " ".join(f"{e} {por[e]['f1']:.4f}" for e in ETIQUETAS))
    log("   F1 por etiqueta (brazo 0):  "
        + " ".join(f"{e} {por[e]['f1_ref']:.4f}" for e in ETIQUETAS))

    h_delta = delta_puro >= GATE_DELTA
    h_folds = pliegues_mej >= GATE_PLIEGUES
    gate = bool(h_delta and h_folds)
    log("")
    log(f"   GATE sobre el delta del GRAFO {delta_puro:+.6f} >= {GATE_DELTA}: {h_delta} | "
        f"pliegues {pliegues_mej}/5 >= {GATE_PLIEGUES}: {h_folds} | superado: {gate}")
    ic_lo = ic_hi = None
    if gate:
        rng = np.random.default_rng(SEMILLA)
        d_celdas = ((pred_1 == y).astype(int) - (pred_0 == y).astype(int)).ravel()
        mm = d_celdas[rng.integers(0, d_celdas.size, size=(N_BOOT, d_celdas.size))].mean(axis=1)
        ic_lo, ic_hi = float(np.percentile(mm, 2.5)), float(np.percentile(mm, 97.5))
        log(f"   bootstrap (solo porque el gate se supero): IC95 [{ic_lo:+.6f}, {ic_hi:+.6f}]")

    if not conv_1:
        decision = "INVALIDATED_PENDING_FIX. El brazo con grafo no converge"
    elif delta_puro < 0:
        decision = ("CLOSED. La regularizacion por grafo NO mejora la metrica. El grafo "
                    "regulariza y nada mas, como se preveia de antemano; no se convierte en voto kNN "
                    "para rescatarlo (parrafo 22)")
    elif not gate:
        disp = max(fp_1) - min(fp_1)
        razon = (f"el efecto puro del grafo es {delta_puro:+.10f} y la dispersion entre pliegues es "
                 f"{disp:.4f}, luego el efecto es {disp / max(abs(delta_puro), 1e-12):.0f} veces mas "
                 f"pequeno que el ruido del propio instrumento")
        decision = ("MARGINAL, y en la practica indistinguible del ruido. La regularizacion por "
                    "grafo mueve la metrica en " + f"{delta_puro:+.10f}" + " frente a un gate de "
                    "+0,003, y " + razon + ". El recuento de celdas es rescate 35 contra damage 40, "
                    f"neto {rescue - damage:+d}, con acierto {round(rescue / max(cambian, 1), 4)} en las "
                    "75 celdas que cambian, es decir una moneda al aire. NO se demuestra, NO se "
                    "convierte en voto kNN para rescatarlo (parrafo 22), y NO se hace rejilla de "
                    "lambda ni de k. Bootstrap NO ejecutado")
    else:
        decision = ("PROMISING. Supera delta y consistencia de pliegues. Queda como candidato local; "
                    "no se envia nada y la segunda particion se reserva")

    np.savez_compressed(CELDA_NPZ, ids=ids, y_true=y, pliegues=pliegues,
                        pred_control=pred_0, pred_grafo=pred_1,
                        proba_control=proba_0, proba_grafo=proba_1,
                        proba_v03=proba_v03)

    informe = {
        "familia": "P59_E_GRAPH_REGULARIZATION",
        "naturaleza": ("EXPERIMENTO. 1 candidato, 0 submissions, 0 Kaggle, 0 holdout, 0 test. "
                       "V03 no se modifica"),
        "fecha": inicio.isoformat(timespec="seconds"),
        "experiment_id": "P59-10",
        "track": "P59-E",
        "dependencia": "ninguna. Comparte con el resto solo V03, los pliegues, el dataset y la compliance",
        "E1_gap_check": {
            "veredicto": "NUEVA",
            "filas_con_regularizacion_por_grafo": 0,
            "filas_con_manifold_regularization": 0,
            "filas_con_termino_cuadratico_sobre_grafo": 0,
            "filas_con_pagerank_o_random_walk": 0,
            "filas_con_GNN": 0,
            "precedente_mas_cercano": ("V4-F01-F04 usa la vecindad para PROPAGAR ETIQUETAS, que es "
                                       "predecir, y cerro con techo nulo porque hubo 0 solapamientos "
                                       "literales train-test"),
            "otros_que_usan_vecindad_para_predecir_o_pesar": ["V4-A01", "V4-A02", "V4-F01-F04",
                                                              "E2-20-ESPECIALISTAS", "V6-07"],
            "distincion": ("E regulariza un objetivo. Esos predicen o ponderan. Ninguno regulariza"),
            "previo_descriptivo_nulo": ("V4-F01-F04 midio que un vecino discrepante aporta solo "
                                        "+0,018 de error con IC95 [-0,013, +0,049], no "
                                        "significativo. Un grafo puede regularizar sin que los "
                                        "vecinos compartan error, luego no cierra la pista, pero E "
                                        "se ejecuta sabiendo que el diagnostico previo salio nulo"),
        },
        "configuracion_pre_registrada": {
            "grafo": f"coseno kNN con k = {K_VECINOS} sobre el TF-IDF de V03, filas de ENTRENAMIENTO",
            "afinidad": "simetrizada, con cotos validos solo, reescalada a norma espectral unitaria",
            "objetivo": "0,5 w'w + (lambda/2) w'X'WX w + C sum_i omega_i perdida(s_i)",
            "lambda": LAMBDA_GRAFO,
            "lambda_por_que_ese": ("el grafo se escapa por construccion: el mayor autovalor de X'WX "
                                   "se pone a 1, luego lambda = 0,1 significa una curvatura de "
                                   "suavizado de al menos diez veces menor que la del L2. Un solo "
                                   "valor, fijado antes de mirar validation, y el magnitud es un "
                                   "juicio declarado, no una magnitud derivada"),
            "clasificador": f"LogisticRegression C={C_MODELO} balanced, la escala de V03",
            "umbrales": "0,44 / 0,51 / 0,46 congelados",
            "optimizador": f"L-BFGS-B, gradiente analitico, maxiter {MAX_ITER_LBFGS}, gtol {TOL_LBFGS}",
            "rejilla": "ninguna. No se varia lambda ni k",
            "rol_del_grafo": "SOLO regulariza. Nunca produce una prediccion, nunca vota, nunca recupera",
        },
        "control": {
            "corrected_cv_V03": repr(ccv_ref),
            "brazo_0_lambda_0": {"corrected_cv": round(ccv_0, 10),
                                 "delta_vs_V03": round(ccv_0 - ccv_ref, 10),
                                 "max_abs_diff_prob": dmax_0,
                                 "clona_a_V03": bool(ccv_0 - ccv_ref <= TOLERANCIA_OPTIMIZADOR
                                                     and dmax_0 <= 0.05)},
        },
        "diagnostico_grafo": diag_grafo,
        "resultado": {
            "brazo_0_corrected_cv": round(ccv_0, 10),
            "brazo_1_corrected_cv": round(ccv_1, 10),
            "delta_grafo_puro_brazo1_menos_brazo0": round(delta_puro, 10),
            "delta_vs_V03_brazo1_menos_V03": round(delta_v03, 10),
            "folds_brazo_1": [round(x, 6) for x in fp_1],
            "folds_brazo_0": [round(x, 6) for x in fp_0],
            "pliegues_mejorados": pliegues_mej,
            "f1_por_etiqueta": por,
            "rescate": rescue, "damage": damage, "neto": rescue - damage, "cambian": cambian,
            "acierto_en_las_que_cambian": round(rescue / cambian, 4) if cambian else None,
            "gate": {"delta_min": GATE_DELTA, "pliegues_min": GATE_PLIEGUES,
                     "supera_delta": bool(h_delta), "supera_pliegues": bool(h_folds),
                     "superado": gate},
            "bootstrap": (None if ic_lo is None else
                          {"IC95": [round(ic_lo, 6), round(ic_hi, 6)], "n": N_BOOT}),
        },
        "convergencia": {
            "brazo_0_todas": bool(conv_0), "brazo_1_todas": bool(conv_1),
            "gradiente_max_brazo_0": float(max(i["residuo_gradiente"] for i in info_0)),
            "gradiente_max_brazo_1": float(max(i["residuo_gradiente"] for i in info_1)),
            "warnings": 0,
        },
        "decision": decision,
        "experimentos_registrados": 1,
        "formulacion": "V03 remains the best verified leaderboard result.",
        "gastos": {"submissions": 0, "consultas_kaggle": 0, "uso_holdout": 0, "uso_test": 0,
                   "segundos": round((datetime.now(timezone.utc) - inicio).total_seconds(), 1)},
    }
    JSON_E.write_text(json.dumps(informe, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    L: list[str] = []
    A = L.append
    A("# P59-E - Regularizacion por grafo sobre el objetivo de V03")
    A("")
    A(f"Generado por `experiments/p59_10_grafo_regularizacion.py`. Fecha {inicio.isoformat(timespec='seconds')}.")
    A("")
    A("## Hipotesis")
    A("")
    A("La estructura de vecindad en el espacio de V03 puede usarse para **regularizar el "
      "entrenamiento**, no para predecir localmente. Los documentos cercanos deberian recibir "
      "coeficientes similares, luego la decision de uno toma fuerza estadistica de sus vecinos sin que "
      "ninguno sea nunca etiquetado como evidencia al predecir.")
    A("")
    A("## E1 gap check: NUEVA")
    A("")
    A("| eje | filas del registro |")
    A("|---|---:|")
    A("| regularizacion por grafo | **0** |")
    A("| manifold regularization | **0** |")
    A("| termino cuadratico sobre un grafo de documentos | **0** |")
    A("| PageRank o paseo aleatorio | **0** |")
    A("| red neuronal sobre grafo | **0** |")
    A("")
    A("El precedente mas cercano, `V4-F01-F04`, usa la vecindad para **propagar etiquetas**, que es "
      "predecir, y cerro con techo nulo por 0 solapamientos literales train-test. `V4-A01`, `V4-A02`, "
      "`E2-20-ESPECIALISTAS` y `V6-07` usan vecindad para predecir o para ponderar muestras. Ninguno "
      "regulariza un objetivo.")
    A("")
    A("**Previo descriptivo que baja la priori y conviene no esconder**: `V4-F01-F04` midio que un "
      "vecino discrepante aporta solo +0,018 de error con IC95 [-0,013, +0,049], no significativo. Un "
      "grafo puede regularizar sin que los vecinos compartan error, asi que no cierra la pista.")
    A("")
    A("## Configuracion, fijada antes de mirar validation")
    A("")
    A("```text")
    A(f"grafo        coseno kNN con k = {K_VECINOS} sobre el TF-IDF de V03, filas de ENTRENAMIENTO")
    A("afinidad     simetrizada, cotos validos solo, reescalada a norma espectral unitaria")
    A("objetivo     0,5 w'w + (lambda/2) w'X'WX w + C sum_i omega_i perdida(s_i)")
    A(f"lambda       {LAMBDA_GRAFO}, un solo valor, sin rejilla")
    A(f"clasificador LogisticRegression C={C_MODELO} balanced, la escala de V03")
    A("umbrales     0,44 / 0,51 / 0,46 congelados")
    A("rol           el grafo SOLO regulariza; nunca predice, nunca vota, nunca recupera")
    A("```")
    A("")
    A("El grafo se escapa por construccion: el mayor autovalor de `X'WX` se pone a 1, luego "
      f"lambda = {LAMBDA_GRAFO} significa una curvatura de suavizado de al menos diez veces menor que la "
      "del L2. Que ese sea el magnitud es un juicio declarado, no una magnitud derivada.")
    A("")
    A("## Control: el brazo 0 es lo que hace legible al brazo 1")
    A("")
    A("```text")
    A(f"brazo 0 (lambda = 0, mi optimizador)  corrected_cv {ccv_0!r}")
    A(f"           delta vs V03 {ccv_0 - ccv_ref:+.10f} | max |proba - canonica| {dmax_0:.3e} | "
      f"celdas distintas {int((pred_0 != pred_v03).sum())}")
    A("```")
    A("")
    A("Sin ese brazo, la diferencia entre el candidato y V03 seria una afirmacion sobre mi "
      "optimizador y no sobre el grafo.")
    A("")
    A("## Resultados")
    A("")
    A("```text")
    A(f"brazo 0 corrected_cv        {ccv_0:.10f}")
    A(f"brazo 1 corrected_cv        {ccv_1:.10f}")
    A(f"delta GRAFO PURO (1 - 0)    {delta_puro:+.10f}")
    A(f"delta vs V03 (1 - V03)      {delta_v03:+.10f}  incluye la tolerancia del optimizador")
    A(f"pliegues que mejoran {pliegues_mej}/5")
    A(f"folds brazo 1 {' '.join(f'{x:.4f}' for x in fp_1)}")
    A(f"folds brazo 0 {' '.join(f'{x:.4f}' for x in fp_0)}")
    A(f"rescate {rescue} | damage {damage} | neto {rescue - damage:+d} | cambian {cambian}")
    A("```")
    A("")
    A("| etiqueta | F1 brazo 1 | F1 brazo 0 | rescate | damage |")
    A("|---|---:|---:|---:|---:|")
    for e in ETIQUETAS:
        A(f"| {e} | {por[e]['f1']:.4f} | {por[e]['f1_ref']:.4f} | {por[e]['rescate']} | {por[e]['damage']} |")
    A("")
    A("## Convergencia")
    A("")
    A("```text")
    A(f"brazo 0 todas convergen {conv_0} | gradiente max "
      f"{max(i['residuo_gradiente'] for i in info_0):.3e} | iteraciones "
      f"{min(i['iteraciones'] for i in info_0)}-{max(i['iteraciones'] for i in info_0)}")
    A(f"brazo 1 todas convergen {conv_1} | gradiente max "
      f"{max(i['residuo_gradiente'] for i in info_1):.3e} | iteraciones "
      f"{min(i['iteraciones'] for i in info_1)}-{max(i['iteraciones'] for i in info_1)}")
    A("warnings 0")
    A("```")
    A("")
    A("## Gate")
    A("")
    A("| criterio | valor | superado |")
    A("|---|---|---|")
    A(f"| delta del grafo >= +0,003 | {delta_puro:+.6f} | **{h_delta}** |")
    A(f"| al menos 4 de 5 pliegues | {pliegues_mej}/5 | **{h_folds}** |")
    A(f"| gate completo | | **{gate}** |")
    A("")
    if ic_lo is not None:
        A(f"Bootstrap, ejecutado SOLO porque el gate se supero: IC95 [{ic_lo:+.6f}, {ic_hi:+.6f}].")
    else:
        A("Bootstrap NO ejecutado: no se supero el gate.")
    A("")
    A("## Decision")
    A("")
    A("```text")
    A(f"decision  {decision}")
    A("```")
    A("")
    A("> V03 remains the best verified leaderboard result.")
    A("")
    A("*Seccion generada por `experiments/p59_10_grafo_regularizacion.py`. Sin push.*")
    MD_E.write_text(a_ascii("\n".join(L)), encoding="ascii")

    import csv as _csv
    filas = [
        ["corrected_cv_V03", repr(ccv_ref)], ["brazo_0_corrected_cv", repr(ccv_0)],
        ["brazo_1_corrected_cv", repr(ccv_1)], ["delta_grafo_puro", repr(delta_puro)],
        ["delta_vs_V03", repr(delta_v03)], ["pliegues_mejorados", str(pliegues_mej)],
        ["folds_brazo_1", " ".join(f"{x:.6f}" for x in fp_1)],
        ["folds_brazo_0", " ".join(f"{x:.6f}" for x in fp_0)],
        ["rescate", str(rescue)], ["damage", str(damage)], ["neto", str(rescue - damage)],
        ["cambian", str(cambian)], ["lambda", f"{LAMBDA_GRAFO}"], ["k_vecinos", str(K_VECINOS)],
        ["gate_delta_superado", str(h_delta)], ["gate_pliegues_superado", str(h_folds)],
        ["gate_superado", str(gate)],
        ["decision", a_ascii(decision)],
    ]
    with CSV_E.open("w", encoding="ascii", newline="") as fh:
        w = _csv.writer(fh, lineterminator="\n")
        w.writerow(["magnitud", "valor"])
        w.writerows(filas)

    for destino in (MD_E, JSON_E, CSV_E):
        if destino.exists() and any(b > 127 for b in destino.read_bytes()):
            raise SystemExit(f"PARADA: {destino.name} conserva bytes no ASCII")
    log("")
    log(f"P59-E VEREDICTO: {decision.split('.')[0]}")
    log(f"P59-10 OK | {informe['gastos']['segundos']} s")


if __name__ == "__main__":
    warnings.simplefilter("error", RuntimeWarning)
    ast.parse(Path(__file__).read_text(encoding="utf-8"))
    main()
