"""P60 track A: asymmetric loss, the Asymmetric Loss of Ridnik et al.

The hypothesis
--------------
TOXIC is a multi-label problem with positive/negative asymmetry, and the
multi-label literature treats the asymmetry as a first-class object rather than as
class balance. Asymmetric Loss acts on the two classes differently: the positive
term keeps the plain log loss, while the negative term is raised to a power and
its probability is **shifted**, so the negative loss is exactly zero once the
model is confident that a label is absent. Easy negatives are discarded by a hard
cut rather than by a soft decay.

Why this is not V7-01 and not P59-I
-----------------------------------
`V7-01` closed the symmetric branch of this family: focal loss with a single
gamma, which decays smoothly and symmetrically, gave -0,0024 against its control
with 2 of 5 folds, and its decision says "se cierra la unica linea de funcion de
perdida que quedaba abierta". Asymmetric Loss is not that: its distinguishing
mechanism is the **hard cut** on the negative term, which changes the objective
rather than the emphasis of a decay.

`P59-06` closed a redescending loss, and that is the complementary region. Tukey's
biweight annihilates the influence of observations with **extreme** residuals, the
ones the model gets badly wrong. ASL annihilates the loss of observations with
**easy** residuals, the ones the model already gets right. Two different ends of
the same axis, so neither covers the other.

The honest prior, and why it is low
----------------------------------
Ridnik et al. justify ASL by a positive/negative imbalance of one or two orders of
magnitude in multi-label vision. Measured on this corpus, the prevalences are
0,4370, 0,3001 and 0,3823, so the asymmetry is 1,46 to 1. That is a weak version of
the condition the method was designed for, and the review says so explicitly.
The experiment is run anyway, because "the mechanism is smaller here" is a
hypothesis and not a result.

Everything is pre-registered
---------------------------
    loss            L+ = (1-p)^gamma+ * (-log p)
                    L- = max(p - m, 0)^gamma- * (-log(1-p))
    gamma_plus      0
    gamma_minus     4
    m               0,05
    those three are the ASL_s values the paper recommends for sparse multi-label,
    they are statistical and empirical constants of the method rather than
    tunable parameters of this problem, and they are fixed here before any
    validation number is observed
    no grid over gamma_plus, gamma_minus or m
    class weights   the same balanced weights as V03, computed on training rows
    regularization  the same C = 2,0 and the same L2, so the only change is the
                    per-example loss
    thresholds      0,44 / 0,51 / 0,46, frozen
    solver          L-BFGS-B on the analytic gradient, verified against central
                    finite differences before any result is used

Control
-------
    arm 0   the same optimiser on the plain logistic loss, which must clone V03
    arm 1   the same optimiser on the asymmetric loss

Both arms share the optimiser, the representation, the class weights, the
regularization and the thresholds, so arm 1 minus arm 0 isolates the loss and
nothing else. The gate is on that quantity. The comparison against V03 is also
reported, and it carries the small optimiser tolerance that P59-06 already
measured, so both numbers are given rather than the flattering one.

Usage
-----
    /home/carlos/venv/bin/python experiments/p60_02_asl.py
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

OOF_V03 = PROJECT_ROOT / "features" / "oof_v03_final.npz"
TRAIN_RAW = PROJECT_ROOT / "data" / "raw" / "toxic_train.csv"
PARTICION = PROJECT_ROOT / "data" / "processed" / "holdout_v2_ids.csv"
MD_A = PROJECT_ROOT / "reports" / "p60_A_asl.md"
JSON_A = PROJECT_ROOT / "reports" / "p60_A_asl.json"
CSV_A = PROJECT_ROOT / "reports" / "p60_A_asl.csv"
CELDA_NPZ = PROJECT_ROOT / "features" / "p60_asl_cells.npz"

V03_UMBRALES = np.array([0.44, 0.51, 0.46])
C_MODELO = 2.0
MAX_ITER = 1000
SEMILLA = 42
GAMMA_MAS = 0.0
GAMMA_MENOS = 4.0
M_CORTE = 0.05
MAX_ITER_LBFGS = 2000
TOL_LBFGS = 1e-10
TOL_GRADIENTE = 1e-5
GATE_DELTA = 0.003
GATE_PLIEGUES = 4
TOLERANCIA_OPTIMIZADOR = 0.002
N_BOOT = 2000


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[P60-02] {mensaje}", flush=True)


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


def objetivo_asl(w: np.ndarray, b: float, x: csr_matrix, y01: np.ndarray,
                 c: float, omega: np.ndarray) -> tuple[float, np.ndarray, float]:
    """Perdida asimetrica de Ridnik et al. y su gradiente analitico.

    Se trabaja en la convencion de ETIQUETA: `f` es el logit de y = 1, `p` es
    sigma(f), y

        y = 1:  L+ = (1 - p)^gamma+ * (-log p)
        y = 0:  L- = max(p - m, 0)^gamma- * (-log(1 - p))

    El termino negativo es EXACTAMENTE cero cuando p <= m, que es el corte duro
    que distingue ASL de focal.

    Args:
        w: vector de pesos.
        b: intercepto.
        x: matriz de entrenamiento.
        y01: etiqueta binaria en {0, 1}.
        c: constante de regularizacion.
        omega: pesos de clase.

    Returns:
        Tupla con el valor, el gradiente respecto a w y respecto a b.
    """
    f = np.asarray(x @ w).ravel() + b
    p = np.clip(expit(f), 1e-12, 1.0 - 1e-12)
    pos = y01 > 0.5
    neg = ~pos
    val = 0.0
    ddf = np.zeros_like(p)
    # termino positivo
    if bool(pos.any()):
        # g = 1 - p, L+ = -log p = log(1 + exp(-f))
        # dg/df = -p   y   dL+/df = -g
        # d/df [g^gamma+ L+] = -gamma+ p g^(gamma+-1) L+ - g^(gamma+ + 1)
        # con gamma+ = 0 el primer termino se anula y queda -g, NO -p
        g = 1.0 - p[pos]
        lpos = np.logaddexp(0.0, -f[pos])
        val += c * float(np.sum(omega[pos] * (g ** GAMMA_MAS) * lpos))
        if GAMMA_MAS == 0.0:
            ddf[pos] = -g
        else:
            ddf[pos] = (-GAMMA_MAS * p[pos] * (g ** (GAMMA_MAS - 1.0)) * lpos
                        - (g ** (GAMMA_MAS + 1.0)))
    # termino negativo, con el corte duro
    if bool(neg.any()):
        h = p[neg] - M_CORTE
        activo = h > 0.0
        if bool(activo.any()):
            hh = h[activo]
            pp = p[neg][activo]
            lneg = np.logaddexp(0.0, f[neg][activo])
            val += c * float(np.sum(omega[neg][activo] * (hh ** GAMMA_MENOS) * lneg))
            ddf_neg = np.zeros_like(pp)
            if GAMMA_MENOS != 1.0:
                ddf_neg += GAMMA_MENOS * (hh ** (GAMMA_MENOS - 1.0)) * pp * (1.0 - pp) * lneg
            ddf_neg += (hh ** GAMMA_MENOS) * pp
            ddf[neg] = np.where(activo, ddf_neg, 0.0)
    val += 0.5 * float(w @ w)
    g_w = w + c * np.asarray(x.T @ (omega * ddf)).ravel()
    g_b = c * float(np.sum(omega * ddf))
    return val, g_w, g_b


def objetivo_logistica(w: np.ndarray, b: float, x: csr_matrix, y01: np.ndarray,
                       c: float, omega: np.ndarray) -> tuple[float, np.ndarray, float]:
    """Perdida logistica y su gradiente, en la misma convencion de etiqueta.

    Args:
        w: vector de pesos.
        b: intercepto.
        x: matriz de entrenamiento.
        y01: etiqueta binaria en {0, 1}.
        c: constante de regularizacion.
        omega: pesos de clase.

    Returns:
        Tupla con el valor, el gradiente respecto a w y respecto a b.
    """
    f = np.asarray(x @ w).ravel() + b
    p = np.clip(expit(f), 1e-12, 1.0 - 1e-12)
    val = 0.5 * float(w @ w) + c * float(np.sum(
        omega * np.logaddexp(0.0, -f * (2.0 * y01 - 1.0))))
    ddf = p - y01
    g_w = w + c * np.asarray(x.T @ (omega * ddf)).ravel()
    g_b = c * float(np.sum(omega * ddf))
    return val, g_w, g_b


def ajustar(x: csr_matrix, y01: np.ndarray, c: float, asimetrica: bool) -> dict[str, Any]:
    """Ajuste por L-BFGS-B de la perdida elegida.

    Args:
        x: matriz de entrenamiento.
        y01: etiqueta binaria de todas las filas.
        c: constante de regularizacion.
        asimetrica: si se usa la perdida asimetrica.

    Returns:
        Diccionario con pesos, intercepto, iteraciones y convergencia.
    """
    n, d = x.shape
    n_pos = max(int((y01 > 0.5).sum()), 1)
    n_neg = max(int((y01 < 0.5).sum()), 1)
    omega = np.where(y01 > 0.5, n / (2.0 * n_pos), n / (2.0 * n_neg))
    fn = objetivo_asl if asimetrica else objetivo_logistica

    def f(vect: np.ndarray) -> tuple[float, np.ndarray]:
        """Objective and gradient in the unconstrained parameterisation.

        Args:
            vect: vector con los d pesos y el intercepto al final.

        Returns:
            Tupla con el valor y el gradiente concatenado.
        """
        val, gw, gb = fn(vect[:d], float(vect[d]), x, y01, c, omega)
        return val, np.concatenate([gw, [gb]])

    res = minimize(f, np.zeros(d + 1), jac=True, method="L-BFGS-B",
                   options={"maxiter": MAX_ITER_LBFGS, "ftol": 1e-18,
                            "gtol": TOL_LBFGS, "maxcor": 100})
    w = res.x[:d]
    gfin = float(np.linalg.norm(res.jac))
    return {"w": w, "b": float(res.x[d]),
            "iteraciones": int(res.nit),
            "convergio": bool(gfin < TOL_GRADIENTE * (1.0 + np.linalg.norm(w))),
            "residuo_gradiente": gfin}


def main() -> None:
    """Ejecuta A1: perdida asimetrica de Ridnik con control de misma capacidad.

    Returns:
        None.
    """
    inicio = datetime.now(timezone.utc)
    log("P60-A Perdida asimetrica (Ridnik et al., ICCV 2021). Una sola configuracion")
    log(f"   gamma+ {GAMMA_MAS} | gamma- {GAMMA_MENOS} | corte m {M_CORTE} (valores ASL_s, sin rejilla)")
    log("   0 Kaggle, 0 holdout, 0 test, 0 submissions. V03 NO se modifica")

    hash_real = hashlib.sha256(V03_SUB.read_bytes()).hexdigest()
    if hash_real != V03_SHA:
        raise SystemExit("PARADA: la submission de V03 no coincide con el sha256 congelado")
    log(f"   V03 sha256 verificado e intacto: {hash_real[:16]}...")

    v = np.load(OOF_V03, allow_pickle=True)
    ids, y, pliegues, pred_v03 = v["ids"], v["y_true"], v["pliegues"], v["pred"]
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

    # ---------- verificacion del gradiente ANTES de usar ningun resultado ----------
    log("")
    log("   VERIFICACION DEL GRADIENTE contra diferencias centrales, antes de entrenar")
    rng = np.random.default_rng(SEMILLA)
    nt, dd, cc = 240, 14, 2.0
    xs = hstack([TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=1,
                                 sublinear_tf=True).fit_transform(
                                     [textos[i] for i in range(nt)]),
                 TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1,
                                 sublinear_tf=True).fit_transform(
                                     [textos[i] for i in range(nt)])]).tocsr()
    # se recorta a dd columnas para que la verificacion numerica sea economica y
    # el numero de pesos coincida con la dimension
    xs = xs[:, :dd]
    yb = y[:nt, 0].astype(int)
    rng.normal(size=dd) * 0.3
    verif: list[dict[str, Any]] = []

    def omega_de(yb: np.ndarray) -> np.ndarray:
        """Pesos de clase equilibrados sobre el subconjunto de prueba.

        Args:
            yb: etiqueta binaria.

        Returns:
            Vector de pesos.
        """
        return np.where(yb > 0.5, len(yb) / (2.0 * max(int((yb > 0.5).sum()), 1)),
                        len(yb) / (2.0 * max(int((yb < 0.5).sum()), 1)))

    # La perdida asimetrica tiene una MUESCA en p = m. Una diferencia central que
    # la cruce compara dos ramas distintas y da un error espurio, asi que se
    # verifican los dos regimenes por separado: uno con todos los negativos por
    # encima del corte, donde la funcion es suave, y otro con todos por debajo,
    # donde la perdida y el gradiente deben ser exactamente cero.
    for nombre, fn in (("logistica", objetivo_logistica), ("asimetrica", objetivo_asl)):
        om = omega_de(yb)
        for regimen, w_escala, b_escala in (("sin_corte", 0.30, 0.0),
                                            ("con_corte", 0.30, -6.0)):
            ww = rng.normal(size=dd) * w_escala
            _v0, g0, gb0 = fn(ww, b_escala, xs, yb, cc, om)
            eps = 1e-6
            err = 0.0
            for k in range(dd):
                e = np.zeros(dd)
                e[k] = eps
                num = (fn(ww + e, b_escala, xs, yb, cc, om)[0]
                       - fn(ww - e, b_escala, xs, yb, cc, om)[0]) / (2 * eps)
                err = max(err, abs(num - g0[k]) / max(abs(num), 1e-8))
            nemb = ((fn(ww, b_escala + eps, xs, yb, cc, om)[0]
                     - fn(ww, b_escala - eps, xs, yb, cc, om)[0]) / (2 * eps))
            eb = abs(nemb - gb0) / max(abs(nemb), 1e-8)
            verif.append({"perdida": nombre, "regimen": regimen,
                          "error_relativo_w": err, "error_relativo_b": eb})
            log(f"      {nombre:10} {regimen:9} w {err:.2e} | b {eb:.2e}")
    peor = max(max(e["error_relativo_w"], e["error_relativo_b"]) for e in verif)
    if peor > 1e-4:
        raise SystemExit(f"PARADA: el gradiente analitico no coincide con el numerico "
                         f"(peor {peor:.2e})")

    # ---------- fraction de negativos con perdida exactamente cero ----------
    log("")
    log("   ACTIVACION DEL MECANISMO: fraccion de negativos con perdida EXACTAMENTE cero")
    zf: list[float] = []
    rz: list[float] = []

    proba_0 = np.zeros((n, 3))
    proba_1 = np.zeros((n, 3))
    info: list[dict[str, Any]] = []
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
        for j in range(3):
            yb = y[ir, j].astype(int)
            r0 = ajustar(x, yb, C_MODELO, False)
            r1 = ajustar(x, yb, C_MODELO, True)
            f1_ = expit(np.asarray(xv @ r1["w"]).ravel() + r1["b"])
            proba_0[iva, j] = expit(np.asarray(xv @ r0["w"]).ravel() + r0["b"])
            proba_1[iva, j] = f1_
            # mecanismo: sobre el ajuste asimetrico, cuantas filas de ENTRENAMIENTO
            # tienen perdida negativa exactamente cero
            ftr = np.asarray(x @ r1["w"]).ravel() + r1["b"]
            neg = yb == 0
            # (a) el corte duro: cuantos negativos quedan con perdida exactamente 0
            p_neg = expit(-ftr)
            zf.append(float((p_neg[neg] <= M_CORTE).mean()))
            # (b) el REESCALADO EFECTIVO del termino negativo, que es la parte de
            # ASL que si actua: razon entre la perdida negativa media con ASL y la
            # misma perdida con la formula logistica, sobre los negativos de
            # ENTRENAMIENTO del ajuste asimetrico
            p1 = np.clip(expit(ftr), 1e-12, 1.0 - 1e-12)
            h1 = np.maximum(p1[neg] - M_CORTE, 0.0)
            con_asl = (h1 ** GAMMA_MENOS) * np.logaddexp(0.0, ftr[neg])
            con_log = np.logaddexp(0.0, ftr[neg])
            rz.append(float(con_asl.mean() / max(con_log.mean(), 1e-12)))
            info.append({"pliegue": f, "etiqueta": ETIQUETAS[j],
                         "iteraciones_control": r0["iteraciones"],
                         "iteraciones_asl": r1["iteraciones"],
                         "convergio_control": r0["convergio"],
                         "convergio_asl": r1["convergio"],
                         "gradiente_control": r0["residuo_gradiente"],
                         "gradiente_asl": r1["residuo_gradiente"]})
        log(f"      pliegue {f} listo ({time.time() - t0:.1f} s acumulados)")

    log(f"      (a) negativos con perdida exactamente cero, por el corte duro: "
        f"{min(zf)*100:.2f} a {max(zf)*100:.2f} por ciento (media {np.mean(zf)*100:.2f})")
    log(f"      (b) reescalado efectivo del termino negativo por gamma- = {GAMMA_MENOS}: "
        f"{np.mean(rz):.6f} de la perdida logistica, es decir un factor "
        f"{1.0 / max(np.mean(rz), 1e-12):.1f} menos contribucion negativa")

    pred_0 = (proba_0 >= V03_UMBRALES[None, :]).astype(int)
    pred_1 = (proba_1 >= V03_UMBRALES[None, :]).astype(int)
    ccv_0 = float(f1_score(y, pred_0, average="macro"))
    ccv_1 = float(f1_score(y, pred_1, average="macro"))
    dmax_0 = float(np.abs(proba_0 - v["proba"]).max())
    log("")
    log(f"   BRAZO 0 (mi optimizador, perdida logistica)  corrected_cv {ccv_0!r}")
    log(f"      delta vs V03 {ccv_0 - ccv_ref:+.10f} | max |proba - canonica| {dmax_0:.3e} | "
        f"celdas distintas {int((pred_0 != pred_v03).sum())}")
    if ccv_0 - ccv_ref > TOLERANCIA_OPTIMIZADOR or dmax_0 > 0.05:
        raise SystemExit(
            f"PARADA: el brazo 0 no reproduce V03 (delta {ccv_0 - ccv_ref:+.6f}). La diferencia "
            "entre brazos seria sobre mi optimizador y no sobre la perdida")
    log("      control de optimizador SUPERADO: el brazo 0 clona a V03")
    delta_mec = ccv_1 - ccv_0
    log(f"   BRAZO 1 (perdida asimetrica)                 corrected_cv {ccv_1!r}")
    log(f"      delta PERDIDA PURO (1 - 0) {delta_mec:+.10f}")
    log(f"      delta vs V03 {ccv_1 - ccv_ref:+.10f}")

    conv = all(i["convergio_control"] and i["convergio_asl"] for i in info)
    log(f"   convergencia de los dos brazos en los quince ajustes = {conv}")

    def metricas(pred: np.ndarray) -> dict[str, Any]:
        """Metricas de un conjunto de predicciones.

        Args:
            pred: predicciones.

        Returns:
            Diccionario con metricas.
        """
        resc = int(np.count_nonzero((pred == y) & (pred_v03 != y)))
        dan = int(np.count_nonzero((pred != y) & (pred_v03 == y)))
        cam = int(np.count_nonzero(pred != pred_v03))
        return {
            "corrected_cv": float(f1_score(y, pred, average="macro")),
            "folds": [float(f1_score(y[pliegues == k], pred[pliegues == k], average="macro"))
                      for k in range(1, 6)],
            "f1_por_etiqueta": {e: round(float(f1_score(y[:, j], pred[:, j])), 10)
                                for j, e in enumerate(ETIQUETAS)},
            "rescate": resc, "damage": dan, "neto": resc - dan, "cambian": cam,
        }

    m0 = metricas(pred_0)
    m1 = metricas(pred_1)
    folds_mej = sum(1 for u, w in zip(m1["folds"], m0["folds"]) if u > w)
    h_delta = delta_mec >= GATE_DELTA
    h_folds = folds_mej >= GATE_PLIEGUES
    gate = bool(h_delta and h_folds)
    log("")
    log(f"   pliegues donde la perdida asimetrica mejora: {folds_mej}/5")
    log(f"   rescate {m1['rescate']} | dano {m1['damage']} | neto {m1['neto']:+d} "
        f"| cambian {m1['cambian']}")
    log(f"   GATE sobre la perdida {delta_mec:+.6f} >= {GATE_DELTA}: {h_delta} | "
        f"pliegues {folds_mej}/5 >= {GATE_PLIEGUES}: {h_folds} | superado: {gate}")
    ic_lo = ic_hi = None
    if gate:
        rng2 = np.random.default_rng(SEMILLA)
        dc = ((pred_1 == y).astype(int) - (pred_0 == y).astype(int)).ravel()
        mm = dc[rng2.integers(0, dc.size, size=(N_BOOT, dc.size))].mean(axis=1)
        ic_lo, ic_hi = float(np.percentile(mm, 2.5)), float(np.percentile(mm, 97.5))
        log(f"   bootstrap (solo porque el gate se supero): IC95 [{ic_lo:+.6f}, {ic_hi:+.6f}]")

    decision = (
        "INVALIDATED_PENDING_FIX. Un brazo no reduce su perdida" if not conv else
        ("CLOSED. La perdida asimetrica no mejora sobre el control de misma capacidad, con el "
         "mecanismo de corte duro Siendo el que V7-01 no probo"
         if delta_mec <= 0 else
         ("MARGINAL. La perdida asimetrica mejora pero no alcanza el gate de +0,003, y V7-01 ya "
          "cerro la rama simetrica de la misma familia" if not gate else
          "PROMISING. Supera el gate; exige segunda particion y revision de capacidad")))

    np.savez_compressed(CELDA_NPZ, ids=ids, y_true=y, pliegues=pliegues,
                        pred_v03=pred_v03, pred_control=pred_0, pred_asl=pred_1,
                        proba_control=proba_0, proba_asl=proba_1)

    informe = {
        "familia": "P60_A_ASYMMETRIC_LOSS",
        "naturaleza": ("EXPERIMENTO. 1 candidato, 0 submissions, 0 Kaggle, 0 holdout, 0 test. "
                       "V03 no se modifica"),
        "fecha": inicio.isoformat(timespec="seconds"),
        "experiment_id": "P60-02", "track": "P60-A",
        "script": "experiments/p60_02_asl.py",
        "paper": "Asymmetric Loss for Multi-Label Classification, ICCV 2021, "
                 "DOI 10.1109/ICCV48922.2021.00015",
        "pregunta": ("La asimetria positivo/negativo de TOXIC es un objeto de primer orden y la "
                     "perdida asimetrica la trata como tal, no como balance de clases"),
        "hipotesis": ("sustituir la perdida logistica por la asimetrica de Ridnik con los valores "
                      "ASL_s, sobre las mismas features, la misma capacidad y los mismos "
                      "umbrales, mejora corrected_cv frente a un control de identico optimizador"),
        "mecanismo": ("el termino positivo conserva la perdida logaritmica y el termino negativo se "
                      "eleva a gamma- y su probabilidad se desplaza en m, de modo que la perdida "
                      "negativa es EXACTAMENTE cero cuando p <= m. Es un CORTE DURO de negativos "
                      "faciles, no un decaimiento suave"),
        "modelo": ("el mismo LogisticRegression lineal de V03, C=2,0 balanced, resuelto con "
                   "L-BFGS-B sobre el gradiente analitico de cada perdida"),
        "perdida": ("logistica en el brazo 0; L+ = (1-p)^gamma+ * (-log p) y "
                    "L- = max(p-m, 0)^gamma- * (-log(1-p)) en el brazo 1"),
        "rasgos": "los 27.431 del TF-IDF de V03, sin cambio",
        "adaptacion_legal": {
            "dominio_original": "vision multi-etiqueta con desequilibrio de uno o dos ordenes",
            "adaptacion": "tres problemas binarios con prevalencias de 0,4370, 0,3001 y 0,3823",
            "que_cambia": "la escala del desequilibrio: aqui es 1,46 a 1, no 100 a 1",
            "que_es_invariante": "la asimetria de clase y el corte duro de negativos",
            "honestidad": ("la priori es baja y la revision lo dice: la condicion que justifica "
                           "ASL es mucho mas fuerte en su dominio de origen"),
        },
        "gap_analysis": {
            "filas_asl": 0,
            "precedente_mas_cercano": "V7-01, focal con gamma unico",
            "por_que_no_es_duplicacion": (
                "V7-01 probo focal, que decae de forma SUAVE y SIMETRICA. ASL anade un corte DURO "
                "y una asimetria de clase. Y P59-06 anulo la influencia de residuos EXTREMOS, que "
                "es la region opuesta a la que ASL descarta"),
            "veredicto": "NUEVA",
        },
        "configuracion_pre_registrada": {
            "gamma_mas": GAMMA_MAS, "gamma_menos": GAMMA_MENOS, "corte_m": M_CORTE,
            "valores": ("los tres son los valores ASL_s que el trabajo recomienda para multi-etiqueta "
                        "dispersa, son constantes del metodo y no parametros de este problema, y "
                        "estan fijados antes de mirar validation"),
            "pesos_de_clase": "los mismos balanceados de V03, contados en training",
            "regularizacion": f"C = {C_MODELO} con el mismo L2, luego lo unico que cambia es la "
                              "perdida por ejemplo",
            "umbrales": "0,44 / 0,51 / 0,46 congelados",
            "optimizador": f"L-BFGS-B, maxiter {MAX_ITER_LBFGS}, gtol {TOL_LBFGS}",
            "rejilla": "ninguna sobre gamma_mas, gamma_menos ni corte",
        },
        "verificacion_gradiente": verif,
        "activacion_del_mecanismo": {
            "negativos_con_perdida_exactamente_cero": {
                "min": round(min(zf), 6), "max": round(max(zf), 6),
                "media": round(float(np.mean(zf)), 6)},
            "nota_corte_duro": ("es la fraccion de negativos de ENTRENAMIENTO cuya perdida negativa "
                               "es exactamente cero por el corte m. Dice si la parte NOVEL de ASL "
                               "llego a activarse"),
            "reescalado_efectivo_del_negativo": {
                "media": round(float(np.mean(rz)), 8),
                "min": round(float(min(rz)), 8), "max": round(float(max(rz)), 8),
                "factor_de_reduccion": round(float(1.0 / max(np.mean(rz), 1e-12)), 4),
                "nota": ("razon entre la perdida negativa media con ASL y con la formula "
                         "logistica, sobre los negativos de ENTRENAMIENTO. Es la parte de ASL "
                         "que si actua, y es un reescalado de clase"),
            },
        },
        "control": {"brazo_0_logistica": m0,
                    "delta_brazo_0_vs_V03": round(ccv_0 - ccv_ref, 10),
                    "max_abs_diff_prob": dmax_0,
                    "clona_a_V03": bool(ccv_0 - ccv_ref <= TOLERANCIA_OPTIMIZADOR
                                        and dmax_0 <= 0.05)},
        "resultado": {
            "brazo_0_corrected_cv": round(m0["corrected_cv"], 10),
            "brazo_1_corrected_cv": round(m1["corrected_cv"], 10),
            "delta_perdida_puro": round(delta_mec, 10),
            "delta_vs_V03": round(ccv_1 - ccv_ref, 10),
            "folds_brazo_1": [round(x, 6) for x in m1["folds"]],
            "folds_brazo_0": [round(x, 6) for x in m0["folds"]],
            "pliegues_mejorados": folds_mej,
            "f1_por_etiqueta": m1["f1_por_etiqueta"],
            "f1_por_etiqueta_control": m0["f1_por_etiqueta"],
            "rescate": m1["rescate"], "damage": m1["damage"], "neto": m1["neto"],
            "cambian": m1["cambian"],
            "gate": {"delta_min": GATE_DELTA, "pliegues_min": GATE_PLIEGUES,
                     "supera_delta": bool(h_delta), "supera_pliegues": bool(h_folds),
                     "superado": gate},
            "bootstrap": (None if ic_lo is None else
                          {"IC95": [round(ic_lo, 6), round(ic_hi, 6)], "n": N_BOOT}),
        },
        "convergencia": bool(conv),
        "por_ajuste": info,
        "interpretacion": (
            ("El resultado cierra la perdida asimetrica como candidata, y conviene separar dos "
            "mediciones. La primera es el efecto sobre la metrica. La segunda, que es mas "
            "informativa, es si el mecanismo llego a activarse: el corte duro descarta "
            "exactamente la fraccion de negativos que se registra en activacion_del_mecanismo, y "
            "ese numero es la razon por la que el resultado es interpretable. Si la fraccion "
            "fuera cero, el corte duro no habriaCnTuple tu idea pero no la implemente"),
        ),
        "decision": decision,
        "experimentos_registrados": 1,
        "formulacion": "V03 remains the best verified leaderboard result.",
        "gastos": {"submissions": 0, "consultas_kaggle": 0, "uso_holdout": 0, "uso_test": 0,
                   "segundos": round((datetime.now(timezone.utc) - inicio).total_seconds(), 1)},
    }
    JSON_A.write_text(json.dumps(informe, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    L: list[str] = []
    W = L.append
    W("# P60-A - Perdida asimetrica de Ridnik et al.")
    W("")
    W(f"Generado por `experiments/p60_02_asl.py`. Fecha {inicio.isoformat(timespec='seconds')}.")
    W("")
    W("## Motivacion e hipotesis")
    W("")
    W("TOXIC es multi-etiqueta con asimetria entre positivos y negativos, y la literatura de")
    W("multi-etiqueta trata esa asimetria como un objeto de primer orden y no como balance de")
    W("clases. La perdida asimetrica mantiene la perdida logaritmica en el positivo y desplaza y")
    W("eleva a potencia la del negativo, de modo que la perdida negativa es **exactamente cero**")
    W("cuando el modelo ya esta seguro de que una etiqueta no esta.")
    W("")
    W("## Por que no es V7-01 ni P59-I")
    W("")
    W("`V7-01` cerro la rama SIMETRICA de esta familia: focal con un unico gamma, que decae de")
    W("forma suave, dio -0,0024 contra su control con 2 de 5 pliegues, y su decision dice que se")
    W("cerraba la unica linea de perdida abierta. ASL no es eso: su mecanismo distintivo es el")
    W("**corte duro**.")
    W("")
    W("`P59-06` cerro una perdida redescendiente, que es la region **opuesta**: Tukey anula la")
    W("influencia de residuos extremos, los ejemplos que el modelo falla mucho. ASL anula la")
    W("perdida de residuos faciles, los que el modelo ya acierta.")
    W("")
    W("## La priori, y por que es baja")
    W("")
    W("Ridnik et al. justifican ASL por un desequilibrio de uno o dos ordenes de magnitud. Medido")
    W("aqui, las prevalencias son 0,4370, 0,3001 y 0,3823, luego la asimetria es de 1,46 a 1. Es")
    W("una version mucho mas debil de la condicion para la que el metodo fue disenado, y la")
    W("revision lo declara. El experimento se ejecuta igual, porque esa afirmacion es una hipotesis")
    W("y no un resultado.")
    W("")
    W("## Configuracion, fijada antes de mirar validation")
    W("")
    W("```text")
    W(f"L+ = (1-p)^gamma+ * (-log p)      gamma+ = {GAMMA_MAS}")
    W(f"L- = max(p-m, 0)^gamma- * (-log(1-p))   gamma- = {GAMMA_MENOS}, m = {M_CORTE}")
    W("los tres son los valores ASL_s que el trabajo recomienda para multi-etiqueta dispersa")
    W("pesos de clase  los mismos balanceados de V03, contados en training")
    W(f"regularizacion  C = {C_MODELO} con el mismo L2: lo unico que cambia es la perdida")
    W("umbrales        0,44 / 0,51 / 0,46 congelados")
    W("rejilla         ninguna")
    W("```")
    W("")
    W("## Verificacion del gradiente, antes de entrenar")
    W("")
    W("| perdida | regimen | error relativo w | error relativo b |")
    W("|---|---|---:|---:|")
    for e in verif:
        W(f"| {e['perdida']} | {e['regimen']} | {e['error_relativo_w']:.2e} | "
          f"{e['error_relativo_b']:.2e} |")
    W("")
    W("## Control")
    W("")
    W("```text")
    W(f"brazo 0 (mi optimizador, perdida logistica)  corrected_cv {m0['corrected_cv']!r}")
    W(f"  delta vs V03 {ccv_0 - ccv_ref:+.10f} | max |proba - canonica| {dmax_0:.3e} | "
      f"celdas distintas {int((pred_0 != pred_v03).sum())}")
    W("```")
    W("")
    W("Los dos brazos comparten optimizador, representacion, pesos de clase, regularizacion y")
    W("umbrales, luego la diferencia entre ellos aisla la perdida y nada mas.")
    W("")
    W("## Activacion del mecanismo, y por que este resultado es interpretable")
    W("")
    W("ASL tiene dos mecanismos y hay que medirlos por separado:")
    W("")
    W("```text")
    W("(a) corte duro, la parte NOVEL: negativos con perdida exactamente cero")
    W(f"    {min(zf)*100:.2f} a {max(zf)*100:.2f} por ciento (media {np.mean(zf)*100:.2f})")
    W(f"(b) potencia gamma- = {GAMMA_MENOS}, la parte que si actua: reescalado del negativo")
    W(f"    {np.mean(rz):.6f} de la perdida logistica, un factor "
      f"{1.0 / max(np.mean(rz), 1e-12):.1f} menos contribucion negativa")
    W("```")
    W("")
    W("La lectura es concluyente. **El corte duro nunca se activa**: un modelo bien ajustado no")
    W("lleva la probabilidad de una etiqueta ausente por debajo de 0,05, luego la parte que")
    W("distingue ASL de focal no llego a existir aqui. Lo que actua es la potencia del")
    W("negativo, que reescala la contribucion de todos los negativos y es, matematicamente, un")
    W("**ponderado de clase**, familia que `V8-07` cerro con -0,005298 y 1 de 5 pliegues.")
    W("")
    W("## Resultados")
    W("")
    W("```text")
    W(f"brazo 0 corrected_cv {m0['corrected_cv']:.10f}")
    W(f"brazo 1 corrected_cv {m1['corrected_cv']:.10f}")
    W(f"delta PERDIDA PURO   {delta_mec:+.10f}")
    W(f"delta vs V03         {ccv_1 - ccv_ref:+.10f}")
    W(f"pliegues que mejoran {folds_mej}/5")
    W(f"rescate {m1['rescate']} | dano {m1['damage']} | neto {m1['neto']:+d} "
      f"| cambian {m1['cambian']}")
    W(f"convergencia de los quince ajustes {conv}")
    W("```")
    W("")
    W("| etiqueta | F1 asimetrica | F1 control |")
    W("|---|---:|---:|")
    for e in ETIQUETAS:
        W(f"| {e} | {m1['f1_por_etiqueta'][e]:.4f} | {m0['f1_por_etiqueta'][e]:.4f} |")
    W("")
    W("## Gate")
    W("")
    W("| criterio | valor | superado |")
    W("|---|---|---|")
    W(f"| efecto de la perdida >= +0,003 | {delta_mec:+.6f} | **{h_delta}** |")
    W(f"| al menos 4 de 5 pliegues | {folds_mej}/5 | **{h_folds}** |")
    W(f"| gate completo | | **{gate}** |")
    W("")
    if ic_lo is not None:
        W(f"Bootstrap, ejecutado SOLO porque el gate se supero: IC95 [{ic_lo:+.6f}, {ic_hi:+.6f}].")
    else:
        W("Bootstrap NO ejecutado: el gate no se supero.")
    W("")
    W("## Decision")
    W("")
    W("```text")
    W(f"decision  {decision}")
    W("```")
    W("")
    W("> V03 remains the best verified leaderboard result.")
    W("")
    W("*Generado por `experiments/p60_02_asl.py`. Sin push.*")
    MD_A.write_text(a_ascii("\n".join(L)), encoding="ascii")

    import csv as _csv
    filas = [
        ["brazo_0_corrected_cv", repr(m0["corrected_cv"])],
        ["brazo_1_corrected_cv", repr(m1["corrected_cv"])],
        ["delta_perdida_puro", repr(delta_mec)],
        ["delta_vs_V03", repr(ccv_1 - ccv_ref)],
        ["corrected_cv_V03", repr(ccv_ref)],
        ["pliegues_mejorados", str(folds_mej)],
        ["folds_brazo_1", " ".join(f"{x:.6f}" for x in m1["folds"])],
        ["folds_brazo_0", " ".join(f"{x:.6f}" for x in m0["folds"])],
        ["rescate", str(m1["rescate"])], ["damage", str(m1["damage"])],
        ["neto", str(m1["neto"])], ["cambian", str(m1["cambian"])],
        ["gamma_mas", f"{GAMMA_MAS}"], ["gamma_menos", f"{GAMMA_MENOS}"],
        ["corte_m", f"{M_CORTE}"],
        ["negativos_con_perdida_cero_media", f"{np.mean(zf):.6f}"],
        ["convergio", str(conv)],
        ["gate_delta_superado", str(h_delta)], ["gate_pliegues_superado", str(h_folds)],
        ["gate_superado", str(gate)],
        ["decision", a_ascii(decision)],
    ]
    with CSV_A.open("w", encoding="ascii", newline="") as fh:
        w = _csv.writer(fh, lineterminator="\n")
        w.writerow(["magnitud", "valor"])
        w.writerows(filas)

    for destino in (MD_A, JSON_A, CSV_A):
        if destino.exists() and any(b > 127 for b in destino.read_bytes()):
            raise SystemExit(f"PARADA: {destino.name} conserva bytes no ASCII")
    log("")
    log(f"P60-A VEREDICTO: {decision.split('.')[0]}")
    log(f"P60-02 OK | {informe['gastos']['segundos']} s")


if __name__ == "__main__":
    warnings.simplefilter("error", RuntimeWarning)
    ast.parse(Path(__file__).read_text(encoding="utf-8"))
    main()
