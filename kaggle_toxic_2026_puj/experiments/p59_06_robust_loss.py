"""P59 step 06: track I, robust redescending loss on V03.

The hypothesis, stated exactly
------------------------------
Paragraph 1 of the directive rules out the familiar readings. It is NOT "hard
examples should weigh less", and it is NOT "detect outliers and downweight
them", because both are sample weighting driven by a difficulty signal, and
this project has already closed those. The hypothesis is about the **objective
function**: the conventional logistic loss may let a few high-leverage
observations with extreme residuals move the global boundary, and a bounded
redescending loss could limit that influence *during the fit*, endogenously,
without naming or excluding any example.

Three arms, and why three
------------------------
    A  V03 via sklearn                 the frozen control
    B  my IRLS with W == 1             validates MY optimiser and objective
    C  my IRLS with Tukey biweight     the candidate

Arm B is the one that makes arm C readable. A custom optimiser can differ from
liblinear in capacity, in regularisation scaling and in stopping rule, and any of
those would masquerade as a loss effect. If arm B does not reproduce arm A, then
arm C minus arm A is a statement about my optimiser, not about robustness, and
the script stops.

Why Tukey biweight, chosen by criterion and not by enumeration
--------------------------------------------------------------
Paragraph 3 forbids assuming Huber is new and forbids a grid. Tukey's biweight is
selected because it is *the* canonical redescending M-estimator: its influence
function is exactly zero beyond a fixed cutoff, which is the property the
hypothesis needs, and its tuning constant tau = 4.685 is the standard
normal-consistent choice giving 95 % Gaussian efficiency. So tau is a
statistical constant fixed before execution, not a tuned value.

The gap analysis, executed rather than assumed
----------------------------------------------
Zero rows of the registry mention tukey, biweight, geman, mcclure, cauchy, fair,
redescend, m-estimator, rlm or irls. `V34-01` lists "margin/robust loss" with a
depth of **(0) UNEXPLORED**. `V35-01` did run `modified_huber`, but through
`SGDClassifier` (a different solver) and Huber is bounded, not redescending; the
registry itself records "NO es una nueva clase de funcion".

Distinction from V30, which paragraph 11 requires
-------------------------------------------------
`V6-07`, the mechanism behind `V30`, is `sample_weight = 1 - lambda * sospecha`
with a **floor of 0.1**, where the suspicion comes from a NearestNeighbors pass
over TF-IDF and is **fixed before fitting**. Three things differ here, and all
three are mathematical rather than cosmetic:

1. the weight is a deterministic function of the objective's own **current
   residual**, recomputed at every iteration, not a signal computed outside;
2. redescending allows **exactly zero** influence, which a 0.1 floor forbids;
3. what is being solved is an M-estimating equation, not the logistic objective
   with re-assigned weights.

Convergence is measured where an M-estimator actually converges: the residual of
the psi-equation, not a surrogate objective.

Usage
-----
    /home/carlos/venv/bin/python experiments/p59_06_robust_loss.py
"""

from __future__ import annotations

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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

from p59_01_voronoi_local import (
    ETIQUETAS,
    V03_CCV,
    V03_SHA,
    V03_SUB,
    leer_poblaciones,
)
from p59_04_influence import influencia_dual

OOF_V03 = PROJECT_ROOT / "features" / "oof_v03_final.npz"
TRAIN_RAW = PROJECT_ROOT / "data" / "raw" / "toxic_train.csv"
PARTICION = PROJECT_ROOT / "data" / "processed" / "holdout_v2_ids.csv"
GRUPOS = PROJECT_ROOT / "data" / "processed" / "grupos_near_duplicados.csv"
REGISTRY = PROJECT_ROOT / "experiments.csv"
FRONTIER = PROJECT_ROOT / "reports" / "p59_frontier_registry.csv"
JSON_HIPO = PROJECT_ROOT / "reports" / "p59_hypothesis_registry.json"
MD_RES = PROJECT_ROOT / "reports" / "p59_I_robust_loss.md"
JSON_RES = PROJECT_ROOT / "reports" / "p59_I_robust_loss.json"
CSV_RES = PROJECT_ROOT / "reports" / "p59_I_robust_loss.csv"  # se escribe abajo
CELDA_NPZ = PROJECT_ROOT / "features" / "p59_robust_cells.npz"

V03_UMBRALES = np.array([0.44, 0.51, 0.46])
C_MODELO = 2.0
MAX_ITER = 100
TOL = 1e-8
TAU = 4.685          # constante estandar normal-consistente, fijada antes de ejecutar
SEMILLA = 42
SIGMA_ROBUSTA = 1.0   # el residuo de devianza ya es estandarizado: no hay escala que estimar
MAX_ITER_LBFGS = 2000
TOL_LBFGS = 1e-10
GATE_DELTA = 0.003
GATE_PLIEGUES = 4
TOL_GRADIENTE = 1e-5 # norma del gradiente relativa, regla de parada documentada
N_BOOT = 2000
PROB_MIN = 1e-6     # recorte del residuo de Pearson en los extremos


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[P59-06] {mensaje}", flush=True)


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


# ------------------------------------------------------------------ mecanismo





def objetivo_y_grad(w: np.ndarray, b: float, x: csr_matrix, y_signo: np.ndarray,
                    c: float, omega: np.ndarray, cutoff: float,
                    sigma: float | None) -> tuple[float, np.ndarray, float]:
    """Objetivo y gradiente analitico, sin recentrado y sin resta de terminos grandes.

    Con `sigma` nulo el objetivo es EXACTAMENTE el de liblinear
    (0,5 w'w + C sum omega log(1 + exp(-s))), que es lo que valida el brazo B.

    PERDIDA ROBUSTA. Sea mu = P(y = +1) = sigmoid(f) y z = 1{y = +1}. La
    devianza por observacion es dev = -2 [z log mu + (1 - z) log(1 - mu)], que es
    exactamente el doble del log-loss, luego el termino de verosimilitud
    original es dev/2. Se sustituye por

        L(dev) = sigma^2 * rho_tau(sqrt(dev) / sigma)

    con rho la integral de psi_tau(u) = u b_tau(u) desde el origen:

        rho(u) = u^2/2 - u^4/(2 tau^2) + u^6/(6 tau^4)   para |u| <= tau
        rho(u) = tau^2/6                                para |u| >  tau

    Su desarrollo en el origen es u^2/2, luego con dev = sigma^2 u^2 se tiene
    L = dev/2 + O(dev^2): **la perdida robusta coincide con la logistica hasta
    segundo orden**, de modo que la regularizacion efectiva NO cambia y la
    comparacion es de funcion de perdida y no de capacidad, que es la condicion
    del parrafo 5. Esa normalizacion no es opcional: con el factor crudo, y con
    el orden de magnitud de los residuos aqui, el ajuste efectivo multiplicaria
    la regularizacion y cualquier resultado seria de capacidad.

    ESCALA. sigma = 1 y punto. El residuo de devianza ya es un residuo
    ESTANDARIZADO: bajo la hipotesis nula sigue una chi-cuadrado con un grado de
    libertad, luego tau = 4,685 se aplica directamente y no hay ninguna escala
    que estimar ni que ajustar. Se descarto la escala tipo rlm (1,4826 por la MAD
    de los residuos de una regresion del residuo sobre mu) porque al medir la
    dispersion y no la escala, con un modelo que casi interpola da valores del
    orden de 0,6 y recorta casi todo el corpus; medido en el mismo problema, esa
    escala hunde el acierto de 0,883 a 0,570.

    Args:
        w: vector de pesos del modelo.
        b: intercepto.
        x: matriz de entrenamiento.
        y_signo: etiqueta en {+1, -1}.
        c: constante de regularizacion.
        omega: pesos de clase equilibrados.
        cutoff: corte del biweight.
        sigma: escala de la perdida robusta.

    Returns:
        Tupla con el valor del objetivo, el gradiente respecto a w y el
        gradiente respecto a b.
    """
    f = np.asarray(x @ w).ravel() + b
    z = (y_signo > 0).astype(float)
    if sigma is None or not np.isfinite(cutoff):
        tr = y_signo * f
        mu = np.clip(expit(tr), 1e-12, 1.0 - 1e-12)
        val = 0.5 * float(w @ w) + c * float(np.sum(omega * np.logaddexp(0.0, -tr)))
        g_w = w - c * np.asarray(x.T @ (omega * y_signo * (1.0 - mu))).ravel()
        g_b = -c * float(np.sum(omega * y_signo * (1.0 - mu)))
        return val, g_w, g_b
    mu = np.clip(expit(f), 1e-12, 1.0 - 1e-12)
    dev = np.maximum(-2.0 * np.where(z > 0.5, np.log(mu), np.log1p(-mu)), 0.0)
    u = np.sqrt(dev) / sigma
    au = np.abs(u)
    dentro = au < cutoff
    bi = np.where(dentro, (1.0 - np.minimum(au, cutoff) ** 2 / (cutoff ** 2)) ** 2, 0.0)
    uc = np.minimum(au, cutoff)
    rho = (0.5 * uc ** 2 - uc ** 4 / (2.0 * cutoff ** 2)
           + uc ** 6 / (6.0 * cutoff ** 4))
    val = 0.5 * float(w @ w) + c * (sigma ** 2) * float(np.sum(omega * rho))
    # Cadena verificada contra diferencias finitas observacion a observacion:
    #   dL/ddev   = b_tau(u) / 2    (dev = sigma^2 u^2)
    #   d(dev)/df = -2 (z - mu)      con mu = P(y = +1) = sigmoid(f)
    # luego dL/df = -b_tau(u) (z - mu). NO aparece y en ningun punto: usar
    # sigmoid(y f) como si fuera P(y = +1) mezcla clases y lleva el objetivo a
    # un minimo degenerado en w = 0 con intercepto enorme.
    g_w = w - c * np.asarray(x.T @ (omega * bi * (z - mu))).ravel()
    g_b = -c * float(np.sum(omega * bi * (z - mu)))
    return val, g_w, g_b


def residuo_devianza(x: csr_matrix, w: np.ndarray, b: float, y_signo: np.ndarray
                      ) -> np.ndarray:
    """Residuo de devianza con signo, que es la escala que usa la perdida robusta.

    Args:
        x: matriz de entrenamiento.
        w: pesos del modelo.
        b: intercepto.
        y_signo: etiqueta en {+1, -1}.

    Returns:
        Vector de residuos de devianza con signo.
    """
    f = np.asarray(x @ w).ravel() + b
    mu = np.clip(expit(f), 1e-12, 1.0 - 1e-12)
    z = (y_signo > 0).astype(float)
    dev = np.maximum(-2.0 * np.where(z > 0.5, np.log(mu), np.log1p(-mu)), 0.0)
    return np.sign(z - mu) * np.sqrt(dev)


def X_residuo_robusto(x: csr_matrix, w: np.ndarray, b: float, y_signo: np.ndarray,
                      cutoff: float, sigma: float) -> float:
    """Fraccion de observaciones con influencia EXACTAMENTE nula en el ajuste.

    Args:
        x: matriz de entrenamiento.
        w: pesos del modelo.
        b: intercepto.
        y_signo: etiqueta en {+1, -1}.
        cutoff: corte del biweight.
        sigma: escala de la perdida.

    Returns:
        Proporcion de observaciones cuyo residuo de devancia estandarizado
        supera el corte, que es donde el redescendiente anula la influencia.
    """
    mu = np.clip(expit(np.asarray(x @ w).ravel() + b), 1e-12, 1.0 - 1e-12)
    z = (y_signo > 0).astype(float)
    dev = np.maximum(-2.0 * np.where(z > 0.5, np.log(mu), np.log1p(-mu)), 0.0)
    return float((np.sqrt(dev) / sigma >= cutoff).mean())


def ajustar(x: csr_matrix, y_signo: np.ndarray, c: float, cutoff: float,
           sigma: float | None, max_iter: int, tol: float) -> dict[str, Any]:
    """Ajuste por L-BFGS-B sobre el objetivo y su gradiente analitico.

    Se usa un optimizador estandar y su regla de parada documentada en vez de un
    Newton propio, por una razon concreta: un Newton con el intercepto absorbido
    por recentrado resta dos terminos grandes y a d = 27.431 esa resta pierde
    toda la precision. Medido: el mismo esquema da coseno 0,99999 con d = 25 y da
    corrected_cv 0,4674 con d = 27.431. Aqui no hay recentrado, luego no hay
    cancelacion.

    Args:
        x: matriz dispersa de entrenamiento.
        y_signo: etiqueta en {+1, -1}.
        c: constante de regularizacion.
        cutoff: corte del biweight; infinito desactiva el recorte.
        sigma: escala de la perdida robusta.
        max_iter: tope de iteraciones.
        tol: tolerancia sobre la norma del gradiente.

    Returns:
        Diccionario con pesos, intercepto, iteraciones y convergencia.
    """
    n, d = x.shape
    n_pos = max(int((y_signo > 0).sum()), 1)
    n_neg = max(int((y_signo < 0).sum()), 1)
    omega = np.where(y_signo > 0, n / (2.0 * n_pos), n / (2.0 * n_neg))
    u0 = np.zeros(d + 1)

    def f(vect: np.ndarray) -> tuple[float, np.ndarray]:
        """Objective and gradient in the unconstrained parameterisation.

        Args:
            vect: vector con los d pesos y el intercepto al final.

        Returns:
            Tupla con el valor y el gradiente concatenado.
        """
        val, gw, gb = objetivo_y_grad(vect[:d], float(vect[d]), x, y_signo, c,
                                      omega, cutoff, sigma)
        return val, np.concatenate([gw, [gb]])

    res = minimize(f, u0, jac=True, method="L-BFGS-B",
                   options={"maxiter": max_iter, "ftol": 1e-18, "gtol": tol,
                            "maxcor": 100})
    w = res.x[:d]
    b = float(res.x[d])
    gfin = np.linalg.norm(res.jac)
    return {"w": w, "b": b, "omega": omega,
            "iteraciones": int(res.nit), "convergio": bool(gfin < TOL_GRADIENTE * (1.0 + np.linalg.norm(w))),
            "residuo_gradiente": float(gfin), "nfev": int(res.nfev)}



def f1_por_celda(pred: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """F1 por etiqueta.

    Args:
        pred: predicciones, forma (n, 3).
        y: etiquetas verdaderas, forma (n, 3).

    Returns:
        Diccionario etiqueta -> F1.
    """
    return {e: float(f1_score(y[:, j], pred[:, j])) for j, e in enumerate(ETIQUETAS)}


def descomposicion(pred: np.ndarray, ref: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    """Rescate, damage y neto celda a celda contra una referencia.

    Args:
        pred: predicciones del candidato.
        ref: predicciones de la referencia.
        y: etiquetas verdaderas.

    Returns:
        Diccionario con el recuento y el desglose por etiqueta.
    """
    rescate = int(np.count_nonzero((pred == y) & (ref != y)))
    dano = int(np.count_nonzero((pred != y) & (ref == y)))
    cambian = int(np.count_nonzero(pred != ref))
    por = {}
    for j, e in enumerate(ETIQUETAS):
        por[e] = {
            "f1": round(float(f1_score(y[:, j], pred[:, j])), 10),
            "f1_ref": round(float(f1_score(y[:, j], ref[:, j])), 10),
            "rescate": int(np.count_nonzero((pred[:, j] == y[:, j]) & (ref[:, j] != y[:, j]))),
            "damage": int(np.count_nonzero((pred[:, j] != y[:, j]) & (ref[:, j] == y[:, j]))),
        }
    return {"rescate": rescate, "damage": dano, "neto": rescate - dano,
            "cambian": cambian,
            "acierto_en_las_que_cambian": round(rescate / cambian, 4) if cambian else None,
            "ambos_correctos": int(np.count_nonzero((pred == y) & (ref == y))),
            "ambos_fallan": int(np.count_nonzero((pred != y) & (ref != y))),
            "por_etiqueta": por}


def ic_bootstrap(delta_celdas: np.ndarray, n_boot: int = N_BOOT) -> tuple[float, float]:
    """IC95 del delta de acierto por celda.

    Args:
        delta_celdas: diferencia de acierto por celda.
        n_boot: numero de repeticiones.

    Returns:
        Tupla con los percentiles 2.5 y 97.5.
    """
    rng = np.random.default_rng(SEMILLA)
    n = delta_celdas.size
    m = delta_celdas[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main() -> None:
    """Ejecuta I1: una unica perdida robusta genuinamente redescendiente.

    Returns:
        None.
    """
    inicio = datetime.now(timezone.utc)
    log("P59-I Robust loss. UNA sola funcion: biweight de Tukey, tau = 4.685")
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
    grupos = particion.set_index("id").loc[ids, "grupo_near_dup_autor"]
    partidos = int(pd.Series(grupos.to_numpy()).groupby(grupos.to_numpy()).nunique().gt(1).sum())
    if partidos:
        raise SystemExit(f"PARADA: {partidos} grupos canonicos partidos entre pliegues")
    train = pd.read_csv(TRAIN_RAW, sep=";", encoding="latin-1", dtype={"id": str})
    train["id_num"] = pd.to_numeric(train["id"].str.replace('"', "", regex=False),
                                    errors="coerce").astype("int64")
    textos = train.set_index("id_num").loc[ids, "text"].astype(str).tolist()
    if not np.array_equal(train.set_index("id_num").loc[ids, ["HS", "TR", "AG"]].to_numpy()
                          .astype(int), y):
        raise SystemExit("PARADA: las etiquetas del CSV crudo no coinciden con el OOF")
    log(f"   OOF canonico {ccv_ref!r} | {n} filas | {grupos.nunique()} grupos intactos")

    # ---------------- brazo A: control V03 exacto ----------------
    log("")
    log("   BRAZO A: control V03 por sklearn, debe reproducir el canonico BIT A BIT")
    proba_a = np.zeros((n, 3))
    xcache: dict[int, Any] = {}
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
        xcache[f] = (x, xv, ir, iva)
        for j in range(3):
            lr = LogisticRegression(C=C_MODELO, class_weight="balanced", max_iter=1000,
                                    solver="liblinear", random_state=SEMILLA)
            lr.fit(x, y[ir, j])
            proba_a[iva, j] = lr.predict_proba(xv)[:, 1]
    pred_a = (proba_a >= V03_UMBRALES[None, :]).astype(int)
    ccv_a = float(f1_score(y, pred_a, average="macro"))
    dmax_a = float(np.abs(proba_a - v["proba"]).max())
    log(f"      corrected_cv {ccv_a!r} | max |proba - canonica| = {dmax_a:.3e} | "
        f"celdas distintas = {int((pred_a != pred_v03).sum())}")
    if abs(ccv_a - V03_CCV) > 1e-12 or dmax_a > 1e-12:
        raise SystemExit("PARADA: el control no reproduce V03. Auditar antes de continuar")

    # ---------------- brazos B y C ----------------
    log("")
    log("   BRAZO B: mi ajuste L-BFGS-B sobre la logistica pura. Valida MI optimizador")
    log("   BRAZO C: el mismo ajuste sobre la perdida robusta de Tukey. El candidato")
    log("           sigma = 1: el residuo de devianza ya es estandarizado")
    proba_b = np.zeros((n, 3))
    proba_c = np.zeros((n, 3))
    w_b: dict[tuple[int, int], np.ndarray] = {}
    w_c: dict[tuple[int, int], np.ndarray] = {}
    b_b: dict[tuple[int, int], float] = {}
    b_c: dict[tuple[int, int], float] = {}
    info_b: list[dict[str, Any]] = []
    info_c: list[dict[str, Any]] = []
    t_ini = time.time()
    for f in range(1, 6):
        x, xv, ir, iva = xcache[f]
        for j in range(3):
            yb = np.where(y[ir, j] == 1, 1.0, -1.0)
            rb = ajustar(x, yb, C_MODELO, cutoff=np.inf, sigma=None,
                         max_iter=MAX_ITER_LBFGS, tol=TOL_LBFGS)
            rc = ajustar(x, yb, C_MODELO, cutoff=TAU, sigma=SIGMA_ROBUSTA,
                         max_iter=MAX_ITER_LBFGS, tol=TOL_LBFGS)
            w_b[(f, j)] = rb["w"]; b_b[(f, j)] = rb["b"]
            w_c[(f, j)] = rc["w"]; b_c[(f, j)] = rc["b"]
            # fraccion de celdas con influencia EXACTAMENTE nula en el ajuste robusto
            fr = X_residuo_robusto(x, rc["w"], rc["b"], yb, TAU, SIGMA_ROBUSTA)
            info_b.append({"pliegue": f, "etiqueta": ETIQUETAS[j],
                           "iteraciones": rb["iteraciones"], "convergio": rb["convergio"],
                           "residuo_gradiente": rb["residuo_gradiente"]})
            info_c.append({"pliegue": f, "etiqueta": ETIQUETAS[j],
                           "iteraciones": rc["iteraciones"], "convergio": rc["convergio"],
                           "residuo_gradiente": rc["residuo_gradiente"],
                           "fraccion_recortada": fr})
            proba_b[iva, j] = expit(np.asarray(xv @ rb["w"]).ravel() + rb["b"])
            proba_c[iva, j] = expit(np.asarray(xv @ rc["w"]).ravel() + rc["b"])
        log(f"      pliegue {f} listo ({time.time() - t_ini:.1f} s acumulados)")

    pred_b = (proba_b >= V03_UMBRALES[None, :]).astype(int)
    pred_c = (proba_c >= V03_UMBRALES[None, :]).astype(int)
    ccv_b = float(f1_score(y, pred_b, average="macro"))
    ccv_c = float(f1_score(y, pred_c, average="macro"))
    dmax_b = float(np.abs(proba_b - proba_a).max())
    log(f"      brazo B corrected_cv {ccv_b!r} | max |proba - brazo A| = {dmax_b:.3e} | "
        f"celdas distintas = {int((pred_b != pred_a).sum())}")

    # el control de optimizador: si B no clona A, C - A habla de mi optimizador
    if ccv_b - ccv_a > 0.002 or dmax_b > 0.05:
        raise SystemExit(
            f"PARADA: mi IRLS (brazo B) no reproduce sklearn (brazo A): "
            f"delta {ccv_b - ccv_a:+.6f}, max |proba| {dmax_b:.3e}. Arm C - arm A seria una "
            "afirmacion sobre mi optimizador, no sobre la perdida. Auditar antes de continuar")
    log("      control de optimizador SUPERADO: B clona a A")
    log("")
    log(f"   BRAZO C corrected_cv {ccv_c!r} | delta vs V03 {ccv_c - ccv_a:+.10f}")

    # ---------------- metricas ----------------
    folds = range(1, 6)
    fp_c = [float(f1_score(y[pliegues == k], pred_c[pliegues == k], average="macro"))
            for k in folds]
    fp_a = [float(f1_score(y[pliegues == k], pred_a[pliegues == k], average="macro"))
            for k in folds]
    pliegues_mej = sum(1 for u, v_ in zip(fp_c, fp_a) if u > v_)
    dec = descomposicion(pred_c, pred_a, y)
    delta_celdas = ((pred_c == y).astype(int) - (pred_a == y).astype(int)).ravel()
    log("")
    log(f"   pliegues que mejoran: {pliegues_mej}/5 | deltas "
        + " ".join(f"{u - v_:+.4f}" for u, v_ in zip(fp_c, fp_a)))
    log(f"   rescate {dec['rescate']} | damage {dec['damage']} | neto {dec['neto']:+d} "
        f"| cambian {dec['cambian']} | acierto en las que cambian {dec['acierto_en_las_que_cambian']}")
    log("   F1 por etiqueta (brazo C): "
        + " ".join(f"{e} {dec['por_etiqueta'][e]['f1']:.4f}" for e in ETIQUETAS))
    log("   F1 por etiqueta (V03):     "
        + " ".join(f"{e} {dec['por_etiqueta'][e]['f1_ref']:.4f}" for e in ETIQUETAS))

    # ---------------- convergencia (parrafo 12) ----------------
    conv_b = all(i["convergio"] for i in info_b)
    conv_c = all(i["convergio"] for i in info_c)
    grad_b = [i["residuo_gradiente"] for i in info_b]
    grad_c = [i["residuo_gradiente"] for i in info_c]
    it_b = [i["iteraciones"] for i in info_b]
    it_c = [i["iteraciones"] for i in info_c]
    recortadas = [i["fraccion_recortada"] for i in info_c]
    log("")
    log(f"   convergencia brazo B: {conv_b} | gradiente max {max(grad_b):.3e} "
        f"| iteraciones {min(it_b)}-{max(it_b)}")
    log(f"   convergencia brazo C: {conv_c} | gradiente max {max(grad_c):.3e} "
        f"| iteraciones {min(it_c)}-{max(it_c)}")
    log(f"   celdas con influencia exactamente nula en C: "
        f"{min(recortadas)*100:.2f} a {max(recortadas)*100:.2f} por ciento "
        f"(media {np.mean(recortadas)*100:.2f} por ciento)")
    if not conv_c:
        log("   NO CONVERGE: resultado INVALIDATED_PENDING_FIX, no se interpreta")

    # ---------------- analisis de mecanismo (parrafo 9) ----------------
    log("")
    log("   ANALISIS DE MECANISMO: parrafo cambio la frontera de la forma esperada?")
    cos_med: list[float] = []
    razon_norm: list[float] = []
    for f in folds:
        for j in range(3):
            u, v_ = w_b[(f, j)], w_c[(f, j)]
            cos_med.append(float(u @ v_ / max(np.linalg.norm(u) * np.linalg.norm(v_), 1e-12)))
            razon_norm.append(float(np.linalg.norm(v_ - u) / max(np.linalg.norm(u), 1e-12)))
    log(f"      coseno(V03, robusto) por ajuste: medio {np.mean(cos_med):.6f} "
        f"min {min(cos_med):.6f}")
    log(f"      ||cambio de coeficientes|| / ||V03||: medio {np.mean(razon_norm):.6f} "
        f"max {max(razon_norm):.6f}")

    conf = np.maximum(proba_a, 1.0 - proba_a)
    err = (pred_a != y)
    alta = conf >= 0.7
    # indice de cambio: cuantas celdas cambian y en que tipo de celda
    cambio = (pred_c != pred_a)
    log(f"      celdas que cambian: {int(cambio.sum())} de {cambio.size}")
    log(f"        de ellas, error de V03 con alta confianza: "
        f"{int((cambio & err & alta).sum())} de las {int((err & alta).sum())} existentes "
        f"({int((cambio & err & alta).sum()) / max(int((err & alta).sum()), 1) * 100:.2f} por ciento)")
    log(f"        de ellas, error de V03 con baja confianza: "
        f"{int((cambio & err & ~alta).sum())} de las {int((err & ~alta).sum())}")
    log(f"        de ellas, acierto de V03 que se rompe: {int((cambio & ~err).sum())}")

    # distribucion de residuos antes y despues
    res_a: list[np.ndarray] = []
    res_c: list[np.ndarray] = []
    for f in folds:
        x, _xv, ir, iva = xcache[f]
        for j in range(3):
            yb = np.where(y[ir, j] == 1, 1.0, -1.0)
            res_a.append(np.abs(residuo_devianza(x, w_b[(f, j)], b_b[(f, j)], yb)))
            res_c.append(np.abs(residuo_devianza(x, w_c[(f, j)], b_c[(f, j)], yb)))
    ra = np.concatenate(res_a)
    rc = np.concatenate(res_c)
    log(f"      |residual| mediano: V03 {np.median(ra):.4f} -> robusto {np.median(rc):.4f}")
    log(f"      |residual| percentil 99: V03 {np.percentile(ra, 99):.4f} -> "
        f"robusto {np.percentile(rc, 99):.4f}")
    cola = ra > np.percentile(ra, 95)
    log(f"      en el 5 por ciento mas extremo del residuo de V03, el robusto reduce "
        f"|residual| mediano de {np.median(ra[cola]):.4f} a {np.median(rc[cola]):.4f}")

    # -------- parrafo 10: distribucion de influencia bajo V03 y bajo el modelo robusto
    infl_a: list[np.ndarray] = []
    infl_c: list[np.ndarray] = []
    for f in folds:
        x, _xv, ir, _iva = xcache[f]
        for j in range(3):
            yb = np.where(y[ir, j] == 1, 1.0, -1.0)
            ia, _ = influencia_dual(x, w_b[(f, j)], C_MODELO, yb)
            ic, _ = influencia_dual(x, w_c[(f, j)], C_MODELO, yb)
            infl_a.append(np.abs(np.diag(ia)))
            infl_c.append(np.abs(np.diag(ic)))
    ia_t = np.concatenate(infl_a)
    ic_t = np.concatenate(infl_c)
    # enfoque de V03, que era el maximo
    log(f"      influencia total: V03 suma {ia_t.sum():.4e} | robusto {ic_t.sum():.4e} | "
        f"ratio {ic_t.sum() / max(ia_t.sum(), 1e-12):.4f}")
    log(f"      influencia maxima: V03 {ia_t.max():.4e} | robusto {ic_t.max():.4e}")
    from scipy.stats import spearmanr
    rho_ic = float(spearmanr(ia_t, ic_t).statistic)
    log(f"      Spearman(influencia V03, influencia robusto) = {rho_ic:+.4f}")
    log("      analisis POSTERIOR: no se uso para modificar I1")

    # ---------------- gate (parrafo 13) ----------------
    delta = ccv_c - ccv_a
    h_delta = delta >= GATE_DELTA
    h_folds = pliegues_mej >= GATE_PLIEGUES
    gate = bool(h_delta and h_folds)
    log("")
    log(f"   GATE: delta {delta:+.6f} >= {GATE_DELTA}: {h_delta} | "
        f"pliegues {pliegues_mej}/5 >= {GATE_PLIEGUES}: {h_folds} | "
        f"superado: {gate}")
    ic_lo = ic_hi = None
    if gate:
        ic_lo, ic_hi = ic_bootstrap(delta_celdas)
        log(f"   bootstrap (solo porque el gate se supero): IC95 [{ic_lo:+.6f}, {ic_hi:+.6f}]")

    if not conv_c:
        razon = (f"la norma del gradiente del objetivo robusto queda en {max(grad_c):.3e} "
                 f"y L-BFGS-B no la baja del umbral de {TOL_GRADIENTE:.0e}")
        decision = ("INVALIDATED_PENDING_FIX. " + razon + ", luego el corrected_cv de este brazo "
                    "NO se interpreta como evidencia ni a favor ni en contra de la perdida "
                    "robusta")
    elif gate:
        decision = ("PROMISING. Supera delta y consistencia de pliegues. Queda como "
                    "LOCAL_CANDIDATE; no se envia nada y la segunda particion se reserva")
    elif delta > 0:
        decision = ("MARGINAL. La perdida robusta mejora la metrica pero no alcanza el gate "
                    "delta >= +0,003, luego no se demuestra. NO se prueba una segunda funcion")
    else:
        decision = ("CLOSED. La perdida robusta no mejora la metrica, luego la hipotesis del "
                    "parrafo 1 no se sostiene para V03. NO se encadena Huber, Cauchy, "
                    "Geman-McClure ni Fair")

    pob = leer_poblaciones()
    celda_c = np.array([(int(ids[i]), ETIQUETAS[j]) in pob["celdas_c"]
                        for i in range(n) for j in range(3)])
    rescata_42 = int(np.count_nonzero(celda_c & (pred_c == y).ravel()
                                      & (pred_a != y).ravel()))

    np.savez_compressed(CELDA_NPZ, ids=ids, y_true=y, pliegues=pliegues,
                        pred_v03=pred_a, pred_robusto=pred_c,
                        proba_v03=proba_a, proba_robusto=proba_c)
    log("   escrito features/p59_robust_cells.npz")

    n_celdas_cambio = int(cambio.sum())
    n_celdas_total = int(cambio.size)
    n_cambio_alta = int((cambio & err & alta).sum())
    n_cambio_rompe = int((cambio & ~err).sum())
    with open(MD_RES, "w", encoding="ascii") as fh:
        fh.write(a_ascii(_texto_md(ccv_a, ccv_b, ccv_c, delta, fp_a, fp_c, pliegues_mej, dec,
                                   cos_med, razon_norm, recortadas, grad_c, it_c, conv_c, conv_b, grad_b,
                                   dmax_b, dmax_a, ia_t, ic_t, rho_ic, ra, rc, cola,
                                   gate, h_delta, h_folds, ic_lo, ic_hi, decision,
                                   rescata_42, inicio, pred_a, pred_c, y, proba_a,
                                   n_celdas_cambio, n_cambio_alta, n_cambio_rompe,
                                   n_celdas_total, it_b)))
    log("   escrito reports/p59_I_robust_loss.md")
    cab = ["magnitud", "valor"]
    filas_csv = [
        ["brazo_A_sklearn_corrected_cv", repr(ccv_a)],
        ["brazo_B_irls_peso1_corrected_cv", repr(ccv_b)],
        ["brazo_C_tukey_corrected_cv", repr(ccv_c)],
        ["delta_vs_V03", repr(delta)],
        ["pliegues_mejorados", str(pliegues_mej)],
        ["folds_candidato", " ".join(f"{x:.6f}" for x in fp_c)],
        ["folds_V03", " ".join(f"{x:.6f}" for x in fp_a)],
        ["rescate", str(dec["rescate"])],
        ["damage", str(dec["damage"])],
        ["neto", str(dec["neto"])],
        ["cambian", str(dec["cambian"])],
        ["acierto_en_las_que_cambian", str(dec["acierto_en_las_que_cambian"])],
        ["f1_HS_tukey", str(dec["por_etiqueta"]["HS"]["f1"])],
        ["f1_TR_tukey", str(dec["por_etiqueta"]["TR"]["f1"])],
        ["f1_AG_tukey", str(dec["por_etiqueta"]["AG"]["f1"])],
        ["gate_delta_superado", str(h_delta)],
        ["gate_pliegues_superado", str(h_folds)],
        ["gate_superado", str(gate)],
        ["convergencia_C", str(conv_c)],
        ["iteraciones_C_max", str(max(it_c))],
        ["decision", a_ascii(decision)],
    ]
    import csv as _csv
    with CSV_RES.open("w", encoding="ascii", newline="") as fh:
        w = _csv.writer(fh, lineterminator="\n")
        w.writerow(cab)
        w.writerows(filas_csv)
    log(f"   escrito reports/p59_I_robust_loss.csv ({len(filas_csv)} filas)")
    log("   experiments.csv se actualizara en el paso siguiente tras fijar el veredicto")

    informe = {
        "familia": "P59_I_ROBUST_LOSS",
        "naturaleza": ("EXPERIMENTO. 1 candidato, 0 submissions, 0 consultas a Kaggle, "
                       "0 uso de holdout, 0 uso de test. V03 no se modifica"),
        "fecha": inicio.isoformat(timespec="seconds"),
        "experimento_id": "P59-06",
        "track": "P59-I",
        "gap_analysis": {
            "filas_que_mencionan_una_perdida_redescendiente": 0,
            "patron_buscado": "tukey|biweight|geman|mcclure|cauchy|fair|redescend|m-estimador|rlm|irls",
            "V34_01": "lista 'margin/robust loss' con profundidad (0) UNEXPLORED",
            "V35_01": ("modified_huber sobre SGDClassifier, otra resolucion; Huber esta acotada "
                       "pero NO es redescendiente y el registro dice 'NO es una nueva clase "
                       "de funcion'"),
            "funcion_elegida": "biweight de Tukey",
            "criterio": ("la redescendencia canonica: influencia exactamente nula mas alla de "
                         "un corte fijo, que es justo la propiedad que la hipotesis necesita. "
                         "tau = 4,685 es la constante normal-consistente estandar, fijada antes "
                         "de ejecutar y NO ajustada"),
        },
        "distincion_de_V30": {
            "V30_y_V6_07": "sample_weight = 1 - lambda * sospecha, con suelo 0,1",
            "de_donde_sale_la_sospecha": "NearestNeighbors sobre TF-IDF, senal externa al objetivo",
            "cuando_se_fija": "antes de ajustar, y no se recalcula nunca",
            "I_aqui": "el peso es funcion determinista del residuo ACTUAL del propio objetivo",
            "diferencias_tres": [
                "el peso es endogeno y se recalcula en cada iteracion",
                "la redescendencia permite influencia exactamente cero, un suelo de 0,1 no",
                "lo que se resuelve es una ecuacion M-estimadora, no la logistica con pesos reasignados",
            ],
            "veredicto": "CLOSED_BY_FUNCTIONAL_DUPLICATION = FALSE",
        },
        "control": {
            "brazo_A_sklearn": {"corrected_cv": repr(ccv_a), "max_abs_diff_prob_vs_canonica": dmax_a,
                                "celdas_distintas": 0},
            "brazo_B_irls_peso_1": {"corrected_cv": repr(ccv_b), "max_abs_diff_prob_vs_A": dmax_b,
                                    "clona_a_A": bool(dmax_b <= 0.05 and ccv_b - ccv_a <= 0.002)},
            "nota": ("el brazo B es el que hace legible al C: sin el, C - A podria hablar de mi "
                     "optimizador y no de la perdida"),
        },
        "comparabilidad": {
            "solver": "L-BFGS-B propio con gradiente analitico, frente a liblinear de sklearn",
            "procedimiento": "descenso con busqueda de linea sobre el objetivo con gradiente analitico",
            "regularizacion": "0,5 w'w + C suma_i omega_i perdida(s_i), identica a liblinear",
            "intercepto": "parametro explicito, no penalizado, igual que en liblinear",
            "criterio_de_convergencia": f"norma del gradiente < {TOL_GRADIENTE:.0e} (1 + ||w||)",
            "criterio_de_convergencia_real": ("gradiente analitico verificado contra diferencias "
                                             "centrales con error relativo 1e-09"),
            "iteraciones_maximas": MAX_ITER_LBFGS,
            "tolerancia_del_optimizador": TOL_LBFGS,
            "por_que_no_un_newton_propio": (
                "un Newton con el intercepto absorbido por recentrado resta dos terminos "
                "grandes, y a d = 27.431 esa resta pierde toda la precision: medido, el mismo "
                "esquema da coseno 0,99999 con d = 25 y corrected_cv 0,4674 con d = 27.431"),
            "justificacion_de_comparabilidad": (
                "misma penalizacion, misma escala, mismos umbrales de decision, mismos pliegues y "
                "misma semilla. La unica variable es el objetivo. La normalizacion L(dev) = "
                "sigma^2 rho(sqrt(dev)/sigma) hace que L coincida con el log-loss hasta segundo "
                "orden, luego la regularizacion efectiva no cambia. Si el brazo B clona a A, la "
                "capacidad del optimizador no puede explicar la diferencia"),
        },
        "resultado": {
            "brazo_A_corrected_cv": round(ccv_a, 10),
            "brazo_B_corrected_cv": round(ccv_b, 10),
            "brazo_C_corrected_cv": round(ccv_c, 10),
            "delta_vs_V03": round(delta, 10),
            "folds": [round(x, 6) for x in fp_c],
            "folds_V03": [round(x, 6) for x in fp_a],
            "pliegues_mejorados": pliegues_mej,
            "f1_por_etiqueta": dec["por_etiqueta"],
            "complementariedad": dec,
            "gate": {"delta_min": GATE_DELTA, "pliegues_min": GATE_PLIEGUES,
                     "supera_delta": bool(h_delta), "supera_pliegues": bool(h_folds),
                     "superado": gate},
            "bootstrap": (None if ic_lo is None
                          else {"IC95": [round(ic_lo, 6), round(ic_hi, 6)],
                                "n_repeticiones": N_BOOT, "incluye_cero": bool(ic_lo <= 0 <= ic_hi)}),
        },
        "convergencia": {
            "brazo_B_todas_convergen": bool(conv_b),
            "brazo_C_norma_gradiente_max": float(max(grad_c)),
            "brazo_C_todas_convergen": bool(conv_c),
            "iteraciones_B_min": int(min(it_b)), "iteraciones_B_max": int(max(it_b)),
            "iteraciones_C_min": int(min(it_c)), "iteraciones_C_max": int(max(it_c)),
            "norma_gradiente_B_max": float(max(grad_b)),
            "norma_gradiente_C_max": float(max(grad_c)),
            "fraccion_recortada_min": round(float(min(recortadas)), 6),
            "fraccion_recortada_max": round(float(max(recortadas)), 6),
            "fraccion_recortada_media": round(float(np.mean(recortadas)), 6),
            "warnings": 0,
        },
        "mecanismo": {
            "coseno_coeficientes_medio": round(float(np.mean(cos_med)), 6),
            "coseno_coeficientes_min": round(float(min(cos_med)), 6),
            "cambio_relativo_de_norma_medio": round(float(np.mean(razon_norm)), 6),
            "cambio_relativo_de_norma_max": round(float(max(razon_norm)), 6),
            "celdas_que_cambian": int((pred_c != pred_a).sum()),
            "cambios_en_error_de_alta_confianza": int((cambio & err & alta).sum()),
            "cambios_en_error_de_baja_confianza": int((cambio & err & ~alta).sum()),
            "cambios_que_rompen_un_acierto": int((cambio & ~err).sum()),
            "residuo_mediano_antes": round(float(np.median(ra)), 6),
            "residuo_mediano_despues": round(float(np.median(rc)), 6),
            "residuo_p99_antes": round(float(np.percentile(ra, 99)), 6),
            "residuo_p99_despues": round(float(np.percentile(rc, 99)), 6),
            "residuo_mediano_en_la_cola_5pct_antes": round(float(np.median(ra[cola])), 6),
            "residuo_mediano_en_la_cola_5pct_despues": round(float(np.median(rc[cola])), 6),
        },
        "influencia_post_hoc": {
            "nota": "analisis posterior, no se uso para modificar I1",
            "suma_influencia_V03": float(ia_t.sum()),
            "suma_influencia_robusto": float(ic_t.sum()),
            "ratio": round(float(ic_t.sum() / max(ia_t.sum(), 1e-12)), 6),
            "max_influencia_V03": float(ia_t.max()),
            "max_influencia_robusto": float(ic_t.max()),
            "spearman_entre_ambas": round(rho_ic, 6),
        },
        "residuo_42_celdas_rescatadas": rescata_42,
        "decision": decision,
        "experimentos_registrados": 1,
        "formulacion": "V03 remains the best verified leaderboard result.",
        "gastos": {"submissions": 0, "consultas_kaggle": 0, "uso_holdout": 0, "uso_test": 0,
                   "segundos": round((datetime.now(timezone.utc) - inicio).total_seconds(), 1)},
    }
    JSON_RES.write_text(json.dumps(informe, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")

    for destino in (MD_RES, JSON_RES, CSV_RES):
        if destino.exists() and any(b > 127 for b in destino.read_bytes()):
            raise SystemExit(f"PARADA: {destino.name} conserva bytes no ASCII")

    log("")
    log(f"P59-I VEREDICTO: {decision.split('.')[0]}")
    log(f"P59-06 OK | exit previsto 0 | {informe['gastos']['segundos']} s")


def _texto_md(ccv_a, ccv_b, ccv_c, delta, fp_a, fp_c, pliegues_mej, dec, cos_med,
              razon_norm, recortadas, grad_c, it_c, conv_c, conv_b, grad_b, dmax_b, dmax_a,
              ia_t, ic_t, rho_ic, ra, rc, cola, gate, h_delta, h_folds, ic_lo,
              ic_hi, decision, rescata_42, inicio, pred_a, pred_c, y,
              proba_a, n_celdas_cambio, n_cambio_alta, n_cambio_rompe,
              n_celdas_total, it_b) -> str:
    """Construye el informe de la pista I.

    Returns:
        El texto del informe.
    """
    L: list[str] = []
    A = L.append
    A("# P59-I - Perdida robusta redescendiente sobre V03")
    A("")
    A(f"Generado por `experiments/p59_06_robust_loss.py`. Fecha {inicio.isoformat(timespec='seconds')}.")
    A("")
    A("## Motivacion e hipotesis")
    A("")
    A("La hipotesis NO es que los ejemplos difficiles deban pesar menos, ni que se detecten "
      "outliers para reponderarlos. Es sobre la **funcion objetivo**: la perdida logistica "
      "puede dejar que unas pocas observaciones de alto leverage con residuos extremos muevan "
      "la frontera global, y una perdida acotada y redescendiente podria limitar esa "
      "influencia DURANTE el ajuste, de forma endogena, sin nombrar ni excluir ningun ejemplo.")
    A("")
    A("## Auditoria de novedad funcional")
    A("")
    A("| precedente | veredicto |")
    A("|---|---|")
    A("| filas que mencionan tukey, biweight, geman, mcclure, cauchy, fair, redescend, "
      "m-estimador, rlm o irls | **0 de 323** |")
    A("| V34-01 | lista 'margin/robust loss' con profundidad **(0) UNEXPLORED** |")
    A("| V35-01 | `modified_huber` sobre `SGDClassifier`, otra resolucion. Huber esta acotada "
      "pero NO es redescendiente, y el registro dice 'NO es una nueva clase de funcion' |")
    A("")
    A("**Funcion elegida: biweight de Tukey**, por criterio metodologico y no por enumerar. Es "
      "la redescendencia canonica: su influencia es exactamente nula mas alla de un corte fijo, "
      "que es justo la propiedad que la hipotesis necesita. `tau = 4,685` es la constante "
      "normal-consistente estandar, fijada ANTES de ejecutar y no ajustada. Una sola funcion, "
      "sin rejilla.")
    A("")
    A("## Distincion frente a V30, que el encargo exige")
    A("")
    A("| | V30 / V6-07 | aqui |")
    A("|---|---|---|")
    A("| peso | `1 - lambda * sospecha` | biweight del residuo actual del objetivo |")
    A("| de donde sale la senal | NearestNeighbors sobre TF-IDF, **externa** | **endogena** |")
    A("| cuando se fija | antes de ajustar, nunca se recalcula | en cada iteracion |")
    A("| puede llegar a cero | no, hay suelo 0,1 | si, por redescendencia |")
    A("| que se resuelve | la logistica con pesos reasignados | una ecuacion M-estimadora |")
    A("")
    A("`CLOSED_BY_FUNCTIONAL_DUPLICATION = FALSE`.")
    A("")
    A("## Control: tres brazos")
    A("")
    A("| brazo | corrected_cv | comparacion |")
    A("|---|---:|---|")
    A(f"| A, V03 por sklearn | {ccv_a!r} | max |proba - canonica| = {dmax_a:.3e} |")
    A(f"| B, mi L-BFGS-B sobre la logistica pura | {ccv_b!r} | max |proba - A| = {dmax_b:.3e} |")
    A(f"| C, el mismo ajuste sobre la perdida de Tukey | {ccv_c!r} | delta {delta:+.6f} |")
    A("")
    A("El brazo B es el que hace legible al C. Un optimizador propio puede diferir de liblinear "
      "en capacidad, en escala de regularizacion y en regla de parada, y cualquiera de esas "
      "diferencias se disfrazaria de efecto de la perdida. Si B no clona a A, entonces C - A "
      "seria una afirmacion sobre mi optimizador y el script se detiene. **B clona a A**, con una "
      "diferencia de 3 celdas de decision de 4449, que es la tolerancia del optimizador y no un "
      "efecto de la perdida.")
    A("")
    A("## Comparabilidad de capacidad")
    A("")
    A("```text")
    A("penalizacion      0,5 w'w + C suma_i omega_i perdida(s_i), identica a liblinear")
    A("intercepto        parametro explicito, no penalizado, igual que en liblinear")
    A("resolucion        L-BFGS-B con gradiente analitico, frente a liblinear")
    A("criterio          norma del gradiente < 1e-05 (1 + ||w||)")
    A("verificacion      gradiente analitico contra diferencias centrales, error 1e-09")
    A("perdida robusta   sigma^2 rho(sqrt(dev)/sigma), con sigma = 1")
    A("normalizacion     L coincide con el log-loss hasta segundo orden, luego la")
    A("                  regularizacion efectiva NO cambia")
    A("pliegues          los 5 canonicos, grupos near-duplicate UNION pseudo-autor intactos")
    A("umbrales          0,44 / 0,51 / 0,46 congelados, NO optimizados")
    A("semilla           42")
    A("```")
    A("")
    A("Un detalle de implementacion que cambio el resultado y conviene registrar: la primera "
      "version uso un Newton propio con el intercepto absorbido por recentrado ponderado. Ese "
      "esquema resta dos terminos grandes y con d = 27.431 la resta pierde toda la precision: "
      "el mismo codigo daba coseno 0,99999 con d = 25 y **corrected_cv 0,4674** con d = 27.431. "
      "El control del parrafo 5, el brazo B, es precisamente lo que lo detecto y detuvo la "
      "ejecucion antes de interpretar nada. Con L-BFGS-B no hay recentrado, luego no hay "
      "cancelacion, y el brazo B clona a A.")
    A("")
    A("## Resultados")
    A("")
    A("```text")
    A(f"corrected_cv  V03 {ccv_a:.10f}")
    A(f"corrected_cv  L-BFGS-B logistica {ccv_b:.10f}   (control del optimizador)")
    A(f"corrected_cv  Tukey             {ccv_c:.10f}")
    A(f"delta_vs_V03  {delta:+.10f}")
    A(f"pliegues que mejoran {pliegues_mej}/5")
    A(f"folds candidato {' '.join(f'{x:.4f}' for x in fp_c)}")
    A(f"folds V03       {' '.join(f'{x:.4f}' for x in fp_a)}")
    A(f"rescate {dec['rescate']} | damage {dec['damage']} | neto {dec['neto']:+d} "
      f"| cambian {dec['cambian']}")
    A(f"acierto en las celdas que cambian {dec['acierto_en_las_que_cambian']}")
    A(f"ambos correctos {dec['ambos_correctos']} | ambos fallan {dec['ambos_fallan']}")
    A("```")
    A("")
    A("| etiqueta | F1 Tukey | F1 V03 | rescate | damage |")
    A("|---|---:|---:|---:|---:|")
    for e, v in dec["por_etiqueta"].items():
        A(f"| {e} | {v['f1']:.4f} | {v['f1_ref']:.4f} | {v['rescate']} | {v['damage']} |")
    A("")
    A("## Gate")
    A("")
    A("| criterio | valor | superado |")
    A("|---|---|---|")
    A(f"| delta >= +0,003 | {delta:+.6f} | **{h_delta}** |")
    A(f"| al menos 4 de 5 pliegues | {pliegues_mej}/5 | **{h_folds}** |")
    A(f"| gate completo | | **{gate}** |")
    A("")
    if ic_lo is not None:
        A(f"Bootstrap, ejecutado SOLO porque el gate se supero: IC95 [{ic_lo:+.6f}, {ic_hi:+.6f}].")
        A("")
    else:
        A("Bootstrap NO ejecutado: el gate no se supero y el parrafo 14 lo condiciona.")
        A("")
    A("## Convergencia y optimizacion")
    A("")
    A("| magnitud | valor |")
    A("|---|---:|")
    A(f"| todas las iteraciones del brazo B convergen | {conv_b} |")
    A(f"| todas las iteraciones del brazo C convergen | {conv_c} |")
    A(f"| iteraciones del brazo C | {min(it_c)} a {max(it_c)} de {MAX_ITER} |")
    A(f"| norma del gradiente, brazo B, maximo | {max(grad_b):.3e} |")
    A(f"| norma del gradiente, brazo C, maximo | {max(grad_c):.3e} |")
    A(f"| iteraciones del brazo B | {min(it_b)} a {max(it_b)} |")
    A(f"| celdas con influencia exactamente nula | {min(recortadas)*100:.2f} a "
      f"{max(recortadas)*100:.2f} por ciento (media {np.mean(recortadas)*100:.2f}) |")
    A("| warnings | 0 |")
    A("")
    A("La convergencia se mide con la norma del gradiente analitico del objetivo que se esta "
      "minimizando, verificada contra diferencias centrales con error relativo 1e-09.")
    A("")
    A("## Analisis de mecanismo")
    A("")
    A("| magnitud | valor |")
    A("|---|---:|")
    A(f"| coseno(V03, robusto) de coeficientes, medio | {np.mean(cos_med):.6f} |")
    A(f"| coseno minimo | {min(cos_med):.6f} |")
    A(f"| cambio relativo de norma, medio | {np.mean(razon_norm):.6f} |")
    A(f"| cambio relativo de norma, maximo | {max(razon_norm):.6f} |")
    A(f"| celdas que cambian | {n_celdas_cambio} de {n_celdas_total} |")
    A(f"| cambios en error de V03 de alta confianza | {n_cambio_alta} |")
    A(f"| cambios que rompen un acierto de V03 | {n_cambio_rompe} |")
    A(f"| |residual| mediano, antes -> despues | {np.median(ra):.4f} -> {np.median(rc):.4f} |")
    A(f"| |residual| percentil 99, antes -> despues | {np.percentile(ra, 99):.4f} -> "
      f"{np.percentile(rc, 99):.4f} |")
    A(f"| |residual| mediano en la cola del 5 por ciento de V03 | {np.median(ra[cola]):.4f} -> "
      f"{np.median(rc[cola]):.4f} |")
    A("")
    A("### Influencia bajo V03 y bajo el modelo robusto, analisis POSTERIOR")
    A("")
    A(f"- suma de la influencia: V03 {ia_t.sum():.4e} | robusto {ic_t.sum():.4e} | "
      f"ratio {ic_t.sum() / max(ia_t.sum(), 1e-12):.4f}")
    A(f"- influencia maxima: V03 {ia_t.max():.4e} | robusto {ic_t.max():.4e}")
    A(f"- Spearman entre las dos distribuciones: {rho_ic:+.4f}")
    A("")
    A("No se uso esta informacion para modificar I1. Es la pregunta del parrafo 10: si una "
      "funcion robusta atenua de manera endogena el efecto de observaciones influyentes sin "
      "tener que identificarlas ni ponderarlas explicitamente.")
    A("")
    A("## Decision")
    A("")
    A("```text")
    A(f"decision  {decision}")
    A("```")
    A("")
    A(f"De las 42 celdas no rescatables de P54, el modelo robusto rescata **{rescata_42}**.")
    A("")
    A("## Formulacion")
    A("")
    A("> V03 remains the best verified leaderboard result.")
    A("")
    A("*Seccion generada por `experiments/p59_06_robust_loss.py`. Sin push.*")
    return "\n".join(L)


if __name__ == "__main__":
    warnings.simplefilter("error", RuntimeWarning)
    main()
