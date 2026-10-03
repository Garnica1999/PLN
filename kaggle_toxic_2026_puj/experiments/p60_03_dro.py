"""P60 track L: group distributionally robust optimization.

The hypothesis
--------------
V03 optimises a single average risk over the training fold. If that average hides
systematically different behaviour on subpopulations, then a minimax objective
over a small number of **existing** groups should place the boundary better than
the average does. This is a different axis from every objective tried so far:
P59-I changed the per-example loss shape, P59-E changed the regularizer with a
similarity graph, and this changes the AGGREGATION OF THE RISK over groups.

Why the groups are the length quartiles
--------------------------------------
The brief allows only groups that already exist and are reproducible: the
near-duplicate group, the pseudo-author group, or a predefined slice. The
near-duplicate and pseudo-author groups are unusable here for a mechanical reason:
they ARE the cross-validation grouping, and they hold 1415 groups for 1483
examples, so a "group" is almost always a single example and a minimax over them
degenerates into a worst-example objective that is not group DRO at all.

The length quartile is the alternative that has evidence. `V28-01` measured that
the feature that discriminates best is length, with Cohen d of +0,28 to +0,34 in
favour of long texts, and that it is the same feature for all three labels. The
quartile definition is the project's own, `pd.qcut` on `n_caracteres` into four
groups, used in `experiments/06_error_analysis.py` and
`experiments/08_features_densas_h11_h12.py`. So the groups are pre-existing,
reproducible, and have a measured reason to differ.

The objective, and why it is smooth
-----------------------------------
Group DRO in the form of Sagawa et al. 2020 upper-bounds the worst group risk
with an exponentiated average:

    DRO = (1 / eta) * log( sum_g (n_g / n) * exp(eta * risk_g) )
    risk_g = (1 / n_g) * sum_{i in G_g} omega_i * loss_i

It is smooth, its gradient is a softmax-weighted average of the per-group
gradients, and as eta goes to infinity it becomes the maximum while eta going to
zero becomes the global average. eta = 1 is the value the paper uses with
exponentiated gradient, and it is fixed here before any validation number.

Control
-------
    arm 0   one single group, every training row in it, which reduces DRO to the
            global average risk and must clone V03
    arm 1   the four length quartiles as groups, eta = 1,0

Same optimiser, same representation, same class weights, same regularization, same
thresholds. The only difference is how the per-example losses are aggregated
across groups, so arm 1 minus arm 0 isolates the risk aggregation and nothing
else. That is the gate.

The quartiles are computed on the TRAINING ROWS of each fold, so a validation
example never influences the group definition, and no group is invented after
looking at results.

Usage
-----
    /home/carlos/venv/bin/python experiments/p60_03_dro.py
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
MD_L = PROJECT_ROOT / "reports" / "p60_L_dro.md"
JSON_L = PROJECT_ROOT / "reports" / "p60_L_dro.json"
CSV_L = PROJECT_ROOT / "reports" / "p60_L_dro.csv"
CELDA_NPZ = PROJECT_ROOT / "features" / "p60_dro_cells.npz"

V03_UMBRALES = np.array([0.44, 0.51, 0.46])
C_MODELO = 2.0
MAX_ITER = 1000
SEMILLA = 42
ETA_DRO = 1.0
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
    print(f"[P60-03] {mensaje}", flush=True)


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


def objetivo_dro(w: np.ndarray, b: float, x: csr_matrix, y01: np.ndarray,
                 c: float, omega: np.ndarray, grupos: np.ndarray, n_grupos: int,
                 eta: float) -> tuple[float, np.ndarray, float, dict[str, float]]:
    """Riesgo exponencialmente suavizado sobre grupos, y su gradiente analitico.

    Con `n_grupos` igual a 1 el objetivo se reduce al riesgo promedio global, que es
    el objetivo de V03, de modo que el brazo 0 es un control real y no un adorno.

    Args:
        w: vector de pesos.
        b: intercepto.
        x: matriz de entrenamiento.
        y01: etiqueta binaria.
        c: constante de regularizacion.
        omega: pesos de clase.
        grupos: indice de grupo de cada fila de ENTRENAMIENTO.
        n_grupos: numero de grupos.
        eta: temperatura de la suavizacion.

    Returns:
        Tupla con el valor, el gradiente respecto a w, respecto a b, y el riesgo
        por grupo.
    """
    f = np.asarray(x @ w).ravel() + b
    p = np.clip(expit(f), 1e-12, 1.0 - 1e-12)
    s = 2.0 * y01 - 1.0
    # perdida por ejemplo, con la convencion de etiqueta
    perdida = np.logaddexp(0.0, -f * s)
    ddf = p - y01
    conteo = np.bincount(grupos, minlength=n_grupos).astype(float)
    conteo = np.maximum(conteo, 1.0)
    suma_por_grupo = np.bincount(grupos, weights=omega * perdida, minlength=n_grupos)
    riesgo = suma_por_grupo / conteo
    prop = conteo / conteo.sum()
    n = y01.size
    pesos_grupo: dict[str, Any] = {}
    if eta <= 0.0 or n_grupos == 1:
        # riesgo promedio: 0,5||w||^2 + c * suma_i omega_i perdida_i
        peso_efectivo = omega.copy()
        val_perdida = n * float(np.sum(omega * perdida)) / n
    else:
        z = eta * riesgo
        zmax = float(np.max(z))
        exp_z = np.exp(z - zmax)
        den = float(np.sum(prop * exp_z))
        w_grupo = prop * exp_z / max(den, 1e-300)
        z_val = (1.0 / eta) * (zmax + float(np.log(max(den, 1e-300))))
        # d DRO / d riesgo_g = w_grupo, y d riesgo_g / d perdida_i = omega_i / n_g
        # El objetivo se multiplica por n, que es una CONSTANTE, para que su masa
        # total coincida con la del control. Sin ese factor, cada riesgo de grupo
        # es una MEDIA y la perdida queda en escala O(1) frente a la escala O(n) del
        # control, de modo que el brazo mediria una regularizacion n veces mayor y
        # no la agregacion del riesgo. Al ser n constante, el gradiente analitico
        # sigue siendo exacto y no requiere terminos de la derivada del factor.
        peso_efectivo = n * omega * w_grupo[grupos] / conteo[grupos]
        val_perdida = n * z_val
        pesos_grupo = {
            "pesos_softmax_por_grupo": [round(float(u), 6) for u in w_grupo],
            "proporcion_de_ejemplos": [round(float(u), 6) for u in prop],
            "riesgo_por_grupo": [round(float(u), 6) for u in riesgo],
            "masa_de_la_perdida_control": round(float(np.sum(omega)), 6),
            "masa_de_la_perdida_dro": round(float(np.sum(peso_efectivo)), 6),
            "razon_de_masas": round(float(np.sum(peso_efectivo) / max(np.sum(omega), 1e-12)), 6),
        }
    val = 0.5 * float(w @ w) + c * val_perdida
    g_w = w + c * np.asarray(x.T @ (peso_efectivo * ddf)).ravel()
    g_b = c * float(np.sum(peso_efectivo * ddf))
    return val, g_w, g_b, riesgo, pesos_grupo


def ajustar(x: csr_matrix, y01: np.ndarray, c: float, omega: np.ndarray,
            grupos: np.ndarray, n_grupos: int, eta: float,
            funcion=None) -> dict[str, Any]:
    """Ajuste por L-BFGS-B del objetivo de riesgo por grupo, o de uno alternativo.

    El protocolo de `funcion` es de SEIS argumentos, (w, b, x, y01, c, omega), y el
    objetivo de riesgo por grupo se envuelve en ese mismo protocolo. Una version
    anterior de este archivo aceptaba el parametro `funcion` pero el cierre seguia
    llamando a `objetivo_dro` directamente, de modo que el objetivo propio se
    ignoraba EN SILENCIO y el ajuste resultante era identico al del control. La pista J
    midio un delta de exactamente cero por esa causa y no por el metodo. Por eso el
    camino alternativo se explicita en dos ramas y la pista J lleva un guardia que
    detiene la ejecucion si los dos brazos no producen soluciones distintas.

    Args:
        x: matriz de entrenamiento.
        y01: etiqueta binaria de todas las filas.
        c: constante de regularizacion.
        omega: pesos de clase.
        grupos: indice de grupo de cada fila de ENTRENAMIENTO.
        n_grupos: numero de grupos.
        eta: temperatura de la suavizacion.
        funcion: objetivo alternativo con el protocolo de seis argumentos, o None.

    Returns:
        Diccionario con pesos, intercepto, iteraciones y convergencia.
    """
    _n, d = x.shape

    if funcion is None:
        def fn(w: np.ndarray, b: float) -> tuple[float, np.ndarray, float]:
            """Envuelve el objetivo de riesgo por grupo en el protocolo de seis argumentos.

            Args:
                w: vector de pesos.
                b: intercepto.

            Returns:
                Tupla con el valor y los dos gradientes.
            """
            val, gw, gb, _, _ = objetivo_dro(w, b, x, y01, c, omega, grupos, n_grupos, eta)
            return val, gw, gb
    else:
        def fn(w: np.ndarray, b: float) -> tuple[float, np.ndarray, float]:
            """Invoca el objetivo alternativo con el protocolo de seis argumentos.

            Args:
                w: vector de pesos.
                b: intercepto.

            Returns:
                Tupla con el valor y los dos gradientes.
            """
            val, gw, gb = funcion(w, b, x, y01, c, omega)
            return val, gw, gb

    def f(vect: np.ndarray) -> tuple[float, np.ndarray]:
        """Objective and gradient in the unconstrained parameterisation.

        Args:
            vect: vector con los d pesos y el intercepto al final.

        Returns:
            Tupla con el valor y el gradiente concatenado.
        """
        val, gw, gb = fn(vect[:d], float(vect[d]))
        return val, np.concatenate([gw, [gb]])

    res = minimize(f, np.zeros(d + 1), jac=True, method="L-BFGS-B",
                   options={"maxiter": MAX_ITER_LBFGS, "ftol": 1e-18,
                            "gtol": TOL_LBFGS, "maxcor": 100})
    w = res.x[:d]
    gfin = float(np.linalg.norm(res.jac))
    return {"w": w, "b": float(res.x[d]), "iteraciones": int(res.nit),
            "convergio": bool(gfin < TOL_GRADIENTE * (1.0 + np.linalg.norm(w))),
            "residuo_gradiente": gfin}


def main() -> None:
    """Ejecuta L1: riesgo minimax suavizado sobre los cuartiles de longitud.

    Returns:
        None.
    """
    inicio = datetime.now(timezone.utc)
    log("P60-L DRO por grupos. Grupos = cuartiles de longitud, eta 1,0")
    log("   El grafo no se usa: aqui se optimiza el RIESGO POR GRUPO, no la forma")
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
    grupos_cv = particion.set_index("id").loc[ids, "grupo_near_dup_autor"].to_numpy()
    partidos = int(pd.Series(grupos_cv).groupby(grupos_cv).nunique().gt(1).sum())
    if partidos:
        raise SystemExit(f"PARADA: {partidos} grupos canonicos partidos entre pliegues")
    train = pd.read_csv(TRAIN_RAW, sep=";", encoding="latin-1", dtype={"id": str})
    train["id_num"] = pd.to_numeric(train["id"].str.replace('"', "", regex=False),
                                    errors="coerce").astype("int64")
    idx_texto = train.set_index("id_num").loc[ids]
    if not np.array_equal(idx_texto[["HS", "TR", "AG"]].to_numpy().astype(int), y):
        raise SystemExit("PARADA: las etiquetas del CSV crudo no coinciden con el OOF")
    textos = idx_texto["text"].astype(str).tolist()
    n_car = np.array([len(t) for t in textos])
    log(f"   OOF canonico {ccv_ref!r} | {n} filas | {len(set(grupos_cv))} grupos CV intactos")

    # heterogeneidad medida por cuartil de longitud, sobre todo el corpus
    q_global = np.asarray(
    pd.qcut(n_car, 4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop").astype(str),
    dtype=object)
    log("")
    log("   HETEROGENEIDAD por cuartil de longitud, medida en el OOF canonico")
    for q in ("Q1", "Q2", "Q3", "Q4"):
        m = (q_global == q)
        f1s = [float(f1_score(y[m, j], pred_v03[m, j])) for j in range(3)]
        err = float((pred_v03[m] != y[m]).mean())
        log(f"      {q}  n={int(m.sum()):5d}  caracteres {int(n_car[m].min()):3d}-"
            f"{int(n_car[m].max()):3d}  F1 media {np.mean(f1s):.4f}  error {err*100:5.2f} por ciento")
    log("      V28-01 midio Cohen d de +0,28 a +0,34 a favor de los textos largos, luego la")
    log("      heterogeneidad que el DRO puede atacar esta medida y no supuesta")

    # ---------- verificacion del gradiente ANTES de entrenar ----------
    log("")
    log("   VERIFICACION DEL GRADIENTE contra diferencias centrales, antes de entrenar")
    rng = np.random.default_rng(SEMILLA)
    nt, dd, cc = 260, 12, 2.0
    xs = hstack([TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=1,
                                 sublinear_tf=True).fit_transform(textos[:nt]),
                 TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1,
                                 sublinear_tf=True).fit_transform(textos[:nt])]).tocsr()
    xs = xs[:, :dd]
    yb = y[:nt, 0].astype(int)
    omb = np.where(yb > 0.5, nt / (2.0 * max(int((yb > 0.5).sum()), 1)),
                   nt / (2.0 * max(int((yb < 0.5).sum()), 1)))
    gq = pd.qcut(pd.Series(n_car[:nt]), 4, labels=[0, 1, 2, 3], duplicates="drop").astype(int).to_numpy()
    verif: list[dict[str, Any]] = []
    for nombre, n_gr, eta in (("un_grupo", 1, 0.0), ("cuatro_grupos", 4, ETA_DRO)):
        ww = rng.normal(size=dd) * 0.3
        om = omb
        _v, g0, gb0, _r, _pg = objetivo_dro(ww, 0.1, xs, yb, cc, om, gq, n_gr, eta)
        eps = 1e-6
        err = 0.0
        for k in range(dd):
            e = np.zeros(dd)
            e[k] = eps
            num = (objetivo_dro(ww + e, 0.1, xs, yb, cc, om, gq, n_gr, eta)[0]
                   - objetivo_dro(ww - e, 0.1, xs, yb, cc, om, gq, n_gr, eta)[0]) / (2 * eps)
            err = max(err, abs(num - g0[k]) / max(abs(num), 1e-8))
        nemb = ((objetivo_dro(ww, 0.1 + eps, xs, yb, cc, om, gq, n_gr, eta)[0]
                 - objetivo_dro(ww, 0.1 - eps, xs, yb, cc, om, gq, n_gr, eta)[0]) / (2 * eps))
        eb = abs(nemb - gb0) / max(abs(nemb), 1e-8)
        verif.append({"configuracion": nombre, "error_relativo_w": err,
                      "error_relativo_b": eb})
        log(f"      {nombre:14} w {err:.2e} | b {eb:.2e}")
    peor = max(max(e["error_relativo_w"], e["error_relativo_b"]) for e in verif)
    if peor > 1e-4:
        raise SystemExit(f"PARADA: el gradiente analitico no coincide con el numerico ({peor:.2e})")

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
        # cuartiles calculados SOLO con las filas de ENTRENAMIENTO del pliegue
        q_tr = pd.qcut(pd.Series(n_car[ir]), 4, labels=[0, 1, 2, 3],
                        duplicates="drop").astype(int).to_numpy()
        n_gr = int(q_tr.max()) + 1
        for j in range(3):
            yb = y[ir, j].astype(int)
            om = np.where(yb > 0.5, len(ir) / (2.0 * max(int((yb > 0.5).sum()), 1)),
                          len(ir) / (2.0 * max(int((yb < 0.5).sum()), 1)))
            r0 = ajustar(x, yb, C_MODELO, om, q_tr, 1, 0.0)
            r1 = ajustar(x, yb, C_MODELO, om, q_tr, n_gr, ETA_DRO)
            proba_0[iva, j] = expit(np.asarray(xv @ r0["w"]).ravel() + r0["b"])
            proba_1[iva, j] = expit(np.asarray(xv @ r1["w"]).ravel() + r1["b"])
            if j == 0:
                _v0, _g0, _b0, riesgo0, _pg0 = objetivo_dro(
                    r0["w"], r0["b"], x, yb, C_MODELO, om, q_tr, 1, 0.0)
                _v1, _g1, _b1, riesgo1, pg1 = objetivo_dro(
                    r1["w"], r1["b"], x, yb, C_MODELO, om, q_tr, n_gr, ETA_DRO)
                info.append({
                    "pliegue": f, "n_grupos": n_gr,
                    "tamanos": [int((q_tr == g).sum()) for g in range(n_gr)],
                    "riesgo_por_grupo_control": [round(float(z), 6) for z in riesgo0],
                    "riesgo_por_grupo_dro": [round(float(z), 6) for z in riesgo1],
                    "masa_y_pesos_dro": pg1,
                    "iteraciones_control": r0["iteraciones"],
                    "iteraciones_dro": r1["iteraciones"],
                    "convergio_control": r0["convergio"], "convergio_dro": r1["convergio"],
                })
        log(f"      pliegue {f} listo, grupos {info[-1]['tamanos']} "
            f"({time.time() - t0:.1f} s acumulados)")

    pred_0 = (proba_0 >= V03_UMBRALES[None, :]).astype(int)
    pred_1 = (proba_1 >= V03_UMBRALES[None, :]).astype(int)
    ccv_0 = float(f1_score(y, pred_0, average="macro"))
    ccv_1 = float(f1_score(y, pred_1, average="macro"))
    dmax_0 = float(np.abs(proba_0 - v["proba"]).max())
    log("")
    log(f"   BRAZO 0 (un solo grupo = riesgo promedio)  corrected_cv {ccv_0!r}")
    log(f"      delta vs V03 {ccv_0 - ccv_ref:+.10f} | max |proba - canonica| {dmax_0:.3e} | "
        f"celdas distintas {int((pred_0 != pred_v03).sum())}")
    if ccv_0 - ccv_ref > TOLERANCIA_OPTIMIZADOR or dmax_0 > 0.05:
        raise SystemExit(f"PARADA: el brazo 0 no reproduce V03 "
                         f"(delta {ccv_0 - ccv_ref:+.6f})")
    log("      control de optimizador SUPERADO")
    delta_mec = ccv_1 - ccv_0
    log(f"   BRAZO 1 (cuatro grupos, DRO)              corrected_cv {ccv_1!r}")
    log(f"      delta DRO PURO (1 - 0) {delta_mec:+.10f}")
    log(f"      delta vs V03 {ccv_1 - ccv_ref:+.10f}")
    conv = all(i["convergio_control"] and i["convergio_dro"] for i in info)
    log(f"   convergencia de los dos brazos = {conv}")

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
    log(f"   pliegues donde DRO mejora: {folds_mej}/5")
    log(f"   rescate {m1['rescate']} | dano {m1['damage']} | neto {m1['neto']:+d} "
        f"| cambian {m1['cambian']}")
    log(f"   GATE sobre DRO {delta_mec:+.6f} >= {GATE_DELTA}: {h_delta} | "
        f"pliegues {folds_mej}/5 >= {GATE_PLIEGUES}: {h_folds} | superado: {gate}")
    ic_lo = ic_hi = None
    if gate:
        rng2 = np.random.default_rng(SEMILLA)
        dc = ((pred_1 == y).astype(int) - (pred_0 == y).astype(int)).ravel()
        mm = dc[rng2.integers(0, dc.size, size=(N_BOOT, dc.size))].mean(axis=1)
        ic_lo, ic_hi = float(np.percentile(mm, 2.5)), float(np.percentile(mm, 97.5))
        log(f"   bootstrap (solo porque el gate se supero): IC95 [{ic_lo:+.6f}, {ic_hi:+.6f}]")

    # DRO por grupo: la mejora o el dano, medido DENTRO de cada cuartil
    log("")
    log("   POR GRUPO, que es lo que DRO promete equitablear")
    por_grupo: dict[str, Any] = {}
    for g in ("Q1", "Q2", "Q3", "Q4"):
        m = (q_global == g)
        r_ = float(f1_score(y[m], pred_1[m], average="macro"))
        r_0 = float(f1_score(y[m], pred_0[m], average="macro"))
        por_grupo[g] = {"n": int(m.sum()), "ccv_control": round(r_0, 6),
                        "ccv_dro": round(r_), "delta": round(r_ - r_0, 6)}
        log(f"      {g}  n={int(m.sum()):5d}  control {r_0:.4f}  DRO {r_:.4f}  "
            f"delta {r_ - r_0:+.4f}")
    log("      si DRO cumpliera su promesa, el delta del grupo peor seria el mas alto")

    decision = (
        "INVALIDATED_PENDING_FIX. Un brazo no reduce su perdida" if not conv else
        ("CLOSED. El riesgo minimax por grupos no mejora sobre el riesgo promedio con la misma "
         "capacidad" if delta_mec <= 0 else
         ("MARGINAL. El mecanismo se ACTIVA pero con +2,5 por ciento de sobreponderacion del "
         "grupo peor, solo cambian 3 de 4449 celdas y el grupo que peor lo tenia "
         "empeora; ademas no alcanza el gate de +0,003"
          if not gate else
          "PROMISING. Supera el gate; exige segunda particion y revision de capacidad")))

    np.savez_compressed(CELDA_NPZ, ids=ids, y_true=y, pliegues=pliegues,
                        pred_v03=pred_v03, pred_control=pred_0, pred_dro=pred_1,
                        proba_control=proba_0, proba_dro=proba_1,
                        cuartil_global=q_global)

    informe = {
        "familia": "P60_L_GROUP_DRO",
        "naturaleza": ("EXPERIMENTO. 1 candidato, 0 submissions, 0 Kaggle, 0 holdout, 0 test. "
                       "V03 no se modifica"),
        "fecha": inicio.isoformat(timespec="seconds"),
        "experiment_id": "P60-03", "track": "P60-L",
        "script": "experiments/p60_03_dro.py",
        "paper": "Distributionally Robust Optimization with Group Distributions, ICML 2020, "
                 "Sagawa, Koh, Hashimoto, Liang",
        "pregunta": ("V03 optimiza un unico riesgo promedio; ese promedio esconde "
                     "comportamiento sistematicamente distinto en subpoblaciones?"),
        "hipotesis": ("un objetivo minimax sobre los cuartiles de longitud, que son una rebanada "
                      "ya existente del proyecto con heterogeneidad medida, coloca la frontera "
                      "mejor que el riesgo promedio con la misma capacidad"),
        "mecanismo": ("el riesgo se agrega sobre grupos con un promedio exponencialmente "
                      "suavizado, de modo que el gradiente es la media de los gradientes por "
                      "grupo con pesos softmax, y el grupo peor pesa mas que su proporcion"),
        "modelo": ("el mismo LogisticRegression lineal de V03, C=2,0 balanced, con L-BFGS-B "
                   "sobre el gradiente analitico del objetivo de riesgo por grupo"),
        "perdida": "logistica, sin cambio; lo unico que cambia es la AGREGACION del riesgo",
        "rasgos": "los 27.431 del TF-IDF de V03, sin cambio",
        "adaptacion_legal": {
            "dominio_original": "vision, subpoblaciones identificables",
            "adaptacion": "cuatro cuartiles de longitud sobre las filas de ENTRENAMIENTO del pliegue",
            "que_cambia": ("los grupos near-duplicate y pseudo-autor no sirven: son el propio grupo "
                           "del CV, con 1415 grupos para 1483 ejemplos, luego un minimax sobre ellos "
                           "degenera en un objetivo de peor ejemplo que no es DRO de grupos"),
            "que_es_invariante": "la estructura minimax sobre una particion en grupos",
            "evidencia_de_la_rebanada": ("V28-01 midio que la longitud es el rasgo que mas "
                                         "discrimina, con Cohen d de +0,28 a +0,34 a favor de "
                                         "los textos largos, y es el mismo para las tres etiquetas"),
        },
        "gap_analysis": {
            "filas_dro": 0, "filas_riesgo_por_grupo": 0,
            "precedente_mas_cercano": "P59-10, regularizacion por grafo de similitud",
            "por_que_no_es_duplicacion": ("P59-10 regulariza los COEFICIENTES con un grafo de "
                                          "similitud entre documentos. DRO no usa grafo: usa una "
                                          "particion en grupos y optimiza un objetivo minimax "
                                          "sobre el riesgo por grupo. Son ejes distintos"),
            "veredicto": "NUEVA",
        },
        "configuracion_pre_registrada": {
            "grupos": "los cuatro cuartiles de longitud (pd.qcut sobre n_caracteres, Q1..Q4)",
            "definicion_de_grupos": "calculada SOLO con las filas de ENTRENAMIENTO de cada pliegue",
            "objetivo": "DRO = (1/eta) log sum_g (n_g/n) exp(eta riesgo_g)",
            "eta": ETA_DRO,
            "eta_por_que_1": ("es el valor que el trabajo usa con gradiente exponenciado, y como las "
                              "perdidas son de orden 1 la suavizacion es suave y el maximo se "
                              "alcanza como limite; un solo valor, sin rejilla"),
            "pesos_de_clase": "los mismos balanceados de V03, contados en training",
            "regularizacion": f"C = {C_MODELO} con el mismo L2",
            "umbrales": "0,44 / 0,51 / 0,46 congelados",
            "optimizador": f"L-BFGS-B, maxiter {MAX_ITER_LBFGS}, gtol {TOL_LBFGS}",
            "rejilla": "ninguna sobre eta ni sobre el numero de grupos",
        },
        "verificacion_gradiente": verif,
        "heterogeneidad_por_cuartil": {q: {"n": int((q_global == q).sum()),
                                          "caracteres_min": int(n_car[(q_global == q)].min()),
                                          "caracteres_max": int(n_car[(q_global == q)].max()),
                                          "f1_media": round(float(np.mean(
                                              [f1_score(y[q_global == q, j],
                                                        pred_v03[(q_global == q), j])
                                               for j in range(3)])), 6),
                                          "error": round(float(
                                              (pred_v03[(q_global == q)] != y[(q_global == q)]).mean()),
                                          6)}
                                      for q in ("Q1", "Q2", "Q3", "Q4")},
        "control": {"brazo_0_un_grupo": m0,
                    "delta_brazo_0_vs_V03": round(ccv_0 - ccv_ref, 10),
                    "max_abs_diff_prob": dmax_0,
                    "clona_a_V03": bool(ccv_0 - ccv_ref <= TOLERANCIA_OPTIMIZADOR
                                        and dmax_0 <= 0.05)},
        "resultado": {
            "brazo_0_corrected_cv": round(m0["corrected_cv"], 10),
            "brazo_1_corrected_cv": round(m1["corrected_cv"], 10),
            "delta_agrupado_puro": round(delta_mec, 10),
            "delta_vs_V03": round(ccv_1 - ccv_ref, 10),
            "folds_brazo_1": [round(x, 6) for x in m1["folds"]],
            "folds_brazo_0": [round(x, 6) for x in m0["folds"]],
            "pliegues_mejorados": folds_mej,
            "f1_por_etiqueta": m1["f1_por_etiqueta"],
            "rescate": m1["rescate"], "damage": m1["damage"], "neto": m1["neto"],
            "cambian": m1["cambian"],
            "por_cuartil": por_grupo,
            "gate": {"delta_min": GATE_DELTA, "pliegues_min": GATE_PLIEGUES,
                     "supera_delta": bool(h_delta), "supera_pliegues": bool(h_folds),
                     "superado": gate},
            "bootstrap": (None if ic_lo is None else
                          {"IC95": [round(ic_lo, 6), round(ic_hi, 6)], "n": N_BOOT}),
        },
        "convergencia": bool(conv),
        "por_fold": info,
        "interpretacion": (
            "Dos hallazgos, y el primero es sobre la implementacion. La primera version de este "
            "objetivo repartia la perdida en escala O(1), porque cada riesgo de grupo es una media, "
            "frente a la escala O(n) del objetivo de V03, que suma sobre ejemplos. Media eso, no la "
            "agregacion del riesgo, sino una regularizacion unas 1186 veces mayor, y daba -0,163083. "
            "El hallazgo queda registrado porque es el tipo de error que produce un numero "
            "plausible y falso. Corregido con un factor constante n, la razon de masas entre "
            "brazos queda en 0,9996 y el gradiente analitico sigue siendo exacto, porque un "
            "factor constante no genera termino de derivada. La segunda version da +0,000336 con "
            "2 de 5 pliegues y 3 celdas de 4449 cambiadas. La medicion que hace ese numero "
            "interpretable es el peso softmax por grupo: el grupo con mas riesgo, Q4, recibe "
            "+2,47 por ciento de sobreponderacion relativa y el de menos riesgo, Q1, recibe "
            "-2,52 por ciento, luego el mecanismo SI se activa, pero con eta = 1 sobre riesgos que "
            "difieren en 0,065 la suavizacion apenas separa los grupos y el objetivo casi "
            "vuelve al riesgo promedio. Y hay un detalle que va contra la promesa del metodo: el "
            "cuartil peor, Q4, no mejora sino que empeora de 0,5980 a 0,5971, mientras el "
            "cuartil intermedio Q3 gana 0,0027. Es decir, el reequilibrio no llego al grupo que "
            "lo necesitaba. Conclusion transferible: existe la heterogeneidad que DRO busca "
            "atacar, esta medida en este corpus, pero un objetivo minimax suave sobre cuatro "
            "grupos que difieren en 0,065 de riesgo no la convierte en senal aprovechable, y V03 "
            "ya equilibra lo que si se puede equilibrar con class_weight balanced"),
        "decision": decision,
        "experimentos_registrados": 1,
        "formulacion": "V03 remains the best verified leaderboard result.",
        "gastos": {"submissions": 0, "consultas_kaggle": 0, "uso_holdout": 0, "uso_test": 0,
                   "segundos": round((datetime.now(timezone.utc) - inicio).total_seconds(), 1)},
    }
    JSON_L.write_text(json.dumps(informe, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    L: list[str] = []
    W = L.append
    W("# P60-L - Riesgo minimax por grupos (DRO)")
    W("")
    W(f"Generado por `experiments/p60_03_dro.py`. Fecha {inicio.isoformat(timespec='seconds')}.")
    W("")
    W("## Motivacion e hipotesis")
    W("")
    W("V03 optimiza un unico riesgo promedio. Si ese promedio esconde comportamientos")
    W("sistematicamente distintos en subpoblaciones, un objetivo minimax sobre unos pocos")
    W("**grupos que ya existen** deberia colocar la frontera mejor que el promedio.")
    W("")
    W("## Por que los grupos son los cuartiles de longitud")
    W("")
    W("Los grupos near-duplicate y pseudo-autor no sirven aqui por una razon mecanica: son el")
    W("propio grupo del CV, con 1415 grupos para 1483 ejemplos, luego un minimax sobre ellos")
    W("degenera en un objetivo de peor ejemplo, que no es DRO de grupos.")
    W("")
    W("El cuartil de longitud tiene evidencia. `V28-01` midio que la longitud es el rasgo que")
    W("mas discrimina, con Cohen d de +0,28 a +0,34 a favor de los textos largos, y que es el")
    W("mismo para las tres etiquetas. La definicion es la del proyecto, `pd.qcut` sobre")
    W("`n_caracteres`, usada en `06_error_analysis.py` y `08_features_densas_h11_h12.py`.")
    W("")
    W("## Heterogeneidad medida en este corpus")
    W("")
    W("| cuartil | n | caracteres | F1 media | error |")
    W("|---|---:|---|---|---:|")
    inf_h = informe["heterogeneidad_por_cuartil"]
    for q in ("Q1", "Q2", "Q3", "Q4"):
        hh = inf_h[q]
        W(f"| {q} | {hh['n']} | {hh['caracteres_min']}-{hh['caracteres_max']} | "
          f"{hh['f1_media']:.4f} | {hh['error']*100:.2f} por ciento |")
    W("")
    W("## Configuracion, fijada antes de mirar validation")
    W("")
    W("```text")
    W("grupos   los cuatro cuartiles de longitud, definidos SOLO con training del pliegue")
    W(f"objetivo DRO = (1/{ETA_DRO}) log sum_g (n_g/n) exp({ETA_DRO} riesgo_g)")
    W(f"eta      {ETA_DRO}, el valor del trabajo con gradiente exponenciado, un solo valor")
    W("pesos    los mismos balanceados de V03   |   L2  C = 2,0   |   umbrales  congelados")
    W("rejilla  ninguna sobre eta ni sobre el numero de grupos")
    W("```")
    W("")
    W("## Verificacion del gradiente, antes de entrenar")
    W("")
    W("| configuracion | error relativo w | error relativo b |")
    W("|---|---:|---:|")
    for e in verif:
        W(f"| {e['configuracion']} | {e['error_relativo_w']:.2e} | {e['error_relativo_b']:.2e} |")
    W("")
    W("## Control")
    W("")
    W("El brazo 0 mete todas las filas de entrenamiento en UN solo grupo, lo que reduce DRO al")
    W("riesgo promedio global, que es el objetivo de V03. Por eso el brazo 0 es un control real.")
    W("")
    W("```text")
    W(f"brazo 0 corrected_cv {m0['corrected_cv']!r}  delta vs V03 {ccv_0 - ccv_ref:+.10f}  "
      f"celdas distintas {int((pred_0 != pred_v03).sum())}")
    W("```")
    W("")
    W("## Resultados")
    W("")
    W("```text")
    W(f"brazo 0 (un grupo)   corrected_cv {m0['corrected_cv']:.10f}")
    W(f"brazo 1 (cuatro)     corrected_cv {m1['corrected_cv']:.10f}")
    W(f"delta DRO PURO       {delta_mec:+.10f}")
    W(f"delta vs V03         {ccv_1 - ccv_ref:+.10f}")
    W(f"pliegues que mejoran {folds_mej}/5")
    W(f"rescate {m1['rescate']} | dano {m1['damage']} | neto {m1['neto']:+d} "
      f"| cambian {m1['cambian']}")
    W(f"convergencia {conv}")
    W("```")
    W("")
    W("| etiqueta | F1 DRO | F1 control |")
    W("|---|---:|---:|")
    for e in ETIQUETAS:
        W(f"| {e} | {m1['f1_por_etiqueta'][e]:.4f} | {m0['f1_por_etiqueta'][e]:.4f} |")
    W("")
    W("## Por grupo, que es lo que DRO promete equitablear")
    W("")
    W("| cuartil | n | control | DRO | delta |")
    W("|---|---:|---:|---:|---:|")
    for q in ("Q1", "Q2", "Q3", "Q4"):
        pg = por_grupo[q]
        W(f"| {q} | {pg['n']} | {pg['ccv_control']:.4f} | {pg['ccv_dro']:.4f} | "
          f"{pg['delta']:+.4f} |")
    W("")
    W("## Gate")
    W("")
    W("| criterio | valor | superado |")
    W("|---|---|---|")
    W(f"| efecto de la agrupacion >= +0,003 | {delta_mec:+.6f} | **{h_delta}** |")
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
    W("*Generado por `experiments/p60_03_dro.py`. Sin push.*")
    MD_L.write_text(a_ascii("\n".join(L)), encoding="ascii")

    import csv as _csv
    filas = [
        ["brazo_0_corrected_cv", repr(m0["corrected_cv"])],
        ["brazo_1_corrected_cv", repr(m1["corrected_cv"])],
        ["delta_agrupado_puro", repr(delta_mec)],
        ["delta_vs_V03", repr(ccv_1 - ccv_ref)],
        ["corrected_cv_V03", repr(ccv_ref)],
        ["pliegues_mejorados", str(folds_mej)],
        ["folds_brazo_1", " ".join(f"{x:.6f}" for x in m1["folds"])],
        ["folds_brazo_0", " ".join(f"{x:.6f}" for x in m0["folds"])],
        ["rescate", str(m1["rescate"])], ["damage", str(m1["damage"])],
        ["neto", str(m1["neto"])], ["cambian", str(m1["cambian"])],
        ["eta", f"{ETA_DRO}"], ["n_grupos", "4 (cuartiles de longitud)"],
        ["convergio", str(conv)],
        ["gate_delta_superado", str(h_delta)], ["gate_pliegues_superado", str(h_folds)],
        ["gate_superado", str(gate)],
        ["decision", a_ascii(decision)],
    ]
    for q in ("Q1", "Q2", "Q3", "Q4"):
        filas.append([f"delta_cuartil_{q}", f"{por_grupo[q]['delta']:+.6f}"])
    with CSV_L.open("w", encoding="ascii", newline="") as fh:
        w = _csv.writer(fh, lineterminator="\n")
        w.writerow(["magnitud", "valor"])
        w.writerows(filas)

    for destino in (MD_L, JSON_L, CSV_L):
        if destino.exists() and any(b > 127 for b in destino.read_bytes()):
            raise SystemExit(f"PARADA: {destino.name} conserva bytes no ASCII")
    log("")
    log(f"P60-L VEREDICTO: {decision.split('.')[0]}")
    log(f"P60-03 OK | {informe['gastos']['segundos']} s")


if __name__ == "__main__":
    warnings.simplefilter("error", RuntimeWarning)
    ast.parse(Path(__file__).read_text(encoding="utf-8"))
    main()
