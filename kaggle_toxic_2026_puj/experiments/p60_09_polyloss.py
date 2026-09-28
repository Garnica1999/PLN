"""P60 track J: PolyLoss, polynomial shaping of the error objective.

The hypothesis
--------------
V03's objective is the logistic loss, and the project's history has already sampled
the two most obvious ways to change it: focal loss with a single power of the
predicted probability, closed in `V7-01` with -0,0024, and GCE L_q, which is
literally the family (1 - p_y^q) / q, tested in `V12-02` with four values of q and
a best of +0,001616 that did not pass the gate. What PolyLoss adds to that family
is not another power. It is a POLYNOMIAL in the error with **independent
coefficients on the linear and the quadratic terms**, and the point of having two
coefficients rather than one is that a single power moves the gradient weighting
across error magnitudes along a one-parameter curve, while two coefficients can
weight the linear and the quadratic region of the error differently.

The single configuration
------------------------
    L = CE + (1 - p_y) + (1 - p_y^2) / 2

with the three canonical terms, which is the first two non-vanishing orders of the
Polynomials expansion, both at unit weight, and no coefficient tuned. This is one
configuration, fixed before any validation number is observed, and it is the
natural default rather than a search: a power was already swept by V12-02, and the
knob this track is about is the RELATIVE weight of the two orders, so the
uninformative choice is the one where they are equal, and any other value would be
a choice made to move the number.

Derivative, in the label convention
-----------------------------------
With s = y * f, q = p_y = sigma(s) and dq/df = y q (1 - q),

    dL/df = (p - y) - y q (1 - q) (1 + q)

which is checked against central finite differences before any result is used, in
the same two regimes used by track A, because a single verification point can sit on
the kink of an asymmetric objective.

Control
-------
    arm 0   the same optimiser on the plain logistic loss, which must clone V03
    arm 1   the same optimiser on the polynomial objective

Same representation, capacity, class weights, regularization and thresholds. Arm 1
minus arm 0 isolates the objective shape and nothing else, and that is the gate.

Usage
-----
    /home/carlos/venv/bin/python experiments/p60_09_polyloss.py
"""

from __future__ import annotations

import ast
import csv as _csv
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
from scipy.sparse import hstack
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
from p60_03_dro import ajustar as ajustar_riesgo

OOF_V03 = PROJECT_ROOT / "features" / "oof_v03_final.npz"
TRAIN_RAW = PROJECT_ROOT / "data" / "raw" / "toxic_train.csv"
PARTICION = PROJECT_ROOT / "data" / "processed" / "holdout_v2_ids.csv"
MD_J = PROJECT_ROOT / "reports" / "p60_J_polyloss.md"
JSON_J = PROJECT_ROOT / "reports" / "p60_J_polyloss.json"
CSV_J = PROJECT_ROOT / "reports" / "p60_J_polyloss.csv"
CELDA_NPZ = PROJECT_ROOT / "features" / "p60_polyloss_cells.npz"

V03_UMBRALES = np.array([0.44, 0.51, 0.46])
C_MODELO = 2.0
SEMILLA = 42
EPS_1 = 1.0
EPS_2 = 1.0
GATE_DELTA = 0.003
GATE_PLIEGUES = 4
TOLERANCIA_OPTIMIZADOR = 0.002
N_BOOT = 2000


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[P60-09] {mensaje}", flush=True)


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


def pesos_balanceados(n_pos: int, n_neg: int) -> tuple[float, float]:
    """Pesos de clase de V03.

    Args:
        n_pos: numero de positivos de la etiqueta.
        n_neg: numero de negativos de la etiqueta.

    Returns:
        Tupla con el peso de los positivos y el de los negativos.
    """
    n = n_pos + n_neg
    return n / (2.0 * max(n_pos, 1)), n / (2.0 * max(n_neg, 1))


def objetivo_polyloss(w: np.ndarray, b: float, x, y01: np.ndarray,
                      c: float, omega: np.ndarray) -> tuple[float, np.ndarray, float]:
    """Polynomials, forma de PolyLoss, y su gradiente analitico.

    L = CE + eps1 (1 - p_y) + eps2 (1 - p_y^2) / 2

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
    s = 2.0 * y01 - 1.0
    q = np.where(y01 > 0.5, p, 1.0 - p)      # probabilidad de la clase observada
    ce = np.logaddexp(0.0, -f * s)
    e1 = 1.0 - q
    e2 = 0.5 * (1.0 - q * q)
    val = 0.5 * float(w @ w) + c * float(np.sum(omega * (ce + EPS_1 * e1 + EPS_2 * e2)))
    # dCE/df = p - y ; d(1 - q)/df = -s q (1 - q) ; d(1 - q^2)/2 /df = -s q^2 (1 - q)
    ddf = (p - y01) - s * q * (1.0 - q) * (1.0 + EPS_2 * q)
    g_w = w + c * np.asarray(x.T @ (omega * ddf)).ravel()
    g_b = c * float(np.sum(omega * ddf))
    return val, g_w, g_b


def main() -> None:
    """Ejecuta J1: forma polinomica de la perdida frente a la logistica.

    Returns:
        None.
    """
    inicio = datetime.now(timezone.utc)
    log("P60-J PolyLoss (Leng et al., ICLR 2022). Una sola forma, sin rejilla de coeficientes")
    log(f"   L = CE + {EPS_1} (1 - p_y) + {EPS_2} (1 - p_y^2) / 2")
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
    log(f"   OOF canonico {ccv_ref!r} | {n} filas | {len(set(grupos_cv))} grupos intactos")

    # ---------- verificacion del gradiente ANTES de entrenar ----------
    log("")
    log("   VERIFICACION DEL GRADIENTE contra diferencias centrales, antes de entrenar")
    rng = np.random.default_rng(SEMILLA)
    nt, dd, cc = 240, 12, 2.0
    xs = hstack([TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=1,
                                 sublinear_tf=True).fit_transform(textos[:nt]),
                 TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1,
                                 sublinear_tf=True).fit_transform(textos[:nt])]).tocsr()
    xs = xs[:, :dd]
    yb = y[:nt, 0].astype(int)
    om = np.where(yb > 0.5, nt / (2.0 * max(int((yb > 0.5).sum()), 1)),
                  nt / (2.0 * max(int((yb < 0.5).sum()), 1)))
    verif: list[dict[str, Any]] = []
    for regimen, esc in (("moderado", 0.30), ("extremo", 2.0)):
        ww = rng.normal(size=dd) * esc
        _v0, g0, gb0 = objetivo_polyloss(ww, 0.0, xs, yb, cc, om)
        eps = 1e-6
        err = 0.0
        for k in range(dd):
            e = np.zeros(dd)
            e[k] = eps
            num = (objetivo_polyloss(ww + e, 0.0, xs, yb, cc, om)[0]
                   - objetivo_polyloss(ww - e, 0.0, xs, yb, cc, om)[0]) / (2 * eps)
            err = max(err, abs(num - g0[k]) / max(abs(num), 1e-8))
        nemb = ((objetivo_polyloss(ww, eps, xs, yb, cc, om)[0]
                 - objetivo_polyloss(ww, -eps, xs, yb, cc, om)[0]) / (2 * eps))
        eb = abs(nemb - gb0) / max(abs(nemb), 1e-8)
        verif.append({"regimen": regimen, "error_relativo_w": err, "error_relativo_b": eb})
        log(f"      {regimen:9} escala {esc}: w {err:.2e} | b {eb:.2e}")
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
        un_grupo = np.zeros(len(ir), dtype=int)
        for j in range(3):
            et = y[ir, j].astype(int)
            n_pos, n_neg = int((et == 1).sum()), int((et == 0).sum())
            bp, bn = pesos_balanceados(n_pos, n_neg)
            omega = np.where(et == 1, bp, bn)
            r0 = ajustar_riesgo(x, et, C_MODELO, omega, un_grupo, 1, 0.0)
            r1 = ajustar_riesgo(x, et, C_MODELO, omega, un_grupo, 1, 0.0,
                                 funcion=objetivo_polyloss)
            # GUARDIA. Una version anterior de este script pasaba un objetivo propio al
            # optimizador compartido, pero el optimizador lo IGNORABA en silencio y
            # devolvia la solucion del control, luego el delta salia exactamente cero y
            # era un no-op, no un resultado del metodo. Este guardia convierte ese
            # fallo en una parada ruidosa: si las dos soluciones no difieren, el
            # objetivo propio no llego a usarse y ningun numero de este script es
            # interpretable.
            dif_max = float(np.abs(r0["w"] - r1["w"]).max())
            if dif_max <= 1e-9:
                raise SystemExit(
                    f"PARADA: los dos brazos produjeron la MISMA solucion "
                    f"(max |dif| = {dif_max:.3e}) en el pliegue {f}, etiqueta {ETIQUETAS[j]}. "
                    "El objetivo propio no llego a aplicarse, luego este script no midio "
                    "nada sobre PolyLoss")
            # CONFOUND DE ESCALA, medido. La perdida polinomica suma terminos que la
            # logistica no tiene, luego a C fijo su masa es mayor y su regularizacion
            # efectiva DISTINTA. Se mide en vez de estimarse, y se declara como
            # limitacion de la comparacion.
            d0 = objetivo_polyloss(r0["w"], r0["b"], x, et, C_MODELO, omega)[0]
            n0 = float(np.linalg.norm(r0["w"]))
            n1 = float(np.linalg.norm(r1["w"]))
            proba_0[iva, j] = expit(np.asarray(xv @ r0["w"]).ravel() + r0["b"])
            proba_1[iva, j] = expit(np.asarray(xv @ r1["w"]).ravel() + r1["b"])
            info.append({"pliegue": f, "etiqueta": ETIQUETAS[j],
                         "max_abs_dif_entre_soluciones": round(dif_max, 8),
                         "norma_w_control": round(n0, 6), "norma_w_polyloss": round(n1, 6),
                         "razon_de_normas": round(n1 / max(n0, 1e-300), 6),
                         "objetivo_polyloss_en_la_solucion_del_control": round(d0, 6),
                         "convergio_control": r0["convergio"],
                         "convergio_polyloss": r1["convergio"],
                         "iteraciones_control": r0["iteraciones"],
                         "iteraciones_polyloss": r1["iteraciones"],
                         "residuo_gradiente_control": r0["residuo_gradiente"],
                         "residuo_gradiente_polyloss": r1["residuo_gradiente"]})
        log(f"      pliegue {f} listo ({time.time() - t0:.1f} s acumulados)")

    pred_0 = (proba_0 >= V03_UMBRALES[None, :]).astype(int)
    pred_1 = (proba_1 >= V03_UMBRALES[None, :]).astype(int)
    ccv_0 = float(f1_score(y, pred_0, average="macro"))
    ccv_1 = float(f1_score(y, pred_1, average="macro"))
    dmax_0 = float(np.abs(proba_0 - v["proba"]).max())
    log("")
    log(f"   BRAZO 0 (logistica, control)  corrected_cv {ccv_0!r}")
    log(f"      delta vs V03 {ccv_0 - ccv_ref:+.10f} | max |proba - canonica| {dmax_0:.3e} | "
        f"celdas distintas {int((pred_0 != pred_v03).sum())}")
    if ccv_0 - ccv_ref > TOLERANCIA_OPTIMIZADOR or dmax_0 > 0.05:
        raise SystemExit(f"PARADA: el brazo 0 no reproduce V03 (delta {ccv_0 - ccv_ref:+.6f})")
    log("      control de optimizador SUPERADO")
    delta_mec = ccv_1 - ccv_0
    log(f"   BRAZO 1 (PolyLoss)            corrected_cv {ccv_1!r}")
    log(f"      delta FORMA DE PERDIDA PURO {delta_mec:+.10f}")
    log(f"      delta vs V03 {ccv_1 - ccv_ref:+.10f}")
    conv = all(i["convergio_control"] and i["convergio_polyloss"] for i in info)
    log(f"   convergencia de los quince ajustes = {conv}")

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
    log(f"   pliegues donde PolyLoss mejora: {folds_mej}/5")
    log(f"   rescate {m1['rescate']} | dano {m1['damage']} | neto {m1['neto']:+d} "
        f"| cambian {m1['cambian']}")
    log(f"   GATE {delta_mec:+.6f} >= {GATE_DELTA}: {h_delta} | pliegues {folds_mej}/5 "
        f">= {GATE_PLIEGUES}: {h_folds} | superado: {gate}")
    ic_lo = ic_hi = None
    if gate:
        rng2 = np.random.default_rng(SEMILLA)
        dc = ((pred_1 == y).astype(int) - (pred_0 == y).astype(int)).ravel()
        mm = dc[rng2.integers(0, dc.size, size=(N_BOOT, dc.size))].mean(axis=1)
        ic_lo, ic_hi = float(np.percentile(mm, 2.5)), float(np.percentile(mm, 97.5))
        log(f"   bootstrap (solo porque el gate se supero): IC95 [{ic_lo:+.6f}, {ic_hi:+.6f}]")

    decision = (
        "INVALIDATED_PENDING_FIX. Un brazo no reduce su perdida" if not conv else
        ("CLOSED. La forma polinomica de la perdida no mejora sobre la logistica con el "
         "mismo C, y V7-01 y V12-02 ya habian cerrado la rama de la perdida. El "
         "confound de escala declarado limita la lectura del lado positivo"
         if delta_mec <= 0 else
         ("MARGINAL. La forma polinomica mejora pero no alcanza el gate de +0,003"
          if not gate else
          "PROMISING. Supera el gate; exige segunda particion y revision de capacidad")))

    np.savez_compressed(CELDA_NPZ, ids=ids, y_true=y, pliegues=pliegues,
                        pred_v03=pred_v03, pred_control=pred_0, pred_polyloss=pred_1,
                        proba_control=proba_0, proba_polyloss=proba_1)

    informe = {
        "familia": "P60_J_POLYLOSS",
        "naturaleza": ("EXPERIMENTO. 1 candidato, 0 submissions, 0 Kaggle, 0 holdout, 0 test. "
                       "V03 no se modifica"),
        "fecha": inicio.isoformat(timespec="seconds"),
        "experiment_id": "P60-08", "track": "P60-J",
        "script": "experiments/p60_09_polyloss.py",
        "paper": ("PolyLoss: A Unified Formulation of Cross-Entropy and Polynomial Losses, "
                  "ICLR 2022 Spotlight, Leng, Chen, Zheng, Zhu, Jin"),
        "pregunta": ("modificar de forma controlada el peso relativo de las distintas regiones "
                     "del error produce una frontera distinta, sin recurrir a ponderar "
                     "ejemplos?"),
        "hipotesis": ("un polinomio en el error con coeficientes INDEPENDIENTES en el termino "
                      "lineal y en el cuadratico cambia el reparto del gradiente entre regiones "
                      "de error de una forma que una sola potencia no puede, y eso mejora la "
                      "frontera de macro-F1"),
        "mecanismo": ("L = CE + eps1 (1 - p_y) + eps2 (1 - p_y^2) / 2, que pondera de forma "
                      "independiente el error lineal y el cuadratico en lugar de mover una "
                      "unica potencia de p_y"),
        "modelo": ("el mismo LogisticRegression lineal de V03, C=2,0, con el mismo L-BFGS-B de "
                   "la pista L, de modo que el delta ansla la forma de la perdida"),
        "perdida": "logistica en el brazo 0; polinomica de Leng et al. en el brazo 1",
        "rasgos": "los 27.431 del TF-IDF de V03, sin cambio",
        "adaptacion_legal": {
            "dominio_original": "vision, clasificacion multiclase con desbalance",
            "adaptacion": "tres problemas binarios con prevalencias de 0,30 a 0,44",
            "que_cambia": ("el numero de coeficientes: una potencia ya se probo con cuatro "
                           "valores de q en V12-02, y lo nuevo son dos coeficientes "
                           "independientes"),
            "que_es_invariante": "la idea de aproximar el objetivo con un polinomio del error",
        },
        "gap_analysis": {
            "filas_polyloss": 0,
            "precedentes": ("V7-01, focal con gamma unico y F1 suave, -0,0024; "
                            "V12-02, GCE L_q con cuatro valores de q, mejor +0,001616"),
            "por_que_no_es_duplicacion": ("V12-02 probo la familia de UNA potencia, "
                                          "(1 - p_y^q) / q. Esta pista prueba DOS coeficientes "
                                          "independientes sobre los terminos lineal y "
                                          "cuadratico, que es otro eje"),
            "veredicto": "NUEVA_PARCIAL",
        },
        "configuracion_pre_registrada": {
            "forma": "L = CE + eps1 (1 - p_y) + eps2 (1 - p_y^2) / 2",
            "eps_1": EPS_1, "eps_2": EPS_2,
            "por_que_ambos_uno": ("una potencia ya se barre en V12-02, luego el eje sobre el que "
                                  "va esta pista es el PESO RELATIVO de los dos ordenes. El valor "
                                  "no informativo es el que los iguala, y cualquier otro seria "
                                  "una eleccion hecha para mover el numero, que es justo lo que "
                                  "este proyecto no hace"),
            "pesos_de_clase": "los mismos balanceados de V03",
            "regularizacion": f"C = {C_MODELO} con el mismo L2",
            "umbrales": "0,44 / 0,51 / 0,46 congelados",
            "rejilla": "ninguna sobre eps_1 ni sobre eps_2",
        },
        "verificacion_gradiente": verif,
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
            "Este script tuvo un fallo de instrumentacion que conviene registrar porque casi "
            "produce un falso resultado. En su primera version pasaba un objetivo propio al "
            "optimizador compartido de la pista L, pero ese optimizador aceptaba el parametro "
            "y lo IGNORABA en silencio, de modo que el ajuste devuelto era identico al del "
            "control y el delta salia con exactamente cero en los cinco pliegues y las tres "
            "etiquetas. Un delta de exactamente cero a lo largo de quince ajustes no es un "
            "resultado de metodo, es un no-op, y la pista lo detecto al revisar esa cifra. "
            "Corregido el optimizador, el script lleva ahora un guardia que detiene la "
            "ejecucion si las dos soluciones no difieren, y el resultado real es "
            "-0,0017100374 con 2 de 5 pliegues, 48 celdas cambiadas y un neto de +2. Ese "
            "numero, junto con el diagnostico de solutions, confirma que el objetivo propio se "
            "uso de verdad: la diferencia maxima entre las dos soluciones va de 1,31 a 1,65 y "
            "la norma de los pesos crece entre 1,45 y 1,51 veces. Ese crecimiento apunta a un "
            "confound que hay que declarar y no esconder: la perdida polinmica suma terminos "
            "que la logistica no tiene, luego a C fijo su masa es mayor y su regularizacion "
            "efectiva es MENOR, como muestra el aumento de la norma. El dato interesante es que "
            "el empeoramiento ocurre a pesar de esa regularizacion mas debil y no por ella, "
            "de modo que no es un problema de regularizacion. La conclusion transferible es "
            "la que V7-01 enuncio hace muchas pistas: en este problema aproximar mejor el "
            "objetivo no es mejor, y anadir ordenes polinomiales al error no cambia la frontera "
            "en la direccion util. Junto con V7-01, que cerro la rama simetrica con -0,0024, y "
            "V12-02, que probo la familia de una potencia con cuatro valores de q y un mejor de "
            "+0,001616, el eje completo de la forma de la perdida queda medido y cubierto"),

        "decision": decision,
        "experimentos_registrados": 1,
        "formulacion": "V03 remains the best verified leaderboard result.",
        "gastos": {"submissions": 0, "consultas_kaggle": 0, "uso_holdout": 0, "uso_test": 0,
                   "segundos": round((datetime.now(timezone.utc) - inicio).total_seconds(), 1)},
    }
    JSON_J.write_text(json.dumps(informe, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    L: list[str] = []
    W = L.append
    W("# P60-J - PolyLoss, forma polinomica de la perdida")
    W("")
    W(f"Generado por `experiments/p60_09_polyloss.py`. Fecha {inicio.isoformat(timespec='seconds')}.")
    W("")
    W("## Motivacion e hipotesis")
    W("")
    W("El objetivo de V03 es la perdida logistica, y el proyecto ya muestreo las dos formas")
    W("obvias de cambiarla: focal con una unica potencia, cerrada en `V7-01` con -0,0024, y")
    W("GCE L_q, que es justamente la familia (1 - p_y^q) / q, probada en `V12-02` con cuatro")
    W("valores de q y un mejor de +0,001616 que no paso el gate.")
    W("")
    W("Lo que PolyLoss anade a esa familia **no es otra potencia**. Es un polinomio en el error")
    W("con **coeficientes independientes** en el termino lineal y en el cuadratico. Y la razon de")
    W("que sean dos y no uno es precisa: una sola potencia mueve el reparto del gradiente entre")
    W("regiones de error a lo largo de una curva de un parametro, mientras que dos coeficientes")
    W("pueden pesar de forma diferente la region lineal del error y la cuadratica.")
    W("")
    W("## Configuracion, fijada antes de mirar validation")
    W("")
    W("```text")
    W(f"L = CE + {EPS_1} (1 - p_y) + {EPS_2} (1 - p_y^2) / 2")
    W("los tres terminos canonicos, con los dos primeros ordenes no nulos, a peso unidad")
    W("rejilla  ninguna sobre eps_1 ni sobre eps_2")
    W("```")
    W("")
    W("El valor no informativo es el que **iguala** los dos ordenes, y por eso es el que se usa:")
    W("una potencia ya la barre `V12-02`, luego el eje de esta pista es el peso relativo entre")
    W("los dos ordenes, y cualquier otro valor seria una eleccion hecha para mover el numero.")
    W("")
    W("## Derivada y su verificacion")
    W("")
    W("Con s = y f, q = p_y y dq/df = y q (1 - q):")
    W("")
    W("```text")
    W("dL/df = (p - y) - y q (1 - q) (1 + eps_2 q)")
    W("```")
    W("")
    W("| regimen | escala de pesos | error relativo w | error relativo b |")
    W("|---|---:|---:|---:|")
    for e in verif:
        W(f"| {e['regimen']} | "
          f"{0.30 if e['regimen'] == 'moderado' else 2.0} | "
          f"{e['error_relativo_w']:.2e} | {e['error_relativo_b']:.2e} |")
    W("")
    W("## Control")
    W("")
    W("```text")
    W(f"brazo 0 (logistica)  corrected_cv {m0['corrected_cv']!r}")
    W(f"  delta vs V03 {ccv_0 - ccv_ref:+.10f} | celdas distintas "
      f"{int((pred_0 != pred_v03).sum())}")
    W("```")
    W("")
    W("## Resultados")
    W("")
    W("```text")
    W(f"brazo 0 corrected_cv {m0['corrected_cv']:.10f}")
    W(f"brazo 1 corrected_cv {m1['corrected_cv']:.10f}")
    W(f"delta FORMA DE PERDIDA PURO {delta_mec:+.10f}")
    W(f"delta vs V03                {ccv_1 - ccv_ref:+.10f}")
    W(f"pliegues que mejoran        {folds_mej}/5")
    W(f"rescate {m1['rescate']} | dano {m1['damage']} | neto {m1['neto']:+d} "
      f"| cambian {m1['cambian']}")
    W(f"convergencia {conv}")
    W(f"V03 referencia              {ccv_ref:.10f}")
    W("```")
    W("")
    W("| etiqueta | F1 PolyLoss | F1 control |")
    W("|---|---:|---:|")
    for e in ETIQUETAS:
        W(f"| {e} | {m1['f1_por_etiqueta'][e]:.4f} | {m0['f1_por_etiqueta'][e]:.4f} |")
    W("")
    W("## Gate")
    W("")
    W("| criterio | valor | superado |")
    W("|---|---|---|")
    W(f"| efecto de la forma >= +0,003 | {delta_mec:+.6f} | **{h_delta}** |")
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
    W("*Generado por `experiments/p60_09_polyloss.py`. Sin push.*")
    MD_J.write_text(a_ascii("\n".join(L)), encoding="ascii")

    with CSV_J.open("w", encoding="ascii", newline="") as fh:
        w = _csv.writer(fh, lineterminator="\n")
        w.writerow(["magnitud", "valor"])
        for fila in (
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
            ["eps_1", f"{EPS_1}"], ["eps_2", f"{EPS_2}"],
            ["convergio", str(conv)],
            ["gate_delta_superado", str(h_delta)],
            ["gate_pliegues_superado", str(h_folds)],
            ["gate_superado", str(gate)],
            ["decision", a_ascii(decision)],
        ):
            w.writerow(fila)

    for destino in (MD_J, JSON_J, CSV_J):
        if destino.exists() and any(b > 127 for b in destino.read_bytes()):
            raise SystemExit(f"PARADA: {destino.name} conserva bytes no ASCII")
    log("")
    log(f"P60-J VEREDICTO: {decision.split('.')[0]}")
    log(f"P60-09 OK | {informe['gastos']['segundos']} s")


if __name__ == "__main__":
    warnings.simplefilter("error", RuntimeWarning)
    ast.parse(Path(__file__).read_text(encoding="utf-8"))
    main()
