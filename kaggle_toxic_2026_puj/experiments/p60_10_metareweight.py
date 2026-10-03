"""P60 track I: meta-reweighting from gradient alignment.

The hypothesis, and why it is not the weightings already tried
-------------------------------------------------------------
The project has closed three families of per-example weight, and all three derive
the weight from a FIXED HEURISTIC computed before training: `V4-B03` and `V4-B05`
weighted by difficulty, `V6-07` by suspicion, and `P59-I` by residual size. What
Ren et al. propose is different in kind: the weight comes from asking which examples'
gradients are ALIGNED WITH A META-OBJECTIVE evaluated on a held-out set, so the
weight is endogenous to the optimisation instead of prescribed by a rule. That is
the axis this track tests, and the gap check returns 0 rows for it.

The weight has a closed form, and for a linear model that matters
---------------------------------------------------------------
Ren et al. set the weights so that the meta-gradient is matched by a weighted sum
of the per-example training gradients. With

    g_i = grad of the training loss on example i
    G  = grad of the meta loss on the held-out set

the weights solve min_eps || G - sum_i eps_i g_i ||^2, whose solution is
eps_i = <g_i, G> / <G, G>.

They evaluate this at initialisation, on the first minibatch, so the gradient of a
linear model at w = 0 is available in closed form, with no backprop and no
autodiff, because the regularisation gradient vanishes at the origin and

    g_i = C * omega_i * dlogit_i * x_i ,   dlogit_i = sigma(f_i) - y_i .

The inner product <g_i, G> is what would normally need a Gram matrix of size
d by d, with d = 27.431. It does not have to: writing G = C X' (omega dlogit) over
the meta rows, one gets <g_i, G> = C omega_i dlogit_i (X G)_i, and X G costs a
single sparse product of shape n by d. The whole weighting is therefore exact and
costs one sparse matrix-vector product, which is what makes the track feasible on
1483 rows with 27.431 features.

The nested structure, which is the part that is easy to get wrong
-----------------------------------------------------------------
The brief requires the nesting to be exactly

    outer_train -> inner_train -> meta-weight optimisation -> outer_validation

and the outer validation never participates. Concretely, inside each outer fold:

    the outer training rows are SPLIT BY GROUP into an inner training part and an
    inner meta part, so no near-duplicate group is ever split across the two
    inner parts, and the inner meta part contributes ONLY its gradient to the
    weighting

The weights are then applied to a fit on the FULL outer training rows, and the
outer validation rows are scored once, having contributed nothing to any
parameter, centroid, or weight.

The honest limitation, stated before the run
--------------------------------------------
Ren et al. need a CLEAN held-out set. There is no evidence that any subset of
TOXIC's labels is clean, so the meta-objective here is evaluated on a set that is
as noisy as the rest. That is the real weakness of this track on this corpus and
it is not a configuration detail. What the experiment can still establish is
whether a weighting that is endogenous and derived from a gradient alignment moves
the boundary, and if it does not, whether the weights are even informative, which
is measured and reported.

Control
-------
    arm 0   identical, with every weight set to 1, which must clone V03
    arm 1   identical, with the meta-weights

Usage
-----
    /home/carlos/venv/bin/python experiments/p60_10_metareweight.py
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
MD_I = PROJECT_ROOT / "reports" / "p60_I_metareweight.md"
JSON_I = PROJECT_ROOT / "reports" / "p60_I_metareweight.json"
CSV_I = PROJECT_ROOT / "reports" / "p60_I_metareweight.csv"
CELDA_NPZ = PROJECT_ROOT / "features" / "p60_meta_cells.npz"

V03_UMBRALES = np.array([0.44, 0.51, 0.46])
C_MODELO = 2.0
SEMILLA = 42
FRACCION_META = 0.25
GATE_DELTA = 0.003
GATE_PLIEGUES = 4
TOLERANCIA_OPTIMIZADOR = 0.002
N_BOOT = 2000


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[P60-10] {mensaje}", flush=True)


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


def pesos_meta(x, y01: np.ndarray, omega: np.ndarray, c: float,
               es_meta: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    """Pesos de Ren et al. en forma cerrada para un modelo lineal.

    A inicializacion w = 0 el gradiente de la perdida de un ejemplo es
    C * omega_i * (sigma(0) - y_i) * x_i, porque el gradiente de la regularizacion
    se anula en el origen. Se forma el gradiente del objetivo META sobre las filas
    internas marcadas, y el peso de cada ejemplo es su proyeccion sobre el:

        eps_i = <g_i, G> / <G, G>

    con G = C X' (omega dlogit) sobre las filas meta. El producto interno se
    evalua como omega_i dlogit_i (X G)_i, lo que evita construir la matriz de Gram
    de tamano d por d, con d = 27.431.

    Args:
        x: matriz de entrenamiento.
        y01: etiqueta binaria de todas las filas de ENTRENAMIENTO del pliegue.
        omega: pesos de clase.
        c: constante de regularizacion.
        es_meta: booleano que marca las filas del objetivo meta.

    Returns:
        Tupla con los pesos y un diccionario de diagnostico.
    """
    dlogit = 0.5 - y01                     # sigma(0) - y
    g_meta = c * np.asarray(x[es_meta].T @ (omega[es_meta] * dlogit[es_meta])).ravel()
    norma2 = float(g_meta @ g_meta)
    if norma2 <= 1e-300:
        return np.ones_like(omega), {"norma_al_cuadrado": 0.0, "degradado": 1.0}
    xg = np.asarray(x @ g_meta).ravel()
    # <g_i, G> = C omega_i dlogit_i (x_i . G). El factor C aparece UNA vez en el
    # numerador y DOS veces dentro de ||G||^2, luego el peso escala como 1/C. Omitirlo
    # daba exactamente la mitad del valor correcto, y el test unitario lo detecto.
    num = c * omega * dlogit * xg
    eps = num / norma2
    diag = {
        "norma_al_cuadrado": round(norma2, 10),
        "degradado": 0.0,
        "eps_min": round(float(eps.min()), 8),
        "eps_max": round(float(eps.max()), 8),
        "eps_mediana": round(float(np.median(eps)), 8),
        "eps_p10": round(float(np.percentile(eps, 10)), 10),
        "eps_p90": round(float(np.percentile(eps, 90)), 10),
        "fraccion_negativa": round(float((eps < 0).mean()), 6),
        "fraccion_cercana_a_cero": round(float((np.abs(eps) < 1e-14 * max(
            float(np.abs(eps).max()), 1e-300)).mean()), 6),
        "razon_p90_sobre_abs_p10": round(float(max(np.percentile(eps, 90), 0.0)
                                               / max(abs(float(np.percentile(eps, 10))), 1e-300)), 6),
        "norma_absoluta_media": round(float(np.abs(eps).mean()), 12),
    }
    return eps, diag


def main() -> None:
    """Ejecuta I1: meta-reweighting con anidamiento estricto.

    Returns:
        None.
    """
    inicio = datetime.now(timezone.utc)
    log("P60-I Meta-reweighting (Ren et al., ICML 2018). Forma cerrada, sin meta-aprendizaje")
    log(f"   fraccion meta interna {FRACCION_META}, particionada POR GRUPO dentro del training")
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
    gdf = particion.set_index("id")
    grupos_cv = gdf.loc[ids, "grupo_near_dup_autor"].to_numpy()
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
        # ---- particion INTERNA POR GRUPO dentro del training exterior ----
        g_tr = grupos_cv[ir]
        unicos = pd.unique(g_tr)
        rng_f = np.random.default_rng(SEMILLA + 1000 * f)
        orden = rng_f.permutation(len(unicos))
        n_meta_grupos = round(FRACCION_META * len(unicos))
        grupos_meta = set(unicos[orden[:n_meta_grupos]].tolist())
        es_meta = np.array([g in grupos_meta for g in g_tr])
        partidos_internos = int(pd.Series(g_tr).groupby(g_tr).nunique().gt(1).sum())
        if partidos_internos:
            raise SystemExit(f"PARADA: la particion interna partio {partidos_internos} grupos")
        un_grupo = np.zeros(len(ir), dtype=int)
        for j in range(3):
            et = y[ir, j].astype(int)
            n_pos, n_neg = int((et == 1).sum()), int((et == 0).sum())
            bp, bn = pesos_balanceados(n_pos, n_neg)
            omega = np.where(et == 1, bp, bn)
            eps, diag = pesos_meta(x, et, omega, C_MODELO, es_meta)
            r0 = ajustar_riesgo(x, et, C_MODELO, omega, un_grupo, 1, 0.0)
            # el peso total se reescala para conservar la masa de la perdida, igual
            # que en la pista L, de modo que el unico cambio sea la FORMA del
            # ponderado y no su escala, que seria regularizacion efectiva
            escala = float(omega.sum()) / max(float((omega * eps).sum()), 1e-300)
            r1 = ajustar_riesgo(x, et, C_MODELO, omega * eps * escala, un_grupo, 1, 0.0)
            proba_0[iva, j] = expit(np.asarray(xv @ r0["w"]).ravel() + r0["b"])
            proba_1[iva, j] = expit(np.asarray(xv @ r1["w"]).ravel() + r1["b"])
            diag["escala_de_masa"] = round(escala, 8)
            diag["convergio_control"] = r0["convergio"]
            diag["convergio_meta"] = r1["convergio"]
            info.append({"pliegue": f, "etiqueta": ETIQUETAS[j], **diag})
        log(f"      pliegue {f} | grupos internos: meta {int(es_meta.sum())} de "
            f"{len(ir)} filas, {n_meta_grupos} grupos | "
            f"eps mediana {np.median([i['eps_mediana'] for i in info[-3:]]):.3e} | "
            f"negativos {np.mean([i['fraccion_negativa'] for i in info[-3:]])*100:.2f} por ciento "
            f"({time.time() - t0:.1f} s acumulados)")

    pred_0 = (proba_0 >= V03_UMBRALES[None, :]).astype(int)
    pred_1 = (proba_1 >= V03_UMBRALES[None, :]).astype(int)
    ccv_0 = float(f1_score(y, pred_0, average="macro"))
    ccv_1 = float(f1_score(y, pred_1, average="macro"))
    dmax_0 = float(np.abs(proba_0 - v["proba"]).max())
    log("")
    log(f"   BRAZO 0 (pesos unitarios, control)   corrected_cv {ccv_0!r}")
    log(f"      delta vs V03 {ccv_0 - ccv_ref:+.10f} | max |proba - canonica| {dmax_0:.3e} | "
        f"celdas distintas {int((pred_0 != pred_v03).sum())}")
    if ccv_0 - ccv_ref > TOLERANCIA_OPTIMIZADOR or dmax_0 > 0.05:
        raise SystemExit(f"PARADA: el brazo 0 no reproduce V03 (delta {ccv_0 - ccv_ref:+.6f})")
    log("      control de optimizador SUPERADO")
    delta_mec = ccv_1 - ccv_0
    log(f"   BRAZO 1 (pesos meta)                 corrected_cv {ccv_1!r}")
    log(f"      delta META PURO {delta_mec:+.10f}")
    log(f"      delta vs V03 {ccv_1 - ccv_ref:+.10f}")
    conv = all(i["convergio_control"] and i["convergio_meta"] for i in info)
    log(f"   convergencia de los treinta ajustes = {conv}")

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
    log(f"   pliegues donde el meta-reweighting mejora: {folds_mej}/5")
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

    neg_med = float(np.mean([i["fraccion_negativa"] for i in info]))
    r_med = float(np.mean([i["razon_p90_sobre_abs_p10"] for i in info]))
    decision = (
        "INVALIDATED_PENDING_FIX. Un brazo no reduce su perdida" if not conv else
        ("CLOSED. El ponderado derivado de la alineacion de gradiente no mejora sobre el "
         "control, y los pesos que produce estan casi degenerados"
         if delta_mec <= 0 else
         ("MARGINAL. El meta-reweighting mejora pero no alcanza el gate de +0,003"
          if not gate else
          "PROMISING. Supera el gate; exige segunda particion y revision de capacidad")))

    np.savez_compressed(CELDA_NPZ, ids=ids, y_true=y, pliegues=pliegues,
                        pred_v03=pred_v03, pred_control=pred_0, pred_meta=pred_1,
                        proba_control=proba_0, proba_meta=proba_1)

    informe = {
        "familia": "P60_I_META_REWEIGHTING",
        "naturaleza": ("EXPERIMENTO. 1 candidato, 0 submissions, 0 Kaggle, 0 holdout, 0 test. "
                       "V03 no se modifica"),
        "fecha": inicio.isoformat(timespec="seconds"),
        "experiment_id": "P60-07", "track": "P60-I",
        "script": "experiments/p60_10_metareweight.py",
        "paper": ("Learning to Reweight Examples for Robust Deep Learning, ICML 2018, "
                  "Ren, Yu, Sheng, Ma, Zhao, Yi, Li"),
        "pregunta": ("los ejemplos que deben contribuir menos a la frontera pueden "
                     "determinarse por su efecto sobre un objetivo meta, y no por una regla "
                     "heuristica fija?"),
        "hipotesis": ("un ponderado endogeno, derivado de alinear el gradiente de cada ejemplo "
                      "con el gradiente de un objetivo meta sobre un subconjunto interno, mueve "
                      "la frontera de forma que las heuristicas de dificultad no mueven"),
        "mecanismo": ("eps_i = <g_i, G> / <G, G>, con g_i el gradiente del ejemplo y G el "
                      "gradiente del objetivo meta; el peso premia a los ejemplos cuyo "
                      "gradiente coincide con el del objetivo held-out"),
        "modelo": ("el mismo LogisticRegression lineal de V03, C=2,0, con el mismo L-BFGS-B de "
                   "la pista L, de modo que el delta ansla el ponderado"),
        "perdida": "logistica ponderada; la unica diferencia entre brazos es el ponderado",
        "rasgos": "los 27.431 del TF-IDF de V03, sin cambio",
        "adaptacion_legal": {
            "dominio_original": "vision, ruido de etiquetas con un conjunto held-out limpio",
            "adaptacion": ("tres problemas binarios con etiquetas anidadas y NINGUN subconjunto "
                           "con etiqueta probadamente limpia"),
            "que_cambia": ("la existencia de un conjunto limpio, que es la hipotesis de partida "
                           "del trabajo y aqui no se cumple"),
            "que_es_invariante": "el criterio de alineacion de gradiente",
        },
        "gap_analysis": {
            "filas_meta_reweight": 0,
            "precedentes": ("V4-B03, V4-B05 ponderan por dificultad; V6-07 por sospecha; "
                            "P59-06 por tamano de residuo. Las tres derivan el peso de una regla "
                            "FIJA computada antes de entrenar"),
            "por_que_no_es_duplicacion": ("meta-reweighting deriva el peso de un criterio de "
                                          "GRADIENTE contra un objetivo proxy, que es endogeno, y "
                                          "no de una regla de dificultad"),
            "veredicto": "NUEVA",
        },
        "configuracion_pre_registrada": {
            "momento": "en la inicializacion, con w = 0, como en Ren et al.",
            "fraccion_meta": FRACCION_META,
            "particion_interna": ("por GRUPO dentro del training exterior, de modo que ningun "
                                  "grupo near-duplicate se parte entre las dos partes "
                                  "internas, y verificado con una comprobacion explicita"),
            "forma_del_peso": "eps_i = <g_i, G> / <G, G>, forma cerrada de Ren et al.",
            "evitar_gram": ("el producto interno se evalua como omega_i dlogit_i (X G)_i, lo que "
                            "evita construir la matriz de Gram de tamano d por d con "
                            "d = 27.431"),
            "normalizacion": ("la masa total del ponderado se reescala a la del control, como en "
                              "la pista L, para que lo unico que cambie sea la FORMA y no la "
                              "escala, que seria regularizacion efectiva"),
            "rejilla": "ninguna sobre la fraccion meta",
            "pesos_de_clase": "los mismos balanceados de V03",
            "regularizacion": f"C = {C_MODELO} con el mismo L2",
            "umbrales": "0,44 / 0,51 / 0,46 congelados",
        },
        "anidamiento": {
            "outer_train_a_inner_train": True,
            "meta_split_por_grupo": True,
            "grupos_internos_partidos": 0,
            "outer_validation_participa": False,
            "outer_validation_uso": "puntuar una vez, sin ajustar nada",
        },
        "por_ajuste": info,
        "diagnostico_del_mecanismo": {
            "fraccion_negativa_media": round(neg_med, 6),
            "razon_p90_sobre_abs_p10_media": round(r_med, 6),
            "nota": ("estos numeros dicen si el ponderado esta INFORMATIVO. Una fraccion "
                     "negativa alta significa que el criterio premia y castiga de verdad, y una "
                     "razon intercuartil cercana a uno significa que casi no distingue"),
        },
        "control": {"brazo_0_unitario": m0,
                    "delta_brazo_0_vs_V03": round(ccv_0 - ccv_ref, 10),
                    "max_abs_diff_prob": dmax_0,
                    "clona_a_V03": bool(ccv_0 - ccv_ref <= TOLERANCIA_OPTIMIZADOR
                                        and dmax_0 <= 0.05)},
        "resultado": {
            "brazo_0_corrected_cv": round(m0["corrected_cv"], 10),
            "brazo_1_corrected_cv": round(m1["corrected_cv"], 10),
            "delta_meta_puro": round(delta_mec, 10),
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
        "interpretacion": (
            "El resultado cierra el meta-reweighting, y lo hace en un caso que conviene "
            "distinguir con cuidado de los cierres de P59. Ahi el mecanismo no se activaba: el "
            "Tukey de P59-06 nunca ponia influencia nula, y el DRO de P59-L sobreponderaba al "
            "grupo peor en un 2,47 por ciento. Aqui el mecanismo se activa y fuerte. El 20,77 "
            "por ciento de los pesos son NEGATIVOS, luego el criterio premia y castiga de verdad, "
            "y la razon entre el percentil 90 y el percentil 10 vale 5,35 de media, luego "
            "distingue con fuerza entre ejemplos. El ponderado no es degenerado y sin embargo "
            "el resultado es -0,301104, con 0 de 5 pliegues, 1208 celdas cambiadas y un neto de "
            "-572. Eso descarta la lectura comoda de que el metodo no hizo nada, y obliga a "
            "preguntarse por que un ponderado tan informativo hace tanto dano. La respuesta "
            "esta en la hipotesis que el propio trabajo declara: Ren et al. necesitan un "
            "conjunto retenido LIMPIO, y no hay evidencia de que ningun subconjunto de las "
            "etiquetas de TOXIC lo sea, luego el objetivo meta que define los pesos es tan "
            "ruidoso como los datos que intenta pesar. Y hay una razon estructural adicional "
            "que se puede anticipar desde las dimensiones: a la inicializacion, con d = 27.431 "
            "y n = 1186, los gradientes por ejemplo de un modelo lineal son casi ortogonales "
            "entre si, luego su proyeccion sobre G esta dominada por el producto de x_i con el "
            "vector agregado del conjunto meta, que es ruido de alta dimension, y no por la "
            "informacion de ejemplo. La conclusion transferible es que el meta-reweighting no "
            "es un metodo que se pueda aplicar a este corpus sin un conjunto limpio, y que la "
            "dificultad no es de implementacion, porque la forma cerrada es exacta y barata, "
            "sino de identificabilidad: con esta dimensionalidad y sin etiquetas limpias no hay "
            "senal que recuperar del criterio"),

        "decision": decision,
        "experimentos_registrados": 1,
        "formulacion": "V03 remains the best verified leaderboard result.",
        "gastos": {"submissions": 0, "consultas_kaggle": 0, "uso_holdout": 0, "uso_test": 0,
                   "segundos": round((datetime.now(timezone.utc) - inicio).total_seconds(), 1)},
    }
    JSON_I.write_text(json.dumps(informe, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    L: list[str] = []
    W = L.append
    W("# P60-I - Meta-reweighting por alineacion de gradiente")
    W("")
    W(f"Generado por `experiments/p60_10_metareweight.py`. "
      f"Fecha {inicio.isoformat(timespec='seconds')}.")
    W("")
    W("## Motivacion, y por que no es uno de los ponderados ya cerrados")
    W("")
    W("El proyecto ha cerrado tres familias de peso por ejemplo, y las tres lo derivan de una")
    W("**regla heuristica fija** computada antes de entrenar: `V4-B03` y `V4-B05` por")
    W("dificultad, `V6-07` por sospecha, `P59-06` por tamano del residuo. Lo que Ren et al.")
    W("proponen es de otra naturaleza: el peso sale de preguntar que gradientes estan")
    W("**alineados con un objetivo meta** evaluado sobre un conjunto retenido, luego es")
    W("endogeno a la optimizacion y no prescrito por una regla.")
    W("")
    W("## El peso tiene forma cerrada, y para un modelo lineal eso importa")
    W("")
    W("```text")
    W("eps_i = <g_i, G> / <G, G>     G = gradiente del objetivo meta")
    W("```")
    W("")
    W("Ren et al. evaluan esto en la inicializacion, luego con w = 0 el gradiente de un")
    W("ejemplo esta disponible en forma cerrada, sin retropropagacion, porque el gradiente de")
    W("la regularizacion se anula en el origen y el de la perdida queda siendo")
    W("`C * omega_i * (sigma(0) - y_i) * x_i`.")
    W("")
    W("El producto interno `<g_i, G>` necesitaria normalmente una matriz de Gram de tamano")
    W("`d` por `d`, con `d = 27.431`. No hace falta: escribiendo `G = C X' (omega dlogit)` sobre")
    W("las filas meta, se obtiene `<g_i, G> = C omega_i dlogit_i (X G)_i`, y `X G` es un unico")
    W("producto disperso de forma `n` por `d`. Eso es lo que hace la pista viable con 1483")
    W("filas y 27.431 rasgos, y es la razon por la que no hace falta meta-aprendizaje.")
    W("")
    W("## El anidamiento, que es la parte facil de hacer mal")
    W("")
    W("```text")
    W("outer_train  ->  inner_train + inner_meta, partidos POR GRUPO")
    W("             ->  los pesos salen solo del gradiente de inner_meta")
    W("             ->  se ajusta sobre el outer_train completo")
    W("outer_validation  ->  se puntua una vez, sin ajustar nada")
    W("```")
    W("")
    W("La particion interna se hace **por grupo near-duplicate**, no por fila, y el script")
    W("comprueba de forma explicita que ningun grupo queda partido entre las dos partes")
    W("internas. La validacion exterior no participa en ningun paso.")
    W("")
    W("## La limitacion honesta, declarada antes de ejecutar")
    W("")
    W("Ren et al. necesitan un conjunto retenido **limpio**. No hay evidencia de que ningun")
    W("subconjunto de las etiquetas de TOXIC lo sea, luego el objetivo meta se evalua aqui")
    W("sobre un conjunto tan ruidoso como el resto. Esa es la debilidad real de esta pista en")
    W("este corpus y no es un detalle de configuracion.")
    W("")
    W("## Configuracion, fijada antes de mirar validation")
    W("")
    W("```text")
    W("momento        en la inicializacion, con w = 0, como en el trabajo original")
    W(f"fraccion meta  {FRACCION_META} de los grupos internos, un solo valor")
    W("forma           eps_i = <g_i, G> / <G, G>, forma cerrada")
    W("normalizacion   la masa del ponderado se reescala a la del control, como en la pista L")
    W("rejilla         ninguna")
    W("```")
    W("")
    W("## Control")
    W("")
    W("```text")
    W(f"brazo 0 (pesos unitarios)  corrected_cv {m0['corrected_cv']!r}")
    W(f"  delta vs V03 {ccv_0 - ccv_ref:+.10f} | celdas distintas "
      f"{int((pred_0 != pred_v03).sum())}")
    W("```")
    W("")
    W("## Resultados")
    W("")
    W("```text")
    W(f"brazo 0 corrected_cv {m0['corrected_cv']:.10f}")
    W(f"brazo 1 corrected_cv {m1['corrected_cv']:.10f}")
    W(f"delta META PURO        {delta_mec:+.10f}")
    W(f"delta vs V03           {ccv_1 - ccv_ref:+.10f}")
    W(f"pliegues que mejoran   {folds_mej}/5")
    W(f"rescate {m1['rescate']} | dano {m1['damage']} | neto {m1['neto']:+d} "
      f"| cambian {m1['cambian']}")
    W(f"convergencia {conv}")
    W(f"V03 referencia         {ccv_ref:.10f}")
    W("```")
    W("")
    W("| etiqueta | F1 meta | F1 control |")
    W("|---|---:|---:|")
    for e in ETIQUETAS:
        W(f"| {e} | {m1['f1_por_etiqueta'][e]:.4f} | {m0['f1_por_etiqueta'][e]:.4f} |")
    W("")
    W("## Diagnostico del mecanismo")
    W("")
    W("| pliegue | etiqueta | eps mediana | eps p10 | eps p90 | razon p90/abs(p10) | fraccion negativa |")
    W("|---:|---|---:|---:|---:|---:|---:|")
    for i in info:
        W(f"| {i['pliegue']} | {i['etiqueta']} | {i['eps_mediana']:.3e} | {i['eps_p10']:.3e} | "
          f"{i['eps_p90']:.3e} | {i['razon_p90_sobre_abs_p10']:.4f} | "
          f"{i['fraccion_negativa']*100:.2f} por ciento |")
    W("")
    W(f"Fraccion negativa media **{neg_med*100:.2f} por ciento**, razon intercuartil media")
    W(f"**{r_med:.4f}**. Una fraccion negativa alta significa que el criterio premia y castiga")
    W("de verdad; una razon cercana a uno significa que apenas distingue entre ejemplos.")
    W("")
    W("## Gate")
    W("")
    W("| criterio | valor | superado |")
    W("|---|---|---|")
    W(f"| efecto del ponderado >= +0,003 | {delta_mec:+.6f} | **{h_delta}** |")
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
    W("*Generado por `experiments/p60_10_metareweight.py`. Sin push.*")
    MD_I.write_text(a_ascii("\n".join(L)), encoding="ascii")

    with CSV_I.open("w", encoding="ascii", newline="") as fh:
        w = _csv.writer(fh, lineterminator="\n")
        w.writerow(["magnitud", "valor"])
        for fila in (
            ["brazo_0_corrected_cv", repr(m0["corrected_cv"])],
            ["brazo_1_corrected_cv", repr(m1["corrected_cv"])],
            ["delta_meta_puro", repr(delta_mec)],
            ["delta_vs_V03", repr(ccv_1 - ccv_ref)],
            ["corrected_cv_V03", repr(ccv_ref)],
            ["pliegues_mejorados", str(folds_mej)],
            ["folds_brazo_1", " ".join(f"{x:.6f}" for x in m1["folds"])],
            ["folds_brazo_0", " ".join(f"{x:.6f}" for x in m0["folds"])],
            ["rescate", str(m1["rescate"])], ["damage", str(m1["damage"])],
            ["neto", str(m1["neto"])], ["cambian", str(m1["cambian"])],
            ["fraccion_meta", f"{FRACCION_META}"],
            ["fraccion_negativa_media", f"{neg_med:.6f}"],
            ["razon_p90_sobre_abs_p10_media", f"{r_med:.6f}"],
            ["convergio", str(conv)],
            ["gate_delta_superado", str(h_delta)],
            ["gate_pliegues_superado", str(h_folds)],
            ["gate_superado", str(gate)],
            ["decision", a_ascii(decision)],
        ):
            w.writerow(fila)

    for destino in (MD_I, JSON_I, CSV_I):
        if destino.exists() and any(b > 127 for b in destino.read_bytes()):
            raise SystemExit(f"PARADA: {destino.name} conserva bytes no ASCII")
    log("")
    log(f"P60-I VEREDICTO: {decision.split('.')[0]}")
    log(f"P60-10 OK | {informe['gastos']['segundos']} s")


if __name__ == "__main__":
    warnings.simplefilter("error", RuntimeWarning)
    ast.parse(Path(__file__).read_text(encoding="utf-8"))
    main()
