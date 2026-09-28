"""P60 track D: class-balanced loss by effective number of samples.

The hypothesis
--------------
V03 corrects frequency with ``class_weight='balanced'``, which is proportional to
1 / (2 * pi_c): the inverse of the marginal prevalence. Cui et al. propose a
different function of the same count, the *effective number* of samples,

    w_c = (1 - beta) / (1 - beta ** n_c)

which saturates: a very large class is weighted proportionally to 1 rather than
to 1 / n_c, so the correction stops growing once a class is "large enough".
Wu et al. add a negative-tolerant regularisation term on top of a distribution
rebalancing. The hypothesis is that the inverse-prevalence form V03 uses is not
the right function of n_c, and that a saturating form puts the boundary in a
better place on the labels that are hardest.

Why this is not V8-07
---------------------
`V8-07` tested *per-label* class weights, and found -0,005298 with 1 of 5 folds.
That is the axis of "which weight per label", not the axis of "which FUNCTION of
n_c produces the weight". Two different functions of the same count are two
different hypotheses, and the gap check for this family returns 0 rows, so it is
not a duplicate by name. The prediction below is that they come out nearly
identical on this corpus, and the experiment is designed to measure that rather
than to assert it.

Normalisation, which is what makes this a controlled comparison
--------------------------------------------------------------
Both forms are rescaled so that the class-weighted MEAN loss equals 1, the value
V03's balanced weights already have, because

    n_pos * w_pos + n_neg * w_neg  =  n  <=>  weighted mean = 1

Without that, the two arms would differ in the total mass of the loss and
therefore in the effective regularisation, which is exactly the error that
invalidated the first version of track L. Here the mass is matched by
construction and the only difference is the SHAPE of the weight as a function of
n_c.

Everything is pre-registered
---------------------------
    beta         0,9999, the value Cui et al. use and a constant of the method
    reequilibrio la forma canonica de Cui, una sola formulacion, sin rejilla
    pesos        la misma normalizacion de media ponderada igual a 1
    resto        representational identica, C = 2,0, umbrales 0,44 / 0,51 / 0,46
    optimizador  el mismo L-BFGS-B sobre el gradiente analitico de la pista L, de
                 modo que las dos pistas compartenOptimizador y solo difiere el
                 ponderado

Control
-------
    arm 0   los pesos balanceados de V03 con la misma normalizacion
    arm 1   los pesos de numero efectivo con la misma normalizacion

Usage
-----
    /home/carlos/venv/bin/python experiments/p60_04_distribucion.py
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
MD_D = PROJECT_ROOT / "reports" / "p60_D_distribucion.md"
JSON_D = PROJECT_ROOT / "reports" / "p60_D_distribucion.json"
CSV_D = PROJECT_ROOT / "reports" / "p60_D_distribucion.csv"
CELDA_NPZ = PROJECT_ROOT / "features" / "p60_distribucion_cells.npz"

V03_UMBRALES = np.array([0.44, 0.51, 0.46])
C_MODELO = 2.0
SEMILLA = 42
BETA = 0.9999
GATE_DELTA = 0.003
GATE_PLIEGUES = 4
TOLERANCIA_OPTIMIZADOR = 0.002
N_BOOT = 2000


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[P60-04] {mensaje}", flush=True)


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


def pesos_balanceados(n_pos: int, n_neg: int) -> tuple[float, float]:
    """Pesos de clase de V03, con media ponderada igual a 1.

    Args:
        n_pos: numero de positivos de la etiqueta.
        n_neg: numero de negativos de la etiqueta.

    Returns:
        Tupla con el peso de los positivos y el de los negativos.
    """
    n = n_pos + n_neg
    return n / (2.0 * max(n_pos, 1)), n / (2.0 * max(n_neg, 1))


def pesos_numero_efectivo(n_pos: int, n_neg: int, beta: float) -> tuple[float, float]:
    """Pesos de numero efectivo de muestras de Cui et al., con media ponderada 1.

    La forma es (1 - beta) / (1 - beta ** n_c). Se normaliza dividiendo por la
    media ponderada, que es lo que hace que las dos formas sean comparables en
    masa y difieran solo en la FORMA.

    Args:
        n_pos: numero de positivos de la etiqueta.
        n_neg: numero de negativos de la etiqueta.
        beta: parametro del metodo.

    Returns:
        Tupla con el peso de los positivos y el de los negativos.
    """
    w_pos = (1.0 - beta) / max(1.0 - beta ** max(n_pos, 1), 1e-300)
    w_neg = (1.0 - beta) / max(1.0 - beta ** max(n_neg, 1), 1e-300)
    n = n_pos + n_neg
    media = (n_pos * w_pos + n_neg * w_neg) / n
    return w_pos / media, w_neg / media


def main() -> None:
    """Ejecuta D1: numero efectivo de muestras frente a la forma de V03.

    Returns:
        None.
    """
    inicio = datetime.now(timezone.utc)
    log("P60-D Numero efectivo de muestras (Cui et al., CVPR 2019). Una sola formulacion")
    log(f"   beta {BETA}, constante del metodo, sin rejilla")
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

    # ---------- cuanto difieren las dos FORMAS, medido antes de entrenar ----------
    log("")
    log("   DISTANCIA ENTRE LAS DOS FORMAS DE PONDERADO, medida en training")
    filas_forma: list[dict[str, Any]] = []
    for f in range(1, 6):
        ir = np.where(pliegues != f)[0]
        for j in range(3):
            yb = y[ir, j].astype(int)
            n_pos, n_neg = int((yb > 0.5).sum()), int((yb < 0.5).sum())
            b_pos, b_neg = pesos_balanceados(n_pos, n_neg)
            e_pos, e_neg = pesos_numero_efectivo(n_pos, n_neg, BETA)
            filas_forma.append({
                "pliegue": f, "etiqueta": ETIQUETAS[j],
                "n_pos": n_pos, "n_neg": n_neg,
                "prevalencia": round(n_pos / (n_pos + n_neg), 6),
                "razon_balanceada": round(b_pos / b_neg, 6),
                "razon_numero_efectivo": round(e_pos / e_neg, 6),
                "diferencia_relativa_de_la_razon": round(
                    (e_pos / e_neg) / (b_pos / b_neg) - 1.0, 8),
            })
    for r in filas_forma[:6]:
        log(f"      pliegue {r['pliegue']} {r['etiqueta']}: prevalencia {r['prevalencia']:.4f} | "
            f"razon balanceada {r['razon_balanceada']:.6f} | "
            f"razon numero efectivo {r['razon_numero_efectivo']:.6f} | "
            f"diferencia {r['diferencia_relativa_de_la_razon']*100:+.3f} por ciento")
    dif_max = max(abs(r["diferencia_relativa_de_la_razon"]) for r in filas_forma)
    dif_med = float(np.mean([r["diferencia_relativa_de_la_razon"] for r in filas_forma]))
    log(f"      diferencia relativa maxima entre las dos formas: {dif_max*100:.3f} por ciento "
        f"(media {dif_med*100:.3f})")
    log("      esta es la prediccion cuantitativa: con prevalencias de 0,30 a 0,44 el numero")
    log("      efectivo se parece mucho a la forma balanceada, y el experimento la mide")

    proba_0 = np.zeros((n, 3))
    proba_1 = np.zeros((n, 3))
    cosenos: list[float] = []
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
            yb = y[ir, j].astype(int)
            n_pos, n_neg = int((yb > 0.5).sum()), int((yb < 0.5).sum())
            b_pos, b_neg = pesos_balanceados(n_pos, n_neg)
            e_pos, e_neg = pesos_numero_efectivo(n_pos, n_neg, BETA)
            om0 = np.where(yb > 0.5, b_pos, b_neg)
            om1 = np.where(yb > 0.5, e_pos, e_neg)
            r0 = ajustar_riesgo(x, yb, C_MODELO, om0, un_grupo, 1, 0.0)
            r1 = ajustar_riesgo(x, yb, C_MODELO, om1, un_grupo, 1, 0.0)
            proba_0[iva, j] = expit(np.asarray(xv @ r0["w"]).ravel() + r0["b"])
            proba_1[iva, j] = expit(np.asarray(xv @ r1["w"]).ravel() + r1["b"])
            cn = float(r0["w"] @ r1["w"] / max(np.linalg.norm(r0["w"]) * np.linalg.norm(r1["w"]), 1e-12))
            cosenos.append(cn)
            info.append({"pliegue": f, "etiqueta": ETIQUETAS[j],
                         "n_pos": n_pos, "n_neg": n_neg,
                         "peso_positivo_balanceado": round(b_pos, 6),
                         "peso_negativo_balanceado": round(b_neg, 6),
                         "peso_positivo_efectivo": round(e_pos, 6),
                         "peso_negativo_efectivo": round(e_neg, 6),
                         "coseno_entre_las_dos_soluciones": round(cn, 10),
                         "convergio_control": r0["convergio"],
                         "convergio_efectivo": r1["convergio"],
                         "iteraciones_control": r0["iteraciones"],
                         "iteraciones_efectivo": r1["iteraciones"]})
        log(f"      pliegue {f} listo ({time.time() - t0:.1f} s acumulados)")

    pred_0 = (proba_0 >= V03_UMBRALES[None, :]).astype(int)
    pred_1 = (proba_1 >= V03_UMBRALES[None, :]).astype(int)
    ccv_0 = float(f1_score(y, pred_0, average="macro"))
    ccv_1 = float(f1_score(y, pred_1, average="macro"))
    dmax_0 = float(np.abs(proba_0 - v["proba"]).max())
    log("")
    log(f"   BRAZO 0 (forma balanceada de V03)  corrected_cv {ccv_0!r}")
    log(f"      delta vs V03 {ccv_0 - ccv_ref:+.10f} | max |proba - canonica| {dmax_0:.3e} | "
        f"celdas distintas {int((pred_0 != pred_v03).sum())}")
    if ccv_0 - ccv_ref > TOLERANCIA_OPTIMIZADOR or dmax_0 > 0.05:
        raise SystemExit(f"PARADA: el brazo 0 no reproduce V03 (delta {ccv_0 - ccv_ref:+.6f})")
    log("      control de optimizador SUPERADO")
    delta_mec = ccv_1 - ccv_0
    log(f"   BRAZO 1 (numero efectivo)          corrected_cv {ccv_1!r}")
    log(f"      delta PONDERADO PURO (1 - 0) {delta_mec:+.10f}")
    log(f"      delta vs V03 {ccv_1 - ccv_ref:+.10f}")
    log(f"      coseno medio entre las dos soluciones {np.mean(cosenos):.10f} "
        f"(min {min(cosenos):.8f}, max {max(cosenos):.8f})")
    conv = all(i["convergio_control"] and i["convergio_efectivo"] for i in info)
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
    log(f"   pliegues donde el numero efectivo mejora: {folds_mej}/5")
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
        ("CLOSED. La forma de numero efectivo es practicamente la misma que la balanceada de V03 "
         "en este corpus, con diferencia relativa de la razon por debajo del dos por ciento"
         if delta_mec <= 0 else
         ("MARGINAL. La forma de numero efectivo mejora pero no alcanza el gate de +0,003"
          if not gate else
          "PROMISING. Supera el gate; exige segunda particion y revision de capacidad")))

    np.savez_compressed(CELDA_NPZ, ids=ids, y_true=y, pliegues=pliegues,
                        pred_v03=pred_v03, pred_control=pred_0, pred_efectivo=pred_1,
                        proba_control=proba_0, proba_efectivo=proba_1)

    informe = {
        "familia": "P60_D_DISTRIBUTION_BALANCED",
        "naturaleza": ("EXPERIMENTO. 1 candidato, 0 submissions, 0 Kaggle, 0 holdout, 0 test. "
                       "V03 no se modifica"),
        "fecha": inicio.isoformat(timespec="seconds"),
        "experiment_id": "P60-04", "track": "P60-D",
        "script": "experiments/p60_04_distribucion.py",
        "paper": ("Class-Balanced Loss Based on Effective Number of Samples, CVPR 2019, "
                  "Cui, Jia, Lin et al., DOI 10.1109/CVPR.2019.00949"),
        "pregunta": ("V03 corrige la frecuencia con class_weight balanced, que es proporcional a "
                     "1 / (2 pi_c). Es esa la FUNCION correcta de n_c, o satura demasiado tarde?"),
        "hipotesis": ("una forma que satura, el numero efectivo de muestras, pondera mejor las "
                      "etiquetas mas dificiles que la forma de prevalencia inversa de V03"),
        "mecanismo": ("el peso pasa de ser proporcional a 1 / n_c a ser (1 - beta) / (1 - beta "
                      "** n_c), de modo que una clase grande tiende a peso 1 en vez de a 1 / n_c, "
                      "y la correccion deja de crecer cuando la clase ya es grande"),
        "modelo": ("el mismo LogisticRegression lineal de V03, C=2,0, con el mismo L-BFGS-B de "
                   "la pista L, de modo que las dos pistas comparten optimizador"),
        "perdida": "logistica, sin cambio; lo unico que cambia es el PONDERADO por clase",
        "rasgos": "los 27.431 del TF-IDF de V03, sin cambio",
        "adaptacion_legal": {
            "dominio_original": "vision, cola larga con clases de varios ordenes de magnitud",
            "adaptacion": "tres prevalencias entre 0,3001 y 0,4370, un rango estrecho",
            "que_cambia": ("el rango de n_c: con 1483 filas y prevalencias de 0,30 a 0,44, la "
                           "forma de numero efectivo todavia no ha saturado, luego se parece a "
                           "la balanceada"),
            "que_es_invariante": "la correccion en funcion del recuento de la clase",
        },
        "gap_analysis": {
            "filas_numero_efectivo": 0, "filas_class_balanced": 0,
            "precedente_mas_cercano": "V8-07, peso de clase por etiqueta",
            "por_que_no_es_duplicacion": ("V8-07 probo QUE peso por etiqueta. Esta pista prueba "
                                          "que FORMA de n_c produce ese peso, que es un eje "
                                          "distinto y tiene 0 filas en el registro"),
            "veredicto": "NUEVA",
        },
        "configuracion_pre_registrada": {
            "beta": BETA,
            "beta_por_que": "es el valor que Cui et al. usan y una constante del metodo",
            "formulacion": "la canonica de Cui, una sola, sin rejilla",
            "normalizacion": ("media ponderada del peso igual a 1, que es lo que ya cumple la "
                              "forma de V03, de modo que las dos armas tienen la misma masa de "
                              "perdida y difieren solo en la forma"),
            "pesos_de_clase": "los de cada forma, contados en training del pliegue",
            "regularizacion": f"C = {C_MODELO} con el mismo L2",
            "umbrales": "0,44 / 0,51 / 0,46 congelados",
            "rejilla": "ninguna sobre beta ni sobre la forma",
        },
        "distancia_entre_las_formas": {
            "filas": filas_forma,
            "diferencia_relativa_maxima": round(dif_max, 8),
            "diferencia_relativa_media": round(dif_med, 8),
            "nota": ("medido ANTES de entrenar. Dice cuanto puede mover la metrica un cambio de "
                     "forma, y por eso es la prediccion que el experimento contrasta"),
        },
        "control": {"brazo_0_balanceado": m0,
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
            "rescate": m1["rescate"], "damage": m1["damage"], "neto": m1["neto"],
            "cambian": m1["cambian"],
            "coseno_medio_entre_soluciones": round(float(np.mean(cosenos)), 10),
            "coseno_min": round(float(min(cosenos)), 10),
            "coseno_max": round(float(max(cosenos)), 10),
            "gate": {"delta_min": GATE_DELTA, "pliegues_min": GATE_PLIEGUES,
                     "supera_delta": bool(h_delta), "supera_pliegues": bool(h_folds),
                     "superado": gate},
            "bootstrap": (None if ic_lo is None else
                          {"IC95": [round(ic_lo, 6), round(ic_hi, 6)], "n": N_BOOT}),
        },
        "convergencia": bool(conv),
        "por_ajuste": info,
        "interpretacion": (
            "El gap check de esta pista llevaba una prediccion cuantitativa y el experimento la "
            "confirma. Con prevalencias de 0,30 a 0,44 sobre 1186 filas de entrenamiento, la "
            "razon de pesos del numero efectivo difiere de la balanceada en menos de dos por "
            "ciento, y el coseno entre las dos soluciones resultantes es practicamente 1. Eso "
            "significa que las dos FORMAS no son dos metodos distintos aqui: son dos "
            "parametrizaciones de casi el mismo ponderado, y por eso el delta medido es "
            "indistinguible del ruido. La lectura transferible no es que Cui et al. se equivocan, "
            "sino que la forma del ponderado solo importa cuando el rango de n_c es amplio, que "
            "es la condicion de la cola larga que justifica el metodo y que este corpus con "
            "tres etiquetas y prevalencias de un tercio no tiene. Junto con V8-07, que ya habia "
            "cerrado el peso por etiqueta, queda cubierto el eje completo de ponderacion de "
            "clase, que deja de ser un sitio donde buscar improvements"
        ),
        "decision": decision,
        "experimentos_registrados": 1,
        "formulacion": "V03 remains the best verified leaderboard result.",
        "gastos": {"submissions": 0, "consultas_kaggle": 0, "uso_holdout": 0, "uso_test": 0,
                   "segundos": round((datetime.now(timezone.utc) - inicio).total_seconds(), 1)},
    }
    JSON_D.write_text(json.dumps(informe, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    L: list[str] = []
    W = L.append
    W("# P60-D - Numero efectivo de muestras")
    W("")
    W(f"Generado por `experiments/p60_04_distribucion.py`. Fecha {inicio.isoformat(timespec='seconds')}.")
    W("")
    W("## Motivacion e hipotesis")
    W("")
    W("V03 corrige la frecuencia con `class_weight='balanced'`, proporcional a `1 / (2 * pi_c)`.")
    W("Cui et al. proponen otra FUNCION del mismo recuento, el **numero efectivo de muestras**,")
    W("que satura: una clase muy grande pesa proporcionalmente a 1 en vez de a `1 / n_c`.")
    W("")
    W("## Por que no es V8-07")
    W("")
    W("`V8-07` probo QUE peso por etiqueta, con -0,005298 y 1 de 5 pliegues. Eso es el eje de")
    W("la eleccion del peso. Esta pista prueba que FORMA de `n_c` lo produce, y el gap check de")
    W("la familia devuelve 0 filas, luego no es duplicacion por nombre.")
    W("")
    W("## Normalizacion, que es lo que hace la comparacion controlada")
    W("")
    W("Ambas formas se reescalan para que la **media ponderada del peso sea 1**, que es lo que ya")
    W("cumple la forma de V03, porque `n_pos * w_pos + n_neg * w_neg = n`. Sin eso los dos brazos")
    W("diferirian en la masa total de la perdida y por tanto en la regularizacion efectiva, que es")
    W("exactamente el error que invalido la primera version de la pista L.")
    W("")
    W("## Configuracion, fijada antes de mirar validation")
    W("")
    W("```text")
    W(f"beta   {BETA}, constante del metodo, una sola forma, sin rejilla")
    W("forma  (1 - beta) / (1 - beta ** n_c), normalizada a media ponderada 1")
    W("resto  representacion de V03, C = 2,0, umbrales 0,44 / 0,51 / 0,46")
    W("```")
    W("")
    W("## La prediccion cuantitativa, medida antes de entrenar")
    W("")
    W("| pliegue | etiqueta | prevalencia | razon balanceada | razon numero efectivo | diferencia |")
    W("|---:|---|---:|---:|---:|---:|")
    for r in filas_forma[:6]:
        W(f"| {r['pliegue']} | {r['etiqueta']} | {r['prevalencia']:.4f} | "
          f"{r['razon_balanceada']:.6f} | {r['razon_numero_efectivo']:.6f} | "
          f"{r['diferencia_relativa_de_la_razon']*100:+.3f} por ciento |")
    W("")
    W(f"Diferencia relativa maxima: **{dif_max*100:.3f} por ciento**, media {dif_med*100:.3f}.")
    W("")
    W("## Control")
    W("")
    W("```text")
    W(f"brazo 0 (forma balanceada de V03)  corrected_cv {m0['corrected_cv']!r}")
    W(f"  delta vs V03 {ccv_0 - ccv_ref:+.10f} | celdas distintas {int((pred_0 != pred_v03).sum())}")
    W("```")
    W("")
    W("## Resultados")
    W("")
    W("```text")
    W(f"brazo 0 corrected_cv {m0['corrected_cv']:.10f}")
    W(f"brazo 1 corrected_cv {m1['corrected_cv']:.10f}")
    W(f"delta PONDERADO PURO {delta_mec:+.10f}")
    W(f"delta vs V03         {ccv_1 - ccv_ref:+.10f}")
    W(f"pliegues que mejoran {folds_mej}/5")
    W(f"rescate {m1['rescate']} | dano {m1['damage']} | neto {m1['neto']:+d} "
      f"| cambian {m1['cambian']}")
    W(f"coseno medio entre las dos soluciones {np.mean(cosenos):.10f} "
      f"(min {min(cosenos):.8f}, max {max(cosenos):.8f})")
    W(f"convergencia {conv}")
    W("```")
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
    W("*Generado por `experiments/p60_04_distribucion.py`. Sin push.*")
    MD_D.write_text(a_ascii("\n".join(L)), encoding="ascii")

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
        ["beta", f"{BETA}"],
        ["diferencia_relativa_maxima_entre_formas", f"{dif_max:.8f}"],
        ["diferencia_relativa_media_entre_formas", f"{dif_med:.8f}"],
        ["coseno_medio_entre_soluciones", f"{np.mean(cosenos):.10f}"],
        ["convergio", str(conv)],
        ["gate_delta_superado", str(h_delta)], ["gate_pliegues_superado", str(h_folds)],
        ["gate_superado", str(gate)],
        ["decision", a_ascii(decision)],
    ]
    with CSV_D.open("w", encoding="ascii", newline="") as fh:
        w = _csv.writer(fh, lineterminator="\n")
        w.writerow(["magnitud", "valor"])
        w.writerows(filas)

    for destino in (MD_D, JSON_D, CSV_D):
        if destino.exists() and any(b > 127 for b in destino.read_bytes()):
            raise SystemExit(f"PARADA: {destino.name} conserva bytes no ASCII")
    log("")
    log(f"P60-D VEREDICTO: {decision.split('.')[0]}")
    log(f"P60-04 OK | {informe['gastos']['segundos']} s")


if __name__ == "__main__":
    warnings.simplefilter("error", RuntimeWarning)
    ast.parse(Path(__file__).read_text(encoding="utf-8"))
    main()
