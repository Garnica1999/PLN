"""P64 track P8: does the evidence block work ONLY on the long tweets?

Why this question is separate from P64-03 and P64-04
----------------------------------------------------
Both of those blocks failed globally, and both failed against the margin of V03. That is a
statement about the average over all 1483 rows, and the average is dominated by the short ones:
CORTA is 167 rows and MUY_LARGA is 167, so a global number says almost nothing about either
end.

P63 measured where the residual margin actually lives, and the answer was specific:

    CORTA      margin to the threshold ceiling  +0,0000
    MEDIA      +0,0093
    LARGA      +0,0217
    MUY_LARGA  +0,0890

In the very long tweets there is nearly nine hundredths of macro F1 sitting on the table, and
it is on the table in a way that no threshold can reach. If the evidence block carries the
structure that the bag of n-grams flattens, then that is where it should pay, and a global
average would be exactly the statistic designed to hide it.

What is measured, and what is NOT done
--------------------------------------
The bin cuts are the ones frozen in P63 and inherited here unchanged. The PRIMARY metric stays
the global corrected_cv, because adopting something that only helps one stratum is how a
project talks itself into a specialisation that does not transfer. The per-bin numbers are
reported so the reader can see the shape, and the decision is NOT made on the best bin.

The third arm is the one that has to be beaten
---------------------------------------------
V03_margen, again. The reason is now sharp rather than ritual. A block that helps the long
tweets has to beat not V03 but the confidence V03 already exposes on those same tweets, because
if the margin already carries the information then any per-bin gain is bookkeeping. This is
the control that closed the composition family in P63 and the evidence family in P64-03 and
P64-04, and it is the one that decides here.

The two mechanisms that are tested, and they are different
---------------------------------------------------------
    G1 concentrated    the maximum and the top-2 mean of the segment evidence. This is the
                       natural reading of "one segment decides", and it is deliberately NOT
                       the max-pooling that P59-07 used as a decision rule: here the max is a
                       FEATURE that a linear model weighs, not an aggregator that replaces the
                       prediction, so the upward bias that sank P59 does not apply.

    G2 conflict        the signed leave-one-out profile, which measures where the decision
                       rests and whether the pieces contradict each other.

Neither is a grid. One configuration each, declared before execution, both already computed
in the previous two scripts and reused rather than recomputed, so this script costs almost
nothing and introduces no new model.

What a positive here would mean, stated in advance
--------------------------------------------------
If G1 or G2 improves the MUY_LARGA stratum by a clear margin against BOTH V03 and V03_margen
in that stratum, the honest next step is not adoption. It is a candidate that has to pass the
interest gate globally, the bootstrap, and the second grouped partition, and the global gate is
the one that binds. A stratum win is evidence about where the information is, not a licence to
ship.

Usage
-----
    /home/carlos/venv/bin/python experiments/p64_05_especializacion_larga.py
"""

from __future__ import annotations

import ast
import csv as _csv
import hashlib
import json
import subprocess
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
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
)

REPORTS = PROJECT_ROOT / "reports"
OOF = PROJECT_ROOT / "features" / "oof_v03_final.npz"
RAW_TRAIN = PROJECT_ROOT / "data" / "raw" / "toxic_train.csv"
CELDA_EV = PROJECT_ROOT / "features" / "p64_evidencia_cells.npz"
CELDA_LOO = PROJECT_ROOT / "features" / "p64_loo_cells.npz"
MD_S = REPORTS / "p64_especializacion_larga.md"
JSON_S = REPORTS / "p64_especializacion_larga.json"
CSV_S = REPORTS / "p64_candidate_comparison.csv"

V03_UMBRALES = np.array([0.44, 0.51, 0.46])
PARAMS_WORD = {"analyzer": "word", "ngram_range": (1, 2), "min_df": 2, "sublinear_tf": True,
               "lowercase": True}
PARAMS_CHAR = {"analyzer": "char_wb", "ngram_range": (3, 5), "min_df": 2,
               "sublinear_tf": True, "lowercase": True}
LR = {"C": 2.0, "class_weight": "balanced", "max_iter": 1000, "solver": "liblinear",
      "random_state": 42}
GATE_DELTA = 0.0015
GATE_PLIEGUES = 4
SEMILLA = 42
# Cortes CONGELADOS de P63. No se ajustan.
BINS: tuple[tuple[str, int, int], ...] = (
    ("CORTA", 0, 6),
    ("MEDIA", 7, 15),
    ("LARGA", 16, 30),
    ("MUY_LARGA", 31, 10**9),
)
# Las dos columnas que forman cada bloque. Se declaran antes y no se varian.
COLS_G1 = ["max", "top2_media", "top3_fraccion", "margen_max"]
COLS_G2 = ["concentracion", "conflicto", "cambia_de_signo", "entropia"]


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[P64-05] {mensaje}", flush=True)


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


def sub_bloque(bloque: np.ndarray, nombres: np.ndarray, prefijo_rasgo: str,
               columnas: list[str]) -> np.ndarray:
    """Extrae un subconjunto de columnas ya calculadas.

    Los nombres reales del bloque son del tipo HS_seg_max y HS_loo_concentracion, luego el
    objetivo se construye con el nombre completo del rasgo.

    Args:
        bloque: matriz de rasgos completa.
        nombres: nombres de las columnas.
        prefijo_rasgo: prefijo del rasgo, seg o loo.
        columnas: nombres de rasgo que se extraen.

    Returns:
        Matriz con una etiqueta y una columna por combinacion.
    """
    cols: list[int] = []
    for et in ETIQUETAS:
        for c in columnas:
            objetivo = f"{et}_{prefijo_rasgo}_{c}"
            coincide = np.where(nombres == objetivo)[0]
            if coincide.size == 0:
                raise SystemExit(f"PARADA: el bloque no tiene la columna {objetivo}")
            cols.append(int(coincide[0]))
    return bloque[:, cols]


def main() -> None:
    """Evalua los dos bloques por estrato de longitud, con el margen como control.

    Returns:
        None.
    """
    inicio = datetime.now(timezone.utc)
    log("P64-05 los bloques de evidencia, por estrato, y si el margen ya lo sabia")
    log("   cortes CONGELADOS de P63 | metrica PRIMARIA global, las rebanadas se informan")
    log(f"   G1 concentracion: {COLS_G1}")
    log(f"   G2 conflicto:     {COLS_G2}")

    hash_real = hashlib.sha256(V03_SUB.read_bytes()).hexdigest()
    if hash_real != V03_SHA:
        raise SystemExit("PARADA: la submission de V03 no coincide con el sha256 congelado")
    v = np.load(OOF, allow_pickle=True)
    ids, y, pliegues = v["ids"], v["y_true"].astype(int), v["pliegues"]
    pred_v03 = v["pred"].astype(int)
    margen_v03 = v["margen"]
    n = ids.size
    ccv_ref = float(f1_score(y, pred_v03, average="macro"))
    if ccv_ref != V03_CCV:
        raise SystemExit(f"PARADA: el OOF canonico da {ccv_ref!r}, no {V03_CCV!r}")

    ev = np.load(CELDA_EV, allow_pickle=True)
    loo = np.load(CELDA_LOO, allow_pickle=True)
    if not np.array_equal(ev["ids"], ids) or not np.array_equal(loo["ids"], ids):
        raise SystemExit("PARADA: los bloques de evidencia no corresponden a las filas del OOF")
    n_tok = ev["n_tokens"]
    if not np.array_equal(n_tok, loo["n_tokens"]):
        raise SystemExit("PARADA: los dos bloques de evidencia ven longitudes distintas")
    log("")
    log(f"   OOF canonico {ccv_ref!r} | {n} filas | bloques de P64-03 y P64-04 reutilizados")

    G1 = sub_bloque(ev["bloque"], ev["nombres"], "seg", COLS_G1)
    G2 = sub_bloque(loo["bloque"], loo["nombres"], "loo", COLS_G2)
    nombres_g1 = [f"{et}_seg_{c}" for et in ETIQUETAS for c in COLS_G1]
    nombres_g2 = [f"{et}_loo_{c}" for et in ETIQUETAS for c in COLS_G2]
    log(f"   G1: {G1.shape[1]} columnas | G2: {G2.shape[1]} columnas")

    tr = pd.read_csv(RAW_TRAIN, sep=";", encoding="latin-1", dtype=str)
    tr["idn"] = pd.to_numeric(tr["id"].str.replace('"', "", regex=False),
                              errors="coerce").astype("int64")
    textos = tr.set_index("idn").loc[ids, "text"].astype(str).tolist()

    proba: dict[str, np.ndarray] = {b: np.zeros((n, 3)) for b in
                                   ("V03_G1", "V03_G2", "V03_ambos", "V03_random", "V03_margen")}
    for f in range(1, 6):
        ir, iva = np.where(pliegues != f)[0], np.where(pliegues == f)[0]
        pw = TfidfVectorizer(**PARAMS_WORD)
        pc = TfidfVectorizer(**PARAMS_CHAR)
        xw = pw.fit_transform([textos[i] for i in ir])
        xc = pc.fit_transform([textos[i] for i in ir])
        xbase = hstack([xw, xc]).tocsr()
        xbase_v = hstack([pw.transform([textos[i] for i in iva]),
                          pc.transform([textos[i] for i in iva])]).tocsr()
        objetivo = float(np.sqrt(np.mean(xbase.multiply(xbase).sum(axis=1))))

        def escalar(mm: csr_matrix, ref: float = objetivo) -> csr_matrix:
            """Iguala la norma media de fila a la del control.

            Args:
                mm: matriz dispersa.
                ref: norma objetivo.

            Returns:
                La matriz escalada.
            """
            media = float(np.sqrt(np.asarray(mm.multiply(mm).sum(axis=1)).ravel()).mean())
            return mm if media <= 0 else mm.multiply(ref / media)

        rng = np.random.default_rng(SEMILLA + f)
        matrices = {
            "V03_G1": G1,
            "V03_G2": G2,
            "V03_ambos": np.hstack([G1, G2]),
            "V03_random": rng.standard_normal((n, G1.shape[1] + G2.shape[1])),
            "V03_margen": np.column_stack([margen_v03[:, j] for j in range(3)]),
        }
        for brazo, be in matrices.items():
            sd = be[ir].std(axis=0)
            sd[sd < 1e-8] = 1.0
            z = (be - be[ir].mean(axis=0)) / sd[None, :]
            norma = float(np.sqrt(np.asarray((z[ir] ** 2).sum(axis=1))).mean())
            z = z / norma if norma > 0 else z
            xa = escalar(hstack([xbase, csr_matrix(z[ir])]).tocsr())
            xav = escalar(hstack([xbase_v, csr_matrix(z[iva])]).tocsr())
            for j in range(3):
                mo = LogisticRegression(**LR)
                mo.fit(xa, y[ir, j])
                proba[brazo][iva, j] = mo.predict_proba(xav)[:, 1]
        log(f"      pliegue {f} listo")

    # ---------- metricas globales y por estrato ----------
    detalle: dict[str, Any] = {}
    filas: list[dict[str, Any]] = []
    for brazo in ("V03_G1", "V03_G2", "V03_ambos", "V03_random", "V03_margen"):
        p = proba[brazo]
        pr = (p >= V03_UMBRALES[None, :]).astype(int)
        ccv = float(f1_score(y, pr, average="macro"))
        folds = [float(f1_score(y[pliegues == g], pr[pliegues == g], average="macro"))
                 for g in range(1, 6)]
        ac = pr == y
        por_bin = {}
        for nombre, lo, hi in BINS:
            m = (n_tok >= lo) & (n_tok <= hi)
            f1b = float(f1_score(y[m], pr[m], average="macro", zero_division=0))
            f1v = float(f1_score(y[m], pred_v03[m], average="macro", zero_division=0))
            por_bin[nombre] = {"n": int(m.sum()), "F1": round(f1b, 6),
                               "F1_de_V03": round(f1v, 6), "delta": round(f1b - f1v, 6)}
        detalle[brazo] = {"corrected_cv": ccv, "folds": folds, "por_bin": por_bin,
                          "rescate": int(np.count_nonzero(ac & (pred_v03 != y))),
                          "damage": int(np.count_nonzero((~ac) & (pred_v03 == y)))}
        filas.append({"brazo": brazo, "corrected_cv": round(ccv, 10),
                      "delta_vs_V03": round(ccv - ccv_ref, 10)})
    folds_v = [float(f1_score(y[pliegues == g], pred_v03[pliegues == g], average="macro"))
               for g in range(1, 6)]

    log("")
    log("   LOS CINCO BRAZOS, con la metrica primaria global y el detalle por estrato")
    log(f"      {'brazo':14}{'global':>14}{'delta':>12}" + "".join(
        f"{b[0]:>11}" for b, _lo, _hi in BINS))
    for f in filas:
        b = f["brazo"]
        log(f"      {b:14}{f['corrected_cv']:>14.10f}{f['delta_vs_V03']:>+12.6f}"
            + "".join(f"{detalle[b]['por_bin'][n]['delta']:>+11.4f}" for n, _l, _h in BINS))
    log(f"      {'V03':14}{ccv_ref:>14.10f}{0.0:>+12.6f}"
        + "".join(f"{0.0:>+11.4f}" for _n, _l, _h in BINS))

    log("")
    log("   LA PREGUNTA DE ESTE TRACK, por estrato")
    log("   hay que superar al margen EN EL MISMO ESTRATO, no en el promedio")
    for nombre, _lo, _hi in BINS:
        d_g1 = detalle["V03_G1"]["por_bin"][nombre]["delta"]
        d_g2 = detalle["V03_G2"]["por_bin"][nombre]["delta"]
        d_am = detalle["V03_ambos"]["por_bin"][nombre]["delta"]
        d_rd = detalle["V03_random"]["por_bin"][nombre]["delta"]
        d_mg = detalle["V03_margen"]["por_bin"][nombre]["delta"]
        log(f"      {nombre:10} G1 {d_g1:+.4f} | G2 {d_g2:+.4f} | juntos {d_am:+.4f} | "
            f"aleatorio {d_rd:+.4f} | margen {d_mg:+.4f}")
        log(f"      {'':10} frente al margen: G1 {d_g1 - d_mg:+.4f} | G2 {d_g2 - d_mg:+.4f} | "
            f"juntos {d_am - d_mg:+.4f}")

    # ---------- veredicto ----------
    muy_larga = "MUY_LARGA"
    mejor_nombre = max(("V03_G1", "V03_G2", "V03_ambos"),
                       key=lambda b: detalle[b]["por_bin"][muy_larga]["delta"])
    d_mejor = detalle[mejor_nombre]["por_bin"][muy_larga]["delta"]
    d_marg_larga = detalle["V03_margen"]["por_bin"][muy_larga]["delta"]
    d_aleat_larga = detalle["V03_random"]["por_bin"][muy_larga]["delta"]
    d_glob = detalle[mejor_nombre]["corrected_cv"] - ccv_ref
    fm = sum(1 for a, b in zip(detalle[mejor_nombre]["folds"], folds_v) if a > b)
    gate_global = bool(d_glob >= GATE_DELTA and fm >= GATE_PLIEGUES
                       and detalle[mejor_nombre]["corrected_cv"]
                       > detalle["V03_margen"]["corrected_cv"])

    log("")
    log("   EL VEREDICTO, y la regla de decision declarada antes")
    log(f"      el mejor bloque en MUY_LARGA es {mejor_nombre} con {d_mejor:+.6f}, frente a")
    log(f"      {d_aleat_larga:+.6f} del control de capacidad y {d_marg_larga:+.6f} del margen")
    log(f"      el gate GLOBAL de la fase es {GATE_DELTA:+.4f} con 4/5 pliegues, y el global")
    log(f"      de {mejor_nombre} es {d_glob:+.10f} con {fm}/5")

    if d_mejor > 0 and d_mejor > d_marg_larga and d_mejor > d_aleat_larga and gate_global:
        lectura = "PROMISING"
        decision = "PROMISING. Supera el gate global y el estrato donde P63 dijo que estaba"
        veredicto = (f"{mejor_nombre} gana en la region larga y supera en ese mismo estrato "
                     f"al margen y al control de capacidad, y ademas supera el gate global de "
                     f"la fase con {d_glob:+.10f} y {fm} de 5 pliegues. Eso es un finalista y "
                     f"exige bootstrap y segunda particion antes de cualquier adopcion")
    elif d_mejor > 0 and d_mejor > d_marg_larga and d_mejor > d_aleat_larga:
        lectura = "SOLO_EN_ESTRATO"
        decision = "COMPLEMENTARY. Solo funciona en el estrato largo y no globalmente"
        veredicto = (f"{mejor_nombre} gana en la region larga, con {d_mejor:+.6f} frente a "
                     f"{d_marg_larga:+.6f} del margen y {d_aleat_larga:+.6f} del control, y "
                     f"eso es informacion real sobre DONDE esta la senal. Pero el global es "
                     f"{d_glob:+.10f} con {fm} de 5 pliegues, luego no pasa el gate. La regla "
                     f"de la fase es que la metrica primaria es el corrected_cv global, y "
                     f"adoptar algo que solo funciona en un estrato es como se entra en una "
                     f"especializacion que no transfiere. Queda como COMPLEMENTARIA y con la "
                     f"ubicacion de la senal medida")
    else:
        lectura = "NADA_EN_EL_ESTRATO"
        decision = "CLOSED. La evidencia no funciona ni donde P63 dijo que estaba el margen"
        veredicto = (f"ni {mejor_nombre} ni los demas ganan en la region larga por encima del "
                     f"margen: el mejor es {mejor_nombre} con {d_mejor:+.6f} frente a "
                     f"{d_marg_larga:+.6f} del margen en el mismo estrato. La hipotesis de la "
                     f"fase queda confirmada como DESCRIPCION y cerrada como "
                     f"REPRESENTACION, y la region larga no es una excepcion que la "
                     f"resuelva: ahi la estructura interna es mas repartida y mas "
                     f"contradictoria, y V03 ya la lleva dentro de su confianza")
    log("")
    log(f"   LECTURA: {lectura}")
    log(f"   VEREDICTO: {veredicto}")
    log(f"   DECISION: {decision}")

    with CSV_S.open("w", encoding="ascii", newline="") as fh:
        w = _csv.writer(fh, lineterminator="\n")
        w.writerow(["brazo", "corrected_cv", "delta_vs_V03"] + [f"delta_{b[0]}" for b, _l, _h in BINS])
        for f in filas:
            w.writerow([f["brazo"], f["corrected_cv"], f["delta_vs_V03"]]
                       + [detalle[f["brazo"]]["por_bin"][n]["delta"] for n, _l, _h in BINS])
        w.writerow(["V03", round(ccv_ref, 10), 0.0, 0.0, 0.0, 0.0, 0.0])
        w.writerow([])
        w.writerow(["bin", "n", "F1_de_V03", "F1_G1", "F1_G2", "F1_ambos", "F1_margen"])
        for nombre, _lo, _hi in BINS:
            w.writerow([nombre, detalle["V03_G1"]["por_bin"][nombre]["n"]]
                       + [detalle[b]["por_bin"][nombre]["F1"]
                          for b in ("V03_G1", "V03_G2", "V03_ambos", "V03_margen")])

    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                            text=True, cwd=PROJECT_ROOT, check=False).stdout.strip()
    informe = {
        "familia": "P64_ESPECIALIZACION_LARGA",
        "naturaleza": ("EXPERIMENTO. 3 candidatos y 2 controles, 0 submissions, 0 Kaggle, "
                       "0 holdout, 0 test"),
        "fecha": inicio.isoformat(timespec="seconds"), "experiment_id": "P64-04",
        "track": "P8-especializacion-larga",
        "script": "experiments/p64_05_especializacion_larga.py",
        "pregunta": ("los bloques de evidencia funcionan SOLO en LARGA y MUY_LARGA, donde "
                     "P63 midio que esta el margen residual"),
        "por_que_es_una pregunta_separada": (
            "P64-03 y P64-04 fallaron en el promedio sobre 1483 filas, y ese promedio lo "
            "dominan las cortas: CORTA y MUY_LARGA tienen 167 filas cada una, luego un numero "
            "global dice casi nada de ningun extremo"),
        "margen_residual_por_rebanada": {"CORTA": 0.0, "MEDIA": 0.0093, "LARGA": 0.0217,
                                         "MUY_LARGA": 0.0890},
        "metrica_primaria": ("el corrected_cv GLOBAL. Los cortes se heredan congelados de P63 "
                             "y la decision NO se toma sobre la mejor rebanada, porque "
                             "adoptar algo que solo ayuda en un estrato es como se entra en "
                             "una especializacion que no transfiere"),
        "bloques": {
            "G1_concentracion": {"columnas": COLS_G1, "n": int(G1.shape[1]),
                                "mecanismo": "un segmento domina la decision",
                                "por_que_no_es_el_max_pooling_de_P59": (
                                    "aqui el max es un RASGO que un modelo lineal pesa, no un "
                                    "agregador que sustituye la prediccion, luego el sesgo al "
                                    "alza que hundio a P59-07 no aplica")},
            "G2_conflicto": {"columnas": COLS_G2, "n": int(G2.shape[1]),
                             "mecanismo": "la decision descansa en un reparto contradictorio"},
        },
        "nombres": {"G1": nombres_g1, "G2": nombres_g2},
        "protocolo": {
            "bloques_reutilizados": ("los de P64-03 y P64-04, ya calculados con los "
                                     "coeficientes de V03 del pliegue de entrenamiento. No se "
                                     "recalcula nada, luego este script no introduce ningun "
                                     "modelo nuevo"),
            "control_de_capacidad": "V03 mas un bloque aleatorio de la anchura conjunta",
            "control_del_margen": "V03 mas sus margenes, evaluado EN EL MISMO ESTRATO",
            "confound_de_escala": "el bloque denso se normaliza a norma de fila uno",
            "umbrales": "0,44 / 0,51 / 0,46 congelados",
        },
        "resultados": detalle,
        "por_estrado": {nombre: {b: detalle[b]["por_bin"][nombre] for b in detalle}
                        for nombre, _lo, _hi in BINS},
        "mejor_bloque_en_muy_larga": mejor_nombre,
        "delta_del_mejor_en_muy_larga": round(d_mejor, 6),
        "delta_del_margen_en_muy_larga": round(d_marg_larga, 6),
        "delta_del_aleatorio_en_muy_larga": round(d_aleat_larga, 6),
        "gate_global": {"delta_min": GATE_DELTA, "pliegues_min": GATE_PLIEGUES,
                        "delta": round(d_glob, 10), "pliegues_mejorados": fm,
                        "superado": gate_global},
        "lectura": lectura, "veredicto": veredicto, "decision": decision,
        "experimentos_registrados": 1,
        "reproducibilidad": {"nivel": "R2", "git_commit": commit, "seed": SEMILLA,
                             "hash_v03": hash_real, "corrected_cv": V03_CCV},
        "formulacion": "V03 remains the best verified leaderboard result.",
    }
    JSON_S.write_text(json.dumps(informe, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")

    L: list[str] = []
    W = L.append
    W("# P64 - Los bloques de evidencia por estrato, y si el margen ya lo sabia")
    W("")
    W(f"Generado por `experiments/p64_05_especializacion_larga.py`. Fecha "
      f"{inicio.isoformat(timespec='seconds')}.")
    W("")
    W("## Por que esta pregunta es separada")
    W("")
    W("P64-03 y P64-04 fallaron en el promedio sobre 1483 filas, y ese promedio lo dominan")
    W("las cortas: CORTA y MUY_LARGA tienen 167 filas cada una. Un numero global dice casi")
    W("nada de ningun extremo.")
    W("")
    W("P63 midio donde esta el margen residual, y la respuesta fue especifica:")
    W("")
    W("```text")
    W("CORTA      margen hasta el techo de umbral  +0,0000")
    W("MEDIA      +0,0093")
    W("LARGA      +0,0217")
    W("MUY_LARGA  +0,0890")
    W("```")
    W("")
    W("En los tweets muy largos hay casi nueve centesimas de macro F1 sobre la mesa, y estan")
    W("de una forma que ningun umbral alcanza. Si el bloque de evidencia lleva la estructura que")
    W("la bolsa de n-gramas aplana, ahi es donde tiene que pagar, y un promedio global es")
    W("exactamente la estadistica disenada para esconderlo.")
    W("")
    W("## La metrica primaria NO cambia")
    W("")
    W("El corrected_cv **global** sigue siendo la metrica primaria, y la decision **no** se")
    W("toma sobre la mejor rebanada. Adoptar algo que solo ayuda en un estrato es como se entra")
    W("en una especializacion que no transfiere. Las rebanadas se informan para que el lector")
    W("vea la forma.")
    W("")
    W("## Los dos mecanismos, que son distintos")
    W("")
    W("| bloque | columnas | mecanismo |")
    W("|---|---:|---|")
    W(f"| G1 concentracion | {G1.shape[1]} | un segmento domina la decision |")
    W(f"| G2 conflicto | {G2.shape[1]} | la decision descansa en un reparto contradictorio |")
    W("")
    W("G1 **no** es el max-pooling de P59-07: aqui el maximo es un RASGO que un modelo lineal")
    W("pesa, no un agregador que sustituye la prediccion, luego el sesgo al alza que hundio a")
    W("P59 no aplica. Los dos bloques se **reutilizan** de P64-03 y P64-04, ya calculados con")
    W("los coeficientes de V03 del pliegue, luego este script no introduce ningun modelo nuevo.")
    W("")
    W("## Resultados")
    W("")
    W("| brazo | corrected_cv | delta global | " + " | ".join(f"delta {b[0]}"
                                                            for b, _l, _h in BINS) + " |")
    W("|---|---:|---:|" + "---:|" * len(BINS))
    for f in filas:
        b = f["brazo"]
        W(f"| {b} | {f['corrected_cv']:.10f} | {f['delta_vs_V03']:+.6f} | "
          + " | ".join(f"{detalle[b]['por_bin'][n]['delta']:+.4f}" for n, _l, _h in BINS) + " |")
    W(f"| **V03** | {ccv_ref:.10f} | +0.000000 | "
      + " | ".join("+0.0000" for _n, _l, _h in BINS) + " |")
    W("")
    W("### F1 absoluto por estrato")
    W("")
    W("| bin | n | V03 | G1 | G2 | juntos | margen |")
    W("|---|---:|---:|---:|---:|---:|---:|")
    for nombre, _lo, _hi in BINS:
        W(f"| {nombre} | {detalle['V03_G1']['por_bin'][nombre]['n']} | "
          f"{detalle['V03_G1']['por_bin'][nombre]['F1_de_V03']:.4f} | "
          + " | ".join(f"{detalle[b]['por_bin'][nombre]['F1']:.4f}"
                      for b in ("V03_G1", "V03_G2", "V03_ambos", "V03_margen")) + " |")
    W("")
    W("## La pregunta, en el mismo estrato")
    W("")
    W("Un bloque que ayuda a los tweets largos tiene que superar no a V03 sino a la")
    W("**confianza que V03 ya expone en esos mismos tweets**, porque si el margen ya lleva la")
    W("informacion, cualquier ganancia por rebanada es contabilidad.")
    W("")
    W("```text")
    for nombre, _lo, _hi in BINS:
        d_g1 = detalle["V03_G1"]["por_bin"][nombre]["delta"]
        d_g2 = detalle["V03_G2"]["por_bin"][nombre]["delta"]
        d_am = detalle["V03_ambos"]["por_bin"][nombre]["delta"]
        d_mg = detalle["V03_margen"]["por_bin"][nombre]["delta"]
        W(f"{nombre:10} G1 {d_g1:+.4f} ({d_g1 - d_mg:+.4f} contra el margen) | "
          f"G2 {d_g2:+.4f} ({d_g2 - d_mg:+.4f}) | juntos {d_am:+.4f} ({d_am - d_mg:+.4f})")
    W("```")
    W("")
    W("## Veredicto")
    W("")
    W("```text")
    W(veredicto)
    W("")
    W(f"LECTURA: {lectura}")
    W("```")
    W("")
    W("> V03 remains the best verified leaderboard result.")
    W("")
    W("*Generado por `experiments/p64_05_especializacion_larga.py`. Sin push.*")
    MD_S.write_text(a_ascii("\n".join(L)), encoding="ascii")

    for destino in (MD_S, JSON_S, CSV_S):
        if any(b > 127 for b in destino.read_bytes()):
            raise SystemExit(f"PARADA: {destino.name} conserva bytes no ASCII")
    log("")
    log(f"P64-05 OK | mejor en MUY_LARGA {mejor_nombre} {d_mejor:+.6f} | margen en el mismo "
        f"estrato {d_marg_larga:+.6f} | global {d_glob:+.6f} {fm}/5 | decision {decision}")


if __name__ == "__main__":
    ast.parse(Path(__file__).read_text(encoding="utf-8"))
    main()
