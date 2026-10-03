"""P59 step 09: track U, continuous fusion of the global and the local score.

The hypothesis
--------------
Even if the isolated local score is worse than V03, it can carry complementary
evidence. A continuous fusion with V03 would keep the global representation and
exploit the local signal without replacing it.

Why U survives the closure that killed T
-----------------------------------------
Track T was closed because Platt scaling at a fixed threshold is algebraically a
re-selection of the per-label threshold: a monotone map cannot change a ranking.
Fusion is the opposite operation. The map

    p_fused = (1 - alpha) * p_global + alpha * p_local

is not monotone in `p_global` whenever the two scores disagree, so it genuinely
reorders cells. That is the exact algebraic line between T and U, and it is why
U escapes the root closure that `V8-09` placed on calibrations and thresholds.

The fusion family itself is NOT unexplored. `ENS-01` through `ENS-04` ran equal
weight soft voting, a simplex weight grid, per-label stacking on four OOF
probabilities, and a nested best pair/trio; `V4-H01` and `S08` ran a 36-feature
meta; `FIN-08`, `FIN-09` and `FIN-10` sent meta candidates to the leaderboard and
all three fell below V03. What is new here is the **base**, not the mechanism: no
prior fusion used a score computed from a different unit of representation. That
is why the track is run at all, and why the gate is kept strict.

Everything is pre-registered
---------------------------
    p_global = V03, the canonical OOF
    p_local  = B1 raw, the canonical max over windows
    p_fused  = (1 - alpha) * p_global + alpha * p_local, per label
    alpha     ONE value shared by the three labels, chosen by ONE rule:
              maximise the macro-F1 of the fused score at V03's frozen
              thresholds, on the inner out-of-fold predictions of the OUTER
              TRAINING FOLD only
    grid      alpha in {0.00, 0.05, ..., 1.00}, fixed in advance, 21 values
    null      alpha = 0 IS in the grid, so the rule can always fall back on V03
    thresholds 0,44 / 0,51 / 0,46, never retuned
    outer validation, holdout, test and the leaderboard are never used to pick
    alpha

The oracle alpha, the one that would have maximised the OUTER validation score,
is computed as a diagnostic only. It is not a candidate, it is never scored as
one, and it answers a question the nested rule cannot: does the local score carry
complementary signal in principle, or is the nested selection simply failing to
find it.

Usage
-----
    /home/carlos/venv/bin/python experiments/p59_09_fusion_local.py
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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

from p59_01_voronoi_local import (
    ETIQUETAS,
    V03_CCV,
    V03_SHA,
    V03_SUB,
)
from p59_07_local_window import (
    C_MODELO,
    MAX_ITER,
    SEMILLA,
    STRIDE,
    VENTANA,
    cortar_ventanas,
)

OOF_V03 = PROJECT_ROOT / "features" / "oof_v03_final.npz"
CELDA_B1 = PROJECT_ROOT / "features" / "p59_local_window_cells.npz"
TRAIN_RAW = PROJECT_ROOT / "data" / "raw" / "toxic_train.csv"
PARTICION = PROJECT_ROOT / "data" / "processed" / "holdout_v2_ids.csv"
JSON_U = PROJECT_ROOT / "reports" / "p59_U_global_local_fusion.json"
MD_U = PROJECT_ROOT / "reports" / "p59_U_global_local_fusion.md"
CSV_U = PROJECT_ROOT / "reports" / "p59_U_global_local_fusion.csv"
CELDA_NPZ = PROJECT_ROOT / "features" / "p59_fusion_cells.npz"

V03_UMBRALES = np.array([0.44, 0.51, 0.46])
GATE_DELTA = 0.003
GATE_PLIEGUES = 4
N_BOOT = 2000
N_INNER = 4
SEMILLA_INNER = 20260928
# Rejilla de alpha fijada antes de ejecutar. Un solo alpha, compartido por las tres
# etiquetas. El 0 esta dentro, luego la regla puede volver siempre a V03.
ALPHAS = np.round(np.arange(0.0, 1.0001, 0.05), 2)


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[P59-09] {mensaje}", flush=True)


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


def f1_macro(pred: np.ndarray, y: np.ndarray) -> float:
    """Macro-F1 sobre macro.

    Args:
        pred: predicciones binarias.
        y: etiquetas verdaderas.

    Returns:
        El valor de la metrica.
    """
    return float(f1_score(y, pred, average="macro"))


def oof_de_una_etiqueta(x, y_bin, idx_tr, idx_val) -> np.ndarray:
    """Probabilidad fuera de muestra de un modelo global en un pliegue.

    Args:
        x: matriz dispersa del training fold.
        y_bin: etiqueta binaria de todas las filas.
        idx_tr: filas de entrenamiento.
        idx_val: filas de validacion.

    Returns:
        Probabilidades en las filas de validacion.
    """
    lr = LogisticRegression(C=C_MODELO, class_weight="balanced", max_iter=MAX_ITER,
                            solver="liblinear", random_state=SEMILLA)
    lr.fit(x[idx_tr], y_bin[idx_tr])
    return lr.predict_proba(x[idx_val])[:, 1]


def oof_local_una_etiqueta(textos_win_tr, y_bin, padre_win_padre,
                           idx_val, textos_val, padre_val) -> np.ndarray:
    """Probabilidad local fuera de muestra, con max sobre ventanas.

    Args:
        textos_win_tr: textos de las ventanas del training fold.
        padre_win: indice de fila de cada ventana de entrenamiento.
        y_bin: etiqueta binaria de todas las filas.
        padre_win_padre: filasoriginales de las ventanas, para la etiqueta.
        idx_tr: filas de entrenamiento.
        idx_val: filas de validacion.
        textos_val: textos de las ventanas de validacion.
        padre_val: fila de cada ventana de validacion.

    Returns:
        Probabilidad agregada por max en las filas de validacion.
    """
    pw = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2,
                         sublinear_tf=True, lowercase=True)
    pc = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2,
                         sublinear_tf=True, lowercase=True)
    xw = hstack([pw.fit_transform(textos_win_tr), pc.fit_transform(textos_win_tr)]).tocsr()
    xvw = hstack([pw.transform(textos_val), pc.transform(textos_val)]).tocsr()
    lr = LogisticRegression(C=C_MODELO, class_weight="balanced", max_iter=MAX_ITER,
                            solver="liblinear", random_state=SEMILLA)
    lr.fit(xw, y_bin[padre_win_padre])
    pv = lr.predict_proba(xvw)[:, 1]
    padre_val = np.asarray(padre_val)
    salida = np.zeros(len(idx_val))
    for k, fila in enumerate(idx_val):
        sel = pv[padre_val == fila]
        salida[k] = float(sel.max()) if sel.size else 0.0
    return salida


def main() -> None:
    """Ejecuta U1: una sola fusion continua con alpha anidado en training.

    Returns:
        None.
    """
    inicio = datetime.now(timezone.utc)
    log("P59-U Global + local fusion. UNA sola alpha, estimada solo con training")
    log("   0 Kaggle, 0 holdout, 0 test, 0 submissions. V03 NO se modifica")

    hash_real = hashlib.sha256(V03_SUB.read_bytes()).hexdigest()
    if hash_real != V03_SHA:
        raise SystemExit("PARADA: la submission de V03 no coincide con el sha256 congelado")
    log(f"   V03 sha256 verificado e intacto: {hash_real[:16]}...")

    v = np.load(OOF_V03, allow_pickle=True)
    ids, y, pliegues, pred_v03, proba_v03 = v["ids"], v["y_true"], v["pliegues"], v["pred"], v["proba"]
    ccv_ref = f1_macro(pred_v03, y)
    if abs(ccv_ref - V03_CCV) > 1e-12:
        raise SystemExit(f"PARADA: el OOF canonico da {ccv_ref!r}, no {V03_CCV!r}")
    n = ids.size

    if not CELDA_B1.exists():
        raise SystemExit("PARADA: falta features/p59_local_window_cells.npz; ejecuta antes P59-07")
    b1 = np.load(CELDA_B1, allow_pickle=True)
    if not np.array_equal(b1["ids"], ids):
        raise SystemExit("PARADA: B1 y el OOF canonico tienen ids distintos")
    proba_l = b1["proba_b1"]
    pred_l = b1["pred_b1"]
    ccv_l = f1_macro(pred_l, y)
    log(f"   T0/U0 control: V03 {ccv_ref!r} exacto | B1 crudo {ccv_l!r}")

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
    ventanas = [cortar_ventanas(t, VENTANA, STRIDE) for t in textos]
    multi = np.array([len(w) > 1 for w in ventanas])
    log(f"   {n} filas | {len(set(grupos))} grupos intactos | "
        f"tweets multi-ventana {int(multi.sum())}")

    # matrices de features del modelo global, ajustadas por pliegue exterior
    log("")
    log("  (features globales por pliegue exterior, el mismo ajuste que V03)")
    x_por_pliegue: dict[int, Any] = {}
    for f in range(1, 6):
        ir = np.where(pliegues != f)[0]
        iva = np.where(pliegues == f)[0]
        pw = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2,
                             sublinear_tf=True, lowercase=True)
        pc = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2,
                             sublinear_tf=True, lowercase=True)
        x_por_pliegue[f] = hstack([pw.fit_transform([textos[i] for i in ir]),
                                   pc.fit_transform([textos[i] for i in ir])]).tocsr()
    log("      listo")

    # ---------------- seleccion anidada de alpha ----------------
    log("")
    log("   ALPHA anidado: se elige SOLO con el training fold exterior")
    proba_u = np.zeros((n, 3))
    alphas_elegidos: list[float] = []
    t0 = time.time()
    for f in range(1, 6):
        ir = np.where(pliegues != f)[0]
        iva = np.where(pliegues == f)[0]
        # x_por_pliegue[f] esta construida sobre las filas de IR, luego sus indices
        # son POSICIONES dentro de IR y no filas globales. Todo el bucle interno
        # trabaja en esas posiciones, y solo se traducen a filas globales cuando
        # hace falta el texto.
        x = x_por_pliegue[f]
        y_ir = y[ir]
        {int(g): k for k, g in enumerate(ir)}
        filas_ir = np.arange(len(ir))
        estrato = y_ir.sum(axis=1).astype(str)
        skf = StratifiedGroupKFold(n_splits=N_INNER, shuffle=True,
                                    random_state=SEMILLA_INNER)
        pg_in = np.zeros((len(ir), 3))
        pl_in = np.zeros((len(ir), 3))
        for k_tr, k_va in skf.split(filas_ir, estrato, grupos[ir]):
            # ventanas de entrenamiento del inner fold
            tw: list[str] = []
            padre_win: list[int] = []
            for pos_f in k_tr:
                for w in ventanas[int(ir[pos_f])]:
                    tw.append(w)
                    padre_win.append(int(pos_f))
            # ventanas de validacion del inner fold
            tv: list[str] = []
            padre_val: list[int] = []
            for pos_f in k_va:
                for w in ventanas[int(ir[pos_f])]:
                    tv.append(w)
                    padre_val.append(int(pos_f))
            padre_win_arr = np.array(padre_win)
            padre_val_arr = np.array(padre_val)
            for j in range(3):
                yb = y_ir[:, j].astype(int)
                pg_in[k_va, j] = oof_de_una_etiqueta(x, yb, k_tr, k_va)
                pl_in[k_va, j] = oof_local_una_etiqueta(
                    tw, yb, padre_win_arr, k_va, tv, padre_val_arr)
        # una sola regla: maximizar macro-F1 del score fusionado en training
        mejor_a, mejor_f1 = 0.0, -1.0
        curva: list[tuple[float, float]] = []
        for a in ALPHAS:
            pf = (1.0 - a) * pg_in + a * pl_in
            pr = (pf >= V03_UMBRALES[None, :]).astype(int)
            v_f1 = f1_macro(pr, y_ir)
            curva.append((float(a), v_f1))
            if v_f1 > mejor_f1 + 1e-12:
                mejor_a, mejor_f1 = float(a), v_f1
        alphas_elegidos.append(mejor_a)
        proba_u[iva] = (1.0 - mejor_a) * proba_v03[iva] + mejor_a * proba_l[iva]
        log(f"      pliegue {f}: alpha = {mejor_a:.2f} | macro-F1 en training {mejor_f1:.6f} "
            f"({time.time() - t0:.1f} s acumulados)")
    proba_u[proba_u < 0.0] = 0.0
    proba_u[proba_u > 1.0] = 1.0
    pred_u = (proba_u >= V03_UMBRALES[None, :]).astype(int)
    ccv_u = f1_macro(pred_u, y)
    delta = ccv_u - ccv_ref
    log("")
    log(f"   U fused corrected_cv {ccv_u!r} | delta vs V03 {delta:+.10f}")
    log(f"   alphas por pliegue: {alphas_elegidos}")

    # ---------------- oracle, SOLO diagnostico ----------------
    orac = []
    mejor = (-1.0, 0.0)
    for a in ALPHAS:
        pf = (1.0 - a) * proba_v03 + a * proba_l
        pr = (pf >= V03_UMBRALES[None, :]).astype(int)
        v_f1 = f1_macro(pr, y)
        orac.append(round(float(a), 2))
        if v_f1 > mejor[0]:
            mejor = (v_f1, float(a))
    log("   DIAGNOSTICO oracle (NO es candidato): alpha que habria|maximizado el score EXTERNO")
    log(f"      alpha_oracle = {mejor[1]:.2f} | corrected_cv = {mejor[0]:.10f} | "
        f"delta = {mejor[0] - ccv_ref:+.10f}")
    log("      esto mide si hay senal complementaria en principio, no si la regla anidada la encuentra")

    # ---------------- metricas ----------------
    fp_u = [f1_macro(pred_u[pliegues == k], y[pliegues == k]) for k in range(1, 6)]
    fp_g = [f1_macro(pred_v03[pliegues == k], y[pliegues == k]) for k in range(1, 6)]
    pliegues_mej = sum(1 for u, w in zip(fp_u, fp_g) if u > w)
    rescue = int(np.count_nonzero((pred_u == y) & (pred_v03 != y)))
    damage = int(np.count_nonzero((pred_u != y) & (pred_v03 == y)))
    cambian = int(np.count_nonzero(pred_u != pred_v03))
    log("")
    log(f"   pliegues que mejoran: {pliegues_mej}/5 | deltas "
        + " ".join(f"{u - w:+.4f}" for u, w in zip(fp_u, fp_g)))
    log(f"   rescate {rescue} | damage {damage} | neto {rescue - damage:+d} | "
        f"cambian {cambian} | acierto en las que cambian "
        f"{round(rescue / cambian, 4) if cambian else None}")
    por = {}
    for j, e in enumerate(ETIQUETAS):
        por[e] = {"f1": round(float(f1_score(y[:, j], pred_u[:, j])), 10),
                  "f1_ref": round(float(f1_score(y[:, j], pred_v03[:, j])), 10),
                  "rescate": int(np.count_nonzero((pred_u[:, j] == y[:, j]) &
                                                  (pred_v03[:, j] != y[:, j]))),
                  "damage": int(np.count_nonzero((pred_u[:, j] != y[:, j]) &
                                                  (pred_v03[:, j] == y[:, j])))}
    log("   F1 por etiqueta (U):  "
        + " ".join(f"{e} {por[e]['f1']:.4f}" for e in ETIQUETAS))
    log("   F1 por etiqueta (V03): "
        + " ".join(f"{e} {por[e]['f1_ref']:.4f}" for e in ETIQUETAS))

    # complementariedad por estrato
    est_res: dict[str, Any] = {}
    for nombre, mascara in (("multi_ventana", multi), ("una_sola_ventana", ~multi)):
        resc = int(np.count_nonzero((pred_u == y) & (pred_v03 != y) & mascara[:, None]))
        dan = int(np.count_nonzero((pred_u != y) & (pred_v03 == y) & mascara[:, None]))
        est_res[nombre] = {"n_tweets": int(mascara.sum()), "rescate": resc,
                           "damage": dan, "neto": resc - dan}
    log(f"   multi-ventana: rescate {est_res['multi_ventana']['rescate']} "
        f"damage {est_res['multi_ventana']['damage']} "
        f"neto {est_res['multi_ventana']['neto']:+d} | "
        f"una-sola-ventana: rescate {est_res['una_sola_ventana']['rescate']} "
        f"damage {est_res['una_sola_ventana']['damage']} "
        f"neto {est_res['una_sola_ventana']['neto']:+d}")

    h_delta = delta >= GATE_DELTA
    h_folds = pliegues_mej >= GATE_PLIEGUES
    gate = bool(h_delta and h_folds)
    log("")
    log(f"   GATE: delta {delta:+.6f} >= {GATE_DELTA}: {h_delta} | "
        f"pliegues {pliegues_mej}/5 >= {GATE_PLIEGUES}: {h_folds} | superado: {gate}")
    ic_lo = ic_hi = None
    if gate:
        rng = np.random.default_rng(SEMILLA)
        d_celdas = ((pred_u == y).astype(int) - (pred_v03 == y).astype(int)).ravel()
        m = d_celdas[rng.integers(0, d_celdas.size, size=(N_BOOT, d_celdas.size))].mean(axis=1)
        ic_lo, ic_hi = float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))
        log(f"   bootstrap (solo porque el gate se supero): IC95 [{ic_lo:+.6f}, {ic_hi:+.6f}]")

    if delta < 0:
        decision = ("CLOSED. La fusion continua del score local no mejora la metrica, y con alpha "
                    "anidado elumbral 0 deja al candidato igual a V03. NO se crea una segunda familia de "
                    "fusion (parrafo 16)")
    elif not gate:
        decision = ("MARGINAL. La fusion mejora la metrica pero no alcanza el gate delta >= +0,003. "
                    "NO se demuestra. Bootstrap NO ejecutado")
    else:
        decision = ("PROMISING. Supera delta y consistencia de pliegues. Queda como candidato local; "
                    "no se envia nada y la segunda particion se reserva")

    np.savez_compressed(CELDA_NPZ, ids=ids, y_true=y, pliegues=pliegues,
                        pred_v03=pred_v03, pred_u=pred_u,
                        proba_v03=proba_v03, proba_b1=proba_l, proba_fusion=proba_u,
                        alphas=np.array(alphas_elegidos), multi_ventana=multi)

    informe = {
        "familia": "P59_U_GLOBAL_LOCAL_FUSION",
        "naturaleza": ("EXPERIMENTO. 1 candidato, 0 submissions, 0 Kaggle, 0 holdout, 0 test. "
                       "V03 no se modifica"),
        "fecha": inicio.isoformat(timespec="seconds"),
        "experiment_id": "P59-09",
        "track": "P59-U",
        "dependencia": "B1, de la que hereda el score local crudo",
        "U1_gap_check": {
            "veredicto": "BASE NUEVA, MECANISMO YA EXPLORADO",
            "base_nueva": ("ninguna fusion previa uso un score calculado sobre una UNIDAD DE "
                           "REPRESENTACION distinta. ENS-01..04 combinan cuatro bases globales; "
                           "V4-H01 y S08 usan un meta de 36 rasgos; FIN-08/09/10 son candidatos de "
                           "meta"),
            "mecanismo_previo": ("ENS-01 voto suave igual, ENS-02 rejilla de pesos del simplex, "
                                 "ENS-03 stacking por etiqueta, ENS-04 mejor par o trio anidado; "
                                 "ENS-02 dio +0,0023 y no supera el gate"),
            "por_que_U_no_esta_cerrada_como_T": (
                "una reparametrizacion monotona no cambia el ranking, y por eso T cae en el cierre de "
                "V8-09. La fusion si cambia el ranking siempre que los dos scores discrepen, luego "
                "escapa a ese cierre"),
            "riesgo": ("la familia tiene seis o mas fracasos y tres sondas enviadas al leaderboard que "
                       "quedaron por debajo de V03, luego la priori de U es baja"),
        },
        "configuracion_pre_registrada": {
            "p_global": "V03, el OOF canonico",
            "p_local": "B1 crudo, el max sobre ventanas",
            "fusion": "p = (1 - alpha) * p_global + alpha * p_local, por etiqueta",
            "alpha": "UN solo valor compartido por las tres etiquetas",
            "regla_de_seleccion": ("maximizar el macro-F1 del score fusionado a los umbrales "
                                   "congelados de V03, sobre las predicciones inner-OOF del TRAINING "
                                   "FOLD EXTERIOR. Ni una vez se toca validation, holdout, test ni "
                                   "leaderboard"),
            "rejilla_alpha": f"{ALPHAS.min():.2f} a {ALPHAS.max():.2f} en pasos de 0.05, {len(ALPHAS)} valores, fijada antes",
            "nulo": "alpha = 0 esta en la rejilla, luego la regla puede volver siempre a V03",
            "inner": f"{N_INNER} pliegues agrupados con StratifiedGroupKFold, semilla {SEMILLA_INNER}",
            "umbrales": "0,44 / 0,51 / 0,46 congelados, no reajustados",
        },
        "control": {
            "corrected_cv_V03": repr(ccv_ref), "reproduce_exacto": bool(abs(ccv_ref - V03_CCV) <= 1e-12),
            "corrected_cv_B1_crudo": round(ccv_l, 10),
        },
        "resultado": {
            "corrected_cv_U": round(ccv_u, 10), "delta_vs_V03": round(delta, 10),
            "folds_U": [round(x, 6) for x in fp_u], "folds_V03": [round(x, 6) for x in fp_g],
            "pliegues_mejorados": pliegues_mej,
            "alphas_por_pliegue": alphas_elegidos,
            "f1_por_etiqueta": por,
            "rescate": rescue, "damage": damage, "neto": rescue - damage, "cambian": cambian,
            "acierto_en_las_que_cambian": round(rescue / cambian, 4) if cambian else None,
            "complementariedad_por_estrato": est_res,
            "gate": {"delta_min": GATE_DELTA, "pliegues_min": GATE_PLIEGUES,
                     "supera_delta": bool(h_delta), "supera_pliegues": bool(h_folds),
                     "superado": gate},
            "bootstrap": (None if ic_lo is None else
                          {"IC95": [round(ic_lo, 6), round(ic_hi, 6)], "n": N_BOOT}),
        },
        "oracle_diagnostico": {
            "nota": ("NO es un candidato y nunca se puntua como tal. Mide si hay senal "
                     "complementaria en principio, que es una pregunta que la regla anidada no "
                     "puede responder"),
            "alpha_oracle": mejor[1], "corrected_cv_oracle": round(mejor[0], 10),
            "delta_oracle": round(mejor[0] - ccv_ref, 10),
        },
        "decision": decision,
        "experimentos_registrados": 1,
        "formulacion": "V03 remains the best verified leaderboard result.",
        "gastos": {"submissions": 0, "consultas_kaggle": 0, "uso_holdout": 0, "uso_test": 0,
                   "segundos": round((datetime.now(timezone.utc) - inicio).total_seconds(), 1)},
    }
    JSON_U.write_text(json.dumps(informe, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    L: list[str] = []
    A = L.append
    A("# P59-U - Fusion continua del score global y el local")
    A("")
    A(f"Generado por `experiments/p59_09_fusion_local.py`. Fecha {inicio.isoformat(timespec='seconds')}.")
    A("")
    A("## Hipotesis")
    A("")
    A("Aunque el score local aislado sea peor que V03, puede contener evidencia complementaria. Una "
      "fusion continua conserva la representacion global y aprovecha la senal local sin sustituirla.")
    A("")
    A("## U1 gap check: base nueva, mecanismo ya explorado")
    A("")
    A("Lo nuevo es la **base**, no el mecanismo. `ENS-01` a `ENS-04` combinan cuatro bases globales "
      "(voto igual, rejilla del simplex, stacking por etiqueta, mejor par o trio anidado); `V4-H01` y "
      "`S08` usan un meta de 36 rasgos; `FIN-08`, `FIN-09` y `FIN-10` enviaron candidatos de meta al "
      "leaderboard y los tres quedaron por debajo de V03. **Ninguna fusion previa uso un score "
      "calculado sobre una unidad de representacion distinta.**")
    A("")
    A("Por que U no esta cerrada como T: una reparametrizacion monotona no cambia el ranking, luego T "
      "cae en el cierre de V8-09; la fusion si cambia el ranking siempre que los dos scores discrepen.")
    A("")
    A("## Configuracion, fijada antes de mirar validation")
    A("")
    A("```text")
    A("p_global   V03, el OOF canonico")
    A("p_local    B1 crudo, el max sobre ventanas")
    A("fusion     p = (1 - alpha) * p_global + alpha * p_local")
    A("alpha      UN valor, compartido por las tres etiquetas")
    A("regla      maximizar el macro-F1 a los umbrales congelados, sobre las predicciones")
    A("           inner-OOF del TRAINING FOLD EXTERIOR. Validation nunca se toca")
    A(f"rejilla    alpha de {ALPHAS.min():.2f} a {ALPHAS.max():.2f} en pasos de 0.05, {len(ALPHAS)} valores")
    A("nulo       alpha = 0 esta en la rejilla, luego la regla puede volver a V03")
    A("umbrales   0,44 / 0,51 / 0,46 congelados")
    A("```")
    A("")
    A("## Control")
    A("")
    A("```text")
    A(f"V03 {ccv_ref!r} reproduce exacto: {abs(ccv_ref - V03_CCV) <= 1e-12}")
    A(f"B1 crudo {ccv_l!r}")
    A("```")
    A("")
    A("## Resultados")
    A("")
    A("```text")
    A(f"corrected_cv U     {ccv_u:.10f}")
    A(f"corrected_cv V03   {ccv_ref:.10f}")
    A(f"delta_vs_V03       {delta:+.10f}")
    A(f"pliegues que mejoran {pliegues_mej}/5")
    A(f"folds U     {' '.join(f'{x:.4f}' for x in fp_u)}")
    A(f"folds V03   {' '.join(f'{x:.4f}' for x in fp_g)}")
    A(f"alpha por pliegue  {alphas_elegidos}")
    A(f"rescate {rescue} | damage {damage} | neto {rescue - damage:+d} | cambian {cambian}")
    A("```")
    A("")
    A("| etiqueta | F1 U | F1 V03 | rescate | damage |")
    A("|---|---:|---:|---:|---:|")
    for e in ETIQUETAS:
        A(f"| {e} | {por[e]['f1']:.4f} | {por[e]['f1_ref']:.4f} | {por[e]['rescate']} | {por[e]['damage']} |")
    A("")
    A("## Complementariedad por estrato")
    A("")
    A("| estrato | n tweets | rescate | damage | neto |")
    A("|---|---:|---:|---:|---:|")
    for nombre in ("multi_ventana", "una_sola_ventana"):
        e_ = est_res[nombre]
        A(f"| {nombre} | {e_['n_tweets']} | {e_['rescate']} | {e_['damage']} | {e_['neto']:+d} |")
    A("")
    A("## Diagnostico oracle, NO es candidato")
    A("")
    A(f"El alpha que habria maximizado el score **externo** es {mejor[1]:.2f}, con corrected_cv "
      f"{mejor[0]:.10f} y delta {mejor[0] - ccv_ref:+.10f}. Mide si hay senal complementaria en "
      "principio, que es la pregunta que la regla anidada no puede responder. Nunca se puntua como "
      "candidato.")
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
    A("*Seccion generada por `experiments/p59_09_fusion_local.py`. Sin push.*")
    MD_U.write_text(a_ascii("\n".join(L)), encoding="ascii")

    import csv as _csv
    filas = [
        ["corrected_cv_V03", repr(ccv_ref)], ["corrected_cv_B1_crudo", repr(ccv_l)],
        ["corrected_cv_U", repr(ccv_u)], ["delta_vs_V03", repr(delta)],
        ["pliegues_mejorados", str(pliegues_mej)],
        ["folds_U", " ".join(f"{x:.6f}" for x in fp_u)],
        ["alphas_por_pliegue", " ".join(f"{a:.2f}" for a in alphas_elegidos)],
        ["rescate", str(rescue)], ["damage", str(damage)], ["neto", str(rescue - damage)],
        ["cambian", str(cambian)],
        ["gate_delta_superado", str(h_delta)], ["gate_pliegues_superado", str(h_folds)],
        ["gate_superado", str(gate)],
        ["alpha_oracle_diagnostico", f"{mejor[1]:.2f}"],
        ["delta_oracle_diagnostico", f"{mejor[0] - ccv_ref:+.10f}"],
        ["decision", a_ascii(decision)],
    ]
    with CSV_U.open("w", encoding="ascii", newline="") as fh:
        w = _csv.writer(fh, lineterminator="\n")
        w.writerow(["magnitud", "valor"])
        w.writerows(filas)

    for destino in (MD_U, JSON_U, CSV_U):
        if destino.exists() and any(b > 127 for b in destino.read_bytes()):
            raise SystemExit(f"PARADA: {destino.name} conserva bytes no ASCII")
    log("")
    log(f"P59-U VEREDICTO: {decision.split('.')[0]}")
    log(f"P59-09 OK | {informe['gastos']['segundos']} s")


if __name__ == "__main__":
    warnings.simplefilter("error", RuntimeWarning)
    ast.parse(Path(__file__).read_text(encoding="utf-8"))
    main()
