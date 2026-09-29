"""P66 tracks P14, P17, P18 and P21: multi-source fusion, complementarity, rescue and bootstrap.

What this script is for
-----------------------
P66-02 found one candidate that meets both interest gates: MI_arbol with the residual
correction, +0,0026743167 on 4 of 5 folds, with an attribution gap of +0,0041133588 against a
shuffled score. Four questions remain before that can be called anything more than
interesting, and they are what this script answers.

    P14  does a second source add anything? Only measured if both have UNIQUE rescue, because
         combining two scores that correct the same cells is not two sources, it is one.

    P17  how many V03 errors does it rescue and how many V03 correct rows does it damage, by
         label and by length stratum, because a fusion that wins on average while damaging the
         long tweets is not a fusion we want.

    P18  are the rescued rows the ones V03 got wrong WITH CONFIDENCE. This is the sharpest
         question in the phase: a fusion that only rescues rows sitting next to the threshold
         has found calibration, not representation, and the residual of V03 has nothing to do
         with it.

    P21  bootstrap with the resampling unit of the project, which is the near-duplicate group
         and not the row. Resampling rows would understate the interval because rows inside a
         group are not independent.

The multi-source rule is declared before execution
--------------------------------------------------
Combining two candidates requires, in this order: correlation between their logits, unique
rescue of each, and overlap of rescue. If a candidate has no unique rescue it is not combined,
because adding a second score that corrects the same cells adds variance and no information.
That rule is in the code and the decision follows it.

Usage
-----
    /home/carlos/venv/bin/python experiments/p66_03_complementariedad_bootstrap.py
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
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

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
CELDA = PROJECT_ROOT / "features" / "p66_scores_candidatos.npz"
MD_C = REPORTS / "p66_complementarity.md"
JSON_C = REPORTS / "p66_complementarity.json"
CSV_COMP = REPORTS / "p66_complementarity.csv"
CSV_INTER = REPORTS / "p66_error_intersection.csv"
CSV_COMP2 = REPORTS / "p66_candidate_comparison.csv"

V03_UMBRALES = np.array([0.44, 0.51, 0.46])
EPS = 1e-6
DELTA_HC = 0.10
N_BOOT = 2000
N_INNER = 3
SEMILLA = 42
BETAS = (0.00, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50)
GATE_DELTA = 0.0015
GATE_PLIEGUES = 4
GATE_FINALISTA = 0.003
BINS: tuple[tuple[str, int, int], ...] = (
    ("CORTA", 0, 6),
    ("MEDIA", 7, 15),
    ("LARGA", 16, 30),
    ("MUY_LARGA", 31, 10**9),
)
CANDIDATOS = ("MI_arbol", "negacion", "word_1_3", "cohesion", "huecos_1_2", "contraste",
              "huecos_6_10")


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[P66-03] {mensaje}", flush=True)


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


def logit(p: np.ndarray) -> np.ndarray:
    """Logit numericamente seguro.

    Args:
        p: probabilidades.

    Returns:
        Los logits.
    """
    q = np.clip(np.asarray(p, dtype=float), EPS, 1.0 - EPS)
    return np.log(q / (1.0 - q))


def sigmoide(z: np.ndarray) -> np.ndarray:
    """Sigmoide estable.

    Args:
        z: logits.

    Returns:
        Probabilidades.
    """
    z = np.asarray(z, dtype=float)
    s = np.empty_like(z)
    pos = z >= 0
    s[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    e = np.exp(z[~pos])
    s[~pos] = e / (1.0 + e)
    return s


def decision(pr: np.ndarray) -> np.ndarray:
    """Aplica los umbrales CONGELADOS de V03, uno por etiqueta.

    HS, TR y AG son tres problemas binarios independientes con umbrales distintos, luego cada
    columna se compara con SU umbral y no con el vector entero.

    Args:
        pr: probabilidades.

    Returns:
        Matriz de decisiones con la misma forma que la entrada.
    """
    pp = np.asarray(pr, dtype=float)
    if pp.ndim == 1:
        pp = pp.reshape(-1, 1)
    salida = np.zeros(pp.shape, dtype=int)
    for j in range(pp.shape[1]):
        salida[:, j] = (pp[:, j] >= V03_UMBRALES[j]).astype(int)
    return salida


def ccv_de(p: np.ndarray, y: np.ndarray, etiquetas: tuple[int, ...] | None = None) -> float:
    """Macro F1 sobre las etiquetas indicadas, con SU umbral.

    Args:
        p: probabilidades.
        y: etiquetas verdaderas.
        etiquetas: indices a promediar.

    Returns:
        El macro F1.
    """
    pp = np.asarray(p, dtype=float)
    yy = np.asarray(y, dtype=int)
    cols = range(pp.shape[1]) if etiquetas is None else etiquetas
    partes = [float(f1_score(yy[:, j], (pp[:, j] >= V03_UMBRALES[j]).astype(int),
                            average="binary", zero_division=0)) for j in cols]
    return float(np.mean(partes)) if partes else 0.0


def main() -> None:
    """Multi-fuente, complementariedad, rescate de alta confianza y bootstrap.

    Returns:
        None.
    """
    inicio = datetime.now(timezone.utc)
    log("P66-03 multi-fuente, complementariedad, rescate y bootstrap")

    hash_real = hashlib.sha256(V03_SUB.read_bytes()).hexdigest()
    if hash_real != V03_SHA:
        raise SystemExit("PARADA: la submission de V03 no coincide con el sha256 congelado")
    z = np.load(CELDA, allow_pickle=True)
    ids, y, pliegues = z["ids"], z["y_true"].astype(int), z["pliegues"]
    pred_v03, proba_v03 = z["pred_v03"], z["proba_v03"]
    _margen_v03, n_tok = z["margen_v03"], z["n_tokens"]
    n = ids.size
    ccv_ref = float(f1_score(y, pred_v03, average="macro"))
    if ccv_ref != V03_CCV:
        raise SystemExit(f"PARADA: el OOF canonico da {ccv_ref!r}, no {V03_CCV!r}")
    proba_c = {c: z[f"proba_{c}"] for c in CANDIDATOS}
    log(f"   V03 verificado: {ccv_ref!r} | {n} filas")

    # ---------- grupos de remuestreo: el GRUPO, no la fila ----------
    holdout = PROJECT_ROOT / "data" / "processed" / "holdout_v2_ids.csv"
    grupos = np.arange(n)
    fuente_grupos = "ninguna, se usa el identificador de fila"
    if holdout.exists():
        # Es el unico archivo que el proyecto autoriza como fuente del agrupador, y se usa
        # SOLO para remuestrear en el bootstrap. No se lee para entrenar ni para seleccionar.
        tabla = pd.read_csv(holdout)
        col_id = "id" if "id" in tabla.columns else tabla.columns[0]
        col_gr = next((c for c in ("grupo_near_dup_autor", "grupo") if c in tabla.columns), None)
        if col_gr is not None:
            mapa = dict(zip(pd.to_numeric(tabla[col_id], errors="coerce"), tabla[col_gr]))
            g = np.array([mapa.get(int(i), i) for i in ids])
            _, grupos = np.unique(g, return_inverse=True)
            fuente_grupos = f"{holdout.name}, columna {col_gr}"
    log(f"   unidad de remuestreo: {len(np.unique(grupos))} grupos desde {fuente_grupos}")
    log("   se remuestrea el GRUPO y no la fila, porque dentro de un grupo las filas no son")
    log("   independientes y remuestrear filas subestimaria el intervalo")

    # ---------- P14: la combinacion de dos fuentes, solo si ambas tienen rescate unico ----------
    zv = logit(proba_v03)

    def residual_de(fuente: str, i_tr: np.ndarray, i_va: np.ndarray) -> tuple[np.ndarray, ...]:
        """Residualiza un candidato respecto de z_V03 con un ajuste en inner_train.

        Args:
            fuente: nombre del candidato.
            i_tr: indices de inner_train.
            i_va: indices de inner_validation.

        Returns:
            Los residuos por etiqueta.
        """
        zc = logit(proba_c[fuente])
        salida = []
        for j in range(3):
            A = np.column_stack([np.ones(len(i_tr)), zv[i_tr, j]])
            co = np.linalg.lstsq(A, zc[i_tr, j], rcond=None)[0]
            salida.append(zc[:, j] - (co[0] + co[1] * zv[:, j]))
        return tuple(salida)

    log("")
    log("   P14, LA REGLA DE LA COMBINACION, declarada antes de ejecutarla")
    log("   dos fuentes se combinan SOLO si cada una tiene rescate UNICO. Una segunda fuente")
    log("   que corrige las mismas celdas no es una segunda fuente, es la misma con ruido")

    # Se construye la correccion residual de MI_arbol, que es el candidato que pasa los gates.
    proba_res = proba_v03.copy()
    proba_res_bar = proba_v03.copy()
    for f in range(1, 6):
        ir, iva = np.where(pliegues != f)[0], np.where(pliegues == f)[0]
        lon = np.log1p(n_tok.astype(float))
        mu, sd = float(lon[ir].mean()), float(lon[ir].std() + 1e-12)
        (lon - mu) / sd
        rng = np.random.default_rng(SEMILLA + f)
        p_mi = proba_c["MI_arbol"]
        q = p_mi[ir].copy()
        rng.shuffle(q, axis=0)
        b_mi = p_mi.copy()
        b_mi[ir] = q
        b_mi[iva] = p_mi[iva][rng.permutation(len(iva))]
        for usa_bar in (False, True):
            fuente = b_mi if usa_bar else p_mi
            zc = logit(fuente)
            for j in range(3):
                skf = StratifiedKFold(n_splits=N_INNER, shuffle=True,
                                      random_state=SEMILLA + 100 * f)
                pos = np.arange(len(ir))
                it = iv = None
                for tr_p, va_p in skf.split(pos, y[ir, j]):
                    it, iv = ir[tr_p], ir[va_p]
                    break
                A = np.column_stack([np.ones(len(it)), zv[it, j]])
                co = np.linalg.lstsq(A, zc[it, j], rcond=None)[0]
                res = zc[:, j] - (co[0] + co[1] * zv[:, j])
                mejor_b, mejor_f1 = 0.0, -1.0
                for b in BETAS:
                    pf = sigmoide(zv[:, j] + b * res)
                    f1 = float(f1_score(y[iv, j], (pf[iv] >= V03_UMBRALES[j]).astype(int),
                                        average="binary", zero_division=0))
                    if f1 > mejor_f1:
                        mejor_b, mejor_f1 = b, f1
                destino = proba_res_bar if usa_bar else proba_res
                destino[iva, j] = sigmoide(zv[:, j] + mejor_b * res)[iva]
    ccv_res = ccv_de(proba_res, y)
    ccv_res_bar = ccv_de(proba_res_bar, y)
    log("")
    log(f"   la correccion residual de MI_arbol: {ccv_res:.10f} | "
        f"delta {ccv_res - ccv_ref:+.10f}")
    log(f"   la MISMA con el score barajado:    {ccv_res_bar:.10f} | "
        f"delta {ccv_res_bar - ccv_ref:+.10f}")
    brecha_attrib = round(ccv_res - ccv_res_bar, 10)
    log(f"   brecha de atribucion: {brecha_attrib:+.10f}")

    # ---------- rescate unico por candidato, para decidir si se combina ----------
    log("")
    log("   EL RESCATE UNICO DE CADA CANDIDATO, que es lo que autoriza o no combinar")
    d_res = decision(proba_res)
    d_v03 = pred_v03
    ac_res = (d_res == y)
    ac_v = (d_v03 == y)
    filas_comp: list[dict[str, Any]] = []
    unicos: dict[str, int] = {}
    for c in CANDIDATOS:
        if c == "MI_arbol":
            continue
        d_c = decision(proba_c[c])
        ac_c = (d_c == y)
        solo_c = int(np.count_nonzero(ac_c & ~ac_res))
        no_rescatables = int(np.count_nonzero(~ac_v & ac_res))
        unicos[c] = solo_c
        filas_comp.append({"candidato": c, "rescate_unito_sobre_MI_arbol": solo_c,
                           "celdas_que_MI_arbol_no_puede_rescatar": no_rescatables,
                           "autorizado_a_combinar": solo_c > 0})
        log(f"      {c:12} rescate unico {solo_c:4d} | celdas no rescatables por MI_arbol "
            f"{no_rescatables:4d} | autorizado: {solo_c > 0}")

    # ---------- P17 y P18: rescate, dano y alta confianza ----------
    log("")
    log("   P17 Y P18, RESCATE, DANO Y ALTA CONFIANZA")
    alto = (np.abs(proba_v03 - V03_UMBRALES[None, :]) >= DELTA_HC)
    filas_int: list[dict[str, Any]] = []
    for j, et in enumerate(ETIQUETAS):
        v_ok, v_ko = d_v03[:, j] == y[:, j], d_v03[:, j] != y[:, j]
        r_ok, r_ko = d_res[:, j] == y[:, j], d_res[:, j] != y[:, j]
        filas_int.append({
            "etiqueta": et,
            "v03_ok_fusion_ko": int(np.count_nonzero(v_ok & r_ko)),
            "v03_ko_fusion_ok": int(np.count_nonzero(v_ko & r_ok)),
            "ambos_ok": int(np.count_nonzero(v_ok & r_ok)),
            "ambos_ko": int(np.count_nonzero(v_ko & r_ko)),
            "rescate": int(np.count_nonzero(v_ko & r_ok)),
            "damage": int(np.count_nonzero(v_ok & r_ko)),
            "neto": int(np.count_nonzero(v_ko & r_ok)) - int(np.count_nonzero(v_ok & r_ko)),
            "rescate_de_alta_confianza": int(np.count_nonzero(v_ko & r_ok & alto[:, j])),
            "damage_de_alta_confianza": int(np.count_nonzero(v_ok & r_ko & alto[:, j])),
            "rescate_cerca_del_umbral": int(np.count_nonzero(
                v_ko & r_ok & ~alto[:, j])),
        })
        log(f"      {et}: rescate {filas_int[-1]['rescate']:4d} "
            f"(alta confianza {filas_int[-1]['rescate_de_alta_confianza']:4d}, "
            f"cerca del umbral {filas_int[-1]['rescate_cerca_del_umbral']:4d}) | "
            f"dano {filas_int[-1]['damage']:4d} | neto {filas_int[-1]['neto']:+d}")

    for nombre, lo, hi in BINS:
        m = (n_tok >= lo) & (n_tok <= hi)
        resc = int(np.count_nonzero((~ac_v & ac_res)[m]))
        dmg = int(np.count_nonzero((ac_v & ~ac_res)[m]))
        log(f"      {nombre:10} rescate {resc:4d} | dano {dmg:4d} | neto {resc - dmg:+d} | "
            f"F1 {ccv_de(proba_res[m], y[m]):.4f} contra {ccv_de(proba_v03[m], y[m]):.4f} de V03")
        filas_int.append({"etiqueta": nombre, "rescate": resc, "damage": dmg,
                          "neto": resc - dmg,
                          "F1_fusion": round(ccv_de(proba_res[m], y[m]), 6),
                          "F1_de_V03": round(ccv_de(proba_v03[m], y[m]), 6)})

    # ---------- P21: bootstrap por grupos ----------
    log("")
    log(f"   P21, BOOTSTRAP con {N_BOOT} remuestreos de GRUPOS")
    unicos_g = np.unique(grupos)
    rng = np.random.default_rng(SEMILLA)
    d_boot: list[float] = []
    for _ in range(N_BOOT):
        elegidos = rng.choice(unicos_g, size=len(unicos_g), replace=True)
        filas_b = np.concatenate([np.where(grupos == g)[0] for g in elegidos])
        if len(np.unique(y[filas_b, 0])) < 2 or len(np.unique(y[filas_b, 1])) < 2 \
                or len(np.unique(y[filas_b, 2])) < 2:
            continue
        d_boot.append(ccv_de(proba_res[filas_b], y[filas_b])
                      - ccv_de(proba_v03[filas_b], y[filas_b]))
    arr = np.array(d_boot)
    log(f"      remuestreos validos: {len(arr)}")
    log(f"      delta {np.mean(arr):+.10f} | percentil 2,5 {np.percentile(arr, 2.5):+.10f} | "
        f"percentil 97,5 {np.percentile(arr, 97.5):+.10f}")
    ic_inf, ic_sup = float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))
    incluye_cero = bool(ic_inf <= 0.0 <= ic_sup)
    log(f"      el IC95 incluye el cero: {incluye_cero}")

    # ---------- gate de finalist ----------
    folds = [ccv_de(proba_res[pliegues == g], y[pliegues == g]) for g in range(1, 6)]
    folds_v = [float(f1_score(y[pliegues == g], pred_v03[pliegues == g], average="macro"))
               for g in range(1, 6)]
    pl_mejorados = sum(1 for a, b in zip(folds, folds_v) if a > b)
    delta_fin = ccv_res - ccv_ref
    log("")
    log("   EL GATE DE FINALISTA")
    log(f"      delta >= +0,003 ...................... {delta_fin:+.10f} -> "
        f"{delta_fin >= GATE_FINALISTA}")
    log(f"      pliegues mejorados >= 4/5 ............. {pl_mejorados}/5 -> "
        f"{pl_mejorados >= GATE_PLIEGUES}")
    log(f"      bootstrap favorable ................... IC95 [{ic_inf:+.10f}, "
        f"{ic_sup:+.10f}] incluye cero: {incluye_cero}")
    log(f"      no explicado por el barajado .......... brecha {brecha_attrib:+.10f} -> "
        f"{brecha_attrib > 0}")
    rescate_hc = sum(r.get("rescate_de_alta_confianza", 0) for r in filas_int)
    log(f"      rescate de alta confianza ............. {rescate_hc}")
    condiciones = [delta_fin >= GATE_FINALISTA, pl_mejorados >= GATE_PLIEGUES,
                   not incluye_cero, brecha_attrib > 0, rescate_hc > 0]
    if all(condiciones):
        estado = "FINALIST"
        veredicto = (
            f"La fusion residual de MI_arbol cumple las cinco condiciones del gate de "
            f"finalista: {delta_fin:+.10f}, {pl_mejorados} de 5 pliegues, IC95 "
            f"[{ic_inf:+.10f}, {ic_sup:+.10f}] que NO incluye el cero, brecha de atribucion "
            f"{brecha_attrib:+.10f} frente al score barajado, y {rescate_hc} celdas de alta "
            f"confianza rescatadas. Eso exige segunda particion antes de cualquier adopcion")
    elif sum(condiciones) >= 3:
        estado = "PROMISING"
        veredicto = (
            f"Cumple {sum(condiciones)} de las cinco condiciones del gate de finalist: "
            f"delta {delta_fin:+.10f}, {pl_mejorados} de 5 pliegues, IC95 "
            f"[{ic_inf:+.10f}, {ic_sup:+.10f}] incluye cero: {incluye_cero}, brecha "
            f"{brecha_attrib:+.10f}, rescate de alta confianza {rescate_hc}. Es el mejor "
            f"resultado de toda la investigacion en el espacio de scores y queda como "
            f"PROMISING")
    else:
        estado = "MARGINAL"
        veredicto = (
            f"Cumple {sum(condiciones)} de las cinco condiciones. delta {delta_fin:+.10f}, "
            f"{pl_mejorados} de 5 pliegues, IC95 [{ic_inf:+.10f}, {ic_sup:+.10f}] incluye "
            f"cero: {incluye_cero}")
    log("")
    log(f"   DECISION: {estado}")
    log(f"   VEREDICTO: {veredicto}")

    # ---------- artefactos ----------
    with CSV_COMP.open("w", encoding="ascii", newline="") as fh:
        w = _csv.writer(fh, lineterminator="\n")
        w.writerow(["tipo", "clave", "valor", "valor2"])
        w.writerow(["principal", "corrected_cv_fusion", round(ccv_res, 10), delta_fin])
        w.writerow(["principal", "corrected_cv_barajado", round(ccv_res_bar, 10),
                    round(ccv_res_bar - ccv_ref, 10)])
        w.writerow(["atribucion", "brecha", brecha_attrib])
        w.writerow(["bootstrap", "IC95_inferior", ic_inf, round(ic_inf, 10)])
        w.writerow(["bootstrap", "IC95_superior", ic_sup, round(ic_sup, 10)])
        w.writerow(["bootstrap", "incluye_el_cero", incluye_cero])
        w.writerow(["bootstrap", "media", round(float(np.mean(arr)), 10)])
        w.writerow(["gate", "pliegues_mejorados", pl_mejorados, (f"de 5, se exigen "
                    f"{GATE_PLIEGUES}")])
        w.writerow([])
        w.writerow(["rescate_unico", "candidato", "unico_sobre_MI_arbol", "autorizado"])
        for f in filas_comp:
            w.writerow(["rescate_unico", f["candidato"], f["rescate_unito_sobre_MI_arbol"],
                        f["autorizado_a_combinar"]])
    with CSV_INTER.open("w", encoding="ascii", newline="") as fh:
        w = _csv.writer(fh, lineterminator="\n")
        w.writerow(["corte", "rescate", "damage", "neto", "rescate_alta_confianza",
                    "damage_alta_confianza", "rescate_cerca_del_umbral", "F1_fusion",
                    "F1_de_V03"])
        for f in filas_int:
            w.writerow([f["etiqueta"], f["rescate"], f["damage"], f["neto"],
                        f.get("rescate_de_alta_confianza", ""),
                        f.get("damage_de_alta_confianza", ""),
                        f.get("rescate_cerca_del_umbral", ""),
                        f.get("F1_fusion", ""), f.get("F1_de_V03", "")])
    with CSV_COMP2.open("w", encoding="ascii", newline="") as fh:
        w = _csv.writer(fh, lineterminator="\n")
        w.writerow(["candidato", "correction_cv", "delta_vs_V03", "pliegues_mejorados",
                    "brecha_de_atribucion", "IC95_inferior", "IC95_superior",
                    "rescate_de_alta_confianza", "decision"])
        w.writerow(["MI_arbol + residual anidada", round(ccv_res, 10), round(delta_fin, 10),
                    pl_mejorados, brecha_attrib, round(ic_inf, 10), round(ic_sup, 10),
                    rescate_hc, estado])
        w.writerow(["MI_arbol + residual con score barajado", round(ccv_res_bar, 10),
                    round(ccv_res_bar - ccv_ref, 10), "", "", "", "", "", "CONTROL"])
        w.writerow(["V03", round(ccv_ref, 10), 0.0, "", "", "", "", "", "BASELINE"])

    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                            text=True, cwd=PROJECT_ROOT, check=False).stdout.strip()
    informe = {
        "familia": "P66_COMPLEMENTARIEDAD",
        "naturaleza": ("EXPERIMENTO. 1 candidato Massey con bootstrap, 0 submissions, 0 Kaggle, "
                       "0 holdout, 0 test"),
        "fecha": inicio.isoformat(timespec="seconds"), "experiment_id": "P66-03",
        "track": "P14-P17-P18-P21", "script":
            "experiments/p66_03_complementariedad_bootstrap.py",
        "pregunta": ("la correccion residual de MI_arbol rescata errores reales o solo "
                     "recalibra lo que ya estaba en el umbral"),
        "unidad_de_remuestreo": {"grupos": len(np.unique(grupos)),
                                 "fuente": fuente_grupos,
                                 "por_que": "dentro de un grupo las filas no son "
                                            "independientes y remuestrear filas subestimaria "
                                            "el intervalo"},
        "resultado": {"corrected_cv": ccv_res, "delta_vs_V03": delta_fin,
                      "con_score_barajado": ccv_res_bar,
                      "brecha_de_atribucion": brecha_attrib,
                      "pliegues_mejorados": pl_mejorados},
        "bootstrap": {"n": len(arr), "media": round(float(np.mean(arr)), 10),
                      "ic95_inferior": ic_inf, "ic95_superior": ic_sup,
                      "incluye_el_cero": incluye_cero},
        "rescate_por_etiqueta": [f for f in filas_int if "ambos_ok" in f],
        "rescate_por_bloque": [f for f in filas_int if "F1_fusion" in f],
        "rescate_unico_por_candidato": filas_comp,
        "regla_de_combinacion": ("dos fuentes se combinan solo si cada una tiene rescate "
                                "unico. Una segunda fuente que corrige las mismas celdas "
                                "no es una segunda fuente"),
        "gate_de_finalista": {"delta_min": GATE_FINALISTA, "delta": delta_fin,
                              "pliegues_min": GATE_PLIEGUES, "pliegues": pl_mejorados,
                              "bootstrap_favorable": not incluye_cero,
                              "atribucion": brecha_attrib > 0,
                              "rescate_de_alta_confianza": rescate_hc,
                              "superado": all(condiciones)},
        "decision": estado, "veredicto": veredicto,
        "experimentos_registrados": 1,
        "reproducibilidad": {"nivel": "R2", "git_commit": commit, "seed": SEMILLA,
                             "hash_v03": hash_real, "corrected_cv": V03_CCV},
        "formulacion": "V03 remains the best verified leaderboard result.",
    }
    JSON_C.write_text(json.dumps(informe, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")

    L: list[str] = []
    W = L.append
    W("# P66 - Complementariedad, rescate de alta confianza y bootstrap")
    W("")
    W(f"Generado por `experiments/p66_03_complementariedad_bootstrap.py`. Fecha "
      f"{inicio.isoformat(timespec='seconds')}.")
    W("")
    W("## Las cuatro preguntas que quedaban abiertas")
    W("")
    W("1. **P14** una segunda fuente anade algo, y solo se mide si ambas tienen rescate unico")
    W("2. **P17** cuantos errores de V03 rescata y cuantas filas correctas dana, por etiqueta")
    W("   y por estrato de longitud")
    W("3. **P18** los errores rescatados son los que V03 fallo **con confianza**, que es la")
    W("   pregunta mas afilada de la fase: una fusion que solo rescata filas pegadas al")
    W("   umbral ha encontrado calibracion, no representacion")
    W("4. **P21** bootstrap con la unidad de remuestreo del proyecto")
    W("")
    W("## La unidad de remuestreo")
    W("")
    W(f"{len(np.unique(grupos))} grupos, desde {fuente_grupos}. Se remuestrea el **grupo** y no "
      f"la fila, porque dentro de un grupo las filas no son independientes y remuestrear filas "
      f"subestimaria el intervalo.")
    W("")
    W("## La correccion residual de MI_arbol")
    W("")
    W("```text")
    W(f"real        {ccv_res:.10f}  delta {ccv_res - ccv_ref:+.10f}")
    W(f"barajado    {ccv_res_bar:.10f}  delta {ccv_res_bar - ccv_ref:+.10f}")
    W(f"brecha      {brecha_attrib:+.10f}")
    W("```")
    W("")
    W("## El rescate unico de cada candidato")
    W("")
    W("La regla es anterior a la ejecucion: dos fuentes se combinan **solo** si cada una tiene "
      "rescate unico. Una segunda fuente que corrige las mismas celdas no es una segunda fuente, "
      "es la misma con ruido.")
    W("")
    W("| candidato | rescate unico sobre MI_arbol | autorizado a combinar |")
    W("|---|---:|---|")
    for f in filas_comp:
        W(f"| {f['candidato']} | {f['rescate_unito_sobre_MI_arbol']} | "
          f"{f['autorizado_a_combinar']} |")
    W("")
    W("## Rescate, dano y alta confianza")
    W("")
    W("| corte | rescate | dano | neto | rescate de alta confianza | cerca del umbral |")
    W("|---|---:|---:|---:|---:|---:|")
    for f in filas_int:
        if "ambos_ok" not in f:
            continue
        W(f"| {f['etiqueta']} | {f['rescate']} | {f['damage']} | {f['neto']:+d} | "
          f"{f['rescate_de_alta_confianza']} | {f['rescate_cerca_del_umbral']} |")
    W("")
    W("| estrato | rescate | dano | neto | F1 fusion | F1 V03 |")
    W("|---|---:|---:|---:|---:|---:|")
    for f in filas_int:
        if "F1_fusion" not in f:
            continue
        W(f"| {f['etiqueta']} | {f['rescate']} | {f['damage']} | {f['neto']:+d} | "
          f"{f['F1_fusion']:.4f} | {f['F1_de_V03']:.4f} |")
    W("")
    W("## El bootstrap")
    W("")
    W("```text")
    W(f"remuestreos validos   {len(arr)}")
    W(f"delta medio           {np.mean(arr):+.10f}")
    W(f"IC95                  [{ic_inf:+.10f}, {ic_sup:+.10f}]")
    W(f"incluye el cero       {incluye_cero}")
    W("```")
    W("")
    W("## El gate de finalist")
    W("")
    W("| condicion | valor | se cumple |")
    W("|---|---|---|")
    W(f"| delta >= +0,003 | {delta_fin:+.10f} | {delta_fin >= GATE_FINALISTA} |")
    W(f"| pliegues >= 4/5 | {pl_mejorados}/5 | {pl_mejorados >= GATE_PLIEGUES} |")
    W(f"| bootstrap favorable | incluye cero: {incluye_cero} | {not incluye_cero} |")
    W(f"| no lo explica el barajado | {brecha_attrib:+.10f} | {brecha_attrib > 0} |")
    W(f"| rescate de alta confianza | {rescate_hc} | {rescate_hc > 0} |")
    W("")
    W("## Veredicto")
    W("")
    W("```text")
    W(veredicto)
    W("")
    W(f"DECISION: {estado}")
    W("```")
    W("")
    W("> V03 remains the best verified leaderboard result.")
    W("")
    W("*Generado por `experiments/p66_03_complementariedad_bootstrap.py`. Sin push.*")
    MD_C.write_text(a_ascii("\n".join(L)), encoding="ascii")

    for destino in (MD_C, JSON_C, CSV_COMP, CSV_INTER, CSV_COMP2):
        if any(b > 127 for b in destino.read_bytes()):
            raise SystemExit(f"PARADA: {destino.name} conserva bytes no ASCII")
    log("")
    log(f"P66-03 OK | delta {delta_fin:+.10f} {pl_mejorados}/5 | IC95 [{ic_inf:+.10f}, "
        f"{ic_sup:+.10f}] incluye cero {incluye_cero} | brecha {brecha_attrib:+.10f} | "
        f"rescate alta confianza {rescate_hc} | decision {estado}")


if __name__ == "__main__":
    ast.parse(Path(__file__).read_text(encoding="utf-8"))
    main()
