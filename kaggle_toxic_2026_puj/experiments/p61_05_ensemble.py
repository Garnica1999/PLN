"""P61 track: small-ensemble weight and membership optimisation, fully nested.

The flaw this script was rewritten for
--------------------------------------
A first version nested the WEIGHTS but not the MEMBERSHIP. The three historical
members were chosen once, globally, by a diversity-adjusted utility computed over the
whole out-of-fold, and only then were the weights optimised per fold. That is fitting
on the outer data: the question "who belongs in the pool" was answered using the same
rows the pool is then scored on. Nesting the weights does not repair that, because the
set of candidates itself carried the leak.

So both decisions are now made inside every outer fold, from the internal rows only:

    for each outer fold f:
        internal rows  = the other four folds
        rank the historical candidates by utility computed on THOSE rows
        keep the top three
        optimise the weights on THOSE rows
        apply the resulting pool and weights to fold f

and the score is assembled from the five held-out folds. Nothing about membership or
weight is chosen using a row it is scored on.

Why the weights need a derivative-free search
--------------------------------------------
The permitted objective is macro-F1 at frozen thresholds, and that function is not
continuous in the weights: with a fixed threshold the decision is piecewise constant
and so is the F1, so its gradient is zero almost everywhere. A first version used SLSQP
and returned weights exactly uniform, 1 over k, at every size and under both
objectives, because the optimiser saw a null gradient and stopped at its starting
point. Those uniform weights were an artefact of the method, not a solution.

The search used instead is deterministic and derivative-free: the uniform point, the k
unit vertices, and N_PUNTOS_SIMPLEJO Dirichlet(1) draws with a fixed seed. The uniform
point is always a candidate, so the result can never be worse than the plain average,
and the seeded generator makes it reproducible.

The attribution control, and why it matters more than the headline
-----------------------------------------------------------------
The pool already contains V03, so a positive result cannot be attributed without
comparing the pool with and without the new model. Paragraph 33 requires it, and here
it turned out to be the decisive measurement: the historical members alone do better
than the pool that includes LightGBM. So the new model subtracts, and reporting only
the full-pool number would have been misleading.

Usage
-----
    /home/carlos/venv/bin/python experiments/p61_05_ensemble.py
"""

from __future__ import annotations

import ast
import hashlib
import json
import subprocess
import sys
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
FIG_P61 = PROJECT_ROOT / "figuras" / "p61"
OOF = PROJECT_ROOT / "features" / "oof_v03_final.npz"
POOL_HIST = PROJECT_ROOT / "features" / "p50_celda_features.npz"
CELDA_LGB = PROJECT_ROOT / "features" / "p61_lightgbm_cells.npz"
MD_ENS = REPORTS / "p61_ensemble.md"
JSON_ENS = REPORTS / "p61_ensemble.json"
CSV_ENS = REPORTS / "p61_ensemble_results.csv"
CSV_PESOS = REPORTS / "p61_ensemble_weights.csv"
CSV_POOL = REPORTS / "p61_ensemble_pool_selection.csv"
CELDA_NPZ = PROJECT_ROOT / "features" / "p61_ensemble_cells.npz"

V03_UMBRALES = np.array([0.44, 0.51, 0.46])
SEMILLA = 42
N_HISTORICOS = 3
LAMBDA_ENTROPIA = 0.002
N_PUNTOS_SIMPLEJO = 300
GATE_DELTA = 0.003
GATE_PLIEGUES = 4
N_BOOT = 2000
MAX_MEMBROS = 6


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[P61-05] {mensaje}", flush=True)


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


def f1_de_pesos(w: np.ndarray, probas: np.ndarray, y: np.ndarray,
                filas: np.ndarray) -> float:
    """Macro-F1 de un conjunto de pesos sobre un subconjunto de filas.

    Args:
        w: vector de pesos.
        probas: probabilidades de cada miembro, forma (k, n, 3).
        y: etiquetas enteras.
        filas: indices de filas a usar.

    Returns:
        El macro-F1 resultante.
    """
    p = np.tensordot(w, probas[:, filas, :], axes=(0, 0))
    return float(f1_score(y[filas], (p >= V03_UMBRALES[None, :]).astype(int),
                          average="macro"))


def entropia(w: np.ndarray) -> float:
    """Entropia normalizada de un vector de pesos en el simplejo.

    Args:
        w: vector de pesos.

    Returns:
        La entropia dividida por su maximo, entre cero y uno.
    """
    return float(-np.sum(w * np.log(np.maximum(w, 1e-12))) / np.log(max(len(w), 2)))


def optimizar_pesos(probas: np.ndarray, y: np.ndarray, filas: np.ndarray,
                    objetivo: str, k: int) -> np.ndarray:
    """Optimiza los pesos sobre el simplejo con un metodo sin derivadas.

    Args:
        probas: probabilidades de cada miembro, forma (k, n, 3).
        y: etiquetas enteras.
        filas: indices de las filas internas que se pueden usar.
        objetivo: "A" para macro-F1, "B" para macro-F1 mas entropia.
        k: numero de miembros.

    Returns:
        Vector de pesos que suma uno.
    """
    rng = np.random.default_rng(SEMILLA + 7919 * k + (0 if objetivo == "A" else 1))
    candidatos: list[np.ndarray] = [np.full(k, 1.0 / k)]
    for i in range(k):
        v = np.zeros(k)
        v[i] = 1.0
        candidatos.append(v)
    for _ in range(N_PUNTOS_SIMPLEJO):
        candidatos.append(rng.dirichlet(np.ones(k)))

    def valor(w: np.ndarray) -> float:
        """Puntua un vector de pesos con el objetivo elegido.

        Args:
            w: vector de pesos.

        Returns:
            El valor a maximizar.
        """
        base = f1_de_pesos(w, probas, y, filas)
        if objetivo == "A":
            return base
        return base + LAMBDA_ENTROPIA * entropia(w)

    mejor = max(candidatos, key=valor)
    s = float(mejor.sum())
    return mejor / s if s > 0 else np.full(k, 1.0 / k)


def utilidad_ajustada(pred: np.ndarray, pred_ref: np.ndarray, y: np.ndarray,
                      filas: np.ndarray) -> float:
    """Utilidad ajustada por diversidad, calculada SOLO sobre las filas internas.

    La forma es la misma que la del inventario, rescate menos dano mas una penalizacion
    por correlacion de error, pero calculado sobre el subconjunto de filas que la
    funcion recibe. Esto es lo que hace que la SELECCION de pertenencia al pool sea
    anidada y no global.

    Args:
        pred: predicciones del candidato, forma (n, 3).
        pred_ref: predicciones de referencia, forma (n, 3).
        y: etiquetas enteras.
        filas: indices de filas internas.

    Returns:
        La utilidad ajustada por diversidad.
    """
    a = (pred[filas] == y[filas]).ravel().astype(float)
    b = (pred_ref[filas] == y[filas]).ravel().astype(float)
    resc = float(np.count_nonzero((a == 1) & (b == 0)))
    dan = float(np.count_nonzero((a == 0) & (b == 1)))
    if a.std() > 0 and b.std() > 0:
        corr = float(np.corrcoef(a, b)[0, 1])
    else:
        corr = 0.0
    return resc - dan + 0.5 * (1.0 - max(min(corr, 1.0), -1.0))


def nativo(valor: Any) -> Any:
    """Convierte tipos de numpy en tipos de python, para que el JSON sea serializable.

    Los `.to_dict(orient="records")` de pandas devuelven int64 y float64 nativos, que
    el codificador JSON rechaza. Convertirlos aqui evita tener que castear cada campo
    en el diccionario, y evita el fallo silencioso de que un campo se pierda.

    Args:
        valor: valor a convertir, posiblemente anidado.

    Returns:
        El mismo valor con escalares de python.
    """
    if isinstance(valor, dict):
        return {k: nativo(v) for k, v in valor.items()}
    if isinstance(valor, (list, tuple)):
        return [nativo(v) for v in valor]
    if isinstance(valor, np.generic):
        return valor.item()
    if isinstance(valor, float) and not np.isfinite(valor):
        return None
    return valor


def main() -> None:
    """Ejecuta la optimizacion anidada de pool y pesos, con control de atribucion.

    Returns:
        None.
    """
    inicio = datetime.now(timezone.utc)
    log("P61-05 optimizacion ANIDADA de PERTENENCIA y PESOS, con control de atribucion")
    log(f"   {N_HISTORICOS} historicos | {N_PUNTOS_SIMPLEJO} puntos en el simplejo | "
        f"lambda de entropia {LAMBDA_ENTROPIA}")
    FIG_P61.mkdir(parents=True, exist_ok=True)

    hash_real = hashlib.sha256(V03_SUB.read_bytes()).hexdigest()
    if hash_real != V03_SHA:
        raise SystemExit("PARADA: la submission de V03 no coincide con el sha256 congelado")
    log(f"   V03 sha256 verificado e intacto: {hash_real[:16]}...")

    v = np.load(OOF, allow_pickle=True)
    y, pliegues, pred_v03 = v["y_true"].astype(int), v["pliegues"], v["pred"].astype(int)
    proba_v03 = v["proba"]
    n = y.shape[0]
    ccv_ref = float(f1_score(y, pred_v03, average="macro"))
    if ccv_ref != V03_CCV:
        raise SystemExit(f"PARADA: el OOF canonico da {ccv_ref!r}, no {V03_CCV!r}")

    # ---------- los candidatos historicos disponibles ----------
    z = np.load(POOL_HIST, allow_pickle=True)
    if not bool(np.array_equal(z["ids"], v["ids"])):
        raise SystemExit("PARADA: el pool historico no coincide en ids con el OOF canonico")
    if not bool(np.array_equal(z["pliegue"] + 1, pliegues)):
        raise SystemExit("PARADA: el pool historico no usa la misma particion")
    nombres = [str(s) for s in z["nombres"]]
    cprob = z["canal_proba"].astype(np.float64)
    ccv_canal = np.array([f1_score(y, (cprob[i] >= V03_UMBRALES[None, :]).astype(int),
                                 average="macro") for i in range(len(nombres))])
    es_v03 = [bool(ccv_canal[i] == ccv_ref
                   and int(np.count_nonzero(
                       (cprob[i] >= V03_UMBRALES[None, :]).astype(int) != pred_v03)) == 0)
              for i in range(len(nombres))]
    cands = [i for i in range(len(nombres)) if not es_v03[i]]
    log("")
    log("   CANDIDATOS HISTORICOS DISPONIBLES")
    log(f"      {len(nombres)} canales, {sum(es_v03)} es V03, {len(cands)} candidatos")
    log(f"      mejor corrected_cv entre candidatos: "
        f"{max(ccv_canal[i] for i in cands):.6f} contra V03 {ccv_ref:.6f}")

    proba_lgb = None
    if CELDA_LGB.exists():
        zz = np.load(CELDA_LGB, allow_pickle=True)
        if bool(np.array_equal(zz["ids"], v["ids"])) and bool(
                np.array_equal(zz["y_true"], y)) and "proba_lgbm_optuna" in zz.files:
            p = zz["proba_lgbm_optuna"]
            if p.shape == (n, 3) and bool(np.all(np.isfinite(p))):
                proba_lgb = p
                log(f"      modelo nuevo disponible: LightGBM+Optuna, corrected_cv "
                    f"{f1_score(y, (p >= V03_UMBRALES[None, :]).astype(int), average='macro'):.6f}")
    if proba_lgb is None:
        log("      NO hay modelo nuevo utilizable: el pool sera solo historico")

    # ---------- barrido completamente anidado ----------
    log("")
    log("   BARRIDO ANIDADO: pertenencia y pesos se deciden dentro de cada pliegue")
    configs: list[tuple[str, str, int, str]] = []
    for etiqueta_pool in ("con_nuevo", "sin_nuevo"):
        base = 1 + (1 if (etiqueta_pool == "con_nuevo" and proba_lgb is not None) else 0)
        for k in range(1, base + N_HISTORICOS + 1):
            for objetivo in ("A", "B"):
                configs.append((etiqueta_pool, etiqueta_pool, k, objetivo))
    # se eliminan los tamanos imposibles del pool sin el modelo nuevo
    max_sin = 1 + N_HISTORICOS
    configs = [c for c in configs
               if not (c[0] == "sin_nuevo" and c[2] > max_sin)]

    filas: list[dict[str, Any]] = []
    probs: dict[str, np.ndarray] = {}
    sel_log: list[dict[str, Any]] = []
    t0 = time.time()
    for etiqueta_pool, _, k, objetivo in configs:
        clave = f"{etiqueta_pool}_k{k}_{objetivo}"
        proba_ens = np.zeros((n, 3))
        pesos_por_fold: list[list[float]] = []
        nombres_por_fold: list[list[str]] = []
        for f in range(1, 6):
            interna = np.where(pliegues != f)[0]
            externa = np.where(pliegues == f)[0]
            # pertenencia ELEGIDA CON LAS FILAS INTERNAS
            ranked = sorted(cands, key=lambda i: -utilidad_ajustada(
                (cprob[i] >= V03_UMBRALES[None, :]).astype(int), pred_v03, y, interna))
            elegidos = ranked[:min(N_HISTORICOS, k - 1)] if k > 1 else []
            pool_nombres = ["V03"]
            pool_probas = [proba_v03]
            if etiqueta_pool == "con_nuevo" and proba_lgb is not None and k >= 2:
                pool_nombres.append("LightGBM+Optuna")
                pool_probas.append(proba_lgb)
            for pos, i in enumerate(elegidos[:max(k - len(pool_nombres), 0)]):
                pool_nombres.append(nombres[i])
                pool_probas.append(cprob[i])
            while len(pool_nombres) < k and ranked:
                i = ranked[len(pool_nombres) - 1]
                if nombres[i] in pool_nombres:
                    break
                pool_nombres.append(nombres[i])
                pool_probas.append(cprob[i])
            ph = np.stack(pool_probas[:k])
            w = optimizar_pesos(ph, y, interna, objetivo, len(ph))
            pesos_por_fold.append([round(float(x), 6) for x in w])
            nombres_por_fold.append(pool_nombres[:k])
            proba_ens[externa] = np.tensordot(w, ph[:, externa, :], axes=(0, 0))
            if clave.endswith("_k5_A") or clave == f"{etiqueta_pool}_k{N_HISTORICOS + 2}_A":
                sel_log.append({"configuracion": clave, "pliegue": f,
                                "miembros": " | ".join(pool_nombres[:k]),
                                "utilidades": {
                                    nombres[i]: round(utilidad_ajustada(
                                        (cprob[i] >= V03_UMBRALES[None, :]).astype(int),
                                        pred_v03, y, interna), 2) for i in elegidos}})
        pred_ens = (proba_ens >= V03_UMBRALES[None, :]).astype(int)
        ccv = float(f1_score(y, pred_ens, average="macro"))
        resc = int(np.count_nonzero((pred_ens == y) & (pred_v03 != y)))
        dan = int(np.count_nonzero((pred_ens != y) & (pred_v03 == y)))
        folds_ens = [float(f1_score(y[pliegues == g], pred_ens[pliegues == g],
                                    average="macro")) for g in range(1, 6)]
        probs[clave] = proba_ens
        filas.append({
            "ID": clave, "Track": "P61-ensemble", "pool": etiqueta_pool, "k": k,
            "objetivo": objetivo,
            "miembros_medios": " | ".join(nombres_por_fold[0]),
            "Corrected_CV": round(ccv, 10),
            "Fold_1": round(folds_ens[0], 6), "Fold_2": round(folds_ens[1], 6),
            "Fold_3": round(folds_ens[2], 6), "Fold_4": round(folds_ens[3], 6),
            "Fold_5": round(folds_ens[4], 6),
            "Delta_vs_V03": round(ccv - ccv_ref, 10),
            "Rescue": resc, "Damage": dan, "Net": resc - dan,
            "cambian": int(np.count_nonzero(pred_ens != pred_v03)),
            "folds_mejorados": None, "gate_superado": None, "Bootstrap_CI": None,
            "pesos_por_fold": json.dumps(pesos_por_fold, ensure_ascii=False),
            "miembros_por_fold": json.dumps(nombres_por_fold, ensure_ascii=False),
        })
        media = np.mean(np.array(pesos_por_fold, dtype=float), axis=0)
        log(f"      {clave:22} ccv {ccv:.10f} delta {ccv - ccv_ref:+.10f} | resc {resc:4d} "
            f"dan {dan:4d} neto {resc - dan:+4d} | pesos {np.round(media, 3).tolist()}")
    log(f"      barrido completo en {time.time() - t0:.1f} s")
    if sel_log:
        pd.DataFrame(sel_log).to_csv(CSV_POOL, index=False, encoding="utf-8")
        log("      escrito reports/p61_ensemble_pool_selection.csv, que muestra como cambia")
        log("      la pertenencia al pool al excluir cada pliegue")

    df = pd.DataFrame(filas)
    folds_v03 = [float(f1_score(y[pliegues == g], pred_v03[pliegues == g], average="macro"))
                 for g in range(1, 6)]

    # ---------- gate y bootstrap para cada configuracion ----------
    log("")
    log("   GATE Y BOOTSTRAP POR CONFIGURACION")
    rng_b = np.random.default_rng(SEMILLA)
    for idx, r in df.iterrows():
        pk = (probs[r["ID"]] >= V03_UMBRALES[None, :]).astype(int)
        fe = [float(f1_score(y[pliegues == g], pk[pliegues == g], average="macro"))
              for g in range(1, 6)]
        fm = sum(1 for u, w in zip(fe, folds_v03) if u > w)
        d = float(r["Corrected_CV"]) - ccv_ref
        sup = bool(d >= GATE_DELTA and fm >= GATE_PLIEGUES)
        df.at[idx, "folds_mejorados"] = fm
        df.at[idx, "gate_superado"] = sup
        if sup:
            dc = ((pk == y).astype(int) - (pred_v03 == y).astype(int)).ravel()
            mm = dc[rng_b.integers(0, dc.size, size=(N_BOOT, dc.size))].mean(axis=1)
            lo, hi = float(np.percentile(mm, 2.5)), float(np.percentile(mm, 97.5))
            df.at[idx, "Bootstrap_CI"] = f"IC95 [{lo:+.6f}, {hi:+.6f}]"
            log(f"      {r['ID']:22} delta {d:+.6f} | pliegues {fm}/5 | GATE SUPERADO | "
                f"IC95 [{lo:+.6f}, {hi:+.6f}]")
        else:
            df.at[idx, "Bootstrap_CI"] = "NO EJECUTADO (gate no superado)"
            log(f"      {r['ID']:22} delta {d:+.6f} | pliegues {fm}/5 | gate no superado")

    calificadas = df[df["gate_superado"]]
    if calificadas.empty:
        log("")
        log("   NINGUNA configuracion supera el gate de +0,003 con 4/5 pliegues")
        mejor = df.sort_values("Corrected_CV", ascending=False).iloc[0]
    else:
        mejor = calificadas.sort_values("Corrected_CV", ascending=False).iloc[0]
    log("")
    log(f"   MEJOR: {mejor['ID']} con corrected_cv {mejor['Corrected_CV']:.10f} "
        f"y delta {float(mejor['Delta_vs_V03']):+.10f}")

    # ---------- control de atribucion explicito ----------
    log("")
    log("   CONTROL DE ATRIBUCION: que aporta el modelo NUEVO")
    if proba_lgb is None:
        atribucion = ("no hay modelo nuevo en el pool, luego el control no aplica")
        log(f"      {atribucion}")
    else:
        con = df[df["pool"] == "con_nuevo"].sort_values("Corrected_CV", ascending=False)
        sin = df[df["pool"] == "sin_nuevo"].sort_values("Corrected_CV", ascending=False)
        c_mejor, s_mejor = con.iloc[0], sin.iloc[0]
        aporte = float(c_mejor["Corrected_CV"]) - float(s_mejor["Corrected_CV"])
        log(f"      mejor con el modelo nuevo:    {c_mejor['ID']:22} "
            f"{float(c_mejor['Corrected_CV']):.10f} "
            f"delta {float(c_mejor['Delta_vs_V03']):+.10f}")
        log(f"      mejor sin el modelo nuevo:    {s_mejor['ID']:22} "
            f"{float(s_mejor['Corrected_CV']):.10f} "
            f"delta {float(s_mejor['Delta_vs_V03']):+.10f}")
        log(f"      APORTE DEL MODELO NUEVO:       {aporte:+.10f}")
        if aporte <= 0:
            atribucion = (
                f"el modelo nuevo SUBTRAA {aporte:+.10f}. El mejor resultado se obtiene sin "
                f"el, luego la mejora no es atribuible a la capacidad de LightGBM sino al "
                f"promediado de modelos historicos diversos que ya existian"
            )
            log(f"      VEREDICTO: {atribucion}")
        else:
            atribucion = (f"el modelo nuevo aporta {aporte:+.10f}, luego su contribucion es "
                          f"positiva pero pequena")
            log(f"      VEREDICTO: {atribucion}")

    # ---------- desplazamiento de probabilidad frente a decision ----------
    log("")
    log("   DESPLAZAMIENTO DE PROBABILIDAD FRENTE A DESPLAZAMIENTO DE DECISION")
    p_ens = probs[mejor["ID"]]
    pred_ens = (p_ens >= V03_UMBRALES[None, :]).astype(int)
    cambios_p = float(np.abs(p_ens - proba_v03).mean())
    cambios_d = int(np.count_nonzero(pred_ens != pred_v03))
    dist_v03 = np.abs(proba_v03 - V03_UMBRALES[None, :])
    dist_ens = np.abs(p_ens - V03_UMBRALES[None, :])
    log(f"      desplazamiento medio de probabilidad {cambios_p:.6f} | decisiones que "
        f"cambian {cambios_d} de {n * 3}")
    log(f"      distancia media al umbral: V03 {dist_v03.mean():.6f} -> ensemble "
        f"{dist_ens.mean():.6f}")
    n_aleja = int(np.count_nonzero((pred_ens != pred_v03) & (dist_ens > dist_v03)))
    log(f"      celdas que cambian Y se alejan del umbral: {n_aleja}")
    ratio_celdas = (cambios_d / max(n * 3, 1)) / max(cambios_p, 1e-9)
    if cambios_d == 0:
        lectura = "ninguna decision cambia, luego el efecto es de probabilidad y no de decision"
    elif ratio_celdas > 3.0:
        lectura = ("las decisiones cambian proporcionalmente mas de lo que se mueven las "
                   "probabilidades, luego el efecto es de proximidad al umbral")
    elif cambios_p > 0.05:
        lectura = "las probabilidades se mueven mucho y las decisiones pocas, luego el efecto es de probabilidad"
    else:
        lectura = "desplazamiento repartido entre probabilidad y decision"
    log(f"      celdas que cambian por unidad de desplazamiento: {ratio_celdas:.2f}")
    log(f"      lectura: {lectura}")

    # ---------- figuras ----------
    fig, ax = plt.subplots(figsize=(11, 5))
    etiq = [f"{r.pool}\nk={r.k}\n{r.objetivo}" for r in df.itertuples()]
    ax.bar(range(len(df)), df["Corrected_CV"],
           color=["tab:blue" if p == "con_nuevo" else "tab:green" for p in df["pool"]])
    ax.axhline(ccv_ref, color="black", ls="--", lw=1.4, label=f"V03 {ccv_ref:.6f}")
    for i, r in df.reset_index(drop=True).iterrows():
        ax.text(i, r["Corrected_CV"] + 0.0004, f"{r['Delta_vs_V03']:+.5f}", ha="center",
                fontsize=6.5, rotation=90)
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(etiq, fontsize=6.5)
    ax.set_ylim(min(df["Corrected_CV"]) - 0.004, max(df["Corrected_CV"]) + 0.004)
    ax.set_ylabel("corrected_cv")
    ax.set_title("Ensemble anidado de pertenencia y pesos, frente a V03")
    ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(FIG_P61 / "11_ensemble_cv_vs_v03.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    mejor_id = mejor["ID"]
    pf = json.loads(df.loc[df["ID"] == mejor_id, "pesos_por_fold"].iloc[0])
    nmf = json.loads(df.loc[df["ID"] == mejor_id, "miembros_por_fold"].iloc[0])
    pesos_medios = np.array(pf, dtype=float).mean(axis=0)
    nombres_medios = nmf[0]
    ax.bar(range(len(pesos_medios)), pesos_medios, color="tab:blue")
    ax.set_xticks(range(len(pesos_medios)))
    ax.set_xticklabels([s[:18] for s in nombres_medios], rotation=25, ha="right", fontsize=7)
    ax.set_ylabel("peso medio anidado")
    ax.set_title(f"Pesos del ensemble {mejor_id}")
    plt.tight_layout()
    fig.savefig(FIG_P61 / "12_ensemble_weights.png", dpi=130)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for j, e in enumerate(ETIQUETAS):
        axes[j].hist(proba_v03[:, j], bins=40, density=True, alpha=0.5, label="V03",
                     color="tab:blue")
        axes[j].hist(p_ens[:, j], bins=40, density=True, alpha=0.5,
                     label=f"ensemble {mejor_id}", color="tab:orange")
        axes[j].axvline(V03_UMBRALES[j], color="black", ls="--", lw=1.2)
        axes[j].set_title(f"{e}: V03 frente al ensemble")
        axes[j].legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(FIG_P61 / "13_v03_vs_candidate_probability_shift.png", dpi=130)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    ac_v = (pred_v03 == y)
    ac_c = (pred_ens == y)
    [int(np.count_nonzero(ac_v & ac_c)), int(np.count_nonzero(ac_v & ~ac_c)),
                  int(np.count_nonzero(~ac_v & ac_c)), int(np.count_nonzero(~ac_v & ~ac_c))]
    etiquetas_cat = ["V03 bien / ens bien", "V03 bien / ens mal", "V03 mal / ens bien",
                     "ambos mal"]
    for j, e in enumerate(ETIQUETAS):
        cats = [int(np.count_nonzero(ac_v[:, j] & ac_c[:, j])),
                int(np.count_nonzero(ac_v[:, j] & ~ac_c[:, j])),
                int(np.count_nonzero(~ac_v[:, j] & ac_c[:, j])),
                int(np.count_nonzero(~ac_v[:, j] & ~ac_c[:, j]))]
        axes[j].bar(etiquetas_cat, cats, color=["tab:green", "tab:red", "tab:orange",
                                                 "tab:gray"])
        axes[j].tick_params(axis="x", rotation=35, labelsize=6.5)
        axes[j].set_title(f"{e}: desglose de celdas")
    plt.tight_layout()
    fig.savefig(FIG_P61 / "14_error_breakdown_v03_vs_candidate.png", dpi=130)
    plt.close(fig)
    log("      figuras 11, 12, 13 y 14")

    # ---------- escritura ----------
    df_out = df.copy()
    for col in ("pesos_por_fold", "miembros_por_fold"):
        df_out[col] = df_out[col].map(lambda s: a_ascii(str(s)[:400]))
    df_out.to_csv(CSV_ENS, index=False, encoding="utf-8")
    filas_pesos = []
    for r in df.itertuples():
        pf_r = json.loads(r.pesos_por_fold)
        nm_r = json.loads(r.miembros_por_fold)
        for f, (w, noms) in enumerate(zip(pf_r, nm_r), start=1):
            for nom, val in zip(noms, w):
                filas_pesos.append({"ID": r.ID, "pool": r.pool, "objetivo": r.objetivo,
                                    "k": r.k, "pliegue": f, "miembro": nom, "peso": val})
    pd.DataFrame(filas_pesos).to_csv(CSV_PESOS, index=False, encoding="utf-8")
    np.savez_compressed(CELDA_NPZ, ids=v["ids"], y_true=y, pliegues=pliegues,
                        pred_v03=pred_v03, pred_ensemble=pred_ens,
                        proba_v03=proba_v03, proba_ensemble=p_ens)
    log("   escrito reports/p61_ensemble_results.csv, p61_ensemble_weights.csv y "
        "p61_ensemble_pool_selection.csv")

    decision = ("PROMISING. Supera el gate; exige segunda particion" if mejor["gate_superado"]
                else ("MARGINAL. Mejora pero no alcanza el gate de +0,003 con 4/5 pliegues"
                      if float(mejor["Corrected_CV"]) > ccv_ref else
                      "CLOSED. El ensemble anidado no supera a V03"))

    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                            text=True, cwd=PROJECT_ROOT, check=False).stdout.strip()
    informe = {
        "familia": "P61_ENSEMBLE",
        "naturaleza": ("EXPERIMENTO. 0 submissions, 0 Kaggle, 0 holdout, 0 test. "
                       "V03 no se modifica"),
        "fecha": inicio.isoformat(timespec="seconds"),
        "experiment_id": "P61-02", "track": "P61-ensemble",
        "script": "experiments/p61_05_ensemble.py",
        "defecto_corregido_antes_de_aceptar_el_resultado": (
            "una primera version anidaba los PESOS pero no la PERTENENCIA al pool: los tres "
            "historicos se elegian una sola vez, de forma global, por una utilidad ajustada "
            "por diversidad calculada sobre todo el OOF. Eso es ajustar sobre los datos "
            "externos, porque la pregunta de quien pertenece al pool se respondia con las "
            "mismas filas sobre las que despues se puntua. Anidar los pesos no repara eso, "
            "porque el conjunto de candidatos ya arrastraba la fuga. La version vigente "
            "hace la SELECCION de pertenencia tambien dentro de cada pliegue, con las filas "
            "internas, y por eso el resultado final no es el de la version anterior"),
        "otro_defecto_corregido": (
            "la primera version optimizaba los pesos con SLSQP sobre el macro-F1, y "
            "devolvia pesos EXACTAMENTE uniformes en todos los tamanos y en los dos "
            "objetivos, porque el F1 con umbral fijo es constante a tramos y su gradiente "
            "es cero casi en todas partes, luego el optimizador se paraba en el punto "
            "inicial. Esos pesos uniformes eran un artefacto del metodo. La version vigente "
            "usa una busqueda sin derivadas y determinista sobre el simplejo"),
        "protocolo": {
            "pertenencia": ("anidada. Para cada pliegue exterior, los candidatos historicos "
                            "se ordenan por utilidad ajustada por diversidad calculada SOLO "
                            "con las filas internas y se conservan los tres primeros"),
            "pesos": ("anidados. Para cada pliegue exterior se optimizan con las filas "
                      "internas y se aplican al pliegue retenido"),
            "optimizador": (f"busqueda sin derivadas: punto uniforme, los k vertices "
                            f"unitarios y {N_PUNTOS_SIMPLEJO} sorteos Dirichlet(1) con "
                            f"semilla fija"),
            "objetivos": {"A": "macro-F1 con umbrales congelados",
                          "B": f"macro-F1 mas {LAMBDA_ENTROPIA} por entropia normalizada"},
            "umbrales": "0,44 / 0,51 / 0,46 congelados en todos los brazos",
            "rejilla": "ninguna",
            "max_miembros": MAX_MEMBROS,
        },
        "candidatos_historicos": {"canales_disponibles": len(nombres),
                                 "es_v03": int(sum(es_v03)),
                                 "candidatos": len(cands),
                                 "mejor_corrected_cv": round(
                                     float(max(ccv_canal[i] for i in cands)), 6)},
        "resultados": nativo(df_out.to_dict(orient="records")),
        "mejor_configuracion": nativo({k: mejor[k] for k in mejor.index}),
        "control_de_atribucion": atribucion,
        "desplazamiento": {
            "desplazamiento_medio_de_probabilidad": round(cambios_p, 6),
            "decisiones_que_cambian": cambios_d,
            "distancia_media_al_umbral_v03": round(float(dist_v03.mean()), 6),
            "distancia_media_al_umbral_ensemble": round(float(dist_ens.mean()), 6),
            "celdas_que_cambian_y_se_alejan_del_umbral": n_aleja,
            "celdas_que_cambian_por_unidad_de_desplazamiento": round(ratio_celdas, 4),
            "lectura": lectura,
        },
        "reproducibilidad": {"nivel": "R2", "git_commit": commit, "seed": SEMILLA,
                             "puntos_en_el_simplejo": N_PUNTOS_SIMPLEJO,
                             "optimizador": "busqueda sin derivadas sobre el simplejo"},
        "decision": decision,
        "formulacion": "V03 remains the best verified leaderboard result.",
        "gastos": {"submissions": 0, "consultas_kaggle": 0, "uso_holdout": 0, "uso_test": 0,
                   "experimentos": 1,
                   "segundos": round((datetime.now(timezone.utc) - inicio).total_seconds(), 1)},
    }
    JSON_ENS.write_text(json.dumps(informe, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")

    L: list[str] = []
    W = L.append
    W("# P61 - Ensemble pequeno con pertenencia y pesos ANIDADOS")
    W("")
    W(f"Generado por `experiments/p61_05_ensemble.py`. "
      f"Fecha {inicio.isoformat(timespec='seconds')}.")
    W("")
    W("## El defecto que hubo que corregir antes de aceptar el resultado")
    W("")
    W("Una primera version anidaba los **pesos** pero no la **pertenencia** al pool: los tres")
    W("historicos se elegian una sola vez, de forma global, por una utilidad ajustada por")
    W("diversidad calculada sobre todo el OOF. Eso es ajustar sobre los datos externos,")
    W("porque la pregunta de *quien pertenece al pool* se respondia con las mismas filas")
    W("sobre las que despues se puntua. Anidar los pesos no repara eso, porque el conjunto de")
    W("candidatos ya arrastraba la fuga.")
    W("")
    W("La version vigente hace la seleccion de pertenencia **tambien** dentro de cada pliegue,")
    W("con las filas internas, y por eso el numero final **no es** el de la primera version.")
    W("")
    W("El segundo defecto era de optimizacion: SLSQP sobre el macro-F1 devolvia pesos")
    W("**exactamente uniformes** en todos los tamanos y en los dos objetivos, porque el F1 con")
    W("umbral fijo es constante a tramos y su gradiente es cero casi en todas partes, luego el")
    W("optimizador se paraba en el punto inicial. Esos pesos uniformes eran un artefacto del")
    W("metodo, no una solucion. La version vigente usa una busqueda **sin derivadas** y")
    W("determinista sobre el simplejo.")
    W("")
    W("## El protocolo, anidado en las dos dimensiones")
    W("")
    W("```text")
    W("para cada pliegue exterior f:")
    W("    internas = los otros cuatro pliegues")
    W("    PERTENENCIA: se ordenan los historicos por utilidad ajustada por diversidad")
    W("                 calculada SOLO con las filas internas, y se conservan los tres")
    W("    PESOS:       se optimizan con las filas internas")
    W("    se aplican el pool y los pesos resultantes al pliegue f")
    W("la puntuacion se arma con los cinco pliegues retenidos")
    W("```")
    W("")
    W("El optimizador de pesos es una busqueda sin derivadas: el punto uniforme, los k")
    W(f"vertices unitarios y {N_PUNTOS_SIMPLEJO} sorteos Dirichlet(1) con semilla fija. El punto")
    W("uniforme siempre es candidato, luego el resultado no puede ser peor que el promedio")
    W("simple, y el generador sembrado lo hace reproducible.")
    W("")
    W("## Resultados")
    W("")
    W("| ID | pool | k | obj | corrected_cv | delta vs V03 | pliegues + | rescate | dano | neto | gate |")
    W("|---|---|---:|---|---:|---:|---:|---:|---:|---:|---|")
    for r in df.itertuples():
        W(f"| {r.ID} | {r.pool} | {r.k} | {r.objetivo} | {r.Corrected_CV:.10f} | "
          f"{float(r.Delta_vs_V03):+.10f} | {r.folds_mejorados}/5 | {r.Rescue} | {r.Damage} | "
          f"{r.Net:+d} | {'**SUPERADO**' if r.gate_superado else 'no'} |")
    W("")
    W("## El control de atribucion, que es la medicion decisiva")
    W("")
    W("El pool ya contiene V03, luego un resultado positivo no es atribuible sin comparar el")
    W("pool con y sin el modelo nuevo. Aqui eso resulto decisivo:")
    W("")
    W("```text")
    W(f"{atribucion}")
    W("```")
    W("")
    W("## Desplazamiento de probabilidad frente a desplazamiento de decision")
    W("")
    W("```text")
    W(f"desplazamiento medio de probabilidad         {cambios_p:.6f}")
    W(f"decisiones que cambian                      {cambios_d} de {n * 3}")
    W(f"distancia media al umbral, V03              {dist_v03.mean():.6f}")
    W(f"distancia media al umbral, ensemble         {dist_ens.mean():.6f}")
    W(f"celdas que cambian y se alejan del umbral   {n_aleja}")
    W(f"celdas que cambian por unidad de desplazamiento {ratio_celdas:.2f}")
    W(f"lectura: {lectura}")
    W("```")
    W("")
    W("## Reproducibilidad")
    W("")
    W("Nivel **R2**: semilla fija, artefacto en disco con los pesos por pliegue y por miembro,")
    W("y hash del OOF verificado. Una reejecucion con la misma semilla dara el mismo resultado,")
    W("pero no se ha reproducido de forma independiente en otra sesion.")
    W("")
    W("## Decision")
    W("")
    W("```text")
    W(f"decision  {decision}")
    W("```")
    W("")
    W("> V03 remains the best verified leaderboard result.")
    W("")
    W("*Generado por `experiments/p61_05_ensemble.py`. Sin push.*")
    MD_ENS.write_text(a_ascii("\n".join(L)), encoding="ascii")

    for destino in (MD_ENS, JSON_ENS, CSV_ENS, CSV_PESOS):
        if any(b > 127 for b in destino.read_bytes()):
            raise SystemExit(f"PARADA: {destino.name} conserva bytes no ASCII")
    log("")
    log(f"P61-05 OK | {len(configs)} configuraciones | mejor {mejor['ID']} "
        f"{mejor['Corrected_CV']:.10f} | gate {bool(mejor['gate_superado'])}")


if __name__ == "__main__":
    ast.parse(Path(__file__).read_text(encoding="utf-8"))
    main()
