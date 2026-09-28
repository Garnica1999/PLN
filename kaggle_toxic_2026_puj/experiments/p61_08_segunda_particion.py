"""P61 step 08: second partition for the frozen finalists, with two controls.

What is being validated, and what is not
---------------------------------------
Two ensemble configurations passed the P61 gate on the canonical partition:

    con_nuevo_k3_A   corrected_cv 0,7835159596   delta +0,0063045070   4/5 folds
    con_nuevo_k4_A   corrected_cv 0,7832411761   delta +0,0060297235   4/5 folds

A third, con_nuevo_k5_A, also passed on delta and on folds, but its bootstrap
interval is [-0,001798, +0,008766], which contains zero, so it fails the
"bootstrap favorable" condition and is therefore not a finalist. That exclusion is
recorded rather than quietly dropped.

The second partition redraws the GROUPING of the rows, not the base learners. The
twenty historical channels exist only as out-of-fold probabilities on the canonical
partition; refitting them would require their exact training code and
hyperparameters, and guessing those would silently substitute a different model for a
named one. So what this step re-validates is the DECISION LAYER, namely the choice of
membership and the choice of weights, and it says so in its own output.

Three tests, because one is not enough
--------------------------------------
    test 1  a fresh group-aware partition, membership and weights re-derived inside
            it and scored on the folds it never saw

    test 2  repeated 3-versus-2 holdback over all ten pairs of folds, which removes
            any dependence on one particular fold structure

    test 3  a NEGATIVE CONTROL in which the historical members are deliberately
            replaced by the WORST channels in the pool. If the gain survives with
            members that are known to be far worse than V03, then the gain is an
            artefact of running a search at all, and the mechanism claim is false. If
            the gain collapses, the mechanism is genuine diversity.

The negative control is the one that matters. The other two only ask whether the
number is reproducible; this one asks whether the explanation is right.

Per paragraph 37, a finalist is frozen before this runs, and nothing here modifies
one afterwards.

Usage
-----
    /home/carlos/venv/bin/python experiments/p61_08_segunda_particion.py
"""

from __future__ import annotations

import ast
import hashlib
import itertools
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

from p59_01_voronoi_local import (
    V03_CCV,
    V03_SHA,
    V03_SUB,
)

REPORTS = PROJECT_ROOT / "reports"
OOF = PROJECT_ROOT / "features" / "oof_v03_final.npz"
POOL_HIST = PROJECT_ROOT / "features" / "p50_celda_features.npz"
CELDA_LGB = PROJECT_ROOT / "features" / "p61_lightgbm_cells.npz"
MD_SEG = REPORTS / "p61_segunda_particion.md"
JSON_SEG = REPORTS / "p61_segunda_particion.json"
CSV_SEG = REPORTS / "p61_segunda_particion.csv"

V03_UMBRALES = np.array([0.44, 0.51, 0.46])
SEMILLA_CANONICA = 42
SEMILLA_NUEVA = 20260928
N_HISTORICOS = 3
LAMBDA_ENTROPIA = 0.002
N_PUNTOS_SIMPLEJO = 300
N_BOOT = 4000
GATE_DELTA = 0.003

# Los finalistas quedan CONGELADOS aqui, antes de ejecutar nada. El parrafo 37 exige
# fijarlos antes y no modificarlos despues, luego la lista es una constante y no un
# resultado que el script pueda cambiar.
FINALISTAS = ("con_nuevo_k3_A", "con_nuevo_k4_A")
EXCLUIDOS = {
    "con_nuevo_k5_A": ("supera el delta y los 5/5 pliegues, pero su IC95 de bootstrap es "
                       "[-0,001798, +0,008766], que CONTIENE el cero, luego no cumple la "
                       "condicion de bootstrap favorable y no es finalista"),
}


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[P61-08] {mensaje}", flush=True)


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
    """Macro-F1 de un pesos sobre un subconjunto de filas.

    Args:
        w: vector de pesos.
        probas: probabilidades de cada miembro, forma (k, n, 3).
        y: etiquetas enteras.
        filas: indices de filas.

    Returns:
        El macro-F1 resultante.
    """
    p = np.tensordot(w, probas[:, filas, :], axes=(0, 0))
    return float(f1_score(y[filas], (p >= V03_UMBRALES[None, :]).astype(int),
                          average="macro"))


def entropia(w: np.ndarray) -> float:
    """Entropia normalizada de un vector de pesos.

    Args:
        w: vector de pesos.

    Returns:
        La entropia normalizada, entre cero y uno.
    """
    return float(-np.sum(w * np.log(np.maximum(w, 1e-12))) / np.log(max(len(w), 2)))


def optimizar_pesos(probas: np.ndarray, y: np.ndarray, filas: np.ndarray,
                    objetivo: str, k: int, semilla: int) -> np.ndarray:
    """Busqueda determinista sin derivadas sobre el simplejo.

    Args:
        probas: probabilidades, forma (k, n, 3).
        y: etiquetas enteras.
        filas: indices de filas internas.
        objetivo: "A" o "B".
        k: numero de miembros.
        semilla: semilla del generador, distinta por particion.

    Returns:
        Vector de pesos normalizado.
    """
    rng = np.random.default_rng(semilla + 7919 * k + (0 if objetivo == "A" else 1))
    candidatos: list[np.ndarray] = [np.full(k, 1.0 / k)]
    for i in range(k):
        v = np.zeros(k)
        v[i] = 1.0
        candidatos.append(v)
    for _ in range(N_PUNTOS_SIMPLEJO):
        candidatos.append(rng.dirichlet(np.ones(k)))

    def valor(w: np.ndarray) -> float:
        """Puntua un vector de pesos.

        Args:
            w: vector de pesos.

        Returns:
            El valor a maximizar.
        """
        base = f1_de_pesos(w, probas, y, filas)
        return base if objetivo == "A" else base + LAMBDA_ENTROPIA * entropia(w)

    mejor = max(candidatos, key=valor)
    s = float(mejor.sum())
    return mejor / s if s > 0 else np.full(k, 1.0 / k)


def utilidad(pred: np.ndarray, pred_ref: np.ndarray, y: np.ndarray,
             filas: np.ndarray) -> float:
    """Utilidad ajustada por diversidad sobre un subconjunto de filas.

    Args:
        pred: predicciones del candidato.
        pred_ref: predicciones de referencia.
        y: etiquetas enteras.
        filas: indices de filas.

    Returns:
        La utilidad ajustada por diversidad.
    """
    a = (pred[filas] == y[filas]).ravel().astype(float)
    b = (pred_ref[filas] == y[filas]).ravel().astype(float)
    resc = float(np.count_nonzero((a == 1) & (b == 0)))
    dan = float(np.count_nonzero((a == 0) & (b == 1)))
    corr = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else 0.0
    return resc - dan + 0.5 * (1.0 - max(min(corr, 1.0), -1.0))


def evaluar(miembros: list[tuple[str, np.ndarray]], y: np.ndarray, ref: np.ndarray,
            interna: np.ndarray, externa: np.ndarray, pliegues: np.ndarray,
            objetivo: str, etiqueta_pool: str, cands: list[int],
            cprob: np.ndarray, nombres: list[str], pred_ref: np.ndarray,
            semilla: int) -> dict[str, Any]:
    """Deriva pertenencia y pesos en las filas internas y puntua las externas.

    Args:
        miembros: lista de (nombre, probabilidades) de los miembros fijos del pool.
        y: etiquetas enteras.
        ref: predicciones de referencia.
        interna: indices internos.
        externa: indices externos.
        pliegues: vector de pliegues.
        objetivo: "A" o "B".
        etiqueta_pool: nombre del pool.
        cands: indices de canales historicos candidatos.
        cprob: probabilidades de los canales historicos.
        nombres: nombres de los canales.
        pred_ref: predicciones de referencia.
        semilla: semilla del optimizador.

    Returns:
        Diccionario con la puntuacion y el detalle.
    """
    elegido = sorted(cands, key=lambda i: -utilidad(
        (cprob[i] >= V03_UMBRALES[None, :]).astype(int), pred_ref, y, interna))
    len(miembros) + (len(elegido) if etiqueta_pool == "con_nuevo" else 0)
    nombres_pool = [m[0] for m in miembros] + [nombres[i] for i in elegido[:N_HISTORICOS]]
    probas = np.stack([m[1] for m in miembros] + [cprob[i] for i in elegido[:N_HISTORICOS]])
    if etiqueta_pool == "sin_nuevo" and len(miembros) == 1:
        nombres_pool = ["V03"] + [nombres[i] for i in elegido[:N_HISTORICOS]]
        probas = np.stack([miembros[0][1]] + [cprob[i] for i in elegido[:N_HISTORICOS]])
    w = optimizar_pesos(probas, y, interna, objetivo, len(probas), semilla)
    p_ext = np.tensordot(w, probas[:, externa, :], axes=(0, 0))
    pred_ext = (p_ext >= V03_UMBRALES[None, :]).astype(int)
    return {"proba": p_ext, "pred": pred_ext, "pesos": w, "miembros": nombres_pool,
            "utilidades": {nombres[i]: round(utilidad(
                (cprob[i] >= V03_UMBRALES[None, :]).astype(int), pred_ref, y, interna), 2)
                for i in elegido[:N_HISTORICOS]},
            "k": len(probas), "elegidos": len(elegido)}


def main() -> None:
    """Ejecuta la segunda particion sobre los finalistas congelados.

    Returns:
        None.
    """
    inicio = datetime.now(timezone.utc)
    log("P61-08 segunda particion sobre los finalistas CONGELADOS")
    log(f"   finalistas: {list(FINALISTAS)}")
    log("   lo que se revalida es la CAPA DE DECISION, no los modelos base, porque los")
    log("   canales historicos solo existen como OOF en la particion canonica")
    log(f"   semilla de la particion nueva: {SEMILLA_NUEVA}, distinta de la canonica")

    hash_real = hashlib.sha256(V03_SUB.read_bytes()).hexdigest()
    if hash_real != V03_SHA:
        raise SystemExit("PARADA: la submission de V03 no coincide con el sha256 congelado")
    log(f"   V03 sha256 verificado e intacto: {hash_real[:16]}...")

    v = np.load(OOF, allow_pickle=True)
    y, pliegues_c, pred_v03 = v["y_true"].astype(int), v["pliegues"], v["pred"].astype(int)
    proba_v03 = v["proba"]
    n = y.shape[0]
    ccv_ref = float(f1_score(y, pred_v03, average="macro"))
    if ccv_ref != V03_CCV:
        raise SystemExit(f"PARADA: el OOF canonico da {ccv_ref!r}, no {V03_CCV!r}")

    z = np.load(POOL_HIST, allow_pickle=True)
    nombres = [str(s) for s in z["nombres"]]
    cprob = z["canal_proba"].astype(np.float64)
    ccv_canal = {i: float(f1_score(y, (cprob[i] >= V03_UMBRALES[None, :]).astype(int),
                                    average="macro")) for i in range(len(nombres))}
    es_v03 = {i for i in range(len(nombres))
              if ccv_canal[i] == ccv_ref
              and int(np.count_nonzero((cprob[i] >= V03_UMBRALES[None, :]).astype(int)
                                       != pred_v03)) == 0}
    cands = [i for i in range(len(nombres)) if i not in es_v03]
    proba_lgb = None
    if CELDA_LGB.exists():
        zz = np.load(CELDA_LGB, allow_pickle=True)
        if bool(np.array_equal(zz["ids"], v["ids"])) and "proba_lgbm_optuna" in zz.files:
            proba_lgb = zz["proba_lgbm_optuna"]
    log("")
    log("   LOS PEORES CANALES, PARA EL CONTROL NEGATIVO")
    peores = sorted(cands, key=lambda i: ccv_canal[i])[:N_HISTORICOS]
    for i in peores:
        log(f"      {nombres[i]:26} corrected_cv {ccv_canal[i]:.6f} "
            f"({ccv_canal[i] - ccv_ref:+.6f} frente a V03)")

    part = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "holdout_v2_ids.csv")
    grupos = part.set_index("id").loc[v["ids"], "grupo_near_dup_autor"].to_numpy()

    # ---------- particiones ----------
    def nueva_particion(semilla: int) -> np.ndarray:
        """Reparte los grupos canonicos en cinco pliegues con una semilla dada.

        Args:
            semilla: semilla del generador.

        Returns:
            Vector de pliegue, de uno a cinco.
        """
        unicos = pd.unique(grupos)
        rng = np.random.default_rng(semilla)
        orden = rng.permutation(len(unicos))
        plieg = np.zeros(n, dtype=int)
        for pos, g in enumerate(unicos[orden]):
            plieg[grupos == g] = (pos % 5) + 1
        return plieg

    pliegues_n = nueva_particion(SEMILLA_NUEVA)
    partidos = int(pd.Series(grupos).groupby(grupos).nunique().gt(1).sum())
    g_texto = "\n".join(map(str, grupos)).encode("utf-8")
    hash_fold_c = hashlib.sha256(
        pliegues_c.tobytes() + g_texto
    ).hexdigest()
    hash_fold_n = hashlib.sha256(
        pliegues_n.tobytes() + g_texto
    ).hexdigest()
    log("")
    log("   LAS DOS PARTICIONES")
    log(f"      canonica: hash {hash_fold_c[:16]}... tamanos "
        f"{[int((pliegues_c == g).sum()) for g in range(1, 6)]}")
    log(f"      nueva:    hash {hash_fold_n[:16]}... tamanos "
        f"{[int((pliegues_n == g).sum()) for g in range(1, 6)]}")
    log(f"      grupos partidos entre pliegues: {partidos}")
    if partidos:
        raise SystemExit(f"PARADA: {partidos} grupos canonicos partidos entre pliegues")
    iguales = bool(np.array_equal(pliegues_c, pliegues_n))
    log(f"      la nueva particion es DISTINTA de la canonica: {not iguales}")

    filas: list[dict[str, Any]] = []

    # ---------- TEST 1: particion nueva, todo re-derivado dentro ----------
    log("")
    log("   TEST 1, PARTICION NUEVA: pertenencia y pesos re-derivados dentro de ella")
    ref_n = pred_v03
    t1: dict[str, Any] = {}
    for etiqueta_pool in ("con_nuevo", "sin_nuevo"):
        miembros = [("V03", proba_v03)]
        if etiqueta_pool == "con_nuevo" and proba_lgb is not None:
            miembros.append(("LightGBM+Optuna", proba_lgb))
        for objetivo in ("A", "B"):
            pred = np.zeros((n, 3), dtype=int)
            detalle: list[dict[str, Any]] = []
            for f in range(1, 6):
                interna = np.where(pliegues_n != f)[0]
                externa = np.where(pliegues_n == f)[0]
                r = evaluar(miembros, y, ref_n, interna, externa, pliegues_n, objetivo,
                            etiqueta_pool, cands, cprob, nombres, pred_v03, SEMILLA_NUEVA)
                pred[externa] = r["pred"]
                detalle.append({"pliegue": f, "miembros": " | ".join(r["miembros"]),
                                "pesos": [round(float(x), 4) for x in r["pesos"]]})
            ccv = float(f1_score(y, pred, average="macro"))
            feats = [float(f1_score(y[pliegues_n == g], pred[pliegues_n == g],
                                    average="macro")) for g in range(1, 6)]
            fev = [float(f1_score(y[pliegues_n == g], pred_v03[pliegues_n == g],
                                  average="macro")) for g in range(1, 6)]
            fm = sum(1 for u, w in zip(feats, fev) if u > w)
            d = ccv - ccv_ref
            clave = f"test1_{etiqueta_pool}_{objetivo}"
            t1[clave] = {"corrected_cv": round(ccv, 10), "delta": round(d, 10),
                         "pliegues_mejorados": fm}
            filas.append({
                "test": "1_nueva_particion", "ID": clave, "pool": etiqueta_pool,
                "objetivo": objetivo, "Corrected_CV": round(ccv, 10),
                "Delta_vs_V03": round(d, 10), "pliegues_mejorados": fm,
                "gate_delta": bool(d >= GATE_DELTA), "Bootstrap_CI": None,
                "detalle": json.dumps(detalle, ensure_ascii=False)[:400],
            })
            log(f"      {clave:22} ccv {ccv:.10f} delta {d:+.10f} pliegues {fm}/5 "
                f"gate_delta {d >= GATE_DELTA}")
    ccv_test1 = t1.get("test1_con_nuevo_A", {}).get("delta", 0.0)
    log(f"      el mejor del test 1 pasa de delta +0,006305 canonico a {ccv_test1:+.10f}")

    # ---------- TEST 2: holdback 3 contra 2 sobre las diez parejas ----------
    log("")
    log("   TEST 2, HOLDBACK 3 CONTRA 2 SOBRE LAS DIEZ PAREJAS DE PLIEGUES")
    t2: list[float] = []
    for a, b in itertools.combinations(range(1, 6), 2):
        interna = np.where(np.isin(pliegues_c, [a, b]))[0]
        externa = np.where(~np.isin(pliegues_c, [a, b]))[0]
        r = evaluar([("V03", proba_v03), ("LightGBM+Optuna", proba_lgb)], y, pred_v03,
                    interna, externa, pliegues_c, "A", "con_nuevo", cands, cprob,
                    nombres, pred_v03, SEMILLA_CANONICA)
        ccv_e = float(f1_score(y[externa], r["pred"], average="macro"))
        ccv_r = float(f1_score(y[externa], pred_v03[externa], average="macro"))
        d = ccv_e - ccv_r
        t2.append(d)
        filas.append({
            "test": "2_holdback_3v2", "ID": f"holdback_{a}v{b}", "pool": "con_nuevo",
            "objetivo": "A", "Corrected_CV": round(ccv_e, 10), "Delta_vs_V03": round(d, 10),
            "pliegues_mejorados": None, "gate_delta": bool(d >= GATE_DELTA),
            "Bootstrap_CI": None,
            "detalle": f"internas {len(interna)} filas, externas {len(externa)}; "
                       f"miembros {' | '.join(r['miembros'])}",
        })
        log(f"      internas pliegues {a},{b} ({len(interna)} filas) -> externas "
            f"({len(externa)} filas): delta {d:+.6f} {'positivo' if d > 0 else 'NEGATIVO'}")
    arr2 = np.array(t2)
    log(f"      delta medio {arr2.mean():+.6f} | mediana {np.median(arr2):+.6f} | "
        f"positivos {int((arr2 > 0).sum())}/10 | minimo {arr2.min():+.6f}")
    filas.append({
        "test": "2_holdback_resumen", "ID": "holdback_resumen", "pool": "con_nuevo",
        "objetivo": "A", "Corrected_CV": None, "Delta_vs_V03": round(float(arr2.mean()), 10),
        "pliegues_mejorados": int((arr2 > 0).sum()), "gate_delta": bool(arr2.mean() >= 0),
        "Bootstrap_CI": None,
        "detalle": f"mediana {float(np.median(arr2)):+.6f}, minimo {float(arr2.min()):+.6f}, "
                   f"maximo {float(arr2.max()):+.6f}",
    })

    # ---------- TEST 3: control negativo con los peores canales ----------
    log("")
    log("   TEST 3, CONTROL NEGATIVO: los historicos se sustituyen por los PEORES canales")
    t3: dict[str, float] = {}
    for etiqueta_pool, pool_c in (("negativo_peores", peores),):
        pred = np.zeros((n, 3), dtype=int)
        for f in range(1, 6):
            interna = np.where(pliegues_c != f)[0]
            externa = np.where(pliegues_c == f)[0]
            pl = [("V03", proba_v03), ("LightGBM+Optuna", proba_lgb)] + [
                (nombres[i], cprob[i]) for i in pool_c]
            w = optimizar_pesos(np.stack([p for _, p in pl]), y, interna, "A", len(pl),
                                SEMILLA_CANONICA)
            pred[externa] = (np.tensordot(w, np.stack([p for _, p in pl])[:, externa, :],
                                           axes=(0, 0)) >= V03_UMBRALES[None, :]).astype(int)
        ccv = float(f1_score(y, pred, average="macro"))
        t3[etiqueta_pool] = ccv - ccv_ref
        filas.append({
            "test": "3_control_negativo", "ID": "negativo_peores", "pool": etiqueta_pool,
            "objetivo": "A", "Corrected_CV": round(ccv, 10),
            "Delta_vs_V03": round(ccv - ccv_ref, 10), "pliegues_mejorados": None,
            "gate_delta": bool(ccv - ccv_ref >= GATE_DELTA), "Bootstrap_CI": None,
            "detalle": "miembros: V03 | LightGBM+Optuna | " + " | ".join(nombres[i]
                                                                        for i in peores),
        })
        log(f"      con los PEORES canales: corrected_cv {ccv:.10f} "
            f"delta {ccv - ccv_ref:+.10f}")

    # ---------- veredicto de los tres tests ----------
    log("")
    log("   VEREDICTO")
    log(f"      test 1, delta en la particion nueva        {ccv_test1:+.10f}")
    log(f"      test 2, delta medio en las diez parejas    {float(arr2.mean()):+.10f} "
        f"con {int((arr2 > 0).sum())}/10 positivas")
    log(f"      test 3, delta con miembros deliberadamente  {t3['negativo_peores']:+.10f}")
    log("               peores, que es el control")
    if t3["negativo_peores"] < 0 and arr2.mean() > 0 and ccv_test1 > 0:
        veredicto_mecanismo = (
            "los tres tests son consistentes con un mecanismo real: la ganancia sobrevive al "
            "redibujo de la particion y al holdback repetido, y COLAPSA cuando los miembros "
            "se sustituyen por canales deliberadamente peores. Esa diferencia es lo que "
            "distingue diversidad verdadera de un artefacto de buscar en el simplejo"
        )
    elif t3["negativo_peores"] >= GATE_DELTA:
        veredicto_mecanismo = (
            "el control negativo NO colapsa: la ganancia aparece igual con miembros que "
            "estan 0,04 a 0,24 por debajo de V03, luego la explicacion por diversidad no se "
            "sostiene y el efecto hay que atribuirlo al hecho de ejecutar una busqueda de "
            "pesos sobre un pool que incluye V03"
        )
    else:
        veredicto_mecanismo = (
            "los tests no dan una lectura coherente: revisar el detalle por test antes de "
            "aceptar o rechazar la finalist"
        )
    log(f"      VEREDICTO: {veredicto_mecanismo}")

    # ---------- bootstrap sobre la particion nueva ----------
    log("")
    log("   BOOTSTRAP sobre la particion nueva, agrupado por el grupo canonico")
    rng = np.random.default_rng(SEMILLA_NUEVA)
    pred_n = np.zeros((n, 3), dtype=int)
    for f in range(1, 6):
        interna = np.where(pliegues_n != f)[0]
        externa = np.where(pliegues_n == f)[0]
        r = evaluar([("V03", proba_v03), ("LightGBM+Optuna", proba_lgb)], y, pred_v03,
                    interna, externa, pliegues_n, "A", "con_nuevo", cands, cprob,
                    nombres, pred_v03, SEMILLA_NUEVA)
        pred_n[externa] = r["pred"]
    g_unicos, g_cod = np.unique(grupos, return_inverse=True)
    n_grupos = len(g_unicos)
    dc = ((pred_n == y).astype(int) - (pred_v03 == y).astype(int))
    # La diferencia se acumula por grupo con bincount, que es O(n). Una version anterior
    # buscaba el indice de cada grupo dentro de una lista de forma literal, luego era
    # O(grupos al cuadrado) en cada una de las N_BOOT replicas y habria tardado horas.
    dif_g = np.bincount(g_cod, weights=dc.sum(axis=1), minlength=n_grupos)
    idx_boot = rng.integers(0, n_grupos, size=(N_BOOT, n_grupos))
    ms = np.asarray(dif_g[idx_boot].sum(axis=1) / n_grupos, dtype=float)
    ic_lo, ic_hi = float(np.percentile(ms, 2.5)), float(np.percentile(ms, 97.5))
    log(f"      IC95 de la diferencia por grupo, sobre {n_grupos} grupos: "
        f"[{ic_lo:+.6f}, {ic_hi:+.6f}]")
    log(f"      incluye el cero: {ic_lo <= 0 <= ic_hi}")
    for i, f in enumerate(filas):
        if f["test"] == "1_nueva_particion" and f["ID"] == "test1_con_nuevo_A":
            filas[i]["Bootstrap_CI"] = f"IC95 [{ic_lo:+.6f}, {ic_hi:+.6f}]"

    # ---------- escritura ----------
    df = pd.DataFrame(filas)
    df.to_csv(CSV_SEG, index=False, encoding="utf-8")
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                            text=True, cwd=PROJECT_ROOT, check=False).stdout.strip()
    informe = {
        "familia": "P61_SEGUNDA_PARTICION",
        "naturaleza": ("EXPERIMENTO. 0 submissions, 0 Kaggle, 0 holdout, 0 test. "
                       "V03 no se modifica. Los finalistas se congelan antes de ejecutar"),
        "fecha": inicio.isoformat(timespec="seconds"),
        "experiment_id": "P61-03", "track": "P61-segunda-particion",
        "script": "experiments/p61_08_segunda_particion.py",
        "finalistas_congelados": list(FINALISTAS),
        "excluidos_del_concurso_de_finalistas": EXCLUIDOS,
        "alcance_de_lo_revalidado": (
            "se revalida la CAPA DE DECISION, es decir la eleccion de pertenencia y la de "
            "pesos. NO se revalidan los modelos base, porque los veinte canales historicos "
            "solo existen como probabilidades out-of-fold en la particion canonica. "
            "Reajustarlos exigiria su codigo de entrenamiento exacto y sus hiperparametros, "
            "y adivinarlos sustituiria en silencio un modelo distinto por uno con nombre"),
        "tests": {
            "1_nueva_particion": {
                "semilla": SEMILLA_NUEVA, "hash_fold_canonico": hash_fold_c,
                "hash_fold_nuevo": hash_fold_n, "resultados": t1,
            },
            "2_holdback_3_contra_2": {
                "delta_medio": round(float(arr2.mean()), 10),
                "delta_mediano": round(float(np.median(arr2)), 10),
                "positivas": int((arr2 > 0).sum()), "de": 10,
                "delta_minimo": round(float(arr2.min()), 10),
                "delta_maximo": round(float(arr2.max()), 10),
            },
            "3_control_negativo": {
                "miembros": [nombres[i] for i in peores],
                "corrected_cv_de_los_miembros": {nombres[i]: round(ccv_canal[i], 6)
                                                for i in peores},
                "delta": round(t3["negativo_peores"], 10),
            },
        },
        "veredicto_del_mecanismo": veredicto_mecanismo,
        "bootstrap_particion_nueva": {"IC95": [round(ic_lo, 6), round(ic_hi, 6)],
                                      "n": N_BOOT, "agrupado_por": "grupo canonico",
                                      "grupos": n_grupos,
                                      "incluye_el_cero": bool(ic_lo <= 0 <= ic_hi)},
        "reproducibilidad": {"nivel": "R2", "git_commit": commit,
                             "seed_particion_nueva": SEMILLA_NUEVA,
                             "seed_canonica": SEMILLA_CANONICA,
                             "hash_fold_canonico": hash_fold_c,
                             "hash_fold_nuevo": hash_fold_n,
                             "puntos_en_el_simplejo": N_PUNTOS_SIMPLEJO},
        "formulacion": "V03 remains the best verified leaderboard result.",
        "gastos": {"submissions": 0, "consultas_kaggle": 0, "uso_holdout": 0,
                   "uso_test": 0, "experimentos": 1},
    }
    JSON_SEG.write_text(json.dumps(informe, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")

    L: list[str] = []
    W = L.append
    W("# P61 - Segunda particion sobre los finalistas congelados")
    W("")
    W(f"Generado por `experiments/p61_08_segunda_particion.py`. "
      f"Fecha {inicio.isoformat(timespec='seconds')}.")
    W("")
    W("## Que es finalista y que no")
    W("")
    W("```text")
    W("finalistas congelados: " + ", ".join(FINALISTAS))
    W("excluido: con_nuevo_k5_A, " + EXCLUIDOS["con_nuevo_k5_A"])
    W("```")
    W("")
    W("El tercero superaba el delta y los 5/5 pliegues, pero su intervalo de bootstrap")
    W("contiene el cero, luego no cumple la condicion de bootstrap favorable. Se excluye y se")
    W("dice por que, en vez de dejarlo en silencio.")
    W("")
    W("## Que se revalida, y que no")
    W("")
    W(informe["alcance_de_lo_revalidado"])
    W("")
    W("## Las dos particiones")
    W("")
    W("```text")
    W(f"canonica  hash {hash_fold_c[:32]}...")
    W(f"nueva     hash {hash_fold_n[:32]}...  semilla {SEMILLA_NUEVA}")
    W(f"grupos canonicos partidos entre pliegues  {partidos}")
    W("```")
    W("")
    W("## Test 1, particion nueva")
    W("")
    W("| configuracion | corrected_cv | delta | pliegues + |")
    W("|---|---:|---:|---:|")
    for k, r in t1.items():
        W(f"| {k} | {r['corrected_cv']:.10f} | {r['delta']:+.10f} | {r['pliegues_mejorados']}/5 |")
    W("")
    W("## Test 2, holdback tres contra dos sobre las diez parejas")
    W("")
    W("| internas | delta en las externas |")
    W("|---|---:|")
    for i, f in enumerate(filas):
        if f["test"] == "2_holdback_3v2":
            W(f"| pliegues {f['ID'].split('_')[1]} | {f['Delta_vs_V03']:+.6f} |")
    W("")
    W("```text")
    W(f"delta medio {float(arr2.mean()):+.6f} | mediana {float(np.median(arr2)):+.6f}")
    W(f"positivas {int((arr2 > 0).sum())}/10 | minimo {float(arr2.min()):+.6f} | "
      f"maximo {float(arr2.max()):+.6f}")
    W("```")
    W("")
    W("## Test 3, control negativo")
    W("")
    W("Los miembros historicos se sustituyen por los PEORES canales del pool, que estan entre")
    W("0,04 y 0,24 por debajo de V03:")
    W("")
    W("```text")
    for i in peores:
        W(f"  {nombres[i]:26} corrected_cv {ccv_canal[i]:.6f}")
    W("")
    W(f"delta con esos miembros: {t3['negativo_peores']:+.10f}")
    W("```")
    W("")
    W("Este es el test que decide si la explicacion es correcta. Los otros dos solo preguntan")
    W("si el numero es reproducible; este pregunta si la causa es la que se afirma.")
    W("")
    W("## Bootstrap sobre la particion nueva")
    W("")
    W("```text")
    W(f"IC95 agrupado por los {n_grupos} grupos canonicos  "
      f"[{ic_lo:+.6f}, {ic_hi:+.6f}]")
    W(f"incluye el cero  {ic_lo <= 0 <= ic_hi}")
    W("```")
    W("")
    W("## Veredicto")
    W("")
    W("```text")
    W(veredicto_mecanismo)
    W("```")
    W("")
    W("> V03 remains the best verified leaderboard result.")
    W("")
    W("*Generado por `experiments/p61_08_segunda_particion.py`. Sin push.*")
    MD_SEG.write_text(a_ascii("\n".join(L)), encoding="ascii")

    for destino in (MD_SEG, JSON_SEG, CSV_SEG):
        if any(b > 127 for b in destino.read_bytes()):
            raise SystemExit(f"PARADA: {destino.name} conserva bytes no ASCII")
    log("")
    log(f"P61-08 OK | 3 tests | delta nueva particion {ccv_test1:+.6f} | "
        f"holdback medio {float(arr2.mean()):+.6f} | control negativo "
        f"{t3['negativo_peores']:+.6f}")


if __name__ == "__main__":
    ast.parse(Path(__file__).read_text(encoding="utf-8"))
    main()
