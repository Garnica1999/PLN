"""P63 track A: does the collection index carry signal, or is it an artefact?

The question, and why it is worth an experiment at all
-----------------------------------------------------
P63-02 classified the id as D. STRUCTURED_IDENTIFIER with high confidence: an integer
inside the contiguous block [20001, 24999], of which train and test are disjoint samples
that together occupy 2885 of 4999 positions. That is structure, and structure is a
precondition for signal, not a guarantee of it.

The gap analysis then found that the id has never been a model input. V15-00 excluded it
by calling it a counter, which P63-02 disproved. So the live question is narrow and
concrete: does the index carry any label information that V03 does not already have?

Three controls, because a positive here would be the easiest artefact in the project
------------------------------------------------------------------------
An id-block model would be the easiest positive to fake in the whole project, and the
reason is structural rather than moral. If the corpus was assembled in blocks and the
labels drift across blocks for reasons that have nothing to do with the text, then a model
that reads the index is reading the assembly order, and it will look brilliant in
cross-validation and die on the leaderboard. Three controls separate that from a real
effect:

    capacity   the same model on RANDOM features of identical dimensionality. If the
               real id block scores about the same as the random one, there is no specific
               signal, only the capacity to memorise a smooth function of 1483 rows.

    permutation  the same pipeline with the labels shuffled. This destroys any
               id-to-label relation while preserving the id distribution and the marginal
               label rates. A model that scores well here is reading noise, and its
               expected score is the floor that the real arm must beat.

    second partition  the frozen configuration rerun on the holdout2 grouped partition.
               The brief forbids using the holdout during exploration, and this arm is
               still exploration, so it is NOT run here. It is reserved for a finalist, and
               saying so now is cheaper than discovering it later.

The arms
--------
    id_solo     rank percentile of the id, plus coarse bins, plus small moduli
    random_solo the identical pipeline on random features of the same dimensionality
    id_perm     the id_solo pipeline with permuted labels
    V03         the frozen control, rebuilt from source and checked against the OOF

Deliberately not used: the raw id as a category, millions of buckets, target encoding and
label lookup. The brief forbids them and they would each be a way of memorising.

What a result means, decided before running
--------------------------------------------
    id_solo clearly above random_solo AND above id_perm
        there is id-to-label signal, and the next question is whether it is provenance
        that transfers, which is a different and harder question

    id_solo about equal to random_solo or equal to id_perm
        the apparent signal is capacity or noise, and the id is CLOSED as a predictor

    id_solo below V03
        expected, since the block carries a fraction of the information V03 has, and it
        says nothing about whether the fraction is real

Usage
-----
    /home/carlos/venv/bin/python experiments/p63_03_id_auditoria_predictiva.py
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

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

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
RAW_TRAIN = PROJECT_ROOT / "data" / "raw" / "toxic_train.csv"
MD_A = REPORTS / "p63_id_auditoria_predictiva.md"
JSON_A = REPORTS / "p63_id_auditoria_predictiva.json"
CSV_A = REPORTS / "p63_id_auditoria_predictiva.csv"
CELDA_A = PROJECT_ROOT / "features" / "p63_id_auditoria_cells.npz"

V03_UMBRALES = np.array([0.44, 0.51, 0.46])
LR = {"C": 2.0, "class_weight": "balanced", "max_iter": 1000, "solver": "liblinear",
      "random_state": 42}
# Los parametros del bloque id se declaran aqui, antes de ejecutar, y no se ajustan despues.
N_BINS = 10
MODULOS = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29)
N_PERMUTACIONES = 20
GATE_DELTA = 0.0015
GATE_PLIEGUES = 4
SEMILLA = 42


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[P63-03] {mensaje}", flush=True)


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


def bloque_id(ids: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Construye el bloque de rasgos del id.

    Los rasgos se normalizan contra `ref`, que es el vector de ids del pliegue de
    entrenamiento. Asi el percentil de rango de una fila de validacion se mide contra la
    distribucion que el modelo ha visto, y no contra la distribucion completa, que seria
    informacion del conjunto de validacion.

    Args:
        ids: ids del conjunto que se quiere transformar.
        ref: ids del pliegue de entrenamiento.

    Returns:
        Matriz densa de rasgos.
    """
    ref_orden = np.sort(np.asarray(ref, dtype=np.int64))
    pct = np.searchsorted(ref_orden, ids) / max(len(ref_orden), 1)
    cols = [pct, (ids - ids.min()) / max(int(ids.max() - ids.min()), 1)]
    # bins gruesos sobre el percentil, uno por quintil declarado
    for b in range(1, N_BINS):
        cols.append((pct >= b / N_BINS).astype(float))
    for m in MODULOS:
        cols.append((ids % m).astype(float) / m)
    return np.column_stack(cols).astype(float)


def main() -> None:
    """Audita si el indice de coleccion porta senal, con control de capacidad y permutacion.

    Returns:
        None.
    """
    inicio = datetime.now(timezone.utc)
    log("P63-03 el indice de coleccion, control de capacidad y permutacion de etiquetas")
    log(f"   bloque id: percentil de rango, {N_BINS} bins y {len(MODULOS)} modulos")
    log(f"   {N_PERMUTACIONES} permutaciones de etiquetas, cada una con su propio par de "
        f"brazos")

    hash_real = hashlib_sha(V03_SUB)
    if hash_real != V03_SHA:
        raise SystemExit("PARADA: la submission de V03 no coincide con el sha256 congelado")

    v = np.load(OOF, allow_pickle=True)
    ids_oof, y, pliegues = v["ids"], v["y_true"].astype(int), v["pliegues"]
    pred_v03 = v["pred"].astype(int)
    ccv_ref = float(f1_score(y, pred_v03, average="macro"))
    if ccv_ref != V03_CCV:
        raise SystemExit(f"PARADA: el OOF canonico da {ccv_ref!r}, no {V03_CCV!r}")
    n = ids_oof.size
    tr = pd.read_csv(RAW_TRAIN, sep=";", encoding="latin-1", dtype={"id": str})
    tr["id_num"] = pd.to_numeric(tr["id"].str.replace('"', "", regex=False),
                                 errors="coerce").astype("int64")
    idx = tr.set_index("id_num").loc[ids_oof]
    if not np.array_equal(idx[["HS", "TR", "AG"]].to_numpy().astype(int), y):
        raise SystemExit("PARADA: las etiquetas del CSV crudo no coinciden con el OOF")
    log("")
    log(f"   OOF canonico {ccv_ref!r} | {n} filas | V03 sha256 verificado")

    n_cols = 2 + (N_BINS - 1) + len(MODULOS)
    log(f"   dimension del bloque id: {n_cols} columnas")

    def evaluar(y_obj: np.ndarray, rng: np.random.Generator) -> dict[str, Any]:
        """Evalua los cuatro brazos sobre unas etiquetas dadas.

        Args:
            y_obj: objetivo, ya permutado o no.
            rng: generador para el brazo aleatorio.

        Returns:
            Diccionario con las metricas de cada brazo.
        """
        out: dict[str, Any] = {}
        for brazo in ("id_solo", "random_solo"):
            pred = np.zeros((n, 3), dtype=int)
            for f in range(1, 6):
                ir, iva = np.where(pliegues != f)[0], np.where(pliegues == f)[0]
                if brazo == "id_solo":
                    xent = bloque_id(ids_oof[ir], ids_oof[ir])
                    xval = bloque_id(ids_oof[iva], ids_oof[ir])
                else:
                    # El control aleatorio tiene la MISMA dimension y el mismo escalado,
                    # y se genera de forma independiente por pliegue para que un valor alto
                    # no pueda deberse a un afortunado de la semilla global.
                    xent = rng.standard_normal((ir.size, n_cols))
                    xval = rng.standard_normal((iva.size, n_cols))
                esc = StandardScaler().fit(xent)
                xent, xval = esc.transform(xent), esc.transform(xval)
                for j in range(3):
                    mo = LogisticRegression(**LR)
                    mo.fit(xent, y_obj[ir, j])
                    p = mo.predict_proba(xval)[:, 1]
                    pred[iva, j] = (p >= 0.5).astype(int)
            out[brazo] = {
                "corrected_cv": float(f1_score(y_obj, pred, average="macro")),
                "folds": [float(f1_score(y_obj[pliegues == g], pred[pliegues == g],
                                         average="macro")) for g in range(1, 6)],
                "n_predichos": int(pred.sum()),
            }
        return out

    # ---------- brazo real y brazo de capacidad, con las etiquetas verdaderas ----------
    t0 = time.time()
    rng_real = np.random.default_rng(SEMILLA)
    real = evaluar(y, rng_real)
    log("")
    log(f"   BRAZO REAL, sobre las etiquetas verdaderas ({time.time() - t0:.0f} s)")
    for b in ("id_solo", "random_solo"):
        log(f"      {b:14} corrected_cv {real[b]['corrected_cv']:.10f} | positivos "
            f"predichos {real[b]['n_predichos']}")
    brecha = real["id_solo"]["corrected_cv"] - real["random_solo"]["corrected_cv"]
    log(f"      BRECHA id contra aleatorio: {brecha:+.10f}")
    log(f"      y V03 esta en {ccv_ref:.10f}, luego el bloque id queda POR DEBAJO de el en "
        f"{ccv_ref - real['id_solo']['corrected_cv']:.10f}")

    # ---------- el suelo de la permutacion ----------
    log("")
    log(f"   SUELO POR PERMUTACION, {N_PERMUTACIONES} permutaciones con etiquetas mezcladas")
    perm_id: list[float] = []
    perm_ale: list[float] = []
    rng_perm = np.random.default_rng(SEMILLA + 1)
    for p in range(N_PERMUTACIONES):
        yp = np.column_stack([rng_perm.permutation(y[:, j]) for j in range(3)])
        r = evaluar(yp, np.random.default_rng(SEMILLA + 100 + p))
        perm_id.append(r["id_solo"]["corrected_cv"])
        perm_ale.append(r["random_solo"]["corrected_cv"])
    p_id = np.array(perm_id)
    p_ale = np.array(perm_ale)
    log(f"      id_solo permutado: media {p_id.mean():.10f} | min {p_id.min():.10f} | "
        f"max {p_id.max():.10f}")
    log(f"      aleatorio permutado: media {p_ale.mean():.10f} | min {p_ale.min():.10f} | "
        f"max {p_ale.max():.10f}")
    log(f"      el brazo real queda {real['id_solo']['corrected_cv'] - p_id.max():+.10f} por "
        f"encima del MAXIMO de las {N_PERMUTACIONES} permutaciones del mismo brazo")

    # ---------- comparacion con V03 ----------
    delta_v03 = real["id_solo"]["corrected_cv"] - ccv_ref
    log("")
    log("   CONTRA V03")
    log(f"      delta {delta_v03:+.10f} | gate de la fase {GATE_DELTA:+.4f}, asi que el id "
        f"NO PUEDE alcanzar el gate ni aunque fuera perfecto, porque Va solo")

    # ---------- veredicto ----------
    supera_aleatorio = brecha > 0.0
    supera_permutacion = real["id_solo"]["corrected_cv"] > p_id.max()
    hay_senal = bool(supera_aleatorio and supera_permutacion)
    if hay_senal:
        veredicto = (
            f"el bloque id supera al control de capacidad en {brecha:+.10f} y queda "
            f"{real['id_solo']['corrected_cv'] - p_id.max():+.10f} por encima del maximo de "
            f"las {N_PERMUTACIONES} permutaciones, luego hay senal especifica entre el indice "
            f"y la etiqueta que no es capacidad ni ruido. Ahora la pregunta deja de ser si hay "
            f"senal y pasa a ser si esa senal es procedencia que transfiere, que es otra "
            f"cuestion mas dura y que NO se responde aqui")
        decision = "PROMISING_PARA_SEGUNDA_PARTICION. Hay senal; falta transferencia"
    else:
        partes = []
        if not supera_aleatorio:
            partes.append(f"El control de capacidad iguala o supera al bloque id, con brecha "
                          f"{brecha:+.10f}, luego no hay senal especifica y solo capacidad")
        if not supera_permutacion:
            partes.append("el brazo real no supera el maximo de las permutaciones, luego "
                          "parte de lo que parecio senal era ruido")
        veredicto = (". ".join(partes) + ". El id queda CLOSED como predictor: lo que "
                     "parecia estructura se disuelve en capacidad y permutacion, que es "
                     "justo lo que se esperaba de un indice de coleccion cuyo unico contenu "
                     "es la posicion en un ensamblado")
        decision = "CLOSED. El indice de coleccion no aporta senal especifica"
    log("")
    log(f"   VEREDICTO: {veredicto}")

    np.savez_compressed(CELDA_A, ids=ids_oof, y_true=y, pliegues=pliegues, pred_v03=pred_v03,
                        n_bins=np.array([N_BINS]), modulos=np.array(MODULOS),
                        perm_id=p_id, perm_ale=p_ale,
                        ccv_id_solo=np.array([real["id_solo"]["corrected_cv"]]),
                        ccv_random_solo=np.array([real["random_solo"]["corrected_cv"]]))

    filas = [
        {"brazo": "V03", "corrected_cv": round(ccv_ref, 10), "delta_vs_V03": 0.0,
         "descripcion": "control congelado, leido del OOF canonico"},
        {"brazo": "id_solo", "corrected_cv": round(real["id_solo"]["corrected_cv"], 10),
         "delta_vs_V03": round(delta_v03, 10),
         "descripcion": f"percentil de rango, {N_BINS} bins y {len(MODULOS)} modulos"},
        {"brazo": "random_solo",
         "corrected_cv": round(real["random_solo"]["corrected_cv"], 10),
         "delta_vs_V03": round(real["random_solo"]["corrected_cv"] - ccv_ref, 10),
         "descripcion": f"control de capacidad, {n_cols} rasgos aleatorios"},
        {"brazo": "id_solo_permutado_max", "corrected_cv": round(float(p_id.max()), 10),
         "delta_vs_V03": round(float(p_id.max() - ccv_ref), 10),
         "descripcion": f"maximo de {N_PERMUTACIONES} permutaciones, el suelo a superar"},
        {"brazo": "random_solo_permutado_media", "corrected_cv": round(float(p_ale.mean()), 10),
         "delta_vs_V03": round(float(p_ale.mean() - ccv_ref), 10),
         "descripcion": "media del control de capacidad con etiquetas mezcladas"},
    ]
    pd.DataFrame(filas).to_csv(CSV_A, index=False, encoding="utf-8")

    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                            text=True, cwd=PROJECT_ROOT, check=False).stdout.strip()
    informe = {
        "familia": "P63_ID_AUDITORIA_PREDICTIVA",
        "naturaleza": ("EXPERIMENTO. 2 brazos reales y 2 controles, 0 submissions, 0 Kaggle, "
                       "0 holdout, 0 test, 0 leaderboard"),
        "fecha": inicio.isoformat(timespec="seconds"),
        "experiment_id": "P63-02",
        "track": "A-id-predictivo",
        "script": "experiments/p63_03_id_auditoria_predictiva.py",
        "pregunta": "el indice de coleccion porta senal especifica, o es capacidad y ruido",
        "por_que_tres_controles": (
            "un modelo que lee el indice seria el positivo mas facil de falsificar del "
            "proyecto, y por una razon estructural: si el corpus se assemblo en bloques y las "
            "etiquetas derivan entre bloques por motivos que no tienen que ver con el texto, "
            "el modelo lee el orden de ensamblado, brilla en validacion cruzada y muere en el "
            "leaderboard"),
        "controles": {
            "capacidad": (f"{n_cols} rasgos aleatorios de la misma dimension y el mismo "
                          f"escalado, generados por pliegue"),
            "permutacion": (f"{N_PERMUTACIONES} permutaciones de las etiquetas, que destruyen "
                            f"cualquier relacion id-etiqueta y conservan la distribucion de ids "
                            f"y las tasas marginales"),
            "segunda_particion": ("NO se ejecuta. El encargo prohibe usar el holdout durante "
                                  "la exploracion, y esto sigue siendo exploracion. Se reserva "
                                  "para un finalista"),
        },
        "parametros_del_bloque": {"bins": N_BINS, "modulos": list(MODULOS),
                                  "n_columnas": n_cols,
                                  "declarados_antes": True},
        "lo_no_usado": ["el id crudo como categoria", "millones de cubos",
                        "target encoding", "label lookup"],
        "resultados": real,
        "permutaciones": {
            "id_solo": {"media": round(float(p_id.mean()), 10),
                        "min": round(float(p_id.min()), 10),
                        "max": round(float(p_id.max()), 10),
                        "n": N_PERMUTACIONES},
            "random_solo": {"media": round(float(p_ale.mean()), 10),
                            "min": round(float(p_ale.min()), 10),
                            "max": round(float(p_ale.max()), 10),
                            "n": N_PERMUTACIONES},
        },
        "brecha_contra_capacidad": round(brecha, 10),
        "exceso_contra_el_maximo_permutado": round(
            real["id_solo"]["corrected_cv"] - float(p_id.max()), 10),
        "delta_vs_V03": round(delta_v03, 10),
        "gate_de_la_fase": GATE_DELTA,
        "nota_sobre_el_gate": ("el id va solo, luego no PUEDE alcanzar el gate de la fase ni "
                               "aunque fuera perfecto, porque compite contra un modelo que "
                               "tiene todo el texto. Su valor es diagnnstico: dice si la "
                               "estructura del indice contiene informacion de etiqueta"),
        "hay_senal_especifica": hay_senal,
        "veredicto": veredicto,
        "decision": decision,
        "experimentos_registrados": 1,
        "reproducibilidad": {"nivel": "R2", "git_commit": commit, "seed": SEMILLA,
                             "hash_v03": hash_real, "corrected_cv": V03_CCV},
        "formulacion": "V03 remains the best verified leaderboard result.",
    }
    JSON_A.write_text(json.dumps(informe, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")

    L: list[str] = []
    W = L.append
    W("# P63 - Auditoria predictiva del indice de coleccion")
    W("")
    W(f"Generado por `experiments/p63_03_id_auditoria_predictiva.py`. Fecha "
      f"{inicio.isoformat(timespec='seconds')}.")
    W("")
    W("## La pregunta, y por que merece un experimento")
    W("")
    W("P63-02 clasifico el id como `D. STRUCTURED_IDENTIFIER` con confianza alta: un entero")
    W("dentro del bloque contiguo [20001, 24999], del que train y test son muestras disjuntas")
    W("que en conjunto ocupan 2885 de 4999 posiciones. Eso es estructura, y la estructura es")
    W("una precondicion de la senal, no una garantia.")
    W("")
    W("El gap analysis encontro ademas que el id **nunca fue entrada de un modelo**: V15-00 lo")
    W("excluyo declarandolo contador, y P63-02 demonstro que no lo es. La pregunta viva es")
    W("estrecha y concreta: el indice porta informacion de etiqueta que V03 no tenga ya?")
    W("")
    W("## Tres controles, porque un positivo aqui seria el artefacto mas facil del proyecto")
    W("")
    W("Un modelo que leyera el indice seria el positivo mas facil de falsificar del proyecto,")
    W("y por una razon estructural, no moral. Si el corpus se ensamblo en bloques y las")
    W("etiquetas derivan entre bloques por motivos que no tienen que ver con el texto, el")
    W("modelo lee el **orden de ensamblado**, brilla en validacion cruzada y muere en el")
    W("leaderboard. Tres controles separan eso de un efecto real:")
    W("")
    W("- **capacidad**: el mismo modelo sobre rasgos ALEATORIOS de dimension identica. Si el")
    W("  bloque real puntua como el aleatorio, no hay senal especifica, solo la capacidad de")
    W("  memorizar una funcion suave de 1483 filas.")
    W("- **permutacion**: el mismo pipeline con las etiquetas mezcladas. Destruye cualquier")
    W("  relacion id-etiqueta y conserva la distribucion de ids y las tasas marginales.")
    W("- **segunda particion**: **NO se ejecuta.** El encargo prohibe usar el holdout durante")
    W("  la exploracion, y esto sigue siendo exploracion. Se reserva para un finalista.")
    W("")
    W("## El bloque")
    W("")
    W(f"Percentil de rango del id contra el pliegue de entrenamiento, {N_BINS} bins gruesos y")
    W(f"{len(MODULOS)} modulos pequenos: **{n_cols} columnas**. Deliberadamente NO se usa el id")
    W("crudo como categoria, ni millones de cubos, ni target encoding, ni label lookup, que")
    W("serian cuatro maneras de memorizar.")
    W("")
    W("## Resultados")
    W("")
    W("| brazo | corrected_cv | delta vs V03 | descripcion |")
    W("|---|---:|---:|---|")
    for f in filas:
        W(f"| {f['brazo']} | {f['corrected_cv']:.10f} | {f['delta_vs_V03']:+.10f} | "
          f"{f['descripcion']} |")
    W("")
    W("### El suelo de la permutacion")
    W("")
    W("| brazo | media | minimo | maximo |")
    W("|---|---:|---:|---:|")
    W(f"| id_solo con etiquetas mezcladas | {p_id.mean():.10f} | {p_id.min():.10f} | "
      f"{p_id.max():.10f} |")
    W(f"| random_solo con etiquetas mezcladas | {p_ale.mean():.10f} | {p_ale.min():.10f} | "
      f"{p_ale.max():.10f} |")
    W("")
    W(f"El brazo real queda **{real['id_solo']['corrected_cv'] - p_id.max():+.10f}** por encima")
    W(f"del maximo de las {N_PERMUTACIONES} permutaciones del mismo brazo, y la brecha contra")
    W(f"el control de capacidad es **{brecha:+.10f}**.")
    W("")
    W("### Una nota sobre el gate")
    W("")
    W(f"El delta contra V03 es {delta_v03:+.10f}. El id va **solo**, luego no PUEDE alcanzar el")
    W(f"gate de la fase de {GATE_DELTA:+.4f} ni aunque fuera perfecto, porque compite contra")
    W("un modelo que tiene todo el texto. Su valor es diagnostico: dice si la estructura del")
    W("indice contiene informacion de etiqueta, no si mejora el modelo.")
    W("")
    W("## Veredicto")
    W("")
    W("```text")
    W(veredicto)
    W("```")
    W("")
    W("```text")
    W(decision)
    W("```")
    W("")
    W("> V03 remains the best verified leaderboard result.")
    W("")
    W("*Generado por `experiments/p63_03_id_auditoria_predictiva.py`. Sin push.*")
    MD_A.write_text(a_ascii("\n".join(L)), encoding="ascii")

    for destino in (MD_A, JSON_A, CSV_A):
        if any(b > 127 for b in destino.read_bytes()):
            raise SystemExit(f"PARADA: {destino.name} conserva bytes no ASCII")
    log("")
    log(f"P63-03 OK | id_solo {real['id_solo']['corrected_cv']:.6f} | aleatorio "
        f"{real['random_solo']['corrected_cv']:.6f} | brecha {brecha:+.6f} | permutacion max "
        f"{p_id.max():.6f} | senal especifica {hay_senal}")


def hashlib_sha(ruta: Path) -> str:
    """Devuelve el sha256 de un fichero.

    Args:
        ruta: fichero.

    Returns:
        El hash en hexadecimal.
    """
    return hashlib.sha256(ruta.read_bytes()).hexdigest()


if __name__ == "__main__":
    ast.parse(Path(__file__).read_text(encoding="utf-8"))
    main()
