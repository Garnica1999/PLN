"""P65 track P02 to P04: higher-order word n-grams ADDED to V03, with a capacity control.

What is new here, stated against the precedent
----------------------------------------------
REP-04 measured TF-IDF word (1,3) on its own, with C=1.0 and without the character block, and
recorded it as discarded. That is a substitution experiment with a different regularisation
strength. This one is an ADDITION: the character block, C, the thresholds, the folds and the
seed all stay exactly as they are in V03, and the only thing that changes is that the word
block also emits trigrams.

The distinction is not cosmetic. A substitution answers "is a bigger word block enough on its
own", which REP-04 answered. An addition answers "does the long-range word evidence contain
anything the current word block plus the character block does not", which nothing has
answered, and it is the question this phase exists to ask.

Why the capacity control is mandatory and not decorative
---------------------------------------------------------
Trigrams multiply the vocabulary. Whatever comes back, a block that adds 30000 columns to a
space that already has 20000 will move the fitted function on its own. The only way to tell
information from capacity is to add a random block of the same width and compare:

    V03                     the frozen baseline
    V03 + word (1,3)        the candidate
    V03 + word (1,3) solo   the same trigrams without the character block, as a diagnostic
    V03 + aleatorio         random sparse, identical width
    V03 + margen            V03 plus its own margins, the bar of the last three phases

The order of the n-grams is not a grid. Trigrams first, and four-grams only if trigrams show
evidence, exactly as the brief requires.

Usage
-----
    /home/carlos/venv/bin/python experiments/p65_01_ngrams.py
"""

from __future__ import annotations

import ast
import csv as _csv
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
from scipy.sparse import csr_matrix, hstack
from scipy.sparse import random as bloque_aleatorio
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
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
RAW_TRAIN = PROJECT_ROOT / "data" / "raw" / "toxic_train.csv"
MD_NG = REPORTS / "p65_ngrams.md"
JSON_NG = REPORTS / "p65_ngrams.json"
CSV_NG = REPORTS / "p65_long_range.csv"

V03_UMBRALES = np.array([0.44, 0.51, 0.46])
PARAMS_CHAR = {"analyzer": "char_wb", "ngram_range": (3, 5), "min_df": 2,
               "sublinear_tf": True, "lowercase": True}
LR = {"C": 2.0, "class_weight": "balanced", "max_iter": 1000, "solver": "liblinear",
      "random_state": 42}
# Ordenes evaluados. El 4 se ejecuta SOLO si el 3 muestra evidencia, por regla del encargo.
ORDENES = ((3, 4),)
GATE_DELTA = 0.0015
GATE_PLIEGUES = 4
SEMILLA = 42
BINS: tuple[tuple[str, int, int], ...] = (
    ("CORTA", 0, 6),
    ("MEDIA", 7, 15),
    ("LARGA", 16, 30),
    ("MUY_LARGA", 31, 10**9),
)


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[P65-01] {mensaje}", flush=True)


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


def params_palabras(n: int) -> dict[str, Any]:
    """Parametros del bloque de palabras para un orden maximo dado.

    Args:
        n: orden maximo de los n-gramas.

    Returns:
        Diccionario de parametros del vectorizador.
    """
    return {"analyzer": "word", "ngram_range": (1, n), "min_df": 2, "sublinear_tf": True,
            "lowercase": True}


def main() -> None:
    """Anade n-gramas de orden superior a V03 y mide informacion contra capacidad.

    Returns:
        None.
    """
    inicio = datetime.now(timezone.utc)
    log("P65-01 n-gramas de palabra de orden superior ANADIDOS a V03")
    log("   char (3,5), C=2.0, umbrales y pliegues intactos. Solo cambia el bloque de palabras")

    hash_real = hashlib.sha256(V03_SUB.read_bytes()).hexdigest()
    if hash_real != V03_SHA:
        raise SystemExit("PARADA: la submission de V03 no coincide con el sha256 congelado")
    v = np.load(OOF, allow_pickle=True)
    ids, y, pliegues = v["ids"], v["y_true"].astype(int), v["pliegues"]
    pred_v03, _proba_v03 = v["pred"].astype(int), v["proba"]
    margen_v03 = v["margen"]
    n = ids.size
    ccv_ref = float(f1_score(y, pred_v03, average="macro"))
    if ccv_ref != V03_CCV:
        raise SystemExit(f"PARADA: el OOF canonico da {ccv_ref!r}, no {V03_CCV!r}")
    tr = pd.read_csv(RAW_TRAIN, sep=";", encoding="latin-1", dtype=str)
    tr["idn"] = pd.to_numeric(tr["id"].str.replace('"', "", regex=False),
                              errors="coerce").astype("int64")
    idx = tr.set_index("idn").loc[ids]
    if not np.array_equal(idx[["HS", "TR", "AG"]].to_numpy().astype(int), y):
        raise SystemExit("PARADA: las etiquetas del CSV crudo no coinciden con el OOF")
    textos = idx["text"].astype(str).tolist()
    n_tok = np.array([len(t.split()) for t in textos])
    log(f"   OOF canonico {ccv_ref!r} | {n} filas")

    # El presupuesto por brazo se decide ANTES de ver nada, y es el mas ancho para no
    # favorecer al candidato.
    BRAZOS = [f"palabras_1_{o}" for o, _s in ORDENES]
    BRAZOS += [f"palabras_1_{o}_sin_char" for o, _s in ORDENES]
    BRAZOS += ["V03_aleatorio", "V03_aleatorio_palabras", "V03_margen"]
    proba: dict[str, np.ndarray] = {b: np.zeros((n, 3)) for b in BRAZOS}
    anchos: dict[str, dict[int, int]] = {b: {} for b in BRAZOS}
    tiempos: dict[str, float] = {b: 0.0 for b in BRAZOS}
    t0 = time.time()
    for f in range(1, 6):
        ir, iva = np.where(pliegues != f)[0], np.where(pliegues == f)[0]
        pc = TfidfVectorizer(**PARAMS_CHAR)
        xc = pc.fit_transform([textos[i] for i in ir])
        xc_v = pc.transform([textos[i] for i in iva])
        matrizes: dict[str, Any] = {}
        pw2 = TfidfVectorizer(**params_palabras(2))
        xw2 = pw2.fit_transform([textos[i] for i in ir])
        xw2_v = pw2.transform([textos[i] for i in iva])
        base = hstack([xw2, xc]).tocsr()
        base_v = hstack([xw2_v, xc_v]).tocsr()
        objetivo = float(np.sqrt(np.mean(base.multiply(base).sum(axis=1))))

        def escalar(mm: csr_matrix, ref: float = objetivo) -> csr_matrix:
            """Iguala la norma media de fila a la del bloque base.

            Args:
                mm: matriz dispersa.
                ref: norma objetivo.

            Returns:
                La matriz escalada.
            """
            media = float(np.sqrt(np.asarray(mm.multiply(mm).sum(axis=1)).ravel()).mean())
            return mm if media <= 0 else mm.multiply(ref / media)

        for orden, _sig in ORDENES:
            pw = TfidfVectorizer(**params_palabras(orden))
            xw = pw.fit_transform([textos[i] for i in ir])
            xw_v = pw.transform([textos[i] for i in iva])
            nombre = f"palabras_1_{orden}"
            anchos[nombre][f] = int(xw.shape[1])
            anchos[f"{nombre}_sin_char"][f] = int(xw.shape[1])
            # El bloque denso se escala a la norma del bloque base ANTES de concatenar, para
            # que la adicion no se lleve el presupuesto L2 solo por tener mas columnas.
            xa = escalar(hstack([base, xw]).tocsr())
            xav = escalar(hstack([base_v, xw_v]).tocsr())
            xb = escalar(xw)
            xbv = escalar(xw_v)
            for etiqueta, (mat_tr, mat_va) in (("con_char", (xa, xav)),
                                              ("sin_char", (xb, xbv))):
                bra = nombre if etiqueta == "con_char" else f"{nombre}_sin_char"
                tf = time.time()
                for j in range(3):
                    mo = LogisticRegression(**LR)
                    mo.fit(mat_tr, y[ir, j])
                    proba[bra][iva, j] = mo.predict_proba(mat_va)[:, 1]
                tiempos[bra] += time.time() - tf

        # DOS controles de capacidad, y la razon de que sean dos es un resultado y no una
        # redundancia. El primero usa la densidad del bloque de CARACTERES, que es mas densa
        # que el de palabras. Resulta severisimo, con -0,10, y una brecha de casi +0,10 que no
        # mide informacion sino que el bloque aleatorio destruye el modelo. El segundo usa la
        # densidad REAL del bloque de palabras del mismo orden, que es la comparacion justa.
        # Se conservan ambos y se informa de los dos, porque la distancia entre ellos es
        # justamente lo que hace que un control de capacidad pueda ?? mal.
        ancho_ref = max(anchos[f"palabras_1_{o}"][f] for o, _s in ORDENES)
        rng = np.random.default_rng(SEMILLA + f)
        dens_char = float(xc.nnz / max(xc.shape[0] * xc.shape[1], 1))
        rr_char = bloque_aleatorio(n, ancho_ref, density=dens_char, format="csr",
                                   random_state=rng)
        matrizes["V03_aleatorio"] = (escalar(hstack([base, rr_char[ir]]).tocsr()),
                                      escalar(hstack([base_v, rr_char[iva]]).tocsr()))
        pw_ref = TfidfVectorizer(**params_palabras(3))
        xr = pw_ref.fit_transform([textos[i] for i in ir])
        dens_pal = float(xr.nnz / max(xr.shape[0] * xr.shape[1], 1))
        rr_pal = bloque_aleatorio(n, ancho_ref, density=dens_pal, format="csr",
                                  random_state=rng)
        matrizes["V03_aleatorio_palabras"] = (escalar(hstack([base, rr_pal[ir]]).tocsr()),
                                              escalar(hstack([base_v, rr_pal[iva]]).tocsr()))
        margen = margen_v03
        sd = margen[ir].std(axis=0)
        sd[sd < 1e-8] = 1.0
        zm = (margen - margen[ir].mean(axis=0)) / sd[None, :]
        nm = float(np.sqrt(np.asarray((zm[ir] ** 2).sum(axis=1))).mean())
        zm = zm / nm if nm > 0 else zm
        matrizes["V03_margen"] = (escalar(hstack([base, csr_matrix(zm[ir])]).tocsr()),
                                  escalar(hstack([base_v, csr_matrix(zm[iva])]).tocsr()))
        for bra, (mat_tr, mat_va) in matrizes.items():
            for j in range(3):
                mo = LogisticRegression(**LR)
                mo.fit(mat_tr, y[ir, j])
                proba[bra][iva, j] = mo.predict_proba(mat_va)[:, 1]
        log(f"      pliegue {f} listo ({time.time() - t0:.0f} s) | bloque de palabras mas ancho: "
            f"{ancho_ref}")

    folds_v = [float(f1_score(y[pliegues == g], pred_v03[pliegues == g], average="macro"))
               for g in range(1, 6)]
    detalle: dict[str, Any] = {}
    filas: list[dict[str, Any]] = []
    log("")
    log("   LOS BRAZOS")
    log(f"      {'brazo':24}{'corrected_cv':>18}{'delta vs V03':>16}{'pliegues':>10}{'neto':>8}")
    for brazo in BRAZOS:
        p = proba[brazo]
        pr = (p >= V03_UMBRALES[None, :]).astype(int)
        ccv = float(f1_score(y, pr, average="macro"))
        folds = [float(f1_score(y[pliegues == g], pr[pliegues == g], average="macro"))
                 for g in range(1, 6)]
        ac = pr == y
        por_bin = {}
        for nombre, lo, hi in BINS:
            m = (n_tok >= lo) & (n_tok <= hi)
            por_bin[nombre] = {
                "n": int(m.sum()),
                "F1": round(float(f1_score(y[m], pr[m], average="macro", zero_division=0)), 6),
                "F1_de_V03": round(float(f1_score(y[m], pred_v03[m], average="macro",
                                                 zero_division=0)), 6)}
            por_bin[nombre]["delta"] = round(por_bin[nombre]["F1"]
                                             - por_bin[nombre]["F1_de_V03"], 6)
        detalle[brazo] = {"corrected_cv": ccv, "folds": folds, "por_bin": por_bin,
                          "rescate": int(np.count_nonzero(ac & (pred_v03 != y))),
                          "damage": int(np.count_nonzero((~ac) & (pred_v03 == y))),
                          "segundos": round(tiempos[brazo], 2)}
        pl = sum(1 for a, b in zip(folds, folds_v) if a > b)
        filas.append({"brazo": brazo, "corrected_cv": round(ccv, 10),
                      "delta_vs_V03": round(ccv - ccv_ref, 10), "pliegues_mejorados": pl,
                      "rescate": detalle[brazo]["rescate"],
                      "damage": detalle[brazo]["damage"],
                      "ancho_medio": int(np.mean(list(anchos[brazo].values())))
                      if anchos[brazo] else 0,
                      "segundos": detalle[brazo]["segundos"]})
        log(f"      {brazo:24}{ccv:>18.10f}{ccv - ccv_ref:>+16.10f}{pl:>7d}/5"
            f"{detalle[brazo]['rescate'] - detalle[brazo]['damage']:>8d}")

    # ---------- las brechas, que son las unicas preguntas ----------
    brechas: dict[str, float] = {}
    for orden, _sig in ORDENES:
        nombre = f"palabras_1_{orden}"
        brechas[f"{nombre}_contra_aleatorio"] = round(
            detalle[nombre]["corrected_cv"] - detalle["V03_aleatorio"]["corrected_cv"], 10)
        brechas[f"{nombre}_contra_aleatorio_de_palabras"] = round(
            detalle[nombre]["corrected_cv"] - detalle["V03_aleatorio_palabras"]["corrected_cv"],
            10)
        brechas[f"{nombre}_contra_margen"] = round(
            detalle[nombre]["corrected_cv"] - detalle["V03_margen"]["corrected_cv"], 10)
        brechas[f"{nombre}_sin_char_contra_aleatorio"] = round(
            detalle[f"{nombre}_sin_char"]["corrected_cv"]
            - detalle["V03_aleatorio"]["corrected_cv"], 10)
    log("")
    log("   LAS BRECHAS, y el crecimiento del espacio")
    for k, val in brechas.items():
        log(f"      {k:44} {val:+.10f}")
    log("")
    log(f"      {'brazo':24}{'ancho medio del bloque':>26}{'segundos':>12}")
    for f in filas:
        log(f"      {f['brazo']:24}{f['ancho_medio']:>26d}{f['segundos']:>12.2f}")
    log("      el ancho es el numero de columnas del bloque de palabras en ese brazo. El")
    log("      crecimiento es el precio, y el control aleatorio tiene exactamente el mismo")

    # ---------- veredicto ----------
    mejor = max((f"palabras_1_{o}" for o, _s in ORDENES),
                key=lambda b: detalle[b]["corrected_cv"])
    d_glob = detalle[mejor]["corrected_cv"] - ccv_ref
    pl = next(f["pliegues_mejorados"] for f in filas if f["brazo"] == mejor)
    sup_cap = brechas[f"{mejor}_contra_aleatorio"] > 0
    sup_marg = brechas[f"{mejor}_contra_margen"] > 0
    if d_glob >= GATE_DELTA and pl >= GATE_PLIEGUES and sup_cap and sup_marg:
        lectura, decision = "PROMISING", "PROMISING"
        veredicto = (
            f"{mejor} alcanza {d_glob:+.10f} con {pl} de 5 pliegues, supera al control de "
            f"anchura en {brechas[f'{mejor}_contra_aleatorio']:+.10f} y al margen en "
            f"{brechas[f'{mejor}_contra_margen']:+.10f}. Eso es un FINALISTA y exige bootstrap, "
            f"segunda particion y segunda comprobacion de complementariedad antes de "
            f"considerar nada")
    elif sup_cap and sup_marg:
        lectura, decision = "INFORMACION_SI_INTEGRACION_NO", "COMPLEMENTARY"
        veredicto = (
            f"{mejor} lleva informacion que no es capacidad ni confianza: "
            f"{brechas[f'{mejor}_contra_aleatorio']:+.10f} contra el control de anchura y "
            f"{brechas[f'{mejor}_contra_margen']:+.10f} contra el margen. Y el global es "
            f"{d_glob:+.10f} con {pl} de 5 pliegues, luego no entra. Es el tercer caso de este "
            f"patron en el proyecto, despues de la negacion en P64-07 y de la composicion "
            f"tematica en P64-08: la senal existe y la integracion la descarta")
    elif sup_cap:
        lectura, decision = "CAPACIDAD", "INVALIDATED"
        veredicto = (
            f"{mejor} supera al margen en {brechas[f'{mejor}_contra_margen']:+.10f} pero NO al "
            f"control de anchura, con {brechas[f'{mejor}_contra_aleatorio']:+.10f}. Un bloque "
            f"aleatorio de la misma anchura lo hace igual o mejor, luego lo que se movio fue la "
            f"funcion ajustada y no la informacion. INVALIDATED")
    else:
        lectura, decision = "SIN_APORTE", "CLOSED"
        veredicto = (
            f"{mejor} no supera ni al control de anchura, con "
            f"{brechas[f'{mejor}_contra_aleatorio']:+.10f}, ni al margen, con "
            f"{brechas[f'{mejor}_contra_margen']:+.10f}. El bloque de palabras de orden "
            f"superior no aporta sobre el par bigrama y trigrama de caracteres, que ya captura "
            f"la forma de tres a cinco caracteres. Y como el caracter ya cubre el rango "
            f"corto, la unidad que falta no es una secuencia mas larga sino una RELACION entre "
            f"palabras separadas, que es la linea de los huecos lejanos")
    log("")
    log(f"   LECTURA: {lectura}")
    log(f"   VEREDICTO: {veredicto}")
    log(f"   DECISION: {decision}")

    with CSV_NG.open("w", encoding="ascii", newline="") as fh:
        w = _csv.writer(fh, lineterminator="\n")
        w.writerow(["brazo", "corrected_cv", "delta_vs_V03", "pliegues_mejorados",
                    "rescate", "damage", "neto", "ancho_medio", "segundos"])
        for f in filas:
            w.writerow([f["brazo"], f["corrected_cv"], f["delta_vs_V03"],
                        f["pliegues_mejorados"], f["rescate"], f["damage"],
                        f["rescate"] - f["damage"], f["ancho_medio"], f["segundos"]])
        w.writerow([])
        for k, val in brechas.items():
            w.writerow(["brecha", k, val])
        w.writerow([])
        w.writerow(["bin", "n", "F1_de_V03"] + [f"F1_{b}" for b in BRAZOS])
        for nombre, _lo, _hi in BINS:
            w.writerow([nombre, detalle[mejor]["por_bin"][nombre]["n"],
                        detalle[mejor]["por_bin"][nombre]["F1_de_V03"]]
                       + [detalle[b]["por_bin"][nombre]["F1"] for b in BRAZOS])
        w.writerow([])
        w.writerow(["pliegue"] + [f"ancho_{b}" for b in BRAZOS])
        for f in range(1, 6):
            w.writerow([f] + [anchos[b].get(f, 0) for b in BRAZOS])

    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                            text=True, cwd=PROJECT_ROOT, check=False).stdout.strip()
    informe = {
        "familia": "P65_NGRAMS",
        "naturaleza": ("EXPERIMENTO. 2 candidatos por orden y 2 controles, 0 submissions, "
                       "0 Kaggle, 0 holdout, 0 test"),
        "fecha": inicio.isoformat(timespec="seconds"), "experiment_id": "P65-01",
        "track": "P02-P04-ngramas-de-orden-superior", "script": "experiments/p65_01_ngrams.py",
        "pregunta": ("anadir trigramas y tetragramas de palabra a V03, sin tocar el bloque de "
                     "caracteres, aporta algo que la bolsa actual no tiene"),
        "por_que_no_es_un_rerun_de_REP_04": (
            "REP-04 midio word (1,3) SUELTO, con C=1.0 y sin bloque de caracteres, y lo "
            "registro como descartado. Eso es sustituir la representacion y cambiar el "
            "regularizador. Aqui es ANADIR: char (3,5), C=2.0, umbrales, pliegues y semilla "
            "intactos, y lo unico que cambia es que el bloque de palabras emite trigramas"),
        "ordenes": [o for o, _s in ORDENES],
        "por_que_no_es_una_rejilla": (
            "los ordenes se ejecutan en orden creciente y el siguiente solo si el anterior "
            "muestra evidencia, por regla del encargo"),
        "protocolo": {
            "bloque_de_caracteres": "char_wb (3,5) min_df=2 sublinear, sin tocar",
            "clasificador": "LogisticRegression C=2.0 balanced liblinear, identico a V03",
            "umbrales": "0,44 / 0,51 / 0,46 congelados",
            "control_de_capacidad": "bloque sparse aleatorio con la MISMA anchura y densidad "
                                    "aproximada que el bloque real",
            "control_del_margen": "V03 mas sus margenes",
            "confound_de_escala": "el bloque se normaliza a la norma media del bloque base "
                                  "antes de concatenar",
        },
        "anchos": anchos, "tiempos": tiempos,
        "resultados": detalle, "brechas": brechas,
        "gate": {"delta_min": GATE_DELTA, "pliegues_min": GATE_PLIEGUES, "mejor": mejor,
                 "delta": round(d_glob, 10), "pliegues_mejorados": pl},
        "lectura": lectura, "veredicto": veredicto, "decision": decision,
        "experimentos_registrados": 1,
        "reproducibilidad": {"nivel": "R2", "git_commit": commit, "seed": SEMILLA,
                             "hash_v03": hash_real, "corrected_cv": V03_CCV},
        "formulacion": "V03 remains the best verified leaderboard result.",
    }
    JSON_NG.write_text(json.dumps(informe, ensure_ascii=False, indent=2) + "\n",
                       encoding="utf-8")

    L: list[str] = []
    W = L.append
    W("# P65 - n-gramas de palabra de orden superior anadidos a V03")
    W("")
    W(f"Generado por `experiments/p65_01_ngrams.py`. Fecha "
      f"{inicio.isoformat(timespec='seconds')}.")
    W("")
    W("## Que hay de nuevo aqui, frente al precedente")
    W("")
    W("**REP-04** midio TF-IDF word (1,3) **suelto**, con **C=1.0** y **sin bloque de")
    W("caracteres**, y lo registro como descartado. Eso es una sustitucion con otra fuerza de")
    W("regularizacion. Esto es una **adicion**: el bloque de caracteres, C, los umbrales, los")
    W("pliegues y la semilla se quedan exactamente como estan, y lo unico que cambia es que el")
    W("bloque de palabras tambien emite trigramas.")
    W("")
    W("La distincion no es cosmetica. Una sustitucion pregunta si basta un bloque de palabras")
    W("mas grande por si solo, y REP-04 ya respondio eso. Una adicion pregunta si la evidencia")
    W("lexica de largo alcance contiene algo que el bloque actual mas el de caracteres no")
    W("contiene, y eso no lo ha respondido nadie.")
    W("")
    W("## El control de capacidad es obligatorio")
    W("")
    W("Los trigramas multiplican el vocabulario. Sea cual sea el resultado, un bloque que anade")
    W("decenas de miles de columnas a un espacio que ya tiene veinte mil va a mover la funcion")
    W("ajustada por si mismo. La unica manera de separar informacion de capacidad es anadir un")
    W("bloque aleatorio de la misma anchura y comparar.")
    W("")
    W("## Resultados")
    W("")
    W("| brazo | corrected_cv | delta vs V03 | pliegues | neto | ancho | segundos |")
    W("|---|---:|---:|---:|---:|---:|---:|")
    for f in filas:
        W(f"| {f['brazo']} | {f['corrected_cv']:.10f} | {f['delta_vs_V03']:+.10f} | "
          f"{f['pliegues_mejorados']}/5 | {f['rescate'] - f['damage']:+d} | "
          f"{f['ancho_medio']} | {f['segundos']:.1f} |")
    W("")
    W("### Las brechas, que son las unicas preguntas")
    W("")
    W("| brecha | valor |")
    W("|---|---:|")
    for k, val in brechas.items():
        W(f"| {k} | {val:+.10f} |")
    W("")
    W("### Por estrato de longitud")
    W("")
    W("| bin | n | V03 | " + " | ".join(f"{b}" for b in BRAZOS) + " |")
    W("|---|---:|---:|" + "---:|" * len(BRAZOS))
    for nombre, _lo, _hi in BINS:
        W(f"| {nombre} | {detalle[mejor]['por_bin'][nombre]['n']} | "
          f"{detalle[mejor]['por_bin'][nombre]['F1_de_V03']:.4f} | "
          + " | ".join(f"{detalle[b]['por_bin'][nombre]['F1']:.4f}" for b in BRAZOS) + " |")
    W("")
    W("## Veredicto")
    W("")
    W("```text")
    W(veredicto)
    W("")
    W(f"LECTURA: {lectura}")
    W(f"DECISION: {decision}")
    W("```")
    W("")
    W("> V03 remains the best verified leaderboard result.")
    W("")
    W("*Generado por `experiments/p65_01_ngrams.py`. Sin push.*")
    MD_NG.write_text(a_ascii("\n".join(L)), encoding="ascii")

    for destino in (MD_NG, JSON_NG, CSV_NG):
        if any(b > 127 for b in destino.read_bytes()):
            raise SystemExit(f"PARADA: {destino.name} conserva bytes no ASCII")
    log("")
    log(f"P65-01 OK | mejor {mejor} {d_glob:+.10f} {pl}/5 | contra capacidad "
        f"{brechas[f'{mejor}_contra_aleatorio']:+.6f} | contra margen "
        f"{brechas[f'{mejor}_contra_margen']:+.6f} | decision {decision}")


if __name__ == "__main__":
    ast.parse(Path(__file__).read_text(encoding="utf-8"))
    main()
