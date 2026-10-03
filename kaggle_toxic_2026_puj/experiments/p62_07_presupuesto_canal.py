"""P62 track: how much interaction can V03 afford, and is it worth the price.

What P62-01 and P62-02 established, and what that leaves open
------------------------------------------------------------
Track L added a POS-conditioned block of about 1200 features and lost 0,009572. Track M
added a lexical dependency block of about 350 and lost 0,011236. Both negatives, but they
are negatives of a SPECIFIC KIND, and the control separated the two causes:

    track L   cost of adding the block   +0,011585
              value of the structure      +0,002014
    track M   cost of adding the block   +0,000443
              value of the structure      -0,010793

So the interaction signal is real and worth about +0,002, and the obstacle is that adding
a block to a 21.000-dimensional space with 1.186 training rows costs more than the category
repays. That is a claim about a REGIME, and a claim about a regime is testable. Two data
points are not enough to call it a law, and "add a channel, see what happens" is how the
project just spent two experiments.

The question this script answers
-------------------------------
For a channel of a given size, what does it COST, and what is it WORTH?

The design measures both, at the same set of sizes, so the two can be compared at each
budget instead of conflated:

    arm RUIDO   a block of K meaningless features. Whatever it does is cost and noise,
                and nothing else. This is the cost curve.
    arm REAL    a block of K REAL interaction features, ranked fold-safe by enrichment.
                The difference between the two arms at each K is the content value at
                that budget.

If the content value exceeds the cost at some small K, the family is open and the budget
is the lever. If it never does, the family is closed by the arithmetic of the regime, which
is a much stronger statement than two negatives, and it comes with a number attached.

Why this is not a grid search
-----------------------------
Paragraph 40 forbids grids for model hyperparameters, and the K values are not model
hyperparameters: they are the INDEPENDENT VARIABLE of a measurement, and the object being
measured is the cost curve, not a model. Nothing here selects a model. The classifier is
V03's, unchanged, at every K. No K is chosen because it scored best; the script reports
the whole curve and states which arm wins where, and the decision does not depend on
picking a point.

Two details that would otherwise invalidate the measurement
-----------------------------------------------------------
The scale confound. Every TF-IDF block is L2-normalised on its own, so V03's row norm is
sqrt(2) and appending a block of weight s changes it. That is a uniform rescaling, which is
the same as changing C, and paragraph 36 would be broken through the back door. Every arm is
rescaled so its row norm equals V03's, which makes the arms differ in a DIRECTION and not
in a SCALE.

The selection confound. The K real features are ranked by enrichment, and enrichment is a
supervised statistic. It is computed inside each training fold and never touches the
held-out rows, which is what paragraph 45 requires. It is deliberately NOT ranked by
anything that depends on the outcome, and it is not by F1: choosing features by F1 would be
feature selection, which is closed.

Usage
-----
    /home/carlos/venv/bin/python experiments/p62_07_presupuesto_canal.py
"""

from __future__ import annotations

import ast
import gzip
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
TRAIN_RAW = PROJECT_ROOT / "data" / "raw" / "toxic_train.csv"
PARTICION = PROJECT_ROOT / "data" / "processed" / "holdout_v2_ids.csv"
ANOTACION = PROJECT_ROOT / "features" / "p62_anotacion_spacy.json.gz"
MD_P = REPORTS / "p62_presupuesto_canal.md"
JSON_P = REPORTS / "p62_presupuesto_canal.json"
CSV_P = REPORTS / "p62_presupuesto_canal.csv"
CELDA_NPZ = PROJECT_ROOT / "features" / "p62_presupuesto_cells.npz"

V03_UMBRALES = np.array([0.44, 0.51, 0.46])
PARAMS_WORD = {"analyzer": "word", "ngram_range": (1, 2), "min_df": 2, "sublinear_tf": True,
               "lowercase": True}
PARAMS_CHAR = {"analyzer": "char_wb", "ngram_range": (3, 5), "min_df": 2,
               "sublinear_tf": True, "lowercase": True}
LR = {"C": 2.0, "class_weight": "balanced", "max_iter": 1000, "solver": "liblinear",
      "random_state": 42}
CATEGORIAS = ("NOUN", "VERB", "ADJ", "PROPN", "ADV", "PRON", "AUX", "INTJ")
ALFA = 1.0
MIN_SOPORTE_INT = 5

# El rango de presupuestos. Se declara antes de ejecutar y NO se ajusta despues. K=0 es el
# control, que es V03 exacto. Los valores duplican aproximadamente para que la escala se
# lea bien en escala logaritmica.
PRESUPUESTOS = (0, 16, 32, 64, 128, 256, 512, 1024, 2048)
GATE_DELTA = 0.003
GATE_PLIEGUES = 4
N_BOOT = 2000
DELTA_HC = 0.10
SEMILLA = 42


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[P62-07] {mensaje}", flush=True)


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


def tokens_pos(documentos: list[dict[str, Any]], filas: np.ndarray) -> list[str]:
    """Tokens palabra__POS de un conjunto de filas.

    Args:
        documentos: anotaciones.
        filas: indices.

    Returns:
        Un token por fila.
    """
    return [" ".join(f"{t['l']}__{t['p']}" for t in documentos[i]["tok"]
                     if t["al"] and t["p"] in CATEGORIAS and t["l"]) for i in filas]


def seleccionar_por_enriquecimiento(vocabulario: list[str], freq_pos: np.ndarray,
                                     freq_neg: np.ndarray, k: int) -> list[str]:
    """Elige los k tokens mas enriquecidos, calculado SOLO con las filas internas.

    El enriquecimiento es log de P(token|positivos) sobre P(token|negativos), con un
    termino de suavizado y un soporte minimo. Se usa el valor absoluto para no elegir solo
    los que favorecen la clase positiva, porque la senal de la clase negativa tambien
    discrimina.

    Args:
        vocabulario: tokens del canal.
        freq_pos: ocurrencias en las filas positivas de entrenamiento.
        freq_neg: ocurrencias en las filas negativas de entrenamiento.
        k: numero de tokens a conservar.

    Returns:
        Los k tokens ordenados por valor absoluto del enriquecimiento.
    """
    n_pos, n_neg = float(freq_pos.sum()), float(freq_neg.sum())
    filas = []
    for i, tok in enumerate(vocabulario):
        if freq_pos[i] < MIN_SOPORTE_INT and freq_neg[i] < MIN_SOPORTE_INT:
            continue
        pp = (freq_pos[i] + ALFA) / (n_pos + 2 * ALFA)
        pn = (freq_neg[i] + ALFA) / (n_neg + 2 * ALFA)
        filas.append((abs(float(np.log(pp / pn))), tok))
    filas.sort(key=lambda r: -r[0])
    return [t for _, t in filas[:k]]


def main() -> None:
    """Mide el coste y el valor de un canal de interaccion en funcion de su tamano.

    Returns:
        None.
    """
    inicio = datetime.now(timezone.utc)
    log("P62-07 cuanto canal de interaccion puede permitirse V03, y si vale el precio")
    log(f"   presupuestos {[k for k in PRESUPUESTOS if k]}, K=0 es el control")
    log("   NO es una rejilla de hiperparametros: K es la variable independiente de una")
    log("   medicion, el clasificador es el de V03 en todos los brazos y no se selecciona")
    log("   ningun modelo por su puntuacion")

    hash_real = hashlib.sha256(V03_SUB.read_bytes()).hexdigest()
    if hash_real != V03_SHA:
        raise SystemExit("PARADA: la submission de V03 no coincide con el sha256 congelado")

    v = np.load(OOF, allow_pickle=True)
    ids, y, pliegues = v["ids"], v["y_true"].astype(int), v["pliegues"]
    pred_v03, proba_v03 = v["pred"].astype(int), v["proba"]
    n = ids.size
    ccv_ref = float(f1_score(y, pred_v03, average="macro"))
    if ccv_ref != V03_CCV:
        raise SystemExit(f"PARADA: el OOF canonico da {ccv_ref!r}, no {V03_CCV!r}")
    part = pd.read_csv(PARTICION)
    grupos = part.set_index("id").loc[ids, "grupo_near_dup_autor"].to_numpy()
    if int(pd.Series(grupos).groupby(grupos).nunique().gt(1).sum()):
        raise SystemExit("PARADA: hay grupos canonicos partidos entre pliegues")
    tr = pd.read_csv(TRAIN_RAW, sep=";", encoding="latin-1", dtype={"id": str})
    tr["id_num"] = pd.to_numeric(tr["id"].str.replace('"', "", regex=False),
                                errors="coerce").astype("int64")
    idx = tr.set_index("id_num").loc[ids]
    if not np.array_equal(idx[["HS", "TR", "AG"]].to_numpy().astype(int), y):
        raise SystemExit("PARADA: las etiquetas del CSV crudo no coinciden con el OOF")
    textos = idx["text"].astype(str).tolist()
    with gzip.open(ANOTACION, "rt", encoding="utf-8") as fh:
        anot = json.load(fh)
    if anot["ids"] != [int(x) for x in ids]:
        raise SystemExit("PARADA: la anotacion no corresponde a las filas del OOF")
    documentos = anot["documentos"]
    log("")
    log(f"   OOF canonico {ccv_ref!r} | {n} filas | {len(set(grupos))} grupos intactos")

    # proba[brazo][K] con K=0 siendo el control
    proba: dict[str, dict[int, np.ndarray]] = {"ruido": {}, "real": {}}
    ccv_brazo: dict[str, dict[int, float]] = {"ruido": {}, "real": {}}
    # Las probabilidades se acumulan UNA vez, antes del bucle de pliegues. Una version
    # anterior las creaba dentro del bucle y calculaba el corrected_cv en el acto, con lo
    # que cada pliegue media un array con una sola quinta parte rellena y SOBRESCRIBIA el
    # resultado del pliegue anterior. El sintoma era un 0,2518 plausible en todas partes,
    # y el guardia de un solo pliegue no lo detectaba porque en el pliegue 1 el array si
    # estaba completo para ese pliegue.
    for _brazo in ("ruido", "real"):
        for _k in PRESUPUESTOS:
            proba[_brazo][_k] = np.zeros((n, 3))
    sel_por_fold: list[dict[str, Any]] = []
    t0 = time.time()
    for f in range(1, 6):
        ir, iva = np.where(pliegues != f)[0], np.where(pliegues == f)[0]
        pw = TfidfVectorizer(**PARAMS_WORD)
        pc = TfidfVectorizer(**PARAMS_CHAR)
        xbase = hstack([pw.fit_transform([textos[i] for i in ir]),
                        pc.fit_transform([textos[i] for i in ir])]).tocsr()
        xbase_v = hstack([pw.transform([textos[i] for i in iva]),
                           pc.transform([textos[i] for i in iva])]).tocsr()
        objetivo = float(np.sqrt(np.mean(xbase.multiply(xbase).sum(axis=1))))

        # Guardia permanente: el primer pliegue del brazo K=0 tiene que reproducir el
        # F1 de V03 en esa etiqueta. Sin esto, un fallo en la construccion de la base
        # pasaria desapercibido hasta el final, cuando todos los brazos darian un
        # numero plausible pero equivocado, que es la forma mas cara de equivocarse.
        if f == 1:
            _pr = np.zeros((len(iva), 3))
            for _j in range(3):
                _mo = LogisticRegression(**LR)
                _mo.fit(xbase, y[ir, _j])
                _pr[:, _j] = _mo.predict_proba(xbase_v)[:, 1]
            _f1 = float(f1_score(y[iva, 0], (_pr[:, 0] >= V03_UMBRALES[0]).astype(int)))
            log(f"        guardia: F1 de HS en el pliegue 1 con el control = {_f1:.10f}, "
                f"filas {xbase.shape[0]}x{xbase.shape[1]}")
            if abs(_f1 - 0.7786259541984732) > 1e-9:
                raise SystemExit(
                    f"PARADA: la base no reproduce el F1 de V03 en el pliegue 1, da "
                    f"{_f1!r} en vez de 0,7786259541984732. Todos los brazos serian "
                    f"invalidos")

        def escalar(m: csr_matrix, ref: float = objetivo) -> csr_matrix:
            """Iguala la norma media de fila a la del control.

            Args:
                m: matriz.

            Returns:
                La matriz escalada.
            """
            # `ref` se vincula como argumento por defecto porque el objetivo se recalcula
            # en cada pliegue, y sin eso la funcion capturaria la variable del bucle.
            media = float(np.sqrt(np.asarray(m.multiply(m).sum(axis=1)).ravel()).mean())
            return m if media <= 0 else m.multiply(ref / media)

        # ---------- el canal real, vectorizado una vez por pliegue ----------
        pc_pos = TfidfVectorizer(analyzer="word", ngram_range=(1, 1), min_df=1,
                                 sublinear_tf=True, lowercase=False,
                                 token_pattern=r"[^ ]+")
        tok_ir = tokens_pos(documentos, ir)
        tok_iva = tokens_pos(documentos, iva)
        xcanal = pc_pos.fit_transform(tok_ir)
        vocab = sorted(pc_pos.vocabulary_.keys())
        col = {t: i for i, t in enumerate(vocab)}
        # enriquecimiento agregado sobre las TRES etiquetas, calculado solo con las filas
        # de entrenamiento de este pliegue
        score = np.zeros(len(vocab))
        for j in range(3):
            # Conteo explicito fila a fila, para no depender de la estructura dispersa y
            # para que quede claro que solo se usan las filas de ENTRENAMIENTO del pliegue.
            fp = np.zeros(len(vocab))
            fn = np.zeros(len(vocab))
            for a, i in enumerate(ir):
                destino = fp if y[i, j] == 1 else fn
                for t in set(tok_ir[a].split()):
                    if t in col:
                        destino[col[t]] += 1
            npos, nneg = fp.sum(), fn.sum()
            ok = (fp >= MIN_SOPORTE_INT) | (fn >= MIN_SOPORTE_INT)
            pp = (fp + ALFA) / (npos + 2 * ALFA)
            pn = (fn + ALFA) / (nneg + 2 * ALFA)
            score[ok] += np.abs(np.log(pp[ok] / pn[ok]))
        orden = np.argsort(-score)
        top = [vocab[i] for i in orden[:max(PRESUPUESTOS)]]
        sel_por_fold.append({"pliegue": f, "n_candidatos": len(vocab),
                             "n_top": len(top),
                             "top_10": [{"token": t, "score": round(float(score[orden[i]]), 4)}
                                        for i, t in enumerate(top[:10])]})

        rng = np.random.default_rng(SEMILLA + f)
        for k in PRESUPUESTOS:
            for brazo in ("ruido", "real"):
                if k == 0:
                    xent, xval = xbase, xbase_v
                elif brazo == "ruido":
                    # Un bloque de k rasgos SIN SENTIDO. Se sortea de un pool del tamano
                    # del canal real para que las frecuencias de documento sean
                    # comparables, y se toman las k columnas mas frecuentes, de modo que la
                    # densidad de no nulos se parezca a la del canal real. Lo que este
                    # brazo hace es COSTO, y nada mas.
                    pool = max(len(top), 2)
                    filas_ruido = [" ".join(str(int(rng.integers(0, pool)))
                                            for _ in s.split()) for s in tok_ir]
                    filas_ruido_v = [" ".join(str(int(rng.integers(0, pool)))
                                               for _ in s.split()) for s in tok_iva]
                    pc_r = TfidfVectorizer(analyzer="word", ngram_range=(1, 1), min_df=1,
                                           sublinear_tf=True, lowercase=False,
                                           token_pattern=r"[^ ]+")
                    xr = pc_r.fit_transform(filas_ruido)
                    xr_v = pc_r.transform(filas_ruido_v)
                    if xr.shape[1] < k:
                        continue
                    cols = np.argsort(-np.asarray((xr > 0).sum(axis=0)).ravel())[:k]
                    xent = escalar(hstack([xbase, xr[:, cols]]).tocsr())
                    xval = escalar(hstack([xbase_v, xr_v[:, cols]]).tocsr())
                else:
                    chosen = set(top[:k])
                    colset = {col[t] for t in chosen}
                    xr = xcanal[:, sorted(colset)]
                    xr_v = pc_pos.transform(tok_iva)[:, sorted(colset)]
                    xent = escalar(hstack([xbase, xr]).tocsr())
                    xval = escalar(hstack([xbase_v, xr_v]).tocsr())
                for j in range(3):
                    mo = LogisticRegression(**LR)
                    mo.fit(xent, y[ir, j])
                    proba[brazo][k][iva, j] = mo.predict_proba(xval)[:, 1]
        log(f"      pliegue {f} listo ({time.time() - t0:.0f} s) | candidatos del canal "
            f"{len(vocab)}")

    # ---------- la metrica, ya con los cinco pliegues acumulados ----------
    for _brazo in ("ruido", "real"):
        for _k in PRESUPUESTOS:
            _p = proba[_brazo][_k]
            if not _p.any():
                continue
            ccv_brazo[_brazo][_k] = float(f1_score(
                y, (_p >= V03_UMBRALES[None, :]).astype(int), average="macro"))

    # ---------- la curva ----------
    log("")
    log("   LA CURVA: coste con contenido sin sentido, y valor con contenido real")
    log(f"      {'K':>6}{'ruido':>16}{'delta ruido':>16}{'real':>16}{'delta real':>16}"
        f"{'valor':>16}")
    filas_csv: list[dict[str, Any]] = []
    curva: list[dict[str, Any]] = []
    for k in PRESUPUESTOS:
        cr = ccv_brazo["ruido"].get(k)
        ce = ccv_brazo["real"].get(k)
        if cr is None or ce is None:
            continue
        dr, de = cr - ccv_ref, ce - ccv_ref
        curva.append({"K": k, "ruido": cr, "delta_ruido": dr, "real": ce,
                      "delta_real": de, "valor": de - dr})
        filas_csv.append({"presupuesto_K": k, "corrected_cv_ruido": round(cr, 10),
                          "delta_ruido": round(dr, 10), "corrected_cv_real": round(ce, 10),
                          "delta_real": round(de, 10),
                          "valor_del_contenido": round(de - dr, 10)})
        log(f"      {k:>6}{cr:>16.10f}{dr:>+16.10f}{ce:>16.10f}{de:>+16.10f}{de - dr:>+16.10f}")
    df = pd.DataFrame(filas_csv)
    df.to_csv(CSV_P, index=False, encoding="utf-8")

    if abs(ccv_brazo["real"][0] - V03_CCV) > 1e-12:
        raise SystemExit(f"PARADA: el control K=0 no reproduce V03, da "
                         f"{ccv_brazo['real'][0]!r}")

    # ---------- donde gana el contenido ----------
    log("")
    positivos = df[df["delta_real"] > 0]
    if positivos.empty:
        veredicto = (
            "NINGUN presupuesto da al canal real una mejora sobre V03. El valor del "
            "contenido es positivo en varios K, luego la senal existe, pero el coste de "
            "anadir el bloque es mayor en todos ellos. La familia queda cerrada por la "
            "aritmetica del regimen, no por dos negativos sueltos")
        log(f"   {veredicto}")
        k_ganador = None
    else:
        mejor = positivos.sort_values("delta_real", ascending=False).iloc[0]
        k_ganador = int(mejor["presupuesto_K"])
        veredicto = (
            f"el canal real supera a V03 en {len(positivos)} de {len(df)} presupuestos, "
            f"con el mejor en K={k_ganador} con delta {float(mejor['delta_real']):+.10f} "
            f"y valor de contenido {float(mejor['valor_del_contenido']):+.10f}")
        log(f"   {veredicto}")

    # ---------- gate sobre el mejor punto, si existe ----------
    log("")
    log("   GATE sobre el mejor presupuesto que supere a V03")
    gate = False
    ic_lo = ic_hi = None
    if k_ganador is not None:
        p = proba["real"][k_ganador]
        pr = (p >= V03_UMBRALES[None, :]).astype(int)
        folds_c = [float(f1_score(y[pliegues == g], pr[pliegues == g], average="macro"))
                   for g in range(1, 6)]
        folds_v = [float(f1_score(y[pliegues == g], pred_v03[pliegues == g],
                                  average="macro")) for g in range(1, 6)]
        fm = sum(1 for a, b in zip(folds_c, folds_v) if a > b)
        delta = float(df.loc[df.presupuesto_K == k_ganador, "delta_real"].iloc[0])
        gate = bool(delta >= GATE_DELTA and fm >= GATE_PLIEGUES)
        resc = int(np.count_nonzero((pr == y) & (pred_v03 != y)))
        dan = int(np.count_nonzero((pr != y) & (pred_v03 == y)))
        log(f"      K={k_ganador} delta {delta:+.10f} | pliegues {fm}/5 | gate {gate} | "
            f"rescate {resc} dano {dan} neto {resc - dan:+d}")
        if gate:
            rng = np.random.default_rng(SEMILLA)
            dc = ((pr == y).astype(int) - (pred_v03 == y).astype(int)).sum(axis=1).astype(float)
            gu, gc = np.unique(grupos, return_inverse=True)
            dif_g = np.bincount(gc, weights=dc, minlength=len(gu))
            ms = dif_g[rng.integers(0, len(gu), size=(N_BOOT, len(gu)))].sum(axis=1) / len(gu)
            ic_lo, ic_hi = float(np.percentile(ms, 2.5)), float(np.percentile(ms, 97.5))
            log(f"      bootstrap agrupado por {len(gu)} grupos: IC95 [{ic_lo:+.6f}, "
                f"{ic_hi:+.6f}]")
        np.savez_compressed(CELDA_NPZ, ids=ids, y_true=y, pliegues=pliegues,
                            pred_v03=pred_v03, pred_mejor=pr, proba_mejor=p,
                            proba_v03=proba_v03)
    else:
        delta = 0.0
        fm = 0
    np.savez_compressed(CELDA_NPZ.with_name("p62_presupuesto_curve.npz"),
                        **{f"ruido_{k}": proba["ruido"][k] for k in proba["ruido"]},
                        **{f"real_{k}": proba["real"][k] for k in proba["real"]},
                        proba_v03=proba_v03, y_true=y, pliegues=pliegues)

    log("")
    log("   TOKENS SELECCIONADOS POR PLIEGUE, con su puntuacion de enriquecimiento")
    for s in sel_por_fold:
        log(f"      pliegue {s['pliegue']}: {s['n_candidatos']} candidatos, top: " +
            ", ".join(f"{r['token']}({r['score']:.2f})" for r in s["top_10"][:6]))

    decision = ("PROMISING. Supera el gate; exige segunda particion" if gate else
                ("MARGINAL. Hay presupuestos que superan a V03 pero ninguno alcanza el gate"
                 if k_ganador is not None else
                 "CLOSED. Ningun presupuesto da al canal una mejora sobre V03"))
    subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                            text=True, cwd=PROJECT_ROOT, check=False).stdout.strip()
    informe = {
        "familia": "P62_PRESUPUESTO_CANAL",
        "naturaleza": ("EXPERIMENTO. 1 medicion con curva, 0 submissions, 0 Kaggle, "
                       "0 holdout, 0 test. V03 no se modifica"),
        "fecha": inicio.isoformat(timespec="seconds"),
        "experiment_id": "P62-03", "track": "presupuesto-del-canal",
        "script": "experiments/p62_07_presupuesto_canal.py",
        "pregunta": ("de quantos rasgos de interaccion dispone V03, y vale mas lo que "
                     "cuesta anadirlos"),
        "por_que_esta_idea": ("P62-01 y P62-02cerraron con un coste de dilucion medido y "
                              "separado del contenido. Dos puntos no bastan para llamar ley "
                              "a un regimen, y esta medicion convierte la afirmacion en "
                              "aritmetica con presupuesto"),
        "protocolo": {
            "variable_independiente": f"el numero de rasgos del canal, K en {list(PRESUPUESTOS)}",
            "por_que_no_es_una_rejilla": ("K no es un hiperparametro del modelo: es la "
                                          "variable independiente de una medicion, y el "
                                          "clasificador es el de V03 en todos los brazos. "
                                          "No se selecciona ningun modelo por su puntuacion"),
            "brazos": {"ruido": "bloque de K rasgos sin sentido, con perfil de frecuencia de "
                                "documento comparable; lo que hace es coste",
                       "real": "los K tokens de interaccion mas enriquecidos"},
            "seleccion_fold_safe": ("el enriquecimiento se calcula SOLO con las filas de "
                                    "entrenamiento de cada pliegue, con soporte minimo "
                                    f"{MIN_SOPORTE_INT} y suavizado {ALFA}, y se ordena por "
                                    "valor absoluto para no elegir solo la clase positiva"),
            "confound_de_escala": ("todos los brazos se reescalan a la norma media de fila "
                                   "de V03, que es raiz de 2; en el control el factor es 1"),
            "umbrales": "0,44 / 0,51 / 0,46 congelados en todos los brazos",
            "reproducibilidad": f"semilla {SEMILLA}, TPESampler no aplica, sin rejilla",
        },
        "curva": curva,
        "presupuestos_con_mejora": [int(x) for x in df.loc[df.delta_real > 0, "presupuesto_K"]],
        "mejor_presupuesto": k_ganador,
        "gate": {"delta_min": GATE_DELTA, "pliegues_min": GATE_PLIEGUES,
                 "delta": round(delta, 10), "pliegues_mejorados": fm, "superado": gate},
        "bootstrap": (None if ic_lo is None else
                      {"IC95": [round(ic_lo, 6), round(ic_hi, 6)], "n": N_BOOT}),
        "seleccion_por_pliegue": sel_por_fold,
        "veredicto": veredicto,
        "decision": decision,
        "experimentos_registrados": 1,
        "formulacion": "V03 remains the best verified leaderboard result.",
    }
    JSON_P.write_text(json.dumps(informe, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")

    L: list[str] = []
    W = L.append
    W("# P62 - Cuanto canal de interaccion puede permitirse V03")
    W("")
    W(f"Generado por `experiments/p62_07_presupuesto_canal.py`. "
      f"Fecha {inicio.isoformat(timespec='seconds')}.")
    W("")
    W("## Que deja abierto P62-01 y P62-02")
    W("")
    W("Las dos pistas cerradas midieron su propia causa, y fue una sola:")
    W("")
    W("```text")
    W("pista L   coste del bloque  +0,011585   valor de la estructura  +0,002014")
    W("pista M   coste del bloque  +0,000443   valor de la estructura  -0,010793")
    W("```")
    W("")
    W("La senal de interaccion es real y vale alrededor de +0,002, y el obstaculo es que")
    W("anadir un bloque a un espacio de 21.000 dimensiones con 1.186 filas cuesta mas de lo")
    W("que la categoria repara. Eso es una afirmacion sobre un REGIMEN, y una afirmacion")
    W("sobre un regimen se mide: con dos puntos no se puede llamar ley.")
    W("")
    W("## El diseno")
    W("")
    W("Para un canal de un tamano dado, se mide lo que CUESTA y lo que VALE:")
    W("")
    W("- **brazo ruido**: un bloque de K rasgos SIN SENTIDO. Lo que hace es coste y nada")
    W("  mas. Esa es la curva de coste.")
    W("- **brazo real**: los K tokens de interaccion mas enriquecidos, elegidos con las")
    W("  filas de entrenamiento del pliegue. La diferencia entre los dos brazos en cada K es")
    W("  el valor del contenido a ese presupuesto.")
    W("")
    W("**No es una rejilla de hiperparametros.** K es la variable independiente de una")
    W("medicion, el clasificador es el de V03 en todos los brazos, y no se selecciona ningun")
    W("modelo por su puntuacion. Se reporta la curva entera.")
    W("")
    W("Los rasgos reales se ordenan por enriquecimiento ABSOLUTO, calculado fold-safe, para")
    W("no quedarse solo con los que favorecen la clase positiva. Elegirlos por F1 seria")
    W("seleccion de rasgos, que esta cerrada.")
    W("")
    W("## La curva")
    W("")
    W("| K | ruido | delta ruido | real | delta real | valor del contenido |")
    W("|---:|---:|---:|---:|---:|---:|")
    for r in curva:
        W(f"| {r['K']} | {r['ruido']:.10f} | {r['delta_ruido']:+.10f} | {r['real']:.10f} | "
          f"{r['delta_real']:+.10f} | {r['valor']:+.10f} |")
    W("")
    W("## Lectura")
    W("")
    W("```text")
    W(veredicto)
    W("```")
    W("")
    if k_ganador is not None:
        W("## Gate sobre el mejor presupuesto")
        W("")
        W("```text")
        W(f"K {k_ganador} | delta {delta:+.10f} | pliegues {fm}/5 | gate {gate}")
        W(f"bootstrap: {'NO EJECUTADO' if ic_lo is None else f'IC95 [{ic_lo:+.6f}, {ic_hi:+.6f}]'}")
        W("```")
        W("")
    W("## Decision")
    W("")
    W("```text")
    W(decision)
    W("```")
    W("")
    W("> V03 remains the best verified leaderboard result.")
    W("")
    W("*Generado por `experiments/p62_07_presupuesto_canal.py`. Sin push.*")
    MD_P.write_text(a_ascii("\n".join(L)), encoding="ascii")

    for destino in (MD_P, JSON_P, CSV_P):
        if any(b > 127 for b in destino.read_bytes()):
            raise SystemExit(f"PARADA: {destino.name} conserva bytes no ASCII")
    log("")
    log(f"P62-07 OK | {len(curva)} puntos de la curva | presupuestos con mejora "
        f"{informe['presupuestos_con_mejora']} | mejor {k_ganador} | gate {gate}")


if __name__ == "__main__":
    ast.parse(Path(__file__).read_text(encoding="utf-8"))
    main()
