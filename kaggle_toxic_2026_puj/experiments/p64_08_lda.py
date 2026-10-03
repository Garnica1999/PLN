"""P64 track P14: topic mixture as a compositional descriptor, and not as a better classifier.

What the audit found, and why the reserve opens
----------------------------------------------
Five latent factorisations were already measured in this repository and all five are closed:

    DL-01      TF-IDF -> TruncatedSVD(300) -> MLP, the projection was neutral
    V9-05      SVD(6) against a supervised compression, no adoption
    V25-01     a low rank second order interaction on V03, closed
    V38-01     a supervised term-by-label matrix factorised with SVD, invalid because the
               matrix has rank 3, the number of labels
    V40-01     PLSRegression(8) on V03, overfit, closed, and not reopened by V42-01 or V44-02

That is a strong prior against a sixth factorisation used the same way, namely to REPLACE the
representation and classify from the projection. Every one of those five failed at that, and
this script does not repeat it.

The one thing none of them did is the thing P63 showed matters. P63 measured that COMPOSITION
carries information: ratios of morphological features beat a random control by 0,1055 in
isolation, and added 0,00835 of margin to V03, and still did not improve V03. That is a
result about the SHAPE of the document rather than about its content, and topic mixture is the
thematic counterpart of that ratio block: it says what a document is made of, not which words
it contains.

So this is a descriptor and not a projection. LDA is fitted on the training fold of each
outer fold, the document-topic mixture is used as a small dense block, and the classifier is
V03's own LogisticRegression with the block appended. The question is narrow:

    does the thematic composition of a document add anything that its words and their
    confidence do not already carry

If it does not, the sixth factorisation closes the family for good, and the closing is much
stronger than a single negative because it sits on top of five.

One configuration, declared before execution
--------------------------------------------
Eight topics, batch variational, seed 42. No search over the number of topics. The brief says
not to, and a search over topic counts would be indistinguishable from the grids this project
has refused everywhere else.

Usage
-----
    /home/carlos/venv/bin/python experiments/p64_08_lda.py
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
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
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
MD_LDA = REPORTS / "p64_topic_mixture.md"
JSON_LDA = REPORTS / "p64_topic_mixture.json"
CSV_LDA = REPORTS / "p64_topic_mixture.csv"

V03_UMBRALES = np.array([0.44, 0.51, 0.46])
PARAMS_WORD = {"analyzer": "word", "ngram_range": (1, 2), "min_df": 2, "sublinear_tf": True,
               "lowercase": True}
PARAMS_CHAR = {"analyzer": "char_wb", "ngram_range": (3, 5), "min_df": 2,
               "sublinear_tf": True, "lowercase": True}
LR = {"C": 2.0, "class_weight": "balanced", "max_iter": 1000, "solver": "liblinear",
      "random_state": 42}
# Una sola configuracion, escrita antes de ejecutar. No hay busqueda del numero de temas.
LDA_PARAMS = {"n_components": 8, "learning_method": "batch", "random_state": 42,
              "max_iter": 20, "doc_topic_prior": 0.1, "topic_word_prior": 0.01}
COUNT_PARAMS = {"min_df": 3, "max_features": 3000, "lowercase": True}
GATE_DELTA = 0.0015
GATE_PLIEGUES = 4
SEMILLA = 42
BINS: tuple[tuple[str, int, int], ...] = (
    ("CORTA", 0, 6),
    ("MEDIA", 7, 15),
    ("LARGA", 16, 30),
    ("MUY_LARGA", 31, 10**9),
)
RASGOS = ("dominante", "entropia", "concentracion", "segundo", "dispersion")


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[P64-08] {mensaje}", flush=True)


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


def perfil_mezcla(m: np.ndarray) -> list[float]:
    """Perfil acotado de la mezcla de temas de un documento.

    Args:
        m: vector de proporciones de temas, que suma uno.

    Returns:
        Proporcion dominante, entropia normalizada, concentracion, proporcion del segundo
        tema y dispersion. Todos acotados.
    """
    q = np.asarray(m, dtype=float)
    total = max(q.sum(), 1e-12)
    q = q / total
    ent = float(-(q * np.log2(np.clip(q, 1e-12, None))).sum()) / np.log2(len(q))
    orden = np.sort(q)[::-1]
    return [float(orden[0]), ent, float(orden[0]), float(orden[1]), float(q.std())]


def main() -> None:
    """Ajusta LDA por pliegue y evalua la mezcla tematica como bloque denso.

    Returns:
        None.
    """
    inicio = datetime.now(timezone.utc)
    log("P64-08 mezcla de temas como descriptor de composicion")
    log(f"   configuracion unica: {json.dumps(LDA_PARAMS, sort_keys=True)}")
    log("   LDA se ajusta SOLO en el entrenamiento de cada pliegue")

    hash_real = hashlib.sha256(V03_SUB.read_bytes()).hexdigest()
    if hash_real != V03_SHA:
        raise SystemExit("PARADA: la submission de V03 no coincide con el sha256 congelado")
    v = np.load(OOF, allow_pickle=True)
    ids, y, pliegues = v["ids"], v["y_true"].astype(int), v["pliegues"]
    pred_v03, proba_v03 = v["pred"].astype(int), v["proba"]
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
    error = pred_v03 != y
    log(f"   OOF canonico {ccv_ref!r} | {n} filas")

    nombres = [f"{r}" for r in RASGOS]
    bloque = np.zeros((n, len(nombres)), dtype=float)
    vocab: dict[int, int] = {}
    tiempos: dict[int, float] = {}
    time.time()
    mezcla_guardada = np.zeros((n, LDA_PARAMS["n_components"]), dtype=float)
    for f in range(1, 6):
        ir, iva = np.where(pliegues != f)[0], np.where(pliegues == f)[0]
        cv = CountVectorizer(**COUNT_PARAMS)
        xc = cv.fit_transform([textos[i] for i in ir])
        vocab[f] = int(xc.shape[1])
        lda = LatentDirichletAllocation(**LDA_PARAMS)
        tf = time.time()
        lda.fit_transform(xc)
        xd_v = lda.transform(cv.transform([textos[i] for i in iva]))
        tiempos[f] = time.time() - tf
        mezcla_guardada[iva] = xd_v
        for k in iva:
            bloque[k] = perfil_mezcla(xd_v[int(np.where(iva == k)[0][0])])
        log(f"      pliegue {f} | vocabulario {vocab[f]} | LDA {tiempos[f]:.1f} s")
    log(f"   tiempo total de LDA: {sum(tiempos.values()):.1f} s | "
        f"vocabulario medio {np.mean(list(vocab.values())):.0f}")

    # ---------- descriptores por estrato ----------
    error_alto = error & (np.abs(proba_v03 - V03_UMBRALES[None, :]) >= 0.10)
    estratos = [("todas", np.ones(n, dtype=bool)),
                ("V03 correcto", ~error.any(axis=1)),
                ("V03 incorrecto", error.any(axis=1)),
                ("error de alta confianza", error_alto.any(axis=1))]
    for nombre, lo, hi in BINS:
        estratos.append((nombre, (n_tok >= lo) & (n_tok <= hi)))
    log("")
    log("   EL PERFIL DE MEZCLA, por estrato")
    log(f"      {'estrato':>26}{'n':>7}" + "".join(f"{r[:14]:>16}" for r in RASGOS))
    filas_perfil = []
    for nombre, m in estratos:
        if m.sum() < 5:
            continue
        vals = bloque[m].mean(axis=0)
        filas_perfil.append({"estrato": nombre, "n": int(m.sum())}
                            | {r: round(float(x), 6) for r, x in zip(RASGOS, vals)})
        log(f"      {nombre:>26}{int(m.sum()):>7}"
            + "".join(f"{x:>16.4f}" for x in vals))

    # ---------- los brazos ----------
    proba: dict[str, np.ndarray] = {b: np.zeros((n, 3)) for b in
                                   ("V03_lda", "V03_lda_completo", "V03_random", "V03_margen")}
    for f in range(1, 6):
        ir, iva = np.where(pliegues != f)[0], np.where(pliegues == f)[0]
        pw = TfidfVectorizer(**PARAMS_WORD)
        pc = TfidfVectorizer(**PARAMS_CHAR)
        xbase = hstack([pw.fit_transform([textos[i] for i in ir]),
                        pc.fit_transform([textos[i] for i in ir])]).tocsr()
        xbase_v = hstack([pw.transform([textos[i] for i in iva]),
                          pc.transform([textos[i] for i in iva])]).tocsr()
        objetivo = float(np.sqrt(np.mean(xbase.multiply(xbase).sum(axis=1))))

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

        rng = np.random.default_rng(SEMILLA + f)
        matrices = {
            "V03_lda": bloque,
            "V03_lda_completo": mezcla_guardada,
            "V03_random": rng.standard_normal(bloque.shape),
            "V03_margen": margen_v03,
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
        log(f"      brazos del pliegue {f} listos")

    folds_v = [float(f1_score(y[pliegues == g], pred_v03[pliegues == g], average="macro"))
               for g in range(1, 6)]
    orden = ("V03_lda", "V03_lda_completo", "V03_random", "V03_margen")
    detalle: dict[str, Any] = {}
    filas: list[dict[str, Any]] = []
    log("")
    log("   LOS CUATRO BRAZOS")
    log(f"      {'brazo':18}{'corrected_cv':>18}{'delta vs V03':>16}{'pliegues':>10}{'neto':>8}")
    for brazo in orden:
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
        detalle[brazo] = {"corrected_cv": ccv, "folds": folds, "por_bin": por_bin,
                          "rescate": int(np.count_nonzero(ac & (pred_v03 != y))),
                          "damage": int(np.count_nonzero((~ac) & (pred_v03 == y)))}
        pl = sum(1 for a, b in zip(folds, folds_v) if a > b)
        filas.append({"brazo": brazo, "corrected_cv": round(ccv, 10),
                      "delta_vs_V03": round(ccv - ccv_ref, 10), "pliegues_mejorados": pl,
                      "rescate": detalle[brazo]["rescate"],
                      "damage": detalle[brazo]["damage"]})
        log(f"      {brazo:18}{ccv:>18.10f}{ccv - ccv_ref:>+16.10f}{pl:>7d}/5"
            f"{detalle[brazo]['rescate'] - detalle[brazo]['damage']:>8d}")

    brecha_cap = detalle["V03_lda"]["corrected_cv"] - detalle["V03_random"]["corrected_cv"]
    brecha_marg = detalle["V03_lda"]["corrected_cv"] - detalle["V03_margen"]["corrected_cv"]
    brecha_full = (detalle["V03_lda_completo"]["corrected_cv"]
                   - detalle["V03_random"]["corrected_cv"])
    d_glob = detalle["V03_lda"]["corrected_cv"] - ccv_ref
    pl = next(f["pliegues_mejorados"] for f in filas if f["brazo"] == "V03_lda")
    log("")
    log("   LAS BRECHAS")
    log(f"      la mezcla completa contra el control de anchura: {brecha_full:+.10f}")
    log(f"      el perfil de 5 rasgos contra el control de anchura: {brecha_cap:+.10f}")
    log(f"      el perfil de 5 rasgos contra el margen de V03:   {brecha_marg:+.10f}")

    if d_glob >= GATE_DELTA and pl >= GATE_PLIEGUES and brecha_cap > 0 and brecha_marg > 0:
        lectura, decision = "PROMISING", "PROMISING"
        veredicto = (f"la composicion tematica aporta {d_glob:+.10f} con {pl} de 5 y supera "
                     f"los dos controles")
    elif brecha_cap > 0 and brecha_marg > 0:
        lectura, decision = "INFORMACION_SI_INTEGRACION_NO", "COMPLEMENTARY"
        veredicto = (
            f"la composicion tematica lleva informacion que no es capacidad ni confianza: "
            f"supera al control de anchura en {brecha_cap:+.10f} y al margen en "
            f"{brecha_marg:+.10f}. Y a la vez el global es {d_glob:+.10f} con {pl} de 5, luego "
            f"no entra. Es el MISMO patron que la negacion en P64-07: la senal existe y la "
            f"integracion la descarta. Con esto son SEIS factorizaciones latentes las que "
            f"aportan senal y ninguna mejora, y eso convierte el resultado de una costumbre en "
            f"una caracterizacion de la LIMITACION")
    else:
        lectura, decision = "CERRADA", "CLOSED"
        veredicto = (
            f"la sexta factoriazacion latente no aporta: {brecha_cap:+.10f} contra el control "
            f"de anchura y {brecha_marg:+.10f} contra el margen. Y con esto quedan SEIS "
            f"factorizaciones distintas medidas y cerradas, que es la mayor evidencia que "
            f"puede tener una familia cerrada en este repositorio")
    log("")
    log(f"   LECTURA: {lectura}")
    log(f"   VEREDICTO: {veredicto}")
    log(f"   DECISION: {decision}")

    with CSV_LDA.open("w", encoding="ascii", newline="") as fh:
        w = _csv.writer(fh, lineterminator="\n")
        w.writerow(["tipo", "clave"] + list(RASGOS) + ["n"])
        for f in filas_perfil:
            w.writerow(["perfil", f["estrato"]] + [f[r] for r in RASGOS] + [f["n"]])
        for f in filas:
            w.writerow(["brazo", f["brazo"], f["corrected_cv"], f["delta_vs_V03"],
                        f["pliegues_mejorados"], "", "", "", ""])
        w.writerow(["brecha", "perfil_contra_anchura", round(brecha_cap, 10)])
        w.writerow(["brecha", "perfil_contra_margen", round(brecha_marg, 10)])
        w.writerow(["brecha", "mezcla_completa_contra_anchura", round(brecha_full, 10)])
        w.writerow([])
        w.writerow(["pliegue", "vocabulario", "segundos_de_LDA"])
        for f in range(1, 6):
            w.writerow([f, vocab[f], round(tiempos[f], 3)])
        w.writerow([])
        w.writerow(["bin", "n", "F1_de_V03", "F1_lda", "F1_mezcla_completa", "F1_margen"])
        for nombre, _lo, _hi in BINS:
            w.writerow([nombre, detalle["V03_lda"]["por_bin"][nombre]["n"],
                        detalle["V03_lda"]["por_bin"][nombre]["F1_de_V03"]]
                       + [detalle[b]["por_bin"][nombre]["F1"]
                          for b in ("V03_lda", "V03_lda_completo", "V03_margen")])

    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                            text=True, cwd=PROJECT_ROOT, check=False).stdout.strip()
    informe = {
        "familia": "P64_TOPIC_MIXTURE",
        "naturaleza": ("EXPERIMENTO. 2 candidatos y 2 controles, 0 submissions, 0 Kaggle, "
                       "0 holdout, 0 test"),
        "fecha": inicio.isoformat(timespec="seconds"), "experiment_id": "P64-08",
        "track": "P14-mezcla-de-temas", "script": "experiments/p64_08_lda.py",
        "pregunta": ("la composicion tematica de un documento aporta algo que sus palabras y "
                     "su confianza no llevan ya"),
        "auditoria_previa": {
            "DL-01": "TF-IDF -> TruncatedSVD(300) -> MLP, proyeccion neutral",
            "V9-05": "SVD(6) frente a compresion supervisada, sin adopcion",
            "V25-01": "interaccion de segundo orden de bajo rango sobre V03, cerrada",
            "V38-01": "matriz termino por etiqueta factorizada con SVD, invalida por rango 3",
            "V40-01": "PLSRegression(8) sobre V03, sobreajusta, cerrada y no reabierta por "
                      "V42-01 ni V44-02",
            "por_que_esta_se_abre_aun": (
                "las cinco anteriores usaron la proyeccion para REEMPLAZAR la representacion y "
                "clasificar desde ella. Esta usa la mezcla como DESCRIPTOR de composicion, que "
                "es lo que P63 mostro que tiene senal en su analogo morfologico: los ratios de "
                "composicion batieron a un control aleatorio en 0,1055 aislados y anadieron "
                "0,00835 de margen, y aun asi no mejoraron V03"),
        },
        "configuracion": {"lda": LDA_PARAMS, "count_vectorizer": COUNT_PARAMS,
                          "busqueda_de_numero_de_temas": "ninguna, por regla del encargo"},
        "vocabulario_por_pliegue": vocab,
        "segundos_de_ajuste_por_pliegue": {k: round(v, 3) for k, v in tiempos.items()},
        "rasgos": list(RASGOS),
        "perfil_por_estrato": filas_perfil,
        "resultados": detalle,
        "brechas": {"perfil_contra_anchura": round(brecha_cap, 10),
                    "perfil_contra_margen": round(brecha_marg, 10),
                    "mezcla_completa_contra_anchura": round(brecha_full, 10)},
        "lectura": lectura, "veredicto": veredicto, "decision": decision,
        "experimentos_registrados": 1,
        "reproducibilidad": {"nivel": "R2", "git_commit": commit, "seed": SEMILLA,
                             "hash_v03": hash_real, "corrected_cv": V03_CCV},
        "formulacion": "V03 remains the best verified leaderboard result.",
    }
    JSON_LDA.write_text(json.dumps(informe, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")

    L: list[str] = []
    W = L.append
    W("# P64 - Mezcla de temas como descriptor de composicion")
    W("")
    W(f"Generado por `experiments/p64_08_lda.py`. Fecha "
      f"{inicio.isoformat(timespec='seconds')}.")
    W("")
    W("## Lo que encontro la auditoria, y por que se abre la reserva")
    W("")
    W("Hay **cinco** factorizaciones latentes ya medidas en este repositorio, y las cinco")
    W("estan cerradas:")
    W("")
    W("| experimento | que era | por que cerro |")
    W("|---|---|---|")
    W("| DL-01 | TF-IDF -> TruncatedSVD(300) -> MLP | proyeccion neutral |")
    W("| V9-05 | SVD(6) frente a compresion supervisada | sin adopcion |")
    W("| V25-01 | interaccion de segundo orden de bajo rango sobre V03 | cerrada |")
    W("| V38-01 | matriz termino por etiqueta factorizada con SVD | invalida por rango 3 |")
    W("| V40-01 | PLSRegression(8) sobre V03 | sobreajusta; no reabierta por V42 ni V44 |")
    W("")
    W("Ese es un priori fuerte en contra de una sexta factoriazacion usada IGUAL, es decir")
    W("para REEMPLAZAR la representacion y clasificar desde la proyeccion. Este script no")
    W("repite eso.")
    W("")
    W("**Lo que ninguna de las cinco hizo es justo lo que P63 demostro que importa.** P63")
    W("midio que la COMPOSICION lleva informacion: los ratios de rasgos morfologicos")
    W("batieron a un control aleatorio en 0,1055 aislados y anadieron 0,00835 de margen a V03,")
    W("y aun asi no mejoraron V03. Eso es un resultado sobre la FORMA del documento y no sobre")
    W("su contenido, y la mezcla de temas es la contraparte tematica de ese bloque de ratios.")
    W("")
    W("Asi que esto es un descriptor y no una proyeccion. LDA se ajusta en el entrenamiento de")
    W("cada pliegue, la mezcla documento-tema entra como un bloque denso pequeno, y el")
    W("clasificador es el LogisticRegression de V03 con ese bloque anadido.")
    W("")
    W("## Una sola configuracion, declarada antes")
    W("")
    W("```text")
    W(f"lda          {json.dumps(LDA_PARAMS, sort_keys=True)}")
    W(f"count        {json.dumps(COUNT_PARAMS, sort_keys=True)}")
    W("busqueda del numero de temas: NINGUNA, por regla del encargo")
    W("```")
    W("")
    W("## Resultados")
    W("")
    W("| brazo | corrected_cv | delta vs V03 | pliegues | neto |")
    W("|---|---:|---:|---:|---:|")
    for f in filas:
        W(f"| {f['brazo']} | {f['corrected_cv']:.10f} | {f['delta_vs_V03']:+.10f} | "
          f"{f['pliegues_mejorados']}/5 | {f['rescate'] - f['damage']:+d} |")
    W("")
    W("### Las brechas")
    W("")
    W("```text")
    W(f"la mezcla completa contra el control de anchura:  {brecha_full:+.10f}")
    W(f"el perfil de 5 rasgos contra el control de anchura: {brecha_cap:+.10f}")
    W(f"el perfil de 5 rasgos contra el margen de V03:    {brecha_marg:+.10f}")
    W("```")
    W("")
    W("### Coste, porque una sexta factoriazacion tiene que justificarse")
    W("")
    W("| pliegue | vocabulario | segundos de LDA |")
    W("|---:|---:|---:|")
    for f in range(1, 6):
        W(f"| {f} | {vocab[f]} | {tiempos[f]:.1f} |")
    W("")
    W("### El perfil de mezcla por estrato")
    W("")
    W("| estrato | n | " + " | ".join(f"`{r}`" for r in RASGOS) + " |")
    W("|---|---:|" + "---:|" * len(RASGOS))
    for f in filas_perfil:
        W(f"| {f['estrato']} | {f['n']} | " + " | ".join(f"{f[r]:.4f}" for r in RASGOS) + " |")
    W("")
    W("### Por estrato de longitud")
    W("")
    W("| bin | n | V03 | perfil | mezcla completa | margen |")
    W("|---|---:|---:|---:|---:|---:|")
    for nombre, _lo, _hi in BINS:
        W(f"| {nombre} | {detalle['V03_lda']['por_bin'][nombre]['n']} | "
          f"{detalle['V03_lda']['por_bin'][nombre]['F1_de_V03']:.4f} | "
          + " | ".join(f"{detalle[b]['por_bin'][nombre]['F1']:.4f}"
                      for b in ("V03_lda", "V03_lda_completo", "V03_margen")) + " |")
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
    W("*Generado por `experiments/p64_08_lda.py`. Sin push.*")
    MD_LDA.write_text(a_ascii("\n".join(L)), encoding="ascii")

    for destino in (MD_LDA, JSON_LDA, CSV_LDA):
        if any(b > 127 for b in destino.read_bytes()):
            raise SystemExit(f"PARADA: {destino.name} conserva bytes no ASCII")
    log("")
    log(f"P64-08 OK | vocabulario medio {np.mean(list(vocab.values())):.0f} | brecha anchura "
        f"{brecha_cap:+.6f} | brecha margen {brecha_marg:+.6f} | decision {decision}")


if __name__ == "__main__":
    ast.parse(Path(__file__).read_text(encoding="utf-8"))
    main()
