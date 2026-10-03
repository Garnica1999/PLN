"""P62 track M: lexical dependency tokens, added to V03's own feature space.

How this differs from track L, and why it is still worth an experiment
-----------------------------------------------------------------------
Track L gave the model a word together with its own grammatical category, and it lost
0,009572. The random control showed the loss was mostly dilution: adding a block of
about 1200 features to a 21000-dimensional space costs 0,011585 even when the block
carries no meaning at all, and the linguistic content repaid only 17,4 % of that.

Track M asks a different question, and the difference is not cosmetic. A ``word__POS``
token says a word is a noun. A ``head__REL__dependent`` token says that in THIS tweet
this noun is the direct object of THAT verb. The second is the thing that distinguishes
an insult aimed at someone from a bare adjective, which is exactly the composition the
phase is about, and no unigram or bigram of the raw text expresses it.

The relation set is declared, not harvested
--------------------------------------------
Paragraph 30 forbids turning every possible pair into a column. The set below is the
Universal Dependencies content relations, fixed in advance, and a pair is emitted only
when both the head and the dependent have an alphabetic lemma. Choosing the relations by
how often they discriminate would be choosing them after seeing the answer.

The control is the one track L taught this phase to build
---------------------------------------------------------
Not a relabelling. Relabelling the vocabulary produces EXACTLY the same predictions,
because permuting columns is a symmetry of an L2-regularised linear model, so that
control is a provable no-op and cannot discriminate. The control here is an independent
block of the same size, found by a fold-local search over the pool size, which is a
search over the control's dimension and not over any hyperparameter of the model.

Usage
-----
    /home/carlos/venv/bin/python experiments/p62_04_pista_m.py
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
    ETIQUETAS,
    V03_CCV,
    V03_SHA,
    V03_SUB,
)

REPORTS = PROJECT_ROOT / "reports"
OOF = PROJECT_ROOT / "features" / "oof_v03_final.npz"
TRAIN_RAW = PROJECT_ROOT / "data" / "raw" / "toxic_train.csv"
PARTICION = PROJECT_ROOT / "data" / "processed" / "holdout_v2_ids.csv"
ANOTACION = PROJECT_ROOT / "features" / "p62_anotacion_spacy.json.gz"
MD_M = REPORTS / "p62_pista_m_dependencias.md"
JSON_M = REPORTS / "p62_pista_m_dependencias.json"
CELDA_NPZ = PROJECT_ROOT / "features" / "p62_m_dep_cells.npz"

V03_UMBRALES = np.array([0.44, 0.51, 0.46])
PARAMS_WORD = {"analyzer": "word", "ngram_range": (1, 2), "min_df": 2, "sublinear_tf": True,
               "lowercase": True}
PARAMS_CHAR = {"analyzer": "char_wb", "ngram_range": (3, 5), "min_df": 2,
               "sublinear_tf": True, "lowercase": True}
PARAMS_CANAL = {"analyzer": "word", "ngram_range": (1, 1), "min_df": 2, "sublinear_tf": True,
                "lowercase": False, "token_pattern": r"[^ ]+"}
LR = {"C": 2.0, "class_weight": "balanced", "max_iter": 1000, "solver": "liblinear",
      "random_state": 42}

# Relaciones de contenido de Universal Dependencies, DECLARADAS ANTES de ejecutar. No se
# eligen por como discriminan, porque eso seria elegirlas despues de ver la respuesta.
RELACIONES = ("obj", "amod", "obl", "obl:arg", "nsubj", "iobj", "nmod", "acomp", "attr",
              "advmod", "compound", "appos", "parataxis", "ccomp", "xcomp", "acl:relcl")
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
    print(f"[P62-04] {mensaje}", flush=True)


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


def tokens_dep(documentos: list[dict[str, Any]], filas: np.ndarray,
               rng: np.random.Generator | None, tam_pool: int = 0) -> list[str]:
    """Construye los tokens del triple lexico de dependencia.

    Args:
        documentos: anotaciones.
        filas: indices de las filas del conjunto.
        rng: si se da, se sustituyen los dos lemas por identificadores aleatorios, que es
            el control de dimensionalidad.
        tam_pool: numero de entradas del pool aleatorio.

    Returns:
        Un token por fila, separados por espacio.
    """
    salida = []
    for i in filas:
        toks = documentos[i]["tok"]
        partes = []
        for t in toks:
            if t["d"] not in RELACIONES or not t["al"] or not t["l"] or not t["hl"]:
                continue
            # Un lema puede ser MULTIPALABRA, y spaCy devuelve alguno como "quedar el".
            # Al meterlo en el token queda "quedar el__obl__cepeda", y el tokenizador de
            # espacios parte eso en dos trozos, uno de los cuales no es un token valido y
            # ni siquiera tiene la forma esperada. Se exigen lemas de una sola palabra, que
            # es lo que hace falta para que el token sea un token.
            if " " in t["l"] or " " in t["hl"]:
                continue
            if t["l"] == t["hl"] and rng is None:
                continue
            if rng is None:
                partes.append(f"{t['hl']}__{t['d']}__{t['l']}")
            else:
                partes.append(f"z{int(rng.integers(0, tam_pool))}__{t['d']}__"
                              f"w{int(rng.integers(0, tam_pool))}")
        salida.append(" ".join(partes))
    return salida


def main() -> None:
    """Ejecuta la pista M con su control de dimensionalidad.

    Returns:
        None.
    """
    inicio = datetime.now(timezone.utc)
    log("P62-04 pista M, triples lexicos de dependencia anadidos al espacio de V03")
    log(f"   {len(RELACIONES)} relaciones de contenido DECLARADAS antes de ejecutar")
    log("   clasificador IDENTICO a V03: LR C=2,0 balanced liblinear, mismos pliegues, "
        "mismos umbrales")

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
        raise SystemExit("PARADA: la anotacion de spaCy no corresponde a las filas del OOF")
    documentos = anot["documentos"]
    log("")
    log(f"   OOF canonico {ccv_ref!r} | {n} filas | {len(set(grupos))} grupos intactos")
    log("   relaciones: " + ", ".join(RELACIONES))

    proba = {k: np.zeros((n, 3)) for k in ("control", "pista_M", "control_aleatorio")}
    dims: dict[str, list[int]] = {k: [] for k in proba}
    rel_usadas: dict[str, int] = {}
    t0 = time.time()
    for f in range(1, 6):
        ir, iva = np.where(pliegues != f)[0], np.where(pliegues == f)[0]
        pw = TfidfVectorizer(**PARAMS_WORD)
        pc = TfidfVectorizer(**PARAMS_CHAR)
        xw = pw.fit_transform([textos[i] for i in ir])
        xc = pc.fit_transform([textos[i] for i in ir])
        xbase = hstack([xw, xc]).tocsr()
        xbase_v = hstack([pw.transform([textos[i] for i in iva]),
                           pc.transform([textos[i] for i in iva])]).tocsr()
        tok = tokens_dep(documentos, ir, None)
        tok_v = tokens_dep(documentos, iva, None)
        pd_m = TfidfVectorizer(**PARAMS_CANAL)
        xdep = pd_m.fit_transform(tok)
        xdep_v = pd_m.transform(tok_v)
        objetivo = float(np.sqrt(np.mean(xbase.multiply(xbase).sum(axis=1))))

        def escalar(m: csr_matrix, ref: float) -> csr_matrix:
            """Iguala la norma media de fila a la de referencia.

            Args:
                m: matriz dispersa.
                ref: norma objetivo.

            Returns:
                La matriz escalada.
            """
            media = float(np.sqrt(np.asarray(m.multiply(m).sum(axis=1)).ravel()).mean())
            return m if media <= 0 else m.multiply(ref / media)

        x_m = escalar(hstack([xbase, xdep]).tocsr(), objetivo)
        x_m_v = escalar(hstack([xbase_v, xdep_v]).tocsr(), objetivo)

        objetivo_dim = xdep.shape[1]
        mejor = None
        for pool in (objetivo_dim, int(objetivo_dim * 0.6), int(objetivo_dim * 0.4),
                     int(objetivo_dim * 1.6), int(objetivo_dim * 2.5), 200, 50, 12):
            rng_c = np.random.default_rng(SEMILLA + f)
            pc_c = TfidfVectorizer(**PARAMS_CANAL)
            xa = pc_c.fit_transform(tokens_dep(documentos, ir, rng_c, max(pool, 2)))
            if xa.shape[1] == 0:
                continue
            if mejor is None or abs(xa.shape[1] - objetivo_dim) < abs(mejor[0] - objetivo_dim):
                rng_c = np.random.default_rng(SEMILLA + f)
                mejor = (xa.shape[1], pool, (pc_c, tokens_dep(documentos, ir, rng_c,
                                                              max(pool, 2)),
                                             tokens_dep(documentos, iva, rng_c,
                                                       max(pool, 2))))
            if xa.shape[1] == objetivo_dim:
                break
        assert mejor is not None
        dim_c, pool_c, (pc_c, tok_ca, tok_cv) = mejor
        xrand = escalar(hstack([xbase, pc_c.fit_transform(tok_ca)]).tocsr(), objetivo)
        xrand_v = escalar(hstack([xbase_v, pc_c.transform(tok_cv)]).tocsr(), objetivo)

        for clave, xent, xval in (("control", xbase, xbase_v), ("pista_M", x_m, x_m_v),
                                  ("control_aleatorio", xrand, xrand_v)):
            for j in range(3):
                mo = LogisticRegression(**LR)
                mo.fit(xent, y[ir, j])
                proba[clave][iva, j] = mo.predict_proba(xval)[:, 1]
            dims[clave].append(int(xent.shape[1]))
        for term in pd_m.vocabulary_:
            rel = term.split("__")[1]
            rel_usadas[rel] = rel_usadas.get(rel, 0) + 1
        log(f"      pliegue {f}: base {xbase.shape[1]} | +M {x_m.shape[1]} "
            f"(canal {xdep.shape[1]}) | +aleatorio {xrand.shape[1]} "
            f"(pool {pool_c} -> {dim_c}) | {time.time() - t0:.0f} s")

    def metricas(clave: str) -> dict[str, Any]:
        """Metricas de un brazo.

        Args:
            clave: nombre del brazo.

        Returns:
            Diccionario con las metricas.
        """
        p = proba[clave]
        pr = (p >= V03_UMBRALES[None, :]).astype(int)
        ac = pr == y
        por_et = {}
        for j, et in enumerate(ETIQUETAS):
            a = ac[:, j]
            por_et[et] = {
                "F1": round(float(f1_score(y[:, j], pr[:, j])), 6),
                "rescate": int(np.count_nonzero(a & (pred_v03[:, j] != y[:, j]))),
                "damage": int(np.count_nonzero((~a) & (pred_v03[:, j] == y[:, j]))),
                "errores_de_alta_confianza": int(np.count_nonzero(
                    ~a & (np.abs(p[:, j] - V03_UMBRALES[j]) >= DELTA_HC))),
                "celdas_modificadas": int(np.count_nonzero(pr[:, j] != pred_v03[:, j])),
            }
        return {
            "corrected_cv": float(f1_score(y, pr, average="macro")),
            "folds": [float(f1_score(y[pliegues == g], pr[pliegues == g], average="macro"))
                      for g in range(1, 6)],
            "rescate": int(np.count_nonzero(ac & (pred_v03 != y))),
            "damage": int(np.count_nonzero((~ac) & (pred_v03 == y))),
            "cambian": int(np.count_nonzero(pr != pred_v03)),
            "por_etiqueta": por_et,
        }

    mc, mm, mr = metricas("control"), metricas("pista_M"), metricas("control_aleatorio")
    log("")
    log("   LOS TRES BRAZOS")
    log(f"      control, V03 sin canal     {mc['corrected_cv']:.10f}  "
        f"(coincide con V03: {mc['corrected_cv'] == V03_CCV})")
    log(f"      pista M, + triple lexico   {mm['corrected_cv']:.10f}  "
        f"delta {mm['corrected_cv'] - ccv_ref:+.10f}")
    log(f"      control aleatorio          {mr['corrected_cv']:.10f}  "
        f"delta {mr['corrected_cv'] - ccv_ref:+.10f}")
    if mc["corrected_cv"] != V03_CCV:
        raise SystemExit(f"PARADA: el control no reproduce V03, da {mc['corrected_cv']!r}")

    aporte_real = mm["corrected_cv"] - ccv_ref
    aporte_aleatorio = mr["corrected_cv"] - ccv_ref
    valor_estructura = aporte_real - aporte_aleatorio
    coste = -aporte_aleatorio
    log("")
    log("   LA LECTURA EN TRES PARTES, la misma que la pista L deja establecida")
    log(f"      (1) coste de anadir el bloque      {coste:+.10f}")
    log(f"      (2) valor de la estructura lexica  {valor_estructura:+.10f}")
    log(f"      (3) saldo                          {aporte_real:+.10f}")
    if coste > 0 and valor_estructura > 0:
        log(f"      repaga el {valor_estructura / coste * 100:.1f} por ciento del coste")
    log("")
    log("      relaciones que sobrevivieron a min_df=2, con el numero de rasgos:")
    for rel, c in sorted(rel_usadas.items(), key=lambda kv: -kv[1])[:12]:
        log(f"        {rel:12} {c:6d}")

    folds_v03 = [float(f1_score(y[pliegues == g], pred_v03[pliegues == g], average="macro"))
                 for g in range(1, 6)]
    folds_mej = sum(1 for a, b in zip(mm["folds"], folds_v03) if a > b)
    delta = aporte_real
    gate = bool(delta >= GATE_DELTA and folds_mej >= GATE_PLIEGUES)
    log("")
    log("   GATE")
    log(f"      delta {delta:+.10f} >= {GATE_DELTA}: {delta >= GATE_DELTA} | pliegues "
        f"{folds_mej}/5 >= {GATE_PLIEGUES}: {folds_mej >= GATE_PLIEGUES} | superado: {gate}")
    log(f"      rescate {mm['rescate']} | dano {mm['damage']} | neto "
        f"{mm['rescate'] - mm['damage']:+d} | cambian {mm['cambian']}")
    log("")
    log("   ALTA CONFIANZA, que el parrafo 57 exige separar de la baja")
    for j, et in enumerate(ETIQUETAS):
        p = proba["pista_M"]
        pr = (p >= V03_UMBRALES[None, :]).astype(int)
        d_v = np.abs(proba_v03[:, j] - V03_UMBRALES[j])
        d_c = np.abs(p[:, j] - V03_UMBRALES[j])
        e_v, e_c = pred_v03[:, j] != y[:, j], pr[:, j] != y[:, j]
        corr = int(np.count_nonzero(e_v & ~e_c & (d_v >= DELTA_HC)))
        dano = int(np.count_nonzero(~e_v & e_c & (d_c >= DELTA_HC)))
        log(f"      {et}: errores V03 {int(e_v.sum()):4d} -> {int(e_c.sum()):4d} | alta "
            f"confianza corregidos {corr:4d} | introducidos {dano:4d} | neto {corr - dano:+4d}")
    ic_lo = ic_hi = None
    if gate:
        rng = np.random.default_rng(SEMILLA)
        pr = (proba["pista_M"] >= V03_UMBRALES[None, :]).astype(int)
        dc = ((pr == y).astype(int) - (pred_v03 == y).astype(int)).sum(axis=1).astype(float)
        gu, gc = np.unique(grupos, return_inverse=True)
        dif_g = np.bincount(gc, weights=dc, minlength=len(gu))
        ms = dif_g[rng.integers(0, len(gu), size=(N_BOOT, len(gu)))].sum(axis=1) / len(gu)
        ic_lo, ic_hi = float(np.percentile(ms, 2.5)), float(np.percentile(ms, 97.5))
        log(f"      bootstrap: IC95 [{ic_lo:+.6f}, {ic_hi:+.6f}]")

    np.savez_compressed(CELDA_NPZ, ids=ids, y_true=y, pliegues=pliegues, pred_v03=pred_v03,
                        proba_m=proba["pista_M"], proba_aleatorio=proba["control_aleatorio"],
                        proba_v03=proba_v03)

    decision = ("PROMISING. Supera el gate; exige segunda particion" if gate else
                ("MARGINAL. Mejora pero no alcanza el gate de +0,003" if delta > 0 else
                 "CLOSED. El triple lexico de dependencia no aporta sobre V03"))
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                            text=True, cwd=PROJECT_ROOT, check=False).stdout.strip()
    informe = {
        "familia": "P62_TRACK_M",
        "naturaleza": ("EXPERIMENTO. 1 candidato, 1 control de dimensionalidad, 0 "
                       "submissions, 0 Kaggle, 0 holdout, 0 test. V03 no se modifica"),
        "fecha": inicio.isoformat(timespec="seconds"),
        "experiment_id": "P62-02", "track": "M-dependencias-lexicas",
        "script": "experiments/p62_04_pista_m.py",
        "hipotesis": ("no basta saber que dos palabras aparecen juntas; importa su relacion "
                      "sintactica concreta, y el triple lexico cabeza-relacion-dependiente "
                      "expresa algo que el TF-IDF de V03 no dice"),
        "gap_check": {"veredicto": "genuinely_new_en_forma_lexica",
                      "comprobacion": ("E2-04 emite POS_dependiente_POS_cabeza, "
                                       "DEP_cabeza_POS y DEP_cabeza, es decir nunca la palabra "
                                       "dependiente; el triple lexico no existe"),
                      "cercanos": ["E2-04-DEPS -0,2258", "E2-04-DEPS-TXT -0,0119 con 1/5",
                                   "31 columnas densas dep_*"]},
        "protocolo": {
            "relaciones": list(RELACIONES),
            "relaciones_como_se_eligieron": ("declaradas antes de ejecutar, no por como "
                                             "discriminan; elegirselas despues seria "
                                             "elegirlas viendo la respuesta"),
            "restricciones": "se emite el par solo si ambos lemas son alfabeticos y distintos",
            "clasificador": "IDENTICO a V03: LogisticRegression C=2,0 balanced liblinear",
            "pliegues": "los canonicos, por grupo", "umbrales": "0,44 / 0,51 / 0,46",
            "confound_de_escala": "la matriz candidata se reescala a la norma de fila del control",
            "control": ("independiente y de tamano parecido, buscado por tamano de pool LOCAL "
                        "de cada pliegue. Un control por reetiquetacion NO sirve: permutar "
                        "columnas es simetria del modelo lineal regularizado y da resultado "
                        "identico, luego es un no-op demostrable"),
            "rejilla": "ninguna",
        },
        "control_de_vocabulario": {"rasgos_por_brazo": dims,
                                   "relaciones_que_sobrevivieron": rel_usadas,
                                   "min_df": 2, "segundos": round(time.time() - t0, 1)},
        "resultados": {"control": mc, "pista_M": mm, "control_aleatorio": mr},
        "lectura_en_tres_partes": {
            "coste_de_anadir_un_bloque": round(coste, 10),
            "valor_de_la_estructura_lexica": round(valor_estructura, 10),
            "fraccion_repagada": (round(valor_estructura / coste, 6) if coste > 0 else None),
            "saldo": round(aporte_real, 10)},
        "gate": {"delta_min": GATE_DELTA, "pliegues_min": GATE_PLIEGUES,
                 "delta": round(delta, 10), "pliegues_mejorados": folds_mej,
                 "superado": gate},
        "bootstrap": (None if ic_lo is None else
                      {"IC95": [round(ic_lo, 6), round(ic_hi, 6)], "n": N_BOOT}),
        "reproducibilidad": {"nivel": "R2", "git_commit": commit, "seed": SEMILLA,
                             "min_df": 2,
                             "tagger": "spaCy es_core_news_md 3.8.0, exclude=['ner'], D-003"},
        "decision": decision, "experimentos_registrados": 1,
        "formulacion": "V03 remains the best verified leaderboard result.",
    }
    JSON_M.write_text(json.dumps(informe, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")

    L: list[str] = []
    W = L.append
    W("# P62 - Pista M, triples lexicos de dependencia")
    W("")
    W(f"Generado por `experiments/p62_04_pista_m.py`. "
      f"Fecha {inicio.isoformat(timespec='seconds')}.")
    W("")
    W("## En que se diferencia de la pista L")
    W("")
    W("Un token `palabra__POS` dice que una palabra es un sustantivo. Un token")
    W("`cabeza__relacion__dependiente` dice que EN ESTE TWEET ese sustantivo es el objeto")
    W("directo de ESE verbo. Lo segundo es lo que distingue un insulto dirigido a alguien")
    W("de un adjetivo suelto, y ningun unigrama ni bigrama del texto crudo lo dice.")
    W("")
    W("Las relaciones se declaran antes de ejecutar, y no se eligen por como discriminan.")
    W("")
    W("```text")
    W("relaciones  " + ", ".join(RELACIONES))
    W("```")
    W("")
    W("## Los tres brazos")
    W("")
    W("| brazo | corrected_cv | delta vs V03 |")
    W("|---|---:|---:|")
    W(f"| control, V03 sin canal | {mc['corrected_cv']:.10f} | +0.0000000000 |")
    W(f"| pista M, + triple lexico | {mm['corrected_cv']:.10f} | {aporte_real:+.10f} |")
    W(f"| control aleatorio | {mr['corrected_cv']:.10f} | {aporte_aleatorio:+.10f} |")
    W("")
    W("## La lectura en tres partes")
    W("")
    W("```text")
    W(f"(1) coste de anadir el bloque     {coste:+.10f}")
    W(f"(2) valor de la estructura        {valor_estructura:+.10f}")
    W(f"(3) saldo                         {aporte_real:+.10f}")
    W("```")
    W("")
    W("## Control de vocabulario")
    W("")
    W("| brazo | rasgos por pliegue |")
    W("|---|---|")
    for clave in ("control", "pista_M", "control_aleatorio"):
        W(f"| {clave} | {dims[clave]} |")
    W("")
    W("Relaciones que sobrevivieron a min_df=2, con el numero de rasgos:")
    W("")
    W("```text")
    for rel, c in sorted(rel_usadas.items(), key=lambda kv: -kv[1])[:12]:
        W(f"  {rel:12} {c:6d}")
    W("```")
    W("")
    W("## Gate")
    W("")
    W("```text")
    W(f"delta {delta:+.10f} | pliegues {folds_mej}/5 | superado {gate}")
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
    W("*Generado por `experiments/p62_04_pista_m.py`. Sin push.*")
    MD_M.write_text(a_ascii("\n".join(L)), encoding="ascii")

    for destino in (MD_M, JSON_M):
        if any(b > 127 for b in destino.read_bytes()):
            raise SystemExit(f"PARADA: {destino.name} conserva bytes no ASCII")
    log("")
    log(f"P62-04 OK | pista M {mm['corrected_cv']:.10f} delta {delta:+.10f} | control "
        f"aleatorio delta {aporte_aleatorio:+.10f} | valor de la estructura "
        f"{valor_estructura:+.10f} | gate {gate}")


if __name__ == "__main__":
    ast.parse(Path(__file__).read_text(encoding="utf-8"))
    main()
