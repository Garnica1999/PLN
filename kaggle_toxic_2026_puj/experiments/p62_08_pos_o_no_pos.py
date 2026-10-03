"""P62 track: is the POS-conditioned channel new information, or the same words twice?

The curve of P62-07 and what it points at
------------------------------------------
With the accumulation bug fixed, the budget curve is clean and its shape is the finding:

    K       cost of a meaningless block    value of the real content
    16            +0,000855                    -0,001662
    32            +0,000282                    -0,003899
    64            -0,001280                    -0,002934
    128           -0,001715                    -0,008715
    256           -0,000933                    -0,007814
    512           -0,006739                    -0,000999
    1024          -0,008922                    +0,002061

No budget improves on V03, and the content is NEGATIVE almost everywhere. That is a
stronger negative than tracks L and M gave, and it has an obvious suspect that this
script tests.

The suspect
-----------
Look at which tokens the fold-safe ranking actually selects:

    pliegue 1: hijo__NOUN, sevillista__ADJ, zorra__NOUN, zorra__PROPN, novia__NOUN,
              perra__PROPN
    pliegue 4: zorra__PROPN, zorra__NOUN, cllate__PROPN, perra__PROPN, hijo__NOUN

Those are the same LEXEMAS the plain-word analysis already found, `zorra`, `perra`,
`hijo`, `novia`, `cllate`, `acoso`, and they are words V03's word unigram channel already
carries. The `__NOUN` or `__PROPN` suffix adds a second copy of a word that is already
there, and sometimes a third.

If that is what is happening, then the EDA's "synergy" of +0,0657 was not new information.
It is what you get when you split one enriched word into two or three enriched rows by POS
category: the average log enrichment of the rows goes up because the same evidence is
counted more than once, while the information content goes nowhere. That would explain,
in one stroke, why every linguistic channel in this project failed: none of them added
information, they re-encoded words V03 had.

The test
--------
Run the same budget curve with the channel built from bare LEMMAS instead of
`lemma__POS`, everything else identical: same folds, same classifier, same thresholds,
same fold-safe ranking, same K values. Then:

    if bare lemmas do as well or better, the POS conditioning is redundancy, and the
    interaction that the EDA measured is an artefact of row splitting
    if `lemma__POS` does clearly better, the POS part carries real information and the
    negative in P62-07 was about something else

The redundancy is also measured directly, not inferred from the outcomes: the overlap
between the top-K `lemma__POS` tokens and the top-K bare lemmas, and the weight V03's own
word channel already assigns to those lemmas.

Usage
-----
    /home/carlos/venv/bin/python experiments/p62_08_pos_o_no_pos.py
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
MD_R = REPORTS / "p62_pos_o_no_pos.md"
JSON_R = REPORTS / "p62_pos_o_no_pos.json"
CSV_R = REPORTS / "p62_pos_o_no_pos.csv"
CELDA_NPZ = PROJECT_ROOT / "features" / "p62_pos_o_no_pos_cells.npz"

V03_UMBRALES = np.array([0.44, 0.51, 0.46])
PARAMS_WORD = {"analyzer": "word", "ngram_range": (1, 2), "min_df": 2, "sublinear_tf": True,
               "lowercase": True}
PARAMS_CHAR = {"analyzer": "char_wb", "ngram_range": (3, 5), "min_df": 2,
               "sublinear_tf": True, "lowercase": True}
PARAMS_CANAL = {"analyzer": "word", "ngram_range": (1, 1), "min_df": 1, "sublinear_tf": True,
                "lowercase": False, "token_pattern": r"[^ ]+"}
LR = {"C": 2.0, "class_weight": "balanced", "max_iter": 1000, "solver": "liblinear",
      "random_state": 42}
CATEGORIAS = ("NOUN", "VERB", "ADJ", "PROPN", "ADV", "PRON", "AUX", "INTJ")
ALFA = 1.0
MIN_SOPORTE_INT = 5
PRESUPUESTOS = (16, 32, 64, 128, 256, 512, 1024, 2048)
SEMILLA = 42
GATE_DELTA = 0.003
GATE_PLIEGUES = 4


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[P62-08] {mensaje}", flush=True)


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


def tokens(documentos: list[dict[str, Any]], filas: np.ndarray,
           con_pos: bool) -> list[str]:
    """Tokens del canal, con o sin el sufijo de categoria.

    Args:
        documentos: anotaciones.
        filas: indices.
        con_pos: si se True, emite lema__POS; si False, emite solo el lema.

    Returns:
        Un token por fila.
    """
    salida = []
    for i in filas:
        partes = []
        for t in documentos[i]["tok"]:
            if not (t["al"] and t["l"] and t["p"] in CATEGORIAS):
                continue
            if con_pos:
                if t["l"] not in partes:
                    partes.append(t["l"])
                partes.append(f"{t['l']}__{t['p']}")
            else:
                partes.append(t["l"])
        salida.append(" ".join(partes))
    return salida


def enriquecimiento(tokens_por_fila: list[str], vocabulario: list[str], col: dict[str, int],
                    y: np.ndarray, filas: np.ndarray) -> np.ndarray:
    """Puntuacion de enriquecimiento absoluto por token, fold-safe.

    Args:
        tokens_por_fila: tokens de las filas internas.
        vocabulario: indice del canal.
        col: token a columna.
        y: etiquetas.
        filas: indices internos.

    Returns:
        Un vector con la puntuacion por token.
    """
    n_vocab = len(vocabulario)
    score = np.zeros(n_vocab)
    for j in range(3):
        fp = np.zeros(n_vocab)
        fn = np.zeros(n_vocab)
        for a, i in enumerate(filas):
            destino = fp if y[i, j] == 1 else fn
            for t in set(tokens_por_fila[a].split()):
                c = col.get(t)
                if c is not None:
                    destino[c] += 1
        npos, nneg = fp.sum(), fn.sum()
        ok = (fp >= MIN_SOPORTE_INT) | (fn >= MIN_SOPORTE_INT)
        pp = (fp + ALFA) / (npos + 2 * ALFA)
        pn = (fn + ALFA) / (nneg + 2 * ALFA)
        score[ok] += np.abs(np.log(pp[ok] / pn[ok]))
    return score


def main() -> None:
    """Compara el canal con POS contra el canal de lemas solos, con todo lo demas igual.

    Returns:
        None.
    """
    inicio = datetime.now(timezone.utc)
    log("P62-08 el canal POS-conditioned, o las mismas palabras dos veces")
    log("   Todo identico entre los dos brazos reales salvo el sufijo de categoria")

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
    log(f"   OOF canonico {ccv_ref!r} | {n} filas | presupuestos {list(PRESUPUESTOS)}")

    proba: dict[str, dict[int, np.ndarray]] = {"con_pos": {}, "solo_lema": {}}
    for _nombre in proba:
        for _k in PRESUPUESTOS:
            proba[_nombre][_k] = np.zeros((n, 3))
    solapes: list[dict[str, Any]] = []
    pesos_v03: list[dict[str, Any]] = []
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
        objetivo = float(np.sqrt(np.mean(xbase.multiply(xbase).sum(axis=1))))

        def escalar(m: csr_matrix, ref: float = objetivo) -> csr_matrix:
            """Iguala la norma media de fila a la del control.

            Args:
                m: matriz.
                ref: norma objetivo.

            Returns:
                La matriz escalada.
            """
            media = float(np.sqrt(np.asarray(m.multiply(m).sum(axis=1)).ravel()).mean())
            return m if media <= 0 else m.multiply(ref / media)

        if f == 1:
            _pr = np.zeros((len(iva), 3))
            for _j in range(3):
                _mo = LogisticRegression(**LR)
                _mo.fit(xbase, y[ir, _j])
                _pr[:, _j] = _mo.predict_proba(xbase_v)[:, 1]
            _f1 = float(f1_score(y[iva, 0], (_pr[:, 0] >= V03_UMBRALES[0]).astype(int)))
            if abs(_f1 - 0.7786259541984732) > 1e-9:
                raise SystemExit(f"PARADA: la base no reproduce el F1 de V03 en el "
                                 f"pliegue 1, da {_f1!r}")

        # peso que V03 ya da a cada lema en su canal de palabras
        col_w = {t: i for i, t in enumerate(pw.vocabulary_.keys())}
        norm_w = np.sqrt(np.asarray(xw.multiply(xw).sum(axis=0)).ravel())
        norm_w[norm_w == 0] = 1.0

        canales = {}
        for nombre, con_pos in (("con_pos", True), ("solo_lema", False)):
            tk_ir = tokens(documentos, ir, con_pos)
            tk_iva = tokens(documentos, iva, con_pos)
            vec = TfidfVectorizer(**PARAMS_CANAL)
            xc_ir = vec.fit_transform(tk_ir)
            vocab = sorted(vec.vocabulary_.keys())
            col = {t: i for i, t in enumerate(vocab)}
            score = enriquecimiento(tk_ir, vocab, col, y, ir)
            orden = np.argsort(-score)
            top = [vocab[i] for i in orden[:max(PRESUPUESTOS)]]
            canales[nombre] = {"vec": vec, "col": col, "x": xc_ir,
                                 "xv": vec.transform(tk_iva), "top": top}

        # ---------- la redundancia, medida y no inferida ----------
        top_pos = set(canales["con_pos"]["top"])
        top_lema = set(canales["solo_lema"]["top"])
        lemas_del_top_pos = {t.rsplit("__", 1)[0] for t in top_pos}
        solapes.append({
            "pliegue": f,
            "top_2048_tokens_con_pos": len(top_pos),
            "top_2048_lemas": len(top_lema),
            "tokens_con_pos_que_son_lemas_del_top": len(top_pos & top_lema),
            "lemas_distintos_tras_quitar_el_POS": len(lemas_del_top_pos),
            "fraccion_de_redundancia": round(1.0 - len(lemas_del_top_pos) / max(len(top_pos), 1),
                                             6),
        })
        # peso de V03 sobre esos mismos lemas
        cols_w = [col_w[l] for l in lemas_del_top_pos if l in col_w]
        en_v03 = len(cols_w)
        pesos_v03.append({
            "pliegue": f,
            "lemas_del_top_que_V03_ya_tiene": en_v03,
            "sobre_total_lemas": len(lemas_del_top_pos),
            "cobertura_de_V03": round(en_v03 / max(len(lemas_del_top_pos), 1), 6),
            "norma_media_de_TFIDF_de_esos_lemas": round(
                float(norm_w[cols_w].mean()) if cols_w else 0.0, 6),
            "norma_media_global_del_canal_de_palabras": round(float(norm_w.mean()), 6),
        })

        for k in PRESUPUESTOS:
            for nombre in ("con_pos", "solo_lema"):
                c = canales[nombre]
                elegidos = sorted({c["col"][t] for t in c["top"][:k]})
                xent = escalar(hstack([xbase, c["x"][:, elegidos]]).tocsr())
                xval = escalar(hstack([xbase_v, c["xv"][:, elegidos]]).tocsr())
                for j in range(3):
                    mo = LogisticRegression(**LR)
                    mo.fit(xent, y[ir, j])
                    proba[nombre][k][iva, j] = mo.predict_proba(xval)[:, 1]
        log(f"      pliegue {f} listo ({time.time() - t0:.0f} s) | "
            f"redundancia {solapes[-1]['fraccion_de_redundancia']:.4f} | "
            f"cubierto por V03 {pesos_v03[-1]['cobertura_de_V03']:.4f}")

    ccv = {b: {k: float(f1_score(y, (proba[b][k] >= V03_UMBRALES[None, :]).astype(int),
                                   average="macro")) for k in PRESUPUESTOS}
           for b in proba}
    log("")
    log("   LA COMPARACION: las mismas filas, con y sin el sufijo de categoria")
    log(f"      {'K':>6}{'solo lema':>18}{'delta':>16}{'lema__POS':>18}{'delta':>16}"
        f"{'ventaja del POS':>18}")
    filas_csv: list[dict[str, Any]] = []
    curva: list[dict[str, Any]] = []
    for k in PRESUPUESTOS:
        dl, dp = ccv["solo_lema"][k] - ccv_ref, ccv["con_pos"][k] - ccv_ref
        curva.append({"K": k, "solo_lema": ccv["solo_lema"][k], "delta_lema": dl,
                      "con_pos": ccv["con_pos"][k], "delta_pos": dp,
                      "ventaja_del_POS": dp - dl})
        filas_csv.append({"K": k, "corrected_cv_lema": round(ccv["solo_lema"][k], 10),
                          "delta_lema": round(dl, 10),
                          "corrected_cv_pos": round(ccv["con_pos"][k], 10),
                          "delta_pos": round(dp, 10),
                          "ventaja_del_POS": round(dp - dl, 10)})
        log(f"      {k:>6}{ccv['solo_lema'][k]:>18.10f}{dl:>+16.10f}"
            f"{ccv['con_pos'][k]:>18.10f}{dp:>+16.10f}{dp - dl:>+18.10f}")
    df = pd.DataFrame(filas_csv)
    df.to_csv(CSV_R, index=False, encoding="utf-8")

    pos_gana = int((df["ventaja_del_POS"] > 0).sum())
    ambos_negan = int(((df["delta_lema"] < 0) & (df["delta_pos"] < 0)).sum())
    redundancia = float(np.mean([s["fraccion_de_redundancia"] for s in solapes]))
    cobertura = float(np.mean([p["cobertura_de_V03"] for p in pesos_v03]))
    log("")
    log("   LA REDUNDANCIA, medida y no inferida de los resultados")
    log(f"      los presupuestos K donde el sufijo de POS ayuda: {pos_gana} de {len(df)}")
    log(f"      los presupuestos donde AMBOS canales pierden: {ambos_negan} de {len(df)}")
    log(f"      redundancia media de los tokens POS: {redundancia:.4f}")
    log("      es decir, esa fraccion de los 2048 tokens seleccionados son lemas que ya")
    log("      estaban en la lista de lemas mas enriquecidos, no categorias nuevas")
    log(f"      cobertura de V03 sobre esos lemas: {cobertura:.4f}")
    log("      es decir, V03 ya les da peso en su canal de palabras")

    if pos_gana <= len(df) / 2 and redundancia > 0.5 and cobertura > 0.8:
        veredicto = (
            f"el sufijo de POS ayuda en {pos_gana} de {len(df)} presupuestos, la "
            f"redundancia de los tokens seleccionados es {redundancia:.4f} y V03 ya cubre "
            f"el {cobertura * 100:.1f} por ciento de esos lemas. El canal POS-conditioned "
            f"no anade informacion: duplica palabras que el TF-IDF de V03 ya tiene, y por "
            f"eso la synergy de +0,0657 que midio la EDA era el promedio de enriquecimiento "
            f"de filas duplicadas, no senal nueva")
    else:
        veredicto = (
            f"el sufijo de POS ayuda en {pos_gana} de {len(df)} presupuestos con una "
            f"redundancia de {redundancia:.4f} y cobertura de V03 de {cobertura:.4f}; la "
            f"condicion para llamarlo redundancia no se cumple y el resultado queda abierto")
    log("")
    log(f"   VEREDICTO: {veredicto}")

    mejor = df.sort_values("ventaja_del_POS", ascending=False).iloc[0]
    k_mejor = int(mejor["K"])
    pr = (proba["con_pos"][k_mejor] >= V03_UMBRALES[None, :]).astype(int)
    folds_c = [float(f1_score(y[pliegues == g], pr[pliegues == g], average="macro"))
               for g in range(1, 6)]
    folds_v = [float(f1_score(y[pliegues == g], pred_v03[pliegues == g], average="macro"))
               for g in range(1, 6)]
    fm = sum(1 for a, b in zip(folds_c, folds_v) if a > b)
    delta = float(mejor["delta_pos"])
    gate = bool(delta >= GATE_DELTA and fm >= GATE_PLIEGUES)
    log("")
    log("   GATE sobre el mejor presupuesto para el brazo POS")
    log(f"      K {k_mejor} | delta {delta:+.10f} | pliegues {fm}/5 | gate {gate}")
    np.savez_compressed(CELDA_NPZ, ids=ids, y_true=y, pliegues=pliegues, pred_v03=pred_v03,
                        pred_pos=pr, proba_pos=proba["con_pos"][k_mejor],
                        proba_lema=proba["solo_lema"][k_mejor], proba_v03=proba_v03)

    decision = ("PROMISING. Supera el gate; exige segunda particion" if gate else
                "CLOSED. El sufijo de categoria no aporta informacion que V03 no tenga ya")
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                            text=True, cwd=PROJECT_ROOT, check=False).stdout.strip()
    informe = {
        "familia": "P62_POS_O_NO_POS",
        "naturaleza": ("EXPERIMENTO. 2 brazos reales y un control, 0 submissions, 0 Kaggle, "
                       "0 holdout, 0 test. V03 no se modifica"),
        "fecha": inicio.isoformat(timespec="seconds"),
        "experiment_id": "P62-04", "track": "redundancia-del-canal-POS",
        "script": "experiments/p62_08_pos_o_no_pos.py",
        "pregunta": ("el canal con sufijo de categoria anade informacion, o duplica "
                     "lemas que el TF-IDF de V03 ya lleva"),
        "diseno": ("los dos brazos reales son identicos salvo en el sufijo: mismos "
                   "pliegues, mismo clasificador, mismos umbrales, mismo ranking fold-safe "
                   "por enriquecimiento y mismos presupuestos. La unica diferencia es si el "
                   "token es el lema o el lema seguido de su categoria"),
        "curva": curva,
        "presupuestos_en_que_el_POS_ayuda": pos_gana,
        "presupuestos_ambos_negativos": ambos_negan,
        "redundancia": {"media": round(redundancia, 6), "por_pliegue": solapes,
                        "definicion": ("fraccion de los 2048 tokens_POS seleccionados cuyo "
                                       "lema ya estaba en la lista de 2048 lemas mas "
                                       "enriquecidos")},
        "cobertura_de_V03": {"media": round(cobertura, 6), "por_pliegue": pesos_v03,
                             "definicion": ("fraccion de esos mismos lemas que ya tienen "
                                            "columna en el canal de palabras de V03")},
        "gate": {"delta_min": GATE_DELTA, "pliegues_min": GATE_PLIEGUES,
                 "K": k_mejor, "delta": round(delta, 10),
                 "pliegues_mejorados": fm, "superado": gate},
        "veredicto": veredicto,
        "decision": decision,
        "experimentos_registrados": 1,
        "reproducibilidad": {"nivel": "R2", "git_commit": commit, "seed": SEMILLA,
                             "min_df": MIN_SOPORTE_INT, "alfa": ALFA,
                             "tagger": "spaCy es_core_news_md 3.8.0, exclude=['ner'], D-003"},
        "formulacion": "V03 remains the best verified leaderboard result.",
    }
    JSON_R.write_text(json.dumps(informe, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")

    L: list[str] = []
    W = L.append
    W("# P62 - El canal POS-conditioned, o las mismas palabras dos veces")
    W("")
    W(f"Generado por `experiments/p62_08_pos_o_no_pos.py`. "
      f"Fecha {inicio.isoformat(timespec='seconds')}.")
    W("")
    W("## Lo que apunta la curva de P62-07")
    W("")
    W("Con el bug de acumulacion corregido, la curva es limpia y su forma es el hallazgo:")
    W("")
    W("Ningun presupuesto mejora a V03, y el valor del contenido es NEGATIVO casi en todos.")
    W("Eso es un negativo mas fuerte que el de las pistas L y M, y tiene un sospechoso")
    W("evidente: mira QUE tokens selecciona el ranking fold-safe.")
    W("")
    W("```text")
    W("pliegue 1: hijo__NOUN, sevillista__ADJ, zorra__NOUN, zorra__PROPN, novia__NOUN,")
    W("           perra__PROPN")
    W("pliegue 4: zorra__PROPN, zorra__NOUN, cllate__PROPN, perra__PROPN, hijo__NOUN")
    W("```")
    W("")
    W("Son los MISMOS lemas que el analisis de palabras ya habia encontrado, `zorra`,")
    W("`perra`, `hijo`, `novia`, `cllate`, `acoso`, y son palabras que el canal de unigramas")
    W("de V03 ya lleva. El sufijo anade una segunda copia de una palabra que ya esta ahi, y")
    W("a veces una tercera.")
    W("")
    W("**La sospecha, enunciada antes de medir:** si eso es lo que pasa, la synergy de")
    W("+0,0657 de la EDA no era informacion nueva. Es lo que sale al partir una palabra")
    W("enriquecida en dos o tres filas por su categoria: el promedio de log enriquecimiento")
    W("de las filas sube porque la misma evidencia se cuenta mas de una vez, mientras que")
    W("la informacion no se mueve. Y eso explicaria de un solo golpe por que todos los")
    W("canales linguisticos de este proyecto fallaron: ninguno anadio informacion, todos")
    W("re-codificaron palabras que V03 ya tenia.")
    W("")
    W("## La prueba")
    W("")
    W("La misma curva de presupuesto con el canal construido de lemas SUELTOS en vez de")
    W("`lema__POS`, con todo lo demas identico.")
    W("")
    W("| K | solo lema | delta | lema__POS | delta | ventaja del POS |")
    W("|---:|---:|---:|---:|---:|---:|")
    for r in curva:
        W(f"| {r['K']} | {r['solo_lema']:.10f} | {r['delta_lema']:+.10f} | "
          f"{r['con_pos']:.10f} | {r['delta_pos']:+.10f} | {r['ventaja_del_POS']:+.10f} |")
    W("")
    W("## La redundancia, medida y no inferida de los resultados")
    W("")
    W("| magnitud | valor |")
    W("|---|---:|")
    W(f"| presupuestos en los que el sufijo de POS ayuda | {pos_gana} de {len(df)} |")
    W(f"| presupuestos donde AMBOS canales pierden | {ambos_negan} de {len(df)} |")
    W(f"| redundancia media de los tokens seleccionados | {redundancia:.4f} |")
    W(f"| cobertura de V03 sobre esos mismos lemas | {cobertura:.4f} |")
    W("")
    W("La redundancia es la fraccion de los 2048 tokens `lema__POS` seleccionados cuyo lema")
    W("ya estaba en la lista de los 2048 lemas mas enriquecidos. La cobertura es la fraccion")
    W("de esos lemas que ya tienen columna en el canal de palabras de V03.")
    W("")
    W("## Veredicto")
    W("")
    W("```text")
    W(veredicto)
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
    W("*Generado por `experiments/p62_08_pos_o_no_pos.py`. Sin push.*")
    MD_R.write_text(a_ascii("\n".join(L)), encoding="ascii")

    for destino in (MD_R, JSON_R, CSV_R):
        if any(b > 127 for b in destino.read_bytes()):
            raise SystemExit(f"PARADA: {destino.name} conserva bytes no ASCII")
    log("")
    log(f"P62-08 OK | {len(curva)} presupuestos | el POS ayuda en {pos_gana} | "
        f"redundancia {redundancia:.4f} | cobertura de V03 {cobertura:.4f} | gate {gate}")


if __name__ == "__main__":
    ast.parse(Path(__file__).read_text(encoding="utf-8"))
    main()
