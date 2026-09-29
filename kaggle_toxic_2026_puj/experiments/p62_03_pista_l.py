"""P62 track L: POS-conditioned sparse tokens, added to V03's own feature space.

What the EDA said, and what this experiment is for
-------------------------------------------------
The linguistic EDA decomposed the label association into the word, the POS category and
the interaction between them:

    word alone                  +0,3773
    category alone               +0,0095
    word and category together  +0,4525
    synergy                      +0,0657   which is 17,41 % of the word signal

The synergy is the part that cannot be written as a function of the word and the category
separately, so it is the part neither existing representation can express: V03's TF-IDF
knows the word and discards the category, and E2-03's POS channel knows the category and
discards the word. This experiment gives the model the pair.

The gap check said this is new
------------------------------
``p62_00_auditoria_y_gap.py`` verified against the source, not against names, that no
token of the form ``word__POS`` is emitted anywhere in the project, and that the script
ABORTS if one appears. The closest neighbours are 36 dense ``pos_*`` count columns and the
categorial ``POSchild_POShead`` sequence of E2-03, neither of which carries the identity
of the word.

Why this holds representation and model apart
---------------------------------------------
Paragraph 36 forbids changing both at once. The classifier is untouched: the same
LogisticRegression, ``C=2,0``, ``class_weight=balanced``, ``solver=liblinear``,
``max_iter=1000``, ``random_state=42``, the same folds, the same thresholds. The only
thing that changes is that one more sparse block is concatenated.

The scale confound, which is real and is controlled
---------------------------------------------------
Each TF-IDF block is L2-normalised on its own, so V03's row has norm sqrt(2), and
concatenating a third block of weight s raises it to sqrt(2 + s^2). That is a uniform
rescaling of every coefficient, which is the same as changing C, and paragraph 36 would be
violated through the back door. So the candidate matrix is rescaled so its row norm
equals V03's, which makes the two arms differ in a DIRECTION in feature space and not in
a SCALE. At the control's own parameters the factor is exactly one, so the control is
untouched.

The third arm, which is the one that decides whether the gain means anything
----------------------------------------------------------------------------
A candidate that adds features can win for a reason that has nothing to do with
linguistics: more dimensions, more sparsity, a different effective regularisation. So a
third arm adds a block of RANDOM tokens of exactly the same size, drawn from the same
alphabet length, with the same support rule. If track L only matches the random block,
then what was measured was dimensionality, and the mechanism claim is false. That is the
same logic as the negative control in P61, applied to a representation instead of to a
weight search.

Vocabulary control
------------------
Paragraph 34 requires the vocabulary to be reported rather than allowed to explode. The
block is built with ``min_df=2`` on the training rows of each fold, so the vocabulary is
induced inside the fold and never sees the held-out rows.

Usage
-----
    /home/carlos/venv/bin/python experiments/p62_03_pista_l.py
"""

from __future__ import annotations

import ast
import csv as _csv
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
EXPERIMENTS = PROJECT_ROOT / "experiments"
OOF = PROJECT_ROOT / "features" / "oof_v03_final.npz"
TRAIN_RAW = PROJECT_ROOT / "data" / "raw" / "toxic_train.csv"
PARTICION = PROJECT_ROOT / "data" / "processed" / "holdout_v2_ids.csv"
ANOTACION = PROJECT_ROOT / "features" / "p62_anotacion_spacy.json.gz"
CSV_GAP_LING = REPORTS / "p62_linguistic_gap.csv"
MD_L = REPORTS / "p62_pista_l_word_pos.md"
JSON_L = REPORTS / "p62_pista_l_word_pos.json"
CELDA_NPZ = PROJECT_ROOT / "features" / "p62_l_word_pos_cells.npz"

V03_UMBRALES = np.array([0.44, 0.51, 0.46])
PARAMS_WORD = {"analyzer": "word", "ngram_range": (1, 2), "min_df": 2, "sublinear_tf": True,
               "lowercase": True}
PARAMS_CHAR = {"analyzer": "char_wb", "ngram_range": (3, 5), "min_df": 2,
               "sublinear_tf": True, "lowercase": True}
PARAMS_CANAL = {"analyzer": "word", "ngram_range": (1, 1), "min_df": 2, "sublinear_tf": True,
                "lowercase": False, "token_pattern": r"[^ ]+"}
LR = {"C": 2.0, "class_weight": "balanced", "max_iter": 1000, "solver": "liblinear",
      "random_state": 42}
CATEGORIAS_CANAL = ("NOUN", "VERB", "ADJ", "PROPN", "ADV", "PRON", "AUX", "INTJ")
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
    print(f"[P62-03] {mensaje}", flush=True)


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


def escribir_gap_linguistico(eda: dict[str, Any]) -> list[dict[str, Any]]:
    """Escribe el informe de hueco linguistico, ANTES de entrenar.

    Paragraph 27 asks which observed properties are already represented and which are
    genuinely absent. Writing it before the experiment is what makes it a decision
    document instead of a justification.

    Args:
        eda: artefacto de la EDA linguistica.

    Returns:
        Las filas del informe.
    """
    eda["composicion_pos"]
    enf = eda["enriquecimiento"]
    filas = [
        {"linguistic_phenomenon": "categoria POS de cada token",
         "existing_representation": "36 columnas densas pos_X_n y pos_X_frac, mas el canal "
                                   "categorial SYN-POS de E2-03 que descarta la palabra",
         "historical_experiment": "E2-03-POS -0,2135; E2-03-POS-TXT -0,0116 con 0/5",
         "status": "already_tested_como_marginal",
         "potential_new_representation": "ninguna: la etiqueta por si sola esta cerrada",
         "evidencia_eda": f"enriquecimiento de la categoria sola {eda['pregunta_central_seccion_24']['solo_categoria']:+.4f}, "
                          f"la mas baja de la tabla"},
        {"linguistic_phenomenon": "identidad lexical del token, sola",
         "existing_representation": "el TF-IDF word(1,2) de V03, exactamente",
         "historical_experiment": "V03, corrected_cv 0,7772114525165721",
         "status": "already_tested_como_control",
         "potential_new_representation": "ninguna: es el baseline",
         "evidencia_eda": f"enriquecimiento de la palabra sola {eda['pregunta_central_seccion_24']['solo_palabra']:+.4f}"},
        {"linguistic_phenomenon": "IDENTIDAD LEXICAL junto a la CATEGORIA, el mismo lexema "
                                  "con distinto papel predictivo",
         "existing_representation": "NINGUNA. Ningun token palabra__POS se emite en el "
                                   "proyecto, y el script de gap analysis aborta si aparece",
         "historical_experiment": "ninguno: esta es la pista L",
         "status": "genuinely_absent",
         "potential_new_representation": f"canal sparse palabra__POS sobre {len(CATEGORIAS_CANAL)} "
                                         f"categorias con soporte, inducido dentro del "
                                         f"pliegue con min_df=2",
         "evidencia_eda": f"synergy {eda['pregunta_central_seccion_24']['sinergia']:+.4f}, "
                          f"que es el {eda['pregunta_central_seccion_24']['sinergia_sobre_palabra'] * 100:.2f} "
                          f"por ciento de la senal de la palabra y no se puede escribir como "
                          f"funcion de las dos por separado"},
        {"linguistic_phenomenon": "relacion de dependencia entre dos palabras concretas",
         "existing_representation": "31 columnas densas dep_X_n, y el canal E2-04 que emite "
                                   "POS_dependiente_POS_cabeza, DEP_cabeza_POS y DEP_cabeza, "
                                   "sin la palabra dependiente",
         "historical_experiment": "E2-04-DEPS -0,2258; E2-04-DEPS-TXT -0,0119 con 1/5",
         "status": "partially_tested",
         "potential_new_representation": "triple lexical cabeza__relacion__dependiente, "
                                         "que es la pista M y queda para el siguiente "
                                         "experimento",
         "evidencia_eda": "los patrones POS__dep__POS mas frecuentes se concentran en la "
                          "cabeza, lo que sugiere que la parte lexical es la que falta"},
        {"linguistic_phenomenon": "secuencia de POS de longitud dos a cuatro",
         "existing_representation": "exactamente la propuesta: E2-03 vectoriza la secuencia "
                                   "POS_dependiente_POS_cabeza con ngram_range (1,4)",
         "historical_experiment": "E2-03-POS-TXT -0,0116 con 0/5",
         "status": "functionally_equivalent",
         "potential_new_representation": "ninguna: se cierra sin consumir experimento",
         "evidencia_eda": "no aplica, la representacion ya existe y fallo"},
        {"linguistic_phenomenon": "tweets cortos y de clausula simple, donde V03 falla con "
                                  "alta confianza",
         "existing_representation": "parcial: char_wb(3,5) captura la forma pero no la "
                                   "estructura, y no hay ninguna feature de longitud "
                                   "estructural en el canal principal",
         "historical_experiment": "E2-16 uso umbrales por rebanada de longitud y cerro; "
                                  "P60-L uso cuartiles de longitud y fue MARGINAL",
         "status": "already_tested_por_rebanada",
         "potential_new_representation": "NO se abre por rebanada: el parrafo 0 prohibe "
                                         "volver a los selectores y las rebanadas, y la "
                                         "pista L ataca la estructura sin rebanar",
         "evidencia_eda": "delta de F1 de V03 en la rebanada corta menos la larga "
                          "-0,2330 de media, con tasa de error de alta confianza 0,4167 en "
                          "TR corto frente a 0,0433 en TR largo"},
        {"linguistic_phenomenon": "alcance de la negacion",
         "existing_representation": "rasgos densos que cuentan insultos y targets DENTRO del "
                                   "alcance de la negacion, mas un token NEG_cabeza",
         "historical_experiment": "E2-08-NEG -0,2505; E2-08-NEG-TXT -0,0115 con 0/5",
         "status": "functionally_equivalent",
         "potential_new_representation": "ninguna: se cierra sin consumir experimento",
         "evidencia_eda": "los negadores aparecen en el 1,6 por ciento de los tokens de las "
                          "tres clases, sin diferencia apreciable entre ellas"},
        {"linguistic_phenomenon": "perfil pronominal y targeting",
         "existing_representation": "columnas col_segunda_persona, col_densidad y "
                                   "plural_generico, y el canal E2-05 de emparejamiento",
         "historical_experiment": "E2-05-CH-TGT -0,1696 con 0/5",
         "status": "already_tested",
         "potential_new_representation": "ninguna",
         "evidencia_eda": "el perfil pronominal es casi igual en las tres clases, con "
                          "segunda persona entre 0,039 y 0,053"},
    ]
    with CSV_GAP_LING.open("w", encoding="ascii", newline="") as fh:
        w = _csv.writer(fh, lineterminator="\n")
        w.writerow(["linguistic_phenomenon", "existing_representation",
                    "historical_experiment", "status", "potential_new_representation",
                    "evidencia_eda"])
        for f in filas:
            w.writerow([a_ascii(f[k]) for k in
                        ("linguistic_phenomenon", "existing_representation",
                         "historical_experiment", "status",
                         "potential_new_representation", "evidencia_eda")])
    log(f"      escrito reports/p62_linguistic_gap.csv con {len(filas)} fenomenos, "
        f"ANTES de entrenar")
    log(f"      genuinamente ausentes: "
        f"{[f['linguistic_phenomenon'][:46] for f in filas if f['status'] == 'genuinely_absent']}")
    log(f"      Total de verbos enriquecidos en HS, como muestra de que la EDA tiene "
        f"contenido: {[r['lema'] for r in enf['HS']['VERB']['top_enriquecidos'][:6]]}")
    return filas


def tokens_word_pos(documentos: list[dict[str, Any]], filas: np.ndarray,
                    rng: np.random.Generator | None, tam_pool: int = 0) -> list[str]:
    """Construye los tokens del canal POS para un conjunto de filas.

    Args:
        documentos: anotaciones.
        filas: indices de las filas del conjunto.
        rng: si se da, se generan identificadores ALEATORIOS en vez de lemas, que es el
            control de dimensionalidad del tercer brazo. Se conservan las categorias, para
            que el control tenga la misma composicion gramatical y la diferencia sea la
            identidad de la palabra y no la categoria.
        tam_pool: numero de entradas del pool aleatorio.

    Returns:
        Un token por fila, separados por espacio.
    """
    salida = []
    for i in filas:
        if rng is None:
            partes = [f"{t['l']}__{t['p']}" for t in documentos[i]["tok"]
                      if t["al"] and t["p"] in CATEGORIAS_CANAL and t["l"]]
        else:
            partes = [f"z{int(rng.integers(0, tam_pool))}_{t['p']}"
                      for t in documentos[i]["tok"]
                      if t["al"] and t["p"] in CATEGORIAS_CANAL]
        salida.append(" ".join(partes))
    return salida


def main() -> None:
    """Ejecuta la pista L con su control de dimensionalidad.

    Returns:
        None.
    """
    inicio = datetime.now(timezone.utc)
    log("P62-03 pista L, tokens condicionados por POS anadidos al espacio de V03")
    log("   clasificador IDENTICO a V03: LR C=2,0 balanced liblinear, mismos pliegues, "
        "mismos umbrales")

    hash_real = hashlib.sha256(V03_SUB.read_bytes()).hexdigest()
    if hash_real != V03_SHA:
        raise SystemExit("PARADA: la submission de V03 no coincide con el sha256 congelado")

    eda = json.loads((REPORTS / "p62_linguistic_eda.json").read_text(encoding="utf-8"))
    log("")
    log("   INFORME DE HUECO LINGUISTICO, escrito antes de entrenar")
    escribir_gap_linguistico(eda)

    v = np.load(OOF, allow_pickle=True)
    ids, y, pliegues = v["ids"], v["y_true"].astype(int), v["pliegues"]
    pred_v03 = v["pred"].astype(int)
    proba_v03 = v["proba"]
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
    log(f"   OOF canonico {ccv_ref!r} | {n} filas | {len(set(grupos))} grupos intactos | "
        f"anotacion coherente")

    proba = {k: np.zeros((n, 3)) for k in ("control", "pista_L", "control_aleatorio")}
    dims: dict[str, list[int]] = {k: [] for k in proba}
    densidad: dict[str, list[float]] = {k: [] for k in proba}
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
        # El canal se induce SOLO con las filas de entrenamiento del pliegue.
        tok_l = tokens_word_pos(documentos, ir, None)
        tok_v = tokens_word_pos(documentos, iva, None)
        pc_l = TfidfVectorizer(**PARAMS_CANAL)
        xpos = pc_l.fit_transform(tok_l)
        xpos_v = pc_l.transform(tok_v)
        # CONTROL EMPAREJADO. La primera version sorteaba nombres de un pool dimensionado
        # con el numero de lemas__POS distintos, que es mayor que el conjunto que
        # sobrevive a min_df=2, con lo que el bloque aleatorio salia MAS GRANDE que el
        # real y no era un control. Aqui se reetiqueta el vocabulario REAL con nombres
        # nuevos: misma dimension, misma frecuencia de documento, misma categoria, misma
        # matriz de dispersidad, y ninguna semantica. Es el control lo mas fuerte
        # posible para la pregunta "la mejora es dimensionalidad o es estructura".
        # ---------- el control, y por que NO puede ser una reetiquetacion ----------
        # La primera version emparejó el control reetiquetando el vocabulario real con
        # nombres nuevos. Daba EXACTAMENTE el mismo resultado que la pista, y no era un
        # fallo: permutar columnas de la matriz de diseño es una SIMETRIA de un modelo
        # lineal regularizado, porque la penalizacion es la norma al cuadrado del vector de
        # coeficientes, que es invariante bajo permutacion. Luego un control reetiquetado
        # es un no-op demostrable y no puede discriminar nada.
        #
        # Esa invariancia es en si misma una comprobacion util de la instrumentacion, y por
        # eso se mide de forma explicita mas abajo. Para que el control DISCRIMINE tiene que
        # ser INDEPENDIENTE del canal real y del mismo tamaño, y la forma de conseguirlo es
        # buscar el tamano del pool aleatorio que, tras min_df=2, da la dimension del canal
        # real. La busqueda es sobre el tamano, no sobre hiperparametros del modelo, y es
        # local de cada pliegue.
        objetivo_dim = xpos.shape[1]
        rng_c = np.random.default_rng(SEMILLA + f)
        mejor: tuple[int, Any, Any] | None = None
        for pool in (objetivo_dim, int(objetivo_dim * 0.6), int(objetivo_dim * 0.4),
                     int(objetivo_dim * 1.6), int(objetivo_dim * 2.5), 200, 50, 12):
            pc_c = TfidfVectorizer(**PARAMS_CANAL)
            xa = pc_c.fit_transform(tokens_word_pos(documentos, ir, rng_c, max(pool, 2)))
            if xa.shape[1] == 0:
                continue
            if mejor is None or abs(xa.shape[1] - objetivo_dim) < abs(mejor[0] - objetivo_dim):
                rng_c = np.random.default_rng(SEMILLA + f)
                mejor = (xa.shape[1], pool,
                         (pc_c, tokens_word_pos(documentos, ir, rng_c, max(pool, 2)),
                          tokens_word_pos(documentos, iva, rng_c, max(pool, 2))))
            if xa.shape[1] == objetivo_dim:
                break
        assert mejor is not None
        dim_c, pool_c, (pc_c, tok_ca, tok_cv) = mejor
        xrand = pc_c.fit_transform(tok_ca)
        xrand_v = pc_c.transform(tok_cv)
        log(f"        control aleatorio: pool {pool_c} -> {dim_c} rasgos, "
            f"el canal real tiene {objetivo_dim}")

        # Control del confound de escala: se iguala la norma de fila al objetivo del
        # control, de modo que los brazos difieren en una DIRECCION y no en una ESCALA.
        objetivo = float(np.sqrt(np.mean(xbase.multiply(xbase).sum(axis=1))))

        def escalar(m: csr_matrix, ref: float) -> csr_matrix:
            """Iguala la norma media de fila a la de referencia.

            Args:
                m: matriz dispersa.
                ref: norma objetivo.

            Returns:
                La matriz escalada.
            """
            norma = np.sqrt(np.asarray(m.multiply(m).sum(axis=1)).ravel())
            media = float(norma.mean())
            return m if media <= 0 else m.multiply(ref / media)

        x_l = escalar(hstack([xbase, xpos]).tocsr(), objetivo)
        x_l_v = escalar(hstack([xbase_v, xpos_v]).tocsr(), objetivo)
        x_r = escalar(hstack([xbase, xrand]).tocsr(), objetivo)
        x_r_v = escalar(hstack([xbase_v, xrand_v]).tocsr(), objetivo)

        for clave, xent, xval in (("control", xbase, xbase_v),
                                  ("pista_L", x_l, x_l_v),
                                  ("control_aleatorio", x_r, x_r_v)):
            for j in range(3):
                m = LogisticRegression(**LR)
                m.fit(xent, y[ir, j])
                proba[clave][iva, j] = m.predict_proba(xval)[:, 1]
            dims[clave].append(int(xent.shape[1]))
            densidad[clave].append(round(float(xent.nnz / (xent.shape[0] * xent.shape[1])),
                                          6))
        log(f"      pliegue {f}: base {xbase.shape[1]} | +L {x_l.shape[1]} "
            f"(canal {xpos.shape[1]}) | +aleatorio {x_r.shape[1]} "
            f"| norma de fila objetivo {objetivo:.4f} ({time.time() - t0:.0f} s)")

    # ---------- evaluacion ----------
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
        resc = int(np.count_nonzero(ac & (pred_v03 != y)))
        dan = int(np.count_nonzero((~ac) & (pred_v03 == y)))
        folds = [float(f1_score(y[pliegues == g], pr[pliegues == g], average="macro"))
                 for g in range(1, 6)]
        por_et = {}
        for j, et in enumerate(ETIQUETAS):
            a = ac[:, j]
            por_et[et] = {
                "F1": round(float(f1_score(y[:, j], pr[:, j])), 6),
                "rescate": int(np.count_nonzero(a & (pred_v03[:, j] != y[:, j]))),
                "damage": int(np.count_nonzero((~a) & (pred_v03[:, j] == y[:, j]))),
                "neto": int(np.count_nonzero(a & (pred_v03[:, j] != y[:, j]))
                           - np.count_nonzero((~a) & (pred_v03[:, j] == y[:, j]))),
                "errores_de_alta_confianza": int(np.count_nonzero(
                    ~a & (np.abs(p[:, j] - V03_UMBRALES[j]) >= DELTA_HC))),
                "celdas_modificadas": int(np.count_nonzero(pr[:, j] != pred_v03[:, j])),
            }
        return {
            "corrected_cv": float(f1_score(y, pr, average="macro")),
            "folds": folds,
            "rescate": resc, "damage": dan, "neto": resc - dan,
            "cambian": int(np.count_nonzero(pr != pred_v03)),
            "por_etiqueta": por_et,
            "desglose": {
                "ambos_bien": int(np.count_nonzero(ac & (pred_v03 == y))),
                "ambos_mal": int(np.count_nonzero((~ac) & (pred_v03 != y))),
                "v03_bien_candidato_mal": int(np.count_nonzero((~ac) & (pred_v03 == y))),
                "v03_mal_candidato_bien": int(np.count_nonzero(ac & (pred_v03 != y))),
            },
        }

    mc, ml, mr = metricas("control"), metricas("pista_L"), metricas("control_aleatorio")
    log("")
    log("   LOS TRES BRAZOS")
    log(f"      control, V03 sin canal     {mc['corrected_cv']:.10f}  "
        f"(debe ser {V03_CCV!r}, coincide {mc['corrected_cv'] == V03_CCV})")
    log(f"      pista L, + palabra__POS    {ml['corrected_cv']:.10f}  "
        f"delta {ml['corrected_cv'] - ccv_ref:+.10f}")
    log(f"      control aleatorio, misma   {mr['corrected_cv']:.10f}  "
        f"dimension {mr['corrected_cv'] - ccv_ref:+.10f}")
    if mc["corrected_cv"] != V03_CCV:
        raise SystemExit(
            f"PARADA: el control no reproduce V03, da {mc['corrected_cv']!r}. Sin eso "
            f"ningun delta de este script es interpretable")
    log("      el control reproduce V03 exactamente, luego los deltas son atribuibles")

    log("")
    log("   EL CONTROL DE DIMALIDAD, que es el que decide si la senal significa algo")
    aporte_real = ml["corrected_cv"] - ccv_ref
    aporte_aleatorio = mr["corrected_cv"] - ccv_ref
    log(f"      aporte de la pista L frente a V03            {aporte_real:+.10f}")
    log(f"      aporte de un bloque aleatorio de igual tamano  {aporte_aleatorio:+.10f}")
    log(f"      diferencia L menos aleatorio                   "
        f"{aporte_real - aporte_aleatorio:+.10f}")
    # La lectura tiene TRES partes y no conviene reducirla a un signo. El canal real y el
    # aleatorio se restan, y lo que sale es el valor de la estructura linguistica; despues
    # se compara ese valor con el coste de haber anadido el bloque.
    valor_estructura = aporte_real - aporte_aleatorio
    coste = -aporte_aleatorio
    if aporte_real <= 0 and valor_estructura > 0:
        fraccion = valor_estructura / coste if coste > 0 else float("inf")
        veredicto_control = (
            f"lectura en tres partes. (1) Anadir un bloque de {dims['pista_L'][0] - dims['control'][0]} "
            f"rasgos CUESTA {coste:+.10f} de macro-F1, y ese coste lo paga incluso un bloque "
            f"sin significado linguistico. (2) La estructura linguistica REAL vale "
            f"{valor_estructura:+.10f} respecto a ese bloque aleatorio, luego la interaccion "
            f"palabra-POS existe y es medible. (3) Pero repaga solo el "
            f"{fraccion * 100:.1f} por ciento del coste, de modo que el saldo es "
            f"{aporte_real:+.10f}. La senal existe y es pequena frente al precio de pagarla")
    elif aporte_real <= 0:
        veredicto_control = (
            f"la pista L no mejora ({aporte_real:+.10f}) y ademas no supera al bloque "
            f"aleatorio ({valor_estructura:+.10f}), luego la interaccion palabra-POS no "
            f"aporta nada medible")
    else:
        veredicto_control = (
            f"la pista L mejora {aporte_real:+.10f} frente a "
            f"{aporte_aleatorio:+.10f} del bloque aleatorio, luego el efecto no es "
            f"dimensionalidad")
    lectura_tres_partes = {
        "coste_de_anadir_un_bloque": round(coste, 10),
        "valor_de_la_estructura_linguistica": round(valor_estructura, 10),
        "fraccion_repagada": round(valor_estructura / coste, 6) if coste > 0 else None,
        "saldo": round(aporte_real, 10),
    }
    log(f"      VEREDICTO: {veredicto_control}")

    # ---------- probabilidad y error de alta confianza ----------
    log("")
    log("   SECCIONES 37 Y 57, PROBABILIDAD, DISTANCIA AL UMBRAL Y ERROR DE ALTA CONFIANZA")
    analisis: dict[str, Any] = {}
    for clave in ("pista_L", "control_aleatorio"):
        p = proba[clave]
        pr = (p >= V03_UMBRALES[None, :]).astype(int)
        d_v03 = np.abs(proba_v03 - V03_UMBRALES[None, :])
        d_cand = np.abs(p - V03_UMBRALES[None, :])
        hc = {}
        for j, et in enumerate(ETIQUETAS):
            err_v = (pred_v03[:, j] != y[:, j])
            err_c = (pr[:, j] != y[:, j])
            # correccion de errores de alta confianza: V03 se equivoca con confianza y el
            # candidato acierta. Es la unica clase de mejora que el parrafo 57 llama
            # evidencia de nueva representacion.
            corr_hc = int(np.count_nonzero(err_v & ~err_c & (d_v03[:, j] >= DELTA_HC)))
            dano_hc = int(np.count_nonzero(~err_v & err_c & (d_cand[:, j] >= DELTA_HC)))
            hc[et] = {
                "errores_V03": int(err_v.sum()),
                "errores_candidato": int(err_c.sum()),
                "alta_confianza_corregidos": corr_hc,
                "alta_confianza_introducidos": dano_hc,
                "neto_alta_confianza": corr_hc - dano_hc,
                "errores_bajos_confianza_corregidos": int(np.count_nonzero(
                    err_v & ~err_c & (d_v03[:, j] < DELTA_HC))),
                "desplazamiento_medio_de_probabilidad": round(
                    float(np.abs(p[:, j] - proba_v03[:, j]).mean()), 6),
                "distancia_media_al_umbral_V03": round(float(d_v03[:, j].mean()), 6),
                "distancia_media_al_umbral_candidato": round(float(d_cand[:, j].mean()), 6),
            }
        analisis[clave] = hc
        log("")
        log(f"      {clave}:")
        for et in ETIQUETAS:
            h = hc[et]
            log(f"        {et}: errores V03 {h['errores_V03']:4d} -> candidato "
                f"{h['errores_candidato']:4d} | alta confianza corregidos "
                f"{h['alta_confianza_corregidos']:4d} | introducidos "
                f"{h['alta_confianza_introducidos']:4d} | neto "
                f"{h['neto_alta_confianza']:+4d}")
            log(f"            desplazamiento medio de probabilidad "
                f"{h['desplazamiento_medio_de_probabilidad']:.6f} | distancia al umbral "
                f"{h['distancia_media_al_umbral_V03']:.6f} -> "
                f"{h['distancia_media_al_umbral_candidato']:.6f}")

    # ---------- gate ----------
    folds_v03 = [float(f1_score(y[pliegues == g], pred_v03[pliegues == g], average="macro"))
                 for g in range(1, 6)]
    folds_mej = sum(1 for a, b in zip(ml["folds"], folds_v03) if a > b)
    delta = ml["corrected_cv"] - ccv_ref
    h_delta = delta >= GATE_DELTA
    h_folds = folds_mej >= GATE_PLIEGUES
    gate = bool(h_delta and h_folds)
    log("")
    log("   GATE")
    log(f"      delta {delta:+.10f} >= {GATE_DELTA}: {h_delta} | pliegues {folds_mej}/5 "
        f">= {GATE_PLIEGUES}: {h_folds} | superado: {gate}")
    log(f"      rescate {ml['rescate']} | dano {ml['damage']} | neto {ml['neto']:+d} | "
        f"cambian {ml['cambian']}")
    log(f"      desglose: {ml['desglose']}")
    ic_lo = ic_hi = None
    if gate:
        rng = np.random.default_rng(SEMILLA)
        pr = (proba["pista_L"] >= V03_UMBRALES[None, :]).astype(int)
        dc = ((pr == y).astype(int) - (pred_v03 == y).astype(int)).sum(axis=1).astype(float)
        g_unicos, g_cod = np.unique(grupos, return_inverse=True)
        dif_g = np.bincount(g_cod, weights=dc, minlength=len(g_unicos))
        idx = rng.integers(0, len(g_unicos), size=(N_BOOT, len(g_unicos)))
        ms = dif_g[idx].sum(axis=1) / len(g_unicos)
        ic_lo, ic_hi = float(np.percentile(ms, 2.5)), float(np.percentile(ms, 97.5))
        log(f"      bootstrap agrupado por los {len(g_unicos)} grupos canonicos: "
            f"IC95 [{ic_lo:+.6f}, {ic_hi:+.6f}]")

    np.savez_compressed(CELDA_NPZ, ids=ids, y_true=y, pliegues=pliegues,
                        pred_v03=pred_v03, pred_l=proba["pista_L"] >= V03_UMBRALES,
                        proba_l=proba["pista_L"], proba_aleatorio=proba["control_aleatorio"],
                        proba_v03=proba_v03)

    decision = ("PROMISING. Supera el gate; exige segunda particion" if gate else
                ("MARGINAL. Mejora pero no alcanza el gate de +0,003" if delta > 0 else
                 "CLOSED. La interaccion palabra-POS no aporta sobre V03"))
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                            text=True, cwd=PROJECT_ROOT, check=False).stdout.strip()
    informe = {
        "familia": "P62_TRACK_L",
        "naturaleza": ("EXPERIMENTO. 1 candidato, 1 control de dimensionalidad, 0 "
                       "submissions, 0 Kaggle, 0 holdout, 0 test. V03 no se modifica"),
        "fecha": inicio.isoformat(timespec="seconds"),
        "experiment_id": "P62-01", "track": "L-word-POS",
        "script": "experiments/p62_03_pista_l.py",
        "hipotesis": ("el mismo lexema tiene distinto significado predictivo segun su "
                      "categoria gramatical, y el par palabra__POS aporta informacion que "
                      "el TF-IDF de V03 no expresa"),
        "gap_check": {"veredicto": "genuinely_new",
                      "comprobacion": "ningun token palabra__POS se emite en el proyecto; el "
                                      "script de gap analysis aborta si aparece",
                      "cercanos": ["36 columnas densas pos_*", "canal categorial E2-03",
                                    "31 columnas densas dep_*"]},
        "protocolo": {
            "clasificador": "IDENTICO a V03: LogisticRegression C=2,0 balanced liblinear "
                            "max_iter=1000 random_state=42",
            "pliegues": "los canonicos, por grupo, sin cambio",
            "umbrales": "0,44 / 0,51 / 0,46 congelados, aplicados a los tres brazos",
            "canal": f"palabra__POS sobre {len(CATEGORIAS_CANAL)} categorias, inducido DENTRO "
                     f"de cada pliegue con min_df=2, ngram_range (1,1)",
            "confound_de_escala": ("cada bloque TF-IDF se normaliza por separado, luego anadir "
                                   "un tercero cambia la norma de fila, que es lo mismo que "
                                   "cambiar C. La matriz candidata se reescala para que su "
                                   "norma media de fila sea la del control, con lo que los "
                                   "brazos difieren en direccion y no en escala. En el control "
                                   "el factor es exactamente uno"),
            "control_de_dimensionalidad": ("un tercer brazo anade un bloque de tokens "
                                           "ALEATORIOS del mismo numero y con las mismas "
                                           "categorias, que mide cuanto de cualquier mejora "
                                           "es dimensionalidad en vez de estructura"),
            "rejilla": "ninguna",
        },
        "control_de_vocabulario": {
            "vocabulario_por_brazo": {"control": dims["control"], "pista_L": dims["pista_L"],
                                      "control_aleatorio": dims["control_aleatorio"]},
            "densidad_no_nulos": densidad,
            "min_df": 2, "n_estimators": "no aplica",
            "segundos": round(time.time() - t0, 1),
        },
        "resultados": {"control": mc, "pista_L": ml, "control_aleatorio": mr},
        "aportes": {"pista_L": round(aporte_real, 10),
                    "control_aleatorio": round(aporte_aleatorio, 10),
                    "diferencia_L_menos_aleatorio": round(aporte_real - aporte_aleatorio, 10)},
        "veredicto_del_control_de_dimensionalidad": veredicto_control,
        "lectura_en_tres_partes": lectura_tres_partes,
        "nota_de_instrumentacion": (
            "un control que reetiqueta el vocabulario real con nombres nuevos produce "
            "EXACTAMENTE el mismo resultado, y no es un fallo: permutar columnas de la "
            "matriz de diseno es una simetria de un modelo lineal regularizado, porque la "
            "penalizacion es la norma al cuadrado del vector de coeficientes. Ese control "
            "es un no-op demostrable, luego el control que si discrimina tiene que ser "
            "independiente del canal real y del mismo tamano, que es lo que hace la "
            "version vigente"),
        "analisis_de_probabilidad": analisis,
        "gate": {"delta_min": GATE_DELTA, "pliegues_min": GATE_PLIEGUES,
                 "delta": round(delta, 10), "pliegues_mejorados": folds_mej,
                 "supera_delta": bool(h_delta), "supera_pliegues": bool(h_folds),
                 "superado": gate},
        "bootstrap": (None if ic_lo is None else
                      {"IC95": [round(ic_lo, 6), round(ic_hi, 6)], "n": N_BOOT,
                       "agrupado_por": "grupo canonico"}),
        "reproducibilidad": {
            "nivel": "R2", "git_commit": commit, "seed": SEMILLA,
            "min_df": 2, "annotacion": "features/p62_anotacion_spacy.json.gz",
            "tagger": "spaCy es_core_news_md 3.8.0, exclude=['ner'], autorizado por D-003",
            "nota": ("el canal se induce dentro de cada pliegue a partir de las filas de "
                     "entrenamiento, luego ningun rasgo del vocabulario ve las filas "
                     "validadas"),
        },
        "decision": decision,
        "experimentos_registrados": 1,
        "formulacion": "V03 remains the best verified leaderboard result.",
    }
    JSON_L.write_text(json.dumps(informe, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")

    L: list[str] = []
    W = L.append
    W("# P62 - Pista L, tokens condicionados por POS")
    W("")
    W(f"Generado por `experiments/p62_03_pista_l.py`. Fecha {inicio.isoformat(timespec='seconds')}.")
    W("")
    W("## Que decia la EDA")
    W("")
    q = eda["pregunta_central_seccion_24"]
    W("```text")
    W(f"palabra sola              {q['solo_palabra']:+.4f}")
    W(f"categoria sola            {q['solo_categoria']:+.4f}")
    W(f"palabra y categoria       {q['palabra_mas_categoria']:+.4f}")
    W(f"synergy                   {q['sinergia']:+.4f}  "
      f"({q['sinergia_sobre_palabra'] * 100:.2f} por ciento de la senal de la palabra)")
    W("```")
    W("")
    W("La synergy es la parte que no se puede escribir como funcion de la palabra y de la")
    W("categoria por separado. V03 sabe la palabra y tira la categoria; el canal de E2-03")
    W("sabe la categoria y tira la palabra. Esta pista le da el par.")
    W("")
    W("## El hueco, comprobado contra el codigo")
    W("")
    W("`p62_00_auditoria_y_gap.py` verifica contra el fuente que ningun token `palabra__POS`")
    W("se emite en el proyecto, y ABORTA si aparece. No es una busqueda de nombres.")
    W("")
    W("## El confound de escala, que es real")
    W("")
    W("Cada bloque TF-IDF se normaliza por separado, luego V03 tiene norma de fila raiz de 2")
    W("y anadir un tercer bloque la sube. Eso es un reescalado uniforme de los")
    W("coeficientes, que equivale a cambiar C, y el parrafo 36 lo prohibe. La matriz")
    W("candidata se reescala para que su norma media de fila sea la del control. En el")
    W("control el factor es exactamente uno, luego el control es V03 intacto.")
    W("")
    W("## Los tres brazos")
    W("")
    W("| brazo | corrected_cv | delta vs V03 |")
    W("|---|---:|---:|")
    W(f"| control, V03 sin canal | {mc['corrected_cv']:.10f} | +0.0000000000 |")
    W(f"| pista L, + palabra__POS | {ml['corrected_cv']:.10f} | {aporte_real:+.10f} |")
    W(f"| control aleatorio, igual dimension | {mr['corrected_cv']:.10f} | "
      f"{aporte_aleatorio:+.10f} |")
    W("")
    W(f"El control reproduce V03 exactamente: {mc['corrected_cv'] == V03_CCV}.")
    W("")
    W("## El control de dimensionalidad, que es el que decide")
    W("")
    W("Un candidato que anade rasgos puede ganar por una razon que no tiene nada que ver con")
    W("linguistica: mas dimensiones, mas dispersidad, otra regularizacion efectiva. Por eso")
    W("el tercer brazo anade un bloque de tokens ALEATORIOS del mismo numero y con las")
    W("mismas categorias.")
    W("")
    W("```text")
    W(f"aporte de la pista L                    {aporte_real:+.10f}")
    W(f"aporte del bloque aleatorio             {aporte_aleatorio:+.10f}")
    W(f"diferencia                              {valor_estructura:+.10f}")
    W("```")
    W("")
    W("### La lectura en tres partes, que conviene no reducir a un signo")
    W("")
    W("```text")
    W(f"(1) coste de anadir un bloque de rasgos   {coste:+.10f}")
    W(f"(2) valor de la estructura linguistica    {valor_estructura:+.10f}")
    W(f"(3) fraccion repagada                     "
      f"{(valor_estructura / coste * 100 if coste > 0 else 0):.1f} por ciento")
    W(f"    saldo                                 {aporte_real:+.10f}")
    W("```")
    W("")
    W(veredicto_control)
    W("")
    W("## Control de vocabulario")
    W("")
    W("| brazo | rasgos por pliegue | densidad de no nulos |")
    W("|---|---|---|")
    for clave in ("control", "pista_L", "control_aleatorio"):
        W(f"| {clave} | {dims[clave]} | {densidad[clave]} |")
    W("")
    W("El canal usa min_df=2 y se induce DENTRO de cada pliegue, luego ningun rasgo de su")
    W("vocabulario ve las filas validadas.")
    W("")
    W("## Probabilidad, umbral y error de alta confianza")
    W("")
    W("| et | errores V03 | errores candidato | alta conf. corregidos | introducidos | neto |")
    W("|---|---:|---:|---:|---:|---:|")
    for et in ETIQUETAS:
        h = analisis["pista_L"][et]
        W(f"| {et} | {h['errores_V03']} | {h['errores_candidato']} | "
          f"{h['alta_confianza_corregidos']} | {h['alta_confianza_introducidos']} | "
          f"{h['neto_alta_confianza']:+d} |")
    W("")
    W("Una correccion que solo ocurra en errores de BAJA confianza seria un efecto de")
    W("decision y no evidencia de nueva representacion (parrafo 57). Por eso la tabla")
    W("separa las dos clases.")
    W("")
    W("## Gate")
    W("")
    W("```text")
    W(f"delta {delta:+.6f} >= {GATE_DELTA}: {h_delta}")
    W(f"pliegues que mejoran {folds_mej}/5 >= {GATE_PLIEGUES}: {h_folds}")
    W(f"superado: {gate}")
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
    W("*Generado por `experiments/p62_03_pista_l.py`. Sin push.*")
    MD_L.write_text(a_ascii("\n".join(L)), encoding="ascii")

    for destino in (MD_L, JSON_L, CSV_GAP_LING):
        if any(b > 127 for b in destino.read_bytes()):
            raise SystemExit(f"PARADA: {destino.name} conserva bytes no ASCII")
    log("")
    log(f"P62-03 OK | pista L {ml['corrected_cv']:.10f} delta {delta:+.10f} | control "
        f"aleatorio {mr['corrected_cv']:.10f} | diferencia "
        f"{aporte_real - aporte_aleatorio:+.10f} | gate {gate}")


if __name__ == "__main__":
    ast.parse(Path(__file__).read_text(encoding="utf-8"))
    main()
