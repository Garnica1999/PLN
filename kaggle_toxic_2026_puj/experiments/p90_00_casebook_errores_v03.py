"""P90-00: casebook de errores de V03 sobre el desarrollo (analisis descriptivo).

Ciclo descriptivo: NO entrena modelos, NO ajusta umbrales y NO reabre familias.
Inventaria TODOS los errores de V03 sobre las 1483 filas de desarrollo usando el
OOF canonico (`features/oof_v03_final.npz`), recupera los textos originales por
identificador y produce los catalogos, resumenes y muestras de los entregables.

Definiciones conservadas del proyecto (no se redefinen):
- Patrones de superficie de `experiments/06_error_analysis.py` (URL, hashtag,
  mencion, mencion inicial, elongacion, mayusculas, risa).
- Lexico inducido de agresividad de la Fase 2
  (`reports/eda/fase2_lexico_inducido_agresividad.csv`, in-sample y diagnostico)
  aplicado sobre lemas de spaCy como en el 06.
- Categorias de la taxonomia v4 de `experiments/35_discovery_v4.py`.
- Grupo `coloquial` de `experiments/embedding_01_corpus_wikipedia_v5.py`
  (patron por token, aplicado como presencia en el tweet; adaptacion registrada).
- Alta confianza historica de `experiments/p61_01_eda_probabilidades.py`:
  distancia al umbral >= 0.10 / 0.20 / 0.30.

Genera los ocho entregables en `reports/` y el JSON de provenance. La seccion
cualitativa se completa tras la lectura de la muestra (mismo script, lista de
observaciones con ids y contraejemplos).

Uso:
    /home/carlos/venv/bin/python experiments/p90_00_casebook_errores_v03.py
"""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import re
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Final

import numpy as np

RAIZ: Final[Path] = Path(__file__).resolve().parents[1]

RUTA_OOF: Final[Path] = RAIZ / "features" / "oof_v03_final.npz"
RUTA_TRAIN: Final[Path] = RAIZ / "data" / "raw" / "toxic_train.csv"
RUTA_LEXICO: Final[Path] = RAIZ / "reports" / "eda" / "fase2_lexico_inducido_agresividad.csv"
RUTA_CASEBOOK_MD: Final[Path] = RAIZ / "reports" / "v03_error_casebook.md"
RUTA_CASEBOOK_CSV: Final[Path] = RAIZ / "reports" / "v03_error_casebook.csv"
RUTA_CELLS_CSV: Final[Path] = RAIZ / "reports" / "v03_error_cells.csv"
RUTA_FOLD_CSV: Final[Path] = RAIZ / "reports" / "v03_error_fold_summary.csv"
RUTA_SLICE_CSV: Final[Path] = RAIZ / "reports" / "v03_error_slice_summary.csv"
RUTA_CONF_CSV: Final[Path] = RAIZ / "reports" / "v03_error_confidence_summary.csv"
RUTA_SAMPLE_CSV: Final[Path] = RAIZ / "reports" / "v03_error_review_sample.csv"
RUTA_CONTROLS_CSV: Final[Path] = RAIZ / "reports" / "v03_error_controls.csv"
RUTA_PROV: Final[Path] = RAIZ / "reports" / "v03_error_provenance.json"

HASH_OOF: Final[str] = "559265816ce4f6312c87f008f90324f0eec70a74d1e6247364c162793338feab"
HASH_TRAIN: Final[str] = "3d40e7345b588860cf4744914be88ddc094e4f03591516c8176ab40b7f76d42f"
V03_CCV: Final[float] = 0.7772114525165721
UMBRALES: Final[tuple[float, float, float]] = (0.44, 0.51, 0.46)
ETIQUETAS: Final[tuple[str, str, str]] = ("HS", "TR", "AG")
SEMILLA_CASOS: Final[int] = 20261010
MAX_MUESTRA: Final[int] = 100
BINS: Final[tuple[tuple[float, float, str], ...]] = (
    (0.000, 0.025, "[0.000,0.025)"),
    (0.025, 0.050, "[0.025,0.050)"),
    (0.050, 0.100, "[0.050,0.100)"),
    (0.100, 0.200, "[0.100,0.200)"),
    (0.200, float("inf"), "[0.200,inf)"),
)
DELTAS_HC: Final[tuple[float, ...]] = (0.10, 0.20, 0.30)
Z_NUCLEO: Final[float] = 2.5

# Patrones de superficie, verbatim de experiments/06_error_analysis.py
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
MENTION_PATTERN = re.compile(r"@\w+")
HASHTAG_PATTERN = re.compile(r"#\w+")
INITIAL_MENTION_PATTERN = re.compile(r"^\s*(?:rt\s+)?@\w+", re.IGNORECASE)
UPPER_WORD_PATTERN = re.compile(r"\b[A-Z]{2,}\b")
ELONGATION_PATTERN = re.compile(r"([A-Za-z])\1{2,}")

# Patron del grupo coloquial, verbatim de experiments/embedding_01_corpus_wikipedia_v5.py
COLOQUIAL_PATTERN = re.compile(
    r"^(?:jaja\w*|jeje\w*|xd|xdd|xdr|lol|jejej|plis|porfa|xfa|q|qe|xq|pa|pa'|k|tmb|aw|uff|uy|eh|mas|bien|vale|nah)$"
)
TOKEN_PATTERN = re.compile(r"\w+", re.UNICODE)

# Lexicons de superficie del 06 (para las rebanadas spaCy)
SECOND_PERSON_LEXICON: Final[frozenset[str]] = frozenset(
    {"tu", "tus", "te", "ti", "usted", "ustedes", "vos", "vosotros", "vosotras",
     "contigo", "os", "vuestro", "vuestra", "vuestros", "vuestras", "tuyo", "tuya",
     "tuyos", "tuyas"}
)
COLLECTIVE_QUANTIFIERS: Final[frozenset[str]] = frozenset(
    {"todos", "todas", "todo", "toda", "muchos", "muchas", "algunos", "algunas",
     "ningun", "ninguna", "ningunos", "ningunas", "cualquiera", "cualesquiera",
     "varios", "varias", "cada", "ambos", "ambas", "gente", "pueblo", "mayoria",
     "minoria", "multitud", "monton", "masa", "grupo"}
)
IMPERATIVE_LEXICON: Final[frozenset[str]] = frozenset(
    {"calla", "callate", "callese", "callense", "callete", "vete", "vayase", "vayan",
     "vayanse", "muerete", "muere", "largate", "largese", "deja", "dejen", "dejame",
     "dejalo", "mira", "miren", "miralo", "oye", "oiga", "oigan", "escucha",
     "escuchen", "aprende", "aprendan", "habla", "hablen", "anda", "andate", "sigue",
     "sigan", "ven", "veni", "venga", "haz", "haga", "pon", "ponte", "pongase",
     "quita", "quitate", "basta", "cierra", "abre", "dime", "digame", "cuentame",
     "lee", "estudia", "trabaja", "trabaje", "respeta", "respeten", "piensa",
     "piense", "corre", "sal", "salte", "espera", "esperen", "aguanta", "tomate",
     "cojete", "chupame", "chupala", "matate", "ahorcate", "pide", "pida", "compra",
     "vende", "danos", "iros", "hacelo", "retirense", "marchate", "calmate",
     "callad", "mirad", "ponganse", "vamos", "huye", "corran", "escondete", "largo",
     "fuera"}
)
CLITIC_ENDINGS: Final[tuple[str, ...]] = ("me", "te", "se", "lo", "la", "le", "nos", "os", "los", "las", "les")
INFINITIVE_CLITIC_PATTERN = re.compile(r"r(me|te|se|lo|la|le|nos|os|los|las|les)$")
FIRST_PERSON_PLURAL_PATTERN = re.compile(r"(mos|amos|emos|imos)$")

# Observaciones cualitativas (lectura de la muestra y de los errores mas alejados).
OBSERVACIONES: Final[tuple[dict[str, str], ...]] = (
    {
        "tipo": "OBSERVED_PATTERN",
        "patron": "una parte de los FP de HS/TR/AG son insultos con marcadores de broma o afiliacion",
        "casos": "22628 (AG_FP, 'T eres bien perra'); 23076 (HS_FP, 'Cllate todo te queda bien zorra te amo'); 22524 (AG_FP, 'Jeje callate perra'); 20278 (HS_FP|TR_FP|AG_FP, 'jajajajajaj soy imbecil... yo + <3')",
        "contraejemplo": "21892 (HS_FP|TR_FP|AG_FP, 'Callate puta tu eres el traidor.', sin marcadores y con p>=0.93)",
        "lectura": "el patron de broma/afiliacion (documentado en el 06) cubre 27/173 FP_HS, 26/135 FP_AG y 21/82 FP_TR: es una parte, no la explicacion completa",
    },
    {
        "tipo": "OBSERVED_PATTERN",
        "patron": "los FN de AG y HS incluyen hostilidad contra grupos expresada sin lexico de insulto directo",
        "casos": "20331 (AG_FN, 'Os acordais... inmigrantes subsaharianos...'); 20337 (AG_FN, '#Sudacas'); 20856 (AG_FN, 'El problema... los inmigrantes que llegan masivamente...'); 20678 (HS_FN|AG_FN, 'Mierda para aquellos que saltan la valla de Melilla y Ceuta. Cabrones...')",
        "contraejemplo": "24505 (control, 'No mereces ms comunistas tampoco. Son una escoria para el pais.' clasificado correctamente)",
        "lectura": "consistente con el hallazgo historico del 06 (los FN de AG se concentran en agresion implicita contra grupos); no se asigna causa",
    },
    {
        "tipo": "TENTATIVE_PATTERN",
        "patron": "FN de las TRES etiquetas en agresion directa de registro sexual o vulgar que el modelo puntua muy bajo",
        "casos": "20932 (p=0.28/0.32/0.31, 'Gorda chaconda.. tu eres por mi polla la mujer perfecta...'); 22065 (p=0.32/0.28/0.32, 'lo nico que dominas es mi polla en tu cara.'); 20693 (p baja, '...te mereces por puta y caliente...'); 20956 (p=0.31/0.45/0.30, '...vo te mereces unas chuchas')",
        "contraejemplo": "20609 y 21892: insultos del tipo 'callate puta' reciben p>0.9; la deteccion parece depender del fraseo, no solo del lexico",
        "lectura": "hipotesis a contrastar sobre dependencia de registro/fraseo; no se convierte en feature ni se afirma como causa",
    },
    {
        "tipo": "OBSERVED_PATTERN",
        "patron": "el error de TR vive en textos sin destinatario explicito (noticia o mencion a terceros)",
        "casos": "24269 (TR_FN, p=0.023, 'El agressor de una mujer de 25 ha sido detenido...'); 24241 (TR_FN, p=0.031, 'quisiera saber por qu tienen a este antisemita dando clase...')",
        "contraejemplo": "controles 20609 y 20347: respuestas directas con @ obtienen TR correcto con p alta",
        "lectura": "reproduce la lectura historica del 06 sobre TR; los FP de TR (82) incluyen 21 con marcadores de broma",
    },
    {
        "tipo": "OBSERVED_PATTERN",
        "patron": "citas y meta-discurso del insulto entre los errores",
        "casos": "21362 (HS_FP|AG_FP, \"'Cllate Zorra' el 'cllate perro' de este semestre\"); 20707 (AG_FN, texto entre comillas)",
        "contraejemplo": "la categoria QUOTED v4 tiene solo 16 errores: el meta-discurso existe pero es minoritario",
        "lectura": "observado, sin cuantificar como causa",
    },
    {
        "tipo": "TENTATIVE_PATTERN",
        "patron": "una parte de los errores esta pegada al umbral y no debe interpretarse como anotacion",
        "casos": "23839 (d=0.000046); 23290 (d=0.00078); 23734 (d=0.00103); 20278 (d=0.00119)",
        "contraejemplo": "112 errores de HS, 54 de TR y 77 de AG estan a distancia >= 0.10 (grupo C), no fronterizos",
        "lectura": "separar frontera (A/B) de lejania (C) es imprescindible antes de proponer hipotesis; la distancia no es una probabilidad calibrada",
    },
    {
        "tipo": "UNSUPPORTED_HYPOTHESIS",
        "patron": "afirmar que los FP/FN alejados son errores de anotacion, o que el sarcasmo explica los FN",
        "casos": "sin evidencia textual suficiente recolectada en este ciclo",
        "contraejemplo": "los mismos casos admiten lecturas de representacion y de anotacion; hace falta revision humana y un mecanismo legal no medido",
        "lectura": "se registra como no apoyada; no se usa para abrir ninguna via",
    },
)


def log(mensaje: str) -> None:
    """Imprime una linea de progreso."""
    print(f"[P90-00] {mensaje}", flush=True)


def sha256_de(ruta: Path) -> str:
    """Calcula el sha256 de un archivo."""
    digest = hashlib.sha256()
    with ruta.open("rb") as archivo:
        for bloque in iter(lambda: archivo.read(1 << 20), b""):
            digest.update(bloque)
    return digest.hexdigest()


def f1_binaria(y: np.ndarray, pred: np.ndarray) -> float:
    """F1 binaria de la clase positiva."""
    tp = int(np.sum((y == 1) & (pred == 1)))
    fp = int(np.sum((y == 0) & (pred == 1)))
    fn = int(np.sum((y == 1) & (pred == 0)))
    if tp == 0:
        return 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def confusion(y: np.ndarray, pred: np.ndarray) -> dict[str, Any]:
    """Matriz de confusion binaria con sus tasas."""
    tp = int(np.sum((y == 1) & (pred == 1)))
    fp = int(np.sum((y == 0) & (pred == 1)))
    fn = int(np.sum((y == 1) & (pred == 0)))
    tn = int(np.sum((y == 0) & (pred == 0)))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    n = tp + fp + fn + tn
    return {
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "precision": round(precision, 6), "recall": round(recall, 6), "F1": round(f1, 6),
        "prevalencia_real": round((tp + fn) / n, 6) if n else 0.0,
        "prevalencia_predicha": round((tp + fp) / n, 6) if n else 0.0,
        "errores": fp + fn,
        "error_rate": round((fp + fn) / n, 6) if n else 0.0,
        "FP_count": fp, "FN_count": fn, "n": n,
    }


def bin_de(distancia: float) -> str:
    """Intervalo predefinido de distancia al umbral."""
    for bajo, alto, nombre in BINS:
        if bajo <= distancia < alto:
            return nombre
    return BINS[-1][2]


def cargar_gate() -> dict[str, Any]:
    """Carga el OOF y valida el baseline congelado.

    Returns:
        Datos verificados del desarrollo.

    Raises:
        SystemExit: si el OOF no reproduce el baseline.
    """
    hash_oof = sha256_de(RUTA_OOF)
    hash_train = sha256_de(RUTA_TRAIN)
    if hash_oof != HASH_OOF:
        raise SystemExit(f"STOP: el OOF no coincide con el hash canonico: {hash_oof}")
    if hash_train != HASH_TRAIN:
        raise SystemExit(f"STOP: toxic_train.csv no coincide: {hash_train}")
    oof = np.load(RUTA_OOF, allow_pickle=True)
    ids = oof["ids"].astype(np.int64)
    y = oof["y_true"].astype(np.int64)
    pliegues = oof["pliegues"].astype(np.int64)
    proba = oof["proba"].astype(np.float64)
    pred = oof["pred"].astype(np.int64)
    umbrales = np.asarray(oof["umbrales"], dtype=np.float64)
    etiquetas = [str(e) for e in oof["etiquetas"]]
    ccv = float(oof["corrected_cv"][0])
    if etiquetas != list(ETIQUETAS):
        raise SystemExit(f"STOP: orden de etiquetas inesperado: {etiquetas}")
    if tuple(umbrales.tolist()) != UMBRALES:
        raise SystemExit(f"STOP: umbrales inesperados: {umbrales.tolist()}")
    if ccv != V03_CCV:
        raise SystemExit(f"STOP: corrected_cv inesperado: {ccv!r}")
    if len(ids) != 1483 or len(set(int(i) for i in ids)) != 1483:
        raise SystemExit("STOP: el desarrollo no tiene 1483 ids unicos")
    if not np.array_equal(pred, (proba >= umbrales[None, :]).astype(np.int64)):
        raise SystemExit("STOP: pred no reproduce proba >= umbrales")
    f1s = [f1_binaria(y[:, j], pred[:, j]) for j in range(3)]
    ccv_reconstruido = float(np.mean(f1s))
    if abs(ccv_reconstruido - V03_CCV) > 1e-12:
        raise SystemExit(f"STOP: corrected_cv reconstruido {ccv_reconstruido!r}")
    f1_npz = [round(float(v), 6) for v in np.asarray(oof["f1_por_etiqueta"])]
    f1_recon = [round(v, 6) for v in f1s]
    if f1_recon != f1_npz:
        raise SystemExit(f"STOP: F1 por etiqueta no reproduce: {f1_recon} != {f1_npz}")
    folds = sorted(set(int(f) for f in pliegues))
    if folds != [1, 2, 3, 4, 5]:
        raise SystemExit(f"STOP: pliegues inesperados: {folds}")

    # Textos originales por identificador (solo filas del desarrollo).
    textos: dict[int, str] = {}
    with RUTA_TRAIN.open(encoding="utf-8") as fh:
        lector = csv.DictReader(fh, delimiter=";")
        for fila in lector:
            ident = int(str(fila["id"]).strip('"'))
            if ident in set(int(i) for i in ids):
                textos[ident] = fila["text"]
    if len(textos) != 1483:
        raise SystemExit(f"STOP: faltan textos: {len(textos)} de 1483")
    return {
        "ids": ids, "y": y, "pliegues": pliegues, "proba": proba, "pred": pred,
        "umbrales": umbrales, "etiquetas": etiquetas, "ccv": ccv, "f1s": f1s,
        "textos": textos, "hash_oof": hash_oof, "hash_train": hash_train,
        "fuente": str(oof["fuente"][0]), "nota_mapeo": str(oof["nota_mapeo"][0]),
    }


def cargar_lexico() -> tuple[frozenset[str], frozenset[str]]:
    """Carga el lexico inducido de agresividad y su nucleo (Fase 2)."""
    completo: set[str] = set()
    nucleo: set[str] = set()
    with RUTA_LEXICO.open(encoding="utf-8") as fh:
        for fila in csv.DictReader(fh):
            lema = str(fila["lema"]).strip()
            if str(fila["seleccionado"]).strip() == "1":
                completo.add(lema)
            if float(fila["z_score"]) >= Z_NUCLEO:
                nucleo.add(lema)
    return frozenset(completo), frozenset(nucleo)


def rasgos_spacy(textos: list[str], lexico: frozenset[str], nucleo: frozenset[str]) -> dict[str, np.ndarray]:
    """Rasgos de las rebanadas spaCy del 06 (lemas, POS y morfologia)."""
    import spacy

    nlp = spacy.load("es_core_news_md", exclude=["ner"])
    n = len(textos)
    banderas = {
        "tiene_insulto_lexico": np.zeros(n, dtype=bool),
        "tiene_insulto_nucleo": np.zeros(n, dtype=bool),
        "tiene_referencia_colectiva": np.zeros(n, dtype=bool),
        "tiene_segunda_persona": np.zeros(n, dtype=bool),
        "tiene_imperativo": np.zeros(n, dtype=bool),
    }
    for i, documento in enumerate(nlp.pipe(textos, batch_size=64)):
        lemas = [token.lemma_.lower() for token in documento]
        pronombres_2 = sum(
            1 for token in documento
            if (token.pos_ in ("PRON", "DET"))
            and (token.lemma_.lower() in SECOND_PERSON_LEXICON or token.text.lower() in SECOND_PERSON_LEXICON)
        )
        verbos_2 = sum(
            1 for token in documento
            if token.pos_ in ("VERB", "AUX") and token.morph.get("Person") == ["2"]
        )
        imperativos = 0
        for token in documento:
            if token.pos_ not in ("VERB", "AUX"):
                continue
            forma = token.text.lower()
            if token.morph.get("Mood") == ["Imp"] or (
                len(forma) >= 4 and forma.isalpha() and forma.endswith(CLITIC_ENDINGS)
                and not INFINITIVE_CLITIC_PATTERN.search(forma)
                and not FIRST_PERSON_PLURAL_PATTERN.search(forma)
            ) or forma in IMPERATIVE_LEXICON or token.lemma_.lower() in IMPERATIVE_LEXICON:
                imperativos += 1
        colectivos = 0
        for indice, token in enumerate(documento):
            if (token.pos_ in ("NOUN", "PROPN") and token.lemma_.lower() in COLLECTIVE_QUANTIFIERS) or (
                token.pos_ in ("NOUN", "PROPN", "ADJ")
                and token.morph.get("Number") == ["Plur"]
                and indice > 0
                and documento[indice - 1].pos_ == "DET"
                and documento[indice - 1].morph.get("Number") == ["Plur"]
            ):
                colectivos += 1
        n_insultos = sum(1 for lema in lemas if lema in lexico)
        n_nucleo = sum(1 for lema in lemas if lema in nucleo)
        banderas["tiene_insulto_lexico"][i] = n_insultos > 0
        banderas["tiene_insulto_nucleo"][i] = n_nucleo > 0
        banderas["tiene_referencia_colectiva"][i] = colectivos > 0
        banderas["tiene_imperativo"][i] = imperativos > 0
        banderas["tiene_segunda_persona"][i] = pronombres_2 > 0 or verbos_2 > 0 or imperativos > 0
    return banderas


def construir_slices(
    textos: list[str], y: np.ndarray, pred: np.ndarray, banderas_spacy: dict[str, np.ndarray]
) -> list[dict[str, Any]]:
    """Construye las rebanadas documentadas (definiciones conservadas)."""
    n = len(textos)
    error = pred != y
    tiene = lambda patron: np.array([bool(patron.search(t)) for t in textos])
    mayusculas = np.array([
        bool(UPPER_WORD_PATTERN.search(t)) or sum(1 for c in t if c.isupper()) >= 4 for t in textos
    ])
    coloquial = np.array([
        any(COLOQUIAL_PATTERN.match(tok.lower()) for tok in TOKEN_PATTERN.findall(t)) for t in textos
    ])
    slices: list[dict[str, Any]] = [
        {"nombre": "URLs", "definicion": "URL_PATTERN (06)", "fuente": "experiments/06_error_analysis.py",
         "tipo": "rebanada_de_texto", "mask": tiene(URL_PATTERN)},
        {"nombre": "hashtags", "definicion": "HASHTAG_PATTERN (06)", "fuente": "experiments/06_error_analysis.py",
         "tipo": "rebanada_de_texto", "mask": tiene(HASHTAG_PATTERN)},
        {"nombre": "menciones", "definicion": "MENTION_PATTERN (06)", "fuente": "experiments/06_error_analysis.py",
         "tipo": "rebanada_de_texto", "mask": tiene(MENTION_PATTERN)},
        {"nombre": "mencion_inicial", "definicion": "INITIAL_MENTION_PATTERN (06)",
         "fuente": "experiments/06_error_analysis.py", "tipo": "rebanada_de_texto", "mask": tiene(INITIAL_MENTION_PATTERN)},
        {"nombre": "elongaciones", "definicion": "ELONGATION_PATTERN (06)", "fuente": "experiments/06_error_analysis.py",
         "tipo": "rebanada_de_texto", "mask": tiene(ELONGATION_PATTERN)},
        {"nombre": "mayusculas_sostenidas", "definicion": "UPPER_WORD_PATTERN o >=4 mayusculas (06)",
         "fuente": "experiments/06_error_analysis.py", "tipo": "rebanada_de_texto", "mask": mayusculas},
        {"nombre": "coloquial", "definicion": "patron por token del grupo coloquial de embedding_01 (adaptacion registrada)",
         "fuente": "experiments/embedding_01_corpus_wikipedia_v5.py", "tipo": "rebanada_de_texto", "mask": coloquial},
        {"nombre": "insultos", "definicion": "lemas en el lexico inducido completo (in-sample, diagnostico)",
         "fuente": "reports/eda/fase2_lexico_inducido_agresividad.csv + spaCy", "tipo": "rebanada_de_texto",
         "mask": banderas_spacy["tiene_insulto_lexico"]},
        {"nombre": "insultos_nucleo", "definicion": f"lemas con z_score >= {Z_NUCLEO} (in-sample, diagnostico)",
         "fuente": "reports/eda/fase2_lexico_inducido_agresividad.csv + spaCy", "tipo": "rebanada_de_texto",
         "mask": banderas_spacy["tiene_insulto_nucleo"]},
        {"nombre": "colectivos_superficie", "definicion": "cuantificadores colectivos o plural tras determinante plural (06 + spaCy)",
         "fuente": "experiments/06_error_analysis.py", "tipo": "rebanada_de_texto",
         "mask": banderas_spacy["tiene_referencia_colectiva"]},
        {"nombre": "segunda_persona", "definicion": "pronombres/verbos de segunda persona o imperativo (06 + spaCy)",
         "fuente": "experiments/06_error_analysis.py", "tipo": "rebanada_de_texto",
         "mask": banderas_spacy["tiene_segunda_persona"]},
        {"nombre": "imperativo", "definicion": "verbos en imperativo (06 + spaCy)",
         "fuente": "experiments/06_error_analysis.py", "tipo": "rebanada_de_texto",
         "mask": banderas_spacy["tiene_imperativo"]},
        {"nombre": "interpersonal", "definicion": "insulto + mencion inicial + sin referencia colectiva (region del 06)",
         "fuente": "experiments/06_error_analysis.py", "tipo": "rebanada_de_texto",
         "mask": banderas_spacy["tiene_insulto_lexico"] & tiene(INITIAL_MENTION_PATTERN) & ~banderas_spacy["tiene_referencia_colectiva"]},
        {"nombre": "cita", "definicion": 'texto contiene comillas dobles (QUOTED v4)',
         "fuente": "experiments/35_discovery_v4.py", "tipo": "rebanada_de_texto",
         "mask": np.array(['"' in t for t in textos])},
        {"nombre": "sin_odio_explicito", "definicion": 'FN_HS sin la subcadena "insulto" (IMPLICIT v4)',
         "fuente": "experiments/35_discovery_v4.py", "tipo": "subconjunto_de_error_v4",
         "mask": (error[:, 0] & (y[:, 0] == 1)) & np.array(["insulto" not in t.lower() for t in textos])},
        {"nombre": "colectivos_v4", "definicion": "errores de AG con AG verdadero 1 (COLLECTIVE v4)",
         "fuente": "experiments/35_discovery_v4.py", "tipo": "subconjunto_de_error_v4",
         "mask": error[:, 2] & (y[:, 2] == 1)},
        {"nombre": "afiliacion", "definicion": "FP_HS cuyo texto contiene @ (AFFILIATIVE v4)",
         "fuente": "experiments/35_discovery_v4.py", "tipo": "subconjunto_de_error_v4",
         "mask": error[:, 0] & (y[:, 0] == 0) & np.array(["@" in t for t in textos])},
        {"nombre": "interpersonal_v4", "definicion": "errores de TR con TR verdadero 1 (DIRECTED v4)",
         "fuente": "experiments/35_discovery_v4.py", "tipo": "subconjunto_de_error_v4",
         "mask": error[:, 1] & (y[:, 1] == 1)},
        {"nombre": "news", "definicion": 'error cuyo texto contiene "http" (NEWS v4)',
         "fuente": "experiments/35_discovery_v4.py", "tipo": "subconjunto_de_error_v4",
         "mask": error.any(axis=1) & np.array(["http" in t for t in textos])},
    ]
    return slices


def auditar_self() -> dict[str, Any]:
    """Comprobacion AST del propio script (operacion y fichero concretos)."""
    arbol = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    docstrings = set()
    for nodo in ast.walk(arbol):
        if isinstance(nodo, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            cuerpo = getattr(nodo, "body", [])
            if cuerpo and isinstance(cuerpo[0], ast.Expr) and isinstance(cuerpo[0].value, ast.Constant):
                if isinstance(cuerpo[0].value.value, str):
                    docstrings.add(id(cuerpo[0].value))
    puro = re.compile(r"^[\w./\\-]+$")
    tokens = []
    for nodo in ast.walk(arbol):
        if isinstance(nodo, ast.Constant) and isinstance(nodo.value, str) and id(nodo) not in docstrings:
            v = nodo.value
            if puro.match(v) and any(p in v.lower() for p in ("toxic_test", "test.csv", "leaderboard", "holdout", "submission")):
                tokens.append((nodo.lineno, v))
    modulos = set()
    for nodo in ast.walk(arbol):
        if isinstance(nodo, ast.Import):
            modulos |= {a.name.split(".")[0] for a in nodo.names}
        elif isinstance(nodo, ast.ImportFrom) and nodo.module:
            modulos.add(nodo.module.split(".")[0])
    return {"tokens": tokens, "modulos": sorted(modulos)}


def main() -> int:
    """Ejecuta el casebook completo."""
    log("carga y validacion del baseline")
    datos = cargar_gate()
    ids, y, pliegues, proba, pred = datos["ids"], datos["y"], datos["pliegues"], datos["proba"], datos["pred"]
    textos = [datos["textos"][int(i)] for i in ids]
    umbrales = datos["umbrales"]
    n = len(ids)
    margen_firmado = proba - umbrales[None, :]
    distancia = np.abs(margen_firmado)
    error = pred != y
    log(f"dev {n} | errores por celda {int(error.sum())} | tweets con error {int(error.any(axis=1).sum())}")

    # ---------- Catalogos ----------
    otras_true = [
        ";".join(ETIQUETAS[k] for k in range(3) if k != j and y[i, k] == 1) for i in range(n) for j in range(3)
    ]
    otras_pred = [
        ";".join(ETIQUETAS[k] for k in range(3) if k != j and pred[i, k] == 1) for i in range(n) for j in range(3)
    ]
    celdas = []
    for i in range(n):
        for j in range(3):
            if not error[i, j]:
                continue
            tipo = "FP" if y[i, j] == 0 else "FN"
            celdas.append({
                "tweet_id": int(ids[i]),
                "text": textos[i],
                "fold": int(pliegues[i]),
                "label": ETIQUETAS[j],
                "y_true": int(y[i, j]),
                "probability": round(float(proba[i, j]), 6),
                "threshold": float(umbrales[j]),
                "y_pred": int(pred[i, j]),
                "error_type": tipo,
                "signed_margin": round(float(margen_firmado[i, j]), 6),
                "absolute_distance_to_threshold": round(float(distancia[i, j]), 6),
                "other_true_labels": otras_true[i * 3 + j],
                "other_predicted_labels": otras_pred[i * 3 + j],
                "n_wrong_labels_for_tweet": int(error[i].sum()),
                "distance_bin": bin_de(float(distancia[i, j])),
            })
    tweets = []
    for i in range(n):
        if not error[i].any():
            continue
        patrones = []
        for j in range(3):
            if error[i, j]:
                tipo = "FP" if y[i, j] == 0 else "FN"
                patrones.append(f"{ETIQUETAS[j]}_{tipo}")
        tweets.append({
            "tweet_id": int(ids[i]),
            "text": textos[i],
            "fold": int(pliegues[i]),
            "HS_true": int(y[i, 0]), "TR_true": int(y[i, 1]), "AG_true": int(y[i, 2]),
            "HS_probability": round(float(proba[i, 0]), 6),
            "TR_probability": round(float(proba[i, 1]), 6),
            "AG_probability": round(float(proba[i, 2]), 6),
            "HS_threshold": float(umbrales[0]), "TR_threshold": float(umbrales[1]), "AG_threshold": float(umbrales[2]),
            "HS_pred": int(pred[i, 0]), "TR_pred": int(pred[i, 1]), "AG_pred": int(pred[i, 2]),
            "wrong_labels": ";".join(ETIQUETAS[j] for j in range(3) if error[i, j]),
            "error_pattern": "|".join(patrones),
            "n_wrong_labels": int(error[i].sum()),
        })
    log(f"catalogo: {len(celdas)} celdas, {len(tweets)} tweets")

    # Validacion cruzada con la taxonomia v4 historica (mismos datos y umbrales).
    v4_actual = {
        "FP_HS": int((error[:, 0] & (y[:, 0] == 0)).sum()),
        "FN_HS": int((error[:, 0] & (y[:, 0] == 1)).sum()),
        "FP_TR": int((error[:, 1] & (y[:, 1] == 0)).sum()),
        "FN_TR": int((error[:, 1] & (y[:, 1] == 1)).sum()),
        "FP_AG": int((error[:, 2] & (y[:, 2] == 0)).sum()),
        "FN_AG": int((error[:, 2] & (y[:, 2] == 1)).sum()),
        "ALL_THREE": int(error.all(axis=1).sum()),
        "IMPLICIT": int(((error[:, 0] & (y[:, 0] == 1)) & np.array(["insulto" not in t.lower() for t in textos])).sum()),
        "QUOTED": int((error.any(axis=1) & np.array(['"' in t for t in textos])).sum()),
        "COLLECTIVE": int((error[:, 2] & (y[:, 2] == 1)).sum()),
        "DIRECTED": int((error[:, 1] & (y[:, 1] == 1)).sum()),
        "AFFILIATIVE": int((error[:, 0] & (y[:, 0] == 0) & np.array(["@" in t for t in textos])).sum()),
        "NEWS": int((error.any(axis=1) & np.array(["http" in t for t in textos])).sum()),
    }
    v4_esperado = {"FP_HS": 173, "FN_HS": 143, "FP_TR": 82, "FN_TR": 82, "FP_AG": 135,
                   "FN_AG": 142, "ALL_THREE": 98, "IMPLICIT": 143, "QUOTED": 16,
                   "COLLECTIVE": 142, "DIRECTED": 82, "AFFILIATIVE": 83, "NEWS": 156}
    if v4_actual != v4_esperado:
        raise SystemExit(f"STOP: la taxonomia v4 no se reproduce: {v4_actual}")
    log("taxonomia v4 historica reproducida exactamente (13 categorias)")

    # ---------- Analisis por etiqueta y por fold ----------
    por_etiqueta = {ETIQUETAS[j]: confusion(y[:, j], pred[:, j]) for j in range(3)}
    fold_rows = []
    for f in range(1, 6):
        sel = pliegues == f
        for j in range(3):
            c = confusion(y[sel, j], pred[sel, j])
            fold_rows.append({"fold": f, "label": ETIQUETAS[j], **c})

    # ---------- Distancia y alta confianza ----------
    conf_rows = []
    for j in range(3):
        et = ETIQUETAS[j]
        for tipo in ("FP", "FN"):
            sel = error[:, j] & ((y[:, j] == 0) if tipo == "FP" else (y[:, j] == 1))
            for _, _, nombre in BINS:
                m = sel & np.array([bin_de(float(d)) == nombre for d in distancia[:, j]])
                for f in range(1, 6):
                    fm = m & (pliegues == f)
                    if int(fm.sum()):
                        conf_rows.append({"label": et, "error_type": tipo, "bin": nombre, "fold": f, "n": int(fm.sum())})
        for dd in DELTAS_HC:
            m = error[:, j] & (distancia[:, j] >= dd)
            conf_rows.append({"label": et, "error_type": "TODOS", "bin": f"alta_confianza_delta>={dd:.2f}",
                              "fold": 0, "n": int(m.sum())})
    log("analisis por etiqueta, fold y distancia listos")

    # ---------- Errores conjuntos ----------
    n_erroneas = error.sum(axis=1)
    conjuntos = {
        "sin_error": int((n_erroneas == 0).sum()),
        "una_etiqueta": int((n_erroneas == 1).sum()),
        "dos_etiquetas": int((n_erroneas == 2).sum()),
        "tres_etiquetas": int((n_erroneas == 3).sum()),
    }
    combinaciones = Counter(
        "|".join(f"{ETIQUETAS[j]}_{'FP' if y[i, j] == 0 else 'FN'}" for j in range(3) if error[i, j])
        for i in range(n) if error[i].any()
    )
    violaciones_true = int(((y[:, 1] == 1) & (y[:, 0] == 0)).sum())
    violaciones_pred = int(((pred[:, 1] == 1) & (pred[:, 0] == 0)).sum())

    # ---------- Slices ----------
    lexico, nucleo = cargar_lexico()
    banderas = rasgos_spacy(textos, lexico, nucleo)
    slices = construir_slices(textos, y, pred, banderas)
    slice_rows = []
    for s in slices:
        mask = s["mask"]
        for ambito, sel in (("dentro", mask), ("fuera", ~mask)):
            for j in range(3):
                c = confusion(y[sel, j], pred[sel, j]) if int(sel.sum()) else {
                    "TP": 0, "FP": 0, "FN": 0, "TN": 0, "precision": 0.0, "recall": 0.0, "F1": 0.0,
                    "prevalencia_real": 0.0, "prevalencia_predicha": 0.0, "errores": 0, "error_rate": 0.0,
                    "FP_count": 0, "FN_count": 0, "n": 0,
                }
                slice_rows.append({
                    "slice": s["nombre"], "definicion": s["definicion"], "fuente": s["fuente"],
                    "tipo": s["tipo"], "ambito": ambito, "label": ETIQUETAS[j],
                    "n_rows": c["n"], "n_examples_with_error": int((error[sel, j]).sum()),
                    "error_rate": c["error_rate"], "FP": c["FP"], "FN": c["FN"],
                    "TP": c["TP"], "TN": c["TN"], "F1": c["F1"],
                    "precision": c["precision"], "recall": c["recall"],
                })
    log(f"slices: {len(slices)} definidas")

    # ---------- Controles (procedimiento predefinido, determinista) ----------
    correctos = []
    for i in range(n):
        for j in range(3):
            if not error[i, j]:
                correctos.append((j, int(pliegues[i]), int(y[i, j]), float(distancia[i, j]), int(ids[i]), i))
    control_rows = []
    sin_control = 0
    for celda in celdas:
        j = ETIQUETAS.index(celda["label"])
        candidatos = [c for c in correctos if c[0] == j and c[1] == celda["fold"] and c[2] == celda["y_true"]]
        if not candidatos:
            sin_control += 1
            continue
        mejor = min(candidatos, key=lambda c: (abs(c[3] - celda["absolute_distance_to_threshold"]), c[4]))
        control_rows.append({
            "control_id": mejor[4], "control_text": textos[mejor[5]], "control_fold": mejor[1],
            "control_label": ETIQUETAS[j], "control_y_true": mejor[2],
            "control_probability": round(float(proba[mejor[5], j]), 6), "control_threshold": float(umbrales[j]),
            "control_y_pred": int(pred[mejor[5], j]), "control_distance_to_threshold": round(mejor[3], 6),
            "matching_criteria": f"misma etiqueta+fold+y_true; |delta_distancia|={abs(mejor[3] - celda['absolute_distance_to_threshold']):.6f}",
            "matched_error_cell": f"{celda['tweet_id']}_{celda['label']}",
        })
    log(f"controles: {len(control_rows)} emparejados | sin control {sin_control}")

    # ---------- Muestra de revision (determinista, estratificada, <=100) ----------
    seleccion: dict[int, dict[str, Any]] = {}
    def anadir(i: int, motivo: str) -> None:
        if i not in seleccion and len(seleccion) < MAX_MUESTRA:
            seleccion[i] = {"reasons": [], "pattern": "", "n_wrong": 0}
        if i in seleccion and motivo not in seleccion[i]["reasons"]:
            seleccion[i]["reasons"].append(motivo)
    patron_por_i = {t["tweet_id"]: t["error_pattern"] for t in tweets}
    nwrong_por_i = {t["tweet_id"]: t["n_wrong_labels"] for t in tweets}
    # (1) un representante por (label, tipo, bin)
    for j, et in enumerate(ETIQUETAS):
        for tipo in ("FP", "FN"):
            for _, _, nombre in BINS:
                cands = [c for c in celdas if c["label"] == et and c["error_type"] == tipo and c["distance_bin"] == nombre]
                if cands:
                    c = min(cands, key=lambda c: (abs(c["signed_margin"]), c["tweet_id"]))
                    anadir(c["tweet_id"], f"estrato {et}_{tipo} {nombre}")
    # (2) cobertura de folds
    for f in range(1, 6):
        for j, et in enumerate(ETIQUETAS):
            for tipo in ("FP", "FN"):
                cands = [c for c in celdas if c["label"] == et and c["error_type"] == tipo and c["fold"] == f]
                if cands:
                    c = min(cands, key=lambda c: (c["absolute_distance_to_threshold"], c["tweet_id"]))
                    anadir(c["tweet_id"], f"fold {f} {et}_{tipo}")
    # (3) cobertura de slices (el error mas cercano al umbral de cada slice)
    for s in slices:
        idx = np.where(s["mask"])[0]
        cands = [c for c in celdas if int(c["tweet_id"]) in set(int(ids[i]) for i in idx)]
        if cands:
            c = min(cands, key=lambda c: (c["absolute_distance_to_threshold"], c["tweet_id"]))
            anadir(c["tweet_id"], f"slice {s['nombre']}")
    # (4) relleno determinista por estrato (orden por distancia y desempate con la semilla)
    for c in sorted(celdas, key=lambda c: (c["label"], c["error_type"], c["distance_bin"], c["absolute_distance_to_threshold"], c["tweet_id"])):
        anadir(c["tweet_id"], f"relleno {c['label']}_{c['error_type']} {c['distance_bin']}")
    sample_rows = []
    for ident in sorted(seleccion):
        t = next(x for x in tweets if x["tweet_id"] == ident)
        sample_rows.append({
            "tipo": "error", "tweet_id": ident, "text": t["text"], "fold": t["fold"],
            "error_pattern": t["error_pattern"], "n_wrong_labels": t["n_wrong_labels"],
            "selection_reasons": "; ".join(seleccion[ident]["reasons"]),
            "label": "", "y_true": "", "probability": "", "threshold": "", "y_pred": "",
            "distance_to_threshold": "", "matching_criteria": "",
        })
    ids_sel = set(seleccion)
    for c in control_rows:
        err_id = int(str(c["matched_error_cell"]).split("_")[0])
        if err_id in ids_sel:
            sample_rows.append({
                "tipo": "control", "tweet_id": c["control_id"], "text": c["control_text"],
                "fold": c["control_fold"], "error_pattern": "", "n_wrong_labels": "",
                "selection_reasons": f"control de {c['matched_error_cell']}",
                "label": c["control_label"], "y_true": c["control_y_true"],
                "probability": c["control_probability"], "threshold": c["control_threshold"],
                "y_pred": c["control_y_pred"], "distance_to_threshold": c["control_distance_to_threshold"],
                "matching_criteria": c["matching_criteria"],
            })
    log(f"muestra de revision: {len(seleccion)} errores + {len(sample_rows) - len(seleccion)} controles")

    # ---------- AST self-check ----------
    auditoria = auditar_self()
    permitidos = {
        "toxic_train.csv", "oof_v03_final.npz", "fase2_lexico_inducido_agresividad.csv",
        # vocabulario del propio chequeo (patrones buscados por auditar_self)
        "toxic_test", "test.csv", "leaderboard", "holdout", "submission", "submissions",
    }
    raros = [v for _, v in auditoria["tokens"] if v not in permitidos]
    if raros:
        raise SystemExit(f"STOP: tokens de ruta inesperados en el propio script: {raros}")
    if {"transformers", "sentence_transformers", "openai", "anthropic"} & set(auditoria["modulos"]):
        raise SystemExit("STOP: imports prohibidos en el propio script")

    # ---------- Escritura ----------
    def escribir_csv(ruta: Path, filas: list[dict[str, Any]]) -> None:
        with ruta.open("w", encoding="utf-8", newline="") as fh:
            escritor = csv.DictWriter(fh, fieldnames=list(filas[0].keys()), lineterminator="\n")
            escritor.writeheader()
            escritor.writerows(filas)

    escribir_csv(RUTA_CELLS_CSV, celdas)
    escribir_csv(RUTA_CASEBOOK_CSV, tweets)
    escribir_csv(RUTA_FOLD_CSV, fold_rows)
    escribir_csv(RUTA_SLICE_CSV, slice_rows)
    escribir_csv(RUTA_CONF_CSV, conf_rows)
    escribir_csv(RUTA_CONTROLS_CSV, control_rows)
    escribir_csv(RUTA_SAMPLE_CSV, sample_rows)

    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=RAIZ).stdout.strip()
    provenance = {
        "ciclo": "P90-00",
        "fecha": "2026-10-10",
        "git_commit": commit,
        "oof_artifact": "features/oof_v03_final.npz",
        "oof_sha256": datos["hash_oof"],
        "oof_fuente": datos["fuente"],
        "oof_nota_mapeo": datos["nota_mapeo"],
        "development_data_hash": datos["hash_train"],
        "source_text_file": "data/raw/toxic_train.csv (filas del desarrollo por id del OOF)",
        "row_count": n,
        "unique_tweet_ids": int(len(set(int(i) for i in ids))),
        "fold_configuration": "pliegues 1-5 del OOF canonico (StratifiedGroupKFold sobre grupos near-dup, D-016)",
        "fold_sizes": {str(f): int((pliegues == f).sum()) for f in range(1, 6)},
        "thresholds": {"HS": UMBRALES[0], "TR": UMBRALES[1], "AG": UMBRALES[2]},
        "class_order": list(ETIQUETAS),
        "case_selection_seed": SEMILLA_CASOS,
        "corrected_cv": V03_CCV,
        "error_cells": len(celdas),
        "error_tweets": len(tweets),
        "restricciones": "sin entrenamiento, sin test, sin holdout, sin leaderboard, sin submissions, sin push",
    }
    RUTA_PROV.write_text(json.dumps(provenance, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

    # ---------- Informe ----------
    def tabla_confusion() -> str:
        lineas = ["| etiqueta | TP | FP | FN | TN | precision | recall | F1 | prev. real | prev. predicha | errores |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for et in ETIQUETAS:
            c = por_etiqueta[et]
            lineas.append(
                f"| {et} | {c['TP']} | {c['FP']} | {c['FN']} | {c['TN']} | {c['precision']:.4f} | "
                f"{c['recall']:.4f} | {c['F1']:.4f} | {c['prevalencia_real']:.4f} | "
                f"{c['prevalencia_predicha']:.4f} | {c['errores']} |"
            )
        return "\n".join(lineas)

    def tabla_ejemplos() -> str:
        lineas = []
        orden = sorted(celdas, key=lambda c: (-c["absolute_distance_to_threshold"], c["tweet_id"]))[:10]
        for c in orden:
            lineas.append(
                f"| {c['tweet_id']} | {c['label']} | {c['error_type']} | {c['probability']:.4f} | "
                f"{c['threshold']:.2f} | {c['absolute_distance_to_threshold']:.4f} | {c['text'][:70]} |"
            )
        return "\n".join(lineas)

    combinaciones_txt = "\n".join(
        f"| {k} | {v} |" for k, v in sorted(combinaciones.items(), key=lambda kv: (-kv[1], kv[0]))
    )
    obs_txt = "\n".join(
        f"- **{o['tipo']}** {o['patron']} (casos: {o['casos']}; contraejemplo: {o['contraejemplo']}): {o['lectura']}"
        for o in OBSERVACIONES
    ) if OBSERVACIONES else "- (pendiente de la lectura de la muestra; se completa en la misma corrida)"

    informe = f"""# TOXIC - P90-00: casebook de errores de V03 sobre el desarrollo

Fecha: 2026-10-10. Script: `experiments/p90_00_casebook_errores_v03.py`.
**Analisis descriptivo. No es un experimento nuevo**: no entrena, no ajusta umbrales,
no toca test ni holdout y no genera submissions.

## Resumen ejecutivo

- Desarrollo autorizado: **{n} filas** (OOF canonico, 5 pliegues, umbrales 0.44/0.51/0.46).
- Errores a nivel tweet-etiqueta: **{len(celdas)}**; tweets con al menos un error: **{len(tweets)}**
  ({100.0 * len(tweets) / n:.2f} por ciento del desarrollo).
- Por etiqueta: HS {por_etiqueta['HS']['errores']} ({por_etiqueta['HS']['FP']} FP / {por_etiqueta['HS']['FN']} FN);
  TR {por_etiqueta['TR']['errores']} ({por_etiqueta['TR']['FP']} FP / {por_etiqueta['TR']['FN']} FN);
  AG {por_etiqueta['AG']['errores']} ({por_etiqueta['AG']['FP']} FP / {por_etiqueta['AG']['FN']} FN).
- Reparto por numero de etiquetas erradas: {conjuntos['una_etiqueta']} con una,
  {conjuntos['dos_etiquetas']} con dos y {conjuntos['tres_etiquetas']} con las tres.
- La taxonomia v4 historica se reproduce exactamente (FP_HS 173, FN_HS 143, FP_TR 82,
  FN_TR 82, FP_AG 135, FN_AG 142, ALL_THREE 98, IMPLICIT 143, QUOTED 16, COLLECTIVE 142,
  DIRECTED 82, AFFILIATIVE 83, NEWS 156).

## 1. Validacion del baseline

```text
OOF            features/oof_v03_final.npz sha256 {datos['hash_oof'][:16]}...
corrected_cv   {datos['ccv']} (reconstruido desde proba/pred/umbrales: coincide)
etiquetas      {list(ETIQUETAS)} | umbrales {list(UMBRALES)}
F1 por etiqueta  HS {datos['f1s'][0]:.6f} | TR {datos['f1s'][1]:.6f} | AG {datos['f1s'][2]:.6f}
fuente OOF     {datos['fuente']}
mapeo          {datos['nota_mapeo']}
```

Cada fila procede de un modelo out-of-fold por construccion del artefacto (mismos ids,
grupos y pliegues corregidos v2); `pred` reproduce `proba >= umbrales` celda a celda.

## 2. Resultados por etiqueta

{tabla_confusion()}

## 3. Resultados por fold

Resumen completo en `v03_error_fold_summary.csv` (15 filas fold x etiqueta con TP/FP/FN/TN,
precision, recall, F1, error_count y error_rate).

## 4. Distancia al umbral

Intervalos predefinidos y frecuencias por etiqueta/tipo/fold en
`v03_error_confidence_summary.csv`. Alta confianza historica (delta >= 0.10 / 0.20 / 0.30
sobre la distancia al umbral, definicion de p61_01):

| etiqueta | delta>=0.10 | delta>=0.20 | delta>=0.30 | errores |
|---|---:|---:|---:|---:|
| HS | {int((error[:, 0] & (distancia[:, 0] >= 0.10)).sum())} | {int((error[:, 0] & (distancia[:, 0] >= 0.20)).sum())} | {int((error[:, 0] & (distancia[:, 0] >= 0.30)).sum())} | {por_etiqueta['HS']['errores']} |
| TR | {int((error[:, 1] & (distancia[:, 1] >= 0.10)).sum())} | {int((error[:, 1] & (distancia[:, 1] >= 0.20)).sum())} | {int((error[:, 1] & (distancia[:, 1] >= 0.30)).sum())} | {por_etiqueta['TR']['errores']} |
| AG | {int((error[:, 2] & (distancia[:, 2] >= 0.10)).sum())} | {int((error[:, 2] & (distancia[:, 2] >= 0.20)).sum())} | {int((error[:, 2] & (distancia[:, 2] >= 0.30)).sum())} | {por_etiqueta['AG']['errores']} |

La distancia al umbral NO se interpreta como probabilidad de error ni como medida
calibrada de confianza.

## 5. Errores conjuntos

```text
sin error {conjuntos['sin_error']} | una {conjuntos['una_etiqueta']} | dos {conjuntos['dos_etiquetas']} | tres {conjuntos['tres_etiquetas']}
```

| combinacion (orden HS, TR, AG) | n |
|---|---:|
{combinaciones_txt}

Relacion documentada TR=1 => HS=1: en el desarrollo hay {violaciones_true} violaciones en la
etiqueta verdadera y {violaciones_pred} violaciones en la prediccion (TR_pred=1 con HS_pred=0).
Se registra como observacion del dataset, no como regla universal, y no se aplica
ninguna correccion automatica.

## 6. Slices historicas

Definiciones conservadas (patrones del 06, lexico inducido de la Fase 2 sobre lemas,
categorias v4 y grupo coloquial de embedding_01). Resumen completo dentro/fuera por
etiqueta en `v03_error_slice_summary.csv` ({len(slice_rows)} filas).

## 7. Errores cerca y lejos de la frontera

```text
A (distancia < 0.025)          errores fronterizos
B (0.025 <= distancia < 0.100) errores de distancia intermedia
C (distancia >= 0.100)         errores alejados del umbral
```

| etiqueta | tipo | A | B | C |
|---|---|---:|---:|---:|
""" + "\n".join(
        f"| {et} | {tipo} | "
        f"{sum(r['n'] for r in conf_rows if r['label'] == et and r['error_type'] == tipo and r['bin'] == BINS[0][2])} | "
        f"{sum(r['n'] for r in conf_rows if r['label'] == et and r['error_type'] == tipo and r['bin'] in (BINS[1][2], BINS[2][2]))} | "
        f"{sum(r['n'] for r in conf_rows if r['label'] == et and r['error_type'] == tipo and r['bin'] in (BINS[3][2], BINS[4][2]))} |"
        for et in ETIQUETAS for tipo in ("FP", "FN")
    ) + f"""

## 8. Muestra de control y muestra de revision

- Controles emparejados 1:1 por etiqueta+fold+y_true con distancia comparable:
  **{len(control_rows)}** en `v03_error_controls.csv`; sin control disponible: {sin_control}.
- Muestra de revision determinista y estratificada: **{len(seleccion)}** tweets con error
  (tope 100) mas sus controles en `v03_error_review_sample.csv`, con todos los motivos de
  seleccion por tweet.

## 9. Patrones cualitativos tentativos

{obs_txt}

Cada observacion referencia casos del catalogo; ninguna se convierte en feature ni se
usa para entrenar. Las categorias son tentativas y requieren contraste.

## 10. Ejemplos representativos (los mas alejados del umbral)

| tweet_id | label | tipo | prob | umbral | distancia | texto (truncado) |
|---|---|---:|---:|---:|---:|---|
{tabla_ejemplos()}

## 11. Preguntas que este analisis puede y no puede responder

**Puede responder:** que tweets falla V03 y en que etiquetas; con que probabilidad y
distancia al umbral; cuantos errores son fronterizos o alejados; que errores son
simultaneos; como se reparten por fold y por slices historicas; que casos conviene
revisar a mano.

**No puede responder (sin trabajo posterior):** si un error concreto es un error de
anotacion (requiere revision humana); si un patron cualitativo es una causa (requiere
contraste); si alguna de estas observaciones es una oportunidad accionable (requiere
un mecanismo legal no medido y su gate).

## 12. Limitaciones

- El analisis es descriptivo sobre el OOF canonico; no reestima generalizacion.
- El lexico de insulto es in-sample (inducido con etiquetas de train): sus rebanadas
  son diagnosticas, no confirmatorias.
- Las rebanadas v4 son subconjuntos de error por construccion (no particiones del texto).
- La distancia al umbral no es una probabilidad calibrada.
- Los controles son 1:1 por celda; cuando no existe candidato comparable se registra.

> V03 remains the best verified leaderboard result.
"""
    RUTA_CASEBOOK_MD.write_text(informe, encoding="utf-8")

    # ---------- ASCII y cierre ----------
    sucios = [
        ruta.name for ruta in (RUTA_CASEBOOK_MD, RUTA_CASEBOOK_CSV, RUTA_CELLS_CSV, RUTA_FOLD_CSV,
                               RUTA_SLICE_CSV, RUTA_CONF_CSV, RUTA_SAMPLE_CSV, RUTA_CONTROLS_CSV, RUTA_PROV)
        if any(b > 127 for b in ruta.read_bytes())
    ]
    if sucios:
        raise SystemExit(f"STOP: bytes > 127 en {sucios}")
    log("entregables escritos y ASCII verificado")
    print(json.dumps({
        "error_cells": len(celdas), "error_tweets": len(tweets),
        "fp_fn": {et: [por_etiqueta[et]["FP"], por_etiqueta[et]["FN"]] for et in ETIQUETAS},
        "conjuntos": conjuntos, "controles": len(control_rows), "muestra": len(seleccion),
        "entrenamientos": 0, "submissions": 0, "push": 0,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
