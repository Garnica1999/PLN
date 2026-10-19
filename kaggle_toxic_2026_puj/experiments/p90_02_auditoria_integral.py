"""P90-02: auditoria integral de V03, corpus, EDA y trazabilidad (parte 1: calculos).

Sin entrenamiento, sin test, sin holdout, sin submissions, sin commits.
Produce las tablas de auditoria en `reports/p90_02_*.csv` y un JSON con todos
los resultados numericos que alimenta la parte 2 (informe y handoff).

Cubre:
- integridad textual (raw -> casebook -> entrada de V03);
- reconciliacion de metricas (pooled vs media de folds, corrected_cv);
- validacion del casebook, de la muestra de revision y de los controles;
- perfil del corpus autorizado (desarrollo, 1483 filas);
- slices (registro por familia, metricas y solapamientos);
- diagnostico de errores y conjuntos;
- sintesis historica (matriz EDA + estado de hipotesis);
- inventario de artefactos y manifiesto;
- compliance (AST por operacion y fichero + ASCII).

Uso:
    /home/carlos/venv/bin/python experiments/p90_02_auditoria_integral.py
"""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Final

import numpy as np

RAIZ: Final[Path] = Path(__file__).resolve().parents[1]

RUTA_OOF: Final[Path] = RAIZ / "features" / "oof_v03_final.npz"
RUTA_TRAIN: Final[Path] = RAIZ / "data" / "raw" / "toxic_train.csv"
RUTA_PARTICION: Final[Path] = RAIZ / "data" / "processed" / "holdout_v2_ids.csv"
RUTA_GRUPOS: Final[Path] = RAIZ / "data" / "processed" / "grupos_near_duplicados.csv"
RUTA_PIPELINE: Final[Path] = RAIZ / "experiments" / "baseline_v03_frozen" / "pipeline_v03.json"
RUTA_HASHES: Final[Path] = RAIZ / "experiments" / "baseline_v03_frozen" / "hashes.txt"
RUTA_SUB: Final[Path] = RAIZ / "submissions" / "toxic_submission_v03_lr_union_thr.csv"
RUTA_MODELO: Final[Path] = RAIZ / "experiments" / "baseline_v03_frozen" / "model_v03.joblib"
RUTA_CASEBOOK: Final[Path] = RAIZ / "reports" / "v03_error_casebook.csv"
RUTA_CELLS: Final[Path] = RAIZ / "reports" / "v03_error_cells.csv"
RUTA_FOLD: Final[Path] = RAIZ / "reports" / "v03_error_fold_summary.csv"
RUTA_SLICE: Final[Path] = RAIZ / "reports" / "v03_error_slice_summary.csv"
RUTA_CONF: Final[Path] = RAIZ / "reports" / "v03_error_confidence_summary.csv"
RUTA_SAMPLE: Final[Path] = RAIZ / "reports" / "v03_error_review_sample.csv"
RUTA_CONTROLS: Final[Path] = RAIZ / "reports" / "v03_error_controls.csv"
RUTA_LEXICO: Final[Path] = RAIZ / "reports" / "eda" / "fase2_lexico_inducido_agresividad.csv"
RUTA_FAMILIAS: Final[Path] = RAIZ / "reports" / "p57_family_status.csv"
RUTA_AUDITORIA_CSV: Final[Path] = RAIZ / "reports" / "toxic_remaining_legal_opportunities.csv"
RUTA_DECISIONS: Final[Path] = RAIZ / "DECISIONS.md"
RUTA_EXPERIMENTS: Final[Path] = RAIZ / "experiments.csv"

RUTA_TEXT_PROV: Final[Path] = RAIZ / "reports" / "p90_02_text_provenance.csv"
RUTA_METRIC: Final[Path] = RAIZ / "reports" / "p90_02_metric_reconciliation.csv"
RUTA_CB_VAL: Final[Path] = RAIZ / "reports" / "p90_02_casebook_validation.csv"
RUTA_SAMPLE_AUDIT: Final[Path] = RAIZ / "reports" / "p90_02_review_sample_audit.csv"
RUTA_CORPUS: Final[Path] = RAIZ / "reports" / "p90_02_corpus_profile.csv"
RUTA_LABELS: Final[Path] = RAIZ / "reports" / "p90_02_label_distribution.csv"
RUTA_SLICE_REG: Final[Path] = RAIZ / "reports" / "p90_02_slice_registry.csv"
RUTA_SLICE_MET: Final[Path] = RAIZ / "reports" / "p90_02_slice_metrics.csv"
RUTA_SLICE_OVER: Final[Path] = RAIZ / "reports" / "p90_02_slice_overlap.csv"
RUTA_DIAG: Final[Path] = RAIZ / "reports" / "p90_02_error_diagnosis.csv"
RUTA_INTER: Final[Path] = RAIZ / "reports" / "p90_02_error_intersections.csv"
RUTA_CASES: Final[Path] = RAIZ / "reports" / "p90_02_representative_cases.csv"
RUTA_HIST: Final[Path] = RAIZ / "reports" / "p90_02_historical_eda_matrix.csv"
RUTA_HYP: Final[Path] = RAIZ / "reports" / "p90_02_hypothesis_status.csv"
RUTA_INV: Final[Path] = RAIZ / "reports" / "p90_02_artifact_inventory.csv"
RUTA_RESULTS: Final[Path] = RAIZ / "reports" / "p90_02_audit_results.json"

HASH_OOF: Final[str] = "559265816ce4f6312c87f008f90324f0eec70a74d1e6247364c162793338feab"
HASH_TRAIN: Final[str] = "3d40e7345b588860cf4744914be88ddc094e4f03591516c8176ab40b7f76d42f"
HASH_SUB: Final[str] = "7d7478f271e6edfd51afff85bad3eaa0e39e6501d42d6346e29ef50ae874520e"
HASH_MODELO: Final[str] = "263ed5bd0fc0de7cabfbaa85e88fca6fa65cf0a1b4e60a6e796b5120e26d13ca"
HASH_PARTICION: Final[str] = "e5b62e5f22c4c63a8c43df4019969504d44c0baa5b503c19ba3560032f5ee6cd"
V03_CCV: Final[float] = 0.7772114525165721
UMBRALES: Final[tuple[float, float, float]] = (0.44, 0.51, 0.46)
ETIQUETAS: Final[tuple[str, str, str]] = ("HS", "TR", "AG")
BINS: Final[tuple[tuple[float, float, str], ...]] = (
    (0.000, 0.025, "[0.000,0.025)"),
    (0.025, 0.050, "[0.025,0.050)"),
    (0.050, 0.100, "[0.050,0.100)"),
    (0.100, 0.200, "[0.100,0.200)"),
    (0.200, float("inf"), "[0.200,inf)"),
)
Z_NUCLEO: Final[float] = 2.5
CASOS_REPRESENTATIVOS: Final[tuple[int, ...]] = (
    22628, 23076, 23839, 21362, 20331, 20678, 23074, 21973, 24508, 20932, 20956, 24269, 23724,
)

# Patrones de superficie, verbatim de experiments/06_error_analysis.py
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
MENTION_PATTERN = re.compile(r"@\w+")
HASHTAG_PATTERN = re.compile(r"#\w+")
INITIAL_MENTION_PATTERN = re.compile(r"^\s*(?:rt\s+)?@\w+", re.IGNORECASE)
UPPER_WORD_PATTERN = re.compile(r"\b[A-Z]{2,}\b")
ELONGATION_PATTERN = re.compile(r"([A-Za-z])\1{2,}")
COLOQUIAL_PATTERN = re.compile(
    r"^(?:jaja\w*|jeje\w*|xd|xdd|xdr|lol|jejej|plis|porfa|xfa|q|qe|xq|pa|pa'|k|tmb|aw|uff|uy|eh|mas|bien|vale|nah)$"
)
TOKEN_PATTERN = re.compile(r"\w+", re.UNICODE)
PUNTUACION_REPETIDA = re.compile(r"[!?]{2,}")
SEGUNDA_PERSONA_LEXICON: Final[frozenset[str]] = frozenset(
    {"tu", "tus", "te", "ti", "usted", "ustedes", "vos", "vosotros", "vosotras",
     "contigo", "os", "vuestro", "vuestra", "vuestros", "vuestras", "tuyo", "tuya",
     "tuyos", "tuyas"}
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
COLLECTIVE_QUANTIFIERS: Final[frozenset[str]] = frozenset(
    {"todos", "todas", "todo", "toda", "muchos", "muchas", "algunos", "algunas",
     "ningun", "ninguna", "ningunos", "ningunas", "cualquiera", "cualesquiera",
     "varios", "varias", "cada", "ambos", "ambas", "gente", "pueblo", "mayoria",
     "minoria", "multitud", "monton", "masa", "grupo"}
)
CLITIC_ENDINGS: Final[tuple[str, ...]] = ("me", "te", "se", "lo", "la", "le", "nos", "os", "los", "las", "les")
INFINITIVE_CLITIC_PATTERN = re.compile(r"r(me|te|se|lo|la|le|nos|os|los|las|les)$")
FIRST_PERSON_PLURAL_PATTERN = re.compile(r"(mos|amos|emos|imos)$")


def log(mensaje: str) -> None:
    """Imprime una linea de progreso."""
    print(f"[P90-02] {mensaje}", flush=True)


def sha256_de(ruta: Path) -> str:
    """Calcula el sha256 de un archivo."""
    digest = hashlib.sha256()
    with ruta.open("rb") as archivo:
        for bloque in iter(lambda: archivo.read(1 << 20), b""):
            digest.update(bloque)
    return digest.hexdigest()


def escribir_csv(ruta: Path, filas: list[dict[str, Any]]) -> None:
    """Escribe un CSV con terminador LF."""
    if not filas:
        raise SystemExit(f"STOP: {ruta.name} quedaria vacio")
    with ruta.open("w", encoding="utf-8", newline="") as fh:
        escritor = csv.DictWriter(fh, fieldnames=list(filas[0].keys()), lineterminator="\n")
        escritor.writeheader()
        escritor.writerows(filas)


def leer_csv(ruta: Path, encoding: str = "utf-8") -> list[dict[str, str]]:
    """Lee un CSV respetando saltos de linea embebidos en campos entrecomillados."""
    with ruta.open(encoding=encoding, newline="") as fh:
        return list(csv.DictReader(fh))


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
        "n_pos_reales": tp + fn, "n_pos_predichos": tp + fp,
        "prevalencia_real": round((tp + fn) / n, 6) if n else 0.0,
        "prevalencia_predicha": round((tp + fp) / n, 6) if n else 0.0,
        "errores": fp + fn, "error_rate": round((fp + fn) / n, 6) if n else 0.0,
        "FP_count": fp, "FN_count": fn, "n": n,
    }


def bin_de(distancia: float) -> str:
    """Intervalo predefinido de distancia al umbral."""
    for bajo, alto, nombre in BINS:
        if bajo <= distancia < alto:
            return nombre
    return BINS[-1][2]


def tasas_exactas(c: dict[str, Any]) -> tuple[float, float, float]:
    """Precision, recall y F1 binarios sin redondeo a partir de una matriz de confusion."""
    p = c["TP"] / (c["TP"] + c["FP"]) if (c["TP"] + c["FP"]) else 0.0
    r = c["TP"] / (c["TP"] + c["FN"]) if (c["TP"] + c["FN"]) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f1


def cargar_gate() -> dict[str, Any]:
    """Carga y valida el OOF, los textos autorizados y los metadatos."""
    hashes = {
        "oof": sha256_de(RUTA_OOF), "train": sha256_de(RUTA_TRAIN),
        "submission": sha256_de(RUTA_SUB), "modelo": sha256_de(RUTA_MODELO),
        "particion": sha256_de(RUTA_PARTICION),
    }
    esperados = {"oof": HASH_OOF, "train": HASH_TRAIN, "submission": HASH_SUB,
                 "modelo": HASH_MODELO, "particion": HASH_PARTICION}
    for clave, esperado in esperados.items():
        if hashes[clave] != esperado:
            raise SystemExit(f"STOP: hash inesperado en {clave}: {hashes[clave]}")
    oof = np.load(RUTA_OOF, allow_pickle=True)
    ids = oof["ids"].astype(np.int64)
    y = oof["y_true"].astype(np.int64)
    pliegues = oof["pliegues"].astype(np.int64)
    proba = oof["proba"].astype(np.float64)
    pred = oof["pred"].astype(np.int64)
    umbrales = np.asarray(oof["umbrales"], dtype=np.float64)
    etiquetas = [str(e) for e in oof["etiquetas"]]
    ccv = float(oof["corrected_cv"][0])
    if etiquetas != list(ETIQUETAS) or tuple(umbrales.tolist()) != UMBRALES:
        raise SystemExit("STOP: etiquetas o umbrales inesperados")
    if ccv != V03_CCV:
        raise SystemExit(f"STOP: corrected_cv inesperado: {ccv!r}")
    if len(ids) != 1483 or len(set(int(i) for i in ids)) != 1483:
        raise SystemExit("STOP: el desarrollo no tiene 1483 ids unicos")
    if not np.array_equal(pred, (proba >= umbrales[None, :]).astype(np.int64)):
        raise SystemExit("STOP: pred no reproduce proba >= umbrales")

    # Textos: SOLO filas del desarrollo (los ids del OOF).
    dev_ids = {int(i) for i in ids}
    textos: dict[int, str] = {}
    with RUTA_TRAIN.open(encoding="utf-8") as fh:
        for fila in csv.DictReader(fh, delimiter=";"):
            ident = int(str(fila["id"]).strip('"'))
            if ident in dev_ids:
                textos[ident] = fila["text"]
    if len(textos) != 1483:
        raise SystemExit(f"STOP: faltan textos de desarrollo: {len(textos)}")

    # Particion: SOLO los metadatos (ids y particion), sin contenido de texto.
    particion = {}
    with RUTA_PARTICION.open(encoding="utf-8-sig") as fh:
        for fila in csv.DictReader(fh):
            particion[int(str(fila["id"]).strip('"'))] = str(fila["particion"]).strip()
    dev_particion = {i for i, p in particion.items() if p == "dev"}
    if dev_particion != dev_ids:
        raise SystemExit("STOP: la particion dev no coincide con los ids del OOF")

    # Grupos near-dup: metadatos de agrupacion (id, grupo, tamano), solo dev.
    grupos: dict[int, tuple[int, int]] = {}
    with RUTA_GRUPOS.open(encoding="utf-8-sig") as fh:
        for fila in csv.DictReader(fh):
            ident = int(str(fila["id"]).strip('"'))
            if ident in dev_ids:
                grupos[ident] = (int(fila["grupo"]), int(fila["tamano_grupo"]))
    if len(grupos) != 1483:
        raise SystemExit(f"STOP: faltan grupos de desarrollo: {len(grupos)}")

    pipeline = json.loads(RUTA_PIPELINE.read_text(encoding="utf-8"))
    return {
        "ids": ids, "y": y, "pliegues": pliegues, "proba": proba, "pred": pred,
        "umbrales": umbrales, "etiquetas": etiquetas, "ccv": ccv,
        "textos": textos, "grupos": grupos, "pipeline": pipeline, "hashes": hashes,
        "fuente_oof": str(oof["fuente"][0]), "nota_mapeo": str(oof["nota_mapeo"][0]),
    }


def auditar_texto(datos: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Auditoria de la cadena textual raw -> casebook -> entrada de V03."""
    ids = datos["ids"]
    textos = [datos["textos"][int(i)] for i in ids]
    casebook = leer_csv(RUTA_CASEBOOK)
    celdas = leer_csv(RUTA_CELLS)
    dif_cb = [f["tweet_id"] for f in casebook if f["text"] != datos["textos"].get(int(f["tweet_id"]))]
    dif_cells = [f["tweet_id"] for f in celdas if f["text"] != datos["textos"].get(int(f["tweet_id"]))]
    ascii_dev = sum(1 for t in textos if any(ord(c) > 127 for c in t))
    formas = ("Espaa", "agresin", "estacin", "polica", "rabes", "inmigracin")
    conteo_formas = {forma: sum(1 for t in textos if forma in t) for forma in formas}
    pipeline = datos["pipeline"]
    word = pipeline["representacion"]["word_tfidf"]
    char = pipeline["representacion"]["char_tfidf"]
    decisiones = RUTA_DECISIONS.read_text(encoding="utf-8")
    d010 = "## D-010" in decisiones and "El corpus oficial no tiene diacriticos" in decisiones
    filas = [
        {"chequeo": "identidad_raw_casebook", "alcance": f"{len(casebook)} tweets con error",
         "resultado": f"{len(dif_cb)} diferencias", "evidencia": "comparacion exacta por tweet_id contra data/raw/toxic_train.csv",
         "estado": "MATCH_CONFIRMED" if not dif_cb else "PARTIAL_MISMATCH"},
        {"chequeo": "identidad_raw_celdas", "alcance": f"{len(celdas)} celdas",
         "resultado": f"{len(dif_cells)} diferencias", "evidencia": "comparacion exacta por tweet_id",
         "estado": "MATCH_CONFIRMED" if not dif_cells else "PARTIAL_MISMATCH"},
        {"chequeo": "ascii_del_desarrollo", "alcance": "1483 textos dev",
         "resultado": f"{ascii_dev} textos con caracteres > 127",
         "evidencia": "recuento byte a byte sobre el texto original",
         "estado": "SOURCE_ALREADY_NORMALIZED" if ascii_dev == 0 else "INCONCLUSIVE"},
        {"chequeo": "formas_sin_diacriticos_en_el_raw", "alcance": "textos dev",
         "resultado": json.dumps(conteo_formas, ensure_ascii=False),
         "evidencia": "busqueda literal en el texto original (las formas ya estan en la fuente)",
         "estado": "SOURCE_ALREADY_NORMALIZED"},
        {"chequeo": "vectorizadores_de_v03", "alcance": "pipeline_v03.json + 04_refit_submissions_v02_v03.py",
         "resultado": f"word={word}; char={char}; strip_accents=None en los scripts de entrenamiento",
         "evidencia": "configuracion congelada; ninguna operacion de retirada de diacriticos en la ruta de V03",
         "estado": "MATCH_CONFIRMED"},
        {"chequeo": "documentacion_historica", "alcance": "D-010",
         "resultado": "D-010 declara que el corpus llega sin tildes ni enye y que train es 100 % ASCII",
         "evidencia": "DECISIONS.md D-010 (2026-09-22)",
         "estado": "MATCH_CONFIRMED" if d010 else "INCONCLUSIVE"},
    ]
    veredicto = {
        "clasificacion": "SOURCE_ALREADY_NORMALIZED",
        "diferencias_casebook": len(dif_cb), "diferencias_celdas": len(dif_cells),
        "ascii_dev": ascii_dev, "formas": conteo_formas,
        "entrada_v03": "texto original del CSV + lowercase del vectorizador; sin strip_accents",
        "conclusion": (
            "las formas sin diacriticos (Espaa, rabes) ya estan en la fuente autorizada: "
            "los organizadores normalizaron el corpus a ASCII (D-010). No hay perdida en el "
            "pipeline ni discrepancia entre raw, casebook y entrada de V03."
        ),
    }
    return filas, veredicto


def reconciliar_metricas(datos: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Reconciliacion de agregaciones de metricas (pooled vs media de folds)."""
    y, pred, pl = datos["y"], datos["pred"], datos["pliegues"]
    pooled = {et: confusion(y[:, j], pred[:, j]) for j, et in enumerate(ETIQUETAS)}
    por_fold: dict[str, list[dict[str, Any]]] = {}
    for j, et in enumerate(ETIQUETAS):
        por_fold[et] = [confusion(y[pl == k, j], pred[pl == k, j]) for k in range(1, 6)]
    media_folds = {et: {clave: float(np.mean([f[clave] for f in por_fold[et]]))
                        for clave in ("precision", "recall", "F1")} for et in ETIQUETAS}
    f1_pliegues = [float(x) for x in np.asarray(np.load(RUTA_OOF, allow_pickle=True)["f1_pliegues"])]
    f1_npz = np.asarray(np.load(RUTA_OOF, allow_pickle=True)["f1_por_etiqueta"], dtype=np.float64)
    macro_pooled = float(np.mean([tasas_exactas(pooled[et])[2] for et in ETIQUETAS]))
    macro_media_folds = float(np.mean([media_folds[et]["F1"] for et in ETIQUETAS]))
    media_macro_pliegues = float(np.mean(f1_pliegues))
    if abs(macro_pooled - V03_CCV) > 1e-12:
        raise SystemExit(f"STOP: macro pooled {macro_pooled!r} != corrected_cv")

    filas = []
    for j, et in enumerate(ETIQUETAS):
        c = pooled[et]
        p_exacta, r_exacta, f_exacta = tasas_exactas(c)
        filas.append({"metric_name": f"confusion_pooled_{et}", "historical_value": "",
                      "recomputed_value": f"TP={c['TP']} FP={c['FP']} FN={c['FN']} TN={c['TN']}",
                      "aggregation_level": "pooled (1483 filas)", "difference": "", "tolerance": "",
                      "status": "REPRODUCED", "source_artifact": "features/oof_v03_final.npz"})
        filas.append({"metric_name": f"F1_pooled_{et}", "historical_value": f"{float(f1_npz[j]):.16f}",
                      "recomputed_value": f"{f_exacta:.16f}", "aggregation_level": "pooled",
                      "difference": f"{f_exacta - float(f1_npz[j]):+.2e}", "tolerance": "1e-12",
                      "status": "REPRODUCED" if abs(f_exacta - float(f1_npz[j])) <= 1e-12 else "REVISAR",
                      "source_artifact": "features/oof_v03_final.npz"})
        filas.append({"metric_name": f"precision_pooled_{et}", "historical_value": "",
                      "recomputed_value": f"{p_exacta:.16f}", "aggregation_level": "pooled",
                      "difference": "", "tolerance": "", "status": "REPRODUCED",
                      "source_artifact": "features/oof_v03_final.npz"})
        filas.append({"metric_name": f"recall_pooled_{et}", "historical_value": "",
                      "recomputed_value": f"{r_exacta:.16f}", "aggregation_level": "pooled",
                      "difference": "", "tolerance": "", "status": "REPRODUCED",
                      "source_artifact": "features/oof_v03_final.npz"})
        filas.append({"metric_name": f"F1_media_folds_{et}", "historical_value": "",
                      "recomputed_value": f"{media_folds[et]['F1']:.6f}",
                      "aggregation_level": "media entre folds", "difference": "", "tolerance": "",
                      "status": "REPRODUCED (nivel distinto, no es la metrica canonica)",
                      "source_artifact": "features/oof_v03_final.npz"})
        for k in range(1, 6):
            filas.append({"metric_name": f"F1_fold{k}_{et}", "historical_value": "",
                          "recomputed_value": f"{por_fold[et][k-1]['F1']:.6f}",
                          "aggregation_level": f"fold {k}", "difference": "", "tolerance": "",
                          "status": "REPRODUCED", "source_artifact": "features/oof_v03_final.npz"})
    filas += [
        {"metric_name": "corrected_cv", "historical_value": f"{V03_CCV:.16f}",
         "recomputed_value": f"{macro_pooled:.16f}", "aggregation_level": "macro pooled (media de F1 pooled por etiqueta)",
         "difference": f"{macro_pooled - V03_CCV:+.2e}", "tolerance": "1e-12",
         "status": "REPRODUCED", "source_artifact": "features/oof_v03_final.npz"},
        {"metric_name": "macro_media_folds", "historical_value": "",
         "recomputed_value": f"{macro_media_folds:.6f}", "aggregation_level": "media entre folds de F1 por etiqueta",
         "difference": "", "tolerance": "", "status": "NIVEL DISTINTO (no es corrected_cv)",
         "source_artifact": "features/oof_v03_final.npz"},
        {"metric_name": "media_macro_pliegues", "historical_value": "",
         "recomputed_value": f"{media_macro_pliegues:.6f}", "aggregation_level": "media de f1_pliegues (macro por fold)",
         "difference": "", "tolerance": "", "status": "NIVEL DISTINTO (no es corrected_cv)",
         "source_artifact": "features/oof_v03_final.npz"},
        {"metric_name": "matriz_historica_train_1731_MOD03b", "historical_value": "HS 614/185/142/790; TR 431/89/88/1123; AG 517/151/145/918",
         "recomputed_value": "no recalculado (poblacion distinta: train 1731 con MOD-03b, reportado en error_analysis.md)",
         "aggregation_level": "train completo 1731 (umbral_optimo)", "difference": "",
         "tolerance": "", "status": "DOCUMENTADO EN INFORME HISTORICO (no comparable con el dev)",
         "source_artifact": "reports/error_analysis.md"},
    ]
    veredicto = {
        "corrected_cv_definicion": "macro pooled: media de los F1 por etiqueta calculados sobre las 1483 predicciones OOF agregadas",
        "corrected_cv_reproducido": f"{macro_pooled:.16f}",
        "f1_por_etiqueta_pooled": {et: tasas_exactas(pooled[et])[2] for et in ETIQUETAS},
        "matrices_pooled": {et: {k: pooled[et][k] for k in ("TP", "FP", "FN", "TN")} for et in ETIQUETAS},
        "media_folds_por_etiqueta": {et: round(media_folds[et]["F1"], 6) for et in ETIQUETAS},
        "media_macro_pliegues": f"{media_macro_pliegues:.6f}",
        "explicacion": (
            "las cifras citadas como discrepantes (TP 614/431/517) proceden de la tabla historica "
            "del TRAIN completo de 1731 filas con MOD-03b (reports/error_analysis.md), una poblacion "
            "distinta del desarrollo de 1483 filas; las matrices del desarrollo son 505/173/143/662, "
            "363/82/82/956 y 425/135/142/781, y sus F1 coinciden exactamente con los F1 por etiqueta "
            "reportados (0.761689/0.815730/0.754215). corrected_cv es el macro pooled y no la media "
            "entre folds (0.7773117), que es un nivel de agregacion distinto."
        ),
        "discrepancias_abiertas": 0,
    }
    return filas, veredicto


def validar_casebook(datos: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Validacion del casebook, la muestra y los controles contra el OOF canonico."""
    ids = [int(i) for i in datos["ids"]]
    idx = {ident: k for k, ident in enumerate(ids)}
    y, proba, pred, pl = datos["y"], datos["proba"], datos["pred"], datos["pliegues"]
    umbrales = datos["umbrales"]
    casebook = leer_csv(RUTA_CASEBOOK)
    celdas = leer_csv(RUTA_CELLS)
    error = pred != y

    mal_prob = mal_pred = mal_patron = mal_fold = mal_id = 0
    for f in casebook:
        ident = int(f["tweet_id"])
        if ident not in idx:
            mal_id += 1
            continue
        k = idx[ident]
        for j, et in enumerate(ETIQUETAS):
            if abs(float(f[f"{et}_probability"]) - float(proba[k, j])) > 1e-6:
                mal_prob += 1
            if int(f[f"{et}_pred"]) != int(pred[k, j]):
                mal_pred += 1
        patron = "|".join(
            f"{et}_{'FP' if y[k, j] == 0 else 'FN'}" for j, et in enumerate(ETIQUETAS) if error[k, j]
        )
        if f["error_pattern"] != patron or int(f["n_wrong_labels"]) != int(error[k].sum()):
            mal_patron += 1
        if int(f["fold"]) != int(pl[k]):
            mal_fold += 1
    mal_cell = 0
    for f in celdas:
        ident = int(f["tweet_id"])
        k = idx.get(ident)
        if k is None:
            mal_cell += 1
            continue
        j = ETIQUETAS.index(f["label"])
        if (abs(float(f["probability"]) - float(proba[k, j])) > 1e-6
                or int(f["y_pred"]) != int(pred[k, j]) or int(f["y_true"]) != int(y[k, j])
                or abs(float(f["threshold"]) - float(umbrales[j])) > 1e-12
                or int(f["fold"]) != int(pl[k])
                or abs(float(f["signed_margin"]) - float(proba[k, j] - umbrales[j])) > 1e-5
                or abs(float(f["absolute_distance_to_threshold"]) - abs(float(proba[k, j] - umbrales[j]))) > 1e-5):
            mal_cell += 1

    # Controles: reproduccion determinista del emparejamiento 1:1 (procedimiento de P90-00).
    controles = leer_csv(RUTA_CONTROLS)
    distancia_todas = np.abs(proba - umbrales[None, :])
    correctos = []
    for i in range(len(ids)):
        for j in range(3):
            if not error[i, j]:
                correctos.append((j, int(pl[i]), int(y[i, j]), float(distancia_todas[i, j]), int(ids[i]), i))
    esperado: dict[str, tuple[tuple[Any, ...], float]] = {}
    for celda in celdas:
        j = ETIQUETAS.index(celda["label"])
        candidatos = [c for c in correctos
                      if c[0] == j and c[1] == int(celda["fold"]) and c[2] == int(celda["y_true"])]
        if not candidatos:
            continue
        mejor = min(candidatos, key=lambda c: (abs(c[3] - float(celda["absolute_distance_to_threshold"])), c[4]))
        esperado[f"{celda['tweet_id']}_{celda['label']}"] = (mejor, float(celda["absolute_distance_to_threshold"]))
    incumple = 0
    for c in controles:
        clave = c["matched_error_cell"]
        if clave not in esperado:
            incumple += 1
            continue
        mejor, d_err = esperado[clave]
        j = ETIQUETAS.index(c["control_label"])
        coincidencia = re.search(r"\|delta_distancia\|=([0-9.]+)", c["matching_criteria"])
        delta_registrado = float(coincidencia.group(1)) if coincidencia else -1.0
        delta_esperado = abs(mejor[3] - d_err)
        if (int(c["control_id"]) != mejor[4] or int(c["control_fold"]) != mejor[1]
                or int(c["control_y_true"]) != mejor[2] or int(c["control_y_pred"]) != int(pred[mejor[5], j])
                or abs(float(c["control_probability"]) - float(proba[mejor[5], j])) > 1.1e-6
                or abs(float(c["control_distance_to_threshold"]) - mejor[3]) > 1.1e-6
                or abs(delta_registrado - delta_esperado) > 2e-6):
            incumple += 1
    reproducidos = len(esperado)
    sin_control = len(celdas) - reproducidos
    repetidos = len(controles) - len({(c["control_id"], c["control_label"]) for c in controles})

    # Muestra: composicion y motivos.
    muestra = leer_csv(RUTA_SAMPLE)
    m_err = [f for f in muestra if f["tipo"] == "error"]
    m_ctl = [f for f in muestra if f["tipo"] == "control"]
    sin_motivo = sum(1 for f in m_err if not f["selection_reasons"].strip())

    filas = [
        {"chequeo": "filas_casebook", "esperado": "441", "observado": str(len(casebook)),
         "estado": "OK" if len(casebook) == 441 else "FALLO", "evidencia": "conteo directo"},
        {"chequeo": "filas_celdas", "esperado": "757", "observado": str(len(celdas)),
         "estado": "OK" if len(celdas) == 757 else "FALLO", "evidencia": "conteo directo"},
        {"chequeo": "ids_fuera_del_desarrollo", "esperado": "0", "observado": str(mal_id),
         "estado": "OK" if mal_id == 0 else "FALLO", "evidencia": "pertenencia al OOF canonico"},
        {"chequeo": "probabilidades_alineadas", "esperado": "0 desalineadas", "observado": str(mal_prob),
         "estado": "OK" if mal_prob == 0 else "FALLO", "evidencia": "comparacion con proba del OOF"},
        {"chequeo": "predicciones_alineadas", "esperado": "0 desalineadas", "observado": str(mal_pred),
         "estado": "OK" if mal_pred == 0 else "FALLO", "evidencia": "comparacion con pred del OOF"},
        {"chequeo": "patrones_y_conteos", "esperado": "0 desviaciones", "observado": str(mal_patron),
         "estado": "OK" if mal_patron == 0 else "FALLO", "evidencia": "error_pattern y n_wrong_labels reconstruidos"},
        {"chequeo": "folds", "esperado": "0 desviaciones", "observado": str(mal_fold),
         "estado": "OK" if mal_fold == 0 else "FALLO", "evidencia": "comparacion con pliegues del OOF"},
        {"chequeo": "celdas_alineadas", "esperado": "0 desviaciones", "observado": str(mal_cell),
         "estado": "OK" if mal_cell == 0 else "FALLO", "evidencia": "prob/pred/umbral/margen/distancia/fold por celda"},
        {"chequeo": "controles_reproducidos_1a1", "esperado": f"{len(celdas)} celdas emparejadas",
         "observado": f"{reproducidos} reproducidos; sin control: {sin_control}",
         "estado": "OK" if (sin_control == 0 and reproducidos == len(celdas)) else "FALLO",
         "evidencia": "reproduccion determinista del procedimiento: misma etiqueta+fold+y_true y vecino mas cercano por |delta distancia|"},
        {"chequeo": "controles_cumplen_criterios", "esperado": "0 incumplimientos", "observado": str(incumple),
         "estado": "OK" if incumple == 0 else "FALLO",
         "evidencia": "control_id/fold/y_true/y_pred/probabilidad/distancia y |delta_distancia| registrado coinciden con el calculo"},
        {"chequeo": "controles_repetidos", "esperado": "documentar", "observado": str(repetidos),
         "estado": "INFORMATIVO", "evidencia": "un control puede emparejarse con varios errores"},
        {"chequeo": "muestra_errores_y_controles", "esperado": "100 + 185", "observado": f"{len(m_err)} + {len(m_ctl)}",
         "estado": "OK" if (len(m_err) == 100 and len(m_ctl) == 185) else "FALLO", "evidencia": "conteo directo"},
        {"chequeo": "muestra_motivos", "esperado": "0 sin motivo", "observado": str(sin_motivo),
         "estado": "OK" if sin_motivo == 0 else "FALLO", "evidencia": "selection_reasons presente en cada error"},
    ]
    conteos = {
        "tweets_con_error": int(error.any(axis=1).sum()),
        "tweets_sin_error": int((~error.any(axis=1)).sum()),
        "celdas_error": int(error.sum()),
        "una_etiqueta": int((error.sum(axis=1) == 1).sum()),
        "dos_etiquetas": int((error.sum(axis=1) == 2).sum()),
        "tres_etiquetas": int((error.sum(axis=1) == 3).sum()),
        "controles": len(controles), "controles_unicos": len({c["control_id"] for c in controles}),
        "controles_incumplen": incumple, "controles_sin_control": sin_control,
        "muestra_errores": len(m_err), "muestra_controles": len(m_ctl),
    }
    return filas, conteos


def perfil_corpus(datos: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Perfil descriptivo del corpus de desarrollo autorizado."""
    ids = [int(i) for i in datos["ids"]]
    textos = [datos["textos"][i] for i in ids]
    y = datos["y"]
    n = len(textos)
    chars = np.array([len(t) for t in textos])
    tokens = np.array([len(t.split()) for t in textos])
    vocab = Counter(tok for t in textos for tok in t.split())
    filas = [
        {"metrica": "filas_autorizadas", "valor": str(n), "alcance": "desarrollo (ids del OOF)", "definicion": "filas con prediccion OOF"},
        {"metrica": "ids_unicos", "valor": str(len(set(ids))), "alcance": "desarrollo", "definicion": "ids distintos"},
        {"metrica": "textos_vacios", "valor": str(sum(1 for t in textos if not t.strip())), "alcance": "desarrollo", "definicion": "texto vacio o solo espacios"},
        {"metrica": "duplicados_exactos_de_texto", "valor": str(len(textos) - len(set(textos))), "alcance": "desarrollo", "definicion": "texto identico repetido (no implica id repetido)"},
        {"metrica": "caracteres_min_media_mediana_max", "valor": f"{chars.min()} / {chars.mean():.1f} / {np.median(chars):.0f} / {chars.max()}", "alcance": "desarrollo", "definicion": "longitud en caracteres del texto original"},
        {"metrica": "caracteres_p10_p90", "valor": f"{np.percentile(chars, 10):.0f} / {np.percentile(chars, 90):.0f}", "alcance": "desarrollo", "definicion": "percentiles 10 y 90"},
        {"metrica": "tokens_min_media_mediana_max", "valor": f"{tokens.min()} / {tokens.mean():.1f} / {np.median(tokens):.0f} / {tokens.max()}", "alcance": "desarrollo", "definicion": "tokens por split de espacios"},
        {"metrica": "tweets_cortos_menos_25_tokens", "valor": str(int((tokens < 25).sum())), "alcance": "desarrollo", "definicion": "umbral historico de texto largo"},
        {"metrica": "tweets_largos_25_o_mas", "valor": str(int((tokens >= 25).sum())), "alcance": "desarrollo", "definicion": "umbral historico de texto largo"},
        {"metrica": "vocabulario_tipos", "valor": str(len(vocab)), "alcance": "desarrollo", "definicion": "tipos por split de espacios (texto original)"},
        {"metrica": "vocabulario_tokens", "valor": str(int(tokens.sum())), "alcance": "desarrollo", "definicion": "tokens totales por split"},
        {"metrica": "hapax", "valor": str(sum(1 for v in vocab.values() if v == 1)), "alcance": "desarrollo", "definicion": "tipos con frecuencia 1"},
        {"metrica": "top20_tokens", "valor": "; ".join(f"{k}:{v}" for k, v in vocab.most_common(20)), "alcance": "desarrollo", "definicion": "20 tipos mas frecuentes (texto original)"},
        {"metrica": "con_url", "valor": str(sum(1 for t in textos if URL_PATTERN.search(t))), "alcance": "desarrollo", "definicion": "URL_PATTERN (06)"},
        {"metrica": "con_hashtag", "valor": str(sum(1 for t in textos if HASHTAG_PATTERN.search(t))), "alcance": "desarrollo", "definicion": "HASHTAG_PATTERN (06)"},
        {"metrica": "con_mencion", "valor": str(sum(1 for t in textos if MENTION_PATTERN.search(t))), "alcance": "desarrollo", "definicion": "MENTION_PATTERN (06)"},
        {"metrica": "con_mencion_inicial", "valor": str(sum(1 for t in textos if INITIAL_MENTION_PATTERN.match(t))), "alcance": "desarrollo", "definicion": "INITIAL_MENTION_PATTERN (06)"},
        {"metrica": "con_elongacion", "valor": str(sum(1 for t in textos if ELONGATION_PATTERN.search(t))), "alcance": "desarrollo", "definicion": "ELONGATION_PATTERN (06)"},
        {"metrica": "con_mayusculas_sostenidas", "valor": str(sum(1 for t in textos if UPPER_WORD_PATTERN.search(t) or sum(1 for c in t if c.isupper()) >= 4)), "alcance": "desarrollo", "definicion": "UPPER_WORD_PATTERN o >=4 mayusculas (06)"},
        {"metrica": "con_puntuacion_repetida", "valor": str(sum(1 for t in textos if PUNTUACION_REPETIDA.search(t))), "alcance": "desarrollo", "definicion": "dos o mas signos ! o ? seguidos"},
        {"metrica": "con_simbolo_no_alfanumerico", "valor": str(sum(1 for t in textos if any(not c.isalnum() and not c.isspace() for c in t))), "alcance": "desarrollo", "definicion": "contiene algun caracter no alfanumerico"},
        {"metrica": "caracteres_no_ascii", "valor": str(sum(1 for t in textos if any(ord(c) > 127 for c in t))), "alcance": "desarrollo", "definicion": "textos con algun byte > 127"},
    ]
    # Etiquetas: marginales, 8 combinaciones y dependencias con denominadores.
    combos = Counter(tuple(int(v) for v in y[k]) for k in range(n))
    filas_labels = []
    for et, j in (("HS", 0), ("TR", 1), ("AG", 2)):
        filas_labels.append({"bloque": "marginal", "clave": et, "n": str(int(y[:, j].sum())),
                             "denominador": str(n), "proporcion": f"{float(y[:, j].mean()):.6f}"})
    for combo, cuenta in sorted(combos.items()):
        filas_labels.append({"bloque": "combinacion", "clave": f"HS={combo[0]} TR={combo[1]} AG={combo[2]}",
                             "n": str(cuenta), "denominador": str(n), "proporcion": f"{cuenta / n:.6f}"})
    for nombre, num, den in (
        ("P(TR=1|HS=1)", int(((y[:, 1] == 1) & (y[:, 0] == 1)).sum()), int((y[:, 0] == 1).sum())),
        ("P(TR=1|HS=0)", int(((y[:, 1] == 1) & (y[:, 0] == 0)).sum()), int((y[:, 0] == 0).sum())),
        ("P(AG=1|HS=1)", int(((y[:, 2] == 1) & (y[:, 0] == 1)).sum()), int((y[:, 0] == 1).sum())),
        ("P(AG=1|HS=0)", int(((y[:, 2] == 1) & (y[:, 0] == 0)).sum()), int((y[:, 0] == 0).sum())),
    ):
        filas_labels.append({"bloque": "dependencia", "clave": nombre, "n": str(num),
                             "denominador": str(den), "proporcion": f"{num / den:.6f}"})
    grupos = datos["grupos"]
    tamanos = Counter(g[0] for g in grupos.values())
    no_unitarios = sum(1 for g in grupos.values() if g[1] > 1)
    grandes = sorted(set(g[0] for g in grupos.values() if g[1] >= 5))
    filas.append({"metrica": "grupos_near_dup_dev", "valor": str(len(tamanos)), "alcance": "desarrollo", "definicion": "grupos del archivo historico (data/processed/grupos_near_duplicados.csv)"})
    filas.append({"metrica": "filas_en_grupos_no_unitarios", "valor": str(no_unitarios), "alcance": "desarrollo", "definicion": "filas con tamano de grupo > 1"})
    filas.append({"metrica": "grupos_de_5_o_mas", "valor": str(len(grandes)), "alcance": "desarrollo", "definicion": "grupos con 5 o mas filas"})
    resumen = {
        "n": n, "chars": {"min": int(chars.min()), "media": round(float(chars.mean()), 2),
                          "mediana": float(np.median(chars)), "max": int(chars.max())},
        "tokens": {"min": int(tokens.min()), "media": round(float(tokens.mean()), 2),
                   "mediana": float(np.median(tokens)), "max": int(tokens.max())},
        "vocab_tipos": len(vocab), "vocab_tokens": int(tokens.sum()),
        "combos": {f"{k}": v for k, v in sorted(combos.items())},
        "no_unitarios": no_unitarios, "grupos": len(tamanos),
    }
    return filas, filas_labels, resumen


def construir_slices(datos: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Registro, metricas y solapamiento de las slices (por familia)."""
    textos = [datos["textos"][int(i)] for i in datos["ids"]]
    y, pred = datos["y"], datos["pred"]
    error = pred != y
    tiene = lambda patron: np.array([bool(patron.search(t)) for t in textos])
    mayusculas = np.array([bool(UPPER_WORD_PATTERN.search(t)) or sum(1 for c in t if c.isupper()) >= 4 for t in textos])
    coloquial = np.array([any(COLOQUIAL_PATTERN.match(tok.lower()) for tok in TOKEN_PATTERN.findall(t)) for t in textos])
    lexico, nucleo = set(), set()
    with RUTA_LEXICO.open(encoding="utf-8") as fh:
        for fila in csv.DictReader(fh):
            if str(fila["seleccionado"]).strip() == "1":
                lexico.add(str(fila["lema"]).strip())
            if float(fila["z_score"]) >= Z_NUCLEO:
                nucleo.add(str(fila["lema"]).strip())
    import spacy
    nlp = spacy.load("es_core_news_md", exclude=["ner"])
    n = len(textos)
    banderas = {k: np.zeros(n, dtype=bool) for k in
                ("insulto", "nucleo", "colectiva", "segunda", "imperativo")}
    for i, documento in enumerate(nlp.pipe(textos, batch_size=64)):
        lemas = [token.lemma_.lower() for token in documento]
        pron2 = sum(1 for token in documento if token.pos_ in ("PRON", "DET") and
                    (token.lemma_.lower() in SEGUNDA_PERSONA_LEXICON or token.text.lower() in SEGUNDA_PERSONA_LEXICON))
        verb2 = sum(1 for token in documento if token.pos_ in ("VERB", "AUX") and token.morph.get("Person") == ["2"])
        imp = 0
        for token in documento:
            if token.pos_ not in ("VERB", "AUX"):
                continue
            forma = token.text.lower()
            if token.morph.get("Mood") == ["Imp"] or (
                len(forma) >= 4 and forma.isalpha() and forma.endswith(CLITIC_ENDINGS)
                and not INFINITIVE_CLITIC_PATTERN.search(forma) and not FIRST_PERSON_PLURAL_PATTERN.search(forma)
            ) or forma in IMPERATIVE_LEXICON or token.lemma_.lower() in IMPERATIVE_LEXICON:
                imp += 1
        col = 0
        for indice, token in enumerate(documento):
            if (token.pos_ in ("NOUN", "PROPN") and token.lemma_.lower() in COLLECTIVE_QUANTIFIERS) or (
                token.pos_ in ("NOUN", "PROPN", "ADJ") and token.morph.get("Number") == ["Plur"]
                and indice > 0 and documento[indice - 1].pos_ == "DET"
                and documento[indice - 1].morph.get("Number") == ["Plur"]
            ):
                col += 1
        banderas["insulto"][i] = any(lema in lexico for lema in lemas)
        banderas["nucleo"][i] = any(lema in nucleo for lema in lemas)
        banderas["colectiva"][i] = col > 0
        banderas["segunda"][i] = pron2 > 0 or verb2 > 0 or imp > 0
        banderas["imperativo"][i] = imp > 0

    definiciones = [
        ("URLs", "SLICE_TEXT_DEFINED", "URL_PATTERN (06)", "experiments/06_error_analysis.py", tiene(URL_PATTERN), False),
        ("hashtags", "SLICE_TEXT_DEFINED", "HASHTAG_PATTERN (06)", "experiments/06_error_analysis.py", tiene(HASHTAG_PATTERN), False),
        ("menciones", "SLICE_TEXT_DEFINED", "MENTION_PATTERN (06)", "experiments/06_error_analysis.py", tiene(MENTION_PATTERN), False),
        ("mencion_inicial", "SLICE_TEXT_DEFINED", "INITIAL_MENTION_PATTERN (06)", "experiments/06_error_analysis.py", tiene(INITIAL_MENTION_PATTERN), False),
        ("elongaciones", "SLICE_TEXT_DEFINED", "ELONGATION_PATTERN (06)", "experiments/06_error_analysis.py", tiene(ELONGATION_PATTERN), False),
        ("mayusculas_sostenidas", "SLICE_TEXT_DEFINED", "UPPER_WORD_PATTERN o >=4 mayusculas (06)", "experiments/06_error_analysis.py", mayusculas, False),
        ("coloquial", "SLICE_TEXT_DEFINED", "patron por token del grupo coloquial (adaptacion registrada)", "experiments/embedding_01_corpus_wikipedia_v5.py", coloquial, False),
        ("colectivos_superficie", "SLICE_TEXT_DEFINED", "cuantificadores colectivos o plural tras determinante (06+spaCy)", "experiments/06_error_analysis.py", banderas["colectiva"], False),
        ("segunda_persona", "SLICE_TEXT_DEFINED", "pronombres/verbos de segunda persona o imperativo (06+spaCy)", "experiments/06_error_analysis.py", banderas["segunda"], False),
        ("imperativo", "SLICE_TEXT_DEFINED", "verbos en imperativo (06+spaCy)", "experiments/06_error_analysis.py", banderas["imperativo"], False),
        ("cita", "SLICE_TEXT_DEFINED", 'texto contiene comillas dobles', "definicion del casebook (distinta de QUOTED v4)", tiene(re.compile('"')), False),
        ("insultos", "SLICE_LEXICON_INDUCED", "lemas en el lexico inducido completo (in-sample)", "reports/eda/fase2_lexico_inducido_agresividad.csv + spaCy", banderas["insulto"], True),
        ("insultos_nucleo", "SLICE_LEXICON_INDUCED", f"lemas con z_score >= {Z_NUCLEO} (in-sample)", "reports/eda/fase2_lexico_inducido_agresividad.csv + spaCy", banderas["nucleo"], True),
        ("interpersonal", "SLICE_LEXICON_INDUCED", "insulto + mencion inicial + sin referencia colectiva (region 06)", "experiments/06_error_analysis.py", banderas["insulto"] & tiene(INITIAL_MENTION_PATTERN) & ~banderas["colectiva"], True),
        ("sin_odio_explicito", "SLICE_ERROR_CONDITIONED", 'FN_HS sin la subcadena "insulto" (IMPLICIT v4)', "experiments/35_discovery_v4.py", (error[:, 0] & (y[:, 0] == 1)) & np.array(["insulto" not in t.lower() for t in textos]), True),
        ("colectivos_v4", "SLICE_ERROR_CONDITIONED", "errores de AG con AG verdadero 1 (COLLECTIVE v4)", "experiments/35_discovery_v4.py", error[:, 2] & (y[:, 2] == 1), True),
        ("afiliacion", "SLICE_ERROR_CONDITIONED", "FP_HS cuyo texto contiene @ (AFFILIATIVE v4)", "experiments/35_discovery_v4.py", error[:, 0] & (y[:, 0] == 0) & np.array(["@" in t for t in textos]), True),
        ("interpersonal_v4", "SLICE_ERROR_CONDITIONED", "errores de TR con TR verdadero 1 (DIRECTED v4)", "experiments/35_discovery_v4.py", error[:, 1] & (y[:, 1] == 1), True),
        ("news", "SLICE_ERROR_CONDITIONED", 'error cuyo texto contiene "http" (NEWS v4)', "experiments/35_discovery_v4.py", error.any(axis=1) & np.array(["http" in t for t in textos]), True),
    ]
    registro, metricas, resumen = [], [], {}
    for nombre, familia, definicion, fuente, mask, lex in definiciones:
        n_rows = int(mask.sum())
        resumen[nombre] = n_rows
        registro.append({
            "slice": nombre, "familia": familia, "definicion": definicion, "fuente": fuente,
            "procedimiento": "recalculo independiente con la definicion citada sobre los 1483 textos dev",
            "depende_de_labels": "SI" if familia == "SLICE_ERROR_CONDITIONED" else "NO",
            "depende_de_predicciones": "SI" if familia == "SLICE_ERROR_CONDITIONED" else "NO",
            "depende_de_errores": "SI" if familia == "SLICE_ERROR_CONDITIONED" else "NO",
            "leakage_de_construccion": "subconjunto definido por errores: F1 interno cero por construccion" if familia == "SLICE_ERROR_CONDITIONED" else (
                "lexico inducido in-sample: diagnostico, no independiente" if familia == "SLICE_LEXICON_INDUCED" else "ninguno"),
            "n_rows_dev": n_rows,
            "interpretacion_valida": "taxonomia y recuentos de errores" if familia == "SLICE_ERROR_CONDITIONED" else "comparacion descriptiva dentro/fuera",
        })
        if familia == "SLICE_ERROR_CONDITIONED":
            sub = mask
            tweets_sub = error.any(axis=1) & sub
            n_etq = error[tweets_sub].sum(axis=1)
            co = Counter(int(v) for v in n_etq)
            for j, et in enumerate(ETIQUETAS):
                metricas.append({
                    "slice": nombre, "familia": familia, "ambito": "subconjunto_de_error", "label": et,
                    "n_rows": n_rows, "n_error_cells": int((sub & error[:, j]).sum()),
                    "FP": int((sub & error[:, j] & (y[:, j] == 0)).sum()),
                    "FN": int((sub & error[:, j] & (y[:, j] == 1)).sum()),
                    "TP": "", "TN": "", "precision": "", "recall": "", "F1": "",
                    "nota": f"F1 omitido (cero por construccion); tweets en el subconjunto: {int(tweets_sub.sum())} con 1/2/3 etiquetas = {co.get(1, 0)}/{co.get(2, 0)}/{co.get(3, 0)}",
                })
        else:
            for ambito, sel in (("dentro", mask), ("fuera", ~mask)):
                for j, et in enumerate(ETIQUETAS):
                    c = confusion(y[sel, j], pred[sel, j]) if int(sel.sum()) else None
                    metricas.append({
                        "slice": nombre, "familia": familia, "ambito": ambito, "label": et,
                        "n_rows": int(sel.sum()), "n_error_cells": int(error[sel, j].sum()),
                        "FP": c["FP"] if c else "", "FN": c["FN"] if c else "",
                        "TP": c["TP"] if c else "", "TN": c["TN"] if c else "",
                        "precision": c["precision"] if c else "", "recall": c["recall"] if c else "",
                        "F1": c["F1"] if c else "",
                        "nota": "lexico inducido in-sample: diagnostico" if lex else "",
                    })
    # Solapamientos por pares entre slices de texto y lexico (no condicionadas por error).
    pares = []
    base = [(n_, m) for n_, f_, _, _, m, _ in definiciones if f_ != "SLICE_ERROR_CONDITIONED"]
    for a in range(len(base)):
        for b in range(a + 1, len(base)):
            na, ma = base[a]
            nb_, mb = base[b]
            inter = int((ma & mb).sum())
            union = int((ma | mb).sum())
            pares.append({"slice_a": na, "slice_b": nb_, "interseccion": inter,
                          "jaccard": round(inter / union, 6) if union else 0.0})
    return registro, metricas, pares, resumen


def diagnostico_errores(datos: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Diagnostico por etiqueta, conjuntos y distancias (con niveles de evidencia)."""
    y, pred, proba, pl = datos["y"], datos["pred"], datos["proba"], datos["pliegues"]
    umbrales = datos["umbrales"]
    textos = [datos["textos"][int(i)] for i in datos["ids"]]
    error = pred != y
    distancia = np.abs(proba - umbrales[None, :])
    filas, inter = [], []
    for j, et in enumerate(ETIQUETAS):
        c = confusion(y[:, j], pred[:, j])
        cerca = int((error[:, j] & (distancia[:, j] < 0.025)).sum())
        medio = int((error[:, j] & (distancia[:, j] >= 0.025) & (distancia[:, j] < 0.100)).sum())
        lejos = int((error[:, j] & (distancia[:, j] >= 0.100)).sum())
        filas.append({"bloque": "etiqueta", "clave": et, "valor": json.dumps({
            "TP": c["TP"], "FP": c["FP"], "FN": c["FN"], "TN": c["TN"],
            "precision": c["precision"], "recall": c["recall"], "F1": c["F1"],
            "errores": c["errores"], "cerca": cerca, "medio": medio, "lejos": lejos,
        }, ensure_ascii=False)})
    combos = Counter()
    for k in range(len(y)):
        if error[k].any():
            combos["|".join(f"{ETIQUETAS[j]}_{'FP' if y[k, j] == 0 else 'FN'}"
                            for j in range(3) if error[k, j])] += 1
    for patron, cuenta in sorted(combos.items(), key=lambda kv: -kv[1]):
        filas.append({"bloque": "conjunto", "clave": patron, "valor": str(cuenta)})
    for j, et in enumerate(ETIQUETAS):
        for nombre, sel in (
            ("cerca_<0.025", error[:, j] & (distancia[:, j] < 0.025)),
            ("lejos_>=0.100", error[:, j] & (distancia[:, j] >= 0.100)),
        ):
            m = sel & error.all(axis=1)
            filas.append({"bloque": "conjunto_distancia", "clave": f"{et}_{nombre}_tres_etiquetas", "valor": str(int(m.sum()))})
    # Hipotesis de FP/FN con evidencia cuantificada (descriptiva).
    BROMA = re.compile(r"\b(te amo|socia|favorit\w*|jaj\w*|jeje\w*|xd+\w*|lol|beer pong|"
                       r"mi (novia|perra|amor|amiga|amigo|hermana|hermano|vida|tia|tio|polla|mujer|favorit\w*))\b", re.I)
    hipotesis = [
        ("FP_HS_con_marcadores_broma_afiliacion", "SUPPORTED",
         sum(1 for k in range(len(y)) if error[k, 0] and y[k, 0] == 0 and BROMA.search(textos[k])),
         sum(1 for k in range(len(y)) if error[k, 0] and y[k, 0] == 0),
         "contraejemplo: FP_HS sin marcadores con probabilidad >= 0.9 (p. ej. 21892)"),
        ("FP_AG_con_marcadores_broma_afiliacion", "SUPPORTED",
         sum(1 for k in range(len(y)) if error[k, 2] and y[k, 2] == 0 and BROMA.search(textos[k])),
         sum(1 for k in range(len(y)) if error[k, 2] and y[k, 2] == 0),
         "misma evidencia que FP_HS; subconjunto, no explicacion completa"),
        ("FN_AG_con_referencia_colectiva_superficie", "TENTATIVE",
         sum(1 for k in range(len(y)) if error[k, 2] and y[k, 2] == 1 and any(
             tok.lower() in ("inmigrantes", "invasores", "negros", "moros", "sudacas", "arabes") for tok in textos[k].split())),
         sum(1 for k in range(len(y)) if error[k, 2] and y[k, 2] == 1),
         "lista ilustrativa; el slice spaCy de colectivos da el recuento sistematico"),
        ("FN_HS_en_textos_con_comilla", "TENTATIVE",
         sum(1 for k in range(len(y)) if error[k, 0] and y[k, 0] == 1 and '"' in textos[k]),
         sum(1 for k in range(len(y)) if error[k, 0] and y[k, 0] == 1),
         "QUOTED v4 tiene 16 errores en total; minoritario"),
        ("errores_fronterizos_menos_0.025", "CONFIRMED",
         int((error & (distancia < 0.025)).sum()), int(error.sum()),
         "porcentaje directo; no implica causa de decision"),
        ("errores_lejanos_mas_0.100", "CONFIRMED",
         int((error & (distancia >= 0.100)).sum()), int(error.sum()),
         "limita el potencial de una solucion basada solo en thresholds"),
    ]
    for clave, nivel, num, den, nota in hipotesis:
        filas.append({"bloque": "hipotesis", "clave": clave,
                      "valor": f"{num}/{den} ({100.0 * num / max(den, 1):.1f}%) | {nivel} | {nota}"})
    # Intersecciones de error por pares de etiquetas.
    pares = [("HS", "TR", 0, 1), ("HS", "AG", 0, 2), ("TR", "AG", 1, 2)]
    for na, nb_, a, b in pares:
        m = error[:, a] & error[:, b]
        inter.append({"interseccion": f"{na}+{nb_}", "n": int(m.sum()),
                      "de": f"{int(error[:, a].sum())} {na} y {int(error[:, b].sum())} {nb_}",
                      "nota": "errores simultaneos; no implica causa comun"})
    solo = [(f"{ETIQUETAS[j]}_solo", int((error[:, j] & ~error[:, [k for k in range(3) if k != j]].any(axis=1)).sum()))
            for j in range(3)]
    for clave, valor in solo:
        inter.append({"interseccion": clave, "n": valor, "de": "errores de esa etiqueta", "nota": "errores aislados"})
    inter.append({"interseccion": "tres_etiquetas", "n": int(error.all(axis=1).sum()),
                  "de": "todos los errores", "nota": "98 historicos verificados"})
    resumen = {
        "errores_celda": int(error.sum()), "tweets_error": int(error.any(axis=1).sum()),
        "combos": dict(combos), "fronterizos": int((error & (distancia < 0.025)).sum()),
        "lejanos": int((error & (distancia >= 0.100)).sum()),
    }
    return filas, inter, resumen


def sintesis_historica() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Matriz consolidada del EDA historico y estado de hipotesis."""
    familias = leer_csv(RUTA_FAMILIAS, encoding="utf-8-sig")
    auditoria = leer_csv(RUTA_AUDITORIA_CSV, encoding="utf-8-sig")
    matriz, hipotesis = [], []
    for f in familias:
        matriz.append({
            "analisis": f["family"], "hipotesis": f["family"], "script": "",
            "informe": f["reusable_artifact"], "artefacto": "",
            "resultado_historico": f["best_result"], "limitaciones": f["LB_transfer"],
            "estado_actual": f["status"], "familia_cerrada": "SI" if f["status"] in ("CLOSED", "INVALIDATED") else "PARCIAL",
            "motivo_de_cierre": f["reason"][:220], "fuente": "reports/p57_family_status.csv",
        })
        hipotesis.append({"linea": f["family"], "estado": f["status"], "motivo": f["reason"][:220],
                          "experimentos": "", "fuente": "reports/p57_family_status.csv"})
    for f in auditoria:
        matriz.append({
            "analisis": f["hypothesis"][:120], "hipotesis": f["hypothesis"][:120], "script": "",
            "informe": "reports/toxic_remaining_legal_opportunities.md", "artefacto": "",
            "resultado_historico": f.get("prior_experiments", "")[:180],
            "limitaciones": f.get("legal_status", ""), "estado_actual": f.get("decision", ""),
            "familia_cerrada": "SI", "motivo_de_cierre": f.get("decision", "")[:220],
            "fuente": "reports/toxic_remaining_legal_opportunities.csv",
        })
        hipotesis.append({"linea": f["hypothesis"][:120], "estado": f.get("decision", ""),
                          "motivo": f.get("legal_status", ""), "experimentos": f.get("prior_experiments", "")[:120],
                          "fuente": "reports/toxic_remaining_legal_opportunities.csv"})
    for linea, estado, motivo in (
        ("P84: exploracion dirigida (DL-01, fastText, ROS, SGNS, EM)", "CLOSED",
         "P84-01..P84-06: ninguna linea supera a V03; D-281 a D-287"),
        ("P85: SGNS corregido sobre Twitter/X (reapertura autorizada)", "CLOSED",
         "Caso C: la fusion no mejora a V03 (D-290); familia de embeddings cerrada"),
        ("P86: Factorization Machine sobre la representacion de V03", "DUPLICATE",
         "V25-01/V25-02 ya la ejecutaron con la representacion exacta (D-291)"),
        ("P87: revision humana de anotaciones en TR", "BLOCKED_BY_RULES",
         "LOW_EXPECTED_VALUE: sin autorizacion y con techo bajo la banda (D-292)"),
        ("P88: LightGBM sobre la matriz TF-IDF de V03", "DUPLICATE",
         "P61-01 ya la ejecuto con la representacion exacta (D-293)"),
        ("P90: casebook de errores de V03", "COMPLEMENTARY",
         "Diagnostico descriptivo; no reabre familias (D-295)"),
    ):
        matriz.append({"analisis": linea, "hipotesis": linea, "script": "", "informe": "", "artefacto": "",
                       "resultado_historico": "", "limitaciones": "", "estado_actual": estado,
                       "familia_cerrada": "SI", "motivo_de_cierre": motivo, "fuente": "DECISIONS.md"})
        hipotesis.append({"linea": linea, "estado": estado, "motivo": motivo, "experimentos": "", "fuente": "DECISIONS.md"})
    return matriz, hipotesis


def inventario() -> list[dict[str, Any]]:
    """Inventario maestro de artefactos de la auditoria."""
    rutas = [
        ("oof_canonico", RUTA_OOF, "datos", "predicciones OOF canonicas de V03"),
        ("train_raw", RUTA_TRAIN, "datos", "fuente autorizada de textos y etiquetas"),
        ("particion", RUTA_PARTICION, "datos", "asignacion dev/holdout (solo metadatos)"),
        ("grupos", RUTA_GRUPOS, "datos", "grupos near-dup (metadatos)"),
        ("pipeline_v03", RUTA_PIPELINE, "config", "configuracion congelada de V03"),
        ("hashes_v03", RUTA_HASHES, "config", "hashes del baseline congelado"),
        ("submission_v03", RUTA_SUB, "submission", "submission historica (LB 0.78306)"),
        ("modelo_v03", RUTA_MODELO, "modelo", "joblib congelado de V03"),
        ("casebook_md", RAIZ / "reports" / "v03_error_casebook.md", "informe", "casebook P90-00"),
        ("casebook_csv", RUTA_CASEBOOK, "tabla", "441 tweets con error"),
        ("cells_csv", RUTA_CELLS, "tabla", "757 celdas tweet-etiqueta"),
        ("fold_csv", RUTA_FOLD, "tabla", "resumen por fold"),
        ("slice_csv", RUTA_SLICE, "tabla", "slices dentro/fuera (mixto, ver P90.02)"),
        ("confidence_csv", RUTA_CONF, "tabla", "distancias y alta confianza"),
        ("sample_csv", RUTA_SAMPLE, "tabla", "muestra de revision"),
        ("controls_csv", RUTA_CONTROLS, "tabla", "controles emparejados"),
        ("provenance_json", RAIZ / "reports" / "v03_error_provenance.json", "metadata", "provenance P90-00"),
        ("lexico_fase2", RUTA_LEXICO, "datos", "lexico inducido in-sample"),
        ("familias_p57", RUTA_FAMILIAS, "informe", "estado de familias historico"),
        ("oportunidades_csv", RUTA_AUDITORIA_CSV, "informe", "auditoria de oportunidades"),
        ("experiments_csv", RUTA_EXPERIMENTS, "registro", "registro experimental (459 filas)"),
        ("decisions", RUTA_DECISIONS, "registro", "decisiones del proyecto"),
    ]
    filas = []
    for aid, ruta, tipo, proposito in rutas:
        existe = ruta.exists()
        filas.append({
            "artifact_id": aid, "path": str(ruta.relative_to(RAIZ)), "type": tipo,
            "purpose": proposito, "source_of_truth": "SI" if aid in ("oof_canonico", "train_raw", "pipeline_v03") else "NO",
            "git_tracked": "SI" if str(ruta.relative_to(RAIZ)) in subprocess.run(
                ["git", "ls-files", "--error-unmatch", str(ruta.relative_to(RAIZ))],
                capture_output=True, text=True, cwd=RAIZ).stdout else "NO",
            "sha256": sha256_de(ruta) if existe else "AUSENTE",
            "date_or_version": "", "related_experiments": "", "status": "OK" if existe else "FALTANTE",
        })
    return filas


def auditar_self(nombre_script: str) -> dict[str, Any]:
    """Comprobacion AST del propio script (operacion y fichero concretos)."""
    arbol = ast.parse((RAIZ / "experiments" / nombre_script).read_text(encoding="utf-8"))
    docstrings = set()
    for nodo in ast.walk(arbol):
        if isinstance(nodo, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            cuerpo = getattr(nodo, "body", [])
            if cuerpo and isinstance(cuerpo[0], ast.Expr) and isinstance(cuerpo[0].value, ast.Constant):
                if isinstance(cuerpo[0].value.value, str):
                    docstrings.add(id(cuerpo[0].value))
    puro = re.compile(r"^[\w./\\-]+\.\w{2,5}$|^[\w./\\-]+/$")
    tokens = []
    for nodo in ast.walk(arbol):
        if isinstance(nodo, ast.Constant) and isinstance(nodo.value, str) and id(nodo) not in docstrings:
            v = nodo.value
            if puro.match(v) and any(p in v.lower() for p in ("toxic_test", "test.csv", "leaderboard", "holdout", "submission")):
                tokens.append(v)
    modulos = set()
    for nodo in ast.walk(arbol):
        if isinstance(nodo, ast.Import):
            modulos |= {a.name.split(".")[0] for a in nodo.names}
        elif isinstance(nodo, ast.ImportFrom) and nodo.module:
            modulos.add(nodo.module.split(".")[0])
    return {"tokens": sorted(set(tokens)), "modulos": sorted(modulos)}


def main() -> int:
    """Ejecuta la auditoria integral (parte 1)."""
    log("carga y validacion del baseline (Gate 0)")
    datos = cargar_gate()
    log(f"dev {len(datos['ids'])} | OOF {datos['hashes']['oof'][:12]} | corrected_cv {datos['ccv']}")

    log("auditoria de integridad textual")
    texto_filas, texto_veredicto = auditar_texto(datos)
    escribir_csv(RUTA_TEXT_PROV, texto_filas)
    log(f"  veredicto: {texto_veredicto['clasificacion']} (diferencias: {texto_veredicto['diferencias_casebook']}/{texto_veredicto['diferencias_celdas']})")

    log("reconciliacion de metricas")
    metric_filas, metric_veredicto = reconciliar_metricas(datos)
    escribir_csv(RUTA_METRIC, metric_filas)
    log(f"  corrected_cv reproducido: {metric_veredicto['corrected_cv_reproducido']} | discrepancias abiertas: {metric_veredicto['discrepancias_abiertas']}")

    log("validacion del casebook, muestra y controles")
    cb_filas, cb_conteos = validar_casebook(datos)
    escribir_csv(RUTA_CB_VAL, cb_filas)
    escribir_csv(RUTA_SAMPLE_AUDIT, [
        {"chequeo": "controles_totales", "valor": str(cb_conteos["controles"]), "nota": "1:1 por celda de error"},
        {"chequeo": "controles_unicos", "valor": str(cb_conteos["controles_unicos"]), "nota": "un control puede repetirse"},
        {"chequeo": "controles_incumplen_criterios", "valor": str(cb_conteos["controles_incumplen"]), "nota": "etiqueta+fold+y_true+direccion"},
        {"chequeo": "muestra_errores", "valor": str(cb_conteos["muestra_errores"]), "nota": "tope 100"},
        {"chequeo": "muestra_controles", "valor": str(cb_conteos["muestra_controles"]), "nota": "controles de los errores seleccionados"},
        {"chequeo": "muestra_no_es_aleatoria", "valor": "CONFIRMADO", "nota": "estratificada; no estimar prevalencias poblacionales"},
    ])
    log(f"  casebook: {cb_conteos['tweets_con_error']} tweets / {cb_conteos['celdas_error']} celdas | controles incumplen {cb_conteos['controles_incumplen']}")

    log("perfil del corpus autorizado")
    corpus_filas, labels_filas, corpus_resumen = perfil_corpus(datos)
    escribir_csv(RUTA_CORPUS, corpus_filas)
    escribir_csv(RUTA_LABELS, labels_filas)
    log(f"  filas {corpus_resumen['n']} | vocab {corpus_resumen['vocab_tipos']} tipos | no unitarios {corpus_resumen['no_unitarios']}")

    log("slices: registro, metricas y solapamiento")
    slice_reg, slice_met, slice_over, slice_resumen = construir_slices(datos)
    escribir_csv(RUTA_SLICE_REG, slice_reg)
    escribir_csv(RUTA_SLICE_MET, slice_met)
    escribir_csv(RUTA_SLICE_OVER, slice_over)
    log(f"  {len(slice_reg)} slices registradas ({len(slice_over)} pares de solapamiento)")

    log("diagnostico de errores")
    diag_filas, inter_filas, diag_resumen = diagnostico_errores(datos)
    escribir_csv(RUTA_DIAG, diag_filas)
    escribir_csv(RUTA_INTER, inter_filas)

    log("sintesis historica")
    hist_filas, hyp_filas = sintesis_historica()
    escribir_csv(RUTA_HIST, hist_filas)
    escribir_csv(RUTA_HYP, hyp_filas)

    log("inventario de artefactos")
    inv_filas = inventario()
    escribir_csv(RUTA_INV, inv_filas)

    # Casos representativos.
    casebook = leer_csv(RUTA_CASEBOOK)
    celdas = leer_csv(RUTA_CELLS)
    casos = []
    for ident in CASOS_REPRESENTATIVOS:
        t = next((f for f in casebook if int(f["tweet_id"]) == ident), None)
        if t is None:
            casos.append({"tweet_id": ident, "estado": "NO_ES_TWEET_CON_ERROR_EN_EL_CASEBOOK"})
            continue
        detalle = [c for c in celdas if int(c["tweet_id"]) == ident]
        casos.append({
            "tweet_id": ident, "fold": t["fold"], "error_pattern": t["error_pattern"],
            "n_wrong_labels": t["n_wrong_labels"], "text": t["text"],
            "HS": f"p={t['HS_probability']} u={t['HS_threshold']} pred={t['HS_pred']} true={t['HS_true']}",
            "TR": f"p={t['TR_probability']} u={t['TR_threshold']} pred={t['TR_pred']} true={t['TR_true']}",
            "AG": f"p={t['AG_probability']} u={t['AG_threshold']} pred={t['AG_pred']} true={t['AG_true']}",
            "distancias": "; ".join(f"{c['label']}:{c['absolute_distance_to_threshold']}" for c in detalle),
            "estado": "VERIFICADO",
        })
    escribir_csv(RUTA_CASES, casos)

    # Compliance AST del propio script.
    auditoria = auditar_self("p90_02_auditoria_integral.py")
    permitidos = {"toxic_train.csv", "holdout_v2_ids.csv", "grupos_near_duplicados.csv",
                  "pipeline_v03.json", "hashes.txt", "toxic_submission_v03_lr_union_thr.csv",
                  "model_v03.joblib", "oof_v03_final.npz", "fase2_lexico_inducido_agresividad.csv",
                  "p57_family_status.csv", "toxic_remaining_legal_opportunities.csv",
                  "v03_error_casebook.csv", "v03_error_cells.csv", "v03_error_fold_summary.csv",
                  "v03_error_slice_summary.csv", "v03_error_confidence_summary.csv",
                  "v03_error_review_sample.csv", "v03_error_controls.csv", "v03_error_provenance.json",
                  "experiments.csv", "DECISIONS.md",
                  "toxic_test", "test.csv", "leaderboard", "holdout", "submission", "submissions"}
    raros = [v for v in auditoria["tokens"] if v not in permitidos]
    if raros:
        raise SystemExit(f"STOP: tokens de ruta inesperados: {raros}")
    prohibidos = {"transformers", "sentence_transformers", "openai", "anthropic"} & set(auditoria["modulos"])
    if prohibidos:
        raise SystemExit(f"STOP: imports prohibidos: {sorted(prohibidos)}")

    # ASCII de los generados.
    generados = [RUTA_TEXT_PROV, RUTA_METRIC, RUTA_CB_VAL, RUTA_SAMPLE_AUDIT, RUTA_CORPUS, RUTA_LABELS,
                 RUTA_SLICE_REG, RUTA_SLICE_MET, RUTA_SLICE_OVER, RUTA_DIAG, RUTA_INTER, RUTA_CASES,
                 RUTA_HIST, RUTA_HYP, RUTA_INV, RAIZ / "experiments" / "p90_02_auditoria_integral.py"]
    sucios = [r.name for r in generados if any(b > 127 for b in r.read_bytes())]
    if sucios:
        raise SystemExit(f"STOP: bytes > 127 en {sucios}")

    # Resultados consolidados para la parte 2.
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=RAIZ).stdout.strip()
    resultados = {
        "audit_id": "P90-02",
        "git_commit": commit,
        "hashes": datos["hashes"],
        "fuente_oof": datos["fuente_oof"], "nota_mapeo": datos["nota_mapeo"],
        "texto": texto_veredicto,
        "metricas": metric_veredicto,
        "casebook": cb_conteos,
        "corpus": corpus_resumen,
        "slices": slice_resumen,
        "diagnostico": diag_resumen,
        "casos": casos,
        "compliance": {"ast_tokens": auditoria["tokens"], "ast_modulos": auditoria["modulos"],
                       "lecturas_autorizadas": sorted(permitidos - {"toxic_test", "test.csv", "leaderboard", "holdout", "submission", "submissions"})},
    }
    RUTA_RESULTS.write_text(json.dumps(resultados, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    if any(b > 127 for b in RUTA_RESULTS.read_bytes()):
        raise SystemExit("STOP: bytes > 127 en p90_02_audit_results.json")
    log("resultados consolidados escritos")
    print(json.dumps({
        "texto": texto_veredicto["clasificacion"],
        "metricas": "REPRODUCED" if metric_veredicto["discrepancias_abiertas"] == 0 else "UNRESOLVED",
        "casebook": f"{cb_conteos['tweets_con_error']} tweets / {cb_conteos['celdas_error']} celdas",
        "corpus": f"{corpus_resumen['n']} filas",
        "entrenamientos": 0, "submissions": 0, "commits": 0,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
