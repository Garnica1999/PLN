"""P90-03: diagnostico de contribuciones de features de V03 sobre los errores OOF.

No entrena modelos, no reentrena V03, no toca test ni holdout, no genera submissions.
Los modelos y vectorizadores exactos POR PLIEGUE que generaron las predicciones OOF
no estan serializados (`experiments/14_oof_corregido_v2.py` los ajusta en memoria y
guarda solo probabilidades): la atribucion exacta OOF queda BLOCKED_BY_MISSING_ARTIFACTS.
El joblib final de V03 (entrenado con todo el train) se usa como evidencia AUXILIAR
claramente separada (AUXILIARY_FULL_MODEL_ATTRIBUTION).

Produce las tablas `reports/p90_03_*.csv` y `reports/p90_03_audit_results.json`.

Uso:
    /home/carlos/venv/bin/python experiments/p90_03_diagnostico_contribuciones.py
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
from typing import Any

import joblib
import numpy as np

RAIZ: Path = Path(__file__).resolve().parents[1]

RUTA_OOF = RAIZ / "features" / "oof_v03_final.npz"
RUTA_V2BASE = RAIZ / "features" / "v2_oof_base_corrected.npz"
RUTA_MODELO = RAIZ / "experiments" / "baseline_v03_frozen" / "model_v03.joblib"
RUTA_TRAIN = RAIZ / "data" / "raw" / "toxic_train.csv"
RUTA_CASEBOOK = RAIZ / "reports" / "v03_error_casebook.csv"
RUTA_CELLS = RAIZ / "reports" / "v03_error_cells.csv"
RUTA_CONTROLS = RAIZ / "reports" / "v03_error_controls.csv"
RUTA_SAMPLE = RAIZ / "reports" / "v03_error_review_sample.csv"
RUTA_LEXICO = RAIZ / "reports" / "eda" / "fase2_lexico_inducido_agresividad.csv"

RUTA_BY_CELL = RAIZ / "reports" / "p90_03_error_contribution_by_cell.csv"
RUTA_SUMMARY = RAIZ / "reports" / "p90_03_feature_contribution_summary.csv"
RUTA_SIGNATURES = RAIZ / "reports" / "p90_03_error_signature_profiles.csv"
RUTA_CONTROL_CMP = RAIZ / "reports" / "p90_03_error_control_comparison.csv"
RUTA_SLICE_SUM = RAIZ / "reports" / "p90_03_slice_contribution_summary.csv"
RUTA_RESULTS = RAIZ / "reports" / "p90_03_audit_results.json"

HASH_OOF = "559265816ce4f6312c87f008f90324f0eec70a74d1e6247364c162793338feab"
HASH_MODELO = "263ed5bd0fc0de7cabfbaa85e88fca6fa65cf0a1b4e60a6e796b5120e26d13ca"
HASH_V2BASE = "27dfa83736b979518a94bc93e32bb9cae07bc82eaff92234fe9e15d1f9822774"
UMBRALES = (0.44, 0.51, 0.46)
ETIQUETAS = ("HS", "TR", "AG")
ESTRATOS = (("cerca", 0.0, 0.025), ("intermedio", 0.025, 0.100), ("lejos", 0.100, float("inf")))
TOPK = 8
Z_NUCLEO = 2.5

URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
MENTION_PATTERN = re.compile(r"@\w+")
HASHTAG_PATTERN = re.compile(r"#\w+")
INITIAL_MENTION_PATTERN = re.compile(r"^\s*(?:rt\s+)?@\w+", re.IGNORECASE)
UPPER_WORD_PATTERN = re.compile(r"\b[A-Z]{2,}\b")
ELONGATION_PATTERN = re.compile(r"([A-Za-z])\1{2,}")
PUNT_REP_PATTERN = re.compile(r"[!?]{2,}")
SEGUNDA_PERSONA_LEXICON = frozenset(
    {"tu", "tus", "te", "ti", "usted", "ustedes", "vos", "vosotros", "vosotras",
     "contigo", "os", "vuestro", "vuestra", "vuestros", "vuestras", "tuyo", "tuya",
     "tuyos", "tuyas"}
)
COLLECTIVE_QUANTIFIERS = frozenset(
    {"todos", "todas", "todo", "toda", "muchos", "muchas", "algunos", "algunas",
     "ningun", "ninguna", "ningunos", "ningunas", "cualquiera", "cualesquiera",
     "varios", "varias", "cada", "ambos", "ambas", "gente", "pueblo", "mayoria",
     "minoria", "multitud", "monton", "masa", "grupo"}
)
IMPERATIVE_LEXICON = frozenset(
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
CLITIC_ENDINGS = ("me", "te", "se", "lo", "la", "le", "nos", "os", "los", "las", "les")
INFINITIVE_CLITIC_PATTERN = re.compile(r"r(me|te|se|lo|la|le|nos|os|los|las|les)$")
FIRST_PERSON_PLURAL_PATTERN = re.compile(r"(mos|amos|emos|imos)$")


def log(mensaje: str) -> None:
    """Imprime una linea de progreso."""
    print(f"[P90-03] {mensaje}", flush=True)


def sha256_de(ruta: Path) -> str:
    """Calcula el sha256 de un archivo."""
    digest = hashlib.sha256()
    with ruta.open("rb") as archivo:
        for bloque in iter(lambda: archivo.read(1 << 20), b""):
            digest.update(bloque)
    return digest.hexdigest()


def leer_csv(ruta: Path, encoding: str = "utf-8") -> list[dict[str, str]]:
    """Lee un CSV respetando saltos de linea embebidos."""
    with ruta.open(encoding=encoding, newline="") as fh:
        return list(csv.DictReader(fh))


def escribir_csv(ruta: Path, filas: list[dict[str, Any]]) -> None:
    """Escribe un CSV con terminador LF."""
    if not filas:
        raise SystemExit(f"STOP: {ruta.name} quedaria vacio")
    with ruta.open("w", encoding="utf-8", newline="") as fh:
        escritor = csv.DictWriter(fh, fieldnames=list(filas[0].keys()), lineterminator="\n")
        escritor.writeheader()
        escritor.writerows(filas)


def logit(p: Any) -> float:
    """Logit con recorte numerico."""
    v = float(np.clip(p, 1e-12, 1.0 - 1e-12))
    return float(np.log(v / (1.0 - v)))


def estrato(distancia: float) -> str:
    """Estrato historico de distancia al threshold."""
    for nombre, bajo, alto in ESTRATOS:
        if bajo <= distancia < alto:
            return nombre
    return ESTRATOS[-1][0]


def cargar_gate() -> dict[str, Any]:
    """Carga y valida el OOF canonico, el modelo final y los textos de desarrollo."""
    hashes = {
        "oof": sha256_de(RUTA_OOF), "modelo": sha256_de(RUTA_MODELO),
        "v2base": sha256_de(RUTA_V2BASE), "train": sha256_de(RUTA_TRAIN),
        "casebook": sha256_de(RUTA_CASEBOOK), "cells": sha256_de(RUTA_CELLS),
        "controls": sha256_de(RUTA_CONTROLS), "sample": sha256_de(RUTA_SAMPLE),
        "lexico": sha256_de(RUTA_LEXICO),
    }
    for clave, esperado in (("oof", HASH_OOF), ("modelo", HASH_MODELO), ("v2base", HASH_V2BASE)):
        if hashes[clave] != esperado:
            raise SystemExit(f"STOP: hash inesperado en {clave}: {hashes[clave]}")

    oof = np.load(RUTA_OOF, allow_pickle=True)
    ids = [int(i) for i in oof["ids"]]
    y = oof["y_true"].astype(np.int64)
    pliegues = oof["pliegues"].astype(np.int64)
    proba = oof["proba"].astype(np.float64)
    pred = oof["pred"].astype(np.int64)
    umbrales = np.asarray(oof["umbrales"], dtype=np.float64)
    etiquetas = [str(e) for e in oof["etiquetas"]]
    if etiquetas != list(ETIQUETAS) or tuple(umbrales.tolist()) != UMBRALES or len(ids) != 1483:
        raise SystemExit("STOP: estructura del OOF inesperada")
    if not np.array_equal(pred, (proba >= umbrales[None, :]).astype(np.int64)):
        raise SystemExit("STOP: pred no reproduce proba >= umbrales")

    # Linaje: M2 del v2 base reordenado por ids == proba canonica.
    v2 = np.load(RUTA_V2BASE, allow_pickle=True)
    modelos_v2 = [str(m) for m in v2["modelos"]]
    pos_m2 = modelos_v2.index("M2_union_lr_c2")
    orden = {int(i): k for k, i in enumerate(v2["ids"])}
    proba_m2 = v2["proba"][[orden[i] for i in ids], pos_m2, :]
    if float(np.abs(proba_m2 - proba).max()) != 0.0:
        raise SystemExit("STOP: el linaje M2 no reproduce la proba canonica")

    modelo = joblib.load(RUTA_MODELO)
    if list(modelo["etiquetas"]) != list(ETIQUETAS) or len(modelo["modelos"]) != 3:
        raise SystemExit("STOP: estructura del joblib final inesperada")

    dev_ids = set(ids)
    textos: dict[int, str] = {}
    with RUTA_TRAIN.open(encoding="utf-8") as fh:
        for fila in csv.DictReader(fh, delimiter=";"):
            ident = int(str(fila["id"]).strip('"'))
            if ident in dev_ids:
                textos[ident] = fila["text"]
    if len(textos) != 1483:
        raise SystemExit("STOP: faltan textos de desarrollo")

    return {
        "hashes": hashes, "ids": ids, "y": y, "pliegues": pliegues, "proba": proba,
        "pred": pred, "umbrales": umbrales, "modelo": modelo,
        "textos": [textos[i] for i in ids],
        "logit_umbral": [logit(t) for t in umbrales],
    }


def preparar_matriz(g: dict[str, Any]) -> dict[str, Any]:
    """Transforma los textos con el vectorizador final y valida la reconstruccion."""
    fu = g["modelo"]["vectorizador"]
    X = fu.transform(g["textos"]).tocsr()
    nombres = np.array([str(n) for n in fu.get_feature_names_out()])
    es_word = np.array([n.startswith("bloque_0__") for n in nombres])
    k_word = int(es_word.sum())
    if not (es_word[:k_word].all() and (~es_word[k_word:]).all()):
        raise SystemExit("STOP: los bloques word/char no son contiguos")
    limpios = np.array([n.split("__", 1)[1] if "__" in n else n for n in nombres])

    modelos = g["modelo"]["modelos"]
    coefs = [np.asarray(mo.coef_[0], dtype=np.float64) for mo in modelos]
    intercepts = [float(mo.intercept_[0]) for mo in modelos]
    Z = np.column_stack([intercepts[j] + np.asarray(X @ coefs[j]).ravel() for j in range(3)])

    # Validacion: sigmoid(z) == predict_proba (exacto) para las 3 etiquetas.
    dif_max = 0.0
    for j, mo in enumerate(modelos):
        p_manual = 1.0 / (1.0 + np.exp(-Z[:, j]))
        p_sk = mo.predict_proba(X)[:, 1]
        dif_max = max(dif_max, float(np.abs(p_manual - p_sk).max()))
    if dif_max > 1e-12:
        raise SystemExit(f"STOP: sigmoid(z) != predict_proba ({dif_max})")

    C = [X.multiply(coefs[j][None, :]).tocsr() for j in range(3)]
    # Reconstruccion: intercept + suma(contribuciones) == z.
    resid = 0.0
    for j in range(3):
        z_suma = intercepts[j] + np.asarray(C[j].sum(axis=1)).ravel()
        resid = max(resid, float(np.abs(z_suma - Z[:, j]).max()))
    if resid > 1e-9:
        raise SystemExit(f"STOP: la descomposicion no reconstruye z ({resid})")

    return {"X": X, "nombres": limpios, "es_word": es_word, "k_word": k_word, "Z": Z,
            "C": C, "coefs": coefs, "intercepts": intercepts, "dif_sigmoid": dif_max,
            "residuo": resid}


def estadisticas_fila(m: dict[str, Any], i: int, j: int) -> dict[str, Any]:
    """Contribuciones agregadas de la fila i para la etiqueta j (modelo final auxiliar)."""
    Cj = m["C"][j]
    k = m["k_word"]
    ini, fin = Cj.indptr[i], Cj.indptr[i + 1]
    idx = Cj.indices[ini:fin]
    vals = Cj.data[ini:fin]
    word = float(vals[idx < k].sum())
    char = float(vals[idx >= k].sum())
    pos = float(vals[vals > 0].sum())
    neg = float(vals[vals < 0].sum())
    orden = np.argsort(-np.abs(vals))
    top3 = float(np.abs(vals[orden][:3]).sum())
    abs_total = float(np.abs(vals).sum())
    top3_share = top3 / abs_total if abs_total else 0.0
    pos_idx = [int(p) for p in orden if vals[p] > 0][:TOPK]
    neg_idx = [int(p) for p in orden if vals[p] < 0][:TOPK]
    top_pos = [(str(m["nombres"][idx[p]]), float(vals[p]), bool(m["es_word"][idx[p]])) for p in pos_idx]
    top_neg = [(str(m["nombres"][idx[p]]), float(vals[p]), bool(m["es_word"][idx[p]])) for p in neg_idx]
    return {"word": word, "char": char, "pos": pos, "neg": neg, "n_active": int(len(vals)),
            "top3_share": top3_share, "top_pos": top_pos, "top_neg": top_neg}


def categoria_feature(nombre: str, es_word: bool, lexico: frozenset[str]) -> str:
    """Categoria descriptiva de una feature (reglas documentadas; no cambia features)."""
    low = nombre.lower()
    if es_word:
        if URL_PATTERN.search(nombre) or "http" in low or "www." in low:
            return "URLs"
        if nombre.startswith("@"):
            return "menciones"
        if nombre.startswith("#"):
            return "hashtags"
        toks = nombre.split()
        if toks and all(not any(c.isalnum() for c in t) for t in toks):
            return "puntuacion"
        if any(t.isdigit() for t in toks):
            return "numeros"
        if any(t.lower() in lexico for t in toks):
            return "insultos y vulgaridad"
        if any(t.lower() in SEGUNDA_PERSONA_LEXICON for t in toks):
            return "expresiones interpersonales"
        if any(t.lower() in COLLECTIVE_QUANTIFIERS for t in toks):
            return "colectivos y referencias grupales"
        if ELONGATION_PATTERN.search(nombre):
            return "elongaciones"
        return "otras (word)"
    if "@" in nombre:
        return "menciones"
    if "http" in low or "www" in low:
        return "URLs"
    if "#" in nombre:
        return "hashtags"
    if re.search(r"(.)\1\1", nombre):
        return "elongaciones"
    if any(c.isdigit() for c in nombre):
        return "numeros"
    if any(not c.isalnum() and not c.isspace() for c in nombre):
        return "puntuacion"
    if " " in nombre:
        return "char con espacio (fronteras)"
    return "char alfabetico"


def categoria_principal(st: dict[str, Any], es_fp: bool, lexico: frozenset[str]) -> dict[str, Any]:
    """Categoria primaria de cobertura de la explicacion (seccion 6.3) y banderas."""
    w, c = st["word"], st["char"]
    pos, neg = st["pos"], st["neg"]
    conflicto_wc = w * c < 0
    ratio = min(pos, -neg) / max(pos, -neg) if (pos > 0 and neg < 0) else 0.0
    conflicto_pn = pos > 0 and neg < 0 and ratio >= 0.3
    cats_top_pos = [categoria_feature(n, ew, lexico) for n, _, ew in st["top_pos"][:10]]
    tiene_lexico_pos = "insultos y vulgaridad" in cats_top_pos
    tiene_contexto = any(cat in ("menciones", "expresiones interpersonales",
                                 "colectivos y referencias grupales") for cat in cats_top_pos)
    if es_fp:
        top1_cat = (categoria_feature(st["top_pos"][0][0], st["top_pos"][0][2], lexico)
                    if st["top_pos"] else "")
        regla_lex = top1_cat == "insultos y vulgaridad" and not tiene_contexto
        regla_nombre = "lexico dominante sin contexto"
    else:
        regla_lex = not tiene_lexico_pos
        regla_nombre = "sin evidencia hostil activada"
    if conflicto_wc:
        principal = "conflicto word-char"
    elif st["top3_share"] >= 0.5:
        principal = "pocas features dominantes"
    elif st["top3_share"] < 0.25:
        principal = "acumulacion de muchas features"
    elif conflicto_pn:
        principal = "conflicto positivas-negativas"
    elif regla_lex:
        principal = regla_nombre
    else:
        principal = "no interpretable con los artefactos disponibles"
    return {"principal": principal, "conflicto_wc": conflicto_wc, "conflicto_pn": conflicto_pn,
            "ratio_pn": round(ratio, 4), "top3_share": round(st["top3_share"], 4)}


def top1_abs(st: dict[str, Any]) -> tuple[str, float, bool] | None:
    """Feature dominante por |contribucion| de la celda."""
    cand = []
    if st["top_pos"]:
        cand.append(st["top_pos"][0])
    if st["top_neg"]:
        cand.append(st["top_neg"][0])
    return max(cand, key=lambda t: abs(t[1])) if cand else None


def formatear_top(items: list[tuple[str, float, bool]]) -> str:
    """Serializa una lista de (feature, contribucion, es_word) para CSV."""
    return "; ".join(f"{n}|{v:.6f}" for n, v, _ in items)


def banderas_texto(t: str) -> dict[str, bool]:
    """Banderas estructurales descriptivas de un texto (mismas definiciones de P90.02)."""
    return {"url": bool(URL_PATTERN.search(t)), "hashtag": bool(HASHTAG_PATTERN.search(t)),
            "mencion": bool(MENTION_PATTERN.search(t)),
            "mencion_inicial": bool(INITIAL_MENTION_PATTERN.match(t)),
            "elongacion": bool(ELONGATION_PATTERN.search(t)),
            "mayusculas": bool(UPPER_WORD_PATTERN.search(t)) or sum(1 for ch in t if ch.isupper()) >= 4,
            "cita": '"' in t}


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
    puro = re.compile(r"^[\w./\\-]+\.\w{2,5}$|^[\w./\\-]+/$")
    tokens = []
    for nodo in ast.walk(arbol):
        if isinstance(nodo, ast.Constant) and isinstance(nodo.value, str) and id(nodo) not in docstrings:
            v = nodo.value
            if puro.match(v) and any(p in v.lower() for p in
                                     ("toxic_test", "test.csv", "leaderboard", "holdout", "submission")):
                tokens.append(v)
    modulos = set()
    for nodo in ast.walk(arbol):
        if isinstance(nodo, ast.Import):
            modulos |= {a.name.split(".")[0] for a in nodo.names}
        elif isinstance(nodo, ast.ImportFrom) and nodo.module:
            modulos.add(nodo.module.split(".")[0])
    return {"tokens": sorted(set(tokens)), "modulos": sorted(modulos)}


def main() -> int:
    """Ejecuta el diagnostico completo de contribuciones (auxiliar) y guarda las tablas."""
    log("gate: OOF canonico, linaje M2, modelo final y textos dev")
    g = cargar_gate()
    log(f"  dev {len(g['ids'])} | hashes OK")

    log("transformacion con el vectorizador final y validacion matematica")
    m = preparar_matriz(g)
    log(f"  sigmoid(z) == predict_proba: max dif {m['dif_sigmoid']:.2e} | "
        f"reconstruccion z: max residuo {m['residuo']:.2e} | features {len(m['nombres'])} "
        f"(word {m['k_word']}, char {len(m['nombres']) - m['k_word']})")

    lexico = frozenset(
        str(f["lema"]).strip() for f in leer_csv(RUTA_LEXICO)
        if str(f["seleccionado"]).strip() == "1"
    )
    log(f"lexico inducido (in-sample, diagnostico): {len(lexico)} lemas")

    # Estadisticas por fila-etiqueta (todas las 1483 filas).
    stats: dict[tuple[int, int], dict[str, Any]] = {}
    for i in range(len(g["ids"])):
        for j in range(3):
            stats[(i, j)] = estadisticas_fila(m, i, j)
    log(f"estadisticas por fila-etiqueta: {len(stats)}")

    # Validacion de los estratos historicos y recuentos de error.
    distancia = np.abs(g["proba"] - g["umbrales"][None, :])
    error = g["pred"] != g["y"]
    estratos_hist = {"HS": (40, 87, 189), "TR": (13, 40, 111), "AG": (31, 80, 166)}
    conteo_estratos = {}
    for j, et in enumerate(ETIQUETAS):
        c = tuple(int((error[:, j] & (distancia[:, j] >= lo) & (distancia[:, j] < hi)).sum())
                  for _, lo, hi in ESTRATOS)
        conteo_estratos[et] = c
        if c != estratos_hist[et]:
            raise SystemExit(f"STOP: estratos {et} no reproducen el historico: {c}")
    if int(error.sum()) != 757 or int(error.any(axis=1).sum()) != 441:
        raise SystemExit("STOP: recuentos de error no reproducen 757/441")
    log("  estratos historicos verificados: HS 40/87/189 | TR 13/40/111 | AG 31/80/166")

    # Tabla por celda de error.
    celdas = leer_csv(RUTA_CELLS)
    idx = {ident: k for k, ident in enumerate(g["ids"])}
    filas_celda = []
    for celda in celdas:
        i = idx[int(celda["tweet_id"])]
        j = ETIQUETAS.index(celda["label"])
        st = stats[(i, j)]
        sig = categoria_principal(st, celda["error_type"] == "FP", lexico)
        p_oof = float(g["proba"][i, j])
        d = float(distancia[i, j])
        resid_celda = abs((m["intercepts"][j] + st["word"] + st["char"]) - float(m["Z"][i, j]))
        filas_celda.append({
            "tweet_id": int(celda["tweet_id"]), "fold": int(celda["fold"]), "label": celda["label"],
            "error_type": celda["error_type"], "y_true": int(celda["y_true"]),
            "probability_oof": f"{p_oof:.6f}", "threshold": f"{g['umbrales'][j]:.2f}",
            "y_pred": int(celda["y_pred"]), "distance_to_threshold": f"{d:.6f}",
            "distance_bin": estrato(d),
            "oof_margin_logit": f"{logit(p_oof) - g['logit_umbral'][j]:.6f}",
            "aux_probability": f"{float(1.0 / (1.0 + np.exp(-float(m['Z'][i, j])))):.6f}",
            "aux_margin_logit": f"{float(m['Z'][i, j]) - g['logit_umbral'][j]:.6f}",
            "intercept_contribution": f"{m['intercepts'][j]:.6f}",
            "word_contribution": f"{st['word']:.6f}", "char_contribution": f"{st['char']:.6f}",
            "positive_contribution_total": f"{st['pos']:.6f}",
            "negative_contribution_total": f"{st['neg']:.6f}",
            "n_features_active": st["n_active"], "top3_abs_share": sig["top3_share"],
            "word_char_conflict": int(sig["conflicto_wc"]), "pos_neg_conflict": int(sig["conflicto_pn"]),
            "signature_primary": sig["principal"],
            "top_positive_features": formatear_top(st["top_pos"]),
            "top_negative_features": formatear_top(st["top_neg"]),
            "reconstruction_error": f"{resid_celda:.2e}",
            "attribution_status": "AUXILIARY_FULL_MODEL_ATTRIBUTION",
            "model_artifact_sha256": HASH_MODELO[:16],
            "vectorizer_artifact_sha256": HASH_MODELO[:16],
        })
    escribir_csv(RUTA_BY_CELL, filas_celda)
    log(f"por celda: {len(filas_celda)} errores")

    # Resumen por etiqueta y tipo.
    filas_resumen = []
    for et in ETIQUETAS:
        for tipo in ("FP", "FN"):
            sel = [(i, j) for (i, j) in stats
                   if j == ETIQUETAS.index(et) and error[i, j]
                   and ((g["y"][i, j] == 0) if tipo == "FP" else (g["y"][i, j] == 1))]
            w = np.array([stats[k]["word"] for k in sel])
            c = np.array([stats[k]["char"] for k in sel])
            d = np.array([distancia[i, j] for (i, j) in sel])
            estr = Counter(estrato(float(x)) for x in d)
            signos = [int(np.sign(stats[k]["word"]) == np.sign(stats[k]["char"])) for k in sel]
            cats = Counter()
            for k in sel:
                t1 = top1_abs(stats[k])
                if t1:
                    cats[categoria_feature(t1[0], t1[2], lexico)] += 1
            folds = Counter(int(g["pliegues"][i]) for (i, j) in sel)
            filas_resumen.append({
                "label": et, "error_type": tipo, "n_errors": len(sel),
                "near_errors": estr.get("cerca", 0), "intermediate_errors": estr.get("intermedio", 0),
                "far_errors": estr.get("lejos", 0),
                "mean_word_contribution": f"{float(w.mean()):.6f}",
                "mean_char_contribution": f"{float(c.mean()):.6f}",
                "mean_abs_word_contribution": f"{float(np.abs(w).mean()):.6f}",
                "mean_abs_char_contribution": f"{float(np.abs(c).mean()):.6f}",
                "share_word_char_same_sign": f"{float(np.mean(signos)):.4f}",
                "dominant_feature_families": "; ".join(f"{k}:{v}" for k, v in cats.most_common(3)),
                "fold_distribution": "; ".join(f"f{k}:{v}" for k, v in sorted(folds.items())),
                "control_comparison_status": "computed_757_relations",
                "attribution_status": "AUXILIARY_FULL_MODEL_ATTRIBUTION",
            })
    escribir_csv(RUTA_SUMMARY, filas_resumen)
    log("resumen por etiqueta y tipo escrito")

    # Perfiles de firma.
    conteo_firma = Counter()
    for (i, j), st in stats.items():
        if not error[i, j]:
            continue
        es_fp = bool(g["y"][i, j] == 0)
        sig = categoria_principal(st, es_fp, lexico)
        conteo_firma[(ETIQUETAS[j], "FP" if es_fp else "FN", sig["principal"])] += 1
    filas_firma = []
    for et in ETIQUETAS:
        for tipo in ("FP", "FN"):
            total = sum(v for (e, t, _), v in conteo_firma.items() if e == et and t == tipo)
            for (e, t, firma), v in sorted(conteo_firma.items()):
                if e == et and t == tipo:
                    filas_firma.append({"label": e, "error_type": t, "signature": firma,
                                        "n": v, "share_within_label_type": f"{v / total:.4f}"})
    escribir_csv(RUTA_SIGNATURES, filas_firma)

    # Comparacion con controles.
    controles = leer_csv(RUTA_CONTROLS)
    reuso = Counter(str(c["control_id"]) for c in controles)
    filas_ctl = []
    for c in controles:
        err_id, err_label = str(c["matched_error_cell"]).rsplit("_", 1)
        i_e, j_e = idx[int(err_id)], ETIQUETAS.index(err_label)
        i_c, j_c = idx[int(c["control_id"])], ETIQUETAS.index(c["control_label"])
        st_e, st_c = stats[(i_e, j_e)], stats[(i_c, j_c)]
        set_pos_e = {n for n, _, _ in st_e["top_pos"][:5]}
        set_pos_c = {n for n, _, _ in st_c["top_pos"][:5]}
        set_neg_e = {n for n, _, _ in st_e["top_neg"][:5]}
        set_neg_c = {n for n, _, _ in st_c["top_neg"][:5]}
        jac_pos = len(set_pos_e & set_pos_c) / len(set_pos_e | set_pos_c) if (set_pos_e | set_pos_c) else 0.0
        jac_neg = len(set_neg_e & set_neg_c) / len(set_neg_e | set_neg_c) if (set_neg_e | set_neg_c) else 0.0
        b_e = banderas_texto(g["textos"][i_e])
        b_c = banderas_texto(g["textos"][i_c])
        flags_iguales = sum(1 for k in b_e if b_e[k] == b_c[k])
        similar_lex = max(jac_pos, jac_neg) >= 0.4
        similar_est = flags_iguales >= 5
        if similar_lex and similar_est:
            clasif = "parecido"
        elif similar_lex:
            clasif = "lexicalmente_similar"
        elif similar_est:
            clasif = "estructuralmente_parecido"
        else:
            clasif = "sin_control_comparable"
        filas_ctl.append({
            "error_tweet_id": int(err_id), "error_label": err_label,
            "error_type": "FP" if g["y"][i_e, j_e] == 0 else "FN",
            "control_id": int(c["control_id"]), "control_label": c["control_label"],
            "control_fold": int(c["control_fold"]), "control_reuse_count": reuso[str(c["control_id"])],
            "delta_distance_canonical": f"{abs(float(c['control_distance_to_threshold']) - distancia[i_e, j_e]):.6f}",
            "delta_margin_logit": f"{logit(g['proba'][i_c, j_c]) - logit(g['proba'][i_e, j_e]):.6f}",
            "error_word_net": f"{st_e['word']:.6f}", "control_word_net": f"{st_c['word']:.6f}",
            "error_char_net": f"{st_e['char']:.6f}", "control_char_net": f"{st_c['char']:.6f}",
            "top5_pos_jaccard": f"{jac_pos:.4f}", "top5_neg_jaccard": f"{jac_neg:.4f}",
            "structural_flags_equal": flags_iguales,
            "classification": clasif,
            "note": "contribuciones auxiliares (modelo final); distancias y margenes canonicos OOF",
        })
    escribir_csv(RUTA_CONTROL_CMP, filas_ctl)
    clasif_conteo = Counter(f["classification"] for f in filas_ctl)
    log(f"controles: {len(filas_ctl)} relaciones | {dict(clasif_conteo)}")

    # Slices priorizadas (definiciones auditadas en P90.02; mismo codigo).
    import spacy
    nlp = spacy.load("es_core_news_md", exclude=["ner"])
    n = len(g["ids"])
    banderas = {k: np.zeros(n, dtype=bool) for k in ("colectiva", "segunda", "imperativo")}
    for i, doc in enumerate(nlp.pipe(g["textos"], batch_size=64)):
        pron2 = sum(1 for t in doc if t.pos_ in ("PRON", "DET") and
                    (t.lemma_.lower() in SEGUNDA_PERSONA_LEXICON or t.text.lower() in SEGUNDA_PERSONA_LEXICON))
        verb2 = sum(1 for t in doc if t.pos_ in ("VERB", "AUX") and t.morph.get("Person") == ["2"])
        imp = 0
        for t in doc:
            if t.pos_ not in ("VERB", "AUX"):
                continue
            forma = t.text.lower()
            if t.morph.get("Mood") == ["Imp"] or (
                len(forma) >= 4 and forma.isalpha() and forma.endswith(CLITIC_ENDINGS)
                and not INFINITIVE_CLITIC_PATTERN.search(forma)
                and not FIRST_PERSON_PLURAL_PATTERN.search(forma)
            ) or forma in IMPERATIVE_LEXICON or t.lemma_.lower() in IMPERATIVE_LEXICON:
                imp += 1
        col = 0
        for k, t in enumerate(doc):
            if (t.pos_ in ("NOUN", "PROPN") and t.lemma_.lower() in COLLECTIVE_QUANTIFIERS) or (
                t.pos_ in ("NOUN", "PROPN", "ADJ") and t.morph.get("Number") == ["Plur"]
                and k > 0 and doc[k - 1].pos_ == "DET" and doc[k - 1].morph.get("Number") == ["Plur"]
            ):
                col += 1
        banderas["colectiva"][i] = col > 0
        banderas["segunda"][i] = pron2 > 0 or verb2 > 0 or imp > 0
        banderas["imperativo"][i] = imp > 0
    tiene = lambda patron: np.array([bool(patron.search(t)) for t in g["textos"]])
    mayusculas = np.array([bool(UPPER_WORD_PATTERN.search(t)) or sum(1 for c in t if c.isupper()) >= 4
                           for t in g["textos"]])
    slices = {
        "URLs": tiene(URL_PATTERN), "hashtags": tiene(HASHTAG_PATTERN),
        "colectivos_superficie": banderas["colectiva"], "menciones": tiene(MENTION_PATTERN),
        "mencion_inicial": tiene(INITIAL_MENTION_PATTERN), "mayusculas_sostenidas": mayusculas,
        "elongaciones": tiene(ELONGATION_PATTERN), "segunda_persona": banderas["segunda"],
        "imperativo": banderas["imperativo"], "cita": tiene(re.compile('"')),
    }

    def resumen_slice(sel: list[tuple[int, int]]) -> dict[str, Any]:
        if not sel:
            return {"n": 0, "word": 0.0, "char": 0.0, "act": 0.0, "top1": ""}
        w = float(np.mean([stats[k]["word"] for k in sel]))
        c = float(np.mean([stats[k]["char"] for k in sel]))
        act = float(np.mean([stats[k]["n_active"] for k in sel]))
        cat = Counter()
        for k in sel:
            t1 = top1_abs(stats[k])
            if t1:
                cat[categoria_feature(t1[0], t1[2], lexico)] += 1
        return {"n": len(sel), "word": w, "char": c, "act": act,
                "top1": "; ".join(f"{k}:{v}" for k, v in cat.most_common(2))}

    filas_slice = []
    for nombre, mask in slices.items():
        for et in ETIQUETAS:
            j = ETIQUETAS.index(et)
            sel_dentro = [(i, j) for i in range(n) if mask[i] and error[i, j]]
            sel_fuera = [(i, j) for i in range(n) if (not mask[i]) and error[i, j]]
            den, fue = resumen_slice(sel_dentro), resumen_slice(sel_fuera)
            filas_slice.append({
                "slice": nombre, "label": et, "n_rows_slice": int(mask.sum()),
                "n_errors_inside": den["n"], "n_errors_outside": fue["n"],
                "mean_word_inside": f"{den['word']:.6f}", "mean_char_inside": f"{den['char']:.6f}",
                "mean_active_inside": f"{den['act']:.2f}",
                "mean_word_outside": f"{fue['word']:.6f}", "mean_char_outside": f"{fue['char']:.6f}",
                "mean_active_outside": f"{fue['act']:.2f}",
                "dominant_inside": den["top1"], "dominant_outside": fue["top1"],
                "note": "contribuciones auxiliares; definiciones de slice auditadas en P90.02",
            })
    escribir_csv(RUTA_SLICE_SUM, filas_slice)
    log("slices priorizadas: resumen de contribuciones escrito")

    # Estructura de etiquetas (TR).
    tr_err = error[:, 1]
    filas_tr = [
        {"tipo": "TR_FN", "n": int((tr_err & (g["y"][:, 1] == 1)).sum()),
         "con_HS_error_simultaneo": int((tr_err & (g["y"][:, 1] == 1) & error[:, 0]).sum()),
         "con_HS_correcta": int((tr_err & (g["y"][:, 1] == 1) & ~error[:, 0]).sum())},
        {"tipo": "TR_FP", "n": int((tr_err & (g["y"][:, 1] == 0)).sum()),
         "con_HS_error_simultaneo": int((tr_err & (g["y"][:, 1] == 0) & error[:, 0]).sum()),
         "con_HS_correcta": int((tr_err & (g["y"][:, 1] == 0) & ~error[:, 0]).sum())},
    ]
    identidad = int(((g["y"][:, 1] == 1) & (g["y"][:, 0] == 0)).sum())

    # Muestra de revision (subset del universo de relaciones).
    muestra = leer_csv(RUTA_SAMPLE)
    sample_err = {int(f["tweet_id"]) for f in muestra if f["tipo"] == "error"}
    sample_ctl = {int(f["tweet_id"]) for f in muestra if f["tipo"] == "control"}
    clasif_muestra = Counter(f["classification"] for f in filas_ctl
                             if f["error_tweet_id"] in sample_err)

    # Agregados por etiqueta/tipo para el gap analysis (contribuciones auxiliares + canonicas).
    def cats_pos(st: dict[str, Any], k: int = 10) -> list[str]:
        return [categoria_feature(n, ew, lexico) for n, _, ew in st["top_pos"][:k]]

    def cats_neg(st: dict[str, Any], k: int = 10) -> list[str]:
        return [categoria_feature(n, ew, lexico) for n, _, ew in st["top_neg"][:k]]

    agregados: dict[str, Any] = {}
    for et in ETIQUETAS:
        idx_et = ETIQUETAS.index(et)
        for tipo in ("FP", "FN"):
            sel = [(i, jj) for (i, jj) in stats if jj == idx_et and error[i, jj]
                   and ((g["y"][i, jj] == 0) if tipo == "FP" else (g["y"][i, jj] == 1))]
            c_pos: Counter = Counter()
            c_neg: Counter = Counter()
            for k in sel:
                if stats[k]["top_pos"]:
                    c_pos[stats[k]["top_pos"][0][0]] += 1
                if stats[k]["top_neg"]:
                    c_neg[stats[k]["top_neg"][0][0]] += 1
            w = np.array([stats[k]["word"] for k in sel])
            c = np.array([stats[k]["char"] for k in sel])
            agregados[f"{et}|{tipo}"] = {
                "n": len(sel),
                "top1_pos_es_lexico": sum(1 for k in sel if stats[k]["top_pos"] and
                    categoria_feature(stats[k]["top_pos"][0][0], stats[k]["top_pos"][0][2], lexico) == "insultos y vulgaridad"),
                "lexico_en_top10_pos": sum(1 for k in sel if "insultos y vulgaridad" in cats_pos(stats[k])),
                "mencion_en_top10_pos": sum(1 for k in sel if "menciones" in cats_pos(stats[k])),
                "interpersonal_en_top10_pos": sum(1 for k in sel if "expresiones interpersonales" in cats_pos(stats[k])),
                "colectivo_en_top10_pos": sum(1 for k in sel if "colectivos y referencias grupales" in cats_pos(stats[k])),
                "sin_lexico_en_top10_pos": sum(1 for k in sel if "insultos y vulgaridad" not in cats_pos(stats[k])),
                "puntuacion_en_top10": sum(1 for k in sel if "puntuacion" in cats_pos(stats[k]) + cats_neg(stats[k])),
                "char_dominante": sum(1 for k in sel if stats[k]["char"] > 0 and
                    stats[k]["char"] / (abs(stats[k]["word"]) + abs(stats[k]["char"]) + 1e-12) >= 0.6),
                "word_dominante": sum(1 for k in sel if stats[k]["word"] > 0 and
                    stats[k]["word"] / (abs(stats[k]["word"]) + abs(stats[k]["char"]) + 1e-12) >= 0.6),
                "conflicto_wc": sum(1 for k in sel if stats[k]["word"] * stats[k]["char"] < 0),
                "ambos_positivos": int(np.sum((w > 0) & (c > 0))),
                "ambos_negativos": int(np.sum((w < 0) & (c < 0))),
                "mismo_signo": int(np.sum(np.sign(w) == np.sign(c))),
                "pos_y_neg": sum(1 for k in sel if stats[k]["pos"] > 0 and stats[k]["neg"] < 0),
                "neg_domina_ratio_0.3": sum(1 for k in sel if stats[k]["pos"] > 0 and stats[k]["neg"] < 0
                    and min(stats[k]["pos"], -stats[k]["neg"]) / max(stats[k]["pos"], -stats[k]["neg"]) >= 0.3),
                "pocas_dominantes": sum(1 for k in sel if stats[k]["top3_share"] >= 0.5),
                "acumulacion": sum(1 for k in sel if stats[k]["top3_share"] < 0.25),
                "cita_texto": sum(1 for k in sel if '"' in g["textos"][k[0]]),
                "url_texto": sum(1 for k in sel if URL_PATTERN.search(g["textos"][k[0]])),
                "elongacion_texto": sum(1 for k in sel if ELONGATION_PATTERN.search(g["textos"][k[0]])),
                "sin_mencion_texto": sum(1 for k in sel if not MENTION_PATTERN.search(g["textos"][k[0]])),
                "top_features_pos": "; ".join(f"{n}:{v}" for n, v in c_pos.most_common(12)),
                "top_features_neg": "; ".join(f"{n}:{v}" for n, v in c_neg.most_common(12)),
            }
    # Errores alejados (>=0.100): firmas y repeticion entre folds.
    far: dict[str, Any] = {}
    for et in ETIQUETAS:
        idx_et = ETIQUETAS.index(et)
        for tipo in ("FP", "FN"):
            sel = [(i, jj) for (i, jj) in stats if jj == idx_et and error[i, jj]
                   and distancia[i, jj] >= 0.100
                   and ((g["y"][i, jj] == 0) if tipo == "FP" else (g["y"][i, jj] == 1))]
            c_pos: Counter = Counter()
            por_fold_feat = {f: Counter() for f in range(1, 6)}
            for k in sel:
                if stats[k]["top_pos"]:
                    c_pos[stats[k]["top_pos"][0][0]] += 1
                    por_fold_feat[int(g["pliegues"][k[0]])][stats[k]["top_pos"][0][0]] += 1
            far[f"{et}|{tipo}"] = {
                "n": len(sel),
                "pocas_dominantes": sum(1 for k in sel if stats[k]["top3_share"] >= 0.5),
                "acumulacion": sum(1 for k in sel if stats[k]["top3_share"] < 0.25),
                "conflicto_wc": sum(1 for k in sel if stats[k]["word"] * stats[k]["char"] < 0),
                "mismo_signo": sum(1 for k in sel if np.sign(stats[k]["word"]) == np.sign(stats[k]["char"])),
                "top_features_pos": "; ".join(f"{n}:{v}" for n, v in c_pos.most_common(8)),
                "top1_pos_por_fold": "; ".join(
                    f"f{f}:{por_fold_feat[f].most_common(1)[0][0]}({por_fold_feat[f].most_common(1)[0][1]})"
                    if por_fold_feat[f] else f"f{f}:-" for f in range(1, 6)),
            }

    # Coeficientes globales del modelo final (evidencia auxiliar de nivel de modelo).
    coef_resumen = {}
    for j, et in enumerate(ETIQUETAS):
        cf = m["coefs"][j]
        orden = np.argsort(-np.abs(cf))
        top_pos = [(str(m["nombres"][t]), float(cf[t]), bool(m["es_word"][t])) for t in orden if cf[t] > 0][:20]
        top_neg = [(str(m["nombres"][t]), float(cf[t]), bool(m["es_word"][t])) for t in orden if cf[t] < 0][:20]
        masa_word = float(np.abs(cf[:m["k_word"]]).sum())
        masa_char = float(np.abs(cf[m["k_word"]:]).sum())
        coef_resumen[et] = {
            "intercept": m["intercepts"][j],
            "top_pos": "; ".join(f"{n}|{c:.4f}" for n, c, _ in top_pos),
            "top_neg": "; ".join(f"{n}|{c:.4f}" for n, c, _ in top_neg),
            "masa_abs_word": masa_word, "masa_abs_char": masa_char,
            "share_masa_word": masa_word / (masa_word + masa_char),
        }

    # Distorsion auxiliar vs OOF (documentada; no se usa como explicacion).
    p_aux = 1.0 / (1.0 + np.exp(-m["Z"]))
    dif_aux_oof = np.abs(p_aux - g["proba"])

    resultados = {
        "audit_id": "P90-03",
        "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                     text=True, cwd=RAIZ).stdout.strip(),
        "hashes": g["hashes"],
        "disponibilidad": {
            "modelos_por_fold": "AUSENTES: 14_oof_corregido_v2.py ajusta en memoria por pliegue y guarda solo probabilidades",
            "modelo_final": "experiments/baseline_v03_frozen/model_v03.joblib (entrenado con todo el train)",
            "atribucion_exacta_oof": "BLOCKED_BY_MISSING_ARTIFACTS",
            "atribucion_auxiliar": "AUXILIARY_FULL_MODEL_ATTRIBUTION",
        },
        "validacion": {
            "sigmoid_vs_predict_proba_max": m["dif_sigmoid"],
            "reconstruccion_z_max_residuo": m["residuo"],
            "lineaje_m2": "exacto (0.0)",
            "estratos_verificados": {k: list(v) for k, v in conteo_estratos.items()},
            "dif_aux_vs_oof_media": float(dif_aux_oof.mean()),
            "dif_aux_vs_oof_p90": float(np.percentile(dif_aux_oof, 90)),
            "n_word_features": m["k_word"], "n_char_features": int(len(m["nombres"]) - m["k_word"]),
        },
        "controles": {"relaciones": len(filas_ctl), "unicos": len(reuso),
                      "clasificacion": dict(clasif_conteo)},
        "muestra": {"errores": len(sample_err), "controles": len(sample_ctl),
                    "relaciones_de_la_muestra": sum(1 for f in filas_ctl if f["error_tweet_id"] in sample_err),
                    "clasificacion": dict(clasif_muestra)},
        "estructura_tr": {"identidad_violaciones": identidad, "filas": filas_tr},
        "firmas": {f"{e}|{t}|{firma}": v for (e, t, firma), v in conteo_firma.items()},
        "agregados": agregados,
        "far": far,
        "coeficientes": coef_resumen,
        "celdas": len(filas_celda),
    }
    RUTA_RESULTS.write_text(json.dumps(resultados, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

    # ASCII de los generados.
    generados = [RUTA_BY_CELL, RUTA_SUMMARY, RUTA_SIGNATURES, RUTA_CONTROL_CMP, RUTA_SLICE_SUM,
                 RUTA_RESULTS, Path(__file__)]
    sucios = [r.name for r in generados if any(b > 127 for b in r.read_bytes())]
    if sucios:
        raise SystemExit(f"STOP: bytes > 127 en {sucios}")

    auditoria = auditar_self()
    permitidos = {"toxic_train.csv", "holdout_v2_ids.csv", "grupos_near_duplicados.csv",
                  "oof_v03_final.npz", "v2_oof_base_corrected.npz", "model_v03.joblib",
                  "fase2_lexico_inducido_agresividad.csv", "v03_error_casebook.csv",
                  "v03_error_cells.csv", "v03_error_controls.csv", "v03_error_review_sample.csv",
                  "p90_03_diagnostico_contribuciones.py", "p90_03b_informe_handoff.py",
                  "test.csv", "submission", "submissions", "leaderboard", "holdout"}
    raros = [v for v in auditoria["tokens"] if v not in permitidos]
    if raros:
        raise SystemExit(f"STOP: tokens de ruta inesperados: {raros}")
    prohibidos = {"transformers", "sentence_transformers", "openai", "anthropic"} & set(auditoria["modulos"])
    if prohibidos:
        raise SystemExit(f"STOP: imports prohibidos: {sorted(prohibidos)}")

    print(json.dumps({
        "celdas": len(filas_celda), "controles": len(filas_ctl),
        "atribucion_exacta": "BLOCKED_BY_MISSING_ARTIFACTS",
        "atribucion_auxiliar": "OK",
        "entrenamientos": 0, "submissions": 0,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
