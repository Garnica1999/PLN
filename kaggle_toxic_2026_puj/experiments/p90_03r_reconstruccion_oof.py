"""P90-03-R: reconstruccion del pipeline historico por fold y atribucion exacta OOF.

Autorizado por la fase P90.03-R: reconstruye los modelos y vectorizadores de cada
fold con los mismos datos, parametros y procedimiento de `experiments/14_oof_corregido_v2.py`,
verifica que reproducen las probabilidades OOF canonicas (gate estricto predefinido)
y, SOLO si el gate pasa, calcula las contribuciones exactas por fold.

No cambia V03, thresholds, folds ni protocolo. No usa test ni holdout. No genera
submissions. No hace commits. Las nuevas probabilidades se guardan en rutas
diagnosticas independientes.

Uso:
    /home/carlos/venv/bin/python experiments/p90_03r_reconstruccion_oof.py
"""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

RAIZ: Path = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from src.data import read_train  # noqa: E402
from src.evaluation import metricas_multietiqueta  # noqa: E402
from src.splits import ETIQUETAS, RANDOM_STATE, grupos_fusionados, pliegues_corregidos  # noqa: E402

RUTA_OOF = RAIZ / "features" / "oof_v03_final.npz"
RUTA_V2BASE = RAIZ / "features" / "v2_oof_base_corrected.npz"
RUTA_TRAIN = RAIZ / "data" / "raw" / "toxic_train.csv"
RUTA_PARTICION = RAIZ / "data" / "processed" / "holdout_v2_ids.csv"
RUTA_SCRIPT14 = RAIZ / "experiments" / "14_oof_corregido_v2.py"
RUTA_SPLITS = RAIZ / "src" / "splits.py"
RUTA_DATA = RAIZ / "src" / "data.py"
RUTA_LEXICO = RAIZ / "reports" / "eda" / "fase2_lexico_inducido_agresividad.csv"

RUTA_REPRO = RAIZ / "reports" / "p90_03r_oof_reproduction.csv"
RUTA_CONTRIB = RAIZ / "reports" / "p90_03r_feature_contributions.csv"
RUTA_ATRIB = RAIZ / "reports" / "p90_03r_exact_oof_attribution.csv"
RUTA_RESULTS = RAIZ / "reports" / "p90_03r_audit_results.json"

HASH_OOF = "559265816ce4f6312c87f008f90324f0eec70a74d1e6247364c162793338feab"
HASH_V2BASE = "27dfa83736b979518a94bc93e32bb9cae07bc82eaff92234fe9e15d1f9822774"
HASH_SCRIPT14 = "f0e8065bfbbc210e3a05f5b9"
HASH_SPLITS = "755d3cef5b06c5859c654df7"
HASH_DATA = "79c1f04d2be4bae6d36a694d"
HASH_TRAIN = "3d40e7345b588860cf4744914be88ddc094e4f03591516c8176ab40b7f76d42f"
UMBRALES = (0.44, 0.51, 0.46)
CCV_CANONICA = 0.7772114525165721
ESTRATOS = (("cerca", 0.0, 0.025), ("intermedio", 0.025, 0.100), ("lejos", 0.100, float("inf")))
TOPK = 10

# Tolerancias PREDEFINIDAS antes de ejecutar la comparacion (no se amplian despues).
RTOL = 1e-7
ATOL = 1e-8
TOL_CCV = 1e-7

PARAMS_WORD = {"analyzer": "word", "ngram_range": (1, 2), "min_df": 2, "sublinear_tf": True}
PARAMS_CHAR = {"analyzer": "char_wb", "ngram_range": (3, 5), "min_df": 2, "sublinear_tf": True}
LR_C2 = {"C": 2.0, "class_weight": "balanced", "max_iter": 1000, "solver": "liblinear",
         "random_state": RANDOM_STATE}

URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
MENTION_PATTERN = re.compile(r"@\w+")
HASHTAG_PATTERN = re.compile(r"#\w+")
INITIAL_MENTION_PATTERN = re.compile(r"^\s*(?:rt\s+)?@\w+", re.IGNORECASE)
UPPER_WORD_PATTERN = re.compile(r"\b[A-Z]{2,}\b")
ELONGATION_PATTERN = re.compile(r"([A-Za-z])\1{2,}")
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
    print(f"[P90-03R] {mensaje}", flush=True)


def sha256_de(ruta: Path) -> str:
    """Calcula el sha256 de un archivo."""
    digest = hashlib.sha256()
    with ruta.open("rb") as archivo:
        for bloque in iter(lambda: archivo.read(1 << 20), b""):
            digest.update(bloque)
    return digest.hexdigest()


def sha256_bytes(datos: bytes) -> str:
    """Calcula el sha256 de un bloque de bytes."""
    return hashlib.sha256(datos).hexdigest()


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


def logit(p: float) -> float:
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
    """Carga y valida el OOF canonico, el train de desarrollo y el procedimiento historico."""
    hashes = {
        "oof": sha256_de(RUTA_OOF), "v2base": sha256_de(RUTA_V2BASE), "train": sha256_de(RUTA_TRAIN),
        "script14": sha256_de(RUTA_SCRIPT14), "splits": sha256_de(RUTA_SPLITS),
        "data": sha256_de(RUTA_DATA), "lexico": sha256_de(RUTA_LEXICO),
    }
    if hashes["oof"] != HASH_OOF or hashes["v2base"] != HASH_V2BASE or hashes["train"] != HASH_TRAIN:
        raise SystemExit("STOP: hash inesperado en artefactos canonicos")
    if not hashes["script14"].startswith(HASH_SCRIPT14) or not hashes["splits"].startswith(HASH_SPLITS):
        raise SystemExit("STOP: los scripts historicos cambiaron desde la auditoria")
    if not hashes["data"].startswith(HASH_DATA):
        raise SystemExit("STOP: src/data.py cambio desde la auditoria")

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
    if sorted(set(int(p) for p in pliegues)) != [1, 2, 3, 4, 5]:
        raise SystemExit("STOP: los folds del OOF no son 1..5")
    if float(np.min(proba)) <= 0.0 or float(np.max(proba)) >= 1.0:
        raise SystemExit("STOP: probabilidades fuera de (0,1) en el OOF canonico")
    if not np.array_equal(pred, (proba >= umbrales[None, :]).astype(np.int64)):
        raise SystemExit("STOP: pred no reproduce proba >= umbrales")

    # Linaje M2 (revalidado).
    v2 = np.load(RUTA_V2BASE, allow_pickle=True)
    pos_m2 = [str(m) for m in v2["modelos"]].index("M2_union_lr_c2")
    orden = {int(i): k for k, i in enumerate(v2["ids"])}
    if float(np.abs(v2["proba"][[orden[i] for i in ids], pos_m2, :] - proba).max()) != 0.0:
        raise SystemExit("STOP: el linaje M2 no reproduce la proba canonica")

    # Reconstruccion del train_dev con el procedimiento exacto del script 14.
    train = read_train()
    particion = pd.read_csv(RUTA_PARTICION)
    ids_dev = particion.loc[particion["particion"] == "dev", "id"].astype(int).tolist()
    train_dev = train[train["id"].astype(int).isin(ids_dev)].reset_index(drop=True)
    if len(train_dev) != 1483:
        raise SystemExit("STOP: train_dev no tiene 1483 filas")
    if train_dev["id"].astype(int).tolist() != ids:
        raise SystemExit("STOP: el orden de train_dev no coincide con el OOF canonico")
    if not np.array_equal(train_dev[list(ETIQUETAS)].to_numpy(), y):
        raise SystemExit("STOP: y_true de train_dev no coincide con el OOF canonico")

    # Verificacion informativa: recomputar los pliegues con el procedimiento historico.
    grupos_full = grupos_fusionados(train)
    mapa_grupo = dict(zip(train["id"].astype(int), grupos_full.tolist()))
    grupos_dev = np.array([mapa_grupo[i] for i in train_dev["id"].astype(int)])
    pliegues_re = pliegues_corregidos(train_dev, grupos_dev)
    fold_re = np.zeros(len(train_dev), dtype=int)
    for numero, (_, idx_val) in enumerate(pliegues_re, start=1):
        fold_re[idx_val] = numero
    folds_iguales = bool(np.array_equal(fold_re, pliegues))

    return {"hashes": hashes, "ids": ids, "y": y, "pliegues": pliegues, "proba": proba,
            "pred": pred, "umbrales": umbrales, "train_dev": train_dev, "grupos_dev": grupos_dev,
            "folds_iguales": folds_iguales}


def reconstruir_folds(g: dict[str, Any]) -> dict[str, Any]:
    """Reconstruye los modelos y vectorizadores de cada fold con el procedimiento historico."""
    train_dev = g["train_dev"]
    pliegues = g["pliegues"]
    proba_recon = np.zeros((len(train_dev), 3), dtype=float)
    modelos_fold: dict[int, dict[str, Any]] = {}
    checksums: dict[str, str] = {}
    for fold in range(1, 6):
        idx_val = np.where(pliegues == fold)[0]
        idx_tra = np.where(pliegues != fold)[0]
        word = TfidfVectorizer(**PARAMS_WORD)
        char = TfidfVectorizer(**PARAMS_CHAR)
        x_tra = sparse.hstack([
            word.fit_transform(train_dev["text"].iloc[idx_tra]),
            char.fit_transform(train_dev["text"].iloc[idx_tra]),
        ]).tocsr()
        x_val = sparse.hstack([
            word.transform(train_dev["text"].iloc[idx_val]),
            char.transform(train_dev["text"].iloc[idx_val]),
        ]).tocsr()
        modelos = []
        for indice, etiqueta in enumerate(ETIQUETAS):
            modelo = LogisticRegression(**LR_C2)
            modelo.fit(x_tra, train_dev[etiqueta].iloc[idx_tra])
            proba_recon[idx_val, indice] = modelo.predict_proba(x_val)[:, 1]
            modelos.append(modelo)
            checksums[f"fold{fold}_{etiqueta}"] = sha256_bytes(
                np.asarray(modelo.coef_[0], dtype=np.float64).tobytes()
                + np.asarray(modelo.intercept_, dtype=np.float64).tobytes()
            )[:16]
        nombres = np.array([f"word::{n}" for n in word.get_feature_names_out()]
                           + [f"char::{n}" for n in char.get_feature_names_out()])
        modelos_fold[fold] = {"modelos": modelos, "nombres": nombres, "x_val": x_val,
                              "idx_val": idx_val, "k_word": len(word.get_feature_names_out())}
        log(f"  fold {fold}: train {len(idx_tra)} | val {len(idx_val)} | "
            f"features {x_tra.shape[1]} (word {len(word.get_feature_names_out())})")
    return {"proba_recon": proba_recon, "modelos_fold": modelos_fold, "checksums": checksums}


def evaluar_gate(g: dict[str, Any], r: dict[str, Any]) -> dict[str, Any]:
    """Gate de reproduccion con tolerancias predefinidas (no ampliables)."""
    proba_recon = r["proba_recon"]
    canon = g["proba"]
    dif = np.abs(proba_recon - canon)
    allclose = bool(np.allclose(proba_recon, canon, rtol=RTOL, atol=ATOL))
    pred_recon = (proba_recon >= g["umbrales"][None, :]).astype(np.int64)
    discordancias = int((pred_recon != g["pred"]).sum())
    ccv_recon = float(metricas_multietiqueta(g["y"], pred_recon)["f1_media_columnas"])
    dif_ccv = abs(ccv_recon - CCV_CANONICA)
    por_fold = {f"fold{f}": float(dif[g['pliegues'] == f].max()) for f in range(1, 6)}
    por_etiqueta = {et: float(dif[:, j].max()) for j, et in enumerate(ETIQUETAS)}
    gate_ok = (allclose and discordancias == 0 and dif_ccv <= TOL_CCV and g["folds_iguales"])
    return {"allclose_rtol_1e-7_atol_1e-8": allclose, "max_abs_error": float(dif.max()),
            "mean_abs_error": float(dif.mean()), "p99_abs_error": float(np.percentile(dif, 99)),
            "binary_discordances": discordancias, "ccv_reconstructed": ccv_recon,
            "ccv_difference": dif_ccv, "tolerance_ccv": TOL_CCV,
            "max_error_by_fold": por_fold, "max_error_by_label": por_etiqueta,
            "folds_recomputed_equal": g["folds_iguales"], "gate_passed": bool(gate_ok)}


def contribuciones_exactas(g: dict[str, Any], r: dict[str, Any],
                           lexico: frozenset[str]) -> list[dict[str, Any]]:
    """Calcula las contribuciones exactas por fold para todas las filas de validacion."""
    train_dev = g["train_dev"]
    proba = g["proba"]
    distancia = np.abs(proba - g["umbrales"][None, :])
    error = g["pred"] != g["y"]
    filas = []
    for fold in range(1, 6):
        info = r["modelos_fold"][fold]
        nombres = info["nombres"]
        k_word = info["k_word"]
        for j, etiqueta in enumerate(ETIQUETAS):
            modelo = info["modelos"][j]
            coef = np.asarray(modelo.coef_[0], dtype=np.float64)
            intercept = float(modelo.intercept_[0])
            x_val = info["x_val"]
            proba_val = modelo.predict_proba(x_val)[:, 1]
            for pos, i in enumerate(info["idx_val"]):
                contrib = x_val.getrow(pos).multiply(coef[None, :]).tocsr()
                fila_c = contrib.getrow(0)
                idxs = fila_c.indices
                vals = fila_c.data
                word_sum = float(vals[idxs < k_word].sum())
                char_sum = float(vals[idxs >= k_word].sum())
                pos_sum = float(vals[vals > 0].sum())
                neg_sum = float(vals[vals < 0].sum())
                z = intercept + word_sum + char_sum
                p = float(proba_val[pos])
                resid = abs(z - logit(p))
                orden = np.argsort(-np.abs(vals))
                top_pos = [int(t) for t in orden if vals[t] > 0][:TOPK]
                top_neg = [int(t) for t in orden if vals[t] < 0][:TOPK]

                def formato(t: int) -> str:
                    nombre = str(nombres[idxs[t]])
                    canal, limpio = nombre.split("::", 1)
                    tfidf = float(x_val[pos, idxs[t]])
                    return f"{limpio}|{canal}|{tfidf:.6f}|{coef[idxs[t]]:.4f}|{vals[t]:.6f}"

                top3 = float(np.abs(vals[orden][:3]).sum())
                abs_total = float(np.abs(vals).sum())
                top3_share = top3 / abs_total if abs_total else 0.0
                es_fp = bool(g["y"][i, j] == 0)
                cats_pos = [categoria_feature(str(nombres[idxs[t]]).split("::", 1)[1],
                                              str(nombres[idxs[t]]).startswith("word::"), lexico)
                            for t in top_pos[:10]]
                tiene_lexico = "insultos y vulgaridad" in cats_pos
                tiene_contexto = any(c in ("menciones", "expresiones interpersonales",
                                           "colectivos y referencias grupales") for c in cats_pos)
                if es_fp:
                    top1_cat = categoria_feature(str(nombres[idxs[top_pos[0]]]).split("::", 1)[1],
                                                 str(nombres[idxs[top_pos[0]]]).startswith("word::"),
                                                 lexico) if top_pos else ""
                    regla = top1_cat == "insultos y vulgaridad" and not tiene_contexto
                    regla_nombre = "lexico dominante sin contexto"
                else:
                    regla = not tiene_lexico
                    regla_nombre = "sin evidencia hostil activada"
                conflicto_wc = word_sum * char_sum < 0
                ratio = (min(pos_sum, -neg_sum) / max(pos_sum, -neg_sum)
                         if (pos_sum > 0 and neg_sum < 0) else 0.0)
                conflicto_pn = pos_sum > 0 and neg_sum < 0 and ratio >= 0.3
                if conflicto_wc:
                    principal = "conflicto word-char"
                elif top3_share >= 0.5:
                    principal = "pocas features dominantes"
                elif top3_share < 0.25:
                    principal = "acumulacion de muchas features"
                elif conflicto_pn:
                    principal = "conflicto positivas-negativas"
                elif regla:
                    principal = regla_nombre
                else:
                    principal = "no interpretable con los artefactos disponibles"
                filas.append({
                    "example_id": int(g["ids"][i]), "fold": int(fold), "label": etiqueta,
                    "y_true": int(g["y"][i, j]), "oof_probability": f"{p:.8f}",
                    "historical_threshold": f"{g['umbrales'][j]:.2f}",
                    "predicted_label": int(g["pred"][i, j]),
                    "error_type": ("FP" if g["y"][i, j] == 0 else "FN") if error[i, j] else "",
                    "distance_to_threshold": f"{float(distancia[i, j]):.8f}",
                    "logit": f"{z:.8f}", "intercept": f"{intercept:.8f}",
                    "word_contribution_sum": f"{word_sum:.8f}",
                    "char_contribution_sum": f"{char_sum:.8f}",
                    "contribution_sum": f"{word_sum + char_sum:.8f}",
                    "decomposition_residual": f"{resid:.2e}",
                    "n_active_features": int(len(vals)),
                    "top3_abs_share": f"{top3_share:.4f}",
                    "word_char_conflict": int(conflicto_wc),
                    "pos_neg_conflict": int(conflicto_pn),
                    "signature_primary": principal,
                    "top_positive_features": "; ".join(formato(t) for t in top_pos),
                    "top_negative_features": "; ".join(formato(t) for t in top_neg),
                })
    return filas


def categoria_feature(nombre: str, es_word: bool, lexico: frozenset[str]) -> str:
    """Categoria descriptiva de una feature (misma definicion que P90.03)."""
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
    """Reconstruye, verifica el gate y (si pasa) calcula atribuciones exactas."""
    log("auditoria de fuentes y gate de estructura")
    g = cargar_gate()
    log(f"  dev {len(g['ids'])} | folds 1..5 | orden y y_true verificados | "
        f"pliegues recomputados == persistidos: {g['folds_iguales']}")

    log("reconstruccion de los 5 folds con el pipeline historico")
    r = reconstruir_folds(g)

    log("gate de reproduccion (tolerancias predefinidas)")
    gate = evaluar_gate(g, r)
    log(f"  allclose(rtol={RTOL}, atol={ATOL}): {gate['allclose_rtol_1e-7_atol_1e-8']} | "
        f"max {gate['max_abs_error']:.3e} | media {gate['mean_abs_error']:.3e} | "
        f"p99 {gate['p99_abs_error']:.3e} | discordancias {gate['binary_discordances']}")
    log(f"  ccv reconstruido {gate['ccv_reconstructed']:.16f} | dif {gate['ccv_difference']:.3e} "
        f"(tolerancia {TOL_CCV})")

    # Tabla de reproduccion por fila y etiqueta (siempre).
    filas_repro = []
    for i, ident in enumerate(g["ids"]):
        for j, et in enumerate(ETIQUETAS):
            filas_repro.append({
                "example_id": int(ident), "fold": int(g["pliegues"][i]), "label": et,
                "y_true": int(g["y"][i, j]),
                "oof_probability_canonical": f"{float(g['proba'][i, j]):.8f}",
                "oof_probability_reconstructed": f"{float(r['proba_recon'][i, j]):.8f}",
                "abs_error": f"{float(abs(r['proba_recon'][i, j] - g['proba'][i, j])):.3e}",
                "threshold": f"{g['umbrales'][j]:.2f}",
                "predicted_canonical": int(g["pred"][i, j]),
                "predicted_reconstructed": int(r["proba_recon"][i, j] >= g["umbrales"][j]),
                "discordant": int(bool((r["proba_recon"][i, j] >= g["umbrales"][j]) != bool(g["pred"][i, j]))),
            })
    escribir_csv(RUTA_REPRO, filas_repro)

    if not gate["gate_passed"]:
        RUTA_RESULTS.write_text(json.dumps({
            "audit_id": "P90-03-R", "gate": gate, "decision": "EXACT_OOF_REPRODUCTION_FAILED",
            "hashes": g["hashes"], "checksums": r["checksums"],
        }, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        log("STOP = EXACT_OOF_REPRODUCTION_FAILED: no se calculan atribuciones")
        return 2

    log("gate superado: calculo de contribuciones exactas por fold")
    lexico = frozenset(str(f["lema"]).strip() for f in leer_csv(RUTA_LEXICO)
                       if str(f["seleccionado"]).strip() == "1")
    filas_contrib = contribuciones_exactas(g, r, lexico)
    escribir_csv(RUTA_CONTRIB, filas_contrib)
    filas_atrib = [f for f in filas_contrib if f["error_type"]]
    escribir_csv(RUTA_ATRIB, filas_atrib)
    log(f"contribuciones: {len(filas_contrib)} filas-etiqueta | errores: {len(filas_atrib)}")

    # Verificaciones de recuento canonico y descomposicion.
    resid_max = max(float(f["decomposition_residual"]) for f in filas_contrib)
    error = g["pred"] != g["y"]
    distancia = np.abs(g["proba"] - g["umbrales"][None, :])
    estratos = {et: [int((error[:, j] & (distancia[:, j] >= lo) & (distancia[:, j] < hi)).sum())
                     for _, lo, hi in ESTRATOS] for j, et in enumerate(ETIQUETAS)}
    if int(error.sum()) != 757 or int(error.any(axis=1).sum()) != 441:
        raise SystemExit("STOP: recuentos de error no reproducen 757/441")
    if resid_max > 1e-9:
        raise SystemExit(f"STOP: la descomposicion no reconstruye el logit ({resid_max})")

    # Agregados exactos para el informe (mismos campos que P90.03, ahora exactos).
    def stats_de(f: dict[str, Any]) -> dict[str, Any]:
        tops = [t.split("|") for t in f["top_positive_features"].split("; ") if t]
        negs = [t.split("|") for t in f["top_negative_features"].split("; ") if t]
        cats = [categoria_feature(t[0], t[1] == "word", lexico) for t in tops[:10]]
        return {"cats_pos": cats, "top1": tops[0][0] if tops else "",
                "top1_cat": cats[0] if cats else "",
                "neg_names": [t[0] for t in negs[:5]],
                "word": float(f["word_contribution_sum"]), "char": float(f["char_contribution_sum"]),
                "n_active": int(f["n_active_features"]),
                "top3": float(f["top3_abs_share"]), "conf_wc": int(f["word_char_conflict"]),
                "conf_pn": int(f["pos_neg_conflict"]), "firma": f["signature_primary"]}

    detalle = {(int(f["example_id"]), f["label"]): stats_de(f) for f in filas_contrib}
    idx = {int(ident): k for k, ident in enumerate(g["ids"])}
    agregados: dict[str, Any] = {}
    firmas_conteo: Counter = Counter()
    for f in filas_contrib:
        if f["error_type"]:
            firmas_conteo[(f["label"], f["error_type"], f["signature_primary"])] += 1
    for et in ETIQUETAS:
        j = ETIQUETAS.index(et)
        for tipo in ("FP", "FN"):
            sel = [(i, j) for i in range(len(g["ids"])) if error[i, j]
                   and ((g["y"][i, j] == 0) if tipo == "FP" else (g["y"][i, j] == 1))]
            c_pos: Counter = Counter()
            c_neg: Counter = Counter()
            for (i, _) in sel:
                st = detalle[(int(g["ids"][i]), et)]
                if st["top1"]:
                    c_pos[st["top1"]] += 1
                for n in st["neg_names"]:
                    c_neg[n] += 1
            w = np.array([detalle[(int(g["ids"][i]), et)]["word"] for (i, _) in sel])
            c = np.array([detalle[(int(g["ids"][i]), et)]["char"] for (i, _) in sel])
            agregados[f"{et}|{tipo}"] = {
                "n": len(sel),
                "top1_pos_es_lexico": sum(1 for (i, _) in sel
                                          if detalle[(int(g["ids"][i]), et)]["top1_cat"] == "insultos y vulgaridad"),
                "lexico_en_top10_pos": sum(1 for (i, _) in sel
                                           if "insultos y vulgaridad" in detalle[(int(g["ids"][i]), et)]["cats_pos"]),
                "conflicto_wc": sum(1 for (i, _) in sel if detalle[(int(g["ids"][i]), et)]["conf_wc"]),
                "char_dominante": sum(1 for (i, _) in sel
                                      if detalle[(int(g["ids"][i]), et)]["char"] > 0 and
                                      detalle[(int(g["ids"][i]), et)]["char"] /
                                      (abs(detalle[(int(g["ids"][i]), et)]["word"]) +
                                       abs(detalle[(int(g["ids"][i]), et)]["char"]) + 1e-12) >= 0.6),
                "mismo_signo": sum(1 for (i, _) in sel
                                   if np.sign(detalle[(int(g["ids"][i]), et)]["word"]) ==
                                   np.sign(detalle[(int(g["ids"][i]), et)]["char"])),
                "pocas_dominantes": sum(1 for (i, _) in sel if detalle[(int(g["ids"][i]), et)]["top3"] >= 0.5),
                "acumulacion": sum(1 for (i, _) in sel if detalle[(int(g["ids"][i]), et)]["top3"] < 0.25),
                "mean_word": float(w.mean()), "mean_char": float(c.mean()),
                "mean_abs_word": float(np.abs(w).mean()), "mean_abs_char": float(np.abs(c).mean()),
                "top_features_pos": "; ".join(f"{n}:{v}" for n, v in c_pos.most_common(12)),
                "top_features_neg": "; ".join(f"{n}:{v}" for n, v in c_neg.most_common(12)),
            }
    # Casos clave para la comparacion con P90.03.
    casos_clave: dict[str, dict[str, int]] = {"perra_top1_FP": {}, "la_puta_en_neg5": {},
                                              "mi_perra_en_neg5": {}}
    for et in ETIQUETAS:
        j = ETIQUETAS.index(et)
        fp_idx = [i for i in range(len(g["ids"])) if error[i, j] and g["y"][i, j] == 0]
        err_idx = [i for i in range(len(g["ids"])) if error[i, j]]
        casos_clave["perra_top1_FP"][et] = sum(
            1 for i in fp_idx if detalle[(int(g["ids"][i]), et)]["top1"] == "perra")
        casos_clave["la_puta_en_neg5"][et] = sum(
            1 for i in err_idx if "la puta" in detalle[(int(g["ids"][i]), et)]["neg_names"])
        casos_clave["mi_perra_en_neg5"][et] = sum(
            1 for i in err_idx if "mi perra" in detalle[(int(g["ids"][i]), et)]["neg_names"])
    firmas = {f"{e}|{t}|{firma}": v for (e, t, firma), v in firmas_conteo.items()}

    # Slices textuales (definiciones auditadas en P90.02/P90.03; mismo codigo).
    import spacy
    nlp = spacy.load("es_core_news_md", exclude=["ner"])
    n_filas = len(g["ids"])
    banderas = {k: np.zeros(n_filas, dtype=bool) for k in ("colectiva", "segunda", "imperativo")}
    for i, doc in enumerate(nlp.pipe(g["train_dev"]["text"].tolist(), batch_size=64)):
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
    textos = g["train_dev"]["text"].tolist()
    tiene = lambda patron: np.array([bool(patron.search(t)) for t in textos])
    mayusculas = np.array([bool(UPPER_WORD_PATTERN.search(t)) or sum(1 for c in t if c.isupper()) >= 4
                           for t in textos])
    slices = {
        "URLs": tiene(URL_PATTERN), "hashtags": tiene(HASHTAG_PATTERN),
        "colectivos_superficie": banderas["colectiva"], "menciones": tiene(MENTION_PATTERN),
        "mencion_inicial": tiene(INITIAL_MENTION_PATTERN), "mayusculas_sostenidas": mayusculas,
        "elongaciones": tiene(ELONGATION_PATTERN), "segunda_persona": banderas["segunda"],
        "imperativo": banderas["imperativo"], "cita": tiene(re.compile('"')),
    }
    filas_slice = []
    for nombre, mask in slices.items():
        for et in ETIQUETAS:
            j = ETIQUETAS.index(et)
            dentro = [i for i in range(n_filas) if mask[i] and error[i, j]]
            fuera = [i for i in range(n_filas) if (not mask[i]) and error[i, j]]

            def resumen_sel(sel: list[int], etiqueta: str = et) -> dict[str, Any]:
                if not sel:
                    return {"n": 0, "word": 0.0, "char": 0.0, "act": 0.0, "top1": ""}
                w = float(np.mean([detalle[(int(g["ids"][i]), etiqueta)]["word"] for i in sel]))
                c = float(np.mean([detalle[(int(g["ids"][i]), etiqueta)]["char"] for i in sel]))
                act = float(np.mean([detalle[(int(g["ids"][i]), etiqueta)]["n_active"] for i in sel]))
                cat = Counter(detalle[(int(g["ids"][i]), etiqueta)]["top1_cat"] for i in sel
                              if detalle[(int(g["ids"][i]), etiqueta)]["top1_cat"])
                return {"n": len(sel), "word": w, "char": c, "act": act,
                        "top1": "; ".join(f"{k}:{v}" for k, v in cat.most_common(2))}

            den, fue = resumen_sel(dentro), resumen_sel(fuera)
            filas_slice.append({
                "slice": nombre, "label": et, "n_rows_slice": int(mask.sum()),
                "n_errors_inside": den["n"], "mean_word_inside": round(den["word"], 6),
                "mean_char_inside": round(den["char"], 6), "mean_active_inside": round(den["act"], 2),
                "mean_word_outside": round(fue["word"], 6), "mean_char_outside": round(fue["char"], 6),
                "mean_active_outside": round(fue["act"], 2),
                "dominant_inside": den["top1"], "dominant_outside": fue["top1"],
            })

    resultados = {
        "audit_id": "P90-03-R",
        "gate": gate,
        "decision": "EXACT_OOF_ATTRIBUTION_RECOVERED",
        "hashes": g["hashes"], "checksums_coeficientes": r["checksums"],
        "estratos": estratos,
        "decomposition_residual_max": resid_max,
        "agregados": agregados,
        "firmas": firmas,
        "casos_clave": casos_clave,
        "slices": filas_slice,
        "celdas_error": len(filas_atrib),
        "filas_contrib": len(filas_contrib),
    }
    RUTA_RESULTS.write_text(json.dumps(resultados, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

    generados = [RUTA_REPRO, RUTA_CONTRIB, RUTA_ATRIB, RUTA_RESULTS, Path(__file__)]
    sucios = [p.name for p in generados if any(b > 127 for b in p.read_bytes())]
    if sucios:
        raise SystemExit(f"STOP: bytes > 127 en {sucios}")

    auditoria = auditar_self()
    permitidos = {"toxic_train.csv", "holdout_v2_ids.csv", "oof_v03_final.npz",
                  "v2_oof_base_corrected.npz", "14_oof_corregido_v2.py", "splits.py", "data.py",
                  "fase2_lexico_inducido_agresividad.csv", "p90_03r_reconstruccion_oof.py",
                  "test.csv", "submission", "submissions", "leaderboard", "holdout"}
    raros = [v for v in auditoria["tokens"] if v not in permitidos]
    if raros:
        raise SystemExit(f"STOP: tokens de ruta inesperados: {raros}")
    prohibidos = {"transformers", "sentence_transformers", "openai", "anthropic"} & set(auditoria["modulos"])
    if prohibidos:
        raise SystemExit(f"STOP: imports prohibidos: {sorted(prohibidos)}")

    print(json.dumps({
        "gate": "PASSED", "max_abs_error": gate["max_abs_error"],
        "discordancias": gate["binary_discordances"], "ccv_dif": gate["ccv_difference"],
        "celdas_error": len(filas_atrib), "decision": "EXACT_OOF_ATTRIBUTION_RECOVERED",
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
