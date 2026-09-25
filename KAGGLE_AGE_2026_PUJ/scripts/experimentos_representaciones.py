"""E-002..E-007 — Matriz controlada de representaciones.

Pregunta
    ¿Qué rangos de n-gramas (palabra y carácter), qué normalización y qué esquema de pesado
    maximizan macro-F1 bajo el protocolo agrupado por clústeres?

Hipótesis evaluadas
    H6  Los char n-grams aportan señal ortográfica complementaria a las palabras.
    H7  El léxico temático discrimina clases (word n-grams útiles, pero con techo).
    H8  La normalización (URLs/menciones/hashtags/RT) cambia el desempeño.
    H4  Parte de la ganancia por añadir features es redundancia (se vigila el número de
        features frente a la mejora real).

Protocolo
    - Modelo fijo (LinearSVC C=1) para aislar el efecto de la **representación**.
    - Validación: StratifiedGroupKFold, 5 folds, 1 repetición (exploración), agrupado por
      clúster de near-duplicados. La configuración ganadora se re-evaluará con 3 repeticiones.
    - Vectorizadores ajustados solo con el train de cada fold.
    - El script es reanudable: omite las configuraciones ya presentes en el JSON de resultados.

Salidas
    ``experiments/representaciones_resultados.json`` y una fila por configuración en
    ``experiments.csv``.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config as cfg  # noqa: E402
from src import cv, data, preprocessing  # noqa: E402
from src.config import set_seed  # noqa: E402

RESULTS_PATH = cfg.EXPERIMENTS / "representaciones_resultados.json"
MODEL_CONFIG = {"C": 1.0, "max_iter": 3000}

BASE_WORD = {"analyzer": "word", "min_df": 3, "sublinear_tf": True,
             "max_features": 300_000, "lowercase": True}
BASE_CHAR = {"analyzer": "char_wb", "min_df": 5, "sublinear_tf": True,
             "max_features": 300_000, "lowercase": True}

CONFIGS: list[dict] = [
    {"id": "E-002a", "nombre": "word 1-gram", "texto": "crudo",
     "vectores": [dict(BASE_WORD, ngram_range=(1, 1))],
     "hipotesis": "El unigrama de palabra captura el léxico temático básico."},
    {"id": "E-002b", "nombre": "word 1-2 gram", "texto": "crudo",
     "vectores": [dict(BASE_WORD, ngram_range=(1, 2))],
     "hipotesis": "Los bigramas añaden colocaciones y fórmulas de registro."},
    {"id": "E-002c", "nombre": "word 1-3 gram", "texto": "crudo",
     "vectores": [dict(BASE_WORD, ngram_range=(1, 3))],
     "hipotesis": "Los trigramas aportan contexto, con riesgo de dispersión."},
    {"id": "E-003a", "nombre": "char_wb 2-4", "texto": "crudo",
     "vectores": [dict(BASE_CHAR, ngram_range=(2, 4))],
     "hipotesis": "Rangos cortos de carácter capturan morfología y abreviaturas."},
    {"id": "E-003b", "nombre": "char_wb 3-5", "texto": "crudo",
     "vectores": [dict(BASE_CHAR, ngram_range=(3, 5))],
     "hipotesis": "El rango 3-5 es el estándar de la literatura para author profiling."},
    {"id": "E-003c", "nombre": "char_wb 3-6", "texto": "crudo",
     "vectores": [dict(BASE_CHAR, ngram_range=(3, 6))],
     "hipotesis": "Rangos más largos capturan palabras completas y sufijos."},
    {"id": "E-005a", "nombre": "union word 1-2 + char 3-5 (normalizado)", "texto": "normalizado",
     "vectores": [dict(BASE_WORD, ngram_range=(1, 2)), dict(BASE_CHAR, ngram_range=(3, 5))],
     "hipotesis": "Quitar URLs/menciones/RT y unificar minúsculas reduce ruido y mejora la señal."},
    {"id": "E-007a", "nombre": "union sin IDF (TF puro)", "texto": "crudo",
     "vectores": [dict(BASE_WORD, ngram_range=(1, 2), use_idf=False),
                  dict(BASE_CHAR, ngram_range=(3, 5), use_idf=False)],
     "hipotesis": "El IDF aporta poco cuando las clases están balanceadas y el vocabulario es estable."},
    # --- Descomposición de la normalización (explica la caída de E-005a) ---
    {"id": "E-005b", "nombre": "marcadores <url>/<user> + union", "texto": "marcadores",
     "vectores": [dict(BASE_WORD, ngram_range=(1, 2)), dict(BASE_CHAR, ngram_range=(3, 5))],
     "hipotesis": ("Sustituir URLs y menciones por marcadores conserva la señal de presencia y "
                   "posición sin arrastrar el contenido específico.")},
    {"id": "E-005c", "nombre": "solo quitar URLs + union", "texto": "sin_urls",
     "vectores": [dict(BASE_WORD, ngram_range=(1, 2)), dict(BASE_CHAR, ngram_range=(3, 5))],
     "hipotesis": "Si quitar solo las URLs ya degrada, la presencia de enlaces es señal en sí misma."},
    {"id": "E-005d", "nombre": "solo quitar menciones + union", "texto": "sin_menciones",
     "vectores": [dict(BASE_WORD, ngram_range=(1, 2)), dict(BASE_CHAR, ngram_range=(3, 5))],
     "hipotesis": "Las menciones son señal de interacción social; eliminarlas debería costar desempeño."},
    {"id": "E-005e", "nombre": "solo quitar puntuación + union", "texto": "sin_punct",
     "vectores": [dict(BASE_WORD, ngram_range=(1, 2)), dict(BASE_CHAR, ngram_range=(3, 5))],
     "hipotesis": "La puntuación (especialmente ! y ?) es señal estilométrica; quitarla debería costar."},
    # --- Follow-ups dirigidos por el hallazgo de menciones ---
    {"id": "E-004b", "nombre": "char_wb 3-6 + word 1-gram", "texto": "crudo",
     "vectores": [dict(BASE_CHAR, ngram_range=(3, 6)), dict(BASE_WORD, ngram_range=(1, 1))],
     "hipotesis": ("Si el unigrama de palabra (el mejor rango de palabra) complementa a los char "
                   "n-grams sin diluirlos, la unión superará al char solo.")},
    {"id": "E-008", "nombre": "char_wb 3-6 + canal de menciones", "texto": "crudo",
     "vectores": [dict(BASE_CHAR, ngram_range=(3, 6))],
     "extra": {"texto": "menciones", "vector": dict(BASE_WORD, ngram_range=(1, 1), min_df=2)},
     "hipotesis": ("La identidad de las cuentas mencionadas es señal fuerte: un canal TF-IDF "
                   "dedicado a los tokens de mención (además del texto completo) debería aportar "
                   "sobre el char n-gram solo.")},
]


def load_texts(train: pd.DataFrame) -> dict[str, pd.Series]:
    """Devuelve las variantes de texto disponibles para los experimentos.

    Incluye la descomposición de la normalización (quitar solo URLs, solo menciones, solo
    prefijo RT, solo puntuación) y la variante con marcadores ``<url>``/``<user>``, para
    aislar **qué** transformación ayuda o daña.
    """
    text = train[cfg.TEXT_COL]
    return {
        "crudo": text,
        "normalizado": preprocessing.light_normalize(text),
        "sin_urls": preprocessing.remove_urls(text),
        "sin_menciones": preprocessing.remove_mentions(text),
        "sin_punct": preprocessing.remove_punctuation(text),
        "marcadores": preprocessing.url_mention_markers(text),
        "menciones": preprocessing.collapse_spaces(
            text.str.findall(r"@\w+").apply(lambda values: " ".join(values).lower())),
        "menciones_hashtags": preprocessing.collapse_spaces(
            (text.str.findall(r"@\w+").apply(lambda values: " ".join(values).lower()) + " "
             + text.str.findall(r"#\w+").apply(lambda values: " ".join(values).lower()))),
    }


def run_config(config: dict, train: pd.DataFrame, y: np.ndarray, folds: list[dict]) -> dict:
    """Ejecuta una configuración de representación sobre todos los folds."""
    texts_by_variant = load_texts(train)
    channels: list[tuple[pd.Series, dict]] = [
        (texts_by_variant[config["texto"]], vector_config) for vector_config in config["vectores"]
    ]
    if config.get("extra"):
        channels.append((texts_by_variant[config["extra"]["texto"]], config["extra"]["vector"]))

    rows: list[dict] = []
    per_class: list[dict] = []
    n_features = 0
    for fold in folds:
        started = time.time()
        train_idx, valid_idx = fold["train_idx"], fold["val_idx"]
        x_train_parts, x_valid_parts = [], []
        for channel_texts, vectorizer_config in channels:
            vectorizer = TfidfVectorizer(**vectorizer_config)
            x_train_parts.append(vectorizer.fit_transform(channel_texts.iloc[train_idx]))
            x_valid_parts.append(vectorizer.transform(channel_texts.iloc[valid_idx]))
        x_train = hstack(x_train_parts).tocsr()
        x_valid = hstack(x_valid_parts).tocsr()
        n_features = x_train.shape[1]

        model = LinearSVC(**MODEL_CONFIG)
        model.fit(x_train, y[train_idx])
        predictions = model.predict(x_valid)

        rows.append({
            "fold": fold["fold"],
            "macro_f1": f1_score(y[valid_idx], predictions, average="macro"),
            "weighted_f1": f1_score(y[valid_idx], predictions, average="weighted"),
            "micro_f1": f1_score(y[valid_idx], predictions, average="micro"),
            "accuracy": accuracy_score(y[valid_idx], predictions),
            "seconds": round(time.time() - started, 1),
        })
        class_scores = f1_score(y[valid_idx], predictions, average=None, labels=cfg.CLASSES)
        per_class.append(dict(zip(cfg.CLASSES, class_scores, strict=True)))

    frame = pd.DataFrame(rows)
    summary = {
        "id": config["id"],
        "nombre": config["nombre"],
        "texto": config["texto"],
        "hipotesis": config["hipotesis"],
        "n_features": int(n_features),
        "macro_f1_mean": float(frame["macro_f1"].mean()),
        "macro_f1_std": float(frame["macro_f1"].std()),
        "weighted_f1_mean": float(frame["weighted_f1"].mean()),
        "micro_f1_mean": float(frame["micro_f1"].mean()),
        "accuracy_mean": float(frame["accuracy"].mean()),
        "seconds_total": round(float(frame["seconds"].sum()), 1),
        "per_class_f1": {label: float(np.mean([row[label] for row in per_class]))
                         for label in cfg.CLASSES},
        "per_fold": rows,
    }
    return summary


def register(summary: dict, registry_path: Path) -> None:
    """Añade la configuración evaluada a ``experiments.csv``."""
    existing = pd.read_csv(registry_path) if registry_path.exists() else pd.DataFrame()
    if len(existing) and summary["id"] in set(existing["experiment_id"].astype(str)):
        return
    row = {
        "experiment_id": summary["id"], "fecha": time.strftime("%Y-%m-%d"), "bloque": "2",
        "etapa": "B - representacion",
        "pregunta": f"¿Qué desempeño da la representación '{summary['nombre']}'?",
        "hipotesis": summary["hipotesis"], "features": "texto",
        "preprocessing": summary["texto"], "representacion": summary["nombre"],
        "modelo": "LinearSVC", "hiperparametros": json.dumps(MODEL_CONFIG),
        "validacion": "StratifiedGroupKFold 5 folds agrupado por clúster",
        "macro_f1_mean": round(summary["macro_f1_mean"], 5),
        "macro_f1_std": round(summary["macro_f1_std"], 5),
        "weighted_f1_mean": round(summary["weighted_f1_mean"], 5),
        "micro_f1_mean": round(summary["micro_f1_mean"], 5),
        "accuracy_mean": round(summary["accuracy_mean"], 5),
        "resultado": f"macro-F1 {summary['macro_f1_mean']:.4f} ± {summary['macro_f1_std']:.4f} "
                     f"({summary['n_features']:,} features)",
        "interpretacion": "pendiente de la comparación global de la matriz de representaciones",
        "decision": "pendiente", "duracion_seg": summary["seconds_total"],
        "artefactos": "experiments/representaciones_resultados.json",
    }
    updated = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
    updated.to_csv(registry_path, index=False)


def main() -> int:
    """Ejecuta la matriz de representaciones de forma reanudable."""
    set_seed()
    train = data.load_train()
    clusters = pd.read_parquet(cfg.FEATURES / "dup_clusters.parquet")
    cluster_ids = clusters["cluster"].to_numpy()[: len(train)]
    y = train[cfg.TARGET].astype(str).to_numpy()
    folds = cv.make_folds(y, groups=cluster_ids, n_splits=5, n_repeats=1, scheme="group")

    results = json.loads(RESULTS_PATH.read_text(encoding="utf-8")) if RESULTS_PATH.exists() else {}
    registry_path = cfg.ROOT / "experiments.csv"

    for config in CONFIGS:
        if config["id"] in results:
            print(f"[skip] {config['id']} ya evaluado", flush=True)
            continue
        print(f"[run ] {config['id']} — {config['nombre']}", flush=True)
        summary = run_config(config, train, y, folds)
        results[config["id"]] = summary
        RESULTS_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
        register(summary, registry_path)
        print(f"        macro-F1 {summary['macro_f1_mean']:.4f} ± {summary['macro_f1_std']:.4f} "
              f"| features {summary['n_features']:,} | {summary['seconds_total']}s", flush=True)

    table = pd.DataFrame([
        {"id": key, "nombre": value["nombre"], "texto": value["texto"],
         "macro_f1": round(value["macro_f1_mean"], 4), "std": round(value["macro_f1_std"], 4),
         "weighted": round(value["weighted_f1_mean"], 4),
         "accuracy": round(value["accuracy_mean"], 4), "features": value["n_features"]}
        for key, value in results.items()
    ]).sort_values("macro_f1", ascending=False)
    table.to_csv(cfg.TABLES / "representaciones_comparativa.csv", index=False)
    print("\n=== MATRIZ DE REPRESENTACIONES ===")
    print(table.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
