"""FASE 6 / L3 — Separación contenido vs estilo (brief §10, línea prioritaria).

Observación
    Las fronteras centrales resisten (18-24|25-34: 0,486) y el análisis de términos mostró por qué:
    ambas cohortes comparten **contenido** (retuitean el mismo humor y los mismos deportes) y
    difieren sobre todo en **registro**. Sin embargo, el sistema actual mezcla contenido y estilo en
    una sola bolsa de n-gramas.

Hipótesis
    H31  Una representación **solo de estilo** (palabras funcionales, forma de las palabras,
         puntuación, estructura, marcadores de entidad sin su identidad) conserva señal demográfica
         que el modelo de contenido no captura.
    H32  Los dos canales son **complementarios**: añadidos como miembros al ensamble aportan más que
         cualquiera de ellos por separado.

Método
    - ``style_only``: se conservan palabras funcionales (lista cerrada en español), puntuación,
      dígitos, marcadores de entidad (``@USER``, ``#TAG``, ``<url>``) y la **forma** de las palabras
      de contenido (letras -> ``x``/``X`` según mayúsculas, dígitos -> ``9``); se descarta su identidad.
    - ``content_only``: se conservan todas las palabras de contenido y las identidades de entidades;
      se descartan las palabras funcionales.
    - Ambos se vectorizan con char_wb(3,6) + canal de menciones y se entrenan con LinearSVC(C=0.2).

Salida: ``experiments/phase6_contenido_estilo.json`` + matrices OOF para el ensamble.
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config as cfg  # noqa: E402
from src import cv, data  # noqa: E402
from src.config import set_seed  # noqa: E402
from src.textometry import SPANISH_STOPWORDS  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from phase2_ponderacion import build_features, evaluate_global  # noqa: E402

RESULTS_PATH = cfg.EXPERIMENTS / "phase6_contenido_estilo.json"
CHAR_CONFIG = {"analyzer": "char_wb", "ngram_range": (3, 6), "min_df": 5,
               "sublinear_tf": True, "max_features": 300_000, "lowercase": True}
MENTION_CONFIG = {"analyzer": "word", "ngram_range": (1, 1), "min_df": 2,
                  "sublinear_tf": True, "lowercase": True}
TOKEN_RE = re.compile(r"@\w+|#\w+|https?://\S+|[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]+|\d+|[^\s\w]")
FUNCTION_WORDS = SPANISH_STOPWORDS | {"rt", "via", "jajaja", "jeje", "jaja"}


def style_only(series: pd.Series) -> pd.Series:
    """Representación de estilo: función, forma, puntuación y marcadores sin identidad."""
    output: list[str] = []
    for value in series.fillna("").astype(str):
        pieces: list[str] = []
        for token in TOKEN_RE.findall(value):
            lowered = token.lower()
            if token.startswith("@"):
                pieces.append("@USER")
            elif token.startswith("#"):
                pieces.append("#TAG")
            elif lowered.startswith("http"):
                pieces.append("<url>")
            elif lowered in FUNCTION_WORDS:
                pieces.append(token)  # la palabra funcional es señal de estilo legítima
            elif token.isdigit():
                pieces.append("9" * len(token))
            else:
                # Se preserva la FORMA (longitud y patrón de mayúsculas), no la identidad.
                pieces.append("".join("X" if character.isupper() else
                                      ("9" if character.isdigit() else "x") for character in token))
        output.append(" ".join(pieces))
    return pd.Series(output, index=series.index)


def content_only(series: pd.Series) -> pd.Series:
    """Representación de contenido: palabras de contenido e identidades de entidad."""
    output: list[str] = []
    for value in series.fillna("").astype(str):
        pieces = [token for token in TOKEN_RE.findall(value)
                  if token.lower() not in SPANISH_STOPWORDS]
        output.append(" ".join(pieces))
    return pd.Series(output, index=series.index)


def mention_tokens(series: pd.Series) -> pd.Series:
    """Tokens de mención para el canal dedicado."""
    from src import preprocessing
    return preprocessing.collapse_spaces(
        series.str.findall(r"@\w+").apply(lambda values: " ".join(values).lower()))


def fit_member(train_texts: pd.Series, valid_texts: pd.Series) -> np.ndarray:
    """Ajusta un SVM sobre char_wb(3,6) + canal de menciones y devuelve los márgenes."""
    char = TfidfVectorizer(**CHAR_CONFIG)
    mentions = TfidfVectorizer(**MENTION_CONFIG)
    x_train = hstack([char.fit_transform(train_texts),
                      mentions.fit_transform(mention_tokens(train_texts))]).tocsr()
    x_valid = hstack([char.transform(valid_texts),
                      mentions.transform(mention_tokens(valid_texts))]).tocsr()
    return x_train, x_valid


def main() -> int:
    """Construye y evalúa las representaciones de estilo y contenido."""
    set_seed()
    started = time.time()
    train = data.load_train()
    y = train[cfg.TARGET].astype(str).to_numpy()
    clusters = pd.read_parquet(cfg.FEATURES / "dup_clusters.parquet")
    folds = cv.make_folds(y, groups=clusters["cluster"].to_numpy()[: len(y)],
                          n_splits=5, n_repeats=1, scheme="group")

    style_texts = style_only(train[cfg.TEXT_COL])
    content_texts = content_only(train[cfg.TEXT_COL])
    print("ejemplo de estilo :", style_texts.iloc[3][:110], flush=True)
    print("ejemplo de contenido:", content_texts.iloc[3][:110], flush=True)

    margins_style = np.zeros((len(y), len(cfg.CLASSES)), dtype=np.float32)
    margins_content = np.zeros((len(y), len(cfg.CLASSES)), dtype=np.float32)
    per_fold = {"estilo": [], "contenido": []}
    for fold in folds:
        train_idx, valid_idx = fold["train_idx"], fold["val_idx"]
        for name, texts, storage in (("estilo", style_texts, margins_style),
                                     ("contenido", content_texts, margins_content)):
            x_train, x_valid = fit_member(texts.iloc[train_idx], texts.iloc[valid_idx])
            model = LinearSVC(C=0.2, max_iter=3000)
            model.fit(x_train, y[train_idx])
            order = [list(model.classes_).index(label) for label in cfg.CLASSES]
            decision = model.decision_function(x_valid)[:, order]
            storage[valid_idx] = decision
            per_fold[name].append(float(f1_score(
                y[valid_idx], np.array(cfg.CLASSES)[np.argmax(decision, axis=1)],
                average="macro", labels=cfg.CLASSES)))
        print(f"[fold {fold['fold']}] estilo {per_fold['estilo'][-1]:.4f} | "
              f"contenido {per_fold['contenido'][-1]:.4f} ({round(time.time() - started, 1)}s)",
              flush=True)

    np.save(cfg.FEATURES / "oof_style_only.npy", margins_style)
    np.save(cfg.FEATURES / "oof_content_only.npy", margins_content)

    print("\nevaluando el aporte al ensamble con el meta-modelo MLP (LOFO)...", flush=True)
    features, y2, _, _, _ = build_features()
    assert np.array_equal(y, y2)
    base_mean, base_std, base_folds = evaluate_global(features, y, folds, "mlp")
    with_both = np.concatenate([features, margins_style, margins_content], axis=1)
    both_mean, both_std, both_folds = evaluate_global(with_both, y, folds, "mlp")
    with_style = np.concatenate([features, margins_style], axis=1)
    style_mean, style_std, style_folds = evaluate_global(with_style, y, folds, "mlp")

    difference = np.array(both_folds) - np.array(base_folds)
    payload = {
        "estilo_individual": {"macro_f1": round(float(np.mean(per_fold["estilo"])), 4),
                              "std": round(float(np.std(per_fold["estilo"])), 4),
                              "per_fold": [round(v, 4) for v in per_fold["estilo"]]},
        "contenido_individual": {"macro_f1": round(float(np.mean(per_fold["contenido"])), 4),
                                 "std": round(float(np.std(per_fold["contenido"])), 4),
                                 "per_fold": [round(v, 4) for v in per_fold["contenido"]]},
        "referencia_svm_completo": 0.5658,
        "mlp_base": {"macro_f1": base_mean, "std": base_std, "per_fold": base_folds},
        "mlp_con_estilo": {"macro_f1": style_mean, "std": style_std, "per_fold": style_folds},
        "mlp_con_ambos": {"macro_f1": both_mean, "std": both_std, "per_fold": both_folds},
        "delta_ambos": round(both_mean - base_mean, 4),
        "folds_con_mejora": int((difference > 0).sum()),
        "mejora_por_fold": difference.round(4).tolist(),
        "segundos": round(time.time() - started, 1),
    }
    RESULTS_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== L3 — CONTENIDO VS ESTILO ===")
    print(f"estilo solo    : {payload['estilo_individual']['macro_f1']:.4f} "
          f"± {payload['estilo_individual']['std']:.4f}")
    print(f"contenido solo : {payload['contenido_individual']['macro_f1']:.4f} "
          f"± {payload['contenido_individual']['std']:.4f}")
    print(f"referencia (svm con todo el texto): 0,5658")
    print(f"MLP base       : {base_mean:.4f} | + estilo: {style_mean:.4f} "
          f"| + ambos: {both_mean:.4f} ({payload['delta_ambos']:+.4f}, "
          f"{payload['folds_con_mejora']}/5 folds)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
