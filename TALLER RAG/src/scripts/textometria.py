"""Fase 1: textometría descriptiva del corpus y del piloto."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.rag.config import settings
from src.rag.corpus import normalize_language
from src.rag.plotting import ACCENT, BLUE, WARM, apply_theme
from src.rag.plotting import save_figure as save_plot
from src.rag.quality import is_junk
from src.rag.textometry import (
    document_metrics,
    exclusive_terms,
    hamming_neighbors,
    simhash64,
    tokenize,
    top_terms,
)

NOISE_COLUMNS = ["html", "urls", "emails", "emojis", "truncated"]


def stratified_sample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    df = df.copy()
    df["language_iso"] = df["language"].map(normalize_language)
    df["stratum"] = df["collection_name"].fillna("sin_coleccion").astype(str) + "|" + df["language_iso"].fillna("xx")
    rng = np.random.default_rng(seed)
    parts = []
    for _, group in df.groupby("stratum"):
        quota = max(1, round(len(group) / len(df) * n))
        quota = min(quota, len(group))
        index = rng.choice(group.index.to_numpy(), size=quota, replace=False)
        parts.append(df.loc[index])
    sample = pd.concat(parts, ignore_index=True)
    if len(sample) > n:
        sample = sample.sample(n, random_state=seed)
    return sample


def cluster_stats(pairs: list[tuple[int, int]], size: int) -> dict:
    parent = list(range(size))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in pairs:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
    roots = Counter(find(i) for i in range(size))
    involved = {i for pair in pairs for i in pair}
    clusters = [count for count in roots.values() if count > 1]
    return {
        "pares": len(pairs),
        "documentos_con_par": len(involved),
        "pct_documentos_con_par": round(len(involved) / max(size, 1) * 100, 2),
        "grupos": len(clusters),
        "grupo_max": max(clusters) if clusters else 0,
        "grupo_mediano": float(np.median(clusters)) if clusters else 0.0,
    }


def metrics_frame(data: pd.DataFrame, label: str) -> pd.DataFrame:
    rows = []
    for record in data.itertuples(index=False):
        metrics = document_metrics(record.text)
        metrics.update(
            {
                "set": label,
                "document_id": record.document_id,
                "collection_name": record.collection_name,
                "language": normalize_language(record.language),
                "year": record.year,
                "boilerplate": int(is_junk(record.text)),
                "simhash": f"{simhash64(record.text):016x}",
            }
        )
        rows.append(metrics)
    return pd.DataFrame(rows)


def zipf_points(texts: list[str], limit: int = 50) -> list[list[int]]:
    counter = Counter()
    for text in texts:
        counter.update(tokenize(text))
    return [[rank, count] for rank, (_, count) in enumerate(counter.most_common(limit), start=1)]


def save_figure(fig, path: Path) -> None:
    fig.tight_layout()
    save_plot(fig, path)
    plt.close(fig)


def main() -> None:
    apply_theme()
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=settings.parquet_path)
    parser.add_argument("--pilot", default="eval/pilot_docs.parquet")
    parser.add_argument("--sample", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outdir", default="reports")
    args = parser.parse_args()

    tables = Path(args.outdir) / "tablas"
    figures = Path(args.outdir) / "figuras"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    columns = ["document_id", "text", "collection_name", "language", "created_at", "text_sha256"]
    df = pq.read_table(args.input, columns=columns).to_pandas()
    df["year"] = pd.to_datetime(df["created_at"], errors="coerce", utc=True).dt.year
    df["language_iso"] = df["language"].map(normalize_language)

    pilot_ids = set(pd.read_parquet(args.pilot)["document_id"].tolist())
    pilot = df[df["document_id"].isin(pilot_ids)].reset_index(drop=True)
    corpus = stratified_sample(df[~df["document_id"].isin(pilot_ids)], args.sample, args.seed)

    pilot_metrics = metrics_frame(pilot, "piloto")
    corpus_metrics = metrics_frame(corpus, "muestra")
    all_metrics = pd.concat([pilot_metrics, corpus_metrics], ignore_index=True)
    all_metrics.to_csv(tables / "textometria.csv", index=False)

    hashes_pilot = np.array([int(value, 16) for value in pilot_metrics["simhash"]], dtype=np.uint64)
    hashes_corpus = np.array([int(value, 16) for value in corpus_metrics["simhash"]], dtype=np.uint64)
    pairs_pilot = hamming_neighbors(hashes_pilot)
    pairs_corpus = hamming_neighbors(hashes_corpus)
    near_pilot = cluster_stats(pairs_pilot, len(pilot_metrics))
    near_corpus = cluster_stats(pairs_corpus, len(corpus_metrics))

    def rates(frame: pd.DataFrame) -> dict:
        return {
            column: round(float((frame[column] > 0).mean()) * 100, 2) for column in NOISE_COLUMNS
        } | {
            "navegacion": round(float((frame["nav_ratio"] > 0.01).mean()) * 100, 2),
            "boilerplate": round(float(frame["boilerplate"].mean()) * 100, 2),
            "code_switching": round(float(frame["code_switching"].mean()) * 100, 2),
            "vacio": round(float(frame["empty"].mean()) * 100, 2),
        }

    def lexicon(frame: pd.DataFrame) -> dict:
        return {
            "types_mediana": float(frame["types"].median()),
            "ttr_mediana": round(float(frame["ttr"].median()), 4),
            "hapax_ratio_mediana": round(float(frame["hapax_ratio"].median()), 4),
            "guiraud_mediana": round(float(frame["guiraud"].median()), 4),
            "chars_mediana": float(frame["chars"].median()),
            "chars_p99": float(frame["chars"].quantile(0.99)),
        }

    collection_texts = defaultdict(list)
    for record in corpus.itertuples(index=False):
        collection_texts[record.collection_name].append(record.text)

    terms_by_collection = {
        name: top_terms(texts, 20) for name, texts in collection_texts.items()
    }
    terms_by_year = {}
    for year, group in corpus.groupby("year"):
        if pd.isna(year):
            continue
        terms_by_year[str(int(year))] = top_terms(group["text"].tolist(), 15)

    bluesky = corpus[corpus["collection_name"] == "bluesky_posts"]["text"].tolist()
    press = corpus[corpus["collection_name"] == "url_crawls"]["text"].tolist()

    payload = {
        "corpus": {
            "documentos": len(df),
            "muestra": len(corpus),
            "semilla": args.seed,
            "idiomas": {str(k): int(v) for k, v in corpus["language_iso"].value_counts(dropna=False).head(15).items()},
        },
        "piloto": {"documentos": len(pilot)},
        "lexico": {"piloto": lexicon(pilot_metrics), "muestra": lexicon(corpus_metrics)},
        "ruido": {"piloto": rates(pilot_metrics), "muestra": rates(corpus_metrics)},
        "near_duplicates": {"piloto": near_pilot, "muestra": near_corpus},
        "zipf": {"piloto": zipf_points(pilot["text"].tolist()), "muestra": zipf_points(corpus["text"].tolist())},
        "terminos": {
            "por_coleccion": terms_by_collection,
            "por_anio": terms_by_year,
            "exclusivos_bluesky": exclusive_terms(bluesky, press),
            "exclusivos_prensa": exclusive_terms(press, bluesky),
        },
    }
    (tables / "textometria.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(6, 4))
    for label, frame in (("piloto", pilot_metrics), ("muestra", corpus_metrics)):
        values = frame["chars"].clip(lower=50)
        ax.hist(values, bins=60, alpha=0.55, label=label)
    ax.set_xscale("log")
    ax.set_xlabel("Caracteres por documento (log)")
    ax.set_ylabel("Documentos")
    ax.set_title("Longitud de los documentos")
    ax.legend()
    save_figure(fig, figures / "textometria_longitudes.png")

    fig, ax = plt.subplots(figsize=(6, 4))
    for label, points in (("piloto", payload["zipf"]["piloto"]), ("muestra", payload["zipf"]["muestra"])):
        ranks = [p[0] for p in points]
        counts = [p[1] for p in points]
        ax.plot(ranks, counts, marker="o", markersize=2, label=label)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Rango")
    ax.set_ylabel("Frecuencia")
    ax.set_title("Distribución de Zipf")
    ax.legend()
    save_figure(fig, figures / "textometria_zipf.png")

    labels = ["HTML", "URLs", "emails", "emojis", "truncado", "navegación", "boilerplate"]
    pilot_values = [
        payload["ruido"]["piloto"][key] for key in ["html", "urls", "emails", "emojis", "truncated", "navegacion", "boilerplate"]
    ]
    corpus_values = [
        payload["ruido"]["muestra"][key] for key in ["html", "urls", "emails", "emojis", "truncated", "navegacion", "boilerplate"]
    ]
    positions = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(positions - 0.2, pilot_values, width=0.4, label="piloto")
    ax.bar(positions + 0.2, corpus_values, width=0.4, label="muestra")
    ax.set_xticks(positions, labels, rotation=30)
    ax.set_ylabel("% de documentos")
    ax.set_title("Ruido textual")
    ax.legend()
    save_figure(fig, figures / "textometria_ruido.png")

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(
        ["piloto", "muestra"],
        [near_pilot["pct_documentos_con_par"], near_corpus["pct_documentos_con_par"]],
        color=WARM,
    )
    ax.set_ylabel("% documentos con near-duplicate")
    ax.set_title("Near-duplicates (SimHash, distancia <= 3)")
    save_figure(fig, figures / "textometria_near_dups.png")

    fig, ax = plt.subplots(figsize=(6, 4))
    languages = pd.Series(payload["corpus"]["idiomas"]).head(10)
    languages.plot(kind="bar", ax=ax, color=ACCENT)
    ax.set_ylabel("Documentos")
    ax.set_title("Idioma normalizado en la muestra")
    save_figure(fig, figures / "textometria_idiomas.png")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for axis, key, title in (
        (axes[0], "exclusivos_bluesky", "Exclusivos de Bluesky"),
        (axes[1], "exclusivos_prensa", "Exclusivos de prensa"),
    ):
        terms = payload["terminos"][key][:10]
        names = [term for term, *_ in terms][::-1]
        scores = [score for _, score, *_ in terms][::-1]
        axis.barh(names, scores, color=BLUE)
        axis.set_title(title)
        axis.set_xlabel("Tasa relativa")
    save_figure(fig, figures / "textometria_terminos.png")

    print(
        json.dumps(
            {
                "muestra": len(corpus),
                "piloto": len(pilot),
                "near_dup_piloto": near_pilot["pct_documentos_con_par"],
                "near_dup_muestra": near_corpus["pct_documentos_con_par"],
                "boilerplate_muestra": payload["ruido"]["muestra"]["boilerplate"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
