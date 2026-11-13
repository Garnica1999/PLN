"""Perfilado del corpus: integridad, distribuciones y figuras."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import pyarrow.parquet as pq

from src.rag.config import settings
from src.rag.corpus import normalize_language
from src.rag.plotting import ACCENT, BLUE, WARM, apply_theme
from src.rag.plotting import save_figure as save_plot

COLUMNS = [
    "document_id",
    "text",
    "collection_name",
    "title",
    "url",
    "source",
    "source_type",
    "language",
    "created_at",
    "url_domain",
    "word_count",
    "char_count",
    "text_sha256",
]


def native(value):
    if hasattr(value, "item"):
        return value.item()
    return value


def save_figure(fig, path: Path) -> None:
    fig.tight_layout()
    save_plot(fig, path)
    plt.close(fig)


def main() -> None:
    apply_theme()
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=settings.parquet_path)
    parser.add_argument("--outdir", default="reports")
    parser.add_argument("--hash-sample", type=int, default=200)
    args = parser.parse_args()

    out = Path(args.outdir)
    tables = out / "tablas"
    figures = out / "figuras"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    df = pq.read_table(args.input, columns=COLUMNS).to_pandas()
    df["language_iso"] = df["language"].map(normalize_language)
    df["created_date"] = pd.to_datetime(df["created_at"], errors="coerce", utc=True)
    df["year"] = df["created_date"].dt.year

    n = len(df)
    duplicates = int(df["text_sha256"].duplicated().sum())
    groups = df.groupby("text_sha256").size()
    duplicated_groups = int((groups > 1).sum())
    max_copies = int(groups.max()) if len(groups) else 0

    sample = df.sample(min(args.hash_sample, n), random_state=42)
    hash_ok = sum(
        hashlib.sha256(str(text).encode("utf-8")).hexdigest() == str(digest)
        for text, digest in zip(sample["text"], sample["text_sha256"])
    )

    stats = {
        "archivo": args.input,
        "filas": n,
        "columnas_parquet": pq.ParquetFile(args.input).metadata.num_columns,
        "tamano_mb": round(Path(args.input).stat().st_size / 1e6, 1),
        "ids_unicos": bool(df["document_id"].is_unique),
        "texto_vacio": int((df["text"].fillna("").str.strip() == "").sum()),
        "duplicados_texto": duplicates,
        "grupos_duplicados": duplicated_groups,
        "max_copias_mismo_texto": max_copies,
        "duplicados_por_coleccion": {
            str(k): int(v)
            for k, v in df[df["text_sha256"].duplicated(keep=False)]["collection_name"].value_counts().items()
        },
        "hash_verificado": {"muestra": len(sample), "coinciden": int(hash_ok)},
        "colecciones": {str(k): int(v) for k, v in df["collection_name"].value_counts(dropna=False).items()},
        "source_type": {str(k): int(v) for k, v in df["source_type"].value_counts(dropna=False).head(10).items()},
        "nulos_pct": {str(k): round(float(v) * 100, 2) for k, v in df[COLUMNS].isna().mean().items()},
        "idioma_crudo_top": {str(k): int(v) for k, v in df["language"].value_counts(dropna=False).head(20).items()},
        "idioma_normalizado": {str(k): int(v) for k, v in df["language_iso"].value_counts(dropna=False).head(20).items()},
        "sin_idioma": int(df["language_iso"].isna().sum()),
        "word_count": {k: float(native(v)) for k, v in df["word_count"].describe(percentiles=[0.5, 0.9, 0.99]).items()},
        "char_count": {k: float(native(v)) for k, v in df["char_count"].describe(percentiles=[0.5, 0.9, 0.99]).items()},
        "fechas": {
            "min": str(df["created_date"].min()),
            "max": str(df["created_date"].max()),
            "nulas": int(df["created_date"].isna().sum()),
            "por_anio": {str(int(k)): int(v) for k, v in df["year"].value_counts().sort_index().items()},
        },
        "dominios_top": {str(k): int(v) for k, v in df["url_domain"].value_counts().head(20).items()},
    }

    (tables / "perfilado.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    flat = {
        "filas": n,
        "duplicados_texto": duplicates,
        "grupos_duplicados": duplicated_groups,
        "texto_vacio": stats["texto_vacio"],
        "sin_idioma": stats["sin_idioma"],
        "word_mediana": stats["word_count"]["50%"],
        "word_p99": stats["word_count"]["99%"],
        "char_mediana": stats["char_count"]["50%"],
        "char_p99": stats["char_count"]["99%"],
    }
    pd.DataFrame(flat.items(), columns=["metrica", "valor"]).to_csv(tables / "perfilado.csv", index=False)

    counts = df["collection_name"].value_counts()
    fig, ax = plt.subplots(figsize=(6, 3))
    counts.plot(kind="bar", ax=ax, color=BLUE)
    ax.set_title("Documentos por colección")
    ax.set_ylabel("Documentos")
    save_figure(fig, figures / "perfil_colecciones.png")

    fig, ax = plt.subplots(figsize=(6, 3))
    df["word_count"].clip(upper=3000).plot(kind="hist", bins=50, ax=ax, color=BLUE)
    ax.set_title("Distribución de longitud (palabras, tope 3000)")
    ax.set_xlabel("Palabras")
    save_figure(fig, figures / "perfil_longitud.png")

    top_lang = df["language_iso"].value_counts().head(15)
    fig, ax = plt.subplots(figsize=(7, 3))
    top_lang.plot(kind="bar", ax=ax, color=ACCENT)
    ax.set_title("Idioma normalizado (top 15)")
    ax.set_ylabel("Documentos")
    save_figure(fig, figures / "perfil_idiomas.png")

    years = df["year"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(7, 3.2))
    years.plot(kind="bar", ax=ax, color=WARM)
    ax.set_title("Documentos por año")
    ax.set_ylabel("Documentos")
    ax.set_xlabel("Año")
    ax.set_xticks(range(len(years)))
    ax.set_xticklabels([str(int(year)) for year in years.index], rotation=90, fontsize=7)
    save_figure(fig, figures / "perfil_anios.png")

    print(json.dumps({"filas": n, "duplicados": duplicates, "sin_idioma": stats["sin_idioma"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
