"""Muestra estratificada y reproducible del corpus."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.rag.config import settings
from src.rag.corpus import normalize_language


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=settings.parquet_path)
    parser.add_argument("--out", default="eval/pilot_docs.parquet")
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = pq.read_table(
        args.input, columns=["document_id", "collection_name", "language"]
    ).to_pandas()
    df["language_iso"] = df["language"].map(normalize_language)
    df["stratum"] = (
        df["collection_name"].fillna("sin_coleccion").astype(str)
        + "|"
        + df["language_iso"].fillna("xx")
    )

    rng = np.random.default_rng(args.seed)
    total = len(df)
    parts = []
    for _, group in df.groupby("stratum"):
        quota = max(1, round(len(group) / total * args.n))
        quota = min(quota, len(group))
        index = rng.choice(group.index.to_numpy(), size=quota, replace=False)
        parts.append(df.loc[index])

    sample = pd.concat(parts, ignore_index=True)
    if len(sample) > args.n:
        sample = sample.sample(args.n, random_state=args.seed)
    elif len(sample) < args.n:
        rest = df[~df["document_id"].isin(sample["document_id"])]
        extra = rest.sample(min(args.n - len(sample), len(rest)), random_state=args.seed)
        sample = pd.concat([sample, extra], ignore_index=True)

    sample = sample.drop(columns=["stratum"]).sort_values("document_id").reset_index(drop=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    sample.to_parquet(args.out, index=False)

    summary = {
        "n": len(sample),
        "seed": args.seed,
        "origen": args.input,
        "salida": args.out,
        "por_coleccion": {str(k): int(v) for k, v in sample["collection_name"].value_counts().items()},
        "por_idioma": {str(k): int(v) for k, v in sample["language_iso"].value_counts(dropna=False).items()},
    }
    Path("reports/tablas").mkdir(parents=True, exist_ok=True)
    Path("reports/tablas/muestra.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary["por_coleccion"], ensure_ascii=False))


if __name__ == "__main__":
    main()
