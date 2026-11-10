"""Bootstrap pareado por pregunta para comparar dos corridas.

Uso:
    python -m src.scripts.bootstrap_eval --a run_a.jsonl --b run_b.jsonl --field correct
    python -m src.scripts.bootstrap_eval --a ragas_a.csv --b ragas_b.csv \
        --field answer_correctness --kind csv
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
from pathlib import Path


def load_jsonl(path: Path, field: str) -> dict[str, float]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return {row["id"]: float(row[field]) for row in rows if row.get(field) is not None}


def load_csv(path: Path, field: str) -> dict[str, float]:
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    return {
        row["user_input"]: float(row[field])
        for row in rows
        if row.get(field) not in (None, "")
    }


def paired_bootstrap(a: list[float], b: list[float], samples: int = 10000, seed: int = 42) -> dict:
    """Diferencia media (a - b), IC 95% por percentiles y p de dos colas."""
    rng = random.Random(seed)
    diffs = [left - right for left, right in zip(a, b)]
    observed = statistics.mean(diffs)
    means = []
    for _ in range(samples):
        drawn = [diffs[rng.randrange(len(diffs))] for _ in range(len(diffs))]
        means.append(statistics.mean(drawn))
    means.sort()
    low = means[int(0.025 * samples)]
    high = means[int(0.975 * samples) - 1]
    below = sum(1 for value in means if value <= 0) / samples
    above = sum(1 for value in means if value >= 0) / samples
    p_value = min(1.0, 2 * min(below, above))
    return {
        "n": len(diffs),
        "mean_a": round(statistics.mean(a), 4),
        "mean_b": round(statistics.mean(b), 4),
        "diff": round(observed, 4),
        "ci_low": round(low, 4),
        "ci_high": round(high, 4),
        "p_value": round(p_value, 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", required=True)
    parser.add_argument("--b", required=True)
    parser.add_argument("--field", default="correct")
    parser.add_argument("--kind", default="jsonl", choices=["jsonl", "csv"])
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    loader = load_jsonl if args.kind == "jsonl" else load_csv
    left = loader(Path(args.a), args.field)
    right = loader(Path(args.b), args.field)
    keys = sorted(set(left) & set(right))
    if not keys:
        raise SystemExit("no hay claves en comun")
    result = paired_bootstrap([left[key] for key in keys], [right[key] for key in keys], args.samples)
    result.update({"field": args.field, "a": args.a, "b": args.b})
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
