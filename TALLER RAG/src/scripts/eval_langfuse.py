"""Corre el set dorado contra la API y lo registra como experimento en Langfuse.

Uso:
    python -m src.scripts.eval_langfuse --run baseline-hibrido
    python -m src.scripts.eval_langfuse --run graph-hibrido --retriever graph_hybrid
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
import unicodedata
from pathlib import Path

import requests
from langfuse import Evaluation, Langfuse

STOPWORDS = {
    "de", "la", "el", "los", "las", "del", "al", "en", "y", "o", "un", "una",
    "que", "por", "para", "con", "se", "su", "es", "son", "fue", "será",
    "the", "of", "and", "to", "in", "a", "is", "was",
}


def normalize(text) -> str:
    """Minúsculas, sin tildes y sin puntuación."""
    value = unicodedata.normalize("NFKD", str(text or ""))
    value = "".join(char for char in value if not unicodedata.combining(char))
    value = value.lower()
    value = re.sub(r"[^a-z0-9 ]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def tokens(text: str) -> list[str]:
    return [word for word in normalize(text).split() if len(word) > 2 and word not in STOPWORDS]


def exact_match(answer: str, gold: str) -> float:
    """1.0 si el gold aparece literal (normalizado) en la respuesta."""
    gold = str(gold or "")
    if not gold:
        return 0.0
    return 1.0 if normalize(gold) in normalize(answer) else 0.0


def token_recall(answer: str, gold: str) -> float:
    """Fracción de palabras del gold presentes en la respuesta."""
    gold_tokens = tokens(str(gold or ""))
    if not gold_tokens:
        return 0.0
    answer_tokens = set(tokens(answer))
    return sum(1 for word in gold_tokens if word in answer_tokens) / len(gold_tokens)


def numeric_match(answer: str, gold: str) -> float:
    """Para respuestas numéricas: todos los números del gold están en la respuesta."""
    gold = str(gold or "")
    numbers = re.findall(r"\d+", gold)
    if not numbers:
        return 0.0
    found = set(re.findall(r"\d+", answer))
    return 1.0 if all(number in found for number in numbers) else 0.0


def is_correct(answer: str, gold: str) -> float:
    """Mezcla de coincidencia literal, numérica y cobertura de tokens."""
    score = max(exact_match(answer, gold), numeric_match(answer, gold))
    if score == 1.0:
        return 1.0
    return 1.0 if token_recall(answer, gold) >= 0.8 else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--silver", default="eval/silver/silver_standard.jsonl")
    parser.add_argument("--api", default=os.getenv("API_URL", "http://localhost:8010"))
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--dataset", default="silver-taller")
    parser.add_argument("--split", default="")
    parser.add_argument("--prompt-version", default="")
    parser.add_argument("--parent-expansion", default="auto", choices=["auto", "on", "off"])
    parser.add_argument("--query-expansion", default="auto", choices=["auto", "on", "off"])
    parser.add_argument("--graph-boost-weight", type=float, default=None)
    parser.add_argument("--model", default="")
    parser.add_argument("--run", default=None)
    parser.add_argument("--retriever", default="", help="solo se guarda en metadata")
    parser.add_argument("--outdir", default="reports")
    args = parser.parse_args()

    api = args.api.rstrip("/")
    run = args.run or f"eval-{time.strftime('%Y%m%d-%H%M')}"
    rows = [
        json.loads(line)
        for line in Path(args.silver).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if args.split:
        rows = [row for row in rows if row.get("split") == args.split]
    if args.limit:
        rows = rows[: args.limit]

    client = Langfuse(
        host=os.getenv("LANGFUSE_HOST", "http://localhost:3000"),
        public_key=os.getenv("LANGFUSE_PUBLIC_KEY", ""),
        secret_key=os.getenv("LANGFUSE_SECRET_KEY", ""),
    )
    try:
        client.create_dataset(name=args.dataset)
    except Exception:  # noqa: BLE001  # ya existe
        print("dataset ya existe", flush=True)
    for row in rows:
        item_id = f"{args.split}-{row['id']}" if args.split else row["id"]
        try:
            client.create_dataset_item(
                dataset_name=args.dataset,
                id=item_id,
                input={"question": row["question"]},
                expected_output=row.get("reference_answer", ""),
                metadata={
                    "type": row.get("type"),
                    "split": row.get("split"),
                    "span": row.get("reference_span", ""),
                    "original_id": row["id"],
                },
            )
        except Exception as error:  # noqa: BLE001
            print(f"item {row['id']}: {error}", flush=True)

    dataset = client.get_dataset(args.dataset)
    captured: list[dict] = []

    def task(*, item, **kwargs):
        question = item.input["question"] if isinstance(item.input, dict) else item.input
        payload = {
            "query": question,
            "k": args.k,
            "temperature": 0.0,
            "ambiguity_policy": "off",
            "retriever": args.retriever or None,
            "prompt_version": args.prompt_version or None,
            "parent_expansion": {"auto": None, "on": True, "off": False}[args.parent_expansion],
            "query_expansion": {"auto": None, "on": True, "off": False}[args.query_expansion],
            "graph_boost_weight": args.graph_boost_weight,
            "model": args.model or None,
        }
        started = time.perf_counter()
        try:
            response = requests.post(f"{api}/ask", json=payload, timeout=900)
            response.raise_for_status()
            data = response.json()
        except Exception as error:  # noqa: BLE001
            data = {"answer": "", "error": str(error)}
        wall_ms = round((time.perf_counter() - started) * 1000, 2)
        output = {
            "answer": data.get("answer", ""),
            "citations": data.get("citations", []),
            "grounded": not (data.get("grounding") or {}).get("ungrounded"),
            "abstained": bool(data.get("abstained")),
            "trace_id": data.get("trace_id"),
            "latency_ms": wall_ms,
        }
        meta = item.metadata or {}
        captured.append(
            {
                "id": meta.get("original_id", item.id),
                "question": question,
                "gold": item.expected_output or "",
                "answer": output["answer"],
                "contexts": [hit.get("text", "") for hit in data.get("evidence", [])],
                "correct": is_correct(output["answer"], item.expected_output or "")
                if item.expected_output
                else (1.0 if output["abstained"] else 0.0),
                "token_recall": round(token_recall(output["answer"], item.expected_output or ""), 4),
                "grounded": 1.0 if output["grounded"] else 0.0,
                "latency_ms": wall_ms,
                "trace_id": output["trace_id"],
            }
        )
        return output

    def correctness(*, output, expected_output, **kwargs):
        gold = expected_output or ""
        value = is_correct(output.get("answer", ""), gold) if gold else (1.0 if output.get("abstained") else 0.0)
        return Evaluation(name="correct", value=value)

    def recall(*, output, expected_output, **kwargs):
        return Evaluation(name="token_recall", value=round(token_recall(output.get("answer", ""), expected_output or ""), 4))

    def citations(*, output, **kwargs):
        return Evaluation(name="citation_valid", value=1.0 if output.get("citations") else 0.0)

    def grounded(*, output, **kwargs):
        return Evaluation(name="grounded", value=1.0 if output.get("grounded") else 0.0)

    started = time.perf_counter()
    result = dataset.run_experiment(
        name=run,
        task=task,
        evaluators=[correctness, recall, citations, grounded],
        max_concurrency=1,
        metadata={"retriever": args.retriever or os.getenv("RETRIEVER", ""), "k": args.k},
    )
    client.flush()

    outdir = Path(args.outdir)
    (outdir / "corridas").mkdir(parents=True, exist_ok=True)
    (outdir / "tablas").mkdir(parents=True, exist_ok=True)
    jsonl = outdir / "corridas" / f"eval_langfuse_{run}.jsonl"
    jsonl.write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in captured),
        encoding="utf-8",
    )
    total = len(captured)
    accuracy = sum(item["correct"] for item in captured) / total if total else 0.0
    latency = sorted(item["latency_ms"] for item in captured)
    summary = {
        "run": run,
        "experiment_id": getattr(result, "experiment_id", None),
        "n": total,
        "accuracy": round(accuracy, 4),
        "token_recall": round(sum(item["token_recall"] for item in captured) / total, 4) if total else 0.0,
        "grounded": round(sum(item["grounded"] for item in captured) / total, 4) if total else 0.0,
        "p50_ms": latency[len(latency) // 2] if latency else 0.0,
        "total_s": round(time.perf_counter() - started, 1),
    }
    (outdir / "tablas" / f"eval_langfuse_{run}.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
