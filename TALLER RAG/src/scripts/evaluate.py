"""Ejecuta el gold contra la API y guarda resultados crudos en JSONL."""
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

API = os.getenv("API_URL", "http://localhost:8000").rstrip("/")


def post(path: str, payload: dict) -> dict:
    response = requests.post(f"{API}{path}", json=payload, timeout=600)
    response.raise_for_status()
    return response.json()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", default="eval/gold/questions.jsonl")
    parser.add_argument("--split", default="test", choices=["dev", "test", "smoke", "all"])
    parser.add_argument("--mode", default="ask", choices=["ask", "retrieve"])
    parser.add_argument("--models", default="")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--collection", default=None)
    parser.add_argument("--run", default=None)
    parser.add_argument("--outdir", default="reports/corridas")
    args = parser.parse_args()

    questions = [json.loads(line) for line in Path(args.gold).read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.split != "all":
        questions = [q for q in questions if q.get("split") == args.split]
    questions = [q for q in questions if not q.get("excluded")]
    if not questions:
        raise SystemExit("no hay preguntas para ese split")

    models = [m.strip() for m in args.models.split(",") if m.strip()] or [None]
    run = args.run or f"{args.mode}_{args.split}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M')}"
    out = Path(args.outdir) / f"{run}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)

    total = len(questions) * len(models) * args.reps
    done = 0
    with out.open("w", encoding="utf-8") as handle:
        for model in models:
            for question in questions:
                for rep in range(1, args.reps + 1):
                    payload = {
                        "query": question["question"],
                        "k": args.k,
                        "collection": args.collection,
                    }
                    if args.mode == "ask":
                        payload["model"] = model
                        payload["temperature"] = args.temperature
                        endpoint = "/ask"
                    else:
                        endpoint = "/retrieve"
                    started = time.perf_counter()
                    try:
                        response = post(endpoint, payload)
                        error = None
                    except Exception as exc:  # noqa: BLE001  # se registra para no perder la corrida
                        response = {}
                        error = str(exc)
                    record = {
                        "run": run,
                        "question_id": question["id"],
                        "split": question.get("split"),
                        "model": model,
                        "rep": rep,
                        "mode": args.mode,
                        "k": args.k,
                        "temperature": args.temperature,
                        "collection": args.collection,
                        "reference_doc_ids": question.get("reference_doc_ids", []),
                        "reference_chunk_ids": question.get("reference_chunk_ids", []),
                        "reference_span": question.get("reference_span", ""),
                        "wall_ms": round((time.perf_counter() - started) * 1000, 2),
                        "response": response,
                        "error": error,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    handle.flush()
                    done += 1
                    print(f"{done}/{total} {question['id']} {model or ''} rep={rep} error={bool(error)}", flush=True)
    print(json.dumps({"run": run, "registros": done, "salida": str(out)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
