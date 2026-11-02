"""Evalua el RAG con RAGAS usando un juez local de Ollama."""
from __future__ import annotations

import json
import os
from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_openai import ChatOpenAI
from ragas import EvaluationDataset, RunConfig, evaluate
from ragas.dataset_schema import SingleTurnSample
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    answer_correctness,
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

DATASET = Path(os.getenv("RAGAS_DATASET", "reports/ragas/dataset.jsonl"))
NEGATIVES = Path(os.getenv("RAGAS_NEGATIVES", "reports/ragas/negatives.jsonl"))
JUDGE = os.getenv("RAGAS_JUDGE", "gemma2:2b")
EMBEDDINGS = os.getenv("RAGAS_EMBEDDINGS", "qwen3-embedding:8b")
PROVIDER = os.getenv("RAGAS_EMBEDDINGS_PROVIDER", "ollama")
LIMIT = int(os.getenv("RAGAS_LIMIT", "0"))
TIMEOUT = int(os.getenv("RAGAS_TIMEOUT", "900"))
WORKERS = int(os.getenv("RAGAS_WORKERS", "6"))
OUT = Path(os.getenv("RAGAS_OUT", f"reports/tablas/ragas_{JUDGE.replace(':', '-')}.csv"))


def main() -> None:
    rows = [json.loads(line) for line in DATASET.read_text(encoding="utf-8").splitlines() if line.strip()]
    if LIMIT:
        rows = rows[:LIMIT]
    print(f"evaluando {len(rows)} preguntas con juez {JUDGE} y embeddings {EMBEDDINGS} ({PROVIDER})", flush=True)
    samples = [
        SingleTurnSample(
            user_input=row["user_input"],
            response=row["response"],
            retrieved_contexts=row["retrieved_contexts"],
            reference=row["reference"],
        )
        for row in rows
    ]
    dataset = EvaluationDataset(samples=samples)
    llm = LangchainLLMWrapper(
        ChatOpenAI(
            base_url="http://localhost:11435/v1",
            api_key="ollama",
            model=JUDGE,
            temperature=0,
        )
    )
    if PROVIDER == "hf":
        embeddings = LangchainEmbeddingsWrapper(
            HuggingFaceEmbeddings(
                model_name=EMBEDDINGS,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
        )
    else:
        embeddings = LangchainEmbeddingsWrapper(
            OllamaEmbeddings(base_url="http://localhost:11435", model=EMBEDDINGS)
        )
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall, answer_correctness],
        llm=llm,
        embeddings=embeddings,
        run_config=RunConfig(timeout=TIMEOUT, max_retries=3, max_wait=30, max_workers=WORKERS),
        raise_exceptions=False,
    )
    frame = result.to_pandas()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT, index=False)
    numeric = frame.select_dtypes("number")
    print(numeric.mean().round(4).to_string(), flush=True)
    print(f"{len(frame)} filas -> {OUT}", flush=True)
    if NEGATIVES.exists():
        negatives = [json.loads(line) for line in NEGATIVES.read_text(encoding="utf-8").splitlines() if line.strip()]
        abstained = sum(1 for row in negatives if row.get("abstained"))
        print(f"sin respuesta: {abstained}/{len(negatives)} se abstienen", flush=True)


if __name__ == "__main__":
    main()
