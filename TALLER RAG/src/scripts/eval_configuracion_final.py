"""Mide recuperación y respuesta con la configuración final del taller.

Recuperación: bge-m3 denso, ef=800, k=10 sobre el silver ampliado.
Respuesta: gemma2:2b, k=5, juez y condensador gemma2, guardarraíl estricto.
"""
from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.rag.chat import chat
from src.rag.pipeline import retrieve
from src.rag.plotting import ACCENT, BLUE, MUTED, WARM, apply_theme, save_figure
from src.scripts.metrics import row_metrics


def tokens(text: str) -> list[str]:
    """Minúsculas, sin acentos y sin puntuación."""
    text = unicodedata.normalize("NFKD", (text or "").lower())
    text = "".join(char for char in text if not unicodedata.combining(char))
    return re.findall(r"[a-z0-9]+", text)


def token_f1(reference: str, answer: str) -> float:
    """F1 por solapamiento de tokens entre la referencia y la respuesta."""
    ref = tokens(reference)
    pred = tokens(answer)
    if not ref or not pred:
        return 0.0
    overlap = sum((Counter(ref) & Counter(pred)).values())
    if not overlap:
        return 0.0
    precision = overlap / len(pred)
    recall = overlap / len(ref)
    return 2 * precision * recall / (precision + recall)


def contains_reference(reference: str, answer: str) -> bool:
    """Verdadero si todos los tokens de la referencia aparecen en la respuesta."""
    ref = tokens(reference)
    pred = set(tokens(answer))
    return bool(ref) and all(token in pred for token in ref)


def evaluate(silver_path: Path, collection: str, model: str, k_retrieval: int, k_answer: int, think: bool = False) -> pd.DataFrame:
    silver = [json.loads(line) for line in silver_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = []
    for index, question in enumerate(silver, start=1):
        hits, retrieval_ms = retrieve(question["question"], k=k_retrieval, collection=collection)
        record = {
            "run": "configuracion_final",
            "question_id": question["id"],
            "split": question.get("split"),
            "model": model,
            "mode": "retrieve",
            "k": k_retrieval,
            "rep": 1,
            "collection": collection,
            "reference_doc_ids": question.get("reference_doc_ids") or [],
            "reference_chunk_ids": question.get("reference_chunk_ids") or [],
            "reference_span": question.get("reference_span") or "",
            "response": {"chunks": hits, "retrieval_ms": retrieval_ms},
        }
        retrieval = row_metrics(record)
        answer = chat(question["question"], model=model, k=k_answer, collection=collection, think=think)
        reference = question.get("reference_answer") or ""
        generated = answer.get("answer") or ""
        rows.append(
            {
                "id": question["id"],
                "type": question.get("type"),
                "lang": question.get("lang"),
                "question": question["question"],
                "reference_answer": reference,
                "answer": generated,
                "abstained": answer.get("abstained"),
                "abstain_reason": answer.get("abstain_reason"),
                "citations": answer.get("citations") or [],
                "has_refs": retrieval["has_refs"],
                "retrieval_recall": retrieval["recall"],
                "retrieval_ndcg": retrieval["ndcg"],
                "retrieval_p_at_1": retrieval["p_at_1"],
                "retrieval_p_at_5": retrieval["p_at_5"],
                "retrieval_mrr": retrieval["mrr"],
                "retrieval_map": retrieval["map"],
                "retrieval_span_hit": retrieval["span_hit"],
                "retrieval_ms": answer.get("retrieval_ms") or retrieval_ms,
                "generation_ms": answer.get("generation_ms"),
                "total_ms": answer.get("total_ms"),
                "context_tokens": answer.get("context_tokens"),
                "context_chunks": answer.get("context_chunks"),
                "strict_match": contains_reference(reference, generated) if reference else None,
                "token_f1": token_f1(reference, generated) if reference else None,
            }
        )
        print(
            f"{index}/{len(silver)} {question['id']} recall={retrieval['recall']:.2f} "
            f"abst={answer.get('abstained')} match={rows[-1]['strict_match']}",
            flush=True,
        )
    return pd.DataFrame(rows)


def summarize(frame: pd.DataFrame) -> dict:
    direct = frame[frame["type"] == "directa"]
    negative = frame[frame["type"] == "sin_respuesta"]
    with_refs = frame[frame["has_refs"]]
    delivered = direct[~direct["abstained"]]
    return {
        "n_total": len(frame),
        "n_direct": len(direct),
        "n_negative": len(negative),
        "recall_at_10": with_refs["retrieval_recall"].mean(),
        "ndcg_at_10": with_refs["retrieval_ndcg"].mean(),
        "p_at_1": with_refs["retrieval_p_at_1"].mean(),
        "p_at_5": with_refs["retrieval_p_at_5"].mean(),
        "mrr": with_refs["retrieval_mrr"].mean(),
        "map": with_refs["retrieval_map"].mean(),
        "span_hit": with_refs["retrieval_span_hit"].dropna().mean(),
        "delivered_direct": len(delivered),
        "abstained_direct": int(direct["abstained"].sum()),
        "abstained_negative": int(negative["abstained"].sum()),
        "strict_match_direct": int(delivered["strict_match"].sum()),
        "strict_match_rate": delivered["strict_match"].mean(),
        "token_f1_direct": delivered["token_f1"].mean(),
        "citation_rate": delivered["citations"].apply(len).gt(0).mean(),
        "context_tokens_mean": frame["context_tokens"].mean(),
        "retrieval_ms_mean": frame["retrieval_ms"].mean(),
        "total_ms_mean": frame["total_ms"].mean(),
        "total_ms_median": frame["total_ms"].median(),
        "total_ms_p95": frame["total_ms"].quantile(0.95),
    }


def figures(frame: pd.DataFrame, summary: dict, outdir: Path) -> None:
    apply_theme()
    outdir.mkdir(parents=True, exist_ok=True)

    metrics = [
        ("Recall@10", summary["recall_at_10"]),
        ("nDCG@10", summary["ndcg_at_10"]),
        ("P@1", summary["p_at_1"]),
        ("P@5", summary["p_at_5"]),
        ("span hit", summary["span_hit"]),
    ]
    fig, ax = plt.subplots(figsize=(5.2, 2.8))
    labels = [name for name, _ in metrics]
    values = [value for _, value in metrics]
    bars = ax.barh(labels[::-1], values[::-1], color=[ACCENT, ACCENT, BLUE, BLUE, WARM][::-1], height=0.55)
    for bar, value in zip(bars, values[::-1]):
        ax.text(value + 0.015, bar.get_y() + bar.get_height() / 2, f"{value:.3f}", va="center", color=MUTED, fontsize=8)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("valor")
    ax.set_title("Recuperación sobre el silver ampliado")
    save_figure(fig, outdir / "metricas_recuperacion.png")

    direct = frame[frame["type"] == "directa"]
    negative = frame[frame["type"] == "sin_respuesta"]
    delivered = direct[~direct["abstained"]]
    manual_path = Path("reports/tablas/metricas_final_revision_manual.csv")
    manual_equivalent = None
    if manual_path.exists():
        manual = pd.read_csv(manual_path)
        manual_equivalent = (
            delivered["strict_match"].sum() + (manual["categoria"] == "correcta").sum()
        ) / len(delivered)
    rates = [
        ("Entrega directas", len(delivered) / len(direct)),
        ("Match estricto", delivered["strict_match"].mean()),
        ("Abstención negativas", negative["abstained"].mean()),
        ("Con cita", delivered["citations"].apply(len).gt(0).mean()),
    ]
    if manual_equivalent is not None:
        rates.insert(2, ("Equivalentes manual", manual_equivalent))
    fig, ax = plt.subplots(figsize=(5.2, 2.6))
    labels = [name for name, _ in rates]
    values = [value for _, value in rates]
    bars = ax.bar(labels, values, color=[ACCENT, BLUE, WARM, ACCENT], width=0.55)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.3f}", ha="center", color=MUTED, fontsize=8)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("proporción")
    ax.set_title("Respuesta y guardarraíl")
    plt.setp(ax.get_xticklabels(), rotation=12, ha="right")
    save_figure(fig, outdir / "metricas_respuesta.png")

    fig, ax = plt.subplots(figsize=(5.2, 2.6))
    ax.hist(frame["total_ms"].dropna(), bins=12, color=ACCENT, edgecolor="white")
    ax.axvline(summary["total_ms_median"], color=WARM, linestyle="--", linewidth=1)
    ax.set_xlabel("ms")
    ax.set_ylabel("preguntas")
    ax.set_title(f"Latencia total (mediana {summary['total_ms_median']:.0f} ms)")
    save_figure(fig, outdir / "latencia_respuesta.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--silver", type=Path, default=Path("eval/silver/silver_standard_ampliado.jsonl"))
    parser.add_argument("--collection", default="rag_full_bge_m3_d1024_v1")
    parser.add_argument("--model", default="gemma4:e4b")
    parser.add_argument("--k-retrieval", type=int, default=10)
    parser.add_argument("--k-answer", type=int, default=5)
    parser.add_argument("--raw", type=Path, default=Path("reports/corridas/metricas_final_raw.jsonl"))
    parser.add_argument("--summary", type=Path, default=Path("reports/tablas/metricas_final_resumen.csv"))
    parser.add_argument("--by-lang", type=Path, default=Path("reports/tablas/metricas_final_por_idioma.csv"))
    parser.add_argument("--figures-dir", type=Path, default=Path("figuras"))
    parser.add_argument("--figures-only", action="store_true")
    parser.add_argument("--think", action="store_true")
    args = parser.parse_args()

    if args.figures_only:
        frame = pd.read_json(args.raw, lines=True)
        summary = summarize(frame)
        by_lang = frame[frame["has_refs"]].groupby("lang").agg(
            n=("id", "nunique"),
            recall=("retrieval_recall", "mean"),
            ndcg=("retrieval_ndcg", "mean"),
            p_at_1=("retrieval_p_at_1", "mean"),
            p_at_5=("retrieval_p_at_5", "mean"),
            span_hit=("retrieval_span_hit", "mean"),
        ).reset_index()
    else:
        frame = evaluate(args.silver, args.collection, args.model, args.k_retrieval, args.k_answer, args.think)
        args.raw.parent.mkdir(parents=True, exist_ok=True)
        with args.raw.open("w", encoding="utf-8") as handle:
            for row in frame.to_dict(orient="records"):
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        summary = summarize(frame)
        pd.DataFrame([summary]).to_csv(args.summary, index=False)
        with_refs = frame[frame["has_refs"]]
        by_lang = with_refs.groupby("lang").agg(
            n=("id", "nunique"),
            recall=("retrieval_recall", "mean"),
            ndcg=("retrieval_ndcg", "mean"),
            p_at_1=("retrieval_p_at_1", "mean"),
            p_at_5=("retrieval_p_at_5", "mean"),
            span_hit=("retrieval_span_hit", "mean"),
        ).reset_index()
        by_lang.to_csv(args.by_lang, index=False)
    reasons = frame["abstain_reason"].dropna().value_counts()
    print("\nResumen")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    print("\nPor idioma")
    print(by_lang.round(3).to_string(index=False))
    print("\nMotivos de abstención")
    print(reasons.to_string() if not reasons.empty else "  ninguno")
    figures(frame, summary, args.figures_dir)


if __name__ == "__main__":
    main()
