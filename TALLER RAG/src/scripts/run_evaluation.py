"""Pipeline de evaluacion: desempeno, chat y RAGAS, con comparacion historica.

Uso:
    python src/scripts/run_evaluation.py
    python src/scripts/run_evaluation.py --skip-ragas
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
TABLES = ROOT / "reports" / "tablas"


def run(command: list[str], env: dict | None = None, check: bool = True) -> None:
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, env=env or os.environ.copy(), check=check)


def performance() -> None:
    base = [
        "docker", "compose", "--profile", "rocm", "run", "--rm", "-T", "ingest-rocm",
        "python", "-m", "src.scripts.eval_configuracion_final", "--model", "gemma4:e4b", "--k-answer", "5",
    ]
    run(base + [
        "--raw", "reports/corridas/metricas_final_sin.jsonl",
        "--summary", "reports/tablas/metricas_final_sin.csv",
        "--by-lang", "reports/tablas/metricas_final_sin_por_idioma.csv",
        "--figures-dir", "reports/figuras",
    ])
    run(base + [
        "--think",
        "--raw", "reports/corridas/metricas_final_con.jsonl",
        "--summary", "reports/tablas/metricas_final_con.csv",
        "--by-lang", "reports/tablas/metricas_final_con_por_idioma.csv",
        "--figures-dir", "/tmp/fig_eval_con",
    ])


def chat() -> None:
    for think in ("0", "1"):
        for strict in ("0", "1"):
            env = {
                **os.environ,
                "EVAL_K": "5",
                "EVAL_THINK": think,
                "EVAL_STRICT": strict,
            }
            run([sys.executable, "src/scripts/eval_chat_cases.py"], env=env)
            suffix = f"{'con' if think == '1' else 'sin'}_{'strict' if strict == '1' else 'libre'}"
            shutil.copy(TABLES / "chat_eval.csv", TABLES / f"chat_eval_final_{suffix}.csv")


def ragas() -> None:
    """Reproduce la fila hibrida (k=10). La fila densa k=5 exige RETRIEVER=dense,
    TOP_K=5 en la API y exportar con --k 5 --hybrid false."""
    run(
        [
            sys.executable,
            "src/scripts/export_ragas_dataset.py",
            "--k", "10",
            "--out", "reports/ragas/dataset_hibrido.jsonl",
            "--hybrid", "true",
        ]
    )
    run(
        [
            "docker", "compose", "--profile", "eval", "run", "--rm", "-T",
            "-e", "RAGAS_JUDGE=gemma4:e4b",
            "-e", "RAGAS_EMBEDDINGS_PROVIDER=hf",
            "-e", "RAGAS_EMBEDDINGS=BAAI/bge-m3",
            "-e", "RAGAS_LIMIT=0",
            "-e", "RAGAS_DATASET=/eval/reports/ragas/dataset_hibrido.jsonl",
            "-e", "RAGAS_OUT=reports/tablas/ragas_hibrido.csv",
            "eval",
        ],
        check=False,
    )


def read_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path)


def summarize() -> None:
    current = read_csv(TABLES / "metricas_final_sin.csv")
    previous = read_csv(TABLES / "metricas_k15_sin.csv")
    current_con = read_csv(TABLES / "metricas_final_con.csv")
    previous_con = read_csv(TABLES / "metricas_k15_con.csv")
    rerank = read_csv(TABLES / "metricas_rerank_top5.csv")
    manual = read_csv(TABLES / "metricas_final_revision_manual.csv")
    chat_strict = read_csv(TABLES / "chat_eval_final_sin_strict.csv")
    chat_free = read_csv(TABLES / "chat_eval_final_sin_libre.csv")
    chat_strict_con = read_csv(TABLES / "chat_eval_final_con_strict.csv")
    chat_previous = read_csv(TABLES / "chat_eval_k15_sin.csv")
    ragas_current = read_csv(TABLES / "ragas_gemma4-e4b_bge-m3.csv")
    ragas_previous = read_csv(TABLES / "ragas_gemma2-2b.csv")

    rows: list[dict] = []

    def add(family: str, metric: str, actual, before) -> None:
        rows.append({"familia": family, "metrica": metric, "actual": actual, "anterior": before})

    if current is not None and previous is not None:
        c, p = current.iloc[0], previous.iloc[0]
        add("Recuperacion", "Recall@10", f"{c['recall_at_10']:.3f}", f"{p['recall_at_10']:.3f}")
        add("Recuperacion", "nDCG@10", f"{c['ndcg_at_10']:.3f}", f"{p['ndcg_at_10']:.3f}")
        add("Recuperacion", "P@1", f"{c['p_at_1']:.3f}", f"{p['p_at_1']:.3f}")
        add("Recuperacion", "Hit@5", f"{c['p_at_5']:.3f}", f"{p['p_at_5']:.3f}")
        add("Respuesta", "Entregadas", f"{int(c['delivered_direct'])}/38", f"{int(p['delivered_direct'])}/38")
        add("Respuesta", "Abstenciones directas", int(c["abstained_direct"]), int(p["abstained_direct"]))
        add(
            "Respuesta",
            "Cobertura de tokens",
            f"{int(c['strict_match_direct'])}/{int(c['delivered_direct'])} ({c['strict_match_rate']*100:.1f}%)",
            f"{int(p['strict_match_direct'])}/{int(p['delivered_direct'])} ({p['strict_match_rate']*100:.1f}%)",
        )
        add("Respuesta", "F1 tokens", f"{c['token_f1_direct']:.3f}", f"{p['token_f1_direct']:.3f}")
        add("Respuesta", "Latencia media", f"{c['total_ms_mean']:.0f} ms", f"{p['total_ms_mean']:.0f} ms")
        add("Respuesta", "Latencia mediana", f"{c['total_ms_median']:.0f} ms", f"{p['total_ms_median']:.0f} ms")
        add("Respuesta", "Contexto medio", f"{c['context_tokens_mean']:.0f}", f"{p['context_tokens_mean']:.0f}")
    if current_con is not None and previous_con is not None:
        c, p = current_con.iloc[0], previous_con.iloc[0]
        add("Respuesta con razonamiento", "Entregadas", f"{int(c['delivered_direct'])}/38", f"{int(p['delivered_direct'])}/38")
        add("Respuesta con razonamiento", "Latencia media", f"{c['total_ms_mean']:.0f} ms", f"{p['total_ms_mean']:.0f} ms")
    if rerank is not None:
        dense = rerank[rerank["config"] == "dense_top5"]
        reranked = rerank[rerank["config"] == "rerank_top5"]
        if not dense.empty and not reranked.empty:
            d, r = dense.iloc[0], reranked.iloc[0]
            add("Recuperacion top-5", "nDCG con reranker", f"{r['ndcg']:.3f}", f"{d['ndcg']:.3f}")
            add("Recuperacion top-5", "P@1 con reranker", f"{r['p_at_1']:.3f}", f"{d['p_at_1']:.3f}")
            add("Recuperacion top-5", "Recall", f"{r['recall']:.3f}", f"{d['recall']:.3f}")
            add("Recuperacion top-5", "Hit@5", f"{r['p_at_5']:.3f}", f"{d['p_at_5']:.3f}")
    if manual is not None:
        counts = manual["categoria"].value_counts()
        correct = int(counts.get("correcta", 0))
        partial = int(counts.get("parcial", 0))
        equivalent = (25 + correct) / 37 * 100
        with_partial = (25 + correct + partial) / 37 * 100
        add("Revision manual", "Equivalentes", f"{25 + correct}/37 ({equivalent:.1f}%)", "32/37 (86.5%)")
        add("Revision manual", "Con parciales", f"{25 + correct + partial}/37 ({with_partial:.1f}%)", "35/37 (94.6%)")
        add("Revision manual", "Correctas / parciales / incorrectas", f"{correct}/{partial}/{int(counts.get('incorrecta', 0))}", "7/3/2")
    if chat_strict is not None:
        add("Chat", "Con Modo estricto", f"{int(chat_strict['correcto'].sum())}/{len(chat_strict)}", f"{int(chat_previous['correcto'].sum()) if chat_previous is not None else 9}/12")
    if chat_free is not None:
        add("Chat", "Sin Modo estricto", f"{int(chat_free['correcto'].sum())}/{len(chat_free)}", f"{int(chat_previous['correcto'].sum()) if chat_previous is not None else 9}/12")
    if chat_strict_con is not None:
        add("Chat con razonamiento", "Con Modo estricto", f"{int(chat_strict_con['correcto'].sum())}/{len(chat_strict_con)}", "9/12")
    if ragas_current is not None and ragas_previous is not None:
        for metric in ("faithfulness", "answer_relevancy", "context_precision", "context_recall", "answer_correctness"):
            if metric in ragas_current.columns and metric in ragas_previous.columns:
                add("RAGAS", metric, f"{ragas_current[metric].mean():.3f}", f"{ragas_previous[metric].mean():.3f}")
    elif ragas_previous is not None:
        for metric in ("faithfulness", "answer_relevancy", "context_precision", "context_recall", "answer_correctness"):
            if metric in ragas_previous.columns:
                add("RAGAS", metric, "n/d", f"{ragas_previous[metric].mean():.3f}")

    frame = pd.DataFrame(rows)
    frame.to_csv(TABLES / "evaluacion_comparativa.csv", index=False)
    lines = ["# Evaluacion comparativa", "", "| Familia | Metrica | Actual | Anterior |", "|---|---|---|---|"]
    for row in rows:
        lines.append(f"| {row['familia']} | {row['metrica']} | {row['actual']} | {row['anterior']} |")
    (TABLES / "evaluacion_comparativa.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-performance", action="store_true")
    parser.add_argument("--skip-chat", action="store_true")
    parser.add_argument("--skip-ragas", action="store_true")
    args = parser.parse_args()
    if not args.skip_performance:
        performance()
    if not args.skip_chat:
        chat()
    if not args.skip_ragas:
        ragas()
    summarize()


if __name__ == "__main__":
    main()
