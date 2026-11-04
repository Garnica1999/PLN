"""Compara clarificacion, respuesta condicional e hibrida en ambiguedad referencial."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.rag.chat import chat
from src.scripts.eval_configuracion_final import contains_reference

SILVER = Path("eval/silver/silver_ambiguedad.jsonl")
RAW = Path("reports/corridas/ambiguedad_raw.jsonl")
SUMMARY = Path("reports/tablas/ambiguedad_resumen.csv")
PER_QUESTION = Path("reports/tablas/ambiguedad_por_pregunta.csv")


def matches(text: str, aliases: list[str]) -> bool:
    return any(contains_reference(alias, text) for alias in aliases)


def run(silver_path: Path = SILVER, raw_path: Path = RAW, summary_path: Path = SUMMARY, per_question_path: Path = PER_QUESTION) -> None:
    rows = [json.loads(line) for line in silver_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    results = []
    for index, row in enumerate(rows, start=1):
        question = row["question"]
        ambiguous = row["type"] == "ambigua"
        aliases = row.get("intended_aliases") or [row.get("reference_answer") or ""]

        baseline = chat(question, ambiguity_policy="off", history=row.get("history"))
        baseline_answer = baseline.get("answer") or ""
        baseline_correct = (not baseline.get("abstained")) and matches(baseline_answer, aliases)

        clarification = chat(question, ambiguity_policy="clarify", history=row.get("history"))
        detected = clarification.get("type") == "clarification"
        intended_offered = False
        clarification_correct = False
        clarification_turns = 1
        clarification_ms = clarification.get("total_ms") or 0.0
        chosen = None
        if detected:
            for option in clarification.get("clarification", {}).get("options", []):
                if matches(option.get("answer") or "", aliases):
                    chosen = option
                    intended_offered = True
                    break
            if chosen is not None:
                follow = chat(chosen["query"], ambiguity_policy="off")
                follow_answer = follow.get("answer") or ""
                clarification_correct = (not follow.get("abstained")) and matches(follow_answer, aliases)
                clarification_turns = 2
                clarification_ms += follow.get("total_ms") or 0.0
        else:
            clarification_answer = clarification.get("answer") or ""
            clarification_correct = (not clarification.get("abstained")) and matches(clarification_answer, aliases)

        conditional = chat(question, ambiguity_policy="conditional", history=row.get("history"))
        conditional_detected = conditional.get("type") == "conditional"
        conditional_covered = False
        conditional_correct = False
        recommendation_correct = False
        recommendation_present = False
        if conditional_detected:
            conditional_covered = any(
                matches(condition.get("answer") or "", aliases)
                for condition in conditional.get("conditions", [])
            )
            conditional_correct = conditional_covered
            recommendation = conditional.get("recommendation") or {}
            recommendation_present = bool(recommendation)
            recommendation_correct = recommendation_present and matches(recommendation.get("answer") or "", aliases)
        else:
            conditional_answer = conditional.get("answer") or ""
            conditional_correct = (not conditional.get("abstained")) and matches(conditional_answer, aliases)

        hybrid_correct = clarification_correct if detected else baseline_correct
        results.append(
            {
                "id": row["id"],
                "type": row["type"],
                "lang": row["lang"],
                "question": question,
                "detected": detected,
                "intended_offered": intended_offered,
                "chosen_option": chosen["label"] if chosen else None,
                "baseline_correct": baseline_correct,
                "clarification_correct": clarification_correct,
                "conditional_detected": conditional_detected,
                "conditional_covered": conditional_covered,
                "conditional_correct": conditional_correct,
                "conditional_recommendation_present": recommendation_present,
                "conditional_recommendation_correct": recommendation_correct,
                "split": row.get("split"),
                "subtype": row.get("subtype"),
                "hybrid_correct": hybrid_correct,
                "clarification_turns": clarification_turns,
                "baseline_ms": baseline.get("total_ms"),
                "clarification_ms": round(clarification_ms, 2),
                "conditional_ms": conditional.get("total_ms"),
                "baseline_answer": baseline_answer,
                "clarification_answer": clarification.get("answer"),
                "conditional_answer": conditional.get("answer"),
            }
        )
        print(
            f"{index}/{len(rows)} {row['id']} detected={detected} "
            f"baseline={baseline_correct} clarify={clarification_correct} "
            f"conditional={conditional_correct}",
            flush=True,
        )

    frame = pd.DataFrame(results)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    per_question_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(per_question_path, index=False)
    with raw_path.open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")

    ambiguous = frame[frame["type"] == "ambigua"]
    control = frame[frame["type"] == "no_ambigua"]
    tp = int((ambiguous["detected"]).sum())
    fn = int((~ambiguous["detected"]).sum())
    fp = int((control["detected"]).sum())
    tn = int((~control["detected"]).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    summary = {
        "n": len(frame),
        "n_ambiguous": len(ambiguous),
        "n_control": len(control),
        "detector_precision": round(precision, 3),
        "detector_recall": round(recall, 3),
        "detector_f1": round(f1, 3),
        "baseline_accuracy": round(float(frame["baseline_correct"].mean()), 3),
        "clarification_accuracy": round(float(frame["clarification_correct"].mean()), 3),
        "clarification_intended_offered": round(float(ambiguous["intended_offered"].mean()), 3),
        "clarification_turns": round(float(frame["clarification_turns"].mean()), 2),
        "conditional_accuracy": round(float(frame["conditional_correct"].mean()), 3),
        "conditional_intended_covered": round(float(ambiguous["conditional_covered"].mean()), 3),
        "conditional_recommendation_accuracy": round(float(ambiguous["conditional_recommendation_correct"].mean()), 3),
        "conditional_recommendation_coverage": round(float(ambiguous["conditional_recommendation_present"].mean()), 3),
        "conditional_recommendation_precision": round(float(ambiguous.loc[ambiguous["conditional_recommendation_present"], "conditional_recommendation_correct"].mean()), 3) if ambiguous["conditional_recommendation_present"].any() else 0.0,
        "hybrid_accuracy": round(float(frame["hybrid_correct"].mean()), 3),
        "baseline_ms": round(float(frame["baseline_ms"].mean()), 1),
        "clarification_ms": round(float(frame["clarification_ms"].mean()), 1),
        "conditional_ms": round(float(frame["conditional_ms"].mean()), 1),
    }
    pd.DataFrame([summary]).to_csv(summary_path, index=False)
    for key, value in summary.items():
        print(f"{key}: {value}", flush=True)


def main() -> None:
    run()


if __name__ == "__main__":
    main()
