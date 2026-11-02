"""Consolida el gold standard: deduplica, asigna split balanceado y exporta JSONL, CSV y Markdown."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def load(paths: list[str]) -> pd.DataFrame:
    rows = []
    for path in paths:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    frame = pd.DataFrame(rows)
    frame = frame.drop_duplicates(subset=["question"]).reset_index(drop=True)
    return frame


def assign_split(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["source"] = frame["reference_doc_ids"].map(
        lambda ids: ids[0].split(":")[0] if ids else "sin_respuesta"
    )
    frame["split"] = "dev"
    for _, group in frame.groupby(["source", "lang"], dropna=False):
        for position, index in enumerate(group.sort_values("id").index):
            frame.loc[index, "split"] = "dev" if position % 2 == 0 else "test"
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", nargs="+", required=True)
    parser.add_argument("--unanswerable", default=None)
    parser.add_argument("--outdir", default="eval/gold")
    parser.add_argument("--prefix", default="g")
    parser.add_argument("--parquet", default="data/documentos_rag.parquet")
    parser.add_argument("--exclude-ids", default="", help="IDs separados por coma")
    parser.add_argument("--reset-validation", action="store_true", help="Borra reviewer y validated")
    args = parser.parse_args()

    frame = assign_split(load(args.gold))
    if args.unanswerable:
        extra = load([args.unanswerable])
        extra["source"] = "sin_respuesta"
        extra["split"] = "test"
        frame = pd.concat([frame, extra], ignore_index=True)

    frame = frame.reset_index(drop=True)
    frame["id"] = [f"{args.prefix}-{i:03d}" for i in range(1, len(frame) + 1)]
    excluded = {item.strip() for item in args.exclude_ids.split(",") if item.strip()}
    if excluded:
        frame = frame[~frame["id"].isin(excluded)].reset_index(drop=True)
    previous: dict[str, tuple[str, bool, str]] = {}
    previous_path = Path(args.outdir) / "gold_standard.jsonl"
    if previous_path.exists() and not args.reset_validation:
        for line in previous_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                question = record.get("question", "")
                if question:
                    previous[question] = (
                        record.get("reviewer", ""),
                        bool(record.get("validated")),
                        record.get("split", ""),
                    )
    frame["validated"] = frame["question"].map(lambda q: previous.get(q, ("", False, ""))[1])
    frame["reviewer"] = frame["question"].map(lambda q: previous.get(q, ("", False, ""))[0])
    frame["split"] = [
        previous[q][2] if q in previous and previous[q][2] else current
        for q, current in zip(frame["question"], frame["split"])
    ]
    frame["criteria"] = frame.get(
        "criteria", "La respuesta debe coincidir con el span de referencia en el documento citado."
    )
    frame["reference_url"] = ""
    if Path(args.parquet).exists():
        import pyarrow.parquet as pq

        meta = pq.read_table(args.parquet, columns=["document_id", "url"]).to_pandas()
        urls = dict(zip(meta["document_id"], meta["url"]))
        frame["reference_url"] = frame["reference_doc_ids"].map(
            lambda ids: urls.get(ids[0], "") if isinstance(ids, list) and ids else ""
        )

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    columns = [
        "id",
        "split",
        "type",
        "lang",
        "source",
        "question",
        "reference_answer",
        "reference_doc_ids",
        "reference_url",
        "reference_span",
        "criteria",
        "reviewer",
        "validated",
    ]
    records = frame.to_dict(orient="records")
    with (outdir / "gold_standard.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    flat = frame.copy()
    flat["reference_doc_ids"] = flat["reference_doc_ids"].map(lambda ids: ";".join(ids) if isinstance(ids, list) else "")
    flat[columns].to_csv(outdir / "gold_standard.csv", index=False)

    lines = [
        "# Gold standard",
        "",
        "Validación humana pendiente: `validated` está en falso en todas las filas.",
        "",
        "Para validar: revisa la pregunta, confirma que la respuesta y el span estén en el documento citado, corrige lo que haga falta y marca `validated: true` en el JSONL. Las preguntas `sin_respuesta` se evalúan por abstención, no por Recall@k.",
        "",
    ]
    for split in ["dev", "test"]:
        subset = flat[flat["split"] == split]
        lines.append(f"## {split} ({len(subset)})")
        lines.append("")
        lines.append("| id | tipo | idioma | pregunta | respuesta de referencia | documento | revisor | validado |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for row in subset.itertuples(index=False):
            doc = row.reference_doc_ids.split(";")[0] if row.reference_doc_ids else ""
            lines.append(
                f"| {row.id} | {row.type} | {row.lang} | {row.question} | {row.reference_answer} | {doc} | {row.reviewer or ''} | {'sí' if row.validated else ''} |"
            )
        lines.append("")
    (outdir / "gold_standard.md").write_text("\n".join(lines), encoding="utf-8")

    print(
        json.dumps(
            {
                "total": len(frame),
                "dev": int((frame["split"] == "dev").sum()),
                "test": int((frame["split"] == "test").sum()),
                "por_tipo": frame["type"].value_counts().to_dict(),
                "por_idioma": frame["lang"].value_counts().to_dict(),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
