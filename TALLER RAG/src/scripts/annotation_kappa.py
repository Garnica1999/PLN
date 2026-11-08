"""Agrega la anotación humana de E4: kappa, medias por modelo y contraste de H1.

Lee eval/annotations/key.csv y un CSV por anotador. Calcula kappa de Cohen
ponderado lineal por eje, promedia los ejes por modelo y contrasta H1 con
bootstrap pareado por pregunta. Escribe tres CSV en reports/tablas/.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import statistics as st
from pathlib import Path

from openpyxl import load_workbook
from sklearn.metrics import cohen_kappa_score

SEED = 20261031
EJES = ["correccion", "fidelidad", "citas", "abstencion"]
MODELOS = ["gemma2:2b", "qwen2.5:1.5b", "llama3.2:3b", "llama3.2:1b"]


def read_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_key(path: Path) -> dict[str, dict]:
    return {row["item_id"]: row for row in read_rows(path)}


def load_votes(path: Path) -> dict[str, dict[str, int]]:
    if path.suffix == ".xlsx":
        return load_votes_xlsx(path)
    votes: dict[str, dict[str, int]] = {}
    for row in read_rows(path):
        votes[row["item_id"]] = {eje: int(row[eje]) for eje in EJES}
    return votes


def load_votes_xlsx(path: Path) -> dict[str, dict[str, int]]:
    sheet = load_workbook(path, data_only=True)["Anotacion"]
    headers = {cell.value: index for index, cell in enumerate(sheet[1])}
    missing = [eje for eje in EJES + ["item_id"] if eje not in headers]
    if missing:
        raise SystemExit(f"Faltan columnas en {path.name}: {', '.join(missing)}")
    votes: dict[str, dict[str, int]] = {}
    for row in sheet.iter_rows(min_row=2, values_only=True):
        item_id = row[headers["item_id"]]
        if not item_id:
            continue
        values = {eje: row[headers[eje]] for eje in EJES}
        filled = [v for v in values.values() if v is not None and str(v).strip() != ""]
        if not filled:
            continue
        if len(filled) < len(EJES):
            raise SystemExit(f"Fila incompleta en {path.name}: {item_id}")
        votes[str(item_id)] = {eje: int(values[eje]) for eje in EJES}
    return votes


def kappa(y1: list[int], y2: list[int]) -> float | None:
    if len(set(y1 + y2)) < 2:
        return None
    return cohen_kappa_score(y1, y2, weights="linear", labels=[0, 1, 2, 3])


def agreement(y1: list[int], y2: list[int]) -> float:
    return sum(a == b for a, b in zip(y1, y2)) / len(y1)


def per_question_means(
    voters: list[dict[str, dict[str, int]]], key: dict[str, dict]
) -> dict[str, dict[str, float]]:
    """Media por pregunta y modelo, promediando los anotadores que la calificaron."""
    buckets: dict[str, dict[str, list[float]]] = {m: {} for m in MODELOS}
    for item_id, meta in key.items():
        model = meta["model"]
        if not any(item_id in v for v in voters):
            continue
        for eje in EJES:
            per_annotator = [v[item_id][eje] for v in voters if item_id in v]
            buckets[model].setdefault(eje, {}).setdefault(meta["question_id"], []).append(
                sum(per_annotator) / len(per_annotator)
            )
    return {
        model: {eje: {q: sum(vals) / len(vals) for q, vals in data.items()} for eje, data in ejes.items()}
        for model, ejes in buckets.items()
    }


def paired_bootstrap(a: dict[str, float], b: dict[str, float], n: int = 2000) -> tuple[float, float, float]:
    questions = sorted(set(a) & set(b))
    diffs = [a[q] - b[q] for q in questions]
    rng = random.Random(SEED)
    means = sorted(sum(diffs[rng.randrange(len(diffs))] for _ in diffs) / len(diffs) for _ in range(n))
    return sum(diffs) / len(diffs), means[int(0.025 * n)], means[int(0.975 * n) - 1]


def latency_p50(run_path: Path, answerable: set[str]) -> dict[str, float]:
    rows = [json.loads(line) for line in run_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = [
        r
        for r in rows
        if r.get("mode") == "ask"
        and not r.get("error")
        and r.get("response")
        and r.get("question_id") in answerable
    ]
    by_model: dict[str, list[float]] = {}
    for r in rows:
        ms = r["response"].get("generation_ms")
        if ms:
            by_model.setdefault(r["model"], []).append(ms)
    return {m: st.median(v) for m, v in by_model.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations-dir", type=Path, default=Path("eval/annotations"))
    parser.add_argument("--run", type=Path, default=Path("reports/corridas/e4_test.jsonl"))
    parser.add_argument("--out-dir", type=Path, default=Path("reports/tablas"))
    parser.add_argument("--exclude-questions", nargs="*", default=[])
    args = parser.parse_args()

    key_path = args.annotations_dir / "key.csv"
    if not key_path.exists():
        raise SystemExit("No existe key.csv. Abre la app de anotación al menos una vez.")
    key = load_key(key_path)
    if args.exclude_questions:
        key = {i: meta for i, meta in key.items() if meta["question_id"] not in set(args.exclude_questions)}
        print(f"Excluidas {len(args.exclude_questions)} preguntas: {', '.join(args.exclude_questions)}")
    files = sorted(
        p
        for p in args.annotations_dir.iterdir()
        if p.suffix in (".csv", ".xlsx") and p.name != "key.csv" and not p.name.startswith("~$")
    )
    if not files:
        raise SystemExit("No hay CSV de anotadores en eval/annotations/")

    votes = {}
    for path in files:
        loaded = load_votes(path)
        if loaded:
            votes[path.stem] = loaded
        else:
            print(f"Ignorado {path.name}: sin calificaciones")
    if not votes:
        raise SystemExit("Ningún archivo tiene calificaciones.")
    annotators = sorted(votes)
    common = set(key) & set.intersection(*(set(v) for v in votes.values()))
    print(f"Anotadores: {', '.join(annotators)}")
    print(f"Items en key: {len(key)} | anotados por todos: {len(common)}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    kappa_rows = []
    if len(annotators) >= 2:
        pairs = [(a, b) for i, a in enumerate(annotators) for b in annotators[i + 1 :]]
        for eje in EJES:
            values = []
            for a, b in pairs:
                ids = sorted(common)
                y1 = [votes[a][i][eje] for i in ids]
                y2 = [votes[b][i][eje] for i in ids]
                k = kappa(y1, y2)
                if k is not None:
                    values.append(k)
                kappa_rows.append(
                    {"par": f"{a}-{b}", "eje": eje, "kappa_lineal": round(k, 3) if k is not None else None, "acuerdo": round(agreement(y1, y2), 3), "n": len(ids)}
                )
            if values:
                print(f"kappa {eje}: {sum(values) / len(values):.3f} (media de {len(values)} pares)")
            else:
                print(f"kappa {eje}: sin varianza, no calculable")
        with (args.out_dir / "e4_kappa.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["par", "eje", "kappa_lineal", "acuerdo", "n"])
            writer.writeheader()
            writer.writerows(kappa_rows)
    else:
        print("Un solo anotador: no se calcula kappa. Falta la segunda anotación.")

    means = per_question_means([votes[a] for a in annotators], key)
    lat = latency_p50(args.run, {meta["question_id"] for meta in key.values()})
    model_rows = []
    for model in MODELOS:
        data = means.get(model, {})
        row = {"model": model, "n_preguntas": len(data.get("correccion", {}))}
        for eje in EJES:
            row[eje] = round(st.mean(data[eje].values()), 3) if data.get(eje) else None
        row["generation_p50_ms"] = round(lat.get(model, 0), 1)
        model_rows.append(row)
    model_rows.sort(key=lambda r: (r["correccion"] is None, -(r["correccion"] or 0)))
    with (args.out_dir / "e4_anotacion.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model", "n_preguntas", *EJES, "generation_p50_ms"])
        writer.writeheader()
        writer.writerows(model_rows)

    print("\nModelo                 n   correccion  fidelidad  citas  abstencion  p50 ms")
    for r in model_rows:
        print(
            f"{r['model']:<22} {r['n_preguntas']:>2}   {r['correccion']:.3f}      {r['fidelidad']:.3f}"
            f"     {r['citas']:.2f}   {r['abstencion']:.2f}       {r['generation_p50_ms']:.0f}"
        )

    h1_rows = []
    base = "llama3.2:3b"
    for rival in ["qwen2.5:1.5b", "gemma2:2b"]:
        for eje in ["correccion", "fidelidad"]:
            diff, lo, hi = paired_bootstrap(means[base][eje], means[rival][eje])
            h1_rows.append(
                {
                    "comparacion": f"{base} - {rival}",
                    "eje": eje,
                    "diff": round(diff, 3),
                    "ic_lo": round(lo, 3),
                    "ic_hi": round(hi, 3),
                    "excluye_cero": lo > 0 or hi < 0,
                    "latencia_base_ms": round(lat.get(base, 0), 1),
                    "latencia_rival_ms": round(lat.get(rival, 0), 1),
                }
            )
    with (args.out_dir / "e4_h1.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "comparacion",
                "eje",
                "diff",
                "ic_lo",
                "ic_hi",
                "excluye_cero",
                "latencia_base_ms",
                "latencia_rival_ms",
            ],
        )
        writer.writeheader()
        writer.writerows(h1_rows)

    print("\nH1: llama3.2:3b mayor correccion y fidelidad que qwen y gemma")
    for r in h1_rows:
        print(
            f"{r['comparacion']:<26} {r['eje']:<11} {r['diff']:+.3f} "
            f"[{r['ic_lo']:+.3f}; {r['ic_hi']:+.3f}] {'si' if r['excluye_cero'] else 'no'}"
        )

    ranking = sorted(
        [m for m in MODELOS if means.get(m, {}).get("correccion")],
        key=lambda m: -st.mean(means[m]["correccion"].values()),
    )
    top, second = ranking[0], ranking[1]
    winner_rows = []
    for eje in EJES:
        diff, lo, hi = paired_bootstrap(means[top][eje], means[second][eje])
        winner_rows.append(
            {
                "comparacion": f"{top} - {second}",
                "eje": eje,
                "diff": round(diff, 3),
                "ic_lo": round(lo, 3),
                "ic_hi": round(hi, 3),
                "excluye_cero": lo > 0 or hi < 0,
            }
        )
    with (args.out_dir / "e4_ganador.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["comparacion", "eje", "diff", "ic_lo", "ic_hi", "excluye_cero"]
        )
        writer.writeheader()
        writer.writerows(winner_rows)

    print(f"\nRanking por correccion: {', '.join(ranking)}")
    print(f"Ganador {top} contra {second}:")
    for r in winner_rows:
        print(
            f"  {r['eje']:<11} {r['diff']:+.3f} [{r['ic_lo']:+.3f}; {r['ic_hi']:+.3f}] "
            f"{'si' if r['excluye_cero'] else 'no'}"
        )

    fid = {(r["comparacion"], r["eje"]): r for r in h1_rows}
    sostiene = all(
        fid[(f"{base} - {rival}", eje)]["diff"] > 0 and fid[(f"{base} - {rival}", eje)]["excluye_cero"]
        for rival in ["qwen2.5:1.5b", "gemma2:2b"]
        for eje in ["correccion", "fidelidad"]
    )
    refuta = any(
        not (fid[(f"{base} - {rival}", "fidelidad")]["diff"] > 0 and fid[(f"{base} - {rival}", "fidelidad")]["excluye_cero"])
        and lat.get(rival, 1e9) < lat.get(base, 0)
        for rival in ["qwen2.5:1.5b", "gemma2:2b"]
    )
    if sostiene:
        print("\nVeredicto: H1 se sostiene.")
    elif refuta:
        print("\nVeredicto: H1 se refuta (un modelo de 1 a 2B iguala la fidelidad con menor latencia).")
    else:
        print("\nVeredicto: H1 queda indeterminada con esta muestra.")


if __name__ == "__main__":
    main()
