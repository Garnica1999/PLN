"""Concilia la anotacion humana doble del gold v2 de ambiguedad.

Lee las dos hojas, calcula el acuerdo y el kappa de Cohen sobre la etiqueta
de ambiguedad, escribe un informe por pregunta y, si se pide, genera el gold
consolidado. No sobrescribe el gold asistido por LLM.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

UNO = Path("eval/anotacion/anotacion_v2_anotador_1.csv")
DOS = Path("eval/anotacion/anotacion_v2_anotador_2.csv")
INFORME = Path("reports/tablas/anotacion_acuerdo.csv")
SALIDA = Path("eval/gold/gold_ambiguedad_v2_humano.jsonl")
GOLD = Path("eval/gold/gold_ambiguedad_v2.jsonl")


def leer(path: Path) -> dict[str, dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {row["id"]: row for row in rows}


def etiqueta(row: dict) -> str | None:
    value = (row.get("es_ambigua") or "").strip().lower()
    if value in {"si", "sí", "s", "1", "true"}:
        return "si"
    if value in {"no", "n", "0", "false"}:
        return "no"
    return None


def kappa(pares: list[tuple[str, str]]) -> float:
    n = len(pares)
    if n == 0:
        return 0.0
    acuerdo = sum(1 for a, b in pares if a == b) / n
    clases = {valor for par in pares for valor in par}
    esperado = sum(
        (sum(1 for a, _ in pares if a == clase) / n) * (sum(1 for _, b in pares if b == clase) / n)
        for clase in clases
    )
    if esperado >= 1.0:
        return 0.0
    return (acuerdo - esperado) / (1.0 - esperado)


def consolidar(uno: dict[str, dict], dos: dict[str, dict], etiquetas: dict[str, tuple[str, str]]) -> list[dict]:
    original = {
        row["id"]: row
        for row in (
            json.loads(line)
            for line in GOLD.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }
    filas = []
    for qid, (a, _) in etiquetas.items():
        base = dict(original[qid])
        if a == "si":
            row = uno[qid]
            opciones = []
            for index in (1, 2):
                label = (row.get(f"interpretacion_{index}") or "").strip()
                answer = (row.get(f"respuesta_{index}") or "").strip()
                if label and answer:
                    opciones.append({"label": label, "query": label, "answer": answer, "evidence": []})
            base.update(
                {
                    "type": "ambigua",
                    "intended_answer": (row.get("respuesta_1") or "").strip(),
                    "intended_aliases": [
                        value
                        for value in [(row.get("respuesta_1") or "").strip(), (row.get("respuesta_2") or "").strip()]
                        if value
                    ],
                    "options": opciones,
                    "source": "v2_humano",
                }
            )
        else:
            base.update({"type": "no_ambigua", "source": "v2_humano"})
        filas.append(base)
    return filas


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uno", type=Path, default=UNO)
    parser.add_argument("--dos", type=Path, default=DOS)
    parser.add_argument("--informe", type=Path, default=INFORME)
    parser.add_argument("--salida", type=Path, default=SALIDA)
    parser.add_argument("--escribir", action="store_true", help="genera el gold humano consolidado")
    args = parser.parse_args()

    uno, dos = leer(args.uno), leer(args.dos)
    if set(uno) != set(dos):
        raise SystemExit("las dos hojas no tienen los mismos ids")
    faltan = [qid for qid in uno if etiqueta(uno[qid]) is None or etiqueta(dos[qid]) is None]
    if faltan:
        raise SystemExit(f"faltan etiquetas en {len(faltan)} filas: {', '.join(faltan[:8])}")

    etiquetas = {qid: (etiqueta(uno[qid]), etiqueta(dos[qid])) for qid in uno}
    pares = list(etiquetas.values())
    acuerdo = sum(1 for a, b in pares if a == b)
    valor_kappa = kappa(pares)

    args.informe.parent.mkdir(parents=True, exist_ok=True)
    with args.informe.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "anotador_1", "anotador_2", "coincide", "pregunta"])
        for qid, (a, b) in etiquetas.items():
            writer.writerow([qid, a, b, "si" if a == b else "no", uno[qid]["pregunta"]])

    print(f"n {len(pares)}")
    print(f"acuerdo {acuerdo}/{len(pares)} ({acuerdo / len(pares):.3f})")
    print(f"kappa {valor_kappa:.3f}")
    discrepancias = [qid for qid, (a, b) in etiquetas.items() if a != b]
    print(f"discrepancias {len(discrepancias)}: {', '.join(discrepancias) or 'ninguna'}")

    if args.escribir:
        filas = consolidar(uno, dos, etiquetas)
        args.salida.write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in filas) + "\n",
            encoding="utf-8",
        )
        print(f"gold humano escrito en {args.salida}")


if __name__ == "__main__":
    main()
