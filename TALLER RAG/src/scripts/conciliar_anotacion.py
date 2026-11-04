"""Concilia la doble anotacion del gold v2 leyendo el libro de Excel.

Lee las hojas «Anotador 1» y «Anotador 2» de eval/anotacion/anotacion_v2.xlsx,
calcula el acuerdo y el kappa de Cohen sobre la etiqueta de ambiguedad,
escribe un informe por pregunta y, si se pide, genera el gold consolidado.
No sobrescribe el gold asistido por LLM.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from openpyxl import load_workbook

LIBRO = Path("eval/anotacion/anotacion_v2.xlsx")
INFORME = Path("reports/tablas/anotacion_acuerdo.csv")
SALIDA = Path("eval/gold/gold_ambiguedad_v2_humano.jsonl")
GOLD = Path("eval/gold/gold_ambiguedad_v2.jsonl")


def hoja_con_datos(libro: Path) -> str:
    """Primera hoja de anotacion que tenga etiquetas."""
    wb = load_workbook(libro, data_only=True)
    for name in ("Anotador 1", "Anotador 2"):
        if name in wb.sheetnames:
            ws = wb[name]
            if any(ws.cell(row=row, column=5).value for row in range(2, ws.max_row + 1)):
                return name
    raise SystemExit(f"{libro} no tiene etiquetas en ninguna hoja de anotacion")


def leer(libro: Path, hoja: str) -> dict[str, dict]:
    """Devuelve las filas de una hoja indexadas por id."""
    sheet = load_workbook(libro, data_only=True)[hoja]
    header = [cell.value for cell in sheet[1]]
    rows = {}
    for values in sheet.iter_rows(min_row=2, values_only=True):
        row = dict(zip(header, values))
        if row.get("id"):
            rows[str(row["id"])] = row
    return rows


def etiqueta(row: dict) -> str | None:
    value = str(row.get("es_ambigua") or "").strip().lower()
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
                label = str(row.get(f"interpretacion_{index}") or "").strip()
                answer = str(row.get(f"respuesta_{index}") or "").strip()
                if label and answer:
                    opciones.append({"label": label, "query": label, "answer": answer, "evidence": []})
            base.update(
                {
                    "type": "ambigua",
                    "intended_answer": str(row.get("respuesta_1") or "").strip(),
                    "intended_aliases": [
                        value
                        for value in [
                            str(row.get("respuesta_1") or "").strip(),
                            str(row.get("respuesta_2") or "").strip(),
                        ]
                        if value
                    ],
                    "options": opciones,
                    "source": "v2_humano",
                }
            )
        else:
            base.update(
                {
                    "type": "no_ambigua",
                    "intended_answer": "",
                    "intended_aliases": [],
                    "options": [],
                    "conditional_answers": [],
                    "source": "v2_humano",
                }
            )
        filas.append(base)
    return filas


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--libro", type=Path, default=LIBRO)
    parser.add_argument("--uno", type=Path, help="libro del anotador 1")
    parser.add_argument("--dos", type=Path, help="libro del anotador 2")
    parser.add_argument("--informe", type=Path, default=INFORME)
    parser.add_argument("--salida", type=Path, default=SALIDA)
    parser.add_argument("--escribir", action="store_true", help="genera el gold humano consolidado")
    args = parser.parse_args()

    if args.uno and args.dos:
        uno = leer(args.uno, hoja_con_datos(args.uno))
        dos = leer(args.dos, hoja_con_datos(args.dos))
    else:
        uno = leer(args.libro, "Anotador 1")
        dos = leer(args.libro, "Anotador 2")
    if set(uno) != set(dos):
        raise SystemExit("las dos hojas no tienen los mismos ids")
    faltan = [qid for qid in uno if etiqueta(uno[qid]) is None or etiqueta(dos[qid]) is None]
    if faltan:
        raise SystemExit(f"faltan etiquetas en {len(faltan)} filas: {', '.join(sorted(faltan)[:8])}")

    etiquetas = {qid: (etiqueta(uno[qid]), etiqueta(dos[qid])) for qid in uno}
    pares = list(etiquetas.values())
    acuerdo = sum(1 for a, b in pares if a == b)
    valor_kappa = kappa(pares)

    args.informe.parent.mkdir(parents=True, exist_ok=True)
    with args.informe.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle, delimiter=";", lineterminator="\n")
        writer.writerow(["id", "anotador_1", "anotador_2", "coincide", "pregunta"])
        for qid, (a, b) in etiquetas.items():
            writer.writerow([qid, a, b, "si" if a == b else "no", uno[qid]["pregunta"]])

    print(f"n {len(pares)}")
    print(f"acuerdo {acuerdo}/{len(pares)} ({acuerdo / len(pares):.3f})")
    print(f"kappa {valor_kappa:.3f}")
    discrepancias = sorted(qid for qid, (a, b) in etiquetas.items() if a != b)
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
