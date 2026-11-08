"""Evaluacion del silver v2 de ambiguedad (24 ambiguas y 21 controles)."""
from pathlib import Path

from src.scripts.experimento_ambiguedad import run


def main() -> None:
    run(
        Path("eval/silver/silver_ambiguedad_v2.jsonl"),
        Path("reports/corridas/ambiguedad_v2_raw.jsonl"),
        Path("reports/tablas/ambiguedad_v2_resumen.csv"),
        Path("reports/tablas/ambiguedad_v2_por_pregunta.csv"),
    )


if __name__ == "__main__":
    main()
