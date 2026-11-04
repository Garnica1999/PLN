"""Ambiguedad en otros escenarios: temporal y de alcance."""
from pathlib import Path

from src.scripts.experimento_ambiguedad import run


def main() -> None:
    run(
        Path("eval/silver/silver_ambiguedad_otros.jsonl"),
        Path("reports/corridas/ambiguedad_otros_raw.jsonl"),
        Path("reports/tablas/ambiguedad_otros_resumen.csv"),
        Path("reports/tablas/ambiguedad_otros_por_pregunta.csv"),
    )


if __name__ == "__main__":
    main()
