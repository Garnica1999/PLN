"""Evaluacion del silver v2 humano de ambiguedad (42 preguntas, doble anotacion)."""
from pathlib import Path

from src.scripts.experimento_ambiguedad import run


def main() -> None:
    run(
        Path("eval/silver/silver_ambiguedad_v2_humano.jsonl"),
        Path("reports/corridas/ambiguedad_v2_humano_raw.jsonl"),
        Path("reports/tablas/ambiguedad_v2_humano_resumen.csv"),
        Path("reports/tablas/ambiguedad_v2_humano_por_pregunta.csv"),
    )


if __name__ == "__main__":
    main()
