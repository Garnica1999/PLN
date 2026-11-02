"""Ingesta base: variante v1 (filtro de calidad, texto intacto)."""
from src.rag.config import settings
from src.rag.ingest_variant import run

if __name__ == "__main__":
    run("v1", settings.collection, default_log="reports/tablas/ingesta.jsonl")
