"""Variante v1 del experimento de limpieza."""
from src.rag.cleaning import DESCRIPTIONS
from src.rag.ingest_variant import run

if __name__ == "__main__":
    print(DESCRIPTIONS["v1"])
    run("v1", "rag_pilot_e5base_d768_v1_v1")
