"""Variante v4 del experimento de limpieza."""
from src.rag.cleaning import DESCRIPTIONS
from src.rag.ingest_variant import run

if __name__ == "__main__":
    print(DESCRIPTIONS["v4"])
    run("v4", "rag_pilot_e5base_d768_v4_v1")
