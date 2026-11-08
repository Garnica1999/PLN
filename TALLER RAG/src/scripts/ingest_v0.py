"""Variante v0 del experimento de limpieza."""
from src.rag.cleaning import DESCRIPTIONS
from src.rag.ingest_variant import run

if __name__ == "__main__":
    print(DESCRIPTIONS["v0"])
    run("v0", "rag_pilot_e5base_d768_v0_v1")
