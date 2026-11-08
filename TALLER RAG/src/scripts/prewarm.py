"""Precalienta el generador, el recuperador y el reranker antes de la demo."""
from __future__ import annotations

import os
import time

import requests

API = os.getenv("API_URL", "http://localhost:8010").rstrip("/")
QUERY = "¿Cuándo se celebra el Global Big Day?"
MODEL = os.getenv("PREWARM_MODEL", "gemma4:e4b")


def main() -> None:
    started = time.perf_counter()
    response = requests.post(
        f"{API}/chat",
        json={"query": QUERY, "model": MODEL, "k": 5, "think": False, "strict": False},
        timeout=600,
    )
    response.raise_for_status()
    data = response.json()
    elapsed = time.perf_counter() - started
    print(
        f"precalentado en {elapsed:.1f} s: {data.get('model')} con "
        f"{len(data.get('evidence', []))} evidencias y {data.get('total_ms', 0):.0f} ms"
    )


if __name__ == "__main__":
    main()
