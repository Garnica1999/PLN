"""Mide e5-base en el dispositivo configurado."""
from __future__ import annotations

import json
import os
import time

import torch
from sentence_transformers import SentenceTransformer

MODEL = "intfloat/multilingual-e5-base"
TEXTS = [("Texto de prueba sobre economia y politica en Colombia. " * 15)[:900] for _ in range(64)]


def main() -> None:
    model = SentenceTransformer(MODEL, device=os.getenv("EMBEDDING_DEVICE", "cpu"))
    model.encode(["passage: warmup"], normalize_embeddings=True)
    info = {
        "device": str(model.device),
        "torch": torch.__version__,
        "hip": getattr(torch.version, "hip", None),
        "cuda_disponible": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
    rows = []
    for batch in (32, 64, 128):
        started = time.perf_counter()
        vectors = model.encode(
            ["passage: " + text for text in TEXTS],
            normalize_embeddings=True,
            batch_size=batch,
        )
        elapsed = time.perf_counter() - started
        rows.append(
            {
                "batch": batch,
                "segundos": round(elapsed, 2),
                "chunks_por_segundo": round(len(TEXTS) / elapsed, 1),
                "dim": int(vectors.shape[1]),
            }
        )
    print(json.dumps({**info, "resultados": rows}, ensure_ascii=False))


if __name__ == "__main__":
    main()
