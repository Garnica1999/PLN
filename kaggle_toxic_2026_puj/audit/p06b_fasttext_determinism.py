"""P6b: reproducibilidad de FASTTEXT-BASELINE (skipgram, dim 200, seed 42).

1. Entrena el mismo modelo dos veces (misma configuracion) y compara sha256:
   mide determinismo intra-configuracion con 16 hilos.
2. Compara contra el .vec almacenado (referencia read-only del data compartido):
   igualdad de vocabulario, coseno centrado medio y solapamiento de vecinos
   top-10 sobre 12 palabras (replica la metrica del informe de fase v5).
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/home/carlos/Git/PLN-toxic-rerun/kaggle_toxic_2026_puj")
sys.path.insert(0, str(ROOT))

from src.embedding_utils import entrenar_fasttext  # noqa: E402

SHARED = Path("/home/carlos/Git/PLN/kaggle_toxic_2026_puj")
OUT = ROOT / "audit" / "out_p06" / "fasttext"
OUT.mkdir(parents=True, exist_ok=True)
BINARIO = ROOT / "tools" / "fastText" / "build" / "fasttext"
CORPUS = ROOT / "data" / "external" / "corpora" / "wikipedia_es" / "emb_normal" / "tokens_10M.txt"
PALABRAS = ["violencia", "inmigración", "gobierno", "mujer", "niño", "familia",
            "iglesia", "fútbol", "música", "trabajo", "ciudad", "guerra"]


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cargar_vec(path: Path) -> tuple[list[str], np.ndarray]:
    palabras, vectores = [], []
    with path.open(encoding="utf-8", errors="ignore") as fh:
        header = fh.readline().split()
        for line in fh:
            partes = line.rstrip().split(" ")
            if len(partes) == int(header[1]) + 1:
                palabras.append(partes[0])
                vectores.append(np.asarray(partes[1:], dtype=np.float32))
    return palabras, np.vstack(vectores)


def top10(palabra: str, indice: dict[str, int], matriz: np.ndarray,
          palabras: list[str]) -> tuple[set[str], np.ndarray] | None:
    """Top-10 vecinos por coseno centrado y vector centrado de la palabra."""
    if palabra not in indice:
        return None
    centro = matriz.mean(axis=0)
    dif = matriz - centro
    consulta = dif[indice[palabra]]
    sim = dif @ consulta / (np.linalg.norm(dif, axis=1) * np.linalg.norm(consulta) + 1e-12)
    sim[indice[palabra]] = -np.inf
    return {palabras[i] for i in np.argsort(-sim)[:10]}, consulta


def main() -> None:
    registro: dict = {}
    prefijos = {}
    for etiqueta in ("run1", "run2"):
        prefijo = OUT / f"baseline_seed42_{etiqueta}"
        salida = entrenar_fasttext(
            BINARIO, CORPUS, prefijo,
            dim=200, ventana=5, min_count=3, negativos=10, epocas=5,
            hilos=16, semilla=42, minn=3, maxn=6, bucket=200_000,
        )
        prefijos[etiqueta] = Path(f"{prefijo}.vec")
        registro[etiqueta] = {"comando": salida["comando"],
                              "sha256_vec": sha256_of(prefijos[etiqueta])}
    registro["determinismo_intra_config"] = (
        registro["run1"]["sha256_vec"] == registro["run2"]["sha256_vec"])

    almacenado = SHARED / "data" / "external" / "embeddings" / "fasttext_baseline.vec"
    registro["almacenado"] = {"ruta": str(almacenado), "sha256": sha256_of(almacenado)}
    registro["igual_al_almacenado"] = registro["run1"]["sha256_vec"] == registro["almacenado"]["sha256"]

    palabras_a, vec_a = cargar_vec(almacenado)
    palabras_b, vec_b = cargar_vec(prefijos["run1"])
    indice_a = {p: i for i, p in enumerate(palabras_a)}
    indice_b = {p: i for i, p in enumerate(palabras_b)}
    registro["vocabulario"] = {
        "n_almacenado": len(palabras_a), "n_run1": len(palabras_b),
        "igual": set(palabras_a) == set(palabras_b),
        "en_comun": len(set(palabras_a) & set(palabras_b)),
    }

    solapamientos, cosenos = [], []
    for palabra in PALABRAS:
        ta = top10(palabra, indice_a, vec_a, palabras_a)
        tb = top10(palabra, indice_b, vec_b, palabras_b)
        if ta and tb:
            solapamientos.append(len(ta[0] & tb[0]) / 10.0)
            ca, cb = ta[1], tb[1]
            cosenos.append(float(ca @ cb / (np.linalg.norm(ca) * np.linalg.norm(cb) + 1e-12)))
    registro["comparacion_semantica_run1_vs_almacenado"] = {
        "palabras_evaluadas": len(solapamientos),
        "solapamiento_medio_top10": round(float(np.mean(solapamientos)), 4) if solapamientos else None,
        "coseno_centrado_medio": round(float(np.mean(cosenos)), 4) if cosenos else None,
    }
    (OUT / "p06b_fasttext.json").write_text(json.dumps(registro, indent=2), encoding="ascii")
    print(json.dumps({k: v for k, v in registro.items() if k != "run1"}, indent=2, ensure_ascii=True)[:2000])


if __name__ == "__main__":
    main()
