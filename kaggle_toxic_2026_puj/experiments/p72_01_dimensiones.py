"""P1: capacidad de la representacion. 100d, 200d y 300d, todo lo demas congelado.

Que se aisla y que se congela
-----------------------------
La unica variable que cambia es `vector_size`. Todo lo demas queda identico:

    corpus            el mismo de Twitter, 10.000.029 tokens
    tokenizacion      EMB-NORMAL-TW v1
    window 5 | min_count 3 | negative 10 | sample 1e-4 | epochs 5 | sg 1
    seed 20261016 | lote 256 | lr = 0,025 x lote
    implementacion    skip-gram con negative sampling, desde cero, en PyTorch

Las 200 dimensiones ya existen y son validas (`w2v_twitter_lrfix.npz`, de p71_08, con la loss
bajando de 1,272 a 1,095). Se reutilizan en lugar de reentrenarlas, porque el encargo pide tres
entrenamientos y ese ya esta hecho y verificado.

Lo que NO se hace
-----------------
No se prueban 50d, 150d, 250d, 400d ni 500d. No hay grid. No se cambia el corpus, ni min_count,
ni window, ni negative, ni pooling, ni el protocolo de evaluacion.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

from p59_01_voronoi_local import V03_CCV, V03_SHA, V03_SUB

CORPORA = PROJECT_ROOT / "data" / "external" / "corpora" / "twitter_domain"
MODELOS = PROJECT_ROOT / "modelos" / "twitter_domain"
REPORTS = PROJECT_ROOT / "reports"

DIMENSIONES = (100, 300)   # 200 ya existe y es valido
LOTE = 256
LR0_POR_EJEMPLO = 0.025
LR_MINIMO = 1e-3


def log(m: str) -> None:
    """Imprime una linea de progreso.

    Args:
        m: texto a mostrar.
    """
    print(f"[P72-01] {m}", flush=True)


def entrenar(cx: dict, ruta: Path, dim: int, dev) -> dict:
    """Entrena un skip-gram de la dimension pedida.

    Args:
        cx: utilidades.
        ruta: corpus.
        dim: dimension.
        dev: device.

    Returns:
        Diccionario con el resultado.
    """
    m = cx["mod"]
    np.random.default_rng(m.SEMILLA)
    t0 = time.time()
    docs = cx["leer_tokens"](ruta)
    n_tokens = sum(len(d) for d in docs)
    vocab, p, cnt = cx["construir_vocabulario"](docs)
    log(f"   {dim}d: vocabulario {len(vocab):,} | tokens {n_tokens:,}")

    modelo = cx["SkipGram"](len(vocab), dim).to(dev)
    with torch.no_grad():
        modelo.salida.weight.uniform_(-0.05, 0.05)
    lr_ini = LR0_POR_EJEMPLO * LOTE
    optimo = torch.optim.SGD(modelo.parameters(), lr=lr_ini)
    idx = np.array([vocab[w] for w in vocab], dtype=np.int64)
    pesos = np.array([cnt[w] for w in vocab], dtype=np.float64) ** 0.75
    pesos = pesos / pesos.sum()

    planes = []
    for ep in range(m.EPOCAS):
        r2 = np.random.default_rng(m.SEMILLA + ep)
        c, _ = cx["pares_de_epoca"](docs, vocab, p, r2)
        planes.append(c)
    total_lotes = sum(max(1, (len(c) + LOTE - 1) // LOTE) for c in planes)

    perdidas, paso = [], 0
    for ep, _ in enumerate(planes, start=1):
        te = time.time()
        r2 = np.random.default_rng(m.SEMILLA + ep)
        centros, contextos = cx["pares_de_epoca"](docs, vocab, p, r2)
        o = r2.permutation(len(centros))
        centros, contextos = centros[o], contextos[o]
        suma, lotes = 0.0, 0
        for i in range(0, len(centros), LOTE):
            lr = max(LR_MINIMO, lr_ini * (1.0 - paso / max(total_lotes, 1)))
            for g in optimo.param_groups:
                g["lr"] = lr
            c = torch.from_numpy(centros[i:i + LOTE]).to(dev)
            x = torch.from_numpy(contextos[i:i + LOTE]).to(dev)
            neg = torch.from_numpy(r2.choice(idx, size=(len(c), m.NEGATIVOS),
                                             p=pesos)).to(dev)
            optimo.zero_grad()
            perdida = modelo(c, x, neg)
            perdida.backward()
            optimo.step()
            suma += float(perdida.item())
            lotes += 1
            paso += 1
        perdidas.append(round(suma / max(lotes, 1), 6))
        log(f"      epoca {ep}/{m.EPOCAS} | loss {perdidas[-1]:.6f} | {time.time() - te:.0f}s")

    W = modelo.entrada.weight.detach().cpu().numpy().astype(np.float32)
    MODELOS.mkdir(parents=True, exist_ok=True)
    salida = MODELOS / f"w2v_twitter_{dim}d.npz"
    np.savez_compressed(salida, W=W, palabras=np.array(list(vocab.keys()), dtype=object))
    bytes_ = salida.stat().st_size
    log(f"      artefacto {salida.name} | {bytes_ / 1e6:.1f} MB | "
        f"parametros {len(vocab) * dim * 2:,}")
    return {"dim": dim, "vocabulario": len(vocab), "tokens": n_tokens,
            "loss_por_epoca": perdidas, "baja": bool(perdidas[-1] < perdidas[0]),
            "segundos": round(time.time() - t0, 1), "bytes": bytes_,
            "parametros_embedding": len(vocab) * dim,
            "sha256": hashlib.sha256(salida.read_bytes()).hexdigest(),
            "artefacto": str(salida.relative_to(PROJECT_ROOT))}


def main() -> None:
    """Entrena las dimensiones que faltan.

    Returns:
        None.
    """
    inicio = datetime.now(timezone.utc)
    log("P72-01 P1: capacidad de la representacion, 100d y 300d")
    log("   las 200d ya existen y son validas, se reutilizan")
    hash_v03 = hashlib.sha256(V03_SUB.read_bytes()).hexdigest()
    if hash_v03 != V03_SHA:
        raise SystemExit("PARADA: la submission de V03 no coincide con el sha256 congelado")

    spec = importlib.util.spec_from_file_location(
        "base", PROJECT_ROOT / "experiments" / "p71_04_entrenar_word2vec.py")
    base = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(base)
    cx = {"sembrar": base.sembrar, "SkipGram": base.SkipGram, "leer_tokens": base.leer_tokens,
          "construir_vocabulario": base.construir_vocabulario,
          "pares_de_epoca": base.pares_de_epoca, "mod": base}
    dev = cx["sembrar"](base.SEMILLA)
    log(f"   todo congelado salvo vector_size: ventana {base.VENTANA} | "
        f"min_count {base.MIN_COUNT} | negativos {base.NEGATIVOS} | "
        f"muestreo {base.SUBMUESTREO} | epocas {base.EPOCAS} | sg 1 | semilla {base.SEMILLA}")

    resultados = []
    for dim in DIMENSIONES:
        log("")
        resultados.append(entrenar(cx, CORPORA / "corpus_twitter_tokens.txt", dim, dev))

    salida = {
        "familia": "P72_TWITTER_WORD2VEC_DIM", "experiment_id": "P72-01",
        "fase": "P1 capacidad de la representacion",
        "fecha": inicio.isoformat(timespec="seconds"),
        "naturaleza": "ENTRENAMIENTO. 2 dimensiones nuevas, la tercera ya existia",
        "aislado": {"variable": "vector_size", "valores": list(DIMENSIONES) + [200]},
        "congelado": {"corpus": "twitter 10.000.029 tokens",
                      "tokenizacion": "EMB-NORMAL-TW v1", "window": base.VENTANA,
                      "min_count": base.MIN_COUNT, "negative": base.NEGATIVOS,
                      "sample": base.SUBMUESTREO, "epochs": base.EPOCAS, "sg": 1,
                      "seed": base.SEMILLA, "lote": LOTE,
                      "lr": f"{LR0_POR_EJEMPLO} x {LOTE}"},
        "nuevas": resultados,
        "reproducibilidad": {"nivel": "R2", "hash_v03": hash_v03, "corrected_cv_v03": V03_CCV},
        "formulacion": "V03 remains the best verified leaderboard result.",
    }
    (REPORTS / "twitter_word2vec_dimensiones.json").write_text(
        json.dumps(salida, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    log("")
    log("=" * 78)
    for r in resultados:
        log(f"P72-01 {r['dim']}d OK | loss {r['loss_por_epoca'][0]} -> {r['loss_por_epoca'][-1]}"
            f" | baja {r['baja']} | {r['segundos']}s")
    log("=" * 78)


if __name__ == "__main__":
    ast.parse(Path(__file__).read_text(encoding="utf-8"))
    main()
