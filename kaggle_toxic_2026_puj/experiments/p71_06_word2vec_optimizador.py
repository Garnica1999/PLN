"""P1b-bis: el mismo Word2Vec con el optimizador corregido. SGD con decaimiento lineal.

Que se arregla y que NO se toca
--------------------------------
El encargo fija la configuracion del modelo:

    vector_size 200 | window 5 | min_count 3 | negative 10 | sample 1e-4 | epochs 5 | sg 1

y **no fija el optimizador**. Yo puse Adam con un learning rate fijo de 0,025, que es el valor
por defecto de gensim pero para SGD. Con Adam ese valor es demasiado alto, y ademas el
desbalance de 10 negativos por cada positivo hace que Adam amplifique la direccion negativa.

El sintoma es inconfundible y esta en el registro de la corrida anterior:

    twitter   loss 1,706 -> 1,811 -> 1,842 -> 1,866 -> 1,884
    general   loss 1,888 -> 2,068 -> 2,126 -> 2,164 -> 2,190

La loss esperada con embeddings aleatorios y 10 negativos es 7,61. La primera epoca baja a 1,71,
luego SUBE: el modelo aprende algo y el learning rate lo deshace.

Que cambia: el optimizador. SGD con decaimiento lineal del learning rate, que es lo que usa
word2vec desde el articulo original y lo que usa gensim por debajo.

Que NO cambia: nada del modelo. Misma dimension, misma ventana, mismo min_count, mismos
negativos, mismo muestreo, mismas epocas, mismo sg, misma semilla, mismos corpus, misma
tokenizacion.
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
REPORTS = PROJECT_ROOT / "reports"
MODELOS = PROJECT_ROOT / "modelos" / "twitter_domain"

# El learning rate INICIAL es el mismo 0,025 que en la corrida anterior. Lo que cambia es el
# optimizador y el decaimiento, no el punto de partida.
LR_INICIAL = 0.025
LR_MINIMO = 1e-4


def log(m: str) -> None:
    """Imprime una linea de progreso.

    Args:
        m: texto a mostrar.
    """
    print(f"[P71-06] {m}", flush=True)


def cargar_base() -> dict:
    """Carga el modulo del primer entrenamiento sin ejecutar su main.

    Returns:
        Las funciones y constantes que se reutilizan.
    """
    spec = importlib.util.spec_from_file_location(
        "base", PROJECT_ROOT / "experiments" / "p71_04_entrenar_word2vec.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return {"sembrar": mod.sembrar, "SkipGram": mod.SkipGram, "leer_tokens": mod.leer_tokens,
            "construir_vocabulario": mod.construir_vocabulario,
            "pares_de_epoca": mod.pares_de_epoca, "mod": mod}


def entrenar(cx: dict, ruta: Path, etiqueta: str) -> dict:
    """Entrena con SGD y decaimiento lineal.

    Args:
        cx: utilidades cargadas.
        ruta: corpus.
        etiqueta: nombre.

    Returns:
        Diccionario con el modelo y las metricas.
    """
    m = cx["mod"]
    dev = cx["sembrar"](m.SEMILLA)
    rng = np.random.default_rng(m.SEMILLA)

    t0 = time.time()
    docs = cx["leer_tokens"](ruta)
    n_tokens = sum(len(d) for d in docs)
    vocab, p, cnt = cx["construir_vocabulario"](docs)
    log(f"   {etiqueta}: {len(docs):,} documentos | vocabulario {len(vocab):,}")

    modelo = cx["SkipGram"](len(vocab), m.DIM).to(dev)
    # SGD en lugar de Adam: es lo que usa word2vec y lo que tolera un LR de 0,025.
    optimo = torch.optim.SGD(modelo.parameters(), lr=LR_INICIAL)
    idx = np.array([vocab[w] for w in vocab], dtype=np.int64)
    pesos = np.array([cnt[w] for w in vocab], dtype=np.float64) ** 0.75
    pesos = pesos / pesos.sum()

    # El decaimiento es sobre el total de actualizaciones, como en el articulo original.
    total_lotes = 0
    planes = []
    for _ in range(m.EPOCAS):
        c, _ = cx["pares_de_epoca"](docs, vocab, p, rng)
        planes.append(c)
    total_lotes = sum(max(1, (len(c) + m.LOTE - 1) // m.LOTE) for c in planes)
    log(f"      actualizaciones previstas: {total_lotes:,}")

    perdidas = []
    paso = 0
    for ep, centros_base in enumerate(planes, start=1):
        te = time.time()
        rng2 = np.random.default_rng(m.SEMILLA + ep)
        centros, contextos = cx["pares_de_epoca"](docs, vocab, p, rng2)
        orden = rng2.permutation(len(centros))
        centros, contextos = centros[orden], contextos[orden]
        suma, lotes = 0.0, 0
        for i in range(0, len(centros), m.LOTE):
            # Decaimiento lineal del learning rate, como word2vec.
            lr = max(LR_MINIMO, LR_INICIAL * (1.0 - paso / max(total_lotes, 1)))
            for g in optimo.param_groups:
                g["lr"] = lr
            c = torch.from_numpy(centros[i:i + m.LOTE]).to(dev)
            x = torch.from_numpy(contextos[i:i + m.LOTE]).to(dev)
            neg = torch.from_numpy(
                rng2.choice(idx, size=(len(c), m.NEGATIVOS), p=pesos)).to(dev)
            optimo.zero_grad()
            perdida = modelo(c, x, neg)
            perdida.backward()
            optimo.step()
            suma += float(perdida.item())
            lotes += 1
            paso += 1
        media = suma / max(lotes, 1)
        perdidas.append(round(media, 6))
        log(f"      epoca {ep}/{m.EPOCAS} | loss {media:.6f} | lr final {lr:.6f} | "
            f"{time.time() - te:.0f}s")

    W = modelo.entrada.weight.detach().cpu().numpy().astype(np.float32)
    MODELOS.mkdir(parents=True, exist_ok=True)
    salida = MODELOS / f"w2v_{etiqueta}_sgd.npz"
    np.savez_compressed(salida, W=W, palabras=np.array(list(vocab.keys()), dtype=object))
    return {"etiqueta": etiqueta, "tokens": n_tokens, "vocabulario": len(vocab),
            "loss_por_epoca": perdidas, "segundos": round(time.time() - t0, 1),
            "modelo": str(salida.relative_to(PROJECT_ROOT)),
            "sha256_modelo": hashlib.sha256(salida.read_bytes()).hexdigest()}


def main() -> None:
    """Reentrena los dos modelos con el optimizador corregido.

    Returns:
        None.
    """
    inicio = datetime.now(timezone.utc)
    log("P71-06 P1b-bis: mismo modelo, optimizador corregido")
    hash_real = hashlib.sha256(V03_SUB.read_bytes()).hexdigest()
    if hash_real != V03_SHA:
        raise SystemExit("PARADA: la submission de V03 no coincide con el sha256 congelado")

    cx = cargar_base()
    m = cx["mod"]
    log(f"   config del modelo SIN CAMBIOS: dim {m.DIM} | ventana {m.VENTANA} | "
        f"min_count {m.MIN_COUNT} | negativos {m.NEGATIVOS} | muestreo {m.SUBMUESTREO} | "
        f"epocas {m.EPOCAS} | sg 1 | semilla {m.SEMILLA}")
    log(f"   lo unico que cambia: SGD con decaimiento lineal desde {LR_INICIAL}")

    tw = entrenar(cx, CORPORA / "corpus_twitter_tokens.txt", "twitter")
    ge = entrenar(cx, CORPORA / "corpus_general_tokens.txt", "general")

    log("")
    log("   COMPARACION DE LAS CURVAS DE LOSS")
    log(f"      {'epoca':6}{'Adam twitter':>16}{'SGD twitter':>16}{'Adam general':>16}"
        f"{'SGD general':>16}")
    previo = json.loads((REPORTS / "twitter_word2vec_training.json")
                        .read_text(encoding="utf-8"))
    for i in range(m.EPOCAS):
        log(f"      {i + 1:<6}{previo['corpus_twitter']['loss_por_epoca'][i]:>16.6f}"
            f"{tw['loss_por_epoca'][i]:>16.6f}"
            f"{previo['corpus_general']['loss_por_epoca'][i]:>16.6f}"
            f"{ge['loss_por_epoca'][i]:>16.6f}")

    baja_tw = tw["loss_por_epoca"][-1] < tw["loss_por_epoca"][0]
    baja_ge = ge["loss_por_epoca"][-1] < ge["loss_por_epoca"][0]
    log("")
    log(f"      la loss BAJA en twitter: {baja_tw}")
    log(f"      la loss BAJA en general: {baja_ge}")

    salida = {
        "familia": "P71_TWITTER_WORD2VEC", "experiment_id": "P71-06",
        "fase": "P1b-bis optimizador corregido",
        "fecha": inicio.isoformat(timespec="seconds"),
        "por_que": ("la loss subia en la corrida anterior. El encargo fija la configuracion del "
                    "modelo pero NO el optimizador; Adam con lr 0,025 deshacia lo aprendido"),
        "lo_que_cambia": {"optimizador": "Adam -> SGD con decaimiento lineal",
                          "lr_inicial": LR_INICIAL, "lr_minimo": LR_MINIMO},
        "lo_que_NO_cambia": ["dim", "ventana", "min_count", "negativos", "muestreo", "epocas",
                             "sg", "semilla", "corpus", "tokenizacion"],
        "configuracion": {"vector_size": m.DIM, "window": m.VENTANA, "min_count": m.MIN_COUNT,
                          "negative": m.NEGATIVOS, "sample": m.SUBMUESTREO,
                          "epochs": m.EPOCAS, "sg": 1, "seed": m.SEMILLA,
                          "optimizador": "SGD con decaimiento lineal"},
        "corpus_twitter": tw, "corpus_general": ge,
        "loss_anterior_adam": {"twitter": previo["corpus_twitter"]["loss_por_epoca"],
                               "general": previo["corpus_general"]["loss_por_epoca"]},
        "la_loss_baja": {"twitter": bool(baja_tw), "general": bool(baja_ge)},
        "reproducibilidad": {"nivel": "R2", "hash_v03": hash_real, "corrected_cv_v03": V03_CCV},
        "formulacion": "V03 remains the best verified leaderboard result.",
    }
    (REPORTS / "twitter_word2vec_training_sgd.json").write_text(
        json.dumps(salida, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    log("")
    log("=" * 78)
    log(f"P71-06 OK | loss baja en twitter {baja_tw} | general {baja_ge}")
    log(f"   twitter {tw['loss_por_epoca']}")
    log(f"   general {ge['loss_por_epoca']}")
    log("=" * 78)


if __name__ == "__main__":
    ast.parse(Path(__file__).read_text(encoding="utf-8"))
    main()
