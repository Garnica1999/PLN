"""P1b-ter: el learning rate escalado por el lote. El bug, aislado y corregido.

El diagnostico, con el corpus diminuto que lo aisla
---------------------------------------------------
Sobre un corpus de dos familias de diez palabras que nunca se mezclan:

    salida a CERO      lr 0,05   loss 1,3863 -> 1,3682
    salida a CERO      lr 0,5    loss 1,3863 -> 0,2094    APRENDE
    salida ALEATORIA   lr 0,05   loss 1,3860 -> 1,2290
    salida ALEATORIA   lr 0,5    loss 1,3860 -> 0,2063    APRENDE

Dos conclusiones:

    el forward y el backward SON CORRECTOS: con learning rate suficiente el modelo aprende
    y la inicializacion a cero de la salida NO es fatal: solo hace falta que el paso sea grande

**El bug era el learning rate.** Con la loss promediada sobre lotes de 8192, el gradiente es
el promedio, o sea 8192 veces mas pequeno que el gradiente por ejemplo que usa word2vec. El
learning rate efectivo por ejemplo era 0,025/8192 = 3e-6, en vez de 0,025.

Los cuatro intentos anteriores fallaron porque los cuatro probaban otra cosa: la escala de la
entrada, el optimizador, el tamano del lote sin escalar el learning rate, o el decaimiento.
Ninguno tocaba el cociente entre el learning rate y el lote.

La correccion
-------------
learning rate inicial = 0,025 x tamano del lote, que es la regla de escalado lineal. Con lote
256 son 6,4. Se comprobo que con lote 64 y lr 0,5 el modelo aprende, luego el punto de partida
esta en la misma region.

Se anade ademas la inicializacion aleatoria de la salida, que el test mostro que ayuda cuando
el learning rate no es muy grande.

Lo que NO cambia: dim, ventana, min_count, negativos, muestreo, epocas, sg, semilla, corpus,
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

LOTE = 256
LR0_POR_EJEMPLO = 0.025
LR_MINIMO = 1e-3


def log(m: str) -> None:
    """Imprime una linea de progreso.

    Args:
        m: texto a mostrar.
    """
    print(f"[P71-08] {m}", flush=True)


def entrenar(cx: dict, ruta: Path, etiqueta: str, dev) -> dict:
    """Entrena con el learning rate escalado por el lote.

    Args:
        cx: utilidades.
        ruta: corpus.
        etiqueta: nombre.
        dev: device.

    Returns:
        Diccionario con metricas.
    """
    m = cx["mod"]
    np.random.default_rng(m.SEMILLA)
    t0 = time.time()
    docs = cx["leer_tokens"](ruta)
    n_tokens = sum(len(d) for d in docs)
    vocab, p, cnt = cx["construir_vocabulario"](docs)
    log(f"   {etiqueta}: {len(docs):,} documentos | tokens {n_tokens:,} | "
        f"vocabulario {len(vocab):,}")

    modelo = cx["SkipGram"](len(vocab), m.DIM).to(dev)
    with torch.no_grad():
        # La salida ya no se queda a cero: el test mostro que ayuda.
        modelo.salida.weight.uniform_(-0.05, 0.05)
    lr_ini = LR0_POR_EJEMPLO * LOTE
    optimo = torch.optim.SGD(modelo.parameters(), lr=lr_ini)
    log(f"      lr inicial {lr_ini:.4f} = {LR0_POR_EJEMPLO} x {LOTE} (escalado por el lote)")

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
        log(f"      epoca {ep}/{m.EPOCAS} | loss {perdidas[-1]:.6f} | lr final {lr:.4f} | "
            f"{time.time() - te:.0f}s")

    W = modelo.entrada.weight.detach().cpu().numpy().astype(np.float32)
    MODELOS.mkdir(parents=True, exist_ok=True)
    salida = MODELOS / f"w2v_{etiqueta}_lrfix.npz"
    np.savez_compressed(salida, W=W, palabras=np.array(list(vocab.keys()), dtype=object))
    return {"etiqueta": etiqueta, "tokens": n_tokens, "vocabulario": len(vocab),
            "loss_por_epoca": perdidas, "segundos": round(time.time() - t0, 1),
            "modelo": str(salida.relative_to(PROJECT_ROOT)),
            "sha256_modelo": hashlib.sha256(salida.read_bytes()).hexdigest(),
            "baja": bool(perdidas[-1] < perdidas[0])}


def main() -> None:
    """Reentrena con el learning rate corregido.

    Returns:
        None.
    """
    inicio = datetime.now(timezone.utc)
    log("P71-08 P1b-ter: learning rate escalado por el lote")
    hash_real = hashlib.sha256(V03_SUB.read_bytes()).hexdigest()
    if hash_real != V03_SHA:
        raise SystemExit("PARADA: la submission de V03 no coincide con el sha256 congelado")

    spec = importlib.util.spec_from_file_location(
        "base", PROJECT_ROOT / "experiments" / "p71_04_entrenar_word2vec.py")
    base = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(base)
    cx = {"sembrar": base.sembrar, "SkipGram": base.SkipGram, "leer_tokens": base.leer_tokens,
          "construir_vocabulario": base.construir_vocabulario,
          "pares_de_epoca": base.pares_de_epoca, "mod": base}
    dev = cx["sembrar"](base.SEMILLA)
    log(f"   config sin cambios: dim {base.DIM} | ventana {base.VENTANA} | "
        f"min_count {base.MIN_COUNT} | negativos {base.NEGATIVOS} | epocas {base.EPOCAS} | "
        f"sg 1 | semilla {base.SEMILLA}")

    tw = entrenar(cx, CORPORA / "corpus_twitter_tokens.txt", "twitter", dev)
    ge = entrenar(cx, CORPORA / "corpus_general_tokens.txt", "general", dev)

    salida = {
        "familia": "P71_TWITTER_WORD2VEC", "experiment_id": "P71-08",
        "fase": "P1b-ter learning rate corregido",
        "fecha": inicio.isoformat(timespec="seconds"),
        "el_bug_aislado": {
            "que": ("la loss promediada sobre lotes de 8192 hace el gradiente 8192 veces mas "
                    "pequeno. El learning rate efectivo por ejemplo era 0,025/8192 = 3e-6"),
            "como_se_aislo": ("corpus diminuto de dos familias de diez palabras que nunca se "
                              "mezclan. Con lr 0,5 la loss baja de 1,3863 a 0,2094"),
            "el_forward_es_correcto": True,
            "la_salida_a_cero_no_es_fatal": True,
            "por_que_fallaron_los_intentos_previos": ("los cuatro probaban otra cosa: escala de "
                                                      "la entrada, optimizador, lote sin escalar "
                                                      "el lr, o decaimiento. Ninguno tocaba el "
                                                      "cociente entre lr y lote"),
        },
        "la_correccion": {"lr_inicial": LR0_POR_EJEMPLO * LOTE, "regla": "0,025 x lote",
                          "lote": LOTE, "lr_minimo": LR_MINIMO,
                          "salida_inicializada": "aleatoria uniforme [-0,05, 0,05]"},
        "lo_que_no_cambia": ["dim", "ventana", "min_count", "negativos", "muestreo", "epocas",
                             "sg", "semilla", "corpus", "tokenizacion"],
        "corpus_twitter": tw, "corpus_general": ge,
        "reproducibilidad": {"nivel": "R2", "hash_v03": hash_real, "corrected_cv_v03": V03_CCV},
        "formulacion": "V03 remains the best verified leaderboard result.",
    }
    (REPORTS / "twitter_word2vec_training_lrfix.json").write_text(
        json.dumps(salida, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    log("")
    log(f"   twitter baja: {tw['baja']} | {tw['loss_por_epoca']}")
    log(f"   general baja: {ge['baja']} | {ge['loss_por_epoca']}")
    ok = tw["baja"] and ge["baja"] and tw["loss_por_epoca"][-1] < 1.0
    log("")
    log("=" * 78)
    log(f"P71-08 OK | {'LAS DOS BAJAN Y APRENDEN' if ok else 'REVISAR'}")
    log("=" * 78)


if __name__ == "__main__":
    ast.parse(Path(__file__).read_text(encoding="utf-8"))
    main()
