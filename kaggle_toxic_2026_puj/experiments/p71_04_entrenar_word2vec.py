"""P1b: Word2Vec desde cero, sobre los dos corpus, con la misma configuracion.

Por que PyTorch y no gensim
---------------------------
Carlos autorizo usar gensim. gensim **no compila en Python 3.14**: su Cython genera C contra
APIs que esa version elimino (`PyDictObject.ma_version_tag`), y no hay wheel. Se intento y fallo
con el error a la vista.

La propia spec de v5 documenta el respaldo, literalmente:

    "fastText C++ oficial (g++/cmake/make). Word2Vec-control con minn=0, maxn=0.
     PyTorch solo como respaldo/control."

Asi que no es una desviacion: es el respaldo que la spec preveia, y ademas encaja con el stack
del proyecto, que ya es PyTorch con ROCm.

Y cumple el requisito de Carlos con mas fuerza que gensim: **el skip-gram con negative sampling
esta implementado desde cero**, no es una caja negra de una libreria. El notebook puede mostrar
la loss, la construccion de pares y la actualizacion.

La configuracion, la del encargo, sin tocar nada
------------------------------------------------
    vector_size 200 | window 5 | min_count 3 | negative 10
    sample 1e-4 | epochs 5 | sg 1 | seed fija

Que se registra
---------------
vocabulario, cobertura, OOV, loss por epoca, tiempo de entrenamiento y hash del modelo. El
mismo pipeline exacto para los dos corpus: lo unico que cambia es el texto.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

from p59_01_voronoi_local import V03_CCV, V03_SHA, V03_SUB

CORPORA = PROJECT_ROOT / "data" / "external" / "corpora" / "twitter_domain"
REPORTS = PROJECT_ROOT / "reports"
MODELOS = PROJECT_ROOT / "modelos" / "twitter_domain"

DIM = 200
VENTANA = 5
MIN_COUNT = 3
NEGATIVOS = 10
SUBMUESTREO = 1e-4
EPOCAS = 5
SEMILLA = 20261016
LOTE = 8192
LR = 0.025
RE_TOKEN = re.compile(r"[#@]?\w+(?:['\u2019]\w+)*", re.UNICODE)


def sembrar(semilla: int) -> torch.device:
    """Siembra TODO lo que interviene, para que el notebook de la misma salida.

    No basta con torch.manual_seed: hay cuatro generadores distintos en juego y uno de ellos
    vive en la GPU. Y las operaciones de GPU no son deterministas por defecto, asi que ademas
    hay que pedirlo explicitamente. Sin esto, dos ejecuciones del mismo codigo dan embeddings
    distintos y el notebook no seria reproducible.

    Args:
        semilla: semilla.

    Returns:
        El device que se va a usar.
    """
    import os
    import random as _random
    os.environ["PYTHONHASHSEED"] = str(semilla)
    _random.seed(semilla)
    np.random.seed(semilla)
    torch.manual_seed(semilla)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dev.type == "cuda":
        torch.cuda.manual_seed(semilla)
        torch.cuda.manual_seed_all(semilla)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except (RuntimeError, AttributeError) as exc:
            print(f"[P71-04] aviso: no se pudo activar el modo determinista: {exc}")
    return dev


def log(m: str) -> None:
    """Imprime una linea de progreso.

    Args:
        m: texto a mostrar.
    """
    print(f"[P71-04] {m}", flush=True)


def leer_tokens(ruta: Path) -> list[list[str]]:
    """Lee el corpus y devuelve la lista de documentos tokenizados.

    Args:
        ruta: fichero con un documento normalizado por linea.

    Returns:
        Lista de listas de tokens.
    """
    docs = []
    with ruta.open(encoding="utf-8", errors="replace") as fh:
        for linea in fh:
            toks = RE_TOKEN.findall(linea)
            if len(toks) >= 3:
                docs.append(toks)
    return docs


def construir_vocabulario(docs: list[list[str]]) -> tuple[dict[str, int], np.ndarray, Counter]:
    """Construye el vocabulario con min_count y la tabla de subsampling.

    Args:
        docs: documentos tokenizados.

    Returns:
        Indice de palabras, frecuencias y contador.
    """
    cnt = Counter(t for d in docs for t in d)
    vocab = {w: i for i, (w, c) in enumerate(cnt.most_common()) if c >= MIN_COUNT}
    total = sum(cnt.values())
    frec = np.zeros(len(vocab), dtype=np.float64)
    for w, i in vocab.items():
        frec[i] = cnt[w] / total
    # Probabilidad de conservar, con el muestreo clasico de word2vec.
    p = (np.sqrt(frec / SUBMUESTREO) + 1.0) * (SUBMUESTREO / frec)
    p = np.clip(p, 0.0, 1.0)
    return vocab, p, cnt


def pares_de_epoca(docs: list[list[str]], vocab: dict[str, int],
                   p: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Genera los pares centro-contexto de una epoca, con ventana dinamica.

    Args:
        docs: documentos tokenizados.
        vocab: indice de palabras.
        p: probabilidad de conservar cada palabra.
        rng: generador aleatorio.

    Returns:
        Dos arrays con centros y contextos.
    """
    centros, contextos = [], []
    for d in docs:
        ids = [vocab[t] for t in d if t in vocab]
        if len(ids) < 2:
            continue
        ids = [i for i in ids if rng.random() < p[i]]
        for k, c in enumerate(ids):
            ventana = int(rng.integers(1, VENTANA + 1))
            lo = max(0, k - ventana)
            hi = min(len(ids), k + ventana + 1)
            for j in range(lo, hi):
                if j != k:
                    centros.append(c)
                    contextos.append(ids[j])
    return np.asarray(centros, dtype=np.int64), np.asarray(contextos, dtype=np.int64)


class SkipGram(nn.Module):
    """Skip-gram con negative sampling, implementado desde cero."""

    def __init__(self, n_vocab: int, dim: int) -> None:
        """Inicializa las dos matrices de embedding.

        Args:
            n_vocab: tamano del vocabulario.
            dim: dimension.
        """
        super().__init__()
        self.entrada = nn.Embedding(n_vocab, dim)
        self.salida = nn.Embedding(n_vocab, dim)
        lim = 0.5 / dim
        nn.init.uniform_(self.entrada.weight, -lim, lim)
        nn.init.zeros_(self.salida.weight)

    def forward(self, centro: torch.Tensor, ctx: torch.Tensor,
                neg: torch.Tensor) -> torch.Tensor:
        """Calcula la perdida de un lote.

        Args:
            centro: indices centrales.
            ctx: indices de contexto positivos.
            neg: indices negativos, de forma (lote, negativos).

        Returns:
            La perdida media.
        """
        v = self.entrada(centro)
        u_pos = self.salida(ctx)
        u_neg = self.salida(neg)
        pos = torch.nn.functional.logsigmoid((v * u_pos).sum(dim=1)).mean()
        neg_l = torch.nn.functional.logsigmoid(-(v.unsqueeze(1) * u_neg).sum(dim=2)).mean()
        return -(pos + neg_l)


def entrenar(ruta: Path, etiqueta: str) -> dict:
    """Entrena un skip-gram sobre un corpus y devuelve su registro.

    Args:
        ruta: fichero del corpus.
        etiqueta: nombre del corpus.

    Returns:
        Diccionario con el modelo y las metricas.
    """
    dev = sembrar(SEMILLA)
    rng = np.random.default_rng(SEMILLA)

    t0 = time.time()
    docs = leer_tokens(ruta)
    n_tokens = sum(len(d) for d in docs)
    log(f"   {etiqueta}: {len(docs):,} documentos | {n_tokens:,} tokens")
    vocab, p, cnt = construir_vocabulario(docs)
    log(f"      vocabulario con min_count={MIN_COUNT}: {len(vocab):,} palabras")
    cobertura = sum(c for w, c in cnt.items() if w in vocab) / max(sum(cnt.values()), 1)
    log(f"      cobertura de tokens: {100 * cobertura:.2f} por ciento")

    modelo = SkipGram(len(vocab), DIM).to(dev)
    optimo = torch.optim.Adam(modelo.parameters(), lr=LR)
    # Tabla de negativos: ruido unigram elevado a 0.75
    idx = np.array([vocab[w] for w in vocab], dtype=np.int64)
    pesos = np.array([cnt[w] for w in vocab], dtype=np.float64) ** 0.75
    pesos = pesos / pesos.sum()

    perdidas = []
    for ep in range(1, EPOCAS + 1):
        te = time.time()
        centros, contextos = pares_de_epoca(docs, vocab, p, rng)
        orden = rng.permutation(len(centros))
        centros, contextos = centros[orden], contextos[orden]
        suma, lotes = 0.0, 0
        for i in range(0, len(centros), LOTE):
            c = torch.from_numpy(centros[i:i + LOTE]).to(dev)
            x = torch.from_numpy(contextos[i:i + LOTE]).to(dev)
            neg = torch.from_numpy(
                rng.choice(idx, size=(len(c), NEGATIVOS), p=pesos)).to(dev)
            optimo.zero_grad()
            perdida = modelo(c, x, neg)
            perdida.backward()
            optimo.step()
            suma += float(perdida.item())
            lotes += 1
        media = suma / max(lotes, 1)
        perdidas.append(round(media, 6))
        log(f"      epoca {ep}/{EPOCAS} | loss {media:.6f} | pares {len(centros):,} | "
            f"{time.time() - te:.0f}s")

    W = modelo.entrada.weight.detach().cpu().numpy().astype(np.float32)
    MODELOS.mkdir(parents=True, exist_ok=True)
    salida = MODELOS / f"w2v_{etiqueta}.npz"
    np.savez_compressed(salida, W=W, palabras=np.array(list(vocab.keys()), dtype=object))
    return {"etiqueta": etiqueta, "corpus": str(ruta.relative_to(PROJECT_ROOT)),
            "documentos": len(docs), "tokens": n_tokens, "vocabulario": len(vocab),
            "cobertura_tokens": round(cobertura, 6),
            "loss_por_epoca": perdidas, "segundos": round(time.time() - t0, 1),
            "modelo": str(salida.relative_to(PROJECT_ROOT)),
            "sha256_modelo": hashlib.sha256(salida.read_bytes()).hexdigest(),
            "vocab": vocab, "W": W}


def main() -> None:
    """Entrena los dos Word2Vec con la misma configuracion.

    Returns:
        None.
    """
    inicio = datetime.now(timezone.utc)
    log("P71-04 P1b: Word2Vec desde cero sobre los dos corpus")
    hash_real = hashlib.sha256(V03_SUB.read_bytes()).hexdigest()
    if hash_real != V03_SHA:
        raise SystemExit("PARADA: la submission de V03 no coincide con el sha256 congelado")
    log(f"   config: dim {DIM} | ventana {VENTANA} | min_count {MIN_COUNT} | "
        f"negativos {NEGATIVOS} | muestreo {SUBMUESTREO} | epocas {EPOCAS} | sg 1 | "
        f"semilla {SEMILLA}")
    log(f"   device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")

    tw = entrenar(CORPORA / "corpus_twitter_tokens.txt", "twitter")
    ge = entrenar(CORPORA / "corpus_general_tokens.txt", "general")

    # OOV contra el vocabulario de TOXIC: cuanta cobertura tiene cada uno
    from src.data import read_train
    tr = read_train()
    toks_toxic = [RE_TOKEN.findall(str(t)) for t in tr["text"]]
    oov = {}
    for etiqueta, m in (("twitter", tw), ("general", ge)):
        dentro = sum(1 for d in toks_toxic for t in d if t in m["vocab"])
        total = sum(len(d) for d in toks_toxic)
        oov[etiqueta] = {"cobertura_toxic": round(dentro / max(total, 1), 6),
                         "oov": round(1 - dentro / max(total, 1), 6),
                         "tipos_cubiertos": len({t for d in toks_toxic for t in d
                                                 if t in m["vocab"]})}
        log(f"   {etiqueta}: cobertura sobre el vocabulario de TOXIC "
            f"{100 * oov[etiqueta]['cobertura_toxic']:.2f} por ciento")

    salida = {
        "familia": "P71_TWITTER_WORD2VEC", "experiment_id": "P71-04", "fase": "P1b entrenamiento",
        "fecha": inicio.isoformat(timespec="seconds"),
        "implementacion": {
            "que": "skip-gram con negative sampling, implementado desde cero en PyTorch",
            "por_que_no_gensim": ("gensim no compila en Python 3.14: su Cython genera C contra "
                                  "PyDictObject.ma_version_tag, eliminado en esa version. Se "
                                  "intento y fallo"),
            "respaldo_documentado": ("la spec de v5 preveia 'PyTorch solo como respaldo/control'"),
            "device": (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"),
        },
        "configuracion": {"vector_size": DIM, "window": VENTANA, "min_count": MIN_COUNT,
                          "negative": NEGATIVOS, "sample": SUBMUESTREO, "epochs": EPOCAS,
                          "sg": 1, "seed": SEMILLA, "lote": LOTE, "lr": LR},
        "corpus_twitter": {k: v for k, v in tw.items() if k not in ("vocab", "W")},
        "corpus_general": {k: v for k, v in ge.items() if k not in ("vocab", "W")},
        "cobertura_sobre_toxic": oov,
        "reproducibilidad": {"nivel": "R2", "hash_v03": hash_real, "corrected_cv_v03": V03_CCV},
        "formulacion": "V03 remains the best verified leaderboard result.",
    }
    (REPORTS / "twitter_word2vec_training.json").write_text(
        json.dumps(salida, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # Guardar los vocabularios para el downstream sin re-parsear.
    np.savez_compressed(MODELOS / "vocabularios.npz",
                        tw_palabras=np.array(list(tw["vocab"].keys()), dtype=object),
                        ge_palabras=np.array(list(ge["vocab"].keys()), dtype=object))

    log("")
    log("=" * 78)
    log(f"P71-04 OK | twitter {tw['vocabulario']:,} palabras | "
        f"general {ge['vocabulario']:,} palabras")
    log(f"   cobertura TOXIC: twitter {100 * oov['twitter']['cobertura_toxic']:.2f}% | "
        f"general {100 * oov['general']['cobertura_toxic']:.2f}%")
    log(f"   tiempo: twitter {tw['segundos']:.0f}s | general {ge['segundos']:.0f}s")
    log("=" * 78)


if __name__ == "__main__":
    ast.parse(Path(__file__).read_text(encoding="utf-8"))
    main()
