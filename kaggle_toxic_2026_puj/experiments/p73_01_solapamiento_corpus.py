"""Mide el solapamiento por near-duplicados entre el corpus externo y desarrollo.

P0 midio solapamiento EXACTO y NORMALIZADO: cero. El revisor de compliance pidio ademas
NODO-DUPLICADOS: si el corpus externo contuviera tuits de desarrollo (o casi), el Word2Vec habria
visto texto de la competencia, y la autorizacion del profesor exige que no sean los mismos datos.

Metodo: shingles de 3 palabras. Indice invertido de los shingles de desarrollo; una sola pasada
sobre el corpus externo; Jaccard exacto sobre los shingles de los documentos candidatos.

No toca test, no toca holdout, no escribe submission.
"""

from __future__ import annotations

import csv as _csv
import hashlib
import json
import re
import sys
import time
import zlib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import read_train
from src.splits import grupos_fusionados, holdout_v2

REPORTS = PROJECT_ROOT / "reports"
CORPUS = PROJECT_ROOT / "data" / "external" / "corpora" / "twitter_domain" / "corpus_twitter_tokens.txt"
RE_TOKEN = re.compile(r"[#@]?\w+(?:['\u2019]\w+)*", re.UNICODE)
K_SHINGLE = 3
UMBRAL_PODA = 3


def log(m: str) -> None:
    """Imprime una linea de progreso.

    Args:
        m: texto a mostrar.
    """
    print(f"[P73-01] {m}", flush=True)


def shingles(tokens: list[str]) -> set[int]:
    """Conjunto de shingles de k palabras, hasheados.

    Args:
        tokens: lista de tokens.

    Returns:
        Conjunto de enteros.
    """
    if len(tokens) < K_SHINGLE:
        return {zlib.crc32(t.encode("utf-8")) for t in tokens}
    return {zlib.crc32(" ".join(tokens[i:i + K_SHINGLE]).encode("utf-8"))
            for i in range(len(tokens) - K_SHINGLE + 1)}


def main() -> None:
    """Ejecuta la medicion de solapamiento en dos pasadas.

    Pasada 1: cuenta shingles compartidos por par (fila, documento) y poda.
    Pasada 2: relee el corpus UNA vez y calcula el Jaccard exacto de los supervivientes.

    Returns:
        None.
    """
    inicio = datetime.now(timezone.utc)
    log("P73-01 solapamiento por near-duplicados: corpus externo contra desarrollo")
    if not CORPUS.exists():
        raise SystemExit(f"PARADA: no existe el corpus {CORPUS}")
    sha_corpus = hashlib.sha256(CORPUS.read_bytes()).hexdigest()

    tr = read_train()
    g = grupos_fusionados(tr)
    idev, _ = holdout_v2(tr, g)
    des = tr.iloc[idev].reset_index(drop=True)
    dev_tokens = [RE_TOKEN.findall(str(x)) for x in des["text"]]
    dev_sh = [shingles(t) for t in dev_tokens]
    indice: dict[int, list[int]] = {}
    for i, s in enumerate(dev_sh):
        for h in s:
            indice.setdefault(h, []).append(i)
    log(f"   {len(dev_tokens)} filas de desarrollo | {len(indice):,} shingles distintos")

    # ---------- pasada 1 ----------
    t0 = time.time()
    pares: dict[int, dict[int, int]] = {}
    n_docs = 0
    with CORPUS.open(encoding="utf-8", errors="replace") as fh:
        for n_docs, linea in enumerate(fh, start=1):
            toks = linea.split()
            if not toks:
                continue
            for h in shingles(toks):
                emp = indice.get(h)
                if emp is None:
                    continue
                for i in emp:
                    d = pares.setdefault(i, {})
                    d[n_docs] = d.get(n_docs, 0) + 1
            if n_docs % 50000 == 0:
                for i in list(pares):
                    pares[i] = {k: v for k, v in pares[i].items() if v >= UMBRAL_PODA}
                    if not pares[i]:
                        del pares[i]
                log(f"      {n_docs:,} documentos | {len(pares)} filas con candidatos vivos "
                    f"| {time.time() - t0:.0f}s")
    for i in list(pares):
        pares[i] = {k: v for k, v in pares[i].items() if v >= UMBRAL_PODA}
        if not pares[i]:
            del pares[i]
    log(f"   pasada 1: {n_docs:,} documentos en {time.time() - t0:.0f}s")
    log(f"   filas con supervivientes tras podar en {UMBRAL_PODA}: {len(pares)}")
    necesarios = {k for d in pares.values() for k in d}
    log(f"   documentos del corpus a comparar exactamente: {len(necesarios):,}")

    # ---------- pasada 2 ----------
    t0 = time.time()
    corpus_sh: dict[int, set[int]] = {}
    with CORPUS.open(encoding="utf-8", errors="replace") as fh:
        for n_docs, linea in enumerate(fh, start=1):
            if n_docs in necesarios:
                corpus_sh[n_docs] = shingles(linea.split())
    log(f"   pasada 2: shingles de {len(corpus_sh):,} documentos en {time.time() - t0:.0f}s")

    filas = []
    for i, docs in sorted(pares.items()):
        si = dev_sh[i]
        mejor, mejor_doc = 0.0, -1
        for d in sorted(docs):
            sj = corpus_sh.get(d)
            if not sj:
                continue
            j = len(si & sj) / max(len(si | sj), 1)
            if j > mejor:
                mejor, mejor_doc = j, d
        filas.append({"fila_desarrollo": i, "jaccard_max": round(mejor, 6),
                      "documento_corpus": mejor_doc, "n_candidatos": len(docs)})

    arr = np.array([f["jaccard_max"] for f in filas]) if filas else np.zeros(0)
    resumen = {
        "n_filas_con_candidato": len(filas),
        "jaccard_max_global": round(float(arr.max()), 6) if arr.size else 0.0,
        "n_jaccard_ge_0_8": int((arr >= 0.8).sum()) if arr.size else 0,
        "n_jaccard_ge_0_5": int((arr >= 0.5).sum()) if arr.size else 0,
        "n_jaccard_ge_0_2": int((arr >= 0.2).sum()) if arr.size else 0,
        "n_jaccard_ge_0_1": int((arr >= 0.1).sum()) if arr.size else 0,
        "umbral_de_poda_en_shingles": UMBRAL_PODA}
    log("")
    log(f"   RESULTADO: filas con algun shingle en comun {resumen['n_filas_con_candidato']}")
    log(f"      jaccard maximo global {resumen['jaccard_max_global']}")
    for u in (0.8, 0.5, 0.2, 0.1):
        clave = f"n_jaccard_ge_{str(u).replace('.', '_')}"
        log(f"      filas con jaccard >= {u}: {resumen[clave]}")
    veredicto = ("SIN_SOLAPAMIENTO" if resumen["n_jaccard_ge_0_5"] == 0
                 else "SOLAPAMIENTO_A_REVISAR")
    log(f"   VEREDICTO: {veredicto}")

    salida = {
        "familia": "P73_TEXTCNN_WORD2VEC_RANDOM", "experiment_id": "P73-01",
        "fase": "diagnostico de solapamiento corpus externo contra desarrollo",
        "fecha": inicio.isoformat(timespec="seconds"),
        "naturaleza": "DIAGNOSTICO. No entrena nada, no evalua ningun modelo",
        "metodo": {"shingle": f"{K_SHINGLE} palabras", "hash": "crc32",
                   "indice": "invertido de desarrollo, dos pasadas sobre el corpus, poda",
                   "similitud": "Jaccard exacto sobre shingles"},
        "corpus": {"ruta": str(CORPUS.relative_to(PROJECT_ROOT)),
                   "sha256": sha_corpus, "documentos": n_docs,
                   "licencia": "CC BY-SA 4.0",
                   "origen": "mmaguero/covid19-spanishtweets-earlylateapril2020",
                   "periodo": "29-mar a 1-may-2020"},
        "filas_desarrollo": len(dev_tokens),
        "resumen": resumen,
        "por_fila": filas,
        "veredicto": veredicto,
        "que_cierra": ("si no hay solapamiento >= 0.5, el corpus externo no contiene tuits de "
                       "desarrollo ni near-duplicados de ellos, y el Word2Vec no pudo ver texto de "
                       "la competencia"),
        "test_tocado": False, "holdout_tocado": False,
        "formulacion": "V03 remains the best verified leaderboard result.",
    }
    (REPORTS / "p73_solapamiento_corpus.json").write_text(
        json.dumps(salida, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (REPORTS / "p73_solapamiento_corpus.csv").open("w", encoding="ascii",
                                                       newline="") as fh:
        w = _csv.writer(fh, lineterminator="\n")
        w.writerow(["fila_desarrollo", "jaccard_max", "documento_corpus", "n_candidatos"])
        for f in filas:
            w.writerow([f["fila_desarrollo"], f["jaccard_max"], f["documento_corpus"],
                        f["n_candidatos"]])
    sucios = [n for n in ("p73_solapamiento_corpus.json", "p73_solapamiento_corpus.csv")
              if any(b > 127 for b in (REPORTS / n).read_bytes())]
    if sucios:
        raise SystemExit(f"PARADA: documentos con bytes no ASCII: {sucios}")
    log("")
    log("=" * 78)
    log(f"P73-01 OK | {veredicto} | jaccard maximo {resumen['jaccard_max_global']}")
    log("=" * 78)


if __name__ == "__main__":
    main()
