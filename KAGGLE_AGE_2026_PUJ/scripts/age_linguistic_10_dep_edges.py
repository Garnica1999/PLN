"""P5.0 - derivacion determinista de las aristas de dependencia (dep -> POS de la cabeza).

## Que hace y por que

La representacion de P5 (patrones de dependencia tipo `subj->VERB`, `DET->NOUN`) necesita la
categoria de la CABEZA de cada token, que no se almaceno en la anotacion original (solo se
guardo la etiqueta de dependencia por token). Se re-analiza train con el MISMO modelo congelado
(spaCy md/lg, SHA256 registrados; D-076: solo etiquetas) y se construye un artefacto derivado:

```text
token_i = "ROOT->{POS_i}"                si dep_i == ROOT
token_i = "{dep_i}->{POS_cabeza_i}"     en otro caso
```

Validacion obligatoria contra lo almacenado: la etiqueta de dependencia re-derivada debe
coincidir token a token con `dep_seq` del parquet de anotacion, y el numero de tokens por
documento debe ser identico. Si algo no cuadra, el script para.

Salidas: features/linguistic/dep_edges_{md,lg}_train.parquet (hashes en
experiments/age_linguistic_10_dep_edges.json). No usa test, no usa holdout, no entrena nada.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config as cfg  # noqa: E402
from src import data as datos  # noqa: E402
from src.verificacion_numerica import autotest_numerico  # noqa: E402

FECHA = "2026-09-30"
MODELOS = {"md": "es_core_news_md", "lg": "es_core_news_lg"}
ANOTACION = {"md": ROOT / "features" / "linguistic" / "anotacion_md_train.parquet",
             "lg": ROOT / "features" / "linguistic" / "anotacion_lg_train.parquet"}
SALIDA = {clave: ROOT / "features" / "linguistic" / f"dep_edges_{clave}_train.parquet"
          for clave in MODELOS}
RESULTADO = cfg.EXPERIMENTS / "age_linguistic_10_dep_edges.json"


def sha256(ruta: Path) -> str:
    return hashlib.sha256(ruta.read_bytes()).hexdigest()


def deriva(clave: str, textos: list[str]) -> tuple[np.ndarray, list[int]]:
    import spacy
    nlp = spacy.load(MODELOS[clave])
    secuencias = np.empty(len(textos), dtype=object)
    n_tokens = [0] * len(textos)
    inicio = time.time()
    for i, doc in enumerate(nlp.pipe(textos, batch_size=512, n_process=4), 1):
        tokens = []
        for t in doc:
            pos = t.pos_ if t.pos_ else "X"
            if t.dep_ == "ROOT" or not t.dep_:
                tokens.append(f"ROOT->{pos}")
            else:
                tokens.append(f"{t.dep_}->{t.head.pos_ or 'X'}")
        secuencias[i - 1] = " ".join(tokens)
        n_tokens[i - 1] = len(tokens)
        if i % 50000 == 0:
            print(f"    [{clave}] {i}/{len(textos)} docs ({time.time() - inicio:.0f} s)",
                  flush=True)
    return secuencias, n_tokens


def valida(clave: str, secuencias: np.ndarray, n_tokens: list[int]) -> dict:
    """La dependencia re-derivada debe reproducir dep_seq y n_tokens almacenados."""
    almacen = pd.read_parquet(ANOTACION[clave], columns=["dep_seq", "n_tokens"])
    dep_guardado = almacen["dep_seq"].astype(str).to_numpy()
    n_guardado = almacen["n_tokens"].astype(int).to_numpy()
    distintos_tokens = int(np.sum(n_guardado != np.asarray(n_tokens)))
    distintos_dep = 0
    for guardado, derivado in zip(dep_guardado, secuencias):
        dep_real = [tok.split("->")[0] for tok in derivado.split()]
        if guardado.split() != dep_real:
            distintos_dep += 1
    return {"documentos_con_tokens_distintos": distintos_tokens,
            "documentos_con_dep_distinta": distintos_dep}


def main() -> int:
    autotest_numerico()
    inicio = time.time()
    train = datos.load_train()
    textos = train[cfg.TEXT_COL].fillna("").astype(str).tolist()
    print(f"train: {len(textos):,} textos", flush=True)

    fichas = {}
    for clave in MODELOS:
        print(f"=== {clave}: re-analisis determinista ===", flush=True)
        secuencias, n_tokens = deriva(clave, textos)
        validacion = valida(clave, secuencias, n_tokens)
        print(f"  validacion: {validacion}", flush=True)
        if validacion["documentos_con_tokens_distintos"] or validacion["documentos_con_dep_distinta"]:
            print("[STOP] la re-derivacion no reproduce lo almacenado", flush=True)
            return 1
        frame = pd.DataFrame({"fila": range(len(textos)), "dep_edge_seq": secuencias,
                              "n_tokens": n_tokens})
        frame.to_parquet(SALIDA[clave], index=False, compression="zstd")
        sha = sha256(SALIDA[clave])
        vacios = int(sum(1 for s in secuencias if not s.strip()))
        tipos = len({tok for s in secuencias for tok in s.split()})
        fichas[clave] = {"modelo": MODELOS[clave], "filas": len(textos),
                         "parquet": str(SALIDA[clave].relative_to(ROOT)), "sha256": sha,
                         "documentos_vacios": vacios, "vocabulario_aristas": tipos,
                         "validacion": validacion}
        print(f"  {clave}: sha {sha[:16]}... | aristas distintas {tipos} | vacios {vacios}",
              flush=True)

    RESULTADO.write_text(json.dumps({
        "fase": "P5.0 derivacion de aristas de dependencia", "fecha": FECHA,
        "fichas": fichas, "sin_test": True, "sin_holdout": True, "sin_entrenamiento": True,
        "segundos": round(time.time() - inicio, 1),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[fin] {time.time() - inicio:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
