#!/usr/bin/env python
"""
P98-12 (Fase B, C98-1): preentrenamiento LTR y RTL sobre el corpus de 30M.

Pre-registro: reports/p98_architecture.md (seccion 4):
  - hidden 1024 x 2 capas (escalera fijada por C98-0 a 17.133 tok/s).
  - seq_len 96, batch 8192 tokens, Adam lr 1e-3 (x0.5 desde epoca 4), clip 5.
  - <=4 epocas con parada temprana si la val loss no mejora.
  - Validacion: 2% de documentos POR FUENTE (replicando la seleccion del
    constructor con la misma semilla); documento = unidad.
  - Dos unidades de entrenamiento (LTR y RTL); checkpoints separados.

Corpus: data/external/corpora/p98_c98_1/tokens_30m.txt (sha 9f562718...).
Sin etiquetas de la competencia; sin test/holdouts.

Salidas: models/p98_c98_1_ltr.pt, models/p98_c98_1_rtl.pt,
features/p98_c98_1_vocab.json, reports/p98_c98_1_pretraining.json y .md,
reports/p98_pretraining_results.csv, reports/p98_c98_1_progreso.json.

Uso:
    /home/carlos/venv/bin/python experiments/p98_12_preentrenar_c98_1.py
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

RAIZ = Path(__file__).resolve().parents[1]


def _cargar_modulo(ruta: Path):
    """Carga un script del proyecto como modulo (sin efectos al importar)."""
    spec = importlib.util.spec_from_file_location(ruta.stem, ruta)
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


LMReal = _cargar_modulo(RAIZ / "experiments" / "p98_10_piloto_c98_0.py").LMReal

CORPUS = RAIZ / "data" / "external" / "corpora" / "p98_c98_1" / "tokens_30m.txt"
RUTA_P98_11 = RAIZ / "experiments" / "p98_11_construir_c98_1.py"
SAL_VOCAB = RAIZ / "features" / "p98_c98_1_vocab.json"
SAL_PROGRESO = RAIZ / "reports" / "p98_c98_1_progreso.json"
SAL_JSON = RAIZ / "reports" / "p98_c98_1_pretraining.json"
SAL_MD = RAIZ / "reports" / "p98_c98_1_pretraining.md"
SAL_CSV = RAIZ / "reports" / "p98_pretraining_results.csv"

SEMILLA = 20260925
TOP_VOCAB = 50_000
MAX_CHAR = 24
EOS = "<eos>"
VAL_FRACCION = 0.02
EPOCAS_MAX = 4

CFG = {
    "vocab_size": TOP_VOCAB + 2,
    "char_vocab_size": 256,
    "char_dim": 32,
    "nn_char": 128,
    "word_dim": 200,
    "enc_dim": 512,
    "hidden": 1024,
    "capas": 2,
    "dropout": 0.2,
    "max_char": MAX_CHAR,
    "seq_len": 96,
    "batch": 8192,
    "lr": 1e-3,
}


def cargar_modulo_p98_11():
    """Carga p98_11 para reutilizar la logica de seleccion por fuente."""
    spec = importlib.util.spec_from_file_location("p98_11", RUTA_P98_11)
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


def sha256(ruta: Path) -> str:
    """sha256 completo."""
    x = hashlib.sha256()
    with open(ruta, "rb") as fh:
        for bloque in iter(lambda: fh.read(1 << 20), b""):
            x.update(bloque)
    return x.hexdigest()


def hashes_por_fuente(m11) -> dict[str, list[str]]:
    """Replica la seleccion del constructor para asignar validacion por fuente."""
    rng = np.random.default_rng(SEMILLA)
    seleccion: dict[str, list[str]] = {}
    for nombre, spec in m11.FUENTES.items():
        docs = m11.leer_documentos(spec["ruta"])
        if spec["modo"] == "prefijo":
            elegidos, total = [], 0
            for doc, n in docs:
                if total >= spec["objetivo"]:
                    break
                elegidos.append(doc)
                total += n
        else:
            orden = rng.permutation(len(docs))
            elegidos, total = [], 0
            for i in orden:
                doc, n = docs[i]
                if total >= spec["objetivo"]:
                    break
                elegidos.append(doc)
                total += n
        seleccion[nombre] = elegidos
    vistos: set[str] = set()
    por_fuente: dict[str, list[str]] = {}
    for nombre, docs in seleccion.items():
        conservados = []
        for doc in docs:
            clave = hashlib.sha1(m11.normalizar(doc).encode("utf-8")).hexdigest()
            if clave in vistos:
                continue
            vistos.add(clave)
            conservados.append(clave)
        por_fuente[nombre] = conservados
    return por_fuente


def val_set(m11, por_fuente: dict[str, list[str]]) -> set[str]:
    """2% de documentos por fuente (semilla propia del split)."""
    rng = np.random.default_rng(SEMILLA + 7)
    val: set[str] = set()
    for nombre, claves in por_fuente.items():
        n_val = max(1, int(round(len(claves) * VAL_FRACCION)))
        idx = rng.permutation(len(claves))[:n_val]
        val.update(claves[i] for i in idx.tolist())
    return val


def construir_arreglos(documentos: list[str], vocab: dict,
                       claves_val: set[str], m11, invertir: bool) -> dict:
    """Streams train/val: ids de palabra (top-50k), chars por token, EOS."""
    word_id = {tok: i + 1 for i, tok in enumerate(vocab["vocab_top"])}
    char_id = {ch: i + 1 for i, ch in enumerate(vocab["chars"])}
    tabla_char: dict[str, np.ndarray] = {}

    def fila_char(tok: str) -> np.ndarray:
        fila = tabla_char.get(tok)
        if fila is None:
            fila = np.zeros(MAX_CHAR, dtype=np.uint8)
            for j, ch in enumerate(tok[:MAX_CHAR]):
                fila[j] = char_id.get(ch, 0)
            tabla_char[tok] = fila
        return fila

    def stream(docs: list[str]):
        ids: list[int] = []
        chs: list[np.ndarray] = []
        for k, doc in enumerate(docs):
            if k:
                ids.append(0)
                chs.append(fila_char(EOS))
            tokens = doc.split()
            if invertir:
                tokens = tokens[::-1]
            for tok in tokens:
                ids.append(word_id.get(tok, 0))
                chs.append(fila_char(tok))
        return (np.array(ids, dtype=np.int32),
                np.stack(chs).astype(np.uint8))

    train_docs, val_docs = [], []
    for doc in documentos:
        clave = hashlib.sha1(m11.normalizar(doc).encode("utf-8")).hexdigest()
        (val_docs if clave in claves_val else train_docs).append(doc)
    ids_tr, chs_tr = stream(train_docs)
    ids_va, chs_va = stream(val_docs)
    return {"ids_train": ids_tr, "chars_train": chs_tr,
            "ids_val": ids_va, "chars_val": chs_va,
            "n_docs_train": len(train_docs), "n_docs_val": len(val_docs)}


def lotes(ids, chars, seq, batch_tokens):
    """Lotes contiguos (palabras, chars, objetivo)."""
    seqs = batch_tokens // seq
    n = (len(ids) - 1) // seq
    por_lote = seqs * seq
    for b in range(n // seqs):
        idx = np.arange(b * por_lote, (b + 1) * por_lote).reshape(seqs, seq)
        yield ids[idx], chars[idx], ids[idx + 1]


def evaluar(modelo, ids, chars, perdida_fn, dispositivo, max_lotes=40) -> float:
    """Loss de validacion (media de hasta max_lotes lotes)."""
    modelo.eval()
    perdidas = []
    with torch.no_grad():
        for palabras, ch, objetivo in lotes(ids, chars, CFG["seq_len"], CFG["batch"]):
            salida = modelo(
                torch.from_numpy(palabras.astype(np.int64)).to(dispositivo),
                torch.from_numpy(ch.astype(np.int64)).to(dispositivo))
            perdidas.append(float(perdida_fn(
                salida.reshape(-1, CFG["vocab_size"]),
                torch.from_numpy(objetivo.astype(np.int64)).reshape(-1)
                .to(dispositivo)
            ).detach().cpu()))
            if len(perdidas) >= max_lotes:
                break
    modelo.train()
    return float(np.mean(perdidas))


def escribir_progreso(datos: dict) -> None:
    """Escribe el progreso de forma atomica (sufijo .tmp + replace)."""
    tmp = SAL_PROGRESO.with_suffix(".tmp")
    tmp.write_text(json.dumps(datos, indent=2, ensure_ascii=True) + "\n",
                   encoding="utf-8")
    tmp.replace(SAL_PROGRESO)


def entrenar_direccion(direccion: str, arreglos: dict, dispositivo) -> dict:
    """Entrena una direccion del LM (LTR o RTL) con parada temprana."""
    torch.manual_seed(SEMILLA)
    np.random.seed(SEMILLA)
    modelo = LMReal(CFG).to(dispositivo)
    opt = torch.optim.Adam(modelo.parameters(), lr=CFG["lr"])
    perdida_fn = nn.CrossEntropyLoss()
    mejor = None
    historial = []
    t_inicio = time.perf_counter()
    for epoca in range(1, EPOCAS_MAX + 1):
        lr_epoca = CFG["lr"] if epoca < 4 else CFG["lr"] * 0.5
        for grupo in opt.param_groups:
            grupo["lr"] = lr_epoca
        modelo.train()
        tokens = 0
        perdidas_muestra = []
        t0 = time.perf_counter()
        for paso, (palabras, ch, objetivo) in enumerate(
                lotes(arreglos["ids_train"], arreglos["chars_train"],
                      CFG["seq_len"], CFG["batch"])):
            salida = modelo(
                torch.from_numpy(palabras.astype(np.int64)).to(dispositivo),
                torch.from_numpy(ch.astype(np.int64)).to(dispositivo))
            perdida = perdida_fn(salida.reshape(-1, CFG["vocab_size"]),
                                 torch.from_numpy(objetivo.astype(np.int64))
                                 .reshape(-1).to(dispositivo))
            opt.zero_grad()
            perdida.backward()
            nn.utils.clip_grad_norm_(modelo.parameters(), 5.0)
            opt.step()
            tokens += palabras.size
            if paso % 250 == 0:
                perdidas_muestra.append(float(perdida.detach().cpu()))
        torch.cuda.synchronize()
        segundos = time.perf_counter() - t0
        val_loss = evaluar(modelo, arreglos["ids_val"], arreglos["chars_val"],
                           perdida_fn, dispositivo)
        fila = {"epoca": epoca, "lr": lr_epoca, "tokens": tokens,
                "loss_muestra_inicio": perdidas_muestra[0],
                "loss_muestra_fin": perdidas_muestra[-1],
                "val_loss": val_loss, "val_perplexity": float(np.exp(val_loss)),
                "segundos": round(segundos, 1),
                "tokens_por_segundo": round(tokens / segundos, 1)}
        historial.append(fila)
        print(f"  [{direccion}] epoca {epoca}: loss {perdidas_muestra[0]:.3f} -> "
              f"{perdidas_muestra[-1]:.3f} | val ppl {np.exp(val_loss):.1f} | "
              f"{tokens/segundos:,.0f} tok/s ({segundos:.0f} s)", flush=True)
        if mejor is None or val_loss < mejor[1]:
            mejor = (epoca, val_loss)
            torch.save({"cfg": CFG, "state_dict": modelo.state_dict(),
                        "epoca": epoca, "val_loss": val_loss,
                        "direccion": direccion,
                        "vocab_file": str(SAL_VOCAB.name)},
                       RAIZ / "models" / f"p98_c98_1_{direccion}.pt")
        escribir_progreso({"direccion_actual": direccion,
                           "historial": historial,
                           "mejor": {"epoca": mejor[0], "val_loss": mejor[1]}})
        if epoca >= 2 and val_loss >= historial[-2]["val_loss"]:
            print(f"  [{direccion}] parada temprana: sin mejora", flush=True)
            break
    return {"historial": historial, "mejor": {"epoca": mejor[0],
                                              "val_loss": mejor[1]},
            "gpu_horas": round((time.perf_counter() - t_inicio) / 3600, 3)}


def main() -> None:
    """Entrena LTR y RTL y escribe los entregables."""
    for ruta in (SAL_VOCAB, SAL_JSON, SAL_MD, SAL_CSV):
        if ruta.exists():
            raise SystemExit(f"Ya existe {ruta}; no se sobrescribe.")
    if not torch.cuda.is_available():
        raise SystemExit("GPU/ROCm no disponible: parar y documentar")
    dispositivo = torch.device("cuda")
    m11 = cargar_modulo_p98_11()
    inicio = time.perf_counter()

    print("cargando corpus C98-1...", flush=True)
    documentos = []
    with open(CORPUS, encoding="utf-8") as fh:
        for linea in fh:
            if linea.split():
                documentos.append(linea.rstrip("\n"))
    print(f"  documentos: {len(documentos)}", flush=True)

    contador: Counter = Counter()
    primera: dict[str, int] = {}
    posicion = 0
    for doc in documentos:
        for tok in doc.split():
            contador[tok] += 1
            if tok not in primera:
                primera[tok] = posicion
            posicion += 1
    ordenados = sorted(contador.items(), key=lambda kv: (-kv[1], primera[kv[0]]))
    vocab_top = [tok for tok, _ in ordenados[:TOP_VOCAB]]
    chars = sorted({ch for tok in contador for ch in tok} | set(EOS))[:254]
    vocabulario = {"vocab_top": vocab_top, "chars": chars, "eos": EOS,
                   "max_char": MAX_CHAR, "seed": SEMILLA, "corpus": "C98-1"}
    SAL_VOCAB.write_text(json.dumps(vocabulario, ensure_ascii=True), encoding="utf-8")
    cubiertos = sum(contador[t] for t in vocab_top)
    print(f"  vocab top-50k: tipos totales {len(contador):,} | cobertura "
          f"{cubiertos/posicion*100:.2f}% | chars {len(chars)}", flush=True)

    por_fuente = hashes_por_fuente(m11)
    claves_val = val_set(m11, por_fuente)
    print(f"  validacion: {len(claves_val)} documentos (2% por fuente)", flush=True)

    resultados = {"experiment_id": "P98-12", "etapa": "C98-1 (preentrenamiento)",
                  "fecha": "2026-10-13", "cfg": CFG,
                  "corpus": {"ruta": str(CORPUS.relative_to(RAIZ)),
                             "sha256": sha256(CORPUS),
                             "documentos": len(documentos),
                             "tokens": posicion},
                  "vocab": {"tipos": len(contador),
                            "cobertura_top50k": round(cubiertos / posicion, 4),
                            "chars": len(chars)},
                  "val_documentos": len(claves_val),
                  "direcciones": {}}
    proba = {d: {} for d in ("ltr", "rtl")}
    for direccion, invertir in (("ltr", False), ("rtl", True)):
        print(f"entrenando {direccion.upper()} (invertir={invertir})...", flush=True)
        arreglos = construir_arreglos(documentos, vocabulario, claves_val, m11,
                                      invertir)
        print(f"  train {arreglos['n_docs_train']} docs / "
              f"{len(arreglos['ids_train']):,} tokens+EOS | val "
              f"{arreglos['n_docs_val']} docs", flush=True)
        r = entrenar_direccion(direccion, arreglos, dispositivo)
        resultados["direcciones"][direccion] = r
        del arreglos
        torch.cuda.empty_cache()

    ltr, rtl = resultados["direcciones"]["ltr"], resultados["direcciones"]["rtl"]
    resultados["gpu_horas_totales_c98_1"] = round(ltr["gpu_horas"] + rtl["gpu_horas"], 3)
    resultados["tiempo_total_s"] = round(time.perf_counter() - inicio, 1)
    resultados["checkpoints"] = {
        "ltr": {"ruta": "models/p98_c98_1_ltr.pt",
                "sha256": sha256(RAIZ / "models" / "p98_c98_1_ltr.pt"),
                "val_ppl": float(np.exp(ltr["mejor"]["val_loss"])),
                "epocas": len(ltr["historial"])},
        "rtl": {"ruta": "models/p98_c98_1_rtl.pt",
                "sha256": sha256(RAIZ / "models" / "p98_c98_1_rtl.pt"),
                "val_ppl": float(np.exp(rtl["mejor"]["val_loss"])),
                "epocas": len(rtl["historial"])}}
    SAL_JSON.write_text(json.dumps(resultados, indent=2, ensure_ascii=True) + "\n",
                        encoding="utf-8")

    lineas_csv = ["direccion,epoca,lr,tokens,val_loss,val_perplexity,segundos,"
                  "tokens_por_segundo,checkpoint_sha256"]
    for direccion in ("ltr", "rtl"):
        sha = resultados["checkpoints"][direccion]["sha256"]
        for h in resultados["direcciones"][direccion]["historial"]:
            lineas_csv.append(f"{direccion},{h['epoca']},{h['lr']},{h['tokens']},"
                              f"{h['val_loss']:.6f},{h['val_perplexity']:.3f},"
                              f"{h['segundos']},{h['tokens_por_segundo']},{sha}")
    SAL_CSV.write_text("\n".join(lineas_csv) + "\n", encoding="utf-8")

    md = [
        "# P98-12 (C98-1): preentrenamiento LTR y RTL sobre 30M tokens",
        "",
        f"- Corpus: {resultados['corpus']['tokens']:,} tokens, "
        f"{resultados['corpus']['documentos']} documentos "
        f"(sha256 {resultados['corpus']['sha256'][:16]}...).",
        f"- Vocab top-50k: cobertura {resultados['vocab']['cobertura_top50k']*100:.2f}%"
        f"; {resultados['vocab']['chars']} chars.",
        f"- Validacion: {resultados['val_documentos']} documentos (2% por fuente).",
        "",
        "| Direccion | Epoca | loss (muestra ini -> fin) | val ppl | tok/s | s |",
        "|---|---|---|---|---|---|",
    ]
    for direccion in ("ltr", "rtl"):
        for h in resultados["direcciones"][direccion]["historial"]:
            md.append(f"| {direccion.upper()} | {h['epoca']} | "
                      f"{h['loss_muestra_inicio']:.3f} -> {h['loss_muestra_fin']:.3f} | "
                      f"{h['val_perplexity']:.1f} | {h['tokens_por_segundo']:,.0f} | "
                      f"{h['segundos']:.0f} |")
    md += [
        "",
        f"- Checkpoints: LTR val ppl "
        f"{resultados['checkpoints']['ltr']['val_ppl']:.1f} "
        f"({resultados['checkpoints']['ltr']['epocas']} epocas); RTL val ppl "
        f"{resultados['checkpoints']['rtl']['val_ppl']:.1f} "
        f"({resultados['checkpoints']['rtl']['epocas']} epocas).",
        f"- GPU-h C98-1: {resultados['gpu_horas_totales_c98_1']} "
        f"(tope 6). La perplexity NO es criterio de adopcion (P98 seccion 10).",
        "",
        "Siguiente: D98-1 (representaciones congeladas + clasificador por pliegue "
        "+ fusion estilo v5.1 con umbrales de V03).",
    ]
    SAL_MD.write_text("\n".join(md) + "\n", encoding="utf-8")
    escribir_progreso({"estado": "completo",
                       "gpu_horas": resultados["gpu_horas_totales_c98_1"]})
    print(f"C98-1 completo: LTR ppl {resultados['checkpoints']['ltr']['val_ppl']:.1f} "
          f"| RTL ppl {resultados['checkpoints']['rtl']['val_ppl']:.1f} | "
          f"GPU-h {resultados['gpu_horas_totales_c98_1']}", flush=True)


if __name__ == "__main__":
    main()
