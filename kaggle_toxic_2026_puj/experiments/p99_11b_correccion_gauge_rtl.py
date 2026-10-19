#!/usr/bin/env python
"""
P99-11b: correccion del gauge de validacion RTL epoca 1 (S99-01).

El gauge original (ppl 2408.3) era invalido: se evaluo el modelo RTL sobre
streams de validacion en orden LTR. Este script:
  1. reconstruye los streams de validacion por direccion (como en p99_11),
  2. mide el checkpoint preservado models/p99_s99_01_rtl_ep1.pt con el stream
     RTL correcto (agregado y por dominio),
  3. verifica los controles: rtl.pt final y ltr.pt final reproducen sus ppl,
  4. parchea reports/p99_pretraining.json (agrega campos corregidos sin borrar
     los originales), regenera p99_pretraining_results.csv y anexa la errata
     al informe md.

Uso:
    /home/carlos/venv/bin/python experiments/p99_11b_correccion_gauge_rtl.py
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

RAIZ = Path(__file__).resolve().parents[1]


def _cargar(ruta: Path, nombre: str):
    """Carga un script del proyecto como modulo."""
    spec = importlib.util.spec_from_file_location(nombre, ruta)
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


m = _cargar(RAIZ / "experiments" / "p99_11_preentrenar_s99_01.py", "p99_11")

SAL_JSON = RAIZ / "reports" / "p99_pretraining.json"
SAL_CSV = RAIZ / "reports" / "p99_pretraining_results.csv"
SAL_MD = RAIZ / "reports" / "p99_pretraining.md"


def sha256(ruta: Path) -> str:
    """sha256 completo."""
    x = hashlib.sha256()
    with open(ruta, "rb") as fh:
        for bloque in iter(lambda: fh.read(1 << 20), b""):
            x.update(bloque)
    return x.hexdigest()


def evaluar_checkpoint(ruta: Path, streams, dispositivo) -> dict:
    """Evalua un checkpoint sobre streams dados (agregado)."""
    ck = torch.load(ruta, map_location=dispositivo, weights_only=True)
    modelo = m.LMReal(ck["cfg"]).to(dispositivo)
    modelo.load_state_dict(ck["state_dict"])
    modelo.eval()
    vl = m.evaluar(modelo, streams, nn.CrossEntropyLoss(), dispositivo)
    del modelo
    torch.cuda.empty_cache()
    return {"val_loss": vl, "perplexity": float(np.exp(vl))}


def main() -> None:
    """Ejecuta la correccion completa."""
    dispositivo = torch.device("cuda")
    vocab = json.loads(m.VOCAB.read_text(encoding="utf-8"))
    docs = m.cargar_docs()
    docs = [d for d in docs if len(d.split()) > 0]

    por_fuente = m.m12.hashes_por_fuente(m.m11)
    rng = np.random.default_rng(m.SEMILLA + 7)
    claves_val_por_fuente = {}
    for nombre, claves in por_fuente.items():
        n_val = max(1, int(round(len(claves) * 0.02)))
        idx = rng.permutation(len(claves))[:n_val]
        claves_val_por_fuente[nombre] = {claves[i] for i in idx.tolist()}
    claves_val = set().union(*claves_val_por_fuente.values())

    def clave(doc: str) -> str:
        return hashlib.sha1(m.m11.normalizar(doc).encode("utf-8")).hexdigest()

    train_docs, val_docs = [], []
    for doc in docs:
        (val_docs if clave(doc) in claves_val else train_docs).append(doc)

    tokens_todos = set()
    for doc in train_docs:
        tokens_todos.update(doc.split())
    for doc in val_docs:
        tokens_todos.update(doc.split())
    tokens_todos.add("<eos>")
    uid, tabla = m.tabla_caracteres(vocab, tokens_todos)
    wid = {tok: i + 1 for i, tok in enumerate(vocab["vocab_top"])}
    print("tabla reconstruida; evaluando...", flush=True)

    v_f, w_f, v_b, w_b = m.construir_streams(val_docs, uid, tabla, wid)
    streams_rtl = [(v_b, w_b)]
    streams_ltr = [(v_f, w_f)]
    rtl_ep1 = evaluar_checkpoint(RAIZ / "models" / "p99_s99_01_rtl_ep1.pt",
                                 streams_rtl, dispositivo)
    ctrl_rtl = evaluar_checkpoint(RAIZ / "models" / "p99_s99_01_rtl.pt",
                                  streams_rtl, dispositivo)
    ctrl_ltr = evaluar_checkpoint(RAIZ / "models" / "p99_s99_01_ltr.pt",
                                  streams_ltr, dispositivo)
    dominios = {}
    for nombre, claves_f in claves_val_por_fuente.items():
        sub = [d for d in val_docs if clave(d) in claves_f]
        _, _, b_i, b_c = m.construir_streams(sub, uid, tabla, wid)
        dominios[nombre] = evaluar_checkpoint(
            RAIZ / "models" / "p99_s99_01_rtl_ep1.pt", [(b_i, b_c)], dispositivo)
    print(f"rtl ep1 corregido: ppl {rtl_ep1['perplexity']:.1f} "
          f"(original invalido 2408.3)", flush=True)
    print(f"controles: rtl final {ctrl_rtl['perplexity']:.1f} (esperado ~98.3) | "
          f"ltr final {ctrl_ltr['perplexity']:.1f} (esperado ~100.6)", flush=True)
    assert abs(ctrl_rtl["perplexity"] - 98.3) < 1.5
    assert abs(ctrl_ltr["perplexity"] - 100.6) < 1.5

    d = json.loads(SAL_JSON.read_text(encoding="utf-8"))
    d["correccion_gauge_rtl_ep1"] = {
        "fecha": "2026-10-13",
        "motivo": "el gauge original evaluo el modelo RTL sobre streams de "
                  "validacion en orden LTR (ppl 2408.3 no fisica; la loss de "
                  "entrenamiento era normal)",
        "valor_original_invalido": 2408.3,
        "valor_corregido": round(rtl_ep1["perplexity"], 3),
        "por_dominio_corregido": {k: round(v["perplexity"], 3)
                                  for k, v in dominios.items()},
        "checkpoint_evaluado": "models/p99_s99_01_rtl_ep1.pt",
        "checkpoint_sha256": sha256(RAIZ / "models" / "p99_s99_01_rtl_ep1.pt"),
        "controles": {"rtl_final": round(ctrl_rtl["perplexity"], 3),
                      "ltr_final": round(ctrl_ltr["perplexity"], 3)},
    }
    hist_rtl = d["direcciones"]["rtl"]["historial"][0]
    hist_rtl["nota_gauge"] = ("valor original INVALIDO (orden de validacion "
                              "incorrecto); corregido con el checkpoint "
                              "preservado")
    hist_rtl["val_perplexity_corregida"] = round(rtl_ep1["perplexity"], 3)
    hist_rtl["val_por_dominio_corregido"] = {
        k: {"val_loss": v["val_loss"], "perplexity": v["perplexity"]}
        for k, v in dominios.items()}
    SAL_JSON.write_text(json.dumps(d, indent=2, ensure_ascii=True) + "\n",
                        encoding="utf-8")

    filas = ["direccion,epoca,tokens,pasos,val_loss,val_perplexity,"
             "ppl_wikipedia,ppl_mc4,ppl_twitter,segundos,tok_s,checkpoint_sha256"]
    for dir_ in ("ltr", "rtl"):
        sha = d["checkpoints"][dir_]["sha256"]
        for h in d["direcciones"][dir_]["historial"]:
            ppl = h.get("val_perplexity_corregida", h["val_perplexity"])
            pd_ = h.get("val_por_dominio_corregido") or h.get("val_por_dominio")
            dom = (f"{pd_['wikipedia']['perplexity']:.3f},"
                   f"{pd_['mc4']['perplexity']:.3f},"
                   f"{pd_['twitter']['perplexity']:.3f}" if pd_ else "n/a")
            filas.append(f"{dir_},{h['epoca']},{h['tokens']},{h['pasos']},"
                         f"{float(np.log(ppl)):.6f},{ppl:.3f},{dom},"
                         f"{h['segundos']},{h['tokens_por_segundo']},{sha}")
    SAL_CSV.write_text("\n".join(filas) + "\n", encoding="utf-8")

    with open(SAL_MD, "a", encoding="utf-8") as fh:
        fh.write("\n## Errata del gauge RTL (epoca 1)\n\n"
                 f"El valor original de la epoca 1 del RTL (ppl 2408.3) era "
                 f"invalido (evaluado en orden LTR). Corregido con el "
                 f"checkpoint preservado y el stream RTL correcto: "
                 f"**ppl {rtl_ep1['perplexity']:.1f}** "
                 f"(wiki {dominios['wikipedia']['perplexity']:.1f} / mc4 "
                 f"{dominios['mc4']['perplexity']:.1f} / twitter "
                 f"{dominios['twitter']['perplexity']:.1f}). Controles: rtl "
                 f"final {ctrl_rtl['perplexity']:.1f}, ltr final "
                 f"{ctrl_ltr['perplexity']:.1f}.\n")
    print("json, csv y md corregidos y verificados")


if __name__ == "__main__":
    main()
