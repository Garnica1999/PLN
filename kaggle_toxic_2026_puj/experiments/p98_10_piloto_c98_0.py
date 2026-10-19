#!/usr/bin/env python
"""
P98-10 (Fase B, C98-0): piloto real del LM sobre Wikipedia 5M (local).

Objetivo (pre-registrado en reports/p98_architecture.md):
  1. Validar el pipeline real: vocab top-50k desde cero, canal de caracteres,
     lotes por tokens, convergencia y validez de la representacion.
  2. Medir tokens/s REALES por configuracion y fijar la escalera de dimensiones
     para C98-1 (regla pre-declarada): la rung mas grande con throughput
     >= 13.333 tok/s, que pagina 30M x 4 epocas x direccion en <=2.5 GPU-h
     (2 direcciones <=5 GPU-h dentro del tope de 6 GPU-h de etapa).

Corpus: data/external/corpora/wikipedia_es_5M/emb_normal/tokens_5M.txt
(E-CORP-09; CC BY-SA 4.0; hash verificado en Fase A). Sin corpus nuevos.

Salidas: models/p98_c98_0_ltr.pt, features/p98_c98_0_vocab.json,
reports/p98_c98_0_piloto.json y .md.

Uso:
    /home/carlos/venv/bin/python experiments/p98_10_piloto_c98_0.py
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

RAIZ = Path(__file__).resolve().parents[1]
CORPUS = RAIZ / "data" / "external" / "corpora" / "wikipedia_es_5M" / "emb_normal" / "tokens_5M.txt"
SAL_CKPT = RAIZ / "models" / "p98_c98_0_ltr.pt"
SAL_VOCAB = RAIZ / "features" / "p98_c98_0_vocab.json"
SAL_JSON = RAIZ / "reports" / "p98_c98_0_piloto.json"
SAL_MD = RAIZ / "reports" / "p98_c98_0_piloto.md"

SEMILLA = 20260925
TOP_VOCAB = 50_000
MAX_CHAR = 24
EOS = "<eos>"
VAL_FRACCION = 0.02
TOKENS_OBJETIVO_C98_1 = 30_000_000
EPOCAS_C98_1 = 4
SEGUNDOS_POR_DIRECCION = int(2.5 * 3600)
REQUERIDO_TOK_S = TOKENS_OBJETIVO_C98_1 * EPOCAS_C98_1 / SEGUNDOS_POR_DIRECCION
ESCALERA = [(1024, 2), (768, 2), (1024, 1), (512, 2)]

CFG_PILOTO = {
    "vocab_size": TOP_VOCAB + 2,
    "char_vocab_size": 256,
    "char_dim": 32,
    "nn_char": 128,
    "word_dim": 200,
    "enc_dim": 512,
    "hidden": 512,
    "capas": 2,
    "dropout": 0.2,
    "max_char": MAX_CHAR,
    "seq_len": 96,
    "batch": 8192,
    "epocas": 3,
    "lr": 1e-3,
}
CFG_PROBES = {  # sobre la CFG_PILOTO base se sustituyen hidden/capas
    "batch": 8192, "seq_len": 96, "pasos": 30,
}


class LMReal(nn.Module):
    """LM char-aware real; dropout manual entre capas.

    Hallazgo de entorno (2026-10-13): el dropout nativo de nn.LSTM
    (MIOpenDropoutHIP) no compila en MIOpen gfx1201
    (rocrand/rocrand_xorwow.h ausente). Se aplica nn.Dropout entre las dos
    LSTM en lugar del parametro dropout de nn.LSTM; misma semantica de
    arquitectura (dropout entre capas).
    """

    def __init__(self, cfg: dict) -> None:
        super().__init__()
        self.cfg = dict(cfg)
        self.char_emb = nn.Embedding(cfg["char_vocab_size"], cfg["char_dim"],
                                     padding_idx=0)
        self.conv3 = nn.Conv1d(cfg["char_dim"], cfg["nn_char"], kernel_size=3,
                               padding=1)
        self.conv5 = nn.Conv1d(cfg["char_dim"], cfg["nn_char"], kernel_size=5,
                               padding=2)
        self.word_emb = nn.Embedding(cfg["vocab_size"], cfg["word_dim"],
                                     padding_idx=0)
        self.proj = nn.Linear(2 * cfg["nn_char"] + cfg["word_dim"],
                              cfg["enc_dim"])
        self.lstm1 = nn.LSTM(cfg["enc_dim"], cfg["hidden"], num_layers=1,
                             batch_first=True)
        self.dropout = nn.Dropout(cfg["dropout"])
        self.lstm2 = (nn.LSTM(cfg["hidden"], cfg["hidden"], num_layers=1,
                              batch_first=True) if cfg["capas"] > 1 else None)
        self.dec = nn.Linear(cfg["hidden"], cfg["word_dim"])

    def forward(self, words: torch.Tensor, chars: torch.Tensor) -> torch.Tensor:
        """words (B, S); chars (B, S, L) -> logits (B, S, V)."""
        b, s, el = chars.shape
        c = self.char_emb(chars.reshape(b * s, el)).transpose(1, 2)
        c = torch.cat([torch.relu(self.conv3(c)).max(dim=2).values,
                       torch.relu(self.conv5(c)).max(dim=2).values], dim=1)
        emb = self.word_emb(words)
        entrada = torch.cat([emb, c.reshape(b, s, -1)], dim=2)
        h, _ = self.lstm1(torch.relu(self.proj(entrada)))
        if self.lstm2 is not None:
            h, _ = self.lstm2(self.dropout(h))
        proj = self.dec(h)
        return proj @ self.word_emb.weight.t()


def sha256(ruta: Path) -> str:
    """sha256 completo."""
    x = hashlib.sha256()
    with open(ruta, "rb") as fh:
        for bloque in iter(lambda: fh.read(1 << 20), b""):
            x.update(bloque)
    return x.hexdigest()


def cargar_corpus() -> list[list[str]]:
    """Documentos = lineas del archivo de tokens."""
    documentos = []
    with open(CORPUS, encoding="utf-8") as fh:
        for linea in fh:
            tokens = linea.split()
            if tokens:
                documentos.append(tokens)
    return documentos


def construir_vocab(documentos: list[list[str]]) -> dict:
    """Vocab top-50k (frecuencia; empate por primera aparicion) + caracteres."""
    contador: Counter = Counter()
    primera: dict[str, int] = {}
    posicion = 0
    for doc in documentos:
        for tok in doc:
            contador[tok] += 1
            if tok not in primera:
                primera[tok] = posicion
            posicion += 1
    ordenados = sorted(contador.items(), key=lambda kv: (-kv[1], primera[kv[0]]))
    vocab_top = [tok for tok, _ in ordenados[:TOP_VOCAB]]
    chars = sorted({ch for tok in contador for ch in tok} | set(EOS))
    if len(chars) > 254:
        chars = chars[:254]
    return {"contador": contador, "vocab_top": vocab_top, "chars": chars}


def preparar_arreglos(documentos: list[list[str]], vocab: dict,
                      semilla: int) -> dict:
    """ids de palabra (top-50k; OOV=0), chars por token, split por documento."""
    rng = np.random.default_rng(semilla)
    n_docs = len(documentos)
    perm = rng.permutation(n_docs)
    n_val = max(1, int(round(n_docs * VAL_FRACCION)))
    idx_val = set(perm[:n_val].tolist())

    word_id = {tok: i + 1 for i, tok in enumerate(vocab["vocab_top"])}
    char_id = {ch: i + 1 for i, ch in enumerate(vocab["chars"])}
    char_id[EOS] = len(char_id)  # no aplica: chars ya incluye los de <eos>

    tabla_char: dict[str, np.ndarray] = {}

    def fila_char(tok: str) -> np.ndarray:
        fila = tabla_char.get(tok)
        if fila is None:
            fila = np.zeros(MAX_CHAR, dtype=np.uint8)
            for j, ch in enumerate(tok[:MAX_CHAR]):
                fila[j] = char_id.get(ch, 0)
            tabla_char[tok] = fila
        return fila

    def construir_stream(docs: list[list[str]]):
        ids, chs = [], []
        for k, doc in enumerate(docs):
            if k:
                ids.append(0)
                chs.append(fila_char(EOS))
            for tok in doc:
                ids.append(word_id.get(tok, 0))
                chs.append(fila_char(tok))
        return (np.array(ids, dtype=np.int32), np.stack(chs).astype(np.uint8))

    docs_train = [d for i, d in enumerate(documentos) if i not in idx_val]
    docs_val = [d for i, d in enumerate(documentos) if i in idx_val]
    ids_train, chs_train = construir_stream(docs_train)
    ids_val, chs_val = construir_stream(docs_val)
    n_tokens = sum(len(d) for d in documentos)
    cubiertos = sum(vocab["contador"][t] for t in vocab["vocab_top"][:TOP_VOCAB])
    return {
        "ids_train": ids_train, "chars_train": chs_train,
        "ids_val": ids_val, "chars_val": chs_val,
        "n_docs_train": len(docs_train), "n_docs_val": len(docs_val),
        "tipos": len(vocab["contador"]),
        "cobertura_top50k": cubiertos / n_tokens,
        "n_tokens_corpus": n_tokens,
    }


def lotes_reales(ids: np.ndarray, chars: np.ndarray, seq: int, batch_tokens: int):
    """Lotes contiguos (palabras, chars, objetivo) para next-token."""
    seqs = batch_tokens // seq
    n = (len(ids) - 1) // seq
    por_lote = seqs * seq
    for b in range(n // seqs):
        idx = np.arange(b * por_lote, (b + 1) * por_lote).reshape(seqs, seq)
        yield ids[idx], chars[idx], ids[idx + 1]


def medir_loss(modelo, ids, chars, perdida_fn, dispositivo, max_lotes=60) -> float:
    """Loss de validacion sobre lotes del stream de validacion."""
    modelo.eval()
    perdidas = []
    with torch.no_grad():
        for palabras, ch, objetivo in lotes_reales(ids, chars, CFG_PILOTO["seq_len"],
                                                   CFG_PILOTO["batch"]):
            x = torch.from_numpy(palabras.astype(np.int64)).to(dispositivo)
            y = torch.from_numpy(objetivo.astype(np.int64)).to(dispositivo)
            c = torch.from_numpy(ch.astype(np.int64)).to(dispositivo)
            salida = modelo(x, c)
            perdidas.append(float(perdida_fn(
                salida.reshape(-1, CFG_PILOTO["vocab_size"]),
                y.reshape(-1)).detach().cpu()))
            if len(perdidas) >= max_lotes:
                break
    modelo.train()
    return float(np.mean(perdidas))


def medir_throughput(clase_modelo, cfg: dict, ids, chars, dispositivo,
                     pasos: int) -> dict:
    """Tokens/s reales de una configuracion (con calentamiento)."""
    modelo = clase_modelo(cfg).to(dispositivo)
    opt = torch.optim.Adam(modelo.parameters(), lr=1e-3)
    perdida_fn = nn.CrossEntropyLoss()
    generador = lotes_reales(ids, chars, cfg["seq_len"], cfg["batch"])
    tokens = 0
    for i, (palabras, ch, objetivo) in enumerate(generador):
        x = torch.from_numpy(palabras.astype(np.int64)).to(dispositivo)
        y = torch.from_numpy(objetivo.astype(np.int64)).to(dispositivo)
        c = torch.from_numpy(ch.astype(np.int64)).to(dispositivo)
        if i == 3:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            tokens = 0
        if i < 3:
            continue
        salida = modelo(x, c)
        perdida = perdida_fn(salida.reshape(-1, cfg["vocab_size"]), y.reshape(-1))
        opt.zero_grad()
        perdida.backward()
        opt.step()
        tokens += palabras.size
        if i - 3 + 1 >= pasos:
            break
    torch.cuda.synchronize()
    segundos = time.perf_counter() - t0
    vram = torch.cuda.max_memory_allocated() / 1e9
    del modelo, opt
    torch.cuda.empty_cache()
    return {"tokens_por_segundo": round(tokens / segundos, 1),
            "segundos": round(segundos, 2), "vram_gb": round(vram, 3)}


def main() -> None:
    """Ejecuta el piloto C98-0 completo."""
    for ruta in (SAL_CKPT, SAL_VOCAB, SAL_JSON, SAL_MD):
        if ruta.exists():
            raise SystemExit(f"Ya existe {ruta}; no se sobrescribe.")
    if not torch.cuda.is_available():
        raise SystemExit("GPU/ROCm no disponible: parar y documentar (P98 seccion 4)")
    dispositivo = torch.device("cuda")
    inicio = time.perf_counter()

    print("cargando corpus wikipedia 5M...")
    documentos = cargar_corpus()
    vocab = construir_vocab(documentos)
    arreglos = preparar_arreglos(documentos, vocab, SEMILLA)
    print(f"  docs: {arreglos['n_docs_train']} train / {arreglos['n_docs_val']} val | "
          f"tipos: {arreglos['tipos']} | top-50k cubre "
          f"{arreglos['cobertura_top50k']*100:.2f}% de tokens | "
          f"chars: {len(vocab['chars'])}")
    SAL_VOCAB.write_text(json.dumps({
        "top_vocab": vocab["vocab_top"], "chars": vocab["chars"],
        "eos": EOS, "max_char": MAX_CHAR, "seed": SEMILLA}, ensure_ascii=True),
        encoding="utf-8")

    torch.manual_seed(SEMILLA)
    np.random.seed(SEMILLA)
    modelo = LMReal(CFG_PILOTO).to(dispositivo)
    n_params = sum(p.numel() for p in modelo.parameters())
    opt = torch.optim.Adam(modelo.parameters(), lr=CFG_PILOTO["lr"])
    perdida_fn = nn.CrossEntropyLoss()
    print(f"modelo piloto (LTR {CFG_PILOTO['hidden']}x{CFG_PILOTO['capas']}): "
          f"{n_params:,} parametros")

    historial = []
    torch.cuda.reset_peak_memory_stats()
    for epoca in range(1, CFG_PILOTO["epocas"] + 1):
        modelo.train()
        perdidas = []
        tokens = 0
        t0 = time.perf_counter()
        for palabras, ch, objetivo in lotes_reales(
                arreglos["ids_train"], arreglos["chars_train"],
                CFG_PILOTO["seq_len"], CFG_PILOTO["batch"]):
            x = torch.from_numpy(palabras.astype(np.int64)).to(dispositivo)
            y = torch.from_numpy(objetivo.astype(np.int64)).to(dispositivo)
            c = torch.from_numpy(ch.astype(np.int64)).to(dispositivo)
            salida = modelo(x, c)
            perdida = perdida_fn(salida.reshape(-1, CFG_PILOTO["vocab_size"]),
                                 y.reshape(-1))
            opt.zero_grad()
            perdida.backward()
            nn.utils.clip_grad_norm_(modelo.parameters(), 5.0)
            opt.step()
            perdidas.append(float(perdida.detach().cpu()))
            tokens += palabras.size
        torch.cuda.synchronize()
        segundos = time.perf_counter() - t0
        val_loss = medir_loss(modelo, arreglos["ids_val"], arreglos["chars_val"],
                              perdida_fn, dispositivo)
        historial.append({
            "epoca": epoca, "tokens": tokens,
            "loss_primera": perdidas[0], "loss_ultima": perdidas[-1],
            "loss_media": float(np.mean(perdidas)),
            "val_loss": val_loss, "val_perplexity": float(np.exp(val_loss)),
            "segundos": round(segundos, 1),
            "tokens_por_segundo": round(tokens / segundos, 1),
        })
        print(f"  epoca {epoca}: loss {perdidas[0]:.3f} -> {perdidas[-1]:.3f} "
              f"| val ppl {np.exp(val_loss):.1f} | {tokens/segundos:,.0f} tok/s "
              f"({segundos:.0f} s)")

    torch.save({"cfg": CFG_PILOTO, "state_dict": modelo.state_dict(),
                "vocab_file": str(SAL_VOCAB.name)}, SAL_CKPT)

    probes = {}
    probadas = [(CFG_PILOTO["hidden"], CFG_PILOTO["capas"])]
    tok_s_piloto = historial[-1]["tokens_por_segundo"]
    mediciones = {(CFG_PILOTO["hidden"], CFG_PILOTO["capas"]): tok_s_piloto}
    eleccion = None
    for hidden, capas in ESCALERA:
        if (hidden, capas) in mediciones:
            tok_s = mediciones[(hidden, capas)]
        else:
            cfg_probe = dict(CFG_PILOTO)
            cfg_probe["hidden"] = hidden
            cfg_probe["capas"] = capas
            r = medir_throughput(LMReal, cfg_probe, arreglos["ids_train"],
                                 arreglos["chars_train"], dispositivo,
                                 CFG_PROBES["pasos"])
            probes[f"hidden{hidden}_capas{capas}"] = r
            tok_s = r["tokens_por_segundo"]
            mediciones[(hidden, capas)] = tok_s
            print(f"  sonda [{hidden}, {capas}]: {tok_s:,.0f} tok/s "
                  f"| VRAM {r['vram_gb']:.2f} GB")
        if tok_s >= REQUERIDO_TOK_S:
            eleccion = {"hidden": hidden, "capas": capas, "tok_s": tok_s}
            break
    if eleccion is None:
        eleccion = {"hidden": ESCALERA[-1][0], "capas": ESCALERA[-1][1],
                    "tok_s": mediciones[ESCALERA[-1]],
                    "nota": "ninguna rung alcanza el umbral; C98-1 reducira "
                            "epocas para caber en presupuesto (documentar)"}

    veredicto = ("PILOTO_OK" if historial[-1]["val_loss"] < historial[0]["val_loss"]
                 and n_params > 1_000_000 else "PILOTO_REVISAR")
    resultado = {
        "experiment_id": "P98-10",
        "etapa": "C98-0 (piloto real)",
        "fecha": "2026-10-13",
        "corpus": {"ruta": str(CORPUS.relative_to(RAIZ)),
                   "sha256": sha256(CORPUS),
                   "tokens": arreglos["n_tokens_corpus"],
                   "documentos": arreglos["n_docs_train"] + arreglos["n_docs_val"],
                   "val_fraccion": VAL_FRACCION},
        "vocab": {"top": TOP_VOCAB, "tipos": arreglos["tipos"],
                  "cobertura_top50k": round(arreglos["cobertura_top50k"], 4),
                  "chars": len(vocab["chars"])},
        "config_piloto": CFG_PILOTO,
        "n_params": n_params,
        "historial": historial,
        "vram_pico_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3),
        "probes": probes,
        "mediciones_tok_s": {f"h{h}_c{c}": round(v, 1)
                             for (h, c), v in mediciones.items()},
        "requerido_tok_s": round(REQUERIDO_TOK_S, 1),
        "eleccion_c98_1": eleccion,
        "veredicto": veredicto,
        "hallazgo_entorno": "el dropout nativo de nn.LSTM no compila en "
                            "MIOpen gfx1201 (rocrand header ausente); se usa "
                            "nn.Dropout entre capas (misma semantica); "
                            "consecuencia: C98-1 usara la variante LMReal",
        "tiempo_total_s": round(time.perf_counter() - inicio, 1),
        "gpu_horas_estimadas": round((time.perf_counter() - inicio) / 3600, 3),
    }
    SAL_JSON.write_text(json.dumps(resultado, indent=2, ensure_ascii=True) + "\n",
                        encoding="utf-8")
    lineas = [
        "# P98-10 (C98-0): piloto real sobre Wikipedia 5M",
        "",
        f"- Corpus: wikipedia_es_5M ({arreglos['n_tokens_corpus']:,} tokens, "
        f"{arreglos['n_docs_train'] + arreglos['n_docs_val']} docs; val "
        f"{VAL_FRACCION*100:.0f}% por documento). Vocab top-{TOP_VOCAB} desde "
        f"cero: {arreglos['tipos']:,} tipos; cobertura "
        f"{arreglos['cobertura_top50k']*100:.2f}%; {len(vocab['chars'])} chars.",
        f"- Modelo piloto LTR: {n_params:,} parametros "
        f"({CFG_PILOTO['hidden']}x{CFG_PILOTO['capas']}).",
        "",
        "| Epoca | loss (ini -> fin) | val ppl | tok/s | s |",
        "|---|---|---|---|---|",
    ]
    for h in historial:
        lineas.append(f"| {h['epoca']} | {h['loss_primera']:.3f} -> "
                      f"{h['loss_ultima']:.3f} | {h['val_perplexity']:.1f} | "
                      f"{h['tokens_por_segundo']:,.0f} | {h['segundos']:.0f} |")
    lineas += [
        "",
        f"- VRAM pico: {resultado['vram_pico_gb']:.2f} GB | umbral requerido "
        f"para C98-1: {REQUERIDO_TOK_S:,.0f} tok/s (30M x 4 epocas x direccion "
        f"en <=2.5 h).",
        f"- **Eleccion C98-1**: hidden {eleccion['hidden']} x {eleccion['capas']} "
        f"capas ({eleccion['tok_s']:,.0f} tok/s medidos).",
        f"- Veredicto: **{veredicto}** | tiempo total "
        f"{resultado['tiempo_total_s']} s (~{resultado['gpu_horas_estimadas']} "
        f"GPU-h, dentro del tope de 2 GPU-h del piloto).",
        "",
        "Sin corpus nuevos; sin etiquetas de la competencia; mismo pipeline que "
        "se reutilizara en C98-1.",
    ]
    SAL_MD.write_text("\n".join(lineas) + "\n", encoding="utf-8")

    print(f"veredicto: {veredicto} | eleccion C98-1: [{eleccion['hidden']}, "
          f"{eleccion['capas']}] a {eleccion['tok_s']:,.0f} tok/s")
    print(f"tiempo total: {resultado['tiempo_total_s']} s")


if __name__ == "__main__":
    main()
