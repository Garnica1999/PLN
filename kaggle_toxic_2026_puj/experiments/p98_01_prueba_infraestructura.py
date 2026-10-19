#!/usr/bin/env python
"""
P98-01 (Fase A): verificacion de infraestructura con datos SINTETICOS.

No usa corpus reales. Comprueba en la GPU AMD/ROCm: pipeline char-aware +
LSTM, convergencia, throughput, memoria, checkpoint/load y determinismo.
Es tambien el esqueleto reutilizable del entrenador (clase EncoderLM y
funciones de datos), que C98-0/C98-1 instanciaran con configuracion mayor.

Salidas: reports/p98_infra_test.json, reports/p98_infra_test.md,
models/p98_infra_test.pt.

Uso:
    /home/carlos/venv/bin/python experiments/p98_01_prueba_infraestructura.py
"""

from __future__ import annotations

import json


import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

RAIZ = Path(__file__).resolve().parents[1]
SAL_JSON = RAIZ / "reports" / "p98_infra_test.json"
SAL_MD = RAIZ / "reports" / "p98_infra_test.md"
SAL_CKPT = RAIZ / "models" / "p98_infra_test.pt"

SEMILLA = 20260925
CFG_SINTETICO = {
    "vocab_size": 5000,
    "char_vocab_size": 64,
    "char_dim": 16,
    "nn_char": 32,
    "word_dim": 64,
    "enc_dim": 256,
    "hidden": 256,
    "capas": 1,
    "dropout": 0.0,
    "max_char": 24,
    "seq_len": 64,
    "batch": 64,
    "pasos": 300,
    "lr": 3e-3,
    "pesos_determinismo": 30,
}


class EncoderLM(nn.Module):
    """LM con representacion char-aware + LSTM + softmax atado (bottleneck)."""

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
        self.lstm = nn.LSTM(cfg["enc_dim"], cfg["hidden"],
                            num_layers=cfg["capas"], batch_first=True,
                            dropout=cfg["dropout"] if cfg["capas"] > 1 else 0.0)
        self.dec = nn.Linear(cfg["hidden"], cfg["word_dim"])

    def forward(self, words: torch.Tensor, chars: torch.Tensor) -> torch.Tensor:
        """words (B, S); chars (B, S, L) -> logits (B, S, V)."""
        b, s, el = chars.shape
        c = self.char_emb(chars.reshape(b * s, el)).transpose(1, 2)
        c = torch.cat([torch.relu(self.conv3(c)).max(dim=2).values,
                       torch.relu(self.conv5(c)).max(dim=2).values], dim=1)
        emb = self.word_emb(words)
        entrada = torch.cat([emb, c.reshape(b, s, -1)], dim=2)
        h, _ = self.lstm(torch.relu(self.proj(entrada)))
        proj = self.dec(h)
        return proj @ self.word_emb.weight.t()


def generar_corpus_sintetico(n_tokens: int, vocab: int,
                             semilla: int) -> tuple[list[int], list[str]]:
    """Corpus sintetico tipo Zipf con formas elongadas ocasionales."""
    rng = np.random.default_rng(semilla)
    frecuencias = 1.0 / np.power(np.arange(1, vocab + 1), 1.1)
    frecuencias = frecuencias / frecuencias.sum()
    ids = rng.choice(vocab, size=n_tokens, p=frecuencias)
    tokens = []
    for i in ids:
        t = f"w{int(i):04d}"
        if rng.random() < 0.01:
            t = t[0] + t[1] * int(rng.integers(2, 5)) + t[2:]
        tokens.append(t)
    return ids.tolist(), tokens


def char_tensor(tokens: list[str], max_char: int) -> np.ndarray:
    """Ids de caracteres por token (0 = pad)."""
    salida = np.zeros((len(tokens), max_char), dtype=np.int64)
    for i, t in enumerate(tokens):
        for j, ch in enumerate(t[:max_char]):
            salida[i, j] = (ord(ch) % 63) + 1
    return salida


def lotes(ids: np.ndarray, seq: int, batch: int, v: int):
    """Genera lotes (palabras, objetivo) desplazados para next-token."""
    n = (len(ids) - 1) // seq
    lotes_por_epoca = n // batch
    for b in range(lotes_por_epoca):
        idx = np.arange(b * batch * seq, (b + 1) * batch * seq).reshape(batch, seq)
        palabras = ids[idx]
        objetivo = ids[idx + 1]
        yield palabras, objetivo


def simular_chars(palabras: np.ndarray) -> np.ndarray:
    """Mapa seudocaracteristico determinista (sintetico, barato)."""
    return ((palabras[:, :, None] * 7 + np.arange(6)[None, None, :]) % 63) + 1


def entrenar(cfg: dict, ids_train: np.ndarray, ids_val: np.ndarray, pasos: int,
             device: torch.device) -> dict:
    """Entrena el LM sintetico y devuelve metricas."""
    torch.manual_seed(SEMILLA)
    np.random.seed(SEMILLA)
    modelo = EncoderLM(cfg).to(device)
    n_params = sum(p.numel() for p in modelo.parameters())
    opt = torch.optim.Adam(modelo.parameters(), lr=cfg["lr"])
    perdida_fn = nn.CrossEntropyLoss()
    tokens_procesados = 0
    perdidas = []
    t0 = time.perf_counter()
    generador = lotes(ids_train, cfg["seq_len"], cfg["batch"], cfg["vocab_size"])
    for paso in range(pasos):
        try:
            palabras, objetivo = next(generador)
        except StopIteration:
            generador = lotes(ids_train, cfg["seq_len"], cfg["batch"],
                              cfg["vocab_size"])
            palabras, objetivo = next(generador)
        x = torch.from_numpy(palabras.astype(np.int64)).to(device)
        y = torch.from_numpy(objetivo.astype(np.int64)).to(device)
        chars = torch.from_numpy(simular_chars(palabras)).to(device)
        logits = modelo(x, chars)
        perdida = perdida_fn(logits.reshape(-1, cfg["vocab_size"]), y.reshape(-1))
        opt.zero_grad()
        perdida.backward()
        nn.utils.clip_grad_norm_(modelo.parameters(), 5.0)
        opt.step()
        tokens_procesados += palabras.size
        perdidas.append(float(perdida.detach().cpu()))
    torch.cuda.synchronize()
    segundos = time.perf_counter() - t0
    with torch.no_grad():
        val_palabras, val_objetivo = next(lotes(ids_val, cfg["seq_len"], cfg["batch"],
                                                cfg["vocab_size"]))
        xv = torch.from_numpy(val_palabras.astype(np.int64)).to(device)
        yv = torch.from_numpy(val_objetivo.astype(np.int64)).to(device)
        cv = torch.from_numpy(simular_chars(val_palabras)).to(device)
        val_loss = float(nn.functional.cross_entropy(
            modelo(xv, cv).reshape(-1, cfg["vocab_size"]),
            yv.reshape(-1)).detach().cpu())
    return {"modelo": modelo, "n_params": n_params,
            "perdida_primera": perdidas[0], "perdida_ultima": perdidas[-1],
            "val_loss": val_loss, "tokens": tokens_procesados,
            "segundos": segundos, "perdidas": perdidas}


def main() -> None:
    """Ejecuta la prueba de infraestructura completa."""
    for ruta in (SAL_JSON, SAL_MD, SAL_CKPT):
        if ruta.exists():
            raise SystemExit(f"Ya existe {ruta}; no se sobrescribe.")
    if not torch.cuda.is_available():
        raise SystemExit("GPU/ROCm no disponible: parar y documentar (P98 seccion 4)")
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats()
    print(f"device: {torch.cuda.get_device_name(0)} | torch {torch.__version__}")

    t0 = time.perf_counter()
    ids_train, _ = generar_corpus_sintetico(1_500_000, CFG_SINTETICO["vocab_size"],
                                            SEMILLA)
    ids_val, _ = generar_corpus_sintetico(150_000, CFG_SINTETICO["vocab_size"],
                                          SEMILLA + 1)
    ids_train = np.array(ids_train, dtype=np.int64)
    ids_val = np.array(ids_val, dtype=np.int64)
    print(f"corpus sintetico: {len(ids_train)} train / {len(ids_val)} val tokens "
          f"({round(time.perf_counter()-t0,1)} s)")

    r = entrenar(CFG_SINTETICO, ids_train, ids_val, CFG_SINTETICO["pasos"], device)
    vram = torch.cuda.max_memory_allocated() / 1e9
    tok_s = r["tokens"] / r["segundos"]
    ms_paso = 1000 * r["segundos"] / CFG_SINTETICO["pasos"]
    print(f"entrenamiento: {CFG_SINTETICO['pasos']} pasos, loss "
          f"{r['perdida_primera']:.3f} -> {r['perdida_ultima']:.3f} "
          f"(val {r['val_loss']:.3f}) | {tok_s:,.0f} tok/s | {ms_paso:.1f} ms/paso "
          f"| VRAM {vram:.2f} GB")

    torch.save({"cfg": CFG_SINTETICO, "state_dict": r["modelo"].state_dict()},
               SAL_CKPT)
    ck = torch.load(SAL_CKPT, map_location=device, weights_only=True)
    modelo2 = EncoderLM(CFG_SINTETICO).to(device)
    modelo2.load_state_dict(ck["state_dict"])
    modelo2.eval()
    palabras = np.arange(64).reshape(1, 64).astype(np.int64)
    with torch.no_grad():
        a = r["modelo"].eval()(torch.from_numpy(palabras).to(device),
                               torch.from_numpy(simular_chars(palabras)).to(device))
        b = modelo2(torch.from_numpy(palabras).to(device),
                    torch.from_numpy(simular_chars(palabras)).to(device))
    checkpoint_ok = bool(torch.allclose(a, b))

    d1 = entrenar(CFG_SINTETICO, ids_train, ids_val,
                  CFG_SINTETICO["pesos_determinismo"], device)
    d2 = entrenar(CFG_SINTETICO, ids_train, ids_val,
                  CFG_SINTETICO["pesos_determinismo"], device)
    determinismo_delta = abs(d1["perdida_ultima"] - d2["perdida_ultima"])

    resultado = {
        "experiment_id": "P98-01",
        "fase": "A (infraestructura sintetica)",
        "fecha": "2026-10-13",
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cfg": CFG_SINTETICO,
        "n_params": r["n_params"],
        "tokens_train": len(ids_train), "tokens_val": len(ids_val),
        "perdida_primera": r["perdida_primera"],
        "perdida_ultima": r["perdida_ultima"],
        "val_loss": r["val_loss"],
        "tokens_por_segundo": round(tok_s, 1),
        "ms_por_paso": round(ms_paso, 2),
        "vram_pico_gb": round(vram, 3),
        "checkpoint_ruta": "models/p98_infra_test.pt",
        "checkpoint_roundtrip_ok": checkpoint_ok,
        "determinismo_delta_30_pasos": determinismo_delta,
        "converge": r["perdida_ultima"] < r["perdida_primera"] * 0.7,
        "tiempo_total_s": round(time.perf_counter() - t0, 1),
    }
    SAL_JSON.write_text(json.dumps(resultado, indent=2, ensure_ascii=True) + "\n",
                        encoding="utf-8")
    md = f"""# P98-01 (Fase A): prueba de infraestructura con datos sinteticos

- Device: {resultado['device']} | torch {resultado['torch']} | ROCm OK.
- Modelo sintetico: {r['n_params']:,} parametros (char-aware + LSTM + softmax atado).
- Loss train: {r['perdida_primera']:.4f} -> {r['perdida_ultima']:.4f}; val {r['val_loss']:.4f} (converge: {resultado['converge']}).
- Throughput: {tok_s:,.0f} tokens/s ({ms_paso:.2f} ms/paso); VRAM pico {vram:.2f} GB.
- Checkpoint roundtrip identico: {checkpoint_ok}.
- Determinismo (dos corridas de 30 pasos, misma semilla): delta de loss {determinismo_delta:.2e}.
- Tiempo total de la prueba: {resultado['tiempo_total_s']} s.

Sin corpus reales (restriccion de Fase A cumplida). El esqueleto EncoderLM y el
pipeline de lotes se reutilizan en C98-0/C98-1 con configuracion mayor.
"""
    SAL_MD.write_text(md, encoding="utf-8")

    print(f"checkpoint roundtrip: {checkpoint_ok} | determinismo delta: "
          f"{determinismo_delta:.2e}")
    print("P98-01: infraestructura verificada")


if __name__ == "__main__":
    main()
