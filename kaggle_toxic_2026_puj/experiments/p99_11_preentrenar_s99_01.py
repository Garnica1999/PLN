#!/usr/bin/env python
"""
P99-11 (S99-01): preentrenamiento LTR y RTL sobre 60M tokens unicos.

Reglas pre-registradas (reports/p99_preregistration.md seccion 5):
  - Arquitectura, tokenizador y VOCABULARIO CONGELADOS de C98-1
    (features/p98_c98_1_vocab.json, sha 92fecce6...).
  - 2 epocas por direccion (exposicion ~ 4 epocas de 30M); lr = 1e-3 durante
    el primer 75.1% de los pasos y 5e-4 el resto (equivalente por tokens al
    calendario de C98-1); parada temprana identica.
  - Mismos 3.320 documentos de validacion de lenguaje de P98 (por fuente),
    excluidos del stream de entrenamiento; perplexity por dominio.
  - Presupuesto <=6 GPU-h (LTR+RTL); seeds 20260925.

Salidas: models/p99_s99_01_ltr.pt, models/p99_s99_01_rtl.pt,
reports/p99_pretraining.json y .md, reports/p99_pretraining_results.csv,
reports/p99_progreso.json.

Uso:
    /home/carlos/venv/bin/python experiments/p99_11_preentrenar_s99_01.py
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import time
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


LMReal = _cargar(RAIZ / "experiments" / "p98_10_piloto_c98_0.py", "p98_10").LMReal
m11 = _cargar(RAIZ / "experiments" / "p98_11_construir_c98_1.py", "p98_11")
m12 = _cargar(RAIZ / "experiments" / "p98_12_preentrenar_c98_1.py", "p98_12")

CORPUS = RAIZ / "data/external/corpora/p99_s99_01/tokens_60m.txt"
VOCAB = RAIZ / "features/p98_c98_1_vocab.json"
VOCAB_SHA = "92fecce611f164b7"
SAL_JSON = RAIZ / "reports/p99_pretraining.json"
SAL_MD = RAIZ / "reports/p99_pretraining.md"
SAL_CSV = RAIZ / "reports/p99_pretraining_results.csv"
SAL_PROG = RAIZ / "reports/p99_progreso.json"

SEMILLA = 20260925
MAX_CHAR = 24
EPOCAS = 2
FRACCION_LR = 0.751
CFG = dict(m12.CFG)  # arquitectura congelada de C98-1


def sha256(ruta: Path) -> str:
    """sha256 completo."""
    x = hashlib.sha256()
    with open(ruta, "rb") as fh:
        for bloque in iter(lambda: fh.read(1 << 20), b""):
            x.update(bloque)
    return x.hexdigest()


def cargar_docs() -> list[str]:
    """Documentos del corpus S99-01."""
    docs = []
    with open(CORPUS, encoding="utf-8") as fh:
        for linea in fh:
            if linea.split():
                docs.append(linea.rstrip("\n"))
    return docs


def tabla_caracteres(vocab: dict, tokens_todos: set[str]) -> tuple[dict, np.ndarray]:
    """uid por token + tabla (n, 24) de ids de caracteres (0 = pad)."""
    char_id = {ch: i + 1 for i, ch in enumerate(vocab["chars"])}
    uid: dict[str, int] = {}
    filas = []
    for tok in sorted(tokens_todos):
        fila = np.zeros(MAX_CHAR, dtype=np.uint8)
        for j, ch in enumerate(tok[:MAX_CHAR]):
            fila[j] = char_id.get(ch, 0)
        uid[tok] = len(filas)
        filas.append(fila)
    return uid, np.stack(filas)


def construir_streams(docs: list[str], uid: dict, tabla: np.ndarray,
                      wid: dict):
    """Streams LTR y RTL en una sola pasada (EOS antes de cada doc, k>0).

    `uid`/`tabla` indexan la tabla de CARACTERES (todos los tokens unicos);
    `wid` es el mapa de PALABRAS del vocabulario congelado (top-50k, OOV=0).
    """
    eos = uid["<eos>"]
    largo = sum(len(d.split()) for d in docs) + max(len(docs) - 1, 0)
    ids_f = np.empty(largo, dtype=np.int32)
    ids_b = np.empty(largo, dtype=np.int32)
    chs_f = np.empty((largo, MAX_CHAR), dtype=np.uint8)
    chs_b = np.empty((largo, MAX_CHAR), dtype=np.uint8)
    pos = 0
    for k, doc in enumerate(docs):
        toks = doc.split()
        uids = np.fromiter((uid[t] for t in toks), dtype=np.int64, count=len(toks))
        wids = np.fromiter((wid.get(t, 0) for t in toks), dtype=np.int32,
                           count=len(toks))
        n = len(uids) + (1 if k else 0)
        rango = slice(pos, pos + n)
        if k:
            ids_f[pos] = 0
            ids_b[pos] = 0
            chs_f[pos] = tabla[eos]
            chs_b[pos] = tabla[eos]
            base = pos + 1
        else:
            base = pos
        ids_f[base:base + len(uids)] = wids
        ids_b[base:base + len(uids)] = wids[::-1]
        chs_f[base:base + len(uids)] = tabla[uids]
        chs_b[base:base + len(uids)] = tabla[uids[::-1]]
        pos += n
    assert pos == largo
    assert ids_f.max() < CFG["vocab_size"] and ids_b.max() < CFG["vocab_size"]
    return (ids_f, chs_f, ids_b, chs_b)


def lotes(ids, chars, seq, batch_tokens, inicio: int = 0):
    """Lotes contiguos (palabras, chars, objetivo), desde el lote `inicio`."""
    seqs = batch_tokens // seq
    n = (len(ids) - 1) // seq
    por_lote = seqs * seq
    for b in range(inicio, n // seqs):
        idx = np.arange(b * por_lote, (b + 1) * por_lote).reshape(seqs, seq)
        yield ids[idx], chars[idx], ids[idx + 1]


def evaluar(modelo, streams, perdida_fn, dispositivo, max_lotes=40) -> float:
    """Loss media de validacion sobre un conjunto de streams."""
    modelo.eval()
    perdidas = []
    completo = False
    with torch.no_grad():
        for ids, chs in streams:
            for palabras, ch, objetivo in lotes(ids, chs, CFG["seq_len"],
                                                CFG["batch"]):
                salida = modelo(
                    torch.from_numpy(palabras.astype(np.int64)).to(dispositivo),
                    torch.from_numpy(ch.astype(np.int64)).to(dispositivo))
                perdidas.append(float(perdida_fn(
                    salida.reshape(-1, CFG["vocab_size"]),
                    torch.from_numpy(objetivo.astype(np.int64)).reshape(-1)
                    .to(dispositivo)).detach().cpu()))
                if len(perdidas) >= max_lotes:
                    completo = True
                    break
            if completo:
                break
    modelo.train()
    return float(np.mean(perdidas))


def escribir_progreso(d: dict) -> None:
    """Progreso atomico."""
    tmp = SAL_PROG.with_suffix(".tmp")
    tmp.write_text(json.dumps(d, indent=2, ensure_ascii=True) + "\n",
                   encoding="utf-8")
    tmp.replace(SAL_PROG)


def guardar_estado(ruta, modelo, opt, epoca, paso, historial, mejor,
                   paso_global) -> None:
    """Estado de reanudacion atomico (protege ante interrupciones del GPU)."""
    tmp = ruta.with_suffix(".tmp")
    torch.save({"state_dict": modelo.state_dict(),
                "optimizer": opt.state_dict(),
                "epoca": epoca, "paso": paso, "historial": historial,
                "mejor": list(mejor) if mejor else None,
                "paso_global": paso_global}, tmp)
    tmp.replace(ruta)


def main() -> None:
    """Entrena las dos direcciones sobre S99-01."""
    for ruta in (SAL_JSON, SAL_MD, SAL_CSV):
        if ruta.exists():
            raise SystemExit(f"Ya existe {ruta}; no se sobrescribe.")
    if not torch.cuda.is_available():
        raise SystemExit("GPU/ROCm no disponible")
    dispositivo = torch.device("cuda")
    inicio = time.perf_counter()

    real_vocab = sha256(VOCAB)
    if not real_vocab.startswith(VOCAB_SHA):
        raise SystemExit(f"vocab C98-1 alterado: {real_vocab[:16]}")
    vocab = json.loads(VOCAB.read_text(encoding="utf-8"))
    print(f"vocab congelado verificado ({VOCAB_SHA})", flush=True)

    docs = cargar_docs()
    docs = [d for d in docs if len(d.split()) > 0]
    print(f"corpus S99-01: {len(docs)} documentos", flush=True)

    por_fuente = m12.hashes_por_fuente(m11)
    claves_val_por_fuente = {}
    rng = np.random.default_rng(SEMILLA + 7)
    for nombre, claves in por_fuente.items():
        n_val = max(1, int(round(len(claves) * 0.02)))
        idx = rng.permutation(len(claves))[:n_val]
        claves_val_por_fuente[nombre] = {claves[i] for i in idx.tolist()}
    claves_val = set().union(*claves_val_por_fuente.values())
    print(f"val: {len(claves_val)} documentos (por fuente)", flush=True)

    def clave(doc: str) -> str:
        return hashlib.sha1(m11.normalizar(doc).encode("utf-8")).hexdigest()

    train_docs, val_docs = [], []
    for doc in docs:
        (val_docs if clave(doc) in claves_val else train_docs).append(doc)
    print(f"train {len(train_docs)} docs | val {len(val_docs)} docs", flush=True)

    tokens_train = set()
    for doc in train_docs:
        tokens_train.update(doc.split())
    for doc in val_docs:
        tokens_train.update(doc.split())
    tokens_train.add("<eos>")
    uid, tabla = tabla_caracteres(vocab, tokens_train)
    wid = {tok: i + 1 for i, tok in enumerate(vocab["vocab_top"])}
    print(f"tabla de caracteres: {len(uid)} tokens unicos (train+val); "
          f"palabras congeladas: {len(wid)}", flush=True)

    ids_f, chs_f, ids_b, chs_b = construir_streams(train_docs, uid, tabla, wid)
    print(f"streams de train: {len(ids_f):,} posiciones x2 direcciones",
          flush=True)
    streams_val = {}
    for nombre, claves_f in claves_val_por_fuente.items():
        sub = [d for d in val_docs if clave(d) in claves_f]
        a, b, c, d = construir_streams(sub, uid, tabla, wid)
        streams_val[nombre] = {"ltr": (a, b), "rtl": (c, d)}
    v_f, w_f, v_b, w_b = construir_streams(val_docs, uid, tabla, wid)
    val_streams = {"ltr": (v_f, w_f), "rtl": (v_b, w_b)}

    resultados = {"experiment_id": "P99-11", "etapa": "S99-01 (preentrenamiento)",
                  "fecha": "2026-10-13", "cfg": CFG,
                  "corpus": {"ruta": str(CORPUS.relative_to(RAIZ)),
                             "sha256": sha256(CORPUS),
                             "documentos": len(docs),
                             "tokens_archivo": sum(len(d.split()) for d in docs)},
                  "vocab_congelado": {"ruta": str(VOCAB.relative_to(RAIZ)),
                                      "sha256": real_vocab},
                  "val_documentos": len(val_docs),
                  "epocas": EPOCAS, "fraccion_lr": FRACCION_LR,
                  "direcciones": {}}
    proba = {"ltr": (ids_f, chs_f), "rtl": (ids_b, chs_b)}
    for direccion in ("ltr", "rtl"):
        ids_tr, chs_tr = proba[direccion]
        seqs_por_lote = CFG["batch"] // CFG["seq_len"]
        pasos_epoca = ((len(ids_tr) - 1) // CFG["seq_len"]) // seqs_por_lote
        pasos_total = pasos_epoca * EPOCAS
        paso_cambio = int(round(FRACCION_LR * pasos_total))
        ruta_estado = RAIZ / "models" / f"p99_s99_01_{direccion}_estado.pt"
        ruta_ckpt = RAIZ / "models" / f"p99_s99_01_{direccion}.pt"
        ruta_log = RAIZ / "experiments" / "p99_pretraining.log"
        log_txt = (ruta_log.read_text(encoding="utf-8", errors="replace")
                   if ruta_log.exists() else "")
        completada_en_log = re.search(rf"\[{direccion}\] epoca {EPOCAS}:",
                                      log_txt) is not None
        if (not ruta_estado.exists()) and ruta_ckpt.exists() \
                and completada_en_log:
            historial = []
            patron = re.compile(
                rf"\[{direccion}\] epoca (\d+): loss ([\d.]+) -> ([\d.]+) \| "
                rf"val ppl ([\d.]+) \| ([\d,]+) tok/s \((\d+) s\)")
            for m in patron.finditer(log_txt):
                ep, li, lf, ppl, toks_s, seg = m.groups()
                historial.append({
                    "epoca": int(ep), "tokens": pasos_epoca * CFG["batch"],
                    "pasos": int(ep) * pasos_epoca,
                    "loss_muestra_inicio": float(li),
                    "loss_muestra_fin": float(lf),
                    "val_loss": float(np.log(float(ppl))),
                    "val_perplexity": float(ppl),
                    "val_por_dominio": None,
                    "segundos": float(seg),
                    "tokens_por_segundo": float(toks_s.replace(",", "")),
                    "nota": "reconstruida del log (reanudacion tras una "
                            "corrida previa)"})
            ck = torch.load(ruta_ckpt, map_location=dispositivo,
                            weights_only=True)
            modelo_ck = LMReal(ck["cfg"]).to(dispositivo)
            modelo_ck.load_state_dict(ck["state_dict"])
            modelo_ck.eval()
            ids_v, chs_v = val_streams[direccion]
            with torch.no_grad():
                vl = evaluar(modelo_ck, [(ids_v, chs_v)],
                             nn.CrossEntropyLoss(), dispositivo)
            dominios = {}
            for nombre, dirstreams in streams_val.items():
                ids_d, chs_d = dirstreams[direccion]
                vld = evaluar(modelo_ck, [(ids_d, chs_d)],
                              nn.CrossEntropyLoss(), dispositivo, max_lotes=20)
                dominios[nombre] = {"val_loss": vld,
                                    "perplexity": float(np.exp(vld))}
            historial[-1]["val_por_dominio"] = dominios
            historial[-1]["nota"] += (f"; verificacion de reanudacion: ppl "
                                      f"medida {float(np.exp(vl)):.1f}")
            del modelo_ck
            torch.cuda.empty_cache()
            mejor = (historial[-1]["epoca"], historial[-1]["val_loss"])
            seg_total = sum(e["segundos"] for e in historial)
            resultados["direcciones"][direccion] = {
                "historial": historial,
                "mejor": {"epoca": mejor[0], "val_loss": mejor[1]},
                "gpu_horas": round(seg_total / 3600, 3),
                "nota": "direccion completada en una corrida previa; "
                        "historial reconstruido del log persistente"}
            print(f"[{direccion}] ya completada en corrida previa (log); ppl "
                  f"final {historial[-1]['val_perplexity']:.1f}; per-dominio "
                  f"de la ultima epoca medido ahora", flush=True)
            continue
        print(f"[{direccion}] {pasos_epoca} pasos/epoca | cambio de lr en el "
              f"paso {paso_cambio}", flush=True)
        torch.manual_seed(SEMILLA)
        np.random.seed(SEMILLA)
        modelo = LMReal(CFG).to(dispositivo)
        opt = torch.optim.Adam(modelo.parameters(), lr=CFG["lr"])
        perdida_fn = nn.CrossEntropyLoss()
        historial = []
        mejor = None
        paso_global = 0
        epoca_inicio = 1
        paso_inicio = 0
        pre_resumen_s = 0.0
        if ruta_estado.exists():
            est = torch.load(ruta_estado, map_location=dispositivo,
                             weights_only=False)
            modelo.load_state_dict(est["state_dict"])
            opt.load_state_dict(est["optimizer"])
            historial = est["historial"]
            mejor = tuple(est["mejor"]) if est["mejor"] else None
            paso_global = est["paso_global"]
            epoca_inicio = est["epoca"]
            paso_inicio = est["paso"]
            if historial:
                seg_ep = historial[0]["segundos"]
                pre_resumen_s = seg_ep + (paso_inicio /
                                          max(pasos_epoca, 1)) * seg_ep
            print(f"  [{direccion}] reanudando desde epoca {epoca_inicio}, "
                  f"paso {paso_inicio} (previo ~{pre_resumen_s:.0f} s)",
                  flush=True)
        t_dir = time.perf_counter()
        for epoca in range(epoca_inicio, EPOCAS + 1):
            modelo.train()
            tokens = 0
            muestras = []
            arranque = paso_inicio if epoca == epoca_inicio else 0
            t0 = time.perf_counter()
            generador = lotes(ids_tr, chs_tr, CFG["seq_len"], CFG["batch"],
                              inicio=arranque)
            for i, (palabras, ch, objetivo) in enumerate(generador,
                                                         start=arranque):
                lr = CFG["lr"] if paso_global < paso_cambio else CFG["lr"] * 0.5
                for grupo in opt.param_groups:
                    grupo["lr"] = lr
                salida = modelo(
                    torch.from_numpy(palabras.astype(np.int64)).to(dispositivo),
                    torch.from_numpy(ch.astype(np.int64)).to(dispositivo))
                perdida = perdida_fn(
                    salida.reshape(-1, CFG["vocab_size"]),
                    torch.from_numpy(objetivo.astype(np.int64)).reshape(-1)
                    .to(dispositivo))
                opt.zero_grad()
                perdida.backward()
                nn.utils.clip_grad_norm_(modelo.parameters(), 5.0)
                opt.step()
                paso_global += 1
                tokens += palabras.size
                if i % 250 == 0:
                    muestras.append(float(perdida.detach().cpu()))
                if i % 1000 == 999 or i == pasos_epoca - 1:
                    guardar_estado(ruta_estado, modelo, opt, epoca, i + 1,
                                   historial, mejor, paso_global)
            torch.cuda.synchronize()
            segundos = time.perf_counter() - t0
            ids_v, chs_v = val_streams[direccion]
            val_loss = evaluar(modelo, [(ids_v, chs_v)], perdida_fn, dispositivo)
            por_dominio = {}
            for nombre, dirstreams in streams_val.items():
                ids_d, chs_d = dirstreams[direccion]
                vl = evaluar(modelo, [(ids_d, chs_d)], perdida_fn, dispositivo,
                             max_lotes=20)
                por_dominio[nombre] = {"val_loss": vl,
                                       "perplexity": float(np.exp(vl))}
            fila = {"epoca": epoca, "tokens": tokens, "pasos": paso_global,
                    "loss_muestra_inicio": muestras[0],
                    "loss_muestra_fin": muestras[-1],
                    "val_loss": val_loss,
                    "val_perplexity": float(np.exp(val_loss)),
                    "val_por_dominio": por_dominio,
                    "segundos": round(segundos, 1),
                    "tokens_por_segundo": round(tokens / segundos, 1)}
            historial.append(fila)
            print(f"  [{direccion}] epoca {epoca}: loss {muestras[0]:.3f} -> "
                  f"{muestras[-1]:.3f} | val ppl {np.exp(val_loss):.1f} | "
                  f"{tokens/segundos:,.0f} tok/s ({segundos:.0f} s)", flush=True)
            if mejor is None or val_loss < mejor[1]:
                mejor = (epoca, val_loss)
                torch.save({"cfg": CFG, "state_dict": modelo.state_dict(),
                            "epoca": epoca, "val_loss": val_loss,
                            "direccion": direccion,
                            "vocab_file": str(VOCAB.name)}, 
                           RAIZ / "models" / f"p99_s99_01_{direccion}.pt")
            escribir_progreso({"direccion_actual": direccion,
                               "historial": historial})
            guardar_estado(ruta_estado, modelo, opt, epoca + 1, 0, historial,
                           mejor, paso_global)
            if len(historial) >= 2 and epoca >= 2 and \
                    val_loss >= historial[-2]["val_loss"]:
                print(f"  [{direccion}] parada temprana (sin mejora)",
                      flush=True)
                break
        resultados["direcciones"][direccion] = {
            "historial": historial, "mejor": {"epoca": mejor[0],
                                              "val_loss": mejor[1]},
            "gpu_horas": round(((time.perf_counter() - t_dir) + pre_resumen_s)
                               / 3600, 3)}
        if pre_resumen_s:
            resultados["direcciones"][direccion]["nota_reanudacion"] = (
                f"incluye {pre_resumen_s:.0f} s de una corrida previa "
                f"(derivados del log y del estado persistido)")
        if ruta_estado.exists():
            ruta_estado.unlink()
        torch.cuda.empty_cache()

    ltr = resultados["direcciones"]["ltr"]
    rtl = resultados["direcciones"]["rtl"]
    resultados["gpu_horas_totales"] = round(ltr["gpu_horas"] + rtl["gpu_horas"], 3)
    resultados["tiempo_total_s"] = round(time.perf_counter() - inicio, 1)
    resultados["checkpoints"] = {}
    for d in ("ltr", "rtl"):
        r = resultados["direcciones"][d]
        ruta = RAIZ / "models" / f"p99_s99_01_{d}.pt"
        resultados["checkpoints"][d] = {
            "ruta": f"models/p99_s99_01_{d}.pt", "sha256": sha256(ruta),
            "val_ppl": float(np.exp(r["mejor"]["val_loss"])),
            "epocas": len(r["historial"])}
    SAL_JSON.write_text(json.dumps(resultados, indent=2, ensure_ascii=True)
                        + "\n", encoding="utf-8")

    filas = ["direccion,epoca,tokens,pasos,val_loss,val_perplexity,"
             "ppl_wikipedia,ppl_mc4,ppl_twitter,segundos,tok_s,checkpoint_sha256"]
    for d in ("ltr", "rtl"):
        sha = resultados["checkpoints"][d]["sha256"]
        for h in resultados["direcciones"][d]["historial"]:
            pd_ = h.get("val_por_dominio")
            dom = (f"{pd_['wikipedia']['perplexity']:.3f},"
                   f"{pd_['mc4']['perplexity']:.3f},"
                   f"{pd_['twitter']['perplexity']:.3f}" if pd_ else "n/a")
            filas.append(f"{d},{h['epoca']},{h['tokens']},{h['pasos']},"
                         f"{h['val_loss']:.6f},{h['val_perplexity']:.3f},{dom},"
                         f"{h['segundos']},{h['tokens_por_segundo']},{sha}")
    SAL_CSV.write_text("\n".join(filas) + "\n", encoding="utf-8")

    md = ["# P99-11 (S99-01): preentrenamiento sobre 60M tokens unicos", "",
          f"- Corpus: {resultados['corpus']['tokens_archivo']:,} tokens "
          f"({resultados['corpus']['documentos']} documentos; sha "
          f"{resultados['corpus']['sha256'][:16]}...); vocab congelado de C98-1.",
          f"- Val: {len(val_docs)} documentos (mismos de P98), por fuente.", "",
          "| Dir | Epoca | val ppl | ppl wiki/mC4/twitter | tok/s |",
          "|---|---|---|---|---|"]
    for d in ("ltr", "rtl"):
        for h in resultados["direcciones"][d]["historial"]:
            pd_ = h.get("val_por_dominio")
            domstr = (f"{pd_['wikipedia']['perplexity']:.1f}/"
                      f"{pd_['mc4']['perplexity']:.1f}/"
                      f"{pd_['twitter']['perplexity']:.1f}" if pd_ else
                      "n/a (reconstruida)")
            md.append(f"| {d.upper()} | {h['epoca']} | {h['val_perplexity']:.1f} | "
                      f"{domstr} | "
                      f"{h['tokens_por_segundo']:,.0f} |")
    md += ["",
           f"- Checkpoints: LTR ppl {resultados['checkpoints']['ltr']['val_ppl']:.1f};"
           f" RTL ppl {resultados['checkpoints']['rtl']['val_ppl']:.1f}; GPU-h "
           f"{resultados['gpu_horas_totales']} (tope 6).",
           "", "La perplexity NO es criterio de adopcion (P99 seccion 9)."]
    SAL_MD.write_text("\n".join(md) + "\n", encoding="utf-8")
    escribir_progreso({"estado": "completo",
                       "gpu_horas": resultados["gpu_horas_totales"]})
    print(f"S99-01 completo: LTR ppl "
          f"{resultados['checkpoints']['ltr']['val_ppl']:.1f} | RTL ppl "
          f"{resultados['checkpoints']['rtl']['val_ppl']:.1f} | GPU-h "
          f"{resultados['gpu_horas_totales']}", flush=True)


if __name__ == "__main__":
    main()
