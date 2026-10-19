#!/usr/bin/env python
"""
P98-91: registra las unidades de Fase B realizadas (P98-10 piloto, P98-11 corpus).

Solo anade filas nuevas a experiments.csv.

Uso:
    /home/carlos/venv/bin/python experiments/p98_91_registrar_fase_b.py
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
RUTA = RAIZ / "experiments.csv"
COLUMNAS = ["experiment_id", "competition", "question", "hypothesis",
            "preprocessing", "feature_groups", "representation", "model",
            "hyperparameters", "validation", "f1", "macro_f1", "weighted_f1",
            "precision", "recall", "training_time", "dimension", "result",
            "interpretation", "decision", "date", "script", "notes"]


def sha16(ruta: Path) -> str:
    """sha256 truncado a 16 hexdigitos."""
    x = hashlib.sha256()
    with open(ruta, "rb") as fh:
        for bloque in iter(lambda: fh.read(1 << 20), b""):
            x.update(bloque)
    return x.hexdigest()[:16]


def jc(valor: object) -> str:
    """JSON compacto ASCII."""
    return json.dumps(valor, ensure_ascii=True, separators=(",", ":"))


def main() -> None:
    """Anade P98-10 y P98-11."""
    with open(RUTA, encoding="utf-8-sig", newline="") as fh:
        filas = list(csv.DictReader(fh))
    existentes = {f["experiment_id"] for f in filas}
    nuevas_ids = {"P98-10", "P98-11"}
    if existentes & nuevas_ids:
        raise SystemExit("Ya existen filas P98-10/11; no se duplica.")
    n_antes = len(filas)
    piloto = json.load(open(RAIZ / "reports" / "p98_c98_0_piloto.json",
                            encoding="utf-8"))
    corpus = json.load(open(RAIZ / "artifacts" / "p98_c98_1_manifest.json",
                            encoding="utf-8"))
    s10 = sha16(RAIZ / "experiments" / "p98_10_piloto_c98_0.py")
    s11 = sha16(RAIZ / "experiments" / "p98_11_construir_c98_1.py")

    def fila(eid, question, hypothesis, preprocessing, representation, model,
             validation, training_time, result, decision, script, notes):
        """Fila con el esquema de experiments.csv."""
        return {
            "experiment_id": eid, "competition": "KAGGLE_TOXIC_2026_PUJ",
            "question": question, "hypothesis": hypothesis,
            "preprocessing": preprocessing, "feature_groups": "no aplica",
            "representation": representation, "model": model,
            "hyperparameters": "", "validation": validation, "f1": "",
            "macro_f1": "", "weighted_f1": "", "precision": "", "recall": "",
            "training_time": training_time, "dimension": "", "result": result,
            "interpretation": "", "decision": decision, "date": "2026-10-13",
            "script": script, "notes": notes,
        }

    nuevas = [
        fila("P98-10",
             "El pipeline real (vocab top-50k desde cero + char-aware + LSTM) "
             "converge sobre Wikipedia 5M y a que throughput por configuracion?",
             "C98-0: piloto tecnico que fija la escalera de dimensiones para C98-1",
             "tokens EMB-NORMAL locales; split 2% documentos para validacion",
             "char-aware (conv 3/5; 254 chars) + word emb 200 + softmax atado",
             "LTR 2 capas; piloto hidden 512; sondas de throughput",
             "documentos de validacion reservados (no competicion)",
             f"{piloto['tiempo_total_s']} s (~{piloto['gpu_horas_estimadas']} GPU-h)",
             jc({"n_params_piloto": piloto["n_params"],
                 "epocas": [{"epoca": h["epoca"],
                             "val_ppl": round(h["val_perplexity"], 1),
                             "tok_s": h["tokens_por_segundo"]}
                            for h in piloto["historial"]],
                 "sonda_h1024_c2_tok_s": piloto["probes"].get(
                     "hidden1024_capas2", {}).get("tokens_por_segundo"),
                 "requerido_tok_s": piloto["requerido_tok_s"],
                 "eleccion_c98_1": piloto["eleccion_c98_1"],
                 "vram_pico_gb": piloto["vram_pico_gb"],
                 "cobertura_top50k": piloto["vocab"]["cobertura_top50k"]}),
             "PILOTO_OK: val ppl 651.3 -> 351.1 en 3 epocas (5M tokens); la "
             "escalera elige hidden 1024 x 2 capas (17.133 tok/s >= 13.333 "
             "requeridos). Hallazgo de entorno: dropout nativo de nn.LSTM no "
             "compila en MIOpen gfx1201; se usa nn.Dropout entre capas.",
             "experiments/p98_10_piloto_c98_0.py",
             f"seed 20260925 | corpus wikipedia_es_5M "
             f"(sha {piloto['corpus']['sha256'][:16]}) | checkpoint "
             f"models/p98_c98_0_ltr.pt | sha16 script {s10} | json "
             f"{sha16(RAIZ/'reports'/'p98_c98_0_piloto.json')}"),
        fila("P98-11",
             "Se puede construir el corpus C98-1 (30M, 60/25/15) con "
             "deduplicacion cruzada y sin solapamiento con el train?",
             "C98-1: composicion pre-registrada en reports/p98_architecture.md",
             "tokens EMB-NORMAL de wikipedia/mc4/twitter; dedup exacta por "
             "documento; barajado con semilla",
             "no aplica (construccion de corpus)",
             "no aplica (construccion de corpus)",
             "auditoria exacta vs train incluida",
             f"{corpus['tiempo_s']} s",
             jc({"tokens_finales": corpus["tokens_finales"],
                 "documentos_finales": corpus["documentos_finales"],
                 "dedup_removidos": {k: v["duplicados_removidos"]
                                     for k, v in corpus["deduplicacion"].items()},
                 "coincidencias_exactas_vs_train":
                     corpus["coincidencias_exactas_vs_train"],
                 "sha256": corpus["sha256"]}),
             "CORPUS_C98_1_CONSTRUIDO: 29.737.183 tokens, 166.022 documentos "
             "(wikipedia prefijo 18.0M; mc4 7.5M; twitter muestreo 4.5M con "
             "9.656 duplicados internos removidos); 0 coincidencias exactas vs "
             "train; listo para preentrenamiento LTR/RTL.",
             "experiments/p98_11_construir_c98_1.py",
             f"sin descargas nuevas | componentes con hash verificado | sha16 "
             f"script {s11} | manifiesto artifacts/p98_c98_1_manifest.json "
             f"({sha16(RAIZ/'artifacts'/'p98_c98_1_manifest.json')})"),
    ]
    with open(RUTA, "a", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNAS, lineterminator="\n")
        for f in nuevas:
            w.writerow(f)
    with open(RUTA, encoding="utf-8-sig", newline="") as fh:
        filas2 = list(csv.DictReader(fh))
    lineas = (RAIZ / "experiments.csv").read_text(encoding="utf-8").splitlines()
    assert all(all(ord(c) <= 127 for c in l)
               for l in lineas if l.startswith("P98-"))
    print(f"P98-91 registro Fase B: {n_antes} -> {len(filas2)} filas "
          f"(+{len(filas2)-n_antes})")


if __name__ == "__main__":
    main()
