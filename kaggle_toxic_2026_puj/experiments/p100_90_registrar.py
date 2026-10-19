#!/usr/bin/env python
"""P100-90: registra las filas P100 en experiments.csv (idempotente).

Registra P100-00 (pre-registro S2), P100-10 (construccion de B1) y, si
existen sus artefactos, P100-11/12 (entrenamiento LTR/RTL) y P100-13/14
(downstream canonica/particion 47).
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
    h = hashlib.sha256()
    with open(ruta, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:16]


def jc(valor) -> str:
    return json.dumps(valor, ensure_ascii=True, separators=(",", ":"))


def leer():
    with open(RUTA, encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def anexar(nuevas) -> None:
    with open(RUTA, "a", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNAS, lineterminator="\n")
        for f in nuevas:
            w.writerow(f)


def main() -> int:
    filas = leer()
    ids = {f["experiment_id"] for f in filas}
    nuevas = []
    if "P100-00" not in ids:
        nuevas.append({
            "experiment_id": "P100-00",
            "competition": "KAGGLE_TOXIC_2026_PUJ",
            "question": "Una composicion de corpus que sustituye ~9M tokens "
                        "de masa general por dialogo conversacional "
                        "autorizado (ChatSubs es) a escala fija (~29.73M) "
                        "mejora la representacion contextual y el downstream "
                        "frente a V03?",
            "hypothesis": "H100-01 pre-registrada en "
                          "reports/p100_preregistration.md; variante A "
                          "cerrada por indisponibilidad de texto social "
                          "nuevo (sin relleno)",
            "preprocessing": "n/a (pre-registro)",
            "feature_groups": "n/a", "representation": "n/a",
            "model": "n/a (pre-registro)", "hyperparameters": "n/a",
            "validation": "n/a", "f1": "", "macro_f1": "", "weighted_f1": "",
            "precision": "", "recall": "", "training_time": "", "dimension": "",
            "result": jc({
                "variante_a": "CERRADA_POR_MATERIAL (S100-A sin texto; sin "
                              "relleno por duplicacion)",
                "variante_b": {"social": 4227450, "conversacional": 9000000,
                               "wiki": 11650000, "mc4": 4850000,
                               "total_nominal": 29727450},
                "gate": "delta vs V03 >= +0.003 canonica; >=4/5 pliegues; "
                        "complementariedad neta positiva; positiva y no "
                        "marginal en particion 47; sin defectos; atribucion: "
                        "delta vs fusion30 > 0",
                "presupuesto": "<=2 configuraciones (1 usada); <=10 GPU-h",
                "fuentes_s1": {
                    "S100-A": "PROHIBITED (solo IDs; servicio de texto roto)",
                    "S100-B": "ALLOWED_WITH_RESTRICTIONS (ChatSubs es)"}}),
            "interpretation": "",
            "decision": "PRE-REGISTRADO antes de construir B1 y entrenar; "
                        "sin reabrir P99; V03/P98/P99 intocados.",
            "date": "2026-10-13",
            "script": "experiments/p100_90_registrar.py",
            "notes": (
                f"S1 fuentes {sha16(RAIZ / 'reports' / 'p100_corpus_research.md')} | "
                f"registro {sha16(RAIZ / 'reports' / 'p100_corpus_registry.csv')} | "
                f"decisiones {sha16(RAIZ / 'reports' / 'p100_source_decisions.md')} | "
                f"preregistro {sha16(RAIZ / 'reports' / 'p100_preregistration.md')}"),
        })
    man = RAIZ / "artifacts" / "p100_b1_manifest.json"
    if "P100-10" not in ids and man.exists():
        d = json.loads(man.read_text(encoding="utf-8"))
        nuevas.append({
            "experiment_id": "P100-10",
            "competition": "KAGGLE_TOXIC_2026_PUJ",
            "question": "Construir B1 (social 15% + conversacional 30% + "
                        "general 55%) a escala fija con objetivos exactos y "
                        "controles de procedencia?",
            "hypothesis": "H100-01 pre-registrada; composicion autorizada "
                          "(ChatSubs es, condiciones del propietario)",
            "preprocessing": "replicacion determinista de C98-1; muestreo "
                             "uniforme de ChatSubs es (seed); dedup exacta; "
                             "truncamiento del documento limite a objetivo",
            "feature_groups": "n/a", "representation": "n/a",
            "model": "n/a (construccion)", "hyperparameters": "n/a",
            "validation": "val fijo de 3320 documentos (presentes 3189; "
                          "excluibles del stream)",
            "f1": "", "macro_f1": "", "weighted_f1": "", "precision": "",
            "recall": "", "training_time": f"{d['tiempo_s']} s", "dimension": "",
            "result": jc({"tokens_finales": d["tokens_finales"],
                          "documentos_finales": d["documentos_finales"],
                          "tokens_por_fuente": d["tokens_por_fuente"],
                          "subset_verificacion": d["subset_verificacion"],
                          "val_presentes_en_b1": d["val_presentes_en_b1"],
                          "coincidencias_exactas_vs_train":
                              d["coincidencias_exactas_vs_train"],
                          "overlap_conversacional_vs_s99_01":
                              d["overlap_conversacional_vs_s99_01"],
                          "sha256": d["sha256"]}),
            "interpretation": "",
            "decision": "CONSTRUIDO; controles superados (replica exacta de "
                        "C98-1, composicion exacta, overlaps 0); procede S4.",
            "date": "2026-10-13",
            "script": "experiments/p100_10_construir_b1.py",
            "notes": f"manifiesto {sha16(man)}",
        })
    pre = RAIZ / "reports" / "p100_b1_pretraining.json"
    if pre.exists():
        d = json.loads(pre.read_text(encoding="utf-8"))
        for eid, dir_ in (("P100-11", "ltr"), ("P100-12", "rtl")):
            if eid in ids:
                continue
            c = d["checkpoints"][dir_]
            hist = d["direcciones"][dir_]["historial"]
            diag = hist[-1].get("diagnostico_chatsubs")
            nuevas.append({
                "experiment_id": eid,
                "competition": "KAGGLE_TOXIC_2026_PUJ",
                "question": f"B1 {dir_.upper()}: 30M tokens con composicion "
                            f"social+conversacional x 4 epocas converge y "
                            f"con que perplexity?",
                "hypothesis": "H100-01: composicion a escala fija",
                "preprocessing": "tokens de B1; EOS; vocab congelado de C98-1 "
                                 "(92fecce6...)",
                "feature_groups": "n/a",
                "representation": "char-aware + word emb top-50k (MISMO "
                                  "vocab congelado)",
                "model": f"LMReal {dir_.upper()} (arquitectura congelada de "
                         f"C98-1)",
                "hyperparameters": jc({
                    "lr": "1e-3 hasta 75.1% de pasos, luego 5e-4",
                    "batch_tokens": 8192, "epocas": 4}),
                "validation": "medidor fijo de 3320 documentos "
                              "(materializados desde C98-1)",
                "f1": "", "macro_f1": "", "weighted_f1": "", "precision": "",
                "recall": "",
                "training_time": f"{sum(h['segundos'] for h in hist)} s",
                "dimension": "",
                "result": jc({"val_ppl_por_epoca": [
                    round(h["val_perplexity"], 1) for h in hist],
                    "val_ppl_final": round(c["val_ppl"], 1),
                    "tokens_por_epoca": hist[0]["tokens"],
                    "checkpoint": c["ruta"],
                    "checkpoint_sha256": c["sha256"],
                    "ppl_chatsubs_diag_final": (
                        round(diag["perplexity"], 1) if diag else None)}),
                "interpretation": "",
                "decision": f"ENTRENADO (P100-B1): {dir_.upper()} val ppl "
                            f"{c['val_ppl']:.1f} en {c['epocas']} epoca(s).",
                "date": "2026-10-13",
                "script": "experiments/p100_11_preentrenar_b1.py",
                "notes": f"corpus sha {d['corpus']['sha256'][:16]} | GPU-h "
                         f"json {d['gpu_horas_totales']} (reconstruccion "
                         f"parcial del log; reales ~3.78 sumando las 3 "
                         f"corridas: intento interrumpido + reanudacion + "
                         f"supervisor; ver p100_supervisor.log) | json "
                         f"{sha16(pre)}",
            })
    for eid, ruta_json, etiqueta in (
            ("P100-13", RAIZ / "reports" / "p100_b1_downstream_canonica.json",
             "canonica"),
            ("P100-14", RAIZ / "reports" / "p100_b1_downstream_part47.json",
             "particion 47")):
        if eid in ids or not ruta_json.exists():
            continue
        d = json.loads(ruta_json.read_text(encoding="utf-8"))
        nuevas.append({
            "experiment_id": eid,
            "competition": "KAGGLE_TOXIC_2026_PUJ",
            "question": f"Composicion B1 vs 30M (C98-1) vs V03 en la "
                        f"{etiqueta}: hay mejora downstream atribuible?",
            "hypothesis": "H100-01 (gate de composicion + atribucion)",
            "preprocessing": "identico a D98-1 (reps congeladas + fusion "
                             "v5.1)",
            "feature_groups": "bloque denso 2048",
            "representation": "reps B1 y P98 (control)",
            "model": "LR por pliegue (fusion C=2; emb C=1)",
            "hyperparameters": jc({"umbrales": "V03 congelados"}),
            "validation": f"pliegues {etiqueta}",
            "f1": "", "macro_f1": "", "weighted_f1": "", "precision": "",
            "recall": "", "training_time": f"{d.get('tiempo_total_s', 0)} s",
            "dimension": "2048",
            "result": jc(d["comparacion"]),
            "interpretation": "",
            "decision": (f"{d['gate']['veredicto']}; marginal_solo_canonica="
                         f"{d['gate']['marginal_solo_canonica']}; "
                         f"atribucion_falsada="
                         f"{d['gate']['atribucion_falsada']}"),
            "date": "2026-10-13",
            "script": "experiments/p100_12_downstream_b1.py",
            "notes": f"sha16 {sha16(ruta_json)}",
        })
    if nuevas:
        anexar(nuevas)
    n1 = len(leer())
    print(f"P100-90: registradas {len(nuevas)} filas nuevas | total {n1}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
