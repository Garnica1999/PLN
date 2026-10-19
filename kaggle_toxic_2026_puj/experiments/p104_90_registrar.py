#!/usr/bin/env python
"""P104-90: registra las filas P104 en experiments.csv (idempotente).

Registra P104-01 (R4 stemming), P104-02 (R5 clusters) y P104-03 (R7-1
L1/ElasticNet) cuando existan sus artefactos en reports/.
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
    """sha256 truncado a 16 hex."""
    h = hashlib.sha256()
    with open(ruta, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:16]


def jc(valor) -> str:
    """JSON compacto ASCII."""
    return json.dumps(valor, ensure_ascii=True, separators=(",", ":"))


def leer():
    """Lee las filas actuales del registro."""
    with open(RUTA, encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def anexar(nuevas) -> None:
    """Anade filas al final del registro."""
    with open(RUTA, "a", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNAS, lineterminator="\n")
        for f in nuevas:
            w.writerow(f)


def _resumen_variante(r: dict) -> dict:
    """Resumen compacto de una variante para la fila del registro."""
    c = r["canonica"]
    p = r["particion_47"]
    return {"delta": c["delta_vs_v03"],
            "pliegues": c["pliegues_mejorados"],
            "celdas": c["celdas_cambiadas"],
            "ic95": c["bootstrap"]["ic95"],
            "delta_47": p["delta_vs_v03"],
            "gate": r["gate"]["veredicto"]}


def fila_p104_01(d: dict) -> dict:
    """Fila P104-01 (R4) desde su JSON."""
    v = d["variantes"]
    return {
        "experiment_id": "P104-01",
        "competition": "KAGGLE_TOXIC_2026_PUJ",
        "question": "Aporta el stemming Snowball (solo word; char_wb intacto) "
                    "algo que supere el gate A3 frente a V03?",
        "hypothesis": "R4 (RONDA 3): el stemming de la rama de palabra mejora "
                      "la generalizacion",
        "preprocessing": "texto oficial; stemming determinista precomputado "
                         "(sin ajuste, sin test); vectorizador y LR dentro "
                         "del pliegue; umbrales V03 congelados",
        "feature_groups": "S1 = word(1,2) stemmed + char_wb(3,5) crudo; "
                          "S2 = S1 + word(1,2) crudo",
        "representation": "TF-IDF min_df=2 sublinear (stemmer nltk Snowball "
                          "espanol)",
        "model": "LogisticRegression balanced C=2.0 liblinear x3",
        "hyperparameters": jc({"stemmer": "SnowballStemmer spanish"}),
        "validation": "folds canonicos + particion 47",
        "f1": "", "macro_f1": "", "weighted_f1": "", "precision": "",
        "recall": "", "training_time": f"{d['tiempo_s']} s", "dimension": "",
        "result": jc({
            "control_reproduce": d["control"]["reproduce_v03"],
            "S1_word_stem": _resumen_variante(v["S1_word_stem"]),
            "S2_stem_mas_raw": _resumen_variante(v["S2_stem_mas_raw"]),
            "lectura": "efecto NEGATIVO detectable (ambos IC95 excluyen cero "
                       "por abajo)"}),
        "interpretation": "",
        "decision": "R4 CERRADO sin candidato: el stemming PERJUDICA (S2 "
                    "-0.007719 con IC95 [-0.014381,-0.000603]; S1 -0.011893; "
                    "47 -0.007150/-0.009679); coherente con la lematizacion "
                    "PRE-09 (0.7425). Efecto negativo detectable.",
        "date": "2026-10-14",
        "script": "experiments/p104_01_stemming_snowball.py",
        "notes": f"json {sha16(RAIZ / 'reports' / 'p104_01_stemming_snowball.json')}"
                 f" | preregistro {sha16(RAIZ / 'reports' / 'p104_preregistro.md')}",
    }


def fila_p104_02(d: dict) -> dict:
    """Fila P104-02 (R5) desde su JSON."""
    v = d["variantes"]
    return {
        "experiment_id": "P104-02",
        "competition": "KAGGLE_TOXIC_2026_PUJ",
        "question": "Anade una bolsa de clusters de palabra (k-means sobre "
                    "embeddings propios) informacion a la union de V03?",
        "hypothesis": "R5 (RONDA 3): la generalizacion semantica por clusters "
                      "aporta a la representacion",
        "preprocessing": "texto oficial; mapa token->cluster de artefacto "
                         "externo congelado (sin etiquetas/folds/test); "
                         "vectorizador y LR dentro del pliegue",
        "feature_groups": "union V03 + TF-IDF word(1,1) del texto de "
                          "clusters (k 100/300/1000)",
        "representation": "token -> c{cluster} sobre w2v_twitter_lrfix "
                          "(77657 palabras, 200d)",
        "model": "LogisticRegression balanced C=2.0 liblinear x3",
        "hyperparameters": jc({"k": [100, 300, 1000],
                               "kmeans": "MiniBatchKMeans seed 42 batch 4096 "
                                         "n_init 3"}),
        "validation": "folds canonicos + particion 47",
        "f1": "", "macro_f1": "", "weighted_f1": "", "precision": "",
        "recall": "", "training_time": f"{d['tiempo_s']} s", "dimension": "",
        "result": jc({
            "control_reproduce": d["control"]["reproduce_v03"],
            "K100": _resumen_variante(v["K100"]),
            "K300": _resumen_variante(v["K300"]),
            "K1000": _resumen_variante(v["K1000"]),
            "lectura": "efecto NEGATIVO detectable en las tres variantes"}),
        "interpretation": "",
        "decision": "R5 CERRADO sin candidato: los clusters PERJUDICAN (K300 "
                    "-0.014801, IC95 [-0.026471,-0.003431]; K100 -0.015756; "
                    "K1000 -0.017862; 47 negativos); coherente con la familia "
                    "de embeddings cerrada (P85-01). Efecto negativo "
                    "detectable.",
        "date": "2026-10-14",
        "script": "experiments/p104_02_clusters_palabras.py",
        "notes": f"json {sha16(RAIZ / 'reports' / 'p104_02_clusters_palabras.json')}"
                 f" | preregistro {sha16(RAIZ / 'reports' / 'p104_preregistro.md')}",
    }


def fila_p104_03(d: dict) -> dict:
    """Fila P104-03 (R7-1) desde su JSON."""
    c = d["canonica"]
    p = d["particion_47"]
    return {
        "experiment_id": "P104-03",
        "competition": "KAGGLE_TOXIC_2026_PUJ",
        "question": "Supera L1/ElasticNet por etiqueta (seleccion anidada de "
                    "penalty/l1_ratio/C) a la L2 de V03?",
        "hypothesis": "R7-1 (RONDA 3): la seleccion de features por L1 por "
                      "etiqueta aporta senal",
        "preprocessing": "texto oficial; representacion exacta de V03; "
                         "seleccion de configuracion dentro del pliegue",
        "feature_groups": "word(1,2) + char_wb(3,5) (union de V03)",
        "representation": "LR saga penalty l1/elasticnet (16 configs nested)",
        "model": "LogisticRegression saga balanced tol 1e-3 max_iter 3000",
        "hyperparameters": jc({"malla": 16,
                               "seleccion": "CV interna 3 pliegues; F1 de la "
                                            "etiqueta @0.5"}),
        "validation": "folds canonicos + particion 47",
        "f1": "", "macro_f1": "", "weighted_f1": "", "precision": "",
        "recall": "", "training_time": f"{d['tiempo_s']} s", "dimension": "",
        "result": jc({
            "control_reproduce": d["control"]["reproduce_v03"],
            "delta_canonica": c["delta_vs_v03"],
            "pliegues": c["pliegues_mejorados"],
            "celdas": c["celdas_cambiadas"],
            "ic95": c["bootstrap"]["ic95"],
            "piso_deteccion": c["deteccion"]["delta_min_detectable_95"],
            "delta_47": p["delta_vs_v03"],
            "nnz_media": {e: d["nnz_canonica"][e]["media"]
                          for e in ("HS", "TR", "AG")},
            "no_convergencias": len(d["no_convergencias_canonica"]),
            "gate": d["gate"]["veredicto"]}),
        "interpretation": "",
        "decision": "R7-1 CERRADO sin candidato: +0.000917 por debajo del "
                    "piso de deteccion (+0.010222) e IC95 que incluye cero; "
                    "47 -0.000418; seleccion inestable (penalizaciones "
                    "distintas por pliegue); nnz medios HS 4559 / TR 1697 / "
                    "AG 553; 0 no-convergencias. Sin senal detectable.",
        "date": "2026-10-14",
        "script": "experiments/p104_03_l1_elastic.py",
        "notes": f"json {sha16(RAIZ / 'reports' / 'p104_03_l1_elastic.json')}"
                 f" | preregistro {sha16(RAIZ / 'reports' / 'p104_preregistro.md')}",
    }


def main() -> int:
    """Registra las filas P104 pendientes."""
    filas = leer()
    ids = {f["experiment_id"] for f in filas}
    nuevas = []
    pendientes = [
        ("P104-01", "p104_01_stemming_snowball.json", fila_p104_01),
        ("P104-02", "p104_02_clusters_palabras.json", fila_p104_02),
        ("P104-03", "p104_03_l1_elastic.json", fila_p104_03),
    ]
    for eid, nombre, constructor in pendientes:
        ruta = RAIZ / "reports" / nombre
        if eid not in ids and ruta.exists():
            nuevas.append(constructor(json.loads(ruta.read_text(
                encoding="utf-8"))))
    if nuevas:
        anexar(nuevas)
    print(f"P104-90: {len(nuevas)} fila(s) anadida(s); "
          f"total={len(filas) + len(nuevas)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
