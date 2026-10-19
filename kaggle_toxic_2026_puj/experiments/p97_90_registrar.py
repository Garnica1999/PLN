#!/usr/bin/env python
"""
P97-90: registra las unidades de la fase P97 en experiments.csv.

Unidades: P97-00 (documental), P97-01 (+control global-nested), P97-02
(+control A0; +2 instanciaciones invalidas preservadas), P97-03 (+control
torch BCE; +2 instanciaciones invalidas preservadas), P97-04 (+control de la
replica). Solo anade filas nuevas; no modifica filas historicas.

Uso:
    /home/carlos/venv/bin/python experiments/p97_90_registrar.py
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
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
    """JSON compacto ASCII para celdas del registro."""
    return json.dumps(valor, ensure_ascii=True, separators=(",", ":"))


def main() -> None:
    """Anade las unidades P97 al registro."""
    with open(RUTA, encoding="utf-8-sig", newline="") as fh:
        filas = list(csv.DictReader(fh))
    if any(f["experiment_id"].startswith("P97") for f in filas):
        raise SystemExit("Ya hay filas P97 en experiments.csv; no se duplica.")
    n_antes = len(filas)

    d01 = json.load(open(RAIZ / "reports" / "p97_01_reglas_por_segmento.json",
                         encoding="utf-8"))
    d02 = json.load(open(RAIZ / "reports" / "p97_02_group_dro.json",
                         encoding="utf-8"))
    d03 = json.load(open(RAIZ / "reports" / "p97_03_nnpu_tr.json",
                         encoding="utf-8"))
    d04 = json.load(open(RAIZ / "reports" / "p97_04_replicacion.json",
                         encoding="utf-8"))
    d04i = json.load(open(RAIZ / "reports" / "p97_04_replicacion.json",
                          encoding="utf-8"))
    f01 = sha16(RAIZ / "reports" / "p97_01_reglas_por_segmento.json")
    f02 = sha16(RAIZ / "reports" / "p97_02_group_dro.json")
    f03 = sha16(RAIZ / "reports" / "p97_03_nnpu_tr.json")
    f04 = sha16(RAIZ / "reports" / "p97_04_replicacion.json")
    s01 = sha16(RAIZ / "experiments" / "p97_01_reglas_por_segmento.py")
    s02 = sha16(RAIZ / "experiments" / "p97_02_group_dro.py")
    s03 = sha16(RAIZ / "experiments" / "p97_03_nnpu_tr.py")
    s04 = sha16(RAIZ / "experiments" / "p97_04_replicacion.py")
    sh00 = sha16(RAIZ / "experiments" / "p97_00_generador_registro.py")
    reg = list(csv.DictReader(open(RAIZ / "reports" / "p97_method_registry.csv",
                                   encoding="utf-8")))
    mat = list(csv.DictReader(open(RAIZ / "reports" / "p97_novelty_matrix.csv",
                                   encoding="utf-8")))
    comparacion01 = d01["comparaciones"]

    def fila(eid, question, hypothesis, result, decision, script, notes):
        """Construye una fila con el esquema de experiments.csv."""
        return {
            "experiment_id": eid, "competition": "KAGGLE_TOXIC_2026_PUJ",
            "question": question, "hypothesis": hypothesis,
            "preprocessing": "ninguno (sobre OOF canonico guardado)"
            if eid.startswith("P97-01") else
            "TF-IDF word(1,2)+char_wb(3,5) min_df=2 sublinear, ajustado en el "
            "train del pliegue",
            "feature_groups": "no aplica", "representation": "V03 union"
            if not eid.startswith("P97-01") else "proba OOF de V03",
            "model": "capa de decision" if eid.startswith("P97-01") else
            "LogisticRegression C=2 balanced liblinear" if eid.startswith("P97-02")
            else "lineal torch (Adam full-batch, 400 pasos)",
            "hyperparameters": "n/a", "validation": "StratifiedGroupKFold(5, seed 42) "
            "grupos near-dup UNION pseudo-autor" if not eid.startswith("P97-04")
            else "StratifiedGroupKFold(5, seed 43) sobre dev (particion de robustez)",
            "f1": "", "macro_f1": "", "weighted_f1": "", "precision": "",
            "recall": "", "training_time": "", "dimension": "",
            "result": result, "interpretation": "", "decision": decision,
            "date": "2026-10-13", "script": script, "notes": notes,
            "ignorar": "",
        }

    nuevas = []
    nuevas.append(fila(
        "P97-00",
        "Que mecanismos externos (2023-2026 y clasicos) son genuinamente nuevos "
        "frente al historial y compatibles con TOXIC?",
        "Una busqueda independiente con fuentes primarias identifica candidatos "
        "D/B no subsumidos por P94",
        jc({"mecanismos_registrados": len(reg),
            "clases": {c: sum(1 for f in reg if f["clasificacion"].startswith(c))
                       for c in ("A", "B", "C", "D", "E")},
            "candidatos": len(mat),
            "seleccionados": [f["candidate_id"] for f in mat
                              if f["decision"].startswith("SELECCIONADA")]}),
        "REGISTRO: 3 hipotesis seleccionadas (H97-A decision condicional; "
        "H97-B DRO minimax; H97-C nnPU/PNU). 0 entrenamientos, 0 submissions.",
        "experiments/p97_00_generador_registro.py",
        f"git_head e43c594 | 16 busquedas externas 2026-10-13 | fuentes "
        f"primarias arXiv/ACL/PMLR/NeurIPS/JASA | sha16 {sh00} | informes: "
        f"reports/p97_independent_research.md, p97_method_registry.csv, "
        f"p97_novelty_matrix.csv"))
    nuevas.append(fila(
        "P97-01",
        "Un umbral por (etiqueta, cuartil de longitud) elegido solo en el train "
        "del pliegue supera a la regla global anidada en las mismas condiciones?",
        "H97-A: la decision condicionada por segmento mejora sobre la global",
        jc({"segmento_nested": round(
            comparacion01["segmento_vs_frozen"]["delta"]
            + d01["reglas"]["frozen"]["corrected_cv"], 10),
            "global_nested": round(d01["reglas"]["global_nested"]["corrected_cv"], 10),
            "delta_vs_global_nested": round(
                comparacion01["segmento_vs_global_nested"]["delta"], 6),
            "pliegues_mejorados": comparacion01["segmento_vs_global_nested"]
            ["pliegues_mejorados"],
            "ic95_clusters": comparacion01["segmento_vs_global_nested"]
            ["bootstrap_clusters"]["ic95"]}),
        "FALSIFICADA: la regla condicional pierde -0.0078 frente a la global "
        "anidada (1/5 pliegues; IC incluye cero); la global nested reproduce el "
        "+0.000925 historico de P55/V56.",
        "experiments/p97_01_reglas_por_segmento.py",
        f"seed 20260925 bootstrap | cuartiles con fronteras del train por pliegue "
        f"| rejilla 0.20-0.80 paso 0.01 (identica a TH-01) | sha16 {s01} | "
        f"json {f01}"))
    nuevas.append(fila(
        "P97-01C",
        "Control del protocolo nested para H97-A: un umbral global por etiqueta "
        "elegido en los otros pliegues",
        "Control (no consume presupuesto): replica del umbral anidado global",
        jc({"corrected_cv": round(d01["reglas"]["global_nested"]["corrected_cv"], 10),
            "delta_vs_frozen": round(
                d01["reglas"]["global_nested"]["corrected_cv"]
                - d01["reglas"]["frozen"]["corrected_cv"], 6)}),
        "CONTROL: +0.000925 sobre frozen, identico al registrado en "
        "P55/V56-01; confirma que el arnes reproduce el protocolo historico.",
        "experiments/p97_01_reglas_por_segmento.py",
        f"control_registrado | sha16 {s01} | json {f01}"))
    nuevas.append(fila(
        "P97-02-INV1",
        "Ejecucion invalida de H97-B (pre-correccion): sample_weight sin "
        "normalizar frente a la escala de C de liblinear",
        "n/a (unidad de entrenamiento invalida, preservada)",
        jc({"corrected_cv": 0.445735011, "causa": "colapso de escala liblinear "
            "(pesos ~1/n y C=2 multiplicado por la suma de pesos)"}),
        "INVALID_EXECUTION: defecto de escala; artefactos preservados en "
        "p97_02_group_dro_escala_invalida.* y features/p97_02_oof_dro_escala_invalida.npz.",
        "experiments/p97_02_group_dro.py",
        f"no cuenta como resultado del mecanismo | pesos normalizados a media 1 "
        f"en la ejecucion valida | sha16 script {s02}"))
    nuevas.append(fila(
        "P97-02",
        "El minimax por peor grupo (DRO, T=15, eta=0.5) mejora a V03 o compra "
        "robustez sobre el control uniforme?",
        "H97-B: la reponderacion adversaria de grupos transfiere mejor",
        jc({"control_A0": round(d02["control_A0"]["corrected_cv"], 10),
            "dro": round(d02["dro"]["corrected_cv"], 10),
            "delta_pooled": round(d02["comparacion"]["delta_pooled"], 6),
            "pliegues_mejorados": d02["comparacion"]["pliegues_mejorados"],
            "ic95_clusters": d02["comparacion"]["bootstrap_clusters"]["ic95"],
            "std_control": round(d02["control_A0"]["std_pliegues"], 5),
            "std_dro": round(d02["dro"]["std_pliegues"], 5),
            "error_grupo_p90_control": d02["comparacion"]
            ["error_por_grupo_control"]["p90"],
            "error_grupo_p90_dro": d02["comparacion"]["error_por_grupo_dro"]["p90"]}),
        "FALSIFICADA: -0.005170 pooled (3/5 pliegues), sin mejora de cola "
        "(p90/p99 de error por grupo iguales) y con mas dispersion; el minimax "
        "no compra robustez util en este regimen.",
        "experiments/p97_02_group_dro.py",
        f"seed 20260925 | control A0 reproduce V03 exacto "
        f"({d02['control_A0']['reproduce_v03_exacto']}) | sha16 {s02} | json {f02}"))
    nuevas.append(fila(
        "P97-02C",
        "Control A0 de H97-B: pesos uniformes en el mismo arnes",
        "Control (no consume presupuesto): reproduce V03",
        jc({"corrected_cv": round(d02["control_A0"]["corrected_cv"], 10),
            "reproduce_v03_exacto": d02["control_A0"]["reproduce_v03_exacto"]}),
        "CONTROL: reproduce 0.7772114525165721 exacto.",
        "experiments/p97_02_group_dro.py",
        f"control_registrado | sha16 {s02} | json {f02}"))
    nuevas.append(fila(
        "P97-03-INV1",
        "Ejecucion invalida de H97-C (pre-correccion): Adam wd=0.5 (2400x el "
        "equivalente liblinear de C=2)",
        "n/a (unidad de entrenamiento invalida, preservada)",
        jc({"control_interno": 0.6888590064, "nnpu": 0.5994521889,
            "causa": "subentrenamiento por regularizacion excesiva del arnes"}),
        "INVALID_EXECUTION: escala L2; artefactos preservados en "
        "p97_03_nnpu_tr_l2_escala_invalida.*.",
        "experiments/p97_03_nnpu_tr.py",
        f"no cuenta como resultado del mecanismo | sha16 script {s03}"))
    nuevas.append(fila(
        "P97-03-INV2",
        "Ejecucion invalida de H97-C (pre-correccion): codificacion P/N/U "
        "erronea (N entro como positivo; idx_n vacio)",
        "n/a (unidad de entrenamiento invalida, preservada)",
        jc({"control_interno": 0.7748502494, "nnpu": 0.6576847505,
            "causa": "y_efectiva mal codificada; negativos definidos ausentes"}),
        "INVALID_EXECUTION: codificacion; artefactos preservados en "
        "p97_03_nnpu_tr_pn_idx_invalido.*.",
        "experiments/p97_03_nnpu_tr.py",
        f"no cuenta como resultado del mecanismo | sha16 script {s03}"))
    nuevas.append(fila(
        "P97-03",
        "El estimador nnPU/PNU (P=TR1; N=TR0 y HS1; U=HS0) mejora la cabeza TR "
        "frente al control BCE identico en el mismo arnes?",
        "H97-C: tratar HS=0 como mezcla no supervisada aporta senal a TR",
        jc({"propia_control": round(d03["reglas"]["propia_control"]["corrected_cv"], 10),
            "propia_nnpu": round(d03["reglas"]["propia_nnpu"]["corrected_cv"], 10),
            "delta_pooled": round(d03["comparacion"]["delta_pooled"], 6),
            "pliegues_mejorados": d03["comparacion"]["pliegues_mejorados"],
            "delta_F1_TR": round(d03["comparacion"]["delta_F1_TR"], 6),
            "pliegues_TR_mejorados": d03["comparacion"]["pliegues_TR_mejorados"],
            "ic95_clusters": d03["comparacion"]["bootstrap_clusters"]["ic95"],
            "pct_mejora_bootstrap": d03["comparacion"]["bootstrap_clusters"]
            ["pct_mejora"],
            "correccion_vs_control": d03["comparacion"]["correccion"]["vs_control"],
            "correccion_vs_v03": d03["comparacion"]["correccion"]["vs_v03_frozen"]}),
        "SENAL DENTRO DEL ARNES (MARGINAL): +0.002150 pooled con 5/5 pliegues y "
        "TR +0.006450, pero por debajo del gate +0.003 y con IC de clusters "
        "[-0.000461,+0.004896] que incluye cero; frente a V03 en crudo "
        "0.7770004 vs 0.7772115 (-0.0002). Candidata a replicacion.",
        "experiments/p97_03_nnpu_tr.py",
        f"seed 20260925 | P/N/U y wd=1/(C*n) corregidos | sha16 {s03} | json {f03}"))
    nuevas.append(fila(
        "P97-03C",
        "Control de H97-C: torch BCE balanceada identica en el mismo arnes",
        "Control (no consume presupuesto): linea base del arnes torch",
        jc({"corrected_cv": round(d03["reglas"]["propia_control"]["corrected_cv"], 10),
            "delta_vs_v03_crudo": round(
                d03["reglas"]["propia_control"]["corrected_cv"] - 0.7772114525165721, 6)}),
        "CONTROL: 0.7748502494 (-0.002361 vs V03 por diferencias de solver); "
        "sano, sin subentrenamiento.",
        "experiments/p97_03_nnpu_tr.py",
        f"control_registrado | sha16 {s03} | json {f03}"))
    nuevas.append(fila(
        "P97-04",
        "La senal de H97-C (nnpu > control) se replica en una segunda particion "
        "agrupada (seed 43)?",
        "H97-C (replicacion): delta > 0 con >= 4/5 pliegues en particion nueva",
        jc({"propia_control": round(d04["reglas"]["propia_control"]["corrected_cv"], 10),
            "propia_nnpu": round(d04["reglas"]["propia_nnpu"]["corrected_cv"], 10),
            "delta_pooled": round(d04["comparacion"]["delta_pooled"], 6),
            "pliegues_mejorados": d04["comparacion"]["pliegues_mejorados"],
            "delta_F1_TR": round(d04["comparacion"]["delta_F1_TR"], 6),
            "ic95_clusters": d04["comparacion"]["bootstrap_clusters"]["ic95"]}),
        "FALSIFICADA: delta -0.001738 con 1/5 pliegues; la senal de P97-03 no "
        "es robusta a la particion; H97-C queda MARGINAL sin mas pruebas "
        "(condicion de parada pre-registrada).",
        "experiments/p97_04_replicacion.py",
        f"particion seed 43 sobre dev | mismo arnes corregido | sha16 {s04} | "
        f"json {f04}"))
    nuevas.append(fila(
        "P97-04C",
        "Control de la replicacion P97-04: torch BCE en la segunda particion",
        "Control (no consume presupuesto)",
        jc({"corrected_cv": round(d04i["reglas"]["propia_control"]["corrected_cv"], 10)}),
        "CONTROL registrado.",
        "experiments/p97_04_replicacion.py",
        f"control_registrado | sha16 {s04} | json {f04}"))

    with open(RUTA, "a", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNAS, lineterminator="\n")
        for f in nuevas:
            w.writerow({k: f.get(k, "") for k in COLUMNAS})
    with open(RUTA, encoding="utf-8-sig", newline="") as fh:
        filas2 = list(csv.DictReader(fh))
    datos = RUTA.read_bytes()
    no_ascii = sum(1 for b in datos if b > 127)
    print("P97-90 registro: ejecutado")
    print(f"  filas antes: {n_antes} | despues: {len(filas2)} | nuevas: "
          f"{len(filas2) - n_antes}")
    print(f"  ids P97: {sum(1 for f in filas2 if f['experiment_id'].startswith('P97'))}")
    print(f"  bytes > 127 en experiments.csv: {no_ascii} (historico puede tener "
          f"algunos en notas previas; las filas nuevas son ASCII)")
    nuevas_io = [l for l in RUTA.read_text(encoding='utf-8').splitlines()
                 if '"P97-' in l or l.startswith('P97-')]
    print(f"  filas nuevas verificadas: {len(nuevas_io)}")
    if len(filas2) - n_antes != len(nuevas):
        sys.exit(1)


if __name__ == "__main__":
    main()
