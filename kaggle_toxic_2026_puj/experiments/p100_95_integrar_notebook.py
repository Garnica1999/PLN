#!/usr/bin/env python
"""P100-95: integra la seccion 163 (P100: composicion social+conversacional).

Idempotente y ASCII puro; celda de verificacion ejecutada en aislamiento.
Los resultados se LEEN de los artefactos registrados (no se reejecuta nada).

Uso:
    /home/carlos/venv/bin/python experiments/p100_95_integrar_notebook.py
"""

from __future__ import annotations

import hashlib
import json
import uuid
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
RUTA_NOTEBOOK = RAIZ / "notebooks" / "10_final_pipeline.ipynb"
MARCA_SECCION = "## 163. P100"
MARCA_CODIGO = "Verificacion de la seccion 163 (P100"


def sha16(ruta: Path) -> str:
    """sha256 truncado a 16 hexdigitos."""
    x = hashlib.sha256()
    with open(ruta, "rb") as fh:
        for bloque in iter(lambda: fh.read(1 << 20), b""):
            x.update(bloque)
    return x.hexdigest()[:16]


def nueva_celda_markdown(texto: str) -> dict:
    """Celda de markdown con identificador unico."""
    return {"cell_type": "markdown", "id": uuid.uuid4().hex[:8], "metadata": {},
            "source": texto.splitlines(keepends=True)}


def nueva_celda_codigo(texto: str) -> dict:
    """Celda de codigo vacia de salidas y sin ejecutar."""
    return {"cell_type": "code", "id": uuid.uuid4().hex[:8], "metadata": {},
            "execution_count": None, "outputs": [],
            "source": texto.splitlines(keepends=True)}


CODIGO = '''"""Verificacion de la seccion 163 (P100: composicion social+conversacional).

No entrena nada; solo lee artefactos. El hashing completo tarda ~15 s.
"""

import csv
import hashlib
import json
from pathlib import Path

PROY = PROJECT_ROOT
R = PROY / "reports"
man = json.load(open(PROY / "artifacts" / "p100_b1_manifest.json",
                     encoding="utf-8"))
assert man["tokens_finales"] == 29_727_450
assert man["tokens_por_fuente"] == {"social": 4_227_450, "mc4": 4_850_000,
                                    "conversacional": 9_000_000,
                                    "wikipedia": 11_650_000}
assert man["val_presentes_en_b1"] == 3189
assert man["coincidencias_exactas_vs_train"] == 0
assert man["overlap_conversacional_vs_s99_01"] == 0
assert man["subset_verificacion"]["fallos"] == 0

pre = json.load(open(R / "p100_b1_pretraining.json", encoding="utf-8"))
assert pre["checkpoints"]["ltr"]["val_ppl"] < 100.0
assert pre["checkpoints"]["rtl"]["val_ppl"] < 100.0
assert pre["val_fijo"]["presentes_en_b1_excluidos"] == 3189

can = json.load(open(R / "p100_b1_downstream_canonica.json", encoding="utf-8"))
p47 = json.load(open(R / "p100_b1_downstream_part47.json", encoding="utf-8"))
assert can["control_reproduccion_30M"]["ok"]
assert p47["control_reproduccion_30M_47"]["ok"]
assert can["gate"]["veredicto"] == "GATE_FAILED"
assert p47["gate"]["veredicto"] == "GATE_FAILED"
assert can["gate"]["atribucion_falsada"] and p47["gate"]["atribucion_falsada"]
assert abs(can["comparacion"]["delta_b1_vs_v03"] + 0.000591) < 1e-5
assert abs(can["comparacion"]["delta_b1_vs_30"] + 0.005646) < 1e-5
assert abs(p47["comparacion"]["delta_b1_vs_v03"] + 0.000645) < 1e-5
assert abs(p47["comparacion"]["delta_b1_vs_30"] + 0.001797) < 1e-5
assert can["comparacion"]["pliegues_b1_mejorados_vs_v03"] == 2
assert p47["comparacion"]["pliegues_b1_mejorados_vs_v03"] == 2
b1 = can["comparacion"]["bootstrap_b1_vs_v03"]["ic95"]
assert b1[0] < 0.0 < b1[1]

filas = list(csv.DictReader(open(PROY / "experiments.csv",
                                 encoding="utf-8-sig")))
ids = [f["experiment_id"] for f in filas]
assert len(filas) == 491 and len(set(ids)) == 491
assert {"P100-00", "P100-10", "P100-11", "P100-12", "P100-13",
        "P100-14"} <= set(ids)

fuentes = {
    "data/raw/toxic_train.csv": "3d40e7345b588860",
    "features/oof_v03_final.npz": "559265816ce4f631",
    "experiments/baseline_v03_frozen/submission_v03.csv": "7d7478f271e6edfd",
    "experiments/baseline_v03_frozen/model_v03.joblib": "263ed5bd0fc0de7c",
    "data/processed/holdout_v2_ids.csv": "e5b62e5f22c4c63a",
    "data/processed/holdout2_ids.csv": "f6c6ed958abf728b",
    "data/external/corpora/p98_c98_1/tokens_30m.txt": "9f562718ef4e7d45",
    "features/p98_c98_1_vocab.json": "92fecce611f164b7",
    "data/external/corpora/p99_s99_01/tokens_60m.txt": "e627a7f33886c0a1",
    "data/external/corpora/p100_b1/tokens_30m.txt": "5111a1832fe9567b",
}
for ruta, pref in fuentes.items():
    h = hashlib.sha256((PROY / ruta).read_bytes()).hexdigest()
    assert h.startswith(pref), f"fuente alterada: {ruta}"

for nombre in ("p100_corpus_research.md", "p100_corpus_registry.csv",
               "p100_source_decisions.md", "p100_preregistration.md",
               "p100_auditorias_s1.md", "p100_b1_final_report.md"):
    datos = (R / nombre).read_bytes()
    assert all(b <= 127 for b in datos), f"byte > 127 en {nombre}"

print("P100 verificada:")
print("  B1: 29.727.450 tokens exactos | sha 5111a1832fe9567b | val 3189/3320")
print("  LTR ppl 98.7 | RTL ppl 98.6 | controles 30M reproducidos exactos")
print("  canonica -0.000591 (2/5) y particion 47 -0.000645 (2/5) vs V03;")
print("  -0.005646 / -0.001797 vs 30M -> GATE_FAILED, H100-01 FALSIFIED,")
print("  P100 CLOSED, sin adopcion; embedding-only secundario conservado.")'''


def construir_seccion() -> str:
    """Construye el markdown de la seccion 163 con metricas reales."""
    man = json.loads((RAIZ / "artifacts" / "p100_b1_manifest.json").read_text(
        encoding="utf-8"))
    pre = json.loads((RAIZ / "reports" / "p100_b1_pretraining.json").read_text(
        encoding="utf-8"))
    can = json.loads((RAIZ / "reports" / "p100_b1_downstream_canonica.json")
                     .read_text(encoding="utf-8"))
    p47 = json.loads((RAIZ / "reports" / "p100_b1_downstream_part47.json")
                     .read_text(encoding="utf-8"))
    d98b = json.loads((RAIZ / "reports" / "p98_14_replica_d98_1b.json")
                      .read_text(encoding="utf-8"))
    h = {n: sha16(RAIZ / "reports" / n) for n in (
        "p100_b1_pretraining.json", "p100_b1_downstream_canonica.json",
        "p100_b1_downstream_part47.json")}
    rc = can["reglas"]
    rp = p47["reglas"]
    cc = can["comparacion"]
    cp = p47["comparacion"]
    ic30_can = d98b["comparacion"]["referencia_canonica_D98_1"]["ic95"]
    ic30_47 = d98b["comparacion"]["bootstrap_clusters"]["ic95"]

    def f1e(r: dict) -> str:
        return "/".join(f"{r['f1_por_etiqueta'][e]:.3f}"
                        for e in ("HS", "TR", "AG"))

    def folds(r: dict) -> str:
        return "/".join(f"{x:.4f}" for x in r["f1_pliegues"])

    def corr(r: dict) -> str:
        c = r.get("correccion_vs_ref")
        return (f"{c['rescatadas']}/{c['rotas']}/{c['neto']:+d}" if c else "-")

    return (
        "## 163. P100 - COMPOSICION SOCIAL+CONVERSACIONAL (B1) Y GATE DE "
        "COMPOSICION (2026-10-14)\n"
        "\n"
        "**Decision congelada** (autorizacion del propietario 2026-10-14):\n"
        "\n"
        "```text\n"
        "H100-01 = FALSIFIED | P100-B1 = GATE_FAILED | P100 = CLOSED\n"
        "V03 = FROZEN | NO_MODEL_ADOPTION | NO_NEW_SUBMISSIONS\n"
        "```\n"
        "\n"
        "**Configuracion evaluada (H100-01 pre-registrada en "
        "`p100_preregistration.md`)**: a escala fija (paridad C98-1) "
        "sustituir ~9.0M tokens de masa general por dialogo conversacional "
        "espanol autorizado (ChatSubs es, CC BY-NC 4.0 bajo condiciones del "
        "propietario), conservando el bloque social. La variante A (social "
        "reforzado) quedo cerrada por material (S100-A sin texto "
        "utilizable; sin relleno). Corpus B1: "
        f"{man['tokens_finales']:,} tokens EXACTOS (social "
        f"{man['tokens_por_fuente']['social']:,} + conversacional "
        f"{man['tokens_por_fuente']['conversacional']:,} + wikipedia "
        f"{man['tokens_por_fuente']['wikipedia']:,} + mC4 "
        f"{man['tokens_por_fuente']['mc4']:,}); "
        f"{man['documentos_finales']:,} documentos; sha "
        f"{man['sha256'][:16]}. Controles: replica exacta de C98-1 (claves "
        "identicas), subconjunto verificado, val fijo 3.320 (3.189 excluidos "
        "del stream), overlap train 0, conversacional vs S99-01 0.\n"
        "\n"
        "**Preentrenamiento** (arquitectura/vocab/exposicion congelados; 4 "
        "epocas/direccion; lr 75.1%): LTR ppl "
        f"{pre['checkpoints']['ltr']['val_ppl']:.1f} | RTL ppl "
        f"{pre['checkpoints']['rtl']['val_ppl']:.1f} | ~3.78 GPU-h reales "
        "(tope 10; 1 configuracion de 2). Interrupciones del servidor "
        "reanudadas por el supervisor detached; sin perdida.\n"
        "\n"
        "**Downstream (recetas congeladas D98-1; umbrales V03; resultados "
        "REGISTRADOS, no reejecutados)**\n"
        "\n"
        "| Particion | Regla | corrected_cv | delta vs V03 | delta vs 30M | "
        "folds | F1 HS/TR/AG | salv/dam/net | IC95 vs V03 |\n"
        "|---|---|---|---|---|---|---|---|---|\n"
        f"| Canonica | V03 | {rc['v03']['corrected_cv']:.6f} | 0 | "
        f"{rc['v03']['corrected_cv'] - rc['fusion_30M']['corrected_cv']:+.6f} "
        f"| - | {f1e(rc['v03'])} | - | - |\n"
        f"| Canonica | 30M (C98-1) | "
        f"{rc['fusion_30M']['corrected_cv']:.6f} | "
        f"{rc['fusion_30M']['corrected_cv'] - rc['v03']['corrected_cv']:+.6f} "
        f"| 0 | 4/5 | "
        f"{f1e(rc['fusion_30M'])} | {corr(rc['fusion_30M'])} | "
        f"[{ic30_can[0]:+.6f},{ic30_can[1]:+.6f}] |\n"
        f"| Canonica | B1 | {rc['fusion_b1']['corrected_cv']:.6f} | "
        f"{cc['delta_b1_vs_v03']:+.6f} | {cc['delta_b1_vs_30']:+.6f} | "
        f"{cc['pliegues_b1_mejorados_vs_v03']}/5 | {f1e(rc['fusion_b1'])} | "
        f"{corr(rc['fusion_b1'])} | "
        f"[{cc['bootstrap_b1_vs_v03']['ic95'][0]:+.6f},"
        f"{cc['bootstrap_b1_vs_v03']['ic95'][1]:+.6f}] |\n"
        f"| Particion 47 | V03eq | "
        f"{rp['v03_equivalente_47']['corrected_cv']:.6f} | 0 | "
        f"{rp['v03_equivalente_47']['corrected_cv'] - rp['fusion_30M_47']['corrected_cv']:+.6f}"
        f" | - | {f1e(rp['v03_equivalente_47'])} | - | - |\n"
        f"| Particion 47 | 30M (C98-1) | "
        f"{rp['fusion_30M_47']['corrected_cv']:.6f} | "
        f"{rp['fusion_30M_47']['corrected_cv'] - rp['v03_equivalente_47']['corrected_cv']:+.6f}"
        f" | 0 | 2/5 | "
        f"{f1e(rp['fusion_30M_47'])} | {corr(rp['fusion_30M_47'])} | "
        f"[{ic30_47[0]:+.6f},{ic30_47[1]:+.6f}] |\n"
        f"| Particion 47 | B1 | {rp['fusion_b1_47']['corrected_cv']:.6f} | "
        f"{cp['delta_b1_vs_v03']:+.6f} | {cp['delta_b1_vs_30']:+.6f} | "
        f"{cp['pliegues_b1_mejorados_vs_v03']}/5 | {f1e(rp['fusion_b1_47'])} "
        f"| {corr(rp['fusion_b1_47'])} | "
        f"[{cp['bootstrap_b1_vs_v03']['ic95'][0]:+.6f},"
        f"{cp['bootstrap_b1_vs_v03']['ic95'][1]:+.6f}] |\n"
        "\n"
        "Controles de reproduccion obligatorios: 30M canonica "
        f"{can['control_reproduccion_30M']['valor']:.10f} (esperado "
        f"{can['control_reproduccion_30M']['esperado']}) y particion 47 "
        f"{p47['control_reproduccion_30M_47']['valor']:.10f} (esperado "
        f"{p47['control_reproduccion_30M_47']['esperado']}): EXACTOS. "
        "Fold scores B1: canonica [" + folds(rc['fusion_b1']) + "]; 47 ["
        + folds(rp['fusion_b1_47']) + "].\n"
        "\n"
        "**Complementariedad: NO robusta**. B1 tuvo "
        f"{cc['pliegues_b1_mejorados_vs_v03']}/5 pliegues positivos en la "
        f"canonica y {cp['pliegues_b1_mejorados_vs_v03']}/5 en la particion "
        "47; neto vs V03 0 y -1. Los slices positivos de la canonica "
        "(long-q4 +0.0197, hashtag +0.0140, url +0.0093) NO se replican en "
        "la 47 (url -0.0011, hashtag -0.0016; long-q4 +0.0138); q1 cortos "
        "negativo en ambas (-0.0098 / -0.0047).\n"
        "\n"
        "**Resultado secundario (conservado sin adopcion ni reapertura)**: "
        "embedding-only B1 0.660174 > embedding-only 30M 0.652238 en la "
        "canonica (+0.0079); la fusion (metrica del gate) no lo traduce.\n"
        "\n"
        "**GATE_FAILED** (pre-registro seccion 8, sin modificaciones): "
        "fallan los criterios 1 (delta >= +0.003), 2 (>= 4/5 pliegues), 3 "
        "(neto positivo), 4 (particion 47 positiva y no marginal) y 6 "
        "(atribucion: delta vs 30M > 0); cumple 5 (sin defectos). "
        "**H100-01 FALSIFIED -> P100 CLOSED**.\n"
        "\n"
        "**Lectura cientifica (verbatim del propietario)**: la sustitucion "
        "de parte del corpus general por dialogos conversacionales, "
        "manteniendo la escala y la arquitectura, no mejoro de forma "
        "consistente la fusion con V03. La variante B1 fue inferior al "
        "control contextual de 30M en ambas particiones y no supero V03. El "
        "resultado negativo falsifica la hipotesis de composicion evaluada; "
        "no demuestra que toda fuente conversacional sea inutil en cualquier "
        "arquitectura.\n"
        "\n"
        f"**Artefactos**: p100_b1_pretraining.json ({h['p100_b1_pretraining.json']}), "
        f"p100_b1_downstream_canonica.json ({h['p100_b1_downstream_canonica.json']}), "
        f"p100_b1_downstream_part47.json ({h['p100_b1_downstream_part47.json']}), "
        "p100_b1_downstream_results.csv, p100_b1_complementarity.csv, "
        "p100_b1_final_report.md, artifacts/p100_b1_manifest.json, "
        "features/p100_b1_reps_dev.npz, modelos p100_b1_{ltr,rtl}.pt; filas "
        "P100-00/10/11/12/13/14 (total 491).")


def validar(cells: list) -> None:
    """Compila las celdas de codigo y exige ASCII puro en lo generado."""
    for indice, celda in enumerate(cells):
        fuente = "".join(celda.get("source", []))
        if celda["cell_type"] == "code":
            compile(fuente, f"<celda {indice}>", "exec")
    if any(ord(c) > 127 for c in CODIGO):
        raise SystemExit("Byte > 127 en el texto generado (codigo)")
    if any(b > 127 for b in Path(__file__).read_bytes()):
        raise SystemExit("El propio script contiene bytes > 127")


def ejecutar_verificacion() -> None:
    """Ejecuta la celda de verificacion en aislamiento."""
    espacio: dict = {"PROJECT_ROOT": RAIZ, "__name__": "__verificacion__"}
    exec(compile(CODIGO, "<seccion 163>", "exec"), espacio)  # noqa: S102


def main() -> None:
    """Integra la seccion 163, ejecuta su verificacion y guarda el notebook."""
    seccion = construir_seccion()
    if any(ord(c) > 127 for c in seccion):
        raise SystemExit("Byte > 127 en el texto generado (seccion)")
    ejecutar_verificacion()
    notebook = json.loads(RUTA_NOTEBOOK.read_text(encoding="utf-8"))
    cells = notebook["cells"]
    celdas_antes = len(cells)
    while any("".join(c.get("source", [])).startswith(MARCA_SECCION)
              for c in cells):
        cells.pop(next(i for i, c in enumerate(cells)
                       if "".join(c.get("source", [])).startswith(
                           MARCA_SECCION)))
    while any(MARCA_CODIGO in "".join(c.get("source", [])) for c in cells):
        cells.pop(next(i for i, c in enumerate(cells)
                       if MARCA_CODIGO in "".join(c.get("source", []))))
    cells.append(nueva_celda_markdown(seccion))
    cells.append(nueva_celda_codigo(CODIGO))
    validar(cells)
    notebook["cells"] = cells
    RUTA_NOTEBOOK.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8")
    print("Notebook actualizado con la seccion 163:")
    print(f"  celdas antes: {celdas_antes} | despues: {len(cells)}")
    print("  verificacion de la fase en aislamiento: OK")
    print("  celdas de codigo compiladas: OK | ASCII puro en lo generado: OK")


if __name__ == "__main__":
    main()
