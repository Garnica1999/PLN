"""P98-97: integra la seccion 160 (P98 FASE B: C98-1 + D98-1).

Idempotente y ASCII puro; celda de verificacion ejecutada en aislamiento.

Uso:
    /home/carlos/venv/bin/python experiments/p98_97_integrar_notebook_c98_1_d98_1.py
"""

from __future__ import annotations

import hashlib
import json
import uuid
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
RUTA_NOTEBOOK = RAIZ / "notebooks" / "10_final_pipeline.ipynb"
MARCA_SECCION = "## 160. P98 - FASE B"
MARCA_CODIGO = "Verificacion de la seccion 160 (P98"


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


CODIGO = '''"""Verificacion de la seccion 160 (P98 Fase B: C98-1 y D98-1).

No entrena nada; solo lee artefactos.
"""

import csv
import hashlib
import json
from pathlib import Path

R = PROJECT_ROOT / "reports"
pre = json.load(open(R / "p98_c98_1_pretraining.json", encoding="utf-8"))
assert pre["checkpoints"]["ltr"]["epocas"] == 4
assert pre["checkpoints"]["rtl"]["epocas"] == 4
assert pre["gpu_horas_totales_c98_1"] < 6.0
assert pre["checkpoints"]["ltr"]["val_ppl"] < 100
assert pre["checkpoints"]["rtl"]["val_ppl"] < 100

d98 = json.load(open(R / "p98_13_downstream.json", encoding="utf-8"))
assert d98["veredicto"] == "MARGINAL_SENAL"
comp = d98["comparaciones"]["fusion_vs_v03"]
assert 0.004 < comp["delta"] < 0.006 and comp["pliegues_mejorados"] == 4
assert comp["bootstrap_clusters"]["ic95"][0] < 0 < comp["bootstrap_clusters"]["ic95"][1]

filas = list(csv.DictReader(open(PROJECT_ROOT / "experiments.csv",
                                 encoding="utf-8-sig")))
ids = {f["experiment_id"] for f in filas}
assert {"P98-12", "P98-13", "P98-14"} <= ids and len(filas) == 479

plan = (R / "p98_plan_seguimiento.md").read_text(encoding="utf-8")
assert "MARGINAL_SENAL" in plan and "HECHO: LTR ppl 94.8" in plan

fuentes = {
    "features/oof_v03_final.npz": "559265816ce4f631",
    "data/raw/toxic_train.csv": "3d40e7345b588860",
}
for ruta, pref in fuentes.items():
    h = hashlib.sha256((PROJECT_ROOT / ruta).read_bytes()).hexdigest()
    assert h.startswith(pref), f"fuente alterada: {ruta}"

for nombre in ("p98_c98_1_pretraining.json", "p98_c98_1_pretraining.md",
               "p98_13_downstream.json", "p98_13_downstream.md",
               "p98_downstream_results.csv", "p98_complementarity.csv"):
    datos = (R / nombre).read_bytes()
    assert all(b <= 127 for b in datos), f"byte > 127 en {nombre}"

print("P98 Fase B (C98-1 + D98-1) verificada:")
print("  C98-1: LTR ppl 94.8 / RTL ppl 91.7 (4+4 epocas, 3.775 GPU-h)")
print("  D98-1: fusion 0.782266 (delta +0.005055, 4/5 pliegues, IC incluye 0)")
print("  veredicto: MARGINAL_SENAL -> siguiente: D98-1b (replica) / D98-2")'''


def construir_seccion() -> str:
    """Construye el markdown de la seccion 160 con metricas reales."""
    pre = json.loads((RAIZ / "reports" / "p98_c98_1_pretraining.json").read_text(
        encoding="utf-8"))
    d98 = json.loads((RAIZ / "reports" / "p98_13_downstream.json").read_text(
        encoding="utf-8"))
    r = d98["reglas"]["fusion_v51_style"]
    comp = d98["comparaciones"]
    h = {n: sha16(RAIZ / "reports" / n) for n in (
        "p98_c98_1_pretraining.json", "p98_13_downstream.json")}
    return (
        "## 160. P98 - FASE B: C98-1 (PREENTRENAMIENTO) Y D98-1 (DOWNSTREAM) "
        "(2026-10-13)\n"
        "\n"
        "**C98-1**: LTR y RTL entrenados 4 epocas cada uno sobre los 29,3M "
        "tokens de C98-1 (hidden 1024 x 2; tope por escalera).\n"
        "\n"
        f"- LTR: val ppl {pre['checkpoints']['ltr']['val_ppl']:.1f} "
        f"(185.0 -> 126.4 -> 105.6 -> 94.8); RTL: val ppl "
        f"{pre['checkpoints']['rtl']['val_ppl']:.1f} "
        "(180.4 -> 123.4 -> 102.2 -> 91.7); ~17,3k tok/s; "
        f"**{pre['gpu_horas_totales_c98_1']} GPU-h** (tope 6).\n"
        "- Checkpoints: `models/p98_c98_1_ltr.pt` (sha "
        f"{pre['checkpoints']['ltr']['sha256'][:16]}) y "
        f"`models/p98_c98_1_rtl.pt` (sha "
        f"{pre['checkpoints']['rtl']['sha256'][:16]}).\n"
        "\n"
        "**D98-1** (reps congeladas 2048-d, media sobre tokens; umbrales V03):\n"
        "\n"
        "| Regla | corrected_cv | delta vs V03 | F1 HS/TR/AG |\n|---|---|---|---|\n"
        f"| V03 frozen | {d98['reglas']['v03_frozen']['corrected_cv']:.10f} | - | "
        "0.7617/0.8157/0.7542 |\n"
        f"| embedding-only | "
        f"{d98['reglas']['embedding_only']['corrected_cv']:.10f} | "
        f"{d98['comparaciones']['embedding_only_vs_v03']['delta']:+.6f} | "
        "(0.6523 de referencia estatica v5.1) |\n"
        f"| fusion v5.1-style | {r['corrected_cv']:.10f} | "
        f"{comp['fusion_vs_v03']['delta']:+.6f} | "
        f"{r['f1_por_etiqueta']['HS']:.4f}/{r['f1_por_etiqueta']['TR']:.4f}/"
        f"{r['f1_por_etiqueta']['AG']:.4f} |\n"
        "\n"
        f"- Fusion vs V03: {comp['fusion_vs_v03']['pliegues_mejorados']}/5 "
        f"pliegues; IC95 clusters "
        f"{comp['fusion_vs_v03']['bootstrap_clusters']['ic95']} (incluye cero; "
        f"85.4% de remuestreos positivos); neto +11 celdas (103/92).\n"
        "- Patron por slices: longitud_q4 (tuits largos) +0.0271 (neto +17); "
        "hashtag +0.0125; url +0.0118; mencion +0.0059; el unico slice "
        "negativo es longitud_q1 (cortos) -0.0061 (neto -10).\n"
        "\n"
        "**Veredicto: MARGINAL_SENAL** (delta nominal sobre el gate +0.003 y "
        "4/5 pliegues, pero el IC incluye cero: mejora NO demostrada) y la "
        "fusion contextual queda +0.0005 sobre la fusion estatica v5.1 "
        "(0.782266 vs 0.7818), dentro de la clase historica de +0.005 que nunca "
        "transfirio al leaderboard.\n"
        "\n"
        f"**Artefactos**: p98_c98_1_pretraining.json ({h['p98_c98_1_pretraining.json']}), "
        f"p98_13_downstream.json ({h['p98_13_downstream.json']}), "
        "p98_pretraining_results.csv, p98_downstream_results.csv, "
        "p98_complementarity.csv, features/p98_13_reps_dev.npz, "
        "features/p98_13_oof.npz; filas P98-12/13/14 (total 479).\n"
        "\n"
        "> Proximo (dentro del presupuesto): D98-1b = replica de robustez en una "
        "segunda particion agrupada (decision sobre D98-2 despues de ella).")


def validar(cells: list[dict]) -> None:
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
    espacio: dict[str, object] = {"PROJECT_ROOT": RAIZ, "__name__": "__verificacion__"}
    exec(compile(CODIGO, "<seccion 160>", "exec"), espacio)


def main() -> None:
    """Integra la seccion 160, ejecuta su verificacion y guarda el notebook."""
    seccion = construir_seccion()
    if any(ord(c) > 127 for c in seccion):
        raise SystemExit("Byte > 127 en el texto generado (seccion)")
    ejecutar_verificacion()
    notebook = json.loads(RUTA_NOTEBOOK.read_text(encoding="utf-8"))
    cells = notebook["cells"]
    celdas_antes = len(cells)
    while any("".join(c.get("source", [])).startswith(MARCA_SECCION) for c in cells):
        cells.pop(next(i for i, c in enumerate(cells)
                       if "".join(c.get("source", [])).startswith(MARCA_SECCION)))
    while any(MARCA_CODIGO in "".join(c.get("source", [])) for c in cells):
        cells.pop(next(i for i, c in enumerate(cells)
                       if MARCA_CODIGO in "".join(c.get("source", []))))
    cells.append(nueva_celda_markdown(seccion))
    cells.append(nueva_celda_codigo(CODIGO))
    validar(cells)
    notebook["cells"] = cells
    RUTA_NOTEBOOK.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
                             encoding="utf-8")
    print("Notebook actualizado con la seccion 160:")
    print(f"  celdas antes: {celdas_antes} | despues: {len(cells)}")
    print("  verificacion de la fase en aislamiento: OK")
    print("  celdas de codigo compiladas: OK | ASCII puro en lo generado: OK")


if __name__ == "__main__":
    main()
