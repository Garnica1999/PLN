"""P92-19: integra la seccion 146 (P92-Y CALIBRACION v2.1) en el notebook.

Lee los entregables `reports/p92y_*` y el handoff de calibracion y construye una
subseccion compacta. Idempotente y ASCII puro. La celda de verificacion se ejecuta de
forma aislada (solo lee el proyecto).

Uso:
    /home/carlos/venv/bin/python experiments/p92_19_integrar_notebook.py
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
RUTA_NOTEBOOK = RAIZ / "notebooks" / "10_final_pipeline.ipynb"
MARCA_SECCION = "## 146. P92-Y - ENTREGAS DE CALIBRACION v2.1"
MARCA_CODIGO = "Verificacion de la seccion 146 (P92-Y"


def nueva_celda_markdown(texto: str) -> dict:
    """Crea una celda de markdown con identificador unico."""
    return {"cell_type": "markdown", "id": uuid.uuid4().hex[:8], "metadata": {},
            "source": texto.splitlines(keepends=True)}


def nueva_celda_codigo(texto: str) -> dict:
    """Crea una celda de codigo vacia de salidas y sin ejecutar."""
    return {"cell_type": "code", "id": uuid.uuid4().hex[:8], "metadata": {},
            "execution_count": None, "outputs": [], "source": texto.splitlines(keepends=True)}


CODIGO = '''"""Verificacion de la seccion 146 (P92-Y, calibracion v2.1 preparada).

Comprueba el handoff de calibracion, la guia v2.1 y la preservacion del paquete
principal. No entrena nada.
"""

import csv
import hashlib
import json
from pathlib import Path

entregables = ["p92y_calibration_handoff_report.md", "p92y_calibration_handoff_manifest.json"]
for nombre in entregables:
    assert (PROJECT_ROOT / "reports" / nombre).exists(), f"falta {nombre}"
handoff = PROJECT_ROOT / "artifacts" / "human_annotation_v2_1_holdout_safe" / "calibration_handoff"
for letra in ("A", "B"):
    for nombre in ("calibration_sample.csv", "README_annotator.md",
                   "annotation_guidelines_v2_1.md"):
        assert (handoff / f"annotator_{letra}" / nombre).exists(), f"falta {letra}/{nombre}"
assert (handoff / "README.md").exists()

with (handoff / "annotator_A" / "calibration_sample.csv").open(
        encoding="utf-8", newline="") as fh:
    a = list(csv.DictReader(fh))
with (handoff / "annotator_B" / "calibration_sample.csv").open(
        encoding="utf-8", newline="") as fh:
    b = list(csv.DictReader(fh))
assert len(a) == 15 and len(b) == 15
assert {f["review_id"] for f in a} == {f["review_id"] for f in b}
assert [f["review_id"] for f in a] != [f["review_id"] for f in b]
assert all(f[c] == "" for f in a + b
           for c in ("HS_annotation", "TR_annotation", "AG_annotation",
                     "uncertainty_notes", "tweet_note"))

with (PROJECT_ROOT / "artifacts" / "human_annotation_v2_1_holdout_safe" / "calibration"
      / "calibration_sample.csv").open(encoding="utf-8", newline="") as fh:
    comun = {f["review_id"]: f["text"] for f in csv.DictReader(fh)}
assert all(comun[f["review_id"]] == f["text"] for f in a + b), "textos divergen del comun"

guia = (PROJECT_ROOT / "reports" / "p92_annotation_guidelines_v2_1_holdout_safe.md").read_bytes()
for letra in ("A", "B"):
    assert (handoff / f"annotator_{letra}" / "annotation_guidelines_v2_1.md").read_bytes() == guia

m = json.loads((PROJECT_ROOT / "reports" / "p92y_calibration_handoff_manifest.json").read_text(
    encoding="utf-8"))
assert m["distribution"] == "NOT_DISTRIBUTED"
assert m["calibration_sample"]["tweets"] == 15

inf = (PROJECT_ROOT / "reports" / "p92y_calibration_handoff_report.md").read_text(encoding="utf-8")
assert "CALIBRATION_V2_1_PACKAGE_READY" in inf
assert "HUMAN_CALIBRATION_PENDING" in inf

hh = hashlib.sha256(
    (PROJECT_ROOT / "artifacts" / "human_annotation_v2_1_holdout_safe" / "manifest.json")
    .read_bytes()).hexdigest()
assert hh.startswith("00998e6006035c9f"), "el manifiesto del paquete cambio"

for nombre, prefijo in (("annotator_A.csv", "7dfd5b898d3905f2"),
                        ("annotator_B.csv", "956646fa479f0c65")):
    hv = hashlib.sha256(
        (PROJECT_ROOT / "artifacts" / "human_annotation_v1" / nombre).read_bytes()).hexdigest()
    assert hv.startswith(prefijo), f"v1 {nombre} cambio"

sucios = [n for n in entregables
          if any(x > 127 for x in (PROJECT_ROOT / "reports" / n).read_bytes())]
assert not sucios, f"bytes > 127 en {sucios}"

print("P92-Y (calibracion v2.1 preparada) verificado:")
print("  handoffs A/B: 15 tweets, misma membresia, orden distinto, vacios | guia v2.1 identica")
print("  paquete principal intacto | NOT_DISTRIBUTED | HUMAN_CALIBRATION_PENDING")'''


def construir_seccion() -> str:
    """Construye el markdown de la seccion 146."""
    m = json.loads((RAIZ / "reports" / "p92y_calibration_handoff_manifest.json").read_text(
        encoding="utf-8"))
    return f"""## 146. P92-Y - ENTREGAS DE CALIBRACION v2.1 PREPARADAS (2026-10-13)

**Fase de preparacion; sin entrenamiento, sin predicciones, sin distribucion, sin anotaciones simuladas.** La calibracion comun de 15 tweets elegibles se reutilizo **sin regenerar** (no se re-selecciono nada) y se prepararon dos copias ciegas en `calibration_handoff/annotator_A/` y `annotator_B/`: mismos 15 review_id (`P92W-C01..C15`), **orden distinto** (semillas A={m['calibration_sample']['orders_seeds']['A']}, B={m['calibration_sample']['orders_seeds']['B']}), respuestas vacias, con README propio y la guia v2.1 (copia byte a byte; sha256 completo `{m['guide']['sha256_completo'][:16]}`).

**Controles (22/22)**: 15/15 ids en dev2 sin solapes con test/holdout/holdout2 ni con los componentes principales A/B; textos identicos a la fuente (15/15 calibracion; 30/30 handoffs); cegamiento sin fugas en archivos visibles; clave interna fuera; guia v2.1 identica en ambos; manifiesto del paquete principal intacto (`00998e60...`); verificador ASCII oficial OK.

**Instrucciones de entrega** (en el informe): entregar cada subcarpeta por separado; las dos personas completan los 15 tweets de forma independiente sin comentarlos entre si hasta entregar; devuelven `calibration_response_A.csv` / `_B.csv`; la ronda evalua la claridad de la guia (no es estimacion poblacional); v2.1 es politica operativa interna, no definicion oficial.

**Estado**: `CALIBRATION_V2_1_PACKAGE_READY` + `HUMAN_CALIBRATION_PENDING` + `MAIN_ANNOTATION_NOT_DISTRIBUTED` + `V03 = FROZEN` + `NO NEW MODELING PHASE`. La muestra principal de 250 tweets solo se distribuira tras recibir ambas calibraciones y decidir si la guia se mantiene congelada.

> V03 remains the best verified leaderboard result."""


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
    """Ejecuta la celda de verificacion de forma aislada (solo lee el proyecto)."""
    espacio: dict[str, object] = {"PROJECT_ROOT": RAIZ, "__name__": "__verificacion__"}
    exec(compile(CODIGO, "<seccion 146>", "exec"), espacio)


def main() -> None:
    """Integra la seccion 146, ejecuta su verificacion y guarda el notebook."""
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
    print("Notebook actualizado con la seccion 146:")
    print(f"  celdas antes: {celdas_antes} | despues: {len(cells)}")
    print("  handoff de calibracion verificado en aislamiento: OK")
    print("  celdas de codigo compiladas: OK | ASCII puro en lo generado: OK (0 bytes > 127)")


if __name__ == "__main__":
    main()
