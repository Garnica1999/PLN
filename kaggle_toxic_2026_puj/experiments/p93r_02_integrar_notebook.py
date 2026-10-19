"""P93R-02: integra la seccion 149 (P93-R ADJUDICACION CIEGA) en el notebook.

Lee los entregables `reports/p93r_*` y el paquete de adjudicacion y construye una
subseccion compacta. Idempotente y ASCII puro. La celda de verificacion se ejecuta de
forma aislada (solo lee el proyecto).

Uso:
    /home/carlos/venv/bin/python experiments/p93r_02_integrar_notebook.py
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
RUTA_NOTEBOOK = RAIZ / "notebooks" / "10_final_pipeline.ipynb"
MARCA_SECCION = "## 149. P93-R - PAQUETE DE ADJUDICACION CIEGA"
MARCA_CODIGO = "Verificacion de la seccion 149 (P93-R"


def nueva_celda_markdown(texto: str) -> dict:
    """Crea una celda de markdown con identificador unico."""
    return {"cell_type": "markdown", "id": uuid.uuid4().hex[:8], "metadata": {},
            "source": texto.splitlines(keepends=True)}


def nueva_celda_codigo(texto: str) -> dict:
    """Crea una celda de codigo vacia de salidas y sin ejecutar."""
    return {"cell_type": "code", "id": uuid.uuid4().hex[:8], "metadata": {},
            "execution_count": None, "outputs": [], "source": texto.splitlines(keepends=True)}


CODIGO = '''"""Verificacion de la seccion 149 (P93-R, adjudicacion ciega preparada).

Comprueba el paquete de ronda 1, la clave interna, la plantilla de ronda 2 y el
cegamiento. No entrena nada.
"""

import csv
import json
from pathlib import Path

pkg = PROJECT_ROOT / "artifacts" / "p93_adjudication_v1" / "independent_review"
for nombre in ("adjudicator_C.csv", "annotation_guidelines_v2_1.md",
               "README_adjudicator.md", "manifest.json"):
    assert (pkg / nombre).exists(), f"falta {nombre}"
for nombre in ("p93r_adjudication_plan.md", "p93r_blinding_audit.md",
               "p93r_adjudication_manifest.json", "p93r_internal_adjudication_key.csv",
               "p93r_round2_comparison_template.csv"):
    assert (PROJECT_ROOT / "reports" / nombre).exists(), f"falta {nombre}"

with (pkg / "adjudicator_C.csv").open(encoding="utf-8", newline="") as fh:
    c = list(csv.DictReader(fh))
assert len(c) == 127
assert all(f["adjudication_review_id"].startswith("P93R-") for f in c)
assert all(f[x] == "" for f in c
           for x in ("HS_annotation", "TR_annotation", "AG_annotation",
                     "uncertainty_notes", "tweet_note"))
assert "gold" not in " ".join(c[0].keys()).lower()

guia = (PROJECT_ROOT / "reports" / "p92_annotation_guidelines_v2_1_holdout_safe.md").read_bytes()
assert (pkg / "annotation_guidelines_v2_1.md").read_bytes() == guia

m = json.loads((pkg / "manifest.json").read_text(encoding="utf-8"))
assert m["counts"]["unique_tweets"] == 127
assert m["counts"]["candidate_cells"] == 176
assert m["distribution"] == "NOT_DISTRIBUTED"

mg = json.loads((PROJECT_ROOT / "reports" / "p93r_adjudication_manifest.json").read_text(
    encoding="utf-8"))
assert mg["status"] == "ADJUDICATION_PACKAGE_READY"
assert all(mg["checks"].values()), "checks con fallos"

with (PROJECT_ROOT / "reports" / "p93r_internal_adjudication_key.csv").open(
        encoding="utf-8", newline="") as fh:
    clave = list(csv.DictReader(fh))
assert len(clave) == 127
with (PROJECT_ROOT / "reports" / "p93r_round2_comparison_template.csv").open(
        encoding="utf-8", newline="") as fh:
    r2 = list(csv.DictReader(fh))
assert len(r2) == 176
assert all(f["initial_independent_decision"] == "" and f["post_comparison_decision"] == ""
           for f in r2)

sucios = [n for n in ("p93r_adjudication_plan.md", "p93r_blinding_audit.md",
                      "p93r_adjudication_manifest.json")
          if any(x > 127 for x in (PROJECT_ROOT / "reports" / n).read_bytes())]
assert not sucios, f"bytes > 127 en {sucios}"

print("P93-R (adjudicacion ciega preparada) verificado:")
print("  127 tweets unicos (176 celdas) | anotaciones vacias | sin gold ni A/B")
print("  guia v2.1 identica | clave interna aislada | plantilla ronda 2 sin decisiones")'''


def construir_seccion() -> str:
    """Construye el markdown de la seccion 149."""
    m = json.loads((RAIZ / "reports" / "p93r_adjudication_manifest.json").read_text(
        encoding="utf-8"))
    c = m["counts"]
    return f"""## 149. P93-R - PAQUETE DE ADJUDICACION CIEGA PREPARADO (2026-10-13)

**Fase de preparacion; sin entrenamiento, sin predicciones, sin cambios de etiquetas.** Se preparo la ronda 1 de adjudicacion completamente independiente para las {c['candidate_label_cells']} celdas candidatas de P93: **{c['unique_tweets_to_review']} tweets unicos** (agrupados; un tweet puede aportar varias celdas), con las tres etiquetas a anotar y anotaciones vacias.

**Paquete** (`artifacts/p93_adjudication_v1/independent_review/`): `adjudicator_C.csv` (ids opacos P93R-*, orden aleatorizado con semilla {m['seeds']['orden_filas']}, ids con semilla {m['seeds']['asignacion_ids']}), guia v2.1 (copia byte a byte), README del revisor y manifiesto. **Cegamiento verificado** (15/15): sin gold, sin respuestas ni notas de A/B, sin predicciones/OOF/umbrales/folds, sin etiquetas candidatas, sin motivos de seleccion, sin componente ni grupos; no se revela por que fueron seleccionados.

**Internos**: clave de adjudicacion (`p93r_internal_adjudication_key.csv`, 127 filas) y plantilla de ronda 2 (`p93r_round2_comparison_template.csv`, 176 filas con A/B prellenados y **sin decisiones de C** hasta que existan respuestas reales). El gold permanece oculto a C en ambas rondas; la comparacion final con el gold sera una fase separada con estados de adjudicacion (no errores confirmados).

**Estado**: `ADJUDICATION_PACKAGE_READY` + `THIRD_REVIEW_PENDING` + `SOURCE_LABELS_UNCHANGED` + `CALIBRATION_V2_1 = PENDING` + `V03 = FROZEN`. El paquete NO se distribuye automaticamente.

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
    exec(compile(CODIGO, "<seccion 149>", "exec"), espacio)


def main() -> None:
    """Integra la seccion 149, ejecuta su verificacion y guarda el notebook."""
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
    print("Notebook actualizado con la seccion 149:")
    print(f"  celdas antes: {celdas_antes} | despues: {len(cells)}")
    print("  adjudicacion P93-R verificada en aislamiento: OK")
    print("  celdas de codigo compiladas: OK | ASCII puro en lo generado: OK (0 bytes > 127)")


if __name__ == "__main__":
    main()
