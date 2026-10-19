"""P92-03: integra la seccion 139 (P92 HUMAN LABEL REVIEW) en el notebook.

Lee los entregables `reports/p92_*` y `artifacts/human_annotation_v1/` y construye una
subseccion compacta. Idempotente y ASCII puro. La celda de verificacion se ejecuta de
forma aislada (solo lee archivos del proyecto).

Uso:
    /home/carlos/venv/bin/python experiments/p92_03_integrar_notebook.py
"""

from __future__ import annotations

import csv
import json
import uuid
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
RUTA_NOTEBOOK = RAIZ / "notebooks" / "10_final_pipeline.ipynb"
RUTA_RESULTS = RAIZ / "reports" / "p92_audit_results.json"
MARCA_SECCION = "## 139. P92 - HUMAN LABEL REVIEW"
MARCA_CODIGO = "Verificacion de la seccion 139 (P92"


def nueva_celda_markdown(texto: str) -> dict:
    """Crea una celda de markdown con identificador unico."""
    return {"cell_type": "markdown", "id": uuid.uuid4().hex[:8], "metadata": {},
            "source": texto.splitlines(keepends=True)}


def nueva_celda_codigo(texto: str) -> dict:
    """Crea una celda de codigo vacia de salidas y sin ejecutar."""
    return {"cell_type": "code", "id": uuid.uuid4().hex[:8], "metadata": {},
            "execution_count": None, "outputs": [], "source": texto.splitlines(keepends=True)}


CODIGO = '''"""Verificacion de la seccion 139 (P92, paquete de revision humana).

Comprueba los entregables, las verificaciones del paquete, la ausencia de anotaciones
simuladas y el escaneo byte a byte. No entrena nada.
"""

import csv
import json
from pathlib import Path

entregables = [
    "p92_human_annotation_plan.md", "p92_annotation_guidelines_v1.md",
    "p92_sampling_manifest.json", "p92_annotation_readiness_audit.md",
    "p92_internal_sampling_key.csv", "p92_audit_results.json",
]
for nombre in entregables:
    assert (PROJECT_ROOT / "reports" / nombre).exists(), f"falta {nombre}"
paquete = ["annotator_A.csv", "annotator_B.csv", "calibration_sample.csv",
           "adjudication_template.csv", "README_annotators.md", "manifest.json"]
for nombre in paquete:
    assert (PROJECT_ROOT / "artifacts" / "human_annotation_v1" / nombre).exists(), f"falta {nombre}"

d = json.loads((PROJECT_ROOT / "reports" / "p92_audit_results.json").read_text(encoding="utf-8"))
assert d["principal"]["tweets"] == 250
assert d["componente_A"]["tweets"] == 100 and d["componente_B"]["tweets"] == 150
assert d["calibracion"]["tweets"] == 15
assert all(d["checks"].values()), "checks del paquete con fallos"

with (PROJECT_ROOT / "artifacts" / "human_annotation_v1" / "annotator_A.csv").open(encoding="utf-8", newline="") as fh:
    a = list(csv.DictReader(fh))
with (PROJECT_ROOT / "artifacts" / "human_annotation_v1" / "annotator_B.csv").open(encoding="utf-8", newline="") as fh:
    b = list(csv.DictReader(fh))
assert len(a) == 250 and len(b) == 250
assert all(f[c] == "" for f in a + b for c in ("HS_annotation", "TR_annotation", "AG_annotation"))
assert [f["review_id"] for f in a] != [f["review_id"] for f in b]

sucios = [n for n in entregables if any(x > 127 for x in (PROJECT_ROOT / "reports" / n).read_bytes())]
assert not sucios, f"bytes > 127 en {sucios}"

print("P92 (paquete de revision humana) verificado:")
print("  250 tweets principales (A=100 + B=150) + 15 calibracion | anotaciones vacias")
print("  14/14 checks del paquete | 0 bytes > 127 | HUMAN_ANNOTATION_PENDING")'''


def construir_seccion() -> str:
    """Construye el markdown de la seccion 139 a partir de los entregables."""
    d = json.loads(RUTA_RESULTS.read_text(encoding="utf-8"))
    a = d["componente_A"]
    b = d["componente_B"]
    cal = d["calibracion"]
    prin = d["principal"]
    return f"""## 139. P92 - HUMAN LABEL REVIEW: paquete de anotacion ciega preparado (2026-10-09)

**Fase de preparacion; la anotacion humana NO se ha ejecutado.** Sin entrenamiento, sin modificacion de etiquetas, sin test/holdout, sin submissions. La anotacion comienza cuando los archivos se entreguen a dos personas reales.

**Motivacion**: P91 concluyo `NO_CONFIRMED_LABEL_ISSUE` + `ANNOTATION_POLICY_UNRESOLVED` + `GOVERNANCE_REVIEW_REQUIRED` y dejo protocolo y cola; P92 construye el paquete ciego reproducible. Definiciones oficiales (unica fuente normativa): HS "discurso de odio"; TR "dirigido contra una persona o grupo"; AG "texto agresivo". Ambiguedades declaradas (no resueltas): criterios operativos de HS y AG ausentes; relacion HS/TR sin regla formal; casos limite sin instruccion.

**Muestreo** (semillas fijas; clave interna fuera del paquete): **A** = {a['tweets']} tweets de referencia (recuperados de P91 y verificados contra el RNG exacto, semilla {a['semilla']}); **B** = {b['tweets']} enriquecidos de {b['disponibles_enriquecidos']} disponibles (PILI {b['seleccionados_por_tier']['pili']} / DRC {b['seleccionados_por_tier']['drc']} / HCD {b['seleccionados_por_tier']['hcd']} / LSM {b['seleccionados_por_tier']['lsm']}; cap {b['cap_grupo']} por grupo; A y B disjuntos; DRC = identificador heredado del criterio de seleccion, EMPIRICAL_STRUCTURE_ANOMALY, no violacion normativa); **calibracion** = {cal['tweets']} tweets. Principal: **{prin['tweets']} tweets / {prin['celdas']} anotaciones de etiqueta por anotador (3 respuestas posibles por anotacion)**.

**Paquete** (`artifacts/human_annotation_v1/`): `annotator_A.csv` y `annotator_B.csv` (mismos casos, orden independiente), `calibration_sample.csv`, `adjudication_template.csv` ({prin['celdas']} filas sin decisiones), `README_annotators.md` y `manifest.json`. `review_id` opacos (P92R-####); ocultos ids, etiquetas, probabilidades, predicciones, folds, componente y motivos. Anotaciones vacias (sin respuestas simuladas).

**Acuerdo y adjudicacion (plan, no ejecutado)**: por etiqueta acuerdo bruto, positivo/negativo, kappa/alpha, matrices, tasa UNABLE_TO_DETERMINE; adjudicacion separada con tercero cualificado; niveles de evidencia hasta SILVER_REFERENCE_V1 (nunca "gold" automatico). Margenes P91-R: +-4.52 pp por etiqueta (p=0.05, DEFF 1.2); +-2.61 pp agregado de 300 celdas.

**Decision**: `ANNOTATION_PACKAGE_READY_WITH_LIMITATIONS` + `HUMAN_ANNOTATION_PENDING` (situacion B: ambiguedades documentadas; piloto exploratorio bajo guia provisional v1). 14/14 verificaciones del paquete en verde; V03 FROZEN; ninguna fase de modelado autorizada. Entregables: `reports/p92_human_annotation_plan.md`, `p92_annotation_guidelines_v1.md`, `p92_sampling_manifest.json`, `p92_annotation_readiness_audit.md`, `p92_internal_sampling_key.csv`.

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
    exec(compile(CODIGO, "<seccion 139>", "exec"), espacio)


def main() -> None:
    """Integra la seccion 139, ejecuta su verificacion y guarda el notebook."""
    seccion = construir_seccion()
    if any(ord(c) > 127 for c in seccion):
        raise SystemExit("Byte > 127 en el texto generado (seccion)")
    ejecutar_verificacion()
    notebook = json.loads(RUTA_NOTEBOOK.read_text(encoding="utf-8"))
    cells = notebook["cells"]
    celdas_antes = len(cells)
    while any("".join(c.get("source", [])).startswith(MARCA_SECCION) for c in cells):
        cells.pop(next(i for i, c in enumerate(cells) if "".join(c.get("source", [])).startswith(MARCA_SECCION)))
    while any(MARCA_CODIGO in "".join(c.get("source", [])) for c in cells):
        cells.pop(next(i for i, c in enumerate(cells) if MARCA_CODIGO in "".join(c.get("source", []))))
    cells.append(nueva_celda_markdown(seccion))
    cells.append(nueva_celda_codigo(CODIGO))
    validar(cells)
    notebook["cells"] = cells
    RUTA_NOTEBOOK.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print("Notebook actualizado con la seccion 139:")
    print(f"  celdas antes: {celdas_antes} | despues: {len(cells)}")
    print("  paquete P92 verificado en aislamiento: OK")
    print("  celdas de codigo compiladas: OK | ASCII puro en lo generado: OK (0 bytes > 127)")


if __name__ == "__main__":
    main()
