"""P92-11: integra la seccion 142 (P92-T-R AUDITORIA DE PROCEDENCIA) en el notebook.

Lee los entregables `reports/p92tr_*` y las notas aditivas aplicadas, y construye una
subseccion compacta. Idempotente y ASCII puro. La celda de verificacion se ejecuta de
forma aislada (solo lee archivos del proyecto).

Uso:
    /home/carlos/venv/bin/python experiments/p92_11_integrar_notebook.py
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
RUTA_NOTEBOOK = RAIZ / "notebooks" / "10_final_pipeline.ipynb"
MARCA_SECCION = "## 142. P92-T-R - AUDITORIA DE PROCEDENCIA"
MARCA_CODIGO = "Verificacion de la seccion 142 (P92-T-R"


def nueva_celda_markdown(texto: str) -> dict:
    """Crea una celda de markdown con identificador unico."""
    return {"cell_type": "markdown", "id": uuid.uuid4().hex[:8], "metadata": {},
            "source": texto.splitlines(keepends=True)}


def nueva_celda_codigo(texto: str) -> dict:
    """Crea una celda de codigo vacia de salidas y sin ejecutar."""
    return {"cell_type": "code", "id": uuid.uuid4().hex[:8], "metadata": {},
            "execution_count": None, "outputs": [], "source": texto.splitlines(keepends=True)}


CODIGO = '''"""Verificacion de la seccion 142 (P92-T-R, auditoria de procedencia).

Comprueba el addendum, la solicitud de confirmacion, las notas aditivas y la
integridad. No entrena nada.
"""

import csv
import json
from pathlib import Path

entregables = ["p92tr_encoding_provenance_audit.md", "p92tr_ab_confirmation_request.md"]
for nombre in entregables:
    assert (PROJECT_ROOT / "reports" / nombre).exists(), f"falta {nombre}"

addendum = (PROJECT_ROOT / "reports" / "p92tr_encoding_provenance_audit.md").read_text(encoding="utf-8")
assert "ENCODING_UNRESOLVED" in addendum
assert "En las 2 anotaciones" in addendum, "falta la cita de la instruccion registrada"
assert "NO DEMOSTRADA al estandar estricto" in addendum

solicitud = (PROJECT_ROOT / "reports" / "p92tr_ab_confirmation_request.md").read_text(encoding="utf-8")
assert "BORRADOR; NO enviado" in solicitud
assert "uncertainty_notes" in solicitud

con_nota = [
    PROJECT_ROOT / "reports" / "p92_calibration_analysis.md",
    PROJECT_ROOT / "artifacts" / "human_annotation_v1" / "calibration_responses" / "PROVENANCE.md",
    PROJECT_ROOT / "reports" / "p92u_human_discussion_pack.md",
    PROJECT_ROOT / "reports" / "p92_guideline_decision_record.md",
]
for ruta in con_nota:
    assert "NOTA ADITIVA P92-T-R" in ruta.read_text(encoding="utf-8"), f"falta nota en {ruta.name}"

d = json.loads((PROJECT_ROOT / "reports" / "p92t_audit_results.json").read_text(encoding="utf-8"))
for anotador, prefijo in (("A", "a20521989aa3fea9"), ("B", "41e50c1f1d48760e"),
                          ("C", "8de76e21046d5091"), ("D", "763229f9a9caa176")):
    assert d["archivos"]["respuestas"][anotador]["sha256"].startswith(prefijo)

for nombre in ("annotator_A.csv", "annotator_B.csv"):
    with (PROJECT_ROOT / "artifacts" / "human_annotation_v1" / nombre).open(
            encoding="utf-8", newline="") as fh:
        filas = list(csv.DictReader(fh))
    assert len(filas) == 250
    assert all(f[c] == "" for f in filas
               for c in ("HS_annotation", "TR_annotation", "AG_annotation"))

sucios = [n for n in entregables
          if any(b > 127 for b in (PROJECT_ROOT / "reports" / n).read_bytes())]
assert not sucios, f"bytes > 127 en {sucios}"

print("P92-T-R (auditoria de procedencia) verificado:")
print("  addendum con cita e ENCODING_UNRESOLVED | solicitud A/B borrador | 4 notas aditivas")
print("  respuestas intactas | muestra principal sin distribuir | ASCII puro")'''


def construir_seccion() -> str:
    """Construye el markdown de la seccion 142."""
    return """## 142. P92-T-R - AUDITORIA DE PROCEDENCIA DE LA CODIFICACION A/B (2026-10-12)

**Correccion aditiva; sin entrenamiento, sin modificacion de etiquetas, sin distribucion.** La afirmacion `ENCODING_CONFIRMED` de la seccion 140 (y de P92-T) quedo auditada contra la evidencia localizable.

**Evidencia localizada**: la instruccion del usuario existe como mensaje registrado en la sesion de trabajo del 2026-10-12 (cita verbatim en `reports/p92tr_encoding_provenance_audit.md`), pero (1) no estaba persistida en ningun artefacto del proyecto (el addendum la deja registrada), (2) no enumera literalmente la direccion del mapeo de clases y (3) es de nivel coordinador, sin confirmacion directa de A ni de B. La busqueda en disco no encontro ninguna evidencia cruda; solo registros derivados y el estado previo "PENDIENTE DE CONFIRMACION" del informe de recepcion v2.

**Estado corregido**: `ENCODING_UNRESOLVED` al estandar estricto. `uncertainty_notes=1`: contenido literal en la instruccion registrada, pendiente de confirmacion directa. Mapeo `0/1 -> NEGATIVE/POSITIVE`: lectura natural y coherente, pero no demostrada al estandar exigido.

**Impacto por niveles**: validos como crudos (C/D con sus UTD, acuerdo C-D, acuerdo A-B por igualdad de valores crudos 10/11/12 de 15, catalogo de ambiguedades); condicionales a la codificacion (pares con A/B, combinacion exacta, alfas, normalizados de A/B); confirmados: ninguno; no afectado: la utilidad de la calibracion para revisar la guia (v2, registro, gate siguen vigentes).

**Accion minima**: solicitud directa a A y B (`reports/p92tr_ab_confirmation_request.md`, borrador no enviado; dos preguntas de seleccion por anotador, por separado).

**Notas aditivas aplicadas** (marcador "NOTA ADITIVA P92-T-R"): `p92_calibration_analysis.md`, `PROVENANCE.md`, `p92u_human_discussion_pack.md`, `p92_guideline_decision_record.md`. Respuestas humanas, normalizados y resultados historicos intactos.

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
    exec(compile(CODIGO, "<seccion 142>", "exec"), espacio)


def main() -> None:
    """Integra la seccion 142, ejecuta su verificacion y guarda el notebook."""
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
    print("Notebook actualizado con la seccion 142:")
    print(f"  celdas antes: {celdas_antes} | despues: {len(cells)}")
    print("  auditoria P92-T-R verificada en aislamiento: OK")
    print("  celdas de codigo compiladas: OK | ASCII puro en lo generado: OK (0 bytes > 127)")


if __name__ == "__main__":
    main()
