"""P92-21: integra la seccion 147 (P92-Z RESPUESTAS PRINCIPALES) en el notebook.

Lee los entregables `reports/p92z_*` y las respuestas recibidas y construye una
subseccion compacta. Idempotente y ASCII puro. La celda de verificacion se ejecuta de
forma aislada (solo lee el proyecto).

Uso:
    /home/carlos/venv/bin/python experiments/p92_21_integrar_notebook.py
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
RUTA_NOTEBOOK = RAIZ / "notebooks" / "10_final_pipeline.ipynb"
MARCA_SECCION = "## 147. P92-Z - RESPUESTAS PRINCIPALES RECIBIDAS"
MARCA_CODIGO = "Verificacion de la seccion 147 (P92-Z"


def nueva_celda_markdown(texto: str) -> dict:
    """Crea una celda de markdown con identificador unico."""
    return {"cell_type": "markdown", "id": uuid.uuid4().hex[:8], "metadata": {},
            "source": texto.splitlines(keepends=True)}


def nueva_celda_codigo(texto: str) -> dict:
    """Crea una celda de codigo vacia de salidas y sin ejecutar."""
    return {"cell_type": "code", "id": uuid.uuid4().hex[:8], "metadata": {},
            "execution_count": None, "outputs": [], "source": texto.splitlines(keepends=True)}


CODIGO = '''"""Verificacion de la seccion 147 (P92-Z, respuestas principales).

Comprueba las respuestas recibidas, su procedencia y el registro de recepcion. No
entrena nada.
"""

import csv
import json
from pathlib import Path

resp_dir = PROJECT_ROOT / "artifacts" / "human_annotation_v2_1_holdout_safe" / "responses"
for nombre in ("response_A.csv", "response_B.csv", "response_B_sin_texto.csv",
               "annotator_A_completado.csv", "annotator_B_completado.csv", "PROVENANCE.md"):
    assert (resp_dir / nombre).exists(), f"falta {nombre}"

def leer(ruta):
    with open(ruta, encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))

a = leer(resp_dir / "response_A.csv")
b = leer(resp_dir / "response_B.csv")
b2 = leer(resp_dir / "response_B_sin_texto.csv")
assert len(a) == 250 and len(b) == 250 and len(b2) == 250
assert {f["review_id"] for f in a} == {f["review_id"] for f in b} == {f["review_id"] for f in b2}
TOKENS = {"POSITIVE", "NEGATIVE", "UNABLE_TO_DETERMINE"}
for filas in (a, b, b2):
    assert all(f[c] in TOKENS for f in filas
               for c in ("HS_annotation", "TR_annotation", "AG_annotation"))
mapa_b = {f["review_id"]: f for f in b}
mapa_b2 = {f["review_id"]: f for f in b2}
assert all(mapa_b[r][c] == mapa_b2[r][c] for r in mapa_b
           for c in ("HS_annotation", "TR_annotation", "AG_annotation",
                     "uncertainty_notes", "tweet_note"))

m = json.loads((PROJECT_ROOT / "reports" / "p92z_responses_manifest.json").read_text(
    encoding="utf-8"))
assert m["status"] == "MAIN_RESPONSES_RECEIVED_AND_VALIDATED"
assert m["calibration"] == "PENDING_HUMAN_RESPONSES"
assert all(m["checks"].values()), "checks con fallos"

inf = (PROJECT_ROOT / "reports" / "p92z_main_responses_reception_report.md").read_text(
    encoding="utf-8")
assert "MAIN_RESPONSES_RECEIVED_AND_VALIDATED" in inf
assert "desviacion" in inf.lower()

sucios = [n for n in ("p92z_main_responses_reception_report.md", "p92z_responses_manifest.json")
          if any(x > 127 for x in (PROJECT_ROOT / "reports" / n).read_bytes())]
assert not sucios, f"bytes > 127 en {sucios}"

print("P92-Z (respuestas principales recibidas) verificado:")
print("  250x2 respuestas validas (tokens, ids, B vs B2 identicos) | calibracion pendiente")
print("  recepcion documentada | desviacion de protocolo registrada")'''


def construir_seccion() -> str:
    """Construye el markdown de la seccion 147."""
    m = json.loads((RAIZ / "reports" / "p92z_responses_manifest.json").read_text(
        encoding="utf-8"))
    co = m["crude_agreement_A_B"]
    tx = m["texts_B_vs_source"]
    return f"""## 147. P92-Z - RESPUESTAS PRINCIPALES RECIBIDAS (250 x 2) (2026-10-13)

**Recepcion y validacion; sin entrenamiento, sin cambios de etiquetas ni de guia.** Carlos entrego las respuestas de AMBOS anotadores para la **muestra principal de 250 tweets** (P92W-0001..0250): A sin columna text; B con y sin text (anotaciones identicas). Se preservaron verbatim en `responses/` (3 archivos + PROVENANCE) y se generaron copias completadas derivadas (esquema del paquete, orden del paquete, textos de la fuente).

**Desviacion de protocolo registrada**: el plan de P92-Y preveia la calibracion (15 tweets) ANTES de la ronda principal; las respuestas recibidas son de la muestra principal y la calibracion sigue `PENDING_HUMAN_RESPONSES`. No se simulan respuestas ni se revierte nada; queda para decision del propietario.

**Validaciones 19/19**: 250 filas por archivo, ids exactos, tokens validos (sin 0/1), version v2, sin reservados, B vs B2 identicos; textos de B: {tx['exactos']}/250 exactos, {tx['artefacto_comillas']} con artefactos de comillas y {tx['artefacto_comillas_y_espacios']} de comillas+espacios del export, residuales {tx['residuales']} (avisos; anotaciones intactas). Orden de A con una transposicion adyacente sin efecto (mapeo por review_id); orden de B identico al paquete.

**Descriptivo provisional**: coincidencia cruda A-B: HS {co['HS']}/250, TR {co['TR']}/250, AG {co['AG']}/250; tres etiquetas {co['tres_etiquetas']}/250. Sin kappa ni inferencia en esta fase.

**Estado**: `MAIN_RESPONSES_RECEIVED_AND_VALIDATED` + `CALIBRATION = PENDING` + `GUIDELINE = v2.1 (sin cambios)` + `V03 = FROZEN` + `NO NEW MODELING PHASE`.

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
    exec(compile(CODIGO, "<seccion 147>", "exec"), espacio)


def main() -> None:
    """Integra la seccion 147, ejecuta su verificacion y guarda el notebook."""
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
    print("Notebook actualizado con la seccion 147:")
    print(f"  celdas antes: {celdas_antes} | despues: {len(cells)}")
    print("  recepcion P92-Z verificada en aislamiento: OK")
    print("  celdas de codigo compiladas: OK | ASCII puro en lo generado: OK (0 bytes > 127)")


if __name__ == "__main__":
    main()
