"""P91-03: integra la seccion 138 (P91 LABEL QUALITY RESEARCH) en el notebook.

Lee los entregables `reports/p91_*` y construye una seccion compacta. Idempotente y
ASCII puro. La celda de verificacion se ejecuta de forma aislada.

Uso:
    /home/carlos/venv/bin/python experiments/p91_03_integrar_notebook.py
"""

from __future__ import annotations

import csv
import json
import uuid
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
RUTA_NOTEBOOK = RAIZ / "notebooks" / "10_final_pipeline.ipynb"
RUTA_RESULTS = RAIZ / "reports" / "p91_audit_results.json"
MARCA_SECCION = "## 138. P91 - LABEL QUALITY RESEARCH"
MARCA_CODIGO = "Verificacion de la seccion 138 (P91"


def nueva_celda_markdown(texto: str) -> dict:
    """Crea una celda de markdown con identificador unico."""
    return {"cell_type": "markdown", "id": uuid.uuid4().hex[:8], "metadata": {},
            "source": texto.splitlines(keepends=True)}


def nueva_celda_codigo(texto: str) -> dict:
    """Crea una celda de codigo vacia de salidas y sin ejecutar."""
    return {"cell_type": "code", "id": uuid.uuid4().hex[:8], "metadata": {},
            "execution_count": None, "outputs": [], "source": texto.splitlines(keepends=True)}


CODIGO = '''"""Verificacion de la seccion 138 (P91, auditoria de calidad de etiquetas).

Comprueba los entregables: existencia, cifras clave del diagnostico, cola de
candidatos y escaneo byte a byte. No entrena nada.
"""

import csv
import json

entregables = [
    "p91_label_quality_audit.md", "p91_label_quality_diagnostics.csv",
    "p91_label_review_protocol.md", "p91_label_review_candidates.csv",
    "p91_label_state_of_art.md", "p91_annotation_gap_analysis.csv",
    "p91_audit_results.json",
]
for nombre in entregables:
    ruta = PROJECT_ROOT / "reports" / nombre
    assert ruta.exists(), f"falta {nombre}"

d = json.loads((PROJECT_ROOT / "reports" / "p91_audit_results.json").read_text(encoding="utf-8"))
assert d["audit_id"] == "P91-01"
assert d["tr_sin_hs"] == 0 and d["ag_sin_hs"] == 22
assert d["marginales"] == {"HS": 648, "TR": 445, "AG": 567}
assert d["cola"]["celdas"] == 882 and d["cola"]["muestra_referencia_tweets"] == 100

with (PROJECT_ROOT / "reports" / "p91_label_quality_diagnostics.csv").open(encoding="utf-8", newline="") as fh:
    diag = list(csv.DictReader(fh))
assert len(diag) == 40
with (PROJECT_ROOT / "reports" / "p91_label_review_candidates.csv").open(encoding="utf-8", newline="") as fh:
    cola = list(csv.DictReader(fh))
assert len(cola) == 882
assert all(f["annotation_status"] == "pending_authorization" for f in cola)

sucios = [n for n in entregables if any(b > 127 for b in (PROJECT_ROOT / "reports" / n).read_bytes())]
assert not sucios, f"bytes > 127 en {sucios}"

print("P91 (auditoria de calidad de etiquetas) verificada:")
print("  reglas documentadas; TR=>HS sin regla formal (DOCUMENTATION_INSUFFICIENT)")
print("  40 diagnosticos | cola 882 celdas | 100 tweets de referencia | 0 bytes > 127")'''


def construir_seccion() -> str:
    """Construye el markdown de la seccion 138 a partir de los entregables."""
    d = json.loads(RUTA_RESULTS.read_text(encoding="utf-8"))
    c = d["conteos_por_etiqueta"]
    cola = d["cola"]
    return f"""## 138. P91 - LABEL QUALITY RESEARCH: auditoria de etiquetas y protocolo de revision (2026-10-10)

**Fase diagnostica y metodologica; NO modifica etiquetas ni entrena.** Objetivo: determinar si existe un problema verificable de calidad de etiquetas y como comprobarlo con rigor. Sin test, sin holdouts, sin submissions, sin modelos nuevos.

**Reglas oficiales** (`docs/competition_rules.md`, verbatim del organizador): HS "indica si hay discurso de odio"; TR "indica si el discurso esta dirigido contra una persona o grupo"; AG "indica si el texto es agresivo". Son definiciones de una linea: **no existen instrucciones para anotadores ni criterios operativos**. La relacion formal TR=>HS **no esta documentada**: es una regularidad empirica perfecta (0 excepciones en 1731 filas; 445/445 en dev) y la lectura condicionada de D-001 es una interpretacion. AG presenta **22 excepciones AG=1 con HS=0** en dev (26 en train) que contradicen esa lectura para AG. Decision: `DOCUMENTATION_INSUFFICIENT` + pregunta exacta para los organizadores (no se contacta a nadie).

**Evidencia computacional** (40 diagnosticos en `p91_label_quality_diagnostics.csv`): evidencia OOF extrema contra el gold es **escasa** (FN p<=0.05: {c['HS']['fn_p_le_0.05']}/{c['TR']['fn_p_le_0.05']}/{c['AG']['fn_p_le_0.05']}; FP p>=0.95: {c['HS']['fp_p_ge_0.95']}/{c['TR']['fp_p_ge_0.95']}/{c['AG']['fp_p_ge_0.95']}); near-duplicados divergentes: 1 grupo/3 filas (near-dup) y 13 grupos/43 filas (fusionados); el desacuerdo unanime de 4 modelos ({sum(d['unanimidad_4modelos'].values())} celdas) es en su mayoria error de modelo, no senal de etiqueta. `NO_CONFIRMED_LABEL_ISSUE` con la evidencia actual.

**Protocolo y cola** (`p91_label_review_protocol.md`, `p91_label_review_candidates.csv`): revision ciega disenada (2+ anotadores, adjudicacion separada, acuerdo por etiqueta con kappa/alpha y limitaciones por prevalencia, muestreo con FPC: componente de referencia de 100 tweets con semilla fija + componente enriquecida) y cola de {cola['celdas']} celdas con motivos verificables. **La revision NO se ejecuta: falta autorizacion de gobernanza** (P87-00 la declaro LOW_EXPECTED_VALUE y no la solicito).

**Decision**: `ANNOTATION_POLICY_UNRESOLVED` + `GOVERNANCE_REVIEW_REQUIRED` + `REVIEW_QUEUE_PREPARED` + `HUMAN_ANNOTATION_PENDING` (con `NO_CONFIRMED_LABEL_ISSUE`). V03 FROZEN; la fase de modelado permanece cerrada. Entregables: `p91_label_quality_audit.md`, `p91_label_review_protocol.md`, `p91_label_state_of_art.md` y las tablas `p91_*.csv`.

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
    """Ejecuta la celda de verificacion de forma aislada (solo lee reports/)."""
    espacio: dict[str, object] = {"PROJECT_ROOT": RAIZ, "__name__": "__verificacion__"}
    exec(compile(CODIGO, "<seccion 138>", "exec"), espacio)


def main() -> None:
    """Integra la seccion 138, ejecuta su verificacion y guarda el notebook."""
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
    print("Notebook actualizado con la seccion 138:")
    print(f"  celdas antes: {celdas_antes} | despues: {len(cells)}")
    print("  auditoria P91 verificada en aislamiento: OK")
    print("  celdas de codigo compiladas: OK | ASCII puro en lo generado: OK (0 bytes > 127)")


if __name__ == "__main__":
    main()
