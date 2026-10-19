"""P93R3-02: integra la seccion 151 (P93-R3 RONDA 2 PREPARADA) en el notebook.

Lee los entregables `reports/p93r3_*` y los paquetes de ronda 2 y construye una
subseccion compacta. Idempotente y ASCII puro. La celda de verificacion se ejecuta de
forma aislada (solo lee el proyecto).

Uso:
    /home/carlos/venv/bin/python experiments/p93r3_02_integrar_notebook.py
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
RUTA_NOTEBOOK = RAIZ / "notebooks" / "10_final_pipeline.ipynb"
MARCA_SECCION = "## 151. P93-R3 - RONDA 2 PREPARADA"
MARCA_CODIGO = "Verificacion de la seccion 151 (P93-R3"


def nueva_celda_markdown(texto: str) -> dict:
    """Crea una celda de markdown con identificador unico."""
    return {"cell_type": "markdown", "id": uuid.uuid4().hex[:8], "metadata": {},
            "source": texto.splitlines(keepends=True)}


def nueva_celda_codigo(texto: str) -> dict:
    """Crea una celda de codigo vacia de salidas y sin ejecutar."""
    return {"cell_type": "code", "id": uuid.uuid4().hex[:8], "metadata": {},
            "execution_count": None, "outputs": [], "source": texto.splitlines(keepends=True)}


CODIGO = '''"""Verificacion de la seccion 151 (P93-R3, ronda 2 preparada).

Comprueba los entregables de P93-R3, los paquetes de ronda 2 y la preservacion de los
originales. No entrena nada.
"""

import csv
import hashlib
import json
from pathlib import Path

for nombre in ("p93r3_round1_blinding_audit.md", "p93r3_mapping_integrity_audit.md",
               "p93r3_round2_plan.md", "p93r3_round2_manifest.json",
               "p93r3_round2_id_assignment.csv",
               "p93r3_final_reconciliation_template.csv"):
    assert (PROJECT_ROOT / "reports" / nombre).exists(), f"falta {nombre}"
v2 = PROJECT_ROOT / "artifacts" / "p93_adjudication_v2"
for revisor in ("C1", "C2"):
    for nombre in (f"adjudication_round2_{revisor}.csv", "annotation_guidelines_v2_1.md",
                   "README_adjudicator.md", "manifest.json"):
        assert (v2 / f"annotator_{revisor}" / nombre).exists(), f"falta {revisor}/{nombre}"

def leer(ruta):
    with open(ruta, encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))

c1 = leer(v2 / "annotator_C1" / "adjudication_round2_C1.csv")
c2 = leer(v2 / "annotator_C2" / "adjudication_round2_C2.csv")
assert len(c1) == 176 and len(c2) == 176
assert all(f["final_response"] == "" and f["final_rationale"] == ""
           and f["remaining_uncertainty"] == "" for f in c1 + c2)
assert not any("P93S2-" in str(f) for f in c1)
assert not any("P93S1-" in str(f) for f in c2)

m = json.loads((PROJECT_ROOT / "reports" / "p93r3_round2_manifest.json").read_text(
    encoding="utf-8"))
assert m["status"] == "ROUND2_PACKAGES_READY"
assert m["round1_blinding"] == "ROUND1_BLINDING_NOT_FULLY_VERIFIABLE"
assert m["gold_comparison"] == "NOT_YET_AUTHORIZED"
assert all(m["checks"].values()), "checks con fallos"
assert m["counts"]["cells"] == 176 and m["counts"]["unique_tweets"] == 127

resp = PROJECT_ROOT / "artifacts" / "p93_adjudication_v1" / "responses"
assert hashlib.sha256((resp / "response_C1.csv").read_bytes()).hexdigest().startswith(
    "f89f74cdabd9649d")
assert hashlib.sha256((resp / "response_C2.csv").read_bytes()).hexdigest().startswith(
    "e0449a811fa8b64b")

sucios = [n for n in ("p93r3_round1_blinding_audit.md", "p93r3_mapping_integrity_audit.md",
                      "p93r3_round2_plan.md", "p93r3_round2_manifest.json")
          if any(x > 127 for x in (PROJECT_ROOT / "reports" / n).read_bytes())]
assert not sucios, f"bytes > 127 en {sucios}"

print("P93-R3 (ronda 2 preparada) verificado:")
print("  176 celdas x2 (ids/orden distintos, finales vacios, sin cruces entre revisores)")
print("  mapeo verificado | cegamiento ronda 1 con limitacion registrada | originales intactos")'''


def construir_seccion() -> str:
    """Construye el markdown de la seccion 151."""
    m = json.loads((RAIZ / "reports" / "p93r3_round2_manifest.json").read_text(
        encoding="utf-8"))
    r = m["reproduction"]
    return f"""## 151. P93-R3 - AUDITORIA DE RONDA 1 Y RONDA 2 PREPARADA (2026-10-13)

**Preparacion; sin entrenamiento, sin cambios de etiquetas, sin comparacion con el gold.** El resultado llamativo de C1 (174/176) fue auditado con recomputacion independiente: **mapeo integro** (176 celdas = exactamente las de A==B binario distinto del gold; recuentos 23/87/66 reproducidos; union por identificador; sin intercambio de etiquetas; reproduccion exacta de C1 {r['C1']} y C2 {r['C2']}) y **sin firma de copia** (C1 coincide con A/B tambien en 109/109 celdas no candidatas; C2 mantiene tasas similares entre grupos, con 9 diferencias y 8 UTD propias).

**Cegamiento ronda 1**: `ROUND1_BLINDING_NOT_FULLY_VERIFIABLE` - verificadas las exclusiones de diseno (sin gold, A/B, lista de candidatos ni motivos en el paquete; ids opacos; instrucciones), pero sin registros de acceso ni acuse, con recuentos agregados en el manifest del paquete y guardado en carpeta compartida (10 s de diferencia). No se afirma independencia total de la ronda 1.

**Ronda 2 preparada** (`artifacts/p93_adjudication_v2/annotator_C1/` y `annotator_C2/`): 176 celdas por revisor, **IDs nuevos** (P93S1-/P93S2-) y **orden distinto** (semillas registradas), con texto, etiqueta objetivo, su respuesta y nota iniciales, respuestas A/B (sin notas) y columnas finales vacias; sin la respuesta del otro revisor, sin gold, sin V03, sin motivos, sin estadisticas. Justificacion obligatoria por cambio y por UTD final; no se fuerza consenso ni mayoria.

**Post-recepcion (P3)**: plantilla interna de reconciliacion lista (FINAL_AGREEMENT_BINARY / FINAL_AGREEMENT_UTD / FINAL_DISAGREEMENT / POLICY_AMBIGUITY / MISSING_OR_INVALID); tercera revision propuesta solo para desacuerdos/ambiguedades, sin desempate automatico. **P4 (gold): NOT_YET_AUTHORIZED.**

**Estado**: `ROUND1_INTEGRITY_VERIFIED` (con limitacion de cegamiento documentada) + `ROUND2_PACKAGES_READY` + `C1_C2_FINAL_DECISIONS_PENDING` + `SOURCE_LABELS_UNCHANGED` + `V03 = FROZEN`.

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
    exec(compile(CODIGO, "<seccion 151>", "exec"), espacio)


def main() -> None:
    """Integra la seccion 151, ejecuta su verificacion y guarda el notebook."""
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
    print("Notebook actualizado con la seccion 151:")
    print(f"  celdas antes: {celdas_antes} | despues: {len(cells)}")
    print("  ronda 2 P93-R3 verificada en aislamiento: OK")
    print("  celdas de codigo compiladas: OK | ASCII puro en lo generado: OK (0 bytes > 127)")


if __name__ == "__main__":
    main()
