"""P92-09: integra la seccion 141 (P92-U DISCUSION HUMANA PREPARADA) en el notebook.

Lee los entregables `reports/p92u_*` y `reports/p92_guideline_decision_record.md` y
construye una subseccion compacta. Idempotente y ASCII puro. La celda de verificacion se
ejecuta de forma aislada (solo lee archivos del proyecto).

Uso:
    /home/carlos/venv/bin/python experiments/p92_09_integrar_notebook.py
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
RUTA_NOTEBOOK = RAIZ / "notebooks" / "10_final_pipeline.ipynb"
RUTA_GATE = RAIZ / "reports" / "p92u_readiness_gate.md"
MARCA_SECCION = "## 141. P92-U - DISCUSION HUMANA PREPARADA"
MARCA_CODIGO = "Verificacion de la seccion 141 (P92-U"


def nueva_celda_markdown(texto: str) -> dict:
    """Crea una celda de markdown con identificador unico."""
    return {"cell_type": "markdown", "id": uuid.uuid4().hex[:8], "metadata": {},
            "source": texto.splitlines(keepends=True)}


def nueva_celda_codigo(texto: str) -> dict:
    """Crea una celda de codigo vacia de salidas y sin ejecutar."""
    return {"cell_type": "code", "id": uuid.uuid4().hex[:8], "metadata": {},
            "execution_count": None, "outputs": [], "source": texto.splitlines(keepends=True)}


CODIGO = '''"""Verificacion de la seccion 141 (P92-U, discusion humana preparada).

Comprueba los entregables de P92-U, la estructura del registro de decisiones (campos
humanos vacios) y la integridad. No entrena nada.
"""

import csv
import json
from pathlib import Path

entregables = [
    "p92u_human_discussion_pack.md",
    "p92_guideline_decision_record.md",
    "p92u_readiness_gate.md",
]
for nombre in entregables:
    assert (PROJECT_ROOT / "reports" / nombre).exists(), f"falta {nombre}"

pack = (PROJECT_ROOT / "reports" / "p92u_human_discussion_pack.md").read_text(encoding="utf-8")
for q in range(1, 9):
    assert f"### Q{q}." in pack, f"falta Q{q} en el pack"
assert "REQUIRES_GOVERNANCE_CLARIFICATION" in pack
assert "NO aprobada" in pack

registro = (PROJECT_ROOT / "reports" / "p92_guideline_decision_record.md").read_text(encoding="utf-8")
for q in range(1, 9):
    assert f"- question_id: Q{q}" in registro, f"falta Q{q} en el registro"
assert registro.count("- human_decision: (vacio - completar en la discusion)") == 8
assert "CH-08" in registro

gate = (PROJECT_ROOT / "reports" / "p92u_readiness_gate.md").read_text(encoding="utf-8")
assert "NOT_READY (por diseno)" in gate
assert gate.count("PENDIENTE") >= 4

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

print("P92-U (discusion humana preparada) verificado:")
print("  pack Q1-Q8 | registro con 8 decisiones vacias | gate NOT_READY por diseno")
print("  respuestas intactas | muestra principal sin distribuir | ASCII puro")'''


def construir_seccion() -> str:
    """Construye el markdown de la seccion 141 a partir de los resultados reales."""
    gate = RUTA_GATE.read_text(encoding="utf-8")
    pendientes = gate.count("PENDIENTE")
    return f"""## 141. P92-U - DISCUSION HUMANA PREPARADA Y GATE DEFINIDO (2026-10-12)

**Fase de preparacion; sin decisiones automaticas, sin entrenamiento, sin distribucion.** La discusion humana de la calibracion quedo preparada con un paquete final autocontenido (Q1 a Q8: definicion oficial, evidencia, alternativas con ventajas y riesgos, opcion conservadora documentada como respaldo NO aprobado, decision requerida y estado). Estados por pregunta: Q1 y Q3 `REQUIRES_GOVERNANCE_CLARIFICATION`; Q2 y Q4-Q8 `UNRESOLVED`; ninguna `DECIDED` (el agente no decide en nombre de los participantes).

**Registro de decisiones** (`p92_guideline_decision_record.md`): plantilla editable con los 11 campos por pregunta (question_id, issue, official_rule, observed_evidence, options_considered, human_decision, rationale, remaining_uncertainty, decision_owner, decision_date, guideline_change); campos humanos vacios; tabla de correspondencia con CH-01 a CH-08; decision de inicio de anotacion principal (seccion 3, vacia).

**Gate de distribucion** (`p92u_readiness_gate.md`): estado `NOT_READY (por diseno)`; {pendientes} condiciones PENDIENTE; ambiguedades normativas documentadas (cumplida, mantener al dia); dos rutas si persiste la ambiguedad (piloto exploratorio con UTD explicito, o esperar aclaracion via Q2). Auditoria de integridad re-verificada: 13/13 hashes OK (respuestas, normalizados, paquetes, V03, OOF, train, particion, modelo).

**Consulta propuesta al organizador** (redaccion lista, NO enviada): HS con insulto sexista individual a persona concreta y AG agresivo vs. vulgar / generalizacion hostil.

**Siguiente paso**: completar el registro en la discusion humana -> congelar la version aplicable de la guia (changelog + SHA256) -> actualizar ambos paquetes a la version congelada -> re-verificar el gate -> distribuir (solo entonces).

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
    exec(compile(CODIGO, "<seccion 141>", "exec"), espacio)


def main() -> None:
    """Integra la seccion 141, ejecuta su verificacion y guarda el notebook."""
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
    print("Notebook actualizado con la seccion 141:")
    print(f"  celdas antes: {celdas_antes} | despues: {len(cells)}")
    print("  discusion P92-U verificada en aislamiento: OK")
    print("  celdas de codigo compiladas: OK | ASCII puro en lo generado: OK (0 bytes > 127)")


if __name__ == "__main__":
    main()
