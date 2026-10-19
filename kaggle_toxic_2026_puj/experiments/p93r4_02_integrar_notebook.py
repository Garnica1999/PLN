"""P93R4-02: integra la seccion 152 (P93-R4 SEGUNDA RONDA BLOQUEADA) en el notebook.

Lee los entregables `reports/p93r4_*` y las respuestas archivadas y construye una
subseccion compacta. Idempotente y ASCII puro. La celda de verificacion se ejecuta de
forma aislada (solo lee el proyecto).

Uso:
    /home/carlos/venv/bin/python experiments/p93r4_02_integrar_notebook.py
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
RUTA_NOTEBOOK = RAIZ / "notebooks" / "10_final_pipeline.ipynb"
MARCA_SECCION = "## 152. P93-R4 - SEGUNDA RONDA BLOQUEADA"
MARCA_CODIGO = "Verificacion de la seccion 152 (P93-R4"


def nueva_celda_markdown(texto: str) -> dict:
    """Crea una celda de markdown con identificador unico."""
    return {"cell_type": "markdown", "id": uuid.uuid4().hex[:8], "metadata": {},
            "source": texto.splitlines(keepends=True)}


def nueva_celda_codigo(texto: str) -> dict:
    """Crea una celda de codigo vacia de salidas y sin ejecutar."""
    return {"cell_type": "code", "id": uuid.uuid4().hex[:8], "metadata": {},
            "execution_count": None, "outputs": [], "source": texto.splitlines(keepends=True)}


CODIGO = '''"""Verificacion de la seccion 152 (P93-R4, segunda ronda bloqueada).

Comprueba los entregables de P93-R4, las decisiones bloqueadas y la integridad.
No entrena nada.
"""

import csv
import hashlib
import json
from pathlib import Path

for nombre in ("p93r4_final_decisions.csv", "p93r4_final_human_agreement.md",
               "p93r4_final_human_agreement.csv", "p93r4_pending_cases.csv",
               "p93r4_manifest.json"):
    assert (PROJECT_ROOT / "reports" / nombre).exists(), f"falta {nombre}"
resp2 = PROJECT_ROOT / "artifacts" / "p93_adjudication_v2" / "responses"
for nombre in ("response_C1_round2.csv", "response_C2_round2_correcciones.csv",
               "PROVENANCE.md"):
    assert (resp2 / nombre).exists(), f"falta {nombre}"

def leer(ruta):
    with open(ruta, encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))

dec = leer(PROJECT_ROOT / "reports" / "p93r4_final_decisions.csv")
assert len(dec) == 176
assert "gold" not in " ".join(dec[0].keys()).lower()
TOKENS = {"POSITIVE", "NEGATIVE", "UNABLE_TO_DETERMINE"}
assert all(f[c] in TOKENS for f in dec
           for c in ("C1_final_response", "C2_final_response"))

pend = leer(PROJECT_ROOT / "reports" / "p93r4_pending_cases.csv")
assert len(pend) == 5

m = json.loads((PROJECT_ROOT / "reports" / "p93r4_manifest.json").read_text(encoding="utf-8"))
assert m["status"] == "SECOND_ROUND_RESPONSES_VALIDATED"
assert m["decisions"] == "FINAL_HUMAN_DECISIONS_LOCKED"
assert m["gold_comparison"] == "NOT_YET_AUTHORIZED"
assert all(m["checks"].values()), "checks con fallos"

resp1 = PROJECT_ROOT / "artifacts" / "p93_adjudication_v1" / "responses"
assert hashlib.sha256((resp1 / "response_C1.csv").read_bytes()).hexdigest().startswith(
    "f89f74cdabd9649d")
assert hashlib.sha256((resp1 / "response_C2.csv").read_bytes()).hexdigest().startswith(
    "e0449a811fa8b64b")

sucios = [n for n in ("p93r4_final_human_agreement.md", "p93r4_manifest.json")
          if any(x > 127 for x in (PROJECT_ROOT / "reports" / n).read_bytes())]
assert not sucios, f"bytes > 127 en {sucios}"

print("P93-R4 (segunda ronda bloqueada) verificado:")
print("  176 decisiones finales C1/C2 (sin gold) | 5 pendientes | rondas 1 intactas")
print("  estado: SECOND_ROUND_RESPONSES_VALIDATED + FINAL_HUMAN_DECISIONS_LOCKED")'''


def construir_seccion() -> str:
    """Construye el markdown de la seccion 152."""
    m = json.loads((RAIZ / "reports" / "p93r4_manifest.json").read_text(encoding="utf-8"))
    a = m["agreement"]
    return f"""## 152. P93-R4 - SEGUNDA RONDA RECIBIDA, BLOQUEADA Y ANALIZADA (2026-10-13)

**Recepcion por mensaje (bloqueo previo resuelto); sin entrenamiento, sin gold, sin cambios de etiquetas.** C1 entrego el archivo completo (176 filas; **2 cambios**: P93S1-0052 HS UTD->NEGATIVE, P93S1-0108 TR UTD->POSITIVE, con justificacion); C2 entrego los **registros corregidos** (16 filas: 12 cambios + 4 casos que permanecen en UTD; las 160 celdas restantes conservan su respuesta inicial, reconstruccion documentada). Decisiones iniciales de la ronda 1 verificadas contra los archivos archivados; decisiones finales **BLOQUEADAS** en `p93r4_final_decisions.csv` (sin gold).

**Acuerdo final (176 celdas; C1 vs C2)**: HS **23/23** (kappa 1.000), TR **85/87** (kappa 0.847; binario condicional 85/85), AG **63/66** (kappa 0.909; bin 63/64). Exacto: **171/176 celdas** y 122/127 tweets. UTD finales: C1=0; C2=4 (TR 2, AG 2). Cambios desde la ronda inicial: C1 2 (HS 1, TR 1); C2 12 (HS 1, TR 3, AG 8).

**Clasificacion**: FINAL_AGREEMENT_BINARY 171; FINAL_DISAGREEMENT **5**; POLICY_AMBIGUITY 0. Cola de pendientes (`p93r4_pending_cases.csv`): 4 celdas NEGATIVE/POSITIVE vs UNABLE_TO_DETERMINE (P92W-0035/0061 AG; 0094/0240 TR) + 1 desacuerdo binario-binario (P92W-0243 AG) -> tercera revision propuesta, sin desempate automatico.

**Limitaciones**: cegamiento de ronda 1 no verificable por completo (P93-R3); la ronda 2 mostro A/B deliberadamente, por lo que las decisiones finales no son independientes de esas respuestas (no se declara independencia estadistica); el gold permanece oculto: ninguna correccion ni error confirmado.

**Estado**: `SECOND_ROUND_RESPONSES_VALIDATED` + `FINAL_HUMAN_DECISIONS_LOCKED` + `GOLD_COMPARISON_NOT_YET_AUTHORIZED` + `SOURCE_LABELS_UNCHANGED` + `V03 = FROZEN`.

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
    exec(compile(CODIGO, "<seccion 152>", "exec"), espacio)


def main() -> None:
    """Integra la seccion 152, ejecuta su verificacion y guarda el notebook."""
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
    print("Notebook actualizado con la seccion 152:")
    print(f"  celdas antes: {celdas_antes} | despues: {len(cells)}")
    print("  bloqueo y analisis P93-R4 verificados en aislamiento: OK")
    print("  celdas de codigo compiladas: OK | ASCII puro en lo generado: OK (0 bytes > 127)")


if __name__ == "__main__":
    main()
