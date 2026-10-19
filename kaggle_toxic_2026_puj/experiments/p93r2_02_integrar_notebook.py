"""P93R2-02: integra la seccion 150 (P93-R2 REVISIONES C1/C2) en el notebook.

Lee los entregables `reports/p93r2_*` y las revisiones archivadas y construye una
subseccion compacta. Idempotente y ASCII puro. La celda de verificacion se ejecuta de
forma aislada (solo lee el proyecto).

Uso:
    /home/carlos/venv/bin/python experiments/p93r2_02_integrar_notebook.py
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
RUTA_NOTEBOOK = RAIZ / "notebooks" / "10_final_pipeline.ipynb"
MARCA_SECCION = "## 150. P93-R2 - REVISIONES C1/C2 RECIBIDAS"
MARCA_CODIGO = "Verificacion de la seccion 150 (P93-R2"


def nueva_celda_markdown(texto: str) -> dict:
    """Crea una celda de markdown con identificador unico."""
    return {"cell_type": "markdown", "id": uuid.uuid4().hex[:8], "metadata": {},
            "source": texto.splitlines(keepends=True)}


def nueva_celda_codigo(texto: str) -> dict:
    """Crea una celda de codigo vacia de salidas y sin ejecutar."""
    return {"cell_type": "code", "id": uuid.uuid4().hex[:8], "metadata": {},
            "execution_count": None, "outputs": [], "source": texto.splitlines(keepends=True)}


CODIGO = '''"""Verificacion de la seccion 150 (P93-R2, revisiones C1/C2 recibidas).

Comprueba las copias archivadas, la ronda 2 preparada y la integridad. No entrena nada.
"""

import csv
import hashlib
import json
from pathlib import Path

resp = PROJECT_ROOT / "artifacts" / "p93_adjudication_v1" / "responses"
for nombre in ("response_C1.csv", "response_C2.csv", "response_C1_completado.csv",
               "response_C2_completado.csv", "PROVENANCE.md"):
    assert (resp / nombre).exists(), f"falta {nombre}"
for nombre in ("p93r2_reception_report.md", "p93r2_reception_manifest.json",
               "p93r2_round2_comparison_dual_reviewer.csv"):
    assert (PROJECT_ROOT / "reports" / nombre).exists(), f"falta {nombre}"

def leer(ruta):
    with open(ruta, encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))

c1 = leer(resp / "response_C1.csv")
c2 = leer(resp / "response_C2.csv")
assert len(c1) == 127 and len(c2) == 127
TOKENS = {"POSITIVE", "NEGATIVE", "UNABLE_TO_DETERMINE"}
for filas in (c1, c2):
    assert all(f[x] in TOKENS for f in filas
               for x in ("HS_annotation", "TR_annotation", "AG_annotation"))

pkg = PROJECT_ROOT / "artifacts" / "p93_adjudication_v1" / "independent_review"
assert hashlib.sha256((pkg / "adjudicator_C1.csv").read_bytes()).hexdigest().startswith(
    "f89f74cdabd9649d")
assert hashlib.sha256((pkg / "adjudicator_C2.csv").read_bytes()).hexdigest().startswith(
    "e0449a811fa8b64b")
assert hashlib.sha256((pkg / "adjudicator_C.csv").read_bytes()).hexdigest().startswith(
    "84ca2179378fa0d8")

r2 = leer(PROJECT_ROOT / "reports" / "p93r2_round2_comparison_dual_reviewer.csv")
assert len(r2) == 176
assert all(f["post_comparison_decision"] == "" and f["rationale"] == ""
           and f["remaining_uncertainty"] == "" for f in r2)

m = json.loads((PROJECT_ROOT / "reports" / "p93r2_reception_manifest.json").read_text(
    encoding="utf-8"))
assert m["status"] == "C_REVIEWS_RECEIVED_AND_VALIDATED"
assert m["gold_comparison"].startswith("NOT_PERFORMED")
assert all(m["checks"].values()), "checks con fallos"

sucios = [n for n in ("p93r2_reception_report.md", "p93r2_reception_manifest.json")
          if any(x > 127 for x in (PROJECT_ROOT / "reports" / n).read_bytes())]
assert not sucios, f"bytes > 127 en {sucios}"

print("P93-R2 (revisiones C1/C2 recibidas) verificado:")
print("  127x2 respuestas validas | template intacto | ronda 2 preparada sin decisiones finales")
print("  sin comparacion con gold | artefactos ASCII OK")'''


def construir_seccion() -> str:
    """Construye el markdown de la seccion 150."""
    m = json.loads((RAIZ / "reports" / "p93r2_reception_manifest.json").read_text(
        encoding="utf-8"))
    cc = m["crude_C1_C2"]
    cb = m["candidate_cells_vs_AB"]
    return f"""## 150. P93-R2 - REVISIONES C1/C2 RECIBIDAS Y VALIDADAS (2026-10-13)

**Recepcion; sin entrenamiento, sin cambios de etiquetas, sin comparacion con el gold.** Dos personas revisoras (C1 y C2) completaron la ronda 1 independiente sobre los 127 tweets (copias en el sitio; template intacto). Archivadas byte a byte en `responses/` con PROVENANCE y derivados completados.

**Validaciones 17/17**: 127 filas por revision, ids y orden exactos, tokens validos, version v2; C1 sin columna text (7 columnas, como A en la ronda principal), C2 con text (21 filas con artefactos de export en el texto - comillas/espacios - y 1 residual; no afectan anotaciones); notas humanas no-ASCII en C2 (31 filas; excepcion documentada). Guia v2.1 usada por ambas (notas referencian su seccion 6).

**Descriptivo (ronda 1, sin gold)**: C1-C2 crudo: HS {cc['HS']}/127, TR {cc['TR']}/127, AG {cc['AG']}/127; triple {m['triple_C1_C2']}/127. Sobre las 176 celdas candidatas: C1 coincide con A/B en {cb['C1']['coincide_con_A_B']} (UTD {cb['C1']['UTD']}, difiere {cb['C1'].get('difiere_de_A_B', 0)}); C2 coincide en {cb['C2']['coincide_con_A_B']} (UTD {cb['C2']['UTD']}, difiere {cb['C2']['difiere_de_A_B']}).

**Ronda 2 preparada (interna)**: `p93r2_round2_comparison_dual_reviewer.csv` (176 celdas con decisiones iniciales de C1 y C2, A/B y notas; campos de decision final vacios). Con dos revisores la ronda 2 admite variantes (decision conjunta / por revisor / revisor designado): la eleccion corresponde al propietario. El gold permanece oculto.

**Estado**: `C_REVIEWS_RECEIVED_AND_VALIDATED` + `ROUND2_MATERIAL_READY` + `SOURCE_LABELS_UNCHANGED` + `V03 = FROZEN`.

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
    exec(compile(CODIGO, "<seccion 150>", "exec"), espacio)


def main() -> None:
    """Integra la seccion 150, ejecuta su verificacion y guarda el notebook."""
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
    print("Notebook actualizado con la seccion 150:")
    print(f"  celdas antes: {celdas_antes} | despues: {len(cells)}")
    print("  recepcion C1/C2 verificada en aislamiento: OK")
    print("  celdas de codigo compiladas: OK | ASCII puro en lo generado: OK (0 bytes > 127)")


if __name__ == "__main__":
    main()
