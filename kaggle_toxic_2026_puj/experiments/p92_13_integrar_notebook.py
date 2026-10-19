"""P92-13: integra la seccion 143 (P92-V POLITICA V2 CONGELADA) en el notebook.

Lee los entregables `reports/p92v_*`, la guia congelada y los paquetes
`artifacts/human_annotation_v2/` y construye una subseccion compacta. Idempotente y
ASCII puro. La celda de verificacion se ejecuta de forma aislada (solo lee el proyecto).

Uso:
    /home/carlos/venv/bin/python experiments/p92_13_integrar_notebook.py
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
RUTA_NOTEBOOK = RAIZ / "notebooks" / "10_final_pipeline.ipynb"
MARCA_SECCION = "## 143. P92-V - POLITICA OPERATIVA V2 CONGELADA"
MARCA_CODIGO = "Verificacion de la seccion 143 (P92-V"


def nueva_celda_markdown(texto: str) -> dict:
    """Crea una celda de markdown con identificador unico."""
    return {"cell_type": "markdown", "id": uuid.uuid4().hex[:8], "metadata": {},
            "source": texto.splitlines(keepends=True)}


def nueva_celda_codigo(texto: str) -> dict:
    """Crea una celda de codigo vacia de salidas y sin ejecutar."""
    return {"cell_type": "code", "id": uuid.uuid4().hex[:8], "metadata": {},
            "execution_count": None, "outputs": [], "source": texto.splitlines(keepends=True)}


CODIGO = '''"""Verificacion de la seccion 143 (P92-V, politica operativa v2 congelada).

Comprueba la guia congelada (hash interno), los paquetes v2, la conservacion de v1 y el
manifiesto. No entrena nada.
"""

import csv
import hashlib
import json
import re
from pathlib import Path

entregables = ["p92_annotation_guidelines_v2_frozen.md",
               "p92v_operational_policy_decisions.md", "p92v_package_audit.md",
               "p92v_manifest.json"]
for nombre in entregables:
    assert (PROJECT_ROOT / "reports" / nombre).exists(), f"falta {nombre}"
v2 = PROJECT_ROOT / "artifacts" / "human_annotation_v2"
for nombre in ("annotator_A/annotator_A.csv", "annotator_A/README_annotator.md",
               "annotator_A/annotation_guidelines_v2_frozen.md",
               "annotator_B/annotator_B.csv", "annotator_B/README_annotator.md",
               "annotator_B/annotation_guidelines_v2_frozen.md", "manifest.json"):
    assert (v2 / nombre).exists(), f"falta {nombre}"

guia = (PROJECT_ROOT / "reports" / "p92_annotation_guidelines_v2_frozen.md").read_text(encoding="utf-8")
lineas = [l for l in guia.splitlines(keepends=True)
          if not l.startswith("SHA256 (contenido sin la linea")]
h = hashlib.sha256("".join(lineas).encode("utf-8")).hexdigest()
declarado = re.search(r"SHA256 \\(contenido sin la linea que contiene este hash\\): ([0-9a-f]{64})", guia).group(1)
assert declarado == h, "hash interno de la guia no coincide"
assert "PROJECT_OPERATIONAL_ANNOTATION_POLICY_V2" in guia

with (v2 / "annotator_A/annotator_A.csv").open(encoding="utf-8", newline="") as fh:
    a = list(csv.DictReader(fh))
with (v2 / "annotator_B/annotator_B.csv").open(encoding="utf-8", newline="") as fh:
    b = list(csv.DictReader(fh))
assert len(a) == 250 and len(b) == 250
assert {f["review_id"] for f in a} == {f["review_id"] for f in b}
assert [f["review_id"] for f in a] != [f["review_id"] for f in b]
assert all(f[c] == "" for f in a + b
           for c in ("HS_annotation", "TR_annotation", "AG_annotation",
                     "uncertainty_notes", "tweet_note"))
assert all(f["guideline_version"] == "v2" for f in a + b)

for nombre, prefijo in (("annotator_A.csv", "7dfd5b898d3905f2"),
                        ("annotator_B.csv", "956646fa479f0c65")):
    hh = hashlib.sha256(
        (PROJECT_ROOT / "artifacts" / "human_annotation_v1" / nombre).read_bytes()).hexdigest()
    assert hh.startswith(prefijo), f"v1 {nombre} cambio"

m = json.loads((PROJECT_ROOT / "reports" / "p92v_manifest.json").read_text(encoding="utf-8"))
assert m["distribution"] == "NOT_DISTRIBUTED"
assert m["counts"]["principal_tweets"] == 250
assert m["policy_id"] == "PROJECT_OPERATIONAL_ANNOTATION_POLICY_V2"

aud = (PROJECT_ROOT / "reports" / "p92v_package_audit.md").read_text(encoding="utf-8")
assert "MAIN_ANNOTATION_PACKAGES_READY" in aud

reg = (PROJECT_ROOT / "reports" / "p92_guideline_decision_record.md").read_text(encoding="utf-8")
assert "DECISION DEL PROPIETARIO DEL PROYECTO (P92-V" in reg

sucios = [n for n in entregables
          if any(b > 127 for b in (PROJECT_ROOT / "reports" / n).read_bytes())]
assert not sucios, f"bytes > 127 en {sucios}"

print("P92-V (politica operativa v2 congelada) verificado:")
print("  guia con hash interno verificable | paquetes A/B: 250 vacios, mismos casos, orden distinto")
print("  v1 intacto | manifiesto NOT_DISTRIBUTED | MAIN_ANNOTATION_PACKAGES_READY")'''


def construir_seccion() -> str:
    """Construye el markdown de la seccion 143."""
    m = json.loads((RAIZ / "reports" / "p92v_manifest.json").read_text(encoding="utf-8"))
    return f"""## 143. P92-V - POLITICA OPERATIVA V2 CONGELADA Y PAQUETES PRINCIPALES LISTOS (2026-10-13)

**Fase de preparacion; sin entrenamiento, sin predicciones, sin modificacion de etiquetas, sin distribucion.** El propietario del proyecto adopto la politica operativa `PROJECT_OPERATIONAL_ANNOTATION_POLICY_V2` (decisiones Q1-Q8 del 2026-10-13) como marco interno del piloto: definicion oficial literal + incertidumbre explicita, sin nuevas definiciones normativas; `governance_clarification = NOT_REQUESTED`, `official_semantics = UNRESOLVED`, `internal_pilot = AUTHORIZED`.

**Guia congelada**: `reports/p92_annotation_guidelines_v2_frozen.md` con hash interno verificable (sha256 sin la linea de hash: `{m['guideline_sha256'][:16]}`); changelog CH-01..CH-09 frente a v1; preguntas normativas abiertas; instrucciones exactas de UNABLE_TO_DETERMINE; esquema de 8 columnas (nueva `tweet_note` para duda general del tweet).

**Paquetes principales v2** (`artifacts/human_annotation_v2/annotator_A/` y `annotator_B/`): mismos 250 tweets y misma asignacion A/B que v1 (100 + 150), mismos casos con orden distinto, anotaciones vacias, guia congelada + README v2. Cegamiento verificado (sin ids originales, etiquetas, OOF, predicciones, umbrales, folds, motivos ni clave interna); textos identicos a la fuente (500/500); 0 registros de test; 0 del holdout de la particion v2; 35 filas principales coinciden con el pliegue holdout2 (limitacion heredada documentada: holdout2 es un pliegue de dev de V10, no virginal, fuera de la definicion de exclusion del plan P92). **NO distribuidos.**

**Estado final**: `PROJECT_OPERATIONAL_POLICY_V2_FROZEN` + `MAIN_ANNOTATION_PACKAGES_READY` + `MAIN_ANNOTATION_NOT_DISTRIBUTED` + `OFFICIAL_LABEL_SEMANTICS_UNRESOLVED` + `V03 = FROZEN` + `NO NEW MODELING PHASE`. Sin commit; v1 y las respuestas humanas intactas.

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
    exec(compile(CODIGO, "<seccion 143>", "exec"), espacio)


def main() -> None:
    """Integra la seccion 143, ejecuta su verificacion y guarda el notebook."""
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
    print("Notebook actualizado con la seccion 143:")
    print(f"  celdas antes: {celdas_antes} | despues: {len(cells)}")
    print("  politica v2 verificada en aislamiento: OK")
    print("  celdas de codigo compiladas: OK | ASCII puro en lo generado: OK (0 bytes > 127)")


if __name__ == "__main__":
    main()
