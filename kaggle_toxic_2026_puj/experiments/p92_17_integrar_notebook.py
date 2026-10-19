"""P92-17: integra la seccion 145 (P92-X GUIA v2.1) en el notebook.

Lee los entregables `reports/p92x_*` y la guia v2.1 y construye una subseccion compacta.
Idempotente y ASCII puro. La celda de verificacion se ejecuta de forma aislada (solo lee
el proyecto).

Uso:
    /home/carlos/venv/bin/python experiments/p92_17_integrar_notebook.py
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
RUTA_NOTEBOOK = RAIZ / "notebooks" / "10_final_pipeline.ipynb"
MARCA_SECCION = "## 145. P92-X - GUIA v2.1"
MARCA_CODIGO = "Verificacion de la seccion 145 (P92-X"


def nueva_celda_markdown(texto: str) -> dict:
    """Crea una celda de markdown con identificador unico."""
    return {"cell_type": "markdown", "id": uuid.uuid4().hex[:8], "metadata": {},
            "source": texto.splitlines(keepends=True)}


def nueva_celda_codigo(texto: str) -> dict:
    """Crea una celda de codigo vacia de salidas y sin ejecutar."""
    return {"cell_type": "code", "id": uuid.uuid4().hex[:8], "metadata": {},
            "execution_count": None, "outputs": [], "source": texto.splitlines(keepends=True)}


CODIGO = '''"""Verificacion de la seccion 145 (P92-X, guia v2.1 y componente B).

Comprueba los entregables de P92-X, la revision v2.1 en los paquetes y la reconciliacion
del componente B. No entrena nada.
"""

import csv
import hashlib
import json
import re
from pathlib import Path

entregables = ["p92x_guideline_provenance_audit.md", "p92x_component_b_reconciliation.csv",
               "p92x_final_distribution_gate.md",
               "p92_annotation_guidelines_v2_1_holdout_safe.md"]
for nombre in entregables:
    assert (PROJECT_ROOT / "reports" / nombre).exists(), f"falta {nombre}"

guia = (PROJECT_ROOT / "reports" / "p92_annotation_guidelines_v2_1_holdout_safe.md").read_text(encoding="utf-8")
lineas = [l for l in guia.splitlines(keepends=True)
          if not l.startswith("SHA256 (contenido sin la linea")]
h = hashlib.sha256("".join(lineas).encode("utf-8")).hexdigest()
declarado = re.search(r"SHA256 \\(contenido sin la linea que contiene este hash\\): ([0-9a-f]{64})", guia).group(1)
assert declarado == h, "hash interno de la guia v2.1 no coincide"
assert "holdout" not in guia.lower()
assert "## 10." in guia and "## 11." in guia

v21 = PROJECT_ROOT / "artifacts" / "human_annotation_v2_1_holdout_safe"
for letra in ("A", "B"):
    carpeta = v21 / f"annotator_{letra}"
    assert (carpeta / "annotation_guidelines_v2_1_holdout_safe.md").exists()
    assert not (carpeta / "annotation_guidelines_v2_frozen.md").exists()
    assert "v2.1" in (carpeta / "README_annotator.md").read_text(encoding="utf-8")

m = json.loads((v21 / "manifest.json").read_text(encoding="utf-8"))
assert m["guideline_sha256"] == h
assert m["distribution"] == "NOT_DISTRIBUTED"

with (PROJECT_ROOT / "reports" / "p92x_component_b_reconciliation.csv").open(
        encoding="utf-8", newline="") as fh:
    recon = list(csv.DictReader(fh))
tiers = {f["clave"]: int(f["valor"]) for f in recon
         if f["categoria"] == "exclusive_sampling_tier" and f["clave"] != "TOTAL"}
assert tiers == {"PILI": 19, "DRC": 19, "HCD": 112, "LSM_otros": 0}
assert sum(tiers.values()) == 150

aud = (PROJECT_ROOT / "reports" / "p92x_guideline_provenance_audit.md").read_text(encoding="utf-8")
assert "POTENTIAL_HOLDOUT2_INFLUENCE" in aud
assert "C-23" in aud and "UNRESOLVED_PROVENANCE" in aud

gate = (PROJECT_ROOT / "reports" / "p92x_final_distribution_gate.md").read_text(encoding="utf-8")
assert "READY_WITH_DOCUMENTED_GUIDELINE_LIMITATION" in gate
assert "G9" in gate

for letra in ("A", "B"):
    with (v21 / f"annotator_{letra}" / f"annotator_{letra}.csv").open(
            encoding="utf-8", newline="") as fh:
        filas = list(csv.DictReader(fh))
    assert len(filas) == 250
    assert all(f[c] == "" for f in filas
               for c in ("HS_annotation", "TR_annotation", "AG_annotation",
                         "uncertainty_notes", "tweet_note"))

assert (PROJECT_ROOT / "artifacts" / "human_annotation_v2" / "QUARANTINE_NOTICE.md").exists()
for nombre, prefijo in (("annotator_A.csv", "7dfd5b898d3905f2"),
                        ("annotator_B.csv", "956646fa479f0c65")):
    hh = hashlib.sha256(
        (PROJECT_ROOT / "artifacts" / "human_annotation_v1" / nombre).read_bytes()).hexdigest()
    assert hh.startswith(prefijo), f"v1 {nombre} cambio"

sucios = [n for n in entregables
          if any(x > 127 for x in (PROJECT_ROOT / "reports" / n).read_bytes())]
assert not sucios, f"bytes > 127 en {sucios}"

print("P92-X (guia v2.1 y componente B) verificado:")
print("  v2.1 con hash interno y sin vocabulario de particiones | paquetes actualizados")
print("  B: PILI 19 + DRC 19 + HCD 112 + LSM 0 = 150 | READY_WITH_DOCUMENTED_GUIDELINE_LIMITATION")'''


def construir_seccion() -> str:
    """Construye el markdown de la seccion 145."""
    import hashlib
    guia = (RAIZ / "reports" / "p92_annotation_guidelines_v2_1_holdout_safe.md").read_text(
        encoding="utf-8")
    lineas = [l for l in guia.splitlines(keepends=True)
              if not l.startswith("SHA256 (contenido sin la linea")]
    h = hashlib.sha256("".join(lineas).encode("utf-8")).hexdigest()
    return f"""## 145. P92-X - GUIA v2.1 Y COMPONENTE B RECONCILIADOS (2026-10-13)

**Fase documental; sin entrenamiento, sin predicciones, sin cambios de muestra ni etiquetas.** Dos puntos resueltos antes de autorizar la distribucion:

**Procedencia de la guia**: auditoria clausula por clausula (`p92x_guideline_provenance_audit.md`, 24 filas): 3 definiciones oficiales, 19 reglas operativas generales (todas con fundamento documentado y evidencia elegible), 1 clausula con `POTENTIAL_HOLDOUT2_INFLUENCE` (el conteo "6 de 15", que incluia 2 filas excluidas), 0 ejemplos textuales, 0 `UNRESOLVED_PROVENANCE`. Se creo la revision **v2.1** (`reports/p92_annotation_guidelines_v2_1_holdout_safe.md`, sha256 interno `{h[:16]}`): reemplaza el conteo por evidencia elegible (4/12; 10/12), agrega seccion de procedencia y changelog R-01..R-03, sin cambios semanticos. La revision no elimina la exposicion historica: el flag `GUIDELINE_POTENTIAL_INDIRECT_HOLDOUT2_INFLUENCE` sigue vigente.

**Componente B reconciliado** (`p92x_component_b_reconciliation.csv`): tiers exclusivos **PILI 19 + DRC 19 + HCD 112 + LSM 0 = 150** (sin doble contabilizacion; simulacion con la semilla 20261018 reproduce la seleccion registrada); 186 asignaciones de motivo (con solapes) en 11 combinaciones; disponibles vs admitidos documentados (PILI 33->19 por cap 2/grupo). Exclusiones de la cola: 69/69 con causa reservada. A=100, B=150, calibracion=15, disjuntos.

**Instrucciones actualizadas en ambos paquetes** (guia v2.1 + READMEs; CSVs byte a byte sin cambios, respuestas vacias). Gates G1-G9: todos OK; clasificacion final **`READY_WITH_DOCUMENTED_GUIDELINE_LIMITATION`**; `MAIN_ANNOTATION = NOT_DISTRIBUTED` (la entrega la decide el propietario).

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
    exec(compile(CODIGO, "<seccion 145>", "exec"), espacio)


def main() -> None:
    """Integra la seccion 145, ejecuta su verificacion y guarda el notebook."""
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
    print("Notebook actualizado con la seccion 145:")
    print(f"  celdas antes: {celdas_antes} | despues: {len(cells)}")
    print("  auditoria P92-X verificada en aislamiento: OK")
    print("  celdas de codigo compiladas: OK | ASCII puro en lo generado: OK (0 bytes > 127)")


if __name__ == "__main__":
    main()
