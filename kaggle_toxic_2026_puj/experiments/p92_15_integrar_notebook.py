"""P92-15: integra la seccion 144 (P92-W HOLDOUT2 AUDITADO) en el notebook.

Lee los entregables `reports/p92w_*` y el paquete `human_annotation_v2_1_holdout_safe`
y construye una subseccion compacta. Idempotente y ASCII puro. La celda de verificacion
se ejecuta de forma aislada (solo lee el proyecto).

Uso:
    /home/carlos/venv/bin/python experiments/p92_15_integrar_notebook.py
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
RUTA_NOTEBOOK = RAIZ / "notebooks" / "10_final_pipeline.ipynb"
MARCA_SECCION = "## 144. P92-W - HOLDOUT2 AUDITADO"
MARCA_CODIGO = "Verificacion de la seccion 144 (P92-W"


def nueva_celda_markdown(texto: str) -> dict:
    """Crea una celda de markdown con identificador unico."""
    return {"cell_type": "markdown", "id": uuid.uuid4().hex[:8], "metadata": {},
            "source": texto.splitlines(keepends=True)}


def nueva_celda_codigo(texto: str) -> dict:
    """Crea una celda de codigo vacia de salidas y sin ejecutar."""
    return {"cell_type": "code", "id": uuid.uuid4().hex[:8], "metadata": {},
            "execution_count": None, "outputs": [], "source": texto.splitlines(keepends=True)}


CODIGO = '''"""Verificacion de la seccion 144 (P92-W, holdout2 auditado).

Comprueba los entregables de P92-W, el paquete holdout-safe y la preservacion de los
paquetes anteriores. No entrena nada.
"""

import csv
import hashlib
import json
from pathlib import Path

entregables = ["p92w_holdout2_exposure_audit.md", "p92w_eligible_population.csv",
               "p92w_sampling_manifest.json", "p92w_package_audit.md",
               "p92w_internal_sampling_key.csv"]
for nombre in entregables:
    assert (PROJECT_ROOT / "reports" / nombre).exists(), f"falta {nombre}"
v21 = PROJECT_ROOT / "artifacts" / "human_annotation_v2_1_holdout_safe"
for nombre in ("annotator_A/annotator_A.csv", "annotator_A/README_annotator.md",
               "annotator_A/annotation_guidelines_v2_frozen.md",
               "annotator_B/annotator_B.csv", "annotator_B/README_annotator.md",
               "annotator_B/annotation_guidelines_v2_frozen.md",
               "calibration/calibration_sample.csv", "calibration/README_calibration.md",
               "manifest.json", "README.md"):
    assert (v21 / nombre).exists(), f"falta {nombre}"

with (PROJECT_ROOT / "reports" / "p92w_eligible_population.csv").open(
        encoding="utf-8", newline="") as fh:
    poblacion = list(csv.DictReader(fh))
assert len(poblacion) == 1236
assert all(f["eligible"] == "True" for f in poblacion)

with (v21 / "annotator_A" / "annotator_A.csv").open(encoding="utf-8", newline="") as fh:
    a = list(csv.DictReader(fh))
with (v21 / "annotator_B" / "annotator_B.csv").open(encoding="utf-8", newline="") as fh:
    b = list(csv.DictReader(fh))
with (v21 / "calibration" / "calibration_sample.csv").open(encoding="utf-8", newline="") as fh:
    cal = list(csv.DictReader(fh))
assert len(a) == 250 and len(b) == 250 and len(cal) == 15
assert {f["review_id"] for f in a} == {f["review_id"] for f in b}
assert [f["review_id"] for f in a] != [f["review_id"] for f in b]
assert all(f[c] == "" for f in a + b + cal
           for c in ("HS_annotation", "TR_annotation", "AG_annotation",
                     "uncertainty_notes", "tweet_note"))
assert all(f["guideline_version"] == "v2" for f in a + b + cal)

m = json.loads((v21 / "manifest.json").read_text(encoding="utf-8"))
assert m["distribution"] == "NOT_DISTRIBUTED"
assert m["policy_id"] == "PROJECT_OPERATIONAL_ANNOTATION_POLICY_V2"

exp = (PROJECT_ROOT / "reports" / "p92w_holdout2_exposure_audit.md").read_text(encoding="utf-8")
assert "GUIDELINE_POTENTIAL_INDIRECT_HOLDOUT2_INFLUENCE" in exp
assert "QUARANTINED" in exp
pkg = (PROJECT_ROOT / "reports" / "p92w_package_audit.md").read_text(encoding="utf-8")
assert "READY_WITH_VERIFIED_EXCLUSIONS" in pkg
assert (PROJECT_ROOT / "artifacts" / "human_annotation_v2" / "QUARANTINE_NOTICE.md").exists()

with (PROJECT_ROOT / "reports" / "p92w_internal_sampling_key.csv").open(
        encoding="utf-8", newline="") as fh:
    clave = list(csv.DictReader(fh))
assert len(clave) == 265

for nombre, prefijo in (("annotator_A.csv", "7dfd5b898d3905f2"),
                        ("annotator_B.csv", "956646fa479f0c65")):
    hh = hashlib.sha256(
        (PROJECT_ROOT / "artifacts" / "human_annotation_v1" / nombre).read_bytes()).hexdigest()
    assert hh.startswith(prefijo), f"v1 {nombre} cambio"

sucios = [n for n in entregables
          if any(x > 127 for x in (PROJECT_ROOT / "reports" / n).read_bytes())]
assert not sucios, f"bytes > 127 en {sucios}"

print("P92-W (holdout2 auditado, paquete hold-safe) verificado:")
print("  poblacion dev2 1236 | paquete A/B/calibracion 250/250/15 vacios | ordenes distintos")
print("  v2 en cuarentena | influencia indirecta registrada | NOT_DISTRIBUTED")'''


def construir_seccion() -> str:
    """Construye el markdown de la seccion 144."""
    m = json.loads((RAIZ / "reports" / "p92w_sampling_manifest.json").read_text(encoding="utf-8"))
    prec = m["expected_sampling_precision"]
    return f"""## 144. P92-W - HOLDOUT2 AUDITADO: PAQUETE HOLD-SAFE RECONSTRUIDO (2026-10-13)

**Correccion de procedencia; sin entrenamiento, sin predicciones, sin distribucion.** La auditoria formal confirmo que el muestreo de P92 declaro excluir holdout2 pero su marco efectivo (dev 1483) lo contenia: **35 filas principales + 3 de calibracion** expuestas. Los paquetes `human_annotation_v2` quedan `QUARANTINED_NOT_FOR_DISTRIBUTION_PENDING_HOLDOUT2_AUDIT` (aviso aditivo; sin modificar artefactos). `human_annotation_v1` tambien queda fuera de distribucion (mismas filas, historico).

**Exposicion clasificada**: texto DEMOSTRADO (38 filas); anotacion humana DEMOSTRADA en 3 casos de calibracion y NO OCURRIDA en los 35; labels/scores: no leidos por fila en P92, uso historico agregado NO VERIFICABLE por fila; metricas sobre holdout2: NO OCURRIDAS en P92. Registrado `GUIDELINE_POTENTIAL_INDIRECT_HOLDOUT2_INFLUENCE`: 2 de los 3 casos expuestos (P92C-14, P92C-15) eran evidencia central de la guia v2; con los 12 casos elegibles el patron se conserva (UTD coincidente C/D en HS: 4/12), lo que sostiene la direccion operativa sin declarar independencia plena.

**Poblacion reconstruida**: dev2 = 1236 (dev 1483 menos holdout2 247; sin test, sin holdout historico; ninguna fila sin particion demostrable). Muestreo nuevo con semillas registradas (A {m['random_seeds']['A']}, B {m['random_seeds']['B']}, calibracion {m['random_seeds']['calibracion']}); A = SRS 100/1236 (probabilidad 100/1236; margenes recalculados: {prec['por_etiqueta_p005']}, peor caso {prec['por_etiqueta_p050']}); B = 150 por tiers con cap 2 por grupo; calibracion nueva = 15.

**Paquete** `human_annotation_v2_1_holdout_safe/` (A/B + calibration + manifest + README): 16/16 verificaciones (elegibilidad, exclusiones, componentes disjuntos, textos 515/515, cegamiento, ASCII), misma guia congelada, NOT distribuido.

**Estado**: `OLD_P92V_PACKAGES = QUARANTINED` + `NEW_HUMAN_ANNOTATION_PACKAGE = READY_WITH_VERIFIED_EXCLUSIONS` + `MAIN_ANNOTATION = NOT_DISTRIBUTED` + `V03 = FROZEN` + `NO NEW MODELING PHASE`. Sin commit; hashes historicos intactos.

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
    exec(compile(CODIGO, "<seccion 144>", "exec"), espacio)


def main() -> None:
    """Integra la seccion 144, ejecuta su verificacion y guarda el notebook."""
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
    print("Notebook actualizado con la seccion 144:")
    print(f"  celdas antes: {celdas_antes} | despues: {len(cells)}")
    print("  reconstruccion P92-W verificada en aislamiento: OK")
    print("  celdas de codigo compiladas: OK | ASCII puro en lo generado: OK (0 bytes > 127)")


if __name__ == "__main__":
    main()
