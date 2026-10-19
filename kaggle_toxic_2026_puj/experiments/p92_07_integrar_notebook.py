"""P92-07: integra la seccion 140 (P92-T CALIBRACION ANALIZADA) en el notebook.

Lee los entregables `reports/p92_*` y construye una subseccion compacta. Idempotente y
ASCII puro. La celda de verificacion se ejecuta de forma aislada (solo lee archivos del
proyecto).

Uso:
    /home/carlos/venv/bin/python experiments/p92_07_integrar_notebook.py
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
RUTA_NOTEBOOK = RAIZ / "notebooks" / "10_final_pipeline.ipynb"
RUTA_RESULTS = RAIZ / "reports" / "p92t_audit_results.json"
MARCA_SECCION = "## 140. P92-T - CALIBRACION ANALIZADA"
MARCA_CODIGO = "Verificacion de la seccion 140 (P92-T"


def nueva_celda_markdown(texto: str) -> dict:
    """Crea una celda de markdown con identificador unico."""
    return {"cell_type": "markdown", "id": uuid.uuid4().hex[:8], "metadata": {},
            "source": texto.splitlines(keepends=True)}


def nueva_celda_codigo(texto: str) -> dict:
    """Crea una celda de codigo vacia de salidas y sin ejecutar."""
    return {"cell_type": "code", "id": uuid.uuid4().hex[:8], "metadata": {},
            "execution_count": None, "outputs": [], "source": texto.splitlines(keepends=True)}


CODIGO = '''"""Verificacion de la seccion 140 (P92-T, calibracion analizada).

Comprueba los entregables del analisis de calibracion, la clasificacion de la
codificacion A/B, el acuerdo registrado y la integridad (hashes y estados).
No entrena nada.
"""

import csv
import json
from pathlib import Path

entregables = [
    "p92_calibration_analysis.md", "p92_calibration_agreement.csv",
    "p92_calibration_case_review.csv", "p92_annotation_guidelines_v2_proposal.md",
    "p92_calibration_discussion_pack.md", "p92t_audit_results.json",
]
for nombre in entregables:
    assert (PROJECT_ROOT / "reports" / nombre).exists(), f"falta {nombre}"

d = json.loads((PROJECT_ROOT / "reports" / "p92t_audit_results.json").read_text(encoding="utf-8"))
assert d["codificacion_AB"]["clasificacion"] == "ENCODING_CONFIRMED"
assert d["acuerdo"]["por_etiqueta"]["HS"]["pares"]["CD"]["acuerdo_3cat"]["num"] == 12
assert d["acuerdo"]["por_etiqueta"]["TR"]["pares"]["CD"]["acuerdo_3cat"]["num"] == 14
assert d["acuerdo"]["por_etiqueta"]["AG"]["pares"]["CD"]["acuerdo_3cat"]["num"] == 11
assert d["acuerdo"]["combinacion_exacta_global"]["num"] == 1
claves = ["raw_annotation_files_unchanged", "original_labels_unchanged",
          "original_texts_unchanged", "calibration_sample_unchanged",
          "main_sample_not_distributed", "v03_unchanged", "canonical_oof_unchanged",
          "holdouts_untouched", "test_not_accessed", "leaderboard_not_used",
          "no_models_trained", "no_predictions_generated", "no_new_submissions",
          "no_pseudo_labeling", "no_external_classifier", "no_external_labeled_dataset"]
assert all(d["integridad"][k] for k in claves), "check de integridad con fallos"

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

print("P92-T (calibracion analizada) verificado:")
print("  codificacion A/B: ENCODING_CONFIRMED | CD 12/15, 14/15, 11/15 | exacto 4: 1/15")
print("  16/16 checks de integridad | hashes de respuestas intactos | ASCII puro")
print("  guia v2: PROPUESTA pendiente de discusion | muestra principal NO distribuida")'''


def construir_seccion() -> str:
    """Construye el markdown de la seccion 140 a partir de los resultados reales."""
    d = json.loads(RUTA_RESULTS.read_text(encoding="utf-8"))
    a = d["acuerdo"]
    hs = a["por_etiqueta"]["HS"]
    tr = a["por_etiqueta"]["TR"]
    ag = a["por_etiqueta"]["AG"]
    return f"""## 140. P92-T - CALIBRACION ANALIZADA Y GUIA V2 PROPUESTA (2026-10-12)

**Fase diagnostica; sin entrenamiento, sin predicciones, sin modificacion de etiquetas.** La calibracion de 15 tweets con 4 anotadores fue analizada. Estado: `CALIBRATION_ANALYZED` + `GUIDELINE_V2_PROPOSED_PENDING_DISCUSSION` + `HUMAN_DISCUSSION_PENDING` + `MAIN_ANNOTATION_NOT_DISTRIBUTED` + `V03 = FROZEN` + `NO NEW MODELING PHASE`.

**Codificacion A/B**: `ENCODING_CONFIRMED` (aclaracion explicita del responsable del proyecto, 2026-10-12): en columnas de clase 0 = NEGATIVE y 1 = POSITIVE; en notas 1 = dificil de determinar para cualquier clase (flag de fila; NO se convierte en UNABLE_TO_DETERMINE por etiqueta). La coincidencia con C/D no se uso como fundamento.

**Acuerdo (respuestas interpretadas, n=15)**: exacto por etiqueta, par C-D = {hs['pares']['CD']['acuerdo_3cat']['num']}/15 (HS), {tr['pares']['CD']['acuerdo_3cat']['num']}/15 (TR), {ag['pares']['CD']['acuerdo_3cat']['num']}/15 (AG); kappa binario C-D = 1.000 en las tres (sin discrepancias binarias); combinacion exacta de los 4 = {a['combinacion_exacta_global']['num']}/15; alpha de Krippendorff (UTD categoria / faltante) = HS {hs['alpha_utd_categoria']:.3f}/{hs['alpha_utd_faltante']:.3f}, TR {tr['alpha_utd_categoria']:.3f}/{tr['alpha_utd_faltante']:.3f}, AG {ag['alpha_utd_categoria']:.3f}/{ag['alpha_utd_faltante']:.3f} (reconciliadas con P92-S2).

**Hallazgo principal**: 6/15 filas de la muestra con UTD coincidente de C y D en HS (P92C-05/09/11/13/14/15): la guia v1 no define cuando una ofensa sexista individual constituye discurso de odio. Casos adicionales: AG (agresivo vs. vulgar: 06, 13), contexto insuficiente (01, 04, 07) y flag de fila sin motivo (02).

**Entregables**: `p92_calibration_analysis.md`, `p92_calibration_agreement.csv`, `p92_calibration_case_review.csv`, `p92_annotation_guidelines_v2_proposal.md` (propuesta, 10 secciones + changelog v1 -> v2), `p92_calibration_discussion_pack.md`, `p92t_audit_results.json`. Integridad: 16/16 checks en verde; hashes canonicos intactos; respuestas humanas sin cambios.

**Siguiente paso**: discusion humana (paquete) -> decisiones registradas -> decidir v1 vs v2 y congelar la guia con version y hash -> distribuir la muestra principal (250 tweets) solo entonces.

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
    exec(compile(CODIGO, "<seccion 140>", "exec"), espacio)


def main() -> None:
    """Integra la seccion 140, ejecuta su verificacion y guarda el notebook."""
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
    print("Notebook actualizado con la seccion 140:")
    print(f"  celdas antes: {celdas_antes} | despues: {len(cells)}")
    print("  calibracion P92-T verificada en aislamiento: OK")
    print("  celdas de codigo compiladas: OK | ASCII puro en lo generado: OK (0 bytes > 127)")


if __name__ == "__main__":
    main()
