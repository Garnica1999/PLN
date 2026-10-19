"""P90-04: integra la seccion 137 (P90-03 CONTRIBUCIONES) en el notebook consolidado.

Lee los entregables de `reports/p90_03_*` y construye una seccion compacta con los
resultados del diagnostico de contribuciones. Idempotente y ASCII puro. No se ejecuta
el notebook completo: la celda de verificacion se ejecuta de forma aislada.

Uso:
    /home/carlos/venv/bin/python experiments/p90_04_integrar_notebook.py
"""

from __future__ import annotations

import csv
import json
import uuid
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
RUTA_NOTEBOOK = RAIZ / "notebooks" / "10_final_pipeline.ipynb"
RUTA_RESULTS = RAIZ / "reports" / "p90_03_audit_results.json"
RUTA_SUMMARY = RAIZ / "reports" / "p90_03_feature_contribution_summary.csv"
RUTA_GAP = RAIZ / "reports" / "p90_03_hypothesis_gap_analysis.csv"
MARCA_SECCION = "## 137. V03 FEATURE CONTRIBUTION"
MARCA_CODIGO = "Verificacion de la seccion 137 (P90-03)"


def leer_csv(ruta: Path) -> list[dict[str, str]]:
    """Lee un CSV como lista de diccionarios."""
    with ruta.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def nueva_celda_markdown(texto: str) -> dict:
    """Crea una celda de markdown con identificador unico."""
    return {"cell_type": "markdown", "id": uuid.uuid4().hex[:8], "metadata": {},
            "source": texto.splitlines(keepends=True)}


def nueva_celda_codigo(texto: str) -> dict:
    """Crea una celda de codigo vacia de salidas y sin ejecutar."""
    return {"cell_type": "code", "id": uuid.uuid4().hex[:8], "metadata": {},
            "execution_count": None, "outputs": [], "source": texto.splitlines(keepends=True)}


CODIGO = '''"""Verificacion de la seccion 137 (P90-03, contribuciones de features de V03).

Comprueba los entregables del diagnostico: existencia, validacion matematica,
recuentos clave, gap analysis y escaneo byte a byte. No entrena nada.
"""

import csv
import json

entregables = [
    "p90_03_feature_contribution_diagnostic.md", "p90_03_chatgpt_handoff.md",
    "p90_03_artifact_manifest.json", "p90_03_audit_results.json",
    "p90_03_error_contribution_by_cell.csv", "p90_03_feature_contribution_summary.csv",
    "p90_03_error_signature_profiles.csv", "p90_03_error_control_comparison.csv",
    "p90_03_slice_contribution_summary.csv", "p90_03_hypothesis_gap_analysis.csv",
    "p90_03_hypothesis_candidates.csv", "p90_03_key_feature_associations.csv",
]
for nombre in entregables:
    ruta = PROJECT_ROOT / "reports" / nombre
    assert ruta.exists(), f"falta {nombre}"

resultados = json.loads((PROJECT_ROOT / "reports" / "p90_03_audit_results.json").read_text(encoding="utf-8"))
assert resultados["audit_id"] == "P90-03"
assert resultados["disponibilidad"]["atribucion_exacta_oof"] == "BLOCKED_BY_MISSING_ARTIFACTS"
assert resultados["disponibilidad"]["atribucion_auxiliar"] == "AUXILIARY_FULL_MODEL_ATTRIBUTION"
assert resultados["validacion"]["sigmoid_vs_predict_proba_max"] == 0.0
assert resultados["validacion"]["reconstruccion_z_max_residuo"] < 1e-12
assert resultados["celdas"] == 757
assert resultados["controles"]["relaciones"] == 757

with (PROJECT_ROOT / "reports" / "p90_03_error_contribution_by_cell.csv").open(encoding="utf-8", newline="") as fh:
    celdas = list(csv.DictReader(fh))
assert len(celdas) == 757
assert all(float(f["reconstruction_error"]) < 1e-12 for f in celdas)

with (PROJECT_ROOT / "reports" / "p90_03_hypothesis_gap_analysis.csv").open(encoding="utf-8", newline="") as fh:
    gap = list(csv.DictReader(fh))
assert len(gap) == 18
assert sum(1 for f in gap if f["final_decision"] == "CLOSED") == 15
assert sum(1 for f in gap if f["final_decision"] == "DUPLICATE") == 2

sucios = [n for n in entregables if any(b > 127 for b in (PROJECT_ROOT / "reports" / n).read_bytes())]
assert not sucios, f"bytes > 127 en {sucios}"

print("P90-03 (contribuciones de features de V03) verificada:")
print("  atribucion exacta BLOCKED | auxiliar validada (sigmoid 0.0, residuo < 1e-12)")
print("  757 celdas | 18 hipotesis (15 CLOSED, 2 DUPLICATE) | 0 bytes > 127")'''


def construir_seccion() -> str:
    """Construye el markdown de la seccion 137 a partir de los entregables."""
    d = json.loads(RUTA_RESULTS.read_text(encoding="utf-8"))
    v = d["validacion"]
    ag = d["agregados"]
    firmas = d["firmas"]
    resumen = leer_csv(RUTA_SUMMARY)
    gap = leer_csv(RUTA_GAP)
    from collections import Counter

    estados = Counter(f["final_decision"] for f in gap)
    filas_resumen = "\n".join(
        f"| {f['label']} | {f['error_type']} | {f['n_errors']} | {f['near_errors']}/{f['intermediate_errors']}/{f['far_errors']} | "
        f"{f['mean_abs_word_contribution']} | {f['mean_abs_char_contribution']} | {f['dominant_feature_families'][:60]} |"
        for f in resumen
    )
    return f"""## 137. V03 FEATURE CONTRIBUTION: contribuciones, mecanismos y gap analysis (2026-10-10)

**Diagnostico descriptivo; NO es un experimento nuevo.** Sin entrenamiento, sin test, sin holdout, sin submissions y sin commits. Los modelos y vectorizadores **por pliegue** que generaron el OOF no estan serializados (`experiments/14_oof_corregido_v2.py` los ajusta en memoria): la atribucion exacta OOF queda `BLOCKED_BY_MISSING_ARTIFACTS`. El joblib final de V03 se uso como **evidencia auxiliar** separada: `sigmoid(z) == predict_proba` exacto ({v['sigmoid_vs_predict_proba_max']:.1e}) y reconstruccion `intercepto + suma(contribuciones) == z` con residuo maximo {v['reconstruccion_z_max_residuo']:.1e}. El modelo final coincide en direccion con la decision OOF en solo 1.4-32.9 por ciento de las celdas de error: las contribuciones por celda NO se interpretan como causa del error OOF.

**Hallazgos estructurales (auxiliares)**: el error es **acumulativo** (0 celdas "pocas features dominantes" entre los 466 errores lejanos; 76-86 por ciento de los FN son acumulacion); **conflicto word-char** en {ag['HS|FP']['conflicto_wc']}/{ag['TR|FP']['conflicto_wc']}/{ag['AG|FP']['conflicto_wc']} FP de HS/TR/AG; **canal char dominante** en {ag['HS|FN']['char_dominante']}/143 y {ag['AG|FN']['char_dominante']}/142 FN de HS/AG; los FP estan encabezados por el lexico de misoginia (perra/zorra/callate; top-1 del lexico en {ag['HS|FP']['top1_pos_es_lexico']}/173 HS, {ag['TR|FP']['top1_pos_es_lexico']}/82 TR, {ag['AG|FP']['top1_pos_es_lexico']}/135 AG); los FN acumulan evidencia negativa de palabras funcionales, https y tokens con asociacion anti-toxica aprendida (rabe, acoso, hijo, la puta).

| etiqueta | tipo | n | cerca/intermedio/lejos | |word| | |char| | dominantes |
|---|---|---|---|---|---|---|
{filas_resumen}

**Gap analysis**: 18 mecanismos evaluados contra el historial: {estados.get('CLOSED', 0)} CLOSED, {estados.get('DUPLICATE', 0)} DUPLICATE, {estados.get('UNRESOLVED', 0)} UNRESOLVED, 0 POTENTIALLY_NEW. Precedentes directos: HIB-H11/H12/H14, HIB-UNION, ENS-01..04, REP-05/06, contextual_selection, reweighting, structural, augmentation, cita/negacion, P68. **Ninguna hipotesis genuinamente nueva** (archivo de candidatos con fila NO_CANDIDATE).

**Decision**: **D. EXACT_OOF_ATTRIBUTION_BLOCKED + A. NO_NEW_HYPOTHESIS**. V03 FROZEN. Entregables: `reports/p90_03_feature_contribution_diagnostic.md`, `reports/p90_03_chatgpt_handoff.md` y las tablas `p90_03_*.csv`.

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
    exec(compile(CODIGO, "<seccion 137>", "exec"), espacio)


def main() -> None:
    """Integra la seccion 137, ejecuta su verificacion y guarda el notebook."""
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
    print("Notebook actualizado con la seccion 137:")
    print(f"  celdas antes: {celdas_antes} | despues: {len(cells)}")
    print("  diagnostico P90-03 verificado en aislamiento: OK")
    print("  celdas de codigo compiladas: OK | ASCII puro en lo generado: OK (0 bytes > 127)")


if __name__ == "__main__":
    main()
