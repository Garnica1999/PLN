"""P90-05: integra la subseccion 137R (P90.03-R ATRIBUCION EXACTA OOF) en el notebook.

Lee los entregables `reports/p90_03r_*` y construye una subseccion compacta.
Idempotente y ASCII puro. No se ejecuta el notebook completo: la celda de
verificacion se ejecuta de forma aislada.

Uso:
    /home/carlos/venv/bin/python experiments/p90_05_integrar_notebook.py
"""

from __future__ import annotations

import csv
import json
import uuid
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
RUTA_NOTEBOOK = RAIZ / "notebooks" / "10_final_pipeline.ipynb"
RUTA_RESULTS = RAIZ / "reports" / "p90_03r_audit_results.json"
MARCA_SECCION = "## 137R. P90.03-R"
MARCA_CODIGO = "Verificacion de la subseccion 137R (P90.03-R"


def nueva_celda_markdown(texto: str) -> dict:
    """Crea una celda de markdown con identificador unico."""
    return {"cell_type": "markdown", "id": uuid.uuid4().hex[:8], "metadata": {},
            "source": texto.splitlines(keepends=True)}


def nueva_celda_codigo(texto: str) -> dict:
    """Crea una celda de codigo vacia de salidas y sin ejecutar."""
    return {"cell_type": "code", "id": uuid.uuid4().hex[:8], "metadata": {},
            "execution_count": None, "outputs": [], "source": texto.splitlines(keepends=True)}


CODIGO = '''"""Verificacion de la subseccion 137R (P90.03-R, atribucion exacta OOF).

Comprueba los entregables: existencia, gate de reproduccion, recuentos,
residuo de descomposicion y escaneo byte a byte. No entrena nada nuevo.
"""

import csv
import json

entregables = [
    "p90_03r_exact_oof_attribution.md", "p90_03r_exact_oof_attribution.csv",
    "p90_03r_oof_reproduction.csv", "p90_03r_feature_contributions.csv",
    "p90_03r_manifest.json", "p90_03r_audit_results.json",
]
for nombre in entregables:
    ruta = PROJECT_ROOT / "reports" / nombre
    assert ruta.exists(), f"falta {nombre}"

resultados = json.loads((PROJECT_ROOT / "reports" / "p90_03r_audit_results.json").read_text(encoding="utf-8"))
assert resultados["decision"] == "EXACT_OOF_ATTRIBUTION_RECOVERED"
assert resultados["gate"]["gate_passed"] is True
assert resultados["gate"]["max_abs_error"] == 0.0
assert resultados["gate"]["binary_discordances"] == 0
assert abs(resultados["gate"]["ccv_reconstructed"] - 0.7772114525165721) <= 1e-12
assert resultados["decomposition_residual_max"] < 1e-9

with (PROJECT_ROOT / "reports" / "p90_03r_feature_contributions.csv").open(encoding="utf-8", newline="") as fh:
    contrib = list(csv.DictReader(fh))
assert len(contrib) == 4449
with (PROJECT_ROOT / "reports" / "p90_03r_exact_oof_attribution.csv").open(encoding="utf-8", newline="") as fh:
    errores = list(csv.DictReader(fh))
assert len(errores) == 757
assert all(f["error_type"] in ("FP", "FN") for f in errores)

sucios = [n for n in entregables if any(b > 127 for b in (PROJECT_ROOT / "reports" / n).read_bytes())]
assert not sucios, f"bytes > 127 en {sucios}"

print("P90.03-R (atribucion exacta OOF) verificada:")
print("  reproduccion exacta: max_abs_error 0.0 | 0 discordancias | ccv identico")
print("  4449 filas-etiqueta | 757 errores con atribucion exacta | 0 bytes > 127")'''


def construir_seccion() -> str:
    """Construye el markdown de la subseccion 137R a partir de los entregables."""
    d = json.loads(RUTA_RESULTS.read_text(encoding="utf-8"))
    g = d["gate"]
    ag = d["agregados"]
    casos = d["casos_clave"]
    filas_resumen = "\n".join(
        f"| {et} | {tipo} | {ag[f'{et}|{tipo}']['n']} | {ag[f'{et}|{tipo}']['mean_word']:+.3f} | "
        f"{ag[f'{et}|{tipo}']['mean_char']:+.3f} | {ag[f'{et}|{tipo}']['acumulacion']} | "
        f"{ag[f'{et}|{tipo}']['conflicto_wc']} | {ag[f'{et}|{tipo}']['top1_pos_es_lexico']} |"
        for et in ("HS", "TR", "AG") for tipo in ("FP", "FN")
    )
    return f"""## 137R. P90.03-R: ATRIBUCION EXACTA OOF RECUPERADA (2026-10-10)

**Fase diagnostica autorizada; NO es un experimento de modelado.** P90.03 dejo la atribucion exacta como `BLOCKED_BY_MISSING_ARTIFACTS` porque los modelos/vectorizadores por fold no estaban serializados. P90.03-R reconstruye el pipeline historico de `experiments/14_oof_corregido_v2.py` (vectorizadores word+char ajustados por fold; 3 LogisticRegression C=2.0 balanced liblinear por fold) **exclusivamente para recuperar la atribucion**, sin cambiar V03, thresholds, folds ni protocolo, sin test, sin holdout, sin submissions y sin commits.

**Gate de reproduccion superado (exacto)**: `allclose(rtol=1e-7, atol=1e-8)` = True; error absoluto maximo {g['max_abs_error']:.1e}; discordancias binarias {g['binary_discordances']}; `corrected_cv` reconstruido {g['ccv_reconstructed']:.16f} (diferencia {g['ccv_difference']:.1e}); folds recomputados == persistidos.

**Descomposicion exacta**: `logit = intercepto + suma(contribuciones)` verificada en 4449 filas-etiqueta (residuo maximo {d['decomposition_residual_max']:.2e}); 757 errores con contribuciones exactas por el modelo del fold que realmente los predijo.

| etiqueta | tipo | n | word medio | char medio | acumulacion | conflicto wc | top-1 lexico |
|---|---|---|---|---|---|---|---|
{filas_resumen}

**Casos clave**: "perra" es top-1 positivo en FP {casos['perra_top1_FP']['HS']}/{casos['perra_top1_FP']['TR']}/{casos['perra_top1_FP']['AG']} (HS/TR/AG); "mi perra" en top-5 negativos de errores {casos['mi_perra_en_neg5']['HS']}/{casos['mi_perra_en_neg5']['TR']}/{casos['mi_perra_en_neg5']['AG']}; "la puta" {casos['la_puta_en_neg5']['HS']}/{casos['la_puta_en_neg5']['TR']}/{casos['la_puta_en_neg5']['AG']}. Caso 23839: "perra" +0.5589 y "mi perra" -0.5511 casi se cancelan (FP a 4.6e-05 del threshold).

**Comparacion con P90.03 (auxiliar)**: CONFIRMED_BY_EXACT_OOF: acumulacion, "perra", "la puta"/"mi perra" negativas, URLs/colectivos con mas features activas, firmas FP/FN opuestas. PARTIALLY_CONFIRMED: conflicto word-char en FP de HS (43->35 por ciento). NOT_CONFIRMED: conflicto en FP de AG (50->14 por ciento); TR FN mismo signo 100->71 por ciento. CONTRADICTED: char dominante en FN (62/56 -> 1.4/21 por ciento) y conflicto bajo en FN (17/3 -> 48/56 por ciento): eran artefactos del modelo final in-sample.

**Decision**: **EXACT_OOF_ATTRIBUTION_RECOVERED**. Entregables: `reports/p90_03r_exact_oof_attribution.md`, `p90_03r_exact_oof_attribution.csv` (757), `p90_03r_feature_contributions.csv` (4449), `p90_03r_oof_reproduction.csv` (4449), `p90_03r_manifest.json`.

**Limitaciones de interpretacion**: las contribuciones explican algebraicamente el logit; no demuestran causalidad semantica ni que una feature pueda eliminarse o reponderarse sin dano en otras predicciones; los modelos reconstruidos son instrumento de atribucion (no candidatos ni mejoras); el gate es exacto en este entorno (sklearn 1.8.0).

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
    exec(compile(CODIGO, "<subseccion 137R>", "exec"), espacio)


def main() -> None:
    """Integra la subseccion 137R, ejecuta su verificacion y guarda el notebook."""
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
    print("Notebook actualizado con la subseccion 137R:")
    print(f"  celdas antes: {celdas_antes} | despues: {len(cells)}")
    print("  atribucion exacta P90.03-R verificada en aislamiento: OK")
    print("  celdas de codigo compiladas: OK | ASCII puro en lo generado: OK (0 bytes > 127)")


if __name__ == "__main__":
    main()
