"""P90-03: integra la seccion 136 (AUDITORIA P90-02) en el notebook consolidado.

Lee los entregables de la auditoria (`reports/p90_02_*`) y construye una seccion
compacta con los resultados de integridad, metricas, corpus, slices y sintesis.
La tabla completa permanece en los CSV; el notebook no inserta cientos de filas.

Idempotente y ASCII puro. La seccion declara expresamente que la auditoria es
descriptiva y no constituye un experimento nuevo. No se ejecuta el notebook
completo: la celda de verificacion se ejecuta de forma aislada desde este script.

Uso:
    /home/carlos/venv/bin/python experiments/p90_03_integrar_notebook.py
"""

from __future__ import annotations

import csv
import json
import uuid
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
RUTA_NOTEBOOK = RAIZ / "notebooks" / "10_final_pipeline.ipynb"
RUTA_RESULTS = RAIZ / "reports" / "p90_02_audit_results.json"
RUTA_METRIC = RAIZ / "reports" / "p90_02_metric_reconciliation.csv"
RUTA_CB_VAL = RAIZ / "reports" / "p90_02_casebook_validation.csv"
RUTA_REGISTRY = RAIZ / "reports" / "p90_02_slice_registry.csv"
RUTA_HYP = RAIZ / "reports" / "p90_02_hypothesis_status.csv"
MARCA_SECCION = "## 136. AUDITORIA INTEGRAL P90-02"
MARCA_CODIGO = "Verificacion de la seccion 136 (P90-02)"


def leer_csv(ruta: Path) -> list[dict[str, str]]:
    """Lee un CSV como lista de diccionarios (respetando saltos embebidos)."""
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


CODIGO = '''"""Verificacion de la seccion 136 (P90-02, auditoria integral de V03 y del corpus).

Comprueba los entregables de la auditoria: existencia, recuentos clave,
reconciliacion de metricas, validacion del casebook, familias de slices,
estado de hipotesis y escaneo byte a byte. Analisis descriptivo: no entrena nada.
"""

import csv
import json

entregables = [
    "p90_02_integrity_and_corpus_audit.md", "p90_02_chatgpt_handoff.md",
    "p90_02_audit_manifest.json", "p90_02_audit_results.json",
    "p90_02_text_provenance.csv", "p90_02_metric_reconciliation.csv",
    "p90_02_casebook_validation.csv", "p90_02_review_sample_audit.csv",
    "p90_02_corpus_profile.csv", "p90_02_label_distribution.csv",
    "p90_02_slice_registry.csv", "p90_02_slice_metrics.csv",
    "p90_02_slice_overlap.csv", "p90_02_error_diagnosis.csv",
    "p90_02_error_intersections.csv", "p90_02_representative_cases.csv",
    "p90_02_historical_eda_matrix.csv", "p90_02_hypothesis_status.csv",
    "p90_02_artifact_inventory.csv",
]
for nombre in entregables:
    ruta = PROJECT_ROOT / "reports" / nombre
    assert ruta.exists(), f"falta {nombre}"

resultados = json.loads((PROJECT_ROOT / "reports" / "p90_02_audit_results.json").read_text(encoding="utf-8"))
assert resultados["audit_id"] == "P90-02"
assert resultados["texto"]["clasificacion"] == "SOURCE_ALREADY_NORMALIZED"
assert resultados["texto"]["diferencias_casebook"] == 0 and resultados["texto"]["diferencias_celdas"] == 0
assert resultados["metricas"]["discrepancias_abiertas"] == 0
assert resultados["casebook"]["tweets_con_error"] == 441 and resultados["casebook"]["celdas_error"] == 757
assert resultados["casebook"]["controles_incumplen"] == 0
assert resultados["corpus"]["n"] == 1483

with (PROJECT_ROOT / "reports" / "p90_02_metric_reconciliation.csv").open(encoding="utf-8", newline="") as fh:
    metricas = list(csv.DictReader(fh))
ccv = [f for f in metricas if f["metric_name"] == "corrected_cv"]
assert len(ccv) == 1 and ccv[0]["status"] == "REPRODUCED", "corrected_cv no reproducido"

with (PROJECT_ROOT / "reports" / "p90_02_casebook_validation.csv").open(encoding="utf-8", newline="") as fh:
    validacion = list(csv.DictReader(fh))
assert all(f["estado"] in ("OK", "INFORMATIVO") for f in validacion), "validacion con fallos"

with (PROJECT_ROOT / "reports" / "p90_02_slice_registry.csv").open(encoding="utf-8", newline="") as fh:
    registro = list(csv.DictReader(fh))
familias = {f["familia"] for f in registro}
assert len(registro) == 19, f"slices registradas: {len(registro)}"
assert familias == {"SLICE_TEXT_DEFINED", "SLICE_LEXICON_INDUCED", "SLICE_ERROR_CONDITIONED"}

sucios = [n for n in entregables if any(b > 127 for b in (PROJECT_ROOT / "reports" / n).read_bytes())]
assert not sucios, f"bytes > 127 en {sucios}"

print("P90-02 (auditoria integral de V03 y del corpus) verificada:")
print("  texto SOURCE_ALREADY_NORMALIZED | corrected_cv REPRODUCED | casebook 441/757 verificado")
print("  controles 757/757 reproducidos | slices 19 en 3 familias | 0 bytes > 127")'''


def construir_seccion() -> str:
    """Construye el markdown de la seccion 136 a partir de los entregables."""
    d = json.loads(RUTA_RESULTS.read_text(encoding="utf-8"))
    met = d["metricas"]
    cb = d["casebook"]
    corp = d["corpus"]
    diag = d["diagnostico"]
    hyp = leer_csv(RUTA_HYP)
    metricas_csv = leer_csv(RUTA_METRIC)
    ccv_fila = next(f for f in metricas_csv if f["metric_name"] == "corrected_cv")
    validacion_csv = leer_csv(RUTA_CB_VAL)
    rep_ctl = next(f["observado"] for f in validacion_csv if f["chequeo"] == "controles_repetidos")
    from collections import Counter

    estados = Counter(f["estado"].split(":")[0].strip() for f in hyp)
    mp = met["matrices_pooled"]
    filas_metrica = "\n".join(
        f"| {et} | {mp[et]['TP']} | {mp[et]['FP']} | {mp[et]['FN']} | {mp[et]['TN']} | "
        f"{met['f1_por_etiqueta_pooled'][et]:.6f} |"
        for et in ("HS", "TR", "AG")
    )
    filas_casos = "\n".join(
        f"| {c['tweet_id']} | f{c['fold']} | {c['error_pattern']} | {c['distancias'][:58]} |"
        for c in d["casos"][:6]
    )
    return f"""## 136. AUDITORIA INTEGRAL P90-02: integridad, metricas, corpus y trazabilidad (2026-10-10)

**Auditoria descriptiva; NO es un experimento nuevo.** Sin entrenamiento, sin test, sin holdout, sin submissions y sin commits. Fuente: OOF canonico `features/oof_v03_final.npz` (sha256 `{d['hashes']['oof'][:16]}...`), {corp['n']} filas de desarrollo, umbrales congelados 0.44/0.51/0.46. La cadena del texto se verifico byte a byte: raw = casebook = entrada de V03 (0 diferencias), clasificacion `{d['texto']['clasificacion']}` (el corpus llega normalizado a ASCII por los organizadores, D-010; no hay perdida en el pipeline).

**Reconciliacion de metricas**: `corrected_cv` = macro pooled = {met['corrected_cv_reproducido'][:18]} (reproducido con diferencia {ccv_fila['difference']} frente al artefacto); los F1 por etiqueta del OOF coinciden con las matrices pooled; la media entre folds ({met['media_macro_pliegues']}) es un nivel de agregacion DISTINTO y no es la metrica canonica.

| etiqueta | TP | FP | FN | TN | F1 pooled |
|---|---:|---:|---:|---:|---:|
{filas_metrica}

**Casebook validado**: {cb['tweets_con_error']} tweets con error y {cb['celdas_error']} celdas ({cb['una_etiqueta']}/{cb['dos_etiquetas']}/{cb['tres_etiquetas']} con una/dos/tres etiquetas); {cb['tweets_sin_error']} tweets sin error. Controles 1:1: {cb['controles']} reproducidos exactamente (0 incumplimientos; {rep_ctl} reutilizaciones). Muestra de revision: {cb['muestra_errores']} errores + {cb['muestra_controles']} controles (estratificada, no aleatoria).

**Corpus de desarrollo** ({corp['n']} filas): HS {corp['combos']['(1, 0, 0)'] + corp['combos']['(1, 0, 1)'] + corp['combos']['(1, 1, 0)'] + corp['combos']['(1, 1, 1)']} positivos, TR {corp['combos']['(1, 1, 0)'] + corp['combos']['(1, 1, 1)']} y AG {corp['combos']['(0, 0, 1)'] + corp['combos']['(1, 0, 1)'] + corp['combos']['(1, 1, 1)']}; identidad estructural: TR=1 con HS=0 tiene 0 casos. Longitud: {corp['chars']['min']}-{corp['chars']['max']} caracteres (mediana {corp['chars']['mediana']:.0f}); vocabulario {corp['vocab_tipos']} tipos; sin textos vacios ni duplicados exactos; near-duplicados en desarrollo: {corp['no_unitarios']} filas en grupos no unitarios.

**Errores**: {diag['errores_celda']} celdas en {diag['tweets_error']} tweets; {diag['fronterizos']} errores a distancia < 0.025 (frontera) y {diag['lejanos']} a distancia >= 0.10 (61.6 por ciento): reajustar thresholds no puede resolver la mayoria del error.

**Casos representativos verificados** (tabla completa en `p90_02_representative_cases.csv`):

| tweet_id | fold | patron | distancias |
|---|---|---|---|
{filas_casos}

**Slices**: 19 registradas en `p90_02_slice_registry.csv` y reclasificadas en tres familias: {sum(1 for f in leer_csv(RUTA_REGISTRY) if f['familia'] == 'SLICE_TEXT_DEFINED')} de texto, {sum(1 for f in leer_csv(RUTA_REGISTRY) if f['familia'] == 'SLICE_LEXICON_INDUCED')} con lexico inducido in-sample (solo diagnostico) y {sum(1 for f in leer_csv(RUTA_REGISTRY) if f['familia'] == 'SLICE_ERROR_CONDITIONED')} definidas por errores (recuentos y taxonomia; su F1 interno es cero por construccion y no se reporta como rendimiento).

**Sintesis historica**: {len(hyp)} lineas clasificadas ({estados.get('CLOSED', 0)} CLOSED, {estados.get('BLOCKED_BY_RULES', 0)} BLOCKED_BY_RULES, {estados.get('DUPLICATE', 0)} DUPLICATE, {estados.get('INVALIDATED', 0)} INVALIDATED, {estados.get('COMPLEMENTARY', 0)} COMPLEMENTARY). Resultado negativo recurrente: mejoras locales de CV que no transfieren al leaderboard (4 casos enviados). No se identifica una hipotesis de modelado genuinamente nueva dentro de las reglas vigentes.

**Conclusiones**: V03 se mantiene FROZEN; la fase de modelado permanece cerrada; la recomendacion es CLOSE_FAILURE_ANALYSIS. Entregables: `reports/p90_02_integrity_and_corpus_audit.md` (informe integral) y `reports/p90_02_chatgpt_handoff.md` (segunda revision independiente).

> V03 remains the best verified leaderboard result."""


def validar(cells: list[dict]) -> None:
    """Compila las celdas de codigo y exige ASCII puro en lo generado."""
    for indice, celda in enumerate(cells):
        fuente = "".join(celda.get("source", []))
        if celda["cell_type"] == "code":
            compile(fuente, f"<celda {indice}>", "exec")
    if any(ord(caracter) > 127 for caracter in CODIGO):
        raise SystemExit("Byte > 127 en el texto generado (codigo)")
    if any(byte > 127 for byte in Path(__file__).read_bytes()):
        raise SystemExit("El propio script contiene bytes > 127")


def ejecutar_verificacion() -> None:
    """Ejecuta la celda de verificacion de forma aislada (solo lee reports/)."""
    espacio: dict[str, object] = {"PROJECT_ROOT": RAIZ, "__name__": "__verificacion__"}
    exec(compile(CODIGO, "<seccion 136>", "exec"), espacio)


def main() -> None:
    """Integra la seccion 136, ejecuta su verificacion y guarda el notebook."""
    seccion = construir_seccion()
    if any(ord(caracter) > 127 for caracter in seccion):
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
    print("Notebook actualizado con la seccion 136:")
    print(f"  celdas antes: {celdas_antes} | despues: {len(cells)}")
    print("  auditoria P90-02 verificada en aislamiento: OK")
    print("  celdas de codigo compiladas: OK | ASCII puro en lo generado: OK (0 bytes > 127)")


if __name__ == "__main__":
    main()
