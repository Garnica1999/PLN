"""P90-01: integra la seccion 135 (V03 FAILURE CASEBOOK) en el notebook.

Lee los entregables del casebook (`reports/v03_error_*`) y construye una seccion
compacta con las metricas de auditoria y una muestra representativa de ejemplos.
La tabla completa permanece en los CSV; el notebook no inserta cientos de tweets.

Idempotente y ASCII puro. La seccion declara expresamente que el analisis es
descriptivo y no constituye un experimento nuevo.

Uso:
    /home/carlos/venv/bin/python experiments/p90_01_integrar_casebook.py
"""

from __future__ import annotations

import csv
import json
import uuid
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
RUTA_NOTEBOOK = RAIZ / "notebooks" / "10_final_pipeline.ipynb"
RUTA_FOLD = RAIZ / "reports" / "v03_error_fold_summary.csv"
RUTA_CASEBOOK = RAIZ / "reports" / "v03_error_casebook.csv"
RUTA_CELLS = RAIZ / "reports" / "v03_error_cells.csv"
RUTA_SAMPLE = RAIZ / "reports" / "v03_error_review_sample.csv"
RUTA_CONF = RAIZ / "reports" / "v03_error_confidence_summary.csv"
RUTA_PROV = RAIZ / "reports" / "v03_error_provenance.json"
MARCA_SECCION = "## 135. V03 FAILURE CASEBOOK"
MARCA_CODIGO = "Verificacion de la seccion 135 (P90-00)"


def leer_csv(ruta: Path) -> list[dict[str, str]]:
    """Lee un CSV como lista de diccionarios."""
    return list(csv.DictReader(ruta.read_text(encoding="utf-8").splitlines()))


def nueva_celda_markdown(texto: str) -> dict:
    """Crea una celda de markdown con identificador unico."""
    return {"cell_type": "markdown", "id": uuid.uuid4().hex[:8], "metadata": {},
            "source": texto.splitlines(keepends=True)}


def nueva_celda_codigo(texto: str) -> dict:
    """Crea una celda de codigo vacia de salidas y sin ejecutar."""
    return {"cell_type": "code", "id": uuid.uuid4().hex[:8], "metadata": {},
            "execution_count": None, "outputs": [], "source": texto.splitlines(keepends=True)}


CODIGO = '''"""Verificacion de la seccion 135 (P90-00, casebook de errores de V03).

Comprueba los entregables del casebook, sus recuentos, el hash del OOF en la
provenance y el escaneo byte a byte. Analisis descriptivo: no entrena nada.
"""

import csv
import json

entregables = [
    "v03_error_casebook.md", "v03_error_casebook.csv", "v03_error_cells.csv",
    "v03_error_fold_summary.csv", "v03_error_slice_summary.csv",
    "v03_error_confidence_summary.csv", "v03_error_review_sample.csv",
    "v03_error_provenance.json",
]
for nombre in entregables:
    ruta = PROJECT_ROOT / "reports" / nombre
    assert ruta.exists(), f"falta {nombre}"

casebook = list(csv.DictReader((PROJECT_ROOT / "reports" / "v03_error_casebook.csv").read_text(encoding="utf-8").splitlines()))
cells = list(csv.DictReader((PROJECT_ROOT / "reports" / "v03_error_cells.csv").read_text(encoding="utf-8").splitlines()))
assert len(casebook) == 441, f"el casebook debe tener 441 tweets con error, tiene {len(casebook)}"
assert len(cells) == 757, f"el catalogo debe tener 757 celdas, tiene {len(cells)}"

prov = json.loads((PROJECT_ROOT / "reports" / "v03_error_provenance.json").read_text(encoding="utf-8"))
assert prov["oof_sha256"] == "559265816ce4f6312c87f008f90324f0eec70a74d1e6247364c162793338feab"
assert prov["row_count"] == 1483 and prov["unique_tweet_ids"] == 1483

sucios = [n for n in entregables if any(b > 127 for b in (PROJECT_ROOT / "reports" / n).read_bytes())]
assert not sucios, f"bytes > 127 en {sucios}"

print("P90-00 (casebook de errores de V03) verificado:")
print("  casebook 441 tweets | catalogo 757 celdas | OOF 55926581 verificado | 0 bytes > 127")'''


def construir_seccion() -> str:
    """Construye el markdown de la seccion 135 a partir de los entregables."""
    folds = leer_csv(RUTA_FOLD)
    casebook = leer_csv(RUTA_CASEBOOK)
    sample = leer_csv(RUTA_SAMPLE)
    conf = leer_csv(RUTA_CONF)
    prov = json.loads(RUTA_PROV.read_text(encoding="utf-8"))

    totales: dict[str, dict[str, int]] = {}
    for fila in folds:
        t = totales.setdefault(fila["label"], {"TP": 0, "FP": 0, "FN": 0, "TN": 0})
        for clave in ("TP", "FP", "FN", "TN"):
            t[clave] += int(fila[clave])

    def f1(t: dict[str, int]) -> float:
        precision = t["TP"] / (t["TP"] + t["FP"]) if (t["TP"] + t["FP"]) else 0.0
        recall = t["TP"] / (t["TP"] + t["FN"]) if (t["TP"] + t["FN"]) else 0.0
        return 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    filas_metrica = "\n".join(
        f"| {et} | {totales[et]['TP']} | {totales[et]['FP']} | {totales[et]['FN']} | {totales[et]['TN']} | "
        f"{f1(totales[et]):.4f} | {totales[et]['FP'] + totales[et]['FN']} |"
        for et in ("HS", "TR", "AG")
    )
    n_una = sum(1 for t in casebook if t["n_wrong_labels"] == "1")
    n_dos = sum(1 for t in casebook if t["n_wrong_labels"] == "2")
    n_tres = sum(1 for t in casebook if t["n_wrong_labels"] == "3")
    alta = {r["label"]: r["n"] for r in conf if r["error_type"] == "TODOS" and r["bin"] == "alta_confianza_delta>=0.10"}
    ejemplos = [f for f in sample if f["tipo"] == "error"]
    ejemplos = sorted(ejemplos, key=lambda f: f["tweet_id"])[:8]
    filas_ejemplos = "\n".join(
        f"| {e['tweet_id']} | f{e['fold']} | {e['error_pattern']} | {e['text'][:72]} |" for e in ejemplos
    )
    return f"""## 135. V03 FAILURE CASEBOOK: inventario descriptivo de errores (2026-10-10)

**Analisis descriptivo; NO es un experimento nuevo.** Sin entrenamiento, sin ajuste de umbrales, sin test, sin holdout y sin submissions. Fuente: OOF canonico `features/oof_v03_final.npz` (sha256 `{prov['oof_sha256'][:16]}...`), {prov['row_count']} filas de desarrollo, umbrales congelados 0.44/0.51/0.46, pliegues del protocolo D-016. La taxonomia v4 historica se reproduce exactamente (13 categorias).

| etiqueta | TP | FP | FN | TN | F1 | errores |
|---|---:|---:|---:|---:|---:|---:|
{filas_metrica}

**Errores conjuntos**: {len(casebook)} tweets con al menos un error ({n_una} con una etiqueta, {n_dos} con dos, {n_tres} con las tres) y {len(leer_csv(RUTA_CELLS))} celdas tweet-etiqueta. Errores con distancia al umbral >= 0.10 (alta confianza historica): HS {alta.get('HS', 0)}, TR {alta.get('TR', 0)}, AG {alta.get('AG', 0)}.

**Muestra representativa** (8 de los 100 tweets de `v03_error_review_sample.csv`; la tabla completa vive en los CSV):

| tweet_id | fold | patron | texto (truncado) |
|---|---|---|---|
{filas_ejemplos}

**Patrones tentativos** (detalle con casos y contraejemplos en `v03_error_casebook.md`): parte de los FP son insultos con marcadores de broma/afiliacion (27/173 FP_HS, 26/135 FP_AG); los FN de AG/HS incluyen hostilidad contra grupos sin lexico de insulto; hay FN de las tres etiquetas en agresion directa de registro vulgar que el modelo puntua muy bajo; el error de TR vive en textos sin destinatario explicito; una parte de los errores esta pegada al umbral. Ninguna observacion se convierte en feature ni se afirma como causa.

**Entregables**: `reports/v03_error_casebook.md`, `v03_error_casebook.csv` (441 tweets), `v03_error_cells.csv` (757 celdas), `v03_error_fold_summary.csv`, `v03_error_slice_summary.csv` (19 slices historicas dentro/fuera), `v03_error_confidence_summary.csv`, `v03_error_review_sample.csv` (100 errores + controles), `v03_error_controls.csv` (757 controles 1:1), `v03_error_provenance.json`.

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


def main() -> None:
    """Integra la seccion 135 y guarda el notebook."""
    seccion = construir_seccion()
    if any(ord(caracter) > 127 for caracter in seccion):
        raise SystemExit("Byte > 127 en el texto generado (seccion)")

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
    print("Notebook actualizado con la seccion 135:")
    print(f"  celdas antes: {celdas_antes} | despues: {len(cells)}")
    print("  casebook 441 tweets | catalogo 757 celdas | analisis descriptivo declarado")
    print("  celdas de codigo compiladas: OK | ASCII puro en lo generado: OK (0 bytes > 127)")


if __name__ == "__main__":
    main()
