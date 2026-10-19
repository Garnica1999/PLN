"""Integra la seccion 132 (P86-00: auditoria de novedad FM, DUPLICATE) en el notebook.

Lee `reports/toxic_fm_novelty_audit.md` y `.csv` y construye la seccion 132 con el
resultado de la auditoria (clasificacion DUPLICATE, precedente V25-01/V25-02, numeros
clave) mas una celda de verificacion que comprueba los entregables, la fila P86-00 del
registro, la clasificacion y el escaneo byte a byte.

Idempotente y ASCII puro (politica de generacion del proyecto).

Uso:
    /home/carlos/venv/bin/python experiments/p86_01_integrar_notebook.py
"""

from __future__ import annotations

import csv
import json
import uuid
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
RUTA_NOTEBOOK = RAIZ / "notebooks" / "10_final_pipeline.ipynb"
RUTA_MD = RAIZ / "reports" / "toxic_fm_novelty_audit.md"
RUTA_CSV = RAIZ / "reports" / "toxic_fm_novelty_audit.csv"
MARCA_SECCION = "## 132. P86-00"
MARCA_CODIGO = "Verificacion de la seccion 132 (P86-00)"


def nueva_celda_markdown(texto: str) -> dict:
    """Crea una celda de markdown con identificador unico."""
    return {
        "cell_type": "markdown",
        "id": uuid.uuid4().hex[:8],
        "metadata": {},
        "source": texto.splitlines(keepends=True),
    }


def nueva_celda_codigo(texto: str) -> dict:
    """Crea una celda de codigo vacia de salidas y sin ejecutar."""
    return {
        "cell_type": "code",
        "id": uuid.uuid4().hex[:8],
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": texto.splitlines(keepends=True),
    }


CODIGO = '''"""Verificacion de la seccion 132 (P86-00).

Comprueba los entregables de la auditoria, la fila P86-00 del registro, la
clasificacion DUPLICATE y que los ficheros generados son ASCII puro.
"""

import csv

ruta_md = PROJECT_ROOT / "reports" / "toxic_fm_novelty_audit.md"
ruta_csv = PROJECT_ROOT / "reports" / "toxic_fm_novelty_audit.csv"
for ruta in (ruta_md, ruta_csv):
    assert ruta.exists(), f"falta {ruta.name}"

texto = ruta_md.read_text(encoding="utf-8")
assert "DUPLICATE" in texto
assert "V25-01 / V25-02" in texto
assert "0.7826986101342172" in texto

filas = list(csv.DictReader(ruta_csv.read_text(encoding="utf-8").splitlines()))
assert filas[0]["clasificacion"] == "DUPLICATE"

registro = list(csv.DictReader((PROJECT_ROOT / "experiments.csv").read_text(encoding="utf-8-sig").splitlines()))
fila = [f for f in registro if f["experiment_id"] == "P86-00"]
assert fila, "falta la fila P86-00"
assert fila[0]["decision"].startswith("DUPLICATE")

sucios = [ruta.name for ruta in (ruta_md, ruta_csv) if any(b > 127 for b in ruta.read_bytes())]
assert not sucios, f"bytes > 127 en {sucios}"

print("P86-00 (auditoria de novedad FM) verificado:")
print("  clasificacion DUPLICATE | precedente V25-01/V25-02 | fila P86-00 OK | 0 bytes > 127")'''


def construir_seccion() -> str:
    """Devuelve el markdown de la seccion 132 a partir de los entregables.

    Returns:
        Texto markdown de la seccion.
    """
    texto = RUTA_MD.read_text(encoding="utf-8")
    if "DUPLICATE" not in texto:
        raise SystemExit("STOP: el informe no declara DUPLICATE")
    return """## 132. P86-00: auditoria de novedad - la Factorization Machine ya esta cubierta (DUPLICATE)

La directiva del 2026-10-08 propuso una FM dispersa con interacciones de bajo rango entre caracteristicas lexicas sobre la representacion word+char de V03, con su protocolo. La auditoria P86-00 (informe `reports/toxic_fm_novelty_audit.md`) encontro el precedente metodologicamente equivalente: **V25-01 / V25-02 (2026-09-27)**.

| elemento | V25-01 / V25-02 (ejecutado) |
|---|---|
| mecanismo | FM desde cero, interaccion de segundo orden de bajo rango, O(nnz*k), pares nunca materializados |
| representacion | word(1,2) + char_wb(3,5) TF-IDF de V03 (min_df=2, sublinear_tf=True) |
| protocolo | pliegues agrupados D-016; umbrales [0.44, 0.51, 0.46]; el control reproduce V03 bit a bit (0.7772114525165721, diferencia 0.0) |
| dimension latente | k en {1, 4, 8, 16}: k=8 y k=16 identicos a k=4 (rango efectivo agotado en 4) |
| regularizacion | lambda_v de 0.25 a 1e6: optimo exacto en 1, sin margen |
| mejor resultado | k=4, lambda_v=1: corrected_cv 0.782699 (+0.005487), neto -40 (119 rescates / 159 danos), dLB esperado -0.0079 |

Cierre historico: `NO CANDIDATE JUSTIFIES SUBMISSION - CLOSE V25` (v25_04) y `CERRAR EL PROYECTO` (v25_05); D-190 cerro el eje "geometria de representacion" (V25, V27, V26, V43). Clasificacion: **DUPLICATE; no se ejecuta nada.** El delta +0.005487 supera nominalmente el gate de +0.003 de la propuesta, pero el propio proyecto corrigio su gate tras medirlo: neto de celdas negativo (-40) y dLB esperado negativo; mas danos que rescates en HS, TR y AG.

Entregables: `reports/toxic_fm_novelty_audit.md` y `.csv`; fila P86-00 en `experiments.csv`; D-291. 0 entrenamientos, 0 submissions, 0 push.

V03 remains the best verified leaderboard result."""


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
    """Integra la seccion 132 y guarda el notebook."""
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
    RUTA_NOTEBOOK.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )

    print("Notebook actualizado con la seccion 132:")
    print(f"  celdas antes: {celdas_antes} | despues: {len(cells)}")
    print("  clasificacion DUPLICATE | precedente V25-01/V25-02")
    print("  celdas de codigo compiladas: OK | ASCII puro en lo generado: OK (0 bytes > 127)")


if __name__ == "__main__":
    main()
