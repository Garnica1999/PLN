"""Integra la seccion 134 (P88-00: auditoria de novedad LightGBM) en el notebook.

Lee `reports/toxic_lgbm_novelty_audit.md` y `.csv` y construye la seccion 134 con
el resultado de la auditoria (clasificacion DUPLICATE, precedente P61-01, numeros
clave) mas una celda de verificacion que comprueba los entregables, la fila P88-00
del registro, la clasificacion y el escaneo byte a byte.

Idempotente y ASCII puro (politica de generacion del proyecto).

Uso:
    /home/carlos/venv/bin/python experiments/p88_01_integrar_notebook.py
"""

from __future__ import annotations

import csv
import json
import uuid
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
RUTA_NOTEBOOK = RAIZ / "notebooks" / "10_final_pipeline.ipynb"
RUTA_MD = RAIZ / "reports" / "toxic_lgbm_novelty_audit.md"
RUTA_CSV = RAIZ / "reports" / "toxic_lgbm_novelty_audit.csv"
MARCA_SECCION = "## 134. P88-00"
MARCA_CODIGO = "Verificacion de la seccion 134 (P88-00)"


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


CODIGO = '''"""Verificacion de la seccion 134 (P88-00).

Comprueba los entregables de la auditoria, la fila P88-00 del registro, la
clasificacion DUPLICATE y que los ficheros generados son ASCII puro.
"""

import csv

ruta_md = PROJECT_ROOT / "reports" / "toxic_lgbm_novelty_audit.md"
ruta_csv = PROJECT_ROOT / "reports" / "toxic_lgbm_novelty_audit.csv"
for ruta in (ruta_md, ruta_csv):
    assert ruta.exists(), f"falta {ruta.name}"

texto = ruta_md.read_text(encoding="utf-8")
assert "DUPLICATE" in texto
assert "P61-01" in texto
assert "0.7513651502" in texto

filas = list(csv.DictReader(ruta_csv.read_text(encoding="utf-8").splitlines()))
assert filas[0]["clasificacion"] == "DUPLICATE"

registro = list(csv.DictReader((PROJECT_ROOT / "experiments.csv").read_text(encoding="utf-8-sig").splitlines()))
fila = [f for f in registro if f["experiment_id"] == "P88-00"]
assert fila, "falta la fila P88-00"
assert fila[0]["decision"].startswith("DUPLICATE")

sucios = [ruta.name for ruta in (ruta_md, ruta_csv) if any(b > 127 for b in ruta.read_bytes())]
assert not sucios, f"bytes > 127 en {sucios}"

print("P88-00 (auditoria de novedad LightGBM) verificado:")
print("  clasificacion DUPLICATE | precedente P61-01 | fila P88-00 OK | 0 bytes > 127")'''


def construir_seccion() -> str:
    """Devuelve el markdown de la seccion 134 a partir de los entregables."""
    texto = RUTA_MD.read_text(encoding="utf-8")
    if "DUPLICATE" not in texto:
        raise SystemExit("STOP: el informe no declara DUPLICATE")
    return """## 134. P88-00: auditoria de novedad - LightGBM sobre la matriz TF-IDF de V03 (DUPLICATE)

La directiva del 2026-10-09 propuso arboles potenciados (LightGBM) directamente sobre la matriz dispersa word+char de V03, con tres clasificadores binarios y umbrales congelados. La auditoria P88-00 (informe `reports/toxic_lgbm_novelty_audit.md`) encontro el precedente metodologicamente equivalente: **P61-01** (`experiments/p61_03_lightgbm.py`, 2026-09-28).

| elemento | P61-01 (ejecutado) |
|---|---|
| mecanismo | LightGBM, un LGBMClassifier por etiqueta |
| representacion | el TF-IDF de V03 tal cual (27.431 rasgos, `hstack(...).tocsr()`) |
| protocolo | pliegues canonicos del OOF congelado; umbrales [0.44, 0.51, 0.46] |
| exploracion | dos puntos: por defecto y Optuna anidado (450 trials, early stopping interno) |
| resultado | brazo 1 -0.0302397191; brazo 2 -0.0258463023; 2/5 pliegues; neto -28; gate no superado |

Cierre historico: `CLOSED_BY_DUPLICATION` (p61_lightgbm_capacidad.md) y D-143 ("la familia no llega"); el patron de busqueda por OOF ya habia cerrado con OPT-01 (+0.008565 OOF, -0.028 LB). Clasificacion: **DUPLICATE; no se ejecuta nada.** Familias vecinas de arboles (V8-13 sobre subespacio, V4-G sobre SVD-300, MI_arbol de P64-05, HistGB de P66-01 y P68-01): todas cerradas.

Entregables: `reports/toxic_lgbm_novelty_audit.md` y `.csv`; fila P88-00 en `experiments.csv`; D-293. 0 entrenamientos, 0 submissions, 0 push.

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
    """Integra la seccion 134 y guarda el notebook."""
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

    print("Notebook actualizado con la seccion 134:")
    print(f"  celdas antes: {celdas_antes} | despues: {len(cells)}")
    print("  clasificacion DUPLICATE | precedente P61-01")
    print("  celdas de codigo compiladas: OK | ASCII puro en lo generado: OK (0 bytes > 127)")


if __name__ == "__main__":
    main()
