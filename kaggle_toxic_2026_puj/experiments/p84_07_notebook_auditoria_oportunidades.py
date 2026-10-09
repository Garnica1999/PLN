"""Integra la seccion 130 (auditoria de oportunidades legales) en el notebook.

La seccion 130 documenta el resultado de la auditoria adversarial del 2026-10-08:
la decision Caso B (no existe via legal defendible) y el reparto de la tabla de
oportunidades. Incluye una celda de verificacion que comprueba los tres
entregables, el reparto declarado y el escaneo byte a byte (0 bytes > 127), sin
presentar hipotesis no ejecutadas como evidencia experimental.

Politica de generacion (P80 seccion 2.4): ASCII puro en todo lo generado,
verificado con `tools/verificar_caracteres.py` mas un escaneo byte a byte que
rechace cualquier byte mayor que 127. Este script se aplica la regla a si mismo.

El script es idempotente: si la seccion ya existe, la regenera.

Uso:
    /home/carlos/venv/bin/python experiments/p84_07_notebook_auditoria_oportunidades.py
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
RUTA_NOTEBOOK = RAIZ / "notebooks" / "10_final_pipeline.ipynb"
MARCAS_SECCION = ("## 130. Auditoria", "## 130. Auditor\u00eda")
MARCA_CODIGO = "Verificacion de la auditoria de oportunidades legales"

SECCION = '''## 130. Auditoria de oportunidades legales restantes (2026-10-08)

La auditoria adversarial reviso, sin entrenar nada y sin tocar test, holdout ni leaderboard, si quedaba alguna intervencion permitida, no ejecutada y con un mecanismo defendible. **Resultado: no. Caso B.** La investigacion de modelado permanece cerrada y la experimentacion se detiene.

**No se identifica actualmente una via experimental nueva, legal y suficientemente justificada dentro de las restricciones vigentes del proyecto.**

- **Via A (frontera de anotacion de TR): `BLOCKED_BY_RULES`.** El material de revision humana esta listo desde la fase 2 y nunca se completo (80/80 columnas humanas vacias); las reglas son silenciosas sobre corregir etiquetas de entrenamiento y el uso modelable de la revision cae en familias cerradas. V4-A ya midio que la contradiccion de etiquetas (2,8 por ciento) no explica el error.
- **Via B (propagacion de near-duplicados train -> test): `QUARANTINED`.** Exige autorizacion escrita del organizador para un uso transductivo; el techo medido (V4-F04; sonda v18 con 0 celdas; <= 48 celdas de 3462) queda por debajo de la banda de deteccion del leaderboard.
- **Puerta 2 (H1 a H5):** las cinco hipotesis estan ejecutadas y cerradas (fase v9 y posteriores: H5 delta exacto 0,0; H4 +0,0006 y despues negativo; H2 0 celdas; H1 V9-04/V10-11; H3 SDR/PLS).
- **Resto de familias:** 0 `ACTIONABLE`; el reparto de las 20 lineas auditadas es 5 `BLOCKED_BY_RULES`, 1 `DUPLICATE` y 14 `CLOSED`.
- **Candidato downweight:** intacto (`PROMISING_UNCONFIRMED` / `NEGATIVE` / `EXPERIMENTAL_RESEARCH_CLOSED` / `PENDING`; LB externo 0,77522).

Entregables: `reports/toxic_remaining_legal_opportunities.md` y `.csv`, y `reports/toxic_final_research_gap.md`. Decision registrada en D-288. Sin submissions, sin test, sin holdout y sin leaderboard.

V03 remains the best verified leaderboard result.'''

CODIGO = '''"""Verificacion de la auditoria de oportunidades legales (seccion 130).

Comprueba que los tres entregables existen, que la tabla de oportunidades tiene
las columnas del encargo y el reparto declarado (0 ACTIONABLE), y que los
entregables son ASCII puro (escaneo byte a byte: 0 bytes > 127).
"""

import csv

ESPERADAS = [
    "hypothesis", "historical_evidence", "prior_experiments", "novelty",
    "legal_status", "expected_mechanism", "control", "cost", "risk", "decision",
]

ruta_md = PROJECT_ROOT / "reports" / "toxic_remaining_legal_opportunities.md"
ruta_csv = PROJECT_ROOT / "reports" / "toxic_remaining_legal_opportunities.csv"
ruta_gap = PROJECT_ROOT / "reports" / "toxic_final_research_gap.md"

for ruta in (ruta_md, ruta_csv, ruta_gap):
    assert ruta.exists(), f"falta el entregable {ruta.name}"

filas = list(csv.DictReader(ruta_csv.read_text(encoding="utf-8").splitlines()))
assert list(filas[0].keys()) == ESPERADAS, "columnas inesperadas en la tabla de oportunidades"
assert len(filas) == 20, f"la tabla debe tener 20 lineas, tiene {len(filas)}"

reparto = {}
for fila in filas:
    reparto[fila["decision"]] = reparto.get(fila["decision"], 0) + 1

assert reparto.get("ACTIONABLE", 0) == 0, "no debe haber lineas ACTIONABLE"

sucios = [ruta.name for ruta in (ruta_md, ruta_csv, ruta_gap) if any(b > 127 for b in ruta.read_bytes())]
assert not sucios, f"entregables con bytes > 127: {sucios}"

print("Auditoria de oportunidades legales (2026-10-08):")
print(f"  lineas auditadas: {len(filas)} | reparto: {reparto}")
print("  entregables: toxic_remaining_legal_opportunities.md/.csv y toxic_final_research_gap.md")
print("  escaneo byte a byte: 0 bytes > 127 en los entregables")
print("  decision final: Caso B (la investigacion de modelado permanece cerrada)")'''


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


def es_seccion(celda: dict) -> bool:
    """Comprueba si la celda es la seccion 130 (version ASCII o con acentos)."""
    inicio = "".join(celda.get("source", []))[:24]
    return any(inicio.startswith(marca) for marca in MARCAS_SECCION)


def validar(cells: list[dict]) -> None:
    """Compila las celdas de codigo y exige ASCII puro en lo generado."""
    for indice, celda in enumerate(cells):
        fuente = "".join(celda.get("source", []))
        if celda["cell_type"] == "code":
            compile(fuente, f"<celda {indice}>", "exec")
    for nombre, texto in (("seccion", SECCION), ("codigo", CODIGO)):
        if any(ord(caracter) > 127 for caracter in texto):
            raise SystemExit(f"Byte > 127 en el texto generado ({nombre})")
    if any(byte > 127 for byte in Path(__file__).read_bytes()):
        raise SystemExit("El propio script contiene bytes > 127")


def main() -> None:
    """Integra la seccion 130 y guarda el notebook."""
    notebook = json.loads(RUTA_NOTEBOOK.read_text(encoding="utf-8"))
    cells = notebook["cells"]
    celdas_antes = len(cells)

    while any(es_seccion(c) for c in cells):
        cells.pop(next(i for i, c in enumerate(cells) if es_seccion(c)))
    while any(MARCA_CODIGO in "".join(c.get("source", [])) for c in cells):
        cells.pop(next(i for i, c in enumerate(cells) if MARCA_CODIGO in "".join(c.get("source", []))))

    cells.append(nueva_celda_markdown(SECCION))
    cells.append(nueva_celda_codigo(CODIGO))

    validar(cells)
    notebook["cells"] = cells
    RUTA_NOTEBOOK.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )

    print("Notebook actualizado con la seccion 130:")
    print(f"  celdas antes: {celdas_antes} | despues: {len(cells)}")
    print("  celdas de codigo compiladas: OK | ASCII puro en lo generado: OK (0 bytes > 127)")


if __name__ == "__main__":
    main()
