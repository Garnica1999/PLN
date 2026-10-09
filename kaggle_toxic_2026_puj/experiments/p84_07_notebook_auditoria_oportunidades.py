"""Integra la seccion 130 (auditoria de oportunidades legales) en el notebook.

La seccion 130 documenta el resultado de la auditoria adversarial del 2026-10-08:
la decision Caso B (no existe via legal defendible) y el reparto de la tabla de
oportunidades, incluida la hipotesis Twitter/X (Word2Vec de dominio). Incluye
una celda de verificacion que comprueba los tres entregables, el reparto
declarado, la fila de Twitter/X y el escaneo byte a byte (0 bytes > 127), sin
presentar hipotesis no ejecutadas como evidencia experimental.

Ademas anade, una sola vez, una nota de errata a la seccion 116 (documentacion
historica del primer P0 de la linea Twitter): la linea se reabrio y quedo
CLOSED por medicion, no por indisponibilidad.

Politica de generacion (P80 seccion 2.4): ASCII puro en todo lo generado,
verificado con `tools/verificar_caracteres.py` mas un escaneo byte a byte que
rechace cualquier byte mayor que 127. Este script se aplica la regla a si mismo.

El script es idempotente: si la seccion ya existe, la regenera, y la nota de
errata no se duplica.

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
MARCA_NOTA = "Nota de la comprobacion documental del 2026-10-08"

SECCION = '''## 130. Auditoria de oportunidades legales restantes (2026-10-08)

La auditoria adversarial reviso, sin entrenar nada y sin tocar test, holdout ni leaderboard, si quedaba alguna intervencion permitida, no ejecutada y con un mecanismo defendible. **Resultado: no. Caso B.** La investigacion de modelado permanece cerrada y la experimentacion se detiene.

**No se identifica actualmente una via experimental nueva, legal y suficientemente justificada dentro de las restricciones vigentes del proyecto.**

- **Via A (frontera de anotacion de TR): `BLOCKED_BY_RULES`.** El material de revision humana esta listo desde la fase 2 y nunca se completo (80/80 columnas humanas vacias); las reglas son silenciosas sobre corregir etiquetas de entrenamiento y el uso modelable de la revision cae en familias cerradas. V4-A ya midio que la contradiccion de etiquetas (2,8 por ciento) no explica el error.
- **Via B (propagacion de near-duplicados train -> test): `QUARANTINED`.** Exige autorizacion escrita del organizador para un uso transductivo; el techo medido (V4-F04; sonda v18 con 0 celdas; <= 48 celdas de 3462) queda por debajo de la banda de deteccion del leaderboard.
- **Puerta 2 (H1 a H5):** las cinco hipotesis estan ejecutadas y cerradas (fase v9 y posteriores: H5 delta exacto 0,0; H4 +0,0006 y despues negativo; H2 0 celdas; H1 V9-04/V10-11; H3 SDR/PLS).
- **Hipotesis Twitter/X (Word2Vec de dominio): `CLOSED` (ejecutada y medida).** La linea se reabrio con un corpus legal autorizado (mmaguero, CC BY-SA 4.0, periodo cerrado, solapamiento cero con train) y quedo cerrada por medicion: P72 sin ventaja, P73/P74 invalidados, P75 sin transferencia, P78 sin transferencia util (brecha 0.117450 frente a V03) y P84-04 negativo sobre corpus general. El corpus de 597M del encargo original sigue bloqueado; la combinacion SGNS corregido x corpus Twitter no se ejecuto.
- **Resto de familias:** 0 `ACTIONABLE`; el reparto de las 20 lineas auditadas es 5 `BLOCKED_BY_RULES`, 1 `DUPLICATE` y 14 `CLOSED`.
- **Candidato downweight:** intacto (`PROMISING_UNCONFIRMED` / `NEGATIVE` / `EXPERIMENTAL_RESEARCH_CLOSED` / `PENDING`; LB externo 0,77522).

Entregables: `reports/toxic_remaining_legal_opportunities.md` y `.csv`, y `reports/toxic_final_research_gap.md`. Decision registrada en D-288. Sin submissions, sin test, sin holdout y sin leaderboard.

V03 remains the best verified leaderboard result.'''

CODIGO = '''"""Verificacion de la auditoria de oportunidades legales (seccion 130).

Comprueba que los tres entregables existen, que la tabla de oportunidades tiene
las columnas del encargo y el reparto declarado (0 ACTIONABLE), que la fila de
Twitter/X queda CLOSED, y que los entregables son ASCII puro (escaneo byte a
byte: 0 bytes > 127).
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

fila_twitter = next((f for f in filas if "Twitter" in f["hypothesis"]), None)
assert fila_twitter is not None, "falta la fila de Twitter/X en la tabla"
assert fila_twitter["decision"] == "CLOSED", "la fila de Twitter/X debe quedar CLOSED"

sucios = [ruta.name for ruta in (ruta_md, ruta_csv, ruta_gap) if any(b > 127 for b in ruta.read_bytes())]
assert not sucios, f"entregables con bytes > 127: {sucios}"

print("Auditoria de oportunidades legales (2026-10-08):")
print(f"  lineas auditadas: {len(filas)} | reparto: {reparto}")
print("  entregables: toxic_remaining_legal_opportunities.md/.csv y toxic_final_research_gap.md")
print("  hipotesis Twitter/X: CLOSED (ejecutada y medida)")
print("  escaneo byte a byte: 0 bytes > 127 en los entregables")
print("  decision final: Caso B (la investigacion de modelado permanece cerrada)")'''

NOTA_116 = (
    "\n\n*(Nota de la comprobacion documental del 2026-10-08: este bloque documenta el primer P0, "
    "con cuatro candidatos. La linea se reabrio despues (P0 ampliado a 9 candidatos, correccion de "
    "criterio y autorizacion escrita) y quedo CLOSED por medicion, no por indisponibilidad; ver la "
    "seccion 130 y reports/toxic_remaining_legal_opportunities.md, seccion 6.6.)*"
)


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
    for nombre, texto in (("seccion", SECCION), ("codigo", CODIGO), ("nota", NOTA_116)):
        if any(ord(caracter) > 127 for caracter in texto):
            raise SystemExit(f"Byte > 127 en el texto generado ({nombre})")
    if any(byte > 127 for byte in Path(__file__).read_bytes()):
        raise SystemExit("El propio script contiene bytes > 127")


def main() -> None:
    """Integra la seccion 130, la nota de errata y guarda el notebook."""
    notebook = json.loads(RUTA_NOTEBOOK.read_text(encoding="utf-8"))
    cells = notebook["cells"]
    celdas_antes = len(cells)

    def es_gestionada(celda: dict) -> bool:
        fuente = "".join(celda.get("source", []))
        return es_seccion(celda) or MARCA_CODIGO in fuente

    posiciones = [i for i, c in enumerate(cells) if es_gestionada(c)]
    indice_insercion = min(posiciones) if posiciones else len(cells)
    while any(es_gestionada(c) for c in cells):
        cells.pop(next(i for i, c in enumerate(cells) if es_gestionada(c)))

    nuevas = [nueva_celda_markdown(SECCION), nueva_celda_codigo(CODIGO)]
    cells[indice_insercion:indice_insercion] = nuevas

    notas = 0
    for celda in cells:
        fuente = "".join(celda.get("source", []))
        if (
            "domain-specific Word2Vec" in fuente
            and "CLOSED_AT_P0_BY_CORPUS_UNAVAILABILITY" in fuente
            and MARCA_NOTA not in fuente
        ):
            celda["source"] = (fuente.rstrip() + NOTA_116 + "\n").splitlines(keepends=True)
            notas += 1

    validar(cells)
    notebook["cells"] = cells
    RUTA_NOTEBOOK.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )

    print("Notebook actualizado:")
    print(f"  celdas antes: {celdas_antes} | despues: {len(cells)}")
    print(f"  notas de errata anadidas: {notas}")
    print("  celdas de codigo compiladas: OK | ASCII puro en lo generado: OK (0 bytes > 127)")


if __name__ == "__main__":
    main()
