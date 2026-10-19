"""Integra la seccion 133 (P87-00: gobernanza de la anotacion TR) en el notebook.

Lee `reports/toxic_tr_annotation_governance.md` y `.csv` y construye la seccion 133
con la evidencia auditada, el protocolo disenado (no ejecutado) y la decision de
gobernanza (LOW_EXPECTED_VALUE), mas una celda de verificacion que comprueba los
entregables, la fila P87-00 del registro y el escaneo byte a byte.

Idempotente y ASCII puro (politica de generacion del proyecto).

Uso:
    /home/carlos/venv/bin/python experiments/p87_01_integrar_notebook.py
"""

from __future__ import annotations

import csv
import json
import uuid
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
RUTA_NOTEBOOK = RAIZ / "notebooks" / "10_final_pipeline.ipynb"
RUTA_MD = RAIZ / "reports" / "toxic_tr_annotation_governance.md"
RUTA_CSV = RAIZ / "reports" / "toxic_tr_annotation_decision.csv"
MARCA_SECCION = "## 133. P87-00"
MARCA_CODIGO = "Verificacion de la seccion 133 (P87-00)"


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


CODIGO = '''"""Verificacion de la seccion 133 (P87-00).

Comprueba los entregables de gobernanza, la fila P87-00 del registro, la
clasificacion LOW_EXPECTED_VALUE y que los ficheros generados son ASCII puro.
"""

import csv

ruta_md = PROJECT_ROOT / "reports" / "toxic_tr_annotation_governance.md"
ruta_csv = PROJECT_ROOT / "reports" / "toxic_tr_annotation_decision.csv"
for ruta in (ruta_md, ruta_csv):
    assert ruta.exists(), f"falta {ruta.name}"

texto = ruta_md.read_text(encoding="utf-8")
assert "LOW_EXPECTED_VALUE" in texto
assert "no ejecutado" in texto
assert "140 filas" in texto

filas = list(csv.DictReader(ruta_csv.read_text(encoding="utf-8").splitlines()))
assert filas[0]["valor"] == "LOW_EXPECTED_VALUE"

registro = list(csv.DictReader((PROJECT_ROOT / "experiments.csv").read_text(encoding="utf-8-sig").splitlines()))
fila = [f for f in registro if f["experiment_id"] == "P87-00"]
assert fila, "falta la fila P87-00"
assert fila[0]["decision"].startswith("LOW_EXPECTED_VALUE")

sucios = [ruta.name for ruta in (ruta_md, ruta_csv) if any(b > 127 for b in ruta.read_bytes())]
assert not sucios, f"bytes > 127 en {sucios}"

print("P87-00 (gobernanza de la anotacion TR) verificado:")
print("  clasificacion LOW_EXPECTED_VALUE | sin revision ni correccion | fila P87-00 OK | 0 bytes > 127")'''


def construir_seccion() -> str:
    """Devuelve el markdown de la seccion 133 a partir de los entregables."""
    texto = RUTA_MD.read_text(encoding="utf-8")
    if "LOW_EXPECTED_VALUE" not in texto:
        raise SystemExit("STOP: el informe no declara LOW_EXPECTED_VALUE")
    return """## 133. P87-00: gobernanza de la revision humana de anotaciones en TR (LOW_EXPECTED_VALUE)

La directiva del 2026-10-09 pidio auditar la unica via sin clasificador nuevo: la revision humana de posibles ambiguedades de anotacion en TR, con protocolo valido y nota de autorizacion. La auditoria P87-00 (informe `reports/toxic_tr_annotation_governance.md`) no ejecuto ninguna revision ni correccion: es un ciclo de gobernanza.

**Evidencia auditada.** El material de revision esta listo y nunca se completo: `reports/eda/tr_ejemplos_para_revision.csv` tiene 80 filas (40 TR=1 + 40 TR=0, todas HS=1) con las tres columnas humanas vacias; 10 de sus filas estan en el holdout (debe filtrarse a dev: 70 + 32 limite + 43 contradictorios = 140 prioritarias). El "2,8 por ciento" de V4-A01 es la proporcion del train en clusters de texto casi duplicado con etiquetas inconsistentes (HS/TR 2.773 %, AG 2.889 %), no una tasa de error. El error de TR es menor dentro de esos clusters (0.023 vs 0.113); las intervenciones medidas son +-0.0005; V28-03 no tiene potencia (43 filas informativas); P59-C es MARGINAL (0.9730 vs 0.9741); el techo de P57 (~+0,009) queda bajo la banda de particion (0,017).

**Protocolo disenado (no ejecutado).** Alcance dev-only; criterios ERROR/AMBIGUO/CORRECTO con la duda resuelta como AMBIGUO; dos revisores ciegos; adjudicacion; confianza y motivo; kappa; trazabilidad; originales inmutables; reglas de parada (kappa < 0.60; ERROR < 5 %).

**Decision: LOW_EXPECTED_VALUE (Caso B).** La revision podria obtener la primera tasa directa de ambiguedad, el kappa de TR y el cierre documental de D-005, pero ninguna cambia una decision: el camino modelable esta bloqueado por reglas y en familias cerradas, y el techo medido queda bajo la banda. No se solicita autorizacion; no se revisa ni corrige nada; TOXIC permanece cerrado. Near-duplicados: QUARANTINED, con beneficio esperado inferior (transductiva sobre test; techo nulo / <= 0,002 LB).

Entregables: `reports/toxic_tr_annotation_governance.md` y `.csv`; fila P87-00 en `experiments.csv`; D-292. 0 entrenamientos, 0 submissions, 0 push.

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
    """Integra la seccion 133 y guarda el notebook."""
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

    print("Notebook actualizado con la seccion 133:")
    print(f"  celdas antes: {celdas_antes} | despues: {len(cells)}")
    print("  clasificacion LOW_EXPECTED_VALUE | protocolo disenado, no ejecutado")
    print("  celdas de codigo compiladas: OK | ASCII puro en lo generado: OK (0 bytes > 127)")


if __name__ == "__main__":
    main()
