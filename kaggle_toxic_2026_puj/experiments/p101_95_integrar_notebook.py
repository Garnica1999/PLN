#!/usr/bin/env python
"""P101-95: integra la seccion 164 (P101: auditoria de supervision y bloqueo).

Idempotente y ASCII puro; celda de verificacion ejecutada en aislamiento.
Los resultados se LEEN de los artefactos registrados (no se reejecuta nada).

Uso:
    /home/carlos/venv/bin/python experiments/p101_95_integrar_notebook.py
"""

from __future__ import annotations

import hashlib
import json
import uuid
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
RUTA_NOTEBOOK = RAIZ / "notebooks" / "10_final_pipeline.ipynb"
MARCA_SECCION = "## 164. P101"
MARCA_CODIGO = "Verificacion de la seccion 164 (P101"


def sha16(ruta: Path) -> str:
    """sha256 truncado a 16 hexdigitos."""
    x = hashlib.sha256()
    with open(ruta, "rb") as fh:
        for bloque in iter(lambda: fh.read(1 << 20), b""):
            x.update(bloque)
    return x.hexdigest()[:16]


def nueva_celda_markdown(texto: str) -> dict:
    """Celda de markdown con identificador unico."""
    return {"cell_type": "markdown", "id": uuid.uuid4().hex[:8], "metadata": {},
            "source": texto.splitlines(keepends=True)}


def nueva_celda_codigo(texto: str) -> dict:
    """Celda de codigo vacia de salidas y sin ejecutar."""
    return {"cell_type": "code", "id": uuid.uuid4().hex[:8], "metadata": {},
            "execution_count": None, "outputs": [],
            "source": texto.splitlines(keepends=True)}


CODIGO = '''"""Verificacion de la seccion 164 (P101: supervision y bloqueo).

No entrena nada; solo lee artefactos.
"""

import csv
import hashlib
import json
from pathlib import Path

PROY = PROJECT_ROOT
R = PROY / "reports"
with open(R / "p101_provenance_table.csv", encoding="utf-8") as fh:
    tabla = list(csv.DictReader(fh))
assert len(tabla) == 1327
assert sum(1 for r in tabla if r["eligibility_new_audit"] == "True") == 971
assert sum(1 for r in tabla if r["v2_1_main_sample"] == "True") == 250
assert sum(1 for r in tabla if r["v2_1_calibration_no_response"] == "True") == 15
assert sum(1 for r in tabla if int(r["r2_c12_cells"]) > 0) == 127

final = (R / "p101_final_report.md").read_text(encoding="utf-8")
assert "CLOSED_BLOCKED" in final
assert "No evaluable" in final
gob = (R / "p101_governance_note.md").read_text(encoding="utf-8")
assert "AMBIGUOUS" in gob

runlog = (PROY / "RUNLOG.md").read_text(encoding="utf-8")
dec = (PROY / "DECISIONS.md").read_text(encoding="utf-8")
assert "Addendum retrospectivo (2026-10-14): trazabilidad de P91-P93" in runlog
assert "Addendum retrospectivo (2026-10-14): trazabilidad de P91-P93" in dec
assert "D-305" in dec

with open(PROY / "experiments.csv", encoding="utf-8-sig") as fh:
    filas = list(csv.DictReader(fh))
ids = [f["experiment_id"] for f in filas]
assert len(filas) == 492 and len(set(ids)) == 492
assert "P101-00" in ids
fila = next(f for f in filas if f["experiment_id"] == "P101-00")
assert json.loads(fila["result"])["estado"] == "CLOSED_BLOCKED_NO_REVIEWERS"

fuentes = {
    "data/raw/toxic_train.csv": "3d40e7345b588860",
    "features/oof_v03_final.npz": "559265816ce4f631",
    "experiments/baseline_v03_frozen/submission_v03.csv": "7d7478f271e6edfd",
    "experiments/baseline_v03_frozen/model_v03.joblib": "263ed5bd0fc0de7c",
    "data/processed/holdout_v2_ids.csv": "e5b62e5f22c4c63a",
    "data/processed/holdout2_ids.csv": "f6c6ed958abf728b",
}
for ruta, pref in fuentes.items():
    h = hashlib.sha256((PROY / ruta).read_bytes()).hexdigest()
    assert h.startswith(pref), f"fuente alterada: {ruta}"

for nombre in ("p101_provenance_audit.md", "p101_governance_note.md",
               "p101_compliance.md", "p101_final_report.md",
               "p101_reproducibility.md"):
    datos = (R / nombre).read_bytes()
    assert all(b <= 127 for b in datos), f"byte > 127 en {nombre}"

print("P101 verificada:")
print("  P0: tabla 1.327 filas | elegibles 971/1236 | reservados v2.1: 0")
print("  P1: AMBIGUOUS | consulta al profesor redactada (no enviada)")
print("  Bloqueo humano: sin revisores -> P6 no ejecutado; labels intactas")
print("  P101 = CLOSED_BLOCKED; addendum P91-P93 presente; fila P101-00.")'''


def construir_seccion() -> str:
    """Construye el markdown de la seccion 164 con datos reales."""
    with open(RAIZ / "reports" / "p101_provenance_table.csv",
              encoding="utf-8") as fh:
        import csv as _csv
        tabla = list(_csv.DictReader(fh))
    n_eleg = sum(1 for r in tabla if r["eligibility_new_audit"] == "True")
    h = {n: sha16(RAIZ / "reports" / n) for n in (
        "p101_provenance_audit.md", "p101_governance_note.md",
        "p101_final_report.md")}
    return (
        "## 164. P101 - AUDITORIA DE SUPERVISION (CALIDAD DE ETIQUETAS) Y "
        "BLOQUEO HUMANO (2026-10-14)\n"
        "\n"
        "**Objetivo**: buscar senal supervisada nueva admisible (errores de "
        "anotacion verificables) sin violar las restricciones de TOXIC "
        "(sin test/holdouts/leaderboard; sin labels externas; sin "
        "pseudo-labeling; sin transformers).\n"
        "\n"
        f"**P0 (procedencia)**: inventario con hashes de los materiales "
        f"P90-P93 (229 archivos); tabla de procedencia por ejemplo "
        f"({len(tabla)} filas = dev2 1.236 + historico); poblacion elegible "
        f"para auditoria nueva: {n_eleg}/1236 (250 revisados + 15 de "
        "calibracion excluidos por prudencia); reservados en v2.1: 0; "
        "canonicos V03 verificados. Hallazgos: cuarentena v2 (38 ids "
        "holdout2); exposicion holdout2 en muestras internas P90 (35) y P91 "
        "(69); hueco de trazabilidad P91-P93 (addendum retrospectivo en "
        "RUNLOG/DECISIONS).\n"
        "\n"
        "**P1 (gobernanza)**: AMBIGUOUS - las reglas no cubren objetivos "
        "alternativos de entrenamiento; plantilla P95 completada; consulta "
        "al profesor redactada (NO enviada).\n"
        "\n"
        "**Bloqueo humano**: sin dos revisores disponibles -> anotacion "
        "detenida SIN simulacion (regla del encargo); P6 no ejecutado; "
        "labels originales intactas.\n"
        "\n"
        "**Decision**: P101 = CLOSED_BLOCKED (sin evidencia humana "
        "admisible); reabrible solo con revisores reales + gobernanza "
        "resuelta. Resultados historicos: 0 errores confirmados; 14 posibles "
        "(solo componente B enriquecido, P93-R5). No se presentan "
        "anotaciones pendientes como resultados.\n"
        "\n"
        f"**Artefactos**: p101_provenance_audit.md ({h['p101_provenance_audit.md']}), "
        f"p101_governance_note.md ({h['p101_governance_note.md']}), "
        f"p101_final_report.md ({h['p101_final_report.md']}), "
        "p101_compliance.md, p101_reproducibility.md, "
        "p101_provenance_table.csv, p101_materials_inventory.csv; "
        "experiments/p101_00/90/95/99; fila P101-00 (total 492).")


def validar(cells: list) -> None:
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
    """Ejecuta la celda de verificacion en aislamiento."""
    espacio: dict = {"PROJECT_ROOT": RAIZ, "__name__": "__verificacion__"}
    exec(compile(CODIGO, "<seccion 164>", "exec"), espacio)  # noqa: S102


def main() -> None:
    """Integra la seccion 164, ejecuta su verificacion y guarda el notebook."""
    seccion = construir_seccion()
    if any(ord(c) > 127 for c in seccion):
        raise SystemExit("Byte > 127 en el texto generado (seccion)")
    ejecutar_verificacion()
    notebook = json.loads(RUTA_NOTEBOOK.read_text(encoding="utf-8"))
    cells = notebook["cells"]
    celdas_antes = len(cells)
    while any("".join(c.get("source", [])).startswith(MARCA_SECCION)
              for c in cells):
        cells.pop(next(i for i, c in enumerate(cells)
                       if "".join(c.get("source", [])).startswith(
                           MARCA_SECCION)))
    while any(MARCA_CODIGO in "".join(c.get("source", [])) for c in cells):
        cells.pop(next(i for i, c in enumerate(cells)
                       if MARCA_CODIGO in "".join(c.get("source", []))))
    cells.append(nueva_celda_markdown(seccion))
    cells.append(nueva_celda_codigo(CODIGO))
    validar(cells)
    notebook["cells"] = cells
    RUTA_NOTEBOOK.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8")
    print("Notebook actualizado con la seccion 164:")
    print(f"  celdas antes: {celdas_antes} | despues: {len(cells)}")
    print("  verificacion de la fase en aislamiento: OK")
    print("  celdas de codigo compiladas: OK | ASCII puro en lo generado: OK")


if __name__ == "__main__":
    main()
