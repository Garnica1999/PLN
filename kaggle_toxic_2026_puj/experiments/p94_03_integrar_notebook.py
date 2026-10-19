"""P94-03: integra la seccion 154 (P94 REVISION SOTA Y FRONTERA) en el notebook.

Lee los entregables `reports/p94_*` y construye una subseccion compacta. Idempotente
y ASCII puro. La celda de verificacion se ejecuta de forma aislada (solo lee el proyecto).

Uso:
    /home/carlos/venv/bin/python experiments/p94_03_integrar_notebook.py
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
RUTA_NOTEBOOK = RAIZ / "notebooks" / "10_final_pipeline.ipynb"
MARCA_SECCION = "## 154. P94 - REVISION SOTA Y FRONTERA DE MODELADO"
MARCA_CODIGO = "Verificacion de la seccion 154 (P94"


def nueva_celda_markdown(texto: str) -> dict:
    """Crea una celda de markdown con identificador unico."""
    return {"cell_type": "markdown", "id": uuid.uuid4().hex[:8], "metadata": {},
            "source": texto.splitlines(keepends=True)}


def nueva_celda_codigo(texto: str) -> dict:
    """Crea una celda de codigo vacia de salidas y sin ejecutar."""
    return {"cell_type": "code", "id": uuid.uuid4().hex[:8], "metadata": {},
            "execution_count": None, "outputs": [], "source": texto.splitlines(keepends=True)}


CODIGO = '''"""Verificacion de la seccion 154 (P94, revision SOTA y frontera de modelado).

Comprueba los entregables de P94 y la integridad de las fuentes canonicas.
No entrena nada.
"""

import csv
import hashlib
from pathlib import Path

ENTREGABLES = ("p94_experiment_inventory.csv", "p94_sota_review.md",
               "p94_hypothesis_registry.csv", "p94_recommended_experiment.md",
               "p94_reproducibility_and_compliance.md")
for nombre in ENTREGABLES:
    assert (PROJECT_ROOT / "reports" / nombre).exists(), f"falta {nombre}"


def leer(ruta):
    """Lee un CSV de reportes con encabezado estandar."""
    with open(ruta, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


inv = leer(PROJECT_ROOT / "reports" / "p94_experiment_inventory.csv")
reg = leer(PROJECT_ROOT / "reports" / "p94_hypothesis_registry.csv")
assert len(inv) == 24 and all(len(f) == 8 for f in inv)
assert len(reg) == 17
assert not any(f["classification"] == "GENUINELY_NEW" for f in reg)
assert not any(f["recommendation"] == "SI" for f in reg)
rec = (PROJECT_ROOT / "reports" / "p94_recommended_experiment.md").read_text(
    encoding="utf-8")
assert "NO_NEW_MODELING_HYPOTHESIS" in rec
assert "TRAINING_NOT_AUTHORIZED" in rec

fuentes = {
    "data/raw/toxic_train.csv": "3d40e7345b588860",
    "features/oof_v03_final.npz": "559265816ce4f631",
    "experiments/baseline_v03_frozen/submission_v03.csv": "7d7478f271e6edfd",
    "experiments/baseline_v03_frozen/model_v03.joblib": "263ed5bd0fc0de7c",
    "data/processed/holdout_v2_ids.csv": "e5b62e5f22c4c63a",
    "reports/p93r4_final_decisions.csv": "98e4264b0fb98b7a",
    "reports/p93r5_gold_comparison.csv": "cc7ab96b90155980",
}
for ruta, pref in fuentes.items():
    h = hashlib.sha256((PROJECT_ROOT / ruta).read_bytes()).hexdigest()
    assert h.startswith(pref), f"fuente alterada: {ruta}"

for nombre in ENTREGABLES:
    datos = (PROJECT_ROOT / "reports" / nombre).read_bytes()
    assert all(b <= 127 for b in datos), f"byte > 127 en {nombre}"

print("P94 (revision SOTA y frontera de modelado) verificado:")
print("  24 familias inventariadas | 17 propuestas auditadas | 0 genuinamente nuevas")
print("  0 recomendadas | NO_NEW_MODELING_HYPOTHESIS | canonicos intactos")'''


def construir_seccion() -> str:
    """Construye el markdown de la seccion 154."""
    return (
        "## 154. P94 - REVISION SOTA Y FRONTERA DE MODELADO (2026-10-13)\n"
        "\n"
        "**Fase de investigacion y diseno, sin entrenamiento ni submissions.** P0: "
        "inventario verificable de 24 familias historicas (21 CLOSED, 1 MARGINAL, 2 "
        "BLOCKED) con 95 identificadores anclados y verificados contra las 459 filas "
        "reales de experiments.csv; el registro completo (v2, v4, v5/v5.1, v6, v10, v11 "
        "y P84-P93) queda mapeado a familias con su conclusion y razon de cierre. P1: "
        "revision de 17 referencias primarias (12 revisadas por pares; 3 preprints "
        "marcados; 2 informativas) sobre macro-F1/no descomponibles, decision plug-in, "
        "costes, NB-SVM, ponderacion supervisada, negacion, estructura multietiqueta, "
        "aumento, kernels, bayesiano y dominio espanol. P2: 17 propuestas auditadas -> "
        "11 ALREADY_TESTED, 2 DUPLICATE, 3 NOT_FEASIBLE, 1 INSUFFICIENT_EVIDENCE; 0 "
        "GENUINELY_NEW. P3: tres candidatas preseleccionadas (perdida sustituta de "
        "F-beta; marcado de negacion; kNN instance-based) y ninguna supera el criterio "
        "de novedad. P4/P5: veredicto **NO_NEW_MODELING_HYPOTHESIS**; no se disena "
        "experimento y no hay nada que autorizar.\n"
        "\n"
        "**Por que**: la F1-suave literal ya se midio y perdio (V7-01), la familia de "
        "negacion/cita/sintaxis se cerro con medicion (E2-08), la estructura de "
        "etiquetas tiene techo medido no capturable (D-116), los ensembles/augmentation/"
        "kernels/bayesiano estan cerrados, y rige el freno de transferencia (mejoras "
        "locales <= +0.02 no se trasladaron al LB en R1-C1/R2-MINDF5/OPT-01). El "
        "candidato cost-sensitive existente no es nuevo (LB externo 0.77522) y sigue "
        "congelado en gobernanza. La correccion de etiquetas (14 candidatas P93-R5) "
        "sigue PRESERVED_NOT_CONFIRMED, no es justificacion de features.\n"
        "\n"
        "**Entregables**: p94_experiment_inventory.csv, p94_sota_review.md, "
        "p94_hypothesis_registry.csv, p94_recommended_experiment.md, "
        "p94_reproducibility_and_compliance.md. Hashes: inventario 11f376e1d4c1545c, "
        "sota 245cf119df408b4b, registro 17917ce2a1a132d9, recomendacion "
        "7a09af6710b863af, cumplimiento 588c7664e8b7ddf4.\n"
        "\n"
        "**Estado**: `NO_NEW_MODELING_HYPOTHESIS` + `TRAINING_NOT_AUTHORIZED` + "
        "`SUBMISSIONS_NOT_AUTHORIZED` + `V03 = FROZEN` (corrected_cv "
        "0.7772114525165721; LB 0.78306).\n"
        "\n"
        "> V03 remains the best verified leaderboard result.")


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
    """Ejecuta la celda de verificacion de forma aislada (solo lee el proyecto)."""
    espacio: dict[str, object] = {"PROJECT_ROOT": RAIZ, "__name__": "__verificacion__"}
    exec(compile(CODIGO, "<seccion 154>", "exec"), espacio)


def main() -> None:
    """Integra la seccion 154, ejecuta su verificacion y guarda el notebook."""
    seccion = construir_seccion()
    if any(ord(c) > 127 for c in seccion):
        raise SystemExit("Byte > 127 en el texto generado (seccion)")
    ejecutar_verificacion()
    notebook = json.loads(RUTA_NOTEBOOK.read_text(encoding="utf-8"))
    cells = notebook["cells"]
    celdas_antes = len(cells)
    while any("".join(c.get("source", [])).startswith(MARCA_SECCION) for c in cells):
        cells.pop(next(i for i, c in enumerate(cells)
                       if "".join(c.get("source", [])).startswith(MARCA_SECCION)))
    while any(MARCA_CODIGO in "".join(c.get("source", [])) for c in cells):
        cells.pop(next(i for i, c in enumerate(cells)
                       if MARCA_CODIGO in "".join(c.get("source", []))))
    cells.append(nueva_celda_markdown(seccion))
    cells.append(nueva_celda_codigo(CODIGO))
    validar(cells)
    notebook["cells"] = cells
    RUTA_NOTEBOOK.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
                             encoding="utf-8")
    print("Notebook actualizado con la seccion 154:")
    print(f"  celdas antes: {celdas_antes} | despues: {len(cells)}")
    print("  revision SOTA P94 verificada en aislamiento: OK")
    print("  celdas de codigo compiladas: OK | ASCII puro en lo generado: OK (0 bytes > 127)")


if __name__ == "__main__":
    main()
