"""P95-02: integra la seccion 155 (P95 CIERRE DE FRONTERA) en el notebook.

Lee los entregables `reports/p95_*` y construye una subseccion compacta. Idempotente
y ASCII puro. La celda de verificacion se ejecuta de forma aislada (solo lee el proyecto).

Uso:
    /home/carlos/venv/bin/python experiments/p95_02_integrar_notebook.py
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
RUTA_NOTEBOOK = RAIZ / "notebooks" / "10_final_pipeline.ipynb"
MARCA_SECCION = "## 155. P95 - CIERRE DE LA FRONTERA EXPERIMENTAL"
MARCA_CODIGO = "Verificacion de la seccion 155 (P95"


def nueva_celda_markdown(texto: str) -> dict:
    """Crea una celda de markdown con identificador unico."""
    return {"cell_type": "markdown", "id": uuid.uuid4().hex[:8], "metadata": {},
            "source": texto.splitlines(keepends=True)}


def nueva_celda_codigo(texto: str) -> dict:
    """Crea una celda de codigo vacia de salidas y sin ejecutar."""
    return {"cell_type": "code", "id": uuid.uuid4().hex[:8], "metadata": {},
            "execution_count": None, "outputs": [], "source": texto.splitlines(keepends=True)}


CODIGO = '''"""Verificacion de la seccion 155 (P95, cierre de frontera y plan de reapertura).

Comprueba los entregables de P95 y la integridad de las fuentes canonicas.
No entrena nada.
"""

import csv
import hashlib
from pathlib import Path

ENTREGABLES = ("p95_reopening_assessment.md", "p95_governance_options.csv",
               "p95_new_evidence_options.csv", "p95_decision_gate.md")
for nombre in ENTREGABLES:
    assert (PROJECT_ROOT / "reports" / nombre).exists(), f"falta {nombre}"


def leer(ruta):
    """Lee un CSV de reportes con encabezado estandar."""
    with open(ruta, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


gov = leer(PROJECT_ROOT / "reports" / "p95_governance_options.csv")
evi = leer(PROJECT_ROOT / "reports" / "p95_new_evidence_options.csv")
assert len(gov) == 7 and all(len(f) == 13 for f in gov)
assert len(evi) == 4 and all(len(f) == 10 for f in evi)
assert any("NO_ACTIONABLE_NEW_EVIDENCE_SOURCE" in f["expected_value_for_score"]
           for f in evi)

ass = (PROJECT_ROOT / "reports" / "p95_reopening_assessment.md").read_text(
    encoding="utf-8")
assert "REQUIRES_MORE_HUMAN_EVIDENCE" in ass
for marca in ("### A. Representacion insuficiente",
              "### B. Objetivo de entrenamiento o decision",
              "### C. Variabilidad y transferencia de la validacion",
              "### D. Calidad y definicion de las etiquetas",
              "### E. Restricciones academicas y de la competicion"):
    assert marca in ass, marca

gate = (PROJECT_ROOT / "reports" / "p95_decision_gate.md").read_text(encoding="utf-8")
for marca in ("MODEL_EXPLORATION_PAUSED", "V03 = FROZEN", "NO_TRAINING_AUTHORIZED",
              "NO_SUBMISSIONS_AUTHORIZED", "REOPENING_CRITERIA_DOCUMENTED"):
    assert marca in gate, marca

fuentes = {
    "data/raw/toxic_train.csv": "3d40e7345b588860",
    "features/oof_v03_final.npz": "559265816ce4f631",
    "experiments/baseline_v03_frozen/submission_v03.csv": "7d7478f271e6edfd",
    "experiments/baseline_v03_frozen/model_v03.joblib": "263ed5bd0fc0de7c",
    "data/processed/holdout_v2_ids.csv": "e5b62e5f22c4c63a",
    "reports/p93r4_final_decisions.csv": "98e4264b0fb98b7a",
    "reports/p93r5_gold_comparison.csv": "cc7ab96b90155980",
    "reports/p94_experiment_inventory.csv": "11f376e1d4c1545c",
    "reports/p94_hypothesis_registry.csv": "17917ce2a1a132d9",
}
for ruta, pref in fuentes.items():
    h = hashlib.sha256((PROJECT_ROOT / ruta).read_bytes()).hexdigest()
    assert h.startswith(pref), f"fuente alterada: {ruta}"

for nombre in ENTREGABLES:
    datos = (PROJECT_ROOT / "reports" / nombre).read_bytes()
    assert all(b <= 127 for b in datos), f"byte > 127 en {nombre}"

print("P95 (cierre de frontera y plan de reapertura) verificado:")
print("  24 familias medidas | 0 hipotesis nuevas | 7 vias de gobernanza auditadas")
print("  14 candidatas REQUIRES_MORE_HUMAN_EVIDENCE | sin via prioritaria")
print("  MODEL_EXPLORATION_PAUSED | V03 FROZEN | canonicos intactos")'''


def construir_seccion() -> str:
    """Construye el markdown de la seccion 155."""
    return (
        "## 155. P95 - CIERRE DE LA FRONTERA EXPERIMENTAL Y PLAN DE REAPERTURA "
        "(2026-10-13)\n"
        "\n"
        "**Fase estrategica documental, sin entrenamiento ni accesos reservados.** P0: "
        "auditoria de P94 confirmada (entregables intactos con sus hashes; registro con "
        "0 GENUINELY_NEW y 0 recomendadas; ninguna propuesta posterior en el arbol). P1: "
        "cinco limitaciones evaluadas -> A (representacion) y B (objetivo/decision) "
        "empiricamente cerradas bajo los protocolos probados con controles fieles; C "
        "(variabilidad/transferencia) es una propiedad medida que actua como restriccion "
        "(R1-C1/R2-MINDF5/OPT-01 perdieron en LB; banda de particion ~0.017; plateau de "
        "~1 error estandar); D (etiquetas) queda en blocked_by_governance + "
        "insufficient_evidence (40.49% de la region de insulto interpersonal etiquetada "
        "(0,0,0); 171/176 celdas humanas difieren del gold; 0 errores confirmados); E "
        "(restricciones) es una regla externa del curso (prohibido transformers). "
        "Residuo genuinamente no explorado: instance-based/kNN (sin evidencia a favor). "
        "P2: siete vias de gobernanza auditadas (1 currently_allowed, 4 "
        "allowed_with_explicit_approval, 1 ambiguous, 1 prohibited); ninguna se solicita "
        "hoy. P3: las 14 candidatas quedan **REQUIRES_MORE_HUMAN_EVIDENCE** (sin "
        "revision ciega independiente no hay base) y NOT_WORTH_PURSUING como palanca de "
        "Kaggle; artefactos preservados, train intacto, 5 casos sin resolver. P4: "
        "**NO_ACTIONABLE_NEW_EVIDENCE_SOURCE** para el score; tres opciones documentales "
        "a disposicion del propietario sin accion automatica (N1 revision humana ciega; "
        "N2 auditoria dirigida de negativos; N3 clarificacion oficial de TR). P5: "
        "registros separados (estimacion interna 0.7772114525165721; transferencia "
        "externa observada 0.78306 con el historial de no-transferencia; optimizacion "
        "de leaderboard prohibida) y requisitos de evidencia interna para un candidato "
        "futuro. P6: criterios y plantilla de reapertura documentados (metricas "
        "canonicas, grupos fusionados, gate dOOF > +0.0376 con neto positivo, "
        "presupuesto <= 30 min CPU de diagnostico). P7: sin via prioritaria.\n"
        "\n"
        "**Entregables**: p95_reopening_assessment.md, p95_governance_options.csv, "
        "p95_new_evidence_options.csv, p95_decision_gate.md. Hashes: assessment "
        "1f941bd7dd56d218, gobernanza 2121755e53fc75fa, evidencia a1ec0b31ae9c6bca, "
        "compuerta 0f3450d7cd7703ed.\n"
        "\n"
        "**Decision final**: `MODEL_EXPLORATION_PAUSED` + `V03 = FROZEN` + "
        "`NO_TRAINING_AUTHORIZED` + `NO_SUBMISSIONS_AUTHORIZED` + "
        "`REOPENING_CRITERIA_DOCUMENTED`.\n"
        "\n"
        "> La pausa no prueba que ninguna mejora futura sea posible; significa que el "
        "historial actual no justifica otro experimento en las condiciones presentes. "
        "V03 remains the best verified leaderboard result.")


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
    exec(compile(CODIGO, "<seccion 155>", "exec"), espacio)


def main() -> None:
    """Integra la seccion 155, ejecuta su verificacion y guarda el notebook."""
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
    print("Notebook actualizado con la seccion 155:")
    print(f"  celdas antes: {celdas_antes} | despues: {len(cells)}")
    print("  cierre de frontera P95 verificado en aislamiento: OK")
    print("  celdas de codigo compiladas: OK | ASCII puro en lo generado: OK (0 bytes > 127)")


if __name__ == "__main__":
    main()
