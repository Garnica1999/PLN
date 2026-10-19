"""P93R5-02: integra la seccion 153 (P93-R5 COMPARACION CON GOLD) en el notebook.

Lee los entregables `reports/p93r5_*` y construye una subseccion compacta. Idempotente
y ASCII puro. La celda de verificacion se ejecuta de forma aislada (solo lee el proyecto).

Uso:
    /home/carlos/venv/bin/python experiments/p93r5_02_integrar_notebook.py
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
RUTA_NOTEBOOK = RAIZ / "notebooks" / "10_final_pipeline.ipynb"
MARCA_SECCION = "## 153. P93-R5 - COMPARACION CON EL GOLD"
MARCA_CODIGO = "Verificacion de la seccion 153 (P93-R5"


def nueva_celda_markdown(texto: str) -> dict:
    """Crea una celda de markdown con identificador unico."""
    return {"cell_type": "markdown", "id": uuid.uuid4().hex[:8], "metadata": {},
            "source": texto.splitlines(keepends=True)}


def nueva_celda_codigo(texto: str) -> dict:
    """Crea una celda de codigo vacia de salidas y sin ejecutar."""
    return {"cell_type": "code", "id": uuid.uuid4().hex[:8], "metadata": {},
            "execution_count": None, "outputs": [], "source": texto.splitlines(keepends=True)}


CODIGO = '''"""Verificacion de la seccion 153 (P93-R5, comparacion con el gold).

Comprueba los entregables de P93-R5, las candidatas documentadas y la integridad
de las fuentes canonicas. No entrena nada.
"""

import csv
import hashlib
import json
from pathlib import Path

for nombre in ("p93r5_gold_comparison.csv", "p93r5_gold_comparison.md",
               "p93r5_label_correction_candidates.csv",
               "p93r5_future_experiment_recommendation.md", "p93r5_manifest.json"):
    assert (PROJECT_ROOT / "reports" / nombre).exists(), f"falta {nombre}"


def leer(ruta):
    """Lee un CSV de reportes con encabezado estandar."""
    with open(ruta, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


cmp_ = leer(PROJECT_ROOT / "reports" / "p93r5_gold_comparison.csv")
cand = leer(PROJECT_ROOT / "reports" / "p93r5_label_correction_candidates.csv")
assert len(cmp_) == 176 and len(cand) == 176
estados = {(f["example_id"], f["label"]): f["comparison_status"] for f in cmp_}
assert sum(1 for v in estados.values() if v == "HUMAN_FINAL_MATCHES_GOLD") == 0
assert sum(1 for v in estados.values() if v == "POSSIBLE_LABEL_ISSUE") == 14
assert sum(1 for v in estados.values() if v == "HUMAN_FINAL_DIFFERS_FROM_GOLD") == 157
assert sum(1 for v in estados.values() if v == "UNRESOLVED_NO_THIRD_REVIEW") == 5

eleg = [f for f in cand if f["eligible_for_future_experiment"] == "true"]
assert len(eleg) == 14
assert all(estados[(f["example_id"], f["label"])] == "POSSIBLE_LABEL_ISSUE" for f in eleg)
assert all(f["justification"].strip() for f in eleg)
assert all(f["comparison_status"] == estados[(f["example_id"], f["label"])] for f in cand)
assert all(f["eligible_for_future_experiment"] == "false" for f in cand
           if f["comparison_status"] == "UNRESOLVED_NO_THIRD_REVIEW")

m = json.loads((PROJECT_ROOT / "reports" / "p93r5_manifest.json").read_text(encoding="utf-8"))
assert m["status"] == "GOLD_COMPARISON_COMPLETED"
assert m["comparison"]["POSSIBLE_LABEL_ISSUE"] == 14
assert m["comparison"]["CONFIRMED_ANNOTATION_ERROR"] == 0
assert m["decisions"]["source_label_modification"] == "NOT_AUTHORIZED"
assert all(m["checks"].values()), "checks con fallos"

fuentes = {
    "data/raw/toxic_train.csv": "3d40e7345b588860",
    "features/oof_v03_final.npz": "559265816ce4f631",
    "experiments/baseline_v03_frozen/submission_v03.csv": "7d7478f271e6edfd",
    "experiments/baseline_v03_frozen/model_v03.joblib": "263ed5bd0fc0de7c",
    "data/processed/holdout_v2_ids.csv": "e5b62e5f22c4c63a",
    "artifacts/p93_adjudication_v1/responses/response_C1.csv": "f89f74cdabd9649d",
    "artifacts/p93_adjudication_v1/responses/response_C2.csv": "e0449a811fa8b64b",
    "artifacts/p93_adjudication_v2/responses/response_C1_round2.csv": "91911ab70f6d70cc",
    "artifacts/p93_adjudication_v2/responses/response_C2_round2_correcciones.csv":
        "5da8fc1b33135489",
    "reports/p93r4_final_decisions.csv": "98e4264b0fb98b7a",
}
for ruta, pref in fuentes.items():
    h = hashlib.sha256((PROJECT_ROOT / ruta).read_bytes()).hexdigest()
    assert h.startswith(pref), f"fuente alterada: {ruta}"

sucios = [n for n in ("p93r5_gold_comparison.md", "p93r5_future_experiment_recommendation.md",
                      "p93r5_manifest.json")
          if any(x > 127 for x in (PROJECT_ROOT / "reports" / n).read_bytes())]
assert not sucios, f"bytes > 127 en {sucios}"

print("P93-R5 (comparacion con el gold) verificado:")
print("  176 celdas | 0 coincidencias | 157 discrepancias registradas | 14 candidatas")
print("  5 sin resolver (conservados) | 0 errores confirmados | fuentes canonicas intactas")'''


def construir_seccion() -> str:
    """Construye el markdown de la seccion 153."""
    m = json.loads((RAIZ / "reports" / "p93r5_manifest.json").read_text(encoding="utf-8"))
    return (
        "## 153. P93-R5 - COMPARACION CON EL GOLD Y CORRECCIONES CANDIDATAS (2026-10-13)\n"
        "\n"
        "**Comparacion AUTORIZADA explicitamente por el propietario (sin tercera revision "
        "humana).** Las 176 celdas candidatas se compararon contra el gold interno: 171 con "
        "decision humana final binaria (C1==C2) y 5 sin consenso. Resultado: **0 coincidencias "
        "con el gold y 171 discrepancias** (todas registradas; ninguna confirmada como error). "
        "Clasificacion: **POSSIBLE_LABEL_ISSUE 14** (con justificacion documentada de C1/C2 "
        "referida a la guia v2.1; HS 2, TR 4, AG 8; direccion retirar 10 / anadir 4; las 14 del "
        "componente B enriquecido) y **HUMAN_FINAL_DIFFERS_FROM_GOLD 157** (sin justificacion "
        "documentada; la coincidencia C1/C2 o con A/B no se considera suficiente). "
        "CONFIRMED_ANNOTATION_ERROR = 0.\n"
        "\n"
        "**Los 5 casos pendientes** (P92W-0035/0061 AG; 0094/0240 TR; 0243 AG) se conservan "
        "como UNRESOLVED_NO_THIRD_REVIEW: no bloquean el analisis de las 171 y no se abre "
        "tercera revision.\n"
        "\n"
        "**Limitaciones**: muestra enriquecida por discrepancia A/B vs gold (sin estimacion de "
        "prevalencia; componentes A 43 / B 133); la ronda 2 vio A/B por diseno, asi que las "
        "decisiones no son independientes y la concordancia humana no mide de forma "
        "independiente el error del gold; el cegamiento de la ronda 1 no es verificable por "
        "completo.\n"
        "\n"
        "**Entregables**: p93r5_gold_comparison.{md,csv}, p93r5_label_correction_candidates.csv, "
        "p93r5_future_experiment_recommendation.md, p93r5_manifest.json. Recomendacion: "
        "experimento limitado opcional solo con protocolo agrupado pre-registrado; no se espera "
        "mejora medible con 14 celdas; el train oficial y V03 permanecen intactos.\n"
        "\n"
        "**Estado**: `GOLD_COMPARISON_COMPLETED` + `CORRECTION_CANDIDATES_DOCUMENTED` + "
        "`FIVE_CASES_UNRESOLVED` + `SOURCE_LABELS_UNCHANGED` + `V03 = FROZEN` + "
        "`MODEL_RETRAINING = NOT_AUTHORIZED`.\n"
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
    exec(compile(CODIGO, "<seccion 153>", "exec"), espacio)


def main() -> None:
    """Integra la seccion 153, ejecuta su verificacion y guarda el notebook."""
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
    print("Notebook actualizado con la seccion 153:")
    print(f"  celdas antes: {celdas_antes} | despues: {len(cells)}")
    print("  comparacion con gold P93-R5 verificada en aislamiento: OK")
    print("  celdas de codigo compiladas: OK | ASCII puro en lo generado: OK (0 bytes > 127)")


if __name__ == "__main__":
    main()
