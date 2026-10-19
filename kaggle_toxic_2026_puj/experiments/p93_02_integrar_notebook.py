"""P93-02: integra la seccion 148 (P93 ACUERDO HUMANO) en el notebook.

Lee los entregables `reports/p93_*` y construye una subseccion compacta. Idempotente y
ASCII puro. La celda de verificacion se ejecuta de forma aislada (solo lee el proyecto).

Uso:
    /home/carlos/venv/bin/python experiments/p93_02_integrar_notebook.py
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
RUTA_NOTEBOOK = RAIZ / "notebooks" / "10_final_pipeline.ipynb"
MARCA_SECCION = "## 148. P93 - ACUERDO HUMANO ANALIZADO"
MARCA_CODIGO = "Verificacion de la seccion 148 (P93"


def nueva_celda_markdown(texto: str) -> dict:
    """Crea una celda de markdown con identificador unico."""
    return {"cell_type": "markdown", "id": uuid.uuid4().hex[:8], "metadata": {},
            "source": texto.splitlines(keepends=True)}


def nueva_celda_codigo(texto: str) -> dict:
    """Crea una celda de codigo vacia de salidas y sin ejecutar."""
    return {"cell_type": "code", "id": uuid.uuid4().hex[:8], "metadata": {},
            "execution_count": None, "outputs": [], "source": texto.splitlines(keepends=True)}


CODIGO = '''"""Verificacion de la seccion 148 (P93, acuerdo humano y candidatos).

Comprueba los entregables de P93, la procedencia de la guia y la integridad de las
respuestas. No entrena nada.
"""

import csv
import hashlib
import json
from pathlib import Path

entregables = ["p93_interannotator_agreement.md", "p93_interannotator_agreement.csv",
               "p93_human_disagreement_cells.csv", "p93_gold_comparison_cells.csv",
               "p93_gold_discrepancy_candidates.csv", "p93_adjudication_protocol.md",
               "p93_provenance_audit.md", "p93_manifest.json",
               "p93_adjudication_blind_candidates.csv"]
for nombre in entregables:
    assert (PROJECT_ROOT / "reports" / nombre).exists(), f"falta {nombre}"

m = json.loads((PROJECT_ROOT / "reports" / "p93_manifest.json").read_text(encoding="utf-8"))
assert m["guide_classification"] == "GUIDE_V2_1_CONFIRMED"
assert all(m["checks"].values()), "checks con fallos"
assert m["joint"]["triple_exacto"] == 131
assert m["candidates_total"] == 176

with (PROJECT_ROOT / "reports" / "p93_gold_discrepancy_candidates.csv").open(
        encoding="utf-8", newline="") as fh:
    cand = list(csv.DictReader(fh))
assert len(cand) == 176
with (PROJECT_ROOT / "reports" / "p93_adjudication_blind_candidates.csv").open(
        encoding="utf-8", newline="") as fh:
    blind = list(csv.DictReader(fh))
assert len(blind) == 176
assert "gold" not in " ".join(blind[0].keys()).lower()
assert "reason" not in " ".join(blind[0].keys()).lower()

mz = json.loads((PROJECT_ROOT / "reports" / "p92z_responses_manifest.json").read_text(
    encoding="utf-8"))
for nombre in ("response_A.csv", "response_B.csv", "response_B_sin_texto.csv"):
    h = hashlib.sha256(
        (PROJECT_ROOT / "artifacts" / "human_annotation_v2_1_holdout_safe" / "responses"
         / nombre).read_bytes()).hexdigest()
    assert h == mz["files"][nombre], f"{nombre} cambio"

sucios = [n for n in ("p93_interannotator_agreement.md", "p93_interannotator_agreement.csv",
                      "p93_gold_discrepancy_candidates.csv", "p93_adjudication_protocol.md",
                      "p93_provenance_audit.md", "p93_manifest.json")
          if any(x > 127 for x in (PROJECT_ROOT / "reports" / n).read_bytes())]
assert not sucios, f"bytes > 127 en {sucios}"

print("P93 (acuerdo humano analizado) verificado:")
print("  guia v2.1 confirmada (nivel de paquete) | 131/250 triple exacto | 176 candidatos")
print("  respuestas intactas vs P92-Z | blind sin gold | artefactos ASCII OK")'''


def construir_seccion() -> str:
    """Construye el markdown de la seccion 148."""
    m = json.loads((RAIZ / "reports" / "p93_manifest.json").read_text(encoding="utf-8"))
    a = m["agreement_summary"]
    j = m["joint"]
    d = m["disagreement_counts"]
    return f"""## 148. P93 - ACUERDO HUMANO ANALIZADO Y CANDIDATOS IDENTIFICADOS (2026-10-13)

**Fase diagnostica; sin entrenamiento, sin predicciones, sin cambios de etiquetas.** Analisis formal de las 250 anotaciones (A x B, unidad (review_id, etiqueta)) con la guia clasificada como **GUIDE_V2_1_CONFIRMED** (nivel de paquete: v2.1 presente en ambos paquetes con hashes verificados; la columna `guideline_version=v2` es la POLITICA, definida asi por la propia v2.1; limitacion de acuse de recibo registrada).

**Acuerdo**: HS raw {a['HS']['raw_3cat']} (binario condicional {a['HS']['binario_condicional']}, kappa_bin {a['HS']['kappa_bin']}); TR raw {a['TR']['raw_3cat']} ({a['TR']['binario_condicional']}, kappa_bin {a['TR']['kappa_bin']}); AG raw {a['AG']['raw_3cat']} ({a['AG']['binario_condicional']}, kappa_bin {a['AG']['kappa_bin']}). Kappa 3-cat: HS {a['HS']['kappa_3cat']}, TR {a['TR']['kappa_3cat']}, AG {a['AG']['kappa_3cat']}. Triple exacto {j['triple_exacto']}/250; celdas con desacuerdo {j['celdas_con_desacuerdo']}/750; celdas con UTD {j['celdas_con_utd']}/750. El grueso del desacuerdo es UTD-vs-binario ({d.get('UTD_VS_BINARY', 0)} celdas) frente a binario-binario ({d.get('BINARY_DISAGREEMENT', 0)}); UTD_BOTH {d.get('UTD_BOTH', 0)} (acuerdo en incertidumbre).

**Discrepancias con el gold (internas)**: {m['candidates_total']} celdas donde A y B coinciden en binario distinto del gold (HS {m['candidates_by_label'].get('HS', 0)}, TR {m['candidates_by_label'].get('TR', 0)}, AG {m['candidates_by_label'].get('AG', 0)}); NO son errores confirmados. Cola interna + paquete ciego preparados (`p93_gold_discrepancy_candidates.csv`, `p93_adjudication_blind_candidates.csv`, NO distribuidos); protocolo de adjudicacion listo.

**Estado**: `HUMAN_AGREEMENT_ANALYZED` + `GOLD_DISCREPANCIES_CANDIDATES_IDENTIFIED` + `ADJUDICATION_PENDING` + `SOURCE_LABELS_UNCHANGED` + `CALIBRATION_V2_1 = PENDING` + `V03 = FROZEN`.

> V03 remains the best verified leaderboard result."""


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
    exec(compile(CODIGO, "<seccion 148>", "exec"), espacio)


def main() -> None:
    """Integra la seccion 148, ejecuta su verificacion y guarda el notebook."""
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
    print("Notebook actualizado con la seccion 148:")
    print(f"  celdas antes: {celdas_antes} | despues: {len(cells)}")
    print("  analisis P93 verificado en aislamiento: OK")
    print("  celdas de codigo compiladas: OK | ASCII puro en lo generado: OK (0 bytes > 127)")


if __name__ == "__main__":
    main()
