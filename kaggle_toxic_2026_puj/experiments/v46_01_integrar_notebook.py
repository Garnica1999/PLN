"""Integra la fase P46 (auditoria del campeon) en el notebook consolidado.

Anade la seccion 42 al final de `notebooks/10_final_pipeline.ipynb`. La celda de
codigo lee `reports/p46_champion_audit.json` en tiempo de ejecucion, de modo que
ningun score esta escrito a mano en el notebook.

Uso
---
    /home/carlos/venv/bin/python experiments/v46_01_integrar_notebook.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

NOTEBOOK = PROJECT_ROOT / "notebooks" / "10_final_pipeline.ipynb"
INFORME = PROJECT_ROOT / "reports" / "p46_champion_audit.json"

TITULO = """\
## 42. P46 - Leaderboard and Champion Audit (2026-09-27)

La pregunta: cual es el verdadero mejor resultado de leaderboard del proyecto, y
cual debe ser la referencia oficial para futuras decisiones.

La contradiccion a resolver era que `CHAMPION.md` atribuia a `v16_ch1` un LB de
0.78535 y a `v16_ch5` un LB de 0.78650, mientras `submissions/leaderboard.log.csv`
registraba como maximo 0.78306.

Jerarquia de evidencia aplicada:

1. archivo de submission
2. registro externo asociado a esa submission
3. leaderboard log estructurado
4. metadata de experimento
5. reports
6. notebook
7. CHAMPION.md
8. texto narrativo

`LABEL_STATE` no aparece en esta seccion: P46 no abre ninguna familia
experimental. 0 modelos, 0 submissions, 0 consultas a Kaggle.
"""

CODIGO = """\
from pathlib import Path
import json

RAIZ = Path.cwd()
while not (RAIZ / "reports").exists() and RAIZ != RAIZ.parent:
    RAIZ = RAIZ.parent
with (RAIZ / "reports" / "p46_champion_audit.json").open(encoding="utf-8") as f:
    d = json.load(f)

ll = d["leaderboard_log"]
v = d["veredicto"]
rf = d["reproduccion_de_la_formula"]
mg = d["magnitudes_separadas"]

print("42.1  INTEGRIDAD DEL LEADERBOARD LOG")
print(f"  filas {ll['n_rows']} | duplicados {ll['duplicados']} | nulos {ll['scores_nulos']} "
      f"| fuera de rango {ll['scores_fuera_de_rango']}")
sin = [f for f in ll["filas"] if not f["artefacto_existe"]]
print(f"  filas con artefacto en disco: {len(ll['filas']) - len(sin)}/{len(ll['filas'])}")
print()
print("42.2  LAS 22 SUBMISSIONS, ORDENADAS POR LB")
print(f"  {'submission':38}{'LB':>9}  {'artefacto':>10}  estado")
for f in sorted(ll["filas"], key=lambda x: -x["leaderboard_f1"]):
    est = "BEST_VERIFIED_LB" if f["leaderboard_f1"] == v["leaderboard_max_verified"] else "REJECTED"
    art = "si" if f["artefacto_existe"] else "NO"
    print(f"  {f['submission_id']:38}{f['leaderboard_f1']:9.5f}  {art:>10}  {est}")
print()
print("42.3  VEREDICTO")
print(f"  CASE                        : {v['CASE']}")
print(f"  leaderboard_max_verified    : {v['leaderboard_max_verified']:.5f} "
      f"({v['leaderboard_max_verified_submission']})")
print(f"  baseline_operativo          : {v['baseline_operativo']}")
print(f"  sha256                      : {v['champion_ofrecional_sha256']}")
print(f"  CHAMPION_STATUS             : {v['CHAMPION_STATUS']}")
print(f"  afirmaciones sin evidencia  : {v['afirmaciones_sin_evidencia_primaria']}")
print()
print("42.4  LAS AFIRMACIONES NO MEDIDAS SON PREDICCIONES")
print(f"  columna de CHAMPION.md: '{d['columna_de_score']}'")
print(f"  formula declarada: {d['formula_declarada'][:90]}")
for c in rf["casos"]:
    print(f"  {c['clave']}: recalculado {c['lb_combinado_recalculado']:.6f} | "
          f"afirmado {c['valor_afirmado']} | desviacion {c['desviacion']:.2e} | "
          f"reproduce {c['reproduce']}")
print(f"  todas reproducen: {rf['todas_reproducen']}")
print()
print("42.5  LAS MEDICIONES REALES DE ESAS MISMAS SUBMISSIONS")
for c in d["contradiccion_interna"]:
    print(f"  {c['clave']} (linea {c['linea']}): {c['valor_en_tabla_de_estado']:.5f} "
          f"{c['estado']}")
print()
print("42.6  LAS CUATRO MAGNITUDES QUE NO DEBEN CONFUNDIRSE")
print(f"  mejor corrected_cv utilizable : {mg['mejor_corrected_cv_utilizable']['valor']:.16f}")
print(f"  mejor local del log          : {mg['mejor_local_del_log']['local_f1']:.5f} "
      f"({mg['mejor_local_del_log']['submission_id']}), LB "
      f"{mg['mejor_local_del_log']['leaderboard_f1']:.5f}")
print(f"  mejor submission por LB      : "
      f"{mg['mejor_submission_por_LB']['leaderboard_f1']:.5f} "
      f"({mg['mejor_submission_por_LB']['submission_id']})")
print(f"  mejor modelo congelado       : "
      f"{mg['mejor_modelo_congelado']['corrected_cv']:.10f} (V03), LB "
      f"{mg['mejor_modelo_congelado']['leaderboard_f1']:.5f}")
"""


def markdown(texto: str) -> dict[str, Any]:
    """Construye una celda markdown.

    Args:
        texto: fuente de la celda.

    Returns:
        La celda como diccionario de notebook.
    """
    return {"cell_type": "markdown", "metadata": {}, "source": texto.splitlines(keepends=True)}


def codigo(texto: str) -> dict[str, Any]:
    """Construye una celda de codigo sin outputs.

    Args:
        texto: fuente de la celda.

    Returns:
        La celda como diccionario de notebook.
    """
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": texto.splitlines(keepends=True),
    }


def main() -> None:
    """Anade la seccion 42 al notebook consolidado.

    Returns:
        None.
    """
    if not INFORME.exists():
        raise SystemExit("PARADA: falta reports/p46_champion_audit.json")
    json.loads(INFORME.read_text(encoding="utf-8"))
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    antes = len(notebook["cells"])
    marcas = " ".join(
        "".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "markdown"
    )
    if "P46 - Leaderboard and Champion Audit" in marcas:
        print("  la seccion 42 ya existe, no se duplica")
        return
    notebook["cells"].extend([markdown(TITULO), codigo(CODIGO)])
    NOTEBOOK.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"  notebook: {antes} -> {len(notebook['cells'])} celdas")
    print("  seccion 42 integrada en el notebook consolidado")


if __name__ == "__main__":
    main()
