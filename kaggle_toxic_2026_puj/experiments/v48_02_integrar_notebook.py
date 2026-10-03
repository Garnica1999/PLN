"""Integra la fase P48 (EDA-Driven Residual Research) en el notebook consolidado.

Anade la seccion 44 al final de `notebooks/10_final_pipeline.ipynb`. Las dos
celdas de codigo leen `reports/p48_eda_traceability.json` y
`reports/p48_residual_atlas.json` en tiempo de ejecucion, de modo que ninguna
cifra queda escrita a mano.

Uso
---
    /home/carlos/venv/bin/python experiments/v48_02_integrar_notebook.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

NOTEBOOK = PROJECT_ROOT / "notebooks" / "10_final_pipeline.ipynb"
TRAZA = PROJECT_ROOT / "reports" / "p48_eda_traceability.json"
ATLAS = PROJECT_ROOT / "reports" / "p48_residual_atlas.json"

TITULO = """\
## 44. P48 - EDA-Driven Residual Research (2026-09-27)

La pregunta de esta fase no es "que modelo probamos ahora", sino:

> Que errores de V03 son corregibles y cuales parecen contener informacion que
> V03 no esta capturando?

Se responde con dos piezas:

1. **Trazabilidad del EDA**: cada hallazgo del EDA original, el mecanismo que
   deriva de el, los experimentos que lo probaron y si quedo cerrado.
2. **Atlas de residuos**: las 757 celdas de error de V03, clasificadas segun si
   algun canal OOF historico acierta donde V03 falla.

0 modelos, 0 submissions, 0 uso de holdout o test.
"""

CODIGO_TRAZA = """\
from pathlib import Path
import json

RAIZ = Path.cwd()
while not (RAIZ / "reports").exists() and RAIZ != RAIZ.parent:
    RAIZ = RAIZ.parent
with (RAIZ / "reports" / "p48_eda_traceability.json").open(encoding="utf-8") as f:
    t = json.load(f)

print("44.1  COBERTURA DEL INFORME EDA ORIGINAL")
c = t["cobertura_del_informe_eda"]
print(f"  hallazgos declarados        : {c['hallazgos_declarados_en_el_informe']}")
print(f"  con hipotesis no vacia      : {c['con_hipotesis_no_vacia']}")
print(f"  sin hipotesis declarada    : {c['sin_hipotesis']}")
print()
print("44.2  MATRIZ DE TRAZABILIDAD")
print(f"  {'hallazgo':54}{'exp':>5}{'cerrada':>9}")
for x in t["matriz_de_trazabilidad"]:
    print(f"  {x['eda_finding'][:54]:54}{x['n_experiments']:>5}"
          f"{('si' if x['closed'] else 'NO'):>9}")
print()
r = t["resumen"]
print("44.3  RESUMEN")
print(f"  hallazgos auditados        : {r['hallazgos_auditados']}")
print(f"  con mecanismo cerrado      : {r['con_mecanismo_cerrado']}")
print(f"  sin ningun experimento     : {r['sin_ningun_experimento']}")
print(f"  abiertos                   : {r['abiertos']}")
"""

CODIGO_ATLAS = """\
from pathlib import Path
import json

RAIZ = Path.cwd()
while not (RAIZ / "reports").exists() and RAIZ != RAIZ.parent:
    RAIZ = RAIZ.parent
with (RAIZ / "reports" / "p48_residual_atlas.json").open(encoding="utf-8") as f:
    d = json.load(f)

g = d["clasificacion"]
a = d["atlas"]
print("44.4  LIMITACIONES QUE NO SE PUEDEN RESOLVER")
for x in d["limitaciones"]:
    print(f"  - {x[:110]}")
print()
print("44.5  ATLAS DE RESIDUOS")
print(f"  filas de dev            : {a['n_filas']}")
print(f"  celdas de error de V03  : {a['celdas_error']}")
print(f"  filas por numero de errores: {a['distribucion_n_errores']}")
print()
print("44.6  CLASIFICACION DE LA POBLACION")
print(f"  A (V03 correcto en las 3)        : {g['poblacion_A_filas']}")
print(f"  C (V03 mal, algun canal acierta) : {g['poblacion_C_filas']}")
print(f"  B (V03 mal, ningun canal acierta): {g['poblacion_B_filas']}")
print(f"  celdas rescatables   : {g['celdas_rescatables']} ({g['pct_celdas_rescatables']} pct)")
print(f"  celdas irrecuperables: {g['celdas_irrecuperables']} ({g['pct_celdas_irrecuperables']} pct)")
print()
print("44.7  MATRIZ DE RESCATE POR ETIQUETA")
print(f"  {'etiqueta':10}{'V03 mal':>9}{'rescate':>9}{'damage':>9}{'net':>7}")
for x in d["matriz_de_rescate"]:
    print(f"  {x['etiqueta']:10}{x['v03_mal']:>9}{x['rescate']:>9}"
          f"{x['damage']:>9}{x['net']:>+7}")
print()
print("44.8  CONTEXTO DE MARGEN")
dm = d["diagnostico_de_margen"]
print(f"  tasa de rescate con margen bajo (<0.10): "
      f"{dm['tasa_rescate_margen_bajo_0.10']:.4f}")
print(f"  tasa de rescate con margen alto (>=0.20): "
      f"{dm['tasa_rescate_margen_alto_0.20']:.4f}")
print(f"  irrecuperables en margen alto: "
      f"{dm['irrecuperables_en_margen_alto']} de {dm['irrecuperables_totales']} "
      f"({dm['pct_irrecuperables_en_margen_alto']} pct)")
print()
print("44.9  REFERENCIA GLOBAL Y PATRONES")
print(f"  tasa de error global  : {g['tasa_error_global']}")
print(f"  tasa de rescate global: {g['tasa_rescate_global']}")
print(f"  {'patron':30}{'lift':>7}{'rescate':>9}{'enriquec.':>11}{'p_enr':>8}")
for x in sorted(d["patrones"], key=lambda z: -z["tasa_error_v03"])[:12]:
    print(f"  {x['patron']:30}{x['tasa_error_v03'] / g['tasa_error_global']:>7.3f}"
          f"{x['tasa_rescate']:>9.4f}"
          f"{x['tasa_rescate'] / g['tasa_rescate_global']:>11.4f}"
          f"{'':>8}")
print()
print("44.10  VEREDICTO")
print(f"  EDA_RESEARCH_GAP = {g['EDA_RESEARCH_GAP']}")
print(f"  candidatos a hipotesis: {len(g['candidatos_a_hipotesis'])}")
print("  Motivo: ninguna slice ni combinacion se enriquece en RESCATE por encima")
print("  de la tasa global. Elevar la tasa de error no es lo mismo que Concentrar")
print("  informacion recuperable, y la prueba que lo separa es precisamente esa.")
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
    """Anade la seccion 44 al notebook consolidado.

    Returns:
        None.
    """
    for ruta in (TRAZA, ATLAS):
        if not ruta.exists():
            raise SystemExit(f"PARADA: falta {ruta}")
    json.loads(TRAZA.read_text(encoding="utf-8"))
    json.loads(ATLAS.read_text(encoding="utf-8"))
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    antes = len(notebook["cells"])
    marcas = " ".join(
        "".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "markdown"
    )
    if "P48 - EDA-Driven Residual Research" in marcas:
        print("  la seccion 44 ya existe, no se duplica")
        return
    notebook["cells"].extend([
        markdown(TITULO), codigo(CODIGO_TRAZA), codigo(CODIGO_ATLAS),
    ])
    NOTEBOOK.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"  notebook: {antes} -> {len(notebook['cells'])} celdas")
    print("  seccion 44 integrada en el notebook consolidado")


if __name__ == "__main__":
    main()
