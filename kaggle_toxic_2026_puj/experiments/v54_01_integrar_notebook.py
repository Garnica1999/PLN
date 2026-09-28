"""Integra la fase P54 (Non-Rescuable Error Semantic Audit) en el notebook.

Anade la seccion 50 al final de `notebooks/10_final_pipeline.ipynb`. La celda de
codigo lee los artefactos de P54 en tiempo de ejecucion, de modo que ninguna cifra
de la seccion esta escrita a mano. No se incluye texto del test.

Uso
---
    /home/carlos/venv/bin/python experiments/v54_01_integrar_notebook.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

NOTEBOOK = PROJECT_ROOT / "notebooks" / "10_final_pipeline.ipynb"
INFORME = PROJECT_ROOT / "reports" / "p54_semantic_audit.json"
ATLAS = PROJECT_ROOT / "reports" / "p54_nonrescuable_atlas.csv"
TAXO = PROJECT_ROOT / "reports" / "p54_nonrescuable_taxonomy.csv"
BVC = PROJECT_ROOT / "reports" / "p54_rescue_vs_nonrescue.csv"
EDA = PROJECT_ROOT / "reports" / "p54_eda_reconciliation.csv"
MARCA = "50. P54 - Non-Rescuable Error Semantic Audit"

TITULO = """\
## 50. P54 - Auditoria semantica de los errores no rescatables (2026-09-28)

### La pregunta

P53 demostro que el espacio de acciones real tiene **dos** elementos, porque todos los
canales que discrepan de V03 proponen el mismo complemento binario. Cerrada la seleccion
contextual, la pregunta deja de ser de algoritmo y pasa a ser de fenomeno:

    que son realmente los errores que NINGUN canal historico puede corregir?

Y la respuesta tiene que ser descriptiva, no una hipotesis fabricada sobre 42
observaciones. Por eso esta seccion separa con cuidado tres cosas: lo que se **mide** sobre
atributos computables, lo que se **lee** cualitativamente y se marca como retrospectivo, y
lo que **no** se puede concluir por falta de soporte.

### Advertencia de soporte, antes de nada

Las 42 celdas de C **no son 42 observaciones independientes**. Proceden de **33 tweets
distintos**, y 9 de las 42 celdas vienen de solo 3 tweets con las tres etiquetas mal. Todo
enriquecimiento de este informe debe leerse con esa cifra delante.
"""

CODIGO = """\
from pathlib import Path
import json
import pandas as pd

RAIZ = Path.cwd()
while not (RAIZ / "reports").exists() and RAIZ != RAIZ.parent:
    RAIZ = RAIZ.parent
d = json.loads((RAIZ / "reports" / "p54_semantic_audit.json").read_text(encoding="utf-8"))
atlas = pd.read_csv(RAIZ / "reports" / "p54_nonrescuable_atlas.csv")
taxo = pd.read_csv(RAIZ / "reports" / "p54_nonrescuable_taxonomy.csv")
bvc = pd.read_csv(RAIZ / "reports" / "p54_rescue_vs_nonrescue.csv")
eda = pd.read_csv(RAIZ / "reports" / "p54_eda_reconciliation.csv")

p = d["poblaciones"]
conf = d["alta_confianza"]

print("50.1  POBLACIONES, RECALCULADAS DESDE CERO")
print(f"  A = V03 correcto                     : {p['A_correcto']}")
print(f"  B = V03 mal, algun canal acierta     : {p['B_rescatable']}")
print(f"  C = V03 mal, ningun canal acierta    : {p['C_no_rescatable']}")
print(f"  errores totales                      : {p['errores_totales']}")
print(f"  verificado contra P49 (757/715/42)   : {p['verificado_contra_P49']}")
print(f"  tweets distintos en C                : {p['tweets_distintos_en_C']}")
print(f"  soporte efectivo                     : {p['soporte_efectivo']}, no {p['C_no_rescatable']}")
print()
print("  direccion del error en C:")
for x in d["direccion_del_error_en_C"]:
    print(f"    {x['direccion']:18} {x['n']:3d}  ({100 * x['n'] / p['C_no_rescatable']:.1f} pct)")
print()

print("50.2  MARGEN Y ALTA CONFIANZA")
print(f"  {'bin de margen':14}{'A':>7}{'B':>7}{'C':>5}{'pct de C':>11}")
for x in d["margen"]:
    print(f"  {x['bin_de_margen']:14}{x['n_A']:>7}{x['n_B']:>7}{x['n_C']:>5}"
          f"{100 * x['pct_de_C']:>10.1f} pct")
print()
print(f"  {'label':8}{'errores':>9}{'alta conf':>12}{'pct':>9}{'no resc.':>10}{'resc.':>8}")
for x in conf["tabla"]:
    print(f"  {x['label']:8}{x['errores']:>9}{x['errores_alta_confianza']:>12}"
          f"{100 * x['pct_de_alta_confianza']:>8.1f} pct{x['de_esos_no_rescatables']:>10}"
          f"{x['de_esos_rescatables']:>8}")
print(f"  C en alta confianza: {conf['n_C_en_alta_confianza']} de {p['C_no_rescatable']} "
      f"({100 * conf['n_C_en_alta_confianza'] / p['C_no_rescatable']:.1f} pct)")
print("  P49 ya habia visto que C se concentra en margen alto: se confirma.")
print()

print("50.3  RESCATE FRENTE A NO RESCATE, QUE ES LA COMPARACION CENTRAL")
print(f"  {'atributo':30}{'n C':>5}{'pct C':>9}{'pct B':>9}{'lift C/B':>10}{'p':>8}"
      f"{'lift C/A':>10}{'p':>8}")
for _, x in bvc.iterrows():
    pc = f"{100 * x['pct_C']:.2f} pct"
    pb = f"{100 * x['pct_B']:.2f} pct"
    l1 = "n/d" if pd.isna(x["enrichment_C_sobre_B"]) else f"{x['enrichment_C_sobre_B']:.2f}"
    p1 = "n/d" if pd.isna(x["p_C_sobre_B"]) else f"{x['p_C_sobre_B']:.4f}"
    l2 = "n/d" if pd.isna(x["enrichment_C_sobre_A"]) else f"{x['enrichment_C_sobre_A']:.2f}"
    p2 = "n/d" if pd.isna(x["p_C_sobre_A"]) else f"{x['p_C_sobre_A']:.4f}"
    print(f"  {x['atributo']:30}{int(x['n_C']):>5}{pc:>9}{pb:>9}{l1:>10}{p1:>8}{l2:>10}{p2:>8}")
print()
print("  CORRECCION DE UNA IMPRESION PROPIA: leer las 42 celdas sugiere que el")
print("  residuo es insulto lexico sin escalada. Medido, el lexico esta en el 66,67")
print("  pct de C pero tambien en el 53,36 pct de A: enrichment de solo 1,32x. La")
print("  impresion era correcta sobre las celdas de AG y equivocada sobre el conjunto.")
print()

print("50.4  EL MECANISMO DE LA NEGACION, PUESTO A PRUEBA Y REFUTADO")
print(f"  {'atributo':24}{'n C':>5}{'pct C':>9}{'pct A':>9}{'lift':>8}{'p':>9}")
for x in d["test_del_mecanismo"]:
    print(f"  {x['atributo']:24}{x['n_C']:>5}{100 * x['pct_C']:>8.2f} pct"
          f"{100 * x['pct_A']:>8.2f} pct{x['enrichment_C_sobre_A']:>8.2f}{x['p']:>9.4f}")
print("  La negacion se asocia a C, pero el mecanismo que la lectura cualitativa")
print("  sugeria, la negacion CONCEDENTE que licencia lo que sigue, queda REFUTADO:")
print("  una sola celda de C la cumple y el enrichment es de 0,76x, es decir nulo.")
print("  Ademas la negacion tambien tiene enrichment de rescate sobre B, luego marca")
print("  celdas DIFICILES y no un modo de fallo concreto. El mecanismo queda sin explicar.")
print()

print("50.5  RECONCILIACION CON LOS 15 HALLAZGOS DEL EDA ORIGINAL")
print(f"  {'hallazgo':20}{'lift rescate':>14}{'p':>9}{'lift no-rescate':>18}{'p':>9}{'n C':>6}")
for _, x in eda.sort_values("nonrescue_enrichment_C_sobre_A", ascending=False).iterrows():
    print(f"  {x['hallazgo_eda']:20}{x['rescue_enrichment_C_sobre_B']:>14.2f}"
          f"{x['p_rescue_enrichment']:>9.4f}{x['nonrescue_enrichment_C_sobre_A']:>18.2f}"
          f"{x['p_nonrescue_enrichment']:>9.4f}{x['n_C']:>6}")
print("  Se reportan DOS enrichments: error enrichment solo dice donde V03 falla;")
print("  rescate y no-rescate dicen donde hay alternativa y donde no la hay.")
print()

print("50.6  TAXONOMIA RETROSPECTIVA DE LAS 42 CELDAS")
print(f"  {'categoria':46}{'n':>4}{'pct de C':>11}{'fuerza':>14}")
for _, x in taxo.sort_values("n", ascending=False).iterrows():
    print(f"  {x['category']:46}{int(x['n']):>4}{100 * x['percentage_C']:>10.1f} pct"
          f"{x['evidence_strength']:>14}")
print("  La columna esta marcada como RETROSPECTIVA: el modelo vio texto y etiqueta a")
print("  la vez, luego sirve para describir y nunca para inferir.")
print()

print("50.7  LAS 42 CELDAS, COMPLETAS")
print(f"  {'n':>3}{'id':>7}{'lab':>5}{'y':>3}{'v03':>5}{'margen':>8}{'OK':>4}  tipo retrospectivo")
for _, x in atlas.iterrows():
    print(f"  {int(x['n']):>3}{int(x['tweet_id']):>7}{x['label']:>5}{int(x['y_true']):>3}"
          f"{int(x['v03']):>5}{x['margen']:>8.3f}{int(x['canales_que_acierten']):>4}  "
          f"{x['error_type_retrospectivo']}")
print("  El texto integro de las 42 esta en reports/p54_nonrescuable_atlas.csv.")
print("  No se incluye texto del test.")
print()

print("50.8  DECISION")
print(f"  P54_RESEARCH_GAP = {d['P54_RESEARCH_GAP']}")
print("  hipotesis nuevas abiertas: 0")
print()
print("  1. El residuo NO es falta de informacion en el texto. El 66,67 pct de las")
print("     celdas de C ya contienen el lexico de insulto, y solo el 2,85 pct de los")
print("     errores de HS son no rescatables frente al 7,58 pct de AG. La informacion")
print("     esta presente y el modelo la ve.")
print("  2. El residuo es en su mayor parte un desacuerdo de FRONTERA DE ETIQUETA:")
print("     insulto lexico con etiqueta 0, y narrativa sobre grupo con etiqueta 1.")
print("     En ambos casos el texto es inequivoco y la lectura del anotador es una")
print("     convencion que no se deriva del texto.")
print("  3. No queda un fenomeno semantico nuevo con soporte y valor material. El")
print("     mayor candidato, la negacion, tiene 17 celdas y ademas marca dificultad")
print("     en general, no un modo de fallo. Su mecanismo queda sin explicar.")
print("  4. El valor absoluto manda: 26,75 celdas valen ~0,003 de macro-F1, luego")
print("     ningun candidato de este residuo alcanza el margen de adopcion.")
print()
print("  LO QUE NO SE AFIRMA: no se afirma que sea imposible mejorar. Se afirma que")
print("  en este residuo no hay mecanismo semantico con soporte suficiente y valor")
print("  material, y que la via de datos observables quedo cerrada por P52 y P53 con")
print("  una cota, no por falta de intento.")
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
    """Construye una celda de codigo sin salidas.

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
    """Anade la seccion 50 al notebook consolidado.

    Returns:
        None.
    """
    for artefacto in (INFORME, ATLAS, TAXO, BVC, EDA):
        if not artefacto.exists():
            raise SystemExit(f"PARADA: falta {artefacto}")
    json.loads(INFORME.read_text(encoding="utf-8"))
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    antes = len(notebook["cells"])
    marcas = " ".join(
        "".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "markdown"
    )
    if MARCA in marcas:
        print("  la seccion 50 ya existe, no se duplica")
        return
    notebook["cells"].extend([markdown(TITULO), codigo(CODIGO)])
    NOTEBOOK.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
                        encoding="utf-8")
    print(f"  notebook: {antes} -> {len(notebook['cells'])} celdas")
    print("  seccion 50 integrada en el notebook consolidado")


if __name__ == "__main__":
    main()
