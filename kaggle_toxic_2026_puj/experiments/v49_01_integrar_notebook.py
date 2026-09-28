"""Integra la fase P49 (OOF Rescue Reconciliation) en el notebook consolidado.

Anade la seccion 45 al final de `notebooks/10_final_pipeline.ipynb`. La celda de
codigo lee `reports/p49_rescue_reconciliation.json` en tiempo de ejecucion, de
modo que la contradiccion entre 147 y 715 queda documentada con sus
denominadores y unidades, y no reescrita.

Uso
---
    /home/carlos/venv/bin/python experiments/v49_01_integrar_notebook.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

NOTEBOOK = PROJECT_ROOT / "notebooks" / "10_final_pipeline.ipynb"
INFORME = PROJECT_ROOT / "reports" / "p49_rescue_reconciliation.json"

TITULO = """\
## 45. P49 - OOF Rescue Reconciliation (2026-09-28)

P48 uso dos cifras como si fueran la misma magnitud: 715 de 757 celdas de error
rescatables, y 147 de 757 como `cobertura_de_errores_v03`. Esta seccion no oculta
la contradiccion: la resuelve.

Tres cosas se established:

1. **Los tensores OOF SI eran recuperables.** Estan en los worktrees `PLN` y
   `PLN-v8`. La reconstruccion del pool de 20 canales coincide con
   `reports/hardness_v4.csv` en las 4449 celdas, con diferencia 0.
2. **715 y 147 son magnitudes distintas y ambas son observacion directa.**
   147 es el rescate de **una** variante. 715 es el de la **union de veinte**.
3. **715 no es un artefacto de azar**, pero tampoco es explotable: 19 de los 20
   canales tienen `net` negativo.
"""

CODIGO = """\
from pathlib import Path
import json

RAIZ = Path.cwd()
while not (RAIZ / "reports").exists() and RAIZ != RAIZ.parent:
    RAIZ = RAIZ.parent
with (RAIZ / "reports" / "p49_rescue_reconciliation.json").open(encoding="utf-8") as f:
    d = json.load(f)

t = d["tensores"]
r = d["reproduccion_de_147"]
az = d["prueba_de_azar"]

print("45.1  LOS TENSORES ERAN RECUPERABLES")
print(f"  ubicaciones: {t['ubicaciones']}")
print(f"  canales reconstruidos: {t['canales_reconstruidos']}")
print(f"  validacion vs hardness_v4: max|dif| = "
      f"{t['validacion_contra_hardness_v4']['max_abs_diff']:.2e} "
      f"| ok = {t['validacion_contra_hardness_v4']['ok']}")
print()
print("45.2  QUE SIGNIFICA CADA CIFRA")
print(f"  {'metrica':44}{'valor':>10}  {'unidad':12}{'estado':>10}")
for x in d["tabla_de_reconciliacion"]:
    print(f"  {x['metrica']:44}{x['valor']:>10}  {x['unidad']:12}{x['estado']:>10}")
print()
print("45.3  EL 147 SE REPRODUCE EXACTAMENTE")
print(f"  registrado {r['valor_registrado']} | reproducido {r['valor_reproducido']} "
      f"| reproduce = {r['reproduce']}")
print(f"  cobertura registrada {r['cobertura_registrada']:.15f}")
print(f"  cobertura reproducida {r['cobertura_reproducida']:.15f}")
print()
print("45.4  LA PRUEBA DECISIVA: SENAL O AZAR")
print(f"  un canal acierta una celda de error con p = "
      f"{az['p_medio_de_acierto_en_celda_de_error']:.4f} "
      f"({az['canales_medios_que_acierten']} de {az['canales']})")
print(f"  irrecuperables OBSERVADAS: {az['irrecuperables_observadas']}")
print(f"  irrecuperables ESPERADAS si fueran independientes: "
      f"{az['irrecuperables_esperadas_independencia']}")
print(f"  es azar: {az['es_azar']}")
print()
print("45.5  MATRIZ DE RESCATE Y DAMAGE")
print(f"  {'canal':36}{'rescate':>8}{'damage':>8}{'net':>8}")
for x in d["matriz_de_rescate_por_canal"]:
    print(f"  {x['canal']:36}{x['rescate']:>8}{x['damage']:>8}{x['net']:>+8}")
positivos = [x for x in d["matriz_de_rescate_por_canal"] if x["net"] > 0]
print(f"  canales con net positivo: {len(positivos)} de {len(d['matriz_de_rescate_por_canal'])}")
for x in positivos:
    print(f"    {x['canal']} (net {x['net']:+d})")
print()
print("45.6  MARGEN SOBRE EL RESCATE OBSERVADO")
print(f"  {'etiqueta':10}{'bin':14}{'errores':>9}{'rescate':>9}{'irrecup.':>10}")
for x in d["margen_sobre_rescate_observado"]:
    print(f"  {x['etiqueta']:10}{x['bin']:14}{x['n_errores']:>9}"
          f"{x['tasa_rescate_observada']:>9.4f}{x['irrecuperables_observados']:>10}")
print()
print("45.7  ERROR-DENSE FRENTE A RESCUE-DENSE")
print(f"  {'slice':22}{'lift_err':>10}{'lift_resc':>11}{'lift_irrec':>12}  clase")
for x in d["slices_error_dense_vs_rescue_dense"]:
    print(f"  {x['slice']:22}{x['lift_error']:>10.3f}{x['lift_rescate']:>11.3f}"
          f"{x['lift_irrecuperable']:>12.3f}  {x['clase']}")
print()
print("45.8  VEREDICTO")
print(f"  CASE = {d['decision']}")
print("  715 es OBSERVACION DIRECTA, no cota superior. No es azar.")
print("  Pero ningun canal del pool tiene net positivo salvo el propio V03 con")
print("  umbral 0.5, luego el rescate es real y difuso, y no se puede cobrar.")
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
    """Anade la seccion 45 al notebook consolidado.

    Returns:
        None.
    """
    if not INFORME.exists():
        raise SystemExit("PARADA: falta reports/p49_rescue_reconciliation.json")
    json.loads(INFORME.read_text(encoding="utf-8"))
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    antes = len(notebook["cells"])
    marcas = " ".join(
        "".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "markdown"
    )
    if "P49 - OOF Rescue Reconciliation" in marcas:
        print("  la seccion 45 ya existe, no se duplica")
        return
    notebook["cells"].extend([markdown(TITULO), codigo(CODIGO)])
    NOTEBOOK.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"  notebook: {antes} -> {len(notebook['cells'])} celdas")
    print("  seccion 45 integrada en el notebook consolidado")


if __name__ == "__main__":
    main()
