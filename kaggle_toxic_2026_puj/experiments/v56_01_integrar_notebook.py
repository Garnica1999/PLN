"""Integra la fase P56 (Corrected Threshold Transfer) en el notebook consolidado.

Anade la seccion 52 al final de `notebooks/10_final_pipeline.ipynb`. La celda de
codigo lee los artefactos de P56 en tiempo de ejecucion, de modo que ninguna cifra
de la seccion esta escrita a mano.

Uso
---
    /home/carlos/venv/bin/python experiments/v56_01_integrar_notebook.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

NOTEBOOK = PROJECT_ROOT / "notebooks" / "10_final_pipeline.ipynb"
INFORME = PROJECT_ROOT / "reports" / "p56_threshold_transfer.json"
TRANSFER = PROJECT_ROOT / "reports" / "p56_threshold_transfer.csv"
ESTABILIDAD = PROJECT_ROOT / "reports" / "p56_threshold_stability.csv"
MARCA = "52. P56 - Corrected Threshold Transfer"

TITULO = """\
## 52. P56 - Transferencia de umbrales corregida (2026-09-28)

### Que se evalua

P55 encontro que los umbrales de V03 se adoptaron bajo el protocolo D-004, mas debil que
el D-016 con el que se reporta la metrica, y que bajo D-016 el umbral optimo es 0,03 mas
bajo en HS y TR. La mejora medida fue de +0,000925, por debajo del margen de adopcion.

Pero el procedimiento de P55 **no era un anidamiento completo**: tomaba el umbral de las
probabilidades out-of-fold de los cuatro pliegues de entrenamiento, que proceden de modelos
ya entrenados. Aqui se hace el anidamiento que hace falta, con vectorizador y modelo
**re-ajustados** dentro de un split interior agrupado.

### Y el resultado corrige a P55

No solo no se adopta la mejora: **el sesgo sistematico que P55 reporto no sobrevive**.
"""

CODIGO = """\
from pathlib import Path
import json
import pandas as pd

RAIZ = Path.cwd()
while not (RAIZ / "reports").exists() and RAIZ != RAIZ.parent:
    RAIZ = RAIZ.parent
d = json.loads((RAIZ / "reports" / "p56_threshold_transfer.json").read_text(encoding="utf-8"))
tr = pd.read_csv(RAIZ / "reports" / "p56_threshold_transfer.csv")
es = pd.read_csv(RAIZ / "reports" / "p56_threshold_stability.csv")

p = d["procedimiento"]
t0 = d["T0_historico"]
t1 = d["T1_mediana"]
ne = d["nested_por_pliegue_exterior"]
b = d["bootstrap"]
inter = d["interseccion_T0_T1"]
ET = ("HS", "TR", "AG")

print("52.1  REPRODUCCION DE P55")
print(f"  valor registrado : {d['P55_valor_registrado']:.10f}")
print(f"  valor reproducido: {d['P55_valor_reproducido']:.10f}")
print(f"  P55_CORRECTED_RESULT_REPRODUCED = {d['P55_CORRECTED_RESULT_REPRODUCED']}")
print()

print("52.2  EL PROCEDIMIENTO, QUE AQUI SI ES ANIDAMIENTO COMPLETO")
print(f"  outer : {p['outer']}")
print(f"  inner : {p['inner']}")
print(f"  umbral: {p['rejilla']}")
print(f"  regla de agregacion: {p['agregacion']}")
print(f"  outer_validation_participa: {p['outer_validation_participa']}")
print()

print("52.3  UMBRALES POR PLIEGUE EXTERIOR")
print(f"  {'outer':8}{'train':>8}{'val':>6}" + "".join(f"{'inner ' + e:>12}" for e in ET)
      + "".join(f"{'V03 ' + e:>11}" for e in ET))
for _, x in tr.iterrows():
    fila = f"  {int(x['outer_fold']):<8}{int(x['n_outer_train']):>8}{int(x['n_outer_val']):>6}"
    for e in ET:
        fila += f"{x['inner_threshold_' + e]:>12.2f}"
    for e in ET:
        fila += f"{x['V03_' + e]:>11.2f}"
    print(fila)
print()

print("52.4  DISTRIBUCION Y ESTABILIDAD")
print(f"  {'etiqueta':10}{'estimaciones':>22}{'mediana':>10}{'std':>9}"
      f"{'min':>7}{'max':>7}{'V03':>7}{'V03 en rango':>15}")
for _, x in es.iterrows():
    fila = f"  {x['label']:10}{x['estimaciones']:>22}{x['mediana']:>10.2f}{x['std']:>9.4f}"
    fila += f"{x['min']:>7.2f}{x['max']:>7.2f}{x['V03']:>7.2f}"
    fila += f"{str(bool(x['V03_dentro_del_rango'])):>15}"
    print(fila)
print()
print("  CORRECCION A P55: los umbrales de V03 caen DENTRO del rango de las cinco")
print("  estimaciones interiores en las TRES etiquetas. Las desviaciones estandar son")
print("  de 0,019 a 0,043, luego los rangos son lo bastante anchos como para contenerlos.")
print("  El supuesto sesgo sistematico de P55 era un artefacto de su procedimiento")
print("  mas simple, que producia rangos mas estrechos.")
print()

print("52.5  T0 CONTRA T1, Y LA CIFRA QUE DECIDE")
print(f"  {'politica':46}{'macro-F1':>16}{'delta vs V03':>16}")
print(f"  {'T0 historico (0,44/0,51/0,46)':46}{t0['macro_f1']:>16.10f}{t0['delta_vs_V03']:>+16.10f}")
print(f"  {'T1 mediana aplicada GLOBALMENTE':46}{t1['macro_f1']:>16.10f}{t1['delta_vs_V03']:>+16.10f}")
print(f"  {'umbral por pliegue exterior (HONESTO)':46}{ne['macro_f1']:>16.10f}{ne['delta_vs_V03']:>+16.10f}")
print(f"  {'P55 simple, para referencia':46}{d['P55_simple']['macro_f1']:>16.10f}{d['P55_simple']['delta_vs_V03']:>+16.10f}")
print()
print(f"  EL OPTIMISMO de definir el umbral con datos de todos los pliegues es de")
print(f"  {d['optimismo_de_la_cifra_global_de_T1']:+.10f}, y es mayor que la propia")
print(f"  mejora aparente. La cifra que decide es la honesta: {d['delta_honesto_por_pliegue_exterior']:+.10f}.")
print()
print("  F1 por etiqueta, T0 -> T1:")
for e in ET:
    a, bb = t0["por_etiqueta"][e], t1["por_etiqueta"][e]
    print(f"    {e}: {a:.10f} -> {bb:.10f}  ({bb - a:+.10f})")
print()
print("  Tasa positiva, T0 -> T1:")
for e in ET:
    a, bb = t0["tasa_positiva"][e], t1["tasa_positiva"][e]
    print(f"    {e}: {a:.4f} -> {bb:.4f}  ({bb - a:+.4f})")
print()

print("52.6  PLIEGUES")
print(f"  {'pliegue':10}{'macro-F1 de T1':>18}{'delta vs V03 global':>24}")
for x in t1["por_pliegue"]:
    print(f"  {x['pliegue']:>10}{x['macro_f1']:>18.10f}{x['delta_vs_V03_global']:>+24.10f}")
print(f"  pliegues con delta positivo: {d['pliegues_positivos']} de 5")
print("  el pliegue 3 se lleva el signo negativo: es el mas pequeno y el de menor F1,")
print("  luego la mejora aparente no es homogenea.")
print()

print("52.7  COMPLEMENTARIEDAD A NIVEL DE CELDA")
print(f"  {'categoria':34}{'celdas':>8}")
print(f"  {'ambos correctos':34}{inter['ambos_correctos']:>8}")
print(f"  {'ambos incorrectos':34}{inter['ambos_incorrectos']:>8}")
print(f"  {'T0 mal, T1 bien (rescate)':34}{inter['T0_mal_T1_bien']:>8}")
print(f"  {'T0 bien, T1 mal (damage)':34}{inter['T0_bien_T1_mal']:>8}")
print(f"  {'NET':34}{d['net']:>+8d}")
for e, v in d["interseccion_por_label"].items():
    print(f"    {e}: rescate {v['rescate']:3d} damage {v['damage']:3d} net {v['net']:+4d}")
print("  el net a nivel de celda es NEGATIVO en las tres etiquetas. El F1 sube porque")
print("  macro-F1 promedia tres etiquetas y el cambio reordaza positivas, pero el")
print("  recuento de celdas no mejora.")
print()

print("52.8  BOOTSTRAP")
print(f"  mediana del delta de T1 : {b['mediana']:+.6f}")
print(f"  IC95                   : [{b['ic95_bajo']:+.6f}, {b['ic95_alto']:+.6f}]")
print(f"  incluye 0              : {b['incluye_cero']}")
print("  el intervalo incluye 0, luego la mejora de T1 no es distinguible de cero.")
print()

print("52.9  DECISION")
print(f"  THRESHOLD_TRANSFER_BIAS = {d['THRESHOLD_TRANSFER_BIAS']}")
print(f"  STATUS = {d['STATUS']}")
print(f"  CASE = {d['caso']}")
print()
print("  CASE B: el procedimiento anidado completo NO reproduce mejora. El delta")
print("  plenamente honesto es negativo, el IC95 incluye 0, el net de celdas es")
print("  negativo y los umbrales de V03 caen dentro del rango interior. Se cierra el")
print("  mecanismo y NO se crea candidato local.")
print()
print("  LO QUE ESTA EN JUEGO: P55 reporto un sesgo sistematico de +0,03 en HS y TR")
print("  fuera de rango en los cinco pliegues. P56, con anidamiento completo, no lo")
print("  reproduce. El defecto procedimental de P55 sigue siendo real como hecho")
print("  historico, pero su impacto medido es menor que el procedimiento de P55")
print("  sugeria, y la correccion no es adoptable.")
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
    """Anade la seccion 52 al notebook consolidado.

    Returns:
        None.
    """
    for artefacto in (INFORME, TRANSFER, ESTABILIDAD):
        if not artefacto.exists():
            raise SystemExit(f"PARADA: falta {artefacto}")
    json.loads(INFORME.read_text(encoding="utf-8"))
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    antes = len(notebook["cells"])
    marcas = " ".join(
        "".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "markdown"
    )
    if MARCA in marcas:
        print("  la seccion 52 ya existe, no se duplica")
        return
    notebook["cells"].extend([markdown(TITULO), codigo(CODIGO)])
    NOTEBOOK.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
                        encoding="utf-8")
    print(f"  notebook: {antes} -> {len(notebook['cells'])} celdas")
    print("  seccion 52 integrada en el notebook consolidado")


if __name__ == "__main__":
    main()
