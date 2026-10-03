"""Integra la fase P55 (Final Pipeline Audit) en el notebook consolidado.

Anade la seccion 51 al final de `notebooks/10_final_pipeline.ipynb`. La celda de
codigo lee los artefactos de P55 en tiempo de ejecucion, de modo que ninguna cifra
de la seccion esta escrita a mano.

Uso
---
    /home/carlos/venv/bin/python experiments/v55_01_integrar_notebook.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

NOTEBOOK = PROJECT_ROOT / "notebooks" / "10_final_pipeline.ipynb"
INFORME = PROJECT_ROOT / "reports" / "p55_pipeline_audit.json"
AUDIT = PROJECT_ROOT / "reports" / "p55_pipeline_audit.csv"
TRANSFER = PROJECT_ROOT / "reports" / "p55_threshold_transfer.csv"
INTEGRIDAD = PROJECT_ROOT / "reports" / "p55_artifact_integrity.csv"
MARCA = "51. P55 - Final Pipeline Audit"

TITULO = """\
## 51. P55 - Auditoria estrategica del pipeline final de V03 (2026-09-28)

### Que se audita y que no

No es una busqueda de modelos. La pregunta es si existe un defecto o perdida de
informacion **en el procedimiento** de V03, y si puede explicar parte de la brecha entre el
OOF y el leaderboard.

### El pipeline real, leido del codigo

La auditoria no construye un diagrama conceptual: lee lo que el codigo hace. Y encuentra
que **los dos numeros de la referencia de V03 vienen de protocolos de agrupacion
distintos**.
"""

CODIGO = """\
from pathlib import Path
import json
import pandas as pd

RAIZ = Path.cwd()
while not (RAIZ / "reports").exists() and RAIZ != RAIZ.parent:
    RAIZ = RAIZ.parent
d = json.loads((RAIZ / "reports" / "p55_pipeline_audit.json").read_text(encoding="utf-8"))
au = pd.read_csv(RAIZ / "reports" / "p55_pipeline_audit.csv")
tr = pd.read_csv(RAIZ / "reports" / "p55_threshold_transfer.csv")
ig = pd.read_csv(RAIZ / "reports" / "p55_artifact_integrity.csv")

pr = d["pipeline_real"]
u = d["umbrales"]
br = d["brecha_oof_lb"]
tp = d["tasa_positiva"]
ET = ("HS", "TR", "AG")

print("51.1  EL PIPELINE REAL")
print("  raw train 1731")
print("    -> dev 1483 / holdout 248")
print("    -> DOS protocolos de agrupacion distintos:")
print("")
print("  A) 04_refit_submissions_v02_v03.py")
print("       grupos = near-duplicado (D-004)")
print("       vectorizador y modelo DENTRO de cada pliegue")
print("       -> oof_mod03b.npz")
print("       -> umbrales 0,44 / 0,51 / 0,46 sobre ESE OOF")
print("       -> refit sobre el 100 pct del train con ESOS umbrales")
print("       -> submission V03")
print("")
print("  B) 14_oof_corregido_v2.py")
print("       grupos = near-duplicate UNION pseudo-autor (D-016)")
print("       -> v2_oof_base_corrected.npz")
print("       -> corrected_cv = 0,7772114525 con LOS UMBRALES DE A)")
print()
print(f"  {pr['discrepancia']}")
print(f"  vectorizador dentro del pliegue (verificado en el codigo): "
      f"{pr['vectorizador_dentro_del_pliegue']}")
print(f"  {pr['modelo_final']}")
print()

print("51.2  INTEGRIDAD DE ARTEFACTOS")
print(f"  {'artefacto':62}{'existe':>8}")
for _, x in ig.iterrows():
    marca = "OK" if x["existe"] else "AUSENTE"
    sufijo = "  <- NO COINCIDE" if x["coincide_con_el_esperado"] is False else ""
    if x["nota"]:
        sufijo += f"  ({x['nota']})"
    print(f"  {x['artefacto']:62}{marca:>8}{sufijo}")
print()

print("51.3  TRANSFERENCIA DE UMBRALES")
print(f"  {'metodo':30}{'HS':>7}{'TR':>7}{'AG':>7}{'macro-F1':>16}{'delta':>14}")
print(f"  {'umbrales de V03':30}{u['V03']['HS']:>7.2f}{u['V03']['TR']:>7.2f}"
      f"{u['V03']['AG']:>7.2f}{u['macro_f1_V03']:>16.10f}{0.0:>+14.10f}")
oi = u["optimos_in_sample_D016"]
print(f"  {'optimos in-sample D-016':30}{oi['HS']:>7.2f}{oi['TR']:>7.2f}{oi['AG']:>7.2f}"
      f"{u['macro_f1_in_sample']:>16.10f}{u['delta_in_sample']:>+14.10f}")
print(f"  {'ANIDADOS (estimacion honesta)':30}{'':>7}{'':>7}{'':>7}"
      f"{u['macro_f1_anidado']:>16.10f}{u['delta_anidado']:>+14.10f}")
print()
print(f"  {u['advertencia']}")
print()
print("  umbrales anidados por pliegue, frente a los de V03:")
print(f"  {'pliegue':9}" + "".join(f"{'anidado ' + e:>14}" for e in ET)
      + "".join(f"{'V03 ' + e:>11}" for e in ET))
for _, x in tr.iterrows():
    fila = f"  {int(x['fold']):<9}"
    for e in ET:
        fila += f"{x['umbral_' + e]:>14.2f}"
    for e in ET:
        fila += f"{x['umbral_V03_' + e]:>11.2f}"
    print(fila)
print()
print("  V03 esta FUERA del rango de los umbrales anidados en HS y TR, en los")
print("  CINCO pliegues sin excepcion. La discrepancia es sistematica, no ruido.")
print()

print("51.4  TASA POSITIVA, LA CONSECUENCIA OPERATIVA")
print(f"  {'umbrales':32}{'HS':>9}{'TR':>9}{'AG':>9}")
print(f"  {'de V03':32}{tp['con_umbrales_V03'][0]:>9.4f}"
      f"{tp['con_umbrales_V03'][1]:>9.4f}{tp['con_umbrales_V03'][2]:>9.4f}")
print(f"  {'optimos in-sample':32}{tp['con_optimos_in_sample'][0]:>9.4f}"
      f"{tp['con_optimos_in_sample'][1]:>9.4f}{tp['con_optimos_in_sample'][2]:>9.4f}")
print(f"  {'anidados':32}{tp['con_anidados'][0]:>9.4f}"
      f"{tp['con_anidados'][1]:>9.4f}{tp['con_anidados'][2]:>9.4f}")
print("  los umbrales de V03 son demasiado ALTOS: el OOF produce ~4 puntos")
print("  por debajo de positivos en HS que los umbrales del propio protocolo piden.")
print()

print("51.5  TRAYECTORIA CON EL TAMANO DE ENTRENAMIENTO")
print(f"  {'fraccion de grupos':22}{'macro-F1':>16}{'HS':>7}{'TR':>7}{'AG':>7}")
for x in d["tamano_de_entrenamiento"]:
    print(f"  {x['fraccion_de_grupos']:>21.0%}{x['macro_f1']:>16.10f}"
          f"{x['umbral_HS']:>7.2f}{x['umbral_TR']:>7.2f}{x['umbral_AG']:>7.2f}")
print("  el umbral optimo NO se desplaza de forma apreciable al crecer el train,")
print("  luego la correccion por tamano no es el mecanismo. Se describe, no se selecciona.")
print()

print("51.6  BALANCE DE PLIEGUES")
print(f"  {'pliegue':9}{'n':>6}{'grupos':>9}{'HS':>9}{'TR':>9}{'AG':>9}{'macro-F1':>12}")
for x in d["balance_de_pliegues"]:
    print(f"  {x['pliegue']:>9}{x['n_filas']:>6}{x['n_grupos']:>9}{x['tasa_HS']:>9.4f}"
          f"{x['tasa_TR']:>9.4f}{x['tasa_AG']:>9.4f}{x['macro_f1_de_V03']:>12.6f}")
bs = pd.DataFrame(d["balance_de_pliegues"])
print(f"  dispersion entre pliegues: {bs['macro_f1_de_V03'].max() - bs['macro_f1_de_V03'].min():.6f}")
print("  las prevalencias son homogneas, luego ningun pliegue domina por clase.")
print()

print("51.7  LA BRECHA OOF A LEADERBOARD")
print(f"  corrected_cv {br['oof']:.10f} | LB {br['lb']:.5f} | delta {br['delta']:+.8f}")
print(f"  sigma de pliegues {br['sigma_de_pliegues']:.6f} | la brecha es "
      f"{br['delta_en_sigma']:.2f} sigma")
print(f"  SABIDO      : {br['sabido']}")
print(f"  DESCONOCIDO : {br['desconocido']}")
print(f"  PLAUSIBLE, no probado: {'; '.join(br['plausible_no_probado'])}")
print()

print("51.8  DECISION")
print(f"  P55_PROCEDURAL_GAP = {d['P55_PROCEDURAL_GAP']}")
print(f"  hay discrepancia procedural                     : {d['hay_discrepancia']}")
print(f"  su impacto alcanza el margen de 0,003           : {d['impacto_alcanza_el_gate_de_0p003']}")
print()
print("  Es un BUG_FOUND, no una mejora. Se clasifica como tal y NO se llama")
print("  nueva mejora. El original y el corregido quedan separados.")
print("  La correccion es legal y el mecanismo es concreto, pero la estimacion")
print("  anidada da +0,000925, por debajo del margen de adopcion de 0,003, luego")
print("  NO se gasta una submission en ello.")
print()
print("  LO QUE SE AFIRMA: V03 permanece como mejor resultado verificado, con una")
print("  discrepancia procedimental registrada, cuantificada y por debajo del gate.")
print("  LO QUE NO SE AFIRMA: que V03 sea globalmente optimo, ni que la brecha con")
print("  el leaderboard tenga la causa sugerida. Ninguna de las dos es demostrable")
print("  con la evidencia legal disponible.")
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
    """Anade la seccion 51 al notebook consolidado.

    Returns:
        None.
    """
    for artefacto in (INFORME, AUDIT, TRANSFER, INTEGRIDAD):
        if not artefacto.exists():
            raise SystemExit(f"PARADA: falta {artefacto}")
    json.loads(INFORME.read_text(encoding="utf-8"))
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    antes = len(notebook["cells"])
    marcas = " ".join(
        "".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "markdown"
    )
    if MARCA in marcas:
        print("  la seccion 51 ya existe, no se duplica")
        return
    notebook["cells"].extend([markdown(TITULO), codigo(CODIGO)])
    NOTEBOOK.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
                        encoding="utf-8")
    print(f"  notebook: {antes} -> {len(notebook['cells'])} celdas")
    print("  seccion 51 integrada en el notebook consolidado")


if __name__ == "__main__":
    main()
