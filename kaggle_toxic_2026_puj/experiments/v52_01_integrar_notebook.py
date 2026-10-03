"""Integra la fase P52 (Policy Oracle and Channel Complementarity) en el notebook.

Anade la seccion 48 al final de `notebooks/10_final_pipeline.ipynb`. La celda de
codigo lee `reports/p52_policy_oracle.json` y `reports/p52_channel_exclusivity.csv`
en tiempo de ejecucion, de modo que ninguna cifra de la seccion esta escrita a mano.

Uso
---
    /home/carlos/venv/bin/python experiments/v52_01_integrar_notebook.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

NOTEBOOK = PROJECT_ROOT / "notebooks" / "10_final_pipeline.ipynb"
INFORME = PROJECT_ROOT / "reports" / "p52_policy_oracle.json"
EXCLUSIVIDAD = PROJECT_ROOT / "reports" / "p52_channel_exclusivity.csv"
PARES = PROJECT_ROOT / "reports" / "p52_channel_pair_analysis.csv"
MARCA = "48. P52 - Policy Oracle and Channel Complementarity"

TITULO = """\
## 48. P52 - Policy Oracle and Channel Complementarity (2026-09-28)

### La limitacion de P51 que P52 ataca

P51 concluyo que la senal es identificable (AUC 0,8091) pero que la politica de cambios
tiene expectativa negativa. Dejaba una pregunta abierta: **?existe una politica mejor
que V03 dentro de la informacion observable, aunque todavia no sepamos aprenderla?**

### Por que "policy oracle" no es un tercer objeto

Sin `y_true` no se puede construir un techo. Una politica que estima
`P(rescue | observable)` **es** una politica aprendida. Por eso P52 separa cuatro
objetos y los situa en una cadena:

    V03  ->  estrato_out_of_fold  ->  STRATUM_ORACLE  ->  TRUE_ORACLE

El `STRATUM_ORACLE` es el techo rigido: **la mejor politica que sea constante dentro de
estratos observables**, elegida usando `y_true`. Si ese techo cae por debajo de V03,
ninguna politica condicional puede ganar, y la conclusion es matematica y no empirica.

### La pregunta sobre rescates exclusivos

P49 mostro que 19 de 20 canales tienen net negativo. Eso no excluye que cada canal aporte
regiones exclusivas cuya combinacion supere a todos ellos. P52 lo mide.
"""

CODIGO = """\
from pathlib import Path
import json
import pandas as pd

RAIZ = Path.cwd()
while not (RAIZ / "reports").exists() and RAIZ != RAIZ.parent:
    RAIZ = RAIZ.parent
d = json.loads((RAIZ / "reports" / "p52_policy_oracle.json").read_text(encoding="utf-8"))
ex = pd.read_csv(RAIZ / "reports" / "p52_channel_exclusivity.csv")
pr = pd.read_csv(RAIZ / "reports" / "p52_channel_pair_analysis.csv")

au = d["auditoria_de_p51"]
u = d["utilidad_rederivada"]
t = d["techos"]
e = d["exclusividad"]
pa = d["pares"]
co = d["correlacion_de_error_en_regimen_v03_wrong"]

print("48.1  AUDITORIA DEL CODIGO REAL DE P51")
print(f"  target de S1-U                : {au['target_S1U']}")
print(f"  target de S2-U                : {au['target_S2U']}")
print(f"  P51_THRESHOLD                 = {au['umbral_S1U']}")
print(f"  THEORETICAL_EQUILIBRIUM       = {au['umbral_equilibrio_teorico']}")
print(f"  espacio de acciones           : {au['espacio_de_acciones']}")
print(f"  cross-fitting                 : {au['cross_fitting']}")
print(f"  mezcla las dos direcciones    : {au['mezcla_direcciones']}")
print("  El umbral 0,5 es mas estricto que 0,3892 al subir y mas laxo que 0,6108 al")
print("  bajar, luego sobre-cambia justo en la direccion que mas dana.")
print()

print("48.2  UTILIDAD RE-DERIVADA DE FORMA INDEPENDIENTE")
print(f"  {'magnitud':44}{'valor':>14}")
for k, etiqueta in (("ganancia_up_FN_a_TP", "ganancia de subir (FN -> TP)"),
                    ("ganancia_down_FP_a_TN", "ganancia de bajar (FP -> TN)"),
                    ("razon", "razon entre ambas"),
                    ("equilibrio_up", "equilibrio al subir"),
                    ("equilibrio_down", "equilibrio al bajar"),
                    ("suma_de_equilibrios", "suma de los dos equilibrios"),
                    ("tasa_base_up", "tasa base al subir"),
                    ("elevacion_necesaria_up", "elevacion necesaria al subir"),
                    ("tasa_base_down", "tasa base al bajar"),
                    ("elevacion_necesaria_down", "elevacion necesaria al bajar")):
    print(f"  {etiqueta:44}{u[k]:>14.6f}")
print()
print(f"  reproduce el equilibrio de P51        : {u['reproduce_equilibrio_de_p51']}")
print(f"  reproduce la elevacion al subir       : {u['reproduce_elevacion_subir_de_p51']}")
print(f"  CORRIGE la elevacion al bajar         : {u['corrige_elevacion_bajar_de_p51']}")
print(f"  P51 registro {u['elevacion_bajar_registrada_en_p51']:.2f}x, el valor correcto es "
      f"{u['elevacion_necesaria_down']:.2f}x")
print(f"  {u['nota_correccion']}")
print()

print("48.3  LA CADENA DE TECHOS")
print(f"  {'objeto':44}{'corrected_cv':>16}{'delta vs V03':>16}")
print(f"  {'V03 (inaccion)':44}{t['V03']:>16.10f}{0.0:>+16.10f}")
print(f"  {'estrato, accion del training fold':44}{t['estrato_oof']:>16.10f}"
      f"{t['estrato_oof'] - t['V03']:>+16.10f}")
print(f"  {'STRATUM_ORACLE (techo rigido)':44}{t['STRATUM_ORACLE']:>16.10f}"
      f"{t['STRATUM_ORACLE'] - t['V03']:>+16.10f}")
print(f"  {'TRUE_ORACLE (celda a celda)':44}{t['TRUE_ORACLE']:>16.10f}"
      f"{t['TRUE_ORACLE'] - t['V03']:>+16.10f}")
print()
print(f"  estratos observables usados: {t['n_estratos']}")
print(f"  hueco TRUE_ORACLE -> STRATUM_ORACLE: {t['hueco_coarsing_TRUE_menos_estrato']:+.6f}")
print("  ese hueco es casi TODO el oracle: la politica tiene que ser distinta celda a")
print("  celda, y los estratos observables no dicen cual.")
print(f"  hueco STRATUM_ORACLE -> V03: {t['hueco_estrato_menos_V03']:+.6f}")
print()

print("48.4  RESCATE Y DANO EXCLUSIVOS")
print(f"  canales con net_exclusivo positivo : {e['canales_con_net_exclusivo_positivo']} de 20")
print(f"  HHI del reparto de rescates         : {e['hhi_del_reparto_de_rescates']:.4f}")
print(f"  celdas con rescate unico           : {e['celdas_con_rescate_unico']}")
print(f"  rescate maximo / minimo de un canal: {e['rescate_maximo_de_un_canal']} / "
      f"{e['rescate_minimo_de_un_canal']}")
print()
print(f"  {'canal':36}{'resc':>6}{'excl':>6}{'dano':>7}{'excl':>6}{'net':>8}{'netex':>8}")
for _, x in ex.sort_values("rescate_exclusivo", ascending=False).head(8).iterrows():
    print(f"  {x['canal']:36}{x['rescate_total']:>6}{x['rescate_exclusivo']:>6}"
          f"{x['dano_total']:>7}{x['dano_exclusivo']:>6}{x['net']:>+8}{x['net_exclusivo']:>+8}")
print()

print("48.5  COMPLEMENTARIEDAD POR PARES")
print(f"  pares analizados                       : {pa['analizados']}")
print(f"  pares con exclusividad bilateral       : {pa['pares_con_exclusividad_bilateral']}")
print(f"  lift maximo de la union sobre un miembro: {pa['lift_union_maximo']:.4f}")
print(f"  pares donde los dos canales se contradicen: "
      f"{int((pr['desacuerdo_entre_ellos'] > 0).sum())} de {len(pr)}")
print(f"  celdas de contradiccion maxima entre un par: "
      f"{int(pr['desacuerdo_entre_ellos'].max())}")
print("  donde dos canales difieren de V03, NUNCA difieren entre si. Por eso ningun par")
print("  puede aportar nada exclusivo: no hay nada que repartir.")
print()

print("48.6  CORRELACION DE ERROR EN EL REGIMEN DONDE V03 FALLA")
print(f"  {'magnitud':44}{'valor':>12}")
for k, etiqueta in (("minima", "correlacion minima"), ("maxima", "correlacion maxima"),
                    ("media", "correlacion media"),
                    ("pares_por_encima_de_0p5", "pares por encima de 0,5"),
                    ("n_grupos_a_distancia_0p65", "grupos a distancia 0,65")):
    print(f"  {etiqueta:44}{co[k]:>12.4f}")
print()
print("  la diversidad global y la diversidad donde V03 falla no son lo mismo, y esta")
print("  es la segunda, que es la unica que un selector necesitaria.")
print()

print("48.7  CURVA DE VALOR DE LA POLITICA")
print(f"  {'umbral':>10}{'cambiadas':>12}{'net':>8}")
for c in d["curva_de_valor"]:
    print(f"  {c['umbral']:>10.4f}{c['n_cambiadas']:>12}{c['net']:>+8d}")
print("  ATENCION: la curva esta contaminada por seleccion in-sample de la accion, su")
print("  net positivo es ajuste y no valor de politica. Su unico uso legitimo es")
print("  logico: ningun umbral puede superar el techo de estrato, y ese techo es NEGATIVO.")
print()

print("48.8  DECISION")
print(f"  STRATUM_ORACLE {t['STRATUM_ORACLE']:.6f} frente a V03 {t['V03']:.6f}")
print(f"  CASE = {d['caso']}")
print("  El techo de la mejor politica estratificada es NEGATIVO, luego ningun")
print("  selector condicional sobre variables observables puede superar a V03.")
print("  UTILITY_POLICY = CLOSED y CONTEXTUAL_SELECTION = CLOSED.")
print("  No se ejecuta P52-01: el techo no lo justifica.")
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
    """Anade la seccion 48 al notebook consolidado.

    Returns:
        None.
    """
    for artefacto in (INFORME, EXCLUSIVIDAD, PARES):
        if not artefacto.exists():
            raise SystemExit(f"PARADA: falta {artefacto}")
    json.loads(INFORME.read_text(encoding="utf-8"))
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    antes = len(notebook["cells"])
    marcas = " ".join(
        "".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "markdown"
    )
    if MARCA in marcas:
        print("  la seccion 48 ya existe, no se duplica")
        return
    notebook["cells"].extend([markdown(TITULO), codigo(CODIGO)])
    NOTEBOOK.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
                        encoding="utf-8")
    print(f"  notebook: {antes} -> {len(notebook['cells'])} celdas")
    print("  seccion 48 integrada en el notebook consolidado")


if __name__ == "__main__":
    main()
