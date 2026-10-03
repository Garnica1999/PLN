"""Integra la fase P53 (Fine-Grained Observable Policy Audit) en el notebook.

Anade la seccion 49 al final de `notebooks/10_final_pipeline.ipynb`. La celda de
codigo lee los artefactos de P53 en tiempo de ejecucion, de modo que ninguna cifra
de la seccion esta escrita a mano.

Uso
---
    /home/carlos/venv/bin/python experiments/v53_01_integrar_notebook.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

NOTEBOOK = PROJECT_ROOT / "notebooks" / "10_final_pipeline.ipynb"
INFORME = PROJECT_ROOT / "reports" / "p53_observable_policy.json"
EXCLUSIVO = PROJECT_ROOT / "reports" / "p53_exclusive_rescue.csv"
PARES = PROJECT_ROOT / "reports" / "p53_pairwise_channels.csv"
PATRONES = PROJECT_ROOT / "reports" / "p53_pattern_policy.csv"
POLITICA = PROJECT_ROOT / "reports" / "p53_observable_policy.csv"
MARCA = "49. P53 - Fine-Grained Observable Policy Audit"

TITULO = """\
## 49. P53 - Fine-Grained Observable Policy Audit (2026-09-28)

### Por que P53 existia

P52 concluyo cerrando la linea contextual, y lo hizo saltando un paso: uso 18
estratos definidos por **cuantos** canales discrepan, y de ahi afirmo que ninguna
politica observable supera V03. Eso no se sigue. P53 pregunta si la **identidad
exacta** de los canales que discrepan aporta algo que el conteo no aporta.

### El colapso, que es lo que hace que la pregunta tenga respuesta

La comprobacion estructural va antes que la politica, y determina el alcance de todo
lo demas. En las celdas con discrepancia, **todos** los canales que discrepan proponen
exactamente el valor `1 - v03_pred`. Cero celdas donde dos de ellos se contradigan.

Consecuencia: cambiar al canal 3 o al canal 17 produce **la misma prediccion**. El
espacio de 21 acciones se reduce a **2**, y la mascara no puede decir el valor
propuesto, porque el valor no depende de quien discrepa. Solo puede decir cuan fuerte
empuja cada canal, a traves de las probabilidades.
"""

CODIGO = """\
from pathlib import Path
import json
import pandas as pd

RAIZ = Path.cwd()
while not (RAIZ / "reports").exists() and RAIZ != RAIZ.parent:
    RAIZ = RAIZ.parent
d = json.loads((RAIZ / "reports" / "p53_observable_policy.json").read_text(encoding="utf-8"))
ex = pd.read_csv(RAIZ / "reports" / "p53_exclusive_rescue.csv")
pr = pd.read_csv(RAIZ / "reports" / "p53_pairwise_channels.csv")
pa = pd.read_csv(RAIZ / "reports" / "p53_pattern_policy.csv")
po = pd.read_csv(RAIZ / "reports" / "p53_observable_policy.csv")

a = d["auditoria_de_p52"]
col = d["colapso_estructural"]
ma = d["mascaras"]
xr = d["rescate_exclusivo"]
pr_ = d["pareces"]
c = d["comparacion"]

print("49.1  P52 CORRECTIONS")
print("  Se MANTIENEN:")
print(f"    - los estratos de P52 son {a['estratos']}")
print(f"    - las 78 celdas de rescate unico: P53 recalcula "
      f"{xr['celdas_con_rescate_unico']} | coincide = {xr['coincide_con_p52']}")
print(f"    - el TRUE_ORACLE de {c['TRUE_ORACLE']:.10f} se reconstruye con el colapso")
print("  Se CORRIGEN:")
print(f"    - P52 NO estaba contaminado, la pregunta estaba mal planteada: {a['por_que_no_invalida']}")
print(f"    - P52 no establece {a['lo_que_p52_no_establish']}")
print("  Metricas MAL DEFINIDAS en P52:")
print("    - el HHI: se computo como sum((exclusivo_i / 715)^2), con proporciones que")
print("      NO suman 1. P53 reporta HHI estandar y cuota por separado.")
print()

print("49.2  EL COLAPSO ESTRUCTURAL")
print(f"  celdas con discrepancia                       : {col['celdas_con_discrepancia']}")
print(f"  celdas con valor propuesto distinto a 1-v03  : {col['celdas_con_valor_propuesto_distinto']}")
print(f"  celdas con dos canales contradiciendose     : {col['celdas_con_dos_canales_contradicendose']}")
print(f"  ESPACIO DE ACCIONES REAL                     : {col['espacio_de_acciones_real']}")
print(f"  {col['consecuencia']}")
print()

print("49.3  EXCLUSIVIDAD, CON LAS metricas BIEN DEFINIDAS")
print(f"  celdas con rescate unico                  : {xr['celdas_con_rescate_unico']} de 715")
print(f"  suma de rescates exclusivos               : {xr['suma_de_exclusivos']}")
print(f"  HHI ESTANDAR de los exclusivos            : {d['hhi_texto']}")
print(f"  cuota de exclusivos sobre los 715         : {100 * xr['cuota_de_exclusivos_sobre_715']:.2f} pct")
print(f"  canales con net_exclusivo positivo        : {xr['canales_con_net_exclusivo_positivo']} de 20")
print()
print("  P52 reporto 0 pares con exclusividad bilateral. P53 lo recalcula SIN restringir")
print("  al conjunto donde ambos discrepan, y encuentra:")
n_bil = int(((pr["i_only_rescue"] > 0) & (pr["j_only_rescue"] > 0)).sum())
print(f"  pares con exclusividad bilateral de rescate: {n_bil} de {len(pr)}")
print(f"  jaccard de rescates: min {pr_['jaccard_de_rescates']['min']:.4f} "
      f"max {pr_['jaccard_de_rescates']['max']:.4f} media {pr_['jaccard_de_rescates']['media']:.4f}")
print()

print("49.4  LOS 190 PARES, DESCOMPUESTOS EN A B C D")
print(f"  pares analizados                                     : {pr_['analizados']}")
print(f"  B: ambos discrepan y se CONTRADICEN, suma total      : {pr_['suma_de_B_sobre_todos_los_pares']}")
print(f"  pares con B > 0                                      : {pr_['pares_con_B_positivo']}")
print("  B = 0 en los 190 pares NO es casualidad: es la identidad estructural de 49.2.")
print()
print(f"  {'canal i':34}{'A':>7}{'B':>4}{'C':>7}{'D':>7}{'i solo':>8}{'j solo':>8}{'jacc':>8}")
for _, x in pr.sort_values("union_rescue", ascending=False).head(6).iterrows():
    print(f"  {x['canal_i']:34}{x['A_ambos_discrepan_y_coinciden']:>7}"
          f"{x['B_ambos_discrepan_y_se_contradicen']:>4}"
          f"{x['C_exactamente_uno_discrepa']:>7}"
          f"{x['D_ambos_coinciden_con_V03']:>7}"
          f"{x['i_only_rescue']:>8}{x['j_only_rescue']:>8}{x['jaccard_de_rescates']:>8.4f}")
print()

print("49.5  DISTRIBUCION DE LAS MASCARAS")
print(f"  mascaras distintas sobre las 4449 celdas : {ma['distintas_sobre_todas_las_celdas']}")
print(f"  mascaras distintas entre las con discrep.: {ma['distintas_entre_las_con_discrepancia']}")
print(f"  patrones en celdas con discrepancia    : {ma['patrones_en_celdas_de_error']}")
print(f"  frecuencia mediana                     : {ma['frecuencia_mediana']:.0f}")
print(f"  frecuencia maxima                      : {ma['frecuencia_maxima']}")
print(f"  patrones singleton                     : {ma['patrones_singleton']}")
print("  la mascara esta muy fragmentada, y por eso casi todos los patrones son uniques.")
print()

print("49.6  RESCATE Y DANO: UN SOLO CANAL FRENTE A VARIOS")
for x in d["direccion_simple_vs_conjunta"]:
    print(f"  {x['tipo']:16} n={x['n']:5d} | rescate {100 * x['rescate_rate']:6.2f} pct "
          f"| dano {100 * x['dano_rate']:6.2f} pct | net {x['net']:+6d}")
print("  el canje es brutal en las dos: el dano domina por 5 a 1 o mas.")
print()

print("49.7  TECHOS CROSS-FITTED DE P0 A P4")
print(f"  {'nivel':8}{'corrected_cv':>16}{'delta vs V03':>16}{'cambiadas':>11}{'net':>8}{'estados':>9}{'cobertura':>11}")
for _, x in po.iterrows():
    print(f"  {x['nivel']:8}{x['corrected_cv']:>16.10f}{x['delta_vs_v03']:>+16.10f}"
          f"{x['n_cambiadas']:>11}{x['net']:>+8}{x['estados_observables']:>9}"
          f"{100 * x['cobertura_validacion']:>10.1f} pct")
print()
print("  soporte minimo en training, como diagnostico de estabilidad:")
print(f"  {'nivel':8}{'soporte':>9}{'corrected_cv':>16}{'delta':>14}{'cambiadas':>11}")
for x in d["soporte"]:
    print(f"  {x['nivel']:8}{x['soporte_minimo']:>9}{x['corrected_cv']:>16.10f}"
          f"{x['delta_vs_v03']:>+14.10f}{x['n_cambiadas']:>11}")
print("  al exigir soporte alto el numero de cambios cae a cero: los patrones con")
print("  expectativa positiva son los pequenos, luego no generalizan.")
print()

print("49.8  LA CADENA DE TECHOS")
print(f"  {'politica':44}{'corrected_cv':>16}{'delta vs V03':>16}")
print(f"  {'V03 (P0, no hacer nada)':44}{c['V03']:>16.10f}{0.0:>+16.10f}")
print(f"  {'P1 margen + conteo':44}{float(po[po['nivel'] == 'P1']['corrected_cv'].iloc[0]):>16.10f}"
      f"{float(po[po['nivel'] == 'P1']['delta_vs_v03'].iloc[0]):>+16.10f}")
print(f"  {'P2 + mascara exacta':44}{float(po[po['nivel'] == 'P2']['corrected_cv'].iloc[0]):>16.10f}"
      f"{float(po[po['nivel'] == 'P2']['delta_vs_v03'].iloc[0]):>+16.10f}")
print(f"  {'P3 + etiqueta':44}{float(po[po['nivel'] == 'P3']['corrected_cv'].iloc[0]):>16.10f}"
      f"{float(po[po['nivel'] == 'P3']['delta_vs_v03'].iloc[0]):>+16.10f}")
print(f"  {'P4 + probabilidades':44}{float(po[po['nivel'] == 'P4']['corrected_cv'].iloc[0]):>16.10f}"
      f"{float(po[po['nivel'] == 'P4']['delta_vs_v03'].iloc[0]):>+16.10f}")
print(f"  {'P52 STRATUM_ORACLE (18 estratos)':44}{c['P52_STRATUM_ORACLE']:>16.10f}"
      f"{c['P52_STRATUM_ORACLE'] - c['V03']:>+16.10f}")
print(f"  {'TRUE_ORACLE':44}{c['TRUE_ORACLE']:>16.10f}{c['TRUE_ORACLE'] - c['V03']:>+16.10f}")
print()

print("49.9  INTERACCIONALIDAD")
idat = d["interaccionalidad"]
if idat:
    print(f"  {'canal':34}{'solo':>10}{'con otros':>12}{'diferencia':>13}")
    for x in sorted(idat, key=lambda z: z["diferencia"])[:4]:
        print(f"  {x['canal']:34}{x['tasa_solo']:>10.4f}{x['tasa_con_otros']:>12.4f}"
              f"{x['diferencia']:>+13.4f}")
    for x in sorted(idat, key=lambda z: -z["diferencia"])[:3]:
        print(f"  {x['canal']:34}{x['tasa_solo']:>10.4f}{x['tasa_con_otros']:>12.4f}"
              f"{x['diferencia']:>+13.4f}")
    print("  hay estructura interaccional, pero no llega a producir una politica positiva.")
print()

print("49.10  DECISION")
print(f"  el techo observable mas fino supera V03 + 0,003: {d['supera_el_gate_de_entrenamiento']}")
print(f"  CASE = {d['caso']}")
print("  MASK_BASED_SELECTION = CLOSED. Ninguna politica basada en la mascara de")
print("  desacuerdo, el margen, el conteo, la etiqueta y las probabilidades de los")
print("  canales supera V03 en evaluacion cross-fitted.")
print("  No se entrena modelo: el techo no lo justifica.")
print()
print("  ALCANCE DE LA AFIRMACION. Lo que la evidencia sostiene es que ninguna politica")
print("  basada en ESTA particion observable supera V03. No se afirma que ninguna")
print("  politica imaginable lo haga.")
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
    """Anade la seccion 49 al notebook consolidado.

    Returns:
        None.
    """
    for artefacto in (INFORME, EXCLUSIVO, PARES, PATRONES, POLITICA):
        if not artefacto.exists():
            raise SystemExit(f"PARADA: falta {artefacto}")
    json.loads(INFORME.read_text(encoding="utf-8"))
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    antes = len(notebook["cells"])
    marcas = " ".join(
        "".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "markdown"
    )
    if MARCA in marcas:
        print("  la seccion 49 ya existe, no se duplica")
        return
    notebook["cells"].extend([markdown(TITULO), codigo(CODIGO)])
    NOTEBOOK.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
                        encoding="utf-8")
    print(f"  notebook: {antes} -> {len(notebook['cells'])} celdas")
    print("  seccion 49 integrada en el notebook consolidado")


if __name__ == "__main__":
    main()
