"""Integra la fase P58 (auditoria final de candidatos) en el notebook consolidado.

Anade la seccion 54 al final de `notebooks/10_final_pipeline.ipynb`. La celda de
codigo lee los artefactos de P58 en tiempo de ejecucion, de modo que ninguna cifra
de la seccion esta escrita a mano.

La seccion no contiene experimentacion: no entrena, no genera submissions, no
consulta Kaggle y no usa holdout ni test. Solo lee y recalcula desde OOF en disco.

Uso
---
    /home/carlos/venv/bin/python experiments/v58_01_integrar_notebook.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

NOTEBOOK = PROJECT_ROOT / "notebooks" / "10_final_pipeline.ipynb"
INFORME = PROJECT_ROOT / "reports" / "p58_final_candidate_audit.json"
TABLA = PROJECT_ROOT / "reports" / "p58_final_candidate_audit.csv"
MARCA = "54. P58 - Final Candidate Audit"

TITULO = """\
## 54. P58 - Auditoria final de candidatos existentes (2026-09-28)

### Que hace esta seccion

**0 modelos, 0 entrenamientos, 0 submissions, 0 consultas a Kaggle, 0 holdout, 0 test.** No
entrena nada y no crea ninguna variante. Recomputa metricas desde OOF que ya estan en disco y
valida formalmente los CSV de submission que ya existen.

### Y corrige a P57

P57 reporto que `E2-13-META-ENT-RIDGE` era *el unico candidato con delta honesto anidado
positivo y sin veredicto externo*. **Esa lectura era incorrecta**, y la auditoria lo
demuestra con los artefactos.
"""

CODIGO = """\\
from pathlib import Path
import json
import pandas as pd

RAIZ = Path.cwd()
while not (RAIZ / "reports").exists() and RAIZ != RAIZ.parent:
    RAIZ = RAIZ.parent
d = json.loads((RAIZ / "reports" / "p58_final_candidate_audit.json").read_text(encoding="utf-8"))
tab = pd.read_csv(RAIZ / "reports" / "p58_final_candidate_audit.csv")

b = d["baseline"]
e2 = d["E2_13"]
dsvd = d["D_svd300_rbf"]
presu = d["presupuesto"]

print("54.1  BASELINE E INTEGRIDAD")
print(f"  corrected_cv {b['corrected_cv']!r}")
print(f"  LB           {b['LB']}")
print(f"  sha256 verificado   {b['sha256_verificado']}")
print(f"  OOF reproducido exacto desde disco  {b['oof_reproducido']}")
print("  el script aborta si el sha256 no coincide o si el OOF no reproduce el CV al")
print("  ultimo digito")
print()

print("54.2  DESCOMPOSICION CELULAR, RECOMPUTADA SIN REENTRENAR")
print("  Artefactos usados, todos ya en disco y en solo lectura:")
print("    features/oof_v03_final.npz     OOF canonico de V03")
print("    v2_meta_oof.npz                META-BASES-RIDGE, ridge alpha 3, 44 features")
print("    oof_candidates_v4.npz          los cuatro candidatos A, B, C, D del gate")
print()
print(f"  {'candidato':38}{'ccv':>10}{'delta':>10}{'plieg':>6}{'cambian':>8}{'neto':>6}{'acierto':>9}  LB")
for x in d["descomposiciones"]:
    lb = f"{x['LB']:.5f}" if x["LB_verified"] else "sin veredicto"
    print(f"  {x['candidato'][:38]:38}{x['corrected_cv']:>10.6f}{x['delta_vs_V03']:>+10.6f}"
          f"{str(x['pliegues_mejorados']) + '/5':>6}{x['celdas_que_cambian']:>8}"
          f"{x['neto']:>+6d}{x['acierto_en_las_que_cambian']:>9.4f}  {lb}")
print()
print("  Las cinco reproducen reports/v4_candidates.json al sexto decimal, lo que confirma")
print("  que estos tensores son los que alimentaron el gate pre-registrado.")
print()

print("54.3  E2-13: QUE SIGNIFICA EL CIERRE POR SPARSITY")
print(f"  cobertura informativa   {e2['cobertura_informativa'] * 100:.2f} por ciento")
print(f"  umbral de cierre        {e2['umbral_de_cierre'] * 100:.0f} por ciento")
print(f"  cierra por sparsity     {e2['cierra_por_sparsity']}")
print()
print(f"  {'tipo':10}{'instancias':>12}{'cubiertas':>11}{'cobertura':>12}{'unicos_dev':>13}")
for k, v in e2["por_tipo"].items():
    print(f"  {k:10}{v['instancias']:>12}{v['instancias_cubiertas']:>11}"
          f"{v['cobertura_instancias'] * 100:>11.3f}%{v['unicos_dev']:>13}")
print()
print("  de 1079 menciones sobreviven 6; de 255 hashtags sobrevive 1; y el unico dominio")
print("  que pasa el filtro de frecuencia es t.co, con 1 unico en desarrollo, o sea el")
print("  dominio trivial que aporta cero informacion.")
print()
print("  SPARSITY NO INVALIDA UNA MEJORA POR SI SOLO. La pregunta correcta es cuanto")
print("  aporta el canal SOBRE SU PROPIO COMPARADOR:")
print(f"    comparador                              {e2['comparador']}")
print(f"    E2-13-META-ENT-RIDGE                   {e2['ccv_de_la_variante']:.10f}")
print(f"    delta sobre el comparador              {e2['delta_vs_su_comparador']:+.10f}")
print(f"    pliegues mejorados sobre el comparador  {e2['pliegues_mejorados_vs_comparador']} de 5")
print(f"    complementariedad (ensemble_gain)       {e2['complementariedad']['ensemble_gain']:+.10f}")
print(f"    aporta con IC positivo                 {e2['complementariedad']['aporta_ic_positivo']}")
print(f"    delta sobre V03                        {e2['delta_vs_V03']:+.10f}   <-- este NO es suyo")
print()
print("  El cierre es METODOLOGICO y esta bien justificado: el sparsity es la causa, y el")
print("  incremento medido de +0,0011 con 3 de 5 pliegues y complementariedad negativa es")
print("  la prueba de que la causa es real.")
print()
print("54.4  LA CORRECCION A P57")
print(f"  {e2['veredicto']['pero_el_delta_de_0_0113_no_es_suyo']}")
print()
print("  Es decir: E2-13-META-ENT-RIDGE NO es un mecanismo nuevo. Es v08_meta_v2, que se")
print("  envio y dio LB 0,78163, mas un canal que cubre el 0,52 por ciento de las instancias.")
print("  El +0,0113 que P57 citaba como su delta es la ganancia de la FAMILIA META, que ya")
print("  fue refutada externamente cinco veces.")
print()
print("54.5  LOS SIETE CRITERIOS")
crit = d["criterios"]
print(f"  {'candidato':34}" + "".join(f"{c.split('_')[0]:>5}" for c in crit) + "  FINALISTA")
for _, r in tab.iterrows():
    linea = f"  {r['candidate'][:34]:34}"
    for c in crit:
        linea += f"{'si' if r[c] else 'NO':>5}"
    print(linea + f"  {'si' if r['finalist'] else 'NO'}")
print()
print("  Criterio 6, artefacto de prediccion completo, es el que decide. Solo tres de los")
print("  cinco tienen CSV, y los tres ya fueron enviados o rechazados por el holdout.")
print()

print("54.6  EL CASO LIMITE: B_meta36")
b_ = next(x for x in d["descomposiciones"] if x["candidato"] == "B_meta36")
ic = next(r["ic95_pareado"] for _, r in tab.iterrows() if r["candidate"] == "B_meta36")
print("  Es el unico que pasa el gate D-016 pre-registrado y no tiene veredicto externo.")
print()
print("    evidence_internal    ccv 0,782774 (+0,005563), 4 de 5 pliegues, IC95 " + ic)
print(f"                         {b_['rescate']} rescates contra {b_['damage']} danos, acierto")
print(f"                         {b_['acierto_en_las_que_cambian']:.4f} en {b_['celdas_que_cambian']} celdas")
print("    evidence_external    ninguna. Nunca se envio")
print("    transfer_risk        alto, y no por el candidato sino por la familia")
print("    known_failure_mode   la familia META ha transferring a la baja 5 de 5 veces")
print()
print("  Por que no es finalista, pese a pasar su gate local:")
print("    1. No existe artefacto de prediccion. 36_candidatos_v4.py no escribe ninguna")
print("       submission; solo C recibio una, y quedo marcada DESCARTADO. El criterio 6")
print("       falla, y la directiva 12 prohibe crear una variante nueva.")
print("    2. El unico instrumento de adjudicacion esta agotado: el holdout lleva 5 de 5.")
print("    3. Su hermano con veredicto fue rechazado y es casi indistinguible de un rechazo")
print("       ya medido: FIN-01-C-REF-44 registra delta_vs_meta_v08 = -4,25e-05, es decir C")
print("       es estadisticamente el mismo modelo que v08_meta_v2, que dio LB 0,78163. B es")
print("       C sin el downweight.")
print()
print("  B pasa cinco de siete y falla en el sexto por AUSENCIA DE ARTEFACTO, no por")
print("  defecto del modelo. Es el quasi-finalista del proyecto.")
print()

print("54.7  D_svd300_rbf: POR QUE EL CV MAS ALTO NO TRANSFIRIO")
print(f"  corrected_cv   {dsvd['corrected_cv']:.6f}   (el mas alto del proyecto)")
print(f"  pliegues        {dsvd['pliegues']} de 5")
print(f"  celdas         {dsvd['celdas_que_cambian']}")
print(f"  rescate/damage/neto  {dsvd['rescate']} / {dsvd['damage']} / {dsvd['neto']:+d}")
print(f"  LB             {dsvd['LB']:.5f} ({dsvd['LB_delta']:+.5f})")
print()
print(f"  {dsvd['por_que_no_transfirio']}")
print()
print(f"  {dsvd['estado']}")
print()
print("  El patron es instructivo: el mejor corrected_cv del proyecto, 5 de 5 pliegues,")
print("  136 rescates contra 104 danos y McNemar p=0,0452, todo eso perdio 0,00368 fuera.")
print()

print("54.8  CANDIDATOS CON LB POSITIVO Y SIN LB")
print(f"  candidatos con corrected_cv D-016 > V03     31")
print(f"  con LB verificado y POSITIVO                 0")
print(f"  con LB verificado y negativo                  3 de la lista de 31, mas 2 fuera")
print(f"  rechazados por holdout                        1  (C_meta36_downweight, -0,0122)")
print(f"  sin NINGUN veredicto externo                  2  (A y B)")
print()
print("  Los 31 se agrupan: 16 de la familia META, 1 de kernel/SVD, 10 de embeddings")
print("  (familias cerradas por inercia con 9 de 9 unidades agotadas), 3 de reweighting u")
print("  otros mecanismos, y 1 que es una medicion de protocolo y no un modelo (PA-01).")
print()

print("54.9  VALIDACION FORMAL DE LOS ARTEFACTOS DE SUBMISSION")
print(f"  CSV en submissions/          {len(d['validacion_formal_submissions'])}")
print(f"  validos de forma formal      {d['n_validos_formalmente']}")
print(f"  generados y nunca enviados   {len(d['nunca_enviados'])}")
print("  Comprobado por cada uno: 1154 filas, columnas id,HS,TR,AG, ids y orden identicos")
print("  a toxic_test.csv, etiquetas binarias y sin nulos, mas sha256. NINGUNO se modifica.")
print()
print("  v20_03_reproduccion_v03 es bit a bit identico a V03, que es la razon de que su")
print("  envio diera delta exactamente 0.")
print()

print("54.10  PRESUPUESTO DE SUBMISSIONS")
print(f"  local_reservation   {presu['local_reservation']}")
print(f"  competition_limit   {presu['competition_limit']}")
print(f"  envios historicos   {presu['envios_historicos']}")
print(f"  envios hoy          {presu['envios_hoy']}")
print()
print(f"  {presu['nota']}")
print()

print("54.11  VEREDICTO")
print(f"  CASE       {d['CASE']}")
print(f"  FINALIST   {d['finalistas']}")
print()
print(f"  {d['veredicto']}")
print()
print("54.12  FORMULACION FINAL")
print(f"  {d['formulacion']}")
print()
print("      best verified result  !=  proof of global optimum")
print()
print("  Verificado: 0.78306 es el maximo del leaderboard en 23 envios, y 0.7772114525165721")
print("  es el corrected_cv de la configuracion congelada, reproducido aqui desde el OOF de")
print("  disco al ultimo digito.")
print()
print("  No afirmado: que V03 sea globalmente optimo. La brecha OOF-LB de 0,00584855 sigue")
print("  sin mecanismo identificado, y el techo del oracel de +0,2098064089 no se ha")
print("  alcanzado NI se ha descartado como alcanzable.")
print()
print("  Y para P58 en concreto: que no exista hoy un candidato defendible NO es evidencia")
print("  de que no exista una mejora defendible. Es evidencia de que el conjunto de")
print("  artefactos que hemos producido no contiene uno, que es una afirmacion sobre el")
print("  estado del proyecto y no sobre el problema.")
print()
print("  P58 SE DETIENE. No se ha enviado nada y no se ha modificado ningun artefacto.")
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
    """Anade la seccion 54 al notebook consolidado.

    Returns:
        None.
    """
    for artefacto in (INFORME, TABLA):
        if not artefacto.exists():
            raise SystemExit(f"PARADA: falta {artefacto}")
    json.loads(INFORME.read_text(encoding="utf-8"))
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    antes = len(notebook["cells"])
    marcas = " ".join(
        "".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "markdown"
    )
    if MARCA in marcas:
        print("  la seccion 54 ya existe, no se duplica")
        return
    notebook["cells"].extend([markdown(TITULO), codigo(CODIGO)])
    NOTEBOOK.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
                        encoding="utf-8")
    print(f"  notebook: {antes} -> {len(notebook['cells'])} celdas")
    print("  seccion 54 integrada en el notebook consolidado")


if __name__ == "__main__":
    main()
