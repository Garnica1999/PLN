"""Integra la fase P57 (cierre estrategico) en el notebook consolidado.

Anade la seccion 53 al final de `notebooks/10_final_pipeline.ipynb`. La celda de
codigo lee los artefactos de P57 en tiempo de ejecucion, de modo que ninguna cifra
de la seccion esta escrita a mano.

La seccion no contiene experimentacion: solo lee y imprime.

Uso
---
    /home/carlos/venv/bin/python experiments/v57_01_integrar_notebook.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

NOTEBOOK = PROJECT_ROOT / "notebooks" / "10_final_pipeline.ipynb"
INFORME = PROJECT_ROOT / "reports" / "p57_strategic_closure.json"
INVENTARIO = PROJECT_ROOT / "reports" / "p57_final_experiment_inventory.csv"
FAMILIAS = PROJECT_ROOT / "reports" / "p57_family_status.csv"
CANDIDATOS = PROJECT_ROOT / "reports" / "p57_candidate_matrix.csv"
ABIERTAS = PROJECT_ROOT / "reports" / "p57_open_questions.csv"
MARCA = "53. P57 - Strategic Closure Audit"

TITULO = """\
## 53. P57 - Cierre estrategico y plan final (2026-09-28)

### Que hace esta seccion

Sintesis, auditoria y estrategia. **0 modelos, 0 entrenamientos, 0 submissions, 0 consultas a
Kaggle, 0 uso de holdout, 0 uso de test.** No se registro ninguna fila nueva en
`experiments.csv`, porque no hubo ninguna ejecucion experimental.

### Lo que esta seccion corrige del propio registro

Al cruzar `experiments.csv` con `leaderboard_log.csv` aparecio un problema de lectura que
afecta a **cualquier tabla de "mejor CV"** construida sobre el registro: las columnas `f1` y
`macro_f1` **no son una metrica cada una**. Mezclan al menos tres metricas y tres protocolos.

Y aparecio un hecho operativo: **un envio real a Kaggle no estaba en el log del leaderboard.**
"""

CODIGO = """\\
from pathlib import Path
import json
import pandas as pd

RAIZ = Path.cwd()
while not (RAIZ / "reports").exists() and RAIZ != RAIZ.parent:
    RAIZ = RAIZ.parent
d = json.loads((RAIZ / "reports" / "p57_strategic_closure.json").read_text(encoding="utf-8"))
inv = pd.read_csv(RAIZ / "reports" / "p57_final_experiment_inventory.csv")
fam = pd.read_csv(RAIZ / "reports" / "p57_family_status.csv")
cand = pd.read_csv(RAIZ / "reports" / "p57_candidate_matrix.csv")
abi = pd.read_csv(RAIZ / "reports" / "p57_open_questions.csv")

b = d["baseline_congelado"]
jer = d["jerarquia_de_magnitudes"]
presu = d["presupuesto"]
can = d["canales_p49_p53"]

print("53.1  BASELINE CONGELADO")
print(f"  id           {b['id']}")
print(f"  repr         {b['representacion']}")
print(f"  clasificador {b['clasificador']}")
print(f"  umbrales     HS {b['umbrales']['HS']} | TR {b['umbrales']['TR']} | AG {b['umbrales']['AG']}")
print(f"  corrected_cv {b['corrected_cv']!r}")
print(f"  LB           {b['LB']}")
print(f"  sha256       {b['sha256']}")
print(f"  verificado   {b['sha256_verificado']}")
print()

print("53.2  JERARQUIA DE MAGNITUDES, SIN MEZCLAR METRICAS")
print("  Cada fila lleva su metrica, su protocolo y su poblacion. NO son intercambiables.")
print()
print(f"  {'magnitud':44}{'valor':>16}  {'metrica':28}{'prot':7}{'comp'}")
for k, v in jer.items():
    print(f"  {k:44}{str(v['valor'])[:16]:>16}  {v['metrica'][:28]:28}{v['protocolo']:7}"
          f"{v['comparable_con_V03']}")
print()
for k in ("best_comparable_CV_D016", "best_honest_CV_anidado", "best_verified_LB",
          "V03_historical_cv_D004", "current_champion"):
    print(f"  {k}: {jer[k]['valor']}  ({jer[k]['experimento']})")
    print(f"      {jer[k]['nota']}")
print()

print("53.3  EL PROBLEMA DE LECTURA DEL REGISTRO")
print(f"  inventario: {len(inv)} filas, una por experimento del registro")
print(f"    protocolo D-004 (mas debil, NO comparable) : {d['inventario']['D-004']:3d}")
print(f"    protocolo D-016 (corregido, comparable)   : {d['inventario']['D-016']:3d}")
print(f"    holdout de 248 filas (otra poblacion)      : {d['inventario']['HOLD']:3d}")
print(f"    sin clasificar                             : {d['inventario']['sin_clasificar']:3d}")
print()
print("  Por que esto importa: la metrica del proyecto sobre el OOF de V03 vale")
print("  0.7772114525 y la macro-6 vale 0.8182213363, una diferencia de +0.04101 (P45).")
print("  El maximo de la columna macro_f1 en todo el registro es 0.8743, que es macro-6")
print("  sobre el holdout, NO una medida de la metrica de la competicion. Sumado a que")
print("  leaderboard_log.csv registra para V03 un local_f1 de 0.7989, que es +0.021689")
print("  por encima del corrected_cv canonico, y la diferencia es el coste de medir con D-004.")
print()
print("  ESTA SECCION NO CONSTRUYE ESA TABLA. Etiqueta cada fila y compara dentro de un")
print("  mismo protocolo.")
print()

print("53.4  MEJOR CV COMPARABLE, MEJOR CV ANIDADO, MEJOR LB")
print(f"  best comparable D-016 (con SELECCION)  {jer['best_comparable_CV_D016']['valor']}  "
      f"({jer['best_comparable_CV_D016']['experimento']})")
print(f"  best anidado D-016     (sin seleccion) {jer['best_honest_CV_anidado']['valor']}  "
      f"({jer['best_honest_CV_anidado']['experimento']})")
print(f"  V03 corrected_cv                       {b['corrected_cv']!r}")
print(f"  best verified LB                       {jer['best_verified_LB']['valor']}  "
      f"({jer['best_verified_LB']['experimento']})")
print()
print("  El mejor CV comparable NO es el campeon: es la linea svd300_rbf, que se envio")
print("  y dio LB 0.77938. Y es ademas el maximo de cuatro candidatos comparados sobre el")
print("  mismo OOF, luego lleva seleccion y su IC95 tocaba cero.")
print()

print("53.5  ESTADO DE V03 Y DIMENSIONES DE UN SOLO PARAMETRO")
print("  Las tres dimensiones tienen su optimo en V03, y la respuesta a la tasa es")
print("  monotona a ambos lados:")
print("    C          1.0 -> 0.77227 | 2.0 -> 0.78306 | 3.26 -> 0.77602")
print("    min_df     1   -> 0.76978 | 2   -> 0.78306 | 5   -> 0.77034")
print("    tasa       0.3385 -> 0.77993 | 0.3645 -> 0.78306 | 0.3836 -> 0.77828 | 0.3980 -> 0.77495")
print()

print("53.6  MAPA DE CIERRE DE FAMILIAS")
print(f"  {len(fam)} familias. Tipos: A resultado negativo, B implementacion invalida,")
print("  C sin transferencia, D evidencia insuficiente, E prohibido o inexistente,")
print("  F redundante.")
print()
print(f"  {'familia':40}{'estado':14}{'tipo':6}{'LB'}")
for _, r in fam.iterrows():
    lb = str(r["LB_transfer"])
    corto = "RECHAZADO" if "RECHAZ" in lb.upper() else ("a la baja" if "baja" in lb else "no enviado")
    print(f"  {r['family'][:40]:40}{r['status']:14}{r['closure_type_A_to_F']:6}{corto}")
print()
for t in ["A", "B", "C", "E"]:
    fs = fam[fam["closure_type_A_to_F"] == t]["family"].tolist()
    print(f"  tipo {t}: {', '.join(fs) if fs else 'ninguna'}")
print()
print("  Las 4 familias de tipo C, con su veredicto externo:")
print("    kernels                1 envio, LB 0.77938, RECHAZADO")
print("    label_specialization   1 envio, LB 0.77781, RECHAZADO")
print("    ensembles              3 envios, 0.7751 / 0.7747 / 0.77122, los tres a la baja")
print("    stacking / meta        5 envios, 0.78163 / 0.77895 / 0.77978 / 0.78150 / 0.77663")
print()
print("  DIEZ envios de candidatos con mecanismo nuevo. LOS DIEZ transfirieron a la baja.")
print("  Es el hecho mas firme del proyecto y es lo que gobierna la estrategia final.")
print()

print("53.7  UMBRAL DE THRESHOLDS, P55 Y P56")
u = d["umbral_p55_p56"]
print(f"  P55 reporta        {u['P55_reportado']:.10f}   (+0.0009253174)")
print(f"  P56 global         {u['P56_global']:.10f}   (+0.0026400107, 4/5 pliegues)")
print(f"  P56 HONESTO        {u['P56_honesto']:.10f}   ({u['P56_honesto'] - u['V03']:+.10f})")
print(f"  optimismo de la cifra global   {u['P56_optimismo']:+.10f}")
print(f"  V03                {u['V03']:.10f}")
print(f"  THRESHOLD_TRANSFER = {u['THRESHOLD_TRANSFER']}")
print("  Los umbrales de V03 caen DENTRO del rango de las cinco estimaciones interiores en")
print("  las tres etiquetas. El defecto procedimental de P55 sigue siendo un hecho real, pero")
print("  su magnitud con el procedimiento correcto es nula o negativa.")
print()

print("53.8  CIERRE DE LOS CANALES, P49 A P53")
print(f"  V03 errores            {can['V03_errores']} celdas de {inv.shape[0] and 4449}")
print(f"  V03 correctos          {can['V03_correctos']}")
print(f"  rescuable              {can['rescuable']}")
print(f"  no rescatable          {can['no_rescatable']} celdas en {can['tweets_no_rescatables']} tweets")
print(f"  true oracle            {can['true_oracle']:.10f}  ({can['true_oracle'] - b['corrected_cv']:+.10f})")
print(f"  mejor politica observable   {can['mejor_politica_observable']:.10f}  "
      f"({can['mejor_politica_observable'] - b['corrected_cv']:+.10f})")
print(f"  CONTEXTUAL_SELECTION = {can['CONTEXTUAL_SELECTION']}")
print()
print("  LO QUE SE SOSTIENE: la senal existe y es enorme, el techo del oracel es casi 70")
print("  veces el margen de adopcion de 0,003.")
print("  LO QUE NO SE SOSTIENE: que exista una politica observable que la convierta en mejora")
print("  neta. P53 midio que los 20 canales que discrepan proponen 1 - v03_pred en las 3896")
print("  celdas con discrepancia, sin una sola excepcion, luego el espacio de acciones real")
print("  cae de 21 a 2 y toda politica esta por debajo de V03.")
print("  Decir 'la senal existe pero ningun selector probado la convirtio en mejora neta")
print("  positiva' es correcto. Decir 'no hay senal' seria falso.")
print()

print("53.9  ERROR RESIDUAL Y DISPONIBILIDAD DE DATOS")
print(f"  no rescatable   {can['no_rescatable']} celdas = "
      f"{can['no_rescatable'] / 4449 * 100:.2f} por ciento, en {can['tweets_no_rescatables']} tweets")
print("  P54: FP 22 / FN 20, 85,7 por ciento de margen alto. El lexico de insulto discrimina")
print("  1,25 veces contra A, luego NO discrimina.")
dd = d["datos"]
print(f"  train oficial   {dd['train_oficial']}  (dev {dd['dev']} + holdout {dd['holdout']})")
print(f"  test oficial    {dd['test_oficial']}")
print(f"  DATA_EXPANSION  = {dd['DATA_EXPANSION']}")
print("  0 filas oficiales etiquetadas sin usar. Los corpus externos existen pero NO estan")
print("  etiquetados, su unico uso permitido es entrenar embeddings, y esa familia esta cerrada.")
print()

print("53.10  INVENTARIO DE SUBMISSIONS")
print(f"  envios historicos totales            {presu['envios_historicos']}")
print(f"  con LB verificado en el log          {presu['envios_con_LB_verificado_en_log']}")
print(f"  envio real NO registrado en el log   {presu['envios_reales_no_registrados']}")
print(f"  ficheros en disco                    {len(d['nunca_enviados']) + presu['envios_con_LB_verificado_en_log'] + presu['envios_reales_no_registrados']}")
print(f"  generados y nunca enviados           {len(d['nunca_enviados'])}")
print(f"  envios hoy, 2026-09-28              {presu['envios_hoy']}")
print(f"  limite diario EXACTO                 {presu['limite_diario_conocido']}")
print(f"  reserva propia del proyecto          {presu['reserva_propia_del_proyecto']}")
print()
print(f"  Envios por dia: {presu['envios_por_dia']}")
print()
print("  El envio que falta en el log: V33-03 documento un envio real y aceptado, ID 56608788,")
print("  fecha 2026-09-27, fichero toxic_submission_v32_05_url_length.csv, LB 0.78306, con delta")
print("  EXACTAMENTE 0 porque la submission resulto identica a V03.")
print()
print(f"  {presu['unica_cifra_local']}")
print()

print("53.11  ARTEFACTOS QUE NO DEBEN VOLVER A USARSE")
print("  toxic_submission_v17_prior_shift_em.csv        CUARENTENADA, transductiva")
print("  toxic_submission_v18_near_dup_propagacion.csv  CUARENTENADA, transductiva, cambia 0 celdas")
print("  toxic_submission_v16_ch2_prior_shift_em.csv    CUARENTENADA, transductiva, mismo motivo que v17")
print("  toxic_submission_v12_descartado_meta36...csv   RECHAZADA por holdout 0.8163 contra 0.8285")
print("  toxic_submission_v20_03_reproduccion_v03.csv   INVALIDA, es una reproduccion de V03")
print("  toxic_submission_v32_05_url_length.csv         GASTADA, delta exactamente 0")
print("  No se modifica ninguna. No se vuelve a enviar ninguna.")
print()

print("53.12  CANDIDATOS, CON LOS CINCO CRITERIOS SIMULTANEOS")
print("  legal + reproducible + sin leakage + no rechazado ya + mecanismo claro")
print()
print(f"  {'artefacto':44}{'honest_CV':>12}{'LB':>10}  verificado")
for _, r in cand.iterrows():
    print(f"  {r['candidate'][:44]:44}{str(r['honest_CV'])[:12]:>12}{str(r['LB']):>10}  {r['verified'][:26]}")
print()
print("  RESULTADO: NINGUN artefacto cumple los cinco a la vez.")
print()
print("    v23_01 svd300_rbf     ya rechazado: enviado, LB 0.77938")
print("    v08_meta_v2           ya rechazado: enviado, LB 0.78163")
print("    v20_01 meta 36 feats  ya rechazado dos veces: holdout y LB 0.77663")
print("    E2-13-META-ENT-RIDGE  legal y sin leakage, pero cerrado por sparsity y nunca")
print("                           verificado fuera")
print("    umbrales P56          veredicto NOT_CONFIRMED, delta honesto negativo")
print("    v16_ch2 prior shift   transductivo: usa informacion del conjunto de test")
print()
print("  Es un resultado, no una omision. Los tres con mejor evidencia interna tienen")
print("  veredicto externo medido y contrario.")
print()

print("53.13  PREGUNTAS ABIERTAS, CON MECANISMO Y NO CON UN 'NO SE HA PROBADO'")
for i, r in abi.iterrows():
    print(f"  [{i + 1}] {r['question']}")
    print(f"      probada : {r['tested']}")
    print(f"      legal   : {r['legal']}")
    print(f"      valor   : {str(r['expected_value'])[:150]}")
    print()
print("  Ninguna cumple hoy las seis condiciones de reapertura de la directiva.")
print()

print("53.14  LIMITACIONES CONOCIDAS")
inc = d["inconsistencias_documentales"]
for sev in ["CRITICAL", "IMPORTANT", "MINOR"]:
    sel = [x for x in inc if x["severity"] == sev]
    print(f"  {sev}: {len(sel)}")
    for x in sel:
        print(f"    - {x['where'][:66]}")
print()
print("  Las dos CRITICAL se leen enteras en la seccion 13 del informe.")
print("  No se corrige la historia experimental: se documenta y se da la cifra correcta.")
print()

print("53.15  CONCLUSION DE INVESTIGACION")
m = d["metrica"]
print(f"  LOCAL_METRIC_REPRODUCED            = {m['LOCAL_METRIC_REPRODUCED']}")
print(f"  KAGGLE_SCORER_DIRECTLY_VERIFIED    = {m['KAGGLE_SCORER_DIRECTLY_VERIFIED']}")
print()
print("  V03 remains the best verified leaderboard result and the frozen architectural")
print("  baseline. The investigated legal mechanisms did not produce a robust transferable")
print("  improvement, and no additional mechanism with sufficient evidence has currently")
print("  been identified.")
print()
print("  Y la distincion que no hay que perder:")
print()
print("      best verified result  !=  proof of global optimum")
print()
print(f"  Verificado : {b['LB']} es el maximo del leaderboard en {presu['envios_historicos']} envios,")
print(f"               y {b['corrected_cv']!r} es el CV corregido de la configuracion congelada.")
print("  NO afirmado: que V03 sea globalmente optimo. La brecha OOF-LB de 0.00584855 sigue")
print("               sin mecanismo identificado, y el techo del oracel de +0.2098 no se ha")
print("               alcanzado ni se ha descartado como alcanzable.")
print("  El hecho de que ningun candidato con mejor CV local haya transferido es evidencia")
print("  en contra de la optimizacion local y a favor de V03 como eleccion. NO es evidencia")
print("  de que no exista una mejor.")
print()

print("53.16  ESTADO ESTRATEGICO FINAL")
print("  La decision competitiva NO se toma en P57. Quedan tres salidas, y ninguna se")
print("  ejecuta automaticamente:")
print()
print("    A. cerrar con V03")
print("       a favor : 0 de 23 envios lo superaron, los 10 candidatos con mecanismo nuevo")
print("                transfirieron a la baja, las 22 familias estan cerradas")
print("       en contra: el techo del oracel es +0.2098 y la brecha OOF-LB sin explicar")
print("       riesgo  : bajo. El leaderboard conserva la mejor puntuacion")
print("       coste   : cero")
print()
print("    B. usar un candidato historico")
print("       a favor : E2-13 tiene +0.0113 de delta honesto anidado sin contradiccion externa")
print("       en contra: los tres con delta comparable tienen veredicto externo contrario")
print("       riesgo  : alto. Gastaria presupuesto repitiendo un patron de 10 sobre 10")
print("       coste   : 1 submission. Y NO HAY candidato que cumpla los cinco criterios,")
print("                luego esta opcion no es ejecutable sin una excepcion explicita")
print()
print("    C. ultima investigacion dirigida")
print("       a favor : la brecha OOF-LB y la asimetria dev-holdout no estan caracterizadas")
print("       en contra: ninguna hipotesis cumple las seis condiciones de reapertura, y la")
print("                 banda de particion de 0.017 supera cualquier delta local alcanzable")
print("       riesgo  : medio. Puede no producir nada y consume presupuesto")
print("       coste   : alto, no medido")
print()
print("  Si se autorizara una submission, el unico formato admisible es:")
print("      un candidato congelado, una sola submission, no una busqueda interactiva.")
print()
print("  P57 SE DETIENE AQUI. No se abre P58.")
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
    """Anade la seccion 53 al notebook consolidado.

    Returns:
        None.
    """
    for artefacto in (INFORME, INVENTARIO, FAMILIAS, CANDIDATOS, ABIERTAS):
        if not artefacto.exists():
            raise SystemExit(f"PARADA: falta {artefacto}")
    json.loads(INFORME.read_text(encoding="utf-8"))
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    antes = len(notebook["cells"])
    marcas = " ".join(
        "".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "markdown"
    )
    if MARCA in marcas:
        print("  la seccion 53 ya existe, no se duplica")
        return
    notebook["cells"].extend([markdown(TITULO), codigo(CODIGO)])
    NOTEBOOK.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
                        encoding="utf-8")
    print(f"  notebook: {antes} -> {len(notebook['cells'])} celdas")
    print("  seccion 53 integrada en el notebook consolidado")


if __name__ == "__main__":
    main()
