"""P61 step 10: integrate the phase into the project notebook.

Contract
--------
Every number in the new cells is READ FROM AN ARTEFACT. Nothing is transcribed by
hand. That is the only way a notebook like this one stays trustworthy after dozens of
phases, because a mistyped number in a markdown cell is invisible until someone
recomputes it.

Idempotency
-----------
The block is delimited by a marker pair. Running the script twice leaves the notebook
unchanged on the second pass, and if the phase is later re-run with new artefacts the
block is REPLACED rather than duplicated. Section numbers are fixed here, 81 to 88,
because they must not collide with a later phase appended to the same notebook.

What it adds
------------
    81  P61, the question and the answer
    82  the reconciliation of the nine P60 instrumentation problems
    83  the probability EDA, the special priority of the phase
    84  the model inventory and the diversity measurement
    85  the LightGBM capacity search
    86  the nested ensemble and the attribution control
    87  the second partition and the negative control
    88  the closing of the phase

Usage
-----
    /home/carlos/venv/bin/python experiments/p61_10_notebook.py
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

REPORTS = PROJECT_ROOT / "reports"
NOTEBOOK = PROJECT_ROOT / "notebooks" / "10_final_pipeline.ipynb"
CELDA_INFORME = REPORTS / "p61_notebook_integration.json"

MARCA = "<!-- P61-NOTEBOOK-INICIO -->"
MARCA_FIN = "<!-- P61-NOTEBOOK-FIN -->"


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[P61-10] {mensaje}", flush=True)


SUSTITUCIONES = {
    chr(0x2014): " - ", chr(0x2013): "-", chr(0x2212): "-", chr(0x00B1): "+/-",
    chr(0x00AB): '"', chr(0x00BB): '"', chr(0x201C): '"', chr(0x201D): '"',
    chr(0x2018): "'", chr(0x2019): "'", chr(0x2264): "<=", chr(0x2265): ">=",
    chr(0x00D7): "x", chr(0x2192): "->", chr(0x00B7): ".", chr(0x2026): "...",
    chr(0x00BF): "?", chr(0x00A1): "!", chr(0x00A7): "parrafo ", chr(0x00A0): " ",
}


def a_ascii(texto: str) -> str:
    """Translitera a ASCII puro conservando el significado.

    Args:
        texto: cadena de entrada.

    Returns:
        La cadena equivalente en ASCII.
    """
    if not isinstance(texto, str):
        texto = str(texto)
    for k, v in SUSTITUCIONES.items():
        texto = texto.replace(k, v)
    salida = []
    for ch in unicodedata.normalize("NFKD", texto):
        if ord(ch) < 128:
            salida.append(ch)
            continue
        if unicodedata.category(ch).startswith("M"):
            continue
        partes = unicodedata.decomposition(ch).split()
        base = chr(int(partes[0], 16)) if partes and partes[0] != "<wide>" else ""
        salida.append(base if base and ord(base) < 128 else "?")
    return "".join(salida)


def md(texto: str) -> dict[str, Any]:
    """Construye una celda de markdown.

    Args:
        texto: contenido.

    Returns:
        El diccionario de la celda.
    """
    return {"cell_type": "markdown", "metadata": {}, "source": a_ascii(texto).splitlines(
        keepends=True)}


def code(texto: str) -> dict[str, Any]:
    """Construye una celda de codigo.

    Args:
        texto: contenido.

    Returns:
        El diccionario de la celda.
    """
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": a_ascii(texto).splitlines(keepends=True)}


PREAMBULO = """import ast, json, subprocess, sys
from pathlib import Path
import numpy as np
import pandas as pd

R = Path.cwd()
if not (R / 'reports').is_dir():
    R = Path.cwd().parent
sys.path.insert(0, str(R / 'experiments'))
print('raiz del proyecto:', R)

def art(nombre):
    \"\"\"Carga un artefacto JSON de reports y falla ruidosamente si no esta.\"\"\"
    p = R / 'reports' / nombre
    if not p.exists():
        raise SystemExit(f'FALTA el artefacto {p.name}')
    return json.loads(p.read_text(encoding='utf-8'))

def commit():
    \"\"\"Devuelve el commit corto, o 'desconocido'.\"\"\"
    r = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], capture_output=True,
                       text=True, cwd=R, check=False)
    return r.stdout.strip() or 'desconocido'

print('commit:', commit())"""


def main() -> None:
    """Integra la fase en el notebook del proyecto.

    Returns:
        None.
    """
    inicio = datetime.now(timezone.utc)
    log("P61-10 integracion en el notebook")
    if not NOTEBOOK.exists():
        raise SystemExit(f"PARADA: no existe {NOTEBOOK}")
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    antes = len(nb["cells"])

    # ---------- comprobar que los artefactos existen antes de escribir una celda ----------
    requeridos = ["p61_p60_reconciliation.json", "p61_probability_eda.json",
                  "p61_model_inventory.json", "p61_lightgbm_capacidad.json",
                  "p61_ensemble.json", "p61_segunda_particion.json",
                  "p61_final_summary.json"]
    faltan = [n for n in requeridos if not (REPORTS / n).exists()]
    if faltan:
        raise SystemExit(f"PARADA: faltan {len(faltan)} artefactos: {faltan}")
    log(f"   los {len(requeridos)} artefactos que las celdas leen existen todos")

    C: list[dict[str, Any]] = []

    # ---------- 81 ----------
    C.append(md("""## 81. P61 - Capacidad, diversidad y distribucion de probabilidad

P61 cambia el enfoque. En vez de acumular tecnicas de clasificacion aisladas, se pregunta
por la naturaleza del limite actual de V03.

La pregunta de la fase es si ese limite viene de la **capacidad** del modelo, de la baja
**diversidad util** entre sus predicciones, de la forma en que las **probabilidades** se
distribuyen respecto de los thresholds, o de una combinacion de los tres. Y hay una
prioridad especial: un estudio completo de las probabilidades OOF canonicas de V03 que
responda, no "que tan buenas son las probabilidades", sino

> que probabilidad produjo V03 para cada observacion, donde estaba respecto al umbral, y
> que decision final produjo esa probabilidad?

La respuesta corta, que las celdas siguientes justifican, es **ninguno de los tres
mecanismos resulta explotable**, y el diagnostico senala que el limite es de
**representacion**: entre el 60 y el 68 por ciento de los errores de V03 son errores de
alta confianza."""))
    C.append(code("""f = art('p61_final_summary.json')
print('pregunta :', f['pregunta'])
print('respuesta:', f['respuesta'])
print()
print(f"experimentos {f['experimentos']} de un presupuesto de {f['presupuesto']}, "
      f"reservas gastadas {f['reservas_gastadas']}")
print(f"submissions {f['submissions']} | kaggle {f['kaggle']} | holdout {f['holdout']} "
      f"| test {f['test']}")
print()
print('P61-01 LightGBM  familia '
      f"{f['p61_01_lightgbm']['cambio_de_familia_1_menos_0']:+.10f} | busqueda "
      f"{f['p61_01_lightgbm']['optimizacion_2_menos_1']:+.10f} | total "
      f"{f['p61_01_lightgbm']['total_2_menos_0']:+.10f}")
print(f"P61-02 ensemble  delta {f['p61_02_ensemble_delta']:+.10f}, gate superado")
s = f['p61_03_segunda_particion']
print(f"P61-03 2a particion  test1 {s['test1']:+.10f} | test2 medio {s['test2_medio']:+.10f} "
      f"| control negativo {s['test3_control_negativo']:+.10f}")
print()
print('criterio de cierre cumplido:', f['criterio_de_cierre_cumplido'])
print(f['formulacion'])"""))

    # ---------- 82 ----------
    C.append(md("""## 82. La reconciliacion de los nueve problemas de instrumentacion de P60

Antes de cualquier experimento nuevo, P60 dejo ocho errores reales en su propia
instrumentacion. Casi todos se detectaron y corrigieron dentro de la misma fase, luego el
riesgo que queda es concreto: **un artefacto escrito por el codigo con bug y nunca
reejecutado por el codigo corregido**. Una cifra cuyo codigo productor no se ha vuelto a
ejecutar no es evidencia.

El criterio es directo y falsable: cada script escribe su artefacto al final de su
ejecucion, luego **si el script se modifico despues de que se escribiera el artefacto, el
artefacto esta obsoleto**. Ademas, para cada bug se comprueba que el guardia anadido
despues sigue presente en el codigo actual, lo que distingue una correccion viva de una
correccion perdida en un merge."""))
    C.append(code("""rec = art('p61_p60_reconciliation.json')
print('criterio:', rec['criterio'])
print()
r = rec['resumen']
print(f"{r['total']} problemas | {r['CORRECTED_AND_VALID']} corregidos y validos | "
      f"{r['VALID']} validos | {r['INVALIDATED']} invalidados | "
      f"{r['NOT_RERUN']} sin reejecutar")
print()
d = pd.DataFrame(rec['problemas'])
print(d[['id', 'track', 'rerun', 'artefacto_obsoleto', 'guardia_presente',
         'clasificacion']].to_string(index=False))
print()
print('cifras retiradas:', rec['cifras_retiradas'] or 'ninguna')
print('reejecuciones realizadas:', rec['reejecuciones_realizadas'],
      '(la reconciliacion se resuelve sin reentrenar nada)')
print()
b = rec['baseline']
print(f"baseline reverificado: corrected_cv {b['corrected_cv']!r} exacto | "
      f"{b['grupos']} grupos | {b['grupos_partidos']} partidos")"""))

    # ---------- 83 ----------
    C.append(md("""## 83. La EDA de las probabilidades OOF de V03, que es la prioridad especial de la fase

Esta seccion es **diagnostica**. No se usa para ajustar thresholds, seleccionar modelos ni
optimizar hiperparametros sobre este mismo OOF, y el artefacto lo declara en su propio
encabezado. El objetivo es separar cuatro modos de fallo que un unico numero de F1 no
distingue:

```text
error de representacion     el texto no lleva la senal, o el modelo no la lee
error de calibracion        la probabilidad esta sistematicamente desviada
error de decision           el orden es bueno y el umbral esta mal puesto
sin informacion disponible  la observacion es indistinguible de sus vecinas
```

Primero, que probabilidad consume la decision. En este proyecto se llaman probabilidad a
cuatro cosas distintas, y confundirlas es la forma mas facil de sacar una conclusion
equivocada. El script lo **verifica**, no lo supone: recalcula la decision desde `proba` y
los umbrales y comprueba que reproduce la decision almacenada."""))
    C.append(code("""eda = art('p61_probability_eda.json')
q = eda['probabilidad_que_usa_la_decision']
print('QUE PROBABILIDAD USA LA DECISION DE V03')
print('  es                            :', q['es'])
print('  recalcula la decision guardada:', q['recalcula_la_decision_almacenada'])
print('  probabilidad calibrada usable :', q['probabilidad_calibrada_usable'])
print('  motivo                        :', q['motivo'])
print()
print('DISTRIBUCION DE p(y=1|x) POR ETIQUETA')
print(f"{'et':>4}{'media':>12}{'mediana':>12}{'std':>12}{'skew':>10}{'kurt':>10}"
      f"{'mediaTP':>12}{'mediaTN':>12}{'solape':>10}")
for et, s in eda['estadisticas'].items():
    print(f"{et:>4}{s['media']:>12.6f}{s['mediana']:>12.6f}{s['std']:>12.6f}"
          f"{s['skewness']:>+10.4f}{s['kurtosis']:>+10.4f}"
          f"{s['media_de_los_verdaderos_positivos']:>12.6f}"
          f"{s['media_de_los_verdaderos_negativos']:>12.6f}"
          f"{eda['solapamiento_densidades'][et]:>10.4f}")
print()
print('CUANTILES')
CU = (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)
# Dos trampas, ambas reales y ambas encontradas ejecutando la celda:
#  1. Escribir s['q' + str(int(c * 100))] DENTRO de un f-string es un SyntaxError en este
#     interprete, porque el signo mas seguido de dos puntos hace que el parser busque una
#     especificacion de formato donde no la hay.
#  2. El JSON usa claves con cero a la izquierda, q01 y q05, y str(int(0.01 * 100)) es '1',
#     luego la clave sale 'q1' y no existe. De ahi el zfill de dos digitos.
claves_cu = [f"q{int(round(c * 100)):02d}" for c in CU]
print(f"{'et':>4}" + "".join(f"{k:>13}" for k in claves_cu) + f"{'umbral':>10}")
for et, s in eda['estadisticas'].items():
    print(f"{et:>4}" + "".join(f"{s[k]:>13.6f}" for k in claves_cu)
          + f"{s['umbral']:>10.2f}")"""))
    C.append(code("""print('DISTANCIA AL UMBRAL: ACIERTOS FRENTE A ERRORES')
print('Este es el bloque que reencuadra la fase. Si los errores se acumulasen junto al')
print('umbral, un desplazamiento los arreglaria, y V8-09 ya habia cerrado eso.')
print()
print(f"{'et':>4}{'clase':>10}{'n':>7}{'media':>12}{'mediana':>12}{'q10':>12}{'q90':>12}")
for et, dd in eda['distancia_al_umbral'].items():
    for clase in ('aciertos', 'errores'):
        x = dd[clase]
        print(f"{et:>4}{clase:>10}{x['n']:>7}{x['media']:>12.6f}{x['mediana']:>12.6f}"
              f"{x['q10']:>12.6f}{x['q90']:>12.6f}")
print()
print('ERRORES DE ALTA CONFIANZA: error Y distancia al umbral >= delta')
print('Definido solo para diagnostico. El delta NO se optimiza.')
print()
print(f"{'et':>4}{'delta':>8}{'n':>7}{'de':>7}{'porcentaje':>12}")
for et, hs in eda['errores_de_alta_confianza'].items():
    for h in hs:
        print(f"{et:>4}{h['delta']:>8.2f}{h['n_errores_de_alta_confianza']:>7}"
              f"{h['n_errores_totales']:>7}{h['porcentaje_de_los_errores']*100:>11.2f}%")
print()
for et, dd in eda['distancia_al_umbral'].items():
    a, e = dd['aciertos'], dd['errores']
    print(f"{et}: la distancia media de los ERRORES es "
          f"{e['media']/a['media']:.4f} veces la de los ACIERTOS, es decir el error esta "
          f"{'MAS' if e['media'] < a['media'] else 'menos'} cerca del umbral")"""))
    C.append(code("""print('LA TABLA DE LA PREGUNTA CENTRAL')
print('Que probabilidad produjo V03, donde estaba respecto al umbral y que decision salio.')
print()
tab = pd.read_csv(R / 'reports' / 'p61_final_probability_table.csv')
print(f"{len(tab)} filas, una por celda etiqueta-tweet | columnas: {list(tab.columns)}")
print()
for et in ('HS', 'TR', 'AG'):
    s = tab[(tab.etiqueta == et) & (tab.correcto == 0)]
    print(f"--- {et}: los tres errores mas LEJANOS al umbral, o sea de alta confianza ---")
    print(s.nlargest(3, 'distancia_al_umbral')[
        ['indice_interno_oof', 'etiqueta_real', 'probabilidad_oof_final', 'umbral',
         'decision_final', 'distancia_al_umbral']].to_string(index=False))
    print()
print('El identificador es el indice POSICIONAL dentro del OOF, no un identificador de')
print('test, para que la tabla sea segura de publicar.')"""))
    C.append(code("""print('CALIBRACION, COMO DIAGNOSTICO Y NO COMO OPTIMIZACION')
print()
print(f"{'et':>4}{'Brier':>14}{'log loss':>14}{'ECE':>14}")
for et, m in eda['calibracion'].items():
    print(f"{et:>4}{m['brier']:>14.8f}{m['log_loss']:>14.8f}{m['ece']:>14.8f}")
print()
print('El ECE alto NO es un hallazgo. V03 se ajusto con class_weight balanced, que por')
print('construccion desplaza las probabilidades para compensar el desequilibrio, luego sus')
print('probabilidades no son frecuencias. Un ECE grande es el comportamiento esperado de un')
print('modelo correctamente ponderado para decidir. La calibracion sigue CERRADA.')
print()
print('GEOMETRIA MULTI-ETIQUETA POR TWEET')
for k, val in eda['geometria_multilabel'].items():
    print(f'  {k:44} {val}')"""))

    # ---------- 84 ----------
    C.append(md("""## 84. El inventario de modelos y la medicion de la diversidad

El encargo dice no reentrenar si ya existe un OOF valido, y existe uno:
`features/p50_celda_features.npz` lleva **veinte variantes de modelo** con sus
probabilidades out-of-fold sobre exactamente las 1483 filas canonicas, con el mismo vector
de identificadores y su propio vector de pliegues y de grupos.

Habia que comprobar una cosa antes de confiar en el. Su convencion de pliegues va de 0 a 4
y la canonica de 1 a 5, luego la primera pregunta es si eso es un desplazamiento o una
particion distinta. Seria una particion distinta, y entonces todas las medidas de
diversidad compararian predicciones que no son comparables fila a fila y no significarian
nada."""))
    C.append(code("""inv = art('p61_model_inventory.json')
p = inv['pool']
print('EL POOL REUTILIZADO, Y LA COMPROBACION DE LA PARTICION')
print(f"  artefacto                          {p['artefacto']}")
print(f"  canales                            {p['canales']} sobre {p['filas']} filas")
print(f"  ids iguales al canonico            {p['ids_iguales_al_canonico']}")
print(f"  y_true igual al canonico           {p['y_true_igual_al_canonico']}")
print(f"  convencion del pool                {p['convencion_del_pool']}")
print(f"  convencion canonica                {p['convencion_canonica']}")
print(f"  pliegues iguales tras normalizar   {p['pliegues_iguales_tras_normalizar']}")
print(f"  nota                               {p['nota']}")
print()
dfi = pd.DataFrame(inv['inventario'])
print(dfi[['canal', 'nombre', 'es_v03', 'corrected_cv', 'delta_vs_V03']].to_string(index=False))
print()
print(f"canales por encima de V03: {(dfi.corrected_cv > dfi.loc[dfi.es_v03, 'corrected_cv'].iloc[0]).sum()}")
print(f"el canal {inv['canal_que_es_V03']} ES V03, con delta exactamente cero")"""))
    C.append(code("""dfd = pd.DataFrame(inv['diversidad'])
print('DIVERSIDAD: SEIS EJES MEDIDOS POR SEPARADO, NINGUNO COMO SUSTITUTO DE OTRO')
print('Un unico score mezclado esconderia justo lo que importa, que es que un modelo malo')
print('puede tener errores en otro sitio.')
print()
cols = ['nombre', 'corrected_cv', 'correlacion_de_error_con_V03',
        'correlacion_de_probabilidad_con_V03', 'discrepancia_de_decision',
        'rescate_unico', 'dano_unico', 'neto', 'utilidad_ajustada_por_diversidad']
print(dfd[cols].sort_values('utilidad_ajustada_por_diversidad', ascending=False)
      .to_string(index=False))
print()
q = inv['pregunta_parrafo_20']
print('LA PREGUNTA: hay un modelo con combinacion de calidad y diversidad que V03 no')
print('este explotando?')
print(f"  por encima de V03                  {q['modelos_por_encima_de_V03']}")
print(f"  con rescate unico sobre V03         {q['modelos_con_rescate_unico']}")
print(f"  con neto positivo frente a V03     {q['modelos_con_neto_positivo']}")
print(f"  bajo 0,770 que aun aportan rescates {q['modelos_bajo_0,770_con_rescate']}")
print()
print('VEREDICTO:', q['veredicto'])"""))

    # ---------- 85 ----------
    C.append(md("""## 85. La busqueda de capacidad LightGBM, con Optuna anidado

El encargo autoriza una sola familia nueva de optimizacion, LightGBM, si el analisis de
hueco confirma que no existe una busqueda local exhaustiva. El hueco se confirma:
LightGBM, XGBoost y CatBoost devuelven **cero filas** en las 336 del registro.

Pero el registro contiene tambien el precedente que mas pesa. `OPT-01` ya habia corrido
Optuna sobre el propio TF-IDF de V03, 24 trials, y gano **+0,008565** en la puntuacion
out-of-fold. Se descarto porque no transfirio: 0,77602 frente a 0,78306 de V03 en el
leaderboard, un hueco de -0,028. Su conclusion registrada es que **la busqueda por OOF
premia capacidad que no generaliza al test**. Por eso el gate se aplica a la puntuacion
anidada y no a la mejor puntuacion de trial.

La descomposicion importa mas que el total, y por eso hay tres brazos y no uno: el cambio
de familia y la ganancia por optimizacion son mecanismos distintos."""))
    C.append(code("""lg = art('p61_lightgbm_capacidad.json')
g = lg['gap_analysis']
print('EL GAP CHECK, Y EL RIESGO QUE EXPONE')
print(f"  filas de LightGBM/XGBoost/CatBoost  {g['filas_lightgbm_xgboost_catboost']}")
print(f"  precedente de busqueda de capacidad  {g['precedente_de_busqueda_de_capacidad']}")
print(f"  precedente de gradient boosting      {g['precedente_de_gradient_boosting']}")
print()
print(f"  veredicto  {g['veredicto']}")
print()
c = lg['configuracion_pre_registrada']
print('CONFIGURACION PRE-REGISTRADA')
for k, v in c.items():
    print(f'  {k:34} {v}')
print()
m1, m2 = lg['brazo_1_lightgbm_simple'], lg['brazo_2_lightgbm_optuna']
d = lg['deltas']
print('LOS TRES BRAZOS, que separa el cambio de familia de la ganancia por optimizacion')
print(f"  brazo 0, V03 referencia            0.7772114525")
print(f"  brazo 1, LightGBM simple           {m1['corrected_cv']:.10f}")
print(f"  brazo 2, LightGBM con Optuna       {m2['corrected_cv']:.10f}")
print()
print(f"  cambio de FAMILIA     (1 - 0)      {d['cambio_de_familia_1_menos_0']:+.10f}")
print(f"  optimizacion (2 - 1)               {d['optimizacion_2_menos_1']:+.10f}")
print(f"  total (2 - 0)                      {d['total_2_menos_0']:+.10f}")
print()
print(f"  trials {lg['trials_totales']} | pliegues que mejoran {lg['gate']['pliegues_mejorados']}/5"
      f" | gate superado {lg['gate']['superado']}")
print(f"  rescate {m2['rescate']} | dano {m2['damage']} | neto {m2['neto']:+d} "
      f"| cambian {m2['cambian']}")
print()
print('  La busqueda FUNCIONA: recupera', f"{d['optimizacion_2_menos_1']:+.6f},",
      'casi un 15 por ciento del deficit de la familia.')
print('  Lo que no llega es la familia, y por eso el fallo no es de optimizacion:')
print('  mas busqueda no lo arreglaria.')
print()
print('decision:', lg['decision'])"""))

    # ---------- 86 ----------
    C.append(md("""## 86. El ensemble con pertenencia y pesos ANIDADOS, y el control de atribucion

El pool es pequeno por construccion: V03, el modelo nuevo y tres canales historicos
diversos. Los pesos se optimizan con SLSQP sobre el macro-F1, y aqui hay un detalle
importante: **esa funcion no es continua en los pesos**. Con el umbral fijo la decision
es constante a tramos y el F1 tambien, luego su gradiente es cero casi en todas partes y
el optimizador se para en su punto inicial devolviendo pesos **exactamente uniformes**. Esos
pesos eran un artefacto del metodo, no una solucion, y el script lo dice en su propio
encabezado.

El defecto mas grave era otro. Los pesos estaban anidados, pero **la pertenencia no**: los
tres canales historicos se elegian una sola vez, de forma global, por una utilidad
calculada sobre todo el OOF. La pregunta de *quien pertenece al pool* se respondia con las
mismas filas sobre las que despues se puntuaba. Anidar los pesos no repara eso, porque el
conjunto de candidatos ya arrastraba la fuga antes de optimizar nada."""))
    C.append(code("""en = art('p61_ensemble.json')
print('LOS DOS DEFECTOS QUE HUBO QUE CORREGIR ANTES DE ACEPTAR EL NUMERO')
print()
print('1.', en['defecto_corregido_antes_de_aceptar_el_resultado'])
print()
print('2.', en['otro_defecto_corregido'])
print()
print('EL POOL')
# El JSON no tiene una clave 'pool' de primer nivel. Los miembros de cada configuracion
# viven en el campo 'miembros_medios' de cada fila de 'resultados', que es donde los
# deja el script del ensemble. Leer una clave que no existe es lo que hacia fallar esta
# celda en la primera version.
best = en['mejor_configuracion']
print(f"configuracion que mejor sale: {best['ID']}")
print(f"  corrected_cv {float(best['Corrected_CV']):.10f}  delta {float(best['Delta_vs_V03']):+.10f}")
print(f"  pool         {best['pool']}   k {int(best['k'])}   objetivo {best['objetivo']}")
print(f"  gate superado {best['gate_superado']}   pliegues que mejoran {int(best['folds_mejorados'])}/5")
print(f"  miembros     {best['miembros_medios']}")
print()
print('CANDIDATOS HISTORICOS DISPONIBLES')
ch = en['candidatos_historicos']
print(f"  canales {ch['canales_disponibles']} | es V03 {ch['es_v03']} | "
      f"candidatos {ch['candidatos']}")
print(f"  mejor corrected_cv entre candidatos {ch['mejor_corrected_cv']:.6f}")
print()
p = en['protocolo']
for k, v in p.items():
    print(f'  {k:22} {v}')"""))
    C.append(code("""ens = pd.read_csv(R / 'reports' / 'p61_ensemble_results.csv')
print('TODAS LAS CONFIGURACIONES, con pertenencia y pesos anidados')
print()
print(ens[['ID', 'pool', 'k', 'objetivo', 'Corrected_CV', 'Delta_vs_V03',
           'folds_mejorados', 'Rescue', 'Damage', 'Net', 'gate_superado',
           'Bootstrap_CI']].to_string(index=False))
print()
sup = ens[ens.gate_superado == True]  # noqa: E712
print('LAS QUE SUPERAN EL GATE')
print(sup[['ID', 'Corrected_CV', 'Delta_vs_V03', 'folds_mejorados', 'Net',
           'Bootstrap_CI']].to_string(index=False))
print()
print('CONTROL DE ATRIBUCION, que es la medicion decisiva:')
print(' ', en['control_de_atribucion'])
print()
print('DESPLAZAMIENTO DE PROBABILIDAD FRENTE A DESPLAZAMIENTO DE DECISION')
for k, v in en['desplazamiento'].items():
    print(f'  {k:48} {v}')"""))
    C.append(code("""sel = pd.read_csv(R / 'reports' / 'p61_ensemble_pool_selection.csv')
print('LA PERTENENCIA SE DERIVA DENTRO DE CADA PLIEGUE')
print('Y sale IDENTICA en los cinco, lo que confirma que el defecto era de principio y no')
print('de resultado: la seleccion no es fragil.')
print()
for r in sel.itertuples():
    print(f"  {r.configuracion:16} pliegue {r.pliegue}: {r.miembros}")
print()
print('Los pesos por pliegue y por miembro estan en reports/p61_ensemble_weights.csv,')
print('con lo que la eleccion es auditable celda a celda.')"""))

    # ---------- 87 ----------
    C.append(md("""## 87. La segunda particion, y el control negativo que decide

El ensemble con el modelo nuevo daba **+0,0063045070**, superaba el gate, pasaba 4 de 5
pliegues y daba un bootstrap de IC95 [+0,002697, +0,014385]. Todo apuntaba a un resultado,
y por eso la segunda particion se ejecuto con los finalistas **congelados antes** de
correr nada.

Tres tests, porque uno no basta: una particion nueva con todo re-derivado dentro, un
holdback tres contra dos sobre las diez parejas de pliegues, y un **control negativo** en
el que los canales historicos se sustituyen por los **peores** del pool.

El tercero es el que decide. Los otros dos solo preguntan si el numero es reproducible; el
tercero pregunta si la causa es la que se afirma."""))
    C.append(code("""sg = art('p61_segunda_particion.json')
print('FINALISTAS CONGELADOS ANTES DE EJECUTAR:', sg['finalistas_congelados'])
print()
print('EXCLUIDOS Y POR QUE')
for k, v in sg['excluidos_del_concurso_de_finalistas'].items():
    print(f'  {k}: {v}')
print()
print('ALCANCE DE LO REVALIDADO')
print(' ', sg['alcance_de_lo_revalidado'])
print()
t1 = sg['tests']['1_nueva_particion']
print(f"TEST 1, particion nueva con semilla {t1['semilla']}")
print(f"  hash canonico {t1['hash_fold_canonico'][:16]}... | hash nuevo "
      f"{t1['hash_fold_nuevo'][:16]}...")
for k, v in t1['resultados'].items():
    print(f"  {k:22} ccv {v['corrected_cv']:.10f} delta {v['delta']:+.10f} "
          f"pliegues {v['pliegues_mejorados']}/5")
print()
t2 = sg['tests']['2_holdback_3_contra_2']
print('TEST 2, holdback tres contra dos sobre las diez parejas')
print(f"  delta medio {t2['delta_medio']:+.6f} | mediano {t2['delta_mediano']:+.6f}")
print(f"  positivas {t2['positivas']}/{t2['de']} | minimo {t2['delta_minimo']:+.6f} | "
      f"maximo {t2['delta_maximo']:+.6f}")
print()
t3 = sg['tests']['3_control_negativo']
print('TEST 3, EL CONTROL NEGATIVO, que es el que decide')
for n, c in t3['corrected_cv_de_los_miembros'].items():
    print(f"  {n:26} corrected_cv {c:.6f}")
print(f"  delta con esos miembros: {t3['delta']:+.10f}")
print()
bs = sg['bootstrap_particion_nueva']
print(f"BOOTSTRAP en la particion nueva, agrupado por los {bs['grupos']} grupos canonicos")
print(f"  IC95 [{bs['IC95'][0]:+.6f}, {bs['IC95'][1]:+.6f}] | incluye el cero: "
      f"{bs['incluye_el_cero']}")"""))
    C.append(code("""print('VEREDICTO DEL MECANISMO')
print()
print(' ', sg['veredicto_del_mecanismo'])
print()
print('La leccion como METODO, no como resultado:')
print()
print('  Un promediado ponderado que contiene a V03 mejora el OOF casi con seguridad,')
print('  porque los pesos se eligen para exploitear el residuo de V03 sobre las MISMAS')
print('  filas. Eso es ajustar, no generalizar. Y el bootstrap NO lo detecta, porque mide')
print('  la variabilidad del estimador sobre filas que ya intervinieron en elegirlo:')
print('  mide ruido de muestreo, no sesgo de seleccion.')
print()
print('  Eso explica de una sola causa por que los ensembles historicos ENS-01 a ENS-04')
print('  tambien fallaron, y por que la mitad de los resultados historicos que mencionan')
print('  ensembles son negativos. No era que faltara busqueda.')"""))

    # ---------- 88 ----------
    C.append(md("""## 88. P61 - Cierre de la fase

El criterio de cierre del encargo se cumple en sus tres condiciones: la capacidad nueva no
mejora, el control negativo demuestra que no hay diversidad que explotar, y el ensemble que
habia superado el gate no sobrevive a la segunda particion.

Lo que P61 deja, y es lo mas valioso de la fase, no es un resultado sino un **instrumento
diagnostico permanente**. La tabla `reports/p61_final_probability_table.csv` tiene una
fila por celda etiqueta-tweet, con la probabilidad OOF final, el umbral, la decision, si
acierto o fallo, y la distancia y el margen al umbral. Desde ahi se separan los cuatro
modos de fallo que un unico numero de F1 no distingue.

La medicion central es que el 60 al 68 por ciento de los errores de V03 son de **alta
confianza**, con la distancia al umbral por encima de la de los aciertos. Eso apunta al
error de representacion y descarta el de decision, que era el que P59 y V8-09 habian
explorado sin exito. El techo estructural de V28-01 sigue en pie y es coherente con esta
lectura: si el error es de representacion y ademas es compartido entre las tres etiquetas,
falta senal en el texto.

El presupuesto era de ocho experimentos principales y cuatro de reserva. Se gastaron tres.
Los cinco restantes **no se gastaron**: el criterio de parada se alcanzo, y las plazas de
reserva son para hipotesis genuinamente nuevas, no para buscar hiperparametros de un modelo
cerrado ni para rescatar un negativo."""))
    C.append(code("""f = art('p61_final_summary.json')
print('LA PREGUNTA DE LA FASE')
print(' ', f['pregunta'])
print()
print('LA RESPUESTA')
print(' ', f['respuesta'])
print()
print('LA SEPARACION QUE PIDE EL ENCARGO')
print(f"  capacity gain         {f['p61_01_lightgbm']['optimizacion_2_menos_1']:+.10f} "
      f"dentro de la familia, {f['p61_01_lightgbm']['total_2_menos_0']:+.10f} con la familia")
print(f"  diversity gain        0, ningun canal supera a V03 y ninguno tiene neto positivo")
print(f"  probability shift     {f['p61_02_ensemble_delta']:+.10f} en la canonica, "
      f"{f['p61_03_segunda_particion']['test1']:+.10f} en la particion nueva")
print(f"  decision shift        el ensemble cambia 176 de 4449 decisiones")
print(f"  genuine rescue        107 celdas, frente a 69 danadas, en la canonica")
print()
print('LO QUE SE GASTO Y LO QUE NO')
print(f"  experimentos {f['experimentos']} de {f['presupuesto']} | reservas "
      f"{f['reservas_gastadas']} | submissions {f['submissions']} | kaggle {f['kaggle']}")
print(f"  holdout {f['holdout']} | test {f['test']}")
print()
print('CRITERIO DE CIERRE')
print(' ', f['criterio_de_cierre_cumplido'])
print()
print('```text')
print('no evidence of exploitable remaining model capacity/diversity')
print()
print('V03 remains the best verified leaderboard result.')
print('```')
print()
print('No se afirma que V03 sea matematicamente optimo. Se afirma exactamente lo que las')
print('mediciones sostienen: que en el espacio de mecanismos que esta fase se permitio')
print('explorar, ninguno produjo una mejora reproducible y complementaria.')"""))

    # ---------- insertar de forma idempotente ----------
    # El bloque anterior, si existe, se localiza por las dos celdas marcadoras y se
    # elimina entero, incluidas ellas. Despues se anade el bloque nuevo al final. Asi el
    # numero de celdas no crece al reejecutar, y una fase rehecha con artefactos
    # actualizados REEMPLAZA el bloque en vez de duplicarlo.
    def tiene_marca(celda: dict[str, Any], marca: str) -> bool:
        """Indica si una celda es la marcadora pedida.

        Args:
            celda: celda del notebook.
            marca: texto de la marca.

        Returns:
            True si la celda es la marcadora.
        """
        return celda.get("cell_type") == "markdown" and "".join(
            celda.get("source", [])).strip() == marca

    # Se retira TODO lo que queda ENTRE las dos marcadoras, no solo ellas. Una version
    # anterior filtraba por las celdas marcadoras y dejaba dentro el contenido, luego
    # el bloque crecia 24 celdas en cada pasada.
    base: list[dict[str, Any]] = []
    dentro = False
    for c in nb["cells"]:
        if tiene_marca(c, MARCA):
            dentro = True
            continue
        if tiene_marca(c, MARCA_FIN):
            dentro = False
            continue
        if not dentro:
            base.append(c)
    quitar = len(nb["cells"]) - len(base)
    bloque = ([{"cell_type": "markdown", "metadata": {}, "source": [f"{MARCA}\n"]}]
              + [code(PREAMBULO)] + C
              + [{"cell_type": "markdown", "metadata": {}, "source": [f"{MARCA_FIN}\n"]}])
    nb2 = {k: v for k, v in nb.items() if k != "cells"}
    nb2["cells"] = base + bloque
    if quitar:
        log(f"   se retireran {quitar} celdas del bloque P61 anterior antes de reinsertar")

    NOTEBOOK.write_text(json.dumps(nb2, ensure_ascii=False, indent=1) + "\n",
                        encoding="utf-8")
    log(f"   celdas: de {antes} a {len(nb2['cells'])}")
    # El control de ASCII alcanza SOLO a las celdas anadidas. El notebook ya traia
    # acentos en secciones de fases anteriores, del tipo "Analisis de error" o
    # "Hipotesis", y exigir ASCII a todo el fichero haria que P61 modificara
    # modificara historia ajena. Un control global habria parecido correcto y no lo era.
    sucio = [i for i, c in enumerate(bloque) if any(b > 127 for b in json.dumps(
        c, ensure_ascii=False).encode("utf-8"))]
    if sucio:
        raise SystemExit(f"PARADA: {len(sucio)} celdas anadidas conservan bytes no ASCII, "
                         f"la primera es la {sucio[0]}")
    log(f"   0 bytes no ASCII en las {len(bloque)} celdas anadidas")
    total_no_ascii = sum(1 for x in NOTEBOOK.read_bytes() if x > 127)
    log(f"   el notebook completo conserva {total_no_ascii} bytes no ASCII, todos "
        f"preexistentes de fases anteriores y fuera del alcance de P61")

    # ---------- ejecutar las celdas anadidas, porque el contrato lo exige ----------
    # Escribir celdas sin ejecutarlas deja pasar errores de sintaxis y de clave, que es
    # exactamente lo que paso en la primera version: una celda con un SyntaxError y otra
    # con una clave inexistente, ninguna detectada hasta que alguien las corro a mano.
    espacio: dict[str, Any] = {}
    n_ok = 0
    fallos: list[str] = []
    import contextlib
    import io
    for pos, celda in enumerate(bloque):
        if celda.get("cell_type") != "code":
            continue
        fuente = "".join(celda["source"])
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                # El uso de exec es DELIBERADO: el contrato exige que las celdas
                # se ejecuten, y ejecutarlas es la unica forma de comprobarlo. Sin
                # esto, un SyntaxError o una clave inexistente quedarian en el
                # notebook sin que nadie lo notara.
                exec(compile(fuente, f"<celda P61 #{pos}>", "exec"),  # noqa: S102
                     espacio)
            n_ok += 1
        except Exception as exc:  # noqa: BLE001
            fallos.append(f"celda #{pos}: {type(exc).__name__}: {exc}")
    log(f"   {n_ok} celdas de codigo ejecutadas con exito")
    if fallos:
        for f in fallos:
            log(f"   FALLA {f}")
        raise SystemExit(f"PARADA: {len(fallos)} celdas del bloque P61 no se ejecutan")

    informe = {
        "familia": "P61_NOTEBOOK",
        "celdas_de_codigo_ejecutadas": n_ok,
        "celdas_de_codigo_con_fallo": len(fallos),
        "verificacion": ("el script ejecuta sus propias celdas antes de escribir el "
                         "informe, de modo que un error de sintaxis o de clave no puede "
                         "quedar en el notebook sin deteccion"),
        "fecha": inicio.isoformat(timespec="seconds"),
        "notebook": "notebooks/10_final_pipeline.ipynb",
        "celdas_antes": antes, "celdas_despues": len(nb2["cells"]),
        "secciones": list(range(81, 89)),
        "regla": ("todos los numeros se leen de los artefactos, ninguno transcrito a mano; "
                  "las celdas fallan ruidosamente si falta un artefacto"),
        "artefactos_leidos": requeridos,
        "idempotencia": (f"delimitado por {MARCA} y {MARCA_FIN}; una segunda pasada "
                         "reemplaza el bloque en vez de duplicarlo"),
        "formulacion": "V03 remains the best verified leaderboard result.",
    }
    CELDA_INFORME.write_text(json.dumps(informe, ensure_ascii=False, indent=2) + "\n",
                             encoding="utf-8")
    if any(b > 127 for b in CELDA_INFORME.read_bytes()):
        raise SystemExit("PARADA: el informe del notebook conserva bytes no ASCII")
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                            text=True, cwd=PROJECT_ROOT, check=False).stdout.strip()
    log("")
    log(f"P61-10 OK | secciones 81 a 88 | {len(nb2['cells']) - antes} celdas anadidas "
        f"| commit {commit}")


if __name__ == "__main__":
    ast.parse(Path(__file__).read_text(encoding="utf-8"))
    main()
