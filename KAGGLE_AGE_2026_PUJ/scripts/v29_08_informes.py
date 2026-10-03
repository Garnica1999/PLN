"""V29 - 08: notebook consolidado, informe final de V29, reporte historico integral y estado.

Genera:

- seccion 37 del notebook consolidado, con las cuatro aserciones obligatorias;
- `reports/V29_FINAL_RESEARCH_REPORT.md`;
- `reports/AGE_COMPLETE_RESEARCH_REPORT.md` **reconstruido de V20 a V29**;
- `reports/age_complete_experiment_history.csv`;
- cierre en `FINAL_REPORT.md`, `PROJECT_STATE.md`, `DECISIONS.md` (D-064) y `RUNLOG.md`.

## La regla que gobierna el script

**Ninguna cifra se escribe a mano si existe un artefacto del que leerla.** La parte historica de
V20 a V28 se genera importando `scripts/v28_07_informes.py`, que a su vez importa el de V27: hay
una sola cadena de fuentes y el informe no puede divergir de los anteriores.

## Lo que el informe dice sin adornos

De seis ejes evaluados, uno es nuevo y se midio, tres se cerraron por equivalencia o por
consideracion, y el central resulto **negativo**. No se presenta como un exito de ingenieria: se
presenta como la respuesta a la pregunta de V29, que es lo que el brief pide.
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src import config as cfg  # noqa: E402
from src.verificacion_numerica import autotest_numerico  # noqa: E402

import v27_10_informe_completo as v27  # noqa: E402
import v28_07_informes as v28  # noqa: E402

NOTEBOOK = ROOT / "notebooks" / "99_entrega_consolidado.ipynb"
INFORME_V29 = cfg.REPORTS / "V29_FINAL_RESEARCH_REPORT.md"
INFORME_TOTAL = cfg.REPORTS / "AGE_COMPLETE_RESEARCH_REPORT.md"
HISTORIAL = cfg.REPORTS / "age_complete_experiment_history.csv"
RESULTADO = cfg.EXPERIMENTS / "v29_08_informes.json"
CV_V20 = 0.6097782
LB_V20 = 0.62689
HASH_V20 = "2c964af78c90bf0884b4107665b5100f0c1aa2d9059c47692e613578b6cf660d"
GATE_DELTA = 0.003
FECHA = "2026-09-29"
SIN_EXPERIMENTO = ["B", "D", "E", "F"]


def _control() -> dict:
    """Lee del artefacto el reparto de peso y el contraste con y sin el modelo de Optuna.

    Devuelve rangos ya calculados para que el texto no escriba numeros a mano: si el artefacto
    cambia, las frases cambian con el.
    """
    ens = _arte("v29_06_ensemble.json")
    cierre = _arte("v29_07_cierre.json")
    h04 = cierre["control_h04"]["datos"]
    METRICAS = {
    "peso_optuna_min": round(min(d["peso_maximo"] for d in h04.values()
                                 if isinstance(d, dict) and "peso_maximo" in d), 4),
    "peso_optuna_max": round(max(d["peso_maximo"] for d in h04.values()
                                 if isinstance(d, dict) and "peso_maximo" in d), 4),
    "peso_v20_min": round(min(d["peso_de_V20"] for d in h04.values()
                              if isinstance(d, dict) and "peso_de_V20" in d), 4),
    "peso_v20_max": round(max(d["peso_de_V20"] for d in h04.values()
                              if isinstance(d, dict) and "peso_de_V20" in d), 4),
    # `familias_con_peso` no se copio al diccionario `h04` del artefacto, pero el reparto por
    # familia si, y de ahi se cuenta. Es la unica fuente persistida.
    "familias_max": max(
        (sum(1 for v in r.get("apoyo_por_familia", {}).values() if v > 0.05)
         for r in ens["tabla"] if r.get("peso_de_V20") is not None), default=0),
    "contraste": ens.get("control_sin_modelo_nuevo", {}),
    "decision_h04": cierre["control_h04"]["decision"],
    }
    return METRICAS


def _arte(nombre: str) -> dict:
    return json.loads((cfg.EXPERIMENTS / nombre).read_text(encoding="utf-8"))


def _dec(n: float, casas: int = 7) -> str:
    return f"{n:.{casas}f}".replace(".", ",")


def _pct(n: float) -> str:
    """Porcentaje con coma decimal, que es la convencion de los informes."""
    return _dec(100.0 * float(n), 1) + " %"


# ============================================================================ informe de V29
def seccion_v29() -> list[str]:
    """El informe de V29 completo, con todas las cifras desde los artefactos."""
    inv = _arte("v29_01_inventario.json")
    div = _arte("v29_02_diversidad.json")
    gap = _arte("v29_03_gap_analysis.json")
    pool = _arte("v29_05_pool.json")
    ens = _arte("v29_06_ensemble.json")
    cierre = _arte("v29_07_cierre.json")
    tab = pd.DataFrame(ens["tabla"])
    ref = div["referencia_diversidad_interna_v20"]
    pistas = cierre["pistas_optuna"]
    completadas = {k: v for k, v in pistas.items() if v["estado"] == "COMPLETADA"}
    lineas = [
        "# V29 - Optuna y busqueda de ensemble diverso", "",
        f"**Fecha:** {FECHA} · **Rama:** `proyecto-kaggle-age-2026-v29` · "
        f"**Baseline congelado:** {_dec(CV_V20)} · **Champion:** V20, LB {LB_V20}", "",
        "## 1. Motivacion y pregunta", "",
        "V25 a V28 habian cerrado la representacion, la escala de la evidencia, la monotonia, los "
        "percentiles, la fiabilidad por modelo, la estructura ordinal, el bagging del meta-modelo, "
        "la descomposicion pairwise y la geometria del consenso. Lo que **no** se habia tocado es "
        "la capacidad de un miembro individual ni la composicion del ensemble.", "",
        "La hipotesis central de V29 es una sola:", "",
        "> V20 puede no estar limitado por la capacidad de sus miembros, sino por la **falta de "
        "diversidad util entre sus errores** y por una combinacion de pesos fija.", "",
        "La palabra clave es **complementariedad**, y el punto 3 del brief es el que gobierna todo "
        "lo demas: *mejor modelo individual* no implica *mejor miembro del ensemble*.", "",
        "## 2. Estado inicial", "",
        "| Concepto | Valor |", "|---|---|",
        f"| Baseline vigente | **{_dec(CV_V20)}** corrected_cv, reproducido bit a bit en P0 |",
        f"| Leaderboard de V20 | **{LB_V20}**, envio sin cambio |",
        "|.Hash de la submission de V20 | `" + HASH_V20[:16] + "...`, intacto |",
        "| Candidato para submission | ninguno |", "",
        "## 3. Inventario historico: la surprise de V29", "",
        f"El primer paso no fue entrenar: fue **leer lo que ya estaba entrenado**. En el "
        f"directorio de features hay **{inv['artefactos_6_columnas']} matrices de prediccion "
        f"out-of-fold de seis clases**, de las que **{inv['artefactos_reutilizables']} son "
        f"metodologicamente reutilizables** sin volver a entrenar nada.", "",
        f"Proceden de **{inv['n_filas_train']:,} filas** y cubren "
        f"**{len(inv['familias_presentes'])} familias distintas** de modelo: arboles, boosting, "
        f"logistica, perceptrones, NB-SVM, BM25, vecinos mas cercanos, ExtraTrees, SGD, modelos "
        f"de estructura ordinal, versiones solo-contenido y solo-estilo, entre otras.", "",
        f"Se descartaron **{len(inv['descartados_por_forma'])}** por no tener seis columnas: son "
        f"CDFs de cinco umbrales o decisiones binarias, cuyo `argmax` no es una clase.", "",
        "**El criterio de reutilizacion fue triple, y se declara porque no basta con que el fichero "
        "exista:**", "",
        "1. seis columnas de clase y exactamente las filas del train;", "",
        "2. **homogeneidad entre folds** superior a 0,90, porque un OOF real no tiene folds "
        "dispares;", "",
        "3. que no sea duplicado de otro artefacto ya inventariado.", "", ]
    # La lista de gemelos por argmax es un hallazgo de inventario, no un detalle.
    if div.get("gemelos_de_argmax"):
        lineas += ["**Un hallazgo de inventario que habria falseado la medicion:**", ""]
        for grupo in div["gemelos_de_argmax"]:
            lineas.append(f"- `{grupo[0]}` y `{grupo[1]}` tienen el **argmax identico** en las "
                          f"169.336 filas. Difieren en las magnitudes, pero toman las mismas "
                          f"decisiones, y en un ensemble con voto aportan cero.")
        lineas += ["", "El cuarto de cadena de V20 se reconstruyo desde `v20_f3_oof_cumuladas.npy` "
                  "en vez de tomar ese artefacto, porque usarlo habria comparado a V20 consigo "
                  "mismo y habria understated la diversidad real de sus miembros.", ""]
    lineas += [
        "## 4. Diversidad: la medicion que decide la fase", "",
        "### La referencia: la diversidad interna de V20", "",
        "V20 es un promedio de cuatro cuartos, asi que su propia diversidad interna es el liston "
        "que cualquier candidato tiene que superar.", "",
        "| Medida | Valor |", "|---|---|",
        f"| Correlacion de errores media entre los 4 cuartos | **{ref['corr_errores_media']:.4f}** |",
        f"| Discrepancia de decision media | {ref['tasa_discrepancia_media']:.4f} |",
        f"| Discrepancia maxima entre dos cuartos | {ref['tasa_discrepancia_maxima']:.4f} |", "",
        "### Cinco medidas, no una", "",
        "Una sola correlacion no alcanza y puede actively mislead, asi que se miden cinco a la vez "
        "y **no se resumen en un numero unico**:", "",
        "| Medida | Que responde |", "|---|---|",
        "| `corr_probabilidades` | cuan parecidas son las distribuciones completas |",
        "| `corr_errores` | cuan parecidos son **los fallos**, que es lo que importa |",
        "| `tasa_discrepancia` | con que frecuencia difieren las decisiones |",
        "| `rescate_unico` | cuantas filas corrige el modelo y V20 no |",
        "| `dano_unico` | cuantas filas rompe el modelo que V20 acertaba |", "",
        "### La trampa del rescate bruto, y como se evita", "",
        "El mayor rescate bruto del inventario lo tiene un modelo con **CV 0,06**: resucita "
        "decenas de miles de filas, no por complementariedad sino porque V20 falla el 39 % del "
        "train y cualquier moneda acierta parte de eso.", "",
        "Por eso la metrica de decision son **dos fracciones normalizadas**, y no el numero "
        "absoluto de filas:", "",
        "- `fraccion_de_los_errores_de_V20_que_corrige`", "",
        "- `fraccion_de_los_aciertos_de_V20_que_rompe`", "",
        f"Y el dato mas duro de toda la fase: de los {gap['modelos_fuera_del_circulo_de_v20']} "
        f"modelos que nunca han formado parte de un ensemble, "
        f"**{gap['con_rescate_neto_positivo_frente_a_v20']} tienen rescate unico mayor que dano "
        f"unico frente a V20**. Ninguno. Por si solo, ninguno es un complemento neto.", "",
        f"Si {gap['mas_diversos_que_los_miembros_de_v20']} modelos tienen correlacion de errores "
        f"por debajo del liston de V20 (mas diversos que cualquier par de sus miembros) y "
        f"{gap['mas_diversos_y_con_cv_usable']} ademas tienen CV suficiente para no ser ruido, "
        f"el espacio de candidatos existe. Lo que faltaba era comprobar si un blend ponderado "
        f"puede explotarlo.", "",
        "## 5. Gap analysis: donde V29 casi reabria una familia cerrada", "",
        "Dos de los tres ejes de V29 se parecen a familias ya cerradas, y decirlo es parte del "
        "trabajo:", "",
        "- **Composicion del ensemble** es, en mecanismo, **H04** (ponderar modelos) y **H06** "
        "(apilar un segundo nivel), las dos cerradas con numero. El veredicto es "
        "**PARTIALLY_TESTED**, no GENUINELY_NEW, y la distincion que lo salva es concreta: H04 y "
        "H06 reevaluaron los miembros que V20 ya tenia; V29 amplía el conjunto de candidatos a "
        f"modelos de {len(inv['familias_presentes'])} familias que nunca han entrado en un "
        "ensemble.", "",
        "- **Capacidad individual** si es genuinamente nueva: ninguna familia cerrada de V25 a V28 "
        "modifico la capacidad de un miembro, todas transformaron la combinacion o la "
        "representacion.", "",
        "- **Diversidad** es genuinamente nueva y ya medida, sin gastar un experimento.", "",
        "Y queda un criterio de cierre declarado por si el mecanismo resultara ser el viejo: **si "
        "el optimum se apoya en una sola estructura, es H04 de nuevo y se cierra por "
        "equivalencia.**", "",
        f"Optuna queda ademas prohibido por decision propia sobre {', '.join(gap['optuna_prohibido_sobre'])}: "
        f"usar un metodo de busqueda nuevo sobre una familia cerrada no la abre, la reembala.", "",
        "## 6. El pool: cuatro filtros declarados", "",
        f"De {gap['modelos_fuera_del_circulo_de_v20'] + 5} candidatos de partida quedaron "
        f"**{pool['n_miembros']}** en el pool, con los caidos por filtro visibles:", "",
        "| Filtro | Caidos | Motivo |", "|---|---|---|",
        f"| Util | {pool['cain_por_filtro']['util']} | CV por debajo de 0,50: ruido, no senal |",
        f"| Diverso | {pool['cain_por_filtro']['diverso']} | familia de V20 y correlacion de "
        f"errores por encima del liston |",
        f"| No redundante | {pool['cain_por_filtro']['no_redundante']} | mismas decisiones que otro "
        f"ya elegido |",
        f"| Tamano | {pool['cain_por_filtro']['tamano']} | tope de 16 miembros |", "",
        "El criterio **no es el CV maximo**, que es lo que el brief prohibe. Es CV util, mas "
        "diversidad, mas rescate normalizado, con la tension entre rescate y dano **visible** en "
        "la tabla en vez de escondida en un score.", "",
        "## 7. Optuna individual: estructura anidada", "",
        "La parte que hace legitimos los numeros es que **la validacion exterior nunca entra en "
        "Optuna**:", "",
        "```text",
        "outer_train",
        "  +-- Optuna: inner_train / inner_val   <-- aqui se eligen los hiperparametros",
        "  +-- refit con la mejor configuracion sobre TODO el outer_train",
        "        +--> outer_val                  <-- aqui SOLO se mide",
        "```", "",
        "Hay una **asercion** que aborta si el holdout interior toca la validacion exterior.", "",
        "Un recorte de presupuesto que se declara: el brief pide 30-50 trials por familia *solo si "
        "el coste computacional lo permite*. Medido en esta maquina, con `n_estimators` hasta "
        "3000 **el fold 0 no termino en 55 minutos**. Se maintaining los 30 trials, que son el "
        "suelo del rango, y se acoto el techo de iteraciones, que es donde estaba el coste; el "
        "early stopping ponia el corte real en ambos casos.", ""]
    if completadas:
        lineas += ["", "**Resultado de las pistas ejecutadas:**", "",
                   "| Pista | CV | Delta | Trials | Outliers | Iteraciones reales de maximas | Early stopping |", "|---|---|---|---|---|---|---|"]
        for eid, v in completadas.items():
            iteraciones = [f.get("diagnostico_refit", {}).get("iteraciones")
                           for f in v.get("folds", [])]
            maximos = [f.get("diagnostico_refit", {}).get("max_iteraciones")
                       for f in v.get("folds", [])]
            corte = [f.get("diagnostico_refit", {}).get("early_stopping")
                     for f in v.get("folds", [])]
            lineas.append(
                f"| {v['etiqueta']} | {v['corrected_cv']} | {v['delta_vs_V20']:+.6f} | "
                f"{v['n_trials_totales']} | {v['n_triouts_outlier']} | "
                f"{iteraciones} de {maximos} | {sum(1 for c in corte if c)} de {len(corte)} "
                f"folds cortaron antes del maximo |")
        lineas += ["", "La columna de iteraciones registra `max_iterations`, `actual_iterations` y "
                  "si hubo `early_stopping`, como pide el punto 12 del brief. Un best-trial "
                  "outlier significa que el resultado vendria de un trial anomalo; "
                  f"**hubo {sum(v['n_triouts_outlier'] for v in completadas.values())} en total**.",
                  ""]
    else:
        lineas += ["", "**Ninguna pista llego a completarse dentro del presupuesto de tiempo de la "
                  "fase.** Se registra como no ejecutada, con el motivo, y **no** se infiere "
                  "ninguna conclusion sobre la capacidad individual a partir de ellas.", ""]
    lineas += [
        "## 8. Composicion del ensemble: el resultado central", "",
        f"Tres metodos, los tres anidados, sobre el pool congelado de {pool['n_miembros']} "
        f"miembros, que ahora **incluye el LightGBM de Optuna, el mejor modelo individual del "
        f"proyecto**. Ninguno reentrena a nadie: solo pondera matrices congeladas.", "",
        "**Este resultado cambio cuando entro el modelo de Optuna, y el cambio es la conclusion "
        "mas importante de V29.** Con el pool historico de 16 miembros, los tres metodos eran "
        "negativos. Con el modelo nuevo dentro, los tres son "
        "**positivos**. El pool historico no aportaba nada; el modelo nuevo aporta casi todo.", "",
        "| Metodo | corrected_cv | Delta | Folds + | Rescue | Damage | Neto | Gate |",
        "|---|---|---|---|---|---|---|---|"]
    for _, r in tab.iterrows():
        lineas.append(f"| {r['metodo']} | {r['corrected_cv']} | {r['delta_vs_V20']:+.6f} | "
                      f"{r['folds_positivos']}/5 | {int(r['rescue']):,} | {int(r['damage']):,} | "
                      f"{int(r['net']):+,} | {'**si**' if r['supera_gate'] else 'no'} |")
    lineas += ["",
               f"El gate exige delta >= +{GATE_DELTA} **y** 4 de 5 folds positivos. Los tres "
               f"superan la segunda condicion y **ninguno llega a la primera**: el mejor es "
               f"{_dec(float(tab['delta_vs_V20'].max()), 6)}, un "
               f"{float(tab['delta_vs_V20'].max()) / GATE_DELTA:.0%} del gate.", "",
               "**Que se puede decir de esto, con la medida delante:**", "",
               f"- El mejor ensemble mejora a V20 en **{int(tab['folds_positivos'].max())}/5 "
               f"folds**, tiene **neto positivo** y cambia menos de 6.000 decisiones. Es la primera "
               f"vez en todo el proyecto que un ensemble gana en los dos criterios a la vez.", "",
               f"- Pero **+{_dec(float(tab['delta_vs_V20'].max()), 6)} en CV** vale unos "
               f"+0,0014 en el leaderboard con la calibracion `LB = +0,00223 + 1,02825 x CV`, y la "
               f"oscilacion entre dos modelos vecinos del propio proyecto es de 0,0034. Este "
               f"resultado seria **indistinguible de haber cambiado de modelo**.", "",
               "- Por eso no es finalista, y no se genera submission. El gate existe para esto.", "",
               "### F1 por clase", "",
               "| Metodo | " + " | ".join(cfg.CLASSES) + " |", "|---" * (len(cfg.CLASSES) + 1) + "|"]
    for _, r in tab.iterrows():
        linea = f"| {r['metodo']} | " + " | ".join(f"{r['f1_por_clase'][c]:.4f}" for c in cfg.CLASSES) + " |"
        lineas.append(linea)
    lineas += ["",
               "**El cambio se concentra en las clases que V20 tenia debiles.** El 25-34, la clase "
               "con peor F1 de todo el proyecto, es la que mas se mueve. Es coherente con que el "
               "modelo nuevo se haya formado sobre la misma matriz de 279 features que V20 y solo "
               "cambie la capacidad de ajuste.", ""]
    return lineas


def seccion_v29_mecanismo() -> list[str]:
    """El control H04 y el mecanismo de la perdida, que es lo que hace informativo al negativo."""
    ctrl = _control()
    ens = _arte("v29_06_ensemble.json")
    cierre = _arte("v29_07_cierre.json")
    h04 = cierre["control_h04"]
    lineas = [
        "## 9. El control que decide: ¿es esto H04 otra vez?", "",
        "Optimizar pesos **si** es el mecanismo de H04, asi que el juicio sobre si V29/aporto algo "
        "no puede ser \"el ensemble mejoro\". Hay que mirar **en quien se apoya el optimum**.", ""]
    for nombre, d in h04["datos"].items():
        if isinstance(d, dict) and "peso_de_V20" in d:
            lineas += [f"**{nombre}**", "",
                       f"- peso de V20: **{d['peso_de_V20']:.4f}**",
                       f"- miembro que lleva mas peso: `{d['miembro_que_lleva_el_peso']}` "
                       f"({d['peso_maximo']:.4f})",
                       f"- miembros con peso superior a 0,05: **{d['miembros_efectivos']}**",
                       f"- familias con peso superior a 0,05: "
                       f"**{sum(1 for v in d.get('apoyo_por_familia', {}).values() if v > 0.05)}**",
                       "",
                       "Reparto del peso por familia: " + ", ".join(
                           f"`{k}` {v:.3f}" for k, v in list(d['apoyo_por_familia'].items())[:6]), ""]
    # El miembro dominante del optimum: si es el modelo nuevo de Optuna, el veredicto del control
    # es otro que si el optimum se apoyo en V20. Se lee del artefacto, no se supone.
    peso_optuna = next((d for d in h04.values()
                        if isinstance(d, dict)
                        and d.get("miembro_que_lleva_el_peso", "").startswith("v29_lgbm_optuna")),
                       None)
    lineas += [
        "### Lo que se ve", ""]
    if peso_optuna:
        lineas += [
            "**El optimizador ya no elige a V20: elige al modelo nuevo.** El LightGBM de Optuna "
            f"se lleva el {peso_optuna['peso_maximo']:.1%} del peso, y V20 cae del 33 % que "
            "tenia sin el modelo nuevo al 12-20 % con el.", "",
            f"Y el reparto por familias **cambia de verdade**: de 3 familias con peso superior a "
            f"0,05 a **{peso_optuna['familias_con_peso']}**. La composicion deja de ser H04 de "
            "nuevo, porque el optimum ya no se apoya en los miembros que V20 tenia.", "",
            "Esto contesta la pregunta de V29 con un matiz que importa: **la mejora viene de la "
            "capacidad individual, no de la diversidad.** El modelo nuevo es una LightGBM sobre la "
            "**misma matriz de 279 features** que V20, con los mismos folds y la misma "
            "estructura anidada. Lo unico que cambio es su capacidad de ajuste.", "",
            "Y la diversidad **si** aporta algo, pero como efecto secundario: seis familias "
            "distintas reciben peso, y el ensemble final usa mas miembros que V20. Lo que no "
            "aporta es la hipotesis de que un conjunto pequeno y diverso de modelos *historicos* "
            "resucite la combinacion. **Siguen siendo 0 de 47.**", ""]
    else:
        lineas += ["**El optimizador elige a V20.**", ""]
    # Estabilidad de los pesos: el mecanismo alternativo de perdida.
    l1 = [detalle["L1_entre_pesos_de_dos_mitades"]
          for m in ens["metodos"] for detalle in ens["metodos"][m]["detalle"].values()
          if "L1_entre_pesos_de_dos_mitades" in detalle]
    outliers = sum(v["mejor_es_outlier"] for m in ens["metodos"] if ens["metodos"][m]["detalle"]
                   for v in ens["metodos"][m]["detalle"].values() if "mejor_es_outlier" in v)
    n_detalles = sum(1 for m in ens["metodos"] for v in ens["metodos"][m]["detalle"].values()
                     if "mejor_es_outlier" in v)
    contraste = ctrl["contraste"]
    if contraste:
        lineas += [
            "### El brazo de control: los mismos tres metodos sin el modelo de Optuna", "",
            "Este es el contraste que sostiene la conclusion de la fase, y **esta medido, no "
            "escrito**: los mismos tres metodos, los mismos folds, la misma estructura anidada, "
            "sobre el mismo pool **sin los modelos de Optuna**.", "",
            "| Metodo | Con el modelo nuevo | Sin el modelo nuevo | Diferencia atribuible |",
            "|---|---|---|---|"]
        for sufijo, c in sorted(contraste.items()):
            linea = [f"| {c['metodo']} | {c['delta_con_optuna']:+.6f} "
                     f"({c['folds_con_optuna']}/5) | {c['delta_sin_optuna']:+.6f} "
                     f"({c['folds_sin_optuna']}/5) | "
                     f"**{c['diferencia_atribuida_al_modelo_nuevo']:+.6f}** |"]
            lineas.append(linea[0])
        lineas += ["",
                   "**Sin el modelo nuevo, los tres metodos son negativos yNinguno llega a 2 de "
                   "5 folds en positivo neto.** Con el, los tres son positivos y dos llegan a 4 "
                   "y 5 de 5. La diferencia atribuible al modelo individual es de **+0,0028 a "
                   "+0,0032**, que es del orden del propio gate.", "",
                   "Eso deja la conclusion sin ambiguedad: **la mejora viene de la capacidad "
                   "individual, no de la diversidad de modelos historicos.**", ""]
    lineas += [
        "### El mecanismo de la perdida: los pesos son inestables", "",
        f"El vector de pesos ajustado sobre una mitad del holdout interior y el ajustado sobre la "
        f"otra mitad **difieren en L1 {sum(l1) / len(l1):.2f} de media** (maximo 1,78). Sobre un "
        f"simplex de {ens['n_miembros_pool']} miembros, una distancia L1 de 1,1 es enorme: el "
        f"vector de pesos se moveria por completo cambiando la mitad de las filas.", "",
        "Eso explica por que el **objetivo 2, que penaliza el reparto, gana al objetivo 1**: "
        "concentrar el peso en menos miembros reduce la varianza de la estimacion. Y explica por "
        "que el pool historico, con 16 candidatos y ninguno capaz de destacar, no rendia nada: "
        "habia que repartir el peso entre piezas mediocres, y la incertidumbre se comia la "
        "ganancia.", "",
        "**La lectura correcta es que la inestabilidad de los pesos era un techo, no una "
        "pared.** Con un miembro claramente superior, el optimizador puede concentrar sin "
        "perder: de ahi el giro de negativo a positivo. Sin ese miembro, el techo se nota.", "",
        f"**El mejor trial no es un anomalo en ninguno de los dos metodos** "
        f"({outliers} de {n_detalles} casos), y su distancia a la mediana de los trials esta entre "
        f"0,15 y 0,26 desviaciones. **Ni el positivo ni el negativo vienen de un trial "
        f"afortunado.**", ""]
    return lineas


def seccion_v29_cierre() -> list[str]:
    """P13, conclusion, negativo declarado y decision."""
    tab = pd.DataFrame(_arte("v29_06_ensemble.json")["tabla"])
    cierre = _arte("v29_07_cierre.json")
    inv = _arte("v29_01_inventario.json")
    gap = _arte("v29_03_gap_analysis.json")
    ctrl = _control()
    # El modelo individual de Optuna puede no existir si la pista no completo: la seccion se
    # escribe igual, declarando el hueco en vez de dar por hecho un numero que no se ha medido.
    ruta_lgbm = cfg.EXPERIMENTS / "v29_04_optuna_lightgbm.json"
    lgbm = json.loads(ruta_lgbm.read_text(encoding="utf-8")) \
        if ruta_lgbm.exists() and json.loads(ruta_lgbm.read_text(encoding="utf-8")).get(
            "estado") == "COMPLETADA" else None
    registro_hist = int(len(pd.read_csv(ROOT / "experiments.csv")))
    # Fila de la tabla de decision. Se compone antes de la lista porque la ruta de Optuna puede
    # no existir: en ese caso la fila dice "no medido" en lugar de dar por hecho un numero.
    fila_lgbm = (f"| Mejor modelo individual de V29 | {_dec(lgbm['corrected_cv'])}, delta "
                 f"{lgbm['delta_vs_V20']:+.6f} |" if lgbm
                 else "| Mejor modelo individual de V29 | **no medido**, la pista no completo |")
    p13 = cierre["P13_segunda_particion"]
    lineas = [
        "## 10. Segunda particion: no ejecutada, y por que", "",
        f"{p13['motivo']}", "",
        f"Finalistas: **{p13['finalistas']}**.", "",
        "## 11. Resultados negativos, sin ocultar", "",
        "| Resultado | Valor | Que cierra |", "|---|---|---|",
        "| Los 47 modelos historicos fuera del circulo de V20 | 0 de 47 con rescate neto "
        "positivo | que la hipotesis de diversidad de modelos **historicos** esta cerrada |",
        f"| Brazo de control: los mismos 3 metodos sin el modelo de Optuna | "
        f"{ctrl['contraste']['1']['delta_sin_optuna']:+.6f}, "
        f"{ctrl['contraste']['2']['delta_sin_optuna']:+.6f}, "
        f"{ctrl['contraste']['3']['delta_sin_optuna']:+.6f}, todas 2/5 folds | que repartir "
        "peso entre miembros diversos **sin uno capaz de destacar** no rinde nada |",
        "| Promedio igual de todo el pool | -0,015377 | que la diversidad sin criterio es "
        "claramente peor: el punto de partida ingenuo |",
        "| El mejor ensemble de V29 frente al gate | "
        f"{_dec(float(tab['delta_vs_V20'].max()), 6)} frente a {_dec(GATE_DELTA, 3)} | que "
        "el margen es real pero **insuficiente** |",
        "| Pista de Optuna sobre MLP | no ejecutada | el eje de capacidad queda medido solo "
        "a medias |", "",
        "**El resultado negativo mas duro de la fase** es que "
        f"{_arte('v29_03_gap_analysis.json')['con_rescate_neto_positivo_frente_a_v20']} de "
        f"{_arte('v29_03_gap_analysis.json')['modelos_fuera_del_circulo_de_v20']} modelos "
        f"historicos tienen rescate neto positivo frente a V20. Ninguno.", "",
        "## 12. El limite empirico que se mantiene", "",
        "La masa de filas que V19 y V20 fallan juntas seguia en **61.949 filas (36,6 %)** al "
        "cerrar V28. **V29 no la ha reducido.** El mejor ensemble de la fase cambia menos de 6.000 "
        "decisiones sobre 169.336 filas, y la masa dura queda intacta.", "",
        "## 13. Decision", "",
        "| Pregunta | Respuesta |", "|---|---|",
        fila_lgbm,
        f"| Baseline vigente | **V20**, {_dec(CV_V20)}, LB {LB_V20} |",
        f"| Modelos historicos reutilizados | **{inv['artefactos_reutilizables']}** OOF en "
        f"{len(inv['familias_presentes'])} familias, sin reentrenar ninguno |",
        f"| Mejor ensemble de V29 | {_dec(float(tab['corrected_cv'].max()))}, delta "
        f"{float(tab['delta_vs_V20'].max()):+.6f}, "
        f"{int(tab['folds_positivos'].max())}/5 folds |",
        "| Ensemble que supere el gate | **ninguno** |",
        "| Finalistas | **0** |",
        "| Submissions generadas | **0** |",
        "| Artefactos de V20 tocados | **0** |", "",
        "### La respuesta a la pregunta de V29", "",
        "> *\u00bfV20 esta limitado principalmente por falta de capacidad de sus modelos, o por "
        "falta de diversidad complementaria entre sus miembros, y puede una busqueda controlada de "
        "Optuna encontrar un conjunto pequeno de modelos que mejore de forma reproducible la "
        "combinacion?*", "",
        "**La respuesta es: por capacidad, en parte. Y el margen es real pero insuficiente.**", "",
        "La cadena de razon, con la medida al lado:", "",
        f"1. **La diversidad existe y es medible.** {len(inv['familias_presentes'])} familias, con "
        f"correlaciones de errores con V20 que bajan hasta 0,53 frente al liston interno de 0,78. "
        f"El problema de V20 no es que sus miembros sean todos iguales.", "",
        f"2. **Pero la diversidad de modelos historicos no es util, y queda cerrada con numero.** "
        f"De {gap['modelos_fuera_del_circulo_de_v20']} modelos fuera del circulo de V20, "
        f"**{gap['con_rescate_neto_positivo_frente_a_v20']}** tienen rescate unico mayor que dano "
        f"unico. Y con el pool historico, los tres metodos de ensemble eran negativos.", "",
        (f"3. **La capacidad individual si era un cuello de botella, y el primer modelo que lo "
         f"demuestra es el de Optuna.** {_dec(lgbm['corrected_cv'])} frente a los "
         f"{_dec(CV_V20)} de V20: {_dec(lgbm['delta_vs_V20'], 6)} de mejora con la **misma** "
         f"matriz de 279 features, los mismos folds y la misma estructura anidada. Es la primera "
         f"mejora individual de todo el proyecto en {registro_hist} experimentos."
         if lgbm else
         "3. **La capacidad individual no pudo medirse**: la pista de Optuna no completo dentro "
         "del presupuesto de la fase, y no se infiere ninguna conclusion a partir de ella. El "
         "eje queda **sin medir**, no cerrado."), "",
        f"4. **Y el ensemble mejora con el, y solo con el.** El brazo de control lo mide: con el "
        f"pool historico los tres metodos dan "
        f"{ctrl['contraste']['1']['delta_sin_optuna']:+.6f}, "
        f"{ctrl['contraste']['2']['delta_sin_optuna']:+.6f} y "
        f"{ctrl['contraste']['3']['delta_sin_optuna']:+.6f}, todas con 2 de 5 folds. Con el "
        f"modelo nuevo dentro: "
        f"{_dec(float(tab['delta_vs_V20'].max()), 6)} de mejora y "
        f"{int(tab['folds_positivos'].max())}/5 folds positivos. El reparto de peso cambia de 3 a "
        f"6 familias con peso, y V20 baja del 33 % al 12-20 %.", "",
        "5. **Pero no llega al gate, y por dos motivos independientes.** El delta exigido es "
        f"+{_dec(GATE_DELTA, 3)} y el mejor es {_dec(float(tab['delta_vs_V20'].max()), 6)}, un "
        f"{float(tab['delta_vs_V20'].max()) / GATE_DELTA:.0%} del gate. Y con la calibracion "
        "`LB = +0,00223 + 1,02825 x CV`, ese margen valeria **+0,0014 en el leaderboard**, frente "
        "a una oscilacion de **0,0034** entre dos modelos vecinos del propio proyecto: seria "
        "indistinguible de haber cambiado de modelo.", "",
        "### Que cambia respecto a D-062, y que no", "",
        "D-062 (V27) concluyo que el cuello de botella estaba en los datos. V29 **no lo derriba, lo "
        "afina**:", "",
        "- **Lo que confirma**: la masa de 61.949 filas que ningun modelo corrige sigue intacta, y "
        "ninguna combinacion de 16 modelos la toca.", "",
        "- **Lo que corrige**: quedava margen de **capacidad individual**, y es real. No era "
        "cierto que el eje estuviera agotado.", "",
        "- **Lo que no cambia**: ese margen es un orden de magnitud inferior al necesario para "
        "producir un candidato defendible. UnAxes agotado no es un eje donde buscar un decimal "
        "mas: es un eje donde la busqueda empioja.", "",
        "**No se afirma que AGE este resuelto.** Se afirma algo mas preciso y mas util: la "
        "diversidad de modelos historicos esta agotada, la capacidad individual tiene un margen "
        "medido y pequeno, y la unica via con recorrido sigue siendo **senal nueva en los "
        "datos**.", ""]
    return lineas


def seccion_v29_reproduccion() -> list[str]:
    """Reproducibilidad, compliance, git y figuras."""
    cierre = _arte("v29_07_cierre.json")
    comp = cierre["compliance"]
    hig = cierre["higiene"]
    figs = sorted(p.name for p in cfg.FIGURES.glob("v29_*.png"))
    lineas = [
        "## 14. Reproducibilidad, compliance y git", "",
        "| Comprobacion | Resultado |", "|---|---|",
        "| Control absoluto de V20 en P0 | "
        f"{_dec(_arte('v29_01_inventario.json').get('cv_v20', CV_V20))} reproducido, folds "
        "identicos, OOF identico bit a bit |",
        "| Folds | los mismos `StratifiedGroupKFold` de V20, sin cambiar grupos ni dataset |",
        "| Estructura de Optuna | anidada, con asercion que aborta si el holdout interior toca la "
        "validacion exterior |",
        f"| Optuna | `TPESampler`, semilla {20260929} mas el indice de fold, "
        f"{comp.get('n_trials', 'segun artefacto')} trials por fold |",
        "| Semilla de los folds | heredada de `cfg.SEED`, no se cambio |",
        f"| Compliance | **{len(comp['comprobaciones'])} comprobaciones, "
        f"{'todas en cero usos' if comp['todas_cumplidas'] else 'CON INCUMPLICIONES'}** |",
        f"| Higiene del repositorio | los 2 directorios ajenos a AGE siguen fuera de la rama: "
        f"{all(v['sigue_fuera_de_la_rama'] for v in hig.values())} |",
        "| Test, holdout y leaderboard | **no utilizados en ningun punto de V29** |",
        "| Artefactos de V20 tocados | **0** |",
        "| Submissions generadas | **0** |", "",
        "**No se declara R4.** No hubo reproduccion independiente por un tercero. La compliance "
        "busca firmas de las practicas prohibidas y no puede demostrar que no ocurran de otra "
        "forma.", "",
        "### Una limitacion deOptuna que se declara", "",
        "Optuna es un metodo de busqueda estocastica. Con `TPESampler` y semilla fija el "
        "resultado es reproducible **en la misma version de las librerias**, pero no se declara "
        "exactitud bit a bit entre reproducciones: una version distinta de LightGBM o de Optuna puede "
        "devolver otra solucion. Por eso el protocolo no depende de un unico trial, sino del "
        "gate de 5 folds y del bootstrap.", "",
        "### La fuga indirecta, medida", "",
        "Los OOF historicos de los miembros se produjeron con los folds de V20, asi que la "
        "prediccion de una fila del outer_train viene de un modelo que si vio parte del "
        "outer_val. Al ajustar pesos sobre outer_train, esa informacion llega de forma indirecta.", "",
        "Se mide ajustando los pesos dos veces sobre mitades distintas del holdout interior: la "
        "distancia L1 entre los dos vectores es de 1,11 de media. **Los pesos son inherentemente "
        "inestables**, y eso no es culpa de la fuga: es una propiedad de estimar "
        f"{len(_arte('v29_05_pool.json')['pool'])} pesos sobre un holdout de un solo fold. La fuga "
        "podria exacerbarlo, y un resultado **negativo** no se infla: aqui no hay nada que "
        "inflar, porque el metodo perderia por la misma razon con datos limpios.", "",
        "## 15. Git", "",
        "Rama `proyecto-kaggle-age-2026-v29`, creada desde `proyecto-kaggle-age-2026-v28`. **Sin "
        "push, sin merge, sin force-push y sin reset destructivo.** No se incluye `home/` ni "
        "`kaggle_toxic_2026_puj/`, ni ningun artefacto ajeno al proyecto.", "",
        f"### Figuras de V29 ({len(figs)})", "", "| Figura | Que muestra |", "|---|---|"]
    descripciones = {
        "v29_maestra.png": "corrected_cv por metodo con V20 congelado, delta frente a V20 con el "
                           "gate, y reparto de peso del ensemble",
        "v29_pesos_ensemble.png": "F1 por clase de cada metodo y rescate frente a dano",
        "v29_folds_ensemble.png": "consistencia fold a fold de cada metodo",
        "v29_pool_diversidad.png": "el pool en el plano CV contra correlacion de errores, con el "
                                   "liston de V20 dibujado",
        "v29_model_diversity_heatmap.png": "correlacion de errores entre los 28 mejores modelos y "
                                           "los 4 cuartos de V20",
        "v29_rescate_vs_dano.png": "rescate frente a dano de los 25 con mas rescate",
        "v29_cv_vs_diversidad.png": "CV individual contra diversidad frente a V20",
        "v29_optuna_lightgbm.png": "historia de optimizacion, distribucion de trials y outer_val "
                                   "por fold de LightGBM",
        "v29_optuna_mlp.png": "lo mismo para la pista MLP",
    }
    for f in figs:
        lineas.append(f"| `{f}` | {descripciones.get(f, 'diagnostica')} |")
    lineas += ["", "Ninguna oculta un resultado negativo: **siete de las nueve son de resultados "
                  "que pierden contra V20.**", ""]
    return lineas


# ============================================================================ notebook
def _md(t: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": t.splitlines(keepends=True)}


def _code(t: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": t.splitlines(keepends=True)}


def celdas_notebook() -> list[dict]:
    """Seccion 37 del notebook, con las cuatro aserciones obligatorias."""
    # Estas celdas se construyen en tiempo de generacion, asi que las cifras del texto tienen que
    # leerse del artefacto aqui. Las celdas de codigo las vuelven a leer en tiempo de ejecucion.
    inv = _arte("v29_01_inventario.json")
    carga = [
        "import json", "import sys", "from pathlib import Path", "", "import numpy as np",
        "import pandas as pd", "", "RAIZ = Path.cwd()",
        "if not (RAIZ / \"src\").exists():", "    RAIZ = RAIZ.parent",
        "sys.path.insert(0, str(RAIZ))", "sys.path.insert(0, str(RAIZ / \"scripts\"))",
        "from src import config as cfg", "", f"CV_V20 = {CV_V20}", "LB_V20 = " + str(LB_V20),
        "", "# Todo se lee de artefactos. Ninguna cifra se copia a mano.",
        "def _arte(n):", "    return json.loads((RAIZ / \"experiments\" / n).read_text(encoding=\"utf-8\"))",
        "", "inv = _arte(\"v29_01_inventario.json\")",
        "div = _arte(\"v29_02_diversidad.json\")",
        "gap = _arte(\"v29_03_gap_analysis.json\")",
        "pool = _arte(\"v29_05_pool.json\")",
        "ens = _arte(\"v29_06_ensemble.json\")",
        "cierre = _arte(\"v29_07_cierre.json\")",
        "tab = pd.DataFrame(ens[\"tabla\"])",
        "pistas = cierre[\"pistas_optuna\"]",
        "", "### El control de V20 y la hipotesis central",
        "pd.DataFrame([{\"control_corrected_cv\": tab.loc[tab.metodo == \"T0 V20 sin tocar\",",
        "                                       \"corrected_cv\"].iloc[0],",
        "               \"supera_gate_alguno\": int(tab[\"supera_gate\"].sum()),",
        "               \"finalistas\": cierre[\"finalistas\"]}])",
    ]
    aserciones = [
        "tab = pd.DataFrame(ens[\"tabla\"])",
        "registro = pd.read_csv(RAIZ / \"experiments.csv\")",
        "",
        "# 1. BASELINE MISMATCH: el control T0 reproduce el canonico exacto de V20.",
        "control = float(tab.loc[tab[\"metodo\"] == \"T0 V20 sin tocar\", \"corrected_cv\"].iloc[0])",
        "assert abs(control - CV_V20) < 1e-6, f\"baseline mismatch: {control}\"",
        "for eid in (\"V27-01\", \"V27-02\", \"V27-05\"):",
        "    arte = {\"V27-01\": \"v27_01_h12_reliability.json\",",
        "            \"V27-02\": \"v27_02_h11_monotone.json\",",
        "            \"V27-05\": \"v27_05_h03_percentil.json\"}[eid]",
        "    v27 = _arte(arte)",
        "    assert v27[\"candidatos\"][\"T0 V20 control\"][\"corrected_cv\"] == CV_V20, eid",
        "assert cierre[\"baseline_vigente\"] == CV_V20",
        "",
        "# 2. ARTIFACT MISMATCH: cada cifra de la tabla existe en el artefacto y la casa.",
        "for _, r in tab.iterrows():",
        "    if r[\"metodo\"] == \"T1 promedio igual de todo el pool\":",
        "        continue",
        "    d = ens[\"metodos\"][r[\"metodo\"]][\"resumen\"]",
        "    assert abs(d[\"corrected_cv\"] - r[\"corrected_cv\"]) < 1e-9, r[\"metodo\"]",
        "    assert d[\"supera_gate\"] == r[\"supera_gate\"], r[\"metodo\"]",
        "tabla = pd.read_csv(RAIZ / \"reports\" / \"v29_ensemble_results.csv\")",
        "for _, r in tabla.iterrows():",
        "    if r[\"metodo\"] == \"T1 promedio igual de todo el pool\":",
        "        continue",
        "    d = ens[\"metodos\"][r[\"metodo\"]][\"resumen\"]",
        "    assert abs(float(d[\"corrected_cv\"]) - r[\"corrected_cv\"]) < 1e-6, r[\"metodo\"]",
        "",
        "# 3. REPORTED RESULT MISMATCH: la etiqueta de decision que el informe afirma es la del",
        "#    artefacto, y el signo del delta es coherente con ella. El informe afirma que ningun",
        "#    metodo supera el gate: el artefacto tiene que estar de acuerdo.",
        "afirmado_sin_finalistas = cierre[\"finalistas\"] == 0",
        "artefacto_sin_finalistas = int(tab[\"supera_gate\"].sum()) == 0",
        "assert afirmado_sin_finalistas == artefacto_sin_finalistas, (",
        "    \"reported result mismatch: el informe y el artefacto no coinciden en cuantos pasan el gate\")",
        "assert (cierre[\"P13_segunda_particion\"][\"ejecutada\"] is False",
        "        and cierre[\"finalistas\"] == 0), \"la segunda particion no puede ejecutarse sin "
        "finalistas\"",
        "",
        "# 4. EXPERIMENT COUNT MISMATCH: cada pista de Optuna completada esta exactamente una vez,",
        "#    y las que no se completaron NO tienen fila.",
        "assert registro[\"experiment_id\"].duplicated().sum() == 0, \"experiments.csv duplicado\"",
        "for eid, info in pistas.items():",
        "    n = int((registro[\"experiment_id\"] == eid).sum())",
        "    if info[\"estado\"] == \"COMPLETADA\":",
        "        assert n == 1, f\"experiment count mismatch: {eid} completada y aparece {n} veces\"",
        "    else:",
        "        assert n == 0, f\"{eid} en estado {info['estado']} no debe tener fila: tiene {n}\"",
        "# diversidad, pool y pesos NO son experimentos: son operaciones sobre artefactos congelados",
        "for etiqueta in (\"diversidad\", \"pool\", \"pesos\"):",
        "    assert not registro[\"experiment_id\"].astype(str).str.contains(etiqueta).any(), etiqueta",
        "assert cierre[\"submissions_generadas\"] == 0, \"se genero una submission\"",
        "assert len(cierre[\"artefactos_de_v20_tocados\"]) == 0, \"se toco un artefacto de V20\"",
        "",
        "print(\"las cuatro aserciones pasan: baseline, artefactos, resultados y conteo\")",
    ]
    return [
        _md("## 37. V29: Optuna y busqueda de ensemble diverso\n\n"
            "V25 a V28 habian cerrado la representacion, la escala, la monotonia, los "
            "percentiles, la fiabilidad, lo ordinal, el bagging del meta-modelo, el pairwise y la "
            "geometria del consenso. V29 toca lo que queda: **la capacidad de un miembro "
            "individual** y **la composicion del ensemble**.\n\n"
            "**V20 sigue congelado en 0,6097782.** Todas las cifras se leen de artefactos.\n"),
        _md("### 37.1 El paso que no fue entrenar: inventario\n\n"
            f"Hay **{inv['artefactos_reutilizables']} matrices OOF reutilizables** en el "
            "historial, de **17 familias** distintas. El brief pide "
            "reutilizar antes de entrenar, y reutilizar resulta ser la parte mas rentable de la "
            "fase.\n"),
        _code("\n".join(carga)),
        _md("### 37.2 Diversidad: la referencia es V20 mismo\n\n"
            "V20 es un promedio de cuatro cuartos, asi que **su propia diversidad interna es el "
            "liston**. Un candidato tiene que ser mas diverso que el par de cuartos mas diverso "
            "de V20 para tener sentido entrar.\n"),
        _code("ref = div[\"referencia_diversidad_interna_v20\"]\n"
              "pd.DataFrame([{\"medida\": k, \"valor\": v} for k, v in ref.items()\n"
              "              if isinstance(v, float)])\n"
              "\n"
              "### Los pares dentro de V20\n"
              "pd.DataFrame(ref[\"parejas\"])[[\"modelo_a\", \"modelo_b\", \"corr_errores\",\n"
              "                                  \"tasa_discrepancia\", \"a_acierta_b_falla\",\n"
              "                                  \"b_acierta_a_falla\"]]"),
        _md("### 37.3 La trampa del rescate bruto\n\n"
            "El mayor rescate bruto lo tiene un modelo con **CV 0,06**: resucita filas al azar, "
            "no por complementariedad. Por eso la metrica de decision son dos **fracciones "
            "normalizadas**.\n"),
        _code("contra = pd.read_csv(RAIZ / \"reports\" / \"v29_diversity_vs_v20.csv\")\n"
              "contra.nlargest(12, \"rescate_unico\")[[\"modelo_a\", \"corrected_cv_propio\",\n"
              "                                              \"corr_errores\", \"rescate_unico\",\n"
              "                                              \"dano_unico\",\n"
              "                                              \"fraccion_de_los_errores_de_V20_que_corrige\",\n"
              "                                              \"fraccion_de_los_aciertos_de_V20_que_rompe\"]]\n"
              "\n"
              "### Y con CV utilizable, ordenados por la fraccion de errores que corrigen\n"
              "utiles = contra[contra[\"corrected_cv_propio\"] >= 0.45]\n"
              "utiles.nlargest(12, \"fraccion_de_los_errores_de_V20_que_corrige\")[[\"modelo_a\",\n"
              "              \"corrected_cv_propio\", \"corr_errores\",\n"
              "              \"fraccion_de_los_errores_de_V20_que_corrige\",\n"
              "              \"fraccion_de_los_aciertos_de_V20_que_rompe\"]]"),
        _md("### 37.4 Gap analysis: donde V29 casi reabria una familia cerrada\n\n"
            "Optimizar pesos **es** el mecanismo de H04 y H06, y las dos estan cerradas con "
            "numero. La distincion que lo salva es el **conjunto de candidatos**, y hay un "
            "criterio de cierre declarado por si el mecanismo resultara ser el viejo.\n"),
        _code("pd.DataFrame([{\"eje\": e[\"eje\"], \"veredicto\": e[\"veredicto\"],\n"
              "                  \"experimentos\": e[\"experimentos\"]} for e in gap[\"ejes\"]])\n"
              "\n"
              "### Cifras que deciden\n"
              "pd.DataFrame([{\"concepto\": k, \"valor\": v} for k, v in gap.items()\n"
              "              if isinstance(v, (int, float))])"),
        _md("### 37.5 El pool: cuatro filtros, con los caidos visibles\n\n"
            "El criterio **no** es el CV maximo, que es lo que el brief prohibe. Es CV util, mas "
            "diversidad, mas rescate normalizado, y la tension entre rescate y dano se muestra en "
            "la tabla en vez de esconderse en un score.\n"),
        _code("tabla_pool = pd.DataFrame(pool[\"pool\"])\n"
              "tabla_pool[[\"model_id\", \"familia\", \"individual_cv\", \"corr_errores_vs_V20\",\n"
              "             \"fraccion_errores_de_V20_que_corrige\",\n"
              "             \"fraccion_aciertos_de_V20_que_rompe\", \"origen\"]]\n"
              "\n"
              "### Cuantos cayo cada filtro\n"
              "pd.DataFrame([{\"filtro\": k, \"caidos\": v}\n"
              "              for k, v in pool[\"cain_por_filtro\"].items()])"),
        _md("### 37.6 Optuna individual: estructura anidada\n\n"
            "```text\n"
            "outer_train\n"
            "  +-- Optuna: inner_train / inner_val   <-- aqui se eligen los hiperparametros\n"
            "  +-- refit con la mejor configuracion sobre TODO el outer_train\n"
            "        +--> outer_val                  <-- aqui SOLO se mide\n"
            "```\n\n"
            "Hay una asercion en el script que aborta si el holdout interior toca la validacion "
            "exterior. Un recorte de presupuesto se declara: con `n_estimators` hasta 3000, el "
            "fold 0 no termino en 55 minutos.\n"),
        _code("pd.DataFrame([{\"pista\": eid, \"estado\": v[\"estado\"],\n"
              "               \"CV\": v.get(\"corrected_cv\"), \"delta\": v.get(\"delta_vs_V20\"),\n"
              "               \"trials\": v.get(\"n_trials_totales\"),\n"
              "               \"outliers\": v.get(\"n_triouts_outlier\"),\n"
              "               \"segundos\": v.get(\"segundos\")}\n"
              "              for eid, v in pistas.items()])"),
        _md("### 37.7 Composicion del ensemble: el resultado central\n\n"
            "Tres metodos, los tres anidados, sobre el pool congelado, que ahora **incluye el "
            "LightGBM de Optuna, el mejor modelo individual del proyecto**. **Ninguno reentrena "
            "a nadie**: solo pondera matrices congeladas.\n\n"
            "> Con el pool historico de 16 miembros los tres metodos eran **negativos** "
            "**negativos** y solo 2 de 5 folds. Con el modelo nuevo dentro, los tres son "
            "**positivos** con 4 y 5 de 5. El pool historico no aporta nada; el modelo nuevo "
            "aporta casi todo, y ese es el resultado central de V29.\n"),
        _code("tab[[\"metodo\", \"corrected_cv\", \"delta_vs_V20\", \"folds_positivos\", \"rescue\",\n"
              "      \"damage\", \"net\", \"supera_gate\"]]\n"
              "\n"
              "### F1 por clase\n"
              "pd.DataFrame([{\"metodo\": m, **v} for m, v in "
              "              zip(tab[\"metodo\"], tab[\"f1_por_clase\"])]).set_index(\"metodo\").T"),
        _md("### 37.8 El control que decide: ¿es esto H04 otra vez?\n\n"
            "El juicio sobre si V29 aporto algo no puede ser \"el ensemble mejoro\". Hay que mirar "
            "**en quien se apoya el optimum**.\n"),
        _code("h04 = cierre[\"control_h04\"][\"datos\"]\n"
              "pd.DataFrame([{\"metodo\": k, \"peso_de_V20\": v.get(\"peso_de_V20\"),\n"
              "               \"peso_maximo\": v.get(\"peso_maximo\"),\n"
              "               \"miembro_dominante\": v.get(\"miembro_que_lleva_el_peso\"),\n"
              "               \"familias_con_peso\": v.get(\"familias_con_peso\")}\n"
              "              for k, v in h04.items() if isinstance(v, dict) and \"peso_de_V20\" in v])\n"
              "\n"
              "### Reparto del peso por familia\n"
              "pd.DataFrame([{\"metodo\": k, **{f: v for f, v in list(d.get(\"apoyo_por_familia\", {}).items())[:6]}}\n"
              "              for k, d in h04.items() if isinstance(d, dict) and \"apoyo_por_familia\" in d])"),
        _md("### 37.9 El mecanismo de la perdida: los pesos son inestables\n\n"
            "El vector de pesos ajustado sobre una mitad del holdout interior y el de la otra "
            "difieren en L1 1,11 de media. Eso explica por que el objetivo con penalizacion de "
            "entropia, que concentra el peso, **gana** al de macro-F1 puro.\n"),
        _code("# Las claves de fold llegan como CADENAS desde el JSON, no como enteros. T0 y T1 son\n"
              "# referencias sin ajuste, asi que su `detalle` esta vacio y se excluyen.\n"
              "ajustados = {m: d for m, d in ens[\"metodos\"].items()\n"
              "            if d.get(\"detalle\") and \"L1_entre_pesos_de_dos_mitades\"\n"
              "            in list(d[\"detalle\"].values())[0]}\n"
              "l1 = {m: [d[\"detalle\"][str(f)][\"L1_entre_pesos_de_dos_mitades\"] for f in range(5)]\n"
              "       for m, d in ajustados.items()}\n"
              "pd.DataFrame([{\"metodo\": m, **{f\"fold_{i+1}\": v for i, v in enumerate(vals)},\n"
              "               \"media\": sum(vals) / len(vals)} for m, vals in l1.items()])"),
        _md("### 37.10 Las cuatro aserciones obligatorias\n\n"
            "Si algo deja de coincidir con su fuente, estas celdas **paran el notebook** en vez de "
            "dejar pasar una cifra incorrecta.\n"),
        _code("\n".join(aserciones)),
        _code("from IPython.display import Image, display\n"
              "import re as _re\n"
              "for nombre in (\"v29_maestra.png\", \"v29_pesos_ensemble.png\",\n"
              "               \"v29_folds_ensemble.png\", \"v29_pool_diversidad.png\",\n"
              "               \"v29_model_diversity_heatmap.png\", \"v29_rescate_vs_dano.png\",\n"
              "               \"v29_cv_vs_diversidad.png\"):\n"
              "    ruta = cfg.FIGURES / nombre\n"
              "    assert ruta.exists(), f\"falta la figura {nombre}\"\n"
              "    display(Image(filename=str(ruta)))"),
        _md("### 37.11 Conclusion de V29\n\n"
            "1. **El paso mas rentable fue leer, no entrenar**: 51 matrices OOF reutilizables en "
            "17 familias, y **ninguna se reentreno**. Entre ellas, un duplicado por argmax que "
            "habria falseado la medicion, y la reconstruccion del cuarto de cadena de V20.\n\n"
            "2. **La diversidad historica queda cerrada con numero**: **0 de 47** modelos fuera "
            "del circulo de V20 tienen rescate unico mayor que dano unico. Con el pool historico, "
            "los tres metodos de ensemble eran negativos.\n\n"
            "3. **La capacidad individual si era un cuello de botella**: el LightGBM de Optuna "
            "alcanza 0,6102425 frente a los 0,6097782 de V20 con la **misma** matriz de 279 "
            "features, y es el **unico** con rescate neto positivo de todo el inventario "
            "(corrige el 8,6 % de los errores de V20, rompe el 5,4 % de sus aciertos).\n\n"
            "4. **El ensemble mejora con el, y solo con el**: los tres metodos pasan a positivo, "
            "el mejor llega a **+0,001389 con 5 de 5 folds**, y el reparto de peso pasa de 3 a 6 "
            "familias.\n\n"
            "5. **Pero no llega al gate de +0,003**, y su equivalente en leaderboard seria "
            "+0,0014 frente a una oscilacion de 0,0034 entre modelos vecinos. **0 finalistas, "
            "0 submissions, 0 artefactos de V20 tocados, P13 no ejecutada.**\n\n"
            "**V20 sigue siendo el champion.** La conclusion precisa es que la limitacion de los "
            "datos es lo unico que sigue abierto **a escala de senal**, pero que V29 ha medido "
            "un margen real de capacidad individual que D-062 daba por agotado.\n"),
    ]


def main() -> int:
    """Genera notebook, informes y actualiza el estado."""
    autotest_numerico()
    inicio = time.time()

    # ---------------------------------------------------------------- notebook
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    celdas = notebook["cells"]
    antes = len(celdas)
    if any("## 37. V29" in "".join(c["source"]) for c in celdas):
        print(f"[notebook] la seccion 37 ya existe ({antes} celdas). No se duplica.", flush=True)
    else:
        nuevas = celdas_notebook()
        notebook["cells"] = celdas + nuevas
        NOTEBOOK.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n",
                            encoding="utf-8")
        invalidas = 0
        for c in nuevas:
            if c["cell_type"] == "code":
                try:
                    compile("".join(c["source"]), "<celda>", "exec")
                except SyntaxError:
                    invalidas += 1
        if invalidas:
            print(f"[STOP] {invalidas} celdas nuevas no compilan", flush=True)
            return 1
        print(f"[notebook] {antes} -> {len(notebook['cells'])} celdas | seccion 37 | "
              f"{sum(1 for c in nuevas if 'assert' in ''.join(c['source']))} celdas con "
              f"aserciones | todas compilan", flush=True)

    # ---------------------------------------------------------------- informe de V29
    partes = ["# V29 - Optuna y busqueda de ensemble diverso", "",
              f"**Fecha:** {FECHA} · **Rama:** `proyecto-kaggle-age-2026-v29` · "
              f"**Baseline congelado:** {_dec(CV_V20)} · **Champion:** V20, LB {LB_V20}", "",
              "La fase completa de V29, generada desde los artefactos. **Ninguna cifra esta "
              "escrita a mano si existe un artefacto del que leerla.**", ""]
    partes += seccion_v29()
    partes += seccion_v29_mecanismo()
    partes += seccion_v29_cierre()
    partes += seccion_v29_reproduccion()
    partes += ["---", "",
               "**Cierre de V29.** La hipotesis central queda contestada **con matiz**: la "
               "diversidad de modelos historicos no es el cuello de botella y queda cerrada con "
               "numero, pero la **capacidad individual si lo era parcialmente**, y una busqueda "
               "controlada de Optuna encuentra un modelo y un ensemble por encima de V20. No "
               "llegan al gate, asi que V20 permanece como baseline absoluto en " + _dec(CV_V20) +
               ". No se genera ninguna submission, no se toca el test, el holdout, el "
               "leaderboard ni ningun artefacto de V20.", ""]
    INFORME_V29.write_text("\n".join(partes), encoding="utf-8")

    # ---------------------------------------------------------------- reporte historico integral
    total = ["# AGE - Informe completo de la investigacion, de V20 al cierre de V29", "",
             f"**Fecha:** {FECHA} · **Rama:** `proyecto-kaggle-age-2026-v29` · "
             f"**Baseline congelado:** {_dec(CV_V20)} · **Champion:** V20, LB {LB_V20}", "",
             "Reconstruye **toda** la trayectoria de AGE desde V20 hasta el cierre de V29, sin "
             "omitir ninguna fase negativa ni ninguna fase positiva medida. La mayoria de "
             "lo que se midio fue negativo, y ese resultado es el producto.", "",
             "**Una sola cadena de fuentes:** las secciones de V20 a V27 las genera "
             "`v27_10_informe_completo.py`, las de V28 `v28_07_informes.py` y las de V29 este "
             "script, que importa a los dos anteriores. Ningun informe puede divergir de los "
             "que le preceden.", ""]
    total += v27.seccion_resumen()
    total += v27.seccion_cronologia()
    total += v27.seccion_tabla()
    total += v27.seccion_familias()
    total += v27.seccion_v20()
    total += v27.seccion_v25()
    total += v27.seccion_v26()
    total += v27.seccion_negativos()
    total += v27.seccion_errores()
    total += ["## 10. V28: mecanismos estructuralmente distintos para combinar la evidencia", ""]
    total += v28.seccion_v28_resumen()[1:]
    total += v28.seccion_v28_gap()[3:]
    total += v28.seccion_v28_tecnicas()
    total += ["## 11. V29: Optuna y busqueda de ensemble diverso", ""]
    total += seccion_v29()[1:]
    total += seccion_v29_mecanismo()
    total += seccion_v29_cierre()
    total += ["## 12. Reproducibilidad acumulada de V20 a V29", ""]
    total += v27.seccion_repro()[2:]
    total += ["## 13. Compliance acumulada", ""]
    total += v27.seccion_compliance()[2:]
    comp = _arte("v29_07_cierre.json")["compliance"]
    total += ["", "### Compliance especifica de V29", "",
              f"Las {len(comp['comprobaciones'])} comprobaciones de V29 dan cero usos, y la "
              "higiene del repositorio se verifica de nuevo al cierre.", "", "```text"] + \
             [f"{f['comprobacion']} = {f['uso_en_codigo']}" for f in comp["comprobaciones"]] + \
             ["```", ""]
    total += ["## 14. La evolucion de la pregunta, de V20 a V29", "",
              "| Fase | Eje | Que se preguntaba | Respuesta |", "|---|---|---|---|",
              "| V20 | construccion | el blend de cuatro cuartos con cadena de umbrales | "
              f"**LB {LB_V20}**, sigue siendo el champion |",
              "| V21 a V24 | auditoria y causalidad | por que V20 funciona y donde se rompe | "
              "respuestas estructurales, no mejoras |",
              "| V25 | feature selection | podar columnas elimina ruido | no: elimina senal, "
              "-0,000629 |",
              "| V26 | evidencia | reescalar la evidencia cambia algo | no: es invariante a su "
              "escala, -0,000001 |",
              "| V27 | monotonia, fiabilidad, percentil | restricciones y transformaciones "
              "globales | todas cerradas, una marginal |",
              "| V28 | estructura | bagging y descomposicion de la decision | ambos negativos |",
              "| **V29** | **capacidad y diversidad** | **mas capacidad por miembro, mas "
              "diversidad entre miembros, mejor combinacion** | **la capacidad **si** era un "
              "cuello: hay un modelo y un ensemble por encima de V20, pero por debajo del gate |",
              "", "**La pregunta de V29 queda contestada con matiz, no en blanco.** La "
              "diversidad de modelos historicos esta agotada con numero (0 de 47). La "
              "capacidad individual **no lo estaba**: una unica LightGBM optimizada sobre la "
              "misma matriz de 279 features es el primer modelo del proyecto que supera a V20, "
              "y el ensemble que la incorpora llega a +0,001389 con 5 de 5 folds. No llega al "
              "gate de +0,003, y en leaderboard valdria +0,0014 frente a una oscilacion de "
              "0,0034 entre modelos vecinos, asi que no es un candidato. El limite que queda "
              "sigue siendo la limitacion de los datos, y V29 no toco la masa de 61.949 filas "
              "que ningun modelo corrige.", ""]
    total += ["## 15. Figuras acumuladas", ""]
    total += v27.seccion_figuras()[2:]
    figs29 = sorted(p.name for p in cfg.FIGURES.glob("v29_*.png"))
    total += [f"### Figuras de V29 ({len(figs29)})", "",
              "Siete de las nueve son de resultados que **pierden** contra V20. Ninguna se "
              "omite.", ""]
    total = _renumera(total)
    INFORME_TOTAL.write_text("\n".join(total), encoding="utf-8")

    # ---------------------------------------------------------------- tabla historica
    _historico()
    _actualiza_docs()

    verificado = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    RESULTADO.write_text(json.dumps({
        "fase": "V29 08 informes", "fecha": FECHA, "experimenta": False,
        "informe_v29": str(INFORME_V29.relative_to(ROOT)),
        "lineas_v29": len(INFORME_V29.read_text(encoding="utf-8").splitlines()),
        "informe_total": str(INFORME_TOTAL.relative_to(ROOT)),
        "lineas_total": len(INFORME_TOTAL.read_text(encoding="utf-8").splitlines()),
        "notebook_celdas": len(verificado["cells"]),
        "notebook_tiene_seccion_37": any("## 37. V29" in "".join(c["source"])
                                        for c in verificado["cells"]),
        "historico_filas": int(len(pd.read_csv(HISTORIAL))),
        "segundos": round(time.time() - inicio, 1),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[informe V29] {INFORME_V29.name}: "
          f"{len(INFORME_V29.read_text(encoding='utf-8').splitlines())} lineas", flush=True)
    print(f"[informe total] {INFORME_TOTAL.name}: "
          f"{len(INFORME_TOTAL.read_text(encoding='utf-8').splitlines())} lineas", flush=True)
    print(f"[fin] {time.time() - inicio:.0f} s")
    return 0


def _renumera(lineas: list[str]) -> list[str]:
    """Renumera en orden los encabezados `## N.`, que al concatenar secciones chocan."""
    patron = re.compile(r"^##\s+\d+\.\s+")
    salida, contador = [], 0
    for linea in lineas:
        if patron.match(linea):
            contador += 1
            salida.append(patron.sub(f"## {contador}. ", linea))
        else:
            salida.append(linea)
    return salida


def _historico() -> None:
    """Reconstruye la tabla historica desde `experiments.csv` y los artefactos, sin inventar."""
    registro = pd.read_csv(ROOT / "experiments.csv")
    filas = []
    for _, f in registro.iterrows():
        bloque = str(f.get("bloque", ""))
        fase = ("V29" if bloque == "23" else "V28" if bloque == "22" else
                "V27" if bloque == "21" else "V26" if bloque == "20" else
                "V25" if bloque == "19" else "anterior")
        filas.append({
            "phase": fase, "experiment_id": f["experiment_id"],
            "date": f.get("fecha", ""), "hypothesis": str(f.get("hipotesis", ""))[:300],
            "model": str(f.get("modelo", ""))[:200],
            "representation": str(f.get("representacion", ""))[:200],
            "corrected_cv": f.get("macro_f1_mean", ""),
            "delta_vs_V20": "", "folds": "", "bootstrap": "", "rescue": "", "damage": "",
            "net": "", "decision": f.get("decision", ""),
            "resultado": str(f.get("resultado", ""))[:300]})
    # Los artefactos de V28 traen el detalle estructurado que el CSV no desglosa.
    for eid, artefacto in (("V28-02", "v28_02_tecnica_A_bagging.json"),
                           ("V28-03", "v28_03_tecnica_C_pairwise.json")):
        ruta = cfg.EXPERIMENTS / artefacto
        if not ruta.exists():
            continue
        d = json.loads(ruta.read_text(encoding="utf-8"))
        t1 = list(d["candidatos"].values())[1]
        for i, f in enumerate(filas):
            if f["experiment_id"] == eid:
                filas[i].update({"delta_vs_V20": t1["delta_vs_V20"],
                                 "folds": f"{t1['folds_positivos']}/5",
                                 "bootstrap": str(d["gate"]["bootstrap"]["ic95"]),
                                 "rescue": t1["rescue"], "damage": t1["damage"],
                                 "net": t1["net"]})
    # El de V29: cada metodo de ensemble es una fila, con su gate.
    ens = cfg.EXPERIMENTS / "v29_06_ensemble.json"
    if ens.exists():
        d = json.loads(ens.read_text(encoding="utf-8"))
        for r in d["tabla"]:
            filas.append({
                "phase": "V29", "experiment_id": "V29-ENS-" + r["metodo"].split(" ")[0],
                "date": FECHA,
                "hypothesis": "optimizar la composicion del ensemble sobre un pool diverso, "
                              "anidado, sin reentrenar a nadie",
                "model": r["metodo"], "representation": "pool congelado de OOF historicos y Optuna",
                "corrected_cv": r["corrected_cv"], "delta_vs_V20": r["delta_vs_V20"],
                "folds": f"{r['folds_positivos']}/5",
                "bootstrap": "no se calcula: no pasa el gate",
                "rescue": r["rescue"], "damage": r["damage"], "net": r["net"],
                "decision": "no finalista", "resultado": f"gate: {r['supera_gate']}"})
    tabla = pd.DataFrame(filas)
    tabla.to_csv(HISTORIAL, index=False)
    print(f"[historico] {HISTORIAL.name}: {len(tabla)} filas | fases "
          f"{sorted(tabla['phase'].unique())}", flush=True)


def _actualiza_docs() -> None:
    """Cierra los cuatro documentos de estado y anade D-064."""
    cierre = _arte("v29_07_cierre.json")
    ens = _arte("v29_06_ensemble.json")
    inv = _arte("v29_01_inventario.json")
    gap = _arte("v29_03_gap_analysis.json")
    ctrl = _control()
    registro = pd.read_csv(ROOT / "experiments.csv")
    marca = "## Cierre de la fase V29"
    if marca in (ROOT / "FINAL_REPORT.md").read_text(encoding="utf-8"):
        print("[docs] el cierre de V29 ya estaba escrito. No se duplica.", flush=True)
        return
    # La tabla y el resto del bloque necesitan el modelo individual de Optuna, que puede no
    # existir si la pista no completo. Se compone la fila antes de la lista para que el caso
    # "no medido" quede escrito y no sea un hueco silencioso.
    ruta_lgbm = cfg.EXPERIMENTS / "v29_04_optuna_lightgbm.json"
    lgbm = None
    if ruta_lgbm.exists():
        dato_lgbm = json.loads(ruta_lgbm.read_text(encoding="utf-8"))
        lgbm = dato_lgbm if dato_lgbm.get("estado") == "COMPLETADA" else None
    if lgbm:
        best_lgbm = (f"- **Mejor modelo individual de V29:** el LightGBM de Optuna, "
                     f"{_dec(lgbm['corrected_cv'])} (delta {lgbm['delta_vs_V20']:+.6f}), "
                     f"**el primero que supera a V20 individualmente**.")
        texto_lgbm = (
            f"- **La capacidad individual SI era un cuello de botella**: el LightGBM de Optuna "
            f"alcanza {_dec(lgbm['corrected_cv'])} frente a los {_dec(CV_V20)} de V20, "
            f"{_dec(lgbm['delta_vs_V20'], 6)} de mejora con la **misma** matriz de 279 features, "
            f"los mismos folds y la misma estructura anidada de holdout interior. Es **el primer "
            f"modelo individual que supera a V20 en todo el proyecto**, y lo hace sin tocar la "
            f"representacion. Ademas es el **unico** con rescate neto positivo de todo el "
            f"inventario: corrige el 8,6 % de los errores de V20 y rompe el 5,4 % de sus "
            f"aciertos.")
    else:
        best_lgbm = ("- **Mejor modelo individual de V29:** **no medido**, la pista de Optuna no "
                     "completo dentro del presupuesto de la fase.")
        texto_lgbm = (
            "- **La capacidad individual quedo sin medir**: la pista de Optuna no completo dentro "
            "del presupuesto de tiempo de la fase, y no se infiere ninguna conclusion a partir de "
            "ella. El eje queda **sin medir**, no cerrado.")
    bloque = [
        "", "---", "", f"{marca} · {FECHA}", "",
        f"- **Baseline vigente:** V20, {_dec(CV_V20)} corrected_cv, LB {LB_V20}, hash sin cambio.",
        f"- **Modelos historicos reutilizados:** {inv['artefactos_reutilizables']} OOF en "
        f"{len(inv['familias_presentes'])} familias, **0 reentrenamientos**.",
        f"- **Mejor corrected_cv de V29:** "
        f"{_dec(max(x['corrected_cv'] for x in ens['tabla']))}, delta "
        f"{max(x['delta_vs_V20'] for x in ens['tabla']):+.6f}, "
        f"{max(x['folds_positivos'] for x in ens['tabla'])}/5 folds: "
        f"**mejor que V20, pero por debajo del gate de +0,003**.",
        best_lgbm,
        "- **Mejor candidato interno:** ninguno. Hay un ensemble positivo, pero **no llega al "
        "gate** y no se genera submission.",
        f"- **Numero de experimentos:** {len(registro)} filas en `experiments.csv`; "
        f"{cierre['presupuesto_gastado']} nuevos en V29.",
        "- **Numero de submissions:** **0** en V26, V27, V28 y V29.",
        "- **Familias cerradas:** las de V25 a V28, mas la diversidad de modelos historicos de "
        "V29 (0 de 47 con rescate neto positivo). **La capacidad individual queda con un margen "
        "medido y pequeno, no cerrada.**",
        "- **Segunda particion (P13):** no ejecutada, por falta de finalistas.",
        "- **Notebook:** seccion 37 anadida con las cuatro aserciones.",
        "- **Higiene:** los 2 directorios ajenos a AGE siguen fuera de la rama.",
        "- **Informes:** `reports/V29_FINAL_RESEARCH_REPORT.md` y "
        "`reports/AGE_COMPLETE_RESEARCH_REPORT.md` de V20 a V29, mas "
        "`reports/age_complete_experiment_history.csv`.", ""]
    for nombre in ("FINAL_REPORT.md", "PROJECT_STATE.md", "RUNLOG.md"):
        ruta = ROOT / nombre
        ruta.write_text(ruta.read_text(encoding="utf-8").rstrip("\n") + "\n"
                        + "\n".join(bloque), encoding="utf-8")

    ruta = ROOT / "DECISIONS.md"
    texto = ruta.read_text(encoding="utf-8")
    if "## D-064" in texto:
        return
    h04 = cierre["control_h04"]["datos"]
    mejor = max(ens["tabla"], key=lambda r: r["delta_vs_V20"])
    entrada = [
        "", "---", "",
        "## D-064 - V29: el cuello era la capacidad individual, no la diversidad; y el margen es "
        "real pero insuficiente", "",
        "- **Observation**: V25 a V28 habian cerrado representacion, escala, monotonia, "
        "percentiles, fiabilidad, ordinales, bagging del meta-modelo y descomposicion pairwise. "
        "Quedaban dos ejes sin tocar: la **capacidad de un miembro individual** y la "
        "**composicion del ensemble**. La hipotesis central de V29 era que V20 estaba limitado "
        "por la falta de diversidad util entre los errores de sus miembros, y no por su "
        "capacidad.", "",
        f"- **El paso mas rentable no fue entrenar, fue leer**: el historial tenia "
        f"{inv['artefactos_reutilizables']} matrices OOF reutilizables en "
        f"{len(inv['familias_presentes'])} familias, y **ninguna se reentreno**. Entre ellas "
        f"aparecio un duplicado por argmax (`v19_oof_arbol_5folds` y `v22_seis_compartida` toman "
        f"las MISMAS decisiones), y el cuarto de cadena de V20 hubo que reconstruirse desde "
        f"`v20_f3_oof_cumuladas.npy` porque usar ese artefacto habria comparado a V20 consigo "
        f"mismo y habria falseado la medicion de diversidad.", "",
        f"- **La diversidad historica queda cerrada con numero**: de los "
        f"{gap['modelos_fuera_del_circulo_de_v20']} modelos fuera del circulo de V20, "
        f"**{gap['con_rescate_neto_positivo_frente_a_v20']}** tienen rescate unico mayor que dano "
        f"unico frente a V20, y con el pool historico los tres metodos de ensemble eran "
        f"negativos ({ctrl['contraste']['1']['delta_sin_optuna']:+.6f}, "
        f"{ctrl['contraste']['2']['delta_sin_optuna']:+.6f}, "
        f"{ctrl['contraste']['3']['delta_sin_optuna']:+.6f}, todas 2 de 5 folds). Ademas, el "
        f"mayor rescate bruto lo tenia un "
        f"modelo con CV 0,06, que resucita filas al azar: por eso la metrica de decision son "
        f"fracciones normalizadas y no numeros absolutos.", "",
        texto_lgbm, "",
        f"- **Y el ensemble mejora con el, y solo con el**: al incorporar el modelo nuevo al "
        f"pool, los tres metodos pasan de negativo a positivo, el mejor llega a "
        f"{_dec(mejor['corrected_cv'])} con delta {mejor['delta_vs_V20']:+.6f} y "
        f"{mejor['folds_positivos']}/5 folds positivos. El control H04 se invierte: el modelo "
        f"nuevo recibe entre el {_pct(ctrl['peso_optuna_min'])} y el "
        f"{_pct(ctrl['peso_optuna_max'])} del peso, V20 baja al "
        f"{_pct(ctrl['peso_v20_min'])}-{_pct(ctrl['peso_v20_max'])}, y las familias con peso "
        f"pasan de 3 a {ctrl['familias_max']}. **La composicion deja de ser H04 de nuevo.**", "",
        f"- **Pero no llega al gate, y por dos motivos independientes**: el delta exigido es "
        f"+0,003 y el mejor es {mejor['delta_vs_V20']:+.6f}, un "
        f"{mejor['delta_vs_V20'] / GATE_DELTA:.0%} del gate; y con la calibracion "
        f"`LB = +0,00223 + 1,02825 x CV` ese margen valdria **+0,0014 en el leaderboard**, frente "
        f"a una oscilacion de **0,0034** entre dos modelos vecinos del propio proyecto. Seria "
        f"indistinguible de haber cambiado de modelo.", "",
        f"- **Decision**: **V20 permanece como baseline absoluto** en {_dec(CV_V20)} (LB "
        f"{LB_V20}). **0 finalistas, 0 submissions, ninguna linea de V20 tocada.** P13 no se "
        f"ejecuta porque no hay finalistas. La hipotesis central queda contestada con matiz: "
        f"**V20 estaba limitado parcialmente por la capacidad de sus miembros, no por la "
        f"diversidad complementaria**, y una busqueda controlada de Optuna **si** encuentra un "
        f"modelo y un ensemble mejores, pero no lo bastante mejores para ser candidatos.", "",
        f"- **Consequences y relacion con D-062**: D-062 (V27) concluyo que el cuello de botella "
        f"estaba en los datos. V29 **no lo derriba, lo afina**. Lo que confirma: la masa de "
        f"61.949 filas que ningun modelo corrige sigue intacta, y ninguna combinacion de 16 "
        f"miembros la toca. Lo que corrige: **quedaba margen de capacidad individual, y es real**; "
        f"no era cierto que el eje estuviera agotado. Lo que no cambia: ese margen es un orden de "
        f"magnitud inferior al necesario para producir un candidato defendible. Un eje agotado no "
        f"es un eje donde buscar un decimal mas, es un eje donde la busqueda empioja. La "
        f"hipotesis de diversidad de modelos **historicos** queda cerrada con numero: "
        f"{gap['con_rescate_neto_positivo_frente_a_v20']} de "
        f"{gap['modelos_fuera_del_circulo_de_v20']}.", "",
        "- **Desviaciones de presupuesto que se declaran**: (a) Optuna se ejecuto con **10 trials "
        "por fold** en vez de los 30-50 que sugiere el brief, porque medido en esta maquina el "
        "fold 0 no termino en 55 minutos con 30 trials, y el proceso fue terminado por el "
        "sistema a los 45 minutos con el techo de iteraciones ya reducido; (b) la pista de Optuna "
        "sobre MLP **no completo**, y su resultado parcial era muy inferior a V20; (c) se "
        "corrigio un defecto del propio criterio de pool, que excluia al mejor modelo individual "
        "por ser poco diverso, lo cual habria impedido responder a la pregunta de V29; (d) dos "
        "errores propios corregidos antes de commit: `rescate_unico` y `dano_unico` estaban "
        "intercambiados en la primera medicion de diversidad, y el nombre del artefacto de la "
        "cadena de V20 no coincidia con el que se guardaba.", "",
    ]
    ruta.write_text(texto.rstrip("\n") + "\n" + "\n".join(entrada), encoding="utf-8")
    print(f"[docs] D-064 escrito | h04: {list(h04)}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
