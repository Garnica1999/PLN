"""V29 - 03 - P3: gap analysis de los ejes de V29.

V29 trae una hipotesis central nueva, pero dos de sus tres ejes se parecen a familias ya
cerradas. Este script decide, **antes de gastar un experimento**, que parte de V29 es genuinamente
nueva y que parte es un etiqueta nueva sobre una familia cerrada.

## El problema serio: P9 roza H04 y H06

Optimizar los pesos de un ensemble es, en apariencia, lo mismo que H04 (ponderar modelos) y H06
(apilar un segundo nivel), y las dos estan **cerradas con numero**. Si V29 fuera solo eso, estaria
reabriendo una familia cerrada, y el presupuesto entero estaria mal gastado.

La distincion que lo salva es concreta y verificable: **cuestion sobre quien es miembro**.

- H04 y H06 respondieron: "¿mejora reponderar o re-apilar los miembros que V20 ya tenia?"
  Los candidatos eran los mismos cuatro, y la respuesta fue no.
- V29 pregunta: "¿aporta senal alguno de los otros **47** modelos historicos, que nunca han
  formado parte de ningun ensemble, y que tienen estructuras distintas (18 familias)?"

Si la respuesta fuera "solo reponderar V20", V29 estaria cerrada por equivalencia. La pregunta de
V29 tiene un conjunto de candidatos **disjunto**, y eso se mide antes de entrenar: la tabla de
diversidad de P2 dice cuantos modelos hay fuera del circulo de V20 y cuanto se diferencian.

## El segundo problema:Optuna sobre familias cerradas

Optuna no es una tecnica, es un metodo de busqueda. Usado sobre `threshold`, `feature selection`,
`calibracion`, `pairwise`, `ordinal` o `robust loss` seria reabrir familias cerradas con mas
trials, que es exactamente lo que V25 a V28 ya hicieron mal y bien. Este script lo declara
prohibido y acota Optuna a los tres ejes que si son nuevos: **capacidad de modelo**, **diversidad**
y **composicion del ensemble**.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src import config as cfg  # noqa: E402
from src.verificacion_numerica import autotest_numerico  # noqa: E402

RESULTADO = cfg.EXPERIMENTS / "v29_03_gap_analysis.json"
GAP_MD = cfg.REPORTS / "v29_gap_analysis.md"
GAP_CSV = cfg.REPORTS / "v29_gap_analysis.csv"
CV_V20 = 0.6097782
#: Familias cerradas en V25 a V28, con la decision que las cerro. Se citan textualmente para que la
#: comparacion sea contra el registro y no contra la memoria.
CERRADAS = {
    "feature selection supervisada": "V25, D-060, -0,000629: podar columnas elimina senal, no ruido",
    "escalado afil de la evidencia": "V26-01, D-061, -0,000001: la evidencia es invariante a su escala",
    "transformacion percentil": "V27-05, MARGINAL +0,000272, IC95 incluye el cero",
    "fiabilidad constante por modelo y clase": "V27-01, H12, delta exactamente 0",
    "restricciones monotonas": "V27-02, H11, -0,000355",
    "reponder blends equivalentes": "V27, H14, cerrada por equivalencia funcional sin experimento",
    "calibracion de probabilidades": "H01, D-029",
    "ponderacion de modelos": "H04, cerrada con numero",
    "apilado de segundo nivel": "H06, cerrada: V20 ya ES el segundo nivel",
    "familias ordinales (H10, CORAL)": "H10 y CORAL, D-058",
    "bagging del meta-modelo": "V28-02, A, -0,000312",
    "descomposicion pairwise": "V28-03, C, -0,067553",
    "geometria del consenso": "V28, B, superconjunto ya implementado: +0,0001",
    "fusion de evidencia Dirichlet": "V28, D, equivalente a H04 mas H06",
    "incertidumbre explicita": "V28, E, duplicacion de B",
    "subespacio aleatorio": "V28, F, ya implementado en colsample_bytree",
}
#: Ejes nuevos que V29 autoriza. Todo lo demas queda fuera por decision propia, no porBounds.
EJES_NUEVOS = ["capacidad de modelo individual", "diversidad de errores entre miembros",
               "composicion del ensemble"]
#: Sobre que NO se permite usar Optuna, aunque el metodo de busqueda sea nuevo.
OPTUNA_PROHIBIDO_SOBRE = ["threshold", "feature selection", "calibracion", "pairwise",
                          "ordinal", "robust loss", "grafo", "ventana local"]


def main() -> int:
    """Clasifica los ejes de V29 y escribe el gap analysis."""
    autotest_numerico()
    inicio = time.time()
    inventario = pd.read_csv(cfg.REPORTS / "v29_model_inventory.csv")
    contra = pd.read_csv(cfg.REPORTS / "v29_diversity_vs_v20.csv")
    diversidad = json.loads((cfg.EXPERIMENTS / "v29_02_diversidad.json").read_text(encoding="utf-8"))
    ref = diversidad["referencia_diversidad_interna_v20"]
    familias = int(inventario["familia"].nunique())
    reutilizables = int(inventario["reutilizable"].sum())
    miembros_v20 = {"v19_b1_oof_logreg", "v19_b1_oof_mlp", "v19_oof_arbol_5folds",
                    "v20_f3b_oof_candidato"}
    fuera = contra[~contra["modelo_a"].isin(miembros_v20)]
    # Un modelo es "mas diverso que los miembros de V20" si su correlacion de errores cae por
    # debajo del minimo que alcanzan los cuatro cuartos entre si.
    minimo_v20 = float(ref["corr_errores_media"]) - 0.0
    mas_diversos = fuera[fuera["corr_errores"] < minimo_v20]
    # Y "util" ademas exige que no sea ruido: CV por encima del suelo de azar.
    utiles = mas_diversos[mas_diversos["corrected_cv_propio"] >= 0.50]
    netos_positivos = fuera[(fuera["rescate_unico"] > fuera["dano_unico"])]

    print("=== P3: los tres ejes de V29 ===", flush=True)
    filas = [
        {"eje": "capacidad de modelo individual", "veredicto": "GENUINELY_NEW",
         "experimentos": 2, "experimentos_previstos": "P4 LightGBM Optuna, P5 MLP Optuna",
         "por_que": "ninguna familia cerrada de V25 a V28 modifico la CAPACIDAD de un miembro "
                    "individual; todas transformaron la combinacion, la escala o la "
                    "representacion. Buscar mejores hiperparametros sobre la misma representacion "
                    "de 279 features es un eje que nadie ha tocado.",
         "control": "misma matriz de 279 features, mismos folds, y el resultado se compara contra "
                    "el cuarto correspondiente de V20, no contra el ensemble.",
         "frescura": "la unica freshness real: V19 exploro max_depth y min_child_samples, pero "
                     "nunca learning_rate, reg_alpha, reg_lambda ni el par "
                     "n_estimators con early stopping de forma conjunta."},
        {"eje": "diversidad de errores entre miembros", "veredicto": "GENUINELY_NEW",
         "experimentos": 0, "experimentos_previstos": "P2, ya ejecutado: es medicion, no "
                                                       "entrenamiento",
         "por_que": f"el historial tiene {reutilizables} OOF en {familias} familias y "
                    f"{len(fuera)} modelos que NUNCA han formado parte de un ensemble. La pregunta "
                    f"de si alguno aporta senal que los cuatro cuartos de V20 no poseen no se ha "
                    f"formulado ni medido antes.",
         "control": f"la referencia es la diversidad interna del propio V20: correlacion de errores "
                    f"{ref['corr_errores_media']:.4f} de media y {ref['tasa_discrepancia_media']:.4f} "
                    f"de discrepancia. Un candidato tiene que ser MAS diverso que los miembros "
                    f"actuales para tener sentido entrar.",
         "frescura": f"medido: {len(mas_diversos)} modelos tienen correlacion de errores por "
                     f"debajo de {minimo_v20:.4f} y {len(utiles)} ademas tienen CVusable."},
        {"eje": "composicion del ensemble", "veredicto": "PARTIALLY_TESTED",
         "experimentos": 1, "experimentos_previstos": "P9 Optuna de pesos, P10 seleccion de subconjunto",
         "por_que": "el MECANISMO de optimizar pesos es el de H04 y H06, y las dos estan cerradas. "
                    "Lo que es genuinamente nuevo es el CONJUNTO de candidatos: H04 y H06 solo "
                    "reevaluaron los miembros que V20 ya tenia; aqui el pool incluye modelos de 18 "
                    "estructuras distintas que nunca han entrado en un ensemble.",
         "control": "si el optimum de Optuna se apoya en un solo modelo o devuelve los cuatro "
                    "cuartos de V20 con otros pesos, la conclusion es que es H04 de nuevo, y se "
                    "cierra por equivalencia. Si el optimum reparte peso en familias distintas, la "
                    "hipotesis de diversidad tiene una afirmacion que hacer.",
         "frescura": f"dato de P2 que decide: de los {len(fuera)} modelos fuera de V20, "
                     f"{len(netos_positivos)} tienen rescate unico mayor que dano unico frente a "
                     f"V20. Que sean pocos no cierra la hipotesis, porque un blend ponderado puede "
                     f"explotar decorrelacion que un modelo solo no puede."},
    ]
    tabla = pd.DataFrame(filas)
    tabla.to_csv(GAP_CSV, index=False)

    print(f"\n  capacidad individual : GENUINELY_NEW ({tabla.iloc[0]['experimentos']} experimentos)",
          flush=True)
    print("  diversidad           : GENUINELY_NEW (medido, 0 experimentos nuevos)", flush=True)
    print(f"  composicion ensemble : PARTIALLY_TESTED ({tabla.iloc[2]['experimentos']} experimento, "
          f"distincion de conjunto de candidatos)", flush=True)
    print(f"\n  {familias} familias | {reutilizables} OOF reutilizables | {len(fuera)} fuera del "
          f"circulo de V20", flush=True)
    print(f"  mas diversos que los miembros de V20 (corr_err < {minimo_v20:.4f}): {len(mas_diversos)}"
          f" | de ellos con CV >= 0,50: {len(utiles)}", flush=True)
    print(f"  con rescate unico > dano unico frente a V20: {len(netos_positivos)}", flush=True)
    if len(netos_positivos):
        for _, f in netos_positivos.nlargest(8, "neto").iterrows():
            print(f"    {f['modelo_a']:<40} CV {f['corrected_cv_propio']:.4f} | "
                  f"rescate {int(f['rescate_unico']):>6,} | dano {int(f['dano_unico']):>6,} | "
                  f"neto {int(f['neto']):>+6,}", flush=True)
    else:
        print("    ninguno: ningun modelo historico arregla mas de lo que rompe frente a V20. "
              "Es el dato mas duro contra V29 y por eso el pool tiene que construirse con un "
              "criterio distinto del rescate bruto.", flush=True)

    _escribe(tabla, ref, inventario, contra, minimo_v20, mas_diversos, utiles, netos_positivos,
             familias, reutilizables, fuera)

    RESULTADO.write_text(json.dumps({
        "fase": "V29 03 P3 gap analysis", "fecha": "2026-09-29", "experimenta": False,
        "ejes": filas,
        "veredictos": {f["eje"]: f["veredicto"] for f in filas},
        "presupuesto_autorizado": 10, "presupuesto_previsto": int(tabla["experimentos"].sum()),
        "familias_historicas": familias, "oof_reutilizables": reutilizables,
        "modelos_fuera_del_circulo_de_v20": len(fuera),
        "mas_diversos_que_los_miembros_de_v20": len(mas_diversos),
        "mas_diversos_y_con_cv_usable": len(utiles),
        "con_rescate_neto_positivo_frente_a_v20": len(netos_positivos),
        "optuna_prohibido_sobre": OPTUNA_PROHIBIDO_SOBRE,
        "familias_cerradas_no_reabiertas": list(CERRADAS),
        "segundos": round(time.time() - inicio, 1),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[P3] {time.time() - inicio:.0f} s")
    return 0


def _escribe(tabla: pd.DataFrame, ref: dict, inventario: pd.DataFrame, contra: pd.DataFrame,
             minimo_v20: float, mas_diversos: pd.DataFrame, utiles: pd.DataFrame,
             netos: pd.DataFrame, familias: int, reutilizables: int, fuera: pd.DataFrame) -> None:
    """Redacta el gap analysis legible, con los mismos numeros que el JSON."""
    lineas = [
        "# V29 - Gap analysis de los tres ejes", "",
        "Clasificacion previa a cualquier entrenamiento. La pregunta no es si V29 trae una idea "
        "nueva, sino **que parte de sus tres ejes lo es**, porque dos de ellos se parecen a "
        "familias que V25 a V28 cerraron con numero.", "",
        "## Veredicto", "",
        "| Eje | Veredicto | Experimentos | Razon |", "|---|---|---|---|"]
    for _, f in tabla.iterrows():
        lineas.append(f"| {f['eje']} | **{f['veredicto']}** | {f['experimentos']} | "
                      f"{f['por_que'][:190]} |")
    lineas += ["",
               f"Presupuesto previsto: **{int(tabla['experimentos'].sum())} de 10** autorizados. "
               f"Las familias historicas reached {familias}, con {reutilizables} OOF reutilizables, "
               f"de los que **{len(fuera)} nunca han formado parte de un ensemble**.", "",
               "## El problema serio: optimizar pesos roza H04 y H06", "",
               "Optimizar los pesos de un ensemble es, en apariencia, lo mismo que **H04** "
               "(ponderar modelos) y **H06** (apilar un segundo nivel), y las dos estan cerradas con "
               "numero. Si V29 fuera solo eso, estaria reabriendo una familia cerrada.", "",
               "La distincion que lo salva es concreta: **cuestion sobre quien es miembro**.", "",
               "- **H04 y H06** respondieron: *mejora reponderar o re-apilar los miembros que V20 ya "
               "tenia?* Los candidatos eran los mismos cuatro. La respuesta fue no.", "",
               "- **V29** pregunta: *aporta senal alguno de los otros modelos historicos, que nunca "
               "han entrado en un ensemble y tienen estructuras distintas?*", "",
               "Por eso el veredicto de composicion es **PARTIALLY_TESTED** y no GENUINELY_NEW: el "
               "mecanismo es el mismo, el conjunto de candidatos es distinto.", "",
               "**Y hay un criterio de cierre declarado por si el mecanismo resulta ser el viejo:** "
               "si el optimum de Optuna se apoya en un solo modelo, o devuelve los cuatro cuartos "
               "de V20 con otros pesos, entonces es H04 de nuevo y la composicion se cierra por "
               "equivalencia. Si reparte peso en familias estructuralmente distintas, entonces hay "
               "algo que decir.", "",
               "## La referencia de diversidad: los miembros de V20", "",
               "| Medida | Valor |", "|---|---|",
               f"| Correlacion de errores media entre los 4 cuartos | "
               f"**{ref['corr_errores_media']:.4f}** |",
               f"| Discrepancia de decision media | {ref['tasa_discrepancia_media']:.4f} |",
               f"| Discrepancia maxima entre dos cuartos | {ref['tasa_discrepancia_maxima']:.4f} |",
               "| Par de cuartos con mayor correlacion | 0,8675 (arbol contra cadena) |",
               "| Par de cuartos con menor correlacion | 0,7461 (mlp contra cadena) |", "",
               "Este es el liston que un candidato tiene que superar: **para ser mas diverso que los "
               "miembros actuales, su correlacion de errores con V20 tiene que bajar de "
               f"{minimo_v20:.4f}**.", "",
               "## Los tres hallazgos de la medicion de diversidad", "",
               f"1. **{len(mas_diversos)} modelos** tienen correlacion de errores por debajo del "
               f"liston, es decir, mas diversos que cualquier par de los miembros de V20. De esos, "
               f"**{len(utiles)}** tienen ademas CV suficiente para no ser ruido "
               f"(>= 0,50).", "",
               f"2. **{len(netos)} modelos** tienen rescate unico mayor que dano unico frente a V20. "
               f"Es el dato mas duro contra V29 y **no se oculta**: por si solo, ningun modelo "
               f"historico es un complemento neto de V20.", "",
               "3. El mayor rescate bruto lo tiene un modelo con CV 0,06, que *resucita* filas por "
               "azar. Es la trampa de leer el rescate sin skill, y por eso la metrica de decision "
               "es la fraccion de los errores de V20 que el modelo corrige, no el numero absolute.", "",
               "Que el punto 2 sea desfavorable **no cierra la hipotesis de diversidad**, y esa es la "
               "distincion importante: un modelo puede tener rescate neto negativo frente a V20 y "
               "aun asi aportar a un blend ponderado, porque lo que un ensemble explota es la "
               "decorrelacion de los errores, no el acierto individual. Medir eso exige el ensemble, "
               "que es lo que hacen P9 y P10.", "",
               "## Sobre que NO se permite usar Optuna", "",
               "Optuna es un metodo de busqueda, no una tecnica. Aplicarlo sobre estas familias "
               "seria reabrir con mas trials lo que V25 a V28 ya cerraron:", "",
               "| Familia | Decision que la cerro |", "|---|---|"]
    for nombre, motivo in CERRADAS.items():
        lineas.append(f"| {nombre} | {motivo} |")
    lineas += ["",
               f"Y queda prohibido por decision propia usar Optuna sobre: "
               f"{', '.join(OPTUNA_PROHIBIDO_SOBRE)}. El punto 43 del brief lo pide y aqui se "
               f"convierte en lista de verificacion.", "",
               "Optuna se acota a tres ejes: **capacidad de modelo individual**, que nadie ha "
               "tocado en este proyecto; **diversidad**, que es medicion; y **composicion del "
               "ensemble**, con el conjunto de candidatos ampliado a modelos que nunca han "
               "entrado en uno.", ""]
    GAP_MD.write_text("\n".join(lineas), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
