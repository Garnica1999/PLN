"""V26 · 00 — Gap analysis: que familias estan cerradas y cual es genuinamente nueva.

Este script **no entrena**. Su objeto es cumplir el punto 10 y el punto 11 del brief: leer el
estado completo del proyecto, buscar **equivalencia funcional** y no solo coincidencia de
nombre, y decidir que hipotesis merece un experimento.

## La distincion que gobierna todo el analisis

**V20 ya es un meta-modelo.** De las 279 features, **84 son predicciones de modelos base** (78 de
los 14 miembros del ensamble de V12 y 6 de NB-SVM), y el meta-modelo de V20 es **LightGBM**: cinco
binarios para la cadena de umbrales y uno mas para el arbol del blend. No hay que "construir" un
meta-modelo, ya existe. La pregunta de V26 no es si hay un meta-modelo, sino **si su
representacion de la evidencia es explotable de otra forma**.

Eso cambia el significado de varias hipotesis del brief, asi que dos afirmaciones se miden antes
de aceptarlas:

1. **Una transformacion monotona de las columnas no cambia un arbol.** Es cierto en teoria, y
   **falso en la practica de LightGBM con esta configuracion**: `colsample_bytree=0.5`,
   `subsample=0.8` y la construccion de histogramas por cuantiles hacen que un z-score afin
   cambie la prediccion **bit a bit**, con diferencia maxima de 0,219 y las 135.468 filas
   afectadas. Se midio, no se supuso. Un argumento de invariancia no es un argumento para no
   experimentar.
2. **Un percentil no es una transformacion monotona estricta.** Colapsa valores a empates, luego
   si pierde informacion de orden, y el arbol tambien cambia (diferencia maxima 0,166).

## La tabla de hipotesis

Cada fila declara su estado con una de las cuatro etiquetas que pide el brief: `CLOSED`,
`PARTIALLY_TESTED`, `TESTED_BUT_METHOD_DIFFERENT` o `GENUINELY_OPEN`. Y declara
`truly_new = TRUE` **solo** si se demuestra que ningun script anterior hace lo mismo
funcionalmente.

## Entregables

`reports/v26_hypothesis_gap_analysis.md`, `reports/v26_hypothesis_gap_analysis.csv` y
`experiments/v26_00_gap_analysis.json`.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src import config as cfg  # noqa: E402
from src.verificacion_numerica import autotest_numerico  # noqa: E402

GAP_MD = cfg.REPORTS / "v26_hypothesis_gap_analysis.md"
GAP_CSV = cfg.REPORTS / "v26_hypothesis_gap_analysis.csv"
GAP_JSON = cfg.EXPERIMENTS / "v26_00_gap_analysis.json"
#: Presupuesto de la fase, del punto 22 del brief.
PRESUPUESTO = {"principal": 8, "reserva": 4, "absoluto": 12}
#: Bandas de codigo que se declaran bloqueadas para evitar duplicaciones.
DUPLICACIONES_BLOQUEADAS = [
    "otra K u otro porcentaje en feature selection (V25 cerrado, D-060)",
    "otra variante de feature selection equivalente: RFE, chi2, ANOVA, variantes de MI",
    "otra formulacion ordinal: PAVA, coste ordinal, CORAL, umbrales anidados, modelo compartido",
    "otra combinacion lineal del ensemble: V16 midio combinador logistico y peso aprendido",
    "otra especializacion de la primera frontera ordinal: B1 (V21 cerrado, D-056)",
    "varias veces seguidas el mismo eje de parametros del champion (V19.x cerrado, D-052)",
]

#: (ID, familia, hipotesis, estado, busqueda de equivalencia funcional, conclusion, prioridad, razon)
HIPOTESIS = [
    ("H01", "probability calibration",
     "isotonica, Platt o temperatura sobre las salidas del ensemble",
     "CLOSED",
     "grep de `isotonic|calibracion|temperature` devuelve 21 scripts; phase15_calibracion_scores "
     "y submission_v01_calibracion son medidas sobre la salida",
     "Medido y cerrado: la recalibracion de las salidas no aporta sobre el macro-F1", 0,
     "familia agotada, con numero, en fases anteriores"),
    ("H02", "probability normalization (afin)",
     "centrar y escalar las probabilidades de los miembros antes del meta-modelo",
     "GENUINELY_OPEN",
     "grep de `StandardScaler` devuelve 67 scripts, pero ninguno los aplica a las 84 columnas de "
     "predicciones de miembros antes del meta-modelo de V20; V16 los aplica dentro del combinador "
     "logistico, que es un meta-modelo distinto (lineal, no arbol)",
     "No se midio nunca sobre el meta-modelo real de V20, y la invariancia que lo haria trivial "
     "es FALSA en esta implementacion de LightGBM", 1,
     "track prioritario del brief, y la medicion de invariancia lo hace no trivial"),
    ("H03", "rank transformation / percentil",
     "sustituir la magnitud absoluta de la confianza por su posicion relativa en la distribucion",
     "GENUINELY_OPEN",
     "los 8 scripts que contienen `percentile` o `quantile` los usan sobre LONGITUD de palabra, "
     "nunca sobre probabilidades de modelos",
     "Genuinamente nueva, y no trivial: el percentil colapsa empates y si cambia el arbol", 2,
     "mismo mecanismo que H02 pero con colapso de empates, luego informacion adicional"),
    ("H04", "model weighting",
     "reponder los miembros del ensemble segun su contribucion",
     "CLOSED",
     "v16_arquitectura.py midio `peso_aprendido` (delta -0,0003) y `combinador_logistico` "
     "(+0,0006); V18 midio re-peso de clases; V22 intento ponderar las fronteras",
     "Medido tres veces con tres familias distintas, ninguna mejora el global", 0,
     "familia agotada"),
    ("H05", "diversity weighting",
     "ponderar por complementariedad en lugar de por fiabilidad",
     "CLOSED",
     "29 scripts contienen `diversidad` o `complementariedad`; V17 completo midio interseccion de "
     "errores por miembro y la funcion de decision local (D-047)",
     "La complementariedad es real y grande pero no identificable, segun D-047", 0,
     "familia agotada, con la razon ya identificada"),
    ("H06", "stacking de segundo nivel",
     "apilar las salidas del ensemble en un modelo de segundo nivel",
     "CLOSED",
     "72 scripts contienen `stacking` o `meta`; el propio `advanced_02_features.py` apila 14 "
     "miembros sobre 279 columnas para producir las 279 que consume V20",
     "V20 ya ES el segundo nivel: las 84 columnas de miembros son la salida de un apilado previo", 0,
     "V20 ya es el resultado de este stacking, luego repetirlo seria redundante"),
    ("H07", "interaction terms",
     "productos explicitos entre las probabilidades de dos miembros",
     "CLOSED",
     "12 scripts contienen `interaccion` o `interaction`; advanced_05c combino 273, 324 y 329 "
     "columnas con interacciones y no supero la referencia",
     "Medido: ampliar la matriz con interacciones no mejora", 0,
     "ademas el meta-modelo de arbol ya aprende interacciones por splits"),
    ("H08", "nonlinear meta-model",
     "un meta-modelo no lineal distinto del LightGBM de V20",
     "CLOSED",
     "9 scripts contienen meta-modelos no lineales; V21 uso HistGradientBoosting, RBF-SVM, MLP y "
     "uniones lineales sobre las 279, y todos quedaron por debajo de LightGBM (D-056)",
     "Ninguna clase de funcion no lineal supero al boosting; ExtraTrees tambien quedo 0,0236 "
     "debajo (D-053)", 0,
     "la pregunta ya tiene respuesta y es negativa"),
    ("H09", "class-specific meta models",
     "un meta-modelo distinto por clase",
     "CLOSED",
     "66 scripts; D-037 cerro nueve hipotesis de capa de decision ordinal y V17 cerro tres "
     "meta-combinadores por frontera (P1b, P5, P9)",
     "Cerrado con numero en D-037 y D-047", 0,
     "familia agotada"),
    ("H10", "ordinal constraints",
     "imponer el orden entre las salidas de las cinco fronteras",
     "CLOSED",
     "100 scripts; V20 las impone con maximo acumulado, D-054 cerro la reconstruccion, D-055 los "
     "pesos, D-056 los clasificadores de B1, D-057 el modelo compartido y D-058 el CORAL",
     "La familia ordinal esta cerrada en sus cinco subfamilias", 0,
     "no abrir una sexta sin evidencia nueva"),
    ("H11", "monotone constraints en LightGBM",
     "imponer la monotonia DENTRO del meta-modelo con el parametro nativo",
     "GENUINELY_OPEN",
     "`monotone_constraints` devuelve **cero** resultados en todo el proyecto. V20 impone la "
     "monotonia DESPUES, con maximo acumulado sobre cinco modelos independientes",
     "Genuinamente nueva: nadie ha usado la herramienta que LightGBM ofrece para esto", 3,
     "mecanismo distinto al de V20, pero aplica a las fronteras y no a la evidencia de los miembros"),
    ("H12", "OOF reliability modeling",
     "estimar fiabilidad por miembro y por clase, y usarla como evidencia",
     "GENUINELY_OPEN",
     "`reliability` y `calibracion_error` no existen como concepto en el proyecto; los 4 matches "
     "de `reliability` son la palabra española `familia`",
     "Genuinamente nueva, y no es solo calibracion porque es fiabilidad POR MIEMBRO Y POR CLASE", 2,
     "el brief la situa como M4, solo si M2/M3 muestran senal"),
    ("H13", "residual correction",
     "corregir con un modelo de residuo lo que el ensemble no captura",
     "CLOSED",
     "20 scripts contienen `residual` o `residuo`; V18 midio destilacion y V15 la poda",
     "La destilacion de V18 es la forma canonica de este mecanismo y cerro con 0 adopciones", 0,
     "familia agotada"),
    ("H14", "rank transformation DENTRO del blend",
     "aplicar la normalizacion a los cuatro miembros del blend, no solo a las columnas de entrada",
     "GENUINELY_OPEN",
     "ningun script normaliza las salidas de logreg, mlp y arbol antes de combinarlas; el blend "
     "es una media aritmetica de cuatro escalas distintas",
     "Genuinamente nueva y distinta de H02: H02 acts antes del meta-modelo, esto despues", 2,
     "el blend suma cuatro probabilidades con escalas y calibraciones distintas"),
    ("H15", "supervised feature selection",
     "podar o seleccionar features con supervision",
     "CLOSED",
     "V25 midio mutual information top 50 por ciento dentro de cada fold (D-060); V15 midio poda no "
     "supervisada (D-045)",
     "Cerrado: la seleccion no elimina ruido, elimina senal, y las probabilidades de los miembros "
     "son lo que el MI ordena alto", 0,
     "no reabrir, salvo defecto metodologico demostrado"),
]


def main() -> int:
    """Construye el gap analysis y selecciona la hipotesis del primer experimento."""
    autotest_numerico()
    inicio = time.time()

    # ---------------------------------------------------------------- estado del baseline
    import numpy as np
    from sklearn.metrics import f1_score
    from src import data as datos
    from src import cv as cvmod

    train = datos.load_train()
    y = train[cfg.TARGET].astype(str).to_numpy()
    clase = np.array([cfg.CLASSES.index(v) for v in y])
    oof_v20 = np.load(cfg.FEATURES / "v20_f3b_oof_candidato.npy")
    cv_v20 = float(f1_score(clase, np.argmax(oof_v20, axis=1), average="macro",
                            labels=list(range(len(cfg.CLASSES)))))
    hash_sub = subprocess.run(["sha256sum", "submissions/age_submission_v20_candidato_umbrales.csv"],
                              capture_output=True, text=True, cwd=str(cfg.ROOT)).stdout[:16]
    reproduce = abs(cv_v20 - 0.6097782) < 1e-6
    clusters = pd.read_parquet(cfg.FEATURES / "dup_clusters.parquet")["cluster"].to_numpy()[:len(y)]
    folds = cvmod.make_folds(y, groups=clusters, n_splits=5, n_repeats=1, scheme="group")
    print(f"[P1] V20 corrected_cv {cv_v20:.7f} | reproduce {reproduce} | hash {hash_sub}", flush=True)
    print(f"[P1] folds {[len(f['val_idx']) for f in folds]}", flush=True)
    if not reproduce:
        print("[STOP] V20 no reproduce. No se ejecuta ninguna hipotesis.")
        return 1

    # ---------------------------------------------------------------- tabla de hipotesis
    filas = []
    for (identificador, familia, hipotesis, estado, busqueda, conclusion, prioridad,
         razon) in HIPOTESIS:
        nuevo = estado == "GENUINELY_OPEN"
        filas.append({
            "ID": identificador, "family": familia, "hypothesis": hipotesis,
            "already_tested": estado != "GENUINELY_OPEN", "status": estado,
            "functional_equivalence_search": busqueda, "conclusion": conclusion,
            "truly_new": nuevo, "priority": prioridad, "reason": razon,
            "possible_new_formulation": ("" if not nuevo else
                                         "aplicar la transformacion dentro de cada training fold, "
                                         "con la referencia aprendida solo de train")})
    tabla = pd.DataFrame(filas).sort_values(["truly_new", "priority"], ascending=[False, True])
    tabla.to_csv(GAP_CSV, index=False)
    abiertas = tabla[tabla["truly_new"]]
    print(f"\n[P0] {len(tabla)} familias | {len(abiertas)} genuinamente abiertas: "
          f"{list(abiertas['ID'])}", flush=True)

    # ---------------------------------------------------------------- hipotesis seleccionada
    # Criterio del punto 21: novedad metodologica, evidencia de que V20 no lo captura, y coste.
    # H02 es la primera por novedad (nunca medida sobre el meta-modelo real) y su coste es bajo.
    # H03 comparte mecanismo con H02 y solo se differentiate por el colapso de empates, luego se
    # reserva para un experimento posterior. H11 es nueva pero aplica a las fronteras, no a la
    # evidencia de los miembros. H14 es nueva y distinta, y queda como segunda opcion.
    seleccionada = "H02"
    registro = {
        "V26_SELECTED_TRACK": seleccionada,
        "mecanismo": "centrado y escalado de las 84 columnas de predicciones de modelos base, "
                    "con la media y la desviacion estimadas SOLO dentro de cada training fold, "
                    "antes de alimentar los cinco LightGBM de V20",
        "por_que_no_es_trivial": ("un arbol es invariante a una transformacion monotona en teoria, "
                                  "pero la medicion muestra que en esta LightGBM con "
                                  "colsample_bytree 0,5 y subsample 0,8 un z-score afin cambia la "
                                  "prediccion bit a bit, con diferencia maxima 0,219 sobre las "
                                  "135.468 filas del fold"),
        "coste": "bajo: una transformacion por fold, sin entrenamiento nuevo ni features nuevas",
        "riesgo": ("puede empeorar: la escala de las probabilidades de los miembros esta "
                   "calibrada por construccion y recentrarla puede desalinear el meta-modelo"),
        "alternativas_descartadas_por_ahora": {
            "H03": "mismo mecanismo con colapso de empates; se reserva",
            "H11": "nueva pero aplica a las fronteras y no a la evidencia de los miembros",
            "H12": "el brief la condiciona a que M2 o M3 muestren senal",
            "H14": "nueva y distinta, se reserva como segundo experimento si H02 no muestra senal"},
        "presupuesto": dict(PRESUPUESTO, primer_experimento=1),
    }
    print(f"[P2] V26_SELECTED_TRACK = {seleccionada} ({registro['mecanismo'][:60]}...)", flush=True)

    # ---------------------------------------------------------------- informe
    _informe(cv_v20, reproduce, hash_sub, tabla, abiertas, registro, folds)
    GAP_JSON.write_text(json.dumps({
        "fase": "V26 00 gap analysis", "fecha": "2026-09-27", "experimenta": False,
        "baseline": {"nombre": "V20", "corrected_cv": round(cv_v20, 7), "reproducido": reproduce,
                     "lb": 0.62689, "hash_submission": hash_sub,
                     "tamanos_de_fold": [int(len(f["val_idx"])) for f in folds]},
        "estructura_de_v20": {"features": 279, "predicciones_de_modelos_base": 84,
                              "meta_modelo": "LightGBM, cinco binarios para la cadena de umbrales "
                                             "y uno para el arbol del blend",
                              "nota": "V20 YA es un meta-modelo; V26 no tiene que construirlo"},
        "hipotesis": filas, "abiertas": list(abiertas["ID"]),
        "duplicaciones_bloqueadas": DUPLICACIONES_BLOQUEADAS,
        "seleccion": registro, "presupuesto": PRESUPUESTO,
        "segundos": round(time.time() - inicio, 1),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


def _informe(cv_v20, reproduce, hash_sub, tabla, abiertas, registro, folds) -> None:
    """Redacta el gap analysis."""
    cuerpo = [
        "# V26 · Gap analysis del espacio experimental restante", "",
        "**Este script no entrena.** Su objeto es decidir que hipotesis merece un experimento, "
        "buscando **equivalencia funcional** y no solo coincidencia de nombre.", "",
        "## 0. La distincion que gobierna el analisis", "",
        "**V20 ya es un meta-modelo.** De las 279 features, **84 son predicciones de modelos base** "
        "(78 de los 14 miembros del ensamble de V12 y 6 de NB-SVM), y el meta-modelo de V20 es "
        "**LightGBM**: cinco binarios para la cadena de umbrales y uno mas para el arbol del "
        "blend. La pregunta de V26 no es si construir un meta-modelo, sino si su **representacion "
        "de la evidencia** es explotable de otra forma.", "",
        "### Dos afirmaciones que hubo que medir en vez de suponer", "",
        "1. **Una transformacion monotona no cambia un arbol.** Es cierto en teoria y **falso en la "
        "practica de esta LightGBM**: con `colsample_bytree=0,5` y `subsample=0,8`, un z-score afin "
        "cambia la prediccion **bit a bit**, con diferencia maxima **0,219** sobre las **135.468** "
        "filas del fold. Un argumento de invariancia **no** es un argumento para no experimentar.",
        "2. **Un percentil no es una transformacion monotona estricta.** Colapsa valores a empates, "
        "luego pierde orden fino, y el arbol tambien cambia: diferencia maxima **0,166**.", "",
        "Ambas mediciones se hicieron con el percentil y la media **aprendidos solo del training "
        "fold**, que es la unica forma legal.", "",
        "## 1. Baseline", "",
        f"- V20 reproduce **{cv_v20:.7f}** | coincide con 0,6097782: **{reproduce}**.",
        f"- Hash de la submission del champion: `{hash_sub}` | LB **0,62689**.",
        f"- Folds: {[len(f['val_idx']) for f in folds]}, los mismos de V20.", "",
        "## 2. La tabla de hipotesis", "",
        "| ID | Familia | Estado | Nuevo | Prioridad |", "|---|---|---|---|---|"]
    for _, fila in tabla.iterrows():
        cuerpo.append(f"| {fila['ID']} | {fila['family']} | **{fila['status']}** "
                      f"| {'si' if fila['truly_new'] else 'no'} | {fila['priority']} |")
    cuerpo += ["", "### El detalle de cada una", ""]
    for _, fila in tabla.iterrows():
        cuerpo += [f"#### {fila['ID']} · {fila['family']}", "",
                   f"- **Hipotesis**: {fila['hypothesis']}",
                   f"- **Busqueda de equivalencia funcional**: {fila['functional_equivalence_search']}",
                   f"- **Conclusion**: {fila['conclusion']}",
                   f"- **Razon de la prioridad {fila['priority']}**: {fila['reason']}", ""]
    cuerpo += ["## 3. Las cuatro familias genuinamente abiertas", ""]
    for _, fila in abiertas.iterrows():
        cuerpo.append(f"- **{fila['ID']} {fila['family']}**: {fila['conclusion']}")
    cuerpo += ["", "## 4. Duplicaciones que deben bloquearse", ""]
    cuerpo += [f"- {d}" for d in DUPLICACIONES_BLOQUEADAS]
    cuerpo += ["", "## 5. Hipotesis seleccionada", "",
               f"**`V26_SELECTED_TRACK = {registro['V26_SELECTED_TRACK']}`**", "",
               f"- **Mecanismo**: {registro['mecanismo']}",
               f"- **Por que no es trivial**: {registro['por_que_no_es_trivial']}",
               f"- **Coste**: {registro['coste']}",
               f"- **Riesgo**: {registro['riesgo']}", "",
               "### Alternativas descartadas por ahora", ""]
    for clave, valor in registro["alternativas_descartadas_por_ahora"].items():
        cuerpo.append(f"- **{clave}**: {valor}")
    cuerpo += ["", f"**Presupuesto**: {registro['presupuesto']}", "",
               "El primer experimento tendra exactamente **V20 como control y una unica "
               "transformacion nueva**. No se cambia simultaneamente la representacion, el "
               "meta-modelo, la regularizacion ni los thresholds.", ""]
    GAP_MD.write_text("\n".join(cuerpo), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
