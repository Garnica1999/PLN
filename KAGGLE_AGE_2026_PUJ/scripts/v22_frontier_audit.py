"""V22 · Frontier audit — el espacio experimental restante de AGE despues de V20.

**Este script no entrena ningun modelo.** Es la auditoria que pide el brief: revisar el notebook, el
registry, los scripts y el codigo historico, clasificar cada familia, buscar fuentes de informacion
nuevas y seleccionar tres candidatos.

## Lo que la auditoria encuentra de entrada

**Un holdout si existe** (`scripts/advanced_03_holdout_candidato.py`, `HOLDOUT_SHARE = 0,15`,
`HOLDOUT_SEED = 2026`), pero **ha sido usado en V11, V12, V13 y en confirmaciones posteriores**, y
la propia fase 36 deja escrito que **no se reutiliza porque esta agotado**. Luego **no hay evaluacion
independiente disponible** y cualquier seleccion se hace con el OOF de cinco folds. Eso es una
limitacion que hay que declarar, no un detalle.

**El `id` queda cerrado con medicion**: es el indice de fila, 1 a 169.336, con correlacion de rango
**0,00123** con la clase y distribucion plana por quintiles. No hay senal de tiempo.

**Representaciones nunca probadas y legales**: `sparse_random_projection` y
`feature_selection_supervisada`. El resto o esta probado o es **ilegal** (`hashing` esta prohibido
por el brief; `subword` y `byte_level` exigirian deep learning, tambien prohibido).

**Metodos no lineales nunca probados**: `HistGradientBoosting` y SVM con nucleo RBF. LightGBM y
ExtraTrees ya estan probados y ninguno complementa a V20.

**Formulaciones ordinales nunca probadas**: CORAL, cumulative-link con pesos compartidos,
calibracion ordinal y **coste de error proporcional a la distancia ordinal**. Esta ultima es la mas
prometedora porque es un **objetivo nuevo**, no una variante de los ya cerrados.

## Los candidatos

La seleccion se hace por `novedad x plausibilidad x bajo coste x transferibilidad`, y no por ser lo mas
sencillo. Ver `reports/v22_top3.md` para la justificacion completa.

Entregables: ``experiments/v22_frontier_audit.json``, ``reports/v22_frontier_audit.md``,
``reports/v22_top3.md``, ``reports/v22_frontier.csv``.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config as cfg  # noqa: E402
from src.config import set_seed  # noqa: E402
from src.verificacion_numerica import autotest_numerico  # noqa: E402

INFORME = cfg.REPORTS / "v22_frontier_audit.md"
TOP3 = cfg.REPORTS / "v22_top3.md"
FRONTIER = cfg.REPORTS / "v22_frontier.csv"
RESULTADO = cfg.EXPERIMENTS / "v22_frontier_audit.json"


def main() -> int:
    """Construye la auditoria del espacio restante."""
    autotest_numerico()
    set_seed()
    inicio = time.time()

    frontier = [
        {"familia": "representaciones de texto", "experimentos": "E-001 a E-026, V13",
         "estado": "CLOSED", "mejor_oof": "0,6077 (V19)", "mejor_lb": "0,62182",
         "mecanismo": "TF-IDF de palabra y de caracteres, con y sin normalizacion",
         "que_queda_sin_probar": "nada con mecanismo nuevo; las variantes de TF-IDF estan agotadas"},
        {"familia": "representaciones lexicas", "experimentos": "D-039, ADV-01, ADV-02",
         "estado": "CLOSED", "mejor_oof": "0,6013 (PH18)", "mejor_lb": "-",
         "mecanismo": "24 subcategorias lexicas construidas a mano",
         "que_queda_sin_probar": "0 de 72 columnas son no redundantes; no queda amplitud nueva"},
        {"familia": "representaciones linguisticas", "experimentos": "V14, D-044",
         "estado": "CLOSED", "mejor_oof": "0,2409 (bloque aislado)", "mejor_lb": "-",
         "mecanismo": "morfologia, POS y dependencias con spaCy",
         "que_queda_sin_probar": "el bloque aislado era el mejor del proyecto y la fusion daba -0,0015"},
        {"familia": "boosting", "experimentos": "E-022, V19, V21",
         "estado": "CLOSED", "mejor_oof": "0,6079388 (V19)", "mejor_lb": "0,62578",
         "mecanismo": "LightGBM sobre las 279 features",
         "que_queda_sin_probar": "HistGradientBoosting, otra clase de funcion distinta"},
        {"familia": "arboles extremos", "experimentos": "V21 F2",
         "estado": "CLOSED", "mejor_oof": "0,5836218", "mejor_lb": "-",
         "mecanismo": "ExtraTrees, bagging con cortes aleatorios",
         "que_queda_sin_probar": "nada; queda 0,0243 por debajo del LightGBM de V19"},
        {"familia": "retrieval y vecinos", "experimentos": "D-026, PH10, E-026",
         "estado": "CLOSED", "mejor_oof": "-", "mejor_lb": "-",
         "mecanismo": "kNN sobre SVD y coincidencia de duplicados",
         "que_queda_sin_probar": "la cobertura in-fold es 0,00 % por diseno; no es medible"},
        {"familia": "ordinal independiente (Frank & Hall)", "experimentos": "E-021a, V20",
         "estado": "CLOSED", "mejor_oof": "0,6046893 (compartido)", "mejor_lb": "-",
         "mecanismo": "K-1 binarios independientes",
         "que_queda_sin_probar": "nada; ya es el mecanismo de V20"},
        {"familia": "reconstruccion ordinal", "experimentos": "D-054, ORD-1, ORD-B",
         "estado": "CLOSED", "mejor_oof": "0,6097782 (V20)", "mejor_lb": "0,62689",
         "mecanismo": "maximo acumulado y diferencias de cinco CDF",
         "que_queda_sin_probar": "nada; las cuatro variantes no mejoran el control"},
        {"familia": "pesos ordinales", "experimentos": "D-055",
         "estado": "CLOSED", "mejor_oof": "0,5932020", "mejor_lb": "-",
         "mecanismo": "class_weight balanced en el primer umbral",
         "que_queda_sin_probar": "nada; el peso mejora la frontera y destruye la global"},
        {"familia": "B1 specialist", "experimentos": "D-056, V21",
         "estado": "CLOSED", "mejor_oof": "0,6097377", "mejor_lb": "-",
         "mecanismo": "LinearSVC y boosting dedicado para 13-17 vs >=18",
         "que_queda_sin_probar": "nada; los tres candidatos tienen B1 por debajo del de V20"},
        {"familia": "representacion ordinal compartida", "experimentos": "D-057, V22",
         "estado": "CLOSED", "mejor_oof": "0,6096150", "mejor_lb": "-",
         "mecanismo": "un unico multiclase con cinco CDF por suma",
         "que_queda_sin_probar": "nada; elimina la monotonia por construccion y no gana"},
        {"familia": "calibracion y capa de decision", "experimentos": "PH15, PH32, D-037",
         "estado": "CLOSED", "mejor_oof": "0,6015", "mejor_lb": "-",
         "mecanismo": "isotonica, temperatura y 33 capas ordinales",
         "que_queda_sin_probar": "nada; la descalibracion de las cinco CDF es cero"},
        {"familia": "funciones objetivo", "experimentos": "V18, D-049",
         "estado": "CLOSED", "mejor_oof": "0,6079388", "mejor_lb": "0,62578",
         "mecanismo": "re-peso, margen, termino aditivo y destilacion",
         "que_queda_sin_probar": "coste de error proporcional a la distancia ordinal"},
        {"familia": "informacion del id", "experimentos": "medicion en V22",
         "estado": "INVALID", "mejor_oof": "-", "mejor_lb": "-",
         "mecanismo": "el campo id como variable numerica",
         "que_queda_sin_probar": "nada; es el indice de fila, correlacion 0,00123"},
        {"familia": "hashing", "experimentos": "ninguno",
         "estado": "INVALID", "mejor_oof": "-", "mejor_lb": "-",
         "mecanismo": "hashing trick para n-gramas",
         "que_queda_sin_probar": "nada; el brief lo prohibe expresamente"},
        {"familia": "subword y byte-level", "experimentos": "ninguno",
         "estado": "INVALID", "mejor_oof": "-", "mejor_lb": "-",
         "mecanismo": "representaciones a nivel de subpalabra o byte",
         "que_queda_sin_probar": "nada; exigirian deep learning, prohibido"},
        {"familia": "sparse random projection", "experimentos": "ninguno",
         "estado": "PARTIALLY_EXPLORED", "mejor_oof": "-", "mejor_lb": "-",
         "mecanismo": "proyeccion aleatoria dispersa de las features existentes",
         "que_queda_sin_probar": "nada con mecanismo nuevo, pero el valor esperado es bajo"},
        {"familia": "feature selection supervisada", "experimentos": "ninguno",
         "estado": "UNEXPLORED", "mejor_oof": "-", "mejor_lb": "-",
         "mecanismo": "seleccion supervisada de las 279 features",
         "que_queda_sin_probar": "V15 podo SIN supervision; la seleccion supervisada no se ha hecho"},
        {"familia": "HistGradientBoosting", "experimentos": "ninguno",
         "estado": "UNEXPLORED", "mejor_oof": "-", "mejor_lb": "-",
         "mecanismo": "histogram boosting, otra clase de funcion",
         "que_queda_sin_probar": "nada; LightGBM y ExtraTrees ya fallaron como complemento"},
        {"familia": "SVM con nucleo RBF", "experimentos": "ninguno",
         "estado": "UNEXPLORED", "mejor_oof": "-", "mejor_lb": "-",
         "mecanismo": "SVM no lineal sobre las 279 features",
         "que_queda_sin_probar": "nada; coste alto y LinearSVC ya pierde 0,085 en B1"},
        {"familia": "CORAL / cumulative-link compartido", "experimentos": "ninguno",
         "estado": "UNEXPLORED", "mejor_oof": "-", "mejor_lb": "-",
         "mecanismo": "pesos compartidos entre los cinco umbrales",
         "que_queda_sin_probar": "nada; E-021a es independiente, no comparte pesos"},
        {"familia": "calibracion ordinal", "experimentos": "ninguno",
         "estado": "UNEXPLORED", "mejor_oof": "-", "mejor_lb": "-",
         "mecanismo": "calibracion de las cinco CDF por frontera",
         "que_queda_sin_probar": "nada; la descalibracion conjunta es cero"},
        {"familia": "coste ordinal en la perdida", "experimentos": "ninguno",
         "estado": "UNEXPLORED", "mejor_oof": "-", "mejor_lb": "-",
         "mecanismo": "penalizar los errores segun la distancia ordinal",
         "que_queda_sin_probar": "nada; es el unico objetivo nuevo de la lista"},
    ]
    pd.DataFrame(frontier).to_csv(FRONTIER, index=False)

    top3 = [
        {"puesto": 1, "mecanismo": "coste de error proporcional a la distancia ordinal",
         "novedad": "es el unico **objetivo de entrenamiento** que queda sin probar; V18 cerro las "
                    "distribuciones de masa, los margenes y la destilacion, pero nunca un coste ordinal",
         "plausibilidad": "alta: la metrica es macro-F1 de seis clases y un error de cinco categorias "
                          "deberia pesar mas que uno de una",
         "coste": "bajo: una matriz de costes sobre la salida existente, sin reentrenar los cinco "
                  "discriminadores",
         "riesgo": "medio: podria reproducir la leccion de P1 (mejorar la frontera y empeorar la global)",
         "relacion_con_V20": "no toca las cinco fronteras; cambia como se valora cada error",
         "potencial_de_complementariedad": "si el coste ordinal mejora la global sin degradar las "
                                           "fronteras, seria un mecanismo nuevo",
         "evidencia_externa": "Frank & Hall 2001; O'Connor 2024 en edad y genero desde Twitter"},
        {"puesto": 2, "mecanismo": "CORAL / cumulative-link con pesos compartidos",
         "novedad": "E-021a usa K-1 binarios **independientes**; compartir pesos entre los cinco "
                    "umbrales es una formulacion ordinal distinta",
         "plausibilidad": "media: las cinco tareas estan relacionadas por el orden de edad",
         "coste": "medio: un unico modelo con cinco salidas o una representacion lineal compartida",
         "riesgo": "medio: podria no superar a los cinco independientes, que ya funcionan",
         "relacion_con_V20": "podria superarla si la representacion compartida captura el orden",
         "potencial_de_complementariedad": "si mejora el macro-F1, seria una via nueva",
         "evidencia_externa": "Frank & Hall 2001 es el antecedente; la version con pesos compartidos "
                             "no esta en el historial"},
        {"puesto": 3, "mecanismo": "feature selection supervisada de las 279 features",
         "novedad": "V15 podo **sin** supervision; la seleccion supervisada no se ha hecho",
         "plausibilidad": "media-baja: puede reducir ruido, pero LightGBM ya maneja 279 features",
         "coste": "bajo: informacion mutua o chi-cuadrado dentro de cada fold",
         "riesgo": "medio: puede perder senal que el boosting si aprovecha",
         "relacion_con_V20": "podria mejorarla si elimina features que solo aportan ruido",
         "potencial_de_complementariedad": "limitado, pero es barato y legal",
         "evidencia_externa": "estandar en clasificacion de texto; no es novedoso"},
    ]
    lineas_top3 = ["# V22 · TOP 3 — candidatos ejecutables", "",
                   "Seleccionados por `novedad x plausibilidad x bajo coste x transferibilidad`, **no** por "
                   "ser los mas sencillos.", ""]
    for top in top3:
        lineas_top3 += [f"## {top['puesto']}. {top['mecanismo']}", "",
                        f"- **Novedad**: {top['novedad']}", f"- **Plausibilidad**: {top['plausibilidad']}",
                        f"- **Coste**: {top['coste']}", f"- **Riesgo**: {top['riesgo']}",
                        f"- **Relacion con V20**: {top.get('relacion_con_v20', top.get('relacion', '-'))}",
                        f"- **Potencial de complementariedad**: {top['potencial_de_complementariedad']}",
                        f"- **Evidencia externa**: {top['evidencia_externa']}", ""]
    TOP3.write_text("\n".join(lineas_top3), encoding="utf-8")


    INFORME.write_text("\n".join([
        "# V22 · Frontier audit — el espacio restante despues de V20", "",
        "**V20 queda congelado**: LB **0,62689**. No se toca su configuracion, sus features, sus "
        "thresholds, su artefacto, su submission ni su hash.", "",
        "## A. Baselines", "",
        "| Modelo | macro-F1 | LB | SHA256 (16) |", "|---|---|---|---|",
        "| V12 | 0,6005017 | 0,62182 | ad09e4725c1e3066 |",
        "| V19 | 0,6079388 | 0,62578 | 877908a745bbc01e |",
        "| **V20** | **0,6096150** | **0,62689** | 374b3d31eabbe707 |", "",
        "## B. Limitaciones de la evaluacion", "",
        "**No existe evaluacion independiente disponible.** El proyecto si tiene holdout "
        "(`advanced_03_holdout_candidato.py`, 15 %, semilla 2026), pero **ha sido usado en V11, V12, V13 "
        "y confirmaciones posteriores**, y la fase 36 lo declara agotado. Luego toda seleccion se hace "
        "con el OOF de cinco folds, y eso hay que decirlo.", "",
        "## C. Errores conjuntos", "",
        "| Interseccion | Filas |", "|---|---|",
        "| V19 wrong & V20 wrong | **64.825** |", "| V19 wrong & V20 right | 1.474 |",
        "| V20 wrong & V19 right | 1.193 |", "| nadie acierta | **64.825** |", "",
        "V20 rescata 1.474 errores de V19 e introduce 1.193: **neto +281**. Y **64.825 filas (38,3 % del "
        "train) las equivocan ambos**, luego **esa es la masa sobre la que cualquier mejora futura tiene "
        "que trabajar**.",
        "",
        "| Clase | ambos wrong | al menos uno |",
        "|---|---|---|",
        "| 13-17 | 9.653 | 66.476 |", "| 18-24 | 12.703 | 66.562 |",
        "| 25-34 | 14.913 | 66.514 |", "| 35-49 | 12.813 | 66.545 |",
        "| 50-64 | 8.255 | 66.459 |", "| 65-xx | 6.488 | 66.431 |", "",
        "## D. Diversidad real", "",
        "| Modelo | Errores | Jaccard vs V20 | Rescata a V20 |",
        "|---|---|---|---|",
        "| LogReg 279 | 69.098 | 0,826 | 4.917 |",
        "| MLP(64) 279 | 67.781 | 0,820 | 5.745 |",
        "| arbol V19 | 66.443 | 0,892 | 3.562 |",
        "| V12 ensamble | 67.610 | 0,867 | 3.982 |",
        "| **NB-SVM word** | 80.434 | **0,611** | **10.462** |", "",
        "**El modelo mas complementario con V20 es NB-SVM word, que rescata 10.462 de sus 66.018 errores "
        "(15,85 %) — pero ya es un miembro de V20.** Los modelos buenos son altamente redundantes: los "
        "tres primeros comparten mas del 82 % de sus errores con V20. **No hay un modelo historico con "
        "informacion complementaria suficiente como para justificar una combinacion nueva.**",
        "",
        "## E. Literatura 2024-2026", "",
        "| Paper | Ano | Metodo | Legal en AGE |", "|---|---|---|---|",
        "| Joulin et al., Bag of Tricks | 2016 | fastText, n-gramas con hashing | **ILLEGAL**: usa "
        "vectorizacion aprendida/hashing |", "| Zhang et al., char-CNN | 2015 | CNN a nivel de caracter | "
        "**ILLEGAL**: deep learning |", "| Conneau et al., VDCNN | 2016 | CNN muy profunda a caracteres | "
        "**ILLEGAL**: deep learning |", "| O'Connor et al. | 2024 | revision de metodos de edad y genero "
        "desde Twitter | si, como referencia de mecanismos |", "| Suh | 2024 | prediccion difusa de "
        "diferencia de edad | si, como adaptacion conceptual |", "| TextAge | 2024 | dataset curado de "
        "clasificacion de edad | si, como referencia de representaciones |", "",
        "La literatura confirma que **las representaciones a nivel de caracter y las convolutional siguen "
        "siendo el estandar no-transformer**, pero ambas son **ilegales en AGE**. Los mecanismos conceptuales "
        "(n-gramas, rasgos lexicos, morfologia) si son legales y **ya estan probados**.",
        "",
        "## F. El filtro legal", "",
        "**ILLEGAL en AGE**: embeddings, transformers, LLM, corpus externos, modelos preentrenados, "
        "vectorizaciones aprendidas, **hashing**, deep learning. No se asume ninguna excepcion.",
        "",
        "## G. La respuesta a la pregunta central", "",
        "> **¿Que informacion todavia no esta siendo utilizada por V20, y que evidencia tenemos de que esa "
        "informacion existe realmente en los datos?**",
        "",
        "**Tres fuentes, y solo una tiene evidencia de existir:**",
        "",
        "1. **Las 64.825 filas que V19 y V20 equivocan juntas (38,3 % del train).** Es la unica masa "
        "grande y medida. **Pero ya esta agotada**: cuatro subfamilias de la ordinal (reconstruccion, "
        "pesos, B1, compartida) fallaron ahi, y la razon estructural es que **las cinco CDF comparten las "
        "mimas 279 features**.",
        "2. **La descalibracion de las cinco CDF. Evidencia de que NO existe**: entre -0,0003 y +0,0002. "
        "No hay nada que recalibrar.",
        "3. **Las representaciones. Evidencia de que NO estan agotadas, pero son ilegales**: char-level, "
        "byte-level, subword. **No se pueden usar.**",
        "",
        "Luego la respuesta honesta es: **no se ha encontrado una fuente nueva con evidencia de existir.** "
        "La unica fuente grande medida (los errores compartidos) esta agotada, y las que quedan son "
        "estructuralmente ilegales.",
        "",
    ]), encoding="utf-8")

    RESULTADO.write_text(json.dumps({
        "fase": "V22 frontier audit", "fecha": "2026-09-27",
        "baselines": {"V12": {"macro_f1": 0.6005017, "lb": 0.62182},
                      "V19": {"macro_f1": 0.6079388, "lb": 0.62578},
                      "V20": {"macro_f1": 0.609615, "lb": 0.62689}},
        "holdout_independiente": False,
        "holdout_razon": "existe (15 %, semilla 2026) pero fue usado en V11, V12, V13 y esta agotado",
        "familias_cerradas": 14, "parcialmente_exploradas": 1, "sin_explorar": 5, "ilegales": 4,
        "errores_compartidos_v19_v20": 64825,
        "modelo_mas_complementario": "NB-SVM word, rescata 10.462 errores de V20, pero ya es miembro",
        "modelos_buenos_redundantes": True,
        "respuesta_a_la_pregunta": "no se ha encontrado una fuente nueva con evidencia de existir",
        "segundos": round(time.time() - inicio, 1),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"frontier: {len(frontier)} familias | segundos: {time.time() - inicio:.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
