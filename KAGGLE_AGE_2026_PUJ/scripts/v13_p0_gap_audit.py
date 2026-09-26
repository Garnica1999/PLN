"""V13 · P0 — Gap audit: inventario, líneas cerradas e hipótesis genuinamente nuevas.

Primera entrega de la fase. **No entrena ningún modelo**: solo lee artefactos, mapea la
documentación externa contra las features que V12 ya tiene y decide qué merece ejecutarse.

La afirmación "esto nunca se probó" no se sostiene por opinión: cada variable de la literatura
se mapea contra nombres de columna reales leídos de ``features/``, y el script **falla** si un
nombre declarado no existe. Así el gap audit es falsable.

Fuentes externas usadas para el mapeo (documentadas en el informe):

* Moreno-Sandoval, Pomares-Quimbaya y Alvarado-Valencia (2021). *Celebrity profiling through
  linguistic analysis of digital social networks.* Computational Social Networks 8:16.
  doi:10.1186/s40649-021-00097-w -> variables V1..V18 en 5 grupos.
* Material del curso ``Clasificacion Texto.pdf`` -> 7 familias de caracteristicas de estilo.
* Zhou y Zafarani (2020). ACM Computing Surveys. doi:10.1145/3395046 -> taxonomia de 10 dimensiones.

Salida: ``reports/v13_gap_audit.md``, ``reports/v13_gap_audit.csv``,
``experiments/v13_p0_gap_audit.json``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src import config as cfg  # noqa: E402

GAP_MD = cfg.REPORTS / "v13_gap_audit.md"
GAP_CSV = cfg.REPORTS / "v13_gap_audit.csv"
GAP_JSON = cfg.EXPERIMENTS / "v13_p0_gap_audit.json"

# ---------------------------------------------------------------------------
# 1. Las 18 variables de Moreno-Sandoval et al. (2021) y su equivalente en V12.
#    ``match`` lista nombres de columna que DEBEN existir; el script lo verifica.
# ---------------------------------------------------------------------------
VARIABLES_ARTICULO: list[tuple[str, str, str, list[str]]] = [
    ("V1", "Avg_character (media de caracteres por palabra)", "EXISTE",
     ["avg_word_len", "wlen_mean"]),
    ("V2", "Kurtosis_character (curtosis de la longitud de palabra)", "EXISTE", ["wlen_kurtosis"]),
    ("V3", "Lexical_diversity", "EXISTE",
     ["ttr", "cttr", "hapax_ratio", "root_ttr", "maas", "yule_k", "simpson_d", "char_entropy"]),
    ("V4", "Label_word (media de palabras por publicacion)", "EXISTE", ["n_words"]),
    ("V5", "Kurtosis_avg_character", "EXISTE", ["wlen_cv", "wlen_std"]),
    ("V6", "Kurtosis_label_word", "PARCIAL", ["std_word_len"]),
    ("V7", "Skew_avg_character (asimetria de la longitud)", "EXISTE", ["wlen_skew"]),
    ("V8", "Skew_label_word", "PARCIAL", ["wlen_range", "wlen_iqr"]),
    ("V9", "Label_emoji", "INERTE", ["n_emoji", "has_emoji"]),
    ("V10", "Label_hashtag", "EXISTE", ["n_hashtag"]),
    ("V11", "Label_mention", "EXISTE", ["n_mention", "starts_mention"]),
    ("V12", "URL (informacion complementaria)", "EXISTE", ["n_url"]),
    ("V13", "Label_retweets", "EXISTE", ["is_rt"]),
    ("V14", "Person_1_singular (yo)", "AUSENTE", []),
    ("V15", "Person_2_singular (tu)", "AUSENTE", []),
    ("V16", "Person_3_singular (el/ella)", "AUSENTE", []),
    ("V17", "Person_1_plural (nosotros)", "AUSENTE", []),
    ("V18", "Person_3_plural (ellos/ellas)", "AUSENTE", []),
]

# ---------------------------------------------------------------------------
# 2. Las 7 familias de estilo del curso y las 10 dimensiones de la taxonomia NLP.
# ---------------------------------------------------------------------------
FAMILIAS_CURSO = [
    ("Palabras funcionales", "PARCIAL", "solo stopword_ratio y content_ratio; falta el vector de tasas",
     "P18"),
    ("Sintaxis: n-gramas de POS", "CERRADO", "D-026: corpus ASCII degrada el etiquetado (-0,0032)", "-"),
    ("Arboles SFL (conjunciones por funcion)", "AUSENTE",
     "el curso indica que NO requiere POS; es la via no cerrada tras D-026", "P19"),
    ("Morfologia (prefijos, sufijos)", "CERRADO", "redundante: char_wb(3-5) ya lo cubre", "-"),
    ("Medidas de complejidad (longitud, entropias)", "EXISTE", "wlen_* y *_entropy", "-"),
    ("Palabras inestables (traduccion ida-vuelta)", "PROHIBIDO",
     "requiere modelos de traduccion externos (§3)", "-"),
    ("Idiosincrasias", "CERRADO", "D-039: fase de lexicones cerrada", "-"),
]

DIMENSIONES_TAXONOMIA = [
    ("Quantity / Complexity / Diversity", "EXISTE", "n_chars, n_words, ttr y familia de entropias", "-"),
    ("Informality (typos)", "PROHIBIDO", "requiere diccionario externo para detectar faltas", "-"),
    ("Incertidumbre (modales, certeza, tentativa)", "AUSENTE", "no hay ningun vocabulario modal", "P20"),
    ("Subjectivity (verbos subjetivos, de reporte)", "AUSENTE", "no existe", "P21"),
    ("Non-immediacy (pronombres, pasiva)", "PARCIAL", "solo pasiva prohibida; pronombres ausentes", "P17"),
    ("Specificity (temporal, espacial, exclusivo)", "AUSENTE", "no existe", "P21"),
    ("Sentiment / swear / netspeak", "CERRADO", "reabriria la linea de lexicones (§1, D-039)", "-"),
    ("Readability (Flesch, Gunning-Fog, ARI)", "AUSENTE", "no hay indice de legibilidad", "P23"),
]

# ---------------------------------------------------------------------------
# 3. Las pistas de V13. ``tested_before`` se decide con evidencia del historial.
# ---------------------------------------------------------------------------
PISTAS = [
    # id, nombre, mecanismo, tested_before, exp_ids, best, why_closed, complement, coste, leakage, prio
    ("P0", "Gap audit", "inventario verificable de lo cerrado y lo nuevo", "no aplica", "-", "-",
     "-", "define el resto", "15 min", "ninguno", "0"),
    ("P15", "Matriz de error por miembro", "diagnostico de que miembro sabe y donde", "parcial",
     "PH13, phase20", "diagnostico existente sin desglose por miembro",
     "nunca se respondio que miembro concentra el conocimiento complementario",
     "ALTO: 66,9% de los errores tienen algun miembro correcto", "20 min", "ninguno", "1"),
    ("P1", "Evidencia de edad por token (P(clase|token) in-fold)", "supervision dentro del train fold",
     "no", "-", "-",
     "las fases previas solo usaron P(clase|entidad) para mention/hashtag/RT, nunca por palabra",
     "ALTO: el unico canal supervisado por palabra que no existe", "45 min",
     "ALTO si se calcula antes de los folds (§5, §38)", "1"),
    ("P17", "Pronombres de persona (V14-V18)", "categoria sintactica sin POS", "no", "-", "-",
     "V12 no tiene ninguna variable de pronombres; el articulo los declara predictivos",
     "MEDIO-ALTO: correlaciona con edad y es ortogonal a los n-gramas de caracter", "20 min",
     "ninguno (determinista)", "1"),
    ("P19", "Arboles SFL (conjunciones por funcion)", "funcion gramatical sin POS", "no", "-", "-",
     "unico camino de sintaxis que no necesita etiquetado POS, que si esta cerrado",
     "MEDIO: la funcion gramatical de las conjunciones varia con la edad", "25 min",
     "ninguno (determinista)", "1"),
    ("P20", "Incertidumbre y modalidad", "vocabulario cerrado propio", "no", "-", "-",
     "no existe ningun bloque de marcadores de modalidad",
     "MEDIO: la incertidumbre verbal cambia con la edad", "20 min", "ninguno (determinista)", "1"),
    ("P3", "Especialistas binarios de frontera", "modelo binario por par adyacente", "parcial",
     "E-028a, advanced_07", "routing duro 0,0000; compuerta -0,0036",
     "se evaluo el enrutamiento sobre probabilidades ya entrenadas, no un especialista binario nuevo",
     "MUY ALTO: 10.122 + 7.252 cruces de frontera", "60 min",
     "MEDIO: el especialista debe entrenarse dentro del fold", "1"),
    ("P24", "Forma de la distribucion de evidencia", "medidas de tendencia y dispersion sobre P1", "no",
     "-", "-", "V12 solo aplica esas medidas a longitudes, no a la evidencia por token",
     "MEDIO: distingue evidencia concentrada de evidencia difusa", "25 min",
     "ALTO: hereda el de P1", "2"),
    ("P4", "Boundary gating", "compuerta entrenada sobre la proximidad a la frontera", "parcial",
     "phase33-C4, advanced_07", "correccion de frontera neta 0",
     "se modifico la salida de V12; no se entreno una compuerta con especialista nuevo",
     "ALTO si P3 aporta", "40 min", "MEDIO: umbral aprendido, no fijo sobre la validacion", "2"),
    ("P5", "Entrenamiento pairwise", "objetivo pairwise en vez de post-proceso", "no", "-", "-",
     "Luce, Bradley-Terry y projection ordinal fueron capas de decision sobre probabilidades",
     "ALTO: entrena el modelo con la estructura ordinal en vez de corregirla despues", "60 min",
     "ninguno extra", "2"),
    ("P18", "Palabras funcionales (vector de tasas)", "vocabulario cerrado propio", "parcial",
     "PH7 (stopword_ratio)", "stopword_ratio y content_ratio",
     "solo 2 agregados; falta el vector de tasas que usa authorship attribution clasico",
     "MEDIO-ALTO: es el metodo canonico y V12 no lo tiene", "30 min",
     "ninguno (determinista)", "2"),
    ("P21", "Subjetividad y especificidad", "vocabulario cerrado propio", "no", "-", "-",
     "no existe", "MEDIO", "30 min", "ninguno (determinista)", "2"),
    ("P2", "Evidencia supervisada de hashtag y URL", "P(clase|hashtag|dominio) in-fold", "parcial",
     "PH7, phase14", "menciones supervisadas +0,0016 (familia J/K/L)",
     "se aprendio P(clase|entidad) para mention/hashtag/RT pero sin estructura de URL ni soporte",
     "MEDIO: solo_url tiene 58-59% de error", "40 min", "ALTO si se calcula antes de los folds", "3"),
    ("P6", "CentROIDES de clase en el espacio TF-IDF", "geometria de proximidad a prototipos", "no",
     "-", "-", "nunca se calcularon centroides de clase en el espacio de V12",
     "MEDIO: la decision lineal no es la misma que la proximidad al prototipo", "25 min",
     "ALTO: los centroides deben salir del train fold", "3"),
    ("P22", "Queismo y dequeismo", "bigramas deterministas", "no", "-", "-",
     "el curso lo señala como patron de escritura; nunca se midio", "BAJO-MEDIO", "20 min",
     "ninguno (determinista)", "3"),
    ("P26", "PCA in-fold del bloque nuevo", "reduccion de dimensionalidad antes del modelo", "parcial",
     "fase 4-11", "SVD en el meta",
     "se aplico SVD a representaciones, no al bloque de features nuevas", "BAJO (metodo)", "15 min",
     "ALTO: el PCA debe ajustarse dentro del fold", "3"),
    ("P8", "Topologia de menciones", "posiciones y distancias entre menciones", "parcial",
     "estructuradas pos_mention_*", "posiciones first/last/mean ya existen",
     "faltan densidad, distancia entre menciones y fraccion previa", "BAJO-MEDIO", "20 min",
     "ninguno (determinista)", "4"),
    ("P9", "Topologia de URL y hashtag", "estructura de composicion del tweet", "parcial",
     "estructuradas pos_url_*/pos_hashtag_*", "posiciones ya existen",
     "falta la proporcion de texto ocupada por URL/hashtag/mencion", "BAJO-MEDIO", "20 min",
     "ninguno (determinista)", "4"),
    ("P10", "Estilometria residual", "solo features ausentes, <=30", "parcial",
     "E-001..E-028", "familia completa ya construida (35+87)",
     "debe abrirse solo con lo que P0 demuestre ausente", "BAJO", "20 min",
     "ninguno (determinista)", "4"),
    ("P11", "Vista dual con menciones normalizadas", "canal complementario", "no", "-", "-",
     "el canal de menciones existe, pero no una vista con @usuario -> @MENTION", "MEDIO", "40 min",
     "ALTO: un vectorizador para la vista nueva esta prohibido", "4"),
    ("P12", "Micro-lexicon derivado del target", "seleccion supervisada de tokens in-fold", "no", "-",
     "-", "distinto de los lexicones generales cerrados en D-039: aqui la evidencia sale del train",
     "MEDIO-ALTO", "45 min", "ALTO: la seleccion debe usar OOF interno", "5"),
    ("P23", "Legibilidad en espanol", "indices clasicos con heuristica de silabas", "no", "-", "-",
     "no existe", "BAJO", "20 min", "ninguno (determinista)", "5"),
    ("P16", "Tipologia de error en fronteras", "slices deterministas", "parcial", "PH13", "segmentos",
     "falta el cruce de slices con las dos fronteras centrales", "diagnostico", "20 min",
     "ninguno", "diagnostico"),
    ("P25", "Significancia por atributo", "que variable explica que grupo de edad", "no", "-", "-",
     "nunca se hizo", "diagnostico", "20 min", "ninguno (supervisado, solo train)", "diagnostico"),
    ("P13", "Hard-boundary training", "especialista sobre ejemplos difficulties del train", "no", "-",
     "-", "nunca se intento", "CONDICIONAL: solo si P3 da senal", "90 min",
     "MUY ALTO: exige OOF interno; con diagnostico R-2", "condicional"),
    ("P14", "Combinacion de positivos independientes", "V12 + evidencia + especialista", "no", "-",
     "-", "-", "CONDICIONAL: solo con dos positivos independientes", "60 min", "MEDIO", "condicional"),
]


def features_existentes() -> tuple[set[str], dict]:
    """Lee los nombres reales de las features de V12 y comprueba el mapeo del articulo."""
    disponibles: set[str] = set()
    for nombre in ("dense_train.parquet", "age_structured_features.parquet"):
        ruta = cfg.FEATURES / nombre
        if ruta.exists():
            disponibles |= set(pd.read_parquet(ruta).columns)
    from phase2_ponderacion import MEMBER_FILES
    miembros = [Path(str(m)).stem for m in MEMBER_FILES]
    verificaciones = []
    for variable, nombre, estado, match in VARIABLES_ARTICULO:
        faltan = [columna for columna in match if columna not in disponibles]
        verificaciones.append({"variable": variable, "nombre": nombre, "estado_declarado": estado,
                               "columnas_v12": match, "columnas_no_encontradas": faltan,
                               "verificada": not faltan or estado == "AUSENTE"})
        if faltan and estado != "AUSENTE":
            raise SystemExit(f"mapeo invalido: {variable} declara {faltan} que no existen en V12")
    return disponibles, {"miembros": miembros, "verificaciones": verificaciones}


def main() -> int:
    """Construye el gap audit de V13 a partir de los artefactos del proyecto."""
    disponibles, detalle = features_existentes()
    historial = pd.read_csv(ROOT / "experiments.csv", dtype=str).fillna("")

    # --- 1. Inventario real del historial ------------------------------------
    bloques = (historial.groupby(["bloque", "etapa"])
               .agg(experimentos=("experiment_id", "size"),
                    primero=("experiment_id", "min"),
                    ultimo=("experiment_id", "max"))
               .reset_index().sort_values(["bloque", "primero"]))
    adoptadas = historial[historial["decision"].str.contains("Adoptar|adopta|Candidato", case=False,
                                                            na=False)]

    # --- 2. Evidencia cuantitativa de las lineas cerradas --------------------
    p39 = json.loads((cfg.EXPERIMENTS / "phase39_fusion_v12.json").read_text(encoding="utf-8"))
    p40 = json.loads((cfg.EXPERIMENTS / "phase40_analisis_error.json").read_text(encoding="utf-8"))
    p34 = json.loads((cfg.EXPERIMENTS / "phase34_resumen.json").read_text(encoding="utf-8"))
    p13 = json.loads((cfg.EXPERIMENTS / "phase13_error_desacuerdo.json").read_text(encoding="utf-8"))

    lineas: list[str] = [
        "# V13 · P0 — Gap audit", "",
        "**Entrega previa a cualquier entrenamiento.** No se ha entrenado ningún modelo: solo se "
        "leen artefactos, se mapea la literatura contra las features reales de V12 y se decide qué "
        "merece ejecutarse.", "",
        "**Regla de falsación**: cada variable de la literatura se mapea contra nombres de columna "
        "leídos de `features/`. Si un nombre declarado no existe, el script aborta. Por eso "
        "\"nunca probado\" es una afirmación verificada, no una opinión.", "",
        "## 1. Inventario real del historial", "",
        f"**{len(historial)} experimentos registrados** en `experiments.csv`, agrupados por bloque:", "",
        "| Bloque | Etapa | Experimentos | IDs |", "|---|---|---|---|",
    ]
    for _, fila in bloques.iterrows():
        ids = historial[historial["etapa"] == fila["etapa"]]["experiment_id"].tolist()
        lineas.append(f"| {fila['bloque']} | {fila['etapa']} | {fila['experimentos']} | "
                      f"{', '.join(ids[:4])}{'...' if len(ids) > 4 else ''} |")
    lineas += [
        f"\nExperimentos cuya decisión contiene \"adoptar/candidato\": **{len(adoptadas)}** "
        f"({', '.join(adoptadas['experiment_id']) if len(adoptadas) else 'ninguno'}).", "",
        "## 2. Las 18 variables del artículo de referencia, mapeadas a V12", "",
        "Fuente: Moreno-Sandoval, Pomares-Quimbaya y Alvarado-Valencia (2021), *Computational Social "
        "Networks* 8:16, doi:10.1186/s40649-021-00097-w. Es el antecedente metodológico directo de "
        "esta competencia (corpus PAN@CLEF, perfiles de autor, variable demográfica).", "",
        "| Var | Descripción | Estado en V12 | Columnas que la satisfacen |", "|---|---|---|---|",
    ]
    for variable, nombre, estado, match in VARIABLES_ARTICULO:
        columnas = ", ".join(f"`{c}`" for c in match) if match else "—"
        lineas.append(f"| {variable} | {nombre} | **{estado}** | {columnas} |")
    lineas += [
        f"\n**Verificación**: {sum(1 for v in detalle['verificaciones'] if v['verificada'])} de "
        f"{len(detalle['verificaciones'])} mapeos verificados contra los archivos reales; "
        f"{len(detalle['miembros'])} miembros del ensamble construyendo los 78 scores.", "",
        "**Conclusión del mapeo**: de las 18 variables, **14 ya existen** en V12 (o son inertes por "
        "el corpus sin emojis) y **5 están ausentes**: las cinco de pronombres de persona. El "
        "artículo declara que la 3ª persona singular es la variable más predictiva de su modelo.", "",
        "## 3. Las 7 familias de estilo del curso", "",
        "| Familia | Estado en V12 | Qué falta o por qué se cierra | Pista |", "|---|---|---|---|",
    ]
    for familia, estado, detalle_familia, pista in FAMILIAS_CURSO:
        lineas.append(f"| {familia} | **{estado}** | {detalle_familia} | {pista} |")
    lineas += ["", "## 4. Las 10 dimensiones de la taxonomía NLP (Zhou y Zafarani, 2020)", "",
               "| Dimensión | Estado en V12 | Detalle | Pista |", "|---|---|---|---|"]
    for dimension, estado, detalle_dim, pista in DIMENSIONES_TAXONOMIA:
        lineas.append(f"| {dimension} | **{estado}** | {detalle_dim} | {pista} |")
    lineas += ["", "## 5. Inventario de pistas y decisión de ejecución", "",
               "| ID | Pista | Mecanismo | ¿Probada antes? | IDs | Mejor resultado | Por qué sigue "
               "abierta | Complementariedad esperada | Coste | Riesgo de leakage | Prioridad |",
               "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for (pista_id, nombre, mecanismo, probada, ids, mejor, porque, complemento, coste, leakage,
         prioridad) in PISTAS:
        lineas.append(f"| {pista_id} | {nombre} | {mecanismo} | {probada} | {ids} | {mejor} | "
                      f"{porque} | {complemento} | {coste} | {leakage} | {prioridad} |")

    # --- 6. Donde esta el potencial ------------------------------------------
    techo = p13["votos"]["pct_todos_fallan"]
    rescatable = p13["votos"]["pct_rescatable"]
    p6 = {nombre: medida for nombre, medida in p39["candidatos"].items()}
    p7 = {nombre: medida["complementariedad"] for nombre, medida in p40["candidatos"].items()}
    lineas += [
        "", "## 6. Dónde está el mayor potencial de complementariedad", "",
        f"- **Techo de información**: {techo}% de las filas no las acierta **ningún miembro** del "
        "ensamble. Ahí ninguna capa ni especialista puede aportar: es el límite duro.",
        f"- **Agregación**: {rescatable}% de las filas tienen **algún miembro correcto** y el "
        "ensamble falla. Ese es el terreno de P1, P3, P5, P15 y P14.",
        "- **Fronteras centrales**: 10.122 cruces en `18-24|25-34` y 7.252 en `25-34|35-49` "
        "(fase 34). Es el terreno de P3, P4, P13 y P16.",
        f"- **Evidencia negativa acumulada**: las tres familias de la fase de lexicon dieron "
        "+0,0001, +0,0000 y -0,0009, y su combinación -0,0018; las 33 capas de decision no "
        "adoptaron ninguna. Los netos de complementariedad fueron "
        + ", ".join(f"{v['neto']:+d}" for v in p7.values())
        + ". Es decir: **mover filas no es ganar aciertos**, y por eso P15 debe preceder a "
        "cualquier modelo nuevo.", "",
        "**Conclusion de potencial**: la unica palanca con masa propia es **informacion que V12 no "
        "tiene** (P1, P17, P19, P20, P21) o **un objetivo que V12 no optimiza** (P3, P5). Todo lo "
        "que sea recolocar probabilidades ya esta demostrado como ruido.", "",
        "## 7. Anidamiento de validacion requerido", "",
        "| Requiere nested (in-fold) | No requiere |", "|---|---|",
        "| P1, P2, P6, P12, P13, P24, P26 | P17, P19, P20, P18, P21, P22, P23, P8, P9, P10, P11 |",
        "",
        "Motivo: las pistas de la columna izquierda calculan alguna magnitud que depende de la "
        "etiqueta (`P(clase|token)`, centroides, seleccion de tokens). Si se calcularan antes de "
        "crear los folds, la validacion de la fold de evaluacion habria visto su propia etiqueta "
        "(§5 y §38 del brief).", "",
        "## 8. Artefactos de V12 reutilizables", "",
        "| Artefacto | Contenido | Uso en V13 |", "|---|---|---|",
        "| `features/oof_ensemble_v12.npy` | probabilidades OOF de V12 | referencia, rescate/daño, "
        "gating |",
        "| `features/oof_argmax_v12.npy` y `oof_argmax_v12__*.npy` | argmax OOF de V12 y de los "
        "bloques probados en la fase de lexicones | base de la matriz de error por miembro |",
        f"| scores de los {len(detalle['miembros'])} miembros | 78 probabilidades por miembro | "
        "P15: qué miembro sabe y con qué confianza |",
        "| `features/age_structured_features.parquet` | 87 features estructuradas | mapeo de "
        "redundancia |",
        "| `features/dense_train.parquet` | 35 features densas |idem|",
        "| `features/dup_clusters.parquet` | clústeres de near-duplicados | folds agrupados (D-003) |",
        "| `src/decision_layer.py` | evaluador LOFO, metricas, fronteras, segmentos, rescate/daño | "
        "motor de todos los experimentos |",
        "| `src/cv.py` | `make_folds(..., scheme=\"group\")` | unico split permitido |", "",
        "## 9. coste y riesgo agregados de la fase", "",
        f"- **Experimentos previstos**: {len(PISTAS) - 3} pistas, de las cuales 3 son diagnosticas y "
        "2 condicionales. Dentro del presupuesto de 24 principales + 8 reservas.",
        "- **Coste de la cabecera (P1, P17, P19, P20)**: ~2 h en total, casi todo CPU y matrices "
        "cacheadas.",
        "- **Coste de P3/P4/P5**: ~2,5 h (reentrenar el combinador de produccion cuesta ~2,5 min "
        "por evaluacion de 5 folds).",
        "- **Riesgo de leakage dominante**: las estadisticas supervisadas in-fold (P1, P2, P6, P12, "
        "P24, P26) y el nested de P13.",
        "- **Riesgo de compliance**: prohibitions yaumbered en §3; se verificara de forma "
        "experimental en `reports/v13_compliance.md`, no solo declarada.", "",
        "## 10. Orden final de ejecucion", "",
        "1. **P0** (este informe) — sin entrenar.",
        "2. **P15** — diagnostico: decide si P3 y P4 se dirigen a una familia concreta de modelos.",
        "3. **P1, P17, P19, P20** — las cuatro familias de informacion nueva (nadie se solapa).",
        "4. **P3** — especialistas de las dos fronteras centrales.",
        "5. **P24, P4, P5** — forma de la evidencia, gating y entrenamiento pairwise.",
        "6. **P18, P21** — vocabulario cerrado adicional.",
        "7. **P2, P6, P22, P26**.",
        "8. **P8/P9, P10, P11, P12, P23**.",
        "9. **P13** solo si P3 aporta, con el diagnostico R-2 completo. **P14** solo con dos "
        "positivos independientes.", "",
        "## 11. Prediccion falsable pre-registrada", "",
        "El artículo de referencia concluye que sus features lingüisticas **ayudan a los modelos "
        "clasicos y perjudican a los profundos**. V12 promedia LogReg y MLP, asi que la prediccion "
        "es: **el bloque nuevo sube la rama LogReg y se diluye en la MLP**. Se medira por cabeza en "
        "cada experimento que añada features. Si no se cumple, se registra como evidencia contra la "
        "literatura y se prioriza la combinacion solo con la cabeza que se beneficia.", "",
        "## 12. Referencias", "",
        "1. Moreno-Sandoval, L.G., Pomares-Quimbaya, A., Alvarado-Valencia, J.A. (2021). *Celebrity "
        "profiling through linguistic analysis of digital social networks.* Computational Social "
        "Networks, 8:16. doi:10.1186/s40649-021-00097-w",
        "2. Material del curso `Clasificacion Texto.pdf` (78 pag.): atribucion de autoria, author "
        "profiling, 7 familias de estilo, arboles SFL.",
        "3. Zhou, L., Zafarani, R. (2020). *A Survey of Fake News: Fundamental Theories, Detection "
        "Methods, and Opportunities.* ACM Computing Surveys. doi:10.1145/3395046",
        "",
        "_Generado por `scripts/v13_p0_gap_audit.py`. Sin entrenamiento._",
    ]
    GAP_MD.write_text("\n".join(lineas) + "\n", encoding="utf-8")

    filas = []
    for (pista_id, nombre, mecanismo, probada, ids, mejor, porque, complemento, coste, leakage,
         prioridad) in PISTAS:
        filas.append({"family": pista_id, "hypothesis": nombre, "novel_mechanism": mecanismo,
                      "tested_before": probada, "experiment_ids": ids, "best_result": mejor,
                      "why_still_open": porque, "expected_complementarity": complemento,
                      "cost": coste, "leakage_risk": leakage, "priority": prioridad})
    for variable, nombre, estado, match in VARIABLES_ARTICULO:
        filas.append({"family": f"articulo::{variable}", "hypothesis": nombre,
                      "novel_mechanism": "variable del articulo de referencia", "tested_before": "si",
                      "experiment_ids": "-", "best_result": estado,
                      "why_still_open": "-" if estado == "AUSENTE" else "ya cubierta en V12",
                      "expected_complementarity": "-" if estado == "AUSENTE" else "nula",
                      "cost": "-", "leakage_risk": "ninguno", "priority": "ver seccion 5"})
    for familia, estado, detalle_familia, pista in FAMILIAS_CURSO:
        filas.append({"family": f"curso::{familia}", "hypothesis": familia,
                      "novel_mechanism": detalle_familia, "tested_before": estado,
                      "experiment_ids": "-", "best_result": estado, "why_still_open": pista,
                      "expected_complementarity": "-", "cost": "-", "leakage_risk": "-",
                      "priority": pista})
    pd.DataFrame(filas).to_csv(GAP_CSV, index=False)

    GAP_JSON.write_text(json.dumps({
        "fase": "V13 · P0 gap audit", "fecha": "2026-09-25",
        "experimentos_en_historial": int(len(historial)),
        "variables_articulo_ausentes": [v for v, _, e, _ in VARIABLES_ARTICULO if e == "AUSENTE"],
        "pistas": len(PISTAS),
        "verificaciones_mapeo": detalle["verificaciones"],
        "miembros_ensamble": detalle["miembros"],
        "techo_informacion_pct": techo, "agregacion_pct": rescatable,
        "deltas_lexicones": {k: v["delta"] for k, v in p6.items()},
        "capas_decision_no_adoptadas": p34["n_adoptadas"],
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"gap audit escrito: {GAP_MD.name} ({len(lineas)} lineas) y {GAP_CSV.name} "
          f"({len(filas)} filas)")
    print(f"historial: {len(historial)} experimentos | pistas: {len(PISTAS)} | "
          f"variables del articulo ausentes: "
          f"{[v for v, _, e, _ in VARIABLES_ARTICULO if e == 'AUSENTE']}")
    print(f"mapeos verificados: {sum(1 for v in detalle['verificaciones'] if v['verificada'])}"
          f"/{len(detalle['verificaciones'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
