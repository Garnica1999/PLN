"""age_linguistic_13_preregistro_v37.py

Fecha: 2026-10-01
Descripcion: pre-registro CONGELADO de V37 (diagnostico P(POS|edad), fases A-E) mas las
    primitivas numericas compartidas que consume el runner
    `scripts/age_linguistic_14_diagnostico_pos.py`.
Estado del arte aplicado: NAIVE BAYES multinomial con normalizacion por longitud (Laplace
    alpha = 1.0 fijo, priors empiricos, leave-one-fold-out), bootstrap IC95 por grupos de
    duplicados (2000 iteraciones, unidad = cluster, semilla 20261010), y control de fuga
    unilateral `A_perm <= 0.176667` mas `A_perm_val` (enmiendas E1 y E3).

Modos
-----
- Sin argumentos: comprueba los SHA256 contra los anclajes y escribe
  `experiments/age_linguistic_13_preregistro_v37.json`, INMUTABLE: si ya existe, lo verifica,
  no lo reescribe y avisa con codigo 2 si difiere.
- `--verificar`: no escribe nada. Corre las nueve aserciones de integridad del pre-registro:
  (1) SHA256 de los dos parquets POS; (2) SHA256 del corpus de train; (3) SHA256 de los seis
  canonicos contra su manifiesto; (4) SHA256 de `dup_clusters` y del manifiesto del plug-in;
  (5) constantes de gates, umbrales V1..V6 y G1..G4; (6) alfa, semillas y pesos del plug-in;
  (7) CORRIDA real de la sonda declarada E9 (A0, A1, A2, A_perm) comparada contra los valores
  declarados, con tolerancia 1e-6 para A1 y A2 y con el criterio unilateral de E1 mas la banda
  declarada para A_perm; (8) las nueve aserciones de integridad del entorno de datos de T0.2,
  incluida la reproduccion exacta de `fold_idx` con `StratifiedGroupKFold`; (9) inmutabilidad
  del propio fichero. Sale con 0 si pasan las nueve y con 2 nombrando la que falla.

Donde esta la verdad
--------------------
- Constantes de la spec y de las enmiendas: este archivo (modulo `CONSTANTES`).
- Ejecucion: `scripts/age_linguistic_14_diagnostico_pos.py`.

ASCII puro: sin acentos, sin emojis, sin CJK (regla 8.4 de la spec).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config as cfg  # noqa: E402
from src.verificacion_numerica import autotest_numerico  # noqa: E402

FECHA = "2026-10-01"
FASE = "V37-DIAG-POS"
N_CLASES = 6
N_TOKENS_TOTALES_ESPERADOS = 2_590_877
N_FILAS_ESPERADAS = 169_336
N_GRUPOS_ESPERADOS = 164_018
TAG_SPACE = "SPACE"
PREREG_JSON = cfg.EXPERIMENTS / "age_linguistic_13_preregistro_v37.json"

#: Inventario UPOS declarado a priori desde la union observada. No hay seleccion posterior.
ETIQUETAS_POS = [
    "ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM", "PART",
    "PRON", "PROPN", "PUNCT", "SCONJ", "SPACE", "SYM", "VERB", "X",
]
N_ETIQUETAS = len(ETIQUETAS_POS)
INDICE_ETIQUETA = {etiqueta: i for i, etiqueta in enumerate(ETIQUETAS_POS)}

CANONICO = ROOT / "artifacts" / "p33_scores" / "todos"
PARQUETS = {
    "md": ROOT / "features" / "linguistic" / "anotacion_md_train.parquet",
    "lg": ROOT / "features" / "linguistic" / "anotacion_lg_train.parquet",
}
DUP_CLUSTERS = ROOT / "features" / "dup_clusters.parquet"
CORPUS_TRAIN = ROOT / "data" / "raw" / "age_train.csv"
MANIFEST_PLUGIN = ROOT / "reports" / "age_submission_plugin_manifest.json"
MANIFEST_CANONICO = CANONICO / "manifest.json"
SPEC = ROOT / "plan" / "V37_DIAGNOSTICO_POS_PLAN.md"
ENMIENDAS = ROOT / "plan" / "V37_ENMIENDAS_01.md"

#: Anclajes de hash declarados. Si alguno cambia, la spec queda obsoleta y hay que parar.
ANCHAJES = {
    "features/linguistic/anotacion_md_train.parquet":
        "dacbc3055ccde0d9a53f335f65b63df666b2ac1a4eee4a94e5b46fd7cd8c5745",
    "features/linguistic/anotacion_lg_train.parquet":
        "81287d541c082ea888ad1b5d06b83680bc5078c624c2da32f0473139d67a739c",
    "artifacts/p33_scores/todos/y_true.npy":
        "b1c928f9b9f5575547c1090a1fd83cf32a048b694af35a980b460c0bd66d48e6",
    "artifacts/p33_scores/todos/fold_idx.npy":
        "74d18330c40c3598a1936706ae9434c5fdf29119d357d38353cf298812fe6fb5",
    "artifacts/p33_scores/todos/v30_prob.npy":
        "3d77352486605971093ade7214ae82326057845d8bf54a2d88c83843c5a721db",
    "artifacts/p33_scores/todos/cand_prob.npy":
        "57be29623543c2616011f8fc77aff89b71faecdad6b599f1b4f4b133655a8134",
    "artifacts/p33_scores/todos/cand_pred.npy":
        "2fd8ab0292e8b7364dd003ff3e23902b01786a612c3e9a57641d88788efca369",
    "artifacts/p33_scores/todos/ids.npy":
        "d69ecb339f1dc57de2f9cddd558658b5a272ffa42aec15c771773e8eabd59825",
    "artifacts/p33_scores/todos/manifest.json":
        "8881a9f53e8613aa943681e7fd9a42a3a281aa66b5d7c796d13d2758c6226076",
    "features/dup_clusters.parquet":
        "4c4ca6761e0bcf5b6ff93ae6a5c6e3e9f66426e788a5b813c0b6293b2e33cc22",
    "data/raw/age_train.csv":
        "39c6c8270366fe155f3c8dd232e31867bee659af8908121d04b6bd89bd163748",
    "reports/age_submission_plugin_manifest.json":
        "8d635e6a546b5a0e3829872b5e182be745910bd2a0945759a6a44b6eb7e643f7",
    "plan/V37_DIAGNOSTICO_POS_PLAN.md":
        "9e5c8e898199a8adf7f0b8c7cf123073fa09773e79a051e59307bba2bc46e510",
    "plan/V37_ENMIENDAS_01.md":
        "d4d48c18d7028c9d320893dbabf83b41602310015dde8beaafff9b5556aaa18c",
}


CONSTANTES = {
    # ---------------------------------------------------------------- modelo de la cota
    "modelo_cota": ("naive bayes multinomial sobre proporciones por token; sin ajustar, "
                    "sin tuneo, sin grilla"),
    "suavizado": {"tipo": "laplace", "alpha": 1.0,
                  "nota": "valor fijo declarado a priori; no hay grilla"},
    "vocab_seleccion": "ninguna",
    "priores": "frecuencia de clase empirica del train de cada fold",
    "estabilidad_numerica": {"clip_log_theta": [-30.0, 0.0], "eps_log_pi": 1e-12},
    "regla_filas_sin_tokens": ("x = 0 para las filas con n_tokens == 0 (0/0 indefinido), "
                                "luego pred = argmax_c pi_c; se incluyen y se cuentan "
                                "(enmienda E5)"),
    # ------------------------------------------------------------------ semillas
    "semillas": {"bootstrap_cluster": 20261010, "permute_control": 20261010,
                 "hash_agrupador": 0, "sgkf_folds": 42,
                 "muestra_correlacion_fase_c": 20261012},
    "n_bootstrap": 2000,
    "quintiles_longitud": "percentiles 20/40/60/80 de n_tokens sobre la particion canonica",
    # --------------------------------------------------------------------- suelos
    "piso_estricto_macro_f1": 0.0485421,
    "azar_uniforme_macro_f1": 0.166667,
    "margen_sobre_azar": 0.010,
    "criterio_a_perm_max": 0.176667,
    "criterio_a_perm_val_max": 0.176667,
    "banda_a_perm_esperada": [0.0485421, 0.166667],
    "baseline_primario": 0.6159732,
    "baseline_primario_exacto": 0.6159732487657944,
    "baseline_primario_nota": ("la spec escribe a la vez 0.6159732487657944 y '== 0.6159732 con "
                               "tolerancia 1e-9'; ambas cosas no pueden ser ciertas a la vez "
                               "(|0.6159732487657944 - 0.6159732| = 4.88e-8 > 1e-9). Se ancla el "
                               "valor EXACTO con tolerancia 1e-9 y se reporta el redondeo a 7 "
                               "decimales"),
    "pesos_plugin": [1.0, 1.1749, 1.2721, 1.163, 1.001, 0.8128],
    "b_criterio_eliminado": ("|A_perm - A0| <= 0.005 se ELIMINA en los cinco sitios de la "
                             "spec (enmienda E1); el unico criterio de A_perm es el unilateral "
                             "A_perm <= 0.176667"),
    # ------------------------------------------------------- umbrales de los criterios V1..V6
    "umbrales": {
        "V1": {"delta_min": 0.010, "n_etiquetas_min": 3, "n_clases_min": 4,
               "alfa_bonferroni": 0.05, "n_pruebas_por_clase": 18,
               "enunciado": "existe delta[c,t] >= 0.010 con z_bonferroni significativo "
                           "para al menos 3 etiquetas y al menos 4 de las 6 clases"},
        "V2": {"dispersion_max": 0.005,
               "enunciado": "esas mismas etiquetas cumplen dispersion[c,t] <= 0.005"},
        "V3": {"max_dc_md_lg_max": 0.005,
               "enunciado": "esas mismas etiquetas cumplen max_dc_md_lg[c,t] <= 0.005"},
        "V4": {"delta_estandarizado_min": 0.008,
               "enunciado": "la version estandarizada por longitud conserva "
                           "|p_est[c,t] - ref[t]| >= 0.008 para esas mismas etiquetas"},
        "V5": {"a1_min": 0.176667,
               "tipo": "sanity floor (enmienda E7; la novedad NO se decide con V5)",
               "enunciado": "A1 > azar_uniforme + 0.010 en md y lg"},
        "V6": {"a_perm_max": 0.176667, "a_perm_val_max": 0.176667,
               "tipo": "control de fuga unilateral (enmiendas E1 y E3)",
               "enunciado": "A_perm <= 0.176667 y A_perm_val <= 0.176667 en md y lg, y el "
                           "IC95 pareado de A1 - A_perm excluye cero"},
    },
    # ------------------------------------------------------------------------------ gates
    "gates": {
        "G1": {"soporte_min": 1000, "n_etiquetas_min": 12, "n_clases_min": 5,
               "criterio_fuga": "A_perm <= 0.176667 (E1)"},
        "G2": {"n_tokens_min": 5000, "dispersion_max": 0.005, "max_dc_md_lg_max": 0.005,
               "celdas_evaluables_min": 1},
        "G3": {"exigencia": "TC.1 + TC.2 + TC.3 completos; no tiene umbral de paso"},
        "G4": {"exigencia": "auditoria de novedad con las 5 razones de 6.2 y medicion D1 "
                            "completa; no autoriza ninguna corrida predictiva"},
    },
    # ------------------------------------------------------- sonda previa declarada (E9)
    "sonda_previa": {
        "declarada_antes_de_congelar": True,
        "A1": 0.1720532,
        "A2": 0.2052270,
        "A_perm": 0.06575,
        "tolerancia": 1e-06,
        "modelos": "los exactos de la seccion 5.2, sin ajustar nada",
        "nota": "A2 NO es identico a A1 (E2): con priors, A2 - A1 = +0.0331738 y 48440 filas "
                "cambian de prediccion; A2 es el brazo de sensibilidad a la longitud",
    },
    # -------------------------------------------------- regla de reproduccion declarada (E9)
    "regla_reproduccion_sonda_e9": {
        "A1_y_A2": "tolerancia 1e-6; si falla, STOP con codigo 2",
        "A_perm": (
            "El valor exacto de A_perm depende del flujo del RNG, que la spec NO fija (no dice "
            "si el generador se crea una vez o uno por fold, ni cual familia de RNG, ni si la "
            "permutacion es global o por fold). Por eso el criterio de integridad de A_perm no "
            "es la igualdad a 5 decimales sino el que MANDAN las enmiendas: E1 fija un unico "
            "criterio unilateral A_perm <= 0.176667 y obliga a reportar el valor dentro de la "
            "banda declarada [0.0485421, 0.166667] sin pass/fail en el extremo inferior. La "
            "corrida congelada exige: (i) A_perm <= 0.176667, (ii) A_perm dentro de la banda "
            "declarada, (iii) el valor medido y la diferencia frente a la sonda se reportan "
            "explicitas. Si (i) o (ii) falla, STOP con codigo 2 por fuga suspected"),
        "esquema_de_permutacion_declarado": (
            "un unico numpy.random.default_rng(20261010) creado antes del bucle de folds; para "
            "cada k se consume rng.permutation(n_train_k) y se reescriben las etiquetas de las "
            "filas de train_k. Las filas de validacion conservan su etiqueta real"),
    },
    # ------------------------------------------------------ auditoria de hdrechos de la sonda
    "banda_de_expectativa_de_A_perm": {
        "extremo_inferior": "A0 = 0.0485421 (prediccion constante de la mayoritaria)",
        "motivo_inferior": ("E1: al permutar las etiquetas de train_k se conserva el multisej y "
                            "por tanto los priors, pero theta[c,.] de cada clase es un "
                            "subconjunto aleatorio; las clases con menos tokens reciben mas "
                            "suavizado de Laplace y pierden verosimilitud, de modo que el "
                            "permutado DEGENERA HACIA LA CLASE MAYORITARIA en vez de caer al "
                            "azar uniforme"),
        "extremo_superior": "azar uniforme = 0.166667",
    },
    # ------------------------------------------------------ prueba concreta si el gate pasara
    "prueba_concreta_si_pasara": (
        "Si el patron resultara potente (V1..V6 los cinco true), la spec posterior que se "
        "ejecutaria esta escrita por adelantado y es esta: 'P(POS_{i+1} | POS_i, edad) con "
        "LogReg L2 C=1.0 sobre indicadores de uni/bi/trigramas POS multiplicados por un "
        "indicador de edad, LeaveOneOut por cluster, integrado con el campeon por alfa 0.5 con "
        "los pesos congelados, gate delta >= +0.0015 y 4/5 folds'. Esa prueba NO se ejecuta en "
        "V37: V37 no produce ninguna corrida predictiva (enmienda E8)"),
    # --------------------------------------------------------------------- referencias previas
    "referencias_previas": [
        {"acta": "D-026", "que_midio": "26 features POS con es_core_news_md: 18 proporciones "
         "UPOS + pos_unique + pos_entropy + diversidad de bigramas + de trigramas + "
         "content_ratio + function_ratio + verb_noun + adj_noun + pos_tokens",
         "numero": -0.0032, "tipo": "integracion", "unidades": "26 ratios por fila"},
        {"acta": "D-044/V14", "que_midio": "37 ratios agregados (morfologia verbal y nominal, "
         "lemas, entidades, sintaxis)", "aislado_logreg": 0.2409, "aislado_svc": 0.2270,
         "integracion": -0.0015, "folds_positivos": "1/5", "unidades": "37 ratios agregados"},
        {"acta": "V31", "que_midio": "canales regex sin tagger: pronombres 1P/2P/3P/plural, "
         "palabras funcionales, tiempo verbal, negacion, densidad",
         "banco": 0.006820, "integracion": 0.0000642, "unidades": "regex sobre forma lexical"},
        {"acta": "D-041/P1a", "que_midio": "P(clase | token) con suavizado empirico-bayesiano y "
         "cross-fit interno", "bloque_solo": 0.3997, "integracion": -0.0002,
         "folds_positivos": "3/5", "controles_permutados": [0.6000, 0.6003, 0.6000],
         "unidades": "direccion dual de la misma tabla"},
        {"acta": "D-077/P2", "que_midio": "pos_seq -> TfidfVectorizer(word, 1-3 grams, min_df=5, "
         "sublinear_tf) + LogReg C=1.0 por fold",
         "md_solo": 0.2750921, "lg_solo": 0.2725490, "integracion_md": -0.0086357,
         "integracion_lg": -0.0084168, "folds_positivos": "0/5",
         "control_barajado_filas": -0.0039528, "control_barajado_intra": -0.0080310,
         "unidades": "n-gramas 1-3 sin indexar por edad"},
        {"acta": "V13 P17/P20/P19", "que_midio": "pronombres / modalidad / arboles de "
         "conjunciones", "valores": [0.1918, 0.1270, 0.1539], "unidades": "regex"},
        {"acta": "D-078/D-079/D-080", "que_midio": "NER estructural, lemma_POS, texto tipado",
         "numero": "cerrados; 0 rescates de 9780 errores de alta confianza", "unidades": "-"},
        {"acta": "D-088/V36-E1", "que_midio": "bloque de transiciones de clase de caracter "
         "(C1-A)", "numero": "cerrada: senal real y especifica por documento, pero el campeon ya "
         "la expresa mejor", "unidades": "transiciones de caracter"},
        {"acta": "D-087", "que_midio": "techo alcanzable LB", "numero": 0.642, "unidades": "-"},
    ],
    # -------------------------------------------------------------- estado del arte (E14)
    "estado_del_arte": [
        {"referencia": "Flekova, L.; Gurevych, I. (2013). Can We Hide in the Web? Large Scale "
         "Simultaneous Age and Gender Author Profiling in Social Media. CLEF 2013 / PAN, "
         "CEUR-WS Vol-1179.", "dataset": "PAN-2012 AP", "metrica": "edad accuracy 0.57 ES / "
         "0.53 EN", "lectura": "0,57 ES / 0,53 EN son del sistema SOLO-EDAD de 3 GRUPOS "
         "(mayoritario 0,33), NO del conjunto de 6 clases; el conjunto de 6 clases llega a 0,38 ES "
         "y 0,29 EN con mayoritario 0,17. Las cifras de 236.600 / 75.900 documentos son de "
         "PAN-2012 y no aparecen en el paper. Cita clave: en espanol se prefieren trigramas y "
         "cuadrigramas POS"},
        {"referencia": "Cruz, F.L.; Haro, R.; Ortega, F.J. (2013). ITALICA at PAN 2013: An "
         "Ensemble Learning Approach to Author Profiling. CLEF 2013 / PAN, CEUR-WS Vol-1179.",
         "dataset": "PAN-2013 AP", "metrica": "edad 0.62 ES",
         "lectura": "el 0,62 es el ACCURACY del ensemble apilado COMPLETO (BoW + POS + "
         "estilisticos + JRip), NO del bloque POS. No es legitimo citarlo como 'los n-gramas "
         "POS dan 0,62'"},
        {"referencia": "Posadas-Duran, J.P. et al. (2015). Syntactic N-grams as Features for "
         "the Author Profiling Task. CLEF 2015 / PAN, CEUR-WS Vol-1391, paper 136.",
         "dataset": "PAN-2015 AP", "metrica": "RMSE por Rasch (habilidad)",
         "lectura": "negativo para edad: su uso no es muy exitoso para edad y genero"},
        {"referencia": "Modaresi, P.; Liebeck, M.; Conrad, S. (2016). Exploring the Effects of "
         "Cross-Genre Machine Learning for Author Profiling in PAN 2016. CLEF 2016, CEUR-WS "
         "Vol-1609, paper 970.", "dataset": "PAN-2016 AP", "metrica": "edad 0.5179 ES",
         "lectura": "negativo DIRECTO sobre el objeto de la fase A: la distribucion de POS "
         "L2-normalizada empeoro los resultados y el perfil POS no fue una buena feature "
         "independiente del genero"},
        {"referencia": "Johannsen, A.; Hovy, D.; Sogaard, A. (2015). Cross-lingual syntactic "
         "variation over age and gender. CoNLL 2015, ACL Anthology K15-1011.",
         "dataset": "Trustpilot, 11 idiomas", "metrica": "sin metrica de clasificacion",
         "lectura": "el paper reporta GENERO sobre clases lexicas BabelNet (adverbios, adjetivos), "
         "NO sobre treelets POS; se retira la afirmacion de 'tres treelets atomicos para "
         "genero'. Para edad el soporte es menor que para genero y no es pervasivo"},
        {"referencia": "Flekova, L.; Preotiuc-Pietro, D.; Ungar, L. (2016). Exploring Stylistic "
         "Variation with Age and Income on Twitter. ACL 2016, P16-2051.",
         "dataset": "4.279 usuarios con edad (574.095 tweets)", "metrica": "Pearson CV 10 folds",
         "lectura": "antecedente mas proximo a P(POS|edad): mide el ratio de cada POS con "
         "universal tagset. nouns/determiners/adjectives suben con la edad; pronombres e "
         "interjecciones tienen el signo clasico pero DEBIL. Unidad = usuario"},
        {"referencia": "Morgan-Lopez, A.A.; Kim, A.E.; Chew, R.F.; Ruddle, P. (2017). "
         "Predicting age groups of Twitter users based on language and metadata features. "
         "PLoS ONE 12(8):e0183537.", "dataset": "3.184 usuarios, hasta 200 tweets por usuario",
         "metrica": "F1 0.739 mejor modelo", "lectura": "limite de la evidencia publicada: 3 "
         "clases y POR USUARIO; aqui 6 clases y 1 tweet por fila, luego no es comparable"},
    ],
    "nota_estado_del_arte": (
        "Las cuatro referencias centrales se reencuadran como PRIOR ART con metricas de otra "
        "tarea (3 grupos y accuracy contra 6 clases y macro-F1, mayoritario 0,33 contra "
        "0,0485). Nunca como 'otros con mejores numeros' (enmienda E14). La cota de la seccion 5 "
        "es una construccion propia de este plan, cerrada y sin hiperparametros: no se presenta "
        "como replicar literatura"),
    # ------------------------------------------------------------------ familias no medidas
    "enmienda_E15_familias_cerradas": {
        "subtipos_UD_de_dep_seq": {"n_ratios": 22, "sonda": 0.2147},
        "morfologia_completa": {"n_ratios": 23, "sonda": 0.2064,
                                "campos": ["PronType", "Case", "VerbForm", "Degree"]},
        "sonda_SPACE_aislado": 0.04854,
        "nota": "las dos familias derivables de los artefactos congelados caen en la misma banda "
                "0.19-0.24 que las seis ya cerradas, muy por debajo de P2 (0.2751), y con fusion "
                "siempre <= 0. SPACE aislado da 0.04854 = exactamente la prediccion constante de "
                "la mayoritaria: su aporte es renormalizacion, no senal propia. No existe "
                "ninguna familia legal no medida con potencial",
    },
    # -------------------------------------------------------------------- boundaries
    "no_hacer": [
        "test, holdout, leaderboard o submission: nunca, ni para diagnostico",
        "transformers, LLM o embeddings, preentrenados o entrenados",
        "datos, etiquetas o lexicons externos",
        "metadatos (follower, perfil, timestamp)",
        "agregar por autor: un tweet por fila, siempre",
        "grillas de hiperparametros: ni alpha, ni C, ni subconjuntos de etiquetas ni de folds",
        "tocar test o submissions/",
        "reentrenar o reevaluar el campeon: V30 se lee, no se corre",
        "editar o borrar scripts ya ejecutados",
        "fabricar metricas: si algo no se midio, se dice que no se midio",
        "declarar que el paso (d) es nuevo cuando la auditoria dice lo contrario",
        "cambiar el pre-registro una vez congelado (se versiona como _v37b)",
    ],
    "rebalanceo": {"aplica": False, "ratio_max_min": 1.0627,
                   "motivo": "clases casi perfectamente balanceadas; la spec lo declara NO APLICA "
                             "para no introducir rebalanceo por inercia"},
}


# ==============================================================================
# Utilidades de bajo nivel
# ==============================================================================
def sha256_de_archivo(ruta: Path) -> str:
    """Devuelve el SHA256 de un archivo.

    Parameters
    ----------
    ruta : Path
        Ruta al archivo.

    Returns
    -------
    str
        Hexadecimal de 64 caracteres.
    """
    return hashlib.sha256(Path(ruta).read_bytes()).hexdigest()


def sha256_de_array(matriz: np.ndarray) -> str:
    """SHA256 de los bytes crudos de un array, en orden C.

    Parameters
    ----------
    matriz : numpy.ndarray
        Array del que se quiere fijar la huella.

    Returns
    -------
    str
        Hexadecimal de 64 caracteres.
    """
    return hashlib.sha256(np.ascontiguousarray(matriz).tobytes()).hexdigest()


def macro_f1(clase: np.ndarray, pred: np.ndarray,
             mascara: np.ndarray | None = None) -> float:
    """macro-F1 sobre las 6 clases, con el conjunto de etiquetas explicito.

    Parameters
    ----------
    clase, pred : numpy.ndarray
        Etiquetas verdaderas y predicciones, alineadas por fila.
    mascara : numpy.ndarray, opcional
        Subconjunto de filas sobre el que se evalua.

    Returns
    -------
    float
        macro-F1.
    """
    if mascara is not None:
        return float(f1_score(clase[mascara], pred[mascara], average="macro",
                              labels=list(range(N_CLASES))))
    return float(f1_score(clase, pred, average="macro", labels=list(range(N_CLASES))))


def macro_f1_de_conteos(conteos: np.ndarray) -> float:
    """macro-F1 a partir de una matriz de confusion 6x6 (para el bootstrap).

    Parameters
    ----------
    conteos : numpy.ndarray
        Vector de 36 enteros en orden ``clase_verdadera * 6 + clase_predicha``.

    Returns
    -------
    float
        macro-F1.
    """
    matriz = conteos.reshape(N_CLASES, N_CLASES).astype(np.float64)
    verdaderas = matriz.sum(axis=1)
    predichas = matriz.sum(axis=0)
    diagonal = np.diag(matriz)
    denominador = verdaderas + predichas
    f1s = np.where(denominador > 0, 2.0 * diagonal / np.where(denominador > 0, denominador, 1.0),
                   0.0)
    return float(f1s.mean())


# ==============================================================================
# Entradas: canonicos, clusters y anotaciones
# ==============================================================================
def carga_canonicos() -> dict:
    """Lee los `.npy` canonicos y devuelve el diccionario de partida de V37.

    Returns
    -------
    dict
        ``y_true``, ``fold_idx``, ``v30_prob``, ``ids``, ``pesos`` y ``baseline``.
    """
    y_true = np.load(CANONICO / "y_true.npy").astype(np.int64)
    fold_idx = np.load(CANONICO / "fold_idx.npy").astype(np.int64)
    v30_prob = np.load(CANONICO / "v30_prob.npy").astype(np.float64)
    ids = np.load(CANONICO / "ids.npy")
    pesos = np.asarray(CONSTANTES["pesos_plugin"], dtype=np.float64)
    camp_pred = np.argmax(v30_prob * pesos, axis=1)
    return {
        "y_true": y_true,
        "fold_idx": fold_idx,
        "v30_prob": v30_prob,
        "ids": ids,
        "pesos": pesos,
        "camp_pred": camp_pred,
        "baseline": macro_f1(y_true, camp_pred),
    }


def carga_clusters() -> tuple[np.ndarray, int]:
    """Devuelve el indice de cluster compacto por fila de train y el numero de grupos.

    Returns
    -------
    tuple
        ``(cluster_local, n_grupos)`` con ``cluster_local`` en ``0..n_grupos-1`` en el orden de
        las filas de train.
    """
    crudo = pd.read_parquet(DUP_CLUSTERS, columns=["cluster"])["cluster"].to_numpy()
    train = crudo[:N_FILAS_ESPERADAS]
    _, local = np.unique(train, return_inverse=True)
    return local.astype(np.int64), int(local.max()) + 1


def tokens_planos(pos_seq: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Aplana `pos_seq` a un vector de indices de etiqueta, con el id de fila por token.

    Parameters
    ----------
    pos_seq : numpy.ndarray
        Secuencias de etiquetas UPOS, una por fila.

    Returns
    -------
    tuple
        ``(tok_idx, tok_fila, longitudes)``: indice de etiqueta por token, fila duena y numero
        de tokens por fila.
    """
    largo = len(pos_seq)
    n_tokens = np.array([len(str(s).split()) for s in pos_seq], dtype=np.int64)
    total = int(n_tokens.sum())
    tok_idx = np.empty(total, dtype=np.int8)
    tok_fila = np.empty(total, dtype=np.int32)
    inicio = 0
    mapa = INDICE_ETIQUETA
    for i, secuencia in enumerate(pos_seq):
        etiquetas = str(secuencia).split()
        n = len(etiquetas)
        if n:
            tok_idx[inicio:inicio + n] = [mapa[e] for e in etiquetas]
            tok_fila[inicio:inicio + n] = i
        inicio += n
    return tok_idx, tok_fila, n_tokens


def matriz_conteos(anotador: str) -> dict:
    """Construye los conteos por fila de etiqueta y el vector de longitudes.

    Parameters
    ----------
    anotador : str
        ``"md"`` o ``"lg"``.

    Returns
    -------
    dict
        ``conteos`` (CSR ``(n_filas, 18)``), ``n_tokens``, ``pos_seq`` y las columnas crudas
        que hacen falta para las aserciones de integridad.
    """
    marco = pd.read_parquet(PARQUETS[anotador],
                            columns=["fila", "pos_seq", "n_tokens", "n_sents"])
    pos_seq = marco["pos_seq"].astype(str).to_numpy()
    n_tokens = marco["n_tokens"].to_numpy().astype(np.int64)
    largo = len(pos_seq)
    filas: list[int] = []
    columnas: list[int] = []
    valores: list[int] = []
    mapa = INDICE_ETIQUETA
    for i, secuencia in enumerate(pos_seq):
        etiquetas = secuencia.split()
        if not etiquetas:
            continue
        conteo: dict[int, int] = {}
        for etiqueta in etiquetas:
            j = mapa[etiqueta]
            conteo[j] = conteo.get(j, 0) + 1
        for j, valor in conteo.items():
            filas.append(i)
            columnas.append(j)
            valores.append(valor)
    conteos = sparse.csr_matrix((valores, (filas, columnas)),
                                shape=(largo, N_ETIQUETAS), dtype=np.float64)
    return {"conteos": conteos, "n_tokens": n_tokens, "pos_seq": pos_seq,
            "fila": marco["fila"].to_numpy(), "n_sents": marco["n_sents"].to_numpy().astype(
                np.int64)}


# ==============================================================================
# Fase A: perfil P(tag | edad) y cota de potencia
# ==============================================================================
def perfil_por_clase(conteos: np.ndarray, y: np.ndarray, filas_validas: np.ndarray,
                     n_clases: int = N_CLASES) -> tuple[np.ndarray, np.ndarray]:
    """Perfil ``p[c,t]`` crudo y numero de filas de la clase que contienen la etiqueta.

    Parameters
    ----------
    conteos : numpy.ndarray
        Matriz dispersa o densa de conteos por fila y etiqueta.
    y : numpy.ndarray
        Etiqueta verdadera por fila.
    filas_validas : numpy.ndarray
        Filas que entran en los denominadores (las de ``n_tokens == 0`` se excluyen).
    n_clases : int, opcional
        Numero de clases.

    Returns
    -------
    tuple
        ``(n_por_clase_y_tag, n_filas_por_clase_y_tag)`` con formas ``(n_clases, 18)``.
    """
    sub = conteos[filas_validas]
    y_sub = y[filas_validas]
    n_tokens = np.asarray(sub.sum(axis=1)).ravel()
    matriz_n = np.zeros((n_clases, N_ETIQUETAS), dtype=np.float64)
    matriz_filas = np.zeros((n_clases, N_ETIQUETAS), dtype=np.float64)
    for c in range(n_clases):
        mascara = y_sub == c
        if not mascara.any():
            continue
        bloque = sub[mascara]
        matriz_n[c] = np.asarray(bloque.sum(axis=0)).ravel()
        matriz_filas[c] = np.asarray((bloque > 0).sum(axis=0)).ravel()
    assert (n_tokens >= 0).all()
    return matriz_n, matriz_filas


def p_desde_conteos(matriz_n: np.ndarray) -> np.ndarray:
    """Convierte conteos por clase y etiqueta en proporciones que suman 1.

    Parameters
    ----------
    matriz_n : numpy.ndarray
        Conteos ``(n_clases, 18)``.

    Returns
    -------
    numpy.ndarray
        Proporciones ``(n_clases, 18)``; las clases con denominador 0 devuelven 0.
    """
    denominador = matriz_n.sum(axis=1, keepdims=True)
    seguro = np.where(denominador > 0, denominador, 1.0)
    return matriz_n / seguro


def p_sin_space(matriz_n: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Perfil excluyendo `SPACE` del numerador y del denominador (definicion de D-026).

    Parameters
    ----------
    matriz_n : numpy.ndarray
        Conteos ``(n_clases, 18)`` que ya incluyen `SPACE`.

    Returns
    -------
    tuple
        ``(p_sin_space, n_sin_space)`` con la misma forma que la entrada.
    """
    j = INDICE_ETIQUETA[TAG_SPACE]
    copia = matriz_n.copy()
    copia[:, j] = 0.0
    return p_desde_conteos(copia), copia


def estandariza_por_longitud(conteos: np.ndarray, y: np.ndarray, quintil: np.ndarray,
                             filas_validas: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Estandariza el perfil por longitud con quintiles de `n_tokens`.

    Parameters
    ----------
    conteos : numpy.ndarray
        Matriz de conteos por fila y etiqueta.
    y : numpy.ndarray
        Etiqueta verdadera por fila.
    quintil : numpy.ndarray
        Quintil de `n_tokens` por fila, en ``0..4``.
    filas_validas : numpy.ndarray
        Filas con ``n_tokens > 0``.

    Returns
    -------
    tuple
        ``(p_est, p_por_quintil)`` con formas ``(6, 18)`` y ``(5, 6, 18)``. El perfil
        estandarizado es ``sum_q w_q * p_q`` con ``w_q`` la frecuencia del estrato en el corpus.
    """
    y_sub = y[filas_validas]
    q_sub = quintil[filas_validas]
    p_por_quintil = np.zeros((5, N_CLASES, N_ETIQUETAS), dtype=np.float64)
    pesos = np.zeros(5, dtype=np.float64)
    for q in range(5):
        seleccion = q_sub == q
        pesos[q] = float(seleccion.sum())
        if not seleccion.any():
            continue
        # `filas_validas` son indices REALES de fila: se pasa `y` completo y el subconjunto
        # de filas del quintil, nunca `y` ya recortada (doble indexacion).
        matriz_n, _ = perfil_por_clase(conteos, y, filas_validas[seleccion])
        p_por_quintil[q] = p_desde_conteos(matriz_n)
    pesos /= pesos.sum()
    p_est = np.tensordot(pesos, p_por_quintil, axes=(0, 0))
    return p_est, p_por_quintil


def p_media_de_proporciones(conteos: np.ndarray, y: np.ndarray, filas_validas: np.ndarray,
                            ) -> np.ndarray:
    """Media de proporciones por tweet (la version opuesta a la razon de sumas, E12).

    Parameters
    ----------
    conteos : numpy.ndarray
        Matriz de conteos por fila y etiqueta.
    y : numpy.ndarray
        Etiqueta verdadera por fila.
    filas_validas : numpy.ndarray
        Filas con ``n_tokens > 0``.

    Returns
    -------
    numpy.ndarray
        ``(6, 18)`` con la media de las proporciones por tweet dentro de cada clase.
    """
    sub = conteos[filas_validas].toarray() if sparse.issparse(conteos) else conteos[filas_validas]
    totales = sub.sum(axis=1, keepdims=True)
    proporciones = sub / np.where(totales > 0, totales, 1.0)
    salida = np.zeros((N_CLASES, N_ETIQUETAS), dtype=np.float64)
    y_sub = y[filas_validas]
    for c in range(N_CLASES):
        seleccion = y_sub == c
        if seleccion.any():
            salida[c] = proporciones[seleccion].mean(axis=0)
    return salida


def se_y_z(p: np.ndarray, ref: np.ndarray, n_por_clase: np.ndarray, n_pooled: np.ndarray,
           alfa: float = 0.05) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Error estandar y z de la diferencia por clase frente al pooled (dos unidades).

    Parameters
    ----------
    p, ref : numpy.ndarray
        Perfil por clase ``(6, 18)`` y perfil pooled ``(18,)``.
    n_por_clase : numpy.ndarray
        Tokens por clase y etiqueta ``(6, 18)``.
    n_pooled : numpy.ndarray
        Tokens por etiqueta sobre las 6 clases ``(18,)``.
    alfa : float, opcional
        Nivel para la correccion de Bonferroni.

    Returns
    -------
    tuple
        ``(delta, se_token, z_token, z_bonferroni)``. El SE principal usa la unidad token
        (pooled, conservador) y se etiqueta como secundario (E11); la inferencia la gobierna
        el bootstrap por grupos de duplicados de la fase B.
    """
    delta = p - ref[None, :]
    p_seguro = np.clip(p, 1e-12, 1.0 - 1e-12)
    ref_seguro = np.clip(ref, 1e-12, 1.0 - 1e-12)
    n1 = np.maximum(n_por_clase, 1.0)
    n2 = np.maximum(n_pooled, 1.0)
    se = np.sqrt(p_seguro * (1 - p_seguro) / n1 + ref_seguro * (1 - ref_seguro) / n2)
    z = np.where(se > 0, delta / se, 0.0)
    critico = float(np.sqrt(2.0 * np.log(np.sqrt(2.0 * np.pi * alfa / N_ETIQUETAS) / alfa)))
    z_corr = z - critico
    return delta, se, z, z_corr


def naive_bayes_pos(conteos: np.ndarray, n_tokens: np.ndarray, y_evaluacion: np.ndarray,
                    fold_idx: np.ndarray, y_ajuste: np.ndarray | None = None,
                    modo: str = "proporciones", alpha: float = 1.0) -> dict:
    """Cota de potencia: NB multinomial con normalizacion por longitud, leave-one-fold-out.

    Para el fold ``k`` los priors y ``theta`` se estiman SOLO con las filas
    ``fold_idx != k`` y se aplican a las filas ``fold_idx == k``. Las filas con
    ``n_tokens == 0`` reciben ``x = 0`` y por tanto ``pred = argmax_c pi_c`` (E5).

    Parameters
    ----------
    conteos : numpy.ndarray
        Matriz dispersa o densa ``(n_filas, 18)`` de conteos por etiqueta.
    n_tokens : numpy.ndarray
        Longitud en tokens por fila.
    y_evaluacion : numpy.ndarray
        Etiqueta verdadera usada para puntuar.
    fold_idx : numpy.ndarray
        Fold canonico por fila.
    y_ajuste : numpy.ndarray, opcional
        Etiquetas con las que se ajusta ``theta`` y los priors. Si es ``None`` se usan
        ``y_evaluacion``. Es el mecanismo de los controles permutados.
    modo : str, opcional
        ``"proporciones"`` (A1) o ``"conteos"`` (A2).
    alpha : float, opcional
        Suavizado de Laplace, fijo.

    Returns
    -------
    dict
        ``prob`` ``(n_filas, 6)``, ``pred`` ``(n_filas,)``, ``n_predichas_por_fold`` y
        ``detalle_por_fold`` con ``log_pi``/``log_theta`` y sus huellas.
    """
    n_filas = conteos.shape[0]
    etiquetas_ajuste = y_evaluacion if y_ajuste is None else y_ajuste
    probabilidades = np.zeros((n_filas, N_CLASES), dtype=np.float64)
    pred = np.zeros(n_filas, dtype=np.int64)
    n_predichas = np.zeros(5, dtype=np.int64)
    detalle = []
    if modo not in ("proporciones", "conteos"):
        raise ValueError(f"modo no soportado: {modo}")
    for k in range(5):
        train = fold_idx != k
        val = fold_idx == k
        y_tr = etiquetas_ajuste[train]
        matriz_n = np.zeros((N_CLASES, N_ETIQUETAS), dtype=np.float64)
        for c in range(N_CLASES):
            seleccion = y_tr == c
            if seleccion.any():
                matriz_n[c] = np.asarray(conteos[train][seleccion].sum(axis=0)).ravel()
        theta = (matriz_n + alpha) / (matriz_n.sum(axis=1, keepdims=True) + alpha * N_ETIQUETAS)
        filas_clase = np.bincount(y_tr, minlength=N_CLASES).astype(np.float64)
        prior = filas_clase / filas_clase.sum()
        log_theta = np.clip(np.log(theta), -30.0, 0.0)
        log_prior = np.log(prior + 1e-12)
        assert np.isfinite(log_theta).all() and np.isfinite(log_prior).all()
        if modo == "proporciones":
            denominador = n_tokens[val].astype(np.float64)[:, None]
            x = conteos[val].toarray() if sparse.issparse(conteos[val]) else conteos[val]
            x = x / np.where(denominador > 0, denominador, 1.0)
        else:
            x = conteos[val]
        scores = (x @ log_theta.T if sparse.issparse(x) else x.dot(log_theta.T)) + log_prior
        pred[val] = np.argmax(scores, axis=1)
        probabilidades[val] = scores
        n_predichas[k] = int(val.sum())
        detalle.append({
            "fold": k,
            "n_train": int(train.sum()),
            "n_val": int(val.sum()),
            "n_train_por_clase": [int(v) for v in filas_clase],
            "log_prior": [float(v) for v in log_prior],
            "sha256_log_prior": sha256_de_array(log_prior),
            "sha256_log_theta": sha256_de_array(log_theta),
            "n_recortes_log_theta": int((np.log(theta) < -30.0).sum()),
        })
    probabilidades = probabilidades - probabilidades.max(axis=1, keepdims=True)
    probabilidades = np.exp(probabilidades)
    probabilidades /= probabilidades.sum(axis=1, keepdims=True)
    return {"prob": probabilidades, "pred": pred, "n_predichas_por_fold": n_predichas,
            "detalle_por_fold": detalle}


def permuta_etiquetas_train(y: np.ndarray, fold_idx: np.ndarray, semilla: int) -> np.ndarray:
    """Permuta las etiquetas de las filas de `train_k` dentro de cada fold.

    Esquema DECLARADO: un unico ``numpy.random.default_rng(semilla)`` creado antes del bucle;
    para cada ``k`` se consume ``rng.permutation(n_train_k)`` y se reescriben las etiquetas de
    las filas de ``train_k``. Las filas de validacion conservan su etiqueta real.

    Parameters
    ----------
    y : numpy.ndarray
        Etiqueta verdadera por fila.
    fold_idx : numpy.ndarray
        Fold canonico por fila.
    semilla : int
        Semilla del control de permutacion.

    Returns
    -------
    numpy.ndarray
        Vector de etiquetas permutadas, con el multisej intacto dentro de cada fold.
    """
    generador = np.random.default_rng(semilla)
    salida = y.copy()
    for k in range(5):
        filas = np.flatnonzero(fold_idx != k)
        salida[filas] = y[filas][generador.permutation(len(filas))]
    return salida


def permuta_etiquetas_por_cluster(y: np.ndarray, cluster_local: np.ndarray,
                                  semilla: int) -> np.ndarray:
    """Permuta ``y`` a nivel de cluster de duplicados (control del lado de la evaluacion).

    Se sortea una etiqueta por cluster y se asigna a todas sus filas, de modo que la
    estructura de duplicados interna se conserva.

    Parameters
    ----------
    y : numpy.ndarray
        Etiqueta verdadera por fila.
    cluster_local : numpy.ndarray
        Cluster compacto por fila.
    semilla : int
        Semilla de la permutacion.

    Returns
    -------
    numpy.ndarray
        Copia de ``y`` con las etiquetas permutadas por cluster.
    """
    n_grupos = int(cluster_local.max()) + 1
    generador = np.random.default_rng(semilla)
    etiqueta_por_cluster = y.copy()[np.arange(n_grupos)]
    etiqueta_por_cluster = etiqueta_por_cluster[generador.permutation(n_grupos)]
    return etiqueta_por_cluster[cluster_local]


def mitad_por_hash(cluster_local: np.ndarray, semilla: int = 0) -> np.ndarray:
    """Parte las filas en dos mitades disjuntas por un hash DECLARADO del id de cluster.

    El hash es ``splitmix64(cluster_id + semilla)`` y se toma el bit bajo: es determinista,
    no usa aleatoriedad de datos y no depende del fold.

    Parameters
    ----------
    cluster_local : numpy.ndarray
        Cluster compacto por fila.
    semilla : int, opcional
        Semilla declarada del hash (0 en la spec).

    Returns
    -------
    numpy.ndarray
        ``0`` o ``1`` por fila, con los dos lados disjuntos y de tamano comparable.
    """
    x = cluster_local.astype(np.uint64) + np.uint64(semilla) * np.uint64(0x9E3779B97F4A7C15)
    x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    x = x ^ (x >> np.uint64(31))
    return (x & np.uint64(1)).astype(np.int8)


# ==============================================================================
# Bootstrap por grupos de duplicados
# ==============================================================================
def matriz_confusion_por_cluster(y: np.ndarray, pred: np.ndarray, cluster_local: np.ndarray,
                                 n_grupos: int) -> np.ndarray:
    """Matriz de confusion agregada por cluster de duplicados.

    Parameters
    ----------
    y, pred : numpy.ndarray
        Etiqueta verdadera y prediccion por fila.
    cluster_local : numpy.ndarray
        Cluster compacto por fila.
    n_grupos : int
        Numero de grupos.

    Returns
    -------
    numpy.ndarray
        ``(n_grupos, 36)`` en ``float32`` con los 36 conteos por grupo.
    """
    clave = cluster_local * (N_CLASES * N_CLASES) + y * N_CLASES + pred
    conteos = np.bincount(clave, minlength=n_grupos * N_CLASES * N_CLASES)
    return conteos.reshape(n_grupos, N_CLASES * N_CLASES).astype(np.float32)


def ic95_macro_f1_por_grupo(y: np.ndarray, predicciones: dict[str, np.ndarray],
                            cluster_local: np.ndarray, n_grupos: int, filas_por_cluster: np.ndarray,
                            semilla: int, n_iter: int) -> dict:
    """IC95 pareado de macro-F1 de varios brazos a la vez, por bootstrap de clusters.

    Todos los brazos comparten las **mismas** remuestreos, luego las diferencias son pareadas
    por construccion.

    Parameters
    ----------
    y : numpy.ndarray
        Etiqueta verdadera por fila.
    predicciones : dict
        ``nombre -> pred``. Cada fila recibe exactamente una prediccion.
    cluster_local : numpy.ndarray
        Cluster compacto por fila.
    n_grupos : int
        Numero de grupos.
    filas_por_cluster : numpy.ndarray
        Numero de filas por grupo, para el control de cobertura de grupos.
    semilla : int
        Semilla declarada (20261010).
    n_iter : int
        Numero de iteraciones declaradas (2000).

    Returns
    -------
    dict
        ``por_brazo`` con punto, IC95 y desviacion; ``diferencias`` pareadas; y
        ``control_cobertura_grupos`` con el numero de filas efectivas por iteracion.
    """
    nombres = list(predicciones)
    bloques = [matriz_confusion_por_cluster(y, predicciones[nombre], cluster_local, n_grupos)
               for nombre in nombres]
    matriz = np.concatenate(bloques, axis=1)
    generador = np.random.default_rng(semilla)
    muestras = {nombre: np.empty(n_iter, dtype=np.float64) for nombre in nombres}
    n_filas_efectivas = np.empty(n_iter, dtype=np.float64)
    for b in range(n_iter):
        indices = generador.integers(0, n_grupos, n_grupos)
        peso = np.bincount(indices, minlength=n_grupos).astype(np.float32)
        totales = matriz.T @ peso
        for j, nombre in enumerate(nombres):
            muestras[nombre][b] = macro_f1_de_conteos(totales[j * 36:(j + 1) * 36])
        n_filas_efectivas[b] = float(peso @ filas_por_cluster)
    lo, hi = np.percentile(np.vstack([muestras[n] for n in nombres]), [2.5, 97.5], axis=1)
    por_brazo = {}
    for j, nombre in enumerate(nombres):
        por_brazo[nombre] = {
            "bootstrap_media": round(float(muestras[nombre].mean()), 8),
            "ic95": [round(float(lo[j]), 7), round(float(hi[j]), 7)],
            "desviacion_tipica": round(float(muestras[nombre].std(ddof=1)), 8),
        }
    diferencias = {}
    for a in range(len(nombres)):
        for b_ in range(a + 1, len(nombres)):
            clave = f"{nombres[a]}_menos_{nombres[b_]}"
            d = muestras[nombres[a]] - muestras[nombres[b_]]
            lo_d, hi_d = np.percentile(d, [2.5, 97.5])
            diferencias[clave] = {"ic95": [round(float(lo_d), 7), round(float(hi_d), 7)],
                                  "excluye_cero": bool(lo_d > 0 or hi_d < 0),
                                  "p_delta_positivo": round(float((d > 0).mean()), 4)}
    n_filas_ref = float(filas_por_cluster.sum())
    dentro = np.abs(n_filas_efectivas - n_filas_ref) / n_filas_ref <= 0.05
    return {
        "n_iter": int(n_iter),
        "semilla": int(semilla),
        "unidad": "cluster de duplicado",
        "n_grupos": int(n_grupos),
        "por_brazo": por_brazo,
        "diferencias": diferencias,
        "control_cobertura_grupos": {
            "n_filas_efectivas_min": round(float(n_filas_efectivas.min()), 1),
            "n_filas_efectivas_max": round(float(n_filas_efectivas.max()), 1),
            "n_filas_efectivas_ref": round(n_filas_ref, 1),
            "frac_iteraciones_dentro_de_mas_menos_5pct": round(float(dentro.mean()), 6),
        },
    }


def perfil_por_cluster(conteos: np.ndarray, y: np.ndarray, cluster_local: np.ndarray,
                       n_grupos: int) -> np.ndarray:
    """Estadistico suficiente de ``p[c,t]`` y total de tokens por grupo.

    Columns: ``clase * 19 + t`` para ``t`` en ``0..17`` y ``clase * 19 + 18`` para el total de
    tokens de la clase. La spec escribe ``(n_grupos, 115)`` y ``[:, :18]``, que es
    inconsistente con ``celda = clase * 19 + t``; se declara aqui la version consistente
    (``n_grupos, 6 * 19 = 114``) que es la que reproduce la formula del perfil.

    Parameters
    ----------
    conteos : numpy.ndarray
        Matriz de conteos por fila y etiqueta.
    y : numpy.ndarray
        Etiqueta verdadera por fila.
    cluster_local : numpy.ndarray
        Cluster compacto por fila.
    n_grupos : int
        Numero de grupos.

    Returns
    -------
    numpy.ndarray
        ``(n_grupos, 114)`` en ``float32``.
    """
    largo = N_CLASES * 19
    totales = np.asarray(conteos.sum(axis=1)).ravel()
    base = cluster_local * largo + y * 19
    bloque = conteos.tocoo()
    salida = np.zeros(n_grupos * largo, dtype=np.float64)
    np.add.at(salida, base[bloque.row] + bloque.col, bloque.data)
    np.add.at(salida, base + 18, totales)
    return salida.reshape(n_grupos, largo).astype(np.float32)


def ic95_delta_por_grupo(perfil_cluster: np.ndarray, n_iter: int, semilla: int) -> dict:
    """IC95 pareado de ``delta[c,t] = p[c,t] - ref[t]`` por bootstrap de clusters.

    Parameters
    ----------
    perfil_cluster : numpy.ndarray
        ``(n_grupos, 114)`` devuelto por :func:`perfil_por_cluster`.
    n_iter : int
        Iteraciones declaradas.
    semilla : int
        Semilla declarada.

    Returns
    -------
    dict
        ``p`` ``(6, 18)`` del punto, ``ic95_delta`` ``(6, 18)`` con pares, ``excluye_cero``
        ``(6, 18)`` booleano y ``delta_fuera_de_ic`` ``(6, 18)`` booleano.
    """
    n_grupos = perfil_cluster.shape[0]
    largo = N_CLASES * 19
    generador = np.random.default_rng(semilla)
    deltas = np.empty((n_iter, N_CLASES, N_ETIQUETAS), dtype=np.float32)
    for b in range(n_iter):
        indices = generador.integers(0, n_grupos, n_grupos)
        peso = np.bincount(indices, minlength=n_grupos).astype(np.float32)
        totales = (perfil_cluster.T @ peso).reshape(N_CLASES, 19)
        p = totales[:, :N_ETIQUETAS] / np.where(totales[:, 18:19] > 0, totales[:, 18:19], 1.0)
        ref = p.sum(axis=0) / N_CLASES
        deltas[b] = p - ref[None, :]
    lo = np.percentile(deltas, 2.5, axis=0)
    hi = np.percentile(deltas, 97.5, axis=0)
    punto = deltas.mean(axis=0)
    return {"ic95_delta": np.stack([lo, hi], axis=-1).tolist(),
            "excluye_cero": ((lo > 0) | (hi < 0)).tolist(),
            "delta_fuera_de_ic": ((punto < lo) | (punto > hi)).tolist(),
            "bootstrap_media_delta": punto.tolist()}


# ==============================================================================
# Verificaciones
# ==============================================================================
def asercion(nombre: str, condicion: bool, detalle: str, resultados: dict) -> None:
    """Registra el resultado de una asercion y lanza ``AssertionError`` si falla.

    Parameters
    ----------
    nombre : str
        Identificador de la asercion.
    condicion : bool
        Resultado de la comprobacion.
    detalle : str
        Texto con la cifra medida, para que el informe sea trazable.
    resultados : dict
        Diccionario acumulador ``nombre -> {ok, detalle}``.
    """
    resultados[nombre] = {"ok": bool(condicion), "detalle": detalle}
    if not condicion:
        raise AssertionError(f"[STOP] asercion {nombre} falla: {detalle}")


def verificar_integridad(silencioso: bool = False) -> dict:
    """Corre las nueve aserciones de integridad del entorno de datos (T0.2 de la spec).

    Returns
    -------
    dict
        ``ok``, ``n_ok``, ``n_total`` y ``aserciones``.

    Raises
    ------
    AssertionError
        Si alguna asercion falla.
    """
    a: dict = {}
    canonico = carga_canonicos()
    y = canonico["y_true"]
    fold_idx = canonico["fold_idx"]
    cluster_local, n_grupos = carga_clusters()
    marco = pd.read_csv(CORPUS_TRAIN, sep=cfg.TRAIN_SEP, usecols=["id", cfg.TARGET],
                        encoding="utf-8-sig")
    mapa = {clase: i for i, clase in enumerate(cfg.CLASSES)}

    # (a), (b) y (c) de T0.2 son UNA asercion cada una, y cubren md y lg a la vez: por eso el
    # total es 9 y no 12. Se recorren los dos anotadores y se acumulan los detalles.
    longitudes_ok, filas_ok, etiquetas_ok = True, True, True
    detalles_a, detalles_b, detalles_c = [], [], []
    for anotador in ("md", "lg"):
        datos = matriz_conteos(anotador)
        sec = np.array([len(s.split()) for s in datos["pos_seq"]], dtype=np.int64)
        coincide = bool((sec == datos["n_tokens"]).all())
        longitudes_ok = longitudes_ok and coincide
        detalles_a.append(f"{anotador}: {int((sec == datos['n_tokens']).sum())}/{len(sec)} filas")
        es_arange = bool(np.array_equal(datos["fila"], np.arange(len(datos["fila"]))))
        filas_ok = filas_ok and es_arange
        detalles_b.append(f"{anotador}: {'exacto' if es_arange else 'NO'}")
        observadas: set[str] = set()
        for secuencia in datos["pos_seq"]:
            observadas.update(secuencia.split())
        coincide_c = len(observadas) == N_ETIQUETAS and sorted(observadas) == sorted(ETIQUETAS_POS)
        etiquetas_ok = etiquetas_ok and coincide_c
        detalles_c.append(f"{anotador}: n_tags = {len(observadas)}")
    asercion("a_len_pos_seq_igual_n_tokens_100pct_md_y_lg", longitudes_ok,
             " | ".join(detalles_a), a)
    asercion("b_fila_es_arange_md_y_lg", filas_ok, " | ".join(detalles_b), a)
    asercion("c_n_tags_18_en_md_y_lg", etiquetas_ok,
             " | ".join(detalles_c) + f"; inventario declarado {N_ETIQUETAS}", a)

    asercion("d_y_true_coincide_age_range",
             bool(np.array_equal(marco[cfg.TARGET].map(mapa).to_numpy().astype(np.int64), y)),
             f"age_range mapeado == y_true en {len(y)} filas", a)
    asercion("e_ids_coincide_train_id",
             bool(np.array_equal(marco["id"].to_numpy(), canonico["ids"])),
             "ids.npy == train['id'] en el orden de las filas", a)
    conteos_fold = np.bincount(fold_idx, minlength=5)
    asercion("f_fold_idx_5_valores_balanceados",
             conteos_fold.tolist() == [33868, 33868, 33867, 33866, 33867],
             f"bincount(fold_idx) = {conteos_fold.tolist()}", a)

    inicio = time.time()
    sg = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    reconstruido = np.empty(len(y), dtype=np.int64)
    for k, (_, val) in enumerate(sg.split(np.zeros(len(y)), y, cluster_local)):
        reconstruido[val] = k
    coincidencia = float((reconstruido == fold_idx).mean())
    segundos_sgkf = round(time.time() - inicio, 1)
    asercion("g_sgkf_reproduce_fold_idx", coincidencia == 1.0,
             f"coincidencia = {coincidencia:.6f} ({segundos_sgkf} s); "
             f"n_grupos = {n_grupos}", a)
    asercion("h_ancla_campeon",
             abs(canonico["baseline"] - CONSTANTES["baseline_primario_exacto"]) <= 1e-9,
             f"argmax(v30_prob*pesos) = {canonico['baseline']:.16f} "
             f"(anclaje exacto {CONSTANTES['baseline_primario_exacto']}, tolerancia 1e-9; "
             f"redondeo a 7 decimales {CONSTANTES['baseline_primario']})", a)
    minimo = int(np.bincount(y[fold_idx != k], minlength=N_CLASES).min())
    asercion("i_train_por_fold_20k_por_clase", minimo >= 20000,
             f"minimo de filas por clase en el train de un fold = {minimo}", a)

    if not silencioso:
        for nombre, dato in a.items():
            print(f"  [{'OK' if dato['ok'] else 'FALLO'}] {nombre}: {dato['detalle']}", flush=True)
    return {"ok": True, "n_ok": len(a), "n_total": 9, "segundos_sgkf": segundos_sgkf,
            "aserciones": a}


def corre_sondas_e9(silencioso: bool = False) -> dict:
    """Corre los tres brazos de la sonda previa declarada (E9) y compara con el pre-registro.

    Parameters
    ----------
    silencioso : bool, opcional
        Si es ``True`` no imprime.

    Returns
    -------
    dict
        Valores medidos, diferencias, criterios de integridad y el veredicto de reproduccion.
    """
    canonico = carga_canonicos()
    y = canonico["y_true"]
    fold_idx = canonico["fold_idx"]
    datos = matriz_conteos("md")
    conteos = datos["conteos"]
    n_tokens = datos["n_tokens"]

    a0_pred = np.full(len(y), int(np.argmax(np.bincount(y, minlength=N_CLASES))), dtype=np.int64)
    a0 = macro_f1(y, a0_pred)
    brazo1 = naive_bayes_pos(conteos, n_tokens, y, fold_idx, modo="proporciones")
    a1 = macro_f1(y, brazo1["pred"])
    brazo2 = naive_bayes_pos(conteos, n_tokens, y, fold_idx, modo="conteos")
    a2 = macro_f1(y, brazo2["pred"])
    y_perm = permuta_etiquetas_train(y, fold_idx, CONSTANTES["semillas"]["permute_control"])
    brazo_perm = naive_bayes_pos(conteos, n_tokens, y, fold_idx, y_ajuste=y_perm,
                                 modo="proporciones")
    a_perm = macro_f1(y, brazo_perm["pred"])

    sonda = CONSTANTES["sonda_previa"]
    tol = sonda["tolerancia"]
    d1, d2 = abs(a1 - sonda["A1"]), abs(a2 - sonda["A2"])
    banda = CONSTANTES["banda_a_perm_esperada"]
    integridad = {
        "A1_dentro_de_tolerancia": bool(d1 <= tol),
        "A2_dentro_de_tolerancia": bool(d2 <= tol),
        "A_perm_cumple_criterio_unilateral": bool(a_perm <= CONSTANTES["criterio_a_perm_max"]),
        "A_perm_dentro_de_banda_declarada": bool(banda[0] <= a_perm <= banda[1]),
    }
    veredicto = "REPRODUCE" if all(integridad.values()) else "NO_REPRODUCE"
    salida = {
        "anotador": "md",
        "medidos": {"A0": round(a0, 8), "A1": round(a1, 8), "A2": round(a2, 8),
                    "A_perm": round(a_perm, 8)},
        "declarados": {"A0": CONSTANTES["piso_estricto_macro_f1"],
                       "A1": sonda["A1"], "A2": sonda["A2"], "A_perm": sonda["A_perm"]},
        "diferencias": {"A0": round(abs(a0 - CONSTANTES["piso_estricto_macro_f1"]), 9),
                        "A1": round(d1, 9), "A2": round(d2, 9),
                        "A_perm": round(abs(a_perm - sonda["A_perm"]), 9)},
        "tolerancia": tol,
        "integridad": integridad,
        "veredicto_reproduccion": veredicto,
        "n_filas_que_cambian_A1_vs_A2": int((brazo1["pred"] != brazo2["pred"]).sum()),
        "A_perm_predice_clase0_en_filas": int(np.bincount(brazo_perm["pred"],
                                                          minlength=N_CLASES)[0]),
        "A2_menos_A1": round(a2 - a1, 7),
    }
    if not silencioso:
        print(f"  sonda E9 (md): A0={a0:.7f} A1={a1:.7f} A2={a2:.7f} A_perm={a_perm:.7f}",
              flush=True)
        print(f"  declarados  : A0={CONSTANTES['piso_estricto_macro_f1']} A1={sonda['A1']} "
              f"A2={sonda['A2']} A_perm={sonda['A_perm']}", flush=True)
        print(f"  integridad  : {integridad} -> {veredicto}", flush=True)
    return salida


def verifica_anclajes() -> dict:
    """Recomputa los SHA256 declarados y los compara con los anclajes.

    Returns
    -------
    dict
        ``ok`` y el detalle ``ruta -> {anclaje, medido}``.
    """
    detalle = {}
    ok = True
    for relativo, anclaje in ANCHAJES.items():
        medido = sha256_de_archivo(ROOT / relativo)
        coincide = medido == anclaje
        ok = ok and coincide
        detalle[relativo] = {"anclaje": anclaje, "medido": medido, "coincide": coincide}
    return {"ok": bool(ok), "detalle": detalle}


# ==============================================================================
# Pre-registro
# ==============================================================================
def construye_pre_registro() -> dict:
    """Construye el diccionario del pre-registro congelado.

    Returns
    -------
    dict
        Pre-registro completo, con anclajes, constantes, referencias,|no_hacer y sonda previa.
    """
    anclajes = verifica_anclajes()
    manifest = json.loads(MANIFEST_CANONICO.read_text(encoding="utf-8"))
    pre_registro = {
        "fase": FASE,
        "fecha": FECHA,
        "estado": "congelado antes de calcular nada; inmutable una vez escrito",
        "spec": str(SPEC.relative_to(ROOT)),
        "spec_sha256": anclajes["detalle"][str(SPEC.relative_to(ROOT))]["medido"],
        "enmiendas": str(ENMIENDAS.relative_to(ROOT)),
        "enmiendas_sha256": anclajes["detalle"][str(ENMIENDAS.relative_to(ROOT))]["medido"],
        "enmiendas_aplicadas": ["E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8", "E9", "E10",
                                "E11", "E12", "E13", "E14", "E15"],
        "hashes": {relativo: dato["medido"] for relativo, dato in anclajes["detalle"].items()},
        "hashes_coinciden_con_anclajes": anclajes["ok"],
        "manifest_canonico_sha256_declarados": manifest["sha256"],
        "etiquetas_pos": ETIQUETAS_POS,
        "n_etiquetas": N_ETIQUETAS,
        "vocab_seleccion": CONSTANTES["vocab_seleccion"],
        "anotadores": ["md", "lg"],
        "n_clases": N_CLASES,
        "clases": list(cfg.CLASSES),
        "constantes": CONSTANTES,
        "etiqueta_space": TAG_SPACE,
        "unidades_muestreo": {
            "modelo": "un tweet por fila",
            "bootstrap": "cluster de duplicado (164018 grupos)",
            "error_analitico": "tweet; se etiqueta como secundario (E11)",
            "prohibido": "agregar por autor",
        },
        "no_hacer": CONSTANTES["no_hacer"],
        "reproduccion_folds": {
            "definicion": "StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42) "
                          "sobre dup_clusters['cluster'][:169336]",
            "uso": "los folds se LEEN de fold_idx.npy; la reconstruccion es solo una asercion",
            "si_no_reproduce": "[STOP] codigo 2 y se usa el .npy leido",
        },
        "auditoria_de_novedad": {
            "canal_predictivo_del_paso_d": "NO SE EJECUTA",
            "medicion_descriptiva_d1": "SI se ejecuta: divergencia entre clases frente a ruido "
                                        "de estimacion, con IC95 pareado por bootstrap de "
                                        "grupos de duplicados",
            "autorizacion_para_habilitar": "autorizacion expresa de Carlos con hipotesis nueva "
                                           "escrita por adelantado",
        },
    }
    return pre_registro


def escribe_pre_registro(forzar: bool = False) -> tuple[dict, str]:
    """Escribe el pre-registro si no existe; si existe, verifica que coincide y no reescribe.

    Parameters
    ----------
    forzar : bool, opcional
        Si es ``True`` escribe siempre. **No se usa en la corrida normal**: el pre-registro
        es inmutable.

    Returns
    -------
    tuple
        ``(pre_registro, estado)`` con estado en ``"escrito"``, ``"verificado_sin_reescribir"``
        o ``"difiere"``.
    """
    nuevo = construye_pre_registro()
    texto = json.dumps(nuevo, indent=2, ensure_ascii=True, sort_keys=False) + "\n"
    if PREREG_JSON.exists() and not forzar:
        previo = PREREG_JSON.read_text(encoding="utf-8")
        if previo == texto:
            return json.loads(previo), "verificado_sin_reescribir"
        previo_obj = json.loads(previo)
        diferencias = [k for k in previo_obj if previo_obj.get(k) != nuevo.get(k)]
        print("  [STOP] el pre-registro existente difiere del que se construiria.", flush=True)
        print(f"         claves que difieren: {diferencias}", flush=True)
        print("         no se reescribe: el pre-registro es inmutable (se versiona _v37b).",
              flush=True)
        return previo_obj, "difiere"
    PREREG_JSON.write_text(texto, encoding="utf-8")
    return nuevo, "escrito"


def verifica_pre_registro() -> dict:
    """Corre las nueve aserciones de integridad del pre-registro.

    Las nueve son: (1) SHA256 de los dos parquets POS, (2) SHA256 del corpus train,
    (3) SHA256 de los seis canonicos mas el manifiesto, (4) SHA256 de
    ``dup_clusters`` y del manifiesto del plug-in, (5) constantes de gates, (6) alfa, semillas
    y pesos del plug-in, (7) las tres sondas de E9, (8) integridad del entorno de datos
    (T0.2), (9) inmutabilidad del fichero.

    Returns
    -------
    dict
        ``ok``, ``n_ok``, ``n_total`` y ``aserciones``.

    Raises
    ------
    AssertionError
        Si alguna asercion falla.
    """
    a: dict = {}
    if not PREREG_JSON.exists():
        raise AssertionError(
            "[STOP] no existe el pre-registro, asi que sus 9 aserciones de integridad no se "
            "pueden correr: falta " + str(PREREG_JSON.relative_to(ROOT)) + ". Generalo con "
            "'/home/carlos/venv/bin/python scripts/age_linguistic_13_preregistro_v37.py' "
            "(sin --verificar) y vuelve a pasar --verificar. La corrida de V37 no puede "
            "empezar sin el pre-registro congelado.")
    pre = json.loads(PREREG_JSON.read_text(encoding="utf-8"))
    hashes = pre["hashes"]

    ancla = ANCHAJES
    asercion("1_hash_parquets_pos",
             hashes["features/linguistic/anotacion_md_train.parquet"]
             == ancla["features/linguistic/anotacion_md_train.parquet"]
             and hashes["features/linguistic/anotacion_lg_train.parquet"]
             == ancla["features/linguistic/anotacion_lg_train.parquet"],
             "md = " + hashes["features/linguistic/anotacion_md_train.parquet"][:8]
             + " | lg = " + hashes["features/linguistic/anotacion_lg_train.parquet"][:8], a)
    asercion("2_hash_corpus_train",
             hashes["data/raw/age_train.csv"] == ancla["data/raw/age_train.csv"],
             "age_train.csv = " + hashes["data/raw/age_train.csv"][:8], a)
    manifest = json.loads(MANIFEST_CANONICO.read_text(encoding="utf-8"))
    canonicos_ok = all(hashes[f"artifacts/p33_scores/todos/{n}.npy"]
                       == manifest["sha256"][f"{n}.npy"]
                       for n in ("y_true", "fold_idx", "v30_prob", "cand_prob", "cand_pred",
                                 "ids"))
    asercion("3_hash_canonicos_seis_mas_manifest", canonicos_ok,
             "y_true=" + hashes["artifacts/p33_scores/todos/y_true.npy"][:8]
             + " fold_idx=" + hashes["artifacts/p33_scores/todos/fold_idx.npy"][:8]
             + " v30_prob=" + hashes["artifacts/p33_scores/todos/v30_prob.npy"][:8], a)
    asercion("4_hash_dup_clusters_y_manifest_plugin",
             hashes["features/dup_clusters.parquet"] == ancla["features/dup_clusters.parquet"]
             and hashes["reports/age_submission_plugin_manifest.json"]
             == ancla["reports/age_submission_plugin_manifest.json"],
             "dup_clusters=" + hashes["features/dup_clusters.parquet"][:8]
             + " plugin=" + hashes["reports/age_submission_plugin_manifest.json"][:8], a)
    c = pre["constantes"]
    asercion("5_constantes_de_gates",
             c["umbrales"] == CONSTANTES["umbrales"] and c["gates"] == CONSTANTES["gates"],
             f"V1..V6 y G1..G4 literales; V5 es sanity floor {c['umbrales']['V5']['a1_min']} "
             f"(E7) y V6 es control unilateral {c['umbrales']['V6']['a_perm_max']} (E1, E3)", a)
    asercion("6_alfa_semillas_y_pesos",
             c["suavizado"]["alpha"] == 1.0
             and c["semillas"]["bootstrap_cluster"] == 20261010
             and c["semillas"]["permute_control"] == 20261010
             and c["semillas"]["hash_agrupador"] == 0
             and c["n_bootstrap"] == 2000
             and c["pesos_plugin"] == [1.0, 1.1749, 1.2721, 1.163, 1.001, 0.8128],
             "alpha=1.0; semillas 20261010/20261010/0/42; 2000 iteraciones; pesos congelados", a)
    print("  --- corriendo la sonda E9 y comparando con los valores declarados ---", flush=True)
    sonda = corre_sondas_e9(silencioso=False)
    asercion("7_sondas_e9", bool(sonda["integridad"]["A1_dentro_de_tolerancia"]
                                 and sonda["integridad"]["A2_dentro_de_tolerancia"]
                                 and sonda["integridad"]["A_perm_cumple_criterio_unilateral"]
                                 and sonda["integridad"]["A_perm_dentro_de_banda_declarada"]),
             f"A1 medido {sonda['medidos']['A1']:.7f} vs {sonda['declarados']['A1']} "
             f"(|dif| {sonda['diferencias']['A1']:.2e} <= {sonda['tolerancia']:g}); "
             f"A2 medido {sonda['medidos']['A2']:.7f} vs {sonda['declarados']['A2']} "
             f"(|dif| {sonda['diferencias']['A2']:.2e} <= {sonda['tolerancia']:g}); "
             f"A_perm medido {sonda['medidos']['A_perm']:.7f} vs "
             f"{sonda['declarados']['A_perm']} (criterio unilateral E1 "
             f"A_perm <= {CONSTANTES['criterio_a_perm_max']}; banda declarada "
             f"{CONSTANTES['banda_a_perm_esperada']})", a)
    integridad = verificar_integridad(silencioso=True)
    asercion("8_integridad_entorno_datos_9_de_9",
             integridad["n_ok"] == integridad["n_total"] == 9,
             f"integridad del entorno de datos: {integridad['n_ok']}/{integridad['n_total']} "
             f"aserciones T0.2", a)
    reconstruido = construye_pre_registro()
    asercion("9_inmutabilidad_del_fichero",
             json.loads(PREREG_JSON.read_text(encoding="utf-8")) == json.loads(
                 json.dumps(reconstruido, ensure_ascii=True)),
             "el fichero escrito coincide con el que se construiria ahora: no se ha editado en "
             "sitio", a)
    return {"ok": True, "n_ok": len(a), "n_total": 9, "sonda": sonda, "aserciones": a}


# ==============================================================================
# CLI
# ==============================================================================
def main() -> int:
    """Punto de entrada de la fase 0 de V37.

    Returns
    -------
    int
        ``0`` en exito y ``2`` en cualquier ``[STOP]``.
    """
    analizador = argparse.ArgumentParser(
        prog="age_linguistic_13_preregistro_v37.py",
        description=(
            "Fase 0 (T0.1 + T0.2) de V37. SIN argumentos: comprueba los SHA256 contra los "
            "anclajes y escribe experiments/age_linguistic_13_preregistro_v37.json, que es "
            "INMUTABLE: si ya existe, lo verifica, no lo reescribe y avisa si difiere. "
            "Con --verificar: NO escribe nada; corre las 9 aserciones de integridad del "
            "pre-registro, que incluyen (a) los SHA256 de los 2 parquets POS, del corpus de "
            "train, de los 6 canonicos mas su manifiesto, de dup_clusters y del manifiesto del "
            "plug-in; (b) las constantes de gates, alfa, semillas y pesos; (c) la CORRIDA real "
            "de los tres brazos de la sonda declarada E9 (A0, A1, A2, A_perm) con comparacion "
            "contra los valores declarados a tolerancia 1e-6 para A1 y A2 y con el criterio "
            "unilateral de E1 mas la banda declarada para A_perm; (d) las 9 aserciones de "
            "integridad del entorno de datos de T0.2, incluida la reproduccion exacta de "
            "fold_idx con StratifiedGroupKFold; y (e) la inmutabilidad del propio fichero. "
            "Sale con 0 si pasan las nueve y con 2 nombrando la asercion que falla."),
        epilog=("Modos: (por defecto) congelar el pre-registro | --verificar auditar el "
                "pre-registro congelado sin escribir nada."))
    analizador.add_argument("--verificar", action="store_true",
                            help="corre las 9 aserciones de integridad del pre-registro "
                                 "(hashes, constantes, sonda E9 real, T0.2, inmutabilidad), "
                                 "imprime el detalle de cada una y no escribe nada")
    argumentos = analizador.parse_args()
    inicio = time.time()
    autotest_numerico()
    if argumentos.verificar:
        print("=== verificacion del pre-registro de V37 (hashes, constantes, sonda E9, T0.2) ===",
              flush=True)
        try:
            resultado = verifica_pre_registro()
        except Exception as error:  # noqa: BLE001 - cualquier fallo se nombra y se sale con 2
            print(f"  {type(error).__name__}: {error}", flush=True)
            print(f"[STOP] --verificar fallido ({time.time() - inicio:.1f} s)", flush=True)
            return 2
        for nombre, dato in resultado["aserciones"].items():
            print(f"  [{'OK' if dato['ok'] else 'FALLO'}] {nombre}: {dato['detalle']}", flush=True)
        print(f"OK {resultado['n_ok']}/{resultado['n_total']}", flush=True)
        print(f"[fin --verificar] {time.time() - inicio:.1f} s", flush=True)
        return 0
    print("=== pre-registro congelado de V37 ===", flush=True)
    try:
        anclajes = verifica_anclajes()
    except Exception as error:  # noqa: BLE001 - se imprime y se sale con 2
        print(f"  {error}", flush=True)
        return 2
    if not anclajes["ok"]:
        for relativo, dato in anclajes["detalle"].items():
            if not dato["coincide"]:
                print(f"  [STOP] hash distinto del anclaje en {relativo}", flush=True)
        print("[STOP] exit 2: los artefactos_POS cambiaron y la spec queda obsoleta.", flush=True)
        return 2
    print(f"  anclajes de hash: OK ({len(anclajes['detalle'])} ficheros)", flush=True)
    try:
        pre, estado = escribe_pre_registro()
    except Exception as error:  # noqa: BLE001 - se imprime y se sale con 2
        print(f"  [STOP] {error}", flush=True)
        return 2
    print(f"  {PREREG_JSON.relative_to(ROOT)}: {estado}", flush=True)
    print(f"  sha256 del pre-registro: {sha256_de_archivo(PREREG_JSON)}", flush=True)
    print(f"  etiquetas UPOS declaradas: {len(pre['etiquetas_pos'])} | "
          f"alpha = {pre['constantes']['suavizado']['alpha']} | "
          f"semillas = {pre['constantes']['semillas']}", flush=True)
    sonda = pre["constantes"]["sonda_previa"]
    print(f"  sonda previa declarada (E9): A1={sonda['A1']} A2={sonda['A2']} "
          f"A_perm={sonda['A_perm']} (tolerancia {sonda['tolerancia']})", flush=True)
    print(f"[fin] {time.time() - inicio:.1f} s", flush=True)
    return 0 if estado != "difiere" else 2


if __name__ == "__main__":
    raise SystemExit(main())
