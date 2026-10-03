"""v33_01_ftrl_softmax_online.py
Fecha: 2026-09-30
Descripcion: implementacion literal del pre-registro congelado V33-E1: challenger FTRL-Proximal
softmax multiclase (K=6 en un solo juego de pesos) entrenado en UNA sola pasada online sobre la
TF-IDF ajustada por fold (brazo decisional A1) frente al control gemelo SGDClassifier de una
pasada (A2), con puerta de convergencia G1-G6 fail-fast dentro del fold 0 y antes de integrar,
sonda de atribucion |sigma*w|/|g| obligatoria, sondas (a)-(f) antes de entrenar, checkpoints
resumibles por fold, OOF float64 con escritura atomica, sondas de determinismo y de refit,
integracion P6/P7 congelada, gates, caps y arbol de decision.

Estado del arte aplicado: FTRL-Proximal (McMahan et al. 2013) con softmax multiclase y un solo
juego de pesos K=6, sesgo materializado como coordenada con x0=1.0 (algebraicamente X@W + b),
lambda1 0.0, lambda2 1.0, beta 1.0, sin power_t, float64, UNA sola pasada sobre el fold-train
barajado una vez con np.random.default_rng(20261010+fold) y orden congelado reutilizado por el
control; alpha DERIVADO alpha = 3/(2*sqrt(T)) con T = n_train_fold calculado en runtime (cero
valores evaluados, no es un grid); gemelo sklearn.linear_model.SGDClassifier(loss=log_loss,
learning_rate=optimal, alpha=1e-6, max_iter=1, tol=None, shuffle=False para consumir el
orden congelado tal cual) sobre la MISMA matriz; TF-IDF de
palabra (1,2) con min_df 3, max_df 0.5 y sublinear_tf ajustada SOLO en el fold-train; bootstrap
pareado de 2000 remuestreos (semilla 20261010) del delta de A1 y, como diagnostico, de la
diferencia A1-A2.

Fuentes: experiments/v33_00_preregistro.json (protocolo, constantes, hashes, sondas y umbrales
congelados), reports/v33_preregistro.md, reports/v33_convergence_gate.md (puerta G1-G6
NORMATIVA con umbrales DERIVADOS, no calibrados) y plan/V32_REAPERTURA_PLAN.md.
Codigos de salida: 0 fin correcto (incluye el cierre documentado NO_ATRIBUIBLE_BY_GATE); 2
verificacion (constantes, versiones, hashes, corpus, anclas, auditoria AST o invariantes); 4
sonda (a)-(f) fallida antes de entrenar; 6 calibracion de fold 0 excedida; 7 timeout global;
8 RSS por encima del tope declarado; 1 error inesperado con traceback.
"""

from __future__ import annotations

# =====================================================================================
# ---- 0. ENTORNO E IMPORTS (los 32 hilos van ANTES de importar numpy/scipy/sklearn) ----
# =====================================================================================
import os

#: Los 32 hilos congelados se fijan antes de importar numpy (determinismo). El pre-registro
#: exige OMP/MKL/OPENBLAS_NUM_THREADS = 32; el lazo FTRL es mono-hilo de todos modos.
os.environ["OMP_NUM_THREADS"] = "32"
os.environ["MKL_NUM_THREADS"] = "32"
os.environ["OPENBLAS_NUM_THREADS"] = "32"

import argparse
import ast
import csv
import hashlib
import json
import math
import resource
import subprocess
import sys
import time
import traceback
from collections.abc import Callable, Iterable
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import f1_score, log_loss

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src import config as cfg
from src import data as datos
from src.verificacion_numerica import autotest_numerico

# =====================================================================================
# ---- 1. SETUP: CONSTANTES CONGELADAS Y VERIFICACION DEL PRE-REGISTRO ----
# =====================================================================================
FECHA = "2026-09-30"
N_CLASES = 6
N_FILAS = 169336
N_FOLDS = 5
HILOS = 32
BASELINE = 0.6159732
NESTED = 0.6154643
PESOS = [1.0, 1.1749, 1.2721, 1.163, 1.001, 0.8128]
ABORTO_BASELINE = 5e-7
GATE_DELTA = 0.0015
GATE_FOLDS = 4
VARIABILIDAD = 0.0008876
N_BOOTSTRAP = 2000
UMBRAL_ALTA_CONFIANZA = 0.60
SEMILLA = 20261010
SEMILLA_SEGUNDA_PARTICION = 987654
PROYECCION_FACTOR = 9.0
PROYECCION_MAX_H = 20.0
TIMEOUT_H = 24.0
RAM_TOPE_GB = 20.0
L2_CLIP = 1e-12
C_LOGREG = 1.0
MAX_ITER_LOGREG = 2000
TOL_DETERMINISMO = 1e-9
TOL_ANCLA_TRIVIAL = 1e-15
FASE_ESTADO = "V33-E1 ejecucion (checkpoint resumible)"

#: Constantes de la regla DERIVADA de alpha: alpha = NUMERADOR_ALPHA / (DENOMINADOR_ALPHA *
#: sqrt(T)). El pre-registro congela la formula "alpha = 3/(2*sqrt(T))" y T = n_train_fold; los
#: dos coeficientes se escriben como enteros para que la formula sea auditable.
NUMERADOR_ALPHA = 3
DENOMINADOR_ALPHA = 2

#: Puntos de la pasada en los que se mide el log_loss de train (G2), en fraccion de la pasada.
PUNTOS_PASADA = (0.25, 0.5, 1.0)

#: Particiones reservadas y familias de librerias vetadas por el plan. Ningun literal de ruta
#: de este script puede contener las cuatro primeras en minusculas; las diez ultimas se
#: comprueban ADEMAS sobre los modulos importados, no solo sobre los literales de ruta.
PARTICIONES_RESERVADAS = ("test", "holdout", "submission", "leaderboard")
FAMILIAS_VETADAS = ("vowpal", "vw", "torch", "transformers", "gensim", "fasttext",
                    "sentence-transformers", "word2vec", "glove", "spacy-vectors")
#: Extensiones que hacen que un literal de cadena se trate como ruta en la auditoria AST.
EXTENSIONES_RUTA = (".csv", ".npy", ".json", ".npz", ".txt")

BRAZOS = ("A1_ftrl_softmax", "A2_sgd_1p")
NOMBRES_BRAZO = {"A1_ftrl_softmax": "A1 (FTRL softmax multiclase, decisional)",
                 "A2_sgd_1p": "A2 (gemelo SGD de una pasada, control)"}
OOF_ARCHIVO = {"A1_ftrl_softmax": "a1_ftrl_softmax", "A2_sgd_1p": "a2_sgd_1p"}
#: Campos numericos que el checkpoint exige por fold de cada brazo.
CAMPOS_FOLD = ("n_features", "n_vocabulario", "vocab_sha256", "idf_sha256", "n_filas_fit", "T",
               "alpha_efectivo", "tiempo_seg", "ll_train_prefijo", "ll_train_online",
               "macro_f1_standalone_fold_val", "frecuencias_fold_val", "max_abs_suma_menos_uno",
               "n_train_fold", "n_val_fold", "atribucion")

# -------------------------------------------------------------------------------------
# Espejos LITERALES del pre-registro congelado. El codigo lee sus parametros de aqui, de modo
# que no puede haber deriva entre el protocolo congelado y la ejecucion: si el JSON cambia, el
# assert de no-deriva falla y el script sale con codigo 2 sin entrenar.
# -------------------------------------------------------------------------------------
A1_FTRL = {
    "algoritmo": "FTRL-Proximal softmax multiclase (McMahan 2013), implementacion propia exacta",
    "clases": "K=6 en un solo juego de pesos (softmax); p suma 1 y es comparable con la mezcla",
    "alpha": {
        "formula": "alpha = 3/(2*sqrt(T))",
        "T": "n_train_fold del fold",
        "origen": "derivado del desplazamiento acumulado del intercepto: 2*alpha*sqrt(T) debe ser "
                  "del orden de 1.8x el log-odds de la tasa base |log(1/5)|",
        "valores_evaluados": 0,
        "rango_esperado": "0.00404 a 0.00409 (varia con T por construccion, no por busqueda)",
        "nota": "con alpha=0.05 (D-083) el desplazamiento seria 36.8, 36x mayor"
    },
    "lambda1": 0.0,
    "lambda2": 1.0,
    "beta": 1.0,
    "bias_x0": 1.0,
    "power_t": None,
    "pasadas": 1,
    "orden": "fold-train barajado UNA vez con np.random.default_rng(20261010+fold); orden "
             "congelado y reutilizado por el control",
    "dtype": "float64",
    "probabilidades": "softmax de los 6 logits, sin temperatura (L04/C03 cerradas)",
    "alternativa_no_usada": "vowpalwabbit 9.11.8 instalado y NO usado"
}
A2_SGD = {
    "estimator": "sklearn.linear_model.SGDClassifier",
    "loss": "log_loss",
    "learning_rate": "optimal",
    "alpha": 1e-06,
    "max_iter": 1,
    "tol": None,
    "fit_intercept": True,
    "average": False,
    "shuffle": False,
    "random_state": "20261010+fold",
    "n_jobs": 32,
    "pasadas": 1,
    "por_que": "misma representacion, mismo presupuesto (1 pasada), mismo barajado, misma semilla "
               "y misma evaluacion: lo unico libre es el eje proximal/adaptativo",
    "asimetria_declarada": "alpha 1e-6 frente a 0.004 (cuatro ordenes de magnitud) y la"
                           "regularidad del intercepto: por eso la sonda |sigma*w|/|g| es OBLIGATORIA"
}
PUERTA_PREREG = {
    "momento": "dentro del fold 0, ANTES de los folds 1-4 y ANTES de integrar",
    "aplica_a": ["A1_ftrl_softmax", "A2_sgd_1p"],
    "fold_0": "consumido-diagnostico: G3 y G4 leen el fold-val del fold 0, que es parte de la "
              "validacion canonica; su fallo NO toca delta",
    "consecuencia_si_falla": "veredicto NO_ATRIBUIBLE_BY_GATE: sin fallback, sin segundo alpha, sin "
                             "integrar, sin bootstrap; se documenta y termina",
    "prohibido": "reintentar con otro alpha tras el fallo (seria grid)",
    "umbrales_derivados": {
        "G1_ll_train_banda": {
            "rango": [0.05, 1.6125835223052495],
            "anclaje": "suelo 0.05 = detector de humo (con una sola pasada y una actualizacion "
                       "por ejemplo no hay memorizacion posible); techo 0.9*log(6) = "
                       "1.6125835223052495: el ll_train debe mejorar el uniforme en al menos un "
                       "10 % relativo. Enmenda 1 pre-ejecucion: la banda congelada [0.25, 0.60] "
                       "era inalcanzable para cualquier modelo sano (en D-083 el gemelo sano dio "
                       "log_loss multiclase 1.3568 > 0.60) y cerraba el experimento por "
                       "construccion"
        },
        "G2_monotonia": {
            "puntos": [0.25, 0.5, 1.0],
            "tolerancia": 1e-06,
            "anclaje": "caza directamente la firma 11.72 -> 23.03 de D-083"
        },
        "G3_macro_f1_fold_val": {
            "minimo": 0.3,
            "anclaje": "trivial 0.0485421 y gemelo SGD de D-083 0.4934789"
        },
        "G4_frecuencia_por_clase": {
            "rango": [0.1, 0.3],
            "anclaje": "proporciones reales de clase del train"
        },
        "G5_determinismo": {
            "criterio": "max|delta p| = 0.0 exacto entre dos corridas identicas"
        },
        "G6_suma_unidad": {
            "tolerancia": 1e-09,
            "criterio": "max|sum_c p_c - 1|"
        }
    }
}
SONDA_ATRIBUCION_PREREG = {
    "descripcion": "razon media |sigma*w| / |g| por coordenada en el fold 0",
    "obligatoria": True,
    "consecuencia_si_no_se_publica": "el veredicto queda capado a MARGINAL",
    "motivo": "el control cambia dos variables (alpha y regularizacion del intercepto): la razon "
              "sigma/g es la que desambigua la causa (correccion inflada frente a multi-pasada)"
}
SONDAS_PREREG = {
    "a_vectorizado_vs_lazo": {"tolerancia": 1e-12},
    "b_raiz_analitica_lambda0": {"residuo_max": 1e-09},
    "c_totalf1_macro": {"tolerancia": 1e-06},
    "d_sin_nan_logloss_finito": True,
    "e_logloss_multiclase_propio": {
        "tolerancia": 1e-09,
        "nota": "NUEVA en V33: D-083 solo probo el OvR, que no normaliza"
    },
    "f_filas_suman_uno": {"tolerancia": 1e-09},
    "politica": "si alguna falla -> exit 4 ANTES de entrenar; cualquier ajuste exige enmienda "
                "pre-ejecucion (precedente f0bf731)"
}
REFIT_PREREG = {
    "descripcion": "reajustar el vectorizador en los 5 folds y comprobar que los sha256 de "
                   "vocabulario e idf coinciden con los de la corrida",
    "exigido_por": "kaggle-specialist",
    "tolerancia": "identicos bit a bit entre las dos corridas"
}
DETERMINISMO_PREREG = {
    "semilla": 20261010,
    "orden": "default_rng(20261010+fold) una vez por fold",
    "hilos": "OMP/MKL/OPENBLAS_NUM_THREADS = 32 antes de numpy; FTRL mono-hilo",
    "esperado": "exacto bit a bit",
    "sonda": "reentrenar el fold 0 de A1 y de A2 y comparar bit a bit",
    "tolerancias": "0 exacto; <=1e-9 conforme; >1e-9 con rango publicado y delta_gate > rango; "
                   "rango >= 0.0008876 o solape con el gate => no concluyente"
}
INTEGRACION_PREREG = {
    "mecanismo": "combinador logistico anidado sobre [L2(v30_prob) || L2(p_brazo)] (12 features, "
                 "L2 por fila con clip 1e-12); LogReg(C=1.0, max_iter=2000, lbfgs) por fold (fit "
                 "fold!=k, predice fold==k)",
    "decision": "argmax(p_comb * w_plugin CONGELADO)",
    "diagnosticos": ["argmax_sin_regla", "delta vs nested 0.6154643", "log_loss por brazo",
                     "macro-F1 diagonal 6x6", "rescate/dano/neto", "alta confianza"]
}
GATES_PREREG = {
    "baseline_abort": 5e-07,
    "delta_min": 0.0015,
    "folds_min": 4,
    "control": "delta_int(A1) > delta_int(A2)",
    "bootstrap": {
        "n": 2000,
        "semilla": 20261010,
        "ic95_sin_cero": True,
        "diagnostico_adicional": "IC95 pareado de la diferencia (A1-A2)"
    },
    "sin_leakage": True,
    "determinismo_concluyente": True,
    "sonda_atribucion_publicada": True,
    "caps": [
        "sin la sonda |sigma*w|/|g| publicada -> veredicto capado a MARGINAL",
        "determinismo no concluyente -> el veredicto no puede ser PROMISING"
    ]
}
ARBOL_PREREG = {
    "NO_ATRIBUIBLE_BY_GATE": "la puerta G1-G6 falla en A1 o A2: STOP documentado, sin fallback, "
                             "sin integrar (nodo nuevo, no un gate de delta)",
    "PROMISING": "gates completos -> segunda particion SOLO con autorizacion del usuario (semilla "
                 "987654); el script no la ejecuta",
    "MARGINAL": "delta > 0 sin gates completos, o con cap activo (sonda o determinismo)",
    "CLOSED": "delta <= 0 -> se cierra V33; el campeon permanece",
    "submission": "jamas automatica; requeriria gate completo + segunda particion + autorizacion "
                  "explicita"
}
NOVEDAD_PREREG = {
    "que_es": "FTRL-Proximal softmax multiclase propio, semantica online de una pasada, un solo "
              "juego de pesos, sobre la TF-IDF por fold",
    "no_es": "no es un embedding, no es un vector aprendido, no es una representacion nueva y no "
             "usa datos externos",
    "diferencia_con_d083": "D-083 fue OvR con 10 pasadas proximales y alpha 0.05; aqui softmax, "
                           "1 pasada y alpha derivado",
    "registro": "artefactos NUEVOS escritos por el runner: reports/v33_novelty_registry.csv (1 "
                "fila) y una fila anexada en reports/age_complete_experiment_history.csv (mismo "
                "esquema); acta D-084 por ANEXADO en DECISIONS.md; no se edita "
                "scripts/age_cobertura_metodologica.py ni ningun script ejecutado"
}
VIA_B_PREREG = {
    "B1_grafo_transductivo": "ILEGAL (usa features de test sin etiqueta para modificar "
                             "predicciones de train; su forma legal es un kNN re-ponderado = "
                             "mecanismo M09 ya cerrado)",
    "B2_aumentacion_masking": "descartada por analogia con C04 (bagging, -0.000312)",
    "B3_cabeza_multitarea": "refutada por algebra: con tronco lineal es una reparametrizacion de "
                            "la misma matriz de pesos"
}
EJECUCION_PREREG = {
    "coste_estimado": "~13-16 min CPU (1 pasada por fold y por brazo; TF-IDF ~90 s por fold "
                      "compartido; determinismo: 2 reentrenamientos del fold 0)",
    "ram_tope_gb": 20.0,
    "ram_medicion": "resource.getrusage(ru_maxrss)",
    "calibracion": "tras el fold 0: proyeccion = 9 x t_fold0; si supera 20.0 h -> parar y reportar",
    "timeout_global": "24.0 h",
    "checkpoints": "experiments/v33_01_estado.json (resumible por fold)",
    "escritura_atomica": "tmp + os.replace en artifacts/p33_scores/v33_e1/"
}
#: Campos numericos de compliance. El campo de prosa ``ast_audit`` NO se refleja como literal
#: porque describe la propia auditoria y contiene por construccion las palabras vigiladas (con
#: barras inclinadas): su presencia marcaria un falso positivo contra este mismo script.
COMPLIANCE_CIFRAS = {"embeddings": 0, "transformers": 0, "llm": 0, "datos_externos": 0,
                     "test": 0, "holdout": 0, "leaderboard": 0, "submission": 0}
COMPLIANCE_AUDITORIAS = ["6 .npy canonicos + manifest",
                        "sha256 del archivo y del array de textos (nunca el data_hash del manifiesto)",
                        "sha256 del vectorizador", "vocabulario e idf por fold",
                        "sonda de refit en los 5 folds", "sondas (a)-(f)",
                        "sonda de atribucion |sigma*w|/|g|", "sondas de determinismo A1 y A2",
                        "RSS pico con ru_maxrss", "escritura atomica"]

#: Configuracion del vectorizador: IDENTICA a la congelada (hash 9bae1576 del JSON). En el fit
#: se convierte ngram_range a tupla porque sklearn 1.8.0 lo exige; es el mismo valor, y el
#: sha256 congelado del dict se calcula SIEMPRE sobre la CONFIG con la lista.
VECTORIZADOR = {"analyzer": "word", "ngram_range": [1, 2], "min_df": 3, "max_df": 0.5,
                "sublinear_tf": True, "lowercase": True, "strip_accents": None,
                "token_pattern": r"(?u)\b\w+\b"}
VECTORIZADOR_FIT = {**VECTORIZADOR, "ngram_range": (1, 2)}

#: Anclas numericas congeladas (se recalculan y se contrastan en tiempo de ejecucion).
ANCLA_TRIVIAL = 0.04854212180687087
ANCLA_GEMELO_SGD_D083 = 0.4934789
ANCLA_LOG_UNIFORME = 1.791759469228055
ANCLA_N_TRAIN_FOLD = 135468
ANCLA_ALPHA_ESPERADO = 0.0040754249463367

#: Umbrales de la puerta G1-G6, extraidos del pre-registro por el codigo (espejo numerico).
G1_BANDA = (0.05, 1.6125835223052495)
G2_TOLERANCIA = 1e-06
G3_MINIMO = 0.30
G4_RANGO = (0.10, 0.30)
G5_TOLERANCIA = 0.0
G6_TOLERANCIA = 1e-09
#: Descripcion de la magnitud de la puerta, identica para A1 y A2 (ver ``prefijo_a1``).
MAGNITUD_PUERTA = ("log_loss multiclase de train en el 25/50/100 % de la pasada, medido como "
                   "perdida de train del modelo ajustado a ese prefijo del orden congelado "
                   "(misma magnitud en A1 y A2); en A1 se publica ademas la media online del "
                   "stream, que es la lectura directa de la firma de D-083")

#: Parametros derivados de los espejos congelados (no se escriben a mano dos veces).
LAMBDA1 = float(A1_FTRL["lambda1"])
LAMBDA2 = float(A1_FTRL["lambda2"])
BETA = float(A1_FTRL["beta"])
X0 = float(A1_FTRL["bias_x0"])
SGD_ALPHA = float(A2_SGD["alpha"])
SGD_MAX_ITER = int(A2_SGD["max_iter"])
TOL_SONDA_A = float(SONDAS_PREREG["a_vectorizado_vs_lazo"]["tolerancia"])
TOL_SONDA_B = float(SONDAS_PREREG["b_raiz_analitica_lambda0"]["residuo_max"])
TOL_SONDA_C = float(SONDAS_PREREG["c_totalf1_macro"]["tolerancia"])
TOL_SONDA_E = float(SONDAS_PREREG["e_logloss_multiclase_propio"]["tolerancia"])
TOL_SONDA_F = float(SONDAS_PREREG["f_filas_suman_uno"]["tolerancia"])
DIM_SONDA_A = (200, 50)
DIM_SONDA_D = (400, 20)

CANONICO = ROOT / "artifacts" / "p33_scores" / "todos"
OOF_DIR = ROOT / "artifacts" / "p33_scores" / "v33_e1"
PREREG_JSON = cfg.EXPERIMENTS / "v33_00_preregistro.json"
ANCLA_D083 = cfg.EXPERIMENTS / "v32_05_ftrl_challenger.json"
ESTADO_JSON = cfg.EXPERIMENTS / "v33_01_estado.json"
RESULTADOS_JSON = cfg.EXPERIMENTS / "v33_01_ftrl_softmax_online.json"
INFORME = cfg.REPORTS / "v33_e1_ftrl_softmax_online.md"
FIGURA = cfg.FIGURES / "v33_e1_curvas.png"
NOVELTY_CSV = cfg.REPORTS / "v33_novelty_registry.csv"
HISTORIA_CSV = cfg.REPORTS / "age_complete_experiment_history.csv"
RUNLOG = ROOT / "RUNLOG.md"
EXPERIMENTS_CSV = ROOT / "experiments.csv"
ARCHIVO_PRE_REGISTRO = "reports/v33_preregistro.md"
ARCHIVO_PUERTA = "reports/v33_convergence_gate.md"
RUTAS_HASHES = {
    "canonico_y_true": CANONICO / "y_true.npy",
    "canonico_fold_idx": CANONICO / "fold_idx.npy",
    "canonico_v30_prob": CANONICO / "v30_prob.npy",
    "canonico_ids": CANONICO / "ids.npy",
    "canonico_cand_pred": CANONICO / "cand_pred.npy",
    "canonico_cand_prob": CANONICO / "cand_prob.npy",
    "manifest_canonico": CANONICO / "manifest.json",
}
CLAVES_HASH_CORPUS = ("corpus_archivo", "corpus_textos", "vectorizador", "ancla_d083_json")
#: Cabecera congelada del historial de experiments; el runner comprueba que coincide y anexa al
#: final sin reordenar ni reescribir ninguna fila previa.
COLUMNAS_HISTORIA = ["phase", "experiment_id", "date", "hypothesis", "model", "representation",
                     "corrected_cv", "delta_vs_V20", "folds", "bootstrap", "rescue", "damage",
                     "net", "decision", "resultado"]
#: Cabecera del registro de novedad (archivo NUEVO escrito por este runner).
COLUMNAS_NOVEDAD = ["experiment_id", "fecha", "que_es", "no_es_embedding",
                    "no_es_representacion_nueva", "no_usa_datos_externos", "diferencia_con_d083",
                    "verificacion", "artefactos"]


class Detencion(RuntimeError):
    """Detencion pre-registrada: el mensaje se imprime y el proceso sale con ``codigo``."""

    def __init__(self, mensaje: str, codigo: int) -> None:
        super().__init__(mensaje)
        self.codigo = codigo


# =====================================================================================
# ---- 1b. UTILIDADES DE HASH, JSON, METRICAS Y NORMALIZACION ----
# =====================================================================================
def sha256_bytes(datos: bytes) -> str:
    """SHA256 hexadecimal de un bloque de bytes.

    Parameters
    ----------
    datos:
        Bytes a resumir.

    Returns
    -------
    str
        Digest hexadecimal de 64 caracteres.
    """
    return hashlib.sha256(datos).hexdigest()


def sha256_archivo(ruta: Path) -> str:
    """SHA256 hexadecimal de un archivo.

    Parameters
    ----------
    ruta:
        Archivo a resumir.

    Returns
    -------
    str
        Digest hexadecimal de 64 caracteres.
    """
    return sha256_bytes(ruta.read_bytes())


def sha256_textos(textos: Iterable[str]) -> str:
    """SHA256 determinista de una secuencia de textos: longitud (8 bytes LE) + bytes UTF-8.

    Es la receta congelada del pre-registro para el array de textos del corpus; aqui tambien
    se usa para el vocabulario ordenado de cada fold (misma receta, mismo orden).

    Parameters
    ----------
    textos:
        Textos en el orden exacto que se quiere congelar.

    Returns
    -------
    str
        Digest hexadecimal de 64 caracteres.
    """
    resumen = hashlib.sha256()
    for texto in textos:
        bruto = texto.encode("utf-8")
        resumen.update(len(bruto).to_bytes(8, "little"))
        resumen.update(bruto)
    return resumen.hexdigest()


def sha256_f64(valores: np.ndarray) -> str:
    """SHA256 de un arreglo float64 en orden C (usado para el idf de cada fold).

    Parameters
    ----------
    valores:
        Arreglo float64.

    Returns
    -------
    str
        Digest hexadecimal de 64 caracteres.
    """
    return sha256_bytes(np.ascontiguousarray(valores, dtype=np.float64).tobytes())


def ruta_relativa(ruta: Path) -> str:
    """Ruta de artefacto relativa al proyecto, para informes, CSV y estado.

    Parameters
    ----------
    ruta:
        Ruta absoluta o relativa.

    Returns
    -------
    str
        Ruta relativa al proyecto si es posible; si no, la ruta tal cual.
    """
    try:
        return str(ruta.relative_to(ROOT))
    except ValueError:
        return str(ruta)


def ruta_oof(brazo: str) -> Path:
    """Ruta congelada del OOF float64 del brazo (probabilidades que suman 1 por fila).

    Parameters
    ----------
    brazo:
        Nombre del brazo.

    Returns
    -------
    Path
        Ruta del ``.npy`` del brazo.
    """
    return OOF_DIR / f"{OOF_ARCHIVO[brazo]}.npy"


def como_nativo(valor: object) -> object:
    """Convierte recursivamente a escalares nativos de JSON (float/int/bool/str/None).

    Los float no finitos se guardan como ``None`` para que el JSON siga siendo valido y las
    claves de fold se fuerzan a texto.

    Parameters
    ----------
    valor:
        Estructura arbitraria (dict, lista, tuple, ndarray o escalar).

    Returns
    -------
    object
        Estructura con escalares nativos y claves de fold como texto.
    """
    if isinstance(valor, dict):
        return {str(clave): como_nativo(item) for clave, item in valor.items()}
    if isinstance(valor, (list, tuple)):
        return [como_nativo(item) for item in valor]
    if isinstance(valor, np.ndarray):
        return como_nativo(valor.tolist())
    if isinstance(valor, (bool, np.bool_)):
        return bool(valor)
    if isinstance(valor, (float, np.floating)):
        numero = float(valor)
        return numero if math.isfinite(numero) else None
    if isinstance(valor, (int, np.integer)):
        return int(valor)
    return valor


def escribir_json(ruta: Path, datos: object) -> None:
    """Escribe JSON indentado con escalares nativos y codificacion ASCII, de forma atomica.

    Parameters
    ----------
    ruta:
        Archivo destino (se crean sus directorios).
    datos:
        Estructura a serializar.
    """
    escribir_json_atomico(ruta, json.dumps(como_nativo(datos), indent=2, ensure_ascii=True) + "\n")


def escribir_json_atomico(ruta: Path, texto: str) -> None:
    """Escribe un texto con escritura atomica (temporal + ``os.replace``).

    Parameters
    ----------
    ruta:
        Archivo destino (se crean sus directorios).
    texto:
        Contenido completo a escribir.
    """
    ruta.parent.mkdir(parents=True, exist_ok=True)
    temporal = ruta.with_name(ruta.name + ".tmp")
    temporal.write_text(texto, encoding="utf-8")
    os.replace(temporal, ruta)


def cargar_preregistro() -> dict:
    """Lee el JSON congelado del pre-registro V33-E1.

    Returns
    -------
    dict
        Contenido completo de ``experiments/v33_00_preregistro.json``.

    Raises
    ------
    FileNotFoundError
        Si el pre-registro no existe.
    """
    if not PREREG_JSON.exists():
        raise FileNotFoundError(f"no existe el pre-registro {PREREG_JSON}")
    return json.loads(PREREG_JSON.read_text(encoding="utf-8"))


def alpha_derivado(n_ejemplos: int) -> float:
    """Alpha DERIVADO de A1: ``3/(2*sqrt(T))`` con T = numero de ejemplos del ajuste.

    No es un grid ni un valor buscado: la regla esta congelada y T se calcula en runtime, de
    modo que alpha varia con el tamano del fold-train por construccion y no por busqueda.

    Parameters
    ----------
    n_ejemplos:
        T, el numero de ejemplos que ve el ajuste (fold-train completo o su prefijo).

    Returns
    -------
    float
        Alpha efectivo del ajuste.
    """
    if n_ejemplos <= 0:
        raise Detencion("[STOP] alpha_derivado con T no positivo", 2)
    return NUMERADOR_ALPHA / (DENOMINADOR_ALPHA * math.sqrt(float(n_ejemplos)))


def nombres_entregables() -> list[str]:
    """Nombres de artefactos que el pre-registro congela para este runner.

    Returns
    -------
    list[str]
        Rutas relativas de los entregables declarados en el pre-registro.
    """
    return [ruta_relativa(RESULTADOS_JSON), ruta_relativa(ESTADO_JSON), ruta_relativa(INFORME),
            ruta_relativa(FIGURA), ruta_relativa(NOVELTY_CSV), ruta_relativa(HISTORIA_CSV),
            f"scripts/{Path(__file__).name}"] + [OOF_ARCHIVO[brazo] for brazo in BRAZOS]


def verificar_constantes(prereg: dict) -> tuple[bool, dict]:
    """Compara cada constante local contra el JSON congelado, una por una.

    Los bloques del pre-registro (a1_ftrl_softmax, a2_sgd_control, puerta_convergencia,
    sonda_atribucion, sondas, sobre_refit, determinismo, integracion, gates, arbol, novedad,
    via_b_descartada y ejecucion) se comparan como espejo literal: si el protocolo cambia, el
    runner se detiene con codigo 2 antes de entrenar.

    Parameters
    ----------
    prereg:
        Contenido del pre-registro.

    Returns
    -------
    tuple[bool, dict]
        Exito global y diccionario nombre -> booleano con el detalle de cada chequeo.
    """
    entregable = prereg["entregables"]["runner"]
    anclas = prereg["anclas"]
    puerta = PUERTA_PREREG["umbrales_derivados"]
    checks = {
        "fase": prereg["fase"] == "V33-E1 pre-registro congelado",
        "fecha": prereg["fecha"] == FECHA,
        "baseline_primario": float(prereg["baseline_primario"]) == BASELINE,
        "referencia_nested": float(prereg["referencia_nested"]) == NESTED,
        "pesos_plugin": [float(valor) for valor in prereg["pesos_plugin"]] == PESOS,
        "versiones_claves": set(prereg["versiones"]) == {"sklearn", "numpy", "pandas"},
        "hashes_claves": set(prereg["hashes"]) == set(RUTAS_HASHES) | set(CLAVES_HASH_CORPUS),
        "anclas_claves": set(anclas) == {"macro_f1_trivial_clase_mayoritaria", "clase_mayoritaria",
                                        "gemelo_sgd_d083", "log_uniforme_k6",
                                        "n_train_fold_esperado", "alpha_esperado_para_n_train"},
        "anclas_cifras": (float(anclas["macro_f1_trivial_clase_mayoritaria"]) == ANCLA_TRIVIAL
                          and float(anclas["gemelo_sgd_d083"]) == ANCLA_GEMELO_SGD_D083
                          and float(anclas["log_uniforme_k6"]) == ANCLA_LOG_UNIFORME
                          and int(anclas["n_train_fold_esperado"]) == ANCLA_N_TRAIN_FOLD
                          and float(anclas["alpha_esperado_para_n_train"])
                          == ANCLA_ALPHA_ESPERADO),
        "anclas_log_uniforme": abs(math.log(N_CLASES) - ANCLA_LOG_UNIFORME) <= 1e-15,
        "anclas_alpha_derivado": (alpha_derivado(ANCLA_N_TRAIN_FOLD) == ANCLA_ALPHA_ESPERADO
                                  and NUMERADOR_ALPHA == 3 and DENOMINADOR_ALPHA == 2),
        "a1_ftrl": prereg["a1_ftrl_softmax"] == A1_FTRL,
        "a1_parametros": (LAMBDA1 == 0.0 and LAMBDA2 == 1.0 and BETA == 1.0 and X0 == 1.0
                          and A1_FTRL["power_t"] is None and A1_FTRL["pasadas"] == 1
                          and A1_FTRL["dtype"] == "float64"
                          and A1_FTRL["alpha"]["valores_evaluados"] == 0),
        "a1_orden": A1_FTRL["orden"]
                    == f"fold-train barajado UNA vez con np.random.default_rng({SEMILLA}+fold); "
                       "orden congelado y reutilizado por el control",
        "a1_alternativa_no_usada": A1_FTRL["alternativa_no_usada"].startswith("vowpalwabbit")
                                   and "NO usado" in A1_FTRL["alternativa_no_usada"],
        "a1_probabilidades": "sin temperatura" in A1_FTRL["probabilidades"],
        "a2_sgd": prereg["a2_sgd_control"] == A2_SGD,
        "a2_parametros": (SGD_ALPHA == 1e-06 and SGD_MAX_ITER == 1 and A2_SGD["tol"] is None
                          and A2_SGD["fit_intercept"] is True and A2_SGD["average"] is False
                          and A2_SGD["learning_rate"] == "optimal" and A2_SGD["loss"] == "log_loss"
                          and A2_SGD["shuffle"] is False
                          and A2_SGD["estimator"] == "sklearn.linear_model.SGDClassifier"),
        "a2_random_state": A2_SGD["random_state"] == f"{SEMILLA}+fold",
        "a2_n_jobs": A2_SGD["n_jobs"] == HILOS,
        "enmiendas_prereg": (len(prereg.get("enmiendas", [])) == 2
                             and all("pre-ejecucion" in str(e.get("estado_datos", ""))
                                     for e in prereg["enmiendas"])),
        "puerta": prereg["puerta_convergencia"] == PUERTA_PREREG,
        "puerta_aplica_a": PUERTA_PREREG["aplica_a"] == list(BRAZOS),
        "puerta_umbrales": (G1_BANDA == (0.05, 1.6125835223052495)
                            and list(puerta["G1_ll_train_banda"]["rango"])
                            == [0.05, 1.6125835223052495] and G2_TOLERANCIA == 1e-06
                            and list(puerta["G2_monotonia"]["puntos"]) == list(PUNTOS_PASADA)
                            and float(puerta["G2_monotonia"]["tolerancia"]) == G2_TOLERANCIA
                            and G3_MINIMO == 0.3
                            and float(puerta["G3_macro_f1_fold_val"]["minimo"]) == G3_MINIMO
                            and G4_RANGO == (0.1, 0.3)
                            and list(puerta["G4_frecuencia_por_clase"]["rango"]) == [0.1, 0.3]
                            and G5_TOLERANCIA == 0.0
                            and float(puerta["G6_suma_unidad"]["tolerancia"]) == G6_TOLERANCIA),
        "puerta_prohibido": "reintentar con otro alpha" in PUERTA_PREREG["prohibido"],
        "sonda_atribucion": prereg["sonda_atribucion"] == SONDA_ATRIBUCION_PREREG,
        "sonda_atribucion_cap": "capado a MARGINAL"
                                in SONDA_ATRIBUCION_PREREG["consecuencia_si_no_se_publica"],
        "sondas": prereg["sondas"] == SONDAS_PREREG,
        "sondas_tolerancias": (TOL_SONDA_A == 1e-12 and TOL_SONDA_B == 1e-9
                               and TOL_SONDA_C == 1e-6 and TOL_SONDA_E == 1e-9
                               and TOL_SONDA_F == 1e-9
                               and SONDAS_PREREG["d_sin_nan_logloss_finito"] is True
                               and DIM_SONDA_A == (200, 50) and DIM_SONDA_D == (400, 20)),
        "sonda_e_es_nueva": "NUEVA en V33" in SONDAS_PREREG["e_logloss_multiclase_propio"]["nota"],
        "sondas_politica": "exit 4 ANTES de entrenar" in SONDAS_PREREG["politica"],
        "refit": prereg["sobre_refit"] == REFIT_PREREG,
        "refit_5_folds": "5 folds" in REFIT_PREREG["descripcion"],
        "determinismo": prereg["determinismo"] == DETERMINISMO_PREREG,
        "determinismo_seeds": (DETERMINISMO_PREREG["semilla"] == SEMILLA
                               and DETERMINISMO_PREREG["orden"]
                               == f"default_rng({SEMILLA}+fold) una vez por fold"),
        "determinismo_tolerancias": (TOL_DETERMINISMO == 1e-9 and VARIABILIDAD == 0.0008876
                                     and DETERMINISMO_PREREG["tolerancias"]
                                     == "0 exacto; <=1e-9 conforme; >1e-9 con rango publicado y "
                                        "delta_gate > rango; rango >= 0.0008876 o solape con el "
                                        "gate => no concluyente"),
        "hilos_efectivos": all(os.environ[clave] == str(HILOS) for clave in
                               ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")),
        "integracion": prereg["integracion"] == INTEGRACION_PREREG,
        "integracion_parametros": (C_LOGREG == 1.0 and MAX_ITER_LOGREG == 2000
                                   and L2_CLIP == 1e-12 and PESOS[0] == 1.0),
        "gates": prereg["gates"] == GATES_PREREG,
        "gates_parametros": (float(GATES_PREREG["baseline_abort"]) == ABORTO_BASELINE
                             and float(GATES_PREREG["delta_min"]) == GATE_DELTA
                             and int(GATES_PREREG["folds_min"]) == GATE_FOLDS
                             and GATES_PREREG["control"] == "delta_int(A1) > delta_int(A2)"
                             and GATES_PREREG["sin_leakage"] is True
                             and GATES_PREREG["determinismo_concluyente"] is True
                             and GATES_PREREG["sonda_atribucion_publicada"] is True),
        "gates_bootstrap": (GATES_PREREG["bootstrap"]["n"] == N_BOOTSTRAP
                            and GATES_PREREG["bootstrap"]["semilla"] == SEMILLA
                            and GATES_PREREG["bootstrap"]["ic95_sin_cero"] is True
                            and GATES_PREREG["bootstrap"]["diagnostico_adicional"]
                            == "IC95 pareado de la diferencia (A1-A2)"),
        "gates_caps": (GATES_PREREG["caps"]
                       == ["sin la sonda |sigma*w|/|g| publicada -> veredicto capado a MARGINAL",
                           "determinismo no concluyente -> el veredicto no puede ser PROMISING"]),
        "arbol": prereg["arbol"] == ARBOL_PREREG,
        "arbol_nodos": (set(ARBOL_PREREG) == {"NO_ATRIBUIBLE_BY_GATE", "PROMISING", "MARGINAL",
                                              "CLOSED", "submission"}),
        "arbol_segunda_particion": (f"semilla {SEMILLA_SEGUNDA_PARTICION}"
                                    in ARBOL_PREREG["PROMISING"]
                                    and "el script no la ejecuta" in ARBOL_PREREG["PROMISING"]),
        "novedad": prereg["novedad"] == NOVEDAD_PREREG,
        "novedad_no_es": "no es un embedding" in NOVEDAD_PREREG["no_es"],
        "via_b": prereg["via_b_descartada"] == VIA_B_PREREG,
        "via_b_b1_ilegal": VIA_B_PREREG["B1_grafo_transductivo"].startswith("ILEGAL"),
        "ejecucion": prereg["ejecucion"] == EJECUCION_PREREG,
        "ejecucion_ram": (float(EJECUCION_PREREG["ram_tope_gb"]) == RAM_TOPE_GB
                          and "ru_maxrss" in EJECUCION_PREREG["ram_medicion"]),
        "ejecucion_calibracion": (float(PROYECCION_FACTOR) == 9.0
                                  and float(PROYECCION_MAX_H) == 20.0
                                  and "9 x t_fold0" in EJECUCION_PREREG["calibracion"]),
        "ejecucion_timeout": (float(TIMEOUT_H) == 24.0
                              and EJECUCION_PREREG["timeout_global"] == "24.0 h"),
        "ejecucion_checkpoints": ruta_relativa(ESTADO_JSON) in EJECUCION_PREREG["checkpoints"],
        "ejecucion_atomica": ruta_relativa(OOF_DIR) + "/" in EJECUCION_PREREG["escritura_atomica"],
        "vectorizador_config": prereg["representacion"]["vectorizador"] == VECTORIZADOR,
        "vectorizador_ajuste_por_fold": prereg["representacion"]["ajuste"].startswith("POR FOLD"),
        "vectorizador_reutilizacion": "RECETA" in prereg["representacion"]["reutilizacion"],
        "corpus_receta": ("load_train" in prereg["corpus"]["receta"]
                          and "utf-8-sig" in prereg["corpus"]["receta"]),
        "corpus_hash_contenido": prereg["corpus"]["hash_contenido"] == (
            "sha256 con longitud (8 bytes LE) + bytes UTF-8 por texto, en orden"),
        "corpus_prohibido": "invalidado en D-083" in prereg["corpus"]["prohibido"],
        "compliance_cifras": {clave: prereg["compliance"][clave] for clave in COMPLIANCE_CIFRAS}
                            == COMPLIANCE_CIFRAS,
        "compliance_auditorias": prereg["compliance"]["auditorias_exigidas"]
                                 == COMPLIANCE_AUDITORIAS,
        "compliance_ast_audit": (prereg["compliance"]["ast_audit"].startswith("importes Y "
                                   "literales de ruta")
                                 and "spacy-vectors" in prereg["compliance"]["ast_audit"]),
        "entregables": all(nombre in entregable for nombre in nombres_entregables()),
    }
    fallos = [nombre for nombre, correcto in checks.items() if not correcto]
    for nombre in fallos:
        print(f"  [STOP] constante distinta a la congelada: {nombre}", flush=True)
    return (not fallos), checks


def verificar_versiones(prereg: dict) -> tuple[bool, dict]:
    """Comprueba que las versiones instaladas coinciden con las congeladas.

    Parameters
    ----------
    prereg:
        Contenido del pre-registro.

    Returns
    -------
    tuple[bool, dict]
        Exito global y detalle con versiones instaladas, esperadas y el chequeo por clave.
    """
    instaladas = {"sklearn": sklearn.__version__, "numpy": np.__version__,
                  "pandas": pd.__version__}
    checks = {clave: instaladas[clave] == valor for clave, valor in prereg["versiones"].items()}
    for clave, correcto in checks.items():
        if not correcto:
            print(f"  [STOP] version distinta en {clave}: {instaladas[clave]} != "
                  f"{prereg['versiones'][clave]}", flush=True)
    return all(checks.values()), {"instaladas": instaladas, "esperadas": prereg["versiones"],
                                  "checks": checks}


def verificar_hashes(prereg: dict, hashes_corpus: dict) -> tuple[bool, dict]:
    """Verifica el SHA256 de cada artefacto leido contra el JSON congelado.

    Los seis canonicos y el manifiesto se leen de disco; los cuatro del corpus (archivo, array de
    textos con la receta determinista, dict del vectorizador y JSON ancla de D-083) se recomputan.
    El ``data_hash`` del manifiesto NO se usa: quedo invalidado en D-083 (hash de punteros).

    Parameters
    ----------
    prereg:
        Contenido del pre-registro.
    hashes_corpus:
        Hashes recomputados del corpus y del ancla.

    Returns
    -------
    tuple[bool, dict]
        Exito global, detalle nombre -> booleano y los valores recomputados.
    """
    checks: dict[str, bool] = {}
    recomputados: dict[str, str] = {}
    for nombre, ruta in RUTAS_HASHES.items():
        if not ruta.exists():
            print(f"  [STOP] falta el artefacto {ruta}", flush=True)
            checks[nombre] = False
            continue
        recomputados[nombre] = sha256_archivo(ruta)
        checks[nombre] = recomputados[nombre] == prereg["hashes"][nombre]
        if not checks[nombre]:
            print(f"  [STOP] hash distinto en {nombre}", flush=True)
    for nombre in CLAVES_HASH_CORPUS:
        recomputados[nombre] = hashes_corpus[nombre]
        checks[nombre] = recomputados[nombre] == prereg["hashes"][nombre]
        if not checks[nombre]:
            print(f"  [STOP] hash distinto en {nombre}: {recomputados[nombre][:16]} != "
                  f"{prereg['hashes'][nombre][:16]}", flush=True)
    return all(checks.values()), {"checks": checks, "recomputados": recomputados,
                                  "esperados": prereg["hashes"]}


def cargar_canonico() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Carga y valida la particion canonica (y_true, fold_idx, v30_prob, ids, cand).

    Returns
    -------
    tuple
        Etiquetas int64, folds int64, probabilidades del campeon normalizadas, ids int64 y
        predicciones del candidato canonico.

    Raises
    ------
    Detencion
        Si los artefactos no tienen la alineacion congelada (169336 filas y 5 folds).
    """
    clase = np.load(RUTAS_HASHES["canonico_y_true"])
    fold_idx = np.load(RUTAS_HASHES["canonico_fold_idx"])
    v30 = np.load(RUTAS_HASHES["canonico_v30_prob"]).astype(np.float64)
    ids = np.load(RUTAS_HASHES["canonico_ids"])
    cand_pred = np.load(RUTAS_HASHES["canonico_cand_pred"])
    cand_prob = np.load(RUTAS_HASHES["canonico_cand_prob"]).astype(np.float64)
    if (clase.shape != (N_FILAS,) or fold_idx.shape != (N_FILAS,) or ids.shape != (N_FILAS,)
            or cand_pred.shape != (N_FILAS,)):
        raise Detencion(f"[STOP] los artefactos canonicos no tienen {N_FILAS} filas (exit 2)", 2)
    if v30.shape != (N_FILAS, N_CLASES) or cand_prob.shape != (N_FILAS, N_CLASES):
        raise Detencion("[STOP] v30_prob o cand_prob con forma inesperada (exit 2)", 2)
    if int(clase.min()) != 0 or int(clase.max()) != N_CLASES - 1:
        raise Detencion("[STOP] clases fuera del rango canonico (exit 2)", 2)
    if np.unique(fold_idx).size != N_FOLDS:
        raise Detencion(f"[STOP] se esperan {N_FOLDS} folds canonicos (exit 2)", 2)
    return (clase.astype(np.int64), fold_idx.astype(np.int64), normaliza(v30),
            ids.astype(np.int64), cand_pred.astype(np.int64))


def cargar_corpus(ids_canonicos: np.ndarray, clase: np.ndarray) -> tuple[np.ndarray, dict]:
    """Recarga el corpus con la receta congelada y verifica su identidad por hash.

    No se reordena ni se "arregla" nada: si el sha256 del archivo o el del array de textos no
    coinciden con los congelados, el runner se detiene con codigo 2.

    Parameters
    ----------
    ids_canonicos:
        ``ids.npy`` de la particion canonica (orden de referencia).
    clase:
        ``y_true`` canonico (etiquetas de referencia).

    Returns
    -------
    tuple[numpy.ndarray, dict]
        Textos en el orden del CSV y los cuatro hashes recomputados del corpus.

    Raises
    ------
    Detencion
        Si el numero de filas, el orden de los ids o las etiquetas no coinciden.
    """
    marco = datos.load_train()
    if len(marco) != N_FILAS:
        raise Detencion(f"[STOP] el corpus tiene {len(marco)} filas, no {N_FILAS} (exit 2)", 2)
    ids_csv = marco[cfg.ID_COL].to_numpy().astype(np.int64)
    etiquetas = marco[cfg.TARGET].cat.codes.to_numpy().astype(np.int64)
    textos = marco[cfg.TEXT_COL].fillna("").astype(str).to_numpy()
    if not np.array_equal(ids_canonicos, ids_csv):
        raise Detencion("[STOP] el orden de ids del corpus no coincide con ids.npy (exit 2)", 2)
    if not np.array_equal(clase, etiquetas):
        raise Detencion("[STOP] cat.codes del corpus no coincide con y_true canonico (exit 2)", 2)
    hashes = {"corpus_archivo": sha256_archivo(cfg.TRAIN_FILE),
              "corpus_textos": sha256_textos(textos),
              "vectorizador": sha256_bytes(json.dumps(VECTORIZADOR, sort_keys=True)
                                           .encode("utf-8")),
              "ancla_d083_json": sha256_archivo(ANCLA_D083)}
    return textos, hashes


def normaliza(p: np.ndarray) -> np.ndarray:
    """Clip [1e-12, 1] y renormalizacion L1 por fila (convencion del proyecto).

    Parameters
    ----------
    p:
        Matriz de puntuaciones o probabilidades (n, 6).

    Returns
    -------
    numpy.ndarray
        Matriz cuyas filas suman 1.
    """
    p = np.clip(p, L2_CLIP, 1.0)
    return p / p.sum(axis=1, keepdims=True)


def normaliza_l2(p: np.ndarray) -> np.ndarray:
    """Clip [1e-12, 1] y normalizacion L2 por fila (mecanismo P6/P7 y comparador canonico).

    Parameters
    ----------
    p:
        Matriz de puntuaciones o probabilidades (n, 6).

    Returns
    -------
    numpy.ndarray
        Matriz con norma 2 igual a 1 en cada fila.
    """
    p = np.clip(p, L2_CLIP, 1.0)
    norma = np.linalg.norm(p, axis=1, keepdims=True)
    return p / np.where(norma > L2_CLIP, norma, 1.0)


def macro(clase: np.ndarray, pred: np.ndarray) -> float:
    """macro-F1 de sklearn con las 6 clases canonicas declaradas.

    Parameters
    ----------
    clase:
        Etiquetas verdaderas.
    pred:
        Predicciones.

    Returns
    -------
    float
        macro-F1 en [0, 1].
    """
    return float(f1_score(clase, pred, average="macro", labels=list(range(N_CLASES))))


def macro_de_conteos(conteos: np.ndarray) -> float:
    """macro-F1 exacto a partir del histograma aplanado de la matriz de confusion 6x6.

    Parameters
    ----------
    conteos:
        Histograma de 36 celdas (clase verdadera x clase predicha, aplanado).

    Returns
    -------
    float
        macro-F1 en [0, 1].
    """
    matriz = conteos.reshape(N_CLASES, N_CLASES)
    f1s = []
    for c in range(N_CLASES):
        tp = float(matriz[c, c])
        fp = float(matriz[:, c].sum() - tp)
        fn = float(matriz[c, :].sum() - tp)
        denominador = 2.0 * tp + fp + fn
        f1s.append(2.0 * tp / denominador if denominador > 0 else 0.0)
    return float(np.mean(f1s))


def log_loss_propio(p: np.ndarray, y: np.ndarray) -> float:
    """Log-loss multiclase PROPIO sobre probabilidades que ya suman 1 por fila.

    Es la magnitud de la puerta G1/G2 de V33: ``-media_i log p[i, y[i]]`` con clip
    [1e-12, 1]. A diferencia del log-loss OvR de D-083, normaliza por construccion (softmax de
    los 6 logits), y la sonda (e) la contrasta contra ``sklearn.metrics.log_loss``.

    Parameters
    ----------
    p:
        Probabilidades (n, 6) cuyas filas suman 1.
    y:
        Etiquetas enteras en 0..5.

    Returns
    -------
    float
        Log-loss multiclase con las 6 clases canonicas.
    """
    objetivos = p[np.arange(len(y)), y]
    return float(-np.mean(np.log(np.clip(objetivos, L2_CLIP, 1.0))))


def log_loss_multiclase(p: np.ndarray, y: np.ndarray) -> float:
    """Log-loss multiclase de sklearn sobre probabilidades RENORMALIZADAS con la convencion
    del proyecto (clip [1e-12, 1] y L1 por fila).

    En sklearn 1.8.0 ``log_loss`` solo avisa ("The y_prob values do not sum to one") y **no**
    renormaliza por dentro, de modo que sin este paso devolveria una perdida calculada sobre
    valores que no son probabilidades. La renormalizacion es la misma que aplica
    ``combinador_oof``, por lo que la metrica es comparable entre brazos y ciclos.

    Parameters
    ----------
    p:
        Matriz de puntuaciones o probabilidades (n, 6), por ejemplo el OOF normalizado L2.
    y:
        Etiquetas enteras en 0..5.

    Returns
    -------
    float
        Log-loss multiclase con las 6 clases canonicas.
    """
    return float(log_loss(y, normaliza(p), labels=list(range(N_CLASES))))


def frecuencias(clase_pred: np.ndarray) -> list[float]:
    """Frecuencias relativas de las 6 clases predichas (magnitud de la puerta G4).

    Parameters
    ----------
    clase_pred:
        Clase predicha por el brazo, enteros en 0..5.

    Returns
    -------
    list[float]
        Las 6 frecuencias en el orden canonico de clases.
    """
    conteos = np.bincount(np.asarray(clase_pred, dtype=np.int64), minlength=N_CLASES)
    return [float(valor) / float(len(clase_pred)) for valor in conteos]


def max_abs_suma_menos_uno(p: np.ndarray) -> float:
    """Maximo de |sum_c p_c - 1| por fila (magnitud de la puerta G6).

    Parameters
    ----------
    p:
        Probabilidades (n, 6).

    Returns
    -------
    float
        Maximo de la desviacion de la suma por fila respecto de 1.
    """
    return float(np.abs(p.sum(axis=1) - 1.0).max())


def linea_base_mayoritaria(clase: np.ndarray) -> dict:
    """Ancla trivial: macro-F1 de predecir siempre la clase mayoritaria.

    Es el ancla congelada de la puerta G3 (0.04854212180687087) y se recalcula y se contrasta
    en tiempo de ejecucion; si no reproduce, el runner se detiene con codigo 2.

    Parameters
    ----------
    clase:
        Etiquetas canonicas (169336,).

    Returns
    -------
    dict
        Clase mayoritaria (indice y etiqueta), su frecuencia, el macro-F1 de esa linea base y
        las filas.

    Raises
    ------
    Detencion
        Si el macro-F1 recalculado no coincide con el ancla congelada (codigo 2).
    """
    conteos = np.bincount(clase, minlength=N_CLASES)
    mayoritaria = int(np.argmax(conteos))
    pred = np.full(len(clase), mayoritaria, dtype=np.int64)
    valor = macro(clase, pred)
    if abs(valor - ANCLA_TRIVIAL) > TOL_ANCLA_TRIVIAL:
        raise Detencion(f"[STOP] el ancla trivial recalculada {valor!r} no reproduce "
                        f"{ANCLA_TRIVIAL!r} (exit 2)", 2)
    return {"clase_indice": mayoritaria, "clase": cfg.CLASSES[mayoritaria],
            "frecuencia": float(conteos[mayoritaria] / len(clase)),
            "macro_f1_linea_base": valor, "n_filas": len(clase),
            "ancla_congelada": ANCLA_TRIVIAL,
            "diferencia": float(abs(valor - ANCLA_TRIVIAL))}


# =====================================================================================
# ---- 2. AUDITORIA AST DE COMPLIANCE DEL PROPIO SCRIPT ----
# =====================================================================================
def auditar_ast_propio() -> dict:
    """Audita el propio codigo fuente: 0 palabras vigiladas en literales de ruta y de modulos.

    Un literal se considera de ruta si contiene barra inclinada o termina en una extension de
    datos (.csv, .npy, .json, .npz, .txt). Ninguno puede contener, en minusculas, ninguna de
    las cuatro particiones reservadas ni ninguna de las diez familias vetadas. Ademas se
    revisan los MODULOS IMPORTADOS contra las familias vetadas, de modo que la ausencia de
    las familias vetadas no dependa solo de los literales de ruta.

    Returns
    -------
    dict
        ``{"ast_audit": True, "literales_de_ruta_revisados": n, "modulos_revisados": m,
        "violaciones": []}``.

    Raises
    ------
    Detencion
        Si hay algun literal de ruta vigilado o algun import de una familia vetada (codigo 2).
    """
    fuente = Path(__file__).read_text(encoding="utf-8")
    arbol = ast.parse(fuente)
    literales: list[tuple[int, str]] = []
    for nodo in ast.walk(arbol):
        if isinstance(nodo, ast.Constant) and isinstance(nodo.value, str):
            texto = nodo.value
            es_ruta = ("/" in texto) or texto.lower().endswith(EXTENSIONES_RUTA)
            if es_ruta:
                literales.append((int(nodo.lineno), texto))
    reservadas = [(linea, texto) for linea, texto in literales
                  if any(palabra in texto.lower() for palabra in PARTICIONES_RESERVADAS)]
    vetadas = [(linea, texto) for linea, texto in literales
               if any(familia in texto.lower() for familia in FAMILIAS_VETADAS)]
    modulos: list[tuple[int, str]] = []
    for nodo in ast.walk(arbol):
        if isinstance(nodo, ast.Import):
            modulos += [(int(nodo.lineno), alias.name) for alias in nodo.names]
        elif isinstance(nodo, ast.ImportFrom) and nodo.module:
            modulos.append((int(nodo.lineno), nodo.module))
    modulos_vetados = [(linea, modulo) for linea, modulo in modulos
                       if any(familia in modulo.lower() for familia in FAMILIAS_VETADAS)]
    for linea, texto in reservadas:
        print(f"  [STOP] literal de ruta con particion reservada ({texto}) en la linea {linea}",
              flush=True)
    for linea, texto in vetadas:
        print(f"  [STOP] literal de ruta con familia vetada ({texto}) en la linea {linea}",
              flush=True)
    for linea, modulo in modulos_vetados:
        print(f"  [STOP] import de una familia vetada ({modulo}) en la linea {linea}", flush=True)
    if reservadas or vetadas or modulos_vetados:
        raise Detencion("[STOP] auditoria AST de compliance fallida (exit 2)", 2)
    return {"ast_audit": True, "literales_de_ruta_revisados": len(literales),
            "modulos_revisados": len(modulos),
            "particiones_reservadas_vigiladas": list(PARTICIONES_RESERVADAS),
            "familias_vetadas": list(FAMILIAS_VETADAS), "violaciones": []}


# =====================================================================================
# ---- 3. FTRL-PROXIMAL SOFTMAX MULTICLASE PROPIO (UNA PASADA) ----
# =====================================================================================
def softmax(logits: np.ndarray) -> np.ndarray:
    """Softmax multiclase numericamente estable sobre la ultima dimension.

    Resta el maximo por fila antes de exponenciar: con logit muy grande o muy pequeno la
    exponencial cruda daria 0 o ``inf`` y las filas dejarian de sumar 1 (lo mide la sonda (f)).

    Parameters
    ----------
    logits:
        Matriz de logits (n, 6) o vector (6,).

    Returns
    -------
    numpy.ndarray
        Probabilidades con la misma forma, cuyas filas suman 1 hasta el redondeo de float64.
    """
    logits = np.asarray(logits, dtype=np.float64)
    desplazado = logits - logits.max(axis=-1, keepdims=True)
    exponenciales = np.exp(desplazado)
    return exponenciales / exponenciales.sum(axis=-1, keepdims=True)


def ftrl_softmax(x: sp.csr_matrix, y: np.ndarray, alpha: float, lambda1: float, lambda2: float,
                 beta: float, orden: np.ndarray, con_estado: bool = False,
                 con_atribucion: bool = False,
                 al_terminar_ejemplo: Callable[[int, float], None] | None = None
                 ) -> tuple[np.ndarray, dict | None]:
    """FTRL-Proximal softmax MULTICLASE propio, UNA sola pasada online (McMahan et al. 2013).

    K = 6 clases en UN SOLO juego de pesos: para cada ejemplo se calcula ``s = X @ W + b`` con
    el sesgo materializado como coordenada de valor ``x0 = 1.0`` (la ultima columna del CSR), se
    obtiene ``p = softmax(s)`` y el gradiente multiclase es ``g_c = p_c - y_c``. Solo se
    actualizan las coordenadas NO NULAS de la fila, que es la regla proximal congelada:
    ``sigma = (sqrt(n+g^2) - sqrt(n)) / (alpha*beta)``; ``z += g - sigma*w``; ``n += g^2``;
    ``w = 0 si |z| <= lambda1``, si no ``sign(z)*(|z|-lambda1)/((beta+sqrt(n))/alpha+lambda2)``.

    El recorrido es secuencial por ejemplo (FTRL es un metodo online) y la actualizacion se
    vectoriza sobre las clases y sobre las coordenadas no nulas de la fila. No hay
    aleatoriedad dentro del lazo: el unico orden es ``orden``, el barajado congelado.

    Parameters
    ----------
    x:
        Matriz CSR float64 con la coordenada del sesgo ya anadida (ultima columna, x0).
    y:
        Etiquetas enteras en 0..5 (n,).
    alpha, lambda1, lambda2, beta:
        Hiperparametros congelados; alpha es el DERIVADO 3/(2*sqrt(T)) de este ajuste.
    orden:
        Permutacion de ``range(x.shape[0])``: el unico recorrido de la pasada.
    con_estado:
        Si True devuelve tambien los acumuladores ``z`` y ``n`` (sonda analitica).
    con_atribucion:
        Si True acumula la razon de correccion ``|sigma*w|/|g|`` por coordenada (sonda
        obligatoria del pre-registro).
    al_terminar_ejemplo:
        Llamada opcional ``(i_ordinal, perdida_media_acumulada)`` tras cada ejemplo, para
        registrar la curva online de log-loss de train.

    Returns
    -------
    tuple[numpy.ndarray, dict | None]
        Pesos ``w`` con forma (n_coordenadas, 6) y, si se pidio, el estado interno con ``z``,
        ``n`` y, si se pidio, los acumuladores de atribucion.

    Raises
    ------
    Detencion
        Si el recorrido no es una permutacion valida de las filas (codigo 2).
    """
    n_filas, n_coordenadas = x.shape
    if orden.shape[0] != n_filas or not np.array_equal(np.sort(orden),
                                                        np.arange(n_filas)):
        raise Detencion("[STOP] el orden congelado no es una permutacion de las filas (exit 2)", 2)
    w = np.zeros((n_coordenadas, N_CLASES), dtype=np.float64)
    z = np.zeros((n_coordenadas, N_CLASES), dtype=np.float64)
    n = np.zeros((n_coordenadas, N_CLASES), dtype=np.float64)
    acc_sigma = np.zeros((n_coordenadas, N_CLASES), dtype=np.float64) if con_atribucion else None
    acc_g = np.zeros((n_coordenadas, N_CLASES), dtype=np.float64) if con_atribucion else None
    perdida_acumulada = 0.0
    for ordinal, i in enumerate(orden):
        inicio, fin = int(x.indptr[i]), int(x.indptr[i + 1])
        indices = x.indices[inicio:fin]
        valores = x.data[inicio:fin]
        s = np.einsum("ij,i->j", w[indices], valores)
        p = softmax(s)
        g = p.copy()
        g[y[i]] -= 1.0
        g_coord = g[None, :] * valores[:, None]
        n_previo = n[indices]
        n_nuevo = n_previo + g_coord * g_coord
        sigma = (np.sqrt(n_nuevo) - np.sqrt(n_previo)) / (alpha * beta)
        w_previo = w[indices]
        if con_atribucion:
            acc_sigma[indices] += np.abs(sigma * w_previo)
            acc_g[indices] += np.abs(g_coord)
        z[indices] = z[indices] + g_coord - sigma * w_previo
        n[indices] = n_nuevo
        z_nuevo = z[indices]
        w[indices] = np.where(
            np.abs(z_nuevo) <= lambda1, 0.0,
            np.sign(z_nuevo) * (np.abs(z_nuevo) - lambda1)
            / ((beta + np.sqrt(n_nuevo)) / alpha + lambda2))
        if al_terminar_ejemplo is not None:
            perdida_acumulada += -math.log(max(float(p[y[i]]), L2_CLIP))
            al_terminar_ejemplo(ordinal, perdida_acumulada)
    estado: dict | None = None
    if con_estado or con_atribucion:
        estado = {"z": z, "n": n}
        if con_atribucion:
            estado["acc_sigma"] = acc_sigma
            estado["acc_g"] = acc_g
    if con_estado:
        return w, estado
    return w, estado if con_atribucion else None


def ftrl_softmax_lazo_ingenuo(x_denso: np.ndarray, y: np.ndarray, alpha: float,
                              lambda1: float, lambda2: float, beta: float,
                              orden: np.ndarray) -> np.ndarray:
    """Lazo ingenuo de FTRL softmax multiclase (listas de Python, sin numpy en el interior).

    Es la referencia independiente de la sonda (a): misma actualizacion, mismo softmax y misma
    permutacion, pero recorriendo una coordenada y una clase por vez con escalares de Python.

    Parameters
    ----------
    x_denso:
        Matriz densa float64 (n, d) con la coordenada del sesgo incluida.
    y:
        Etiquetas enteras en 0..5 (n,).
    alpha, lambda1, lambda2, beta:
        Hiperparametros congelados.
    orden:
        Permutacion de las filas: el unico recorrido de la pasada.

    Returns
    -------
    numpy.ndarray
        Matriz de pesos (d, 6) al terminar la pasada.
    """
    n_coordenadas = x_denso.shape[1]
    w = [[0.0] * N_CLASES for _ in range(n_coordenadas)]
    z = [[0.0] * N_CLASES for _ in range(n_coordenadas)]
    n = [[0.0] * N_CLASES for _ in range(n_coordenadas)]
    for i in orden:
        fila = x_denso[i]
        s = []
        for c in range(N_CLASES):
            suma = 0.0
            for j in range(n_coordenadas):
                suma += w[j][c] * fila[j]
            s.append(suma)
        maximo = max(s)
        exponenciales = [math.exp(valor - maximo) for valor in s]
        total = 0.0
        for valor in exponenciales:
            total += valor
        p = [valor / total for valor in exponenciales]
        p[y[i]] -= 1.0
        for j in range(n_coordenadas):
            xj = fila[j]
            for c in range(N_CLASES):
                g = p[c] * xj
                raiz_previa = math.sqrt(n[j][c])
                raiz = math.sqrt(n[j][c] + g * g)
                sigma = (raiz - raiz_previa) / (alpha * beta)
                z[j][c] += g - sigma * w[j][c]
                n[j][c] += g * g
                w[j][c] = (0.0 if abs(z[j][c]) <= lambda1 else
                           math.copysign(1.0, z[j][c]) * (abs(z[j][c]) - lambda1)
                           / ((beta + math.sqrt(n[j][c])) / alpha + lambda2))
    return np.array(w, dtype=np.float64)


def problema_aleatorio(n_filas: int, n_features: int, semilla: int
                       ) -> tuple[sp.csr_matrix, np.ndarray]:
    """Problema aleatorio denso (n, d+1) con sesgo, empaquetado en CSR canonico.

    La ultima columna es la coordenada del sesgo con valor ``x0``; las primeras ``n_features``
    son continuas en [0, 1). Se usa en las sondas (a), (d), (e) y (f).

    Parameters
    ----------
    n_filas:
        Numero de ejemplos.
    n_features:
        Numero de columnas SIN la coordenada del sesgo.
    semilla:
        Semilla del generador.

    Returns
    -------
    tuple[scipy.sparse.csr_matrix, numpy.ndarray]
        Matriz CSR con sesgo y etiquetas enteras en 0..5.
    """
    generador = np.random.default_rng(semilla)
    denso = generador.random((n_filas, n_features + 1))
    denso[:, -1] = X0
    y = generador.integers(0, N_CLASES, size=n_filas)
    return sp.csr_matrix(denso), y.astype(np.int64)


def razon_atribucion(estado: dict, n_coordenadas: int, n_ejemplos: int) -> dict:
    """Razon media ``|sigma*w| / |g|`` por coordenada a partir de los acumuladores.

    Para cada coordenada ``j`` se divide la correccion acumulada ``|sigma*w|`` por el gradiente
    acumulado ``|g|`` (sumando sobre clases y sobre las actualizaciones que tocaron ``j``); el
    resultado es UN numero por coordenada. Una razon mayor que 1 significa que la correccion
    proximal domina al gradiente: es la firma que la sonda debe desambujar en D-083 (correccion
    inflada) frente a la multi-pasada.

    Parameters
    ----------
    estado:
        Estado devuelto por ``ftrl_softmax`` con ``con_atribucion=True``.
    n_coordenadas:
        Numero de coordenadas del problema.
    n_ejemplos:
        Numero de ejemplos de la pasada (numero de actualizaciones acumuladas).

    Returns
    -------
    dict
        Media, mediana, percentil 90 y maximo de la razon por coordenada, la razon global, la
        razon por clase, la razon de la coordenada del sesgo, el numero de coordenadas tocadas y
        el de actualizaciones.
    """
    acc_sigma = estado["acc_sigma"].sum(axis=1)
    acc_g = estado["acc_g"].sum(axis=1)
    tocadas = acc_g > 0.0
    razones = np.zeros(n_coordenadas, dtype=np.float64)
    razones[tocadas] = acc_sigma[tocadas] / acc_g[tocadas]
    por_clase = []
    for c in range(N_CLASES):
        denominador = float(estado["acc_g"][:, c].sum())
        por_clase.append(float(estado["acc_sigma"][:, c].sum() / denominador)
                         if denominador > 0 else 0.0)
    sesgo = float(razones[n_coordenadas - 1]) if n_coordenadas > 0 else 0.0
    seleccionada = razones[tocadas]
    return {
        "definicion": "razon media |sigma*w|/|g| por coordenada (coordenadas no nulas de la "
                      "pasada); 1 significa correccion igual al gradiente",
        "n_coordenadas": int(n_coordenadas),
        "n_coordenadas_tocadas": int(tocadas.sum()),
        "n_actualizaciones": int(n_ejemplos),
        "razon_media_coordenadas": float(np.mean(seleccionada)) if seleccionada.size else 0.0,
        "razon_mediana_coordenadas": float(np.median(seleccionada)) if seleccionada.size else 0.0,
        "razon_p90_coordenadas": float(np.percentile(seleccionada, 90)) if seleccionada.size
                                 else 0.0,
        "razon_max_coordenadas": float(np.max(seleccionada)) if seleccionada.size else 0.0,
        "razon_global": float(acc_sigma.sum() / acc_g.sum()) if acc_g.sum() > 0 else 0.0,
        "razon_por_clase": por_clase,
        "razon_coordenada_sesgo": sesgo,
        "fraccion_coordenadas_con_razon_supera_1": float((seleccionada > 1.0).mean())
                                               if seleccionada.size else 0.0,
    }


# =====================================================================================
# ---- 4. SONDAS OBLIGATORIAS (a)-(f), ANTES DE ENTRENAR (exit 4) ----
# =====================================================================================
def sonda_a_vectorizado_vs_lazo() -> dict:
    """Sonda (a): la version vectorizada de FTRL softmax coincide con el lazo ingenuo.

    Problema aleatorio de 200 filas x 50 features (mas la coordenada del sesgo) x 6 clases,
    misma permutacion sembrada y una sola pasada; se compara el maximo de la diferencia
    absoluta de TODOS los pesos de las 6 clases.

    Returns
    -------
    dict
        Dimensiones, maximo de la diferencia, tolerancia y ``pass``.
    """
    n_filas, n_features = DIM_SONDA_A
    x, y = problema_aleatorio(n_filas, n_features, 7)
    denso = np.asarray(x.todense(), dtype=np.float64)
    alpha = alpha_derivado(n_filas)
    orden = np.random.default_rng(SEMILLA).permutation(n_filas)
    w_vector, _ = ftrl_softmax(x, y, alpha, LAMBDA1, LAMBDA2, BETA, orden)
    w_lazo = ftrl_softmax_lazo_ingenuo(denso, y, alpha, LAMBDA1, LAMBDA2, BETA, orden)
    peor = float(np.abs(w_lazo - w_vector).max())
    return {"descripcion": f"FTRL softmax multiclase vectorizado vs lazo ingenuo, problema "
                           f"aleatorio {n_filas}x{n_features}x{N_CLASES}, una pasada",
            "alpha": alpha, "max_abs_diff": peor, "tolerancia": TOL_SONDA_A,
            "pass": bool(peor <= TOL_SONDA_A)}


def sonda_b_raiz_analitica_lambda0() -> dict:
    """Sonda (b): con lambda1=lambda2=0 el peso es la raiz analitica de FTRL.

    Con las dos penalizaciones a cero, el peso de cada coordenada es el minimo analitico de la
    cuadratica proximal ``(1/2)((beta+sqrt(n))/alpha) w^2 - z w``, es decir
    ``w = alpha*z/(beta+sqrt(n))``. Se comprueba con dos residuales exactos: el residuo de esa
    raiz sobre el estado final (feature constante repetida, clase 0) y el valor de un solo paso
    (softmax de 6 ceros -> p = 1/6 -> ``g_0 = -5/6`` y ``g_c = 1/6``).

    Returns
    -------
    dict
        Residuos de la raiz analitica y del paso unico, umbral y ``pass``.
    """
    alpha = alpha_derivado(25)
    constante = sp.csr_matrix(np.ones((25, 1), dtype=np.float64))
    y = np.zeros(25, dtype=np.int64)
    orden = np.arange(25, dtype=np.int64)
    w, estado = ftrl_softmax(constante, y, alpha, 0.0, 0.0, BETA, orden, con_estado=True)
    raiz = estado["z"][0] * alpha / (BETA + np.sqrt(estado["n"][0]))
    residuo_raiz = float(np.abs(w[0] - raiz).max())
    w_uno, _ = ftrl_softmax(sp.csr_matrix(np.ones((1, 1))), np.zeros(1, dtype=np.int64), alpha,
                            0.0, 0.0, BETA, np.zeros(1, dtype=np.int64))
    esperado = alpha * np.array([-5.0 / 6.0] + [1.0 / 6.0] * (N_CLASES - 1)) / (
        BETA + np.array([5.0 / 6.0] + [1.0 / 6.0] * (N_CLASES - 1)))
    residuo_paso = float(np.abs(w_uno[0] - esperado).max())
    return {"descripcion": "lambda1=lambda2=0 con feature constante: raiz analitica "
                           "alpha*z/(beta+sqrt(n)) y paso unico alpha*g/(beta+|g|) con p=1/6",
            "alpha": alpha, "residuo_raiz": residuo_raiz, "residuo_un_paso": residuo_paso,
            "residuo_max": max(residuo_raiz, residuo_paso), "umbral": TOL_SONDA_B,
            "pass": bool(max(residuo_raiz, residuo_paso) < TOL_SONDA_B)}


def sonda_c_totalf1_macro() -> dict:
    """Sonda (c): ``TotalF1:average=Macro`` de CatBoost equivale al macro-F1 de sklearn.

    Se conservan las dos metricas como diagnostico: el default ``TotalF1`` de CatBoost
    promedia Weighted y NO coincide con macro-F1 (por eso el pre-registro fija la variante
    Macro). El criterio de PASS es solo el delta de la variante Macro.

    Returns
    -------
    dict
        TotalF1 default, TotalF1 average=Macro, macro-F1 de sklearn, deltas, tolerancia y
        ``pass``.
    """
    from catboost.utils import eval_metric as metrica_catboost

    generador = np.random.default_rng(3)
    n_filas = 4000
    y = generador.integers(0, N_CLASES, size=n_filas)
    p = normaliza(generador.random((n_filas, N_CLASES)) ** 3)
    total_macro = float(metrica_catboost(y, p, "TotalF1:average=Macro")[0])
    total_default = float(metrica_catboost(y, p, "TotalF1")[0])
    macro_sklearn = macro(y, p.argmax(axis=1))
    delta = abs(total_macro - macro_sklearn)
    return {"metrica": "TotalF1:average=Macro", "total_f1_average_macro": total_macro,
            "total_f1_default": total_default, "macro_f1_sklearn": macro_sklearn,
            "delta_criterio": delta, "delta_default_vs_macro": abs(total_default - macro_sklearn),
            "tolerancia": TOL_SONDA_C, "pass": bool(delta <= TOL_SONDA_C)}


def sonda_d_sin_nan_logloss_finito() -> dict:
    """Sonda (d): la p del softmax no tiene NaN/inf y el log_loss multiclase es finito.

    Returns
    -------
    dict
        ``p_finita``, ``log_loss_finito``, valores y ``pass``.
    """
    n_filas, n_features = DIM_SONDA_D
    x, y = problema_aleatorio(n_filas, n_features, 13)
    alpha = alpha_derivado(n_filas)
    orden = np.random.default_rng(SEMILLA).permutation(n_filas)
    w, _ = ftrl_softmax(x, y, alpha, LAMBDA1, LAMBDA2, BETA, orden)
    p = softmax(x.dot(w))
    valor = log_loss_propio(p, y)
    finita = bool(np.isfinite(p).all() and np.isfinite(w).all() and math.isfinite(valor))
    return {"descripcion": "softmax multiclase sin NaN/inf y log_loss multiclase finito",
            "n_filas": int(x.shape[0]), "p_finita": bool(np.isfinite(p).all()),
            "w_finito": bool(np.isfinite(w).all()), "log_loss_multiclase": valor,
            "log_loss_finito": bool(math.isfinite(valor)), "p_min": float(p.min()),
            "p_max": float(p.max()), "pass": finita}


def sonda_e_logloss_multiclase_propio() -> dict:
    """Sonda (e) NUEVA: el log_loss multiclase propio coincide con el de sklearn.

    D-083 solo probo el OvR, que no normaliza. Aqui el contraste es sobre probabilidades que
    SI suman 1: se compara ``-media_i log p[i, y[i]]`` propio contra
    ``sklearn.metrics.log_loss`` con las 6 clases declaradas, y se publica tambien la
    desviacion maxima de la suma por fila respecto de 1.

    Returns
    -------
    dict
        Valor propio, valor de sklearn, delta, tolerancia, desviacion de la suma y ``pass``.
    """
    n_filas, n_features = DIM_SONDA_D
    x, y = problema_aleatorio(n_filas, n_features, 29)
    w, _ = ftrl_softmax(x, y, alpha_derivado(n_filas), LAMBDA1, LAMBDA2, BETA,
                        np.random.default_rng(SEMILLA).permutation(n_filas))
    p = softmax(x.dot(w))
    propio = log_loss_propio(p, y)
    referencia = float(log_loss(y, p, labels=list(range(N_CLASES))))
    delta = abs(propio - referencia)
    suma = max_abs_suma_menos_uno(p)
    return {"descripcion": "log_loss multiclase propio -media log p[i,y[i]] contra "
                           "sklearn.metrics.log_loss con filas que suman 1",
            "log_loss_propio": propio, "log_loss_sklearn": referencia, "delta": delta,
            "max_abs_suma_menos_uno": suma, "tolerancia": TOL_SONDA_E,
            "nota": SONDAS_PREREG["e_logloss_multiclase_propio"]["nota"],
            "pass": bool(delta <= TOL_SONDA_E)}


def sonda_f_filas_suman_uno() -> dict:
    """Sonda (f): las filas del softmax de 6 logits suman 1 (con y sin temperatura).

    Se prueban tres familias de logits: la salida real de una pasada FTRL, logits aleatorios
    moderados y logits extremos (1e3), donde una exponencial cruda se desbordaria.

    Returns
    -------
    dict
        Desviaciones maximas por familia, tolerancia y ``pass``.
    """
    n_filas, n_features = DIM_SONDA_D
    x, y = problema_aleatorio(n_filas, n_features, 31)
    w, _ = ftrl_softmax(x, y, alpha_derivado(n_filas), LAMBDA1, LAMBDA2, BETA,
                        np.random.default_rng(SEMILLA).permutation(n_filas))
    generador = np.random.default_rng(37)
    logits = generador.normal(0.0, 3.0, size=(n_filas, N_CLASES))
    extremos = logits.copy()
    extremos[::2] += 1000.0
    extremos[1::2] -= 1000.0
    casos = {"ftrl_una_pasada": softmax(x.dot(w)), "aleatorio": softmax(logits),
             "extremos_1e3": softmax(extremos)}
    desvios = {nombre: max_abs_suma_menos_uno(valores) for nombre, valores in casos.items()}
    peor = float(max(desvios.values()))
    return {"descripcion": "max|sum_c p_c - 1| del softmax de 6 logits, sin temperatura",
            "max_abs_suma_menos_uno_por_caso": desvios, "peor": peor,
            "tolerancia": TOL_SONDA_F, "pass": bool(peor <= TOL_SONDA_F)}


NOMBRES_SONDAS = ("a_vectorizado_vs_lazo", "b_raiz_analitica_lambda0", "c_totalf1_macro",
                  "d_sin_nan_logloss_finito", "e_logloss_multiclase_propio", "f_filas_suman_uno")


def ejecutar_sondas() -> dict:
    """Ejecuta las sondas obligatorias (a)-(f) antes de cualquier ajuste.

    Returns
    -------
    dict
        Detalle de las seis sondas con valores, tolerancias y ``todas_pasan``.

    Raises
    ------
    Detencion
        Si alguna sonda falla (codigo 4), tal como fija la politica del pre-registro.
    """
    inicio = time.time()
    sondas = {"a_vectorizado_vs_lazo": sonda_a_vectorizado_vs_lazo(),
              "b_raiz_analitica_lambda0": sonda_b_raiz_analitica_lambda0(),
              "c_totalf1_macro": sonda_c_totalf1_macro(),
              "d_sin_nan_logloss_finito": sonda_d_sin_nan_logloss_finito(),
              "e_logloss_multiclase_propio": sonda_e_logloss_multiclase_propio(),
              "f_filas_suman_uno": sonda_f_filas_suman_uno()}
    detalle = dict(sondas)
    detalle["todas_pasan"] = bool(all(sondas[nombre]["pass"] for nombre in NOMBRES_SONDAS))
    detalle["politica"] = SONDAS_PREREG["politica"]
    detalle["duracion_seg"] = float(time.time() - inicio)
    for nombre in NOMBRES_SONDAS:
        print(f"  sonda ({nombre[0]}) {nombre}: pass={sondas[nombre]['pass']}", flush=True)
    if not detalle["todas_pasan"]:
        fallidas = [nombre for nombre in NOMBRES_SONDAS if not sondas[nombre]["pass"]]
        raise Detencion(f"[STOP] sondas obligatorias fallidas {fallidas} (exit 4)", 4)
    return detalle


# =====================================================================================
# ---- 5. CHECKPOINT, VECTORIZADOR POR FOLD Y ENTRENAMIENTO DE UN BRAZO ----
# =====================================================================================
def estado_nuevo() -> dict:
    """Estado resumible inicial del checkpoint.

    Returns
    -------
    dict
        Checkpoint vacio con inicio, guardas, calibracion, vectorizador por fold, la puerta y
        los dos brazos.
    """
    return {
        "fase": FASE_ESTADO,
        "inicio_ts": float(time.time()),
        "timeout_global_h": TIMEOUT_H,
        "ram_tope_gb": RAM_TOPE_GB,
        "rss_pico_gb": 0.0,
        "sondas": None,
        "calibracion": None,
        "puerta": None,
        "refit": None,
        "vectorizador_por_fold": {},
        "brazos": {brazo: {"folds_completados": [], "folds": {},
                           "mascara_relleno_por_fila": [0] * N_FILAS,
                           "sonda_determinismo": None} for brazo in BRAZOS},
    }


def guardar_estado(estado: dict) -> None:
    """Persiste el checkpoint con escritura atomica (temporal + os.replace).

    Parameters
    ----------
    estado:
        Estado a persistir.
    """
    escribir_json(ESTADO_JSON, estado)


def cargar_estado() -> dict:
    """Carga el checkpoint resumible o lo crea si no existe.

    Returns
    -------
    dict
        Estado con folds completados, magnitudes por fold, mascara de relleno y guardas.

    Raises
    ------
    Detencion
        Si el checkpoint existe pero no corresponde a V33-E1 o esta incompleto (codigo 2).
    """
    if not ESTADO_JSON.exists():
        estado = estado_nuevo()
        guardar_estado(estado)
        return estado
    estado = json.loads(ESTADO_JSON.read_text(encoding="utf-8"))
    if estado.get("fase") != FASE_ESTADO:
        raise Detencion(f"[STOP] el checkpoint no corresponde a {FASE_ESTADO} (exit 2)", 2)
    for clave in ("inicio_ts", "rss_pico_gb", "calibracion", "puerta", "refit",
                  "vectorizador_por_fold", "brazos"):
        if clave not in estado:
            raise Detencion(f"[STOP] checkpoint sin la clave {clave} (exit 2)", 2)
    for brazo in BRAZOS:
        datos = estado["brazos"].get(brazo)
        if not isinstance(datos, dict):
            raise Detencion(f"[STOP] checkpoint sin el brazo {brazo} (exit 2)", 2)
        if len(datos.get("mascara_relleno_por_fila", [])) != N_FILAS:
            raise Detencion(f"[STOP] mascara de relleno invalida en {brazo} (exit 2)", 2)
        if datos.get("folds_completados") != sorted(datos.get("folds", {})):
            raise Detencion(f"[STOP] folds completados y detalle incoherentes en {brazo} (exit 2)",
                            2)
        for fold, detalle in datos.get("folds", {}).items():
            faltan = [campo for campo in CAMPOS_FOLD if campo not in detalle]
            if faltan:
                raise Detencion(f"[STOP] checkpoint sin campos {faltan} en {brazo}/{fold} (exit 2)",
                                2)
    return estado


def mascara_de_folds(fold_idx: np.ndarray, folds: list[str]) -> np.ndarray:
    """Mascara por fila con 1 en las filas de los folds indicados.

    Parameters
    ----------
    fold_idx:
        Fold canonico por fila.
    folds:
        Folds incluidos.

    Returns
    -------
    numpy.ndarray
        Mascara int64 de 169336 posiciones.
    """
    mascara = np.zeros(N_FILAS, dtype=np.int64)
    for fold in folds:
        mascara[fold_idx == int(fold)] = 1
    return mascara


def guardar_oof(brazo: str, oof: np.ndarray) -> str:
    """Persiste el OOF del brazo con escritura atomica (temporal + os.replace).

    El OOF son las probabilidades CRUDAS del brazo (softmax de 6 logits en A1, ``predict_proba``
    en A2): ambas suman 1 por fila por construccion, que es justo lo que mide la puerta G6.

    Parameters
    ----------
    brazo:
        Nombre del brazo.
    oof:
        Matriz float64 (169336, 6).

    Returns
    -------
    str
        Ruta relativa del artefacto escrito.
    """
    OOF_DIR.mkdir(parents=True, exist_ok=True)
    ruta = ruta_oof(brazo)
    temporal = ruta.with_name(ruta.name + ".tmp")
    with temporal.open("wb") as manejador:
        np.save(manejador, oof)
    os.replace(temporal, ruta)
    return ruta_relativa(ruta)


def asegurar_brazo_cargado(estado: dict, brazo: str, fold_idx: np.ndarray) -> np.ndarray:
    """Devuelve el OOF del brazo (ceros si es nuevo) y valida su coherencia.

    Parameters
    ----------
    estado:
        Checkpoint cargado.
    brazo:
        Nombre del brazo.
    fold_idx:
        Fold canonico por fila.

    Returns
    -------
    numpy.ndarray
        OOF float64 (169336, 6).

    Raises
    ------
    Detencion
        Si el OOF y el checkpoint no son coherentes (codigo 2).
    """
    datos = estado["brazos"][brazo]
    completados = [str(valor) for valor in datos["folds_completados"]]
    ruta = ruta_oof(brazo)
    if not ruta.exists():
        if completados:
            raise Detencion(f"[STOP] el checkpoint dice que {brazo} tiene folds completados y no "
                            "existe su OOF (exit 2)", 2)
        return np.zeros((N_FILAS, N_CLASES), dtype=np.float64)
    with ruta.open("rb") as manejador:
        oof = np.load(manejador)
    if oof.shape != (N_FILAS, N_CLASES) or oof.dtype != np.float64:
        raise Detencion(f"[STOP] OOF de {brazo} con forma/tipo inesperado (exit 2)", 2)
    relleno = (oof.sum(axis=1) > 0.0).astype(np.int64)
    if not np.array_equal(relleno, mascara_de_folds(fold_idx, completados)):
        raise Detencion(f"[STOP] el OOF de {brazo} no coincide con los folds completados (exit 2)",
                        2)
    if not np.isfinite(oof).all():
        raise Detencion(f"[STOP] el OOF de {brazo} contiene no finitos (exit 2)", 2)
    return oof.astype(np.float64)


def anadir_sesgo(x: sp.csr_matrix, x0: float = X0) -> sp.csr_matrix:
    """Anade la coordenada del sesgo (ultima columna, valor constante ``x0``) a un CSR.

    El sesgo es una coordenada MAS de FTRL, con su propio ``n`` y su propio ``z``: por eso se
    materializa como columna y no como termino aparte, de modo que ``s = X @ W + b`` queda
    escrito como un unico producto contra una matriz de pesos de (d+1, 6). Se anade
    explicitamente en todas las filas (incluidas las que quedaron sin terminos tras
    ``min_df``/``max_df``) y el resultado queda en formato CSR canonico.

    Parameters
    ----------
    x:
        Matriz CSR float64 (n, d) sin sesgo.
    x0:
        Valor de la coordenada del sesgo (1.0 congelado).

    Returns
    -------
    scipy.sparse.csr_matrix
        Matriz (n, d+1) con indices ordenados y formato canonico.
    """
    n_filas, n_features = x.shape
    filas = np.concatenate([np.repeat(np.arange(n_filas, dtype=np.int64), np.diff(x.indptr)),
                            np.arange(n_filas, dtype=np.int64)])
    columnas = np.concatenate([x.indices.astype(np.int64),
                               np.full(n_filas, n_features, dtype=np.int64)])
    valores = np.concatenate([x.data.astype(np.float64), np.full(n_filas, x0)])
    matriz = sp.csr_matrix((valores, (filas, columnas)), shape=(n_filas, n_features + 1))
    if not matriz.has_sorted_indices:
        raise Detencion("[STOP] la matriz con sesgo no tiene indices ascendentes (exit 2)", 2)
    if not matriz.has_canonical_format:
        raise Detencion("[STOP] la matriz con sesgo no esta en forma canonica (exit 2)", 2)
    return matriz


def ajustar_vectorizador_fold(textos: np.ndarray, tr_idx: np.ndarray, va_idx: np.ndarray
                             ) -> dict:
    """Ajusta el TF-IDF SOLO en las filas de fold-train y transforma el fold-val.

    Los ``min_df``/``max_df``/``idf`` nunca son globales: se recalculan en cada fold, que es el
    precedente canonico del manifiesto. Se registran los hashes del vocabulario ordenado y del
    idf para poder verificar cualquier reentreno posterior.

    Parameters
    ----------
    textos:
        Array de textos del corpus en orden canonico.
    tr_idx:
        Indices de fold-train.
    va_idx:
        Indices de fold-val.

    Returns
    -------
    dict
        Vectorizador ajustado, matrices de train y val con sesgo y hashes del fold.
    """
    vectorizador = TfidfVectorizer(**VECTORIZADOR_FIT)
    x_tr = vectorizador.fit_transform(textos[tr_idx])
    x_va = vectorizador.transform(textos[va_idx])
    return {"vectorizador": vectorizador, "x_tr": anadir_sesgo(x_tr),
            "x_va": anadir_sesgo(x_va),
            "hashes": {"n_filas_fit": int(tr_idx.size),
                       "n_vocabulario": len(vectorizador.vocabulary_),
                       "vocab_sha256": sha256_textos(sorted(vectorizador.vocabulary_)),
                       "idf_sha256": sha256_f64(vectorizador.idf_)}}


def indices_fold(fold_idx: np.ndarray, fold: int) -> tuple[np.ndarray, np.ndarray]:
    """Indices de fold-train y fold-val del fold canonico.

    Parameters
    ----------
    fold_idx:
        Fold canonico por fila.
    fold:
        Fold a tomar como validacion.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        ``tr_idx`` y ``va_idx`` (orden ascendente, sin solapes).
    """
    return np.flatnonzero(fold_idx != fold), np.flatnonzero(fold_idx == fold)


def orden_congelado(fold: int, n_filas_fold: int) -> np.ndarray:
    """Orden de datos CONGELADO del fold: una sola permutacion sembrada, reutilizada por A2.

    Parameters
    ----------
    fold:
        Fold canonico.
    n_filas_fold:
        Numero de filas del fold-train.

    Returns
    -------
    numpy.ndarray
        Permutacion int64 de ``range(n_filas_fold)``.
    """
    return np.random.default_rng(SEMILLA + fold).permutation(n_filas_fold).astype(np.int64)


def kwargs_sgd(fold: int) -> dict:
    """Argumentos del estimador A2 derivados de la config congelada.

    Parameters
    ----------
    fold:
        Fold canonico (la semilla es ``20261010 + fold``).

    Returns
    -------
    dict
        Argumentos de ``SGDClassifier`` identicos a los congelados.
    """
    return {"loss": A2_SGD["loss"], "alpha": SGD_ALPHA, "max_iter": SGD_MAX_ITER,
            "tol": A2_SGD["tol"], "fit_intercept": A2_SGD["fit_intercept"],
            "average": A2_SGD["average"], "learning_rate": A2_SGD["learning_rate"],
            "shuffle": A2_SGD["shuffle"],
            "random_state": SEMILLA + fold, "n_jobs": A2_SGD["n_jobs"]}


def filas_prefijo(x: sp.csr_matrix, orden: np.ndarray, fraccion: float) -> sp.csr_matrix:
    """Submatriz del fold-train que corresponde a un prefijo del orden congelado.

    Es la instrumentacion de la puerta G1/G2: el modelo que produce el OOF es SIEMPRE el de la
    pasada completa; estas corridas de prefijo no son modelos, son la medida de la magnitud
    ``ll_train`` en el 25 %, 50 % y 100 % de la pasada, y no entran en ningun otro sitio.

    Parameters
    ----------
    x:
        Fold-train con sesgo (CSR float64), en el orden natural del fold.
    orden:
        Permutacion congelada de las filas del fold-train.
    fraccion:
        Fraccion de la pasada que se conserva.

    Returns
    -------
    scipy.sparse.csr_matrix
        Submatriz CSR con las filas del prefijo, en el orden congelado.
    """
    n_prefijo = max(math.floor(fraccion * orden.shape[0]), 1)
    sub = x[orden[:n_prefijo]].tocsr()
    sub.sort_indices()
    return sub


def entrenar_a1(x_tr: sp.csr_matrix, y_tr: np.ndarray, orden: np.ndarray, con_atribucion: bool
                ) -> dict:
    """Entrena A1 (FTRL softmax, UNA pasada) y devuelve pesos, curva online y atribucion.

    Registra la curva online de log-loss de train (media acumulada del stream) en los tres
    puntos de la pasada, que es la lectura directa de la firma de D-083, y los acumuladores de
    la razon ``|sigma*w|/|g|`` cuando se piden.

    Parameters
    ----------
    x_tr:
        Fold-train con sesgo (CSR float64), en el orden natural del fold.
    y_tr:
        Etiquetas del fold-train (reales, sin permutar).
    orden:
        Permutacion congelada de las filas del fold-train (unica, compartida con A2).
    con_atribucion:
        Si True acumula la razon de correccion por coordenada.

    Returns
    -------
    dict
        Pesos (d, 6), curva online en los tres puntos y, si se pidio, el detalle de atribucion.
    """
    alpha = alpha_derivado(orden.shape[0])
    n_total = orden.shape[0]
    marcas = {round(fraccion * n_total): fraccion for fraccion in PUNTOS_PASADA}
    curva: dict[str, float] = {}

    def registrar(ordinal: int, acumulada: float) -> None:
        """Guarda la media online de la perdida en los puntos congelados de la pasada."""
        if (ordinal + 1) in marcas:
            curva[str(marcas[ordinal + 1])] = float(acumulada / (ordinal + 1))

    w, estado = ftrl_softmax(x_tr, y_tr, alpha, LAMBDA1, LAMBDA2, BETA, orden,
                             con_atribucion=con_atribucion, al_terminar_ejemplo=registrar)
    return {"w": w, "alpha": alpha, "ll_train_online": curva,
            "atribucion": (razon_atribucion(estado, x_tr.shape[1], n_total)
                           if con_atribucion else None)}


def entrenar_a2(x_tr: sp.csr_matrix, y_tr: np.ndarray, orden: np.ndarray,
                fold: int) -> dict:
    """Entrena A2 (gemelo SGD de UNA pasada) y devuelve el modelo y su curva de train.

    Las filas se entregan EN EL ORDEN CONGELADO y con ``shuffle=False`` (enmenda 2
    pre-ejecucion del pre-registro), de modo que el control consume exactamente la misma
    permutacion que A1. El ``random_state=20261010+fold`` se conserva por fidelidad al
    protocolo, aunque con ``shuffle=False`` no altera el resultado (verificado: delta P = 0.0).

    Parameters
    ----------
    x_tr:
        Fold-train con sesgo (CSR float64), en el orden natural del fold.
    y_tr:
        Etiquetas del fold-train.
    orden:
        Permutacion congelada de las filas del fold-train.
    fold:
        Fold canonico; la semilla del control es ``20261010 + fold``.

    Returns
    -------
    dict
        Modelo ajustado, alpha efectivo (None: el control usa su alpha congelado) y tiempo.
    """
    inicio = time.time()
    filas = x_tr[orden].tocsr()
    filas.sort_indices()
    modelo = SGDClassifier(**kwargs_sgd(fold))
    modelo.fit(filas, y_tr[orden])
    return {"modelo": modelo, "alpha": None, "n_iter": int(np.max(modelo.n_iter_)),
            "tiempo_seg": float(time.time() - inicio)}


def prefijo_a1(x_tr: sp.csr_matrix, y_tr: np.ndarray, orden: np.ndarray, fraccion: float
               ) -> dict:
    """Mide el log_loss de train de A1 en un prefijo de la pasada (instrumentacion de G1/G2).

    Reajusta el MISMO algoritmo congelado (alpha derivado de su propio T) sobre el prefijo del
    orden congelado y devuelve su perdida de train sobre ese mismo prefijo. No es un modelo
    alternativo: no produce predicciones, no entra en el OOF y no altera ningun gate salvo
    G1/G2.

    Parameters
    ----------
    x_tr:
        Fold-train con sesgo (CSR float64), en el orden natural del fold.
    y_tr:
        Etiquetas del fold-train.
    orden:
        Permutacion congelada de las filas del fold-train.
    fraccion:
        Fraccion de la pasada (0.25, 0.5 o 1.0).

    Returns
    -------
    dict
        T del prefijo, alpha efectivo del prefijo y log_loss multiclase de train.
    """
    n_prefijo = max(math.floor(fraccion * orden.shape[0]), 1)
    sub = filas_prefijo(x_tr, orden, fraccion)
    alpha = alpha_derivado(n_prefijo)
    w, _ = ftrl_softmax(sub, y_tr[orden[:n_prefijo]], alpha, LAMBDA1, LAMBDA2, BETA,
                        np.arange(n_prefijo, dtype=np.int64))
    return {"n": int(n_prefijo), "alpha": alpha, "ll": log_loss_propio(softmax(sub.dot(w)),
                                                                        y_tr[orden[:n_prefijo]])}


def prefijo_a2(x_tr: sp.csr_matrix, y_tr: np.ndarray, orden: np.ndarray, fraccion: float,
               fold: int) -> dict:
    """Mide el log_loss de train de A2 en un prefijo de la pasada (instrumentacion de G1/G2).

    Ajusta el MISMO estimador congelado sobre el prefijo del orden congelado y devuelve su
    perdida de train sobre ese mismo prefijo. Es el gemelo exacto de ``prefijo_a1`` para que la
    puerta se aplique con la misma magnitud a los dos brazos.

    Parameters
    ----------
    x_tr:
        Fold-train con sesgo (CSR float64), en el orden natural del fold.
    y_tr:
        Etiquetas del fold-train.
    orden:
        Permutacion congelada de las filas del fold-train.
    fraccion:
        Fraccion de la pasada (0.25, 0.5 o 1.0).
    fold:
        Fold canonico (semilla del control).

    Returns
    -------
    dict
        T del prefijo, alpha efectivo (None: el control conserva 1e-6) y log_loss multiclase.
    """
    n_prefijo = max(math.floor(fraccion * orden.shape[0]), 1)
    sub = filas_prefijo(x_tr, orden, fraccion)
    modelo = SGDClassifier(**kwargs_sgd(fold))
    modelo.fit(sub, y_tr[orden[:n_prefijo]])
    p = modelo.predict_proba(sub)
    return {"n": int(n_prefijo), "alpha": float(SGD_ALPHA),
            "ll": log_loss_propio(p, y_tr[orden[:n_prefijo]])}


def registrar_fold(estado: dict, brazo: str, fold: int, resultado: dict, oof: np.ndarray,
                   hashes_fold: dict, tr_idx: np.ndarray, va_idx: np.ndarray,
                   clase_va: np.ndarray) -> dict:
    """Incorpora el fold al OOF y al checkpoint y persiste ambos (OOF primero, estado luego).

    La mascara de relleno se escribe una sola vez por fila (assert) y el fold-val jamas entra
    al ajuste.

    Parameters
    ----------
    estado:
        Checkpoint a actualizar.
    brazo:
        Nombre del brazo.
    fold:
        Fold entrenado.
    resultado:
        Salida de ``entrenar_fold``.
    oof:
        OOF del brazo (se modifica in situ).
    hashes_fold:
        Hashes del vectorizador del fold.
    tr_idx:
        Indices de fold-train.
    va_idx:
        Indices de fold-val.
    clase_va:
        Etiquetas reales del fold-val.

    Returns
    -------
    dict
        Detalle del fold tal como queda en el checkpoint.

    Raises
    ------
    Detencion
        Si la mascara detecta una fila escrita dos veces o un solape de folds (exit 2).
    """
    if np.intersect1d(tr_idx, va_idx).size != 0:
        raise Detencion(f"[STOP] train y val del fold {fold} se solapan (exit 2)", 2)
    p_va = resultado["p_va"]
    datos = estado["brazos"][brazo]
    mascara = np.asarray(datos["mascara_relleno_por_fila"], dtype=np.int64)
    if va_idx.size and int(mascara[va_idx].max()) != 0:
        raise Detencion(f"[STOP] {brazo} fold {fold}: fila ya rellenada en el OOF (exit 2)",
                        2)
    oof[va_idx] = p_va
    mascara[va_idx] += 1
    guardar_oof(brazo, oof)
    detalle = {
        "n_features": int(hashes_fold["n_features"]),
        "n_vocabulario": int(hashes_fold["n_vocabulario"]),
        "vocab_sha256": hashes_fold["vocab_sha256"],
        "idf_sha256": hashes_fold["idf_sha256"],
        "n_filas_fit": int(hashes_fold["n_filas_fit"]),
        "T": int(resultado["T"]),
        "alpha_efectivo": resultado["alpha"],
        "tiempo_seg": float(resultado["tiempo_seg"]),
        "ll_train_prefijo": {str(clave): float(valor)
                             for clave, valor in resultado["ll_train_prefijo"].items()},
        "ll_train_online": {str(clave): float(valor)
                            for clave, valor in resultado["ll_train_online"].items()},
        "macro_f1_standalone_fold_val": float(resultado["macro_f1_standalone_fold_val"]),
        "frecuencias_fold_val": [float(valor) for valor in resultado["frecuencias_fold_val"]],
        "max_abs_suma_menos_uno": float(resultado["max_abs_suma_menos_uno"]),
        "n_train_fold": int(tr_idx.size),
        "n_val_fold": int(va_idx.size),
        "atribucion": resultado["atribucion"],
    }
    datos["folds"][str(fold)] = detalle
    datos["folds_completados"] = sorted(datos["folds"])
    datos["mascara_relleno_por_fila"] = mascara.tolist()
    guardar_estado(estado)
    print(f"  [{brazo}] fold {fold}: {detalle['n_features']} coordenadas, vocab "
          f"{detalle['n_vocabulario']}, T {detalle['T']}, alpha {detalle['alpha_efectivo']}, "
          f"ll_train 25/50/100 "
          f"{detalle['ll_train_prefijo']['0.25']:.4f}/"
          f"{detalle['ll_train_prefijo']['0.5']:.4f}/"
          f"{detalle['ll_train_prefijo']['1.0']:.4f}, macro-F1 standalone "
          f"{detalle['macro_f1_standalone_fold_val']:.4f}, {detalle['tiempo_seg']:.1f} s",
          flush=True)
    return detalle


def entrenar_fold(brazo: str, x_tr: sp.csr_matrix, x_va: sp.csr_matrix, y_tr: np.ndarray,
                  clase_va: np.ndarray, orden: np.ndarray, fold: int) -> dict:
    """Entrena UN fold de UN brazo con el presupuesto congelado de una sola pasada.

    El unico recorrido de A1 es el orden congelado; el unico ajuste de A2 es el mismo
    presupuesto de una pasada sobre las mismas filas en ese orden. La magnitud ``ll_train`` se
    mide en los tres puntos de la pasada con el par de prefijos (misma magnitud para los dos
    brazos) y, en A1, tambien con la media online del stream.

    Parameters
    ----------
    brazo:
        Nombre del brazo.
    x_tr:
        Fold-train con sesgo (CSR float64), en el orden natural del fold.
    x_va:
        Fold-val con sesgo (CSR float64), en el orden natural del fold.
    y_tr:
        Etiquetas del fold-train.
    clase_va:
        Etiquetas reales del fold-val (solo reporte y puerta G3/G4).
    orden:
        Permutacion congelada de las filas del fold-train.
    fold:
        Fold canonico.

    Returns
    -------
    dict
        Probabilidades del fold-val y de fold-train, magnitudes de la pasada, alpha, tiempo y
        (en A1) la sonda de atribucion.
    """
    inicio = time.time()
    if brazo == "A1_ftrl_softmax":
        ajuste = entrenar_a1(x_tr, y_tr, orden, con_atribucion=True)
        w = ajuste["w"]
        p_va = softmax(x_va.dot(w))
        p_tr = softmax(x_tr[orden].tocsr().dot(w))
        ll_prefijo = {str(fraccion): prefijo_a1(x_tr, y_tr, orden, fraccion)["ll"]
                      for fraccion in PUNTOS_PASADA}
        alpha = ajuste["alpha"]
        atribucion = ajuste["atribucion"]
        ll_online = ajuste["ll_train_online"]
    else:
        ajuste = entrenar_a2(x_tr, y_tr, orden, fold)
        modelo = ajuste["modelo"]
        p_va = modelo.predict_proba(x_va)
        p_tr = modelo.predict_proba(x_tr[orden].tocsr())
        ll_prefijo = {str(fraccion): prefijo_a2(x_tr, y_tr, orden, fraccion, fold)["ll"]
                      for fraccion in PUNTOS_PASADA}
        alpha = float(SGD_ALPHA)
        atribucion = None
        ll_online = {}
    pred_va = p_va.argmax(axis=1)
    return {"p_va": p_va, "p_tr": p_tr, "T": int(orden.shape[0]), "alpha": alpha,
            "ll_train_prefijo": ll_prefijo, "ll_train_online": ll_online,
            "macro_f1_standalone_fold_val": macro(clase_va, pred_va),
            "frecuencias_fold_val": frecuencias(pred_va),
            "max_abs_suma_menos_uno": max(max_abs_suma_menos_uno(p_va),
                                          max_abs_suma_menos_uno(p_tr)),
            "atribucion": atribucion, "tiempo_seg": float(time.time() - inicio)}


def rss_gb() -> float:
    """RSS de proceso en GB (``ru_maxrss`` viene en KiB en Linux).

    Returns
    -------
    float
        Pico de memoria residente en GB.
    """
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0 / 1024.0


def verificar_ram(estado: dict) -> float:
    """Registra el pico de RSS y detiene si supera el tope congelado de 20 GB (exit 8).

    Parameters
    ----------
    estado:
        Checkpoint.

    Returns
    -------
    float
        RSS actual en GB.

    Raises
    ------
    Detencion
        Si el RSS supera ``RAM_TOPE_GB`` (codigo 8, estado guardado; no se anade
        ``max_features`` sobre la marcha).
    """
    actual = rss_gb()
    estado["rss_pico_gb"] = float(max(float(estado.get("rss_pico_gb", 0.0)), actual))
    if actual > RAM_TOPE_GB:
        guardar_estado(estado)
        raise Detencion(f"[STOP] RSS {actual:.2f} GB supera el tope de {RAM_TOPE_GB:g} GB; estado "
                        "guardado y no se anade max_features (exit 8)", 8)
    return actual


def verificar_deadline(estado: dict) -> None:
    """Comprueba el timeout global de 24 h antes de cada ajuste (exit 7 con estado).

    Parameters
    ----------
    estado:
        Checkpoint.

    Raises
    ------
    Detencion
        Si se superaron las 24 h (codigo 7, estado guardado).
    """
    transcurrido = time.time() - float(estado["inicio_ts"])
    if transcurrido >= TIMEOUT_H * 3600.0:
        guardar_estado(estado)
        raise Detencion(f"[STOP] timeout global de {TIMEOUT_H:g} h superado "
                        f"({transcurrido / 3600.0:.2f} h); estado guardado (exit 7)", 7)


def asegurar_calibracion(estado: dict) -> None:
    """Aplica la calibracion congelada tras el fold 0 (exit 6 si proyecta > 20 h).

    La proyeccion es ``9 x t_fold0`` con ``t_fold0`` la suma del tiempo de entrenamiento de los
    dos brazos en el fold 0 (la lectura mas conservadora de "t_fold0").

    Parameters
    ----------
    estado:
        Checkpoint.

    Raises
    ------
    Detencion
        Si la proyeccion 9 x t_fold0 supera las 20 h (codigo 6, estado guardado).
    """
    if estado["calibracion"] is not None:
        if not estado["calibracion"].get("ok", True):
            raise Detencion(f"[STOP] calibracion pre-registrada excedida: proyeccion "
                            f"{estado['calibracion']['proyeccion_horas']:.2f} h > "
                            f"{PROYECCION_MAX_H} h; estado guardado (exit 6)", 6)
        return
    if "0" not in estado["brazos"]["A1_ftrl_softmax"]["folds"]:
        return
    tiempos = {brazo: float(estado["brazos"][brazo]["folds"]["0"]["tiempo_seg"])
               for brazo in BRAZOS if "0" in estado["brazos"][brazo]["folds"]}
    tiempo_fold0 = float(sum(tiempos.values()))
    proyeccion_seg = PROYECCION_FACTOR * tiempo_fold0
    estado["calibracion"] = {"t_fold0_seg": tiempo_fold0, "tiempos_por_brazo_seg": tiempos,
                             "factor": PROYECCION_FACTOR, "proyeccion_seg": proyeccion_seg,
                             "proyeccion_horas": proyeccion_seg / 3600.0,
                             "limite_horas": PROYECCION_MAX_H,
                             "nota": "t_fold0 = suma de los dos brazos del fold 0 (lectura "
                                     "conservadora del pre-registro)",
                             "ok": bool(proyeccion_seg <= PROYECCION_MAX_H * 3600.0)}
    guardar_estado(estado)
    if not estado["calibracion"]["ok"]:
        raise Detencion(f"[STOP] calibracion: {PROYECCION_FACTOR:g} x t_fold0 = "
                        f"{estado['calibracion']['proyeccion_horas']:.2f} h > "
                        f"{PROYECCION_MAX_H} h; estado guardado (exit 6)", 6)


# =====================================================================================
# ---- 6. PUERTA DE CONVERGENCIA G1-G6 (FOLD 0, FAIL-FAST) ----
# =====================================================================================
def evaluar_puerta_preliminar(estado: dict, brazo: str) -> dict:
    """Evalua G1, G2, G3, G4 y G6 de un brazo con las magnitudes ya registradas del fold 0.

    G5 (determinismo) NO se evalua aqui: exige una segunda corrida identica del fold 0, que
    solo se lanza si estos cinco gates pasan (fail-fast: no se gasta CPU si ya se sabe que el
    veredicto es NO_ATRIBUIBLE_BY_GATE).

    Parameters
    ----------
    estado:
        Checkpoint con el fold 0 del brazo ya registrado.
    brazo:
        Nombre del brazo.

    Returns
    -------
    dict
        Valor de cada gate, su ``pass`` y el detalle numerico.

    Raises
    ------
    Detencion
        Si el fold 0 del brazo no esta registrado (codigo 2).
    """
    detalle = estado["brazos"][brazo]["folds"].get("0")
    if detalle is None:
        raise Detencion(f"[STOP] la puerta necesita el fold 0 de {brazo} y no existe (exit 2)", 2)
    prefijo = detalle["ll_train_prefijo"]
    puntos = [float(prefijo[str(fraccion)]) for fraccion in PUNTOS_PASADA]
    ll_100 = puntos[-1]
    decreciente = bool(puntos[0] >= puntos[1] - G2_TOLERANCIA
                       and puntos[1] >= puntos[2] - G2_TOLERANCIA)
    frec = [float(valor) for valor in detalle["frecuencias_fold_val"]]
    suma = float(detalle["max_abs_suma_menos_uno"])
    g1_cumple = bool(G1_BANDA[0] <= ll_100 <= G1_BANDA[1])
    g2_cumple = decreciente
    g3_cumple = bool(float(detalle["macro_f1_standalone_fold_val"]) >= G3_MINIMO)
    g4_cumple = bool(all(G4_RANGO[0] <= valor <= G4_RANGO[1] for valor in frec))
    g6_cumple = bool(suma <= G6_TOLERANCIA)
    return {
        "G1_ll_train_banda": {
            "criterio": f"{G1_BANDA[0]} <= ll_train(100%) <= {G1_BANDA[1]}",
            "valor": ll_100, "banda": list(G1_BANDA), "anclaje": ANCLA_LOG_UNIFORME,
            "pass": g1_cumple},
        "G2_monotonia": {
            "criterio": "ll_train no creciente en 25/50/100 % (prefijo del orden congelado)",
            "valores": puntos, "tolerancia": G2_TOLERANCIA, "ll_train_online":
            detalle["ll_train_online"], "pass": g2_cumple},
        "G3_macro_f1_fold_val": {
            "criterio": f"macro-F1 standalone en el fold-val >= {G3_MINIMO}",
            "valor": float(detalle["macro_f1_standalone_fold_val"]), "minimo": G3_MINIMO,
            "anclas": {"trivial": ANCLA_TRIVIAL, "gemelo_sgd_d083": ANCLA_GEMELO_SGD_D083},
            "pass": g3_cumple},
        "G4_frecuencia_por_clase": {
            "criterio": f"frecuencia predicha por clase en [{G4_RANGO[0]}, {G4_RANGO[1]}]",
            "valores": frec, "rango": list(G4_RANGO), "pass": g4_cumple},
        "G5_determinismo": {"criterio": "max|delta p| = 0.0 exacto entre dos corridas identicas",
                            "valor": None, "pass": None,
                            "estado": "no evaluado: exige la segunda corrida del fold 0"},
        "G6_suma_unidad": {
            "criterio": f"max|sum_c p_c - 1| <= {G6_TOLERANCIA}", "valor": suma,
            "tolerancia": G6_TOLERANCIA, "pass": g6_cumple},
    }


def evaluar_puerta(estado: dict, puerta_preliminar: dict, determinismos: dict) -> dict:
    """Cierra la puerta G1-G6 con G5 y decide el veredicto de la puerta.

    Parameters
    ----------
    estado:
        Checkpoint ya actualizado con las sondas de determinismo.
    puerta_preliminar:
        Gates G1-G4 y G6 ya evaluados por brazo.
    determinismos:
        Sondas de determinismo por brazo (segunda corrida del fold 0).

    Returns
    -------
    dict
        Gates por brazo, ``pasa`` global, brazos fallidos y el detalle del fold 0 consumido.
    """
    brazos = dict(puerta_preliminar)
    for brazo in BRAZOS:
        sonda = determinismos[brazo]
        valor = float(sonda["max_abs_diff"])
        brazos[brazo]["G5_determinismo"] = {
            "criterio": "max|delta p| = 0.0 exacto entre dos corridas identicas",
            "valor": valor, "tolerancia": G5_TOLERANCIA, "clasificacion":
            sonda["clasificacion"], "pass": bool(valor == G5_TOLERANCIA)}
    fallidos = {}
    for brazo in BRAZOS:
        gates = brazos[brazo]
        malos = [nombre for nombre, info in gates.items() if not info["pass"]]
        fallidos[brazo] = malos
    pasa = bool(all(not fallidos[brazo] for brazo in BRAZOS))
    return {"momento": PUERTA_PREREG["momento"],
            "fold_0_consumido_diagnostico": True,
            "nota_fold_0": PUERTA_PREREG["fold_0"],
            "magnitud": MAGNITUD_PUERTA,
            "gates": brazos, "brazos_fallidos": {brazo: fallidos[brazo] for brazo in BRAZOS},
            "pasa": pasa,
            "g5_evaluada": True,
            "consecuencia_si_falla": PUERTA_PREREG["consecuencia_si_falla"],
            "prohibido": PUERTA_PREREG["prohibido"]}


def sonda_determinismo(brazo: str, x_tr: sp.csr_matrix, x_va: sp.csr_matrix, y_tr: np.ndarray,
                       clase_va: np.ndarray, orden: np.ndarray, referencia: np.ndarray,
                       fold: int, estado: dict) -> dict:
    """Reentrena el fold 0 del brazo desde cero y mide max|delta p| contra la primera corrida.

    Es la MISMA corrida que exige el gate G5: se hace una sola vez y alimenta tanto G5 como la
    escalera de tolerancias del pre-registro (0 exacto; <=1e-9 conforme; >1e-9 con rango
    publicado; rango >= variabilidad o solape con el gate -> no concluyente).

    Parameters
    ----------
    brazo:
        Brazo a sondear.
    x_tr:
        Fold-train con sesgo (CSR float64), en el orden natural del fold.
    x_va:
        Fold-val con sesgo (CSR float64).
    y_tr:
        Etiquetas del fold-train.
    clase_va:
        Etiquetas reales del fold-val (solo reporte).
    orden:
        Permutacion congelada del fold-train del fold 0.
    referencia:
        Probabilidades de la primera corrida del fold 0 (fold-val).
    fold:
        Fold canonico (siempre 0 en esta sonda).
    estado:
        Checkpoint.

    Returns
    -------
    dict
        Delta de la repeticion, clasificacion, segunda repeticion si hace falta, rango,
        ``no_concluyente`` y solape con el gate.
    """
    verificar_deadline(estado)
    resultado = entrenar_fold(brazo, x_tr, x_va, y_tr, clase_va, orden, fold)
    delta_1 = float(np.abs(resultado["p_va"] - referencia).max())
    sonda = {"descripcion": f"reentrenamiento de {brazo} fold 0 desde cero con config identica",
             "max_abs_diff_repeticion_1": delta_1, "max_abs_diff": delta_1,
             "segunda_repeticion": None, "rango": None, "clasificacion": None,
             "no_concluyente": False, "solape_con_gate": False,
             "tolerancia_declarada": TOL_DETERMINISMO, "variabilidad_referencia": VARIABILIDAD,
             "gate_delta": GATE_DELTA, "referencia": DETERMINISMO_PREREG["esperado"]}
    if delta_1 == 0.0:
        sonda["clasificacion"] = "exacto"
    elif delta_1 <= TOL_DETERMINISMO:
        sonda["clasificacion"] = "tolerancia_declarada"
    else:
        verificar_deadline(estado)
        repetida = float(np.abs(entrenar_fold(brazo, x_tr, x_va, y_tr, clase_va, orden,
                                               fold)["p_va"] - referencia).max())
        rango = max(delta_1, repetida) - min(delta_1, repetida)
        sonda.update({"clasificacion": "rango_publicado", "segunda_repeticion": repetida,
                      "max_abs_diff": max(delta_1, repetida), "rango": rango,
                      "solape_con_gate": bool(rango >= GATE_DELTA),
                      "no_concluyente": bool(rango >= VARIABILIDAD or rango >= GATE_DELTA)})
    print(f"  sonda de determinismo {brazo} fold 0: max|delta p| = "
          f"{sonda['max_abs_diff']:.3e} ({sonda['clasificacion']})", flush=True)
    return sonda


# =====================================================================================
# ---- 7. ENTRENAMIENTO POR BRAZOS CON CHECKPOINTS, PUERTA Y REFIT ----
# =====================================================================================
def refit_vectorizador(textos: np.ndarray, fold_idx: np.ndarray,
                       estado: dict) -> dict:
    """Sonda de refit: reajusta el vectorizador en los 5 folds y compara los hashes bit a bit.

    No reentrena ningun modelo ni escribe ningun OOF: solo comprueba que el vocabulario ordenado
    y el idf de cada fold se reproducen identicos, que es lo que exige el pre-registro
    (exigido por kaggle-specialist).

    Parameters
    ----------
    textos:
        Textos del corpus en orden canonico.
    fold_idx:
        Fold canonico por fila.
    estado:
        Checkpoint.

    Returns
    -------
    dict
        Detalle por fold y ``reproducido`` global.
    """
    detalle: dict[str, dict] = {}
    todo_ok = True
    for fold in range(N_FOLDS):
        verificar_deadline(estado)
        verificar_ram(estado)
        tr_idx, va_idx = indices_fold(fold_idx, fold)
        ajuste = ajustar_vectorizador_fold(textos, tr_idx, va_idx)
        guardado = estado["vectorizador_por_fold"].get(str(fold), {})
        coincide = bool(ajuste["hashes"]["vocab_sha256"] == guardado.get("vocab_sha256")
                        and ajuste["hashes"]["idf_sha256"] == guardado.get("idf_sha256"))
        todo_ok = todo_ok and coincide
        detalle[str(fold)] = {"n_vocabulario": int(ajuste["hashes"]["n_vocabulario"]),
                              "vocab_sha256": ajuste["hashes"]["vocab_sha256"],
                              "idf_sha256": ajuste["hashes"]["idf_sha256"],
                              "vocab_sha256_corrida": guardado.get("vocab_sha256"),
                              "idf_sha256_corrida": guardado.get("idf_sha256"),
                              "reproducido": coincide}
        print(f"  refit fold {fold}: vocab e idf reproducidos bit a bit = {coincide}", flush=True)
        del ajuste
    return {"descripcion": REFIT_PREREG["descripcion"], "tolerancia": REFIT_PREREG["tolerancia"],
            "exigido_por": REFIT_PREREG["exigido_por"], "folds": detalle,
            "reproducido": todo_ok}


def entrenar_con_puerta(estado: dict, textos: np.ndarray, clase: np.ndarray,
                        fold_idx: np.ndarray) -> tuple[dict[str, np.ndarray], dict | None]:
    """Entrena los 2 brazos x 5 folds con la puerta G1-G6 fail-fast en el fold 0.

    Por fold se ajusta el vectorizador UNA vez (compartido por los dos brazos) y luego se
    recorren los brazos en orden A1, A2 con el MISMO orden de datos congelado. Al terminar el
    fold 0 se evalua la puerta completa (G1-G4 y G6 con las magnitudes ya registradas, y G5 con
    una segunda corrida identica del fold 0). Si falla, se devuelven los OOF parciales y la
    puerta fallida: el llamante cierra con NO_ATRIBUIBLE_BY_GATE sin integrar ni bootstrap.

    Parameters
    ----------
    estado:
        Checkpoint.
    textos:
        Textos del corpus en orden canonico.
    clase:
        Etiquetas canonicas.
    fold_idx:
        Fold canonico por fila.

    Returns
    -------
    tuple
        OOF por brazo y ``None`` si todo pasa, o el detalle de la puerta si falla.

    Raises
    ------
    Detencion
        Timeout global (7), calibracion excedida (6), RSS excedido (8) o invariantes (2).
    """
    oofs = {brazo: asegurar_brazo_cargado(estado, brazo, fold_idx) for brazo in BRAZOS}
    for fold in range(N_FOLDS):
        verificar_deadline(estado)
        tr_idx, va_idx = indices_fold(fold_idx, fold)
        ajuste = ajustar_vectorizador_fold(textos, tr_idx, va_idx)
        hashes_fold = dict(ajuste["hashes"])
        inicio_fit = time.time()
        hashes_fold["n_features"] = int(ajuste["x_tr"].shape[1])
        hashes_fold["tiempo_fit_seg"] = float(time.time() - inicio_fit)
        hashes_fold["n_train_fold"] = int(tr_idx.size)
        estado["vectorizador_por_fold"][str(fold)] = hashes_fold
        guardar_estado(estado)
        orden = orden_congelado(fold, tr_idx.size)
        y_tr = clase[tr_idx].astype(np.int64)
        clase_va = clase[va_idx].astype(np.int64)
        pendientes = [brazo for brazo in BRAZOS if str(fold) not in estado["brazos"][brazo]["folds"]]
        print(f"  [fold {fold}] TF-IDF ajustado solo en fold-train: {ajuste['x_tr'].shape[1]} "
              f"coordenadas, {ajuste['x_tr'].nnz} no nulos, T {tr_idx.size}, vocab "
              f"{hashes_fold['vocab_sha256'][:12]}; orden congelado sha256 "
              f"{sha256_f64(orden.astype(np.float64))[:12]}", flush=True)
        for brazo in pendientes:
            verificar_deadline(estado)
            resultado = entrenar_fold(brazo, ajuste["x_tr"], ajuste["x_va"], y_tr, clase_va,
                                      orden, fold)
            registrar_fold(estado, brazo, fold, resultado, oofs[brazo], hashes_fold, tr_idx,
                           va_idx, clase_va)
            verificar_ram(estado)
        del ajuste
        if fold != 0:
            continue
        # ---- puerta de convergencia G1-G6 (dentro del fold 0, antes de los folds 1-4) ----
        if tr_idx.size != ANCLA_N_TRAIN_FOLD:
            raise Detencion(f"[STOP] T del fold 0 = {tr_idx.size}, distinto del ancla congelada "
                            f"{ANCLA_N_TRAIN_FOLD} (exit 2)", 2)
        alpha_fold0 = alpha_derivado(tr_idx.size)
        if alpha_fold0 != ANCLA_ALPHA_ESPERADO:
            raise Detencion(f"[STOP] alpha efectivo del fold 0 {alpha_fold0!r} no reproduce el "
                            f"ancla {ANCLA_ALPHA_ESPERADO!r} (exit 2)", 2)
        preliminar = {brazo: evaluar_puerta_preliminar(estado, brazo) for brazo in BRAZOS}
        if not all(all(info["pass"] for nombre, info in brazo.items()
                       if nombre != "G5_determinismo") for brazo in preliminar.values()):
            for brazo in BRAZOS:
                malos = [nombre for nombre, info in preliminar[brazo].items()
                         if nombre != "G5_determinismo" and not info["pass"]]
                for nombre in malos:
                    print(f"  [GATE] {brazo} falla {nombre}: {preliminar[brazo][nombre]}",
                          flush=True)
            puerta = {"momento": PUERTA_PREREG["momento"],
                      "fold_0_consumido_diagnostico": True,
                      "nota_fold_0": PUERTA_PREREG["fold_0"],
                      "magnitud": MAGNITUD_PUERTA,
                      "gates": preliminar,
                      "brazos_fallidos": {brazo: [nombre for nombre, info
                                                 in preliminar[brazo].items()
                                                 if nombre != "G5_determinismo"
                                                 and not info["pass"]]
                                          for brazo in BRAZOS},
                      "pasa": False,
                      "g5_evaluada": False,
                      "consecuencia_si_falla": PUERTA_PREREG["consecuencia_si_falla"],
                      "prohibido": PUERTA_PREREG["prohibido"]}
            estado["puerta"] = puerta
            guardar_estado(estado)
            return oofs, puerta
        # ---- G5: segunda corrida identica del fold 0 (misma corrida que el determinismo) ----
        ajuste = ajustar_vectorizador_fold(textos, tr_idx, va_idx)
        sondas = {}
        for brazo in BRAZOS:
            referencia = oofs[brazo][va_idx]
            sondas[brazo] = sonda_determinismo(brazo, ajuste["x_tr"], ajuste["x_va"], y_tr,
                                              clase_va, orden, referencia, fold, estado)
            estado["brazos"][brazo]["sonda_determinismo"] = sondas[brazo]
            guardar_estado(estado)
        del ajuste
        puerta = evaluar_puerta(estado, preliminar, sondas)
        estado["puerta"] = puerta
        guardar_estado(estado)
        for brazo in BRAZOS:
            for nombre, info in puerta["gates"][brazo].items():
                if not info["pass"]:
                    print(f"  [GATE] {brazo} falla {nombre}: {info}", flush=True)
        if not puerta["pasa"]:
            return oofs, puerta
        print("  [GATE] puerta G1-G6 superada en A1 y A2 (fold 0 consumido-diagnostico); "
              "se continua con los folds 1-4", flush=True)
        asegurar_calibracion(estado)
    for brazo in BRAZOS:
        datos = estado["brazos"][brazo]
        if datos["folds_completados"] != [str(fold) for fold in range(N_FOLDS)]:
            raise Detencion(f"[STOP] el brazo {brazo} quedo incompleto (exit 2)", 2)
        mascara = np.asarray(datos["mascara_relleno_por_fila"], dtype=np.int64)
        if not bool((mascara == 1).all()):
            raise Detencion(f"[STOP] la mascara de relleno del OOF de {brazo} no es 1 por fila "
                            "(exit 2)", 2)
        if not np.isfinite(oofs[brazo]).all():
            raise Detencion(f"[STOP] el OOF de {brazo} tiene no finitos (exit 2)", 2)
        print(f"  [{brazo}] OOF completo: {oofs[brazo].shape} float64, mascara 1 por fila",
              flush=True)
    return oofs, None


# =====================================================================================
# ---- 8. EVALUACION: INTEGRACION P6/P7, COMPLEMENTARIEDAD Y BOOTSTRAP ----
# =====================================================================================
def bootstrap_delta(clase: np.ndarray, pred_base: np.ndarray, pred_cand: np.ndarray) -> dict:
    """Bootstrap pareado (N_BOOTSTRAP, semilla congelada) de F1(cand) - F1(base).

    Parameters
    ----------
    clase:
        Etiquetas canonicas.
    pred_base:
        Prediccion de referencia.
    pred_cand:
        Prediccion del candidato.

    Returns
    -------
    dict
        Delta medio, IC95 por percentiles, ``p_delta_positivo`` e ``ic95_sin_cero``.
    """
    generador = np.random.default_rng(SEMILLA)
    codigo_base = clase * N_CLASES + pred_base
    codigo_cand = clase * N_CLASES + pred_cand
    n = len(clase)
    deltas = np.empty(N_BOOTSTRAP, dtype=np.float64)
    for b in range(N_BOOTSTRAP):
        idx = generador.integers(0, n, n)
        deltas[b] = (macro_de_conteos(np.bincount(codigo_cand[idx], minlength=N_CLASES ** 2))
                     - macro_de_conteos(np.bincount(codigo_base[idx], minlength=N_CLASES ** 2)))
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return {"delta_medio": float(deltas.mean()), "ic95": [float(lo), float(hi)],
            "p_delta_positivo": float((deltas > 0).mean()),
            "ic95_sin_cero": bool(lo > 0.0 or hi < 0.0)}


def combinador_oof(v30: np.ndarray, p_brazo: np.ndarray, clase: np.ndarray,
                   fold_idx: np.ndarray) -> tuple[np.ndarray, dict]:
    """Combinador logistico anidado congelado sobre [L2(v30) || L2(p_brazo)] (12 features).

    Parameters
    ----------
    v30:
        Probabilidades del campeon (169336, 6) ya normalizadas L1.
    p_brazo:
        OOF del brazo (169336, 6), probabilidades que suman 1.
    clase:
        Etiquetas canonicas.
    fold_idx:
        Fold canonico por fila.

    Returns
    -------
    tuple[numpy.ndarray, dict]
        ``p_comb`` (169336, 6) normalizada y el detalle del combinador.

    Raises
    ------
    AssertionError
        Si el combinador de algun fold no ve las 6 clases en orden canonico.
    """
    caracteristicas = np.hstack([normaliza_l2(v30), normaliza_l2(p_brazo)])
    p = np.zeros((len(clase), N_CLASES), dtype=np.float64)
    ratio_por_fold, iteraciones = [], []
    for fold in range(N_FOLDS):
        tr = fold_idx != fold
        va = fold_idx == fold
        modelo = LogisticRegression(C=C_LOGREG, max_iter=MAX_ITER_LOGREG, solver="lbfgs")
        modelo.fit(caracteristicas[tr], clase[tr])
        if list(modelo.classes_) != list(range(N_CLASES)):
            raise Detencion("[STOP] el combinador no vio las 6 clases en orden canonico "
                            "(exit 2)", 2)
        p[va] = modelo.predict_proba(caracteristicas[va])
        coeficientes = np.abs(modelo.coef_)
        ratio_por_fold.append(float(coeficientes[:, N_CLASES:].mean()
                                    / max(coeficientes[:, :N_CLASES].mean(), L2_CLIP)))
        iteraciones.append(int(np.max(modelo.n_iter_)))
    return normaliza(p), {"ratio_bloque_brazo_por_fold": ratio_por_fold,
                          "n_iter_por_fold": iteraciones}


def evaluar_combinacion(p_comb: np.ndarray, clase: np.ndarray, fold_idx: np.ndarray,
                        champ_pred: np.ndarray, pesos: np.ndarray) -> dict:
    """Metricas del combinador con la regla congelada (argmax p_comb * pesos) y diagnosticos.

    Returns
    -------
    dict
        macro-F1 global, deltas vs baseline y nested, folds y deltas por fold, folds
        positivos, IC95 del bootstrap, ``argmax_sin_regla``, filas que cambian y la
        prediccion combinada (no serializable: el llamador la retira antes del JSON).
    """
    pred = np.argmax(p_comb * pesos, axis=1)
    cv = macro(clase, pred)
    folds = [macro(clase[fold_idx == k], pred[fold_idx == k]) for k in range(N_FOLDS)]
    folds_base = [macro(clase[fold_idx == k], champ_pred[fold_idx == k])
                  for k in range(N_FOLDS)]
    deltas_por_fold = [float(a - b) for a, b in zip(folds, folds_base)]
    intervalo = bootstrap_delta(clase, champ_pred, pred)
    sin_regla = macro(clase, np.argmax(p_comb, axis=1))
    return {"corrected_cv": cv, "delta_vs_baseline": cv - BASELINE,
            "delta_vs_nested": cv - NESTED, "folds": folds, "folds_base": folds_base,
            "deltas_por_fold": deltas_por_fold,
            "folds_positivos": int(sum(1 for delta in deltas_por_fold if delta > 0)),
            "filas_que_cambian": int((pred != champ_pred).sum()),
            "argmax_sin_regla": sin_regla,
            "delta_sin_regla_vs_baseline": sin_regla - BASELINE,
            "delta_sin_regla_vs_nested": sin_regla - NESTED,
            "log_loss_p_comb": log_loss_multiclase(p_comb, clase),
            "delta_medio_bootstrap": intervalo["delta_medio"], "ic95": intervalo["ic95"],
            "p_delta_positivo": intervalo["p_delta_positivo"],
            "ic95_sin_cero": intervalo["ic95_sin_cero"], "pred_comb": pred}


def diagnostico_brazo(p_brazo: np.ndarray, clase: np.ndarray, fold_idx: np.ndarray,
                      champ_pred: np.ndarray, conf: np.ndarray) -> dict:
    """Standalone del brazo (sin pesos), matriz esperada 6x6, log_loss y rescate/dano.

    La normalizacion L2 por fila escala cada fila por una constante POSITIVA, de modo que el
    argmax (y por tanto el macro-F1 standalone) es identico al del OOF crudo.

    Parameters
    ----------
    p_brazo:
        OOF del brazo (169336, 6), probabilidades que suman 1.
    clase:
        Etiquetas canonicas.
    fold_idx:
        Fold canonico por fila.
    champ_pred:
        Decision del campeon (argmax v30 * pesos).
    conf:
        Confianza del campeon (max v30), para el subconjunto de alta confianza.

    Returns
    -------
    dict
        macro-F1 standalone global y por fold, F1 por clase, matriz ``E[p[:, j] | y = i]``
        con su diagonal, los dos log_loss y rescate/dano/neto globales y por clase.
    """
    pred = np.argmax(p_brazo, axis=1)
    f1_por_clase = f1_score(clase, pred, average=None, labels=list(range(N_CLASES)))
    ok_cand = pred == clase
    ok_campeon = champ_pred == clase
    alta_confianza = (~ok_campeon) & (conf >= UMBRAL_ALTA_CONFIANZA)
    por_clase = []
    for c in range(N_CLASES):
        mascara = clase == c
        por_clase.append({
            "clase": cfg.CLASSES[c], "filas": int(mascara.sum()),
            "f1": float(f1_por_clase[c]),
            "rescate": int((~ok_campeon & ok_cand & mascara).sum()),
            "danio": int((ok_campeon & ~ok_cand & mascara).sum()),
            "rescates_alta_confianza": int((alta_confianza & ok_cand & mascara).sum())})
    esperada = np.stack([p_brazo[clase == i].mean(axis=0) for i in range(N_CLASES)])
    diagonal_ok = bool(all(int(np.argmax(esperada[i])) == i for i in range(N_CLASES)))
    rescate = int((~ok_campeon & ok_cand).sum())
    danio = int((ok_campeon & ~ok_cand).sum())
    return {"standalone_cv": macro(clase, pred),
            "standalone_folds": [macro(clase[fold_idx == k], pred[fold_idx == k])
                                 for k in range(N_FOLDS)],
            "log_loss_prob": log_loss_propio(normaliza(p_brazo), clase),
            "log_loss_l2": log_loss_multiclase(p_brazo, clase),
            "f1_por_clase": [float(valor) for valor in f1_por_clase],
            "matriz_esperada": esperada.tolist(), "diagonal_ok": diagonal_ok,
            "max_abs_suma_menos_uno": max_abs_suma_menos_uno(p_brazo),
            "vs_campeon": {"rescate": rescate, "danio": danio, "neto": rescate - danio,
                           "rescates_alta_confianza": int((alta_confianza & ok_cand).sum()),
                           "errores_alta_confianza_campeon": int(alta_confianza.sum()),
                           "por_clase": por_clase}}


def verificar_invariantes(estado: dict, clase: np.ndarray, fold_idx: np.ndarray,
                          oofs: dict[str, np.ndarray], refit: dict) -> dict:
    """Evidencias de no-leakage derivadas de los artefactos (no declaraciones a mano).

    Returns
    -------
    dict
        Chequeos booleanos: alineacion, folds canonicos, folds completos, mascara 1 por fila,
        OOF finito y con filas que suman 1, vocabulario e idf registrados, vectorizador
        ajustado solo en fold-train, coherencia de dimensiones y refit reproducible.
    """
    completos, mascaras, finitos, suman_uno, hashes_ok, filas_fit, dimensiones, sumas = (
        [], [], [], [], [], [], [], [])
    for brazo in BRAZOS:
        datos = estado["brazos"][brazo]
        completos.append(datos["folds_completados"] == [str(fold) for fold in range(N_FOLDS)])
        mascara = np.asarray(datos["mascara_relleno_por_fila"], dtype=np.int64)
        mascaras.append(bool((mascara == 1).all()))
        finitos.append(bool(np.isfinite(oofs[brazo]).all()))
        suman_uno.append(bool(max_abs_suma_menos_uno(oofs[brazo]) <= G6_TOLERANCIA))
        for fold in range(N_FOLDS):
            detalle = datos["folds"][str(fold)]
            hashes_ok.append(len(detalle["vocab_sha256"]) == 64
                             and len(detalle["idf_sha256"]) == 64)
            filas_fit.append(detalle["n_filas_fit"] == detalle["n_train_fold"])
            dimensiones.append(detalle["n_features"] == detalle["n_vocabulario"] + 1)
            sumas.append(detalle["max_abs_suma_menos_uno"] <= G6_TOLERANCIA)
    return {
        "alineacion_filas": bool(len(clase) == N_FILAS),
        "cinco_folds_canonicos": bool(np.unique(fold_idx).size == N_FOLDS),
        "folds_completos_por_brazo": bool(all(completos)),
        "oof_mascara_relleno_uno_por_fila": bool(all(mascaras)),
        "oof_sin_no_finitos": bool(all(finitos)),
        "oof_filas_suman_uno": bool(all(suman_uno)),
        "vocabulario_e_idf_registrados_por_fold": bool(all(hashes_ok)),
        "vectorizador_ajustado_solo_en_fold_train": bool(all(filas_fit)),
        "n_features_coherente_vocab_mas_sesgo": bool(all(dimensiones)),
        "suma_unidad_por_fold": bool(all(sumas)),
        "refit_reproduce_vocabulario_e_idf_en_5_folds": bool(refit["reproducido"]),
    }


# =====================================================================================
# ---- 9. GATES, CAPS Y VEREDICTO ----
# =====================================================================================
def calcular_gates(baseline_ok: bool, integracion: dict, invariantes: dict,
                   determinismos: dict) -> tuple[dict, dict, str]:
    """Aplica los gates congelados, los caps y el arbol de decision.

    Gates: baseline recomputado, delta de A1 >= +0.0015, folds positivos >= 4/5, control
    favorable A1 sobre A2 (delta puntual), IC95 del delta de A1 sin cero, sin leakage,
    determinismo concluyente y sonda de atribucion publicada. Caps: si la sonda de atribucion no
    se publica el veredicto queda capado a MARGINAL, y si el determinismo no es concluyente el
    veredicto no puede ser PROMISING.

    Parameters
    ----------
    baseline_ok:
        Resultado del chequeo del baseline.
    integracion:
        Metricas del combinador por brazo.
    invariantes:
        Evidencias de no leakage.
    determinismos:
        Sondas de determinismo por brazo.

    Returns
    -------
    tuple[dict, dict, str]
        Gates, caps y veredicto del arbol.
    """
    a1 = integracion["A1_ftrl_softmax"]
    a2 = integracion["A2_sgd_1p"]
    sonda_publicada = bool(determinismos["atribucion_publicada"])
    no_concluyente = bool(determinismos["no_concluyente"])
    gates = {
        "baseline_ok": bool(baseline_ok),
        "delta_A1_suficiente": bool(a1["delta_vs_baseline"] >= GATE_DELTA),
        "folds_positivos_A1_suficiente": bool(a1["folds_positivos"] >= GATE_FOLDS),
        "control_favorable_A1_sobre_A2": bool(a1["delta_vs_baseline"]
                                              > a2["delta_vs_baseline"]),
        "ic95_A1_sin_cero": bool(a1["ic95_sin_cero"]),
        "sin_leakage": bool(all(invariantes.values())),
        "determinismo_concluyente": bool(not no_concluyente),
        "sonda_atribucion_publicada": sonda_publicada,
    }
    caps = {"sonda_atribucion_no_publicada": bool(not sonda_publicada),
            "criterio_sonda": GATES_PREREG["caps"][0],
            "determinismo_no_concluyente": bool(no_concluyente),
            "criterio_determinismo": GATES_PREREG["caps"][1]}
    if all(gates.values()) and not caps["sonda_atribucion_no_publicada"] \
            and not caps["determinismo_no_concluyente"]:
        veredicto = "PROMISING"
    elif a1["delta_vs_baseline"] > 0:
        veredicto = "MARGINAL"
    else:
        veredicto = "CLOSED"
    return gates, caps, veredicto


# =====================================================================================
# ---- 10. SALIDAS (JSON, informe, figura, novedad, historial, RUNLOG, CSV) ----
# =====================================================================================
def figura_curvas(estado: dict, puerta: dict, integracion: dict | None, ruta: Path) -> None:
    """Figura 1x3: log_loss de train en la pasada, razon de atribucion y deltas por fold.

    Panel 1: log_loss multiclase de train en el 25/50/100 % de la pasada, por fold y por brazo,
    con la banda G1 sombreada. Panel 2: razon media |sigma*w|/|g| por clase de A1 en el fold 0
    frente a la linea 1 (correccion igual al gradiente). Panel 3: delta por fold del macro-F1
    combinado frente al campeon, con la linea del gate.

    Parameters
    ----------
    estado:
        Checkpoint con las magnitudes por fold de los dos brazos.
    puerta:
        Detalle de la puerta G1-G6.
    integracion:
        Metricas del combinador por brazo, o None si la puerta cerro el ciclo.
    ruta:
        Destino del PNG.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figura, ejes = plt.subplots(1, 3, figsize=(19.0, 5.0))
    fracciones = [str(fraccion) for fraccion in PUNTOS_PASADA]
    for brazo in BRAZOS:
        for fold in range(N_FOLDS):
            detalle = estado["brazos"][brazo]["folds"].get(str(fold))
            if detalle is None:
                continue
            color = plt.cm.viridis((fold + (0.35 if brazo == "A2_sgd_1p" else 0.0))
                                   / max(N_FOLDS, 1))
            eje = 0 if brazo == "A1_ftrl_softmax" else 1
            valores = [detalle["ll_train_prefijo"][clave] for clave in fracciones]
            ejes[eje].plot([float(fraccion) * 100 for fraccion in PUNTOS_PASADA], valores,
                           marker="o", color=color, linewidth=1.2,
                           label=f"{brazo} fold {fold}")
    for indice, brazo in enumerate(BRAZOS):
        eje = 0 if brazo == "A1_ftrl_softmax" else 1
        ejes[eje].axhspan(G1_BANDA[0], G1_BANDA[1], color="g", alpha=0.12,
                         label="banda G1" if indice == 0 else None)
    ejes[0].set_title("A1 FTRL softmax: log_loss de train en la pasada (banda G1)")
    ejes[1].set_title("A2 gemelo SGD: log_loss de train en la pasada (banda G1)")
    for eje in (ejes[0], ejes[1]):
        eje.set_xlabel("porcentaje de la pasada")
        eje.set_ylabel("log_loss multiclase de train")
        eje.grid(alpha=0.3)
        eje.legend(fontsize=5, ncol=2)
    atribucion = (estado["brazos"]["A1_ftrl_softmax"]["folds"].get("0", {}) or {}).get("atribucion")
    if atribucion:
        valores_clase = atribucion["razon_por_clase"]
        ejes[2].bar(np.arange(N_CLASES), valores_clase, color="tab:orange")
        ejes[2].axhline(1.0, color="k", linewidth=0.8, linestyle=":",
                        label="correccion = gradiente")
        ejes[2].axhline(0.0, color="k", linewidth=0.8)
        for indice, valor in enumerate(valores_clase):
            ejes[2].text(indice, valor, f"{valor:.3g}", ha="center", va="bottom", fontsize=7)
    ejes[2].set_xticks(np.arange(N_CLASES))
    ejes[2].set_xticklabels([nombre for nombre in cfg.CLASSES], rotation=30)
    ejes[2].set_title("Sonda de atribucion A1 (fold 0): razon media |sigma*w|/|g| por clase")
    ejes[2].set_ylabel("razon")
    ejes[2].grid(alpha=0.3, axis="y")
    ejes[2].legend(fontsize=7)
    figura.tight_layout()
    figura.savefig(ruta, dpi=140)
    plt.close(figura)
    print(f"  figura: {ruta_relativa(ruta)}", flush=True)


def escribir_informe(resultado: dict) -> None:
    """Escribe el informe Markdown del experimento con tablas, puerta, gates y asimetrias.

    Parameters
    ----------
    resultado:
        JSON de resultados ya construido.
    """
    puerta = resultado["puerta"]
    integracion = resultado.get("integracion") or {}
    linea = [
        "# V33-E1 - Ejecucion: FTRL-Proximal softmax multiclase de una pasada online",
        "",
        f"- Fecha de ejecucion: {resultado['fecha']}",
        (f"- Pre-registro congelado: `experiments/v33_00_preregistro.json` (commit "
         f"{resultado['prereg']['commit_congelado'][:7]}) y {ARCHIVO_PRE_REGISTRO}"),
        f"- Puerta de convergencia (normativa): {ARCHIVO_PUERTA}",
        f"- Commit de ejecucion: `{resultado['commit']}`",
        (f"- Baseline primario recomputado: "
         f"{resultado['verificacion']['baseline_recomputado']:.7f} (referencia nested "
         f"{NESTED:.7f})"),
        (f"- Ancla trivial recalculada: "
         f"{resultado['verificacion']['linea_base_mayoritaria']['macro_f1_linea_base']!r} "
         f"(congelada {ANCLA_TRIVIAL!r}, diferencia "
         f"{resultado['verificacion']['linea_base_mayoritaria']['diferencia']:.3e})"),
        (f"- {HILOS} hilos fijos; TF-IDF ajustada por fold; UNA pasada por fold y por brazo; "
         f"alpha efectivo derivado por fold; RSS pico "
         f"{resultado['rss_pico_gb']:.2f} GB (tope {RAM_TOPE_GB:g} GB)"),
        (f"- Compliance AST: {resultado['compliance']['literales_de_ruta_revisados']} literales "
         f"de ruta y {resultado['compliance']['modulos_revisados']} modulos revisados, "
         "0 violaciones"),
        "",
        "## Diagnostico que motiva el diseno (D-083)",
        "",
        ("- D-083 congelo FTRL-Proximal **OvR** con alpha 0.05, lambda2 1.0 y **10 pasadas**: "
         "el log_loss medio de train fue 11.721967 en la epoca 1 y 23.025869 en la 10, las "
         "probabilidades quedaron planas (~0.408 por clase tras normalizar L2) y el macro-F1 "
         "standalone fue 0.040169, POR DEBAJO de la linea base trivial 0.048542. La "
         "implementacion se habia verificado exacta (tres implementaciones coinciden a 6.7e-15 "
         "y el punto fijo analitico da 0.0), luego el fallo era de CONFIGURACION: con alpha "
         "0.05 el termino de correccion `sigma` queda inflado y diez pasadas acumulan `z` y `n` "
         "en un regimen para el que McMahan calibro alpha con T del orden 1e6-1e9."),
        ("- La hipotesis de que solo alpha tiene palanca queda registrada como HIPOTESIS del "
         "arquitecto, no como evidencia: por eso la sonda `|sigma*w|/|g|` es OBLIGATORIA y "
         "publica la razon por coordenada de la correccion frente al gradiente."),
        ("- En este diseno alpha pasa a DERIVADO (`3/(2*sqrt(T))`, con T = n_train_fold "
         "calculado en runtime, cero valores evaluados), la descomposicion pasa a softmax "
         "multiclase con un solo juego de pesos y las 10 pasadas pasan a UNA."),
        "",
        "## alpha efectivo por fold (DERIVADO en runtime, no buscado)",
        "",
        "| fold | T | alpha efectivo | ll_train 25% | ll_train 50% | ll_train 100% |",
        "|---|---|---|---|---|---|",
    ]
    for fold in range(N_FOLDS):
        detalle = resultado["entrenamiento"]["A1_ftrl_softmax"]["folds"].get(str(fold))
        if detalle is None:
            continue
        linea.append(f"| {fold} | {detalle['T']} | {detalle['alpha_efectivo']!r} | "
                     f"{detalle['ll_train_prefijo']['0.25']:.4f} | "
                     f"{detalle['ll_train_prefijo']['0.5']:.4f} | "
                     f"{detalle['ll_train_prefijo']['1.0']:.4f} |")
    linea += [
        "",
        ("Nota: `3/(2*sqrt(T))` con T = "
         f"{ANCLA_N_TRAIN_FOLD} da alpha = {ANCLA_ALPHA_ESPERADO!r}, exactamente el ancla "
         "congelada del pre-registro; el runner lo verifica en runtime y sale con codigo 2 si no "
         "reproduce."),
        "",
        "## Sondas obligatorias (a)-(f), ejecutadas antes de cualquier ajuste",
        "",
        "| Sonda | Valor | Tolerancia | pass |",
        "|---|---|---|---|",
        (f"| (a) vectorizado vs lazo | max delta = "
         f"{resultado['sondas']['a_vectorizado_vs_lazo']['max_abs_diff']:.3e} | "
         f"{resultado['sondas']['a_vectorizado_vs_lazo']['tolerancia']:.0e} | "
         f"{resultado['sondas']['a_vectorizado_vs_lazo']['pass']} |"),
        (f"| (b) raiz analitica lambda 0 | residuo raiz = "
         f"{resultado['sondas']['b_raiz_analitica_lambda0']['residuo_raiz']:.3e}, un paso = "
         f"{resultado['sondas']['b_raiz_analitica_lambda0']['residuo_un_paso']:.3e} | "
         f"{resultado['sondas']['b_raiz_analitica_lambda0']['umbral']:.0e} | "
         f"{resultado['sondas']['b_raiz_analitica_lambda0']['pass']} |"),
        (f"| (c) TotalF1 average=Macro | delta vs macro-F1 = "
         f"{resultado['sondas']['c_totalf1_macro']['delta_criterio']:.3e} | "
         f"{resultado['sondas']['c_totalf1_macro']['tolerancia']:.0e} | "
         f"{resultado['sondas']['c_totalf1_macro']['pass']} |"),
        (f"| (d) sin NaN y log_loss finito | log_loss multiclase = "
         f"{resultado['sondas']['d_sin_nan_logloss_finito']['log_loss_multiclase']:.4f} | "
         f"finito | {resultado['sondas']['d_sin_nan_logloss_finito']['pass']} |"),
        (f"| (e) log_loss propio vs sklearn | delta = "
         f"{resultado['sondas']['e_logloss_multiclase_propio']['delta']:.3e} | "
         f"{resultado['sondas']['e_logloss_multiclase_propio']['tolerancia']:.0e} | "
         f"{resultado['sondas']['e_logloss_multiclase_propio']['pass']} |"),
        (f"| (f) filas que suman 1 | peor max|sum-1| = "
         f"{resultado['sondas']['f_filas_suman_uno']['peor']:.3e} | "
         f"{resultado['sondas']['f_filas_suman_uno']['tolerancia']:.0e} | "
         f"{resultado['sondas']['f_filas_suman_uno']['pass']} |"),
        "",
        "## Puerta de convergencia G1-G6 (fail-fast en el fold 0, ANTES de integrar)",
        "",
        f"- Momento: {puerta['momento']}.",
        f"- Magnitud: {puerta['magnitud']}.",
        f"- {puerta['nota_fold_0']}.",
        f"- Resultado: pasa = {puerta['pasa']}.",
        "",
        "| Gate | Criterio | A1 valor | A1 pass | A2 valor | A2 pass |",
        "|---|---|---|---|---|---|",
    ]
    for nombre in ("G1_ll_train_banda", "G2_monotonia", "G3_macro_f1_fold_val",
                   "G4_frecuencia_por_clase", "G5_determinismo", "G6_suma_unidad"):
        a1 = puerta["gates"].get("A1_ftrl_softmax", {}).get(nombre, {})
        a2 = puerta["gates"].get("A2_sgd_1p", {}).get(nombre, {})
        linea.append(f"| {nombre} | {a1.get('criterio', '-')} | {a1.get('valor', '-')} | "
                     f"{a1.get('pass')} | {a2.get('valor', '-')} | {a2.get('pass')} |")
    linea += [
        "",
        ("- El fold 0 queda registrado como **consumido-diagnostico**: G3 y G4 leen su "
         "fold-val, que es parte de la validacion canonica, y su fallo **no toca delta**."),
        f"- Consecuencia si falla: {puerta['consecuencia_si_falla']}.",
        f"- Prohibido por el pre-registro: {puerta['prohibido']}.",
        "",
        "## Sonda de atribucion OBLIGATORIA |sigma*w| / |g| (fold 0)",
        "",
        (f"- Descripcion congelada: {SONDA_ATRIBUCION_PREREG['descripcion']}."),
        f"- Motivo: {SONDA_ATRIBUCION_PREREG['motivo']}.",
        f"- Consecuencia si no se publica: {SONDA_ATRIBUCION_PREREG['consecuencia_si_no_se_publica']}.",
        "",
    ]
    atribucion = resultado["determinismo"].get("atribucion_fold_0")
    if atribucion:
        linea += [
            (f"- Razon media por coordenada: {atribucion['razon_media_coordenadas']!r} (mediana "
             f"{atribucion['razon_mediana_coordenadas']!r}, p90 "
             f"{atribucion['razon_p90_coordenadas']!r}, max "
             f"{atribucion['razon_max_coordenadas']!r})."),
            (f"- Razon global: {atribucion['razon_global']!r} | coordenada del sesgo: "
             f"{atribucion['razon_coordenada_sesgo']!r} | fraccion de coordenadas con razon > 1: "
             f"{atribucion['fraccion_coordenadas_con_razon_supera_1']!r}."),
            (f"- Razon por clase: {atribucion['razon_por_clase']}."),
            (f"- Coordenadas tocadas: {atribucion['n_coordenadas_tocadas']} de "
             f"{atribucion['n_coordenadas']}."),
        ]
    else:
        linea.append("- Sonda de atribucion NO publicada: el veredicto queda capado a MARGINAL.")
    linea += [
        "",
        "## Entrenamiento por fold (el fold-val solo se uso para el OOF, G3 y G4)",
        "",
        ("| Brazo | fold | n_feat | vocab | T | alpha | ll_train 25/50/100 | media online | "
         "macro-F1 standalone | suma-1 | tiempo (s) |"),
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for brazo in BRAZOS:
        for fold in range(N_FOLDS):
            detalle = resultado["entrenamiento"][brazo]["folds"].get(str(fold))
            if detalle is None:
                continue
            online = detalle["ll_train_online"]
            cadena_online = "/".join(f"{online[clave]:.4f}" for clave in
                                     ("0.25", "0.5", "1.0") if clave in online) or "-"
            linea.append(
                f"| {brazo} | {fold} | {detalle['n_features']} | {detalle['n_vocabulario']} | "
                f"{detalle['T']} | {detalle['alpha_efectivo']!r} | "
                f"{detalle['ll_train_prefijo']['0.25']:.4f}/"
                f"{detalle['ll_train_prefijo']['0.5']:.4f}/"
                f"{detalle['ll_train_prefijo']['1.0']:.4f} | {cadena_online} | "
                f"{detalle['macro_f1_standalone_fold_val']:.4f} | "
                f"{detalle['max_abs_suma_menos_uno']:.3e} | {detalle['tiempo_seg']:.1f} |")
    linea += [
        "",
        ("- La magnitud `ll_train` es la perdida de train del modelo ajustado **a ese prefijo** "
         "del orden congelado: es la misma en A1 y en A2, de modo que la puerta se aplica con un "
         "criterio unico. El modelo que produce el OOF es SIEMPRE el de la pasada completa; las "
         "corridas de prefijo son instrumentacion, no modelos."),
        ("- `media online` solo existe en A1 (es el running mean del stream, la lectura directa "
         "de la firma 11.72 -> 23.03 de D-083). A2 no puede exponerla porque "
         "`SGDClassifier` no ofrece un callback por ejemplo: se reporta como '-' y su "
         "asimetria queda declarada."),
        "",
        "## Vocabulario e idf por fold (sha256, primeros 16) y sonda de refit",
        "",
        "| fold | n_vocabulario | vocab_sha256 | idf_sha256 | refit reproduce |",
        "|---|---|---|---|---|",
    ]
    refit = resultado["refit"] or {}
    for fold in range(N_FOLDS):
        detalle = resultado["vocabulario_idf_por_fold"].get(str(fold))
        if detalle is None:
            continue
        coincide = (refit.get("folds", {}).get(str(fold), {}) or {}).get("reproducido")
        linea.append(f"| {fold} | {detalle['n_vocabulario']} | `{detalle['vocab_sha256'][:16]}`"
                     f" | `{detalle['idf_sha256'][:16]}` | {coincide} |")
    linea += [
        "",
        (f"- Sonda de refit: {REFIT_PREREG['descripcion']}; tolerancia: "
         f"{REFIT_PREREG['tolerancia']}; reproducido en los 5 folds = "
         f"{refit.get('reproducido')} (exigido por {REFIT_PREREG['exigido_por']})."),
        "",
        "## Sondas de determinismo (reentreno del fold 0 desde cero, config identica)",
        "",
    ]
    for brazo in BRAZOS:
        sonda = resultado["determinismo"].get(f"sonda_{brazo}") or {}
        if not sonda:
            linea.append(f"- {brazo}: no medido (el cierre por puerta precede a las sondas "
                         f"de determinismo).")
            continue
        linea.append(f"- {brazo}: max|delta p| = {sonda['max_abs_diff']:.3e} "
                     f"({sonda['clasificacion']}), rango = {sonda['rango']}, no concluyente = "
                     f"{sonda['no_concluyente']}, solape con el gate = "
                     f"{sonda['solape_con_gate']}.")
    linea += [
        "",
        "## Asimetria del control (declarada, no corregida)",
        "",
        (f"- `{A2_SGD['asimetria_declarada']}`"),
        (f"- `alpha` del control: {SGD_ALPHA:g} frente al alpha DERIVADO de A1 (~0.004075), "
         "cuatro ordenes de magnitud; y el intercepto de `SGDClassifier` no se regulariza, "
         "mientras que en A1 el sesgo es una coordenada mas de FTRL con su propio `n` y `z`."),
        ("- Las filas se le entregan al control YA EN EL ORDEN CONGELADO y con `shuffle=False` "
         "(enmenda 2 pre-ejecucion), de modo que consume la MISMA permutacion que A1; el "
         "`random_state` congelado `20261010+fold` se mantiene aunque con `shuffle=False` no "
         "altera el resultado (verificado: delta P = 0.0)."),
        ("- **Enmiendas pre-ejecucion registradas** (sin resultados): (1) banda de G1 de "
         "`[0.25, 0.60]` a `[0.05, 0.9*log(6)]`, porque el techo original era inalcanzable para "
         "un modelo sano (D-083 midio log_loss multiclase 1.3568) y cerraba el experimento por "
         "construccion; (2) `shuffle=False` en el control para respetar el orden congelado."),
        "",
    ]
    if integracion:
        linea += [
            "## Integracion P6/P7 (combinador congelado, regla argmax(p_comb * w))",
            "",
            ("| Brazo | macro-F1 comb | delta vs baseline | folds+ | IC95 | delta sin regla | "
             "delta vs nested | log_loss p_comb | sha256 OOF (16) |"),
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for brazo in BRAZOS:
            met = integracion[brazo]
            sha = resultado["entrenamiento"][brazo]["oof_artefacto"]["sha256"][:16]
            linea.append(
                f"| {brazo} | {met['corrected_cv']:.7f} | {met['delta_vs_baseline']:+.7f} | "
                f"{met['folds_positivos']}/5 | [{met['ic95'][0]:+.7f}, {met['ic95'][1]:+.7f}] | "
                f"{met['delta_sin_regla_vs_baseline']:+.7f} | {met['delta_vs_nested']:+.7f} | "
                f"{met['log_loss_p_comb']:.4f} | `{sha}` |")
        linea += [
            "",
            "## Complementariedad standalone (argmax del OOF del brazo, sin pesos)",
            "",
            ("| Brazo | macro-F1 solo | diagonal 6x6 | log_loss | rescate | dano | neto | rescate "
             "alta conf. | errores alta conf. campeon |"),
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for brazo in BRAZOS:
            diag = resultado["complementariedad"][brazo]
            contra = diag["vs_campeon"]
            linea.append(
                f"| {brazo} | {diag['standalone_cv']:.7f} | {diag['diagonal_ok']} | "
                f"{diag['log_loss_prob']:.4f} | {contra['rescate']} | {contra['danio']} | "
                f"{contra['neto']:+d} | {contra['rescates_alta_confianza']} | "
                f"{contra['errores_alta_confianza_campeon']} |")
        control = resultado["determinismo"]["control"]
        linea += [
            "",
            "## Gates, caps y veredicto",
            "",
            "| Gate | Resultado |",
            "|---|---|",
        ]
        for nombre, valor in resultado["gates"].items():
            linea.append(f"| {nombre} | {valor} |")
        linea += [
            "",
            (f"- Bootstrap pareado de la diferencia (F1_combA1 - F1_combA2), "
             f"{N_BOOTSTRAP} remuestreos con semilla {SEMILLA}: delta medio "
             f"{control['delta_medio']:+.7f}, IC95 [{control['ic95'][0]:+.7f}, "
             f"{control['ic95'][1]:+.7f}] (DIAGNOSTICO, el gate de control es el delta puntual)."),
            (f"- Caps: sonda de atribucion no publicada = "
             f"{resultado['caps']['sonda_atribucion_no_publicada']}; determinismo no concluyente = "
             f"{resultado['caps']['determinismo_no_concluyente']}."),
            (f"- Calibracion: t_fold0 = "
             f"{(resultado['calibracion'] or {}).get('t_fold0_seg', 0.0):.1f} s, proyeccion "
             f"{PROYECCION_FACTOR:g} x = "
             f"{(resultado['calibracion'] or {}).get('proyeccion_horas', 0.0):.4f} h (limite "
             f"{PROYECCION_MAX_H} h)."),
        ]
    else:
        linea += [
            "## Integracion, bootstrap y gates de delta: NO EJECUTADOS",
            "",
            ("La puerta G1-G6 fallo en el fold 0: por el pre-registro NO se integra, NO se hace "
             "bootstrap, NO hay segundo alpha y NO hay fallback. Los OOF quedan parciales "
             "diagnosticos y no se tocan los artefactos congelados del campeon."),
        ]
    linea += [
        "",
        "## Novedad registrada",
        "",
        f"- Que es: {NOVEDAD_PREREG['que_es']}.",
        f"- Lo que no es: {NOVEDAD_PREREG['no_es']}.",
        f"- Diferencia con D-083: {NOVEDAD_PREREG['diferencia_con_d083']}.",
        ("- Cruces explicitos: no es embedding; no es representacion nueva; no usa datos "
         "externos. Registro en artefactos NUEVOs (registro de novedad y fila anexada en el "
         "historial)."),
        (f"- Via B descartada: B1 {VIA_B_PREREG['B1_grafo_transductivo'][:60]}; B2 "
         f"{VIA_B_PREREG['B2_aumentacion_masking'][:40]}; B3 "
         f"{VIA_B_PREREG['B3_cabeza_multitarea'][:40]}."),
        "",
        "## Auditorias y asimetrias tecnicas",
        "",
        ("- **D-083 data_hash del manifiesto**: "
         f"{resultado['verificacion']['corpus_prohibido']}. Se congilan y verifican el sha256 del "
         "archivo del corpus, el sha256 del array de textos con la receta determinista "
         "(longitud de 8 bytes LE + bytes UTF-8 por texto, en orden) y el sha256 del dict del "
         "vectorizador. No se reordena nada para hacer cuadrar un hash."),
        (f"- **Receta de hash de textos**: {resultado['verificacion']['receta_hash_textos']}."),
        ("- **ngram_range**: la config congelada lo declara como lista y sklearn 1.8.0 exige "
         "tupla; en el fit se convierte a tupla (mismo valor) y el sha256 congelado del dict se "
         "calcula SIEMPRE sobre la CONFIG con la lista."),
        ("- **Orden de datos**: una sola permutacion por fold con "
         f"`np.random.default_rng({SEMILLA}+fold)`, congelada y reutilizada por los dos brazos; "
         "la curva online de A1 se registra con un callback que no altera la aritmetica."),
        ("- **log_loss**: A1 y A2 entregan probabilidades que suman 1 por construccion (softmax "
         "de 6 logits y `predict_proba`), que es la diferencia frente al OvR de D-083; la "
         "sonda (e) lo verifica contra sklearn y la (f) verifica la suma de las filas."),
        ("- **Escritura atomica**: temporal + `os.replace` en el OOF y en el checkpoint."),
        ("- **Familias vetadas**: la auditoria AST revisa este mismo archivo, importes Y "
         "literales de ruta, contra las diez familias vetadas y las cuatro particiones "
         "reservadas. La libreria de bandits instalada en el venv se descarta y no se importa: "
         "el FTRL-Proximal es implementacion propia verificada por las sondas (a), (b) y (e)."),
        "",
        "## Veredicto",
        "",
        f"**{resultado['veredicto']}**: {resultado['siguiente_paso']}",
        "",
        "## Compliance",
        "",
        ("- 0 embeddings, 0 transformers, 0 LLM, 0 datos externos; sin particiones reservadas "
         "leidas ni escritas; el campeon V30 + plug-in permanece intacto."),
        ("- Un experimento = un script nuevo; no se edito ningun script ya ejecutado ni el "
         "acta de D-083."),
        ("- La segunda particion (semilla 987654) NO se ejecuta: la autoriza el usuario y el "
         "runner solo la declara en `siguiente_paso`."),
        "",
    ]
    INFORME.write_text("\n".join(linea), encoding="utf-8")
    print(f"  informe: {ruta_relativa(INFORME)}", flush=True)


def anotar_runlog(resultado: dict) -> bool:
    """Anade el bloque del experimento a RUNLOG.md una sola vez (append de texto).

    Parameters
    ----------
    resultado:
        JSON de resultados.

    Returns
    -------
    bool
        True si se anadio el bloque; False si ya estaba.
    """
    marca = "V33-E1 ejecutado"
    if marca in RUNLOG.read_text(encoding="utf-8"):
        return False
    puerta = resultado["puerta"]
    linea = [
        "",
        "---",
        "",
        f"## V33-E1 ejecutado: FTRL-Proximal softmax multiclase de una pasada - {FECHA}",
        "",
        f"- Veredicto: {resultado['veredicto']}.",
        (f"- Puerta G1-G6 en el fold 0 (consumido-diagnostico): pasa = {puerta['pasa']}; "
         f"brazos con gates fallidos: "
         f"{ {brazo: puerta['brazos_fallidos'][brazo] for brazo in BRAZOS} }."),
        (f"- Baseline recomputado {BASELINE:.7f}; ancla trivial "
         f"{ANCLA_TRIVIAL!r} reproducida; sondas (a)-(f) todas pasan."),
        (f"- RSS pico {resultado['rss_pico_gb']:.2f} GB (tope {RAM_TOPE_GB:g}); {HILOS} hilos; "
         f"duracion {resultado['duracion_seg']:.0f} s."),
        ("- Sin particiones reservadas: sin evaluacion sobre la particion bloqueada del "
         "campeon, sin envio."),
        (f"- Artefactos: {ruta_relativa(RESULTADOS_JSON)} | {ruta_relativa(ESTADO_JSON)} | "
         f"{ruta_relativa(INFORME)} | {ruta_relativa(FIGURA)} | {ruta_relativa(NOVELTY_CSV)} | "
         + " | ".join(ruta_relativa(ruta_oof(brazo)) for brazo in BRAZOS) + "."),
    ]
    with RUNLOG.open("a", encoding="utf-8") as manejador:
        manejador.write("\n" + "\n".join(linea) + "\n")
    return True


def anotar_experiments_csv(resultado: dict) -> bool:
    """Anade la fila V33-E1 a experiments.csv (22 columnas de la cabecera, sin CR).

    Parameters
    ----------
    resultado:
        JSON de resultados.

    Returns
    -------
    bool
        True si se anadio la fila; False si ya estaba.

    Raises
    ------
    Detencion
        Si la fila nueva no respeta las columnas de la cabecera (codigo 1).
    """
    with EXPERIMENTS_CSV.open("r", encoding="utf-8", newline="") as manejador:
        filas = list(csv.reader(manejador))
    cabecera = filas[0]
    if "V33-E1" in {fila[0] for fila in filas[1:] if fila}:
        return False
    if len(cabecera) != 22:
        raise Detencion(f"[STOP] la cabecera de experiments.csv tiene {len(cabecera)} columnas, "
                        "no 22 (exit 1)", 1)
    puerta = resultado["puerta"]
    integracion = resultado.get("integracion") or {}
    a1 = integracion.get("A1_ftrl_softmax")
    if a1 is None:
        resumen = (f"puerta G1-G6 cerrada en el fold 0 (pasa = {puerta['pasa']}); sin "
                   "integracion, sin bootstrap y sin segundo alpha por pre-registro")
        media, desviacion, decision = "", "", ("cierre documentado NO_ATRIBUIBLE_BY_GATE; "
                                               "revisar la puerta con el arquitecto")
        interpretacion = (f"veredicto {resultado['veredicto']}: la puerta fail-fast de "
                          "convergencia detuvo el ciclo antes de integrar; el fold 0 queda "
                          "consumido-diagnostico y su fallo no toca delta")
    else:
        a2 = integracion["A2_sgd_1p"]
        media = f"{float(np.mean(a1['folds'])):.7f}"
        desviacion = f"{float(np.std(a1['folds'])):.7f}"
        resumen = (f"A1 FTRL softmax comb {a1['corrected_cv']:.7f} (delta "
                   f"{a1['delta_vs_baseline']:+.7f}, {a1['folds_positivos']}/5 folds, IC95 "
                   f"[{a1['ic95'][0]:+.7f}, {a1['ic95'][1]:+.7f}]); A2 gemelo SGD delta "
                   f"{a2['delta_vs_baseline']:+.7f}")
        decision = resultado["siguiente_paso"]
        interpretacion = (f"veredicto {resultado['veredicto']} por el arbol pre-registrado; "
                          "asimetrias del control declaradas en el informe; sonda de atribucion "
                          "publicada")
    registro = {
        "experiment_id": "V33-E1", "fecha": FECHA, "bloque": "33.0",
        "etapa": "V33-E1 ejecucion: challenger FTRL-Proximal softmax multiclase de una pasada "
                 "online (A1 decisional) + gemelo SGD de una pasada (A2 control)",
        "pregunta": "el online proximal con la semantica correcta (una pasada, softmax "
                    "multiclase, alpha derivado) aporta complementariedad al campeon?",
        "hipotesis": "la divergencia de D-083 fue de configuracion (sigma inflado por alpha y "
                     "multi-pasada), no de implementacion",
        "features": "TF-IDF por fold (receta v31_02: word 1-2, min_df 3, max_df 0.5, sublinear_tf) "
                    "ajustada SOLO en fold-train; 0 embeddings",
        "preprocessing": "vocabulario e idf por fold; sesgo como coordenada regularizada con "
                         "x0=1.0; p = softmax de 6 logits sin temperatura",
        "representacion": "TF-IDF dispersa (misma receta, artefactos nuevos recalculados y "
                          "congelados por fold)",
        "modelo": "FTRL-Proximal softmax K=6 propio (A1) / SGDClassifier log_loss 1 pasada (A2)",
        "hiperparametros": f"congelados: alpha = 3/(2*sqrt(T)) DERIVADO en runtime (0 valores "
                           f"evaluados), lambda1 {LAMBDA1}, lambda2 {LAMBDA2}, beta {BETA}, "
                           f"x0 {X0}, sin power_t, 1 pasada, control alpha {SGD_ALPHA:g}, "
                           f"{HILOS} hilos, seed {SEMILLA}+fold",
        "validacion": "particion canonica; puerta G1-G6 fail-fast en el fold 0; integracion "
                      "P6/P7; bootstrap pareado 2000 (semilla 20261010); gates de la serie; "
                      "sondas (a)-(f), de atribucion, de determinismo y de refit en 5 folds",
        "macro_f1_mean": media, "macro_f1_std": desviacion,
        "weighted_f1_mean": "", "micro_f1_mean": "", "accuracy_mean": "",
        "resultado": resumen, "interpretacion": interpretacion, "decision": decision,
        "duracion_seg": f"{float(resultado['duracion_seg']):.1f}",
        "artefactos": f"{ruta_relativa(RESULTADOS_JSON)} | {ruta_relativa(INFORME)} | "
                      f"{ruta_relativa(FIGURA)} | {ruta_relativa(NOVELTY_CSV)} | "
                      + " | ".join(ruta_relativa(ruta_oof(brazo)) for brazo in BRAZOS),
    }
    if set(registro) != set(cabecera):
        raise Detencion("[STOP] la fila nueva no respeta las columnas de experiments.csv (exit 1)",
                        1)
    with EXPERIMENTS_CSV.open("a", encoding="utf-8", newline="") as manejador:
        escritor = csv.writer(manejador, lineterminator="\n")
        escritor.writerow([registro[columna] for columna in cabecera])
    return True


def escribir_registro_novedad(resultado: dict) -> bool:
    """Escribe el registro de novedad (archivo NUEVO) con una sola fila y su cabecera.

    Parameters
    ----------
    resultado:
        JSON de resultados.

    Returns
    -------
    bool
        True si se escribio; False si la fila ya estaba.
    """
    if NOVELTY_CSV.exists():
        with NOVELTY_CSV.open("r", encoding="utf-8", newline="") as manejador:
            filas = list(csv.reader(manejador))
        if filas and filas[0] != COLUMNAS_NOVEDAD:
            raise Detencion("[STOP] la cabecera del registro de novedad no coincide (exit 1)", 1)
        if "V33-E1" in {fila[0] for fila in filas[1:] if fila}:
            return False
    fila = {
        "experiment_id": "V33-E1", "fecha": FECHA, "que_es": NOVEDAD_PREREG["que_es"],
        "no_es_embedding": "no", "no_es_representacion_nueva": "no",
        "no_usa_datos_externos": "no", "diferencia_con_d083": NOVEDAD_PREREG["diferencia_con_d083"],
        "verificacion": "sondas (a)-(f) obligatorias, sonda de atribucion |sigma*w|/|g| "
                        "obligatoria, sonda de refit del vectorizador en los 5 folds, sonda de "
                        "determinismo de A1 y A2, escritura atomica del OOF y del checkpoint",
        "artefactos": f"{ruta_relativa(RESULTADOS_JSON)} | {ruta_relativa(INFORME)} | "
                      f"{ruta_relativa(NOVELTY_CSV)} | "
                      + " | ".join(ruta_relativa(ruta_oof(brazo)) for brazo in BRAZOS),
    }
    if set(fila) != set(COLUMNAS_NOVEDAD):
        raise Detencion("[STOP] la fila de novedad no respeta su cabecera (exit 1)", 1)
    nuevo = not NOVELTY_CSV.exists()
    with NOVELTY_CSV.open("a", encoding="utf-8", newline="") as manejador:
        escritor = csv.writer(manejador, lineterminator="\n")
        if nuevo:
            escritor.writerow(COLUMNAS_NOVEDAD)
        escritor.writerow([fila[columna] for columna in COLUMNAS_NOVEDAD])
    return True


def anotar_historial(resultado: dict) -> bool:
    """Anexa UNA fila V33-E1 al historial de experiments respetando su cabecera existente.

    Parameters
    ----------
    resultado:
        JSON de resultados.

    Returns
    -------
    bool
        True si se anadio la fila; False si ya estaba.

    Raises
    ------
    Detencion
        Si la cabecera del historial no coincide con el esquema congelado (codigo 1).
    """
    if not HISTORIA_CSV.exists():
        raise Detencion(f"[STOP] no existe el historial {HISTORIA_CSV} (exit 1)", 1)
    with HISTORIA_CSV.open("r", encoding="utf-8", newline="") as manejador:
        filas = list(csv.reader(manejador))
    cabecera = filas[0]
    if cabecera != COLUMNAS_HISTORIA:
        raise Detencion(f"[STOP] el esquema del historial no coincide con el congelado: "
                        f"{cabecera} (exit 1)", 1)
    if "V33-E1" in {fila[1] for fila in filas[1:] if len(fila) > 1}:
        return False
    integracion = resultado.get("integracion") or {}
    a1 = integracion.get("A1_ftrl_softmax")
    puerta = resultado["puerta"]
    if a1 is None:
        registro = {
            "phase": "V33", "experiment_id": "V33-E1", "date": FECHA,
            "hypothesis": "la divergencia de D-083 fue de configuracion (sigma inflado por "
                          "alpha y multi-pasada), no de implementacion",
            "model": "FTRL-Proximal softmax K=6 propio, 1 pasada (A1) / SGDClassifier log_loss "
                     "1 pasada (A2)",
            "representation": "TF-IDF por fold (receta v31_02), artefactos nuevos por fold",
            "corrected_cv": "", "delta_vs_V20": "", "folds": "",
            "bootstrap": f"no se calcula: la puerta G1-G6 cerro el ciclo (pasa = "
                         f"{puerta['pasa']})",
            "rescue": "", "damage": "", "net": "",
            "decision": "NO_ATRIBUIBLE_BY_GATE",
            "resultado": f"gate: False | puerta G1-G6: {puerta['pasa']} | veredicto "
                         f"{resultado['veredicto']}"}
    else:
        diag = resultado["complementariedad"]["A1_ftrl_softmax"]["vs_campeon"]
        contra = resultado["determinismo"]["control"]
        registro = {
            "phase": "V33", "experiment_id": "V33-E1", "date": FECHA,
            "hypothesis": "la divergencia de D-083 fue de configuracion (sigma inflado por "
                          "alpha y multi-pasada), no de implementacion",
            "model": "FTRL-Proximal softmax K=6 propio, 1 pasada (A1) / SGDClassifier log_loss "
                     "1 pasada (A2)",
            "representation": "TF-IDF por fold (receta v31_02), artefactos nuevos por fold",
            "corrected_cv": f"{a1['corrected_cv']:.7f}",
            "delta_vs_V20": f"{a1['delta_vs_baseline']:+.7f}",
            "folds": f"{a1['folds_positivos']}/5",
            "bootstrap": f"[{a1['ic95'][0]:+.7f}, {a1['ic95'][1]:+.7f}]",
            "rescue": f"{diag['rescate']}", "damage": f"{diag['danio']}",
            "net": f"{diag['neto']}", "decision": resultado["siguiente_paso"][:80],
            "resultado": f"gate: {resultado['gates']['delta_A1_suficiente']} | marginal: "
                         f"{resultado['veredicto'] == 'MARGINAL'} | veredicto "
                         f"{resultado['veredicto']} | control A1-A2 "
                         f"{contra['delta_medio']:+.7f}"}
    if set(registro) != set(COLUMNAS_HISTORIA):
        raise Detencion("[STOP] la fila del historial no respeta su cabecera (exit 1)", 1)
    with HISTORIA_CSV.open("a", encoding="utf-8", newline="") as manejador:
        csv.writer(manejador, lineterminator="\n").writerow(
            [registro[columna] for columna in COLUMNAS_HISTORIA])
    return True


def resumen_consola(resultado: dict) -> None:
    """Resumen final en consola (maximo 15 lineas).

    Parameters
    ----------
    resultado:
        JSON de resultados.
    """
    puerta = resultado["puerta"]
    integracion = resultado.get("integracion") or {}
    print()
    print(f"[V33-E1] fin en {resultado['duracion_seg']:.0f} s | baseline "
          f"{resultado['verificacion']['baseline_recomputado']:.7f} | {HILOS} hilos | sondas "
          f"(a)-(f) pass | RSS pico {resultado['rss_pico_gb']:.2f} GB")
    print(f"  puerta G1-G6 (fold 0, consumido-diagnostico): pasa = {puerta['pasa']}")
    for brazo in BRAZOS:
        fallidos = puerta["brazos_fallidos"][brazo]
        detalle = resultado["entrenamiento"][brazo]["folds"].get("0", {})
        print(f"  {brazo}: T {detalle.get('T')} alpha {detalle.get('alpha_efectivo')!r} "
              f"ll_train {detalle.get('ll_train_prefijo', {}).get('1.0')} macro-F1 "
              f"{detalle.get('macro_f1_standalone_fold_val')} | fallidos: {fallidos}")
    atribucion = resultado["determinismo"].get("atribucion_fold_0")
    if atribucion:
        print(f"  sonda de atribucion (fold 0): razon media |sigma*w|/|g| por coordenada = "
              f"{atribucion['razon_media_coordenadas']!r} (sesgo "
              f"{atribucion['razon_coordenada_sesgo']!r})")
    for brazo in BRAZOS:
        sonda = resultado["determinismo"].get(f"sonda_{brazo}")
        if sonda:
            print(f"  determinismo {brazo}: max|delta p| = {sonda['max_abs_diff']:.3e} "
                  f"({sonda['clasificacion']})")
    if integracion:
        for brazo in BRAZOS:
            met = integracion[brazo]
            print(f"  integracion {brazo}: comb {met['corrected_cv']:.7f} (delta "
                  f"{met['delta_vs_baseline']:+.7f}, {met['folds_positivos']}/5 folds, IC95 "
                  f"[{met['ic95'][0]:+.7f}, {met['ic95'][1]:+.7f}])")
        print(f"  gates: {resultado['gates']}")
    caps = resultado.get("caps") or {}
    print(f"  caps: sonda no publicada = {caps.get('sonda_atribucion_no_publicada')} | "
          f"determinismo no concluyente = {caps.get('determinismo_no_concluyente')}")
    print(f"  VEREDICTO: {resultado['veredicto']}")
    print(f"  siguiente: {resultado['siguiente_paso']}")
    print(f"  artefactos: {ruta_relativa(RESULTADOS_JSON)} | {ruta_relativa(INFORME)} | "
          f"{ruta_relativa(FIGURA)} | {ruta_relativa(NOVELTY_CSV)}")


# =====================================================================================
# ---- 11. CONSTRUCCION DEL JSON DE RESULTADOS ----
# =====================================================================================
def construir_resultado(inicio: float, prereg: dict, auditoria: dict, baseline: float,
                        linea_base: dict, hashes_corpus: dict, detalle_constantes: dict,
                        detalle_versiones: dict, detalle_hashes: dict, sondas: dict,
                        estado: dict, puerta: dict, refit: dict, integracion: dict | None,
                        complementariedad: dict | None, invariantes: dict | None,
                        determinismos: dict, gates: dict, caps: dict, veredicto: str) -> dict:
    """Ensambla el JSON de resultados con config, hashes, sondas, puerta, gates y veredicto.

    Parameters
    ----------
    inicio:
        Marca temporal del inicio de la corrida.
    prereg:
        Contenido del pre-registro.
    auditoria:
        Salida de ``auditar_ast_propio``.
    baseline:
        Baseline recomputado.
    linea_base:
        Ancla trivial recalculada.
    hashes_corpus:
        Hashes recomputados del corpus.
    detalle_constantes:
        Detalle de los asserts de no-deriva.
    detalle_versiones:
        Detalle de las versiones instaladas.
    detalle_hashes:
        Detalle de los hashes verificados.
    sondas:
        Detalle de las sondas (a)-(f).
    estado:
        Checkpoint final.
    puerta:
        Detalle de la puerta G1-G6.
    refit:
        Detalle de la sonda de refit.
    integracion:
        Metricas del combinador por brazo, o None si la puerta cerro el ciclo.
    complementariedad:
        Diagnosticos standalone por brazo, o None.
    invariantes:
        Evidencias de no leakage, o None.
    determinismos:
        Sondas de determinismo y publicacion de la sonda de atribucion.
    gates:
        Gates calculados.
    caps:
        Caps aplicados.
    veredicto:
        Veredicto del arbol.

    Returns
    -------
    dict
        JSON de resultados completo.
    """
    oof_artefactos = {}
    for brazo in BRAZOS:
        ruta = ruta_oof(brazo)
        oof_artefactos[brazo] = {"archivo": ruta_relativa(ruta),
                                "existe": bool(ruta.exists()),
                                "sha256": sha256_archivo(ruta) if ruta.exists() else None}
    entrenamiento = {brazo: {"folds": estado["brazos"][brazo]["folds"],
                             "folds_completados": estado["brazos"][brazo]["folds_completados"],
                             "oof_artefacto": oof_artefactos[brazo]} for brazo in BRAZOS}
    alphas = {str(fold): estado["brazos"]["A1_ftrl_softmax"]["folds"][str(fold)]["alpha_efectivo"]
              for fold in sorted(estado["brazos"]["A1_ftrl_softmax"]["folds"])}
    return {
        "fase": "V33-E1 ejecucion: challenger FTRL-Proximal softmax multiclase de una pasada",
        "fecha": FECHA,
        "commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=False,
                                 capture_output=True, text=True, timeout=30).stdout.strip(),
        "prereg": {"json": ruta_relativa(PREREG_JSON), "sha256": sha256_archivo(PREREG_JSON),
                   "informe": ARCHIVO_PRE_REGISTRO, "puerta": ARCHIVO_PUERTA,
                   "commit_congelado": str(prereg["commit"])},
        "verificacion": {
            "constantes": detalle_constantes, "versiones": detalle_versiones,
            "hashes": detalle_hashes["checks"], "hashes_recomputados": detalle_hashes[
                "recomputados"], "hashes_esperados": detalle_hashes["esperados"],
            "hashes_corpus_recomputados": hashes_corpus,
            "receta_hash_textos": prereg["corpus"]["hash_contenido"],
            "corpus_prohibido": prereg["corpus"]["prohibido"],
            "corpus_receta": prereg["corpus"]["receta"],
            "baseline_recomputado": baseline, "baseline_esperado": BASELINE, "baseline_ok": True,
            "ids_alineados": True, "etiquetas_alineadas": True,
            "linea_base_mayoritaria": linea_base,
            "anclas": prereg["anclas"]},
        "compliance": {**auditoria, **COMPLIANCE_CIFRAS,
                       "auditorias_exigidas": COMPLIANCE_AUDITORIAS,
                       "nota_ast_audit": prereg["compliance"]["ast_audit"],
                       "alternativa_no_usada": A1_FTRL["alternativa_no_usada"]},
        "config": {
            "a1_ftrl_softmax": A1_FTRL, "a2_sgd_control": A2_SGD,
            "vectorizador": VECTORIZADOR, "vectorizador_fit": {"ngram_range": list(VECTORIZADOR[
                "ngram_range"])},
            "nota_ngram_range": "la config congelada declara ngram_range como lista; sklearn "
                                "1.8.0 exige tupla y en el fit se convierte (mismo valor); el "
                                "sha256 congelado del dict usa SIEMPRE la lista",
            "nota_orden": A1_FTRL["orden"] + "; la permutacion se entrega como filas al control, "
                          "que con shuffle=False (enmenda 2) la consume tal cual",
            "norm_dtype": prereg["representacion"]["norm_dtype"],
            "ajuste_vectorizador": prereg["representacion"]["ajuste"],
            "pesos_plugin": PESOS, "hilos": HILOS,
            "alpha_por_fold": alphas,
            "regla_alpha": {"formula": A1_FTRL["alpha"]["formula"], "T": A1_FTRL["alpha"]["T"],
                            "origen": A1_FTRL["alpha"]["origen"],
                            "valores_evaluados": A1_FTRL["alpha"]["valores_evaluados"],
                            "rango_esperado": A1_FTRL["alpha"]["rango_esperado"],
                            "ancla_n_train_fold": ANCLA_N_TRAIN_FOLD,
                            "ancla_alpha": ANCLA_ALPHA_ESPERADO},
            "seeds": {"global": SEMILLA, "orden_datos": DETERMINISMO_PREREG["orden"],
                      "sgd_random_state": A2_SGD["random_state"]},
            "integracion": {"c_logreg": C_LOGREG, "max_iter": MAX_ITER_LOGREG, "clip": L2_CLIP,
                            "decision": INTEGRACION_PREREG["decision"]},
            "puntos_de_la_pasada": list(PUNTOS_PASADA),
            "magnitud_ll_train": "perdida de train del modelo ajustado a ese prefijo del orden "
                                 "congelado (misma en A1 y A2); en A1 se publica ademas la media "
                                 "online del stream",
            "nota_capas": "la libreria de bandits instalada en el venv se descarta y NO se "
                          "importa: el FTRL-Proximal es implementacion propia verificada por "
                          "las sondas (a), (b), (e) y (f)"},
        "sondas": sondas,
        "puerta": puerta,
        "refit": refit,
        "vocabulario_idf_por_fold": estado["vectorizador_por_fold"],
        "determinismo": determinismos,
        "calibracion": estado["calibracion"],
        "rss_pico_gb": float(estado["rss_pico_gb"]), "ram_tope_gb": RAM_TOPE_GB,
        "rss_medicion": EJECUCION_PREREG["ram_medicion"],
        "sin_leakage": invariantes,
        "entrenamiento": entrenamiento,
        "integracion": integracion,
        "complementariedad": complementariedad,
        "gates": gates,
        "caps": caps,
        "veredicto": veredicto,
        "siguiente_paso": ARBOL_PREREG[veredicto],
        "duracion_seg": float(time.time() - inicio),
        "duracion_sondas_seg": float(sondas["duracion_seg"]),
        "tiempos_por_brazo_fold_seg": {brazo: {fold: float(detalle["tiempo_seg"]) for fold,
                                               detalle in estado["brazos"][brazo]["folds"]
                                               .items()} for brazo in BRAZOS},
    }


def construir_cierre_por_puerta(inicio: float, prereg: dict, auditoria: dict, baseline: float,
                                 linea_base: dict, hashes_corpus: dict, detalle_constantes: dict,
                                 detalle_versiones: dict, detalle_hashes: dict, sondas: dict,
                                 estado: dict, puerta: dict) -> dict:
    """Ensambla el JSON del cierre documentado por la puerta G1-G6 (sin integracion).

    Parameters
    ----------
    inicio:
        Marca temporal del inicio de la corrida.
    prereg:
        Contenido del pre-registro.
    auditoria:
        Salida de ``auditar_ast_propio``.
    baseline:
        Baseline recomputado.
    linea_base:
        Ancla trivial recalculada.
    hashes_corpus:
        Hashes recomputados del corpus.
    detalle_constantes:
        Detalle de los asserts de no-deriva.
    detalle_versiones:
        Detalle de las versiones instaladas.
    detalle_hashes:
        Detalle de los hashes verificados.
    sondas:
        Detalle de las sondas (a)-(f).
    estado:
        Checkpoint final.
    puerta:
        Detalle de la puerta fallida.

    Returns
    -------
    dict
        JSON de resultados del cierre por puerta.
    """
    determinismos = {f"sonda_{brazo}": estado["brazos"][brazo]["sonda_determinismo"]
                     for brazo in BRAZOS}
    determinismos["atribucion_fold_0"] = (estado["brazos"]["A1_ftrl_softmax"]["folds"].get("0")
                                          or {}).get("atribucion")
    determinismos["atribucion_publicada"] = bool(determinismos["atribucion_fold_0"])
    determinismos["no_concluyente"] = bool(
        any((determinismos[f"sonda_{brazo}"] or {}).get("no_concluyente", False)
            for brazo in BRAZOS))
    determinismos["tolerancia_declarada"] = TOL_DETERMINISMO
    determinismos["variabilidad_referencia"] = VARIABILIDAD
    determinismos["esperado"] = DETERMINISMO_PREREG["esperado"]
    determinismos["hilos"] = DETERMINISMO_PREREG["hilos"]
    determinismos["tolerancias"] = DETERMINISMO_PREREG["tolerancias"]
    resultado = construir_resultado(inicio, prereg, auditoria, baseline, linea_base,
                                    hashes_corpus, detalle_constantes, detalle_versiones,
                                    detalle_hashes, sondas, estado, puerta,
                                    {"descripcion": REFIT_PREREG["descripcion"],
                                     "tolerancia": REFIT_PREREG["tolerancia"],
                                     "exigido_por": REFIT_PREREG["exigido_por"],
                                     "folds": {}, "reproducido": None,
                                     "nota": "no se ejecuta: la puerta cerro el ciclo antes de "
                                             "terminar los folds"},
                                    None, None, None, determinismos, {}, {},
                                    "NO_ATRIBUIBLE_BY_GATE")
    resultado["siguiente_paso"] = puerta["consecuencia_si_falla"] + ". " + ARBOL_PREREG[
        "NO_ATRIBUIBLE_BY_GATE"]
    resultado["nota_cierre"] = ("sin fallback, sin segundo alpha, sin integrar y sin bootstrap: "
                                "el pre-registro prohibe reintentar con otro alpha (seria grid). "
                                "El fold 0 queda consumido-diagnostico y su fallo no toca delta.")
    return resultado


# =====================================================================================
# ---- 12. PIPELINE COMPLETO ----
# =====================================================================================
def ejecutar(verificar: bool) -> int:
    """Pipeline completo: verificacion, sondas, puerta, entrenamiento resumible, evaluacion.

    Parameters
    ----------
    verificar:
        Si True solo verifica y ejecuta las sondas, sin entrenar (imprime
        ``VERIFICACION OK``).

    Returns
    -------
    int
        0 si todo correcto, incluido el cierre documentado por la puerta G1-G6.

    Raises
    ------
    Detencion
        Con el codigo pre-registrado de la guarda que fallo.
    """
    inicio = time.time()
    autotest_numerico()
    prereg = cargar_preregistro()
    constantes_ok, detalle_constantes = verificar_constantes(prereg)
    versiones_ok, detalle_versiones = verificar_versiones(prereg)
    if not (constantes_ok and versiones_ok):
        raise Detencion("[STOP] el pre-registro congelado no se respeta (exit 2)", 2)
    auditoria = auditar_ast_propio()
    clase, fold_idx, v30, ids_canonicos, _cand_pred = cargar_canonico()
    textos, hashes_corpus = cargar_corpus(ids_canonicos, clase)
    hashes_ok, detalle_hashes = verificar_hashes(prereg, hashes_corpus)
    if not hashes_ok:
        raise Detencion("[STOP] hashes del pre-registro distintos a los recomputados (exit 2)", 2)
    pesos = np.array(PESOS, dtype=np.float64)
    champ_pred = np.argmax(v30 * pesos, axis=1)
    baseline = macro(clase, champ_pred)
    print(f"  baseline primario recomputado: {baseline:.7f} (esperado {BASELINE:.7f})", flush=True)
    if abs(baseline - BASELINE) > ABORTO_BASELINE:
        raise Detencion("[STOP] el baseline canonico no reproduce (exit 2)", 2)
    linea_base = linea_base_mayoritaria(clase)
    print(f"  corpus: {len(textos)} textos, ids en el MISMO orden que el canonico, cat.codes == "
          f"y_true; hashes de archivo, array de textos, vectorizador y ancla de D-083 OK",
          flush=True)
    print(f"  ancla trivial recalculada: {linea_base['macro_f1_linea_base']!r} (clase "
          f"{linea_base['clase']})", flush=True)
    print(f"  compliance AST: {auditoria['literales_de_ruta_revisados']} literales de ruta y "
          f"{auditoria['modulos_revisados']} modulos, 0 violaciones", flush=True)
    sondas = ejecutar_sondas()
    print(f"  sondas (a)-(f) en {sondas['duracion_seg']:.1f} s: todas pasan", flush=True)
    if verificar:
        print("VERIFICACION OK", flush=True)
        return 0

    estado = cargar_estado()
    estado["sondas"] = {nombre: bool(sondas[nombre]["pass"]) for nombre in NOMBRES_SONDAS}
    guardar_estado(estado)
    oofs, puerta_fallida = entrenar_con_puerta(estado, textos, clase, fold_idx)
    if puerta_fallida is not None:
        resultado = construir_cierre_por_puerta(inicio, prereg, auditoria, baseline, linea_base,
                                               hashes_corpus, detalle_constantes,
                                               detalle_versiones, detalle_hashes, sondas,
                                               estado, puerta_fallida)
        resultado["veredicto"] = "NO_ATRIBUIBLE_BY_GATE"
        escribir_json(RESULTADOS_JSON, resultado)
        print(f"  resultados: {ruta_relativa(RESULTADOS_JSON)}", flush=True)
        figura_curvas(estado, puerta_fallida, None, FIGURA)
        escribir_informe(resultado)
        print(f"  RUNLOG.md: {'anotado' if anotar_runlog(resultado) else 'ya estaba'}", flush=True)
        print(f"  experiments.csv: "
              f"{'fila V33-E1 anadida' if anotar_experiments_csv(resultado) else 'ya estaba'}",
              flush=True)
        print(f"  registro de novedad: "
              f"{'escrito' if escribir_registro_novedad(resultado) else 'ya estaba'}", flush=True)
        print(f"  historial: {'fila anexada' if anotar_historial(resultado) else 'ya estaba'}",
              flush=True)
        resumen_consola(resultado)
        return 0

    puerta = estado["puerta"]
    refit = refit_vectorizador(textos, fold_idx, estado)
    estado["refit"] = refit
    guardar_estado(estado)
    if not refit["reproducido"]:
        raise Detencion("[STOP] la sonda de refit no reproduce los hashes de vocabulario e idf "
                        "en los 5 folds (exit 2)", 2)
    invariantes = verificar_invariantes(estado, clase, fold_idx, oofs, refit)
    print(f"  invariantes de no leakage: {sum(bool(v) for v in invariantes.values())}/"
          f"{len(invariantes)} OK", flush=True)
    verificar_ram(estado)

    integracion: dict[str, dict] = {}
    preds_comb: dict[str, np.ndarray] = {}
    for brazo in BRAZOS:
        p_comb, detalle_combinador = combinador_oof(v30, oofs[brazo], clase, fold_idx)
        met = evaluar_combinacion(p_comb, clase, fold_idx, champ_pred, pesos)
        met["combinador"] = detalle_combinador
        preds_comb[brazo] = met.pop("pred_comb")
        integracion[brazo] = met
        print(f"  integracion {brazo}: comb {met['corrected_cv']:.7f} (delta "
              f"{met['delta_vs_baseline']:+.7f}, {met['folds_positivos']}/5)", flush=True)
    confianza = v30.max(axis=1)
    complementariedad = {brazo: diagnostico_brazo(oofs[brazo], clase, fold_idx, champ_pred,
                                                  confianza) for brazo in BRAZOS}
    diferencia = bootstrap_delta(clase, preds_comb["A2_sgd_1p"],
                                 preds_comb["A1_ftrl_softmax"])
    control = {"delta_A1": integracion["A1_ftrl_softmax"]["delta_vs_baseline"],
               "delta_A2": integracion["A2_sgd_1p"]["delta_vs_baseline"],
               "diferencia_puntual": (integracion["A1_ftrl_softmax"]["delta_vs_baseline"]
                                      - integracion["A2_sgd_1p"]["delta_vs_baseline"]),
               "delta_medio": diferencia["delta_medio"], "ic95": diferencia["ic95"],
               "p_delta_positivo": diferencia["p_delta_positivo"],
               "ic95_sin_cero": diferencia["ic95_sin_cero"],
               "criterio": GATES_PREREG["control"],
               "nota": "IC95 pareado de la diferencia como DIAGNOSTICO adicional; el gate de "
                       "control del pre-registro es el delta puntual"}
    determinismos = {f"sonda_{brazo}": estado["brazos"][brazo]["sonda_determinismo"]
                     for brazo in BRAZOS}
    determinismos["atribucion_fold_0"] = (estado["brazos"]["A1_ftrl_softmax"]["folds"]["0"]
                                          or {}).get("atribucion")
    determinismos["atribucion_publicada"] = bool(determinismos["atribucion_fold_0"])
    determinismos["no_concluyente"] = bool(
        any(determinismos[f"sonda_{brazo}"]["no_concluyente"] for brazo in BRAZOS))
    determinismos["tolerancia_declarada"] = TOL_DETERMINISMO
    determinismos["variabilidad_referencia"] = VARIABILIDAD
    determinismos["esperado"] = DETERMINISMO_PREREG["esperado"]
    determinismos["hilos"] = DETERMINISMO_PREREG["hilos"]
    determinismos["tolerancias"] = DETERMINISMO_PREREG["tolerancias"]
    determinismos["control"] = control
    determinismos["puerta"] = {"pasa": puerta["pasa"], "brazos_fallidos":
                               puerta["brazos_fallidos"]}
    gates, caps, veredicto = calcular_gates(True, integracion, invariantes, determinismos)
    print(f"  veredicto: {veredicto}", flush=True)
    resultado = construir_resultado(inicio, prereg, auditoria, baseline, linea_base,
                                    hashes_corpus, detalle_constantes, detalle_versiones,
                                    detalle_hashes, sondas, estado, puerta, refit, integracion,
                                    complementariedad, invariantes, determinismos, gates, caps,
                                    veredicto)
    escribir_json(RESULTADOS_JSON, resultado)
    print(f"  resultados: {ruta_relativa(RESULTADOS_JSON)}", flush=True)
    figura_curvas(estado, puerta, integracion, FIGURA)
    escribir_informe(resultado)
    print(f"  RUNLOG.md: {'anotado' if anotar_runlog(resultado) else 'ya estaba'}", flush=True)
    print(f"  experiments.csv: "
          f"{'fila V33-E1 anadida' if anotar_experiments_csv(resultado) else 'fila ya estaba'}",
          flush=True)
    print(f"  registro de novedad: "
          f"{'escrito' if escribir_registro_novedad(resultado) else 'ya estaba'}", flush=True)
    print(f"  historial: {'fila anexada' if anotar_historial(resultado) else 'ya estaba'}",
          flush=True)
    resumen_consola(resultado)
    return 0


def parsear_argumentos() -> argparse.Namespace:
    """Argumentos de linea de comandos.

    Returns
    -------
    argparse.Namespace
        ``--verificar`` (solo verificacion y sondas, sin entrenar).
    """
    analizador = argparse.ArgumentParser(
        description="V33-E1: implementacion del pre-registro congelado del challenger "
                    "FTRL-Proximal softmax multiclase de una pasada online, con control gemelo "
                    "SGD y puerta de convergencia G1-G6 fail-fast en el fold 0.")
    analizador.add_argument("--verificar", action="store_true",
                            help="solo verifica constantes, versiones, hashes, corpus, anclas, "
                                 "auditoria AST, baseline y sondas (a)-(f) (no entrena; "
                                 "imprime VERIFICACION OK)")
    return analizador.parse_args()


def main() -> int:
    """Entrada del script con manejo de los codigos de salida pre-registrados.

    Returns
    -------
    int
        Codigo de salida (0 correcto, incluido el cierre documentado por la puerta; 2, 4, 6, 7
        y 8 para las guardas; 1 inesperado).
    """
    argumentos = parsear_argumentos()
    try:
        return ejecutar(argumentos.verificar)
    except Detencion as detencion:
        print(str(detencion), flush=True)
        return detencion.codigo
    except Exception:  # noqa: BLE001 - se reporta el traceback y se sale con 1
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
