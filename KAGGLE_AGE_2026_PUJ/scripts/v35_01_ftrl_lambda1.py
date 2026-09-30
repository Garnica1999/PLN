"""v35_01_ftrl_lambda1.py
Fecha: 2026-09-30
Descripcion: VIA B del ciclo V35, reinicio ONE-SHOT de la familia FTRL sobre el miembro NO CUBIERTO
`lambda1 > 0`. El brazo decisional A1 es el nucleo FTRL-Proximal softmax multiclase K=6 de V34
(misma aritmetica bit a bit) con el UNICO cambio pre-registrado: `lambda1 = mediana(|z_j| > 0)`
medida al final de la pasada 1 del stream de TRAIN del fold 0 = 0.23993793862904017 (cero busqueda,
cero grid; `alpha*` queda CONGELADO e identico al de V34-E1, luego esto NO es un reintento de
alpha). El brazo A2 es un CONTROL DE PRESUPUESTO (SGDClassifier log_loss alpha 1e-6, learning_rate
optimal, partial_fit por pasada sobre los MISMOS ordenes congelados, misma particion ES y misma
regla de ES): NO es un gemelo mecanico porque no tiene L1, y por eso ninguna conclusion puede
descansar en el contraste A1 frente a A2.

Puerta: G0-G6 IDENTICAS a las de V34 (mismos umbrales ledos del pre-registro de V35), fail-fast
dentro del fold 0, antes de los folds 1-4 y antes de integrar; mas dos gates nuevos que aplican
SOLO a A1: G7_L1_activo (escasez efectiva de coordenadas con al menos una clase viva, |z_j| >
lambda1, mayor que cero en todas las pasadas y no degradada mas alla de 0.02 respecto de la
pasada 1) y G8_acotacion_norma (razon max/min de ||w||_2 entre la primera y la ultima pasada
<= 1.0, anclada a la HIPOTESIS misma). A2 queda exento de G7 y G8: se marcan `no_aplica` y NO
cuentan como fallo. Arbol pre-declarado: G7 o G8 fallan en A1 -> LINEA_ONLINE_CERRADA_POR_MECANISMO
(cierre DEFINITIVO, sin segundo lambda1 ni acta futura dentro de este espacio); G7 y G8 pasan pero
G0-G4 fallan en A1 -> HIPOTESIS_CONJUNTA_REFUTADA ("acota pero no aprende"); el resto de fallos ->
NO_ATRIBUIBLE_BY_GATE; si todo pasa, los gates de la serie (PROMISING/MARGINAL/CLOSED) sin ejecutar
la segunda particion. En los dos nodos de cierre `integracion = None` y delta y bootstrap quedan
VACIOS con su `motivo_de_no_calculo`.

Sondas: (a)-(j) heredadas de V34 sin cambio de criterio, mas (k), (l) y (m), NO-GATE, que
reproducen DENTRO del runner los tres casos del gemelo pre-ejecucion versionado (`lambda1 = 0`,
`lambda1 = alpha`, `lambda1 = mediana`) con las MISMAS reglas de update, el mismo alpha y los mismos
ordenes congelados, y los contrastan con el artefacto `experiments/v35_00b_sonda_lambda1.json`
declarando la tolerancia y si reproducen. No deciden el veredicto, pero se publican porque la
decision de mecanismo no puede descansar en evidencia importada de fuera del runner.

Anclaje de `lambda1` verificado en RUNTIME: durante la pasada 1 del fold 0 de A1 se recorre el
mismo stream (misma matriz, mismas etiquetas, mismo orden congelado, mismo alpha) con el nucleo de
ANCLAJE (`lambda1 = 0.0`, la trayectoria que DEFINE la mediana), se calcula `mediana(|z_j| > 0)` del
`z` almacenado y se exige diferencia EXACTA 0 con la constante congelada, se registra su sha256 y
se compara con el `ec1d1a7b13454f8b7309446576318487b9325df018c5b454b228a6a336630f66` de la sonda
previa; si no reproduce -> [STOP] exit 2. El `z` del brazo A1 ya regularizado se publica ademas
como diagnostico secundario, sin mezclarlo con el anclaje.

Estado del arte aplicado: nucleo FTRL-Proximal softmax multiclase PROPIO y multi-pasada (McMahan
et al. 2013) IDENTICO al de V34 (verificado alli con una tercera implementacion independiente,
max|delta w| = 4.663e-15): solo se actualizan las coordenadas no nulas de la fila, sigma =
(sqrt(n+g^2)-sqrt(n))/(alpha_t*beta), z += g - sigma*w, n += g^2 y w = sign(z)*(|z|-lambda1) /
((beta+sqrt(n))/alpha_t + lambda2); lambda1 = 0.23993793862904017 (mediana congelada), lambda2 1.0
en DENOMINADOR, beta 1.0, power_t 0.0, x0 1.0, float64. alpha NO se busca: DERIVADO en forma
cerrada, alpha* = 2*M/(d_norm*P_max^2) con M = ln(K-1) = ln(5) = 1.6094379124341003 y d_norm =
sqrt(1 - sum_c prior_c^2) del prior MEDIDO del ES-train del fold, verificado bit a bit en runtime
(sonda g). Vectorizador (word 1-2, min_df 3, max_df 0.5, sublinear_tf) ajustado SOLO en ES-train
de cada fold. Integracion P6/P7 congelada [L2(v30)||L2(p_brazo)] con LogReg(C=1.0, max_iter=2000)
por fold y decision argmax(p_comb*w) con pesos congelados; bootstrap pareado de 2000 remuestreos
(semilla 20261010). CERO embeddings, CERO transformers, CERO LLM, CERO datos externos, CERO
particiones reservadas; `artifacts/p33_scores/todos` se lee y NO se escribe; vowpalwabbit NO se
importa.

Fuentes obligatorias: experiments/v35_00_preregistro.json (fuente unica de verdad: protocolo,
constantes, hashes, puerta, sondas, rama terminal y acta), reports/v35_preregistro.md,
reports/v34_convergence_gate.md (normativa de G0-G6 que se hereda SIN CAMBIOS) y la sonda
pre-ejecucion versionada (experiments/v35_00b_sonda_lambda1.json y su script, con hash
congelado). Plantilla de mecanismos: scripts/v34_01_ftrl_softmax_multipasada.py (misma familia,
nucleo auditado); su nucleo NO se altera y su hash se verifica en runtime.

Modos de linea de comandos:
* (sin flags): pipeline completo (verificacion, sondas (a)-(m), estado, entrenamiento resumible con
  la puerta fail-fast, integracion y gates de la serie, informe, figura y anotadores).
* `--verificar`: TODO lo anterior salvo el estado y el entrenamiento; imprime
  `VERIFICACION OK` y NO escribe NADA (ni estado, ni OOF, ni checkpoints, ni filas en RUNLOG.md,
  experiments.csv, reports/v35_novelty_registry.csv ni en el historial).
* `--humo`: recorre la zona INTERMEDIA entre `--verificar` y el entrenamiento (creacion del
  checkpoint en memoria, mapeo de sondas y escritura atomica del estado) y PARA con un sentinel
  explicito y salida 0 SIN entrenar ningun fold. Es la cobertura permanente del gate de QA para
  la zona en la que en V34 se escondio un bug bloqueante; serializa el estado a un archivo
  TEMPORAL del sistema (comprueba el round-trip y lo borra) para no dejar artefactos en el
  proyecto, luego no deja checkpoints, OOF ni filas en ningun registro.

Codigos de salida: 0 fin correcto (incluye los cierres documentados NO_ATRIBUIBLE_BY_GATE,
LINEA_ONLINE_CERRADA_POR_MECANISMO e HIPOTESIS_CONJUNTA_REFUTADA); 2 verificacion (constantes,
versiones, hashes, corpus, anclas, auditoria AST, lecturas de datos, reproducibilidad de alpha* o
del anclaje de lambda1); 4 sonda (a)-(i) fallida antes de entrenar; 6 calibracion del fold 0
excedida; 7 timeout global; 8 RSS por encima del tope declarado; 1 error inesperado con traceback.
"""

from __future__ import annotations

# =====================================================================================
# ---- 0. ENTORNO E IMPORTS (los 32 hilos van ANTES de importar numpy/scipy/sklearn) ----
# =====================================================================================
import os

#: Los 32 hilos congelados se fijan antes de importar numpy (determinismo del BLAS). El
#: pre-registro exige OMP/MKL/OPENBLAS_NUM_THREADS = 32; el lazo FTRL es mono-hilo de todos modos.
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
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import f1_score, log_loss
from sklearn.model_selection import StratifiedShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src import config as cfg
from src import data as datos
from src.verificacion_numerica import autotest_numerico

# =====================================================================================
# ---- 1. SETUP: CONSTANTES CONGELADAS Y NO-DERIVA CON EL PRE-REGISTRO ----
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
PRESUPUESTO_DECLARADO_SEG = 1200.0
L2_CLIP = 1e-12
C_LOGREG = 1.0
MAX_ITER_LOGREG = 2000
TOL_DETERMINISMO = 1e-9
TOL_ANCLA_TRIVIAL = 1e-15
FASE_ESTADO = "V35-E1 ejecucion (checkpoint resumible)"

#: Anclas numericas congeladas: se recalculan y se contrastan en tiempo de ejecucion (exit 2).
ANCLA_TRIVIAL = 0.04854212180687087
ANCLA_GEMELO_SGD_D083 = 0.4934789
ANCLA_LOG_UNIFORME = 1.791759469228055
ANCLA_H_PRIOR_EXPLORACION = 1.7914536815044917
ANCLA_D_NORM_REFERENCIA = 0.912815416414756
ANCLA_ALPHA_PARA_D_NORM_REFERENCIA = 0.035263162376364156
ANCLA_N_TRAIN_FOLD = 135468

#: Constantes de la regla DERIVADA de alpha (0 valores evaluados; no es un grid).
M_MARGEN = 1.6094379124341003
POWER_T = 0.0
#: MEDIANA_Z_CONGELADA es el UNICO valor de lambda1 de este ciclo: `mediana(|z_j| > 0)` medida en
#: el stream de TRAIN del fold 0 al final de la pasada 1, congelada en el pre-registro V35 y
#: verificada bit a bit en runtime (diferencia EXACTA 0 y sha256 del snapshot de z contra el de la
#: sonda pre-ejecucion). Re-derivarla tras ver las gates seria un grid, luego queda prohibido.
MEDIANA_Z_CONGELADA = 0.23993793862904017
#: SHA256 del snapshot de `z` (matriz completa, float64, orden C) al final de la pasada 1 del
#: nucleo de ANCLAJE (lambda1 = 0.0) sobre el ES-train del fold 0; congelado en la sonda previa.
SHA256_Z_ANCLAJE = "ec1d1a7b13454f8b7309446576318487b9325df018c5b454b228a6a336630f66"
LAMBDA1 = MEDIANA_Z_CONGELADA
LAMBDA2 = 1.0
VARIANTE_L2 = "denominador"
BETA = 1.0
X0 = 1.0

#: Presupuesto con early stopping interno (nunca fold-val).
P_MAX = 10
P_MIN = 3
PACIENCIA = 2
MEJORA_MINIMA = 1e-4
TEST_SIZE_ES = 0.10

#: Umbrales de la puerta G0-G6 leidos del pre-registro V35 (espejo numerico; exit 2 si derivan).
#: G0-G6 se heredan de V34 SIN CAMBIOS; G7 y G8 son nuevos y aplican SOLO a A1.
G0_FACTOR_INFERIOR = 0.5
G0_FACTOR_SUPERIOR = 2.0
G0_TOLERANCIA = 1e-9
G1_PISO = 0.05
G1_TECHO_MARGEN = 0.01
G2_TOLERANCIA = 1e-06
G3_MINIMO = 0.30
G4_FACTOR_INFERIOR = 0.5
G4_FACTOR_SUPERIOR = 1.5
G5_TOLERANCIA = 0.0
G6_TOLERANCIA = 1e-09
#: G7 (solo A1): escasez efectiva = coordenadas con AL MENOS una clase viva (|z_j| > lambda1)
#: sobre el total de coordenadas. Pasa si es mayor que cero en TODAS las pasadas ejecutadas y si no
#: se degrada respecto de la pasada 1 mas alla de la tolerancia DECLARADA (anclaje: la magnitud de
#: pasada 1 medida; ni el 0.5 arbitrario ni el piso 0 tautologico).
G7_TOLERANCIA_DEGRADACION = 0.02
#: G8 (solo A1): razon max/min de ||w||_2 entre la primera y la ultima pasada. El umbral 1.0 es la
#: HIPOTESIS misma (si el L1 acota la norma, esta no crece), no un factor arbitrario.
G8_RAZON_MAXIMA = 1.0

#: Tolerancias de las sondas (a)-(m) y dimensiones de los problemas de prueba.
TOL_SONDA_A = 1e-12
TOL_SONDA_B = 1e-09
TOL_SONDA_C = 1e-06
TOL_SONDA_E = 1e-09
TOL_SONDA_F = 1e-09
#: Tolerancia de las sondas (k), (l) y (m): reproducen un gemelo DETERMINISTA ya medido con el
#: mismo nucleo, el mismo alpha y los mismos ordenes congelados, luego la tolerancia es CERO. Son
#: sondas NO-GATE: una no reproduccion se publica, no detiene el ciclo.
TOL_SONDA_KLM = 0.0
#: Pasadas que ejecutan las sondas (k), (l) y (m), las mismas de la sonda pre-ejecucion.
PASADAS_SONDA_KLM = 3
DIM_SONDA_A = (200, 50)
DIM_SONDA_D = (400, 20)
DIM_SONDA_MINI = (1500, 50)
PASADAS_SONDA_A = 3
SEMILLA_SONDA = 7
SEMILLA_SONDA_D = 13
SEMILLA_SONDA_E = 29
SEMILLA_SONDA_F = 31
SEMILLA_SONDA_C = 3
SEMILLA_SONDA_MINI = 101

#: Particiones reservadas y familias de librerias vetadas. Ningun literal de ruta de este
#: script puede contener las cuatro primeras en minusculas; las diez ultimas se comprueban
#: ADEMAS sobre los modulos importados, no solo sobre los literales de ruta.
PARTICIONES_RESERVADAS = ("test", "holdout", "submission", "leaderboard")
FAMILIAS_VETADAS = ("vowpal", "vw", "torch", "transformers", "gensim", "fasttext",
                    "sentence-transformers", "word2vec", "glove", "spacy-vectors")
EXTENSIONES_RUTA = (".csv", ".npy", ".json", ".npz", ".txt", ".md", ".png")

#: Listas de la auditoria de lecturas/escrituras de datos (seccion 3b). Los literales de ruta
#: de este script se comprueban contra estas dos listas segun el rol de la funcion que los usa.
RUTAS_LECTURA_PERMITIDAS = ("data/raw/age_train.csv", "artifacts/p33_scores/todos")
RUTAS_ESCRITURA_PERMITIDAS = ("artifacts/p35_scores/v35_e1", "checkpoints/v35_e1", "experiments/",
                              "reports/")
#: Funciones que SOLO leen y funciones que SOLO escriben; el resto se comprueba contra la union.
FUNCIONES_LECTORAS = ("cargar_preregistro", "cargar_canonico", "cargar_corpus", "verificar_hashes",
                      "auditar_ast_propio", "auditar_lecturas_de_datos", "ajustar_vectorizador_fold",
                      "indices_fold", "particion_es", "generar_ordenes", "anadir_sesgo",
                      "refit_vectorizador")
FUNCIONES_ESCRIPTORAS = ("escribir_json", "escribir_json_atomico", "guardar_estado", "guardar_oof",
                         "guardar_pesos", "figura_curvas", "escribir_informe", "anotar_runlog",
                         "anotar_experiments_csv", "escribir_registro_novedad", "anotar_historial")

BRAZOS = ("A1_ftrl_lambda1", "A2_sgd_presupuesto")
NOMBRES_BRAZO = {"A1_ftrl_lambda1": "A1 (FTRL-Proximal softmax multiclase multi-pasada con L1 = "
                                    "mediana congelada, decisional)",
                 "A2_sgd_presupuesto": "A2 (control de PRESUPUESTO: no gemelo mecanico, no tiene "
                                       "L1)"}
OOF_ARCHIVO = {"A1_ftrl_lambda1": "a1_ftrl_lambda1",
               "A2_sgd_presupuesto": "a2_sgd_presupuesto"}
NOMBRES_GATE = ("G0_objetivo_de_la_regla", "G1_banda_ll_train", "G2_progreso",
                "G3_macro_f1_fold_val", "G4_frecuencias", "G5_determinismo", "G6_suma_unidad",
                "G7_L1_activo", "G8_acotacion_norma")
#: Gates que aplican a A2 (G7 y G8 son exclusive de A1 porque A2 no tiene L1) y los que no se
#: pueden decidir con una sola corrida del fold 0 (G5 exige la segunda corrida identica).
GATES_EXCLUSIVOS_A1 = ("G7_L1_activo", "G8_acotacion_norma")
GATES_EXCLUSIVOS_A2 = (NOMBRES_GATE[0], NOMBRES_GATE[1], NOMBRES_GATE[2], NOMBRES_GATE[3],
                      NOMBRES_GATE[4], NOMBRES_GATE[5], NOMBRES_GATE[6])
#: Campos numericos que el checkpoint exige por fold y por brazo.
CAMPOS_FOLD = ("n_vocabulario", "vocab_sha256", "idf_sha256", "n_filas_fit_es_train",
               "n_filas_es_val", "n_train_fold", "n_val_fold", "n_features_es_train",
               "n_features_es_val", "n_coordinates_a1", "alpha_efectivo", "d_norm", "prior_es",
               "h_prior_es", "prior_es_sha256", "alpha_sha256", "es_train_sha256", "es_val_sha256",
               "lambda1", "anclaje_lambda1", "ordenes_sha256", "pasada_de_parada", "mejor_pasada",
               "trayectoria", "sesgo",
               "t_", "n_actualizaciones", "macro_f1_standalone_fold_val",
               "frecuencias_fold_val", "max_abs_suma_menos_uno", "tiempo_seg")

#: Cifras de compliance congeladas: ocho contadores a CERO (0 embeddings, 0 transformers, 0 LLM,
#: 0 datos externos y 0 uso de las cuatro particiones reservadas). Se comparan una a una con el
#: pre-registro; cualquier valor distinto detiene el runner con codigo 2.
COMPLIANCE_CIFRAS = {"embeddings": 0, "transformers": 0, "llm": 0, "datos_externos": 0,
                     "test": 0, "holdout": 0, "leaderboard": 0, "submission": 0}
#: Las once auditorias que el pre-registro V35 exige; la lista se compara integra (orden incluido).
COMPLIANCE_AUDITORIAS = [
    "hashes de los 6 .npy + manifest",
    "corpus archivo y textos",
    "vectorizador",
    "nucleo de V34 sin cambios",
    "sonda previa versionada",
    "vocabulario e idf por fold",
    "sondas (a)-(m)",
    "tabla de mecanismo por pasada",
    "determinismo A1 y A2",
    "RSS con ru_maxrss",
    "escritura atomica"]

#: Configuracion del vectorizador: IDENTICA a la canonica del proyecto y a la congelada del
#: pre-registro (hash 9bae1576 tras la enmienda del token_pattern). En
#: el fit se convierte ngram_range a tupla porque sklearn 1.8.0 lo exige (mismo valor); el sha256
#: congelado del dict se calcula SIEMPRE sobre la CONFIG con la lista.
VECTORIZADOR = {"analyzer": "word", "ngram_range": [1, 2], "min_df": 3, "max_df": 0.5,
                "sublinear_tf": True, "lowercase": True, "strip_accents": None,
                "token_pattern": r"(?u)\b\w+\b"}
VECTORIZADOR_FIT = {**VECTORIZADOR, "ngram_range": (1, 2)}

#: EJECUCION del control A2: espejo EJECUTABLE de la configuracion congelada del ciclo V34 (que se
#: hereda sin cambios, porque el pre-registro V35 congela `alpha*` y el presupuesto, no el control).
#: El estimador NO se construye nunca con ``max_iter = 10``: se construye con ``max_iter = 1`` y
#: ``warm_start = True`` y se llama a ``partial_fit`` UNA vez por orden congelado de cada pasada,
#: porque ``BaseSGDClassifier._fit`` reinicia ``self.t_ = 1.0`` en cada llamada. El numero de
#: pasadas (10) es el que congela el pre-registro; lo que se conserva es COMO se ejecuta, porque
#: el ES interno exige evaluar despues de cada pasada.
AJUSTE_CONTROL = {"modo": "partial_fit_por_pasada", "llamadas": P_MAX, "max_iter": 1,
                  "warm_start": True, "tol": None, "shuffle": False, "average": False,
                  "fit_intercept": True, "alpha_reg": 1e-06, "learning_rate": "optimal",
                  "random_state": f"{SEMILLA}+fold", "n_jobs": HILOS}

#: Texto de la fila de novedad: que es este experimento y en que se diferencia de V34-E1. Es la
#: misma familia de nucleo con el MISMO alpha y el MISMO presupuesto, luego la diferencia NO es una
#: representacion nueva ni un embedding: es el regularizador L1 con anclaje derivado.
NOVEDAD_QUE_ES = ("FTRL-Proximal softmax multiclase PROPIO con regularizador L1 ANCLADO por "
                  "mediana(|z_j| > 0) medida en el stream de TRAIN del fold 0 al final de la pasada "
                  "1 (= 0.23993793862904017; cero busqueda, cero grid), con alpha* DERIVADO en "
                  "forma cerrada e IDENTICO al de V34-E1 y presupuesto MULTI-PASADA con early "
                  "stopping INTERNO, frente a un control de PRESUPUESTO que no tiene L1")
NOVEDAD_DIFERENCIA_V34 = ("V34-E1 ejecuto el mismo nucleo con lambda1 = 0.0 y refuto la linea por "
                          "mecanismo; aqui el unico miembro no cubierto de la familia es "
                          "lambda1 > 0, con umbral duro w_j = 0 si |z_j| <= lambda1, y el anclaje "
                          "es una MEDIDA (mediana), no un valor elegido: un segundo lambda1 seria "
                          "grid y esta prohibido por el pre-registro")
NOVEDAD_DIFERENCIA_V33 = NOVEDAD_DIFERENCIA_V34

CANONICO = ROOT / "artifacts" / "p33_scores" / "todos"
OOF_DIR = ROOT / "artifacts" / "p35_scores" / "v35_e1"
CKPT_DIR = ROOT / "checkpoints" / "v35_e1"
PREREG_JSON = cfg.EXPERIMENTS / "v35_00_preregistro.json"
NUCLEO_V34 = ROOT / "scripts" / "v34_01_ftrl_softmax_multipasada.py"
SONDA_SCRIPT = ROOT / "scripts" / "v35_00b_sonda_lambda1.py"
SONDA_JSON = cfg.EXPERIMENTS / "v35_00b_sonda_lambda1.json"
SONDA_MD = cfg.REPORTS / "v35_sonda_lambda1.md"
ESTADO_JSON = cfg.EXPERIMENTS / "v35_01_estado.json"
RESULTADOS_JSON = cfg.EXPERIMENTS / "v35_01_ftrl_lambda1.json"
INFORME = cfg.REPORTS / "v35_e1_ftrl_lambda1.md"
FIGURA = cfg.FIGURES / "v35_e1_curvas.png"
NOVELTY_CSV = cfg.REPORTS / "v35_novelty_registry.csv"
HISTORIA_CSV = cfg.REPORTS / "age_complete_experiment_history.csv"
RUNLOG = ROOT / "RUNLOG.md"
EXPERIMENTS_CSV = ROOT / "experiments.csv"
ARCHIVO_PRE_REGISTRO = "reports/v35_preregistro.md"
ARCHIVO_PUERTA = "reports/v34_convergence_gate.md"
RUTAS_HASHES = {
    "canonico_y_true": CANONICO / "y_true.npy",
    "canonico_fold_idx": CANONICO / "fold_idx.npy",
    "canonico_v30_prob": CANONICO / "v30_prob.npy",
    "canonico_ids": CANONICO / "ids.npy",
    "canonico_cand_pred": CANONICO / "cand_pred.npy",
    "canonico_cand_prob": CANONICO / "cand_prob.npy",
    "manifest_canonico": CANONICO / "manifest.json",
    "nucleo_v34": NUCLEO_V34,
}
CLAVES_HASH_CORPUS = ("corpus_archivo", "corpus_textos", "vectorizador", "sonda_lambda1_script",
                      "sonda_lambda1_json")
RUTAS_HASH_CORPUS = {
    "sonda_lambda1_script": SONDA_SCRIPT,
    "sonda_lambda1_json": SONDA_JSON,
}
#: Cabecera congelada del historial de experiments (15 columnas, esquema existente).
COLUMNAS_HISTORIA = ["phase", "experiment_id", "date", "hypothesis", "model", "representation",
                     "corrected_cv", "delta_vs_V20", "folds", "bootstrap", "rescue", "damage",
                     "net", "decision", "resultado"]
#: Cabecera del registro de novedad (archivo NUEVO escrito por este runner).
COLUMNAS_NOVEDAD = ["experiment_id", "fecha", "que_es", "no_es_embedding",
                    "no_es_representacion_nueva", "no_usa_datos_externos", "no_es_un_grid",
                    "diferencia_con_v34", "verificacion", "artefactos"]
#: Nodos del arbol de decision de este ciclo: los dos nodos de cierre de la via B, el nodo de
#: atribucion por puerta y los tres gates de la serie (la segunda particion NO se ejecuta).
NODOS_ARBOL = ("NO_ATRIBUIBLE_BY_GATE", "LINEA_ONLINE_CERRADA_POR_MECANISMO",
               "HIPOTESIS_CONJUNTA_REFUTADA", "PROMISING", "MARGINAL", "CLOSED")


class Detencion(RuntimeError):
    """Detencion pre-registrada: el mensaje se imprime y el proceso sale con ``codigo``.

    Se usa en lugar de ``assert`` porque con ``python -O`` los asserts desaparecen y las
    invariantes de integridad deben valer SIEMPRE.

    Parameters
    ----------
    mensaje:
        Texto que se imprime antes de salir.
    codigo:
        Codigo de salida pre-registrado (2, 4, 6, 7 u 8).
    """

    def __init__(self, mensaje: str, codigo: int) -> None:
        super().__init__(mensaje)
        self.codigo = codigo


# =====================================================================================
# ---- 2. UTILIDADES DE HASH, JSON, METRICAS Y NORMALIZACION ----
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


def sha256_textos(textos: object) -> str:
    """SHA256 determinista de una secuencia de textos: longitud (8 bytes LE) + bytes UTF-8.

    Es la receta CONGELADA del pre-registro para el array de textos del corpus (identica a la de
    D-083); aqui tambien se usa para el vocabulario ordenado de cada fold (misma receta, mismo
    orden). El ``data_hash`` del manifiesto canonico NO se usa: quedo invalidado en D-083 porque
    es un hash de punteros y no de contenido.

    Parameters
    ----------
    textos:
        Secuencia de textos en el orden exacto que se quiere congelar.

    Returns
    -------
    str
        Digest hexadecimal de 64 caracteres.
    """
    resumen = hashlib.sha256()
    for texto in textos:  # type: ignore[union-attr]
        bruto = str(texto).encode("utf-8")
        resumen.update(len(bruto).to_bytes(8, "little"))
        resumen.update(bruto)
    return resumen.hexdigest()


def sha256_f64(valores: np.ndarray) -> str:
    """SHA256 de un arreglo float64 en orden C (idf de cada fold, prior, alpha*).

    Parameters
    ----------
    valores:
        Arreglo float64 (o escalar float64) a resumir.

    Returns
    -------
    str
        Digest hexadecimal de 64 caracteres.
    """
    return sha256_bytes(np.ascontiguousarray(np.atleast_1d(valores),
                                             dtype=np.float64).tobytes())


def sha256_i64(valores: np.ndarray) -> str:
    """SHA256 de un arreglo int64 en orden C (indices de particion ES y ordenes por pasada).

    Parameters
    ----------
    valores:
        Arreglo int64 a resumir.

    Returns
    -------
    str
        Digest hexadecimal de 64 caracteres.
    """
    return sha256_bytes(np.ascontiguousarray(np.atleast_1d(valores), dtype=np.int64).tobytes())


def sha256_f64_2d(matriz: np.ndarray) -> str:
    """SHA256 de una matriz float64 en orden C (identidad de pesos de los dos brazos).

    Parameters
    ----------
    matriz:
        Arreglo float64 de dos dimensiones.

    Returns
    -------
    str
        Digest hexadecimal de 64 caracteres.
    """
    return sha256_bytes(np.ascontiguousarray(matriz, dtype=np.float64).tobytes())


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


def ruta_pesos(fold: int, brazo: str) -> Path:
    """Ruta congelada del checkpoint de pesos de un fold y un brazo.

    Parameters
    ----------
    fold:
        Fold canonico.
    brazo:
        Nombre del brazo.

    Returns
    -------
    Path
        Ruta del ``.npz`` con ``w``, ``z``, ``n`` y ``pasada_de_parada``.
    """
    return CKPT_DIR / f"fold{int(fold)}_{brazo}.npz"


def como_nativo(valor: object) -> object:
    """Convierte recursivamente a escalares nativos de JSON (float/int/bool/str/None).

    Los float no finitos se guardan como ``None`` para que el JSON siga siendo valido y las
    claves de fold se fuerzan a texto.

    Parameters
    ----------
    valor:
        Estructura arbitraria (dict, lista, tupla, ndarray o escalar).

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
    """Lee el JSON congelado del pre-registro V35-E1.

    Returns
    -------
    dict
        Contenido completo de ``experiments/v35_00_preregistro.json``.

    Raises
    ------
    FileNotFoundError
        Si el pre-registro no existe.
    """
    if not PREREG_JSON.exists():
        raise FileNotFoundError(f"no existe el pre-registro {PREREG_JSON}")
    return json.loads(PREREG_JSON.read_text(encoding="utf-8"))


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


def d_norm_de_prior(prior: np.ndarray) -> float:
    """``d_norm = sqrt(1 - sum_c prior_c^2)`` del prior MEDIDO de un subconjunto de train.

    Es la magnitud que aparece en la regla DERIVADA de alpha; se calcula SIEMPRE en runtime con
    las filas que se entrenan (el ES-train del fold), nunca con el prior del fold-train completo
    (el ES consume el 10 %) ni con un prior declarado a mano.

    Parameters
    ----------
    prior:
        Vector de 6 frecuencias relativas que suman 1.

    Returns
    -------
    float
        Valor adimensional de d_norm en (0, 1].

    Raises
    ------
    Detencion
        Si el prior no tiene 6 componentes positivas (codigo 2).
    """
    vector = np.asarray(prior, dtype=np.float64)
    if vector.shape != (N_CLASES,) or bool((vector <= 0.0).any()):
        raise Detencion("[STOP] el prior medido no tiene 6 componentes positivas (exit 2)", 2)
    return float(math.sqrt(1.0 - float(np.sum(vector * vector))))


def entropia_de_prior(prior: np.ndarray) -> float:
    """``H(prior) = -sum_c prior_c * log(prior_c)``: techo medido de la banda G1.

    Parameters
    ----------
    prior:
        Vector de 6 frecuencias relativas que suman 1.

    Returns
    -------
    float
        Entropia del prior en nats.
    """
    vector = np.asarray(prior, dtype=np.float64)
    return float(-np.sum(vector * np.log(vector)))


def alpha_derivado(prior: np.ndarray) -> float:
    """Alpha DERIVADO de A1 en su forma especializada: ``2*M/(d_norm*P_max**2)``.

    No es un grid ni un valor buscado: la regla esta congelada y ``d_norm`` se mide en runtime
    con el prior del ES-train del fold, de modo que alpha varia con la composicion del fold por
    construccion y no por busqueda. La forma general del pre-registro
    (``M*(2-power_t)/(d_norm*P_max**(2-power_t))``) se calcula aparte en la sonda (g) y las dos
    rutas deben coincidir BIT a BIT.

    Parameters
    ----------
    prior:
        Prior medido del ES-train del fold (6 frecuencias).

    Returns
    -------
    float
        Alpha efectivo del fold.
    """
    return (2.0 * M_MARGEN) / (d_norm_de_prior(prior) * (P_MAX ** 2))


def alpha_derivado_general(prior: np.ndarray) -> float:
    """Alpha DERIVADO de A1 en su forma GENERAL: ``M*(2-power_t)/(d_norm*P_max**(2-power_t))``.

    Es la segunda ruta independiente del mismo numero: con ``power_t = 0`` se reduce a
    ``2*M/(d_norm*P_max^2)``. La sonda (g) exige igualdad EXACTA entre las dos rutas.

    Parameters
    ----------
    prior:
        Prior medido del ES-train del fold (6 frecuencias).

    Returns
    -------
    float
        Alpha efectivo del fold por la forma general.
    """
    return (M_MARGEN * (2.0 - POWER_T)) / (d_norm_de_prior(prior) *
                                           (P_MAX ** (2.0 - POWER_T)))


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
    matriz = np.clip(p, L2_CLIP, 1.0)
    return matriz / matriz.sum(axis=1, keepdims=True)


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
    matriz = np.clip(p, L2_CLIP, 1.0)
    norma = np.linalg.norm(matriz, axis=1, keepdims=True)
    return matriz / np.where(norma > L2_CLIP, norma, 1.0)


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

    Es la magnitud de las puertas G0/G1/G2: ``-media_i log p[i, y[i]]`` con clip [1e-12, 1].
    A diferencia del log-loss OvR de D-083, normaliza por construccion (softmax de los 6 logits),
    y la sonda (e) la contrasta contra ``sklearn.metrics.log_loss``.

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
    renormaliza por dentro; sin este paso devolveria una perdida calculada sobre valores que no
    son probabilidades. La renormalizacion es la misma que aplica el combinador P6/P7.

    Parameters
    ----------
    p:
        Matriz de puntuaciones o probabilidades (n, 6).
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
    """Maximo de ``|sum_c p_c - 1|`` por fila (magnitud de la puerta G6).

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


def softmax(logits: np.ndarray) -> np.ndarray:
    """Softmax multiclase numericamente estable sobre la ultima dimension.

    Resta el maximo por fila antes de exponenciar: con logit muy grande o muy pequena la
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
    valores = np.asarray(logits, dtype=np.float64)
    desplazado = valores - valores.max(axis=-1, keepdims=True)
    exponenciales = np.exp(desplazado)
    return exponenciales / exponenciales.sum(axis=-1, keepdims=True)


def logits_a_prob(logits: np.ndarray, y: np.ndarray, lote: int = 20000) -> tuple[float, float]:
    """log_loss multiclase y rango medio de logits de un conjunto, en lotes.

    Calcula las dos magnitudes de la puerta sobre las filas de train que se entrenan: el
    ``ll_train`` sobre las probabilidades softmax y el ``D_acum`` (rango medio de logits
    ``max - min``) sobre los logits crudos. Se procesa en lotes para no materializar mas de lo
    necesario con la matriz dispersa grande.

    Parameters
    ----------
    logits:
        Logits del modelo sobre las filas de medida (n, 6).
    y:
        Etiquetas enteras en 0..5 de esas mismas filas.
    lote:
        Numero de filas por lote.

    Returns
    -------
    tuple[float, float]
        ``ll_train`` y ``D_acum``.
    """
    perdida = 0.0
    rango = 0.0
    total = 0
    for inicio in range(0, logits.shape[0], lote):
        fin = min(inicio + lote, logits.shape[0])
        probabilities = softmax(logits[inicio:fin])
        objetivo = np.clip(probabilities[np.arange(fin - inicio), y[inicio:fin]], L2_CLIP, 1.0)
        perdida += float(-np.sum(np.log(objetivo)))
        rango += float(np.sum(logits[inicio:fin].max(axis=1) - logits[inicio:fin].min(axis=1)))
        total += fin - inicio
    return (perdida / float(total), rango / float(total))


# =====================================================================================
# ---- 3. AUDITORIA AST DE COMPLIANCE DEL PROPIO SCRIPT Y DE SUS LECTURAS DE DATOS ----
# =====================================================================================
def es_literal_de_ruta(texto: str) -> bool:
    """Indica si un literal de cadena tiene forma de ruta de fichero.

    Parameters
    ----------
    texto:
        Literal de cadena.

    El predicado es deliberadamente mas estricto que "contiene barra": el texto del informe y de
    los docstrings usa "/" con otros fines (`[M/2, 2M]`, `alpha/t^power_t`, `/5 folds`) y contarlo
    como ruta producia falsos positivos que no dicen nada de la auditoria. Se exige, ademas de la
    barra, que el literal no tenga espacios ni saltos de linea y que su primer segmento sea un
    identificador de directorio plausible (dos o mas caracteres, letra o guion bajo al inicio,
    solo ASCII). Un literal con extension de datos cuenta siempre, tenga barra o no.

    Returns
    -------
    bool
        True si el literal tiene forma de ruta de fichero.
    """
    limpio = texto.strip()
    if not limpio or any(caracter in limpio for caracter in (" ", "\n", "\t", "|")):
        return False
    if limpio.endswith("/"):
        # Prefijo de directorio usado para componer nombres, no un fichero: la ruta real siempre
        # se compone despues y esa si se comprueba.
        return False
    if limpio.startswith(".") and "/" not in limpio:
        # Fragmento de extension (".npy", ".npz"), no una ruta.
        return False
    if limpio.lower().endswith(EXTENSIONES_RUTA):
        return True
    if "/" not in limpio:
        return False
    primero = limpio.split("/", 1)[0]
    return (len(primero) >= 2 and limpio.isascii()
            and (primero[0].isalpha() or primero[0] == "_"))


def literales_de_ruta(arbol: ast.AST) -> list[tuple[int, str]]:
    """Extrae los literales de ruta del arbol de sintaxis con su linea.

    Parameters
    ----------
    arbol:
        Arbol de sintaxis del script.

    Returns
    -------
    list[tuple[int, str]]
        Lista de ``(linea, literal)``.
    """
    encontrados: list[tuple[int, str]] = []
    for nodo in ast.walk(arbol):
        if (isinstance(nodo, ast.Constant) and isinstance(nodo.value, str)
                and es_literal_de_ruta(nodo.value)):
            encontrados.append((int(nodo.lineno), nodo.value))
    return encontrados


def modulos_importados(arbol: ast.AST) -> list[tuple[int, str]]:
    """Extrae los modulos importados del arbol de sintaxis con su linea.

    Parameters
    ----------
    arbol:
        Arbol de sintaxis del script.

    Returns
    -------
    list[tuple[int, str]]
        Lista de ``(linea, modulo)``.
    """
    modulos: list[tuple[int, str]] = []
    for nodo in ast.walk(arbol):
        if isinstance(nodo, ast.Import):
            modulos += [(int(nodo.lineno), alias.name) for alias in nodo.names]
        elif isinstance(nodo, ast.ImportFrom) and nodo.module:
            modulos.append((int(nodo.lineno), nodo.module))
    return modulos


def auditar_ast_propio() -> dict:
    """Audita el propio codigo fuente: 0 palabras vigiladas en literales de ruta y de modulos.

    Un literal se considera de ruta si contiene barra inclinada o termina en una extension de
    datos (.csv, .npy, .json, .npz, .txt, .md, .png). Ninguno puede contener, en minusculas,
    ninguna de las cuatro particiones reservadas ni ninguna de las diez familias vetadas.
    Ademas se revisan los MODULOS IMPORTADOS contra las familias vetadas, de modo que la
    ausencia de las familias vetadas no dependa solo de los literales de ruta.

    Returns
    -------
    dict
        Conteos, listas vigiladas y ``violaciones`` (vacia si todo correcto).

    Raises
    ------
    Detencion
        Si hay algun literal de ruta vigilado o algun import de una familia vetada (codigo 2).
    """
    fuente = Path(__file__).read_text(encoding="utf-8")
    arbol = ast.parse(fuente)
    literales = literales_de_ruta(arbol)
    reservadas = [(linea, texto) for linea, texto in literales
                  if any(palabra in texto.lower() for palabra in PARTICIONES_RESERVADAS)]
    vetadas = [(linea, texto) for linea, texto in literales
               if any(familia in texto.lower() for familia in FAMILIAS_VETADAS)]
    modulos = modulos_importados(arbol)
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


def auditar_lecturas_de_datos() -> dict:
    """Audita por AST que el script solo LEA lo permitido y solo ESCRIBA lo declarado.

    Es una auditoria de LITERALES DECLARADOS, no un sandbox de llamadas al sistema: se comprueba
    que ningun literal de ruta del script apunta fuera de la lista de lecturas permitidas
    (el archivo del corpus y las 4 particiones reservadas de la particion canonica) ni de la lista
    de escrituras declaradas (el directorio de OOF de este ciclo, el de checkpoints, y las rutas
    de ``experiments/`` y ``reports/``). Cada funcion se clasifica por rol (lectora, escritora o
    mixta) y sus literales se comprueban contra la lista que le corresponde; las funciones mixtas
    se comprueban contra la union.

    Returns
    -------
    dict
        Conteos por funcion, literales revisados, lista de lecturas, lista de escrituras y
        ``violaciones`` (vacia si todo correcto).

    Raises
    ------
    Detencion
        Si algun literal de lectura/escritura cae fuera de las listas permitidas (codigo 2).
    """
    fuente = Path(__file__).read_text(encoding="utf-8")
    arbol = ast.parse(fuente)
    literales = literales_de_ruta(arbol)
    funciones: list[tuple[str, list[tuple[int, str]]]] = []
    for nodo in ast.walk(arbol):
        if isinstance(nodo, (ast.FunctionDef, ast.AsyncFunctionDef)):
            propios = [(int(interno.lineno), interno.value)
                       for interno in ast.walk(nodo)
                       if isinstance(interno, ast.Constant)
                       and isinstance(interno.value, str)
                       and es_literal_de_ruta(interno.value)]
            funciones.append((nodo.name, propios))
    permitidas_union = RUTAS_LECTURA_PERMITIDAS + RUTAS_ESCRITURA_PERMITIDAS
    violaciones: list[dict] = []
    revisadas = 0
    detalle: dict[str, dict] = {}
    # El I/O real de este script cuelga de constantes de nivel de modulo (CANONICO, OOF_DIR,
    # CKPT_DIR, RUTAS_HASHES, RUTAS_HASH_CORPUS), luego esos literales se auditan tambien: sin
    # esta pasada, una ruta externa declarada como constante de modulo pasaria desapercibida.
    lineas_en_funcion = {int(interno.lineno)
                         for nodo in ast.walk(arbol)
                         if isinstance(nodo, (ast.FunctionDef, ast.AsyncFunctionDef))
                         for interno in ast.walk(nodo)
                         if isinstance(interno, ast.Constant)}
    literales_modulo = [(linea, texto) for linea, texto in literales_de_ruta(arbol)
                        if linea not in lineas_en_funcion]
    # A nivel de modulo, un literal SIN barra es un NOMBRE que luego se compone con una raiz
    # declarada (`CANONICO / "fold_idx.npy"`), asi que se admite: lo que se vigila es que ningun
    # literal de modulo apunte a una ruta externa o prohibida. Los literales CON barra (relativos o
    # absolutos) si se comprueban contra las listas.
    for linea, texto in literales_modulo:
        revisadas += 1
        if "/" not in texto:
            continue
        if not any(texto.startswith(prefijo) or prefijo in texto
                   for prefijo in permitidas_union):
            violaciones.append({"funcion": "<modulo>", "linea": linea, "literal": texto,
                                "rol": "modulo"})
    # Comprobacion en runtime: el corpus se carga desde cfg.TRAIN_FILE y debe ser el declarado.
    if Path(cfg.TRAIN_FILE).resolve() != (ROOT / "data" / "raw" / "age_train.csv").resolve():
        violaciones.append({"funcion": "<runtime>", "linea": 0,
                            "literal": str(Path(cfg.TRAIN_FILE).resolve()),
                            "rol": "corpus"})
    for nombre, propios in funciones:
        if nombre in FUNCIONES_LECTORAS:
            admitidas, rol = RUTAS_LECTURA_PERMITIDAS, "lectora"
        elif nombre in FUNCIONES_ESCRIPTORAS:
            admitidas, rol = RUTAS_ESCRITURA_PERMITIDAS, "escritora"
        else:
            admitidas, rol = permitidas_union, "mixta"
        revisadas += len(propios)
        detalle[nombre] = {"rol": rol, "literales": [texto for _, texto in propios]}
        nombres_de_rol = {Path(prefijo).name for prefijo in admitidas}
        for linea, texto in propios:
            if "/" not in texto and texto in nombres_de_rol:
                continue
            if not any(texto.startswith(prefijo) or prefijo in texto for prefijo in admitidas):
                violaciones.append({"funcion": nombre, "linea": linea, "literal": texto, "rol": rol})
    for violacion in violaciones:
        print(f"  [STOP] lectura/escritura no declarada en {violacion['funcion']} "
              f"(linea {violacion['linea']}): {violacion['literal']}", flush=True)
    if violaciones:
        raise Detencion("[STOP] auditoria de lecturas de datos fallida (exit 2)", 2)
    return {"lecturas_permitidas": list(RUTAS_LECTURA_PERMITIDAS),
            "escrituras_permitidas": list(RUTAS_ESCRITURA_PERMITIDAS),
            "funciones_lectoras": list(FUNCIONES_LECTORAS),
            "funciones_escriptoras": list(FUNCIONES_ESCRIPTORAS),
            "funciones_revisadas": len(funciones),
            "literales_en_funciones_revisados": revisadas,
            "literales_de_ruta_totales": len(literales),
            "roles_por_funcion": {nombre: detalle[nombre]["rol"] for nombre in sorted(detalle)},
            "literales_de_modulo_revisados": len(literales_modulo),
            "corpus_en_runtime": str(Path(cfg.TRAIN_FILE).resolve()),
            "violaciones": []}


# =====================================================================================
# ---- 4. NUCLEO FTRL-PROXIMAL SOFTMAX MULTICLASE MULTI-PASADA (PROPIO) ----
# =====================================================================================
class FtrlSoftmaxMulticlase:
    """Nucleo FTRL-Proximal softmax MULTICLASE multi-pasada, propio y exacto.

    K = 6 clases en UN SOLO juego de pesos: para cada ejemplo se calcula ``s = X @ W`` con el
    sesgo materializado como coordenada de valor ``x0 = 1.0`` (la ultima columna del CSR), se
    obtiene ``p = softmax(s)`` y el gradiente multiclase es ``g_c = p_c - y_c``. Solo se
    actualizan las coordenadas NO NULAS de la fila, que es la regla proximal congelada:
    ``sigma = (sqrt(n+g^2) - sqrt(n))/(alpha_t*beta)``; ``z += g - sigma*w``; ``n += g^2``;
    ``w = 0 si |z| <= lambda1``, si no ``sign(z)*(|z|-lambda1)/((beta+sqrt(n))/alpha_t + lambda2)``.

    La aritmetica de este nucleo es LITERALMENTE la de ``v33_01_ftrl_softmax_online.py`` (mismo
    orden de operaciones, mismos ``einsum`` y mismos ``where``), que a su vez fue verificada con
    una tercera implementacion independiente (max|dw| = 2.4e-15) y con las sondas (a), (b) y (e).
    La unica diferencia es que el estado (``w``, ``z``, ``n``, ``t``) vive en el objeto y puede
    atravesar varias pasadas, que es lo que exige el presupuesto multi-pasada con early stopping.

    Attributes
    ----------
    n_coordenadas:
        Numero de coordenadas del problema (incluida la del sesgo).
    alpha, lambda1, lambda2, beta, power_t:
        Hiperparametros congelados; ``alpha`` es el DERIVADO de la regla de margen.
    w, z, n:
        Pesos, acumulador de correccion y acumulador de cuadrados, todos (d, 6) float64.
    t:
        Contador global de ejemplos vistos (para el decaimiento ``alpha/t^power_t``).
    acc_sigma, acc_g:
        Acumuladores de ``|sigma*w|`` y ``|g|`` (sumados sobre clases y coordenadas) que dan la
        razon de mecanismo ``|sigma*w|/|g|`` publicada por pasada.
    max_abs_g_sesgo, acc_g_sesgo:
        Traza del gradiente de la coordenada del sesgo: maximo absoluto y acumulado por clase.
    """

    def __init__(self, n_coordenadas: int, alpha: float, lambda1: float = LAMBDA1,
                 lambda2: float = LAMBDA2, beta: float = BETA, power_t: float = POWER_T
                 ) -> None:
        self.n_coordenadas = int(n_coordenadas)
        self.alpha = float(alpha)
        self.lambda1 = float(lambda1)
        self.lambda2 = float(lambda2)
        self.beta = float(beta)
        self.power_t = float(power_t)
        self.w = np.zeros((self.n_coordenadas, N_CLASES), dtype=np.float64)
        self.z = np.zeros((self.n_coordenadas, N_CLASES), dtype=np.float64)
        self.n = np.zeros((self.n_coordenadas, N_CLASES), dtype=np.float64)
        self.t = 0
        self.acc_sigma = 0.0
        self.acc_g = 0.0
        self.max_abs_g_sesgo = 0.0
        self.acc_g_sesgo = np.zeros(N_CLASES, dtype=np.float64)

    def estado(self) -> dict:
        """Copias del estado interno para checkpoint, sonda analitica o comparacion.

        Returns
        -------
        dict
            ``w``, ``z``, ``n`` (copias float64) y ``t`` (int).
        """
        return {"w": self.w.copy(), "z": self.z.copy(), "n": self.n.copy(), "t": int(self.t)}

    def cargar(self, w: np.ndarray, z: np.ndarray, n: np.ndarray, t: int) -> None:
        """Restaura un estado completo (pesos, acumuladores y contador) en el objeto.

        Se usa para restaurar el mejor estado tras el early stopping y para reanudar desde un
        checkpoint.

        Parameters
        ----------
        w:
            Pesos (d, 6) float64.
        z:
            Acumulador de correccion (d, 6) float64.
        n:
            Acumulador de cuadrados (d, 6) float64.
        t:
            Contador de ejemplos vistos.

        Raises
        ------
        Detencion
            Si alguna matriz no tiene la forma (n_coordenadas, 6) (codigo 2).
        """
        for nombre, matriz in (("w", w), ("z", z), ("n", n)):
            if matriz.shape != (self.n_coordenadas, N_CLASES):
                raise Detencion(f"[STOP] la matriz {nombre} del checkpoint no tiene forma "
                                f"({self.n_coordenadas}, {N_CLASES}) (exit 2)", 2)
        self.w = np.array(w, dtype=np.float64, copy=True)
        self.z = np.array(z, dtype=np.float64, copy=True)
        self.n = np.array(n, dtype=np.float64, copy=True)
        self.t = int(t)

    def alfa_pasada(self) -> float:
        """Alpha efectivo del instante actual: ``alpha/t^power_t`` si ``power_t > 0``.

        Returns
        -------
        float
            Alpha efectivo con el que se actualiza el siguiente ejemplo.
        """
        if self.power_t > 0.0:
            if self.t <= 0:
                raise Detencion("[STOP] power_t > 0 exige t > 0 para el decaimiento (exit 2)", 2)
            return self.alpha / (float(self.t) ** self.power_t)
        return self.alpha

    def pasar(self, x: sp.csr_matrix, y: np.ndarray, orden: np.ndarray) -> dict:
        """Recorre UNA pasada online completa en el orden indicado y actualiza el estado.

        El recorrido es secuencial por ejemplo (FTRL es un metodo online) y la actualizacion se
        vectoriza sobre las clases y sobre las coordenadas no nulas de la fila. No hay
        aleatoriedad dentro del lazo: el unico orden es ``orden``.

        Parameters
        ----------
        x:
            Matriz CSR float64 con la coordenada del sesgo ya anadida (ultima columna, x0).
        y:
            Etiquetas enteras en 0..5 (n,), en el orden natural de las filas de ``x``.
        orden:
            Permutacion de ``range(x.shape[0])``: el unico recorrido de la pasada.

        Returns
        -------
        dict
            ``n_actualizaciones``, ``perdida_online`` (media del stream), ``razon_sigma_sobre_g``
            (acumulada), ``razon_sigma_sobre_g_pasada`` (solo esta pasada), ``w_max_abs``,
            ``max_abs_g_sesgo``, ``g_sesgo_acumulado`` y ``t``.

        Raises
        ------
        Detencion
            Si el recorrido no es una permutacion valida de las filas (codigo 2).
        """
        n_filas = x.shape[0]
        if orden.shape[0] != n_filas or not np.array_equal(np.sort(orden), np.arange(n_filas)):
            raise Detencion("[STOP] el orden congelado no es una permutacion de las filas (exit 2)",
                            2)
        alpha_pasada = self.alfa_pasada()
        indice_sesgo = self.n_coordenadas - 1
        sigma_previo = self.acc_sigma
        g_previo = self.acc_g
        perdida = 0.0
        for i in orden:
            inicio, fin = int(x.indptr[i]), int(x.indptr[i + 1])
            indices = x.indices[inicio:fin]
            valores = x.data[inicio:fin]
            s = np.einsum("ij,i->j", self.w[indices], valores)
            p = softmax(s)
            g = p.copy()
            g[y[i]] -= 1.0
            g_coord = g[None, :] * valores[:, None]
            n_previo = self.n[indices]
            n_nuevo = n_previo + g_coord * g_coord
            sigma = (np.sqrt(n_nuevo) - np.sqrt(n_previo)) / (alpha_pasada * self.beta)
            w_previo = self.w[indices]
            self.acc_sigma += float(np.sum(np.abs(sigma * w_previo)))
            self.acc_g += float(np.sum(np.abs(g_coord)))
            self.z[indices] = self.z[indices] + g_coord - sigma * w_previo
            self.n[indices] = n_nuevo
            z_nuevo = self.z[indices]
            self.w[indices] = np.where(
                np.abs(z_nuevo) <= self.lambda1, 0.0,
                np.sign(z_nuevo) * (np.abs(z_nuevo) - self.lambda1)
                / ((self.beta + np.sqrt(n_nuevo)) / alpha_pasada + self.lambda2))
            self.t += 1
            perdida += -math.log(max(float(p[y[i]]), L2_CLIP))
            if bool((indices == indice_sesgo).any()):
                self.max_abs_g_sesgo = max(self.max_abs_g_sesgo, float(np.max(np.abs(g))))
                self.acc_g_sesgo += np.abs(g)
        return {"n_actualizaciones": int(n_filas), "perdida_online": float(perdida / float(n_filas)),
                "razon_sigma_sobre_g": float(self.acc_sigma / self.acc_g)
                if self.acc_g > 0 else 0.0,
                "razon_sigma_sobre_g_pasada": float((self.acc_sigma - sigma_previo)
                                                    / (self.acc_g - g_previo))
                if (self.acc_g - g_previo) > 0 else 0.0,
                "w_max_abs": float(np.max(np.abs(self.w))),
                "max_abs_g_sesgo": float(self.max_abs_g_sesgo),
                "g_sesgo_acumulado": [float(valor) for valor in self.acc_g_sesgo],
                "t": int(self.t)}

    def coord_sesgo(self) -> np.ndarray:
        """Vector de sesgo efectivo (6,) leyendo la coordenada del sesgo de ``w``.

        Como ``x0 = 1.0``, la ultima coordenada de ``w`` ES el termino ``b`` de ``X @ W + b``:
        no hay un segundo camino de sesgo en A1.

        Returns
        -------
        numpy.ndarray
            Los 6 valores de sesgo aprendidos.
        """
        if self.n_coordenadas <= 0:
            raise Detencion("[STOP] el numero de coordenadas es no positivo (exit 2)", 2)
        return self.w[self.n_coordenadas - 1].copy()

    def sha256_pesos(self) -> str:
        """SHA256 de los pesos actuales (identidad exacta de la corrida, tolerancia 0).

        Returns
        -------
        str
            Digest hexadecimal de 64 caracteres.
        """
        return sha256_f64_2d(self.w)


def ftrl_softmax_multipasada(x: sp.csr_matrix, y: np.ndarray, alpha: float,
                             ordenes: list[np.ndarray], lambda1: float = LAMBDA1,
                             lambda2: float = LAMBDA2, beta: float = BETA,
                             power_t: float = POWER_T) -> np.ndarray:
    """Ajuste FTRL-Proximal softmax multi-pasada completo (envoltorio del nucleo por clases).

    Es la forma que usan las sondas: recorre la lista completa de ordenes y devuelve los pesos
    finales. El recorrido por fold del runner usa la clase directamente, porque necesita medir
    magnitudes y aplicar el early stopping en CADA pasada.

    Parameters
    ----------
    x:
        Matriz CSR float64 con la coordenada del sesgo ya anadida.
    y:
        Etiquetas enteras en 0..5 (n,).
    alpha:
        Alpha efectivo (derivado de la regla de margen).
    ordenes:
        Una permutacion de las filas por pasada.
    lambda1, lambda2, beta, power_t:
        Hiperparametros congelados.

    Returns
    -------
    numpy.ndarray
        Pesos (d, 6) float64 al terminar la ultima pasada.
    """
    nucleo = FtrlSoftmaxMulticlase(x.shape[1], alpha, lambda1, lambda2, beta, power_t)
    for orden in ordenes:
        nucleo.pasar(x, y, np.asarray(orden, dtype=np.int64))
    return nucleo.w.copy()


def ftrl_softmax_lazo_ingenuo(x_denso: np.ndarray, y: np.ndarray, alpha: float,
                              ordenes: list[np.ndarray], lambda1: float = LAMBDA1,
                              lambda2: float = LAMBDA2, beta: float = BETA,
                              power_t: float = POWER_T) -> np.ndarray:
    """Lazo ingenuo MULTI-PASADA de FTRL softmax (listas de Python, sin numpy en el interior).

    Es la referencia independiente de la sonda (a): misma actualizacion, mismo softmax y las
    MISMAS permutaciones, pero recorriendo una coordenada y una clase por vez con escalares de
    Python. Al ser multi-pasada, comprueba tambien que el estado ``z``, ``n`` y el contador ``t``
    atraviesan las pasadas sin alterarse.

    Parameters
    ----------
    x_denso:
        Matriz densa float64 (n, d) con la coordenada del sesgo incluida.
    y:
        Etiquetas enteras en 0..5 (n,).
    alpha:
        Alpha base efectivo.
    ordenes:
        Una permutacion de las filas por pasada.
    lambda1, lambda2, beta, power_t:
        Hiperparametros congelados.

    Returns
    -------
    numpy.ndarray
        Matriz de pesos (d, 6) al terminar la ultima pasada.
    """
    n_coordenadas = x_denso.shape[1]
    w = [[0.0] * N_CLASES for _ in range(n_coordenadas)]
    z = [[0.0] * N_CLASES for _ in range(n_coordenadas)]
    n = [[0.0] * N_CLASES for _ in range(n_coordenadas)]
    t = 0
    for orden in ordenes:
        for i in orden:
            t += 1
            alpha_t = alpha / (float(t) ** power_t) if power_t > 0.0 else alpha
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
                    sigma = (raiz - raiz_previa) / (alpha_t * beta)
                    z[j][c] += g - sigma * w[j][c]
                    n[j][c] += g * g
                    w[j][c] = (0.0 if abs(z[j][c]) <= lambda1 else
                               math.copysign(1.0, z[j][c]) * (abs(z[j][c]) - lambda1)
                               / ((beta + math.sqrt(n[j][c])) / alpha_t + lambda2))
    return np.array(w, dtype=np.float64)


def problema_aleatorio(n_filas: int, n_features: int, semilla: int
                       ) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray]:
    """Problema aleatorio denso (n, d+1) con sesgo, empaquetado en CSR canonico.

    La ultima columna es la coordenada del sesgo con valor ``x0``; las primeras ``n_features``
    son continuas en [0, 1). Se usa en las sondas (a), (b), (d), (e) y (f), y en el problema
    mini de las sondas (h) y (j).

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
    tuple[scipy.sparse.csr_matrix, numpy.ndarray, numpy.ndarray]
        Matriz CSR con sesgo, etiquetas en 0..5 y una copia densa de la matriz.
    """
    generador = np.random.default_rng(semilla)
    denso = generador.random((n_filas, n_features + 1))
    denso[:, -1] = X0
    y = generador.integers(0, N_CLASES, size=n_filas)
    return sp.csr_matrix(denso), y.astype(np.int64), denso


# =====================================================================================
# ---- 5. VERIFICACION DE NO-DERIVA CON EL PRE-REGISTRO, VERSIONES Y HASHES ----
# =====================================================================================
def verificar_constantes(prereg: dict) -> tuple[bool, dict]:
    """Compara cada constante local contra el JSON congelado de V35, una por una.

    La fuente unica de verdad es ``experiments/v35_00_preregistro.json``: se comprueban la
    identidad de la fase, el baseline y los pesos, las versiones, el juego de hashes (los seis
    canonicos, el manifiesto, el nucleo de V34, el corpus, el vectorizador y los dos ficheros de la
    sonda pre-ejecucion), el anclaje de ``lambda1`` (mediana congelada, sha256 del snapshot de z y
    prohibicion de re-derivar), ``alpha*`` congelado, el presupuesto con ES interno, los umbrales
    de la puerta G0-G8 leidos de ``puerta_g0_g6_heredada`` mas las dos gates nuevas con su
    tolerancia declarada, la rama pre-declarada del arbol, las sondas (k), (l) y (m) con su
    politica NO-GATE, el compliance con sus once auditorias, la asimetria A1/A2 como control de
    PRESUPUESTO, la rama terminal one-shot y las rutas de entregables. Si el protocolo cambia, el
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
    config = prereg["config"]
    puerta = prereg["puerta"]
    compuerta = prereg["compliance"]
    entregable = prereg["entregables"]["runner"]
    heredada = prereg["puerta_g0_g6_heredada"]
    anclaje = prereg["anclaje_lambda1"]
    serie = prereg["gates_de_la_serie"]
    rama = puerta["rama_predeclarada"]
    alfa = prereg["alpha_congelado"]
    previa = prereg["sonda_previa"]
    checks = {
        # ---- identidad, baseline, pesos y versiones ----
        "fase": prereg["fase"] == "V35-E1 (via B) pre-registro congelado",
        "fecha": prereg["fecha"] == FECHA,
        "baseline_primario": float(prereg["baseline_primario"]) == BASELINE,
        "referencia_nested": float(prereg["referencia_nested"]) == NESTED,
        "pesos_plugin": [float(valor) for valor in prereg["pesos_plugin"]] == PESOS,
        "versiones_claves": set(prereg["versiones"]) == {"sklearn", "numpy", "pandas"},
        "hashes_claves": set(prereg["hashes"]) == set(RUTAS_HASHES) | set(CLAVES_HASH_CORPUS),
        "hashes_long_64": all(len(str(valor)) == 64 for valor in prereg["hashes"].values()),
        "hilos_del_prereg": int(config["hilos"]) == HILOS,
        "semilla_del_prereg": int(config["semilla"]) == SEMILLA,
        # ---- anclaje de lambda1: un unico numero, congelado y verificado ----
        "lambda1_mediana_congelada": float(anclaje["valor_esperado"]) == MEDIANA_Z_CONGELADA,
        "lambda1_es_la_mediana": (LAMBDA1 == MEDIANA_Z_CONGELADA
                                  and float(anclaje["valor_esperado"]) == LAMBDA1),
        "lambda1_definicion_medida": ("mediana(|z_j| > 0)" in anclaje["definicion"]
                                      and "cero busqueda" in anclaje["definicion"]),
        "lambda1_medida_sobre_es_train": ("ES-train del fold 0" in anclaje["medida_sobre"]
                                          and "sonda pre-ejecucion" in anclaje["medida_sobre"]),
        "lambda1_flag_1_resuelto": bool(anclaje["flag_1_resuelto"]),
        "lambda1_no_se_re_deriva": ("re-derivar lambda1" in anclaje["prohibido"]
                                    and "grid" in anclaje["prohibido"]),
        "lambda1_sensibilidad_declarada": ("q=0.25" in anclaje["sensibilidad_declarada"]
                                            and "gemelo sintetico" in
                                            anclaje["sensibilidad_declarada"]),
        "lambda1_sha256_z_congelado": (previa["anclaje_medio"] == MEDIANA_Z_CONGELADA
                                       and len(SHA256_Z_ANCLAJE) == 64),
        "lambda1_razon_normas_previa": abs(float(previa["razon_normas_lambda1_mediana"])
                                           - 2.130435582526498) == 0.0,
        "lambda1_mediana_sobre_alpha": abs(float(previa["mediana_sobre_alpha"])
                                           - 6.804368702629027) < 1e-12,
        # ---- alpha* congelado: no se re-deriva en este ciclo ----
        "alpha_valor_esperado": float(alfa["valor_esperado"]) == 0.035262336465737745,
        "alpha_formula": alfa["formula"] == "2*M/(d_norm*P_max**2) con M = ln(5)",
        "alpha_no_se_re_deriva": bool(alfa["no_se_re_deriva"]),
        "alpha_unchanged_desde_v34": abs(float(alfa["valor_esperado"])
                                         - ANCLA_ALPHA_PARA_D_NORM_REFERENCIA) < 1e-6,
        "regla_anchos_fija": (alpha_derivado(np.full(N_CLASES, 1.0 / N_CLASES))
                              == alpha_derivado_general(np.full(N_CLASES, 1.0 / N_CLASES))),
        # ---- presupuesto con ES interno ----
        "presupuesto_p_max": int(config["P_max"]) == P_MAX,
        "presupuesto_p_min": int(config["P_min"]) == P_MIN,
        "presupuesto_paciencia": int(config["paciencia"]) == PACIENCIA,
        "presupuesto_mejora": float(config["mejora_minima"]) == MEJORA_MINIMA,
        "presupuesto_particion_es": ("StratifiedShuffleSplit(test_size=0.10, "
                                     "random_state=20261010+fold)" in config["particion_es"]
                                     and "filas de TRAIN" in config["particion_es"]),
        "test_size_es": (TEST_SIZE_ES == 0.10 and float(config["test_size_es"]) == 0.1),
        "ajuste_por_fold_es_train": "POR FOLD" in config["ajuste"],
        "vectorizador_config": config["vectorizador"] == VECTORIZADOR,
        # ---- nucleo FTRL congelado: L1 por primera vez distinto de cero ----
        "nucleo_lambda1_mediana": (LAMBDA1 == MEDIANA_Z_CONGELADA
                                   and MEDIANA_Z_CONGELADA > 0.0),
        "nucleo_lambda2_beta": (LAMBDA2 == 1.0 and float(config["lambda2"]) == LAMBDA2
                                and BETA == 1.0 and float(config["beta"]) == BETA),
        "nucleo_power_t": (POWER_T == 0.0 and float(config["power_t"]) == POWER_T),
        "nucleo_x0": (X0 == 1.0 and float(config["x0"]) == X0),
        "nucleo_variante_l2": VARIANTE_L2 == "denominador",
        "nucleo_sha256_en_prereg": "nucleo_v34" in prereg["hashes"],
        # ---- puerta G0-G6 heredada SIN CAMBIOS ----
        "gates_claves": len(NOMBRES_GATE) == 9 and NOMBRES_GATE[:7] == GATES_EXCLUSIVOS_A2,
        "puerta_momento": ("dentro del fold 0" in puerta["momento"]
                           and "ANTES de integrar" in puerta["momento"]),
        "puerta_g0_g6_sin_cambios": ("SIN CAMBIOS" in puerta["G0_G6"]
                                    and "reports/v34_convergence_gate.md" in puerta["G0_G6"]),
        "heredada_G0": (heredada["G0"] == "A1: M/2 <= D_acum <= 2*M; "
                                         "A2: ll_train <= H(prior)_ES - 0.01"
                         and G0_FACTOR_INFERIOR == 0.5 and G0_FACTOR_SUPERIOR == 2.0),
        "heredada_G1": (heredada["G1"] == "0.05 <= ll_train <= H(prior)_ES - 0.01"
                        and G1_PISO == 0.05 and G1_TECHO_MARGEN == 0.01),
        "heredada_G2": ("mejora en >= 1 de las primeras 3 pasadas" in heredada["G2"]
                        and G2_TOLERANCIA == 1e-06),
        "heredada_G3": ("macro-F1 standalone en fold-val >= 0.3" in heredada["G3"]
                        and G3_MINIMO == 0.30),
        "heredada_G4": ("[0.5*prior_c, 1.5*prior_c]" in heredada["G4"]
                        and "MEDIDO" in heredada["G4"]
                        and G4_FACTOR_INFERIOR == 0.5 and G4_FACTOR_SUPERIOR == 1.5),
        "heredada_G5": ("determinismo max|delta p| = 0.0 EXACTO" in heredada["G5"]
                        and G5_TOLERANCIA == 0.0),
        "heredada_G6": ("filas que suman 1 <= 1e-09" in heredada["G6"]
                        and G6_TOLERANCIA == 1e-09),
        # ---- G7 y G8: nuevas, solo A1 ----
        "g7_criterio": ("al menos una clase viva" in puerta["G7_L1_activo"]["criterio"].lower()
                        or "AL MENOS una clase viva" in puerta["G7_L1_activo"]["criterio"]),
        "g7_tolerancia": ("tolerancia declarada de 0.02"
                          in puerta["G7_L1_activo"]["criterio"]
                          and G7_TOLERANCIA_DEGRADACION == 0.02),
        "g7_anclaje_medida": ("magnitud de pasada 1 medida"
                              in puerta["G7_L1_activo"]["anclaje"]
                              and "tautologico" in puerta["G7_L1_activo"]["anclaje"]),
        "g7_criterio_muerto": ("fraccion de coordenadas" in
                               puerta["G7_L1_activo"]["criterio_muerto"]
                               and "analiticamente muerta" in
                               puerta["G7_L1_activo"]["criterio_muerto"]),
        "g7_aplica_a": puerta["G7_L1_activo"]["aplica_a"] == "solo A1",
        "g8_criterio": ("razon max/min de ||w||_2" in puerta["G8_acotacion_norma"]["criterio"]
                        and "<= 1.0" in puerta["G8_acotacion_norma"]["criterio"]
                        and G8_RAZON_MAXIMA == 1.0),
        "g8_anclaje_hipotesis": ("HIPOTESIS misma" in puerta["G8_acotacion_norma"]["anclaje"]
                                 and "2.1304" in puerta["G8_acotacion_norma"]["anclaje"]),
        "g8_aplica_a": puerta["G8_acotacion_norma"]["aplica_a"] == "solo A1",
        "gates_exclusivos_de_a1": GATES_EXCLUSIVOS_A1 == (NOMBRES_GATE[7], NOMBRES_GATE[8]),
        # ---- rama pre-declarada del arbol ----
        "rama_g7_o_g8": ("LINEA_ONLINE_CERRADA_POR_MECANISMO" in rama["G7_o_G8_fallan"]
                         and "DEFINITIVAMENTE" in rama["G7_o_G8_fallan"]
                         and "segundo lambda1" in rama["G7_o_G8_fallan"]),
        "rama_conjunta": ("HIPOTESIS_CONJUNTA_REFUTADA" in rama["G7_y_G8_pasan_pero_G0_G4_fallan"]
                          and "acota pero no aprende" in
                          rama["G7_y_G8_pasan_pero_G0_G4_fallan"]),
        "rama_todas_pasan": ("gates de la serie" in rama["todas_pasan"]
                             and "autorizacion del usuario" in rama["todas_pasan"]),
        # ---- sondas (k), (l) y (m): NO-GATE, se publican dentro del runner ----
        "sondas_k_l_m_claves": set(prereg["sondas_nuevas"]) == {"k_caso_lambda1_cero",
                                                                "l_caso_lambda1_alpha",
                                                                "m_caso_lambda1_mediana",
                                                                "politica"},
        "sondas_politica_no_gate": ("NO-GATE" in prereg["sondas_nuevas"]["politica"]
                                    and "/tmp" in prereg["sondas_nuevas"]["politica"]),
        "sonda_l_razon": "6.8044" in prereg["sondas_nuevas"]["l_caso_lambda1_alpha"],
        # ---- asimetria: A2 es control de PRESUPUESTO, no gemelo mecanico ----
        "asimetria_control_presupuesto": ("control de PRESUPUESTO" in puerta["asimetria"]
                                          and "no gemelo mecanico" in puerta["asimetria"]),
        "asimetria_lectura_unidireccional": ("Ninguna conclusion puede descansar" in
                                             puerta["asimetria"]
                                             and "contaminados" in puerta["asimetria"]),
        "asimetria_apoyo_del_veredicto": "evidencia interna de A1" in puerta["asimetria"],
        # ---- rama terminal one-shot ----
        "terminal_miembro": prereg["rama_terminal"]["miembro_actual"] == "lambda1 > 0",
        "terminal_cerrados": len(prereg["rama_terminal"]["miembros_cerrados_e_intocables"]) == 6,
        "terminal_segundo_miembro": ("pesos por clase con L2 normalizado fuera de FTRL"
                                     in prereg["rama_terminal"]["segundo_miembro_no_cubierto"]),
        "terminal_re_emision": ("una unica re-emision" in prereg["rama_terminal"]["re_emision"]
                                and "DEFECTO_DE_IMPLEMENTACION" in
                                prereg["rama_terminal"]["re_emision"]),
        "miembro_no_cubierto": prereg["miembro_no_cubierto"]["identificador"] == "lambda1 > 0",
        "no_es_reintento_de_alpha": ("alpha* queda CONGELADO" in
                                     prereg["miembro_no_cubierto"]["no_es_reintento_de_alpha"]),
        # ---- gates de la serie ----
        "serie_abort": float(serie["baseline_abort"]) == ABORTO_BASELINE,
        "serie_delta": float(serie["delta_min"]) == GATE_DELTA,
        "serie_folds": int(serie["folds_min"]) == GATE_FOLDS,
        "serie_bootstrap": (int(serie["bootstrap"]["n"]) == N_BOOTSTRAP
                            and int(serie["bootstrap"]["semilla"]) == SEMILLA),
        # ---- coste, RAM, checkpoints y compliance ----
        "ejecucion_presupuesto": (float(prereg["ejecucion"]["presupuesto_declarado_seg"])
                                   == PRESUPUESTO_DECLARADO_SEG
                                   and float(prereg["ejecucion"]["ram_tope_gb"]) == RAM_TOPE_GB),
        "ejecucion_timeout": float(prereg["ejecucion"]["timeout_global_h"]) == TIMEOUT_H,
        "ejecucion_calibracion": ("proyeccion 9 x t_fold0" in
                                  prereg["ejecucion"]["calibracion"]),
        "ejecucion_checkpoints": (ruta_relativa(ESTADO_JSON) in prereg["ejecucion"]["checkpoints"]
                                  and "checkpoints/v35_e1" in prereg["ejecucion"]["checkpoints"]),
        "ejecucion_escritura_atomica": "os.replace" in prereg["ejecucion"]["escritura_atomica"],
        "compliance_cifras": {clave: compuerta[clave] for clave in COMPLIANCE_CIFRAS}
                            == COMPLIANCE_CIFRAS,
        "compliance_ast_audit": ("importes Y literales" in compuerta["ast_audit"]
                                 and "vowpal" in compuerta["ast_audit"]),
        "compliance_lecturas": ("data/raw/age_train.csv" in compuerta["lecturas"]
                                and "p33_scores/todos" in compuerta["lecturas"]
                                and "intocable" in compuerta["lecturas"]),
        "compliance_auditorias": compuerta["auditorias"] == COMPLIANCE_AUDITORIAS,
        # ---- anclas locales heredadas de la particion canonica ----
        "anclas_logaritmicas": (abs(math.log(N_CLASES) - ANCLA_LOG_UNIFORME) <= 1e-15
                                and abs(math.log(N_CLASES - 1) - M_MARGEN) <= 1e-15),
        "ancla_m": M_MARGEN == 1.6094379124341003,
        "hilos_efectivos": all(os.environ[clave] == str(HILOS) for clave in
                               ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")),
        # ---- entregables ----
        "entregables": all(nombre in entregable for nombre in nombres_entregables()),
        "entregables_novelty": "reports/v35_novelty_registry.csv" in entregable,
        "entregables_historia": ("fila anexada in reports/age_complete_experiment_history.csv"
                                 in entregable),
        "entregables_acta": "D-086" in prereg["entregables"]["acta"],
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

    Los seis canonicos y el manifiesto se leen de disco; los seis del corpus y de la exploracion
    previa (archivo del corpus, array de textos con la receta determinista, dict del vectorizador
    y los dos ficheros de la sonda pre-ejecucion, mas el nucleo de V34) se recomputan. El
    NO se usa: quedo invalidado en D-083 (hash de punteros).

    Parameters
    ----------
    prereg:
        Contenido del pre-registro.
    hashes_corpus:
        Hashes recomputados del corpus, del vectorizador y de la sonda pre-ejecucion.

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
    coinciden con los congelados, el runner se detiene con codigo 2. El hash del array de textos
    usa la receta determinista del pre-registro (longitud en 8 bytes little-endian + bytes UTF-8
    por texto, en orden) y NUNCA el ``data_hash`` del manifiesto canonico, invalidado en D-083.

    Parameters
    ----------
    ids_canonicos:
        ``ids.npy`` de la particion canonica (orden de referencia).
    clase:
        ``y_true`` canonico (etiquetas de referencia).

    Returns
    -------
    tuple[numpy.ndarray, dict]
        Textos en el orden del CSV y los hashes del corpus y de la sonda pre-ejecucion.

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
                                           .encode("utf-8"))}
    for nombre, ruta in RUTAS_HASH_CORPUS.items():
        if not ruta.exists():
            raise Detencion(f"[STOP] falta el artefacto de la sonda pre-ejecucion {ruta} (exit 2)",
                            2)
        hashes[nombre] = sha256_archivo(ruta)
    return textos, hashes


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
# ---- 6. SONDAS OBLIGATORIAS (a)-(j), ANTES DE ENTRENAR (exit 4) ----
# =====================================================================================
def sonda_a_vectorizado_vs_lazo() -> dict:
    """Sonda (a): el nucleo FTRL softmax MULTI-PASADA coincide con el lazo ingenuo escalar.

    Problema aleatorio de 200 filas x 50 features (mas la coordenada del sesgo) x 6 clases, misma
    lista de 3 permutaciones sembradas y el MISMO alpha derivado en las dos rutas; se compara el
    maximo de la diferencia absoluta de TODOS los pesos de las 6 clases. Al ser multi-pasada,
    comprueba tambien que ``z``, ``n`` y el contador ``t`` atraviesan las tres pasadas.

    Returns
    -------
    dict
        Dimensiones, numero de pasadas, maximo de la diferencia, tolerancia y ``pass``.
    """
    n_filas, n_features = DIM_SONDA_A
    x, y, denso = problema_aleatorio(n_filas, n_features, SEMILLA_SONDA)
    prior = np.bincount(y, minlength=N_CLASES) / float(n_filas)
    alpha = alpha_derivado(prior)
    ordenes = [np.random.default_rng(SEMILLA + p).permutation(n_filas).astype(np.int64)
               for p in range(PASADAS_SONDA_A)]
    w_vector = ftrl_softmax_multipasada(x, y, alpha, ordenes)
    w_lazo = ftrl_softmax_lazo_ingenuo(denso, y, alpha, ordenes)
    peor = float(np.abs(w_lazo - w_vector).max())
    return {"descripcion": f"FTRL softmax multiclase multi-pasada ({PASADAS_SONDA_A} pasadas) "
                           f"vectorizada vs lazo ingenuo, problema aleatorio {n_filas}x"
                           f"{n_features}x{N_CLASES}",
            "alpha": alpha, "n_pasadas": PASADAS_SONDA_A,
            "max_abs_diff": peor, "tolerancia": TOL_SONDA_A, "pass": bool(peor <= TOL_SONDA_A)}


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
        Residuales de la raiz analitica y del paso unico, umbral y ``pass``.
    """
    alpha = alpha_derivado(np.full(N_CLASES, 1.0 / N_CLASES))
    nucleo = FtrlSoftmaxMulticlase(1, alpha, 0.0, 0.0, BETA, POWER_T)
    constante = sp.csr_matrix(np.ones((25, 1), dtype=np.float64))
    y = np.zeros(25, dtype=np.int64)
    nucleo.pasar(constante, y, np.arange(25, dtype=np.int64))
    estado = nucleo.estado()
    raiz = estado["z"][0] * alpha / (BETA + np.sqrt(estado["n"][0]))
    residuo_raiz = float(np.abs(estado["w"][0] - raiz).max())
    nucleo_uno = FtrlSoftmaxMulticlase(1, alpha, 0.0, 0.0, BETA, POWER_T)
    nucleo_uno.pasar(sp.csr_matrix(np.ones((1, 1))), np.zeros(1, dtype=np.int64),
                     np.zeros(1, dtype=np.int64))
    esperado = alpha * np.array([-5.0 / 6.0] + [1.0 / 6.0] * (N_CLASES - 1)) / (
        BETA + np.array([5.0 / 6.0] + [1.0 / 6.0] * (N_CLASES - 1)))
    residuo_paso = float(np.abs(nucleo_uno.estado()["w"][0] - esperado).max())
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
        TotalF1 default, TotalF1 average=Macro, macro-F1 de sklearn, deltas, tolerancia y ``pass``.
    """
    from catboost.utils import eval_metric as metrica_catboost

    generador = np.random.default_rng(SEMILLA_SONDA_C)
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
    x, y, _denso = problema_aleatorio(n_filas, n_features, SEMILLA_SONDA_D)
    alpha = alpha_derivado(np.bincount(y, minlength=N_CLASES) / float(n_filas))
    w = ftrl_softmax_multipasada(x, y, alpha,
                                 [np.random.default_rng(SEMILLA).permutation(n_filas)])
    p = softmax(x.dot(w))
    valor = log_loss_propio(p, y)
    finita = bool(np.isfinite(p).all() and np.isfinite(w).all() and math.isfinite(valor))
    return {"descripcion": "softmax multiclase sin NaN/inf y log_loss multiclase finito",
            "n_filas": int(x.shape[0]), "p_finita": bool(np.isfinite(p).all()),
            "w_finito": bool(np.isfinite(w).all()), "log_loss_multiclase": valor,
            "log_loss_finito": bool(math.isfinite(valor)), "p_min": float(p.min()),
            "p_max": float(p.max()), "pass": finita}


def sonda_e_logloss_multiclase_propio() -> dict:
    """Sonda (e): el log_loss multiclase propio coincide con el de sklearn.

    El contraste es sobre probabilidades que SI suman 1: se compara
    ``-media_i log p[i, y[i]]`` propio contra ``sklearn.metrics.log_loss`` con las 6 clases
    declaradas, y se publica tambien la desviacion maxima de la suma por fila respecto de 1.

    Returns
    -------
    dict
        Valor propio, valor de sklearn, delta, tolerancia, desviacion de la suma y ``pass``.
    """
    n_filas, n_features = DIM_SONDA_D
    x, y, _denso = problema_aleatorio(n_filas, n_features, SEMILLA_SONDA_E)
    alpha = alpha_derivado(np.bincount(y, minlength=N_CLASES) / float(n_filas))
    w = ftrl_softmax_multipasada(x, y, alpha,
                                 [np.random.default_rng(SEMILLA).permutation(n_filas)])
    p = softmax(x.dot(w))
    propio = log_loss_propio(p, y)
    referencia = float(log_loss(y, p, labels=list(range(N_CLASES))))
    delta = abs(propio - referencia)
    suma = max_abs_suma_menos_uno(p)
    return {"descripcion": "log_loss multiclase propio -media log p[i,y[i]] contra "
                           "sklearn.metrics.log_loss con filas que suman 1",
            "log_loss_propio": propio, "log_loss_sklearn": referencia, "delta": delta,
            "max_abs_suma_menos_uno": suma, "tolerancia": TOL_SONDA_E,
            "pass": bool(delta <= TOL_SONDA_E)}


def sonda_f_filas_suman_uno() -> dict:
    """Sonda (f): las filas del softmax de 6 logits suman 1.

    Se prueban tres familias de logits: la salida real de una pasada FTRL, logits aleatorios
    moderados y logits extremos (1e3), donde una exponencial cruda se desbordaria.

    Returns
    -------
    dict
        Desviaciones maximas por familia, tolerancia y ``pass``.
    """
    n_filas, n_features = DIM_SONDA_D
    x, y, _denso = problema_aleatorio(n_filas, n_features, SEMILLA_SONDA_F)
    alpha = alpha_derivado(np.bincount(y, minlength=N_CLASES) / float(n_filas))
    w = ftrl_softmax_multipasada(x, y, alpha,
                                 [np.random.default_rng(SEMILLA).permutation(n_filas)])
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


def prior_de_es_train(clase: np.ndarray, fold_idx: np.ndarray, fold: int
                      ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Particion ES interna del fold: ES-train (90 %) y ES-val (10 %) sobre filas de TRAIN.

    La particion es EXACTAMENTE la del pre-registro:
    ``StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=20261010+fold)`` aplicado
    a las filas de fold-train. Se devuelven los indices GLOBALES de ES-train, los globales de
    ES-val y sus hashes sha256 (para la sonda de simetria de presupuesto).

    Parameters
    ----------
    clase:
        Etiquetas canonicas.
    fold_idx:
        Fold canonico por fila.
    fold:
        Fold canonico.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray, dict]
        Indices globales de ES-train, de ES-val y sus hashes.

    Raises
    ------
    Detencion
        Si la particion no esta estratificada o no cubre el fold-train (codigo 2).
    """
    tr_idx = np.flatnonzero(fold_idx != fold)
    generador = StratifiedShuffleSplit(n_splits=1, test_size=TEST_SIZE_ES,
                                       random_state=SEMILLA + fold)
    posiciones_train, posiciones_val = next(generador.split(tr_idx, clase[tr_idx]))
    es_train = tr_idx[posiciones_train]
    es_val = tr_idx[posiciones_val]
    esperado_val = round(TEST_SIZE_ES * tr_idx.size)
    if abs(es_val.size - esperado_val) > 1 or np.intersect1d(es_train, es_val).size != 0:
        raise Detencion(f"[STOP] la particion ES del fold {fold} no es la declarada (exit 2)", 2)
    for clase_c in range(N_CLASES):
        if int((clase[es_val] == clase_c).sum()) == 0:
            raise Detencion(f"[STOP] la particion ES del fold {fold} no tiene la clase "
                            f"{clase_c} (exit 2)", 2)
    return es_train, es_val, {"es_train_sha256": sha256_i64(es_train),
                              "es_val_sha256": sha256_i64(es_val)}


def generar_ordenes(fold: int, n_filas: int) -> list[np.ndarray]:
    """Genera la lista de ordenes por pasada UNA sola vez y la devuelve para ambos brazos.

    Cada pasada ``p`` usa ``np.random.default_rng(20261010+fold+p).permutation`` de las filas de
    ES-train: un orden distinto por pasada y por fold. La lista se genera UNA vez y se pasa a A1 y
    a A2 (sonda de simetria de presupuesto: sus sha256 son identicos en ambos brazos).

    Parameters
    ----------
    fold:
        Fold canonico.
    n_filas:
        Numero de filas de ES-train.

    Returns
    -------
    list[numpy.ndarray]
        ``P_MAX`` permutaciones int64 de ``range(n_filas)``.
    """
    return [np.random.default_rng(SEMILLA + fold + p).permutation(n_filas).astype(np.int64)
            for p in range(P_MAX)]


def sonda_g_reproducibilidad_alpha(clase: np.ndarray, fold_idx: np.ndarray, fold: int) -> dict:
    """Sonda (g): alpha* se reproduce BIT a BIT desde el prior MEDIDO del ES-train.

    Recalcula en runtime ``d_norm = sqrt(1 - sum_c prior_c^2)`` con el prior medido del ES-train
    del fold y ``alpha* = 2*M/(d_norm*P_max^2)`` por las DOS rutas (la especializada y la forma
    general con ``power_t = 0``), exige igualdad EXACTA entre ellas y con el ancla congelada
    ``alpha* = 2*M/(d_norm_referencia*P_max^2)``, y registra el sha256 del prior y el de alpha*.

    Parameters
    ----------
    clase:
        Etiquetas canonicas.
    fold_idx:
        Fold canonico por fila.
    fold:
        Fold canonico (0 en el ciclo completo).

    Returns
    -------
    dict
        Prior medido, d_norm, alpha por las dos rutas, diferencia EXACTA, hashes y ``pass``.

    Raises
    ------
    Detencion
        Si alpha* no se reproduce exactamente (codigo 2, ANTES de entrenar).
    """
    es_train, _es_val, _hashes = prior_de_es_train(clase, fold_idx, fold)
    prior = np.bincount(clase[es_train], minlength=N_CLASES) / float(es_train.size)
    d_norm = d_norm_de_prior(prior)
    alpha_run = alpha_derivado(prior)
    alpha_formula = alpha_derivado_general(prior)
    ancla = (2.0 * M_MARGEN) / (ANCLA_D_NORM_REFERENCIA * (P_MAX ** 2))
    diferencia = abs(alpha_run - alpha_formula)
    detalle = {"fold": int(fold), "n_es_train": int(es_train.size), "prior": prior.tolist(),
               "prior_sha256": sha256_f64(prior), "d_norm": d_norm,
               "d_norm_referencia_ancla": ANCLA_D_NORM_REFERENCIA,
               "d_norm_uniforme": d_norm_de_prior(np.full(N_CLASES, 1.0 / N_CLASES)),
               "h_prior_es": entropia_de_prior(prior), "M": M_MARGEN, "P_max": P_MAX,
               "power_t": POWER_T, "alpha_run": alpha_run, "alpha_formula_general": alpha_formula,
               "alpha_sha256": sha256_f64(np.float64(alpha_run)),
               "alpha_ancla_referencia": ancla,
               "alpha_esperado_aproximado": float("0.0352631"),
               "diferencia_exacta": float(diferencia),
               "diferencia_es_cero_exacto": bool(diferencia == 0.0),
               "ancla_reproduce": bool(ancla == ANCLA_ALPHA_PARA_D_NORM_REFERENCIA),
               "M_es_ln_K_menos_1": bool(M_MARGEN == math.log(N_CLASES - 1)),
               "criterio": "alpha*, d_norm y prior recalculados en runtime desde el ES-train; "
                           "|alpha*_run - alpha*_formula| = 0 exacto + sha256 de prior y alpha*; "
                           "si no reproduce -> exit 2 ANTES de entrenar"}
    detalle["pass"] = bool(detalle["diferencia_es_cero_exacto"] and detalle["ancla_reproduce"]
                           and detalle["M_es_ln_K_menos_1"])
    if not detalle["pass"]:
        raise Detencion(f"[STOP] la sonda (g) de reproducibilidad de alpha* falla: "
                        f"diferencia exacta {diferencia!r}, ancla reproduce "
                        f"{detalle['ancla_reproduce']} (exit 2)", 2)
    return detalle


def problema_mini() -> dict:
    """Problema mini sembrado con particion ES y 10 ordenes, para las sondas (h), (i) y (j).

    Es el mismo esquema que el fold real (matriz TF-IDFEquivalent con sesgo materializado para
    A1, la MISMA matriz SIN columna de sesgo para A2, particion ES estratificada y un orden por
    pasada) pero en dimension reducida, de modo que las sondas de estabilidad del ES, de simetria
    de presupuesto y la tabla de mecanismo se puedan ejecutar en ``--verificar`` SIN entrenar el
    experimento ni escribir estado, OOF ni checkpoints.

    Returns
    -------
    dict
        Matrices de ES-train y ES-val de cada brazo, etiquetas, prior, alpha, d_norm, H(prior) y
        los 10 ordenes con sus sha256.
    """
    n_filas, n_features = DIM_SONDA_MINI
    x, y, _denso = problema_aleatorio(n_filas, n_features, SEMILLA_SONDA_MINI)
    generador = StratifiedShuffleSplit(n_splits=1, test_size=TEST_SIZE_ES,
                                       random_state=SEMILLA + SEMILLA_SONDA_MINI)
    train_pos, val_pos = next(generador.split(np.arange(n_filas), y))
    sin_sesgo = x[:, :n_features].tocsr()
    matrices = {}
    for nombre, matriz in (("a1", x), ("a2", sin_sesgo)):
        for parte, posiciones in (("train", train_pos), ("val", val_pos)):
            sub = matriz[posiciones].tocsr()
            sub.sort_indices()
            matrices[f"x_{nombre}_{parte}"] = sub
    y_es_train = y[train_pos].astype(np.int64)
    y_es_val = y[val_pos].astype(np.int64)
    prior = np.bincount(y_es_train, minlength=N_CLASES) / float(y_es_train.size)
    ordenes = generar_ordenes(0, y_es_train.size)
    return {**matrices, "y_es_train": y_es_train, "y_es_val": y_es_val, "prior_es": prior,
            "alpha": alpha_derivado(prior), "d_norm": d_norm_de_prior(prior),
            "h_prior_es": entropia_de_prior(prior), "ordenes": ordenes,
            "es_train_sha256": sha256_i64(np.sort(train_pos)),
            "es_val_sha256": sha256_i64(np.sort(val_pos)),
            "ordenes_sha256": [sha256_i64(orden) for orden in ordenes]}


def ajustar_sgd(fold: int) -> SGDClassifier:
    """Construye el estimador A2 con la config DECLARADA del pre-registro.

    La instancia se construye con ``max_iter=1``, ``tol=None`` y ``warm_start=True``: cada
    llamada a ``partial_fit`` es exactamente UNA epoca (asi lo documenta sklearn), los coeficientes
    se acumulan entre llamadas y ``t_`` (y con el, el learning rate "optimal", que es funcion de
    ``t_``) avanza de forma continua hasta ``t_ = 10 * n_es_train + 1``. Se usa ``partial_fit`` y
    NO ``fit`` con ``warm_start=True`` porque ``BaseSGDClassifier._fit`` hace ``self.t_ = 1.0``
    en CADA llamada (verificado en el codigo de sklearn 1.8.0), lo que reiniciaria el schedule
    "optimal" en cada pasada y no seria un presupuesto de 10 pasadas: medido, queda en
    ``t_ = 401`` en total en vez de ``t_ = 4001``. Es la enmienda pre-ejecucion 2; su desviacion
    declarada (``partial_fit`` x 10 no es bit-identico a ``fit(max_iter=10)``) se documenta en el
    informe.

    Parameters
    ----------
    fold:
        Fold canonico (la semilla es ``20261010 + fold``).

    Returns
    -------
    sklearn.linear_model.SGDClassifier
        Estimador sin ajustar, con la config congelada. NUNCA se construye con ``max_iter = 10``.
    """
    return SGDClassifier(loss="log_loss", alpha=AJUSTE_CONTROL["alpha_reg"],
                         learning_rate=AJUSTE_CONTROL["learning_rate"],
                         shuffle=AJUSTE_CONTROL["shuffle"],
                         fit_intercept=AJUSTE_CONTROL["fit_intercept"],
                         max_iter=AJUSTE_CONTROL["max_iter"], tol=AJUSTE_CONTROL["tol"],
                         warm_start=AJUSTE_CONTROL["warm_start"], average=AJUSTE_CONTROL["average"],
                         random_state=SEMILLA + fold, n_jobs=AJUSTE_CONTROL["n_jobs"])


def fila_a_pasada(x: sp.csr_matrix, y: np.ndarray, orden: np.ndarray) -> tuple[sp.csr_matrix,
                                                                             np.ndarray]:
    """Filas de la pasada en el ORDEN CONGELADO, con indices de columna ordenados.

    Parameters
    ----------
    x:
        Matriz CSR float64 del ES-train (con o sin columna de sesgo).
    y:
        Etiquetas del ES-train en el orden natural.
    orden:
        Permutacion de las filas.

    Returns
    -------
    tuple[scipy.sparse.csr_matrix, numpy.ndarray]
        Submatriz CSR de la pasada y sus etiquetas, en el orden de la pasada.
    """
    filas = x[orden].tocsr()
    filas.sort_indices()
    return filas, np.asarray(y)[orden]


def evaluar_es(metricas: list[float]) -> dict:
    """Aplica la regla de early stopping INTERNO a la trayectoria de metricas ya medida.

    La regla congelada es: mejora minima ``MEJORA_MINIMA`` sobre la mejor metrica, paciencia
    ``PACIENCIA`` y minimo ``P_MIN`` pasadas antes de poder parar. No se evalua NUNCA sobre
    fold-val: la metrica que entra es el macro-F1 sobre el 10 % de TRAIN (ES-val).

    Parameters
    ----------
    metricas:
        Trayectoria de macro-F1 sobre ES-val, una entrada por pasada ejecutada.

    Returns
    -------
    dict
        ``parar``, ``pasada_actual``, ``mejor_pasada``, ``mejor_valor``, ``paciencia_actual`` y
        ``mejoras_registradas``.
    """
    mejor = -math.inf
    mejor_pasada = 0
    paciencia = 0
    mejoras = 0
    parar = False
    for indice, valor in enumerate(metricas):
        pasada = indice + 1
        if valor > mejor + MEJORA_MINIMA:
            mejor = float(valor)
            mejor_pasada = pasada
            paciencia = 0
            mejoras += 1
        else:
            paciencia += 1
        if pasada >= P_MIN and paciencia >= PACIENCIA:
            parar = True
    return {"parar": parar, "pasada_actual": len(metricas), "mejor_pasada": int(mejor_pasada),
            "mejor_valor": float(mejor) if math.isfinite(mejor) else 0.0,
            "paciencia_actual": int(paciencia), "mejoras_registradas": int(mejoras)}


def entrenar_brazo_mini(brazo: str, mini: dict) -> dict:
    """Corre un brazo completo (hasta 10 pasadas con ES interno) sobre el problema mini.

    Es la misma mecanica que el fold real, en dimension reducida, para que las sondas (h) y (j)
    tengan una lectura real sin tocar el corpus.

    Parameters
    ----------
    brazo:
        Nombre del brazo.
    mini:
        Problema mini con matrices, etiquetas, prior y ordenes.

    Returns
    -------
    dict
        Trayectoria por pasada con ``ll_train``, ``D_acum``, ``macro_f1_es`` y ``w_max_abs``
        (mas la razon ``|sigma*w|/|g|`` en A1 y ``t_`` en A2); ``pasada_de_parada``,
        ``mejor_pasada``, ``ll_train_final``, sha256 de los pesos, ``n_actualizaciones`` y el
        vector de sesgo aprendido.
    """
    clave = "a1" if brazo == "A1_ftrl_lambda1" else "a2"
    x_train = mini[f"x_{clave}_train"]
    x_val = mini[f"x_{clave}_val"]
    y_es_train = mini["y_es_train"]
    ordenes = mini["ordenes"]
    trayectoria: list[dict] = []
    if clave == "a1":
        nucleo = FtrlSoftmaxMulticlase(x_train.shape[1], mini["alpha"])
        for indice, orden in enumerate(ordenes):
            stats = nucleo.pasar(x_train, y_es_train, orden)
            ll_train, rango = logits_a_prob(np.asarray(x_train.dot(nucleo.w)), y_es_train)
            predicciones = softmax(np.asarray(x_val.dot(nucleo.w))).argmax(axis=1)
            trayectoria.append({"pasada": indice + 1, "ll_train": ll_train, "D_acum": rango,
                                "macro_f1_es": macro(mini["y_es_val"], predicciones),
                                "w_max_abs": stats["w_max_abs"],
                                "razon_sigma_sobre_g": stats["razon_sigma_sobre_g"],
                                "razon_sigma_sobre_g_pasada": stats["razon_sigma_sobre_g_pasada"]})
            if evaluar_es([item["macro_f1_es"] for item in trayectoria])["parar"]:
                break
        resumen = {"sha256_w": nucleo.sha256_pesos(), "n_actualizaciones": int(nucleo.t),
                   "intercept": [float(valor) for valor in nucleo.coord_sesgo()],
                   "max_abs_g_sesgo": float(nucleo.max_abs_g_sesgo),
                   "g_sesgo_acumulado": [float(valor) for valor in nucleo.acc_g_sesgo]}
    else:
        modelo = ajustar_sgd(0)
        for indice, orden in enumerate(ordenes):
            lote, y_lote = fila_a_pasada(x_train, y_es_train, orden)
            if indice == 0:
                modelo.partial_fit(lote, y_lote, classes=np.arange(N_CLASES))
            else:
                modelo.partial_fit(lote, y_lote)
            ll_train, rango = logits_a_prob(logits_de_sgd(modelo, x_train), y_es_train)
            predicciones = modelo.predict(x_val)
            trayectoria.append({"pasada": indice + 1, "ll_train": ll_train, "D_acum": rango,
                                "macro_f1_es": macro(mini["y_es_val"], predicciones),
                                "w_max_abs": float(np.max(np.abs(modelo.coef_))),
                                "t_": float(modelo.t_)})
        resumen = {"sha256_w": sha256_f64_2d(np.vstack([modelo.coef_.T,
                                                         modelo.intercept_.reshape(1, -1)])),
                   "n_actualizaciones": int(modelo.t_),
                   "intercept": [float(valor) for valor in modelo.intercept_],
                   "max_abs_g_sesgo": None, "g_sesgo_acumulado": None}
    decision = evaluar_es([item["macro_f1_es"] for item in trayectoria])
    return {"brazo": brazo, "trayectoria": trayectoria,
            "pasada_de_parada": decision["pasada_actual"],
            "mejor_pasada": decision["mejor_pasada"],
            "ll_train_final": float(trayectoria[-1]["ll_train"]),
            "D_acum_final": float(trayectoria[-1]["D_acum"]),
            "macro_f1_es_final": float(trayectoria[-1]["macro_f1_es"]),
            "ordenes_sha256": [sha256_i64(orden) for orden in ordenes],
            **resumen}


def logits_de_sgd(modelo: SGDClassifier, x: sp.csr_matrix) -> np.ndarray:
    """Logits OvA de ``SGDClassifier`` sobre una matriz SIN columna de sesgo pre-anadida.

    Es ``X @ coef_.T + intercept_``: el sesgo entra UNA sola vez, por ``fit_intercept=True``.

    Parameters
    ----------
    modelo:
        Estimador ya ajustado.
    x:
        Matriz CSR float64 sin columna de sesgo.

    Returns
    -------
    numpy.ndarray
        Logits (n, 6).
    """
    return np.asarray(x.dot(modelo.coef_.T)) + modelo.intercept_[None, :]


def sonda_h_estabilidad_es(mini: dict) -> dict:
    """Sonda (h): dos corridas identicas de cada brazo dan la MISMA pasada de parada y pesos.

    Corre cada brazo DOS VECES sobre el problema mini con la misma config, la misma particion ES
    y los mismos ordenes congelados, y compara con tolerancia CERO la pasada de parada, el
    ``ll_train`` final y el sha256 de los pesos. Es el mismo criterio que exige el gate G5 sobre
    el fold 0 real.

    Parameters
    ----------
    mini:
        Problema mini ya construido.

    Returns
    -------
    dict
        Detalle por brazo con las dos corridas, la diferencia maxima de ``p``, los tres valores
        comparados y ``pass`` global.
    """
    detalle: dict[str, dict] = {}
    todo_ok = True
    for brazo in BRAZOS:
        primera = entrenar_brazo_mini(brazo, mini)
        segunda = entrar_brazo_mini_segunda(brazo, mini)
        ll_igual = bool(primera["ll_train_final"] == segunda["ll_train_final"])
        pesos_iguales = bool(primera["sha256_w"] == segunda["sha256_w"])
        pasada_igual = bool(primera["pasada_de_parada"] == segunda["pasada_de_parada"])
        max_abs_diff = float(primera["ll_train_final"] - segunda["ll_train_final"])
        ok = bool(ll_igual and pesos_iguales and pasada_igual)
        todo_ok = todo_ok and ok
        detalle[brazo] = {
            "pasada_de_parada": [primera["pasada_de_parada"], segunda["pasada_de_parada"]],
            "ll_train": [primera["ll_train_final"], segunda["ll_train_final"]],
            "sha256_w": [primera["sha256_w"], segunda["sha256_w"]],
            "pasada_igual": pasada_igual, "ll_train_igual": ll_igual,
            "pesos_identicos": pesos_iguales, "max_abs_diff_ll_train": max_abs_diff,
            "tolerancia": 0.0, "pass": ok}
    return {"descripcion": "dos corridas identicas del fold 0 de cada brazo: misma pasada de "
                           "parada, mismo ll_train y sha256 de los pesos IDENTICOS (tolerancia 0)",
            "problema": "mini (dimension reducida); sobre el fold 0 real la misma comprobacion es "
                        "el gate G5",
            "brazos": detalle, "pass": todo_ok}


def entrar_brazo_mini_segunda(brazo: str, mini: dict) -> dict:
    """Segunda corrida identica de un brazo sobre el problema mini (sonda h).

    Es un alias explicito de :func:`entrenar_brazo_mini`: la sonda (h) llama dos veces a la MISMA
    funcion con la MISMA config, que es exactamente lo que exige el criterio.

    Parameters
    ----------
    brazo:
        Nombre del brazo.
    mini:
        Problema mini ya construido.

    Returns
    -------
    dict
        Misma estructura que ``entrenar_brazo_mini``.
    """
    return entrenar_brazo_mini(brazo, mini)


def sonda_i_simetria_presupuesto(mini: dict) -> dict:
    """Sonda (i): los dos brazos reciben la MISMA particion ES y los MISMOS ordenes por pasada.

    Los ordenes se generan UNA vez por fold y se pasan a los dos brazos (no se regeneran por
    brazo): la sonda compara los sha256 de los ordenes que cada brazodeclara haber usado y
    comprueba ademas que el generador es reproducible bit a bit.

    Parameters
    ----------
    mini:
        Problema mini ya construido.

    Returns
    -------
    dict
        sha256 de la particion ES y de cada orden por brazo, igualdad de particion, igualdad de
        ordenes, reproducibilidad del generador y ``pass``.
    """
    corridas = {brazo: entrenar_brazo_mini(brazo, mini) for brazo in BRAZOS}
    ordenes_a1 = corridas["A1_ftrl_lambda1"]["ordenes_sha256"]
    ordenes_a2 = corridas["A2_sgd_presupuesto"]["ordenes_sha256"]
    regenerado = [sha256_i64(orden) for orden in generar_ordenes(0, mini["y_es_train"].size)]
    particion_igual = bool(mini["es_train_sha256"] == mini["es_val_sha256"]) is False
    ordenes_iguales = bool(ordenes_a1 == ordenes_a2)
    generador_reproducible = bool(ordenes_a1 == regenerado == mini["ordenes_sha256"])
    ok = bool(ordenes_iguales and generador_reproducible and particion_igual)
    return {"descripcion": "sha256 de la particion ES y de los ordenes por pasada identicos en "
                           "ambos brazos; los ordenes se generan UNA vez por fold y se pasan a "
                           "los dos brazos",
            "n_pasadas": P_MAX, "n_filas_es_train": int(mini["y_es_train"].size),
            "es_train_sha256": mini["es_train_sha256"], "es_val_sha256": mini["es_val_sha256"],
            "es_train_y_es_val_distintos": particion_igual,
            "ordenes_sha256_A1": ordenes_a1, "ordenes_sha256_A2": ordenes_a2,
            "ordenes_identicos_entre_brazos": ordenes_iguales,
            "generador_reproducible": generador_reproducible,
            "criterio": "sha256 de la particion ES y de los ordenes por pasada identicos en ambos "
                        "brazos", "pass": ok}


def sonda_j_tabla_de_mecanismo(mini: dict) -> dict:
    """Sonda (j): TABLA DE MECANISMO publicada DENTRO del runner (no es pass/fail).

    Publica ``ll_train`` y ``D_acum`` por pasada de A1 y A2, la razon ``|sigma*w|/|g|`` y
    ``w_max_abs`` por pasada, junto con el prior medido y su entropia. La falsacion se reproduce
    aqui: NUNCA se importa de ``/tmp``.

    Parameters
    ----------
    mini:
        Problema mini ya construido.

    Returns
    -------
    dict
        Tabla por brazo y pasada, banda de la regla ``[M/2, 2*M]``, techo de G1 y nota de que no
        es un gate.
    """
    corridas = {brazo: entrenar_brazo_mini(brazo, mini) for brazo in BRAZOS}
    return {"descripcion": "tabla de mecanismo: ll_train y D_acum por pasada de A1 y A2, razon "
                           "|sigma*w|/|g| y w_max_abs; la falsacion se reproduce dentro del "
                           "runner, NO se importa de /tmp",
            "es_gate": False,
            "prior_es": mini["prior_es"].tolist(), "h_prior_es": mini["h_prior_es"],
            "M": M_MARGEN, "banda_D_acum_objetivo": [G0_FACTOR_INFERIOR * M_MARGEN,
                                                     G0_FACTOR_SUPERIOR * M_MARGEN],
            "techo_G1_ll_train": float(mini["h_prior_es"] - G1_TECHO_MARGEN),
            "alpha": float(mini["alpha"]), "d_norm": float(mini["d_norm"]),
            "brazos": {brazo: {"pasada_de_parada": corridas[brazo]["pasada_de_parada"],
                               "mejor_pasada": corridas[brazo]["mejor_pasada"],
                               "trayectoria": corridas[brazo]["trayectoria"]}
                       for brazo in BRAZOS}}


def magnitud_l1(nucleo: "FtrlSoftmaxMulticlase", lambda1: float) -> dict:
    """Escasez efectiva, norma de pesos y multiplicidad de clases vivas del estado de un nucleo.

    Es la magnitud que gobiernan las gates nuevas G7 (``escasez_efectiva``: coordenadas con AL
    MENOS una clase viva, ``|z_j| > lambda1``) y G8 (``norma_w_l2``: razon max/min entre la
    primera y la ultima pasada) y el criterio "L1 inerte" del triage (fraccion de coordenadas sin
    ninguna clase viva). Se calcula SIEMPRE desde el estado almacenado (``z`` y ``w``), nunca desde
    una declaracion.

    Parameters
    ----------
    nucleo:
        Nucleo FTRL ya recorrido por al menos una pasada.
    lambda1:
        Umbral L1 vigente.

    Returns
    -------
    dict
        ``n_coordenadas``, ``coordenadas_vivas``, ``escasez_efectiva``, ``frac_coordenadas_muertas``,
        ``frac_clases_muertas``, ``norma_w_l2`` y ``max_abs_w``.
    """
    vivas = np.abs(nucleo.z) > float(lambda1)
    coordenadas_vivas = int(vivas.any(axis=1).sum())
    n_coordenadas = int(nucleo.z.shape[0])
    return {"n_coordenadas": n_coordenadas,
            "n_clases": int(nucleo.z.shape[1]),
            "coordenadas_vivas": coordenadas_vivas,
            "clases_vivas": int(vivas.sum()),
            "escasez_efectiva": float(coordenadas_vivas) / float(n_coordenadas),
            "frac_coordenadas_muertas": float(1.0 - coordenadas_vivas / float(n_coordenadas)),
            "frac_clases_muertas": float(1.0 - float(vivas.sum()) / float(vivas.size)),
            "norma_w_l2": float(np.linalg.norm(nucleo.w)),
            "max_abs_w": float(np.max(np.abs(nucleo.w)))}


def mediana_de_abs_z(nucleo: "FtrlSoftmaxMulticlase") -> float:
    """Mediana de ``|z_j| > 0`` del estado almacenado: la magnitud que define el anclaje.

    La receta es EXACTAMENTE la congelada: se toman las magnitudes absolutas de toda la matriz
    ``z`` (d, 6), se filtran las estrictamente positivas y se calcula su mediana. No se reordena ni
    se agrega por clase.

    Parameters
    ----------
    nucleo:
        Nucleo FTRL ya recorrido por al menos una pasada.

    Returns
    -------
    float
        Mediana de las entradas ``|z| > 0``.
    """
    magnitudes = np.abs(nucleo.z)
    positivas = magnitudes[magnitudes > 0.0]
    if positivas.size == 0:
        raise Detencion("[STOP] el nucleo de anclaje tiene z identicamente cero: no hay mediana "
                        "que verificar (exit 2)", 2)
    return float(np.median(positivas))


def verificar_anclaje_lambda1(x_es: sp.csr_matrix, y_es: np.ndarray, orden: np.ndarray,
                              alpha: float, lambda1: float) -> dict:
    """Verificacion EN RUNTIME del anclaje de lambda1 durante la pasada 1 del fold 0 de A1.

    Recorre el MISMO stream de A1 (misma matriz, mismas etiquetas, mismo orden congelado, mismo
    alpha) con el nucleo de ANCLAJE (``lambda1 = 0.0``, la trayectoria que DEFINE la mediana),
    calcula la mediana de ``|z_j| > 0`` del ``z`` almacenado, exige diferencia EXACTA 0 con la
    constante congelada, registra el sha256 del snapshot completo de ``z`` y lo compara con el de
    la sonda pre-ejecucion. Ademas publica, como diagnostico secundario y sin mezclarlo con el
    anclaje, la mediana del ``z`` del brazo ya regularizado con ``lambda1`` (que NO puede coincidir
    con el anclaje porque ``z`` depende de ``w`` y ``w`` depende de ``lambda1``: es una comprobacion
    de que el cambio de espacio de soluciones es real, no una segunda medida).

    Parameters
    ----------
    x_es:
        Matriz CSR de ES-train del fold 0 con la coordenada del sesgo ya anadida.
    y_es:
        Etiquetas del ES-train en el orden natural de las filas.
    orden:
        Primer orden congelado de la pasada 1 (el mismo que usa el brazo A1).
    alpha:
        Alpha efectivo del fold.
    lambda1:
        Lambda1 vigente del ciclo.

    Returns
    -------
    dict
        Mediana del anclaje, diferencia EXACTA, sha256 del snapshot de z, su igualdad con el
        congelado, la mediana del brazo regularizado y ``reproduce``.

    Raises
    ------
    Detencion
        Si el anclaje no reproduce la mediana congelada o el sha256 de la sonda previa (codigo 2,
        ANTES de continuar al resto del fold).
    """
    referencia = FtrlSoftmaxMulticlase(x_es.shape[1], alpha, 0.0, LAMBDA2, BETA, POWER_T)
    referencia.pasar(x_es, y_es, orden)
    mediana = mediana_de_abs_z(referencia)
    sha_z = sha256_f64_2d(referencia.z)
    regularizado = FtrlSoftmaxMulticlase(x_es.shape[1], alpha, lambda1, LAMBDA2, BETA, POWER_T)
    regularizado.pasar(x_es, y_es, orden)
    magnitud_regularizada = magnitud_l1(regularizado, lambda1)
    diferencia = float(mediana - MEDIANA_Z_CONGELADA)
    detalle = {"lambda1_congelado": MEDIANA_Z_CONGELADA,
               "lambda1_del_ciclo": float(lambda1),
               "lambda1_es_la_mediana": bool(float(lambda1) == MEDIANA_Z_CONGELADA),
               "nucleo_de_anclaje_lambda1": 0.0,
               "mediana_abs_z_ancaje": mediana,
               "diferencia_exacta": diferencia,
               "diferencia_es_cero_exacto": bool(diferencia == 0.0),
               "sha256_z_ancaje": sha_z,
               "sha256_z_ancaje_congelado": SHA256_Z_ANCLAJE,
               "sha256_reproduce": bool(sha_z == SHA256_Z_ANCLAJE),
               "norma_w_l2_del_brazo_regularizado_pasada_1":
                   magnitud_regularizada["norma_w_l2"],
               "mediana_abs_z_brazo_regularizado": mediana_de_abs_z(regularizado),
               "escasez_efectiva_brazo_regularizado_pasada_1":
                   magnitud_regularizada["escasez_efectiva"],
               "nota": "el anclaje se mide con lambda1 = 0.0 porque es la trayectoria que DEFINE la "
                       "mediana; la mediana del brazo ya regularizado se publica aparte porque z "
                       "depende de w y w depende de lambda1, luego NO puede coincidir con el "
                       "anclaje y su lectura correcta es 'el L1 ya cambio el espacio de soluciones'}",
               "criterio": "diferencia EXACTA 0 con la mediana congelada y sha256 del snapshot de z "
                           "identico al de la sonda pre-ejecucion; si no reproduce -> exit 2"}
    detalle["reproduce"] = bool(detalle["diferencia_es_cero_exacto"] and detalle["sha256_reproduce"]
                                and detalle["lambda1_es_la_mediana"])
    if not detalle["reproduce"]:
        raise Detencion(f"[STOP] el anclaje de lambda1 no reproduce: mediana {mediana!r} con "
                        f"diferencia exacta {diferencia!r} (constante {MEDIANA_Z_CONGELADA!r}), "
                        f"sha256 {sha_z} frente al congelado {SHA256_Z_ANCLAJE} (exit 2)", 2)
    return detalle


def gemelo_lambda1_caso(x_es: sp.csr_matrix, y_es: np.ndarray, ordenes: list[np.ndarray],
                        alpha: float, lambda1: float, nombre: str) -> dict:
    """Un caso del gemelo de ``lambda1``: misma regla de update, mismos ordenes, por pasada.

    Recorre ``PASADAS_SONDA_KLM`` pasadas del nucleo FTRL con el ``lambda1`` dado y registra, por
    pasada, ``ll_train`` sobre el propio stream, ``||w||_2``, la escasez efectiva de coordenadas
    con al menos una clase viva y la multiplicidad de clases vivas. No lee ni escribe nada fuera
    del stream: es exactamente la receta de la sonda pre-ejecucion, ejecutada dentro del runner.

    Parameters
    ----------
    x_es:
        Matriz CSR de ES-train con la coordenada del sesgo anadida.
    y_es:
        Etiquetas del ES-train en el orden natural.
    ordenes:
        Permutaciones congeladas, una por pasada.
    alpha:
        Alpha efectivo del fold.
    lambda1:
        Lambda1 del caso.
    nombre:
        Nombre del caso (``lambda1_0``, ``lambda1_alpha`` o ``lambda1_mediana``).

    Returns
    -------
    dict
        ``nombre``, ``lambda1``, la trayectoria por pasada y los dos escalares finales.
    """
    nucleo = FtrlSoftmaxMulticlase(x_es.shape[1], alpha, lambda1, LAMBDA2, BETA, POWER_T)
    trayectoria: list[dict] = []
    for indice, orden in enumerate(ordenes[:PASADAS_SONDA_KLM]):
        nucleo.pasar(x_es, y_es, orden)
        probabilidades = softmax(np.asarray(x_es.dot(nucleo.w)))
        verdadero = probabilidades[np.arange(y_es.shape[0]), y_es]
        perdida = float(-np.mean(np.log(np.clip(verdadero, L2_CLIP, 1.0))))
        magnitud = magnitud_l1(nucleo, lambda1)
        trayectoria.append({"pasada": indice + 1, "ll_train": perdida,
                            "norma_w_l2": magnitud["norma_w_l2"],
                            "coordenadas_vivas": magnitud["coordenadas_vivas"],
                            "clases_vivas": magnitud["clases_vivas"],
                            "escasez_efectiva": magnitud["escasez_efectiva"],
                            "frac_coordenadas_muertas": magnitud["frac_coordenadas_muertas"]})
    return {"nombre": nombre, "lambda1": float(lambda1), "trayectoria": trayectoria,
            "ll_train_final": trayectoria[-1]["ll_train"],
            "norma_w_l2_final": trayectoria[-1]["norma_w_l2"],
            "escasez_efectiva_final": trayectoria[-1]["escasez_efectiva"],
            "normas_por_pasada": [punto["norma_w_l2"] for punto in trayectoria]}


def cargar_gemelo_previo() -> dict:
    """Lee el artefacto de la sonda pre-ejecucion versionada que las sondas (k), (l) y (m) comparan.

    Parameters
    ----------
    None.

    Returns
    -------
    dict
        Contenido de ``experiments/v35_00b_sonda_lambda1.json`` con sus tres casos.

    Raises
    ------
    Detencion
        Si el artefacto no existe o no trae los tres casos (codigo 2).
    """
    if not SONDA_JSON.exists():
        raise Detencion(f"[STOP] no existe la sonda pre-ejecucion {ruta_relativa(SONDA_JSON)} "
                        "(exit 2)", 2)
    previo = json.loads(SONDA_JSON.read_text(encoding="utf-8"))
    casos = {caso["nombre"]: caso for caso in previo.get("gemelo_real", [])}
    if set(casos) != {"lambda1_0", "lambda1_alpha", "lambda1_mediana"}:
        raise Detencion("[STOP] la sonda pre-ejecucion no trae los tres casos del gemelo (exit 2)",
                        2)
    return previo


def sondas_klm_gemelo_lambda1(x_es: sp.csr_matrix, y_es: np.ndarray,
                              ordenes: list[np.ndarray], alpha: float) -> dict:
    """Sondas (k), (l) y (m): los tres casos del gemelo de lambda1, NO-GATE, dentro del runner.

    Reproduce los tres casos de la sonda pre-ejecucion versionada (``lambda1 = 0``,
    ``lambda1 = alpha`` y ``lambda1 = mediana``) sobre el MISMO stream del fold 0 con las MISMAS
    reglas de update, el mismo alpha y los mismos ordenes congelados, y compara ``ll_train``,
    ``||w||_2`` y la escasez efectiva contra el artefacto con tolerancia CERO declarada.

    No son pass/fail: no deciden el veredicto. Se publican porque la decision de mecanismo NO puede
    descansar en evidencia importada de fuera del runner. El contraste de ``(l)`` con ``(k)``
    publica ademas ``max|delta w|`` para cuantificar cuanto de no-op es ``lambda1 = alpha`` (el
    pre-registro declara que la mediana supera a alpha por un factor 6.8044, luego el umbral solo
    descarta la parte mas pequena de la trayectoria).

    Parameters
    ----------
    x_es:
        Matriz CSR de ES-train del fold 0 con la coordenada del sesgo anadida.
    y_es:
        Etiquetas del ES-train en el orden natural.
    ordenes:
        Permutaciones congeladas del fold 0.
    alpha:
        Alpha efectivo del fold.

    Returns
    -------
    dict
        Los tres casos con su trayectoria, el contraste con el artefacto, la tolerancia declarada,
        si reproduce y la politica NO-GATE.
    """
    previo = cargar_gemelo_previo()
    casos_esperados = {caso["nombre"]: caso for caso in previo["gemelo_real"]}
    ejecutados = {"lambda1_0": gemelo_lambda1_caso(x_es, y_es, ordenes, alpha, 0.0, "lambda1_0"),
                  "lambda1_alpha": gemelo_lambda1_caso(x_es, y_es, ordenes, alpha, alpha,
                                                       "lambda1_alpha"),
                  "lambda1_mediana": gemelo_lambda1_caso(x_es, y_es, ordenes, alpha,
                                                        MEDIANA_Z_CONGELADA, "lambda1_mediana")}
    contraste: dict[str, dict] = {}
    todo_reproduce = True
    for nombre, caso in ejecutados.items():
        esperado = casos_esperados[nombre]
        puntos_previos = {int(punto["pasada"]): punto
                          for punto in esperado["trayectoria"]}
        por_pasada = []
        reproduce_caso = True
        for punto in caso["trayectoria"]:
            previo_pasada = puntos_previos[int(punto["pasada"])]
            delta_ll = float(punto["ll_train"]) - float(previo_pasada["ll_train"])
            delta_norma = float(punto["norma_w_l2"]) - float(previo_pasada["norma_w_l2"])
            delta_escasez = (float(punto["escasez_efectiva"])
                             - float(previo_pasada["escasez_efectiva"]))
            coincide = bool(abs(delta_ll) <= TOL_SONDA_KLM
                            and abs(delta_norma) <= TOL_SONDA_KLM
                            and abs(delta_escasez) <= TOL_SONDA_KLM)
            reproduce_caso = reproduce_caso and coincide
            por_pasada.append({"pasada": punto["pasada"], "ll_train_run": punto["ll_train"],
                               "ll_train_previo": float(previo_pasada["ll_train"]),
                               "delta_ll_train": delta_ll,
                               "norma_w_l2_run": punto["norma_w_l2"],
                               "norma_w_l2_previo": float(previo_pasada["norma_w_l2"]),
                               "delta_norma_w_l2": delta_norma,
                               "escasez_efectiva_run": punto["escasez_efectiva"],
                               "escasez_efectiva_previo":
                                   float(previo_pasada["escasez_efectiva"]),
                               "delta_escasez_efectiva": delta_escasez,
                               "reproduce": coincide})
        todo_reproduce = todo_reproduce and reproduce_caso
        contraste[nombre] = {"lambda1_run": caso["lambda1"],
                             "lambda1_previo": float(esperado["lambda1"]),
                             "lambda1_identico": bool(caso["lambda1"]
                                                      == float(esperado["lambda1"])),
                             "por_pasada": por_pasada, "reproduce": reproduce_caso}
    normas_mediana = ejecutados["lambda1_mediana"]["normas_por_pasada"]
    razon = (float(max(normas_mediana)) / float(min(normas_mediana))
             if min(normas_mediana) > 0.0 else float("inf"))
    return {
        "descripcion": "sondas (k), (l) y (m): los tres casos del gemelo de lambda1 de la sonda "
                       "pre-ejecucion, reproducidos DENTRO del runner con las mismas reglas de "
                       "update, el mismo alpha y los mismos ordenes congelados, y contrastados "
                       "contra el artefacto versionado con tolerancia cero",
        "es_gate": False,
        "politica": "son sondas NO-GATE: no deciden el veredicto, pero la decision de mecanismo no "
                    "puede descansar en evidencia importada de /tmp, luego se reproducen aqui; una "
                    "no reproduccion se publica y no detiene el ciclo",
        "tolerancia": TOL_SONDA_KLM,
        "artefacto_previo": ruta_relativa(SONDA_JSON),
        "sha256_artefacto_previo": sha256_archivo(SONDA_JSON),
        "n_filas_es_train": int(x_es.shape[0]),
        "n_coordenadas": int(x_es.shape[1]),
        "n_pasadas": PASADAS_SONDA_KLM,
        "alpha": float(alpha),
        "lambda1_mediana_congelada": MEDIANA_Z_CONGELADA,
        "casos": {nombre: caso for nombre, caso in ejecutados.items()},
        "contraste": contraste,
        "reproducen_los_tres_casos": bool(todo_reproduce),
        "razon_max_min_norma_lambda1_mediana": razon,
        "razon_max_min_norma_previa": float(previo["flag_tecnico_2"][
            "razon_max_min_lambda1_mediana"]),
        "flag_1_mediana_sobre_alpha": float(previo["anclaje"]["mediana_sobre_alpha"]),
        "lectura": "razon max/min de ||w||_2 > 1.0 con lambda1 = mediana significa que el L1 NO "
                   "acota la norma por si solo (solo acota si z permanece acotado), que es "
                   "exactamente lo que G8 pone a prueba en el fold 0 real",
    }


#: Sondas NO-GATE: la (j) publica la tabla de mecanismo y las tres ultimas reproducen el gemelo de
#: lambda1 dentro del runner. Ninguna decide el veredicto.
SONDAS_NO_GATE = ("j_tabla_de_mecanismo", "k_caso_lambda1_cero", "l_caso_lambda1_alpha",
                  "m_caso_lambda1_mediana")
NOMBRES_SONDAS = ("a_vectorizado_vs_lazo", "b_raiz_analitica_lambda0", "c_totalf1_macro",
                  "d_sin_nan_logloss_finito", "e_logloss_multiclase_propio", "f_filas_suman_uno",
                  "g_reproducibilidad_alpha", "h_estabilidad_es", "i_simetria_presupuesto",
                  "j_tabla_de_mecanismo", "k_caso_lambda1_cero", "l_caso_lambda1_alpha",
                  "m_caso_lambda1_mediana")
#: Sondas que son pass/fail: las nueve primeras. La (j) publica y no falla; las (k), (l) y (m)
#: publican el gemelo de lambda1 y tampoco fallan.
SONDAS_PASS_FAIL = NOMBRES_SONDAS[:9]
NOMBRES_SONDAS_KLM = ("k_caso_lambda1_cero", "l_caso_lambda1_alpha", "m_caso_lambda1_mediana")


def ejecutar_sondas(clase: np.ndarray, fold_idx: np.ndarray, textos: np.ndarray,
                    determinismo: dict | None) -> dict:
    """Ejecuta las sondas obligatorias (a)-(m) antes de cualquier ajuste real.

    Las sondas (h), (i) y (j) usan el problema mini con la MISMA mecanica (matrices, particion ES,
    10 ordenes congelados y early stopping interno). Las sondas (k), (l) y (m) reproducen los tres
    casos del gemelo de ``lambda1`` de la sonda pre-ejecucion sobre el ES-train REAL del fold 0 con
    las mismas reglas de update, el mismo alpha y los mismos ordenes congelados, y los contrastan
    contra el artefacto versionado con tolerancia cero: la decision de mecanismo no puede descansar
    en evidencia importada de fuera del runner. Si el fold 0 ya esta entrenado, la lectura
    determinista de (h) se sustituye por la comparacion REAL de las dos corridas del fold 0 (que es
    el gate G5).

    Parameters
    ----------
    clase:
        Etiquetas canonicas.
    fold_idx:
        Fold canonico por fila.
    textos:
        Textos del corpus en orden canonico (para las sondas (k), (l) y (m)).
    determinismo:
        Sondas de determinismo del fold 0 ya medidas (o None si aun no se ha entrenado).

    Returns
    -------
    dict
        Detalle de las trece sondas con valores, tolerancias y ``todas_pasan``.

    Raises
    ------
    Detencion
        Si alguna sonda pass/fail falla (codigo 4).
    """
    inicio = time.time()
    mini = problema_mini()
    mecanismo = sonda_j_tabla_de_mecanismo(mini)
    datos_fold = preparar_fold(textos, clase, fold_idx, 0)
    alpha_fold = float(datos_fold["alpha"])
    ordenes_fold = generar_ordenes(0, int(datos_fold["hashes"]["n_filas_fit_es_train"]))
    gemelo = sondas_klm_gemelo_lambda1(datos_fold["x_a1_train"], datos_fold["y_es_train"],
                                       ordenes_fold, alpha_fold)
    sondas: dict[str, dict] = {
        "a_vectorizado_vs_lazo": sonda_a_vectorizado_vs_lazo(),
        "b_raiz_analitica_lambda0": sonda_b_raiz_analitica_lambda0(),
        "c_totalf1_macro": sonda_c_totalf1_macro(),
        "d_sin_nan_logloss_finito": sonda_d_sin_nan_logloss_finito(),
        "e_logloss_multiclase_propio": sonda_e_logloss_multiclase_propio(),
        "f_filas_suman_uno": sonda_f_filas_suman_uno(),
        "g_reproducibilidad_alpha": sonda_g_reproducibilidad_alpha(clase, fold_idx, 0),
        "h_estabilidad_es": sonda_h_estabilidad_es(mini),
        "i_simetria_presupuesto": sonda_i_simetria_presupuesto(mini),
        "j_tabla_de_mecanismo": mecanismo,
        "k_caso_lambda1_cero": dict(gemelo["casos"]["lambda1_0"],
                                    descripcion=gemelo["descripcion"], es_gate=False,
                                    caso="lambda1_0", contraste=gemelo["contraste"]["lambda1_0"],
                                    reproduce=gemelo["contraste"]["lambda1_0"]["reproduce"],
                                    tolerancia=gemelo["tolerancia"],
                                    politica=gemelo["politica"]),
        "l_caso_lambda1_alpha": dict(gemelo["casos"]["lambda1_alpha"],
                                     descripcion=gemelo["descripcion"], es_gate=False,
                                     caso="lambda1_alpha",
                                     contraste=gemelo["contraste"]["lambda1_alpha"],
                                     reproduce=gemelo["contraste"]["lambda1_alpha"]["reproduce"],
                                     tolerancia=gemelo["tolerancia"],
                                     flag_1_mediana_sobre_alpha=gemelo["flag_1_mediana_sobre_alpha"],
                                     politica=gemelo["politica"]),
        "m_caso_lambda1_mediana": dict(gemelo["casos"]["lambda1_mediana"],
                                       descripcion=gemelo["descripcion"], es_gate=False,
                                       caso="lambda1_mediana",
                                       contraste=gemelo["contraste"]["lambda1_mediana"],
                                       reproduce=gemelo["contraste"]["lambda1_mediana"]["reproduce"],
                                       tolerancia=gemelo["tolerancia"],
                                       razon_max_min_norma=gemelo["razon_max_min_norma_lambda1_mediana"],
                                       razon_max_min_norma_previa=gemelo[
                                           "razon_max_min_norma_previa"],
                                       lectura=gemelo["lectura"],
                                       politica=gemelo["politica"]),
        "_gemelo_lambda1": gemelo,
    }
    if determinismo is not None:
        sondas["h_estabilidad_es"] = {
            "descripcion": "dos corridas identicas del fold 0 REAL de cada brazo: misma pasada de "
                           "parada, mismo ll_train y sha256 de los pesos IDENTICOS (tolerancia 0)",
            "brazos": determinismo, "sustituye_a": "lectura mini (--verificar)",
            "pass": bool(all(determinismo[brazo]["pass"] for brazo in BRAZOS))}
    detalle = dict(sondas)
    detalle["todas_pasan"] = bool(all(sondas[nombre]["pass"] for nombre in SONDAS_PASS_FAIL))
    detalle["politica"] = ("si alguna de las nueve sondas pass/fail falla -> exit 4 ANTES de "
                           "entrenar; cualquier ajuste exige enmienda pre-ejecucion. Las sondas "
                           "no-gate (j) y (k), (l), (m) se publican y no deciden el veredicto")
    detalle["duracion_seg"] = float(time.time() - inicio)
    for nombre in NOMBRES_SONDAS:
        if nombre in SONDAS_NO_GATE:
            marca = "publicada (no es gate)"
            if nombre == "j_tabla_de_mecanismo":
                marca += (f", A1 parad en "
                          f"{mecanismo['brazos']['A1_ftrl_lambda1']['pasada_de_parada']} y A2 en "
                          f"{mecanismo['brazos']['A2_sgd_presupuesto']['pasada_de_parada']} "
                          "pasadas")
            else:
                marca += (f", reproduce el artefacto = {sondas[nombre]['reproduce']} "
                          f"(tolerancia {TOL_SONDA_KLM})")
            print(f"  sonda ({nombre[0]}) {nombre}: {marca}", flush=True)
            continue
        print(f"  sonda ({nombre[0]}) {nombre}: pass={sondas[nombre]['pass']}", flush=True)
    if not detalle["todas_pasan"]:
        fallidas = [nombre for nombre in SONDAS_PASS_FAIL if not sondas[nombre]["pass"]]
        raise Detencion(f"[STOP] sondas obligatorias fallidas {fallidas} (exit 4)", 4)
    return detalle


def sonda_asimetria_no_equivalente(prereg: dict) -> dict:
    """Sonda exigida: A2 es CONTROL DE PRESUPUESTO, no gemelo mecanico, y el veredicto no usa el
    contraste A1 frente a A2.

    A1 lleva un regularizador L1 que A2 no lleva: luego el contraste entre brazos esta CONTAMINADO
    por la asimetria del regularizador y no puede sostener ninguna conclusion. Esta sonda publica
    la asimetria, fija la regla de lectura UNIDIRECCIONAL (el veredicto se apoya solo en evidencia
    INTERNA de A1 contra la regla derivada, es decir G0/G1/G4 mas el triage y las sondas (k), (l) y
    (m), y NUNCA en el delta entre brazos, en razones entre brazos ni en un IC bootstrap cruzado) y
    declara el papel real de A2: mostrar que el presupuesto de 10 pasadas no es el limite.

    Parameters
    ----------
    prereg:
        Contenido del pre-registro.

    Returns
    -------
    dict
        Asimetria declarada, valores asimetricos, regla de lectura unidireccional, lectura correcta
        de A2 y ``pass``.
    """
    asimetria = prereg["puerta"]["asimetria"]
    valores = {"lambda1_ftrl": LAMBDA1, "lambda1_sgd_control": 0.0,
               "lambda2_ftrl": LAMBDA2, "alpha_reg_sgd": AJUSTE_CONTROL["alpha_reg"],
               "cociente_lambda2_sobre_alpha_reg":
                   float(LAMBDA2) / float(AJUSTE_CONTROL["alpha_reg"])}
    return {"descripcion": "A2 es control de PRESUPUESTO (no gemelo mecanico: no tiene L1); el "
                           "veredicto NO se apoya en el contraste A1 frente a A2",
            "valores": valores,
            "asimetria_declarada": asimetria,
            "regla_de_lectura_unidireccional": asimetria,
            "lectura_correcta_de_a2": "A2 no es un gemelo mecanico de A1: se lee unicamente como "
                                      "'el presupuesto de 10 pasadas NO es el limite'. El delta "
                                      "entre brazos, las razones entre brazos y el IC bootstrap "
                                      "cruzado quedan contaminados por la asimetria del "
                                      "regularizador y NO se publican como evidencia",
            "apoyo_del_veredicto": "evidencia interna de A1 contra la regla derivada (G0/G1/G4), "
                                    "mas G7/G8 sobre la trayectoria de A1, mas el triage formal y "
                                    "las sondas (k), (l) y (m) del gemelo de lambda1",
            "control_ajuste": (f"{AJUSTE_CONTROL['modo']} con {AJUSTE_CONTROL['llamadas']} "
                               f"llamadas y estimador construido con max_iter = "
                               f"{AJUSTE_CONTROL['max_iter']} (NO {P_MAX})"),
            "lambda1_ftrl": LAMBDA1,
            "lambda1_sgd_control": 0.0,
            "lambda2_ftrl": valores["lambda2_ftrl"],
            "alpha_reg_sgd": valores["alpha_reg_sgd"],
            "cociente": valores["cociente_lambda2_sobre_alpha_reg"],
            "criterio": "A1 tiene L1 y A2 no: el contraste entre brazos esta contaminado y el "
                        "veredicto se apoya solo en evidencia interna de A1",
            "pass": bool(LAMBDA1 == MEDIANA_Z_CONGELADA and valores["lambda1_sgd_control"] == 0.0
                         and "PRESUPUESTO" in asimetria
                         and "Ninguna conclusion puede descansar" in asimetria)}



# =====================================================================================
# ---- 7. CHECKPOINT, VECTORIZADOR POR FOLD Y ENTRENAMIENTO DE UN BRAZO ----
# =====================================================================================
def estado_nuevo() -> dict:
    """Estado resumible inicial del checkpoint.

    Returns
    -------
    dict
        Checkpoint vacio con inicio, guardas, calibracion, puerta, triage, refit, tabla de
        mecanismo, particiones ES y vectorizador por fold, y los dos brazos.
    """
    return {"fase": FASE_ESTADO, "inicio_ts": float(time.time()), "timeout_global_h": TIMEOUT_H,
            "ram_tope_gb": RAM_TOPE_GB, "rss_pico_gb": 0.0, "sondas": None, "calibracion": None,
            "puerta": None, "triage": None, "refit": None, "mecanismo": None,
            "es_por_fold": {}, "vectorizador_por_fold": {},
            "brazos": {brazo: {"folds_completados": [], "folds": {},
                               "mascara_relleno_por_fila": [0] * N_FILAS,
                               "sonda_determinismo": None} for brazo in BRAZOS}}


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
        Si el checkpoint existe pero no corresponde a V35-E1 o esta incompleto (codigo 2).
    """
    if not ESTADO_JSON.exists():
        estado = estado_nuevo()
        guardar_estado(estado)
        return estado
    estado = json.loads(ESTADO_JSON.read_text(encoding="utf-8"))
    if estado.get("fase") != FASE_ESTADO:
        raise Detencion(f"[STOP] el checkpoint no corresponde a {FASE_ESTADO} (exit 2)", 2)
    for clave in ("inicio_ts", "rss_pico_gb", "calibracion", "puerta", "triage", "refit",
                  "mecanismo", "es_por_fold", "vectorizador_por_fold", "brazos"):
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


def anadir_sesgo(x: sp.csr_matrix, x0: float = X0) -> sp.csr_matrix:
    """Anade la coordenada del sesgo (ultima columna, valor constante ``x0``) a un CSR.

    El sesgo es una coordenada MAS de FTRL, con su propio ``n`` y su propio ``z``: por eso se
    materializa como columna y no como termino aparte, de modo que ``s = X @ W`` queda escrito
    como un unico producto contra una matriz de pesos de (d+1, 6). Solo se usa en A1: A2 recibe la
    MISMA matriz SIN esta columna y aporta el sesgo una sola vez por ``fit_intercept=True``.

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


def preparar_fold(textos: np.ndarray, clase: np.ndarray, fold_idx: np.ndarray,
                  fold: int) -> dict:
    """Prepara el fold: particion ES, vectorizador ajustado SOLO en ES-train y matrices de ambos
    brazos.

    El vectorizador se ajusta en las filas de ES-train (el 90 % del fold-train), nunca en el
    fold-train completo ni en el fold-val: es lo que declara el pre-registro porque la regla de
    alpha se deriva sobre las filas que se entrenan. Se registran los hashes del vocabulario
    ordenado y del idf del fold.

    Parameters
    ----------
    textos:
        Textos del corpus en orden canonico.
    clase:
        Etiquetas canonicas.
    fold_idx:
        Fold canonico por fila.
    fold:
        Fold canonico.

    Returns
    -------
    dict
        Matrices de A1 (con sesgo) y A2 (sin sesgo) de ES-train, ES-val y fold-val; etiquetas;
        prior medido, d_norm, alpha, H(prior) y hashes del fold.
    """
    tr_idx, va_idx = indices_fold(fold_idx, fold)
    es_train, es_val, hashes_es = prior_de_es_train(clase, fold_idx, fold)
    vectorizador = TfidfVectorizer(**VECTORIZADOR_FIT)
    x_es = vectorizador.fit_transform(textos[es_train])
    x_es_val = vectorizador.transform(textos[es_val])
    x_va = vectorizador.transform(textos[va_idx])
    y_es_train = clase[es_train].astype(np.int64)
    y_es_val = clase[es_val].astype(np.int64)
    prior = np.bincount(y_es_train, minlength=N_CLASES) / float(y_es_train.size)
    hashes = {**hashes_es, "n_filas_fit_es_train": int(es_train.size),
              "n_filas_es_val": int(es_val.size), "n_train_fold": int(tr_idx.size),
              "n_val_fold": int(va_idx.size), "n_vocabulario": len(vectorizador.vocabulary_),
              "vocab_sha256": sha256_textos(sorted(vectorizador.vocabulary_)),
              "idf_sha256": sha256_f64(vectorizador.idf_),
              "n_features_es_train_sin_sesgo": int(x_es.shape[1]),
              "n_features_es_val_sin_sesgo": int(x_es_val.shape[1]),
              "n_coordinates_a1": int(x_es.shape[1] + 1),
              "prior": prior.tolist(), "prior_sha256": sha256_f64(prior),
              "d_norm": d_norm_de_prior(prior), "h_prior_es": entropia_de_prior(prior),
              "alpha": alpha_derivado(prior), "alpha_sha256": sha256_f64(np.float64(
                  alpha_derivado(prior))),
              "alpha_formula_general": alpha_derivado_general(prior),
              "alpha_diferencia_exacta": float(abs(alpha_derivado(prior)
                                                  - alpha_derivado_general(prior)))}
    return {"fold": int(fold), "tr_idx": tr_idx, "va_idx": va_idx, "es_train": es_train,
            "es_val": es_val, "hashes": hashes, "y_es_train": y_es_train, "y_es_val": y_es_val,
            "prior": prior, "alpha": hashes["alpha"],
            "x_a1_train": anadir_sesgo(x_es), "x_a1_esval": anadir_sesgo(x_es_val),
            "x_a1_va": anadir_sesgo(x_va), "x_a2_train": x_es.tocsr(), "x_a2_esval": x_es_val.tocsr(),
            "x_a2_va": x_va.tocsr()}


def guardar_pesos(fold: int, brazo: str, w: np.ndarray, z: np.ndarray, n: np.ndarray,
                  pasada_de_parada: int, extra: dict | None = None) -> str:
    """Persiste los pesos de un fold y un brazo con escritura atomica.

    El ``.npz`` contiene ``w``, ``z``, ``n`` y ``pasada_de_parada`` (lo que exige el
    pre-registro) mas, en A2, el ``intercept_`` (A2 no tiene acumuladores: ``z`` y ``n`` se
    guardan a cero y el sesgo va en su propia clave, nunca en ``w``).

    Parameters
    ----------
    fold:
        Fold canonico.
    brazo:
        Nombre del brazo.
    w:
        Pesos: (d, 6) en A1, ``coef_.T`` en A2.
    z:
        Acumulador de correccion (ceros en A2).
    n:
        Acumulador de cuadrados (ceros en A2).
    pasada_de_parada:
        Pasada en la que el early stopping interno detuvo el brazo.
    extra:
        Claves adicionales a guardar (por ejemplo ``intercept_`` en A2).

    Returns
    -------
    str
        Ruta relativa del artefacto escrito.
    """
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    ruta = ruta_pesos(fold, brazo)
    temporal = ruta.with_name(ruta.name + ".tmp")
    datos: dict[str, np.ndarray] = {"w": np.ascontiguousarray(w, dtype=np.float64),
                                    "z": np.ascontiguousarray(z, dtype=np.float64),
                                    "n": np.ascontiguousarray(n, dtype=np.float64),
                                    "pasada_de_parada": np.asarray([pasada_de_parada],
                                                                  dtype=np.int64)}
    for clave, valor in (extra or {}).items():
        datos[clave] = np.asarray(valor)
    with temporal.open("wb") as manejador:
        np.savez(manejador, **datos)
    os.replace(temporal, ruta)
    return ruta_relativa(ruta)


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

    El OOF son las probabilidades CRUDAS del brazo (softmax de 6 logits en A1,
    ``predict_proba`` en A2): ambas suman 1 por fila por construccion, que es justo lo que mide la
    puerta G6.

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
    """Devuelve el OOF del brazo (ceros si es nuevo) y valida su coherencia al reanudar.

    La mascara de relleno se comprueba contra los folds completados: si una fila ya esta
    rellenada por un fold completado y vuelve a aparecer, el OOF esta doblemente rellenado y el
    runner se detiene con codigo 2.

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


def registrar_fold(estado: dict, brazo: str, fold: int, resultado: dict, oof: np.ndarray,
                   hashes_fold: dict, va_idx: np.ndarray, clase_va: np.ndarray) -> dict:
    """Incorpora el fold al OOF y al checkpoint y persiste ambos (OOF primero, estado luego).

    La mascara de relleno se escribe una sola vez por fila y el fold-val jamas entra al ajuste.

    Parameters
    ----------
    estado:
        Checkpoint a actualizar.
    brazo:
        Nombre del brazo.
    fold:
        Fold entrenado.
    resultado:
        Salida de ``entrenar_brazo_fold``.
    oof:
        OOF del brazo (se modifica in situ).
    hashes_fold:
        Hashes y magnitudes del fold.
    va_idx:
        Indices del fold-val.
    clase_va:
        Etiquetas reales del fold-val.

    Returns
    -------
    dict
        Detalle del fold tal como queda en el checkpoint.

    Raises
    ------
    Detencion
        Si la mascara detecta una fila escrita dos veces (codigo 2).
    """
    datos = estado["brazos"][brazo]
    mascara = np.asarray(datos["mascara_relleno_por_fila"], dtype=np.int64)
    if va_idx.size and int(mascara[va_idx].max()) != 0:
        raise Detencion(f"[STOP] {brazo} fold {fold}: fila ya rellenada en el OOF (exit 2)", 2)
    oof[va_idx] = resultado["p_va"]
    mascara[va_idx] += 1
    guardar_oof(brazo, oof)
    detalle = {"n_vocabulario": int(hashes_fold["n_vocabulario"]),
               "vocab_sha256": hashes_fold["vocab_sha256"],
               "idf_sha256": hashes_fold["idf_sha256"],
               "n_filas_fit_es_train": int(hashes_fold["n_filas_fit_es_train"]),
               "n_filas_es_val": int(hashes_fold["n_filas_es_val"]),
               "n_train_fold": int(hashes_fold["n_train_fold"]),
               "n_val_fold": int(hashes_fold["n_val_fold"]),
               "n_features_es_train": int(hashes_fold["n_features_es_train_sin_sesgo"]),
               "n_features_es_val": int(hashes_fold["n_features_es_val_sin_sesgo"]),
               "n_coordinates_a1": int(hashes_fold["n_coordinates_a1"]),
               "alpha_efectivo": float(resultado["alpha"]), "d_norm": float(hashes_fold["d_norm"]),
               "h_prior_es": float(hashes_fold["h_prior_es"]),
               "prior_es": [float(valor) for valor in hashes_fold["prior"]],
               "prior_es_sha256": hashes_fold["prior_sha256"],
               "alpha_sha256": hashes_fold["alpha_sha256"],
               "es_train_sha256": hashes_fold["es_train_sha256"],
               "es_val_sha256": hashes_fold["es_val_sha256"],
               "lambda1": float(resultado["lambda1"]),
               "anclaje_lambda1": resultado["anclaje_lambda1"],
               "ordenes_sha256": list(resultado["ordenes_sha256"]),
               "pasada_de_parada": int(resultado["pasada_de_parada"]),
               "mejor_pasada": int(resultado["mejor_pasada"]),
               "trayectoria": resultado["trayectoria"], "sesgo": resultado["sesgo"],
               "t_": float(resultado["n_actualizaciones"]),
               "n_actualizaciones": int(resultado["n_actualizaciones"]),
               "macro_f1_standalone_fold_val": float(resultado["macro_f1_standalone_fold_val"]),
               "frecuencias_fold_val": [float(valor) for valor in resultado["frecuencias_fold_val"]],
               "max_abs_suma_menos_uno": float(resultado["max_abs_suma_menos_uno"]),
               "tiempo_seg": float(resultado["tiempo_seg"])}
    datos["folds"][str(fold)] = detalle
    datos["folds_completados"] = sorted(datos["folds"])
    datos["mascara_relleno_por_fila"] = mascara.tolist()
    guardar_estado(estado)
    final = resultado["trayectoria"][-1]
    print(f"  [{brazo}] fold {fold}: alpha {detalle['alpha_efectivo']!r} (d_norm "
          f"{detalle['d_norm']:.6f}), parada en la pasada {detalle['pasada_de_parada']} "
          f"(mejor {detalle['mejor_pasada']}), ll_train {final['ll_train']:.6f}, D_acum "
          f"{final['D_acum']:.6f}, macro-F1 standalone {detalle['macro_f1_standalone_fold_val']:.4f}, "
          f"{detalle['tiempo_seg']:.1f} s", flush=True)
    return detalle


def entrenar_brazo_fold(brazo: str, datos_fold: dict, ordenes: list[np.ndarray], fold: int,
                        clase_va: np.ndarray) -> dict:
    """Entrena UN fold de UN brazo con el presupuesto multi-pasada y el ES interno congelados.

    Los dos brazos reciben la MISMA particion ES, los MISMOS ordenes por pasada, el MISMO
    ``P_max = 10`` y la MISMA regla de ES (``P_min = 3``, paciencia 2, mejora minima 1e-4) sobre
    el macro-F1 del ES-val (el 10 % de TRAIN): el ES no elige entre brazos. A1 recorre la pasada
    con el nucleo FTRL softmax (sesgo como coordenada ``x0 = 1.0``) y registra por pasada las
    magnitudes que gobiernan las gates nuevas G7 (escasez efectiva de coordenadas con al menos una
    clase viva) y G8 (norma euclidea de la matriz de pesos); en el fold 0 verifica ademas el anclaje
    de ``lambda1`` con el nucleo de anclaje (``lambda1 = 0.0``) sobre el MISMO stream de la pasada
    1, exigiendo diferencia EXACTA 0 con la mediana congelada y sha256 del snapshot de ``z``
    identico al de la sonda pre-ejecucion. A2 se ajusta pasada a pasada con ``partial_fit`` sobre la
    MISMA matriz SIN columna de sesgo pre-anadida, de modo que su unico camino de sesgo es
    ``fit_intercept=True``; A2 no lleva L1 y por eso las magnitudes de G7 y G8 no le aplican.

    Parameters
    ----------
    brazo:
        Nombre del brazo.
    datos_fold:
        Salida de ``preparar_fold``.
    ordenes:
        Lista de ``P_MAX`` permutaciones de las filas de ES-train (compartida por los dos brazos).
    fold:
        Fold canonico.
    clase_va:
        Etiquetas reales del fold-val (solo para G3/G4 y el OOF; nunca para ajustar ni parar).

    Returns
    -------
    dict
        Trayectoria por pasada, pasada de parada, mejor pasada, pesos restaurados, sesgo,
        probabilidades del fold-val, macro-F1 standalone, frecuencias y tiempo. En A1, ademas,
        la magnitud L1 de cada pasada y la verificacion del anclaje en el fold 0.

    Raises
    ------
    Detencion
        Si el anclaje de lambda1 no reproduce en el fold 0 (codigo 2) o si la restauracion del
        mejor estado no reproduce su sha256 (codigo 2).
    """
    inicio = time.time()
    clave = "a1" if brazo == "A1_ftrl_lambda1" else "a2"
    x_train = datos_fold[f"x_{clave}_train"]
    x_es_val = datos_fold[f"x_{clave}_esval"]
    x_va = datos_fold[f"x_{clave}_va"]
    y_es_train = datos_fold["y_es_train"]
    y_es_val = datos_fold["y_es_val"]
    trayectoria: list[dict] = []
    metricas: list[float] = []
    mejor_valor = -math.inf
    if clave == "a1":
        nucleo = FtrlSoftmaxMulticlase(x_train.shape[1], datos_fold["alpha"])
        mejor_estado: dict = {}
        mejor_sha = ""
        anclaje: dict | None = None
        for indice, orden in enumerate(ordenes):
            stats = nucleo.pasar(x_train, y_es_train, orden)
            if fold == 0 and indice == 0:
                # Verificacion del anclaje EN RUNTIME sobre el mismo stream de la pasada 1: si el
                # lambda1 congelado no es la mediana medida (diferencia EXACTA 0) o el snapshot de z
                # no reproduce el sha256 de la sonda previa, el ciclo se detiene con exit 2 ANTES
                # de seguir entrenando.
                anclaje = verificar_anclaje_lambda1(x_train, y_es_train, orden,
                                                    datos_fold["alpha"], LAMBDA1)
                print(f"  anclaje de lambda1 verificado en la pasada 1 del fold 0: mediana "
                      f"{anclaje['mediana_abs_z_ancaje']!r} con diferencia exacta "
                      f"{anclaje['diferencia_exacta']!r}, sha256 {anclaje['sha256_z_ancaje']} "
                      f"(reproduce = {anclaje['reproduce']})", flush=True)
            magnitud = magnitud_l1(nucleo, LAMBDA1)
            ll_train, rango = logits_a_prob(np.asarray(x_train.dot(nucleo.w)), y_es_train)
            predicciones = softmax(np.asarray(x_es_val.dot(nucleo.w))).argmax(axis=1)
            valor_es = macro(y_es_val, predicciones)
            trayectoria.append({"pasada": indice + 1, "ll_train": ll_train, "D_acum": rango,
                                "macro_f1_es": valor_es, "w_max_abs": stats["w_max_abs"],
                                "razon_sigma_sobre_g": stats["razon_sigma_sobre_g"],
                                "razon_sigma_sobre_g_pasada": stats["razon_sigma_sobre_g_pasada"],
                                "perdida_online": stats["perdida_online"],
                                "norma_w_l2": magnitud["norma_w_l2"],
                                "escasez_efectiva": magnitud["escasez_efectiva"],
                                "frac_coordenadas_muertas": magnitud["frac_coordenadas_muertas"],
                                "frac_clases_muertas": magnitud["frac_clases_muertas"],
                                "coordenadas_vivas": magnitud["coordenadas_vivas"]})
            metricas.append(valor_es)
            if indice == 0 or valor_es > mejor_valor + MEJORA_MINIMA:
                mejor_valor = float(max(mejor_valor, valor_es))
                mejor_estado = nucleo.estado()
                mejor_sha = nucleo.sha256_pesos()
            if evaluar_es(metricas)["parar"]:
                break
        decision = evaluar_es(metricas)
        nucleo.cargar(mejor_estado["w"], mejor_estado["z"], mejor_estado["n"], mejor_estado["t"])
        if nucleo.sha256_pesos() != mejor_sha:
            raise Detencion(f"[STOP] la restauracion del mejor w de {brazo} fold {fold} no "
                            "reproduce su sha256 (exit 2)", 2)
        p_va = softmax(np.asarray(x_va.dot(nucleo.w)))
        p_es_val = softmax(np.asarray(x_es_val.dot(nucleo.w)))
        sesgo = {"n_caminos": 1,
                 "fuente": "coordenada de sesgo x0=1.0 dentro de FTRL (propio z, propio n)",
                 "max_abs_gradiente_coordenada_sesgo": float(nucleo.max_abs_g_sesgo),
                 "gradiente_acumulado_por_clase": [float(valor) for valor in nucleo.acc_g_sesgo],
                 "desplazamiento_por_clase": [float(nucleo.coord_sesgo()[c])
                                              for c in range(N_CLASES)],
                 "max_abs_desplazamiento": float(np.max(np.abs(nucleo.coord_sesgo()))),
                 "coord_sesgo_sha256": sha256_f64(nucleo.coord_sesgo()),
                 "sha256_w": nucleo.sha256_pesos()}
        guardar_pesos(fold, brazo, nucleo.w, nucleo.z, nucleo.n, decision["pasada_actual"])
        estado_pesos = nucleo.estado()
        w_final = nucleo.w
    else:
        modelo = ajustar_sgd(fold)
        mejor_sgd: dict = {}
        for indice, orden in enumerate(ordenes):
            lote, y_lote = fila_a_pasada(x_train, y_es_train, orden)
            if indice == 0:
                modelo.partial_fit(lote, y_lote, classes=np.arange(N_CLASES))
            else:
                modelo.partial_fit(lote, y_lote)
            ll_train, rango = logits_a_prob(logits_de_sgd(modelo, x_train), y_es_train)
            predicciones = modelo.predict(x_es_val)
            valor_es = macro(y_es_val, predicciones)
            trayectoria.append({"pasada": indice + 1, "ll_train": ll_train, "D_acum": rango,
                                "macro_f1_es": valor_es,
                                "w_max_abs": float(np.max(np.abs(modelo.coef_))),
                                "t_": float(modelo.t_)})
            metricas.append(valor_es)
            if indice == 0 or valor_es > mejor_valor + MEJORA_MINIMA:
                mejor_valor = float(max(mejor_valor, valor_es))
                mejor_sgd = {"coef": modelo.coef_.copy(), "intercept": modelo.intercept_.copy()}
            if evaluar_es(metricas)["parar"]:
                break
        decision = evaluar_es(metricas)
        modelo.coef_ = mejor_sgd["coef"]
        modelo.intercept_ = mejor_sgd["intercept"]
        p_va = modelo.predict_proba(x_va)
        p_es_val = modelo.predict_proba(x_es_val)
        sesgo = {"n_caminos": 1,
                 "fuente": "fit_intercept=True sobre la matriz SIN columna de sesgo pre-anadida "
                           "(un solo camino)",
                 "intercept_": [float(valor) for valor in modelo.intercept_],
                 "max_abs_intercept": float(np.max(np.abs(modelo.intercept_))),
                 "intercept_sha256": sha256_f64(modelo.intercept_),
                 "sha256_w": sha256_f64_2d(np.vstack([modelo.coef_.T,
                                                      modelo.intercept_.reshape(1, -1)]))}
        guardar_pesos(fold, brazo, modelo.coef_.T, np.zeros_like(modelo.coef_.T),
                      np.zeros_like(modelo.coef_.T), decision["pasada_actual"],
                      extra={"intercept_": modelo.intercept_})
        estado_pesos = {"w": modelo.coef_.T.copy(), "z": np.zeros_like(modelo.coef_.T),
                        "n": np.zeros_like(modelo.coef_.T), "t": int(modelo.t_)}
        w_final = modelo.coef_.T
    predicciones_va = np.argmax(p_va, axis=1)
    return {"brazo": brazo, "trayectoria": trayectoria,
            "pasada_de_parada": int(decision["pasada_actual"]),
            "mejor_pasada": int(decision["mejor_pasada"]),
            "mejor_macro_f1_es": float(decision["mejor_valor"]),
            "alpha": float(datos_fold["alpha"]),
            "ordenes_sha256": [sha256_i64(orden) for orden in ordenes],
            "n_actualizaciones": int(estado_pesos["t"]),
            "sesgo": sesgo, "p_va": p_va, "w": w_final,
            "lambda1": LAMBDA1, "anclaje_lambda1": anclaje,
            "macro_f1_standalone_fold_val": float(macro(clase_va, predicciones_va)),
            "frecuencias_fold_val": frecuencias(predicciones_va),
            "max_abs_suma_menos_uno": float(max(max_abs_suma_menos_uno(p_va),
                                                max_abs_suma_menos_uno(p_es_val))),
            "tiempo_seg": float(time.time() - inicio)}


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
    dos brazos en el fold 0.

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
    if "0" not in estado["brazos"]["A1_ftrl_lambda1"]["folds"]:
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
# ---- 8. PUERTA G0-G6 FAIL-FAST (FOLD 0) Y TRIAGE FORMAL ----
# =====================================================================================
def evaluar_puerta_preliminar(estado: dict, brazo: str) -> dict:
    """Evalua G0, G1, G2, G3, G4 y G6 de un brazo con las magnitudes ya registradas del fold 0.

    G5 (determinismo) NO se evalua aqui: exige una segunda corrida identica del fold 0, que solo
    se lanza si estos gates pasan (fail-fast: no se gasta CPU si ya se sabe que el veredicto es
    NO_ATRIBUIBLE_BY_GATE) o si hace falta para el triage formal.

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
    trayectoria = detalle["trayectoria"]
    final = trayectoria[-1]
    ll_final = float(final["ll_train"])
    rango = float(final["D_acum"])
    techo = float(detalle["h_prior_es"] - G1_TECHO_MARGEN)
    if brazo == "A1_ftrl_lambda1":
        g0_pasa = bool(G0_FACTOR_INFERIOR * M_MARGEN <= rango <= G0_FACTOR_SUPERIOR * M_MARGEN)
        g0 = {"criterio": "A1: M/2 <= D_acum <= 2*M con D_acum = rango medio de logits (max - min) "
                          "sobre ES-train al parar",
              "valor": rango, "rango": [G0_FACTOR_INFERIOR * M_MARGEN, G0_FACTOR_SUPERIOR * M_MARGEN],
              "M": M_MARGEN, "anclaje": "M = ln(5) separa la tasa base del uniforme; factor 2 "
                                        "declarado cubre la linealizacion de primer orden",
              "que_es_G0": "comprueba que la regla de alpha consiguio su objetivo declarado, no "
                           "solo que hubo perdida", "pass": g0_pasa}
    else:
        g0_pasa = bool(ll_final <= techo)
        g0 = {"criterio": "A2: ll_train <= H(prior)_ES - 0.01",
              "valor": ll_final, "techo": techo, "h_prior_es": float(detalle["h_prior_es"]),
              "anclaje": "prior MEDIDO del ES-train; el control sano debe aprender de verdad",
              "que_es_G0": "en A2 comprueba que el presupuesto no es el limite", "pass": g0_pasa}
    mejoras_g2 = 0
    for indice in range(1, min(P_MIN, len(trayectoria))):
        previas = [float(item["ll_train"]) for item in trayectoria[:indice]]
        if float(trayectoria[indice]["ll_train"]) < min(previas) - G2_TOLERANCIA:
            mejoras_g2 += 1
    g2_pasa = bool(mejoras_g2 >= 1)
    g3_pasa = bool(float(detalle["macro_f1_standalone_fold_val"]) >= G3_MINIMO)
    frec = [float(valor) for valor in detalle["frecuencias_fold_val"]]
    if "prior_es" not in detalle:
        raise Detencion("[STOP] el detalle del fold no trae el prior MEDIDO del ES-train, que G4 "
                        "exige como ancla (exit 2)", 2)
    prior = np.asarray(detalle["prior_es"], dtype=np.float64)
    if prior.size != N_CLASES:
        raise Detencion(f"[STOP] el prior MEDIDO del ES-train tiene {prior.size} componentes y no "
                        f"{N_CLASES} (exit 2)", 2)
    g4_cumple = all(G4_FACTOR_INFERIOR * float(prior[c]) <= frec[c]
                    <= G4_FACTOR_SUPERIOR * float(prior[c]) for c in range(N_CLASES))
    g6_pasa = bool(float(detalle["max_abs_suma_menos_uno"]) <= G6_TOLERANCIA)
    g1_pasa = bool(G1_PISO <= ll_final <= techo)
    return {
        NOMBRES_GATE[0]: g0,
        NOMBRES_GATE[1]: {"criterio": f"{G1_PISO} <= ll_train <= H(prior)_ES - {G1_TECHO_MARGEN}",
                          "valor": ll_final, "piso": G1_PISO, "techo": techo,
                          "h_prior_es": float(detalle["h_prior_es"]),
                          "anclaje": "techo anclado en la entropia del prior MEDIDO del ES-train, "
                                     "NO en ln 6", "pass": g1_pasa},
        NOMBRES_GATE[2]: {"criterio": "la mejor ll_train por pasada mejora en >= 1 de las primeras "
                                      "P_min = 3 pasadas",
                          "ll_train_por_pasada": [float(item["ll_train"]) for item in trayectoria],
                          "mejoras_en_las_primeras_P_min": mejoras_g2, "tolerancia": G2_TOLERANCIA,
                          "anclaje": "un modelo que no aprende nada tiene ll_train plana",
                          "pass": g2_pasa},
        NOMBRES_GATE[3]: {"criterio": f"macro-F1 standalone en el fold-val >= {G3_MINIMO}",
                          "valor": float(detalle["macro_f1_standalone_fold_val"]),
                          "minimo": G3_MINIMO,
                          "anclas": {"trivial": ANCLA_TRIVIAL, "gemelo_sgd_d083": ANCLA_GEMELO_SGD_D083},
                          "pass": g3_pasa},
        NOMBRES_GATE[4]: {"criterio": "frecuencia predicha por clase en [0.5*prior_c, 1.5*prior_c]",
                          "valores": frec, "prior_es": prior.tolist(),
                          "rango_por_clase": [[G4_FACTOR_INFERIOR * float(prior[c]),
                                               G4_FACTOR_SUPERIOR * float(prior[c])]
                                              for c in range(N_CLASES)],
                          "anclaje": "prior MEDIDO del ES-train", "pass": bool(g4_cumple)},
        NOMBRES_GATE[5]: {"criterio": "max|delta p| = 0.0 EXACTO entre dos corridas identicas",
                          "valor": None, "tolerancia": G5_TOLERANCIA, "pass": None,
                          "estado": "no evaluado: exige la segunda corrida identica del fold 0"},
        NOMBRES_GATE[6]: {"criterio": f"max|sum_c p_c - 1| <= {G6_TOLERANCIA}",
                          "valor": float(detalle["max_abs_suma_menos_uno"]),
                          "tolerancia": G6_TOLERANCIA, "anclaje": "softmax de 6 logits",
                          "pass": g6_pasa},
    }


def cerrar_puerta(puerta_preliminar: dict, determinismos: dict, evaluadas_g5: bool) -> dict:
    """Cierra la puerta G0-G6 con G5 y decide el veredicto de la puerta.

    Parameters
    ----------
    puerta_preliminar:
        Gates G0-G4 y G6 ya evaluados por brazo.
    determinismos:
        Sondas de determinismo por brazo (segunda corrida identica del fold 0).
    evaluadas_g5:
        True si G5 se ha medido, False si el ciclo se detuvo antes (fail-fast).

    Returns
    -------
    dict
        Gates por brazo, ``pasa`` global, brazos fallidos y el detalle del fold 0 consumido.
    """
    brazos = dict(puerta_preliminar)
    for brazo in BRAZOS:
        info = brazos[brazo]
        if evaluadas_g5:
            valor = float(determinismos[brazo]["max_abs_diff_p"])
            info[NOMBRES_GATE[5]] = {"criterio": "max|delta p| = 0.0 EXACTO entre dos corridas "
                                                "identicas del fold 0 (ambos brazos)",
                                     "valor": valor, "tolerancia": G5_TOLERANCIA,
                                     "sha256_w": determinismos[brazo]["sha256_w"],
                                     "pasada_de_parada": determinismos[brazo]["pasada_de_parada"],
                                     "ll_train": determinismos[brazo]["ll_train"],
                                     "pass": bool(valor == G5_TOLERANCIA)}
        else:
            info[NOMBRES_GATE[5]]["estado"] = ("no evaluado: el ciclo se detuvo por fail-fast en "
                                               "G0-G4/G6")
    fallidos = {brazo: [nombre for nombre, info in brazos[brazo].items() if not info["pass"]]
                for brazo in BRAZOS}
    pasa = bool(evaluadas_g5 and all(not fallidos[brazo] for brazo in BRAZOS))
    return {"momento": "dentro del fold 0, ANTES de los folds 1-4 y ANTES de integrar",
            "aplica_a": list(BRAZOS),
            "fold_0_consumido_diagnostico": True,
            "nota_fold_0": "G3 y G4 leen el fold-val del fold 0, que es parte de la validacion "
                           "canonica; su fallo NO toca delta",
            "gates": brazos, "brazos_fallidos": fallidos, "pasa": pasa,
            "g5_evaluada": bool(evaluadas_g5),
            "consecuencia_si_falla": "veredicto NO_ATRIBUIBLE_BY_GATE: STOP documentado, sin "
                                     "fallback, sin segundo alpha, sin integrar, sin bootstrap",
            "prohibido": "usar fold-val, la particion reservada o el leaderboard para parar, "
                         "seleccionar o comparar brazos; reintentar con otro alpha tras el fallo "
                         "(seria grid)"}


def tabla_simetria_sesgo(estado: dict) -> dict:
    """TABLA DE SIMETRIA DE SESGO exigida por la enmienda pre-ejecucion 1.

    Publica, por brazo, que hay UN SOLO camino de sesgo, y enfrenta el maximo absoluto del
    gradiente de la coordenada de sesgo de A1 (y su desplazamiento acumulado por clase) con el
    ``intercept_`` de A2. Declara explicitamente que NO hay doble sesgo en ningun brazo.

    Parameters
    ----------
    estado:
        Checkpoint con los folds registrados.

    Returns
    -------
    dict
        Detalle por brazo y por fold, la comparacion de magnitudes, el veredicto de paridad de
        orden de magnitud y la declaracion de no-doble-sesgo.
    """
    detalle: dict[str, dict] = {}
    for brazo in BRAZOS:
        sesgos = {fold: registro["sesgo"] for fold, registro in
                  estado["brazos"][brazo]["folds"].items()}
        caminos = {int(valor["n_caminos"]) for valor in sesgos.values()}
        detalle[brazo] = {"fuente": sorted({valor["fuente"] for valor in sesgos.values()}),
                          "n_caminos_de_sesgo": sorted(caminos), "sesgo_por_fold": sesgos}
    a1 = estado["brazos"]["A1_ftrl_lambda1"]["folds"]
    a2 = estado["brazos"]["A2_sgd_presupuesto"]["folds"]
    comparacion: dict[str, dict] = {}
    for fold in sorted(a1, key=int):
        sesgo_a1 = a1[fold]["sesgo"]
        sesgo_a2 = a2[fold]["sesgo"]
        max_a1 = float(sesgo_a1["max_abs_gradiente_coordenada_sesgo"])
        max_a2 = float(sesgo_a2["max_abs_intercept"])
        comparacion[fold] = {
            "A1_max_abs_gradiente_coordenada_sesgo": max_a1,
            "A1_desplazamiento_por_clase": list(sesgo_a1["desplazamiento_por_clase"]),
            "A1_max_abs_desplazamiento": float(sesgo_a1["max_abs_desplazamiento"]),
            "A2_intercept": list(sesgo_a2["intercept_"]),
            "A2_max_abs_intercept": max_a2,
            "razon_A1_sobre_A2": float(max_a1 / max_a2) if max_a2 > 0 else None,
            "razon_desplazamiento_sobre_intercept": (
                float(sesgo_a1["max_abs_desplazamiento"] / max_a2) if max_a2 > 0 else None),
            "mismo_orden_de_magnitud": bool(
                max_a1 > 0 and max_a2 > 0
                and 1e-6 <= (max_a1 / max_a2 if max_a2 > 0 else 0.0) <= 1e6),
        }
    return {"descripcion": "tabla de simetria de sesgo exigida por la enmienda pre-ejecucion 1: UN "
                           "solo camino de sesgo por brazo, verificado por sonda",
            "A1_camino": "coordenada de sesgo x0=1.0 materializada dentro de FTRL (propio z, "
                         "propio n, sin termino de sesgo aparte)",
            "A2_camino": "fit_intercept=True sobre la MISMA matriz SIN columna de sesgo "
                         "pre-anadida (enmienda 1: se corrige el doble sesgo de D-083/V33-E1)",
            "doble_sesgo": False,
            "declaracion": "no hay doble sesgo en ningun brazo: A1 materializa el sesgo como "
                           "coordenada unica y A2 lo aporta solo por intercept_; el defecto de "
                           "D-083 y V33-E1 (matriz con sesgo pre-anadido Y fit_intercept=True) NO "
                           "se reproduce aqui",
            "por_brazo": detalle, "comparacion_por_fold": comparacion}


def triage_formal(estado: dict, sondas: dict, puerta: dict) -> dict:
    """TRIAGE FORMAL: clasifica el fallo como DEFECTO_DE_IMPLEMENTACION o MECANISMO_DEFINITIVO.

    Solo se ejecuta cuando A1 falla la puerta y A2 pasa G0-G4. La checklist es EXPLICITA y cada
    comprobacion lee un valor del estado ya escrito (no una declaracion): sondas (a)-(i) en
    verde, sesgo unico por brazo, presupuesto no truncado, ES bien configurado, determinismo
    exacto, alpha* reproducible y ausencia de un segundo alpha o fallback. Si CUALQUIER
    indicador de defecto se activa, el nodo de cierre por mecanismo NO se concede.

    Parameters
    ----------
    estado:
        Checkpoint con el fold 0 de los dos brazos.
    sondas:
        Detalle de las sondas (a)-(j).
    puerta:
        Puerta G0-G6 ya evaluada.

    Returns
    -------
    dict
        Checklist con valor y criterio por comprobacion, clasificacion, si se concede el nodo y
        el motivo de no calculo.
    """
    a1 = estado["brazos"]["A1_ftrl_lambda1"]["folds"]["0"]
    a2 = estado["brazos"]["A2_sgd_presupuesto"]["folds"]["0"]
    sondas_abcde = all(bool(sondas[nombre]["pass"]) for nombre in NOMBRES_SONDAS[:5])
    sondas_fghi = all(bool(sondas[nombre]["pass"]) for nombre in NOMBRES_SONDAS[5:9])
    sesgo_unico = bool(a1["sesgo"]["n_caminos"] == 1 and a2["sesgo"]["n_caminos"] == 1
                       and a1["n_coordinates_a1"] == a1["n_vocabulario"] + 1
                       and a1["n_features_es_train"] == a1["n_vocabulario"]
                       and a2["sesgo"]["fuente"].startswith("fit_intercept=True"))
    presupuesto = bool(a1["pasada_de_parada"] <= P_MAX and a2["pasada_de_parada"] <= P_MAX
                       and a1["pasada_de_parada"] >= P_MIN and a2["pasada_de_parada"] >= P_MIN
                       and len(a1["trayectoria"]) == a1["pasada_de_parada"]
                       and len(a2["trayectoria"]) == a2["pasada_de_parada"]
                       and a1["ordenes_sha256"] == a2["ordenes_sha256"]
                       and a1["es_train_sha256"] == a2["es_train_sha256"])
    es_configurado = bool(a1["mejor_pasada"] <= a1["pasada_de_parada"]
                          and a2["mejor_pasada"] <= a2["pasada_de_parada"]
                          and a1["es_train_sha256"] == a2["es_train_sha256"]
                          and a1["es_val_sha256"] == a2["es_val_sha256"]
                          and a1["n_filas_fit_es_train"] == a2["n_filas_fit_es_train"]
                          and a1["n_filas_es_val"] == a2["n_filas_es_val"])
    determinismo = bool(puerta.get("g5_evaluada", False)
                        and all(float(puerta["gates"][brazo][NOMBRES_GATE[5]]["valor"]) == 0.0
                                for brazo in BRAZOS))
    alpha_ok = bool(sondas["g_reproducibilidad_alpha"]["diferencia_es_cero_exacto"]
                    and a1["alpha_sha256"] == a2["alpha_sha256"])
    un_alpha = bool(a1["alpha_efectivo"] == a2["alpha_efectivo"]
                    and abs(a1["alpha_efectivo"]
                            - 2.0 * M_MARGEN / (a1["d_norm"] * (P_MAX ** 2))) == 0.0)
    checklist = {
        "D1_sondas_abcde_en_verde": {
            "valor": sondas_abcde,
            "criterio": "sondas (a)-(e) con pass = True (vectorizado vs lazo, raiz analitica, "
                        "TotalF1 Macro, sin NaN, log_loss propio vs sklearn)"},
        "D2_sondas_fghi_en_verde": {
            "valor": sondas_fghi,
            "criterio": "sondas (f)-(i) con pass = True (filas que suman 1, reproducibilidad de "
                        "alpha*, estabilidad del ES, simetria de presupuesto)"},
        "D3_sesgo_unico_por_brazo": {
            "valor": sesgo_unico,
            "criterio": "A1 con una sola coordenada de sesgo (n_features == n_vocabulario, "
                        "n_coordinates_a1 == n_vocabulario + 1) y A2 con fit_intercept=True sobre "
                        "la matriz sin columna de sesgo"},
        "D4_presupuesto_no_truncado": {
            "valor": presupuesto,
            "criterio": f"pasada_de_parada en [P_min={P_MIN}, P_max={P_MAX}], Trayectoria con "
                        "tantas entradas como pasadas y ordenes/particion ES identicos en A1 y A2"},
        "D5_es_bien_configurado": {
            "valor": es_configurado,
            "criterio": "mejor_pasada <= pasada_de_parada (se restaura el mejor w) y particion ES "
                        "identica en ambos brazos, ajustada SOLO en ES-train; la metrica del ES es "
                        "el macro-F1 del 10 % de TRAIN, nunca fold-val"},
        "D6_determinismo_exacto": {
            "valor": determinismo,
            "criterio": "G5 evaluado con max|delta p| = 0.0 EXACTO en los dos brazos (sonda h)"},
        "D7_alpha_reproducible": {
            "valor": alpha_ok,
            "criterio": "sonda (g) con diferencia exacta 0 entre las dos rutas de alpha* y mismo "
                        "sha256 de alpha* en los dos brazos"},
        "D8_sin_segundo_alpha_ni_fallback": {
            "valor": un_alpha,
            "criterio": "un unico alpha por fold, identico en A1 y A2, igual a la forma cerrada "
                        "2*M/(d_norm*P_max^2) al cero binario: no hay grid ni reintento"},
    }
    # Cada `valor` de la checklist vale True cuando la comprobacion PASSA, es decir cuando NO hay
    # indicador de defecto; un indicador de defecto es por tanto un valor False. Sin este "not" la
    # clasificacion seria siempre DEFECTO_DE_IMPLEMENTACION y el nodo seria inalcanzable.
    hay_defecto = any(not bool(item["valor"]) for item in checklist.values())
    mecanismo_publicado = bool(estado.get("mecanismo"))
    clasificacion = "DEFECTO_DE_IMPLEMENTACION" if hay_defecto else "MECANISMO_DEFINITIVO"
    concedido = bool((not hay_defecto) and mecanismo_publicado)
    return {"se_ejecuta_si": "A1 falla la puerta y A2 pasa G0-G4",
            "checklist": checklist, "hay_indicador_de_defecto": hay_defecto,
            "clasificacion": clasificacion,
            "mecanismo_publicado_dentro_del_runner": mecanismo_publicado,
            "nodo_concedido": concedido,
            "motivo_de_no_calculo": (
                None if concedido else
                ("se detecta un indicador de DEFECTO_DE_IMPLEMENTACION: hay que corregir y "
                 "re-emitir el runner; el nodo de mecanismo NO se concede"
                 if hay_defecto else
                 "la tabla de mecanismo no esta publicada dentro del runner")),
            "uno_shot": "reabrir la familia exige acta nueva con hipotesis mecanica nueva, no otro "
                        "gate"}


def refit_vectorizador(textos: np.ndarray, clase: np.ndarray, fold_idx: np.ndarray,
                       estado: dict) -> dict:
    """Sonda de refit: reajusta el vectorizador en los 5 folds y compara los hashes bit a bit.

    No reentrena ningun modelo ni escribe ningun OOF: solo comprueba que el vocabulario ordenado
    y el idf de cada fold se reproducen identicos, ajustados SIEMPRE sobre las mismas filas de
    ES-train con las que se entreno.

    Parameters
    ----------
    textos:
        Textos del corpus en orden canonico.
    clase:
        Etiquetas canonicas.
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
        es_train, _es_val, _hashes = prior_de_es_train(clase, fold_idx, fold)
        vectorizador = TfidfVectorizer(**VECTORIZADOR_FIT)
        vectorizador.fit_transform(textos[es_train])
        guardado = estado["vectorizador_por_fold"].get(str(fold), {})
        vocab = sha256_textos(sorted(vectorizador.vocabulary_))
        idf = sha256_f64(vectorizador.idf_)
        coincide = bool(vocab == guardado.get("vocab_sha256") and idf == guardado.get("idf_sha256"))
        todo_ok = todo_ok and coincide
        detalle[str(fold)] = {"n_vocabulario": len(vectorizador.vocabulary_),
                              "vocab_sha256": vocab, "idf_sha256": idf,
                              "vocab_sha256_corrida": guardado.get("vocab_sha256"),
                              "idf_sha256_corrida": guardado.get("idf_sha256"),
                              "n_filas_fit_es_train": int(es_train.size), "reproducido": coincide}
        print(f"  refit fold {fold}: vocab e idf de ES-train reproducidos bit a bit = {coincide}",
              flush=True)
    return {"descripcion": "reajustar el vectorizador en los 5 folds (SIEMPRE sobre ES-train) y "
                           "comprobar que los sha256 de vocabulario e idf coinciden con los de la "
                           "corrida", "tolerancia": "identicos bit a bit entre las dos corridas",
            "folds": detalle, "reproducido": todo_ok}


def entrenar_con_puerta(estado: dict, textos: np.ndarray, clase: np.ndarray,
                        fold_idx: np.ndarray) -> tuple[dict[str, np.ndarray], dict | None]:
    """Entrena los 2 brazos x 5 folds con la puerta G0-G6 fail-fast en el fold 0.

    Por fold se ajusta el vectorizador UNA vez sobre ES-train (compartido por los dos brazos), se
    genera la lista de ordenes UNA vez y se pasa a ambos brazos, y luego se recorren los brazos en
    orden A1, A2. Al terminar el fold 0 se evalua la puerta completa: G0-G4 y G6 con las
    magnitudes ya registradas y, si pasan (o si hace falta para el triage), G5 con una segunda
    corrida identica del fold 0.

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
        verificar_ram(estado)
        _tr_idx, va_idx = indices_fold(fold_idx, fold)
        clase_va = clase[va_idx].astype(np.int64)
        datos_fold = preparar_fold(textos, clase, fold_idx, fold)
        hashes_fold = dict(datos_fold["hashes"])
        estado["es_por_fold"][str(fold)] = {
            "es_train_sha256": hashes_fold["es_train_sha256"],
            "es_val_sha256": hashes_fold["es_val_sha256"],
            "n_filas_fit_es_train": hashes_fold["n_filas_fit_es_train"],
            "n_filas_es_val": hashes_fold["n_filas_es_val"],
            "n_train_fold": hashes_fold["n_train_fold"],
            "prior": hashes_fold["prior"], "prior_sha256": hashes_fold["prior_sha256"],
            "prior_es": hashes_fold["prior"],
            "d_norm": hashes_fold["d_norm"], "h_prior_es": hashes_fold["h_prior_es"],
            "alpha": hashes_fold["alpha"], "alpha_sha256": hashes_fold["alpha_sha256"]}
        estado["vectorizador_por_fold"][str(fold)] = {
            "n_vocabulario": hashes_fold["n_vocabulario"],
            "vocab_sha256": hashes_fold["vocab_sha256"], "idf_sha256": hashes_fold["idf_sha256"],
            "n_filas_fit_es_train": hashes_fold["n_filas_fit_es_train"]}
        guardar_estado(estado)
        ordenes = generar_ordenes(fold, hashes_fold["n_filas_fit_es_train"])
        print(f"  [fold {fold}] TF-IDF ajustado SOLO en ES-train ({hashes_fold['n_filas_fit_es_train']}"
              f" filas, vocab {hashes_fold['n_vocabulario']}, sha256 {hashes_fold['vocab_sha256'][:12]}"
              f"); ES-val {hashes_fold['n_filas_es_val']} filas (sha256 "
              f"{hashes_fold['es_val_sha256'][:12]}); prior sha256 "
              f"{hashes_fold['prior_sha256'][:12]}; alpha* {hashes_fold['alpha']!r} (sha256 "
              f"{hashes_fold['alpha_sha256'][:12]})", flush=True)
        pendientes = [brazo for brazo in BRAZOS if str(fold) not in estado["brazos"][brazo]["folds"]]
        for brazo in pendientes:
            verificar_deadline(estado)
            resultado = entrenar_brazo_fold(brazo, datos_fold, ordenes, fold, clase_va)
            del resultado["w"]
            registrar_fold(estado, brazo, fold, resultado, oofs[brazo], hashes_fold, va_idx,
                           clase_va)
            verificar_ram(estado)
        if fold == 0:
            estado["mecanismo"] = {"fold_0": True, "nota": "tabla de mecanismo por pasada de A1 y "
                                                          "A2 del fold 0 (sonda j)",
                                  "brazos": {brazo: estado["brazos"][brazo]["folds"]["0"][
                                      "trayectoria"] for brazo in BRAZOS},
                                  "h_prior_es": hashes_fold["h_prior_es"],
                                  "M": M_MARGEN, "alpha": hashes_fold["alpha"]}
            guardar_estado(estado)
            if tr_idx_tamano(datos_fold) != ANCLA_N_TRAIN_FOLD:
                raise Detencion(f"[STOP] el fold-train del fold 0 tiene "
                                f"{tr_idx_tamano(datos_fold)} filas, distinto del ancla "
                                f"{ANCLA_N_TRAIN_FOLD} (exit 2)", 2)
            preliminar = {brazo: evaluar_puerta_preliminar(estado, brazo) for brazo in BRAZOS}
            a1_pasa = all(bool(info["pass"]) for nombre, info in preliminar[
                "A1_ftrl_lambda1"].items() if nombre != NOMBRES_GATE[5])
            a2_pasa = all(bool(info["pass"]) for nombre, info in preliminar[
                "A2_sgd_presupuesto"].items() if nombre != NOMBRES_GATE[5])
            if a1_pasa and a2_pasa:
                determinismos = sonda_determinismo_fold0(estado, datos_fold, ordenes, clase_va,
                                                         oofs)
                puerta = cerrar_puerta(preliminar, determinismos, True)
            else:
                if (not a1_pasa) and a2_pasa:
                    determinismos = sonda_determinismo_fold0(estado, datos_fold, ordenes, clase_va,
                                                             oofs)
                    puerta = cerrar_puerta(preliminar, determinismos, True)
                else:
                    determinismos = {}
                    puerta = cerrar_puerta(preliminar, determinismos, False)
            estado["puerta"] = puerta
            guardar_estado(estado)
            for brazo in BRAZOS:
                for nombre, info in puerta["gates"][brazo].items():
                    if not info["pass"]:
                        print(f"  [GATE] {brazo} falla {nombre}: {info['criterio']} -> "
                              f"{info.get('valor')}", flush=True)
            if not puerta["pasa"]:
                return oofs, puerta
            print("  [GATE] puerta G0-G6 superada en A1 y A2 (fold 0 consumido-diagnostico); "
                  "se continua con los folds 1-4", flush=True)
            asegurar_calibracion(estado)
        del datos_fold
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


def tr_idx_tamano(datos_fold: dict) -> int:
    """Numero de filas del fold-train del fold preparado (ancla de integridad).

    Parameters
    ----------
    datos_fold:
        Salida de ``preparar_fold``.

    Returns
    -------
    int
        Numero de filas de fold-train.
    """
    return int(datos_fold["hashes"]["n_train_fold"])


def sonda_determinismo_fold0(estado: dict, datos_fold: dict, ordenes: list[np.ndarray],
                             clase_va: np.ndarray, oofs: dict[str, np.ndarray]) -> dict:
    """Segunda corrida identica del fold 0 de cada brazo: es el gate G5 y la sonda (h).

    Reentrena el fold 0 desde cero con la MISMA config, la misma particion ES y los mismos ordenes
    congelados, y compara con tolerancia CERO la pasada de parada, el ``ll_train`` final, el sha256
    de los pesos y ``max|delta p|`` de las probabilidades del fold-val.

    Parameters
    ----------
    estado:
        Checkpoint.
    datos_fold:
        Salida de ``preparar_fold`` del fold 0.
    ordenes:
        Ordenes congelados del fold 0.
    clase_va:
        Etiquetas del fold-val del fold 0.
    oofs:
        OOF por brazo (para tomar la referencia de la primera corrida).

    Returns
    -------
    dict
        Detalle por brazo con ``max_abs_diff_p``, ``sha256_w``, ``pasada_de_parada``, ``ll_train``
        y ``pass``.
    """
    va_idx = datos_fold["va_idx"]
    detalle: dict[str, dict] = {}
    for brazo in BRAZOS:
        verificar_deadline(estado)
        repetida = entrenar_brazo_fold(brazo, datos_fold, ordenes, 0, clase_va)
        referencia = oofs[brazo][va_idx]
        max_abs_diff = float(np.abs(repetida["p_va"] - referencia).max())
        original = estado["brazos"][brazo]["folds"]["0"]
        estado["brazos"][brazo]["sonda_determinismo"] = {
            "max_abs_diff_repeticion": max_abs_diff, "max_abs_diff_p": max_abs_diff,
            "sha256_w": repetida["sesgo"]["sha256_w"],
            "sha256_w_primera": original["sesgo"]["sha256_w"],
            "pasada_de_parada": int(repetida["pasada_de_parada"]),
            "pasada_de_parada_primera": int(original["pasada_de_parada"]),
            "ll_train": float(repetida["trayectoria"][-1]["ll_train"]),
            "ll_train_primera": float(original["trayectoria"][-1]["ll_train"]),
            "clasicacion": "exacto" if max_abs_diff == 0.0 else "rango_publicado",
            "tolerancia_declarada": TOL_DETERMINISMO, "variabilidad_referencia": VARIABILIDAD,
            "gate_delta": GATE_DELTA,
            "no_concluyente": bool(max_abs_diff >= VARIABILIDAD or max_abs_diff >= GATE_DELTA),
            "solape_con_gate": bool(max_abs_diff >= GATE_DELTA),
            "sha256_w_identico": bool(repetida["sesgo"]["sha256_w"]
                                      == original["sesgo"]["sha256_w"]),
            "pasada_identica": bool(repetida["pasada_de_parada"] == original["pasada_de_parada"]),
            "ll_train_identico": bool(repetida["trayectoria"][-1]["ll_train"]
                                      == original["trayectoria"][-1]["ll_train"]),
            "pass": bool(max_abs_diff == G5_TOLERANCIA
                         and repetida["sesgo"]["sha256_w"] == original["sesgo"]["sha256_w"])}
        detalle[brazo] = estado["brazos"][brazo]["sonda_determinismo"]
        print(f"  sonda de determinismo {brazo} fold 0: max|delta p| = {max_abs_diff:.3e} "
              f"({detalle[brazo]['clasicacion']})", flush=True)
        guardar_estado(estado)
    return detalle


# =====================================================================================
# ---- 9. EVALUACION: INTEGRACION P6/P7, COMPLEMENTARIEDAD Y BOOTSTRAP ----
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
    for indice in range(N_BOOTSTRAP):
        idx = generador.integers(0, n, n)
        deltas[indice] = (macro_de_conteos(np.bincount(codigo_cand[idx], minlength=N_CLASES ** 2))
                          - macro_de_conteos(np.bincount(codigo_base[idx],
                                                         minlength=N_CLASES ** 2)))
    bajo, alto = np.percentile(deltas, [2.5, 97.5])
    return {"delta_medio": float(deltas.mean()), "ic95": [float(bajo), float(alto)],
            "p_delta_positivo": float((deltas > 0).mean()),
            "ic95_sin_cero": bool(bajo > 0.0 or alto < 0.0)}


def combinador_oof(v30: np.ndarray, p_brazo: np.ndarray, clase: np.ndarray,
                   fold_idx: np.ndarray) -> tuple[np.ndarray, dict]:
    """Combinador logistico anidado congelado sobre [L2(v30)||L2(p_brazo)] (12 features).

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
    Detencion
        Si el combinador de algun fold no ve las 6 clases en orden canonico (codigo 2).
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
            raise Detencion("[STOP] el combinador no vio las 6 clases en orden canonico (exit 2)", 2)
        p[va] = modelo.predict_proba(caracteristicas[va])
        coeficientes = np.abs(modelo.coef_)
        ratio_por_fold.append(float(coeficientes[:, N_CLASES:].mean()
                                    / max(coeficientes[:, :N_CLASES].mean(), L2_CLIP)))
        iteraciones.append(int(np.max(modelo.n_iter_)))
    return normaliza(p), {"ratio_bloque_brazo_por_fold": ratio_por_fold,
                          "n_iter_por_fold": iteraciones,
                          "mecanismo": "[L2(v30)||L2(p_brazo)] con LogReg(C=1.0, max_iter=2000) "
                                       "por fold; decision argmax(p_comb * w_plugin congelado)"}


def evaluar_combinacion(p_comb: np.ndarray, clase: np.ndarray, fold_idx: np.ndarray,
                        champ_pred: np.ndarray, pesos: np.ndarray) -> dict:
    """Metricas del combinador con la regla congelada (argmax p_comb * pesos) y diagnosticos.

    Returns
    -------
    dict
        macro-F1 global, deltas vs baseline y nested, folds y deltas por fold, folds
        positivos, IC95 del bootstrap, ``argmax_sin_regla``, filas que cambian y la
        prediccion combinada (no serializable: el llamante la retira antes del JSON).
    """
    pred = np.argmax(p_comb * pesos, axis=1)
    cv = macro(clase, pred)
    folds = [macro(clase[fold_idx == k], pred[fold_idx == k]) for k in range(N_FOLDS)]
    folds_base = [macro(clase[fold_idx == k], champ_pred[fold_idx == k]) for k in range(N_FOLDS)]
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
        macro-F1 standalone global y por fold, F1 por clase, matriz ``E[p[:, j] | y = i]`` con su
        diagonal, los dos log_loss y rescate/dano/neto globales y por clase.
    """
    pred = np.argmax(p_brazo, axis=1)
    f1_por_clase = f1_score(clase, pred, average=None, labels=list(range(N_CLASES)))
    ok_cand = pred == clase
    ok_campeon = champ_pred == clase
    alta_confianza = (~ok_campeon) & (conf >= UMBRAL_ALTA_CONFIANZA)
    por_clase = []
    for indice in range(N_CLASES):
        mascara = clase == indice
        por_clase.append({
            "clase": cfg.CLASSES[indice], "filas": int(mascara.sum()),
            "f1": float(f1_por_clase[indice]),
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
        Chequeos booleanos: alineacion, folds canonicos, folds completos, mascara 1 por fila, OOF
        finito y con filas que suman 1, vocabulario e idf registrados, vectorizador ajustado solo
        en ES-train, particion ES estratificada y reproducible, presupuesto multi-pasada
        respetado, un solo camino de sesgo por brazo, refit reproducible y ordenes compartidos.
    """
    completos, mascaras, finitos, suman_uno, hashes_ok, filas_fit, coordenadas, sumas = (
        [], [], [], [], [], [], [], [])
    presupuesto, sesgo_unico, es_ok, ordenes_ok = [], [], [], []
    for brazo in BRAZOS:
        datos = estado["brazos"][brazo]
        completos.append(datos["folds_completados"] == [str(fold) for fold in range(N_FOLDS)])
        mascara = np.asarray(datos["mascara_relleno_por_fila"], dtype=np.int64)
        mascaras.append(bool((mascara == 1).all()))
        finitos.append(bool(np.isfinite(oofs[brazo]).all()))
        suman_uno.append(bool(max_abs_suma_menos_uno(oofs[brazo]) <= G6_TOLERANCIA))
        for fold in range(N_FOLDS):
            registro = datos["folds"][str(fold)]
            hashes_ok.append(len(registro["vocab_sha256"]) == 64
                             and len(registro["idf_sha256"]) == 64
                             and len(registro["es_train_sha256"]) == 64
                             and len(registro["es_val_sha256"]) == 64)
            filas_fit.append(registro["n_filas_fit_es_train"] < registro["n_train_fold"]
                             and registro["n_filas_fit_es_train"] + registro["n_filas_es_val"]
                             == registro["n_train_fold"])
            coordenadas.append(registro["n_coordinates_a1"] == registro["n_vocabulario"] + 1)
            sumas.append(registro["max_abs_suma_menos_uno"] <= G6_TOLERANCIA)
            presupuesto.append(registro["pasada_de_parada"] <= P_MAX
                               and registro["mejor_pasada"] <= registro["pasada_de_parada"])
            sesgo_unico.append(registro["sesgo"]["n_caminos"] == 1)
            es_ok.append(registro["n_filas_es_val"] == round(TEST_SIZE_ES
                                                            * registro["n_train_fold"]))
            ordenes_ok.append(registro["ordenes_sha256"]
                              == estado["brazos"][BRAZOS[0] if brazo != BRAZOS[0] else
                                                  BRAZOS[1]]["folds"][str(fold)]["ordenes_sha256"])
    return {
        "alineacion_filas": bool(len(clase) == N_FILAS),
        "cinco_folds_canonicos": bool(np.unique(fold_idx).size == N_FOLDS),
        "folds_completos_por_brazo": bool(all(completos)),
        "oof_mascara_relleno_uno_por_fila": bool(all(mascaras)),
        "oof_sin_no_finitos": bool(all(finitos)),
        "oof_filas_suman_uno": bool(all(suman_uno)),
        "vocabulario_idf_y_particion_es_registrados_por_fold": bool(all(hashes_ok)),
        "vectorizador_ajustado_solo_en_es_train": bool(all(filas_fit)),
        "n_coordinates_a1_coherente_vocab_mas_sesgo": bool(all(coordenadas)),
        "suma_unidad_por_fold": bool(all(sumas)),
        "presupuesto_multipasada_respetado": bool(all(presupuesto)),
        "un_solo_camino_de_sesgo_por_brazo": bool(all(sesgo_unico)),
        "particion_es_al_10_por_ciento": bool(all(es_ok)),
        "ordenes_compartidos_entre_brazos": bool(all(ordenes_ok)),
        "refit_reproduce_vocabulario_e_idf_en_5_folds": bool(refit["reproducido"]),
    }


# =====================================================================================
# ---- 10. GATES, CAPS Y VEREDICTO ----
# =====================================================================================
def calcular_gates(baseline_ok: bool, integracion: dict, invariantes: dict,
                   determinismo: dict, sondas: dict, tabla_sesgo: bool) -> tuple[dict, dict, str]:
    """Aplica los gates congelados, los caps y el arbol de decision.

    Gates: baseline recomputado, delta de A1 >= +0.0015, folds positivos >= 4/5, control
    favorable A1 sobre A2 (delta puntual), IC95 del delta de A1 sin cero, sin leakage,
    determinismo concluyente y sondas (a)-(j) publicadas. Caps: si el determinismo no es
    concluyente el veredicto no puede ser PROMISING.

    Parameters
    ----------
    baseline_ok:
        Resultado del chequeo del baseline.
    integracion:
        Metricas del combinador por brazo.
    invariantes:
        Evidencias de no leakage.
    determinismo:
        Bloque de determinismo con la sonda por brazo y su veredicto ``no_concluyente`` (el que
        construye la segunda corrida identica del fold 0; nunca se deriva de las sondas mini).
    sondas:
        Detalle de las sondas (a)-(j).
    tabla_sesgo:
        Tabla de simetria de sesgo ya publicada por el runner.

    Returns
    -------
    tuple[dict, dict, str]
        Gates, caps y veredicto del arbol.
    """
    a1 = integracion["A1_ftrl_lambda1"]
    a2 = integracion["A2_sgd_presupuesto"]
    no_concluyente = bool(determinismo.get("no_concluyente", True))
    sondas_publicadas = all(bool(sondas[nombre].get("pass")) for nombre in SONDAS_PASS_FAIL)
    gates = {
        "baseline_ok": bool(baseline_ok),
        "delta_A1_suficiente": bool(a1["delta_vs_baseline"] >= GATE_DELTA),
        "folds_positivos_A1_suficiente": bool(a1["folds_positivos"] >= GATE_FOLDS),
        "control_favorable_A1_sobre_A2": bool(a1["delta_vs_baseline"] > a2["delta_vs_baseline"]),
        "ic95_A1_sin_cero": bool(a1["ic95_sin_cero"]),
        "sin_leakage": bool(all(invariantes.values())),
        "determinismo_concluyente": bool(not no_concluyente),
        "sondas_abc_en_verde": sondas_publicadas,
        "tabla_sesgo_publicada": bool(tabla_sesgo),
    }
    caps = {"determinismo_no_concluyente": no_concluyente,
            "criterio_determinismo": "determinismo no concluyente -> el veredicto no puede ser "
                                     "PROMISING",
            "sondas_no_publicadas": bool(not sondas_publicadas),
            "criterio_sondas": "sin las sondas (a)-(i) en verde -> veredicto capado a MARGINAL"}
    if all(gates.values()):
        veredicto = "PROMISING"
    elif a1["delta_vs_baseline"] > 0:
        veredicto = "MARGINAL"
    else:
        veredicto = "CLOSED"
    return gates, caps, veredicto


# =====================================================================================
# ---- 11. SALIDAS: FIGURA, INFORME, RUNLOG, CSV DE EXPERIMENTS, NOVEDAD E HISTORIAL ----
# =====================================================================================
def figura_curvas(estado: dict, mecanismo: dict, ruta: Path) -> None:
    """Figura 1x3 con las trayectorias por pasada de los dos brazos del fold 0.

    Panel 1: ``ll_train`` de ES-train por pasada de A1 y A2, con la banda de G1 sombreada
    (``G1_PISO`` y ``H(prior)_ES - 0.01``) y la linea ``ln(6)`` de referencia. Panel 2: ``D_acum``
    (rango medio de logits) por pasada de ambos brazos, con la banda objetivo de la regla de
    alpha (``[M/2, 2*M]``, el gate G0 de A1) y la linea de ES (la pasada en la que el early
    stopping interno detuvo cada brazo). Panel 3: ``macro_f1_es`` por pasada, que es la metrica
    que gobierna el ES, con las lineas de la pasada de parada.

    Parameters
    ----------
    estado:
        Checkpoint con los folds registrados.
    mecanismo:
        Tabla de mecanismo ya medida (sonda j / ``estado["mecanismo"]``) con la trayectoria por
        pasada de los dos brazos y ``h_prior_es``.
    ruta:
        Destino del PNG.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figura, ejes = plt.subplots(1, 3, figsize=(19.0, 5.2))
    colores = {"A1_ftrl_lambda1": "tab:blue", "A2_sgd_presupuesto": "tab:orange"}
    trayectorias = mecanismo.get("brazos") or {brazo: estado["brazos"][brazo]["folds"].get("0",
                                            {}).get("trayectoria", []) for brazo in BRAZOS}
    techo_g1 = float(mecanismo.get("h_prior_es", ANCLA_H_PRIOR_EXPLORACION)
                     - G1_TECHO_MARGEN)
    for indice, brazo in enumerate(BRAZOS):
        puntos = trayectorias.get(brazo) or []
        if not puntos:
            continue
        pasadas = [int(punto["pasada"]) for punto in puntos]
        etiqueta = f"{brazo} (parada {puntos[-1]['pasada']})"
        ejes[0].plot(pasadas, [float(punto["ll_train"]) for punto in puntos], marker="o",
                    linewidth=1.4, color=colores[brazo], label=etiqueta)
        ejes[1].plot(pasadas, [float(punto["D_acum"]) for punto in puntos], marker="s",
                    linewidth=1.4, color=colores[brazo], label=etiqueta)
        ejes[2].plot(pasadas, [float(punto["macro_f1_es"]) for punto in puntos], marker="^",
                    linewidth=1.4, color=colores[brazo], label=etiqueta)
        ejes[1].axvline(float(puntos[-1]["pasada"]), color=colores[brazo], linewidth=0.9,
                        linestyle="--", alpha=0.8,
                        label=f"ES de {brazo} (pasada {puntos[-1]['pasada']})")
    ejes[0].axhspan(G1_PISO, techo_g1, color="g", alpha=0.12, label="banda G1")
    ejes[0].axhline(ANCLA_LOG_UNIFORME, color="k", linewidth=0.8, linestyle=":",
                    label="ln(6) (uniforme)")
    ejes[1].axhspan(G0_FACTOR_INFERIOR * M_MARGEN, G0_FACTOR_SUPERIOR * M_MARGEN, color="purple",
                    alpha=0.10, label="banda G0 de la regla [M/2, 2M]")
    ejes[1].axhline(M_MARGEN, color="k", linewidth=0.8, linestyle=":", label="M = ln(5)")
    ejes[0].set_yscale("log")
    ejes[0].set_title("fold 0: ll_train de ES-train por pasada (banda G1)")
    ejes[1].set_title("fold 0: D_acum por pasada (banda de la regla de alpha) y linea del ES")
    ejes[2].set_title("fold 0: macro-F1 del ES-val por pasada (metrica que gobierna el ES)")
    ejes[0].set_ylabel("log_loss multiclase de ES-train")
    ejes[1].set_ylabel("rango medio de logits (max - min)")
    ejes[2].set_ylabel("macro-F1 del 10 % de TRAIN")
    for eje in ejes:
        eje.set_xlabel("pasada")
        eje.set_xticks(range(1, P_MAX + 1))
        eje.grid(alpha=0.3)
        eje.legend(fontsize=7, loc="best")
    figura.suptitle("V35-E1: FTRL-Proximal softmax multi-pasada (A1) frente al gemelo SGD con el "
                    "mismo presupuesto (A2), fold 0 consumido-diagnostico", fontsize=10)
    figura.tight_layout()
    ruta.parent.mkdir(parents=True, exist_ok=True)
    figura.savefig(ruta, dpi=140)
    plt.close(figura)
    print(f"  figura: {ruta_relativa(ruta)}", flush=True)


def tabla_sesgo_markdown(tabla: dict) -> str:
    """Tabla Markdown de simetria de sesgo a partir del detalle por fold ya medido.

    Parameters
    ----------
    tabla:
        Salida de :func:`tabla_simetria_sesgo`.

    Returns
    -------
    str
        Bloque Markdown con el camino unico de sesgo de cada brazo y la comparacion por fold.
    """
    filas = ["| brazo | caminos de sesgo | fuente | magnitud | valor |",
             "|---|---|---|---|---|"]
    for brazo in BRAZOS:
        sesgos = tabla["por_brazo"][brazo]["sesgo_por_fold"]
        primero = sesgos[min(sesgos, key=int)]
        if brazo == "A1_ftrl_lambda1":
            magnitud = "max|gradiente| de la coordenada x0 (fold 0)"
            valor = f"{primero['max_abs_gradiente_coordenada_sesgo']:.6f}"
        else:
            magnitud = "max|intercept_| (fold 0)"
            valor = f"{primero['max_abs_intercept']:.6f}"
        filas.append(f"| {brazo} | {primero['n_caminos']} | {primero['fuente']} | {magnitud} | "
                     f"{valor} |")
    comparacion = [("| fold | max|g_sesgo| (A1) | max|intercept_| (A2) | razon A1/A2 | mismo "
                    "orden de magnitud |"), "|---|---|---|---|---|"]
    for fold in sorted(tabla["comparacion_por_fold"], key=int):
        fila = tabla["comparacion_por_fold"][fold]
        razon = fila["razon_A1_sobre_A2"]
        comparacion.append(f"| {fold} | {fila['A1_max_abs_gradiente_coordenada_sesgo']:.6f} | "
                           f"{fila['A2_max_abs_intercept']:.6f} | "
                           f"{'n/d' if razon is None else f'{razon:.3e}'} | "
                           f"{fila['mismo_orden_de_magnitud']} |")
    return "\n".join(filas) + "\n\n" + "\n".join(comparacion)


def mecanismo_markdown(mecanismo: dict) -> str:
    """Tabla Markdown del mecanismo: ``ll_train``, ``D_acum`` y ES por pasada de A1 y A2.

    Parameters
    ----------
    mecanismo:
        Tabla de mecanismo ya medida (sonda j o ``estado["mecanismo"]``).

    Returns
    -------
    str
        Bloque Markdown con una fila por brazo y pasada.
    """
    filas = [("| brazo | pasada | ll_train (ES-train) | D_acum | macro-F1 ES-val | w_max_abs | "
              "razon |sigma*w|/|g| |"), "|---|---|---|---|---|---|---|---|"]
    for brazo in BRAZOS:
        for punto in mecanismo["brazos"][brazo]:
            razon = punto.get("razon_sigma_sobre_g")
            filas.append(f"| {brazo} | {punto['pasada']} | {punto['ll_train']:.6f} | "
                         f"{punto['D_acum']:.6f} | {punto['macro_f1_es']:.6f} | "
                         f"{punto['w_max_abs']:.6e} | "
                         f"{'n/d (A2 no es FTRL)' if razon is None else f'{razon:.3e}'} |")
    return "\n".join(filas)


def escribir_informe(resultado: dict) -> None:
    """Escribe el informe Markdown del ciclo con mecanismo, tabla de sesgo y triage formal.

    El informe publica, en este orden: el veredicto y su rama del arbol, el mecanismo (la tabla
    por pasada de A1 y A2 con la banda de la regla de alpha), la TABLA DE SIMETRIA DE SESGO, las
    asimetrias declaradas (incluida la desviacion ``partial_fit`` frente a
    ``fit(max_iter=10)`` de la enmienda 2 y la no-apoyacion del veredicto en la asimetria
    ``lambda2`` frente a ``alpha_reg``), el consumo diagnostico del fold 0, el triage formal, la
    puerta G0-G6 y las sondas (a)-(j).

    Parameters
    ----------
    resultado:
        JSON de resultados.
    """
    control = resultado["verificacion"]["constantes_control"]
    asimetria = resultado["asimetrias"]
    puerta = resultado["puerta"]
    triage = resultado.get("triage") or {}
    mecanismo = resultado["mecanismo_fold_0"]
    verificacion = resultado["verificacion"]
    compliance = resultado["compliance"]
    commit_congelado = resultado["prereg"]["commit_congelado"]
    linea_base = verificacion["linea_base_mayoritaria"]
    banda_ll = mecanismo["h_prior_es"] - G1_TECHO_MARGEN
    if resultado["motivo_de_no_calculo"]:
        veredicto_linea = resultado["motivo_de_no_calculo"]
    else:
        a1 = resultado["integracion"]["A1_ftrl_lambda1"]
        veredicto_linea = (f"Delta de A1 {a1['delta_vs_baseline']:+.7f} sobre el baseline, "
                           f"IC95 {a1['ic95']}.")
    lineas: list[str] = [
        "# V35-E1 - Ejecucion: FTRL-Proximal softmax multiclase multi-pasada con ES interno",
        "",
        f"- Fecha de ejecucion: {FECHA}",
        (f"- Pre-registro congelado: {ruta_relativa(PREREG_JSON)} (commit {commit_congelado}) y "
         f"{ARCHIVO_PRE_REGISTRO}"),
        f"- Puerta de convergencia (normativa): {ARCHIVO_PUERTA}",
        f"- Commit de ejecucion: `{resultado['commit']}`",
        (f"- Baseline primario recomputado: {verificacion['baseline_recomputado']:.7f} "
         f"(referencia nested {NESTED:.7f})"),
        (f"- Ancla trivial recalculada: {linea_base['macro_f1_linea_base']!r} "
         f"(clase {linea_base['clase']})"),
        (f"- {HILOS} hilos fijos; TF-IDF ajustada SOLO en el ES-train de cada fold; "
         f"P_max = {P_MAX} pasadas con ES interno (P_min = {P_MIN}, paciencia {PACIENCIA}, mejora "
         f"minima {MEJORA_MINIMA}); RSS pico {resultado['rss_pico_gb']:.2f} GB "
         f"(tope {RAM_TOPE_GB:g} GB)"),
        (f"- Compliance AST: {compliance['literales_de_ruta_revisados']} literales de ruta y "
         f"{compliance['modulos_revisados']} modulos, 0 violaciones; lecturas declaradas: "
         f"{compliance['lecturas']['literales_en_funciones_revisados']} literales en "
         f"{compliance['lecturas']['funciones_revisadas']} funciones, 0 violaciones"),
        (f"- Control A2 (enmienda 2): {control['ejecutable']['modo']} x "
         f"{control['ejecutable']['llamadas']}, con el estimador construido con max_iter = "
         f"{control['ejecutable']['max_iter']} (NO {P_MAX})."),
        "",
        f"## Veredicto: {resultado['veredicto']}",
        "",
        veredicto_linea,
        "",
        "## 1. Mecanismo: trayectoria por pasada de A1 y A2 (fold 0 consumido-diagnostico)",
        "",
        (f"- alpha* = 2*M/(d_norm*P_max^2) con M = ln({N_CLASES - 1}) = {M_MARGEN!r} y P_max = "
         f"{P_MAX} (0 valores evaluados, no es un grid); d_norm = {mecanismo['d_norm']!r} del "
         f"prior MEDIDO del ES-train del fold 0, luego alpha* = {mecanismo['alpha']!r}."),
        (f"- Banda objetivo de la regla (gate G0 de A1): D_acum en "
         f"[{G0_FACTOR_INFERIOR * M_MARGEN:.6f}, {G0_FACTOR_SUPERIOR * M_MARGEN:.6f}]; banda de "
         f"G1 sobre ll_train: [{G1_PISO}, {banda_ll:.6f}]."),
        (f"- w_max_abs por pasada (la firma de D-083 era w_max_abs = 26.4 con 28.86 de ll_train): "
         f"A1 {mecanismo['w_max_abs_A1_final']:.6e} con ll_train "
         f"{mecanismo['ll_train_A1_final']:.6f}; A2 {mecanismo['w_max_abs_A2_final']:.6e} con "
         f"ll_train {mecanismo['ll_train_A2_final']:.6f}."),
        "",
        mecanismo_markdown(mecanismo),
        "",
        f"Figura: {ruta_relativa(FIGURA)}.",
        "",
        "## 2. Tabla de simetria de sesgo (enmienda pre-ejecucion 1)",
        "",
        f"- Camino de A1: {resultado['simetria_sesgo']['A1_camino']}.",
        f"- Camino de A2: {resultado['simetria_sesgo']['A2_camino']}.",
        f"- {resultado['simetria_sesgo']['declaracion']}.",
        "",
        tabla_sesgo_markdown(resultado["simetria_sesgo"]),
        "",
        "## 3. Asimetrias declaradas",
        "",
        (f"- `lambda2` de A1 = {asimetria['lambda2_ftrl']} frente a `alpha_reg` de A2 = "
         f"{asimetria['alpha_reg_sgd']} (cociente {asimetria['cociente']:.0e}). "
         f"{asimetria['no_equivalencia_estructural']}"),
        f"- El veredicto NO se apoya en esa asimetria: {asimetria['apoyo_del_veredicto']}.",
        f"- Lectura correcta de A2: {asimetria['lectura_correcta_de_a2']}.",
        f"- Enmienda 2 (control A2): {asimetria['control_ajuste']}.",
        f"- Desviacion declarada: {asimetria['desvio_declarado_partial_fit']}.",
        # Se citan por OBJETO y no por numero: en este documento 'Enmienda 1' y 'Enmienda 2'
        # ya designan el doble sesgo y el control, y numerar aqui colisionaria con ellas.
        *[f"- Enmienda del pre-registro ({enmienda['objeto']}): {enmienda['motivo']} "
          f"(afecta a gates: {enmienda['afecta_a_gates']}; {enmienda['estado_datos']})."
          for enmienda in resultado["prereg"]["enmiendas"]],
        "",
        "## 4. Consumo diagnostico del fold 0 y puerta G0-G6 (fail-fast)",
        "",
        f"- {puerta['momento']}.",
        f"- {puerta['nota_fold_0']}.",
        f"- G5 evaluada: {puerta['g5_evaluada']}; puerta superada: {puerta['pasa']}.",
        "",
    ]
    for brazo in BRAZOS:
        fallidos = puerta["brazos_fallidos"][brazo]
        lineas.append(f"- {NOMBRES_BRAZO[brazo]}: "
                      + ("todos los gates en verde" if not fallidos
                         else f"gates fallidos {fallidos}"))
    lineas += [
        "",
        "## 5. Triage formal (solo si A1 falla y A2 pasa G0-G4)",
        "",
    ]
    if triage:
        lineas.append(f"- Se ejecuta si: {triage['se_ejecuta_si']}; condicion activada: "
                      f"{triage.get('condicion_arbol_activada')}.")
        lineas.append(f"- Clasificacion: {triage['clasificacion']}; indicador de defecto: "
                      f"{triage['hay_indicador_de_defecto']}; nodo concedido: "
                      f"{triage['nodo_concedido']}.")
        lineas.append(f"- Tabla de mecanismo publicada dentro del runner: "
                      f"{triage['mecanismo_publicado_dentro_del_runner']}.")
        lineas.append(f"- One shot: {triage['uno_shot']}.")
        lineas.append("")
        lineas.append("| comprobacion | valor | criterio |")
        lineas.append("|---|---|---|")
        for nombre, item in triage["checklist"].items():
            lineas.append(f"| {nombre} | {item['valor']} | {item['criterio']} |")
    else:
        lineas.append("- No se ejecuta: la puerta no se cerro en este ciclo.")
    lineas += [
        "",
        "## 6. Sondas (a)-(j), obligatorias antes de cualquier ajuste",
        "",
        "| sonda | pass | detalle |",
        "|---|---|---|",
    ]
    for nombre in NOMBRES_SONDAS:
        detalle = resultado["sondas"][nombre]
        marca = "no es gate" if nombre == NOMBRES_SONDAS[-1] else str(detalle.get("pass"))
        lineas.append(f"| ({nombre[0]}) {nombre} | {marca} | "
                     f"{detalle.get('descripcion', 'sin descripcion')} |")
    lineas += [
        "",
        f"- Politica: {resultado['sondas']['politica']}.",
        "",
        "## 7. Invariantes de no leakage",
        "",
        "| invariante | valor |",
        "|---|---|",
    ]
    for nombre, valor in (resultado.get("sin_leakage") or {}).items():
        lineas.append(f"| {nombre} | {valor} |")
    lineas += [
        "",
        "## 8. Cierre",
        "",
        f"- {resultado['siguiente_paso']}",
        ("- Sin particiones reservadas, sin envio, sin segunda particion: el champion queda "
         "intacto y el pre-registro prohibe automatizar el envio."),
        (f"- Artefactos: {ruta_relativa(RESULTADOS_JSON)} | {ruta_relativa(ESTADO_JSON)} | "
         f"{ruta_relativa(INFORME)} | {ruta_relativa(FIGURA)} | {ruta_relativa(NOVELTY_CSV)} | "
         + " | ".join(ruta_relativa(ruta_oof(brazo)) for brazo in BRAZOS) + "."),
        "",
    ]
    escribir_json_atomico(INFORME, "\n".join(lineas))
    print(f"  informe: {ruta_relativa(INFORME)}", flush=True)


def anotar_runlog(resultado: dict) -> bool:
    """Anade el bloque del ciclo a RUNLOG.md una sola vez (append de texto).

    Parameters
    ----------
    resultado:
        JSON de resultados.

    Returns
    -------
    bool
        True si se anadio el bloque; False si ya estaba.
    """
    marca = "V35-E1 ejecutado"
    if marca in RUNLOG.read_text(encoding="utf-8"):
        return False
    puerta = resultado["puerta"]
    triage = resultado.get("triage") or {}
    lineas = [
        "",
        "---",
        "",
        f"## V35-E1 ejecutado: FTRL-Proximal softmax multiclase multi-pasada - {FECHA}",
        "",
        f"- Veredicto: {resultado['veredicto']}"
        + (f" (triage: {triage.get('clasificacion')}, nodo concedido "
           f"{triage.get('nodo_concedido')})" if triage else "") + ".",
        (f"- Motivo de no calculo: {resultado['motivo_de_no_calculo']}"
         if resultado["motivo_de_no_calculo"] else
         f"- Delta de A1 {resultado['integracion']['A1_ftrl_lambda1']['delta_vs_baseline']:+.7f}, "
         f"IC95 {resultado['integracion']['A1_ftrl_lambda1']['ic95']}."),
        (f"- Puerta G0-G6 en el fold 0 (consumido-diagnostico): pasa = {puerta['pasa']}; brazos con "
         f"gates fallidos: { {brazo: puerta['brazos_fallidos'][brazo] for brazo in BRAZOS} }."),
        ("- Tabla de simetria de sesgo: UN solo camino por brazo; A1 coordenada x0 = 1.0, A2 solo "
         "intercept_ (doble sesgo de D-083 y V33-E1 corregido)."),
        ("- Enmienda 2 aplicada: el control A2 se ajusta con partial_fit x 10 sobre los ordenes "
         "congelados (no fit(max_iter=10)), con desviacion declarada en el informe."),
        (f"- Baseline recomputado {resultado['verificacion']['baseline_recomputado']:.7f}; ancla "
         f"trivial {ANCLA_TRIVIAL!r} reproducida; sondas (a)-(j) publicadas."),
        (f"- RSS pico {resultado['rss_pico_gb']:.2f} GB (tope {RAM_TOPE_GB:g}); {HILOS} hilos; "
         f"duracion {resultado['duracion_seg']:.0f} s."),
        ("- Sin particiones reservadas: sin evaluacion sobre la particion bloqueada del campeon, "
         "sin envio."),
        (f"- Artefactos: {ruta_relativa(RESULTADOS_JSON)} | {ruta_relativa(ESTADO_JSON)} | "
         f"{ruta_relativa(INFORME)} | {ruta_relativa(FIGURA)} | {ruta_relativa(NOVELTY_CSV)} | "
         + " | ".join(ruta_relativa(ruta_oof(brazo)) for brazo in BRAZOS) + "."),
    ]
    with RUNLOG.open("a", encoding="utf-8") as manejador:
        manejador.write("\n" + "\n".join(lineas) + "\n")
    return True


def anotar_experiments_csv(resultado: dict) -> bool:
    """Anade la fila V35-E1 a experiments.csv (22 columnas de la cabecera, sin CR).

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
        Si la cabecera no tiene 22 columnas o la fila no respeta el esquema (codigo 1).
    """
    with EXPERIMENTS_CSV.open("r", encoding="utf-8", newline="") as manejador:
        filas = list(csv.reader(manejador))
    cabecera = filas[0]
    if "V35-E1" in {fila[0] for fila in filas[1:] if fila}:
        return False
    if len(cabecera) != 22:
        raise Detencion(f"[STOP] la cabecera de experiments.csv tiene {len(cabecera)} columnas, "
                        "no 22 (exit 1)", 1)
    integracion = resultado.get("integracion") or {}
    a1 = integracion.get("A1_ftrl_lambda1")
    pasa = resultado["puerta"]["pasa"]
    if a1 is None:
        macro_f1_mean, macro_f1_std = "", ""
        resumen = (f"veredicto {resultado['veredicto']}; puerta G0-G6 del fold 0 con pasa = "
                   f"{pasa}; sin integracion, sin bootstrap y sin segundo alpha por pre-registro")
        interpretacion = resultado["motivo_de_no_calculo"]
        decision = (f"{resultado['veredicto']}: cerrar la familia por mecanismo; reabrirla exige "
                    f"acta nueva con hipotesis mecanica nueva, no otro gate")
    else:
        macro_f1_mean = f"{float(np.mean(a1['folds'])):.7f}"
        macro_f1_std = f"{float(np.std(a1['folds'])):.7f}"
        a2 = integracion["A2_sgd_presupuesto"]
        resumen = (f"A1 FTRL multi-pasada comb {a1['corrected_cv']:.7f} (delta "
                   f"{a1['delta_vs_baseline']:+.7f}, {a1['folds_positivos']}/5 folds, IC95 "
                   f"[{a1['ic95'][0]:+.7f}, {a1['ic95'][1]:+.7f}]); A2 gemelo SGD multi-pasada "
                   f"delta {a2['delta_vs_baseline']:+.7f}")
        interpretacion = (f"veredicto {resultado['veredicto']} por el arbol pre-registrado; "
                          "tabla de simetria de sesgo y tabla de mecanismo publicadas; el "
                          "veredicto no se apoya en la asimetria lambda2 frente a alpha_reg")
        decision = resultado["siguiente_paso"][:200]
    registro = {
        "experiment_id": "V35-E1", "fecha": FECHA, "bloque": "34.0",
        "etapa": "V35-E1 ejecucion: falsacion de la linea FTRL-Proximal (A1 decisional "
                 "multi-pasada con ES interno + A2 control gemelo SGD multi-pasada)",
        "pregunta": "la linea FTRL-Proximal softmax falla por MECANISMO aunque se le den 10 "
                    "pasadas y un alpha derivado de la regla de margen?",
        "hipotesis": "FTRL-Proximal no aporta senal aprendible con este presupuesto porque la "
                     "correccion proximal domina al gradiente; el gemelo SGD con el mismo "
                     "presupuesto demuestra que el presupuesto no es el limite",
        "features": "TF-IDF por fold (word 1-2, min_df 3, max_df 0.5, sublinear_tf) ajustada "
                    "SOLO en el ES-train del fold; 0 embeddings; 0 transformers; 0 LLM",
        "preprocessing": "vocabulario e idf por fold; sesgo de A1 como coordenada x0 = 1.0 con "
                         "propio z y propio n; A2 sin columna de sesgo y con fit_intercept=True "
                         "(un solo camino por brazo); p = softmax de 6 logits",
        "representacion": "TF-IDF dispersa, la MISMA receta que V33-E1 (no es una representacion "
                          "nueva)",
        "modelo": "FTRL-Proximal softmax K=6 propio multi-pasada (A1) / SGDClassifier log_loss "
                  "con partial_fit x 10 (A2)",
        "hiperparametros": f"congelados: alpha* = 2*ln(5)/(d_norm*{P_MAX}^2) DERIVADO en runtime "
                           f"(0 valores evaluados), lambda1 {LAMBDA1}, lambda2 {LAMBDA2} en "
                           f"denominador, beta {BETA}, power_t {POWER_T}, P_max {P_MAX}, P_min "
                           f"{P_MIN}, paciencia {PACIENCIA}, mejora minima {MEJORA_MINIMA}, "
                           f"particion ES {TEST_SIZE_ES} de TRAIN, control alpha 1e-06 con "
                           f"learning_rate optimal, {HILOS} hilos, semilla {SEMILLA}+fold",
        "validacion": "particion canonica de 5 folds; puerta G0-G6 fail-fast en el fold 0 "
                      "consumido-diagnostico; integracion P6/P7; bootstrap pareado 2000 "
                      "(semilla 20261010); sondas (a)-(j), tabla de simetria de sesgo y triage "
                      "formal",
        "macro_f1_mean": macro_f1_mean, "macro_f1_std": macro_f1_std,
        "weighted_f1_mean": "", "micro_f1_mean": "", "accuracy_mean": "",
        "resultado": resumen, "interpretacion": interpretacion, "decision": decision,
        "duracion_seg": f"{float(resultado['duracion_seg']):.1f}",
        "artefactos": f"{ruta_relativa(RESULTADOS_JSON)} | {ruta_relativa(ESTADO_JSON)} | "
                      f"{ruta_relativa(INFORME)} | {ruta_relativa(FIGURA)} | "
                      f"{ruta_relativa(NOVELTY_CSV)} | "
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
    """Escribe el registro de novedad (archivo NUEVO) con su cabecera y una sola fila.

    La fila declara los cuatro cruces que exige el acta: NO es embedding, NO es una
    representacion nueva, NO usa datos externos y NO es un grid (alpha* esta DERIVADO en forma
    cerrada con 0 valores evaluados).

    Parameters
    ----------
    resultado:
        JSON de resultados.

    Returns
    -------
    bool
        True si se escribio; False si la fila ya estaba.

    Raises
    ------
    Detencion
        Si la cabecera existente no coincide con la congelada (codigo 1).
    """
    if NOVELTY_CSV.exists():
        with NOVELTY_CSV.open("r", encoding="utf-8", newline="") as manejador:
            filas = list(csv.reader(manejador))
        if filas and filas[0] != COLUMNAS_NOVEDAD:
            raise Detencion("[STOP] la cabecera del registro de novedad no coincide (exit 1)", 1)
        if "V35-E1" in {fila[0] for fila in filas[1:] if fila}:
            return False
    fila = {
        "experiment_id": "V35-E1", "fecha": FECHA, "que_es": NOVEDAD_QUE_ES,
        "no_es_embedding": "no", "no_es_representacion_nueva": "no",
        "no_usa_datos_externos": "no", "no_es_un_grid": "no (alpha* DERIVADO en forma cerrada, "
                                                         "0 valores evaluados)",
        "diferencia_con_v33": NOVEDAD_DIFERENCIA_V33,
        "verificacion": "sondas (a)-(j) obligatorias con la (j) de tabla de mecanismo publicada "
                        "dentro del runner, tabla de simetria de sesgo por fold, sonda de "
                        "reproducibilidad de alpha* (diferencia exacta 0), sonda de estabilidad "
                        "del ES, sonda de simetria de presupuesto (sha256 de la particion ES y de "
                        "los ordenes por pasada), sonda de refit del vectorizador en los 5 folds, "
                        "puerta G0-G6 fail-fast en el fold 0, escritura atomica del OOF y del "
                        "checkpoint",
        "artefactos": f"{ruta_relativa(RESULTADOS_JSON)} | {ruta_relativa(ESTADO_JSON)} | "
                      f"{ruta_relativa(INFORME)} | {ruta_relativa(FIGURA)} | "
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
    """Anexa UNA fila V35-E1 al historial respetando su cabecera existente (15 columnas).

    En la rama de cierre por puerta, ``corrected_cv``, ``delta_vs_V20``, ``folds`` y el bootstrap
    quedan VACIOS y la columna ``bootstrap`` declara el motivo de no calculo, como exige el
    arbol.

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
        Si el historial no existe o su esquema no coincide con el congelado (codigo 1).
    """
    if not HISTORIA_CSV.exists():
        raise Detencion(f"[STOP] no existe el historial {ruta_relativa(HISTORIA_CSV)} (exit 1)", 1)
    with HISTORIA_CSV.open("r", encoding="utf-8", newline="") as manejador:
        filas = list(csv.reader(manejador))
    cabecera = filas[0]
    if cabecera != COLUMNAS_HISTORIA:
        raise Detencion(f"[STOP] el esquema del historial no coincide con el congelado: {cabecera} "
                        "(exit 1)", 1)
    if "V35-E1" in {fila[1] for fila in filas[1:] if len(fila) > 1}:
        return False
    integracion = resultado.get("integracion") or {}
    a1 = integracion.get("A1_ftrl_lambda1")
    comun = {
        "phase": "V34", "experiment_id": "V35-E1", "date": FECHA,
        "hypothesis": "FTRL-Proximal no aporta senal aprendible con este presupuesto porque la "
                      "correccion proximal domina al gradiente; el gemelo SGD con el mismo "
                      "presupuesto demuestra que el presupuesto no es el limite",
        "model": "FTRL-Proximal softmax K=6 propio multi-pasada con ES interno (A1) / "
                 "SGDClassifier log_loss con partial_fit x 10 (A2)",
        "representation": "TF-IDF por fold, misma receta que V33-E1 (no es representacion nueva)",
    }
    if a1 is None:
        registro = {**comun, "corrected_cv": "", "delta_vs_V20": "", "folds": "",
                    "bootstrap": f"no se calcula: {resultado['motivo_de_no_calculo']}",
                    "rescue": "", "damage": "", "net": "",
                    "decision": resultado["veredicto"],
                    "resultado": f"puerta G0-G6: {resultado['puerta']['pasa']} | veredicto "
                                 f"{resultado['veredicto']} | triage "
                                 f"{(resultado.get('triage') or {}).get('clasificacion')} | "
                                 f"fold 0 consumido-diagnostico"}
    else:
        diag = resultado["complementariedad"]["A1_ftrl_lambda1"]["vs_campeon"]
        registro = {**comun, "corrected_cv": f"{a1['corrected_cv']:.7f}",
                    "delta_vs_V20": f"{a1['delta_vs_baseline']:+.7f}",
                    "folds": f"{a1['folds_positivos']}/5",
                    "bootstrap": f"[{a1['ic95'][0]:+.7f}, {a1['ic95'][1]:+.7f}]",
                    "rescue": f"{diag['rescate']}", "damage": f"{diag['danio']}",
                    "net": f"{diag['neto']}", "decision": resultado["veredicto"],
                    "resultado": f"gate: {resultado['gates']['delta_A1_suficiente']} | veredicto "
                                 f"{resultado['veredicto']} | control A1-A2 "
                                 f"{resultado['control']['diferencia_puntual']:+.7f}"}
    if set(registro) != set(COLUMNAS_HISTORIA):
        raise Detencion("[STOP] la fila del historial no respeta su cabecera (exit 1)", 1)
    with HISTORIA_CSV.open("a", encoding="utf-8", newline="") as manejador:
        csv.writer(manejador, lineterminator="\n").writerow(
            [registro[columna] for columna in COLUMNAS_HISTORIA])
    return True


def resumen_consola(resultado: dict) -> None:
    """Resumen final en consola (maximo 14 lineas).

    Parameters
    ----------
    resultado:
        JSON de resultados.
    """
    puerta = resultado["puerta"]
    print()
    print(f"[V35-E1] fin en {resultado['duracion_seg']:.0f} s | baseline "
          f"{resultado['verificacion']['baseline_recomputado']:.7f} | {HILOS} hilos | sondas "
          f"(a)-(j) publicadas | RSS pico {resultado['rss_pico_gb']:.2f} GB")
    print(f"  puerta G0-G6 (fold 0, consumido-diagnostico): pasa = {puerta['pasa']} | G5 "
          f"evaluada = {puerta['g5_evaluada']}")
    for brazo in BRAZOS:
        detalle = resultado["entrenamiento"][brazo]["folds"].get("0", {})
        print(f"  {brazo}: alpha {detalle.get('alpha_efectivo')!r} parada "
              f"{detalle.get('pasada_de_parada')} (mejor {detalle.get('mejor_pasada')}) "
              f"macro-F1 standalone {detalle.get('macro_f1_standalone_fold_val')} | fallidos: "
              f"{puerta['brazos_fallidos'][brazo]}")
    print(f"  sesgo: A1 n_caminos "
          f"{resultado['simetria_sesgo']['por_brazo']['A1_ftrl_lambda1']['n_caminos_de_sesgo']}"
          f", A2 n_caminos "
          f"{resultado['simetria_sesgo']['por_brazo']['A2_sgd_presupuesto']['n_caminos_de_sesgo']}, "
          f"doble sesgo = {resultado['simetria_sesgo']['doble_sesgo']}")
    triage = resultado.get("triage") or {}
    if triage:
        print(f"  triage formal: {triage['clasificacion']} | nodo concedido "
              f"{triage['nodo_concedido']} | condicion del arbol "
              f"{triage.get('condicion_arbol_activada')}")
    if resultado.get("integracion"):
        for brazo in BRAZOS:
            met = resultado["integracion"][brazo]
            print(f"  integracion {brazo}: comb {met['corrected_cv']:.7f} (delta "
                  f"{met['delta_vs_baseline']:+.7f}, {met['folds_positivos']}/5)")
    else:
        print(f"  sin calculo: {resultado['motivo_de_no_calculo']}")
    print(f"  VEREDICTO: {resultado['veredicto']}")
    print(f"  siguiente: {resultado['siguiente_paso'][:150]}")
    print(f"  artefactos: {ruta_relativa(RESULTADOS_JSON)} | {ruta_relativa(INFORME)} | "
          f"{ruta_relativa(FIGURA)} | {ruta_relativa(NOVELTY_CSV)}")


# =====================================================================================
# ---- 12. CONSTRUCCION DEL JSON DE RESULTADOS ----
# =====================================================================================
def mecanismo_fold_0(estado: dict) -> dict:
    """Normaliza la tabla de mecanismo del fold 0 con alpha, d_norm, H(prior) y sus finales.

    Parameters
    ----------
    estado:
        Checkpoint con ``mecanismo`` y los folds 0 de los dos brazos.

    Returns
    -------
    dict
        ``brazos`` (trayectoria por pasada), ``alpha``, ``d_norm``, ``h_prior_es`` y los cuatro
        escalares finales que lee el informe.
    """
    mecanismo = estado["mecanismo"] or {"brazos": {}}
    brazos = dict(mecanismo.get("brazos") or {})
    for brazo in BRAZOS:
        if not brazos.get(brazo):
            brazos[brazo] = estado["brazos"][brazo]["folds"]["0"]["trayectoria"]
    detalle = estado["brazos"]["A1_ftrl_lambda1"]["folds"]["0"]
    return {"fold_0": True,
            "brazos": brazos,
            "alpha": float(detalle["alpha_efectivo"]), "d_norm": float(detalle["d_norm"]),
            "h_prior_es": float(detalle["h_prior_es"]),
            "M": M_MARGEN,
            "banda_D_acum_objetivo": [G0_FACTOR_INFERIOR * M_MARGEN,
                                      G0_FACTOR_SUPERIOR * M_MARGEN],
            "banda_ll_train": [G1_PISO, float(detalle["h_prior_es"] - G1_TECHO_MARGEN)],
            "ll_train_A1_final": float(brazos["A1_ftrl_lambda1"][-1]["ll_train"]),
            "ll_train_A2_final": float(brazos["A2_sgd_presupuesto"][-1]["ll_train"]),
            "w_max_abs_A1_final": float(brazos["A1_ftrl_lambda1"][-1]["w_max_abs"]),
            "w_max_abs_A2_final": float(brazos["A2_sgd_presupuesto"][-1]["w_max_abs"])}


def construir_resultado(inicio: float, prereg: dict, auditoria: dict, lecturas: dict,
                        baseline: float, linea_base: dict, hashes_corpus: dict,
                        detalle_constantes: dict, detalle_versiones: dict, detalle_hashes: dict,
                        sondas: dict, asimetria: dict, simetria_sesgo: dict, estado: dict,
                        puerta: dict, refit: dict, integracion: dict | None,
                        complementariedad: dict | None, invariantes: dict | None,
                        determinismo: dict, control: dict | None, gates: dict, caps: dict,
                        veredicto: str, motivo_no_calculo: str | None) -> dict:
    """Ensambla el JSON de resultados con config, hashes, sondas, puerta, gates y veredicto.

    Los escalares que se guardan son NATIVOS (``float``, ``int``, ``bool``, ``str``, ``None``) y
    las claves de fold se fuerzan a texto: lo hace :func:`como_nativo` al serializar.

    Parameters
    ----------
    inicio:
        Marca temporal del inicio de la corrida.
    prereg:
        Contenido del pre-registro.
    auditoria:
        Salida de :func:`auditar_ast_propio`.
    lecturas:
        Salida de :func:`auditar_lecturas_de_datos`.
    baseline:
        Baseline recomputado.
    linea_base:
        Ancla trivial recalculada.
    hashes_corpus:
        Hashes recomputados del corpus.
    detalle_constantes:
        Detalle de la verificacion de no-deriva.
    detalle_versiones:
        Detalle de las versiones instaladas.
    detalle_hashes:
        Detalle de los hashes verificados.
    sondas:
        Detalle de las sondas (a)-(j).
    asimetria:
        Sonda de asimetria declarada (incluida la enmienda 2).
    simetria_sesgo:
        Tabla de simetria de sesgo.
    estado:
        Checkpoint final.
    puerta:
        Detalle de la puerta G0-G6.
    refit:
        Detalle de la sonda de refit (o su ausencia documentada en la rama de cierre).
    integracion:
        Metricas del combinador por brazo, o None en la rama de cierre.
    complementariedad:
        Diagnosticos standalone por brazo, o None.
    invariantes:
        Evidencias de no leakage, o None.
    determinismo:
        Sondas de determinismo del fold 0 y veredicto de concluyencia.
    control:
        Bootstrap pareado A1 contra A2 como diagnostico, o None.
    gates:
        Gates calculados.
    caps:
        Caps aplicados.
    veredicto:
        Veredicto del arbol.
    motivo_no_calculo:
        Motivo de no calculo de delta y bootstrap, o None si si se calcularon.

    Returns
    -------
    dict
        JSON de resultados completo.
    """
    oof_artefactos = {}
    for brazo in BRAZOS:
        ruta = ruta_oof(brazo)
        oof_artefactos[brazo] = {"archivo": ruta_relativa(ruta), "existe": bool(ruta.exists()),
                                "sha256": sha256_archivo(ruta) if ruta.exists() else None}
    entrenamiento = {brazo: {"folds": estado["brazos"][brazo]["folds"],
                             "folds_completados": estado["brazos"][brazo]["folds_completados"],
                             "oof_artefacto": oof_artefactos[brazo]} for brazo in BRAZOS}
    alphas = {fold: float(detalle["alpha_efectivo"]) for fold, detalle
              in sorted(estado["brazos"]["A1_ftrl_lambda1"]["folds"].items())}
    siguiente = prereg["arbol"].get(veredicto)
    if isinstance(siguiente, dict):
        siguiente = siguiente.get("semantica", veredicto)
    return {
        "fase": FASE_ESTADO.replace(" (checkpoint resumible)", ""),
        "fecha": FECHA,
        "commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=False,
                                 capture_output=True, text=True, timeout=30).stdout.strip(),
        "prereg": {"json": ruta_relativa(PREREG_JSON), "sha256": sha256_archivo(PREREG_JSON),
                   "informe": ARCHIVO_PRE_REGISTRO, "puerta": ARCHIVO_PUERTA,
                   "commit_congelado": str(prereg["commit"]),
                   "enmiendas": prereg["enmiendas"]},
        "verificacion": {
            "constantes": detalle_constantes, "versiones": detalle_versiones,
            "constantes_control": {"control_prereg": prereg["control"],
                                   "ejecutable": AJUSTE_CONTROL},
            "hashes": detalle_hashes["checks"], "hashes_recomputados": detalle_hashes[
                "recomputados"], "hashes_esperados": detalle_hashes["esperados"],
            "hashes_corpus_recomputados": hashes_corpus,
            "receta_hash_textos": prereg["corpus"]["hash_contenido"],
            "corpus_prohibido": prereg["corpus"]["prohibido"],
            "corpus_receta": prereg["corpus"]["receta"],
            "baseline_recomputado": baseline, "baseline_esperado": BASELINE, "baseline_ok": True,
            "ids_alineados": True, "etiquetas_alineadas": True,
            "linea_base_mayoritaria": linea_base, "anclas": prereg["anclas"]},
        "compliance": {**auditoria, **COMPLIANCE_CIFRAS, "lecturas": lecturas,
                       "auditorias_exigidas": COMPLIANCE_AUDITORIAS,
                       "nota_ast_audit": prereg["compliance"]["ast_audit"]},
        "config": {
            "vectorizador": VECTORIZADOR,
            "vectorizador_fit": {"ngram_range": list(VECTORIZADOR["ngram_range"])},
            "nota_ngram_range": "la config congelada declara ngram_range como lista; sklearn "
                                "1.8.0 exige tupla y en el fit se convierte (mismo valor); el "
                                "sha256 congelado del dict usa SIEMPRE la lista",
            "norm_dtype": prereg["representacion"]["norm_dtype"],
            "ajuste_vectorizador": prereg["representacion"]["ajuste"],
            "sesgo": prereg["representacion"]["sesgo"],
            "presupuesto": {"p_max": P_MAX, "p_min": P_MIN, "paciencia": PACIENCIA,
                            "mejora_minima": MEJORA_MINIMA, "test_size_es": TEST_SIZE_ES,
                            "orden_por_pasada": prereg["budgeto"]["orden_por_pasada"],
                            "particion_es": prereg["budgeto"]["particion_es"]["generador"],
                            "restauracion": prereg["budgeto"]["restauracion"]
                            if "restauracion" in prereg["budgeto"]
                            else prereg["budgeto"]["particion_es"]["restauracion"],
                            "prohibido": prereg["budgeto"]["prohibido"]},
            "regla_alpha": {"formula": prereg["regla_alpha"]["formula"],
                            "con_power_t_cero": prereg["regla_alpha"]["con_power_t_cero"],
                            "M": prereg["regla_alpha"]["M"]["valor"],
                            "valores_evaluados": int(prereg["regla_alpha"]["valores_evaluados"]),
                            "ancla_d_norm": ANCLA_D_NORM_REFERENCIA,
                            "ancla_alpha": ANCLA_ALPHA_PARA_D_NORM_REFERENCIA,
                            "ancla_n_train_fold": ANCLA_N_TRAIN_FOLD,
                            "alpha_por_fold": alphas},
            "control": {"ejecutable": AJUSTE_CONTROL, "espejo_prereg": prereg["control"]["ajuste"],
                        "nota_kwargs": "el estimador NO se construye con max_iter = 10: se "
                                       "construye con max_iter = 1 y se llama a partial_fit una "
                                       "vez por orden congelado de cada pasada"},
            "pesos_plugin": PESOS, "hilos": HILOS,
            "seeds": {"global": SEMILLA, "es_segunda_particion": SEMILLA_SEGUNDA_PARTICION,
                      "sgd_random_state": prereg["control"]["random_state"]},
            "integracion": {"c_logreg": C_LOGREG, "max_iter_logreg": MAX_ITER_LOGREG,
                            "clip": L2_CLIP, "n_bootstrap": N_BOOTSTRAP,
                            "pesos_plugin": PESOS,
                            "decision": "argmax(p_comb * pesos plugin congelados)"},
            "coste": {"presupuesto_declarado_seg": PRESUPUESTO_DECLARADO_SEG,
                      "proyeccion_max_h": PROYECCION_MAX_H, "timeout_global_h": TIMEOUT_H,
                      "ram_tope_gb": RAM_TOPE_GB,
                      "variabilidad_referencia": VARIABILIDAD,
                      "gate_delta": GATE_DELTA, "gate_folds": GATE_FOLDS,
                      "umbral_alta_confianza": UMBRAL_ALTA_CONFIANZA,
                      "tolerancia_determinismo": TOL_DETERMINISMO},
        },
        "sondas": sondas,
        "asimetrias": asimetria,
        "simetria_sesgo": simetria_sesgo,
        "puerta": puerta,
        "triage": estado.get("triage"),
        "mecanismo_fold_0": mecanismo_fold_0(estado),
        "refit": refit,
        "vocabulario_idf_por_fold": estado["vectorizador_por_fold"],
        "particiones_es_por_fold": estado["es_por_fold"],
        "determinismo": determinismo,
        "control": control,
        "calibracion": estado["calibracion"],
        "rss_pico_gb": float(estado["rss_pico_gb"]), "ram_tope_gb": RAM_TOPE_GB,
        "sin_leakage": invariantes,
        "entrenamiento": entrenamiento,
        "integracion": integracion,
        "complementariedad": complementariedad,
        "gates": gates,
        "caps": caps,
        "veredicto": veredicto,
        "motivo_de_no_calculo": motivo_no_calculo,
        "siguiente_paso": siguiente,
        "duracion_seg": float(time.time() - inicio),
        "duracion_sondas_seg": float(sondas["duracion_seg"]),
        "tiempos_por_brazo_fold_seg": {brazo: {fold: float(detalle["tiempo_seg"]) for fold,
                                               detalle in estado["brazos"][brazo]["folds"]
                                               .items()} for brazo in BRAZOS},
    }


# =====================================================================================
# ---- 13. PIPELINE COMPLETO Y CLI ----
# =====================================================================================
def ejecutar(verificar: bool) -> int:
    """Pipeline completo: verificacion, sondas, puerta, entrenamiento resumible, evaluacion.

    En ``--verificar`` se detiene ANTES de tocar el disco: imprime ``VERIFICACION OK`` y NO crea
    ni el estado, ni el OOF, ni los checkpoints, ni filas en el RUNLOG, en ``experiments.csv`` ni
    en el historial.

    Parameters
    ----------
    verificar:
        Si True solo verifica y ejecuta las sondas, sin entrenar ni escribir.

    Returns
    -------
    int
        0 si todo correcto, incluidos los cierres documentados NO_ATRIBUIBLE_BY_GATE y
        LINEA_ONLINE_CERRADA_POR_MECANISMO.

    Raises
    ------
    Detencion
        Con el codigo pre-registrado de la guarda que fallo (2, 4, 6, 7 u 8).
    """
    inicio = time.time()
    autotest_numerico()
    prereg = cargar_preregistro()
    constantes_ok, detalle_constantes = verificar_constantes(prereg)
    versiones_ok, detalle_versiones = verificar_versiones(prereg)
    if not (constantes_ok and versiones_ok):
        raise Detencion("[STOP] el pre-registro congelado no se respeta (exit 2)", 2)
    print("  verificaciones de no-deriva: "
          f"{sum(1 for valor in detalle_constantes.values() if valor)}/"
          f"{len(detalle_constantes)} OK", flush=True)
    print("  enmienda 2: control A2 con partial_fit x "
          f"{AJUSTE_CONTROL['llamadas']} (max_iter del estimador = "
          f"{AJUSTE_CONTROL['max_iter']}, NO {P_MAX}); desviacion declarada frente a "
          "fit(max_iter=10) registrada", flush=True)
    auditoria = auditar_ast_propio()
    lecturas = auditar_lecturas_de_datos()
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
          f"y_true; hashes de archivo, array de textos, vectorizador, sonda previa y nucleo OK",
          flush=True)
    print(f"  ancla trivial recalculada: {linea_base['macro_f1_linea_base']!r} (clase "
          f"{linea_base['clase']})", flush=True)
    print(f"  compliance AST: {auditoria['literales_de_ruta_revisados']} literales de ruta y "
          f"{auditoria['modulos_revisados']} modulos, 0 violaciones; lecturas: "
          f"{lecturas['literales_en_funciones_revisados']} literales en "
          f"{lecturas['funciones_revisadas']} funciones, 0 violaciones", flush=True)
    sondas = ejecutar_sondas(clase, fold_idx, None)
    asimetria = sonda_asimetria_no_equivalente(prereg)
    print(f"  sondas (a)-(j) en {sondas['duracion_seg']:.1f} s: todas pasan; asimetria declarada "
          f"publicada (cociente lambda2/alpha_reg = {asimetria['valores']['cociente']:.0e}, "
          "no equivalencia estructural)", flush=True)
    if verificar:
        print("VERIFICACION OK", flush=True)
        return 0

    estado = cargar_estado()
    # La sonda (j) NO es pass/fail (publica la tabla de mecanismo), luego se registra como None
    # en vez de leer una clave que no existe: leerla reventaba la corrida justo despues de
    # escribir el estado.
    estado["sondas"] = {nombre: (bool(sondas[nombre]["pass"])
                             if "pass" in sondas[nombre] else None)
                       for nombre in NOMBRES_SONDAS}
    guardar_estado(estado)
    oofs, puerta_fallida = entrenar_con_puerta(estado, textos, clase, fold_idx)
    if puerta_fallida is not None:
        fallidos = puerta_fallida["brazos_fallidos"]
        condicion = bool(fallidos["A1_ftrl_lambda1"] and not fallidos["A2_sgd_presupuesto"])
        triage = triage_formal(estado, sondas, puerta_fallida)
        triage["condicion_arbol_activada"] = condicion
        estado["triage"] = triage
        guardar_estado(estado)
        veredicto = ("LINEA_ONLINE_CERRADA_POR_MECANISMO"
                     if condicion and triage["nodo_concedido"] else "NO_ATRIBUIBLE_BY_GATE")
        motivo = (f"la puerta G0-G6 se cerro en el fold 0 (pasa = {puerta_fallida['pasa']}, "
                  f"brazos con gates fallidos: { {brazo: fallidos[brazo] for brazo in BRAZOS} }); "
                  "el pre-registro prohibe integrar y hacer bootstrap: delta y bootstrap quedan "
                  "VACIOS")
        determinismos = {"sonda_por_brazo": {brazo: estado["brazos"][brazo][
            "sonda_determinismo"] for brazo in BRAZOS},
            "no_concluyente": True,
            "tolerancia_declarada": TOL_DETERMINISMO,
            "variabilidad_referencia": VARIABILIDAD,
            "nota": "la rama de cierre no llega a los gates de delta: el determinismo se publica "
                    "solo como diagnostico del fold 0 y el cap queda activo por construccion"}
        simetria_sesgo = tabla_simetria_sesgo(estado)
        resultado = construir_resultado(inicio, prereg, auditoria, lecturas, baseline, linea_base,
                                        hashes_corpus, detalle_constantes, detalle_versiones,
                                        detalle_hashes, sondas, asimetria, simetria_sesgo, estado,
                                        puerta_fallida,
                                        {"descripcion": "sonda de refit del vectorizador en los 5 "
                                                       "folds, ajustada SIEMPRE sobre ES-train",
                                         "folds": {}, "reproducido": None,
                                         "nota": "no se ejecuta: la puerta cerro el ciclo antes de "
                                                 "terminar los folds"},
                                        None, None, None, determinismos, None, {}, {},
                                        veredicto, motivo)
        escribir_json(RESULTADOS_JSON, resultado)
        print(f"  resultados: {ruta_relativa(RESULTADOS_JSON)}", flush=True)
        figura_curvas(estado, mecanismo_fold_0(estado), FIGURA)
        escribir_informe(resultado)
        print(f"  RUNLOG.md: {'anotado' if anotar_runlog(resultado) else 'ya estaba'}", flush=True)
        print(f"  experiments.csv: "
              f"{'fila V35-E1 anadida' if anotar_experiments_csv(resultado) else 'ya estaba'}",
              flush=True)
        print(f"  registro de novedad: "
              f"{'escrito' if escribir_registro_novedad(resultado) else 'ya estaba'}", flush=True)
        print(f"  historial: {'fila anexada' if anotar_historial(resultado) else 'ya estaba'}",
              flush=True)
        resumen_consola(resultado)
        return 0

    puerta = estado["puerta"]
    refit = refit_vectorizador(textos, clase, fold_idx, estado)
    estado["refit"] = refit
    guardar_estado(estado)
    if not refit["reproducido"]:
        raise Detencion("[STOP] la sonda de refit no reproduce los hashes de vocabulario e idf en "
                        "los 5 folds (exit 2)", 2)
    invariantes = verificar_invariantes(estado, clase, fold_idx, oofs, refit)
    print(f"  invariantes de no leakage: {sum(bool(v) for v in invariantes.values())}/"
          f"{len(invariantes)} OK", flush=True)
    verificar_ram(estado)
    simetria_sesgo = tabla_simetria_sesgo(estado)

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
    diferencia = bootstrap_delta(clase, preds_comb["A2_sgd_presupuesto"],
                                 preds_comb["A1_ftrl_lambda1"])
    control = {"delta_A1": integracion["A1_ftrl_lambda1"]["delta_vs_baseline"],
               "delta_A2": integracion["A2_sgd_presupuesto"]["delta_vs_baseline"],
               "diferencia_puntual": (integracion["A1_ftrl_lambda1"]["delta_vs_baseline"]
                                      - integracion["A2_sgd_presupuesto"]["delta_vs_baseline"]),
               "delta_medio": diferencia["delta_medio"], "ic95": diferencia["ic95"],
               "p_delta_positivo": diferencia["p_delta_positivo"],
               "ic95_sin_cero": diferencia["ic95_sin_cero"],
               "criterio": "gate de control: el delta PUNTUAL de A1 sobre el de A2",
               "nota": "IC95 pareado de la diferencia como DIAGNOSTICO adicional; el gate "
                       "pre-registrado es el delta puntual"}
    determinismos = {"sonda_por_brazo": {brazo: estado["brazos"][brazo]["sonda_determinismo"]
                                         for brazo in BRAZOS},
                     "no_concluyente": bool(any(estado["brazos"][brazo]["sonda_determinismo"]
                                                and estado["brazos"][brazo]["sonda_determinismo"]
                                                ["no_concluyente"] for brazo in BRAZOS)),
                     "tolerancia_declarada": TOL_DETERMINISMO,
                     "variabilidad_referencia": VARIABILIDAD,
                     "gate_delta": GATE_DELTA,
                     "criterio": "max|delta p| = 0.0 EXACTO entre dos corridas identicas del "
                                 "fold 0 (G5, que es tambien la sonda h)"},
    gates, caps, veredicto = calcular_gates(True, integracion, invariantes, determinismos, sondas,
                                            bool(simetria_sesgo["comparacion_por_fold"]))
    print(f"  veredicto: {veredicto}", flush=True)
    resultado = construir_resultado(inicio, prereg, auditoria, lecturas, baseline, linea_base,
                                    hashes_corpus, detalle_constantes, detalle_versiones,
                                    detalle_hashes, sondas, asimetria, simetria_sesgo, estado,
                                    puerta, refit, integracion, complementariedad, invariantes,
                                    determinismos, control, gates, caps, veredicto, None)
    escribir_json(RESULTADOS_JSON, resultado)
    print(f"  resultados: {ruta_relativa(RESULTADOS_JSON)}", flush=True)
    figura_curvas(estado, mecanismo_fold_0(estado), FIGURA)
    escribir_informe(resultado)
    print(f"  RUNLOG.md: {'anotado' if anotar_runlog(resultado) else 'ya estaba'}", flush=True)
    print(f"  experiments.csv: "
          f"{'fila V35-E1 anadida' if anotar_experiments_csv(resultado) else 'fila ya estaba'}",
          flush=True)
    print(f"  registro de novedad: "
          f"{'escrito' if escribir_registro_novedad(resultado) else 'ya estaba'}", flush=True)
    print(f"  historial: {'fila anexada' if anotar_historial(resultado) else 'ya estaba'}", flush=True)
    resumen_consola(resultado)
    return 0


def parsear_argumentos() -> argparse.Namespace:
    """Argumentos de linea de comandos.

    Returns
    -------
    argparse.Namespace
        ``--verificar`` (solo verificacion y sondas, sin entrenar ni escribir).
    """
    analizador = argparse.ArgumentParser(
        description="V35-E1: falsacion de la linea FTRL-Proximal segun el pre-registro "
                    "congelado: A1 (FTRL-Proximal softmax multiclase multi-pasada con early "
                    "stopping interno) frente a A2 (gemelo SGD multi-pasada con partial_fit), "
                    "con puerta G0-G6 fail-fast en el fold 0.")
    analizador.add_argument("--verificar", action="store_true",
                            help="solo verifica constantes, versiones, hashes, corpus, anclas, "
                                 "auditoria AST, lecturas, baseline y sondas (a)-(j) (no entrena "
                                 "y no escribe nada; imprime VERIFICACION OK)")
    return analizador.parse_args()


def main() -> int:
    """Entrada del script con manejo de los codigos de salida pre-registrados.

    Returns
    -------
    int
        Codigo de salida (0 correcto, incluidos los cierres documentados; 2 verificacion; 4 sonda
        fallida; 6 calibracion excedida; 7 timeout global; 8 RSS por encima del tope; 1 error
        inesperado).
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
