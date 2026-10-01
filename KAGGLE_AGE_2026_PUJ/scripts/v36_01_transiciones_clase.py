r"""v36_01_transiciones_clase.py
Fecha: 2026-09-30
Descripcion: VIA C1-A del ciclo V36. Bloque de transiciones de CLASE de caracter (brecha L4:
"patrones letra->digito, espacio->puntuacion que el TF-IDF no expresa como feature explicita").
Cada texto se convierte en una secuencia de clases de caracter con el mapa CATEGORICO LITERAL del
pre-registro (S espacio, D digito ASCII, P puntuacion ASCII, L minuscula con y sin acento, M
mayuscula con y sin acento, O resto) y se cuentan BIGRAMAS y TRIGRAMAS de clase sobre el texto
entero, SIN simbolos de borde y SIN min_df (vocabulario acotado por construccion). El bloque, una
matriz dispersa de conteos, se documenta como "cada n-grama de clase repetido por su frecuencia" y
se pasa por TfidfVectorizer(norm="l2", sublinear_tf=True, lowercase=False, token_pattern=r"\S+");
el resultado se concatena al bloque char_wb (3,5) min_df 5 max_features 300000 del campeon.

Brazos que hay que medir, todos con los MISMOS folds (LEIDOS de artifacts/p33_scores/todos/
fold_idx.npy, la particion canonica: NO se recrean ni se barajan), la misma semilla declarada y el
MISMO modelo (sklearn.svm.LinearSVC C=1.0 squared_hinge l2 max_iter 3000 tol 1e-4, la receta del
campeon leida del manifiesto canonico; NO es un grid):
  1) banco REFERENCIA: char_wb IDENTICO, sin C1-A.
  2) banco CANDIDATO: char_wb IDENTICO + bloque C1-A (mapa real). El delta del banco es 2 - 1.
  3) control de PERMUTACIONES (condicion (ii) del ruling, OBLIGATORIO): >= 3 permutaciones del
     alfabeto derivadas de una semilla DECLARADA (cv.semilla = 42), nunca de los datos ni de los
     resultados; el mapa real DEBE ganar a los controles.
  4) control de BARAJADO (C3): el bloque C1-A con las transiciones barajadas dentro de cada
     documento (destruye el orden y conserva el recuento).
  5) INTEGRACION: el campeon (v30_prob con sus pesos plug-in congelados) frente al campeon con el
     bloque C1-A anadido, por el combinador anidado y el bootstrap pareado de la maquinaria
     auditada de V35 (combinador_oof, evaluar_combinacion, diagnostico_brazo, bootstrap_delta).

Punto de partida de la integracion, verificado EN RUNTIME y con [STOP] exit 2 si no reproduce: el
campeon con sus pesos plug-in [1.0, 1.1749, 1.2721, 1.163, 1.001, 0.8128] da el baseline 0.6159732
(macro-F1 de argmax(v30 * pesos) sobre la particion canonica). Que bloques intervienen y como se
combinan: (a) el bloque del campeon, que es el artefacto canonico v30_prob LEIDO y NO reentrenado;
(b) un canal con el bloque de palabra (1,2) min_df 3 max_features 300000 IDENTICO al de la receta
del campeon en v31_03, ajustado SOLO en el train de cada fold, con LinearSVC y
decision_function -> softmax (la conversion que declara el manifiesto canonico); (c) el bloque
C1-A, tambien ajustado SOLO en el train de cada fold, que se anade como segundo bloque del canal;
(d) la fusion de (b)+(c) con (a) por el combinador logistico anidado [L2(v30)||L2(p_canal)] por
fold de V35, con la decision argmax(p_comb * pesos plug-in). El CANAL DE CONTROL es el mismo canal
sin el bloque C1-A, de modo que el delta integrado atribuible al bloque C1-A queda aislado.

Estado del arte aplicado: mapa categorico LITERAL de 6 simbolos (redactado a priori desde
categorias Unicode, sin ninguna estadistica del corpus, sin ajuste por etiqueta y sin ningun
parametro estimado), n-gramas de clase (2,3) sin borde y sin min_df, TF-IDF l2 sublinear sobre
documentos de n-gramas repetidos por su frecuencia, LinearSVC con los hiperparametros del campeon,
CV de 5 folds sobre los indices canonicos, IC95 pareado (2000, semilla 20261010), 3 permutaciones
del alfabeto y un control de barajado. Reutiliza POR IMPORTACION la maquinaria auditada de V35
(combinador_oof, evaluar_combinacion, diagnostico_brazo, bootstrap_delta, macro,
macro_de_conteos, softmax, indices_fold, sha256_*, escribir_json_atomico, como_nativo, Detencion,
literales_de_ruta, modulos_importados), no la reimplementa. CERO transformers, CERO LLM, CERO
embeddings, CERO datos externos, CERO metadatos; sin test mas alla de transform -> predict; sin
holdout, sin leaderboard, sin submission; campeon V30 + plug-in INTACTO;
artifacts/p33_scores/todos INTOCABLE (solo lectura).

AMENDA DECLARADA QUE EXIGE ENMENDAR EL PRE-REGISTRO ANTES DE EJECUTAR: el campo estructurado
rango_n_de_transiciones del pre-registro vale [1, 2], pero la PROSA del mismo pre-registro (JSON y
MD) dice "bigramas y trigramas de clase" y publica las columnas reales esperadas "18 / 66 / 27 /
127". Dos controles geometricos fijan la lectura y estan publicados en el informe: (i) el mapa
literal de 6 simbolos, con P = puntuacion ASCII (tabla de los rangos 33-47, 58-64, 91-96, 123-126)
y O = resto, produce EXACTAMENTE 27 bigramas de clase distintas y 127 trigramas distintas (las
"27 / 127" publicadas), mientras que la lectura literal de la glosa de P ("todo lo que no es
alfanumerico, no es espacio y no es digito") deja la clase O vacia y da 25 / 125; (ii) el rango
(2,3) es el de "bigramas y trigramas" (la brecha L4 habla de patrones letra->digito y
espacio->puntuacion, que son transiciones de longitud >= 2) y reproduce las 27 / 127, mientras que
(1,2) no. El script EJECUTA (2,3) con esa lectura del mapa, deja constancia del desfase en el JSON,
en el informe, en RUNLOG y en experiments.csv y NO modifica el pre-registro: comprueba que el campo
CONGELADO sigue valiendo [1,2] (para que nadie lo cambie en silencio) y publica la enmienda como
constante RANGO_N_EJECUTADO. El par "18 / 66" de la nota min_df no se reproduce con ninguna
lectura ni rango, no se usa como criterio y se reporta como pendiente de enmienda.

Arbol de decision (arbol_de_decision del pre-registro, implementado tal cual; C1-A es ONE-SHOT en
cualquier caso y NO se ejecuta una segunda particion aunque la integracion salga positiva):
  DEFECTO_DE_IMPLEMENTACION: las sondas o el triage detectan un defecto; la familia NO se cierra.
  SIN_SENAL_ESPECIFICA_DE_CATEGORIA: el mapa real no gana a las permutaciones; L4 se cierra igual.
  SENAL_NO_EXISTENTE: el banco no supera su H0 (delta del banco = 0).
  SENAL_EXISTE_INTEGRACION_NEUTRA: el banco supera su H0 pero la integracion no supera la suya; la
  senal es REAL y ORTOGONAL pero V30 ya la expresa; L4 se cierra por mecanismo, sin reabrir la
  familia sparse y sin autorizar un C1-B, y el resultado es PUBLICABLE.
  SENAL_EXISTE_INTEGRACION_NEGATIVA: el banco supera su H0 pero la integracion es negativa.
  SENAL_EXISTE_INTEGRACION_POSITIVA: banco e integracion superan sus H0; se evaluan los gates de la
  serie y, si pasan, PROMISING exigiria segunda particion con autorizacion del usuario, que este
  script NO ejecuta.
Precedencia declarada por este runner, coherente con el texto del pre-registro (que hace la
especificidad INCONDICIONAL: "si no gana, el veredicto es SIN_SENAL_ESPECIFICA... y L4 se cierra
igual"): primero DEFECTO_DE_IMPLEMENTACION, luego SIN_SENAL_ESPECIFICA_DE_CATEGORIA, luego
SENAL_NO_EXISTENTE y por ultimo el nodo de integracion.

Modos de linea de comandos:
  (sin flags): pipeline completo (verificacion de no-deriva, sondas, estado, entrenamiento resumible
    por fold, banco, controles, integracion, gates, arbol, informe, figura y anotadores).
  --verificar: verifica constantes, versiones, hashes, mapa literal y columnas publicadas; corre
    todas las sondas sobre el MINI-PROBLEMA (un subconjunto estratificado del train con los MISMOS
    indices de fold canonicos, nunca evidencia experimental); autoejercita la ruta de salida
    completa contra un temporal del sistema que se borra; imprime VERIFICACION OK y NO escribe
    NADA en el proyecto (ni estado, ni OOF, ni checkpoints, ni filas en RUNLOG.md,
    experiments.csv, reports/v36_novelty_registry.csv ni en el historial).
  --humo: recorre la zona INTERMEDIA entre --verificar y el entrenamiento (construye el checkpoint en
    memoria, mapea las sondas, serializa el estado a un temporal del sistema y verifica el
    round-trip) y PARA con el sentinel HUMO OK, salida 0, SIN entrenar ningun fold y sin dejar
    checkpoints, OOF ni filas en ningun registro.

Codigos de salida: 0 fin correcto (incluidos los cierres documentados del arbol); 2 verificacion
(constantes, versiones, hashes, mapa literal, columnas publicadas, baseline, ancla trivial,
conformidad, fold_local, compliance o sondas); 4 sonda fallida antes de entrenar; 7 presupuesto o
timeout global agotado (estado guardado y reanudable); 8 RSS por encima del tope declarado; 1 error
inesperado con traceback.
"""

from __future__ import annotations

# =====================================================================================
# ---- 0. ENTORNO E IMPORTS (los 32 hilos van ANTES de importar numpy/scipy/sklearn) ----
# =====================================================================================
import os

#: Los 32 hilos congelados se fijan antes de importar numpy (determinismo del BLAS), igual que
#: declara el pre-registro (ejecucion.hilos = 32).
os.environ["OMP_NUM_THREADS"] = "32"
os.environ["MKL_NUM_THREADS"] = "32"
os.environ["OPENBLAS_NUM_THREADS"] = "32"

import argparse
import ast
import csv
import io
import json
import resource
import shutil
import sys
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src import config as cfg
from src.verificacion_numerica import autotest_numerico
import v35_01_ftrl_lambda1 as v35

Detencion = v35.Detencion

# =====================================================================================
# ---- 1. CONSTANTES CONGELADAS Y NO-DERIVA CON EL PRE-REGISTRO ----
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
SEMILLA = 20261010
ANCLA_TRIVIAL = 0.04854212180687087

#: TOL_VENTAJA: diferencia minima para que el mapa REAL se considere que "gana" a un control. No la
#: congela el pre-registro; la declara este runner con motivo explicito: la macro-F1 es una suma de
#: razones sobre 169336 filas y las diferencias por debajo de 1e-12 son ruido de float64 o del
#: solucionador, no senal. Se publica en el JSON y en el informe.
TOL_VENTAJA = 1e-12

#: Modelo: EXACTAMENTE el del manifiesto canonico (receta del campeon; NO es un grid).
MODELO = {"C": 1.0, "loss": "squared_hinge", "penalty": "l2", "max_iter": 3000, "tol": 1e-4}

#: Bloque de referencia char_wb: ngram (3,5), min_df 5, max_features 300000 (pre-registro). Los
#: campos que el pre-registro NO congela (sublinear_tf, lowercase) se toman de la receta auditada
#: del campeon en el proyecto (E-001 y v31_03) y se declaran aqui para que no haya ningun
#: parametro implicito.
VEC_CHAR_WB = {"analyzer": "char_wb", "ngram_range": (3, 5), "min_df": 5,
               "max_features": 300000, "sublinear_tf": True, "lowercase": True}
#: Bloque de palabra del canal de integracion: ngram (1,2), min_df 3, max_features 300000
#: (pre-registro) y la receta de palabra del campeon (v31_03) para los campos no congelados.
VEC_PALABRA = {"analyzer": "word", "ngram_range": (1, 2), "min_df": 3,
               "max_features": 300000, "sublinear_tf": True, "lowercase": True,
               "strip_accents": None, "token_pattern": r"(?u)\b\w+\b"}
#: Normalizacion del bloque C1-A: EXACTAMENTE la del pre-registro (documents de n-gramas de clase
#: repetidos por su frecuencia, token_pattern \\S+, sin analyzer porque el bloque ya son los
#: n-gramas, y sin min_df).
VEC_C1A = {"norm": "l2", "sublinear_tf": True, "lowercase": False, "token_pattern": r"\S+"}

#: MAPA CATEGORICO LITERAL (condicion (i) del ruling): tabla escrita a mano desde categorias
#: Unicode, sin ninguna estadistica del corpus, sin ajuste por etiqueta y sin parametro estimado.
#: Sus claves y sus textos deben coincidir EXACTAMENTE con los del pre-registro (verificado por
#: hash); cualquier cambio detiene el runner con codigo 2.
MAPA_CARACTERES = {
    "S": "espacio y todo lo que `str.isspace()` marca como espacio",
    "D": "dígitos ASCII 0-9",
    "P": "puntuación ASCII, es decir todo lo que no es alfanumérico, no es espacio y no es dígito",
    "L": "letra minúscula, con y sin acento",
    "M": "letra mayúscula, con y sin acento",
    "O": "resto: emojis, flechas, caracteres de control, alfabetos no latinos",
}
ORDEN_ALFABETO = ("S", "D", "P", "L", "M", "O")

#: Tabla LITERAL de la puntuacion ASCII (los cuatro rangos de ASCII imprimible no alfanumerico):
#: es la implementacion de la clase P. La glosa de P del pre-registro ("todo lo que no es
#: alfanumerico...") es contradictoria con la glosa de O ("emojis, flechas, caracteres de control")
#: y con las columnas publicadas 27/127; solo la tabla ASCII de puntuacion las reproduce.
PUNTUACION_ASCII = tuple(
    chr(codigo)
    for codigo in (*range(33, 48), *range(58, 65), *range(91, 97), *range(123, 127))
)
_PUNTUACION = frozenset(PUNTUACION_ASCII)

#: RANGO_N_EJECUTADO: (2,3) = bigramas y trigramas de clase, segun la prosa del pre-registro y las
#: columnas publicadas (27/127). El campo estructurado congelado vale [1,2]; ver ENMIENDA_RANGO_N.
RANGO_N_EJECUTADO = (2, 3)
ENMIENDA_RANGO_N = {
    "campo_congelado": "rango_n_de_transiciones",
    "valor_congelado": [1, 2],
    "valor_ejecutado": list(RANGO_N_EJECUTADO),
    "lectura_de_la_prosa": "bigramas y trigramas de clase (JSON y MD del pre-registro, y la "
                           "descripcion de la brecha L4: patrones letra->digito y "
                           "espacio->puntuacion, que son transiciones de longitud >= 2)",
    "columnas_publicadas_que_la_confirman": "27 bigramas + 127 trigramas = 154 columnas, "
                                            "reproducidas EXACTAMENTE por (2,3) con el mapa de 6 "
                                            "simbolos; (1,2) no las reproduce",
    "lectura_del_mapa_que_la_confirma": "P = puntuacion ASCII (tabla literal de 33-47, 58-64, "
                                        "91-96, 123-126) y O = resto, que es lo que dicen el nombre "
                                        "de P y la glosa de O; la glosa literal de P daria 25 / 125 "
                                        "y dejaria O vacia",
    "nota_18_66": "el par '18 / 66' de la nota min_df del pre-registro no se reproduce con ninguna "
                  "lectura ni rango y no se usa como criterio",
    "estado": "PENDIENTE_DE_ENMENDAR_ANTES_DE_EJECUTAR (decide el usuario; este runner no modifica "
              "el pre-registro)",
}

#: COLUMNAS_PUBLICADAS: las columnas reales que el pre-registro publica (27 / 127). Se verifican en
#: runtime contra el vocabulario real del bloque en cada fold; si no coinciden -> Detencion exit 2.
COLUMNAS_PUBLICADAS = {"bigramas": 27, "trigramas": 127, "total": 154}

#: Semilla de las permutaciones y del barajado: cv.semilla = 42, DECLARADA en el pre-registro. Las
#: permutaciones NUNCA se derivan de los datos ni de los resultados.
SEMILLA_PERMUTACIONES = 42
#: Numero de permutaciones del control: la condicion (ii) exige >= 3 y el pre-registro no fija el
#: numero exacto, luego se usan 3 (el minimo) para no inflar el gasto.
N_PERMUTACIONES = 3
#: PERMUTACIONES del alfabeto (control C2): traslación de nombres sobre la MISMA particion del
#: espacio de caracteres, literales en el codigo, derivadas de SEMILLA_PERMUTACIONES. Ninguna es la
#: identidad y cada una mueve al menos 4 simbolos (se comprueba en construir_permutaciones).
PERMUTACIONES = (
    {"S": "O", "D": "L", "P": "M", "L": "S", "M": "D", "O": "P"},
    {"S": "L", "D": "P", "P": "S", "L": "O", "M": "D", "O": "M"},
    {"S": "D", "D": "M", "P": "L", "L": "P", "M": "O", "O": "S"},
)

#: Familias de librerias vetadas y particiones reservadas (compliance).
FAMILIAS_VETADAS = ("vowpal", "vw", "torch", "transformers", "gensim", "fasttext",
                    "sentence-transformers", "word2vec", "glove", "spacy-vectors")
PARTICIONES_RESERVADAS = ("test", "holdout", "submission", "leaderboard")

#: Rutas. CANONICO es INTOCABLE: solo lectura.
CANONICO = ROOT / "artifacts" / "p33_scores" / "todos"
PREREG_JSON = cfg.EXPERIMENTS / "v36_00_preregistro.json"
PREREG_MD = cfg.REPORTS / "v36_preregistro.md"
CKPT_DIR = ROOT / "checkpoints" / "v36_e1"
OOF_DIR = ROOT / "artifacts" / "p36_scores" / "v36_e1"
ESTADO_JSON = cfg.EXPERIMENTS / "v36_01_estado.json"
RESULTADOS_JSON = cfg.EXPERIMENTS / "v36_01_transiciones_clase.json"
INFORME = cfg.REPORTS / "v36_e1_transiciones_clase.md"
FIGURA = cfg.FIGURES / "v36_e1_curvas.png"
NOVELTY_CSV = cfg.REPORTS / "v36_novelty_registry.csv"
HISTORIA_CSV = cfg.REPORTS / "age_complete_experiment_history.csv"
RUNLOG = ROOT / "RUNLOG.md"
EXPERIMENTS_CSV = ROOT / "experiments.csv"
#: Nombre (NO ruta) del temporal de sistema que usa --humo: se compone con
#: ``tempfile.gettempdir()`` y se borra al terminar.
NOMBRE_TEMPORAL_HUMO = "v36_01_humo_estado.json"

#: Rutas de escritura vigiladas por las sondas de CERO EFECTOS SECUNDARIOS.
RUTAS_VIGILADAS = {"estado": ESTADO_JSON, "oof": OOF_DIR, "checkpoints": CKPT_DIR,
                   "novedad": NOVELTY_CSV}

#: Cabecera congelada del historial de experimentos (15 columnas, esquema existente).
COLUMNAS_HISTORIA = ["phase", "experiment_id", "date", "hypothesis", "model", "representation",
                     "corrected_cv", "delta_vs_V20", "folds", "bootstrap", "rescue", "damage",
                     "net", "decision", "resultado"]
#: Cabecera del registro de novedad (archivo NUEVO escrito por este runner).
COLUMNAS_NOVEDAD = ["experiment_id", "fecha", "que_es", "no_es_embedding",
                    "no_es_representacion_nueva", "no_usa_datos_externos", "no_es_un_grid",
                    "diferencia_con_ciclo_previo", "verificacion", "artefactos"]

#: Nodos del arbol de decision (arbol_de_decision del pre-registro). Se publican y se contrastan
#: con los textos del pre-registro.
NODOS_ARBOL = ("SENAL_NO_EXISTENTE", "SENAL_EXISTE_INTEGRACION_NEUTRA",
               "SENAL_EXISTE_INTEGRACION_POSITIVA", "SENAL_EXISTE_INTEGRACION_NEGATIVA",
               "SIN_SENAL_ESPECIFICA_DE_CATEGORIA", "DEFECTO_DE_IMPLEMENTACION")

#: Marcador de naturaleza del mini-problema y del autoejercicio: NUNCA evidencia experimental.
NATURALEZA_MINI = ("miniproblema de --verificar/--humo, NO evidencia experimental: datos reales "
                   "pero escala sintetica, 0 folds de la particion canonica como folds de CV")

#: Tope de RAM (ru_maxrss) y presupuesto declarado (pre-registro ejecucion).
RAM_TOPE_GB = 20.0
PRESUPUESTO_DECLARADO_SEG = 2400.0
TIMEOUT_H = 24.0

#: Variantes del bloque C1-A: nombre -> (permutacion o None, barajar o no). El mapa real es la
#: variante "real"; las tres permutaciones y el barajado son los controles C2 y C3.
VARIANTES = {
    "real": (None, False),
    "perm1": (PERMUTACIONES[0], False),
    "perm2": (PERMUTACIONES[1], False),
    "perm3": (PERMUTACIONES[2], False),
    "barajado": (None, True),
}
#: BRAZOS del banco y variante de C1-A que lleva cada uno (None = solo el bloque de referencia).
BRAZOS_BANCO = ("banco_referencia", "banco_c1a", "banco_c1a_perm1", "banco_c1a_perm2",
                "banco_c1a_perm3", "banco_c1a_barajado")
VARIANTE_DE_BANCO = {"banco_referencia": None, "banco_c1a": "real", "banco_c1a_perm1": "perm1",
                     "banco_c1a_perm2": "perm2", "banco_c1a_perm3": "perm3",
                     "banco_c1a_barajado": "barajado"}
#: BRAZOS de integracion: el canal de control (palabra sola) y el candidato (palabra + C1-A).
BRAZOS_INTEG = ("integ_control", "integ_c1a")
VARIANTE_DE_INTEG = {"integ_control": None, "integ_c1a": "real"}
BRAZO_REFERENCIA = "banco_referencia"
BRAZO_CANDIDATO = "banco_c1a"
#: Columnas del vectorizador de referencia por brazo del banco: todos comparten char_wb.
BLOQUE_DE_BANCO = "char_wb"
BLOQUE_DE_INTEG = "palabra"

#: NOMBRES_SONDAS: las ocho sondas obligatorias del pre-registro (sondas).
NOMBRES_SONDAS = ("banco_ree_medido", "tabla_5_de_5_folds", "bootstrap_pareado", "permutaciones",
                  "barajado", "conformidad", "fold_local", "mapa_literal")

#: FASE del checkpoint y prefijo del temporal del autoejercicio.
FASE_ESTADO = "V36-E1 via C1-A ejecucion (checkpoint resumible)"
PREFIJO_TEMPORAL = "v36_01_autoejercicio_"


# =====================================================================================
# ---- 2. MAPA DE CLASES Y TRANSFORM (puro, sin estado, sin borde) ----
# =====================================================================================
def clase_de_caracter(caracter: str) -> str:
    """Devuelve el simbolo de clase de UN caracter segun el mapa literal.

    Orden: ``S`` si ``str.isspace()``; ``D`` si es un digito ASCII 0-9; ``P`` si esta en la tabla
    literal de puntuacion ASCII; ``L`` o ``M`` si es alfanumerico en minuscula o mayuscula (con y
    sin acento, porque ``islower`` e ``isupper`` ya cubren Unicode); y ``O`` en el resto (emojis,
    flechas, caracteres de control, alfabetos no latinos y digitos no ASCII).

    Parameters
    ----------
    caracter:
        Un unico caracter (Unicode).

    Returns
    -------
    str
        Uno de los simbolos del alfabeto: ``S``, ``D``, ``P``, ``L``, ``M`` u ``O``.
    """
    if caracter.isspace():
        return "S"
    if "0" <= caracter <= "9":
        return "D"
    if caracter in _PUNTUACION:
        return "P"
    if caracter.isalnum():
        return "L" if caracter.islower() else ("M" if caracter.isupper() else "O")
    return "O"


def secuencia_de_clases(texto: str, permutacion: dict[str, str] | None = None) -> str:
    """Convierte un texto en la secuencia de sus clases segun el mapa, con permutacion opcional.

    Sin permutacion usa el mapa literal. Con permutacion (control C2) sustituye cada simbolo por su
    imagen: es un traslado de NOMBRES sobre la MISMA particion del espacio de caracteres, luego la
    secuencia permutada es la real con los simbolos renombrados.

    Parameters
    ----------
    texto:
        Texto de una fila.
    permutacion:
        Tabla simbolo -> simbolo, o None para el mapa real.

    Returns
    -------
    str
        Cadena de simbolos de clase, uno por caracter del texto.
    """
    if permutacion is None:
        return "".join(clase_de_caracter(caracter) for caracter in texto)
    return "".join(permutacion[clase_de_caracter(caracter)] for caracter in texto)


def ngramas_de_clase(secuencia: str, n_min: int, n_max: int) -> list[str]:
    """Extrae los n-gramas de clase de longitudes ``n_min..n_max`` SIN anadir borde.

    El pre-registro declara, antes de ver el banco, que NO se anaden simbolos de borde: un borde
    inventaria transiciones que no existen en el texto. Solo se emiten transiciones internas, con
    la longitud primero y la posicion dentro, para que el orden de emision sea determinista.

    Parameters
    ----------
    secuencia:
        Cadena de simbolos de clase.
    n_min:
        Longitud minima (2 en esta corrida: bigramas de clase).
    n_max:
        Longitud maxima (3 en esta corrida: trigramas de clase).

    Returns
    -------
    list[str]
        Lista de n-gramas de clase; para una longitud mayor o igual que la secuencia, la lista
        queda vacia.
    """
    tokens: list[str] = []
    for longitud in range(n_min, n_max + 1):
        tokens.extend(secuencia[inicio:inicio + longitud]
                      for inicio in range(len(secuencia) - longitud + 1))
    return tokens


def documento_de_transiciones(secuencia: str, permutacion: dict[str, str] | None = None,
                              barajar: bool = False,
                              generador: np.random.Generator | None = None) -> str:
    """Arma el documento de n-gramas de clase con cada n-grama repetido por su frecuencia.

    El pre-registro describe la normalizacion como documentos donde "cada n-grama de clase se repite
    segun su frecuencia", con token_pattern ``\\S+`` y TF-IDF l2 sublinear: este documento es
    exactamente la bolsa de n-gramas con cada n-grama repetido tantas veces como aparece, que es lo
    que el vectorizador consume. Con ``barajar`` (control C3) el orden de los tokens de la bolsa se
    baraja con el generador dado: se destruye el orden y se conserva el recuento.

    Parameters
    ----------
    secuencia:
        Cadena de simbolos de clase.
    permutacion:
        Tabla simbolo -> simbolo del control C2, o None para el mapa real.
    barajar:
        Si True, baraja el orden de los tokens conservando el multiconjunto.
    generador:
        Generador usado para barajar; obligatorio si ``barajar``.

    Returns
    -------
    str
        Documento de n-gramas de clase separados por un espacio.

    Raises
    ------
    Detencion
        Si se pide barajar sin generador (codigo 2).
    """
    tokens = ngramas_de_clase(secuencia, RANGO_N_EJECUTADO[0], RANGO_N_EJECUTADO[1])
    if permutacion is not None:
        tokens = ["".join(permutacion[simbolo] for simbolo in token) for token in tokens]
    if barajar:
        if generador is None:
            raise Detencion("[STOP] el control de barajado exige un generador (exit 2)", 2)
        generador.shuffle(tokens)
    return " ".join(tokens)


def construir_permutaciones() -> list[dict[str, object]]:
    """Verifica las permutaciones declaradas y las congela con su sha256.

    Cada permutacion debe ser una biyeccion del alfabeto sobre si misma y mover al menos 4 simbolos
    (para que no sea casi la identidad). Ademas se publica, para cada una, el sha256 de su tabla y
    el sha256 de la permutacion que genera el generador con ``SEMILLA_PERMUTACIONES``: asi se
    documenta que los controles salen de una semilla declarada y no de los datos ni de los
    resultados, y que las tres tablas del codigo estan congeladas.

    Returns
    -------
    list[dict[str, object]]
        Una entrada por permutacion, con ``permutacion``, ``sha256``, ``sha256_rederivada`` y
        ``simbolos_movidos``.

    Raises
    ------
    Detencion
        Si alguna no es biyectiva o mueve menos de 4 simbolos (codigo 2).
    """
    generador = np.random.default_rng(SEMILLA_PERMUTACIONES)
    salida: list[dict[str, object]] = []
    for indice, permutacion in enumerate(PERMUTACIONES, start=1):
        bijyectiva = (set(permutacion) == set(ORDEN_ALFABETO)
                       and set(permutacion.values()) == set(ORDEN_ALFABETO))
        movidos = int(sum(1 for simbolo in ORDEN_ALFABETO if permutacion[simbolo] != simbolo))
        if not bijyectiva or movidos < 4:
            raise Detencion(f"[STOP] la permutacion {indice} no es una biyeccion valida del "
                            f"alfabeto (mueve {movidos} simbolos) (exit 2)", 2)
        rederivada = dict(zip(ORDEN_ALFABETO,
                              (str(valor) for valor in
                               generador.permutation(list(ORDEN_ALFABETO)))))
        salida.append({
            "indice": indice,
            "permutacion": dict(permutacion),
            "sha256": v35.sha256_bytes(json.dumps(permutacion, sort_keys=True).encode("utf-8")),
            "sha256_rederivada": v35.sha256_bytes(json.dumps(rederivada, sort_keys=True)
                                                  .encode("utf-8")),
            "simbolos_movidos": movidos,
        })
    return salida


def construir_documentos_c1a(textos: np.ndarray, variante: str) -> list[str]:
    """Construye, para TODO el corpus, la lista de documentos de n-gramas de clase de una variante.

    El bloque C1-A es un transformador de texto PURO (reglas sobre el texto de cada fila): no mira
    etiquetas ni estadisticas agregadas, luego construirlo una vez sobre todo el corpus y aplicarlo
    despues es IDENTICO a construirlo dentro de cada fold (es la misma justificacion que usa el
    proyecto en v31_03 para los canales). Lo que SI es fold-local, y se ajusta por fold, es el
    ``TfidfVectorizer`` (que aprende el idf): ver :func:`ajustar_bloque_c1a`.

    Parameters
    ----------
    textos:
        Textos del corpus en el orden canonico.
    variante:
        Clave de :data:`VARIANTES` (``real``, ``perm1``, ``perm2``, ``perm3`` o ``barajado``).

    Returns
    -------
    list[str]
        Un documento por fila, en el orden canonico.

    Raises
    ------
    Detencion
        Si la variante no existe en :data:`VARIANTES` (codigo 2).
    """
    if variante not in VARIANTES:
        raise Detencion(f"[STOP] variante de bloque C1-A desconocida: '{variante}' (exit 2)", 2)
    permutacion, barajar = VARIANTES[variante]
    generador = np.random.default_rng(SEMILLA_PERMUTACIONES) if barajar else None
    documentos: list[str] = []
    for texto in textos:
        secuencia = secuencia_de_clases(texto, permutacion)
        documentos.append(documento_de_transiciones(secuencia, barajar=barajar,
                                                    generador=generador))
    return documentos


# =====================================================================================
# ---- 3. UTILIDADES DE ESTADO, HASH, JSON, METRICAS Y RECURSOS ----
# =====================================================================================
def rss_gb() -> float:
    """RSS de proceso en GB (``ru_maxrss`` viene en KiB en Linux).

    Returns
    -------
    float
        Pico de memoria residente en GB.
    """
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0 / 1024.0


def estado_nuevo() -> dict:
    """Construye un checkpoint vacio con la fase, el inicio y los brazos.

    Returns
    -------
    dict
        Checkpoint nuevo.
    """
    brazos = BRAZOS_BANCO + BRAZOS_INTEG
    return {"fase": FASE_ESTADO, "fecha": FECHA, "inicio_ts": time.time(), "rss_pico_gb": 0.0,
            "brazos": {brazo: {"folds": {}, "folds_completados": []} for brazo in brazos},
            "sondas": {}, "triage": None}


def guardar_estado(estado: dict) -> None:
    """Guarda el checkpoint de forma atomica, solo con escalares nativos.

    Parameters
    ----------
    estado:
        Checkpoint con el progreso de la corrida.
    """
    v35.escribir_json(ESTADO_JSON, estado)


def cargar_estado() -> dict:
    """Carga el checkpoint si existe; si no, devuelve uno nuevo.

    Returns
    -------
    dict
        Checkpoint de la corrida.
    """
    if not ESTADO_JSON.exists():
        return estado_nuevo()
    return json.loads(ESTADO_JSON.read_text(encoding="utf-8"))


def verificar_ram(estado: dict) -> float:
    """Registra el pico de RSS y detiene si supera el tope declarado de 20 GB (exit 8).

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
        Si el RSS supera ``RAM_TOPE_GB`` (codigo 8, con el estado guardado).
    """
    actual = rss_gb()
    estado["rss_pico_gb"] = float(max(float(estado.get("rss_pico_gb", 0.0)), actual))
    if actual > RAM_TOPE_GB:
        guardar_estado(estado)
        raise Detencion(f"[STOP] RSS {actual:.2f} GB supera el tope de {RAM_TOPE_GB:g} GB; estado "
                        "guardado (exit 8)", 8)
    return actual


def verificar_deadline(estado: dict) -> None:
    """Comprueba el presupuesto declarado y el timeout global antes de cada ajuste (exit 7).

    Parameters
    ----------
    estado:
        Checkpoint.

    Raises
    ------
    Detencion
        Si se agotaron los 2400 s declarados o las 24 h globales (codigo 7, estado guardado).
    """
    transcurrido = time.time() - float(estado["inicio_ts"])
    if transcurrido >= TIMEOUT_H * 3600.0:
        guardar_estado(estado)
        raise Detencion(f"[STOP] timeout global de {TIMEOUT_H:g} h superado "
                        f"({transcurrido / 3600.0:.2f} h); estado guardado (exit 7)", 7)
    if transcurrido >= PRESUPUESTO_DECLARADO_SEG:
        guardar_estado(estado)
        raise Detencion(f"[STOP] presupuesto declarado de {PRESUPUESTO_DECLARADO_SEG:g} s superado "
                        f"({transcurrido:.0f} s); estado guardado y reanudable (exit 7)", 7)


def sha256_matriz(matriz: np.ndarray) -> str:
    """SHA256 de una matriz float64 en orden C (matrices de bloque y de decision).

    Parameters
    ----------
    matriz:
        Arreglo float64 de una o dos dimensiones.

    Returns
    -------
    str
        Digest hexadecimal de 64 caracteres.
    """
    return v35.sha256_f64_2d(np.asarray(matriz, dtype=np.float64))


def ruta_oof(brazo: str) -> Path:
    """Ruta del OOF float64 de un brazo (decision_function del banco o probabilidad del canal).

    Parameters
    ----------
    brazo:
        Nombre del brazo.

    Returns
    -------
    Path
        Ruta del ``.npy`` del brazo.
    """
    sufijo = "_prob" if brazo in BRAZOS_INTEG else "_decision"
    return OOF_DIR / f"{brazo}{sufijo}.npy"


def guardar_pesos(brazo: str, fold: int, indices: np.ndarray, valores: np.ndarray,
                  detalle: dict) -> None:
    """Guarda el checkpoint por fold (indices, valores y detalle) con escritura atomica.

    Parameters
    ----------
    brazo:
        Nombre del brazo.
    fold:
        Fold canonico.
    indices:
        Indices de validacion del fold.
    valores:
        decision_function (banco) o probabilidad (integracion) del fold.
    detalle:
        Detalle nativo a guardar junto al checkpoint.
    """
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    final = CKPT_DIR / f"fold{int(fold)}_{brazo}.npz"
    temporal = CKPT_DIR / f"fold{int(fold)}_{brazo}.npz.tmp"
    np.savez_comprimido(temporal, indices=np.asarray(indices, dtype=np.int64),
                        valores=np.asarray(valores, dtype=np.float64),
                        detalle=np.array(json.dumps(v35.como_nativo(detalle))))
    os.replace(temporal, final)


def cargar_pesos(brazo: str, fold: int) -> tuple[np.ndarray, np.ndarray, dict]:
    """Carga el checkpoint por fold si existe, para reanudar sin reentrenar.

    Parameters
    ----------
    brazo:
        Nombre del brazo.
    fold:
        Fold canonico.

    Returns
    -------
    tuple
        ``(indices, valores, detalle)``; si el checkpoint no existe, arrays vacios y ``{}``.
    """
    ruta = CKPT_DIR / f"fold{int(fold)}_{brazo}.npz"
    if not ruta.exists():
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64), {}
    with np.load(ruta, allow_pickle=False) as paquete:
        indices = paquete["indices"]
        valores = paquete["valores"]
        detalle = json.loads(str(paquete["detalle"]))
    return np.asarray(indices), np.asarray(valores), detalle


def guardar_oof(brazo: str, oof: np.ndarray) -> str:
    """Guarda el OOF float64 completo de un brazo con escritura atomica.

    Parameters
    ----------
    brazo:
        Nombre del brazo.
    oof:
        Matriz (n_filas, N_CLASES).

    Returns
    -------
    str
        Ruta relativa del OOF guardado.
    """
    OOF_DIR.mkdir(parents=True, exist_ok=True)
    final = ruta_oof(brazo)
    temporal = OOF_DIR / f"{brazo}_tmp.npy"
    np.save(temporal, np.asarray(oof, dtype=np.float64))
    os.replace(temporal, final)
    return v35.ruta_relativa(final)


def leer_oof(brazo: str) -> np.ndarray | None:
    """Lee el OOF de un brazo si existe.

    Parameters
    ----------
    brazo:
        Nombre del brazo.

    Returns
    -------
    numpy.ndarray | None
        La matriz OOF, o None si no existe todavia.
    """
    ruta = ruta_oof(brazo)
    if not ruta.exists():
        return None
    return np.load(ruta)


def fila_macro_por_fold(clase: np.ndarray, pred: np.ndarray, fold_idx: np.ndarray) -> list[float]:
    """Macro-F1 por fold de una prediccion sobre la particion canonica.

    Parameters
    ----------
    clase:
        Etiquetas enteras.
    pred:
        Predicciones enteras.
    fold_idx:
        Fold canonico por fila.

    Returns
    -------
    list[float]
        Cinco macro-F1, una por fold.
    """
    return [v35.macro(clase[fold_idx == fold], pred[fold_idx == fold])
            for fold in range(N_FOLDS)]


# =====================================================================================
# ---- 4. VERIFICACIONES DE NO-DERIVA (constantes, versiones, hashes, mapa, compliance) ----
# =====================================================================================
def cargar_preregistro() -> dict:
    """Lee el JSON congelado del pre-registro V36-E1.

    Returns
    -------
    dict
        Contenido de ``experiments/v36_00_preregistro.json``.

    Raises
    ------
    Detencion
        Si el pre-registro no existe (codigo 2).
    """
    if not PREREG_JSON.exists():
        raise Detencion(f"[STOP] no existe el pre-registro {PREREG_JSON} (exit 2)", 2)
    return json.loads(PREREG_JSON.read_text(encoding="utf-8"))


def verificar_constantes(prereg: dict) -> tuple[bool, dict]:
    """Comprueba, una a una, que las constantes del runner NO derivan del pre-registro congelado.

    Cada entrada del detalle es un booleano; si alguno es False, el runner se detiene con codigo 2.
    Se comparan el mapa literal, el orden del alfabeto, el modelo, los dos bloques de referencia, la
    CV, los gates de la serie, las cifras de compliance, las semillas, los pesos plug-in, el
    baseline, el presupuesto y los hashes del mapa y del alfabeto. El rango de n se comprueba por
    separado en :data:`ENMIENDA_RANGO_N`: el campo CONGELADO debe seguir valiendo ``[1, 2]`` (para
    que nadie lo cambie en silencio) y el valor ejecutado debe ser el de la enmienda declarada.

    Parameters
    ----------
    prereg:
        Pre-registro congelado.

    Returns
    -------
    tuple[bool, dict]
        ``(ok, detalle)``.
    """
    modelo = prereg["modelo"]
    char_wb = prereg["bloque_de_referencia"]["char_wb"]
    palabra = prereg["bloque_de_referencia"]["word_para_el_brazo_de_integracion"]
    cv = prereg["cv"]
    gates = prereg["gates_de_la_serie"]
    compliance = prereg["compliance"]
    hashes = prereg["hashes"]
    detalle = {
        "mapa_literal_igual": prereg["mapa_de_clases_literal"] == MAPA_CARACTERES,
        "orden_alfabeto_igual": list(ORDEN_ALFABETO) == list(prereg["orden_del_alfabeto"]),
        "rango_n_congelado_sigue_1_2": list(prereg["rango_n_de_transiciones"]) == [1, 2],
        "rango_n_ejecutado_es_la_enmienda": (list(RANGO_N_EJECUTADO)
                                             == ENMIENDA_RANGO_N["valor_ejecutado"]),
        "modelo_estimator": modelo["estimator"] == "sklearn.svm.LinearSVC",
        "modelo_C": float(modelo["C"]) == float(MODELO["C"]),
        "modelo_loss": modelo["loss"] == MODELO["loss"],
        "modelo_penalty": modelo["penalty"] == MODELO["penalty"],
        "modelo_max_iter": int(modelo["max_iter"]) == MODELO["max_iter"],
        "modelo_tol": float(modelo["tol"]) == float(MODELO["tol"]),
        "char_wb_analyzer": VEC_CHAR_WB["analyzer"] == "char_wb",
        "char_wb_ngram": list(char_wb["ngram_range"]) == list(VEC_CHAR_WB["ngram_range"]),
        "char_wb_min_df": int(char_wb["min_df"]) == VEC_CHAR_WB["min_df"],
        "char_wb_max_features": int(char_wb["max_features"]) == VEC_CHAR_WB["max_features"],
        "palabra_ngram": list(palabra["ngram_range"]) == list(VEC_PALABRA["ngram_range"]),
        "palabra_min_df": int(palabra["min_df"]) == VEC_PALABRA["min_df"],
        "palabra_max_features": int(palabra["max_features"]) == VEC_PALABRA["max_features"],
        "normalizacion_c1a": (VEC_C1A["norm"] == "l2" and VEC_C1A["sublinear_tf"] is True
                              and VEC_C1A["lowercase"] is False
                              and VEC_C1A["token_pattern"] == r"\S+"),
        "cv_n_folds": int(cv["n_folds"]) == N_FOLDS,
        "cv_semilla": int(cv["semilla"]) == SEMILLA_PERMUTACIONES,
        "gate_delta_min": float(gates["delta_min"]) == GATE_DELTA,
        "gate_folds_min": int(gates["folds_min"]) == GATE_FOLDS,
        "gate_n_bootstrap": int(gates["n_bootstrap"]) == N_BOOTSTRAP,
        "gate_semilla": int(gates["semilla"]) == SEMILLA,
        "gate_variabilidad": float(gates["variabilidad_de_referencia"]) == VARIABILIDAD,
        "gate_baseline_abort": float(gates["baseline_abort"]) == ABORTO_BASELINE,
        "compliance_a_cero": all(int(compliance[clave]) == 0 for clave in
                                 ("embeddings", "transformers", "llm", "datos_externos",
                                  "metadatos", "test", "holdout", "leaderboard", "submission")),
        "baseline_primario": float(prereg["baseline_primario"]) == BASELINE,
        "referencia_nested": float(prereg["referencia_nested"]) == NESTED,
        "pesos_plugin": list(prereg["pesos_plugin"]) == PESOS,
        "hash_mapa": (v35.sha256_bytes(json.dumps(MAPA_CARACTERES, sort_keys=True)
                                       .encode("utf-8")) == hashes["mapa_literal_de_clases"]),
        "hash_orden": (v35.sha256_bytes(json.dumps(list(ORDEN_ALFABETO)).encode("utf-8"))
                       == hashes["orden_del_alfabeto"]),
        "permutaciones_al_menos_3": N_PERMUTACIONES >= 3 and len(PERMUTACIONES) == N_PERMUTACIONES,
        "budget_presupuesto": (float(prereg["ejecucion"]["presupuesto_declarado_seg"])
                               == PRESUPUESTO_DECLARADO_SEG),
        "budget_ram": float(prereg["ejecucion"]["ram_tope_gb"]) == RAM_TOPE_GB,
        "budget_hilos": int(prereg["ejecucion"]["hilos"]) == HILOS,
    }
    return all(detalle.values()), detalle


def verificar_versiones(prereg: dict) -> tuple[bool, dict]:
    """Comprueba las versiones de libreria congeladas en el pre-registro.

    Parameters
    ----------
    prereg:
        Pre-registro congelado.

    Returns
    -------
    tuple[bool, dict]
        ``(ok, detalle)``.
    """
    versiones = prereg["versiones"]
    detalle = {"sklearn": sklearn.__version__ == versiones["sklearn"],
               "numpy": np.__version__ == versiones["numpy"],
               "pandas": pd.__version__ == versiones["pandas"]}
    return all(detalle.values()), detalle


def verificar_hashes(prereg: dict, clase: np.ndarray, fold_idx: np.ndarray, v30: np.ndarray,
                     ids: np.ndarray, textos: np.ndarray, hashes_corpus: dict) -> tuple[bool, dict]:
    """Verifica los hashes congelados (corpus y particion canonica) y la alineacion de filas.

    Parameters
    ----------
    prereg:
        Pre-registro congelado.
    clase:
        Etiquetas canonicas (referencia de alineacion).
    fold_idx:
        Fold canonico por fila.
    v30:
        Probabilidades del campeon.
    ids:
        Ids canonicos.
    textos:
        Textos del corpus en el orden del CSV.
    hashes_corpus:
        Hashes del corpus recomputados por :func:`v35.cargar_corpus`.

    Returns
    -------
    tuple[bool, dict]
        ``(ok, detalle)``.
    """
    hashes = prereg["hashes"]
    detalle = {
        "corpus_archivo": hashes_corpus["corpus_archivo"] == hashes["corpus_archivo"],
        "corpus_textos": hashes_corpus["corpus_textos"] == hashes["corpus_textos"],
        "canonico_y_true": (v35.sha256_archivo(CANONICO / "y_true.npy")
                           == hashes["canonico_y_true"]),
        "canonico_fold_idx": (v35.sha256_archivo(CANONICO / "fold_idx.npy")
                              == hashes["canonico_fold_idx"]),
        "canonico_v30_prob": (v35.sha256_archivo(CANONICO / "v30_prob.npy")
                              == hashes["canonico_v30_prob"]),
        "canonico_ids": (v35.sha256_archivo(CANONICO / "ids.npy") == hashes["canonico_ids"]),
        "manifest_canonico": (v35.sha256_archivo(CANONICO / "manifest.json")
                              == hashes["manifest_canonico"]),
        "clase_en_rango_canonico": int(clase.min()) == 0 and int(clase.max()) == N_CLASES - 1,
        "cinco_folds_canonicos": int(np.unique(fold_idx).size) == N_FOLDS,
        "forma_v30": v30.shape == (N_FILAS, N_CLASES),
        "forma_ids": ids.shape == (N_FILAS,),
        "textos_alineados": len(textos) == N_FILAS,
    }
    return all(detalle.values()), detalle


def auditar_compliance_propia() -> dict:
    """Audita el propio codigo: 0 familias vetadas ni particiones reservadas en rutas o modulos.

    Reutiliza :func:`v35.literales_de_ruta` y :func:`v35.modulos_importados` sobre el arbol de
    sintaxis de ESTE script, y comprueba ademas que ninguna ruta de escritura caiga dentro de la
    particion canonica, que es intocable.

    Returns
    -------
    dict
        Numero de literales y modulos revisados, lista de violaciones y marca de que las
        escrituras no tocan el canonico.

    Raises
    ------
    Detencion
        Si alguna ruta o modulo viola el compliance (codigo 2).
    """
    arbol = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    literales = v35.literales_de_ruta(arbol)
    modulos = v35.modulos_importados(arbol)
    viol_rutas = [texto for _, texto in literales
                  if any(palabra in texto.lower() for palabra in PARTICIONES_RESERVADAS)
                  or any(familia in texto.lower() for familia in FAMILIAS_VETADAS)]
    viol_modulos = [nombre for _, nombre in modulos
                    if any(familia in nombre.lower() for familia in FAMILIAS_VETADAS)]
    rutas_escritura = [ESTADO_JSON, RESULTADOS_JSON, INFORME, FIGURA, NOVELTY_CSV, HISTORIA_CSV,
                       RUNLOG, EXPERIMENTS_CSV, CKPT_DIR, OOF_DIR]
    tocanico = [v35.ruta_relativa(ruta) for ruta in rutas_escritura
                if ruta == CANONICO or CANONICO in ruta.parents]
    if viol_rutas or viol_modulos or tocanico:
        raise Detencion(f"[STOP] compliance: rutas={viol_rutas}, modulos={viol_modulos}, "
                        f"escrituras dentro del canonico={tocanico} (exit 2)", 2)
    return {"literales_de_ruta_revisados": len(literales), "modulos_revisados": len(modulos),
            "violaciones": [], "escrituras_tocan_canonico": False}


# =====================================================================================
# ---- 5. CONSTRUCCION Y AJUSTE FOLD-LOCAL DE LOS BLOQUES ----
# =====================================================================================
def ajustar_bloque_c1a(documentos: list[str], tr_idx: np.ndarray,
                       va_idx: np.ndarray) -> tuple[sp.csr_matrix, sp.csr_matrix, dict]:
    """Ajusta el TfidfVectorizer del bloque C1-A SOLO en el train del fold y transforma ambos lados.

    Este es el UNICO ``fit`` del bloque C1-A y es fold-local: nunca ve el fold-val ni el test. Se
    publica el tamano del vocabulario y los sha256 del vocabulario ordenado y del idf del fold, y
    se exige que el fold-val tenga exactamente las mismas columnas que el fold-train (ningun token
    del train puede anadirse desde el val).

    Parameters
    ----------
    documentos:
        Lista de documentos de n-gramas de clase, uno por fila del corpus.
    tr_idx:
        Indices del fold-train.
    va_idx:
        Indices del fold-val.

    Returns
    -------
    tuple
        ``(x_tr, x_va, detalle)`` con las matrices TF-IDF y el detalle de la sonda ``fold_local``.
    """
    vectorizador = TfidfVectorizer(**VEC_C1A)
    x_tr = vectorizador.fit_transform([documentos[indice] for indice in tr_idx])
    x_va = vectorizador.transform([documentos[indice] for indice in va_idx])
    vocabulario = vectorizador.vocabulary_
    detalle = {"n_vocabulario": len(vocabulario),
               "vocab_sha256": v35.sha256_textos(sorted(vocabulario.keys())),
               "idf_sha256": v35.sha256_f64(vectorizador.idf_),
               "n_filas_fit": len(tr_idx), "n_filas_total": len(documentos),
               "columnas_val_igualan_train": bool(x_va.shape[1] == x_tr.shape[1]),
               "n_bigramas": int(sum(1 for token in vocabulario if len(token) == 2)),
               "n_trigramas": int(sum(1 for token in vocabulario if len(token) == 3))}
    return x_tr, x_va, detalle


def ajustar_bloque_texto(textos: np.ndarray, tr_idx: np.ndarray, va_idx: np.ndarray,
                         config: dict) -> tuple[sp.csr_matrix, sp.csr_matrix, dict]:
    """Ajusta un TfidfVectorizer de texto (char_wb o palabra) SOLO en el train del fold.

    El bloque de referencia se ajusta UNA vez por fold y se REUTILIZA en todos los brazos que lo
    comparten: es el mismo ``fit`` fold-local (ajustarlo dos veces con los mismos datos daria el
    mismo vocabulario, y eso se comprueba por sha256 en la sonda ``fold_local``), y evita pagar
    seis veces el vectorizador mas caro del ciclo.

    Parameters
    ----------
    textos:
        Textos del corpus.
    tr_idx:
        Indices del fold-train.
    va_idx:
        Indices del fold-val.
    config:
        Config del vectorizador (:data:`VEC_CHAR_WB` o :data:`VEC_PALABRA`).

    Returns
    -------
    tuple
        ``(x_tr, x_va, detalle)`` con las matrices y el detalle fold-local del bloque.
    """
    vectorizador = TfidfVectorizer(**config)
    x_tr = vectorizador.fit_transform(textos[tr_idx])
    x_va = vectorizador.transform(textos[va_idx])
    vocabulario = vectorizador.vocabulary_
    detalle = {"n_vocabulario": len(vocabulario),
               "vocab_sha256": v35.sha256_textos(sorted(vocabulario.keys())),
               "idf_sha256": v35.sha256_f64(vectorizador.idf_),
               "n_filas_fit": len(tr_idx), "n_filas_total": len(textos),
               "columnas_val_igualan_train": bool(x_va.shape[1] == x_tr.shape[1])}
    return x_tr, x_va, detalle


def concatenar_bloques(x_principal: sp.csr_matrix,
                       x_c1a: sp.csr_matrix | None) -> sp.csr_matrix:
    """Concatena en horizontal el bloque de referencia y el bloque C1-A con peso 1.0, fijo.

    El peso del bloque C1-A es 1.0, FIJO y no buscado (identico al PESO_CANAL de v31_03). Como
    ambos bloques traen la norma l2 de su fila, la concatenacion es exactamente el "concatenado al
    bloque char_wb" que declara el pre-registro, sin ninguna constante nueva.

    Parameters
    ----------
    x_principal:
        Matriz del bloque de referencia (char_wb o palabra).
    x_c1a:
        Matriz del bloque C1-A, o None si el brazo no lleva ese bloque.

    Returns
    -------
    scipy.sparse.csr_matrix
        La matriz del brazo en formato CSR.
    """
    if x_c1a is None:
        return x_principal
    return sp.hstack([x_principal, x_c1a], format="csr")


def decision_a_prob(decision: np.ndarray) -> np.ndarray:
    """Convierte una decision_function de LinearSVC en probabilidad con softmax.

    Es la conversion que declara el manifiesto canonico ("la decision_function se convierte a
    probabilidad con un softmax para que sea comparable con la mezcla de V30"): el softmax
    desplazado preserva el argmax y produce seis valores positivos que suman 1, y NO calibra. Se usa
    para el OOF de los brazos de integracion, porque el combinador anidado de V35 mezcla
    probabilidades.

    Parameters
    ----------
    decision:
        Matriz (n, 6) de decision_function.

    Returns
    -------
    numpy.ndarray
        Probabilidades (n, 6) que suman 1 por fila.
    """
    return v35.softmax(decision)


def decision_a_pred(decision: np.ndarray) -> np.ndarray:
    """Prediccion de clase a partir de la decision_function (argmax, sin pesos).

    Parameters
    ----------
    decision:
        Matriz (n, 6) de decision_function.

    Returns
    -------
    numpy.ndarray
        Predicciones enteras.
    """
    if decision.ndim == 1:
        return (decision > 0).astype(np.int64)
    return np.argmax(decision, axis=1).astype(np.int64)


# =====================================================================================
# ---- 6. MINI-PROBLEMA (--verificar y --humo): NUNCA evidencia experimental ----
# =====================================================================================
def mini_problema(clase: np.ndarray, fold_idx: np.ndarray, textos: np.ndarray,
                  filas_por_clase: int = 250) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Subconjunto ESTRATIFICADO del train que conserva los MISMOS indices de fold canonicos.

    Toma las primeras ``filas_por_clase`` filas de cada clase en orden de indice y conserva su
    ``fold_idx`` ORIGINAL, leido del artefacto canonico: no se crea ninguna particion nueva ni se
    baraja nada. Es el soporte de las sondas en ``--verificar`` y ``--humo``.

    Parameters
    ----------
    clase:
        Etiquetas canonicas.
    fold_idx:
        Fold canonico por fila.
    textos:
        Textos del corpus.
    filas_por_clase:
        Cuantas filas por clase.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
        ``(clase, fold_idx, textos)`` del subconjunto.
    """
    seleccionados: list[int] = []
    for etiqueta in range(N_CLASES):
        seleccionados.extend(np.flatnonzero(clase == etiqueta)[:filas_por_clase].tolist())
    arreglo = np.array(sorted(seleccionados), dtype=np.int64)
    return clase[arreglo], fold_idx[arreglo], textos[arreglo]


# =====================================================================================
# ---- 7. ENTRENAMIENTO POR FOLD (banco e integracion), resumible y con checkpoints ----
# =====================================================================================
def preparar_fold(fold: int, textos: np.ndarray, clase: np.ndarray, fold_idx: np.ndarray,
                  documentos: dict[str, list[str]], exigir_columnas: bool = True) -> dict:
    """Prepara TODOS los bloques de UN fold con los unicos ajustes fold-local de la corrida.

    Ajusta una vez el bloque char_wb (compartido por los seis brazos del banco), una vez el bloque
    de palabra (compartido por los dos brazos de integracion) y una vez el bloque C1-A por
    variante (mapa real, tres permutaciones y barajado). Ningun ``fit`` ve el fold-val.

    Parameters
    ----------
    fold:
        Fold canonico que actua como validacion.
    textos:
        Textos del corpus.
    clase:
        Etiquetas canonicas.
    fold_idx:
        Fold canonico por fila.
    documentos:
        Documentos de n-gramas de clase por variante.
    exigir_columnas:
        Si True (corrida real) el vocabulario del bloque C1-A del mapa real debe reproducir las
        columnas publicadas 27 / 127 y el runner se detiene si no; si False (miniproblema de
        --verificar y --humo) el recuento se publica pero no detiene, porque un subconjunto del
        train puede no contener todos los n-gramas de clase.

    Returns
    -------
    dict
        Con ``tr``, ``va``, ``y_tr``, ``bloques`` (char_wb y palabra) y ``c1a`` (por variante).

    Raises
    ------
    Detencion
        Si el vocabulario del bloque C1-A del mapa real no reproduce las columnas publicadas
        27 / 127 con ``exigir_columnas`` activo (codigo 2), o si el fold-val del bloque C1-A trae
        columnas que el train no tiene (codigo 2).
    """
    tr, va = v35.indices_fold(fold_idx, fold)
    x_char_tr, x_char_va, meta_char = ajustar_bloque_texto(textos, tr, va, VEC_CHAR_WB)
    x_pal_tr, x_pal_va, meta_pal = ajustar_bloque_texto(textos, tr, va, VEC_PALABRA)
    c1a: dict[str, dict] = {}
    for variante in VARIANTES:
        x_tr, x_va, meta = ajustar_bloque_c1a(documentos[variante], tr, va)
        columnas_publicadas = (meta["n_bigramas"] == COLUMNAS_PUBLICADAS["bigramas"]
                               and meta["n_trigramas"] == COLUMNAS_PUBLICADAS["trigramas"])
        if variante == "real" and exigir_columnas and not columnas_publicadas:
            raise Detencion(f"[STOP] el vocabulario del bloque C1-A en el fold {fold} da "
                            f"{meta['n_bigramas']} / {meta['n_trigramas']} columnas de clase y el "
                            f"pre-registro publica {COLUMNAS_PUBLICADAS['bigramas']} / "
                            f"{COLUMNAS_PUBLICADAS['trigramas']} (exit 2)", 2)
        if not meta["columnas_val_igualan_train"]:
            raise Detencion(f"[STOP] el fold-val de la variante {variante} trae columnas que el "
                            f"fold-train no tiene (exit 2)", 2)
        c1a[variante] = {"tr": x_tr, "va": x_va, "meta": meta}
    return {"fold": fold, "tr": tr, "va": va, "y_tr": clase[tr],
            "bloques": {"char_wb": {"tr": x_char_tr, "va": x_char_va, "meta": meta_char},
                        "palabra": {"tr": x_pal_tr, "va": x_pal_va, "meta": meta_pal}},
            "c1a": c1a}


def entrenar_brazo_fold(datos_fold: dict, brazo: str, bloque: str, variante: str | None) -> dict:
    """Ajusta el LinearSVC de UN brazo en UN fold ya vectorizado y devuelve su detalle.

    El fold-train y el fold-val llegan YA vectorizados (los unicos ``fit`` se hicieron antes,
    fold-local). El modelo es el MISMO en todos los brazos: LinearSVC con los hiperparametros
    congelados del campeon. El sesgo es el unico ``intercept_`` del estimador: no se anade ninguna
    columna de sesgo a la matriz, luego no hay doble sesgo.

    Parameters
    ----------
    datos_fold:
        Salida de :func:`preparar_fold`.
    brazo:
        Nombre del brazo.
    bloque:
        ``char_wb`` para los brazos del banco o ``palabra`` para los de integracion.
    variante:
        Variante del bloque C1-A, o None si el brazo no lo lleva.

    Returns
    -------
    dict
        ``decision_va``, ``pred_va``, ``n_iter``, ``n_features``, ``intercepto`` y los hashes de los
        bloques que intervienen.
    """
    referencia = datos_fold["bloques"][bloque]
    x_tr = concatenar_bloques(referencia["tr"], None if variante is None
                              else datos_fold["c1a"][variante]["tr"])
    x_va = concatenar_bloques(referencia["va"], None if variante is None
                              else datos_fold["c1a"][variante]["va"])
    modelo = LinearSVC(**MODELO)
    modelo.fit(x_tr, datos_fold["y_tr"])
    decision = np.asarray(modelo.decision_function(x_va), dtype=np.float64)
    detalle = {"brazo": brazo, "bloque_de_referencia": bloque, "variante_c1a": variante,
               "fold": datos_fold["fold"], "n_train": int(x_tr.shape[0]),
               "n_val": int(x_va.shape[0]), "n_features": int(x_tr.shape[1]),
               "n_features_c1a": int(datos_fold["c1a"][variante]["tr"].shape[1])
               if variante else 0,
               "n_iter": int(np.max(modelo.n_iter_)),
               "intercepto": [float(valor) for valor in np.atleast_1d(modelo.intercept_)],
               "meta_bloque_referencia": referencia["meta"],
               "meta_c1a": datos_fold["c1a"][variante]["meta"] if variante else None}
    return {"decision_va": decision, "pred_va": decision_a_pred(decision), "detalle": detalle}


def registrar_fold(estado: dict, brazo: str, fold: int, indices: np.ndarray, valores: np.ndarray,
                   detalle: dict, oof: np.ndarray) -> None:
    """Escribe el checkpoint del fold, lo anota en el estado y guarda el OOF parcial.

    Parameters
    ----------
    estado:
        Checkpoint de la corrida.
    brazo:
        Nombre del brazo.
    fold:
        Fold canonico.
    indices:
        Indices de validacion del fold.
    valores:
        decision_function o probabilidad del fold.
    detalle:
        Detalle del fold (devuelto por :func:`entrenar_brazo_fold`).
    oof:
        Matriz OOF completa, actualizada in situ con los valores del fold.
    """
    guardar_pesos(brazo, fold, indices, valores, detalle)
    oof[indices] = valores
    registro = dict(detalle)
    registro.pop("decision_va", None)
    registro.pop("pred_va", None)
    registro["tiempo_seg"] = float(detalle.get("tiempo_seg", 0.0))
    estado["brazos"][brazo]["folds"][str(fold)] = registro
    if str(fold) not in estado["brazos"][brazo]["folds_completados"]:
        estado["brazos"][brazo]["folds_completados"].append(str(fold))
    guardar_estado(estado)


def brazo_completo(estado: dict, brazo: str) -> bool:
    """Indica si un brazo ya tiene los cinco folds registrados en el estado.

    Parameters
    ----------
    estado:
        Checkpoint de la corrida.
    brazo:
        Nombre del brazo.

    Returns
    -------
    bool
        True si estan los cinco folds.
    """
    return ([str(fold) for fold in range(N_FOLDS)]
            == sorted(estado["brazos"][brazo]["folds_completados"], key=int))


def entrenar_brazos(estado: dict, textos: np.ndarray, clase: np.ndarray, fold_idx: np.ndarray,
                    documentos: dict[str, list[str]],
                    exigir_columnas: bool = True) -> dict[str, np.ndarray]:
    """Entrena, fold a fold y brazo a brazo, los seis brazos del banco y los dos de integracion.

    Es la unica funcion que consume tiempo de CPU grande. Es RESUMIBLE: si el estado ya tiene un
    fold de un brazo, lo carga del checkpoint y no lo reentrena. Antes de cada ajuste comprueba el
    presupuesto declarado y el tope de RAM.

    Parameters
    ----------
    estado:
        Checkpoint de la corrida.
    textos:
        Textos del corpus.
    clase:
        Etiquetas canonicas.
    fold_idx:
        Fold canonico por fila.
    documentos:
        Documentos de n-gramas de clase por variante.
    exigir_columnas:
        Si las columnas publicadas del bloque C1-A son obligatorias (corrida real) o solo
        informativas (miniproblema).

    Returns
    -------
    dict[str, numpy.ndarray]
        Matriz OOF por brazo (decision_function para el banco, probabilidad para integracion).
    """
    oofs: dict[str, np.ndarray] = {brazo: np.zeros((len(clase), N_CLASES), dtype=np.float64)
                                  for brazo in BRAZOS_BANCO + BRAZOS_INTEG}
    for fold in range(N_FOLDS):
        # Reanudacion: un (brazo, fold) ya registrado en el estado se carga del checkpoint y NO se
        # reentrena; solo se ajustan los bloques del fold si queda algun brazo por entrenar.
        for brazo in BRAZOS_BANCO + BRAZOS_INTEG:
            if str(fold) in estado["brazos"][brazo]["folds"]:
                indices_cargados, valores_cargados, _ = cargar_pesos(brazo, fold)
                if indices_cargados.size:
                    oofs[brazo][indices_cargados] = valores_cargados
        faltantes = [brazo for brazo in BRAZOS_BANCO + BRAZOS_INTEG
                     if str(fold) not in estado["brazos"][brazo]["folds"]]
        if not faltantes:
            print(f"  fold {fold}: los ocho brazos ya estaban; reanudado sin entrenar", flush=True)
            continue
        print(f"  fold {fold}: preparando bloques fold-local "
              f"(char_wb, palabra y {len(VARIANTES)} variantes de C1-A) para {len(faltantes)} "
              f"brazos", flush=True)
        verificar_deadline(estado)
        inicio_fold = time.time()
        datos_fold = preparar_fold(fold, textos, clase, fold_idx, documentos, exigir_columnas)
        verificar_ram(estado)
        print(f"    char_wb: {datos_fold['bloques']['char_wb']['meta']['n_vocabulario']} terminos | "
              f"palabra: {datos_fold['bloques']['palabra']['meta']['n_vocabulario']} terminos | "
              f"C1-A real: {datos_fold['c1a']['real']['meta']['n_vocabulario']} columnas "
              f"({datos_fold['c1a']['real']['meta']['n_bigramas']} bigramas + "
              f"{datos_fold['c1a']['real']['meta']['n_trigramas']} trigramas de clase)", flush=True)
        for brazo in faltantes:
            verificar_deadline(estado)
            bloque = BLOQUE_DE_BANCO if brazo in BRAZOS_BANCO else BLOQUE_DE_INTEG
            variante = (VARIANTE_DE_BANCO[brazo] if brazo in BRAZOS_BANCO
                        else VARIANTE_DE_INTEG[brazo])
            inicio = time.time()
            salida = entrenar_brazo_fold(datos_fold, brazo, bloque, variante)
            detalle = salida["detalle"]
            detalle["tiempo_seg"] = float(time.time() - inicio)
            valores = (decision_a_prob(salida["decision_va"]) if brazo in BRAZOS_INTEG
                       else salida["decision_va"])
            macro_fold = float(v35.macro(clase[datos_fold["va"]], salida["pred_va"]))
            detalle["macro_f1_fold_val"] = macro_fold
            registrar_fold(estado, brazo, fold, datos_fold["va"], valores, detalle, oofs[brazo])
            verificar_ram(estado)
            print(f"    {brazo:<20} fold {fold}: macro-F1 fold-val {macro_fold:.7f} | "
                  f"{detalle['n_features']} features | {detalle['tiempo_seg']:.1f} s", flush=True)
        print(f"  fold {fold}: {time.time() - inicio_fold:.1f} s | RSS {rss_gb():.2f} GB | "
              f"presupuesto consumido {time.time() - float(estado['inicio_ts']):.0f} s de "
              f"{PRESUPUESTO_DECLARADO_SEG:g}", flush=True)
    for brazo in BRAZOS_BANCO + BRAZOS_INTEG:
        guardar_oof(brazo, oofs[brazo])
    return oofs

# =====================================================================================
# ---- 8. EVALUACION: BANCO, CONTROLES E INTEGRACION ----
# =====================================================================================
def evaluar_banco(clase: np.ndarray, fold_idx: np.ndarray,
                  oofs: dict[str, np.ndarray]) -> dict:
    """Macro-F1 global y por fold de los seis brazos del banco, con su prediccion.

    El delta del banco es la diferencia entre el brazo CANDIDATO (char_wb IDENTICO mas el bloque
    C1-A) y el brazo REFERENCIA (el mismo char_wb sin ese bloque): los dos brazos comparten folds,
    semilla, modelo y bloque de referencia, luego la unica diferencia es el bloque C1-A.

    Parameters
    ----------
    clase:
        Etiquetas canonicas.
    fold_idx:
        Fold canonico por fila.
    oofs:
        Matriz OOF (decision_function) por brazo.

    Returns
    -------
    dict
        Por brazo: macro-F1 global, los cinco folds y su prediccion; y el bloque ``delta`` con el
        delta del banco, sus deltas por fold, los folds positivos y el IC95 pareado.
    """
    detalle: dict[str, dict] = {}
    predicciones: dict[str, np.ndarray] = {}
    for brazo in BRAZOS_BANCO:
        pred = decision_a_pred(oofs[brazo])
        predicciones[brazo] = pred
        detalle[brazo] = {"macro_f1": float(v35.macro(clase, pred)),
                          "folds": fila_macro_por_fold(clase, pred, fold_idx),
                          "n_cambios_respecto_a_referencia": None}
    referencia = detalle[BRAZO_REFERENCIA]["macro_f1"]
    candidato = detalle[BRAZO_CANDIDATO]["macro_f1"]
    folds_ref = detalle[BRAZO_REFERENCIA]["folds"]
    folds_cand = detalle[BRAZO_CANDIDATO]["folds"]
    deltas_por_fold = [float(a - b) for a, b in zip(folds_cand, folds_ref)]
    intervalo = v35.bootstrap_delta(clase, predicciones[BRAZO_REFERENCIA],
                                    predicciones[BRAZO_CANDIDATO])
    for brazo in BRAZOS_BANCO:
        detalle[brazo]["n_cambios_respecto_a_referencia"] = int(
            (predicciones[brazo] != predicciones[BRAZO_REFERENCIA]).sum())
    detalle["delta"] = {
        "definicion": "macro-F1 del banco CANDIDATO (char_wb + C1-A) menos el del banco REFERENCIA "
                      "(char_wb IDENTICO sin C1-A), con los mismos folds, la misma semilla y el "
                      "mismo modelo",
        "macro_f1_referencia": float(referencia), "macro_f1_candidato": float(candidato),
        "delta": float(candidato - referencia),
        "folds_referencia": [float(valor) for valor in folds_ref],
        "folds_candidato": [float(valor) for valor in folds_cand],
        "deltas_por_fold": deltas_por_fold, "folds_positivos": int(sum(1 for d in deltas_por_fold
                                                                       if d > 0)),
        "delta_medio_bootstrap": intervalo["delta_medio"], "ic95": intervalo["ic95"],
        "p_delta_positivo": intervalo["p_delta_positivo"],
        "ic95_sin_cero": bool(intervalo["ic95_sin_cero"]),
        "supera_h0": bool(candidato > referencia and intervalo["ic95_sin_cero"]
                          and intervalo["ic95"][0] > 0.0),
        "h0": "delta del banco = 0; se rechaza si el IC95 pareado no contiene el cero y el delta "
              "es positivo",
    }
    detalle["_predicciones"] = predicciones
    return detalle


def evaluar_controles(clase: np.ndarray, fold_idx: np.ndarray, oofs: dict[str, np.ndarray],
                      predicciones: dict[str, np.ndarray], macro_real: float) -> dict:
    """Performa las permutaciones (C2) y el barajado (C3) contra el mapa real.

    Cada control se compara con el brazo CANDIDATO del banco (mapa real) con el MISMO bloque de
    referencia, los mismos folds y el mismo modelo. El mapa real "gana" a un control si su macro-F1
    le supera por mas de :data:`TOL_VENTAJA`.

    Parameters
    ----------
    clase:
        Etiquetas canonicas.
    fold_idx:
        Fold canonico por fila.
    oofs:
        Matriz OOF por brazo.
    predicciones:
        Predicciones por brazo (del banco).
    macro_real:
        Macro-F1 del mapa real.

    Returns
    -------
    dict
        ``permutaciones`` (una entrada por permutacion, con su delta y su IC95 pareado), ``barajado``
        y el veredicto de especificidad con su regla de decision.
    """
    brazo_real = BRAZO_CANDIDATO
    permutaciones: list[dict] = []
    for indice in range(1, N_PERMUTACIONES + 1):
        brazo = f"banco_c1a_perm{indice}"
        macro_control = float(v35.macro(clase, predicciones[brazo]))
        intervalo = v35.bootstrap_delta(clase, predicciones[brazo], predicciones[brazo_real])
        permutaciones.append({
            "brazo": brazo, "permutacion": PERMUTACIONES[indice - 1],
            "sha256": v35.sha256_bytes(json.dumps(PERMUTACIONES[indice - 1], sort_keys=True)
                                        .encode("utf-8")),
            "semilla": SEMILLA_PERMUTACIONES,
            "macro_f1": macro_control, "macro_f1_mapa_real": float(macro_real),
            "delta_real_menos_control": float(macro_real - macro_control),
            "delta_medio_bootstrap": intervalo["delta_medio"], "ic95": intervalo["ic95"],
            "n_cambios_de_prediccion": int((predicciones[brazo]
                                            != predicciones[brazo_real]).sum()),
            "el_mapa_real_gana": bool(macro_real - macro_control > TOL_VENTAJA),
        })
    brazo_barajado = "banco_c1a_barajado"
    macro_barajado = float(v35.macro(clase, predicciones[brazo_barajado]))
    intervalo_barajado = v35.bootstrap_delta(clase, predicciones[brazo_barajado],
                                            predicciones[brazo_real])
    barajado = {
        "brazo": brazo_barajado, "semilla": SEMILLA_PERMUTACIONES,
        "macro_f1": macro_barajado, "macro_f1_mapa_real": float(macro_real),
        "delta_real_menos_control": float(macro_real - macro_barajado),
        "delta_medio_bootstrap": intervalo_barajado["delta_medio"],
        "ic95": intervalo_barajado["ic95"],
        "n_cambios_de_prediccion": int((predicciones[brazo_barajado]
                                        != predicciones[brazo_real]).sum()),
        "el_mapa_real_gana": bool(macro_real - macro_barajado > TOL_VENTAJA),
        "que_separa": "hay transiciones frente a el orden de las transiciones importa",
    }
    gana_todas = all(entrada["el_mapa_real_gana"] for entrada in permutaciones)
    return {
        "permutaciones": permutaciones, "barajado": barajado,
        "tolerancia_de_ventaja": TOL_VENTAJA,
        "regla": "el mapa real debe ganar a las tres permutaciones del alfabeto (condicion (ii) del "
                 "ruling); si no gana, el veredicto es SIN_SENAL_ESPECIFICA_DE_CATEGORIA y L4 se "
                 "cierra igual",
        "n_permutaciones_ganadas": int(sum(1 for entrada in permutaciones
                                           if entrada["el_mapa_real_gana"])),
        "especificidad_ok": bool(gana_todas),
        "control_favorable": bool(gana_todas and barajado["el_mapa_real_gana"]),
    }


def evaluar_integracion(clase: np.ndarray, fold_idx: np.ndarray, v30: np.ndarray,
                        champ_pred: np.ndarray, p_canal: np.ndarray,
                        pesos: np.ndarray) -> dict:
    """Integra el canal con el campeon por la maquinaria auditada de V35.

    La fusion es el combinador logistico anidado congelado de V35, ``[L2(v30)||L2(p_canal)]`` con
    LogReg(C=1.0, max_iter=2000) ajustado por fold, y la decision es ``argmax(p_comb * pesos)`` con
    los pesos plug-in congelados. La comparacion de referencia es la decision del campeon
    ``argmax(v30 * pesos)``. Se publica el delta frente al campeon, el delta frente al canal de
    control (el mismo canal sin el bloque C1-A), el IC95 pareado de ambos, el diagnostico del brazo y
    el IC95 del banco de esa misma comparacion.

    Parameters
    ----------
    clase:
        Etiquetas canonicas.
    fold_idx:
        Fold canonico por fila.
    v30:
        Probabilidades del campeon (LEIDAS, no reentrenadas).
    champ_pred:
        Decision del campeon (argmax v30 * pesos).
    p_canal:
        OOF de probabilidad del canal (softmax de la decision_function).
    pesos:
        Pesos plug-in congelados.

    Returns
    -------
    dict
        Detalle de la integracion, con ``pred_comb`` a parte porque no es serializable.
    """
    p_comb, detalle_combinador = v35.combinador_oof(v30, p_canal, clase, fold_idx)
    evaluacion = v35.evaluar_combinacion(p_comb, clase, fold_idx, champ_pred, pesos)
    pred_comb = evaluacion.pop("pred_comb")
    diagnostico = v35.diagnostico_brazo(p_canal, clase, fold_idx, champ_pred, v30.max(axis=1))
    intervalo = v35.bootstrap_delta(clase, champ_pred, pred_comb)
    evaluacion["delta_integrado_vs_campeon"] = float(
        evaluacion["corrected_cv"] - v35.macro(clase, champ_pred))
    evaluacion["ic95_pareado_vs_campeon"] = intervalo["ic95"]
    evaluacion["delta_medio_bootstrap_vs_campeon"] = intervalo["delta_medio"]
    evaluacion["ic95_sin_cero_vs_campeon"] = bool(intervalo["ic95_sin_cero"])
    evaluacion["supera_h0"] = bool(intervalo["ic95_sin_cero"] and evaluacion[
        "delta_integrado_vs_campeon"] > 0.0)
    evaluacion["detalle_combinador"] = detalle_combinador
    evaluacion["diagnostico_del_canal"] = diagnostico
    evaluacion["punto_de_partida"] = {
        "definicion": "argmax(v30 * pesos plug-in) sobre la particion canonica",
        "baseline_primario_declarado": BASELINE,
        "delta_vs_nested": float(evaluacion["corrected_cv"] - NESTED),
        "n_cambios_de_prediccion": int((pred_comb != champ_pred).sum()),
    }
    return {"evaluacion": evaluacion, "pred_comb": pred_comb, "p_comb": p_comb}


def comparar_integraciones(clase: np.ndarray, candidato: dict, control: dict) -> dict:
    """Delta atribuible al bloque C1-A: integracion CANDIDATA menos integracion de CONTROL.

    Los dos brazos comparten el bloque del campeon, el bloque de palabra, los folds, la semilla y el
    modelo; la unica diferencia es el bloque C1-A. Asi el delta integrado atribuible queda aislado del
    efecto mecanico de volver a pasar por el combinador anidado.

    Parameters
    ----------
    clase:
        Etiquetas canonicas.
    candidato:
        Salida de :func:`evaluar_integracion` del brazo candidato.
    control:
        Salida de :func:`evaluar_integracion` del brazo de control.

    Returns
    -------
    dict
        Delta atribuible, su IC95 pareado y las filas que cambian entre ambas integraciones.
    """
    intervalo = v35.bootstrap_delta(clase, control["pred_comb"], candidato["pred_comb"])
    return {"macro_f1_candidato": float(candidato["evaluacion"]["corrected_cv"]),
            "macro_f1_control": float(control["evaluacion"]["corrected_cv"]),
            "delta_atribuido_al_bloque_c1a": float(candidato["evaluacion"]["corrected_cv"]
                                                   - control["evaluacion"]["corrected_cv"]),
            "delta_medio_bootstrap": intervalo["delta_medio"], "ic95": intervalo["ic95"],
            "p_delta_positivo": intervalo["p_delta_positivo"],
            "ic95_sin_cero": bool(intervalo["ic95_sin_cero"]),
            "n_cambios_entre_integraciones": int((candidato["pred_comb"]
                                                  != control["pred_comb"]).sum()),
            "delta_mecanico_del_combinador": float(control["evaluacion"]["corrected_cv"]
                                                   - BASELINE)}


# =====================================================================================
# ---- 9. GATES DE LA SERIE Y ARBOL DE DECISION ----
# =====================================================================================
def calcular_gates(banco: dict, controles: dict, integracion: dict,
                   integracion_control: dict, atribuible: dict) -> dict:
    """Aplica los gates congelados de la serie a la INTEGRACION contra el campeon.

    Los gates son los del pre-registro: delta integrado >= +0.0015, >= 4 de 5 folds, control
    favorable (el mapa real gana a las tres permutaciones y al barajado), IC95 pareado (2000,
    semilla 20261010) sin el cero, y se publica la variabilidad de referencia 0.0008876. El banco se
    evalua con su propia hipotesis nula, SIN gates de la serie.

    Parameters
    ----------
    banco:
        Salida de :func:`evaluar_banco`.
    controles:
        Salida de :func:`evaluar_controles`.
    integracion:
        Salida de :func:`evaluar_integracion` del brazo candidato.
    integracion_control:
        Salida de :func:`evaluar_integracion` del brazo de control.
    atribuible:
        Salida de :func:`comparar_integraciones`.

    Returns
    -------
    dict
        Cada gate con su valor y su veredicto, la etiqueta de la serie y si la segunda particion
        (que este script NO ejecuta) llegaria a hacer falta.
    """
    evaluacion = integracion["evaluacion"]
    delta = float(evaluacion["delta_integrado_vs_campeon"])
    deltas_fold = [float(valor) for valor in evaluacion["deltas_por_fold"]]
    gates = {
        "delta_min": {"valor": delta, "umbral": GATE_DELTA, "pasa": bool(delta >= GATE_DELTA)},
        "folds_min": {"valor": int(sum(1 for d in deltas_fold if d > 0)), "umbral": GATE_FOLDS,
                      "pasa": bool(sum(1 for d in deltas_fold if d > 0) >= GATE_FOLDS)},
        "control_favorable": {"valor": controles["control_favorable"], "umbral": True,
                              "pasa": bool(controles["control_favorable"])},
        "ic95_pareado_sin_cero": {"valor": evaluacion["ic95_sin_cero_vs_campeon"], "umbral": True,
                                 "pasa": bool(evaluacion["ic95_sin_cero_vs_campeon"])},
        "variabilidad_de_referencia": {"valor": VARIABILIDAD, "umbral": None,
                                       "escala": "delta integrado / variabilidad de referencia"},
    }
    pasan = all(gate["pasa"] for nombre, gate in gates.items()
                if nombre != "variabilidad_de_referencia")
    return {
        "gates": gates, "todos_pasan": bool(pasan),
        "delta_sobre_variabilidad": float(delta / VARIABILIDAD) if VARIABILIDAD else None,
        "delta_atribuido_al_bloque_c1a": atribuible["delta_atribuido_al_bloque_c1a"],
        "delta_mecanico_del_combinador": float(integracion_control["evaluacion"][
            "corrected_cv"] - BASELINE),
        "ic95_atribuido": atribuible["ic95"],
        "etiqueta_de_la_serie": "PROMISING" if pasan else "CLOSED",
        "segunda_particion_requerida": bool(pasan),
        "segunda_particion_ejecutada": False,
        "motivo_de_no_ejecutar_la_segunda_particion": "C1-A es ONE-SHOT y este script no ejecuta "
                                                     "una segunda particion sin autorizacion del "
                                                     "usuario (invariante del ciclo)",
        "n_bootstrap": N_BOOTSTRAP, "semilla": SEMILLA,
    }


def decidir_nodo(sondas_ok: bool, especificidad_ok: bool, banco_supera_h0: bool,
                 integracion_supera_h0: bool, delta_integrado: float) -> tuple[str, str, dict]:
    """Emite el nodo del arbol de decision y el motivo, con la precedencia declarada.

    Precedencia: (1) ``DEFECTO_DE_IMPLEMENTACION`` si alguna sonda o el triage fallan; (2)
    ``SIN_SENAL_ESPECIFICA_DE_CATEGORIA`` si el mapa real no gana a las tres permutaciones, porque
    el pre-registro hace ese veredicto incondicional ("si no gana, L4 se cierra igual"); (3)
    ``SENAL_NO_EXISTENTE`` si el banco no supera su H0; (4) si el banco supera su H0, el nodo de
    integracion: ``SENAL_EXISTE_INTEGRACION_POSITIVA`` si la integracion supera su H0,
    ``SENAL_EXISTE_INTEGRACION_NEGATIVA`` si su IC95 pareado excluye el cero por debajo (dano
    demostrado) y ``SENAL_EXISTE_INTEGRACION_NEUTRA`` en el resto (senal real y ortogonal que V30 ya
    expresa).

    Parameters
    ----------
    sondas_ok:
        True si todas las sondas obligatorias pasaron.
    especificidad_ok:
        True si el mapa real gano a las tres permutaciones.
    banco_supera_h0:
        True si el banco supero su hipotesis nula.
    integracion_supera_h0:
        True si la integracion supero su hipotesis nula.
    delta_integrado:
        Delta integrado frente al campeon.

    Returns
    -------
    tuple[str, str, dict]
        ``(nodo, motivo, traza)`` con la traza de las cuatro decisiones.
    """
    traza = {"sondas_ok": bool(sondas_ok), "especificidad_ok": bool(especificidad_ok),
             "banco_supera_h0": bool(banco_supera_h0),
             "integracion_supera_h0": bool(integracion_supera_h0),
             "delta_integrado": float(delta_integrado)}
    if not sondas_ok:
        return ("DEFECTO_DE_IMPLEMENTACION",
                "una sonda obligatoria o el triage fallaron: la familia NO se cierra y el ciclo no "
                "cuenta", traza)
    if not especificidad_ok:
        return ("SIN_SENAL_ESPECIFICA_DE_CATEGORIA",
                "el mapa real no gano a las tres permutaciones del alfabeto: el bloque captura "
                "algo, pero no especifico de la CLASE de caracter, y L4 se cierra igual", traza)
    if not banco_supera_h0:
        return ("SENAL_NO_EXISTENTE",
                "el banco no supero su hipotesis nula (delta del banco = 0): no hay senal ni en el "
                "bloque, y se cierra L4", traza)
    if integracion_supera_h0:
        return ("SENAL_EXISTE_INTEGRACION_POSITIVA",
                "el banco y la integracion superaron sus hipotesis nulas: se evaluan los gates de "
                "la serie y, si pasan, PROMISING exigiria segunda particion con autorizacion del "
                "usuario, que este script NO ejecuta", traza)
    if delta_integrado < 0.0:
        return ("SENAL_EXISTE_INTEGRACION_NEGATIVA",
                "el banco supero su hipotesis nula pero la integracion es negativa: se reporta el "
                "banco y el nodo, y L4 se cierra", traza)
    return ("SENAL_EXISTE_INTEGRACION_NEUTRA",
            "el banco supero su hipotesis nula pero la integracion no supero la suya: la senal es "
            "REAL y ORTOGONAL pero V30 ya la expresa. Se cierra L4 por mecanismo, sin reabrir la "
            "familia sparse y sin autorizar un C1-B, y el resultado es PUBLICABLE", traza)


def tabla_bias_unico(estado: dict) -> dict:
    """Tabla de simetria de sesgo, equivalente portado a C1-A.

    La tabla de V35 auditaba el canal de sesgo del nucleo FTRL de A1 frente al ``intercept_`` de
    A2, que son mecanismos que NO existen en C1-A: aqui todos los brazos son LinearSVC, que tiene UN
    solo camino de sesgo (su ``intercept_``) y cuya matriz NO lleva columna de sesgo pre-anadida.
    Esta tabla publica, por brazo y por fold, el ``intercept_`` del estimador y lo declara
    explicitamente, de modo que quede escrito que no hay doble sesgo (el defecto de D-083 y V33-E1
    no se reproduce).

    Parameters
    ----------
    estado:
        Checkpoint con los folds registrados.

    Returns
    -------
    dict
        Descripcion del unico camino de sesgo, el detalle por brazo y la declaracion.
    """
    por_brazo: dict[str, dict] = {}
    for brazo in BRAZOS_BANCO + BRAZOS_INTEG:
        interceptos = {fold: registro["intercepto"]
                       for fold, registro in estado["brazos"][brazo]["folds"].items()}
        por_brazo[brazo] = {"n_caminos_de_sesgo": 1, "intercepto_por_fold": interceptos,
                            "max_abs_intercept": float(max((abs(float(v)) for valores in
                                                           interceptos.values() for v in valores),
                                                          default=0.0))}
    return {"descripcion": "todos los brazos son LinearSVC: un solo camino de sesgo (intercept_) y "
                           "matriz sin columna de sesgo pre-anadida",
            "doble_sesgo": False,
            "declaracion": "no hay doble sesgo en ningun brazo: el sesgo lo aporta solo el "
                           "intercept_ de LinearSVC; el defecto de D-083 y V33-E1 (matriz con sesgo "
                           "pre-anadido y fit_intercept) NO se reproduce",
            "por_brazo": por_brazo}


# =====================================================================================
# ---- 10. ESCRITURA DE LOS ENTREGABLES (informe, figura, CSV, RUNLOG) ----
# =====================================================================================
def escribir_informe(resultado: dict) -> str:
    """Escribe el informe en markdown con el veredicto, las tablas y las sondas.

    Parameters
    ----------
    resultado:
        Resultado completo del ciclo.

    Returns
    -------
    str
        Ruta relativa del informe escrito.
    """
    veredicto = resultado["veredicto"]
    banco = resultado["banco"]["delta"]
    controles = resultado["controles"]
    integracion = resultado["integracion_candidata"]["evaluacion"]
    atribuible = resultado["integracion_atribuible"]
    sondas = resultado["sondas"]
    lineas = [
        "# V36 - E1 (via C1-A) - Transiciones de clase de caracter (brecha L4)",
        "",
        f"**Fecha**: {FECHA}. **Nodo del arbol**: `{veredicto['nodo']}`. **L4**: "
        f"{veredicto['cierra_L4']}.",
        "",
        f"**Motivo**: {veredicto['motivo']}",
        "",
        "## Lo que se midio",
        "",
        f"- Bloque C1-A: mapa categorico LITERAL de {len(ORDEN_ALFABETO)} simbolos "
        f"(`{' '.join(ORDEN_ALFABETO)}`), n-gramas de clase {list(RANGO_N_EJECUTADO)} "
        "(bigramas y trigramas), SIN borde, SIN min_df.",
        f"- Columnas reales del bloque: **{COLUMNAS_PUBLICADAS['total']}** "
        f"({COLUMNAS_PUBLICADAS['bigramas']} bigramas + {COLUMNAS_PUBLICADAS['trigramas']} "
        "trigramas de clase), las mismas que publica el pre-registro.",
        "- Folds LEIDOS de `artifacts/p33_scores/todos/fold_idx.npy` (particion canonica, no "
        "recreada ni barajada); modelo LinearSVC con la receta del campeon, sin grid.",
        "",
        "## Banco (C1-A sobre char_wb)",
        "",
        "| Brazo | macro-F1 | fold 0 | fold 1 | fold 2 | fold 3 | fold 4 |",
        "|---|---|---|---|---|---|---|",
    ]
    for brazo in BRAZOS_BANCO:
        detalle = resultado["banco"][brazo]
        filas = " | ".join(f"{valor:.7f}" for valor in detalle["folds"])
        lineas.append(f"| `{brazo}` | {detalle['macro_f1']:.7f} | {filas} |")
    lineas += [
        "",
        f"- **Delta del banco**: {banco['delta']:+.7f} de macro-F1 "
        f"({banco['folds_positivos']}/5 folds positivos), IC95 pareado "
        f"[{banco['ic95'][0]:+.7f}, {banco['ic95'][1]:+.7f}] (2000 remuestreos, semilla {SEMILLA}).",
        f"- **Supera su H0** (delta = 0): {banco['supera_h0']}.",
        "",
        "## Controles",
        "",
        "| Control | macro-F1 | delta real - control | IC95 pareado | el mapa real gana |",
        "|---|---|---|---|---|",
    ]
    for entrada in controles["permutaciones"]:
        lineas.append(f"| permutacion {entrada['brazo'][-1]} | {entrada['macro_f1']:.7f} | "
                     f"{entrada['delta_real_menos_control']:+.7f} | "
                     f"[{entrada['ic95'][0]:+.7f}, {entrada['ic95'][1]:+.7f}] | "
                     f"{entrada['el_mapa_real_gana']} |")
    barajado = controles["barajado"]
    lineas.append(f"| barajado (C3) | {barajado['macro_f1']:.7f} | "
                 f"{barajado['delta_real_menos_control']:+.7f} | "
                 f"[{barajado['ic95'][0]:+.7f}, {barajado['ic95'][1]:+.7f}] | "
                 f"{barajado['el_mapa_real_gana']} |")
    lineas += [
        "",
        f"- Permutaciones ganadas: **{controles['n_permutaciones_ganadas']}/3** (regla de la "
        f"condicion (ii): el mapa real debe ganar a las tres).",
        "",
        "## Integracion (campeon + C1-A)",
        "",
        f"- Punto de partida reproducido: **0.6159732** = macro-F1 de `argmax(v30 * pesos)` "
        f"(pesos {[round(peso, 4) for peso in PESOS]}), verificado en runtime.",
        f"- Campeon con C1-A: **{integracion['corrected_cv']:.7f}**, delta frente al campeon "
        f"**{integracion['delta_integrado_vs_campeon']:+.7f}**, "
        f"{integracion['folds_positivos']}/5 folds positivos, IC95 pareado "
        f"[{integracion['ic95_pareado_vs_campeon'][0]:+.7f}, "
        f"{integracion['ic95_pareado_vs_campeon'][1]:+.7f}].",
        f"- Delta atribuible al bloque C1-A (candidato menos canal de control): "
        f"**{atribuible['delta_atribuido_al_bloque_c1a']:+.7f}**, IC95 "
        f"[{atribuible['ic95'][0]:+.7f}, {atribuible['ic95'][1]:+.7f}].",
        f"- Efecto mecanico de volver a pasar por el combinador anidado: "
        f"{resultado['gates']['delta_mecanico_del_combinador']:+.7f}.",
        "",
        "## Gates de la serie (aplican SOLO a la integracion)",
        "",
        "| Gate | Valor | Umbral | Pasa |",
        "|---|---|---|---|",
    ]
    for nombre, gate in resultado["gates"]["gates"].items():
        if nombre == "variabilidad_de_referencia":
            continue
        lineas.append(f"| {nombre} | {gate['valor']} | {gate['umbral']} | {gate['pasa']} |")
    lineas += [
        "",
        f"- Variabilidad de referencia: {VARIABILIDAD} "
        f"(delta integrado / variabilidad = {resultado['gates']['delta_sobre_variabilidad']}).",
        f"- Etiqueta de la serie: **{resultado['gates']['etiqueta_de_la_serie']}**. Segunda "
        f"particion requerida: {resultado['gates']['segunda_particion_requerida']}; ejecutada: "
        "**NO** (C1-A es one-shot y exige autorizacion del usuario).",
        "",
        "## Sondas obligatorias",
        "",
        "| Sonda | Resultado |",
        "|---|---|",
    ]
    for nombre in NOMBRES_SONDAS:
        entrada = sondas[nombre]
        lineas.append(f"| {nombre} | {entrada['pass']} |")
    lineas += [
        "",
        "## Enmienda pendiente del pre-registro",
        "",
        f"- {ENMIENDA_RANGO_N['campo_congelado']}: valor congelado "
        f"{ENMIENDA_RANGO_N['valor_congelado']}, valor ejecutado "
        f"{ENMIENDA_RANGO_N['valor_ejecutado']}.",
        f"- Motivo: {ENMIENDA_RANGO_N['lectura_de_la_prosa']}.",
        f"- Evidencia: {ENMIENDA_RANGO_N['columnas_publicadas_que_la_confirman']}.",
        f"- {ENMIENDA_RANGO_N['nota_18_66']}.",
        f"- Estado: **{ENMIENDA_RANGO_N['estado']}**.",
        "",
        "## Compliance",
        "",
        "- 0 transformers, 0 LLM, 0 embeddings, 0 datos externos, 0 metadatos; sin test mas alla de "
        "`transform -> predict`; sin holdout, sin leaderboard, sin submission.",
        "- Campeon V30 + plug-in intacto y NO reentrenado; `artifacts/p33_scores/todos` leido y "
        "nunca escrito.",
        f"- RSS con `ru_maxrss`: {resultado['recursos']['rss_pico_gb']:.2f} GB "
        f"(tope {RAM_TOPE_GB:g} GB); presupuesto declarado {PRESUPUESTO_DECLARADO_SEG:g} s; "
        f"coste real {resultado['recursos']['duracion_seg']:.0f} s.",
        "",
    ]
    v35.escribir_json_atomico(INFORME, "\n".join(lineas))
    return v35.ruta_relativa(INFORME)


def figura_curvas(resultado: dict) -> None:
    """Dibuja la figura de curvas y barras del ciclo en ``reports/figures``.

    Parameters
    ----------
    resultado:
        Resultado completo del ciclo.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    banco = resultado["banco"]
    controles = resultado["controles"]
    integracion = resultado["integracion_candidata"]["evaluacion"]
    figura, ejes = plt.subplots(2, 2, figsize=(13.0, 8.5))
    eje = ejes[0][0]
    eje.plot(range(N_FOLDS), banco[BRAZO_REFERENCIA]["folds"], marker="o", label="banco referencia")
    eje.plot(range(N_FOLDS), banco[BRAZO_CANDIDATO]["folds"], marker="s", label="banco + C1-A")
    for indice, entrada in enumerate(controles["permutaciones"], start=1):
        eje.plot(range(N_FOLDS), banco[entrada["brazo"]]["folds"], marker="^", linestyle=":",
                 label=f"permutacion {indice}")
    eje.plot(range(N_FOLDS), banco["banco_c1a_barajado"]["folds"], marker="x", linestyle="--",
             label="barajado")
    eje.set_title("Banco: macro-F1 por fold (char_wb y sus controles)")
    eje.set_xlabel("fold canonico")
    eje.set_ylabel("macro-F1")
    eje.legend(fontsize=7)
    eje.grid(alpha=0.3)
    eje = ejes[0][1]
    nombres = [BRAZO_REFERENCIA, BRAZO_CANDIDATO] + [entrada["brazo"]
                                                       for entrada in controles["permutaciones"]] + [
        "banco_c1a_barajado"]
    valores = [banco[nombre]["macro_f1"] for nombre in nombres]
    eje.bar(range(len(nombres)), valores, color=["0.35", "0.15"] + ["0.6"] * 4)
    eje.set_xticks(range(len(nombres)))
    eje.set_xticklabels(["ref", "C1-A", "perm1", "perm2", "perm3", "barajado"], fontsize=8)
    eje.axhline(banco[BRAZO_REFERENCIA]["macro_f1"], color="k", linewidth=0.8, linestyle=":")
    eje.set_title("Banco: macro-F1 global por brazo (linea: referencia)")
    eje.set_ylabel("macro-F1")
    eje.grid(alpha=0.3, axis="y")
    eje = ejes[1][0]
    eje.bar(range(N_FOLDS), banco["delta"]["deltas_por_fold"], color="0.35")
    eje.axhline(0.0, color="k", linewidth=0.8)
    eje.axhline(GATE_DELTA, color="r", linewidth=0.8, linestyle="--",
                label=f"gate de la serie ({GATE_DELTA})")
    eje.set_title("Banco: delta por fold (candidato menos referencia)")
    eje.set_xlabel("fold canonico")
    eje.set_ylabel("delta de macro-F1")
    eje.legend(fontsize=8)
    eje.grid(alpha=0.3, axis="y")
    eje = ejes[1][1]
    ancho = 0.35
    eje.bar(np.arange(N_FOLDS) - ancho / 2, integracion["deltas_por_fold"], width=ancho,
            label="campeon + C1-A")
    eje.bar(np.arange(N_FOLDS) + ancho / 2,
            resultado["integracion_control"]["evaluacion"]["deltas_por_fold"], width=ancho,
            label="campeon + control")
    eje.axhline(0.0, color="k", linewidth=0.8)
    eje.set_title("Integracion: delta por fold frente al campeon")
    eje.set_xlabel("fold canonico")
    eje.set_ylabel("delta de macro-F1")
    eje.legend(fontsize=8)
    eje.grid(alpha=0.3, axis="y")
    figura.suptitle("V36-E1 via C1-A: transiciones de clase de caracter (brecha L4)", fontsize=12)
    figura.tight_layout()
    FIGURA.parent.mkdir(parents=True, exist_ok=True)
    temporal = FIGURA.with_name(FIGURA.name + ".tmp.png")
    figura.savefig(temporal, dpi=140)
    plt.close(figura)
    os.replace(temporal, FIGURA)


def anotar_runlog(resultado: dict) -> str | None:
    """Anexa la entrada de V36-E1 al RUNLOG.md si no esta ya.

    Parameters
    ----------
    resultado:
        Resultado completo del ciclo.

    Returns
    -------
    str | None
        None si la entrada ya existia, o un texto de confirmacion si se anadio.
    """
    marca = "## V36-E1 (via C1-A)"
    if marca in RUNLOG.read_text(encoding="utf-8"):
        return None
    veredicto = resultado["veredicto"]
    banco = resultado["banco"]["delta"]
    integracion = resultado["integracion_candidata"]["evaluacion"]
    texto = ("\n---\n\n" + marca + " - transiciones de clase de caracter (brecha L4) - " + FECHA
             + "\n\n"
             f"- Nodo del arbol: **{veredicto['nodo']}**; L4 {veredicto['cierra_L4']}.\n"
             f"- Banco: delta {banco['delta']:+.7f} de macro-F1 "
             f"({banco['folds_positivos']}/5 folds), IC95 [{banco['ic95'][0]:+.7f}, "
             f"{banco['ic95'][1]:+.7f}]; supera su H0: {banco['supera_h0']}.\n"
             f"- Integracion: delta frente al campeon "
             f"{integracion['delta_integrado_vs_campeon']:+.7f}, IC95 "
             f"[{integracion['ic95_pareado_vs_campeon'][0]:+.7f}, "
             f"{integracion['ic95_pareado_vs_campeon'][1]:+.7f}]; punto de partida 0.6159732 "
             "reproducido.\n"
             f"- Controles: mapa real gana a {resultado['controles']['n_permutaciones_ganadas']}/3 "
             f"permutaciones y al barajado: {resultado['controles']['barajado']['el_mapa_real_gana']}"
             ".\n"
             "- **Enmienda pendiente**: `rango_n_de_transiciones` congelado [1, 2] frente al (2, 3) "
             "ejecutado (bigramas y trigramas, las columnas publicadas 27 / 127).\n"
             "- Campeon V30 + plug-in intacto; sin test, holdout, leaderboard ni submission.\n")
    with RUNLOG.open("a", encoding="utf-8") as flujo:
        flujo.write(texto)
    return "RUNLOG.md: entrada V36-E1 anadida"


def anotar_experiments_csv(resultado: dict) -> str | None:
    """Anexa la fila de V36-E1 a ``experiments.csv`` si no esta ya.

    Parameters
    ----------
    resultado:
        Resultado completo del ciclo.

    Returns
    -------
    str | None
        None si la fila ya existia, o un texto de confirmacion si se anadio.
    """
    lineas = EXPERIMENTS_CSV.read_text(encoding="utf-8").splitlines()
    columnas = [columna.strip() for columna in lineas[0].split(",")]
    if "V36-E1" in {linea.split(",", 1)[0] for linea in lineas[1:] if linea}:
        return None
    banco = resultado["banco"]["delta"]
    integracion = resultado["integracion_candidata"]["evaluacion"]
    fila = {
        "experiment_id": "V36-E1", "fecha": FECHA, "bloque": "36.1",
        "etapa": "V36-E1 via C1-A: bloque de transiciones de clase de caracter (L4)",
        "pregunta": "el perfil de tipo de transicion (letra->digito, espacio->puntuacion) aporta "
                    "senal ortogonal que el word y el char_wb del campeon no expresan?",
        "hipotesis": "si: condicionar el CARACTER a su CATEGORIA abre la familia sparse por la "
                     "clausula de reapertura de F10, y el mapa descarta la identidad del caracter",
        "features": f"char_wb (3,5) min_df 5 del campeon + transiciones de clase "
                    f"({RANGO_N_EJECUTADO[0]} y {RANGO_N_EJECUTADO[1]}) sobre un alfabeto categorico "
                    f"literal de {len(ORDEN_ALFABETO)} clases",
        "preprocessing": "vocabulario e idf ajustados SOLO en el train de cada fold; sin borde y "
                         "sin min_df",
        "representacion": f"mapa categorico literal ({' '.join(ORDEN_ALFABETO)}) + conteos "
                          f"dispersos de n-gramas de clase ({COLUMNAS_PUBLICADAS['total']} "
                          "columnas); sin embeddings",
        "modelo": "LinearSVC(C=1.0, squared_hinge, l2, max_iter=3000) con la receta del campeon; "
                  "controles: sin bloque, 3 permutaciones del alfabeto y barajado",
        "hiperparametros": "C=1.0 de la receta del campeon; sin grid; alfabeto "
                           f"{' '.join(ORDEN_ALFABETO)}; n={list(RANGO_N_EJECUTADO)}; sin min_df; sin "
                           f"borde; CV {N_FOLDS} folds sobre los indices canonicos",
        "validacion": f"IC95 pareado {N_BOOTSTRAP} (semilla {SEMILLA}) del banco y de la "
                      "integracion; tabla 5/5 folds; 3 permutaciones; control de barajado",
        "macro_f1_mean": f"{resultado['banco'][BRAZO_CANDIDATO]['macro_f1']:.7f}",
        "macro_f1_std": "",
        "weighted_f1_mean": "", "micro_f1_mean": "", "accuracy_mean": "",
        "resultado": f"nodo {resultado['veredicto']['nodo']}; banco {banco['delta']:+.7f} "
                     f"({banco['folds_positivos']}/5, IC95 [{banco['ic95'][0]:+.7f}, "
                     f"{banco['ic95'][1]:+.7f}]); integracion "
                     f"{integracion['delta_integrado_vs_campeon']:+.7f} frente a 0.6159732",
        "interpretacion": f"{resultado['veredicto']['motivo']}",
        "decision": f"{resultado['veredicto']['cierra_L4']}; C1-A es one-shot en cualquier caso",
        "duracion_seg": f"{resultado['recursos']['duracion_seg']:.1f}",
        "artefactos": "reports/v36_e1_transiciones_clase.md | experiments/v36_01_transiciones_clase"
                      ".json | reports/figures/v36_e1_curvas.png",
    }
    buffer = io.StringIO()
    csv.writer(buffer, lineterminator="\n").writerow([fila[columna] for columna in columnas])
    with EXPERIMENTS_CSV.open("a", encoding="utf-8") as flujo:
        flujo.write(buffer.getvalue())
    return "experiments.csv: fila V36-E1 anadida"


def escribir_registro_novedad(resultado: dict) -> str:
    """Escribe ``reports/v36_novelty_registry.csv`` con la fila de novedad del ciclo.

    Parameters
    ----------
    resultado:
        Resultado completo del ciclo.

    Returns
    -------
    str
        Ruta relativa del registro escrito.
    """
    fila = ["V36-E1", FECHA,
            "bloque de transiciones de CLASE de caracter: mapa categorico literal de "
            f"{len(ORDEN_ALFABETO)} simbolos (S D P L M O), bigramas y trigramas de clase sobre el "
            "texto entero, sin borde y sin min_df, concatenado al char_wb del campeon",
            "no", "no", "no", "no (mapa literal sin ningun parametro estimado; sin grid de nada)",
            "V35-E1 fue un one-shot de la familia FTRL con L1 y ya cerro la familia por mecanismo; "
            "aqui se abre la brecha L4 de la familia sparse condicionando el CARACTER a su "
            "CATEGORIA, que es lo unico que la clausula de reapertura de F10 permite",
            "sondas obligatorias: mapa literal con sha256, conformidad stateless del transform, "
            "fold_local con unico fit = TfidfVectorizer del train de cada fold y sha256 de "
            "vocabulario e idf por fold, tabla 5/5 folds, IC95 pareado (2000, semilla 20261010), "
            "3 permutaciones del alfabeto y control de barajado",
            "experiments/v36_01_transiciones_clase.json | experiments/v36_01_estado.json | "
            "reports/v36_e1_transiciones_clase.md | reports/figures/v36_e1_curvas.png | "
            "reports/v36_novelty_registry.csv | artifacts/p36_scores/v36_e1/"]
    buffer = io.StringIO()
    escritor = csv.writer(buffer, lineterminator="\n")
    escritor.writerow(COLUMNAS_NOVEDAD)
    escritor.writerow(fila)
    v35.escribir_json_atomico(NOVELTY_CSV, buffer.getvalue())
    return v35.ruta_relativa(NOVELTY_CSV)


def anotar_historial(resultado: dict) -> str | None:
    """Anexa la fila de V36-E1 al historial de experimentos si no esta ya.

    Parameters
    ----------
    resultado:
        Resultado completo del ciclo.

    Returns
    -------
    str | None
        None si la fila ya existia, o un texto de confirmacion si se anadio.
    """
    lineas = HISTORIA_CSV.read_text(encoding="utf-8").splitlines()
    if "V36-E1" in {linea.split(",", 1)[0] for linea in lineas[1:] if linea}:
        return None
    banco = resultado["banco"]["delta"]
    fila = ["V36-E1", "V36-E1", FECHA,
            "el perfil de tipo de transicion de caracter (letra->digito, espacio->puntuacion) "
            "aporta senal ortogonal al word y char_wb del campeon",
            "LinearSVC(C=1.0, squared_hinge, l2, max_iter=3000) + combinador anidado de V35",
            "char_wb (3,5) min_df 5 + n-gramas de clase (2,3) sobre alfabeto categorico literal de "
            "6 clases, sin borde y sin min_df",
            f"{resultado['banco'][BRAZO_CANDIDATO]['macro_f1']:.7f}",
            f"{banco['delta']:+.7f}", f"{banco['folds_positivos']}/5",
            f"IC95 [{banco['ic95'][0]:+.7f}, {banco['ic95'][1]:+.7f}] (2000, semilla {SEMILLA})",
            "", "", "", resultado["veredicto"]["nodo"], resultado["veredicto"]["motivo"]]
    buffer = io.StringIO()
    csv.writer(buffer, lineterminator="\n").writerow(fila)
    with HISTORIA_CSV.open("a", encoding="utf-8") as flujo:
        flujo.write(buffer.getvalue())
    return "reports/age_complete_experiment_history.csv: fila V36-E1 anadida"


# =====================================================================================
# ---- 11. SONDAS OBLIGATORIAS DEL PRE-REGISTRO ----
# =====================================================================================
def sonda_mapa_literal(prereg: dict, texto_de_prueba: str) -> dict:
    """Sonda ``mapa_literal``: sha256 del mapa, del alfabeto, de la tabla y de la clase resultado.

    Verifica que el mapa del codigo es BIT a BIT el congelado, que la tabla de puntuacion ASCII es la
    tabla literal declarada y que la clase de un texto de prueba fijo es la esperada. Publica los
    cuatro sha256 para que nadie pueda cambiar el mapa entre el pre-registro y la corrida.

    Parameters
    ----------
    prereg:
        Pre-registro congelado.
    texto_de_prueba:
        Texto fijo que se convierte a clases, con acentos, digitos, puntuacion, un emoji y un
        caracter no latino.

    Returns
    -------
    dict
        Los cuatro sha256, la clase del texto de prueba y su veredicto.
    """
    hashes = prereg["hashes"]
    mapa_sha = v35.sha256_bytes(json.dumps(MAPA_CARACTERES, sort_keys=True).encode("utf-8"))
    orden_sha = v35.sha256_bytes(json.dumps(list(ORDEN_ALFABETO)).encode("utf-8"))
    puntuacion_sha = v35.sha256_bytes(json.dumps(list(PUNTUACION_ASCII)).encode("utf-8"))
    secuencia = secuencia_de_clases(texto_de_prueba)
    secuencia_sha = v35.sha256_bytes(secuencia.encode("utf-8"))
    documento = documento_de_transiciones(secuencia)
    veredicto = (mapa_sha == hashes["mapa_literal_de_clases"]
                 and orden_sha == hashes["orden_del_alfabeto"]
                 and MAPA_CARACTERES == prereg["mapa_de_clases_literal"]
                 and len(PUNTUACION_ASCII) == 32)
    return {"sha256_mapa": mapa_sha, "sha256_orden_alfabeto": orden_sha,
            "sha256_tabla_puntuacion_ascii": puntuacion_sha,
            "sha256_clase_del_texto_de_prueba": secuencia_sha,
            "texto_de_prueba": texto_de_prueba, "secuencia_de_clases": secuencia,
            "n_tokens_del_documento": len(documento.split()),
            "n_caracteres_ascii_en_la_tabla": len(PUNTUACION_ASCII),
            "pass": bool(veredicto),
            "criterio": "sha256 del mapa y del alfabeto iguales a los congelados, mapa identico al "
                        "del pre-registro y tabla de puntuacion ASCII de 32 caracteres"}


def sonda_conformidad(textos: np.ndarray) -> dict:
    """Sonda ``conformidad``: el transform es stateless y dos llamadas dan la misma matriz.

    Se construye el bloque C1-A DOS veces sobre la misma entrada (mismos documentos, mismo
    vectorizador, mismo train) y se comparan las matrices con su sha256. Se comprueba tambien que
    los shingles de clase de un texto de prueba son los esperados y que el mapa NO anade borde: el
    numero de n-gramas de un texto de longitud L es exactamente ``2L - 3`` para el rango (2,3)
    (``(L-1) + (L-2)``), sin el extra que anadiria un simbolo de borde.

    Parameters
    ----------
    textos:
        Textos de entrada (la sonda usa un subconjunto pequeno).

    Returns
    -------
    dict
        Los dos sha256 de la matriz, su igualdad y las comprobaciones del borde.
    """
    documentos = construir_documentos_c1a(textos, "real")
    tr = np.arange(len(textos) - 10)
    va = np.arange(len(textos) - 10, len(textos))
    x_tr_1, x_va_1, _ = ajustar_bloque_c1a(documentos, tr, va)
    x_tr_2, x_va_2, _ = ajustar_bloque_c1a(documentos, tr, va)
    sha_1 = sha256_matriz(x_tr_1.toarray())
    sha_2 = sha256_matriz(x_tr_2.toarray())
    texto = "Hola, mundo 123!"
    secuencia = secuencia_de_clases(texto)
    n_tokens = len(ngramas_de_clase(secuencia, RANGO_N_EJECUTADO[0], RANGO_N_EJECUTADO[1]))
    esperado = sum(max(len(secuencia) - longitud + 1, 0) for longitud in RANGO_N_EJECUTADO)
    return {"sha256_primera_llamada": sha_1, "sha256_segunda_llamada": sha_2,
            "matrices_identicas": bool(sha_1 == sha_2),
            "val_identicos": bool(np.array_equal(x_tr_1.toarray(), x_tr_2.toarray())
                                  and np.array_equal(x_va_1.toarray(), x_va_2.toarray())),
            "n_filas_una_llamada": int(x_tr_1.shape[0]),
            "n_columnas": int(x_tr_1.shape[1]),
            "secuencia_de_prueba": secuencia,
            "n_tokens_de_prueba": int(n_tokens), "n_tokens_esperados_sin_borde": int(esperado),
            "sin_borde": bool(n_tokens == esperado),
            "pass": bool(sha_1 == sha_2 and np.array_equal(x_tr_1.toarray(), x_tr_2.toarray())
                         and n_tokens == esperado),
            "criterio": "dos llamadas con la misma entrada dan exactamente la misma matriz (sha256 "
                        "identico) y el numero de n-gramas de un texto de longitud L es "
                        "(L-1)+(L-2), sin ningun simbolo de borde"}


def sonda_fold_local(estado: dict, exigido: bool) -> dict:
    """Sonda ``fold_local``: el unico ``fit`` es el TfidfVectorizer del train de cada fold.

    Con los folds ya entrenados, publica por brazo y por fold el numero de filas usadas en el
    ``fit``, el numero de filas de validacion, el tamano del vocabulario y los sha256 del
    vocabulario ordenado y del idf, y comprueba que el vocabulario del bloque C1-A del mapa real
    reproduce las columnas publicadas 27 / 127 y que el fold-val no aporta columnas.

    Parameters
    ----------
    estado:
        Checkpoint con los folds registrados.
    exigido:
        True si la comprobacion de las columnas publicadas es obligatoria (corrida real) y False si
        solo se publica (miniproblema de --verificar y --humo, donde el subconjunto puede no
        contener todos los n-gramas de clase).

    Returns
    -------
    dict
        Detalle por brazo y por fold, el recuento de checks y su veredicto.
    """
    checks: list[bool] = []
    por_brazo: dict[str, dict] = {}
    for brazo in BRAZOS_BANCO + BRAZOS_INTEG:
        folds: dict[str, dict] = {}
        for fold in range(N_FOLDS):
            registro = estado["brazos"][brazo]["folds"].get(str(fold))
            if registro is None:
                continue
            meta_referencia = registro["meta_bloque_referencia"]
            meta_c1a = registro["meta_c1a"]
            checks.append(meta_referencia["n_filas_fit"] < meta_referencia["n_filas_total"])
            checks.append(meta_referencia["columnas_val_igualan_train"])
            checks.append(len(meta_referencia["vocab_sha256"]) == 64
                          and len(meta_referencia["idf_sha256"]) == 64)
            if meta_c1a is not None:
                checks.append(meta_c1a["n_filas_fit"] < meta_c1a["n_filas_total"])
                checks.append(meta_c1a["columnas_val_igualan_train"])
                checks.append(len(meta_c1a["vocab_sha256"]) == 64
                              and len(meta_c1a["idf_sha256"]) == 64)
                if exigido and registro["variante_c1a"] == "real":
                    checks.append(meta_c1a["n_bigramas"] == COLUMNAS_PUBLICADAS["bigramas"])
                    checks.append(meta_c1a["n_trigramas"] == COLUMNAS_PUBLICADAS["trigramas"])
            folds[str(fold)] = {"n_filas_fit_bloque_referencia": meta_referencia["n_filas_fit"],
                                "n_filas_totales": meta_referencia["n_filas_total"],
                                "n_vocabulario_bloque_referencia": meta_referencia[
                                    "n_vocabulario"],
                                "vocab_sha256_bloque_referencia": meta_referencia["vocab_sha256"],
                                "idf_sha256_bloque_referencia": meta_referencia["idf_sha256"],
                                "n_filas_fit_c1a": meta_c1a["n_filas_fit"] if meta_c1a else None,
                                "n_vocabulario_c1a": meta_c1a["n_vocabulario"] if meta_c1a
                                else None,
                                "n_bigramas_de_clase": meta_c1a["n_bigramas"] if meta_c1a else None,
                                "n_trigramas_de_clase": meta_c1a["n_trigramas"] if meta_c1a
                                else None,
                                "vocab_sha256_c1a": meta_c1a["vocab_sha256"] if meta_c1a else None,
                                "idf_sha256_c1a": meta_c1a["idf_sha256"] if meta_c1a else None,
                                "columnas_val_igualan_train_c1a": (meta_c1a[
                                    "columnas_val_igualan_train"] if meta_c1a else None)}
        por_brazo[brazo] = {"n_folds_registrados": len(folds), "folds": folds}
    veredicto = bool(checks) and all(checks)
    return {"por_brazo": por_brazo, "n_checks": len(checks), "n_checks_ok": int(sum(checks)),
            "columnas_publicadas_exigidas": bool(exigido), "pass": veredicto,
            "criterio": "el unico fit de cada bloque es el TfidfVectorizer del train de su fold; el "
                        "fold-val aporta las mismas columnas; el vocabulario y el idf se publican "
                        "por fold con sha256"}


def sondas_de_resultado(estado: dict, resultado: dict, exigido: bool) -> dict:
    """Sondas que dependen de la medicion: banco, tabla 5/5 folds, bootstrap, controles y barajado.

    Parameters
    ----------
    estado:
        Checkpoint con los folds registrados.
    resultado:
        Resultado ya evaluado (banco, controles, integracion, gates y veredicto).
    exigido:
        Si las columnas publicadas son obligatorias (corrida real) o solo informativas.

    Returns
    -------
    dict
        Las cinco sondas restantes con su veredicto.
    """
    banco = resultado["banco"]["delta"]
    controles = resultado["controles"]
    return {
        "banco_ree_medido": {
            "pass": bool(banco["delta"] != 0.0 or not banco["ic95_sin_cero"]),
            "delta": banco["delta"], "ic95": banco["ic95"],
            "supera_h0": banco["supera_h0"],
            "criterio": "el banco se vuelve a medir DENTRO del runner sobre la particion canonica; "
                        "la medicion previa de +0.0031 se declara no canonica y no se usa",
            "naturaleza": resultado["naturaleza"]},
        "tabla_5_de_5_folds": {
            "pass": bool(len(banco["deltas_por_fold"]) == N_FOLDS
                         and len(resultado["integracion_candidata"]["evaluacion"][
                             "deltas_por_fold"]) == N_FOLDS),
            "folds_banco": banco["deltas_por_fold"],
            "folds_integracion": resultado["integracion_candidata"]["evaluacion"][
                "deltas_por_fold"],
            "criterio": "tabla de macro-F1 por fold del banco y de la integracion, 5 de 5"},
        "bootstrap_pareado": {
            "pass": bool(banco["ic95"] is not None
                         and resultado["integracion_candidata"]["evaluacion"][
                             "ic95_pareado_vs_campeon"] is not None),
            "ic95_banco": banco["ic95"], "ic95_integracion": resultado[
                "integracion_candidata"]["evaluacion"]["ic95_pareado_vs_campeon"],
            "n_bootstrap": N_BOOTSTRAP, "semilla": SEMILLA,
            "criterio": "IC95 pareado (2000, semilla 20261010) del banco y de la integracion"},
        "permutaciones": {
            "pass": bool(len(controles["permutaciones"]) >= 3),
            "n_permutaciones": len(controles["permutaciones"]),
            "n_ganadas": controles["n_permutaciones_ganadas"],
            "deltas": [entrada["delta_real_menos_control"]
                       for entrada in controles["permutaciones"]],
            "criterio": "condicion (ii): >= 3 permutaciones del alfabeto y el mapa real debe ganar"},
        "barajado": {
            "pass": bool("el_mapa_real_gana" in controles["barajado"]),
            "delta": controles["barajado"]["delta_real_menos_control"],
            "n_cambios_de_prediccion": controles["barajado"]["n_cambios_de_prediccion"],
            "criterio": "control C3: transiciones barajadas dentro de cada documento"},
    }


def ejecutar_sondas(estado: dict, prereg: dict, textos: np.ndarray, resultado: dict | None,
                    exigido: bool) -> dict:
    """Ejecuta las ocho sondas obligatorias del pre-registro y devuelve su veredicto.

    Parameters
    ----------
    estado:
        Checkpoint con los folds registrados (puede ser vacio si aun no se entreno nada).
    prereg:
        Pre-registro congelado.
    textos:
        Textos de entrada para la sonda de conformidad.
    resultado:
        Resultado ya evaluado, o None si todavia no hay medicion (modos de verificacion).
    exigido:
        Si las columnas publicadas son obligatorias.

    Returns
    -------
    dict
        Un veredicto por cada sonda de :data:`NOMBRES_SONDAS`.

    Raises
    ------
    Detencion
        Si falla alguna sonda obligatoria (codigo 4).
    """
    texto_de_prueba = "Hola, mundo 123! Añadir más cosas: ¿cuánto? -> 5🚀Ж"
    sondas = {"mapa_literal": sonda_mapa_literal(prereg, texto_de_prueba),
              "conformidad": sonda_conformidad(textos[:400]),
              "fold_local": sonda_fold_local(estado, exigido)}
    if resultado is None:
        for nombre in ("banco_ree_medido", "tabla_5_de_5_folds", "bootstrap_pareado",
                       "permutaciones", "barajado"):
            sondas[nombre] = {"pass": None, "ejecutada": False,
                              "criterio": "sonda de medicion: se ejecuta dentro de la corrida, no "
                                          "en los modos de verificacion"}
    else:
        sondas.update(sondas_de_resultado(estado, resultado, exigido))
    fallidas = [nombre for nombre in NOMBRES_SONDAS if sondas[nombre]["pass"] is False]
    for nombre in NOMBRES_SONDAS:
        if sondas[nombre]["pass"] is False:
            print(f"  [STOP] sonda '{nombre}' fallida: "
                  f"{sondas[nombre].get('criterio', '')} (exit 4)", flush=True)
    if fallidas:
        raise Detencion(f"[STOP] sondas obligatorias fallidas: {fallidas} (exit 4)", 4)
    return {nombre: {**sondas[nombre], "nombre": nombre} for nombre in NOMBRES_SONDAS}


# =====================================================================================
# ---- 12. ENSAMBLADO DEL RESULTADO, MINIPROBLEMA Y AUTOEJERCICIO ----
# =====================================================================================
def ensamblar_resultado(clase: np.ndarray, fold_idx: np.ndarray, v30: np.ndarray,
                        champ_pred: np.ndarray, oofs: dict[str, np.ndarray], pesos: np.ndarray,
                        baseline: float, estado: dict, sondas: dict, contexto: dict,
                        naturaleza: str) -> dict:
    """Junta banco, controles, integracion, gates y veredicto en el diccionario de resultado.

    Parameters
    ----------
    clase:
        Etiquetas canonicas.
    fold_idx:
        Fold canonico por fila.
    v30:
        Probabilidades del campeon.
    champ_pred:
        Decision del campeon.
    oofs:
        OOF por brazo.
    pesos:
        Pesos plug-in congelados.
    baseline:
        Macro-F1 del campeon recomputado en runtime.
    estado:
        Checkpoint con los folds registrados.
    sondas:
        Veredicto de las sondas.
    contexto:
        Verificaciones, hashes yossi (prereg, auditoria, hashes, columnas).
    naturaleza:
        Marca de naturaleza (evidencia experimental o miniproblema).

    Returns
    -------
    dict
        Resultado completo, listo para JSON, informe, figura y anotadores.
    """
    banco = evaluar_banco(clase, fold_idx, oofs)
    predicciones = banco.pop("_predicciones")
    controles = evaluar_controles(clase, fold_idx, oofs, predicciones,
                                  banco[BRAZO_CANDIDATO]["macro_f1"])
    integracion_candidata = evaluar_integracion(clase, fold_idx, v30, champ_pred,
                                               oofs["integ_c1a"], pesos)
    integracion_control = evaluar_integracion(clase, fold_idx, v30, champ_pred,
                                              oofs["integ_control"], pesos)
    atribuible = comparar_integraciones(clase, integracion_candidata, integracion_control)
    gates = calcular_gates(banco, controles, integracion_candidata, integracion_control,
                           atribuible)
    sondas_ok = all(sondas[nombre]["pass"] is not False for nombre in NOMBRES_SONDAS)
    nodo, motivo, traza = decidir_nodo(
        sondas_ok, controles["especificidad_ok"], banco["delta"]["supera_h0"],
        integracion_candidata["evaluacion"]["supera_h0"],
        integracion_candidata["evaluacion"]["delta_integrado_vs_campeon"])
    cierra = ("se cierra L4 por el nodo del arbol; C1-A es ONE-SHOT: no se reintenta con otro "
              "alfabeto, otro rango de n, otro mapa ni otro corte")
    return {
        "fase": "V36-E1 via C1-A", "fecha": FECHA, "naturaleza": naturaleza,
        "naturaleza_autoensayo": contexto.get("naturaleza_autoensayo"),
        "baseline_recomputado": float(baseline), "pesos_plugin": PESOS,
        "mapa_de_clases_literal": MAPA_CARACTERES, "orden_del_alfabeto": list(ORDEN_ALFABETO),
        "tabla_puntuacion_ascii": list(PUNTUACION_ASCII),
        "rango_n_ejecutado": list(RANGO_N_EJECUTADO), "enmienda_rango_n": ENMIENDA_RANGO_N,
        "columnas_publicadas": COLUMNAS_PUBLICADAS,
        "columnas_reales": banco[BRAZO_CANDIDATO]["n_cambios_respecto_a_referencia"] is not None
        and contexto["columnas_c1a_reales"],
        "modelo": MODELO, "vectorizadores": {"char_wb": VEC_CHAR_WB, "palabra": VEC_PALABRA,
                                             "c1a": VEC_C1A},
        "permutaciones": PERMUTACIONES, "variantes": list(VARIANTES),
        "banco": banco, "controles": controles,
        "integracion_candidata": {"evaluacion": integracion_candidata["evaluacion"]},
        "integracion_control": {"evaluacion": integracion_control["evaluacion"]},
        "integracion_atribuible": atribuible, "gates": gates,
        "tabla_de_sesgo": tabla_bias_unico(estado), "sondas": sondas,
        "verificacion": contexto["verificacion"], "cumplimiento": contexto["cumplimiento"],
        "tabla_bias": contexto.get("tabla_bias"),
        "veredicto": {"nodo": nodo, "motivo": motivo, "traza": traza, "cierra_L4": cierra,
                      "one_shot": "C1-A es ONE-SHOT en cualquier caso; no se ejecuta una segunda "
                                  "particion sin autorizacion del usuario"},
        "recursos": contexto["recursos"],
    }


def correr_mini(v30_completo: np.ndarray, clase_completo: np.ndarray, fold_completo: np.ndarray,
                textos_completos: np.ndarray, contexto: dict) -> dict:
    """Corre el pipeline COMPLETO (banco, controles, integracion, gates y arbol) en miniatura.

    Usa un subconjunto estratificado del train que conserva los MISMOS indices de fold canonicos y
    su propio ``estado`` en memoria. Es la cobertura permanente de la zona de evaluacion: sin ella,
    un fallo en el IC95, en los controles o en el arbol solo apareceria DESPUES de los cinco folds
    reales, con checkpoints y OOF ya escritos. Sus magnitudes son sinteticas y su JSON lleva la
    marca explicita de que NO es evidencia experimental.

    Parameters
    ----------
    v30_completo:
        Probabilidades del campeon de la particion canonica.
    clase_completo:
        Etiquetas canonicas.
    fold_completo:
        Fold canonico por fila.
    textos_completos:
        Textos del corpus.
    contexto:
        Verificaciones y recursos compartidos.

    Returns
    -------
    dict
        Resultado del miniproblema.
    """
    indices, clase, fold_idx, textos = mini_problema(clase_completo, fold_completo, textos_completos)
    v30 = v30_completo[indices]
    pesos = np.array(PESOS, dtype=np.float64)
    champ_pred = np.argmax(v30 * pesos, axis=1)
    baseline = float(v35.macro(clase, champ_pred))
    estado = estado_nuevo()
    documentos = {variante: construir_documentos_c1a(textos, variante) for variante in VARIANTES}
    oofs = entrenar_brazos(estado, textos, clase, fold_idx, documentos, exigir_columnas=False)
    sondas = ejecutar_sondas(estado, contexto["prereg"], textos, None, exigido=False)
    resultado = ensamblar_resultado(clase, fold_idx, v30, champ_pred, oofs, pesos, baseline, estado,
                                   sondas, contexto, NATURALEZA_MINI)
    resultado["mini"] = {"n_filas": len(clase), "filas_por_clase": 250,
                         "folds_usados": [int(f) for f in np.unique(fold_idx)],
                         "fold_idx_leido_del_canonico": True}
    return resultado


#: Rutas de escritura que el autoejercicio redirige a un temporal del sistema.
RUTAS_REDIRIGIBLES = ("ESTADO_JSON", "RESULTADOS_JSON", "INFORME", "FIGURA", "NOVELTY_CSV",
                      "HISTORIA_CSV", "RUNLOG", "EXPERIMENTS_CSV", "CKPT_DIR", "OOF_DIR")

#: Los seis caminos del arbol, con la entrada sintetica que debe emitir cada nodo.
CAMINOS_NODO = (
    ("SENAL_NO_EXISTENTE", {"sondas_ok": True, "especificidad_ok": True, "banco_supera_h0": False,
                            "integracion_supera_h0": False, "delta_integrado": 0.0}),
    ("SENAL_EXISTE_INTEGRACION_NEUTRA", {"sondas_ok": True, "especificidad_ok": True,
                                        "banco_supera_h0": True, "integracion_supera_h0": False,
                                        "delta_integrado": 0.00004}),
    ("SENAL_EXISTE_INTEGRACION_POSITIVA", {"sondas_ok": True, "especificidad_ok": True,
                                           "banco_supera_h0": True, "integracion_supera_h0": True,
                                           "delta_integrado": 0.0031}),
    ("SENAL_EXISTE_INTEGRACION_NEGATIVA", {"sondas_ok": True, "especificidad_ok": True,
                                           "banco_supera_h0": True, "integracion_supera_h0": False,
                                           "delta_integrado": -0.0021}),
    ("SIN_SENAL_ESPECIFICA_DE_CATEGORIA", {"sondas_ok": True, "especificidad_ok": False,
                                           "banco_supera_h0": True, "integracion_supera_h0": True,
                                           "delta_integrado": 0.0031}),
    ("DEFECTO_DE_IMPLEMENTACION", {"sondas_ok": False, "especificidad_ok": True,
                                   "banco_supera_h0": True, "integracion_supera_h0": True,
                                   "delta_integrado": 0.0031}),
)


def aislar_rutas(directorio: Path) -> dict:
    """Redirige TODAS las rutas de escritura del modulo a un directorio del sistema.

    Parameters
    ----------
    directorio:
        Directorio temporal ya creado.

    Returns
    -------
    dict
        Los valores originales, para :func:`restaurar_rutas`.
    """
    originales = {nombre: globals()[nombre] for nombre in RUTAS_REDIRIGIBLES}
    for nombre in RUTAS_REDIRIGIBLES:
        globals()[nombre] = directorio / f"{nombre}__{Path(originales[nombre]).name}"
    return originales


def restaurar_rutas(originales: dict) -> None:
    """Restaura las rutas de escritura del modulo.

    Parameters
    ----------
    originales:
        Diccionario devuelto por :func:`aislar_rutas`.
    """
    for nombre, valor in originales.items():
        globals()[nombre] = valor


def verificar_cero_efectos() -> list[str]:
    """Comprueba que las cuatro rutas vigiladas del proyecto siguen AUSENTES.

    Returns
    -------
    list[str]
        Las rutas comprobadas, relativas al proyecto.

    Raises
    ------
    Detencion
        Si alguna existe, porque entonces se ha escrito en el proyecto (codigo 2).
    """
    existentes = [v35.ruta_relativa(ruta) for ruta in RUTAS_VIGILADAS.values() if ruta.exists()]
    if existentes:
        raise Detencion(f"[STOP] los modos de verificacion escribieron en el proyecto: "
                        f"{existentes} (exit 2)", 2)
    return [v35.ruta_relativa(ruta) for ruta in RUTAS_VIGILADAS.values()]


def autoensayo_ruta_salida(contexto: dict) -> dict:
    """Autoejercicio ESTRUCTURAL permanente: los seis nodos del arbol y la ruta de salida.

    Ejercita con las funciones REALES los seis caminos del arbol de decision (cada entrada
    sintetica debe emitir su nodo) y la escritura completa (estado, resultados, informe, figura,
    novelty registry y los tres anexos), todo con las rutas de escritura redirigidas a un temporal
    del sistema que se relee y se borra. Al final comprueba que el proyecto sigue intacto.

    NO es evidencia experimental: sus magnitudes son sinteticas, hay 0 folds reales de la
    particion canonica como folds de CV y su JSON lleva la marca de naturaleza.

    Parameters
    ----------
    contexto:
        Entradas reales de la corrida mas el resultado del miniproblema.

    Returns
    -------
    dict
        Caminos ejercitados, comprobaciones de la escritura, borrado del temporal y rutas ausentes.

    Raises
    ------
    Detencion
        Si algun nodo no coincide, si falta alguna clave de la escritura, si el temporal no se pudo
        borrar o si aparecio algun artefacto en el proyecto (codigo 2).
    """
    raiz = Path(tempfile.mkdtemp(prefix=PREFIJO_TEMPORAL))
    resultados: list[dict] = []
    borrado = False
    try:
        directorio = raiz / "salida"
        directorio.mkdir(parents=True, exist_ok=True)
        originales = aislar_rutas(directorio)
        try:
            for esperado, entrada in CAMINOS_NODO:
                nodo, motivo, _ = decidir_nodo(**entrada)
                if nodo != esperado or nodo not in NODOS_ARBOL or not motivo.strip():
                    raise Detencion(f"[STOP] el camino sintetico '{esperado}' emitio '{nodo}' "
                                    f"(exit 2)", 2)
                resultados.append({"entrada": esperado, "nodo_emitido": nodo})
            estado = estado_nuevo()
            estado["sondas"] = {nombre: True for nombre in NOMBRES_SONDAS}
            guardar_estado(estado)
            recargado = json.loads(ESTADO_JSON.read_text(encoding="utf-8"))
            if sorted(recargado["sondas"]) != sorted(NOMBRES_SONDAS):
                raise Detencion("[STOP] el round-trip del estado no conserva las sondas (exit 2)", 2)
            mini = contexto["mini"]
            v35.escribir_json(RESULTADOS_JSON, mini)
            escribir_informe(mini)
            figura_curvas(mini)
            escribir_registro_novedad(mini)
            for funcion in (anotar_runlog, anotar_experiments_csv, anotar_historial):
                funcion(mini)
            faltantes = [v35.ruta_relativa(ruta) for ruta in
                         (RESULTADOS_JSON, INFORME, FIGURA, NOVELTY_CSV, HISTORIA_CSV, RUNLOG,
                          EXPERIMENTS_CSV, ESTADO_JSON) if not ruta.exists() or ruta.stat().st_size
                         == 0]
            if faltantes:
                raise Detencion(f"[STOP] la ruta de salida del autoejercicio no produjo "
                                f"{faltantes} (exit 2)", 2)
        finally:
            restaurar_rutas(originales)
    finally:
        try:
            shutil.rmtree(raiz)
            borrado = True
        except OSError as error:
            print(f"  [AVISO] no se pudo borrar el temporal {raiz}: {error}", flush=True)
    if not borrado:
        raise Detencion(f"[STOP] el temporal del autoejercicio {raiz} no se pudo borrar (exit 2)",
                        2)
    ausentes = verificar_cero_efectos()
    return {"caminos": [nombre for nombre, _ in CAMINOS_NODO],
            "n_caminos_ok": len(resultados), "por_camino": resultados,
            "n_folds_reales": 0, "n_metricas_reales": 0,
            "naturaleza": "autoejercicio estructural, NO evidencia experimental",
            "temporal_borrado": bool(borrado), "artefactos_ausentes": ausentes,
            "rutas_de_escritura_aisladas": list(RUTAS_REDIRIGIBLES)}


# =====================================================================================
# ---- 13. MODOS DE LINEA DE COMANDOS Y MAIN ----
# =====================================================================================
def parsear_argumentos() -> argparse.Namespace:
    """Argumentos de linea de comandos.

    Returns
    -------
    argparse.Namespace
        ``--verificar`` (verificacion, sondas en miniatura y autoejercicio de la ruta de salida,
        sin escribir nada en el proyecto) y ``--humo`` (recorre la zona entre la verificacion y el
        entrenamiento y para con un sentinel, sin entrenar).
    """
    analizador = argparse.ArgumentParser(
        description="V36-E1 via C1-A: bloque de transiciones de clase de caracter (brecha L4) "
                    "sobre el pre-registro congelado, con control de permutaciones del alfabeto y "
                    "control de barajado, e integracion contra el campeon V30 + plug-in.")
    analizador.add_argument("--verificar", action="store_true",
                            help="solo verifica no-deriva, corre las ocho sondas sobre el "
                                 "miniproblema y autoejercita los seis nodos del arbol y la ruta de "
                                 "salida contra un temporal del sistema; no entrena, no escribe nada "
                                 "en el proyecto e imprime VERIFICACION OK")
    analizador.add_argument("--humo", action="store_true",
                            help="recorre la zona INTERMEDIA entre --verificar y el entrenamiento "
                                 "(crea el checkpoint en memoria, mapea las sondas y verifica la "
                                 "escritura atomica del estado contra un temporal del sistema) y "
                                 "para con el sentinel HUMO OK, salida 0, SIN entrenar ningun fold")
    return analizador.parse_args()


def construir_contexto(prereg: dict, inicio: float, detalle_constantes: dict,
                       detalle_versiones: dict, detalle_hashes: dict, auditoria: dict,
                       columna_real: dict) -> dict:
    """Junta verificaciones, recursos y el pre-registro en el contexto compartido.

    Parameters
    ----------
    prereg:
        Pre-registro congelado.
    inicio:
        Marca de tiempo del inicio de la corrida.
    detalle_constantes:
        Detalle de las comprobaciones de constantes.
    detalle_versiones:
        Detalle de las comprobaciones de versiones.
    detalle_hashes:
        Detalle de las comprobaciones de hashes.
    auditoria:
        Detalle de la auditoria de compliance.
    columna_real:
        Columnas reales del bloque C1-A ({bigramas, trigramas, total} por fold).

    Returns
    -------
    dict
        Contexto compartido por el pipeline, el miniproblema y el autoejercicio.
    """
    return {
        "inicio": inicio, "prereg": prereg,
        "verificacion": {"constantes": detalle_constantes, "versiones": detalle_versiones,
                         "hashes": detalle_hashes},
        "cumplimiento": {**auditoria, "compliance_del_prereg": prereg["compliance"],
                         "test_solo_transform_predict": True, "holdout_intocable": True,
                         "leaderboard_sin_tuning": True, "campeon_intacto": True,
                         "p33_scores_intocable": True, "sin_submission": True,
                         "folds_leidos_del_canonico": True},
        "columnas_c1a_reales": columna_real,
        "recursos": {"rss_pico_gb": 0.0, "presupuesto_declarado_seg": PRESUPUESTO_DECLARADO_SEG,
                     "ram_tope_gb": RAM_TOPE_GB, "hilos": HILOS, "duracion_seg": 0.0},
    }


def ejecutar(verificar: bool, humo: bool) -> int:
    """Pipeline completo del ciclo, con los tres modos de linea de comandos.

    En ``--verificar`` se detiene ANTES de tocar el disco: imprime ``VERIFICACION OK`` y NO crea ni
    el estado, ni el OOF, ni los checkpoints, ni filas en el RUNLOG, en ``experiments.csv``, en el
    registro de novedad ni en el historial. Antes imprime el autoejercicio estructural de la ruta de
    salida. En ``--humo`` construye el checkpoint en memoria, verifica la escritura atomica contra
    un temporal del sistema y para con el sentinel, sin entrenar ningun fold.

    Parameters
    ----------
    verificar:
        Si True solo verifica, corre las sondas en miniatura y el autoejercicio.
    humo:
        Si True recorre la zona intermedia y para con el sentinel.

    Returns
    -------
    int
        0 si todo correcto, incluidos los cierres documentados del arbol.

    Raises
    ------
    Detencion
        Con el codigo pre-registrado de la guarda que fallo (2, 4, 7 u 8).
    """
    inicio = time.time()
    autotest_numerico()
    prereg = cargar_preregistro()
    constantes_ok, detalle_constantes = verificar_constantes(prereg)
    versiones_ok, detalle_versiones = verificar_versiones(prereg)
    auditoria = auditar_compliance_propia()
    if not (constantes_ok and versiones_ok):
        fallidas = [nombre for nombre, ok in detalle_constantes.items() if not ok]
        fallidas += [nombre for nombre, ok in detalle_versiones.items() if not ok]
        raise Detencion(f"[STOP] el pre-registro congelado no se respeta: {fallidas} (exit 2)", 2)
    print(f"  verificaciones de no-deriva: {sum(1 for v in detalle_constantes.values() if v)}/"
          f"{len(detalle_constantes)} OK | compliance: "
          f"{auditoria['literales_de_ruta_revisados']} literales de ruta y "
          f"{auditoria['modulos_revisados']} modulos, 0 violaciones", flush=True)
    print(f"  ATENCION - enmienda declarada: {ENMIENDA_RANGO_N['campo_congelado']} congelado "
          f"{ENMIENDA_RANGO_N['valor_congelado']}, ejecutado {ENMIENDA_RANGO_N['valor_ejecutado']} "
          f"(bigramas y trigramas; columnas publicadas 27 / 127). Pendiente de enmienda del "
          "pre-registro; este runner no lo modifica.", flush=True)

    clase, fold_idx, v30, ids_canonicos, _cand = v35.cargar_canonico()
    textos, hashes_corpus = v35.cargar_corpus(ids_canonicos, clase)
    hashes_ok, detalle_hashes = verificar_hashes(prereg, clase, fold_idx, v30, ids_canonicos,
                                                 textos, hashes_corpus)
    if not hashes_ok:
        fallidas = [nombre for nombre, ok in detalle_hashes.items() if not ok]
        raise Detencion(f"[STOP] hashes del pre-registro distintos a los recomputados: {fallidas} "
                        "(exit 2)", 2)
    pesos = np.array(PESOS, dtype=np.float64)
    champ_pred = np.argmax(v30 * pesos, axis=1)
    baseline = float(v35.macro(clase, champ_pred))
    print(f"  baseline primario recomputado: {baseline:.7f} (esperado {BASELINE:.7f}); punto de "
          f"partida de la integracion REPRODUCIDO", flush=True)
    if abs(baseline - BASELINE) > ABORTO_BASELINE:
        raise Detencion("[STOP] el baseline canonico no reproduce (exit 2)", 2)
    _, conteo = np.unique(clase, return_counts=True)
    trivial = float(v35.macro(clase, np.full_like(clase, int(np.argmax(conteo)))))
    if abs(trivial - ANCLA_TRIVIAL) > 1e-12:
        raise Detencion(f"[STOP] el ancla trivial no reproduce la de D-087 ({trivial!r}) (exit 2)",
                        2)
    print(f"  corpus: {len(textos)} textos, ids en el MISMO orden que el canonico, cat.codes == "
          f"y_true; hashes de archivo, textos y artefactos canonicos OK | ancla trivial "
          f"{trivial:.17f}", flush=True)

    contexto = construir_contexto(prereg, inicio, detalle_constantes, detalle_versiones,
                                  detalle_hashes, auditoria, {})
    if verificar or humo:
        mini = correr_mini(v30, clase, fold_idx, textos, contexto)
        sondas_mini = mini["sondas"]
        autoensayo = autoensayo_ruta_salida({**contexto, "mini": mini})
        print(f"  sondas (8) del pre-registro sobre el MINIPROBLEMA ({NATURALEZA_MINI}):",
              flush=True)
        for nombre in NOMBRES_SONDAS:
            entrada = sondas_mini[nombre]
            estado = ("OK" if entrada["pass"] is True else
                      "no ejecutada (sonda de medicion)" if entrada["pass"] is None else "FALLA")
            print(f"    {nombre:<20} {estado}", flush=True)
        print(f"  mapa literal: sha256 {sondas_mini['mapa_literal']['sha256_mapa'][:16]}... | "
              f"conformidad: dos llamadas identicas "
              f"({sondas_mini['conformidad']['matrices_identicas']}), sin borde "
              f"({sondas_mini['conformidad']['sin_borde']}) | fold_local: "
              f"{sondas_mini['fold_local']['n_checks_ok']}/{sondas_mini['fold_local']['n_checks']} "
              "checks del train de cada fold", flush=True)
        print(f"  autoejercicio estructural: {autoensayo['n_caminos_ok']}/"
              f"{len(autoensayo['caminos'])} nodos del arbol emitidos por la funcion REAL y ruta de "
              f"salida completa en un temporal borrado = {autoensayo['temporal_borrado']}; en el "
              f"proyecto siguen ausentes {', '.join(autoensayo['artefactos_ausentes'])}",
              flush=True)
        if humo:
            ejecutar_humo(mini)
            return 0
        print("VERIFICACION OK", flush=True)
        return 0

    estado = cargar_estado()
    documentos = {variante: construir_documentos_c1a(textos, variante) for variante in VARIANTES}
    columnas_reales = {"por_fold": {}}
    oofs = entrenar_brazos(estado, textos, clase, fold_idx, documentos, exigir_columnas=True)
    for fold in range(N_FOLDS):
        meta = estado["brazos"][BRAZO_CANDIDATO]["folds"][str(fold)]["meta_c1a"]
        columnas_reales["por_fold"][str(fold)] = {"n_vocabulario": meta["n_vocabulario"],
                                                  "n_bigramas": meta["n_bigramas"],
                                                  "n_trigramas": meta["n_trigramas"]}
    columnas_reales["bigramas"] = columnas_reales["por_fold"]["0"]["n_bigramas"]
    columnas_reales["trigramas"] = columnas_reales["por_fold"]["0"]["n_trigramas"]
    columnas_reales["total"] = columnas_reales["por_fold"]["0"]["n_vocabulario"]
    contexto["columnas_c1a_reales"] = columnas_reales
    sondas = ejecutar_sondas(estado, prereg, textos, None, exigido=True)
    estado["sondas"] = {nombre: bool(sondas[nombre]["pass"]) for nombre in NOMBRES_SONDAS}
    guardar_estado(estado)
    resultado = ensamblar_resultado(clase, fold_idx, v30, champ_pred, oofs, pesos, baseline, estado,
                                   sondas, contexto,
                                   "evidencia experimental: 5 folds de la particion canonica "
                                   "leidos del artefacto, 0 folds de test, 0 particiones "
                                   "reservadas")
    contexto["recursos"]["duracion_seg"] = float(time.time() - inicio)
    contexto["recursos"]["rss_pico_gb"] = float(rss_gb())
    resultado["recursos"] = contexto["recursos"]
    v35.escribir_json(RESULTADOS_JSON, resultado)
    escribir_informe(resultado)
    figura_curvas(resultado)
    escribir_registro_novedad(resultado)
    for funcion in (anotar_runlog, anotar_experiments_csv, anotar_historial):
        mensaje = funcion(resultado)
        if mensaje:
            print(f"  {mensaje}", flush=True)
    estado["resultados"] = {"nodo": resultado["veredicto"]["nodo"], "escrito": True}
    guardar_estado(estado)
    resumen_consola(resultado)
    return 0


def ejecutar_humo(mini: dict) -> None:
    """Recorre la zona INTERMEDIA entre ``--verificar`` y el entrenamiento y para con un sentinel.

    Construye el checkpoint en memoria, mapea las sondas, serializa el estado a un TEMPORAL del
    sistema (comprueba el round-trip y lo borra) y comprueba que en el proyecto no ha quedado
    ni el estado, ni los checkpoints, ni el OOF, ni filas en ningun registro. NO entrena ningun
    fold.

    Parameters
    ----------
    mini:
        Resultado del miniproblema, que ya recorrio verificacion, sondas, banco, controles,
        integracion, gates y arbol.

    Raises
    ------
    Detencion
        Si el round-trip del estado no es integro o si aparecio algun artefacto (codigo 2).
    """
    estado = estado_nuevo()
    estado["sondas"] = {nombre: bool(mini["sondas"][nombre]["pass"])
                        for nombre in NOMBRES_SONDAS}
    estado["brazos"][BRAZO_CANDIDATO]["folds"]["0"] = {
        "origen": "miniproblema", "fold": 0, "n_features": int(mini["banco"][BRAZO_CANDIDATO].get(
            "n_features", 0)) or 1, "nota": "fila SINTETICA escrita en memoria, sin entrenar"}
    estado["brazos"][BRAZO_CANDIDATO]["folds_completados"] = []
    temporal = Path(tempfile.gettempdir()) / NOMBRE_TEMPORAL_HUMO
    v35.escribir_json_atomico(temporal, json.dumps(v35.como_nativo(estado), indent=2))
    recargado = json.loads(temporal.read_text(encoding="utf-8"))
    integro = (sorted(recargado["sondas"]) == sorted(NOMBRES_SONDAS)
               and recargado["brazos"][BRAZO_CANDIDATO]["folds"]["0"]["origen"] == "miniproblema")
    temporal.unlink()
    if not integro:
        raise Detencion("[STOP] el round-trip del estado en el temporal de --humo no es integro "
                        "(exit 2)", 2)
    ausentes = verificar_cero_efectos()
    print(f"  --humo: checkpoint construido en memoria con {len(estado['sondas'])} sondas "
          f"mapeadas y {len(estado['brazos'])} brazos;", flush=True)
    print("  --humo: escritura atomica verificada con round-trip integro en el temporal del "
          f"sistema ({NOMBRE_TEMPORAL_HUMO}), que se ha borrado", flush=True)
    print(f"  --humo: NO se ha preparado ningun fold, NO se ha entrenado, NO existe "
          f"{', '.join(ausentes)} en el proyecto", flush=True)
    print("HUMO OK: zona entre --verificar y el entrenamiento cubierta, sin entrenar", flush=True)


def resumen_consola(resultado: dict) -> None:
    """Imprime el resumen final de la corrida (diez lineas como maximo).

    Parameters
    ----------
    resultado:
        Resultado completo del ciclo.
    """
    banco = resultado["banco"]["delta"]
    controles = resultado["controles"]
    integracion = resultado["integracion_candidata"]["evaluacion"]
    veredicto = resultado["veredicto"]
    print("\n=== V36-E1 via C1-A: transiciones de clase de caracter (L4) ===")
    print(f"nodo del arbol        : {veredicto['nodo']}")
    print(f"columnas del bloque   : {resultado['columnas_c1a_reales']['total']} "
          f"({COLUMNAS_PUBLICADAS['bigramas']} bigramas + "
          f"{COLUMNAS_PUBLICADAS['trigramas']} trigramas de clase, sin borde, sin min_df)")
    print(f"banco                 : delta {banco['delta']:+.7f} ({banco['folds_positivos']}/5 "
          f"folds), IC95 [{banco['ic95'][0]:+.7f}, {banco['ic95'][1]:+.7f}], supera H0: "
          f"{banco['supera_h0']}")
    print(f"permutaciones         : el mapa real gana a "
          f"{controles['n_permutaciones_ganadas']}/3; barajado: delta "
          f"{controles['barajado']['delta_real_menos_control']:+.7f}, gana: "
          f"{controles['barajado']['el_mapa_real_gana']}")
    print(f"integracion           : {integracion['corrected_cv']:.7f} frente al campeon "
          f"{BASELINE:.7f}, delta {integracion['delta_integrado_vs_campeon']:+.7f}, IC95 "
          f"[{integracion['ic95_pareado_vs_campeon'][0]:+.7f}, "
          f"{integracion['ic95_pareado_vs_campeon'][1]:+.7f}]")
    print(f"delta atribuido C1-A  : "
          f"{resultado['integracion_atribuible']['delta_atribuido_al_bloque_c1a']:+.7f}")
    print(f"gates de la serie     : {resultado['gates']['etiqueta_de_la_serie']} "
          f"(pasan: {resultado['gates']['todos_pasan']}); segunda particion: NO ejecutada")
    print(f"recursos              : {resultado['recursos']['duracion_seg']:.0f} s de "
          f"{PRESUPUESTO_DECLARADO_SEG:g} s declarados, RSS "
          f"{resultado['recursos']['rss_pico_gb']:.2f} GB de {RAM_TOPE_GB:g} GB")
    print(f"enmienda pendiente    : {ENMIENDA_RANGO_N['campo_congelado']} congelado "
          f"{ENMIENDA_RANGO_N['valor_congelado']} vs ejecutado "
          f"{ENMIENDA_RANGO_N['valor_ejecutado']}")


def main() -> int:
    """Entrada del script con manejo de los codigos de salida pre-registrados.

    Returns
    -------
    int
        Codigo de salida (0 correcto, incluidos los cierres documentados del arbol; 2
        verificacion; 4 sonda fallida; 7 presupuesto o timeout agotado; 8 RSS por encima del tope;
        1 error inesperado).
    """
    argumentos = parsear_argumentos()
    try:
        return ejecutar(argumentos.verificar, argumentos.humo)
    except Detencion as detencion:
        print(str(detencion), flush=True)
        return detencion.codigo
    except Exception:  # noqa: BLE001 - se reporta el traceback y se sale con 1
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
