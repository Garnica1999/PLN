"""v32_05_ftrl_challenger.py
Fecha: 2026-09-30
Descripcion: implementacion exacta del pre-registro congelado V32-E3 (P3): challenger
FTRL-Proximal propio (McMahan et al. 2013) sobre TF-IDF ajustada por fold (brazo decisional
A1), gemelo SGDClassifier con el mismo presupuesto de 10 pasadas sobre la MISMA matriz
(control de atribucion A2) y el mismo FTRL con las etiquetas del fold-train permutadas
(control negativo A3); ejecucion resumible por fold con escritura atomica del OOF, sondas
(a)-(d) obligatorias antes de entrenar, regla de fallback escrita antes de ejecutar, sonda
de determinismo bit a bit, integracion P6/P7 congelada, gates, caps y arbol de decision.

Estado del arte aplicado: FTRL-Proximal con regularizacion L1/L2 explicita y sesgo como
coordenada regularizada (x0 = 1.0), 6 sigmoides one-vs-rest, 10 pasadas fijas con barajado
sembrado por epoca, float64 y orden CSR canonico (lazo mono-hilo); gemelo
sklearn.linear_model.SGDClassifier (loss log_loss, alpha 1e-6, max_iter 10, tol nulo) sobre
la misma matriz; TF-IDF de palabra (1,2) con min_df 3, max_df 0.5 y sublinear_tf ajustada
SOLO en el fold-train; bootstrap pareado de 2000 remuestreos (semilla 20261010) para los
intervalos de confianza del delta de A1 y de la diferencia A1-A2.

Fuentes: reports/v32_p3_preregistro.md y experiments/v32_04_preregistro_ftrl.json.
Codigos de salida: 0 fin correcto; 2 verificacion (constantes, hashes, corpus, auditoria
AST o invariantes); 4 sonda (a)-(d) fallida antes de entrenar; 5 gate A3 incoherente
(sospecha de fuga); 6 calibracion de fold 0 excedida; 7 timeout global; 8 RSS por encima
del tope declarado; 1 error inesperado con traceback.
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
SEMILLA_BARAJADO = 20261011
SEMILLA_SEGUNDA_PARTICION = 987654
PROYECCION_FACTOR = 9.0
PROYECCION_MAX_H = 20.0
TIMEOUT_H = 24.0
RAM_TOPE_GB = 20.0
L2_CLIP = 1e-12
C_LOGREG = 1.0
MAX_ITER_LOGREG = 2000
TOL_DETERMINISMO = 1e-9
FASE_ESTADO = "V32-E3 ejecucion (checkpoint resumible)"

#: Palabras vigiladas por el pre-registro. Ningun literal de ruta de este script puede
#: contenerlas en minusculas: las cuatro primeras son particiones reservadas, la quinta es
#: la alternativa declarada y descartada, y las seis ultimas son familias vetadas por el
#: plan V32 (la quinta aparece aqui como elemento de la lista, no como ruta).
PALABRAS_VIGILADAS = ("test", "holdout", "submission", "leaderboard", "vowpal", "vw", "torch",
                      "transformers", "gensim", "fasttext")
#: Familias de librerias vetadas por el plan V32. Se comprueban sobre los MODULOS IMPORTADOS
#: (no solo sobre literales de ruta): sin esto, un ``import`` vetado pasaria inadvertido.
FAMILIAS_VETADAS = ("vowpal", "vw", "torch", "transformers", "gensim", "fasttext")
BRAZOS = ("A1_ftrl", "A2_sgd_twin", "A3_ftrl_barajado")
BRAZOS_FTRL = ("A1_ftrl", "A3_ftrl_barajado")
BRAZOS_CON_SONDA = ("A1_ftrl", "A2_sgd_twin")
NOMBRES_BRAZO = {"A1_ftrl": "A1 (FTRL, decisional)",
                 "A2_sgd_twin": "A2 (gemelo SGD, control de atribucion)",
                 "A3_ftrl_barajado": "A3 (FTRL con etiquetas barajadas, control negativo)"}
CAMPOS_FOLD = ("n_features", "n_vocabulario", "vocab_sha256", "idf_sha256", "n_filas_fit",
               "alpha", "tiempo_seg", "log_loss_train_ovr", "log_loss_val_ovr",
               "log_loss_crudo_fold_val", "macro_f1_crudo_fold_val", "n_iter_sgd",
               "n_train_fold", "n_val_fold")

# -------------------------------------------------------------------------------------
# Espejos LITERALES del pre-registro congelado. El codigo lee sus parametros de aqui, de
# modo que no puede haber deriva entre el protocolo congelado y la ejecucion: si el JSON
# cambia, el assert de no-deriva falla y el script sale con codigo 2 sin entrenar.
# -------------------------------------------------------------------------------------
FTRL = {
    "algoritmo": "FTRL-Proximal (McMahan et al. 2013), implementacion propia exacta",
    "alpha": 0.05, "lambda1": 0.0, "lambda2": 1.0, "beta": 1.0, "bias": True, "x0": 1.0,
    "power_t": None, "epocas": 10, "criterio_parada": "ninguno (10 pasadas fijas)",
    "ovr": "6 modelos binarios con sigmoid; p no suma 1",
    "barajado": "np.random.default_rng(20261010+fold) sobre los indices de fold-train por epoca",
    "combinador_p": "normalizacion L2 por fila (comparador canonico); se reportan ademas "
                    "argmax crudo, macro-F1 diagonal y log_loss",
    "actualizacion": "sigma=(sqrt(n+g^2)-sqrt(n))/(alpha*beta); z+=g-sigma*w; n+=g^2; w=0 si "
                     "|z|<=lambda1; si no sign(z)*(|z|-lambda1)/((beta+sqrt(n))/alpha+lambda2)",
    "alternativa_no_usada": "vowpalwabbit 9.11.8 instalado en el venv y NO usado "
                            "(parametrizacion y bit_precision distintas; sin sondesn de "
                            "verificacion)",
}
SGD_TWIN = {
    "estimator": "sklearn.linear_model.SGDClassifier", "loss": "log_loss", "alpha": 1e-06,
    "max_iter": 10, "tol": None, "fit_intercept": True, "average": False,
    "learning_rate": "optimal", "random_state": "20261010+fold", "n_jobs": 32,
    "desvio_de_precedente": "E-011 usaba max_iter=30 y tol=1e-4; aqui 10/tol=None para igualar "
                            "el presupuesto de pasadas con A1 (desvio declarado, no fallo)",
    "limitacion": "en A1 el bias es coordenada regularizada de FTRL; en A2 no se regulariza "
                  "(limite de sklearn): asimetria declarada en el informe",
}
A3_CONTROL = {
    "descripcion": "los mismos 6 FTRL con etiquetas de fold-train permutadas con "
                   "np.random.default_rng(20261011+fold); no decisional",
    "umbral_fuga": "A3 debe quedar por debajo de la linea base de la clase mayoritaria; si la "
                   "supera -> STOP por sospecha de fuga (gate A3)",
}
SONDAS_PREREG = {
    "a_vectorizado_vs_lazo": {"problema": "aleatorio 200x50x6", "tolerancia": 1e-12},
    "b_punto_fijo_analitico": {"config": "lambda1=lambda2=0 con feature constante",
                               "residuo_max": 1e-09},
    "c_totalf1_macro": {"comparacion": "TotalF1:average=Macro == macro-F1 de sklearn",
                        "tolerancia": 1e-06},
    "d_sin_nan_logloss_finito": True,
    "politica": "si alguna sonda falla -> exit 4 ANTES de entrenar; cualquier ajuste exige "
                "enmienda pre-ejecucion (precedente f0bf731)",
}
FALLBACK_PREREG = {
    "criterio": "en el fold 0 de A1, sobre las filas de fold-train: log_loss OvR medio tras 10 "
                "pasadas <= 0.8 * log_loss tras la pasada 1, y sin NaN/inf",
    "si_falla": "usar la alternativa declarada alpha=0.005 (lambda2=1.0 resto igual) y registrar "
                "config_fallback_usada=True en el JSON de resultados y en el informe",
    "prohibido": "decidir con fold-val o con el leaderboard; si el criterio se fijara despues "
                 "de observar fold-0, el resultado se marca post-hoc en D-083",
}
DETERMINISMO_PREREG = {
    "seeds": {"global": 20261010, "barajado_etiquetas": 20261011,
              "orden_datos": "default_rng(20261010+fold) por epoca"},
    "hilos": "OMP/MKL/OPENBLAS_NUM_THREADS = 32 fijos antes de importar numpy; el lazo FTRL es "
             "mono-hilo",
    "esperado": "exacto bit a bit (float64, orden CSR, permutacion sembrada, sin reducciones "
                "multi-hilo ni atomicos ni BLAS denso)",
    "sonda": "reentrenar fold 0 de A1 y de A2 con config identica y comparar bit a bit",
    "tolerancias": "0 exacto; <=1e-9 conforme; >1e-9 declarar rango y exigir delta_gate > rango; "
                   "rango >= 0.0008876 o solape con el gate => no concluyente",
}
INTEGRACION_PREREG = {
    "mecanismo": "combinador logistico anidado sobre [L2(v30_prob) || L2(p_brazo)] (12 features, "
                 "L2 por fila con clip 1e-12); LogReg(C=1.0, max_iter=2000, lbfgs) por fold "
                 "(fit fold!=k, predice fold==k)",
    "decision": "argmax(p_comb * w_plugin CONGELADO)",
    "diagnosticos": ["argmax_sin_regla", "delta vs nested 0.6154643", "log_loss por brazo",
                     "macro-F1 diagonal 6x6"],
}
GATES_PREREG = {
    "baseline_abort": 5e-07, "delta_min": 0.0015, "folds_min": 4,
    "control": "delta_int(A1) > delta_int(A2) Y bootstrap pareado de la diferencia "
               "(F1_combA1 - F1_combA2; 2000; semilla 20261010) con IC95 sin cero",
    "control_negativo": "si delta_int(A3) >= delta_int(A1): NO_ATRIBUIBLE (precedencia)",
    "a3_gate": "A3 debe quedar por debajo de la linea base de la clase mayoritaria; si la supera "
               "-> STOP por sospecha de fuga (gate A3)",
    "bootstrap": {"n": 2000, "semilla": 20261010, "ic95_sin_cero": True},
    "sin_leakage": True, "determinismo_concluyente": True,
    "cap_ic95_diferencia": "si el IC95 de la diferencia contiene 0, el veredicto queda capado a "
                           "MARGINAL aunque A1 pase el resto de gates (decision del lead, "
                           "registrada; puede degradar un resultado genuinamente bueno)",
    "cap_determinismo": "sonda no concluyente -> el veredicto no puede ser PROMISING",
}
ARBOL_PREREG = {
    "NO_ATRIBUIBLE": "si delta_int(A3) >= delta_int(A1) (precedencia)",
    "PROMISING": "gates completos y IC95 de la diferencia excluye 0 -> segunda particion "
                 "congelada (semilla 987654) en este mismo ciclo",
    "MARGINAL": "delta > 0 sin gates completos (incluye el cap por IC95) -> no se adopta",
    "CLOSED": "delta <= 0 -> se cierra P3 y la reapertura V32 termina",
    "nota": "la formulacion del arquitecto 'A1 <= A2 -> NO_ATRIBUIBLE' queda subsumida por este "
            "arbol, que es el validado por QA en el ciclo 2 (P2)",
    "submission": "jamas automatica; requeriria gate completo + segunda particion + "
                  "autorizacion explicita",
}
EJECUCION_PREREG = {
    "coste_estimado": "~25-30 min CPU (fit TF-IDF ~90 s por fold compartido por los 3 brazos; A1 "
                      "~60 s; A2 ~20 s; A3 ~60 s)",
    "ram_tope_gb": 20.0,
    "ram_medicion": "resource.getrusage(ru_maxrss); si se supera -> STOP registrado; NO anadir "
                    "max_features sobre la marcha",
    "calibracion": "tras el fold 0 de A1: proyeccion = 9 x t_fold0; si supera 20.0 h -> parar y "
                   "reportar",
    "timeout_global": "24.0 h",
    "checkpoints": "experiments/v32_05_estado.json (resumible por fold)",
    "escritura_atomica": "tmp + os.replace en artifacts/p33_scores/v32_p3/",
}
#: Campos numericos de compliance. Los dos campos de prosa del pre-registro (nota y
#: ast_audit) NO se reflejan como literales porque describen la propia auditoria y
#: contienen por construccion las palabras vigiladas: su presencia marcaria un falso positivo.
COMPLIANCE_CIFRAS = {"embeddings": 0, "transformers": 0, "llm": 0, "datos_externos": 0,
                     "test": 0, "holdout": 0, "leaderboard": 0, "submission": 0}
COMPLIANCE_AUDITORIAS = ["hashes del manifest", "sha256 del archivo y del array de textos",
                        "sha256 del vectorizador", "vocabulario e idf por fold", "sondas (a)-(d)",
                        "sonda de reentreno", "RSS pico", "config_fallback_usada",
                        "escritura atomica"]

#: Configuracion del vectorizador: IDENTICA a la congelada (hash 9bae1576 del JSON). En el
#: fit se convierte ngram_range a tupla porque sklearn 1.8.0 lo exige; es el mismo valor.
VECTORIZADOR = {"analyzer": "word", "ngram_range": [1, 2], "min_df": 3, "max_df": 0.5,
                "sublinear_tf": True, "lowercase": True, "strip_accents": None,
                "token_pattern": r"(?u)\b\w+\b"}
VECTORIZADOR_FIT = {**VECTORIZADOR, "ngram_range": (1, 2)}

#: Parametros derivados de los espejos congelados (no se escriben a mano dos veces).
ALPHA = float(FTRL["alpha"])
LAMBDA1 = float(FTRL["lambda1"])
LAMBDA2 = float(FTRL["lambda2"])
BETA = float(FTRL["beta"])
X0 = float(FTRL["x0"])
EPOCAS = int(FTRL["epocas"])
ALPHA_ALTERNATIVO = 0.005
FALLBACK_FACTOR = 0.8
SGD_ALPHA = float(SGD_TWIN["alpha"])
SGD_MAX_ITER = int(SGD_TWIN["max_iter"])
TOL_SONDA_A = float(SONDAS_PREREG["a_vectorizado_vs_lazo"]["tolerancia"])
TOL_SONDA_B = float(SONDAS_PREREG["b_punto_fijo_analitico"]["residuo_max"])
TOL_SONDA_C = float(SONDAS_PREREG["c_totalf1_macro"]["tolerancia"])
DIM_SONDA_A = (200, 50, 6)

CANONICO = ROOT / "artifacts" / "p33_scores" / "todos"
OOF_DIR = ROOT / "artifacts" / "p33_scores" / "v32_p3"
OOF_ARCHIVO = {"A1_ftrl": "a1_ftrl", "A2_sgd_twin": "a2_sgd_twin",
               "A3_ftrl_barajado": "a3_ftrl_barajado"}
#: OOF CRUDO (OvR sin normalizar) de A2: solo diagnostico, no entra en la integracion.
OOF_CRUDO_ARCHIVO = {"A2_sgd_twin": "a2_sgd_twin_crudo"}
PREREG_JSON = cfg.EXPERIMENTS / "v32_04_preregistro_ftrl.json"
ESTADO_JSON = cfg.EXPERIMENTS / "v32_05_estado.json"
RESULTADOS_JSON = cfg.EXPERIMENTS / "v32_05_ftrl_challenger.json"
INFORME = cfg.REPORTS / "v32_p3_ftrl_challenger.md"
FIGURA = cfg.FIGURES / "v32_p3_curvas.png"
RUNLOG = ROOT / "RUNLOG.md"
EXPERIMENTS_CSV = ROOT / "experiments.csv"
ARCHIVO_PRE_REGISTRO = "reports/v32_p3_preregistro.md"
RUTAS_HASHES = {
    "canonico_y_true": CANONICO / "y_true.npy",
    "canonico_fold_idx": CANONICO / "fold_idx.npy",
    "canonico_v30_prob": CANONICO / "v30_prob.npy",
    "canonico_ids": CANONICO / "ids.npy",
    "manifest_canonico": CANONICO / "manifest.json",
}
CLAVES_HASH_CORPUS = ("corpus_archivo", "corpus_textos", "vectorizador")


class Detencion(RuntimeError):
    """Detencion pre-registrada: el mensaje se imprime y el proceso sale con ``codigo``."""

    def __init__(self, mensaje: str, codigo: int) -> None:
        super().__init__(mensaje)
        self.codigo = codigo


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
    """Ruta congelada del OOF float64 (normalizado L2) del brazo.

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


def ruta_oof_crudo(brazo: str) -> Path:
    """Ruta del OOF crudo (OvR sin normalizar) de A2, solo diagnostico.

    Parameters
    ----------
    brazo:
        Nombre del brazo (solo A2 tiene artefacto crudo).

    Returns
    -------
    Path
        Ruta del ``.npy`` crudo.
    """
    return OOF_DIR / f"{OOF_CRUDO_ARCHIVO[brazo]}.npy"


def como_nativo(valor: object) -> object:
    """Convierte recursivamente a escalares nativos de JSON (float/int/bool/str/None).

    Los float no finitos se guardan como ``None`` para que el JSON siga siendo valido.

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
    """Escribe JSON indentado con escalares nativos y codificacion ASCII.

    Parameters
    ----------
    ruta:
        Archivo destino (se crean sus directorios).
    datos:
        Estructura a serializar.
    """
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(json.dumps(como_nativo(datos), indent=2, ensure_ascii=True) + "\n",
                    encoding="utf-8")


def cargar_preregistro() -> dict:
    """Lee el JSON congelado del pre-registro V32-E3.

    Returns
    -------
    dict
        Contenido completo de ``experiments/v32_04_preregistro_ftrl.json``.

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
        Rutas relativas de los entregables (JSON, estado, informe, figura y los tres OOF).
    """
    nombres = [ruta_relativa(RESULTADOS_JSON), ruta_relativa(ESTADO_JSON),
               ruta_relativa(INFORME), ruta_relativa(FIGURA), Path(__file__).name]
    nombres += [OOF_ARCHIVO[brazo] for brazo in BRAZOS]
    return nombres


def verificar_constantes(prereg: dict) -> tuple[bool, dict]:
    """Compara cada constante local contra el JSON congelado, una por una.

    Los bloques del pre-registro (ftrl, a2_sgd_twin, a3_control_negativo, sondas, fallback,
    determinismo, integracion, gates, arbol y ejecucion) se comparan como espejo literal: si
    el protocolo cambia, el runner se detiene con codigo 2 antes de entrenar.

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
    checks = {
        "fase": prereg["fase"] == "V32-E3 (P3) pre-registro congelado",
        "fecha": prereg["fecha"] == FECHA,
        "baseline_primario": float(prereg["baseline_primario"]) == BASELINE,
        "referencia_nested": float(prereg["referencia_nested"]) == NESTED,
        "pesos_plugin": [float(valor) for valor in prereg["pesos_plugin"]] == PESOS,
        "versiones_claves": set(prereg["versiones"]) == {"sklearn", "numpy", "pandas"},
        "hashes_claves": set(prereg["hashes"]) == set(RUTAS_HASHES) | set(CLAVES_HASH_CORPUS),
        "ftrl_config": prereg["ftrl"] == FTRL,
        "ftrl_parametros": (ALPHA == 0.05 and LAMBDA1 == 0.0 and LAMBDA2 == 1.0 and BETA == 1.0
                            and X0 == 1.0 and EPOCAS == 10 and FTRL["bias"] is True
                            and FTRL["power_t"] is None
                            and "10 pasadas fijas" in FTRL["criterio_parada"]),
        "ftrl_barajado": FTRL["barajado"]
                         == f"np.random.default_rng({SEMILLA}+fold) sobre los indices de "
                            "fold-train por epoca",
        "ftrl_alternativa_no_usada": FTRL["alternativa_no_usada"].startswith("vowpalwabbit")
                                     and "NO usado" in FTRL["alternativa_no_usada"],
        "ftrl_combinador_p": "normalizacion L2 por fila" in FTRL["combinador_p"],
        "a2_sgd_twin": prereg["a2_sgd_twin"] == SGD_TWIN,
        "a2_parametros": (SGD_ALPHA == 1e-06 and SGD_MAX_ITER == 10 and SGD_TWIN["tol"] is None
                          and SGD_TWIN["fit_intercept"] is True
                          and SGD_TWIN["average"] is False
                          and SGD_TWIN["learning_rate"] == "optimal"
                          and SGD_TWIN["loss"] == "log_loss"
                          and SGD_TWIN["estimator"] == "sklearn.linear_model.SGDClassifier"),
        "a2_random_state": SGD_TWIN["random_state"] == f"{SEMILLA}+fold",
        "a2_n_jobs": SGD_TWIN["n_jobs"] == HILOS,
        "a3_control": prereg["a3_control_negativo"] == A3_CONTROL,
        "a3_semilla": (SEMILLA_BARAJADO == 20261011
                       and f"np.random.default_rng({SEMILLA_BARAJADO}+fold)"
                       in A3_CONTROL["descripcion"]),
        "sondas": prereg["sondas"] == SONDAS_PREREG,
        "sondas_tolerancias": (TOL_SONDA_A == 1e-12 and TOL_SONDA_B == 1e-9
                               and TOL_SONDA_C == 1e-6
                               and SONDAS_PREREG["d_sin_nan_logloss_finito"] is True
                               and DIM_SONDA_A == (200, 50, 6)),
        "sondas_politica": "exit 4 ANTES de entrenar" in SONDAS_PREREG["politica"],
        "fallback": prereg["fallback"] == FALLBACK_PREREG,
        "fallback_criterio": (FALLBACK_FACTOR == 0.8
                              and "<= 0.8 * log_loss tras la pasada 1"
                              in FALLBACK_PREREG["criterio"]),
        "fallback_alternativa": (ALPHA_ALTERNATIVO == 0.005
                                 and "alpha=0.005" in FALLBACK_PREREG["si_falla"]
                                 and "lambda2=1.0" in FALLBACK_PREREG["si_falla"]),
        "determinismo": prereg["determinismo"] == DETERMINISMO_PREREG,
        "determinismo_seeds": (DETERMINISMO_PREREG["seeds"]["global"] == SEMILLA
                               and DETERMINISMO_PREREG["seeds"]["barajado_etiquetas"]
                               == SEMILLA_BARAJADO
                               and DETERMINISMO_PREREG["seeds"]["orden_datos"]
                               == f"default_rng({SEMILLA}+fold) por epoca"),
        "determinismo_tolerancias": (TOL_DETERMINISMO == 1e-9 and VARIABILIDAD == 0.0008876
                                     and DETERMINISMO_PREREG["tolerancias"]
                                     == "0 exacto; <=1e-9 conforme; >1e-9 declarar rango y "
                                        "exigir delta_gate > rango; rango >= 0.0008876 o "
                                        "solape con el gate => no concluyente"),
        "hilos_efectivos": all(os.environ[clave] == str(HILOS) for clave in
                               ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")),
        "integracion": prereg["integracion"] == INTEGRACION_PREREG,
        "integracion_parametros": (C_LOGREG == 1.0 and MAX_ITER_LOGREG == 2000
                                   and L2_CLIP == 1e-12 and PESOS[0] == 1.0),
        "gates": prereg["gates"] == GATES_PREREG,
        "gates_parametros": (float(GATES_PREREG["baseline_abort"]) == ABORTO_BASELINE
                             and float(GATES_PREREG["delta_min"]) == GATE_DELTA
                             and int(GATES_PREREG["folds_min"]) == GATE_FOLDS
                             and GATES_PREREG["sin_leakage"] is True
                             and GATES_PREREG["determinismo_concluyente"] is True),
        "gates_bootstrap": GATES_PREREG["bootstrap"] == {"n": N_BOOTSTRAP, "semilla": SEMILLA,
                                                          "ic95_sin_cero": True},
        "gates_control": GATES_PREREG["control"] == (
            f"delta_int(A1) > delta_int(A2) Y bootstrap pareado de la diferencia "
            f"(F1_combA1 - F1_combA2; {N_BOOTSTRAP}; semilla {SEMILLA}) con IC95 sin cero"),
        "cap_ic95": ("capado a MARGINAL" in GATES_PREREG["cap_ic95_diferencia"]
                     and "puede degradar un resultado genuinamente bueno"
                     in GATES_PREREG["cap_ic95_diferencia"]),
        "cap_determinismo": GATES_PREREG["cap_determinismo"]
                            == "sonda no concluyente -> el veredicto no puede ser PROMISING",
        "a3_gate": GATES_PREREG["a3_gate"] == A3_CONTROL["umbral_fuga"],
        "arbol": prereg["arbol"] == ARBOL_PREREG,
        "arbol_semilla_2": f"semilla {SEMILLA_SEGUNDA_PARTICION}" in ARBOL_PREREG["PROMISING"],
        "ejecucion": prereg["ejecucion"] == EJECUCION_PREREG,
        "ejecucion_ram": (float(EJECUCION_PREREG["ram_tope_gb"]) == RAM_TOPE_GB
                          and "ru_maxrss" in EJECUCION_PREREG["ram_medicion"]
                          and "max_features" in EJECUCION_PREREG["ram_medicion"]),
        "ejecucion_calibracion": (float(PROYECCION_FACTOR) == 9.0
                                  and float(PROYECCION_MAX_H) == 20.0
                                  and "9 x t_fold0" in EJECUCION_PREREG["calibracion"]),
        "ejecucion_timeout": (float(TIMEOUT_H) == 24.0
                              and EJECUCION_PREREG["timeout_global"] == "24.0 h"),
        "ejecucion_checkpoints": ruta_relativa(ESTADO_JSON) in EJECUCION_PREREG["checkpoints"],
        "ejecucion_atomica": ruta_relativa(OOF_DIR) + "/" in EJECUCION_PREREG[
            "escritura_atomica"],
        "vectorizador_config": prereg["representacion"]["vectorizador"] == VECTORIZADOR,
        "vectorizador_ajuste_por_fold": prereg["representacion"]["ajuste"].startswith(
            "POR FOLD"),
        "vectorizador_hash_forma": json.dumps(VECTORIZADOR, sort_keys=True) == json.dumps(
            prereg["representacion"]["vectorizador"], sort_keys=True),
        "corpus_receta": ("load_train" in prereg["corpus"]["receta"]
                          and "utf-8-sig" in prereg["corpus"]["receta"]),
        "corpus_hash_contenido": prereg["corpus"]["hash_contenido"] == (
            "sha256 con longitud (8 bytes LE) + bytes UTF-8 por texto, en orden"),
        "corpus_desvio_manifest": (
            prereg["corpus"]["desvio_manifest"]["decision"].startswith("no se reordena nada")
            and prereg["corpus"]["desvio_manifest"]["data_hash_manifest"]
            == "2ada573b27b9cb503cda0bb80344b092b9601a3105fe9b56d2127e261f2edf8e"),
        "compliance_cifras": {clave: prereg["compliance"][clave] for clave in COMPLIANCE_CIFRAS}
                            == COMPLIANCE_CIFRAS,
        "compliance_auditorias": prereg["compliance"]["auditorias_exigidas"]
                                 == COMPLIANCE_AUDITORIAS,
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

    Los cinco canonicos se leen de disco; los tres del corpus (archivo, array de textos con
    la receta determinista y dict del vectorizador) se recomputan del corpus cargado.

    Parameters
    ----------
    prereg:
        Contenido del pre-registro.
    hashes_corpus:
        Hashes recomputados del corpus.

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


def cargar_canonico() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Carga y valida la particion canonica (y_true, fold_idx, v30_prob, ids).

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]
        Etiquetas int64, folds int64, probabilidades del campeon normalizadas e ids int64.

    Raises
    ------
    Detencion
        Si los artefactos no tienen la alineacion congelada (169336 filas y 5 folds).
    """
    clase = np.load(RUTAS_HASHES["canonico_y_true"])
    fold_idx = np.load(RUTAS_HASHES["canonico_fold_idx"])
    v30 = np.load(RUTAS_HASHES["canonico_v30_prob"]).astype(np.float64)
    ids = np.load(RUTAS_HASHES["canonico_ids"])
    if clase.shape != (N_FILAS,) or fold_idx.shape != (N_FILAS,) or ids.shape != (N_FILAS,):
        raise Detencion(f"[STOP] los artefactos canonicos no tienen {N_FILAS} filas (exit 2)", 2)
    if v30.shape != (N_FILAS, N_CLASES):
        raise Detencion("[STOP] v30_prob con forma inesperada (exit 2)", 2)
    if int(clase.min()) != 0 or int(clase.max()) != N_CLASES - 1:
        raise Detencion("[STOP] clases fuera del rango canonico (exit 2)", 2)
    if np.unique(fold_idx).size != N_FOLDS:
        raise Detencion(f"[STOP] se esperan {N_FOLDS} folds canonicos (exit 2)", 2)
    return (clase.astype(np.int64), fold_idx.astype(np.int64), normaliza(v30),
            ids.astype(np.int64))


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
        Textos en el orden del CSV y los tres hashes recomputados del corpus.

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
        raise Detencion("[STOP] cat.codes del corpus no coincide con y_true canonico (exit 2)",
                        2)
    hashes = {"corpus_archivo": sha256_archivo(cfg.TRAIN_FILE),
              "corpus_textos": sha256_textos(textos),
              "vectorizador": sha256_bytes(json.dumps(VECTORIZADOR, sort_keys=True)
                                           .encode("utf-8"))}
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


def log_loss_ovr(p_ovr: np.ndarray, y: np.ndarray) -> float:
    """log_loss OvR medio: media sobre las 6 clases del log-loss binario con sigmoid.

    Las probabilidades OvR de FTRL no suman 1; por eso el pre-registro define la magnitud de
    la regla de fallback como la media de los seis log-loss binarios (no con log_loss de
    sklearn, que exige filas que sumen 1).

    Parameters
    ----------
    p_ovr:
        Probabilidades OvR (n, 6), sin normalizar.
    y:
        Etiquetas enteras en 0..5.

    Returns
    -------
    float
        Log-loss OvR medio (finito siempre gracias al clip).
    """
    objetivo = np.zeros((len(y), N_CLASES), dtype=np.float64)
    objetivo[np.arange(len(y)), y] = 1.0
    recortado = np.clip(p_ovr, L2_CLIP, 1.0 - L2_CLIP)
    perdida = -(objetivo * np.log(recortado) + (1.0 - objetivo) * np.log1p(-recortado))
    return float(np.mean(perdida.sum(axis=1)) / N_CLASES)


def log_loss_multiclase(p: np.ndarray, y: np.ndarray) -> float:
    """log_loss multiclase de sklearn sobre probabilidades RENORMALIZADAS con la convencion
    del proyecto (clip [1e-12, 1] y L1 por fila).

    Es obligatorio renormalizar antes de llamar a ``sklearn.metrics.log_loss``: en sklearn
    1.8.0 esa funcion solo avisa ("The y_prob values do not sum to one") y **no** renormaliza
    por dentro, de modo que sin este paso devolveria una perdida calculada sobre valores que
    no son probabilidades (demasiado baja). La renormalizacion es la misma que aplica
    ``combinador_oof``, por lo que la metrica es comparable entre brazos.

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


def linea_base_mayoritaria(clase: np.ndarray) -> dict:
    """Linea base de la clase mayoritaria (umbral del gate A3 del pre-registro).

    Se obtiene prediciendo la clase mas frecuente del corpus canonico en todas las filas y
    midiendo su macro-F1; se registra como evidencia, no se ajusta.

    Parameters
    ----------
    clase:
        Etiquetas canonicas (169336,).

    Returns
    -------
    dict
        Clase mayoritaria, su frecuencia, el macro-F1 de esa linea base y las filas.
    """
    conteos = np.bincount(clase, minlength=N_CLASES)
    mayoritaria = int(np.argmax(conteos))
    pred = np.full(len(clase), mayoritaria, dtype=np.int64)
    return {"clase_indice": mayoritaria, "clase": cfg.CLASSES[mayoritaria],
            "frecuencia": float(conteos[mayoritaria] / len(clase)),
            "macro_f1_linea_base": macro(clase, pred), "n_filas": len(clase)}


# =====================================================================================
# ---- 2. AUDITORIA AST DE COMPLIANCE DEL PROPIO SCRIPT ----
# =====================================================================================
def auditar_ast_propio() -> dict:
    """Audita el propio codigo fuente: 0 palabras vigiladas en literales de ruta.

    Un literal se considera de ruta si contiene barra inclinada o termina en extension de
    datos (.csv, .npy, .json, .npz, .txt). Ninguno de esos literales puede contener, en
    minusculas, ninguna de las palabras vigiladas del pre-registro. Si aparece una, el
    runner se detiene. Ademas se revisan los MODULOS IMPORTADOS contra las familias vetadas
    (vowpal, vw, torch, transformers, gensim, fasttext), de modo que la ausencia de las
    familias vetadas no dependa solo de los literales de ruta.

    Returns
    -------
    dict
        ``{"ast_audit": True, "literales_de_ruta_revisados": n, "importaciones_revisadas": m,
        "violaciones": []}``.

    Raises
    ------
    Detencion
        Si hay algun literal de ruta con palabra vigilada o algun import de una familia
        vetada (codigo 2).
    """
    fuente = Path(__file__).read_text(encoding="utf-8")
    arbol = ast.parse(fuente)
    literales: list[tuple[int, str]] = []
    for nodo in ast.walk(arbol):
        if isinstance(nodo, ast.Constant) and isinstance(nodo.value, str):
            texto = nodo.value
            es_ruta = ("/" in texto) or texto.lower().endswith((".csv", ".npy", ".json", ".npz",
                                                                 ".txt"))
            if es_ruta:
                literales.append((int(nodo.lineno), texto))
    violaciones = [(linea, texto) for linea, texto in literales
                   if any(palabra in texto.lower() for palabra in PALABRAS_VIGILADAS)]
    modulos: list[tuple[int, str]] = []
    for nodo in ast.walk(arbol):
        if isinstance(nodo, ast.Import):
            modulos += [(int(nodo.lineno), alias.name) for alias in nodo.names]
        elif isinstance(nodo, ast.ImportFrom) and nodo.module:
            modulos.append((int(nodo.lineno), nodo.module))
    importaciones_vetadas = [(linea, modulo) for linea, modulo in modulos
                            if any(familia in modulo.lower() for familia in FAMILIAS_VETADAS)]
    for linea, texto in violaciones:
        print(f"  [STOP] literal de ruta con palabra vigilada ({texto}) en la linea "
              f"{linea}", flush=True)
    for linea, modulo in importaciones_vetadas:
        print(f"  [STOP] import de una familia vetada ({modulo}) en la linea {linea}",
              flush=True)
    if violaciones or importaciones_vetadas:
        raise Detencion("[STOP] auditoria AST de compliance fallida (exit 2)", 2)
    return {"ast_audit": True, "literales_de_ruta_revisados": len(literales),
            "importaciones_revisadas": len(modulos), "familias_vetadas": list(FAMILIAS_VETADAS),
            "violaciones": []}


# =====================================================================================
# ---- 3. SONDAS OBLIGATORIAS (a)-(d): FTRL PROPIO Y METRICAS ----
# =====================================================================================
def sigmoid(valores: np.ndarray) -> np.ndarray:
    """Sigmoid numericamente estable (evita overflow en ``exp`` con logit muy grande).

    Parameters
    ----------
    valores:
        Logit(s) de cualquier forma.

    Returns
    -------
    numpy.ndarray
        Probabilidades en (0, 1) con la misma forma.
    """
    valores = np.asarray(valores, dtype=np.float64)
    salida = np.empty_like(valores)
    positivo = valores >= 0.0
    salida[positivo] = 1.0 / (1.0 + np.exp(-valores[positivo]))
    exponencial = np.exp(valores[~positivo])
    salida[~positivo] = exponencial / (1.0 + exponencial)
    return salida


def ftrl_ovr(x: sp.csr_matrix, y: np.ndarray, alpha: float, lambda1: float, lambda2: float,
             beta: float, epocas: int, semilla: int,
             al_terminar_epoca: Callable[[int, np.ndarray], None] | None = None,
             con_estado: bool = False) -> tuple[np.ndarray, dict | None]:
    """FTRL-Proximal propio one-vs-rest con la actualizacion exacta del pre-registro.

    Vectorizacion: el bucle es secuencial por ejemplo (FTRL es un metodo online) y la
    actualizacion se vectoriza sobre las 6 clases con los indices de la fila CSR. Para cada
    ejemplo y clase el gradiente es ``g = (sigmoid(w.x) - y) * x`` y la coordenada se
    actualiza con ``sigma = (sqrt(n+g^2)-sqrt(n))/(alpha*beta); z += g - sigma*w; n += g^2;
    w = 0 si |z| <= lambda1; si no sign(z)*(|z|-lambda1)/((beta+sqrt(n))/alpha+lambda2)``.

    Parameters
    ----------
    x:
        Matriz CSR float64 con la coordenada del sesgo ya anadida (ultima columna, x0).
    y:
        Etiquetas enteras en 0..5 (n,).
    alpha, lambda1, lambda2, beta:
        Hiperparametros congelados de FTRL.
    epocas:
        Numero de pasadas fijas (10).
    semilla:
        Semilla del generador de barajado de indices por epoca.
    al_terminar_epoca:
        Callback opcional ``(epoca, w)`` al final de cada pasada, para registrar curvas.
    con_estado:
        Si True devuelve tambien los acumuladores ``z`` y ``n`` (sonda analitica).

    Returns
    -------
    tuple[numpy.ndarray, dict | None]
        Pesos (6, n_coordenadas) y, si se pidio, el estado interno con ``z`` y ``n``.
    """
    n_filas, n_coordenadas = x.shape
    w = np.zeros((N_CLASES, n_coordenadas), dtype=np.float64)
    z = np.zeros((N_CLASES, n_coordenadas), dtype=np.float64)
    n = np.zeros((N_CLASES, n_coordenadas), dtype=np.float64)
    objetivo = np.zeros((n_filas, N_CLASES), dtype=np.float64)
    objetivo[np.arange(n_filas), y] = 1.0
    generador = np.random.default_rng(semilla)
    for epoca in range(epocas):
        for i in generador.permutation(n_filas):
            inicio, fin = int(x.indptr[i]), int(x.indptr[i + 1])
            indices = x.indices[inicio:fin]
            valores = x.data[inicio:fin]
            p = sigmoid(w[:, indices] @ valores)
            g = (p - objetivo[i])[:, None] * valores[None, :]
            n_previo = n[:, indices]
            n_nuevo = n_previo + g * g
            sigma = (np.sqrt(n_nuevo) - np.sqrt(n_previo)) / (alpha * beta)
            z[:, indices] = z[:, indices] + g - sigma * w[:, indices]
            n[:, indices] = n_nuevo
            z_nuevo = z[:, indices]
            w[:, indices] = np.where(
                np.abs(z_nuevo) <= lambda1, 0.0,
                np.sign(z_nuevo) * (np.abs(z_nuevo) - lambda1)
                / ((beta + np.sqrt(n_nuevo)) / alpha + lambda2))
        if al_terminar_epoca is not None:
            al_terminar_epoca(epoca, w)
    if con_estado:
        return w, {"z": z, "n": n}
    return w, None


def ftrl_lazo_ingenuo(x_denso: np.ndarray, objetivo: np.ndarray, alpha: float, lambda1: float,
                      lambda2: float, beta: float, epocas: int, semilla: int) -> np.ndarray:
    """Lazo ingenuo de FTRL para UNA clase (listas de Python, sin numpy en el bucle interno).

    Es la referencia independiente de la sonda (a): misma actualizacion y misma permutacion
    por epoca, pero recorriendo una coordenada por vez con escalares de Python.

    Parameters
    ----------
    x_denso:
        Matriz densa float64 (n, d) de una sola clase (coordenada del sesgo incluida).
    objetivo:
        Objetivo binario float64 (n,) en {0.0, 1.0}.
    alpha, lambda1, lambda2, beta:
        Hiperparametros congelados de FTRL.
    epocas:
        Numero de pasadas.
    semilla:
        Semilla del barajado (mismo stream que la version vectorizada).

    Returns
    -------
    numpy.ndarray
        Vector de pesos (d,) al terminar las pasadas.
    """
    n_filas, n_coordenadas = x_denso.shape
    w = [0.0] * n_coordenadas
    z = [0.0] * n_coordenadas
    n = [0.0] * n_coordenadas
    generador = np.random.default_rng(semilla)
    for _ in range(epocas):
        for i in generador.permutation(n_filas):
            fila = x_denso[i]
            suma = 0.0
            for j in range(n_coordenadas):
                suma += w[j] * fila[j]
            p = (1.0 / (1.0 + math.exp(-suma)) if suma >= 0.0
                 else math.exp(suma) / (1.0 + math.exp(suma)))
            error = p - objetivo[i]
            for j in range(n_coordenadas):
                g = error * fila[j]
                raiz_previa = math.sqrt(n[j])
                raiz = math.sqrt(n[j] + g * g)
                sigma = (raiz - raiz_previa) / (alpha * beta)
                z[j] += g - sigma * w[j]
                n[j] += g * g
                w[j] = (0.0 if abs(z[j]) <= lambda1 else
                        math.copysign(1.0, z[j]) * (abs(z[j]) - lambda1)
                        / ((beta + math.sqrt(n[j])) / alpha + lambda2))
    return np.array(w, dtype=np.float64)


def problema_aleatorio(n_filas: int, n_features: int, semilla: int
                       ) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray]:
    """Problema aleatorio denso (n, d) con sesgo, empaquetado en CSR canonico.

    Se usa en las sondas (a), (c) y (d) con las dimensiones congeladas.

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
        Matriz CSR con sesgo, clases y matriz objetivo one-hot.
    """
    generador = np.random.default_rng(semilla)
    denso = generador.random((n_filas, n_features + 1))
    denso[:, -1] = X0
    y = generador.integers(0, N_CLASES, size=n_filas)
    objetivo = np.zeros((n_filas, N_CLASES), dtype=np.float64)
    objetivo[np.arange(n_filas), y] = 1.0
    return sp.csr_matrix(denso), y, objetivo


def sonda_a_vectorizado_vs_lazo() -> dict:
    """Sonda (a): la version vectorizada de FTRL coincide con el lazo ingenuo.

    Problema aleatorio de 200 filas x 50 features x 6 clases (dimension congelada), misma
    semilla de barajado y mismas 10 pasadas; se compara el maximo de la diferencia absoluta
    de los pesos de las 6 clases.

    Returns
    -------
    dict
        Dimensiones, maximo de la diferencia, tolerancia y ``pass``.
    """
    n_filas, n_features, n_clases = DIM_SONDA_A
    x, y, objetivo = problema_aleatorio(n_filas, n_features, 7)
    denso = np.asarray(x.todense(), dtype=np.float64)
    w_vector, _ = ftrl_ovr(x, y, ALPHA, LAMBDA1, LAMBDA2, BETA, EPOCAS, SEMILLA)
    peor = 0.0
    for clase in range(n_clases):
        w_lazo = ftrl_lazo_ingenuo(denso, objetivo[:, clase], ALPHA, LAMBDA1, LAMBDA2, BETA,
                                   EPOCAS, SEMILLA)
        peor = max(peor, float(np.abs(w_lazo - w_vector[clase]).max()))
    return {"descripcion": f"vectorizado (sobre 6 clases) vs lazo ingenuo, problema aleatorio "
                           f"{n_filas}x{n_features}x{n_clases}",
            "max_abs_diff": peor, "tolerancia": TOL_SONDA_A, "pass": bool(peor <= TOL_SONDA_A)}


def sonda_b_punto_fijo_analitico() -> dict:
    """Sonda (b): con lambda1=lambda2=0 el peso es la raiz analitica de FTRL.

    Con las dos penalizaciones a cero, el peso de cada coordenada es el minimo analitico de
    la cuadratica proximal ``(1/2)((beta+sqrt(n))/alpha) w^2 - z w``, es decir
    ``w = alpha*z/(beta+sqrt(n))``. Se comprueba con dos residuales exactos: el residuo de
    esa raiz sobre el estado final (feature constante repetida) y el valor de un solo paso
    (``g = sigmoid(0) - 1 = -0.5`` -> ``w = alpha*g/(beta+|g|)``).

    Returns
    -------
    dict
        Residuos de la raiz analitica y del paso unico, umbral y ``pass``.
    """
    constante = sp.csr_matrix(np.ones((25, 1), dtype=np.float64))
    y = np.zeros(25, dtype=np.int64)
    w, estado = ftrl_ovr(constante, y, ALPHA, 0.0, 0.0, BETA, 3, 11, con_estado=True)
    raiz = estado["z"][0] * ALPHA / (BETA + np.sqrt(estado["n"][0]))
    residuo_raiz = float(np.abs(w[0] - raiz).max())
    w_uno, _ = ftrl_ovr(sp.csr_matrix(np.ones((1, 1))), np.zeros(1, dtype=np.int64), ALPHA,
                        0.0, 0.0, BETA, 1, 11)
    gradiente = 0.5 - 1.0
    esperado = ALPHA * gradiente / (BETA + abs(gradiente))
    residuo_paso = float(abs(w_uno[0, 0] - esperado))
    return {"descripcion": "lambda1=lambda2=0 con feature constante: raiz analitica "
                           "alpha*z/(beta+sqrt(n)) y paso unico alpha*g/(beta+|g|)",
            "residuo_raiz": residuo_raiz, "residuo_un_paso": residuo_paso,
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
            "delta_criterio": delta,
            "delta_default_vs_macro": abs(total_default - macro_sklearn),
            "tolerancia": TOL_SONDA_C, "pass": bool(delta <= TOL_SONDA_C)}


def sonda_d_sin_nan_logloss_finito() -> dict:
    """Sonda (d): las p OvR no tienen NaN/inf y el log_loss OvR es finito.

    Returns
    -------
    dict
        ``p_finita``, ``log_loss_finito``, valores y ``pass``.
    """
    x, y, _ = problema_aleatorio(400, 20, 13)
    w, _ = ftrl_ovr(x, y, ALPHA, LAMBDA1, LAMBDA2, BETA, 3, SEMILLA)
    p = sigmoid(x.dot(w.T))
    valor = log_loss_ovr(p, y)
    finita = bool(np.isfinite(p).all() and math.isfinite(valor))
    return {"descripcion": "p OvR sin NaN/inf y log_loss OvR medio finito",
            "n_filas": int(x.shape[0]), "p_finita": bool(np.isfinite(p).all()),
            "log_loss_ovr": valor, "log_loss_finito": bool(math.isfinite(valor)),
            "p_min": float(p.min()), "p_max": float(p.max()), "pass": finita}


def ejecutar_sondas() -> dict:
    """Ejecuta las sondas obligatorias (a)-(d) antes de cualquier ajuste.

    Returns
    -------
    dict
        Detalle de las cuatro sondas con valores, tolerancias y ``todas_pasan``.

    Raises
    ------
    Detencion
        Si alguna sonda falla (codigo 4), tal como fija la politica del pre-registro.
    """
    inicio = time.time()
    sondas = {"a_vectorizado_vs_lazo": sonda_a_vectorizado_vs_lazo(),
              "b_punto_fijo_analitico": sonda_b_punto_fijo_analitico(),
              "c_totalf1_macro": sonda_c_totalf1_macro(),
              "d_sin_nan_logloss_finito": sonda_d_sin_nan_logloss_finito()}
    detalle = dict(sondas)
    detalle["todas_pasan"] = bool(all(sondas[nombre]["pass"] for nombre in sondas))
    detalle["politica"] = SONDAS_PREREG["politica"]
    detalle["duracion_seg"] = float(time.time() - inicio)
    for nombre in ("a_vectorizado_vs_lazo", "b_punto_fijo_analitico", "c_totalf1_macro",
                   "d_sin_nan_logloss_finito"):
        print(f"  sonda ({nombre[0]}) {nombre}: pass={sondas[nombre]['pass']}", flush=True)
    if not detalle["todas_pasan"]:
        fallidas = [nombre for nombre in sondas if not sondas[nombre]["pass"]]
        raise Detencion(f"[STOP] sondas obligatorias fallidas {fallidas} (exit 4)", 4)
    return detalle


# =====================================================================================
# ---- 4. ENTRENAMIENTO POR BRAZOS CON CHECKPOINTS RESUMIBLES ----
# =====================================================================================
def guardar_oof(brazo: str, oof: np.ndarray) -> str:
    """Persiste el OOF del brazo con escritura atomica (temporal + os.replace).

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


def guardar_oof_crudo(brazo: str, oof: np.ndarray) -> str:
    """Persiste el OOF crudo (diagnostico de A2) con escritura atomica.

    Parameters
    ----------
    brazo:
        Nombre del brazo.
    oof:
        Matriz float64 (169336, 6) con las seis sigmoides sin normalizar.

    Returns
    -------
    str
        Ruta relativa del artefacto escrito.
    """
    OOF_DIR.mkdir(parents=True, exist_ok=True)
    ruta = ruta_oof_crudo(brazo)
    temporal = ruta.with_name(ruta.name + ".tmp")
    with temporal.open("wb") as manejador:
        np.save(manejador, oof)
    os.replace(temporal, ruta)
    return ruta_relativa(ruta)


def estado_nuevo() -> dict:
    """Estado resumible inicial del checkpoint.

    Returns
    -------
    dict
        Checkpoint vacio con inicio, guardas, vocabulario por fold y los tres brazos.
    """
    return {
        "fase": FASE_ESTADO,
        "inicio_ts": float(time.time()),
        "timeout_global_h": TIMEOUT_H,
        "ram_tope_gb": RAM_TOPE_GB,
        "rss_pico_gb": 0.0,
        "sondas": None,
        "fallback": {"evaluado": False, "criterio": FALLBACK_PREREG["criterio"],
                     "factor": FALLBACK_FACTOR, "alpha_inicial": ALPHA,
                     "alpha_efectivo": ALPHA, "config_fallback_usada": False,
                     "log_loss_train_epoca1": None, "log_loss_train_epoca10": None,
                     "finito": None, "cumple": None, "reinicios": []},
        "calibracion": None,
        "vectorizador_por_fold": {},
        "brazos": {brazo: {"folds_completados": [], "folds": {},
                           "mascara_relleno_por_fila": [0] * N_FILAS,
                           "alpha_efectivo": None if brazo == "A2_sgd_twin" else ALPHA,
                           "sonda_determinismo": None} for brazo in BRAZOS},
    }


def guardar_estado(estado: dict) -> None:
    """Persiste el checkpoint con escritura atomica (temporal + os.replace).

    Parameters
    ----------
    estado:
        Estado a persistir.
    """
    ESTADO_JSON.parent.mkdir(parents=True, exist_ok=True)
    temporal = ESTADO_JSON.with_name(ESTADO_JSON.name + ".tmp")
    temporal.write_text(json.dumps(como_nativo(estado), indent=2, ensure_ascii=True) + "\n",
                        encoding="utf-8")
    os.replace(temporal, ESTADO_JSON)


def cargar_estado() -> dict:
    """Carga el checkpoint resumible o lo crea si no existe.

    Returns
    -------
    dict
        Estado con folds completados, curvas, mascara de relleno y guardas.

    Raises
    ------
    Detencion
        Si el checkpoint existe pero no corresponde a V32-E3 o esta incompleto (codigo 2).
    """
    if not ESTADO_JSON.exists():
        estado = estado_nuevo()
        guardar_estado(estado)
        return estado
    estado = json.loads(ESTADO_JSON.read_text(encoding="utf-8"))
    if estado.get("fase") != FASE_ESTADO:
        raise Detencion(f"[STOP] el checkpoint no corresponde a {FASE_ESTADO} (exit 2)", 2)
    for clave in ("inicio_ts", "rss_pico_gb", "fallback", "calibracion",
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
            raise Detencion(f"[STOP] folds completados y detalle incoherentes en {brazo} "
                            "(exit 2)", 2)
        for fold, detalle in datos.get("folds", {}).items():
            faltan = [campo for campo in CAMPOS_FOLD if campo not in detalle]
            if faltan:
                raise Detencion(f"[STOP] checkpoint sin campos {faltan} en {brazo}/{fold} "
                                "(exit 2)", 2)
    if estado["fallback"].get("evaluado") and estado["fallback"].get(
            "alpha_efectivo") not in (ALPHA, ALPHA_ALTERNATIVO):
        raise Detencion("[STOP] el checkpoint registra un alpha de fallback desconocido "
                        "(exit 2)", 2)
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
            raise Detencion(f"[STOP] el checkpoint dice que {brazo} tiene folds completados y "
                            "no existe su OOF (exit 2)", 2)
        return np.zeros((N_FILAS, N_CLASES), dtype=np.float64)
    with ruta.open("rb") as manejador:
        oof = np.load(manejador)
    if oof.shape != (N_FILAS, N_CLASES) or oof.dtype != np.float64:
        raise Detencion(f"[STOP] OOF de {brazo} con forma/tipo inesperado (exit 2)", 2)
    relleno = (oof.sum(axis=1) > 0.0).astype(np.int64)
    if not np.array_equal(relleno, mascara_de_folds(fold_idx, completados)):
        raise Detencion(f"[STOP] el OOF de {brazo} no coincide con los folds completados "
                        "(exit 2)", 2)
    if not np.isfinite(oof).all():
        raise Detencion(f"[STOP] el OOF de {brazo} contiene no finitos (exit 2)", 2)
    return oof.astype(np.float64)


def anadir_sesgo(x: sp.csr_matrix, x0: float = X0) -> sp.csr_matrix:
    """Anade la coordenada del sesgo (ultima columna, valor constante ``x0``) a un CSR.

    El sesgo es una coordenada MAS de FTRL, con su propio ``n`` y su propio ``z``: por eso
    se materializa como columna y no como termino aparte. Se anade explicitamente en todas
    las filas (incluidas las que quedaron sin terminos tras ``min_df``/``max_df``) y el
    resultado queda en formato CSR canonico con indices ascendentes.

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
    filas = np.concatenate([np.repeat(np.arange(n_filas, dtype=np.int64),
                                      np.diff(x.indptr)),
                            np.arange(n_filas, dtype=np.int64)])
    columnas = np.concatenate([x.indices.astype(np.int64),
                               np.full(n_filas, n_features, dtype=np.int64)])
    valores = np.concatenate([x.data.astype(np.float64), np.full(n_filas, x0)])
    matriz = sp.csr_matrix((valores, (filas, columnas)), shape=(n_filas, n_features + 1))
    assert matriz.has_sorted_indices, "la matriz con sesgo debe tener indices ascendentes"
    assert matriz.has_canonical_format, "la matriz con sesgo debe estar en formato canonico"
    return matriz


def ajustar_vectorizador_fold(textos: np.ndarray, tr_idx: np.ndarray, va_idx: np.ndarray
                             ) -> dict:
    """Ajusta el TF-IDF SOLO en las filas de fold-train y transforma el fold-val.

    Los ``min_df``/``max_df``/``idf`` nunca son globales: se recalculan en cada fold, que es
    el precedente canonico del manifiesto. Se registran los hashes del vocabulario ordenado y
    del idf para poder verificar cualquier reentreno posterior.

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


def etiquetas_objetivo(brazo: str, clase: np.ndarray, fold_idx: np.ndarray,
                       fold: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Indices del fold y etiquetas objetivo del fold-train (permutadas solo en A3).

    A3 permuta las etiquetas del fold-train con ``default_rng(20261011 + fold)``; el fold-val
    conserva las etiquetas reales (solo para el OOF). El fold-val jamas entra al ajuste.

    Parameters
    ----------
    brazo:
        Nombre del brazo.
    clase:
        Etiquetas canonicas.
    fold_idx:
        Fold canonico por fila.
    fold:
        Fold canonico.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
        ``tr_idx``, ``va_idx`` y ``y_tr``.
    """
    tr_idx, va_idx = indices_fold(fold_idx, fold)
    y_tr = clase[tr_idx].astype(np.int64)
    if brazo == "A3_ftrl_barajado":
        generador = np.random.default_rng(SEMILLA_BARAJADO + fold)
        y_tr = y_tr[generador.permutation(tr_idx.size)]
    return tr_idx, va_idx, y_tr


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
    return {"loss": SGD_TWIN["loss"], "alpha": SGD_ALPHA, "max_iter": SGD_MAX_ITER,
            "tol": SGD_TWIN["tol"], "fit_intercept": SGD_TWIN["fit_intercept"],
            "average": SGD_TWIN["average"], "learning_rate": SGD_TWIN["learning_rate"],
            "random_state": SEMILLA + fold, "n_jobs": SGD_TWIN["n_jobs"]}


def entrenar_fold_ftrl(x_tr: sp.csr_matrix, y_tr: np.ndarray, x_va: sp.csr_matrix,
                       clase_va: np.ndarray, alpha: float, fold: int) -> dict:
    """Entrena un fold de A1/A3 con FTRL-Proximal y devuelve el OOF crudo del fold-val.

    Registra por epoca el log_loss OvR medio de las filas de fold-train (magnitud de la regla
    de fallback) y el del fold-val (solo reporte, nunca decision).

    Parameters
    ----------
    x_tr:
        Fold-train con sesgo (CSR float64).
    y_tr:
        Etiquetas objetivo del fold-train (permutadas en A3).
    x_va:
        Fold-val con sesgo (CSR float64).
    clase_va:
        Etiquetas reales del fold-val (solo reporte).
    alpha:
        Alpha efectivo del fold (0.05 o la alternativa declarada 0.005).
    fold:
        Fold canonico (solo para la semilla del barajado).

    Returns
    -------
    dict
        ``p_crudo`` (n_val, 6), curvas de log-loss por epoca, tiempo, alpha y n_features.
    """
    inicio = time.time()
    curvas_train: list[float] = []
    instantaneas: list[np.ndarray] = []

    def registrar(_epoca: int, w: np.ndarray) -> None:
        """Registra el log_loss OvR medio de fold-train y guarda el estado de la epoca."""
        curvas_train.append(log_loss_ovr(sigmoid(x_tr.dot(w.T)), y_tr))
        instantaneas.append(w.copy())

    w, _ = ftrl_ovr(x_tr, y_tr, alpha, LAMBDA1, LAMBDA2, BETA, EPOCAS, SEMILLA + fold,
                    al_terminar_epoca=registrar)
    curvas_val = [log_loss_ovr(sigmoid(x_va.dot(w_epoca.T)), clase_va)
                  for w_epoca in instantaneas]
    p_crudo = sigmoid(x_va.dot(w.T))
    return {"p_crudo": p_crudo, "curvas_train": curvas_train, "curvas_val": curvas_val,
            "tiempo_seg": float(time.time() - inicio), "alpha": alpha,
            "n_features": int(x_tr.shape[1]), "n_iter_sgd": None}


def entrenar_fold_sgd(x_tr: sp.csr_matrix, y_tr: np.ndarray, x_va: sp.csr_matrix,
                      clase_va: np.ndarray, fold: int) -> dict:
    """Entrena un fold de A2 (gemelo SGD) sobre la MISMA matriz y devuelve su OOF crudo.

    Parameters
    ----------
    x_tr:
        Fold-train con sesgo (CSR float64).
    y_tr:
        Etiquetas del fold-train (reales; A2 no permuta).
    x_va:
        Fold-val con sesgo (CSR float64).
    clase_va:
        Etiquetas reales del fold-val (solo reporte).
    fold:
        Fold canonico (semilla ``20261010 + fold``).

    Returns
    -------
    dict
        ``p_crudo`` y ``p_crudo_train`` (diagnostico), curvas, tiempo y n_iter_sgd.

    Raises
    ------
    Detencion
        Si la config congelada produce probabilidades no finitas (codigo 2).
    """
    inicio = time.time()
    p_va = np.zeros((x_va.shape[0], N_CLASES), dtype=np.float64)
    p_tr = np.zeros((x_tr.shape[0], N_CLASES), dtype=np.float64)
    iteraciones: list[int] = []
    for c in range(N_CLASES):
        modelo = SGDClassifier(**kwargs_sgd(fold))
        modelo.fit(x_tr, (y_tr == c).astype(np.int64))
        p_va[:, c] = modelo.predict_proba(x_va)[:, 1]
        p_tr[:, c] = modelo.predict_proba(x_tr)[:, 1]
        iteraciones.append(int(np.max(modelo.n_iter_)))
    if not np.isfinite(p_va).all() or not np.isfinite(p_tr).all():
        raise Detencion(f"[STOP] A2 fold {fold} produjo probabilidades no finitas con la config "
                        "congelada (alpha 1e-6 con learning_rate optimal); se detiene y se "
                        "registra como fallo de la config, sin ajustar nada (exit 2)", 2)
    return {"p_crudo": p_va, "p_crudo_train": p_tr, "n_iter_sgd": iteraciones,
            "curvas_train": [log_loss_ovr(p_tr, y_tr)],
            "curvas_val": [log_loss_ovr(p_va, clase_va)], "tiempo_seg": float(time.time() - inicio),
            "alpha": None, "n_features": int(x_tr.shape[1])}


def registrar_fold(estado: dict, brazo: str, fold: int, resultado: dict, oof: np.ndarray,
                   oof_crudo: np.ndarray | None, hashes_fold: dict, tr_idx: np.ndarray,
                   va_idx: np.ndarray, clase_va: np.ndarray) -> None:
    """Incorpora el fold al OOF y al checkpoint y persiste ambos (OOF primero, estado luego).

    El OOF que se guarda es el normalizado L2 por fila (comparador canonico); el crudo de A2
    se guarda aparte como diagnostico. La mascara de relleno se escribe una sola vez por fila
    (assert), y el fold-val jamas entra al ajuste.

    Parameters
    ----------
    estado:
        Checkpoint a actualizar.
    brazo:
        Nombre del brazo.
    fold:
        Fold entrenado.
    resultado:
        Salida de ``entrenar_fold_ftrl`` o ``entrenar_fold_sgd``.
    oof:
        OOF normalizado del brazo (se modifica in situ).
    oof_crudo:
        OOF crudo del brazo (solo A2; None en los demas).
    hashes_fold:
        Hashes del vectorizador del fold.
    tr_idx:
        Indices de fold-train.
    va_idx:
        Indices de fold-val.
    clase_va:
        Etiquetas reales del fold-val.

    Raises
    ------
    AssertionError
        Si la mascara detecta una fila escrita dos veces o un solape de folds.
    """
    assert np.intersect1d(tr_idx, va_idx).size == 0, "train y val del fold se solapan"
    p_crudo = resultado["p_crudo"]
    p_crudo_train = resultado.get("p_crudo_train")
    datos = estado["brazos"][brazo]
    mascara = np.asarray(datos["mascara_relleno_por_fila"], dtype=np.int64)
    assert mascara[va_idx].max() == 0, f"{brazo} fold {fold}: fila ya rellenada en el OOF"
    oof[va_idx] = normaliza_l2(p_crudo)
    mascara[va_idx] += 1
    guardar_oof(brazo, oof)
    if oof_crudo is not None and p_crudo_train is not None:
        oof_crudo[va_idx] = p_crudo
        guardar_oof_crudo(brazo, oof_crudo)
    detalle = {
        "n_features": int(resultado["n_features"]),
        "n_vocabulario": int(hashes_fold["n_vocabulario"]),
        "vocab_sha256": hashes_fold["vocab_sha256"],
        "idf_sha256": hashes_fold["idf_sha256"],
        "n_filas_fit": int(hashes_fold["n_filas_fit"]),
        "alpha": resultado["alpha"],
        "tiempo_seg": float(resultado["tiempo_seg"]),
        "log_loss_train_ovr": [float(valor) for valor in resultado["curvas_train"]],
        "log_loss_val_ovr": [float(valor) for valor in resultado["curvas_val"]],
        "log_loss_crudo_fold_val": float(log_loss_ovr(p_crudo, clase_va)),
        "macro_f1_crudo_fold_val": macro(clase_va, p_crudo.argmax(axis=1)),
        "n_iter_sgd": resultado["n_iter_sgd"],
        "n_train_fold": int(tr_idx.size),
        "n_val_fold": int(va_idx.size)}
    datos["folds"][str(fold)] = detalle
    datos["folds_completados"] = sorted(datos["folds"])
    datos["mascara_relleno_por_fila"] = mascara.tolist()
    if detalle["alpha"] is not None:
        datos["alpha_efectivo"] = detalle["alpha"]
    guardar_estado(estado)
    print(f"  [{brazo}] fold {fold}: {detalle['n_features']} coordenadas, vocab "
          f"{detalle['n_vocabulario']}, ll_train {detalle['log_loss_train_ovr'][-1]:.4f}, "
          f"ll_val {detalle['log_loss_crudo_fold_val']:.4f}, macro-F1 crudo "
          f"{detalle['macro_f1_crudo_fold_val']:.4f}, {detalle['tiempo_seg']:.1f} s", flush=True)


def reiniciar_fold(estado: dict, brazo: str, fold: int, oof: np.ndarray,
                   fold_idx: np.ndarray) -> None:
    """Borra un fold del checkpoint y del OOF (usado por el reppaso del fallback).

    Parameters
    ----------
    estado:
        Checkpoint a actualizar.
    brazo:
        Nombre del brazo.
    fold:
        Fold a reiniciar.
    oof:
        OOF del brazo.
    fold_idx:
        Fold canonico por fila.
    """
    va_idx = np.flatnonzero(fold_idx == fold)
    datos = estado["brazos"][brazo]
    mascara = np.asarray(datos["mascara_relleno_por_fila"], dtype=np.int64)
    if str(fold) in datos["folds"]:
        assert int(mascara[va_idx].min()) == 1, f"{brazo} fold {fold}: mascara incoherente"
        mascara[va_idx] -= 1
        oof[va_idx] = 0.0
        del datos["folds"][str(fold)]
    datos["folds_completados"] = sorted(datos["folds"])
    datos["mascara_relleno_por_fila"] = mascara.tolist()
    guardar_oof(brazo, oof)
    guardar_estado(estado)


def preparar_fold(estado: dict, brazo: str, fold: int, alpha: float, oof: np.ndarray,
                  fold_idx: np.ndarray) -> bool:
    """Decide si hay que entrenar (brazo, fold); reinicia si el alpha ya no corresponde.

    Parameters
    ----------
    estado:
        Checkpoint.
    brazo:
        Nombre del brazo.
    fold:
        Fold canonico.
    alpha:
        Alpha efectivo vigente.
    oof:
        OOF del brazo.
    fold_idx:
        Fold canonico por fila.

    Returns
    -------
    bool
        True si el fold debe entrenarse ahora.
    """
    datos = estado["brazos"][brazo]
    if str(fold) not in datos["folds"]:
        return True
    if brazo in BRAZOS_FTRL and datos["folds"][str(fold)].get("alpha") != alpha:
        reiniciar_fold(estado, brazo, fold, oof, fold_idx)
        return True
    return False


def resolver_fallback(estado: dict, oofs: dict[str, np.ndarray], fold_idx: np.ndarray) -> dict:
    """Aplica (o recupera) la regla de fallback CONGELADA a partir de las curvas guardadas.

    Criterio: en el fold 0 de A1, sobre las filas de fold-train, log_loss OvR medio tras 10
    pasadas <= 0.8 x el log_loss tras la pasada 1 y sin NaN/inf. Si falla se pasa al alpha
    alternativo declarado (0.005) para TODOS los folds de A1 (y de A3, que es A1 con
    etiquetas barajadas) y se repasa el fold 0. Jamas se decide con fold-val ni con el
    ranking.

    Parameters
    ----------
    estado:
        Checkpoint.
    oofs:
        OOF por brazo (para reiniciar el fold 0 si hay repasso).
    fold_idx:
        Fold canonico por fila.

    Returns
    -------
    dict
        Registro del fallback con ``alpha_efectivo`` y ``config_fallback_usada``.
    """
    fallback = estado["fallback"]
    if fallback["evaluado"]:
        return fallback
    datos = estado["brazos"]["A1_ftrl"]
    if "0" not in datos["folds"]:
        return fallback
    curva = datos["folds"]["0"]["log_loss_train_ovr"]
    primero, ultimo = float(curva[0]), float(curva[-1])
    finito = bool(np.isfinite([primero, ultimo]).all())
    cumple = bool(finito and ultimo <= FALLBACK_FACTOR * primero)
    fallback.update({"evaluado": True, "log_loss_train_epoca1": primero,
                     "log_loss_train_epoca10": ultimo, "finito": finito, "cumple": cumple,
                     "config_fallback_usada": bool(not cumple),
                     "alpha_efectivo": ALPHA if cumple else ALPHA_ALTERNATIVO,
                     "reinicios": []})
    guardar_estado(estado)
    if not cumple:
        for brazo in BRAZOS_FTRL:
            detalle = estado["brazos"][brazo]["folds"].get("0")
            if detalle is not None and detalle.get("alpha") != ALPHA_ALTERNATIVO:
                reiniciar_fold(estado, brazo, 0, oofs[brazo], fold_idx)
                fallback["reinicios"].append(f"{brazo}/0")
    for brazo in BRAZOS_FTRL:
        estado["brazos"][brazo]["alpha_efectivo"] = fallback["alpha_efectivo"]
    guardar_estado(estado)
    print(f"  fallback: ll_train ep1 {primero:.4f} -> ep{EPOCAS} {ultimo:.4f} (umbral "
          f"{FALLBACK_FACTOR:g} x ep1) cumple={cumple} alpha_efectivo "
          f"{fallback['alpha_efectivo']}", flush=True)
    return fallback


def asegurar_calibracion(estado: dict) -> None:
    """Aplica la calibracion congelada tras el fold 0 de A1 (exit 6 si proyecta > 20 h).

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
    datos = estado["brazos"]["A1_ftrl"]
    if "0" not in datos["folds"]:
        return
    tiempo_fold0 = float(datos["folds"]["0"]["tiempo_seg"])
    proyeccion_seg = PROYECCION_FACTOR * tiempo_fold0
    estado["calibracion"] = {"t_fold0_A1_seg": tiempo_fold0, "factor": PROYECCION_FACTOR,
                             "proyeccion_seg": proyeccion_seg,
                             "proyeccion_horas": proyeccion_seg / 3600.0,
                             "limite_horas": PROYECCION_MAX_H,
                             "ok": bool(proyeccion_seg <= PROYECCION_MAX_H * 3600.0)}
    guardar_estado(estado)
    if not estado["calibracion"]["ok"]:
        raise Detencion(f"[STOP] calibracion: {PROYECCION_FACTOR:g} x t_fold0 = "
                        f"{estado['calibracion']['proyeccion_horas']:.2f} h > "
                        f"{PROYECCION_MAX_H} h; estado guardado (exit 6)", 6)


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
        raise Detencion(f"[STOP] RSS {actual:.2f} GB supera el tope de {RAM_TOPE_GB:g} GB; "
                        "estado guardado y no se anade max_features (exit 8)", 8)
    return actual


def entrenar_con_checkpoints(textos: np.ndarray, clase: np.ndarray, fold_idx: np.ndarray,
                             estado: dict) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Entrena los 3 brazos x 5 folds de forma resumible y devuelve los OOF.

    Por fold se ajusta el vectorizador UNA vez (compartido por los tres brazos) y luego se
    recorren los brazos en orden A1, A2, A3. La regla de fallback se evalua en cuanto A1
    termina su fold 0: si se activa, se reinicia el fold 0 de A1 (y de A3 si ya estaba) y se
    reentrena con el alpha alternativo antes de seguir con el resto del fold.

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
    tuple[dict[str, numpy.ndarray], numpy.ndarray]
        OOF normalizado L2 por brazo y OOF crudo de A2.

    Raises
    ------
    Detencion
        Timeout global (7), calibracion excedida (6), RSS excedido (8) o invariantes (2).
    """
    oofs = {brazo: asegurar_brazo_cargado(estado, brazo, fold_idx) for brazo in BRAZOS}
    oof_crudo = np.zeros((N_FILAS, N_CLASES), dtype=np.float64)
    if estado["brazos"]["A2_sgd_twin"]["folds"]:
        # Reanudacion: el OOF crudo ya persistido se vuelve a leer para que el diagnostico de
        # A2 no se calcule sobre una matriz de ceros (el OOF crudo no entra en ningun gate).
        ruta_cruda = ruta_oof_crudo("A2_sgd_twin")
        if not ruta_cruda.exists():
            raise Detencion("[STOP] el checkpoint declara folds de A2 pero falta su OOF crudo "
                            "diagnostico; se detiene para no reportar un diagnostico de ceros "
                            "(exit 2)", 2)
        oof_crudo = np.load(ruta_cruda).astype(np.float64)
        if oof_crudo.shape != (N_FILAS, N_CLASES) or not np.isfinite(oof_crudo).all():
            raise Detencion("[STOP] el OOF crudo de A2 tiene forma o valores invalidos (exit 2)",
                            2)
    elif ruta_oof_crudo("A2_sgd_twin").exists():
        ruta_oof_crudo("A2_sgd_twin").unlink()
    asegurar_calibracion(estado)
    for fold in range(N_FOLDS):
        verificar_deadline(estado)
        tr_idx, va_idx = indices_fold(fold_idx, fold)
        inicio_fit = time.time()
        ajuste = ajustar_vectorizador_fold(textos, tr_idx, va_idx)
        hashes_fold = dict(ajuste["hashes"])
        hashes_fold["tiempo_fit_seg"] = float(time.time() - inicio_fit)
        estado["vectorizador_por_fold"][str(fold)] = hashes_fold
        guardar_estado(estado)
        print(f"  [fold {fold}] TF-IDF ajustado solo en fold-train: "
              f"{ajuste['x_tr'].shape[1]} coordenadas, {ajuste['x_tr'].nnz} no nulos, "
              f"{hashes_fold['tiempo_fit_seg']:.1f} s, vocab "
              f"{hashes_fold['vocab_sha256'][:12]}", flush=True)
        pendientes = True
        while pendientes:
            pendientes = False
            for brazo in BRAZOS:
                alpha = float(resolver_fallback(estado, oofs, fold_idx)["alpha_efectivo"])
                if not preparar_fold(estado, brazo, fold, alpha, oofs[brazo], fold_idx):
                    continue
                verificar_deadline(estado)
                tr_brazo, va_brazo, y_tr = etiquetas_objetivo(brazo, clase, fold_idx, fold)
                if brazo == "A2_sgd_twin":
                    resultado = entrenar_fold_sgd(ajuste["x_tr"], y_tr, ajuste["x_va"],
                                                  clase[va_brazo], fold)
                else:
                    resultado = entrenar_fold_ftrl(ajuste["x_tr"], y_tr, ajuste["x_va"],
                                                   clase[va_brazo], alpha, fold)
                registrar_fold(estado, brazo, fold, resultado, oofs[brazo],
                               oof_crudo if brazo == "A2_sgd_twin" else None, hashes_fold,
                               tr_brazo, va_brazo, clase[va_brazo])
                verificar_ram(estado)
                if brazo == "A1_ftrl" and fold == 0:
                    antes = json.dumps(estado["fallback"], sort_keys=True)
                    resolver_fallback(estado, oofs, fold_idx)
                    pendientes = json.dumps(estado["fallback"], sort_keys=True) != antes
        if fold == 0:
            asegurar_calibracion(estado)
        verificar_ram(estado)
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
    return oofs, oof_crudo


def sonda_determinismo(brazo: str, textos: np.ndarray, clase: np.ndarray, fold_idx: np.ndarray,
                       oof: np.ndarray, estado: dict, alpha: float) -> dict:
    """Reentrena el fold 0 del brazo desde cero y mide max|delta p| contra el OOF guardado.

    Ademas comprueba que el reajuste del vectorizador del fold 0 reproduce los mismos hashes
    de vocabulario e idf (sonda de refit del pre-registro). Tolerancias congeladas: 0 ->
    exacto; <=1e-9 -> conforme; >1e-9 -> segunda repeticion con rango publicado; rango >=
    variabilidad o solape con el gate -> no concluyente.

    Parameters
    ----------
    brazo:
        Brazo a sondear (A1 o A2).
    textos:
        Textos del corpus.
    clase:
        Etiquetas canonicas.
    fold_idx:
        Fold canonico por fila.
    oof:
        OOF normalizado guardado del brazo.
    estado:
        Checkpoint.
    alpha:
        Alpha efectivo de A1 (A2 no lo usa).

    Returns
    -------
    dict
        Deltas de cada repeticion, rango, clasificacion, refit y ``no_concluyente``.
    """
    verificar_deadline(estado)
    referencia = oof[fold_idx == 0]
    tr_idx, va_idx, y_tr = etiquetas_objetivo(brazo, clase, fold_idx, 0)
    ajuste = ajustar_vectorizador_fold(textos, tr_idx, va_idx)
    guardado = estado["vectorizador_por_fold"].get("0", {})
    refit = {"vocab_sha256": ajuste["hashes"]["vocab_sha256"],
             "idf_sha256": ajuste["hashes"]["idf_sha256"],
             "n_vocabulario": int(ajuste["hashes"]["n_vocabulario"])}
    refit_ok = bool(refit["vocab_sha256"] == guardado.get("vocab_sha256")
                    and refit["idf_sha256"] == guardado.get("idf_sha256"))
    if brazo == "A2_sgd_twin":
        resultado = entrenar_fold_sgd(ajuste["x_tr"], y_tr, ajuste["x_va"], clase[va_idx], 0)
    else:
        resultado = entrenar_fold_ftrl(ajuste["x_tr"], y_tr, ajuste["x_va"], clase[va_idx], alpha,
                                       0)
    delta_1 = float(np.abs(normaliza_l2(resultado["p_crudo"]) - referencia).max())
    sonda = {"descripcion": f"reentrenamiento de {brazo} fold 0 con config identica",
             "max_abs_diff_repeticion_1": delta_1, "max_abs_diff": delta_1,
             "segunda_repeticion": None, "rango": None, "clasificacion": None,
             "no_concluyente": False, "solape_con_gate": False,
             "tolerancia_declarada": TOL_DETERMINISMO, "variabilidad_referencia": VARIABILIDAD,
             "refit_vocab_idf_reproducidos": refit_ok, "refit": refit,
             "vocab_sha256_guardado": guardado.get("vocab_sha256"),
             "idf_sha256_guardado": guardado.get("idf_sha256")}
    if delta_1 == 0.0:
        sonda["clasificacion"] = "exacto"
    elif delta_1 <= TOL_DETERMINISMO:
        sonda["clasificacion"] = "tolerancia_declarada"
    else:
        verificar_deadline(estado)
        if brazo == "A2_sgd_twin":
            repetida = entrenar_fold_sgd(ajuste["x_tr"], y_tr, ajuste["x_va"], clase[va_idx], 0)
        else:
            repetida = entrenar_fold_ftrl(ajuste["x_tr"], y_tr, ajuste["x_va"], clase[va_idx],
                                          alpha, 0)
        delta_2 = float(np.abs(normaliza_l2(repetida["p_crudo"]) - referencia).max())
        rango = max(delta_1, delta_2) - min(delta_1, delta_2)
        sonda.update({"clasificacion": "rango_publicado", "segunda_repeticion": delta_2,
                      "max_abs_diff": max(delta_1, delta_2), "rango": rango,
                      "solape_con_gate": bool(rango >= GATE_DELTA),
                      "no_concluyente": bool(rango >= VARIABILIDAD or rango >= GATE_DELTA)})
    print(f"  sonda de determinismo {brazo} fold 0: max|delta p| = "
          f"{sonda['max_abs_diff']:.3e} ({sonda['clasificacion']}), refit idem={refit_ok}",
          flush=True)
    return sonda


def ejecutar_sondas_determinismo(textos: np.ndarray, clase: np.ndarray, fold_idx: np.ndarray,
                                 oofs: dict[str, np.ndarray], estado: dict) -> dict:
    """Ejecuta y guarda las sondas de determinismo de A1 y de A2 al terminar los brazos.

    Parameters
    ----------
    textos:
        Textos del corpus.
    clase:
        Etiquetas canonicas.
    fold_idx:
        Fold canonico por fila.
    oofs:
        OOF por brazo.
    estado:
        Checkpoint.

    Returns
    -------
    dict
        Sondas por brazo y bandera global ``no_concluyente``.
    """
    alpha = float(estado["fallback"]["alpha_efectivo"])
    no_concluyente = False
    for brazo in BRAZOS_CON_SONDA:
        if estado["brazos"][brazo]["sonda_determinismo"] is not None:
            sonda = estado["brazos"][brazo]["sonda_determinismo"]
        else:
            sonda = sonda_determinismo(brazo, textos, clase, fold_idx, oofs[brazo], estado,
                                       alpha)
            estado["brazos"][brazo]["sonda_determinismo"] = sonda
            guardar_estado(estado)
        no_concluyente = bool(no_concluyente or sonda["no_concluyente"])
    verificar_ram(estado)
    return {"no_concluyente": no_concluyente, "tolerancia_declarada": TOL_DETERMINISMO,
            "variabilidad_referencia": VARIABILIDAD}


def verificar_gate_a3(oof_a3: np.ndarray, clase: np.ndarray, linea_base: dict,
                      estado: dict) -> dict:
    """Gate A3: el control negativo debe quedar por debajo de la linea base mayoritaria.

    Parameters
    ----------
    oof_a3:
        OOF normalizado de A3 (169336, 6).
    clase:
        Etiquetas canonicas.
    linea_base:
        Linea base de la clase mayoritaria.
    estado:
        Checkpoint (se guarda antes de detener).

    Returns
    -------
    dict
        ``macro_f1_standalone``, linea base y ``coherente``.

    Raises
    ------
    Detencion
        Si A3 alcanza o supera la linea base de la clase mayoritaria (codigo 5, sospecha de
        fuga).
    """
    standalone = macro(clase, oof_a3.argmax(axis=1))
    coherente = bool(standalone < linea_base["macro_f1_linea_base"])
    if not coherente:
        guardar_estado(estado)
        raise Detencion(f"[STOP] A3 incoherente (sospecha de fuga): macro-F1 standalone "
                        f"{standalone:.7f} >= linea base de la clase mayoritaria "
                        f"{linea_base['macro_f1_linea_base']:.7f}; estado guardado (exit 5)", 5)
    print(f"  gate A3: macro-F1 standalone {standalone:.7f} < linea base mayoritaria "
          f"{linea_base['macro_f1_linea_base']:.7f} ({linea_base['clase']})", flush=True)
    return {"macro_f1_standalone": standalone,
            "linea_base_mayoritaria": linea_base["macro_f1_linea_base"],
            "clase_mayoritaria": linea_base["clase"], "coherente": coherente,
            "criterio": A3_CONTROL["umbral_fuga"]}


def verificar_invariantes(estado: dict, clase: np.ndarray, fold_idx: np.ndarray,
                          oofs: dict[str, np.ndarray]) -> dict:
    """Evidencias de no-leakage derivadas de los artefactos (no declaraciones a mano).

    Returns
    -------
    dict
        Chequeos booleanos: alineacion, folds canonicos, folds completos, mascara 1 por fila,
        OOF finito y normalizado L2, vocabulario e idf registrados, vectorizador ajustado
        solo en fold-train, coherencia de dimensiones y refit reproducible.
    """
    completos, mascaras, finitos, normalizados, hashes_ok, filas_fit, dimensiones = (
        [], [], [], [], [], [], [])
    for brazo in BRAZOS:
        datos = estado["brazos"][brazo]
        completos.append(datos["folds_completados"] == [str(fold) for fold in range(N_FOLDS)])
        mascara = np.asarray(datos["mascara_relleno_por_fila"], dtype=np.int64)
        mascaras.append(bool((mascara == 1).all()))
        finitos.append(bool(np.isfinite(oofs[brazo]).all()))
        normas = np.linalg.norm(oofs[brazo], axis=1)
        normalizados.append(bool(np.all(np.abs(normas - 1.0) <= 1e-9)))
        for fold in range(N_FOLDS):
            detalle = datos["folds"][str(fold)]
            hashes_ok.append(len(detalle["vocab_sha256"]) == 64
                             and len(detalle["idf_sha256"]) == 64)
            filas_fit.append(detalle["n_filas_fit"] == detalle["n_train_fold"])
            dimensiones.append(detalle["n_features"] == detalle["n_vocabulario"] + 1)
    refits = [bool((estado["brazos"][brazo]["sonda_determinismo"] or {}).get(
        "refit_vocab_idf_reproducidos", False)) for brazo in BRAZOS_CON_SONDA]
    return {
        "alineacion_filas": bool(len(clase) == N_FILAS),
        "cinco_folds_canonicos": bool(np.unique(fold_idx).size == N_FOLDS),
        "folds_completos_por_brazo": bool(all(completos)),
        "oof_mascara_relleno_uno_por_fila": bool(all(mascaras)),
        "oof_sin_no_finitos": bool(all(finitos)),
        "oof_normalizado_l2_por_fila": bool(all(normalizados)),
        "vocabulario_e_idf_registrados_por_fold": bool(all(hashes_ok)),
        "vectorizador_ajustado_solo_en_fold_train": bool(all(filas_fit)),
        "n_features_coherente_vocab_mas_sesgo": bool(all(dimensiones)),
        "refit_reproduce_vocabulario_e_idf": bool(all(refits)),
    }


# =====================================================================================
# ---- 5. EVALUACION: INTEGRACION P6/P7, COMPLEMENTARIEDAD Y BOOTSTRAP ----
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
                     - macro_de_conteos(np.bincount(codigo_base[idx],
                                                    minlength=N_CLASES ** 2)))
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
        OOF normalizado L2 del brazo (169336, 6).
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
        assert list(modelo.classes_) == list(range(N_CLASES)), \
            "el combinador no vio las 6 clases en orden canonico"
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
    argmax (y por tanto el macro-F1 standalone) es identico al del OOF crudo: el comparador
    canonico L2 no cambia la decision, solo hace comparables las filas entre brazos.

    Parameters
    ----------
    p_brazo:
        OOF normalizado L2 del brazo (169336, 6).
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
        con su diagonal, log_loss y rescate/dano/neto globales y por clase.
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
            "log_loss_l2": log_loss_multiclase(p_brazo, clase),
            "log_loss_ovr_medio": log_loss_ovr(p_brazo, clase),
            "f1_por_clase": [float(valor) for valor in f1_por_clase],
            "matriz_esperada": esperada.tolist(), "diagonal_ok": diagonal_ok,
            "argmax_identico_al_crudo": True,
            "vs_campeon": {"rescate": rescate, "danio": danio, "neto": rescate - danio,
                           "rescates_alta_confianza": int((alta_confianza & ok_cand).sum()),
                           "errores_alta_confianza_campeon": int(alta_confianza.sum()),
                           "por_clase": por_clase}}


def diagnostico_crudo_a2(oof_crudo: np.ndarray, clase: np.ndarray) -> dict:
    """Diagnostico del OOF CRUDO de A2 (seis sigmoides independientes, no suman 1).

    Se reportan el argmax crudo, su macro-F1, el log_loss OvR, el log_loss tras normalizar L2
    y la suma media por fila como evidencia de que el comparador canonico (L2) es el que se
    integra.

    Parameters
    ----------
    oof_crudo:
        OOF crudo de A2 (169336, 6).
    clase:
        Etiquetas canonicas.

    Returns
    -------
    dict
        Diagnosticos escalares del OOF crudo de A2.
    """
    filas = oof_crudo.sum(axis=1)
    return {"n_filas": int(oof_crudo.shape[0]),
            "filas_que_suman_1": int(np.isclose(filas, 1.0).sum()),
            "suma_media_fila": float(filas.mean()),
            "macro_f1_argmax_crudo": macro(clase, oof_crudo.argmax(axis=1)),
            "log_loss_ovr_medio": log_loss_ovr(oof_crudo, clase),
            "log_loss_l2_tras_normalizar": log_loss_multiclase(oof_crudo, clase),
            "nota": "A2 entrega seis sigmoides independientes (no suman 1); el OOF crudo se "
                    "guarda solo como diagnostico y el comparador canonico es el L2"}


# =====================================================================================
# ---- 6. GATES, CAPS Y VEREDICTO ----
# =====================================================================================
def calcular_gates(baseline_ok: bool, integracion: dict, control_atribucion: dict,
                   gate_a3: dict, invariantes: dict, no_concluyente: bool
                   ) -> tuple[dict, dict, dict, str]:
    """Aplica los gates congelados, los caps y el arbol de decision.

    Gates: baseline recomputado, delta de A1 >= +0.0015, folds positivos >= 4/5, control
    favorable A1 sobre A2 (delta puntual y bootstrap pareado de la diferencia sin cero), IC95
    del delta de A1 sin cero, A3 coherente, sin leakage y determinismo concluyente. Caps: si
    el IC95 de la diferencia contiene 0 el veredicto queda capado a MARGINAL, y si la sonda de
    determinismo no es concluyente el veredicto no puede ser PROMISING. Precedencia: si
    delta_int(A3) >= delta_int(A1) el veredicto es NO_ATRIBUIBLE.

    Parameters
    ----------
    baseline_ok:
        Resultado del chequeo del baseline.
    integracion:
        Metricas del combinador por brazo.
    control_atribucion:
        Bootstrap pareado de la diferencia A1-A2.
    gate_a3:
        Resultado del gate A3.
    invariantes:
        Evidencias de no leakage.
    no_concluyente:
        True si alguna sonda de determinismo no es concluyente.

    Returns
    -------
    tuple[dict, dict, dict, str]
        Gates, control negativo, caps y veredicto del arbol.
    """
    a1 = integracion["A1_ftrl"]
    a2 = integracion["A2_sgd_twin"]
    a3 = integracion["A3_ftrl_barajado"]
    gates = {
        "baseline_ok": bool(baseline_ok),
        "delta_A1_suficiente": bool(a1["delta_vs_baseline"] >= GATE_DELTA),
        "folds_positivos_A1_suficiente": bool(a1["folds_positivos"] >= GATE_FOLDS),
        "control_favorable_A1_sobre_A2": bool(
            a1["delta_vs_baseline"] > a2["delta_vs_baseline"]
            and control_atribucion["ic95_sin_cero"]),
        "ic95_A1_sin_cero": bool(a1["ic95_sin_cero"]),
        "a3_coherente": bool(gate_a3["coherente"]),
        "sin_leakage": bool(all(invariantes.values())),
        "determinismo_concluyente": bool(not no_concluyente),
    }
    control_negativo = {"delta_A1": a1["delta_vs_baseline"],
                        "delta_A3": a3["delta_vs_baseline"],
                        "criterio": GATES_PREREG["control_negativo"],
                        "no_atribuible": bool(a3["delta_vs_baseline"]
                                              >= a1["delta_vs_baseline"])}
    caps = {"ic95_diferencia_contiene_cero": bool(not control_atribucion["ic95_sin_cero"]),
            "criterio": GATES_PREREG["cap_ic95_diferencia"],
            "determinismo_no_concluyente": bool(no_concluyente),
            "criterio_determinismo": GATES_PREREG["cap_determinismo"]}
    if control_negativo["no_atribuible"]:
        veredicto = "NO_ATRIBUIBLE"
    elif all(gates.values()) and not caps["ic95_diferencia_contiene_cero"]:
        veredicto = "PROMISING"
    elif a1["delta_vs_baseline"] > 0:
        veredicto = "MARGINAL"
    else:
        veredicto = "CLOSED"
    return gates, control_negativo, caps, veredicto


# =====================================================================================
# ---- 7. SALIDAS (JSON, informe, figura, RUNLOG, CSV, consola) ----
# =====================================================================================
def figura_curvas(entrenamiento: dict, integracion: dict, ruta: Path) -> None:
    """Figura 1x3: log_loss por epoca de A1 y de A2 por fold, y deltas por fold.

    Panel 1: log_loss OvR de A1 por epoca (fold-train y fold-val) para los 5 folds. Panel 2:
    log_loss de A2 por fold (SGDClassifier no expone curvas por epoca: se representa el
    valor final por fold, replicado en las 10 pasadas y rotulado como tal). Panel 3: delta
    por fold frente al campeon de los tres brazos con la linea del gate.

    Parameters
    ----------
    entrenamiento:
        Detalle de folds por brazo.
    integracion:
        Metricas del combinador por brazo.
    ruta:
        Destino del PNG.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figura, ejes = plt.subplots(1, 3, figsize=(19.0, 5.0))
    for fold in range(N_FOLDS):
        detalle = entrenamiento["A1_ftrl"]["folds"][str(fold)]
        color = plt.cm.viridis(fold / max(N_FOLDS - 1, 1))
        ejes[0].plot(range(1, EPOCAS + 1), detalle["log_loss_train_ovr"], color=color,
                     linewidth=1.2, label=f"fold {fold} train")
        ejes[0].plot(range(1, EPOCAS + 1), detalle["log_loss_val_ovr"], color=color,
                     linewidth=1.0, linestyle="--", label=f"fold {fold} val")
    ejes[0].set_title("A1 FTRL-Proximal: log_loss OvR por epoca (train solido, val discontinuo)")
    ejes[0].set_xlabel("epoca")
    ejes[0].set_ylabel("log_loss OvR medio")
    ejes[0].grid(alpha=0.3)
    ejes[0].legend(fontsize=6, ncol=2)
    for fold in range(N_FOLDS):
        detalle = entrenamiento["A2_sgd_twin"]["folds"][str(fold)]
        color = plt.cm.plasma(fold / max(N_FOLDS - 1, 1))
        ejes[1].plot(range(1, EPOCAS + 1), [float(np.mean(detalle["log_loss_train_ovr"]))]
                     * EPOCAS, color=color, linewidth=1.2, label=f"fold {fold} train")
        ejes[1].plot(range(1, EPOCAS + 1), [detalle["log_loss_crudo_fold_val"]] * EPOCAS,
                     color=color, linewidth=1.0, linestyle="--", label=f"fold {fold} val")
    ejes[1].set_title("A2 gemelo SGD: valor final por fold (sin curva por epoca)")
    ejes[1].set_xlabel("epoca (10 pasadas internas)")
    ejes[1].set_ylabel("log_loss OvR medio")
    ejes[1].grid(alpha=0.3)
    ejes[1].legend(fontsize=6, ncol=2)
    posiciones = np.arange(N_FOLDS)
    ancho = 0.27
    for indice, brazo in enumerate(BRAZOS):
        ejes[2].bar(posiciones + (indice - 1) * ancho, integracion[brazo]["deltas_por_fold"],
                    width=ancho, label=NOMBRES_BRAZO[brazo])
    ejes[2].axhline(0.0, color="k", linewidth=0.8)
    ejes[2].axhline(GATE_DELTA, color="r", linewidth=0.8, linestyle=":",
                    label=f"gate +{GATE_DELTA}")
    ejes[2].set_xticks(posiciones)
    ejes[2].set_xticklabels([f"fold {fold}" for fold in range(N_FOLDS)])
    ejes[2].set_title("Delta por fold del macro-F1 combinado vs campeon")
    ejes[2].set_ylabel("delta macro-F1")
    ejes[2].grid(alpha=0.3, axis="y")
    ejes[2].legend(fontsize=7)
    figura.tight_layout()
    figura.savefig(ruta, dpi=140)
    plt.close(figura)
    print(f"  figura: {ruta_relativa(ruta)}", flush=True)


def escribir_informe(resultado: dict) -> None:
    """Escribe el informe Markdown del experimento con tablas, gates y asimetrias.

    Parameters
    ----------
    resultado:
        JSON de resultados ya construido.
    """
    integracion = resultado["integracion"]
    entrenamiento = resultado["entrenamiento"]
    sondas = resultado["sondas"]
    fallback = resultado["fallback"]
    linea = [
        "# V32-E3 (P3) - Ejecucion: challenger FTRL-Proximal sobre TF-IDF por fold",
        "",
        f"- Fecha de ejecucion: {resultado['fecha']}",
        (f"- Pre-registro: {ARCHIVO_PRE_REGISTRO} (commit "
         f"{resultado['prereg']['commit_congelado'][:7]}) y `experiments/"
         "v32_04_preregistro_ftrl.json`"),
        f"- Commit de ejecucion: `{resultado['commit']}`",
        (f"- Baseline primario recomputado: "
         f"{resultado['verificacion']['baseline_recomputado']:.7f} (referencia nested "
         f"{NESTED:.7f})"),
        (f"- {HILOS} hilos fijos; TF-IDF ajustada por fold; FTRL {EPOCAS} pasadas fijas; alpha "
         f"efectivo {fallback['alpha_efectivo']}; RSS pico {resultado['rss_pico_gb']:.2f} GB "
         f"(tope {RAM_TOPE_GB:g} GB)"),
        (f"- Compliance AST: {resultado['compliance']['literales_de_ruta_revisados']} literales "
         f"de ruta y {resultado['compliance']['importaciones_revisadas']} modulos importados "
         "revisados, 0 violaciones"),
        "",
        "## Sondas obligatorias (a)-(d), ejecutadas antes de cualquier ajuste",
        "",
        "| Sonda | Valor | Tolerancia | pass |",
        "|---|---|---|---|",
        (f"| (a) vectorizado vs lazo | max delta = "
         f"{sondas['a_vectorizado_vs_lazo']['max_abs_diff']:.3e} | "
         f"{sondas['a_vectorizado_vs_lazo']['tolerancia']:.0e} | "
         f"{sondas['a_vectorizado_vs_lazo']['pass']} |"),
        (f"| (b) raiz analitica | residuo raiz = "
         f"{sondas['b_punto_fijo_analitico']['residuo_raiz']:.3e}, un paso = "
         f"{sondas['b_punto_fijo_analitico']['residuo_un_paso']:.3e} | "
         f"{sondas['b_punto_fijo_analitico']['umbral']:.0e} | "
         f"{sondas['b_punto_fijo_analitico']['pass']} |"),
        (f"| (c) TotalF1 average=Macro | delta vs macro-F1 = "
         f"{sondas['c_totalf1_macro']['delta_criterio']:.3e} | "
         f"{sondas['c_totalf1_macro']['tolerancia']:.0e} | "
         f"{sondas['c_totalf1_macro']['pass']} |"),
        (f"| (d) sin NaN y log_loss finito | log_loss OvR = "
         f"{sondas['d_sin_nan_logloss_finito']['log_loss_ovr']:.4f} | finito | "
         f"{sondas['d_sin_nan_logloss_finito']['pass']} |"),
        "",
        "## Entrenamiento por fold (el fold-val solo se uso para el OOF y como reporte)",
        "",
        ("| Brazo | fold | n_feat | vocab | ll train ep1 | ll train ep10 | ll val | macro-F1 "
         "crudo val | alpha | tiempo (s) |"),
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for brazo in BRAZOS:
        for fold in range(N_FOLDS):
            detalle = entrenamiento[brazo]["folds"][str(fold)]
            alpha = detalle["alpha"] if detalle["alpha"] is not None else "-"
            linea.append(
                f"| {brazo} | {fold} | {detalle['n_features']} | {detalle['n_vocabulario']} | "
                f"{detalle['log_loss_train_ovr'][0]:.4f} | "
                f"{detalle['log_loss_train_ovr'][-1]:.4f} | "
                f"{detalle['log_loss_crudo_fold_val']:.4f} | "
                f"{detalle['macro_f1_crudo_fold_val']:.4f} | {alpha} | "
                f"{detalle['tiempo_seg']:.1f} |")
    iteraciones_a2 = sorted({int(n) for f in range(N_FOLDS)
                             for n in entrenamiento["A2_sgd_twin"]["folds"][str(f)]["n_iter_sgd"]})
    linea += [
        "",
        (f"Nota: en A2 (gemelo SGD) la tabla muestra UN solo punto de log-loss por fold (las "
         f"columnas ep1 y ep10 repiten el mismo valor) porque `SGDClassifier` corre sus "
         f"{SGD_MAX_ITER} pasadas internas sin callback por epoca; `n_iter_` = {iteraciones_a2} "
         f"en los {N_FOLDS} folds, el mismo presupuesto de pasadas que A1. A1 y A3 si tienen "
         f"curva de {EPOCAS} puntos por epoca."),
        "",
        "## Vocabulario e idf por fold (sha256, primeros 16)",
        "",
        "| fold | n_vocabulario | vocab_sha256 | idf_sha256 | tiempo fit (s) |",
        "|---|---|---|---|---|",
    ]
    for fold in range(N_FOLDS):
        detalle = resultado["vocabulario_idf_por_fold"][str(fold)]
        linea.append(f"| {fold} | {detalle['n_vocabulario']} | `{detalle['vocab_sha256'][:16]}`"
                     f" | `{detalle['idf_sha256'][:16]}` | {detalle['tiempo_fit_seg']:.1f} |")
    linea += [
        "",
        "## Regla de fallback (congelada antes de ejecutar)",
        "",
        f"- Criterio: {fallback['criterio']}",
        (f"- Fold 0 de A1: log_loss train epoca 1 = {fallback['log_loss_train_epoca1']:.4f}, "
         f"epoca {EPOCAS} = {fallback['log_loss_train_epoca10']:.4f} (umbral "
         f"{FALLBACK_FACTOR:g} x ep1 = "
         f"{FALLBACK_FACTOR * fallback['log_loss_train_epoca1']:.4f})"),
        (f"- Cumple: {fallback['cumple']} | config_fallback_usada: "
         f"{fallback['config_fallback_usada']} | alpha efectivo: {fallback['alpha_efectivo']} | "
         f"repasos: {fallback['reinicios']}"),
        "- La decision se tomo con las filas de fold-train; jamas con el fold-val ni con el ranking.",
        "",
        "## Sondas de determinismo (reentreno del fold 0 con config identica)",
        "",
    ]
    for brazo in BRAZOS_CON_SONDA:
        sonda = resultado["determinismo"][f"sonda_{brazo}"]
        linea.append(f"- {brazo}: max delta p = {sonda['max_abs_diff']:.3e} "
                     f"({sonda['clasificacion']}), rango = {sonda['rango']}, no concluyente = "
                     f"{sonda['no_concluyente']}, refit de vocabulario e idf identico = "
                     f"{sonda['refit_vocab_idf_reproducidos']}")
    linea += [
        "",
        "## Integracion P6/P7 (combinador congelado, regla argmax(p_comb * w))",
        "",
        ("| Brazo | macro-F1 comb | delta vs baseline | folds+ | IC95 | delta sin regla | delta "
         "vs nested | log_loss p_comb | sha256 OOF (16) |"),
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for brazo in BRAZOS:
        met = integracion[brazo]
        sha = entrenamiento[brazo]["oof_artefacto"]["sha256"][:16]
        linea.append(
            f"| {brazo} | {met['corrected_cv']:.7f} | {met['delta_vs_baseline']:+.7f} | "
            f"{met['folds_positivos']}/5 | [{met['ic95'][0]:+.7f}, {met['ic95'][1]:+.7f}] | "
            f"{met['delta_sin_regla_vs_baseline']:+.7f} | {met['delta_vs_nested']:+.7f} | "
            f"{met['log_loss_p_comb']:.4f} | `{sha}` |")
    linea += [
        "",
        "## Complementariedad standalone (argmax del OOF normalizado L2, sin pesos)",
        "",
        ("| Brazo | macro-F1 solo | diagonal 6x6 | log_loss L2 | rescate | dano | neto | rescate "
         "alta conf. | errores alta conf. campeon |"),
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for brazo in BRAZOS:
        diag = resultado["complementariedad"][brazo]
        contra = diag["vs_campeon"]
        linea.append(
            f"| {brazo} | {diag['standalone_cv']:.7f} | {diag['diagonal_ok']} | "
            f"{diag['log_loss_l2']:.4f} | {contra['rescate']} | {contra['danio']} | "
            f"{contra['neto']:+d} | {contra['rescates_alta_confianza']} | "
            f"{contra['errores_alta_confianza_campeon']} |")
    crudo = resultado.get("diagnostico_crudo_a2") or {}
    if crudo:
        linea += [
            "",
            (f"- OOF crudo de A2 (diagnostico): {crudo['macro_f1_argmax_crudo']:.7f} de macro-F1 "
             f"con argmax crudo, log_loss OvR {crudo['log_loss_ovr_medio']:.4f}, suma media por "
             f"fila {crudo['suma_media_fila']:.4f} y {crudo['filas_que_suman_1']} filas que suman "
             "1: no suma 1 por construccion (seis sigmoides independientes)."),
            f"- Artefacto: `{crudo['artefacto']}` (sha256 `{crudo['sha256'][:16]}`). {crudo['uso']}.",
            ("- La normalizacion L2 escala cada fila por una constante positiva: el argmax (y el "
             "macro-F1 standalone) es identico al del OOF crudo."),
        ]
    else:
        linea += ["", "- OOF crudo de A2: diagnostico no disponible en esta corrida."]
    linea += [
        "",
        "## Gates",
        "",
        "| Gate | Resultado |",
        "|---|---|",
    ]
    for nombre, valor in resultado["gates"].items():
        linea.append(f"| {nombre} | {valor} |")
    linea += [
        "",
        "## Caps y controles",
        "",
        (f"- Bootstrap pareado de la diferencia (F1_combA1 - F1_combA2): delta medio "
         f"{resultado['control_atribucion']['delta_medio']:+.7f}, IC95 "
         f"[{resultado['control_atribucion']['ic95'][0]:+.7f}, "
         f"{resultado['control_atribucion']['ic95'][1]:+.7f}], favorable = "
         f"{resultado['control_atribucion']['favorable']}"),
        (f"- delta_int(A3) = {resultado['control_negativo']['delta_A3']:+.7f} vs delta_int(A1) = "
         f"{resultado['control_negativo']['delta_A1']:+.7f}: "
         f"{'NO_ATRIBUIBLE' if resultado['control_negativo']['no_atribuible'] else 'atribuible'}"),
        (f"- Gate A3: macro-F1 standalone {resultado['gate_a3']['macro_f1_standalone']:.7f} vs "
         f"linea base de la clase mayoritaria "
         f"{resultado['gate_a3']['linea_base_mayoritaria']:.7f} "
         f"({resultado['gate_a3']['clase_mayoritaria']}): coherente = "
         f"{resultado['gate_a3']['coherente']}"),
        (f"- Cap por IC95 de la diferencia: "
         f"{resultado['caps']['ic95_diferencia_contiene_cero']}; cap por determinismo: "
         f"{resultado['caps']['determinismo_no_concluyente']}"),
        (f"- Calibracion: t_fold0 = {resultado['calibracion']['t_fold0_A1_seg']:.1f} s, "
         f"proyeccion {PROYECCION_FACTOR:g} x = {resultado['calibracion']['proyeccion_horas']:.2f}"
         f" h (limite {PROYECCION_MAX_H} h)"),
        "",
        "## Veredicto",
        "",
        f"**{resultado['veredicto']}**: {resultado['siguiente_paso']}",
        "",
        "## Asimetrias declaradas y auditorias",
        "",
        ("- **Sesgo**: en A1 el sesgo es una coordenada mas de FTRL, con su propio acumulador y "
         "su propia penalizacion lambda2; en A2 el intercepto de sklearn no se regulariza. La "
         "asimetria la fija el pre-registro y se reporta, no se corrige."),
        f"- **A2 se aparta de E-011**: {SGD_TWIN['desvio_de_precedente']}.",
        f"- **Cap por IC95**: {GATES_PREREG['cap_ic95_diferencia']}.",
        f"- **Cap por determinismo**: {GATES_PREREG['cap_determinismo']}.",
        ("- **Auditoria del data_hash del manifiesto**: "
         f"{resultado['verificacion']['desvio_manifest']['problema']}. Decision: "
         f"{resultado['verificacion']['desvio_manifest']['decision']}. Los tres hashes "
         "congelados (archivo, array de textos con receta determinista y dict del vectorizador) "
         "si se reproducen; no se reordeno nada."),
        ("- **ngram_range**: la config congelada lo declara como lista y sklearn 1.8.0 exige "
         "tupla; en el fit se convierte a tupla (mismo valor) y el hash del dict congelado usa "
         "la lista."),
        ("- **A2 sin curva por epoca**: SGDClassifier corre sus 10 pasadas internas sin "
         "callback; se reporta el log_loss final por fold (figura, panel 2)."),
        ("- **log_loss de la tabla de complementariedad**: el OOF normalizado L2 no suma 1 por "
         "fila y en sklearn 1.8.0 `log_loss` solo avisa, no renormaliza; por eso la metrica se "
         "calcula sobre el OOF renormalizado con la convencion del proyecto (clip [1e-12, 1] y "
         "L1 por fila), que es la misma de `combinador_oof`. Sin ese paso la perdida se "
         "subestimaba."),
        ("- **p OvR**: FTRL entrega seis sigmoides que no suman 1; el comparador canonico es "
         "la normalizacion L2 por fila y la magnitud de la regla de fallback es el log_loss "
         "OvR medio."),
        "",
        "## Compliance",
        "",
        ("- 0 embeddings, 0 transformers, 0 LLM, 0 datos externos; ninguna particion "
         "reservada se lee ni se escribe; el campeon no se toca."),
        ("- Auditoria AST del propio runner: 0 palabras vigiladas en literales de ruta y 0 "
         "imports de las familias vetadas (vowpal, vw, torch, transformers, gensim, fasttext), "
         "que se comprueban tanto en los literales de ruta como en los modulos importados. La "
         "alternativa declarada en el pre-registro (vowpalwabbit 9.11.8, instalada en el venv) "
         "no se importa ni se usa: el FTRL-Proximal es implementacion propia verificada por "
         "las sondas (a) y (b)."),
        ("- Campeon V30 + plug-in intacto; el challenger es un candidato separado; sin tuning "
         "posterior."),
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
    marca = "V32-E3 ejecutado"
    if marca in RUNLOG.read_text(encoding="utf-8"):
        return False
    a1 = resultado["integracion"]["A1_ftrl"]
    a2 = resultado["integracion"]["A2_sgd_twin"]
    a3 = resultado["integracion"]["A3_ftrl_barajado"]
    fallback = resultado["fallback"]
    linea = [
        "",
        "---",
        "",
        f"## V32-E3 ejecutado: challenger FTRL-Proximal sobre TF-IDF por fold - {FECHA}",
        "",
        (f"- Veredicto: {resultado['veredicto']}. A1 comb {a1['corrected_cv']:.7f} (delta "
         f"{a1['delta_vs_baseline']:+.7f}, {a1['folds_positivos']}/5 folds); A2 control delta "
         f"{a2['delta_vs_baseline']:+.7f}; A3 negativo delta {a3['delta_vs_baseline']:+.7f}."),
        (f"- Baseline recomputado {BASELINE:.7f}; sondas (a)-(d) todas pasan; fallback: "
         f"config_fallback_usada={fallback['config_fallback_usada']}, alpha efectivo "
         f"{fallback['alpha_efectivo']}."),
        (f"- RSS pico {resultado['rss_pico_gb']:.2f} GB (tope {RAM_TOPE_GB:g}); {HILOS} hilos; "
         f"duracion {resultado['duracion_seg']:.0f} s."),
        "- Sin particiones reservadas: sin test, sin holdout, sin envio.",
        (f"- Artefactos: {ruta_relativa(RESULTADOS_JSON)} | {ruta_relativa(ESTADO_JSON)} | "
         f"{ruta_relativa(INFORME)} | {ruta_relativa(FIGURA)} | "
         + " | ".join(ruta_relativa(ruta_oof(brazo)) for brazo in BRAZOS) + " | "
         + ruta_relativa(ruta_oof_crudo("A2_sgd_twin")) + "."),
    ]
    with RUNLOG.open("a", encoding="utf-8") as manejador:
        manejador.write("\n" + "\n".join(linea) + "\n")
    return True


def anotar_experiments_csv(resultado: dict) -> bool:
    """Anade la fila V32-E3 a experiments.csv (22 columnas de la cabecera, sin CR).

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
    if "V32-E3" in {fila[0] for fila in filas[1:] if fila}:
        return False
    a1 = resultado["integracion"]["A1_ftrl"]
    a2 = resultado["integracion"]["A2_sgd_twin"]
    a3 = resultado["integracion"]["A3_ftrl_barajado"]
    fallback = resultado["fallback"]
    if fallback["config_fallback_usada"]:
        nota_alpha = (f"alpha {fallback['alpha_efectivo']} ACTIVADO por la regla de fallback "
                      f"pre-registrada (inicial {ALPHA})")
    else:
        nota_alpha = (f"alpha {fallback['alpha_efectivo']} sin fallback (alternativa declarada "
                      f"{ALPHA_ALTERNATIVO} no activada)")
    resumen = (f"A1 FTRL comb {a1['corrected_cv']:.7f} (delta {a1['delta_vs_baseline']:+.7f}, "
               f"{a1['folds_positivos']}/5 folds, IC95 [{a1['ic95'][0]:+.7f}, "
               f"{a1['ic95'][1]:+.7f}]); A2 gemelo SGD delta {a2['delta_vs_baseline']:+.7f}; "
               f"A3 barajado delta {a3['delta_vs_baseline']:+.7f}; alpha efectivo "
               f"{fallback['alpha_efectivo']}")
    registro = {
        "experiment_id": "V32-E3", "fecha": FECHA, "bloque": "32.3",
        "etapa": "V32-E3 ejecucion: challenger FTRL-Proximal sobre TF-IDF por fold (A1 "
                 "decisional) + gemelo SGD (A2 control) + barajado (A3 negativo)",
        "pregunta": "FTRL-Proximal sobre TF-IDF aporta complementariedad al campeon mas alla "
                    "del SGD ya usado?",
        "hipotesis": "el aprendizaje online proximal con L1-L2 explicito extrae otra region "
                     "del espacio de texto",
        "features": "TF-IDF por fold (word 1-2, min_df 3, max_df 0.5, sublinear_tf) ajustada "
                    "solo en fold-train; 0 embeddings",
        "preprocessing": "vocabulario e idf por fold; sesgo como coordenada regularizada con "
                         "x0=1.0; p OvR normalizado L2 por fila",
        "representacion": "TF-IDF dispersa (nueva en este ciclo: no existia matriz canonica)",
        "modelo": "FTRL-Proximal propio OvR 6 (A1) / SGDClassifier log_loss gemelo (A2) / FTRL "
                  "con etiquetas barajadas (A3)",
        "hiperparametros": f"congelados: {nota_alpha}, lambda1 {LAMBDA1}, "
                           f"lambda2 {LAMBDA2}, beta {BETA}, {EPOCAS} pasadas, {HILOS} hilos, "
                           f"seed {SEMILLA}",
        "validacion": "particion canonica; integracion P6/P7; bootstrap pareado de la "
                      "diferencia A1-A2; gates +0.0015 y 4-5 folds; sondas (a)-(d) y sonda bit "
                      "a bit; calibracion, timeout y RSS declarados",
        "macro_f1_mean": f"{float(np.mean(a1['folds'])):.7f}",
        "macro_f1_std": f"{float(np.std(a1['folds'])):.7f}",
        "weighted_f1_mean": "", "micro_f1_mean": "", "accuracy_mean": "",
        "resultado": resumen,
        "interpretacion": f"veredicto {resultado['veredicto']} por el arbol pre-registrado; A2 "
                          "control de atribucion y A3 control negativo; asimetrias declaradas "
                          "en el informe",
        "decision": resultado["siguiente_paso"],
        "duracion_seg": f"{float(resultado['duracion_seg']):.1f}",
        "artefactos": f"{ruta_relativa(RESULTADOS_JSON)} | {ruta_relativa(INFORME)} | "
                      f"{ruta_relativa(FIGURA)} | " + " | ".join(
                          ruta_relativa(ruta_oof(brazo)) for brazo in BRAZOS),
    }
    if set(registro) != set(cabecera):
        raise Detencion("[STOP] la fila nueva no respeta las columnas de experiments.csv "
                        "(exit 1)", 1)
    with EXPERIMENTS_CSV.open("a", encoding="utf-8", newline="") as manejador:
        escritor = csv.writer(manejador, lineterminator="\n")
        escritor.writerow([registro[columna] for columna in cabecera])
    return True


def resumen_consola(resultado: dict) -> None:
    """Resumen final en consola (maximo 15 lineas).

    Parameters
    ----------
    resultado:
        JSON de resultados.
    """
    integracion = resultado["integracion"]
    print()
    print(f"[V32-E3] fin en {resultado['duracion_seg']:.0f} s | baseline "
          f"{resultado['verificacion']['baseline_recomputado']:.7f} | {HILOS} hilos | sondas "
          f"(a)-(d) pass | RSS pico {resultado['rss_pico_gb']:.2f} GB")
    for brazo in BRAZOS:
        met = integracion[brazo]
        print(f"  {brazo}: comb {met['corrected_cv']:.7f} (delta {met['delta_vs_baseline']:+.7f}, "
              f"{met['folds_positivos']}/5 folds, IC95 [{met['ic95'][0]:+.7f}, "
              f"{met['ic95'][1]:+.7f}])")
    print(f"  control A1 vs A2: {resultado['control_atribucion']['delta_medio']:+.7f} IC95 "
          f"[{resultado['control_atribucion']['ic95'][0]:+.7f}, "
          f"{resultado['control_atribucion']['ic95'][1]:+.7f}] favorable = "
          f"{resultado['control_atribucion']['favorable']}")
    print(f"  control negativo: A3 {resultado['control_negativo']['delta_A3']:+.7f} vs A1 "
          f"{resultado['control_negativo']['delta_A1']:+.7f} -> "
          f"{'NO_ATRIBUIBLE' if resultado['control_negativo']['no_atribuible'] else 'ok'} | "
          f"gate A3 {resultado['gate_a3']['macro_f1_standalone']:.4f} < "
          f"{resultado['gate_a3']['linea_base_mayoritaria']:.4f}")
    print(f"  fallback: config_fallback_usada={resultado['fallback']['config_fallback_usada']} "
          f"alpha_efectivo={resultado['fallback']['alpha_efectivo']} | determinismo no concluyente "
          f"={resultado['determinismo']['no_concluyente']}")
    print(f"  caps: IC95 de la diferencia contiene cero = "
          f"{resultado['caps']['ic95_diferencia_contiene_cero']} | determinismo no concluyente = "
          f"{resultado['caps']['determinismo_no_concluyente']}")
    print(f"  gates: {resultado['gates']}")
    print(f"  VEREDICTO: {resultado['veredicto']} | siguiente: {resultado['siguiente_paso']}")
    print(f"  artefactos: {ruta_relativa(RESULTADOS_JSON)} | {ruta_relativa(INFORME)} | "
          f"{ruta_relativa(FIGURA)}")


# =====================================================================================
# ---- 8. PIPELINE COMPLETO ----
# =====================================================================================
def construir_resultado(inicio: float, prereg: dict, clase: np.ndarray,
                        detalle_constantes: dict, detalle_versiones: dict,
                        detalle_hashes: dict, auditoria: dict, baseline: float,
                        linea_base: dict, estado: dict, sondas: dict, hashes_corpus: dict,
                        oof_crudo: np.ndarray, entrenamiento: dict, integracion: dict,
                        complementariedad: dict, invariantes: dict, control_atribucion: dict,
                        gate_a3: dict, gates: dict, control_negativo: dict, caps: dict,
                        veredicto: str) -> dict:
    """Ensambla el JSON de resultados con config, hashes, sondas, gates y veredicto.

    Parameters
    ----------
    inicio:
        Marca temporal del inicio de la corrida.
    prereg:
        Contenido del pre-registro.
    clase:
        Etiquetas canonicas (para el diagnostico del OOF crudo de A2).
    detalle_constantes:
        Detalle de los asserts de no-deriva.
    detalle_versiones:
        Detalle de las versiones instaladas.
    detalle_hashes:
        Detalle de los hashes verificados.
    auditoria:
        Salida de ``auditar_ast_propio``.
    baseline:
        Baseline recomputado.
    linea_base:
        Linea base de la clase mayoritaria.
    estado:
        Checkpoint final.
    sondas:
        Detalle de las sondas (a)-(d).
    hashes_corpus:
        Hashes recomputados del corpus.
    oof_crudo:
        OOF crudo de A2.
    entrenamiento:
        Detalle de folds por brazo.
    integracion:
        Metricas del combinador por brazo.
    complementariedad:
        Diagnosticos standalone por brazo.
    invariantes:
        Evidencias de no leakage.
    control_atribucion:
        Bootstrap pareado de la diferencia A1-A2.
    gate_a3:
        Resultado del gate A3.
    gates:
        Gates calculados.
    control_negativo:
        Control negativo (precedencia NO_ATRIBUIBLE).
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
        oof_artefactos[brazo] = {"archivo": ruta_relativa(ruta), "sha256": sha256_archivo(ruta)}
    tiempos = {brazo: {str(fold): float(entrenamiento[brazo]["folds"][str(fold)]
                                        ["tiempo_seg"]) for fold in range(N_FOLDS)}
               for brazo in BRAZOS}
    tiempos_fit = {str(fold): float(estado["vectorizador_por_fold"][str(fold)]["tiempo_fit_seg"])
                   for fold in range(N_FOLDS)}
    no_concluyente = bool(
        estado["brazos"]["A1_ftrl"]["sonda_determinismo"]["no_concluyente"]
        or estado["brazos"]["A2_sgd_twin"]["sonda_determinismo"]["no_concluyente"])
    if bool(oof_crudo.any()):
        diagnostico_crudo = diagnostico_crudo_a2(oof_crudo, clase)
        ruta_cruda = ruta_oof_crudo("A2_sgd_twin")
        diagnostico_crudo["artefacto"] = ruta_relativa(ruta_cruda)
        diagnostico_crudo["sha256"] = sha256_archivo(ruta_cruda)
        diagnostico_crudo["uso"] = ("diagnostico exclusivo: no entra en la integracion, en los "
                                   "gates, en el arbol ni en los OOF congelados de los brazos")
    else:
        diagnostico_crudo = {}
    return {
        "fase": "V32-E3 (P3) ejecucion: challenger FTRL-Proximal sobre TF-IDF por fold",
        "fecha": FECHA,
        "commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=False,
                                 capture_output=True, text=True, timeout=30).stdout.strip(),
        "prereg": {"json": ruta_relativa(PREREG_JSON), "sha256": sha256_archivo(PREREG_JSON),
                   "informe": ARCHIVO_PRE_REGISTRO,
                   "commit_congelado": str(prereg["commit"])},
        "verificacion": {
            "constantes": detalle_constantes, "versiones": detalle_versiones,
            "hashes": detalle_hashes["checks"], "hashes_recomputados": detalle_hashes[
                "recomputados"], "hashes_esperados": detalle_hashes["esperados"],
            "hashes_corpus_recomputados": hashes_corpus,
            "receta_hash_textos": prereg["corpus"]["hash_contenido"],
            "desvio_manifest": prereg["corpus"]["desvio_manifest"],
            "baseline_recomputado": baseline, "baseline_esperado": BASELINE, "baseline_ok": True,
            "ids_alineados": True, "etiquetas_alineadas": True,
            "linea_base_mayoritaria": linea_base},
        "compliance": {**auditoria, **COMPLIANCE_CIFRAS,
                       "auditorias_exigidas": COMPLIANCE_AUDITORIAS,
                       "nota": prereg["compliance"]["nota"],
                       "alternativa_no_usada": FTRL["alternativa_no_usada"]},
        "config": {
            "ftrl": FTRL, "alpha_efectivo": float(estado["fallback"]["alpha_efectivo"]),
            "config_fallback_usada": bool(estado["fallback"]["config_fallback_usada"]),
            "a2_sgd_twin": SGD_TWIN, "a3_control_negativo": A3_CONTROL,
            "vectorizador": VECTORIZADOR, "norm_dtype": prereg["representacion"]["norm_dtype"],
            "pesos_plugin": PESOS, "hilos": HILOS,
            "seeds": {"global": SEMILLA, "barajado_etiquetas": SEMILLA_BARAJADO,
                      "orden_datos": f"default_rng({SEMILLA}+fold) por epoca",
                      "sgd_random_state": f"{SEMILLA}+fold"},
            "integracion": {"c_logreg": C_LOGREG, "max_iter": MAX_ITER_LOGREG, "clip": L2_CLIP,
                            "decision": INTEGRACION_PREREG["decision"]},
            "nota_ngram_range": "la config congelada declara ngram_range como lista; sklearn "
                                "1.8.0 exige tupla y en el fit se convierte (mismo valor); el "
                                "hash del dict congelado usa la lista",
            "nota_capas": "vowpalwabbit esta instalado en el venv y NO se importa: el FTRL es "
                          "implementacion propia verificada por las sondas (a) y (b)"},
        "sondas": sondas,
        "fallback": estado["fallback"],
        "vocabulario_idf_por_fold": estado["vectorizador_por_fold"],
        "determinismo": {
            "sonda_A1_ftrl": estado["brazos"]["A1_ftrl"]["sonda_determinismo"],
            "sonda_A2_sgd_twin": estado["brazos"]["A2_sgd_twin"]["sonda_determinismo"],
            "no_concluyente": no_concluyente, "tolerancia_declarada": TOL_DETERMINISMO,
            "variabilidad_referencia": VARIABILIDAD,
            "esperado": DETERMINISMO_PREREG["esperado"], "hilos": DETERMINISMO_PREREG["hilos"]},
        "calibracion": estado["calibracion"],
        "rss_pico_gb": float(estado["rss_pico_gb"]), "ram_tope_gb": RAM_TOPE_GB,
        "rss_medicion": EJECUCION_PREREG["ram_medicion"],
        "gate_a3": gate_a3,
        "sin_leakage": invariantes,
        "entrenamiento": {brazo: {**entrenamiento[brazo], "oof_artefacto": oof_artefactos[brazo]}
                          for brazo in BRAZOS},
        "integracion": integracion,
        "complementariedad": complementariedad,
        "diagnostico_crudo_a2": diagnostico_crudo,
        "control_atribucion": control_atribucion,
        "gates": gates,
        "control_negativo": control_negativo,
        "caps": caps,
        "veredicto": veredicto,
        "siguiente_paso": ARBOL_PREREG[veredicto],
        "duracion_seg": float(time.time() - inicio),
        "duracion_sondas_seg": float(sondas["duracion_seg"]),
        "tiempos_por_fold_seg": tiempos,
        "tiempos_fit_tfidf_por_fold_seg": tiempos_fit,
    }


def ejecutar(verificar: bool) -> int:
    """Pipeline completo: verificacion, sondas, entrenamiento resumible, integracion, gates.

    Parameters
    ----------
    verificar:
        Si True solo verifica y ejecuta las sondas, sin entrenar (imprime
        ``VERIFICACION OK``).

    Returns
    -------
    int
        0 si todo correcto.

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
    clase, fold_idx, v30, ids_canonicos = cargar_canonico()
    textos, hashes_corpus = cargar_corpus(ids_canonicos, clase)
    hashes_ok, detalle_hashes = verificar_hashes(prereg, hashes_corpus)
    if not hashes_ok:
        raise Detencion("[STOP] hashes del pre-registro distintos a los recomputados (exit 2)",
                        2)
    pesos = np.array(PESOS, dtype=np.float64)
    champ_pred = np.argmax(v30 * pesos, axis=1)
    baseline = macro(clase, champ_pred)
    print(f"  baseline primario recomputado: {baseline:.7f} (esperado {BASELINE:.7f})", flush=True)
    if abs(baseline - BASELINE) > ABORTO_BASELINE:
        raise Detencion("[STOP] el baseline canonico no reproduce (exit 2)", 2)
    linea_base = linea_base_mayoritaria(clase)
    print(f"  corpus: {len(textos)} textos, ids en el MISMO orden que el canonico, etiquetas "
          "coincidentes; hashes de archivo, array de textos y vectorizador OK", flush=True)
    print(f"  compliance AST: {auditoria['literales_de_ruta_revisados']} literales de ruta y "
          f"{auditoria['importaciones_revisadas']} modulos, 0 violaciones", flush=True)
    sondas = ejecutar_sondas()
    print(f"  sondas (a)-(d) en {sondas['duracion_seg']:.1f} s: todas pasan", flush=True)
    if verificar:
        print("VERIFICACION OK", flush=True)
        return 0

    estado = cargar_estado()
    estado["sondas"] = {nombre: bool(sondas[nombre]["pass"]) for nombre in sondas
                        if isinstance(sondas[nombre], dict)}
    oofs, oof_crudo = entrenar_con_checkpoints(textos, clase, fold_idx, estado)
    determinismo = ejecutar_sondas_determinismo(textos, clase, fold_idx, oofs, estado)
    invariantes = verificar_invariantes(estado, clase, fold_idx, oofs)
    gate_a3 = verificar_gate_a3(oofs["A3_ftrl_barajado"], clase, linea_base, estado)

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
    diferencia = bootstrap_delta(clase, preds_comb["A2_sgd_twin"], preds_comb["A1_ftrl"])
    control_atribucion = {
        "delta_A1": integracion["A1_ftrl"]["delta_vs_baseline"],
        "delta_A2": integracion["A2_sgd_twin"]["delta_vs_baseline"],
        "diferencia_puntual": (integracion["A1_ftrl"]["delta_vs_baseline"]
                               - integracion["A2_sgd_twin"]["delta_vs_baseline"]),
        "delta_medio": diferencia["delta_medio"], "ic95": diferencia["ic95"],
        "p_delta_positivo": diferencia["p_delta_positivo"],
        "ic95_sin_cero": diferencia["ic95_sin_cero"], "criterio": GATES_PREREG["control"],
        "favorable": bool(integracion["A1_ftrl"]["delta_vs_baseline"]
                          > integracion["A2_sgd_twin"]["delta_vs_baseline"]
                          and diferencia["ic95_sin_cero"])}
    gates, control_negativo, caps, veredicto = calcular_gates(
        True, integracion, control_atribucion, gate_a3, invariantes, determinismo[
            "no_concluyente"])
    print(f"  veredicto: {veredicto}", flush=True)

    entrenamiento = {brazo: {"folds": estado["brazos"][brazo]["folds"]} for brazo in BRAZOS}
    resultado = construir_resultado(inicio, prereg, clase, detalle_constantes,
                                    detalle_versiones, detalle_hashes, auditoria, baseline,
                                    linea_base, estado, sondas, hashes_corpus, oof_crudo,
                                    entrenamiento, integracion, complementariedad, invariantes,
                                    control_atribucion, gate_a3, gates, control_negativo, caps,
                                    veredicto)
    escribir_json(RESULTADOS_JSON, resultado)
    print(f"  resultados: {ruta_relativa(RESULTADOS_JSON)}", flush=True)
    figura_curvas(entrenamiento, integracion, FIGURA)
    escribir_informe(resultado)
    print(f"  RUNLOG.md: {'anotado' if anotar_runlog(resultado) else 'ya estaba'}", flush=True)
    print(f"  experiments.csv: "
          f"{'fila V32-E3 anadida' if anotar_experiments_csv(resultado) else 'fila ya estaba'}",
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
        description="V32-E3 (P3): implementacion del pre-registro congelado del challenger "
                    "FTRL-Proximal sobre TF-IDF ajustada por fold, con gemelo SGD y control "
                    "barajado.")
    analizador.add_argument("--verificar", action="store_true",
                            help="solo verifica constantes, versiones, hashes, corpus, "
                                 "auditoria AST, baseline y sondas (a)-(d) (no entrena; "
                                 "imprime VERIFICACION OK)")
    return analizador.parse_args()


def main() -> int:
    """Entrada del script con manejo de los codigos de salida pre-registrados.

    Returns
    -------
    int
        Codigo de salida (0 correcto; 2, 4, 5, 6, 7, 8 para las guardas; 1 inesperado).
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
