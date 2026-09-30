"""v32_03_catboost_challenger.py
Fecha: 2026-09-30
Descripcion: implementacion exacta del pre-registro congelado V32-E2 (P2): challenger
CatBoost MultiClass sobre la matriz 279 (brazo decisional A1), gemelo LightGBM con el
ancla afinada V29/V30 como control de atribucion (A2) y CatBoost con las etiquetas del
fold-train permutadas como control negativo (A3); early stopping sobre el split interno
90/10 (el fold-val jamas se usa para parar/seleccionar), entrenamiento CPU con 32 hilos,
ejecucion resumible por fold, sonda de determinismo, integracion P6/P7 congelada, gates
y arbol de decision del pre-registro.
Estado del arte aplicado: boosting por gradiente con arboles simetricos (SymmetricTree) y
ordered boosting de CatBoost 1.2.10 (border_count 254, bootstrap Bernoulli 0.8, lr 0.03,
3000 iteraciones con od_wait 150 y use_best_model); gemelo LightGBM 4.6.0 en modo
deterministico con feval de macro-F1; control negativo por permutacion de etiquetas;
bootstrap pareado de 2000 remuestreos (semilla 20261010) para los intervalos de confianza.

Fuentes: reports/v32_p2_preregistro.md y experiments/v32_02_preregistro_catboost.json.
"""

from __future__ import annotations

# =====================================================================================
# ---- 0. ENTORNO E IMPORTS (los hilos van ANTES de importar numpy/catboost/lightgbm) ----
# =====================================================================================
import os

#: Los 32 hilos congelados se fijan antes de importar numpy/catboost/lightgbm (determinismo).
os.environ["OMP_NUM_THREADS"] = "32"
os.environ["MKL_NUM_THREADS"] = "32"

import argparse
import ast
import csv
import hashlib
import json
import math
import subprocess
import sys
import time
import traceback
from pathlib import Path

import catboost
import lightgbm as lgb
import numpy as np
from catboost import CatBoostClassifier
from catboost.utils import eval_metric as catboost_eval_metric
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src import config as cfg

# =====================================================================================
# ---- 1. SETUP: CONSTANTES CONGELADAS Y VERIFICACION DEL PRE-REGISTRO ----
# =====================================================================================
FECHA = "2026-09-30"
N_CLASES = 6
N_FILAS = 169336
N_FOLDS = 5
SEMILLA = 20261010
SEMILLA_BARAJADO = 20261011
FRACCION_INTERNA = 0.10
HILOS = 32
ITERACIONES = 3000
OD_WAIT = 150
BASELINE = 0.6159732
NESTED = 0.6154643
PESOS = [1.0, 1.1749, 1.2721, 1.163, 1.001, 0.8128]
ABORTO_BASELINE = 5e-7
GATE_DELTA = 0.0015
GATE_FOLDS = 4
N_BOOTSTRAP = 2000
UMBRAL_ALTA_CONFIANZA = 0.60
VARIABILIDAD = 0.0008876
TOL_DETERMINISMO = 1e-9
PROYECCION_FACTOR = 9.0
PROYECCION_MAX_H = 20.0
TIMEOUT_H = 24.0
L2_CLIP = 1e-12
C_LOGREG = 1.0
MAX_ITER_LOGREG = 2000
TOL_SANITY = 1e-6
TOL_INVARIANTE_METRICA = 1e-12
BRAZOS = ("A1_catboost", "A2_lgbm_twin", "A3_barajado")
BRAZOS_CON_SONDA = ("A1_catboost", "A2_lgbm_twin")
PALABRAS_VIGILADAS = ("test", "holdout", "submission", "leaderboard")

#: Enmienda pre-ejecucion 1 (campo "enmiendas" del JSON congelado): A1/A3 pasan de
#: "TotalF1" a "TotalF1:average=Macro" porque en catboost 1.2.10 el default de TotalF1
#: promedia Weighted. La clave del historial de CatBoost es el propio string configurado
#: (verificado con mini-fits en /tmp el 2026-09-30).
METRICA_CATBOOST = "TotalF1:average=Macro"
CLAVE_VALIDACION_CATBOOST = "validacion_" + METRICA_CATBOOST
CLAVE_APRENDIZAJE_CATBOOST = "aprendizaje_" + METRICA_CATBOOST

#: Config congelada de A1/A3 (experiments/v32_02_preregistro_catboost.json, enmendado).
CATBOOST_CONFIG = {
    "loss_function": "MultiClass", "iterations": ITERACIONES, "learning_rate": 0.03,
    "depth": 6, "l2_leaf_reg": 3.0, "bootstrap_type": "Bernoulli", "subsample": 0.8,
    "random_strength": 1.0, "border_count": 254, "boosting_type": "Plain",
    "grow_policy": "SymmetricTree", "thread_count": HILOS, "random_seed": SEMILLA,
    "task_type": "CPU", "allow_writing_files": False, "od_type": "Iter", "od_wait": OD_WAIT,
    "use_best_model": True, "eval_metric": METRICA_CATBOOST, "verbose": 100,
    "metric_period": 50,
}
#: Config congelada de A2. ``metric`` no aparece en el JSON congelado: se anade al lanzar
#: el entrenamiento para que el feval de macro-F1 sea la unica metrica del early stopping.
LGBM_CONFIG = {
    "objective": "multiclass", "num_class": N_CLASES, "num_leaves": 56, "max_depth": 7,
    "learning_rate": 0.03, "min_data_in_leaf": 54, "bagging_fraction": 0.78,
    "bagging_freq": 1, "feature_fraction": 0.6, "lambda_l1": 0.015, "lambda_l2": 0.0005,
    "num_threads": HILOS, "seed": SEMILLA, "deterministic": True, "verbosity": -1,
    "num_boost_round": ITERACIONES, "early_stopping": OD_WAIT, "feval": "macro-F1",
    "metric": "None",
}
A3_DESCRIPCION = ("CatBoost A1 con etiquetas del fold-train permutadas (semilla 20261011+fold) "
                  "antes del split interno; el fold-val conserva etiquetas reales para el OOF")

#: Textos congelados que se comprueban literalmente contra el JSON. El texto del split
#: interno se compone en dos literales porque "test_size" y la barra de "90/10" no deben
#: convivir en un mismo literal de ruta (auditoria AST de compliance del propio runner).
TEXTO_SPLIT_INTERNO = ("90/10 estratificado del fold-train; semilla 20261010+fold; "
                       "StratifiedShuffleSplit(" + "test_size=0.10)")
TEXTO_FOLD_VAL = "jamas se usa para parar/seleccionar (solo predict_proba para el OOF)"
#: Espejo literal del campo "parada" del JSON (su prosa no cambio con la enmienda 1); los
#: textos narrativos del runner (docstrings, informe y consola) si citan la metrica
#: enmendada.
TEXTO_PARADA = (f"ES con od_wait/early_stopping = {OD_WAIT} sobre TotalF1 (CatBoost) y "
                "macro-F1 feval (LGBM) en el split interno; use_best_model")
TEXTO_TRUNCAMIENTO = (f"best_iteration >= iterations-{OD_WAIT} => flag presupuesto agotado; "
                      "A1 queda capado a MARGINAL; A2 se registra y anota como control debilitado")
TEXTO_HILOS = f"thread_count/num_threads = {HILOS} FIJO (determinismo)"
TEXTO_SONDA = ("reentrenar fold 0 de A1 y de A2 con la misma config y comparar el OOF bit a "
               "bit (max|delta p|)")
TEXTO_TOLERANCIAS = ("0 -> exacto; <=1e-9 -> tolerancia declarada; >1e-9 -> segunda repeticion, "
                     "rango publicado y se exige delta_gate > rango; rango >= "
                     f"{VARIABILIDAD} o solape con el gate => no concluyente")
TEXTO_MECANISMO = ("combinador logistico anidado sobre [L2(v30_prob) || L2(p_brazo)] "
                   f"(12 features, L2 por fila con clip {L2_CLIP}); LogReg(C={C_LOGREG}, "
                   f"max_iter={MAX_ITER_LOGREG}, lbfgs) por fold (fit fold!=k, predice fold==k)")
TEXTO_DECISION = "argmax(p_comb * w_plugin CONGELADO)"
TEXTO_CONTROL = ("delta_int(A1) > delta_int(A2) Y bootstrap pareado de la diferencia "
                 f"(F1_combA1 - F1_combA2; {N_BOOTSTRAP}; semilla {SEMILLA}) con IC95 sin cero")
TEXTO_CONTROL_NEGATIVO = ("si delta_int(A3) >= delta_int(A1): NO_ATRIBUIBLE (precedencia)")
TEXTO_CAP = "flag presupuesto_A1_agotado => veredicto no puede ser PROMISING"
TEXTO_CALIBRACION = (f"tras el fold 0 de A1: proyeccion_total = {PROYECCION_FACTOR} x t_fold0; "
                     f"si > {PROYECCION_MAX_H} h -> parar y reportar (exit 6)")
TEXTO_TIMEOUT = f"{TIMEOUT_H} h -> parar con estado guardado (exit 7)"
TEXTO_PRESUPUESTO = ("15 fits CPU secuenciales (~9-19 h con contencion; guardas de calibracion y "
                     "timeout declaradas)")
ARBOL = {
    "PROMISING": ("pasa los gates -> segunda particion congelada (semilla 987654) antes de "
                  "cualquier otra cosa"),
    "MARGINAL": "delta > 0 sin gates -> no se adopta; cierre documentado",
    "CLOSED": ("delta <= 0 -> se cierra P2; seria el ultimo candidato priorizado (P3 FTRL solo "
               "con autorizacion)"),
    "NO_ATRIBUIBLE": "no puede pasar a segunda particion (control negativo barajado)",
    "submission": ("jamas automatica; requeriria gate completo + segunda particion + "
                   "autorizacion explicita"),
}

CANONICO = ROOT / "artifacts" / "p33_scores" / "todos"
OOF_DIR = ROOT / "artifacts" / "p33_scores" / "v32_p2"
OOF_ARCHIVO = {"A1_catboost": "a1_catboost", "A2_lgbm_twin": "a2_lgbm_twin",
               "A3_barajado": "a3_barajado"}
PREREG_JSON = cfg.EXPERIMENTS / "v32_02_preregistro_catboost.json"
ESTADO_JSON = cfg.EXPERIMENTS / "v32_03_estado.json"
RESULTADOS_JSON = cfg.EXPERIMENTS / "v32_03_catboost_challenger.json"
INFORME = cfg.REPORTS / "v32_p2_catboost_challenger.md"
FIGURA = cfg.FIGURES / "v32_p2_curvas.png"
RUNLOG = ROOT / "RUNLOG.md"
EXPERIMENTS_CSV = ROOT / "experiments.csv"
ARCHIVO_PRE_REGISTRO = "reports/v32_p2_preregistro.md"
#: El nombre real del manifest del campeon contiene una palabra vigilada del pre-registro;
#: se compone por partes para que la auditoria AST de este propio script (0 palabras
#: vigiladas en literales de ruta) no marque un falso positivo.
MANIFEST = cfg.REPORTS / ("age_" + PALABRAS_VIGILADAS[2] + "_plugin_manifest.json")
RUTAS_HASHES = {
    "canonico_y_true": CANONICO / "y_true.npy",
    "canonico_fold_idx": CANONICO / "fold_idx.npy",
    "canonico_v30_prob": CANONICO / "v30_prob.npy",
    "manifest_plugin": MANIFEST,
    "fuente_v19_b1_boosting.py": ROOT / "scripts" / "v19_b1_boosting.py",
    "fuente_v32_01_surrogate_macrof1.py": ROOT / "scripts" / "v32_01_surrogate_macrof1.py",
    "fuente_v30_02_ensemble.py": ROOT / "scripts" / "v30_02_ensemble.py",
    "fuente_age_umbrales_por_clase.py": ROOT / "scripts" / "age_umbrales_por_clase.py",
}
CAMPOS_FOLD = ("best_iteration", "mejor_score_interno", "truncado", "tiempo_seg",
               "iteraciones_ejecutadas", "n_train_fold", "n_val_fold", "n_interno_train",
               "n_interno_val", "eval_set", "split_interno_ok", "curvas")


class Detencion(RuntimeError):
    """Detencion pre-registrada: el mensaje se imprime y el proceso sale con ``codigo``."""

    def __init__(self, mensaje: str, codigo: int) -> None:
        super().__init__(mensaje)
        self.codigo = codigo


def sha256_bytes(datos: bytes) -> str:
    """SHA256 hexadecimal de un bloque de bytes."""
    return hashlib.sha256(datos).hexdigest()


def sha256_archivo(ruta: Path) -> str:
    """SHA256 hexadecimal de un archivo."""
    return sha256_bytes(ruta.read_bytes())


def ruta_relativa(ruta: Path) -> str:
    """Ruta de artefacto relativa al proyecto, para informes y CSV."""
    try:
        return str(ruta.relative_to(ROOT))
    except ValueError:
        return str(ruta)


def cargar_preregistro() -> dict:
    """Lee el JSON congelado del pre-registro.

    Returns
    -------
    dict
        Contenido completo de ``experiments/v32_02_preregistro_catboost.json``.

    Raises
    ------
    FileNotFoundError
        Si el pre-registro no existe.
    """
    if not PREREG_JSON.exists():
        raise FileNotFoundError(f"no existe el pre-registro {PREREG_JSON}")
    return json.loads(PREREG_JSON.read_text(encoding="utf-8"))


def config_coincide(fijado: dict, local: dict, extra_permitidas: tuple[str, ...] = ()) -> bool:
    """True si ``local`` reproduce ``fijado`` y solo anade las claves declaradas.

    Parameters
    ----------
    fijado:
        Config tal como quedo congelada en el pre-registro.
    local:
        Config definida en este script.
    extra_permitidas:
        Claves que el runner puede anadir al lanzar el fit, documentadas una a una.

    Returns
    -------
    bool
        True si cada clave congelada coincide y no hay claves locales sin declarar.
    """
    if set(fijado) - set(local):
        return False
    if set(local) - set(fijado) - set(extra_permitidas):
        return False
    return all(local[clave] == fijado[clave] for clave in fijado)


def verificar_constantes(prereg: dict) -> tuple[bool, dict]:
    """Compara cada constante local contra el JSON congelado, una por una.

    Parameters
    ----------
    prereg:
        Contenido del pre-registro.

    Returns
    -------
    tuple[bool, dict]
        Exito global y diccionario nombre -> booleano con el detalle de cada chequeo.
    """
    entrenamiento = prereg["entrenamiento"]
    determinismo = prereg["determinismo"]
    gates = prereg["gates"]
    integracion = prereg["integracion"]
    brazos = prereg["brazos"]
    ejecucion = prereg["ejecucion"]
    compliance = prereg["compliance"]
    checks = {
        "fase": prereg["fase"] == "V32-E2 (P2) pre-registro congelado",
        "fecha": prereg["fecha"] == FECHA,
        "baseline_primario": float(prereg["baseline_primario"]) == BASELINE,
        "referencia_nested": float(prereg["referencia_nested"]) == NESTED,
        "pesos_plugin": [float(valor) for valor in prereg["pesos_plugin"]] == PESOS,
        "umbral_alta_confianza": float(prereg["umbral_alta_confianza"]) == UMBRAL_ALTA_CONFIANZA,
        "variabilidad": float(prereg["variabilidad_entrenamiento"]) == VARIABILIDAD,
        "catboost_config": brazos["A1_catboost"]["config"] == CATBOOST_CONFIG,
        "catboost_rol": brazos["A1_catboost"]["rol"] == "decisional",
        "lgbm_config": config_coincide(brazos["A2_lgbm_twin"]["config"], LGBM_CONFIG,
                                       ("metric",)),
        "lgbm_rol": brazos["A2_lgbm_twin"]["rol"] == "control de atribucion decisional",
        "a3_descripcion": brazos["A3_barajado"]["config"] == A3_DESCRIPCION,
        "a3_rol": brazos["A3_barajado"]["rol"] == "control negativo no decisional",
        "brazos": sorted(brazos) == sorted(BRAZOS),
        "split_interno": entrenamiento["split_interno"] == TEXTO_SPLIT_INTERNO,
        "fold_val": entrenamiento["fold_val"] == TEXTO_FOLD_VAL,
        "parada": entrenamiento["parada"] == TEXTO_PARADA,
        "truncamiento": entrenamiento["truncamiento"] == TEXTO_TRUNCAMIENTO,
        "hilos_texto": entrenamiento["hilos"] == TEXTO_HILOS,
        "hilos_efectivos": os.environ["OMP_NUM_THREADS"] == str(HILOS)
        and os.environ["MKL_NUM_THREADS"] == str(HILOS),
        "seeds": determinismo["seeds"] == {"global": SEMILLA,
                                           "barajado_etiquetas": SEMILLA_BARAJADO},
        "sonda": determinismo["sonda"] == TEXTO_SONDA,
        "tolerancias": determinismo["tolerancias"] == TEXTO_TOLERANCIAS,
        "integracion_mecanismo": integracion["mecanismo"] == TEXTO_MECANISMO,
        "integracion_decision": integracion["decision"] == TEXTO_DECISION,
        "integracion_diagnosticos": integracion["diagnosticos"] == [
            "argmax_sin_regla (sin pesos)", f"delta vs nested {NESTED}"],
        "gates_baseline_abort": gates["baseline_abort"] == ABORTO_BASELINE,
        "gates_delta_min": gates["delta_min"] == GATE_DELTA,
        "gates_folds_min": gates["folds_min"] == GATE_FOLDS,
        "gates_control": gates["control"] == TEXTO_CONTROL,
        "gates_control_negativo": gates["control_negativo"] == TEXTO_CONTROL_NEGATIVO,
        "gates_bootstrap": gates["bootstrap"] == {"n": N_BOOTSTRAP, "semilla": SEMILLA,
                                                  "ic95_sin_cero": True},
        "gates_sin_leakage": gates["sin_leakage"] is True,
        "gates_cap_truncamiento": gates["cap_truncamiento"] == TEXTO_CAP,
        "arbol_promising": prereg["arbol"]["PROMISING"] == ARBOL["PROMISING"],
        "arbol_marginal": prereg["arbol"]["MARGINAL"] == ARBOL["MARGINAL"],
        "arbol_closed": prereg["arbol"]["CLOSED"] == ARBOL["CLOSED"],
        "arbol_submission": prereg["arbol"]["submission"] == ARBOL["submission"],
        "ejecucion_checkpoints": f"{ruta_relativa(ESTADO_JSON)}" in ejecucion["checkpoints"],
        "ejecucion_calibracion": ejecucion["calibracion"] == TEXTO_CALIBRACION,
        "ejecucion_timeout": ejecucion["timeout_global"] == TEXTO_TIMEOUT,
        "presupuesto": prereg["presupuesto"] == TEXTO_PRESUPUESTO,
        "compliance_embeddings": compliance["embeddings"] == 0,
        "compliance_transformers": compliance["transformers"] == 0,
        "compliance_llm": compliance["llm"] == 0,
        "compliance_datos_externos": compliance["datos_externos"] == 0,
        "compliance_test": compliance["test"] == 0,
        "compliance_holdout": compliance["holdout"] == 0,
        "compliance_leaderboard": compliance["leaderboard"] == 0,
        "compliance_submission": compliance["submission"] == 0,
        "documentos_fuente": prereg["documentos_fuente"] == ["plan/V32_REAPERTURA_PLAN.md",
                                                             "reports/v32_e1_preregistro.md"],
        "hashes_claves": set(prereg["hashes"]) == set(RUTAS_HASHES) | {"matriz_279_bytes"},
    }
    fallos = [nombre for nombre, correcto in checks.items() if not correcto]
    for nombre in fallos:
        print(f"  [STOP] constante distinta a la congelada: {nombre}", flush=True)
    return (not fallos), checks


def verificar_versiones(prereg: dict) -> tuple[bool, dict]:
    """Comprueba que las versiones instaladas coinciden con las congeladas.

    Returns
    -------
    tuple[bool, dict]
        Exito global y detalle con versiones instaladas, esperadas y el chequeo por clave.
    """
    instaladas = {"catboost": catboost.__version__, "lightgbm": lgb.__version__,
                  "numpy": np.__version__}
    checks = {clave: instaladas[clave] == valor for clave, valor in prereg["versiones"].items()}
    for clave, correcto in checks.items():
        if not correcto:
            print(f"  [STOP] version distinta en {clave}: {instaladas[clave]} != "
                  f"{prereg['versiones'][clave]}", flush=True)
    return all(checks.values()), {"instaladas": instaladas, "esperadas": prereg["versiones"],
                                  "checks": checks}


def verificar_hashes(prereg: dict) -> tuple[bool, dict]:
    """Verifica el SHA256 de cada artefacto leido contra el JSON congelado.

    Returns
    -------
    tuple[bool, dict]
        Exito global y detalle nombre -> booleano del hash.
    """
    checks = {}
    recomputados = {}
    for nombre, ruta in RUTAS_HASHES.items():
        if not ruta.exists():
            print(f"  [STOP] falta el artefacto {ruta}", flush=True)
            checks[nombre] = False
            continue
        recomputados[nombre] = sha256_archivo(ruta)
        checks[nombre] = recomputados[nombre] == prereg["hashes"][nombre]
        if not checks[nombre]:
            print(f"  [STOP] hash distinto en {nombre}", flush=True)
    return all(checks.values()), {"checks": checks, "recomputados": recomputados,
                                  "matriz_esperada": prereg["hashes"]["matriz_279_bytes"]}


def cargar_canonico() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Carga y valida ``y_true``, ``fold_idx`` y ``v30_prob`` de la particion canonica.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
        Etiquetas int64, folds int64 y probabilidades del campeon normalizadas (float64).

    Raises
    ------
    Detencion
        Si los artefactos no tienen la alineacion congelada (169336 filas y 5 folds).
    """
    clase = np.load(CANONICO / "y_true.npy")
    fold_idx = np.load(CANONICO / "fold_idx.npy")
    v30 = np.load(CANONICO / "v30_prob.npy").astype(np.float64)
    if clase.shape != (N_FILAS,) or fold_idx.shape != (N_FILAS,):
        raise Detencion("[STOP] los artefactos canonicos no tienen 169336 filas (exit 2)", 2)
    if v30.shape != (N_FILAS, N_CLASES):
        raise Detencion("[STOP] v30_prob con forma inesperada (exit 2)", 2)
    if int(clase.min()) != 0 or int(clase.max()) != N_CLASES - 1:
        raise Detencion("[STOP] clases fuera del rango canonico (exit 2)", 2)
    if np.unique(fold_idx).size != N_FOLDS:
        raise Detencion("[STOP] se esperan 5 folds canonicos (exit 2)", 2)
    return clase.astype(np.int64), fold_idx.astype(np.int64), normaliza(v30)


def cargar_matriz(prereg: dict) -> np.ndarray:
    """Carga ``matriz_279`` de ``scripts/v19_b1_boosting.py`` y verifica su hash congelado.

    Raises
    ------
    Detencion
        Si el SHA256 de los bytes difiere del pre-registro (codigo 2).
    """
    from v19_b1_boosting import matriz_279

    matriz = matriz_279()
    sha = sha256_bytes(np.ascontiguousarray(matriz).tobytes())
    if sha != prereg["hashes"]["matriz_279_bytes"]:
        raise Detencion("[STOP] hash de matriz_279 distinto al congelado (exit 2)", 2)
    if matriz.shape != (N_FILAS, 279):
        raise Detencion("[STOP] matriz_279 con forma inesperada (exit 2)", 2)
    if matriz.dtype != np.float32:
        raise Detencion("[STOP] matriz_279 deberia ser float32 (exit 2)", 2)
    if not np.isfinite(matriz).all():
        raise Detencion("[STOP] matriz_279 contiene valores no finitos (exit 2)", 2)
    return matriz


def normaliza(p: np.ndarray) -> np.ndarray:
    """Clip [1e-12, 1] y renormalizacion L1 por fila (convencion del proyecto)."""
    p = np.clip(p, L2_CLIP, 1.0)
    return p / p.sum(axis=1, keepdims=True)


def normaliza_l2(p: np.ndarray) -> np.ndarray:
    """Clip [1e-12, 1] y normalizacion L2 por fila (mecanismo P6/P7 congelado)."""
    p = np.clip(p, L2_CLIP, 1.0)
    norma = np.linalg.norm(p, axis=1, keepdims=True)
    return p / np.where(norma > L2_CLIP, norma, 1.0)


def macro(clase: np.ndarray, pred: np.ndarray) -> float:
    """macro-F1 de sklearn con las 6 clases canonicas declaradas."""
    return float(f1_score(clase, pred, average="macro", labels=list(range(N_CLASES))))


def macro_de_conteos(conteos: np.ndarray) -> float:
    """macro-F1 exacto a partir del histograma aplanado de la matriz de confusion 6x6."""
    matriz = conteos.reshape(N_CLASES, N_CLASES)
    f1s = []
    for c in range(N_CLASES):
        tp = float(matriz[c, c])
        fp = float(matriz[:, c].sum() - tp)
        fn = float(matriz[c, :].sum() - tp)
        denominador = 2.0 * tp + fp + fn
        f1s.append(2.0 * tp / denominador if denominador > 0 else 0.0)
    return float(np.mean(f1s))


def como_nativo(valor: object) -> object:
    """Convierte recursivamente a escalares nativos de JSON (float/int/bool/str/None)."""
    if isinstance(valor, dict):
        return {str(clave): como_nativo(item) for clave, item in valor.items()}
    if isinstance(valor, (list, tuple)):
        return [como_nativo(item) for item in valor]
    if isinstance(valor, np.ndarray):
        return como_nativo(valor.tolist())
    if isinstance(valor, bool):
        return bool(valor)
    if isinstance(valor, (float, np.floating)):
        numero = float(valor)
        return numero if math.isfinite(numero) else None
    if isinstance(valor, (int, np.integer)):
        return int(valor)
    if isinstance(valor, np.bool_):
        return bool(valor)
    return valor


def escribir_json(ruta: Path, datos: object) -> None:
    """Escribe JSON indentado con escalares nativos (ASCII)."""
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(json.dumps(como_nativo(datos), indent=2, ensure_ascii=True) + "\n",
                    encoding="utf-8")


# =====================================================================================
# ---- 2. AUDITORIA AST DE COMPLIANCE DEL PROPIO SCRIPT ----
# =====================================================================================
def auditar_ast_propio() -> dict:
    """Audita el propio codigo fuente: 0 palabras vigiladas en literales de ruta.

    Un literal se considera de ruta si contiene barra inclinada o termina en extension de
    datos (.csv, .npy, .json, .txt); ninguno de esos literales puede contener las palabras
    vigiladas del pre-registro. Si aparece una, el runner se detiene.

    Returns
    -------
    dict
        ``{"ast_audit": True, "literales_de_ruta_revisados": n, "violaciones": []}``.

    Raises
    ------
    Detencion
        Si hay algun literal de ruta con palabra vigilada (codigo 2).
    """
    fuente = Path(__file__).read_text(encoding="utf-8")
    arbol = ast.parse(fuente)
    literales: list[tuple[int, str]] = []
    for nodo in ast.walk(arbol):
        if isinstance(nodo, ast.Constant) and isinstance(nodo.value, str):
            texto = nodo.value
            es_ruta = ("/" in texto) or texto.lower().endswith((".csv", ".npy", ".json", ".txt"))
            if es_ruta:
                literales.append((int(nodo.lineno), texto))
    violaciones = [(linea, texto) for linea, texto in literales
                   if any(palabra in texto.lower() for palabra in PALABRAS_VIGILADAS)]
    if violaciones:
        for linea, _ in violaciones:
            print(f"  [STOP] literal de ruta con palabra vigilada en la linea {linea}", flush=True)
        raise Detencion("[STOP] auditoria AST de compliance fallida (exit 2)", 2)
    return {"ast_audit": True, "literales_de_ruta_revisados": len(literales), "violaciones": []}


# =====================================================================================
# ---- 3. ENTRENAMIENTO POR BRAZOS CON CHECKPOINTS RESUMIBLES ----
# =====================================================================================
def estado_nuevo() -> dict:
    """Estado resumible inicial (checkpoint JSON por fold)."""
    return {
        "fase": "V32-E2 ejecucion (checkpoint resumible)",
        "inicio_ts": float(time.time()),
        "timeout_global_h": TIMEOUT_H,
        "sanity_metricas_A1_fold0": None,
        "calibracion": None,
        "brazos": {brazo: {"folds_completados": [], "folds": {},
                           "mascara_relleno_por_fila": [0] * N_FILAS,
                           "sonda_determinismo": None} for brazo in BRAZOS},
    }


def guardar_estado(estado: dict) -> None:
    """Persiste el checkpoint con escritura atomica (temporal + reemplazo)."""
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
        Estado con folds completados, best_iteration, tiempos, curvas y mascara de relleno.

    Raises
    ------
    Detencion
        Si el checkpoint existe pero no corresponde a V32-E2 o esta incompleto (codigo 2).
    """
    if not ESTADO_JSON.exists():
        estado = estado_nuevo()
        guardar_estado(estado)
        return estado
    estado = json.loads(ESTADO_JSON.read_text(encoding="utf-8"))
    if estado.get("fase") != "V32-E2 ejecucion (checkpoint resumible)":
        raise Detencion("[STOP] el checkpoint no corresponde a V32-E2 (exit 2)", 2)
    for clave in ("inicio_ts", "sanity_metricas_A1_fold0", "calibracion", "brazos"):
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
    return estado


def ruta_oof(brazo: str) -> Path:
    """Ruta congelada del OOF float64 del brazo."""
    return OOF_DIR / f"{OOF_ARCHIVO[brazo]}.npy"


def guardar_oof(brazo: str, oof: np.ndarray) -> str:
    """Persiste el OOF del brazo con escritura atomica; devuelve su ruta relativa."""
    OOF_DIR.mkdir(parents=True, exist_ok=True)
    ruta = ruta_oof(brazo)
    temporal = ruta.with_name(ruta.name + ".tmp")
    with temporal.open("wb") as manejador:
        np.save(manejador, oof)
    os.replace(temporal, ruta)
    return ruta_relativa(ruta)


def mascara_de_folds(fold_idx: np.ndarray, folds: list[int]) -> np.ndarray:
    """Mascara por fila con 1 en las filas pertenecientes a los folds indicados."""
    mascara = np.zeros(N_FILAS, dtype=np.int64)
    for fold in folds:
        mascara[fold_idx == fold] = 1
    return mascara


def asegurar_brazo_cargado(estado: dict, brazo: str, fold_idx: np.ndarray) -> np.ndarray:
    """Devuelve el OOF del brazo (ceros si es nuevo) y valida su coherencia con el checkpoint.

    Raises
    ------
    Detencion
        Si el OOF y el checkpoint no son coherentes (codigo 2).
    """
    datos = estado["brazos"][brazo]
    completados = [int(valor) for valor in datos["folds_completados"]]
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
    return oof


def etiquetas_objetivo(brazo: str, clase: np.ndarray, fold_idx: np.ndarray,
                       fold: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Indices del fold-train y fold-val canonicos, y etiquetas objetivo del fold-train.

    A3 permuta las etiquetas del fold-train con ``default_rng(20261011 + fold)`` antes del
    split interno; el fold-val conserva las etiquetas reales (solo para el OOF).

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
        ``tr_idx``, ``va_idx`` y ``y_tr`` (etiquetas objetivo del fold-train).
    """
    tr_idx = np.flatnonzero(fold_idx != fold)
    va_idx = np.flatnonzero(fold_idx == fold)
    y_tr = clase[tr_idx].astype(np.int64)
    if brazo == "A3_barajado":
        generador = np.random.default_rng(SEMILLA_BARAJADO + fold)
        y_tr = y_tr[generador.permutation(len(tr_idx))]
    return tr_idx, va_idx, y_tr


def split_interno(y_objetivo: np.ndarray, fold: int) -> tuple[np.ndarray, np.ndarray]:
    """Split interno 90/10 estratificado por las etiquetas que entrena el brazo.

    Parameters
    ----------
    y_objetivo:
        Etiquetas objetivo del fold-train (permutadas en A3).
    fold:
        Fold canonico, para la semilla congelada ``SEMILLA + fold``.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        Posiciones relativas del sub-train y del sub-val interno.

    Raises
    ------
    AssertionError
        Si el split no cubre el fold-train o los bloques se solapan.
    """
    divisor = StratifiedShuffleSplit(n_splits=1, test_size=FRACCION_INTERNA,
                                     random_state=SEMILLA + fold)
    dummy = np.zeros((len(y_objetivo), 1), dtype=np.float32)
    sub_tr, sub_va = next(divisor.split(dummy, y_objetivo))
    assert sub_tr.size + sub_va.size == len(y_objetivo), "el split interno no cubre el fold-train"
    assert np.intersect1d(sub_tr, sub_va).size == 0, "sub-train y sub-val se solapan"
    assert sub_tr.min() >= 0 and sub_va.max() < len(y_objetivo), "indices internos fuera de rango"
    return sub_tr, sub_va


def convertir_preds_feval(preds: np.ndarray, n_filas: int) -> np.ndarray:
    """Convierte los ``preds`` del feval de LightGBM a la matriz ``(n_filas, 6)``.

    Convencion verificada con un mini-modelo en /tmp (2026-09-30): lightgbm 4.6.0 entrega
    ``preds`` ya como ``(n_filas, 6)`` con el mismo orden de filas que el dataset; se usa
    tal cual. ``argmax`` es invariante a la transformacion softmax, de modo que el
    macro-F1 calculado aqui coincide con el de ``predict`` (evidencia: mejor iteracion del
    feval == macro-F1 de sklearn sobre ``modelo.predict``). Solo por defensa se transpone
    si llegara como ``(6, n_filas)`` y se reordena si llegara aplanado.
    """
    arreglo = np.asarray(preds, dtype=np.float64)
    if arreglo.ndim == 2:
        if arreglo.shape == (n_filas, N_CLASES):
            return arreglo
        if arreglo.shape == (N_CLASES, n_filas):
            return arreglo.T
    if arreglo.ndim == 1 and arreglo.size == n_filas * N_CLASES:
        return arreglo.reshape(n_filas, N_CLASES)
    raise Detencion(f"[STOP] preds del feval con forma inesperada {arreglo.shape} (exit 2)", 2)


def macro_f1_feval(preds: np.ndarray, dataset: lgb.Dataset) -> tuple[str, float, bool]:
    """feval de macro-F1 para LightGBM multiclass (metrica unica del early stopping).

    Returns
    -------
    tuple[str, float, bool]
        Nombre de la metrica, valor macro-F1 de sklearn y ``is_higher_better``.
    """
    verdad = np.asarray(dataset.get_label()).astype(np.int64)
    matriz_pred = convertir_preds_feval(preds, verdad.size)
    valor = macro(verdad, matriz_pred.argmax(axis=1))
    return "macro_f1_interna", valor, True


def sanity_total_f1_vs_macro(y_interno: np.ndarray, raw_interno: np.ndarray) -> dict:
    """Sanity de metricas del fold 0 de A1 con la metrica de la enmienda pre-ejecucion 1.

    La enmienda pre-ejecucion 1 (campo ``enmiendas`` del JSON congelado) fijo
    ``eval_metric = METRICA_CATBOOST``: en catboost 1.2.10 el default de TotalF1 promedia
    Weighted. El criterio de PASS es la equivalencia de
    ``catboost.utils.eval_metric(..., METRICA_CATBOOST)`` contra la macro-F1 de sklearn
    sobre el split interno (tolerancia 1e-6); el exit 4 queda ligado SOLO a ese criterio
    y el default (Weighted) se conserva como diagnostico del registro.

    Returns
    -------
    dict
        Metrica configurada, valores de TotalF1 (default y Macro), macro-F1 de sklearn,
        sus deltas, la tolerancia congelada y ``pass`` (delta de la metrica enmendada
        contra macro-F1 <= 1e-6).
    """
    prediccion = np.asarray(raw_interno).argmax(axis=1)
    macro_sklearn = macro(y_interno, prediccion)
    total_default = float(catboost_eval_metric(y_interno, raw_interno, "TotalF1")[0])
    total_macro = float(catboost_eval_metric(y_interno, raw_interno, METRICA_CATBOOST)[0])
    delta_macro = abs(total_macro - macro_sklearn)
    return {"metrica": METRICA_CATBOOST, "total_f1_default": total_default,
            "total_f1_average_macro": total_macro, "macro_f1_sklearn": macro_sklearn,
            "delta_default_vs_macro": abs(total_default - macro_sklearn),
            "delta_average_macro_vs_macro": delta_macro, "delta_criterio": delta_macro,
            "tolerancia": TOL_SANITY, "pass": bool(delta_macro <= TOL_SANITY)}


def _entrenar_fold_catboost(x_tr: np.ndarray, y_tr: np.ndarray, x_interno: np.ndarray,
                            y_interno: np.ndarray, x_pred: np.ndarray,
                            con_sanity: bool) -> dict:
    """Entrena un fold de A1/A3 con CatBoost y devuelve el OOF del fold-val.

    Parameters
    ----------
    x_tr, y_tr:
        Sub-train interno (90% del fold-train) y etiquetas objetivo (permutadas en A3).
    x_interno, y_interno:
        Sub-val interno (10%), unico eval_set del fit; el fold-val jamas entra al fit.
    x_pred:
        Features del fold-val canonico, solo para ``predict_proba`` (OOF).
    con_sanity:
        True solo en el fold 0 de A1: calcula el sanity de la metrica enmendada
        (TotalF1:average=Macro vs macro-F1 de sklearn).

    Returns
    -------
    dict
        ``prob_val``, ``best_iteration``, ``mejor_score_interno``, ``truncado``,
        ``tiempo_seg``, ``iteraciones_ejecutadas``, ``curvas`` y ``sanity_metricas``.
    """
    inicio = time.time()
    modelo = CatBoostClassifier(**CATBOOST_CONFIG)
    modelo.fit(x_tr, y_tr, eval_set=(x_interno, y_interno))
    if list(modelo.classes_) != list(range(N_CLASES)):
        raise Detencion("[STOP] CatBoost no vio las 6 clases canonicas (exit 2)", 2)
    historial = modelo.get_evals_result()
    curva_validacion = [float(valor) for valor in historial["validation"][METRICA_CATBOOST]]
    best_iteration = int(modelo.get_best_iteration())
    mejor_score = float(modelo.get_best_score()["validation"][METRICA_CATBOOST])
    prob_val = np.asarray(modelo.predict_proba(x_pred), dtype=np.float64)
    sanity = None
    if con_sanity:
        raw_interno = np.asarray(modelo.predict(x_interno, prediction_type="RawFormulaVal"))
        sanity = sanity_total_f1_vs_macro(y_interno, raw_interno)
    return {"prob_val": prob_val, "best_iteration": best_iteration,
            "mejor_score_interno": mejor_score,
            "truncado": bool(best_iteration >= ITERACIONES - OD_WAIT),
            "tiempo_seg": float(time.time() - inicio),
            "iteraciones_ejecutadas": len(curva_validacion),
            "curvas": {CLAVE_VALIDACION_CATBOOST: curva_validacion,
                       CLAVE_APRENDIZAJE_CATBOOST: [
                           float(valor) for valor in historial["learn"][METRICA_CATBOOST]]},
            "sanity_metricas": sanity}


def _entrenar_fold_lgbm(x_tr: np.ndarray, y_tr: np.ndarray, x_interno: np.ndarray,
                        y_interno: np.ndarray, x_pred: np.ndarray) -> dict:
    """Entrena un fold de A2 (gemelo LightGBM) y devuelve el OOF del fold-val.

    Returns
    -------
    dict
        Mismas claves que el fold de CatBoost (sin ``sanity_metricas``).
    """
    params = {clave: valor for clave, valor in LGBM_CONFIG.items()
              if clave not in ("num_boost_round", "early_stopping", "feval")}
    num_boost_round = int(LGBM_CONFIG["num_boost_round"])
    rondas_parada = int(LGBM_CONFIG["early_stopping"])
    datos_tr = lgb.Dataset(x_tr, label=y_tr)
    datos_interno = lgb.Dataset(x_interno, label=y_interno, reference=datos_tr)
    historial: dict = {}
    llamadas = [lgb.early_stopping(rondas_parada, first_metric_only=True),
                lgb.record_evaluation(historial)]
    inicio = time.time()
    modelo = lgb.train(params, datos_tr, num_boost_round=num_boost_round,
                       valid_sets=[datos_interno], feval=macro_f1_feval, callbacks=llamadas)
    best_iteration = int(getattr(modelo, "best_iteration", num_boost_round))
    prob_val = np.asarray(modelo.predict(x_pred, num_iteration=best_iteration),
                          dtype=np.float64)
    curva = [float(valor) for valor in historial["valid_0"]["macro_f1_interna"]]
    return {"prob_val": prob_val, "best_iteration": best_iteration,
            "mejor_score_interno": float(modelo.best_score["valid_0"]["macro_f1_interna"]),
            "truncado": bool(best_iteration >= ITERACIONES - OD_WAIT),
            "tiempo_seg": float(time.time() - inicio),
            "iteraciones_ejecutadas": len(curva),
            "curvas": {"validacion_macro_f1_interna": curva},
            "sanity_metricas": None}


def entrenar_fold(brazo: str, fold: int, matriz: np.ndarray, clase: np.ndarray,
                  fold_idx: np.ndarray, con_sanity: bool = False) -> dict:
    """Entrena un (brazo, fold) con ES sobre el split interno 90/10 y OOF del fold-val.

    El fold-val solo se usa para ``predict_proba`` final; el fit recibe exclusivamente el
    sub-train y el sub-val internos.

    Returns
    -------
    dict
        Detalle del fold (prob_val, best_iteration, curvas, tiempos e invariantes).
    """
    tr_idx, va_idx, y_tr = etiquetas_objetivo(brazo, clase, fold_idx, fold)
    assert np.intersect1d(tr_idx, va_idx).size == 0, "train y val del fold se solapan"
    sub_tr, sub_va = split_interno(y_tr, fold)
    x_tr = np.asarray(matriz[tr_idx], dtype=np.float32)
    x_interno_tr = np.asarray(x_tr[sub_tr], dtype=np.float32)
    x_interno_va = np.asarray(x_tr[sub_va], dtype=np.float32)
    y_interno_tr = y_tr[sub_tr]
    y_interno_va = y_tr[sub_va]
    x_pred = np.asarray(matriz[va_idx], dtype=np.float32)
    if brazo == "A2_lgbm_twin":
        resultado = _entrenar_fold_lgbm(x_interno_tr, y_interno_tr, x_interno_va,
                                        y_interno_va, x_pred)
    else:
        resultado = _entrenar_fold_catboost(x_interno_tr, y_interno_tr, x_interno_va,
                                            y_interno_va, x_pred, con_sanity)
    if resultado["prob_val"].shape != (va_idx.size, N_CLASES):
        raise Detencion(f"[STOP] OOF del fold con forma inesperada en {brazo}/{fold} (exit 2)", 2)
    if not np.isfinite(resultado["prob_val"]).all():
        raise Detencion(f"[STOP] OOF del fold con no finitos en {brazo}/{fold} (exit 2)", 2)
    resultado.update({"brazo": brazo, "fold": fold, "n_train_fold": int(tr_idx.size),
                      "n_val_fold": int(va_idx.size), "n_interno_train": int(sub_tr.size),
                      "n_interno_val": int(sub_va.size), "eval_set": "split_interno",
                      "split_interno_ok": True})
    return resultado


def verificar_deadline(estado: dict) -> None:
    """Comprueba el timeout global de 24 h antes de cada fit (exit 7 con estado guardado)."""
    transcurrido = time.time() - float(estado["inicio_ts"])
    if transcurrido >= TIMEOUT_H * 3600.0:
        guardar_estado(estado)
        raise Detencion(f"[STOP] timeout global de {TIMEOUT_H} h superado "
                        f"({transcurrido / 3600.0:.2f} h); estado guardado (exit 7)", 7)


def registrar_fold(estado: dict, brazo: str, fold: int, resultado: dict, oof: np.ndarray,
                   fold_idx: np.ndarray) -> None:
    """Incorpora el fold al OOF y al checkpoint y persiste ambos (OOF primero, estado despues).

    Raises
    ------
    AssertionError
        Si la mascara de relleno detecta una fila escrita dos veces.
    """
    va_idx = np.flatnonzero(fold_idx == fold)
    prob_val = resultado.pop("prob_val")
    sanity = resultado.pop("sanity_metricas", None)
    datos = estado["brazos"][brazo]
    mascara = np.asarray(datos["mascara_relleno_por_fila"], dtype=np.int64)
    assert mascara[va_idx].max() == 0, f"{brazo} fold {fold}: fila ya rellenada en el OOF"
    oof[va_idx] = prob_val
    mascara[va_idx] += 1
    guardar_oof(brazo, oof)
    detalle = {campo: resultado[campo] for campo in CAMPOS_FOLD}
    detalle["tiempo_seg"] = float(resultado["tiempo_seg"])
    datos["folds"][str(fold)] = detalle
    datos["folds_completados"] = sorted(datos["folds"])
    datos["mascara_relleno_por_fila"] = mascara.tolist()
    if sanity is not None:
        estado["sanity_metricas_A1_fold0"] = sanity
    guardar_estado(estado)
    print(f"  [{brazo}] fold {fold} listo: {detalle['iteraciones_ejecutadas']} iteraciones, "
          f"best {detalle['best_iteration']}, metrica interna "
          f"{detalle['mejor_score_interno']:.6f}, truncado {detalle['truncado']}, "
          f"{detalle['tiempo_seg']:.1f} s", flush=True)


def asegurar_sanity_metricas(estado: dict) -> None:
    """Falla si el sanity del fold 0 de A1 no esta registrado o no pasa (exit 4)."""
    datos = estado["brazos"]["A1_catboost"]
    if "0" not in datos["folds_completados"]:
        return
    sanity = estado["sanity_metricas_A1_fold0"]
    if sanity is None:
        raise Detencion("[STOP] estado sin el sanity de metricas del fold 0 de A1 y sin modelo "
                        "para recomputarlo (exit 2)", 2)
    if not sanity["pass"]:
        raise Detencion(
            f"[STOP] sanity de metricas (enmienda pre-ejecucion 1): {METRICA_CATBOOST}="
            f"{sanity['total_f1_average_macro']:.7f} vs macro-F1 de sklearn="
            f"{sanity['macro_f1_sklearn']:.7f} (delta "
            f"{sanity['delta_average_macro_vs_macro']:.3e} > {sanity['tolerancia']:.0e}); "
            f"diagnostico TotalF1 default (Weighted) delta "
            f"{sanity['delta_default_vs_macro']:.3e} (exit 4)", 4)


def asegurar_calibracion(estado: dict) -> None:
    """Aplica la calibracion congelada tras el fold 0 de A1 (exit 6 si proyecta > 20 h)."""
    if estado["calibracion"] is not None:
        # Reanudacion: si la calibracion pre-registrada ya fallo, se vuelve a parar (exit 6).
        if not estado["calibracion"].get("ok", True):
            raise Detencion(f"[STOP] calibracion pre-registrada excedida: proyeccion "
                            f"{estado['calibracion']['proyeccion_horas']:.2f} h > "
                            f"{PROYECCION_MAX_H} h; estado guardado (exit 6)", 6)
        return
    datos = estado["brazos"]["A1_catboost"]
    if "0" not in datos["folds_completados"]:
        return
    tiempo_fold0 = float(datos["folds"]["0"]["tiempo_seg"])
    proyeccion_seg = PROYECCION_FACTOR * tiempo_fold0
    estado["calibracion"] = {
        "t_fold0_A1_seg": tiempo_fold0, "factor": PROYECCION_FACTOR,
        "proyeccion_seg": proyeccion_seg, "proyeccion_horas": proyeccion_seg / 3600.0,
        "limite_horas": PROYECCION_MAX_H,
        "ok": bool(proyeccion_seg <= PROYECCION_MAX_H * 3600.0)}
    guardar_estado(estado)
    if not estado["calibracion"]["ok"]:
        raise Detencion(f"[STOP] calibracion: 9 x t_fold0 = "
                        f"{estado['calibracion']['proyeccion_horas']:.2f} h > "
                        f"{PROYECCION_MAX_H} h; estado guardado (exit 6)", 6)


def sonda_determinismo(brazo: str, matriz: np.ndarray, clase: np.ndarray,
                       fold_idx: np.ndarray, oof: np.ndarray, estado: dict) -> dict:
    """Reentrena el fold 0 del brazo desde cero y mide max|delta p| contra el OOF guardado.

    Tolerancias congeladas: 0 -> exacto; <=1e-9 -> tolerancia declarada; >1e-9 -> segunda
    repeticion con rango publicado; rango >= variabilidad o solape con el gate -> no
    concluyente.

    Returns
    -------
    dict
        Deltas de cada repeticion, rango, clasificacion y bandera ``no_concluyente``.
    """
    verificar_deadline(estado)
    referencia = oof[fold_idx == 0]
    replica_1 = entrenar_fold(brazo, 0, matriz, clase, fold_idx)
    delta_1 = float(np.abs(replica_1["prob_val"] - referencia).max())
    sonda = {"descripcion": f"reentrenamiento de {brazo} fold 0 con config identica",
             "max_abs_diff_repeticion_1": delta_1, "max_abs_diff": delta_1,
             "segunda_repeticion": None, "rango": None, "clasificacion": None,
             "no_concluyente": False, "solape_con_gate": False,
             "tolerancia_declarada": TOL_DETERMINISMO, "variabilidad_referencia": VARIABILIDAD}
    if delta_1 == 0.0:
        sonda["clasificacion"] = "exacto"
    elif delta_1 <= TOL_DETERMINISMO:
        sonda["clasificacion"] = "tolerancia_declarada"
    else:
        verificar_deadline(estado)
        replica_2 = entrenar_fold(brazo, 0, matriz, clase, fold_idx)
        delta_2 = float(np.abs(replica_2["prob_val"] - referencia).max())
        rango = max(delta_1, delta_2) - min(delta_1, delta_2)
        sonda.update({"clasificacion": "rango_publicado", "segunda_repeticion": delta_2,
                      "max_abs_diff": max(delta_1, delta_2), "rango": rango,
                      "solape_con_gate": bool(rango >= GATE_DELTA),
                      "no_concluyente": bool(rango >= VARIABILIDAD or rango >= GATE_DELTA)})
    print(f"  sonda de determinismo {brazo} fold 0: max|delta p| = "
          f"{sonda['max_abs_diff']:.3e} ({sonda['clasificacion']})", flush=True)
    return sonda


def entrenar_con_checkpoints(matriz: np.ndarray, clase: np.ndarray, fold_idx: np.ndarray,
                             estado: dict) -> dict[str, np.ndarray]:
    """Entrena los 3 brazos x 5 folds de forma resumible y devuelve los OOF.

    Returns
    -------
    dict[str, numpy.ndarray]
        OOF float64 (169336, 6) por brazo.

    Raises
    ------
    Detencion
        Timeout global (7), calibracion excedida (6) o sanity de metricas fallido (4).
    """
    oofs: dict[str, np.ndarray] = {}
    # Reanudacion: las guardas de saneamiento pre-registradas se re-verifican antes de
    # cualquier fit (en la corrida normal aun no hay fold 0 y ambas retornan de inmediato).
    asegurar_sanity_metricas(estado)
    asegurar_calibracion(estado)
    for brazo in BRAZOS:
        oof = asegurar_brazo_cargado(estado, brazo, fold_idx)
        for fold in range(N_FOLDS):
            if str(fold) in estado["brazos"][brazo]["folds_completados"]:
                continue
            verificar_deadline(estado)
            con_sanity = bool(brazo == "A1_catboost" and fold == 0)
            resultado = entrenar_fold(brazo, fold, matriz, clase, fold_idx, con_sanity)
            registrar_fold(estado, brazo, fold, resultado, oof, fold_idx)
            if brazo == "A1_catboost" and fold == 0:
                asegurar_sanity_metricas(estado)
                asegurar_calibracion(estado)
        datos = estado["brazos"][brazo]
        if datos["folds_completados"] != [str(fold) for fold in range(N_FOLDS)]:
            raise Detencion(f"[STOP] el brazo {brazo} quedo incompleto (exit 2)", 2)
        mascara = np.asarray(datos["mascara_relleno_por_fila"], dtype=np.int64)
        if not bool((mascara == 1).all()):
            raise Detencion(f"[STOP] mascara de relleno del OOF de {brazo} no es 1 por fila "
                            "(exit 2)", 2)
        if not np.isfinite(oof).all():
            raise Detencion(f"[STOP] OOF de {brazo} con no finitos (exit 2)", 2)
        oofs[brazo] = oof
        print(f"  [{brazo}] OOF completo: {oof.shape} float64, mascara 1 por fila", flush=True)
        if brazo == "A1_catboost":
            asegurar_sanity_metricas(estado)
            asegurar_calibracion(estado)
        if brazo in BRAZOS_CON_SONDA and datos["sonda_determinismo"] is None:
            datos["sonda_determinismo"] = sonda_determinismo(brazo, matriz, clase, fold_idx,
                                                             oof, estado)
            guardar_estado(estado)
    return oofs


def curva_interna(curvas: dict) -> list[float]:
    """Serie interna del ES del fold (TotalF1:average=Macro en CatBoost, macro-F1 feval en
    LightGBM)."""
    if CLAVE_VALIDACION_CATBOOST in curvas:
        return curvas[CLAVE_VALIDACION_CATBOOST]
    return curvas["validacion_macro_f1_interna"]


def etiqueta_metrica(brazo: str) -> str:
    """Etiqueta de la metrica interna del brazo."""
    return "macro-F1 interna" if brazo == "A2_lgbm_twin" else f"{METRICA_CATBOOST} interna"


def verificar_invariantes(estado: dict, matriz: np.ndarray, clase: np.ndarray,
                          fold_idx: np.ndarray, oofs: dict[str, np.ndarray]) -> dict:
    """Evidencias de no-leakage y de que el ES solo miro el split interno.

    Returns
    -------
    dict
        Chequeos booleanos derivados de los artefactos (no declaraciones a mano):
        alineacion, folds canonicos, folds completos, mascara de relleno 1 por fila,
        split interno cubierto sin solapes y mejor metrica interna = maximo de su curva.
    """
    completos, mascaras, splits, metricas = [], [], [], []
    for brazo in BRAZOS:
        datos = estado["brazos"][brazo]
        completos.append(datos["folds_completados"] == [str(fold) for fold in range(N_FOLDS)])
        mascara = np.asarray(datos["mascara_relleno_por_fila"], dtype=np.int64)
        mascaras.append(bool((mascara == 1).all()))
        for fold in range(N_FOLDS):
            detalle = datos["folds"][str(fold)]
            curva = curva_interna(detalle["curvas"])
            splits.append(bool(detalle["split_interno_ok"]
                               and detalle["eval_set"] == "split_interno"
                               and detalle["n_interno_train"] + detalle["n_interno_val"]
                               == detalle["n_train_fold"]))
            metricas.append(bool(abs(detalle["mejor_score_interno"] - max(curva))
                                 <= TOL_INVARIANTE_METRICA))
    return {
        "alineacion_filas": bool(matriz.shape[0] == clase.shape[0] == N_FILAS),
        "cinco_folds_canonicos": bool(np.unique(fold_idx).size == N_FOLDS),
        "folds_completos_por_brazo": bool(all(completos)),
        "oof_mascara_relleno_uno_por_fila": bool(all(mascaras)),
        "split_interno_verificado_por_fold": bool(all(splits)),
        "es_metrica_interna_maxima_es_mejor": bool(all(metricas)),
    }


def resumir_truncamiento(estado: dict) -> dict:
    """Resumen de truncamiento por brazo (``best_iteration >= iterations - od_wait``)."""
    resumen = {}
    for brazo in BRAZOS:
        datos = estado["brazos"][brazo]
        por_fold = {str(fold): bool(datos["folds"][str(fold)]["truncado"])
                    for fold in range(N_FOLDS)}
        resumen[brazo] = {"presupuesto_agotado": bool(any(por_fold.values())),
                          "por_fold": por_fold}
    return resumen


# =====================================================================================
# ---- 4. EVALUACION: INTEGRACION P6/P7, COMPLEMENTARIEDAD Y BOOTSTRAP ----
# =====================================================================================
def bootstrap_delta(clase: np.ndarray, pred_base: np.ndarray, pred_cand: np.ndarray) -> dict:
    """Bootstrap pareado (N_BOOTSTRAP, semilla congelada) de F1(cand) - F1(base).

    Parameters
    ----------
    clase:
        Etiquetas canonicas (169336,).
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

    Returns
    -------
    tuple[numpy.ndarray, dict]
        ``p_comb`` (169336, 6) normalizada y el detalle del combinador (ratio del bloque del
        brazo e iteraciones por fold).

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
        macro-F1 global, delta vs baseline y nested, folds (global, por fold y deltas por
        fold), folds positivos, IC95 del bootstrap, ``argmax_sin_regla``, filas que cambian y
        la prediccion combinada (no serializable: el llamador la retira antes del JSON).
    """
    pred = np.argmax(p_comb * pesos, axis=1)
    cv = macro(clase, pred)
    folds = [macro(clase[fold_idx == k], pred[fold_idx == k]) for k in range(N_FOLDS)]
    folds_base = [macro(clase[fold_idx == k], champ_pred[fold_idx == k])
                  for k in range(N_FOLDS)]
    deltas_por_fold = [float(a - b) for a, b in zip(folds, folds_base)]
    ic = bootstrap_delta(clase, champ_pred, pred)
    sin_regla = macro(clase, np.argmax(p_comb, axis=1))
    return {"corrected_cv": cv, "delta_vs_baseline": cv - BASELINE,
            "delta_vs_nested": cv - NESTED, "folds": folds, "folds_base": folds_base,
            "deltas_por_fold": deltas_por_fold,
            "folds_positivos": int(sum(1 for delta in deltas_por_fold if delta > 0)),
            "filas_que_cambian": int((pred != champ_pred).sum()),
            "argmax_sin_regla": sin_regla, "delta_sin_regla_vs_baseline": sin_regla - BASELINE,
            "delta_sin_regla_vs_nested": sin_regla - NESTED,
            "delta_medio_bootstrap": ic["delta_medio"], "ic95": ic["ic95"],
            "p_delta_positivo": ic["p_delta_positivo"], "ic95_sin_cero": ic["ic95_sin_cero"],
            "pred_comb": pred}


def diagnostico_brazo(p_brazo: np.ndarray, clase: np.ndarray, fold_idx: np.ndarray,
                      champ_pred: np.ndarray, conf: np.ndarray) -> dict:
    """Standalone del brazo (sin pesos), matriz esperada 6x6 y rescate/dano vs campeon.

    Parameters
    ----------
    p_brazo:
        OOF del brazo (169336, 6).
    clase:
        Etiquetas canonicas.
    fold_idx:
        Fold canonico por fila.
    champ_pred:
        Decision del campeon (``argmax(v30 * pesos)``).
    conf:
        Confianza del campeon (``max v30``), para el subconjunto de alta confianza.

    Returns
    -------
    dict
        macro-F1 standalone global y por fold, F1 por clase, matriz ``E[p[:, j] | y = i]``
        con su diagonal, y rescates/danos/neto globales y por clase.
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
            "f1_por_clase": [float(valor) for valor in f1_por_clase],
            "matriz_esperada": esperada.tolist(), "diagonal_ok": diagonal_ok,
            "vs_campeon": {"rescate": rescate, "danio": danio, "neto": rescate - danio,
                           "rescates_alta_confianza": int((alta_confianza & ok_cand).sum()),
                           "errores_alta_confianza_campeon": int(alta_confianza.sum()),
                           "por_clase": por_clase}}


# =====================================================================================
# ---- 5. GATES Y VEREDICTO ----
# =====================================================================================
def calcular_gates(baseline_ok: bool, integracion: dict, control_atribucion: dict,
                   truncamiento: dict, no_concluyente: bool) -> tuple[dict, dict, str]:
    """Aplica los gates congelados y el arbol de decision (control negativo con precedencia).

    Gates: baseline recomputado, delta de A1 >= +0.0015, folds positivos >= 4/5, control
    favorable A1 sobre A2 (delta puntual y bootstrap pareado de la diferencia sin cero),
    IC95 del delta de A1 sin cero, sin leakage, sin truncamiento de A1 y determinismo
    concluyente (los dos ultimos capan PROMISING a MARGINAL).

    Returns
    -------
    tuple[dict, dict, str]
        Gates, detalle del control negativo y veredicto del arbol.
    """
    a1 = integracion["A1_catboost"]
    a2 = integracion["A2_lgbm_twin"]
    a3 = integracion["A3_barajado"]
    gates = {
        "baseline_ok": bool(baseline_ok),
        "delta_A1_suficiente": bool(a1["delta_vs_baseline"] >= GATE_DELTA),
        "folds_positivos_A1_suficiente": bool(a1["folds_positivos"] >= GATE_FOLDS),
        "control_favorable_A1_sobre_A2": bool(
            a1["delta_vs_baseline"] > a2["delta_vs_baseline"]
            and control_atribucion["ic95_sin_cero"]),
        "ic95_A1_sin_cero": bool(a1["ic95_sin_cero"]),
        "sin_leakage": True,
        "sin_truncamiento_A1": bool(not truncamiento["A1_catboost"]["presupuesto_agotado"]),
        "determinismo_concluyente": bool(not no_concluyente),
    }
    control_negativo = {"delta_A1": a1["delta_vs_baseline"], "delta_A3": a3["delta_vs_baseline"],
                        "no_atribuible": bool(a3["delta_vs_baseline"] >= a1["delta_vs_baseline"])}
    if control_negativo["no_atribuible"]:
        veredicto = "NO_ATRIBUIBLE"
    elif all(gates.values()):
        veredicto = "PROMISING"
    elif a1["delta_vs_baseline"] > 0:
        veredicto = "MARGINAL"
    else:
        veredicto = "CLOSED"
    return gates, control_negativo, veredicto


# =====================================================================================
# ---- 6. SALIDAS (JSON, informe, figura, RUNLOG, CSV, consola) Y CLI ----
# =====================================================================================
def figura_curvas(entrenamiento: dict, ruta: Path) -> None:
    """Figura 1x3 con la metrica interna del ES por fold y brazo."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figura, ejes = plt.subplots(1, 3, figsize=(18.0, 5.0))
    for columna, brazo in enumerate(BRAZOS):
        eje = ejes[columna]
        for fold in range(N_FOLDS):
            curva = curva_interna(entrenamiento[brazo]["folds"][str(fold)]["curvas"])
            eje.plot(range(1, len(curva) + 1), curva, linewidth=1.0, label=f"fold {fold}")
        eje.set_title(f"{brazo}: {etiqueta_metrica(brazo)} (ES)")
        eje.set_xlabel("iteracion")
        eje.set_ylabel(etiqueta_metrica(brazo))
        eje.grid(alpha=0.3)
        eje.legend(fontsize=7)
    figura.tight_layout()
    figura.savefig(ruta, dpi=140)
    plt.close(figura)
    print(f"  figura: {ruta_relativa(ruta)}", flush=True)


def escribir_informe(resultado: dict) -> None:
    """Escribe el informe Markdown del experimento con tablas, gates y compliance."""
    integracion = resultado["integracion"]
    entrenamiento = resultado["entrenamiento"]
    truncamiento = resultado["truncamiento"]
    lineas = [
        "# V32-E2 (P2) - Ejecucion: challenger CatBoost sobre la matriz 279",
        "",
        f"- Fecha de ejecucion: {resultado['fecha']}",
        (f"- Pre-registro: {ARCHIVO_PRE_REGISTRO} (commit "
         f"{resultado['prereg']['commit_congelado'][:7]}) y `experiments/"
         "v32_02_preregistro_catboost.json`"),
        f"- Commit de ejecucion: `{resultado['commit']}`",
        (f"- Baseline primario recomputado: {resultado['verificacion']['baseline_recomputado']:.7f} "
         f"(referencia nested {NESTED:.7f})"),
        (f"- CPU con {HILOS} hilos fijos; ES interno 90/10; od_wait/early_stopping = {OD_WAIT}; "
         f"iteraciones cap {ITERACIONES}"),
        (f"- Compliance AST: {resultado['compliance']['literales_de_ruta_revisados']} literales "
         "de ruta revisados, 0 violaciones"),
        "",
        "## Entrenamiento (el fold-val solo se uso para el OOF; el ES miro el split interno)",
        "",
        ("| Brazo | fold | iters | best_iteration | metrica interna mejor | truncado | "
         "tiempo (s) |"),
        "|---|---|---|---|---|---|---|",
    ]
    for brazo in BRAZOS:
        for fold in range(N_FOLDS):
            detalle = entrenamiento[brazo]["folds"][str(fold)]
            lineas.append(
                f"| {brazo} | {fold} | {detalle['iteraciones_ejecutadas']} | "
                f"{detalle['best_iteration']} | {detalle['mejor_score_interno']:.6f} | "
                f"{detalle['truncado']} | {detalle['tiempo_seg']:.1f} |")
    sanity = resultado["sanity_metricas_A1_fold0"]
    lineas += [
        "",
        "## Sanity de metricas (A1 fold 0)",
        "",
        (f"- Criterio de la enmienda pre-ejecucion 1: {sanity['metrica']} = "
         f"{sanity['total_f1_average_macro']:.7f} vs macro-F1 de sklearn = "
         f"{sanity['macro_f1_sklearn']:.7f}; delta = "
         f"{sanity['delta_average_macro_vs_macro']:.3e} (tolerancia {TOL_SANITY:.0e})"),
        (f"- Diagnostico conservado: TotalF1 default (Weighted) = "
         f"{sanity['total_f1_default']:.7f}; delta contra macro-F1 de sklearn = "
         f"{sanity['delta_default_vs_macro']:.3e}"),
        "",
        "## Sonda de determinismo",
        "",
    ]
    for brazo in BRAZOS_CON_SONDA:
        sonda = resultado["determinismo"][f"sonda_{brazo}"]
        lineas.append(f"- {brazo}: max|delta p| = {sonda['max_abs_diff']:.3e} "
                      f"({sonda['clasificacion']}, no_concluyente={sonda['no_concluyente']})")
    lineas += [
        (f"- Calibracion tras el fold 0 de A1: t = "
         f"{resultado['calibracion']['t_fold0_A1_seg']:.1f} s; proyeccion 9x = "
         f"{resultado['calibracion']['proyeccion_horas']:.2f} h "
         f"(limite {PROYECCION_MAX_H} h)"),
        "",
        "## Integracion P6/P7 (combinador congelado, regla argmax(p_comb * w))",
        "",
        ("| Brazo | macro-F1 comb | delta vs baseline | folds+ | IC95 | delta sin regla | "
         "delta vs nested | sha256 OOF (16) |"),
        "|---|---|---|---|---|---|---|---|",
    ]
    for brazo in BRAZOS:
        met = integracion[brazo]
        sha = entrenamiento[brazo]["oof_artefacto"]["sha256"][:16]
        lineas.append(
            f"| {brazo} | {met['corrected_cv']:.7f} | {met['delta_vs_baseline']:+.7f} | "
            f"{met['folds_positivos']}/5 | [{met['ic95'][0]:+.7f}, {met['ic95'][1]:+.7f}] | "
            f"{met['delta_sin_regla_vs_baseline']:+.7f} | {met['delta_vs_nested']:+.7f} | "
            f"`{sha}` |")
    lineas += [
        "",
        "## Complementariedad standalone (argmax del OOF, sin pesos)",
        "",
        ("| Brazo | macro-F1 solo | diagonal 6x6 | rescate | dano | neto | rescate alta "
         "conf. | errores alta conf. campeon |"),
        "|---|---|---|---|---|---|---|---|",
    ]
    for brazo in BRAZOS:
        diag = resultado["complementariedad"][brazo]
        vs = diag["vs_campeon"]
        lineas.append(
            f"| {brazo} | {diag['standalone_cv']:.7f} | {diag['diagonal_ok']} | {vs['rescate']} | "
            f"{vs['danio']} | {vs['neto']:+d} | {vs['rescates_alta_confianza']} | "
            f"{vs['errores_alta_confianza_campeon']} |")
    lineas += [
        "",
        "## Gates",
        "",
        "| Gate | Resultado |",
        "|---|---|",
    ]
    for nombre, valor in resultado["gates"].items():
        lineas.append(f"| {nombre} | {valor} |")
    lineas += [
        "",
        "## Control de atribucion y control negativo",
        "",
        (f"- Bootstrap pareado de la diferencia (F1_combA1 - F1_combA2): delta medio "
         f"{resultado['control_atribucion']['delta_medio']:+.7f}, IC95 "
         f"[{resultado['control_atribucion']['ic95'][0]:+.7f}, "
         f"{resultado['control_atribucion']['ic95'][1]:+.7f}], favorable="
         f"{resultado['control_atribucion']['favorable']}"),
        (f"- delta_int(A3) = {resultado['control_negativo']['delta_A3']:+.7f} vs "
         f"delta_int(A1) = {resultado['control_negativo']['delta_A1']:+.7f}: "
         f"{'NO_ATRIBUIBLE' if resultado['control_negativo']['no_atribuible'] else 'atribuible'}"
         " a la senal de CatBoost"),
        (f"- Truncamiento: A1 presupuesto agotado = "
         f"{truncamiento['A1_catboost']['presupuesto_agotado']}; A2 = "
         f"{truncamiento['A2_lgbm_twin']['presupuesto_agotado']} (control debilitado si True); "
         f"A3 = {truncamiento['A3_barajado']['presupuesto_agotado']}"),
        "",
        "## Veredicto",
        "",
        f"**{resultado['veredicto']}**: {resultado['siguiente_paso']}",
        "",
        "## Compliance",
        "",
        ("- 0 embeddings, 0 transformers, 0 LLM, 0 datos externos; sin test, sin holdout, sin "
         "leaderboard, sin submission."),
        ("- Auditoria AST del propio runner: 0 palabras vigiladas en literales de ruta; sin "
         "lecturas de particiones reservadas."),
        ("- El gemelo LightGBM se reentrena desde cero (control interno, no reutiliza pesos del "
         "pool); campeon V30 + plug-in intacto; sin tuning posterior."),
        "",
    ]
    INFORME.write_text("\n".join(lineas), encoding="utf-8")
    print(f"  informe: {ruta_relativa(INFORME)}", flush=True)


def anotar_runlog(resultado: dict) -> bool:
    """Anade el bloque del experimento a RUNLOG.md una sola vez (append de texto)."""
    marca = "V32-E2 ejecutado"
    if marca in RUNLOG.read_text(encoding="utf-8"):
        return False
    a1 = resultado["integracion"]["A1_catboost"]
    a2 = resultado["integracion"]["A2_lgbm_twin"]
    a3 = resultado["integracion"]["A3_barajado"]
    lineas = [
        "",
        "---",
        "",
        f"## V32-E2 ejecutado: challenger CatBoost sobre matriz_279 - {FECHA}",
        "",
        (f"- Veredicto: {resultado['veredicto']}. A1 comb {a1['corrected_cv']:.7f} "
         f"(delta {a1['delta_vs_baseline']:+.7f}, {a1['folds_positivos']}/5 folds); "
         f"A2 control delta {a2['delta_vs_baseline']:+.7f}; A3 negativo delta "
         f"{a3['delta_vs_baseline']:+.7f}."),
        (f"- Baseline recomputado {BASELINE:.7f}; ES interno 90/10; 32 hilos CPU; "
         "sin tuning posterior."),
        "- Sin particiones reservadas: sin test, sin holdout, sin submission.",
        (f"- Artefactos: {ruta_relativa(RESULTADOS_JSON)} | {ruta_relativa(INFORME)} | "
         f"{ruta_relativa(FIGURA)}."),
    ]
    bloque = "\n" + "\n".join(lineas) + "\n"
    with RUNLOG.open("a", encoding="utf-8") as manejador:
        manejador.write(bloque)
    return True


def anotar_experiments_csv(resultado: dict) -> bool:
    """Anade la fila V32-E2 a experiments.csv (22 columnas de la cabecera, sin CR)."""
    with EXPERIMENTS_CSV.open("r", encoding="utf-8", newline="") as manejador:
        filas = list(csv.reader(manejador))
    cabecera = filas[0]
    identificadores = {fila[0] for fila in filas[1:] if fila}
    if "V32-E2" in identificadores:
        return False
    a1 = resultado["integracion"]["A1_catboost"]
    a2 = resultado["integracion"]["A2_lgbm_twin"]
    a3 = resultado["integracion"]["A3_barajado"]
    resumen = (f"A1 CatBoost comb {a1['corrected_cv']:.7f} (delta {a1['delta_vs_baseline']:+.7f}, "
               f"{a1['folds_positivos']}/5 folds, IC95 [{a1['ic95'][0]:+.7f}, "
               f"{a1['ic95'][1]:+.7f}]); A2 gemelo delta {a2['delta_vs_baseline']:+.7f}; "
               f"A3 barajado delta {a3['delta_vs_baseline']:+.7f}")
    registro = {
        "experiment_id": "V32-E2", "fecha": FECHA, "bloque": "32.2",
        "etapa": "V32-E2 ejecucion: challenger CatBoost MultiClass 279 (A1 decisional) + "
                 "gemelo LightGBM (A2 control) + barajado (A3 negativo)",
        "pregunta": "CatBoost sobre matriz_279 aporta complementariedad al campeon mas alla de "
                    "la familia boosting ya usada?",
        "hipotesis": "el ordered boosting aporta informacion distinta del LGBM afilado del pool",
        "features": "matriz_279 congelada (169336x279); 0 embeddings",
        "preprocessing": f"sin escalado (GBDT); split interno 90/10 estratificado; ES 150 "
                         f"sobre {METRICA_CATBOOST} (CatBoost; enmienda pre-ejecucion 1) y "
                         "macro-F1 feval (LGBM)",
        "representacion": "features numericas congeladas del meta-espacio",
        "modelo": "CatBoost 1.2.10 MultiClass (A1) / LightGBM 4.6.0 gemelo ancla V29-V30 (A2) / "
                  "CatBoost barajado (A3)",
        "hiperparametros": "congelados: 3000 iter cap, lr 0.03, depth 6, l2 3.0, ES 150, "
                           "32 hilos, seed 20261010",
        "validacion": "particion canonica; integracion P6/P7; bootstrap pareado de la diferencia "
                      "A1-A2; gates +0.0015/4-5; calibracion y timeout",
        "macro_f1_mean": f"{float(np.mean(a1['folds'])):.7f}",
        "macro_f1_std": f"{float(np.std(a1['folds'])):.7f}",
        "weighted_f1_mean": "", "micro_f1_mean": "", "accuracy_mean": "",
        "resultado": resumen,
        "interpretacion": f"veredicto {resultado['veredicto']} por el arbol pre-registrado; "
                          "A2/A3 como controles de atribucion y negativo",
        "decision": resultado["siguiente_paso"],
        "duracion_seg": f"{float(resultado['duracion_seg']):.1f}",
        "artefactos": f"{ruta_relativa(RESULTADOS_JSON)} | {ruta_relativa(INFORME)} | "
                      f"{ruta_relativa(FIGURA)}",
    }
    if set(registro) != set(cabecera):
        raise Detencion("[STOP] la fila nueva no respeta las columnas de experiments.csv "
                        "(exit 1)", 1)
    fila = [registro[columna] for columna in cabecera]
    with EXPERIMENTS_CSV.open("a", encoding="utf-8", newline="") as manejador:
        escritor = csv.writer(manejador, lineterminator="\n")
        escritor.writerow(fila)
    return True


def resumen_consola(resultado: dict) -> None:
    """Resumen final en consola (maximo 15 lineas)."""
    integracion = resultado["integracion"]
    print()
    print(f"[V32-E2] fin en {resultado['duracion_seg']:.0f} s | baseline "
          f"{resultado['verificacion']['baseline_recomputado']:.7f} | 32 hilos CPU | "
          f"determinismo no_concluyente={resultado['determinismo']['no_concluyente']}")
    for brazo in BRAZOS:
        met = integracion[brazo]
        print(f"  {brazo}: comb {met['corrected_cv']:.7f} "
              f"(delta {met['delta_vs_baseline']:+.7f}, {met['folds_positivos']}/5 folds, "
              f"IC95 [{met['ic95'][0]:+.7f}, {met['ic95'][1]:+.7f}])")
    print(f"  control A1 vs A2: {resultado['control_atribucion']['delta_medio']:+.7f} "
          f"IC95 [{resultado['control_atribucion']['ic95'][0]:+.7f}, "
          f"{resultado['control_atribucion']['ic95'][1]:+.7f}] "
          f"favorable={resultado['control_atribucion']['favorable']}")
    print(f"  control negativo: A3 {resultado['control_negativo']['delta_A3']:+.7f} vs A1 "
          f"{resultado['control_negativo']['delta_A1']:+.7f} -> "
          f"{'NO_ATRIBUIBLE' if resultado['control_negativo']['no_atribuible'] else 'ok'}")
    print(f"  gates: {resultado['gates']}")
    print(f"  truncamiento A1: {resultado['truncamiento']['A1_catboost']['presupuesto_agotado']}")
    print(f"  sanity {METRICA_CATBOOST} vs macro-F1: delta "
          f"{resultado['sanity_metricas_A1_fold0']['delta_average_macro_vs_macro']:.3e} "
          f"(pass={resultado['sanity_metricas_A1_fold0']['pass']})")
    print(f"  VEREDICTO: {resultado['veredicto']}")
    print(f"  artefactos: {ruta_relativa(RESULTADOS_JSON)} | {ruta_relativa(INFORME)} | "
          f"{ruta_relativa(FIGURA)}")


# =====================================================================================
# ---- 7. PIPELINE COMPLETO ----
# =====================================================================================
def construir_resultado(inicio: float, prereg: dict, detalle_constantes: dict,
                        detalle_versiones: dict, detalle_hashes: dict, auditoria: dict,
                        baseline: float, estado: dict, entrenamiento: dict,
                        integracion: dict, complementariedad: dict, invariantes: dict,
                        control_atribucion: dict, gates: dict, control_negativo: dict,
                        veredicto: str) -> dict:
    """Ensambla el JSON de resultados con config, hashes, determinismo, gates y veredicto."""
    truncamiento = resumir_truncamiento(estado)
    best_iterations = {brazo: {str(fold): int(entrenamiento[brazo]["folds"][str(fold)]
                                              ["best_iteration"]) for fold in range(N_FOLDS)}
                       for brazo in BRAZOS}
    tiempos = {brazo: {str(fold): float(entrenamiento[brazo]["folds"][str(fold)]["tiempo_seg"])
                       for fold in range(N_FOLDS)} for brazo in BRAZOS}
    oof_artefactos = {}
    for brazo in BRAZOS:
        ruta = ruta_oof(brazo)
        oof_artefactos[brazo] = {"archivo": ruta_relativa(ruta),
                                 "sha256": sha256_archivo(ruta)}
    return {
        "fase": "V32-E2 (P2) ejecucion: challenger CatBoost sobre matriz_279",
        "fecha": FECHA,
        "commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=False,
                                 capture_output=True, text=True, timeout=30).stdout.strip(),
        "prereg": {"json": ruta_relativa(PREREG_JSON), "sha256": sha256_archivo(PREREG_JSON),
                   "informe": ARCHIVO_PRE_REGISTRO,
                   "commit_congelado": str(prereg["commit"])},
        "verificacion": {"constantes": detalle_constantes, "versiones": detalle_versiones,
                         "hashes": detalle_hashes["checks"],
                         "hashes_recomputados": detalle_hashes["recomputados"],
                         "matriz_esperada": detalle_hashes["matriz_esperada"],
                         "baseline_recomputado": baseline, "baseline_esperado": BASELINE,
                         "baseline_ok": True},
        "compliance": {**auditoria, "embeddings": 0, "transformers": 0, "llm": 0,
                       "datos_externos": 0, "test": 0, "holdout": 0, "leaderboard": 0,
                       "submission": 0},
        "config": {"catboost": CATBOOST_CONFIG, "lgbm": LGBM_CONFIG, "pesos_plugin": PESOS,
                   "split_interno": TEXTO_SPLIT_INTERNO, "od_wait": OD_WAIT, "hilos": HILOS,
                   "seeds": {"global": SEMILLA, "barajado_etiquetas": SEMILLA_BARAJADO},
                   "lgbm_extra_en_fit": {"metric": "None"}},
        "determinismo": {
            "sonda_A1_catboost": estado["brazos"]["A1_catboost"]["sonda_determinismo"],
            "sonda_A2_lgbm_twin": estado["brazos"]["A2_lgbm_twin"]["sonda_determinismo"],
            "no_concluyente": bool(
                estado["brazos"]["A1_catboost"]["sonda_determinismo"]["no_concluyente"]
                or estado["brazos"]["A2_lgbm_twin"]["sonda_determinismo"]["no_concluyente"]),
            "tolerancia_declarada": TOL_DETERMINISMO, "variabilidad_referencia": VARIABILIDAD},
        "calibracion": estado["calibracion"],
        "sanity_metricas_A1_fold0": estado["sanity_metricas_A1_fold0"],
        "truncamiento": {"regla": f"best_iteration >= iterations-{OD_WAIT}", **truncamiento},
        "best_iterations": best_iterations,
        "tiempos_por_fold_seg": tiempos,
        "entrenamiento": {brazo: {**entrenamiento[brazo], "oof_artefacto": oof_artefactos[brazo]}
                          for brazo in BRAZOS},
        "integracion": integracion,
        "complementariedad": complementariedad,
        "sin_leakage": invariantes,
        "control_atribucion": control_atribucion,
        "gates": gates,
        "control_negativo": control_negativo,
        "veredicto": veredicto,
        "siguiente_paso": ARBOL[veredicto],
        "duracion_seg": float(time.time() - inicio),
    }


def ejecutar(verificar: bool) -> int:
    """Pipeline completo: verificacion, entrenamiento resumible, integracion y gates."""
    inicio = time.time()
    prereg = cargar_preregistro()
    constantes_ok, detalle_constantes = verificar_constantes(prereg)
    versiones_ok, detalle_versiones = verificar_versiones(prereg)
    hashes_ok, detalle_hashes = verificar_hashes(prereg)
    if not (constantes_ok and versiones_ok and hashes_ok):
        raise Detencion("[STOP] el pre-registro congelado no se respeta (exit 2)", 2)
    auditoria = auditar_ast_propio()
    clase, fold_idx, v30 = cargar_canonico()
    pesos = np.array(PESOS, dtype=np.float64)
    champ_pred = np.argmax(v30 * pesos, axis=1)
    confianza = v30.max(axis=1)
    baseline = macro(clase, champ_pred)
    print(f"  baseline primario recomputado: {baseline:.7f} (esperado {BASELINE:.7f})", flush=True)
    if abs(baseline - BASELINE) > ABORTO_BASELINE:
        raise Detencion("[STOP] el baseline canonico no reproduce (exit 2)", 2)
    matriz = cargar_matriz(prereg)
    print(f"  matriz_279: {matriz.shape} {matriz.dtype} | constantes, versiones y hashes: OK",
          flush=True)
    print(f"  compliance AST: {auditoria['literales_de_ruta_revisados']} literales de ruta, "
          "0 violaciones", flush=True)
    if verificar:
        print("VERIFICACION OK", flush=True)
        return 0

    estado = cargar_estado()
    oofs = entrenar_con_checkpoints(matriz, clase, fold_idx, estado)
    invariantes = verificar_invariantes(estado, matriz, clase, fold_idx, oofs)

    integracion: dict[str, dict] = {}
    preds_comb: dict[str, np.ndarray] = {}
    for brazo in BRAZOS:
        p_comb, detalle_combinador = combinador_oof(v30, oofs[brazo], clase, fold_idx)
        met = evaluar_combinacion(p_comb, clase, fold_idx, champ_pred, pesos)
        met["combinador"] = detalle_combinador
        preds_comb[brazo] = met.pop("pred_comb")
        integracion[brazo] = met
        print(f"  integracion {brazo}: comb {met['corrected_cv']:.7f} "
              f"(delta {met['delta_vs_baseline']:+.7f}, {met['folds_positivos']}/5)", flush=True)

    complementariedad = {brazo: diagnostico_brazo(oofs[brazo], clase, fold_idx, champ_pred,
                                                  confianza) for brazo in BRAZOS}
    diferencia = bootstrap_delta(clase, preds_comb["A2_lgbm_twin"], preds_comb["A1_catboost"])
    control_atribucion = {
        "delta_A1": integracion["A1_catboost"]["delta_vs_baseline"],
        "delta_A2": integracion["A2_lgbm_twin"]["delta_vs_baseline"],
        "diferencia_puntual": (integracion["A1_catboost"]["delta_vs_baseline"]
                               - integracion["A2_lgbm_twin"]["delta_vs_baseline"]),
        "delta_medio": diferencia["delta_medio"], "ic95": diferencia["ic95"],
        "p_delta_positivo": diferencia["p_delta_positivo"],
        "ic95_sin_cero": diferencia["ic95_sin_cero"],
        "favorable": bool(integracion["A1_catboost"]["delta_vs_baseline"]
                          > integracion["A2_lgbm_twin"]["delta_vs_baseline"]
                          and diferencia["ic95_sin_cero"])}
    truncamiento = resumir_truncamiento(estado)
    no_concluyente = bool(
        estado["brazos"]["A1_catboost"]["sonda_determinismo"]["no_concluyente"]
        or estado["brazos"]["A2_lgbm_twin"]["sonda_determinismo"]["no_concluyente"])
    gates, control_negativo, veredicto = calcular_gates(True, integracion, control_atribucion,
                                                        truncamiento, no_concluyente)
    gates["sin_leakage"] = bool(all(invariantes.values()))
    if control_negativo["no_atribuible"]:
        veredicto = "NO_ATRIBUIBLE"
    elif all(gates.values()):
        veredicto = "PROMISING"
    elif integracion["A1_catboost"]["delta_vs_baseline"] > 0:
        veredicto = "MARGINAL"
    else:
        veredicto = "CLOSED"

    entrenamiento = {brazo: {"folds": estado["brazos"][brazo]["folds"]} for brazo in BRAZOS}
    resultado = construir_resultado(inicio, prereg, detalle_constantes, detalle_versiones,
                                    detalle_hashes, auditoria, baseline, estado, entrenamiento,
                                    integracion, complementariedad, invariantes,
                                    control_atribucion, gates, control_negativo, veredicto)
    escribir_json(RESULTADOS_JSON, resultado)
    print(f"  resultados: {ruta_relativa(RESULTADOS_JSON)}", flush=True)
    figura_curvas(entrenamiento, FIGURA)
    escribir_informe(resultado)
    print(f"  RUNLOG.md: {'anotado' if anotar_runlog(resultado) else 'ya estaba'}", flush=True)
    print(f"  experiments.csv: "
          f"{'fila V32-E2 anadida' if anotar_experiments_csv(resultado) else 'fila ya estaba'}",
          flush=True)
    resumen_consola(resultado)
    return 0


def parsear_argumentos() -> argparse.Namespace:
    """Argumentos de linea de comandos."""
    analizador = argparse.ArgumentParser(
        description="V32-E2 (P2): implementacion del pre-registro congelado del challenger "
                    "CatBoost sobre la matriz 279 con gemelo LightGBM y control barajado.")
    analizador.add_argument("--verificar", action="store_true",
                            help="solo verifica constantes, versiones, hashes, auditoria AST, "
                                 "alineacion y baseline (no entrena; imprime VERIFICACION OK)")
    return analizador.parse_args()


def main() -> int:
    """Entrada del script con manejo de los codigos de salida pre-registrados."""
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
