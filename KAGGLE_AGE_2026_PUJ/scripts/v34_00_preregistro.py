"""V34 - E1 - Pre-registro CONGELADO: falsacion de la linea FTRL-Proximal con regla de margen.

## Que congela

Unico experimento del ciclo V34 (autorizado por Carlos el 2026-09-30, opcion B de la consulta de
cierre). Su caracter no es el de un intento de mejora: es una **FALSACION con la forma mas fuerte
de la hipotesis**. La exploracion metodologica previa (`scripts/v34_00b_exploracion_alpha.py`,
sha256 `9d898ab8cf5908e...`, 27 celdas) mostro que **no existe alpha sano**: con
alpha en [.0005, .5] x lambda2 en {.05, .2, 1, 1e-6} x power_t en {0, .25, .5, 1} x L2 en
acumulador o denominador, **0 de 27 celdas** bajan de la entropia del prior `H(prior) = 1.7914537`
del subconjunto explorado; la mejor celda (alpha 0.176316, lambda2 1.0, power_t 1.0, denominador)
termina en 1.7917638, es decir `ln 6 = 1.7917595` mas 4.3e-6: **uniforme, cero aprendizaje**. La
peor celda (alpha 0.08, power_t 0.0) reproduce la firma de colapso de D-083 (ll_train 6.62 ->
28.86). El gemelo SGD con el mismo presupuesto llega a 0.1580395, luego el presupuesto NO es el
limite: lo es el mecanismo.

Este pre-registro congela la regla de alpha en forma cerrada, el presupuesto con early stopping
interno, el control gemelo con su enmienda declarada, la puerta fail-fast G0-G6, la lista
exhaustiva de la familia que se cierra y el nodo de arbol que solo puede concederse tras un triage
formal. El runner (`scripts/v34_01_ftrl_softmax_multipasada.py`) **no repite las 27 celdas**: solo
publica la falsacion reproducible de su propia corrida (`ll_train` y `D_acum` por pasada de A1 y
A2, razon `|sigma*w|/|g|`).

## Diagnostico que fija el diseno

El punto fijo de FTRL es `w_j ~ alpha*sqrt(k)`: norma **no acotada**, que persigue las features
consistentes. SGD con schedule `1/(alpha*(t0+t))` tiene pesos **acotados**. Con un solo `alpha` se
reescala una direccion no acotada, asi que o el amortiguamiento proximal domina (V33-E1: razon
`|sigma*w|/|g|` = 0.042, inercia) o los pesos desbordan (D-083: divergencia). El diseno ataca el
segundo efecto con una **regla de margen de una sola ecuacion**, y el mecanismo queda medido.

## Condiciones incorporadas (kaggle-specialist, LEGAL con condiciones)

- `d_norm` y el conteo de filas se calculan sobre el **subconjunto ES-train** (el 90 % del
  fold-train que realmente se entrena), no sobre el fold-train completo: el ES consume el 10 %,
  luego derivar sobre el conjunto completo describiria un dataset que no es el que se entrena.
- Las tres anclas se declaran con **unidad, fuente y sensibilidad**: `M = ln(K-1) = 1.6094379124341003`
  es una constante DECLARADA y arbitraria (`alpha*` es lineal en `M`, luego la sensibilidad es
  proporcional); `d_norm` viene del prior **medido**; `P_max = 10` es el ancla de D-083 y entra a la
  vez como presupuesto y como ancla, luego hay **un solo grado de libertad** y cualquier cambio
  posterior de `P_max` re-deriva `alpha*` (prohibido tocarlo tras la ejecucion).
- Las 27 celdas se **declaran** (semilla, config, script, sha256) como exploracion metodologica
  previa: datos de fold-train, metrica de perdida, sin seleccion, y **ningun umbral de G0-G4 se
  calibro con ellas**. No se repiten dentro del runner (seria tuning disfrazado).
- El nodo `LINEA_ONLINE_CERRADA_POR_MECANISMO` exige **triage formal** (defecto de implementacion vs
  mecanismo definitivo) decidido ANTES de elegir nodo, la lista exhaustiva de la familia con el
  estado de cada miembro (lo no medido queda EXCLUIDO/NO CUBIERTO, nunca "cerrado") y es **one-shot**.
- Enmienda pre-ejecucion 1 (control): el control de D-083/V33-E1 recibia la matriz **con sesgo
  pre-anadido** y ademas `fit_intercept=True` (doble sesgo). Se corrige: **un solo camino de sesgo
  por brazo**, verificado por sonda (max|gradiente de la coordenada de sesgo| en A1 y paridad con
  `intercept_` de A2). Motivacion de correccion, no de conveniencia.
- `lambda2 = 1.0` en FTRL frente a `alpha_reg = 1e-6` en SGD se declara **NO-EQUIVALENCIA
  ESTRUCTURAL** (con `lambda2 = 1e-6` el L2-en-acumulador y el L2-en-denominador dan resultado
  identico: es un no-op, medido). Se anade sonda de que el veredicto **no se apoya** en esa
  asimetria: el cierre se sostiene en A1 frente a la DERIVACION, no en A1 frente a A2.
- "Se restaura el mejor w" es seleccion de modelo **sobre train**, nunca sobre fold-val; el ES
  interno **no elige entre brazos** (si lo hiciera seria seleccion de hiperparametro y ademas
  romperia la simetria).

## Invariantes

Sin test, sin holdout, sin leaderboard, sin submission, sin datos externos, sin transformers, LLMs
ni embeddings (0); campeon V30 + plug-in intacto; un experimento = un script nuevo; no se edita
ningun script ya ejecutado (`v33_01` y sus artefactos quedan congelados) ni se toca
`artifacts/p33_scores/todos`.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src import config as cfg
from src import data as datos
from src.verificacion_numerica import autotest_numerico

FECHA = "2026-09-30"
CANONICO = ROOT / "artifacts" / "p33_scores" / "todos"
PREREG_MD = cfg.REPORTS / "v34_preregistro.md"
PUERTA_MD = cfg.REPORTS / "v34_convergence_gate.md"
PREREG_JSON = cfg.EXPERIMENTS / "v34_00_preregistro.json"
EXPLORA_JSON = cfg.EXPERIMENTS / "v34_00b_exploracion_alpha.json"
EXPLORA_MD = cfg.REPORTS / "v34_exploracion_alpha.md"
EXPLORA_SCRIPT = ROOT / "scripts" / "v34_00b_exploracion_alpha.py"

BASELINE = 0.6159732
NESTED = 0.6154643
PESOS = [1.0, 1.1749, 1.2721, 1.163, 1.001, 0.8128]
N_CLASES = 6
SEMILLA = 20261010
SEMILLA_EXPLORA = 7
HILOS = 32
N_BOOTSTRAP = 2000
GATE_DELTA = 0.0015
GATE_FOLDS = 4
VARIABILIDAD = 0.0008876
LIMITE_PROYECTADO_H = 20.0
LIMITE_GLOBAL_H = 24.0
RAM_TOPE_GB = 20.0
SEGUNDOS_DECLARADOS = 1200.0
TRIVIAL_MAYORIA_ESPERADO = 0.04854212180687087
GEMELO_SGD_D083 = 0.4934789
LOG_UNIFORME_K6 = 1.791759469228055
LOG_K_MENOS_1 = 1.6094379124341003

P_MAX = 10
P_MIN = 3
PACIENCIA = 2
MEJORA_MINIMA = 1e-4
TOL_PASADA = 1e-6
TEST_SIZE_ES = 0.10
POWER_T = 0.0
LAMBDA1 = 0.0
LAMBDA2 = 1.0
BETA = 1.0
X0 = 1.0
M_MARGEN = LOG_K_MENOS_1
FACTOR_G0 = 2.0
SUELO_G1 = 0.05
H_PRIOR_EXPLORA = 1.7914536815044917
FACTOR_G4 = (0.5, 1.5)
TOL_SUMAS = 1e-9
N_FILAS_EXPLORA = 30000

VECTORIZADOR = {"analyzer": "word", "ngram_range": [1, 2], "min_df": 3, "max_df": 0.5,
                "sublinear_tf": True, "lowercase": True, "strip_accents": None,
                "token_pattern": r"(?u)\\b\\w+\\b"}

ALPHA_REGLA = {
    "formula": "alpha* = M*(2 - power_t)/(d_norm * P_max**(2 - power_t))",
    "con_power_t_cero": "alpha* = 2*M/(d_norm * P_max**2)",
    "M": {"valor": M_MARGEN, "definicion": "ln(K-1) = ln(5)", "unidad": "log-odds (nats)",
          "fuente": "constante DECLARADA, ancla ya usada en V33-E1 (|log(1/5)|)",
          "sensibilidad": "alpha* es lineal en M: si M se multiplica por c, alpha* se multiplica "
                          "por c (sensibilidad proporcional 1.0)"},
    "d_norm": {"definicion": "sqrt(1 - sum_c prior_c^2) sobre el prior MEDIDO del subconjunto "
                             "ES-train del fold",
               "unidades": "adimensional (0, 1]",
               "fuente": "medido en runtime con las filas que se entrenan; NO el prior del "
                         "fold-train completo (el ES consume el 10 %)",
               "sensibilidad": "alpha* es proporcional a 1/d_norm; con prior uniforme d_norm "
                               "= 0.9128709 y alpha* seria 0.0352631 (valor de referencia "
                               "calculado con el prior medido del fold-train completo en la "
                               "exploracion del arquitecto)"},
    "P_max": {"valor": P_MAX, "unidad": "pasadas", "fuente": "ancla D-083: el presupuesto con el "
                                                            "que el gemelo SGD aprendio",
              "nota": "entra A LA VEZ como presupuesto y como ancla: hay UN SOLO grado de libertad "
                      "(no es busqueda); cambiarlo re-deriva alpha* y esta prohibido tras la "
                      "ejecucion"},
    "valores_evaluados": 0,
    "aplicacion": "se calcula UNA vez (fold 0), se congela y se aplica a los 5 folds; si no "
                  "reproduce la formula en runtime -> exit 2 ANTES de entrenar",
    "esperado_aproximado": 0.0352631,
}
BUDGETO = {
    "pasadas_maximas": P_MAX, "pasadas_minimas": P_MIN, "paciencia": PACIENCIA,
    "mejora_minima": MEJORA_MINIMA,
    "orden_por_pasada": "np.random.default_rng(20261010+fold+pasada).permutation de las filas "
                        "ES-train; un orden distinto por pasada y por fold",
    "particion_es": {
        "generador": "sklearn.model_selection.StratifiedShuffleSplit(n_splits=1, "
                     f"test_size={TEST_SIZE_ES}, random_state=20261010+fold) sobre las filas de "
                     "fold-train",
        "metrica": "macro-F1 del brazo sobre el 10 % de TRAIN (nunca fold-val)",
        "restauracion": "al parar se restaura el mejor w (seleccion sobre TRAIN)",
        "simetria": "ambos brazos reciben el MISMO P_max, la MISMA particion (sha256 de los "
                    "indices), los MISMOS ordenes por pasada (sha256) y reportan su pasada de "
                    "parada; el ES no elige entre brazos",
    },
    "prohibido": "usar fold-val, test o el leaderboard para parar, seleccionar o comparar brazos",
}
CONTROL = {
    "estimator": "sklearn.linear_model.SGDClassifier",
    "loss": "log_loss", "alpha_reg": 1e-06, "learning_rate": "optimal", "shuffle": False,
    "fit_intercept": True, "max_iter": P_MAX, "tol": None, "random_state": "20261010+fold",
    "n_jobs": HILOS, "pasadas": P_MAX,
    "matriz": "la MISMA TF-IDF por fold, **SIN columna de sesgo pre-anadida** (un solo camino de "
              "sesgo por brazo)",
    "por_que": "es el precedente que SI aprende (0.3956246 en V33-E1 con una pasada; 0.4934789 con "
               "diez en D-083) y por eso demuestra que el presupuesto no es el limite",
    "enmienda_pre_ejecucion_1": {
        "defecto_corregido": "D-083 y V33-E1 entregaron al control la matriz con sesgo pre-anadido "
                             "Y fit_intercept=True (doble sesgo)",
        "motivacion": "correccion de un defecto de implementacion, NO de conveniencia: no altera "
                      "el criterio de exito y no favorece a A1",
        "exigencia": "tabla de simetria de sesgo con UN solo camino por brazo, verificada por "
                     "sonda: max|gradiente de la coordenada de sesgo| en A1 y paridad con "
                     "intercept_ de A2",
    },
    "asimetria_declarada": {
        "lambda2_ftrl": LAMBDA2, "alpha_reg_sgd": 1e-06,
        "no_equivalencia_estructural": "con lambda2 = 1e-6 el L2-en-acumulador y el "
                                      "L2-en-denominador dan resultado IDENTICO (es un no-op, "
                                      "medido en la exploracion previa): por eso no existe una "
                                      "equivalencia FTRL<->SGD a escala 1e-6",
        "sonda_exigida": "publicar que el veredicto NO se apoya en esta asimetria: el cierre se "
                         "sostiene en A1 frente a la DERIVACION (la regla y la exploracion), no "
                         "en A1 frente a A2",
    },
}
PUERTA = {
    "momento": "dentro del fold 0, ANTES de los folds 1-4 y ANTES de integrar",
    "aplica_a": ["A1_ftrl_multipasada", "A2_sgd_multipasada"],
    "fold_0": "consumido-diagnostico: G3 y G4 leen el fold-val del fold 0, que es parte de la "
              "validacion canonica; su fallo NO toca delta",
    "consecuencia_si_falla": "veredicto NO_ATRIBUIBLE_BY_GATE: STOP documentado, sin fallback, sin "
                             "segundo alpha, sin integrar, sin bootstrap",
    "gates": {
        "G0_objetivo_de_la_regla": {
            "criterio": "A1: M/2 <= D_acum <= 2*M con D_acum = rango medio de logits (max - min) "
                       "sobre ES-train al parar; A2: ll_train <= H(prior)_ES - 0.01",
            "anclaje": "M = ln(5) separa la tasa base del uniforme; factor 2 declarado cubre la "
                       "linealizacion de primer orden",
            "que_es_G0": "es un gate NUEVO de este ciclo: comprueba que la regla de alpha "
                         "consiguio su objetivo declarado, no solo que hubo perdida",
        },
        "G1_banda_ll_train": {
            "criterio": f"{SUELO_G1} <= ll_train <= H(prior)_ES - 0.01",
            "anclaje": "piso 0.05 = detector de humo; techo anclado en la entropia del prior "
                       "MEDIDO del ES-train, NO en ln 6: exigir 0.01 bajo el prior obliga a "
                       "aprender algo real (el gemelo sano da 0.1580)",
            "nota": "el techo de D-084 (0.9*ln6 = 1.6125835223052496) queda muy por debajo del "
                    "prior medido y habria feito por amplitud, no pormargen de aprendizaje",
        },
        "G2_progreso": {
            "criterio": "la mejor ll_train por pasada mejora en >= 1 de las primeras P_min = 3 "
                        "pasadas",
            "anclaje": "un modelo que no aprende nada tiene ll_train plana; tolerancia 1e-6",
        },
        "G3_macro_f1_fold_val": {
            "criterio": "macro-F1 standalone en el fold-val >= 0.30",
            "anclaje": f"trivial {TRIVIAL_MAYORIA_ESPERADO} y gemelo SGD de D-083 {GEMELO_SGD_D083}",
        },
        "G4_frecuencias": {
            "criterio": "frecuencia predicha por clase en [0.5*prior_c, 1.5*prior_c]",
            "anclaje": "prior MEDIDO del ES-train: deriva de los datos de train, no de un objetivo",
            "mejora_sobre_D084": "D-084 uso la banda arbitraria [0.10, 0.30], que rechazo al "
                                 "gemelo sano (42.6 % en la clase 5) y al propio prior",
        },
        "G5_determinismo": {"criterio": "max|delta p| = 0.0 EXACTO entre dos corridas identicas "
                                        "del fold 0 (ambos brazos)"},
        "G6_suma_unidad": {"criterio": f"max|sum_c p_c - 1| <= {TOL_SUMAS:g}", "anclaje": "softmax "
                                                                                  "de 6 logits"},
    },
}
ARBOL = {
    "NO_ATRIBUIBLE_BY_GATE": "la puerta falla en A1 o A2: STOP documentado, sin fallback, sin "
                             "segundo alpha, sin integrar, sin bootstrap; delta intacto",
    "LINEA_ONLINE_CERRADA_POR_MECANISMO": {
        "semantica": "cierre de FAMILIA, NO un gate de delta: se concede SOLO si (a) triage formal "
                     "registrado que clasifica el fallo como MECANISMO_DEFINITIVO y no como "
                     "DEFECTO_DE_IMPLEMENTACION (bug, sesgo duplicado, presupuesto truncado, ES "
                     "mal configurado) decidido ANTES de elegir nodo; si es defecto -> se corrige "
                     "y se re-emite; (b) la tabla de mecanismo esta PUBLICADA dentro del runner "
                     "(ll_train y D_acum por pasada de A1 y A2, razon |sigma*w|/|g|), no "
                     "importada de /tmp; (c) la lista exhaustiva de la familia esta registrada "
                     "con el estado de cada miembro",
        "condiciones": "si A1 falla y A2 pasa G0-G4; el delta queda intacto y vacio en informe e "
                       "historial, con el motivo de no calculo",
        "one_shot": "reabrir la familia exige acta nueva con hipotesis mecanica nueva, no otro gate",
    },
    "PROMISING": "gates completos y control favorable -> segunda particion SOLO con autorizacion "
                 "del usuario; el script no la ejecuta",
    "MARGINAL": "delta > 0 sin gates completos, o con cap activo",
    "CLOSED": "delta <= 0",
    "submission": "jamas automatica; requeriria gate completo + segunda particion + "
                  "autorizacion explicita",
}
FAMILIA = [
    {"miembro": "FTRL-Proximal propio, K=6 softmax (un solo juego de pesos)", "estado": "CUBIERTO",
     "evidencia": "V33-E1 (D-084, una pasada) y este ciclo (V34, multi-pasada con ES interno)"},
    {"miembro": "FTRL-Proximal OvR (6 sigmoides)", "estado": "CUBIERTO",
     "evidencia": "D-083 (diez pasadas, alpha 0.05, divergencia)"},
    {"miembro": "power_t (decaimiento) en {0.25, 0.5, 1.0}", "estado": "CUBIERTO",
     "evidencia": "exploracion previa (27 celdas): power_t con su alpha de margen da ll_train "
                  "1.8094 / 1.7929 / 1.7918, nunca bajo H(prior)"},
    {"miembro": "lambda2 en {.05, .2, 1, 1e-6} y L2 en acumulador vs denominador",
     "estado": "CUBIERTO", "evidencia": "exploracion previa (27 celdas); con lambda2 = 1e-6 ambas "
                                        "variantes L2 son identicas (no-op medido)"},
    {"miembro": "presupuesto de una pasada", "estado": "CUBIERTO",
     "evidencia": "V33-E1 (A1 y A2 con una pasada) y D-083 (diez pasadas)"},
    {"miembro": "FTRL-Proximal con features preentrenadas o normalizacion aprendida",
     "estado": "EXCLUIDO", "evidencia": "prohibido por las reglas (embeddings, preentrenados o "
                                        "entrenados) y sin cobertura historica"},
    {"miembro": "FTRL-Proximal con regularizacion L1 (lambda1 > 0)", "estado": "NO CUBIERTO",
     "evidencia": "no medido; queda EXCLUIDO del cierre de familia, no se declara cerrado"},
    {"miembro": "FTRL-Proximal multiclase con matriz de pesos por clase y L2 normalizado fuera "
                "de FTRL", "estado": "NO CUBIERTO", "evidencia": "no medido"},
]
SONDAS = {
    "a_vectorizado_vs_lazo": {"tolerancia": 1e-12, "nota": "reutilizada de D-084"},
    "b_raiz_analitica_lambda0": {"residuo_max": 1e-9},
    "c_totalf1_macro": {"tolerancia": 1e-6},
    "d_sin_nan_logloss_finito": True,
    "e_logloss_multiclase_propio": {"tolerancia": 1e-9},
    "f_filas_suman_uno": {"tolerancia": 1e-9},
    "g_reproducibilidad_alpha": {"criterio": "alpha*, d_norm y prior recalculados en runtime desde "
                                             "el ES-train; |alpha*_run - alpha*_formula| = 0 "
                                             "exacto + sha256 de prior y de alpha*; si no "
                                             "reproduce -> exit 2 ANTES de entrenar"},
    "h_estabilidad_es": {"criterio": "dos corridas identicas: misma pasada de parada, mismo "
                                     "ll_train y sha256 de w identicos (tolerancia 0)"},
    "i_simetria_presupuesto": {"criterio": "sha256 de la particion ES y de los ordenes por pasada "
                                           "identicos en ambos brazos"},
    "j_tabla_de_mecanismo": {"criterio": "publicar ll_train y D_acum por pasada de A1 y A2, mas la "
                                         "razon |sigma*w|/|g|: la falsacion se reproduce DENTRO "
                                         "del runner, NO se importa de /tmp"},
    "politica": "si alguna falla -> exit 4 ANTES de entrenar; cualquier ajuste exige enmienda "
                "pre-ejecucion",
}
ENTREGABLES = {
    "exploracion_previa": "scripts/v34_00b_exploracion_alpha.py (ya ejecutado) + "
                          "experiments/v34_00b_exploracion_alpha.json + "
                          "reports/v34_exploracion_alpha.md",
    "preregistro": "scripts/v34_00_preregistro.py -> experiments/v34_00_preregistro.json, "
                   "reports/v34_preregistro.md, reports/v34_convergence_gate.md (NORMATIVO)",
    "runner": "scripts/v34_01_ftrl_softmax_multipasada.py -> "
              "experiments/v34_01_ftrl_softmax_multipasada.json, experiments/v34_01_estado.json, "
              "artifacts/p34_scores/v34_e1/{a1_ftrl_multipasada,a2_sgd_multipasada}.npy, "
              "checkpoints/v34_e1/fold{k}_{brazo}.npz (w, z, n, pasada_de_parada) con escritura "
              "atomica, reports/v34_e1_ftrl_multipasada.md, "
              "reports/figures/v34_e1_curvas.png, reports/v34_novelty_registry.csv (NUEVO), fila "
              "anexada en reports/age_complete_experiment_history.csv, RUNLOG.md y fila V34-E1 en "
              "experiments.csv",
    "acta": "D-085 con: clasificacion de triage, tabla de las 27 celdas y su lectura, anclas "
            "M/d_norm/P_max con sensibilidad, tabla de simetria de sesgo, estado de sondas (a)-(j), "
            "lista de miembros de familia cubiertos vs no cubiertos, y confirmacion de que NO se "
            "toco p33_scores/todos ni v33_01, sin submission, sin LB y sin segunda particion",
    "rutas_prohibidas": "artifacts/p33_scores/todos/** y cualquier script ya ejecutado",
}


def sha256_bytes(datos_bytes: bytes) -> str:
    return hashlib.sha256(datos_bytes).hexdigest()


def sha256_archivo(ruta: Path) -> str:
    return sha256_bytes(ruta.read_bytes())


def main() -> int:
    autotest_numerico()
    inicio = time.time()

    clase = np.load(CANONICO / "y_true.npy")
    fold_idx = np.load(CANONICO / "fold_idx.npy")
    v30 = np.load(CANONICO / "v30_prob.npy").astype(np.float64)
    ids_canon = np.load(CANONICO / "ids.npy")
    v30 = np.clip(v30, 1e-12, 1.0)
    v30 = v30 / v30.sum(axis=1, keepdims=True)
    w = np.array(PESOS, dtype=float)
    baseline = float(f1_score(clase, np.argmax(v30 * w, axis=1), average="macro",
                              labels=list(range(N_CLASES))))
    print(f"baseline primario recomputado: {baseline:.7f} (esperado {BASELINE})", flush=True)
    if abs(baseline - BASELINE) > 5e-7:
        print("[STOP] el baseline canonico no reproduce", flush=True)
        return 1
    assert clase.shape == fold_idx.shape == ids_canon.shape == (169336,), "canonico desalineado"
    assert np.unique(fold_idx).size == 5, "se esperan 5 folds canonicos"

    train = datos.load_train()
    if not np.array_equal(ids_canon.astype(np.int64),
                          train[cfg.ID_COL].to_numpy().astype(np.int64)):
        print("[STOP] el orden de ids del corpus no coincide con la particion canonica", flush=True)
        return 1
    if not np.array_equal(clase.astype(np.int64),
                          train[cfg.TARGET].cat.codes.to_numpy().astype(np.int64)):
        print("[STOP] las etiquetas del corpus no coinciden con y_true canonico", flush=True)
        return 1

    explora = json.loads(EXPLORA_JSON.read_text(encoding="utf-8"))
    sha_script = sha256_archivo(EXPLORA_SCRIPT)
    if sha_script != explora["sha256_del_script"]:
        print("[STOP] el sha256 del script de exploracion no coincide con el declarado", flush=True)
        return 1
    celdas = explora["celdas"]
    if len(celdas) != 27 or not explora["ninguna_celda_bajo_h_prior"]:
        print("[STOP] la exploracion previa no respata lo declarado (27 celdas, 0 bajo el prior)",
              flush=True)
        return 1
    mejor = explora["mejor_celda"]["ll_train"]
    peor = explora["peor_celda"]["ll_train"]
    if not (explora["datos"]["h_prior"] - 1e-12 <= mejor and peor > 10.0):
        print("[STOP] las cifras de la exploracion no son las congeladas", flush=True)
        return 1
    print(f"exploracion previa: 27 celdas | mejor {mejor!r} | peor {peor!r} | "
          f"H(prior) {explora['datos']['h_prior']!r}", flush=True)

    # Ancla de G3 recalculada desde la particion canonica (mismo valor que en D-083/D-084).
    etiquetas, conteo = np.unique(clase, return_counts=True)
    idx_mayor = int(np.argmax(conteo))
    trivial = float(f1_score(clase, np.full_like(clase, etiquetas[idx_mayor]), average="macro",
                             labels=list(range(N_CLASES))))
    if abs(trivial - TRIVIAL_MAYORIA_ESPERADO) > 1e-12:
        print("[STOP] el ancla trivial no reproduce la de D-083/D-084", flush=True)
        return 1
    print(f"ancla G3 - macro-F1 trivial (clase {cfg.CLASSES[idx_mayor]}): {trivial:.16f}", flush=True)

    sha_archivo = sha256_archivo(cfg.TRAIN_FILE)
    textos = train[cfg.TEXT_COL].fillna("").astype(str).to_numpy()
    h = hashlib.sha256()
    for t in textos:
        b = t.encode("utf-8")
        h.update(len(b).to_bytes(8, "little"))
        h.update(b)
    sha_textos = h.hexdigest()
    sha_vectorizador = sha256_bytes(json.dumps(VECTORIZADOR, sort_keys=True).encode("utf-8"))

    hashes = {"corpus_archivo": sha_archivo, "corpus_textos": sha_textos,
              "vectorizador": sha_vectorizador,
              "manifest_canonico": sha256_archivo(CANONICO / "manifest.json"),
              "exploracion_script": sha_script, "exploracion_json": sha256_archivo(EXPLORA_JSON),
              "exploracion_md": sha256_archivo(EXPLORA_MD)}
    for nombre in ("y_true", "fold_idx", "v30_prob", "ids", "cand_pred", "cand_prob"):
        hashes[f"canonico_{nombre}"] = sha256_archivo(CANONICO / f"{nombre}.npy")

    import sklearn
    versiones = {"sklearn": sklearn.__version__, "numpy": np.__version__,
                 "pandas": pd.__version__}
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT.parent, check=False,
                            capture_output=True, text=True, timeout=30).stdout.strip()

    prior_ref = np.asarray(explora["datos"]["prior"], dtype=np.float64)
    prior_ref = prior_ref / prior_ref.sum()
    d_norm_ref = float(np.sqrt(1.0 - np.sum(prior_ref ** 2)))
    alpha_ref = 2.0 * M_MARGEN / (d_norm_ref * P_MAX ** 2)
    print(f"d_norm de referencia (prior medido): {d_norm_ref!r} | alpha* = {alpha_ref!r}", flush=True)

    congelado = {
        "fase": "V34-E1 pre-registro congelado", "fecha": FECHA, "commit": commit,
        "experimento": "FALSACION de la linea FTRL-Proximal: multi-pasada con ES interno y regla "
                       "de margen en forma cerrada para alpha (control: SGD multi-pasada)",
        "carácter": "falsacion con la forma mas fuerte de la hipotesis; NO es un intento de "
                    "mejora; su valor es cerrar la familia con evidencia reproducible",
        "baseline_primario": round(baseline, 7), "referencia_nested": NESTED,
        "pesos_plugin": PESOS, "hashes": hashes, "versiones": versiones,
        "exploracion_previa": {
            "categoria": explora["categoria"], "no_es_un_grid": explora["no_es_un_grid"],
            "justificacion": explora["justificacion"],
            "n_celdas": len(celdas), "presupuesto": explora["presupuesto"], "semilla":
            explora["semilla"], "mejor_celda": explora["mejor_celda"], "peor_celda":
            explora["peor_celda"], "h_prior_subconjunto": explora["datos"]["h_prior"],
            "log_uniforme": explora["datos"]["log_uniforme"],
            "referencia_gemelo_sgd": explora["referencia_gemelo_sgd"],
            "ninguna_celda_bajo_h_prior": True,
            "no_repetida_en_el_runner": True,
            "declaracion": "exploracion metodologica previa sobre filas de fold-train con la sola "
                           "metrica de perdida; ningun valor se selecciono y NINGUN umbral de "
                           "G0-G4 se calibro con estas celdas; no se repite dentro del runner "
                           "(seria tuning disfrazado)",
        },
        "anclas": {"macro_f1_trivial_clase_mayoritaria": trivial,
                   "clase_mayoritaria": cfg.CLASSES[idx_mayor],
                   "gemelo_sgd_d083": GEMELO_SGD_D083,
                   "log_uniforme_k6": LOG_UNIFORME_K6, "log_k_menos_1": LOG_K_MENOS_1,
                   "h_prior_subconjunto_exploracion": H_PRIOR_EXPLORA,
                   "d_norm_referencia": d_norm_ref, "alpha_esperado_para_d_norm_referencia":
                   alpha_ref},
        "corpus": {"receta": "src.data.load_train() sobre data/raw/age_train.csv (sep=cfg."
                             "TRAIN_SEP, encoding utf-8-sig, clean_text aplicado); orden de filas "
                             "= orden del CSV = orden canonico de ids.npy",
                   "hash_contenido": "sha256 con longitud (8 bytes LE) + bytes UTF-8 por texto, "
                                     "en orden",
                   "prohibido": "el data_hash del manifiesto canonico (2ada573b...) queda "
                                "invalidado en D-083: es un hash de punteros, no de contenido"},
        "representacion": {"vectorizador": VECTORIZADOR,
                           "norm_dtype": "defaults de sklearn (norm='l2', dtype float64)",
                           "ajuste": "POR FOLD: fit solo en ES-train (el 10 % del fold-train va "
                                     "al ES interno y NO se vectoriza con el) — declarado por "
                                     "coherencia con la regla de alpha, que se deriva sobre las "
                                     "filas que se entrenan",
                           "sesgo": "A1 materializa el sesgo como coordenada x0=1.0 (un solo "
                                    "camino); A2 usa fit_intercept=True SIN columna de sesgo "
                                    "pre-anadida (enmienda pre-ejecucion 1)"},
        "regla_alpha": ALPHA_REGLA, "budgeto": BUDGETO, "control": CONTROL, "puerta": PUERTA,
        "arbol": ARBOL, "familia_cerrada": FAMILIA, "sondas": SONDAS,
        "publicado_no_gate": ["ll_train por pasada de A1 y A2", "D_acum por pasada de A1 y A2",
                              "razon |sigma*w|/|g| por pasada", "w_max_abs por pasada",
                              "frecuencias predichas por clase al parar"],
        "compliance": {
            "embeddings": 0, "transformers": 0, "llm": 0, "datos_externos": 0, "test": 0,
            "holdout": 0, "leaderboard": 0, "submission": 0,
            "ast_audit": "importes Y literales de ruta: vowpal, vw, torch, transformers, gensim, "
                         "fasttext, sentence-transformers, word2vec, glove, spacy-vectors; y 0 "
                         "rutas de test/holdout/submission/leaderboard",
            "lecturas_de_datos": "solo data/raw/age_train.csv y las 4 particiones reservadas de "
                                 "artifacts/p33_scores/todos; se audita que no se lea nada mas",
            "auditorias_exigidas": ["6 .npy canonicos + manifest", ("sha256 del archivo y del "
                                    "array de textos (nunca el data_hash del manifiesto)"),
                                    "sha256 del vectorizador", "vocabulario e idf por fold",
                                    "sonda de refit en los 5 folds", "sondas (a)-(j)",
                                    "tabla de simetria de sesgo", ("sonda de reproducibilidad de "
                                    "alpha*"), "estabilidad del ES", "simetria de presupuesto",
                                    "RSS pico con ru_maxrss", "escritura atomica"],
        },
        "ejecucion": {
            "coste_medido_en_exploracion": explora["duracion_seg"],
            "coste_estimado_runner": "~590 s (FTRL ~77 s por fold con 10 pasadas, A2 ~1.5 s por "
                                    "fold, sondas ~1 s, refit ~15 s, sondas de determinismo y "
                                    "estabilidad del ES ~158 s)",
            "presupuesto_declarado_seg": SEGUNDOS_DECLARADOS,
            "limite_por_proyeccion_h": LIMITE_PROYECTADO_H, "timeout_global_h": LIMITE_GLOBAL_H,
            "ram_tope_gb": RAM_TOPE_GB, "ram_medicion": "resource.getrusage(ru_maxrss)",
            "rss_esperado": "< 1.5 GB (z, n, w = 3 x 118410 x 6 x 8 B = 17 MB) vs 0.71 GB medido "
                            "en V33-E1",
            "checkpoints": "experiments/v34_01_estado.json y checkpoints/v34_e1/*.npz, resumible "
                           "por fold y por brazo",
        },
        "entregables": ENTREGABLES,
    }

    lineas = [
        "# V34 - E1 - Pre-registro congelado: falsacion de la linea FTRL-Proximal",
        "",
        f"**Fecha**: {FECHA}. **Estado**: CONGELADO antes de ejecutar (commit `{commit[:7]}`).",
        f"**Baseline primario recomputado**: {baseline:.7f} (nested {NESTED}).",
        "",
        "## Caracter del experimento",
        "",
        "> Este NO es un intento de mejora: es una **FALSACION con la forma mas fuerte de la",
        "> hipotesis**. La exploracion previa mostro que no existe alpha sano, y su valor es",
        "> cerrar la familia online con evidencia **reproducible dentro del runner**, sin gastar",
        "> mas gates de delta.",
        "",
        "## Evidencia previa (exploracion metodologica, no es un grid)",
        "",
        (f"- Script `{EXPLORA_SCRIPT.name}` (sha256 `{sha_script[:16]}...`), "
        f"**{len(celdas)} celdas**, presupuesto {explora['presupuesto']['filas']} filas x "
        f"{explora['presupuesto']['pasadas']} pasadas, semilla "
        f"{explora['semilla']['subconjunto']} para el subconjunto y "
        f"`{explora['semilla']['ordenes_por_pasada']}` para los ordenes."),
        (f"- `H(prior)` del subconjunto = {explora['datos']['h_prior']!r}; `ln 6` = "
        f"{explora['datos']['log_uniforme']!r}."),
        (f"- Mejor celda (alpha {explora['mejor_celda']['alpha']}, lambda2 "
        f"{explora['mejor_celda']['lambda2']}, power_t {explora['mejor_celda']['power_t']}, "
        f"{explora['mejor_celda']['variante_l2']}): ll_train = {mejor!r} = uniforme + 4.3e-6."),
        (f"- Peor celda (alpha {explora['peor_celda']['alpha']}, power_t "
        f"{explora['peor_celda']['power_t']}): ll_train = {peor!r}, la firma de colapso de D-083."),
        (f"- **0 de {len(celdas)} celdas** bajan de `H(prior)`; el gemelo SGD con el mismo "
        f"presupuesto llega a {explora['referencia_gemelo_sgd']['ll_train']!r}: el presupuesto NO "
        "es el limite."),
        ("- Declaracion exigida: ningun umbral de G0-G4 se calibro con estas celdas; **no se "
        "repite dentro del runner** (seria tuning disfrazado)."),
        "",
        "## Mecanismo que fija el diseno",
        "",
        "El punto fijo de FTRL es `w_j ~ alpha*sqrt(k)`: norma **no acotada**. SGD con schedule",
        "`1/(alpha*(t0+t))` tiene pesos **acotados**. Con un solo alpha se reescala una direccion no",
        "acotada: o el amortiguamiento proximal domina (V33-E1, razon `|sigma*w|/|g|` = 0.042,",
        "inercia) o los pesos desbordan (D-083, divergencia).",
        "",
        "## Regla de alpha (forma cerrada, 0 valores evaluados)",
        "",
        (f"- `alpha* = 2*M/(d_norm * P_max^2)` con `M = ln(K-1) = {M_MARGEN!r}` "
         f"(constante DECLARADA, linealidad: sensibilidad proporcional 1.0),"),
        (f"  `d_norm = sqrt(1 - sum_c prior_c^2)` del prior **MEDIDO del subconjunto ES-train** "
         f"(NO el del fold-train completo: el ES consume el 10 %), y `P_max = {P_MAX}` "
         f"(ancla D-083)."),
        (f"- Con el prior de referencia (`d_norm = {d_norm_ref!r}`) sale `alpha* = {alpha_ref!r}`; "
         "el runner lo recalcula por fold y verifica reproducibilidad exacta (sonda g, exit 2 si "
         "no reproduce)."),
        ("- `P_max` entra **a la vez** como presupuesto y como ancla: hay **un solo grado de "
        "libertad** (no es busqueda) y cambiarlo re-deriva `alpha*` (prohibido tras ejecutar)."),
        "",
        "## Presupuesto con early stopping interno (nunca fold-val)",
        "",
        (f"- `P_max = {P_MAX}` pasadas, `P_min = {P_MIN}`, paciencia {PACIENCIA}, mejora minima "
         f"{MEJORA_MINIMA:g}; una permutacion distinta por pasada "
         f"(`default_rng(20261010+fold+pasada)`)."),
        (f"- ES interno: `StratifiedShuffleSplit(test_size={TEST_SIZE_ES}, "
        f"random_state=20261010+fold)` sobre **filas de TRAIN**, metrica macro-F1; al parar se "
        "restaura el mejor `w` (seleccion sobre TRAIN)."),
        "- Simetria: ambos brazos reciben el mismo `P_max`, la misma particion (sha256), los mismos",
        "  ordenes (sha256) y reportan su pasada de parada. El ES **no elige entre brazos**.",
        "",
        "## Control y enmienda pre-ejecucion 1",
        "",
        ("- `SGDClassifier(log_loss, alpha=1e-6, learning_rate='optimal', shuffle=False, "
         "fit_intercept=True)`"),
        (f"  con `max_iter = {P_MAX}` sobre la MISMA matriz y **sin columna de sesgo pre-anadida** "
         "(un solo camino de sesgo por brazo)."),
        "- Defecto corregido: D-083 y V33-E1 dar(on al control la matriz con sesgo pre-anadido Y",
        "  `fit_intercept=True` (doble sesgo). Motivacion de correccion, no de conveniencia;",
        "  verificado por sonda (max|gradiente de la coordenada de sesgo| y paridad con",
        "  `intercept_` de A2).",
        "- Asimetria declarada: `lambda2 = 1.0` (FTRL) frente a `alpha_reg = 1e-6` (SGD) es",
        "  **NO-EQUIVALENCIA ESTRUCTURAL** (con `lambda2 = 1e-6` el L2 en acumulador y en",
        "  denominador son identicos: no-op medido). El cierre se sostiene en **A1 frente a la",
        "  DERIVACION**, no en A1 frente a A2, y se publica esa sonda.",
        "",
        "## Puerta fail-fast G0-G6 (fold 0, antes de folds 1-4 y de integrar; ambos brazos)",
        "",
        ("- **G0 (nuevo)**: objetivo de la regla cumplido: A1 `M/2 <= D_acum <= 2*M`; A2 "
         "`ll_train <= H(prior)_ES - 0.01`."),
        (f"- **G1**: banda `[{SUELO_G1}, H(prior)_ES - 0.01]`; el techo se ancla en el prior "
         "**medido** (el de D-084, `0.9*ln6`, quedaba por debajo del prior y fallaba por "
         "amplitud)."),
        f"- **G2**: la mejor `ll_train` mejora en >= 1 de las primeras {P_MIN} pasadas.",
        (f"- **G3**: macro-F1 standalone en fold-val >= 0.30 (anclas {TRIVIAL_MAYORIA_ESPERADO:.7f} "
        f"y {GEMELO_SGD_D083})."),
        "- **G4**: frecuencias por clase en `[0.5*prior_c, 1.5*prior_c]` del prior medido (la banda",
        "  arbitraria [0.10, 0.30] de D-084 rechazo al gemelo sano: clase 5 al 42.6 %).",
        "- **G5**: determinismo `max|delta p| = 0.0` exacto; **G6**: filas que suman 1.",
        "- Fallo -> `NO_ATRIBUIBLE_BY_GATE`: STOP, sin fallback, sin segundo alpha, sin integrar,",
        "  sin bootstrap; el fold 0 queda consumido-diagnostico y su fallo **no toca delta**.",
        "",
        "## Nodo nuevo de cierre por mecanismo (condiciones)",
        "",
        "- Se concede **solo** si (a) hay **triage formal registrado** que clasifica el fallo como",
        "  `MECANISMO_DEFINITIVO` y no como `DEFECTO_DE_IMPLEMENTACION` (bug, sesgo duplicado,",
        "  presupuesto truncado, ES mal configurado), decidido ANTES de elegir nodo; (b) la tabla",
        "  de mecanismo esta **publicada dentro del runner**; (c) la lista exhaustiva de la familia",
        "  esta registrada con el estado de cada miembro. Si un miembro no se midio queda",
        "  `NO CUBIERTO`, nunca \"cerrado\". **One-shot**: reabrir exige acta nueva.",
        "",
        "| Miembro de la familia | Estado | Evidencia |",
        "|---|---|---|",
    ]
    for miembro in FAMILIA:
        lineas.append(f"| {miembro['miembro']} | {miembro['estado']} | {miembro['evidencia']} |")
    lineas += [
        "",
        "## Sondas obligatorias antes de entrenar (exit 4)",
        "",
        ("(a) vectorizado == lazo (1e-12); (b) raiz analitica con lambda 0 (1e-9); "
        "(c) TotalF1:average=Macro == macro-F1 (1e-6);"),
        ("(d) sin NaN/inf; (e) log_loss multiclase propio == sklearn (1e-9); (f) filas que suman 1 "
        "(1e-9);"),
        ("(g) **reproducibilidad de `alpha*`** (error 0 + sha256 de prior y alpha*, exit 2 si no "
        "reproduce);"),
        ("(h) **estabilidad del ES** (dos corridas identicas: misma pasada de parada, sha256 de w "
        "identico, tolerancia 0);"),
        ("(i) **simetria de presupuesto** (sha256 de particion ES y de ordenes iguales en ambos "
        "brazos);"),
        ("(j) **tabla de mecanismo** publicada dentro del runner (`ll_train` y `D_acum` por pasada "
        "de A1 y A2, razon"),
        "  `|sigma*w|/|g|`): la falsacion se reproduce aqui, NO se importa de /tmp.",
        "",
        "## Materiales congelados (SHA256, 16)",
        "",
        "| Artefacto | SHA256 |",
        "|---|---|",
    ]
    for nombre, valor in hashes.items():
        lineas.append(f"| {nombre} | `{valor[:16]}...` |")
    lineas += [
        "",
        "## Compliance, invariantes y coste",
        "",
        "- 0 embeddings, 0 transformers, 0 LLM, 0 datos externos; sin test, sin holdout, sin",
        "  leaderboard, sin submission; un script nuevo; campeon V30 + plug-in (LB 0.63575)",
        "  intacto; NO se edita `v33_01` ni ningun script ejecutado ni se toca",
        "  `artifacts/p33_scores/todos`.",
        ("- AST audit de imports y literales; lecturas de datos limitadas a `data/raw/age_train.csv` "
        "y las 4 particiones reservadas."),
        (f"- Coste esperado ~590 s con presupuesto declarado de {SEGUNDOS_DECLARADOS:g} s; RSS < "
        f"1.5 GB con tope {RAM_TOPE_GB:g} GB medido con `ru_maxrss`; checkpoints por fold y por "
        "brazo con escritura atomica."),
        f"- Referencia de variabilidad de entrenamiento: {VARIABILIDAD}.",
        (f"- La exploracion previa tardo {explora['duracion_seg']:.0f} s con RSS "
        f"{explora['rss_pico_gb']:.2f} GB."),
    ]
    texto = "\n".join(lin.rstrip() for lin in lineas) + "\n"
    PREREG_MD.write_text(texto, encoding="utf-8")
    print(f"  {PREREG_MD.name}: {len(texto.splitlines())} lineas", flush=True)

    filas_puerta = [
        "| Gate | Criterio | Anclaje | Tolerancia |",
        "|---|---|---|---|",
        (f"| G0 | A1: M/2 <= D_acum <= 2*M con D_acum = rango medio de logits; A2: ll_train <= "
         f"H(prior)_ES - 0.01 | M = ln(5) = {LOG_K_MENOS_1!r}; factor 2 declarado | 1e-9 |"),
        (f"| G1 | {SUELO_G1} <= ll_train <= H(prior)_ES - 0.01 | piso = detector de humo; techo "
         f"anclado en el prior MEDIDO del ES-train (NO en ln 6 = {LOG_UNIFORME_K6!r}) | 1e-6 |"),
        (f"| G2 | la mejor ll_train mejora en >= 1 de las primeras {P_MIN} pasadas | un modelo que "
         f"no aprende tiene ll_train plana | {TOL_PASADA:g} |"),
        (f"| G3 | macro-F1 standalone en el fold-val >= 0.30 | trivial "
         f"{TRIVIAL_MAYORIA_ESPERADO:.7f} (clase {cfg.CLASSES[idx_mayor]}) y gemelo SGD de D-083 "
         f"{GEMELO_SGD_D083} | 1e-9 |"),
        ("| G4 | frecuencia predicha por clase en [0.5*prior_c, 1.5*prior_c] | prior MEDIDO del "
         "ES-train | 1e-9 |"),
        ("| G5 | max|delta p| = 0.0 entre dos corridas identicas del fold 0 | determinismo CPU "
         "esperado exacto | 0 |"),
        (f"| G6 | max|sum_c p_c - 1| <= {TOL_SUMAS:g} | softmax de 6 logits | {TOL_SUMAS:g} |"),
        "",
        ("Los seis umbrales estan **DERIVADOS**, no calibrados con datos: G1 y G4 se anclan en el "
        "prior"),
        ("**medido** del ES-train (datos de train), G3 en las referencias ya congeladas de D-083 y "
        "D-084, y"),
        ("G0 en el margen declarado de la propia regla. **Ninguno se calibro con las 27 celdas de la "
        "exploracion"),
        "metodologica previa.**",
        "",
        "## Cuando corre y que pasa si falla",
        "",
        ("- Corre dentro del **fold 0**, antes de los folds 1-4 y antes de integrar, e **igual** a A1 "
        "y a A2."),
        ("- G3 y G4 leen el fold-val del fold 0, parte de la validacion canonica: el fold 0 queda "
        "registrado"),
        "  como **consumido-diagnostico** y su fallo **no toca delta**.",
        ("- Si falla cualquiera: `NO_ATRIBUIBLE_BY_GATE`, sin fallback, sin segundo alpha (reintentar "
        "seria grid),"),
        "  sin integracion y sin bootstrap, con salida 0 tras dejar JSON, estado e informe.",
        ("- Si A1 falla y A2 pasa G0-G4, el nodo candidato es `LINEA_ONLINE_CERRADA_POR_MECANISMO` "
        "(cierre de FAMILIA),"),
        ("  y **solo** se concede tras triage formal registrado y con la tabla de mecanismo "
        "publicada dentro del runner."),
    ]
    PUERTA_MD.write_text("\n".join(lin.rstrip() for lin in filas_puerta) + "\n", encoding="utf-8")
    print(f"  {PUERTA_MD.name}: "
          f"{len(PUERTA_MD.read_text(encoding='utf-8').splitlines())} lineas", flush=True)

    PREREG_JSON.write_text(json.dumps(congelado, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  {PREREG_JSON.name}", flush=True)

    runlog = ROOT / "RUNLOG.md"
    if "Pre-registro de V34-E1" not in runlog.read_text(encoding="utf-8"):
        with runlog.open("a", encoding="utf-8") as f:
            f.write("\n---\n\n## Pre-registro de V34-E1 (falsacion de la linea FTRL) · " + FECHA
                    + "\n\n"
                    "- Exploracion metodologica previa versionada: 27 celdas, 0 bajo H(prior); mejor "
                    "celda = uniforme\n  (1.7917638) y peor celda = 28.864 (firma de D-083); gemelo "
                    "SGD sano = 0.1580395.\n"
                    "- Congelado: alpha* = 2*ln(5)/(d_norm*P_max^2) con d_norm del prior MEDIDO del "
                    "ES-train y P_max = 10 (ancla\n  D-083), forma cerrada con 0 valores evaluados; "
                    "10 pasadas con ES interno 90/10 sobre TRAIN (paciencia 2, minimo 3);\n"
                    "  control SGD multi-pasada con enmienda pre-ejecucion del doble sesgo; puerta "
                    "G0-G6 con G0 nuevo y anclas en el\n  prior medido; nodo nuevo "
                    "LINEA_ONLINE_CERRADA_POR_MECANISMO con triage formal; sondas (a)-(j).\n"
                    "- La falsacion se reproduce DENTRO del runner; las 27 celdas NO se repiten.\n"
                    f"- Baseline recomputado: {baseline:.7f}; ancla trivial: {trivial:.16f}. Sin "
                    "test, sin holdout,\n  sin submission, campeon intacto.\n")
        print("  RUNLOG.md: pre-registro anotado", flush=True)

    ruta_exp = ROOT / "experiments.csv"
    lineas_exp = ruta_exp.read_text(encoding="utf-8").splitlines()
    cols = [c.strip() for c in lineas_exp[0].split(",")]
    if "V34-E1-PREREG" not in {lin.split(",", 1)[0] for lin in lineas_exp[1:] if lin}:
        fila = {"experiment_id": "V34-E1-PREREG", "fecha": FECHA, "bloque": "34.0",
                "etapa": "V34-E1 pre-registro congelado: falsacion de la linea FTRL-Proximal con "
                        "regla de margen en forma cerrada",
                "pregunta": "existe algun alpha con el que FTRL-Proximal supere el predictor de "
                            "prior sobre TF-IDF L2-normalizada?",
                "hipotesis": "no: el punto fijo w_j ~ alpha*sqrt(k) tiene norma no acotada frente a "
                             "los pesos acotados de SGD, luego alpha es la palanca equivocada",
                "features": "TF-IDF por fold (receta v31_02) ajustada en ES-train; 0 embeddings",
                "preprocessing": "vocabulario e idf ajustados SOLO en ES-train; sesgo como "
                                 "coordenada x0=1.0 solo en A1",
                "representacion": "TF-IDF dispersa (misma receta, artefactos nuevos)",
                "modelo": "FTRL-Proximal softmax K=6 multi-pasada con ES interno (A1) / "
                          "SGDClassifier log_loss multi-pasada (A2)",
                "hiperparametros": "alpha* = 2*ln(5)/(d_norm*P_max^2) DERIVADO (0 valores "
                                   "evaluados), P_max 10, P_min 3, paciencia 2, lambda1 0, "
                                   "lambda2 1, beta 1, power_t 0, x0 1, seed 20261010+fold",
                "validacion": "puerta G0-G6 fail-fast en fold 0 (ambos brazos) + gates de la "
                              "serie; sondas (a)-(j); tabla de mecanismo publicada en el runner",
                "macro_f1_mean": "", "macro_f1_std": "", "weighted_f1_mean": "",
                "micro_f1_mean": "", "accuracy_mean": "",
                "resultado": "pre-registro escrito y congelado; sin resultados aun",
                "interpretacion": "la exploracion previa de 27 celdas (0 bajo H(prior)) es "
                                  "evidencia de falsacion, no un grid",
                "decision": "ejecutar V34-E1 tal cual",
                "duracion_seg": "0.0",
                "artefactos": "reports/v34_preregistro.md | "
                              "reports/v34_convergence_gate.md | "
                              "experiments/v34_00_preregistro.json | "
                              "reports/v34_exploracion_alpha.md"}
        buf = io.StringIO()
        csv.writer(buf, lineterminator="\n").writerow([fila[c] for c in cols])
        with ruta_exp.open("a", encoding="utf-8") as f:
            f.write(buf.getvalue())
        print("  experiments.csv: fila V34-E1-PREREG", flush=True)

    print(f"\n[fin] {time.time() - inicio:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
