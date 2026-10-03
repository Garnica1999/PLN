"""V33 - E1 - Pre-registro CONGELADO: FTRL-Proximal softmax multiclase, una pasada online.

## Que congela

Primer experimento del expediente V33 (autorizado por Carlos el 2026-09-30 al final de la
reapertura V32, cerrada con D-081, D-082 y D-083). Un challenger **FTRL-Proximal softmax
multiclase** entrenado en **una sola pasada online** sobre la TF-IDF ajustada por fold, frente
al control gemelo SGD de una pasada, comparado con el campeon V30 + plug-in (baseline interno
0.6159732) mediante el mecanismo de integracion establecido (combinador logistico anidado +
regla y pesos del plug-in CONGELADOS).

## Diagnostico que motiva el diseno (D-083)

La configuracion congelada en P3 (FTRL-Proximal **OvR**, alpha 0.05, lambda2 1.0, **10 pasadas**
con semantica proximal) degenero: log_loss medio de train 11.72 en la epoca 1 y 23.03 en la 10,
probabilidades planas y macro-F1 standalone 0.040169 (por debajo de la linea base trivial
0.048542). La IMPLEMENTACION se verifico exacta (tres implementaciones independientes coinciden
a 6.7e-15; punto fijo analitico 0.0), luego el fallo es de la configuracion: con alpha 0.05 el
termino de correccion `sigma` queda inflado ~10x frente al gradiente en coordenadas raras, y diez
pasadas acumulan `z, n` en un regimen para el que McMahan calibro alpha con T del orden 1e6-1e9
(aqui T = n_train_fold x 10). Ademas la descomposicion OvR deja el log-odds de la tasa base en
las coordenadas mas sensibles. De ahi las tres decisiones de este diseno: **softmax multiclase
(K=6 en un solo juego de pesos)**, **una sola pasada** y **alpha DERIVADO**.

## Condiciones incorporadas (kaggle-specialist, GO condicionado)

- `alpha = 3/(2*sqrt(T))` con T = n_train_fold: **derivado, cero valores evaluados**; se
  documentan la formula, su fuente, T y alpha por fold, y que no es un grid.
- La hipotesis "solo alpha tiene palanca" queda registrada como HIPOTESIS del arquitecto, no
  como evidencia.
- El control SGD cambia **dos** variables (alpha 1e-6 frente a 0.004, cuatro ordenes de
  magnitud), asi que la sonda de atribucion `|sigma*w|/|g|` pasa de diagnostico a
  **OBLIGATORIA**: si no se publica, el veredicto queda capado a MARGINAL.
- La puerta G1-G6 se aplica **igual a A1 y a A2**; si falla cualquiera, veredicto
  `NO_ATRIBUIBLE_BY_GATE` (nodo nuevo del arbol, no un gate de delta), **sin fallback y sin
  segundo alpha** (reintentar seria grid).
- Umbrales **DERIVADOS, no calibrados**: G1 anclado en `log(6) = 1.7918` (uniforme); G3 en
  referencias ya congeladas (trivial 0.0485421 y gemelo SGD de D-083 0.4934789); G4 en las
  proporciones reales de clase del train. El fold 0 queda registrado como
  **consumido-diagnostico**; la puerta corre antes de los folds 1-4 y antes de integrar, y su
  fallo NO toca `delta`.
- Se reusa la RECETA TF-IDF, no los artefactos: los sha256 de vocabulario e idf por fold se
  recalculan y se congelan aqui.
- Sondas: ademas de las de D-083, una **nueva obligatoria** de log_loss multiclase propio
  contra sklearn (1e-9) y de filas que suman 1 (D-083 solo probo el OvR, que no normaliza).
- Novedad registrada en artefactos NUEVOS escritos por el runner (`reports/v33_novelty_registry.csv`
  y una fila anexada en `reports/age_complete_experiment_history.csv`); **no se edita** ningun
  script ya ejecutado; el acta se anade a DECISIONS.md como D-084.
- Via B (ideas de otra magnitud) queda **DESCARTADA** en este expediente: B1 (grafo coseno kNN
  transductivo) es ILEGAL (usa features de test para modificar predicciones de train; su forma
  legal es un kNN re-ponderado = mecanismo M09 ya cerrado), B2 se descarta por analogia con C04
  (bagging, -0.000312) y B3 se refuta por algebra (tronco lineal = misma matriz de pesos).

## Invariantes

Sin test, sin holdout, sin leaderboard, sin submission, sin datos externos, sin transformers,
LLMs ni embeddings (0); campeon V30 + plug-in intacto; un experimento = un script nuevo.
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
PREREG_MD = cfg.REPORTS / "v33_preregistro.md"
PUERTA_MD = cfg.REPORTS / "v33_convergence_gate.md"
PREREG_JSON = cfg.EXPERIMENTS / "v33_00_preregistro.json"
ANCLA_D083 = cfg.EXPERIMENTS / "v32_05_ftrl_challenger.json"

BASELINE = 0.6159732
NESTED = 0.6154643
PESOS = [1.0, 1.1749, 1.2721, 1.163, 1.001, 0.8128]
N_CLASES = 6
SEMILLA = 20261010
SEMILLA_SEGUNDA_PARTICION = 987654
HILOS = 32
N_BOOTSTRAP = 2000
GATE_DELTA = 0.0015
GATE_FOLDS = 4
VARIABILIDAD = 0.0008876
UMBRAL_ALTA_CONFIANZA = 0.60
LIMITE_PROYECTADO_H = 20.0
LIMITE_GLOBAL_H = 24.0
RAM_TOPE_GB = 20.0
TRIVIAL_MAYORIA_ESPERADO = 0.04854212180687087
GEMELO_SGD_D083 = 0.4934789
LOG_UNIFORME_K6 = 1.791759469228055  # log(6)

VECTORIZADOR = {"analyzer": "word", "ngram_range": [1, 2], "min_df": 3, "max_df": 0.5,
                "sublinear_tf": True, "lowercase": True, "strip_accents": None,
                "token_pattern": r"(?u)\b\w+\b"}

A1_CONFIG = {
    "algoritmo": "FTRL-Proximal softmax multiclase (McMahan 2013), implementacion propia exacta",
    "clases": "K=6 en un solo juego de pesos (softmax); p suma 1 y es comparable con la mezcla",
    "alpha": {"formula": "alpha = 3/(2*sqrt(T))", "T": "n_train_fold del fold",
              "origen": "derivado del desplazamiento acumulado del intercepto: 2*alpha*sqrt(T) "
                        "debe ser del orden de 1.8x el log-odds de la tasa base |log(1/5)|",
              "valores_evaluados": 0,
              "rango_esperado": "0.00404 a 0.00409 (varia con T por construccion, no por busqueda)",
              "nota": "con alpha=0.05 (D-083) el desplazamiento seria 36.8, 36x mayor"},
    "lambda1": 0.0, "lambda2": 1.0, "beta": 1.0, "bias_x0": 1.0, "power_t": None,
    "pasadas": 1,
    "orden": "fold-train barajado UNA vez con np.random.default_rng(20261010+fold); orden "
             "congelado y reutilizado por el control",
    "dtype": "float64",
    "probabilidades": "softmax de los 6 logits, sin temperatura (L04/C03 cerradas)",
    "alternativa_no_usada": "vowpalwabbit 9.11.8 instalado y NO usado",
}
A2_CONTROL = {
    "estimator": "sklearn.linear_model.SGDClassifier", "loss": "log_loss",
    "learning_rate": "optimal", "alpha": 1e-6, "max_iter": 1, "tol": None,
    "fit_intercept": True, "average": False, "random_state": "20261010+fold", "n_jobs": HILOS,
    "pasadas": 1,
    "por_que": "misma representacion, mismo presupuesto (1 pasada), mismo barajado, misma "
               "semilla y misma evaluacion: lo unico libre es el eje proximal/adaptativo",
    "asimetria_declarada": "alpha 1e-6 frente a 0.004 (cuatro ordenes de magnitud) y laregularidad "
                           "del intercepto: por eso la sonda |sigma*w|/|g| es OBLIGATORIA",
}
PUERTA = {
    "momento": "dentro del fold 0, ANTES de los folds 1-4 y ANTES de integrar",
    "aplica_a": ["A1_ftrl_softmax", "A2_sgd_1p"],
    "fold_0": "consumido-diagnostico: G3 y G4 leen el fold-val del fold 0, que es parte de la "
              "validacion canonica; su fallo NO toca delta",
    "consecuencia_si_falla": "veredicto NO_ATRIBUIBLE_BY_GATE: sin fallback, sin segundo alpha, "
                             "sin integrar, sin bootstrap; se documenta y termina",
    "prohibido": "reintentar con otro alpha tras el fallo (seria grid)",
    "umbrales_derivados": {
        "G1_ll_train_banda": {"rango": [0.25, 0.60], "anclaje": "uniforme log(6) = 1.7918; la cota "
                                                             "alta caza la divergencia de D-083"},
        "G2_monotonia": {"puntos": [0.25, 0.50, 1.00], "tolerancia": 1e-6,
                         "anclaje": "caza directamente la firma 11.72 -> 23.03 de D-083"},
        "G3_macro_f1_fold_val": {"minimo": 0.30,
                                 "anclaje": "trivial 0.0485421 y gemelo SGD de D-083 0.4934789"},
        "G4_frecuencia_por_clase": {"rango": [0.10, 0.30],
                                    "anclaje": "proporciones reales de clase del train"},
        "G5_determinismo": {"criterio": "max|delta p| = 0.0 exacto entre dos corridas identicas"},
        "G6_suma_unidad": {"tolerancia": 1e-9, "criterio": "max|sum_c p_c - 1|"},
    },
}
SONDA_ATRIBUCION = {
    "descripcion": "razon media |sigma*w| / |g| por coordenada en el fold 0",
    "obligatoria": True,
    "consecuencia_si_no_se_publica": "el veredicto queda capado a MARGINAL",
    "motivo": "el control cambia dos variables (alpha y regularizacion del intercepto): la razon "
              "sigma/g es la que desambigua la causa (correccion inflada frente a multi-pasada)",
}
SONDAS = {
    "a_vectorizado_vs_lazo": {"tolerancia": 1e-12},
    "b_raiz_analitica_lambda0": {"residuo_max": 1e-9},
    "c_totalf1_macro": {"tolerancia": 1e-6},
    "d_sin_nan_logloss_finito": True,
    "e_logloss_multiclase_propio": {"tolerancia": 1e-9,
                                    "nota": "NUEVA en V33: D-083 solo probo el OvR, que no "
                                            "normaliza"},
    "f_filas_suman_uno": {"tolerancia": 1e-9},
    "politica": "si alguna falla -> exit 4 ANTES de entrenar; cualquier ajuste exige enmienda "
                "pre-ejecucion (precedente f0bf731)",
}
SOBRE_REFIT = {
    "descripcion": "reajustar el vectorizador en los 5 folds y comprobar que los sha256 de "
                   "vocabulario e idf coinciden con los de la corrida",
    "exigido_por": "kaggle-specialist",
    "tolerancia": "identicos bit a bit entre las dos corridas",
}


def sha256_bytes(datos_bytes: bytes) -> str:
    return hashlib.sha256(datos_bytes).hexdigest()


def sha256_archivo(ruta: Path) -> str:
    return sha256_bytes(ruta.read_bytes())


def hash_textos(textos: np.ndarray) -> str:
    """SHA256 determinista del contenido: longitud (8 bytes LE) + bytes UTF-8, en orden."""
    h = hashlib.sha256()
    for t in textos:
        b = t.encode("utf-8")
        h.update(len(b).to_bytes(8, "little"))
        h.update(b)
    return h.hexdigest()


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

    # Ancla de G3: linea base trivial de la clase mayoritaria, recalculada y contrastada.
    mayor, conteo = np.unique(clase, return_counts=True)
    idx_mayor = int(np.argmax(conteo))
    pred_trivial = np.full_like(clase, mayor[idx_mayor])
    trivial = float(f1_score(clase, pred_trivial, average="macro", labels=list(range(N_CLASES))))
    print(f"ancla G3 - macro-F1 trivial (clase {mayor[idx_mayor]}): {trivial:.16f} "
          f"(esperado {TRIVIAL_MAYORIA_ESPERADO})", flush=True)
    if abs(trivial - TRIVIAL_MAYORIA_ESPERADO) > 1e-12:
        print("[STOP] el ancla trivial no reproduce la de D-083", flush=True)
        return 1

    train = datos.load_train()
    ids_csv = train[cfg.ID_COL].to_numpy().astype(np.int64)
    etiquetas = train[cfg.TARGET].cat.codes.to_numpy().astype(np.int64)
    textos = train[cfg.TEXT_COL].fillna("").astype(str).to_numpy()
    if not np.array_equal(ids_canon.astype(np.int64), ids_csv):
        print("[STOP] el orden de ids del corpus no coincide con la particion canonica", flush=True)
        return 1
    if not np.array_equal(clase.astype(np.int64), etiquetas):
        print("[STOP] las etiquetas del corpus no coinciden con y_true canonico", flush=True)
        return 1
    sha_archivo = sha256_archivo(cfg.TRAIN_FILE)
    sha_textos = hash_textos(textos)
    sha_vectorizador = sha256_bytes(json.dumps(VECTORIZADOR, sort_keys=True).encode("utf-8"))
    print(f"corpus: archivo {sha_archivo[:16]} | textos {sha_textos[:16]} | "
          f"vectorizador {sha_vectorizador[:16]}", flush=True)

    hashes = {"corpus_archivo": sha_archivo, "corpus_textos": sha_textos,
              "vectorizador": sha_vectorizador,
              "ancla_d083_json": sha256_archivo(ANCLA_D083),
              "manifest_canonico": sha256_archivo(CANONICO / "manifest.json")}
    for nombre in ("y_true", "fold_idx", "v30_prob", "ids", "cand_pred", "cand_prob"):
        hashes[f"canonico_{nombre}"] = sha256_archivo(CANONICO / f"{nombre}.npy")

    import sklearn
    versiones = {"sklearn": sklearn.__version__, "numpy": np.__version__,
                 "pandas": pd.__version__}
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT.parent, check=False,
                            capture_output=True, text=True, timeout=30).stdout.strip()

    n_train_esperado = 135468  # filas de fold-train en la particion canonica (de D-083)
    alpha_esperado = 3.0 / (2.0 * float(np.sqrt(n_train_esperado)))

    congelado = {
        "fase": "V33-E1 pre-registro congelado", "fecha": FECHA, "commit": commit,
        "experimento": "challenger FTRL-Proximal softmax multiclase, una pasada online "
                       "(control: SGD log_loss de una pasada)",
        "baseline_primario": round(baseline, 7), "referencia_nested": NESTED,
        "pesos_plugin": PESOS, "hashes": hashes, "versiones": versiones,
        "anclas": {"macro_f1_trivial_clase_mayoritaria": trivial,
                   "clase_mayoritaria": str(mayor[idx_mayor]),
                   "gemelo_sgd_d083": GEMELO_SGD_D083,
                   "log_uniforme_k6": LOG_UNIFORME_K6,
                   "n_train_fold_esperado": n_train_esperado,
                   "alpha_esperado_para_n_train": alpha_esperado},
        "corpus": {
            "receta": "src.data.load_train() sobre data/raw/age_train.csv (sep=cfg.TRAIN_SEP, "
                      "encoding utf-8-sig, clean_text aplicado); texto = columna TEXT_COL; "
                      "orden de filas = orden del CSV = orden canonico de ids.npy",
            "hash_contenido": "sha256 con longitud (8 bytes LE) + bytes UTF-8 por texto, en orden",
            "prohibido": "el data_hash del manifiesto canonico (2ada573b...) queda invalidado en "
                         "D-083: es un hash de punteros de memoria, no de contenido",
        },
        "representacion": {
            "vectorizador": VECTORIZADOR,
            "norm_dtype": "defaults de sklearn (norm='l2', dtype float64)",
            "ajuste": "POR FOLD: fit solo en fold-train (idf, min_df y max_df nunca globales)",
            "reutilizacion": "se reusa la RECETA de v31_02_banco_representaciones.py, NO los "
                             "artefactos de D-083: los sha256 de vocabulario e idf por fold se "
                             "recalculan y se congelan en este ciclo",
        },
        "a1_ftrl_softmax": A1_CONFIG, "a2_sgd_control": A2_CONTROL,
        "puerta_convergencia": PUERTA, "sonda_atribucion": SONDA_ATRIBUCION, "sondas": SONDAS,
        "sobre_refit": SOBRE_REFIT,
        "determinismo": {
            "semilla": SEMILLA, "orden": "default_rng(20261010+fold) una vez por fold",
            "hilos": f"OMP/MKL/OPENBLAS_NUM_THREADS = {HILOS} antes de numpy; FTRL mono-hilo",
            "esperado": "exacto bit a bit",
            "sonda": "reentrenar el fold 0 de A1 y de A2 y comparar bit a bit",
            "tolerancias": "0 exacto; <=1e-9 conforme; >1e-9 con rango publicado y "
                           f"delta_gate > rango; rango >= {VARIABILIDAD} o solape con el gate "
                           "=> no concluyente",
        },
        "integracion": {
            "mecanismo": "combinador logistico anidado sobre [L2(v30_prob) || L2(p_brazo)] "
                         "(12 features, L2 por fila con clip 1e-12); LogReg(C=1.0, max_iter=2000, "
                         "lbfgs) por fold (fit fold!=k, predice fold==k)",
            "decision": "argmax(p_comb * w_plugin CONGELADO)",
            "diagnosticos": ["argmax_sin_regla", "delta vs nested 0.6154643", "log_loss por brazo",
                             "macro-F1 diagonal 6x6", "rescate/dano/neto", "alta confianza"],
        },
        "gates": {
            "baseline_abort": 5e-7, "delta_min": GATE_DELTA, "folds_min": GATE_FOLDS,
            "control": "delta_int(A1) > delta_int(A2)",
            "bootstrap": {"n": N_BOOTSTRAP, "semilla": SEMILLA, "ic95_sin_cero": True,
                          "diagnostico_adicional": "IC95 pareado de la diferencia (A1-A2)"},
            "sin_leakage": True, "determinismo_concluyente": True,
            "sonda_atribucion_publicada": True,
            "caps": ["sin la sonda |sigma*w|/|g| publicada -> veredicto capado a MARGINAL",
                     "determinismo no concluyente -> el veredicto no puede ser PROMISING"],
        },
        "arbol": {
            "NO_ATRIBUIBLE_BY_GATE": "la puerta G1-G6 falla en A1 o A2: STOP documentado, sin "
                                     "fallback, sin integrar (nodo nuevo, no un gate de delta)",
            "PROMISING": "gates completos -> segunda particion SOLO con autorizacion del usuario "
                         f"(semilla {SEMILLA_SEGUNDA_PARTICION}); el script no la ejecuta",
            "MARGINAL": "delta > 0 sin gates completos, o con cap activo (sonda o determinismo)",
            "CLOSED": "delta <= 0 -> se cierra V33; el campeon permanece",
            "submission": "jamas automatica; requeriria gate completo + segunda particion + "
                          "autorizacion explicita",
        },
        "novedad": {
            "que_es": "FTRL-Proximal softmax multiclase propio, semantica online de una pasada, "
                      "un solo juego de pesos, sobre la TF-IDF por fold",
            "no_es": "no es un embedding, no es un vector aprendido, no es una representacion "
                     "nueva y no usa datos externos",
            "diferencia_con_d083": "D-083 fue OvR con 10 pasadas proximales y alpha 0.05; aqui "
                                   "softmax, 1 pasada y alpha derivado",
            "registro": "artefactos NUEVOS escritos por el runner: "
                        "reports/v33_novelty_registry.csv (1 fila) y una fila anexada en "
                        "reports/age_complete_experiment_history.csv (mismo esquema); acta D-084 "
                        "por ANEXADO en DECISIONS.md; no se edita "
                        "scripts/age_cobertura_metodologica.py ni ningun script ejecutado",
        },
        "via_b_descartada": {
            "B1_grafo_transductivo": "ILEGAL (usa features de test sin etiqueta para modificar "
                                     "predicciones de train; su forma legal es un kNN "
                                     "re-ponderado = mecanismo M09 ya cerrado)",
            "B2_aumentacion_masking": "descartada por analogia con C04 (bagging, -0.000312)",
            "B3_cabeza_multitarea": "refutada por algebra: con tronco lineal es una "
                                    "reparametrizacion de la misma matriz de pesos",
        },
        "compliance": {
            "embeddings": 0, "transformers": 0, "llm": 0, "datos_externos": 0, "test": 0,
            "holdout": 0, "leaderboard": 0, "submission": 0,
            "ast_audit": "importes Y literales de ruta: vowpal, vw, torch, transformers, "
                         "gensim, fasttext, sentence-transformers, word2vec, glove, "
                         "spacy-vectors; mas 0 rutas de test/holdout/submission/leaderboard",
            "auditorias_exigidas": ["6 .npy canonicos + manifest", ("sha256 del archivo y del "
                                    "array de textos (nunca el data_hash del manifiesto)"),
                                    "sha256 del vectorizador", "vocabulario e idf por fold",
                                    "sonda de refit en los 5 folds", "sondas (a)-(f)",
                                    "sonda de atribucion |sigma*w|/|g|", ("sondas de determinismo "
                                    "A1 y A2"), "RSS pico con ru_maxrss", "escritura atomica"],
        },
        "ejecucion": {
            "coste_estimado": "~13-16 min CPU (1 pasada por fold y por brazo; TF-IDF ~90 s por "
                              "fold compartido; determinismo: 2 reentrenamientos del fold 0)",
            "ram_tope_gb": RAM_TOPE_GB, "ram_medicion": "resource.getrusage(ru_maxrss)",
            "calibracion": f"tras el fold 0: proyeccion = 9 x t_fold0; si supera "
                           f"{LIMITE_PROYECTADO_H} h -> parar y reportar",
            "timeout_global": f"{LIMITE_GLOBAL_H} h",
            "checkpoints": "experiments/v33_01_estado.json (resumible por fold)",
            "escritura_atomica": "tmp + os.replace en artifacts/p33_scores/v33_e1/",
        },
        "entregables": {
            "preregistro": "scripts/v33_00_preregistro.py -> experiments/v33_00_preregistro.json, "
                           "reports/v33_preregistro.md, reports/v33_convergence_gate.md",
            "runner": "scripts/v33_01_ftrl_softmax_online.py -> "
                      "experiments/v33_01_ftrl_softmax_online.json, experiments/v33_01_estado.json, "
                      "artifacts/p33_scores/v33_e1/{a1_ftrl_softmax,a2_sgd_1p}.npy, "
                      "reports/v33_e1_ftrl_softmax_online.md, reports/figures/v33_e1_curvas.png, "
                      "reports/v33_novelty_registry.csv, fila anexada en "
                      "reports/age_complete_experiment_history.csv, RUNLOG.md y fila V33-E1 en "
                      "experiments.csv",
        },
    }

    lineas = [
        "# V33 - E1 - Pre-registro congelado: FTRL-Proximal softmax, una pasada online",
        "",
        f"**Fecha**: {FECHA}. **Estado**: CONGELADO antes de ejecutar (commit `{commit[:7]}`).",
        f"**Baseline primario recomputado**: {baseline:.7f} (nested {NESTED}).",
        (f"**Ancla G3 - macro-F1 trivial** (clase {mayor[idx_mayor]}): {trivial:.16f}; "
        f"gemelo SGD de D-083: {GEMELO_SGD_D083}; uniforme K=6: log(6) = {LOG_UNIFORME_K6:.6f}."),
        "",
        "## La pregunta y el diagnostico que la motiva",
        "",
        "> El aprendizaje online proximal con la semantica correcta (una pasada, softmax",
        "> multiclase, alpha derivado) aporta informacion complementaria al campeon?",
        "",
        ("D-083: la implementacion FTRL de P3 era exacta pero la configuracion (OvR, alpha 0.05, "
        "10 pasadas proximales) DEGENERO (log_loss de train 11.72 -> 23.03; standalone 0.040169 "
        "bajo el trivial 0.048542). Causa: `sigma` inflado ~10x por alpha=0.05 y diez pasadas en "
        "un regimen donde McMahan calibro alpha con T ~ 1e6-1e9."),
        "",
        "## Diseno congelado",
        "",
        ("- **A1 (decisional)**: FTRL-Proximal **softmax multiclase** (K=6, un solo juego de "
        "pesos; p suma 1), **una sola pasada online**, orden = fold-train barajado una vez con "
        "`default_rng(20261010+fold)` y reutilizado por el control."),
        (f"- `alpha = 3/(2*sqrt(T))` con T = n_train_fold: para T = {n_train_esperado} sale "
         f"alpha ~ {alpha_esperado:.6f} (rango 0.00404-0.00409). **Derivado, cero valores "
         "evaluados**: no es un grid. Con alpha=0.05 el desplazamiento del intercepto seria "
         "36.8, 36x mayor."),
        ("- lambda1 0.0, lambda2 1.0, beta 1.0, intercepto x0=1.0, sin power_t, float64, softmax "
        "sin temperatura."),
        ("- **A2 (control)**: `SGDClassifier(loss='log_loss', learning_rate='optimal', "
        "alpha=1e-6, max_iter=1, tol=None)` con la MISMA matriz, el mismo barajado y la misma "
        "1 pasada. Asimetria declarada (alpha 1e-6 frente a 0.004) y por eso la sonda "
        "`|sigma*w|/|g|` es **obligatoria** (si no se publica, veredicto capado a MARGINAL)."),
        ("- La hipotesis 'solo alpha tiene palanca' queda como HIPOTESIS del arquitecto, no como "
        "evidencia."),
        "",
        "## Puerta de convergencia fail-fast (G1-G6)",
        "",
        ("- Corre dentro del fold 0, **antes** de los folds 1-4 y **antes** de integrar; se aplica "
        "**igual a A1 y a A2**."),
        "- G1 banda de log_loss multiclase de train [0.25, 0.60] (anclaje: uniforme log 6).",
        "- G2 log_loss en 25/50/100 % de la pasada no creciente (tolerancia 1e-6).",
        "- G3 macro-F1 standalone en el fold-val >= 0.30 (anclas: 0.0485421 y 0.4934789).",
        "- G4 frecuencia predicha por clase en [0.10, 0.30] (proporciones reales del train).",
        "- G5 determinismo: max|delta p| = 0.0 entre dos corridas identicas.",
        "- G6 filas que suman 1: max|sum_c p_c - 1| <= 1e-9.",
        ("- Si falla cualquiera: veredicto **NO_ATRIBUIBLE_BY_GATE** (nodo nuevo del arbol), "
        "**sin fallback y sin segundo alpha**; el fallo NO toca delta. El fold 0 queda "
        "registrado como consumido-diagnostico."),
        "",
        "## Sondas obligatorias antes de entrenar (exit 4)",
        "",
        ("1. Vectorizado == lazo ingenuo (1e-12). 2. Raiz analitica con lambda 0 (1e-9). "
        "3. TotalF1:average=Macro == macro-F1 (1e-6). 4. Sin NaN/inf. 5. **NUEVA**: log_loss "
        "multiclase propio == sklearn (1e-9) y filas que suman 1 (D-083 solo probo el OvR)."),
        ("6. Sonda de refit: los sha256 de vocabulario e idf de los 5 folds deben reproducirse "
        "bit a bit."),
        "",
        "## Gates de la serie (heredados sin cambios)",
        "",
        (f"baseline +-5e-7; delta_int(A1) >= +{GATE_DELTA:.4f}; folds+ >= {GATE_FOLDS}/5; "
         f"control delta(A1) > delta(A2); IC95 pareado {N_BOOTSTRAP} (semilla {SEMILLA}) sin "
         "cero; sin leakage; determinismo concluyente; sonda de atribucion publicada."),
        ("Caps: sin sonda de atribucion o con determinismo no concluyente el veredicto no puede "
        "ser PROMISING. Arbol: NO_ATRIBUIBLE_BY_GATE / PROMISING (segunda particion solo con "
        "autorizacion) / MARGINAL / CLOSED."),
        "",
        "## Materiales congelados (SHA256)",
        "",
        "| Artefacto | SHA256 (16) |",
        "|---|---|",
    ]
    for nombre, valor in hashes.items():
        lineas.append(f"| {nombre} | `{valor[:16]}...` |")
    lineas += [
        "",
        "## Novedad y lo que NO se hace",
        "",
        ("- Novedad: FTRL-Proximal **softmax multiclase propio**, semantica online de una pasada, "
        "un solo juego de pesos, sobre la TF-IDF por fold. No es embedding, no es vector "
        "aprendido, no es representacion nueva, no usa datos externos."),
        ("- Se registra en artefactos NUEVOS escritos por el runner (`reports/v33_novelty_registry.csv`, "
        "fila anexada en `reports/age_complete_experiment_history.csv`) y en el acta D-084; **no "
        "se edita ningun script ya ejecutado**."),
        ("- **Via B DESCARTADA**: B1 (grafo coseno kNN transductivo) es ILEGAL (usa features de "
        "test sin etiqueta para modificar predicciones de train; su forma legal es kNN = M09 "
        "cerrado); B2 (masking/spans) descartada por analogia con C04; B3 (cabeza multitarea) "
        "refutada por algebra con tronco lineal."),
        "",
        "## Compliance y limites",
        "",
        ("- 0 embeddings, 0 transformers, 0 LLM, 0 datos externos; sin test, sin holdout, sin "
        "leaderboard, sin submission; un script nuevo; campeon V30 + plug-in (LB 0.63575) intacto."),
        ("- AST audit de imports Y literales (vowpal, vw, torch, transformers, gensim, fasttext, "
        "sentence-transformers, word2vec, glove, spacy-vectors) y 0 rutas de test/holdout/"
        "submission/leaderboard."),
        (f"- Coste esperado ~13-16 min de CPU; RAM con tope {RAM_TOPE_GB:g} GB medido con "
        "`ru_maxrss`; calibracion de fold 0; checkpoints por fold; escritura atomica."),
        f"- Referencia de variabilidad de entrenamiento: {VARIABILIDAD}.",
    ]
    texto = "\n".join(lin.rstrip() for lin in lineas) + "\n"
    PREREG_MD.write_text(texto, encoding="utf-8")
    print(f"  {PREREG_MD.name}: {len(texto.splitlines())} lineas", flush=True)

    puerta_lineas = [
        "# V33 - E1 - Puerta de convergencia fail-fast (G1-G6)",
        "",
        "Documento normativo de la puerta. Se congela con el pre-registro; sus umbrales estan",
        "**DERIVADOS** (no calibrados con datos) y las sondas en `/tmp` solo sirven para",
        "programar la regla: si una sonda fijara un umbral, seria tuning y esta prohibido.",
        "",
        "| Gate | Criterio | Anclaje | Tolerancia |",
        "|---|---|---|---|",
        (f"| G1 | 0.25 <= log_loss multiclase de train <= 0.60 | uniforme `log(6)` = "
         f"{LOG_UNIFORME_K6:.6f}; la cota alta caza la divergencia 11.72 -> 23.03 de D-083 | 1e-9 |"),
        "| G2 | log_loss no creciente en 25 %, 50 % y 100 % de la pasada | caza la firma de D-083 | 1e-6 |",
        (f"| G3 | macro-F1 standalone en el fold-val >= 0.30 | trivial {TRIVIAL_MAYORIA_ESPERADO:.7f} "
         f"(clase {mayor[idx_mayor]}) y gemelo SGD de D-083 {GEMELO_SGD_D083} | 1e-9 |"),
        "| G4 | frecuencia predicha por clase en [0.10, 0.30] | proporciones reales del train | 1e-9 |",
        "| G5 | max|delta p| = 0.0 entre dos corridas identicas | determinismo CPU esperado exacto | 0 |",
        "| G6 | max|sum_c p_c - 1| <= 1e-9 | softmax de 6 logits | 1e-9 |",
        "",
        "## Cuando corre y que pasa si falla",
        "",
        "- Corre dentro del **fold 0**, antes de los folds 1-4 y antes de integrar.",
        "- Se aplica **igual a A1 y a A2**: si el control falla, tambien se detiene (asi el",
        "  diagnostico distingue 'el presupuesto no alcanza' de 'el optimizador no converge').",
        "- G3 y G4 leen el fold-val del fold 0, que es parte de la validacion canonica: el",
        "  fold 0 queda registrado como **consumido-diagnostico** y su fallo **no toca delta**.",
        "- Si falla cualquiera: veredicto `NO_ATRIBUIBLE_BY_GATE`, **sin fallback**, **sin",
        "  segundo alpha** (reintentar seria grid), sin integracion y sin bootstrap.",
        "",
        "## Sonda de atribucion obligatoria",
        "",
        (f"{SONDA_ATRIBUCION['descripcion']}. {SONDA_ATRIBUCION['motivo']}. "
         f"Consecuencia si no se publica: {SONDA_ATRIBUCION['consecuencia_si_no_se_publica']}."),
    ]
    puerta_texto = "\n".join(lin.rstrip() for lin in puerta_lineas) + "\n"
    PUERTA_MD.write_text(puerta_texto, encoding="utf-8")
    print(f"  {PUERTA_MD.name}: {len(puerta_texto.splitlines())} lineas", flush=True)

    PREREG_JSON.write_text(json.dumps(congelado, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  {PREREG_JSON.name}", flush=True)

    runlog = ROOT / "RUNLOG.md"
    marca = "Pre-registro de V33-E1"
    if marca not in runlog.read_text(encoding="utf-8"):
        bloque = ("\n---\n\n## Pre-registro de V33-E1 (FTRL softmax, una pasada) · " + FECHA + "\n\n"
                  "- Congelado: FTRL-Proximal softmax multiclase (K=6, un juego de pesos), 1 pasada "
                  "online, alpha = 3/(2*sqrt(T))\n  derivado (cero valores evaluados), lambda1 0, "
                  "lambda2 1, beta 1, intercepto x0=1, sin power_t; control SGD log_loss de 1 "
                  "pasada con el\n  mismo barajado; puerta de convergencia G1-G6 fail-fast en el "
                  "fold 0, aplicada a ambos brazos;\n  sonda |sigma*w|/|g| obligatoria; sondas "
                  "(a)-(f) y sonda de refit; gates de la serie sin cambios.\n"
                  f"- Baseline recomputado: {baseline:.7f}; ancla trivial: {trivial:.16f}. "
                  "Via B descartada (B1 ilegal; B2 analogia; B3 algebra).\n"
                  "- Sin test, sin holdout, sin leaderboard, sin submission.\n")
        with runlog.open("a", encoding="utf-8") as f:
            f.write(bloque)
        print("  RUNLOG.md: pre-registro anotado", flush=True)

    ruta_exp = ROOT / "experiments.csv"
    lineas_exp = ruta_exp.read_text(encoding="utf-8").splitlines()
    cols = [c.strip() for c in lineas_exp[0].split(",")]
    if "V33-E1-PREREG" not in {lin.split(",", 1)[0] for lin in lineas_exp[1:] if lin}:
        reg = {"experiment_id": "V33-E1-PREREG", "fecha": FECHA, "bloque": "33.0",
               "etapa": "V33-E1 pre-registro congelado del challenger FTRL-Proximal softmax "
                       "multiclase de una pasada",
               "pregunta": "el online proximal con la semantica correcta aporta complementariedad "
                           "al campeon?",
               "hipotesis": "la divergencia de D-083 fue de configuracion (sigma inflado por "
                            "alpha y multi-pasada), no de implementacion",
               "features": "TF-IDF por fold (receta v31_02) recalculada; 0 embeddings",
               "preprocessing": "vocabulario e idf ajustados SOLO en fold-train; sha256 por fold",
               "representacion": "TF-IDF dispersa (misma receta, artefactos nuevos)",
               "modelo": "FTRL-Proximal softmax K=6 propio, 1 pasada (A1) / SGDClassifier "
                         "log_loss 1 pasada (A2)",
               "hiperparametros": "alpha = 3/(2*sqrt(T)) DERIVADO (0 valores evaluados), lambda1 0, "
                                  "lambda2 1, beta 1, x0 1, sin power_t, 1 pasada, seed "
                                  "20261010+fold",
               "validacion": "particion canonica; puerta G1-G6 fail-fast en fold 0; integracion "
                             "P6/P7; gates de la serie; sonda de atribucion obligatoria",
               "macro_f1_mean": "", "macro_f1_std": "", "weighted_f1_mean": "",
               "micro_f1_mean": "", "accuracy_mean": "",
               "resultado": "pre-registro escrito y congelado; sin resultados aun",
               "interpretacion": "diagnostico de D-083 y condiciones del especialista incorporados",
               "decision": "ejecutar V33-E1 tal cual",
               "duracion_seg": "0.0",
               "artefactos": "reports/v33_preregistro.md | reports/v33_convergence_gate.md | "
                             "experiments/v33_00_preregistro.json"}
        buf = io.StringIO()
        csv.writer(buf, lineterminator="\n").writerow([reg[c] for c in cols])
        with ruta_exp.open("a", encoding="utf-8") as f:
            f.write(buf.getvalue())
        print("  experiments.csv: fila V33-E1-PREREG", flush=True)

    print(f"\n[fin] {time.time() - inicio:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
