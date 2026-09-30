"""V32 - E3 (P3) - Pre-registro CONGELADO: challenger FTRL-Proximal sobre TF-IDF por fold.

## Que congela

El tercer y ultimo experimento de la reapertura V32: un challenger **FTRL-Proximal** (McMahan
et al. 2013, implementacion propia) entrenado sobre una representacion **TF-IDF ajustada por
fold**, comparado contra el campeon V30 + plug-in (baseline interno 0.6159732) con el
mecanismo de integracion establecido (combinador logistico anidado + regla y pesos del plug-in
CONGELADOS).

## Decisiones incorporadas (agentes + usuario)

- ml-architect: FTRL no implementado (novedad verificada); no existe matriz TF-IDF canonica
  -> se construye con la config IDENTICA a `VECTORIZADOR` de `v31_02_banco_representaciones.py`
  y **vocabulario ajustado por fold** (precedente canonico del manifest); config unica sin
  grid; control de atribucion = gemelo SGD con el mismo presupuesto de pasadas; A3 barajado.
  Costes: ~25-30 min CPU; RAM con tope declarado; calibracion de fold 0.
- kaggle-specialist: APROBADO CON EXIGENCIAS: reproducir la identidad del corpus antes de
  entrenar; sondas (a)-(d) obligatorias antes de entrenar; AST audit que pruebe ausencia de
  vowpal/vw/torch/transformers/gensim/fasttext; registrar que `vowpalwabbit 9.11.8` esta
  instalado y NO usado; RSS pico con `ru_maxrss`; escritura atomica; cap por IC95 de la
  diferencia; criterio de fallback escrito y congelado ANTES de ejecutar.
- Usuario: autorizo P3 (ultimo candidato de la lista priorizada) el 2026-09-30.

## Hallazgo de auditoria registrado (no bloquea la ejecucion)

El `data_hash` del manifiesto canonico (`2ada573b...`) se calculo con
`train["text"].fillna("").astype(str).to_numpy().tobytes()` sobre un array de **objetos**:
`.tobytes()` serializa PUNTEROS de memoria, no el texto (mismo valor dentro de un proceso,
distinto entre procesos: se observaron `68b9a0f2...` y `40342f7c...`). No es reproducible y no
identifica el contenido. Este pre-registro congela en su lugar: sha256 del ARCHIVO del corpus
(bytes), sha256 del ARRAY DE TEXTOS con receta determinista de longitudes + bytes en orden, y
la receta misma. No se reordena nada para "hacer cuadrar" el hash.
"""

from __future__ import annotations

import hashlib
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
PREREG_MD = cfg.REPORTS / "v32_p3_preregistro.md"
PREREG_JSON = cfg.EXPERIMENTS / "v32_04_preregistro_ftrl.json"

BASELINE = 0.6159732
NESTED = 0.6154643
PESOS = [1.0, 1.1749, 1.2721, 1.163, 1.001, 0.8128]
N_CLASES = 6
SEMILLA = 20261010
SEMILLA_BARAJADO = 20261011
HILOS = 32
N_BOOTSTRAP = 2000
GATE_DELTA = 0.0015
GATE_FOLDS = 4
VARIABILIDAD = 0.0008876
UMBRAL_ALTA_CONFIANZA = 0.60
LIMITE_PROYECTADO_H = 20.0
LIMITE_GLOBAL_H = 24.0
RAM_TOPE_GB = 20.0
SEMILLA_SEGUNDA_PARTICION = 987654

#: Configuracion del vectorizador: IDENTICA a `VECTORIZADOR` de v31_02_banco_representaciones.py
#: (norm y dtype quedan en los defaults de sklearn: norm="l2", dtype float64).
VECTORIZADOR = {"analyzer": "word", "ngram_range": [1, 2], "min_df": 3, "max_df": 0.5,
                "sublinear_tf": True, "lowercase": True, "strip_accents": None,
                "token_pattern": r"(?u)\b\w+\b"}

FTRL_CONFIG = {
    "algoritmo": "FTRL-Proximal (McMahan et al. 2013), implementacion propia exacta",
    "alpha": 0.05, "lambda1": 0.0, "lambda2": 1.0, "beta": 1.0, "bias": True, "x0": 1.0,
    "power_t": None, "epocas": 10, "criterio_parada": "ninguno (10 pasadas fijas)",
    "ovr": "6 modelos binarios con sigmoid; p no suma 1",
    "barajado": "np.random.default_rng(20261010+fold) sobre los indices de fold-train por epoca",
    "combinador_p": "normalizacion L2 por fila (comparador canonico); se reportan ademas "
                    "argmax crudo, macro-F1 diagonal y log_loss",
    "actualizacion": "sigma=(sqrt(n+g^2)-sqrt(n))/(alpha*beta); z+=g-sigma*w; n+=g^2; "
                     "w=0 si |z|<=lambda1; si no sign(z)*(|z|-lambda1)/((beta+sqrt(n))/alpha+lambda2)",
    "alternativa_no_usada": "vowpalwabbit 9.11.8 instalado en el venv y NO usado "
                            "(parametrizacion y bit_precision distintas; sin sondesn de verificacion)",
}
SGD_TWIN = {
    "estimator": "sklearn.linear_model.SGDClassifier", "loss": "log_loss", "alpha": 1e-6,
    "max_iter": 10, "tol": None, "fit_intercept": True, "average": False,
    "learning_rate": "optimal", "random_state": "20261010+fold", "n_jobs": HILOS,
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
SONDAS = {
    "a_vectorizado_vs_lazo": {"problema": "aleatorio 200x50x6", "tolerancia": 1e-12},
    "b_punto_fijo_analitico": {"config": "lambda1=lambda2=0 con feature constante",
                               "residuo_max": 1e-9},
    "c_totalf1_macro": {"comparacion": "TotalF1:average=Macro == macro-F1 de sklearn",
                        "tolerancia": 1e-6},
    "d_sin_nan_logloss_finito": True,
    "politica": "si alguna sonda falla -> exit 4 ANTES de entrenar; cualquier ajuste exige "
                "enmienda pre-ejecucion (precedente f0bf731)",
}
FALLBACK = {
    "criterio": "en el fold 0 de A1, sobre las filas de fold-train: log_loss OvR medio tras 10 "
                "pasadas <= 0.8 * log_loss tras la pasada 1, y sin NaN/inf",
    "si_falla": "usar la alternativa declarada alpha=0.005 (lambda2=1.0 resto igual) y registrar "
                "config_fallback_usada=True en el JSON de resultados y en el informe",
    "prohibido": "decidir con fold-val o con el leaderboard; si el criterio se fijara despues "
                 "de observar fold-0, el resultado se marca post-hoc en D-083",
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

    manifest = json.loads((CANONICO / "manifest.json").read_text(encoding="utf-8"))
    hashes = {
        "canonico_y_true": sha256_archivo(CANONICO / "y_true.npy"),
        "canonico_fold_idx": sha256_archivo(CANONICO / "fold_idx.npy"),
        "canonico_v30_prob": sha256_archivo(CANONICO / "v30_prob.npy"),
        "canonico_ids": sha256_archivo(CANONICO / "ids.npy"),
        "manifest_canonico": sha256_archivo(CANONICO / "manifest.json"),
        "corpus_archivo": sha_archivo,
        "corpus_textos": sha_textos,
        "vectorizador": sha_vectorizador,
    }
    import sklearn
    versiones = {"sklearn": sklearn.__version__, "numpy": np.__version__,
                 "pandas": pd.__version__}

    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT.parent, check=False,
                            capture_output=True, text=True, timeout=30).stdout.strip()

    congelado = {
        "fase": "V32-E3 (P3) pre-registro congelado", "fecha": FECHA, "commit": commit,
        "experimento": "challenger FTRL-Proximal sobre TF-IDF ajustada por fold (ciclo 3)",
        "baseline_primario": round(baseline, 7), "referencia_nested": NESTED,
        "pesos_plugin": PESOS, "hashes": hashes, "versiones": versiones,
        "corpus": {
            "receta": "src.data.load_train() sobre data/raw/age_train.csv (sep=cfg.TRAIN_SEP, "
                      "encoding utf-8-sig, clean_text aplicado); texto = columna TEXT_COL; "
                      "orden de filas = orden del CSV = orden canonico de ids.npy",
            "ids_alineados": True, "etiquetas_alineadas": True,
            "hash_contenido": "sha256 con longitud (8 bytes LE) + bytes UTF-8 por texto, en orden",
            "desvio_manifest": {
                "data_hash_manifest": manifest.get("data_hash"),
                "problema": "el manifiesto lo calculo con to_numpy().tobytes() sobre un array de "
                            "OBJETOS: serializa punteros de memoria, no el texto; no es "
                            "reproducible entre procesos (se observaron 68b9a0f2 y 40342f7c)",
                "decision": "no se reordena nada; se congelan sha256 del archivo + sha256 del "
                            "array de textos con receta determinista + la receta",
            },
        },
        "representacion": {
            "vectorizador": VECTORIZADOR,
            "norm_dtype": "defaults de sklearn (norm='l2', dtype float64)",
            "ajuste": "POR FOLD: fit solo en fold-train (idf, min_df y max_df nunca globales)",
            "precedente": "scripts/v31_02_banco_representaciones.py (VECTORIZADOR) y el propio "
                          "manifest canonico ('se ajusta el vectorizador solo sobre train de "
                          "ese fold')",
            "verificabilidad": "sha256 de config + sha256 del vocabulario ordenado y del idf "
                               "por fold + sonda de refit que reproduce los mismos hashes",
        },
        "ftrl": FTRL_CONFIG, "a2_sgd_twin": SGD_TWIN, "a3_control_negativo": A3_CONTROL,
        "sondas": SONDAS, "fallback": FALLBACK,
        "determinismo": {
            "seeds": {"global": SEMILLA, "barajado_etiquetas": SEMILLA_BARAJADO,
                      "orden_datos": "default_rng(20261010+fold) por epoca"},
            "hilos": f"OMP/MKL/OPENBLAS_NUM_THREADS = {HILOS} fijos antes de importar numpy; el "
                     "lazo FTRL es mono-hilo",
            "esperado": "exacto bit a bit (float64, orden CSR, permutacion sembrada, sin "
                        "reducciones multi-hilo ni atomicos ni BLAS denso)",
            "sonda": "reentrenar fold 0 de A1 y de A2 con config identica y comparar bit a bit",
            "tolerancias": "0 exacto; <=1e-9 conforme; >1e-9 declarar rango y exigir "
                           f"delta_gate > rango; rango >= {VARIABILIDAD} o solape con el gate "
                           "=> no concluyente",
        },
        "integracion": {
            "mecanismo": "combinador logistico anidado sobre [L2(v30_prob) || L2(p_brazo)] "
                         "(12 features, L2 por fila con clip 1e-12); LogReg(C=1.0, "
                         "max_iter=2000, lbfgs) por fold (fit fold!=k, predice fold==k)",
            "decision": "argmax(p_comb * w_plugin CONGELADO)",
            "diagnosticos": ["argmax_sin_regla", "delta vs nested 0.6154643",
                             "log_loss por brazo", "macro-F1 diagonal 6x6"],
        },
        "gates": {
            "baseline_abort": 5e-7, "delta_min": GATE_DELTA, "folds_min": GATE_FOLDS,
            "control": "delta_int(A1) > delta_int(A2) Y bootstrap pareado de la diferencia "
                       f"(F1_combA1 - F1_combA2; {N_BOOTSTRAP}; semilla {SEMILLA}) con IC95 sin cero",
            "control_negativo": "si delta_int(A3) >= delta_int(A1): NO_ATRIBUIBLE (precedencia)",
            "a3_gate": A3_CONTROL["umbral_fuga"],
            "bootstrap": {"n": N_BOOTSTRAP, "semilla": SEMILLA, "ic95_sin_cero": True},
            "sin_leakage": True, "determinismo_concluyente": True,
            "cap_ic95_diferencia": "si el IC95 de la diferencia contiene 0, el veredicto queda "
                                   "capado a MARGINAL aunque A1 pase el resto de gates "
                                   "(decision del lead, registrada; puede degradar un resultado "
                                   "genuinamente bueno)",
            "cap_determinismo": "sonda no concluyente -> el veredicto no puede ser PROMISING",
        },
        "arbol": {
            "NO_ATRIBUIBLE": "si delta_int(A3) >= delta_int(A1) (precedencia)",
            "PROMISING": f"gates completos y IC95 de la diferencia excluye 0 -> segunda particion "
                         f"congelada (semilla {SEMILLA_SEGUNDA_PARTICION}) en este mismo ciclo",
            "MARGINAL": "delta > 0 sin gates completos (incluye el cap por IC95) -> no se adopta",
            "CLOSED": "delta <= 0 -> se cierra P3 y la reapertura V32 termina",
            "nota": "la formulacion del arquitecto 'A1 <= A2 -> NO_ATRIBUIBLE' queda subsumida "
                    "por este arbol, que es el validado por QA en el ciclo 2 (P2)",
            "submission": "jamas automatica; requeriria gate completo + segunda particion + "
                          "autorizacion explicita",
        },
        "ejecucion": {
            "coste_estimado": "~25-30 min CPU (fit TF-IDF ~90 s por fold compartido por los 3 "
                              "brazos; A1 ~60 s; A2 ~20 s; A3 ~60 s)",
            "ram_tope_gb": RAM_TOPE_GB,
            "ram_medicion": "resource.getrusage(ru_maxrss); si se supera -> STOP registrado; NO "
                            "anadir max_features sobre la marcha",
            "calibracion": f"tras el fold 0 de A1: proyeccion = 9 x t_fold0; si supera "
                           f"{LIMITE_PROYECTADO_H} h -> parar y reportar",
            "timeout_global": f"{LIMITE_GLOBAL_H} h",
            "checkpoints": "experiments/v32_05_estado.json (resumible por fold)",
            "escritura_atomica": "tmp + os.replace en artifacts/p33_scores/v32_p3/",
        },
        "novedad": {
            "ftrl_en_codigo": False,
            "verificacion": "search_graph del proyecto home-carlos-Git-PLN: 0 nodos; search_code "
                            "y grep solo literales con el nombre del candidato",
            "cobertura_registrada_en": "DECISIONS D-083, PROJECT_STATE.md e informe del "
                                      "experimento; NO se edita scripts/age_cobertura_metodologica.py",
        },
        "compliance": {
            "embeddings": 0, "transformers": 0, "llm": 0, "datos_externos": 0, "test": 0,
            "holdout": 0, "leaderboard": 0, "submission": 0,
            "nota": "TF-IDF + modelo lineal ya usados por el proyecto (LogReg/LinearSVC/SGD/NB-SVM): "
                    "no cuenta como embedding vetado; vowpalwabbit 9.11.8 instalado y NO usado",
            "ast_audit": "auditoria de literales de ruta que pruebe ausencia de vowpal/vw/torch/"
                         "transformers/gensim/fasttext y 0 rutas de test/holdout/submission",
            "auditorias_exigidas": [
                "hashes del manifest",
                "sha256 del archivo y del array de textos",
                "sha256 del vectorizador",
                "vocabulario e idf por fold",
                "sondas (a)-(d)",
                "sonda de reentreno",
                "RSS pico",
                "config_fallback_usada",
                "escritura atomica",
            ],
        },
        "entregables": {
            "preregistro": "scripts/v32_04_preregistro_ftrl.py -> "
                           "experiments/v32_04_preregistro_ftrl.json + reports/v32_p3_preregistro.md",
            "runner": "scripts/v32_05_ftrl_challenger.py -> experiments/v32_05_ftrl_challenger.json, "
                      "experiments/v32_05_estado.json, "
                      "artifacts/p33_scores/v32_p3/{a1_ftrl,a2_sgd_twin,a3_ftrl_barajado}.npy, "
                      "reports/v32_p3_ftrl_challenger.md, reports/figures/v32_p3_curvas.png, "
                      "fila V32-E3 en experiments.csv y seccion en RUNLOG.md",
        },
    }

    lineas = [
        "# V32 - E3 (P3) - Pre-registro congelado: challenger FTRL-Proximal sobre TF-IDF",
        "",
        f"**Fecha**: {FECHA}. **Estado**: CONGELADO antes de ejecutar (commit `{commit[:7]}`).",
        (f"**Baseline primario recomputado**: {baseline:.7f} "
         f"(referencia secundaria nested {NESTED})."),
        "",
        "## La pregunta y la hipotesis",
        "",
        "> El aprendizaje online proximal (FTRL-Proximal) sobre TF-IDF aporta informacion",
        "> complementaria al campeon V30 + plug-in que la familia lineal ya usada no captura?",
        "",
        "Ultimo candidato de la lista priorizada de la reapertura V32 (P1 D-081, P2 D-082).",
        "FTRL no esta implementado en el repositorio (novedad verificada en el grafo y por grep).",
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
        "## Corpus (identidad verificada, sin reordenar nada)",
        "",
        "- Receta: `src.data.load_train()` sobre `data/raw/age_train.csv` (sep `;`, encoding",
        "  `utf-8-sig`, `clean_text` aplicado); texto = columna `text`; orden de filas = orden",
        "  del CSV = orden canonico de `ids.npy` (verificado elemento a elemento) y etiquetas",
        "  = `y_true` canonico (verificado).",
        "- Hash de contenido determinista: longitud (8 bytes LE) + bytes UTF-8 por texto, en orden.",
        f"- **Desvio del manifiesto** (auditoria): su `data_hash` ({manifest.get('data_hash')})",
        "  se calculo con `to_numpy().tobytes()` sobre un array de OBJETOS -> serializa punteros",
        "  de memoria; no es reproducible (se observaron `68b9a0f2` y `40342f7c` en dos procesos).",
        "  Se congela en su lugar el hash del archivo + el hash de contenido + la receta.",
        "",
        "## Representacion (TF-IDF por fold)",
        "",
        "- Config IDENTICA a `VECTORIZADOR` de `v31_02_banco_representaciones.py`: word, (1,2),",
        "  min_df 3, max_df 0.5, sublinear_tf, lowercase, strip_accents None, token_pattern",
        "  `(?u)\\b\\w+\\b`; norm y dtype en defaults de sklearn.",
        "- **Ajuste por fold**: `fit` solo en fold-train (idf, min_df y max_df nunca globales);",
        "  precedente canonico del propio manifest. Verificabilidad: sha256 de config + vocabulario",
        "  ordenado + idf por fold + sonda de refit.",
        "",
        "## Brazos",
        "",
        "| Brazo | Definicion | Rol |",
        "|---|---|---|",
        ("| A1_ftrl | FTRL-Proximal propio (alpha 0.05, lambda1 0.0, lambda2 1.0, beta 1.0, bias "
         "regularizado x0=1, sin power_t, 10 pasadas fijas, OvR 6 sigmoides, barajado por epoca "
         f"con {SEMILLA}+fold) | decisional |"),
        ("| A2_sgd_twin | 6 x SGDClassifier(log_loss, alpha 1e-6, max_iter 10, tol None) con el "
         "mismo presupuesto de pasadas y la misma matriz | control de atribucion (decisional) |"),
        ("| A3_ftrl_barajado | A1 con etiquetas de fold-train permutadas (20261011+fold) | control "
         "negativo (no decisional) |"),
        "",
        "## Sondas obligatorias antes de entrenar (exit 4 si fallan)",
        "",
        "1. Vectorizado == lazo ingenuo (problema aleatorio 200x50x6, tolerancia 1e-12).",
        "2. Punto fijo analitico de FTRL con lambda1=lambda2=0 y feature constante (residuo < 1e-9).",
        "3. TotalF1:average=Macro == macro-F1 de sklearn (<= 1e-6).",
        "4. p sin NaN/inf y log_loss finito.",
        "",
        "## Regla de fallback (congelada antes de ejecutar, no es grid)",
        "",
        "- Criterio: en el fold 0 de A1, sobre las filas de fold-train, log_loss OvR medio tras",
        "  10 pasadas <= 0.8 x el log_loss tras la pasada 1, y sin NaN/inf.",
        "- Si falla: alternativa declarada alpha=0.005 (resto igual) y `config_fallback_usada=True`",
        "  en el JSON de resultados y en el informe.",
        "- Prohibido decidir con fold-val o con el leaderboard.",
        "",
        "## Determinismo",
        "",
        "- Semillas fijas; OMP/MKL/OPENBLAS = 32 antes de importar numpy; lazo FTRL mono-hilo.",
        "- Sonda: reentrenar fold 0 de A1 y de A2 y comparar bit a bit. Tolerancias: 0 exacto;",
        f"  <=1e-9 conforme; >1e-9 con rango publicado y delta_gate > rango; rango >= {VARIABILIDAD}",
        "  o solape con el gate => no concluyente (entonces no puede ser PROMISING).",
        "",
        "## Integracion y gates (mecanismo P6/P7, sin cambios)",
        "",
        "- Combinador anidado [L2(v30) || L2(p_brazo)] (12 features); LogReg C=1.0 por fold;",
        "  decision `argmax(p_comb * w)` con los pesos del plug-in CONGELADOS.",
        f"- Gates: baseline +-5e-7; delta_int(A1) >= +{GATE_DELTA:.4f}; folds+ >= {GATE_FOLDS}/5;",
        "  control favorable (delta A1 > delta A2 Y IC95 pareado de la diferencia sin cero);",
        "  A3 coherente (por debajo de la linea base de la clase mayoritaria); IC95 A1 sin cero;",
        "  sin leakage; determinismo concluyente.",
        "- **Caps**: IC95 de la diferencia con cero -> veredicto capado a MARGINAL; determinismo no",
        "  concluyente -> no PROMISING.",
        "",
        "## Arbol de decision",
        "",
        "| Resultado | Estado | Siguiente paso |",
        "|---|---|---|",
        "| A3 >= A1 | NO_ATRIBUIBLE | cierre documentado |",
        ("| gates completos + IC95 diferencia excluye 0 | PROMISING | segunda particion "
         f"(semilla {SEMILLA_SEGUNDA_PARTICION}) en este ciclo |"),
        "| delta > 0 sin gates (o cap) | MARGINAL | no se adopta |",
        "| delta <= 0 | CLOSED | se cierra P3 y termina la reapertura |",
        "",
        "## Compliance, novedad y limites",
        "",
        "- 0 embeddings, 0 transformers, 0 LLM, 0 datos externos; sin test, sin holdout, sin",
        "  leaderboard, sin submission; un script nuevo; campeon intacto. TF-IDF + lineal ya",
        "  usados por el proyecto: no cuenta como embedding vetado.",
        "- `vowpalwabbit 9.11.8` instalado y **no usado** (alternativa disponible descartada).",
        "- AST audit: ausencia de vowpal/vw/torch/transformers/gensim/fasttext y 0 rutas de",
        "  test/holdout/submission. RSS pico con `ru_maxrss` (tope 20 GB). Escritura atomica.",
        "- Asimetria declarada: FTRL regulariza el bias; SGD de sklearn no.",
        "- A2 se aparta de E-011 (max_iter 30/tol 1e-4 -> 10/tol=None) para igualar pasadas.",
        "- El cap por IC95 puede degradar un resultado genuinamente bueno a MARGINAL: decision",
        "  del lead, registrada.",
    ]
    texto = "\n".join(lin.rstrip() for lin in lineas) + "\n"
    PREREG_MD.write_text(texto, encoding="utf-8")
    print(f"  {PREREG_MD.name}: {len(texto.splitlines())} lineas", flush=True)

    PREREG_JSON.write_text(json.dumps(congelado, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  {PREREG_JSON.name}", flush=True)

    runlog = ROOT / "RUNLOG.md"
    marca = "Pre-registro de V32-E3"
    if marca not in runlog.read_text(encoding="utf-8"):
        bloque = ("\n---\n\n## Pre-registro de V32-E3 (P3 FTRL-Proximal) · " + FECHA + "\n\n"
                  "- Congelado: FTRL-Proximal propio sobre TF-IDF ajustada por fold; A2 gemelo "
                  "SGD con 10 pasadas; A3 barajado;\n  sondas (a)-(d) obligatorias; regla de "
                  "fallback escrita antes de ejecutar; cap por IC95 de la diferencia; "
                  "determinismo con\n  sonda bit a bit. Auditoria: el data_hash del manifest es un "
                  "hash de punteros, no reproducible; se congelan hashes de contenido.\n"
                  f"- Baseline recomputado: {baseline:.7f}. Sin test, sin holdout, sin submission.\n")
        with runlog.open("a", encoding="utf-8") as f:
            f.write(bloque)
        print("  RUNLOG.md: pre-registro anotado", flush=True)

    ruta_exp = ROOT / "experiments.csv"
    lineas_exp = ruta_exp.read_text(encoding="utf-8").splitlines()
    cols = [c.strip() for c in lineas_exp[0].split(",")]
    if "V32-E3-PREREG" not in {lin.split(",", 1)[0] for lin in lineas_exp[1:] if lin}:
        import csv as _csv
        import io as _io
        reg = {"experiment_id": "V32-E3-PREREG", "fecha": FECHA, "bloque": "32.3",
               "etapa": "V32-E3 (P3) pre-registro congelado del challenger FTRL-Proximal + "
                       "gemelo SGD + control barajado",
               "pregunta": "FTRL-Proximal sobre TF-IDF aporta complementariedad al campeon mas "
                           "alla del SGD ya usado?",
               "hipotesis": "el aprendizaje online proximal con L1/L2 explicito extrae otra "
                            "region del espacio de texto",
               "features": "TF-IDF por fold (word 1-2, min_df 3, max_df 0.5, sublinear_tf); 0 "
                           "embeddings",
               "preprocessing": "vocabulario e idf ajustados SOLO en fold-train",
               "representacion": "TF-IDF dispersa (nueva en este ciclo: no existia matriz canonica)",
               "modelo": "FTRL-Proximal propio OvR 6 (A1) / SGDClassifier log_loss gemelo (A2) / "
                         "FTRL barajado (A3)",
               "hiperparametros": "congelados: alpha 0.05 (alt 0.005 por regla de fallback), "
                                  "lambda1 0.0, lambda2 1.0, beta 1.0, 10 pasadas, 32 hilos, "
                                  "seed 20261010",
               "validacion": "particion canonica; integracion P6/P7; bootstrap de la diferencia "
                             "A1-A2; cap por IC95; sondas obligatorias",
               "macro_f1_mean": "", "macro_f1_std": "", "weighted_f1_mean": "",
               "micro_f1_mean": "", "accuracy_mean": "",
               "resultado": "pre-registro escrito y congelado; sin resultados aun",
               "interpretacion": "auditoria: el data_hash canonico es un hash de punteros no "
                                 "reproducible; se congelan hashes de contenido",
               "decision": "ejecutar V32-E3 tal cual",
               "duracion_seg": "0.0",
               "artefactos": "reports/v32_p3_preregistro.md | "
                             "experiments/v32_04_preregistro_ftrl.json"}
        buf = _io.StringIO()
        _csv.writer(buf, lineterminator="\n").writerow([reg[c] for c in cols])
        with ruta_exp.open("a", encoding="utf-8") as f:
            f.write(buf.getvalue())
        print("  experiments.csv: fila V32-E3-PREREG", flush=True)

    print(f"\n[fin] {time.time() - inicio:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
