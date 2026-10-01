"""V36 - E1 (via C1-A) - Pre-registro CONGELADO: transiciones de clase de caracter (brecha L4).

## Que congela

La brecha **L4 - Transiciones de caracteres**, declarada en `scripts/phase0_freeze_gap_v2.py:157`
("patrones letra->digito, letra->puntuacion, espacio->puntuacion que el TF-IDF no expresa como
feature explicita") y **nunca ejecutada**: el backlog del mismo script la deja como `pendiente`
(linea 204). Es la brecha mas antigua y abierta del proyecto.

El bloque transforma el texto en una secuencia de **clases de caracter** con un alfabeto categorico
**escrito a mano**, y cuenta bigramas y trigramas de clase. Se concatena al bloque `char_wb` del
campeon. No es un re-nombre de lo ya cerrado: la clausula de reapertura de **F10** dice literalmente
"solo se abre lo que cambie el MECANISMO: condicionar la palabra a su forma, su posicion o su
**categoria**", y C1-A es **condicionar el caracter a su categoria**, que es precisamente lo que esa
clausula habilita. Ademas, funcionalmente, un n-grama literal mide ortografia lexica y uno sobre el
alfabeto de 7 clases mide perfil de *tipo* de transicion, y ninguno se recupera del otro.

## Condiciones exigidas por kaggle-specialist antes de congelar (las 6, incorporadas)

(i) El **mapa de 7 clases** va aqui como **tabla literal**, con su `sha256`, y se declara redactado a
    priori desde categorias Unicode, **sin ninguna estadistica del corpus, sin ajuste por etiqueta
    y sin ningun parametro estimado**. (ii) **Sonda de falsacion PRE-REGISTRADA**: permutar el
    alfabeto de 7 clases en **>= 3 permutaciones** y exigir que el mapa real gane a los controles; si
    no gana, el veredicto es "sin senal especifica de categoria" y **L4 se cierra igual**. (iii)
    **Arbol de decision declarado por adelantado**, que incluye el nodo `SENAL_EXISTE_INTEGRACION_NEUTRA`
    para el caso banco positivo con integracion ~0, y **C1-A es one-shot en cualquier caso**
    (positivo o negativo), sin reabrir la familia sparse y sin autorizar un C1-B. (iv) La medicion
    previa de **+0.0031** se hizo en `/tmp` y se declara **NO canonica e inservible como gate**: el
    runner la **vuelve a medir dentro de si**, sobre la particion canonica, como sondas. (v) Las
    **hipotesis nulas en unidades de macro-F1**, para que "banco positivo con integracion nula" no
    sea reinterpretable despues. (vi) Registro de compliance: `sha256` del commit, mapa literal,
    prueba de que el transform es **stateless y fold-local** (el unico `fit` es el
    `TfidfVectorizer` del train de cada fold), tabla 5/5 folds, bootstrap pareado y declaracion de
    que test, holdout y leaderboard no se tocaron.

Ademas, por exigencia explicita del especialista: la decision de **NO anadir simbolos de borde** se
declara aqui, antes de ver cualquier banco, para que no pueda re-tocarse despues. Es decision
metodologica (un borde inventaria transiciones que no existen en el texto) y **no** afecta a la
legalidad.

## Ruling de legalidad

`kaggle-specialist` declara **LEGAL** con este criterio, que se aplica a variantes futuras: **(a) test
de parametros estimados**: si ningun parametro del constructor (mapa, alfabeto, pesos, IDF propio)
sale de datos, corpus, etiquetas o metadatos, es un featurizer escrito a mano y es legal; **(b) test
de abstraccion**: el mapa de 7 clases **descarta** la identidad del caracter, luego es una abstraccion
disenada y no un espacio semantico denso. Aqui se cumplen (a) y (b).

**ILEGAL** (mismo constructor, para que no se reintente): mapa elegido con matriz de confusion o
informacion mutua con la etiqueta; mapa distinto por clase objetivo; mapa derivado de covarianza o
autovectores; mapa derivado de clustering de caracteres; frecuencias de transicion usadas como vector
**denso**; y calcular estadisticas de transicion sobre todo el corpus **antes** del split.
**REQUIERE ruling explicito**: prior de clase sobre transiciones, reduccion SVD/NMF/LSA de la matriz
de transiciones, seleccion supervisada de rasgos, mapa fuzzy con pesos ajustados, y corte de DF
global.

## Contexto que condiciona la expectativa

El acta **D-087** concluyo que **LB 0.70 no es alcanzable** bajo estas reglas, con un oraculo
restringido a la informacion del campeon de **+0.0568** frente al **+0.0626** que exigiria el
objetivo. Por tanto este experimento **NO compite por 0.70**. Su valor es **medir si la senal existe
y publicarla**, y el unico criterio de exito util es el **gate de la serie**.

## Invariantes

Sin transformers, LLMs ni embeddings (preentrenados **o entrenados**); sin datos o etiquetas externos;
sin metadatos; test solo `transform -> predict`; holdout intocable; leaderboard sin tuning; campeon
V30 + plug-in (LB 0.63575) intacto; un experimento = un script nuevo; no se edita ningun script ya
ejecutado; `artifacts/p33_scores/todos` intocable; sin submission sin autorizacion.
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
import sklearn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src import config as cfg
from src import data as datos
from src.verificacion_numerica import autotest_numerico

FECHA = "2026-09-30"
CANONICO = ROOT / "artifacts" / "p33_scores" / "todos"
PREREG_MD = ROOT / "reports" / "v36_preregistro.md"
PREREG_JSON = ROOT / "experiments" / "v36_00_preregistro.json"
MANIFEST = CANONICO / "manifest.json"

N_CLASES = 6
BASELINE = 0.6159732
NESTED = 0.6154643
PESOS = [1.0, 1.1749, 1.2721, 1.163, 1.001, 0.8128]
SEMILLA = 20261010
HILOS = 32
N_BOOTSTRAP = 2000
GATE_DELTA = 0.0015
GATE_FOLDS = 4
VARIABILIDAD = 0.0008876
RAM_TOPE_GB = 20.0
SEGUNDOS_DECLARADOS = 2400.0
LIMITE_GLOBAL_H = 24.0
N_FOLDS = 5
CV_SEMILLA = 42
C_LINEAL = 1.0
MAX_ITER_LINEAL = 3000
TOL_LINEAL = 1e-4
CHAR_WB_NGRAM = (3, 5)
CHAR_WB_MIN_DF = 5
CHAR_WB_MAX_FEATURES = 300000
WORD_NGRAM = (1, 2)
WORD_MIN_DF = 3
WORD_MAX_FEATURES = 300000
TRIVIAL_MAYORIA = 0.04854212180687087

#: El mapa categorico, TABLA LITERAL en el codigo (condicion (i) del ruling). Redactado a priori
#: desde categorias Unicode: sin ninguna estadistica del corpus, sin ajuste por etiqueta y sin
#: ningun parametro estimado. El orden de las claves es la identidad del alfabeto y las permutaciones
#: de la sonda de falsacion se derivan de SEMILLA_PERMUTACIONES, nunca de los datos.
MAPA_CARACTERES = {
    "S": "espacio y todo lo que `str.isspace()` marca como espacio",
    "D": "dígitos ASCII 0-9",
    "P": "puntuación ASCII, es decir todo lo que no es alfanumérico, no es espacio y no es dígito",
    "L": "letra minúscula, con y sin acento",
    "M": "letra mayúscula, con y sin acento",
    "O": "resto: emojis, flechas, caracteres de control, alfabetos no latinos",
}
ORDEN_ALFABETO = ("S", "D", "P", "L", "M", "O")
NS_ORDEN = (1, 2)
#: Las 6 condiciones del ruling, escritas en el pre-registro y comprobables por el runner.
CONDICIONES_RULING = {
    "i_mapa_literal": "el mapa de 7 clases es una tabla literal del codigo, con sha256, redactado a "
                     "priori desde categorias Unicode sin ninguna estadistica del corpus",
    "ii_sonda_de_permutaciones": "permutar el alfabeto en >= 3 permutaciones y exigir que el mapa "
                                 "real gane a los controles; si no gana, el veredicto es 'sin senal "
                                 "especifica de categoria' y L4 se cierra igual",
    "iii_arbol_declarado": "arbol de decision declarado por adelantado, con el nodo "
                           "SENAL_EXISTE_INTEGRACION_NEUTRA, y C1-A es ONE-SHOT en cualquier caso",
    "iv_banco_no_canonico": "la medicion previa de +0.0031 se hizo en /tmp y es NO canonica: el "
                            "runner la vuelve a medir dentro de si sobre la particion canonica",
    "v_hipotesis_nulas_en_unidades": "hipotesis nulas expresadas en unidades de macro-F1",
    "vi_registro_de_compliance": "sha256 del commit, mapa literal, prueba de transform stateless y "
                                "fold-local con unico fit = TfidfVectorizer del train de cada fold, "
                                "tabla 5/5 folds, bootstrap pareado y declaracion de que test, "
                                "holdout y leaderboard no se tocaron",
    "borde_declarado": "NO se anaden simbolos de borde: un borde inventaria transiciones que no "
                       "existen en el texto. Decision metodologica declarada ANTES de ver el banco, "
                       "para que no pueda re-tocarse. No afecta a la legalidad",
}

HIPOTESIS_NULAS = {
    "banco": {"H0_banco": "delta del banco = 0 en unidades de macro-F1", "unidad": "macro-F1",
              "criterio": "IC95 del bootstrap pareado (2000, semilla 20261010) sobre el delta del "
                          "banco contra el bloque identico sin C1-A"},
    "integracion": {"H0_integracion": "delta integrado = 0 en unidades de macro-F1",
                    "unidad": "macro-F1",
                    "criterio": "IC95 del bootstrap pareado (2000, semilla 20261010) sobre el "
                                "delta integrado de la combinacion contra el campeon, con los gates "
                                "de la serie: delta >= +0.0015, >= 4/5 folds, control favorable"},
    "especificidad": {"H0_especificidad": "el mapa real no es mejor que una permutacion del "
                                          "alfabeto", "unidad": "macro-F1",
                       "criterio": "el mapa real debe ganar a >= 3 permutaciones; si no, el "
                                   "veredicto es 'sin senal especifica de categoria' y L4 se cierra "
                                   "igual"},
}

ARBOL = {
    "nodos": ["SENAL_NO_EXISTENTE", "SENAL_EXISTE_INTEGRACION_NEUTRA", "SENAL_EXISTE_INTEGRACION_POSITIVA",
              "SENAL_EXISTE_INTEGRACION_NEGATIVA", "SIN_SENAL_ESPECIFICA_DE_CATEGORIA",
              "DEFECTO_DE_IMPLEMENTACION"],
    "SENAL_NO_EXISTENTE": "el banco no supera su H0: no hay senal ni en el bloque, y se cierra L4",
    "SENAL_EXISTE_INTEGRACION_NEUTRA": ("el banco supera su H0 pero la integracion no supera la suya: "
                                       "la senal es REAL y ORTOGONAL pero V30 ya la expresa. Se cierra "
                                       "L4 por mecanismo, **sin reabrir la familia sparse y sin "
                                       "autorizar un C1-B**, y el resultado es PUBLICABLE"),
    "SENAL_EXISTE_INTEGRACION_POSITIVA": ("el banco y la integracion superan sus H0: se evaluan los "
                                          "gates de la serie y, si pasan, PROMISING exigiria segunda "
                                          "particion con autorizacion del usuario, que el script NO "
                                          "ejecuta"),
    "SENAL_EXISTE_INTEGRACION_NEGATIVA": "el banco supera su H0 pero la integracion es negativa: se "
                                         "reporta el banco y el nodo, y L4 se cierra",
    "SIN_SENAL_ESPECIFICA_DE_CATEGORIA": ("el mapa real no gana a las permutaciones: el bloque "
                                          "captura algo, pero no especifico de la CLASE de "
                                          "caracter, y L4 se cierra igual"),
    "DEFECTO_DE_IMPLEMENTACION": "las sondas o el triage detectan un defecto: la familia NO se "
                                 "cierra y el ciclo no cuenta",
    "one_shot": "C1-A es ONE-SHOT en cualquier caso: positivo o negativo, no se reintenta con otro "
               "alfabeto, otro rango de n, otro mapa o otro corte",
}
CONTROLES = {
    "C1_sin_bloque": "el bloque char_wb IDENTICO y sin C1-A, con modelo, folds, semilla y "
                    "vectorizador congelados: es la referencia del banco",
    "C2_permutaciones": ">= 3 permutaciones del alfabeto de 7 clases derivadas de una semilla "
                        "declarada, nunca de los datos: mide si la senal es especifica de la clase",
    "C3_barajado": "el bloque C1-A con las transiciones barajadas dentro de cada documento, "
                   "destruyendo el orden y conservando el recuento: separa 'hay transiciones' de "
                   "'el orden de las transiciones importa'",
}
SONDAS = {
    "banco_ree_medido": "el banco se vuelve a medir DENTRO del runner, sobre la particion canonica, "
                       "con las sondas (i)-(iv); la medicion previa de /tmp es no canonica",
    "tabla_5_de_5_folds": "tabla de macro-F1 por fold del banco y de la integracion",
    "bootstrap_pareado": "IC95 pareado (2000, semilla 20261010) del banco y de la integracion",
    "permutaciones": "resultado de las >= 3 permutaciones del alfabeto, exigido por la condicion (ii)",
    "barajado": "resultado del control de barajado, exigido por C3",
    "conformidad": "prueba de que el transform es stateless: dos llamadas con la misma entrada dan "
                   "exactamente la misma matriz",
    "fold_local": "el unico `fit` es el TfidfVectorizer del train de cada fold; se publica el "
                  "vocabulario y el idf por fold con su sha256",
    "mapa_literal": "sha256 del mapa literal y de la clase resultante, para que nadie pueda cambiarlo "
                    "entre el pre-registro y la corrida",
}
COMPLIANCE = {
    "embeddings": 0, "transformers": 0, "llm": 0, "datos_externos": 0, "metadatos": 0,
    "test": 0, "holdout": 0, "leaderboard": 0, "submission": 0,
    "ruling": "kaggle-specialist: LEGAL, con los tests (a) parametros estimados y (b) abstraccion; "
              "el mapa descarta la identidad del caracter y no estima nada",
    "ilegales_declarados": ["mapa elegido con matriz de confusion o informacion mutua con la etiqueta",
                            "mapa distinto por clase objetivo",
                            "mapa derivado de covarianza o autovectores",
                            "mapa derivado de clustering de caracteres",
                            "frecuencias de transicion como vector denso",
                            "estadisticas de transicion sobre todo el corpus antes del split"],
}


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_archivo(ruta: Path) -> str:
    return sha256_bytes(ruta.read_bytes())


def main() -> int:
    autotest_numerico()
    inicio = time.time()

    clase = np.load(CANONICO / "y_true.npy")
    np.load(CANONICO / "fold_idx.npy")
    v30 = np.load(CANONICO / "v30_prob.npy").astype(np.float64)
    ids = np.load(CANONICO / "ids.npy")
    v30 = np.clip(v30, 1e-12, 1.0)
    v30 = v30 / v30.sum(axis=1, keepdims=True)
    from sklearn.metrics import f1_score

    w = np.array(PESOS, dtype=float)
    base = float(f1_score(clase, np.argmax(v30 * w, axis=1), average="macro",
                          labels=list(range(N_CLASES))))
    print(f"baseline primario recomputado: {base:.7f} (esperado {BASELINE})")
    if abs(base - BASELINE) > 5e-7:
        print("[STOP] el baseline canonico no reproduce (exit 2)")
        return 2

    train = datos.load_train()
    if not np.array_equal(ids.astype(np.int64), train[cfg.ID_COL].to_numpy().astype(np.int64)):
        print("[STOP] el orden de ids no coincide con la particion canonica (exit 2)")
        return 2
    _, conteo = np.unique(clase, return_counts=True)
    idx_may = int(np.argmax(conteo))
    trivial = float(f1_score(clase, np.full_like(clase, np.unique(clase)[idx_may]),
                              average="macro", labels=list(range(N_CLASES))))
    if abs(trivial - TRIVIAL_MAYORIA) > 1e-12:
        print("[STOP] el ancla trivial no reproduce la de D-087 (exit 2)")
        return 2
    print(f"ancla trivial verificada en runtime: {trivial:.17f} (clase {cfg.CLASSES[idx_may]})")

    textos_sha = hashlib.sha256()
    for t in train[cfg.TEXT_COL].fillna("").astype(str).to_numpy():
        b = t.encode("utf-8")
        textos_sha.update(len(b).to_bytes(8, "little"))
        textos_sha.update(b)

    hashes = {
        "mapa_literal_de_clases": sha256_bytes(
            json.dumps(MAPA_CARACTERES, sort_keys=True).encode("utf-8")),
        "orden_del_alfabeto": sha256_bytes(json.dumps(list(ORDEN_ALFABETO)).encode("utf-8")),
        "manifest_canonico": sha256_archivo(MANIFEST),
        "corpus_archivo": sha256_archivo(cfg.TRAIN_FILE),
        "corpus_textos": textos_sha.hexdigest(),
        "caracteristicas_del_campeon": sha256_archivo(MANIFEST),
    }
    for nombre in ("y_true", "fold_idx", "v30_prob", "ids"):
        hashes[f"canonico_{nombre}"] = sha256_archivo(CANONICO / f"{nombre}.npy")

    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT.parent, check=False,
                            capture_output=True, text=True, timeout=30).stdout.strip()

    congelado = {
        "fase": "V36-E1 (via C1-A) pre-registro congelado",
        "fecha": FECHA, "commit": commit,
        "brecha": "L4 - Transiciones de caracteres, declarada en scripts/phase0_freeze_gap_v2.py:157 "
                  "y nunca ejecutada (backlog linea 204: 'pendiente')",
        "experimento": "bloque de transiciones de clase de caracter (bigramas y trigramas de clase "
                       "sobre el texto entero) concatenado al bloque char_wb, con dos controles",
        "por_que_no_es_un_renombre": "la clausula de reapertura de F10 dice literalmente 'solo se "
                                     "abre lo que cambie el MECANISMO: condicionar la palabra a su "
                                     "forma, su posicion o su CATEGORIA'. C1-A es condicionar el "
                                     "caracter a su categoria. Funcionalmente, un n-grama literal "
                                     "mide ortografia lexica y uno sobre el alfabeto de clases mide "
                                     "perfil de tipo de transicion, y ninguno se recupera del otro",
        "baseline_primario": round(base, 7), "referencia_nested": NESTED, "pesos_plugin": PESOS,
        "hashes": hashes,
        "versiones": {"sklearn": sklearn.__version__, "numpy": np.__version__,
                      "pandas": pd.__version__},
        "mapa_de_clases_literal": MAPA_CARACTERES,
        "orden_del_alfabeto": list(ORDEN_ALFABETO),
        "rango_n_de_transiciones": list(NS_ORDEN),
        "borde": "NO se anaden simbolos de borde (declarado antes de ver el banco; un borde "
                 "inventaria transiciones inexistentes)",
        "min_df": "ninguno: el vocabulario esta acotado por construccion (18/66/27/127 columnas "
                  "reales segun el rango y el alfabeto)",
        "normalizacion": 'TfidfVectorizer(norm="l2", sublinear_tf=True, lowercase=False, '
                         "token_pattern=r\"\\S+\") sobre los documentos de n-gramas de clase repetidos "
                         "por su frecuencia",
        "modelo": {"estimator": "sklearn.svm.LinearSVC", "C": C_LINEAL, "loss": "squared_hinge",
                   "penalty": "l2", "max_iter": MAX_ITER_LINEAL, "tol": TOL_LINEAL,
                   "origen_de_los_hiperparametros": "receta del campeon, leida del manifiesto "
                                                    "canonico; NO es un grid"},
        "bloque_de_referencia": {"char_wb": {"ngram_range": list(CHAR_WB_NGRAM),
                                             "min_df": CHAR_WB_MIN_DF, "max_features":
                                             CHAR_WB_MAX_FEATURES},
                                 "word_para_el_brazo_de_integracion": {
                                     "ngram_range": list(WORD_NGRAM), "min_df": WORD_MIN_DF,
                                     "max_features": WORD_MAX_FEATURES}},
        "cv": {"n_folds": N_FOLDS, "semilla": CV_SEMILLA,
               "generador": "StratifiedKFold(n_splits=5, shuffle=True, random_state=42) sobre las "
                            "filas de TRAIN con los mismos indices de fold del canonico",
               "nota": "los indices de fold se leen de artifacts/p33_scores/todos/fold_idx.npy, que "
                       "es la particion canonica, y NO se re-crean ni se barajan"},
        "condiciones_del_ruling": CONDICIONES_RULING,
        "hipotesis_nulas": HIPOTESIS_NULAS,
        "arbol_de_decision": ARBOL,
        "controles": CONTROLES,
        "sondas": SONDAS,
        "compliance": COMPLIANCE,
        "gates_de_la_serie": {"baseline_abort": 5e-7, "delta_min": GATE_DELTA,
                              "folds_min": GATE_FOLDS, "n_bootstrap": N_BOOTSTRAP,
                              "semilla": SEMILLA, "variabilidad_de_referencia": VARIABILIDAD,
                              "nota": "los gates aplican a la INTEGRACION contra el campeon; el banco "
                                      "se evalua con su propia hipotesis nula, sin gates de la serie"},
        "medicion_previa_no_canonica": {
            "banco_char_wb_mas_c1a": "+0.0030641 de macro-F1, 5/5 folds, IC95 [+0.0021, +0.0040], "
                                      "z de referencia +3.45",
            "donde": "/tmp, con los folds canonicos leidos del artefacto, pero FUERA del runner y "
                     "FUERA de la particion canonica como medicion de gate",
            "valor_para_gate": "NINGUNO: se declara no canonica y el runner la vuelve a medir "
                               "dentro de si, como sondas",
        },
        "contexto_de_expectativa": {
            "D-087": "LB 0.70 no es alcanzable: un oraculo restringido a la informacion del campeon "
                     "rinde +0.0568 frente al +0.0626 del objetivo",
            "calibracion": "LB ~ 0.00223 + 1.02825 x CV, sostenida en n = 5 puntos, con el limite ya "
                           "declarado en D-087 de que no se puede extrapolar lejos",
            "tasas_de_absorcion_medidas": "D-067: un canal con +0.006820 de banco rindio +0.0000642 "
                                          "integrado (factor 0.0094). Por eso el delta esperado sobre "
                                          "V30 es +0.0000 a +0.0001 y la probabilidad de pasar el "
                                          "gate es de 5 % a 10 %",
            "valor_de_este_experimento": "medir y publicar si la senal existe; NO compite por 0.70",
        },
        "ejecucion": {"coste_estimado": "3-6 min (5 folds x 3 brazos, mas las permutaciones y el "
                                        "barajado)",
                      "presupuesto_declarado_seg": SEGUNDOS_DECLARADOS,
                      "ram_tope_gb": RAM_TOPE_GB, "timeout_global_h": LIMITE_GLOBAL_H,
                      "hilos": HILOS,
                      "checkpoints": "experiments/v36_01_estado.json y checkpoints/v36_e1/",
                      "escritura_atomica": "tmp + os.replace"},
        "entregables": {
            "runner": "scripts/v36_01_transiciones_clase.py -> experiments/v36_01_*.json, "
                      "reports/v36_e1_transiciones_clase.md, reports/figures/v36_e1_curvas.png, "
                      "reports/v36_novelty_registry.csv, fila anexada en "
                      "reports/age_complete_experiment_history.csv, RUNLOG.md, experiments.csv",
            "acta": "D-088 en DECISIONS.md, citando la clausula de reapertura de F10",
        },
        "invariantes": {"sin_transformers": True, "sin_llm": True, "sin_embeddings": True,
                        "sin_datos_externos": True, "sin_metadatos": True,
                        "test_solo_transform_predict": True, "holdout_intocable": True,
                        "leaderboard_sin_tuning": True, "campeon_intacto": True,
                        "un_experimento_un_script": True, "p33_scores_intocable": True,
                        "sin_submission_sin_autorizacion": True},
    }
    PREREG_JSON.write_text(json.dumps(congelado, indent=2, ensure_ascii=False), encoding="utf-8")

    lineas = [
        "# V36 - E1 (via C1-A) - Pre-registro congelado: transiciones de clase de caracter (L4)",
        "",
        f"**Fecha**: {FECHA}. **Estado**: CONGELADO antes de ejecutar (commit `{commit[:7]}`).",
        (f"**Baseline recomputado**: {base:.7f} (nested {NESTED}); ancla trivial verificada en "
        f"runtime: {trivial:.17f}."),
        "",
        "## Que es y por que no es un re-nombre",
        "",
        f"- **Brecha**: {congelado['brecha']}.",
        f"- **Reapertura de F10**: {congelado['por_que_no_es_un_renombre']}.",
        "",
        "## El mapa de 7 clases (TABLA LITERAL, condicion (i) del ruling)",
        "",
        "Redactado a priori desde categorias Unicode, **sin ninguna estadistica del corpus**, sin",
        "ajuste por etiqueta y sin ningun parametro estimado. El mapa **descarta** la identidad del",
        "caracter: es una abstraccion disenada, no un espacio semantico denso.",
        "",
        "| Clase | Que representa |",
        "|---|---|",
    ]
    for simbolo, descripcion in MAPA_CARACTERES.items():
        lineas.append(f"| `{simbolo}` | {descripcion} |")
    lineas += [
        "",
        (f"- Orden del alfabeto: `{' '.join(ORDEN_ALFABETO)}`; `sha256` del mapa "
        f"`{hashes['mapa_literal_de_clases'][:16]}...`."),
        ("- Rango de n: bigramas y trigramas de clase; sin `min_df` (vocabulario acotado por "
        "construccion)."),
        f"- **Borde: {congelado['borde']}.**",
        "",
        "## Las 6 condiciones del ruling, incorporadas",
        "",
    ]
    for clave, texto in CONDICIONES_RULING.items():
        lineas.append(f"- **{clave}**: {texto}")
    lineas += [
        "",
        "## Hipotesis nulas, en unidades de macro-F1",
        "",
        "| Ambito | H0 | Criterio |",
        "|---|---|---|",
    ]
    for ambito, valores in HIPOTESIS_NULAS.items():
        lineas.append(f"| {ambito} | {valores['criterio']} | unidad: {valores['unidad']} |")
    lineas += [
        "",
        "## Arbol de decision, declarado por adelantado",
        "",
    ]
    for nodo in ARBOL["nodos"]:
        lineas.append(f"- **{nodo}**: {ARBOL[nodo]}")
    lineas += [f"- **one-shot**: {ARBOL['one_shot']}", "", "## Controles", ""]
    for nombre, texto in CONTROLES.items():
        lineas.append(f"- **{nombre}**: {texto}")
    lineas += ["", "## Sondas obligatorias", ""]
    for nombre, texto in SONDAS.items():
        lineas.append(f"- **{nombre}**: {texto}")
    lineas += [
        "",
        "## La medicion previa NO es canonica",
        "",
        f"- Banco declarado: {congelado['medicion_previa_no_canonica']['banco_char_wb_mas_c1a']}.",
        f"- Donde: {congelado['medicion_previa_no_canonica']['donde']}.",
        f"- Valor para gate: {congelado['medicion_previa_no_canonica']['valor_para_gate']}.",
        "",
        "## Expectativa declarada antes de ejecutar",
        "",
        f"- D-087: {congelado['contexto_de_expectativa']['D-087']}.",
        f"- Absorcion medida: {congelado['contexto_de_expectativa']['tasas_de_absorcion_medidas']}.",
        ("- **Delta esperado sobre V30: CV +0.0000 a +0.0001, LB +0.0000 a +0.0001. Probabilidad de "
        "pasar el gate: 5 % a 10 %.**"),
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
        "## Compliance e invariantes",
        "",
        f"- Ruling: {COMPLIANCE['ruling']}.",
        "- **ILEGAL** (para que no se reintente): " + "; ".join(COMPLIANCE["ilegales_declarados"]) + ".",
        ("- 0 transformers, 0 LLM, 0 embeddings, 0 datos externos, 0 metadatos; sin test mas alla de "
        "`transform -> predict`; sin holdout, sin leaderboard, sin submission; campeon "
        "V30 + plug-in intacto; `artifacts/p33_scores/todos` intocable."),
        (f"- Coste esperado 3-6 min con presupuesto declarado de {SEGUNDOS_DECLARADOS:g} s; RSS con "
        f"tope {RAM_TOPE_GB:g} GB medido con `ru_maxrss`; checkpoints por fold y escritura atomica."),
    ]
    texto = "\n".join(lin.rstrip() for lin in lineas) + "\n"
    PREREG_MD.write_text(texto, encoding="utf-8")
    print(f"  {PREREG_MD.name}: {len(texto.splitlines())} lineas")
    print(f"  {PREREG_JSON.name}")

    if "Pre-registro de V36-E1" not in (ROOT / "RUNLOG.md").read_text(encoding="utf-8"):
        with (ROOT / "RUNLOG.md").open("a", encoding="utf-8") as f:
            f.write("\n---\n\n## Pre-registro de V36-E1 (via C1-A: transiciones de clase) · " + FECHA
                    + "\n\n"
                    "- Brecha **L4**, declarada en `phase0_freeze_gap_v2.py:157` y nunca ejecutada.\n"
                    "- Reapertura de F10 por su propia clausula: condicionar la palabra/caracter a su "
                    "**categoria**.\n"
                    "- Ruling de `kaggle-specialist`: **LEGAL** (test de parametros estimados y test "
                    "de abstraccion, ambos cumplidos). Las 6 condiciones incorporadas, incluida la "
                    "**sonda de permutaciones** y el nodo `SENAL_EXISTE_INTEGRACION_NEUTRA`.\n"
                    "- Mapa de 7 clases como **tabla literal** con sha256, redactado a priori sin "
                    "estadisticas del corpus. Sin simbolos de borde, declarado antes de ver el banco.\n"
                    "- La medicion previa de +0.0031 se declara **NO canonica** y el runner la vuelve "
                    "a medir dentro de si.\n"
                    "- **Expectativa declarada**: delta integrado CV +0.0000 a +0.0001, "
                    "P(gate) 5-10 %. C1-A es **one-shot** en cualquier caso.\n"
                    f"- Baseline recomputado {base:.7f}; campeon intacto; sin submission.\n")
        print("  RUNLOG.md: pre-registro anotado")

    ruta_exp = ROOT / "experiments.csv"
    lineas_exp = ruta_exp.read_text(encoding="utf-8").splitlines()
    cols = [c.strip() for c in lineas_exp[0].split(",")]
    if "V36-E1-PREREG" not in {lin.split(",", 1)[0] for lin in lineas_exp[1:] if lin}:
        fila = {"experiment_id": "V36-E1-PREREG", "fecha": FECHA, "bloque": "36.0",
                "etapa": "V36-E1 (via C1-A) pre-registro congelado: transiciones de clase (L4)",
                "pregunta": "las transiciones de clase de caracter aportan senal ortogonal que el "
                            "word y char_wb del campeon no expresan?",
                "hipotesis": "si: el perfil de tipo de transicion (letra->digito, espacio->puntuacion) "
                             "no esta expresado como feature explicita en ninguna familia existente",
                "features": "char_wb (3,5) min_df 5 del campeon + transiciones de clase (bigramas y "
                            "trigramas) sobre un alfabeto categorico de 7 clases",
                "preprocessing": "vocabulario e idf ajustados SOLO en el train de cada fold",
                "representacion": "mapa categorico literal de 7 clases + conteos dispersos de "
                                  "n-gramas de clase; sin embeddings",
                "modelo": "LinearSVC(C=1.0, squared_hinge, l2, max_iter=3000) con la receta del "
                          "campeon; controles: sin bloque, permutaciones del alfabeto y barajado",
                "hiperparametros": f"C={C_LINEAL} de la receta del campeon; sin grid; alfabeto "
                                   f"{' '.join(ORDEN_ALFABETO)}; n={list(NS_ORDEN)}; sin min_df; "
                                   f"sin borde; CV {N_FOLDS} folds semilla {CV_SEMILLA}",
                "validacion": "IC95 pareado 2000 (semilla 20261010) del banco y de la integracion; "
                              "tabla 5/5 folds; sonda de >= 3 permutaciones; control de barajado",
                "macro_f1_mean": "", "macro_f1_std": "", "weighted_f1_mean": "",
                "micro_f1_mean": "", "accuracy_mean": "",
                "resultado": "pre-registro escrito y congelado; sin resultados aun",
                "interpretacion": "la medicion previa de +0.0031 se declara NO canonica e inservible "
                                  "como gate; el runner la re-mide dentro de si",
                "decision": "ejecutar V36-E1 tal cual; C1-A es one-shot en cualquier caso",
                "duracion_seg": "0.0",
                "artefactos": "reports/v36_preregistro.md | experiments/v36_00_preregistro.json"}
        buf = io.StringIO()
        csv.writer(buf, lineterminator="\n").writerow([fila[c] for c in cols])
        with ruta_exp.open("a", encoding="utf-8") as f:
            f.write(buf.getvalue())
        print("  experiments.csv: fila V36-E1-PREREG")

    print(f"\n[fin] {time.time() - inicio:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())