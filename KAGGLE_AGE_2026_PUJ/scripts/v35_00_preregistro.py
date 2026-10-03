"""V35 - E1 (via B) - Pre-registro CONGELADO: lambda1 > 0 sobre el nucleo FTRL de V34.

## Que congela

Reinicio **one-shot** de la familia FTRL-Proximal (cerrada por mecanismo en D-085) sobre su primer
miembro declarado **NO CUBIERTO**: `lambda1 > 0`. El acta de reinicio es **D-086** (se anexa a
`DECISIONS.md` en el cierre de este ciclo, y su hash se fija aqui). Carlos firmo la via B el
2026-09-30.

Este NO es un reintento de `alpha`: `alpha*` queda **congelado e identico** al de V34-E1
(`2*ln(5)/(d_norm*P_max^2)`), y lo que cambia es el espacio de soluciones (un umbral duro
`w_j = 0` si `|z_j| <= lambda1`), no el punto del learning-rate.

## Condiciones incorporadas (kaggle-specialist, REINICIO LEGAL con condiciones)

1. El acta D-086 debe quedar **escrita y congelada (sha256 + commit) ANTES de ejecutar**, nombrando
   el miembro NO CUBIERTO por su identificador y repitiendo como **intocables** los miembros ya
   cerrados: OvR, `power_t`, multi-pasada, `lambda2` en {.05, .2, 1, 1e-6}, L2 en acumulador vs
   denominador y `alpha`.
2. **Criterio que distingue "L1 activo" de "L1 inerte"**, declarado: si en pasadas >= 2 la fraccion
   de coordenadas con `|z_j| <= lambda1` fuese 0, la hipotesis esta **analiticamente muerta** y se
   reporta asi, sin culpar a la puerta. Este criterio es la **columna vertebral de G7**.
3. **G7 y G8 se ancran a magnitudes medidas**, no a numeros arbitrarios: G7 reproduce el valor de
   la pasada 1 dentro de una tolerancia nombrada y exige **no degradacion declarada**; G8 se ancla a
   la hipotesis misma (`||w||_2` no creciente, razon <= 1.0), que es exactamente la propiedad que
   el L1 deberia aportar y la que la sonda pre-ejecucion **ya refuto**.
4. **G7 y G8 aplican SOLO a A1**: A2 (control) no tiene L1, luego aplicarle G7/G8 fabricaria un
   `NO_ATRIBUIBLE_BY_GATE` propio. A2 se somete a G0-G6, que son las gates de la serie.
5. **Rama pre-declarada**: si G7 y G8 **pasan** y G0-G4 **fallan**, la hipotesis conjunta queda
   **refutada** (acota pero no aprende) y se emite `HIPOTESIS_CONJUNTA_REFUTADA`, que **no** es cierre
   por conveniencia. Si G7 o G8 **fallan**, la acotacion queda refutada y la familia se cierra
   **DEFINITIVAMENTE** con `LINEA_ONLINE_CERRADA_POR_MECANISMO`.
6. **Asimetria declarada y lectura unidireccional**: A1 tiene un regularizador extra que A2 no
   tiene, luego **A2 es control de PRESUPUESTO, no gemelo mecanico**. Ninguna conclusion puede
   descansar en el contraste A1 vs A2 (delta, razones entre brazos, IC bootstrap cruzado quedan
   **contaminados**): el veredicto se apoya solo en evidencia interna de A1 contra la regla derivada
   (G0/G1/G4) mas el triage.
7. **Regla terminal one-shot**: si `lambda1 > 0` falla, el segundo miembro NO CUBIERTO (pesos por
   clase con L2 normalizado fuera de FTRL) queda **cerrado por declaracion** con su evidencia de
   techo, y la familia queda cerrada sin Acta futura dentro de este espacio. No hay segundo
   `lambda1`, sin fallback, sin re-derivacion posterior a las gates; una unica re-emision se
   permite solo por `DEFECTO_DE_IMPLEMENTACION`, nunca por cambio de hipotesis.
8. **El gemelo sintetico del arquitecto y la sonda pre-ejecucion se versionan como artefactos
   trazables** y ademas se **reproducen DENTRO del runner** como sondas no-gate (k), (l), (m), porque
   D-085 prohibio que una decision de mecanismo descanse en evidencia importada de `/tmp`.

## Los dos flags tecnicos, YA RESUELTOS con medicion (scripts/v35_00b_sonda_lambda1.py)

- **Flag 1** ("`lambda1 = alpha` es un no-op" exige `mediana(|z|) > alpha`): **resuelto**. Medido en
  TRAIN del fold 0: `mediana(|z|>0) = 0.23993793862904017` con `alpha* = 0.035262336465737745`, luego
  `mediana/alpha = 6.8044`. Cuantiles: q1 1.07e-4, q25 0.0637, q50 0.2399, q75 0.6777, q99 7.9102;
  fraccion con `|z| <= alpha` = 0.144558; `sha256` del snapshot de `z` = `ec1d1a7b13454f8b...`.
- **Flag 2** ("un `lambda1` fijo acota `||w||_2`"): **resuelto en negativo**. Con las mismas reglas de
  update y los mismos ordenes congelados, 3 pasadas:
  `lambda1 = 0` -> `ll_train` 10.086314 -> 19.174421, `||w||_2` 29.19 -> 60.98;
  `lambda1 = alpha` -> 10.083241 -> 19.172685, 29.03 -> 60.86 (no-op, coherente con flag 1);
  `lambda1 = mediana` -> 10.058343 -> 19.159444, **28.27 -> 60.24**. La razon max/min de `||w||_2`
  con `lambda1 = mediana` es **2.1304** (> 1.0): el umbral L1 **amortigua pero no acota**, y la
  divergencia queda casi intacta.

**Prediccion registrada antes de ejecutar**: con esta evidencia, el veredicto mas probable es
`LINEA_ONLINE_CERRADA_POR_MECANISMO` por fallo de G8 (la hipotesis de acotacion ya esta refutada
por medicion). El ciclo **no** puede terminar en PROMISING: `alpha` esta congelado e identico al que
ya divergio, y el unico miembro que cambia es el L1.

## Invariantes

Sin test, sin holdout, sin leaderboard, sin submission, sin datos externos, sin transformers, LLMs
ni embeddings; campeon V30 + plug-in intacto; un experimento = un script nuevo; no se edita ningun
script ya ejecutado; `artifacts/p33_scores/todos` intocable.
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
PREREG_MD = cfg.REPORTS / "v35_preregistro.md"
PREREG_JSON = cfg.EXPERIMENTS / "v35_00_preregistro.json"
SONDA_JSON = cfg.EXPERIMENTS / "v35_00b_sonda_lambda1.json"
SONDA_SCRIPT = ROOT / "scripts" / "v35_00b_sonda_lambda1.py"
NUCLEO_V34 = ROOT / "scripts" / "v34_01_ftrl_softmax_multipasada.py"

BASELINE = 0.6159732
NESTED = 0.6154643
PESOS = [1.0, 1.1749, 1.2721, 1.163, 1.001, 0.8128]
N_CLASES = 6
SEMILLA = 20261010
HILOS = 32
N_BOOTSTRAP = 2000
GATE_DELTA = 0.0015
GATE_FOLDS = 4
VARIABILIDAD = 0.0008876
RAM_TOPE_GB = 20.0
LIMITE_PROYECTADO_H = 20.0
LIMITE_GLOBAL_H = 24.0
SEGUNDOS_DECLARADOS = 1200.0
TRIVIAL_MAYORIA = 0.04854212180687087
LOG_UNIFORME_K6 = 1.791759469228055
LOG_K_MENOS_1 = 1.6094379124341003
P_MAX = 10
P_MIN = 3
PACIENCIA = 2
MEJORA_MINIMA = 1e-4
TEST_SIZE_ES = 0.10
LAMBDA2 = 1.0
BETA = 1.0
POWER_T = 0.0
X0 = 1.0
G1_PISO = 0.05
G1_TECHO_MARGEN = 0.01
G3_MINIMO = 0.30
G4_FACTOR_INFERIOR = 0.5
G4_FACTOR_SUPERIOR = 1.5
G6_TOLERANCIA = 1e-9
G2_TOLERANCIA = 1e-06
#: Tolerancia de G7 para reproducir la magnitud de pasada 1 (declarada, no derivada de datos).
G7_TOLERANCIA = 0.02
#: G8: ancla a la HIPOTESIS misma (acotar => la norma no crece), no a un factor arbitrario.
G8_RAZON_MAXIMA = 1.0

ALPHA_ESPERADO = 0.035262336465737745
MEDIANA_Z_ESPERADA = 0.23993793862904017
RAZON_MEDIANA_ALPHA = 6.8044

VECTORIZADOR = {"analyzer": "word", "ngram_range": [1, 2], "min_df": 3, "max_df": 0.5,
                "sublinear_tf": True, "lowercase": True, "strip_accents": None,
                "token_pattern": r"(?u)\\b\\w+\\b"}

ANCLAJE = {
    "definicion": "lambda1 := mediana(|z_j| > 0) medida en el stream de TRAIN del fold 0 al final de "
                  "la pasada 1 (un unico numero, cero busqueda)",
    "valor_esperado": MEDIANA_Z_ESPERADA,
    "medida_sobre": "solo filas del ES-train del fold 0; recipe exacta y sha256 del snapshot de z "
                    "declarados en la sonda pre-ejecucion (experiments/v35_00b_sonda_lambda1.json)",
    "flag_1_resuelto": "lambda1 = alpha es no-op porque mediana/alpha = 6.8044 > 1 (medido)",
    "sensibilidad_declarada": "q=0.25 y q=0.75 escalan lambda1 ~ 1/sqrt(q); esa sensibilidad SOLO "
                              "se ejecuta en gemelo sintetico, jamas sobre datos reales",
    "prohibido": "re-derivar lambda1 tras ver las gates; un segundo lambda1 seria grid",
}
PUERTA = {
    "momento": "dentro del fold 0, ANTES de los folds 1-4 y ANTES de integrar",
    "G0_G6": "las seis de reports/v34_convergence_gate.md, SIN CAMBIOS y con sus umbrales ledos del "
             "pre-registro de V34 (techo anclado en H(prior)_ES - 0.01; banda de frecuencias anclada "
             "en el prior MEDIDO del ES-train)",
    "G7_L1_activo": {
        "criterio": "escasez efectiva (coordenadas con AL MENOS una clase viva, |z_j| > lambda1) "
                   "mayor que cero en todas las pasadas Y no degradada respecto de la pasada 1 mas "
                   f"allá de una tolerancia declarada de {G7_TOLERANCIA}",
        "anclaje": "la magnitud de pasada 1 medida, con tolerancia nombrada; NO el 0.5 arbitrario "
                   "propuesto antes (reescribia el anclaje) ni el piso 0 (tautologico)",
        "criterio_muerto": "si en pasadas >= 2 la fraccion de coordenadas con |z_j| <= lambda1 fuese "
                           "0, la hipotesis esta analiticamente muerta y se reporta asi, sin culpar "
                           "a la puerta",
        "aplica_a": "solo A1",
    },
    "G8_acotacion_norma": {
        "criterio": f"razon max/min de ||w||_2 entre la primera y la ultima pasada <= "
                    f"{G8_RAZON_MAXIMA}",
        "anclaje": "la HIPOTESIS misma: si el L1 acota la norma, esta no crece. NO un factor "
                   "arbitrario como el 1.5 propuesto antes. La sonda pre-ejecucion ya midio 2.1304 "
                   "con lambda1 = mediana, luego se espera que G8 falle",
        "aplica_a": "solo A1",
    },
    "rama_predeclarada": {
        "G7_o_G8_fallan": "LINEA_ONLINE_CERRADA_POR_MECANISMO: la acotacion queda refutada y la "
                          "familia se cierra DEFINITIVAMENTE (no hay segundo lambda1 ni acta futura "
                          "dentro de este espacio)",
        "G7_y_G8_pasan_pero_G0_G4_fallan": "HIPOTESIS_CONJUNTA_REFUTADA (acota pero no aprende); NO "
                                          "es cierre por conveniencia y se registra con su propia "
                                          "evidencia",
        "todas_pasan": "se evaluan los gates de la serie; PROMISING exigiria segunda particion con "
                       "autorizacion del usuario, que el script no ejecuta",
    },
    "asimetria": "A1 tiene un regularizador (L1) que A2 no tiene: A2 es control de PRESUPUESTO, no "
                 "gemelo mecanico. Ninguna conclusion puede descansar en el contraste A1 vs A2 "
                 "(delta, razones entre brazos e IC bootstrap cruzado quedan contaminados): el "
                 "veredicto se apoya solo en evidencia interna de A1 contra la regla derivada "
                 "(G0/G1/G4) mas el triage",
}
SONDAS_NUEVAS = {
    "k_caso_lambda1_cero": "reproduce dentro del runner el caso lambda1 = 0 del gemelo pre-ejecucion "
                           "(ll_train y ||w||_2 por pasada) y compara con el artefacto versionado",
    "l_caso_lambda1_alpha": "reproduce el caso lambda1 = alpha, que debe ser no-op (mediana/alpha = "
                            "6.8044)",
    "m_caso_lambda1_mediana": "reproduce el caso lambda1 = mediana, con su ||w||_2 y su escasez",
    "politica": "son sondas NO-GATE: no deciden el veredicto, pero la decision de mecanismo NO puede "
                "descansar en evidencia importada de /tmp, luego se reproducen aqui",
}
RAMA_TERMINAL = {
    "miembro_actual": "lambda1 > 0",
    "miembros_cerrados_e_intocables": ["OvR (6 sigmoides)", "power_t (decaimiento)",
                                       "multi-pasada", "lambda2 en {.05, .2, 1, 1e-6}",
                                       "L2 en acumulador vs denominador", "alpha"],
    "segundo_miembro_no_cubierto": "pesos por clase con L2 normalizado fuera de FTRL",
    "destino_del_segundo_miembro": "CERRADO POR DECLARACION si lambda1 > 0 falla, con su evidencia "
                                   "de techo: su geometria (norma acotada como SGD) es la del "
                                   "control sano A2, que ya midio 0.4934789 standalone en D-083 y "
                                   "-0.0017002 integrado con 0/5 folds",
    "re_emision": "una unica re-emision, solo por DEFECTO_DE_IMPLEMENTACION, nunca por cambio de "
                  "hipotesis",
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

    etiquetas, conteo = np.unique(clase, return_counts=True)
    idx_mayor = int(np.argmax(conteo))
    trivial = float(f1_score(clase, np.full_like(clase, etiquetas[idx_mayor]), average="macro",
                             labels=list(range(N_CLASES))))
    if abs(trivial - TRIVIAL_MAYORIA) > 1e-12:
        print("[STOP] el ancla trivial no reproduce la de D-084", flush=True)
        return 1
    print(f"ancla G3 - macro-F1 trivial (clase {cfg.CLASSES[idx_mayor]}): {trivial:.16f}", flush=True)

    sonda = json.loads(SONDA_JSON.read_text(encoding="utf-8"))
    if not sonda["flag_tecnico_1"]["resuelto"]:
        print("[STOP] el flag 1 (mediana > alpha) quedo sin resolver en la sonda previa", flush=True)
        return 1
    if abs(float(sonda["anclaje"]["valor"]) - MEDIANA_Z_ESPERADA) > 1e-12:
        print("[STOP] la mediana medida no coincide con la congelada", flush=True)
        return 1
    print(f"sonda previa: mediana(|z|>0) = {sonda['anclaje']['valor']!r} | mediana/alpha = "
          f"{sonda['anclaje']['mediana_sobre_alpha']:.4f} | razon ||w||_2 con mediana = "
          f"{sonda['flag_tecnico_2']['razon_max_min_lambda1_mediana']:.4f}", flush=True)

    sha_sonda = sha256_archivo(SONDA_SCRIPT)
    sha_sonda_json = sha256_archivo(SONDA_JSON)
    sha_nucleo = sha256_archivo(NUCLEO_V34)
    if sha_nucleo != sonda["sha256_del_nucleo"]:
        print("[STOP] el nucleo FTRL de V34 cambio desde la sonda previa", flush=True)
        return 1
    print(f"nucleo de V34 sin cambios (sha256 {sha_nucleo[:16]})", flush=True)

    hashes = {"sonda_lambda1_script": sha_sonda, "sonda_lambda1_json": sha_sonda_json,
              "nucleo_v34": sha_nucleo, "manifest_canonico": sha256_archivo(CANONICO / "manifest.json")}
    for nombre in ("y_true", "fold_idx", "v30_prob", "ids", "cand_pred", "cand_prob"):
        hashes[f"canonico_{nombre}"] = sha256_archivo(CANONICO / f"{nombre}.npy")
    sha_archivo = sha256_archivo(cfg.TRAIN_FILE)
    textos = train[cfg.TEXT_COL].fillna("").astype(str).to_numpy()
    h = hashlib.sha256()
    for t in textos:
        b = t.encode("utf-8")
        h.update(len(b).to_bytes(8, "little"))
        h.update(b)
    hashes["corpus_archivo"] = sha_archivo
    hashes["corpus_textos"] = h.hexdigest()
    hashes["vectorizador"] = sha256_bytes(json.dumps(VECTORIZADOR, sort_keys=True).encode("utf-8"))

    import sklearn
    versiones = {"sklearn": sklearn.__version__, "numpy": np.__version__,
                 "pandas": pd.__version__}
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT.parent, check=False,
                            capture_output=True, text=True, timeout=30).stdout.strip()

    congelado = {
        "fase": "V35-E1 (via B) pre-registro congelado", "fecha": FECHA, "commit": commit,
        "experimento": "reinicio one-shot de la familia FTRL sobre el miembro NO CUBIERTO "
                       "lambda1 > 0 (control: gemelo SGD de presupuesto)",
        "acta_de_reinicio": "D-086 (se anexa a DECISIONS.md al cierre; su hash se fija ahi)",
        "baseline_primario": round(baseline, 7), "referencia_nested": NESTED,
        "pesos_plugin": PESOS, "hashes": hashes, "versiones": versiones,
        "miembro_no_cubierto": {"identificador": "lambda1 > 0",
                                "no_es_reintento_de_alpha": "alpha* queda CONGELADO e identico al "
                                "de V34-E1; lo que cambia es el espacio de soluciones (umbral duro "
                                "w_j = 0 si |z_j| <= lambda1), no el punto del learning-rate"},
        "miembros_cerrados_e_intocables": RAMA_TERMINAL["miembros_cerrados_e_intocables"],
        "anclaje_lambda1": ANCLAJE, "puerta": PUERTA, "sondas_nuevas": SONDAS_NUEVAS,
        "rama_terminal": RAMA_TERMINAL,
        "sonda_previa": {"script": str(SONDA_SCRIPT.relative_to(ROOT.parent)),
                         "sha256": sha_sonda,
                         "categoria": sonda["categoria"],
                         "anclaje_medio": sonda["anclaje"]["valor"],
                         "mediana_sobre_alpha": sonda["anclaje"]["mediana_sobre_alpha"],
                         "flag_1_resuelto": sonda["flag_tecnico_1"]["resuelto"],
                         "razon_normas_lambda1_mediana":
                         sonda["flag_tecnico_2"]["razon_max_min_lambda1_mediana"],
                         "gemelo_real": [{"nombre": caso["nombre"], "lambda1": caso["lambda1"],
                                          "ll_train_final": caso["ll_train_final"],
                                          "norma_w_l2_final": caso["norma_w_l2_final"]}
                                         for caso in sonda["gemelo_real"]]},
        "alpha_congelado": {"valor_esperado": ALPHA_ESPERADO,
                            "formula": "2*M/(d_norm*P_max**2) con M = ln(5)",
                            "no_se_re_deriva": True},
        "config": {"lambda2": LAMBDA2, "beta": BETA, "power_t": POWER_T, "x0": X0,
                   "P_max": P_MAX, "P_min": P_MIN, "paciencia": PACIENCIA,
                   "mejora_minima": MEJORA_MINIMA, "test_size_es": TEST_SIZE_ES,
                   "particion_es": "StratifiedShuffleSplit(test_size=0.10, "
                                   "random_state=20261010+fold) sobre filas de TRAIN",
                   "semilla": SEMILLA, "hilos": HILOS,
                   "vectorizador": VECTORIZADOR,
                   "ajuste": "POR FOLD: solo ES-train"},
        "puerta_g0_g6_heredada": {
            "G0": f"A1: M/2 <= D_acum <= 2*M; A2: ll_train <= H(prior)_ES - {G1_TECHO_MARGEN}",
            "G1": f"{G1_PISO} <= ll_train <= H(prior)_ES - {G1_TECHO_MARGEN}",
            "G2": f"mejora en >= 1 de las primeras {P_MIN} pasadas (tol {G2_TOLERANCIA})",
            "G3": f"macro-F1 standalone en fold-val >= {G3_MINIMO}",
            "G4": f"frecuencias en [{G4_FACTOR_INFERIOR}*prior_c, {G4_FACTOR_SUPERIOR}*prior_c] del "
                  "prior MEDIDO del ES-train",
            "G5": "determinismo max|delta p| = 0.0 EXACTO",
            "G6": f"filas que suman 1 <= {G6_TOLERANCIA}",
        },
        "gates_de_la_serie": {"baseline_abort": 5e-7, "delta_min": GATE_DELTA,
                              "folds_min": GATE_FOLDS,
                              "bootstrap": {"n": N_BOOTSTRAP, "semilla": SEMILLA},
                              "nota": "solo se evaluan si la puerta pasa; si falla, integracion y "
                                      "bootstrap quedan prohibidos por el pre-registro"},
        "compliance": {"embeddings": 0, "transformers": 0, "llm": 0, "datos_externos": 0, "test": 0,
                       "holdout": 0, "leaderboard": 0, "submission": 0,
                       "ast_audit": "importes Y literales (vowpal, vw, torch, transformers, gensim, "
                                    "fasttext, sentence-transformers, word2vec, glove, "
                                    "spacy-vectors) y 0 rutas de test/holdout/submission/"
                                    "leaderboard",
                       "lecturas": "solo data/raw/age_train.csv y artifacts/p33_scores/todos "
                                   "(intocable, sin escritura)",
                       "auditorias": ["hashes de los 6 .npy + manifest", "corpus archivo y textos",
                                      "vectorizador", "nucleo de V34 sin cambios", ("sonda previa "
                                      "versionada"), "vocabulario e idf por fold", "sondas (a)-(m)",
                                      "tabla de mecanismo por pasada", "determinismo A1 y A2",
                                      "RSS con ru_maxrss", "escritura atomica"]},
        "ejecucion": {"coste_estimado": "~51 s (medido en V34-E1 con la misma maquina y nucleo)",
                      "presupuesto_declarado_seg": SEGUNDOS_DECLARADOS,
                      "ram_tope_gb": RAM_TOPE_GB, "timeout_global_h": LIMITE_GLOBAL_H,
                      "calibracion": f"proyeccion 9 x t_fold0; limite {LIMITE_PROYECTADO_H} h",
                      "checkpoints": "experiments/v35_01_estado.json y checkpoints/v35_e1/",
                      "escritura_atomica": "tmp + os.replace"},
        "entregables": {
            "runner": "scripts/v35_01_ftrl_lambda1.py -> experiments/v35_01_ftrl_lambda1.json, "
                      "experiments/v35_01_estado.json, checkpoints/v35_e1/, "
                      "artifacts/p35_scores/v35_e1/{a1_ftrl_lambda1,a2_sgd_presupuesto}.npy, "
                      "reports/v35_e1_ftrl_lambda1.md, reports/figures/v35_e1_curvas.png, "
                      "reports/v35_novelty_registry.csv, fila anexada en "
                      "reports/age_complete_experiment_history.csv, RUNLOG.md, fila V35-E1 en "
                      "experiments.csv",
            "acta": "D-086 en DECISIONS.md con el miembro NO CUBIERTO, la lista de miembros "
                    "cerrados e intocables, el destino del segundo miembro NO CUBIERTO y la regla "
                    "terminal one-shot",
        },
        "prediccion_registrada": "con la sonda pre-ejecucion ya falsada la hipotesis de acotacion "
                                 "(razon ||w||_2 = 2.1304 con lambda1 = mediana), el veredicto mas "
                                 "probable es LINEA_ONLINE_CERRADA_POR_MECANISMO por fallo de G8. "
                                 "El ciclo NO puede terminar en PROMISING: alpha esta congelado e "
                                 "identico al que ya divergio",
    }

    lineas = [
        "# V35 - E1 (via B) - Pre-registro congelado: `lambda1 > 0` sobre el nucleo FTRL de V34",
        "",
        f"**Fecha**: {FECHA}. **Estado**: CONGELADO antes de ejecutar (commit `{commit[:7]}`).",
        (f"**Baseline recomputado**: {baseline:.7f} (nested {NESTED}); ancla trivial {trivial:.16f} "
        f"(clase {cfg.CLASSES[idx_mayor]})."),
        "",
        "## Que es y que NO es",
        "",
        ("- **Acta de reinicio**: **D-086**, anexada a `DECISIONS.md` al cierre. Miembro NO CUBIERTO: "
        f"`{RAMA_TERMINAL['miembro_actual']}`."),
        ("- **No es un reintento de `alpha`**: `alpha*` queda CONGELADO e identico al de V34-E1 "
        f"({ALPHA_ESPERADO!r}). Lo que cambia es el espacio de soluciones."),
        f"- **Miembros cerrados e INTOCABLES**: {'; '.join(RAMA_TERMINAL['miembros_cerrados_e_intocables'])}.",
        "",
        "## Anclaje de lambda1 (derivado, cero grid)",
        "",
        (f"- `lambda1 := mediana(|z_j| > 0)` al final de la pasada 1 sobre **TRAIN** del fold 0 = "
        f"**{MEDIANA_Z_ESPERADA!r}** (medido; `alpha*` = {ALPHA_ESPERADO!r}, luego "
        f"mediana/alpha = {RAZON_MEDIANA_ALPHA:.4f})."),
        ("- **Flag 1 resuelto**: `lambda1 = alpha` es no-op precisamente porque la mediana supera a "
        "`alpha` por un factor 6.80."),
        (f"- **Flag 2 resuelto en negativo**: `lambda1` fijo **no acota** `||w||_2` (razon max/min = "
        f"{sonda['flag_tecnico_2']['razon_max_min_lambda1_mediana']:.4f} con la mediana), luego la "
        "hipotesis de acotacion queda **refutada por medicion**."),
        "- Prohibido re-derivar `lambda1` tras ver las gates: un segundo `lambda1` seria grid.",
        "",
        "## Puerta",
        "",
        ("- **G0-G6**: las de V34 **sin cambios** (techo anclado en `H(prior)_ES` - 0.01; banda de "
        "frecuencias en `[0.5*prior_c, 1.5*prior_c]` del prior **medido**)."),
        (f"- **G7 (L1 activo)**: escasez efectiva > 0 en todas las pasadas y no degradada mas alla de "
        f"una tolerancia declarada de {G7_TOLERANCIA} respecto de la pasada 1. Anclada a la magnitud "
        "medida (NO al 0.5 arbitrario, NO al piso 0 tautologico). **Solo A1**."),
        (f"- **G8 (acotacion)**: razon max/min de `||w||_2` <= **{G8_RAZON_MAXIMA}**, anclada a la "
        "hipotesis misma (acotar implica no crecer), no a un factor arbitrario. **Solo A1**."),
        ("- **Rama pre-declarada**: G7 o G8 fallan -> `LINEA_ONLINE_CERRADA_POR_MECANISMO` (cierre "
        "DEFINITIVO de la familia); G7 y G8 pasan y G0-G4 fallan -> `HIPOTESIS_CONJUNTA_REFUTADA` "
        "(acota pero no aprende; no es cierre por conveniencia)."),
        ("- **Asimetria declarada**: A2 es control de **PRESUPUESTO**, no gemelo mecanico; ninguna "
        "conclusion puede descansar en el contraste A1 vs A2 (queda contaminado)."),
        "",
        "## Sondas nuevas (no-gate) dentro del runner",
        "",
        ("- (k) caso `lambda1 = 0`, (l) caso `lambda1 = alpha`, (m) caso `lambda1 = mediana`: se "
        "reproducen aqui los tres casos del gemelo, porque la decision de mecanismo no puede "
        "descansar en evidencia importada de `/tmp`."),
        "",
        "## Regla terminal one-shot",
        "",
        (f"- Si `lambda1 > 0` falla, el segundo miembro NO CUBIERTO ({RAMA_TERMINAL['segundo_miembro_no_cubierto']}) "
        f"queda {RAMA_TERMINAL['destino_del_segundo_miembro'].split(':')[0].lower()}."),
        "- Una unica re-emision, solo por `DEFECTO_DE_IMPLEMENTACION`.",
        "",
        "## Prediccion registrada antes de ejecutar",
        "",
        f"- {congelado['prediccion_registrada']}",
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
        ("- 0 embeddings, 0 transformers, 0 LLM, 0 datos externos; sin test, sin holdout, sin "
        "leaderboard, sin submission; un script nuevo; NO se edita ningun script ya ejecutado; "
        "`artifacts/p33_scores/todos` intocable."),
        ("- AST audit de imports y literales; lecturas limitadas al corpus y a las particiones "
        "reservadas."),
        (f"- Coste esperado ~51 s con presupuesto declarado de {SEGUNDOS_DECLARADOS:g} s; RSS con "
        f"tope {RAM_TOPE_GB:g} GB medido con `ru_maxrss`; checkpoints por fold y por brazo con "
        "escritura atomica."),
    ]
    texto = "\n".join(lin.rstrip() for lin in lineas) + "\n"
    PREREG_MD.write_text(texto, encoding="utf-8")
    print(f"  {PREREG_MD.name}: {len(texto.splitlines())} lineas", flush=True)

    PREREG_JSON.write_text(json.dumps(congelado, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  {PREREG_JSON.name}", flush=True)

    if "Pre-registro de V35-E1" not in (ROOT / "RUNLOG.md").read_text(encoding="utf-8"):
        with (ROOT / "RUNLOG.md").open("a", encoding="utf-8") as f:
            f.write("\n---\n\n## Pre-registro de V35-E1 (via B: lambda1 > 0) · " + FECHA + "\n\n"
                    "- Reinicio one-shot de la familia FTRL sobre el miembro NO CUBIERTO "
                    "`lambda1 > 0`; acta D-086. `alpha*` CONGELADO e identico a V34-E1.\n"
                    f"- Anclaje derivado: mediana(|z|>0) = {MEDIANA_Z_ESPERADA!r} "
                    f"(mediana/alpha = {RAZON_MEDIANA_ALPHA:.4f}); flag 1 resuelto.\n"
                    f"- Flag 2 resuelto en negativo: `lambda1` fijo NO acota `||w||_2` (razon "
                    f"{sonda['flag_tecnico_2']['razon_max_min_lambda1_mediana']:.4f}), luego la "
                    "hipotesis de acotacion queda falsada por medicion.\n"
                    "- Puerta G0-G6 sin cambios + G7 (L1 activo, anclada a la magnitud medida) y G8 "
                    "(acotacion, anclada a la hipotesis), ambas solo sobre A1; rama pre-declarada; "
                    "sondas (k)-(m) no-gate.\n"
                    f"- Baseline recomputado {baseline:.7f}; sin test, sin holdout, sin submission, "
                    "campeon intacto.\n")
        print("  RUNLOG.md: pre-registro anotado", flush=True)

    ruta_exp = ROOT / "experiments.csv"
    lineas_exp = ruta_exp.read_text(encoding="utf-8").splitlines()
    cols = [c.strip() for c in lineas_exp[0].split(",")]
    if "V35-E1-PREREG" not in {lin.split(",", 1)[0] for lin in lineas_exp[1:] if lin}:
        fila = {"experiment_id": "V35-E1-PREREG", "fecha": FECHA, "bloque": "35.0",
                "etapa": "V35-E1 (via B) pre-registro congelado: lambda1 > 0 sobre el nucleo de V34",
                "pregunta": "el umbral L1 acota la norma de FTRL y elimina la divergencia?",
                "hipotesis": "si: w_j = 0 si |z_j| <= lambda1 acota la norma, que es la causa "
                             "medida de la divergencia",
                "features": "TF-IDF por fold (receta v31_02) ajustada en ES-train; 0 embeddings",
                "preprocessing": "vocabulario e idf ajustados SOLO en ES-train",
                "representacion": "TF-IDF dispersa identica a V34-E1 (misma receta, hashes nuevos)",
                "modelo": "FTRL-Proximal softmax K=6 con L1 (A1) / SGD log_loss partial_fit x 10 "
                          "(A2, control de PRESUPUESTO)",
                "hiperparametros": f"lambda1 = mediana(|z|>0) = {MEDIANA_Z_ESPERADA!r} (DERIVADO); "
                                   f"alpha* = {ALPHA_ESPERADO!r} CONGELADO; lambda2 1, beta 1, "
                                   "power_t 0, P_max 10, P_min 3, paciencia 2",
                "validacion": "puerta G0-G6 sin cambios + G7 (L1 activo) + G8 (acotacion), solo A1; "
                              "rama pre-declarada; sondas (a)-(m)",
                "macro_f1_mean": "", "macro_f1_std": "", "weighted_f1_mean": "",
                "micro_f1_mean": "", "accuracy_mean": "",
                "resultado": "pre-registro escrito y congelado; sin resultados aun",
                "interpretacion": "la sonda pre-ejecucion ya refuto la hipotesis de acotacion "
                                  "(razon ||w||_2 = 2.1304 con lambda1 = mediana)",
                "decision": "ejecutar V35-E1 tal cual",
                "duracion_seg": "0.0",
                "artefactos": "reports/v35_preregistro.md | "
                              "experiments/v35_00_preregistro.json | reports/v35_sonda_lambda1.md"}
        buf = io.StringIO()
        csv.writer(buf, lineterminator="\n").writerow([fila[c] for c in cols])
        with ruta_exp.open("a", encoding="utf-8") as f:
            f.write(buf.getvalue())
        print("  experiments.csv: fila V35-E1-PREREG", flush=True)

    print(f"\n[fin] {time.time() - inicio:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
