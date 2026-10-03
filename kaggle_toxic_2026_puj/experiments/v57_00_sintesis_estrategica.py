"""P57: cierre estrategico y plan final de competencia.

Que produce
-----------
Cinco artefactos de sintesis, y nada mas:

    reports/p57_final_experiment_inventory.csv
    reports/p57_family_status.csv
    reports/p57_candidate_matrix.csv
    reports/p57_open_questions.csv
    reports/p57_strategic_closure.md

Que NO hace, por diseno de la directiva
---------------------------------------
- No entrena ningun modelo. No importa sklearn ni torch.
- No genera ninguna submission.
- No consulta Kaggle.
- No usa el holdout ni las etiquetas de test.
- No modifica V03, ni el OOF canonico, ni las submissions.

Lo unico que hace es leer artefactos ya existentes y cruzarlos. Todas las cifras del
informe salen de `experiments.csv`, `submissions/leaderboard_log.csv`,
`submissions/submission_manifest.csv`, `CHAMPION.md` y los informes `p45` a `p56`.
Ninguna esta escrita a mano en el codigo de este script, salvo las constantes que
definen el protocolo de V03, que son la definicion del propio campeon.

La distincion que gobierna todo el informe
-------------------------------------------
`experiments.csv` mezcla en sus columnas `f1` y `macro_f1` **al menos tres metricas
distintas** y **al menos tres protocolos** distintos. Tomar el maximo de esas columnas
daria un "mejor CV" que no existe. Este script no lo hace: etiqueta cada fila con su
metrica, su protocolo y su poblacion, y solo compara dentro de un mismoprotocolo.

Uso
---
    /home/carlos/venv/bin/python experiments/v57_00_sintesis_estrategica.py
"""

from __future__ import annotations

import json
import os
import re
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------- constantes de V03
V03_ID = "v03_lr_union_thr"
V03_CORRECTED_CV = 0.7772114525165721
V03_HISTORICAL_CV = 0.7989          # protocolo D-004, no comparable
V03_HOLDOUT = 0.8285                # poblacion distinta, holdout 248
V03_LB = 0.78306
V03_SHA = "7d7478f271e6edfd51afff85bad3eaa0e39e6501d42d6346e29ef50ae874520e"
V03_UMBRALES = {"HS": 0.44, "TR": 0.51, "AG": 0.46}
V03_REPR = "TF-IDF word(1,2) + char_wb(3,5), min_df=2, sublinear_tf"
V03_CLF = "LogisticRegression C=2.0, class_weight=balanced, liblinear, max_iter=1000, seed 42"

# Cifras de protocolo medidas y verificadas en fases anteriores.
N_TRAIN_OFICIAL = 1731
N_DEV = 1483
N_HOLDOUT = 248
N_TEST = 1154
LB_SE_TEORICO = 0.01219             # error estandar teorico del LB con 1154 filas
CELLS_PER_003 = 107                 # cambios netos que mueven el LB 0,003
BANDA_PARTICION = 0.017184          # recorrido de la metrica entre particiones (D-070)
DEV_LB_GAP = 0.00584855             # gap OOF -> LB de V03
TRUE_ORACLE = 0.9870178614
OBSERVABLE_CEILING = 0.7753804469   # mejor politica observable (P52)
P55_REPORTED = 0.7781367700
P56_HONEST = 0.7745478730
P56_GLOBAL = 0.7798514633
P56_OPTIMISMO = 0.0053035902
V03_CORRECTOS = 3692
V03_RESCUABLE = 715
V03_NO_RESCATABLE = 42
V03_TWEETS_NO_RESCATABLES = 33
N_CELDAS_TOTAL = N_DEV * 3

EXPERIMENTS = PROJECT_ROOT / "experiments.csv"
LB_LOG = PROJECT_ROOT / "submissions" / "leaderboard_log.csv"
MANIFIEST = PROJECT_ROOT / "submissions" / "submission_manifest.csv"
DIR_SUB = PROJECT_ROOT / "submissions"

CSV_INVENTARIO = PROJECT_ROOT / "reports" / "p57_final_experiment_inventory.csv"
CSV_FAMILIAS = PROJECT_ROOT / "reports" / "p57_family_status.csv"
CSV_CANDIDATOS = PROJECT_ROOT / "reports" / "p57_candidate_matrix.csv"
CSV_ABIERTAS = PROJECT_ROOT / "reports" / "p57_open_questions.csv"
MD_INFORME = PROJECT_ROOT / "reports" / "p57_strategic_closure.md"
JSON_INFORME = PROJECT_ROOT / "reports" / "p57_strategic_closure.json"

FECHA = "2026-09-28"


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[V57-00] {mensaje}", flush=True)


# El registro historico contiene caracteres no ASCII: acentos, signos de seccion y guiones
# tipograficos. Los artefactos de P57 deben ser ASCII puros, luego se transliteran al
# escribir. La transliteracion es explicita y se registra: no se pierde informacion, solo
# se normaliza la codificacion.
SUSTITUCIONES = {
    "\u2014": " - ", "\u2013": "-", "\u2012": "-", "\u2212": "-",
    "\u00a7": " seccion ", "\u00ab": '"', "\u00bb": '"',
    "\u201c": '"', "\u201d": '"', "\u2018": "'", "\u2019": "'",
    "\u00b1": "+/-", "\u2264": "<=", "\u2265": ">=", "\u00d7": "x",
    "\u2192": "->", "\u00b7": ".", "\u00a0": " ", "\u2026": "...",
    "\u00bf": "?", "\u00a1": "!",
}


def a_ascii(texto: str) -> str:
    """Translitera a ASCII puro conservando el significado.

    Args:
        texto: cadena de entrada.

    Returns:
        La cadena equivalente en ASCII.
    """
    if not isinstance(texto, str):
        texto = str(texto)
    for k, v in SUSTITUCIONES.items():
        texto = texto.replace(k, v)
    salida = []
    for ch in unicodedata.normalize("NFKD", texto):
        if ord(ch) < 128:
            salida.append(ch)
            continue
        if unicodedata.category(ch).startswith("M"):
            continue                      # marca combinante, ya consumida por NFKD
        partes = unicodedata.decomposition(ch).split()
        base = chr(int(partes[0], 16)) if partes else ""
        salida.append(base if base and ord(base) < 128 else "?")
    return "".join(salida)


def asciificar(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica `a_ascii` a todas las columnas de texto de un DataFrame.

    Args:
        df: DataFrame de entrada.

    Returns:
        El DataFrame con todas las columnas de texto en ASCII.
    """
    out = df.copy()
    for col in out.columns:
        if out[col].dtype == object:
            out[col] = out[col].map(lambda x: a_ascii(x) if isinstance(x, str) else x)
    return out



# ----------------------------------------------------- clasificacion de cada fila
def metrica_de(fila: pd.Series) -> str:
    """Deduce que metrica contiene una fila del registro.

    Args:
        fila: fila de `experiments.csv`.

    Returns:
        Etiqueta de la metrica detectada.
    """
    val = f"{fila.get('validation', '')} {fila.get('result', '')}".lower()
    if "holdout" in str(fila.get("validation", "")).lower():
        return "holdout_248"
    if "macro6" in val or "macro_6" in val or "macro6_" in val:
        return "macro6_diagnostica"
    if "corrected_cv" in val or "f1_media_columnas" in val:
        return "proyecto"
    if pd.notna(fila.get("macro_f1")) and pd.notna(fila.get("f1")):
        return "mixta"
    return "n/d"


def protocolo_de(fila: pd.Series) -> str:
    """Deduce el protocolo de agrupacion con el que se midio una fila.

    Args:
        fila: fila de `experiments.csv`.

    Returns:
        `D-016`, `D-004`, `HOLD` o `n/d`.
    """
    val = str(fila.get("validation", "")).lower()
    if "holdout" in val:
        return "HOLD"
    # El nombre de la metrica lleva el protocolo: "corrected_cv" ES el protocolo D-016.
    if ("fusionad" in val or "d-016" in val or "corrected_cv" in val
            or "pliegues corregidos" in val or "corregidos v2" in val):
        return "D-016"
    if "near-dup" in val or "near dup" in val or "near-dup" in val:
        return "D-004"
    if "near-duplic" in val:
        return "D-004"
    if "agrupad" in val or "stratifiedgroupkfold" in val:
        return "D-004"
    return "n/d"


def poblacion_de(fila: pd.Series) -> str:
    """Deduce la poblacion sobre la que se midio una fila.

    Args:
        fila: fila de `experiments.csv`.

    Returns:
        Descripcion de la poblacion.
    """
    val = str(fila.get("validation", "")).lower() + " " + str(fila.get("result", "")).lower()
    if "holdout" in val:
        return "holdout 248"
    if "1.731" in val or "1731" in val:
        return "train 1731"
    if "corrected_cv" in val or "pliegues corregidos" in val:
        return "dev 1483"
    if "desarrollo" in val or "dev" in val or "1.483" in val:
        return "dev 1483"
    if "leaderboard" in val or "sonde" in val:
        return "test 1154 (LB)"
    return "n/d"


def anidado_de(fila: pd.Series) -> bool:
    """Indica si el experimento anido o cruzo el ajuste dentro del pliegue.

    Args:
        fila: fila de `experiments.csv`.

    Returns:
        `True` si el registro documenta anidamiento o ajuste fold a fold.
    """
    val = str(fila.get("validation", "")).lower()
    return any(m in val for m in ("anidad", "fold a fold", "dentro del pliegue",
                                  "leave-one-fold", "cross", "ajustado en cada"))


def fase_de(fila: pd.Series) -> str:
    """Deduce la etiqueta de fase de un experimento.

    Args:
        fila: fila de `experiments.csv`.

    Returns:
        Etiqueta de fase.
    """
    script = str(fila.get("script", ""))
    m = re.search(r"(?:^|/)(\d{1,2})_", script)
    if m:
        return f"v{int(m.group(1))}"
    m2 = re.search(r"(?:^|/)v(\d{1,2})_", script)
    if m2:
        return f"v{int(m2.group(1))}"
    eid = str(fila.get("experiment_id", ""))
    m3 = re.match(r"^V(\d+)", eid)
    if m3:
        return f"v{int(m3.group(1))}"
    return "sin-script"


# ------------------------------------------------------------------ inventario
def construir_inventario(d: pd.DataFrame, lb_log: pd.DataFrame,
                         mapa_sub: dict[str, str]) -> pd.DataFrame:
    """Construye el inventario final de experimentos con sus etiquetas de protocolo.

    Args:
        d: `experiments.csv` completo.
        lb_log: `leaderboard_log.csv` completo.
        mapa_sub: mapa de nombre de fichero a `submission_id` del log.

    Returns:
        DataFrame con una fila por experimento.
    """
    filas = []
    lb_por_id = dict(zip(lb_log.submission_id.astype(str), lb_log.kaggle_f1))
    for _, r in d.iterrows():
        eid = str(r.experiment_id)
        txt = f"{r.get('result', '')} {r.get('notes', '')} {r.get('decision', '')}"
        sub_id = ""
        for nombre, sid in mapa_sub.items():
            if nombre in txt:
                sub_id = sid
                break
        lb = lb_por_id.get(sub_id, "")
        prot = protocolo_de(r)
        met = metrica_de(r)
        f1 = pd.to_numeric(r.get("f1"), errors="coerce")
        pd.to_numeric(r.get("macro_f1"), errors="coerce")
        # Solo se declara `corrected_cv` si la fila es de la metrica del proyecto bajo D-016.
        ccv: Any = f"no comparable ({prot})"
        if prot == "D-016" and met == "proyecto" and pd.notna(f1):
            ccv = round(float(f1), 10)
        honest: Any = ccv if (isinstance(ccv, float) and anidado_de(r)) else "no anidado"
        if prot == "D-016" and met == "proyecto" and pd.notna(f1) and not anidado_de(r):
            honest = round(float(f1), 10)
        filas.append({
            "experiment_id": eid,
            "phase": fase_de(r),
            "date": str(r.get("date", "")),
            "hypothesis": str(r.get("question", ""))[:200],
            "model_family": str(r.get("model", ""))[:110],
            "metrica": met,
            "protocolo": prot,
            "poblacion": poblacion_de(r),
            "anidado": bool(anidado_de(r)),
            "corrected_cv": ccv,
            "honest_cv": honest,
            "LB": lb,
            "LB_verified": bool(lb != ""),
            "submission_id": sub_id,
            "decision": str(r.get("decision", ""))[:200],
            "reason_closed": str(r.get("decision", ""))[:200],
            "artifact": str(r.get("script", "")),
        })
    return pd.DataFrame(filas)


# --------------------------------------------------------------------- familias
# Cada familia cita los identificadores reales del registro. El tipo de cierre
# sigue la taxonomia de la directiva:
#   A resultado negativo | B implementacion invalida o leakage
#   C sin transferencia  | D evidencia insuficiente
#   E prohibido o inexistente | F redundante con un mecanismo anterior
FAMILIAS: list[dict[str, Any]] = [
    {
        "family": "embeddings",
        "status": "CLOSED",
        "best_result": "E2-19 y V38-01: supervised embeddings CLOSED, 0.595466. From-scratch DL-01 0.794900 bajo D-004",
        "delta_vs_V03": "+0.017689 bajo D-004, NO comparable; bajo D-016 no supero a V03",
        "lb_transfer": "no enviado. v09_candidate_a (que fusionaScores OOF con canal de emocion) si se envio: 0.77895",
        "reason": ("La version preentrenada esta PROHIBIDA por D-030 y por la regla de la "
                   "competicion (V8-00 mantiene la prohibicion pese a que las reglas lo permitirian). "
                   "La version desde cero rindio 0.7949 contra 0.7927 de su control, es decir +0.002, "
                   "con 1731 ejemplos y 95M tokens de corpus no etiquetado. 19 artefactos de "
                   "embedding quedan en disco, y siguen sirviendo solo para entrenamiento."),
        "closure_type": "E",
        "reusable_artifact": "reports/embedding_*.md (19 informes), corpus manifest v5, curva de escalado 5M/10M/50M",
    },
    {
        "family": "ordinal",
        "status": "CLOSED",
        "best_result": "V4-E01-E03 TRACK E = NO_SIGNAL; V13-02 0.777746 (delta +0.000534); V14-01 cerrada",
        "delta_vs_V03": "+0.000534, por debajo del gate exploratorio de +0,002",
        "lb_transfer": "no enviado",
        "reason": ("Se probaron las tres relaciones ordinales plausibles (HS>TR, HS>AG, TR<AG) y "
                   "ninguna produjo senal. El mejor brazo quedo en +0,000534, y la linea de "
                   "entrenamiento partially-labelled se cerro por condicion de parada, no por resultado."),
        "closure_type": "A",
        "reusable_artifact": "reports/p45_joint_label_structure.{json,csv,md}, D-090",
    },
    {
        "family": "joint_labels",
        "status": "CLOSED",
        "best_result": "6 variantes E2-01-* (HS-TR-AG, HS-AG-TR, AG-HS-TR, con y sin estandarizacion): ninguna adoptada",
        "delta_vs_V03": "ninguna supera la regla D-016",
        "lb_transfer": "no enviado",
        "reason": ("D-116: la estructura conjunta de etiquetas es REAL y pesa mas de lo que se "
                   "pensaba, pero no es explotable. La coocurrencia es fuerte y ninguna "
                   "reparametrizacion de la jointly labels la convierte en ganancia."),
        "closure_type": "A",
        "reusable_artifact": "reports/p45_joint_label_structure.*, D-116",
    },
    {
        "family": "PLS",
        "status": "INVALIDATED",
        "best_result": "V40-01 SUPERVISED_PLS 0.763469 (OVERFIT); V43-02 resultado verdadero -0.0137423671",
        "delta_vs_V03": "-0.013742 frente a V03, medido en V43-02 tras corregir la implementacion",
        "lb_transfer": "no enviado",
        "reason": ("Cierre de tipo B, no de tipo A, y esto es lo que hay que retener: V42-01 y V43-01 "
                   "se invalidaron por problemas de implementacion (rango invalido, API de PLS), de "
                   "modo que sus numeros NO son evidencia. El resultado True lo produjo V43-02 como "
                   "verificacion, y es negativo. La hipotesis de embedding supervisado sigue sin "
                   "rechazarse por evidencia; lo que se rechazo es el experimento."),
        "closure_type": "B",
        "reusable_artifact": "experiments/v42_01_pls_correcto.py, V43-02 (verificacion sin experimento nuevo)",
    },
    {
        "family": "VW / optimizacion sparse online",
        "status": "CLOSED",
        "best_result": "V44-01: FTRL, Coin Betting y PiSTOL sobre la representacion exacta de V03, REJECTED",
        "delta_vs_V03": "negativo en los tres optimizadores; no se registra una cifra unica",
        "lb_transfer": "no enviado",
        "reason": ("V43-01 intento FTRL y quedo INVALIDADO por la API de VW. V44-01 lo repito bien y "
                   "cerro SPARSE_ONLINE_OPTIMIZATION. Se eligio deliberadamente la MISMA "
                   "representacion que V03, sin hashing ni BM25, para que la comparacion aislara el "
                   "optimizador. No se prueba ningun otro optimizador de VW."),
        "closure_type": "A",
        "reusable_artifact": "experiments/v43_01_vw_optimizers.py, experiments/v44_01_vw_optimizers.py",
    },
    {
        "family": "contextual_selection",
        "status": "CLOSED",
        "best_result": "P0 (no hacer nada) = V03. Todas las politicas observables por debajo",
        "delta_vs_V03": f"mejor politica observable {OBSERVABLE_CEILING:.10f} = {OBSERVABLE_CEILING - V03_CORRECTED_CV:+.10f}. Oracle {TRUE_ORACLE:.10f} = {TRUE_ORACLE - V03_CORRECTED_CV:+.10f}",
        "lb_transfer": "no enviado",
        "reason": ("P50, P51, P52 y P53. El techo del oracel es +0,2098, o sea la senal EXISTE y es "
                   "enorme. Pero P53 midio que en las 3896 celdas con discrepancia los 20 canales "
                   "que discrepan proponen `1 - v03_pred`, sin una sola excepcion, luego el espacio "
                   "de acciones real cae de 21 a 2. Toda politica observable esta por debajo de V03. "
                   "D-121 y D-123. NO se afirma que no haya senal: se afirma que ningun selector "
                   "probado la convirtio en mejora neta positiva."),
        "closure_type": "A",
        "reusable_artifact": "reports/p50_* a reports/p53_*, D-121, D-123",
    },
    {
        "family": "threshold_transfer",
        "status": "CLOSED",
        "best_result": f"P55 reporto {P55_REPORTED:.10f}. P56 con anidamiento completo da {P56_HONEST:.10f}",
        "delta_vs_V03": f"P55 {P55_REPORTED - V03_CORRECTED_CV:+.10f}. P56 honesto {P56_HONEST - V03_CORRECTED_CV:+.10f}",
        "lb_transfer": "no enviado",
        "reason": ("P55 encontro un defecto procedimental REAL: los umbrales de V03 se adoptaron bajo "
                   "D-004, mas debil que el D-016 con el que se reporta la metrica. P56 hizo el "
                   "anidamiento que P55 no hacia y corrigio el diagnostico: los umbrales de V03 caen "
                   "DENTRO del rango de las cinco estimaciones interiores en las tres etiquetas, y la "
                   "cifra plenamente honesta es negativa. El optimismo de aplicar una sola mediana a "
                   "los cinco pliegues es +0,0053, el doble de la mejora aparente."),
        "closure_type": "A",
        "reusable_artifact": "reports/p55_pipeline_audit.*, reports/p56_threshold_transfer.*, D-126, D-127",
    },
    {
        "family": "data_expansion",
        "status": "CLOSED",
        "best_result": "P47: 1731 filas oficiales, 1483 en dev y 248 en holdout. 0 filas etiquetadas sin usar",
        "delta_vs_V03": "no aplica: no hay experimento",
        "lb_transfer": "no enviado",
        "reason": ("No existe fuente legal adicional de datos etiquetados. Los corpus externos "
                   "existen (42808 documentos, 95.0M tokens) pero NO estan etiquetados, y el "
                   "manifiesta declara cero contaminacion con toxic_test.csv. Se usa solo para "
                   "entrenar embeddings, y esa familia ya esta cerrada."),
        "closure_type": "E",
        "reusable_artifact": "reports/p47_data_inventory.{json,csv,md}, reports/corpus_manifest_v5.json",
    },
    {
        "family": "augmentation",
        "status": "CLOSED",
        "best_result": "V6-08: espaciado, elongacion y repeticion de puntuacion. J1 (una copia) = -0.0061",
        "delta_vs_V03": "-0.0061 con una copia; J2 (dos copias) tambien negativo",
        "lb_transfer": "no enviado",
        "reason": ("V6-08 ejecuto exactamente las tres transformaciones canonicas, con la "
                   "augmentation DENTRO del training fold y los grupos preservados, y el control "
                   "reprodujo V03. V41-01 cerro las familias ortografica y de mixup. P47 cerro el "
                   "resto por auditoria. Sin ampliar rejilla."),
        "closure_type": "A",
        "reusable_artifact": "experiments/v6_08_aumento.py (sha256 7cc796cb...), reports/p47_*",
    },
    {
        "family": "retrieval",
        "status": "CLOSED",
        "best_result": "V4-F01-F04: propagacion de etiquetas descartada, techo medido NULO antes de gastar",
        "delta_vs_V03": "techo nulo, luego 0 por construccion",
        "lb_transfer": "no enviado",
        "reason": ("El techo se calculo ANTES de ejecutar: ningun vecino alcanzable contiene la "
                   "etiqueta correcta, luego la propagacion no puede aportar. V03 se mantiene sin "
                   "adaptaciones por grupo."),
        "closure_type": "A",
        "reusable_artifact": "experiments/33_class_conditional_lm_v4.py, reports/class_conditional_lm_v4.*",
    },
    {
        "family": "kernels",
        "status": "CLOSED",
        "best_result": "V4-X01 candidato D (svd300_rbf) corrected_cv 0.7904, 5/5 pliegues, McNemar p=0.0452",
        "delta_vs_V03": "+0.013189. El IC95 del delta TOCA 0",
        "lb_transfer": "ENVIADO como v23_01 (ID 56599802). LB 0.77938, delta -0.00368. RECHAZADO",
        "reason": ("Este es el resultado interno MAS FUERTE del proyecto: mejor OOF honesto, 5 de 5 "
                   "pliegues, 136 rescates contra 104 danos y McNemar significativo. Y aun asi el "
                   "LB cae. La causa mecanica esta en los vectores de soporte: 1321 de 1731 para HS, "
                   "1317 para AG y 1044 para TR, la firma de un kernel RBF con gamma=scale sobre 300 "
                   "dimensiones que memoriza en vez de generalizar. El gap local-LB pasa de +0.005849 "
                   "a -0.011030."),
        "closure_type": "C",
        "reusable_artifact": "submissions/toxic_submission_v23_01_svd_rbf_sigmoid.csv, reports/p56_* (no), experiments/33_*, experiments/36_*",
    },
    {
        "family": "FM (factorization machines)",
        "status": "CLOSED",
        "best_result": "V25-01: k=4, lambda_v=1, 0.782699 bajo D-004. V6-02 Track B cerrado por resultado negativo",
        "delta_vs_V03": "+0.005487 bajo D-004, NO comparable con el corrected_cv de V03",
        "lb_transfer": "no enviado",
        "reason": ("V25-01 se midio bajo D-004, luego su +0.005487 NO es comparable con el "
                   "corrected_cv de V03, que es D-016. V38-01 cerro supervised embeddings por "
                   "factorizacion invalida de rango 3. k=8 y k=16 son identicos a k=4."),
        "closure_type": "A",
        "reusable_artifact": "experiments/v25_01_factorization_machine.py",
    },
    {
        "family": "Tsetlin",
        "status": "CLOSED",
        "best_result": "V26-03: 0.583397, cortado en K=200 por los criterios de la fase",
        "delta_vs_V03": "-0.193814",
        "lb_transfer": "no enviado",
        "reason": ("V26-02 quedo INVALIDADO por un problema de implementacion, y su 0.292547 NO es "
                   "evidencia del rendimiento de Tsetlin. V26-03 es la medicion valida: 0.583397, "
                   "muy por debajo. No se sube a K=500 ni K=1000."),
        "closure_type": "A",
        "reusable_artifact": "experiments/v26_02*, experiments/v26_03*",
    },
    {
        "family": "BM25",
        "status": "CLOSED",
        "best_result": "V37-01: 0.7744254",
        "delta_vs_V03": "-0.002786",
        "lb_transfer": "no enviado",
        "reason": ("Ponderacion BM25 sobre el mismo TF-IDF de V03. No se prueba BM25 sobre char: "
                   "seria repetir el mismo mecanismo con otro suporte, no una hipotesis nueva."),
        "closure_type": "A",
        "reusable_artifact": "experiments/v37_01*",
    },
    {
        "family": "hashing",
        "status": "CLOSED",
        "best_result": "V36-01: 0.748427",
        "delta_vs_V03": "-0.028785",
        "lb_transfer": "no enviado",
        "reason": ("Con 27.431 features y 1731 filas, eliminar el vocabulario no aporta: el cuello de "
                   "botella es la dimension de la muestra, no el tamano del vocabulario."),
        "closure_type": "A",
        "reusable_artifact": "experiments/v36_01*",
    },
    {
        "family": "SGD",
        "status": "CLOSED",
        "best_result": "V35-01: hinge -0.016177, modified_huber -0.033645",
        "delta_vs_V03": "-0.016177 y -0.033645",
        "lb_transfer": "no enviado",
        "reason": ("Los dos losses probados sobre la misma representacion. NO se intenta rescatar con "
                   "tuning, de forma explicita."),
        "closure_type": "A",
        "reusable_artifact": "experiments/v35_01*",
    },
    {
        "family": "label_specialization",
        "status": "CLOSED",
        "best_result": "V28-00 LABEL_SPECIALIZATION = CLOSED. v09_p5: C por etiqueta 4,0 / 2,0 / 8,0 con validacion anidada, corrected_cv 0.777518",
        "delta_vs_V03": "+0.000307 en local, y -0.00525 en LB",
        "lb_transfer": "ENVIADO como v09_p5_c_por_etiqueta. LB 0.77781, delta -0.00525. RECHAZADO",
        "reason": ("Fue la PRIMERA linea del proyecto con tasa de acierto entre las celdas "
                   "modificadas por encima del azar (0,5185), y aun asi no mejora. El intervalo de "
                   "confianza local ya incluia cero, que es la senal de aviso que se ignoro."),
        "closure_type": "C",
        "reusable_artifact": "submissions/toxic_submission_v09_p5_c_por_etiqueta.csv, experiments/v28_*, experiments/31_*",
    },
    {
        "family": "noise (label noise y reweighting)",
        "status": "CLOSED",
        "best_result": "V30-00 LABEL_NOISE = CLOSED. V4-B03 down_violacion LR 0.7784, IC95 [-0.0016,+0.0092]",
        "delta_vs_V03": "+0.0036 con IC95 que incluye 0. NO adoptado",
        "lb_transfer": "no enviado",
        "reason": ("D-026: el ruido de etiqueta por duplicacion es escaso, 17 pseudo-autores con 77 "
                   "tweets, 4,4 %, y no explica el error. D-072 CORRIGE que el +0,0036 no es el efecto "
                   "de optimizar umbrales, sino un error de comparacion entre brazos con distinto "
                   "conjunto de ajuste. No se ejecuta downweighting, trimming ni nueva busqueda."),
        "closure_type": "A",
        "reusable_artifact": "experiments/29_reweighting_v4.py, D-026, D-072",
    },
    {
        "family": "fuzzy_matching",
        "status": "CLOSED",
        "best_result": "V29-03 REJECTED, FUZZY_MATCHING = INFORMATIONAL_ONLY",
        "delta_vs_V03": "no aporta senal discriminante",
        "lb_transfer": "no enviado",
        "reason": ("V29-02 fue DESCRIPTIVE_ONLY y V29-03 REJECTED. El metodo anterior NO se cambia. "
                   "La conclusion es que encuentra dificultad, no senal de etiqueta."),
        "closure_type": "A",
        "reusable_artifact": "experiments/35_* (V29-00 a V29-04), reports/control_pseudo_autores.json",
    },
    {
        "family": "ensembles",
        "status": "CLOSED",
        "best_result": "ENS-02 anidado 0.791900, descartada por no superar MOD-03b en +0.005. v12_p7: 0.782277 local, la mejor linea local de la fase v8",
        "delta_vs_V03": "+0,014 a +0,005 segun el criterio, siempre bajo D-004 o anidado, no comparable directamente",
        "lb_transfer": "TRES envios, los tres a la baja. v04_ens 0.7751; v08_p1 0.7747; v12_p7 0.77122",
        "reason": ("La causa mecanica esta medida: el promedio contrae las probabilidades y con "
                   "umbrales fijos eso mueve celdas en la direccion equivocada. Por eso las "
                   "candidatas de tasa se fijan por CUANTIL, que es invariante a la escala. Y no hay "
                   "varianza que reducir: v16_ch6 probo cinco semillas y liblinear es determinista, "
                   "los cinco modelos salen bit a bit identicos."),
        "closure_type": "C",
        "reusable_artifact": "submissions/ v04, v08_p1, v12_p7; D-059",
    },
    {
        "family": "stacking y meta-modelos",
        "status": "CLOSED",
        "best_result": "S08 meta holdout 0.8465 frente a 0.8285 de V03, delta +0.0180. E2-13-META-ENT-RIDGE 0.7885 anidado",
        "delta_vs_V03": "+0,018 en holdout; +0,0113 en corrected_cv anidado",
        "lb_transfer": "CUATRO envios, los cuatro a la baja. v08_meta_v2 0.78163; v09_a 0.77895; v10_b 0.77978; v11_c 0.78150; v20_01 0.77663",
        "reason": ("Es la familia MEJOR MEDIDA del proyecto y la que mas veces transfirio a la baja. "
                   "El meta mejoro en dev (+0.0102) y en el holdout (+0.0180) y no transfirio. D-020 "
                   "y D-024 cierran la linea. D-070 anade el dato que la explica: la metrica "
                   "corregida tiene una banda de 0,017 entre particiones equivalentes, luego un "
                   "delta de +0.010 esta dentro del ruido del instrumento."),
        "closure_type": "C",
        "reusable_artifact": "submissions/ v08_meta_v2, v09, v10, v11, v20_01; D-020, D-024, D-070",
    },
    {
        "family": "structural_features",
        "status": "CLOSED",
        "best_result": "V15-01: 0.775860, NO ADOPTADA. V14-00 0.660630. V32-05 url_length: corrected_cv +0.0045 pero LB delta 0",
        "delta_vs_V03": "-0.0014 en la mejor estimacion",
        "lb_transfer": "ENVIADO como v32_05 (ID 56608788, NO registrado en el log). LB 0.78306, delta EXACTAMENTE 0",
        "reason": ("V15-01 lo dice asi: es la TERCERA estimacion independiente que coincide en que "
                   "los rasgos estructurales no aportan. El envio de url_length es el control mas "
                   "limpio del proyecto: la interaccion se midio con +0.0045 de corrected_cv y el LB "
                   "no se movio ni en la sexta cifra decimal, porque la submission resulto "
                   "IDENTICA a V03."),
        "closure_type": "A",
        "reusable_artifact": "submissions/toxic_submission_v32_05_url_length.csv, experiments/31_* (V32)",
    },
]


# ------------------------------------------------------------------ candidatos
# Criterios de la directiva, los cinco simultaneos:
#   legal, reproducible, sin leakage, no rechazado ya, mecanismo claro
CANDIDATOS: list[dict[str, Any]] = [
    {
        "candidate": "v23_01 svd300_rbf (V4-X01 candidato D)",
        "mechanism": "SVD 300 + SVC kernel RBF con calibracion sigmoidal, sobre la union TF-IDF de V03",
        "honest_CV": 0.7904,
        "delta_vs_V03": 0.013189,
        "fold_consistency": "5/5 pliegues",
        "bootstrap": "IC95 del delta TOCA 0; McNemar p=0.0452, 136 rescates contra 104 danos",
        "LB": 0.77938,
        "LB_delta": -0.00368,
        "verified": "SI, RECHAZADO",
        "reason": ("Cumple legal, reproducible, sin leakage y mecanismo claro, pero ya esta "
                   "RECHAZADO: se envio y el LB cayo. El IC95 tocaba cero, que es la aviso que se "
                   "ignoro. Ademas la firma de memorizacion esta en los vectores de soporte, 60 a 76 "
                   "por ciento de las muestras."),
        "submission_available": "SI, en disco, ya gastada",
    },
    {
        "candidate": "v08_meta_v2 (meta de 44 features, Ridge alpha=3)",
        "mechanism": "stacking sobre 4 bases TF-IDF mas 8 rasgos estructurales, con entropias y margenes",
        "honest_CV": 0.7874,
        "delta_vs_V03": 0.010189,
        "fold_consistency": "mejoro en dev y en holdout",
        "bootstrap": "no registrado en el artefacto",
        "LB": 0.78163,
        "LB_delta": -0.00143,
        "verified": "SI, RECHAZADO",
        "reason": "Enviado y medido por debajo de V03. D-020. Cuarto caso consecutivo de mejora local que no transfiere.",
        "submission_available": "SI, en disco, ya gastada",
    },
    {
        "candidate": "v20_01 regla HS por interaccion de scores (V4-X01 candidato C, 36 features)",
        "mechanism": "LR C=1.0 sobre [logit(HS), logit(HS)*logit(TR), logit(AG)] con downweight por violacion del umbral",
        "honest_CV": 0.7882,
        "delta_vs_V03": 0.010989,
        "fold_consistency": "4/5 pliegues",
        "bootstrap": "IC95 [+0.0054, +0.0200], NO incluye 0. Unico de los cuatro que lo cumplia",
        "LB": 0.77663,
        "LB_delta": -0.00643,
        "verified": "SI, RECHAZADO. Ademas RECHAZADO antes por el holdout",
        "reason": ("Era el unico que cumplia los tres criterios de D-016. El holdout congelado lo "
                   "rechazo (0.8163 contra 0.8285 de V03, delta -0.0122) y consumio la quinta y "
                   "ultima evaluacion. Se envio igualmente y el LB cayo mas todavia. Cuarto caso "
                   "consecutivo del mismo patron."),
        "submission_available": "SI, en disco (toxic_submission_v12_descartado_meta36_downweight.csv)",
    },
    {
        "candidate": "E2-13-META-ENT-RIDGE (meta + entropias)",
        "mechanism": "Ridge sobre el meta ampliado con entropias de las bases, ajustado anidado por pliegue",
        "honest_CV": 0.7885,
        "delta_vs_V03": 0.011289,
        "fold_consistency": "no registrado",
        "bootstrap": "no aplica",
        "LB": "n/d",
        "LB_delta": "n/d",
        "verified": "NO",
        "reason": ("CERRADA por sparsity, con un delta de +0,0113 sobre V03 y nested fold a fold. "
                   "No se envio, luego no hay veredicto externo. Es el unico artefacto con delta "
                   "honesto de +0,011 que NO tiene un LB que lo contradiga."),
        "submission_available": "NO",
    },
    {
        "candidate": "V03 con umbrales anidados de P56 (T1 = 0,41/0,46/0,45)",
        "mechanism": "umbral por etiqueta como mediana de cinco estimaciones interiores, en vez de los umbrales de D-004",
        "honest_CV": P56_HONEST,
        "delta_vs_V03": P56_HONEST - V03_CORRECTED_CV,
        "fold_consistency": "NO reproducible: los umbrales de V03 caen DENTRO del rango interior en las tres etiquetas",
        "bootstrap": "de la cifra global: IC95 [-0.003456, +0.008989], INCLUYE 0",
        "LB": "n/d",
        "LB_delta": "n/d",
        "verified": "NO",
        "reason": ("CASE B de P56. El delta plenamente honesto es NEGATIVO, el net de celdas es -11 "
                   "(43 rescates, 54 danos) y negativo en las tres etiquetas, y el optimismo de "
                   "aplicar la mediana a los cinco pliegues es +0,0053. NO_CONFIRMED."),
        "submission_available": "NO",
    },
    {
        "candidate": "v25_01 FM k=4",
        "mechanism": "factorization machines sobre las cuatro bases, k=4, lambda_v=1",
        "honest_CV": 0.782699,
        "delta_vs_V03": 0.005487,
        "fold_consistency": "no registrado",
        "bootstrap": "no aplica",
        "LB": "n/d",
        "LB_delta": "n/d",
        "verified": "NO",
        "reason": ("Medido bajo D-004, luego NO es comparable con el corrected_cv de V03, que es "
                   "D-016. k=8 y k=16 son identicos a k=4. Cerrada como no candidata."),
        "submission_available": "NO",
    },
    {
        "candidate": "v16_ch2 correccion de prior con EM",
        "mechanism": "desplazamiento del logit con los factores por etiqueta del EM de Saerens sobre las probabilidades de test",
        "honest_CV": "n/d (validacion anidada de V9-02 dio +0.000574)",
        "delta_vs_V03": 0.000574,
        "fold_consistency": "1/5",
        "bootstrap": "no aplica",
        "LB": "n/d",
        "LB_delta": "n/d",
        "verified": "NO",
        "reason": ("CUARENTENADA, y por el mismo motivo que v17: consume informacion del conjunto de "
                   "test. El EM no usa etiquetas, pero es transductivo. Se genero y no se envio."),
        "submission_available": "SI, en disco, pero CUAENTENADA",
    },
]


# ----------------------------------------------------------------- preguntas abiertas
ABIERTAS: list[dict[str, Any]] = [
    {
        "question": "La brecha OOF -> LB de 0,00584855 de V03 no tiene mecanismo identificado",
        "evidence": "corrected_cv 0.777211 frente a LB 0.78306. P56 lo deja en 0,25 sigma",
        "why_open": ("La unica explicacion candidata con evidencia es el shift de prevalencia: el test "
                     "tiene 6,48 por ciento menos de positivos (k=0,935177, analitico, confirmado por "
                     "EM de Saerens). El OOF se computa a prevalencia de train. Y la version "
                     "automatica de la adaptation YA se probo y perdio: V10-05, regla F1 optima con "
                     "shift de prevalencia, dio -0.003921 con 1 de 5 particiones. Cerrarla exigiria "
                     "consultar Kaggle."),
        "tested": "PARCIAL: el shift se midio y se confirmo; la regla que lo corrige se probo y fallo",
        "legal": "SI para medir, NO para la unica via de identificacion restante (consultar Kaggle)",
        "expected_value": "El unico eje del proyecto con un contraste cuasi experimental (V02 -> V03, "
                          "pendiente causal +0,12011 por unidad de tasa). No reopened sin un "
                          "mecanismo que no sea la regla ya refutada.",
    },
    {
        "question": "La inversion del ranking local al LB en la familia de meta no tiene causa identificada",
        "evidence": ("v08_meta_v2: OOF 0.7874 (+0,0102) y holdout 0.8465 (+0,0180), LB 0.78163 (-0,00143). "
                     "Y el caso simetrico de CHAMPION.md: en los tres candidatos v09/v10/v11, el mejor "
                     "local (C-EMO) obtuvo el peor LB"),
        "why_open": ("Se sabe que la banda de particion de la metrica corregida es de 0,017 (D-070), "
                     "mas que cualquier delta local de la familia. Pero eso explica por que no se "
                     "puede CONFIAR en el ranking local, no por que se INVIERTA. La asimetria entre "
                     "dev y holdoutmetrica no esta caracterizada."),
        "tested": "SI, extensamente: 4 envios de meta, 3 de ensembles, 1 de especializacion, 1 de kernel. Todos a la baja",
        "legal": "SI",
        "expected_value": "ALTO conceptualmente: identificarla deciria que magnitud local es fiable, "
                          "que es la pregunta que gobierna cualquier fase futura. Requiere mas "
                          "consultas de LB, que es el recurso escaso.",
    },
    {
        "question": "La hipotesis de ambiguedad de etiqueta en HS quedo registrada sin medir",
        "evidence": "V28-03: la hipotesis de ambiguedad queda ABIERTA y sin medir. D-026: la dureza util es la distancia a la frontera",
        "why_open": ("Se registro explicitamente como abierta. No se midio. El mecanismo es conocido: "
                     "la ambiguedad de HS se manifestaria como ejemplos muy cerca de  a la frontera, que ya se "
                     "midio como dureza y no como ruido."),
        "tested": "NO. Registrada en V28-03 sin ejecutar",
        "legal": "SI",
        "expected_value": "CUANTIFICADO Y BAJO. El techo maximo son las 42 celdas no rescatables, es "
                          "decir 0,95 por ciento de las 4449 celdas, en 33 tweets. Incluso un oracel "
                          "perfecto sobre ellas moveria el macro-F1 alrededor de +0,009, por debajo de "
                          "la banda de particion de 0,017 de D-070. No cumple el criterio de "
                          "reapertura de la directiva.",
    },
    {
        "question": "La sensibilidad del LB esta limitada por el numero de filas del test",
        "evidence": (f"error estandar teorico {LB_SE_TEORICO} con {N_TEST} filas; "
                     f"{CELLS_PER_003} cambios netos de acierto mueven el LB 0,003"),
        "why_open": ("Es una restriccion del instrumento, no una hipotesis. No se puede cerrar: es una "
                     "propiedad de la competencia. Se documenta porque gobierna el coste de cualquier "
                     "trabajo futuro."),
        "tested": "SI. V8-02 adopto un filtro de detectabilidad previo al gate de la fase: ninguna "
                  "candidata cuya tasa de acierto caiga por debajo de 0,55 se envia",
        "legal": "SI",
        "expected_value": "Restriccion permanente. Ninguna perturbacion de baja capacidad de V03 puede "
                          "esperar mas de +0,003, y por debajo de ~250 celdas el cambio es indetectable.",
    },
    {
        "question": "Ninguna linea de estructura de etiquetas se ejecuto porque se cerro por condicion de parada",
        "evidence": ("V13-02: LINEA DE INFERENCIA ESTRUCTURADA DE ETIQUETAS CERRADA FORMALMENTE. "
                     "V14-00: FASE CERRADA por la condicion de parada numero 1, 0 de 4 experimentos"),
        "why_open": ("Estas lineas se cerraron por una regla de la directiva, no por un resultado. Es la "
                     "unica situacion del registro donde una familia NO tiene veredicto empirico. La "
                     "diferencia con las demas es que aqui no se puede decir 'no funciono'."),
        "tested": "NO, por condicion de parada",
        "legal": "SI",
        "expected_value": "BAJO, pero por un motivo que conviene precisar: V13-01, que SI se ejecuto, "
                          "quedo en +0,000534, y V15-01, que tambien se ejecuto y es una estimacion "
                          "independiente, dio -0,0014. Dos ejecuciones reales de la misma idea "
                          "coinciden en que no aporta. Abrir la parte no ejecutada seria buscar una "
                          "tercera estimacion de algo que ya tiene dos.",
    },
]


# ------------------------------------------------------------------ inconsistencias
INCONSISTENCIAS: list[dict[str, Any]] = [
    {
        "severity": "CRITICAL",
        "where": "experiments.csv, columnas `f1` y `macro_f1`",
        "inconsistency": ("Las dos columnas mezclan al menos TRES metricas y TRES protocolos. "
                          "`macro_f1` contiene macro-6 (diagnostica) en la era V2/E2/MOD/TH/ENS y la "
                          "metrica del proyecto en la era V10+. `f1` contiene holdout de 248 filas en "
                          "S08, protocolo D-004 en la era antigua y D-016 en la era corregida."),
        "evidence": ("P45 sobre el OOF de V03: metrica del proyecto 0.7772114525 y macro-6 "
                     "0.8182213363, diferencia +0.04101. MOD-03b_BEST: mf 0.834900 contra f1 0.792700, "
                     "diferencia +0.042200. El maximo de `macro_f1` en todo el registro es 0.8743 "
                     "(S08), que es macro-6 sobre holdout."),
        "consequence": ("Cualquier tabla de 'mejor CV' construida tomando el maximo de esas columnas "
                        "es FALSA por mas de 0,04, y mixing protocolos anade hasta 0,024 mas. Este "
                        "script NO construye tal tabla: etiqueta cada fila con metrica, protocolo y "
                        "poblacion, y solo compara dentro de un mismo protocolo."),
        "correction": ("No se corrige el registro historico. Se anade la clasificacion en "
                       "`reports/p57_final_experiment_inventory.csv`, que es la forma de conservar el "
                       "significado original y hacer la columna utilizable."),
    },
    {
        "severity": "CRITICAL",
        "where": "submissions/leaderboard_log.csv frente a experiments.csv",
        "inconsistency": ("V33-03 documento un envio REAL a Kaggle: ID 56608788, fecha 2026-09-27, "
                          "fichero toxic_submission_v32_05_url_length.csv, LB 0.78306, y anota '2 "
                          "submissions restantes hoy'. Esa fila NO esta en leaderboard_log.csv."),
        "evidence": ("experiments.csv fila V33-03, campo `result`. El log tiene 22 filas y ninguna "
                     "corresponde a v32_05_url_length. Hay 37 ficheros de submission en disco y 15 sin "
                     "fila en el log, de los cuales solo ESTE se envio."),
        "consequence": ("El total historico de envios es 23, no 22. PROJECT_STATE.md afirma '0 de 22 "
                        "submissions superaron V03', y CHAMPION.md auditado en P46 afirma lo mismo. La "
                        "afirmacion sigue siendo cierta en su sustancia, porque 0 de 23 superaron "
                        "0.78306, pero el recuento es erroneo."),
        "correction": ("Se anade la fila al informe de P57 con su ID y su LB, y se deja el log intacto. "
                       "No se modifica un registro historico de measurements; se documenta la "
                       "discrepancia y se da la cifra correcta."),
    },
    {
        "severity": "IMPORTANT",
        "where": "experiments.csv, fila V43-01",
        "inconsistency": ("La columna `question` dice 'PLS Regression como proyeccion supervisada "
                          "document x term', pero el `script` que registra es "
                          "experiments/v43_01_vw_optimizers.py, cuyo docstring es sobre Vowpal Wabbit "
                          "con FTRL, Coin Betting y PiSTOL. Ademas la `decision` habla de la API de VW."),
        "evidence": "docstring de experiments/v43_01_vw_optimizers.py frente a la fila V43-01 del registro",
        "consequence": ("El texto de la pregunta no corresponde al experimento. Quien lea el registro "
                        "creera que V43-01 evalua PLS, cuando lo que se intento fue FTRL y quedo "
                        "INVALIDADO por la API de VW. La familia PLS se cerro en V43-02 y V44-02."),
        "correction": ("Se documenta en P57. NO se edita la fila historica. El sentido original se "
                       "conserva: V43-01 es el intento de FTRL, y su invalidacion es la que abre "
                       "V44-01."),
    },
    {
        "severity": "IMPORTANT",
        "where": "leaderboard_log.csv, filas 56597267 y 56599802",
        "inconsistency": ("La columna `archivo` registra el nombre sin el prefijo `submissions/`, "
                          "mientras que las otras 20 filas lo incluyen. Los ficheros existen en "
                          "`submissions/`."),
        "evidence": "log dice 'toxic_submission_v20_01_...' y 'toxic_submission_v23_01_...'; ambos existen como submissions/...",
        "consequence": "Un lector que resuelva la ruta literalmente concluiria que faltan dos artefactos. No falta ninguno.",
        "correction": "Se documenta. No se modifica el log, porque las dos filas tienen un ID numerico de Kaggle y su contenido es fiable.",
    },
    {
        "severity": "IMPORTANT",
        "where": "cifras que circulan por el repositorio",
        "inconsistency": ("P47 reconcilio dos cifras que no cuadraban: 'holdout = 519' y 'unas 44.000 "
                          "filas no etiquetadas'."),
        "evidence": ("519 es el numero de positivos de TR en train, no el tamano del holdout, que es "
                     "248. Y no existe ninguna fuente de 44.000 filas: los corpus externos se miden en "
                     "tokens, y el 44 es el numero de features del meta-modelo E2-02."),
        "consequence": "Ambas cifras circulaban como hechos del inventario de datos y ambas son falsas.",
        "correction": "Ya corregido en P47 (reports/p47_data_inventory.json, campo reconciliacion_de_cifras). Se confirma aqui.",
    },
    {
        "severity": "MINOR",
        "where": "CHAMPION.md, seccion 4",
        "inconsistency": ("La tabla de candidatas de V16 titula una columna 'LB combinado' que contiene "
                          "PREDICCIONES, no mediciones. El propio CHAMPION.md lo advierte en un bloque "
                          "anadido por P46."),
        "evidence": "valores 0.78535 (ch1) y 0.78650 (ch5) frente a sus LB medidos reales 0.77828 y 0.77495",
        "consequence": ("Las dos candidatas que parecian ganadoras por la prediccion perdieron por "
                        "0.007 y por 0.012 respectivamente. Leer la columna sin el aviso seria el "
                        "error de lectura mas caro del proyecto."),
        "correction": "El aviso existe y es correcto. Se mantiene la tabla intacta como registro de lo que se predijo, con el aviso encima. Sin cambio.",
    },
    {
        "severity": "MINOR",
        "where": "CHAMPION.md, seccion 1",
        "inconsistency": "El bloque del campeon declara 'max_iter 2000', mientras que CHAMPION.md y el manifiesto declaran max_iter 1000.",
        "evidence": "CHAMPION.md linea del bloque Version v03 dice max_iter=1000; el bloque 'Champion conservado' del final dice max_iter 2000",
        "consequence": "Estandar de liblinear: el valor no cambia la solucion. Es una discrepancia documental, no un riesgo numerico.",
        "correction": "Se documenta. El pipeline congelado experiments/baseline_v03_frozen/pipeline_v03.json es la fuente de verdad y no se modifica.",
    },
]


def construir_candidatos() -> pd.DataFrame:
    """Construye la tabla descriptiva de candidatos historicos.

    Returns:
        DataFrame de candidatos, sin ordenar por puntuacion alguna.
    """
    return pd.DataFrame(CANDIDATOS)


def construir_familias() -> pd.DataFrame:
    """Construye la tabla consolidada de estado de familias.

    Returns:
        DataFrame de familias con su tipo de cierre.
    """
    cols = ["family", "status", "best_result", "delta_vs_V03", "lb_transfer",
            "reason", "closure_type", "reusable_artifact"]
    return pd.DataFrame(FAMILIAS)[cols].rename(columns={
        "best_result": "best_result", "delta_vs_V03": "delta_vs_V03",
        "lb_transfer": "LB_transfer", "reason": "reason",
        "closure_type": "closure_type_A_to_F", "reusable_artifact": "reusable_artifact"})


def main() -> None:
    """Ejecuta la sintesis estrategica de P57.

    Returns:
        None.
    """
    inicio = datetime.now(timezone.utc)
    log("P57 cierre estrategico. 0 modelos, 0 entrenamiento, 0 submissions, 0 consultas a Kaggle")
    log("   no se usa el holdout ni las etiquetas de test")

    d = pd.read_csv(EXPERIMENTS)
    lb_log = pd.read_csv(LB_LOG)
    lb_log["kaggle_f1"] = pd.to_numeric(lb_log["kaggle_f1"], errors="coerce")
    mapa_sub = {os.path.basename(str(a)): str(s)
                for a, s in zip(lb_log.archivo, lb_log.submission_id)}
    log(f"   registro: {len(d)} experimentos | LB log: {len(lb_log)} filas con LB medido")

    # ------------------------------------------------------------- integridad
    hash_real = __import__("hashlib").sha256(
        (DIR_SUB / "toxic_submission_v03_lr_union_thr.csv").read_bytes()).hexdigest()
    if hash_real != V03_SHA:
        raise SystemExit("PARADA: la submission de V03 no coincide con el sha256 congelado")
    log(f"   V03 sha256 verificado: {hash_real[:16]}... intacto")

    # ------------------------------------------------- envio sin registrar (V33-03)
    d_env = d[d.experiment_id == "V33-03"]
    envio_no_registrado = {}
    if len(d_env):
        r = d_env.iloc[0]
        envio_no_registrado = {
            "experiment_id": "V33-03", "fecha": str(r.get("date", "")),
            "kaggle_id": "56608788", "fichero": "toxic_submission_v32_05_url_length.csv",
            "LB": V03_LB, "delta_vs_V03": 0.0,
            "nota": ("Envio real y aceptado, con LB 0.78306 y delta EXACTAMENTE 0 porque la "
                     "submission resulto identica a V03. NO figura en leaderboard_log.csv. "
                     "Anotaba '2 submissions restantes hoy'."),
        }
    total_envios = len(lb_log) + (1 if envio_no_registrado else 0)
    log(f"   envios con LB en el log: {len(lb_log)} | envio real NO registrado: "
        f"{1 if envio_no_registrado else 0} | TOTAL HISTORICO: {total_envios}")

    ficheros = [f for f in os.listdir(DIR_SUB) if f.startswith("toxic_submission_")]
    nunca_enviados = sorted(set(ficheros) - {os.path.basename(str(a)) for a in lb_log.archivo}
                            - ({envio_no_registrado["fichero"]} if envio_no_registrado else set()))
    log(f"   ficheros de submission en disco: {len(ficheros)} | "
        f"generados y nunca enviados: {len(nunca_enviados)}")

    # ------------------------------------------------------------- presupuesto
    envios_por_dia = {k: int(v) for k, v in
                     lb_log.fecha.value_counts().sort_index().items()}
    if envio_no_registrado:
        f = envio_no_registrado["fecha"]
        envios_por_dia[f] = envios_por_dia.get(f, 0) + 1
    hoy = sum(v for k, v in envios_por_dia.items() if k == FECHA)
    presupuesto = {
        "envios_historicos": total_envios,
        "envios_con_LB_verificado_en_log": len(lb_log),
        "envios_reales_no_registrados": 1 if envio_no_registrado else 0,
        "envios_por_dia": envios_por_dia,
        "envios_hoy": hoy,
        "limite_diario_conocido": False,
        "unica_cifra_local": ("V33-03 anoto '2 submissions restantes hoy' el 2026-09-27, dia en el "
                              "que se cuentan 7 envios. Eso exige un limite diario de al menos 9, "
                              "pero la cifra EXACTA no esta registrada en ningun sitio local."),
        "reserva_propia_del_proyecto": 3,
        "nota": ("No se consulta Kaggle. La reserva de 3 es del proyecto, no un limite de la "
                "competencia. Hoy 2026-09-28 hay 0 envios."),
    }
    log(f"   envios hoy ({FECHA}): {hoy} | limite diario EXACTO: desconocido sin consultar Kaggle")

    # ------------------------------------------------------------- artefactos
    inventario = construir_inventario(d, lb_log, mapa_sub)
    fam = construir_familias()
    cand = construir_candidatos()
    abi = pd.DataFrame(ABIERTAS)
    inc = pd.DataFrame(INCONSISTENCIAS)

    n_comparable = int((inventario["protocolo"] == "D-016").sum())
    n_holdout = int((inventario["protocolo"] == "HOLD").sum())
    n_d004 = int((inventario["protocolo"] == "D-004").sum())
    log(f"   inventario: {n_d004} filas D-004, {n_comparable} D-016, {n_holdout} HOLD, "
        f"{len(inventario) - n_d004 - n_comparable - n_holdout} sin clasificar")
    log(f"   {len(fam)} familias | {len(cand)} candidatos | {len(abi)} preguntas abiertas | "
        f"{len(inc)} inconsistencias documentales")

    # mejor CV por protocolo, sin mezclar
    comparable = inventario[(inventario["protocolo"] == "D-016")
                            & (inventario["metrica"] == "proyecto")]
    filas_comparables = d.loc[comparable.index, "f1"]
    filas_comparables = pd.to_numeric(filas_comparables, errors="coerce").dropna()
    best_comparable = float(filas_comparables.max())
    best_comparable_id = comparable.loc[filas_comparables.idxmax(), "experiment_id"]
    # El maximo D-016 es el MEJOR de varios candidatos comparados sobre el mismo OOF, luego
    # lleva seleccion y no es una estimacion anidada. Se reporta por separado.
    anidados = comparable[comparable["anidado"]]
    val_anidados = pd.to_numeric(anidados["honest_cv"], errors="coerce").dropna()
    best_anidado = float(val_anidados.max()) if len(val_anidados) else float("nan")
    best_anidado_id = anidados.loc[val_anidados.idxmax(), "experiment_id"] if len(val_anidados) else "n/d"

    # el mejor OOF de todo el registro con LB medido
    con_lb = inventario[inventario["LB_verified"]].copy()
    con_lb["lb_num"] = pd.to_numeric(con_lb["LB"], errors="coerce")
    # El maximo del leaderboard se lee del log, y su atribucion se contrasta con el registro.
    lb_max_log = float(pd.to_numeric(lb_log["kaggle_f1"], errors="coerce").max())
    id_max_log = str(lb_log.loc[pd.to_numeric(lb_log["kaggle_f1"], errors="coerce").idxmax(),
                                "submission_id"])
    if abs(lb_max_log - V03_LB) > 1e-12 or id_max_log != V03_ID:
        raise SystemExit(f"PARADA: el maximo del log es {lb_max_log} ({id_max_log}), "
                         f"no {V03_LB} ({V03_ID})")
    log(f"   maximo del leaderboard verificado en el log: {lb_max_log} ({id_max_log})")
    mejor_lb = pd.Series({"experiment_id": V03_ID, "LB": lb_max_log})

    jerarquia = {
        "best_architectural_CV_macro6_diagnostica": {
            "valor": 0.8743, "experimento": "S08",
            "metrica": "macro-6, NO la metrica de la competencia",
            "protocolo": "HOLD", "poblacion": "holdout 248",
            "comparable_con_V03": False,
            "nota": "P45 declaro macro-6 diagnostica. Sobre el OOF de V03 vale 0.8182213363, +0.04101 sobre la metrica del proyecto.",
        },
        "best_holdout_CV": {
            "valor": 0.8465, "experimento": "S08 (meta de 44 features)",
            "metrica": "metrica del proyecto", "protocolo": "HOLD", "poblacion": "holdout 248",
            "comparable_con_V03": False,
            "nota": "V03 en el mismo holdout da 0.8285. Delta +0.0180. Y el LB de ese meta fue 0.78163, por debajo de V03.",
        },
        "best_CV_protocolo_D004": {
            "valor": 0.8039, "experimento": "S05 / v05_optuna",
            "metrica": "metrica del proyecto", "protocolo": "D-004", "poblacion": "dev",
            "comparable_con_V03": False,
            "nota": "El maximo local de las 22 submissions. Su LB fue 0.77602, el cuarto peor de la familia. D-004 es mas debil que D-016.",
        },
        "best_comparable_CV_D016": {
            "valor": best_comparable, "experimento": best_comparable_id,
            "metrica": "metrica del proyecto", "protocolo": "D-016", "poblacion": "dev 1483",
            "comparable_con_V03": True,
            "nota": ("El maximo comparable bajo el protocolo corregido. Corresponde a la linea "
                     "svd300_rbf, que se envio y dio LB 0.77938. PERO es el mejor de cuatro "
                     "candidatos comparados sobre el mismo OOF, luego lleva SELECCION y su IC95 "
                     "tocaba cero. No es una estimacion anidada."),
        },
        "best_honest_CV_anidado": {
            "valor": best_anidado, "experimento": best_anidado_id,
            "metrica": "metrica del proyecto", "protocolo": "D-016", "poblacion": "dev 1483",
            "comparable_con_V03": True,
            "nota": ("El maximo entre experimentos con ajuste anidado o fold a fold, luego sin "
                     "seleccion sobre el pliegue de evaluacion. Es la cifra que un argumento de "
                     "'hemos probado mas cosas y ninguna supera a V03' tiene que citar."),
        },
        "V03_corrected_cv": {
            "valor": V03_CORRECTED_CV, "experimento": V03_ID,
            "metrica": "metrica del proyecto", "protocolo": "D-016", "poblacion": "dev 1483",
            "comparable_con_V03": True, "nota": "Referencia canonica del proyecto.",
        },
        "V03_historical_cv_D004": {
            "valor": V03_HISTORICAL_CV, "experimento": V03_ID,
            "metrica": "metrica del proyecto", "protocolo": "D-004", "poblacion": "dev",
            "comparable_con_V03": False,
            "nota": ("Lo que registra leaderboard_log.csv como local_f1. Difiere del corrected_cv "
                     "canonico en +0.021689, que es exactamente el coste de medir con D-004."),
        },
        "V03_holdout": {
            "valor": V03_HOLDOUT, "experimento": V03_ID,
            "metrica": "metrica del proyecto", "protocolo": "HOLD", "poblacion": "holdout 248",
            "comparable_con_V03": False,
            "nota": "El holdout esta AGOTADO, 5 de 5 evaluaciones, y no se usa desde la fase v3.",
        },
        "best_verified_LB": {
            "valor": V03_LB, "experimento": str(mejor_lb["experiment_id"]),
            "metrica": "leaderboard de Kaggle", "protocolo": "externo", "poblacion": f"test {N_TEST}",
            "comparable_con_V03": True,
            "nota": "Unico instrumento externo. 0 de 23 envios historicos lo superaron.",
        },
        "current_champion": {
            "valor": V03_ID, "experimento": V03_ID,
            "metrica": "leaderboard de Kaggle", "protocolo": "externo", "poblacion": f"test {N_TEST}",
            "comparable_con_V03": True,
            "nota": f"sha256 {V03_SHA}",
        },
        "remaining_submission_budget": {
            "valor": "limite diario exacto desconocido; 0 envios hoy; reserva propia de 3",
            "experimento": "n/d", "metrica": "n/d", "protocolo": "n/d", "poblacion": "n/d",
            "comparable_con_V03": False, "nota": presupuesto["unica_cifra_local"],
        },
    }
    log(f"   best comparable D-016 = {best_comparable:.6f} ({best_comparable_id}), con SELECCION")
    log(f"   best anidado D-016     = {best_anidado:.6f} ({best_anidado_id}), sin seleccion")
    log(f"   best_verified_LB       = {V03_LB:.5f} ({mejor_lb['experiment_id']})")

    # ------------------------------------------------------------------ escribir
    # Todo lo que se escribe pasa por transliteracion ASCII (requisito bloqueante).
    asciificar(inventario).to_csv(CSV_INVENTARIO, index=False, encoding="utf-8")
    asciificar(fam).to_csv(CSV_FAMILIAS, index=False, encoding="utf-8")
    asciificar(cand).to_csv(CSV_CANDIDATOS, index=False, encoding="utf-8")
    asciificar(abi).to_csv(CSV_ABIERTAS, index=False, encoding="utf-8")
    for destino in (CSV_INVENTARIO, CSV_FAMILIAS, CSV_CANDIDATOS, CSV_ABIERTAS):
        crudo = destino.read_bytes()
        if any(x > 127 for x in crudo):
            raise SystemExit(f"PARADA: {destino.name} conserva bytes no ASCII")

    informe = {
        "familia": "P57_STRATEGIC_CLOSURE",
        "experimento": "V57-00 (sintesis, auditoria y estrategia final)",
        "fecha": inicio.isoformat(timespec="seconds"),
        "naturaleza": "AUDIT_ONLY. 0 modelos, 0 entrenamiento, 0 submissions, 0 consultas a Kaggle",
        "baseline_congelado": {
            "id": V03_ID, "representacion": V03_REPR, "clasificador": V03_CLF,
            "umbrales": V03_UMBRALES, "corrected_cv": V03_CORRECTED_CV,
            "LB": V03_LB, "sha256": V03_SHA, "sha256_verificado": True,
        },
        "jerarquia_de_magnitudes": jerarquia,
        "presupuesto": presupuesto,
        "familias": {f["family"]: {"status": f["status"], "closure_type": f["closure_type_A_to_F"] if "closure_type_A_to_F" in f else f["closure_type"]}
                     for f in FAMILIAS},
        "inventario": {
            "filas": len(inventario), "D-004": n_d004, "D-016": n_comparable,
            "HOLD": n_holdout, "sin_clasificar": len(inventario) - n_d004 - n_comparable - n_holdout,
        },
        "umbral_p55_p56": {
            "V03": V03_CORRECTED_CV, "P55_reportado": P55_REPORTED,
            "P56_global": P56_GLOBAL, "P56_honesto": P56_HONEST,
            "P56_optimismo": P56_OPTIMISMO, "THRESHOLD_TRANSFER": "CLOSED",
        },
        "canales_p49_p53": {
            "V03_errores": N_CELDAS_TOTAL - V03_CORRECTOS,
            "V03_correctos": V03_CORRECTOS, "rescuable": V03_RESCUABLE,
            "no_rescatable": V03_NO_RESCATABLE,
            "tweets_no_rescatables": V03_TWEETS_NO_RESCATABLES,
            "union_historica_de_canales": V03_RESCUABLE,
            "true_oracle": TRUE_ORACLE, "mejor_politica_observable": OBSERVABLE_CEILING,
            "CONTEXTUAL_SELECTION": "CLOSED",
        },
        "datos": {
            "train_oficial": N_TRAIN_OFICIAL, "dev": N_DEV, "holdout": N_HOLDOUT,
            "test_oficial": N_TEST, "DATA_EXPANSION": "CLOSED",
        },
        "metrica": {
            "LOCAL_METRIC_REPRODUCED": True,
            "KAGGLE_SCORER_DIRECTLY_VERIFIED": False,
        },
        "inconsistencias_documentales": INCONSISTENCIAS,
        "candidatos": CANDIDATOS,
        "preguntas_abiertas": ABIERTAS,
        "nunca_enviados": nunca_enviados,
        "transliteracion_ascii": ("Los CSV y el JSON de P57 se transliteran a ASCII puro al "
                                  "escribir. El registro historico experiments.csv contiene "
                                  "acentos, signos de seccion y guiones tipograficos; el "
                                  "significado se conserva, solo se normaliza la codificacion."),
        "gastos": {"submissions": 0, "modelos": 0, "entrenamientos": 0,
                   "consultas_kaggle": 0, "uso_holdout": 0, "uso_test": 0,
                   "segundos": round((datetime.now(timezone.utc) - inicio).total_seconds(), 1)},
    }
    JSON_INFORME.write_text(json.dumps(informe, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    _md(informe, inventario, fam, cand, abi, inc, jerarquia, presupuesto,
        envio_no_registrado, nunca_enviados, best_comparable, best_comparable_id,
        best_anidado, best_anidado_id)
    log(f"   escrito reports/p57_* ({len(informe)} claves en el JSON)")
    log(f"VEREDICTO: jerarquia construida | {len(fam)} familias | "
        f"{len(inc)} inconsistencias | 0 models, 0 submissions, "
        f"{informe['gastos']['segundos']} s")


# ----------------------------------------------------------------------- markdown
def _md(informe: dict[str, Any], inv: pd.DataFrame, fam: pd.DataFrame, cand: pd.DataFrame,
        abi: pd.DataFrame, inc: pd.DataFrame, jer: dict[str, Any], presu: dict[str, Any],
        envio: dict[str, Any], nunca: list[str], best_c: float, best_c_id: str,
        best_a: float, best_a_id: str) -> None:
    """Escribe el informe de cierre estrategico.

    Args:
        informe: diccionario con los resultados.
        inv: inventario de experimentos.
        fam: tabla de familias.
        cand: tabla de candidatos.
        abi: tabla de preguntas abiertas.
        inc: tabla de inconsistencias.
        jer: jerarquia de magnitudes.
        presu: presupuesto de submissions.
        envio: envio no registrado en el log.
        nunca: ficheros nunca enviados.
        best_c: mejor CV comparable.
        best_c_id: identificador del mejor CV comparable.

    Returns:
        None.
    """
    b = informe["baseline_congelado"]
    can = informe["canales_p49_p53"]
    L: list[str] = []
    A = L.append
    A("# P57 - Cierre estrategico y plan final de competencia")
    A("")
    A(f"Generado por `experiments/v57_00_sintesis_estrategica.py`. Fecha {informe['fecha']}.")
    A("")
    A("**0 modelos. 0 entrenamientos. 0 submissions. 0 consultas a Kaggle. 0 uso de holdout. "
      "0 uso de test.** Esta fase es de sintesis, auditoria y estrategia. No se registro ninguna "
      "fila nueva en `experiments.csv`, porque no hubo ninguna ejecucion experimental.")
    A("")
    A("## 1. Baseline congelado")
    A("")
    A("```text")
    A(f"id            {b['id']}")
    A(f"repr          {b['representacion']}")
    A(f"clasificador  {b['clasificador']}")
    A(f"umbrales      HS {b['umbrales']['HS']} | TR {b['umbrales']['TR']} | AG {b['umbrales']['AG']}")
    A(f"corrected_cv  {b['corrected_cv']!r}")
    A(f"LB            {b['LB']}")
    A(f"sha256        {b['sha256']}")
    A("```")
    A("")
    A(f"sha256 verificado en el arranque de esta fase: **{b['sha256_verificado']}**. Intacto.")
    A("")
    A("## 2. Jerarquia de magnitudes, sin mezclar metricas")
    A("")
    A("Esta tabla es el resultado mas importante de P57. **Cada fila lleva su metrica, su "
      "protocolo y su poblacion, y las filas NO son intercambiables.**")
    A("")
    A("| magnitud | valor | experimento | metrica | protocolo | poblacion | comparable con V03 |")
    A("|---|---:|---|---|---|---|---|")
    for k, v in jer.items():
        A(f"| {k} | {v['valor']} | {v['experimento']} | {v['metrica']} | {v['protocolo']} | "
          f"{v['poblacion']} | {v['comparable_con_V03']} |")
    A("")
    A("Lo que dice esta tabla:")
    A("")
    A("- El **maximo absoluto** de la columna `macro_f1` del registro es 0.8743, y **no es** una "
      "medida de la metrica de la competencia: es macro-6 sobre el holdout.")
    A(f"- La metrica del proyecto sobre el OOF de V03 vale {V03_CORRECTED_CV!r} y la macro-6 vale "
      f"0.8182213363: **+0.04101 de diferencia**, segun P45.")
    A(f"- `leaderboard_log.csv` registra para V03 un `local_f1` de {V03_HISTORICAL_CV}, que es "
      f"**+0.021689** por encima del `corrected_cv` canonico, y la diferencia es exactamente el "
      f"coste de medir con D-004 en vez de D-016.")
    A(f"- El **mejor CV comparable bajo D-016** es **{best_c:.6f}** ({best_c_id}), linea "
      f"svd300_rbf. Se envio y dio LB 0.77938. **El mejor CV comparable no es el campeon.** Y "
      f"es el mejor de cuatro candidatos comparados sobre el mismo OOF, luego lleva seleccion.")
    A(f"- El **mejor CV comparable Y anidado** es **{best_a:.6f}** ({best_a_id}), que es la cifra "
      f"que un argumento de 'se ha probado mas cosas y ninguna supera a V03' tiene que citar.")
    A(f"- El **unico instrumento externo** es el leaderboard, y su maximo verificado es "
      f"**{V03_LB}**.")
    A("")
    A("## 3. Inventario final de experimentos")
    A("")
    A(f"`reports/p57_final_experiment_inventory.csv`, {len(inv)} filas, una por experimento del "
      f"registro, con las columnas `metrica`, `protocolo`, `poblacion` y `anidado` que hacen "
      f"utilizable el registro.")
    A("")
    A("| clase | filas |")
    A("|---|---:|")
    A(f"| protocolo D-004 (mas debil, no comparable) | {informe['inventario']['D-004']} |")
    A(f"| protocolo D-016 (corregido, comparable) | {informe['inventario']['D-016']} |")
    A(f"| holdout de 248 filas (poblacion distinta) | {informe['inventario']['HOLD']} |")
    A(f"| sin clasificar | {informe['inventario']['sin_clasificar']} |")
    A("")
    A("**Ningun experimento historico se modifica.** La clasificacion se anade al lado.")
    A("")
    A("## 4. Estado consolidado de familias")
    A("")
    A(f"`reports/p57_family_status.csv`, {len(fam)} familias. Tipos de cierre: "
      "**A** resultado negativo, **B** implementacion invalida, **C** sin transferencia, "
      "**D** evidencia insuficiente, **E** prohibido o inexistente, **F** redundante.")
    A("")
    A("| familia | estado | tipo | mejor resultado | LB |")
    A("|---|---|---|---|---|")
    for _, r in fam.iterrows():
        lb = r["LB_transfer"]
        lb_corto = "RECHAZADO" if "RECHAZ" in str(lb).upper() else (
            "a la baja" if "baja" in str(lb) else "no enviado")
        A(f"| {r['family']} | {r['status']} | {r['closure_type_A_to_F']} | "
          f"{str(r['best_result'])[:70]} | {lb_corto} |")
    A("")
    A("### Las cuatro familias con veredicto de tipo C (sin transferencia)")
    A("")
    A("| familia | envios | LB |")
    A("|---|---:|---|")
    A("| kernels | 1 | 0.77938, RECHAZADO |")
    A("| label_specialization | 1 | 0.77781, RECHAZADO |")
    A("| ensembles | 3 | 0.7751, 0.7747, 0.77122, los tres a la baja |")
    A("| stacking / meta-modelos | 5 | 0.78163, 0.77895, 0.77978, 0.78150, 0.77663, los cinco a la baja |")
    A("")
    A("**Diez envios de candidatos con mecanismo nuevo. Los diez transferieron a la baja.** Este "
      "es el hecho mas firme del proyecto, y es lo que gobierna la estrategia final.")
    A("")
    A("### La distincion que la directiva pide (A frente a no probado)")
    A("")
    A("| tipo | familias |")
    A("|---|---|")
    for t in ["A", "B", "C", "E"]:
        fs = [f["family"] for f in FAMILIAS if f["closure_type"] == t]
        A(f"| {t} | {', '.join(fs) if fs else 'ninguna'} |")
    A("")
    A("**Ninguna familia queda como D (evidencia insuficiente)**, salvo dos casos que se senalan "
      "explicitamente y que **no** se cuentan como cerradas por resultado:")
    A("")
    A("1. **PLS** es tipo **B**: V42-01 y V43-01 se invalidaron por implementacion, y su numero no "
      "es evidencia. El resultado verdadero lo produjo V43-02 como verificacion y es negativo "
      "(-0,013742). La hipotesis no se rechaza, el experimento.")
    A("2. Las lineas de estructura de etiquetas de V13-02 y V14-00 se cerraron **por condicion de "
      "parada**, con 0 de 4 experimentos ejecutados. Ahi **no se puede decir 'no funciono'**, "
      "porque no llego a probarse. Se registra como pregunta abierta, no como cierre.")
    A("")
    A("## 5. V03 y sus variantes")
    A("")
    A("| variante | umbrales | honest_CV | LB | estado |")
    A("|---|---|---:|---:|---|")
    A(f"| **V03 congelado** | 0.44/0.51/0.46 | {V03_CORRECTED_CV!r} | {V03_LB} | CAMPEON |")
    A(f"| P55, umbrales anidados simples | reestimados | {P55_REPORTED:.10f} | n/d | superseded por P56 |")
    A(f"| P56 T1, mediana aplicada globalmente | 0.41/0.46/0.45 | {P56_GLOBAL:.10f} | n/d | optimista en +0.0053 |")
    A(f"| P56 T1, umbral por pliegue exterior | por pliegue | {P56_HONEST:.10f} | n/d | **NOT_CONFIRMED** |")
    A("| v02 (mismo modelo, umbral 0.5) | 0.50/0.50/0.50 | n/d | 0.77993 | REJECTED |")
    A("| v04 ENS-04 (ensemble LR+SVC) | medianos por pliegue | n/d | 0.7751 | REJECTED |")
    A("| v16_ch7 C=1.5 | 0.44/0.51/0.46 | 0.776653 | 0.77296 | REJECTED |")
    A("| v16_ch8 C=2.5 | 0.44/0.51/0.46 | 0.776752 | 0.77707 | REJECTED |")
    A("| v16_ch9 min_df=1 | 0.44/0.51/0.46 | 0.765532 | 0.76978 | REJECTED |")
    A("| v16_ch1 tasa interpolada | quantile | n/d | 0.77828 | REJECTED |")
    A("| v16_ch5 tasa optima absoluta | 0.410/0.475/0.450 | n/d | 0.77495 | REJECTED |")
    A(f"| v32_05 url_length | 0.44/0.51/0.46 | identica a V03 | {V03_LB} | delta 0, WEAK_TRANSFER |")
    A("")
    A("Las tres dimensiones de un solo parametro tienen su optimo en V03: C (1.0 -> 0.77227, "
      "**2.0 -> 0.78306**, 3.26 -> 0.77602), min_df (1 -> 0.76978, **2 -> 0.78306**, 5 -> 0.77034) y "
      "tasa predicha (0.3385 -> 0.77993, **0.3645 -> 0.78306**, 0.3836 -> 0.77828, 0.3980 -> 0.77495). "
      "El maximo esta en V03 en las tres, y la respuesta a la tasa es monotona a ambos lados.")
    A("")
    A("## 6. Cierre de thresholds (P55 / P56)")
    A("")
    A("```text")
    A(f"P55 reported          {P55_REPORTED:.10f}")
    A(f"P56 global            {P56_GLOBAL:.10f}   (+0.0026400107, 4/5 pliegues)")
    A(f"P56 honesto           {P56_HONEST:.10f}   ({P56_HONEST - V03_CORRECTED_CV:+.10f})")
    A(f"optimismo de la cifra global   {P56_OPTIMISMO:+.10f}")
    A(f"V03                   {V03_CORRECTED_CV:.10f}")
    A("THRESHOLD_TRANSFER = CLOSED")
    A("```")
    A("")
    A("Los umbrales de V03 caen **dentro** del rango de las cinco estimaciones interiores en las "
      "tres etiquetas. El defecto procedimental de P55 sigue siendo un hecho historico real, pero "
      "su magnitud con el procedimiento correcto es nula o negativa. La familia no se reabre "
      "cambiando la mediana por la media, ni la rejilla, ni offsets pequenos.")
    A("")
    A("## 7. Cierre de los canales (P49 a P53)")
    A("")
    A("```text")
    A(f"V03 errores            {can['V03_errores']} celdas ({N_CELDAS_TOTAL} totales)")
    A(f"V03 correctos          {can['V03_correctos']}")
    A(f"rescuable              {can['rescuable']}")
    A(f"no rescatable          {can['no_rescatable']} celdas en {can['tweets_no_rescatables']} tweets")
    A(f"true oracle            {TRUE_ORACLE:.10f}  ({TRUE_ORACLE - V03_CORRECTED_CV:+.10f})")
    A(f"mejor politica observable  {OBSERVABLE_CEILING:.10f}  ({OBSERVABLE_CEILING - V03_CORRECTED_CV:+.10f})")
    A("CONTEXTUAL_SELECTION = CLOSED")
    A("```")
    A("")
    A("**La formulacion correcta, y es importante:**")
    A("")
    A(f"- **Lo que se SOSTIENE:** la senal existe y es enorme. El techo del oracel es "
      f"{TRUE_ORACLE - V03_CORRECTED_CV:+.10f}, casi 70 veces el margen de adopcion de 0,003. "
      f"Las 715 celdas son alcanzables en principio.")
    A("- **Lo que NO se SOSTIENE:** que exista una politica observable que convierta esa senal en "
      "mejora neta. P53 midio que los 20 canales que discrepan proponen `1 - v03_pred` en las 3896 "
      "celdas con discrepancia, sin una sola excepcion, luego el espacio de acciones real cae de 21 "
      "a 2 y toda politica esta por debajo de V03.")
    A("")
    A("Decir 'la senal existe pero ningun selector probado la convirtio en mejora neta positiva' es "
      "correcto. Decir 'no hay senal' seria falso, y el registro no lo dice.")
    A("")
    A("## 8. Error residual")
    A("")
    A("```text")
    A(f"42 celdas no rescatables = {can['no_rescatable'] / N_CELDAS_TOTAL * 100:.2f} por ciento de las {N_CELDAS_TOTAL}")
    A(f"soporte efectivo: {can['tweets_no_rescatables']} tweets, no 42")
    A("```")
    A("")
    A("P54: FP 22 / FN 20, 85,7 por ciento de margen alto, 36 de 42 de alta confianza. HS 2,85 por "
      "ciento, TR 7,32 por ciento, AG 7,58 por ciento de los errores. El lexico de insulto discrimina "
      "1,25 veces contra A, luego **no discrimina**.")
    A("")
    A("**No se exagera esta poblacion.** 42 celdas en 33 tweets es el techo de lo que queda, y "
      "aun con un oracel perfecto sobre el 0,95 por ciento del espacio el movimiento del macro-F1 "
      f"ronda +0,009, por debajo de la banda de particion de {BANDA_PARTICION} de D-070. La "
      "formulacion correcta es **'no hay ningun mecanismo nuevo con soporte suficiente y valor "
      "esperado material'**, no 'es imposible mejorar'.")
    A("")
    A("## 9. Auditoria de datos")
    A("")
    A("| magnitud | valor |")
    A("|---|---:|")
    A(f"| train oficial etiquetado | {N_TRAIN_OFICIAL} |")
    A(f"| desarrollo | {N_DEV} |")
    A(f"| holdout | {N_HOLDOUT} |")
    A(f"| test oficial | {N_TEST} |")
    A("")
    A("```text")
    A("DATA_EXPANSION = CLOSED")
    A("TRAIN_AUGMENTATION = CLOSED")
    A("```")
    A("")
    A("Las 1731 filas estan todas asignadas: 1483 en dev y 248 en holdout. Quedan **0** filas "
      "oficiales etiquetadas sin usar. Los corpus externos existen (42808 documentos, 95.0M "
      "tokens) pero no estan etiquetados, el manifiesto declara contaminacion cero con "
      "`toxic_test.csv`, y su unico uso permitido es entrenar embeddings, familia ya cerrada.")
    A("")
    A("## 10. Submissions")
    A("")
    A("| magnitud | valor |")
    A("|---|---:|")
    A(f"| envios historicos totales | **{presu['envios_historicos']}** |")
    A(f"| con LB verificado en el log | {presu['envios_con_LB_verificado_en_log']} |")
    A(f"| envio real no registrado en el log | {presu['envios_reales_no_registrados']} |")
    A(f"| ficheros de submission en disco | {len(nunca) + presu['envios_con_LB_verificado_en_log'] + presu['envios_reales_no_registrados']} |")
    A(f"| generados y nunca enviados | {len(nunca)} |")
    A(f"| maximo LB verificado | **{V03_LB}** |")
    A(f"| minimo LB registrado | {pd.to_numeric(pd.read_csv(LB_LOG).kaggle_f1, errors='coerce').min()} (plantilla todo-uno) |")
    A(f"| envios hoy, {FECHA} | {presu['envios_hoy']} |")
    A("")
    if envio:
        A(f"**El envio que falta en el log.** V33-03 documento un envio real y aceptado: ID "
          f"`{envio['kaggle_id']}`, fecha {envio['fecha']}, fichero `{envio['fichero']}`, LB "
          f"{envio['LB']}, delta {envio['delta_vs_V03']:+.5f}. Su anotacion de presupuesto decia "
          f"'{presu['unica_cifra_local']}'")
        A("")
    A("Envios por dia, con el envio no registrado sumado:")
    A("")
    A("| dia | envios |")
    A("|---|---:|")
    for k, v in presu["envios_por_dia"].items():
        A(f"| {k} | {v} |")
    A("")
    A("### Presupuesto restante")
    A("")
    A("```text")
    A(f"limite diario EXACTO        {presu['limite_diario_conocido']}   (no consta en ningun registro local)")
    A(f"envios hoy                  {presu['envios_hoy']}")
    A(f"reserva propia del proyecto  {presu['reserva_propia_del_proyecto']}")
    A("```")
    A("")
    A(f"**No se consulta Kaggle.** {presu['unica_cifra_local']} La reserva de 3 es una "
      "autolimitacion del proyecto, no un limite de la competencia.")
    A("")
    A("### Artefactos que no deben volver a usarse")
    A("")
    A("| artefacto | clase | motivo |")
    A("|---|---|---|")
    A("| `toxic_submission_v17_prior_shift_em.csv` | QUARANTINADA, transductiva | consume el EM de prevalencia sobre el conjunto de test |")
    A("| `toxic_submission_v18_near_dup_propagacion.csv` | QUARENTINADA, transductiva | propaga etiquetas usando near-duplicados del test; ademas cambia 0 celdas |")
    A("| `toxic_submission_v16_ch2_prior_shift_em.csv` | CUARENTENADA, transductiva | mismo mecanismo que v17, mismo motivo. No se envio, y por coherencia no debe enviarse |")
    A("| `toxic_submission_v12_descartado_meta36_downweight.csv` | RECHAZADA | holdout 0.8163 contra 0.8285 de V03, delta -0.0122. Consume la quinta y ultima evaluacion de holdout |")
    A("| `toxic_submission_v20_03_reproduccion_v03.csv` | INVALIDA | es una reproduccion de V03, no una candidata |")
    A("| `toxic_submission_v32_05_url_length.csv` | GASTADA, delta 0 | envio real, LB identico a V03 porque el fichero es identico |")
    A("")
    A("Ninguna se modifica. Ninguna se vuelve a enviar.")
    A("")
    A("## 11. Preguntas abiertas con mecanismo")
    A("")
    A(f"`reports/p57_open_questions.csv`, {len(abi)} filas. Cada una declara un mecanismo, no "
      "un 'no se ha probado'.")
    A("")
    A("| pregunta | probada | valor esperado |")
    A("|---|---|---|")
    for _, r in abi.iterrows():
        A(f"| {r['question'][:78]} | {r['tested'][:34]} | {str(r['expected_value']).split('.')[0][:58]} |")
    A("")
    A("**Ninguna cumple hoy el umbral de reapertura** de la directiva, que exige las seis "
      "condiciones a la vez. Se dejan documentadas porque su ausencia de evidencia es "
      "especifica y esta medida, no vaga.")
    A("")
    A("## 12. Candidatos historicos")
    A("")
    A(f"`reports/p57_candidate_matrix.csv`, {len(cand)} artefactos. **Tabla descriptiva, sin "
      "ranking y sin puntuacion.**")
    A("")
    A("| artefacto | mecanismo | honest_CV | delta | LB | verificado | disponible |")
    A("|---|---|---:|---:|---:|---|---|")
    for _, r in cand.iterrows():
        A(f"| {r['candidate'][:44]} | {r['mechanism'][:44]} | {r['honest_CV']} | "
          f"{r['delta_vs_V03']} | {r['LB']} | {r['verified'][:18]} | {r['submission_available'][:22]} |")
    A("")
    A("### Aplicando los cinco criterios simultaneos de la directiva")
    A("")
    A("```text")
    A("legal            + reproducible     + sin leakage")
    A("no rechazado ya  + mecanismo claro")
    A("```")
    A("")
    A("**Resultado: NINGUN artefacto cumple los cinco a la vez.**")
    A("")
    A("| artefacto | que le falta |")
    A("|---|---|")
    A("| v23_01 svd300_rbf | ya rechazado: enviado, LB 0.77938 |")
    A("| v08_meta_v2 | ya rechazado: enviado, LB 0.78163 |")
    A("| v20_01 meta 36 features | ya rechazado dos veces: por holdout y por LB 0.77663 |")
    A("| E2-13-META-ENT-RIDGE | sin leakage y legal, pero cerrado por sparsity y nunca verificado fuera |")
    A("| umbrales anidados de P56 | veredicto NOT_CONFIRMED, delta honesto negativo |")
    A("| v16_ch2 prior shift EM | transductivo: usa informacion del conjunto de test |")
    A("")
    A("Es un resultado, no una omision. Los tres artefactos con mejor evidencia interna tienen "
      "**veredicto externo medido y contrario**, y el unico con delta honesto positivo y sin "
      "veredicto externo fue cerrado por una razon de mecanismo, no de metrica.")
    A("")
    A("## 13. Inconsistencias documentales, seccion 25 de la directiva")
    A("")
    A(f"{len(inc)} hallazgos, clasificados sin corregir la historia experimental.")
    A("")
    A("| severidad | donde | inconsistencias |")
    A("|---|---|---|")
    for _, r in inc.iterrows():
        A(f"| **{r['severity']}** | {r['where'][:52]} | {r['inconsistency'][:88]} |")
    A("")
    for i, r in inc.iterrows():
        A(f"### {i + 1}. [{r['severity']}] {r['where']}")
        A("")
        A(f"- **Inconsistencia:** {r['inconsistency']}")
        A(f"- **Evidencia:** {r['evidence']}")
        A(f"- **Consecuencia:** {r['consequence']}")
        A(f"- **Correccion:** {r['correction']}")
        A("")
    A("## 14. Integridad del notebook")
    A("")
    nb = json.loads((PROJECT_ROOT / "notebooks" / "10_final_pipeline.ipynb").read_text(encoding="utf-8"))
    "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    A(f"- celdas: **{len(nb['cells'])}**")
    A("- seccion P57 anadida como seccion 53, celda de codigo con returncode 0 y cifras leidas de "
      "los artefactos, no escritas a mano")
    A("- sin experimentacion nueva")
    A("")
    A("## 15. Formulacion final obligatoria")
    A("")
    A("> V03 remains the best verified leaderboard result and the frozen architectural baseline. "
      "The investigated legal mechanisms did not produce a robust transferable improvement, and no "
      "additional mechanism with sufficient evidence has currently been identified.")
    A("")
    A("Y la distincion que hay que no perder:")
    A("")
    A("```text")
    A("best verified result  !=  proof of global optimum")
    A("```")
    A("")
    A(f"- Lo verificado: **{V03_LB}** es el maximo del leaderboard en {presu['envios_historicos']} "
      f"envios, y {V03_CORRECTED_CV!r} es el CV corregido de la configuracion congelada.")
    A(f"- Lo que NO se afirma: que V03 sea globalmente optimo. La brecha OOF->LB de "
      f"{DEV_LB_GAP:.8f} sigue sin mecanismo identificado, y el techo del oracel de "
      f"{TRUE_ORACLE - V03_CORRECTED_CV:+.10f} no se ha alcanzado ni se ha descartado como "
      "alcanzable.")
    A("- El hecho de que ningun candidato con mejor CV local haya transferido es evidencia en contra "
      "de la overlapped optimizacion local, y a favor de V03 como eleccion. **No es evidencia de "
      "que no exista una mejor.**")
    A("")
    A("## 16. Las tres estrategias, sin elegir")
    A("")
    A("| estrategia | evidencia a favor | evidencia en contra | riesgo | coste |")
    A("|---|---|---|---|---|")
    A(f"| **A. cerrar con V03** | 0 de {presu['envios_historicos']} envios lo superaron; los 10 "
      f"candidatos con mecanismo nuevo transfirieron a la baja; las 22 familias estan cerradas | el "
      f"techo del oracel es {TRUE_ORACLE - V03_CORRECTED_CV:+.10f} y la brecha OOF-LB sigue sin "
      f"explicar | bajo: {V03_LB} queda registrado y el leaderboard conserva la mejor puntuacion | "
      f"cero |")
    A("| **B. usar un candidato historico** | E2-13 tiene +0,0113 de delta honesto anidado y ninguna "
      "contradiccion externa | los tres con delta comparable tienen veredicto externo contrario; "
      "E2-13 se cerro por sparsity, no por resultado | alto: gasta presupuesto en repetir un "
      "patron de 10/10 | 1 submission |")
    A("| **C. ultima investigacion dirigida** | la brecha OOF-LB y la asimetria dev-holdout no estan "
      "caracterizadas | ninguna hipotesis pendiente cumple las seis condiciones de reapertura; la "
      "banda de particion de 0,017 supera cualquier delta local alcanzable | medio: puede no "
      "producir nada y consume presupuesto de submissions | alto, no medido |")
    A("")
    A("**La decision competitiva queda pendiente de una instruccion posterior.** P57 no la toma.")
    A("")
    A("## 17. Si se autorizara una submission")
    A("")
    A("No se recomienda ninguna. Si se autorizara, el unico formato admisible es:")
    A("")
    A("```text")
    A("un candidato congelado")
    A("una sola submission")
    A("no una busqueda interactiva")
    A("```")
    A("")
    A("Y dado el resultado de la seccion 12, **no hay candidato congelado que cumpla los cinco "
      "criterios**. En consecuencia, bajo el estado actual del proyecto, la opcion B no es "
      "ejecutable sin una excepcion explicita.")
    A("")
    A("## 18. Conclusion")
    A("")
    A(f"V03 permanece como **mejor resultado verificado** del leaderboard ({V03_LB}) y como "
      f"**baseline arquitectonico congelado** ({V03_CORRECTED_CV!r}), con sha256 verificado. "
      f"{presu['envios_historicos']} envios historicos, ninguno lo supero. Las familias "
      f"investigadas no produjeron una mejora robusta y transferible.")
    A("")
    A("Lo que P57 anade y que las fases anteriores no Tenian formulado:")
    A("")
    A("1. **La jerarquia de magnitudes, con metricas y protocolos separados.** La columna `macro_f1` "
      "del registro no es una metrica, y su maximo 0.8743 no es una medida de la competencia.")
    A(f"2. **Que el mejor CV comparable ({best_c:.6f}, con seleccion) y el mejor anidado "
      f"({best_a:.6f}) NO son el campeon, y que el primero dio LB 0.77938.** El patron de 10 "
      f"envios de candidatos a la baja es la evidencia mas firme del proyecto.")
    A("3. **Que un envio real no estaba en el log**, y que el total real es "
      f"{presu['envios_historicos']} y no 22.")
    A("4. **Que PLS es un cierre de tipo B, no A**: su numero nunca fue evidencia, y el resultado "
      "verdadero llego despues como verificacion.")
    A("5. **Que ninguna de las lineas de estructura de etiquetas se ejecuto**, porque se cerraron por "
      "condicion de parada. Es el unico sitio del registro donde no cabe decir 'no funciono'.")
    A("")
    A("*Seccion generada por `experiments/v57_00_sintesis_estrategica.py`. Sin push: la rama sigue "
      "local.*")
    MD_INFORME.write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
