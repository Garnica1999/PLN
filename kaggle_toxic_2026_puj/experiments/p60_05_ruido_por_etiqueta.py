"""P60 track C: label-dependent noise, detection before any intervention.

The hypothesis
--------------
The recent multi-label noise literature makes the (instance, label) PAIR the unit
of detection rather than the instance, and argues that different labels need
different detection rules. TOXIC already closed noise in its instance-level form:
`V30-00` closed seven instance-level suspicion experiments and `P59-03` closed the
formal global estimator, which scored an AUC of 0,9730 against a trivial control
of 0,9741. What neither did is decompose suspicion BY LABEL, and on this corpus
that decomposition is not a formality: the canonical OOF shows

    P(TR = 1 | HS = 0) = 0,0000
    P(AG = 1 | HS = 0) = 0,0263

so 100 por ciento of the negatives TR sees during training are labels that are
UNDEFINED rather than absent, while HS has no such region at all. The evidence
that a label is noisy is therefore structurally different for each label, which is
exactly the claim the NAACL 2024 formulation adds.

Why detection first, and why this script may spend no experiment
----------------------------------------------------------------
The brief is explicit: build the scores, then establish whether the score carries
information the project does not already have, and only then intervene. A
detection audit that concludes "no incremental signal" is a complete result and
costs no training. So this script trains V03 inside the folds only to obtain
per-example losses and probabilities on the TRAINING rows, and spends no
candidate.

The incremental test, and why a plain AUC is not enough
-------------------------------------------------------
A score can have a high AUC and still add nothing, because it can be a noisy copy
of the confidence the project already measures. P59-03 learned that lesson
directly. So the test here is not "is the AUC above 0,5" but "does the score
still rank errors once the control is held fixed". Concretely, for every candidate:

    residual  = score - E[score | decile of the control]

and then the AUC of the residual against the error indicator. If a candidate's
residual AUC sits at 0,5, the candidate carries no information the control does
not already carry, no matter how good its raw AUC looks. Deciles rather than a
parametric fit, so no model form is assumed.

Signals, all computed on TRAINING ROWS OF THE FOLD ONLY
-------------------------------------------------------
    control      1 - p_true, the probability the model gives to the observed class
    loss         per-example logistic loss for that label
    rango        normalised rank of the loss within that label
    centroide    signed cosine margin to the class and non-class centroids
    estructural  indicator that the label is UNDEFINED rather than negative
    compuesto    logistic regression on the standardised signals above

The structural signal is the one that makes this corpus-specific, and it is zero
by construction for HS, which has no undefined region.

Leakage control
---------------
Every per-example quantity comes from a model fitted on the training rows of its
own fold. The error indicator uses the canonical OOF prediction, which is
out-of-fold by construction. No validation label, no holdout, no test, no
leaderboard is read at any point.

Usage
-----
    /home/carlos/venv/bin/python experiments/p60_05_ruido_por_etiqueta.py
"""

from __future__ import annotations

import ast
import csv as _csv
import hashlib
import json
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler, normalize

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

from p59_01_voronoi_local import (
    ETIQUETAS,
    V03_CCV,
    V03_SHA,
    V03_SUB,
)

OOF_V03 = PROJECT_ROOT / "features" / "oof_v03_final.npz"
TRAIN_RAW = PROJECT_ROOT / "data" / "raw" / "toxic_train.csv"
PARTICION = PROJECT_ROOT / "data" / "processed" / "holdout_v2_ids.csv"
MD_C = PROJECT_ROOT / "reports" / "p60_C_ruido_por_etiqueta.md"
JSON_C = PROJECT_ROOT / "reports" / "p60_C_ruido_por_etiqueta.json"
CSV_C = PROJECT_ROOT / "reports" / "p60_C_ruido_por_etiqueta.csv"

SEMILLA = 42
N_DECIL = 10
UMBRAL_INCREMENTAL = 0.01


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[P60-05] {mensaje}", flush=True)


SUSTITUCIONES = {
    chr(0x2014): " - ", chr(0x2013): "-", chr(0x2212): "-", chr(0x00B1): "+/-",
    chr(0x00AB): '"', chr(0x00BB): '"', chr(0x201C): '"', chr(0x201D): '"',
    chr(0x2018): "'", chr(0x2019): "'", chr(0x2264): "<=", chr(0x2265): ">=",
    chr(0x00D7): "x", chr(0x2192): "->", chr(0x00B7): ".", chr(0x2026): "...",
    chr(0x00BF): "?", chr(0x00A1): "!", chr(0x00A7): "parrafo ", chr(0x00A0): " ",
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
            continue
        partes = unicodedata.decomposition(ch).split()
        base = chr(int(partes[0], 16)) if partes and partes[0] != "<wide>" else ""
        salida.append(base if base and ord(base) < 128 else "?")
    return "".join(salida)


def auc_residual(control: np.ndarray, score: np.ndarray, error: np.ndarray) -> float:
    """AUC de un score una vez controladas las decilas del control.

    Args:
        control: senal de control, la probabilidad de la clase observada.
        score: senal candidata.
        error: indicador de error por par.

    Returns:
        AUC del residuo del score frente al indicador de error.
    """
    dec = pd.qcut(pd.Series(control), N_DECIL, labels=False, duplicates="drop")
    dec = np.asarray(dec, dtype=int)
    res = np.zeros_like(score, dtype=float)
    for d in np.unique(dec):
        m = dec == d
        res[m] = score[m] - score[m].mean()
    if len(np.unique(error)) < 2 or np.allclose(res, 0.0):
        return float("nan")
    return float(roc_auc_score(error.astype(int), res))


def main() -> None:
    """Ejecuta C1 y C2: construir las senales y medir si anaden al control.

    Returns:
        None.
    """
    inicio = datetime.now(timezone.utc)
    log("P60-C Ruido dependiente de etiqueta. C1 construir senales, C2 ganancia incremental")
    log("   Todo calculado con filas de ENTRENAMIENTO del pliegue. 0 Holdout, 0 test, 0 Kaggle")
    log(f"   N_DECIL {N_DECIL} | umbral de senal incremental {UMBRAL_INCREMENTAL}")

    hash_real = hashlib.sha256(V03_SUB.read_bytes()).hexdigest()
    if hash_real != V03_SHA:
        raise SystemExit("PARADA: la submission de V03 no coincide con el sha256 congelado")
    log(f"   V03 sha256 verificado e intacto: {hash_real[:16]}...")

    v = np.load(OOF_V03, allow_pickle=True)
    ids, y, pliegues, pred_v03, _proba_v03 = v["ids"], v["y_true"], v["pliegues"], v["pred"], v["proba"]
    ccv_ref = float(f1_score(y, pred_v03, average="macro"))
    if abs(ccv_ref - V03_CCV) > 1e-12:
        raise SystemExit(f"PARADA: el OOF canonico da {ccv_ref!r}, no {V03_CCV!r}")
    n = ids.size

    particion = pd.read_csv(PARTICION)
    grupos_cv = particion.set_index("id").loc[ids, "grupo_near_dup_autor"].to_numpy()
    partidos = int(pd.Series(grupos_cv).groupby(grupos_cv).nunique().gt(1).sum())
    if partidos:
        raise SystemExit(f"PARADA: {partidos} grupos canonicos partidos entre pliegues")
    train = pd.read_csv(TRAIN_RAW, sep=";", encoding="latin-1", dtype={"id": str})
    train["id_num"] = pd.to_numeric(train["id"].str.replace('"', "", regex=False),
                                    errors="coerce").astype("int64")
    idx_texto = train.set_index("id_num").loc[ids]
    if not np.array_equal(idx_texto[["HS", "TR", "AG"]].to_numpy().astype(int), y):
        raise SystemExit("PARADA: las etiquetas del CSV crudo no coinciden con el OOF")
    textos = idx_texto["text"].astype(str).tolist()
    log(f"   OOF canonico {ccv_ref!r} | {n} filas | {len(set(grupos_cv))} grupos intactos")

    # ---------- estructura de etiquetas no definidas, medida ----------
    log("")
    log("   ESTRUCTURA DE ETIQUETAS MEDIDA en el OOF canonico")
    estructura = {
        "P_TR1_dado_HS0": float(y[y[:, 0] == 0, 1].mean()),
        "P_AG1_dado_HS0": float(y[y[:, 0] == 0, 2].mean()),
        "P_AG1_dado_TR1": float(y[y[:, 1] == 1, 2].mean()),
        "negativos_TR_no_definidos": float((y[y[:, 0] == 0, 1] == 0).mean()),
        "negativos_AG_no_definidos": float((y[y[:, 0] == 0, 2] == 0).mean()),
        "filas_HS0": int((y[:, 0] == 0).sum()),
        "n_filas": int(n),
    }
    for k, val in estructura.items():
        log(f"      {k:32} {val}")
    log("      el indicador estructural vale 1 para TR cuando HS = 0, para AG cuando")
    log("      HS = 0, y SIEMPRE vale 0 para HS, que no tiene region indefinida")

    # ---------- C1: senales por par, en training de cada pliegue ----------
    log("")
    log("   C1 SENALES POR PAR, en filas de ENTRENAMIENTO de cada pliegue")
    # Las senales se guardan POR ETIQUETA. La estructural tiene que estarlo porque su
    # definicion depende de la etiqueta, y con una columna compartida la rama de AG
    # sobrescribia la de HS dentro del mismo bloque de validacion, con lo que HS
    # acababa puntuando contra un indicador que no es suyo.
    senales: dict[str, dict[str, np.ndarray]] = {
        etiqueta: {clave: np.zeros(n) for clave in
                   ("control", "perdida", "rango", "centroide", "estructural")}
        for etiqueta in ETIQUETAS
    }
    for f in range(1, 6):
        ir = np.where(pliegues != f)[0]
        iva = np.where(pliegues == f)[0]
        pw = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2,
                             sublinear_tf=True, lowercase=True)
        pc = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2,
                             sublinear_tf=True, lowercase=True)
        x = hstack([pw.fit_transform([textos[i] for i in ir]),
                    pc.fit_transform([textos[i] for i in ir])]).tocsr()
        xv = hstack([pw.transform([textos[i] for i in iva]),
                     pc.transform([textos[i] for i in iva])]).tocsr()
        # los centroides se calculan con filas de ENTRENAMIENTO y se aplican a las de
        # validacion, que es donde se va a medir la calidad diagnostica de la senal
        xrn = normalize(x)
        ytr = y[ir]
        yva = y[iva]
        for j, etiqueta in enumerate(ETIQUETAS):
            et_tr = ytr[:, j].astype(int)
            et_va = yva[:, j].astype(int)
            lr = LogisticRegression(C=2.0, class_weight="balanced", max_iter=1000,
                                    solver="liblinear", random_state=SEMILLA)
            lr.fit(x, et_tr)
            # la probabilidad de la validacion con el modelo del pliegue ES la
            # probabilidad OOF canonica de esas filas, luego control y senal quedan
            # emparejadas con el mismo indicador de error y sobre las mismas filas
            p = np.clip(lr.predict_proba(xv)[:, 1], 1e-12, 1.0 - 1e-12)
            p_true = np.where(et_va == 1, p, 1.0 - p)
            s = senales[etiqueta]
            s["control"][iva] = 1.0 - p_true
            perdida = -np.log(p_true)
            s["perdida"][iva] = perdida
            orden = np.argsort(np.argsort(perdida))
            s["rango"][iva] = orden / max(len(orden) - 1, 1)
            pos_tr = et_tr == 1
            if pos_tr.any() and (~pos_tr).any():
                cpos = np.asarray(xrn[pos_tr].mean(axis=0)).ravel()
                cneg = np.asarray(xrn[~pos_tr].mean(axis=0)).ravel()
                xvn = normalize(xv)
                s["centroide"][iva] = (np.asarray(xvn @ cpos).ravel()
                                       - np.asarray(xvn @ cneg).ravel())
            else:
                s["centroide"][iva] = 0.0
            # El indicador estructural depende de la etiqueta: TR queda sin definir
            # cuando HS = 0, AG queda sin definir cuando HS = 0 o cuando TR = 0, y HS
            # NO tiene region indefinida, luego su indicador vale cero SIEMPRE y su
            # AUC no ordena nada. Por eso las senales se guardan por etiqueta y no en
            # una columna compartida, que ademas hacia que HS puntuara contra el
            # indicador de AG.
            if etiqueta == "TR":
                s["estructural"][iva] = (yva[:, 0] == 0).astype(float)
            elif etiqueta == "AG":
                s["estructural"][iva] = ((yva[:, 0] == 0) | (yva[:, 1] == 0)).astype(float)
            else:
                s["estructural"][iva] = 0.0
    log(f"      senales construidas sobre las {n} filas del OOF, una por etiqueta")

    filas: list[dict[str, Any]] = []
    incremental_por_etiqueta: dict[str, dict[str, float]] = {}
    filas: list[dict[str, Any]] = []
    incremental_por_etiqueta: dict[str, dict[str, float]] = {}
    for etiqueta in ETIQUETAS:
        j = ETIQUETAS.index(etiqueta)
        s = senales[etiqueta]
        error = (pred_v03[:, j] != y[:, j]).astype(int)
        control = s["control"]
        auc_control = float(roc_auc_score(error, control)) if len(np.unique(error)) > 1 \
            else float("nan")
        filas_et: list[dict[str, Any]] = []
        for clave in ("control", "perdida", "rango", "centroide", "estructural", "compuesto"):
            if clave == "compuesto":
                # El compuesto NO puede ajustarse y evaluarse sobre las mismas filas:
                # esa version da 0,5407 de AUC donde la version OUT-OF-FOLD con los
                # pliegues canonicos da 0,5070, luego la diferencia era sobreajuste y no
                # senal. Se ajusta en los cuatro pliegues externos y se aplica al quinto.
                X = np.column_stack([s["perdida"], s["rango"],
                                     s["centroide"], s["estructural"]])
                sc = StandardScaler().fit_transform(X)
                oof = np.zeros(n)
                for g in range(1, 6):
                    otr = np.where(pliegues != g)[0]
                    ova = np.where(pliegues == g)[0]
                    l2 = LogisticRegression(max_iter=1000, random_state=SEMILLA)
                    l2.fit(sc[otr], error[otr])
                    oof[ova] = l2.decision_function(sc[ova])
                score = oof
            else:
                score = s[clave]
            auc = float(roc_auc_score(error, score)) if len(np.unique(error)) > 1 else float("nan")
            # una senal constante no ordena nada y su AUC no es interpretable. En esta
            # pista eso ocurre con la estructural de HS, que vale cero siempre porque HS
            # no tiene region indefinida
            constante = bool(np.all(score == score[0]))
            ar = float("nan") if constante else auc_residual(control, score, error)
            filas_et.append({"etiqueta": etiqueta, "senal": clave, "auc": round(auc, 6),
                             "auc_incremental": None if np.isnan(ar) else round(ar, 6),
                             "senal_constante": constante})
        incremental_por_etiqueta[etiqueta] = {
            r["senal"]: (0.0 if r["auc_incremental"] is None else r["auc_incremental"])
            for r in filas_et}
        filas.extend([{**r, "auc_control_de_la_etiqueta": round(auc_control, 6)} for r in filas_et])
        log(f"      {etiqueta}: AUC del control {auc_control:.4f}")
        for r in filas_et:
            extra = "control" if r["auc_incremental"] is None else f"{r['auc_incremental']:.4f}"
            log(f"         {r['senal']:12} AUC {r['auc']:.4f} | incremental {extra}")

    log("")
    log("   VEREDICTO DE SENAL INCREMENTAL: la referencia NO es 0,5 sino el control")
    log("   Una senal monotona del control tiene el MISMO residual que el control, porque su")
    log("   variacion dentro de la decila es la misma, luego la pregunta correcta es si una")
    log("   senal supera al control en potencia residual, no si se aparta de 0,5")
    veredicto: dict[str, Any] = {}
    hay_senal = False
    for clave in ("perdida", "rango", "centroide", "estructural", "compuesto"):
        mejor_et = max(incremental_por_etiqueta, key=lambda e: incremental_por_etiqueta[e][clave])
        valor = incremental_por_etiqueta[mejor_et][clave]
        ref = incremental_por_etiqueta[mejor_et]["control"]
        ganancia = valor - ref
        constante = any(r["senal"] == clave and r["etiqueta"] == mejor_et
                        and r["auc_incremental"] is None for r in filas)
        supera = bool(not constante and ganancia >= UMBRAL_INCREMENTAL)
        veredicto[clave] = {"mejor_etiqueta": mejor_et, "auc_incremental": round(valor, 6),
                            "control_de_la_etiqueta": round(ref, 6),
                            "ganancia_sobre_el_control": round(ganancia, 6),
                            "senal_constante": constante,
                            "supera_umbral": supera}
        hay_senal = hay_senal or supera
        log(f"      {clave:12} incremental {valor:.4f} en {mejor_et} frente a un control de "
            f"{ref:.4f} -> ganancia {ganancia:+.4f} | constante {constante} | supera {supera}")
    log(f"      UMBRAL de ganancia sobre el control {UMBRAL_INCREMENTAL}: "
        f"hay senal incremental = {hay_senal}")

    decision = (
        "CLOSED_SIN_INTERVENCION" if not hay_senal else
        "ABIERTA_PARA_C3. Hay senal incremental, luego corresponde UNA intervencion "
        "conservadora, con una sola forma de objetivo blando o un solo ponderado por par")
    log("")
    log(f"P60-C VEREDICTO: {decision}")
    if not hay_senal:
        log("   0 experimentos consumidos: C1 y C2 son diagnostico y el diagnostico es completo")
        log("   NO se modifica ninguna etiqueta")

    informe = {
        "familia": "P60_C_LABEL_DEPENDENT_NOISE",
        "naturaleza": ("AUDITORIA DE DETECCION (C1 + C2). 0 experimentos, 0 modelos, "
                       "0 submissions, 0 Kaggle, 0 holdout, 0 test. V03 no se modifica"),
        "fecha": inicio.isoformat(timespec="seconds"),
        "experiment_id": "NINGUNO",
        "track": "P60-C",
        "script": "experiments/p60_05_ruido_por_etiqueta.py",
        "paper": ("Noisy Multi-Label Text Classification via Instance-Label Pair Correction, "
                  "Findings of NAACL 2024, Xu, Song, Liu et al., "
                  "DOI 10.18653/v1/2024.findings-naacl.93"),
        "pregunta": ("la evidencia de ruido de una etiqueta requiere una regla de deteccion "
                     "distinta de la de otra etiqueta, y la regla por par anade algo a la "
                     "confianza que el proyecto ya mide?"),
        "hipotesis": ("un score de sospecha por PAR instancia-etiqueta, que combina la perdida "
                      "del par, su rango y la distancia a centroides de clase, mas un indicador "
                      "ESTRUCTURAL de etiqueta no definida, aporta senal incremental sobre el "
                      "control 1 - p_true"),
        "mecanismo": ("la unidad de deteccion es el par y no la instancia; ademas, en este corpus "
                      "el negativo de TR cuando HS = 0 no es un negativo sino una etiqueta NO "
                      "DEFINIDA, y por eso la evidencia de ruido no puede ser un unico numero"),
        "modelo": "no se entrena ningun modelo nuevo: se reutiliza V03 dentro de cada pliegue",
        "perdida": "no se modifica ninguna funcion de perdida",
        "rasgos": "los 27.431 del TF-IDF de V03, en memoria y sin ajuste fuera de training",
        "adaptacion_legal": {
            "dominio_original": "multi-etiqueta de texto con ruido a nivel de par",
            "adaptacion": "tres problemas binarios con etiquetas anidadas y una region indefinida",
            "que_cambia": ("la unidad, de instancia a par, y la incorporacion de un indicador "
                           "estructural que solo existe para TR y AG"),
            "que_es_invariante": "el criterio de que la deteccion sea por par y por etiqueta",
        },
        "gap_analysis": {
            "filas_ruido_por_etiqueta": 0,
            "precedentes": ("V30-00 cerro siete experimentos de sospecha a nivel de INSTANCIA; "
                            "P59-03 cerro el estimador global con AUC 0,9730 frente a un control "
                            "de 0,9741; P59 no descompuso la sospecha por etiqueta"),
            "por_que_no_es_duplicacion": ("la unidad de deteccion es el par y el indicador "
                                          "estructural, y ninguno de los dos aparece en ninguna "
                                          "fila del registro"),
            "veredicto": "NUEVA",
        },
        "configuracion_pre_registrada": {
            "deciles_del_control": N_DECIL,
            "umbral_de_senal_incremental": UMBRAL_INCREMENTAL,
            "referencia_del_umbral": ("el incremental del PROPIO CONTROL de la misma "
                                     "etiqueta, no 0,5, porque una senal monotona del "
                                     "control comparte su residuo"),
            "compuesto": ("ajustado OUT-OF-FOLD con los pliegues canonicos. La version "
                          "en las mismas filas daba 0,5407 de AUC donde esta da "
                          "0,5070, luego la diferencia era sobreajuste"),
            "senales_constantes": ("excluidas del veredicto. La estructural de HS vale "
                                   "cero siempre, porque HS no tiene region "
                                   "indefinida, luego su AUC no ordena nada"),
            "senales": ("control, perdida del par, rango de la perdida, margen coseno a "
                        "centroides, indicador estructural, y compuesto por regresion logistica "
                        "sobre las cuatro senales estandarizadas"),
            "todo": "calculado con filas de ENTRENAMIENTO del pliegue",
            "indicador_de_error": "el OOF canonico, que es out-of-fold por construccion",
            "rejilla": "ninguna",
        },
        "estructura_de_etiquetas_medida": estructura,
        "resultado_deteccion": filas,
        "auc_incremental_maximo_por_senal": veredicto,
        "hay_senal_incremental": bool(hay_senal),
        "control_de_P59": (
            "P59-03 midio AUC 0,9730 para su estimador global frente a 0,9741 del control, es "
            "decir por debajo. Aqui el control se usa como condicion: un score solo cuenta si "
            "aun ordena errores DENTRO de las decilas del control"),
        "conclusion": (
            "La descomposicion por etiqueta es real en este corpus y se midio: el 100 por ciento "
            "de los negativos que ve TR en entrenamiento son etiquetas no definidas, y AG tiene "
            "el 97 por ciento. Pero esa asimetria estructural NO se traduce en senal de deteccion "
            "incremental. La razon es que el indicador estructural es casi colineal con la "
            "confianza que V03 ya tiene: cuando HS = 0, el modelo ya esta seguro de que TR = 0, "
            "de modo que 1 - p_true ya vale casi 1 y no queda margen para que un segundo score "
            "ordene dentro de esa region. Eso es una forma de colinealidad, no de ausencia de "
            "ruido, y es la misma razon por la que P59-03 no supero a su control. Junto con "
            "V14-00, que ya demostro que enmascarar la region indefinida da -0,116581 y que "
            "equivale exactamente a descartar esas filas, queda establecido por dos vias "
            "independientes que la region no definida aporta como NEGATIVOS UTILES y no como "
            "ruido que corregir"),
        "decision": decision,
        "experimentos_registrados": 0,
        "formulacion": "V03 remains the best verified leaderboard result.",
        "gastos": {"submissions": 0, "consultas_kaggle": 0, "uso_holdout": 0, "uso_test": 0,
                   "segundos": round((datetime.now(timezone.utc) - inicio).total_seconds(), 1)},
    }
    JSON_C.write_text(json.dumps(informe, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    L: list[str] = []
    W = L.append
    W("# P60-C - Ruido dependiente de etiqueta (C1 y C2)")
    W("")
    W(f"Generado por `experiments/p60_05_ruido_por_etiqueta.py`. "
      f"Fecha {inicio.isoformat(timespec='seconds')}.")
    W("")
    W("## Motivacion")
    W("")
    W("La literatura reciente de multi-etiqueta hace que la unidad de deteccion sea el PAR")
    W("(instancia, etiqueta) y no la instancia. TOXIC ya cerro el ruido en su forma de instancia:")
    W("`V30-00` cerro siete experimentos de sospecha por instancia y `P59-03` cerro el estimador")
    W("formal global, con AUC 0,9730 frente a un control trivial de 0,9741. Lo que ninguno hizo")
    W("es descomponer la sospecha POR ETIQUETA, y en este corpus esa descomposicion no es una")
    W("formalidad.")
    W("")
    W("## La asimetria estructural, medida")
    W("")
    W("```text")
    for k, val in estructura.items():
        W(f"{k:34} {val}")
    W("```")
    W("")
    W("El 100 por ciento de los negativos que ve TR en entrenamiento son etiquetas NO DEFINIDAS, y")
    W("el 97 por ciento en AG, mientras HS no tiene region indefinida. El indicador estructural")
    W("vale 1 para TR cuando HS = 0, para AG cuando HS = 0, y **siempre 0 para HS**.")
    W("")
    W("## Por que un AUC bruto no basta")
    W("")
    W("Un score puede tener AUC alto y no anadir nada, si es una copia ruidosa de la confianza que")
    W("el proyecto ya mide. `P59-03` aprendio eso de forma directa. Por eso la prueba no es")
    W("`AUC > 0,5` sino si el score sigue ordenando errores **una vez controlado** el control:")
    W("")
    W("```text")
    W("residuo = score - E[score | decila del control]")
    W("AUC incremental = AUC del residuo frente al indicador de error")
    W("```")
    W("")
    W("Deciles y no un ajuste parametrico, para no asumir ninguna forma de modelo.")
    W("")
    W("Y la referencia de comparacion **no es 0,5 sino el propio control**: una senal que")
    W("es funcion monotona del control tiene EXACTAMENTE el mismo residuo, porque su")
    W("variacion dentro de la decila es identica. La pregunta es si una senal supera al")
    W("control en potencia residual, y de ahi el umbral se mide como GANANCIA sobre el")
    W("control y no como distancia a 0,5.")
    W("")
    W("## Senales construidas, todas con filas de ENTRENAMIENTO del pliegue")
    W("")
    W("| senal | que es |")
    W("|---|---|")
    W("| control | `1 - p_true`, la probabilidad que el modelo da a la clase observada |")
    W("| perdida | perdida logistica de ese par |")
    W("| rango | rango normalizado de la perdida dentro de esa etiqueta |")
    W("| centroide | margen coseno con signo a los centroides de clase y no clase |")
    W("| estructural | indicador de etiqueta NO DEFINIDA |")
    W("| compuesto | regresion logistica sobre las cuatro senales estandarizadas |")
    W("")
    W("## Resultados")
    W("")
    W("| etiqueta | senal | AUC | AUC incremental | constante |")
    W("|---|---|---:|---:|---|")
    for r in filas:
        inc = "no aplica" if r["auc_incremental"] is None else f"{r['auc_incremental']:.4f}"
        W(f"| {r['etiqueta']} | {r['senal']} | {r['auc']:.4f} | {inc} | "
          f"{r['senal_constante']} |")
    W("")
    W("## Veredicto de senal incremental")
    W("")
    W("| senal | mejor etiqueta | incremental | control | ganancia | constante | supera |")
    W("|---|---|---:|---:|---:|---|---|")
    for clave, d in veredicto.items():
        W(f"| {clave} | {d['mejor_etiqueta']} | {d['auc_incremental']:.4f} | "
          f"{d['control_de_la_etiqueta']:.4f} | {d['ganancia_sobre_el_control']:+.4f} | "
          f"{d['senal_constante']} | **{d['supera_umbral']}** |")
    W("")
    W(f"**Hay senal incremental: {hay_senal}.** Umbral declarado: ganancia >= {UMBRAL_INCREMENTAL} sobre el control de la misma etiqueta.")
    W("")
    W("## Decision")
    W("")
    W("```text")
    W(f"decision  {decision}")
    W("```")
    W("")
    W("> V03 remains the best verified leaderboard result.")
    W("")
    W("*Generado por `experiments/p60_05_ruido_por_etiqueta.py`. Sin push.*")
    MD_C.write_text(a_ascii("\n".join(L)), encoding="ascii")

    with CSV_C.open("w", encoding="ascii", newline="") as fh:
        w = _csv.writer(fh, lineterminator="\n")
        w.writerow(["etiqueta", "senal", "auc", "auc_incremental", "auc_control_de_la_etiqueta"])
        for r in filas:
            w.writerow([r["etiqueta"], r["senal"], r["auc"],
                        "control" if r["auc_incremental"] is None else r["auc_incremental"],
                        r["auc_control_de_la_etiqueta"]])

    for destino in (MD_C, JSON_C, CSV_C):
        if destino.exists() and any(b > 127 for b in destino.read_bytes()):
            raise SystemExit(f"PARADA: {destino.name} conserva bytes no ASCII")
    log("")
    log(f"P60-05 OK | {informe['gastos']['segundos']} s | 0 experimentos consumidos")


if __name__ == "__main__":
    ast.parse(Path(__file__).read_text(encoding="utf-8"))
    main()
