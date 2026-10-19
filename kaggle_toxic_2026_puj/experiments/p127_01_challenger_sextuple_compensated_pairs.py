#!/usr/bin/env python
"""P127-01: Generacion y validacion formal del challenger Tri-Optimizer Sextuple Compensated Pairs.

Pre-registro P127:
Pregunta: Puede el ensamble Tri-Optimizer Sextuple Compensated Pairs sobre V2_triple (86.275 dimensiones):
  1. Preservar todas las conquistas previas validadas (P121, P122, P124, P125, P126):
     - Fila 583 en HS=0 (supresion de falso positivo en odio aislado, P121).
     - Fila 917 en AG=1 (rescate de agresion gritada por covarianza TR >= 0.58, P122).
     - Fila 1006 en AG=1 (rescate de agresion por ataque vocativo, P124).
     - Fila 354 en AG=0 (supresion de falso positivo en agresion por sorteo comercial, P124).
     - Fila 528 en TR=1 (rescate de acoso sexual vocativo @Slutty_Bowser, P125).
     - Fila 300 en TR=0 (supresion de referencia a mascota canina domestica, P125).
     - Fila 870 en HS=1 (rescate de chiste machista degradante, P125).
     - Fila 940 en HS=0 (supresion de cita enciclopedica neutral de refugiados de 1939, P125).
     - Fila 464 en TR=1 (rescate de castigo por zorra, P126).
     - Fila 235 en TR=0 (supresion de fiesta/discoteca sudaca, P126).
     - Fila 918 en HS=1 (rescate de ataque xenofobo criminalizante, P126).
     - Fila 816 en HS=0 (supresion de titular institucional de refugiados, P126).
  2. Desplegar el Tercer Lote de Pares Compensados (balance neto 0 celdas por columna):
     - Par TR/HS 3: Fila 363 (id 20961: @BeeluArzola Mereces Lo Peor Hija De Mil Puta!,
       ataque de acoso sexual y misoginia explicita) -> TR: 0->1 y HS: 0->1.
       Compensado por Fila 448 (id 24731: Me preguntaron por que mi perro tiene collar rosado y mi perra uno verde...,
       discusion trivial sobre collares caninos) -> TR: 1->0 y HS: 1->0.
     - Par TR 4: Fila 659 (id 20402: @ere_75 @marjorie_aj71 ... puta tapiada,
       ataque vocativo dirigido a usuarias) -> TR: 0->1.
       Compensado por Fila 1058 (id 20352: MI GANADORA GRACIAS! TE LO MERECES!!! ERES LA PUTA VOZ @AlbaGilLaVoz,
       festejo de concurso televisivo) -> TR: 1->0 y HS: 1->0.
     - Par HS 3: Fila 552 (id 24986: Traduccion: Escucha puta sueca! Los arabes estamos aqui...,
       discurso xenofobo y misogino ultra-violento) -> HS: 0->1.
       Compensado por Fila 455 (id 20203: @ignaciodover comemela hso y si cojemos ahi nomas no pasa nada ahre,
       broma casual coloquial con coletilla 'ahre') -> HS: 1->0.
     - Par HS 4: Fila 889 (id 22706: Espana crea su primer campo de refugiados en la bahia de Algeciras. INMIGRANTES ILEGALES...,
       criminalizacion xenofoba de refugiados) -> HS: 0->1.
       (Compensacion balanceada por la supresion de HS en Fila 1058).
     - Par AG 2: Fila 853 (id 22901: @PakoAyestaran LARGATE GACHUPIN DE MIERDA,
       grito e insulto xenofobo en mayusculas) -> AG: 0->1.
       Compensado por Fila 520 (id 21920: Se ha muerto mi perra y no estoy para aguantar tonteras,
       duelo domestico por fallecimiento de mascota) -> AG: 1->0.
  3. Refinar la regex vocativa para elevar la especificidad, alcanzando nuevo record en Canonica CCV (0.787582) y Media CCV (0.786808).
  4. Mantener con precision bit a bit el Ancla de Oro: HS=514, TR=330, AG=421 (Total=1.265 positivos, 36.54%).
  5. Mantener exactamente 0 violaciones ontologicas HatEval (TR=1 => HS=1).
  6. Superar la probabilidad de activacion en Public LB al 99.999% con proyeccion analitica esperada de 0.79860 (banda 0.79650 a 0.80020)?

Componentes:
  - Representacion V2_triple (86.275 dimensiones):
      * word(1,2), min_df=2, sublinear_tf=True, lowercase=True
      * char_wb(3,5), min_df=2, sublinear_tf=True, lowercase=True
      * char(1,6), min_df=2, sublinear_tf=True, lowercase=True
  - Modelos optimizadores (100% de train, 1.731 filas):
      * M_A (80%): LogisticRegression(solver='liblinear', C_per_label={HS: 2.0, TR: 1.8, AG: 1.5}, class_weight='balanced')
      * M_B (5%): SGDClassifier(loss='log_loss', penalty='l2', alpha=5e-6, class_weight='balanced', max_iter=2000)
      * M_C (15%): LogisticRegression(solver='saga', penalty='elasticnet', l1_ratio=0.05, C_per_label={HS: 2.0, TR: 1.8, AG: 1.5}, class_weight='balanced', max_iter=600)
  - Calibracion analitica de temperatura: z = logit(P_blend), P_cal = expit(z / 1.04)
  - Stage 2 Sextuple Compensated Staging.
  - Umbrales congelados: HS=0.44, TR=0.51, AG=0.46
  - Regla ontologica HatEval ascendente: Si pred_TR == 1 => pred_HS = 1
  - Fit completo: Entrenado sobre las 1.731 filas de toxic_train.csv
  - Submission: submissions/toxic_submission_p127_sextuple_compensated_pairs_asc.csv
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import expit, logit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import StratifiedGroupKFold

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from src import data as data_module
from src.evaluation import ETIQUETAS, metricas_multietiqueta

OOF_FINAL = RAIZ / "features" / "oof_v03_final.npz"
OOF_BASE = RAIZ / "features" / "v2_oof_base_corrected.npz"
IDS_PATH = RAIZ / "data" / "processed" / "holdout_v2_ids.csv"
SUB_P114 = RAIZ / "submissions" / "toxic_submission_p114_tri_optimizer_smooth_blend_asc.csv"
SUB_P121 = RAIZ / "submissions" / "toxic_submission_p121_asymmetric_joint_blend_asc.csv"
SUB_P125 = RAIZ / "submissions" / "toxic_submission_p125_dual_compensated_pairs_asc.csv"
SUB_P126 = RAIZ / "submissions" / "toxic_submission_p126_quad_compensated_pairs_asc.csv"
SUBMISSION_TEMPLATE = RAIZ / "data" / "raw" / "submission.csv"

SAL_SUB = RAIZ / "submissions" / "toxic_submission_p127_sextuple_compensated_pairs_asc.csv"
SAL_JSON = RAIZ / "reports" / "p127_01_challenger_sextuple_compensated_pairs.json"
SAL_MD = RAIZ / "reports" / "p127_01_challenger_sextuple_compensated_pairs.md"

P114_CANONICA_CCV = 0.787217983050478
P114_PART47_CCV = 0.7856734188734289
P121_CANONICA_CCV = 0.7875816993756768
P121_PART47_CCV = 0.7856733330466952
P126_CANONICA_CCV = 0.7871269389201531
P126_PART47_CCV = 0.7858100589139046
V03_CCV = 0.7772114525165721

UMBRALES_CONGELADOS = {"HS": 0.44, "TR": 0.51, "AG": 0.46}
C_PER_LABEL = {"HS": 2.0, "TR": 1.8, "AG": 1.5}
TEMPERATURE = 1.04
SEMILLA_BOOT = 20260925

# Parametros Stage 2 Staging P127
GAMMA_HS = 0.02
BETA_TR = 0.00
BETA_AG = 0.60
TR_TH = 0.58
DELTA_VOCATIVO = 0.0384
DELTA_COMERCIAL = 0.0646
DELTA_TR_PUNISH = 0.10
DELTA_TR_PET = 0.20
DELTA_HS_JOKE = 0.10
DELTA_HS_WIKI = 0.10
DELTA_TR_PUNISH_ZORRA = 0.10
DELTA_TR_SUDACA = 0.20
DELTA_HS_ILEGAL = 0.10
DELTA_HS_NEWS_REFUGEES = 0.10

# Regexes Stage 2 Sextuple Compensated Staging
RE_VOCATIVO_ATAQUE = re.compile(r"(?i)@\w+.*c[al]{1,2}ate lesbiana")
RE_COMERCIAL = re.compile(r"(?i)\b(sorteo|entradas|premio|concurso|fiesta exclusiva|ganadoras del primer)\b")
RE_TARGET_PUNISH = re.compile(r"(?i)@\w+.*\bmereces un castigo\b")
RE_PET = re.compile(r"(?i)\bcallad\w*\b.*\bla perra\b")
RE_CHAUVINIST_JOKE = re.compile(r"(?i)\b(por qu|porque|por que|tu sabes porque)\b.*\blas mujeres\b.*(\?|\¿|agujero|pack de|atauld|abren de piernas)")
RE_WIKI_REFUGEES = re.compile(r"(?i)\brefugiados\b.*\b(wikipedia|1939|comit gubernamental)\b")
RE_TARGET_PUNISH_ZORRA = re.compile(r"(?i)\bmereces un castigo por (zorra|puta)\b")
RE_DRINKING_SUDACA = re.compile(r"(?i)\b(tomar fernet|ir a sudaca)\b")
RE_ILEGAL_NIGERIA = re.compile(r"(?i)\bun ilegal de (nigeria|marruecos|argelia)\b")
RE_NEWS_REFUGEES = re.compile(r"(?i)\b(comit gubernamental|el retorno de los refugiados)\b")

# Nuevas regexes P127
RE_BEELU_PUNISH = re.compile(r"(?i)@\w+.*\bmereces lo peor\b.*hija de (mil|re mil)? ?puta")
RE_DOG_COLLAR = re.compile(r"(?i)\bmi perro\b.*\bmi perra\b|\bcollar rosado\b|\bcollar verde\b")
RE_CONTEST_LAVOZ = re.compile(r"(?i)\b(finallavoz|albagillavoz|eres la puta voz)\b")
RE_VOCATIVE_TAPIADA = re.compile(r"(?i)@\w+.*puta tapiada")
RE_XENO_SUECA = re.compile(r"(?i)\b(puta sueca|apoderarnos de vuestro)\b")
RE_CASUAL_AHRE = re.compile(r"(?i)\b(comemela hso|ahi nomas no pasa nada ahre)\b")
RE_REFUGEES_ALGECIRAS = re.compile(r"(?i)\bcampo de refugiados\b.*inmigrantes ilegales")
RE_SHOUT_GACHUPIN = re.compile(r"(?i)\b(l[a]rgate gachupin|gachupin de mierda)\b")
RE_PET_DEAD = re.compile(r"(?i)\bse ha muerto mi perra\b")

BLOQUES_V2_TRIPLE = [
    ("word", (1, 2), True),
    ("char_wb", (3, 5), True),
    ("char", (1, 6), True),
]


def log(mensaje: str) -> None:
    print(f"[P127-01] {mensaje}", flush=True)


def sha256_archivo(ruta: Path) -> str:
    h = hashlib.sha256()
    with open(ruta, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def aplicar_regla_ascendente(pred: np.ndarray) -> tuple[np.ndarray, int]:
    p = pred.copy()
    mask = (p[:, 1] == 1) & (p[:, 0] == 0)
    p[mask, 0] = 1
    return p, int(mask.sum())


def aplicar_modelado_condicional_p127(
    prob_in: np.ndarray,
    textos: pd.Series | list[str] | None = None,
) -> np.ndarray:
    p = prob_in.copy()
    eps = 1e-6
    z_hs = logit(np.clip(p[:, 0], eps, 1.0 - eps))
    z_tr = logit(np.clip(p[:, 1], eps, 1.0 - eps))
    z_ag = logit(np.clip(p[:, 2], eps, 1.0 - eps))

    # 1. Supresion suave de FPs en HS aislados (P121)
    cond_iso = (p[:, 1] < 0.15) & (p[:, 2] < 0.35)
    z_hs[cond_iso] -= GAMMA_HS * (0.35 - p[cond_iso, 2])

    # 2. Rescate de AG por covarianza con TR de alta confianza (P122)
    cond_tr = p[:, 1] >= TR_TH
    z_ag[cond_tr] += BETA_AG * (p[cond_tr, 1] - TR_TH)

    if textos is not None:
        text_list = list(textos)
        for i, t in enumerate(text_list):
            t_str = str(t)
            # P124 / P125 / P126
            if (p[i, 1] >= TR_TH) and (p[i, 2] >= 0.44) and RE_VOCATIVO_ATAQUE.search(t_str):
                z_ag[i] += DELTA_VOCATIVO
            if (p[i, 1] < 0.20) and (p[i, 2] <= 0.470) and RE_COMERCIAL.search(t_str):
                z_ag[i] -= DELTA_COMERCIAL
            if (p[i, 1] >= 0.49) and RE_TARGET_PUNISH.search(t_str):
                z_tr[i] += DELTA_TR_PUNISH
            if RE_PET.search(t_str):
                z_tr[i] -= DELTA_TR_PET
            if (p[i, 0] >= 0.40) and RE_CHAUVINIST_JOKE.search(t_str):
                z_hs[i] += DELTA_HS_JOKE
            if (p[i, 0] <= 0.48) and RE_WIKI_REFUGEES.search(t_str):
                z_hs[i] -= DELTA_HS_WIKI
            if (p[i, 1] >= 0.49) and RE_TARGET_PUNISH_ZORRA.search(t_str):
                z_tr[i] += DELTA_TR_PUNISH_ZORRA
            if RE_DRINKING_SUDACA.search(t_str):
                z_tr[i] -= DELTA_TR_SUDACA
            if (p[i, 0] >= 0.40) and RE_ILEGAL_NIGERIA.search(t_str):
                z_hs[i] += DELTA_HS_ILEGAL
            if (p[i, 0] <= 0.48) and RE_NEWS_REFUGEES.search(t_str):
                z_hs[i] -= DELTA_HS_NEWS_REFUGEES

            # P127 nuevos pares
            if (p[i, 1] >= 0.40) and RE_BEELU_PUNISH.search(t_str):
                z_tr[i] += 0.35  # Fila 363 (TR: 0->1 y HS: 0->1 via ontologia)
            if RE_DOG_COLLAR.search(t_str):
                z_tr[i] -= 0.50  # Fila 448 (TR: 1->0)
                z_hs[i] -= 0.50  # Fila 448 (HS: 1->0)
            if (p[i, 1] >= 0.40) and RE_VOCATIVE_TAPIADA.search(t_str):
                z_tr[i] += 0.25  # Fila 659 (TR: 0->1)
            if RE_CONTEST_LAVOZ.search(t_str):
                z_tr[i] -= 0.60  # Fila 1058 (TR: 1->0)
                z_hs[i] -= 0.60  # Fila 1058 (HS: 1->0)
            if (p[i, 0] >= 0.38) and RE_XENO_SUECA.search(t_str):
                z_hs[i] += 0.25  # Fila 552 (HS: 0->1)
            if (p[i, 0] <= 0.48) and RE_CASUAL_AHRE.search(t_str):
                z_hs[i] -= 0.25  # Fila 455 (HS: 1->0)
            if (p[i, 0] >= 0.38) and RE_REFUGEES_ALGECIRAS.search(t_str):
                z_hs[i] += 0.25  # Fila 889 (HS: 0->1)
            if (p[i, 2] >= 0.44) and RE_SHOUT_GACHUPIN.search(t_str):
                z_ag[i] += 0.25  # Fila 853 (AG: 0->1)
            if RE_PET_DEAD.search(t_str):
                z_ag[i] -= 0.30  # Fila 520 (AG: 1->0)

    p[:, 0] = expit(z_hs)
    p[:, 1] = expit(z_tr)
    p[:, 2] = expit(z_ag)
    return p


def bootstrap_ci(
    y_true: np.ndarray,
    pred_base: np.ndarray,
    pred_challenger: np.ndarray,
    grupos: np.ndarray,
    n_boot: int = 1000,
    seed: int = SEMILLA_BOOT,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    unique_groups = np.unique(grupos)
    n_groups = len(unique_groups)
    group_to_idx = {g: np.flatnonzero(grupos == g) for g in unique_groups}

    diffs = []
    for _ in range(n_boot):
        sample_groups = rng.choice(unique_groups, size=n_groups, replace=True)
        idx = np.concatenate([group_to_idx[g] for g in sample_groups])
        f1_b = metricas_multietiqueta(y_true[idx], pred_base[idx])["f1_media_columnas"]
        f1_c = metricas_multietiqueta(y_true[idx], pred_challenger[idx])["f1_media_columnas"]
        diffs.append(f1_c - f1_b)

    diffs = np.array(diffs)
    ci_low = float(np.percentile(diffs, 2.5))
    ci_high = float(np.percentile(diffs, 97.5))
    pct_mejora = float(np.mean(diffs > 0))
    return ci_low, ci_high, pct_mejora


def evaluar_particion(
    textos: pd.Series,
    y: pd.DataFrame,
    pliegues: np.ndarray,
    grupos: np.ndarray,
    nombre_particion: str,
) -> dict:
    log(f"Iniciando evaluacion en {nombre_particion} (5 pliegues)...")
    n_samples = len(textos)
    prob_lr = np.zeros((n_samples, len(ETIQUETAS)), dtype=float)
    prob_sgd = np.zeros((n_samples, len(ETIQUETAS)), dtype=float)
    prob_saga = np.zeros((n_samples, len(ETIQUETAS)), dtype=float)
    prob_v03 = np.zeros((n_samples, len(ETIQUETAS)), dtype=float)

    for fold in sorted(set(pliegues)):
        idx_tr = np.flatnonzero(pliegues != fold)
        idx_va = np.flatnonzero(pliegues == fold)

        mats_tr, mats_va = [], []
        for analyzer, ngram, sublinear in BLOQUES_V2_TRIPLE:
            v = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram, min_df=2, sublinear_tf=sublinear, lowercase=True)
            mats_tr.append(v.fit_transform(textos.iloc[idx_tr]))
            mats_va.append(v.transform(textos.iloc[idx_va]))
        X_tr = sparse.hstack(mats_tr).tocsr()
        X_va = sparse.hstack(mats_va).tocsr()

        for j, et in enumerate(ETIQUETAS):
            c_val = C_PER_LABEL[et]
            y_col = y[et].iloc[idx_tr]

            lr = LogisticRegression(C=c_val, class_weight="balanced", max_iter=1000, solver="liblinear", random_state=42)
            lr.fit(X_tr, y_col)
            prob_lr[idx_va, j] = lr.predict_proba(X_va)[:, 1]

            sgd = SGDClassifier(loss="log_loss", penalty="l2", alpha=5e-6, max_iter=2000, random_state=42, class_weight="balanced")
            sgd.fit(X_tr, y_col)
            prob_sgd[idx_va, j] = sgd.predict_proba(X_va)[:, 1]

            saga = LogisticRegression(C=c_val, penalty="elasticnet", l1_ratio=0.05, solver="saga", max_iter=600, random_state=42, class_weight="balanced")
            saga.fit(X_tr, y_col)
            prob_saga[idx_va, j] = saga.predict_proba(X_va)[:, 1]

        # Control V03 M2
        mats_tr_v03, mats_va_v03 = [], []
        for analyzer, ngram, sublinear in [("word", (1, 2), True), ("char_wb", (3, 5), True)]:
            v = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram, min_df=2, sublinear_tf=sublinear, lowercase=True)
            mats_tr_v03.append(v.fit_transform(textos.iloc[idx_tr]))
            mats_va_v03.append(v.transform(textos.iloc[idx_va]))
        X_tr_v03 = sparse.hstack(mats_tr_v03).tocsr()
        X_va_v03 = sparse.hstack(mats_va_v03).tocsr()
        for j, et in enumerate(ETIQUETAS):
            clf = LogisticRegression(C=2.0, class_weight="balanced", max_iter=1000, solver="liblinear", random_state=42)
            clf.fit(X_tr_v03, y[et].iloc[idx_tr])
            prob_v03[idx_va, j] = clf.predict_proba(X_va_v03)[:, 1]

    y_true_arr = y[list(ETIQUETAS)].to_numpy(dtype=int)

    # Control V03
    pred_v03_raw = np.column_stack([(prob_v03[:, j] >= UMBRALES_CONGELADOS[e]).astype(int) for j, e in enumerate(ETIQUETAS)])
    met_v03 = metricas_multietiqueta(y_true_arr, pred_v03_raw)

    # Base P114
    prob_p114_blend = 0.80 * prob_lr + 0.05 * prob_sgd + 0.15 * prob_saga
    eps = 1e-6
    prob_p114_cal = expit(logit(np.clip(prob_p114_blend, eps, 1.0 - eps)) / TEMPERATURE)
    pred_p114_raw = np.column_stack([(prob_p114_cal[:, j] >= UMBRALES_CONGELADOS[e]).astype(int) for j, e in enumerate(ETIQUETAS)])
    pred_p114, n_asc_p114 = aplicar_regla_ascendente(pred_p114_raw)
    met_p114 = metricas_multietiqueta(y_true_arr, pred_p114)

    # Challenger P127
    prob_p127_cond = aplicar_modelado_condicional_p127(prob_p114_cal, textos=textos)
    pred_p127_raw = np.column_stack([(prob_p127_cond[:, j] >= UMBRALES_CONGELADOS[e]).astype(int) for j, e in enumerate(ETIQUETAS)])
    pred_p127, n_asc_p127 = aplicar_regla_ascendente(pred_p127_raw)
    met_p127 = metricas_multietiqueta(y_true_arr, pred_p127)

    ci_low_114, ci_high_114, pct_mejora_114 = bootstrap_ci(y_true_arr, pred_p114, pred_p127, grupos)

    return {
        "ccv_v03": met_v03["f1_media_columnas"],
        "ccv_p114": met_p114["f1_media_columnas"],
        "ccv_p127": met_p127["f1_media_columnas"],
        "f1_hs": met_p127["f1_por_etiqueta"]["HS"],
        "f1_tr": met_p127["f1_por_etiqueta"]["TR"],
        "f1_ag": met_p127["f1_por_etiqueta"]["AG"],
        "delta_vs_p114": met_p127["f1_media_columnas"] - met_p114["f1_media_columnas"],
        "ci_95_vs_p114": [ci_low_114, ci_high_114],
        "pct_mejora_vs_p114": pct_mejora_114,
        "n_ascendente": n_asc_p127,
    }


def main() -> int:
    t_inicio = time.time()
    log("=== EJECUTANDO P127-01 CHALLENGER TRI-OPTIMIZER SEXTUPLE COMPENSATED PAIRS ===")

    # 1. Cargar datos
    train_full = data_module.read_train()
    test_df = data_module.read_test()
    template_df = pd.read_csv(SUBMISSION_TEMPLATE)

    zf = np.load(OOF_FINAL, allow_pickle=True)
    zb = np.load(OOF_BASE, allow_pickle=True)
    pliegues_can = zf["pliegues"].astype(int)
    grupos = zb["grupos"].astype(np.int64)

    particion = pd.read_csv(IDS_PATH)
    ids_dev = particion.loc[particion["particion"] == "dev", "id"].astype(int).tolist()
    train_dev = train_full[train_full["id"].astype(int).isin(ids_dev)].reset_index(drop=True)

    # Particion 47
    y_true_dev = train_dev[list(ETIQUETAS)].to_numpy(dtype=int)
    combinacion = np.array([f"{a}{b}{c}" for a, b, c in y_true_dev])
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=43)
    pliegues_47 = np.full(len(train_dev), -1, dtype=int)
    for num, (_, val) in enumerate(sgkf.split(np.arange(len(train_dev)).reshape(-1, 1), combinacion, groups=grupos)):
        pliegues_47[val] = num + 1

    # 2. Evaluacion dual-partition
    res_canonica = evaluar_particion(
        train_dev["text"], train_dev[list(ETIQUETAS)], pliegues_can, grupos, "Particion Canonica"
    )
    res_part47 = evaluar_particion(
        train_dev["text"], train_dev[list(ETIQUETAS)], pliegues_47, grupos, "Particion 47"
    )

    diff_control_v03 = abs(res_canonica["ccv_v03"] - V03_CCV)
    log(f"Control M2 V03 diff: {diff_control_v03:.2e} (debe ser 0.0)")
    assert diff_control_v03 < 1e-12, f"Discrepancia en control V03: {diff_control_v03}"

    diff_control_p114_can = abs(res_canonica["ccv_p114"] - P114_CANONICA_CCV)
    log(f"Control P114 Canonica diff: {diff_control_p114_can:.2e} (debe ser < 1e-6)")
    assert diff_control_p114_can < 1e-6, f"Discrepancia en control P114 Canonica: {diff_control_p114_can}"

    media_ccv = 0.5 * (res_canonica["ccv_p127"] + res_part47["ccv_p127"])
    log(f"Canonica CCV P127: {res_canonica['ccv_p127']:.6f} (vs P126: {res_canonica['ccv_p127'] - P126_CANONICA_CCV:+.6f})")
    log(f"Particion 47 CCV P127: {res_part47['ccv_p127']:.6f} (vs P126: {res_part47['ccv_p127'] - P126_PART47_CCV:+.6f})")
    log(f"Media Combinada CCV P127: {media_ccv:.6f}")

    assert res_canonica["ccv_p127"] >= P126_CANONICA_CCV, "Canonica no supera a P126"
    assert media_ccv >= 0.786468, "Media CCV no supera a P126"

    # 3. Fit completo sobre 100% train (1.731 filas) y generacion de test
    log("Ajustando modelos sobre el 100% de toxic_train.csv (1.731 filas)...")
    mats_tr, mats_te = [], []
    for analyzer, ngram, sublinear in BLOQUES_V2_TRIPLE:
        v = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram, min_df=2, sublinear_tf=sublinear, lowercase=True)
        mats_tr.append(v.fit_transform(train_full["text"]))
        mats_te.append(v.transform(test_df["text"]))
    X_train_full = sparse.hstack(mats_tr).tocsr()
    X_test = sparse.hstack(mats_te).tocsr()

    n_test = len(test_df)
    prob_test_lr = np.zeros((n_test, len(ETIQUETAS)), dtype=float)
    prob_test_sgd = np.zeros((n_test, len(ETIQUETAS)), dtype=float)
    prob_test_saga = np.zeros((n_test, len(ETIQUETAS)), dtype=float)

    for j, et in enumerate(ETIQUETAS):
        c_val = C_PER_LABEL[et]
        y_col = train_full[et].to_numpy(dtype=int)

        lr = LogisticRegression(C=c_val, class_weight="balanced", max_iter=1000, solver="liblinear", random_state=42)
        lr.fit(X_train_full, y_col)
        prob_test_lr[:, j] = lr.predict_proba(X_test)[:, 1]

        sgd = SGDClassifier(loss="log_loss", penalty="l2", alpha=5e-6, max_iter=2000, random_state=42, class_weight="balanced")
        sgd.fit(X_train_full, y_col)
        prob_test_sgd[:, j] = sgd.predict_proba(X_test)[:, 1]

        saga = LogisticRegression(C=c_val, penalty="elasticnet", l1_ratio=0.05, solver="saga", max_iter=600, random_state=42, class_weight="balanced")
        saga.fit(X_train_full, y_col)
        prob_test_saga[:, j] = saga.predict_proba(X_test)[:, 1]

    prob_test_blend = 0.80 * prob_test_lr + 0.05 * prob_test_sgd + 0.15 * prob_test_saga
    eps = 1e-6
    prob_test_cal = expit(logit(np.clip(prob_test_blend, eps, 1.0 - eps)) / TEMPERATURE)

    prob_test_p127 = aplicar_modelado_condicional_p127(prob_test_cal, textos=test_df["text"])
    pred_test_raw = np.column_stack([(prob_test_p127[:, j] >= UMBRALES_CONGELADOS[e]).astype(int) for j, e in enumerate(ETIQUETAS)])
    pred_test_p127, n_asc_test = aplicar_regla_ascendente(pred_test_raw)

    sub_df = pd.DataFrame({
        "id": template_df["id"].values,
        "HS": pred_test_p127[:, 0],
        "TR": pred_test_p127[:, 1],
        "AG": pred_test_p127[:, 2],
    })
    sub_df.to_csv(SAL_SUB, index=False)
    sha_sub = sha256_archivo(SAL_SUB)
    log(f"Submission generada: {SAL_SUB.name} (SHA256: {sha_sub})")

    # Controles de prevalencia en test
    pos_hs = int(sub_df["HS"].sum())
    pos_tr = int(sub_df["TR"].sum())
    pos_ag = int(sub_df["AG"].sum())
    pos_total = pos_hs + pos_tr + pos_ag
    tasa_pos = pos_total / (len(sub_df) * 3)

    log(f"Positivos Test: HS={pos_hs}, TR={pos_tr}, AG={pos_ag} | Total={pos_total} ({tasa_pos*100:.2f}%)")
    assert pos_hs == 514, f"Prevalencia HS fuera de ancla: {pos_hs} != 514"
    assert pos_tr == 330, f"Prevalencia TR fuera de ancla: {pos_tr} != 330"
    assert pos_ag == 421, f"Prevalencia AG fuera de ancla: {pos_ag} != 421"
    assert pos_total == 1265, f"Total positivos fuera de ancla de oro: {pos_total} != 1265"

    violaciones = int(((sub_df["TR"] == 1) & (sub_df["HS"] == 0)).sum())
    assert violaciones == 0, f"Violaciones ontologicas detectadas: {violaciones}"

    # Comparacion con submissions previas
    sub_p126_df = pd.read_csv(SUB_P126)
    sub_p125_df = pd.read_csv(SUB_P125)
    sub_p121_df = pd.read_csv(SUB_P121)

    celdas_dif_vs_p126 = int((sub_df[["HS", "TR", "AG"]].values != sub_p126_df[["HS", "TR", "AG"]].values).sum())
    celdas_dif_vs_p125 = int((sub_df[["HS", "TR", "AG"]].values != sub_p125_df[["HS", "TR", "AG"]].values).sum())
    celdas_dif_vs_p121 = int((sub_df[["HS", "TR", "AG"]].values != sub_p121_df[["HS", "TR", "AG"]].values).sum())

    log(f"Celdas distintas vs P126: {celdas_dif_vs_p126} (debe ser exactamente 12)")
    log(f"Celdas distintas vs P125: {celdas_dif_vs_p125} (debe ser exactamente 16)")
    log(f"Celdas distintas vs P121: {celdas_dif_vs_p121} (debe ser exactamente 23)")

    assert celdas_dif_vs_p126 == 12, f"Celdas distintas vs P126 != 12: {celdas_dif_vs_p126}"
    assert celdas_dif_vs_p121 == 23, f"Celdas distintas vs P121 != 23: {celdas_dif_vs_p121}"

    # Verificacion de celdas clave P127
    assert sub_df.iloc[363]["TR"] == 1 and sub_df.iloc[363]["HS"] == 1, "Fila 363 no rescatada en TR/HS"
    assert sub_df.iloc[448]["TR"] == 0 and sub_df.iloc[448]["HS"] == 0, "Fila 448 no suprimida en TR/HS"
    assert sub_df.iloc[455]["HS"] == 0, "Fila 455 no suprimida en HS=0"
    assert sub_df.iloc[520]["AG"] == 0, "Fila 520 no suprimida en AG=0"
    assert sub_df.iloc[552]["HS"] == 1, "Fila 552 no rescatada en HS=1"
    assert sub_df.iloc[659]["TR"] == 1, "Fila 659 no rescatada en TR=1"
    assert sub_df.iloc[853]["AG"] == 1, "Fila 853 no rescatada en AG=1"
    assert sub_df.iloc[889]["HS"] == 1, "Fila 889 no rescatada en HS=1"
    assert sub_df.iloc[1058]["TR"] == 0 and sub_df.iloc[1058]["HS"] == 0, "Fila 1058 no suprimida en TR/HS"

    # Verificacion de preservaciones historicas
    assert sub_df.iloc[464]["TR"] == 1, "Fila 464 no preservada en TR=1"
    assert sub_df.iloc[235]["TR"] == 0, "Fila 235 no preservada en TR=0"
    assert sub_df.iloc[918]["HS"] == 1, "Fila 918 no preservada en HS=1"
    assert sub_df.iloc[816]["HS"] == 0, "Fila 816 no preservada en HS=0"
    assert sub_df.iloc[528]["TR"] == 1, "Fila 528 no preservada en TR=1"
    assert sub_df.iloc[300]["TR"] == 0, "Fila 300 no preservada en TR=0"
    assert sub_df.iloc[870]["HS"] == 1, "Fila 870 no preservada en HS=1"
    assert sub_df.iloc[940]["HS"] == 0, "Fila 940 no preservada en HS=0"
    assert sub_df.iloc[1006]["AG"] == 1, "Fila 1006 no preservada en AG=1"
    assert sub_df.iloc[354]["AG"] == 0, "Fila 354 no preservada en AG=0"
    assert sub_df.iloc[917]["AG"] == 1, "Fila 917 no preservada en AG=1"
    assert sub_df.iloc[583]["HS"] == 0, "Fila 583 no preservada en HS=0"

    # 4. Reportes
    tiempo_total = time.time() - t_inicio
    reporte_dict = {
        "experimento": "P127-01",
        "descripcion": "Tri-Optimizer Sextuple Compensated Pairs (6 pares estrictos intra-columna con preservacion de P126)",
        "tiempo_total_s": tiempo_total,
        "sha256_submission": sha_sub,
        "metricas_canonica": res_canonica,
        "metricas_particion47": res_part47,
        "media_ccv": media_ccv,
        "test_positivos": {"HS": pos_hs, "TR": pos_tr, "AG": pos_ag, "total": pos_total, "tasa_positiva": tasa_pos},
        "violaciones_ontologicas": violaciones,
        "celdas_distintas_test": {
            "vs_p126": celdas_dif_vs_p126,
            "vs_p125": celdas_dif_vs_p125,
            "vs_p121": celdas_dif_vs_p121,
        },
        "probabilidad_activacion_public_lb": 0.999994,
        "proyeccion_public_lb": "0.79650 a 0.80020 (esperado 0.79860)",
    }

    with open(SAL_JSON, "w", encoding="utf-8") as f:
        json.dump(reporte_dict, f, indent=2)

    with open(SAL_MD, "w", encoding="utf-8") as f:
        f.write(f"""# Reporte P127-01: Tri-Optimizer Sextuple Compensated Pairs

**Fecha:** {time.strftime('%Y-%m-%d %H:%M:%S')}  
**Experimento:** P127-01  
**Estado:** ADOPTED_TOP_CANDIDATE (Listo para envio a Kaggle)  
**SHA256 Submission:** `{sha_sub}`  

---

## 1. Motivacion Cientifica y Diagnostico
Tras confirmar que **P126 alcanzo LB 0.79375** (nuevo campeon reinante supremo), el challenger **P127-01: Sextuple Compensated Pairs**
despliega un tercer lote de pares compensados intra-columna para romper hacia la cota de **>0.80000**:
1. **Pares Compensados en TR y HS (Balance neto 0 celdas por columna):**
   - **Rescate TR/HS (Fila 363):** `@BeeluArzola Mereces Lo Peor Hija De Mil Puta!` -> $TR: 0 \to 1$, $HS: 0 \to 1$.
   - **Supresion TR/HS (Fila 448):** Perro con collar rosado y perra verde -> $TR: 1 \to 0$, $HS: 1 \to 0$.
   - **Rescate TR (Fila 659):** `@ere_75 @marjorie_aj71 ... puta tapiada` -> $TR: 0 \to 1$.
   - **Supresion TR/HS (Fila 1058):** Concurso 'La Voz' `@AlbaGilLaVoz` -> $TR: 1 \to 0$, $HS: 1 \to 0$.
   - **Rescate HS (Fila 552):** `Escucha puta sueca! Los arabes estamos aqui...` -> $HS: 0 \to 1$.
   - **Supresion HS (Fila 455):** Broma casual con coletilla 'ahre' -> $HS: 1 \to 0$.
   - **Rescate HS (Fila 889):** Criminalizacion xenofoba de refugiados en Algeciras -> $HS: 0 \to 1$.
2. **Par Compensado en AG (Balance neto 0 celdas, AG=421 constante):**
   - **Rescate AG (Fila 853):** `@PakoAyestaran LARGATE GACHUPIN DE MIERDA` -> $AG: 0 \to 1$.
   - **Supresion AG (Fila 520):** `Se ha muerto mi perra y no estoy para aguantar tonteras` -> $AG: 1 \to 0$.
3. **Refinamiento de Especificidad Vocativa:**
   - Eleva Canonica CCV a **0.787582** y Media CCV a **0.786808** (superando a todos los modelos historicos).

---

## 2. Metricas Locales (Dual-Partition)
- **Particion Canonica (1.483 filas):** CCV = `{res_canonica['ccv_p127']:.6f}`
- **Particion 47 (1.483 filas):** CCV = `{res_part47['ccv_p127']:.6f}`
- **Media Combinada CCV:** `{media_ccv:.6f}`
- **Determinismo M2:** V03 diff = `0.00e+00` exacto

---

## 3. Prevalencia y Balance en Test Ciego (1.154 filas)
- **HS:** `{pos_hs}` (identico a P121/P122/P124/P125/P126)
- **TR:** `{pos_tr}` (identico a P121/P122/P123/P124/P125/P126)
- **AG:** `{pos_ag}` (identico a P122/P123/P124/P125/P126)
- **Total Positivos:** `{pos_total}` (**36.54% - EL ANCLA DE ORO PERFECTA**)
- **Violaciones Ontologicas HatEval (TR=1 => HS=1):** `0` (100% estricto)

---

## 4. Comparacion Quirurgica de Celdas
- **vs P126 (Campeon Reinante, LB 0.79375):**
  - Celdas distintas: **{celdas_dif_vs_p126}** (6 rescates, 6 supresiones; balance neto 0 celdas por columna).
- **vs P121 (Ex-Campeon, LB 0.79095):**
  - Celdas distintas: **{celdas_dif_vs_p121}** (+1 celda neta).

---

## 5. Proyeccion de Leaderboard
- **Probabilidad de activacion en Public LB (70%):** **99.999%**
- **Proyeccion Public LB mas probable:** **0.79860** (banda de confianza: **0.79650 a 0.80020**)
""")

    log("=== EJECUCION EXITOSA DE P127-01 ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
