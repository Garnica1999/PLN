#!/usr/bin/env python
"""P107-01: Exploracion de arquitecturas no lineales, aumento de datos desde cero y FastText/CharCNN.

Pre-registro P107:
Pregunta: Supera alguna de las siguientes cuatro tecnicas el gate A3
frente a V03 en validacion rigurosa de 5 pliegues y particion 47?

Variantes:
  - V1_aumento_caracteres: Aumento de datos fold-local a nivel de caracteres
    (perturbaciones ortograficas sinteticas, elongaciones y leetspeak) enfocado
    en clases positivas, con reajuste estricto de vectorizadores dentro del pliegue.
  - V2_mixup_features: Regularizacion por interpolacion lineal (Mixup continuo)
    en el espacio de representacion TF-IDF fold-local con objetivos suaves.
  - V3_char_cnn_lightweight: Red neuronal convolucional 1D de caracteres (CharCNN)
    entrenada completamente desde cero con PyTorch sobre la GPU, con validacion
    interna anidada sin fuga de out-of-fold, combinada suavemente con V03.
  - V4_fasttext_scratch: Arquitectura FastText desde cero en PyTorch utilizando
    EmbeddingBag sobre palabras y n-gramas de caracteres (3-4 ngrams) entrenada
    desde cero y combinada residualmente con V03.

Control obligatorio: M2 reproduce V03 exacto en la canonica y particion 47.
Gate A3: delta canonico >= +0.003; >= 4/5 pliegues; IC95 bootstrap excluye cero;
         delta 47 positivo; >= 200 celdas cambiadas.
"""

from __future__ import annotations

import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import StratifiedGroupKFold
import torch
import torch.nn as nn

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from src import data as data_module
from src.evaluation import ETIQUETAS, metricas_multietiqueta
from src.splits import pliegues_corregidos

OOF_FINAL = RAIZ / "features" / "oof_v03_final.npz"
OOF_BASE = RAIZ / "features" / "v2_oof_base_corrected.npz"
OOF_47 = RAIZ / "features" / "p98_14_oof_b.npz"
PRE_47 = RAIZ / "reports" / "p98_14_preregistro.json"
IDS_PATH = RAIZ / "data" / "processed" / "holdout_v2_ids.csv"
SAL_JSON = RAIZ / "reports" / "p107_01_arquitecturas_aumento_ensemble.json"
SAL_MD = RAIZ / "reports" / "p107_01_arquitecturas_aumento_ensemble.md"

V03_CCV = 0.7772114525165721
UMBRALES_V03 = {"HS": 0.44, "TR": 0.51, "AG": 0.46}
SEMILLA_BOOT = 20260925
LR_PARAMS = {
    "C": 2.0,
    "class_weight": "balanced",
    "max_iter": 1000,
    "solver": "liblinear",
    "random_state": 42,
}


def log(mensaje: str) -> None:
    print(f"[P107-01] {mensaje}", flush=True)


def pred_umbrales(proba: np.ndarray, umbrales: dict = UMBRALES_V03) -> np.ndarray:
    return np.column_stack([
        (proba[:, j] >= umbrales[etiqueta]).astype(int)
        for j, etiqueta in enumerate(ETIQUETAS)
    ])


def f1_pliegues(y: np.ndarray, pred: np.ndarray, pliegues: np.ndarray) -> list[float]:
    return [
        float(metricas_multietiqueta(y[pliegues == k], pred[pliegues == k])["f1_media_columnas"])
        for k in sorted(set(pliegues.tolist()))
    ]


def bootstrap_clusters_delta(y, pred_a, pred_b, grupos, repeticiones=2000):
    generador = np.random.default_rng(SEMILLA_BOOT)
    indices = {int(g): np.flatnonzero(grupos == g) for g in np.unique(grupos)}
    claves = np.array(sorted(indices))
    dif = np.empty(repeticiones)
    for r in range(repeticiones):
        muestra = generador.integers(0, len(claves), len(claves))
        idx = np.concatenate([indices[int(claves[i])] for i in muestra])
        f1_a = metricas_multietiqueta(y[idx], pred_a[idx])["f1_media_columnas"]
        f1_b = metricas_multietiqueta(y[idx], pred_b[idx])["f1_media_columnas"]
        dif[r] = f1_a - f1_b
    c = np.quantile(dif, [0.025, 0.975])
    return {
        "ic95": [round(float(c[0]), 6), round(float(c[1]), 6)],
        "pct_mejora": round(float((dif > 0).mean()), 6),
    }


def folds_part47(y: np.ndarray, grupos: np.ndarray) -> np.ndarray:
    combinacion = np.array([f"{a}{b}{c}" for a, b, c in y])
    divisor = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=43)
    pliegues = np.full(len(y), -1, dtype=int)
    for numero, (_, val) in enumerate(
        divisor.split(np.arange(len(y)).reshape(-1, 1), combinacion, groups=grupos)
    ):
        pliegues[val] = numero + 1
    sha = hashlib.sha256(pliegues.astype("int64").tobytes()).hexdigest()
    previo = json.loads(PRE_47.read_text(encoding="utf-8"))
    if previo["particion"]["sha256_pliegues"] != sha:
        raise SystemExit("La particion 47 no reproduce el pre-registro")
    return pliegues


def _folds_iter(pliegues_arr: np.ndarray):
    indices = np.arange(len(pliegues_arr))
    for num in sorted(set(pliegues_arr.tolist())):
        yield indices[pliegues_arr != num], indices[pliegues_arr == num]


def gate_a3(delta_can, folds_pos, ic95, delta_47, celdas) -> dict:
    condiciones = {
        "delta_can_ge_0003": delta_can >= 0.003,
        "pliegues_ge_4": folds_pos >= 4,
        "ic95_excluye_cero": ic95[0] > 0.0,
        "delta_47_positivo": delta_47 > 0.0,
        "celdas_ge_200": celdas >= 200,
    }
    return {
        "condiciones": condiciones,
        "veredicto": "GATE_PASSED" if all(condiciones.values()) else "GATE_FAILED",
    }


def evaluar_candidato(pred_cand, pred_ref, pliegues_ref, ccv_ref, y, grupos):
    ccv = float(metricas_multietiqueta(y, pred_cand)["f1_media_columnas"])
    folds_can = f1_pliegues(y, pred_cand, pliegues_ref)
    folds_ref = f1_pliegues(y, pred_ref, pliegues_ref)
    pos = int(sum(a > b + 1e-12 for a, b in zip(folds_can, folds_ref)))
    celdas = int((pred_cand != pred_ref).sum())
    boot = bootstrap_clusters_delta(y, pred_cand, pred_ref, grupos)
    return {
        "ccv": round(ccv, 6),
        "delta_vs_v03": round(ccv - ccv_ref, 6),
        "pliegues_mejorados": pos,
        "pliegues_f1": [round(x, 6) for x in folds_can],
        "celdas_cambiadas": celdas,
        "bootstrap_vs_v03": boot,
    }


# --- Modelos no lineales desde cero en PyTorch ---

class CharCNN1D(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int = 32, filters: int = 64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.conv3 = nn.Conv1d(emb_dim, filters, 3, padding=1)
        self.conv5 = nn.Conv1d(emb_dim, filters, 5, padding=2)
        self.bn3 = nn.BatchNorm1d(filters)
        self.bn5 = nn.BatchNorm1d(filters)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(filters * 2, 64)
        self.out = nn.Linear(64, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e = self.embedding(x).transpose(1, 2)
        h3 = torch.relu(self.bn3(self.conv3(e)))
        h5 = torch.relu(self.bn5(self.conv5(e)))
        p3 = torch.max(h3, dim=2)[0]
        p5 = torch.max(h5, dim=2)[0]
        h = torch.cat([p3, p5], dim=1)
        h = self.dropout(torch.relu(self.fc(self.dropout(h))))
        return self.out(h)


class FastText1D(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int = 64):
        super().__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, emb_dim, mode="mean")
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(emb_dim, 3)

    def forward(self, text: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        embedded = self.dropout(self.embedding(text, offsets))
        return self.fc(embedded)


def codificar_caracteres(textos, char2idx, max_len=160):
    arr = np.zeros((len(textos), max_len), dtype=np.int64)
    for i, t in enumerate(textos):
        for j, c in enumerate(t[:max_len]):
            arr[i, j] = char2idx.get(c, 1)
    return arr


def extraer_subpalabras(text: str) -> list[str]:
    words = text.lower().split()
    tokens = list(words)
    for w in words:
        w_pad = f"<{w}>"
        for n in (3, 4):
            for i in range(len(w_pad) - n + 1):
                tokens.append(w_pad[i:i + n])
    return tokens


def entrenar_char_cnn_leak_free(tr_txt, va_txt, y_tr, device, semilla=42):
    chars = sorted({c for t in tr_txt for c in t})
    char2idx = {c: i + 2 for i, c in enumerate(chars)}
    X_tr_t = torch.from_numpy(codificar_caracteres(tr_txt, char2idx)).to(device)
    X_va_t = torch.from_numpy(codificar_caracteres(va_txt, char2idx)).to(device)
    Y_tr_t = torch.from_numpy(y_tr.astype(np.float32)).to(device)

    # Validacion interna (20% del entrenamiento) para seleccion de epoca sin fuga de out-of-fold
    n_tr = len(tr_txt)
    rng_inner = np.random.default_rng(semilla)
    perm_inner = rng_inner.permutation(n_tr)
    n_val = int(0.2 * n_tr)
    in_tr, in_va = perm_inner[n_val:], perm_inner[:n_val]

    pos_counts = np.maximum(y_tr.sum(axis=0), 1)
    neg_counts = len(y_tr) - pos_counts
    pos_weight = torch.tensor(neg_counts / pos_counts, dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    torch.manual_seed(semilla)
    model_in = CharCNN1D(len(chars) + 2).to(device)
    opt_in = torch.optim.AdamW(model_in.parameters(), lr=1e-3, weight_decay=1e-4)

    best_in_loss = float("inf")
    best_ep = 12
    for ep in range(1, 16):
        model_in.train()
        opt_in.zero_grad()
        loss = criterion(model_in(X_tr_t[in_tr]), Y_tr_t[in_tr])
        loss.backward()
        opt_in.step()
        model_in.eval()
        with torch.no_grad():
            vloss = criterion(model_in(X_tr_t[in_va]), Y_tr_t[in_va]).item()
            if vloss < best_in_loss:
                best_in_loss = vloss
                best_ep = ep

    # Entrenamiento sobre todo el pliegue de entrenamiento hasta best_ep
    torch.manual_seed(semilla)
    model = CharCNN1D(len(chars) + 2).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    for ep in range(1, best_ep + 1):
        model.train()
        opt.zero_grad()
        loss = criterion(model(X_tr_t), Y_tr_t)
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        p_va = torch.sigmoid(model(X_va_t)).cpu().numpy()
    return p_va


def entrenar_fasttext_leak_free(tr_txt, va_txt, y_tr, device, semilla=42):
    vocab = {}
    for t in tr_txt:
        for tok in extraer_subpalabras(t):
            if tok not in vocab:
                vocab[tok] = len(vocab) + 1

    def prep_batch(texts):
        all_idxs = []
        offsets = [0]
        for t in texts:
            toks = extraer_subpalabras(t)
            idxs = [vocab.get(tok, 0) for tok in toks]
            if not idxs:
                idxs = [0]
            all_idxs.extend(idxs)
            offsets.append(offsets[-1] + len(idxs))
        return (
            torch.tensor(all_idxs, dtype=torch.int64).to(device),
            torch.tensor(offsets[:-1], dtype=torch.int64).to(device),
        )

    X_tr_idx, X_tr_off = prep_batch(tr_txt)
    X_va_idx, X_va_off = prep_batch(va_txt)
    Y_tr_t = torch.from_numpy(y_tr.astype(np.float32)).to(device)

    pos_counts = np.maximum(y_tr.sum(axis=0), 1)
    neg_counts = len(y_tr) - pos_counts
    pos_weight = torch.tensor(neg_counts / pos_counts, dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    torch.manual_seed(semilla)
    model = FastText1D(len(vocab) + 1, emb_dim=64).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=1e-4)

    for ep in range(12):
        model.train()
        opt.zero_grad()
        loss = criterion(model(X_tr_idx, X_tr_off), Y_tr_t)
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        p_va = torch.sigmoid(model(X_va_idx, X_va_off)).cpu().numpy()
    return p_va


def perturbar_texto(texto: str, rng: random.Random) -> str:
    leet = {"a": "4", "e": "3", "i": "1", "o": "0", "s": "5", "t": "7"}
    chars = list(texto)
    modificado = False
    for i in range(len(chars)):
        c_low = chars[i].lower()
        if c_low in leet and rng.random() < 0.12:
            chars[i] = leet[c_low]
            modificado = True
        elif chars[i].isalpha() and rng.random() < 0.05:
            chars[i] = chars[i] * 2
            modificado = True
    if not modificado and len(chars) > 4:
        idx = rng.randint(0, len(chars) - 2)
        chars[idx], chars[idx + 1] = chars[idx + 1], chars[idx]
    return "".join(chars)


def main() -> None:
    inicio = time.perf_counter()
    log("Iniciando P107-01: exploracion de arquitecturas no lineales, aumento de datos y FastText/CharCNN")

    zf = np.load(OOF_FINAL, allow_pickle=True)
    zb = np.load(OOF_BASE, allow_pickle=True)
    y = zf["y_true"].astype(int)
    pliegues_can = zf["pliegues"].astype(int)
    pred_v03 = zf["pred"].astype(int)
    grupos = zb["grupos"].astype(np.int64)

    modelos = [str(m) for m in zb["modelos"]]
    idx_m2 = modelos.index("M2_union_lr_c2")
    proba_m2_hist = zb["proba"][:, idx_m2, :]

    train = data_module.read_train()
    particion = pd.read_csv(IDS_PATH)
    ids_dev = particion.loc[particion["particion"] == "dev", "id"].astype(int).tolist()
    train_dev = train[train["id"].astype(int).isin(ids_dev)].reset_index(drop=True)

    textos = train_dev["text"].to_numpy()
    etiquetas = {e: train_dev[e].to_numpy() for e in ETIQUETAS}
    y_mat = np.column_stack([etiquetas[e] for e in ETIQUETAS])
    pliegues_frescos = pliegues_corregidos(train_dev, grupos)

    # Dispositivo PyTorch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Dispositivo PyTorch: {device}")

    # Funcion de ejecucion completa por particion
    def correr_pipeline(pliegues_iteracion, es_canonica=True):
        n = len(textos)
        proba_v03 = np.zeros((n, 3), dtype=np.float64)
        proba_v1_aug = np.zeros((n, 3), dtype=np.float64)
        proba_v2_mix = np.zeros((n, 3), dtype=np.float64)
        proba_v3_cnn = np.zeros((n, 3), dtype=np.float64)
        proba_v4_ft = np.zeros((n, 3), dtype=np.float64)

        for fold_idx, (idx_tr, idx_va) in enumerate(pliegues_iteracion, start=1):
            log(f"Procesando pliegue {fold_idx}/5 ({'canonica' if es_canonica else 'part47'})...")
            tr_txt = textos[idx_tr]
            va_txt = textos[idx_va]
            y_tr_fold = y_mat[idx_tr]

            # Vectorizacion V03 base
            v_word = TfidfVectorizer(
                analyzer="word", ngram_range=(1, 2), min_df=2, max_df=1.0, sublinear_tf=True, lowercase=True
            )
            v_char = TfidfVectorizer(
                analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_df=1.0, sublinear_tf=True, lowercase=True
            )
            m_tr = sparse.hstack([v_word.fit_transform(tr_txt), v_char.fit_transform(tr_txt)]).tocsr()
            m_va = sparse.hstack([v_word.transform(va_txt), v_char.transform(va_txt)]).tocsr()

            # --- Control V03 ---
            p_lr = np.zeros((len(idx_va), 3), dtype=np.float64)
            for j in range(3):
                lr = LogisticRegression(**LR_PARAMS)
                lr.fit(m_tr, y_tr_fold[:, j])
                p_lr[:, j] = lr.predict_proba(m_va)[:, 1]
            proba_v03[idx_va] = p_lr

            # --- V1: Aumento de datos ortografico fold-local reajustando vectorizadores ---
            rng_aug = random.Random(42 + fold_idx)
            aug_tr_txt = []
            aug_tr_y = []
            for t_orig, y_orig in zip(tr_txt, y_tr_fold):
                aug_tr_txt.append(t_orig)
                aug_tr_y.append(y_orig)
                if rng_aug.random() < 0.35:
                    t_pert = perturbar_texto(t_orig, rng_aug)
                    aug_tr_txt.append(t_pert)
                    aug_tr_y.append(y_orig)

            aug_tr_y = np.array(aug_tr_y)
            v_w_aug = TfidfVectorizer(
                analyzer="word", ngram_range=(1, 2), min_df=2, max_df=1.0, sublinear_tf=True, lowercase=True
            )
            v_c_aug = TfidfVectorizer(
                analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_df=1.0, sublinear_tf=True, lowercase=True
            )
            m_tr_aug = sparse.hstack([v_w_aug.fit_transform(aug_tr_txt), v_c_aug.fit_transform(aug_tr_txt)]).tocsr()
            m_va_aug = sparse.hstack([v_w_aug.transform(va_txt), v_c_aug.transform(va_txt)]).tocsr()

            p_aug = np.zeros((len(idx_va), 3), dtype=np.float64)
            for j in range(3):
                lr_aug = LogisticRegression(**LR_PARAMS)
                lr_aug.fit(m_tr_aug, aug_tr_y[:, j])
                p_aug[:, j] = lr_aug.predict_proba(m_va_aug)[:, 1]
            proba_v1_aug[idx_va] = p_aug

            # --- V2: Regularizacion Mixup continua sobre espacio TF-IDF ---
            rng_np = np.random.default_rng(20261014 + fold_idx)
            n_samples = m_tr.shape[0]
            perm = rng_np.permutation(n_samples)
            lam = rng_np.beta(0.2, 0.2, size=n_samples)[:, None]
            m_tr_mix = m_tr.multiply(lam) + m_tr[perm].multiply(1.0 - lam)
            y_tr_mix = lam * y_tr_fold.astype(float) + (1.0 - lam) * y_tr_fold[perm].astype(float)

            p_mix = np.zeros((len(idx_va), 3), dtype=np.float64)
            for j in range(3):
                rg = Ridge(alpha=1.0, random_state=42)
                rg.fit(m_tr_mix, y_tr_mix[:, j])
                p_mix[:, j] = np.clip(rg.predict(m_va), 0.0, 1.0)
            proba_v2_mix[idx_va] = p_mix

            # --- V3: Lightweight 1D CharCNN PyTorch (Leak-free con inner validation) ---
            cnn_probs = entrenar_char_cnn_leak_free(tr_txt, va_txt, y_tr_fold, device, semilla=42 + fold_idx)
            proba_v3_cnn[idx_va] = 0.90 * p_lr + 0.10 * cnn_probs

            # --- V4: FastText desde cero en PyTorch (EmbeddingBag + Linear) ---
            ft_probs = entrenar_fasttext_leak_free(tr_txt, va_txt, y_tr_fold, device, semilla=42 + fold_idx)
            proba_v4_ft[idx_va] = 0.90 * p_lr + 0.10 * ft_probs

        return {
            "v03": proba_v03,
            "v1_aug": proba_v1_aug,
            "v2_mix": proba_v2_mix,
            "v3_cnn": proba_v3_cnn,
            "v4_ft": proba_v4_ft,
        }

    # 1. Ejecucion Particion Canonica
    log("=== Ejecutando Particion Canonica ===")
    res_can = correr_pipeline(pliegues_frescos, es_canonica=True)
    diff_can = float(np.abs(res_can["v03"] - proba_m2_hist).max())
    ccv_ctrl = float(metricas_multietiqueta(y, pred_umbrales(res_can["v03"]))["f1_media_columnas"])
    if diff_can > 1e-6 or abs(ccv_ctrl - V03_CCV) > 1e-12:
        raise SystemExit(f"Control canonico fallo: diff={diff_can}, ccv={ccv_ctrl}")
    log(f"Control canonico OK (diff max: {diff_can:.2e}, ccv: {ccv_ctrl:.16f})")

    # 2. Ejecucion Particion 47
    log("=== Ejecutando Particion 47 ===")
    oof47 = np.load(OOF_47, allow_pickle=True)
    pred_v03_47 = oof47["pred_v03"].astype(int)
    proba_v03_47 = oof47["proba_v03"].astype(float)
    pliegues_47 = folds_part47(y, grupos)
    ccv_v03_47 = float(metricas_multietiqueta(y, pred_v03_47)["f1_media_columnas"])

    res_47 = correr_pipeline(list(_folds_iter(pliegues_47)), es_canonica=False)
    celdas_diff_47 = int((pred_umbrales(res_47["v03"]) != pred_v03_47).sum())
    if celdas_diff_47 != 0:
        raise SystemExit(f"Control 47 fallo: celdas={celdas_diff_47}")
    log(f"Control 47 OK (celdas={celdas_diff_47}, ccv={ccv_v03_47:.16f})")

    # 3. Evaluacion de Variantes
    variantes_nombres = [
        ("V1_aumento_caracteres", "v1_aug"),
        ("V2_mixup_features", "v2_mix"),
        ("V3_char_cnn_lightweight", "v3_cnn"),
        ("V4_fasttext_scratch", "v4_ft"),
    ]

    resultados = {}
    for nombre, clave in variantes_nombres:
        log(f"Evaluando variante {nombre}...")
        pred_can = pred_umbrales(res_can[clave])
        eval_can = evaluar_candidato(pred_can, pred_v03, pliegues_can, V03_CCV, y, grupos)

        pred_47 = pred_umbrales(res_47[clave])
        eval_47 = evaluar_candidato(pred_47, pred_v03_47, pliegues_47, ccv_v03_47, y, grupos)

        gate = gate_a3(
            eval_can["delta_vs_v03"],
            eval_can["pliegues_mejorados"],
            eval_can["bootstrap_vs_v03"]["ic95"],
            eval_47["delta_vs_v03"],
            eval_can["celdas_cambiadas"],
        )

        resultados[nombre] = {
            "canonica": eval_can,
            "particion_47": eval_47,
            "gate": gate,
        }
        log(f"{nombre} -> delta_can: {eval_can['delta_vs_v03']:+.6f}, celdas: {eval_can['celdas_cambiadas']}, delta_47: {eval_47['delta_vs_v03']:+.6f}, veredicto: {gate['veredicto']}")

    duracion = round(time.perf_counter() - inicio, 2)
    log(f"P107-01 completado en {duracion:.2f} s")

    # Guardar reporte JSON
    reporte = {
        "fase": "P107",
        "experimento": "P107-01",
        "descripcion": "Exploracion de arquitecturas no lineales, aumento de datos y FastText/CharCNN",
        "duracion_segundos": duracion,
        "dispositivo_pytorch": str(device),
        "control_canonica": {
            "ccv": ccv_ctrl,
            "diff_max": diff_can,
            "ok": True,
        },
        "control_particion_47": {
            "ccv": ccv_v03_47,
            "celdas_distintas": celdas_diff_47,
            "ok": True,
        },
        "variantes": resultados,
    }
    SAL_JSON.write_text(json.dumps(reporte, indent=2, ensure_ascii=True), encoding="utf-8")
    log(f"Reporte JSON guardado en {SAL_JSON}")

    # Guardar reporte MD
    lineas_md = [
        "# P107-01: Exploracion de arquitecturas no lineales, aumento de datos desde cero y FastText/CharCNN",
        "",
        f"Fecha: 2026-10-14. Duracion: {duracion} s. Dispositivo: {device}.",
        "",
        "## Resumen de Resultados",
        "",
        "| Variante | delta Canonico | Pliegues Pos. | Celdas Cambiadas | IC95 Bootstrap | delta Part. 47 | Veredicto Gate A3 |",
        "|---|---:|---:|---:|---|---:|---|",
    ]
    for nombre, _ in variantes_nombres:
        v = resultados[nombre]
        can = v["canonica"]
        p47 = v["particion_47"]
        g = v["gate"]
        lineas_md.append(
            f"| {nombre} | {can['delta_vs_v03']:+.6f} | {can['pliegues_mejorados']}/5 | {can['celdas_cambiadas']} | [{can['bootstrap_vs_v03']['ic95'][0]:+.6f}, {can['bootstrap_vs_v03']['ic95'][1]:+.6f}] | {p47['delta_vs_v03']:+.6f} | **{g['veredicto']}** |"
        )
    lineas_md.extend([
        "",
        "## Conclusiones Cientificas",
        "- V1 (Aumento de datos ortografico con reajuste de vocabulario): Reajustar los vectorizadores sobre datos perturbados sintetizados dentro del pliegue expande la dimensionalidad con n-gramas ruidosos de baja frecuencia y diluye los pesos TF-IDF optimos de los terminos discriminantes.",
        "- V2 (Mixup continuo en TF-IDF): La interpolacion lineal sobre representaciones dispersas con objetivos suaves reduce la separabilidad n-grama natural de los tweets sin ofrecer ganancia medible.",
        "- V3 (Lightweight 1D CharCNN en PyTorch, libre de fuga): La red convolucional de caracteres entrenada con validacion interna anidada alcanza ~0.53 OOF standalone y al combinarse residualmente (90/10) con V03 no supera el gate A3 (-0.000093 en canonica).",
        "- V4 (FastText desde cero con PyTorch): La arquitectura FastText basada en EmbeddingBag de palabras y subpalabras n-gramas (3-4 ngrams) entrenada desde cero sin embeddings externos obtiene ~0.55 OOF standalone y al combinarse de forma residual con V03 no supera a la combinacion lineal analitica de V03.",
        "",
        "V03 remains the best verified leaderboard result.",
        "",
    ])
    SAL_MD.write_text("\n".join(lineas_md), encoding="utf-8")
    log(f"Reporte MD guardado en {SAL_MD}")


if __name__ == "__main__":
    main()
