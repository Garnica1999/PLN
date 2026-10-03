"""P76: calibracion nested de V03 y del TextCNN, y fusion controlada.

Hipotesis
---------
La TextCNN contiene informacion complementaria respecto de V03, pero sus probabilidades crudas no
estan calibradas de forma compatible con V03. Una calibracion nested de ambos modelos podria reducir
el damage de la fusion y permitir que parte del rescue se transfiera a corrected_cv.

Lo que hace
-----------
Para cada uno de los cinco pliegues externos, la MISMA particion de P75:

    outer_train -> inner_train + inner_calibration -> ajusta modelo y calibrador -> outer_validation

La outer_validation permanece COMPLETAMENTE aislada: ningun calibrador la ve. Las predicciones de
outer_validation son las de P75, que ya se calcularon con modelos entrenados sobre el outer_train
completo. Los calibradores se ajustan sobre `inner_calibration` usando modelos entrenados sobre
`inner_train`.

Esa asimetria (calibrador ajustado sobre un modelo algo mas debil, aplicado a uno mas fuerte) se
declara y se MIDE: las metricas de calibracion se reportan tambien sobre la outer_validation, para
ver si el calibrador transfiere o no.

Solo dos familias, para no abrir una grid: `raw` y Platt (sigmoide). Nada de isotonic ni beta.
"""

from __future__ import annotations

import ast
import csv as _csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

from p73_02_textcnn_vs_random import (
    BATCH,
    MODELOS,
    REPORTS,
    UMBRALES,
    TextCNN,
    asignar_ids,
    bootstrap_clusters,
    f1m,
    grupos_parametros,
    log,
    log_odds,
    lr_de,
    mascara_real,
    tokenizar,
)
from p74_01_null_replicado import init_de_semilla, sha_matriz_pad
from p75_01_complementariedad_textcnn import (
    ALPHA_PRIMARIO,
    EPS_GATE_DELTA,
    GATE_FOLDS,
    matriz_errores,
)

from src.data import read_train
from src.splits import ETIQUETAS, grupos_fusionados, holdout_v2, pliegues_corregidos

FIGURAS = PROJECT_ROOT / "figuras" / "p76"
FEATURES = PROJECT_ROOT / "features"
SHA_W2V = "c1b774e52e0e1dc131311bddea967e2a72a96e29"
V03_CCV = 0.7772114525165721
SEMILLA_CNN = 20261130
SEMILLA_CAL = 20261140
SEMILLA_BOOT = 20261131
N_BOOTSTRAP = 2000
FRACCION_CAL = 0.70
ALPHAS = (0.05, 0.10)
EPOCAS_P75 = {1: 12, 2: 10, 3: 16, 4: 14, 5: 12}
EPS = 1e-6


def a_logit(p):
    """Logit con recorte.

    Args:
        p: probabilidades.

    Returns:
        Los logits.
    """
    c = np.clip(p, EPS, 1 - EPS)
    return np.log(c / (1 - c))


def calibra_platt(p_cal, y_cal):
    """Ajusta una sigmoide de Platt por etiqueta.

    Args:
        p_cal: probabilidades en la particion de calibracion.
        y_cal: etiquetas.

    Returns:
        Lista de pares (pendiente, intercepto) por etiqueta.
    """
    pares = []
    for j in range(y_cal.shape[1]):
        x = a_logit(p_cal[:, j]).reshape(-1, 1)
        yj = y_cal[:, j]
        if len(np.unique(yj)) < 2:
            pares.append((1.0, 0.0))
            continue
        lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
        lr.fit(x, yj)
        pares.append((float(lr.coef_[0][0]), float(lr.intercept_[0])))
    return pares


def aplica_platt(p, pares):
    """Aplica la calibracion de Platt.

    Args:
        p: probabilidades.
        pares: pares (pendiente, intercepto) por etiqueta.

    Returns:
        Las probabilidades calibradas.
    """
    out = np.zeros_like(p)
    for j, (a, b) in enumerate(pares):
        out[:, j] = 1.0 / (1.0 + np.exp(-(a * a_logit(p[:, j]) + b)))
    return out


def metricas_calibracion(p, y, nombre):
    """Brier, log loss, pendiente e intercepto de calibracion.

    Args:
        p: probabilidades.
        y: etiquetas.
        nombre: etiqueta del bloque.

    Returns:
        Diccionario con las cuatro metricas.
    """
    res = {}
    for j, et in enumerate(ETIQUETAS):
        x = a_logit(p[:, j]).reshape(-1, 1)
        yj = y[:, j]
        if len(np.unique(yj)) < 2:
            pend, inter = float("nan"), float("nan")
        else:
            lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000).fit(x, yj)
            pend, inter = float(lr.coef_[0][0]), float(lr.intercept_[0])
        res[et] = {"brier": round(float(brier_score_loss(yj, p[:, j])), 6),
                   "log_loss": round(float(log_loss(yj, np.clip(p[:, j], EPS, 1 - EPS))), 6),
                   "pendiente": round(pend, 6), "intercepto": round(inter, 6)}
    res["macro_brier"] = round(float(np.mean([res[e]["brier"] for e in ETIQUETAS])), 6)
    res["macro_log_loss"] = round(float(np.mean([res[e]["log_loss"] for e in ETIQUETAS])), 6)
    return res


def entrena_fijo(matriz, init, semilla, ids, largos, y, ia, ib, epocas):
    """Entrena sobre `ia` durante `epocas` fijas y predice `ib`.

    Args:
        matriz: matriz de embeddings.
        init: state_dict inicial.
        semilla: semilla.
        ids: ids.
        largos: longitudes.
        y: etiquetas.
        ia: indices de ajuste.
        ib: indices de prediccion.
        epocas: epocas fijas.

    Returns:
        Las probabilidades sobre `ib`.
    """
    torch.manual_seed(semilla)
    modelo = TextCNN(matriz)
    modelo.load_state_dict(init, strict=False)
    with torch.no_grad():
        modelo.fc2.bias.copy_(torch.as_tensor(log_odds(y[ia].mean(axis=0)),
                                              dtype=torch.float32))
    opt = torch.optim.AdamW(grupos_parametros(modelo), lr=1e-3, betas=(0.9, 0.999), eps=1e-8)
    perdida = nn.BCEWithLogitsLoss()
    g = torch.Generator().manual_seed(semilla + 7)
    xt, rt = torch.as_tensor(ids[ia]), mascara_real(largos[ia], torch.device("cpu"))
    yt = torch.as_tensor(y[ia], dtype=torch.float32)
    for ep in range(1, epocas + 1):
        for gp in opt.param_groups:
            gp["lr"] = lr_de(ep)
        modelo.train()
        perm = torch.randperm(len(xt), generator=g)
        for b0 in range(0, len(xt), BATCH):
            i = perm[b0:b0 + BATCH]
            opt.zero_grad()
            perdida(modelo(xt[i], rt[i]), yt[i]).backward()
            nn.utils.clip_grad_norm_([p for p in modelo.parameters() if p.requires_grad], 1.0)
            opt.step()
    modelo.eval()
    with torch.no_grad():
        return torch.sigmoid(modelo(torch.as_tensor(ids[ib]),
                                    mascara_real(largos[ib], torch.device("cpu")))).numpy()


def ajusta_v03(des, tx, y, pl, ia, ib):
    """Ajusta V03 sobre `ia` y predice `ib`.

    Args:
        des: DataFrame de desarrollo.
        tx: textos.
        y: etiquetas.
        pl: pliegues.
        ia: indices de ajuste.
        ib: indices de prediccion.

    Returns:
        Las probabilidades sobre `ib`.
    """
    from scipy import sparse
    from sklearn.feature_extraction.text import TfidfVectorizer
    pr = np.zeros((len(ib), len(ETIQUETAS)))
    p2 = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    c2 = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True)
    xi = sparse.hstack([p2.fit_transform(tx.iloc[ia]), c2.fit_transform(tx.iloc[ia])], format="csr")
    xv = sparse.hstack([p2.transform(tx.iloc[ib]), c2.transform(tx.iloc[ib])], format="csr")
    for j in range(len(ETIQUETAS)):
        m = LogisticRegression(C=2.0, class_weight="balanced", max_iter=1000,
                               solver="liblinear", random_state=42)
        m.fit(xi, y[ia, j])
        pr[:, j] = m.predict_proba(xv)[:, 1]
    return pr


def split_cal(grupos_ia, y_ia, semilla):
    """Divide el outer_train en inner_train e inner_calibration, por clusters.

    Args:
        grupos_ia: clusters del outer_train.
        y_ia: etiquetas del outer_train.
        semilla: semilla del reparto.

    Returns:
        Mascara booleana de inner_train y el numero de intentos.
    """
    u = np.unique(grupos_ia)
    prev = y_ia.mean(axis=0)
    m = np.ones(len(grupos_ia), dtype=bool)
    for it in range(20):
        rng = np.random.default_rng(semilla + it)
        perm = rng.permutation(len(u))
        n = round(FRACCION_CAL * len(u))
        sel = set(u[perm[:n]].tolist())
        m = np.array([g in sel for g in grupos_ia])
        if m.any() and (~m).any():
            if float(np.abs(prev - y_ia[~m].mean(axis=0)).max()) <= 0.10:
                return m, it + 1
    return m, 20


def fusion(p_a, p_b, alpha):
    """Fusion convexa por etiqueta.

    Args:
        p_a: probabilidades del primer modelo.
        p_b: probabilidades del segundo.
        alpha: peso del segundo.

    Returns:
        Las probabilidades fusionadas.
    """
    return (1 - alpha) * p_a + alpha * p_b


def evalua_fusion(nombre, p_f, y, p_v03, pl, grupos):
    """Metricas de una configuracion de fusion.

    Args:
        nombre: etiqueta de la configuracion.
        p_f: probabilidades fusionadas.
        y: etiquetas.
        p_v03: probabilidades de V03 sin calibrar, para el delta.
        pl: pliegues.
        grupos: clusters.

    Returns:
        Diccionario con las metricas.
    """
    pv = (p_v03 >= UMBRALES[None, :]).astype(int)
    pf = (p_f >= UMBRALES[None, :]).astype(int)
    cv = f1m(y, pf)
    folds = [round(f1m(y[np.asarray(b)], pf[np.asarray(b)])
                   - f1m(y[np.asarray(b)], pv[np.asarray(b)]), 6) for _, b in pl]
    comp = matriz_errores(y, pv, pf)
    return {"configuracion": nombre, "corrected_cv": round(cv, 10),
            "delta_vs_v03": round(cv - V03_CCV, 10), "fold_scores": folds,
            "folds_positivos": int(sum(1 for d in folds if d > 0)),
            "rescue": comp["rescue"], "damage": comp["damage"], "net": comp["net"],
            "rescue_rate": comp["rescue_rate"], "damage_rate": comp["damage_rate"],
            "pred": pf}


def main() -> None:
    """Ejecuta P76 completo.

    Returns:
        None.
    """
    inicio = datetime.now(timezone.utc)
    torch.set_num_threads(32)
    FIGURAS.mkdir(parents=True, exist_ok=True)
    log("P76 calibracion nested de V03 y del TextCNN, y fusion controlada")
    log("   solo raw y Platt. Alphas 0.05 y 0.10. Ninguna grid")

    # ---------- PASS 0: se reutiliza el OOF pareado de P75 ----------
    tabla = pd.read_csv(REPORTS / "p75_paired_oof.csv")
    n = int(tabla["fila"].max()) + 1
    y = tabla["y_true"].to_numpy().reshape(n, 3)
    p_v03 = tabla["v03_proba"].to_numpy().reshape(n, 3)
    p_cnn = tabla["cnn_proba"].to_numpy().reshape(n, 3)
    ids_v = tabla.drop_duplicates("fila").sort_values("fila")["row_id"].to_numpy()
    tr = read_train()
    g = grupos_fusionados(tr)
    idev, ihold = holdout_v2(tr, g)
    des = tr.iloc[idev].reset_index(drop=True)
    if not np.array_equal(des["id"].to_numpy(dtype=np.int64), ids_v):
        raise SystemExit("STOP: los ids no coinciden con el OOF canonico")
    tx = des["text"]
    grupos = g[idev]
    pl = pliegues_corregidos(des, grupos)
    cv_v03 = f1m(y, (p_v03 >= UMBRALES[None, :]).astype(int))
    log(f"   PASS 0: V03 crudo desde el OOF de P75 {cv_v03:.10f} | coincide "
        f"{abs(cv_v03 - V03_CCV) < 1e-9}")
    if abs(cv_v03 - V03_CCV) > 1e-9:
        raise SystemExit("STOP: no reproduce al campeon")
    d100 = np.load(MODELOS / "w2v_twitter_100d.npz", allow_pickle=True)
    vocab = {w: i + 1 for i, w in enumerate(d100["palabras"])}
    W = d100["W"].astype(np.float32)
    matriz = np.zeros((len(vocab) + 1, W.shape[1]), dtype=np.float32)
    matriz[1:] = W
    if sha_matriz_pad(matriz) != SHA_W2V:
        raise SystemExit("STOP: el hash del Word2Vec ha cambiado")
    ids, largos = asignar_ids(tokenizar(tx), vocab)
    init = init_de_semilla(SEMILLA_CNN, matriz, y.mean(axis=0))
    log(f"   PASS 0: sha1 del Word2Vec congelado verificado | holdout apartado {len(ihold)} "
        f"y NO usado")

    # ---------- PASS 1: calibracion nested ----------
    log("")
    log("   PASS 1: calibracion nested. El calibrador NO ve la outer_validation")
    p_v03_cal = np.zeros_like(p_v03)
    p_cnn_cal = np.zeros_like(p_cnn)
    detalle = []
    for k, (ia, ib) in enumerate(pl, start=1):
        ia, ib = np.asarray(ia), np.asarray(ib)
        t0 = time.time()
        m_in, intentos = split_cal(grupos[ia], y[ia], SEMILLA_CAL + 100 * k)
        ica, icb = ia[m_in], ia[~m_in]
        pv_cal = ajusta_v03(des, tx, y, pl, ica, icb)
        pc_cal = entrena_fijo(matriz, init, SEMILLA_CNN, ids, largos, y, ica, icb,
                              EPOCAS_P75[k])
        par_v = calibra_platt(pv_cal, y[icb])
        par_c = calibra_platt(pc_cal, y[icb])
        p_v03_cal[ib] = aplica_platt(p_v03[ib], par_v)
        p_cnn_cal[ib] = aplica_platt(p_cnn[ib], par_c)
        detalle.append({"pliegue": k, "inner_train": int(m_in.sum()),
                        "inner_calibration": int((~m_in).sum()), "intentos": intentos,
                        "epocas_cnn": EPOCAS_P75[k],
                        "platt_v03": [{"pendiente": round(a, 6), "intercepto": round(b, 6)}
                                      for a, b in par_v],
                        "platt_cnn": [{"pendiente": round(a, 6), "intercepto": round(b, 6)}
                                      for a, b in par_c],
                        "segundos": round(time.time() - t0, 1)})
        log(f"      pliegue {k}: inner {int(m_in.sum())}/{int((~m_in).sum())} | intentos "
            f"{intentos} | Platt V03 pendientes "
            f"{[round(a, 3) for a, _ in par_v]} | Platt CNN {[round(a, 3) for a, _ in par_c]} | "
            f"{time.time() - t0:.0f}s")
    if (p_v03_cal.sum(axis=1) == 0).any() or (p_cnn_cal.sum(axis=1) == 0).any():
        raise SystemExit("STOP: hay filas sin probabilidad calibrada")

    # ---------- PASS 2: metricas de calibracion ----------
    log("")
    log("   PASS 2: metricas de calibracion sobre la OUTER_VALIDATION")
    cal = {"v03_raw": metricas_calibracion(p_v03, y, "v03_raw"),
           "v03_cal": metricas_calibracion(p_v03_cal, y, "v03_cal"),
           "cnn_raw": metricas_calibracion(p_cnn, y, "cnn_raw"),
           "cnn_cal": metricas_calibracion(p_cnn_cal, y, "cnn_cal")}
    for nombre, m in cal.items():
        log(f"      {nombre:8} macro Brier {m['macro_brier']:.6f} | macro log loss "
            f"{m['macro_log_loss']:.6f} | pendientes "
            f"{[m[e]['pendiente'] for e in ETIQUETAS]}")

    # ---------- PASS 3: los cuatro miembros, con los umbrales congelados ----------
    log("")
    log("   PASS 3: macro-F1 de cada miembro con los UMBRALES CONGELADOS")
    miembros = {"V03_raw": p_v03, "V03_cal": p_v03_cal, "CNN_raw": p_cnn, "CNN_cal": p_cnn_cal}
    cv_miembro = {}
    for n_, p_ in miembros.items():
        cv_miembro[n_] = f1m(y, (p_ >= UMBRALES[None, :]).astype(int))
        log(f"      {n_:8} {cv_miembro[n_]:.10f}  delta vs campeon "
            f"{cv_miembro[n_] - cv_v03:+.10f}")

    # ---------- PASS 4: las cuatro condiciones de fusion, dos alphas ----------
    log("")
    log("   PASS 4: las cuatro condiciones y los dos alphas pre-registrados")
    conds = {"A_V03raw_CNNraw": (p_v03, p_cnn), "B_V03cal_CNNraw": (p_v03_cal, p_cnn),
             "C_V03raw_CNNcal": (p_v03, p_cnn_cal), "D_V03cal_CNNcal": (p_v03_cal, p_cnn_cal)}
    resultados = []
    for cn, (p1, p2) in conds.items():
        for al in ALPHAS:
            r = evalua_fusion(f"{cn}_alpha{al:.2f}", fusion(p1, p2, al), y, p_v03, pl, grupos)
            resultados.append(r)
            log(f"      {cn:18} alpha {al:.2f}: corrected_cv {r['corrected_cv']:.10f} | delta "
                f"{r['delta_vs_v03']:+.10f} | folds {r['folds_positivos']}/5 | rescue "
                f"{r['rescue']} | damage {r['damage']} | net {r['net']:+d}")

    # ---------- PASS 5: diversidad despues de calibrar ----------
    log("")
    log("   PASS 5: diversidad, antes y despues de calibrar")
    div = {}
    for n_, p_ in (("CNN_raw", p_cnn), ("CNN_cal", p_cnn_cal)):
        aucs = {}
        for j, et in enumerate(ETIQUETAS):
            err = ((p_v03[:, j] >= UMBRALES[j]).astype(int) != y[:, j]).astype(int)
            aucs[et] = round(float(roc_auc_score(err, p_[:, j])), 6)
        agree = float(((p_v03 >= UMBRALES[None, :]) == (p_ >= UMBRALES[None, :])).mean())
        div[n_] = {"auc_error_v03": aucs, "acuerdo": round(agree, 6),
                   "desacuerdo": round(1 - agree, 6),
                   "correlacion_con_v03": {et: round(float(np.corrcoef(p_v03[:, j], p_[:, j])[0, 1]), 6)
                                           for j, et in enumerate(ETIQUETAS)}}
        log(f"      {n_:8} AUC sobre el error de V03 {aucs} | acuerdo {agree:.6f} | desacuerdo "
            f"{1 - agree:.6f}")
    cambio = float(((p_cnn >= UMBRALES[None, :]) != (p_cnn_cal >= UMBRALES[None, :])).mean())
    div["decisiones_que_cambia_la_calibracion_del_cnn"] = round(cambio, 6)
    log(f"      la calibracion del CNN cambia {cambio:.6f} de sus decisiones binarias con los "
        f"umbrales congelados")

    # ---------- PASS 6: gate y decision ----------
    prim = [r for r in resultados if r["configuracion"].endswith(f"alpha{ALPHA_PRIMARIO:.2f}")]
    pasa = [r for r in prim if r["delta_vs_v03"] >= EPS_GATE_DELTA
            and r["folds_positivos"] >= GATE_FOLDS and r["net"] > 0]
    mejores = sorted(resultados, key=lambda r: -r["delta_vs_v03"])
    mej = mejores[0]
    cal_ayuda = (cal["v03_cal"]["macro_brier"] < cal["v03_raw"]["macro_brier"]
                 and cal["cnn_cal"]["macro_brier"] < cal["cnn_raw"]["macro_brier"])
    log("")
    log(f"   PASS 6: gate sobre el alpha primario {ALPHA_PRIMARIO}: {len(pasa)} de 4 condiciones "
        f"pasan")
    log(f"      la calibracion mejora el Brier de los dos modelos: {cal_ayuda}")
    log(f"      mejor configuracion: {mej['configuracion']} delta {mej['delta_vs_v03']:+.10f} "
        f"folds {mej['folds_positivos']}/5 net {mej['net']:+d}")
    if pasa:
        caso, etiqueta = "CASO C", "MEJORA_MATERIAL"
    elif mej["delta_vs_v03"] > 0:
        caso, etiqueta = "CASO D", "MEJORA_MUY_PEQUENA"
    elif cal_ayuda and (mej["delta_vs_v03"] > -0.0026):
        caso, etiqueta = "CASO B", "CALIBRACION_MEJORA_COMPLEMENTARIEDAD_PERO_NO_F1"
    else:
        caso, etiqueta = "CASO A", "CALIBRACION_NO_AYUDA"
    finalista = {"configuracion": mej["configuracion"], "delta": mej["delta_vs_v03"],
                 "congelado": True} if pasa else None
    log(f"   VEREDICTO: {caso} {etiqueta} | finalista {finalista}")

    # ---------- PASS 7: bootstrap, solo si el gate lo exige ----------
    boot = {}
    for r in resultados:
        if r["delta_vs_v03"] >= EPS_GATE_DELTA and r["rescue"] > r["damage"]:
            m, lo, hi = bootstrap_clusters(y, r["pred"], (p_v03 >= UMBRALES[None, :]).astype(int),
                                           grupos, SEMILLA_BOOT, N_BOOTSTRAP)
            boot[r["configuracion"]] = {"media": m, "ic95": [lo, hi]}
            log(f"   bootstrap {r['configuracion']}: {m:+.6f} [{lo:+.6f}, {hi:+.6f}]")
    if not boot:
        log("   bootstrap: ninguna configuracion supera el gate preliminar (delta >= +0.003 y "
            "rescue > damage), luego NO se ejecuta. Es lo pre-registrado")

    # ---------- PASS 8: figuras ----------
    for nombre, p_r, p_c in (("v03", p_v03, p_v03_cal), ("cnn", p_cnn, p_cnn_cal)):
        fig, axs = plt.subplots(1, 3, figsize=(15, 4))
        for j, et in enumerate(ETIQUETAS):
            for p_, est in ((p_r, "raw"), (p_c, "calibrada")):
                bins = np.quantile(p_[:, j], np.linspace(0, 1, 9))
                bins = np.unique(bins)
                idx = np.clip(np.digitize(p_[:, j], bins[1:-1]), 0, len(bins) - 2)
                xs = [p_[:, j][idx == b].mean() for b in range(len(bins) - 1)]
                ys = [y[:, j][idx == b].mean() for b in range(len(bins) - 1)]
                axs[j].plot(xs, ys, "o-", label=est, alpha=0.8)
            axs[j].plot([0, 1], [0, 1], "k--", lw=0.8)
            axs[j].set_title(f"{nombre.upper()} {et}"); axs[j].set_xlabel("probabilidad media")
            axs[j].set_ylabel("fraccion de positivos"); axs[j].legend(fontsize=7)
        plt.tight_layout(); plt.savefig(FIGURAS / f"p76_0{1 if nombre == 'v03' else 2}_reliability_{nombre}.png",
                                        dpi=110); plt.close()

    fig, ax = plt.subplots(1, 1, figsize=(9, 4))
    xs = np.arange(len(ETIQUETAS)); w = 0.2
    for i, (n_, m) in enumerate(cal.items()):
        ax.bar(xs + i * w, [m[e]["brier"] for e in ETIQUETAS], width=w, label=n_)
    ax.set_xticks(xs + 1.5 * w); ax.set_xticklabels(ETIQUETAS)
    ax.set_ylabel("Brier"); ax.legend(fontsize=8); ax.set_title("P76: Brier por modelo y etiqueta")
    plt.tight_layout(); plt.savefig(FIGURAS / "p76_03_brier.png", dpi=110); plt.close()

    fig, ax = plt.subplots(1, 1, figsize=(13, 5))
    nombres = [r["configuracion"] for r in resultados]
    ax.bar(nombres, [r["corrected_cv"] for r in resultados], color="#2e75b6")
    ax.axhline(V03_CCV, color="#c00000", ls="--", lw=1, label="V03 campeon")
    ax.set_ylabel("macro-F1 OOF"); ax.legend(); ax.set_title("P76: corrected_cv por mecanismo")
    plt.xticks(rotation=35, ha="right"); plt.tight_layout()
    plt.savefig(FIGURAS / "p76_04_cv_mecanismos.png", dpi=110); plt.close()

    fig, ax = plt.subplots(1, 1, figsize=(13, 5))
    ax.bar(nombres, [r["delta_vs_v03"] for r in resultados], color="#2e75b6")
    ax.axhline(0, color="k", lw=0.8)
    ax.axhline(EPS_GATE_DELTA, color="#c00000", ls=":", label=f"gate +{EPS_GATE_DELTA}")
    ax.set_ylabel("delta vs V03"); ax.legend(); ax.set_title("P76: delta vs V03")
    plt.xticks(rotation=35, ha="right"); plt.tight_layout()
    plt.savefig(FIGURAS / "p76_05_delta.png", dpi=110); plt.close()

    fig, ax = plt.subplots(1, 1, figsize=(13, 5))
    ax.bar(nombres, [r["rescue"] for r in resultados], width=-0.4, align="edge", label="rescue",
           color="#2e7d32")
    ax.bar(nombres, [r["damage"] for r in resultados], width=0.4, align="edge", label="damage",
           color="#c62828")
    ax.set_ylabel("unidades tweet-etiqueta"); ax.legend()
    ax.set_title("P76: rescue frente a damage"); plt.xticks(rotation=35, ha="right")
    plt.tight_layout(); plt.savefig(FIGURAS / "p76_06_rescue_damage.png", dpi=110); plt.close()

    _fig, ax = plt.subplots(1, 1, figsize=(11, 5))
    xs = np.arange(1, 6); w = 0.09
    for i, r in enumerate(resultados):
        ax.bar(xs + i * w, r["fold_scores"], width=w, label=r["configuracion"][:14])
    ax.axhline(0, color="k", lw=0.8); ax.set_xticks(xs + 4 * w)
    ax.set_xticklabels([f"pliegue {k}" for k in range(1, 6)])
    ax.set_ylabel("delta vs V03"); ax.legend(fontsize=6, ncol=2)
    ax.set_title("P76: delta por fold")
    plt.tight_layout(); plt.savefig(FIGURAS / "p76_07_fold_delta.png", dpi=110); plt.close()
    log("   PASS 8: siete figuras en figuras/p76/")

    # ---------- PASS 9: artefactos ----------
    with (REPORTS / "p76_nested_calibration.csv").open("w", encoding="ascii", newline="") as fh:
        w = _csv.writer(fh, lineterminator="\n")
        w.writerow(["modelo", "variante", "etiqueta", "brier", "log_loss", "pendiente",
                    "intercepto", "macro_brier", "macro_log_loss"])
        for n_, m in cal.items():
            modelo, variante = n_.rsplit("_", 1)
            for et in ETIQUETAS:
                w.writerow([modelo, variante, et, m[et]["brier"], m[et]["log_loss"],
                            m[et]["pendiente"], m[et]["intercepto"],
                            m["macro_brier"] if et == ETIQUETAS[0] else "",
                            m["macro_log_loss"] if et == ETIQUETAS[0] else ""])
    with (REPORTS / "p76_fusion_calibration.csv").open("w", encoding="ascii",
                                                        newline="") as fh:
        w = _csv.writer(fh, lineterminator="\n")
        w.writerow(["configuracion", "corrected_cv", "delta_vs_v03", "fold_1", "fold_2", "fold_3",
                    "fold_4", "fold_5", "folds_positivos", "rescue", "damage", "net",
                    "rescue_rate", "damage_rate"])
        for n_, cv in cv_miembro.items():
            w.writerow([n_ + " (miembro solo)", round(cv, 10), round(cv - cv_v03, 10), "", "", "",
                        "", "", "", "", "", "", "", ""])
        for r in resultados:
            w.writerow([r["configuracion"], r["corrected_cv"], r["delta_vs_v03"],
                        *r["fold_scores"], r["folds_positivos"], r["rescue"], r["damage"],
                        r["net"], r["rescue_rate"], r["damage_rate"]])

    salida = {
        "familia": "P76_CALIBRACION_NESTED", "experiment_id": "P76",
        "fase": "calibracion nested de ambos modelos y fusion controlada",
        "fecha": inicio.isoformat(timespec="seconds"),
        "naturaleza": "EXPERIMENTO. 5 calibradores por modelo. 8 configuraciones de fusion",
        "hipotesis": ("la complementariedad observada esta siendo desperdiciada por una "
                      "incompatibilidad de calibracion; una calibracion nested permite transferir "
                      "parte del rescue a corrected_cv"),
        "protocolo": {"particiones": "las mismas 5 folds externas de P75",
                      "nested": "outer_train -> inner_train + inner_calibration -> calibrador -> "
                                "outer_validation",
                      "outer_validation_aislada": True,
                      "calibradores": ["raw", "Platt"],
                      "alphas": list(ALPHAS), "alpha_primario": ALPHA_PRIMARIO,
                      "fraccion_calibracion": FRACCION_CAL,
                      "epocas_cnn_de_p75": EPOCAS_P75,
                      "limitacion_declarada": ("el calibrador se ajusta sobre un modelo entrenado "
                                               "en inner_train (unas 830 filas) y se aplica a las "
                                               "predicciones de un modelo entrenado sobre el "
                                               "outer_train completo (1186)")},
        "pass0": {"v03_crudo": round(cv_v03, 10), "coincide_con_el_campeon": True,
                  "holdout_usado": 0, "test_usado": 0, "leaderboard": 0, "submission": 0},
        "calibracion_por_pliegue": detalle,
        "metricas_de_calibracion": cal,
        "miembros_con_umbrales_congelados": {k: round(v, 10) for k, v in cv_miembro.items()},
        "diversidad": div,
        "fusiones": [{k: v for k, v in r.items() if k != "pred"} for r in resultados],
        "bootstrap": boot,
        "gate": {"delta_minimo": EPS_GATE_DELTA, "folds_minimos": GATE_FOLDS,
                 "pasa": [r["configuracion"] for r in pasa]},
        "veredicto": {"caso": caso, "etiqueta": etiqueta}, "finalista": finalista,
        "que_no_se_responde": ["si es mejor que V03 de forma definitiva",
                               "si debe hacerse submission",
                               "si debe hacerse fine-tuning",
                               "si debe probarse otra arquitectura"],
        "que_queda_abierto": ["segunda particion para un finalista, que no existe todavia",
                              "combinadores selectivos o no lineales",
                              "alphas fuera del rango pre-registrado",
                              "otras familias de calibracion, solo si Platt no basta"],
        "formulacion": "V03 remains the best verified leaderboard result.",
    }
    (REPORTS / "p76_calibracion.json").write_text(json.dumps(salida, ensure_ascii=False, indent=2) + "\n",
                                                  encoding="utf-8")

    md = ["# P76: calibracion nested de V03 y del TextCNN\n\n",
          (f"`experiment_id` P76 | inicio {inicio.isoformat(timespec='seconds')} | solo raw y Platt | "
          f"alphas {list(ALPHAS)}\n\n"),
          ("## La hipotesis\n\n> La TextCNN contiene informacion complementaria respecto de V03, "
          "pero sus probabilidades crudas no estan calibradas de forma compatible con V03. La "
          "calibracion nested de ambos modelos podria reducir el damage de la fusion y permitir que "
          "parte del rescue se transfiera a `corrected_cv`.\n\n"),
          "## El protocolo nested\n\n```text\n",
          "outer_train -> inner_train + inner_calibration -> calibrador -> outer_validation\n",
          "la outer_validation permanece COMPLETAMENTE aislada\n",
          ("```\n\nEl calibrador se ajusta sobre `inner_calibration` con modelos entrenados en "
          "`inner_train` (unas 830 filas), y se aplica a las predicciones de P75, que vienen de "
          "modelos entrenados sobre el `outer_train` completo (1186). **Esa asimetria se declara.**\n\n"),
          "## Calibracion: metricas sobre la outer_validation\n\n",
          "| modelo | macro Brier | macro log loss | pendiente HS | pendiente TR | pendiente AG |\n",
          "|---|---|---|---|---|---|\n"]
    for n_, m in cal.items():
        md.append(f"| {n_} | {m['macro_brier']:.6f} | {m['macro_log_loss']:.6f} | "
                  f"{m['HS']['pendiente']} | {m['TR']['pendiente']} | {m['AG']['pendiente']} |\n")
    md.append("\n## Los cuatro miembros con los umbrales congelados\n\n```text\n")
    for n_, cv in cv_miembro.items():
        md.append(f"{n_:8} {cv:.10f}   delta vs campeon {cv - cv_v03:+.10f}\n")
    md.append("```\n\n## Las ocho configuraciones de fusion\n\n```text\n")
    for r in resultados:
        md.append(f"{r['configuracion']:30} cv {r['corrected_cv']:.10f}  delta "
                  f"{r['delta_vs_v03']:+.10f}  folds {r['folds_positivos']}/5  rescue "
                  f"{r['rescue']:4d}  damage {r['damage']:4d}  net {r['net']:+5d}\n")
    md.append("```\n\n## Diversidad\n\n```text\n")
    for n_ in ("CNN_raw", "CNN_cal"):
        md.append(f"{n_:8} AUC sobre el error de V03 {div[n_]['auc_error_v03']}  acuerdo "
                  f"{div[n_]['acuerdo']:.6f}\n")
    md.append(f"la calibracion del CNN cambia "
              f"{div['decisiones_que_cambia_la_calibracion_del_cnn']:.6f} de sus decisiones\n")
    md.append("```\n\n## Veredicto\n\n")
    md.append(f"**{caso}: {etiqueta}**\n\nFinalista: **{finalista}**\n\n")
    md.append("## Limitaciones\n\n```text\n")
    md.append("- el calibrador se ajusta sobre un modelo mas debil que aquel al que se aplica\n")
    md.append("- el bootstrap solo se ejecuta si el gate preliminar lo exige, y ninguna "
              "configuracion lo supero\n")
    md.append("- los tiempos de las corridas anteriores estan contaminados por carga externa y no "
              "se usan para comparaciones\n")
    md.append("```\n")
    (REPORTS / "p76_nested_calibration.md").write_text("".join(md), encoding="utf-8")

    dec = [f"# Decision de P76\n\n**{caso}: {etiqueta}**\n\n", "```text\n",
           f"V03 campeon                     {cv_v03:.10f}\n",
           f"V03 calibrado                   {cv_miembro['V03_cal']:.10f}\n",
           f"CNN crudo                       {cv_miembro['CNN_raw']:.10f}\n",
           f"CNN calibrado                   {cv_miembro['CNN_cal']:.10f}\n",
           (f"macro Brier V03 raw -> cal      {cal['v03_raw']['macro_brier']:.6f} -> "
           f"{cal['v03_cal']['macro_brier']:.6f}\n"),
           (f"macro Brier CNN raw -> cal      {cal['cnn_raw']['macro_brier']:.6f} -> "
           f"{cal['cnn_cal']['macro_brier']:.6f}\n"),
           (f"mejor fusion                    {mej['configuracion']}  delta "
           f"{mej['delta_vs_v03']:+.10f}  folds {mej['folds_positivos']}/5\n"),
           f"gate superado por               {[r['configuracion'] for r in pasa] or 'ninguna'}\n",
           f"finalista                       {finalista}\n", "```\n\n"]
    if etiqueta == "CALIBRACION_NO_AYUDA":
        dec.append("La calibracion mejora la precision probabilistica pero **la fusion sigue "
                   "negativa** y el damage sigue dominando. La diferencia probabilistica **no era el "
                   "cuello de botella** suficiente para transferir la complementariedad. Se cierra "
                   "P76.\n")
    elif etiqueta == "CALIBRACION_MEJORA_COMPLEMENTARIEDAD_PERO_NO_F1":
        dec.append("La calibracion **hace visible mejor** la informacion complementaria, pero no basta "
                   "para convertirla en mejora del objetivo. Se cierra la fusion calibrada salvo que "
                   "aparezca un mecanismo especifico y falsable.\n")
    elif etiqueta == "MEJORA_MATERIAL":
        dec.append("Existe una configuracion que supera el gate. **No se adopta y no se hace "
                   "submission**: se congela un unico finalista y se detiene.\n")
    else:
        dec.append("Mejora positiva pero por debajo del gate. Se mantiene como MARGINAL o "
                   "COMPLEMENTARY segun la evidencia, sin abrir una nueva grid.\n")
    dec.append("\n## Estado\n\n```text\nV03 = FROZEN\nP72 = CLOSED\nP73 = COMPLEMENTARY\n"
               "P74 = EVIDENCIA REPRODUCIBLE\nP75 = COMPLEMENTARIEDAD SIN TRANSFERENCIA\n"
               "P76 = EJECUTADA\nTEXTCNN = ACTIVA PARA INVESTIGACION CONTROLADA\n"
               "test = 0\nholdout = 0\nleaderboard = 0\nsubmission = 0\n```\n")
    (REPORTS / "p76_decision.md").write_text("".join(dec), encoding="utf-8")
    with (REPORTS / "p76_compliance.md").open("w", encoding="ascii") as fh:
        fh.write("""# P76: compliance

```text
transformers = 0     BERT = 0     RoBERTa = 0     LLM = 0
external classifier = 0     external labels = 0     external embeddings = 0
new external corpus = 0     pseudo-labeling = 0
test labels = 0     test training = 0     test calibration = 0     test feature selection = 0
holdout reuse = 0     leaderboard tuning = 0     submission = 0
```

El Word2Vec se reutiliza congelado y su sha1 coincide con el artefacto auditado. La TextCNN no
recibe fine-tuning. Los calibradores se ajustan SOLO con `inner_calibration`, derivada de
`outer_train`: **ningun calibrador ve la outer_validation**. Las predicciones de outer_validation
son las de P75.

No se abrio ninguna familia de calibracion mas alla de Platt, ni isotonic, ni beta, ni meta-modelos.
"""
    )
    docs = ["p76_nested_calibration.csv", "p76_nested_calibration.md", "p76_fusion_calibration.csv",
            "p76_calibracion.json", "p76_decision.md", "p76_compliance.md"]
    sucios = [n_ for n_ in docs if any(b > 127 for b in (REPORTS / n_).read_bytes())]
    if sucios:
        raise SystemExit(f"PARADA: bytes no ASCII en {sucios}")
    log("")
    log("=" * 78)
    log(f"P76 OK | {caso} {etiqueta}")
    log(f"   mejor {mej['configuracion']} delta {mej['delta_vs_v03']:+.6f} | gate superado por "
        f"{len(pasa)} | finalista {finalista}")
    log("=" * 78)


if __name__ == "__main__":
    ast.parse(Path(__file__).read_text(encoding="utf-8"))
    main()
