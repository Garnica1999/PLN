"""P75: complementariedad de Word2Vec + TextCNN frente a V03, y una fusion minima.

Pre-registro: docs/p75_preregistro_fusion.md (ver la cabecera de constantes). Un solo
experimento con tres configuraciones de alpha del mismo experimento.

Lo que responde
--------------
La informacion que Word2Vec + TextCNN contiene, corrige errores que V03 no corrige, o es una
representacion distinta pero esencialmente redundante?

La cadena causal que este script NO rompe:
    P72  mean pooling no demuestra ventaja
    P73  la TextCNN revela una posible diferencia
    P74  esa diferencia sobrevive al nulo aleatorio replicado (10 de 10)
    P75  comprobar si esa informacion es complementaria y TRANSFERIBLE a V03

Nivel 1 es "la representacion contiene informacion de tarea", ya establecido por P74.
Nivel 2 es "los errores son complementarios", descriptivo.
Nivel 3 es "la fusion mejora a V03", y no se sigue de 1 ni de 2.
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
from sklearn.metrics import roc_auc_score
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
    partir_interna,
    tokenizar,
)
from p74_01_null_replicado import init_de_semilla, sha_matriz_pad

from src.data import read_train
from src.splits import ETIQUETAS, grupos_fusionados, holdout_v2, pliegues_corregidos

FIGURAS = PROJECT_ROOT / "figuras" / "p75"
FEATURES = PROJECT_ROOT / "features"
SHA_W2V = "c1b774e52e0e1dc131311bddea967e2a72a96e29"
V03_CCV = 0.7772114525165721
SEMILLA_CNN = 20261130
SEMILLA_BOOT = 20261131
N_BOOTSTRAP = 2000
ALPHAS = (0.05, 0.10, 0.15)
ALPHA_PRIMARIO = 0.10
EPS_GATE_DELTA = 0.003
GATE_FOLDS = 4
N_INTERNOS = 3
# Lista declarada solo para rebanar el diagnostico. NO es una feature y no entra en ningun modelo.
LISTA_INSULTOS = ("puta", "puto", "mierda", "gilipollas", "idiota", "imbecil", "estupido",
                  "cabron", "pendejo", "verga", "joder", "cojones", "zorra", "marica",
                  "malparido", "gonorrea", "hp", "hijueputa")


def f1_filas(y, pred, idx=None):
    """Macro-F1 sobre un subconjunto de filas.

    Args:
        y: etiquetas.
        pred: predicciones binarias.
        idx: indices, o None para todas.

    Returns:
        El macro-F1.
    """
    if idx is None:
        return f1m(y, pred)
    return f1m(y[idx], pred[idx])


def epoca_estrella(matriz, init, semilla, ids, largos, y, grupos, ia, k):
    """Epoca E* como MEDIANA de tres splits internos por clusters.

    Args:
        matriz: matriz de embeddings.
        init: state_dict inicial.
        semilla: semilla de modelo.
        ids: ids.
        largos: longitudes.
        y: etiquetas.
        grupos: clusters.
        ia: indices del pliegue de entrenamiento externo.
        k: numero de pliegue, desde 1.

    Returns:
        La epoca E* y las epocas de los tres splits.
    """
    epocas = []
    for j in range(N_INTERNOS):
        m_in, _ = partir_interna(grupos[ia], y[ia], 20261101 + 100 * k + 7 * j)
        torch.manual_seed(semilla + j)
        modelo = TextCNN(matriz)
        modelo.load_state_dict(init, strict=False)
        with torch.no_grad():
            modelo.fc2.bias.copy_(torch.as_tensor(
                log_odds(y[ia][m_in].mean(axis=0)), dtype=torch.float32))
        opt = torch.optim.AdamW(grupos_parametros(modelo), lr=1e-3, betas=(0.9, 0.999),
                                eps=1e-8)
        perdida = nn.BCEWithLogitsLoss()
        g_perm = torch.Generator().manual_seed(semilla + 7 + j)
        xt = torch.as_tensor(ids[ia][m_in])
        rt = mascara_real(largos[ia][m_in], torch.device("cpu"))
        yt = torch.as_tensor(y[ia][m_in], dtype=torch.float32)
        xv = torch.as_tensor(ids[ia][~m_in])
        rv = mascara_real(largos[ia][~m_in], torch.device("cpu"))
        yv = torch.as_tensor(y[ia][~m_in], dtype=torch.float32)
        suaves, mejor, mejor_ep = [], float("inf"), 1
        for epoca in range(1, 41):
            for gp in opt.param_groups:
                gp["lr"] = lr_de(epoca)
            modelo.train()
            perm = torch.randperm(len(xt), generator=g_perm)
            for b0 in range(0, len(xt), BATCH):
                idx = perm[b0:b0 + BATCH]
                opt.zero_grad()
                perdida(modelo(xt[idx], rt[idx]), yt[idx]).backward()
                nn.utils.clip_grad_norm_([p for p in modelo.parameters()
                                          if p.requires_grad], 1.0)
                opt.step()
            modelo.eval()
            with torch.no_grad():
                pv = float(perdida(modelo(xv, rv), yv).item())
            suaves.append(float(np.mean([pv] + suaves[-2:])) if suaves else pv)
            if suaves[-1] < mejor - 1e-3:
                mejor, mejor_ep = suaves[-1], epoca
            if epoca - mejor_ep >= 6:
                break
        epocas.append(mejor_ep)
    return int(np.median(epocas)), epocas


def entrenar_fijo(matriz, init, semilla, ids, largos, y, ia, ib, epocas):
    """Reentrena sobre el outer_train COMPLETO durante E* epocas fijas.

    Args:
        matriz: matriz de embeddings.
        init: state_dict inicial.
        semilla: semilla de modelo.
        ids: ids.
        largos: longitudes.
        y: etiquetas.
        ia: indices de entrenamiento externo.
        ib: indices de validacion externa.
        epocas: numero de epocas, fijo.

    Returns:
        Las probabilidades sobre la validacion externa.
    """
    torch.manual_seed(semilla)
    modelo = TextCNN(matriz)
    modelo.load_state_dict(init, strict=False)
    with torch.no_grad():
        modelo.fc2.bias.copy_(torch.as_tensor(log_odds(y[ia].mean(axis=0)),
                                              dtype=torch.float32))
    opt = torch.optim.AdamW(grupos_parametros(modelo), lr=1e-3, betas=(0.9, 0.999), eps=1e-8)
    perdida = nn.BCEWithLogitsLoss()
    g_perm = torch.Generator().manual_seed(semilla + 7)
    xt = torch.as_tensor(ids[ia])
    rt = mascara_real(largos[ia], torch.device("cpu"))
    yt = torch.as_tensor(y[ia], dtype=torch.float32)
    for epoca in range(1, epocas + 1):
        for gp in opt.param_groups:
            gp["lr"] = lr_de(epoca)
        modelo.train()
        perm = torch.randperm(len(xt), generator=g_perm)
        for b0 in range(0, len(xt), BATCH):
            idx = perm[b0:b0 + BATCH]
            opt.zero_grad()
            perdida(modelo(xt[idx], rt[idx]), yt[idx]).backward()
            nn.utils.clip_grad_norm_([p for p in modelo.parameters() if p.requires_grad], 1.0)
            opt.step()
    modelo.eval()
    with torch.no_grad():
        out1 = torch.sigmoid(modelo(torch.as_tensor(ids[ib]),
                                    mascara_real(largos[ib], torch.device("cpu"))))
        out2 = torch.sigmoid(modelo(torch.as_tensor(ids[ib]),
                                    mascara_real(largos[ib], torch.device("cpu"))))
    if not torch.equal(out1, out2):
        raise SystemExit("STOP: el modo eval no es determinista")
    return out1.numpy()


def n_alcanzables(p_v03, p_cnn, alpha):
    """Unidades que PUEDEN voltear con ese alpha, por la condicion geometrica exacta.

    Args:
        p_v03: probabilidades de V03 (filas, etiquetas).
        p_cnn: probabilidades del CNN.
        alpha: peso del CNN.

    Returns:
        Numero de unidades alcanzables.
    """
    return int((np.abs(UMBRALES[None, :] - p_v03) <= alpha * np.abs(p_cnn - p_v03)).sum())


def matriz_errores(y, pa, pb, mascara=None):
    """Matriz de interseccion de errores en unidades tweet-etiqueta.

    Args:
        y: etiquetas.
        pa: predicciones del modelo A (V03).
        pb: predicciones del modelo B (CNN).
        mascara: mascara booleana de filas, o None.

    Returns:
        Diccionario con los cuatro recuentos y los derivados.
    """
    if mascara is None:
        mascara = np.ones(len(y), dtype=bool)
    ya = y[mascara].ravel()
    ok_a = ya == pa[mascara].ravel()
    ok_b = ya == pb[mascara].ravel()
    rescue = int((~ok_a & ok_b).sum())
    damage = int((ok_a & ~ok_b).sum())
    ambos_ok = int((ok_a & ok_b).sum())
    ambos_ko = int((~ok_a & ~ok_b).sum())
    n = int(ya.size)
    return {"unidades": n, "rescue": rescue, "damage": damage, "ambos_ok": ambos_ok,
            "ambos_ko": ambos_ko, "net": rescue - damage,
            "rescue_rate": round(rescue / max(n, 1), 6),
            "damage_rate": round(damage / max(n, 1), 6),
            "net_rate": round((rescue - damage) / max(n, 1), 6),
            "v03_error": int((~ok_a).sum()), "cnn_error": int((~ok_b).sum())}


def main() -> None:
    """Ejecuta P75 completo.

    Returns:
        None.
    """
    inicio = datetime.now(timezone.utc)
    torch.set_num_threads(32)
    FIGURAS.mkdir(parents=True, exist_ok=True)
    log("P75 complementariedad de Word2Vec + TextCNN frente a V03, y fusion minima")
    log(f"   alpha primario {ALPHA_PRIMARIO} | sensibilidad {list(ALPHAS)}")
    log(f"   gate: delta >= {EPS_GATE_DELTA} y >= {GATE_FOLDS}/5 folds y net positivo")

    # ---------- PASS 0: integridad ----------
    v = np.load(FEATURES / "oof_v03_final.npz", allow_pickle=True)
    y = v["y_true"].astype(int)
    p_v03 = v["proba"].astype(float)
    ids_v03 = v["ids"].astype(np.int64)
    tr = read_train()
    g = grupos_fusionados(tr)
    idev, ihold = holdout_v2(tr, g)
    des = tr.iloc[idev].reset_index(drop=True)
    if not np.array_equal(des["id"].to_numpy(dtype=np.int64), ids_v03):
        raise SystemExit("STOP: los ids no coinciden con el OOF canonico")
    grupos = g[idev]
    pl = pliegues_corregidos(des, grupos)
    cv_v03 = f1m(y, (p_v03 >= UMBRALES[None, :]).astype(int))
    log("")
    log(f"   PASS 0: V03 desde el OOF canonico {cv_v03:.10f} | esperado {V03_CCV:.10f} | "
        f"coincide {abs(cv_v03 - V03_CCV) < 1e-9}")
    if abs(cv_v03 - V03_CCV) > 1e-9:
        raise SystemExit("STOP: la reconstruccion de V03 no reproduce el campeon")
    cubre = np.zeros(len(y), dtype=int)
    folde = np.zeros(len(y), dtype=int)
    for k, (_, b) in enumerate(pl, start=1):
        cubre[np.asarray(b)] += 1
        folde[np.asarray(b)] = k
    if not (cubre == 1).all():
        raise SystemExit("STOP: los pliegues no cubren las filas exactamente una vez")

    d100 = np.load(MODELOS / "w2v_twitter_100d.npz", allow_pickle=True)
    vocab = {w: i + 1 for i, w in enumerate(d100["palabras"])}
    W = d100["W"].astype(np.float32)
    matriz = np.zeros((len(vocab) + 1, W.shape[1]), dtype=np.float32)
    matriz[1:] = W
    sha = sha_matriz_pad(matriz)
    log(f"   PASS 0: sha1 del Word2Vec congelado {sha} | coincide con el auditado "
        f"{sha == SHA_W2V}")
    if sha != SHA_W2V:
        raise SystemExit("STOP: el hash del Word2Vec ha cambiado")

    ids, largos = asignar_ids(tokenizar(des["text"]), vocab)
    prev_global = y.mean(axis=0)
    init = init_de_semilla(SEMILLA_CNN, matriz, prev_global)

    # ---------- PASS 1: OOF pareado del CNN ----------
    log("")
    log("   PASS 1: OOF pareado del TextCNN, reentrenado sobre el outer_train completo a E*")
    p_cnn = np.zeros_like(p_v03)
    epocas_por_fold = []
    for k, (ia, ib) in enumerate(pl, start=1):
        ia, ib = np.asarray(ia), np.asarray(ib)
        t0 = time.time()
        e_star, e3 = epoca_estrella(matriz, init, SEMILLA_CNN, ids, largos, y, grupos, ia, k)
        p_cnn[ib] = entrenar_fijo(matriz, init, SEMILLA_CNN, ids, largos, y, ia, ib, e_star)
        epocas_por_fold.append({"pliegue": k, "E_estrella": e_star, "epocas_de_los_tres": e3,
                                "n_outer_train": len(ia), "n_outer_val": len(ib),
                                "segundos": round(time.time() - t0, 1)})
        log(f"      pliegue {k}: E* {e_star} (tres splits {e3}) | outer_train {len(ia)} | "
            f"outer_val {len(ib)} | {time.time() - t0:.0f}s")
    if (p_cnn.sum(axis=1) == 0).any() or not np.isfinite(p_cnn).all():
        raise SystemExit("STOP: hay filas sin prediccion del CNN")

    pred_v03 = (p_v03 >= UMBRALES[None, :]).astype(int)
    pred_cnn = (p_cnn >= UMBRALES[None, :]).astype(int)
    cv_cnn = f1m(y, pred_cnn)
    log(f"      CNN sola (diagnostica): {cv_cnn:.10f} | V03 {cv_v03:.10f}")

    # ---------- la tabla unica, y la comprobacion de alineamiento ----------
    filas_tabla = []
    for i in range(len(y)):
        for j, et in enumerate(ETIQUETAS):
            filas_tabla.append({"row_id": int(ids_v03[i]), "fila": i, "fold": int(folde[i]),
                                "label": et, "y_true": int(y[i, j]),
                                "v03_proba": round(float(p_v03[i, j]), 8),
                                "cnn_proba": round(float(p_cnn[i, j]), 8),
                                "v03_pred": int(pred_v03[i, j]),
                                "cnn_pred": int(pred_cnn[i, j])})
    tabla = pd.DataFrame(filas_tabla)
    y_t = tabla["y_true"].to_numpy().reshape(-1, 3)
    tabla["v03_pred"].to_numpy().reshape(-1, 3)
    if len(tabla) != 1483 * 3 or not np.array_equal(y_t, y):
        raise SystemExit("STOP: la tabla unica no tiene 4449 unidades alineadas")
    cv_tabla = f1m(tabla["y_true"].to_numpy().reshape(-1, 3),
                   tabla["v03_pred"].to_numpy().reshape(-1, 3))
    log(f"      la tabla unica tiene {len(tabla)} unidades tweet-etiqueta | V03 recalculado "
        f"desde la tabla {cv_tabla:.10f} | coincide {abs(cv_tabla - V03_CCV) < 1e-9}")
    if abs(cv_tabla - V03_CCV) > 1e-9:
        raise SystemExit("STOP: la tabla unica no reproduce a V03. El pareado esta desalineado")

    # ---------- PASS 2: complementariedad ----------
    log("")
    log("   PASS 2: matriz de interseccion de errores, en unidades tweet-etiqueta")
    comp_global = matriz_errores(y, pred_v03, pred_cnn)
    log(f"      GLOBAL  n {comp_global['unidades']} | rescue {comp_global['rescue']} | damage "
        f"{comp_global['damage']} | ambos ok {comp_global['ambos_ok']} | ambos ko "
        f"{comp_global['ambos_ko']} | net {comp_global['net']}")
    comp_etiqueta = {}
    for j, et in enumerate(ETIQUETAS):
        d = matriz_errores(y[:, [j]], pred_v03[:, [j]], pred_cnn[:, [j]])
        comp_etiqueta[et] = d
        log(f"      {et}      rescue {d['rescue']:4d} | damage {d['damage']:4d} | net "
            f"{d['net']:+5d} | rescue_rate {d['rescue_rate']:.4f} | damage_rate "
            f"{d['damage_rate']:.4f}")

    feats = pd.read_parquet(FEATURES / "v2_structured_features.parquet")
    feats = feats.set_index("id").reindex(ids_v03)
    tipos = des["text"].astype(str)
    bajo = tipos.str.lower()
    slic = {
        "MENCION": feats["tiene_mencion"].to_numpy() == 1,
        "COLECTIVO": feats["tiene_colectivo"].to_numpy() == 1,
        "AFILIACION_proxy_densidad": feats["col_densidad"].to_numpy() > 0,
        "INTERPERSONAL": feats["tiene_segunda_persona"].to_numpy() == 1,
        "URL": feats["tiene_url"].to_numpy() == 1,
        "HASHTAG": feats["tiene_hashtag"].to_numpy() == 1,
        "CITA": feats["tiene_cita"].to_numpy() == 1,
        "ELONGACION": feats["tiene_elongacion"].to_numpy() == 1,
        "COLOQUIAL_proxy_risas_leet": ((feats["tiene_risas"].to_numpy() == 1)
                                       | (feats["tiene_leet"].to_numpy() == 1)),
        "INSULTOS_lista_declarada": bajo.apply(
            lambda s: any(w in s for w in LISTA_INSULTOS)).to_numpy(),
        "SIN_ODIO_EXPLICITO": y[:, 0] == 0,
    }
    comp_slice = {}
    for nombre, m in slic.items():
        if m.sum() < 20:
            continue
        d = matriz_errores(y, pred_v03, pred_cnn, m)
        comp_slice[nombre] = dict(d, filas=int(m.sum()))
        log(f"      {nombre:28} filas {int(m.sum()):4d} | rescue {d['rescue']:4d} | damage "
            f"{d['damage']:4d} | net {d['net']:+5d}")

    # ---------- PASS 3: diversidad ----------
    log("")
    log("   PASS 3: diversidad de predicciones entre V03 y el CNN")
    div = {"correlacion_proba": {}, "correlacion_margen": {},
           "auc_v03_error_con_score_cnn": {}, "auc_cnn_error_con_score_v03": {}}
    for j, et in enumerate(ETIQUETAS):
        div["correlacion_proba"][et] = round(float(np.corrcoef(p_v03[:, j], p_cnn[:, j])[0, 1]), 6)
        mv, mc = p_v03[:, j] - UMBRALES[j], p_cnn[:, j] - UMBRALES[j]
        div["correlacion_margen"][et] = round(float(np.corrcoef(mv, mc)[0, 1]), 6)
        err_v = (pred_v03[:, j] != y[:, j]).astype(int)
        err_c = (pred_cnn[:, j] != y[:, j]).astype(int)
        div["auc_v03_error_con_score_cnn"][et] = round(float(roc_auc_score(err_v, p_cnn[:, j])), 6)
        div["auc_cnn_error_con_score_v03"][et] = round(float(roc_auc_score(err_c, p_v03[:, j])), 6)
        log(f"      {et}: corr proba {div['correlacion_proba'][et]:+.4f} | corr margen "
            f"{div['correlacion_margen'][et]:+.4f} | AUC(error V03 | score CNN) "
            f"{div['auc_v03_error_con_score_cnn'][et]:.4f}")
    acuerdo = float((pred_v03 == pred_cnn).mean())
    div["acuerdo_unidades"] = round(acuerdo, 6)
    div["desacuerdo_unidades"] = round(1 - acuerdo, 6)
    log(f"      acuerdo {acuerdo:.6f} | desacuerdo {1 - acuerdo:.6f} sobre 4449 unidades")

    # ---------- PASS 4: la fusion minima y pre-registrada ----------
    log("")
    log("   PASS 4: fusion minima, tres alphas pre-registrados, ninguna busqueda")
    fusiones = []
    for a in ALPHAS:
        pf = (1 - a) * p_v03 + a * p_cnn
        predf = (pf >= UMBRALES[None, :]).astype(int)
        cv = f1m(y, predf)
        delta = cv - cv_v03
        pl_f = [round(f1m(y[np.asarray(b)], predf[np.asarray(b)])
                      - f1m(y[np.asarray(b)], pred_v03[np.asarray(b)]), 6) for _, b in pl]
        n_ok = sum(1 for d in pl_f if d > 0)
        reach = np.abs(UMBRALES[None, :] - p_v03) <= a * np.abs(p_cnn - p_v03)
        nreach = int(reach.sum())
        flips = int((predf != pred_v03).sum())
        ce = predf.copy()
        ce[reach & (predf != y)] = y[reach & (predf != y)]
        fl = predf.copy()
        fl[reach & (predf == y)] = 1 - y[reach & (predf == y)]
        techo, piso = f1m(y, ce) - cv_v03, f1m(y, fl) - cv_v03
        media, lo, hi = bootstrap_clusters(y, predf, pred_v03, grupos, SEMILLA_BOOT,
                                           N_BOOTSTRAP)
        comp_f = matriz_errores(y, pred_v03, predf)
        fusiones.append({"alpha": a, "corrected_cv": round(cv, 10),
                         "delta_vs_v03": round(delta, 10), "fold_scores": pl_f,
                         "folds_positivos": n_ok, "n_alcanzables": nreach,
                         "n_volteadas": flips, "tasa_volteo": round(flips / max(nreach, 1), 4),
                         "techo_oraculo": round(techo, 10), "piso_oraculo": round(piso, 10),
                         "bootstrap_media": media, "bootstrap_ic95": [lo, hi],
                         "rescue_vs_v03": comp_f["rescue"], "damage_vs_v03": comp_f["damage"],
                         "net_vs_v03": comp_f["net"]})
        log(f"      alpha {a:.2f}: corrected_cv {cv:.10f} | delta {delta:+.10f} | folds "
            f"{n_ok}/5 | alcanzables {nreach} | volteadas {flips} | techo {techo:+.6f} | "
            f"bootstrap [{lo:+.6f}, {hi:+.6f}]")
        log(f"         fold_scores {pl_f}")

    prim = next(f for f in fusiones if f["alpha"] == ALPHA_PRIMARIO)
    pasa = (prim["delta_vs_v03"] >= EPS_GATE_DELTA and prim["folds_positivos"] >= GATE_FOLDS
            and prim["net_vs_v03"] > 0)
    signos = [f["delta_vs_v03"] > 0 for f in fusiones]
    coherentes = all(signos) or not any(signos)
    log("")
    log(f"   GATE sobre el alpha primario {ALPHA_PRIMARIO}: delta {prim['delta_vs_v03']:+.10f} "
        f">= {EPS_GATE_DELTA} | folds {prim['folds_positivos']}/5 | net {prim['net_vs_v03']:+d} "
        f"-> PASA {pasa}")
    log(f"   coherencia de signo entre los tres alphas: {coherentes} {signos}")

    net = comp_global["net"]
    if not pasa and net > 0 and comp_global["rescue"] > 0:
        caso, etiqueta = "CASO 1", "COMPLEMENTARIEDAD_SIN_TRANSFERENCIA"
    elif pasa and prim["delta_vs_v03"] >= 0.005:
        caso, etiqueta = "CASO 3", "MEJORA_FUERTE"
    elif pasa:
        caso, etiqueta = "CASO 2", "MEJORA_PEQUENA"
    elif net <= 0 and abs(net) <= max(1, 0.05 * comp_global["unidades"]) and comp_global["rescue"] > 0:
        caso, etiqueta = "CASO 4", "TODO_REDUNDANTE"
    elif all(f["delta_vs_v03"] < 0 for f in fusiones):
        caso, etiqueta = "CASO 5", "CNN_PERJUDICA"
    else:
        caso, etiqueta = "SIN_CASO", "NO_CONCLUYENTE"
    finalista = None
    if pasa:
        finalista = {"alpha": ALPHA_PRIMARIO, "congelado_antes_de_la_segunda_particion": True,
                     "delta": prim["delta_vs_v03"], "folds_positivos": prim["folds_positivos"]}
    log("")
    log(f"   VEREDICTO: {caso} {etiqueta} | finalista para segunda particion: {finalista}")

    # ---------- PASS 5: graficas ----------
    fig, ax = plt.subplots(1, 1, figsize=(9, 5))
    etq = ["V03", "CNN sola"] + [f"V03+CNN a={f['alpha']:.2f}" for f in fusiones]
    vals = [cv_v03, cv_cnn] + [f["corrected_cv"] for f in fusiones]
    ax.bar(etq, vals, color=["#1f4e79", "#a6a6a6", "#2e75b6", "#2e75b6", "#2e75b6"])
    ax.axhline(V03_CCV, color="#c00000", ls="--", lw=1)
    ax.set_ylabel("macro-F1 OOF concatenado"); ax.set_ylim(min(vals) - 0.02, max(vals) + 0.02)
    ax.set_title("P75: V03, CNN sola y las tres fusiones"); plt.xticks(rotation=20, ha="right")
    plt.tight_layout(); plt.savefig(FIGURAS / "p75_01_comparacion.png", dpi=110); plt.close()

    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    a_s = [f["alpha"] for f in fusiones]; d_s = [f["delta_vs_v03"] for f in fusiones]
    ax.axhline(0, color="k", lw=0.8); ax.axhline(EPS_GATE_DELTA, color="#c00000", ls=":",
                                                label=f"gate +{EPS_GATE_DELTA}")
    ax.plot(a_s, d_s, "o-", color="#1f4e79")
    ax.set_xlabel("alpha"); ax.set_ylabel("delta vs V03"); ax.legend()
    ax.set_title("P75: delta vs V03 para los tres alphas")
    plt.tight_layout(); plt.savefig(FIGURAS / "p75_02_delta_alpha.png", dpi=110); plt.close()

    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    ax.bar(["rescue", "damage"], [comp_global["rescue"], comp_global["damage"]],
           color=["#2e7d32", "#c62828"])
    ax.set_ylabel("unidades tweet-etiqueta")
    ax.set_title(f"P75: rescue frente a damage (net {comp_global['net']:+d})")
    plt.tight_layout(); plt.savefig(FIGURAS / "p75_03_rescue_damage.png", dpi=110); plt.close()

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for j, et in enumerate(ETIQUETAS):
        ax[j].scatter(p_v03[:, j], p_cnn[:, j], s=6, alpha=0.35, color="#1f4e79")
        ax[j].axhline(UMBRALES[j], color="#c00000", lw=0.8)
        ax[j].axvline(UMBRALES[j], color="#c00000", lw=0.8)
        ax[j].set_title(f"{et}: corr {div['correlacion_proba'][et]:+.3f}")
        ax[j].set_xlabel("V03 proba")
    plt.tight_layout(); plt.savefig(FIGURAS / "p75_04_desacuerdo.png", dpi=110); plt.close()

    _, ax = plt.subplots(1, 1, figsize=(8, 5))
    xs = np.arange(1, 6); w = 0.25
    for i, f in enumerate(fusiones):
        ax.bar(xs + i * w, f["fold_scores"], width=w, label=f"alpha {f['alpha']:.2f}")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(xs + w); ax.set_xticklabels([f"pliegue {k}" for k in range(1, 6)])
    ax.set_ylabel("delta vs V03"); ax.legend()
    ax.set_title("P75: delta por fold y por alpha")
    plt.tight_layout(); plt.savefig(FIGURAS / "p75_05_fold_delta.png", dpi=110); plt.close()

    nombres_s = list(comp_slice)
    _fig, ax = plt.subplots(1, 1, figsize=(11, 5))
    ax.bar(nombres_s, [comp_slice[n]["net"] for n in nombres_s], color="#2e75b6")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("net (rescue - damage)"); ax.set_title("P75: net por slice")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout(); plt.savefig(FIGURAS / "p75_06_slice_net.png", dpi=110); plt.close()
    log("")
    log("   PASS 5: seis figuras en figuras/p75/")

    # ---------- PASS 6: artefactos ----------
    tabla.to_csv(REPORTS / "p75_paired_oof.csv", index=False, encoding="ascii",
                 lineterminator="\n")
    with (REPORTS / "p75_textcnn_complementarity.csv").open("w", encoding="ascii",
                                                            newline="") as fh:
        w = _csv.writer(fh, lineterminator="\n")
        w.writerow(["ambito", "unidades", "rescue", "damage", "ambos_ok", "ambos_ko", "net",
                    "rescue_rate", "damage_rate", "net_rate", "v03_error", "cnn_error", "filas"])
        w.writerow(["GLOBAL", comp_global["unidades"], comp_global["rescue"], comp_global["damage"],
                    comp_global["ambos_ok"], comp_global["ambos_ko"], comp_global["net"],
                    comp_global["rescue_rate"], comp_global["damage_rate"],
                    comp_global["net_rate"], comp_global["v03_error"], comp_global["cnn_error"], ""])
        for et, d in comp_etiqueta.items():
            w.writerow([f"ETIQUETA_{et}"] + [d[k] for k in
                                             ("unidades", "rescue", "damage", "ambos_ok",
                                              "ambos_ko", "net", "rescue_rate", "damage_rate",
                                              "net_rate", "v03_error", "cnn_error")] + [""])
        for n, d in comp_slice.items():
            w.writerow([f"SLICE_{n}", d["unidades"], d["rescue"], d["damage"], d["ambos_ok"],
                        d["ambos_ko"], d["net"], d["rescue_rate"], d["damage_rate"],
                        d["net_rate"], d["v03_error"], d["cnn_error"], d["filas"]])
        w.writerow([])
        w.writerow(["NOTA", ("unidades tweet-etiqueta. NO son comparables con corrected_cv ni con "
                            "ningun macro-F1. net es un recuento de celdas, no una mejora de metrica")])

    with (REPORTS / "p75_fusion_results.csv").open("w", encoding="ascii", newline="") as fh:
        w = _csv.writer(fh, lineterminator="\n")
        w.writerow(["modelo", "alpha", "corrected_cv", "delta_vs_v03", "fold_1", "fold_2",
                    "fold_3", "fold_4", "fold_5", "folds_positivos", "n_alcanzables",
                    "n_volteadas", "techo_oraculo", "piso_oraculo", "bootstrap_bajo",
                    "bootstrap_alto", "net_vs_v03"])
        w.writerow(["V03", "", round(cv_v03, 10), 0.0, "", "", "", "", "", "", "", "", "", "", "",
                    "", ""])
        w.writerow(["CNN_sola", "", round(cv_cnn, 10), round(cv_cnn - cv_v03, 10),
                    *[round(f1m(y[np.asarray(b)], pred_cnn[np.asarray(b)])
                           - f1m(y[np.asarray(b)], pred_v03[np.asarray(b)]), 6) for _, b in pl],
                    "", "", "", "", "", "", ""])
        for f in fusiones:
            w.writerow(["V03_mas_CNN", f["alpha"], f["corrected_cv"], f["delta_vs_v03"],
                        *f["fold_scores"], f["folds_positivos"], f["n_alcanzables"],
                        f["n_volteadas"], f["techo_oraculo"], f["piso_oraculo"],
                        f["bootstrap_ic95"][0], f["bootstrap_ic95"][1], f["net_vs_v03"]])

    salida = {
        "familia": "P75_COMPLEMENTARIEDAD_Y_FUSION", "experiment_id": "P75",
        "fase": "complementariedad de Word2Vec + TextCNN frente a V03, y fusion minima",
        "fecha": inicio.isoformat(timespec="seconds"),
        "naturaleza": "EXPERIMENTO. OOF pareado de dos modelos. 0 entrenamientos de embeddings",
        "pregunta": ("la informacion de Word2Vec + TextCNN corrige errores que V03 no corrige, o "
                     "es una representacion distinta pero redundante?"),
        "alpha_primario": ALPHA_PRIMARIO, "alphas_de_sensibilidad": [0.05, 0.15],
        "gate": {"delta_minimo": EPS_GATE_DELTA, "folds_minimos": GATE_FOLDS,
                 "requiere_net_positivo": True},
        "pass0_integridad": {"v03_reconstruido": round(cv_v03, 10), "v03_esperado": V03_CCV,
                             "coincide": bool(abs(cv_v03 - V03_CCV) < 1e-9),
                             "sha_w2v": sha, "sha_auditado": SHA_W2V,
                             "holdout_apartado": len(ihold), "holdout_usado": 0,
                             "test_usado": 0, "leaderboard_consultado": 0, "submission": 0,
                             "unidades_tabla": len(tabla),
                             "v03_recalculado_desde_la_tabla": round(cv_tabla, 10),
                             "tabla_reproduce_al_campeon": bool(abs(cv_tabla - V03_CCV) < 1e-9)},
        "pass1_oof_cnn": {"cv_cnn_sola": round(cv_cnn, 10), "epocas_por_fold": epocas_por_fold,
                          "semilla": SEMILLA_CNN,
                          "protocolo": ("E* como mediana de tres splits internos por clusters, y "
                                        "reentrenamiento sobre el outer_train completo durante E* "
                                        "epocas fijas, para usar las mismas filas que V03 sin "
                                        "elegir la epoca sobre outer_validation")},
        "pass2_complementariedad": {"global": comp_global, "por_etiqueta": comp_etiqueta,
                                    "por_slice": comp_slice,
                                    "proxies_declarados": {
                                        "AFILIACION_proxy_densidad": "col_densidad > 0",
                                        "COLOQUIAL_proxy_risas_leet": "tiene_risas o tiene_leet",
                                        "INSULTOS_lista_declarada": "lista del autor, el corte menos "
                                                                    "fiable",
                                        "SIN_ODIO_EXPLICITO": "HS = 0, corte basado en etiqueta"}},
        "pass3_diversidad": div,
        "pass4_fusion": fusiones, "gate_pasa": bool(pasa), "signos_coherentes": bool(coherentes),
        "finalista": finalista,
        "veredicto": {"caso": caso, "etiqueta": etiqueta},
        "atribucion": {
            "nivel_1_informacion": "establecido por P74, no se reabre",
            "nivel_2_complementariedad": "descriptivo, medido en P75-B/C",
            "nivel_3_transferencia": "solo si el gate pasa y, para una afirmacion estable, si "
                                     "replica en la segunda particion",
            "prohibido": ["Word2Vec es mejor que V03", ("que el delta sea atribuible al embedding: "
                          "la comparacion es sistema embedding+CNN contra sistema n-gramas+LR")]},
        "deuda_heredada": [("los umbrales 0.44/0.51/0.46 y la C de V03 se eligieron historicamente "
                           "con acceso a la OOF, luego el liston de +0.003 esta sesgado a favor "
                           "del campeon")],
        "precision_del_corpus": ("sin near-duplicados por encima de Jaccard 0.142857, no 'sin "
                                 "ningun solapamiento'"),
        "formulacion": "V03 remains the best verified leaderboard result.",
    }
    (REPORTS / "p75_fusion.json").write_text(json.dumps(salida, ensure_ascii=False, indent=2) + "\n",
                                             encoding="utf-8")

    md = ["# P75: complementariedad de Word2Vec + TextCNN frente a V03\n\n",
          (f"`experiment_id` P75 | inicio {inicio.isoformat(timespec='seconds')} | alpha primario "
          f"{ALPHA_PRIMARIO} | sensibilidad 0.05 y 0.15\n\n"),
          ("## Lo que habilita P75\n\nP74 demostro con 10 replicas pareadas que la TextCNN sobre "
          "Word2Vec supera a la TextCNN sobre embedding aleatorio (10 de 10, media del delta "
          "+0.0242705458, IC95 t [+0.0139795245, +0.0345615672]). Eso establece que **la "
          "representacion contiene informacion de tarea**. P75 no reabre eso: pregunta si esa "
          "informacion **transfiere a V03**.\n\n"),
          ("## OOF pareado\n\nV03 y el TextCNN entrenan los dos sobre el `outer_train` completo y "
          "predicen sobre exactamente las mismas filas. El TextCNN usa `E*` como **mediana de tres "
          "splits internos por clusters** y luego reentrena sobre el `outer_train` completo durante "
          "`E*` epocas fijas: mismo presupuesto de datos que V03 y **ninguna epoca elegida sobre "
          "`outer_validation`**.\n\n"), "```text\n",
          f"V03 desde el OOF canonico        {cv_v03:.10f}   coincide {abs(cv_v03 - V03_CCV) < 1e-9}\n",
          f"V03 recalculado desde la tabla   {cv_tabla:.10f}   coincide {abs(cv_tabla - V03_CCV) < 1e-9}\n",
          f"unidades tweet-etiqueta          {len(tabla)}\n",
          f"CNN sola (diagnostica)           {cv_cnn:.10f}\n", "```\n\n",
          "## Matriz de interseccion y rescue/damage/net\n\n```text\n",
          (f"GLOBAL  unidades {comp_global['unidades']}  rescue {comp_global['rescue']}  damage "
          f"{comp_global['damage']}  net {comp_global['net']:+d}\n"),
          (f"        ambos correctos {comp_global['ambos_ok']}  ambos incorrectos "
          f"{comp_global['ambos_ko']}\n"),
          (f"        rescue_rate {comp_global['rescue_rate']:.4f}  damage_rate "
          f"{comp_global['damage_rate']:.4f}  net_rate {comp_global['net_rate']:.4f}\n"),
          ("```\n\n**Unidades tweet-etiqueta. NO son comparables con `corrected_cv` ni con ningun "
          "macro-F1.** `net` es un recuento de celdas, no una mejora de metrica.\n\n")]
    md.append("| etiqueta | rescue | damage | net |\n|---|---|---|---|\n")
    for et, d in comp_etiqueta.items():
        md.append(f"| {et} | {d['rescue']} | {d['damage']} | {d['net']:+d} |\n")
    md.append("\n## Diversidad\n\n```text\n")
    for et in ETIQUETAS:
        md.append(f"{et}  corr proba {div['correlacion_proba'][et]:+.4f}  corr margen "
                  f"{div['correlacion_margen'][et]:+.4f}  AUC(error V03 | score CNN) "
                  f"{div['auc_v03_error_con_score_cnn'][et]:.4f}\n")
    md.append(f"acuerdo {div['acuerdo_unidades']:.6f}  desacuerdo {div['desacuerdo_unidades']:.6f}\n")
    md.append("```\n\n## Fusion: los tres alphas, completos\n\n```text\n")
    for f in fusiones:
        md.append(f"alpha {f['alpha']:.2f}  corrected_cv {f['corrected_cv']:.10f}  delta "
                  f"{f['delta_vs_v03']:+.10f}  folds {f['folds_positivos']}/5  "
                  f"alcanzables {f['n_alcanzables']}  volteadas {f['n_volteadas']}  "
                  f"techo {f['techo_oraculo']:+.6f}  piso {f['piso_oraculo']:+.6f}\n")
    md.append("```\n\nEl **alpha primario es 0.10** y es el unico que carga la puerta. 0.05 y 0.15 "
              "se reportan completos y no promueven ni rescatan un fallo del primario.\n\n")
    md.append(f"## Veredicto\n\n**{caso}: {etiqueta}**\n\n")
    md.append(f"Finalista para segunda particion: **{finalista}**\n")
    (REPORTS / "p75_textcnn_complementarity.md").write_text("".join(md), encoding="utf-8")

    dec = [f"# Decision de P75\n\n**{caso}: {etiqueta}**\n\n", "```text\n",
           f"V03                            {cv_v03:.10f}\n",
           f"CNN sola (diagnostica)         {cv_cnn:.10f}\n",
           (f"alpha primario {ALPHA_PRIMARIO}  delta {prim['delta_vs_v03']:+.10f}  folds "
           f"{prim['folds_positivos']}/5\n"),
           (f"gate delta >= {EPS_GATE_DELTA} y >= {GATE_FOLDS}/5 folds y net positivo  ->  "
           f"{'PASA' if pasa else 'NO PASA'}\n"),
           (f"rescue {comp_global['rescue']}  damage {comp_global['damage']}  net "
           f"{comp_global['net']:+d}   (unidades tweet-etiqueta)\n"),
           f"finalista                      {finalista}\n", "```\n\n"]
    dec.append("## Los dos niveles que no se confunden\n\n```text\n")
    dec.append("Word2Vec + TextCNN produce informacion complementaria   -> solo si los errores "
               "muestran correccion especifica\n")
    dec.append("Word2Vec + TextCNN mejora V03                           -> solo si la fusion "
               "demuestra transferencia en corrected_cv\n")
    dec.append("```\n\n")
    if etiqueta == "COMPLEMENTARIEDAD_SIN_TRANSFERENCIA":
        dec.append("Hay correcciones especificas del CNN sobre V03, pero **el mecanismo simple de "
                   "combinacion no consigue transferirlas** a `corrected_cv`. No se abre "
                   "automaticamente otro blend: se cierra la fusion simple y se conserva la "
                   "evidencia de complementariedad.\n")
    elif etiqueta == "MEJORA_PEQUENA":
        dec.append("Existe evidencia de transferencia potencial. **No se adopta todavia**: se congela "
                   "un unico finalista (`alpha` 0.10) y se ejecuta la segunda particion.\n")
    elif etiqueta == "MEJORA_FUERTE":
        dec.append("Existe un candidato suficientemente fuerte para una validacion mas profunda. "
                   "**Aun asi**: no hay submission, ni leaderboard, ni adopcion inmediata. Primero "
                   "segunda particion, auditoria y analisis de errores.\n")
    elif etiqueta == "TODO_REDUNDANTE":
        dec.append("La representacion no aporta informacion util para la decision de V03 bajo este "
                   "probe. Se cierra esta via de fusion.\n")
    elif etiqueta == "CNN_PERJUDICA":
        dec.append("Las probabilidades del CNN desplazan decisiones de forma netamente negativa. Se "
                   "cierra la fusion y no se rescata automaticamente: nada de probar alpha mas "
                   "pequeno, alpha por etiqueta ni stacking sin una hipotesis nueva.\n")
    else:
        dec.append("El resultado no encaja en ninguno de los cinco casos pre-registrados. Se reporta "
                   "como no concluyente.\n")
    dec.append("\n## Estado\n\n```text\nV03 = FROZEN\nP72 = CLOSED\nP73 = COMPLEMENTARY\n"
               "P74 = EVIDENCIA REPRODUCIBLE\nP75 = EJECUTADA\nTEXTCNN = WAITING (se cierra aqui "
               "salvo que P75 justifique una fase posterior)\ntest = 0\nholdout = 0\n"
               "leaderboard = 0\nsubmission = 0\n```\n")
    (REPORTS / "p75_decision.md").write_text("".join(dec), encoding="utf-8")

    with (REPORTS / "p75_compliance.md").open("w", encoding="ascii") as fh:
        fh.write("""# P75: compliance

```text
transformers = 0     BERT = 0     RoBERTa = 0     LLM = 0
external classifier = 0     external labels = 0     external embeddings = 0
new external corpus = 0     pseudo-labeling = 0
test labels = 0     test training = 0     test calibration = 0     test feature selection = 0
holdout reuse = 0     leaderboard tuning = 0     submission = 0
```

El Word2Vec se reutiliza **congelado**, sin reentrenar y sin fine-tuning. Su sha1 coincide con el
artefacto auditado. La TextCNN se entrena supervisada solo con etiquetas de `outer_train`, desde
cero, sin pesos preentrenados de terceros.

El corpus externo siguio usandose solo como texto NO etiquetado, cubierto por la autorizacion
archivada en `docs/p73_provenance_autorizacion_corpus.md`. Precision literal de la auditoria: sin
near-duplicados por encima de Jaccard 0.142857, no "sin ningun solapamiento".

Los umbrales 0.44/0.51/0.46 son parte de la definicion de la metrica y se aplican igual a los tres
brazos. No se recalibran.
""")

    docs_out = ["p75_paired_oof.csv", "p75_textcnn_complementarity.csv",
                "p75_textcnn_complementarity.md", "p75_fusion_results.csv", "p75_fusion.json",
                "p75_decision.md", "p75_compliance.md"]
    sucios = [n for n in docs_out if any(b > 127 for b in (REPORTS / n).read_bytes())]
    if sucios:
        raise SystemExit(f"PARADA: documentos con bytes no ASCII: {sucios}")
    log("")
    log("=" * 78)
    log(f"P75 OK | {caso} {etiqueta}")
    log(f"   V03 {cv_v03:.6f} | CNN {cv_cnn:.6f} | alpha {ALPHA_PRIMARIO} delta "
        f"{prim['delta_vs_v03']:+.6f} | gate {'PASA' if pasa else 'NO PASA'}")
    log(f"   rescue {comp_global['rescue']} | damage {comp_global['damage']} | net "
        f"{comp_global['net']:+d} | finalista {finalista}")
    log("=" * 78)


if __name__ == "__main__":
    ast.parse(Path(__file__).read_text(encoding="utf-8"))
    main()
