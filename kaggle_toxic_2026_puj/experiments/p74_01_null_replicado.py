"""P74: replicacion del null aleatorio. Diez replicas pareadas A_i contra B_i.

Pre-registro en docs/p74_preregistro_null_replicado.md, commit c8856cd, anterior a la corrida.

La pregunta, y es una sola
--------------------------
Puede la ventaja de +0.0316 medida en P73 explicarse por una realizacion particularmente debil del
embedding aleatorio, o permanece cuando el null aleatorio se replica?

Lo que hace
-----------
Diez replicas. En cada una, DOS entrenamientos que comparten la semilla de modelo:

    A_i  Word2Vec Twitter 100d CONGELADO    + TextCNN con la semilla de modelo i
    B_i  embedding ALEATORIO 100d CONGELADO + TextCNN con la MISMA semilla de modelo i

Las dos ramas se REENTRENAN en cada replica. Congelar una sola TextCNN A habria roto el
emparejamiento: B_1 compartiria init con A_1 y B_2..B_10 no, y la varianza de A no se propagaria.

Dentro de una replica, la unica diferencia entre las ramas es la matriz de embeddings.

La unidad de inferencia es el delta pareado d_i = CV(A_i) - CV(B_i), y el intervalo primario es el
t de Student sobre las diez replicas. El bootstrap por clusters es secundario y no desplaza al
primario.
"""

from __future__ import annotations

import ast
import csv as _csv
import hashlib
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

from p73_02_textcnn_vs_random import (
    BATCH,
    EPOCAS,
    MAXLEN,
    MODELOS,
    PACIENCIA,
    REPORTS,
    TOL_ES,
    UMBRALES,
    VENTANA_ES,
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
    sha_arr,
    tokenizar,
)

from src.data import read_train
from src.splits import ETIQUETAS, grupos_fusionados, holdout_v2, pliegues_corregidos

# ---------- las diez semillas, fijadas en el pre-registro c8856cd ----------
SEMILLAS_MODELO = (20261111, 20261112, 20261113, 20261114, 20261115,
                   20261116, 20261117, 20261118, 20261119, 20261120)
SEMILLAS_SORTEO = (20261211, 20261212, 20261213, 20261214, 20261215,
                   20261216, 20261217, 20261218, 20261219, 20261220)
SEMILLA_CONTROL_A = 20261110
SEMILLA_CONTROL_A2 = 20261109
SEMILLA_BOOT = 20261102
N_BOOTSTRAP = 2000
EPSILON = 0.010
CONSISTENTE_MIN = 8
SHA_W2V_REGISTRADO_P73 = "c1b774e52e0e1dc131311bddea967e2a72a96e29"
N_FOLDS = 5


def sha_matriz_pad(m: np.ndarray) -> str:
    """sha1 de una matriz, con el mismo metodo que uso P73.

    Args:
        m: matriz, con la fila de padding incluida.

    Returns:
        Los primeros 40 caracteres del sha1.
    """
    return hashlib.sha1(np.ascontiguousarray(m).tobytes()).hexdigest()[:40]


def init_de_semilla(semilla: int, matriz_ref: np.ndarray, prev_global: np.ndarray) -> dict:
    """Snapshot de los parametros entrenables, determinista por semilla.

    Args:
        semilla: semilla de modelo.
        matriz_ref: matriz cualquiera con la forma correcta (no afecta al init entrenable).
        prev_global: prevalencias para el sesgo de salida de referencia.

    Returns:
        state_dict de los parametros entrenables, sin la matriz ni el sesgo de salida.
    """
    torch.manual_seed(semilla)
    m = TextCNN(matriz_ref)
    with torch.no_grad():
        m.fc2.bias.copy_(torch.as_tensor(log_odds(prev_global), dtype=torch.float32))
    return {k: v.detach().clone() for k, v in m.state_dict().items()
            if not k.startswith("emb.") and k != "fc2.bias"}


def entrenar_con_semilla(matriz: np.ndarray, init: dict, semilla: int,
                         ids_tr: np.ndarray, largos_tr: np.ndarray, y_tr: np.ndarray,
                         ids_va: np.ndarray, largos_va: np.ndarray, y_va: np.ndarray,
                         prev_prior: np.ndarray) -> tuple[np.ndarray, dict]:
    """Entrena un pliegue con la semilla dada. Espejo de la funcion de P73.

    Args:
        matriz: matriz de embeddings.
        init: state_dict inicial, comun a las dos ramas dentro de una replica.
        semilla: semilla de modelo de la replica.
        ids_tr: ids de entrenamiento.
        largos_tr: longitudes de entrenamiento.
        y_tr: etiquetas de entrenamiento.
        ids_va: ids de validacion externa.
        largos_va: longitudes de validacion externa.
        y_va: etiquetas de validacion externa.
        prev_prior: prevalencias para el sesgo de salida.

    Returns:
        Probabilidades sobre la validacion externa y la traza.
    """
    t0 = time.time()
    torch.manual_seed(semilla)
    modelo = TextCNN(matriz)
    modelo.load_state_dict(init, strict=False)
    with torch.no_grad():
        modelo.fc2.bias.copy_(torch.as_tensor(log_odds(prev_prior), dtype=torch.float32))
    theta0 = torch.cat([p.detach().reshape(-1) for p in modelo.parameters()
                        if p.requires_grad]).clone()
    opt = torch.optim.AdamW(grupos_parametros(modelo), lr=LR_PICO_LOCAL,
                            betas=(0.9, 0.999), eps=1e-8)
    perdida = nn.BCEWithLogitsLoss()
    g_perm = torch.Generator().manual_seed(semilla + 7)
    xt = torch.as_tensor(ids_tr)
    rt = mascara_real(largos_tr, torch.device("cpu"))
    yt = torch.as_tensor(y_tr, dtype=torch.float32)
    suaves: list[float] = []
    mejor_estado = None
    mejor_suave = float("inf")
    mejor_epoca = 0
    toco_tope = False
    for epoca in range(1, EPOCAS + 1):
        for gp in opt.param_groups:
            gp["lr"] = lr_de(epoca)
        modelo.train()
        perm = torch.randperm(len(ids_tr), generator=g_perm)
        for b0 in range(0, len(ids_tr), BATCH):
            idx = perm[b0:b0 + BATCH]
            opt.zero_grad()
            perdida(modelo(xt[idx], rt[idx]), yt[idx]).backward()
            nn.utils.clip_grad_norm_([p for p in modelo.parameters() if p.requires_grad], 1.0)
            opt.step()
        perdida_va = _perdida_eval(modelo, ids_va, largos_va, y_va)
        suaves.append(float(np.mean([perdida_va] + suaves[-(VENTANA_ES - 1):]))
                      if suaves else perdida_va)
        if suaves[-1] < mejor_suave - TOL_ES:
            mejor_suave = suaves[-1]
            mejor_epoca = epoca
            mejor_estado = {k: v.detach().clone() for k, v in modelo.state_dict().items()}
        if epoca - mejor_epoca >= PACIENCIA:
            break
        if epoca == EPOCAS:
            toco_tope = True
    modelo.load_state_dict(mejor_estado)
    _, proba_va = _perdida_y_proba(modelo, ids_va, largos_va, y_va)
    _, proba_tr = _perdida_y_proba(modelo, ids_tr, largos_tr, y_tr)
    th1 = torch.cat([p.detach().reshape(-1) for p in modelo.parameters() if p.requires_grad])
    registro = {"epocas": len(suaves), "mejor_epoca": mejor_epoca,
                "perdida_val_interna_minima": round(mejor_suave, 6),
                "deriva_parametros_epoca_1": None,
                "toco_tope_epocas": toco_tope,
                "macro_f1_train_final": round(
                    f1m(y_tr, (proba_tr >= UMBRALES[None, :]).astype(int)), 6),
                "perdida_train_final": round(_perdida_eval(modelo, ids_tr, largos_tr, y_tr), 6),
                "segundos": round(time.time() - t0, 1)}
    registro["deriva_parametros_epoca_1"] = round(float(torch.linalg.vector_norm(
        th1 - theta0).item()), 6)
    return proba_va, registro


@torch.no_grad()
def _perdida_y_proba(modelo: nn.Module, ids: np.ndarray, largos: np.ndarray,
                     y: np.ndarray) -> tuple[float, np.ndarray]:
    """Perdida BCE media y probabilidades, en modo eval.

    Args:
        modelo: red.
        ids: ids.
        largos: longitudes.
        y: etiquetas.

    Returns:
        La perdida y las probabilidades.
    """
    modelo.eval()
    logits = modelo(torch.as_tensor(ids), mascara_real(largos, torch.device("cpu")))
    p = torch.sigmoid(logits).numpy()
    perdida = float(nn.BCEWithLogitsLoss()(
        logits, torch.as_tensor(y, dtype=torch.float32)).item())
    return perdida, p


def _perdida_eval(modelo: nn.Module, ids: np.ndarray, largos: np.ndarray,
                  y: np.ndarray) -> float:
    """Perdida BCE media en modo eval.

    Args:
        modelo: red.
        ids: ids.
        largos: longitudes.
        y: etiquetas.

    Returns:
        La perdida.
    """
    return _perdida_y_proba(modelo, ids, largos, y)[0]


LR_PICO_LOCAL = 1e-3


def corre_rama(matriz, init, semilla, ids, largos, y, grupos, pl):
    """Corre una rama completa: los cinco pliegues.

    Args:
        matriz: matriz de embeddings.
        init: state_dict inicial comun.
        semilla: semilla de modelo.
        ids: ids de todas las filas.
        largos: longitudes.
        y: etiquetas.
        grupos: clusters.
        pl: pliegues.

    Returns:
        Probabilidades OOF y las trazas por pliegue.
    """
    pr = np.zeros((len(y), len(ETIQUETAS)))
    regs = []
    for k, (ia, ib) in enumerate(pl, start=1):
        ia, ib = np.asarray(ia), np.asarray(ib)
        m_in, _ = partir_interna(grupos[ia], y[ia], 20261101 + 100 * k)
        proba, reg = entrenar_con_semilla(
            matriz, init, semilla,
            ids[ia][m_in], largos[ia][m_in], y[ia][m_in],
            ids[ib], largos[ib], y[ib],
            y[ia][m_in].mean(axis=0))
        pr[ib] = proba
        reg["pliegue"] = k
        reg["f1_externo"] = round(f1m(y[ib], (proba >= UMBRALES[None, :]).astype(int)), 6)
        regs.append(reg)
    return pr, regs


def verificar_pareo(matriz_a, matriz_b, init, semilla, ids, largos, y, grupos, pl, iva):
    """Verifica que las dos ramas son comparables salvo por la matriz.

    Args:
        matriz_a: matriz entrenada.
        matriz_b: matriz aleatoria.
        init: state_dict inicial.
        semilla: semilla de modelo.
        ids: ids.
        largos: longitudes.
        y: etiquetas.
        grupos: clusters.
        pl: pliegues.
        iva: indice del pliegue a instrumentar.

    Returns:
        Diccionario con los resultados de la verificacion.
    """
    ia = np.asarray(pl[iva][0])
    m_in, _ = partir_interna(grupos[ia], y[ia], 20261101 + 100 * (iva + 1))

    def instrumentar(matriz):
        torch.manual_seed(semilla)
        modelo = TextCNN(matriz)
        modelo.load_state_dict(init, strict=False)
        with torch.no_grad():
            modelo.fc2.bias.copy_(torch.as_tensor(
                log_odds(y[ia][m_in].mean(axis=0)), dtype=torch.float32))
        cap, est = [], []

        def gancho(_m, entrada, salida):
            cap.append((entrada[0].detach().clone(), salida.detach().clone()))

        modelo.drop.register_forward_hook(gancho)
        opt = torch.optim.AdamW(grupos_parametros(modelo), lr=lr_de(1), betas=(0.9, 0.999),
                                eps=1e-8)
        g_perm = torch.Generator().manual_seed(semilla + 7)
        xt = torch.as_tensor(ids[ia][m_in])
        rt = mascara_real(largos[ia][m_in], torch.device("cpu"))
        yt = torch.as_tensor(y[ia][m_in], dtype=torch.float32)
        modelo.train()
        lote = []
        perm = torch.randperm(len(xt), generator=g_perm)
        for b0 in range(0, min(3 * BATCH, len(xt)), BATCH):
            idx = perm[b0:b0 + BATCH]
            lote.append(idx.numpy().tolist())
            if not est:
                est.append(torch.get_rng_state().clone())
            opt.zero_grad()
            nn.BCEWithLogitsLoss()(modelo(xt[idx], rt[idx]), yt[idx]).backward()
            opt.step()
        return lote, cap, est[0]

    la, ca, ea = instrumentar(matriz_a)
    lb, cb, eb = instrumentar(matriz_b)
    acuerdos, total = 0, 0
    for (aa, oa), (ab, ob) in zip(ca, cb, strict=True):
        obs = (aa != 0) & (ab != 0)
        total += int(obs.sum())
        acuerdos += int(((oa != 0) == (ob != 0))[obs].sum())
    return {"lotes_identicos": la == lb, "rng_identico": bool(torch.equal(ea, eb)),
            "mascara_dropout_identica": bool(total > 0 and acuerdos == total),
            "posiciones_observables": total, "acuerdos": acuerdos}


def t_ic(deltas: np.ndarray) -> tuple[float, float, float, float]:
    """Media y IC 95 t de Student de las diferencias pareadas.

    Args:
        deltas: las diferencias por replica.

    Returns:
        Media, extremo inferior, extremo superior y desviacion tipica.
    """
    from scipy import stats
    n = len(deltas)
    media = float(np.mean(deltas))
    sd = float(np.std(deltas, ddof=1))
    se = sd / math.sqrt(n)
    t = float(stats.t.ppf(0.975, n - 1))
    return round(media, 10), round(media - t * se, 10), round(media + t * se, 10), round(sd, 10)


def genera_aleatoria(semilla: int, W: np.ndarray, normas: np.ndarray) -> np.ndarray:
    """Matriz aleatoria con direccion uniforme y norma por token exacta.

    Args:
        semilla: semilla del sorteo.
        W: matriz entrenada, para la forma.
        normas: normas por token de la matriz entrenada.

    Returns:
        La matriz aleatoria, con la fila 0 a cero.
    """
    rng = np.random.default_rng(semilla)
    U = rng.standard_normal(W.shape)
    U /= np.maximum(np.linalg.norm(U, axis=1, keepdims=True), 1e-12)
    m = np.zeros((W.shape[0] + 1, W.shape[1]), dtype=np.float32)
    m[1:] = (normas[:, None] * U).astype(np.float32)
    return m


def main() -> None:
    """Ejecuta P74 completo.

    Returns:
        None.
    """
    inicio = datetime.now(timezone.utc)
    torch.set_num_threads(32)
    log("P74 null aleatorio replicado: diez replicas pareadas A_i contra B_i")
    log("   pre-registro: docs/p74_preregistro_null_replicado.md, commit c8856cd")

    tr = read_train()
    g = grupos_fusionados(tr)
    idev, ihold = holdout_v2(tr, g)
    des = tr.iloc[idev].reset_index(drop=True)
    y = np.load(PROJECT_ROOT / "features" / "oof_v03_final.npz", allow_pickle=True)["y_true"]
    y = y.astype(int)
    ids_canonicos = np.load(PROJECT_ROOT / "features" / "oof_v03_final.npz",
                            allow_pickle=True)["ids"].astype(np.int64)
    if not np.array_equal(des["id"].to_numpy(dtype=np.int64), ids_canonicos):
        raise SystemExit("PARADA: los ids no coinciden con el OOF canonico")
    grupos = g[idev]
    pl = pliegues_corregidos(des, grupos)
    log(f"   {len(y)} filas | {len(np.unique(grupos))} clusters | {N_FOLDS} pliegues | "
        f"holdout apartado {len(ihold)} filas (NO se usa)")

    # ---------- auditoria del Word2Vec congelado ----------
    d100 = np.load(MODELOS / "w2v_twitter_100d.npz", allow_pickle=True)
    vocab = {w: i + 1 for i, w in enumerate(d100["palabras"])}
    W = d100["W"].astype(np.float32)
    normas = np.linalg.norm(W, axis=1)
    matriz_a = np.zeros((len(vocab) + 1, W.shape[1]), dtype=np.float32)
    matriz_a[1:] = W
    sha_a = sha_matriz_pad(matriz_a)
    log("")
    log(f"   auditoria del Word2Vec: sha1 de la matriz con padding {sha_a}")
    log(f"      registrado por P73 {SHA_W2V_REGISTRADO_P73} | coinciden "
        f"{sha_a == SHA_W2V_REGISTRADO_P73}")
    if sha_a != SHA_W2V_REGISTRADO_P73:
        raise SystemExit("STOP: el hash del Word2Vec ha cambiado. No se continua")

    docs = tokenizar(des["text"])
    ids, largos = asignar_ids(docs, vocab)
    n_cub = np.array([sum(1 for t in d if t in vocab) for d in docs])
    n_tok = np.array([len(d) for d in docs])
    log(f"      ids {ids.shape} | longitudes {largos.min()} a {largos.max()} | cobertura "
        f"{100 * n_cub.sum() / n_tok.sum():.2f}% | filas sin cobertura {int((n_cub == 0).sum())}")
    assert ids.shape == (len(y), MAXLEN)

    # ---------- las diez matrices aleatorias ----------
    matrices_b = [genera_aleatoria(s, W, normas) for s in SEMILLAS_SORTEO]
    hashes_b = [sha_matriz_pad(m) for m in matrices_b]
    if len(set(hashes_b)) != len(hashes_b):
        raise SystemExit("PARADA: dos matrices aleatorias coinciden")
    log("")
    log("   auditoria de los embeddings aleatorios")
    for i, (s, m, h) in enumerate(zip(SEMILLAS_SORTEO, matrices_b, hashes_b, strict=True), start=1):
        nm = float(np.linalg.norm(m[1:], axis=1).mean())
        ns = float(np.linalg.norm(m[1:], axis=1).std())
        dn = float(np.abs(np.linalg.norm(m[1:], axis=1) - normas).max())
        log(f"      B{i:2d} seed {s} | dim {m.shape[1]} | sha1 {h[:16]} | norma media {nm:.6f} "
            f"sd {ns:.6f} | desviacion de normas {dn:.2e}")
        assert m.shape == matriz_a.shape and dn < 1e-6
    assert all(h != sha_a for h in hashes_b)

    # ---------- verificacion de pareamiento ----------
    log("")
    log("   VERIFICACION DE PAREAMIENTO, antes de las replicas")
    prev_global = y.mean(axis=0)
    init_ctrl = init_de_semilla(SEMILLAS_MODELO[0], matriz_a, prev_global)
    vp = verificar_pareo(matriz_a, matrices_b[0], init_ctrl, SEMILLAS_MODELO[0], ids, largos, y,
                         grupos, pl, 0)
    cubre = np.zeros(len(y), dtype=int)
    for _, b in pl:
        cubre[np.asarray(b)] += 1
    n_par = sum(p.numel() for p in TextCNN(matriz_a).parameters() if p.requires_grad)
    log(f"      mismas filas por pliegue: cada fila en exactamente uno "
        f"{bool((cubre == 1).all())} | suma {int(cubre.sum())}")
    log(f"      mismo orden de lotes {vp['lotes_identicos']} | mismo estado del RNG "
        f"{vp['rng_identico']} | misma mascara de dropout {vp['mascara_dropout_identica']} "
        f"({vp['acuerdos']} de {vp['posiciones_observables']})")
    log(f"      misma arquitectura: {n_par} parametros entrenables en las dos ramas")
    log(f"      mismos folds, mismo preprocessing, misma cobertura, mismo vocabulario, mismos "
        f"umbrales {UMBRALES.tolist()}, misma funcion de evaluacion")
    log(f"      la unica diferencia permitida es la matriz: A {sha_a[:16]} frente a B "
        f"{hashes_b[0][:16]}")
    if not (vp["lotes_identicos"] and vp["rng_identico"] and vp["mascara_dropout_identica"]):
        raise SystemExit("STOP: la verificacion de pareamiento ha fallado")

    # ---------- control A contra A ----------
    log("")
    log("   CONTROL A CONTRA A, para verificar el instrumento")
    pr_a1, _ = corre_rama(matriz_a, init_ctrl, SEMILLAS_MODELO[0], ids, largos, y, grupos, pl)
    pr_a2, _ = corre_rama(matriz_a, init_ctrl, SEMILLAS_MODELO[0], ids, largos, y, grupos, pl)
    ctrl_det = float(f1m(y, (pr_a1 >= UMBRALES[None, :]).astype(int))
                     - f1m(y, (pr_a2 >= UMBRALES[None, :]).astype(int)))
    log(f"      A contra A con la MISMA semilla: delta {ctrl_det:+.12f} | identicas "
        f"{bool(np.array_equal(pr_a1, pr_a2))}")
    init_alt = init_de_semilla(SEMILLA_CONTROL_A2, matriz_a, prev_global)
    pr_a3, _ = corre_rama(matriz_a, init_alt, SEMILLA_CONTROL_A2, ids, largos, y, grupos, pl)
    cv_a1 = f1m(y, (pr_a1 >= UMBRALES[None, :]).astype(int))
    cv_a3 = f1m(y, (pr_a3 >= UMBRALES[None, :]).astype(int))
    ctrl_init = cv_a1 - cv_a3
    log(f"      A contra A con DISTINTA semilla: A(s{SEMILLAS_MODELO[0]}) {cv_a1:.10f} | "
        f"A(s{SEMILLA_CONTROL_A2}) {cv_a3:.10f} | delta {ctrl_init:+.10f}")
    log("      el segundo mide la varianza de inicializacion; es control del instrumento, NO "
        "evidencia de mejora")
    if not np.array_equal(pr_a1, pr_a2):
        raise SystemExit("STOP: el pipeline no es determinista con la misma semilla")

    # ---------- las diez replicas ----------
    log("")
    log("   LAS DIEZ REPLICAS PAREADAS")
    replicas = []
    t0 = time.time()
    for i, (sm, mat_b, hb) in enumerate(zip(SEMILLAS_MODELO, matrices_b, hashes_b, strict=True),
                                          start=1):
        init = init_de_semilla(sm, matriz_a, prev_global)
        h_init = sha_arr(np.concatenate([v.numpy().reshape(-1) for v in init.values()]))
        ta = time.time()
        pr_a, reg_a = corre_rama(matriz_a, init, sm, ids, largos, y, grupos, pl)
        pr_b, reg_b = corre_rama(mat_b, init, sm, ids, largos, y, grupos, pl)
        pa = (pr_a >= UMBRALES[None, :]).astype(int)
        pb = (pr_b >= UMBRALES[None, :]).astype(int)
        cva, cvb = f1m(y, pa), f1m(y, pb)
        fa = [r["f1_externo"] for r in reg_a]
        fb = [r["f1_externo"] for r in reg_b]
        df = [round(fa[k] - fb[k], 6) for k in range(N_FOLDS)]
        media, lo, hi = bootstrap_clusters(y, pa, pb, grupos, SEMILLA_BOOT, N_BOOTSTRAP)
        replicas.append({
            "replica_id": i, "seed_modelo": sm, "seed_sorteo": SEMILLAS_SORTEO[i - 1],
            "embedding_B_hash": hb, "cnn_initialization_hash": h_init,
            "A_CV": round(cva, 10), "B_CV": round(cvb, 10),
            "delta_A_minus_B": round(cva - cvb, 10),
            "fold_A": fa, "fold_B": fb, "fold_delta": df,
            "folds_positivos": int(sum(1 for d in df if d > 0)),
            "bootstrap_replica": {"media": media, "ic95_bajo": lo, "ic95_alto": hi},
            "segundos": round(time.time() - ta, 1),
            "convergencia": {"A": [{"pliegue": r["pliegue"], "mejor_epoca": r["mejor_epoca"],
                                    "epocas": r["epocas"], "toco_tope": r["toco_tope_epocas"],
                                    "deriva_p1": r["deriva_parametros_epoca_1"],
                                    "perdida_train": r["perdida_train_final"],
                                    "macro_f1_train": r["macro_f1_train_final"]} for r in reg_a],
                             "B": [{"pliegue": r["pliegue"], "mejor_epoca": r["mejor_epoca"],
                                    "epocas": r["epocas"], "toco_tope": r["toco_tope_epocas"],
                                    "deriva_p1": r["deriva_parametros_epoca_1"],
                                    "perdida_train": r["perdida_train_final"],
                                    "macro_f1_train": r["macro_f1_train_final"]} for r in reg_b]}})
        log(f"      replica {i:2d} seed {sm}: A {cva:.6f} | B {cvb:.6f} | delta "
            f"{cva - cvb:+.6f} | folds A {fa} | folds B {fb} | A gana {replicas[-1]['folds_positivos']}/5 "
            f"| {time.time() - ta:.0f}s")

    deltas = np.array([r["delta_A_minus_B"] for r in replicas])
    media, lo, hi, sd = t_ic(deltas)
    ganadas = int((deltas > 0).sum())
    consistentes = int(sum(1 for r in replicas
                           if r["delta_A_minus_B"] > 0 and r["folds_positivos"] >= 3))
    excluye = bool(lo > 0 or hi < 0)
    dentro_eps = bool(abs(media) < EPSILON)
    log("")
    log("   DISTRIBUCION DE LOS DIEZ DELTAS")
    log(f"      deltas {[round(float(x), 6) for x in deltas]}")
    log(f"      media {media:+.10f} | mediana {float(np.median(deltas)):+.10f} | sd {sd:.10f} | "
        f"min {float(deltas.min()):+.6f} | max {float(deltas.max()):+.6f}")
    log(f"      IC 95 t de Student sobre replicas [{lo:+.10f}, {hi:+.10f}] | excluye el cero "
        f"{excluye}")
    log(f"      A gana en {ganadas} de 10 replicas | con el criterio de direccion pre-registrado "
        f"(delta > 0 y >= 3/5 pliegues) {consistentes} de 10")
    log(f"      relacion con la banda de equivalencia {EPSILON}: |media| {abs(media):.6f} -> "
        f"dentro de la banda {dentro_eps}")

    cvs_a = np.array([r["A_CV"] for r in replicas])
    cvs_b = np.array([r["B_CV"] for r in replicas])
    log("")
    log("   VARIABILIDAD DE CADA RAMA")
    log(f"      A media {cvs_a.mean():.10f} sd {cvs_a.std(ddof=1):.10f} rango "
        f"[{cvs_a.min():.6f}, {cvs_a.max():.6f}]")
    log(f"      B media {cvs_b.mean():.10f} sd {cvs_b.std(ddof=1):.10f} rango "
        f"[{cvs_b.min():.6f}, {cvs_b.max():.6f}]")
    mejor_a = int(np.argmax(cvs_a)) + 1
    mejor_b = int(np.argmax(cvs_b)) + 1
    peor_b = int(np.argmin(cvs_b)) + 1
    log(f"      mejor A replica {mejor_a} | mejor B replica {mejor_b} | peor B replica {peor_b}")
    log(f"      sd de A {cvs_a.std(ddof=1):.6f} frente a sd de B {cvs_b.std(ddof=1):.6f} | "
        f"la variabilidad viene {'de las dos' if abs(cvs_a.std(ddof=1) - cvs_b.std(ddof=1)) < 0.5 * max(cvs_a.std(ddof=1), cvs_b.std(ddof=1)) else ('mas de B' if cvs_b.std(ddof=1) > cvs_a.std(ddof=1) else 'mas de A')}")

    if consistentes >= CONSISTENTE_MIN and excluye and not dentro_eps:
        caso, etiqueta = "CASO 1", "EVIDENCIA_REPRODUCIBLE"
    elif ganadas <= 2:
        caso, etiqueta = "CASO 4", "A_NO_SUPERA_EL_NULL"
    elif dentro_eps:
        caso, etiqueta = "CASO 3", "EQUIVALENCIA_PRACTICA"
    elif consistentes >= CONSISTENTE_MIN and not excluye:
        caso, etiqueta = "CASO 2", "SENAL_POSITIVA_PERO_INCIERTA"
    else:
        caso, etiqueta = "SIN_CASO", "NO_CONCLUYENTE"
    log("")
    log(f"   VEREDICTO: {caso} {etiqueta} | A gana {ganadas}/10 | IC95 t [{lo:+.6f}, {hi:+.6f}] | "
        f"media {media:+.6f} | dentro de epsilon {dentro_eps}")
    log(f"   segundos totales {time.time() - t0:.0f}")

    # ---------- artefactos ----------
    salida = {
        "familia": "P74_NULL_REPLICADO", "experiment_id": "P74",
        "fase": "replicacion del null aleatorio: diez replicas pareadas",
        "fecha": inicio.isoformat(timespec="seconds"),
        "preregistro": {"documento": "docs/p74_preregistro_null_replicado.md",
                        "commit": "c8856cd", "anterior_a_la_corrida": True},
        "naturaleza": "EXPERIMENTO. 20 corridas downstream (10 replicas x 2 ramas) mas 3 de control. "
                      "0 entrenamientos de Word2Vec",
        "pregunta": ("puede la ventaja de +0.0316 medida en P73 explicarse por una realizacion "
                     "particularmente debil del embedding aleatorio, o permanece cuando el null "
                     "aleatorio se replica?"),
        "diseno": {"A": "Word2Vec Twitter 100d CONGELADO + TextCNN con la semilla de modelo i",
                   "B": "embedding ALEATORIO 100d CONGELADO + TextCNN con la MISMA semilla i",
                   "las_dos_ramas_se_reentrenan_por_replica": True,
                   "unica_diferencia_dentro_de_una_replica": "la matriz de embeddings",
                   "replicas": 10, "semillas_modelo": list(SEMILLAS_MODELO),
                   "semillas_sorteo": list(SEMILLAS_SORTEO),
                   "protocolo_congelado_de_P73": True,
                   "folds": "los 5 congelados, la particion canonica de P73"},
        "auditoria_word2vec": {"sha1_matriz_con_padding": sha_a,
                               "registrado_por_P73": SHA_W2V_REGISTRADO_P73,
                               "coinciden": sha_a == SHA_W2V_REGISTRADO_P73,
                               "shape": list(matriz_a.shape),
                               "nota": "la comprobacion inicial hasheaba W sin la fila de padding y "
                                       "dio una falsa alarma; el hash de P73 es el de la matriz "
                                       "con padding"},
        "auditoria_datos": {"filas_desarrollo": len(y), "clusters": len(np.unique(grupos)),
                            "holdout_apartado": len(ihold), "holdout_usado": 0,
                            "test_usado": 0, "leaderboard_consultado": 0, "submission": 0,
                            "ids": list(ids.shape), "respuestas": {},
                            "cobertura_por_ciento": round(100 * n_cub.sum() / n_tok.sum(), 4),
                            "filas_sin_cobertura": int((n_cub == 0).sum())},
        "auditoria_embeddings_aleatorios": [
            {"replica": i + 1, "seed": SEMILLAS_SORTEO[i], "sha1": hashes_b[i],
             "dim": int(matrices_b[i].shape[1]),
             "norma_media": round(float(np.linalg.norm(matrices_b[i][1:], axis=1).mean()), 6),
             "norma_sd": round(float(np.linalg.norm(matrices_b[i][1:], axis=1).std()), 6),
             "desviacion_de_normas": round(float(np.abs(
                 np.linalg.norm(matrices_b[i][1:], axis=1) - normas).max()), 12)}
            for i in range(10)],
        "verificacion_de_pareamiento": dict(vp, mismas_filas_por_pliegue=bool((cubre == 1).all()),
                                            parametros_entrenables=n_par,
                                            unica_diferencia="la matriz de embedding congelada"),
        "control_a_contra_a": {
            "misma_semilla_delta": ctrl_det, "salidas_identicas": bool(np.array_equal(pr_a1, pr_a2)),
            "distinta_semilla_delta": round(ctrl_init, 10),
            "A_semilla_1": round(cv_a1, 10), "A_semilla_control": round(cv_a3, 10),
            "nota": "el primero verifica determinismo; el segundo mide varianza de inicializacion. "
                    "Ninguno es evidencia de mejora"},
        "replicas": replicas,
        "analisis": {"deltas": [round(float(x), 10) for x in deltas],
                     "media": media, "mediana": round(float(np.median(deltas)), 10), "sd": sd,
                     "ic95_t": [lo, hi], "grados_de_libertad": 9,
                     "min": round(float(deltas.min()), 10), "max": round(float(deltas.max()), 10),
                     "replicas_positivas": ganadas, "replicas_negativas": int((deltas < 0).sum()),
                     "replicas_con_direccion_consistente": consistentes,
                     "epsilon": EPSILON, "dentro_de_la_banda": dentro_eps,
                     "variabilidad_A": {"media": round(float(cvs_a.mean()), 10),
                                        "sd": round(float(cvs_a.std(ddof=1)), 10),
                                        "min": round(float(cvs_a.min()), 10),
                                        "max": round(float(cvs_a.max()), 10)},
                     "variabilidad_B": {"media": round(float(cvs_b.mean()), 10),
                                        "sd": round(float(cvs_b.std(ddof=1)), 10),
                                        "min": round(float(cvs_b.min()), 10),
                                        "max": round(float(cvs_b.max()), 10)},
                     "replica_mejor_A": mejor_a, "replica_mejor_B": mejor_b,
                     "replica_peor_B": peor_b},
        "bootstrap_secundario": {
            "por_replica": [r["bootstrap_replica"] for r in replicas],
            "cuantos_excluyen_cero": int(sum(1 for r in replicas
                                             if r["bootstrap_replica"]["ic95_bajo"] > 0
                                             or r["bootstrap_replica"]["ic95_alto"] < 0)),
            "nota": "secundario. El intervalo primario es el t de Student sobre replicas. No se "
                    "utiliza para desplazar al primario aunque sea mas estrecho"},
        "referencia_p73": {"delta_A_minus_B": 0.0316259472,
                           "ic95_bootstrap_clusters": [0.0100657802, 0.0534837793],
                           "ic95_t_por_pliegues": [-0.031652, 0.093580],
                           "P73": "COMPLEMENTARY, evidencia direccionalmente consistente y "
                                  "cuantitativamente inestable. NO se modifica"},
        "veredicto": {"caso": caso, "etiqueta": etiqueta, "A_gana": ganadas,
                      "ic95_t": [lo, hi], "media": media, "dentro_de_epsilon": dentro_eps},
        "que_no_se_responde": [
            ("si es mejor que V03: la comparacion sigue siendo descriptiva, porque la TextCNN no "
            "tiene por que ser competitiva con el campeon y la diferencia mezcla representacion y "
            "arquitectura"),
            "si debe hacerse una fusion",
            "si debe hacerse fine-tuning",
            "si debe enviarse una submission"],
        "que_no_se_puede_afirmar": [
            "que la representacion Word2Vec funcione en general",
            ("que la ventaja provenga de relaciones semanticas y no de estructura direccional mas "
            "estadistica de frecuencia: solo se iguala la norma por token"),
            "que el efecto se replique con otra parte del corpus, otra dimension u otra arquitectura",
            ("que la diferencia sea atribuible a la representacion y no a la interaccion con este "
            "lector concreto")],
        "formulacion": "V03 remains the best verified leaderboard result.",
    }
    (REPORTS / "p74_null_replicado.json").write_text(
        json.dumps(salida, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    with (REPORTS / "p74_replicated_random_null.csv").open("w", encoding="ascii",
                                                           newline="") as fh:
        w = _csv.writer(fh, lineterminator="\n")
        w.writerow(["replica_id", "seed", "embedding_B_hash", "cnn_initialization_hash",
                    "A_CV", "B_CV", "delta_A_minus_B", "fold_1_A", "fold_2_A", "fold_3_A",
                    "fold_4_A", "fold_5_A", "fold_1_B", "fold_2_B", "fold_3_B", "fold_4_B",
                    "fold_5_B", "folds_positivos", "bootstrap_bajo", "bootstrap_alto", "segundos"])
        for r in replicas:
            w.writerow([r["replica_id"], r["seed_modelo"], r["embedding_B_hash"],
                        r["cnn_initialization_hash"], r["A_CV"], r["B_CV"],
                        r["delta_A_minus_B"], *r["fold_A"], *r["fold_B"], r["folds_positivos"],
                        r["bootstrap_replica"]["ic95_bajo"], r["bootstrap_replica"]["ic95_alto"],
                        r["segundos"]])

    with (REPORTS / "p74_replication_summary.csv").open("w", encoding="ascii", newline="") as fh:
        w = _csv.writer(fh, lineterminator="\n")
        w.writerow(["estadistico", "valor"])
        for k, v in [("media_A", round(float(cvs_a.mean()), 10)),
                     ("media_B", round(float(cvs_b.mean()), 10)),
                     ("media_delta", media), ("mediana_delta", round(float(np.median(deltas)), 10)),
                     ("sd_delta", sd), ("ic95_t_bajo", lo), ("ic95_t_alto", hi),
                     ("min_delta", round(float(deltas.min()), 10)),
                     ("max_delta", round(float(deltas.max()), 10)),
                     ("replicas_positivas", ganadas), ("replicas_negativas", int((deltas < 0).sum())),
                     ("epsilon", EPSILON), ("dentro_de_epsilon", dentro_eps),
                     ("sd_A", round(float(cvs_a.std(ddof=1)), 10)),
                     ("sd_B", round(float(cvs_b.std(ddof=1)), 10)),
                     ("control_A_contra_A_misma_semilla", ctrl_det),
                     ("control_A_contra_A_distinta_semilla", round(ctrl_init, 10)),
                     ("caso", caso), ("etiqueta", etiqueta)]:
            w.writerow([k, v])

    with (REPORTS / "p74_compliance.md").open("w", encoding="ascii") as fh:
        fh.write("""# P74: auditoria de compliance

```text
transformers = 0            BERT = 0              RoBERTa = 0
LLM = 0                     classifier externo = 0
external labels = 0         external embeddings = 0
new external corpus = 0     pseudo-labeling = 0
test labels = 0             test training = 0
holdout reuse = 0           leaderboard tuning = 0
submission = 0
dependencias: torch, numpy, sklearn, scipy, pandas
```

## El Word2Vec

Entrenado desde cero por el proyecto (skip-gram con negative sampling implementado a mano en
PyTorch). Corpus externo `mmaguero/covid19-spanishtweets-earlylateapril2020`, licencia CC BY-SA 4.0
con atribucion, periodo cerrado de 2020, cero solapamiento exacto, normalizado y por
near-duplicados (Jaccard maximo 0.142857). Autorizacion del profesor archivada en
`docs/p73_provenance_autorizacion_corpus.md` con el texto verbatim.

## Verificaciones antes de ejecutar

```text
sha1 de la matriz con padding     == el registrado por P73     OK
holdout apartado, no usado        248 filas, 0 usadas por P74  OK
el texto de test no entra en      vocabulario, tokenizer, entrenamiento, embedding, calibracion,
                                  seleccion ni evaluacion      OK
las diez matrices aleatorias      distintas entre si por hash   OK
norma por token en la rama B      desviacion maxima < 1e-6     OK
ninguna matriz aleatoria coincide con la entrenada              OK
```

## Nota de transparencia

La primera comprobacion del hash dio **falsa alarma**: hasheaba `W` sin la fila de padding
(77657 x 100) mientras que P73 registro el hash de la matriz **con** padding (77658 x 100). Se
verifico la causa antes de escalarla y el artefacto esta intacto. Se deja escrito para que el
procedimiento quede documentado.
""")

    md = []
    md.append("# P74: null aleatorio replicado\n\n")
    md.append(f"`experiment_id` P74 | pre-registro `c8856cd` | inicio "
              f"{inicio.isoformat(timespec='seconds')}\n\n")
    md.append("## La amenaza que ataca\n\nP73 midio `delta A - B = +0.0316259472` con un intervalo "
              "por clusters que excluia el cero, pero con **una sola tirada** del embedding "
              "aleatorio. La pregunta es si esa ventaja se explica por una realizacion "
              "particularmente debil del null.\n\n")
    md.append("## El diseno\n\nDiez replicas. En cada una, las dos ramas se **reentrenan** con la "
              "misma semilla de modelo:\n\n```text\nA_i  Word2Vec 100d CONGELADO    + TextCNN "
              "semilla i\nB_i  ALEATORIO 100d CONGELADO + TextCNN semilla i\n```\n\n"
              "La unica diferencia dentro de una replica es la matriz de embeddings. "
              "Congelar una sola TextCNN A habria roto el emparejamiento.\n\n")
    md.append("## Verificacion de pareamiento\n\n```text\n")
    md.append(f"mismas filas por pliegue     {bool((cubre == 1).all())}\n")
    md.append(f"mismo orden de lotes         {vp['lotes_identicos']}\n")
    md.append(f"mismo estado del RNG         {vp['rng_identico']}\n")
    md.append(f"misma mascara de dropout     {vp['mascara_dropout_identica']} "
              f"({vp['acuerdos']} de {vp['posiciones_observables']})\n")
    md.append(f"misma arquitectura           {n_par} parametros entrenables\n")
    md.append("mismos folds, preprocessing, cobertura, vocabulario, umbrales y evaluacion\n")
    md.append("```\n\n## Control A contra A\n\n```text\n")
    md.append(f"misma semilla     delta {ctrl_det:+.12f}   identicas "
              f"{bool(np.array_equal(pr_a1, pr_a2))}\n")
    md.append(f"distinta semilla  delta {ctrl_init:+.10f}   (varianza de inicializacion)\n")
    md.append("```\n\nControl del instrumento, **no** evidencia de mejora.\n\n")
    md.append("## Resultado por replica\n\n| # | seed | A_CV | B_CV | delta | folds A | "
              "folds B | A gana |\n|---|---|---|---|---|---|---|---|\n")
    for r in replicas:
        md.append(f"| {r['replica_id']} | {r['seed_modelo']} | {r['A_CV']:.6f} | "
                  f"{r['B_CV']:.6f} | {r['delta_A_minus_B']:+.6f} | {r['folds_positivos']}/5 | "
                  f"{sum(1 for d in r['fold_delta'] if d < 0)}/5 | "
                  f"{'si' if r['delta_A_minus_B'] > 0 else 'no'} |\n")
    md.append("\n## Distribucion de los diez deltas\n\n```text\n")
    md.append(f"media     {media:+.10f}      mediana  {float(np.median(deltas)):+.10f}\n")
    md.append(f"sd        {sd:.10f}      min      {float(deltas.min()):+.10f}      "
              f"max {float(deltas.max()):+.10f}\n")
    md.append(f"IC 95 t de Student sobre replicas  [{lo:+.10f}, {hi:+.10f}]\n")
    md.append(f"A gana en {ganadas} de 10 replicas\n")
    md.append("```\n\n## Variabilidad de cada rama\n\n```text\n")
    md.append(f"A  media {cvs_a.mean():.10f}  sd {cvs_a.std(ddof=1):.10f}  rango "
              f"[{cvs_a.min():.6f}, {cvs_a.max():.6f}]\n")
    md.append(f"B  media {cvs_b.mean():.10f}  sd {cvs_b.std(ddof=1):.10f}  rango "
              f"[{cvs_b.min():.6f}, {cvs_b.max():.6f}]\n")
    md.append("```\n\n## Banda de equivalencia\n\n```text\n")
    md.append(f"epsilon {EPSILON} | |media| {abs(media):.6f} | dentro de la banda {dentro_eps}\n")
    md.append("```\n\nSe reporta por separado: diferencia estimada, IC95, relacion con epsilon y "
              "numero de replicas ganadas. **La banda no convierte un positivo en negativo.**\n\n")
    md.append("## Bootstrap secundario\n\n```text\n")
    md.append(f"replicas cuyo intervalo por clusters excluye el cero: "
              f"{salida['bootstrap_secundario']['cuantos_excluyen_cero']} de 10\n")
    md.append("```\n\nSecundario. El primario es el t sobre replicas y no se desplaza.\n\n")
    md.append(f"## Veredicto\n\n**{caso} {etiqueta}**\n")
    (REPORTS / "p74_replicated_random_null.md").write_text("".join(md), encoding="utf-8")

    dec = [f"# Decision de P74\n\n**{caso}: {etiqueta}**\n\n", "```text\n",
           f"A gana en {ganadas} de 10 replicas\n",
           f"media del delta {media:+.10f}\n",
           f"IC 95 t de Student sobre replicas [{lo:+.10f}, {hi:+.10f}]\n",
           f"epsilon {EPSILON} | dentro de la banda {dentro_eps}\n",
           f"control A contra A, misma semilla {ctrl_det:+.12f}\n", "```\n\n"]
    if etiqueta == "EVIDENCIA_REPRODUCIBLE":
        dec.append("Existe evidencia reproducible de que la representacion Word2Vec entrenada desde "
                   "cero aporta informacion que el lector TextCNN puede utilizar y que **no se "
                   "explica por una unica realizacion del embedding aleatorio**.\n\n"
                   "**No** significa que la TextCNN sea competitiva con V03. **No** significa que "
                   "exista todavia una mejora sobre V03. **No** significa que haya que hacer una "
                   "submission.\n")
    elif etiqueta == "SENAL_POSITIVA_PERO_INCIERTA":
        dec.append("La direccion es consistente, pero la magnitud no queda establecida con "
                   "suficiencia. Se mantiene la linea como COMPLEMENTARY / INCONCLUSIVE y se "
                   "detiene. No se abre automaticamente otra arquitectura.\n")
    elif etiqueta == "EQUIVALENCIA_PRACTICA":
        dec.append("No hay evidencia suficiente de una ventaja material de Word2Vec sobre el null "
                   "aleatorio para este lector. Se cierra la hipotesis de ventaja de geometria "
                   "aprendida bajo este protocolo.\n")
    elif etiqueta == "A_NO_SUPERA_EL_NULL":
        dec.append("La ventaja observada en P73 no se reproduce. Se cierra esta rama de evidencia "
                   "para Word2Vec + TextCNN. No se intenta rescatarla cambiando la arquitectura.\n")
    else:
        dec.append("El resultado no encaja en ninguno de los cuatro casos pre-registrados. Se "
                   "reporta como no concluyente y no se fuerza dentro de ningun caso.\n")
    dec.append("\n## Estado\n\n```text\nV03 = FROZEN\nP72 = CLOSED\nP73 = COMPLEMENTARY "
               "(evidencia direccionalmente consistente y cuantitativamente inestable)\n"
               "Word2Vec Twitter 100d = FROZEN\nP74 = EJECUTADA\nTEXTCNN = WAITING\ntest = 0\n"
               "holdout = 0\nleaderboard = 0\nsubmission = 0\n```\n")
    (REPORTS / "p74_decision.md").write_text("".join(dec), encoding="utf-8")

    docs_out = ["p74_replicated_random_null.csv", "p74_replicated_random_null.md",
                "p74_replication_summary.csv", "p74_decision.md", "p74_compliance.md",
                "p74_null_replicado.json"]
    sucios = [n for n in docs_out if any(b > 127 for b in (REPORTS / n).read_bytes())]
    if sucios:
        raise SystemExit(f"PARADA: documentos con bytes no ASCII: {sucios}")
    log("")
    log("=" * 78)
    log(f"P74 OK | {caso} {etiqueta}")
    log(f"   A gana {ganadas}/10 | media {media:+.6f} | IC95 t [{lo:+.6f}, {hi:+.6f}]")
    log("=" * 78)


if __name__ == "__main__":
    ast.parse(Path(__file__).read_text(encoding="utf-8"))
    main()
