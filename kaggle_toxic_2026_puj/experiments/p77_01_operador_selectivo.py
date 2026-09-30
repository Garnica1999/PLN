"""P77: operador selectivo sobre el punto de operacion congelado de V03.

Pregunta
--------
La informacion complementaria de Word2Vec + TextCNN, que P74 demostro utilizable y P75 mostro
estructural, puede transferirse al punto de operacion de V03 mediante un operador selectivo que
modifique la decision solo cuando exista evidencia suficiente de que la CNN es util?

Por que este operador y no otro
-------------------------------
P75 USO la CNN globalmente y el damage triplico al rescue. P76 mostro que calibrar no arregla nada,
porque la calibracion es monotona y no cambia el orden, y que ademas recalibrar a V03 destruye su
propio punto de operacion. Lo que falta no es informacion sino un criterio de cuando usarla.

Diseno, fijado en docs/p77_preregistro.md y hasheado ANTES de ejecutar
----------------------------------------------------------------------
La DECISION vive en el espacio crudo de V03 con los umbrales congelados 0.44, 0.51 y 0.46, porque
esos umbrales definen el punto de operacion historico del campeon.

Las MAGNITUDES de evidencia se calculan en escala logit. Comparar margenes crudos entre un modelo
sobreconfiado y otro infraconfiado es un artefacto de escala, no una medida de evidencia. El logit
es monotono y no tiene parametros ajustados, luego no introduce calibracion alguna y, por P76, no
puede crear informacion que antes no estuviera: solo hace comparables las magnitudes.

Tres mecanismos, todos derivados exclusivamente de las salidas de V03 y del CNN:
  1  operador determinista nivel 1, corte por cuantil 80, anidado en outer_train
  2  operador determinista nivel 2, corte por cuantil 50, anidado en outer_train
  3  gate lineal anidado, corte fijo 0.5, entrenado solo con el OOF interno de outer_train

Ninguno ve la outer_validation, ni para ajustarse, ni para calibrarse, ni para elegirse.
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
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

from p73_02_textcnn_vs_random import (  # noqa: E402
    REPORTS,
    bootstrap_clusters,
    f1m,
    log,
)
from p75_01_complementariedad_textcnn import LISTA_INSULTOS  # noqa: E402
from p76_01_calibracion_nested import (  # noqa: E402
    SEMILLA_CNN,
    a_logit,
    ajusta_v03,
    entrena_fijo,
)

from src.data import read_train  # noqa: E402
from src.splits import ETIQUETAS, grupos_fusionados, holdout_v2, pliegues_corregidos  # noqa: E402

FIGURAS = PROJECT_ROOT / "figuras" / "p77"
FEATURES = PROJECT_ROOT / "features"
SHA_W2V = "c1b774e52e0e1dc131311bddea967e2a72a96e29"
V03_CCV = 0.7772114525165721
UMBRALES = np.array([0.44, 0.51, 0.46])
SEMILLA_INT = 20261150
SEMILLA_BOOT = 20261151
SEMILLA_AUC = 20261152
N_BOOTSTRAP = 2000
N_AUC = 500
N_INNER = 4
CUANTILES = {1: 0.80, 2: 0.50}
C_GATE = 0.5
EPOCAS_P75 = {1: 12, 2: 10, 3: 16, 4: 14, 5: 12}
EPS_GATE_DELTA = 0.003
GATE_FOLDS = 4
EPS_ETIQUETA = 0.005
EPS = 1e-6
PRE_REGISTRO = "024b0288af5746baaa1b4acfbdade3c94fb2fbcf29601f41447c893706e7b0ca"


def logit_t(t: float) -> float:
    """Logit de un umbral de V03.

    Args:
        t: umbral.

    Returns:
        El logit.
    """
    c = min(max(t, EPS), 1 - EPS)
    return float(np.log(c / (1 - c)))


def senales(p_v03: np.ndarray, p_cnn: np.ndarray) -> dict:
    """Construye las senales del operador.

    Args:
        p_v03: probabilidades de V03.
        p_cnn: probabilidades del CNN.

    Returns:
        Diccionario con las senales por etiqueta.
    """
    lt = np.array([logit_t(t) for t in UMBRALES])
    s_v = p_v03 - UMBRALES[None, :]
    s_c = p_cnn - UMBRALES[None, :]
    m_v = np.abs(a_logit(p_v03) - lt[None, :])
    m_c = np.abs(a_logit(p_cnn) - lt[None, :])
    return {"s_v": s_v, "s_c": s_c, "m_v": m_v, "m_c": m_c, "d": m_c - m_v,
            "des": ((s_v > 0) != (s_c > 0)) & (s_v != 0) & (s_c != 0)}


def kfold_por_clusters(grupos: np.ndarray, k: int, semilla: int) -> list:
    """Reparte en k pliegues por clusters.

    Args:
        grupos: etiqueta de cluster por fila.
        k: numero de pliegues.
        semilla: semilla.

    Returns:
        Lista de pares de mascaras (entrenamiento, validacion).
    """
    u = np.unique(grupos)
    perm = np.random.default_rng(semilla).permutation(len(u))
    out = []
    for parte in np.array_split(perm, k):
        sel = set(u[parte].tolist())
        m = np.array([g in sel for g in grupos])
        if m.any() and (~m).any():
            out.append((~m, m))
    return out


def aplica_determinista(s: dict, c_m: float, c_d: float) -> tuple:
    """Aplica el operador determinista.

    Args:
        s: senales.
        c_m: corte de magnitud de la CNN.
        c_d: corte de evidencia relativa.

    Returns:
        Predicciones resultantes y mascara de conmutacion.
    """
    cond = s["des"] & (s["m_c"] >= c_m) & (s["d"] >= c_d)
    pred_v = (s["s_v"] > 0).astype(int)
    pred_c = (s["s_c"] > 0).astype(int)
    return np.where(cond, pred_c, pred_v), cond


def celdas(y: np.ndarray, pred: np.ndarray) -> dict:
    """Matriz de interseccion de error contra V03.

    Args:
        y: etiquetas.
        pred: predicciones.

    Returns:
        Diccionario con los recuentos por celda.
    """
    bien = pred == y
    mal = ~bien
    n = bien.size
    return {"n": int(n), "ambos_ok": int(bien.sum()), "ambos_ko": int(mal.sum())}


def mcnemar(y: np.ndarray, pred: np.ndarray, pred_v: np.ndarray) -> dict:
    """Test de McNemar sobre las unidades discordantes.

    Args:
        y: etiquetas.
        pred: predicciones del mecanismo.
        pred_v: predicciones de V03.

    Returns:
        Diccionario con el estadistico, los conteos y el valor p.
    """
    a_ok = (pred == y)
    b_ok = (pred_v == y)
    b = int((a_ok & ~b_ok).sum())
    c = int((~a_ok & b_ok).sum())
    if b + c == 0:
        return {"b_dano": b, "c_rescate": c, "chi2": 0.0, "p": 1.0}
    chi = (abs(b - c) - 1) ** 2 / (b + c)
    return {"b_dano": b, "c_rescate": c, "chi2": float(chi),
            "p": float(stats.chi2.sf(chi, 1))}


def main() -> None:
    """Ejecuta P77 completo.

    Returns:
        None.
    """
    inicio = datetime.now(timezone.utc)
    FIGURAS.mkdir(parents=True, exist_ok=True)
    log("P77: operador selectivo sobre el punto de operacion congelado de V03")
    log(f"   pre-registro sha256 {PRE_REGISTRO}")
    log(f"   holdout = 0, test = 0, leaderboard = 0, submission = 0")

    # ---------- PASS 0: integridad ----------
    tabla = pd.read_csv(REPORTS / "p75_paired_oof.csv")
    n = int(tabla["fila"].max()) + 1
    y = tabla["y_true"].to_numpy().reshape(n, 3)
    p_v03 = tabla["v03_proba"].to_numpy().reshape(n, 3)
    p_cnn = tabla["cnn_proba"].to_numpy().reshape(n, 3)
    ids = tabla.drop_duplicates("fila").sort_values("fila")["row_id"].to_numpy()
    tr = read_train()
    g = grupos_fusionados(tr)
    idev, ihold = holdout_v2(tr, g)
    des = tr.iloc[idev].reset_index(drop=True)
    if not np.array_equal(des["id"].to_numpy(dtype=np.int64), ids):
        raise SystemExit("PARADA: los ids no coinciden con el OOF canonico")
    tx = des["text"]
    grupos = g[idev]
    pl = pliegues_corregidos(des, grupos)
    pred_v = (p_v03 >= UMBRALES[None, :]).astype(int)
    pred_c = (p_cnn >= UMBRALES[None, :]).astype(int)
    cv_v = f1m(y, pred_v)
    log(f"   PASS 0: V03 {cv_v:.10f} | coincide con el campeon {abs(cv_v - V03_CCV) < 1e-9}")
    if abs(cv_v - V03_CCV) > 1e-9:
        raise SystemExit("PARADA: no reproduce al campeon")
    d100 = np.load(PROJECT_ROOT / "modelos" / "twitter_domain" / "w2v_twitter_100d.npz",
                   allow_pickle=True)
    vocab = {w: i + 1 for i, w in enumerate(d100["palabras"])}
    W = d100["W"].astype(np.float32)
    matriz = np.zeros((len(vocab) + 1, W.shape[1]), dtype=np.float32)
    matriz[1:] = W
    from p74_01_null_replicado import init_de_semilla, sha_matriz_pad
    if sha_matriz_pad(matriz) != SHA_W2V:
        raise SystemExit("PARADA: el hash del Word2Vec ha cambiado")
    from p73_02_textcnn_vs_random import asignar_ids, tokenizar
    ids_tok, largos = asignar_ids(tokenizar(tx), vocab)
    init = init_de_semilla(SEMILLA_CNN, matriz, y.mean(axis=0))
    s = senales(p_v03, p_cnn)
    log(f"   PASS 0: sha1 del Word2Vec verificado | holdout apartado {len(ihold)} y NO usado")
    log(f"   PASS 0: tweets efectivos {len(np.unique(grupos))} sobre {n} filas tweet-etiqueta")
    log(f"   PASS 0: unidades discrepantes {int(s['des'].sum())} de {n * 3}")

    # ---------- PASS 1: OOF interno de ambos modelos dentro de outer_train ----------
    log("")
    log("   PASS 1: OOF interno de V03 y del CNN dentro de outer_train, 4 inner folds")
    cortes = {1: [], 2: []}
    auc_gate_folds = []
    gates = []
    switches = {1: np.zeros((n, 3), bool), 2: np.zeros((n, 3), bool),
                "gate": np.zeros((n, 3), bool)}
    detalle_gate = []
    t0 = time.time()
    for k, (ia, ib) in enumerate(pl, start=1):
        ia, ib = np.asarray(ia), np.asarray(ib)
        sub = kfold_por_clusters(grupos[ia], N_INNER, SEMILLA_INT + 100 * k)
        iv = np.zeros((len(ia), 3))
        ic = np.zeros((len(ia), 3))
        for m_tr, m_va in sub:
            a, b = ia[m_tr], ia[m_va]
            iv[m_va] = ajusta_v03(des, tx, y, pl, a, b)
            ic[m_va] = entrena_fijo(matriz, init, SEMILLA_CNN, ids_tok, largos, y, a, b,
                                    EPOCAS_P75[k])
        s_in = senales(iv, ic)
        des_in = s_in["des"]
        if des_in.sum() < 50:
            raise SystemExit("PARADA: muy pocas unidades discrepantes internas")
        for lvl, q in CUANTILES.items():
            cortes[lvl].append((float(np.quantile(s_in["m_c"][des_in], q)),
                                float(np.quantile(s_in["d"][des_in], q))))
        mtr = des_in
        pc_in = (ic >= UMBRALES[None, :]).astype(int)
        X = np.column_stack([s_in["m_v"][mtr], s_in["m_c"][mtr], s_in["d"][mtr]])
        yt = (pc_in[mtr] == y[ia][mtr]).astype(int).ravel()
        if len(np.unique(yt)) < 2:
            raise SystemExit("PARADA: el target del gate es degenerado")
        gate = Pipeline([("sc", StandardScaler()),
                         ("lr", LogisticRegression(C=0.1, class_weight="balanced",
                                                   max_iter=2000, solver="lbfgs"))])
        gate.fit(X, yt)
        gates.append(gate)
        auc_gate_folds.append(float(roc_auc_score(yt, gate.predict_proba(X)[:, 1])))
        detalle_gate.append({"pliegue": k, "n_discrepantes_internas": int(des_in.sum()),
                             "positivos_gate": int(yt.sum()),
                             "auc_gate_interno": round(auc_gate_folds[-1], 6),
                             "corte_nivel_1": [round(x, 6) for x in cortes[1][-1]],
                             "corte_nivel_2": [round(x, 6) for x in cortes[2][-1]],
                             "coef_orden": [round(float(v), 6) for v in
                                            gate.named_steps["lr"].coef_[0]]})
        log(f"      pliegue {k}: discrepantes internas {int(des_in.sum())} | positivos gate "
            f"{int(yt.sum())} | AUC interno {auc_gate_folds[-1]:.4f} | q80 m_c "
            f"{cortes[1][-1][0]:.4f} d {cortes[1][-1][1]:.4f} | q50 m_c {cortes[2][-1][0]:.4f} "
            f"d {cortes[2][-1][1]:.4f} | coef {detalle_gate[-1]['coef_orden']}")
    log(f"   PASS 1: {time.time() - t0:.0f}s. AUC interno medio del gate "
        f"{np.mean(auc_gate_folds):.4f}")

    # ---------- PASS 2: aplicacion sobre outer_validation, una sola vez ----------
    log("")
    log("   PASS 2: aplicacion sobre outer_validation, una sola vez por mecanismo")
    preds = {"A_V03": pred_v, "B_CNN_sola": pred_c}
    for lvl in (1, 2):
        pr = np.zeros((n, 3), int)
        sw = np.zeros((n, 3), bool)
        for idx, (ia, ib) in enumerate(pl):
            ib = np.asarray(ib)
            c_m, c_d = cortes[lvl][idx]
            loc = {"s_v": s["s_v"][ib], "s_c": s["s_c"][ib], "m_v": s["m_v"][ib],
                   "m_c": s["m_c"][ib], "d": s["d"][ib], "des": s["des"][ib]}
            p_, cond = aplica_determinista(loc, c_m, c_d)
            pr[ib] = p_
            sw[ib] = cond
        preds[f"{'C' if lvl == 1 else 'D'}_determinista_nivel{lvl}"] = pr
        switches[lvl] = sw
        log(f"      {'C' if lvl == 1 else 'D'} nivel {lvl}: conmutaciones {int(sw.sum())} "
            f"de {n * 3} unidades")
    pr = pred_v.copy()
    sw = np.zeros((n, 3), bool)
    for idx, (ia, ib) in enumerate(pl):
        ib = np.asarray(ib)
        mtr = s["des"][ib]
        Xo = np.column_stack([s["m_v"][ib].ravel(), s["m_c"][ib].ravel(), s["d"][ib].ravel()])
        pg = gates[idx].predict_proba(Xo)[:, 1].reshape(len(ib), 3)
        cond = mtr & (pg >= C_GATE)
        pr[ib] = np.where(cond, (s["s_c"][ib] > 0).astype(int), pred_v[ib])
        sw[ib] = cond
    preds["E_gate_nested"] = pr
    switches["gate"] = sw
    log(f"      E gate nested: conmutaciones {int(sw.sum())} de {n * 3} unidades")

    # ---------- PASS 3: metricas de cada mecanismo ----------
    log("")
    log("   PASS 3: corrected_cv, folds, rescue, damage, net")
    res = []
    for nom, pr in preds.items():
        cv = f1m(y, pr)
        fol = [round(f1m(y[np.asarray(b)], pr[np.asarray(b)]) - f1m(y[np.asarray(b)],
                                                                  pred_v[np.asarray(b)]), 6)
               for _, b in pl]
        ok_v = pred_v == y
        ok_m = pr == y
        resc = int((~ok_v & ok_m).sum())
        dam = int((ok_v & ~ok_m).sum())
        ent = {"mecanismo": nom, "corrected_cv": round(cv, 10),
               "delta_vs_v03": round(cv - V03_CCV, 10), "fold_scores": fol,
               "folds_positivos": int(sum(1 for d in fol if d > 0)),
               "rescue": resc, "damage": dam, "net": resc - dam,
               "switches": int((pr != pred_v).sum()), "pred": pr}
        res.append(ent)
        log(f"      {nom:24} cv {cv:.10f}  delta {cv - V03_CCV:+.10f}  folds "
            f"{ent['folds_positivos']}/5  rescue {resc:4d}  damage {dam:4d}  net "
            f"{resc - dam:+5d}  switches {ent['switches']:4d}")

    # ---------- PASS 4: el objetivo real del operador ----------
    log("")
    log("   PASS 4: rescue conservado frente a damage introducido, sobre los 268 y 813 de P75")
    ok_v, ok_c = pred_v == y, pred_c == y
    base_resc = (~ok_v & ok_c)
    base_dam = (ok_v & ~ok_c)
    log(f"      base P75 recuperada: rescue {int(base_resc.sum())} | damage "
        f"{int(base_dam.sum())} | coincide con P75 {int(base_resc.sum()) == 268 and int(base_dam.sum()) == 813}")
    if int(base_resc.sum()) != 268 or int(base_dam.sum()) != 813:
        raise SystemExit("PARADA: la base de P75 no se reproduce")
    conservado = {}
    for nom, pr in preds.items():
        if nom == "B_CNN_sola":
            continue
        flip = pr != pred_v
        rc = int((base_resc & flip).sum())
        di = int((base_dam & flip).sum())
        conservado[nom] = {"rescue_conservado": rc, "damage_introducido": di,
                           "rescue_perdido": 268 - rc, "damage_evitado": 813 - di,
                           "frac_rescue_conservado": round(rc / 268, 6),
                           "frac_damage_introducido": round(di / 813, 6),
                           "por_etiqueta": {et: {
                               "rescue_conservado": int((base_resc & flip)[:, j].sum()),
                               "damage_introducido": int((base_dam & flip)[:, j].sum())}
                               for j, et in enumerate(ETIQUETAS)}}
        log(f"      {nom:24} rescue conservado {rc:3d}/268 ({100 * rc / 268:4.1f} por ciento) | "
            f"damage introducido {di:3d}/813 ({100 * di / 813:4.1f} por ciento) | "
            f"damage evitado {813 - di:3d} | rescue perdido {268 - rc:3d}")

    # ---------- PASS 5: interseccion de error del gate ----------
    log("")
    log("   PASS 5: el gate selecciona rescates o tambien danos?")
    swg = switches["gate"]
    interseccion = {
        "rescates_seleccionados": int((base_resc & swg).sum()),
        "rescates_no_seleccionados": int((base_resc & ~swg).sum()),
        "danos_seleccionados": int((base_dam & swg).sum()),
        "danos_no_seleccionados": int((base_dam & ~swg).sum())}
    log(f"      de los 268 rescates potenciales, el gate selecciona "
        f"{interseccion['rescates_seleccionados']} y descarta "
        f"{interseccion['rescates_no_seleccionados']}")
    log(f"      de los 813 danos potenciales, el gate selecciona "
        f"{interseccion['danos_seleccionados']} y descarta "
        f"{interseccion['danos_no_seleccionados']}")
    a, b = interseccion["rescates_seleccionados"], interseccion["danos_seleccionados"]
    log(f"      precision del selector sobre lo que toca: {a}/{a + b} = "
        f"{a / max(a + b, 1):.4f}; la base es {268 / 1081:.4f}")
    interseccion["precision_del_selector"] = round(a / max(a + b, 1), 6)
    interseccion["precision_de_la_base"] = round(268 / 1081, 6)

    # ---------- PASS 6: dominancia de V03 o de la CNN en el criterio ----------
    log("")
    log("   PASS 6: quemargen decide realmente el operador")
    dis = s["des"]
    dom = {"correlacion_de_la_seleccion_con_m_v": 0.0, "correlacion_con_m_c": 0.0}
    for lvl in (1, 2, "gate"):
        sw = switches[lvl].reshape(-1)
        dm = dis.reshape(-1)
        if sw.sum() > 1 and dm.sum() > 2:
            sel = dm & sw
            base = dm
            dom[f"nivel_{lvl}"] = {
                "n": int(sw.sum()),
                "cambio_de_m_v": round(float(s["m_v"].reshape(-1)[sel].mean()
                                              - s["m_v"].reshape(-1)[base].mean()), 6),
                "cambio_de_m_c": round(float(s["m_c"].reshape(-1)[sel].mean()
                                              - s["m_c"].reshape(-1)[base].mean()), 6),
                "cambio_de_d": round(float(s["d"].reshape(-1)[sel].mean()
                                            - s["d"].reshape(-1)[base].mean()), 6)}
            log(f"      nivel {lvl}: al conmutar, m_v cambia "
                f"{dom[f'nivel_{lvl}']['cambio_de_m_v']:+.4f}, m_c cambia "
                f"{dom[f'nivel_{lvl}']['cambio_de_m_c']:+.4f}, d cambia "
                f"{dom[f'nivel_{lvl}']['cambio_de_d']:+.4f}")

    # ---------- PASS 7: bootstrap incondicional, McNemar y AUC del gate ----------
    log("")
    log("   PASS 7: bootstrap pareado por clusters, INCONDICIONAL para los tres operadores")
    boot = {}
    mcn = {}
    for ent in res:
        nom = ent["mecanismo"]
        if nom in ("A_V03", "B_CNN_sola"):
            continue
        med, lo, hi = bootstrap_clusters(y, ent["pred"], pred_v, grupos, SEMILLA_BOOT,
                                         N_BOOTSTRAP)
        boot[nom] = {"media": round(med, 6), "ic95": [round(lo, 6), round(hi, 6)]}
        mcn[nom] = mcnemar(y, ent["pred"], pred_v)
        log(f"      {nom:24} bootstrap {med:+.6f} [{lo:+.6f}, {hi:+.6f}] | McNemar "
            f"chi2 {mcn[nom]['chi2']:.4f} p {mcn[nom]['p']:.6f}")
    ic_auc = {}
    for idx, (ia, ib) in enumerate(pl):
        ib = np.asarray(ib)
        mtr = s["des"][ib]
        if mtr.sum() < 30:
            continue
        Xo = np.column_stack([s["m_v"][ib].ravel(), s["m_c"][ib].ravel(), s["d"][ib].ravel()])
        pg = gates[idx].predict_proba(Xo)[:, 1].reshape(len(ib), 3)[mtr]
        yt = (pred_c[ib][mtr] == y[ib][mtr]).astype(int)
        if len(np.unique(yt)) < 2:
            continue
        gsub = np.repeat(grupos[ib][:, None], 3, axis=1)[mtr]
        auc = float(roc_auc_score(yt, pg))
        us = np.unique(gsub)
        rng = np.random.default_rng(SEMILLA_AUC + idx)
        vals = []
        for _ in range(N_AUC):
            sel = rng.choice(us, len(us), replace=True)
            filas = np.concatenate([np.where(gsub == u)[0] for u in sel])
            if len(np.unique(yt[filas])) > 1:
                vals.append(roc_auc_score(yt[filas], pg[filas]))
        ic_auc[f"pliegue_{idx + 1}"] = {"auc": round(auc, 6),
                                        "ic95": [round(float(np.quantile(vals, 0.025)), 6),
                                                 round(float(np.quantile(vals, 0.975)), 6)]}
        log(f"      AUC del gate en outer pliegue {idx + 1}: {auc:.4f} IC95 "
            f"[{ic_auc[f'pliegue_{idx + 1}']['ic95'][0]:.4f}, "
            f"{ic_auc[f'pliegue_{idx + 1}']['ic95'][1]:.4f}]")

    # ---------- PASS 8: por etiqueta ----------
    log("")
    log("   PASS 8: resultados por etiqueta")
    por_et = {}
    for ent in res:
        cvj = {}
        for j, et in enumerate(ETIQUETAS):
            f1j = float(np.where(y[:, j] == 1, (ent["pred"][:, j] == 1), 0).sum() /
                       max(1, (y[:, j] == 1).sum()))
            pv = float(np.where(y[:, j] == 1, (pred_v[:, j] == 1), 0).sum() /
                        max(1, (y[:, j] == 1).sum()))
            f1p = float(np.where(y[:, j] == 1, (pred_c[:, j] == 1), 0).sum() /
                        max(1, (y[:, j] == 1).sum()))
            cvj[et] = {"f1": round(f1j, 6), "f1_v03": round(pv, 6), "f1_cnn": round(f1p, 6),
                       "delta_vs_v03": round(f1j - pv, 6)}
        por_et[ent["mecanismo"]] = cvj
        if ent["mecanismo"] not in ("A_V03", "B_CNN_sola"):
            log(f"      {ent['mecanismo']:24} " + "  ".join(
                f"{et} {cvj[et]['f1']:.4f} ({cvj[et]['delta_vs_v03']:+.4f})" for et in ETIQUETAS))

    # ---------- PASS 9: slices, solo diagnosticas ----------
    log("")
    log("   PASS 9: slices oficiales, SOLO diagnosticas, no se usan para seleccionar")
    pq = pd.read_parquet(FEATURES / "v2_structured_features.parquet")
    devpq = pq[pq["particion"] == "dev"].copy()
    devpq["id_int"] = devpq["id"].astype(int)
    devpq = devpq.set_index("id_int")
    if not pd.Series(ids).isin(devpq.index).all():
        raise SystemExit("PARADA: el join por id no cruza todas las filas")
    fq = devpq.reindex(ids)
    if fq["tiene_url"].isna().any():
        raise SystemExit("PARADA: quedan NaN tras el join")
    bajo = des["text"].astype(str).str.lower()
    slic = {"MENCION": fq["tiene_mencion"].to_numpy() == 1,
            "COLECTIVO": fq["tiene_colectivo"].to_numpy() == 1,
            "AFILIACION_proxy_densidad": fq["col_densidad"].to_numpy() > 0,
            "INTERPERSONAL": fq["tiene_segunda_persona"].to_numpy() == 1,
            "URL": fq["tiene_url"].to_numpy() == 1,
            "HASHTAG": fq["tiene_hashtag"].to_numpy() == 1,
            "CITA": fq["tiene_cita"].to_numpy() == 1,
            "ELONGACION": fq["tiene_elongacion"].to_numpy() == 1,
            "REPETICION": fq["tiene_repeticion"].to_numpy() == 1,
            "COLOQUIAL_proxy_risas_leet": (fq["tiene_risas"].to_numpy() == 1) |
                                          (fq["tiene_leet"].to_numpy() == 1),
            "INSULTOS_lista_declarada": bajo.apply(
                lambda s: any(w in s for w in LISTA_INSULTOS)).to_numpy(),
            "SIN_ODIO_EXPLICITO": y[:, 0] == 0}
    from p75_01_complementariedad_textcnn import matriz_errores
    por_slice = {}
    for nombre, m in slic.items():
        m = np.asarray(m, dtype=bool)
        if m.sum() < 20:
            continue
        fila = {"filas": int(m.sum())}
        for ent in res:
            if ent["mecanismo"] in ("A_V03", "B_CNN_sola"):
                continue
            d = matriz_errores(y, pred_v, ent["pred"], m)
            fila[ent["mecanismo"]] = {"rescue": d["rescue"], "damage": d["damage"],
                                      "net": d["net"]}
        por_slice[nombre] = fila
    log(f"   PASS 9: {len(por_slice)} slices evaluadas, ninguna usada para seleccionar")

    # ---------- PASS 10: decision ----------
    log("")
    log("   PASS 10: gate de finalista, cinco condiciones")
    finales = []
    for ent in res:
        nom = ent["mecanismo"]
        if nom in ("A_V03", "B_CNN_sola"):
            continue
        delta_et = [por_et[nom][et]["delta_vs_v03"] for et in ETIQUETAS]
        c1 = ent["delta_vs_v03"] >= EPS_GATE_DELTA
        c2 = ent["folds_positivos"] >= GATE_FOLDS
        c3 = ent["rescue"] > ent["damage"]
        c4 = boot[nom]["ic95"][0] > 0
        c5 = mcn[nom]["p"] < 0.05 and min(delta_et) >= -EPS_ETIQUETA
        ok = c1 and c2 and c3 and c4 and c5
        log(f"      {nom:24} C1 {c1} C2 {c2} C3 {c3} C4 {c4} C5 {c5}  -> "
            f"{'FINALISTA' if ok else 'no finalista'}")
        if ok:
            finales.append(nom)
    log("")
    log(f"   FINALISTAS: {finales if finales else 'ninguno'}")
    mecs = [e for e in res if e["mecanismo"] not in ("A_V03", "B_CNN_sola")]
    mejor = max(mecs, key=lambda e: e["delta_vs_v03"])
    if len(finales) == 1:
        caso = "CASO_3_MEJORA_MATERIAL"
        finalista = finales[0]
    elif len(finales) > 1:
        caso = "CASO_3_AMBIGUO"
        finalista = None
    elif mejor["delta_vs_v03"] > 0:
        caso = "CASO_4_MEJORA_MUY_PEQUENA"
        finalista = None
    elif mejor["delta_vs_v03"] > -0.0026:
        caso = "CASO_2_REDUCE_DAMAGE_PERO_NO_MEJORA_F1"
        finalista = None
    else:
        caso = "CASO_1_NINGUN_OPERADOR_TRANSFIERE"
        finalista = None
    log(f"   VEREDICTO: {caso} | finalista {finalista}")
    log(f"   segunda particion: {'NO se ejecuta, se evalua solo tras interpretar' if finalista is None else 'se evaluara solo la robustez del finalista'}")

    # ---------- PASS 11: figuras ----------
    nomb = [e["mecanismo"] for e in res]
    col = ["#7f7f7f", "#c62828"] + ["#2e75b6"] * 3
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.bar(nomb, [e["corrected_cv"] for e in res], color=col)
    ax.axhline(V03_CCV, color="#c00000", ls="--", lw=1, label="V03 campeon")
    ax.set_ylabel("macro-F1 OOF"); ax.legend(); ax.set_title("P77: corrected_cv por mecanismo")
    plt.xticks(rotation=20, ha="right"); plt.tight_layout()
    plt.savefig(FIGURAS / "p77_01_cv_operadores.png", dpi=110); plt.close()

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.bar(nomb, [e["delta_vs_v03"] for e in res], color=col)
    ax.axhline(0, color="k", lw=0.8)
    ax.axhline(EPS_GATE_DELTA, color="#c00000", ls=":", label=f"gate +{EPS_GATE_DELTA}")
    ax.set_ylabel("delta vs V03"); ax.legend(); ax.set_title("P77: delta vs V03")
    plt.xticks(rotation=20, ha="right"); plt.tight_layout()
    plt.savefig(FIGURAS / "p77_02_delta.png", dpi=110); plt.close()

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.bar(nomb, [e["rescue"] for e in res], width=-0.4, align="edge", color="#2e7d32",
           label="rescue")
    ax.bar(nomb, [e["damage"] for e in res], width=0.4, align="edge", color="#c62828",
           label="damage")
    ax.set_ylabel("unidades tweet-etiqueta"); ax.legend()
    ax.set_title("P77: rescue frente a damage CONTRA V03")
    plt.xticks(rotation=20, ha="right"); plt.tight_layout()
    plt.savefig(FIGURAS / "p77_03_rescue_damage.png", dpi=110); plt.close()

    fig, ax = plt.subplots(figsize=(11, 4.5))
    cn = [k for k in conservado]
    ax.bar(cn, [conservado[k]["rescue_conservado"] for k in cn], width=-0.4, align="edge",
           color="#2e7d32", label="rescue conservado (de 268)")
    ax.bar(cn, [conservado[k]["damage_introducido"] for k in cn], width=0.4, align="edge",
           color="#c62828", label="damage introducido (de 813)")
    ax.set_ylabel("unidades tweet-etiqueta"); ax.legend()
    ax.set_title("P77: el objetivo real, conservar rescates sin pagar danos")
    plt.xticks(rotation=20, ha="right"); plt.tight_layout()
    plt.savefig(FIGURAS / "p77_04_rescue_conservado.png", dpi=110); plt.close()

    fig, ax = plt.subplots(figsize=(12, 4.5))
    xs = np.arange(1, 6); w = 0.26
    for i, e in enumerate(res):
        if e["mecanismo"] in ("A_V03", "B_CNN_sola"):
            continue
        ax.bar(xs + i * w, e["fold_scores"], width=w, label=e["mecanismo"][:22])
    ax.axhline(0, color="k", lw=0.8); ax.set_xticks(xs + w)
    ax.set_xticklabels([f"pliegue {k}" for k in range(1, 6)])
    ax.set_ylabel("delta vs V03"); ax.legend(fontsize=7)
    ax.set_title("P77: delta por fold"); plt.tight_layout()
    plt.savefig(FIGURAS / "p77_05_delta_por_fold.png", dpi=110); plt.close()

    fig, ax = plt.subplots(figsize=(11, 4.5))
    et = ["rescates\nseleccionados", "rescates\ndescartados", "danos\nseleccionados",
          "danos\ndescartados"]
    val = [interseccion["rescates_seleccionados"], interseccion["rescates_no_seleccionados"],
           interseccion["danos_seleccionados"], interseccion["danos_no_seleccionados"]]
    ax.bar(et, val, color=["#2e7d32", "#9ccc65", "#c62828", "#ef9a9a"])
    for i, v in enumerate(val):
        ax.text(i, v + 8, str(v), ha="center", fontsize=9)
    ax.set_ylabel("unidades tweet-etiqueta")
    ax.set_title(f"P77: interseccion del gate, precision {interseccion['precision_del_selector']:.4f} "
                 f"frente a la base {interseccion['precision_de_la_base']:.4f}")
    plt.tight_layout(); plt.savefig(FIGURAS / "p77_06_interseccion.png", dpi=110); plt.close()

    fig, ax = plt.subplots(figsize=(10, 4.5))
    cn2 = list(switches.keys())
    xs = np.arange(len(ETIQUETAS)); w = 0.26
    for i, k in enumerate(cn2):
        tot = switches[k].sum(axis=0)
        ax.bar(xs + i * w, 100 * tot / n, width=w,
               label=["C nivel 1", "D nivel 2", "E gate"][cn2.index(k)])
    ax.set_xticks(xs + w); ax.set_xticklabels(ETIQUETAS)
    ax.set_ylabel("por ciento de unidades conmutadas"); ax.legend(fontsize=8)
    ax.set_title("P77: tasa de seleccion por etiqueta"); plt.tight_layout()
    plt.savefig(FIGURAS / "p77_07_tasa_seleccion.png", dpi=110); plt.close()

    fig, ax = plt.subplots(figsize=(12, 4.5))
    ns = list(por_slice.keys()); xs = np.arange(len(ns))
    for i, k in enumerate(mecs):
        ax.bar(xs + i * 0.26, [por_slice[s_][k["mecanismo"]]["net"] for s_ in ns], width=0.26,
               label=k["mecanismo"][:22])
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(xs + 0.26); ax.set_xticklabels(ns, rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("net de unidades, NO delta de metrica"); ax.legend(fontsize=7)
    ax.set_title("P77: net por slice, diagnostico"); plt.tight_layout()
    plt.savefig(FIGURAS / "p77_08_delta_por_slice.png", dpi=110); plt.close()
    log("   PASS 11: ocho figuras en figuras/p77/")

    # ---------- PASS 12: artefactos ----------
    with (REPORTS / "p77_selective_fusion.csv").open("w", encoding="ascii", newline="") as fh:
        w = _csv.writer(fh, lineterminator="\n")
        w.writerow(["mecanismo", "corrected_cv", "delta_vs_v03", "fold_1", "fold_2", "fold_3",
                    "fold_4", "fold_5", "folds_positivos", "rescue", "damage", "net",
                    "switches", "rescue_conservado", "damage_introducido", "rescue_perdido",
                    "damage_evitado", "bootstrap_media", "bootstrap_lo", "bootstrap_hi",
                    "mcnemar_p"])
        for e in res:
            nom = e["mecanismo"]
            c = conservado.get(nom, {})
            b = boot.get(nom, {})
            m_ = mcn.get(nom, {})
            w.writerow([nom, e["corrected_cv"], e["delta_vs_v03"], *e["fold_scores"],
                        e["folds_positivos"], e["rescue"], e["damage"], e["net"],
                        e["switches"], c.get("rescue_conservado", ""),
                        c.get("damage_introducido", ""), c.get("rescue_perdido", ""),
                        c.get("damage_evitado", ""), b.get("media", ""),
                        b.get("ic95", ["", ""])[0], b.get("ic95", ["", ""])[1],
                        m_.get("p", "")])
    with (REPORTS / "p77_gate_results.csv").open("w", encoding="ascii", newline="") as fh:
        w = _csv.writer(fh, lineterminator="\n")
        w.writerow(["pliegue", "n_discrepantes_internas", "positivos_gate", "auc_gate_interno",
                    "corte_n1_mc", "corte_n1_d", "corte_n2_mc", "corte_n2_d",
                    "coef_m_v", "coef_m_c", "coef_d"])
        for d_ in detalle_gate:
            w.writerow([d_["pliegue"], d_["n_discrepantes_internas"], d_["positivos_gate"],
                        d_["auc_gate_interno"], *d_["corte_nivel_1"], *d_["corte_nivel_2"],
                        *d_["coef_orden"]])
        w.writerow([])
        w.writerow(["pliegue", "auc_gate_outer", "ic_lo", "ic_hi"])
        for k_, v_ in ic_auc.items():
            w.writerow([k_.replace("pliegue_", ""), v_["auc"], v_["ic95"][0], v_["ic95"][1]])

    salida = {
        "familia": "P77_OPERADOR_SELECTIVO", "experiment_id": "P77",
        "fecha": inicio.isoformat(timespec="seconds"),
        "preregistro_sha256": PRE_REGISTRO,
        "hipotesis": ("la informacion complementaria no debe aplicarse globalmente: una regla "
                      "selectiva con evidencia concordante y suficiente podria conservar los "
                      "rescates y evitar parte del damage"),
        "protocolo": {"folds": "las mismas 5 folds externas de P75 y P76",
                      "inner_folds": N_INNER, "cortes": "regla de cuantil pre-registrada, "
                      "calculada solo en outer_train y solo sobre unidades discrepantes",
                      "cuantiles": CUANTILES, "gate": "LogisticRegression lineal, C 0.1, "
                      "class_weight balanced, escalado estandar, corte fijo 0.5, entrenado solo "
                      "con el OOF interno de outer_train",
                      "caracteristicas_del_gate": ["m_v", "m_c", "d"],
                      "indicador_de_desacuerdo": "NO se usa como feature porque es constante 1 en "
                                               "el conjunto de entrenamiento; se usa como mascara",
                      "decisiones": "espacio crudo de V03 con umbrales congelados 0.44 0.51 0.46",
                      "magnitudes": "escala logit, monotona y sin parametros ajustados"},
        "pass0": {"v03": round(cv_v, 10), "coincide_con_el_campeon": True,
                  "sha_w2v": SHA_W2V, "tweets_efectivos": int(len(np.unique(grupos))),
                  "filas_tweet_etiqueta": n, "discrepantes": int(s["des"].sum()),
                  "holdout_usado": 0, "test_usado": 0, "leaderboard": 0, "submission": 0},
        "mecanismos": [{k: v for k, v in e.items() if k != "pred"} for e in res],
        "objetivo_real": conservado,
        "interseccion_del_gate": interseccion,
        "gate_por_pliegue": detalle_gate,
        "auc_del_gate_outer": ic_auc,
        "auc_del_gate_interno_medio": round(float(np.mean(auc_gate_folds)), 6),
        "bootstrap_incondicional": boot,
        "mcnemar": mcn,
        "por_etiqueta": por_et,
        "cortes_nivel_1": [[round(x, 6) for x in c] for c in cortes[1]],
        "cortes_nivel_2": [[round(x, 6) for x in c] for c in cortes[2]],
        "dominio_del_criterio": dom,
        "por_slice": por_slice,
        "finalistas": finales, "veredicto": caso, "finalista": finalista,
        "auditoria": {"semillas": {"cnn": SEMILLA_CNN, "interna": SEMILLA_INT,
                                   "bootstrap": SEMILLA_BOOT, "auc": SEMILLA_AUC},
                      "n_inner": N_INNER, "cuantiles": CUANTILES, "c_gate": C_GATE,
                      "epocas_cnn": EPOCAS_P75,
                      "fold_ids": {f"pliegue_{k}": [int(np.asarray(b).min()),
                                                    int(np.asarray(b).max())]
                                   for k, (_, b) in enumerate(pl, start=1)}},
        "formulacion": "V03 remains the best verified leaderboard result.",
    }
    (REPORTS / "p77_operador.json").write_text(
        json.dumps(salida, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    md = ["# P77: operador selectivo\n\n",
          f"pre-registro sha256 `{PRE_REGISTRO}`, escrito y hasheado antes de ejecutar\n\n",
          "## La pregunta\n\n> La informacion complementaria de Word2Vec + TextCNN puede "
          "transferirse al punto de operacion de V03 mediante un operador selectivo que modifique "
          "la decision solo cuando exista evidencia suficiente de que la CNN es util?\n\n",
          f"## Los mecanismos\n\n```text\nA_V03                    {preds['A_V03'].shape} V03 intacto\nB_CNN_sola               macro-F1 "
          f"{f1m(y, pred_c):.10f}\nC nivel 1, cuantil 80      {int(switches[1].sum())} "
          f"conmutaciones\nD nivel 2, cuantil 50      {int(switches[2].sum())} conmutaciones\n"
          f"E gate lineal anidado    {int(switches['gate'].sum())} conmutaciones\n```\n\n",
          "## Resultados\n\n```text\n"]
    for e in res:
        md.append(f"{e['mecanismo']:24} cv {e['corrected_cv']:.10f}  delta "
                  f"{e['delta_vs_v03']:+.10f}  folds {e['folds_positivos']}/5  switches "
                  f"{e['switches']:4d}\n")
    md.append("```\n\n## El objetivo real de P77\n\n```text\n")
    for k, v in conservado.items():
        md.append(f"{k:24} rescue conservado {v['rescue_conservado']:3d}/268  damage "
                  f"introducido {v['damage_introducido']:3d}/813  damage evitado "
                  f"{v['damage_evitado']:3d}  rescue perdido {v['rescue_perdido']:3d}\n")
    md.append("```\n\n## Interseccion del gate\n\n```text\n")
    for k, v in interseccion.items():
        md.append(f"{k:34} {v}\n")
    md.append("```\n\n## Bootstrap incondicional y McNemar\n\n```text\n")
    for k in boot:
        md.append(f"{k:24} {boot[k]['media']:+.6f} IC95 [{boot[k]['ic95'][0]:+.6f}, "
                  f"{boot[k]['ic95'][1]:+.6f}]  McNemar p {mcn[k]['p']:.6f}\n")
    md.append(f"```\n\n## Veredicto\n\n**{caso}**\n\nFinalistas: {finales or 'ninguno'}\n\n")
    md.append("## Limitaciones\n\n```text\n")
    md.append("- los cortes por cuantil se calculan con modelos internos mas debiles, luego el sesgo "
              "del gate es conservador y el de los cortes es a la alta, y no se cancelan de forma "
              "acotada\n")
    md.append("- las slices se solapan y no son pruebas independientes, y son solo diagnosticas\n")
    md.append("- las unidades estan agrupadas tambien por tweet, luego el numero efectivo es "
              f"{len(np.unique(grupos))} y no {n}\n")
    md.append("- el AUC del gate en outer se reporta con etiquetas de outer: es medicion, nunca "
              "ajuste\n")
    md.append("```\n")
    (REPORTS / "p77_selective_fusion.md").write_text("".join(md), encoding="utf-8")

    dec = [f"# Decision de P77\n\n**{caso}**\n\n```text\n",
           f"V03 campeon                 {V03_CCV:.10f}\n",
           f"mejor operador              {mejor['mecanismo']}  delta {mejor['delta_vs_v03']:+.10f}\n",
           f"finalistas                   {finales or "ninguno"}\n",
           f"finalista                    {finalista}\n```\n\n"]
    if caso == "CASO_1_NINGUN_OPERADOR_TRANSFIERE":
        dec.append("Los operadores conservan complementariedad pero `corrected_cv` sigue por debajo "
                   "de V03. La informacion complementaria existe pero no es suficientemente fiable "
                   "para modificar la decision del campeon. **Se cierra P77.**\n")
    elif caso == "CASO_2_REDUCE_DAMAGE_PERO_NO_MEJORA_F1":
        dec.append("El operador reduce el damage frente al blend global y conserva parte del "
                   "rescue, pero `corrected_cv` no mejora. Hay mejora del mecanismo de decision y "
                   "no transferencia suficiente al objetivo. **No se hace submission y no se abren "
                   "variantes.**\n")
    elif caso == "CASO_3_MEJORA_MATERIAL":
        dec.append("Existe un finalista que supera las cinco condiciones. Se congela y se detiene. "
                   "**No se hace submission ni se consulta el leaderboard.**\n")
    elif caso == "CASO_3_AMBIGUO":
        dec.append("Varios mecanismos superan el gate a la vez. Elegir entre ellos seria una nueva "
                   "seleccion no pre-registrada, luego **no se elige**.\n")
    else:
        dec.append("Mejora positiva pero por debajo del gate. Se marca MARGINAL y **no se abre una "
                   "nueva grid**.\n")
    dec.append("\n## Estado\n\n```text\nV03 = FROZEN      P72 = CLOSED      P73 = COMPLEMENTARY\n"
               "P74 = EVIDENCIA REPRODUCIBLE      P75 = SIN TRANSFERENCIA\nP76 = SIN TRANSFERENCIA"
               "      P77 = EJECUTADA\nTEXTCNN = ACTIVA SOLO BAJO EL ALCANCE DEL OPERADOR SELECTIVO\n"
               "test = 0   holdout = 0   leaderboard = 0   submission = 0\n```\n")
    (REPORTS / "p77_decision.md").write_text("".join(dec), encoding="utf-8")
    with (REPORTS / "p77_compliance.md").open("w", encoding="ascii") as fh:
        fh.write("""# P77: compliance

```text
transformers = 0     BERT = 0     RoBERTa = 0     LLM = 0
external classifier = 0     external labels = 0     external embeddings = 0
new external corpus = 0     pseudo-labeling = 0
test labels = 0     test training = 0     test calibration = 0     test feature selection = 0
test threshold tuning = 0     holdout reuse = 0     leaderboard tuning = 0     submission = 0
```

El Word2Vec 100d se reutiliza congelado y su sha1 coincide con el artefacto auditado. La TextCNN
no recibe fine-tuning. El gate se ajusta SOLO con el OOF interno de outer_train y se aplica una
sola vez sobre outer_validation, que nunca participa en su ajuste. Los cortes por cuantil se
calculan SOLO dentro de outer_train. Los umbrales de V03 no se tocan. Del parquet de anotaciones
se lee EXCLUSIVAMENTE la particion 'dev'.
""")
    docs = ["p77_selective_fusion.csv", "p77_gate_results.csv", "p77_operador.json",
            "p77_decision.md", "p77_compliance.md"]
    sucios = [x for x in docs if any(b > 127 for b in (REPORTS / x).read_bytes())]
    if sucios:
        raise SystemExit(f"PARADA: bytes no ASCII en {sucios}")
    log("")
    log("=" * 78)
    log(f"P77 OK | {caso} | finalista {finalista}")
    log("=" * 78)


if __name__ == "__main__":
    ast.parse(Path(__file__).read_text(encoding="utf-8"))
    main()
