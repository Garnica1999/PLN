#!/usr/bin/env python
"""P104-03 (R7-1, RONDA 3): L1 / Elastic Net por etiqueta con seleccion anidada.

Pre-registro: reports/p104_preregistro.md (seccion 4). Pregunta: la
regularizacion L1/ElasticNet por etiqueta (malla de 16 configuraciones,
seleccion anidada) supera a la L2 de V03?

- Representacion: la union exacta de V03 (word(1,2) + char_wb(3,5)).
- Malla: penalty l1 y elasticnet (l1_ratio {0.2,0.5,0.8}) x C {0.5,1,2,4};
  solver saga, balanced, seed 42, max_iter 3000, tol 1e-3.
- Seleccion ANIDADA por pliegue y etiqueta (CV interna de 3 pliegues;
  criterio F1 de la etiqueta @0.5). Reporta nnz por etiqueta.
Control obligatorio: M2 reproduce V03 exacto en la canonica y la particion 47.
Sin test; sin holdouts; sin leaderboard. Salidas:
  reports/p104_03_l1_elastic.json y .md

Uso:
    /home/carlos/venv/bin/python experiments/p104_03_l1_elastic.py
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from src import data as data_module
from src.evaluation import ETIQUETAS, metricas_multietiqueta
from src.splits import pliegues_corregidos

warnings.filterwarnings("ignore", category=ConvergenceWarning)

OOF_FINAL = RAIZ / "features" / "oof_v03_final.npz"
OOF_BASE = RAIZ / "features" / "v2_oof_base_corrected.npz"
OOF_47 = RAIZ / "features" / "p98_14_oof_b.npz"
PRE_47 = RAIZ / "reports" / "p98_14_preregistro.json"
IDS_PATH = RAIZ / "data" / "processed" / "holdout_v2_ids.csv"
SAL_JSON = RAIZ / "reports" / "p104_03_l1_elastic.json"
SAL_MD = RAIZ / "reports" / "p104_03_l1_elastic.md"

V03_CCV = 0.7772114525165721
UMBRALES_V03 = {"HS": 0.44, "TR": 0.51, "AG": 0.46}
SEMILLA_BOOT = 20260925
LR_PARAMS = {"C": 2.0, "class_weight": "balanced", "max_iter": 1000,
             "solver": "liblinear", "random_state": 42}
LR_MAX_ITER = 3000
LR_TOL = 1e-3
N_SPLITS_INNER = 3
SEMILLA_INNER = 42
ANCLA = (2, 1.0, (1, 2), (3, 5))


def malla_16() -> list[tuple]:
    """Malla pre-registrada de 16 configuraciones (penalty, l1_ratio, C)."""
    configs = []
    for pen, l1r in (("l1", None), ("elasticnet", 0.2),
                     ("elasticnet", 0.5), ("elasticnet", 0.8)):
        for c in (0.5, 1.0, 2.0, 4.0):
            configs.append((pen, l1r, c))
    return configs


CONFIGS = malla_16()


def log(mensaje: str) -> None:
    """Progreso."""
    print(f"[P104-03] {mensaje}", flush=True)


def sha256(ruta: Path) -> str:
    """sha256 completo."""
    x = hashlib.sha256()
    with open(ruta, "rb") as fh:
        for bloque in iter(lambda: fh.read(1 << 20), b""):
            x.update(bloque)
    return x.hexdigest()


def construir_matrices(textos_tr, textos_va, cfg):
    """Ajusta TF-IDF (word+char_wb) dentro del pliegue y transforma validacion."""
    min_df, max_df, word, char = cfg
    mats_tr, mats_va = [], []
    for analyzer, ngram in (("word", word), ("char_wb", char)):
        vectorizador = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram,
                                       min_df=min_df, max_df=max_df,
                                       sublinear_tf=True, lowercase=True)
        mats_tr.append(vectorizador.fit_transform(textos_tr))
        mats_va.append(vectorizador.transform(textos_va))
    return sparse.hstack(mats_tr).tocsr(), sparse.hstack(mats_va).tocsr()


def construir_lr(cfg) -> LogisticRegression:
    """LR con la configuracion (penalty, l1_ratio, C) de la malla."""
    pen, l1r, c = cfg
    return LogisticRegression(penalty=pen, l1_ratio=l1r, C=c,
                              class_weight="balanced", solver="saga",
                              random_state=42, max_iter=LR_MAX_ITER,
                              tol=LR_TOL)


def combinaciones_texto(train_dev: pd.DataFrame) -> np.ndarray:
    """Etiqueta de combinacion HS x TR x AG por fila (3 digitos)."""
    return (train_dev["HS"].astype(str) + train_dev["TR"].astype(str)
            + train_dev["AG"].astype(str)).to_numpy()


def pliegues_internos(train_dev: pd.DataFrame, grupos: np.ndarray,
                      idx_tr: np.ndarray) -> list:
    """CV interna de 3 pliegues dentro del train del pliegue externo."""
    divisor = StratifiedGroupKFold(n_splits=N_SPLITS_INNER, shuffle=True,
                                   random_state=SEMILLA_INNER)
    combinacion = combinaciones_texto(train_dev)
    return list(divisor.split(np.zeros(len(idx_tr)), combinacion[idx_tr],
                              groups=grupos[idx_tr]))


def seleccionar_configs(train_dev: pd.DataFrame, grupos: np.ndarray,
                        idx_tr: np.ndarray) -> dict:
    """Elige configuracion por etiqueta (CV interna; empate: orden de malla)."""
    textos = train_dev["text"].to_numpy()
    etiquetas = {e: train_dev[e].to_numpy() for e in ETIQUETAS}
    internos = pliegues_internos(train_dev, grupos, idx_tr)
    matrices = []
    for i_tr, i_va in internos:
        pos_tr, pos_va = idx_tr[i_tr], idx_tr[i_va]
        m_tr, m_va = construir_matrices(textos[pos_tr], textos[pos_va], ANCLA)
        matrices.append((pos_tr, pos_va, m_tr, m_va))
    tabla = np.zeros((len(CONFIGS), len(ETIQUETAS)))
    for ci, cfg in enumerate(CONFIGS):
        for pos_tr, pos_va, m_tr, m_va in matrices:
            for j, etiqueta in enumerate(ETIQUETAS):
                modelo = construir_lr(cfg)
                modelo.fit(m_tr, etiquetas[etiqueta][pos_tr])
                pred = (modelo.predict_proba(m_va)[:, 1] >= 0.5).astype(int)
                tabla[ci, j] += f1_score(etiquetas[etiqueta][pos_va], pred,
                                         zero_division=0)
    tabla /= len(internos)
    return {e: CONFIGS[int(np.argmax(tabla[:, j]))]
            for j, e in enumerate(ETIQUETAS)}


def oof_nested(train_dev: pd.DataFrame, grupos: np.ndarray,
               pliegues_list: list) -> tuple[np.ndarray, list, list]:
    """OOF con seleccion anidada; devuelve proba, seleccion y no-convergencias."""
    proba = np.zeros((len(train_dev), len(ETIQUETAS)))
    posiciones = {e: j for j, e in enumerate(ETIQUETAS)}
    textos = train_dev["text"].to_numpy()
    etiquetas = {e: train_dev[e].to_numpy() for e in ETIQUETAS}
    seleccion = []
    no_conv = []
    for numero, (idx_tr, idx_va) in enumerate(pliegues_list, start=1):
        elegidas = seleccionar_configs(train_dev, grupos, idx_tr)
        m_tr, m_va = construir_matrices(textos[idx_tr], textos[idx_va], ANCLA)
        registro = {"pliegue": numero, "nnz": {}}
        for etiqueta in ETIQUETAS:
            cfg = elegidas[etiqueta]
            registro[etiqueta] = [cfg[0], cfg[1], cfg[2]]
            modelo = construir_lr(cfg)
            modelo.fit(m_tr, etiquetas[etiqueta][idx_tr])
            proba[idx_va, posiciones[etiqueta]] = (
                modelo.predict_proba(m_va)[:, 1])
            registro["nnz"][etiqueta] = int((modelo.coef_ != 0).sum())
            if int(modelo.n_iter_[0]) >= LR_MAX_ITER:
                no_conv.append(f"pliegue{numero}-{etiqueta}")
        seleccion.append(registro)
        log(f"  pliegue {numero}: HS {elegidas['HS']} nnz "
            f"{registro['nnz']['HS']} | TR {elegidas['TR']} nnz "
            f"{registro['nnz']['TR']} | AG {elegidas['AG']} nnz "
            f"{registro['nnz']['AG']}")
    return proba, seleccion, no_conv


def pred_umbrales(proba: np.ndarray) -> np.ndarray:
    """Aplica los umbrales congelados de V03."""
    return np.column_stack([
        (proba[:, j] >= UMBRALES_V03[etiqueta]).astype(int)
        for j, etiqueta in enumerate(ETIQUETAS)])


def f1_pliegues(y: np.ndarray, pred: np.ndarray, pliegues: np.ndarray) -> list:
    """F1 media por columna en cada pliegue."""
    return [float(metricas_multietiqueta(
        y[pliegues == k], pred[pliegues == k])["f1_media_columnas"])
        for k in sorted(set(pliegues.tolist()))]


def bootstrap_clusters_delta(y, pred_a, pred_b, grupos, repeticiones=2000):
    """IC95 del delta pooled (a-b) por clusters (maquinaria de p98_13)."""
    generador = np.random.default_rng(SEMILLA_BOOT)
    indices = {int(g): np.flatnonzero(grupos == g) for g in np.unique(grupos)}
    claves = np.array(sorted(indices))
    dif = np.empty(repeticiones)
    for r in range(repeticiones):
        muestra = generador.integers(0, len(claves), len(claves))
        idx = np.concatenate([indices[int(claves[i])] for i in muestra])
        dif[r] = (metricas_multietiqueta(y[idx], pred_a[idx])
                  ["f1_media_columnas"]
                  - metricas_multietiqueta(y[idx], pred_b[idx])
                  ["f1_media_columnas"])
    c = np.quantile(dif, [0.025, 0.975])
    return {"ic95": [round(float(c[0]), 6), round(float(c[1]), 6)],
            "pct_mejora": round(float((dif > 0).mean()), 6)}


def deteccion_poder(ic95: list) -> dict:
    """Piso de deteccion implicado por el IC95 del bootstrap."""
    se = (ic95[1] - ic95[0]) / (2 * 1.959964)
    return {"se_bootstrap": round(se, 6),
            "delta_min_detectable_95": round(1.959964 * se, 6)}


def folds_part47(y: np.ndarray, grupos: np.ndarray) -> np.ndarray:
    """Recrea la particion 47 y verifica su hash contra el pre-registro."""
    combinacion = np.array([f"{a}{b}{c}" for a, b, c in y])
    divisor = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=43)
    pliegues = np.full(len(y), -1, dtype=int)
    for numero, (_, val) in enumerate(divisor.split(
            np.arange(len(y)).reshape(-1, 1), combinacion, groups=grupos)):
        pliegues[val] = numero + 1
    sha = hashlib.sha256(pliegues.astype("int64").tobytes()).hexdigest()
    previo = json.loads(PRE_47.read_text(encoding="utf-8"))
    if previo["particion"]["sha256_pliegues"] != sha:
        raise SystemExit("la particion 47 no reproduce el pre-registro")
    return pliegues


def gate_a3(delta_can, folds_pos, ic95, delta_47, celdas) -> dict:
    """Gate A3 del pre-registro P104 (todas las condiciones obligatorias)."""
    condiciones = {
        "delta_can_ge_0003": delta_can >= 0.003,
        "pliegues_ge_4": folds_pos >= 4,
        "ic95_excluye_cero": ic95[0] > 0.0,
        "delta_47_positivo": delta_47 > 0.0,
        "celdas_ge_200": celdas >= 200,
    }
    return {"condiciones": condiciones,
            "veredicto": "GATE_PASSED" if all(condiciones.values())
            else "GATE_FAILED"}


def evaluar_candidato(proba, pred_ref, pliegues_ref, ccv_ref, y, grupos):
    """Metricas completas de un OOF candidato contra la referencia."""
    pred = pred_umbrales(proba)
    ccv = float(metricas_multietiqueta(y, pred)["f1_media_columnas"])
    folds_can = f1_pliegues(y, pred, pliegues_ref)
    folds_ref = f1_pliegues(y, pred_ref, pliegues_ref)
    pos = int(sum(a > b + 1e-12 for a, b in zip(folds_can, folds_ref)))
    celdas = int((pred != pred_ref).sum())
    boot = bootstrap_clusters_delta(y, pred, pred_ref, grupos)
    return {"corrected_cv": round(ccv, 6),
            "delta_vs_v03": round(ccv - ccv_ref, 6),
            "folds": [round(x, 6) for x in folds_can],
            "pliegues_mejorados": pos,
            "celdas_cambiadas": celdas,
            "bootstrap": boot,
            "deteccion": deteccion_poder(boot["ic95"])}


def resumen_estabilidad(seleccion: list) -> dict:
    """Conteo de configuraciones elegidas por etiqueta a lo largo de pliegues."""
    resumen = {}
    for etiqueta in ETIQUETAS:
        conteo: dict = {}
        for registro in seleccion:
            clave = json.dumps(registro[etiqueta], ensure_ascii=True)
            conteo[clave] = conteo.get(clave, 0) + 1
        resumen[etiqueta] = conteo
    return resumen


def resumen_nnz(seleccion: list) -> dict:
    """Estadisticos de nnz por etiqueta a lo largo de pliegues."""
    resumen = {}
    for etiqueta in ETIQUETAS:
        valores = [registro["nnz"][etiqueta] for registro in seleccion]
        resumen[etiqueta] = {"valores": valores,
                             "media": round(float(np.mean(valores)), 1),
                             "min": int(np.min(valores)),
                             "max": int(np.max(valores))}
    return resumen


def _folds_iter(pliegues: np.ndarray):
    """Itera (train, val) desde un vector de pliegues."""
    for k in sorted(set(pliegues.tolist())):
        val = np.flatnonzero(pliegues == k)
        tr = np.flatnonzero(pliegues != k)
        yield tr, val


def main() -> None:
    """Ejecuta P104-03 completo."""
    for ruta in (SAL_JSON, SAL_MD):
        if ruta.exists():
            raise SystemExit(f"Ya existe {ruta}; no se sobrescribe.")
    inicio = time.perf_counter()

    zf = np.load(OOF_FINAL, allow_pickle=True)
    zb = np.load(OOF_BASE, allow_pickle=True)
    pliegues_can = zf["pliegues"].astype(int)
    y = zf["y_true"].astype(int)
    pred_v03 = zf["pred"].astype(int)
    grupos = zb["grupos"].astype(np.int64)
    modelos = [str(m) for m in zb["modelos"]]
    idx_m2 = modelos.index("M2_union_lr_c2")
    proba_m2_hist = zb["proba"][:, idx_m2, :]
    pred_desde_m2 = pred_umbrales(proba_m2_hist)
    if not np.array_equal(pred_desde_m2, pred_v03):
        raise SystemExit("pred de M2 con umbrales V03 != pred congelado")

    train = data_module.read_train()
    particion = pd.read_csv(IDS_PATH)
    ids_dev = particion.loc[particion["particion"] == "dev", "id"].astype(
        int).tolist()
    train_dev = train[train["id"].astype(int).isin(ids_dev)].reset_index(
        drop=True)
    if not np.array_equal(train_dev["id"].astype(int).to_numpy(),
                          zf["ids"].astype(int)):
        raise SystemExit("orden de dev distinto al congelado")
    log(f"desarrollo: {len(train_dev)} filas; malla: {len(CONFIGS)} configs")

    pliegues_frescos = pliegues_corregidos(train_dev, grupos)
    etiquetas_fold = np.empty(len(train_dev), dtype=int)
    for num, (_, va) in enumerate(pliegues_frescos, start=1):
        etiquetas_fold[va] = num
    if not np.array_equal(etiquetas_fold, pliegues_can):
        log("AVISO: los folds frescos no coinciden 1:1 con el congelado")

    textos = train_dev["text"].to_numpy()
    etiquetas = {e: train_dev[e].to_numpy() for e in ETIQUETAS}

    log("control M2 canonico (debe reproducir el OOF congelado)")
    proba_control = np.zeros((len(train_dev), len(ETIQUETAS)))
    for idx_tr, idx_va in pliegues_frescos:
        m_tr, m_va = construir_matrices(textos[idx_tr], textos[idx_va], ANCLA)
        for j, etiqueta in enumerate(ETIQUETAS):
            modelo = LogisticRegression(**LR_PARAMS)
            modelo.fit(m_tr, etiquetas[etiqueta][idx_tr])
            proba_control[idx_va, j] = modelo.predict_proba(m_va)[:, 1]
    diff_control = float(np.abs(proba_control - proba_m2_hist).max())
    ccv_control = float(metricas_multietiqueta(
        y, pred_umbrales(proba_control))["f1_media_columnas"])
    if diff_control > 1e-6 or abs(ccv_control - V03_CCV) > 1e-12:
        raise SystemExit(f"control NO reproduce: diff {diff_control}, "
                         f"ccv {ccv_control}")
    log(f"control OK (diff max {diff_control:.2e}; ccv {ccv_control:.16f})")

    oof47 = np.load(OOF_47, allow_pickle=True)
    pred_v03_47 = oof47["pred_v03"].astype(int)
    proba_v03_47 = oof47["proba_v03"].astype(float)
    pliegues_47 = folds_part47(y, grupos)
    ccv_v03_47 = float(metricas_multietiqueta(y, pred_v03_47)
                       ["f1_media_columnas"])
    log("particion 47 verificada por hash")

    log("control M2 en la particion 47")
    proba_control_47 = np.zeros((len(train_dev), len(ETIQUETAS)))
    for idx_tr, idx_va in _folds_iter(pliegues_47):
        m_tr, m_va = construir_matrices(textos[idx_tr], textos[idx_va], ANCLA)
        for j, etiqueta in enumerate(ETIQUETAS):
            modelo = LogisticRegression(**LR_PARAMS)
            modelo.fit(m_tr, etiquetas[etiqueta][idx_tr])
            proba_control_47[idx_va, j] = modelo.predict_proba(m_va)[:, 1]
    diff_47_control = float(np.abs(proba_control_47 - proba_v03_47).max())
    celdas_control_47 = int((pred_umbrales(proba_control_47)
                             != pred_v03_47).sum())
    if celdas_control_47 != 0:
        raise SystemExit(f"control 47 no reproduce: {celdas_control_47} celdas")
    log(f"control 47 OK (celdas distintas {celdas_control_47}; "
        f"diff proba {diff_47_control:.2e})")

    log("OOF anidado canonico")
    proba_nested, seleccion_can, no_conv_can = oof_nested(
        train_dev, grupos, pliegues_frescos)
    res_can = evaluar_candidato(proba_nested, pred_v03, pliegues_can, V03_CCV,
                                y, grupos)
    log(f"  canonica: ccv {res_can['corrected_cv']:.6f} | delta "
        f"{res_can['delta_vs_v03']:+.6f} | pliegues "
        f"{res_can['pliegues_mejorados']}/5 | celdas "
        f"{res_can['celdas_cambiadas']}")

    log("OOF anidado particion 47")
    proba_nested_47, seleccion_47, no_conv_47 = oof_nested(
        train_dev, grupos, list(_folds_iter(pliegues_47)))
    res_47 = evaluar_candidato(proba_nested_47, pred_v03_47, pliegues_47,
                               ccv_v03_47, y, grupos)
    log(f"  47: ccv {res_47['corrected_cv']:.6f} | delta "
        f"{res_47['delta_vs_v03']:+.6f} | celdas {res_47['celdas_cambiadas']}")

    gate = gate_a3(res_can["delta_vs_v03"], res_can["pliegues_mejorados"],
                   res_can["bootstrap"]["ic95"], res_47["delta_vs_v03"],
                   res_can["celdas_cambiadas"])
    ic = res_can["bootstrap"]["ic95"]
    if gate["veredicto"] == "GATE_PASSED":
        conclusion = ("GATE_PASSED: candidato para el slot 2 (frente a "
                      "v08_meta_v2); V03 no se toca.")
    elif ic[1] < 0.0:
        conclusion = (f"efecto NEGATIVO detectable: IC95 del delta "
                      f"[{ic[0]:+.6f},{ic[1]:+.6f}] excluye cero por abajo; "
                      "la variante perjudica.")
    elif ic[0] <= 0.0:
        conclusion = (f"sin senal detectable: IC95 del delta "
                      f"[{ic[0]:+.6f},{ic[1]:+.6f}] incluye cero; piso de "
                      f"deteccion "
                      f"{res_can['deteccion']['delta_min_detectable_95']:+.6f}.")
    else:
        conclusion = ("delta positivo por encima del piso de deteccion, pero "
                      "gate A3 no superado (otras condiciones).")

    nnz_can = resumen_nnz(seleccion_can)
    salida = {
        "experiment_id": "P104-03",
        "fase": "P104 RONDA 3 (R7-1): L1/ElasticNet por etiqueta",
        "fecha": "2026-10-14",
        "control": {"diff_max_proba": diff_control, "ccv": ccv_control,
                    "reproduce_v03": True,
                    "particion_47": {"celdas_distintas": celdas_control_47,
                                     "diff_max_proba": diff_47_control}},
        "v03_referencia": {"corrected_cv": V03_CCV, "umbrales": UMBRALES_V03},
        "malla": [[c[0], c[1], c[2]] for c in CONFIGS],
        "canonica": res_can,
        "particion_47": res_47,
        "gate": gate,
        "conclusion": conclusion,
        "seleccion_por_pliegue_canonica": seleccion_can,
        "seleccion_por_pliegue_47": seleccion_47,
        "estabilidad_canonica": resumen_estabilidad(seleccion_can),
        "nnz_canonica": nnz_can,
        "no_convergencias_canonica": no_conv_can,
        "no_convergencias_47": no_conv_47,
        "sha_oof_final": sha256(OOF_FINAL)[:16],
        "tiempo_s": round(time.perf_counter() - inicio, 1),
    }
    SAL_JSON.write_text(json.dumps(salida, indent=2, ensure_ascii=True)
                        + "\n", encoding="utf-8")

    md = ["# P104-03 (R7-1) - L1 / Elastic Net por etiqueta (nested)", "",
          (f"Fecha: 2026-10-14. Control reproduce V03 exacto (diff proba "
           f"{diff_control:.2e}; ccv {ccv_control:.16f}); particion 47 "
           f"{celdas_control_47} celdas distintas."),
          (f"Piso de deteccion (pre-declarado): se bootstrap "
           f"{res_can['deteccion']['se_bootstrap']:.6f}; delta minimo "
           f"detectable 95% "
           f"{res_can['deteccion']['delta_min_detectable_95']:+.6f}."), "",
          "| Metrica | Canonica | Particion 47 |",
          "|---|---|---|",
          (f"| corrected_cv | {res_can['corrected_cv']:.6f} | "
           f"{res_47['corrected_cv']:.6f} |"),
          (f"| delta vs V03 | {res_can['delta_vs_v03']:+.6f} | "
           f"{res_47['delta_vs_v03']:+.6f} |"),
          (f"| pliegues mejorados | {res_can['pliegues_mejorados']}/5 | - |"),
          (f"| celdas cambiadas | {res_can['celdas_cambiadas']} | "
           f"{res_47['celdas_cambiadas']} |"),
          (f"| IC95 delta | "
           f"[{ic[0]:+.6f},{ic[1]:+.6f}] | - |"), "",
          "## Configuracion elegida y nnz por pliegue (canonica)", "",
          "| Pliegue | HS (pen,l1,C) | nnz HS | TR | nnz TR | AG | nnz AG |",
          "|---|---|---|---|---|---|---|"]
    for registro in seleccion_can:
        md.append(f"| {registro['pliegue']} | {registro['HS']} | "
                  f"{registro['nnz']['HS']} | {registro['TR']} | "
                  f"{registro['nnz']['TR']} | {registro['AG']} | "
                  f"{registro['nnz']['AG']} |")
    md += ["", "## Estabilidad (conteo de elecciones, canonica)", ""]
    for etiqueta in ETIQUETAS:
        for clave, conteo in sorted(
                salida["estabilidad_canonica"][etiqueta].items(),
                key=lambda par: -par[1]):
            md.append(f"- {etiqueta}: {clave} x{conteo}")
    md += ["", "## Features no nulas (nnz) por etiqueta", ""]
    for etiqueta in ETIQUETAS:
        r = nnz_can[etiqueta]
        md.append(f"- {etiqueta}: media {r['media']} "
                  f"(min {r['min']}, max {r['max']}) de 20441 rasgos")
    md += ["", (f"Gate A3: {gate['veredicto']} | "
                f"{json.dumps(gate['condiciones'], ensure_ascii=True)}"),
           (f"No convergencias (max_iter {LR_MAX_ITER}): canonica "
            f"{len(no_conv_can)}; 47 {len(no_conv_47)}"),
           "", f"Conclusion: {conclusion}", "",
           ("Sin test; sin holdouts; sin leaderboard; umbrales V03 "
            "congelados."), ""]
    SAL_MD.write_text("\n".join(md), encoding="ascii")

    log(f"resultados: {SAL_JSON.name} y {SAL_MD.name} | tiempo "
        f"{salida['tiempo_s']} s")


if __name__ == "__main__":
    main()
