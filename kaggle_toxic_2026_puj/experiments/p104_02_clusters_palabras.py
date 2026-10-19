#!/usr/bin/env python
"""P104-02 (R5, RONDA 3): clusters de palabras como bolsa adicional.

Pre-registro: reports/p104_preregistro.md (seccion 3). Pregunta: una bolsa de
clusters de palabra (k-means sobre los embeddings propios) anade informacion a
la union de V03?

- Embedding: modelos/twitter_domain/w2v_twitter_lrfix.npz (P72-00; 77.657
  palabras, 200d; corpus externo Twitter/X sin etiquetas, autorizado P71-02).
- k-means (MiniBatchKMeans, seed 42) sobre las palabras del artefacto (solo
  vocabulario externo; sin etiquetas, sin folds, sin texto de dev/test: no
  hay fuga; el mapa token->cluster es una funcion fija del artefacto).
- Variantes: k en {100, 300, 1000}; bloque = TF-IDF word(1,1) del texto de
  clusters; hstack a la union exacta de V03.
Control obligatorio: M2 reproduce V03 exacto en la canonica y la particion 47.
Sin test; sin holdouts; sin leaderboard. Salidas:
  reports/p104_02_clusters_palabras.json y .md

Uso:
    /home/carlos/venv/bin/python experiments/p104_02_clusters_palabras.py
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold

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
EMB_PATH = RAIZ / "modelos" / "twitter_domain" / "w2v_twitter_lrfix.npz"
SAL_JSON = RAIZ / "reports" / "p104_02_clusters_palabras.json"
SAL_MD = RAIZ / "reports" / "p104_02_clusters_palabras.md"

V03_CCV = 0.7772114525165721
UMBRALES_V03 = {"HS": 0.44, "TR": 0.51, "AG": 0.46}
SEMILLA_BOOT = 20260925
LR_PARAMS = {"C": 2.0, "class_weight": "balanced", "max_iter": 1000,
             "solver": "liblinear", "random_state": 42}
PUNTUACION = ".,;:!?¡¿\"'()[]{}<>-…*@#"
VALORES_K = (100, 300, 1000)


def log(mensaje: str) -> None:
    """Progreso."""
    print(f"[P104-02] {mensaje}", flush=True)


def sha256(ruta: Path) -> str:
    """sha256 completo."""
    x = hashlib.sha256()
    with open(ruta, "rb") as fh:
        for bloque in iter(lambda: fh.read(1 << 20), b""):
            x.update(bloque)
    return x.hexdigest()


def construir_mapa(palabras, etiquetas_cluster) -> dict:
    """Mapa palabra -> cluster a partir del artefacto congelado."""
    return {str(p): int(c) for p, c in zip(palabras, etiquetas_cluster)}


def texto_de_clusters(texto: str, mapa: dict) -> str:
    """Sustituye cada token por 'c{cluster}' (o 'coov' si no esta)."""
    piezas = []
    for token in texto.split():
        clave = token.lower()
        etiqueta = mapa.get(clave)
        if etiqueta is None:
            etiqueta = mapa.get(clave.strip(PUNTUACION))
        piezas.append(f"c{etiqueta}" if etiqueta is not None else "coov")
    return " ".join(piezas)


def construir_matrices(textos_tr, textos_va, bloques):
    """Ajusta los bloques TF-IDF dentro del pliegue y transforma validacion."""
    mats_tr, mats_va = [], []
    for analyzer, ngram, fuente in bloques:
        vectorizador = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram,
                                       min_df=2, max_df=1.0,
                                       sublinear_tf=True, lowercase=True)
        mats_tr.append(vectorizador.fit_transform(textos_tr[fuente]))
        mats_va.append(vectorizador.transform(textos_va[fuente]))
    return sparse.hstack(mats_tr).tocsr(), sparse.hstack(mats_va).tocsr()


def oof_variante(textos, etiquetas, pliegues_list, bloques) -> np.ndarray:
    """OOF de una variante; vectorizador y LR ajustados dentro del pliegue."""
    proba = np.zeros((len(textos["raw"]), len(ETIQUETAS)))
    for idx_tr, idx_va in pliegues_list:
        tr = {fuente: arreglo[idx_tr] for fuente, arreglo in textos.items()}
        va = {fuente: arreglo[idx_va] for fuente, arreglo in textos.items()}
        m_tr, m_va = construir_matrices(tr, va, bloques)
        for j, etiqueta in enumerate(ETIQUETAS):
            modelo = LogisticRegression(**LR_PARAMS)
            modelo.fit(m_tr, etiquetas[etiqueta][idx_tr])
            proba[idx_va, j] = modelo.predict_proba(m_va)[:, 1]
    return proba


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


def _folds_iter(pliegues: np.ndarray):
    """Itera (train, val) desde un vector de pliegues."""
    for k in sorted(set(pliegues.tolist())):
        val = np.flatnonzero(pliegues == k)
        tr = np.flatnonzero(pliegues != k)
        yield tr, val


def main() -> None:
    """Ejecuta P104-02 completo."""
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
    log(f"desarrollo: {len(train_dev)} filas")

    pliegues_frescos = pliegues_corregidos(train_dev, grupos)
    etiquetas_fold = np.empty(len(train_dev), dtype=int)
    for num, (_, va) in enumerate(pliegues_frescos, start=1):
        etiquetas_fold[va] = num
    if not np.array_equal(etiquetas_fold, pliegues_can):
        log("AVISO: los folds frescos no coinciden 1:1 con el congelado")

    emb = np.load(EMB_PATH, allow_pickle=True)
    W = emb["W"].astype(np.float64)
    palabras = emb["palabras"]
    log(f"embedding: {W.shape[0]} palabras x {W.shape[1]}d")

    crudo = train_dev["text"].to_numpy()
    textos = {"raw": crudo}
    info_clusters = {}
    for k in VALORES_K:
        t0 = time.perf_counter()
        km = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=4096,
                             n_init=3)
        km.fit(W)
        mapa = construir_mapa(palabras, km.labels_)
        textos[f"clus{k}"] = np.array([texto_de_clusters(t, mapa)
                                       for t in crudo])
        tamanos = np.bincount(km.labels_, minlength=k)
        info_clusters[f"K{k}"] = {
            "n_clusters": k,
            "tam_min": int(tamanos.min()),
            "tam_max": int(tamanos.max()),
            "tam_mediana": int(np.median(tamanos)),
            "segundos": round(time.perf_counter() - t0, 1),
        }
        log(f"k-means k={k}: {info_clusters[f'K{k}']['segundos']} s | "
            f"tam min/med/max {tamanos.min()}/{int(np.median(tamanos))}/"
            f"{tamanos.max()}")

    variantes = {
        "control_M2": [("word", (1, 2), "raw"), ("char_wb", (3, 5), "raw")],
    }
    for k in VALORES_K:
        variantes[f"K{k}"] = [("word", (1, 2), "raw"),
                              ("char_wb", (3, 5), "raw"),
                              ("word", (1, 1), f"clus{k}")]

    etiquetas = {e: train_dev[e].to_numpy() for e in ETIQUETAS}

    log("control M2 canonico (debe reproducir el OOF congelado)")
    proba_control = oof_variante(textos, etiquetas, pliegues_frescos,
                                 variantes["control_M2"])
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
    proba_control_47 = oof_variante(textos, etiquetas,
                                    list(_folds_iter(pliegues_47)),
                                    variantes["control_M2"])
    diff_47_control = float(np.abs(proba_control_47 - proba_v03_47).max())
    celdas_control_47 = int((pred_umbrales(proba_control_47)
                             != pred_v03_47).sum())
    if celdas_control_47 != 0:
        raise SystemExit(f"control 47 no reproduce: {celdas_control_47} celdas")
    log(f"control 47 OK (celdas distintas {celdas_control_47}; "
        f"diff proba {diff_47_control:.2e})")

    resultados = {}
    for nombre, bloques in variantes.items():
        if nombre == "control_M2":
            continue
        log(f"variante {nombre}")
        proba = oof_variante(textos, etiquetas, pliegues_frescos, bloques)
        res_can = evaluar_candidato(proba, pred_v03, pliegues_can, V03_CCV,
                                    y, grupos)
        proba_47 = oof_variante(textos, etiquetas,
                                list(_folds_iter(pliegues_47)), bloques)
        res_47 = evaluar_candidato(proba_47, pred_v03_47, pliegues_47,
                                   ccv_v03_47, y, grupos)
        gate = gate_a3(res_can["delta_vs_v03"], res_can["pliegues_mejorados"],
                       res_can["bootstrap"]["ic95"], res_47["delta_vs_v03"],
                       res_can["celdas_cambiadas"])
        resultados[nombre] = {"canonica": res_can, "particion_47": res_47,
                              "gate": gate}
        log(f"  ccv {res_can['corrected_cv']:.6f} | delta "
            f"{res_can['delta_vs_v03']:+.6f} | pliegues "
            f"{res_can['pliegues_mejorados']}/5 | celdas "
            f"{res_can['celdas_cambiadas']} | 47 "
            f"{res_47['delta_vs_v03']:+.6f} | {gate['veredicto']}")

    mejor = max(resultados, key=lambda n:
                resultados[n]["canonica"]["delta_vs_v03"])
    rm = resultados[mejor]["canonica"]
    ic = rm["bootstrap"]["ic95"]
    gate_m = resultados[mejor]["gate"]
    if gate_m["veredicto"] == "GATE_PASSED":
        conclusion = (f"{mejor}: GATE_PASSED; candidato para el slot 2 "
                      "(frente a v08_meta_v2).")
    elif ic[1] < 0.0:
        conclusion = (f"{mejor}: efecto NEGATIVO detectable (IC95 "
                      f"[{ic[0]:+.6f},{ic[1]:+.6f}] excluye cero por abajo): "
                      "la variante perjudica; no es candidata.")
    elif ic[0] <= 0.0:
        conclusion = (f"{mejor}: sin senal detectable (IC95 "
                      f"[{ic[0]:+.6f},{ic[1]:+.6f}] incluye cero; piso de "
                      f"deteccion "
                      f"{rm['deteccion']['delta_min_detectable_95']:+.6f}).")
    else:
        conclusion = (f"{mejor}: delta positivo sobre el piso de deteccion, "
                      "pero gate A3 no superado.")
    log(f"conclusion: {conclusion}")

    salida = {
        "experiment_id": "P104-02",
        "fase": "P104 RONDA 3 (R5): clusters de palabras",
        "fecha": "2026-10-14",
        "embedding": {"ruta": "modelos/twitter_domain/w2v_twitter_lrfix.npz",
                      "palabras": int(W.shape[0]),
                      "dim": int(W.shape[1]),
                      "nota": "corpus externo Twitter/X sin etiquetas "
                              "(P71-02); sin folds ni texto de dev/test"},
        "clusters": info_clusters,
        "control": {"diff_max_proba": diff_control, "ccv": ccv_control,
                    "reproduce_v03": True,
                    "particion_47": {"celdas_distintas": celdas_control_47,
                                     "diff_max_proba": diff_47_control}},
        "v03_referencia": {"corrected_cv": V03_CCV, "umbrales": UMBRALES_V03},
        "variantes": resultados,
        "mejor_variante": mejor,
        "conclusion": conclusion,
        "sha_oof_final": sha256(OOF_FINAL)[:16],
        "tiempo_s": round(time.perf_counter() - inicio, 1),
    }
    SAL_JSON.write_text(json.dumps(salida, indent=2, ensure_ascii=True)
                        + "\n", encoding="utf-8")

    md = [("# P104-02 (R5) - Clusters de palabras (k-means sobre "
           "embeddings propios)"), "",
          (f"Fecha: 2026-10-14. Control reproduce V03 exacto (diff proba "
           f"{diff_control:.2e}; ccv {ccv_control:.16f}); particion 47 "
           f"{celdas_control_47} celdas distintas."),
          ("Piso de deteccion del arnes (pre-declarado): se ~0.0030; delta "
           "minimo detectable 95% ~+0.0059."), "",
          ("| Variante | corrected_cv | delta V03 | pliegues | celdas | "
           "IC95 vs V03 | delta 47 | Gate |"),
          "|---|---|---|---|---|---|---|---|"]
    for nombre, r in resultados.items():
        rc = r["canonica"]
        r47 = r["particion_47"]
        icv = rc["bootstrap"]["ic95"]
        md.append(
            f"| {nombre} | {rc['corrected_cv']:.6f} | "
            f"{rc['delta_vs_v03']:+.6f} | {rc['pliegues_mejorados']}/5 | "
            f"{rc['celdas_cambiadas']} | [{icv[0]:+.6f},{icv[1]:+.6f}] | "
            f"{r47['delta_vs_v03']:+.6f} | {r['gate']['veredicto']} |")
    md += ["", f"Conclusion: {conclusion}", "",
           ("Sin test; sin holdouts; sin leaderboard; umbrales V03 "
            "congelados."), ""]
    SAL_MD.write_text("\n".join(md), encoding="ascii")

    log(f"resultados: {SAL_JSON.name} y {SAL_MD.name} | tiempo "
        f"{salida['tiempo_s']} s")


if __name__ == "__main__":
    main()
