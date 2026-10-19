#!/usr/bin/env python
"""
P97-02 (H97-B): Group DRO minimax sobre los grupos del protocolo.

Reapertura B-class de M-P94-12 con cambio de hipotesis causal: lo probado
fueron reglas ESTATICAS de peso (V6-03, V4-B); aqui se ejecuta el objetivo
minimax iterativo (pesos adversarios por peor perdida de grupo, Sagawa 2020).

Pre-registro: reports/p97_independent_research.md, seccion 5 (H97-B):
  T = 15 rondas, eta = 0.5, q normalizado, sample_weight = q_g / n_g.
  Control A0 identico con pesos uniformes (debe reproducir V03).
  Metrica primaria: corrected_cv pooled con umbrales V03 congelados.

Salidas: features/p97_02_oof_dro.npz, reports/p97_02_group_dro.json y .md,
figura reports/figuras/p97_02_group_dro.png.

Uso:
    /home/carlos/venv/bin/python experiments/p97_02_group_dro.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from src import data as data_module  # noqa: E402
from src.evaluation import metrica_macro, metricas_multietiqueta  # noqa: E402

SAL_NPZ = RAIZ / "features" / "p97_02_oof_dro.npz"
SAL_JSON = RAIZ / "reports" / "p97_02_group_dro.json"
SAL_MD = RAIZ / "reports" / "p97_02_group_dro.md"
SAL_FIG = RAIZ / "reports" / "figuras" / "p97_02_group_dro.png"

UMBRALES_V03 = np.array([0.44, 0.51, 0.46])
T_RONDAS = 15
ETA = 0.5
SEMILLA = 20260925


def construir_matriz(textos_train: pd.Series, textos_aplicar: pd.Series):
    """TF-IDF word(1,2) + char_wb(3,5) de V03, ajustado SOLO en el train.

    Returns:
        (matriz_train, matriz_aplicar) como CSR.
    """
    palabra = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2,
                              sublinear_tf=True)
    caracter = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2,
                               sublinear_tf=True)
    m_pal = palabra.fit_transform(textos_train)
    m_car = caracter.fit_transform(textos_train)
    return (sparse.hstack([m_pal, m_car]).tocsr(),
            sparse.hstack([palabra.transform(textos_aplicar),
                           caracter.transform(textos_aplicar)]).tocsr())


def logloss_por_fila(proba: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Perdida logaritmica por fila para una etiqueta."""
    p = np.clip(proba, 1e-12, 1 - 1e-12)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def ajustar_lr(matriz, y, pesos_muestra):
    """LogisticRegression con la especificacion exacta de V03."""
    modelo = LogisticRegression(class_weight="balanced", C=2.0, max_iter=1000,
                                solver="liblinear", random_state=42)
    modelo.fit(matriz, y, sample_weight=pesos_muestra)
    return modelo


def arm_control(matriz_train, y_train, matriz_val):
    """A0: pesos uniformes."""
    proba_val = np.zeros((matriz_val.shape[0], 3))
    for posicion in range(3):
        modelo = ajustar_lr(matriz_train, y_train[:, posicion],
                            np.ones(matriz_train.shape[0]))
        proba_val[:, posicion] = modelo.predict_proba(matriz_val)[:, 1]
    return proba_val


def arm_dro(matriz_train, y_train, matriz_val, grupos_train, rng,
            registro_pesos):
    """DRO: minimax iterativo por peor perdida de grupo, por etiqueta.

    Nota de implementacion (correccion documentada): en liblinear el
    ``sample_weight`` escala el termino de perdida frente a ``C``; los pesos
    se normalizan para que sumen ``n_train`` (media 1) y la escala de ``C``
    quede intacta. La primera corrida con pesos sin normalizar (media ~1/n)
    produjo un colapso de escala, no del mecanismo; quedo preservada como
    ``*_escala_invalida``.
    """
    proba_val = np.zeros((matriz_val.shape[0], 3))
    n_train = matriz_train.shape[0]
    claves = np.unique(grupos_train)
    for posicion in range(3):
        q = {int(g): 1.0 / len(claves) for g in claves}
        for ronda in range(T_RONDAS):
            pesos = np.array([q[int(g)] / np.sum(grupos_train == g)
                              for g in grupos_train]) * n_train
            modelo = ajustar_lr(matriz_train, y_train[:, posicion], pesos)
            proba_train = modelo.predict_proba(matriz_train)[:, 1]
            perdida = logloss_por_fila(proba_train, y_train[:, posicion])
            nuevas = {}
            for g in claves:
                filas = grupos_train == g
                perdida_g = float(perdida[filas].mean())
                nuevas[int(g)] = q[int(g)] * float(np.exp(ETA * perdida_g))
            total = sum(nuevas.values())
            q = {k: v / total for k, v in nuevas.items()}
            registro_pesos.append({
                "etiqueta": posicion, "ronda": ronda,
                "peso_max": max(q.values()), "peso_min": min(q.values()),
            })
        pesos = np.array([q[int(g)] / np.sum(grupos_train == g)
                          for g in grupos_train]) * n_train
        modelo = ajustar_lr(matriz_train, y_train[:, posicion], pesos)
        proba_val[:, posicion] = modelo.predict_proba(matriz_val)[:, 1]
    return proba_val


def cuantiles_error_por_grupo(y, pred, grupos, minimo=3):
    """Cuantiles de la tasa de error por celda agregada por grupo."""
    errores = (pred != y).mean(axis=1)
    filas = []
    for g in np.unique(grupos):
        mascara = grupos == g
        if mascara.sum() >= minimo:
            filas.append(float(errores[mascara].mean()))
    arr = np.array(filas)
    return {"p50": float(np.quantile(arr, 0.50)), "p90": float(np.quantile(arr, 0.90)),
            "p99": float(np.quantile(arr, 0.99)), "n_grupos": int(len(arr))}


def bootstrap_clusters_delta(y, pred_a, pred_b, grupos, repeticiones=2000):
    """IC95 del delta pooled (a - b) remuestreando clusters."""
    generador = np.random.default_rng(SEMILLA)
    indices = {int(g): np.flatnonzero(grupos == g) for g in np.unique(grupos)}
    claves = np.array(sorted(indices))
    diferencias = np.empty(repeticiones)
    for r in range(repeticiones):
        muestra = generador.integers(0, len(claves), len(claves))
        idx = np.concatenate([indices[int(claves[i])] for i in muestra])
        diferencias[r] = metrica_macro(y[idx], pred_a[idx]) - metrica_macro(y[idx], pred_b[idx])
    c = np.quantile(diferencias, [0.025, 0.975])
    return {"ic95": [round(float(c[0]), 6), round(float(c[1]), 6)],
            "pct_mejora": round(float((diferencias > 0).mean()), 6)}


def main() -> None:
    """Ejecuta el pre-registro H97-B."""
    for ruta in (SAL_NPZ, SAL_JSON, SAL_MD, SAL_FIG):
        if ruta.exists():
            raise SystemExit(f"Ya existe {ruta}; no se sobrescribe.")
    inicio = time.perf_counter()
    z = np.load(RAIZ / "features" / "oof_v03_final.npz", allow_pickle=True)
    zb = np.load(RAIZ / "features" / "v2_oof_base_corrected.npz", allow_pickle=True)
    ids = z["ids"].astype(np.int64)
    y = z["y_true"].astype(int)
    pliegues = z["pliegues"].astype(int)
    grupos = zb["grupos"].astype(np.int64)
    assert np.array_equal(ids, zb["ids"].astype(np.int64))

    train = data_module.read_train()
    mapa = train.assign(_id=train["id"].astype(int)).set_index("_id")
    textos = mapa.loc[ids, "text"].reset_index(drop=True)

    proba_control = np.zeros((len(ids), 3))
    proba_dro = np.zeros((len(ids), 3))
    registro_pesos: list[dict] = []
    rng = np.random.default_rng(SEMILLA)
    tiempos = []
    for k in sorted(np.unique(pliegues).tolist()):
        t0 = time.perf_counter()
        idx_train = np.flatnonzero(pliegues != k)
        idx_val = np.flatnonzero(pliegues == k)
        m_train, m_val = construir_matriz(textos.iloc[idx_train], textos.iloc[idx_val])
        proba_control[idx_val] = arm_control(m_train, y[idx_train], m_val)
        proba_dro[idx_val] = arm_dro(m_train, y[idx_train], m_val,
                                     grupos[idx_train], rng, registro_pesos)
        tiempos.append(round(time.perf_counter() - t0, 2))
        print(f"  pliegue {k}: control y DRO listos ({tiempos[-1]} s)")

    pred_control = (proba_control >= UMBRALES_V03).astype(int)
    pred_dro = (proba_dro >= UMBRALES_V03).astype(int)
    met_control = metricas_multietiqueta(y, pred_control)
    met_dro = metricas_multietiqueta(y, pred_dro)
    f1_control = [float(metricas_multietiqueta(
        y[pliegues == k], pred_control[pliegues == k])["f1_media_columnas"])
        for k in sorted(np.unique(pliegues).tolist())]
    f1_dro = [float(metricas_multietiqueta(
        y[pliegues == k], pred_dro[pliegues == k])["f1_media_columnas"])
        for k in sorted(np.unique(pliegues).tolist())]
    pliegues_mejorados = int(np.sum(np.array(f1_dro) > np.array(f1_control) + 1e-12))
    delta = met_dro["f1_media_columnas"] - met_control["f1_media_columnas"]
    boot = bootstrap_clusters_delta(y, pred_dro, pred_control, grupos)
    std_control = float(np.std(f1_control, ddof=1))
    std_dro = float(np.std(f1_dro, ddof=1))

    referencia_reproduce = abs(met_control["f1_media_columnas"] - 0.7772114525165721) < 1e-9
    if (delta >= 0.003 and pliegues_mejorados >= 4):
        veredicto = "SENAL_CV"
    elif (std_dro < 0.9 * std_control
          and met_dro["f1_media_columnas"] >= 0.7772114525165721 - 0.001):
        veredicto = "ESTABILIZA_SIN_PERDIDA_MATERIAL"
    else:
        veredicto = "FALSIFICADA"

    resumen = {
        "experiment_id": "P97-02",
        "hipotesis": "H97-B",
        "fecha": "2026-10-13",
        "pre_registro": {"T_rondas": T_RONDAS, "eta": ETA,
                         "grupos": "near-duplicate UNION pseudo-autor",
                         "umbrales": "V03 congelados", "metric": "corrected_cv"},
        "control_A0": {
            "corrected_cv": met_control["f1_media_columnas"],
            "f1_por_etiqueta": met_control["f1_por_etiqueta"],
            "f1_pliegues": f1_control, "std_pliegues": std_control,
            "reproduce_v03_exacto": bool(referencia_reproduce),
        },
        "dro": {
            "corrected_cv": met_dro["f1_media_columnas"],
            "f1_por_etiqueta": met_dro["f1_por_etiqueta"],
            "f1_pliegues": f1_dro, "std_pliegues": std_dro,
        },
        "comparacion": {
            "delta_pooled": delta, "pliegues_mejorados": pliegues_mejorados,
            "bootstrap_clusters": boot,
            "error_por_grupo_control": cuantiles_error_por_grupo(y, pred_control, grupos),
            "error_por_grupo_dro": cuantiles_error_por_grupo(y, pred_dro, grupos),
        },
        "pesos_finales_diagnostico": registro_pesos[-6:],
        "correccion_implementacion": "pesos normalizados a media 1 (suma n_train) "
                                     "para preservar la escala de C de liblinear; "
                                     "la corrida previa sin normalizar quedo en "
                                     "p97_02_group_dro_escala_invalida.*",
        "tiempos_por_pliegue_s": tiempos,
        "veredicto_pre_registrado": veredicto,
        "tiempo_total_s": round(time.perf_counter() - inicio, 1),
    }
    np.savez_compressed(SAL_NPZ, ids=ids, y_true=y, grupos=grupos, pliegues=pliegues,
                        proba_control=proba_control, proba_dro=proba_dro)
    SAL_JSON.write_text(json.dumps(resumen, indent=2, ensure_ascii=True) + "\n",
                        encoding="utf-8")

    lineas = [
        "# P97-02 (H97-B): Group DRO minimax sobre grupos del protocolo",
        "",
        "Pre-registro en `p97_independent_research.md` seccion 5 (T=15, eta=0.5, "
        "pesos q_g/n_g, umbrales V03 congelados).",
        "",
        f"- Control A0 (pesos uniformes): corrected_cv "
        f"{met_control['f1_media_columnas']:.10f} "
        f"(reproduce V03 exacto: {referencia_reproduce}), std pliegues "
        f"{std_control:.5f}.",
        f"- DRO: corrected_cv {met_dro['f1_media_columnas']:.10f}, std pliegues "
        f"{std_dro:.5f}.",
        f"- Delta pooled {delta:+.6f}; pliegues mejorados {pliegues_mejorados}/5; "
        f"IC95 clusters {boot['ic95']} (positivos {boot['pct_mejora']}).",
        f"- Error por grupo (celda agregada): control "
        f"{resumen['comparacion']['error_por_grupo_control']} | "
        f"dro {resumen['comparacion']['error_por_grupo_dro']}.",
        "",
        f"**Veredicto pre-registrado: {veredicto}**",
    ]
    SAL_MD.write_text("\n".join(lineas) + "\n", encoding="utf-8")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ejes = plt.subplots(1, 2, figsize=(11, 4.2))
    xs = np.arange(5)
    ejes[0].bar(xs - 0.18, f1_control, 0.36, label="A0 control")
    ejes[0].bar(xs + 0.18, f1_dro, 0.36, label="DRO")
    ejes[0].set_xticks(xs, [f"p{k}" for k in sorted(np.unique(pliegues).tolist())])
    ejes[0].set_title("F1 por pliegue (umbrales V03)")
    ejes[0].legend(fontsize=8)
    pesos = [p["peso_max"] for p in registro_pesos if p["etiqueta"] == 0]
    ejes[1].plot(range(len(pesos)), pesos, marker="o", ms=2)
    ejes[1].set_title("DRO TR: peso maximo de grupo por ronda")
    ejes[1].set_xlabel("ronda (pliegues concatenados)")
    fig.tight_layout()
    SAL_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(SAL_FIG, dpi=130)
    plt.close(fig)

    print("P97-02 H97-B: ejecutado")
    print(f"  A0: {met_control['f1_media_columnas']:.10f} (V03 exacto: "
          f"{referencia_reproduce}) | DRO: {met_dro['f1_media_columnas']:.10f}")
    print(f"  delta={delta:+.6f} | pliegues {pliegues_mejorados}/5 | "
          f"IC clusters {boot['ic95']}")
    print(f"  veredicto: {veredicto} | tiempo total {resumen['tiempo_total_s']} s")


if __name__ == "__main__":
    main()
