"""Sonda V24-03 (P2): kNN / retrieval como PREDICTOR, no como medida.

Pregunta que responde
---------------------
El gap analysis de V6 (`experiments/v6_00_gap_analysis.py`, entrada
`E_knn_predictor`) dejo abierto un hueco explicito: los vecinos de 3 se
construyeron en V4-F03 como MEDIDA de ambiguedad (`knn_disagreement`,
`sim_cercana`), pero nunca se usaron como PREDICTOR. Esta sonda ejecuta esa
pieza que faltaba y mide si puede competir con V03.

Lo que NO hace
--------------
- No genera submission. Diagnostico de gate.
- No toca el holdout.
- No reentrena V03: reutiliza su OOF corregido cacheado.

Salida
------
`reports/v24_03_sonda_knn_predictor.md` y `reports/v24_03_sonda_knn.json`.
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
from sklearn.metrics import f1_score
from sklearn.neighbors import NearestNeighbors

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src import data as data_module
from src import splits as splits_module
from src.evaluation import aplicar_postproceso_jerarquico

REPORTS_DIR = PROJECT_ROOT / "reports"
OOF_V03 = PROJECT_ROOT / "features" / "oof_v03_final.npz"

ETIQUETAS = ("HS", "TR", "AG")
UMBRALES_V03 = {"HS": 0.44, "TR": 0.51, "AG": 0.46}

PARAMS_WORD = {
    "analyzer": "word",
    "ngram_range": (1, 2),
    "min_df": 2,
    "sublinear_tf": True,
    "lowercase": True,
}
PARAMS_CHAR = {
    "analyzer": "char_wb",
    "ngram_range": (3, 5),
    "min_df": 2,
    "sublinear_tf": True,
    "lowercase": True,
}

SALIDA: list[str] = []


def log(mensaje: str) -> None:
    """Acumula y muestra una linea de log."""
    SALIDA.append(mensaje)
    print(mensaje, flush=True)


# ---------------------------------------------------------------------------
# Utilidades de metricas
# ---------------------------------------------------------------------------


def f1_columnas(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, list[float]]:
    """F1 positiva por etiqueta y su media (la metrica de la competencia)."""
    por_etiqueta = [
        f1_score(y_true[:, i], y_pred[:, i], zero_division=0) for i in range(3)
    ]
    return float(np.mean(por_etiqueta)), [float(v) for v in por_etiqueta]


def predecir(proba: np.ndarray, umbrales: np.ndarray, jerarquico: bool = False) -> np.ndarray:
    """Aplica umbrales por etiqueta, opcionalmente con el post-proceso HS=0.

    El protocolo de referencia de V03 (``pipeline_v03.json`` y la nota de
    mapeo de ``oof_v03_final.npz``) es **umbrales solos**: la prediccion
    congelada es ``proba >= [0.44, 0.51, 0.46]`` sin post-proceso jerarquico.
    Por defecto se reproduce ese protocolo para que la comparacion sea exacta.
    """
    bruto = (proba >= umbrales[None, :]).astype(int)
    if jerarquico:
        return aplicar_postproceso_jerarquico(bruto)
    return bruto


UMBRALES_VEC = np.array([UMBRALES_V03[e] for e in ETIQUETAS])


# ---------------------------------------------------------------------------
# Vecindarios
# ---------------------------------------------------------------------------


def vecinos_ples(
    matriz_entrenamiento: sparse.csr_matrix,
    matriz_validacion: sparse.csr_matrix,
    y_entrenamiento: np.ndarray,
    k: int = 25,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vecinos mas cercanos por cosimilitud y voto local ponderado.

    Args:
        matriz_entrenamiento: TF-IDF del pliegue de entrenamiento.
        matriz_validacion: TF-IDF del pliegue de validacion.
        y_entrenamiento: etiquetas binarias del pliegue de entrenamiento.
        k: numero de vecinos a recuperar.

    Returns:
        Tupla ``(ids_vecinos, similitudes, prob_knn)`` con forma
        ``(n_validacion, k)``, ``(n_validacion, k)``, ``(n_validacion, 3)``.
    """
    buscador = NearestNeighbors(n_neighbors=k, metric="cosine", algorithm="brute")
    buscador.fit(matriz_entrenamiento)
    distancias, indices = buscador.kneighbors(matriz_validacion)
    similitudes = np.clip(1.0 - distancias, 0.0, 1.0)

    vecinos = indices
    sims = similitudes
    y_vec = y_entrenamiento[vecinos]  # (n_val, k, 3)
    pesos = sims[:, :, None]
    total = pesos.sum(axis=1)
    total[total == 0.0] = 1.0
    prob_knn = (pesos * y_vec).sum(axis=1) / total
    return vecinos, sims, prob_knn


def construir_union(entrenamiento: pd.Series, validacion: pd.Series):
    """TF-IDF union palabra+caracteres ajustado solo en entrenamiento."""
    word = TfidfVectorizer(**PARAMS_WORD)
    char = TfidfVectorizer(**PARAMS_CHAR)
    me = sparse.hstack([word.fit_transform(entrenamiento), char.fit_transform(entrenamiento)]).tocsr()
    mv = sparse.hstack([word.transform(validacion), char.transform(validacion)]).tocsr()
    return me, mv


def construir_char(entrenamiento: pd.Series, validacion: pd.Series):
    """Solo char_wb(3,5), el espacio donde la geometria local difiere mas."""
    char = TfidfVectorizer(**PARAMS_CHAR)
    return char.fit_transform(entrenamiento).tocsr(), char.transform(validacion).tocsr()


# ---------------------------------------------------------------------------
# Cuerpo
# ---------------------------------------------------------------------------


def main() -> None:
    inicio = time.time()
    log("# V24-03 (P2): sonda de kNN como predictor")
    log("")
    log("Diagnostico de gate. Sin submission. Sin tocar el holdout.")
    log("")

    # --- Datos -----------------------------------------------------------
    train = data_module.read_train()
    grupos = splits_module.grupos_fusionados(train)
    idx_dev, idx_holdout = splits_module.holdout_v2(train, grupos)
    train_dev = train.iloc[idx_dev].reset_index(drop=True)
    grupos_dev = grupos[idx_dev]
    pliegues = splits_module.pliegues_corregidos(train_dev, grupos_dev)

    log("## 0. Marco")
    log(f"- train {len(train)} | dev {len(train_dev)} | holdout {len(idx_holdout)} (intacto)")
    log(f"- pliegues: {[(len(a), len(b)) for a, b in pliegues]}")
    log("")

    y_dev = train_dev[list(ETIQUETAS)].to_numpy()

    # --- V03 cacheado ---------------------------------------------------
    cache = np.load(OOF_V03, allow_pickle=True)
    assert np.array_equal(cache["ids"].astype(int), train_dev["id"].to_numpy().astype(int)), (
        "los ids del OOF cacheado no coinciden con el desarrollo"
    )
    proba_v03 = cache["proba"]
    pred_v03 = cache["pred"].astype(int)
    f1_v03, por_etiqueta_v03 = f1_columnas(y_dev, pred_v03)
    celdas_v03 = int((pred_v03 != y_dev).sum())
    celdas_totales = int(y_dev.size)
    log("## 1. V03 referencia (OOF corregido, umbrales congelados)")
    log(f"- corrected_cv = **{f1_v03:.4f}** (por etiqueta {', '.join(f'{v:.4f}' for v in por_etiqueta_v03)})")
    log(f"- celdas falladas = **{celdas_v03}** de {celdas_totales}")
    log(f"- reproduce el valor canonico 0.777211: {abs(f1_v03 - 0.77721145) < 1e-6}")
    log(f"- celdas que el post-proceso jerarquico cambiaria: "
        f"{int((pred_v03 != predecir(proba_v03, UMBRALES_VEC, jerarquico=True)).sum())} "
        f"(protocolo V03 = umbrales solos, el post-proceso NO se aplica)")
    log("")

    # --- Vecindarios en dos espacios ------------------------------------
    resultados_espacios = {}
    for nombre, constructor in (("union", construir_union), ("char", construir_char)):
        prob_knn = np.zeros((len(train_dev), 3), dtype=float)
        sim_top1 = np.zeros(len(train_dev), dtype=float)
        sim_top5 = np.zeros(len(train_dev), dtype=float)
        prob_knn_top1 = np.zeros((len(train_dev), 3), dtype=float)
        prob_knn_top3 = np.zeros((len(train_dev), 3), dtype=float)
        prob_knn_top5 = np.zeros((len(train_dev), 3), dtype=float)
        prob_knn_p5 = np.zeros((len(train_dev), 3), dtype=float)
        for idx_tr, idx_va in pliegues:
            me, mv = constructor(train_dev["text"].iloc[idx_tr], train_dev["text"].iloc[idx_va])
            y_tr = y_dev[idx_tr]
            vecinos, sims, p25 = vecinos_ples(me, mv, y_tr, k=25)
            prob_knn[idx_va] = p25
            sim_top1[idx_va] = sims[:, 0]
            sim_top5[idx_va] = sims[:, :5].mean(axis=1)
            prob_knn_top1[idx_va] = y_tr[vecinos[:, 0]]
            for kk in (3, 5):
                w = sims[:, :kk]
                tot = w.sum(axis=1)
                tot[tot == 0.0] = 1.0
                p_kk = (w[:, :, None] * y_tr[vecinos[:, :kk]]).sum(axis=1) / tot[:, None]
                if kk == 3:
                    prob_knn_top3[idx_va] = p_kk
                else:
                    prob_knn_top5[idx_va] = p_kk
            idx_p5 = vecinos[:, :5]
            w_p5 = sims[:, :5]
            tot_p5 = w_p5.sum(axis=1)
            tot_p5[tot_p5 == 0.0] = 1.0
            prob_knn_p5[idx_va] = (w_p5[:, :, None] * y_tr[idx_p5]).sum(axis=1) / tot_p5[:, None]

        resultados_espacios[nombre] = {
            "p25": prob_knn,
            "p1": prob_knn_top1,
            "p3": prob_knn_top3,
            "p5": prob_knn_top5,
            "p5w": prob_knn_p5,
            "sim_top1": sim_top1,
            "sim_top5": sim_top5,
        }
        log(f"## 2. kNN en espacio `{nombre}` (OOF agrupado, sin fuga)")
        for clave in ("p1", "p3", "p5w", "p25"):
            f1_k, _ = f1_columnas(y_dev, predecir(resultados_espacios[nombre][clave], UMBRALES_VEC))
            log(f"- corrected_cv kNN {clave:>4} = {f1_k:.4f}  (V03 = {f1_v03:.4f}, delta {f1_k - f1_v03:+.4f})")
        log(f"- similitud top-1: mediana {np.median(resultados_espacios[nombre]['sim_top1']):.4f}, "
            f"p90 {np.percentile(resultados_espacios[nombre]['sim_top1'], 90):.4f}, "
            f"max {resultados_espacios[nombre]['sim_top1'].max():.4f}")
        log(f"- filas con sim top-1 > 0.5: {int((resultados_espacios[nombre]['sim_top1'] > 0.5).sum())} "
            f"({(resultados_espacios[nombre]['sim_top1'] > 0.5).mean() * 100:.1f} %)")
        log(f"- filas con sim top-1 > 0.7: {int((resultados_espacios[nombre]['sim_top1'] > 0.7).sum())} "
            f"({(resultados_espacios[nombre]['sim_top1'] > 0.7).mean() * 100:.1f} %)")
        log("")

    # --- Puntos B a E sobre el espacio mas promesa -----------------------
    # Usamos `union` (mismo espacio que V03) y `char` (geometria distinta).
    analisis = {}
    for nombre in ("union", "char"):
        res = resultados_espacios[nombre]
        sims = res["sim_top1"]
        # Vecino "confiable": top-1 con cosimilitud por encima del percentil 90
        # y ademas coincidencia de etiqueta en los 5 vecinos (desacuerdo bajo).
        p90 = float(np.percentile(sims, 90))
        flag_cercano = sims >= p90
        desacuerdo = (res["p5"].round() != y_dev).sum(axis=1)
        unidad = flag_cercano & (desacuerdo <= 1)

        fila = {"p90_sim": p90, "n_confiables": int(unidad.sum()), "frac_confiables": float(unidad.mean())}

        # B: cobertura de los errores de V03 por un vecino confiable
        error_v03 = pred_v03 != y_dev
        fila["celdas_error_v03"] = int(error_v03.sum())
        fila["celdas_error_v03_con_vecino_confiable"] = int((error_v03 & unidad[:, None]).sum())
        fila["cobertura_B"] = float((error_v03 & unidad[:, None]).sum() / max(error_v03.sum(), 1))

        # C+E: precision de kNN en las celdas donde kNN contradice a V03
        # Regla realizable: donde el vecino top-1 es confidently cercano y su
        # etiqueta difiere de la de V03, se adopta la etiqueta del vecino.
        flip = unidad[:, None] & (res["p1"].astype(int) != pred_v03)
        fila["celdas_flip"] = int(flip.sum())
        fila["flip_correctas"] = int((flip & (res["p1"].astype(int) == y_dev)).sum())
        fila["flip_falladas"] = int((flip & (res["p1"].astype(int) != y_dev)).sum())
        fila["precision_flip_C"] = float(
            (flip & (res["p1"].astype(int) == y_dev)).sum() / max(flip.sum(), 1)
        )
        # Volumen neto exigido para superar el umbral de gate (+0.03756 dOOF)
        fila["rescates_flip"] = fila["flip_correctas"]
        fila["danos_flip"] = fila["flip_falladas"]

        # Aplicacion realizable de la regla
        pred_flip = pred_v03.copy()
        pred_flip[flip] = res["p1"].astype(int)[flip]
        f1_flip, _ = f1_columnas(y_dev, pred_flip)
        fila["f1_regla_flip"] = f1_flip
        fila["dOOF_regla_flip"] = f1_flip - f1_v03

        # Mezcla convexa de probabilidades V03 + kNN (20% de peso kNN)
        for peso in (0.10, 0.20, 0.30):
            mezcla = (1 - peso) * proba_v03 + peso * res["p25"]
            f1_m, _ = f1_columnas(y_dev, predecir(mezcla, UMBRALES_VEC))
            fila[f"f1_mezcla_w{peso:.2f}"] = f1_m
            fila[f"dOOF_mezcla_w{peso:.2f}"] = f1_m - f1_v03

        # D: complementariedad. McNemar sobre las celdas donde kNN y V03 discrepan.
        difieren = (res["p1"].astype(int) != pred_v03)
        a_v03_gana = int((difieren & (pred_v03 == y_dev)).sum())
        b_knn_gana = int((difieren & (res["p1"].astype(int) == y_dev)).sum())
        fila["discrepan"] = int(difieren.sum())
        fila["solo_v03_gana"] = a_v03_gana
        fila["solo_knn_gana"] = b_knn_gana
        # McNemar exacto (binomial, sin correccion de continuidad)
        from math import comb
        n = a_v03_gana + b_knn_gana
        if n > 0:
            k = min(a_v03_gana, b_knn_gana)
            p_exact = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / (2 ** n))
        else:
            p_exact = 1.0
        fila["mcnemar_p"] = p_exact
        # Jaccard de errores: kNN solo, V03 solo
        err_knn = res["p1"].astype(int) != y_dev
        union = (error_v03 | err_knn).sum()
        fila["jaccard_errores"] = float((error_v03 & err_knn).sum() / max(union, 1))
        fila["err_knn_solo"] = int((err_knn & ~error_v03).sum())
        fila["err_v03_solo"] = int((error_v03 & ~err_knn).sum())
        fila["err_ambos"] = int((error_v03 & err_knn).sum())

        # ORACLE: techo absoluto de cualquier regla de seleccion por celda
        # entre V03 y el vecino top-1. No es realizable; es la cota.
        pred_oracle = np.where(
            (pred_v03 == y_dev) | (res["p1"].astype(int) != y_dev), pred_v03, res["p1"].astype(int)
        )
        f1_oracle, _ = f1_columnas(y_dev, pred_oracle)
        fila["f1_oracle_techo"] = f1_oracle
        fila["dOOF_oracle_techo"] = f1_oracle - f1_v03

        analisis[nombre] = fila
        log(f"## 3. Puntos B-F en espacio `{nombre}`")
        log(f"- B. Vecinos confiables (sim top-1 >= p90 = {p90:.4f} y desacuerdo 5-vecinos <= 1): "
            f"**{fila['n_confiables']} de {len(train_dev)} filas** ({fila['frac_confiables'] * 100:.1f} %)")
        log(f"   Celdas de error de V03 cubiertas por un vecino confiable: "
            f"**{fila['celdas_error_v03_con_vecino_confiable']} de {fila['celdas_error_v03']}** "
            f"= **{fila['cobertura_B'] * 100:.1f} %**")
        log(f"- C. Precision de kNN al contradecir a V03: "
            f"**{fila['flip_correctas']} correctas / {fila['celdas_flip']} volteos = {fila['precision_flip_C']:.4f}**")
        log(f"   McNemar: V03 gana {fila['solo_v03_gana']} celdas donde discrepan, kNN gana "
            f"{fila['solo_knn_gana']}, p = {fila['mcnemar_p']:.4f}")
        log(f"- D. Complementariedad: errors solo-V03 {fila['err_v03_solo']}, solo-kNN {fila['err_knn_solo']}, "
            f"ambos {fila['err_ambos']}, Jaccard {fila['jaccard_errores']:.3f}")
        log(f"- E. Volumen: red = {fila['rescates_flip']} - {fila['danos_flip']} = "
            f"{fila['rescates_flip'] - fila['danos_flip']} celdas. regla realizable -> corrected_cv "
            f"{f1_flip:.4f} (dOOF {f1_flip - f1_v03:+.4f})")
        for peso in (0.10, 0.20, 0.30):
            log(f"   mezcla de probabilidades peso kNN {peso:.2f} -> corrected_cv "
                f"{fila[f'f1_mezcla_w{peso:.2f}']:.4f} (dOOF {fila[f'dOOF_mezcla_w{peso:.2f}']:+.4f})")
        log(f"- ORACLE (cota superior, NO realizable): corrected_cv {f1_oracle:.4f} "
            f"(dOOF {f1_oracle - f1_v03:+.4f})")
        log("")

    # --- F: se solapa con V4-F03 o con V4-F01-F04? ----------------------
    # V4-F03 uso el vecindario como MEDIDA de ambiguedad (desacuerdo del vecino
    # 1, desacuerdo ponderado de 3, sim_cercana) y luego como feature de una
    # segunda etapa. Aqui el vecindario ES la prediccion. Para medir el
    # solapamiento, correlacionamos la prediccion kNN con el desacuerdo que
    # V4-F03 ya uso como feature: si coinciden, la formulacion no es nueva.
    res_union = resultados_espacios["union"]
    desacuerdo_3 = np.abs(res_union["p5"].round() - y_dev).mean(axis=1)
    difieren_union = (res_union["p1"].astype(int) != pred_v03)
    difieren_fila = difieren_union.mean(axis=1)
    from scipy import stats as st
    corr_disagree_vs_diff, _ = st.pearsonr(desacuerdo_3, difieren_fila)
    # Y con la propia sim_cercana de V4-F03
    corr_sim_vs_diff, _ = st.pearsonr(res_union["sim_top1"], difieren_fila)
    log("## 4. Punto F: distincion frente a V4-F03 y V4-F01-F04")
    log("- V4-F03 uso `knn_disagreement` (desacuerdo ponderado de 3) y `sim_cercana` (cos del vecino 1)")
    log("  como FEATURES para explicar el error. Aqui el voto de los vecinos es la PREDICCION.")
    log("- Correlacion Pearson entre el desacuerdo 3-vecinos (medida V4-F03) y la tasa de celdas donde")
    log(f"  el voto kNN contradice a V03: **{corr_disagree_vs_diff:+.4f}**")
    log(f"- Correlacion Pearson entre `sim_cercana` (medida V4-F03) y esa misma tasa: "
        f"**{corr_sim_vs_diff:+.4f}**")
    log(f"- Correlacion Pearson entre el voto kNN ponderado y la probabilidad de V03 "
        f"(mismo espacio `union`): **{st.pearsonr(res_union['p25'].ravel(), proba_v03.ravel())[0]:+.4f}**")
    log(f"- Correlacion Pearson entre el voto kNN (espacio `char`) y la probabilidad de V03: "
        f"**{st.pearsonr(resultados_espacios['char']['p25'].ravel(), proba_v03.ravel())[0]:+.4f}**")
    log("")

    # --- Barrido amplio: generosidad maxima al candidato -----------------
    log("## 4b. Barrido amplio de la mezcla convexa V03 + kNN (reproducibilidad por pliegue)")
    pliegues_dev = cache["pliegues"]
    mejor = None
    for nombre in ("union", "char"):
        p_knn = resultados_espacios[nombre]["p25"]
        for peso in np.arange(0.05, 0.85, 0.05):
            mezcla = (1 - peso) * proba_v03 + peso * p_knn
            pred_m = predecir(mezcla, UMBRALES_VEC)
            f1_m, _ = f1_columnas(y_dev, pred_m)
            por_pliegue = [
                f1_columnas(y_dev[pliegues_dev == p], pred_m[pliegues_dev == p])[0]
                for p in np.unique(pliegues_dev)
            ]
            base_por_pliegue = [
                f1_columnas(y_dev[pliegues_dev == p], pred_v03[pliegues_dev == p])[0]
                for p in np.unique(pliegues_dev)
            ]
            gana = sum(1 for a, b in zip(por_pliegue, base_por_pliegue) if a > b + 1e-9)
            if mejor is None or f1_m > mejor[0]:
                mejor = (f1_m, nombre, float(peso), gana, por_pliegue, base_por_pliegue)
    f1_best, espacio_best, peso_best, gana_best, pp_best, bp_best = mejor
    log(f"- Mejor mezcla convexa: espacio `{espacio_best}`, peso kNN {peso_best:.2f} -> "
        f"corrected_cv **{f1_best:.4f}** (dOOF {f1_best - f1_v03:+.4f}), gana en **{gana_best} de 5 pliegues**")
    log(f"- corrected_cv por pliegue kNN: {', '.join(f'{v:.4f}' for v in pp_best)}")
    log(f"- corrected_cv por pliegue V03: {', '.join(f'{v:.4f}' for v in bp_best)}")
    log("- dOOF necesario para superar V03 en LB segun el ajuste del historial: +0.03756")
    log(f"- La mejor mezcla real queda en {f1_best - f1_v03:+.4f}: "
        f"**{(f1_best - f1_v03) / 0.03756 * 100:.0f} % del umbral de gate**")
    log("")

    # --- Sintesis -------------------------------------------------------
    log("## 5. Sintesis del gate")
    for nombre in ("union", "char"):
        fila = analisis[nombre]
        techo = fila["dOOF_oracle_techo"]
        mejor_f = max(
            fila["dOOF_regla_flip"],
            fila["dOOF_mezcla_w0.10"],
            fila["dOOF_mezcla_w0.20"],
            fila["dOOF_mezcla_w0.30"],
        )
        log(f"- espacio `{nombre}`: mejor regla REALIZABLE dOOF = {mejor_f:+.4f}; "
            f"techo ORACLE = {techo:+.4f}; umbral de gate dOOF = +0.03756")
    log(f"- mejor mezcla convexa en todo el barrido: **{f1_best - f1_v03:+.4f}** de dOOF, "
        f"gana en {gana_best} de 5 pliegues")
    log("")

    payload = {
        "version": "V24-03",
        "fecha": time.strftime("%Y-%m-%d"),
        "f1_v03_corrected": f1_v03,
        "celdas_error_v03": celdas_v03,
        "analisis": analisis,
        "segundos": round(time.time() - inicio, 1),
    }
    (REPORTS_DIR / "v24_03_sonda_knn.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (REPORTS_DIR / "v24_03_sonda_knn_predictor.md").write_text(
        "\n".join(SALIDA) + "\n", encoding="utf-8"
    )
    print(f"\nCompletado en {time.time() - inicio:.1f} s")


if __name__ == "__main__":
    main()
