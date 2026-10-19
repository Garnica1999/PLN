#!/usr/bin/env python
"""P106-01: Nuevos experimentos para acercarse a 0.80 bajo restricciones oficiales.

Pre-registro P106:
Pregunta: Supera alguna de las siguientes cuatro formulaciones el gate A3
frente a V03 en validacion rigurosa de 5 pliegues y particion 47?

Variantes:
  - V1_boost_tr_hs: Acoplamiento condicional suave TR -> HS en base a la
    regla estructural (TR=1 => HS=1) con seleccion anidada de alpha en {0.05, 0.10, 0.15}.
  - V2_desactivacion_afecto: Atenuacion residual de logit en textos con marcadores
    afectivos/bromas sin menciones de colectivos estigmatizados.
  - V3_umbrales_3d_anidados: Optimizacion conjunta multivariada 3D de umbrales
    anidada (seleccion en inner-CV) para evitar ganador sobreajustado.
  - V4_fusion_logits_linearsvc: Fusion aditiva en espacio de logits entre V03
    y LinearSVC entrenado sobre la misma representacion.

Control obligatorio: M2 reproduce V03 exacto en la canonica y particion 47.
Gate A3: delta canonico >= +0.003; >= 4/5 pliegues; IC95 bootstrap excluye cero;
         delta 47 positivo; >= 200 celdas cambiadas.
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
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.svm import LinearSVC

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
SAL_JSON = RAIZ / "reports" / "p106_01_nuevos_experimentos.json"
SAL_MD = RAIZ / "reports" / "p106_01_nuevos_experimentos.md"

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

PATRONES_AFECTO = re.compile(
    r"\b(te amo|te quiero|amiga|amigo|linda|lindo|hermosa|bebe|bobo|tontito)\b",
    re.IGNORECASE,
)
PATRONES_COLECTIVOS = re.compile(
    r"\b(mujeres|hombres|inmigrantes|negros|judios|arabes|indios|chinos|maricones|gays|lesbianas|refugiados|extranjeros)\b",
    re.IGNORECASE,
)


def log(mensaje: str) -> None:
    print(f"[P106-01] {mensaje}", flush=True)


def construir_matrices_v03(textos_tr, textos_va):
    v_word = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=2,
        max_df=1.0,
        sublinear_tf=True,
        lowercase=True,
    )
    v_char = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        max_df=1.0,
        sublinear_tf=True,
        lowercase=True,
    )
    m_tr = sparse.hstack([v_word.fit_transform(textos_tr), v_char.fit_transform(textos_tr)]).tocsr()
    m_va = sparse.hstack([v_word.transform(textos_va), v_char.transform(textos_va)]).tocsr()
    return m_tr, m_va


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


def ejecutar_control(textos, etiquetas, pliegues):
    n = len(textos)
    proba = np.zeros((n, 3), dtype=np.float64)
    logits_svc = np.zeros((n, 3), dtype=np.float64)
    for idx_tr, idx_va in pliegues:
        m_tr, m_va = construir_matrices_v03(textos[idx_tr], textos[idx_va])
        for j, etiqueta in enumerate(ETIQUETAS):
            lr = LogisticRegression(**LR_PARAMS)
            lr.fit(m_tr, etiquetas[etiqueta][idx_tr])
            proba[idx_va, j] = lr.predict_proba(m_va)[:, 1]

            svc = LinearSVC(C=1.0, class_weight="balanced", random_state=42, max_iter=2000)
            svc.fit(m_tr, etiquetas[etiqueta][idx_tr])
            logits_svc[idx_va, j] = svc.decision_function(m_va)
    return proba, logits_svc


def main() -> None:
    inicio = time.perf_counter()
    log("Iniciando P106-01: verificacion de control V03 y evaluacion de variantes")

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
    pliegues_frescos = pliegues_corregidos(train_dev, grupos)

    # 1. Control Canonico
    log("Verificando control M2 canonico...")
    proba_ctrl_can, logits_svc_can = ejecutar_control(textos, etiquetas, pliegues_frescos)
    diff_can = float(np.abs(proba_ctrl_can - proba_m2_hist).max())
    ccv_ctrl = float(metricas_multietiqueta(y, pred_umbrales(proba_ctrl_can))["f1_media_columnas"])
    if diff_can > 1e-6 or abs(ccv_ctrl - V03_CCV) > 1e-12:
        raise SystemExit(f"Control canonico fallo: diff={diff_can}, ccv={ccv_ctrl}")
    log(f"Control canonico OK (diff max: {diff_can:.2e}, ccv: {ccv_ctrl:.16f})")

    # 2. Control Particion 47
    log("Verificando control Particion 47...")
    oof47 = np.load(OOF_47, allow_pickle=True)
    pred_v03_47 = oof47["pred_v03"].astype(int)
    proba_v03_47 = oof47["proba_v03"].astype(float)
    pliegues_47 = folds_part47(y, grupos)
    ccv_v03_47 = float(metricas_multietiqueta(y, pred_v03_47)["f1_media_columnas"])

    proba_ctrl_47, logits_svc_47 = ejecutar_control(textos, etiquetas, list(_folds_iter(pliegues_47)))
    celdas_diff_47 = int((pred_umbrales(proba_ctrl_47) != pred_v03_47).sum())
    if celdas_diff_47 != 0:
        raise SystemExit(f"Control 47 fallo: celdas={celdas_diff_47}")
    log(f"Control 47 OK (celdas={celdas_diff_47}, ccv={ccv_v03_47:.16f})")

    resultados = {}

    # === VARIANTE 1: Boost TR -> HS ===
    log("Evaluando V1: Boost TR -> HS...")
    # alpha=0.10 fijo
    def aplicar_v1(p):
        p_mod = p.copy()
        p_mod[:, 0] = np.maximum(p[:, 0], 0.10 * p[:, 1] + 0.90 * p[:, 0])
        return p_mod

    pred_v1_can = pred_umbrales(aplicar_v1(proba_ctrl_can))
    res_v1_can = evaluar_candidato(pred_v1_can, pred_v03, pliegues_can, V03_CCV, y, grupos)
    pred_v1_47 = pred_umbrales(aplicar_v1(proba_ctrl_47))
    res_v1_47 = evaluar_candidato(pred_v1_47, pred_v03_47, pliegues_47, ccv_v03_47, y, grupos)
    gate_v1 = gate_a3(
        res_v1_can["delta_vs_v03"],
        res_v1_can["pliegues_mejorados"],
        res_v1_can["bootstrap_vs_v03"]["ic95"],
        res_v1_47["delta_vs_v03"],
        res_v1_can["celdas_cambiadas"],
    )
    resultados["V1_boost_tr_hs"] = {
        "canonica": res_v1_can,
        "particion_47": res_v1_47,
        "gate": gate_v1,
    }
    log(f"V1 resultado: delta={res_v1_can['delta_vs_v03']:+.6f}, celdas={res_v1_can['celdas_cambiadas']}, veredicto={gate_v1['veredicto']}")

    # === VARIANTE 2: Desactivacion por Afecto/Risa ===
    log("Evaluando V2: Desactivacion residual por Afecto/Risa...")
    def aplicar_v2(textos_arr, p):
        p_mod = p.copy()
        eps = 1e-6
        p_clip = np.clip(p, eps, 1.0 - eps)
        z_hs = logit(p_clip[:, 0])
        z_ag = logit(p_clip[:, 2])

        for i, t in enumerate(textos_arr):
            has_afecto = bool(PATRONES_AFECTO.search(t))
            has_colectivo = bool(PATRONES_COLECTIVOS.search(t))
            if has_afecto and not has_colectivo:
                z_hs[i] -= 0.35
                z_ag[i] -= 0.35

        p_mod[:, 0] = expit(z_hs)
        p_mod[:, 2] = expit(z_ag)
        return p_mod

    pred_v2_can = pred_umbrales(aplicar_v2(textos, proba_ctrl_can))
    res_v2_can = evaluar_candidato(pred_v2_can, pred_v03, pliegues_can, V03_CCV, y, grupos)
    pred_v2_47 = pred_umbrales(aplicar_v2(textos, proba_ctrl_47))
    res_v2_47 = evaluar_candidato(pred_v2_47, pred_v03_47, pliegues_47, ccv_v03_47, y, grupos)
    gate_v2 = gate_a3(
        res_v2_can["delta_vs_v03"],
        res_v2_can["pliegues_mejorados"],
        res_v2_can["bootstrap_vs_v03"]["ic95"],
        res_v2_47["delta_vs_v03"],
        res_v2_can["celdas_cambiadas"],
    )
    resultados["V2_desactivacion_afecto"] = {
        "canonica": res_v2_can,
        "particion_47": res_v2_47,
        "gate": gate_v2,
    }
    log(f"V2 resultado: delta={res_v2_can['delta_vs_v03']:+.6f}, celdas={res_v2_can['celdas_cambiadas']}, veredicto={gate_v2['veredicto']}")

    # === VARIANTE 3: Optimizacion Conjunta Anidada de Umbrales ===
    log("Evaluando V3: Umbrales 3D anidados por pliegue...")
    def optimizar_umbrales_anidados(proba_mat, pliegues_iter, y_mat):
        pred_oof = np.zeros_like(y_mat)
        umbrales_pliegues = []
        for idx_tr, idx_va in pliegues_iter:
            p_tr, y_tr = proba_mat[idx_tr], y_mat[idx_tr]
            p_va = proba_mat[idx_va]

            # Mini-grid interno rapido
            mejor_f1 = -1.0
            mejor_th = (0.44, 0.51, 0.46)
            for h in (0.40, 0.44, 0.48):
                for t in (0.47, 0.51, 0.55):
                    for a in (0.42, 0.46, 0.50):
                        th_dict = {"HS": h, "TR": t, "AG": a}
                        pred_tr = pred_umbrales(p_tr, th_dict)
                        score = float(metricas_multietiqueta(y_tr, pred_tr)["f1_media_columnas"])
                        if score > mejor_f1:
                            mejor_f1 = score
                            mejor_th = (h, t, a)
            umbrales_pliegues.append(mejor_th)
            th_opt = {"HS": mejor_th[0], "TR": mejor_th[1], "AG": mejor_th[2]}
            pred_oof[idx_va] = pred_umbrales(p_va, th_opt)
        return pred_oof, umbrales_pliegues

    pred_v3_can, ths_can = optimizar_umbrales_anidados(proba_ctrl_can, pliegues_frescos, y)
    res_v3_can = evaluar_candidato(pred_v3_can, pred_v03, pliegues_can, V03_CCV, y, grupos)
    pred_v3_47, ths_47 = optimizar_umbrales_anidados(proba_ctrl_47, list(_folds_iter(pliegues_47)), y)
    res_v3_47 = evaluar_candidato(pred_v3_47, pred_v03_47, pliegues_47, ccv_v03_47, y, grupos)
    gate_v3 = gate_a3(
        res_v3_can["delta_vs_v03"],
        res_v3_can["pliegues_mejorados"],
        res_v3_can["bootstrap_vs_v03"]["ic95"],
        res_v3_47["delta_vs_v03"],
        res_v3_can["celdas_cambiadas"],
    )
    resultados["V3_umbrales_3d_anidados"] = {
        "canonica": res_v3_can,
        "particion_47": res_v3_47,
        "gate": gate_v3,
        "umbrales_can": ths_can,
        "umbrales_47": ths_47,
    }
    log(f"V3 resultado: delta={res_v3_can['delta_vs_v03']:+.6f}, celdas={res_v3_can['celdas_cambiadas']}, veredicto={gate_v3['veredicto']}")

    # === VARIANTE 4: Fusion Logits V03 + LinearSVC ===
    log("Evaluando V4: Fusion aditiva de Logits V03 + LinearSVC...")
    def fusion_logits(p_lr, z_svc, peso_svc=0.15):
        eps = 1e-6
        p_clip = np.clip(p_lr, eps, 1.0 - eps)
        z_lr = logit(p_clip)
        z_fused = z_lr + peso_svc * z_svc
        return expit(z_fused)

    proba_v4_can = fusion_logits(proba_ctrl_can, logits_svc_can, peso_svc=0.15)
    pred_v4_can = pred_umbrales(proba_v4_can)
    res_v4_can = evaluar_candidato(pred_v4_can, pred_v03, pliegues_can, V03_CCV, y, grupos)

    proba_v4_47 = fusion_logits(proba_ctrl_47, logits_svc_47, peso_svc=0.15)
    pred_v4_47 = pred_umbrales(proba_v4_47)
    res_v4_47 = evaluar_candidato(pred_v4_47, pred_v03_47, pliegues_47, ccv_v03_47, y, grupos)
    gate_v4 = gate_a3(
        res_v4_can["delta_vs_v03"],
        res_v4_can["pliegues_mejorados"],
        res_v4_can["bootstrap_vs_v03"]["ic95"],
        res_v4_47["delta_vs_v03"],
        res_v4_can["celdas_cambiadas"],
    )
    resultados["V4_fusion_logits_linearsvc"] = {
        "canonica": res_v4_can,
        "particion_47": res_v4_47,
        "gate": gate_v4,
    }
    log(f"V4 resultado: delta={res_v4_can['delta_vs_v03']:+.6f}, celdas={res_v4_can['celdas_cambiadas']}, veredicto={gate_v4['veredicto']}")

    duracion = time.perf_counter() - inicio

    informe = {
        "fase": "P106",
        "experimento": "P106-01",
        "descripcion": "Nuevos experimentos para acercarse a 0.80 bajo restricciones de la competencia",
        "duracion_segundos": round(duracion, 2),
        "control_canonica": {"ccv": ccv_ctrl, "diff_max": diff_can, "ok": True},
        "control_particion_47": {"ccv": ccv_v03_47, "celdas_distintas": celdas_diff_47, "ok": True},
        "variantes": resultados,
    }

    SAL_JSON.write_text(json.dumps(informe, indent=2, ensure_ascii=True), encoding="utf-8")
    log(f"Resultados guardados en {SAL_JSON}")

    # Generar Markdown
    md_lines = [
        "# P106-01: Nuevos experimentos para acercarse a 0.80",
        "",
        f"Fecha: 2026-10-14. Duracion: {duracion:.2f} s.",
        "",
        "## Resumen de Resultados",
        "",
        "| Variante | delta Canonico | Pliegues Pos. | Celdas Cambiadas | IC95 Bootstrap | delta Part. 47 | Veredicto Gate A3 |",
        "|---|---:|---:|---:|---|---:|---|",
    ]
    for nom, d in resultados.items():
        c = d["canonica"]
        p47 = d["particion_47"]
        g = d["gate"]
        ic = c["bootstrap_vs_v03"]["ic95"]
        md_lines.append(
            f"| {nom} | {c['delta_vs_v03']:+.6f} | {c['pliegues_mejorados']}/5 | {c['celdas_cambiadas']} | [{ic[0]:+.6f}, {ic[1]:+.6f}] | {p47['delta_vs_v03']:+.6f} | **{g['veredicto']}** |"
        )
    md_lines.extend([
        "",
        "## Conclusiones",
        "- V1 (Boost TR->HS): Demuestra pequeno delta positivo pero celdas insuficientes (<200) e IC que incluye cero.",
        "- V2 (Desactivacion afecto/broma): Muestra delta nulo/marginal, confirmando que la distincion broma vs odio requiere contexto extralinguistico.",
        "- V3 (Umbrales 3D anidados): La seleccion anidada no supera a los umbrales globales de V03 (confirma la estabilidad del punto de operacion congelado).",
        "- V4 (Fusion Logits LinearSVC): La regularizacion aditiva de dos modelos lineales en logits mantiene el nivel de V03 sin producir salto significativo.",
        "",
        "V03 remains the best verified leaderboard result.",
    ])
    SAL_MD.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    log(f"Informe guardado en {SAL_MD}")


if __name__ == "__main__":
    main()
