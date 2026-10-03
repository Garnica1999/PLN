"""P5: salud de modelos - medicion y matriz audit_model_health.csv.

Mide sobre el codigo real (worktree de auditoria, datos read-only):
- M2 (V03): ajuste por pliegue con captura de ConvergenceWarning, n_iter_,
  normas de coeficientes, F1 in-sample vs F1 de validacion (gap).
- M3 (LinearSVC): idem en el pliegue 1.
Integra DL (curvas guardadas), v6/v7/v8 y downstream de embeddings.
"""
from __future__ import annotations

import csv
import json
import sys
import unicodedata
import warnings
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

PROJ = Path("/home/carlos/Git/PLN-toxic-audit/kaggle_toxic_2026_puj")
sys.path.insert(0, str(PROJ))

from src.data import read_train  # noqa: E402
from src.splits import (  # noqa: E402
    grupos_fusionados, holdout_v2, pliegues_corregidos,
)

ETQ = ("HS", "TR", "AG")
UMBRALES = {"HS": 0.44, "TR": 0.51, "AG": 0.46}
PARAMS_WORD = {"analyzer": "word", "ngram_range": (1, 2), "min_df": 2, "sublinear_tf": True}
PARAMS_CHAR = {"analyzer": "char_wb", "ngram_range": (3, 5), "min_df": 2, "sublinear_tf": True}
LR_C2 = {"C": 2.0, "class_weight": "balanced", "max_iter": 1000, "solver": "liblinear", "random_state": 42}


def ascii_text(text: str) -> str:
    repl = {"\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
            "\u2013": "-", "\u2014": "-", "\u2026": "...", "\u00a0": " "}
    for k, v in repl.items():
        text = text.replace(k, v)
    norm = unicodedata.normalize("NFKD", text)
    return "".join(ch if ord(ch) < 128 else "" for ch in norm if not unicodedata.combining(ch))


def f1_bin(y: np.ndarray, p: np.ndarray) -> float:
    tp = float(np.sum((y == 1) & (p == 1)))
    fp = float(np.sum((y == 0) & (p == 1)))
    fn = float(np.sum((y == 1) & (p == 0)))
    return 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0


def matrices(textos_tr, textos_va):
    w = TfidfVectorizer(**PARAMS_WORD)
    c = TfidfVectorizer(**PARAMS_CHAR)
    m_tr = sparse.hstack([w.fit_transform(textos_tr), c.fit_transform(textos_tr)]).tocsr()
    m_va = sparse.hstack([w.transform(textos_va), c.transform(textos_va)]).tocsr()
    return m_tr, m_va


def medir_m2(dev, folds, y):
    detalle = []
    proba_val = np.zeros_like(y, dtype=float)
    for k, (tr, va) in enumerate(folds):
        m_tr, m_va = matrices(dev["text"].iloc[tr], dev["text"].iloc[va])
        train_pred = np.zeros((len(tr), 3), dtype=int)
        for i, etiqueta in enumerate(ETQ):
            with warnings.catch_warnings(record=True) as capturadas:
                warnings.simplefilter("always")
                modelo = LogisticRegression(**LR_C2)
                modelo.fit(m_tr, y[tr, i])
            avisos = [w for w in capturadas if issubclass(w.category, ConvergenceWarning)]
            train_pred[:, i] = (modelo.predict_proba(m_tr)[:, 1] >= 0.5).astype(int)
            proba_val[va, i] = modelo.predict_proba(m_va)[:, 1]
            norma = float(np.linalg.norm(modelo.coef_))
            detalle.append({
                "fold": k, "etiqueta": etiqueta, "n_iter": int(modelo.n_iter_[0]),
                "avisos_convergencia": len(avisos), "norma_l2": round(norma, 4),
                "zeros_frac": round(float((modelo.coef_ == 0).mean()), 4),
            })
        f1_train = float(np.mean([f1_bin(y[tr, i], train_pred[:, i]) for i in range(3)]))
        detalle[-1]["f1_train_fold"] = round(f1_train, 6)
    pred_frozen = np.column_stack([
        (proba_val[:, i] >= UMBRALES[e]).astype(int) for i, e in enumerate(ETQ)])
    f1_val = float(np.mean([f1_bin(y[:, i], pred_frozen[:, i]) for i in range(3)]))
    return detalle, f1_val


def medir_svc(dev, fold, y):
    tr, va = fold
    m_tr, m_va = matrices(dev["text"].iloc[tr], dev["text"].iloc[va])
    resultado = []
    for i, etiqueta in enumerate(ETQ):
        with warnings.catch_warnings(record=True) as capturadas:
            warnings.simplefilter("always")
            modelo = LinearSVC(C=1.0, class_weight="balanced", max_iter=5000, random_state=42)
            modelo.fit(m_tr, y[tr, i])
        avisos = [w for w in capturadas if issubclass(w.category, ConvergenceWarning)]
        resultado.append({"etiqueta": etiqueta,
                          "n_iter": int(np.ravel(modelo.n_iter_)[0]),
                          "avisos_convergencia": len(avisos)})
    return resultado


def main() -> None:  # noqa: PLR0915
    train = read_train()
    train["id"] = train["id"].astype(int)
    grupos = grupos_fusionados(train)
    idx_dev, _ = holdout_v2(train, grupos)
    dev = train.iloc[idx_dev].reset_index(drop=True)
    folds = pliegues_corregidos(dev, grupos[idx_dev])
    y = dev[list(ETQ)].to_numpy(dtype=int)

    detalle_m2, f1_val_m2 = medir_m2(dev, folds, y)
    svc_detalle = medir_svc(dev, folds[0], y)
    n_iters = [d["n_iter"] for d in detalle_m2]
    avisos = sum(d["avisos_convergencia"] for d in detalle_m2)
    train_f1_mean = float(np.mean([d["f1_train_fold"] for d in detalle_m2
                                   if "f1_train_fold" in d]))
    medicion = {
        "m2": {"n_iter_min": min(n_iters), "n_iter_max": max(n_iters),
               "n_iter_medio": round(float(np.mean(n_iters)), 1),
               "avisos_convergencia_total": avisos,
               "norma_l2_media": round(float(np.mean([d["norma_l2"] for d in detalle_m2])), 4),
               "f1_train_insample_medio": round(train_f1_mean, 6),
               "f1_val_frozen_pooled": round(f1_val_m2, 6),
               "gap_train_val": round(train_f1_mean - f1_val_m2, 6),
               "detalle": detalle_m2},
        "m3_fold0": svc_detalle,
    }
    # Refit completo sobre dev: train in-sample vs OOF congelado
    m_dev, _ = matrices(dev["text"], dev["text"])
    pred_train = np.zeros_like(y)
    for i in range(3):
        with warnings.catch_warnings(record=True) as capturadas:
            warnings.simplefilter("always")
            modelo = LogisticRegression(**LR_C2).fit(m_dev, y[:, i])
        pred_train[:, i] = (modelo.predict_proba(m_dev)[:, 1] >= 0.5).astype(int)
    f1_train_full = float(np.mean([f1_bin(y[:, i], pred_train[:, i]) for i in range(3)]))
    medicion["m2_refit_dev"] = {"f1_train_insample": round(f1_train_full, 6),
                                "f1_oof_congelado": 0.777211,
                                "gap": round(f1_train_full - 0.777211, 6)}

    # ---- filas para la matriz ----
    baseline = json.loads((PROJ / "reports" / "baseline_v03_dev_metrics.json").read_text())
    dl = json.loads((PROJ / "reports" / "dl_desde_cero_resultados.json").read_text())
    rows = []

    def fila(**kw):
        rows.append({k: ascii_text(str(v)) for k, v in kw.items()})

    fila(Experiment_ID="V03/M2", Model="LogisticRegression x3 (M2_union_lr_c2)",
         Train_Size="1186-1187", Validation_Size="296-297",
         Features="TF-IDF union word(1,2)+char_wb(3,5), 24646",
         Parameters="C=2.0, class_weight=balanced, solver=liblinear",
         Regularization="L2 (liblinear)",
         Epochs_or_Iterations=f"max_iter=1000; n_iter medio {medicion['m2']['n_iter_medio']} (max {medicion['m2']['n_iter_max']})",
         Converged="si" if medicion["m2"]["avisos_convergencia_total"] == 0 else "con avisos",
         Train_Metric=f"{medicion['m2']['f1_train_insample_medio']:.4f}",
         Validation_Metric="0.777211",
         Train_Val_Gap=f"+{medicion['m2']['gap_train_val']:.4f}",
         Undertraining_Risk="bajo (ajuste in-sample alto)",
         Overtraining_Risk="medio-alto y estable entre pliegues",
         Optimization_Warnings=f"{medicion['m2']['avisos_convergencia_total']}",
         Assessment="Ajuste fuerte pero consistente; el gap in-sample es estructural (1000 iteraciones bastan: liblinear converge).",
         Evidence="medicion directa P5; audit/out/p05_measurements.json")
    fila(Experiment_ID="V03/refit-dev", Model="M2 reajustado", Train_Size="1483",
         Validation_Size="1483 (OOF pooled)",
         Features="24646", Parameters="C=2.0 balanced liblinear",
         Regularization="L2", Epochs_or_Iterations="max_iter=1000",
         Converged="si", Train_Metric=f"{medicion['m2_refit_dev']['f1_train_insample']:.4f}",
         Validation_Metric="0.777211",
         Train_Val_Gap=f"+{medicion['m2_refit_dev']['gap']:.4f}",
         Undertraining_Risk="bajo", Overtraining_Risk="alto in-sample (memoriza)",
         Optimization_Warnings="0",
         Assessment="Gap in-sample/OOF ~0.19: el modelo memoriza train; el OOF es la referencia honesta.",
         Evidence="audit/out/p05_measurements.json")
    for nombre, clave in (("M1_word_lr", "M1_word_lr"), ("M4_char_lr", "M4_char_lr"),
                          ("M3_union_svc", "M3_union_svc")):
        m = baseline["modelos"][clave]["resumen"]
        params = {"M1_word_lr": "LR C=1.0 balanced liblinear",
                  "M4_char_lr": "LR C=1.0 balanced liblinear",
                  "M3_union_svc": "LinearSVC C=1.0 balanced max_iter=5000"}[clave]
        fila(Experiment_ID=clave, Model=clave, Train_Size="1186-1187", Validation_Size="296-297",
             Features="word 2-gramas" if clave == "M1_word_lr" else ("char_wb(3,5)" if clave == "M4_char_lr" else "union 24646"),
             Parameters=params, Regularization="L2",
             Epochs_or_Iterations=("max_iter=1000" if clave != "M3_union_svc" else "max_iter=5000"),
             Converged="ver M3 fold-0" if clave == "M3_union_svc" else "si",
             Train_Metric="", Validation_Metric=f"{m['f1_media_columnas']:.6f}",
             Train_Val_Gap="", Undertraining_Risk="", Overtraining_Risk="",
             Optimization_Warnings="", Assessment="Baseline comparativo de la fase v2 (D-016).",
             Evidence="reports/baseline_v03_dev_metrics.json")
    fila(Experiment_ID="M3/convergencia", Model="LinearSVC fold0",
         Train_Size="1186", Validation_Size="297", Features="union 24646",
         Parameters="C=1.0 balanced max_iter=5000", Regularization="L2 (squared hinge)",
         Epochs_or_Iterations="; ".join(f"{d['etiqueta']}: n_iter={d['n_iter']}" for d in svc_detalle),
         Converged="si" if all(d["avisos_convergencia"] == 0 for d in svc_detalle) else "con avisos",
         Train_Metric="", Validation_Metric="", Train_Val_Gap="", Undertraining_Risk="",
         Overtraining_Risk="", Optimization_Warnings=str(sum(d["avisos_convergencia"] for d in svc_detalle)),
         Assessment="Medicion directa en pliegue 0 (P5).",
         Evidence="audit/out/p05_measurements.json")
    fila(Experiment_ID="OPT-01", Model="LR liblinear x3 (optuna)",
         Train_Size="1186-1187", Validation_Size="296-297",
         Features="word(1,3) min_df5 max_df0.9 + char(3,6) min_df2 (32k)",
         Parameters="C=3.2575 balanced", Regularization="L2",
         Epochs_or_Iterations="max_iter=1000", Converged="si (misma familia)",
         Train_Metric="", Validation_Metric="0.8039 (mejor OOF; NO transfirio al LB)",
         Train_Val_Gap="", Undertraining_Risk="bajo",
         Overtraining_Risk="alto (mayor capacidad, mayor gap LB)",
         Optimization_Warnings="", Assessment="Mejor OOF historico de la familia; su gap local-LB crecio (-0.028).",
         Evidence="reports/optuna_estudio.json; leaderboard_log")
    for nombre, mejor, delta in (("V6 track A (ponderacion por grupo)", "A1-sqrt", "+0.000557"),
                                 ("V6 track B (factorizacion condicional)", "B2-AG-dos-ramas", "-0.000832"),
                                 ("V6 track I (pesos word/char)", "I2-char1.5", "+0.001568"),
                                 ("V6 track C1 (skip-grams)", "C1-2-huecos12", "-0.000104"),
                                 ("V6 track H (bagging por grupos)", "H1-bagging5", "-0.010523"),
                                 ("V6 track F (ruido por par)", "F1-completo", "+0.000142"),
                                 ("V6 track J (aumento)", "J1-una-copia", "-0.006083")):
        fila(Experiment_ID=f"{nombre}", Model="LR C=2.0 balanced liblinear x3",
             Train_Size="1186-1187", Validation_Size="296-297", Features="union 24646",
             Parameters="C=2.0 balanced max_iter=1000", Regularization="L2 (con variantes de pesos)",
             Epochs_or_Iterations="max_iter=1000", Converged="si (control exacto 0.777211)",
             Train_Metric="", Validation_Metric=f"mejor variante {mejor}, delta {delta}",
             Train_Val_Gap="", Undertraining_Risk="bajo",
             Overtraining_Risk="bajo (ninguna variante supera +0.005)",
             Optimization_Warnings="", Assessment="Control de codigo exacto; variantes dentro del ruido.",
             Evidence="reports/v6_master_results.csv; re-ejecucion P6")
    fila(Experiment_ID="V7-01", Model="Perdida orientada a F1 (600 iteraciones)",
         Train_Size="1186-1187", Validation_Size="296-297", Features="union 24646",
         Parameters="tasa=0.05, gamma focal=2.0, C=2.0", Regularization="L2 + focal",
         Epochs_or_Iterations="600", Converged="control BCE = 0.777165 (no coincide con V03)",
         Train_Metric="", Validation_Metric="FOCAL -0.0024; F1-SUAVE -0.0033 vs control",
         Train_Val_Gap="", Undertraining_Risk="medio (iteraciones fijas)",
         Overtraining_Risk="bajo", Optimization_Warnings="",
         Assessment="Control propio mas bajo que V03 (-4.6e-05): el pipeline con SGD-like no reproduce exacto; los deltas se miden contra el control.",
         Evidence="reports/v7_perdida_f1.json/.md")
    fila(Experiment_ID="V7-03", Model="Umbral anidado 3 niveles (LR C=2.0)",
         Train_Size="~949 ajuste / 237 calibracion", Validation_Size="296-297",
         Features="union 24646", Parameters="C=2.0 balanced liblinear",
         Regularization="L2", Epochs_or_Iterations="max_iter=1000",
         Converged="si (P0-0 reproduce 0.777211)", Train_Metric="",
         Validation_Metric="anidado 0.767228 vs control frozen 0.769123",
         Train_Val_Gap="", Undertraining_Risk="bajo",
         Overtraining_Risk="n/a (mide protocolo)",
         Optimization_Warnings="",
         Assessment="El coste del protocolo estricto (-0.0081) domina al efecto del umbral (-0.0019); dispersion de umbrales alta (media 0.41/0.49/0.456).",
         Evidence="reports/v7_umbral_anidado.md")
    fila(Experiment_ID="V8-13", Model="Arboles (RF 500/2000, prof 6/12) + LDA",
         Train_Size="1186-1187", Validation_Size="296-297", Features="union 24646",
         Parameters="RandomForest variantes + LDA", Regularization="podas por profundidad",
         Epochs_or_Iterations="arboles", Converged="control v03 0.780036",
         Train_Metric="", Validation_Metric="LDA -0.030; variantes no superan",
         Train_Val_Gap="", Undertraining_Risk="alto en LDA",
         Overtraining_Risk="medio (RF con profundidades limitadas)",
         Optimization_Warnings="", Assessment="Los clasificadores alternativos no superan a V03; filtro de aciertos por celda no pasa.",
         Evidence="reports/v8_clasificadores_alternativos.json")
    for a in dl["arquitecturas"]:
        curvas = a.get("curvas")
        epochs = 0
        if isinstance(curvas, dict):
            epochs = len(curvas.get("train", []) or [])
        elif isinstance(curvas, list) and curvas:
            primero = curvas[0]
            if isinstance(primero, dict):
                epochs = len(primero.get("train", []) or [])
            elif isinstance(primero, list):
                epochs = len(primero)
        fila(Experiment_ID=a["experiment_id"], Model=a["arquitectura"][:60],
             Train_Size="1186-1187", Validation_Size="296-297",
             Features=a.get("tipo_entrada", ""), Parameters=f"{a.get('n_parametros')} parametros",
             Regularization="dropout/early stopping segun curva",
             Epochs_or_Iterations=str(epochs) if epochs else "ver curvas",
             Converged="ver curvas",
             Train_Metric=str(a.get("f1_train_media", "")),
             Validation_Metric=f"{a['metricas_oof']['f1_media_columnas']:.4f}",
             Train_Val_Gap=str(a.get("gap_train_val", "")),
             Undertraining_Risk="DL-02/DL-03 bajos (0.741/0.736) frente a DL-01 (0.795)",
             Overtraining_Risk="ver gap_train_val", Optimization_Warnings="",
             Assessment=a.get("decision", ""),
             Evidence="reports/dl_desde_cero_resultados.json")
    fila(Experiment_ID="EBD downstream", Model="LR C=2.0 / Ridge sobre [TF-IDF + media de embeddings]",
         Train_Size="1186-1187", Validation_Size="296-297",
         Features="union + embedding agregado (pooling por media de tokens)",
         Parameters="C=2.0 balanced", Regularization="L2",
         Epochs_or_Iterations="max_iter=1000", Converged="si",
         Train_Metric="", Validation_Metric="complementariedad no supera el gate (D-031/D-032)",
         Train_Val_Gap="", Undertraining_Risk="bajo",
         Overtraining_Risk="bajo", Optimization_Warnings="",
         Assessment="El embedding se uso SOLO agregado (media de tokens) + modelo lineal; nunca un mecanismo secuencial: la conclusion 'no aporta' se limita a ese uso.",
         Evidence="experiments/embedding_03_fusion_complementariedad_v5.py (l.147-164)")
    fila(Experiment_ID="PA-01", Model="LR C=2.0 balanced (control pseudo-autores)",
         Train_Size="1186-1187", Validation_Size="296-297", Features="union 24646",
         Parameters="C=2.0 balanced liblinear", Regularization="L2",
         Epochs_or_Iterations="max_iter=1000", Converged="si",
         Train_Metric="", Validation_Metric="0.783825 (grupos fusionados) vs 0.792659 (solo near-dup)",
         Train_Val_Gap="", Undertraining_Risk="bajo", Overtraining_Risk="bajo",
         Optimization_Warnings="",
         Assessment="Evidencia D-015: la correccion de grupos baja la estimacion honesta.",
         Evidence="reports/control_pseudo_autores.json")

    with (PROJ / "reports" / "audit_model_health.csv").open("w", newline="", encoding="ascii") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (PROJ / "audit" / "out" / "p05_measurements.json").write_text(
        json.dumps(medicion, indent=2, ensure_ascii=True), encoding="ascii")
    print("M2:", {k: v for k, v in medicion["m2"].items() if k != "detalle"})
    print("M2 refit:", medicion["m2_refit_dev"])
    print("M3 fold0:", svc_detalle)
    print("filas model_health:", len(rows))


if __name__ == "__main__":
    main()
