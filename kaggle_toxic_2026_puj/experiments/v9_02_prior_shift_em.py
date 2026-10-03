"""V9-02 (H4): correccion de prior shift por EM sobre las probabilidades de V03.

Pregunta
--------
El test tiene ~7 % menos positivos que train (sonda E00). Si V03 usa los
umbrales congelados, ¿se puede recuperar parte del gap local-LB corrigiendo el
prior con EM (Saerens) sin usar etiquetas de test?

Metodo
------
1. Se reajusta el pipeline congelado de V03 (M2 union + LR C=2 x3) con TODO el
   train y se predicen probabilidades en test (receta de submission 04).
2. Calibracion Platt por etiqueta (1 feature sobre el logit) ajustada en el OOF
   del desarrollo; en la simulacion se usa la version anidada (4 pliegues -> 1).
3. EM de Saerens sobre las probabilidades calibradas de test para estimar la
   prevalencia por etiqueta (sin etiquetas).
4. Simulacion del mecanismo: rejilla de shifts conocidos sobre el OOF del
   desarrollo (reponderacion de positivos) comparando umbrales congelados
   (sobre probabilidad cruda) contra umbrales re-optimizados (sobre calibrada).
5. Submission sonda v17 con los umbrales corregidos por el shift EM (no se envia).

Salidas: reports/v9_02_prior_shift_em.json/.md/.csv,
submissions/toxic_submission_v17_prior_shift_em.csv y fila V9-02.

Ejecucion:
    /home/carlos/venv/bin/python experiments/v9_02_prior_shift_em.py
"""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src import data as data_module
from src.splits import ETIQUETAS, grupos_fusionados, holdout_v2, pliegues_corregidos

REF_PRECISION = 0.7772114525165721
UMBRALES_V03 = {"HS": 0.44, "TR": 0.51, "AG": 0.46}
REJILLA = np.round(np.arange(0.05, 0.951, 0.01), 2)
REPORTE_JSON = PROJECT_ROOT / "reports" / "v9_02_prior_shift_em.json"
REPORTE_MD = PROJECT_ROOT / "reports" / "v9_02_prior_shift_em.md"
REPORTE_CSV = PROJECT_ROOT / "reports" / "v9_02_prior_shift_em.csv"
SUBMISSION = PROJECT_ROOT / "submissions" / "toxic_submission_v17_prior_shift_em.csv"
EXPERIMENTS_CSV = PROJECT_ROOT / "experiments.csv"


def log(mensaje: str) -> None:
    """Imprime un mensaje con prefijo de fase."""
    print(f"[V9-02] {mensaje}", flush=True)


def cargar_modulo14():
    """Carga experiments/14 para reutilizar el pipeline congelado de V03."""
    ruta = PROJECT_ROOT / "experiments" / "14_oof_corregido_v2.py"
    spec = importlib.util.spec_from_file_location("mod14b", ruta)
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


def logit(p: np.ndarray) -> np.ndarray:
    """Logit con recorte para evitar infinitos."""
    pc = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(pc / (1 - pc))


def f1_ponderada_columna(y: np.ndarray, pred: np.ndarray, w: np.ndarray) -> float:
    """F1 binaria de una columna con pesos de fila."""
    yb = np.asarray(y).reshape(-1)
    pb = np.asarray(pred).reshape(-1)
    tp = float(np.sum(w * ((yb == 1) & (pb == 1))))
    fp = float(np.sum(w * ((yb == 0) & (pb == 1))))
    fn = float(np.sum(w * ((yb == 1) & (pb == 0))))
    den = 2 * tp + fp + fn
    return 2 * tp / den if den > 0 else 0.0


def f1_media(y: np.ndarray, pred: np.ndarray) -> float:
    """F1 media column-wise sin pesos."""
    return float(np.mean([f1_ponderada_columna(y[:, i], pred[:, i], np.ones(len(y)))
                          for i in range(3)]))


def optimizar_umbral_columna(y_col: np.ndarray, proba_col: np.ndarray,
                             w: np.ndarray) -> tuple[float, float]:
    """Umbral que maximiza la F1 ponderada de una columna."""
    mejor, mejor_u = -1.0, 0.5
    for u in REJILLA:
        valor = f1_ponderada_columna(y_col, (proba_col >= u).astype(int), w)
        if valor > mejor + 1e-12:
            mejor, mejor_u = valor, float(u)
    return mejor_u, mejor


def em_saerens(proba: np.ndarray, prior_inicial: float) -> tuple[float, int]:
    """EM de Saerens-Latinne-Decaestecker sobre probabilidades calibradas."""
    pi = prior_inicial
    iteraciones = 0
    for _ in range(500):
        iteraciones += 1
        w_pos = pi / prior_inicial
        w_neg = (1 - pi) / (1 - prior_inicial)
        num = w_pos * proba
        den = num + w_neg * (1 - proba)
        posteriores = num / np.maximum(den, 1e-12)
        nuevo = float(np.mean(posteriores))
        if abs(nuevo - pi) < 1e-8:
            pi = nuevo
            break
        pi = nuevo
    return pi, iteraciones


def calibrar(proba: np.ndarray, y: np.ndarray) -> LogisticRegression:
    """Calibracion Platt sobre el logit de la probabilidad."""
    modelo = LogisticRegression(C=1e3, solver="lbfgs", max_iter=2000)
    modelo.fit(logit(proba).reshape(-1, 1), y)
    return modelo


def pesos_shift(prevalencia_train: float, prevalencia_objetivo: float) -> float:
    """Peso de las filas positivas para simular la prevalencia objetivo."""
    odds_train = prevalencia_train / (1 - prevalencia_train)
    odds_obj = prevalencia_objetivo / (1 - prevalencia_objetivo)
    return odds_obj / odds_train


def main() -> None:  # noqa: PLR0915
    inicio = time.perf_counter()
    modulo14 = cargar_modulo14()
    train = data_module.read_train()
    test = data_module.read_test()
    grupos = grupos_fusionados(train)
    indices_dev, _ = holdout_v2(train, grupos)
    dev = train.iloc[indices_dev].reset_index(drop=True)
    pliegues = pliegues_corregidos(dev, grupos[indices_dev])
    y_dev = dev[list(ETIQUETAS)].to_numpy()
    oof = np.load(PROJECT_ROOT / "features" / "oof_v03_final.npz", allow_pickle=True)
    proba_dev = oof["proba"]
    prevalencia_train = y_dev.mean(axis=0)

    # --- Calibracion ---
    calibrada_dev = np.zeros_like(proba_dev)
    for etiqueta_idx in range(3):
        for _k, (_tr, va) in enumerate(pliegues):
            mascara_ajuste = np.ones(len(dev), dtype=bool)
            mascara_ajuste[va] = False
            modelo = calibrar(proba_dev[mascara_ajuste, etiqueta_idx],
                              y_dev[mascara_ajuste, etiqueta_idx])
            calibrada_dev[va, etiqueta_idx] = modelo.predict_proba(
                logit(proba_dev[va, etiqueta_idx]).reshape(-1, 1))[:, 1]
    calibradores_finales = [calibrar(proba_dev[:, i], y_dev[:, i]) for i in range(3)]

    # --- Probabilidades de test con el pipeline congelado (refit full train) ---
    log("reajuste de V03 con todo el train y prediccion de test")
    matriz_train, matriz_test = modulo14.construir_matrices(
        train["text"], test["text"], "union")
    proba_test = np.zeros((len(test), 3))
    for i in range(3):
        modelo = LogisticRegression(**modulo14.MODELOS["M2_union_lr_c2"]["params"])
        modelo.fit(matriz_train, train[ETIQUETAS[i]])
        proba_test[:, i] = modelo.predict_proba(matriz_test)[:, 1]
    calibrada_test = np.column_stack([
        calibradores_finales[i].predict_proba(
            logit(proba_test[:, i]).reshape(-1, 1))[:, 1] for i in range(3)])

    # --- EM de prevalencia por etiqueta ---
    em_resultados = {}
    for i, etiqueta in enumerate(ETIQUETAS):
        pi_em, iteraciones = em_saerens(calibrada_test[:, i], float(prevalencia_train[i]))
        em_resultados[etiqueta] = {
            "prevalencia_train": round(float(prevalencia_train[i]), 6),
            "prevalencia_em_test": round(pi_em, 6),
            "factor_vs_train": round(pi_em / float(prevalencia_train[i]), 4),
            "iteraciones": iteraciones,
        }
        log(f"  {etiqueta}: train {prevalencia_train[i]:.4f} -> EM test {pi_em:.4f} "
            f"(factor {pi_em / prevalencia_train[i]:.3f})")

    # --- Simulacion del mecanismo con shifts conocidos ---
    filas_sim = []
    detalle_sim = []
    for factor in (1.0, 0.95, 0.93, 0.90, 0.85, 0.80, 0.75):
        fila = {"factor_simulado": factor}
        for i, etiqueta in enumerate(ETIQUETAS):
            pi_train = float(prevalencia_train[i])
            w_pos = pesos_shift(pi_train, pi_train * factor)
            pesos_i = np.where(y_dev[:, i] == 1, w_pos, 1.0)
            pred_fijo = (proba_dev[:, i] >= UMBRALES_V03[etiqueta]).astype(int)
            f1_fijo = f1_ponderada_columna(y_dev[:, i], pred_fijo, pesos_i)
            umbral_corr, f1_corr = optimizar_umbral_columna(
                y_dev[:, i], calibrada_dev[:, i], pesos_i)
            fila[etiqueta] = round(f1_corr - f1_fijo, 6)
            if factor == 0.93:
                detalle_sim.append({"etiqueta": etiqueta, "w_pos": round(w_pos, 4),
                                    "f1_fijo": round(f1_fijo, 6),
                                    "f1_corregido": round(f1_corr, 6),
                                    "umbral_fijo": UMBRALES_V03[etiqueta],
                                    "umbral_corregido": umbral_corr})
        filas_sim.append(fila)

    # --- Decision final: umbrales corregidos por el shift estimado por EM ---
    umbrales_em = {}
    ganancia_em = {}
    for i, etiqueta in enumerate(ETIQUETAS):
        pi_train = float(prevalencia_train[i])
        pi_em = em_resultados[etiqueta]["prevalencia_em_test"]
        w_pos = pesos_shift(pi_train, pi_em)
        pesos_i = np.where(y_dev[:, i] == 1, w_pos, 1.0)
        umbral, _ = optimizar_umbral_columna(y_dev[:, i], calibrada_dev[:, i], pesos_i)
        umbrales_em[etiqueta] = umbral
        pred_fijo = (proba_dev[:, i] >= UMBRALES_V03[etiqueta]).astype(int)
        pred_corr = (calibrada_dev[:, i] >= umbral).astype(int)
        ganancia_em[etiqueta] = round(
            f1_ponderada_columna(y_dev[:, i], pred_corr, pesos_i)
            - f1_ponderada_columna(y_dev[:, i], pred_fijo, pesos_i), 6)

    pred_fijo_dev = np.column_stack([
        (proba_dev[:, i] >= UMBRALES_V03[e]).astype(int) for i, e in enumerate(ETIQUETAS)])
    pred_corr_dev = np.column_stack([
        (calibrada_dev[:, i] >= umbrales_em[e]).astype(int) for i, e in enumerate(ETIQUETAS)])
    f1_congelado_dev = f1_media(y_dev, pred_fijo_dev)
    f1_corregido_dev = f1_media(y_dev, pred_corr_dev)

    # --- Validacion anidada del mecanismo (leave-one-fold-out) ---
    validacion_anidada = {}
    for i, etiqueta in enumerate(ETIQUETAS):
        pi_train = float(prevalencia_train[i])
        pi_em = em_resultados[etiqueta]["prevalencia_em_test"]
        w_pos = pesos_shift(pi_train, pi_em)
        pesos_i = np.where(y_dev[:, i] == 1, w_pos, 1.0)
        pred_corr_nested = np.zeros(len(dev), dtype=int)
        for _k, (_tr, va) in enumerate(pliegues):
            fuera = np.ones(len(dev), dtype=bool)
            fuera[va] = False
            umbral, _ = optimizar_umbral_columna(
                y_dev[fuera, i], calibrada_dev[fuera, i], pesos_i[fuera])
            pred_corr_nested[va] = (calibrada_dev[va, i] >= umbral).astype(int)
        pred_fijo = (proba_dev[:, i] >= UMBRALES_V03[etiqueta]).astype(int)
        f1_fijo_pond = f1_ponderada_columna(y_dev[:, i], pred_fijo, pesos_i)
        f1_nested_pond = f1_ponderada_columna(y_dev[:, i], pred_corr_nested, pesos_i)
        validacion_anidada[etiqueta] = {
            "f1_fijo_ponderado": round(f1_fijo_pond, 6),
            "f1_nested_ponderado": round(f1_nested_pond, 6),
            "ganancia_nested": round(f1_nested_pond - f1_fijo_pond, 6),
        }
    ganancia_nested_media = float(np.mean(
        [validacion_anidada[e]["ganancia_nested"] for e in ETIQUETAS]))
    veredicto_mecanismo = (
        "GANANCIA NESTED POSITIVA" if ganancia_nested_media >= 0.002 else
        ("GANANCIA NESTED PEQUENA" if ganancia_nested_media > 0 else
         "SIN GANANCIA NESTED (el efecto lax era seleccion)"))

    # --- Submission sonda ---
    pred_test = np.column_stack([
        (calibrada_test[:, i] >= umbrales_em[e]).astype(int)
        for i, e in enumerate(ETIQUETAS)])
    v03 = pd.read_csv(PROJECT_ROOT / "submissions" / "toxic_submission_v03_lr_union_thr.csv")
    celdas_cambian = int((pred_test != v03[["HS", "TR", "AG"]].to_numpy()).sum())
    marco = pd.DataFrame({e: pred_test[:, i] for i, e in enumerate(ETIQUETAS)})
    marco.insert(0, "id", test["id"].to_numpy())
    SUBMISSION.parent.mkdir(parents=True, exist_ok=True)
    marco.to_csv(SUBMISSION, index=False, lineterminator="\n")
    integridad = {
        "filas": len(marco),
        "columnas": list(marco.columns),
        "ids_iguales": bool((marco["id"].astype(int) == v03["id"].astype(int)).all()),
        "binario": bool(set(marco[list(ETIQUETAS)].to_numpy().ravel()) <= {0, 1}),
    }

    reporte = {
        "experimento": "V9-02 (H4)",
        "fecha": "2026-09-26",
        "referencia_v03": REF_PRECISION,
        "em_por_etiqueta": em_resultados,
        "umbrales_congelados": UMBRALES_V03,
        "umbrales_corregidos_em": umbrales_em,
        "ganancia_simulada_em": ganancia_em,
        "simulacion_shifts": filas_sim,
        "detalle_simulacion_0_93": detalle_sim,
        "f1_dev_congelado": round(f1_congelado_dev, 6),
        "f1_dev_con_umbrales_em": round(f1_corregido_dev, 6),
        "validacion_anidada": validacion_anidada,
        "ganancia_nested_media": round(ganancia_nested_media, 6),
        "veredicto_mecanismo": veredicto_mecanismo,
        "nota": ("El F1 dev con umbrales EM no es una mejora local comparable: es la "
                 "decision adaptada al shift estimado; la validacion es la rejilla de "
                 "simulacion (ganancia bajo shift conocido)."),
        "submission_sonda": str(SUBMISSION.relative_to(PROJECT_ROOT)),
        "celdas_que_cambian_vs_v03": celdas_cambian,
        "integridad_submission": integridad,
        "duracion_segundos": round(time.perf_counter() - inicio, 2),
    }
    REPORTE_JSON.write_text(json.dumps(reporte, indent=2, ensure_ascii=False), encoding="utf-8")
    with REPORTE_CSV.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(filas_sim[0].keys()))
        writer.writeheader()
        writer.writerows(filas_sim)
    lineas = [
        "# V9-02 (H4): correccion de prior shift por EM",
        "",
        "## Prevalencias (train vs EM en test, sin etiquetas)",
        "",
        "| etiqueta | train | EM test | factor |",
        "|---|---|---|---|",
    ]
    for etiqueta, datos in em_resultados.items():
        lineas.append(f"| {etiqueta} | {datos['prevalencia_train']:.4f} | "
                      f"{datos['prevalencia_em_test']:.4f} | {datos['factor_vs_train']:.3f} |")
    lineas += [
        "",
        "## Ganancia simulada (F1 ponderada: umbrales re-optimizados - congelados)",
        "",
        "| factor de shift | HS | TR | AG |",
        "|---|---|---|---|",
    ]
    for fila in filas_sim:
        lineas.append(f"| {fila['factor_simulado']} | {fila['HS']:+.4f} | "
                      f"{fila['TR']:+.4f} | {fila['AG']:+.4f} |")
    lineas += [
        "",
        f"- Umbrales congelados: {UMBRALES_V03}",
        f"- Umbrales corregidos (EM): { {e: round(v, 3) for e, v in umbrales_em.items()} }",
        f"- Ganancia simulada al shift EM (lax): {ganancia_em}",
        f"- Validacion anidada (leave-one-fold-out): ganancia media "
        f"{ganancia_nested_media:+.6f} -> {veredicto_mecanismo}",
        f"- Detalle anidado: {validacion_anidada}",
        f"- Submission sonda: `{reporte['submission_sonda']}` "
        f"({celdas_cambian} celdas cambian frente a V03).",
        "",
        "Lectura: la correccion aporta una ganancia pequena y creciente con el shift",
        "simulado; la sonda queda pre-registrada para decision de Carlos.",
        f"\nFecha: 2026-09-26. Duracion: {reporte['duracion_segundos']} s.",
    ]
    REPORTE_MD.write_text("\n".join(lineas), encoding="utf-8")

    fila_reg = {
        "experiment_id": "V9-02",
        "competition": "KAGGLE_TOXIC_2026_PUJ",
        "question": "Recupera parte del gap local-LB una correccion de prior shift por EM sobre V03?",
        "hypothesis": "El test tiene ~7% menos positivos; reestimar la prevalencia por EM y ajustar umbrales mejora la F1 esperada",
        "preprocessing": "texto crudo (pipeline congelado de V03)",
        "feature_groups": "WORD_NGRAM+CHAR_NGRAM",
        "representation": "union TF-IDF word(1,2)+char_wb(3,5), min_df=2, sublinear_tf",
        "model": "LogisticRegression x3 (C=2.0 balanced) + calibracion Platt + EM de Saerens",
        "hyperparameters": json.dumps({"C": 2.0, "calibracion": "Platt por etiqueta",
                                       "umbrales_em": {e: round(v, 3) for e, v in umbrales_em.items()}},
                                      ensure_ascii=False),
        "validation": "simulacion por reponderacion del OOF (rejilla de shifts conocidos); calibracion anidada",
        "f1": round(f1_congelado_dev, 4),
        "macro_f1": "", "weighted_f1": "", "precision": "", "recall": "",
        "training_time": reporte["duracion_segundos"],
        "dimension": "",
        "result": json.dumps({"em_por_etiqueta": {e: em_resultados[e]["prevalencia_em_test"]
                                                  for e in ETIQUETAS},
                              "ganancia_sim_em": ganancia_em,
                              "celdas_que_cambian": celdas_cambian}, ensure_ascii=False),
        "interpretation": "Capa de decision nueva; la mejora local esperada depende del shift real y es pequena",
        "decision": "Sonda v17 generada y pre-registrada; no adoptada ni enviada",
        "date": "2026-09-26",
        "script": "experiments/v9_02_prior_shift_em.py",
        "notes": "Artefactos: reports/v9_02_prior_shift_em.json/.md/.csv; submissions/toxic_submission_v17_prior_shift_em.csv",
    }
    existente = pd.read_csv(EXPERIMENTS_CSV) if EXPERIMENTS_CSV.exists() else pd.DataFrame()
    if "V9-02" in existente.get("experiment_id", pd.Series(dtype=str)).tolist():
        existente = existente[existente["experiment_id"] != "V9-02"]
        existente.to_csv(EXPERIMENTS_CSV, index=False)
    pd.DataFrame([fila_reg]).to_csv(EXPERIMENTS_CSV, mode="a",
                                    header=not EXPERIMENTS_CSV.exists(), index=False)
    log(f"EM: {[em_resultados[e]['prevalencia_em_test'] for e in ETIQUETAS]} | "
        f"umbrales {umbrales_em} | celdas que cambian {celdas_cambian}")


if __name__ == "__main__":
    main()
