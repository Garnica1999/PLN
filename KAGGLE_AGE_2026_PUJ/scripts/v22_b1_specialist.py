"""V22 · B1_SPECIALIST — Los dos candidatos de B1 que faltan por evaluar.

El script `v22_ordinal.py` evalua el modelo ordinal **compartido** (paso 5). Este evalua los dos
candidatos de **sustitucion de B1** (pasos 3 y 4), que son los que faltan:

| Candidato | Mecanismo | Por que |
|---|---|---|
| **LinearSVC C 0,2** | control de solver/loss, no geometria nueva | comprueba si el problema de B1 es del estimador lineal |
| **LinearSVC C 1,0** | una variante cercana de regularizacion | sin grid |
| **boosting especializado** | LightGBM binario, pequeno y regularizado | la hipotesis principal: las interacciones no lineales ayudan a B1 |

**V20 queda congelado** y es el control exacto: el script se aborta si su OOF no reproduce
**0,6097782**. Las otras cuatro fronteras y la reconstruccion son **exactamente** las de V20; la unica
variable es el modelo que genera B1.

Entregables: ``experiments/v22_b1_specialist.json``, ``reports/v22_b1_specialist.md``,
``reports/v22_b1_specialist.csv``.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from v19_b1_boosting import HIERBA, matriz_279  # noqa: E402
from v13_p1b_fusion_tokens import bootstrap_diferencia  # noqa: E402

from src import config as cfg  # noqa: E402
from src import cv as cvmod  # noqa: E402
from src import data as datos  # noqa: E402
from src.config import set_seed  # noqa: E402
from src.verificacion_numerica import autotest_numerico  # noqa: E402

INFORME = cfg.REPORTS / "v22_b1_specialist.md"
TABLA = cfg.REPORTS / "v22_b1_specialist.csv"
RESULTADO = cfg.EXPERIMENTS / "v22_b1_specialist.json"
ACUMULADAS = cfg.FEATURES / "v20_f3_oof_cumuladas.npy"
OOF_V20 = cfg.FEATURES / "v20_f3b_oof_candidato.npy"
CV_V20 = 0.6097782
ETIQUETAS = list(range(len(cfg.CLASSES)))
#: Hiperparametros del boosting especializado: binario, pequeno y fuertemente regularizado.
BOOST_B1 = {"objective": "binary", "n_estimators": 300, "learning_rate": 0.05, "num_leaves": 8,
            "max_depth": 3, "min_child_samples": 200, "colsample_bytree": 0.5, "subsample": 0.8,
            "subsample_freq": 1, "reg_lambda": 1.0, "n_jobs": 32, "random_state": cfg.SEED,
            "verbose": -1}


def recon_2d(acumuladas: np.ndarray) -> np.ndarray:
    """Las seis clases por diferencias de las cinco CDF, con monotonia por maximo acumulado."""
    n_clases = acumuladas.shape[1] + 1
    mono = np.maximum.accumulate(acumuladas, axis=1)
    seis = np.zeros((len(mono), n_clases))
    seis[:, 0] = mono[:, 0]
    seis[:, 1:-1] = np.diff(mono, axis=1)
    seis[:, -1] = 1.0 - mono[:, -1]
    seis = np.clip(seis, 0.0, None)
    return seis / np.clip(seis.sum(axis=1, keepdims=True), 1e-9, None)


def macro(clase: np.ndarray, prob: np.ndarray) -> float:
    """macro-F1 pooled de seis clases."""
    return float(f1_score(clase, np.argmax(prob, axis=1), average="macro", labels=ETIQUETAS))


def oof_b1(folds: list[dict], matriz: np.ndarray, clase: np.ndarray, modo: str) -> np.ndarray:
    """OOF del primer discriminador, con cache por modo."""
    cache = cfg.FEATURES / f"v22_b1_oof_{modo}.npy"
    if cache.exists():
        guardada = np.load(cache)
        if guardada.shape == (len(clase),):
            print(f"    cache: {cache.name}", flush=True)
            return guardada
    salida = np.zeros(len(clase))
    for fold in folds:
        validacion = fold["val_idx"]
        entrenamiento = np.concatenate([o["val_idx"] for o in folds
                                        if o["fold"] != fold["fold"]])
        objetivo = (clase[entrenamiento] <= 0).astype(int)
        if modo == "lineal_c02":
            escalador = StandardScaler().fit(matriz[entrenamiento])
            modelo = LinearSVC(C=0.2, max_iter=3000)
            modelo.fit(escalador.transform(matriz[entrenamiento]), objetivo)
            salida[validacion] = modelo.decision_function(escalador.transform(matriz[validacion]))
        elif modo == "lineal_c10":
            escalador = StandardScaler().fit(matriz[entrenamiento])
            modelo = LinearSVC(C=1.0, max_iter=3000)
            modelo.fit(escalador.transform(matriz[entrenamiento]), objetivo)
            salida[validacion] = modelo.decision_function(escalador.transform(matriz[validacion]))
        else:
            modelo = lgb.LGBMClassifier(**BOOST_B1)
            modelo.fit(matriz[entrenamiento], objetivo)
            salida[validacion] = modelo.predict_proba(matriz[validacion])[:, 1]
    np.save(cache, salida)
    return salida


def main() -> int:
    """Evalua los tres candidatos de B1 contra el control exacto de V20."""
    autotest_numerico()
    set_seed()
    inicio = time.time()
    train = datos.load_train()
    y = train[cfg.TARGET].astype(str).to_numpy()
    clase = np.array([cfg.CLASSES.index(v) for v in y])
    clusters = pd.read_parquet(cfg.FEATURES / "dup_clusters.parquet")["cluster"].to_numpy()[:len(y)]
    folds = cvmod.make_folds(y, groups=clusters, n_splits=5, n_repeats=1, scheme="group")
    matriz = matriz_279()
    acumuladas = np.load(ACUMULADAS)
    oof_v20 = np.load(OOF_V20)
    logreg = np.load(cfg.FEATURES / "v19_b1_oof_logreg.npy")
    mlp = np.load(cfg.FEATURES / "v19_b1_oof_mlp.npy")
    arbol_v19 = np.load(cfg.FEATURES / "v19_oof_arbol_5folds.npy")

    # ---------------------------------------------------------------- control exacto
    seis_v20 = recon_2d(acumuladas)
    blend_v20 = (0.25 * logreg + 0.25 * mlp + 0.25 * arbol_v19 + 0.25 * seis_v20)
    valor_v20 = macro(clase, blend_v20)
    cumple = abs(valor_v20 - CV_V20) < 1e-6
    print(f"[P1] V20 reconstruido {valor_v20:.7f} frente a {CV_V20:.7f} | reproduce: {cumple}", flush=True)
    if not cumple:
        print("[ABORTO] V20 no se reproduce")
        return 1

    candidatos = [("V20 (control exacto)", "v20", None),
                  ("LinearSVC C 0.2", "lineal_c02", "lineal_c02"),
                  ("LinearSVC C 1.0", "lineal_c10", "lineal_c10"),
                  ("boosting especializado B1", "boosting", "boosting")]
    resultados = []
    for nombre, clave, modo in candidatos:
        print(f"[{nombre}]", flush=True)
        if modo is None:
            b1 = acumuladas[:, 0].copy()
        else:
            b1 = oof_b1(folds, matriz, clase, modo)
        acumuladas_c = acumuladas.copy()
        acumuladas_c[:, 0] = b1
        seis = recon_2d(acumuladas_c)
        blend = (0.25 * logreg + 0.25 * mlp + 0.25 * arbol_v19 + 0.25 * seis)
        valor = macro(clase, blend)
        por_fold, deltas = [], []
        for fold in folds:
            idx = fold["val_idx"]
            a, b = macro(clase[idx], blend[idx]), macro(clase[idx], oof_v20[idx])
            por_fold.append(a)
            deltas.append(a - b)
        intervalo = bootstrap_diferencia(oof_v20, blend, clase, len(cfg.CLASSES))
        f1 = f1_score(clase, np.argmax(blend, axis=1), average=None, labels=ETIQUETAS)
        pred = np.array(cfg.CLASSES)[np.argmax(blend, axis=1)]
        pred_v20 = np.array(cfg.CLASSES)[np.argmax(oof_v20, axis=1)]
        acierta, acierta_v20 = pred == y, pred_v20 == y
        cambios = pred != pred_v20
        lado_alto = clase >= 1
        score_alto = 1.0 - b1
        decision = score_alto > 0.5
        bal = 0.5 * (decision[lado_alto].mean() + (~decision[~lado_alto]).mean())
        f1_b1 = float(f1_score(lado_alto.astype(int), decision.astype(int)))
        from sklearn.metrics import roc_auc_score
        auc = float(roc_auc_score(lado_alto.astype(int), score_alto))
        brier = float(np.mean((score_alto - lado_alto.astype(int)) ** 2))
        # Lo primero que decide el criterio es si **B1 mejora**; se calcula aqui, donde ya existe.
        b1_mejora = True if nombre.startswith("V20") else bal > 0.7632
        item = {"candidato": nombre, "macro_f1": round(valor, 7),
                "delta_contra_v20": round(valor - valor_v20, 6),
                "media_folds": round(float(np.mean(por_fold)), 6),
                "desviacion_folds": round(float(np.std(por_fold)), 6),
                "delta_por_fold": [round(v, 6) for v in deltas],
                "folds_positivos": sum(1 for v in deltas if v > 0),
                "bootstrap_ic95_contra_v20": [round(v, 6) for v in intervalo[:2]],
                "f1_por_clase": {c: round(float(f1[i]), 4) for i, c in enumerate(cfg.CLASSES)},
                "f1_13_17": round(float(f1[0]), 4), "f1_18_24": round(float(f1[1]), 4),
                "b1": {"exactitud_balanceada": round(float(bal), 4), "f1_binario": round(f1_b1, 4),
                       "auc": round(auc, 4), "brier": round(brier, 6)},
                "complementariedad": {
                    "ambos_correctos": int((acierta & acierta_v20).sum()),
                    "ambos_incorrectos": int((~acierta & ~acierta_v20).sum()),
                    "solo_candidato": int((acierta & ~acierta_v20).sum()),
                    "solo_v20": int((acierta_v20 & ~acierta).sum()),
                    "neto": int((acierta & ~acierta_v20).sum())
                            - int((acierta_v20 & ~acierta).sum()),
                    "cambios": int(cambios.sum())}}
        item["rescates"] = item["complementariedad"]["solo_candidato"]
        item["danos"] = item["complementariedad"]["solo_v20"]
        item["neto"] = item["complementariedad"]["neto"]
        # Criterios del brief, en este orden: lo primero que decide es si **B1 mejora**.
        if item["candidato"].startswith("V20"):
            item["estado"] = "CONTROL"
        elif not b1_mejora:
            # B1 no mejora: se cierra el mecanismo evaluado y se pasa al siguiente candidato.
            item["estado"] = "NO_SIGNAL: B1 no mejora, mecanismo cerrado"
        elif item["delta_contra_v20"] > 0 and item["folds_positivos"] >= 4 \
                and abs(valor - valor_v20) > 0.001:
            item["estado"] = "NEW_SIGNAL"
        elif item["delta_contra_v20"] > 0:
            item["estado"] = "WEAK_SIGNAL"
        else:
            item["estado"] = "WRONG_OBJECTIVE"
        resultados.append(item)
        print(f"    macro-F1 {valor:.7f} | contra V20 {item['delta_contra_v20']:+.4f} | "
              f"{item['folds_positivos']}/5 | B1 balanceada {bal:.4f} AUC {auc:.4f} | "
              f"neto {item['neto']:+,} | cambios {item['complementariedad']['cambios']:,} | {item['estado']}", flush=True)

    pd.DataFrame([{k: v for k, v in r.items() if k not in ("f1_por_clase", "b1",
                                                           "complementariedad", "delta_por_fold")}
                  for r in resultados]).to_csv(TABLA, index=False)
    INFORME.write_text("\n".join([
        "# V22 · B1_SPECIALIST — Sustituir unicamente el primer discriminador", "",
        "**V20 queda congelado**: LB **0,62689**, OOF **0,6097782**.", "",
        "## 1. Auditoria historica", "",
        "Hecha buscando **codigo**, no nombres. Ningun experimento habia sustituido especificamente B1, "
        "entrenado un modelo dedicado a `13-17` frente a `>=18`, ni usado una representacion especifica "
        "para B1. El antecedente cercano **E-028 no es equivalente**: entrena un `LinearSVC` sobre el "
        "**par adyacente** (dos clases, 57.539 filas), no sobre el umbral **anidado** (todas las filas, "
        "17/83).", "",
        "## 2. V20 como control exacto", "",
        f"- Reconstruido: **{valor_v20:.7f}** frente a **{CV_V20:.7f}**. **Reproduce: {cumple}**.",
        "- El script **se aborta** si no cuadra, porque sin ese control ninguna variante es "
        "interpretable.", "",
        "## 3. Resultados, con las otras cuatro fronteras intactas", "",
        "| Candidato | macro-F1 | Delta vs V20 | Folds | B1 balanceada | B1 AUC | Rescates | Danos | "
        "Neto | Cambios | Estado |", "|---|---|---|---|---|---|---|---|---|---|---|",
        *[f"| {r['candidato']} | {r['macro_f1']} | {r['delta_contra_v20']:+.4f} | "
          f"{r['folds_positivos']}/5 | {r['b1']['exactitud_balanceada']} | {r['b1']['auc']} | "
          f"{r['rescates']:,} | {r['danos']:,} | {r['neto']:+,} | {r['complementariedad']['cambios']:,} | "
          f"**{r['estado']}** |" for r in resultados], "",
        "## 4. Delta por fold", "", "| Candidato | Delta por fold |", "|---|---|",
        *[f"| {r['candidato']} | {r['delta_por_fold']} |" for r in resultados], "",
        "## 5. F1 por clase", "",
        "| Candidato | " + " | ".join(cfg.CLASSES) + " |", "|---|" + "---|" * len(cfg.CLASSES),
        *[f"| {r['candidato']} | " + " | ".join(f"{r['f1_por_clase'][c]}" for c in cfg.CLASSES) + " |"
          for r in resultados], "",
    ]), encoding="utf-8")
    RESULTADO.write_text(json.dumps({
        "fase": "V22 B1_SPECIALIST", "fecha": "2026-09-27",
        "champion_congelado": {"nombre": "V20", "lb": 0.62689, "cv": CV_V20, "intacto": True},
        "control_exacto": {"reproducido": cumple, "macro_f1": round(valor_v20, 7)},
        "candidatos": resultados, "submission_generada": False,
        "segundos": round(time.time() - inicio, 1),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nestados: " + " | ".join(f"{r['candidato']}={r['estado']}" for r in resultados))
    print(f"segundos: {time.time() - inicio:.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
