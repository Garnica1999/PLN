"""V21 — Mejorar unicamente la primera frontera ordinal: `13-17` frente a `>=18`.

V20 queda **congelado**: LB **0,62689**, OOF **0,6097782**. No se toca su entrenamiento, sus
features, su configuracion, sus cinco fronteras, su metodo de combinacion, sus thresholds, su
submission, su hash ni su score. Todo aqui es un challenger independiente.

## Paso 0, la auditoria historica (hecha buscando codigo, no nombres)

Se busco en el codigo si algun experimento habia **sustituido especificamente B1**, entrenado un
modelo dedicado a `13-17` frente a `>=18`, usado boosting/SVM/arboles solo en esa frontera, o usado una
representacion especifica unicamente para B1. **Ninguno existia.** El unico antecedente cercano es
**E-028**, y **no es equivalente**: entrena un `LinearSVC` sobre las filas del **par adyacente**
`("13-17", "18-24")`, es decir **dos clases y 57.539 filas**, mientras que B1 es el umbral **anidado**
`13-17` frente a `>=18`, es decir **todas las 169.336 filas** y un desbalance 17/83. Luego **los dos
experimentos de esta fase son genuinamente nuevos**.

## Paso 1, V20 como control exacto

Antes de tocar nada se reproduce V20 desde sus artefactos ya medidos y se exige que el OOF sea
**0,6097782**. Si no cuadra, el script se para: sin ese control, ninguna variante seria interpretable.
Se guardan las cinco predicciones acumuladas, las seis clases finales, el SHA256 y el F1 por clase.

## Lo que mide, y lo que no

Se registran las metricas de la frontera (exactitud balanceada, F1, AUC, score por fold, distribucion,
calibracion y hueco train a OOF), pero **no se usan como criterio**. El criterio es el **macro-F1 de
seis clases**, y una mejora grande de B1 con deterioro de las seis clases **es un fracaso**, no un exito.

## Los dos candidatos, y solo estos dos

1. **Control de solver**: `LinearSVC` para B1, con `C` base y una variante cercana. Mismas features,
   mismos folds, mismas labels, misma informacion y las otras cuatro fronteras intactas.
2. **Boosting especializado**: un LightGBM **binario, pequeno y regularizado** para B1, con las mismas
   279 features y sin introducir ninguna nueva. Hipotesis: V19 demostro que el boosting extrae
   interacciones no lineales que los combinadores lineales no capturan, y quizan esas interacciones sean
   especialmente utiles en esta frontera.

La **unica variable** de toda la fase es el modelo que genera B1.

Entregables: ``experiments/v21_b1.json``, ``reports/v21_b1.md``, ``reports/v21_b1_comparativa.csv``,
``reports/v21_b1_complementariedad.csv``, ``reports/v21_b1_ordinal.csv`` y una figura.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from v19_b1_boosting import HIERBA, matriz_279  # noqa: E402
from v13_p1b_fusion_tokens import PARES, bootstrap_diferencia, metricas_frontera  # noqa: E402

from src import config as cfg  # noqa: E402
from src import cv as cvmod  # noqa: E402
from src import data as datos  # noqa: E402
from src.config import set_seed  # noqa: E402
from src.verificacion_numerica import autotest_numerico  # noqa: E402

INFORME = cfg.REPORTS / "v21_b1.md"
COMPARATIVA = cfg.REPORTS / "v21_b1_comparativa.csv"
COMPLEMENTARIEDAD = cfg.REPORTS / "v21_b1_complementariedad.csv"
ORDINAL = cfg.REPORTS / "v21_b1_ordinal.csv"
RESULTADO = cfg.EXPERIMENTS / "v21_b1.json"
FIGURA = cfg.FIGURES / "v21_b1.png"
ACUMULADAS = cfg.FEATURES / "v20_f3_oof_cumuladas.npy"
OOF_V20 = cfg.FEATURES / "v20_f3b_oof_candidato.npy"

CV_V20 = 0.6097782
LB_V20 = 0.62689
N_UMBRAL = 5
ETIQUETAS = list(range(len(cfg.CLASSES)))
#: Hiperparametros del boosting especializado para B1: binario, pequeno y fuertemente regularizado.
BOOST_B1 = {"objective": "binary", "n_estimators": 300, "learning_rate": 0.05, "num_leaves": 8,
            "max_depth": 3, "min_child_samples": 200, "colsample_bytree": 0.5, "subsample": 0.8,
            "subsample_freq": 1, "reg_lambda": 1.0, "n_jobs": 32, "random_state": cfg.SEED,
            "verbose": -1}


def reconstruir(acumuladas: np.ndarray) -> np.ndarray:
    """Las seis clases por diferencias de las cinco CDF, con monotonia por maximo acumulado."""
    n_clases = acumuladas.shape[1] + 1
    mono = acumuladas.copy()
    for k in range(1, acumuladas.shape[1]):
        mono[:, k] = np.maximum(mono[:, k], mono[:, k - 1])
    seis = np.zeros((len(mono), n_clases))
    seis[:, 0] = mono[:, 0]
    for k in range(1, acumuladas.shape[1]):
        seis[:, k] = mono[:, k] - mono[:, k - 1]
    seis[:, n_clases - 1] = 1.0 - mono[:, n_clases - 2]
    seis = np.clip(seis, 0.0, None)
    return seis / np.clip(seis.sum(axis=1, keepdims=True), 1e-9, None)


def macro(clase: np.ndarray, prob: np.ndarray) -> float:
    """macro-F1 pooled de seis clases."""
    return float(f1_score(clase, np.argmax(prob, axis=1), average="macro", labels=ETIQUETAS))


def oof_b1(matriz: np.ndarray, y: np.ndarray, clase: np.ndarray, folds: list[dict],
           constructor: str | None = None) -> np.ndarray:
    """OOF del primer discriminador. `constructor` elige el estimador."""
    salida = np.zeros(len(y))
    cache = cfg.FEATURES / f"v21_b1_oof_{constructor}.npy"
    if cache.exists():
        guardada = np.load(cache)
        if guardada.shape == (len(y),):
            print(f"    cache: {cache.name}", flush=True)
            return guardada
    for fold in folds:
        validacion = fold["val_idx"]
        entrenamiento = np.concatenate([o["val_idx"] for o in folds
                                        if o["fold"] != fold["fold"]])
        objetivo = (clase[entrenamiento] <= 0).astype(int)
        if constructor == "lineal_c02":
            escalador = StandardScaler().fit(matriz[entrenamiento])
            modelo = LinearSVC(C=0.2, max_iter=3000)
            modelo.fit(escalador.transform(matriz[entrenamiento]), objetivo)
            salida[validacion] = modelo.decision_function(escalador.transform(matriz[validacion]))
        elif constructor == "lineal_c10":
            escalador = StandardScaler().fit(matriz[entrenamiento])
            modelo = LinearSVC(C=1.0, max_iter=3000)
            modelo.fit(escalador.transform(matriz[entrenamiento]), objetivo)
            salida[validacion] = modelo.decision_function(escalador.transform(matriz[validacion]))
        elif constructor == "boosting":
            modelo = lgb.LGBMClassifier(**BOOST_B1)
            modelo.fit(matriz[entrenamiento], objetivo)
            salida[validacion] = modelo.predict_proba(matriz[validacion])[:, 1]
        else:
            raise ValueError(f"constructor desconocido: {constructor}")
    np.save(cache, salida)
    return salida


def main() -> int:
    """Reproduce V20 como control exacto y mide los dos candidatos para B1."""
    autotest_numerico()
    set_seed()
    inicio = time.time()
    train = datos.load_train()
    y = train[cfg.TARGET].astype(str).to_numpy()
    clase = np.array([cfg.CLASSES.index(valor) for valor in y])
    clusters = pd.read_parquet(cfg.FEATURES / "dup_clusters.parquet")["cluster"].to_numpy()[:len(y)]
    folds = cvmod.make_folds(y, groups=clusters, n_splits=5, n_repeats=1, scheme="group")
    matriz = matriz_279()
    acumuladas = np.load(ACUMULADAS)
    oof_v20 = np.load(OOF_V20)

    # ---------------------------------------------------------------- paso 1: control exacto
    print("[P1] reproduciendo V20 desde sus artefactos ya medidos", flush=True)
    seis_v20 = reconstruir(acumuladas)
    blend_v20 = (0.25 * np.load(cfg.FEATURES / "v19_b1_oof_logreg.npy")
                 + 0.25 * np.load(cfg.FEATURES / "v19_b1_oof_mlp.npy")
                 + 0.25 * np.load(cfg.FEATURES / "v19_oof_arbol_5folds.npy")
                 + 0.25 * seis_v20)
    valor_v20 = macro(clase, blend_v20)
    cumple = abs(valor_v20 - CV_V20) < 1e-6
    print(f"  V20 reconstruido {valor_v20:.7f} frente a {CV_V20:.7f} | reproduce el control: {cumple}",
          flush=True)
    if not cumple:
        print("[ABORTO] V20 no se reproduce: sin este control ninguna variante es interpretable")
        return 1
    f1_v20 = f1_score(clase, np.argmax(blend_v20, axis=1), average=None, labels=ETIQUETAS)
    hash_v20 = hashlib.sha256(np.ascontiguousarray(blend_v20).tobytes()).hexdigest()
    print(f"  SHA256 del OOF de V20: {hash_v20[:16]}... | F1 por clase "
          f"{{c: round(float(f1_v20[i]), 4) for i, c in enumerate(cfg.CLASSES)}}", flush=True)

    # ---------------------------------------------------------------- paso 2: que es B1 hoy
    b1_actual = {"clase": "LightGBMClassifier", "hiperparametros": HIERBA,
                 "objetivo": "binario, clase <= 0 contra clase >= 1",
                 "representacion": "las 279 features de V12, sin ninguna nueva",
                 "regularizacion": "min_child_samples 200, colsample 0.5, subsample 0.8, reg_lambda 1.0",
                 "class_weights": "ninguno", "threshold": 0.5,
                 "calibracion": "ninguna; la probabilidad cruda del booster",
                 "scores": "predict_proba[:, 1], que es P(clase <= 0)"}
    print(f"[P2] B1 actual: {b1_actual['clase']} con {b1_actual['hiperparametros']}", flush=True)

    # ---------------------------------------------------------------- pasos 3 y 4
    candidatos = [("V20 (control exacto)", "v20", None),
                  ("LinearSVC C 0.2", "lineal_c02", "lineal_c02"),
                  ("LinearSVC C 1.0", "lineal_c10", "lineal_c10"),
                  ("boosting especializado B1", "boosting", "boosting")]
    logreg = np.load(cfg.FEATURES / "v19_b1_oof_logreg.npy")
    mlp = np.load(cfg.FEATURES / "v19_b1_oof_mlp.npy")
    arbol_v19 = np.load(cfg.FEATURES / "v19_oof_arbol_5folds.npy")
    resultados = []
    for nombre, clave, constructor in candidatos:
        print(f"[{nombre}]", flush=True)
        if constructor is None:
            b1 = acumuladas[:, 0].copy()
        else:
            b1 = oof_b1(matriz, y, clase, folds, constructor)
        acumuladas_candidato = acumuladas.copy()
        acumuladas_candidato[:, 0] = b1
        seis = reconstruir(acumuladas_candidato)
        blend = (0.25 * logreg + 0.25 * mlp + 0.25 * arbol_v19 + 0.25 * seis)
        valor = macro(clase, blend)
        por_fold, deltas = [], []
        for fold in folds:
            idx = fold["val_idx"]
            a = macro(clase[idx], blend[idx])
            b = macro(clase[idx], oof_v20[idx])
            por_fold.append(a)
            deltas.append(a - b)
        intervalo = bootstrap_diferencia(oof_v20, blend, clase, len(cfg.CLASSES))
        fronteras = metricas_frontera(blend, clase, PARES)
        fronteras_v20 = metricas_frontera(oof_v20, clase, PARES)
        pred = np.array(cfg.CLASSES)[np.argmax(blend, axis=1)]
        pred_v20 = np.array(cfg.CLASSES)[np.argmax(oof_v20, axis=1)]
        acierta, acierta_v20 = pred == y, pred_v20 == y
        f1 = f1_score(clase, np.argmax(blend, axis=1), average=None, labels=ETIQUETAS)
        # Paso 6: metricas de la frontera, que NO son el criterio.
        lado_alto = clase >= 1
        score_alto = 1.0 - b1
        decision = score_alto > 0.5
        bal = 0.5 * (decision[lado_alto].mean() + (~decision[~lado_alto]).mean())
        f1_b1 = float(f1_score(lado_alto.astype(int), decision.astype(int)))
        from sklearn.metrics import roc_auc_score
        auc = float(roc_auc_score(lado_alto.astype(int), score_alto))
        brier = float(np.mean((score_alto - lado_alto.astype(int)) ** 2))
        # train->OOF gap de B1, medido en el fold 0 con el mismo modelo.
        gap = None
        if constructor is not None:
            fold = folds[0]
            validacion = fold["val_idx"]
            entrenamiento = np.concatenate([o["val_idx"] for o in folds
                                            if o["fold"] != fold["fold"]])
            objetivo = (clase[entrenamiento] <= 0).astype(int)
            if constructor.startswith("lineal"):
                c = 0.2 if constructor == "lineal_c02" else 1.0
                escalador = StandardScaler().fit(matriz[entrenamiento])
                modelo = LinearSVC(C=c, max_iter=3000)
                modelo.fit(escalador.transform(matriz[entrenamiento]), objetivo)
                train_score = modelo.decision_function(escalador.transform(matriz[entrenamiento]))
                oof_score = modelo.decision_function(escalador.transform(matriz[validacion]))
            else:
                modelo = lgb.LGBMClassifier(**BOOST_B1)
                modelo.fit(matriz[entrenamiento], objetivo)
                train_score = modelo.predict_proba(matriz[entrenamiento])[:, 1]
                oof_score = modelo.predict_proba(matriz[validacion])[:, 1]
            train_f1 = float(f1_score((clase[entrenamiento] <= 0).astype(int),
                                      (1 - train_score) > 0.5))
            oof_f1 = float(f1_score((clase[validacion] <= 0).astype(int),
                                    (1 - oof_score) > 0.5))
            gap = {"train_f1_binario": round(train_f1, 6), "oof_f1_binario": round(oof_f1, 6),
                   "hueco": round(train_f1 - oof_f1, 6)}
        # Paso 9: distancia ordinal de los cambios.
        cambios = pred != pred_v20
        distancia = np.abs(clase - np.argmax(blend, axis=1))
        distancia_v20 = np.abs(clase - np.argmax(oof_v20, axis=1))
        ordinal = {"cambios": int(cambios.sum()),
                   "cambios_por_clase": {c: int((cambios & (pred_v20 == i)).sum())
                                         for i, c in enumerate(cfg.CLASSES)},
                   "cambios_por_clase_verdadera": {c: int((cambios & (clase == i)).sum())
                                                   for i, c in enumerate(cfg.CLASSES)},
                   "corrige_error_adyacente": int((cambios & acierta & ~acierta_v20
                                                   & (distancia_v20 == 1)).sum()),
                   "crea_error_adyacente": int((cambios & ~acierta & acierta_v20
                                               & (distancia == 1)).sum()),
                   "crea_salto_lejano": int((cambios & ~acierta & acierta_v20
                                            & (distancia >= 2)).sum()),
                   "distancia_media_de_los_nuevos_errores": round(
                       float(distancia[cambios & ~acierta].mean())
                       if (cambios & ~acierta).any() else 0.0, 4)}
        item = {"candidato": nombre, "macro_f1": round(valor, 7),
                "delta_contra_v20": round(valor - valor_v20, 6),
                "media_folds": round(float(np.mean(por_fold)), 6),
                "desviacion_folds": round(float(np.std(por_fold)), 6),
                "delta_por_fold": [round(v, 6) for v in deltas],
                "folds_positivos": sum(1 for v in deltas if v > 0),
                "bootstrap_ic95_contra_v20": [round(v, 6) for v in intervalo],
                "f1_por_clase": {c: round(float(f1[i]), 4) for i, c in enumerate(cfg.CLASSES)},
                "f1_13_17": round(float(f1[0]), 4), "f1_18_24": round(float(f1[1]), 4),
                "frontera_1": fronteras["18-24|25-34"]["accuracy"],
                "frontera_1_v20": fronteras_v20["18-24|25-34"]["accuracy"],
                "frontera_2": fronteras["25-34|35-49"]["accuracy"],
                "frontera_2_v20": fronteras_v20["25-34|35-49"]["accuracy"],
                "b1": {"exactitud_balanceada": round(float(bal), 4), "f1_binario": round(f1_b1, 4),
                       "auc": round(auc, 4), "brier": round(brier, 6),
                       "hueco_train_oof": gap},
                "complementariedad": {
                    "ambos_correctos": int((acierta & acierta_v20).sum()),
                    "ambos_incorrectos": int((~acierta & ~acierta_v20).sum()),
                    "solo_candidato": int((acierta & ~acierta_v20).sum()),
                    "solo_v20": int((acierta_v20 & ~acierta).sum()),
                    "neto": int((acierta & ~acierta_v20).sum())
                            - int((acierta_v20 & ~acierta).sum())},
                "ordinal": ordinal}
        item["rescates"] = item["complementariedad"]["solo_candidato"]
        item["danos"] = item["complementariedad"]["solo_v20"]
        item["cambios"] = ordinal["cambios"]
        resultados.append(item)
        print(f"    macro-F1 {valor:.7f} | contra V20 {item['delta_contra_v20']:+.4f} | "
              f"{item['folds_positivos']}/5 | IC95 [{intervalo[0]:+.4f}, {intervalo[1]:+.4f}] | "
              f"F1 13-17 {item['f1_13_17']} y 18-24 {item['f1_18_24']} | B1 balanceada {bal:.4f} "
              f"AUC {auc:.4f} | rescate {item['rescates']:,} dano {item['danos']:,} "
              f"neto {item['complementariedad']['neto']:+,}", flush=True)

    # ---------------------------------------------------------------- decision
    def decidir(item: dict) -> str:
        if item["candidato"].startswith("V20"):
            return "CONTROL"
        b1_mejor = item["b1"]["exactitud_balanceada"] > 0.7632
        final_mejor = item["delta_contra_v20"] > 0 and item["folds_positivos"] >= 4
        if b1_mejor and final_mejor:
            return "NEW_SIGNAL"
        if b1_mejor and abs(item["delta_contra_v20"]) < 0.0005:
            return "INFORMATIONAL_ONLY"
        if b1_mejor and item["delta_contra_v20"] < 0:
            return "WRONG_OBJECTIVE"
        return "NO_SIGNAL"

    for item in resultados:
        item["estado"] = decidir(item)
        print(f"  {item['candidato']:<28} B1 {item['b1']['exactitud_balanceada']} | final "
              f"{item['delta_contra_v20']:+.4f} | {item['estado']}", flush=True)

    tabla = pd.DataFrame([{k: v for k, v in r.items()
                           if k not in ("f1_por_clase", "b1", "complementariedad", "ordinal",
                                        "delta_por_fold")}
                          for r in resultados])
    tabla.to_csv(COMPARATIVA, index=False)
    pd.DataFrame([{"candidato": r["candidato"], **r["complementariedad"]}
                  for r in resultados]).to_csv(COMPLEMENTARIEDAD, index=False)
    pd.DataFrame([{"candidato": r["candidato"], **r["ordinal"]} for r in resultados]).to_csv(
        ORDINAL, index=False)

    fig, ejes = plt.subplots(1, 2, figsize=(16, 5.5))
    eje = ejes[0]
    nombres = [r["candidato"] for r in resultados]
    eje.bar(range(len(resultados)), [r["delta_contra_v20"] * 1000 for r in resultados],
            color=["#1a9850"] + ["#2166ac"] * (len(resultados) - 1))
    eje.axhline(0, color="grey", linewidth=1)
    eje.set_xticks(range(len(resultados)))
    eje.set_xticklabels(nombres, rotation=20, fontsize=7)
    eje.set_ylabel("delta de macro-F1 contra V20 (milesimas)")
    eje.set_title("Sustituir solo B1, con las otras cuatro fronteras intactas", fontsize=10)
    eje = ejes[1]
    eje.bar(np.arange(len(resultados)) - 0.2,
            [r["b1"]["exactitud_balanceada"] for r in resultados], 0.4,
            label="exactitud balanceada de B1", color="#b2182b")
    eje.bar(np.arange(len(resultados)) + 0.2,
            [r["b1"]["auc"] for r in resultados], 0.4, label="AUC de B1", color="#2166ac")
    eje.axhline(0.7632, color="grey", linestyle=":", linewidth=1, label="B1 de V20")
    eje.set_xticks(range(len(resultados)))
    eje.set_xticklabels(nombres, rotation=20, fontsize=7)
    eje.set_ylim(0.5, 1.0)
    eje.set_ylabel("valor")
    eje.set_title("Las metricas de la frontera NO son el criterio", fontsize=10)
    eje.legend(fontsize=8)
    fig.suptitle("V21: mejorar unicamente la primera frontera ordinal", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGURA, dpi=130)
    plt.close(fig)

    INFORME.write_text("\n".join([
        "# V21 — Mejorar unicamente la primera frontera ordinal", "",
        "**V20 queda congelado**: LB **0,62689**, OOF **0,6097782**. No se toca nada de V20.", "",
        "## 1. Auditoria historica, hecha buscando codigo", "",
        "Se busco si algun experimento habia **sustituido especificamente B1**, entrenado un modelo "
        "dedicado a `13-17` frente a `>=18`, usado boosting/SVM/arboles solo en esa frontera, o usado una "
        "representacion especifica unicamente para B1. **Ninguno existia.**", "",
        "El unico antecedente cercano es **E-028**, y **no es equivalente**: entrena un `LinearSVC` sobre "
        "las filas del **par adyacente** `(\"13-17\", \"18-24\")`, es decir **dos clases y 57.539 "
        "filas**, mientras que B1 es el umbral **anidado** `13-17` frente a `>=18`, es decir **todas las "
        "169.336 filas** y un desbalance 17/83. **Los dos experimentos de esta fase son genuinamente "
        "nuevos.**", "",
        "## 2. V20 como control exacto", "",
        f"- V20 reconstruido desde sus artefactos: **{valor_v20:.7f}**, frente a **{CV_V20:.7f}**.",
        f"- **Reproduce el control: {cumple}**. Sin ese control, ninguna variante seria interpretable.",
        f"- SHA256 del OOF de V20: `{hash_v20}`.",
        "- F1 por clase de V20: "
        + ", ".join(f"{c} {round(float(f1_v20[i]), 4)}" for i, c in enumerate(cfg.CLASSES)), "",
        "## 3. Que modelo usa B1 hoy", "",
        f"- **Clase**: {b1_actual['clase']}, objetivo binario `clase <= 0` contra `clase >= 1`.",
        f"- **Hiperparametros**: {json.dumps(HIERBA, ensure_ascii=False)}",
        f"- **Representacion**: {b1_actual['representacion']}.",
        f"- **Regularizacion**: {b1_actual['regularizacion']}.",
        f"- **Class weights**: {b1_actual['class_weights']}. **Threshold**: {b1_actual['threshold']}.",
        f"- **Calibracion**: {b1_actual['calibracion']}.",
        f"- **Scores**: {b1_actual['scores']}.", "",
        "## 4. Resultados, con las otras cuatro fronteras intactas", "",
        "| Candidato | macro-F1 | Delta vs V20 | Folds | IC95 | F1 13-17 | F1 18-24 | B1 balanceada | "
        "B1 AUC | Rescates | Danos | Neto | Cambios | Estado |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
        *[f"| {r['candidato']} | {r['macro_f1']} | {r['delta_contra_v20']:+.4f} | "
          f"{r['folds_positivos']}/5 | [{r['bootstrap_ic95_contra_v20'][0]:+.4f}, "
          f"{r['bootstrap_ic95_contra_v20'][1]:+.4f}] | {r['f1_13_17']} | {r['f1_18_24']} | "
          f"{r['b1']['exactitud_balanceada']} | {r['b1']['auc']} | {r['rescates']:,} | "
          f"{r['danos']:,} | {r['complementariedad']['neto']:+,} | {r['cambios']:,} | "
          f"**{r['estado']}** |" for r in resultados], "",
        "## 5. Delta por fold y estabilidad", "",
        "| Candidato | Delta por fold |", "|---|---|",
        *[f"| {r['candidato']} | {r['delta_por_fold']} |" for r in resultados], "",
        "## 6. Distancia ordinal de los cambios", "",
        "| Candidato | Cambios | Corrige error adyacente | Crea error adyacente | Crea salto lejano | "
        "Distancia media de los nuevos errores |", "|---|---|---|---|---|---|",
        *[f"| {r['candidato']} | {r['ordinal']['cambios']:,} | "
          f"{r['ordinal']['corrige_error_adyacente']:,} | {r['ordinal']['crea_error_adyacente']:,} | "
          f"{r['ordinal']['crea_salto_lejano']:,} | "
          f"{r['ordinal']['distancia_media_de_los_nuevos_errores']} |" for r in resultados], "",
        "## 7. Cambios por clase verdadera", "",
        "| Candidato | " + " | ".join(cfg.CLASSES) + " |", "|---|" + "---|" * len(cfg.CLASSES),
        *[f"| {r['candidato']} | " + " | ".join(
            str(r['ordinal']['cambios_por_clase_verdadera'][c]) for c in cfg.CLASSES) + " |"
          for r in resultados], "",
    ]), encoding="utf-8")

    RESULTADO.write_text(json.dumps({
        "fase": "V21 mejorar unicamente B1", "fecha": "2026-09-27",
        "champion_congelado": {"nombre": "V20 cadena de umbrales anidados", "lb": LB_V20,
                               "cv": CV_V20, "intacto": True},
        "auditoria_historica": {"sustituido_especificamente_B1": False,
                                 "modelo_dedicado_a_13_17_vs_mayor_igual_18": False,
                                 "boosting_solo_en_B1": False, "svm_solo_en_B1": False,
                                 "arboles_solo_en_B1": False,
                                 "representacion_especifica_para_B1": False,
                                 "antecedente_cercano": "E-028 entrena un LinearSVC sobre el par ADYACENTE "
                                                        "(dos clases, 57.539 filas), no sobre el umbral "
                                                        "anidado (todas las filas, 17/83)"},
        "control_exacto": {"v20_reconstruido": round(valor_v20, 7), "referencia": CV_V20,
                            "reproducido": cumple, "sha256": hash_v20,
                            "f1_por_clase": {c: round(float(f1_v20[i]), 4)
                                             for i, c in enumerate(cfg.CLASSES)}},
        "b1_actual": b1_actual,
        "candidatos": resultados,
        "unica_variable": "el modelo que genera B1",
        "submission_generada": False,
        "segundos": round(time.time() - inicio, 1),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nestados: " + " | ".join(f"{r['candidato']}={r['estado']}" for r in resultados))
    print(f"segundos: {time.time() - inicio:.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
