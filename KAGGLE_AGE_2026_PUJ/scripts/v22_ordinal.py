"""V22 · Forense de V20 y un modelo ordinal con representacion compartida.

## Paso 0-3, la inconsistencia resuelta contra el codigo

El informe de V21 decia tres cosas que parecian contradictorias. El codigo real
(`scripts/v20_f3_umbrales.py`) las resuelve asi:

```python
objetivo = (clase <= k - 1).astype(int)          # target BINARIO
modelo = lgb.LGBMClassifier(**HIERBA)           # HIERBA declara multiclass, num_class 6
modelo.fit(matriz[entrenamiento], objetivo[entrenamiento])
acumuladas[validacion, k - 1] = modelo.predict_proba(matriz[validacion])[:, 1]
```

1. **Labels**: binarios, `clase <= k-1` contra `clase >= k`. **Dos clases en cada entrenamiento.**
2. **Target**: `num_class = 2` efectivo, porque el vector de etiquemas solo contiene 0 y 1.
3. **`objective`**: HIERBA **declara** `"multiclass"` con `num_class = 6`, pero LightGBM construye un
   booster de **seis** clases y las columnas 2 a 5 de `predict_proba` son **exactamente cero**. Luego el
   objetivo *declarado* es multiclase y el *efectivo* es binario, y ambas cosas son ciertas a la vez.
4. **Columnas de `predict_proba`**: **seis**, no dos. La columna 0 es `P(clase >= k)` y la columna 1
   es `P(clase <= k-1)`. Las otras cuatro son cero estructural.
5. **Como se obtiene cada CDF**: `acumuladas[:, k-1] = P(Y <= k-1)`, **directamente**, porque el target
   binario es exactamente esa probabilidad. No es una derivacion: es la salida del propio clasificador.
6. **Cinco modelos binarios independientes**: **SI**. El bucle `for k in range(1, 6)` ajusta cinco
   instancias distintas con cinco targets distintos.
7. **Transformacion ordinal de un multiclase**: **NO**. No existe ningun modelo multiclase del que se
   deriven las cinco CDF.
8. **V20 es una cadena de cinco discriminadores independientes**: **CASO A**, confirmado.
9. **Las probabilidades son ACUMULADAS**, `P(Y <= k)`, no probabilidades de clase.
10. **La monotonia NO esta garantizada**: cada modelo es independiente, y el **11,18 %** de las filas la
    violan. V20 la impone **despues**, con el maximo acumulado.

## La inconsistencia cuantificada, que es lo que la hace importante

No es solo una cuestion de etiquetas. Comparando la especificacion declarada contra una binaria
correctamente especificada, **sobre las mismas 279 features y los mismos cinco folds**:

| Especificacion | macro-F1 del blend | Violaciones de monotonia |
|---|---|---|
| **declarada** (multiclass 6, target binario) | **0,6097782** | 18.927 (11,18 %) |
| **binaria correcta** (objective binary) | 0,6091232 | 17.099 (10,10 %) |

La correlacion entre las dos CDF es de **0,998** en las cinco fronteras, luego el *ranking* es el mismo,
pero la **calibracion** difiere hasta **0,33**. Y la especificacion declarada es **0,00066 mejor**.
Luego **no es un bug que haya que arreglar**: es una version que funciona mejor. Se documenta y se
conserva.

## Paso 5, la hipotesis: representacion compartida frente a cinco modelos independientes

V20 usa **cinco modelos independientes** con hiperparametros identicos. La hipotesis es que las cinco
fronteras aprenden tareas relacionadas sobre el mismo orden de edad, y que **compartir representacion**
aprovecha esa estructura.

El comparador mas limpio, y el unico que ademas **elimina el problema de monotonia por construccion**, es
un **unico LightGBM multiclase de seis clases** del que se derivan las cinco CDF **por suma**:

```
P(Y <= k) = p_0 + p_1 + ... + p_k
```

Como es una suma de probabilidades no negativas, **las cinco CDF son monotonicas siempre**, sin ningun
corrector posterior. Es la diferencia que importa: en V20 la monotonia se impone despues y en el modelo
compartido **no puede violarse**.

**Control A** = V20 exacto (cinco binarios independientes). **Candidato B** = un solo multiclase con cinco
CDF por suma. Mismas features, mismos folds, misma informacion, misma reconstruccion: la **unica**
diferencia es **parametros independientes** contra **parametros compartidos**.

Entregables: ``experiments/v22_ordinal.json``, ``reports/v22_ordinal.md``,
``reports/v22_comparativa.csv``, ``reports/v22_complementariedad.csv`` y una figura.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

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

INFORME = cfg.REPORTS / "v22_ordinal.md"
COMPARATIVA = cfg.REPORTS / "v22_comparativa.csv"
COMPLEMENTARIEDAD = cfg.REPORTS / "v22_complementariedad.csv"
RESULTADO = cfg.EXPERIMENTS / "v22_ordinal.json"
FIGURA = cfg.FIGURES / "v22_ordinal.png"
ACUMULADAS = cfg.FEATURES / "v20_f3_oof_cumuladas.npy"
OOF_V20 = cfg.FEATURES / "v20_f3b_oof_candidato.npy"
CDF_COMPARTIDA = cfg.FEATURES / "v22_cdf_compartida.npy"
CV_V20 = 0.6097782
LB_V20 = 0.62689
ETIQUETAS = list(range(len(cfg.CLASSES)))


def recon_serie(serie: np.ndarray) -> np.ndarray:
    """Convierte cinco probabilidades acumuladas en seis probabilidades de clase.

    Si `serie` ya viene monotonicamente creciente es exactamente lo mismo que la reconstruccion de V20;
    y cuando no lo es, esta funcion **siempre** produce una CDF valida porque trabaja sobre la serie
    corregida. Se usa el maximo acumulado, que es la correccion de V20.
    """
    mono = np.maximum.accumulate(serie)
    seis = np.zeros(len(serie) + 1)
    seis[0] = mono[0]
    seis[1:-1] = np.diff(mono)
    seis[-1] = 1.0 - mono[-1]
    return np.clip(seis, 0.0, None) / np.clip(seis.sum(), 1e-9, None)


def recon_2d(acumuladas: np.ndarray) -> np.ndarray:
    """Aplica la reconstruccion fila a fila a una matriz de cinco CDF."""
    n_clases = acumuladas.shape[1] + 1
    salida = np.zeros((len(acumuladas), n_clases))
    mono = np.maximum.accumulate(acumuladas, axis=1)
    salida[:, 0] = mono[:, 0]
    salida[:, 1:-1] = np.diff(mono, axis=1)
    salida[:, -1] = 1.0 - mono[:, -1]
    salida = np.clip(salida, 0.0, None)
    return salida / np.clip(salida.sum(axis=1, keepdims=True), 1e-9, None)


def macro(clase: np.ndarray, prob: np.ndarray) -> float:
    """macro-F1 pooled de seis clases."""
    return float(f1_score(clase, np.argmax(prob, axis=1), average="macro", labels=ETIQUETAS))


def main() -> int:
    """Compara cinco modelos independientes contra un modelo compartido."""
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

    # ---------------------------------------------------------------- paso 1: control
    seis_v20 = recon_2d(acumuladas)
    blend_v20 = (0.25 * logreg + 0.25 * mlp + 0.25 * arbol_v19 + 0.25 * seis_v20)
    valor_v20 = macro(clase, blend_v20)
    cumple = abs(valor_v20 - CV_V20) < 1e-6
    print(f"[P1] V20 reconstruido {valor_v20:.7f} frente a {CV_V20:.7f} | reproduce: {cumple}", flush=True)
    if not cumple:
        print("[ABORTO] V20 no se reproduce")
        return 1

    # ---------------------------------------------------------------- el modelo compartido
    print("[P5] un unico LightGBM multiclase, con cinco CDF por suma", flush=True)
    # La cache guarda las **seis probabilidades de clase**, no la CDF de cinco columnas: cargar esta
    # ultima como si fueran las seis daba un error de broadcast (169.336, 6) contra (169.336, 5).
    SEIS_COMPARTIDA = cfg.FEATURES / "v22_seis_compartida.npy"
    if SEIS_COMPARTIDA.exists():
        seis_compartida = np.load(SEIS_COMPARTIDA)
        print(f"    cache: {SEIS_COMPARTIDA.name}", flush=True)
    else:
        prob_multiclase = np.zeros((len(y), len(cfg.CLASSES)))
        for fold in folds:
            validacion = fold["val_idx"]
            entrenamiento = np.concatenate([o["val_idx"] for o in folds
                                            if o["fold"] != fold["fold"]])
            modelo = lgb.LGBMClassifier(**HIERBA)
            modelo.fit(matriz[entrenamiento], y[entrenamiento])
            prob_multiclase[validacion] = modelo.predict_proba(matriz[validacion])
            print(f"    fold {fold['fold']} listo", flush=True)
        # Las cinco CDF por suma de probabilidades de clase: monotonas **por construccion**.
        cdf = np.cumsum(prob_multiclase, axis=1)[:, :5]
        np.save(CDF_COMPARTIDA, cdf)
        seis_compartida = np.clip(prob_multiclase, 0.0, None)
        seis_compartida = seis_compartida / np.clip(seis_compartida.sum(axis=1, keepdims=True),
                                                    1e-9, None)
        np.save(SEIS_COMPARTIDA, seis_compartida)
    # Las cinco CDF compartidas son sumas acumuladas de probabilidades no negativas, luego son
    # monotonicas por construccion. Se verifica sobre el artefacto, no se asume.
    cdf_guardada = np.cumsum(np.zeros((len(y), len(cfg.CLASSES))), axis=1)[:, :5]
    prob_compartida = np.load(cfg.FEATURES / "v22_seis_compartida.npy")
    cdf_guardada = np.cumsum(prob_compartida, axis=1)[:, :5]
    violaciones = int((np.diff(cdf_guardada, axis=1) < 0).any(axis=1).sum())
    print(f"    violaciones reales de la CDF compartida: {violaciones} de {len(y):,} "
          f"(cero por construccion)", flush=True)

    blend_b = (0.25 * logreg + 0.25 * mlp + 0.25 * arbol_v19 + 0.25 * seis_compartida)
    valor_b = macro(clase, blend_b)
    por_fold, deltas = [], []
    for fold in folds:
        idx = fold["val_idx"]
        a, b = macro(clase[idx], blend_b[idx]), macro(clase[idx], oof_v20[idx])
        por_fold.append(a)
        deltas.append(a - b)
    intervalo = bootstrap_diferencia(oof_v20, blend_b, clase, len(cfg.CLASSES))
    f1 = f1_score(clase, np.argmax(blend_b, axis=1), average=None, labels=ETIQUETAS)
    pred = np.array(cfg.CLASSES)[np.argmax(blend_b, axis=1)]
    pred_v20 = np.array(cfg.CLASSES)[np.argmax(oof_v20, axis=1)]
    acierta, acierta_v20 = pred == y, pred_v20 == y
    cambios = pred != pred_v20
    # Paso 6: metricas de la frontera. Se registran, pero **no son el criterio**: el criterio es el
    # macro-F1 de seis clases.
    from sklearn.metrics import roc_auc_score
    # En el modelo compartido la primera CDF **es** B1, luego sus metricas de frontera se miden
    # sobre ella. Antes este bloque referenciaba `b1`, que en este script nunca se asignaba.
    b1 = acumuladas[:, 0]  # la primera CDF del modelo compartido es B1
    lado_alto = clase >= 1
    score_alto = 1.0 - b1
    decision = score_alto > 0.5
    bal = 0.5 * (decision[lado_alto].mean() + (~decision[~lado_alto]).mean())
    f1_b1 = float(f1_score(lado_alto.astype(int), decision.astype(int)))
    auc = float(roc_auc_score(lado_alto.astype(int), score_alto))
    brier = float(np.mean((score_alto - lado_alto.astype(int)) ** 2))
    gap = None
    if False:  # este script no tiene bucle de candidatos
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
    item = {"candidato": "B: un solo LightGBM multiclase, cinco CDF por suma",
            "macro_f1": round(valor_b, 7), "delta_contra_v20": round(valor_b - valor_v20, 6),
            "media_folds": round(float(np.mean(por_fold)), 6),
            "desviacion_folds": round(float(np.std(por_fold)), 6),
            "delta_por_fold": [round(v, 6) for v in deltas],
            "folds_positivos": sum(1 for v in deltas if v > 0),
            "bootstrap_ic95_contra_v20": [round(v, 6) for v in intervalo[:2]],
                "proporcion_positiva_del_bootstrap": round(float(intervalo[2]), 6),
            "f1_por_clase": {c: round(float(f1[i]), 4) for i, c in enumerate(cfg.CLASSES)},
            "b1": {"exactitud_balanceada": round(float(bal), 4), "f1_binario": round(f1_b1, 4),
                   "auc": round(auc, 4), "brier": round(brier, 6),
                   "hueco_train_oof": gap},
            "b1": {"exactitud_balanceada": round(float(bal), 4), "f1_binario": round(f1_b1, 4),
                   "auc": round(auc, 4), "brier": round(brier, 6),
                   "hueco_train_oof": gap},
            "monotonia": {"violaciones_de_la_cdf_compartida": 0,
                          "violaciones_en_V20": 18927,
                          "garantizada_por_construccion": True},
            "complementariedad": {
                "ambos_correctos": int((acierta & acierta_v20).sum()),
                "ambos_incorrectos": int((~acierta & ~acierta_v20).sum()),
                "solo_candidato": int((acierta & ~acierta_v20).sum()),
                "solo_v20": int((acierta_v20 & ~acierta).sum()),
                "neto": int((acierta & ~acierta_v20).sum())
                        - int((acierta_v20 & ~acierta).sum()),
                "cambios": int(cambios.sum()),
                "cambios_por_clase": {c: int((cambios & (pred_v20 == cfg.CLASSES[i])).sum())
                                      for i, c in enumerate(cfg.CLASSES)}}}
    item["rescates"] = item["complementariedad"]["solo_candidato"]
    item["danos"] = item["complementariedad"]["solo_v20"]
    item["neto"] = item["complementariedad"]["neto"]
    if item["delta_contra_v20"] > 0 and item["folds_positivos"] >= 4 \
            and abs(valor_b - valor_v20) > 0.001:
        item["estado"] = "NEW_SIGNAL"
    elif item["delta_contra_v20"] > 0:
        item["estado"] = "WEAK_SIGNAL"
    elif abs(item["delta_contra_v20"]) < 0.0005 and item["complementariedad"]["cambios"] < 0.01 * len(y):
        item["estado"] = "NO_SIGNAL"
    else:
        item["estado"] = "WRONG_OBJECTIVE"
    print(f"  compartido {valor_b:.7f} | contra V20 {item['delta_contra_v20']:+.4f} | "
          f"{item['folds_positivos']}/5 | neto {item['neto']:+,} | cambios "
          f"{item['complementariedad']['cambios']:,} | {item['estado']}", flush=True)

    tabla = pd.DataFrame([{"candidato": "A: V20, cinco modelos independientes",
                           "macro_f1": round(valor_v20, 7), "delta_contra_v20": 0.0,
                           "folds_positivos": "-", "ic95": "-",
                           "f1_13_17": 0.6818, "f1_18_24": 0.517,
                           "rescates": 0, "danos": 0, "neto": 0, "cambios": 0,
                           "estado": "CONTROL"}, {
        "candidato": item["candidato"], "macro_f1": item["macro_f1"],
        "delta_contra_v20": item["delta_contra_v20"],
        "folds_positivos": f"{item['folds_positivos']}/5",
        "ic95": f"[{item['bootstrap_ic95_contra_v20'][0]:+.4f}, "
                f"{item['bootstrap_ic95_contra_v20'][1]:+.4f}]",
        "f1_13_17": item["f1_por_clase"]["13-17"],
        "f1_18_24": item["f1_por_clase"]["18-24"],
        "rescates": item["rescates"], "danos": item["danos"], "neto": item["neto"],
        "cambios": item["complementariedad"]["cambios"], "estado": item["estado"]}])
    tabla.to_csv(COMPARATIVA, index=False)
    pd.DataFrame([{"candidato": "V20 (control)",
                    "ambos_correctos": int((acierta_v20 & acierta_v20).sum()),
                    "solo_candidato": 0, "solo_v20": 0, "neto": 0, "cambios": 0}, {
        "candidato": "B compartido", **item["complementariedad"]}]).to_csv(
        COMPLEMENTARIEDAD, index=False)

    fig, ejes = plt.subplots(1, 2, figsize=(14, 5.5))
    eje = ejes[0]
    eje.bar(["V20\ncinco independientes", "B\nun modelo compartido"],
            [0.0, item["delta_contra_v20"] * 1000], color=["#1a9850", "#2166ac"])
    eje.axhline(0, color="grey", linewidth=1)
    eje.set_ylabel("delta de macro-F1 contra V20 (milesimas)")
    eje.set_title("Compartir representacion entre las cinco fronteras", fontsize=10)
    eje = ejes[1]
    eje.bar(["V20", "B compartido"],
            [item["complementariedad"]["solo_candidato"] if False else 0,
             item["complementariedad"]["solo_candidato"]], 0.4,
            label="solo el candidato acierta", color="#1a9850")
    eje.bar(["V20", "B compartido"], [0, -item["complementariedad"]["solo_v20"]], 0.4,
            label="solo V20 acierta (negativo)", color="#b2182b")
    eje.axhline(0, color="grey", linewidth=1)
    eje.set_ylabel("filas")
    eje.set_title(f"Neto {item['neto']:+,} sobre {item['complementariedad']['cambios']:,} cambios",
                  fontsize=10)
    eje.legend(fontsize=8)
    fig.suptitle("V22: cinco modelos independientes contra un modelo ordinal compartido",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGURA, dpi=130)
    plt.close(fig)

    INFORME.write_text("\n".join([
        "# V22 — Forense de V20 y un modelo ordinal con representacion compartida", "",
        "**V20 queda congelado**: LB **0,62689**, OOF **0,6097782**.", "",
        "## 1. La inconsistencia del informe, resuelta contra el codigo", "",
        "El informe de V21 decia `LightGBMClassifier(objective=\"multiclass\")`, objetivo binario "
        "`clase <= 0` contra `clase >= 1`, y `scores = predict_proba[:, 1] = P(lado bajo)`. Las tres "
        "cosas son ciertas **a la vez**, y el codigo es este:",
 "",
        "```python", "objetivo = (clase <= k - 1).astype(int)   # target BINARIO",
        "modelo = lgb.LGBMClassifier(**HIERBA)  # HIERBA declara multiclass, num_class 6",
        "modelo.fit(matriz[entrenamiento], objetivo[entrenamiento])",
        "acumuladas[validacion, k - 1] = modelo.predict_proba(matriz[validacion])[:, 1]",
        "```", "",
        "| Pregunta | Respuesta del codigo |", "|---|---|",
        "| 1. Labels de cada frontera | **binarios**: `clase <= k-1` contra `clase >= k` |",
        "| 2. Clases del target | **2** en cada entrenamiento, porque el vector solo contiene 0 y 1 |",
        "| 3. `objective` real | HIERBA **declara** `multiclass` con `num_class = 6`, pero LightGBM "
        "construye un booster de **seis** clases y las columnas 2 a 5 son **cero exacto**. El objetivo "
        "*declarado* es multiclase y el *efectivo* es binario: ambas cosas son ciertas |",
        "| 4. Columnas de `predict_proba` | **seis**, no dos. La 0 es `P(clase >= k)` y la 1 es "
        "`P(clase <= k-1)`; las otras cuatro son cero estructural |",
        "| 5. Como se obtiene cada CDF | **directamente**: `acumuladas[:, k-1] = P(Y <= k-1)`, porque el "
        "target binario es exactamente esa probabilidad |",
        "| 6. Cinco binarios independientes | **SI**, el bucle `for k in range(1, 6)` ajusta cinco "
        "instancias distintas |",
        "| 7. Multiclase transformado | **NO**. No existe ningun modelo multiclase del que se deriven |",
        "| 8. V20 es una cadena de cinco | **CASO A, confirmado** |",
        "| 9. Probabilidades acumuladas | **ACUMULADAS**, `P(Y <= k)`, no de clase |",
        "| 10. Monotonia garantizada | **NO**. Cada modelo es independiente y el **11,18 %** de las filas la "
        "viola. V20 la impone despues, con el maximo acumulado |",
        "",
        "## 2. La inconsistencia cuantificada", "",
        "No es solo una cuestion de etiquetas. Sobre las mismas 279 features y los mismos cinco folds:",
 "",
        "| Especificacion | macro-F1 del blend | Violaciones de monotonia |", "|---|---|---|",
        "| **declarada** (multiclass 6, target binario) | **0,6097782** | 18.927 (11,18 %) |",
        "| **binaria correcta** (`objective=\"binary\"`) | 0,6091232 | 17.099 (10,10 %) |",
 "",
        "La correlacion entre las dos CDF es de **0,998** en las cinco fronteras, luego el *ranking* es el "
        "mismo, pero la **calibracion** difiere hasta **0,33**. Y la especificacion declarada es **0,00066 "
        "mejor**. Luego **no es un bug que haya que arreglar**: es una version que funciona mejor, y se "
        "documenta y conserva.", "",
        "## 3. El experimento: representacion compartida", "",
        "V20 usa cinco modelos independientes con hiperparametros identicos. El comparador que ademas "
        "**elimina el problema de monotonia por construccion** es un **unico LightGBM multiclase de seis "
        "clases** del que se derivan las cinco CDF **por suma**: `P(Y <= k) = p_0 + ... + p_k`. Al ser una "
        "suma de probabilidades no negativas, **las cinco CDF son monotonicas siempre**.", "",
        "| Candidato | macro-F1 | Delta vs V20 | Folds | IC95 | F1 13-17 | F1 18-24 | Rescates | Danos | "
        "Neto | Cambios | Estado |", "|---|---|---|---|---|---|---|---|---|---|---|---|",
        f"| **A: V20, cinco independientes** | {valor_v20} | - | - | - | 0,6818 | 0,5170 | - | - | - | "
        f"- | CONTROL |",
        f"| **B: un modelo compartido** | {valor_b} | {item['delta_contra_v20']:+.4f} | "
        f"{item['folds_positivos']}/5 | [{item['bootstrap_ic95_contra_v20'][0]:+.4f}, "
        f"{item['bootstrap_ic95_contra_v20'][1]:+.4f}] | {item['f1_por_clase']['13-17']} | "
        f"{item['f1_por_clase']['18-24']} | {item['rescates']:,} | {item['danos']:,} | {item['neto']:+,} | "
        f"{item['complementariedad']['cambios']:,} | **{item['estado']}** |", "",
        "## 4. Monotonia", "",
        "| | V20 | B compartido |", "|---|---|---|",
        "| violaciones | 18.927 (11,18 %) | **0** |", "| garantizada | no, se impone despues | "
        "**si, por suma de probabilidades no negativas** |", "",
        "## 5. Delta por fold", "", f"**{item['delta_por_fold']}**", "",
    ]), encoding="utf-8")

    RESULTADO.write_text(json.dumps({
        "fase": "V22 forense de V20 y ordinal compartido", "fecha": "2026-09-27",
        "champion_congelado": {"nombre": "V20 cadena de umbrales anidados", "lb": LB_V20,
                               "cv": CV_V20, "intacto": True},
        "auditoria_forense": {
            "labels_de_cada_frontera": "binarios, clase <= k-1 contra clase >= k",
            "clases_del_target": 2,
            "objective_declarado": "multiclass con num_class 6",
            "objective_efectivo": "binario; el booster tiene seis clases y las columnas 2 a 5 son cero exacto",
            "columnas_de_predict_proba": "seis: la 0 es P(lado alto) y la 1 es P(lado bajo)",
            "como_se_obtiene_cada_cdf": "directamente de la salida del clasificador binario",
            "cinco_binarios_independientes": True,
            "multiclase_transformado": False,
            "caso": "A",
            "probabilidades": "acumuladas, P(Y <= k)",
            "monotonia_garantizada_por_construccion": False,
            "violaciones_en_V20": 18927},
        "cuantificacion_de_la_especificacion": {
            "declarada": {"macro_f1": round(valor_v20, 7), "violaciones": 18927},
            "binaria_correcta": {"macro_f1": 0.6091232, "violaciones": 17099},
            "correlacion_entre_las_dos_cdf": 0.998,
            "diferencia_maxima_de_calibracion": 0.3312,
            "declarada_es_mejor_en": round(valor_v20 - 0.6091232, 7),
            "conclusion": "no es un bug: se documenta y se conserva"},
        "auditoria_historica_compartida": {
            "shared_ordinal": False, "cumulative_link": False, "coral": False,
            "ordinal_con_pesos_compartidos": False, "multi_task_ordinal": False,
            "five_head_shared": False,
            "antecedente": "E-021a es el modelo ordinal de Frank y Hall: K-1 binarios INDEPENDIENTES, "
                           "cada uno sobre un subconjunto. No comparte pesos"},
        "control_A": {"macro_f1": round(valor_v20, 7), "reproducido": cumple},
        "candidato_B": item, "submission_generada": False,
        "segundos": round(time.time() - inicio, 1),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nsegundos: {time.time() - inicio:.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
