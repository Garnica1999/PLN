"""V27 · 05 — H03: transformacion a percentil de la evidencia de los modelos base.

**Unica configuracion, sin grid de percentiles.** V20 congelado, mismos folds, mismo `HIERBA`,
misma cadena de umbrales y mismo blend. Lo unico que cambia es la representacion de las 84 columnas
que son predicciones de modelos base.

## Que la distingue de H02, que ya esta cerrado

| | H02 (cerrada) | H03 (esta) |
|---|---|---|
| Transformacion | `z = (p - media) / desviacion` | `percentil` empirico de `p` dentro de la distribucion de train |
| Tipo | **afin**, estrictamente creciente e inyectiva | **monotona debil**: colapsa valores a empates |
| Consecuencia | no pierde informacion de orden | **pierde** el orden fino dentro de cada tramo |

La distincion no es academica y esta medida: en V26 se comprobó que un z-score afín cambia la
prediccion del arbol **bit a bit** (diferencia maxima 0,219), y que un percentil tambien la cambia
(diferencia maxima 0,166) **por un motivo distinto**: el colapso de empates. H03 conserva el orden
global entre filas y destruye la distancia, que es la unica forma en que un percentil puede
aportar algo que H02 no apportaba.

## Una unica configuracion

Percentil empirico, sin discretizar a 10, 25, 50, 75 ni 90. La referencia se aprende **dentro de
cada training fold** y se aplica al fold de validacion, que nunca participa.

## Que se aplica

Solo las **84 columnas de evidencia de modelos base**, que son las que el gap analysis marco como
la senal fuerte. Las otras 195 quedan bit a bit iguales.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import f1_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src import config as cfg  # noqa: E402
from src import cv as cvmod  # noqa: E402
from src import data as datos  # noqa: E402
from src.verificacion_numerica import autotest_numerico  # noqa: E402
from v13_p1b_fusion_tokens import bootstrap_diferencia, metricas_frontera  # noqa: E402
from v19_b1_boosting import HIERBA, matriz_279  # noqa: E402

RESULTADO = cfg.EXPERIMENTS / "v27_05_h03_percentil.json"
FIG = cfg.FIGURES / "v27_h03_delta_por_fold.png"
CUMULADAS_V20 = cfg.FEATURES / "v20_f3_oof_cumuladas.npy"
CV_V20 = 0.6097782
LB_V20 = 0.62689
ETIQUETAS = list(range(len(cfg.CLASSES)))
GATE_DELTA = 0.003
GATE_FOLDS = 4
PARES = {("18-24", "25-34"): "18-24|25-34", ("25-34", "35-49"): "25-34|35-49"}


def macro(clase: np.ndarray, prob: np.ndarray) -> float:
    """macro-F1 pooled de las seis clases."""
    return float(f1_score(clase, np.argmax(prob, axis=1), average="macro", labels=ETIQUETAS))


def reconstruye(acumuladas: np.ndarray) -> np.ndarray:
    """La reconstruccion exacta de V20."""
    n_umbral = acumuladas.shape[1]
    mono = acumuladas.copy()
    for k in range(1, n_umbral):
        mono[:, k] = np.maximum(mono[:, k], mono[:, k - 1])
    seis = np.zeros((len(mono), n_umbral + 1))
    seis[:, 0] = mono[:, 0]
    for k in range(1, n_umbral):
        seis[:, k] = mono[:, k] - mono[:, k - 1]
    seis[:, -1] = 1.0 - mono[:, n_umbral - 1]
    seis = np.clip(seis, 0.0, None)
    return seis / np.clip(seis.sum(axis=1, keepdims=True), 1e-9, None)


def percentil_contra_referencia(valores: np.ndarray, referencia: np.ndarray) -> np.ndarray:
    """Percentil empirico de `valores` dentro de la distribucion ordenada `referencia`."""
    return np.searchsorted(referencia, valores, side="right") / len(referencia)


def main() -> int:
    """Ejecuta H03 con una unica configuracion de percentil."""
    autotest_numerico()
    inicio = time.time()
    train = datos.load_train()
    y = train[cfg.TARGET].astype(str).to_numpy()
    clase = np.array([cfg.CLASSES.index(v) for v in y])
    clusters = pd.read_parquet(cfg.FEATURES / "dup_clusters.parquet")["cluster"].to_numpy()[:len(y)]
    folds = cvmod.make_folds(y, groups=clusters, n_splits=5, n_repeats=1, scheme="group")
    matriz = matriz_279()
    n_clases = len(cfg.CLASSES)
    logreg = np.load(cfg.FEATURES / "v19_b1_oof_logreg.npy")
    mlp = np.load(cfg.FEATURES / "v19_b1_oof_mlp.npy")
    arbol_v19 = np.load(cfg.FEATURES / "v19_oof_arbol_5folds.npy")
    inventario = pd.read_csv(cfg.REPORTS / "v25_feature_inventory.csv")
    evidencia = inventario.loc[
        inventario["bloque"].isin(["miembro_ensamble", "nbsvm_palabra"]), "indice"].to_numpy()
    print(f"[H03] {matriz.shape[1]} features | {len(evidencia)} de evidencia a percentil | "
          f"{matriz.shape[1] - len(evidencia)} sin tocar", flush=True)

    seis_t0 = reconstruye(np.load(CUMULADAS_V20))
    oof_t0 = 0.25 * logreg + 0.25 * mlp + 0.25 * arbol_v19 + 0.25 * seis_t0
    valor_t0 = macro(clase, oof_t0)
    reproduce = abs(valor_t0 - CV_V20) < 1e-6
    print(f"[T0] V20 {valor_t0:.7f} | reproduce: {reproduce}", flush=True)
    if not reproduce:
        print("[STOP] V20 no reproduce. No se ejecuta H03.")
        return 1

    acumuladas_t1 = np.zeros((len(y), n_clases - 1))
    colisiones = []
    for fold in folds:
        validacion = fold["val_idx"]
        entrenamiento = np.concatenate([o["val_idx"] for o in folds
                                        if o["fold"] != fold["fold"]])
        x_entrenamiento = matriz[entrenamiento].copy()
        x_validacion = matriz[validacion].copy()
        for columna in evidencia:
            referencia = np.sort(matriz[entrenamiento, columna])
            x_entrenamiento[:, columna] = percentil_contra_referencia(
                matriz[entrenamiento, columna], referencia)
            x_validacion[:, columna] = percentil_contra_referencia(
                matriz[validacion, columna], referencia)
        colisiones.append(int(len(np.unique(x_entrenamiento[:, evidencia[0]]))
                            / len(x_entrenamiento[:, evidencia[0]])))
        for k in range(1, n_clases):
            objetivo = (clase <= k - 1).astype(int)
            modelo = lgb.LGBMClassifier(**HIERBA)
            modelo.fit(x_entrenamiento, objetivo[entrenamiento])
            acumuladas_t1[validacion, k - 1] = modelo.predict_proba(x_validacion)[:, 1]
        print(f"    fold {fold['fold']}: percentil listo", flush=True)

    seis_t1 = reconstruye(acumuladas_t1)
    oof_t1 = 0.25 * logreg + 0.25 * mlp + 0.25 * arbol_v19 + 0.25 * seis_t1
    valor_t1 = macro(clase, oof_t1)
    print(f"[T1] {valor_t1:.7f} | delta {valor_t1 - valor_t0:+.6f}", flush=True)

    filas, detalle = [], {}
    for nombre, prob in (("T0 V20 control", oof_t0), ("T1 evidencia en percentil", oof_t1)):
        pred = np.argmax(prob, axis=1)
        pred_t0 = np.argmax(oof_t0, axis=1)
        por = [macro(clase[f["val_idx"]], prob[f["val_idx"]]) for f in folds]
        base = [macro(clase[f["val_idx"]], oof_t0[f["val_idx"]]) for f in folds]
        deltas = [round(a - b, 6) for a, b in zip(por, base)]
        rescate = int(((pred_t0 != clase) & (pred == clase)).sum())
        dano = int(((pred_t0 == clase) & (pred != clase)).sum())
        item = {"modelo": nombre, "corrected_cv": round(macro(clase, prob), 7),
                "delta_vs_V20": round(macro(clase, prob) - valor_t0, 6),
                "fold_1": round(por[0], 7), "fold_2": round(por[1], 7), "fold_3": round(por[2], 7),
                "fold_4": round(por[3], 7), "fold_5": round(por[4], 7),
                "media": round(float(np.mean(por)), 6), "desviacion": round(float(np.std(por)), 6),
                "delta_por_fold": deltas, "folds_positivos": sum(1 for d in deltas if d > 0),
                "rescue": rescate, "damage": dano, "net": rescate - dano,
                "changed_rows": int((pred != pred_t0).sum()),
                "f1_por_clase": {c: round(float(v), 4) for c, v in zip(
                    cfg.CLASSES, f1_score(clase, pred, average=None, labels=ETIQUETAS))},
                "complementariedad_por_clase": {
                    c: {"rescata": int(((pred_t0 != clase) & (pred == clase) & (clase == i)).sum()),
                        "danio": int(((pred_t0 == clase) & (pred != clase) & (clase == i)).sum())}
                    for i, c in enumerate(cfg.CLASSES)},
                "frontera_1": round(float(metricas_frontera(prob, clase, PARES)["18-24|25-34"]
                                          ["accuracy"]), 4)}
        filas.append(item)
        detalle[nombre] = item
        print(f"[{nombre}] {item['corrected_cv']:.7f} | delta {item['delta_vs_V20']:+.6f} | "
              f"folds +{item['folds_positivos']}/5 | neto {item['net']:+,} | "
              f"filas {item['changed_rows']:,}", flush=True)

    t1 = detalle["T1 evidencia en percentil"]
    cumple_delta = t1["delta_vs_V20"] >= GATE_DELTA
    cumple_folds = t1["folds_positivos"] >= GATE_FOLDS
    intervalo = bootstrap_diferencia(oof_t0, oof_t1, clase, n_clases)
    bootstrap = {"delta": round(t1["delta_vs_V20"], 6),
                 "ic95": [round(float(intervalo[0]), 6), round(float(intervalo[1]), 6)],
                 "proporcion_positiva": round(float(intervalo[2]), 6),
                 "ic95_incluye_cero": bool(intervalo[0] <= 0.0 <= intervalo[1])}
    print(f"[bootstrap] delta {bootstrap['delta']:+.6f} IC95 "
          f"[{bootstrap['ic95'][0]:+.6f}, {bootstrap['ic95'][1]:+.6f}]", flush=True)
    if not cumple_delta and t1["delta_vs_V20"] <= 0:
        decision, estado = "H03 CLOSED", "CLOSED"
    elif not cumple_delta:
        decision, estado = "H03 MARGINAL", "MARGINAL"
    elif cumple_folds and not bootstrap["ic95_incluye_cero"]:
        decision, estado = "H03 PROMISING (sin submission)", "PROMISING"
    else:
        decision, estado = "H03 MARGINAL", "MARGINAL"
    print(f"[gate] delta {t1['delta_vs_V20']:+.6f} (>= {GATE_DELTA}: {cumple_delta}) | "
          f"folds {t1['folds_positivos']}/5 (>= {GATE_FOLDS}: {cumple_folds})", flush=True)
    print(f"[decision] {decision}", flush=True)

    _figura(filas)
    pd.DataFrame([{"ID": "H03-T0" if f["modelo"].startswith("T0") else "V27-05",
                   **{k: v for k, v in f.items()
                      if k not in {"f1_por_clase", "complementariedad_por_clase", "delta_por_fold"}}}
                  for f in filas]).to_csv(cfg.REPORTS / "v26_master_results.csv", mode="a",
                                           header=False, index=False)
    RESULTADO.write_text(json.dumps({
        "fase": "V27 05 H03 percentil", "fecha": "2026-09-28",
        "hipotesis": "la evidencia de los modelos base tiene una estructura de rango util que una "
                     "transformacion afín no puede expresar",
        "distincion_con_H02": ("H02 es afín, estrictamente creciente e inyectiva, luego conserva el "
                               "orden fino. H03 colapsa empates: conserva el orden entre filas y "
                               "destruye la distancia, que es la unica forma de aportar algo que "
                               "H02 no aportaba"),
        "configuracion_unica": "percentil empirico, sin discretizar a ningun numero de cortes",
        "referencia": "ordenada dentro de cada training fold, aplicada despues a la validacion",
        "columnas_transformadas": int(len(evidencia)),
        "fraccion_de_valores_unicos_del_primer_canal": colisiones,
        "control_t0": {"corrected_cv": round(valor_t0, 7), "reproducido": reproduce},
        "candidatos": detalle, "delta": round(t1["delta_vs_V20"], 6),
        "gate": {"delta_minimo": GATE_DELTA, "folds_minimos": GATE_FOLDS,
                 "cumple_delta": cumple_delta, "cumple_folds": cumple_folds,
                 "bootstrap": bootstrap},
        "decision": decision, "estado": estado, "submission_generada": False,
        "segundos": round(time.time() - inicio, 1),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


def _figura(filas: list[dict]) -> None:
    """Delta por fold y evolucion."""
    fig, ejes = plt.subplots(1, 2, figsize=(13, 4.5))
    eje = ejes[0]
    t0, t1 = filas[0], filas[1]
    eje.bar(np.arange(5) - 0.2, t0["delta_por_fold"], 0.4, label="T0", color="#999999")
    eje.bar(np.arange(5) + 0.2, t1["delta_por_fold"], 0.4, label="T1 percentil", color="#984ea3")
    eje.axhline(0, color="grey", linewidth=1)
    eje.axhline(GATE_DELTA, color="black", linestyle="--", linewidth=1)
    eje.set_xticks(range(5))
    eje.set_xticklabels([f"fold {i + 1}" for i in range(5)])
    eje.set_ylabel("delta de macro-F1 frente a T0")
    eje.legend(fontsize=8)
    eje.set_title(f"Delta {t1['delta_vs_V20']:+.6f}, {t1['folds_positivos']}/5", fontsize=10)
    eje = ejes[1]
    eje.plot(["V20", "H03"], [0.6097782, t1["corrected_cv"]], marker="o", color="#984ea3")
    eje.set_ylabel("corrected_cv")
    eje.set_title("H03 frente a V20", fontsize=10)
    fig.suptitle("H03: percentil de la evidencia de los modelos base", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
