"""V28 - 02 - P2/A1: bagging del meta-modelo sobre las mismas features de V20.

**Unico experimento de la tecnica A.** N fijo, agregacion fijada antes de mirar la validacion, sin
rejilla y sin rescate posterior.

## Que cambia y que no

| Se mantiene | Se cambia |
|---|---|
| la matriz de 279 features de V20 | las filas que ve cada ajuste del meta-modelo |
| las predicciones de los 3 miembros del blend | |
| los folds y las agrupaciones de V20 | |
| la cadena de cinco umbrales con maximo acumulado | |
| la reconstruccion de seis clases | |
| los thresholds y el blend de cuatro cuartos | |

La tecnica **no** anade modelos base, **no** anade representaciones, **no** selecciona features y
**no** toca el espacio de informacion. Solo cambia como se estima el meta-modelo final.

## Agregacion, fijada antes de mirar la validacion

**Promedio de probabilidades** de los N miembros. Se declara aqui, no despues, para que la
comparacion sea una comparacion y no una busqueda.

## El remuestreo: bootstrap de Poisson

Cada miembro ve las **mismas filas** del training fold, pero con pesos `Poisson(1)` independientes
por fila. Es el bootstrap de Poisson, equivalente en distribucion a sortear filas con repeticion, y
se prefiere por dos razones Practicas: no duplica filas en memoria y se puede pasar a LightGBM
como `sample_weight` sin cambiar nada del pipeline.

## El confundidor que el brief obliga a declarar

LightGBM ya embolsado: `HIERBA` tiene `subsample=0.8` con `subsample_freq=1`, de modo que **cada
uno de los 300 arboles de un ajuste ya ve el 80% de las filas**. A anade un segundo nivel de
remuestreo, y ademas multiplica el numero de estimadores por N. Cualquier mejora podria ser
**capacidad y no reduccion de varianza**. El numero se declara aqui, antes del resultado, y se vuelve
a declarar en la decision: es la razon por la que un positivo pequeno NO se puede leer como prueba de
que el bagging reduce varianza.
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

RESULTADO = cfg.EXPERIMENTS / "v28_02_tecnica_A_bagging.json"
MAESTRA_V28 = cfg.REPORTS / "v28_master_results.csv"
FIG = cfg.FIGURES / "v28_A_bagging_diagnostico.png"
CUMULADAS_V20 = cfg.FEATURES / "v20_f3_oof_cumuladas.npy"
CV_V20 = 0.6097782
LB_V20 = 0.62689
#: Numero de miembros del ensemble. **Fijo, sin rejilla**, y declarado antes de entrenar.
N_MIEMBROS = 5
#: Regla de agregacion, fijada antes de mirar la validacion.
AGREGACION = "promedio de probabilidades"
GATE_DELTA = 0.003
GATE_FOLDS = 4
PARES = {("18-24", "25-34"): "18-24|25-34", ("25-34", "35-49"): "25-34|35-49"}


def macro(clase: np.ndarray, prob: np.ndarray) -> float:
    """macro-F1 pooled de las seis clases."""
    return float(f1_score(clase, np.argmax(prob, axis=1), average="macro",
                          labels=list(range(len(cfg.CLASSES)))))


def reconstruye(acumuladas: np.ndarray) -> np.ndarray:
    """La reconstruccion exacta de V20: monotonia por maximo acumulado y diferencias."""
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


def main() -> int:
    """Ejecuta la tecnica A y decide."""
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
    print(f"[A] {matriz.shape[1]} features | N = {N_MIEMBROS} miembros | agregacion: {AGREGACION}",
          flush=True)
    print(f"[A] LightGBM ya embolsado dentro: subsample={HIERBA['subsample']}, "
          f"subsample_freq={HIERBA['subsample_freq']} | estimadores por miembro "
          f"{HIERBA['n_estimators']}, total {HIERBA['n_estimators'] * N_MIEMBROS}", flush=True)

    # ---------------------------------------------------------------- T0: control
    seis_t0 = reconstruye(np.load(CUMULADAS_V20))
    oof_t0 = 0.25 * logreg + 0.25 * mlp + 0.25 * arbol_v19 + 0.25 * seis_t0
    valor_t0 = macro(clase, oof_t0)
    reproduce = abs(valor_t0 - CV_V20) < 1e-6
    print(f"[T0] V20 {valor_t0:.7f} | reproduce: {reproduce}", flush=True)
    if not reproduce:
        print("[STOP] V20 no reproduce. No se ejecuta A.")
        return 1

    # ---------------------------------------------------------------- T1: meta-modelo embolsado
    acumuladas_t1 = np.zeros((len(y), n_clases - 1))
    # A2 necesita la prediccion de **cada miembro** por separado, para medir cuanto coinciden entre
    # si. Se guardan las N matrices completas de acumuladas: (N, filas, umbrales).
    por_miembro = np.zeros((N_MIEMBROS, len(y), n_clases - 1))
    for fold in folds:
        validacion = fold["val_idx"]
        entrenamiento = np.concatenate([o["val_idx"] for o in folds
                                        if o["fold"] != fold["fold"]])
        for k in range(1, n_clases):
            objetivo = (clase <= k - 1).astype(int)
            for miembro in range(N_MIEMBROS):
                # Semilla derivada y reproducible: la misma para todos los folds y umbrales, con
                # el indice del miembro dentro. Cambiarla invalidaria el entrepares.
                semilla = int(cfg.SEED) + 1000 * miembro + 10 * int(fold["fold"]) + k
                pesos = np.random.default_rng(semilla).poisson(1.0, size=len(entrenamiento))
                modelo = lgb.LGBMClassifier(**{**HIERBA, "random_state": semilla})
                modelo.fit(matriz[entrenamiento], objetivo[entrenamiento],
                           sample_weight=pesos.astype(np.float64))
                por_miembro[miembro, validacion, k - 1] = \
                    modelo.predict_proba(matriz[validacion])[:, 1]
        # Agregacion pre-registrada: promedio de probabilidades.
        acumuladas_t1[validacion, :] = por_miembro[:, validacion, :].mean(axis=0)
        print(f"    fold {fold['fold']}: cadena embolsada lista ({time.time() - inicio:.0f} s)",
              flush=True)

    seis_t1 = reconstruye(acumuladas_t1)
    oof_t1 = 0.25 * logreg + 0.25 * mlp + 0.25 * arbol_v19 + 0.25 * seis_t1
    valor_t1 = macro(clase, oof_t1)
    print(f"[T1] {valor_t1:.7f} | delta {valor_t1 - valor_t0:+.6f}", flush=True)

    # ---------------------------------------------------------------- A2: control de varianza
    # Dos medidas, y ninguna de las dos se interpreta como mejora:
    #   1. dispersion de las acumuladas frente a las de V20;
    #   2. **acuerdo entre los N miembros**, que es la cantidad que el bagging deberia reducir.
    # Menor varianza NO es mejor F1: se reporta y se decide con el gate, no con la dispersion.
    argmax_miembros = np.argmax(por_miembro, axis=2)          # (N, filas, umbrales)
    acuerdos = [float((argmax_miembros[i] == argmax_miembros[j]).mean())
                for i in range(N_MIEMBROS) for j in range(i + 1, N_MIEMBROS)]
    dispersion = {
        "desviacion_typica_de_acumuladas_V20": float(np.std(np.load(CUMULADAS_V20), axis=0).mean()),
        "desviacion_typica_de_acumuladas_embolsadas": float(np.std(acumuladas_t1, axis=0).mean()),
        "max_abs_diferencia_de_acumuladas": float(
            np.abs(np.load(CUMULADAS_V20) - acumuladas_t1).max()),
        "mediana_abs_diferencia_de_acumuladas": float(
            np.median(np.abs(np.load(CUMULADAS_V20) - acumuladas_t1))),
        "acuerdo_de_prediccion_con_V20": float(
            (np.argmax(oof_t1, axis=1) == np.argmax(oof_t0, axis=1)).mean()),
        "acuerdo_medio_entre_miembros": round(float(np.mean(acuerdos)), 6),
        "acuerdo_minimo_entre_miembros": round(float(np.min(acuerdos)), 6),
        "desviacion_typica_entre_miembros": float(
            np.mean([np.std(por_miembro[i] - acumuladas_t1, axis=0).mean()
                     for i in range(N_MIEMBROS)])),
    }
    print(f"[A2] acuerdo medio entre los {N_MIEMBROS} miembros: "
          f"{dispersion['acuerdo_medio_entre_miembros']:.4f} | acuerdo con V20: "
          f"{dispersion['acuerdo_de_prediccion_con_V20']:.4f} | mediana |dif| de acumuladas: "
          f"{dispersion['mediana_abs_diferencia_de_acumuladas']:.6f}", flush=True)

    # ---------------------------------------------------------------- metricas
    filas, detalle = [], {}
    for nombre, prob in (("T0 V20 control", oof_t0),
                         ("T1 meta-modelo embolsado", oof_t1)):
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
                    cfg.CLASSES, f1_score(clase, pred, average=None,
                                          labels=list(range(n_clases))))},
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

    # ---------------------------------------------------------------- gate
    t1 = detalle["T1 meta-modelo embolsado"]
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
        decision, estado = "A CLOSED", "CLOSED"
    elif not cumple_delta:
        decision, estado = "A MARGINAL", "MARGINAL"
    elif cumple_folds and not bootstrap["ic95_incluye_cero"]:
        decision, estado = "A PROMISING (sin submission)", "PROMISING"
    else:
        decision, estado = "A MARGINAL", "MARGINAL"
    print(f"[gate] delta {t1['delta_vs_V20']:+.6f} (>= {GATE_DELTA}: {cumple_delta}) | "
          f"folds {t1['folds_positivos']}/5 (>= {GATE_FOLDS}: {cumple_folds})", flush=True)
    print(f"[decision] {decision}", flush=True)

    _figura(filas, dispersion)
    pd.DataFrame([{"ID": "A-T0" if f["modelo"].startswith("T0") else "V28-02",
                   **{k: v for k, v in f.items()
                      if k not in {"f1_por_clase", "complementariedad_por_clase",
                                   "delta_por_fold"}}}
                  for f in filas]).to_csv(
        cfg.REPORTS / "v28_master_results.csv", mode="a",
        # En la primera escritura hay que poner la cabecera; en las siguientes, anadir sin ella.
        header=not MAESTRA_V28.exists(), index=False)
    RESULTADO.write_text(json.dumps({
        "fase": "V28 02 P2 tecnica A bagging del meta-modelo", "fecha": "2026-09-28",
        "tecnica": "A", "hipotesis": "la estimacion de la frontera tiene varianza residual, y "
                                    "embolsar el meta-modelo sobre las mismas features la reduce",
        "novedad": "PARTIALLY_TESTED: LightGBM ya embolsaba filas por arbol via subsample, A anade "
                   "un segundo nivel de ajustes independientes",
        "n_miembros": N_MIEMBROS, "agregacion": AGREGACION,
        "remuestreo": "bootstrap de Poisson(1) como sample_weight, semilla derivada de cfg.SEED, "
                      "miembro, fold y umbral",
        "confundidor_declarado": f"los estimadores totales pasan de {HIERBA['n_estimators']} a "
                                 f"{HIERBA['n_estimators'] * N_MIEMBROS}: parte de cualquier "
                                 f"mejora puede ser capacidad y no reduccion de varianza",
        "control_t0": {"corrected_cv": round(valor_t0, 7), "reproducido": reproduce},
        "candidatos": detalle, "delta": round(t1["delta_vs_V20"], 6),
        "control_de_varianza_A2": dispersion,
        "gate": {"delta_minimo": GATE_DELTA, "folds_minimos": GATE_FOLDS,
                 "cumple_delta": cumple_delta, "cumple_folds": cumple_folds,
                 "bootstrap": bootstrap},
        "decision": decision, "estado": estado, "submission_generada": False,
        "segundos": round(time.time() - inicio, 1),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


def _figura(filas: list[dict], dispersion: dict) -> None:
    """Delta por fold, F1 por clase y el diagnostico de dispersion de A2."""
    fig, ejes = plt.subplots(1, 3, figsize=(16, 4.8))
    eje = ejes[0]
    t0, t1 = filas[0], filas[1]
    eje.bar(np.arange(5) - 0.2, t0["delta_por_fold"], 0.4, label="T0", color="#999999")
    eje.bar(np.arange(5) + 0.2, t1["delta_por_fold"], 0.4, label="T1 embolsado", color="#08519c")
    eje.axhline(0, color="grey", linewidth=1)
    eje.axhline(GATE_DELTA, color="black", linestyle="--", linewidth=1)
    eje.set_xticks(range(5))
    eje.set_xticklabels([f"fold {i + 1}" for i in range(5)])
    eje.set_ylabel("delta de macro-F1 frente a T0")
    eje.set_title(f"A: delta {t1['delta_vs_V20']:+.6f}, {t1['folds_positivos']}/5", fontsize=10)
    eje.legend(fontsize=8)

    eje = ejes[1]
    clases = list(t0["f1_por_clase"].keys())
    x = np.arange(len(clases))
    eje.bar(x - 0.2, [t0["f1_por_clase"][c] for c in clases], 0.4, label="V20", color="#333333")
    eje.bar(x + 0.2, [t1["f1_por_clase"][c] for c in clases], 0.4, label="embolsado", color="#08519c")
    eje.set_xticks(x)
    eje.set_xticklabels(clases, fontsize=8)
    eje.set_ylabel("F1")
    eje.set_title("F1 por clase", fontsize=10)
    eje.legend(fontsize=8)

    eje = ejes[2]
    eje.bar(["std V20", "std embolsado"],
            [dispersion["desviacion_typica_de_acumuladas_V20"],
             dispersion["desviacion_typica_de_acumuladas_embolsadas"]],
            color=["#999999", "#08519c"])
    eje.set_ylabel("desviacion tipica media de las acumuladas")
    eje.set_title("A2: dispersion (menor NO es mejor F1)", fontsize=10)
    fig.suptitle("Tecnica A: bagging del meta-modelo de V20", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
