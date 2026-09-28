"""V28 - 03 - P6/C1: descomposicion pairwise de la decision en 5 pares adyacentes disjuntos.

**Unico experimento de la tecnica C.** Pares fijados antes de entrenar, una sola regla de
agregacion, sin rejilla y sin rescate posterior.

## Que cambia y que no

| Se mantiene | Se cambia |
|---|---|
| la matriz de 279 features de V20 | el problema que resuelve cada uno de los 5 clasificadores |
| los folds y las agrupaciones de V20 | |
| los 3 miembros del blend | |
| la reconstruccion con maximo acumulado | |
| la capacidad por clasificador (300 estimadores, `HIERBA`) | |
| los thresholds y los pesos del blend | |

## La distincion con V20, que es lo que hace nueva a la tecnica

V20 entrena **cinco binarios anidados** de `clase <= k-1`. Cada uno mezcla **todas** las clases por
debajo del umbral, luego cada clasificador ve las seis clases y decide donde cortar una frontera
global.

C entrena **cinco binarios disjuntos**, uno por par adyacente, donde el clasificador `k` solo ve
las clases `k` y `k+1` y decide si la fila esta mas cerca de la superior. Los cinco problemas **no se
solapan**.

No es CORAL (pesos compartidos y umbrales ordenados) ni cumulative-link. Aqui cada frontera la
aprende un clasificador independiente con la matriz completa de features.

## Los pares, fijados antes de mirar la validacion

Salen de la confusion del **control V20**, que es informacion de train y out-of-fold, con un corte
declarado de 5.000 confusiones mutuas. Los seleccionaron exactamente los cinco pares adyacentes, que
concentran el 59,9 % de los 66.005 errores de V20. El criterio esta en
`experiments/v28_01_gap_analysis.json` y **no se toca despues**.

## Agregacion, fijada antes de mirar la validacion

Cadena de probabilidad independiente:

    p_k = P(clase > k) que devuelve el clasificador del par (k, k+1)
    P(clase <= k) = producto_{j < k} de (1 - p_j)
    P(clase = k)  = p_k por el producto_{j < k} de (1 - p_j)

Da una distribucion de seis clases que es **monotona por construccion** (los factores son menores o
iguales que 1), y se pasa por la **misma** `reconstruye` de V20, para que el post-proceso sea
identico. Una sola implementacion: no se prueban Borda, rank, voting duro ni acoplamiento.

## El coste mecanico de ser disjunta, declarado antes del resultado

Al ser disjuntos, **cada clasificador se entrena con las filas de sus dos clases, unas 56.000 en vez
de las 169.000 de V20**. Menos datos por clasificador. No es un defecto de la tecnica, es su
mecanismo: la descomposicion particiona las filas. Se declara para que un resultado negativo se
lea como el de la tecnica y no como un fallo de implementacion.
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
from sklearn.metrics import confusion_matrix, f1_score

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

RESULTADO = cfg.EXPERIMENTS / "v28_03_tecnica_C_pairwise.json"
MAESTRA_V28 = cfg.REPORTS / "v28_master_results.csv"
FIG = cfg.FIGURES / "v28_C_pairwise_diagnostico.png"
CUMULADAS_V20 = cfg.FEATURES / "v20_f3_oof_cumuladas.npy"
CV_V20 = 0.6097782
#: Corte declarado en el gap analysis, sobre la confusion del control V20.
CORTE_PARES = 5000
AGREGACION = "cadena de probabilidad independiente: P(clase<=k) = producto de (1 - p_j)"
GATE_DELTA = 0.003
GATE_FOLDS = 4
PARES = {("18-24", "25-34"): "18-24|25-34", ("25-34", "35-49"): "25-34|35-49"}


def macro(clase: np.ndarray, prob: np.ndarray) -> float:
    """macro-F1 pooled de las seis clases."""
    return float(f1_score(clase, np.argmax(prob, axis=1), average="macro",
                          labels=list(range(len(cfg.CLASSES)))))


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


def pares_seleccionados(clase: np.ndarray) -> list[tuple[int, int]]:
    """Pares con al menos CORTE_PARES confusiones mutuas en el control V20.

    Se calcula sobre las predicciones **almacenadas** de V20, que son out-of-fold, y se fija una
    sola vez. Devuelve pares de indices contiguos porque el criterio sobre la confusion bandada
    selecciona frontiers, no combinaciones arbitrarias.
    """
    almacenado = np.load(cfg.FEATURES / "v20_f3b_oof_candidato.npy")
    matriz = confusion_matrix(clase, np.argmax(almacenado, axis=1),
                              labels=list(range(len(cfg.CLASSES))))
    salida = []
    for k in range(len(cfg.CLASSES) - 1):
        if matriz[k, k + 1] + matriz[k + 1, k] >= CORTE_PARES:
            salida.append((k, k + 1))
    return salida


def cadena(cumprob: np.ndarray) -> np.ndarray:
    """Cadena de probabilidad independiente a partir de los p_k de cada par.

    `cumprob[:, k]` es P(clase > k) segun el clasificador del par (k, k+1). Devuelve la CDF
    acumulada de cinco columnas, identica en forma a la que produce la cadena de V20.
    """
    n_filas, n_umbral = cumprob.shape
    cdf = np.zeros((n_filas, n_umbral))
    vivo = np.ones(n_filas)
    for k in range(n_umbral):
        cdf[:, k] = vivo
        vivo = vivo * (1.0 - cumprob[:, k])
    return cdf


def main() -> int:
    """Ejecuta la tecnica C y decide."""
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

    pares = pares_seleccionados(clase)
    nombres = [f"{cfg.CLASSES[k]}|{cfg.CLASSES[j]}" for k, j in pares]
    filas_por_par = [int(((clase == k) | (clase == j)).sum()) for k, j in pares]
    print(f"[C] pares seleccionados con corte {CORTE_PARES}: {nombres}", flush=True)
    print(f"[C] filas por par: {filas_por_par} (V20 entrena cada binario con "
          f"{len(y):,} filas) | agregacion: {AGREGACION}", flush=True)

    # ---------------------------------------------------------------- T0: control
    seis_t0 = reconstruye(np.load(CUMULADAS_V20))
    oof_t0 = 0.25 * logreg + 0.25 * mlp + 0.25 * arbol_v19 + 0.25 * seis_t0
    valor_t0 = macro(clase, oof_t0)
    reproduce = abs(valor_t0 - CV_V20) < 1e-6
    print(f"[T0] V20 {valor_t0:.7f} | reproduce: {reproduce}", flush=True)
    if not reproduce:
        print("[STOP] V20 no reproduce. No se ejecuta C.")
        return 1

    # ---------------------------------------------------------------- T1: cinco binarios disjuntos
    acumuladas_t1 = np.zeros((len(y), n_clases - 1))
    for fold in folds:
        validacion = fold["val_idx"]
        entrenamiento = np.concatenate([o["val_idx"] for o in folds
                                        if o["fold"] != fold["fold"]])
        for (k, j), nombre in zip(pares, nombres):
            # Solo las filas de las dos clases del par. Es el coste mecanico de ser disjunta.
            filas = entrenamiento[(clase[entrenamiento] == k) | (clase[entrenamiento] == j)]
            objetivo = (clase[filas] == j).astype(int)
            modelo = lgb.LGBMClassifier(**{**HIERBA, "random_state": int(cfg.SEED)
                                           + 10 * int(fold["fold"]) + k})
            modelo.fit(matriz[filas], objetivo)
            # p = P(clase superior | clase en {k, j}); se aplica a TODAS las filas de validacion.
            acumuladas_t1[validacion, k] = modelo.predict_proba(matriz[validacion])[:, 1]
        print(f"    fold {fold['fold']}: 5 pares disjuntos listos "
              f"({time.time() - inicio:.0f} s)", flush=True)

    seis_t1 = reconstruye(cadena(acumuladas_t1))
    oof_t1 = 0.25 * logreg + 0.25 * mlp + 0.25 * arbol_v19 + 0.25 * seis_t1
    valor_t1 = macro(clase, oof_t1)
    print(f"[T1] {valor_t1:.7f} | delta {valor_t1 - valor_t0:+.6f}", flush=True)

    # ---------------------------------------------------------------- metricas
    filas, detalle = [], {}
    for nombre, prob in (("T0 V20 control", oof_t0), ("T1 cinco pares disjuntos", oof_t1)):
        pred = np.argmax(prob, axis=1)
        pred_t0 = np.argmax(oof_t0, axis=1)
        por = [macro(clase[f["val_idx"]], prob[f["val_idx"]]) for f in folds]
        base = [macro(clase[f["val_idx"]], oof_t0[f["val_idx"]]) for f in folds]
        deltas = [round(a - b, 6) for a, b in zip(por, base)]
        rescate = int(((pred_t0 != clase) & (pred == clase)).sum())
        dano = int(((pred_t0 == clase) & (pred != clase)).sum())
        cm = confusion_matrix(clase, pred, labels=list(range(n_clases)))
        item = {"modelo": nombre, "corrected_cv": round(macro(clase, prob), 7),
                "delta_vs_V20": round(macro(clase, prob) - valor_t0, 6),
                "fold_1": round(por[0], 7), "fold_2": round(por[1], 7), "fold_3": round(por[2], 7),
                "fold_4": round(por[3], 7), "fold_5": round(por[4], 7),
                "media": round(float(np.mean(por)), 6), "desviacion": round(float(np.std(por)), 6),
                "delta_por_fold": deltas, "folds_positivos": sum(1 for d in deltas if d > 0),
                "rescue": rescate, "damage": dano, "net": rescate - dano,
                "changed_rows": int((pred != pred_t0).sum()),
                "f1_por_clase": {c: round(float(v), 4) for c, v in zip(
                    cfg.CLASSES, f1_score(clase, pred, average=None, labels=list(range(n_clases))))},
                "complementariedad_por_clase": {
                    c: {"rescata": int(((pred_t0 != clase) & (pred == clase) & (clase == i)).sum()),
                        "danio": int(((pred_t0 == clase) & (pred != clase) & (clase == i)).sum())}
                    for i, c in enumerate(cfg.CLASSES)},
                "confusion": cm.tolist(),
                "errores_totales": int(len(clase) - np.trace(cm)),
                "errores_en_pares_adyacentes": int(sum(
                    cm[i, i + 1] + cm[i + 1, i] for i in range(n_clases - 1))),
                "frontera_1": round(float(metricas_frontera(prob, clase, PARES)["18-24|25-34"]
                                          ["accuracy"]), 4)}
        filas.append(item)
        detalle[nombre] = item
        print(f"[{nombre}] {item['corrected_cv']:.7f} | delta {item['delta_vs_V20']:+.6f} | "
              f"folds +{item['folds_positivos']}/5 | neto {item['net']:+,} | "
              f"filas {item['changed_rows']:,}", flush=True)

    # ---------------------------------------------------------------- la masa del 38,3 %
    # Se separa lo que C corrige de la masa que V19 y V20 fallan juntas, como pide el punto 35.
    pred_v19 = np.argmax(arbol_v19, axis=1)
    pred_t0 = np.argmax(oof_t0, axis=1)
    pred_t1 = np.argmax(oof_t1, axis=1)
    error_v19 = pred_v19 != clase
    error_v20 = pred_t0 != clase
    masa = error_v19 & error_v20
    corrects = pred_t1 == clase
    masa_analisis = {
        "filas": int(masa.sum()),
        "proporcion_del_train": round(float(masa.mean()), 6),
        "de_esas_C_acerta": int((masa & corrects).sum()),
        "de_esas_C_sigue_fallando": int((masa & ~corrects).sum()),
        "proporcion_de_la_masa_corregida": round(
            float((masa & corrects).sum() / max(masa.sum(), 1)), 6),
        "de_las_no_masa_C_acerta": int((~masa & corrects).sum()),
        "de_las_no_masa_C_falla": int((~masa & ~corrects & (pred_t0 != clase)).sum()),
    }
    print(f"[35] masa conjunta V19+V20: {masa_analisis['filas']:,} filas "
          f"({masa_analisis['proporcion_del_train']:.1%}) | C acierta "
          f"{masa_analisis['de_esas_C_acerta']:,} de ellas "
          f"({masa_analisis['proporcion_de_la_masa_corregida']:.2%})", flush=True)

    # ---------------------------------------------------------------- gate
    t1 = detalle["T1 cinco pares disjuntos"]
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
        decision, estado = "C CLOSED", "CLOSED"
    elif not cumple_delta:
        decision, estado = "C MARGINAL", "MARGINAL"
    elif cumple_folds and not bootstrap["ic95_incluye_cero"]:
        decision, estado = "C PROMISING (sin submission)", "PROMISING"
    else:
        decision, estado = "C MARGINAL", "MARGINAL"
    print(f"[gate] delta {t1['delta_vs_V20']:+.6f} (>= {GATE_DELTA}: {cumple_delta}) | "
          f"folds {t1['folds_positivos']}/5 (>= {GATE_FOLDS}: {cumple_folds})", flush=True)
    print(f"[decision] {decision}", flush=True)

    _figura(filas, masa_analisis)
    # La tabla maestra tiene un esquema **unico y fijo** para toda la fase. C aporta dos columnas
    # extra (`errores_totales` y `errores_en_pares_adyacentes`) que A no tiene: si se escribieran,
    # las filas de C tendrian mas campos que la cabecera y el fichero quedaria ilegible para pandas.
    # Las columnas por tecnica viven en su artefacto, no en la tabla compartida.
    pd.DataFrame([{"ID": "C-T0" if f["modelo"].startswith("T0") else "V28-03",
                   **{k: v for k, v in f.items()
                      if k not in {"f1_por_clase", "complementariedad_por_clase",
                                   "delta_por_fold", "confusion", "errores_totales",
                                   "errores_en_pares_adyacentes"}}}
                  for f in filas]).to_csv(MAESTRA_V28, mode="a",
                                           header=not MAESTRA_V28.exists(), index=False)
    RESULTADO.write_text(json.dumps({
        "fase": "V28 03 P6 tecnica C descomposicion pairwise", "fecha": "2026-09-28",
        "tecnica": "C",
        "hipotesis": "las confusiones dominantes son entre pares adyacentes, y un clasificador "
                     "disjunto por par puede aprender fronteras que el multiclase no representa",
        "novedad": "GENUINELY_NEW: OneVsOne, VotingClassifier y ClassifierChain tienen 0 usos; "
                   "V20 entrena cinco binarios ANIDADOS de clase<=k-1 y C cinco DISJUNTOS por par",
        "pares_seleccionados": nombres, "corte_de_seleccion": CORTE_PARES,
        "criterio_de_seleccion": "pares con al menos 5000 confusiones mutuas en la confusion "
                                 "out-of-fold del control V20, fijado antes de entrenar",
        "filas_por_par": filas_por_par,
        "agregacion": AGREGACION,
        "coste_mecanico_declarado": f"cada clasificador ve {filas_por_par[0]:,} filas en vez de "
                                    f"{len(y):,}: la descomposicion disjunta particiona las "
                                    f"filas, y eso es su mecanismo, no un defecto",
        "control_t0": {"corrected_cv": round(valor_t0, 7), "reproducido": reproduce},
        "candidatos": detalle, "delta": round(t1["delta_vs_V20"], 6),
        "masa_conjunta_38pct": masa_analisis,
        "gate": {"delta_minimo": GATE_DELTA, "folds_minimos": GATE_FOLDS,
                 "cumple_delta": cumple_delta, "cumple_folds": cumple_folds,
                 "bootstrap": bootstrap},
        "decision": decision, "estado": estado, "submission_generada": False,
        "segundos": round(time.time() - inicio, 1),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


def _figura(filas: list[dict], masa: dict) -> None:
    """Delta por fold, F1 por clase y correccion de la masa conjunta del 38,3 %."""
    fig, ejes = plt.subplots(1, 3, figsize=(16, 4.8))
    t0, t1 = filas[0], filas[1]
    eje = ejes[0]
    eje.bar(np.arange(5) - 0.2, t0["delta_por_fold"], 0.4, label="T0", color="#999999")
    eje.bar(np.arange(5) + 0.2, t1["delta_por_fold"], 0.4, label="T1 pares", color="#a50f15")
    eje.axhline(0, color="grey", linewidth=1)
    eje.axhline(GATE_DELTA, color="black", linestyle="--", linewidth=1)
    eje.set_xticks(range(5))
    eje.set_xticklabels([f"fold {i + 1}" for i in range(5)])
    eje.set_ylabel("delta de macro-F1 frente a T0")
    eje.set_title(f"C: delta {t1['delta_vs_V20']:+.6f}, {t1['folds_positivos']}/5", fontsize=10)
    eje.legend(fontsize=8)

    eje = ejes[1]
    clases = list(t0["f1_por_clase"].keys())
    x = np.arange(len(clases))
    eje.bar(x - 0.2, [t0["f1_por_clase"][c] for c in clases], 0.4, label="V20", color="#333333")
    eje.bar(x + 0.2, [t1["f1_por_clase"][c] for c in clases], 0.4, label="pares", color="#a50f15")
    eje.set_xticks(x)
    eje.set_xticklabels(clases, fontsize=8)
    eje.set_ylabel("F1")
    eje.set_title("F1 por clase", fontsize=10)
    eje.legend(fontsize=8)

    eje = ejes[2]
    cm0 = np.array(t0["confusion"])
    cm1 = np.array(t1["confusion"])
    adyacentes = [f"{i}|{i+1}" for i in range(5)]
    v0 = [(cm0[i, i + 1] + cm0[i + 1, i]) for i in range(5)]
    v1 = [(cm1[i, i + 1] + cm1[i + 1, i]) for i in range(5)]
    eje.bar(np.arange(5) - 0.2, v0, 0.4, label="V20", color="#333333")
    eje.bar(np.arange(5) + 0.2, v1, 0.4, label="pares", color="#a50f15")
    eje.set_xticks(range(5))
    eje.set_xticklabels(adyacentes, fontsize=8)
    eje.set_ylabel("confusiones mutuas")
    eje.set_title("Errores en los 5 pares adyacentes", fontsize=10)
    eje.legend(fontsize=8)
    fig.suptitle(f"Tecnica C: pares disjuntos. Corrige {masa['de_esas_C_acerta']:,} de las "
                 f"{masa['filas']:,} filas de la masa conjunta", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
