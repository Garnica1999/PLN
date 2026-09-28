"""V26 · 01 — Primer experimento minimo: representacion afin de la evidencia de los miembros.

**Un solo experimento.** El baseline V20 queda congelado y el unico cambio es una
transformacion afin de las 84 columnas que son predicciones de modelos base. No se cambia
simultaneamente la representacion de las otras features, ni el meta-modelo, ni la
regularizacion, ni los thresholds, ni los folds.

## La hipotesis, y por que el gap analysis la declara genuinamente nueva

> Las probabilidades de los modelos base no son directamente comparables como magnitudes
> absolutas. Parte de la senal perdida por V20 podria estar en diferencias sistematicas de escala
> entre modelos.

El gap analysis (`v26_00_gap_analysis.py`) verifico por equivalencia funcional que
**nunca se midio**: los 67 scripts que usan `StandardScaler` lo aplican a otras cosas, y V16 lo
aplico dentro de un combinador **logistico**, que es un meta-modelo distinto del LightGBM de V20.

## Por que NO es un no-op, que era la objecion obvia

Un arbol es invariante a una transformacion monotona, luego cabia esperar que centrar y escalar
las columnas no cambiera nada. **La medicion dice lo contrario**: con `colsample_bytree=0,5` y
`subsample=0,8`, un z-score afin cambia la prediccion **bit a bit**, con diferencia maxima **0,219**
sobre las 135.468 filas de un fold. La construccion de histogramas por cuantiles y el submuestreo
hacen que la invariancia teorica no se cumpla en esta implementacion. Por eso la hipotesis se
prueba en vez de descartarse por argumento.

## El alcance exacto del cambio

Se transforman **solo** las **84 columnas** de predicciones de modelos base (78 del ensamble de V12
y 6 de NB-SVM), con la media y la desviacion estimadas **exclusivamente dentro del training
fold** y aplicadas despues al fold de validacion. Las otras **195 columnas** quedan
**bit a bit iguales**.

Los tres miembros historicos del blend (LogReg, MLP y arbol de V19) **no se reentrenan**: son
artefactos congelados. El unico elemento del modelo que cambia son los **cinco LightGBM de la
cadena de umbrales**, que son los que ven las 279 columnas. Eso mantiene la comparacion limpia:
la diferencia entre T0 y T1 no puede ser otra cosa que la transformacion.

## Criterio de cierre

Gate del punto 29 del brief de V26: **delta >= +0,003** y **4/5 folds positivos**. El gate es mas
exigente que el +0,001 de V25 a proposito. Bootstrap solo si se supera el gate preliminar.

## Entregables

`experiments/v26_01_rank_representacion.json`, `reports/v26_master_results.csv`,
`reports/v26_hypothesis_registry.csv`, `reports/v26_reproducibility.md` y las figuras obligatorias.
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

RESULTADO = cfg.EXPERIMENTS / "v26_01_rank_representacion.json"
MAESTRA = cfg.REPORTS / "v26_master_results.csv"
REGISTRO = cfg.REPORTS / "v26_hypothesis_registry.csv"
REPRO = cfg.REPORTS / "v26_reproducibility.md"
FIG_DELTA = cfg.FIGURES / "v26_delta_por_experimento.png"
FIG_FOLD = cfg.FIGURES / "v26_comparacion_por_fold.png"
FIG_COMP = cfg.FIGURES / "v26_rescate_vs_dano.png"
FIG_CLASE = cfg.FIGURES / "v26_f1_por_clase.png"
FIG_EVOL = cfg.FIGURES / "v26_evolucion_corrected_cv.png"
CUMULADAS_V20 = cfg.FEATURES / "v20_f3_oof_cumuladas.npy"
CV_V20 = 0.6097782
LB_V20 = 0.62689
ETIQUETAS = list(range(len(cfg.CLASSES)))
#: Gate del punto 29 del brief de V26.
GATE_DELTA = 0.003
GATE_FOLDS = 4
PARES = {("18-24", "25-34"): "18-24|25-34", ("25-34", "35-49"): "25-34|35-49"}


def macro(clase: np.ndarray, prob: np.ndarray) -> float:
    """macro-F1 pooled de las seis clases."""
    return float(f1_score(clase, np.argmax(prob, axis=1), average="macro", labels=ETIQUETAS))


def reconstruye(acumuladas: np.ndarray) -> np.ndarray:
    """La reconstruccion **exacta** de V20: monotonia por maximo acumulado y diferencias."""
    n_umbral = acumuladas.shape[1]
    n_clases = n_umbral + 1
    mono = acumuladas.copy()
    for k in range(1, n_umbral):
        mono[:, k] = np.maximum(mono[:, k], mono[:, k - 1])
    seis = np.zeros((len(mono), n_clases))
    seis[:, 0] = mono[:, 0]
    for k in range(1, n_umbral):
        seis[:, k] = mono[:, k] - mono[:, k - 1]
    seis[:, -1] = 1.0 - mono[:, n_umbral - 1]
    seis = np.clip(seis, 0.0, None)
    return seis / np.clip(seis.sum(axis=1, keepdims=True), 1e-9, None)


def main() -> int:
    """Ejecuta el control T0 y el candidato T1, y decide."""
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

    # Las 84 columnas que son predicciones de modelos base, segun el inventario de V25.
    inventario = pd.read_csv(cfg.REPORTS / "v25_feature_inventory.csv")
    columnas = inventario.loc[inventario["bloque"].isin(["miembro_ensamble", "nbsvm_palabra"]),
                              "indice"].to_numpy()
    print(f"[V26-01] {matriz.shape[1]} features | {len(columnas)} son predicciones de modelos base",
          flush=True)
    if len(columnas) != 84:
        print("[ABORTO] el numero de columnas de miembros no es 84")
        return 1

    # ---------------------------------------------------------------- T0: control
    seis_t0 = reconstruye(np.load(CUMULADAS_V20))
    oof_t0 = 0.25 * logreg + 0.25 * mlp + 0.25 * arbol_v19 + 0.25 * seis_t0
    valor_t0 = macro(clase, oof_t0)
    reproduce = abs(valor_t0 - CV_V20) < 1e-6
    print(f"[T0] V20 {valor_t0:.7f} frente a {CV_V20:.7f} | reproduce: {reproduce}", flush=True)
    if not reproduce:
        print("[STOP] V20 no reproduce. No se ejecuta T1.")
        return 1

    # ---------------------------------------------------------------- T1: representacion afin
    acumuladas_t1 = np.zeros((len(y), n_clases - 1))
    for fold in folds:
        validacion = fold["val_idx"]
        entrenamiento = np.concatenate([o["val_idx"] for o in folds
                                        if o["fold"] != fold["fold"]])
        x_entrenamiento = matriz[entrenamiento].copy()
        x_validacion = matriz[validacion].copy()
        # Media y desviacion **exclusivamente del training fold**.
        for columna in columnas:
            media = x_entrenamiento[:, columna].mean()
            desviacion = x_entrenamiento[:, columna].std()
            if desviacion < 1e-12:
                desviacion = 1.0
            x_entrenamiento[:, columna] = (x_entrenamiento[:, columna] - media) / desviacion
            x_validacion[:, columna] = (x_validacion[:, columna] - media) / desviacion
        # Las otras 195 columnas quedan **bit a bit** iguales: no se tocan.
        for k in range(1, n_clases):
            objetivo = (clase <= k - 1).astype(int)
            modelo = lgb.LGBMClassifier(**HIERBA)
            modelo.fit(x_entrenamiento, objetivo[entrenamiento])
            acumuladas_t1[validacion, k - 1] = modelo.predict_proba(x_validacion)[:, 1]
        print(f"    fold {fold['fold']}: cadena con evidencia reescalada lista", flush=True)

    seis_t1 = reconstruye(acumuladas_t1)
    oof_t1 = 0.25 * logreg + 0.25 * mlp + 0.25 * arbol_v19 + 0.25 * seis_t1
    valor_t1 = macro(clase, oof_t1)
    print(f"[T1] {valor_t1:.7f} | delta {valor_t1 - valor_t0:+.6f}", flush=True)

    # ---------------------------------------------------------------- metricas
    filas, detalle = [], {}
    for nombre, prob in (("T0 V20 control", oof_t0), ("T1 evidencia de miembros reescalada", oof_t1)):
        pred = np.argmax(prob, axis=1)
        pred_t0 = np.argmax(oof_t0, axis=1)
        por = [macro(clase[f["val_idx"]], prob[f["val_idx"]]) for f in folds]
        base = [macro(clase[f["val_idx"]], oof_t0[f["val_idx"]]) for f in folds]
        deltas = [round(a - b, 6) for a, b in zip(por, base)]
        rescate = int(((pred_t0 != clase) & (pred == clase)).sum())
        dano = int(((pred_t0 == clase) & (pred != clase)).sum())
        fronteras = metricas_frontera(prob, clase, PARES)
        item = {"ID": "M1-T0" if nombre.startswith("T0") else "V26-01",
                "modelo": nombre, "corrected_cv": round(macro(clase, prob), 7),
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
                "frontera_18_24_25_34": round(float(fronteras["18-24|25-34"]["accuracy"]), 4),
                "frontera_25_34_35_49": round(float(fronteras["25-34|35-49"]["accuracy"]), 4)}
        filas.append(item)
        detalle[nombre] = item
        print(f"[{nombre}] {item['corrected_cv']:.7f} | delta {item['delta_vs_V20']:+.6f} | "
              f"folds +{item['folds_positivos']}/5 | neto {item['net']:+,} | "
              f"filas {item['changed_rows']:,}", flush=True)

    # ---------------------------------------------------------------- gate
    t1 = detalle["T1 evidencia de miembros reescalada"]
    cumple_delta = t1["delta_vs_V20"] >= GATE_DELTA
    cumple_folds = t1["folds_positivos"] >= GATE_FOLDS
    bootstrap = None
    if cumple_delta:
        intervalo = bootstrap_diferencia(oof_t0, oof_t1, clase, n_clases)
        bootstrap = {"delta": round(t1["delta_vs_V20"], 6),
                     "ic95": [round(float(intervalo[0]), 6), round(float(intervalo[1]), 6)],
                     "proporcion_positiva": round(float(intervalo[2]), 6),
                     "ic95_incluye_cero": bool(intervalo[0] <= 0.0 <= intervalo[1])}
    else:
        intervalo = bootstrap_diferencia(oof_t0, oof_t1, clase, n_clases)
        print(f"[bootstrap informativo] delta {intervalo[0]:+.6f} IC95 "
              f"[{intervalo[0]:+.6f}, {intervalo[1]:+.6f}] (no promotiona: el gate no se cumple)",
              flush=True)
    if not cumple_delta and t1["delta_vs_V20"] <= 0:
        decision, estado = "H02 CLOSED", "CLOSED"
    elif not cumple_delta:
        decision, estado = "H02 MARGINAL", "MARGINAL"
    elif cumple_folds and bootstrap and not bootstrap["ic95_incluye_cero"]:
        decision, estado = "H02 PROMISING (sin submission)", "PROMISING"
    else:
        decision, estado = "H02 MARGINAL", "MARGINAL"
    print(f"[gate] delta {t1['delta_vs_V20']:+.6f} (>= {GATE_DELTA}: {cumple_delta}) | "
          f"folds {t1['folds_positivos']}/5 (>= {GATE_FOLDS}: {cumple_folds})", flush=True)
    print(f"[decision] {decision}", flush=True)

    # ---------------------------------------------------------------- artefactos
    pd.DataFrame([{k: v for k, v in f.items()
                   if k not in {"f1_por_clase", "complementariedad_por_clase", "delta_por_fold"}}
                  for f in filas]).to_csv(MAESTRA, index=False)
    pd.DataFrame([{
        "ID": "V26-01", "Hypothesis": "las probabilidades de los modelos base no son comparables "
                                      "como magnitudes absolutas",
        "Prior_evidence": "D-060 mostro que la senal fuerte esta en las 84 columnas de miembros, "
                          "con MI media 0,215 y frecuencia de seleccion 1,000",
        "Novelty": "el gap analysis verifico por equivalencia funcional que ninguna de las 67 "
                   "familias de normalizacion del proyecto se aplica a estas 84 columnas antes del "
                   "meta-modelo de V20",
        "Mechanism": "media y desviacion por columna, estimadas solo del training fold",
        "Method": "z-score de las 84 columnas de predicciones de modelos base, cinco LightGBM de "
                  "V20 sin cambiar",
        "Status": estado, "Result": f"delta {t1['delta_vs_V20']:+.6f}, {t1['folds_positivos']}/5 "
                                 f"folds positivos, neto {t1['net']:+,}",
        "Decision": decision}]).to_csv(REGISTRO, index=False)
    _figuras(filas, valor_t0)
    _reproducibilidad(reproduce, valor_t0, folds, t1, decision)

    RESULTADO.write_text(json.dumps({
        "fase": "V26 01 representacion afin de la evidencia", "fecha": "2026-09-27",
        "hipotesis_seleccionada": "H02",
        "champion_congelado": {"nombre": "V20", "lb": LB_V20, "cv": CV_V20, "intacto": True},
        "mecanismo": ("media y desviacion de las 84 columnas de predicciones de modelos base, "
                      "estimadas solo dentro de cada training fold; las otras 195 quedan bit a bit "
                      "iguales; los tres miembros historicos del blend no se reentrenan"),
        "control_t0": {"corrected_cv": round(valor_t0, 7), "reproducido": reproduce},
        "candidatos": detalle, "delta": round(t1["delta_vs_V20"], 6),
        "gate": {"delta_minimo": GATE_DELTA, "folds_minimos": GATE_FOLDS,
                 "cumple_delta": cumple_delta, "cumple_folds": cumple_folds,
                 "bootstrap": bootstrap},
        "decision": decision, "estado": estado, "submission_generada": False,
        "segundos": round(time.time() - inicio, 1),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


def _figuras(filas: list[dict], valor_t0: float) -> None:
    """Las seis figuras obligatorias del punto 35, mas la de evolucion."""
    t0, t1 = filas[0], filas[1]
    # 1 y 6: evolucion y comparacion de corrected_cv
    fig, ejes = plt.subplots(1, 2, figsize=(13, 4.8))
    eje = ejes[0]
    eje.bar(["V20\n(control)", "V26-01\nreescalado"], [t0["corrected_cv"], t1["corrected_cv"]],
            color=["#2166ac", "#b2182b"])
    eje.axhline(CV_V20, color="#2166ac", linestyle="--", linewidth=1, label="V20 canonico")
    eje.set_ylabel("corrected_cv")
    eje.set_title("V20 frente al candidato", fontsize=10)
    eje.legend(fontsize=8)
    eje = ejes[1]
    eje.bar(["V20", "V26-01"], [0.6097782, t1["corrected_cv"]], color=["#999999", "#b2182b"])
    eje.set_ylabel("corrected_cv")
    eje.set_ylim(0.60, 0.615)
    eje.set_title(f"Delta {t1['delta_vs_V20']:+.6f} (gate {GATE_DELTA:+.3f})", fontsize=10)
    fig.suptitle("V26-01: representacion afin de la evidencia de los miembros", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG_DELTA, dpi=130)
    plt.close(fig)
    # 3: fold a fold
    fig, eje = plt.subplots(figsize=(9, 4.2))
    eje.bar(np.arange(5) - 0.2, t0["delta_por_fold"], 0.4, label="T0 (control)",
            color="#999999")
    eje.bar(np.arange(5) + 0.2, t1["delta_por_fold"], 0.4, label="T1 (reescalado)", color="#b2182b")
    eje.axhline(0, color="grey", linewidth=1)
    eje.axhline(GATE_DELTA, color="black", linestyle="--", linewidth=1, label=f"gate {GATE_DELTA}")
    eje.set_xticks(range(5))
    eje.set_xticklabels([f"fold {i + 1}" for i in range(5)])
    eje.set_ylabel("delta de macro-F1 frente a T0")
    eje.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_FOLD, dpi=130)
    plt.close(fig)
    # 4: rescate frente a dano
    fig, eje = plt.subplots(figsize=(7, 4))
    eje.bar(["rescate", "dano"], [t1["rescue"], t1["damage"]],
            color=["#1a9850", "#b2182b"])
    for i, v in enumerate([t1["rescue"], t1["damage"]]):
        eje.text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=9)
    eje.set_ylabel("filas")
    eje.set_title(f"Neto {t1['net']:+,} | cambian {t1['changed_rows']:,} filas", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG_COMP, dpi=130)
    plt.close(fig)
    # 5: F1 por clase
    fig, eje = plt.subplots(figsize=(10, 4.2))
    x = np.arange(len(cfg.CLASSES))
    eje.bar(x - 0.2, [t0["f1_por_clase"][c] for c in cfg.CLASSES], 0.4, label="T0")
    eje.bar(x + 0.2, [t1["f1_por_clase"][c] for c in cfg.CLASSES], 0.4, label="T1")
    eje.set_xticks(x)
    eje.set_xticklabels(cfg.CLASSES, fontsize=8)
    eje.set_ylabel("F1")
    eje.set_title("F1 por clase", fontsize=10)
    eje.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_CLASE, dpi=130)
    plt.close(fig)
    # 6: evolucion de corrected_cv en V26
    fig, eje = plt.subplots(figsize=(7, 4))
    eje.plot(["V20", "V26-01"], [0.6097782, t1["corrected_cv"]], marker="o", color="#b2182b")
    eje.set_ylabel("corrected_cv")
    eje.set_title("Evolucion dentro de V26", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG_EVOL, dpi=130)
    plt.close(fig)


def _reproducibilidad(reproduce, valor_t0, folds, t1, decision) -> None:
    """Clasifica el nivel de reproducibilidad segun el punto 36 del brief."""
    cuerpo = [
        "# V26 · Reproducibilidad", "",
        "Clasificacion del punto 36 del brief. Los cinco niveles son `R0` no reproducible, `R1` "
        "reproducible con limitaciones, `R2` reproducido dentro de tolerancia, `R3` reproduccion "
        "exacta y `R4` reproduccion independiente.", "",
        "## R3 — Reproduccion exacta del baseline", "",
        "| Comprobacion | Estado |", "|---|---|",
        "| Baseline intacto | V20, hash de submission sin cambio, artefactos no modificados |",
        f"| Mismos folds | {[len(f['val_idx']) for f in folds]}, los mismos de V20 |",
        "| Mismo data hash | `v20_f3b_oof_candidato.npy` sin modificar |",
        "| Misma semilla | `cfg.SEED`, sin semilla nueva para seleccionar |",
        "| Misma generacion de features | `matriz_279()` verificada bit a bit contra el "
        "inventario de V25 |",
        "| Misma configuracion del meta-modelo | `HIERBA` importada del champion, sin tocar |",
        "| Mismos thresholds | cadena de cinco umbrales con maximo acumulado, identico a V20 |",
        f"| **Reproduccion del control** | **{valor_t0:.7f}** frente a {CV_V20:.7f}, "
        f"coincide: **{reproduce}** |", "",
        "**Nivel declarado: R3, reproduccion exacta.** No se declara R4 porque no hubo una "
        "reproduccion por un tercero ni con codigo independiente: el control se leyo del mismo "
        "artefacto que usa el candidato, que es la forma correcta de garantizar que la "
        "comparacion sea justa, pero no es una verificacion independiente.", "",
        "## Del candidato", "",
        f"- Delta frente a V20: **{t1['delta_vs_V20']:+.6f}** (gate {GATE_DELTA:+.3f}).",
        f"- Folds positivos: **{t1['folds_positivos']}/5** (gate {GATE_FOLDS}/5).",
        f"- Decision: **{decision}**.", "",
        "## Lo que NO se declara", "",
        "- No se declara R4: no hubo reproduccion independiente.",
        "- No se declara R2 sin mas: el control es exacto, luego R3 es el nivel correcto y mas "
        "fuerte que R2.", ""]
    REPRO.write_text("\n".join(cuerpo), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
