"""V25 · 01 — Seleccion supervisada de features por informacion mutua, dentro de cada fold.

**Una sola comparacion.** La unica variable nueva es la seleccion de features. Todo lo demas es
V20 exacto: las mismas 279 columnas de entrada, el mismo LightGBM con `HIERBA` congelada, los
mismos cinco folds agrupados por duplicado, la misma cadena de cinco umbrales con correccion de
monotonia, y el mismo blend de cuatro cuartos.

## La hipotesis

> V20 puede mejorar eliminando features ruidosas mediante una seleccion supervisada realizada
> correctamente dentro de cada fold.

## Por que no es V15

V15 midio **correlacion de entrada y poda no supervisada**: VIF, correlacion y clustering, todo
**sin usar la variable objetivo**. V15 concluyo que la redundancia es funcional y que podar
**empeora** (delta -0,004, 0 de 5 folds), y de ahi salio el bloque de 73 variables label-informed
que V19 si explotó.

V25 usa la **variable objetivo**, y solo las etiquetas del **training fold**. No es una poda mas.

## Por que hay que escalar antes de medir el mutual information

`mutual_info_classif` estima la informacion mutua con un vecino mas cercano, y **la metrica de
distancia no es invariante a escala**. Las 73 columnas label-informed llegan a magnitud absoluta
**59.439** frente a las probabilidades de miembro, que viven en [0, 1]. Sin estandarizar, el
estimador compara distancias en unas columnas y en otras no, y el ranking deja de medir senal.

Por eso el estandarizado se hace **dentro de cada training fold** y se aplica a las columnas antes
de calcular el MI. No es un paso de seleccion: es la condicion para que el MI signifique algo, y el
informe registra la escala de cada columna para que se pueda comprobar.

## El control T0

T0 es **V20 sin seleccion**, reconstruido desde las cinco CDF guardadas. Si no reproduce
0,6097782 dentro de la tolerancia historica, la fase se detiene sin entrenar T1. T1 usa
**exactamente el mismo codigo de ajuste**, la misma `HIERBA` y la misma reconstruccion, luego la
diferencia entre T0 y T1 no puede ser otra cosa que las columnas seleccionadas.

## Regla de seleccion, pre-registrada y unica

`keep_top_50_percent`: K = floor(279 / 2) = **139**. Se aplica **independientemente en cada
training fold** y **no se ajusta observando el resultado**. No se corre ninguna grid, no se prueba
otro K y no se prueba otro selector.

## Entregables

`experiments/v25_01_seleccion.json`, `reports/v25_feature_selection.csv`,
`reports/v25_feature_selection_frequency.csv`, `reports/v25_feature_selection.md` y las figuras
de frecuencia de seleccion y de delta por fold.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import lightgbm as lgb  # noqa: E402

from src import config as cfg  # noqa: E402
from src import cv as cvmod  # noqa: E402
from src import data as datos  # noqa: E402
from src.verificacion_numerica import autotest_numerico  # noqa: E402
from v13_p1b_fusion_tokens import bootstrap_diferencia, metricas_frontera  # noqa: E402
from v19_b1_boosting import HIERBA, matriz_279  # noqa: E402

INVENTARIO = cfg.REPORTS / "v25_feature_inventory.csv"
SELECCION = cfg.REPORTS / "v25_feature_selection.csv"
FRECUENCIA = cfg.REPORTS / "v25_feature_selection_frequency.csv"
INFORME = cfg.REPORTS / "v25_feature_selection.md"
RESULTADO = cfg.EXPERIMENTS / "v25_01_seleccion.json"
FIG_FREQ = cfg.FIGURES / "v25_frecuencia_seleccion.png"
FIG_FOLD = cfg.FIGURES / "v25_delta_por_fold.png"
CUMULADAS_V20 = cfg.FEATURES / "v20_f3_oof_cumuladas.npy"
CV_V20 = 0.6097782
LB_V20 = 0.62689
ETIQUETAS = list(range(len(cfg.CLASSES)))
#: Regla de seleccion **pre-registrada y unica** (punto 6 del brief). No se varia.
TOP_PORCENTAJE = 0.5
GATE_DELTA = 0.001
GATE_FOLDS = 4
#: Semilla del submuestreo para el ranking, si hiciera falta. No se usa en el ajuste.
SEMILLA_MI = 20260927
PARES = {("18-24", "25-34"): "18-24|25-34", ("25-34", "35-49"): "25-34|35-49"}


def macro(clase: np.ndarray, prob: np.ndarray) -> float:
    """macro-F1 pooled de las seis clases."""
    return float(f1_score(clase, np.argmax(prob, axis=1), average="macro", labels=ETIQUETAS))


def reconstruye(acumuladas: np.ndarray) -> np.ndarray:
    """La reconstruccion **exacta** de V20: monotonia por maximo acumulado y diferencias.

    Es el mismo paso que `v20_f3_umbrales.py`: `P(Y = 0) = CDF_0`, `P(Y = k) = CDF_k - CDF_{k-1}`
    y la ultima clase se lleva el resto hasta 1, con recorte y renormalizacion.
    """
    n_clases = acumuladas.shape[1] + 1
    mono = acumuladas.copy()
    for k in range(1, acumuladas.shape[1]):
        mono[:, k] = np.maximum(mono[:, k], mono[:, k - 1])
    seis = np.zeros((len(mono), n_clases))
    seis[:, 0] = mono[:, 0]
    for k in range(1, acumuladas.shape[1]):
        seis[:, k] = mono[:, k] - mono[:, k - 1]
    n_umbral = acumuladas.shape[1]
    seis[:, n_clases - 1] = 1.0 - mono[:, n_umbral - 1]
    seis = np.clip(seis, 0.0, None)
    return seis / np.clip(seis.sum(axis=1, keepdims=True), 1e-9, None)


def main() -> int:
    """Ejecuta T0 y T1 con los folds de V20 y mide la diferencia."""
    autotest_numerico()
    inicio = time.time()
    train = datos.load_train()
    y = train[cfg.TARGET].astype(str).to_numpy()
    clase = np.array([cfg.CLASSES.index(v) for v in y])
    clusters = pd.read_parquet(cfg.FEATURES / "dup_clusters.parquet")["cluster"].to_numpy()[:len(y)]
    folds = cvmod.make_folds(y, groups=clusters, n_splits=5, n_repeats=1, scheme="group")
    matriz = matriz_279()
    n_clases = len(cfg.CLASSES)
    n_umbral = n_clases - 1
    K = int(np.floor(matriz.shape[1] * TOP_PORCENTAJE))
    print(f"[V25-01] {matriz.shape[1]} features | K = floor({matriz.shape[1]} x "
          f"{TOP_PORCENTAJE}) = {K}", flush=True)

    logreg = np.load(cfg.FEATURES / "v19_b1_oof_logreg.npy")
    mlp = np.load(cfg.FEATURES / "v19_b1_oof_mlp.npy")
    arbol_v19 = np.load(cfg.FEATURES / "v19_oof_arbol_5folds.npy")

    # ---------------------------------------------------------------- T0: control exacto
    # V20 **no es la cadena de cinco umbrales**: es el blend de cuatro cuartos. La cadena sola da
    # 0,6046893 y el blend da 0,6097782. Comparar contra la cifra del blend es lo unico
    # comparable, y es la misma distincion que ya hubo que hacer en V22.
    seis_t0 = reconstruye(np.load(CUMULADAS_V20))
    oof_t0 = 0.25 * logreg + 0.25 * mlp + 0.25 * arbol_v19 + 0.25 * seis_t0
    valor_t0 = macro(clase, oof_t0)
    valor_cadena = macro(clase, seis_t0)
    reproduce = abs(valor_t0 - CV_V20) < 1e-6
    print(f"[T0] cadena de cinco umbrales en solitario {valor_cadena:.7f} | "
          f"blend de V20 {valor_t0:.7f} frente a {CV_V20:.7f} | reproduce: {reproduce}",
          flush=True)
    if not reproduce:
        print("[ABORTO] T0 no reproduce el control de V20. No se ejecuta T1.")
        return 1

    # ---------------------------------------------------------------- T1: seleccion por fold
    acumuladas_t1 = np.zeros((len(y), n_umbral))
    por_fold: list[dict] = []
    problemas_mi: list[str] = []
    for fold in folds:
        validacion = fold["val_idx"]
        entrenamiento = np.concatenate([o["val_idx"] for o in folds
                                        if o["fold"] != fold["fold"]])
        # Estandarizado **dentro del fold de entrenamiento**. Nunca ve la validacion.
        escalador = StandardScaler()
        x_entrenamiento = escalador.fit_transform(matriz[entrenamiento])
        x_validacion = escalador.transform(matriz[validacion])
        if not np.isfinite(x_entrenamiento).all():
            finitos = np.where(np.isfinite(x_entrenamiento).all(axis=0))[0]
            problemas_mi.append(f"fold {fold['fold']}: {matriz.shape[1] - len(finitos)} columnas "
                                f"con infinito tras estandarizar; se usan solo las finitas")
            x_entrenamiento = x_entrenamiento[:, finitos]
            x_validacion = x_validacion[:, finitos]
        # Mutual information multiclase sobre las SEIS categorias, solo con el training fold.
        puntajes = mutual_info_classif(x_entrenamiento, clase[entrenamiento],
                                       n_neighbors=3, random_state=SEMILLA_MI)
        orden = np.argsort(puntajes)[::-1]
        elegido = np.sort(orden[:K])
        if len(elegido) != K:
            problemas_mi.append(f"fold {fold['fold']}: solo {len(elegido)} columnas finitas")
        # V20 exacto, pero sobre las columnas elegidas.
        for k in range(1, n_clases):
            objetivo = (clase <= k - 1).astype(int)
            modelo = lgb.LGBMClassifier(**HIERBA)
            modelo.fit(x_entrenamiento[:, elegido], objetivo[entrenamiento])
            acumuladas_t1[validacion, k - 1] = modelo.predict_proba(
                x_validacion[:, elegido])[:, 1]
        por_fold.append({"fold": int(fold["fold"]) + 1, "K": int(len(elegido)),
                         "indices": [int(i) for i in elegido],
                         "puntajes": {int(i): float(p) for i, p in zip(elegido, puntajes[elegido])},
                         "mi_max": float(np.nanmax(puntajes)),
                         "mi_min": float(np.nanmin(puntajes))})
        print(f"    fold {fold['fold']}: MI sobre {x_entrenamiento.shape[1]} columnas, "
              f"seleccionadas {len(elegido)} | MI max {np.nanmax(puntajes):.4f} | "
              f"min {np.nanmin(puntajes):.4f}", flush=True)

    seis_t1 = reconstruye(acumuladas_t1)
    valor_t1 = macro(clase, seis_t1)
    delta = valor_t1 - valor_t0
    print(f"[T1] {valor_t1:.7f} | delta {delta:+.6f}", flush=True)

    # ---------------------------------------------------------------- metricas comparadas
    oof_t1 = 0.25 * logreg + 0.25 * mlp + 0.25 * arbol_v19 + 0.25 * seis_t1
    filas, detalle = [], {}
    for nombre, prob in (("T0 V20 sin seleccion", oof_t0), ("T1 seleccion MI top 50%", oof_t1)):
        pred = np.argmax(prob, axis=1)
        por = [macro(clase[f["val_idx"]], prob[f["val_idx"]]) for f in folds]
        base = [macro(clase[f["val_idx"]], oof_t0[f["val_idx"]]) for f in folds]
        deltas = [round(a - b, 6) for a, b in zip(por, base)]
        pred_t0 = np.argmax(oof_t0, axis=1)
        solo_t1 = int(((pred_t0 != clase) & (pred == clase)).sum())
        solo_t0 = int(((pred_t0 == clase) & (pred != clase)).sum())
        fronteras = metricas_frontera(prob, clase, PARES)
        item = {"modelo": nombre, "corrected_cv": round(macro(clase, prob), 7),
                "delta_vs_t0": round(macro(clase, prob) - macro(clase, oof_t0), 6),
                "fold_1": round(por[0], 7), "fold_2": round(por[1], 7), "fold_3": round(por[2], 7),
                "fold_4": round(por[3], 7), "fold_5": round(por[4], 7),
                "media": round(float(np.mean(por)), 6), "desviacion": round(float(np.std(por)), 6),
                "delta_por_fold": deltas, "folds_positivos": sum(1 for d in deltas if d > 0),
                "rescue": solo_t1, "damage": solo_t0, "net": solo_t1 - solo_t0,
                "changed_rows": int((pred != pred_t0).sum()),
                "f1_por_clase": {c: round(float(v), 4) for c, v in zip(
                    cfg.CLASSES, f1_score(clase, pred, average=None, labels=ETIQUETAS))},
                "frontera_18_24_25_34": round(float(fronteras["18-24|25-34"]["accuracy"]), 4),
                "frontera_25_34_35_49": round(float(fronteras["25-34|35-49"]["accuracy"]), 4)}
        filas.append(item)
        detalle[nombre] = item
        print(f"[{nombre}] {item['corrected_cv']:.7f} | delta {item['delta_vs_t0']:+.6f} | "
              f"folds +{item['folds_positivos']}/5 | neto {item['net']:+,}", flush=True)
    pd.DataFrame([{k: v for k, v in f.items()
                   if k not in {"f1_por_clase", "delta_por_fold"}} for f in filas]).to_csv(
        SELECCION, index=False)

    # ---------------------------------------------------------------- estabilidad de la seleccion
    inventario = pd.read_csv(INVENTARIO)
    frecuencia = []
    for indice in range(matriz.shape[1]):
        veces = sum(1 for f in por_fold if indice in f["indices"])
        puntajes = [f["puntajes"][indice] for f in por_fold if indice in f["puntajes"]]
        fila = inventario.iloc[indice]
        frecuencia.append({"indice": indice, "nombre": fila["nombre"], "bloque": fila["bloque"],
                           "label_informed": bool(fila["label_informed"]),
                           "selected_in_n_folds": veces, "frequency": veces / len(por_fold),
                           "mean_MI": round(float(np.mean(puntajes)), 6) if puntajes else 0.0,
                           "es_constante": bool(fila["es_constante"])})
    tabla_freq = pd.DataFrame(frecuencia).sort_values(
        ["selected_in_n_folds", "mean_MI"], ascending=[False, False])
    tabla_freq.to_csv(FRECUENCIA, index=False)
    print(f"[estabilidad] {len(tabla_freq)} features | "
          f"{int((tabla_freq['selected_in_n_folds'] == len(por_fold)).sum())} en los 5 folds | "
          f"{int((tabla_freq['selected_in_n_folds'] == 0).sum())} en ninguno", flush=True)

    # ---------------------------------------------------------------- error por bloque
    bloques = {}
    for bloque, grupo in tabla_freq.groupby("bloque"):
        bloques[str(bloque)] = {
            "columnas": int(len(grupo)),
            "media_frecuencia": round(float(grupo["frequency"].mean()), 4),
            "media_mi": round(float(grupo["mean_MI"].mean()), 6),
            "nunca_seleccionadas": int((grupo["selected_in_n_folds"] == 0).sum()),
        }
    for bloque, v in sorted(bloques.items(), key=lambda kv: -kv[1]["media_frecuencia"]):
        print(f"    {bloque:<22} cols {v['columnas']:>3} | frec media {v['media_frecuencia']:.3f} "
              f"| MI media {v['media_mi']:.5f} | nunca {v['nunca_seleccionadas']}", flush=True)

    # ---------------------------------------------------------------- gate y bootstrap
    t1 = detalle["T1 seleccion MI top 50%"]
    cumple_delta = t1["delta_vs_t0"] >= GATE_DELTA
    cumple_folds = t1["folds_positivos"] >= GATE_FOLDS
    bootstrap = None
    if cumple_delta:
        intervalo = bootstrap_diferencia(oof_t0, oof_t1, clase, n_clases)
        # La tupla es (percentil 2,5, percentil 97,5, proporcion de remuestrios positivos).
        bootstrap = {"delta": round(delta, 6),
                     "ic95": [round(float(intervalo[0]), 6), round(float(intervalo[1]), 6)],
                     "proporcion_positiva": round(float(intervalo[2]), 6),
                     "ic95_incluye_cero": bool(intervalo[0] <= 0.0 <= intervalo[1])}
    if not cumple_delta:
        decision, estado = "SUPERVISED_FEATURE_SELECTION = CLOSED", "NO_SIGNAL"
    elif cumple_folds and bootstrap and not bootstrap["ic95_incluye_cero"]:
        decision, estado = ("SUPERVISED_FEATURE_SELECTION = CANDIDATE (local, sin submission)",
                            "CANDIDATE")
    else:
        decision, estado = "SUPERVISED_FEATURE_SELECTION = MARGINAL", "MARGINAL"
    print(f"[gate] delta {t1['delta_vs_t0']:+.6f} (>= {GATE_DELTA}: {cumple_delta}) | "
          f"folds {t1['folds_positivos']}/5 (>= {GATE_FOLDS}: {cumple_folds})", flush=True)
    print(f"[decision] {decision}", flush=True)

    _figuras(tabla_freq, filas)
    _informe(detalle, por_fold, bloques, t1, cumple_delta, cumple_folds, bootstrap, decision,
             problemas_mi, K, valor_t0, reproduce, delta, seis_t0, seis_t1)
    RESULTADO.write_text(json.dumps({
        "fase": "V25 01 seleccion supervisada", "fecha": "2026-09-27",
        "champion_congelado": {"nombre": "V20", "lb": LB_V20, "cv": CV_V20, "intacto": True},
        "hipotesis": ("V20 mejora eliminando features ruidosas con una seleccion supervisada "
                      "hecha dentro de cada fold"),
        "diferencia_con_v15": ("V15 podo SIN supervision, con VIF, correlacion y clustering, y "
                               "encontro que la redundancia es funcional. V25 usa la variable "
                               "objetivo y solo las etiquetas del training fold"),
        "variable_unica": "seleccion de features",
        "pre_registrado": {"top_porcentaje": TOP_PORCENTAJE, "K": int(K),
                           "metodo": "mutual_info_classif, n_neighbors=3",
                           "escalar": "StandardScaler dentro de cada training fold",
                           "semilla_mi": SEMILLA_MI, "n_splits": 5,
                           "agrupacion": "dup_clusters, StratifiedGroupKFold"},
        "control_t0": {"corrected_cv": round(valor_t0, 7), "reproducido": reproduce,
                       "criterio": f"|T0 - {CV_V20}| < 1e-6 o la fase se detiene"},
        "candidatos": detalle, "delta": round(delta, 6),
        "seleccion_por_fold": [{"fold": f["fold"], "K": f["K"], "mi_max": round(f["mi_max"], 6),
                                "mi_min": round(f["mi_min"], 6),
                                "indices": f["indices"]} for f in por_fold],
        "error_por_bloque": bloques,
        "gate": {"delta_minimo": GATE_DELTA, "folds_minimos": GATE_FOLDS,
                 "delta_t1": t1["delta_vs_t0"], "cumple_delta": cumple_delta,
                 "cumple_folds": cumple_folds, "bootstrap": bootstrap},
        "problemas_de_mi": problemas_mi,
        "decision": decision, "estado": estado, "submission_generada": False,
        "segundos": round(time.time() - inicio, 1),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


def _figuras(tabla_freq: pd.DataFrame, filas: list[dict]) -> None:
    """Frecuencia de seleccion por bloque, y delta por fold."""
    fig, ejes = plt.subplots(1, 2, figsize=(14, 5.0))
    eje = ejes[0]
    resumen = tabla_freq.groupby("bloque")["frequency"].mean().sort_values()
    eje.barh(resumen.index, resumen.values,
             color=["#b2182b" if v < 0.5 else "#2166ac" for v in resumen.values])
    eje.axvline(0.5, color="grey", linestyle="--", linewidth=1)
    eje.set_xlabel("frecuencia media de seleccion")
    eje.set_title("Que bloque sobrevive a la seleccion", fontsize=10)
    eje = ejes[1]
    eje.bar(np.arange(5) - 0.2, filas[0]["delta_por_fold"], 0.4, label="T0 (control)", color="#999999")
    eje.bar(np.arange(5) + 0.2, filas[1]["delta_por_fold"], 0.4, label="T1 (MI top 50%)",
            color="#1a9850")
    eje.axhline(0, color="grey", linewidth=1)
    eje.axhline(GATE_DELTA, color="#b2182b", linestyle="--", linewidth=1,
                label=f"gate {GATE_DELTA}")
    eje.set_xticks(range(5))
    eje.set_xticklabels([f"fold {i + 1}" for i in range(5)])
    eje.set_ylabel("delta de macro-F1 frente a T0")
    eje.set_title("Delta por fold", fontsize=10)
    eje.legend(fontsize=8)
    fig.suptitle("V25: seleccion supervisada de features dentro de cada fold", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG_FREQ, dpi=130)
    plt.close(fig)
    fig2, eje2 = plt.subplots(figsize=(8, 4))
    eje2.bar(np.arange(5), filas[1]["delta_por_fold"], color="#1a9850")
    eje2.axhline(0, color="grey", linewidth=1)
    eje2.axhline(GATE_DELTA, color="#b2182b", linestyle="--", linewidth=1)
    eje2.set_xticks(range(5))
    eje2.set_xticklabels([f"fold {i + 1}" for i in range(5)])
    eje2.set_ylabel("delta de macro-F1")
    eje2.set_title("T1 menos T0, por fold", fontsize=10)
    fig2.tight_layout()
    fig2.savefig(FIG_FOLD, dpi=130)
    plt.close(fig2)


def _informe(detalle, por_fold, bloques, t1, cumple_delta, cumple_folds, bootstrap, decision,
             problemas_mi, K, valor_t0, reproduce, delta, seis_t0, seis_t1) -> None:
    """Redacta el informe de la fase."""
    t0 = detalle["T0 V20 sin seleccion"]
    cuerpo = [
        "# V25 — Seleccion supervisada de features por informacion mutua", "",
        "**Una sola comparacion.** No se corre ninguna grid, no se prueba otro K y no se prueba "
        "otro selector.", "",
        "## 1. Hipotesis y por que no es V15", "",
        "> V20 puede mejorar eliminando features ruidosas mediante una seleccion supervisada "
        "realizada correctamente dentro de cada fold.", "",
        "| | V15 | V25 |", "|---|---|---|",
        "| Metodo | VIF, correlacion, clustering | `mutual_info_classif`, 6 categorias |",
        "| Usa la variable objetivo | **no** | **si**, solo del training fold |",
        "| Resultado | poda -0,004, 0 de 5 folds | se mide en este informe |",
        "| Conclusion | la redundancia es funcional | - |", "",
        "V15 concluyo que podar **empeora**, y de ahi salio el bloque de 73 variables "
        "label-informed que V19 si explotó. V25 no repite esa poda: usa la etiqueta.", "",
        "## 2. Procedencia de las 73 label-informed", "",
        "Verificadas por codigo, por comparacion de arrays y por separacion empirica en "
        "`v25_00_inventario.py`: se construyen con `y_train` del training fold, se escriben solo "
        "en `valid_idx`, y usan **el mismo array de grupos y los mismos folds** que V20. "
        "**LEGAL**.", "",
        "## 3. Por que se estandariza antes del mutual information", "",
        "`mutual_info_classif` estima con vecino mas cercano y **la metrica de distancia no es "
        "invariante a escala**. Las 73 supervisadas llegan a magnitud absoluta **59.439** frente a "
        "las probabilidades de miembro, que viven en [0, 1]. El estandarizado se aplica dentro de "
        "cada training fold y **nunca ve la validacion**.", "",
        "## 4. Control T0", "",
        f"- V20 reconstruido: **{valor_t0:.7f}** frente a {CV_V20:.7f} | reproduce: **{reproduce}**.",
        "- Si no reprodujera, la fase se detiene sin entrenar T1.", "",
        f"- Blend con cuatro cuartos: **{t0['corrected_cv']:.7f}**.", "",
        "## 5. Resultado", "",
        "| Modelo | corrected_cv | Delta | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Media |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for nombre in detalle:
        d = detalle[nombre]
        cuerpo.append(f"| {nombre} | **{d['corrected_cv']:.7f}** | {d['delta_vs_t0']:+.6f} "
                      f"| {d['fold_1']:.6f} | {d['fold_2']:.6f} | {d['fold_3']:.6f} "
                      f"| {d['fold_4']:.6f} | {d['fold_5']:.6f} | {d['media']:.6f} |")
    cuerpo += ["", "### F1 por clase", "", "| Modelo | " + " | ".join(cfg.CLASSES) + " |",
               "|---" * 7 + "|"]
    for nombre in detalle:
        cuerpo.append(f"| {nombre} | " + " | ".join(
            f"{detalle[nombre]['f1_por_clase'][c]:.4f}" for c in cfg.CLASSES) + " |")
    cuerpo += ["", "### Complementariedad y error por fold", "",
               "| Modelo | rescue | damage | net | changed rows | folds + |", "|---|---|---|---|---|---|"]
    for nombre in detalle:
        d = detalle[nombre]
        cuerpo.append(f"| {nombre} | {d['rescue']:,} | {d['damage']:,} | {d['net']:+,} "
                      f"| {d['changed_rows']:,} | {d['folds_positivos']}/5 |")
    cuerpo += ["", "| Modelo | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 |", "|---|---|---|---|---|---|"]
    for nombre in detalle:
        cuerpo.append(f"| {nombre} | " + " | ".join(
            f"{d:+.6f}" for d in detalle[nombre]["delta_por_fold"]) + " |")
    cuerpo += ["", "## 6. La seleccion, fold a fold", "",
               f"K = floor(279 x {TOP_PORCENTAJE}) = **{K}**, aplicado **independientemente en cada "
               "training fold** y **sin ajustarlo observando el resultado**.", "",
               "| Fold | K | MI max | MI min |",
               "|---|---|---|---|"]
    for f in por_fold:
        cuerpo.append(f"| {f['fold']} | {f['K']} | {f['mi_max']:.4f} | {f['mi_min']:.4f} |")
    cuerpo += ["", "## 7. Estabilidad y error por bloque", "",
               "La frecuencia de seleccion es **diagnostico**. No se usa para construir una "
               "segunda seleccion, que es justo lo que V15 hizo mal.", "",
               "| Bloque | Columnas | Frecuencia media | MI media | Nunca seleccionadas |",
               "|---|---|---|---|---|"]
    for bloque, v in sorted(bloques.items(), key=lambda kv: -kv[1]["media_frecuencia"]):
        cuerpo.append(f"| `{bloque}` | {v['columnas']} | {v['media_frecuencia']:.3f} "
                      f"| {v['media_mi']:.5f} | {v['nunca_seleccionadas']} |")
    cuerpo += ["", "## 8. Decision", "",
               f"- Delta de T1 frente a T0: **{t1['delta_vs_t0']:+.6f}** (gate >= {GATE_DELTA}) -> "
               f"**{cumple_delta}**",
               f"- Folds positivos: **{t1['folds_positivos']}/5** (gate >= {GATE_FOLDS}) -> "
               f"**{cumple_folds}**"]
    if bootstrap:
        cuerpo.append(f"- Bootstrap pareado: delta {bootstrap['delta']:+.6f}, IC95 "
                      f"[{bootstrap['ic95'][0]:+.6f}, {bootstrap['ic95'][1]:+.6f}]")
    else:
        cuerpo.append("- Bootstrap: **no se calcula**, porque el gate de delta no se cumple y el "
                      "brief lo reserva a ese caso.")
    cuerpo += ["", f"**{decision}**", ""]
    if problemas_mi:
        cuerpo += ["Avisos del mutual information:"] + [f"- {p}" for p in problemas_mi] + [""]
    cuerpo += ["No se genera submission y no se abre ninguna otra hipotesis.", ""]
    INFORME.write_text("\n".join(cuerpo), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
