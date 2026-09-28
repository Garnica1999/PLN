"""V27 · H12 — Evidencia de fiabilidad por modelo y clase para el meta-modelo.

**Primer experimento de H12, y unico.** V20 congelado, una sola representacion nueva, sin grid y
sin segunda variante salvo evidencia concreta del resultado.

## La hipotesis

> Si la fiabilidad de cada miembro del ensemble depende de la clase de edad, puede el meta-modelo
> beneficiarse de conocer explicitamente esa fiabilidad?

## Novedad funcional, verificada antes de entrenar

Se buscaron con `grep` cuatro familias y ninguna es esto:

- `phase2_ponderacion.py` **nombra** que la fiabilidad de cada miembro depende del tipo de tweet,
  pero no la calcula ni la entrega al meta-modelo.
- `phase32_correccion_probabilistica.py` mide la fiabilidad del **top-1** de la decision final
  (D-029), que es reliability de la salida, no de cada miembro por clase.
- `v20_p0_fronteras.py` calcula reliability **por frontera** en su diagnostico, no por miembro.

**Ninguna construye una tabla fiabilidad(miembro, clase) como entrada del meta-modelo.** H12 es
nueva.

## La medida, predefinida y compacta

Para cada modelo base `i` con sus seis columnas y cada clase `j`:

    fiabilidad[i][j] = media de la columna i-clase j sobre las filas de clase j del TRAINING FOLD

Son **13 modelos con seis clases x 6 clases = 78 columnas**, y se anaden a las 279 existentes.
Quedan fuera dos bloques incompletos: `E028a_pares` tiene 5 columnas y `E015_ordinal` tiene 1, luego
no pueden cubrir las seis clases y no se inventan las que faltan.

La medida es la **media del bloque del modelo en las filas de esa clase**: para los miembros
`probabilidades` es la confianza media en esa clase, y para los `margenes` es el margen medio. No se
confunde una cosa con la otra y se documenta.

## Leakage

La media se calcula **exclusivamente sobre las filas del training fold**, y se aplica al fold de
validacion. La outer validacion no participa ni en el calculo ni en ninguna decision. Las
probabilidades de los miembros ya son OOF de su propio apilado, luego no hay informacion de la
etiqueta de la fila en su propia feature.

## Lo que no cambia

Los mismos folds, el mismo `HIERBA`, la misma cadena de cinco umbrales con maximo acumulado, el
mismo blend de cuatro cuartos y los mismos thresholds. El unico cambio son las 78 columnas
anadidas.

## Gate

`delta >= +0,003` y `4/5` folds positivos, el mismo gate de V26. Bootstrap solo si se supera.
"""

from __future__ import annotations

import json
import re
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

RESULTADO = cfg.EXPERIMENTS / "v27_01_h12_reliability.json"
FIG_CALOR = cfg.FIGURES / "v27_h12_mapa_fiabilidad.png"
FIG_DELTA = cfg.FIGURES / "v27_h12_delta_por_fold.png"
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


def modelos_completos(inventario: pd.DataFrame) -> dict[str, list[int]]:
    """Agrupa las columnas de evidencia por modelo y se queda con los que tienen las seis clases."""
    sub = inventario[inventario["bloque"].isin(["miembro_ensamble", "nbsvm_palabra"])]
    grupos: dict[str, dict[int, int]] = {}
    for _, fila in sub.iterrows():
        coincidencia = re.match(r"^(.*)_c(\d+)$", str(fila["nombre"]))
        if not coincidencia:
            continue
        grupos.setdefault(coincidencia.group(1), {})[int(coincidencia.group(2))] = int(fila["indice"])
    completos = {nombre: [mapa[c] for c in range(6)] for nombre, mapa in grupos.items()
                 if len(mapa) == 6}
    excluidos = {nombre: len(mapa) for nombre, mapa in grupos.items() if len(mapa) != 6}
    return completos, excluidos


def main() -> int:
    """Ejecuta H12 y decide."""
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
    completos, excluidos = modelos_completos(inventario)
    print(f"[H12] {matriz.shape[1]} features | {len(completos)} modelos con las 6 clases "
          f"-> {len(completos) * n_clases} columnas de fiabilidad | excluidos {excluidos}", flush=True)

    # ---------------------------------------------------------------- T0: control
    seis_t0 = reconstruye(np.load(CUMULADAS_V20))
    oof_t0 = 0.25 * logreg + 0.25 * mlp + 0.25 * arbol_v19 + 0.25 * seis_t0
    valor_t0 = macro(clase, oof_t0)
    reproduce = abs(valor_t0 - CV_V20) < 1e-6
    print(f"[T0] V20 {valor_t0:.7f} | reproduce: {reproduce}", flush=True)
    if not reproduce:
        print("[STOP] V20 no reproduce. No se ejecuta H12.")
        return 1

    # ---------------------------------------------------------------- T1: fiabilidad por modelo y clase
    acumuladas_t1 = np.zeros((len(y), n_clases - 1))
    tablas_fiabilidad: list[pd.DataFrame] = []
    for fold in folds:
        validacion = fold["val_idx"]
        entrenamiento = np.concatenate([o["val_idx"] for o in folds
                                        if o["fold"] != fold["fold"]])
        # La tabla de fiabilidad se calcula SOLO con las filas del training fold.
        filas_tabla, nombres_tabla = [], []
        for modelo, columnas in completos.items():
            for c in range(n_clases):
                mascara = clase[entrenamiento] == c
                filas_tabla.append(float(matriz[entrenamiento[mascara], columnas[c]].mean()))
                nombres_tabla.append(f"fiab_{modelo}_{cfg.CLASSES[c]}")
        tabla = np.tile(np.array(filas_tabla, dtype=float), (len(matriz), 1))
        x_entrenamiento = np.hstack([matriz[entrenamiento], tabla[entrenamiento]])
        x_validacion = np.hstack([matriz[validacion], tabla[validacion]])
        for k in range(1, n_clases):
            objetivo = (clase <= k - 1).astype(int)
            modelo_o = lgb.LGBMClassifier(**HIERBA)
            modelo_o.fit(x_entrenamiento, objetivo[entrenamiento])
            acumuladas_t1[validacion, k - 1] = modelo_o.predict_proba(x_validacion)[:, 1]
        tablas_fiabilidad.append(pd.DataFrame(
            {"modelo": [n.rsplit("_", 1)[0] for n in nombres_tabla],
             "clase": [n.rsplit("_", 1)[1] for n in nombres_tabla],
             "fiabilidad": filas_tabla, "fold": int(fold["fold"]) + 1}))
        print(f"    fold {fold['fold']}: fiabilidad de {len(nombres_tabla)} pares lista", flush=True)

    tabla_fiabilidad = pd.concat(tablas_fiabilidad, ignore_index=True)
    tabla_fiabilidad.to_csv(cfg.REPORTS / "v27_h12_fiabilidad.csv", index=False)
    seis_t1 = reconstruye(acumuladas_t1)
    oof_t1 = 0.25 * logreg + 0.25 * mlp + 0.25 * arbol_v19 + 0.25 * seis_t1
    valor_t1 = macro(clase, oof_t1)
    print(f"[T1] {valor_t1:.7f} | delta {valor_t1 - valor_t0:+.6f}", flush=True)

    # ---------------------------------------------------------------- metricas
    filas, detalle = [], {}
    for nombre, prob in (("T0 V20 control", oof_t0), ("T1 con fiabilidad por modelo y clase", oof_t1)):
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

    # ---------------------------------------------------------------- gate
    t1 = detalle["T1 con fiabilidad por modelo y clase"]
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
        decision, estado = "H12 CLOSED", "CLOSED"
    elif not cumple_delta:
        decision, estado = "H12 MARGINAL", "MARGINAL"
    elif cumple_folds and not bootstrap["ic95_incluye_cero"]:
        decision, estado = "H12 PROMISING (sin submission)", "PROMISING"
    else:
        decision, estado = "H12 MARGINAL", "MARGINAL"
    print(f"[gate] delta {t1['delta_vs_V20']:+.6f} (>= {GATE_DELTA}: {cumple_delta}) | "
          f"folds {t1['folds_positivos']}/5 (>= {GATE_FOLDS}: {cumple_folds})", flush=True)
    print(f"[decision] {decision}", flush=True)

    # ---------------------------------------------------------------- mecanismo
    # ¿La fiabilidad varia entre modelos y entre clases? Si es casi plana, la feature no puede
    # discriminar y el resultado negativo esta explicado antes de mirar la metrica.
    amplitud_por_clase = tabla_fiabilidad.groupby("clase")["fiabilidad"].agg(["min", "max", "std"])
    rango_por_modelo = (tabla_fiabilidad.groupby("modelo")["fiabilidad"]
                         .agg(["min", "max"]).assign(rango=lambda d: d["max"] - d["min"]))
    mecanismo = {
        "rango_global": round(float(rango_por_modelo["rango"].max()), 4),
        "modelo_mas_variable": str(rango_por_modelo["rango"].idxmax()),
        "desviacion_tipica_por_clase": {str(k): round(float(v), 4) for k, v
                                        in amplitud_por_clase["std"].items()},
        "distincion_entre_clases": round(float(amplitud_por_clase["max"].max()
                                                - amplitud_por_clase["min"].min()), 4),
    }
    print(f"[mecanismo] rango de fiabilidad entre modelos {mecanismo['rango_global']} "
          f"(el mas variable: {mecanismo['modelo_mas_variable']}) | distincion entre clases "
          f"{mecanismo['distincion_entre_clases']}", flush=True)

    _figuras(tabla_fiabilidad, filas)
    pd.DataFrame([{"ID": "H12-T0" if f["modelo"].startswith("T0") else "V27-01",
                   **{k: v for k, v in f.items()
                      if k not in {"f1_por_clase", "complementariedad_por_clase", "delta_por_fold"}}}
                  for f in filas]).to_csv(cfg.REPORTS / "v26_master_results.csv", mode="a",
                                           header=False, index=False)
    RESULTADO.write_text(json.dumps({
        "fase": "V27 01 H12 fiabilidad por modelo y clase", "fecha": "2026-09-28",
        "hipotesis": "el meta-modelo se beneficia de conocer explicitamente la fiabilidad de cada "
                     "miembro y para cada clase de edad",
        "novedad_funcional": ("ninguna de las tres familias existentes construye una tabla "
                              "fiabilidad(miembro, clase): phase32 mide la del top-1 de la "
                              "decision final (D-029), v20_p0_fronteras la de cada frontera, y "
                              "phase2 solo la nombra como objetivo"),
        "medida": "media de la columna del modelo y clase sobre las filas de esa clase del "
                  "training fold",
        "columnas_anadidas": int(len(completos) * n_clases),
        "modelos_incluidos": sorted(completos), "modelos_excluidos": excluidos,
        "modelos_excluidos_motivo": "no cubren las seis clases, luego no pueden aportar fiabilidad "
                                    "de todas; no se inventan las que faltan",
        "control_t0": {"corrected_cv": round(valor_t0, 7), "reproducido": reproduce},
        "candidatos": detalle, "delta": round(t1["delta_vs_V20"], 6),
        "mecanismo": mecanismo,
        "gate": {"delta_minimo": GATE_DELTA, "folds_minimos": GATE_FOLDS,
                 "cumple_delta": cumple_delta, "cumple_folds": cumple_folds,
                 "bootstrap": bootstrap},
        "decision": decision, "estado": estado, "submission_generada": False,
        "segundos": round(time.time() - inicio, 1),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


def _figuras(tabla: pd.DataFrame, filas: list[dict]) -> None:
    """Mapa de fiabilidad por modelo y clase, y delta por fold."""
    fig, ejes = plt.subplots(1, 2, figsize=(15, 6))
    eje = ejes[0]
    media = tabla.pivot_table(index="modelo", columns="clase", values="fiabilidad", aggfunc="mean")
    imagen = eje.imshow(media.values, aspect="auto", cmap="viridis")
    eje.set_xticks(range(len(media.columns)))
    eje.set_xticklabels(media.columns, fontsize=8)
    eje.set_yticks(range(len(media.index)))
    eje.set_yticklabels(media.index, fontsize=6)
    eje.set_title("Fiabilidad media por modelo y clase (agregada sobre folds)", fontsize=10)
    fig.colorbar(imagen, ax=eje, fraction=0.02)
    eje = ejes[1]
    t0, t1 = filas[0], filas[1]
    eje.bar(np.arange(5) - 0.2, t0["delta_por_fold"], 0.4, label="T0", color="#999999")
    eje.bar(np.arange(5) + 0.2, t1["delta_por_fold"], 0.4, label="T1 fiabilidad", color="#1a9850")
    eje.axhline(0, color="grey", linewidth=1)
    eje.axhline(GATE_DELTA, color="black", linestyle="--", linewidth=1)
    eje.set_xticks(range(5))
    eje.set_xticklabels([f"fold {i + 1}" for i in range(5)])
    eje.set_ylabel("delta de macro-F1 frente a T0")
    eje.set_title(f"Delta {t1['delta_vs_V20']:+.6f}, {t1['folds_positivos']}/5", fontsize=10)
    eje.legend(fontsize=8)
    fig.suptitle("H12: fiabilidad por modelo y clase para el meta-modelo", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG_DELTA, dpi=130)
    plt.close(fig)
    fig2, eje2 = plt.subplots(figsize=(9, 6))
    imagen2 = eje2.imshow(media.values, aspect="auto", cmap="viridis")
    eje2.set_xticks(range(len(media.columns)))
    eje2.set_xticklabels(media.columns, fontsize=8)
    eje2.set_yticks(range(len(media.index)))
    eje2.set_yticklabels(media.index, fontsize=6)
    eje2.set_title("Mapa de fiabilidad", fontsize=10)
    fig2.colorbar(imagen2, ax=eje2, fraction=0.02)
    fig2.tight_layout()
    fig2.savefig(FIG_CALOR, dpi=130)
    plt.close(fig2)


if __name__ == "__main__":
    raise SystemExit(main())
