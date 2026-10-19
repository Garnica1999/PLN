#!/usr/bin/env python
"""
P97-01 (H97-A): reglas de decision condicionadas por segmento, nested.

Pregunta: un umbral por (etiqueta, cuartil de longitud) elegido SOLO en el
train del pliegue supera a la regla global anidada en las mismas condiciones?

Pre-registro: reports/p97_independent_research.md, seccion 5 (H97-A).

No entrena; usa el OOF canonico guardado (features/oof_v03_final.npz) y los
grupos del protocolo. Compara tres reglas sobre las mismas 1483 filas:
  R0 frozen          : umbrales V03 [0.44, 0.51, 0.46]
  R1 global_nested   : un umbral por etiqueta, elegido en los otros pliegues
  R2 segmento_nested : umbral por etiqueta y cuartil de longitud (fronteras
                       calculadas en el train del pliegue)
  R3/R4 descriptivos : mencion (2) y hashtag (2)

Salidas: features/p97_01_pred_segmentos.npz, reports/p97_01_reglas_por_segmento.json
y .md, figura reports/figuras/p97_01_segmentos.png.

Uso:
    /home/carlos/venv/bin/python experiments/p97_01_reglas_por_segmento.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from src import data as data_module  # noqa: E402
from src.evaluation import metrica_macro, metricas_multietiqueta  # noqa: E402

SAL_NPZ = RAIZ / "features" / "p97_01_pred_segmentos.npz"
SAL_JSON = RAIZ / "reports" / "p97_01_reglas_por_segmento.json"
SAL_MD = RAIZ / "reports" / "p97_01_reglas_por_segmento.md"
SAL_FIG = RAIZ / "reports" / "figuras" / "p97_01_segmentos.png"

UMBRALES_V03 = np.array([0.44, 0.51, 0.46])
REJILLA = np.round(np.arange(0.20, 0.8001, 0.01), 2)
SEMILLA = 20260925


def f1_por_etiqueta(y: np.ndarray, pred: np.ndarray) -> list[float]:
    """F1 binaria de la clase positiva por etiqueta."""
    return [float(f1_score(y[:, i], pred[:, i], zero_division=0)) for i in range(3)]


def elegir_umbral_global(y: np.ndarray, proba: np.ndarray, filas: np.ndarray,
                         etiqueta: int) -> float:
    """Umbral que maximiza la F1 de una etiqueta en las filas dadas."""
    mejor = (0.5, -1.0)
    for t in REJILLA:
        f1 = f1_score(y[filas, etiqueta], (proba[filas, etiqueta] >= t).astype(int),
                      zero_division=0)
        if f1 > mejor[1]:
            mejor = (float(t), float(f1))
    return mejor[0]


def aplicar_por_segmento(proba: np.ndarray, segmentos: np.ndarray,
                         umbrales: dict[int, dict[int, float]]) -> np.ndarray:
    """Predice con umbrales por (etiqueta, segmento)."""
    n = proba.shape[0]
    pred = np.zeros((n, 3), dtype=int)
    for etiqueta in range(3):
        for seg in np.unique(segmentos).tolist():
            filas = np.flatnonzero(segmentos == seg)
            t = umbrales[etiqueta].get(int(seg), 0.5)
            pred[filas, etiqueta] = (proba[filas, etiqueta] >= t).astype(int)
    return pred


def bootstrap_clusters_delta(y: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray,
                             grupos: np.ndarray, repeticiones: int = 2000) -> dict:
    """IC95 del delta pooled (a - b) remuestreando clusters (unidad = grupo)."""
    generador = np.random.default_rng(SEMILLA)
    indices_por_cluster: dict[int, np.ndarray] = {}
    for valor in np.unique(grupos):
        indices_por_cluster[int(valor)] = np.flatnonzero(grupos == valor)
    claves = np.array(sorted(indices_por_cluster))
    n_clusters = len(claves)
    diferencias = np.empty(repeticiones)
    for r in range(repeticiones):
        muestra = generador.integers(0, n_clusters, n_clusters)
        idx = np.concatenate([indices_por_cluster[int(claves[i])] for i in muestra])
        diferencias[r] = metrica_macro(y[idx], pred_a[idx]) - metrica_macro(y[idx], pred_b[idx])
    cuantiles = np.quantile(diferencias, [0.025, 0.975])
    return {"ic95": [round(float(cuantiles[0]), 6), round(float(cuantiles[1]), 6)],
            "pct_mejora": round(float((diferencias > 0).mean()), 6),
            "repeticiones": repeticiones}


def matrices_correccion(y: np.ndarray, pred: np.ndarray, referencia: np.ndarray) -> dict:
    """Salvage/damage/neto por celda (fila, etiqueta) frente a la referencia."""
    aciertos_ref = (referencia == y)
    aciertos_nuevo = (pred == y)
    rescatadas = int((~aciertos_ref & aciertos_nuevo).sum())
    rotas = int((aciertos_ref & ~aciertos_nuevo).sum())
    return {"rescatadas": rescatadas, "rotas": rotas, "neto": rescatadas - rotas}


def main() -> None:
    """Ejecuta el pre-registro H97-A."""
    for ruta in (SAL_NPZ, SAL_JSON, SAL_MD, SAL_FIG):
        if ruta.exists():
            raise SystemExit(f"Ya existe {ruta}; no se sobrescribe.")
    inicio = time.perf_counter()
    z = np.load(RAIZ / "features" / "oof_v03_final.npz", allow_pickle=True)
    zb = np.load(RAIZ / "features" / "v2_oof_base_corrected.npz", allow_pickle=True)
    ids = z["ids"].astype(np.int64)
    y = z["y_true"].astype(int)
    proba = z["proba"].astype(float)
    pliegues = z["pliegues"].astype(int)
    assert np.array_equal(ids, zb["ids"].astype(np.int64))
    grupos = zb["grupos"].astype(np.int64)

    train = data_module.read_train()
    mapa = train.assign(_id=train["id"].astype(int)).set_index("_id")
    texto = mapa.loc[ids, "text"]
    n_tokens = texto.str.split().str.len().to_numpy()
    tiene_mencion = texto.str.contains("@", regex=False).to_numpy().astype(int)
    tiene_hashtag = texto.str.contains("#", regex=False).to_numpy().astype(int)

    folds = sorted(np.unique(pliegues).tolist())

    # R0 frozen
    pred_frozen = (proba >= UMBRALES_V03).astype(int)

    # R1 global nested
    pred_global = np.zeros_like(pred_frozen)
    umbrales_global: dict[int, dict[int, float]] = {}
    for k in folds:
        filas_train = np.flatnonzero(pliegues != k)
        filas_val = np.flatnonzero(pliegues == k)
        umbrales_global[k] = {}
        for etiqueta in range(3):
            t = elegir_umbral_global(y, proba, filas_train, etiqueta)
            umbrales_global[k][etiqueta] = t
            pred_global[filas_val, etiqueta] = (proba[filas_val, etiqueta] >= t).astype(int)

    # R2 segmento nested (cuartiles de longitud con fronteras del train del pliegue)
    pred_seg = np.zeros_like(pred_frozen)
    umbrales_seg: dict[int, dict[int, dict[int, float]]] = {}
    fronteras_por_k: dict[int, list[float]] = {}
    for k in folds:
        filas_train = np.flatnonzero(pliegues != k)
        filas_val = np.flatnonzero(pliegues == k)
        fronteras = [float(np.quantile(n_tokens[filas_train], q))
                     for q in (0.25, 0.50, 0.75)]
        fronteras_por_k[k] = fronteras
        seg_train = np.digitize(n_tokens[filas_train], fronteras)
        seg_val = np.digitize(n_tokens[filas_val], fronteras)
        umbrales_seg[k] = {}
        for etiqueta in range(3):
            umbrales_seg[k][etiqueta] = {}
            for seg in np.unique(seg_train).tolist():
                sub = filas_train[seg_train == seg]
                t = elegir_umbral_global(y, proba, sub, etiqueta)
                umbrales_seg[k][etiqueta][int(seg)] = t
            for seg in np.unique(seg_val).tolist():
                sub = filas_val[seg_val == seg]
                t = umbrales_seg[k][etiqueta].get(int(seg))
                if t is None:
                    t = umbrales_seg[k][etiqueta][int(min(umbrales_seg[k][etiqueta]))]
                pred_seg[sub, etiqueta] = (proba[sub, etiqueta] >= t).astype(int)

    # R3/R4 descriptivos: mencion y hashtag (por etiqueta, nested)
    pred_men = np.zeros_like(pred_frozen)
    pred_hash = np.zeros_like(pred_frozen)
    for k in folds:
        filas_train = np.flatnonzero(pliegues != k)
        filas_val = np.flatnonzero(pliegues == k)
        for etiqueta in range(3):
            for segmentos, destino, val_seg in (
                    (tiene_mencion, pred_men, tiene_mencion[filas_val]),
                    (tiene_hashtag, pred_hash, tiene_hashtag[filas_val])):
                for seg in (0, 1):
                    sub = filas_train[segmentos[filas_train] == seg]
                    if len(sub) == 0:
                        continue
                    t = elegir_umbral_global(y, proba, sub, etiqueta)
                    filas = filas_val[val_seg == seg]
                    destino[filas, etiqueta] = (proba[filas, etiqueta] >= t).astype(int)

    reglas = {"frozen": pred_frozen, "global_nested": pred_global,
              "segmento_nested": pred_seg, "mencion_nested": pred_men,
              "hashtag_nested": pred_hash}
    resumen: dict[str, dict] = {}
    for nombre, pred in reglas.items():
        metrica = metricas_multietiqueta(y, pred)
        por_pliegue = [
            float(metricas_multietiqueta(y[pliegues == k], pred[pliegues == k])
                  ["f1_media_columnas"]) for k in folds]
        resumen[nombre] = {
            "corrected_cv": metrica["f1_media_columnas"],
            "f1_por_etiqueta": metrica["f1_por_etiqueta"],
            "f1_pliegues": por_pliegue,
            "std_pliegues": float(np.std(por_pliegue, ddof=1)),
            "correccion_vs_frozen": matrices_correccion(y, pred, pred_frozen),
        }

    delta_principal = (resumen["segmento_nested"]["corrected_cv"]
                       - resumen["global_nested"]["corrected_cv"])
    delta_vs_frozen = (resumen["segmento_nested"]["corrected_cv"]
                       - resumen["frozen"]["corrected_cv"])
    pliegues_mejorados_global = int(np.sum(
        np.array(resumen["segmento_nested"]["f1_pliegues"])
        > np.array(resumen["global_nested"]["f1_pliegues"]) + 1e-12))
    pliegues_mejorados_frozen = int(np.sum(
        np.array(resumen["segmento_nested"]["f1_pliegues"])
        > np.array(resumen["frozen"]["f1_pliegues"]) + 1e-12))

    boot_global = bootstrap_clusters_delta(y, pred_seg, pred_global, grupos)
    boot_frozen = bootstrap_clusters_delta(y, pred_seg, pred_frozen, grupos)

    veredicto = ("SENAL" if (delta_principal > 0 and pliegues_mejorados_global >= 3
                             and boot_global["ic95"][0] > 0) else "FALSIFICADA")
    resultado = {
        "experiment_id": "P97-01",
        "hipotesis": "H97-A",
        "fecha": "2026-10-13",
        "protocolo": "nested por pliegues canonicos; umbrales elegidos solo en el train "
                     "del pliegue; cuartiles de longitud con fronteras del train",
        "n": int(len(ids)),
        "reglas": resumen,
        "comparaciones": {
            "segmento_vs_global_nested": {
                "delta": delta_principal,
                "pliegues_mejorados": pliegues_mejorados_global,
                "bootstrap_clusters": boot_global},
            "segmento_vs_frozen": {
                "delta": delta_vs_frozen,
                "pliegues_mejorados": pliegues_mejorados_frozen,
                "bootstrap_clusters": boot_frozen},
        },
        "fronteras_longitud_por_pliegue": fronteras_por_k,
        "umbrales_global_por_pliegue": umbrales_global,
        "umbrales_segmento_por_pliegue": umbrales_seg,
        "veredicto_pre_registrado": veredicto,
        "tiempo_seg": round(time.perf_counter() - inicio, 2),
    }
    np.savez_compressed(SAL_NPZ, ids=ids, y_true=y, grupos=grupos, pliegues=pliegues,
                        pred_frozen=pred_frozen, pred_global_nested=pred_global,
                        pred_segmento_nested=pred_seg, pred_mencion_nested=pred_men,
                        pred_hashtag_nested=pred_hash)
    SAL_JSON.write_text(json.dumps(resultado, indent=2, ensure_ascii=True) + "\n",
                        encoding="utf-8")

    lineas = [
        "# P97-01 (H97-A): reglas de decision condicionadas por segmento",
        "",
        "Pre-registro en `p97_independent_research.md` seccion 5. Sin entrenamiento: "
        "evaluacion nested sobre el OOF canonico (1483 filas, pliegues 1..5).",
        "",
        "| Regla | corrected_cv | delta vs frozen | pliegues mejorados vs frozen |",
        "|---|---|---|---|",
    ]
    for nombre in ("frozen", "global_nested", "segmento_nested", "mencion_nested",
                   "hashtag_nested"):
        r = resumen[nombre]
        delta = r["corrected_cv"] - resumen["frozen"]["corrected_cv"]
        mejorados = int(np.sum(np.array(r["f1_pliegues"])
                               > np.array(resumen["frozen"]["f1_pliegues"]) + 1e-12))
        lineas.append(f"| {nombre} | {r['corrected_cv']:.10f} | {delta:+.6f} | "
                      f"{mejorados}/5 |")
    lineas += [
        "",
        f"**Comparacion principal (segmento - global_nested)**: delta "
        f"{delta_principal:+.6f}, pliegues {pliegues_mejorados_global}/5, IC95 por "
        f"clusters {boot_global['ic95']}, porcentaje de remuestreos positivos "
        f"{boot_global['pct_mejora']}.",
        "",
        f"**Segmento vs frozen**: delta {delta_vs_frozen:+.6f}, pliegues "
        f"{pliegues_mejorados_frozen}/5, IC95 {boot_frozen['ic95']}.",
        "",
        f"**Veredicto pre-registrado: {veredicto}** "
        "(SENAL exige delta > 0, >= 3/5 pliegues y IC95 sin cero contra la regla "
        "global anidada).",
        "",
        "Nota: la comparacion contra `frozen` mezcla condicionamiento y "
        "re-optimizacion; la comparacion principal aisla el efecto de condicionar.",
    ]
    SAL_MD.write_text("\n".join(lineas) + "\n", encoding="utf-8")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ejes = plt.subplots(1, 2, figsize=(11, 4.2))
    nombres = list(reglas.keys())
    valores = [resumen[n]["corrected_cv"] for n in nombres]
    ejes[0].bar(nombres, valores, color=["#444444", "#1f77b4", "#d62728", "#7f7f7f",
                                         "#7f7f7f"])
    ejes[0].set_ylim(0.770, 0.782)
    ejes[0].set_ylabel("corrected_cv (pooled)")
    ejes[0].set_title("Reglas de decision (nested sobre OOF)")
    ejes[0].tick_params(axis="x", rotation=20)
    ancho = 0.25
    xs = np.arange(len(folds))
    for i, regla in enumerate(("frozen", "global_nested", "segmento_nested")):
        ejes[1].bar(xs + (i - 1) * ancho, resumen[regla]["f1_pliegues"], ancho,
                    label=regla)
    ejes[1].set_xticks(xs, [f"p{k}" for k in folds])
    ejes[1].set_title("F1 por pliegue")
    ejes[1].legend(fontsize=8)
    fig.tight_layout()
    SAL_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(SAL_FIG, dpi=130)
    plt.close(fig)

    print("P97-01 H97-A: ejecutado")
    for nombre in ("frozen", "global_nested", "segmento_nested"):
        r = resumen[nombre]
        print(f"  {nombre:18s} corrected_cv={r['corrected_cv']:.10f} "
              f"std={r['std_pliegues']:.5f} "
              f"salvage={r['correccion_vs_frozen']['rescatadas']} "
              f"damage={r['correccion_vs_frozen']['rotas']}")
    print(f"  principal: delta={delta_principal:+.6f} | pliegues "
          f"{pliegues_mejorados_global}/5 | IC clusters {boot_global['ic95']}")
    print(f"  veredicto: {veredicto}")
    print(f"  salidas: {SAL_JSON.name}, {SAL_MD.name}, figuras/{SAL_FIG.name}, "
          f"{SAL_NPZ.name}")


if __name__ == "__main__":
    main()
