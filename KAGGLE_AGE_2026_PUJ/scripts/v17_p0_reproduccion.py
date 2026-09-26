"""V17 · P0' — Reconciliacion de la referencia y verificacion inerte de A2 de V16.

Este script no entrena nada. Cumplen dos obligaciones del brief antes de que se pague un solo
minuto de CPU:

**1. Reconciliar 0,6004 contra 0,6005** (brief seccion 1). Los dos numeros aparecen en el
proyecto desde hace meses y no son el mismo numero redondeado: son **dos estimadores distintos** de
la misma magnitud. Uno es el macro-F1 de la confusion **pooled** (todas las filas a la vez, que es lo
que calcula ``metricas_globales``); el otro es la **media de los cinco macro-F1 por fold**. No
coinciden porque el macro-F1 de una confusion agregada no es la media de los macro-F1 de sus
partes: es el efecto Jensen aplicado a la media por clases. Aqui se cuantifica la diferencia con
diez decimales y se deja constancia de cual es el canonico y por que.

**2. Reproducir A2 de V16** (brief seccion 7). Se recalcula, desde los artefactos guardados, todo lo
que V16 publiko: macro-F1 global, las dos fronteras, los cinco folds y el intervalo de bootstrap
pareado. La comparacion es **inerte**: no se reentrena nada, se releen las matrices OOF que el
propio V16 guardo. Eso verifica la consistencia interna del resultado y de los artefactos, que es
lo que permite detectar que un informe y su JSON no cuadran. Lo que **no** verifica es que el
entrenamiento sea reproducible bit a bit, porque V16 no guardo las predicciones de su segunda
particion de 4 folds; esos numeros se citan del JSON de V16 y se marcan como tales.

Que no se reproduzca aqui significa que los artefactos de V16 estan mal, no que V16 este mal. La
reproduccion de entrenamiento, que si costs CPU, se decide aparte.

**Canonico para V17: 0,6005 en escala pooled.** El argumento no es aesthetico: el umbral de efecto
de D-042 se calibro contra controles permutados medidos en pooled (0,6000 a 0,6003), el IC95 del
criterio 3 vive en pooled porque ``bootstrap_diferencia`` remuestrea la confusion conjunta, y las
cinco fases modernas (V13 P1b, P3a, V14b, V15b, V16) miden su referencia en pooled. Usar 0,6004
regalaria +0,0001 de delta y rompería el emparejamiento con todo lo anterior.

Entregables: ``experiments/v17_p0_reproduccion.json`` y ``reports/v17_p0_reproduccion.md``.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from v13_p1b_fusion_tokens import (PARES, bootstrap_diferencia, metricas_frontera,  # noqa: E402
                                   metricas_globales)

from src import config as cfg  # noqa: E402
from src import cv as cvmod  # noqa: E402
from src import data as datos  # noqa: E402
from src.config import set_seed  # noqa: E402
from src.verificacion_numerica import autotest_numerico  # noqa: E402

INFORME = cfg.REPORTS / "v17_p0_reproduccion.md"
RESULTADO = cfg.EXPERIMENTS / "v17_p0_reproduccion.json"
#: Sensibilidad del estimador: la diferencia entre ambos valores es de 9,1e-5, y se redondea a
#: cuatro decimales para ver si basta con redondear o si de verdad son dos estimadores.
DECIMALES = 10


def main() -> int:
    """Reconcilia la referencia y verifica A2 de V16 desde los artefactos guardados."""
    autotest_numerico()
    set_seed()
    inicio = time.time()
    train = datos.load_train()
    y = train[cfg.TARGET].astype(str).to_numpy()
    clase = np.array([cfg.CLASSES.index(valor) for valor in y])
    clusters_ids = pd.read_parquet(cfg.FEATURES / "dup_clusters.parquet")["cluster"].to_numpy()[:len(y)]
    folds = cvmod.make_folds(y, groups=clusters_ids, n_splits=5, n_repeats=1, scheme="group")

    # ---------------------------------------------------------------- 1. reconciliacion
    referencia = np.load(cfg.FEATURES / "oof_ensemble_v12.npy")
    # `metricas_globales` redondea a cuatro decimales, y comparar ese valor redondeado con
    # la media de folds sin redondear subestima la diferencia. Aqui se calcula el pooled SIN
    # redondear, que es el que hace falta para cuantificar la discrepancia.
    pooled = float(f1_score(clase, np.argmax(referencia, axis=1), average="macro",
                           labels=list(range(len(cfg.CLASSES)))))
    por_fold = [float(f1_score(y[fold["val_idx"]],
                               np.array(cfg.CLASSES)[np.argmax(referencia[fold["val_idx"]], axis=1)],
                               average="macro", labels=cfg.CLASSES)) for fold in folds]
    media_folds = float(np.mean(por_fold))
    diferencia = media_folds - pooled
    print(f"[referencia] pooled {pooled:.{DECIMALES}f} | media de folds {media_folds:.{DECIMALES}f} | "
          f"diferencia {diferencia:+.10f}", flush=True)
    print(f"[referencia] es la misma magnitud redondeada? "
          f"{'si' if abs(diferencia) < 5e-5 else 'NO: son dos estimadores'}", flush=True)

    # ---------------------------------------------------------------- 2. A2 de V16
    registro_v16 = json.loads((cfg.EXPERIMENTS / "v16_arquitectura.json").read_text(encoding="utf-8"))
    verificaciones = []
    for arquitectura, archivo in (("promedio_fijo", "oof_ensemble_v12.npy"),
                                  ("peso_aprendido", "v16_oof_peso_aprendido.npy"),
                                  ("combinador_logistico", "v16_oof_combinador_logistico.npy")):
        matriz = np.load(cfg.FEATURES / archivo)
        metricas = metricas_globales(matriz, clase)
        fronteras = metricas_frontera(matriz, clase, PARES)
        propio = [float(f1_score(y[fold["val_idx"]],
                                  np.array(cfg.CLASSES)[np.argmax(matriz[fold["val_idx"]], axis=1)],
                                  average="macro", labels=cfg.CLASSES)) for fold in folds]
        publicado = (registro_v16["referencia"]["macro_f1"] if arquitectura == "promedio_fijo"
                     else registro_v16["arquitecturas"][arquitectura]["macro_f1"])
        delta = abs(metricas["macro_f1"] - publicado)
        coincide = bool(delta < 5e-5)
        verificaciones.append({
            "arquitectura": arquitectura, "macro_f1": round(metricas["macro_f1"], 6),
            "publicado": publicado, "coincide": coincide, "delta_absoluto": round(delta, 10),
            "18-24|25-34": fronteras["18-24|25-34"]["accuracy"],
            "25-34|35-49": fronteras["25-34|35-49"]["accuracy"],
            "por_fold": [round(v, 6) for v in propio],
        })
        print(f"[A2 verificacion] {arquitectura:24s} {metricas['macro_f1']} frente a publicado "
              f"{publicado} -> {'coincide' if coincide else 'NO COINCIDE'} | fronteras "
              f"{fronteras['18-24|25-34']['accuracy']} y {fronteras['25-34|35-49']['accuracy']}",
              flush=True)

    a2 = np.load(cfg.FEATURES / "v16_oof_combinador_logistico.npy")
    intervalo = bootstrap_diferencia(referencia, a2, clase, len(cfg.CLASSES))
    delta_a2 = metricas_globales(a2, clase)["macro_f1"] - pooled
    fronteras_a2 = metricas_frontera(a2, clase, PARES)
    fronteras_v12 = metricas_frontera(referencia, clase, PARES)
    todo_coincide = all(v["coincide"] for v in verificaciones)
    print(f"[A2] delta {delta_a2:+.4f} | IC95 [{intervalo[0]:+.4f}, {intervalo[1]:+.4f}] | "
          f"fronteras +{fronteras_a2['18-24|25-34']['accuracy'] - fronteras_v12['18-24|25-34']['accuracy']:+.4f} "
          f"y +{fronteras_a2['25-34|35-49']['accuracy'] - fronteras_v12['25-34|35-49']['accuracy']:+.4f}",
          flush=True)

    # ---------------------------------------------------------------- 3. bandas de ruido
    #: El umbral de efecto de D-042 se calibro contra los controles permutados de P1b, que se
    #: movieron en esta banda. Se citan del registro de V13, no se recalculan.
    banda_permutados = [0.6000, 0.6003, 0.6000]
    print(f"[banda de ruido] controles permutados de P1b: {banda_permutados} | gate exploratorio de "
          f"V17 +0,0015 y de adopcion +0,003, ambos por encima de la banda", flush=True)

    # ---------------------------------------------------------------- informe
    lineas = [
        "# V17 · P0' — Referencia reconciliada y A2 de V16 verificado", "",
        "## 1. La discrepancia 0,6004 contra 0,6005", "",
        "No son el mismo numero redondeado. Son **dos estimadores de la misma magnitud**:", "",
        f"- **Pooled** (macro-F1 de la confusion de todas las filas, que es `metricas_globales`): "
        f"**{pooled:.{DECIMALES}f}**, que redondea a **{pooled:.4f}**.",
        f"- **Media de los cinco folds**: **{media_folds:.{DECIMALES}f}**, que redondea a "
        f"**{media_folds:.4f}**.",
        f"- Diferencia: **{diferencia:+.10f}**, unas {abs(diferencia) / 5e-5:.0f} veces la unidad de "
        f"redondeo del ultimo digito. No se arregla redondeando.", "",
        "**Por que difieren.** El macro-F1 de una confusion agregada no es la media de los macro-F1 "
        "de las partes que la componen, porque cada clase entra con su propio peso en cada "
        "subconjunto. Es la desigualdad de Jensen aplicada a la media por clases.", "",
        "| Fold | macro-F1 |", "|---|---|",
    ]
    for indice, valor in enumerate(por_fold, start=1):
        lineas.append(f"| {indice} | {valor:.8f} |")
    lineas += ["", f"**Valor canonico de V17: {pooled:.4f} (pooled).** El motivo no es "
                  "estetico: el umbral de efecto de D-042 se calibro contra los controles permutados "
                  "de P1b, medidos en pooled; el IC95 del criterio 3 vive en pooled porque "
                  "`bootstrap_diferencia` remuestrea la confusion conjunta; y las cinco fases "
                  "modernas miden su referencia en pooled. Usar 0,6004 regalaria +0,0001 de delta y "
                  "romperia el emparejamiento con V13, V14, V15 y V16.", "",
              "**Regla operativa**: todo candidato de V17 se mide con `metricas_globales` (pooled). Si "
              "en un informe aparece 0,6004, es que se uso la media por fold y su delta no es "
              "comparable. Cuando se cite el historico hay que escribir el estimador.", "",
              "## 2. A2 de V16 verificado en colderte desde los artefactos", "",
              "| Arquitectura | macro-F1 recalculado | Publicado | Coincide | 18-24\\|25-34 | "
              "25-34\\|35-49 |", "|---|---|---|---|---|---|"]
    for verificacion in verificaciones:
        lineas.append(
            f"| {verificacion['arquitectura']} | {verificacion['macro_f1']} | "
            f"{verificacion['publicado']} | {'si' if verificacion['coincide'] else '**no**'} | "
            f"{verificacion['18-24|25-34']} | {verificacion['25-34|35-49']} |")
    lineas += ["", f"**Delta de A2 sobre la referencia pooled: {delta_a2:+.4f}** · IC95 pareado "
                  f"**[{intervalo[0]:+.4f}, {intervalo[1]:+.4f}]** · fronteras "
                  f"**+{fronteras_a2['18-24|25-34']['accuracy'] - fronteras_v12['18-24|25-34']['accuracy']:.4f}** "
                  f"y **+{fronteras_a2['25-34|35-49']['accuracy'] - fronteras_v12['25-34|35-49']['accuracy']:.4f}**.", "",
              "### Limite de esta verificacion", "",
              "Es **inerte**: relee las matrices OOF que V16 guardo y recalcula las metricas. "
              "Verifica que los artefactos y los informes de V16 sean coherentes entre si, que es lo "
              "que detecta un informe desalineado de su JSON. **No** verifica que el entrenamiento "
              "sea reproducible bit a bit, porque V16 no guardo las predicciones de su segunda "
              "particion de 4 folds; los numeros de esa particion se citan de "
              "`experiments/v16_arquitectura.json` y no se recalculan aqui.", "",
              f"**Veredicto: {'A2 se reproduce' if todo_coincide else 'A2 NO se reproduce: STOP'}**"
              f" sobre la evidencia inerte disponible.", "",
              "## 3. Bandas de ruido y umbrales de V17", "",
              f"- Controles permutados de P1b (V13), en pooled: {banda_permutados}. Son la banda que "
              f"calibro el umbral de D-042.",
              "- Gate exploratorio de V17: delta >= +0,0015 con >= 4/5 folds positivos.",
              "- Gate de adopcion: delta >= +0,003 con IC95 consistente, segunda particion positiva y "
              "mejora no concentrada en una sola fold.", ""]
    INFORME.write_text("\n".join(lineas) + "\n", encoding="utf-8")
    RESULTADO.write_text(json.dumps({
        "fase": "V17 · P0' reconciliacion de referencia y verificacion de A2",
        "fecha": "2026-09-26", "entrenamiento": "ninguno (inerte)",
        "referencia": {"pooled": round(pooled, DECIMALES), "pooled_redondeado": round(pooled, 4),
                       "media_folds": round(media_folds, DECIMALES),
                       "media_folds_redondeada": round(media_folds, 4),
                       "diferencia": round(diferencia, 10),
                       "canonico": round(pooled, 4), "estimador_canonico": "pooled",
                       "motivo": ("el umbral de D-042 y el IC95 del bootstrap viven en pooled, y las "
                                  "cinco fases modernas miden alli; usar la media por fold regalaria "
                                  "+0,0001 y romperia la comparabilidad"),
                       "por_fold": [round(v, 8) for v in por_fold]},
        "a2_verificado": verificaciones, "todo_coincide": todo_coincide,
        "a2": {"delta": round(delta_a2, 6), "bootstrap_ic95": [round(v, 6) for v in intervalo],
               "frontera_18_24_25_34": fronteras_a2["18-24|25-34"]["accuracy"],
               "frontera_25_34_35_49": fronteras_a2["25-34|35-49"]["accuracy"],
               "ganancia_frontera_18_24_25_34": round(
                   fronteras_a2["18-24|25-34"]["accuracy"] - fronteras_v12["18-24|25-34"]["accuracy"], 6),
               "ganancia_frontera_25_34_35_49": round(
                   fronteras_a2["25-34|35-49"]["accuracy"] - fronteras_v12["25-34|35-49"]["accuracy"], 6),
               "segunda_particion": registro_v16["segunda_particion"],
               "fuente_segunda_particion": "citada de experiments/v16_arquitectura.json; no recalculada"},
        "banda_ruido_permutados": banda_permutados,
        "segundos": round(time.time() - inicio, 1),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\ncanonico {pooled:.4f} (pooled) | A2 {'reproducida' if todo_coincide else 'NO reproducida'}"
          f" | delta {delta_a2:+.4f} | IC95 [{intervalo[0]:+.4f}, {intervalo[1]:+.4f}]")
    return 0 if todo_coincide else 1


if __name__ == "__main__":
    raise SystemExit(main())
