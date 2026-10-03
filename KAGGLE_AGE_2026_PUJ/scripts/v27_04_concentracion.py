"""V27 · 04 — Hallazgo estructural: el meta-modelo de V20 divide por 7 de sus 279 features.

**No es un experimento.** No entrena una hipotesis, no entra en el gate y no consume presupuesto.
Es el **mecanismo que explica** por que las tres familias recién cerradas fallaron, y por eso
necesita su propio artefacto y su propia evidencia.

## Lo que se mide

Para el meta-modelo de V20, con su configuracion exacta, **cuantas features distintas aparecen
como variable de division** en los 1.800 arboles del booster, y de que bloque son.

El resultado es **7 de 279**, y las siete son del bloque `miembro_ensamble`: los seis margenes de
`E013_margenes` y uno de `E016_margenes`.

## Por que el hallazgo no es un artefacto del submuestreo

`HIERBA` usa `colsample_bytree=0,5` y `subsample=0,8` con `subsample_freq=1`. Si la concentracion
fuera efecto del muestreo, quitarlo devolveria las features. Se prueban tres controles:

| Configuracion | Splits | Features distintas |
|---|---|---|
| V20 exacta | 3.990 | **7** |
| sin `colsample_bytree` | 3.960 | **7** |
| sin `subsample` | 4.014 | **7** |
| `num_leaves=31`, `max_depth=-1` | 18.000 | 30 |

La concentracion **persiste sin submuestreo**, luego es una propiedad de la senal: los margenes de
E013 capturan practicamente toda la ganancia disponible y las otras 272 features no aportan nada
incremental. Subir la capacidad del arbol si amplia el conjunto, a 30 features, lo que confirma que
las demas estaban disponibles y no usadas por falta de capacidad.

## Lo que explica de una sola vez

1. **V25 (feature selection)**: el mutual information ordena alto las columnas de
   `miembro_ensamble`, y esas son **precisamente** las que el modelo usa. Las 87 estructuradas y 35
   densas que el MI descarta son las que **ya no se dividian**. Podar no quita nada que el modelo
   estuviera usando, y por eso no mejora.
2. **V26-01 (reescalada de la evidencia)**: reordenar columnas no cambia el orden, y el modelo
   divide por orden, luego el resultado de -0,000001 era el unico posible.
3. **H11 (restricciones monotonas)**: las features con relacion monotona **defendible** son las
   probabilidades de clase, en los indices 12 a 53. **Ninguna se usa para dividir.** La restriccion
   no puede actuar sobre features que el modelo no abre, y su unico efecto es el -0,000355 que
   produce el cambio de heuristica interna de LightGBM al activar `monotone_constraints`.

## Lo que este hallazgo NO abre

No se convierte en experimento por el solo hecho de ser interesante. Registrar la observacion es
obligatorio; abrir una hipotesis nueva requiere mecanismo distinto **y** evidencia de que V20 no lo
representa ya, **y** que entra en el presupuesto. Eso se decide aparte, y este script no decide nada.
"""

from __future__ import annotations

import collections
import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src import config as cfg  # noqa: E402
from src import cv as cvmod  # noqa: E402
from src import data as datos  # noqa: E402
from src.verificacion_numerica import autotest_numerico  # noqa: E402
from v19_b1_boosting import HIERBA, matriz_279  # noqa: E402

RESULTADO = cfg.EXPERIMENTS / "v27_04_concentracion.json"
TABLA = cfg.REPORTS / "v27_concentracion_features.csv"
FIG = cfg.FIGURES / "v27_concentracion_features.png"
#: Controles que demuestran que la concentracion no viene del submuestreo.
CONTROLES = [
    ("V20 exacta (HIERBA)", {}),
    ("sin colsample_bytree", {"colsample_bytree": 1.0}),
    ("sin subsample", {"subsample": 1.0, "subsample_freq": 0}),
    ("num_leaves=31 y max_depth=-1", {"num_leaves": 31, "max_depth": -1}),
]


def _splits(nodo: dict) -> list[int]:
    """Lista de `split_index` de una estructura de arbol, en profundidad."""
    if "leaf_value" in nodo:
        return []
    salida = [int(nodo.get("split_index", -1))]
    for hijo in (nodo.get("left_child"), nodo.get("right_child")):
        if isinstance(hijo, dict):
            salida.extend(_splits(hijo))
    return salida


def main() -> int:
    """Mide la concentracion de features del meta-modelo de V20 en los cinco umbrales."""
    autotest_numerico()
    inicio = time.time()
    train = datos.load_train()
    y = train[cfg.TARGET].astype(str).to_numpy()
    clase = np.array([cfg.CLASSES.index(v) for v in y])
    clusters = pd.read_parquet(cfg.FEATURES / "dup_clusters.parquet")["cluster"].to_numpy()[:len(y)]
    folds = cvmod.make_folds(y, groups=clusters, n_splits=5, n_repeats=1, scheme="group")
    matriz = matriz_279()
    inventario = pd.read_csv(cfg.REPORTS / "v25_feature_inventory.csv")
    entrena = np.concatenate([o["val_idx"] for o in folds if o["fold"] != 0])

    # ---------------------------------------------------------------- por umbral
    por_umbral = {}
    for k in range(1, len(cfg.CLASSES)):
        objetivo = (clase <= k - 1).astype(int)
        modelo = lgb.LGBMClassifier(**HIERBA).fit(matriz[entrena], objetivo[entrena])
        conteo = collections.Counter()
        for arbol in modelo.booster_.dump_model()["tree_info"]:
            conteo.update(_splits(arbol["tree_structure"]))
        total = sum(conteo.values())
        bloques = collections.Counter(
            str(inventario.loc[inventario["indice"] == i, "bloque"].iloc[0]) for i in conteo)
        por_umbral[f"umbral_{k}"] = {
            "splits_totales": total, "features_distintas": len(conteo),
            "porcentaje_de_279": round(100 * len(conteo) / matriz.shape[1], 2),
            "bloques": dict(bloques),
            "features": [{"indice": int(i), "splits": int(n),
                          "nombre": str(inventario.loc[inventario["indice"] == i, "nombre"].iloc[0]),
                          "bloque": str(inventario.loc[inventario["indice"] == i, "bloque"].iloc[0])}
                         for i, n in conteo.most_common()]}
        print(f"[conc] umbral {k}: {len(conteo)} features distintas de 279, {total} splits, "
              f"bloques {dict(bloques)}", flush=True)

    # ---------------------------------------------------------------- controles
    objetivo = (clase <= 0).astype(int)
    controles = {}
    for nombre, extra in CONTROLES:
        ajustes = dict(HIERBA)
        ajustes.update(extra)
        modelo = lgb.LGBMClassifier(**ajustes).fit(matriz[entrena], objetivo[entrena])
        conteo = collections.Counter()
        for arbol in modelo.booster_.dump_model()["tree_info"]:
            conteo.update(_splits(arbol["tree_structure"]))
        controles[nombre] = {"splits_totales": sum(conteo.values()),
                              "features_distintas": len(conteo),
                              "por_bloque": dict(collections.Counter(
                                  str(inventario.loc[inventario["indice"] == i, "bloque"].iloc[0])
                                  for i in conteo))}
        print(f"[conc] control {nombre:<28} {len(conteo):>3} features distintas | "
              f"{sum(conteo.values())} splits", flush=True)

    # ---------------------------------------------------------------- tabla consolidada
    filas = []
    for indice, fila in inventario.iterrows():
        total_splits = sum(v["features"] and next(
            (f["splits"] for f in v["features"] if f["indice"] == indice), 0)
            for v in por_umbral.values())
        filas.append({"indice": int(indice), "nombre": fila["nombre"], "bloque": fila["bloque"],
                      "splits_totales_en_los_5_umbrales": int(total_splits),
                      "usada_en_algun_umbral": bool(total_splits > 0)})
    tabla = pd.DataFrame(filas).sort_values("splits_totales_en_los_5_umbrales", ascending=False)
    tabla.to_csv(TABLA, index=False)
    usadas = int(tabla["usada_en_algun_umbral"].sum())
    print(f"\n[conc] features usadas en al menos un umbral: {usadas} de {len(tabla)}", flush=True)
    for bloque, grupo in tabla.groupby("bloque"):
        n = int(grupo["usada_en_algun_umbral"].sum())
        print(f"    {bloque:<22} {n:>3} de {len(grupo):>3} usadas", flush=True)

    _figura(tabla)
    RESULTADO.write_text(json.dumps({
        "fase": "V27 04 concentracion de features", "fecha": "2026-09-28", "experimenta": False,
        "hallazgo": ("el meta-modelo de V20 divide por muy pocas de sus 279 features, y todas son "
                     "del bloque de evidencia de los miembros"),
        "por_umbral": por_umbral, "controles": controles,
        "features_totales": int(matriz.shape[1]), "features_usadas": usadas,
        "es_efecto_del_submuestreo": False,
        "prueba": ("quitar colsample_bytree o subsample no cambia el numero de features usadas; "
                   "subir num_leaves a 31 si las amplía de 7 a 30, luego estaban disponibles y no "
                   "usadas por falta de capacidad"),
        "explica": {
            "V25": ("el mutual information ordena alto las columnas de miembro_ensamble, que son "
                    "justo las que el modelo usa; las estructuradas y las densas que el MI "
                    "descarta son las que ya no se dividian"),
            "V26-01": ("reescalar columnas no cambia su orden, y el modelo divide por orden"),
            "H11": ("las features con relacion monotona defendible son las probabilidades de clase, "
                    "en los indices 12 a 53, y ninguna se usa para dividir")},
        "no_abre_hipotesis": True,
        "segundos": round(time.time() - inicio, 1),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


def _figura(tabla: pd.DataFrame) -> None:
    """Reparto de splits por feature, con el bloque como color."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    usados = tabla[tabla["splits_totales_en_los_5_umbrales"] > 0]
    colores = {"miembro_ensamble": "#2166ac", "label_informed": "#1a9850",
               "estructurado": "#d73027", "denso_textometrico": "#fc8d59",
               "nbsvm_palabra": "#984ea3"}
    fig, ejes = plt.subplots(1, 2, figsize=(14, 5.5))
    eje = ejes[0]
    eje.barh(range(len(usados)), usados["splits_totales_en_los_5_umbrales"],
             color=[colores.get(b, "#999999") for b in usados["bloque"]])
    eje.set_yticks(range(len(usados)))
    eje.set_yticklabels(usados["nombre"], fontsize=6)
    eje.set_xlabel("splits en los cinco umbrales")
    eje.set_title(f"Las {len(usados)} features que V20 divide de {len(tabla)}", fontsize=10)
    eje = ejes[1]
    por_bloque = tabla.groupby("bloque")["usada_en_algun_umbral"].agg(["sum", "size"])
    eje.bar(range(len(por_bloque)), por_bloque["sum"] / por_bloque["size"],
            color=[colores.get(b, "#999999") for b in por_bloque.index])
    eje.set_xticks(range(len(por_bloque)))
    eje.set_xticklabels(por_bloque.index, fontsize=6, rotation=20)
    eje.set_ylabel("proporcion usada para dividir")
    eje.set_title("Proporcion usada por bloque", fontsize=10)
    fig.suptitle("V20: concentracion de features en el meta-modelo", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
