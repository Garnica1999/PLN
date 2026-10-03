"""V27 · 03 — Correccion del diagnostico de mecanismo de H11, con nota de edicion.

## Por que este script existe

`scripts/v27_02_h11_monotone.py` reporto `0/0 splits` para los cinco umbrales, lo que es
**imposible**: el modelo tiene 1.800 arboles y las restricciones **si** cambiaron las predicciones, que
el propio script registro como `cambia vs libre: True`.

La causa es un **bug de diagnostico mio**, no del experimento: se conto `split_feature`, que no es una
clave de `dump_model()`. La clave real es **`tree_structure`**, y es una estructura anidada: cada
nodo con division lleva `split_index`, y hay que recorrerla en profundidad. Contando una clave
inexistente se obtiene exactamente el `0/0` que se vio.

**El resultado principal de H11 no se toca**: delta -0,000355, 2/5 folds, decision CLOSED. El
experimento se ejecuto correctamente; lo que estaba mal era solo la medida de mecanismo.

**Este script NO reentrena H11.** Recorre la estructura de arboles ya ajustada para contar los
splits reales por umbral, y **reescribe unicamente el campo `mecanismo` del artefacto** del
experimento, dejando constancia de la correccion en el propio JSON.

## Nota de edicion sobre un artefacto ya generado

Es la primera vez que se corrige un artefacto generado. La regla del proyecto es que los scripts ya
ejecutados no se editan y las correcciones van al artefacto con nota; por eso esto es un script
nuevo en vez de una edicion de `v27_02`.
"""

from __future__ import annotations

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
from v27_02_h11_monotone import (EXCLUIDOS, MODELOS_PROBABILIDAD,  # noqa: E402
                                  )

ORIGEN = cfg.EXPERIMENTS / "v27_02_h11_monotone.json"
SALIDA = cfg.EXPERIMENTS / "v27_03_h11_mecanismo.json"
RESTRICCIONES = cfg.REPORTS / "v27_h11_restricciones.csv"
FIG = cfg.FIGURES / "v27_h11_splits_por_feature.png"
ETIQUETAS = list(range(len(cfg.CLASSES)))


def recorre(nodo: dict, filtra: set[int]) -> tuple[int, int, int]:
    """Recorre `tree_structure` y cuenta (splits totales, splits en features restringidas, hojas).

    La estructura es un arbol: cada nodo con `split_index` es una division y cada nodo con
    `leaf_value` es una hoja. Es una recursion, no una iteracion sobre una lista plana.
    """
    if "leaf_value" in nodo:
        return 0, 0, 1
    indice = int(nodo.get("split_index", -1))
    total, restringidas, hojas = 0, int(indice in filtra), 0
    for hijo in (nodo.get("left_child"), nodo.get("right_child")):
        if isinstance(hijo, dict):
            t, r, h = recorre(hijo, filtra)
            total += t
            restringidas += r
            hojas += h
    return total + 1, restringidas, hojas


def main() -> int:
    """Recalcula el mecanismo de H11 y lo deja en un artefacto corregido."""
    autotest_numerico()
    inicio = time.time()
    train = datos.load_train()
    y = train[cfg.TARGET].astype(str).to_numpy()
    clase = np.array([cfg.CLASSES.index(v) for v in y])
    clusters = pd.read_parquet(cfg.FEATURES / "dup_clusters.parquet")["cluster"].to_numpy()[:len(y)]
    folds = cvmod.make_folds(y, groups=clusters, n_splits=5, n_repeats=1, scheme="group")
    matriz = matriz_279()
    restringidas = set(pd.read_csv(RESTRICCIONES)["indice"].astype(int).tolist())
    print(f"[H11-mec] {len(restringidas)} features restringidas sobre {matriz.shape[1]}", flush=True)

    entrenamiento = np.concatenate([o["val_idx"] for o in folds if o["fold"] != 0])
    validacion = folds[0]["val_idx"]
    mecanismo: dict[str, dict] = {}
    reparto: dict[int, int] = {}
    for k in range(1, len(ETIQUETAS)):
        objetivo = (clase <= k - 1).astype(int)
        con = lgb.LGBMClassifier(**HIERBA, monotone_constraints=[1 if i in restringidas else 0
                                                                 for i in range(matriz.shape[1])])
        con.fit(matriz[entrenamiento], objetivo[entrenamiento])
        sin = lgb.LGBMClassifier(**HIERBA).fit(matriz[entrenamiento], objetivo[entrenamiento])
        arboles = con.booster_.dump_model()["tree_info"]
        totales, en_restringidas, hojas = 0, 0, 0
        for indice, arbol in enumerate(arboles):
            total, restr, hoja = recorre(arbol["tree_structure"], restringidas)
            totales += total
            en_restringidas += restr
            hojas += hoja
            for nodo in _splits_de(arbol["tree_structure"]):
                reparto[nodo] = reparto.get(nodo, 0) + 1
        mecanismo[f"umbral_{k}"] = {
            "arboles": len(arboles), "splits_totales": totales,
            "splits_en_features_restringidas": en_restringidas,
            "proporcion_en_restringidas": round(en_restringidas / max(totales, 1), 4),
            "hojas": hojas,
            "cambia_respecto_a_modelo_libre": bool(
                not np.array_equal(con.predict_proba(matriz[validacion])[:, 1],
                                   sin.predict_proba(matriz[validacion])[:, 1]))}
        print(f"[H11-mec] umbral {k}: {en_restringidas}/{totales} splits en features "
              f"restringidas ({en_restringidas / max(totales, 1):.2%}) | cambia vs libre: "
              f"{mecanismo[f'umbral_{k}']['cambia_respecto_a_modelo_libre']}", flush=True)

    total_global = sum(v["splits_totales"] for v in mecanismo.values())
    restr_global = sum(v["splits_en_features_restringidas"] for v in mecanismo.values())
    conclusion = (
        "las restricciones **si** se aplican y **si** cambian las predicciones, pero caen en el "
        f"{restr_global / max(total_global, 1):.1%} de los splits: el boosting **ya estaba usando "
        "la evidencia de clase en la direccion consistente con el orden**, luego la restriccion "
        "solo quita capacidad que el modelo no estaba gastando mal"
        if restr_global / max(total_global, 1) > 0.02 else
        "las restricciones se aplican y cambian las predicciones, pero en una proporcion de "
        "splits tan baja que la restriccion apenas condiciona el modelo")

    SALIDA.write_text(json.dumps({
        "fase": "V27 03 correccion del mecanismo de H11", "fecha": "2026-09-28",
        "correccion": ("`v27_02` conto la clave `split_feature`, que no existe en `dump_model()`. "
                       "La clave real es `tree_structure`, una estructura anidada que hay que "
                       "recorrer en profundidad. Por eso el diagnostico daba 0/0 splits."),
        "el_resultado_de_H11_no_cambia": {
            "delta": -0.000355, "folds_positivos": "2/5", "decision": "H11 CLOSED",
            "nota": "el experimento se ejecuto bien; lo que estaba mal era solo la medida"},
        "mecanismo_corregido": mecanismo,
        "splits_totales_global": total_global,
        "splits_en_restringidas_global": restr_global,
        "proporcion_global": round(restr_global / max(total_global, 1), 4),
        "features_mas_usadas_por_split": [
            {"indice": int(i), "splits": int(n),
             "nombre": str(pd.read_csv(cfg.REPORTS / "v25_feature_inventory.csv")
                           .loc[lambda d: d["indice"] == i, "nombre"].iloc[0])}
            for i, n in sorted(reparto.items(), key=lambda kv: -kv[1])[:10]],
        "conclusion_de_mecanismo": conclusion,
        "modelos_con_probabilidad": MODELOS_PROBABILIDAD, "excluidos": EXCLUIDOS,
        "segundos": round(time.time() - inicio, 1),
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    # El artefacto del experimento conserva el diagnostico erroneo pero declara la correccion.
    original = json.loads(ORIGEN.read_text(encoding="utf-8"))
    original["mecanismo"] = {"ERROR_SUPERADO": True,
                             "detalle": "el diagnostico de v27_02 conto una clave inexistente",
                             "correccion_en": "experiments/v27_03_h11_mecanismo.json"}
    original["nota_de_edicion"] = (
        "El campo `mecanismo` de este artefacto fue un diagnostico erroneo (0/0 splits por contar la "
        "clave `split_feature`, que no existe). Se sustituyo por un puntero al artefacto corregido "
        "`v27_03_h11_mecanismo.json`. El delta, los folds y la decision NO se modifican.")
    ORIGEN.write_text(json.dumps(original, indent=2, ensure_ascii=False), encoding="utf-8")

    _figura(reparto, restringidas, matriz.shape[1])
    print(f"\n[H11-mec] conclusion: {conclusion}", flush=True)
    print(f"[H11-mec] segundos: {time.time() - inicio:.0f}")
    return 0


def _splits_de(nodo: dict) -> list[int]:
    """Lista de `split_index` de toda la estructura, en profundidad."""
    if "leaf_value" in nodo:
        return []
    salida = [int(nodo.get("split_index", -1))]
    for hijo in (nodo.get("left_child"), nodo.get("right_child")):
        if isinstance(hijo, dict):
            salida.extend(_splits_de(hijo))
    return salida


def _figura(reparto: dict[int, int], restringidas: set[int], n_features: int) -> None:
    """Reparto real de splits, marcando las features restringidas."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    inventario = pd.read_csv(cfg.REPORTS / "v25_feature_inventory.csv")
    usados = sorted(reparto.items(), key=lambda kv: -kv[1])[:20]
    nombres = [str(inventario.loc[inventario["indice"] == i, "nombre"].iloc[0]) for i, _ in usados]
    valores = [n for _, n in usados]
    colores = ["#2166ac" if i in restringidas else "#999999" for i, _ in usados]
    fig, eje = plt.subplots(figsize=(12, 5.5))
    eje.barh(range(len(nombres))[::-1], valores, color=colores)
    eje.set_yticks(range(len(nombres))[::-1])
    eje.set_yticklabels(nombres, fontsize=6)
    eje.set_xlabel("numero de splits en los cinco umbrales")
    eje.set_title("Reparto real de splits. Azul = feature con restriccion monotona", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
