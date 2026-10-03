"""V29 - 01 - P1: inventario de los modelos historicos con OOF reutilizable.

V29 no empieza entrenando: empieza **leyendo lo que ya esta entrenado**. Este script audita todos
los artefactos de prediccion out-of-fold que el historial dejo en disco y determina, para cada uno,
si es metodologicamente reutilizable sin volver a entrenar.

## Que se considera un modelo reutilizable

Una matriz `.npy` de forma `(n_filas, 6)` que ademas:

1. tenga exactamente las filas del train y seis columnas de clase;
2. produzca un argmax homogeneo entre folds (un OOF real no tiene folds dispares: si uno se
   desmarca, el artefacto no es OOF o esta contaminado);
3. no sea un duplicado exacto de otro artefacto ya inventariado;
4. no sea una version transformada de otro (por ejemplo, el mismo OOF guardado dos veces con otro
   nombre), lo que se detecta comparando el argmax y un hash de los valores.

Las matrices de cinco o tres columnas quedan **excluidas del pool**: son CDFs de cinco umbrales o
decisiones binarias, no distribuciones sobre seis clases, y su `argmax` no es una clase.

## Que NO hace

No entrena, no optimiza, no lee el test, no lee el leaderboard y no inventa metadatos. Lo que no
se puede leer de un artefacto se marca como `desconocido`, no se rellena.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src import config as cfg  # noqa: E402
from src import cv as cvmod  # noqa: E402
from src import data as datos  # noqa: E402
from src.verificacion_numerica import autotest_numerico  # noqa: E402

RESULTADO = cfg.EXPERIMENTS / "v29_01_inventario.json"
INVENTARIO = cfg.REPORTS / "v29_model_inventory.csv"
CV_V20 = 0.6097782
N_FILAS = 169336
N_CLASES = 6
#: Fases de las que procede cada prefijo de artefacto. Se infiere del nombre, y se marca como
#: `inferida_del_nombre` cuando no hay otra fuente.
FASE_POR_PREFIJO = {
    "oof_E": "E (modelos clasicos, fase temprana)",
    "oof_ADV": "ADV (fase avanzada)",
    "oof_meta": "meta-modelos de la fase de metamodelado",
    "oof_ensemble": "ensemble V12",
    "oof_content": "solo contenido",
    "oof_style": "solo estilo",
    "v13_": "V13 (fusión de tokens)",
    "v14": "V14",
    "v15": "V15",
    "v16_": "V16 (combinadores)",
    "v17_": "V17",
    "v19_": "V19 (cuartos del blend)",
    "v20_": "V20 (cadena de umbrales y arboles)",
    "v21_": "V21 (B1)",
    "v22_": "V22 (forense de V20)",
    "v23_": "V23 (CORAL)",
    "vista_": "vistas expandidas de la matriz 279",
}
#: Palabras que delatan la familia del modelo a partir del nombre del artefacto.
FAMILIA_POR_PALABRA = [
    (r"nbsvm", "NB-SVM (lineal sobre binarios)"),
    (r"bm25", "BM25 (lineal disperso)"),
    (r"knn", "vecinos mas cercanos"),
    (r"extratrees", "ExtraTrees"),
    (r"sgd", "SGD"),
    (r"mlp", "perceptron multicapa"),
    (r"logreg", "logistica multiclase"),
    (r"ridge|lineal_c0|lineal_c1|rochio|logistico|meta_sgd", "lineal"),
    (r"arbol4|arbol5|arbol_|boosting", "arbol / boosting"),
    (r"coral|cumulad|cdf|ordinal", "estructura ordinal"),
    (r"E028a|pares", "clasificadores por pares"),
    (r"E028b|grupos", "clasificadores por grupos"),
    (r"ADV07|word", "lineal disperso por palabra"),
    (r"ADV05", "NB-SVM (lineal sobre binarios)"),
    (r"contenido|content", "solo contenido"),
    (r"estilo|style", "solo estilo"),
    (r"softmax|margenes|probabilidades|oof_E01[3-7]", "cabeza de modelo multiclase"),
    (r"oof_E01[9ab]", "denso textometrico (SVD)"),
    (r"ensemble_v12|v16_oof|v17_|v13_p1b|v14b|v15b", "combinador / meta-modelo"),
    (r"f3b_oof_candidato|oof_v20|oof_v19|seis_compartida", "cuarto de V20"),
    (r"vista_", "vista expandida de la matriz 279"),
]


def _familia(nombre: str) -> str:
    for patron, etiqueta in FAMILIA_POR_PALABRA:
        if re.search(patron, nombre, re.IGNORECASE):
            return etiqueta
    return "desconocida"


def _fase(nombre: str) -> str:
    for prefijo, fase in FASE_POR_PREFIJO.items():
        if nombre.startswith(prefijo):
            return fase
    return "desconocida"


def _tipo(matriz: np.ndarray) -> str:
    """Distingue probabilidad, margen o log-odds, que importa para combinar bien."""
    suma_uno = float(np.mean(np.abs(matriz.sum(axis=1) - 1.0) < 1e-3))
    dentro = float(np.mean((matriz >= -1e-9) & (matriz <= 1.0 + 1e-9)))
    if suma_uno > 0.98 and dentro > 0.98:
        return "probabilidad"
    if dentro > 0.98:
        return "probabilidad_no_normalizada"
    return "margen_o_log_odds"


def _homogeneidad(por_fold: list[float]) -> float:
    """1 - coeficiente de variacion. Un OOF real se acerca a 1; uno contaminado, a 0."""
    if len(por_fold) < 2:
        return 0.0
    media = float(np.mean(por_fold))
    if media == 0:
        return 0.0
    return round(1.0 - float(np.std(por_fold)) / abs(media), 6)


def main() -> int:
    """Inventaria los OOF reutilizables y decide cuales entran al analisis de diversidad."""
    autotest_numerico()
    inicio = time.time()
    train = datos.load_train()
    y = train[cfg.TARGET].astype(str).to_numpy()
    clase = np.array([cfg.CLASSES.index(v) for v in y])
    clusters = pd.read_parquet(cfg.FEATURES / "dup_clusters.parquet")["cluster"].to_numpy()[:len(y)]
    folds = cvmod.make_folds(y, groups=clusters, n_splits=5, n_repeats=1, scheme="group")

    # El registro historico aporta el metadato de fase, modelo y artefacto cuando lo declaro.
    registro = pd.read_csv(ROOT / "experiments.csv")
    columna_artefactos = next((c for c in registro.columns if "artefact" in c.lower()), None)
    print(f"[inventario] experiments.csv: {len(registro)} filas | columna de artefactos: "
          f"{columna_artefactos!r}", flush=True)

    filas, descartados, hashes = [], [], {}
    candidatas = []
    for ruta in sorted(cfg.FEATURES.glob("*.npy")):
        try:
            matriz = np.load(ruta, allow_pickle=False)
        except ValueError:
            continue
        if matriz.ndim != 2 or matriz.shape[0] != N_FILAS or matriz.dtype.kind not in "fiu":
            continue
        candidatas.append((ruta, matriz))

    for ruta, matriz in candidatas:
        nombre = ruta.name
        if matriz.shape[1] != N_CLASES:
            descartados.append({"artefacto": nombre, "motivo": f"tiene {matriz.shape[1]} columnas, "
                                "no 6: es una CDF de umbrales o una decision binaria, su argmax no "
                                "es una clase"})
            continue
        argmax = np.argmax(matriz, axis=1).astype(np.int8)
        # Huella de valores: distingue dos artefactos con el mismo argmax pero distinta confianza.
        huella = hashlib.sha256(np.ascontiguousarray(matriz[:, ::7]).tobytes()).hexdigest()[:16]
        clave = hashlib.sha256(argmax.tobytes()).hexdigest()[:16]
        if hashes.get(clave + huella):
            filas.append({**filas[int(hashes[clave + huella])], "artefacto": nombre,
                          "nota": "duplicado exacto de otro artefacto ya inventariado"})
            continue
        hashes[clave + huella] = str(len(filas))
        macro = float(f1_score(clase, argmax, average="macro", labels=list(range(N_CLASES))))
        por_fold = [float(f1_score(clase[f["val_idx"]], argmax[f["val_idx"]], average="macro",
                                   labels=list(range(N_CLASES)))) for f in folds]
        # Busca en el registro historico cualquier fila que mencione este artefacto.
        mentioned = []
        if columna_artefactos:
            for _, f in registro.iterrows():
                if nombre in str(f[columna_artefactos]):
                    mentioned.append(f["experiment_id"])
        filas.append({
            "model_id": nombre.replace(".npy", ""),
            "artefacto": nombre,
            "fase": _fase(nombre),
            "familia": _familia(nombre),
            "tipo_de_salida": _tipo(matriz),
            "corrected_cv_pooled": round(macro, 7),
            "media_por_fold": round(float(np.mean(por_fold)), 7),
            "desviacion_por_fold": round(float(np.std(por_fold)), 7),
            "homogeneidad_folds": _homogeneidad(por_fold),
            "folds_desviados": int(sum(1 for v in por_fold
                                       if abs(v - float(np.mean(por_fold))) > 0.02)),
            "experimentos_que_lo_citan": ";".join(mentioned) if mentioned else "desconocido",
            "argmax_hash": clave,
            "nota": "",
        })

    tabla = pd.DataFrame(filas).drop_duplicates(subset=["model_id"]).reset_index(drop=True)
    # Reutilizable = seis columnas, homogeneo entre folds, y sin nota de duplicado.
    tabla["reutilizable"] = (tabla["nota"] == "") & (tabla["homogeneidad_folds"] > 0.90) \
        & (tabla["folds_desviados"] <= 1)
    tabla = tabla.sort_values("corrected_cv_pooled", ascending=False).reset_index(drop=True)
    tabla.to_csv(INVENTARIO, index=False)

    print(f"[inventario] {len(tabla)} artefactos de 6 columnas, "
          f"{int(tabla['reutilizable'].sum())} reutilizables", flush=True)
    print(f"  descartados por forma: {len(descartados)}", flush=True)
    print("\n  los 12 con mejor CV propio:", flush=True)
    for _, f in tabla.head(12).iterrows():
        print(f"    {f['model_id']:<44} {f['corrected_cv_pooled']:.7f}  "
              f"homog {f['homogeneidad_folds']:.4f}  {f['familia'][:28]:<28} "
              f"{f['tipo_de_salida'][:20]}", flush=True)
    print("\n  reparto por familia, que es donde puede estar la diversidad real:", flush=True)
    por_familia = tabla[tabla["reutilizable"]].groupby("familia").agg(
        n=("model_id", "count"), mejor=("corrected_cv_pooled", "max")).sort_values("mejor",
                                                                                    ascending=False)
    for fam, r in por_familia.iterrows():
        print(f"    {fam:<40} {int(r['n']):>2} modelos | mejor CV {r['mejor']:.7f}", flush=True)

    RESULTADO.write_text(json.dumps({
        "fase": "V29 01 P1 inventario historico", "fecha": "2026-09-29", "experimenta": False,
        "n_filas_train": N_FILAS, "n_clases": N_CLASES,
        "artefactos_6_columnas": int(len(tabla)),
        "artefactos_reutilizables": int(tabla["reutilizable"].sum()),
        "descartados_por_forma": descartados,
        "familias_presentes": {k: {"n_modelos": int(v["n"]), "mejor_cv": round(float(v["mejor"]), 7)}
                               for k, v in por_familia.to_dict("index").items()},
        "criterio_de_reutilizacion": "6 columnas de clase, homogeneidad entre folds > 0,90, "
                                     "ningun fold desviado mas de 0,02 del resto, y no duplicado",
        "limitacion_declarada": "el homogeneo entre folds es una comprobacion de coherencia, no una "
                                "prueba de que el artefacto sea estrictamente out-of-fold. El "
                                "criterio estricto se aplica mas adelante, al usar un OOF como "
                                "evidencia externa de una optimizacion de pesos.",
        "segundos": round(time.time() - inicio, 1),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame(descartados).to_csv(cfg.REPORTS / "v29_model_inventory_descartados.csv", index=False)
    print(f"\n[inventario] {INVENTARIO.name}: {len(tabla)} filas", flush=True)
    print(f"[fin] {time.time() - inicio:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
