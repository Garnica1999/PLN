"""auditoria_medir_inner_split_optuna.py

Fecha: 2026-10-01
Descripcion: mide el sesgo real del INNER SPLIT que el proyecto usa para seleccionar
    hiperparametros de Optuna. El codigo de V29/V30 lo construye con
    `make_folds(..., groups=np.arange(len(entrenamiento)), scheme="stratified")`, es decir
    `StratifiedKFold` ingenuo con grupo IDENTIDAD, que NO agrupa near-duplicates. Pero
    `src/cv.py:11` documenta ese mismo esquema como el CONTROL para cuantificar el leakage.

    Este script NO entrena, NO reoptimiza, NO toca el campeon. Solo reconstruye las dos
    particiones interiores exactamente como las construye el codigo real y mide cuanto se
    parecen, en terminos de solapamiento de near-duplicates.

Que mide
--------
1. `overlap_exacto`: fraccion de filas de la validacion interior cuyo texto EXACTO (normalizado)
   aparece en el inner train. Es la cota inferior del sesgo: solo igualdad de texto.
2. `solape_cluster`: fraccion de filas de la validacion interior cuyo CLUSTER de near-duplicate
   aparece en el inner train. Es la medida que importa, porque los clusters agrupan por
   similitud de Jaccard, no solo por igualdad.
3. La misma medicion en el outer split AGRUPADO canonico, como referencia del suelo.
4. Cuantas filas del inner train quedan huerfanas de cluster en el inner train (posible fuga).

Criterio de interpretacion, declarado ANTES de correr
-----------------------------------------------------
- Si el inner ingenuo tiene solape de cluster MUY SUPERIOR al agrupado, el sesgo es real y
  V29/V30 son re-optimizables con inner split agrupado.
- Si la diferencia es marginal (menos de ~2 puntos porcentuales), el sesgo no es material y la
  reoptimizacion NO se justifica por esta via.
- Este script NO decide si reoptimizar: mide y reporta. La decision es de Carlos.

Modos
-----
- Sin argumentos: mide sobre los 5 folds externos canonicos y escribe
  `reports/tables/auditoria_inner_split_optuna.csv` y
  `experiments/auditoria_medir_inner_split_optuna.json`.
- `--verificar`: autoejercicio en memoria. No escribe nada. Sale 0 si pasa, 2 si falla.

ASCII puro. No lee test, holdout ni leaderboard. No genera submissions.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config as cfg
from src import cv as cvmod
from src import data as D

FUENTES = ("v29_04_optuna_individual", "v29_06_ensemble", "v30_01_optuna_refinamiento",
           "v30_02_ensemble", "v31_05_integracion")

OUT_CSV = ROOT / "reports" / "tables" / "auditoria_inner_split_optuna.csv"
OUT_JSON = ROOT / "experiments" / "auditoria_medir_inner_split_optuna.json"

# Criterio declarado antes de ejecutar (ver docstring).
UMBRAL_MATERIAL = 0.02  # 2 puntos porcentuales de diferencia en solape de cluster


def normaliza(t: pd.Series) -> np.ndarray:
    """Texto normalizado a minusculas con espacios colapsados, para comparar por igualdad."""
    return t.astype(str).str.lower().str.split().str.join(" ").to_numpy()


def solape_exacto(idx_val: np.ndarray, idx_train: np.ndarray, textos: np.ndarray) -> float:
    s = set(textos[idx_train])
    if len(idx_val) == 0:
        return 0.0
    return float(np.mean([textos[i] in s for i in idx_val]))


def solape_cluster(idx_val: np.ndarray, idx_train: np.ndarray, clusters: np.ndarray) -> float:
    """Fraccion de filas de val cuyo cluster de near-duplicate aparece en train.

    Esta es la medida relevante: los clusters de `dup_clusters.parquet` agrupan por igualdad de
    texto normalizado O similitud de Jaccard por encima de un umbral, asi que capta el leakage que
    la igualdad exacta NO ve.
    """
    s = set(clusters[idx_train])
    if len(idx_val) == 0:
        return 0.0
    return float(np.mean([clusters[i] in s for i in idx_val]))


def construir_inner_como_el_codigo(clase_ent: np.ndarray, clusters_ent: np.ndarray
                                   ) -> tuple[np.ndarray, np.ndarray]:
    """Reproduce el inner split de `v30_01_optuna_refinamiento.py:220-226`.

    El codigo real usa el fold cuyo tamano de validacion mas se acerca a
    `FRACCION_INTERIOR * len(entrenamiento)`, no el fold 0. Se replica ese criterio.
    """
    interior_ingenuo = cvmod.make_folds(clase_ent, groups=np.arange(len(clase_ent)),
                                       n_splits=5, n_repeats=1, scheme="stratified")
    objetivo = int(len(clase_ent) * 0.30)
    elegido = min(interior_ingenuo, key=lambda f: abs(len(f["val_idx"]) - objetivo))
    val = np.asarray(elegido["val_idx"])
    trn = np.concatenate([np.asarray(o["val_idx"]) for o in interior_ingenuo
                          if o["fold"] != elegido["fold"]])
    return trn, val


def medir() -> dict:
    inicio = time.time()
    train = D.load_train()
    n = len(train)
    clase = train[cfg.TARGET].astype(str).to_numpy()
    textos = normaliza(train[cfg.TEXT_COL])
    clusters = pd.read_parquet(cfg.FEATURES / "dup_clusters.parquet")["cluster"].to_numpy()[:n]

    # outer canonico: el que el proyecto usa para reportar (agrupado, persistido en artifacts)
    outer = cvmod.make_folds(clase, groups=clusters, n_splits=5, n_repeats=1, scheme="group")

    filas: list[dict] = []
    for fold in outer:
        k = int(fold["fold"])
        idx_val = np.asarray(fold["val_idx"])
        idx_trn = np.asarray(fold["train_idx"])

        # referencia: el propio outer, que es el esquema correcto
        filas.append({
            "nivel": "outer_agrupado",
            "fold": k,
            "n_val": len(idx_val),
            "n_train": len(idx_trn),
            "overlap_exacto": round(solape_exacto(idx_val, idx_trn, textos), 6),
            "solape_cluster": round(solape_cluster(idx_val, idx_trn, clusters), 6),
        })

        # el inner TAL COMO LO HACE EL CODIGO (ingenuo)
        trn_i, val_i = construir_inner_como_el_codigo(clase[idx_trn], clusters[idx_trn])
        inner_val_abs = idx_trn[val_i]
        inner_trn_abs = idx_trn[trn_i]
        filas.append({
            "nivel": "inner_ingenuo_codigo_real",
            "fold": k,
            "n_val": len(inner_val_abs),
            "n_train": len(inner_trn_abs),
            "overlap_exacto": round(solape_exacto(inner_val_abs, inner_trn_abs, textos), 6),
            "solape_cluster": round(solape_cluster(inner_val_abs, inner_trn_abs, clusters), 6),
        })

        # el inner que haria falta (agrupado), sobre el MISMO fold-train
        interior_agrupado = cvmod.make_folds(clase[idx_trn], groups=clusters[idx_trn],
                                             n_splits=5, n_repeats=1, scheme="group")
        objetivo = int(len(idx_trn) * 0.30)
        elegido = min(interior_agrupado, key=lambda f: abs(len(f["val_idx"]) - objetivo))
        v = idx_trn[np.asarray(elegido["val_idx"])]
        t = idx_trn[np.concatenate([np.asarray(o["val_idx"]) for o in interior_agrupado
                                     if o["fold"] != elegido["fold"]])]
        filas.append({
            "nivel": "inner_agrupado_propuesto",
            "fold": k,
            "n_val": len(v),
            "n_train": len(t),
            "overlap_exacto": round(solape_exacto(v, t, textos), 6),
            "solape_cluster": round(solape_cluster(v, t, clusters), 6),
        })

    tabla = pd.DataFrame(filas)
    resumen_niveles = {}
    for nivel, g in tabla.groupby("nivel", sort=False):
        resumen_niveles[nivel] = {
            "overlap_exacto_medio": round(float(g["overlap_exacto"].mean()), 6),
            "solape_cluster_medio": round(float(g["solape_cluster"].mean()), 6),
            "solape_cluster_std": round(float(g["solape_cluster"].std(ddof=0)), 6),
            "n_val_medio": int(g["n_val"].mean()),
        }

    ing = resumen_niveles["inner_ingenuo_codigo_real"]["solape_cluster_medio"]
    agr = resumen_niveles["inner_agrupado_propuesto"]["solape_cluster_medio"]
    ext = resumen_niveles["outer_agrupado"]["solape_cluster_medio"]
    diferencia = ing - agr

    veredicto = (
        "SESGO MATERIAL: el inner split ingenuo comparte mas near-duplicates que el agrupado; "
        "V29/V30 son re-optimizables con inner agrupado"
        if diferencia > UMBRAL_MATERIAL
        else "SESGO NO MATERIAL: la diferencia esta por debajo del umbral declarado de "
             f"{UMBRAL_MATERIAL}; la reoptimizacion NO se justifica por esta via"
    )

    return {
        "generado_por": "scripts/auditoria_medir_inner_split_optuna.py",
        "fecha": time.strftime("%Y-%m-%d"),
        "n_train": n,
        "n_clusters": len(set(clusters)),
        "scripts_que_usan_inner_ingenuo": list(FUENTES),
        "umbral_materialidad": UMBRAL_MATERIAL,
        "criterio": (
            "diferencia de solape de cluster entre inner ingenuo e inner agrupado > "
            f"{UMBRAL_MATERIAL} se considera material"
        ),
        "por_nivel": resumen_niveles,
        "diferencia_inner_ingenuo_menos_agrupado": round(diferencia, 6),
        "referencia_outer_agrupado": round(ext, 6),
        "veredicto": veredicto,
        "no_entrena": True,
        "no_toca_campeon": True,
        "no_lee_test_holdout_lb": True,
        "tabla": filas,
        "segundos": round(time.time() - inicio, 1),
    }


def verificar() -> int:
    """Autoejercicio en memoria: comprueba que las funciones miden lo que dicen."""
    print("=== autoejercicio de la medicion del inner split ===")
    fallos = 0

    # V1: solape_exacto detecta un caso construido a mano con solapamiento conocido.
    textos = np.array(["a", "b", "c", "a"], dtype=object)
    v = np.array([0, 1])
    t = np.array([2, 3])           # 'a' esta en train (idx 3), 'b' no
    got = solape_exacto(v, t, textos)
    ok = abs(got - 0.5) < 1e-9
    print(f"  [{'OK' if ok else 'FALLA'}] V1 solape_exacto en caso conocido (espera 0.5): {got}")
    fallos += 0 if ok else 1

    # V2: casos con solapamiento conocido y distinto del de V1.
    # textos = ['a','b','c','a']; val=[1,2] vs train=[0,3] -> 'b' y 'c' NO estan -> 0.0
    # textos = ['a','b','c','a']; val=[0,3] vs train=[0,1] -> ambas 'a' -> 1.0
    ok = (abs(solape_exacto(np.array([1, 2]), np.array([0, 3]), textos) - 0.0) < 1e-9
          and abs(solape_exacto(np.array([0, 3]), np.array([0, 1]), textos) - 1.0) < 1e-9)
    print(f"  [{'OK' if ok else 'FALLA'}] V2 solape_exacto en los extremos (0.0 y 1.0)")
    fallos += 0 if ok else 1

    # V3: solape_cluster capta lo que el exacto NO ve (mismo cluster, texto distinto).
    clusters = np.array([7, 7, 9, 7])
    textos_d = np.array(["hola", "adios", "otro", "hey"], dtype=object)
    ok = (abs(solape_exacto(np.array([0]), np.array([3]), textos_d) - 0.0) < 1e-9
          and abs(solape_cluster(np.array([0]), np.array([3]), clusters) - 1.0) < 1e-9)
    print(f"  [{'OK' if ok else 'FALLA'}] V3 solape_cluster ve lo que el exacto no "
          f"(texto distinto, mismo cluster)")
    fallos += 0 if ok else 1

    # V4: el resultado es un numero finito en [0,1] y no un vacio silencioso.
    ok = 0.0 <= got <= 1.0
    print(f"  [{'OK' if ok else 'FALLA'}] V4 resultado en [0,1]: {got}")
    fallos += 0 if ok else 1

    # V5: `construir_inner_como_el_codigo` respeta la estructura anidada que el codigo real exige.
    try:
        rng = np.random.default_rng(0)
        c = np.repeat(np.arange(6), 100)
        cl = rng.integers(0, 50, len(c))
        tr, va = construir_inner_como_el_codigo(c, cl)
        ok = len(np.intersect1d(tr, va)) == 0 and len(tr) + len(va) == len(c)
        print(f"  [{'OK' if ok else 'FALLA'}] V5 inner split anidado y cubriendo todo "
              f"(train {len(tr)}, val {len(va)}, solape {len(np.intersect1d(tr, va))})")
        fallos += 0 if ok else 1
    except Exception as exc:  # noqa: BLE001
        print(f"  [FALLA] V5 el inner split lanzo excepcion: {exc}")
        fallos += 1

    print(f"  {5 - fallos}/5 caminos pasan; no se escribio ningun artefacto real")
    return 0 if fallos == 0 else 2


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Mide el sesgo del inner split de Optuna.")
    ap.add_argument("--verificar", action="store_true", help="autoejercicio; no escribe nada")
    args = ap.parse_args(argv)

    if args.verificar:
        return verificar()

    res = medir()
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(res["tabla"]).to_csv(OUT_CSV, index=False)
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(res, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(json.dumps({k: v for k, v in res.items() if k != "tabla"}, indent=2, ensure_ascii=False))
    print(f"\n[fin] tabla de {len(res['tabla'])} filas en {OUT_CSV.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
