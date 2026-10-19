#!/usr/bin/env python
"""
P96-02: auditoria de particiones, folds y leakage (P4 de P96).

Solo lectura y computo determinista. Verifica, contra los artefactos guardados:

1. La particion congelada dev/holdout: completitud, disjuncion y conteos
   (solo membresia; no se abren etiquetas ni scores del holdout).
2. Los grupos fusionados (near-duplicado UNION pseudo-autor) recomputados con
   src.splits y comparados con los guardados en v2_oof_base_corrected.npz.
3. Los pliegues corregidos recomputados (StratifiedGroupKFold(5, seed 42)) y
   comparados con los guardados en v2_oof_base_corrected.npz y en
   oof_v03_final.npz.
4. La ausencia de fuga de grupos entre entrenamiento y validacion en cada
   pliegue (comprobacion directa, no solo la del constructor).
5. Escaneos estaticos de referencia a toxic_test, holdout y holdout2 en los
   scripts (evidencia cruda a reports/p96_static_scans.txt).

Salida: reports/p96_folds_evidence.json y reports/p96_static_scans.txt.

Uso:
    /home/carlos/venv/bin/python experiments/p96_02_folds_leakage.py
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from src import data as data_module  # noqa: E402
from src.splits import (  # noqa: E402
    combinaciones,
    grupos_fusionados,
    pliegues_corregidos,
)

SALIDA_JSON = RAIZ / "reports" / "p96_folds_evidence.json"
SALIDA_SCAN = RAIZ / "reports" / "p96_static_scans.txt"


def ids_numericos(serie: pd.Series) -> pd.Series:
    """Normaliza ids con comillas malformadas a enteros."""
    extraidos = serie.astype(str).str.extract(r"(\d+)")[0]
    if extraidos.isna().any():
        raise ValueError("Hay ids sin digitos extraibles")
    return extraidos.astype(int)


def main() -> None:
    """Ejecuta la auditoria de particiones y folds."""
    if SALIDA_JSON.exists() or SALIDA_SCAN.exists():
        raise SystemExit("Ya existen salidas de P96-02; no se sobrescriben.")
    evidencia: dict[str, object] = {}

    train = data_module.read_train()
    train = train.assign(_id=ids_numericos(train["id"]))
    ids_train = set(train["_id"].tolist())
    particion = pd.read_csv(RAIZ / "data" / "processed" / "holdout_v2_ids.csv",
                            usecols=["id", "particion"])
    particion = particion.assign(_id=ids_numericos(particion["id"]))
    dev_ids = set(particion.loc[particion["particion"] == "dev", "_id"].tolist())
    holdout_ids = set(particion.loc[particion["particion"] == "holdout", "_id"].tolist())

    evidencia["particion"] = {
        "n_train": len(ids_train),
        "n_dev": len(dev_ids),
        "n_holdout": len(holdout_ids),
        "dev_y_holdout_particionan_train": (dev_ids | holdout_ids) == ids_train,
        "dev_y_holdout_son_disjuntos": len(dev_ids & holdout_ids) == 0,
    }
    print("=== A. Particion congelada ===")
    for clave, valor in evidencia["particion"].items():
        print(f"  {clave}: {valor}")

    # orden de desarrollo tal como lo usa 14_oof_corregido_v2.py
    orden_dev = train.loc[train["_id"].isin(dev_ids)].reset_index(drop=True)
    print(f"  filas de desarrollo en orden de train: {len(orden_dev)}")

    # --- grupos fusionados recomputados ---
    print()
    print("=== B. Grupos fusionados (near-dup UNION pseudo-autor) ===")
    grupos_full = grupos_fusionados(train)
    mapa_grupo = dict(zip(train["_id"].tolist(), grupos_full.tolist()))
    grupos_dev = np.array([mapa_grupo[i] for i in orden_dev["_id"].tolist()])

    bases = np.load(RAIZ / "features" / "v2_oof_base_corrected.npz",
                    allow_pickle=True)
    ids_art = bases["ids"].astype(np.int64)
    grupos_art = np.asarray(bases["grupos"]).astype(np.int64)
    mismo_orden = bool(np.array_equal(ids_art, orden_dev["_id"].to_numpy()))
    evidencia["grupos"] = {
        "mismo_orden_ids_que_train_dev": mismo_orden,
        "iguales_etiqueta_a_etiqueta": bool(
            mismo_orden and np.array_equal(grupos_art, grupos_dev)),
    }
    if mismo_orden:
        # equivalencia de particion (independiente del etiquetado de raices)
        _, inv_re = np.unique(grupos_dev, return_inverse=True)
        _, inv_art = np.unique(grupos_art, return_inverse=True)
        evidencia["grupos"]["equivalencia_de_particion"] = bool(
            np.array_equal(inv_re, inv_art))
    print(f"  mismos ids y orden que train_dev: {evidencia['grupos']['mismo_orden_ids_que_train_dev']}")
    print(f"  grupos iguales etiqueta a etiqueta: {evidencia['grupos']['iguales_etiqueta_a_etiqueta']}")
    if mismo_orden:
        print(f"  equivalencia de particion: {evidencia['grupos']['equivalencia_de_particion']}")
    n_grupos_dev = len(np.unique(grupos_dev))
    print(f"  componentes en dev: {n_grupos_dev}")

    # --- pliegues corregidos recomputados ---
    print()
    print("=== C. Pliegues corregidos (StratifiedGroupKFold(5, seed 42)) ===")
    pliegues = pliegues_corregidos(orden_dev, grupos_dev)
    pliegue_de_fila = np.full(len(orden_dev), -1)
    for numero, (_, validacion) in enumerate(pliegues):
        pliegue_de_fila[validacion] = numero
    tamanos = [int(np.sum(pliegue_de_fila == k)) for k in range(5)]
    print(f"  tamanos por pliegue: {tamanos}")
    sin_asignar = int(np.sum(pliegue_de_fila == -1))
    print(f"  filas sin pliegue: {sin_asignar}")

    # fuga de grupos: verificacion directa por pliegue
    fugas = []
    for numero, (entrena, valida) in enumerate(pliegues):
        g_e = set(grupos_dev[entrena].tolist())
        g_v = set(grupos_dev[valida].tolist())
        if g_e & g_v:
            fugas.append(numero)
    print(f"  pliegues con grupos compartidos entrena/valida: {fugas or 'ninguno'}")

    pliegues_en_bases = "pliegues" in bases.keys()
    oof = np.load(RAIZ / "features" / "oof_v03_final.npz", allow_pickle=True)
    pliegues_final = np.asarray(oof["pliegues"]).astype(np.int64)
    iguales_estricto = bool(np.array_equal(pliegue_de_fila, pliegues_final))
    # comparacion robusta: mapeo de numeracion k_recomputado -> etiqueta guardada
    mapeo: dict[int, list[int]] = {}
    for k in sorted(np.unique(pliegue_de_fila).tolist()):
        mapeo[int(k)] = sorted(np.unique(pliegues_final[pliegue_de_fila == k]).tolist())
    biyeccion = all(len(v) == 1 for v in mapeo.values()) and \
        len({v[0] for v in mapeo.values()}) == len(mapeo)
    particion_igual = False
    if biyeccion:
        traducida = np.array([mapeo[int(v)][0] for v in pliegue_de_fila])
        particion_igual = bool(np.array_equal(traducida, pliegues_final))
    print(f"  pliegues en v2_oof_base_corrected: "
          f"{'si' if pliegues_en_bases else 'no (no se guardaron)'}")
    print(f"  igualdad estricta (misma numeracion): {iguales_estricto}")
    print(f"  particion identica tras traducir numeracion: {particion_igual} "
          f"| mapeo recomputado->guardado: "
          f"{ {k: v[0] if len(v) == 1 else v for k, v in mapeo.items()} }")
    evidencia["pliegues"] = {
        "tamanos": tamanos,
        "sin_asignar": sin_asignar,
        "pliegues_con_grupos_compartidos": fugas,
        "v2_oof_base_corrected_guarda_pliegues": pliegues_en_bases,
        "igualdad_estricta_misma_numeracion": iguales_estricto,
        "particion_identica_tras_traducir_numeracion": particion_igual,
        "mapeo_recomputado_a_guardado": {str(k): v for k, v in mapeo.items()},
        "seed": 42,
        "n_splits": 5,
    }

    # y_true alineado en el artefacto de bases
    y_art = np.asarray(bases["y_true"])
    y_train_dev = orden_dev[["HS", "TR", "AG"]].to_numpy()
    evidencia["alineacion"] = {
        "y_true_v2_bases_igual_a_train_dev": bool(mismo_orden and np.array_equal(y_art, y_train_dev)),
    }
    print(f"  y_true de v2_oof_base_corrected == etiquetas de train_dev: "
          f"{evidencia['alineacion']['y_true_v2_bases_igual_a_train_dev']}")

    # --- escaneos estaticos ---
    print()
    print("=== D. Escaneos estaticos (referencias a test/holdout/holdout2) ===")
    scripts = sorted((RAIZ / "experiments").glob("*.py"))
    lineas_scan: list[str] = ["# P96-02: escaneos estaticos de scripts\n"]

    def scan(nombre: str, patron: str) -> list[str]:
        encontrados = []
        for s in scripts:
            try:
                texto = s.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            if re.search(patron, texto):
                encontrados.append(s.name)
        lineas_scan.append(f"\n## {nombre}: {len(encontrados)} scripts\n")
        lineas_scan.extend(f"- {n}" for n in encontrados)
        return encontrados

    ref_test = scan("referencias a toxic_test / test.csv", r"toxic_test|test\.csv")
    ref_holdout = scan("referencias a holdout (historico)", r"holdout")
    ref_holdout2 = scan("referencias a holdout2", r"holdout2")
    print(f"  scripts que mencionan test: {len(ref_test)}")
    print(f"  scripts que mencionan holdout: {len(ref_holdout)}")
    print(f"  scripts que mencionan holdout2: {len(ref_holdout2)}: {ref_holdout2}")

    # contexto de fit sobre test: lineas con fit y toxic_test cercanos
    sospechosos: list[str] = []
    for s in scripts:
        try:
            lineas = s.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        for k, linea in enumerate(lineas):
            if "toxic_test" in linea or "test.csv" in linea:
                vecindad = "\n".join(lineas[max(0, k - 12):k + 12])
                if re.search(r"\.fit\(|fit_transform\(", vecindad):
                    sospechosos.append(f"{s.name}:{k + 1}")
    lineas_scan.append(f"\n## vecindades de test con fit cercano: "
                       f"{len(sospechosos)}\n")
    lineas_scan.extend(f"- {n}" for n in sospechosos)
    print(f"  vecindades test+fit (revision manual): {sospechosos or 'ninguna'}")

    inv_fits = subprocess.run(
        ["grep", "-n", "-E", r"\.fit\(|fit_transform\(", "--include=*.py", "-r",
         str(RAIZ / "experiments")],
        capture_output=True, text=True).stdout
    lineas_scan.append(f"\n## inventario de .fit/fit_transform "
                       f"({len(inv_fits.splitlines())} ocurrencias)\n")
    lineas_scan.append(inv_fits)
    SALIDA_SCAN.write_text("\n".join(lineas_scan), encoding="utf-8")

    evidencia["escaneos"] = {
        "scripts_con_test": len(ref_test),
        "scripts_con_holdout": len(ref_holdout),
        "scripts_con_holdout2": ref_holdout2,
        "vecindades_test_fit": sospechosos,
        "ocurrencias_fit_total": len(inv_fits.splitlines()),
    }
    SALIDA_JSON.write_text(json.dumps(evidencia, indent=2, ensure_ascii=True) + "\n",
                           encoding="utf-8")
    print()
    print(f"P96-02 folds/leakage: ejecutado | evidencia: {SALIDA_JSON.relative_to(RAIZ)}")
    print(f"  escaneos: {SALIDA_SCAN.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()
