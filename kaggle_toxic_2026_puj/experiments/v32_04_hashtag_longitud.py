"""V32-04: Fase 4 de V32. El mecanismo de interacciones aplicado a hashtag.

Primero solo las tres de hashtag x longitud; las de posicion solo si hay senal.

Uso
---
    /home/carlos/venv/bin/python experiments/v32_04_hashtag_longitud.py
"""

from __future__ import annotations

import json
import resource
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import read_train
from src.splits import grupos_fusionados, pliegues_corregidos
from src.structural_eval import (
    ETIQUETAS,
    LR_PARAMS,
    UMBRALES_V03,
    comparar,
    evaluar_bloque,
    matriz_v03,
    metricas_multietiqueta,
    resumen,
    verificar_control,
)
from src.structural_features import columnas, extraer, interacciones

EXPERIMENTO = "V32-04 (Fase 4 de V32: interacciones de hashtag con la longitud)"
REPORTE_JSON = PROJECT_ROOT / "reports" / "v32_04_hashtag_longitud.json"
REPORTE_MD = PROJECT_ROOT / "reports" / "v32_04_hashtag_longitud.md"
PARTICION = PROJECT_ROOT / "data" / "processed" / "holdout_v2_ids.csv"
BLOQUE = "mencion"
TRES = ("int__hashtag_x_corto", "int__hashtag_x_medio", "int__hashtag_x_largo")
DOS = ("int__hashtag_inicial_x_medio", "int__hashtag_final_x_largo")


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[V32-04] {mensaje}", flush=True)


def memoria_pico_mb() -> float:
    """Resident set maximum del proceso, en megabytes.

    Returns:
        La memoria maxima residente en MB.
    """
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _evaluar(
    train_dev: pd.DataFrame,
    pliegues: list[tuple[np.ndarray, np.ndarray]],
    permitidas: tuple[str, ...],
) -> dict[str, Any]:
    """Evalua V03 mas las interacciones de hashtag permitidas.

    Args:
        train_dev: DataFrame del desarrollo.
        pliegues: pliegues canonicos.
        permitidas: nombres de columna a conservar.

    Returns:
        El resultado de la evaluacion.
    """
    from scipy import sparse

    y_true = train_dev[list(ETIQUETAS)].to_numpy(dtype=np.int64)
    n = len(train_dev)
    proba = np.zeros((n, 3), dtype=float)
    base = pd.DataFrame(extraer(BLOQUE, train_dev["text"].astype(str).tolist()))
    base = base[columnas(BLOQUE)]
    tokens = np.array([len(t.split()) for t in train_dev["text"].astype(str)], dtype=float)
    filas = [
        {k: v for k, v in interacciones(BLOQUE, fila, n).items() if k in permitidas}
        for fila, n in zip(base.to_dict("records"), tokens, strict=True)
    ]
    features = pd.DataFrame(filas).reset_index(drop=True)
    metricas = []
    train_f1s = []
    for numero, (idx_tr, idx_va) in enumerate(pliegues, start=1):
        train, val = matriz_v03(train_dev["text"].iloc[idx_tr], train_dev["text"].iloc[idx_va])
        train = sparse.hstack([train, features.iloc[idx_tr].to_numpy()]).tocsr()
        val = sparse.hstack([val, features.iloc[idx_va].to_numpy()]).tocsr()
        pred_train = np.zeros((len(idx_tr), 3), dtype=np.int64)
        for j, etiqueta in enumerate(ETIQUETAS):
            modelo = LogisticRegression(**LR_PARAMS)
            modelo.fit(train, train_dev[etiqueta].iloc[idx_tr])
            proba[idx_va, j] = modelo.predict_proba(val)[:, 1]
            pred_train[:, j] = (
                modelo.predict_proba(train)[:, 1] >= UMBRALES_V03[etiqueta]
            ).astype(int)
        pred_pliegue = np.column_stack(
            [(proba[idx_va, j] >= UMBRALES_V03[et]).astype(int) for j, et in enumerate(ETIQUETAS)]
        )
        metrica = metricas_multietiqueta(y_true[idx_va], pred_pliegue)
        metrica["pliegue"] = numero
        metricas.append(metrica)
        train_f1s.append(metricas_multietiqueta(y_true[idx_tr], pred_train)["f1_media_columnas"])
    pred = np.column_stack(
        [(proba[:, j] >= UMBRALES_V03[et]).astype(int) for j, et in enumerate(ETIQUETAS)]
    )
    return {"proba": proba, "pred": pred, "metricas_pliegues": metricas,
            "y_true": y_true, "train_f1": float(np.mean(train_f1s))}


def main() -> None:
    """Ejecuta la Fase 4 de V32.

    Returns:
        None.
    """
    inicio = time.time()
    log(f"{EXPERIMENTO}. NINGUNA submission")

    train = read_train()
    particion = pd.read_csv(PARTICION)
    ids_dev = particion.loc[particion["particion"] == "dev", "id"].astype(int).tolist()
    desarrollo = train[train["id"].astype(int).isin(ids_dev)].reset_index(drop=True)
    pliegues = pliegues_corregidos(desarrollo, grupos_fusionados(desarrollo))
    y = desarrollo[list(ETIQUETAS)].to_numpy(dtype=np.int64)
    log(f"   desarrollo {len(desarrollo)} filas | {len(pliegues)} pliegues")

    log("1/3 control de V03")
    control = resumen(evaluar_bloque(desarrollo, pliegues, bloque=None))
    verificar_control(control["corrected_cv"])
    log(f"   control exacto: {control['corrected_cv']:.10f}")

    log("2/3 V32-04a: solo las tres de hashtag x longitud")
    a = resumen(_evaluar(desarrollo, pliegues, TRES))
    delta_a = a["corrected_cv"] - control["corrected_cv"]
    log(f"   V32-04a: {a['corrected_cv']:.10f} | delta {delta_a:+.10f}")

    log("3/3 decision e informe")
    hay_senal = delta_a > 0
    if hay_senal:
        b = resumen(_evaluar(desarrollo, pliegues, TRES + DOS))
        comp_b = comparar(y, evaluar_bloque(desarrollo, pliegues, bloque=None)["pred"],
                          _evaluar(desarrollo, pliegues, TRES + DOS)["pred"])
        log(f"   V32-04b: {b['corrected_cv']:.10f} | delta {b['corrected_cv'] - control['corrected_cv']:+.10f}")
        log(f"   V32-04b: rescates {comp_b['rescates']} danios {comp_b['danios']} neto {comp_b['neto']:+d}")
    else:
        b = None
        comp_b = None
        log("   sin senal en 04a, no se prueban las de posicion")

    informe = {
        "familia": "STRUCTURAL_TEXT_FEATURES",
        "experimento": EXPERIMENTO,
        "fecha": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "estado": "OOF_EVALUATED",
        "bloque": "hashtag",
        "fase": "4_HASHTAG_LONGITUD",
        "features_tres": list(TRES),
        "features_dos": list(DOS),
        "control": control,
        "V32_04a_tres_interacciones": a,
        "V32_04b_con_posicion": b,
        "hay_senal_para_posicion": hay_senal,
        "comparacion_04a": comparar(
            y, evaluar_bloque(desarrollo, pliegues, bloque=None)["pred"],
            _evaluar(desarrollo, pliegues, TRES)["pred"],
        ),
        "comparacion_04b": comp_b,
        "gastos": {"submissions": 0, "modelos_entrenados": 3 if hay_senal else 2,
                   "memoria_pico_mb": round(memoria_pico_mb(), 1),
                   "segundos": round(time.time() - inicio, 1)},
    }
    REPORTE_JSON.write_text(
        json.dumps(informe, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _informe_md(informe)
    log("   informe escrito en reports/v32_04_hashtag_longitud.{json,md}")
    log(f"   memoria pico {memoria_pico_mb():.0f} MB | tiempo {round(time.time() - inicio, 1)} s")
    log("VEREDICTO: Fase 4 evaluada, 0 submissions")


def _informe_md(informe: dict[str, Any]) -> None:
    """Escribe el informe legible.

    Args:
        informe: informe con los datos.

    Returns:
        None.
    """
    c = informe["control"]
    a = informe["V32_04a_tres_interacciones"]
    b = informe["V32_04b_con_posicion"]
    g = informe["gastos"]
    lineas = [
        "# V32-04: Fase 4 de V32, interacciones de hashtag con la longitud",
        "",
        f"Generado por `experiments/v32_04_hashtag_longitud.py` ({EXPERIMENTO}).",
        "",
        f"**Estado `{informe['estado']}`. Alcance: interacciones de hashtag con la longitud.**",
        "",
        "| variante | corrected_cv | delta vs V03 |",
        "|---|---:|---:|",
        f"| V03 (control) | {c['corrected_cv']:.10f} | 0 |",
        (f"| V32-04a tres interacciones | {a['corrected_cv']:.10f} | "
        f"{a['corrected_cv'] - c['corrected_cv']:+.10f} |"),
    ]
    if b is not None:
        lineas.append(
            f"| V32-04b + posicion | {b['corrected_cv']:.10f} | "
            f"{b['corrected_cv'] - c['corrected_cv']:+.10f} |"
        )
    else:
        lineas.append("| V32-04b + posicion | no ejecutada | sin senal en 04a |")
    lineas += [
        "",
        (f"Rescates y danos de 04a: {informe['comparacion_04a']['rescates']} / "
        f"{informe['comparacion_04a']['danios']} / neto {informe['comparacion_04a']['neto']:+d}."),
        "",
        "## Lo que este script no hace",
        "",
        "- No usa el contenido del hashtag.",
        "- No combina con URL ni con mencion.",
        "- No genera submission. No lee el test.",
        "",
        (f"Coste: {g['segundos']} s, memoria {g['memoria_pico_mb']:.0f} MB, "
        f"{g['modelos_entrenados']} modelos, {g['submissions']} submissions."),
        "",
    ]
    REPORTE_MD.write_text("\n".join(lineas) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
