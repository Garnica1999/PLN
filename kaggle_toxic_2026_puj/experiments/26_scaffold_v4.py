"""Scaffold de la fase v4: verificacion de la referencia V03 y del protocolo.

Contexto
--------
La fase v2 (D-021) se cerro con 24/24 lineas evaluadas y V03 como modelo final
(LB 0.78306). La fase v4 (Advanced Research v4) exige, antes de cualquier
experimento nuevo, tres controles:

1. que V03 siga intacta y sea reconstruible byte a byte (hashes congelados);
2. que la metrica de referencia con el protocolo v2 (D-016) se reproduzca
   exactamente desde el OOF canonico congelado;
3. que el presupuesto de control (holdout congelado y sondas de leaderboard)
   este accounted antes de empezar, porque es un recurso agotable.

Este script no entrena nada: solo verifica, recalcula metricas desde el OOF
almacenado y escribe un informe JSON. No lee el conjunto de test en ningun
momento (regla D-002/D-016).

Uso
---
    /home/carlos/venv/bin/python experiments/26_scaffold_v4.py

Salidas
-------
- ``reports/baseline_v03_v4_check.json``: informe de verificacion de la
  referencia y del protocolo para la fase v4.
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import read_train  # noqa: E402
from src.evaluation import metricas_multietiqueta  # noqa: E402
from src.splits import (  # noqa: E402
    ETIQUETAS,
    grupos_fusionados,
    holdout_v2,
    pliegues_corregidos,
)

FROZEN_DIR = PROJECT_ROOT / "experiments" / "baseline_v03_frozen"
HASHES_PATH = FROZEN_DIR / "hashes.txt"
PIPELINE_PATH = FROZEN_DIR / "pipeline_v03.json"
OOF_CANONICO_PATH = PROJECT_ROOT / "features" / "oof_v03_final.npz"
HOLDOUT_EVALUACIONES_PATH = PROJECT_ROOT / "reports" / "holdout_evaluaciones_v2.csv"
LEADERBOARD_LOG_PATH = PROJECT_ROOT / "submissions" / "leaderboard_log.csv"
REPORTE_PATH = PROJECT_ROOT / "reports" / "baseline_v03_v4_check.json"

HOLDOUT_MAX_EVALUACIONES = 5
TOLERANCIA_METRICA = 1e-6


def log(mensaje: str) -> None:
    """Imprime una linea de progreso con marca de seccion.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[V4-00] {mensaje}", flush=True)


# ---------------------------------------------------------------------------
# 1. Integridad de la referencia congelada
# ---------------------------------------------------------------------------
def verificar_hashes() -> dict[str, Any]:
    """Recalcula el sha256 de cada artefacto congelado y lo contrasta.

    Returns:
        Diccionario con ``ok`` global, el numero de artefactos verificados y
        el detalle por archivo (ruta, hash esperado y hash calculado).
    """
    log("1/4 Integridad de V03 congelada (sha256)")
    esperado: dict[str, str] = {}
    for linea in HASHES_PATH.read_text(encoding="utf-8").splitlines():
        if not linea.strip():
            continue
        hash_esperado, ruta = linea.split(maxsplit=1)
        esperado[ruta.strip()] = hash_esperado

    detalles = []
    todo_ok = True
    for ruta_relativa, hash_esperado in esperado.items():
        # Las dos ultimas entradas son rutas relativas a la carpeta congelada.
        ruta = (FROZEN_DIR / ruta_relativa).resolve()
        if not ruta.exists():
            ruta = (PROJECT_ROOT / ruta_relativa).resolve()
        if not ruta.exists():
            detalles.append(
                {"ruta": ruta_relativa, "estado": "FALTA", "hash_esperado": hash_esperado}
            )
            todo_ok = False
            continue
        calculado = hashlib.sha256(ruta.read_bytes()).hexdigest()
        coincide = calculado == hash_esperado
        todo_ok = todo_ok and coincide
        detalles.append(
            {
                "ruta": str(ruta.relative_to(PROJECT_ROOT)),
                "estado": "OK" if coincide else "DIFIERE",
                "hash_esperado": hash_esperado,
                "hash_calculado": calculado,
            }
        )
        marca = "OK  " if coincide else "FALLA"
        log(f"   {marca} {ruta.name}")
    return {"ok": bool(todo_ok), "n_artefactos": len(detalles), "detalle": detalles}


# ---------------------------------------------------------------------------
# 2. Reproduccion de la metrica de referencia desde el OOF canonico
# ---------------------------------------------------------------------------
def verificar_oof_canonico() -> dict[str, Any]:
    """Recalcula la metrica de referencia de V03 en el protocolo v2.

    El OOF canonico (``features/oof_v03_final.npz``) contiene las
    probabilidades fuera de pliegue de V03 sobre el conjunto de desarrollo, el
    vector de pliegues y los umbrales congelados. La verificacion comprueba que
    la metrica recalculada coincide con la almacenada y reporta el desglose por
    etiqueta y por pliegue.

    Returns:
        Diccionario con la metrica recalculada, la almacenada, su diferencia y
        el desglose por etiqueta y pliegue.
    """
    log("2/4 Reproduccion de la referencia con el protocolo v2")
    datos = np.load(OOF_CANONICO_PATH, allow_pickle=True)
    y_true = datos["y_true"]
    proba = datos["proba"]
    pred = datos["pred"]
    pliegues = datos["pliegues"]
    umbrales = datos["umbrales"]

    metricas_umbrales = metricas_multietiqueta(y_true, pred)
    metricas_cinco = metricas_multietiqueta(y_true, (proba >= 0.5).astype(int))
    metricas_umbral_cinco = metricas_multietiqueta(
        y_true, (proba >= umbrales[None, :]).astype(int)
    )

    referencia = float(datos["corrected_cv"][0])
    recalculada = metricas_umbrales["f1_media_columnas"]
    diferencia = abs(referencia - recalculada)

    f1_por_pliegue = []
    for pliegue in sorted(np.unique(pliegues)):
        mascara = pliegues == pliegue
        f1_por_pliegue.append(
            float(
                metricas_multietiqueta(y_true[mascara], pred[mascara])[
                    "f1_media_columnas"
                ]
            )
        )

    log(f"   corrected_cv almacenada : {referencia:.6f}")
    log(f"   corrected_cv recalculada: {recalculada:.6f}")
    log(f"   M2 con umbral 0.5       : {metricas_cinco['f1_media_columnas']:.6f}")
    log(f"   umbrales congelados     : {list(np.round(umbrales, 2))}")
    log(f"   F1 por pliegue          : {[round(v, 4) for v in f1_por_pliegue]}")

    return {
        "ok": bool(diferencia <= TOLERANCIA_METRICA),
        "n_dev": int(y_true.shape[0]),
        "corrected_cv_almacenada": referencia,
        "corrected_cv_recalculada": recalculada,
        "diferencia": diferencia,
        "corrected_cv_umbral_0_5": metricas_cinco["f1_media_columnas"],
        "corrected_cv_umbrales_frozen": metricas_umbral_cinco["f1_media_columnas"],
        "umbrales": [float(valor) for valor in umbrales],
        "f1_por_etiqueta": metricas_umbrales["f1_por_etiqueta"],
        "f1_por_pliegue": f1_por_pliegue,
        "std_pliegues": float(np.std(f1_por_pliegue, ddof=1)),
        "fuente_oof": str(datos["fuente"][0]),
    }


# ---------------------------------------------------------------------------
# 3. Protocolo: desarrollo, holdout y ausencia de fuga entre grupos
# ---------------------------------------------------------------------------
def verificar_protocolo() -> dict[str, Any]:
    """Recalcula las particiones del protocolo v2 y comprueba su integridad.

    Returns:
        Diccionario con los tamaños de desarrollo y holdout, el numero de
        grupos, la distribucion de combinaciones y la confirmacion de que las
        particiones no comparten grupos.
    """
    log("3/4 Protocolo v2 (grupos fusionados, desarrollo/holdout)")
    train = read_train()
    grupos = grupos_fusionados(train)
    indices_dev, indices_holdout = holdout_v2(train, grupos)
    desarrollo = train.iloc[indices_dev].reset_index(drop=True)
    grupos_dev = grupos[indices_dev]
    pliegues = pliegues_corregidos(desarrollo, grupos_dev)

    combinaciones_dev = (
        desarrollo["HS"].astype(str)
        + desarrollo["TR"].astype(str)
        + desarrollo["AG"].astype(str)
    ).value_counts().sort_index().to_dict()
    combinaciones_holdout = (
        train.iloc[indices_holdout]["HS"].astype(str)
        + train.iloc[indices_holdout]["TR"].astype(str)
        + train.iloc[indices_holdout]["AG"].astype(str)
    ).value_counts().sort_index().to_dict()

    # `_verificar_sin_fuga` lanza excepcion si hay solape; aqui se comprueba
    # explicitamente para dejar constancia en el informe.
    grupos_compartidos = len(set(grupos_dev.tolist()) & set(grupos[indices_holdout].tolist()))

    log(f"   train total            : {len(train)}")
    log(f"   desarrollo             : {len(indices_dev)}")
    log(f"   holdout congelado      : {len(indices_holdout)}")
    log(f"   grupos en train        : {len(set(grupos.tolist()))}")
    log(f"   pliegues corregidos    : {len(pliegues)}")
    log(f"   grupos compartidos     : {grupos_compartidos} (debe ser 0)")

    return {
        "ok": bool(grupos_compartidos == 0 and len(pliegues) == 5),
        "n_train": int(len(train)),
        "n_dev": int(len(indices_dev)),
        "n_holdout": int(len(indices_holdout)),
        "n_grupos_train": int(len(set(grupos.tolist()))),
        "n_pliegues_dev": len(pliegues),
        "grupos_compartidos_dev_holdout": int(grupos_compartidos),
        "distribucion_combinaciones_dev": {str(k): int(v) for k, v in combinaciones_dev.items()},
        "distribucion_combinaciones_holdout": {
            str(k): int(v) for k, v in combinaciones_holdout.items()
        },
        "etiquetas": list(ETIQUETAS),
    }


# ---------------------------------------------------------------------------
# 4. Presupuesto de control: holdout y sondas de leaderboard
# ---------------------------------------------------------------------------
def verificar_presupuesto() -> dict[str, Any]:
    """Cuenta las evaluaciones de holdout usadas y las sondas de LB registradas.

    Returns:
        Diccionario con el consumo de ambos presupuestos y las decisiones
        vigentes que los gobiernan.
    """
    log("4/4 Presupuesto de control (holdout y leaderboard)")
    evaluaciones = pd.read_csv(HOLDOUT_EVALUACIONES_PATH)
    usadas = int(len(evaluaciones))

    registro_lb = pd.read_csv(LEADERBOARD_LOG_PATH)
    sondas = int(len(registro_lb))
    mejor = registro_lb.loc[registro_lb["kaggle_f1"].idxmax()]

    log(f"   evaluaciones de holdout: {usadas}/{HOLDOUT_MAX_EVALUACIONES}")
    log(f"   sondas de leaderboard  : {sondas}")
    log(f"   mejor score de LB      : {mejor['kaggle_f1']} ({mejor['submission_id']})")

    return {
        "ok": bool(usadas <= HOLDOUT_MAX_EVALUACIONES),
        "holdout_evaluaciones_usadas": usadas,
        "holdout_evaluaciones_maximas": HOLDOUT_MAX_EVALUACIONES,
        "holdout_evaluaciones_restantes": HOLDOUT_MAX_EVALUACIONES - usadas,
        "sondas_lb_registradas": sondas,
        "mejor_lb": float(mejor["kaggle_f1"]),
        "mejor_lb_submission": str(mejor["submission_id"]),
        "politica_v4": (
            "La ultima evaluacion de holdout se reserva para el unico mejor "
            "finalista de v4; el envio al leaderboard queda a decision de Carlos "
            "solo si el candidato pasa todos los gates (D-025)."
        ),
    }


# ---------------------------------------------------------------------------
# Orquestacion
# ---------------------------------------------------------------------------
def main() -> None:
    """Ejecuta la verificacion completa del scaffold de la fase v4."""
    inicio = datetime.now(timezone.utc)
    log("Scaffold v4: verificacion de la referencia y del protocolo (sin entrenar)")

    pipeline = json.loads(PIPELINE_PATH.read_text(encoding="utf-8"))
    hashes = verificar_hashes()
    oof = verificar_oof_canonico()
    protocolo = verificar_protocolo()
    presupuesto = verificar_presupuesto()

    informe = {
        "experiment_id": "V4-00-SCAFFOLD",
        "fase": "P0 - scaffold y verificacion de la referencia",
        "fecha": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "duracion_s": round((datetime.now(timezone.utc) - inicio).total_seconds(), 2),
        "referencia": {
            "modelo": pipeline["modelo"],
            "version": pipeline["version"],
            "umbrales": pipeline["umbrales"],
            "kaggle_lb": pipeline["referencias"]["kaggle_lb"],
            "corrected_cv_referencia": pipeline["referencias_protocolo_v2"][
                "corrected_cv_m2_con_umbrales"
            ],
        },
        "integridad_v03": hashes,
        "metrica_referencia": oof,
        "protocolo": protocolo,
        "presupuesto": presupuesto,
        "uso_del_test": "ninguno: el script no lee data/raw/toxic_test.csv",
    }
    veredicto = bool(
        hashes["ok"] and oof["ok"] and protocolo["ok"] and presupuesto["ok"]
    )
    informe["veredicto"] = "OK" if veredicto else "REVISAR"

    REPORTE_PATH.write_text(
        json.dumps(informe, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    log(f"Veredicto: {informe['veredicto']}")
    log(f"Informe escrito en {REPORTE_PATH.relative_to(PROJECT_ROOT)}")
    if not veredicto:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
