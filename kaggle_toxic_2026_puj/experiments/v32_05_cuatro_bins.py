"""V32-05: Fase 6 de V32. Representacion de longitud con 4 bins en vez de 3.

Que responde
------------
La Fase 6 de la directiva V32. Las interacciones URL x longitud se reproducen y son reales, asi
que toca probar si una discretizacion algo mas fina aporta algo.

La regla de la directiva: **no hacer un barrido**. Como maximo comparar 3 bins contra 4 bins, y
los limites deben fijarse usando unicamente el training fold, nunca mirando el resultado OOF.

Los limites
----------
Se usan cuantiles del training fold, no umbrales fijos. Con 3 bins los cortes son el percentil 33 y
el 67 de los tokens del training fold. Con 4 bins son el 25, 50 y 75. Asi los bins se adaptan a la
distribucion de cada pliegue y no se eligen mirando el resultado.

Uso
---
    /home/carlos/venv/bin/python experiments/v32_05_cuatro_bins.py
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
from src.structural_features import columnas, extraer

EXPERIMENTO = "V32-05 (Fase 6 de V32: 4 bins de longitud en vez de 3)"
REPORTE_JSON = PROJECT_ROOT / "reports" / "v32_05_cuatro_bins.json"
REPORTE_MD = PROJECT_ROOT / "reports" / "v32_05_cuatro_bins.md"
PARTICION = PROJECT_ROOT / "data" / "processed" / "holdout_v2_ids.csv"
BLOQUE = "url"


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[V32-05] {mensaje}", flush=True)


def memoria_pico_mb() -> float:
    """Resident set maximum del proceso, en megabytes.

    Returns:
        La memoria maxima residente en MB.
    """
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def interacciones_n_bins(
    fila: dict[str, float], n_tokens: float, cortes: tuple[float, ...]
) -> dict[str, float]:
    """Interacciones de URL con la longitud, con un numero de bins arbitrario.

    Los cortes vienen dados y no se calculan aqui: el script que llama es el que los toma del
    training fold, que es la regla de la directiva.

    Args:
        fila: features del bloque URL.
        n_tokens: numero de tokens del texto.
        cortes: limites entre bins, en orden ascendente.

    Returns:
        Diccionario con las flags de interaccion.
    """
    tramo = 0
    for corte in cortes:
        if n_tokens > corte:
            tramo += 1
    tiene = fila["url_n"] > 0
    # Las claves son fijas y no dependen del tramo: si se generaran dinamicamente, cada fila
    # tendria claves distintas y pd.DataFrame rellena las que faltan con NaN, que es justo lo
    # que rompio la primera corrida.
    salida = {f"url_x_b{k}": 0.0 for k in range(len(cortes) + 1)}
    salida[f"url_x_b{tramo}"] = float(tiene)
    salida["url_query_x_largo"] = float(
        fila["url_con_query"] > 0 and tramo == len(cortes)
    )
    salida["url_corta_x_corto"] = float(fila["url_es_corta"] > 0 and tramo == 0)
    return salida


def evaluar_n_bins(
    train_dev: pd.DataFrame,
    pliegues: list[tuple[np.ndarray, np.ndarray]],
    n_bins: int,
) -> dict[str, Any]:
    """Evalua V03 + URL aditivo + interacciones con un numero de bins dado.

    Los cortes se calculan **dentro de cada training fold**, nunca sobre todo el desarrollo.

    Args:
        train_dev: DataFrame del desarrollo.
        pliegues: pliegues canonicos.
        n_bins: numero de bins de longitud.

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
    metricas = []
    train_f1s = []
    for numero, (idx_tr, idx_va) in enumerate(pliegues, start=1):
        # Los cortes se toman del training fold, que es la regla de la directiva.
        cortes = tuple(float(x) for x in np.percentile(tokens[idx_tr], np.linspace(0, 100, n_bins + 1)[1:-1]))
        filas = [
            {f"int_{k}": v for k, v in interacciones_n_bins(fila, tok, cortes).items()}
            for fila, tok in zip(base.to_dict("records"), tokens, strict=True)
        ]
        features = pd.DataFrame(filas).reset_index(drop=True)
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
    """Ejecuta la Fase 6 de V32.

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

    log("1/4 control de V03")
    control = resumen(evaluar_bloque(desarrollo, pliegues, bloque=None))
    verificar_control(control["corrected_cv"])
    log(f"   control exacto: {control['corrected_cv']:.10f}")

    log("2/4 3 bins (referencia de V31-04)")
    tres = resumen(evaluar_n_bins(desarrollo, pliegues, 3))
    log(f"   3 bins: {tres['corrected_cv']:.10f} | delta {tres['corrected_cv'] - control['corrected_cv']:+.10f}")

    log("3/4 4 bins")
    cuatro = resumen(evaluar_n_bins(desarrollo, pliegues, 4))
    log(f"   4 bins: {cuatro['corrected_cv']:.10f} | delta {cuatro['corrected_cv'] - control['corrected_cv']:+.10f}")
    log(f"   4 bins sobre 3 bins: {cuatro['corrected_cv'] - tres['corrected_cv']:+.10f}")

    log("4/4 informe")
    comp_4 = comparar(y, evaluar_bloque(desarrollo, pliegues, bloque=None)["pred"],
                      evaluar_n_bins(desarrollo, pliegues, 4)["pred"])
    log(f"   4 bins: rescates {comp_4['rescates']} danios {comp_4['danios']} neto {comp_4['neto']:+d}")

    informe = {
        "familia": "STRUCTURAL_TEXT_FEATURES",
        "experimento": EXPERIMENTO,
        "fecha": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "estado": "OOF_EVALUATED",
        "fase": "6_CUATRO_BINS",
        "pregunta": "una discretizacion mas fina de la longitud aporta algo sobre los 3 bins",
        "alcance": "solo 3 bins contra 4 bins, con cortes tomados del training fold",
        "regla_de_cortes": "cuartiles del training fold, nunca umbrales fijos ni elegidos mirando OOF",
        "control": control,
        "tres_bins": tres,
        "cuatro_bins": cuatro,
        "cuatro_menos_tres": cuatro["corrected_cv"] - tres["corrected_cv"],
        "comparacion_cuatro": comp_4,
        "gastos": {"submissions": 0, "modelos_entrenados": 3,
                   "memoria_pico_mb": round(memoria_pico_mb(), 1),
                   "segundos": round(time.time() - inicio, 1)},
    }
    REPORTE_JSON.write_text(
        json.dumps(informe, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _informe_md(informe)
    log("   informe escrito en reports/v32_05_cuatro_bins.{json,md}")
    log(f"   memoria pico {memoria_pico_mb():.0f} MB | tiempo {round(time.time() - inicio, 1)} s")
    log("VEREDICTO: Fase 6 evaluada, 0 submissions")


def _informe_md(informe: dict[str, Any]) -> None:
    """Escribe el informe legible.

    Args:
        informe: informe con los datos.

    Returns:
        None.
    """
    c = informe["control"]
    t = informe["tres_bins"]
    b = informe["cuatro_bins"]
    comp = informe["comparacion_cuatro"]
    g = informe["gastos"]
    lineas = [
        "# V32-05: Fase 6 de V32, 4 bins de longitud en vez de 3",
        "",
        f"Generado por `experiments/v32_05_cuatro_bins.py` ({EXPERIMENTO}).",
        "",
        f"**Estado `{informe['estado']}`. Alcance: {informe['alcance']}**",
        "",
        f"{informe['regla_de_cortes']}.",
        "",
        "## 1. Resultado",
        "",
        "| variante | corrected_cv | delta vs V03 |",
        "|---|---:|---:|",
        f"| V03 (control) | {c['corrected_cv']:.10f} | 0 |",
        f"| 3 bins | {t['corrected_cv']:.10f} | {t['corrected_cv'] - c['corrected_cv']:+.10f} |",
        f"| 4 bins | {b['corrected_cv']:.10f} | {b['corrected_cv'] - c['corrected_cv']:+.10f} |",
        "",
        f"**4 bins sobre 3 bins: {informe['cuatro_menos_tres']:+.10f}**",
        "",
        (f"Rescates y danios de 4 bins: {comp['rescates']} / {comp['danios']} / "
        f"neto {comp['neto']:+d}."),
        "",
        "## 2. Lo que este script no hace",
        "",
        "- No hace un barrido de bins: solo 3 contra 4.",
        "- No elige los cortes mirando el OOF.",
        "- No genera submission. No lee el test.",
        "",
        (f"Coste: {g['segundos']} s, memoria {g['memoria_pico_mb']:.0f} MB, "
        f"{g['modelos_entrenados']} modelos, {g['submissions']} submissions."),
        "",
    ]
    REPORTE_MD.write_text("\n".join(lineas) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
