"""V32-02: aislar la fuente de la ganancia de V31-04.

Que responde
------------
La Fase 2 de la directiva V32. V31-04 gana +0,004167, pero no se sabe si esa ganancia viene de
las interacciones o del bloque URL aditivo. Este script ejecuta exactamente las tres variantes que
pide la directiva:

- **V32-A**: URL aditivo (el Control A de V31).
- **V32-B**: solo las cinco interacciones, sin las variables URL aditivas.
- **V32-C**: URL aditivo + interacciones (debe reproducir V31-04).

Y eso permite descomponer:

    V03 -> URL                    (lo que aporta el bloque aditivo)
    URL -> URL + interacciones    (lo que aportan las interacciones)

La descomposicion es la que evita la interpretacion incorrecta: si V32-B da casi todo el resultado
de V32-C, la ganancia esta en las interacciones y el bloque aditivo es casi irrelevante. Si V32-B
da poco y V32-A tambien, entonces la ganancia esta en la combinacion.

Uso
---
    /home/carlos/venv/bin/python experiments/v32_02_aislar_fuente.py
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
from src.structural_features import (
    columnas,
    extraer,
    interacciones,
)

EXPERIMENTO = "V32-02 (aislar la fuente de la ganancia de V31-04)"
REPORTE_JSON = PROJECT_ROOT / "reports" / "v32_02_aislar_fuente.json"
REPORTE_MD = PROJECT_ROOT / "reports" / "v32_02_aislar_fuente.md"
PARTICION = PROJECT_ROOT / "data" / "processed" / "holdout_v2_ids.csv"
BLOQUE = "url"
TRAMOS = (12, 22)


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[V32-02] {mensaje}", flush=True)


def memoria_pico_mb() -> float:
    """Resident set maximum del proceso, en megabytes.

    Returns:
        La memoria maxima residente en MB.
    """
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _solo_interacciones(textos: list[str]) -> pd.DataFrame:
    """Construye la matriz de solo las cinco interacciones, sin las variables URL aditivas.

    Args:
        textos: textos a procesar.

    Returns:
        DataFrame con las cinco columnas de interaccion.
    """
    base = pd.DataFrame(extraer(BLOQUE, textos))
    base = base[columnas(BLOQUE)]
    tokens = np.array([len(t.split()) for t in textos], dtype=float)
    filas = [
        {f"int_{k}": v for k, v in interacciones(BLOQUE, fila, n).items()}
        for fila, n in zip(base.to_dict("records"), tokens, strict=True)
    ]
    return pd.DataFrame(filas).reset_index(drop=True)


def evaluar_solo_interacciones(
    train_dev: pd.DataFrame,
    pliegues: list[tuple[np.ndarray, np.ndarray]],
) -> dict[str, Any]:
    """Evalua V03 mas solo las interacciones, sin las variables URL aditivas.

    Args:
        train_dev: DataFrame del desarrollo.
        pliegues: pliegues canonicos.

    Returns:
        El resultado de ``evaluar_bloque`` con la matriz sustituida.
    """

    y_true = train_dev[list(ETIQUETAS)].to_numpy(dtype=np.int64)
    n = len(train_dev)
    proba = np.zeros((n, 3), dtype=float)
    features = _solo_interacciones(train_dev["text"].astype(str).tolist())
    metricas = []
    train_f1s = []
    for numero, (idx_tr, idx_va) in enumerate(pliegues, start=1):
        train, val = matriz_v03(train_dev["text"].iloc[idx_tr], train_dev["text"].iloc[idx_va])
        train = sparse_hstack([train, features.iloc[idx_tr].to_numpy()])
        val = sparse_hstack([val, features.iloc[idx_va].to_numpy()])
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


def sparse_hstack(partes: list[Any]) -> Any:
    """Apila matrices en horizontal, dispersas o densas.

    Args:
        partes: matrices a apilar.

    Returns:
        La matriz apilada en formato CSR.
    """
    from scipy import sparse

    return sparse.hstack(partes).tocsr()


def main() -> None:
    """Ejecuta la Fase 2 de V32.

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

    log("2/4 V32-A: URL aditivo")
    a = resumen(evaluar_bloque(desarrollo, pliegues, bloque=BLOQUE, con_interacciones=False))
    log(f"   V32-A: {a['corrected_cv']:.10f} | delta {a['corrected_cv'] - control['corrected_cv']:+.10f}")

    log("3/4 V32-B: solo las cinco interacciones")
    b = resumen(evaluar_solo_interacciones(desarrollo, pliegues))
    log(f"   V32-B: {b['corrected_cv']:.10f} | delta {b['corrected_cv'] - control['corrected_cv']:+.10f}")

    log("4/4 V32-C: URL aditivo + interacciones (debe reproducir V31-04)")
    c = resumen(evaluar_bloque(desarrollo, pliegues, bloque=BLOQUE, con_interacciones=True))
    log(f"   V32-C: {c['corrected_cv']:.10f} | delta {c['corrected_cv'] - control['corrected_cv']:+.10f}")

    base = control["corrected_cv"]
    descomposicion = {
        "V03_a_URL_aditivo": a["corrected_cv"] - base,
        "URL_aditivo_a_URL_mas_interacciones": c["corrected_cv"] - a["corrected_cv"],
        "V03_a_solo_interacciones": b["corrected_cv"] - base,
        "V03_a_URL_mas_interacciones": c["corrected_cv"] - base,
        "parte_de_la_ganancia_explicada_por_interacciones": (
            (c["corrected_cv"] - a["corrected_cv"]) / (c["corrected_cv"] - base)
            if c["corrected_cv"] != base else 0.0
        ),
    }
    log("   descomposicion de la ganancia:")
    for clave, valor in descomposicion.items():
        log(f"      {clave:<48} {valor:+.10f}")

    comp_a = comparar(y, evaluar_bloque(desarrollo, pliegues, bloque=None)["pred"],
                      evaluar_bloque(desarrollo, pliegues, bloque=BLOQUE)["pred"])
    comp_b = comparar(y, evaluar_bloque(desarrollo, pliegues, bloque=None)["pred"],
                      evaluar_solo_interacciones(desarrollo, pliegues)["pred"])
    comp_c = comparar(y, evaluar_bloque(desarrollo, pliegues, bloque=None)["pred"],
                      evaluar_bloque(desarrollo, pliegues, bloque=BLOQUE, con_interacciones=True)["pred"])
    for nombre, comp in (("V32-A", comp_a), ("V32-B", comp_b), ("V32-C", comp_c)):
        log(
            f"   {nombre}: rescates {comp['rescates']} danios {comp['danios']} "
            f"neto {comp['neto']:+d} celdas {comp['celdas_modificadas']}"
        )

    informe = {
        "familia": "STRUCTURAL_TEXT_FEATURES",
        "experimento": EXPERIMENTO,
        "fecha": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "estado": "OOF_EVALUATED",
        "pregunta": "la ganancia de V31-04 viene de las interacciones o del bloque URL aditivo",
        "alcance": "tres variantes: URL aditivo, solo interacciones, y ambas. Sin combinar con otros bloques",
        "control": control,
        "V32_A_url_aditivo": a,
        "V32_B_solo_interacciones": b,
        "V32_C_url_mas_interacciones": c,
        "descomposicion": descomposicion,
        "comparacion": {"V32_A": comp_a, "V32_B": comp_b, "V32_C": comp_c},
        "gastos": {"submissions": 0, "modelos_entrenados": 4,
                   "memoria_pico_mb": round(memoria_pico_mb(), 1),
                   "segundos": round(time.time() - inicio, 1)},
    }
    REPORTE_JSON.write_text(
        json.dumps(informe, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _informe_md(informe)
    log("   informe escrito en reports/v32_02_aislar_fuente.{json,md}")
    log(f"   memoria pico {memoria_pico_mb():.0f} MB | tiempo {round(time.time() - inicio, 1)} s")
    log("VEREDICTO: Fase 2 evaluada, 0 submissions")


def _informe_md(informe: dict[str, Any]) -> None:
    """Escribe el informe legible.

    Args:
        informe: informe con los datos.

    Returns:
        None.
    """
    c = informe["control"]
    a = informe["V32_A_url_aditivo"]
    b = informe["V32_B_solo_interacciones"]
    d = informe["V32_C_url_mas_interacciones"]
    dec = informe["descomposicion"]
    g = informe["gastos"]
    lineas = [
        "# V32-02: aislar la fuente de la ganancia de V31-04",
        "",
        f"Generado por `experiments/v32_02_aislar_fuente.py` ({EXPERIMENTO}).",
        "",
        f"**Estado `{informe['estado']}`. Alcance: {informe['alcance']}**",
        "",
        "## 1. Las tres variantes",
        "",
        "| variante | corrected_cv | delta vs V03 | rescates | danios | net |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for nombre, res, comp in (
        ("V32-A URL aditivo", a, informe["comparacion"]["V32_A"]),
        ("V32-B solo interacciones", b, informe["comparacion"]["V32_B"]),
        ("V32-C URL + interacciones", d, informe["comparacion"]["V32_C"]),
    ):
        lineas.append(
            f"| {nombre} | {res['corrected_cv']:.10f} | {res['corrected_cv'] - c['corrected_cv']:+.10f} | "
            f"{comp['rescates']} | {comp['danios']} | {comp['neto']:+d} |"
        )
    lineas += [
        "",
        "## 2. La descomposicion de la ganancia",
        "",
        "| paso | delta |",
        "|---|---:|",
        f"| V03 -> URL aditivo | {dec['V03_a_URL_aditivo']:+.10f} |",
        f"| URL aditivo -> URL + interacciones | {dec['URL_aditivo_a_URL_mas_interacciones']:+.10f} |",
        f"| V03 -> solo interacciones | {dec['V03_a_solo_interacciones']:+.10f} |",
        f"| V03 -> URL + interacciones | {dec['V03_a_URL_mas_interacciones']:+.10f} |",
        "",
        (f"**El {100 * dec['parte_de_la_ganancia_explicada_por_interacciones']:.1f} % de la ganancia "
        f"se explica por las interacciones.**"),
        "",
        "## 3. Lo que este script no hace",
        "",
        "- No combina con mencion ni con enfasis.",
        "- No genera submission. No lee el test.",
        "",
        (f"Coste: {g['segundos']} s, memoria {g['memoria_pico_mb']:.0f} MB, "
        f"{g['modelos_entrenados']} modelos, {g['submissions']} submissions."),
        "",
    ]
    REPORTE_MD.write_text("\n".join(lineas) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
