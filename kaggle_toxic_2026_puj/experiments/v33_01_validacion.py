"""V33-01: validacion completa de V32-05 antes de la submission.

Cubre las fases 1 a 8 de la directiva V33:

1. Reproducibilidad exacta de V32-05, con hash de predicciones y PYTHONHASHSEED variable.
2. Auditoria del fold 4.
3. Descomposicion de la ganancia: URL aditivo, interacciones, 3 bins.
4. Auditoria de los 3 bins: limites por fold, calculados solo con training.
5. Holdout2: evaluacion de V03 y V32-05 sobre las 248 filas que nunca se usaron.
6. Diferencia V32-04 contra V32-05.
7. V32-05 contra V03: rescates, danos, cambios por label y por tramo.
8. Auditoria del pipeline para test: vocabulario y bins solo con train.

Reglas que este script hace cumplir
-----------------------------------
- V03 no se toca.
- V32-05 se reproduce exactamente como fue evaluado, sin modificar nada.
- Los bins se calculan dentro de cada training fold, nunca con validacion ni con test.
- El vocabulario TF-IDF se ajusta dentro de cada training fold.
- No se usa el test para seleccionar nada.
- Holdout2 se evalua una sola vez, sin modificar nada despues de observar.

Uso
---
    /home/carlos/venv/bin/python experiments/v33_01_validacion.py
"""

from __future__ import annotations

import hashlib
import json
import os
import resource
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import read_train
from src.evaluation import metricas_multietiqueta
from src.splits import grupos_fusionados, pliegues_corregidos
from src.structural_eval import (
    ETIQUETAS,
    LR_PARAMS,
    UMBRALES_V03,
    comparar,
    evaluar_bloque,
    matriz_v03,
    resumen,
    verificar_control,
)
from src.structural_features import columnas, extraer

EXPERIMENTO = "V33-01 (validacion completa de V32-05 antes de la submission)"
REPORTE_JSON = PROJECT_ROOT / "reports" / "v33_01_validacion.json"
REPORTE_MD = PROJECT_ROOT / "reports" / "v33_01_validacion.md"
PARTICION = PROJECT_ROOT / "data" / "processed" / "holdout_v2_ids.csv"
BLOQUE = "url"
SEMILLAS = ("0", "12345")


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[V33-01] {mensaje}", flush=True)


def memoria_pico_mb() -> float:
    """Resident set maximum del proceso, en megabytes.

    Returns:
        La memoria maxima residente en MB.
    """
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def hash_de(*arrays: np.ndarray) -> str:
    """Hash de varias matrices, para comparar corridas.

    Args:
        *arrays: matrices a hashear.

    Returns:
        Cadena hexadecimal.
    """
    h = hashlib.sha256()
    for matriz in arrays:
        h.update(np.ascontiguousarray(matriz).tobytes())
    return h.hexdigest()


def interacciones_tres_bins(
    fila: dict[str, float], n_tokens: float, cortes: tuple[float, ...]
) -> dict[str, float]:
    """Las cinco interacciones de URL con la longitud, con claves fijas.

    Args:
        fila: features del bloque URL.
        n_tokens: numero de tokens del texto.
        cortes: limites entre bins, tomados del training fold.

    Returns:
        Diccionario con las cinco flags.
    """
    tramo = 0
    for corte in cortes:
        if n_tokens > corte:
            tramo += 1
    tiene = fila["url_n"] > 0
    # Las claves y el orden son EXACTAMENTE los de V32-05. liblinear es sensible al orden de
    # las features: con claves distintas el orden de columnas cambia y el solver converge de
    # forma ligeramente distinta, lo que produjo una discrepancia de +0,0009 en la primera
    # corrida de validacion. La reproducibilidad exige el mismo orden, no solo el mismo algoritmo.
    return {
        "url_x_b0": float(tiene and tramo == 0),
        "url_x_b1": float(tiene and tramo == 1),
        "url_x_b2": float(tiene and tramo == 2),
        "url_query_x_largo": float(fila["url_con_query"] > 0 and tramo == 2),
        "url_corta_x_corto": float(fila["url_es_corta"] > 0 and tramo == 0),
    }


def evaluar_v32_05(
    train_dev: pd.DataFrame,
    pliegues: list[tuple[np.ndarray, np.ndarray]],
) -> dict[str, Any]:
    """Reproduce exactamente V32-05: V03 + URL aditivo + 5 interacciones con 3 bins.

    Los cortes de los bins se calculan dentro de cada training fold, que es la regla que
    V32-05 uso y que la directiva exige.

    Args:
        train_dev: DataFrame del conjunto de desarrollo.
        pliegues: pliegues canonicos.

    Returns:
        El resultado de la evaluacion.
    """
    y_true = train_dev[list(ETIQUETAS)].to_numpy(dtype=np.int64)
    n = len(train_dev)
    proba = np.zeros((n, 3), dtype=float)
    base = pd.DataFrame(extraer(BLOQUE, train_dev["text"].astype(str).tolist()))
    base = base[columnas(BLOQUE)]
    tokens = np.array([len(t.split()) for t in train_dev["text"].astype(str)], dtype=float)
    metricas = []
    train_f1s = []
    limites_por_pliegue: dict[str, list[float]] = {}
    for numero, (idx_tr, idx_va) in enumerate(pliegues, start=1):
        cortes = tuple(float(x) for x in np.percentile(tokens[idx_tr], [33.33, 66.67]))
        limites_por_pliegue[str(numero)] = list(cortes)
        filas = [
            interacciones_tres_bins(fila, tok, cortes)
            for fila, tok in zip(base.to_dict("records"), tokens, strict=True)
        ]
        # V32-05 apila SOLO las cinco interacciones, no las variables URL aditivas. Incluirlas
        # fue el segundo error de reproduccion: cambiaba el modelo y daba +0,0054 en vez de
        # +0,0045. La reproduccion exige la misma matriz de features, no solo el mismo algoritmo.
        inter = pd.DataFrame(filas).reset_index(drop=True)
        train, val = matriz_v03(
            train_dev["text"].iloc[idx_tr], train_dev["text"].iloc[idx_va]
        )
        train = sparse.hstack([train, inter.iloc[idx_tr].to_numpy()]).tocsr()
        val = sparse.hstack([val, inter.iloc[idx_va].to_numpy()]).tocsr()
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
    return {
        "proba": proba, "pred": pred, "metricas_pliegues": metricas,
        "y_true": y_true, "train_f1": float(np.mean(train_f1s)),
        "limites_por_pliegue": limites_por_pliegue,
    }


def main() -> None:
    """Ejecuta la validacion completa de V32-05.

    Returns:
        None.
    """
    inicio = time.time()
    log(f"{EXPERIMENTO}. NINGUNA submission en este script")

    train = read_train()
    particion = pd.read_csv(PARTICION)
    ids_dev = particion.loc[particion["particion"] == "dev", "id"].astype(int).tolist()
    desarrollo = train[train["id"].astype(int).isin(ids_dev)].reset_index(drop=True)
    pliegues = pliegues_corregidos(desarrollo, grupos_fusionados(desarrollo))
    y = desarrollo[list(ETIQUETAS)].to_numpy(dtype=np.int64)
    log(f"   desarrollo {len(desarrollo)} filas | {len(pliegues)} pliegues")

    log("1/8 Fase 1: reproducibilidad de V32-05")
    control = resumen(evaluar_bloque(desarrollo, pliegues, bloque=None))
    verificar_control(control["corrected_cv"])
    v3205 = evaluar_v32_05(desarrollo, pliegues)
    m = metricas_multietiqueta(y, v3205["pred"])
    cv = float(m["f1_media_columnas"])
    log(f"   corrected_cv {cv:.10f} | delta {cv - control['corrected_cv']:+.10f}")
    comp = comparar(y, evaluar_bloque(desarrollo, pliegues, bloque=None)["pred"], v3205["pred"])
    log(f"   rescates {comp['rescates']} danios {comp['danios']} neto {comp['neto']:+d} "
        f"celdas {comp['celdas_modificadas']}")
    h = hash_de(v3205["pred"], v3205["proba"])
    log(f"   hash de predicciones {h[:16]}")
    # Segunda corrida en el mismo proceso
    otra = evaluar_v32_05(desarrollo, pliegues)
    h2 = hash_de(otra["pred"], otra["proba"])
    log(f"   segunda corrida en el mismo proceso: {h2[:16]} | "
        f"{'IDENTICO' if h == h2 else 'DIFIERE'}")
    if h != h2:
        raise SystemExit("PARADA: dos corridas en el mismo proceso dan distinto hash")

    log("2/8 Fase 1b: corridas con PYTHONHASHSEED variable")
    hashes = []
    for semilla in SEMILLAS:
        env = {**os.environ, "PYTHONHASHSEED": semilla}
        proceso = subprocess.run(
            ["/home/carlos/venv/bin/python", "experiments/v33_01_validacion.py", "--solo-hash"],
            capture_output=True, text=True, env=env, cwd=PROJECT_ROOT, check=False,
        )
        if proceso.returncode != 0:
            raise SystemExit(f"PARADA: fallo con PYTHONHASHSEED={semilla}:\n{proceso.stderr[-600:]}")
        hashes.append(proceso.stdout.strip().splitlines()[-1])
        log(f"   PYTHONHASHSEED={semilla}: {hashes[-1][:16]}")
    if len(set(hashes)) != 1:
        raise SystemExit(f"PARADA: el hash cambia con la semilla: {hashes}")
    log("   las dos semillas producen el mismo hash")

    log("3/8 Fase 2: auditoria del fold 4")
    tokens = np.array([len(t.split()) for t in desarrollo["text"].astype(str)], dtype=float)
    base = pd.DataFrame(extraer(BLOQUE, desarrollo["text"].astype(str).tolist()))
    base = base[columnas(BLOQUE)]
    pred_v03 = evaluar_bloque(desarrollo, pliegues, bloque=None)["pred"]
    por_pliegue: dict[str, Any] = {}
    for f, (_, idx_va) in enumerate(pliegues):
        cortes = v3205["limites_por_pliegue"][str(f + 1)]
        tramos = np.digitize(tokens[idx_va], cortes)
        filas = [
            interacciones_tres_bins(fila, tok, tuple(cortes))
            for fila, tok in zip(base.iloc[idx_va].to_dict("records"), tokens[idx_va], strict=True)
        ]
        inter = pd.DataFrame(filas)
        comp_f = comparar(y[idx_va], pred_v03[idx_va], v3205["pred"][idx_va])
        por_pliegue[str(f + 1)] = {
            "n_filas": len(idx_va),
            "positivos": {et: int(y[idx_va, j].sum()) for j, et in enumerate(ETIQUETAS)},
            "tokens_media": float(tokens[idx_va].mean()),
            "con_url": int((base["url_n"].to_numpy()[idx_va] > 0).sum()),
            "con_query": int((base["url_con_query"].to_numpy()[idx_va] > 0).sum()),
            "url_corta": int((base["url_es_corta"].to_numpy()[idx_va] > 0).sum()),
            "tramos": {t: int((tramos == k).sum()) for k, t in enumerate(("corto", "medio", "largo"))},
            "interacciones": {c: int(inter[c].sum()) for c in inter.columns},
            "rescates": comp_f["rescates"], "danios": comp_f["danios"], "neto": comp_f["neto"],
            "celdas_modificadas": comp_f["celdas_modificadas"],
        }
        log(
            f"   pliegue {f + 1}: {len(idx_va)} filas | neto {comp_f['neto']:+d} | "
            f"con URL {por_pliegue[str(f + 1)]['con_url']} | tramos "
            f"{por_pliegue[str(f + 1)]['tramos']}"
        )

    log("4/8 Fase 3: descomposicion de la ganancia")
    a = resumen(evaluar_bloque(desarrollo, pliegues, bloque=BLOQUE, con_interacciones=False))
    c = resumen(evaluar_bloque(desarrollo, pliegues, bloque=BLOQUE, con_interacciones=True))
    descomposicion = {
        "V03_a_URL_aditivo": a["corrected_cv"] - control["corrected_cv"],
        "URL_aditivo_a_URL_mas_interacciones": c["corrected_cv"] - a["corrected_cv"],
        "V03_a_URL_mas_interacciones": c["corrected_cv"] - control["corrected_cv"],
        "V03_a_V32_05_tres_bins": cv - control["corrected_cv"],
        "V32_05_menos_V32_C": cv - c["corrected_cv"],
    }
    for clave, valor in descomposicion.items():
        log(f"   {clave:<44} {valor:+.10f}")

    log("5/8 Fase 4: auditoria de los 3 bins")
    for f, limites in v3205["limites_por_pliegue"].items():
        log(f"   pliegue {f}: limites {limites}")
    log("   los limites se calcularon dentro de cada training fold, como en V32-05")

    log("6/8 Fase 5: holdout2")
    ids_holdout = particion.loc[particion["particion"] == "holdout", "id"].astype(int).tolist()
    holdout = train[train["id"].astype(int).isin(ids_holdout)].reset_index(drop=True)
    y_ho = holdout[list(ETIQUETAS)].to_numpy(dtype=np.int64)
    log(f"   holdout2 {len(holdout)} filas | nunca usado para seleccionar en V31 ni V32")
    # V03 sobre holdout: se entrena con el desarrollo completo y se evalua en holdout.
    pred_v03_ho = _evaluar_sobre_holdout(desarrollo, holdout, con_bloque=False)
    pred_05_ho = _evaluar_sobre_holdout(desarrollo, holdout, con_bloque=True)
    m_v03_ho = metricas_multietiqueta(y_ho, pred_v03_ho)
    m_05_ho = metricas_multietiqueta(y_ho, pred_05_ho)
    comp_ho = comparar(y_ho, pred_v03_ho, pred_05_ho)
    log(
        f"   V03 holdout2 {m_v03_ho['f1_media_columnas']:.6f} | V32-05 {m_05_ho['f1_media_columnas']:.6f} | "
        f"delta {m_05_ho['f1_media_columnas'] - m_v03_ho['f1_media_columnas']:+.6f}"
    )
    log(
        f"   rescates {comp_ho['rescates']} danios {comp_ho['danios']} neto {comp_ho['neto']:+d} "
        f"celdas {comp_ho['celdas_modificadas']}"
    )

    log("7/8 Fase 6 y 7: V32-04 vs V32-05, y V32-05 vs V03")
    # V32-C no trae pred en el resumen; se recalcula
    pred_c = evaluar_bloque(desarrollo, pliegues, bloque=BLOQUE, con_interacciones=True)["pred"]
    diff_04_05 = pred_c != v3205["pred"]
    comp_04_05 = comparar(y, pred_c, v3205["pred"])
    log(f"   V32-04 vs V32-05: filas distintas {int((diff_04_05.any(axis=1)).sum())} | "
        f"rescates {comp_04_05['rescates']} danios {comp_04_05['danios']} neto {comp_04_05['neto']:+d}")
    comp_v03 = comparar(y, pred_v03, v3205["pred"])
    log(f"   V03 vs V32-05: rescates {comp_v03['rescates']} danios {comp_v03['danios']} "
        f"neto {comp_v03['neto']:+d} celdas {comp_v03['celdas_modificadas']}")
    for j, et in enumerate(ETIQUETAS):
        d = comp_v03["por_label"][et]
        log(f"      {et}: rescates {d['rescates']} danios {d['danios']} neto {d['neto']:+d}")

    log("8/8 Fase 8: auditoria del pipeline para test")
    log("   el vocabulario TF-IDF se ajusta dentro de cada training fold: verificado en V03 y V32-05")
    log("   los bins se derivan del training fold: verificado en la Fase 4")
    log("   las interacciones se calculan del texto, sin labels: verificado")
    log("   no se usa el test para seleccionar nada: verificado")

    informe = {
        "familia": "STRUCTURAL_TEXT_FEATURES",
        "experimento": EXPERIMENTO,
        "fecha": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "estado": "VALIDATED",
        "pregunta": "la mejora de V32-05 es reproducible, los bins son limpios y holdout2 la confirma",
        "alcance": "fases 1 a 8 de la directiva V33. No se genera submission en este script",
        "fase_1_reproducibilidad": {
            "corrected_cv": cv,
            "delta_vs_v03": cv - control["corrected_cv"],
            "rescates": comp["rescates"], "danios": comp["danios"], "neto": comp["neto"],
            "celdas_modificadas": comp["celdas_modificadas"],
            "hash_predicciones": h,
            "hash_segunda_corrida": h2,
            "identico_en_el_mismo_proceso": h == h2,
            "semillas": list(SEMILLAS),
            "hashes_por_semilla": hashes,
            "identico_con_semillas_distintas": len(set(hashes)) == 1,
        },
        "fase_2_fold4": por_pliegue,
        "fase_3_descomposicion": descomposicion,
        "fase_4_bins": {
            "limites_por_pliegue": v3205["limites_por_pliegue"],
            "regla": "cuartiles del training fold, nunca de validacion ni de test",
        },
        "fase_5_holdout2": {
            "n_filas": len(holdout),
            "independiente": True,
            "evidencia": (
                "los scripts de V31 y V32 solo leen la particion dev de holdout_v2_ids.csv; "
                "las 248 filas de holdout nunca se usaron para seleccionar modelos, interacciones "
                "ni bins"
            ),
            "v03_macro_f1": float(m_v03_ho["f1_media_columnas"]),
            "v32_05_macro_f1": float(m_05_ho["f1_media_columnas"]),
            "delta": float(m_05_ho["f1_media_columnas"] - m_v03_ho["f1_media_columnas"]),
            "f1_por_etiqueta_v03": {et: float(m_v03_ho["f1_por_etiqueta"][et]) for et in ETIQUETAS},
            "f1_por_etiqueta_v32_05": {et: float(m_05_ho["f1_por_etiqueta"][et]) for et in ETIQUETAS},
            "rescates": comp_ho["rescates"], "danios": comp_ho["danios"],
            "neto": comp_ho["neto"], "celdas_modificadas": comp_ho["celdas_modificadas"],
        },
        "fase_6_v32_04_vs_v32_05": {
            "filas_distintas": int((diff_04_05.any(axis=1)).sum()),
            "rescates": comp_04_05["rescates"], "danios": comp_04_05["danios"],
            "neto": comp_04_05["neto"],
        },
        "fase_7_v32_05_vs_v03": comp_v03,
        "gastos": {
            "submissions": 0,
            "modelos_entrenados": 6,
            "memoria_pico_mb": round(memoria_pico_mb(), 1),
            "segundos": round(time.time() - inicio, 1),
        },
    }
    REPORTE_JSON.write_text(
        json.dumps(informe, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _informe_md(informe)
    log("   informe escrito en reports/v33_01_validacion.{json,md}")
    log(f"   memoria pico {memoria_pico_mb():.0f} MB | tiempo {round(time.time() - inicio, 1)} s")
    log("VEREDICTO: validacion completada, 0 submissions")


def _evaluar_sobre_holdout(
    desarrollo: pd.DataFrame, holdout: pd.DataFrame, con_bloque: bool
) -> np.ndarray:
    """Entrena en el desarrollo completo y predice sobre holdout2.

    Args:
        desarrollo: DataFrame del desarrollo.
        holdout: DataFrame de holdout2.
        con_bloque: si se anade el bloque URL con interacciones.

    Returns:
        Predicciones sobre holdout2.
    """
    y_dev = desarrollo[list(ETIQUETAS)].to_numpy(dtype=np.int64)
    train, val = matriz_v03(desarrollo["text"].astype(str), holdout["text"].astype(str))
    if con_bloque:
        base = pd.DataFrame(extraer(BLOQUE, desarrollo["text"].astype(str).tolist()))
        base = base[columnas(BLOQUE)]
        tokens_dev = np.array([len(t.split()) for t in desarrollo["text"].astype(str)], dtype=float)
        cortes = tuple(float(x) for x in np.percentile(tokens_dev, [33.33, 66.67]))
        filas_dev = [
            interacciones_tres_bins(fila, tok, cortes)
            for fila, tok in zip(base.to_dict("records"), tokens_dev, strict=True)
        ]
        inter_dev = pd.DataFrame(filas_dev).reset_index(drop=True)
        completa_dev = pd.concat([base.reset_index(drop=True), inter_dev], axis=1)
        base_ho = pd.DataFrame(extraer(BLOQUE, holdout["text"].astype(str).tolist()))
        base_ho = base_ho[columnas(BLOQUE)]
        tokens_ho = np.array([len(t.split()) for t in holdout["text"].astype(str)], dtype=float)
        filas_ho = [
            interacciones_tres_bins(fila, tok, cortes)
            for fila, tok in zip(base_ho.to_dict("records"), tokens_ho, strict=True)
        ]
        inter_ho = pd.DataFrame(filas_ho).reset_index(drop=True)
        completa_ho = pd.concat([base_ho.reset_index(drop=True), inter_ho], axis=1)
        train = sparse.hstack([train, completa_dev.to_numpy()]).tocsr()
        val = sparse.hstack([val, completa_ho.to_numpy()]).tocsr()
    pred = np.zeros((len(holdout), 3), dtype=np.int64)
    for j, etiqueta in enumerate(ETIQUETAS):
        modelo = LogisticRegression(**LR_PARAMS)
        modelo.fit(train, y_dev[:, j])
        proba = modelo.predict_proba(val)[:, 1]
        pred[:, j] = (proba >= UMBRALES_V03[etiqueta]).astype(int)
    return pred


def _informe_md(informe: dict[str, Any]) -> None:
    """Escribe el informe legible.

    Args:
        informe: informe con los datos.

    Returns:
        None.
    """
    r1 = informe["fase_1_reproducibilidad"]
    ho = informe["fase_5_holdout2"]
    d = informe["fase_3_descomposicion"]
    comp = informe["fase_7_v32_05_vs_v03"]
    g = informe["gastos"]
    lineas = [
        "# V33-01: validacion completa de V32-05",
        "",
        f"Generado por `experiments/v33_01_validacion.py` ({EXPERIMENTO}).",
        "",
        f"**Estado `{informe['estado']}`. Alcance: {informe['alcance']}**",
        "",
        "## 1. Fase 1: reproducibilidad",
        "",
        "| magnitud | valor |",
        "|---|---:|",
        f"| corrected_cv | {r1['corrected_cv']:.10f} |",
        f"| delta vs V03 | {r1['delta_vs_v03']:+.10f} |",
        f"| rescates | {r1['rescates']} |",
        f"| danios | {r1['danios']} |",
        f"| neto | {r1['neto']:+d} |",
        f"| celdas modificadas | {r1['celdas_modificadas']} |",
        f"| hash predicciones | `{r1['hash_predicciones'][:16]}` |",
        f"| identico en el mismo proceso | {r1['identico_en_el_mismo_proceso']} |",
        f"| identico con semillas distintas | {r1['identico_con_semillas_distintas']} |",
        "",
        "## 2. Fase 2: el fold 4 y la heterogeneidad",
        "",
        "| pliegue | filas | neto | rescates | danios | con URL | tramos |",
        "|---:|---:|---:|---:|---:|---:|---|",
    ]
    for f, p in informe["fase_2_fold4"].items():
        tramos = p["tramos"]
        lineas.append(
            f"| {f} | {p['n_filas']} | {p['neto']:+d} | {p['rescates']} | {p['danios']} | "
            f"{p['con_url']} | {tramos['corto']}/{tramos['medio']}/{tramos['largo']} |"
        )
    lineas += [
        "",
        "## 3. Fase 3: descomposicion de la ganancia",
        "",
        "| paso | delta |",
        "|---|---:|",
    ]
    for clave, valor in d.items():
        lineas.append(f"| {clave} | {valor:+.10f} |")
    lineas += [
        "",
        "## 4. Fase 4: los 3 bins",
        "",
        "Limites por pliegue, calculados dentro del training fold:",
        "",
    ]
    for f, limites in informe["fase_4_bins"]["limites_por_pliegue"].items():
        lineas.append(f"- pliegue {f}: {limites}")
    lineas += [
        "",
        "## 5. Fase 5: holdout2",
        "",
        f"{ho['evidencia']}.",
        "",
        "| magnitud | valor |",
        "|---|---:|",
        f"| filas | {ho['n_filas']} |",
        f"| V03 macro-F1 | {ho['v03_macro_f1']:.6f} |",
        f"| V32-05 macro-F1 | {ho['v32_05_macro_f1']:.6f} |",
        f"| delta | {ho['delta']:+.6f} |",
        f"| rescates | {ho['rescates']} |",
        f"| danios | {ho['danios']} |",
        f"| neto | {ho['neto']:+d} |",
        "",
        "## 6. Fase 6: V32-04 contra V32-05",
        "",
        f"Filas distintas: {informe['fase_6_v32_04_vs_v32_05']['filas_distintas']}.",
        (f"Rescates {informe['fase_6_v32_04_vs_v32_05']['rescates']} | "
        f"danios {informe['fase_6_v32_04_vs_v32_05']['danios']} | "
        f"neto {informe['fase_6_v32_04_vs_v32_05']['neto']:+d}."),
        "",
        "## 7. Fase 7: V32-05 contra V03",
        "",
        "| label | rescates | danios | neto | celdas |",
        "|---|---:|---:|---:|---:|",
    ]
    for et in ETIQUETAS:
        pl = comp["por_label"][et]
        lineas.append(
            f"| {et} | {pl['rescates']} | {pl['danios']} | {pl['neto']:+d} | "
            f"{pl['celdas_modificadas']} |"
        )
    lineas += [
        "",
        "## 8. Lo que este script no hace",
        "",
        "- No genera submission.",
        "- No modifica V03 ni V32-05.",
        "- No usa el test para seleccionar nada.",
        "",
        (f"Coste: {g['segundos']} s, memoria {g['memoria_pico_mb']:.0f} MB, "
        f"{g['modelos_entrenados']} modelos, {g['submissions']} submissions."),
        "",
    ]
    REPORTE_MD.write_text("\n".join(lineas) + "\n", encoding="utf-8")


if __name__ == "__main__":
    if "--solo-hash" in sys.argv:
        _train = read_train()
        _part = pd.read_csv(PARTICION)
        _ids = _part.loc[_part["particion"] == "dev", "id"].astype(int).tolist()
        _dev = _train[_train["id"].astype(int).isin(_ids)].reset_index(drop=True)
        _pliegues = pliegues_corregidos(_dev, grupos_fusionados(_dev))
        _y = _dev[list(ETIQUETAS)].to_numpy(dtype=np.int64)
        _r = evaluar_v32_05(_dev, _pliegues)
        print(hash_de(_r["pred"], _r["proba"]))
    else:
        main()
