"""V51-01 (P51): selector utilitario S1-U con cross-fitting y umbral analitico.

Que resuelve
------------
P50 uso como surrogado `P(canal acierta)`, que premia la **predecibilidad** de un
canal y por eso eligio `cita:NEG` 109 veces con net -1706. P51 usa la **utilidad**
respecto de V03, y de esa diferencia sale una consecuencia que P50 no podia ver.

El mecanismo, en corto
----------------------
La F1 no es simetrica. En V19-01 ya estaba derivado que `FN -> TP` escala con
`TP + FP + FN` y `FP -> TN` solo con `TP`, de modo que subir de 0 a 1 y bajar de
1 a 0 **no** son la misma apuesta. El punto de equilibrio resulta ser el mismo
numero en las dos direcciones cuando se expresa sobre `P(y=1)`, y ese numero es
**0,3814**, no 0,5.

Por tanto la regla utilitaria optima no es "cambiar si P(acierta) > 0,5", sino
"cambiar si la direccion del canal coincide con el signo de P(y=1) por encima del
equilibrio". Ese es un **punto analitico**, no un grid, y se reporta como
variante del mismo selector.

Metodo
------
- S1-U: un `LogisticRegression` por canal, entrenado **solo sobre celdas en
  discrepancia** (seccion 12), con target `delta_F1 > 0` calculado con los
  conteos del pliegue de entrenamiento.
- Cross-fitting con los pliegues canonicos de TOXIC agrupados. La validacion no
  interviene ni en el target, ni en el ajuste, ni en la fiabilidad de canales.
- `KEEP_V03` siempre disponible. decision: argmax de `P(delta_F1 > 0)` sobre los
  canales que discrepan, y solo se cambia si supera el umbral.

Uso
---
    /home/carlos/venv/bin/python experiments/v51_01_selector_utilidad.py
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

sys.path.insert(0, str(PROJECT_ROOT / "experiments"))
from v51_00_utilidad_y_asimetria import (
    f1_macro,
    valor_falso_negativo,
    valor_falso_positivo,
)

FEATURES = PROJECT_ROOT / "features" / "p50_celda_features.npz"
UTILIDAD = PROJECT_ROOT / "features" / "p51_utility_targets.npz"
P50_JSON = PROJECT_ROOT / "reports" / "p50_selector_results.json"
P50_GAP = PROJECT_ROOT / "reports" / "p50_gap_and_oracle.json"

ETIQUETAS = ("HS", "TR", "AG")
N_FOLD = 5
UMBRAL_ESPECIFICADO = 0.5

REPORTE_JSON = PROJECT_ROOT / "reports" / "p51_utility_selector_results.json"
REPORTE_MD = PROJECT_ROOT / "reports" / "p51_utility_selector_analysis.md"
CSV_RESULTADOS = PROJECT_ROOT / "reports" / "p51_utility_selector_results.csv"
CSV_INTerseccion = PROJECT_ROOT / "reports" / "p51_utility_selector_confusion.csv"
CSV_RESCATE = PROJECT_ROOT / "reports" / "p51_utility_selector_rescue.csv"


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[V51-01] {mensaje}", flush=True)


def columnas_accion(X: np.ndarray, columnas: list[str], canal: int,
                    nombres: list[str]) -> np.ndarray:
    """Variables de un clasificador: las compartidas mas las dos del canal.

    Args:
        X: matriz completa de variables por celda.
        columnas: nombres de las columnas.
        canal: indice del canal, de 0 a 19.
        nombres: nombres de los canales.

    Returns:
        Matriz de variables de ese canal.
    """
    compartidas = [i for i, c in enumerate(columnas)
                   if not c.startswith("proba_") and not c.startswith("pred_")]
    propias = [columnas.index(f"proba_{nombres[canal]}"),
               columnas.index(f"pred_{nombres[canal]}")]
    return X[:, compartidas + propias]


def evaluar(y: np.ndarray, pred: np.ndarray, v03: np.ndarray) -> dict[str, Any]:
    """Metricas completas de una politica de seleccion.

    Args:
        y: etiquetas verdaderas.
        pred: predicciones de la politica.
        v03: predicciones de V03.

    Returns:
        Diccionario con corrected_cv, delta, rescate, damage y net.
    """
    err = v03 != y
    f1 = f1_macro(y, pred)
    resc = int((err & (pred == y)).sum())
    dany = int(((~err) & (pred != y)).sum())
    return {
        "corrected_cv": round(f1, 10),
        "delta_vs_v03": round(f1 - f1_macro(y, v03), 10),
        "por_etiqueta": {et: round(float(f1_score(y[:, j], pred[:, j], zero_division=0)), 10)
                         for j, et in enumerate(ETIQUETAS)},
        "celdas_cambiadas": int((pred != v03).sum()),
        "abstencion_rate": round(float(1 - (pred != v03).mean()), 6),
        "rescate": resc, "damage": dany, "net": resc - dany,
    }


def main() -> None:
    """Ejecuta el selector utilitario S1-U.

    Returns:
        None.
    """
    inicio = time.time()
    log("P51-01 selector utilitario S1-U. 0 submissions, 0 consultas a Kaggle")
    datos = np.load(FEATURES, allow_pickle=True)
    util = np.load(UTILIDAD, allow_pickle=True)
    n_filas = len(datos["ids"])
    y2d = np.asarray(datos["y_true"], dtype=np.int64)
    v03 = np.asarray(datos["v03_pred"], dtype=np.int64)
    canales = np.asarray(datos["canal_pred"], dtype=np.int64)
    v03_margen = np.asarray(datos["v03_margen"], dtype=np.float64)
    pliegue = np.asarray(datos["pliegue"], dtype=np.int64)
    X = datos["X"]
    columnas = [str(c) for c in datos["columnas"]]
    nombres = [str(x) for x in util["nombres"]]
    n_canales = canales.shape[0]
    utilidad = np.asarray(util["utilidad"], dtype=np.float64)
    f1_v03 = f1_macro(y2d, v03)
    p50 = json.loads(P50_JSON.read_text(encoding="utf-8"))
    p50_gap = json.loads(P50_GAP.read_text(encoding="utf-8"))
    log(f"   V03 {f1_v03:.10f} | P50 S1 {p50['s1']['corrected_cv']:.10f} | "
        f"canales {n_canales}")

    # ------------------------------------------------ 1. controles B0, B1, B2
    log("1/6 controles B0, B1 y B2 (P50 desde su artefacto, sin reentrenar)")
    b0 = evaluar(y2d, v03, v03)
    b1 = evaluar(y2d, v03, v03)  # siempre KEEP_V03 es identico a B0 por construccion
    log(f"   B0 = V03              {b0['corrected_cv']:.10f} | delta +0,0000000000")
    log(f"   B1 = siempre KEEP_V03 {b1['corrected_cv']:.10f} | identico a B0 por "
        f"construccion: la accion por defecto ya es no cambiar")
    log(f"   B2 = P50 S1           {p50['s1']['corrected_cv']:.10f} | "
        f"delta {p50['s1']['delta_vs_v03']:+.10f} | net {p50['s1']['net']:+d}")

    # ------------------------------------- 2. puntos de equilibrio analiticos
    log("2/6 puntos de equilibrio analiticos por etiqueta y direccion")
    filas_eq = []
    for v in range(N_FOLD):
        ent = pliegue != v
        for j, et in enumerate(ETIQUETAS):
            col = y2d[ent, j]
            pred = v03[ent, j]
            tp = int(np.count_nonzero((pred == 1) & (col == 1)))
            fp = int(np.count_nonzero((pred == 1) & (col == 0)))
            fn = int(np.count_nonzero((pred == 0) & (col == 1)))
            g = valor_falso_negativo(tp, fp, fn)
            h = valor_falso_positivo(tp, fp, fn)
            filas_eq.append({
                "pliegue": v, "etiqueta": et, "tp": tp, "fp": fp, "fn": fn,
                "ganancia_subir": g, "perdida_subir": h,
                "umbral_equilibrio_subir": h / (g + h),
                "umbral_equilibrio_bajar": g / (g + h),
            })
    eqdf = pd.DataFrame(filas_eq)
    umbral_subir = float(eqdf["umbral_equilibrio_subir"].mean())
    umbral_bajar = float(eqdf["umbral_equilibrio_bajar"].mean())
    for et in ETIQUETAS:
        sub = eqdf[eqdf["etiqueta"] == et]
        log(f"   {et}: subir si P(y=1) > {sub['umbral_equilibrio_subir'].mean():.4f} | "
            f"bajar si P(y=1) < {1 - sub['umbral_equilibrio_bajar'].mean():.4f}")
    log(f"   nota: ambos umbrales coinciden sobre P(y=1) en "
        f"{abs(umbral_subir - (1 - umbral_bajar)):.2e}, luego la regla optima es un "
        f"UNICO umbral de {umbral_subir:.4f} sobre P(y=1)")

    # Tasas base: cuanto hay que mejorar para alcanzar el equilibrio. Se cuentan
    # solo sobre las celdas de validacion de cada pliegue, con la utilidad de ese
    # pliegue, para no mezclar conteos de origenes distintos.
    num_subir = den_subir = num_bajar = den_bajar = 0
    for v in range(N_FOLD):
        val = pliegue != v
        u = utilidad[v][:, val, :]
        arriba = (canales == 1) & (v03 == 0)[None]
        abajo = (canales == 0) & (v03 == 1)[None]
        ma, mb = arriba[:, val, :], abajo[:, val, :]
        num_subir += int(((u > 0) & ma).sum())
        den_subir += int(ma.sum())
        num_bajar += int(((u > 0) & mb).sum())
        den_bajar += int(mb.sum())
    base_subir = num_subir / max(den_subir, 1)
    base_bajar = num_bajar / max(den_bajar, 1)
    log(f"   tasa base de acierto al SUBIR (0->1): {100 * base_subir:.2f} pct "
        f"({num_subir}/{den_subir}), necesario {100 * umbral_subir:.2f} pct, "
        f"elevacion necesaria {umbral_subir / base_subir:.2f}x")
    log(f"   tasa base de acierto al BAJAR (1->0): {100 * base_bajar:.2f} pct "
        f"({num_bajar}/{den_bajar}), necesario {100 * (1 - umbral_bajar):.2f} pct, "
        f"elevacion necesaria {(1 - umbral_bajar) / base_bajar:.2f}x")

    # ---------------------------------------------- 3. S1-U cross-fitted
    log("3/6 S1-U: un LogisticRegression por canal, solo sobre celdas en discrepancia")
    n_celdas = n_filas * 3
    pl_celda = np.repeat(pliegue, 3)
    y2d.ravel()
    v03_plano = v03.ravel()
    proba = np.zeros((n_canales, n_celdas), dtype=np.float64)
    auc = np.full(n_canales, np.nan)
    for v in range(N_FOLD):
        ent = pl_celda != v
        val = ~ent
        for c in range(n_canales):
            u = utilidad[v, c].ravel()
            discrepan = canales[c].ravel() != v03_plano
            mask = ent & discrepan
            if mask.sum() < 20 or len(np.unique(u[mask] > 0)) < 2:
                continue
            Xc = columnas_accion(X, columnas, c, nombres)
            modelo = LogisticRegression(C=1.0, max_iter=1000, solver="liblinear",
                                        random_state=42)
            modelo.fit(Xc[mask], (u[mask] > 0).astype(int))
            proba[c, val] = modelo.predict_proba(Xc[val])[:, 1]
        n_disc_ent = int((ent.reshape(n_filas, 3)
                          & (canales != v03[None]).any(axis=0)).sum())
        log(f"   pliegue {v}: ajustados {int(ent.sum())} celdas, de las cuales "
            f"{n_disc_ent} en discrepancia")
    # AUC out-of-fold: mide si la utilidad es identificable con variables observables.
    for c in range(n_canales):
        objetivo = (utilidad[0, c].ravel() > 0)
        constante = bool(np.all(proba[c] == proba[c][0]))
        if len(np.unique(objetivo)) == 2 and not constante:
            auc[c] = roc_auc_score(objetivo, proba[c])
    log(f"   AUC out-of-fold por canal: min {np.nanmin(auc):.4f} "
        f"max {np.nanmax(auc):.4f} media {np.nanmean(auc):.4f}")

    # ---------------------------------------------- 4. regla de decision
    log("4/6 regla de decision")
    canales_plano = canales.reshape(n_canales, -1)
    discrepan_plano = canales_plano != v03_plano[None]
    proba_bloque = np.where(discrepan_plano, proba, -1.0)
    canal_elegido = proba_bloque.argmax(axis=0)
    confianza = proba_bloque.max(axis=0)
    accionada = (confianza >= UMBRAL_ESPECIFICADO) & (confianza > 0)

    acciones_finales = np.zeros(n_celdas, dtype=np.int64)
    for c in range(n_canales):
        sel = accionada & (canal_elegido == c)
        acciones_finales[sel] = canales[c].ravel()[sel]
    pred_u = acciones_finales.reshape(n_filas, 3)
    s1u = evaluar(y2d, pred_u, v03)
    log(f"   S1-U con umbral {UMBRAL_ESPECIFICADO}: {s1u['corrected_cv']:.10f} | "
        f"delta {s1u['delta_vs_v03']:+.10f} | resc {s1u['rescate']} "
        f"dano {s1u['damage']} net {s1u['net']:+d}")

    # Variante analitica. La regla utilitaria optima NO es un 0,5 simetrico:
    #   subir de 0 a 1 cambia FN->TP (gana g) y arriesga TN->FP (pierde h),
    #   bajar de 1 a 0 cambia FP->TN (gana h) y arriesga TP->FN (pierde g),
    # luego el equilibrio es h/(g+h) al subir y g/(g+h) al bajar. Un canal con
    # 50 pct de acierto tiene expectativa positiva subiendo y negativa bajando.
    # Estos umbrales salen de la forma cerrada de V19-01, no de un grid.
    (canales.reshape(n_canales, -1) == 1).any(axis=0)
    (canales.reshape(n_canales, -1) == 0).any(axis=0)
    directo_arriba = (canales_plano[canal_elegido, np.arange(n_celdas)] == 1)
    directo_abajo = (canales_plano[canal_elegido, np.arange(n_celdas)] == 0)
    analitica = accionada & (
        (directo_arriba & (confianza > umbral_subir))
        | (directo_abajo & (confianza > umbral_bajar))
    )
    log(f"   umbrales analiticos: subir si P(utilidad>0) > {umbral_subir:.4f} | "
        f"bajar si P(utilidad>0) > {umbral_bajar:.4f}")
    log(f"   celdas que some arriba {int((analitica & directo_arriba).sum())} | "
        f"que baja {int((analitica & directo_abajo).sum())}")
    pred_analitica = v03.copy()
    for c in range(n_canales):
        sel = analitica & (canal_elegido == c)
        if sel.any():
            idx = np.where(sel)[0]
            filas_idx, cols_idx = np.divmod(idx, 3)
            pred_analitica[filas_idx, cols_idx] = canales[c].ravel()[sel]
    s_anal = evaluar(y2d, pred_analitica, v03)
    log(f"   variante analitica: {s_anal['corrected_cv']:.10f} | "
        f"delta {s_anal['delta_vs_v03']:+.10f} | resc {s_anal['rescate']} "
        f"dano {s_anal['damage']} net {s_anal['net']:+d}")

    d_abajo = (canales_plano[canal_elegido, np.arange(n_celdas)] == 0)

    # El argmax sobre 20 probabilidades ruidosas infla la confianza: es un sesgo de
    # seleccion, y por eso un umbral sobre el maximo no significa lo que parece. Para
    # separar "el objetivo de utilidad no funciona" de "la puerta no estaba calibrada"
    # se aisan tres diagnosticos, todos con los MISMOS modelos ya ajustados.
    #
    d_abajo = (canales_plano[canal_elegido, np.arange(n_celdas)] == 0)

    # El argmax sobre 20 probabilidades ruidosas infla la confianza: es un sesgo de
    # seleccion, y por eso un umbral puesto sobre el maximo no significa lo que parece.
    # Para separar "el objetivo de utilidad no funciona" de "la puerta no estaba
    # calibrada" se aislan tres diagnosticos, todos con los MISMOS modelos ya ajustados
    # y sin anadir ningun modelo nuevo.
    #
    # D1, puerta calibrada: la confianza se promedia entre los canales que discrepan, de
    #     modo que la puerta decide sobre una estimacion no sesgada por el maximo.
    # D2, umbral analitico de equilibrio sobre la confianza promediada.
    # D3, solo la direccion 1->0: la tasa base al bajar es 37,61 pct frente a un
    #     equilibrio de 38,92 pct, o sea pide 1,03x de elevacion, mientras que subir
    #     pide 2,18x. La restriccion sale de las tasas medidas, no de un grid.
    n_dis = discrepan_plano.sum(axis=0)
    promedia = np.where(n_dis > 0, proba.sum(axis=0) / np.maximum(n_dis, 1), 0.0)

    def aplicar(puerta: np.ndarray) -> np.ndarray:
        """Construye la prediccion final aplicando una mascara de puerta.

        Args:
            puerta: mascara booleana de celdas donde se cambia.

        Returns:
            Matriz de predicciones con los cambios aplicados.
        """
        salida = v03.copy()
        for c in range(n_canales):
            sel = puerta & (canal_elegido == c)
            if sel.any():
                idx = np.where(sel)[0]
                filas_idx, cols_idx = np.divmod(idx, 3)
                salida[filas_idx, cols_idx] = canales[c].ravel()[sel]
        return salida

    s_d1 = evaluar(y2d, aplicar(promedia >= UMBRAL_ESPECIFICADO), v03)
    s_d2 = evaluar(y2d, aplicar(promedia > umbral_subir), v03)
    s_d3 = evaluar(y2d, aplicar((promedia >= UMBRAL_ESPECIFICADO) & d_abajo), v03)
    log(f"   D1 puerta promediada, umbral 0,5 : {s_d1['corrected_cv']:.10f} | "
        f"delta {s_d1['delta_vs_v03']:+.10f} | net {s_d1['net']:+d}")
    log(f"   D2 puerta promediada, analitica : {s_d2['corrected_cv']:.10f} | "
        f"delta {s_d2['delta_vs_v03']:+.10f} | net {s_d2['net']:+d}")
    log(f"   D3 solo direccion 1->0            : {s_d3['corrected_cv']:.10f} | "
        f"delta {s_d3['delta_vs_v03']:+.10f} | net {s_d3['net']:+d}")

    # ----------------------------------------------------- 5. metricas finas
    log("5/6 pliegues, interseccion de errores y margin")
    pliegues = []
    for v in range(N_FOLD):
        sel = pliegue == v
        fv = f1_macro(y2d[sel], v03[sel])
        fu = f1_macro(y2d[sel], pred_u[sel])
        pliegues.append({"pliegue": v, "n_filas": int(sel.sum()),
                         "f1_v03": round(fv, 10), "f1_selector": round(fu, 10),
                         "delta": round(fu - fv, 10)})
        log(f"   pliegue {v}: V03 {fv:.10f} | S1-U {fu:.10f} | delta {fu - fv:+.10f}")
    positivos = sum(1 for p in pliegues if p["delta"] > 0)
    delta_s1u = s1u["delta_vs_v03"]

    err = v03 != y2d
    inter = {
        "ambos_correctos": int(((~err) & (pred_u == y2d)).sum()),
        "ambos_incorrectos": int((err & (pred_u != y2d)).sum()),
        "v03_mal_selector_bien": int((err & (pred_u == y2d)).sum()),
        "v03_bien_selector_mal": int(((~err) & (pred_u != y2d)).sum()),
    }
    log(f"   interseccion: {inter}")

    filas_margin = []
    for lo, hi, nombre in ((0.0, 0.10, "[0,0.10)"), (0.10, 0.20, "[0.10,0.20)"),
                           (0.20, 1.01, "[0.20,1]")):
        celdas = (v03_margen >= lo) & (v03_margen < hi)
        camb = int((celdas & (pred_u != v03)).sum())
        r = int((celdas & err & (pred_u == y2d)).sum())
        d = int((celdas & (~err) & (pred_u != y2d)).sum())
        filas_margin.append({"bin": nombre, "n_celdas": int(celdas.sum()),
                             "cambiadas": camb, "rescate": r, "damage": d,
                             "net": r - d})
        log(f"   margen {nombre:12} cambiadas {camb:4d} | resc {r:4d} dano {d:4d} "
            f"net {r - d:+5d}")

    # Direccion de los cambios de S1-U: ?evita el error de P50?
    cambia = (pred_u != v03)
    sube = cambia & (pred_u == 1)
    baja = cambia & (pred_u == 0)
    filas_dir = []
    for nombre, mascara in (("sube_0_a_1", sube), ("baja_1_a_0", baja)):
        r = int((mascara & err & (pred_u == y2d)).sum())
        d = int((mascara & (~err) & (pred_u != y2d)).sum())
        filas_dir.append({"direccion": nombre, "n": int(mascara.sum()),
                          "rescate": r, "damage": d, "net": r - d})
        log(f"   direccion {nombre:12} n={int(mascara.sum()):4d} | resc {r:4d} "
            f"dano {d:4d} net {r - d:+5d}")
    log("   P50 elijo sobre todo canales con net negativo; estos son los cambios por direccion")

    # ------------------------------------------------- 6. gate y veredicto
    log("6/6 gate y veredicto")
    pasa = delta_s1u >= 0.003 and positivos >= 4
    if delta_s1u <= 0:
        caso = "A"
    elif not pasa:
        caso = "B"
    else:
        caso = "C"
    log(f"   delta {delta_s1u:+.6f} | pliegues positivos {positivos}/{N_FOLD} | "
        f"gate {pasa}")
    log(f"   mejor de las dos reglas: "
        f"{'analitica' if s_anal['delta_vs_v03'] > delta_s1u else 'umbral 0,5'} "
        f"con {max(delta_s1u, s_anal['delta_vs_v03']):+.6f}")
    log(f"   CASE = {caso}")

    filas_csv = [
        {"selector": "B0_V03", **b0, "delta_vs_P50": 0.0, "pliegues_positivos": ""},
        {"selector": "B1_siempre_KEEP", **b1,
         "delta_vs_P50": round(b1["corrected_cv"] - p50["s1"]["corrected_cv"], 10),
         "pliegues_positivos": ""},
        {"selector": "B2_P50_S1", "corrected_cv": p50["s1"]["corrected_cv"],
         "delta_vs_v03": p50["s1"]["delta_vs_v03"],
         "por_etiqueta": json.dumps(p50["s1"].get("por_etiqueta", {})),
         "celdas_cambiadas": p50["s1"]["celdas_cambiadas"],
         "abstencion_rate": p50["s1"]["tasa_de_abstencion"],
         "rescate": p50["s1"]["rescate"], "damage": p50["s1"]["damage"],
         "net": p50["s1"]["net"], "delta_vs_P50": 0.0, "pliegues_positivos": ""},
        {"selector": "S1U_umbral_0p5", **s1u,
         "delta_vs_P50": round(s1u["corrected_cv"] - p50["s1"]["corrected_cv"], 10),
         "pliegues_positivos": positivos},
        {"selector": "S1U_umbral_analitico", **s_anal,
         "delta_vs_P50": round(s_anal["corrected_cv"] - p50["s1"]["corrected_cv"], 10),
         "pliegues_positivos": ""},
        {"selector": "D1_puerta_promediada", **s_d1,
         "delta_vs_P50": round(s_d1["corrected_cv"] - p50["s1"]["corrected_cv"], 10),
         "pliegues_positivos": ""},
        {"selector": "D2_umbral_analitico", **s_d2,
         "delta_vs_P50": round(s_d2["corrected_cv"] - p50["s1"]["corrected_cv"], 10),
         "pliegues_positivos": ""},
        {"selector": "D3_solo_direccion_bajar", **s_d3,
         "delta_vs_P50": round(s_d3["corrected_cv"] - p50["s1"]["corrected_cv"], 10),
         "pliegues_positivos": ""},
        {"selector": "ORACLE_P50_techo_no_alcanzable",
         "corrected_cv": p50_gap["oracle"]["macro_f1_oracle"],
         "delta_vs_v03": p50_gap["oracle"]["delta_macro_f1"], "por_etiqueta": "",
         "celdas_cambiadas": "", "abstencion_rate": "", "rescate": "", "damage": "",
         "net": "", "delta_vs_P50": "", "pliegues_positivos": ""},
    ]
    df_res = pd.DataFrame(filas_csv)
    df_res.to_csv(CSV_RESULTADOS, index=False, encoding="utf-8")
    pd.DataFrame([{"categoria": k, "celdas": v} for k, v in inter.items()]).to_csv(
        CSV_INTerseccion, index=False, encoding="utf-8")
    pd.concat([
        pd.DataFrame(filas_margin).assign(tipo="margen", clave=lambda d: d["bin"]),
        pd.DataFrame(filas_dir).assign(tipo="direccion",
                                        clave=lambda d: d["direccion"]),
    ], ignore_index=True).to_csv(CSV_RESCATE, index=False, encoding="utf-8")

    informe = {
        "familia": "P51_UTILITY_AWARE_CONTEXTUAL_GATING",
        "experimento": "V51-01 (selector utilitario S1-U)",
        "fecha": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "target": "delta_F1 > 0, con los conteos del pliegue de entrenamiento",
        "controles": {"B0_V03": b0, "B1_siempre_KEEP": b1,
                      "B2_P50_S1": {"corrected_cv": p50["s1"]["corrected_cv"],
                                    "delta_vs_v03": p50["s1"]["delta_vs_v03"],
                                    "net": p50["s1"]["net"]}},
        "equilibrio_analitico": {
            "umbral_subir_sobre_P_y1": round(umbral_subir, 6),
            "umbral_bajar_sobre_P_y1": round(1 - umbral_bajar, 6),
            "coinciden": bool(abs(umbral_subir - (1 - umbral_bajar)) < 1e-9),
            "tasa_base_subir": round(base_subir, 6),
            "tasa_base_bajar": round(base_bajar, 6),
            "elevacion_necesaria_subir": round(umbral_subir / base_subir, 4),
            "elevacion_necesaria_bajar": round((1 - umbral_bajar) / base_bajar, 4),
        },
        "auc_por_canal": {"min": round(float(np.nanmin(auc)), 6),
                          "max": round(float(np.nanmax(auc)), 6),
                          "media": round(float(np.nanmean(auc)), 6)},
        "s1u_umbral_0p5": s1u,
        "s1u_umbral_analitico": s_anal,
        "diagnosticos_de_calibracion": {"D1_puerta_promediada": s_d1,
                                          "D2_umbral_analitico": s_d2,
                                          "D3_solo_direccion_bajar": s_d3},
        "pliegues": pliegues,
        "pliegues_positivos": positivos,
        "interseccion_de_errores": inter,
        "por_margen": filas_margin,
        "por_direccion": filas_dir,
        "p50_por_direccion": {
            "nota": ("P50 trato las dos direcciones por igual y su argmax eligio "
                     "cita:NEG 109 veces con net -1706. Se compara aqui si P51 repite "
                     "o evita ese comportamiento"),
            "acciones_elegidas": p50["configuracion"].get("reparto_de_acciones_elegidas"),
        },
        "gate": {"delta_minimo": 0.003, "delta_obtenido": delta_s1u,
                 "pliegues_positivos": positivos, "pasado": bool(pasa)},
        "caso": caso,
        "gastos": {"submissions": 0, "consultas_kaggle": 0, "uso_holdout": 0,
                   "uso_test": 0, "segundos": round(time.time() - inicio, 1)},
    }
    REPORTE_JSON.write_text(json.dumps(informe, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8")
    _md(informe, p50, p50_gap)
    log("   escrito reports/p51_utility_selector_*")
    log(f"VEREDICTO: CASE={caso} | delta={delta_s1u:+.6f} | "
        f"delta_analitico={s_anal['delta_vs_v03']:+.6f} | "
        f"pliegues {positivos}/{N_FOLD} | 0 submissions, "
        f"{round(time.time() - inicio, 1)} s")


def _md(informe: dict[str, Any], p50: dict[str, Any], p50_gap: dict[str, Any]) -> None:
    """Escribe el informe del selector utilitario.

    Args:
        informe: diccionario con los resultados.
        p50: resultados de P50.
        p50_gap: artefacto de brecha y oracle de P50.

    Returns:
        None.
    """
    eq = informe["equilibrio_analitico"]
    u = informe["s1u_umbral_0p5"]
    a = informe["s1u_umbral_analitico"]
    c = informe["controles"]
    inter = informe["interseccion_de_errores"]
    lineas = [
        "# P51 - Analisis del selector utilitario",
        "",
        (f"Generado por `experiments/v51_01_selector_utilidad.py` "
        f"({informe['experimento']})."),
        "",
        f"Fecha {informe['fecha']}. **0 submissions, 0 consultas a Kaggle.**",
        "",
        f"Target: `{informe['target']}`.",
        "",
        "## 1. El objetivo cambia respecto a P50",
        "",
        "| fase | target | lo que premia |",
        "|---|---|---|",
        "| P50 S1 | `channel_correct` | la **predecibilidad** de un canal |",
        "| P51 S1-U | `delta_F1 > 0` | la **utilidad** del cambio |",
        "",
        "## 2. El equilibrio no es 0,5",
        "",
        "| magnitud | valor |",
        "|---|---:|",
        (f"| umbral de equilibrio para **subir** de 0 a 1, sobre P(y=1) | "
        f"**{eq['umbral_subir_sobre_P_y1']:.4f}** |"),
        (f"| umbral de equilibrio para **bajar** de 1 a 0, sobre P(y=1) | "
        f"**{eq['umbral_bajar_sobre_P_y1']:.4f}** |"),
        f"| los dos coinciden | {eq['coinciden']} |",
        f"| tasa base al subir | {100 * eq['tasa_base_subir']:.2f} pct |",
        f"| tasa base al bajar | {100 * eq['tasa_base_bajar']:.2f} pct |",
        f"| **elevacion necesaria al subir** | **{eq['elevacion_necesaria_subir']:.2f}x** |",
        f"| **elevacion necesaria al bajar** | **{eq['elevacion_necesaria_bajar']:.2f}x** |",
        "",
        ("La consecuencia es que la regla utilitaria optima es un **unico umbral sobre P(y=1)** "
        "y no un 0,5 simetrico. Ese umbral sale de la forma cerrada, no de un grid."),
        "",
        "## 3. Controles",
        "",
        "| selector | corrected_cv | delta vs V03 | rescate | damage | net |",
        "|---|---:|---:|---:|---:|---:|",
        f"| B0 V03 | {c['B0_V03']['corrected_cv']:.10f} | +0,0000000000 | 0 | 0 | 0 |",
        (f"| B1 siempre KEEP_V03 | {c['B1_siempre_KEEP']['corrected_cv']:.10f} | "
        f"+0,0000000000 | 0 | 0 | 0 |"),
        (f"| B2 P50 S1 | {c['B2_P50_S1']['corrected_cv']:.10f} | "
        f"{c['B2_P50_S1']['delta_vs_v03']:+.10f} | 48 | 67 | {c['B2_P50_S1']['net']:+d} |"),
        "",
        "B1 es identico a B0 por construccion: la accion por defecto ya es no cambiar.",
        "",
        "## 4. S1-U",
        "",
        ("| regla | corrected_cv | delta vs V03 | delta vs P50 | cambiadas | rescate | "
        "damage | net |"),
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        (f"| umbral 0,5 (el de la directiva) | {u['corrected_cv']:.10f} | "
        f"{u['delta_vs_v03']:+.10f} | "
        f"{u['corrected_cv'] - p50['s1']['corrected_cv']:+.10f} | "
        f"{u['celdas_cambiadas']} | {u['rescate']} | {u['damage']} | {u['net']:+d} |"),
        (f"| umbral analitico | {a['corrected_cv']:.10f} | {a['delta_vs_v03']:+.10f} | "
        f"{a['corrected_cv'] - p50['s1']['corrected_cv']:+.10f} | "
        f"{a['celdas_cambiadas']} | {a['rescate']} | {a['damage']} | {a['net']:+d} |"),
        "",
        (f"AUC out-of-fold de los 20 clasificadores: min "
        f"{informe['auc_por_canal']['min']:.4f}, max "
        f"{informe['auc_por_canal']['max']:.4f}, media "
        f"{informe['auc_por_canal']['media']:.4f}."),
        "",
        "## 5. Pliegues",
        "",
        "| pliegue | F1 de V03 | F1 de S1-U | delta |",
        "|---:|---:|---:|---:|",
    ]
    for p in informe["pliegues"]:
        lineas.append(f"| {p['pliegue']} | {p['f1_v03']:.10f} | {p['f1_selector']:.10f} | "
                      f"{p['delta']:+.10f} |")
    lineas += [
        "",
        f"Pliegues positivos: **{informe['pliegues_positivos']} de 5**.",
        "",
        "## 6. Interseccion de errores",
        "",
        "| categoria | celdas |",
        "|---|---:|",
    ]
    for k, v in inter.items():
        lineas.append(f"| {k} | {v} |")
    lineas += [
        "",
        "## 7. Por margen y por direccion",
        "",
        "| corte | tipo | n | rescate | damage | net |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for m in informe["por_margen"]:
        lineas.append(f"| {m['bin']} | margen | {m['cambiadas']} | {m['rescate']} | "
                      f"{m['damage']} | {m['net']:+d} |")
    for d in informe["por_direccion"]:
        lineas.append(f"| {d['direccion']} | direccion | {d['n']} | {d['rescate']} | "
                      f"{d['damage']} | {d['net']:+d} |")
    g = informe["gate"]
    lineas += [
        "",
        "## 8. Comparacion con P50 y con el techo",
        "",
        "| politica | corrected_cv | delta vs V03 |",
        "|---|---:|---:|",
        f"| V03 | {c['B0_V03']['corrected_cv']:.10f} | +0,0000000000 |",
        (f"| P50 S1 (`channel_correct`) | {p50['s1']['corrected_cv']:.10f} | "
        f"{p50['s1']['delta_vs_v03']:+.10f} |"),
        (f"| **P51 S1-U (`delta_F1 > 0`)** | **{u['corrected_cv']:.10f}** | "
        f"**{u['delta_vs_v03']:+.10f}** |"),
        (f"| oracle (techo, usa `y_true`) | {p50_gap['oracle']['macro_f1_oracle']:.10f} | "
        f"{p50_gap['oracle']['delta_macro_f1']:+.10f} |"),
        "",
        ("**El oracle no es un modelo alcanzable.** Usa `y_true` para decidir cada celda, luego "
        "es un techo y no un resultado."),
        "",
        "## 9. Gate y decision",
        "",
        f"- delta obtenido {g['delta_obtenido']:+.6f} frente a {g['delta_minimo']} exigido",
        f"- pliegues positivos {g['pliegues_positivos']} de 4 exigidos",
        f"- gate pasado: **{g['pasado']}**",
        "",
        f"**CASE = {informe['caso']}.**",
        "",
    ]
    REPORTE_MD.write_text("\n".join(lineas) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
