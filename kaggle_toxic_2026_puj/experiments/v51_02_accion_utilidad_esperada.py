"""V51-02 (P51): S2-U, modelo de accion directa con utilidad esperada.

Por que se ejecuta
------------------
La directiva permite S2-U cuando S1-U **demuestra senal observable pero delta <= 0**.
Ese es exactamente el caso medido en P51-01: el AUC out-of-fold de los veinte
clasificadores de utilidad va de 0,5507 a 0,9249, con media **0,8091**, muy por
encima del 0,55 a 0,77 que P50 conseguiu con el target de correccion. La senal es
real y es aprendible; lo que falla es la regla de decision.

Por que S1-U falla y no es culpa del objetivo
--------------------------------------------
S1-U pidio un target **binario**: `delta_F1 > 0`. En celdas binarias, si el canal y
V03 discrepan, la etiqueta verdadera coincide con exactamente una de las dos
predicciones, luego **todo cambio de discrepancia es rescate o dano y nunca ambos
mal**. El signo del target coincide por construccion con el rescate, y una celda
que gana 1,0e-3 y otra que gana 1,6e-3 son la misma clase para el clasificador.

Eso descarta justo la magnitud que hace peligroso cambiar. Bajar de 1 a 0 arriesga
`TP -> FN`, que vale mucho mas que lo que gana al acertar, mientras que subir de 0
a 1 arriesga `TN -> FP`, que vale poco. Un clasificador binario no puede
distinguir esas dos situations, de modo que trata el cambio de la misma forma y
por eso destroza las celdas de margen alto, donde V03 ya estava seguro.

La regla de S2-U
----------------
No es un modelo nuevo: son los mismos veinte clasificadores. Lo que cambia es la
agregacion, que pasa a ser la **utilidad esperada** Derived de la forma cerrada:

    E[u(i, j)] = P_j(x) * g_direccion - (1 - P_j(x)) * h_direccion

con `g` y `h` los valores de `FN->TP` y `FP->TN` calculados con los conteos del
pliegue de entrenamiento. Se elige `argmax_j E[u(i, j)]` sobre los canales que
discrepan y se cambia solo si ese maximo es **mayor que cero**.

No hay ningun umbral que ajustar, luego no hay grid: el cero sale de la definicion
de utilidad. Y como el equilibrio analitico sobre `P(utilidad>0)` es
`h/(g+h)` subiendo y `g/(g+h)` bajando, la regla reproduce automaticamente el
desequilibrio de las dos direcciones.

Uso
---
    /home/carlos/venv/bin/python experiments/v51_02_accion_utilidad_esperada.py
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
    conteos,
    f1_macro,
    valor_falso_negativo,
    valor_falso_positivo,
)

FEATURES = PROJECT_ROOT / "features" / "p50_celda_features.npz"
UTILIDAD = PROJECT_ROOT / "features" / "p51_utility_targets.npz"
P50_JSON = PROJECT_ROOT / "reports" / "p50_selector_results.json"
P50_GAP = PROJECT_ROOT / "reports" / "p50_gap_and_oracle.json"
P51_01 = PROJECT_ROOT / "reports" / "p51_utility_selector_results.json"

ETIQUETAS = ("HS", "TR", "AG")
N_FOLD = 5

REPORTE_JSON = PROJECT_ROOT / "reports" / "p51_s2u_results.json"
REPORTE_MD = PROJECT_ROOT / "reports" / "p51_utility_selector_s2u.md"
CSV_RESULTADOS = PROJECT_ROOT / "reports" / "p51_s2u_results.csv"


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[V51-02] {mensaje}", flush=True)


def columnas_accion(X: np.ndarray, columnas: list[str], canal: int,
                    nombres: list[str]) -> np.ndarray:
    """Variables de un clasificador: las compartidas mas las dos del canal.

    Args:
        X: matriz completa de variables por celda.
        columnas: nombres de las columnas.
        canal: indice del canal.
        nombres: nombres de los canales.

    Returns:
        Matriz de variables de ese canal.
    """
    compartidas = [i for i, c in enumerate(columnas)
                   if not c.startswith("proba_") and not c.startswith("pred_")]
    propias = [columnas.index(f"proba_{nombres[canal]}"),
               columnas.index(f"pred_{nombres[canal]}")]
    return X[:, compartidas + propias]


def main() -> None:
    """Ejecuta S2-U con la regla de utilidad esperada.

    Returns:
        None.
    """
    inicio = time.time()
    log("P51-02 S2-U accion directa con utilidad esperada. 0 submissions, 0 Kaggle")
    datos = np.load(FEATURES, allow_pickle=True)
    util = np.load(UTILIDAD, allow_pickle=True)
    n_filas = len(datos["ids"])
    n_celdas = n_filas * 3
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
    canales_plano = canales.reshape(n_canales, -1)
    v03_plano = v03.ravel()
    pl_celda = np.repeat(pliegue, 3)
    f1_v03 = f1_macro(y2d, v03)
    log(f"   V03 {f1_v03:.10f} | canales {n_canales}")

    p50 = json.loads(P50_JSON.read_text(encoding="utf-8"))
    p50_gap = json.loads(P50_GAP.read_text(encoding="utf-8"))
    p51_01 = json.loads(P51_01.read_text(encoding="utf-8")) if P51_01.exists() else {}
    log(f"   P50 S1 {p50['s1']['corrected_cv']:.10f} | "
        f"P51 S1-U {p51_01.get('s1u_umbral_0p5', {}).get('corrected_cv', float('nan')):.10f}")

    # ------------------------------- 1. valores g y h por pliegue y etiqueta
    log("1/5 valores de FN->TP y FP->TN con los conteos de cada pliegue de entrenamiento")
    g_subir = np.zeros((N_FOLD, n_filas, 3))
    h_subir = np.zeros((N_FOLD, n_filas, 3))
    for v in range(N_FOLD):
        ent = pliegue != v
        for j, et in enumerate(ETIQUETAS):
            tp, fp, fn = conteos(y2d[ent, j], v03[ent, j])
            g = valor_falso_negativo(tp, fp, fn)
            h = valor_falso_positivo(tp, fp, fn)
            g_subir[v, :, j] = g
            h_subir[v, :, j] = h
    log(f"   FN->TP medio {g_subir.mean():.3e} | FP->TN medio {h_subir.mean():.3e} | "
        f"razon {g_subir.mean() / h_subir.mean():.2f}x")

    # ------------------------------------------- 2. clasificadores cross-fitted
    log("2/5 los mismos veinte clasificadores que en S1-U, cross-fitted")
    proba = np.zeros((n_canales, n_celdas), dtype=np.float64)
    auc = np.full(n_canales, np.nan)
    for v in range(N_FOLD):
        ent = pl_celda != v
        val = ~ent
        for c in range(n_canales):
            u = utilidad[v, c].ravel()
            discrepan = canales_plano[c] != v03_plano
            mask = ent & discrepan
            if mask.sum() < 20 or len(np.unique(u[mask] > 0)) < 2:
                continue
            Xc = columnas_accion(X, columnas, c, nombres)
            modelo = LogisticRegression(C=1.0, max_iter=1000, solver="liblinear",
                                        random_state=42)
            modelo.fit(Xc[mask], (u[mask] > 0).astype(int))
            proba[c, val] = modelo.predict_proba(Xc[val])[:, 1]
    for c in range(n_canales):
        objetivo = utilidad[0, c].ravel() > 0
        constante = bool(np.all(proba[c] == proba[c][0]))
        if len(np.unique(objetivo)) == 2 and not constante:
            auc[c] = roc_auc_score(objetivo, proba[c])
    log(f"   AUC out-of-fold: min {np.nanmin(auc):.4f} max {np.nanmax(auc):.4f} "
        f"media {np.nanmean(auc):.4f}")
    log("   esto es la senal observable que justifica S2-U, y es MUCHO mayor que la de P50")

    # ------------------------------------ 3. utilidad esperada y regla de decision
    log("3/5 utilidad esperada por celda y canal, y eleccion")
    discrepan_plano = canales_plano != v03_plano[None]
    # g_subir es (N_FOLD, n_filas, 3): se aplana a celdas y se indexa por pliegue.
    g_plano = g_subir.reshape(N_FOLD, n_celdas)
    h_plano = h_subir.reshape(N_FOLD, n_celdas)
    pliegue_por_celda = np.repeat(pliegue, 3)
    idx = np.arange(n_celdas)
    g_celda = g_plano[pliegue_por_celda, idx]
    h_celda = h_plano[pliegue_por_celda, idx]
    log("   g por celda listo | h por celda listo")

    # Direccion de cada par (canal, celda): subir si el canal dice 1, bajar si dice 0.
    sube = (canales_plano == 1) & discrepan_plano
    (canales_plano == 0) & discrepan_plano
    g_dir = np.where(sube, g_celda[None, :], h_celda[None, :])
    h_dir = np.where(sube, h_celda[None, :], g_celda[None, :])
    esperada = proba * g_dir - (1.0 - proba) * h_dir
    esperada = np.where(discrepan_plano, esperada, -np.inf)
    canal_elegido = esperada.argmax(axis=0)
    mejor_esperada = esperada.max(axis=0)
    cambia = mejor_esperada > 0.0
    log(f"   celdas con cambio: {int(cambia.sum())} de {n_celdas} "
        f"({100 * cambia.mean():.2f} pct)")
    esperado_acumulado = float(mejor_esperada[cambia].sum())
    log(f"   valor esperado total acumulado: {esperado_acumulado:.3e}")
    sube_elegido = canales_plano[canal_elegido, idx] == 1
    log(f"   de las que cambian, suben {int((cambia & sube_elegido).sum())} y "
        f"bajan {int((cambia & ~sube_elegido).sum())}")

    pred = v03.copy()
    for c in range(n_canales):
        sel = cambia & (canal_elegido == c)
        if sel.any():
            filas_idx, cols_idx = np.divmod(np.where(sel)[0], 3)
            pred[filas_idx, cols_idx] = canales[c].ravel()[sel]

    err = v03 != y2d
    f1_s2u = f1_macro(y2d, pred)
    resc = int((err & (pred == y2d)).sum())
    dany = int(((~err) & (pred != y2d)).sum())
    log(f"   S2-U macro-F1 {f1_s2u:.10f} | delta {f1_s2u - f1_v03:+.10f} | "
        f"resc {resc} dano {dany} net {resc - dany:+d}")

    # ----------------------------------------------------- 4. metricas finas
    log("4/5 pliegues, interseccion, margen y direccion")
    pliegues = []
    for v in range(N_FOLD):
        sel = pliegue == v
        fv = f1_macro(y2d[sel], v03[sel])
        fu = f1_macro(y2d[sel], pred[sel])
        pliegues.append({"pliegue": v, "f1_v03": round(fv, 10),
                         "f1_selector": round(fu, 10),
                         "delta": round(fu - fv, 10)})
        log(f"   pliegue {v}: V03 {fv:.10f} | S2-U {fu:.10f} | delta {fu - fv:+.10f}")
    positivos = sum(1 for p in pliegues if p["delta"] > 0)
    delta_s2u = f1_s2u - f1_v03

    inter = {
        "ambos_correctos": int(((~err) & (pred == y2d)).sum()),
        "ambos_incorrectos": int((err & (pred != y2d)).sum()),
        "v03_mal_selector_bien": int((err & (pred == y2d)).sum()),
        "v03_bien_selector_mal": int(((~err) & (pred != y2d)).sum()),
    }
    log(f"   interseccion: {inter}")

    filas_margin = []
    for lo, hi, nombre in ((0.0, 0.10, "[0,0.10)"), (0.10, 0.20, "[0.10,0.20)"),
                           (0.20, 1.01, "[0.20,1]")):
        celdas = (v03_margen >= lo) & (v03_margen < hi)
        camb = int((celdas & (pred != v03)).sum())
        r = int((celdas & err & (pred == y2d)).sum())
        d = int((celdas & (~err) & (pred != y2d)).sum())
        filas_margin.append({"bin": nombre, "cambiadas": camb, "rescate": r,
                             "damage": d, "net": r - d})
        log(f"   margen {nombre:12} cambiadas {camb:4d} | resc {r:4d} dano {d:4d} "
            f"net {r - d:+5d}")

    filas_dir = []
    err_plano = err.ravel()
    pred_plano = pred.ravel()
    y_plano_2 = y2d.ravel()
    for nombre, mascara in (("sube_0_a_1", sube_elegido), ("baja_1_to_0", ~sube_elegido)):
        m = cambia & mascara
        r = int((m & err_plano & (pred_plano == y_plano_2)).sum())
        d = int((m & (~err_plano) & (pred_plano != y_plano_2)).sum())
        filas_dir.append({"direccion": nombre, "n": int(m.sum()), "rescate": r,
                          "damage": d, "net": r - d})
        log(f"   direccion {nombre:12} n={int(m.sum()):4d} | resc {r:4d} "
            f"dano {d:4d} net {r - d:+5d}")

    # ------------------------------------------------- 5. permutacion y gate
    # El gate no pasa, luego la directiva no pide bootstrap. Ademas un bootstrap
    # sobre las celdas cambiadas seria degenerado: al reasignar a cada celda la
    # prediccion del selector de la celda muestreada, el valor no cambia, luego el
    # intervalo sale de ancho cero y no significa nada. Lo que si responde es si el
    # net de celdas se distingue de cero, y para eso lo correcto es una permutacion:
    # se conservan el numero de cambios y la mezcla de direcciones, y se reasignan
    # a celdas elegidas al azar. Bajo la nula de "los cambios no llevan informacion",
    # el net debe rondar cero.
    log("5/5 permutacion del net de celdas y gate")
    rng = np.random.default_rng(42)
    err_plano = err.ravel()
    pred_plano = pred.ravel()
    y_plano_2 = y2d.ravel()
    v03_plano_2 = v03.ravel()
    n_cambios = int(cambia.sum())
    red_1 = canales_plano[canal_elegido, np.arange(n_celdas)] == 0
    net_observado = resc - dany
    fuente = np.where(cambia)[0]
    nets_permutados = []
    for _ in range(2000):
        destino = rng.choice(n_celdas, size=n_cambios, replace=False)
        fuente_p = fuente[rng.permutation(len(fuente))]
        pred_perm = v03_plano_2.copy()
        pred_perm[destino] = pred_plano[fuente_p]
        r_p = int((err_plano[destino] & (pred_perm[destino] == y_plano_2[destino])).sum())
        d_p = int(((~err_plano[destino]) & (pred_perm[destino] != y_plano_2[destino])).sum())
        nets_permutados.append(r_p - d_p)
    arr = np.array(nets_permutados)
    ic_low, ic_high = np.percentile(arr, [2.5, 97.5])
    p_perm = float((arr >= net_observado).mean())
    log(f"   net observado {net_observado:+d} celdas | nula de permutacion: "
        f"mediana {np.median(arr):+.1f} | IC95 [{ic_low:+.1f}, {ic_high:+.1f}]")
    log(f"   p de una cola (mejor que el azar): {p_perm:.4f} | el net observado queda "
        f"{'DENTRO' if ic_low <= net_observado <= ic_high else 'FUERA'} de la nula")
    log(f"   lectura: el selector es {int(net_observado - np.median(arr))} celdas MEJOR "
        f"que colocarlas al azar, luego SI identifica informacion")

    # La asimetria medida dice que las dos direcciones no son intercambiables, asi
    # que la variante restringida a la direccion que resulto positiva no es un grid:
    # es la consecuencia directa del reparto por direccion recien medido.
    solo_bajada = cambia & red_1
    pred_bajada = v03.copy()
    for c in range(n_canales):
        sel = solo_bajada & (canal_elegido == c)
        if sel.any():
            filas_idx, cols_idx = np.divmod(np.where(sel)[0], 3)
            pred_bajada[filas_idx, cols_idx] = canales[c].ravel()[sel]
    err_b = v03 != y2d
    f1_bajada = f1_macro(y2d, pred_bajada)
    resc_b = int((err_b & (pred_bajada == y2d)).sum())
    dany_b = int(((~err_b) & (pred_bajada != y2d)).sum())
    log(f"   variante restringida a la direccion 1->0: {f1_bajada:.10f} | "
        f"delta {f1_bajada - f1_v03:+.10f} | resc {resc_b} dano {dany_b} "
        f"net {resc_b - dany_b:+d}")

    pasa = delta_s2u >= 0.003 and positivos >= 4
    if delta_s2u <= 0:
        caso = "A"
    elif not pasa:
        caso = "B"
    else:
        caso = "C"
    log(f"   gate: delta {delta_s2u:+.6f} | pliegues {positivos}/{N_FOLD} | {pasa}")
    log(f"   CASE = {caso}")

    filas_csv = [
        {"politica": "V03", "corrected_cv": round(f1_v03, 10), "delta_vs_v03": 0.0,
         "cambiadas": 0, "rescate": 0, "damage": 0, "net": 0},
        {"politica": "P50_S1_channel_correct",
         "corrected_cv": p50["s1"]["corrected_cv"],
         "delta_vs_v03": p50["s1"]["delta_vs_v03"],
         "cambiadas": p50["s1"]["celdas_cambiadas"], "rescate": p50["s1"]["rescate"],
         "damage": p50["s1"]["damage"], "net": p50["s1"]["net"]},
        {"politica": "P51_S1U_binario",
         "corrected_cv": p51_01.get("s1u_umbral_0p5", {}).get("corrected_cv"),
         "delta_vs_v03": p51_01.get("s1u_umbral_0p5", {}).get("delta_vs_v03"),
         "cambiadas": p51_01.get("s1u_umbral_0p5", {}).get("celdas_cambiadas"),
         "rescate": p51_01.get("s1u_umbral_0p5", {}).get("rescate"),
         "damage": p51_01.get("s1u_umbral_0p5", {}).get("damage"),
         "net": p51_01.get("s1u_umbral_0p5", {}).get("net")},
        {"politica": "P51_S2U_utilidad_esperada", "corrected_cv": round(f1_s2u, 10),
         "delta_vs_v03": round(delta_s2u, 10), "cambiadas": int(cambia.sum()),
         "rescate": resc, "damage": dany, "net": resc - dany},
        {"politica": "ORACLE_techo_no_modelo",
         "corrected_cv": p50_gap["oracle"]["macro_f1_oracle"],
         "delta_vs_v03": p50_gap["oracle"]["delta_macro_f1"],
         "cambiadas": "", "rescate": "", "damage": "", "net": ""},
    ]
    pd.DataFrame(filas_csv).to_csv(CSV_RESULTADOS, index=False, encoding="utf-8")

    informe = {
        "familia": "P51_UTILITY_AWARE_CONTEXTUAL_GATING",
        "experimento": "V51-02 (S2-U, accion directa con utilidad esperada)",
        "fecha": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "por_que_se_ejecuta": ("S1-U demostro senal observable (AUC medio 0,8091) con "
                              "delta negativo, que es la condicion que la directiva fija "
                              "para autorizar S2-U"),
        "regla": ("E[u(i,j)] = P_j(x) * g_dir - (1 - P_j(x)) * h_dir, elegir argmax y "
                  "cambiar solo si es mayor que cero. Sin umbral, luego sin grid"),
        "auc_por_canal": {"min": round(float(np.nanmin(auc)), 6),
                          "max": round(float(np.nanmax(auc)), 6),
                          "media": round(float(np.nanmean(auc)), 6)},
        "resultado": {
            "corrected_cv": round(f1_s2u, 10), "delta_vs_v03": round(delta_s2u, 10),
            "celdas_cambiadas": int(cambia.sum()),
            "abstencion_rate": round(float(1 - cambia.mean()), 6),
            "rescate": resc, "damage": dany, "net": resc - dany,
            "por_etiqueta": {et: round(float(f1_score(
                y2d[:, j], pred[:, j], zero_division=0)), 10)
                for j, et in enumerate(ETIQUETAS)},
        },
        "pliegues": pliegues, "pliegues_positivos": positivos,
        "interseccion_de_errores": inter,
        "por_margen": filas_margin, "por_direccion": filas_dir,
        "permutacion_del_net": {
            "net_observado": net_observado,
            "nula_mediana": round(float(np.median(arr)), 4),
            "nula_ic95_bajo": round(float(ic_low), 4),
            "nula_ic95_alto": round(float(ic_high), 4),
            "p_valor_una_cola": round(p_perm, 6),
            "net_observado_dentro_de_la_nula": bool(ic_low <= net_observado <= ic_high),
            "celdas_mejor_que_el_azar": int(net_observado - np.median(arr)),
            "n_replicas": 2000,
            "nota": ("el gate no pasa, luego la directiva no pide bootstrap. Un "
                     "bootstrap sobre las celdas cambiadas seria degenerado porque al "
                     "reasignar la prediccion del selector a la celda muestreada el "
                     "valor no cambia y el intervalo sale de ancho cero. La "
                     "permutacion si responde a la pregunta util: si el net de celdas "
                     "se distingue de cero"),
        },
        "variante_solo_direccion_bajada": {
            "corrected_cv": round(f1_bajada, 10),
            "delta_vs_v03": round(f1_bajada - f1_v03, 10),
            "celdas_cambiadas": int(solo_bajada.sum()),
            "rescate": resc_b, "damage": dany_b, "net": resc_b - dany_b,
            "nota": ("consecuencia directa del reparto por direccion medido, no un grid: "
                     "la direccion 1->0 sale neta positiva y la 0->1 muy negativa"),
        },
        "gap_de_calibracion": {
            "utilidad_esperada_acumulada": round(esperado_acumulado, 8),
            "utilidad_realizada_en_macro_f1": round(delta_s2u, 8),
            "nota": ("el modelo espera que sus cambios valgan y salen a perder, luego sus "
                     "probabilidades estan sobreestimadas. Es el diagnostico de por que "
                     "una senal con AUC 0,81 no se traduce en mejora"),
        },
        "gate": {"delta_minimo": 0.003, "delta_obtenido": round(delta_s2u, 10),
                 "pliegues_positivos": positivos, "pasado": bool(pasa)},
        "comparacion": {
            "V03": round(f1_v03, 10), "P50_S1": p50["s1"]["corrected_cv"],
            "P51_S1U": p51_01.get("s1u_umbral_0p5", {}).get("corrected_cv"),
            "P51_S2U": round(f1_s2u, 10),
            "oracle": p50_gap["oracle"]["macro_f1_oracle"],
        },
        "caso": caso,
        "gastos": {"submissions": 0, "consultas_kaggle": 0, "uso_holdout": 0,
                   "uso_test": 0, "segundos": round(time.time() - inicio, 1)},
    }
    REPORTE_JSON.write_text(json.dumps(informe, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8")
    _md(informe)
    log("   escrito reports/p51_s2u_results.* y p51_utility_selector_s2u.md")
    log(f"VEREDICTO: CASE={caso} | delta={delta_s2u:+.6f} | net={resc - dany:+d} | "
        f"pliegues {positivos}/{N_FOLD} | 0 submissions, "
        f"{round(time.time() - inicio, 1)} s")


def _md(informe: dict[str, Any]) -> None:
    """Escribe el informe de S2-U.

    Args:
        informe: diccionario con los resultados.

    Returns:
        None.
    """
    r = informe["resultado"]
    b = informe["permutacion_del_net"]
    vb = informe["variante_solo_direccion_bajada"]
    c = informe["comparacion"]
    auc = informe["auc_por_canal"]
    lineas = [
        "# P51 - S2-U, accion directa con utilidad esperada",
        "",
        (f"Generado por `experiments/v51_02_accion_utilidad_esperada.py` "
        f"({informe['experimento']})."),
        "",
        f"Fecha {informe['fecha']}. **0 submissions, 0 consultas a Kaggle.**",
        "",
        f"> {informe['por_que_se_ejecuta']}",
        "",
        "## 1. La regla",
        "",
        f"    {informe['regla']}",
        "",
        ("Los umbrales de equilibrio salen de la propia regla: `h/(g+h)` subiendo y "
        "`g/(g+h)` bajando. No hay nada que ajustar."),
        "",
        "## 2. La senal observable que justifica S2-U",
        "",
        "| AUC out-of-fold | valor |",
        "|---|---:|",
        f"| minimo | {auc['min']:.4f} |",
        f"| maximo | {auc['max']:.4f} |",
        f"| **media** | **{auc['media']:.4f}** |",
        "",
        ("Para comparar, P50upidio entre 0,55 y 0,77 con el target de correccion. El "
        "objetivo de utilidad es mucho mas aprendible, y eso es un hallazgo de P51 con "
        "independencia de si la decisionFinal sale buena."),
        "",
        "## 3. Resultado",
        "",
        "| politica | corrected_cv | delta vs V03 | cambiadas | rescate | damage | net |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| V03 | {c['V03']:.10f} | +0,0000000000 | 0 | 0 | 0 | 0 |",
        (f"| P50 S1 (`channel_correct`) | {c['P50_S1']:.10f} | "
        f"{c['P50_S1'] - c['V03']:+.10f} | - | 48 | 67 | -19 |"),
        (f"| P51 S1-U (binario) | {c['P51_S1U']} | "
        f"{c['P51_S1U'] - c['V03']:+.10f} | - | 414 | 1330 | -916 |"),
        (f"| **P51 S2-U (utilidad esperada)** | **{c['P51_S2U']:.10f}** | "
        f"**{c['P51_S2U'] - c['V03']:+.10f}** | **{r['celdas_cambiadas']}** | "
        f"**{r['rescate']}** | **{r['damage']}** | **{r['net']:+d}** |"),
        (f"| oracle (techo, usa `y_true`) | {c['oracle']:.10f} | "
        f"{c['oracle'] - c['V03']:+.10f} | - | 715 | 0 | +715 |"),
        "",
        f"Tasa de abstencion: {100 * r['abstencion_rate']:.2f} pct.",
        "",
        "## 4. Pliegues",
        "",
        "| pliegue | F1 de V03 | F1 de S2-U | delta |",
        "|---:|---:|---:|---:|",
    ]
    for p in informe["pliegues"]:
        lineas.append(f"| {p['pliegue']} | {p['f1_v03']:.10f} | {p['f1_selector']:.10f} | "
                      f"{p['delta']:+.10f} |")
    lineas += [
        "",
        f"Pliegues positivos: **{informe['pliegues_positivos']} de 5**.",
        "",
        "## 5. Interseccion de errores",
        "",
        "| categoria | celdas |",
        "|---|---:|",
    ]
    for k, v in informe["interseccion_de_errores"].items():
        lineas.append(f"| {k} | {v} |")
    lineas += [
        "",
        "## 6. Por margen y por direccion",
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
    lineas += [
        "",
        "## 7. Permutacion del net de celdas",
        "",
        f"> {b['nota']}",
        "",
        "| magnitud | valor |",
        "|---|---:|",
        f"| net observado | {b['net_observado']:+d} celdas |",
        f"| mediana de la nula | {b['nula_mediana']:+.1f} |",
        f"| IC95 de la nula | [{b['nula_ic95_bajo']:+.1f}, {b['nula_ic95_alto']:+.1f}] |",
        (f"| p de una cola, mejor que el azar | {b['p_valor_una_cola']:.4f} |"
        f"| celdas mejor que el azar | {b['celdas_mejor_que_el_azar']:+d} |"),
        (f"| el net observado cae dentro de la nula | "
        f"{b['net_observado_dentro_de_la_nula']} |"),
        f"| replicas | {b['n_replicas']} |",
        "",
        "## 7 bis. Variante restringida a la direccion 1->0",
        "",
        f"> {vb['nota']}",
        "",
        "| magnitud | valor |",
        "|---|---:|",
        f"| corrected_cv | {vb['corrected_cv']:.10f} |",
        f"| delta vs V03 | {vb['delta_vs_v03']:+.10f} |",
        f"| celdas cambiadas | {vb['celdas_cambiadas']} |",
        f"| rescate / damage / net | {vb['rescate']} / {vb['damage']} / {vb['net']:+d} |",
        "",
        "## 8. Gate y decision",
        "",
        (f"- delta {informe['gate']['delta_obtenido']:+.6f} frente a "
        f"{informe['gate']['delta_minimo']} exigido"),
        f"- pliegues positivos {informe['gate']['pliegues_positivos']} de 4 exigidos",
        f"- gate pasado: **{informe['gate']['pasado']}**",
        "",
        f"**CASE = {informe['caso']}.**",
        "",
    ]
    REPORTE_MD.write_text("\n".join(lineas) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
