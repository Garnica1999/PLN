"""V50-01 (P50): baselines y selector de canal por label-cell, con cross-fitting.

Que resuelve
------------
P50-00 dejo el techo del oracle medido: **+0,2098 de macro-F1**, unas 70 veces el
margen de 0,003 que la especificacion fija. La oportunidad es enorme. Falta la
otra mitad: ?se puede identificar, con variables observables en inferencia, que
canal conviene escuchar en cada celda?

La evidencia previa es mala y hay que ponerla a prueba. V18-01, que es el caso
degenerado de esta hipotesis, dio veredicto negativo con AUC 0,6181. Y la tabla
de fiabilidad de P50-00 muestra que **19 de los 20 canales tienen accuracy inferior
a 0,5 sobre el conjunto de desacuerdo**, es decir, sobre exactamente las celdas
donde un selector los usaria. El unico que supera 0,5 es el propio V03 con otro
umbral, que no es un canal alternativo sino la accion de abstenerse.

Metodo
------
1. Baselines de diagnostico: B0 V03, B1 votacion mayoritaria de los 20 canales,
   B2 canal de mayor fiabilidad global, B3 V03 salvo que la mayoria discrepe.
2. S1, selector de correccion: 21 clasificadores binarios, uno por accion, cada
   uno estima P(esa accion acierta | variables observables). Se elige el argmax,
   de modo que la accion KEEP_V03 esta siempre disponible y el selector puede
   abstenerse. No hay umbral que ajustar.
3. Cross-fitting con los pliegues canonicos de TOXIC, agrupados por grupo, de modo
   que ninguna label-cell de un tweet aparece a la vez en entrenamiento y en
   validacion.
4. Todo lo que se reporta es out-of-fold.

Reglas que el selector respeta
------------------------------
- El selector **nunca** esta obligado a cambiar V03: KEEP_V03 es una accion mas.
- No entra como variable ni ``y_true``, ni la correccion de V03, ni la identidad
  del canal correcto. Solo salidas de V03, salidas de los canales y la identidad
  de la etiqueta.
- El modelo es ``LogisticRegression``: pequeno, transparente y suficiente. No
  se usan XGBoost, LightGBM, redes ni apilamientos.

Que NO hace
-----------
- No genera ninguna submission.
- No consulta Kaggle.
- No modifica V03, ni el holdout, ni el test.

Uso
---
    /home/carlos/venv/bin/python experiments/v50_01_selector.py
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
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

FEATURES = PROJECT_ROOT / "features" / "p50_celda_features.npz"
ETIQUETAS = ("HS", "TR", "AG")
ACCION_KEEP = 0
N_FOLD = 5

REPORTE_JSON = PROJECT_ROOT / "reports" / "p50_selector_results.json"
REPORTE_MD = PROJECT_ROOT / "reports" / "p50_selector_analysis.md"
CSV_RESULTADOS = PROJECT_ROOT / "reports" / "p50_selector_results.csv"


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[V50-01] {mensaje}", flush=True)


def f1_macro(y_true: np.ndarray, pred: np.ndarray) -> float:
    """Macro-F1 multietiqueta: la media de los F1 por etiqueta.

    Aplastar las celdas con ``ravel`` NO sirve, porque sklearn veria un unico
    problema binario sobre 4449 muestras y devolveria el F1 agregado en vez del
    macro-F1 multietiqueta. El valor canonico ``corrected_cv`` de V03 es 0,7772114525.

    Args:
        y_true: etiquetas verdaderas, forma ``(n_filas, 3)``.
        pred: etiquetas predichas, forma ``(n_filas, 3)``.

    Returns:
        La media de los tres F1 por etiqueta.
    """
    return float(f1_score(y_true, pred, average="macro"))


def partir_celda(datos: np.lib.npyio.NpzFile) -> tuple[np.ndarray, ...]:
    """Reordena la matriz de celdas de filas a filas y celdas a columnas.

    Las features se construyeron recorriendo filas y luego etiquetas, asi que la
    celda ``(i, j)`` esta en la fila ``3 * i + j``. Aqui se aplana a las formas que
    necesitan los clasificadores.

    Args:
        datos: diccionario de features persistido por P50-00.

    Returns:
        Tupla con ``X``, ``y``, ``acciones``, ``pliegue``, ``ids`` y ``nombres``.
    """
    X = datos["X"]
    n_canales = datos["canal_pred"].shape[0]
    n_acciones = n_canales + 1
    y = np.asarray(datos["y_true"], dtype=np.int64).ravel()
    v03 = np.asarray(datos["v03_pred"], dtype=np.int64).ravel()
    canales = np.asarray(datos["canal_pred"], dtype=np.int64)
    acciones = np.empty((n_acciones, len(y)), dtype=np.int64)
    acciones[ACCION_KEEP] = v03
    for c in range(n_canales):
        acciones[c + 1] = np.transpose(canales[c]).ravel()
    pliegue = np.repeat(np.asarray(datos["pliegue"], dtype=np.int64), 3)
    ids = np.repeat(np.asarray(datos["ids"], dtype=np.int64), 3)
    nombres = [str(x) for x in datos["nombres"]]
    return X, y, acciones, pliegue, ids, nombres


def columnas_accion(X: np.ndarray, columnas: list[str], accion: int,
                    nombres: list[str]) -> np.ndarray:
    """Construye la matriz de variables de un clasificador de accion.

    Combina las variables compartidas de la celda con las dos variables propias
    de la accion: su probabilidad y su prediccion.

    Args:
        X: matriz completa de variables por celda.
        columnas: nombres de las columnas de ``X``.
        accion: indice de la accion, 0 para KEEP_V03.
        nombres: nombres de los canales.

    Returns:
        Matriz de variables para ese clasificador.
    """
    compartidas = [i for i, c in enumerate(columnas)
                   if not c.startswith("proba_") and not c.startswith("pred_")]
    if accion == ACCION_KEEP:
        propias = [columnas.index("v03_proba"), columnas.index("v03_pred")]
    else:
        nombre = f"proba_{nombres[accion - 1]}"
        propias = [columnas.index(nombre), columnas.index(f"pred_{nombres[accion - 1]}")]
    return X[:, compartidas + propias]


def main() -> None:
    """Ejecuta los baselines y el selector cross-fitted.

    Returns:
        None.
    """
    inicio = time.time()
    log("P50-01 baselines y selector contextual. 0 submissions, 0 consultas a Kaggle")
    datos = np.load(FEATURES, allow_pickle=True)
    X, y, acciones, pliegue, _ids, nombres = partir_celda(datos)
    columnas = [str(c) for c in datos["columnas"]]
    n_acciones, n_celdas = acciones.shape
    log(f"   celdas {n_celdas} | acciones {n_acciones} | variables {X.shape[1]}")
    log(f"   pliegues: {[int((pliegue == v).sum()) for v in range(N_FOLD)]}")

    # Reconstruye las formas 2D para puntuar con la metrica canonica.
    n_filas = len(datos["ids"])
    y2d = y.reshape(n_filas, 3)
    acc2d = acciones.reshape(n_acciones, n_filas, 3)

    v03_2d = acc2d[ACCION_KEEP]
    err_v03 = v03_2d != y2d
    n_celdas_error = int(err_v03.sum())
    f1_v03 = f1_macro(y2d, v03_2d)
    log(f"   B0 = V03: macro-F1 {f1_v03:.10f} | celdas de error {n_celdas_error}")

    # ------------------------------------------------------------- B1 a B3
    log("2/7 baselines de diagnostico")
    votos = acc2d[1:].sum(axis=0)
    mayoria = (votos * 2 > (n_acciones - 1)).astype(np.int64)
    canales_2d = acc2d[1:].reshape(n_acciones - 1, n_filas, 3)

    fiabilidad = []
    for c in range(n_acciones - 1):
        pred_c = canales_2d[c]
        desac = pred_c != v03_2d
        fiab = float((pred_c == y2d)[desac].mean()) if desac.any() else float("nan")
        fiab += float((pred_c == y2d).mean())
        fiabilidad.append(fiab)
    mejor_canal = int(np.argmax(fiabilidad))
    b2 = canales_2d[mejor_canal]

    filas_base = []
    for nombre, pred in (("B0_V03", v03_2d), ("B1_mayoria_canales", mayoria),
                         (f"B2_canal_mas_fiable_{nombres[mejor_canal]}", b2),
                         ("B3_V03_salvo_mayoria", np.where(mayoria != v03_2d, mayoria,
                                                            v03_2d))):
        f1 = f1_macro(y2d, pred)
        resc = int((err_v03 & (pred == y2d)).sum())
        dany = int(((~err_v03) & (pred != y2d)).sum())
        filas_base.append({
            "selector": nombre, "corrected_cv": round(f1, 10),
            "delta_vs_v03": round(f1 - f1_v03, 10),
            "celdas_cambiadas": int((pred != v03_2d).sum()),
            "rescate": resc, "damage": dany, "net": resc - dany,
        })
        log(f"   {nombre:34} macro-F1 {f1:.10f} | delta {f1 - f1_v03:+.10f} | "
            f"cambiadas {int((pred != v03_2d).sum()):4d} | resc {resc:4d} "
            f"dano {dany:4d} net {resc - dany:+5d}")

    # ------------------------------------------------------------- S1
    log("3/7 S1 selector de correccion, 21 clasificadores con cross-fitting")
    log(f"   accion KEEP_V03 disponible siempre (indice {ACCION_KEEP})")
    probabilidades = np.zeros((n_acciones, n_celdas), dtype=np.float64)
    for v in range(N_FOLD):
        ent = pliegue != v
        val = ~ent
        for a in range(n_acciones):
            objetivo = (acciones[a] == y).astype(int)
            Xa = columnas_accion(X, columnas, a, nombres)
            if objetivo[ent].sum() < 2 or objetivo[ent].sum() == ent.sum():
                probabilidades[a, val] = float(objetivo[ent].mean())
                continue
            modelo = LogisticRegression(C=1.0, max_iter=1000, solver="liblinear",
                                        random_state=42)
            modelo.fit(Xa[ent], objetivo[ent])
            probabilidades[a, val] = modelo.predict_proba(Xa[val])[:, 1]
        log(f"   pliegue {v}: entrenado {int(ent.sum())} celdas, "
            f"validado {int(val.sum())}")

    # AUC out-of-fold: mide si la accion correcta es identificable con variables
    # observables. Se calcula sobre las probabilidades ya acumuladas fuera de
    # pliegue, nunca sobre el ajuste del propio entrenamiento.
    log("   AUC calculado sobre las predicciones OUT-OFOLD de los 5 pliegues")
    auc_por_accion = np.full(n_acciones, np.nan)
    for a in range(n_acciones):
        objetivo = (acciones[a] == y).astype(int)
        if len(np.unique(objetivo)) == 2:
            auc_por_accion[a] = roc_auc_score(objetivo, probabilidades[a])

    # Seleccion out-of-fold: argmax sobre las 21 acciones.
    accion_elegida = probabilidades.argmax(axis=0)
    confianza = probabilidades.max(axis=0)
    pred_s1_1d = acciones[accion_elegida, np.arange(n_celdas)]
    pred_s1 = pred_s1_1d.reshape(n_filas, 3)
    accionada = accion_elegida != ACCION_KEEP
    f1_s1 = f1_macro(y2d, pred_s1)
    resc_s1 = int((err_v03 & (pred_s1 == y2d)).sum())
    dany_s1 = int(((~err_v03) & (pred_s1 != y2d)).sum())
    log(f"   S1: macro-F1 {f1_s1:.10f} | delta {f1_s1 - f1_v03:+.10f}")
    log(f"   S1: celdas cambiadas {int(accionada.sum())} | rescate {resc_s1} "
        f"damage {dany_s1} net {resc_s1 - dany_s1:+d}")
    log(f"   S1: tasa de abstencion {100 * (1 - accionada.mean()):.2f} pct")

    filas_fiab = json.loads(
        (REPORTE_JSON.parent / "p50_gap_and_oracle.json").read_text(encoding="utf-8")
    )["fiabilidad_por_canal"]
    # Que accion elige S1 de verdad. La regla es argmax de P(acierta), y por eso
    # tiende a premiar al canal mas predecible, que no es el mas utile.
    frecuentes = pd.Series(accion_elegida).value_counts()
    top = frecuentes.head(6)
    log("   acciones que S1 elige de verdad:")
    for accion_id, cuenta in top.items():
        etiqueta_accion = ("KEEP_V03" if accion_id == ACCION_KEEP else nombres[accion_id - 1])
        net_accion = (0 if accion_id == ACCION_KEEP
                      else int([f["net"] for f in filas_fiab][accion_id - 1]))
        log(f"      accion {accion_id:2d} {etiqueta_accion:34} {int(cuenta):4d} veces "
            f"| net real del canal {net_accion:+d}")
    reparto_acciones = [
        {"accion": ("KEEP_V03" if a == ACCION_KEEP else nombres[a - 1]),
         "n_elegida": int(frecuentes.get(a, 0)),
         "net_real_del_canal": (0 if a == ACCION_KEEP
                                else int([f["net"] for f in filas_fiab][a - 1]))}
        for a in range(n_acciones)
    ]

    # ------------------------------------------- 4. AUC: la cifra que decide
    log("4/7 AUC out-of-sample de cada clasificador de accion (identificabilidad)")
    orden_auc = np.argsort(-np.nan_to_num(auc_por_accion, nan=-1))
    for a in orden_auc:
        etiqueta_accion = ("KEEP_V03" if a == ACCION_KEEP else nombres[a - 1])
        log(f"   accion {a:2d} {etiqueta_accion:34} AUC {auc_por_accion[a]:.4f}")
    mejor_auc = float(np.nanmax(auc_por_accion))
    log(f"   MEJOR AUC = {mejor_auc:.4f} | el objetivo es que sea identificable la "
        f"accion correcta, luego interesa la accuracy del ACCION ELEGIDA")

    # El AUC de un canal mide lo predecible de SU salida, no el valor de actuar
    # sobre ella. Se contrasta con el net real de cada canal para ver si la
    # confianza del selector apunta a los canales que conviene escuchar.
    net_por_canal = np.array([f["net"] for f in filas_fiab], dtype=float)
    auc_canales = auc_por_accion[1:]
    mask = ~np.isnan(auc_canales)
    rho, pval = spearmanr(auc_canales[mask], net_por_canal[mask])
    orden = np.argsort(-auc_canales)
    log(f"   Spearman entre el AUC de cada canal y su net real: rho {rho:+.4f} "
        f"p {pval:.4f} sobre {int(mask.sum())} canales")
    log(f"   canal con MAS AUC : {nombres[int(orden[0])]} AUC "
        f"{auc_canales[orden[0]]:.4f} net {net_por_canal[orden[0]]:+.0f}")
    log(f"   canal con MENOS AUC: {nombres[int(orden[-1])]} AUC "
        f"{auc_canales[orden[-1]]:.4f} net {net_por_canal[orden[-1]]:+.0f}")
    correlacion_auc_net = {"spearman_rho": round(float(rho), 6),
                           "p_valor": round(float(pval), 6),
                           "n_canales": int(mask.sum())}

    # AUC de la accion que S1 elige de verdad, que es la cifra relevante.
    correcto_elegido = (acciones[accion_elegida, np.arange(n_celdas)] == y)
    log(f"   la accion elegida acierta {int(correcto_elegido.sum())} de {n_celdas} "
        f"celdas ({100 * correcto_elegido.mean():.2f} pct)")

    # ------------------------------------------------- 5. metricas por pliegue
    log("5/7 corrected_cv por pliegue")
    pliegues_f1 = []
    for v in range(N_FOLD):
        sel = pliegue == v
        filas_sel = (sel.reshape(n_filas, 3)).any(axis=1)
        pv = v03_2d[filas_sel]
        ps = pred_s1[filas_sel]
        yv = y2d[filas_sel]
        f1_v = f1_macro(yv, pv)
        f1_s = f1_macro(yv, ps)
        pliegues_f1.append({
            "pliegue": v, "n_filas": int(filas_sel.sum()),
            "f1_v03": round(f1_v, 10), "f1_selector": round(f1_s, 10),
            "delta": round(f1_s - f1_v, 10),
        })
        log(f"   pliegue {v}: V03 {f1_v:.10f} | selector {f1_s:.10f} | "
            f"delta {f1_s - f1_v:+.10f}")
    deltas = [p["delta"] for p in pliegues_f1]
    pliegues_positivos = sum(1 for d in deltas if d > 0)
    log(f"   pliegues positivos {pliegues_positivos} de {N_FOLD} | "
        f"delta medio {np.mean(deltas):+.10f}")

    # ------------------------------------------------ 6. interseccion de errores
    log("6/7 interseccion de errores y analisis por margen")
    ok_v03 = ~err_v03
    interseccion = {
        "ambos_correctos": int((ok_v03 & (pred_s1 == y2d)).sum()),
        "ambos_malos": int((err_v03 & (pred_s1 != y2d)).sum()),
        "v03_mal_selector_bien": int((err_v03 & (pred_s1 == y2d)).sum()),
        "v03_bien_selector_mal": int((ok_v03 & (pred_s1 != y2d)).sum()),
    }
    for k, v in interseccion.items():
        log(f"   {k:26} {v}")
    log(f"   verificacion: {interseccion['ambos_correctos']} + "
        f"{interseccion['v03_bien_selector_mal']} + "
        f"{interseccion['v03_mal_selector_bien']} + {interseccion['ambos_malos']} "
        f"= {sum(interseccion.values())} de {n_filas * 3}")

    v03_margen = np.asarray(datos["v03_margen"], dtype=np.float64)
    margen_filas = []
    for lo, hi, nombre_bin in ((0.0, 0.10, "[0,0.10)"), (0.10, 0.20, "[0.10,0.20)"),
                               (0.20, 1.01, "[0.20,1]")):
        celdas = (v03_margen >= lo) & (v03_margen < hi)
        n_e = int((celdas & err_v03).sum())
        n_c = int((celdas & (pred_s1 != v03_2d)).sum())
        n_r = int((celdas & err_v03 & (pred_s1 == y2d)).sum())
        n_d = int((celdas & ok_v03 & (pred_s1 != y2d)).sum())
        margen_filas.append({
            "bin": nombre_bin, "n_celdas": int(celdas.sum()), "n_errores_v03": n_e,
            "n_cambiadas": n_c, "rescate": n_r, "damage": n_d, "net": n_r - n_d,
        })
        log(f"   margen {nombre_bin:12} celdas {int(celdas.sum()):5d} | errores "
            f"{n_e:4d} | cambiadas {n_c:4d} | resc {n_r:4d} dano {n_d:4d} "
            f"net {n_r - n_d:+5d}")

    # -------------------------------------------------- 7. bins de confianza
    filas_conf = []
    for lo, hi, nombre_bin in ((0.0, 0.45, "[0,0.45)"), (0.45, 0.55, "[0.45,0.55)"),
                               (0.55, 0.70, "[0.55,0.70)"), (0.70, 1.01, "[0.70,1]")):
        sel = (confianza >= lo) & (confianza < hi)
        if not sel.any():
            continue
        filas_conf.append({
            "bin": nombre_bin, "n": int(sel.sum()),
            "frac_accionada": round(float(accionada[sel].mean()), 4),
            "accuracy_de_la_accion_elegida": round(float(correcto_elegido[sel].mean()), 4),
        })
        log(f"   confianza {nombre_bin:12} n={int(sel.sum()):5d} | acciona "
            f"{100 * accionada[sel].mean():5.1f} pct | acierta "
            f"{100 * correcto_elegido[sel].mean():5.1f} pct")

    # ------------------------------------------------------------- veredicto
    delta_s1 = f1_s1 - f1_v03
    pasa_delta = delta_s1 >= 0.003
    pasa_pliegues = pliegues_positivos >= 4
    if delta_s1 <= 0:
        caso = "B"
    elif pasa_delta and pasa_pliegues:
        caso = "C"
    else:
        caso = "B"
    log(f"   GATE 22: delta {delta_s1:+.6f} | pliegues positivos "
        f"{pliegues_positivos}/{N_FOLD} | pasa {pasa_delta and pasa_pliegues}")
    log(f"   CASE = {caso}")

    filas_res = []
    for f in filas_base:
        filas_res.append({
            "selector": f["selector"], "corrected_cv": f["corrected_cv"],
            "delta_vs_v03": f["delta_vs_v03"], "celdas_cambiadas": f["celdas_cambiadas"],
            "rescate": f["rescate"], "damage": f["damage"], "net": f["net"],
            "auc_maxima": "", "pliegues_positivos": "", "bootstrap_ic95": "",
        })
    filas_res.append({
        "selector": "S1_correccion", "corrected_cv": round(f1_s1, 10),
        "delta_vs_v03": round(delta_s1, 10), "celdas_cambiadas": int(accionada.sum()),
        "rescate": resc_s1, "damage": dany_s1, "net": resc_s1 - dany_s1,
        "auc_maxima": round(mejor_auc, 4), "pliegues_positivos": pliegues_positivos,
        "bootstrap_ic95": "",
    })
    pd.DataFrame(filas_res).to_csv(CSV_RESULTADOS, index=False, encoding="utf-8")

    informe = {
        "familia": "P50_CONTEXTUAL_CHANNEL_SELECTION",
        "experimento": "V50-01 (baselines y selector contextual cross-fitted)",
        "fecha": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "pregunta": ("dado un techo del oracle de +0,2098, ?se puede identificar "
                     "por celda que canal escuchar?"),
        "configuracion": {
            "acciones": n_acciones, "incluye_keep_v03": True,
            "modelo": "LogisticRegression(C=1.0, liblinear)",
            "variables": int(X.shape[1]), "variables_por_accion": 15,
            "cross_fitting": f"{N_FOLD} pliegues canonicos de TOXIC, agrupados",
            "auc_maxima": round(mejor_auc, 4),
            "correlacion_auc_con_net": correlacion_auc_net,
            "reparto_de_acciones_elegidas": reparto_acciones,
        },
        "v03": {"corrected_cv": f1_v03, "celdas_de_error": n_celdas_error},
        "baselines": filas_base,
        "s1": {
            "corrected_cv": round(f1_s1, 10),
            "delta_vs_v03": round(delta_s1, 10),
            "celdas_cambiadas": int(accionada.sum()),
            "tasa_de_abstencion": round(float(1 - accionada.mean()), 6),
            "rescate": resc_s1, "damage": dany_s1, "net": resc_s1 - dany_s1,
            "accuracy_de_la_accion_elegida": round(float(correcto_elegido.mean()), 6),
            "auc_por_accion": [
                {"accion": (ACCION_KEEP if a == ACCION_KEEP else nombres[a - 1]),
                 "auc": (None if np.isnan(auc_por_accion[a])
                         else round(float(auc_por_accion[a]), 6))}
                for a in range(n_acciones)
            ],
        },
        "pliegues": pliegues_f1,
        "pliegues_positivos": pliegues_positivos,
        "interseccion_de_errores": interseccion,
        "margen": margen_filas,
        "confianza": filas_conf,
        "gate": {"delta_minimo": 0.003, "delta_obtenido": round(delta_s1, 10),
                 "pliegues_positivos_requeridos": 4,
                 "pasado": bool(pasa_delta and pasa_pliegues)},
        "caso": caso,
        "gastos": {"submissions": 0, "consultas_kaggle": 0, "uso_holdout": 0,
                   "uso_test": 0, "segundos": round(time.time() - inicio, 1)},
    }
    REPORTE_JSON.write_text(json.dumps(informe, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8")
    _md(informe)
    log("   escrito reports/p50_selector_results.{json,csv} y p50_selector_analysis.md")
    log(f"VEREDICTO: CASE={caso} | delta={delta_s1:+.6f} | mejor_auc={mejor_auc:.4f} | "
        f"0 submissions, {round(time.time() - inicio, 1)} s")


def _md(informe: dict[str, Any]) -> None:
    """Escribe el informe del selector.

    Args:
        informe: diccionario con los resultados.

    Returns:
        None.
    """
    s1 = informe["s1"]
    correlacion = informe["configuracion"]["correlacion_auc_con_net"]
    base = informe["baselines"]
    pl = informe["pliegues"]
    inter = informe["interseccion_de_errores"]
    lineas = [
        "# P50 - Analisis del selector de canal por label-cell",
        "",
        f"Generado por `experiments/v50_01_selector.py` ({informe['experimento']}).",
        "",
        f"Fecha {informe['fecha']}. **0 submissions, 0 consultas a Kaggle.**",
        "",
        f"> {informe['pregunta']}",
        "",
        "## 1. Configuracion",
        "",
        (f"- acciones: **{informe['configuracion']['acciones']}**, con `KEEP_V03` siempre "
        "disponible, luego el selector puede abstenerse"),
        (f"- modelo: `{informe['configuracion']['modelo']}`, "
        f"{informe['configuracion']['variables_por_accion']} variables por accion"),
        f"- cross-fitting: {informe['configuracion']['cross_fitting']}",
        "- ninguna variable es `y_true`, la correccion de V03 o el canal correcto",
        "",
        "## 2. Baselines de diagnostico",
        "",
        "| selector | corrected_cv | delta | cambiadas | rescate | damage | net |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for b in base:
        lineas.append(f"| {b['selector']} | {b['corrected_cv']:.10f} | "
                      f"{b['delta_vs_v03']:+.10f} | {b['celdas_cambiadas']} | "
                      f"{b['rescate']} | {b['damage']} | {b['net']:+d} |")
    lineas += [
        "",
        "## 3. S1, selector de correccion",
        "",
        "| magnitud | valor |",
        "|---|---:|",
        f"| corrected_cv | **{s1['corrected_cv']:.10f}** |",
        f"| delta vs V03 | **{s1['delta_vs_v03']:+.10f}** |",
        f"| celdas cambiadas | {s1['celdas_cambiadas']} |",
        f"| tasa de abstencion | {100 * s1['tasa_de_abstencion']:.2f} pct |",
        f"| rescate | {s1['rescate']} |",
        f"| damage | {s1['damage']} |",
        f"| net | {s1['net']:+d} |",
        f"| accuracy de la accion elegida | {s1['accuracy_de_la_accion_elegida']:.4f} |",
        (f"| **AUC maxima de los 21 clasificadores** | "
        f"**{informe['configuracion']['auc_maxima']:.4f}** |"),
        "",
        (f"Correlacion de Spearman entre el AUC de cada canal y su net real: "
         f"**{correlacion['spearman_rho']:+.4f}** (p = {correlacion['p_valor']:.4f}, "
         f"n = {correlacion['n_canales']} canales). El AUC mide lo predecible de la "
         "salida de un canal, no el valor de actuar sobre ella."),
        ("El AUC importa mas que el delta: mide si la accion correcta es "
        "**identificable** con variables observables. Un AUC de 0,50 significa que "
        "elegir es indistinguible de tirar una moneda."),
        "",
        "### AUC por accion",
        "",
        "| accion | AUC |",
        "|---|---:|",
    ]
    for a in sorted(s1["auc_por_accion"], key=lambda d: -(d["auc"] or 0)):
        valor = "n/d" if a["auc"] is None else f"{a['auc']:.4f}"
        lineas.append(f"| `{a['accion']}` | {valor} |")
    lineas += [
        "",
        "## 4. Pliegues",
        "",
        "| pliegue | filas | F1 de V03 | F1 del selector | delta |",
        "|---:|---:|---:|---:|---:|",
    ]
    for p in pl:
        lineas.append(f"| {p['pliegue']} | {p['n_filas']} | {p['f1_v03']:.10f} | "
                      f"{p['f1_selector']:.10f} | {p['delta']:+.10f} |")
    lineas += [
        "",
        f"Pliegues con delta positivo: **{informe['pliegues_positivos']} de 5**.",
        "",
        "## 5. Interseccion de errores",
        "",
        "| categoria | celdas |",
        "|---|---:|",
        f"| ambos correctos | {inter['ambos_correctos']} |",
        f"| ambos incorrectos | {inter['ambos_malos']} |",
        f"| V03 mal, selector bien (rescate) | {inter['v03_mal_selector_bien']} |",
        f"| V03 bien, selector mal (damage) | {inter['v03_bien_selector_mal']} |",
        "",
        "## 6. Por margen de V03",
        "",
        "| bin | celdas | errores | cambiadas | rescate | damage | net |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for m in informe["margen"]:
        lineas.append(f"| {m['bin']} | {m['n_celdas']} | {m['n_errores_v03']} | "
                      f"{m['n_cambiadas']} | {m['rescate']} | {m['damage']} | {m['net']:+d} |")
    lineas += [
        "",
        "## 7. Por confianza de la accion elegida",
        "",
        "| bin | n | fraccion accionada | accuracy de la accion |",
        "|---|---:|---:|---:|",
    ]
    for c in informe["confianza"]:
        lineas.append(f"| {c['bin']} | {c['n']} | {c['frac_accionada']:.4f} | "
                      f"{c['accuracy_de_la_accion_elegida']:.4f} |")
    g = informe["gate"]
    lineas += [
        "",
        "## 8. Gate y decision",
        "",
        f"- delta obtenido {g['delta_obtenido']:+.6f} frente a {g['delta_minimo']} exigido",
        (f"- pliegues positivos {informe['pliegues_positivos']} de "
        f"{g['pliegues_positivos_requeridos']} exigidos"),
        f"- gate pasado: **{g['pasado']}**",
        "",
        (f"**CASE = {informe['caso']}.** Ver `reports/p50_selector_gap_audit.md` para la "
        "auditoria de brecha y el techo del oracle."),
        "",
    ]
    REPORTE_MD.write_text("\n".join(lineas) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
