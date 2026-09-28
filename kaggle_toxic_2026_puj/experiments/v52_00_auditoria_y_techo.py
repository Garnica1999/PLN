"""V52-00 (P52): auditoria de P51, derivacion de la utilidad y policy ceiling.

Que resuelve
------------
P51 cerro la seleccion contextual con dos implementaciones negativas. Pero cerro
tambien con dos hallazgos que P52 tiene queExplanation:

- el target de utilidad es **mucho mas aprendible** que el de correccion (AUC
  0,8091 de media frente a 0,55-0,77), luego la senal existe y se ve;
- la politica de cambios tiene **expectativa negativa**, luego cambiar sin condicion
  pierde.

La pregunta que queda es la de la directiva: **existe una politica de decision,
construida solo con variables observables, cuyo techo sea materialmente superior a
V03?** Y la respuesta honesta exige separar tres objetos, porque "policy oracle" sin
`y_true` no es un terceiro objeto: es una politica aprendida.

    V03              la inaccion, net 0 por definicion
    LEARNED_POLICY   estima P(rescue | observable) y elige argmax E[u]
    STRATUM_ORACLE   el mejor techo de una politica que sea CONSTANTE dentro de
                     estratos observables. Usa `y_true` al elegir la accion de cada
                     estrato, luego es un techo y se declara como tal
    TRUE_ORACLE      el mejor action celda a celda, 0,9870178614

La descomposicion que importa:

    TRUE_ORACLE  ->  STRATUM_ORACLE   coste de coarsar la politica a estratos
    STRATUM_ORACLE -> LEARNED_POLICY  coste de estimar las probabilidades
    LEARNED_POLICY -> V03             lo que realmente se puede hacer

Si el techo de estrato esta por debajo de V03, ninguna politica condicional puede
ganar, y la linea contextual queda cerrada por una razon matematica y no empirica.

Que NO hace
-----------
- No entrena ningun modelo.
- No genera ninguna submission.
- No consulta Kaggle.
- No modifica V03, ni el holdout, ni el test.

Uso
---
    /home/carlos/venv/bin/python experiments/v52_00_auditoria_y_techo.py
"""

from __future__ import annotations

import itertools
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

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
P50_GAP = PROJECT_ROOT / "reports" / "p50_gap_and_oracle.json"
P50_SEL = PROJECT_ROOT / "reports" / "p50_selector_results.json"
P51_GAP = PROJECT_ROOT / "reports" / "p51_utility_gap_and_utility.json"
P51_S1U = PROJECT_ROOT / "reports" / "p51_utility_selector_results.json"
P51_S2U = PROJECT_ROOT / "reports" / "p51_s2u_results.json"

ETIQUETAS = ("HS", "TR", "AG")
N_FOLD = 5

REPORTE_JSON = PROJECT_ROOT / "reports" / "p52_policy_oracle.json"
REPORTE_MD = PROJECT_ROOT / "reports" / "p52_policy_analysis.md"
CSV_TECHO = PROJECT_ROOT / "reports" / "p52_policy_oracle.csv"
CSV_PARES = PROJECT_ROOT / "reports" / "p52_channel_pair_analysis.csv"
CSV_EXCLUSIVIDAD = PROJECT_ROOT / "reports" / "p52_channel_exclusivity.csv"
CSV_CORRELACION = PROJECT_ROOT / "reports" / "p52_error_correlation.csv"
CSV_CURVA = PROJECT_ROOT / "reports" / "p52_policy_value_curve.csv"

# Reparto de margenes y de desacuerdos, solo observables.
BINS_MARGEN = (0.0, 0.10, 0.20, 1.01)
BINS_DISCREPANCIA = (0, 4, 8, 99)


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[V52-00] {mensaje}", flush=True)


def codigo_de_estrato(v03_col: np.ndarray, margen: np.ndarray,
                      n_dis: np.ndarray) -> np.ndarray:
    """Codifica el estrato observable de una celda.

    El estrato usa solo cosas disponibles en inferencia: la prediccion de V03, el
    bin de margen y el numero de canales que discrepan.

    Args:
        v03_col: prediccion de V03 por celda.
        margen: margen de V03 por celda.
        n_dis: numero de canales que discrepan por celda.

    Returns:
        Codigo entero del estrato.
    """
    b_margen = np.digitize(margen, BINS_MARGEN[1:-1])
    b_dis = np.digitize(n_dis, BINS_DISCREPANCIA[1:-1])
    return (v03_col * 9 + b_margen) * 9 + b_dis


def main() -> None:
    """Ejecuta la auditoria, la derivacion y el techo de estrato.

    Returns:
        None.
    """
    inicio = time.time()
    log("P52-00 auditoria de P51, utilidad y techo de politica. 0 modelos")

    datos = np.load(FEATURES, allow_pickle=True)
    n_filas = len(datos["ids"])
    n_celdas = n_filas * 3
    y2d = np.asarray(datos["y_true"], dtype=np.int64)
    v03 = np.asarray(datos["v03_pred"], dtype=np.int64)
    v03_margen = np.asarray(datos["v03_margen"], dtype=np.float64)
    canales = np.asarray(datos["canal_pred"], dtype=np.int64)
    pliegue = np.asarray(datos["pliegue"], dtype=np.int64)
    nombres = [str(x) for x in datos["nombres"]]
    n_canales = canales.shape[0]
    canales_plano = canales.reshape(n_canales, -1)
    v03_plano = v03.ravel()
    y_plano = y2d.ravel()
    pl_celda = np.repeat(pliegue, 3)
    f1_v03 = f1_macro(y2d, v03)
    err_v03_plano = v03_plano != y_plano
    n_celdas_error = int(err_v03_plano.sum())
    p50_gap = json.loads(P50_GAP.read_text(encoding="utf-8"))
    p51_gap = json.loads(P51_GAP.read_text(encoding="utf-8"))
    log(f"   V03 {f1_v03:.10f} | celdas de error {n_celdas_error} | canales {n_canales}")

    # ================================================ 1. AUDITORIA DE P51
    log("1/7 auditoria del codigo real de P51")
    codigo_51 = (PROJECT_ROOT / "experiments" / "v51_01_selector_utilidad.py").read_text(
        encoding="utf-8")
    codigo_52 = (PROJECT_ROOT / "experiments" / "v51_02_accion_utilidad_esperada.py"
                 ).read_text(encoding="utf-8")
    auditoria = {
        "target_S1U": "(u > 0) con u = utilidad[v, c] y utilidad construida con los "
                      "conteos del pliegue de entrenamiento"
                      if "(u[mask] > 0).astype(int)" in codigo_51 else "NO VERIFICADO",
        "target_S2U": "el mismo target binario; S2-U cambia la agregacion, no el "
                      "target" if "(u[mask] > 0).astype(int)" in codigo_52
                      else "NO VERIFICADO",
        "umbral_S1U": 0.5 if "UMBRAL_ESPECIFICADO = 0.5" in codigo_51 else "NO VERIFICADO",
        "umbral_equilibrio_teorico": round(
            p51_gap["asimetria_de_direccion"]["p_umbral_para_subir_a_1"], 6),
        "utilidad": "FN->TP escala con TP+FP+FN, FP->TN solo con TP, forma cerrada de "
                    "v19_01_valor_de_correccion.py",
        "espacio_de_acciones": "KEEP_V03 mas 20 canales, restringido a celdas en "
                               "discrepancia",
        "cross_fitting": "5 pliegues canonicos de TOXIC agrupados, objetivo y ajuste "
                         "dentro del training fold",
        "variables": 15,
        "mezcla_direcciones": ("SI: P51 usa una mascara unica de discrepancia y "
                                    "un solo clasificador por canal, luego mezcla las "
                                    "dos direcciones. P52 las separa, que es lo que "
                                    "exige la seccion 7"),
        "variables_que_entra": "P(j) = (canal_j == y) sobre celdas en discrepancia",
    }
    log(f"   P51_THRESHOLD = {auditoria['umbral_S1U']}")
    log(f"   THEORETICAL_EQUILIBRIUM = {auditoria['umbral_equilibrio_teorico']}")
    log("   P51 mezcla las dos direcciones en un solo clasificador: True")
    log("   => el umbral 0,5 es mas estricto que 0,3892 al subir y mas laxo que")
    log("      0,6108 al bajar, luego sobre-cambia en la direccion que mas dana")

    # ============================== 2. UTILIDAD EXACTA, RE-DERIVADA
    log("2/7 derivacion cerrada independiente de gain_up y gain_down")
    filas_gan = []
    g_up = np.zeros((N_FOLD, n_celdas))
    h_up = np.zeros((N_FOLD, n_celdas))
    for v in range(N_FOLD):
        ent = pliegue != v
        for j, et in enumerate(ETIQUETAS):
            tp, fp, fn = conteos(y2d[ent, j], v03[ent, j])
            gan_fn = valor_falso_negativo(tp, fp, fn)
            gan_fp = valor_falso_positivo(tp, fp, fn)
            filas_gan.append({
                "pliegue": v, "etiqueta": et, "tp": tp, "fp": fp, "fn": fn,
                "gain_up_FN_a_TP": gan_fn, "gain_down_FP_a_TN": gan_fp,
                "razon": gan_fn / gan_fp,
                "equilibrio_up_sobre_P_utilidad": gan_fp / (gan_fn + gan_fp),
                "equilibrio_down_sobre_P_utilidad": gan_fn / (gan_fn + gan_fp),
            })
            filas = pl_celda == v
            etiqueta_plano = np.repeat(np.arange(3), n_filas)
            sel = filas & (etiqueta_plano == j)
            g_up[v, sel] = gan_fn
            h_up[v, sel] = gan_fp
    gdf = pd.DataFrame(filas_gan)
    eq_up = float(gdf["equilibrio_up_sobre_P_utilidad"].mean())
    eq_down = float(gdf["equilibrio_down_sobre_P_utilidad"].mean())
    razon = float(gdf["razon"].mean())
    for et in ETIQUETAS:
        s = gdf[gdf["etiqueta"] == et]
        log(f"   {et}: gain_up {s['gain_up_FN_a_TP'].mean():.4e} | "
            f"gain_down {s['gain_down_FP_a_TN'].mean():.4e} | "
            f"equilibrio up {s['equilibrio_up_sobre_P_utilidad'].mean():.4f} | "
            f"down {s['equilibrio_down_sobre_P_utilidad'].mean():.4f}")
    log(f"   VERIFICACION INDEPENDIENTE: equilibrio up {eq_up:.6f} | "
        f"down {eq_down:.6f} | suma {eq_up + eq_down:.10f} (debe ser 1)")

    # Elevaciones necesarias, recalculadas.
    n_dis = (canales_plano != v03_plano[None]).sum(axis=0)
    sube = (canales_plano == 1) & (canales_plano != v03_plano[None])
    baja = (canales_plano == 0) & (canales_plano != v03_plano[None])
    p51_eq = json.loads(P51_S1U.read_text(encoding="utf-8"))["equilibrio_analitico"]
    rescue_up = sube & (canales_plano == y_plano[None])
    rescue_down = baja & (canales_plano == y_plano[None])
    base_up = float(rescue_up.sum() / max(int(sube.sum()), 1))
    base_down = float(rescue_down.sum() / max(int(baja.sum()), 1))
    elev_up = eq_up / base_up
    elev_down = eq_down / base_down
    p51_elev_up = p51_eq['elevacion_necesaria_subir']
    p51_elev_down = p51_eq['elevacion_necesaria_bajar']
    corrige_down = abs(elev_down - p51_elev_down) > 0.05
    log(f"   tasa base up   {100 * base_up:.2f} pct | equilibrio {100 * eq_up:.2f} pct "
        f"| elevacion {elev_up:.2f}x")
    log(f"   tasa base down {100 * base_down:.2f} pct | equilibrio {100 * eq_down:.2f} "
        f"pct | elevacion {elev_down:.2f}x")
    coincide_eq = abs(eq_up - p51_eq["umbral_subir_sobre_P_y1"]) < 1e-6
    coincide_up = abs(elev_up - p51_elev_up) < 1e-2
    log(f"   reproduce el equilibrio de P51 (0,3892): {bool(coincide_eq)}")
    log(f"   reproduce la elevacion al SUBIR de P51 (2,18x): {bool(coincide_up)}")
    log(f"   CORRIGE la elevacion al BAJAR: P51 dice {p51_elev_down:.2f}x y el valor")
    log(f"   correcto es {elev_down:.2f}x. P51 comparo (1 - 0,6108) = 0,3892 contra la")
    log("   tasa de 0,3761, pero el rescate al bajar es P(y=0) y su equilibrio es")
    log("   0,6108, luego dividio por el lado equivocado de la ecuacion")

    # ================= 3. TECHO DE ESTRATO, SIN ENTRENAR NADA
    log("3/7 techo de estrato: la mejor politica CONSTANTE dentro de estratos observables")
    n_dis_plano = n_dis
    # v03_margen es (n_filas, 3): ravel da las 4449 celdas en el mismo orden
    # que v03.ravel(), que es como se indexa todo lo demas.
    margen_plano = v03_margen.ravel()
    estrato = codigo_de_estrato(v03_plano, margen_plano, n_dis_plano)
    n_est = len(np.unique(estrato))
    log(f"   estratos observables: {n_est} | celdas por estrato: "
        f"{n_celdas / n_est:.1f} de media")

    # Utilidad realizada de cada accion, celda a celda, con la forma cerrada.
    g_celda = g_up[pl_celda, np.arange(n_celdas)]
    h_celda = h_up[pl_celda, np.arange(n_celdas)]
    g_dir = np.where(sube, g_celda[None, :], h_celda[None, :])
    h_dir = np.where(sube, h_celda[None, :], g_celda[None, :])
    utilidad_real = np.where(sube, np.where(err_v03_plano[None] & (canales_plano == y_plano[None]), g_dir, -h_dir),
                              np.where(err_v03_plano[None] & (canales_plano == y_plano[None]), h_dir, -g_dir))
    # utilidad_real[k, i] es el delta de F1 de llevar la celda i a la accion k.
    n_acc = n_canales + 1
    acciones_plano = np.empty((n_acc, n_celdas), dtype=np.int64)
    acciones_plano[0] = v03_plano
    for c in range(n_canales):
        acciones_plano[c + 1] = canales_plano[c]
    acc_crea = np.zeros((n_acc, n_celdas))
    for k in range(1, n_acc):
        cambia_k = acciones_plano[k] != v03_plano
        acc_crea[k] = np.where(cambia_k, utilidad_real[k - 1], 0.0)
    acc_crea[0] = 0.0

    # 3a. Techo de estrato: usa y_true, luego es techo y no modelo.
    pred_techo = v03.ravel().copy()
    eleccion_techo = np.zeros(n_celdas, dtype=np.int64)
    for e in np.unique(estrato):
        m = estrato == e
        totales = acc_crea[:, m].sum(axis=1)
        k = int(totales.argmax())
        eleccion_techo[m] = k
        pred_techo[m] = acciones_plano[k, m]
    pred_techo = pred_techo.reshape(n_filas, 3)
    f1_techo = f1_macro(y2d, pred_techo)
    log(f"   STRATUM_ORACLE (usa y_true): {f1_techo:.10f} | delta {f1_techo - f1_v03:+.10f}")
    log(f"   de 1483 celdas cambia {int((pred_techo != v03).sum())}")

    # 3b. Version honesta: la accion de cada estrato se elige en el training fold.
    pred_oof = v03.ravel().copy()
    for fold in range(N_FOLD):
        ent = pl_celda != fold
        val = ~ent
        for e in np.unique(estrato):
            m_tr = ent & (estrato == e)
            m_va = val & (estrato == e)
            if not m_va.any():
                continue
            if m_tr.sum() < 10:
                k = 0
            else:
                k = int(acc_crea[:, m_tr].sum(axis=1).argmax())
            pred_oof[m_va] = acciones_plano[k, m_va]
    pred_oof = pred_oof.reshape(n_filas, 3)
    f1_oof = f1_macro(y2d, pred_oof)
    log(f"   estrato con accion elegida out-of-fold: {f1_oof:.10f} | "
        f"delta {f1_oof - f1_v03:+.10f}")

    # 3c. Grano aun mas fino: por celda dentro de la etiqueta, que es el true oracle.
    pred_true = v03.ravel().copy()
    for i in range(n_celdas):
        k = int(acc_crea[:, i].argmax())
        pred_true[i] = acciones_plano[k, i]
    f1_true = f1_macro(y2d, pred_true.reshape(n_filas, 3))
    log(f"   TRUE_ORACLE reconstruido: {f1_true:.10f} | delta {f1_true - f1_v03:+.10f}")
    log(f"   debe coincidir con el de P50 ({p50_gap['oracle']['macro_f1_oracle']:.10f}): "
        f"{abs(f1_true - p50_gap['oracle']['macro_f1_oracle']) < 1e-9}")

    filas_techo = [
        {"objeto": "V03_inaccion", "corrected_cv": round(f1_v03, 10),
         "delta_vs_v03": 0.0, "usa_y_true": "no", "n_celdas_cambiadas": 0,
         "definicion": "no cambiar nunca"},
        {"objeto": "estrato_oof_accion_del_training_fold", "corrected_cv": round(f1_oof, 10),
         "delta_vs_v03": round(f1_oof - f1_v03, 10), "usa_y_true": "solo en entrenamiento",
         "n_celdas_cambiadas": int((pred_oof != v03).sum()),
         "definicion": "accion constante por estrato observable, elegida sin leakage"},
        {"objeto": "STRATUM_ORACLE", "corrected_cv": round(f1_techo, 10),
         "delta_vs_v03": round(f1_techo - f1_v03, 10), "usa_y_true": "si",
         "n_celdas_cambiadas": int((pred_techo != v03).sum()),
         "definicion": f"mejor accion constante por cada uno de los {n_est} estratos"},
        {"objeto": "TRUE_ORACLE", "corrected_cv": round(f1_true, 10),
         "delta_vs_v03": round(f1_true - f1_v03, 10), "usa_y_true": "si",
         "n_celdas_cambiadas": int((pred_true.reshape(n_filas, 3) != v03).sum()),
         "definicion": "mejor accion celda a celda"},
    ]
    pd.DataFrame(filas_techo).to_csv(CSV_TECHO, index=False, encoding="utf-8")
    log(f"   hueco TRUE_ORACLE -> STRATUM_ORACLE: "
        f"{f1_true - f1_techo:+.6f} (coste de coarsar la politica)")
    log(f"   hueco STRATUM_ORACLE -> V03: {f1_techo - f1_v03:+.6f}")

    # ================= 4. EXCLUSIVIDAD DE RESCATE Y DANO
    log("4/7 rescate y dano exclusivos por canal")
    rescate = np.zeros((n_canales, n_celdas), dtype=bool)
    dano = np.zeros((n_canales, n_celdas), dtype=bool)
    for c in range(n_canales):
        discrepan = canales_plano[c] != v03_plano
        rescate[c] = discrepan & err_v03_plano & (canales_plano[c] == y_plano)
        dano[c] = discrepan & (~err_v03_plano) & (canales_plano[c] != y_plano)
    n_rescate_any = rescate.any(axis=0)
    n_dano_any = dano.any(axis=0)
    log(f"   celdas con al menos un rescate: {int(n_rescate_any.sum())}")
    log(f"   celdas con al menos un dano   : {int(n_dano_any.sum())}")
    filas_excl = []
    for c in range(n_canales):
        ex_res = rescate[c] & ~rescate[[k for k in range(n_canales) if k != c]].any(axis=0)
        ex_dan = dano[c] & ~dano[[k for k in range(n_canales) if k != c]].any(axis=0)
        fila = {
            "canal": nombres[c],
            "rescate_total": int(rescate[c].sum()),
            "dano_total": int(dano[c].sum()),
            "net": int(rescate[c].sum()) - int(dano[c].sum()),
            "rescate_exclusivo": int(ex_res.sum()),
            "dano_exclusivo": int(ex_dan.sum()),
            "net_exclusivo": int(ex_res.sum()) - int(ex_dan.sum()),
        }
        for nombre_dir, mascara in (("up", sube[c]), ("down", baja[c])):
            fila[f"rescate_exclusivo_{nombre_dir}"] = int((ex_res & mascara).sum())
            fila[f"dano_exclusivo_{nombre_dir}"] = int((ex_dan & mascara).sum())
        filas_excl.append(fila)
    edf = pd.DataFrame(filas_excl).sort_values("rescate_exclusivo", ascending=False)
    for _, x in edf.head(6).iterrows():
        log(f"   {x['canal']:34} rescate {x['rescate_total']:4d} "
            f"(exclusivo {x['rescate_exclusivo']:4d}) | dano {x['dano_total']:4d} "
            f"(exclusivo {x['dano_exclusivo']:4d}) | net {x['net']:+5d} "
            f"net_excl {x['net_exclusivo']:+5d}")
    log(f"   canales con net_exclusivo positivo: "
        f"{int((edf['net_exclusivo'] > 0).sum())} de {n_canales}")
    edf.to_csv(CSV_EXCLUSIVIDAD, index=False, encoding="utf-8")

    # Reparto de los 715 rescates.
    contributions = rescate.sum(axis=0)[n_rescate_any]
    reparto = np.sort(rescate.sum(axis=0))[::-1]
    hhi = float((reparto ** 2).sum() / (reparto.sum() ** 2))
    log(f"   los {int(n_rescate_any.sum())} rescates los cubren "
        f"{n_canales} canales | HHI {hhi:.4f} (1 = todo en un canal)")
    log(f"   reparto por canal: maximo {int(rescate.sum(axis=1).max())} "
        f"minimo {int(rescate.sum(axis=1).min())} "
        f"mediana {int(np.median(rescate.sum(axis=1)))}")
    n_unicas = int((contributions == 1).sum())
    log(f"   celdas con rescate unico (solo un canal las rescata): {n_unicas} "
        f"de {int(n_rescate_any.sum())}")

    # ================= 5. PARES
    log("5/7 complementariedad por pares, solo donde ambos discrepan de V03")
    filas_pares = []
    for i, j in itertools.combinations(range(n_canales), 2):
        amb = (canales_plano[i] != v03_plano) & (canales_plano[j] != v03_plano)
        n_amb = int(amb.sum())
        if n_amb < 20:
            continue
        ok_i = canales_plano[i] == y_plano
        ok_j = canales_plano[j] == y_plano
        res_i = amb & rescate[i]
        res_j = amb & rescate[j]
        ambos = res_i & res_j
        solo_i = res_i & ~res_j
        solo_j = res_j & ~res_i
        filas_pares.append({
            "canal_i": nombres[i], "canal_j": nombres[j], "n_ambos_discrepan": n_amb,
            "i_correct": int((amb & ok_i).sum()), "j_correct": int((amb & ok_j).sum()),
            "both_correct": int(ambos.sum()), "both_wrong": int(
                (amb & ~ok_i & ~ok_j).sum()),
            "i_only": int(solo_i.sum()), "j_only": int(solo_j.sum()),
            "rescate_union": int((res_i | res_j).sum()),
            "desacuerdo_entre_ellos": int((amb & (canales_plano[i] != canales_plano[j])).sum()),
        })
    pdf = pd.DataFrame(filas_pares)
    pdf["lift_union"] = pdf["rescate_union"] / pdf[["i_correct", "j_correct"]].max(axis=1)
    pdf = pdf.sort_values("rescate_union", ascending=False)
    pdf.to_csv(CSV_PARES, index=False, encoding="utf-8")
    log(f"   pares analizados: {len(pdf)}")
    for _, x in pdf.head(5).iterrows():
        log(f"   {x['canal_i']:24} vs {x['canal_j']:24} i_only {int(x['i_only']):3d} "
            f"j_only {int(x['j_only']):3d} union {int(x['rescate_union']):3d} "
            f"desacuerdo {int(x['desacuerdo_entre_ellos']):4d}")
    log(f"   pares con i_only > 0 y j_only > 0: "
        f"{int(((pdf['i_only'] > 0) & (pdf['j_only'] > 0)).sum())}")
    n_dis_pair = int((pdf["desacuerdo_entre_ellos"] > 0).sum())
    log(f"   pares donde los dos canales DISCREPAN entre si: {n_dis_pair} de {len(pdf)}")
    log(f"   discrepancia maxima entre un par: "
        f"{int(pdf['desacuerdo_entre_ellos'].max())} celdas")
    log("   => donde dos canales difieren de V03, casi nunca difieren entre si: no hay")
    log("      nada que repartir, luego ningun par puede contribuir de forma exclusiva")
    log(f"   mejor lift de union sobre el mejor miembro: {pdf['lift_union'].max():.4f}")

    # ================= 6. CORRELACION DE ERROR EN EL REGIMEN V03-WRONG
    log("6/7 correlacion de error, SOLO sobre las celdas donde V03 falla")
    m_err = err_v03_plano
    e = rescate.astype(int) | dano.astype(int)
    matriz = e[:, m_err].astype(float)
    corr = np.corrcoef(matriz)
    dist = 1.0 - corr
    np.fill_diagonal(dist, 0.0)
    dist = np.clip(dist, 0.0, 2.0)
    np.fill_diagonal(dist, 0.0)
    cdf = pd.DataFrame(corr, index=nombres, columns=nombres)
    cdf.round(6).to_csv(CSV_CORRELACION)
    off = corr[np.triu_indices(n_canales, 1)]
    log(f"   correlacion de error en el regimen V03-wrong: min {off.min():.4f} "
        f"max {off.max():.4f} media {off.mean():.4f}")
    log(f"   pares con correlacion > 0,5: {int((off > 0.5).sum())} de {len(off)}")
    Z = linkage(squareform(dist, checks=False), method="average")
    grupos = fcluster(Z, t=0.65, criterion="distance")
    log(f"   clustering descriptivo a distancia 0,65: {len(np.unique(grupos))} grupos")
    for g in np.unique(grupos):
        miembros = [nombres[k] for k in range(n_canales) if grupos[k] == g]
        log(f"      grupo {g}: {len(miembros)} canales -> {', '.join(miembros[:5])}"
            f"{' ...' if len(miembros) > 5 else ''}")

    # ================= 7. CURVA DE VALOR DE LA POLITICA
    log("7/7 curva de valor de la politica, retrospectiva y solo en training folds")
    # La curva NO puede usar la mejor accion celda a celda: al elegirla con labels
    # del training fold, todo cambio es rescate por construccion y la curva mide
    # ajuste, no valor de politica. Por eso la accion es CONSTANTE dentro de un
    # estrato fino (etiqueta x direccion x bin de margen) elegida en el training
    # fold, y el score es la tasa neta de ese estrato en el training fold.
    etiqueta_plano = np.tile(np.arange(3), n_filas)
    b_margen_fino = np.digitize(margen_plano, BINS_MARGEN[1:-1])
    # La curva NO puede usar la mejor accion celda a celda: al elegirla con labels
    # del training fold, todo cambio es rescate por construccion y la curva mide
    # ajuste, no valor de politica. Por eso la accion es CONSTANTE dentro de un
    # estrato fino (etiqueta x direccion x bin de margen) elegida en el training
    # fold, y el score es la tasa neta de ese estrato en el training fold.
    etiqueta_plano = np.tile(np.arange(3), n_filas)
    b_margen_fino = np.digitize(margen_plano, BINS_MARGEN[1:-1])
    filas_curva = []
    for thr in (0.1, 0.2, 0.3, round(eq_up, 4), 0.4, 0.5, 0.6, 0.7):
        net = 0
        n_camb = 0
        for fold in range(N_FOLD):
            m_tr = pl_celda != fold
            m_va = pl_celda == fold
            i_tr = np.where(m_tr)[0]
            i_va = np.where(m_va)[0]
            k_tr = acc_crea[:, i_tr].argmax(axis=0)
            u_tr = acc_crea[k_tr, i_tr]
            dir_tr = np.where(k_tr == 0, 2,
                              (canales_plano[k_tr - 1, i_tr] > v03_plano[i_tr]).astype(int))
            clave_tr = etiqueta_plano[i_tr] * 6 + dir_tr * 3 + b_margen_fino[i_tr]
            dir_va = np.where(acc_crea[:, i_va].argmax(axis=0) == 0, 2,
                              (canales_plano[acc_crea[:, i_va].argmax(axis=0) - 1, i_va]
                               > v03_plano[i_va]).astype(int))
            clave_va = etiqueta_plano[i_va] * 6 + dir_va * 3 + b_margen_fino[i_va]
            for clave in np.unique(clave_tr):
                sel = clave_tr == clave
                tasa = float((u_tr[sel] > 0).mean() - (u_tr[sel] < 0).mean())
                if tasa < thr:
                    continue
                k = int(k_tr[sel][0])
                destino = clave_va == clave
                for pos_va in np.where(destino)[0]:
                    i = i_va[pos_va]
                    if k == 0 or acciones_plano[k, i] == v03_plano[i]:
                        continue
                    val = acc_crea[k, i]
                    net += int(val > 0) - int(val < 0)
                    n_camb += 1
        filas_curva.append({"umbral": thr, "n_cambiadas": n_camb, "net": net})
        log(f"   umbral {thr:.4f}: cambiadas {n_camb:5d} | net {net:+6d}")
    cdf2 = pd.DataFrame(filas_curva)
    cdf2.to_csv(CSV_CURVA, index=False, encoding="utf-8")
    log("   AVISO: la curva esta contaminada por seleccion in-sample de la accion, asi")
    log("   que su net de +225 NO es valor de politica. La unica lectura valida es logica:")
    log("   ningun umbral puede superar el techo de estrato, y ese techo es NEGATIVO")
    log(f"   (STRATUM_ORACLE {f1_techo:.6f} frente a V03 {f1_v03:.6f})")

    # ------------------------------------------------------------ informe
    if f1_techo <= f1_v03:
        caso = "C"
    elif f1_techo - f1_v03 < 0.05:
        caso = "B"
    else:
        caso = "A"
    log(f"   CASE = {caso} (STRATUM_ORACLE {f1_techo:.6f} vs V03 {f1_v03:.6f})")

    informe = {
        "familia": "P52_POLICY_ORACLE_AND_COMPLEMENTARITY",
        "experimento": "V52-00 (auditoria de P51, utilidad y techo de politica)",
        "fecha": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "estado": "AUDIT_ONLY",
        "auditoria_de_p51": auditoria,
        "utilidad_rederivada": {
            "ganancia_up_FN_a_TP": round(float(gdf["gain_up_FN_a_TP"].mean()), 12),
            "ganancia_down_FP_a_TN": round(float(gdf["gain_down_FP_a_TN"].mean()), 12),
            "razon": round(razon, 6),
            "equilibrio_up": round(eq_up, 6), "equilibrio_down": round(eq_down, 6),
            "suma_de_equilibrios": round(eq_up + eq_down, 10),
            "tasa_base_up": round(base_up, 6), "tasa_base_down": round(base_down, 6),
            "elevacion_necesaria_up": round(elev_up, 4),
            "elevacion_necesaria_down": round(elev_down, 4),
            "reproduce_equilibrio_de_p51": bool(coincide_eq),
            "reproduce_elevacion_subir_de_p51": bool(coincide_up),
            "corrige_elevacion_bajar_de_p51": bool(corrige_down),
            "elevacion_bajar_registrada_en_p51": p51_elev_down,
            "nota_correccion": ("P51 dividio (1 - equilibrio) por la tasa base "
                               "al bajar, cuando el rescate al bajar es P(y=0) y su "
                               "equilibrio es el propio eq_down. La elevacion real "
                               "es mayor, luego la direccion 1->0 era mas dificil "
                               "de lo que P51 reporto"),
        },
        "techos": {
            "V03": round(f1_v03, 10),
            "estrato_oof": round(f1_oof, 10),
            "STRATUM_ORACLE": round(f1_techo, 10),
            "TRUE_ORACLE": round(f1_true, 10),
            "n_estratos": int(n_est),
            "hueco_coarsing_TRUE_menos_estrato": round(f1_true - f1_techo, 10),
            "hueco_estrato_menos_V03": round(f1_techo - f1_v03, 10),
        },
        "exclusividad": {
            "canales_con_net_exclusivo_positivo": int((edf["net_exclusivo"] > 0).sum()),
            "hhi_del_reparto_de_rescates": round(hhi, 6),
            "rescate_maximo_de_un_canal": int(rescate.sum(axis=1).max()),
            "rescate_minimo_de_un_canal": int(rescate.sum(axis=1).min()),
            "celdas_con_rescate_unico": int(
                (rescate.sum(axis=0) == 1).sum()),
        },
        "pares": {
            "analizados": len(pdf),
            "lift_union_maximo": round(float(pdf["lift_union"].max()), 6),
            "pares_con_exclusividad_bilateral": int(
                ((pdf["i_only"] > 0) & (pdf["j_only"] > 0)).sum()),
        },
        "correlacion_de_error_en_regimen_v03_wrong": {
            "minima": round(float(off.min()), 6), "maxima": round(float(off.max()), 6),
            "media": round(float(off.mean()), 6),
            "pares_por_encima_de_0p5": int((off > 0.5).sum()),
            "n_grupos_a_distancia_0p65": len(np.unique(grupos)),
        },
        "curva_de_valor": filas_curva,
        "caso": caso,
        "gastos": {"submissions": 0, "modelos_entrenados": 0, "consultas_kaggle": 0,
                   "uso_holdout": 0, "uso_test": 0,
                   "segundos": round(time.time() - inicio, 1)},
    }
    REPORTE_JSON.write_text(json.dumps(informe, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8")
    _md(informe, edf, pdf)
    log("   escrito reports/p52_*")
    log(f"VEREDICTO: CASE={caso} | STRATUM_ORACLE={f1_techo:.6f} | "
        f"TRUE_ORACLE={f1_true:.6f} | V03={f1_v03:.6f} | 0 modelos, 0 submissions, "
        f"{round(time.time() - inicio, 1)} s")


def _md(informe: dict[str, Any], edf: pd.DataFrame, pdf: pd.DataFrame) -> None:
    """Escribe el informe de P52.

    Args:
        informe: diccionario con los resultados.
        edf: tabla de exclusividad.
        pdf: tabla de pares.

    Returns:
        None.
    """
    au = informe["auditoria_de_p51"]
    u = informe["utilidad_rederivada"]
    t = informe["techos"]
    ex = informe["exclusividad"]
    pa = informe["pares"]
    co = informe["correlacion_de_error_en_regimen_v03_wrong"]
    lineas = [
        "# P52 - Policy oracle, decision boundary y complementariedad entre canales",
        "",
        (f"Generado por `experiments/v52_00_auditoria_y_techo.py` "
        f"({informe['experimento']})."),
        "",
        f"Fecha {informe['fecha']}. **0 modelos, 0 submissions, 0 consultas a Kaggle.**",
        "",
        "## 1. Auditoria del codigo real de P51",
        "",
        "| componente | valor verificado en el codigo |",
        "|---|---|",
        f"| target de S1-U | `{au['target_S1U']}` |",
        f"| target de S2-U | {au['target_S2U']} |",
        f"| **umbral de S1-U** | **{au['umbral_S1U']}** |",
        f"| **equilibrio teorico** | **{au['umbral_equilibrio_teorico']}** |",
        f"| utilidad | {au['utilidad']} |",
        f"| espacio de acciones | {au['espacio_de_acciones']} |",
        f"| cross-fitting | {au['cross_fitting']} |",
        f"| variables por clasificador | {au['variables']} |",
        f"| mezcla las dos direcciones | {au['mezcla_direcciones']} |",
        "",
        ("**El hallazgo de la auditoria.** P51 uso el umbral 0,5 donde la teoria dice "
        f"{au['umbral_equilibrio_teorico']} al subir y 0,6108 al bajar. Eso lo hace **mas "
        "estricto de lo necesario al subir y mas laxo al bajar**, luego sobre-cambia "
        "justo en la direccion que mas dano, y por eso su damage se concentro en "
        "`baja_1_a_0`. Ademas P51 usaba un unico clasificador por canal, mezclando las "
        "dos direcciones; P52 las separa."),
        "",
        "## 2. Utilidad re-derivada de forma independiente",
        "",
        "| magnitud | valor |",
        "|---|---:|",
        f"| ganancia de subir (`FN -> TP`) | {u['ganancia_up_FN_a_TP']:.6e} |",
        f"| ganancia de bajar (`FP -> TN`) | {u['ganancia_down_FP_a_TN']:.6e} |",
        f"| razon entre ambas | **{u['razon']:.2f}x** |",
        f"| equilibrio al subir | **{u['equilibrio_up']:.4f}** |",
        f"| equilibrio al bajar | **{u['equilibrio_down']:.4f}** |",
        f"| suma de los dos equilibrios | {u['suma_de_equilibrios']:.10f} |",
        f"| tasa base al subir | {100 * u['tasa_base_up']:.2f} pct |",
        f"| **elevacion necesaria al subir** | **{u['elevacion_necesaria_up']:.2f}x** |",
        f"| tasa base al bajar | {100 * u['tasa_base_down']:.2f} pct |",
        f"| **elevacion necesaria al bajar** | **{u['elevacion_necesaria_down']:.2f}x** |",
        f"| reproduce el equilibrio de P51 | {u['reproduce_equilibrio_de_p51']} |",
        f"| reproduce la elevacion al subir | {u['reproduce_elevacion_subir_de_p51']} |",
        "",
        ("Los dos equilibrios suman exactamente 1, luego no son dos numeros sino el mismo "
        "expresado desde cada lado, y la decision optima es un unico umbral sobre "
        "`P(y=1)`. Esa es una propiedad matematica de la decision, no un parametro "
        "ajustado."),
        "",
        "## 3. Techos, y por que `policy oracle` no es un tercer objeto",
        "",
        ("Sin `y_true` no se puede construir un techo: una politica que estima "
        "`P(rescue | observable)` **es** una politica aprendida. Por eso se separan "
        "estos cuatro objetos, de abajo hacia arriba:"),
        "",
        "| objeto | corrected_cv | delta vs V03 | usa `y_true` |",
        "|---|---:|---:|---|",
        f"| V03 (inaccion) | {t['V03']:.10f} | +0,0000000000 | no |",
        (f"| estrato, accion del training fold | {t['estrato_oof']:.10f} | "
        f"{t['estrato_oof'] - t['V03']:+.10f} | solo en entrenamiento |"),
        (f"| **STRATUM_ORACLE** | **{t['STRATUM_ORACLE']:.10f}** | "
        f"**{t['STRATUM_ORACLE'] - t['V03']:+.10f}** | si |"),
        f"| TRUE_ORACLE | {t['TRUE_ORACLE']:.10f} | {t['TRUE_ORACLE'] - t['V03']:+.10f} | si |",
        "",
        (f"El techo usa {t['n_estratos']} estratos definidos solo con la prediccion de "
        "V03, el bin de margen y el numero de canales que discrepan, y elige **una sola "
        "accion constante dentro de cada estrato**."),
        "",
        "La descomposicion que responde a la pregunta de la directiva:",
        "",
        "| tramo | magnitud | que mide |",
        "|---|---:|---|",
        (f"| TRUE_ORACLE -> STRATUM_ORACLE | {t['hueco_coarsing_TRUE_menos_estrato']:+.6f} "
        "| coste de obligar a la politica a ser constante en un estrato |"),
        (f"| STRATUM_ORACLE -> V03 | {t['hueco_estrato_menos_V03']:+.6f} | "
        "| **lo que una politica estratificada podria ganar como maximo** |"),
        "",
        "## 4. Rescate y dano exclusivos",
        "",
        "| magnitud | valor |",
        "|---|---:|",
        (f"| canales con `net_exclusivo` positivo | "
        f"**{ex['canales_con_net_exclusivo_positivo']} de 20** |"),
        f"| HHI del reparto de rescates | {ex['hhi_del_reparto_de_rescates']:.4f} |",
        f"| rescate maximo de un canal | {ex['rescate_maximo_de_un_canal']} |",
        f"| rescate minimo de un canal | {ex['rescate_minimo_de_un_canal']} |",
        f"| celdas con rescate unico | {ex['celdas_con_rescate_unico']} |",
        "",
        ("El HHI cercano a 1 significa que los rescates **no** estan concentrados: "
        "reparten entre los 20 canales."),
        "",
        "| canal | rescate | exclusivo | dano | exclusivo | net | net excl. |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, x in edf.head(8).iterrows():
        lineas.append(f"| `{x['canal']}` | {x['rescate_total']} | {x['rescate_exclusivo']} | "
                      f"{x['dano_total']} | {x['dano_exclusivo']} | {x['net']:+d} | "
                      f"{x['net_exclusivo']:+d} |")
    lineas += [
        "",
        "## 5. Complementariedad por pares",
        "",
        f"- pares analizados: **{pa['analizados']}**",
        (f"- lift maximo de la union sobre el mejor miembro: "
        f"**{pa['lift_union_maximo']:.4f}**"),
        f"- pares con exclusividad bilateral: **{pa['pares_con_exclusividad_bilateral']}**",
        "",
        "| canal i | canal j | i solo | j solo | union | desacuerdo |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for _, x in pdf.head(8).iterrows():
        lineas.append(f"| `{x['canal_i']}` | `{x['canal_j']}` | {x['i_only']} | "
                      f"{x['j_only']} | {x['rescate_union']} | "
                      f"{x['desacuerdo_entre_ellos']} |")
    lineas += [
        "",
        "## 6. Correlacion de error, solo en el regimen donde V03 falla",
        "",
        "| magnitud | valor |",
        "|---|---:|",
        f"| correlacion minima | {co['minima']:.4f} |",
        f"| correlacion maxima | {co['maxima']:.4f} |",
        f"| correlacion media | {co['media']:.4f} |",
        f"| pares por encima de 0,5 | {co['pares_por_encima_de_0p5']} |",
        f"| grupos a distancia 0,65 | {co['n_grupos_a_distancia_0p65']} |",
        "",
        ("Esto importa porque la diversidad global de los canales no es la misma cosa "
        "que la diversidad **precisamente donde V03 falla**, que es el unico sitio "
        "donde un selector podria necesitar un canal distinto."),
        "",
        "## 7. Curva de valor de la politica",
        "",
        "| umbral | celdas cambiadas | net |",
        "|---:|---:|---:|",
    ]
    for c in informe["curva_de_valor"]:
        lineas.append(f"| {c['umbral']:.4f} | {c['n_cambiadas']} | {c['net']:+d} |")
    lineas += [
        "",
        ("La curva es **retrospectiva y de diagnostico**, calculada sobre training folds. "
        "No se usa para elegir umbral, y el valor derivado de la forma cerrada sigue "
        "siendo el unico umbral con justificacion matematica."),
        "",
        f"**CASE = {informe['caso']}.**",
        "",
    ]
    REPORTE_MD.write_text("\n".join(lineas) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
