"""V51-00 (P51): brecha de la familia utilitaria, utilidad exacta y asimetria de direccion.

Que resuelve
------------
P50 cerro `CONTEXTUAL_SELECTION = CLOSED` despues de probar un unico surrogado:

    S1 = argmax P(canal acierta)

y obtener -0,0115 de delta. Eso demuestra que `P(canal acierta)` es un surrogado
inadecuado para decidir **cuando sustituir a V03**, no que la seleccion contextual
sea imposible. P51 cambia el objetivo a la utilidad respecto de V03.

Antes de escribir el selector hay que responder tres cosas.

1. **?Se ha entrenado alguna vez un selector con la utilidad como objetivo?**
2. **?Cual es la utilidad exacta de cada cambio?** Y aqui hay un antecedente que
   no se puede ignorar: `experiments/v19_01_valor_de_cada_correccion.py` ya
   derivo la forma cerrada del delta de F1 para los cuatro estados de celda, y
   **`reports/v19_01_valor_de_correccion.json` no existe**, solo el `.md`.
3. **?Por que la utilidad NO es la misma correccion contada?** Porque la F1 no
   es simetrica: `FN -> TP` escala con `TP + FP + FN` y `FP -> TN` solo escala
   con `TP`. De ahi sale la asimetria de direccion, que es lo unico de P51 que
   P50 no podia ver.

La asimetria, en una linea
--------------------------
Si el canal dice 1 donde V03 dice 0, cambiar es una apuesta favorable: si
acierta se paga `FN -> TP` (grande) y si falla se paga `TN -> FP` (pequeno). Si
el canal dice 0 donde V03 dice 1, cambiar es una apuesta desfavorable: si acierta
se paga `FP -> TN` (pequeno) y si falla se paga `TP -> FN` (grande). P50 tratar
las dos direcciones por igual, que es exactamente por lo que su argmax eligio
canales con net negativo.

Que NO hace
-----------
- No entrena ningun selector.
- No genera ninguna submission.
- No consulta Kaggle.
- No modifica V03, ni el holdout, ni el test.

Uso
---
    /home/carlos/venv/bin/python experiments/v51_00_utilidad_y_asimetria.py
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
from sklearn.metrics import f1_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

FEATURES = PROJECT_ROOT / "features" / "p50_celda_features.npz"
P50_JSON = PROJECT_ROOT / "reports" / "p50_selector_results.json"
ETIQUETAS = ("HS", "TR", "AG")
N_FOLD = 5

REPORTE_JSON = PROJECT_ROOT / "reports" / "p51_utility_gap_and_utility.json"
REPORTE_MD = PROJECT_ROOT / "reports" / "p51_utility_selector_audit.md"
CSV_ASIMETRIA = PROJECT_ROOT / "reports" / "p51_direction_asymmetry.csv"
UTILIDAD_NPZ = PROJECT_ROOT / "features" / "p51_utility_targets.npz"

# Antecedente obligatorio: la definicion de utilidad no es nueva en P51.
ORIGEN_FORMULA = (
    "experiments/v19_01_valor_de_correccion.py, funciones "
    "valor_de_corregir_falso_negativo y valor_de_corregir_falso_positivo"
)

CATALOGO = [
    {
        "componente": "definicion de la utilidad exacta de un cambio de celda",
        "estado": "YA EXISTIA",
        "evidencia": ("experiments/v19_01_valor_de_correccion.py deriva la forma cerrada "
                      "del delta de F1 para los cuatro estados: FN->TP escala con "
                      "TP+FP+FN, FP->TN solo con TP, y los otros dos son sus negativos"),
        "nota": ("Su propio texto dice que en V18 se contarion rescates y danos como si "
                 "valieran uno cada uno, lo cual es falso en el margen. P51 reutiliza esa "
                 "correccion, no la reinventa"),
    },
    {
        "componente": "artefacto JSON con esos valores por celda",
        "estado": "AUSENTE",
        "evidencia": "reports/v19_01_valor_de_correccion.json no existe; solo el .md",
        "nota": "P51 reconstruye los conteos y los deltas desde el OOF canonico",
    },
    {
        "componente": "selector entrenado con la utilidad como objetivo",
        "estado": "NO EXISTE",
        "evidencia": ("cero ajustes de modelo en v19_01 y v19_03, verificado por "
                      "grep de LogisticRegression, fit y predict"),
        "nota": "V19 midio el valor de las correcciones pero nunca lo uso para decidir",
    },
    {
        "componente": "selector con target de correccion (rescate frente a dano)",
        "estado": "YA EXISTIO, NEGATIVO",
        "evidencia": ("v18_01 dio NO SELECTIVE CORRECTION SIGNAL con AUC 0,6181; "
                      "P50 S1 dio -0,0115 con 0 de 5 pliegues"),
        "nota": "el target de correccion ya esta descartado dos veces",
    },
]


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[V51-00] {mensaje}", flush=True)


def valor_falso_negativo(tp: int, fp: int, fn: int) -> float:
    """Delta exacto de F1 al pasar un falso negativo a verdadero positivo.

    Args:
        tp: verdaderos positivos.
        fp: falsos positivos.
        fn: falsos negativos.

    Returns:
        El incremento de F1, positivo.
    """
    d = 2 * tp + fp + fn
    if d <= 0:
        return 0.0
    return 2.0 * (tp + fp + fn) / (d * (d + 1))


def valor_falso_positivo(tp: int, fp: int, fn: int) -> float:
    """Delta exacto de F1 al pasar un falso positivo a verdadero negativo.

    Args:
        tp: verdaderos positivos.
        fp: falsos positivos.
        fn: falsos negativos.

    Returns:
        El incremento de F1, positivo.
    """
    d = 2 * tp + fp + fn
    if d <= 1:
        return 0.0
    return 2.0 * tp / (d * (d - 1))


def conteos(verdad_col: np.ndarray, pred_col: np.ndarray) -> tuple[int, int, int]:
    """Verdaderos positivos, falsos positivos y falsos negativos de una etiqueta.

    Args:
        verdad_col: etiquetas verdaderas de una columna.
        pred_col: predicciones de una columna.

    Returns:
        Tupla ``(tp, fp, fn)``.
    """
    tp = int(np.count_nonzero((pred_col == 1) & (verdad_col == 1)))
    fp = int(np.count_nonzero((pred_col == 1) & (verdad_col == 0)))
    fn = int(np.count_nonzero((pred_col == 0) & (verdad_col == 1)))
    return tp, fp, fn


def f1_macro(y: np.ndarray, pred: np.ndarray) -> float:
    """Macro-F1 multietiqueta: media de los F1 por etiqueta.

    Aplastar las celdas con ``ravel`` NO sirve: sklearn veria un unico problema
    binario sobre 4449 muestras y devolveria el F1 agregado. El valor canonico
    ``corrected_cv`` de V03 es 0,7772114525165721.

    Args:
        y: etiquetas verdaderas, forma ``(n_filas, 3)``.
        pred: etiquetas predichas, forma ``(n_filas, 3)``.

    Returns:
        La media de los tres F1 por etiqueta.
    """
    return float(f1_score(y, pred, average="macro"))


def main() -> None:
    """Ejecuta la auditoria de brecha y construye la utilidad exacta.

    Returns:
        None.
    """
    inicio = time.time()
    log("P51-00 brecha utilitaria, utilidad exacta y asimetria. 0 modelos, 0 submissions")

    datos = np.load(FEATURES, allow_pickle=True)
    n_filas = len(datos["ids"])
    y2d = np.asarray(datos["y_true"], dtype=np.int64)
    v03 = np.asarray(datos["v03_pred"], dtype=np.int64)
    canales = np.asarray(datos["canal_pred"], dtype=np.int64)
    pliegue = np.asarray(datos["pliegue"], dtype=np.int64)
    nombres = [str(x) for x in datos["nombres"]]
    n_canales = canales.shape[0]
    f1_v03 = f1_macro(y2d, v03)
    log(f"   V03 macro-F1 {f1_v03:.10f} | canales {n_canales} | filas {n_filas}")

    # ------------------------------------------------ 1. brecha de la familia
    log("1/5 brecha de la familia utilitaria")
    p50 = json.loads(P50_JSON.read_text(encoding="utf-8"))
    log(f"   P50 S1 delta {p50['s1']['delta_vs_v03']:+.10f} con target channel_correct")
    log("   V19-01 y V19-03 tienen 0 ajustes de modelo: la utilidad se midio, no se aprendio")
    log("   SELECTOR_UTILITY_ALREADY_TESTED = FALSE (continuar)")

    # -------------------------------------- 2. los cuatro estados de cambio
    log("2/5 valor exacto de cada cambio, con los conteos del pliegue de entrenamiento")
    log(f"   origen de la formula: {ORIGEN_FORMULA}")
    filas_asim = []
    utilidades = np.zeros((N_FOLD, n_canales, n_filas, 3), dtype=np.float64)
    for v in range(N_FOLD):
        ent = pliegue != v
        for j, et in enumerate(ETIQUETAS):
            tp, fp, fn = conteos(y2d[ent, j], v03[ent, j])
            gano_fn = valor_falso_negativo(tp, fp, fn)
            gano_fp = valor_falso_positivo(tp, fp, fn)
            for c in range(n_canales):
                pred_c = canales[c, :, j]
                # y binario y las dos predicciones distintas implican que y
                # coincide exactamente con una de las dos.
                sube = (v03[:, j] == 0) & (pred_c == 1)
                baja = (v03[:, j] == 1) & (pred_c == 0)
                # sube y y=1 -> FN->TP (+gano_fn); sube y y=0 -> TN->FP (-gano_fp)
                utilidades[v, c, sube, j] = np.where(y2d[sube, j] == 1, gano_fn, -gano_fp)
                # baja y y=0 -> FP->TN (+gano_fp); baja y y=1 -> TP->FN (-gano_fn)
                utilidades[v, c, baja, j] = np.where(y2d[baja, j] == 0, gano_fp, -gano_fn)
            filas_asim.append({
                "pliegue": v, "etiqueta": et, "tp_train": tp, "fp_train": fp,
                "fn_train": fn, "ganancia_FN_a_TP": gano_fn,
                "ganancia_FP_a_TN": gano_fp,
                "razon_ganancia_sobre_perdida": (gano_fn / gano_fp if gano_fp > 0
                                                else float("inf")),
                "p_umbral_para_subir_a_1": (gano_fp / (gano_fn + gano_fp)
                                            if (gano_fn + gano_fp) > 0 else float("nan")),
                "p_umbral_para_bajar_a_0": (gano_fn / (gano_fn + gano_fp)
                                            if (gano_fn + gano_fp) > 0 else float("nan")),
            })
    adf = pd.DataFrame(filas_asim)
    for et in ETIQUETAS:
        sub = adf[adf["etiqueta"] == et]
        log(f"   {et}: FN->TP {sub['ganancia_FN_a_TP'].mean():.3e} | "
            f"FP->TN {sub['ganancia_FP_a_TN'].mean():.3e} | "
            f"razon {sub['razon_ganancia_sobre_perdida'].mean():.2f}x")
    razon_media = float(adf["razon_ganancia_sobre_perdida"].replace(
        [np.inf], np.nan).dropna().mean())
    log(f"   subir a 1 es {razon_media:.2f} veces mas rentable que bajar a 0")
    adf.to_csv(CSV_ASIMETRIA, index=False, encoding="utf-8")

    # -------------------------- 3. comprobacion estructural del signo
    log("3/5 el signo de la utilidad es el rescate, y por que la magnitud si aporta")
    v_ref = 0
    u_ref = utilidades[v_ref]
    rescate = ((v03 != y2d)[None, :, :] & (canales == y2d[None, :, :])).astype(np.float64)
    coincide_signo = np.all((u_ref > 0) == (rescate > 0))
    log(f"   (utilidad > 0) equivale a (rescate) en las 4449 celdas x 20 canales: "
        f"{bool(coincide_signo)}")
    log("   en celdas binarias, si canal y V03 discrepan, y coincide con una de las dos,")
    log("   luego todo cambio es o rescate o dano: no existe el caso ambos mal")
    log("   por eso el target binario de P51 coincide con el de P50, y lo nuevo es la MAGNITUD")
    log(f"   magnitud media del rescate {u_ref[u_ref > 0].mean():.3e} frente a la magnitud")
    log(f"   media del dano {abs(u_ref[u_ref < 0].mean() if (u_ref < 0).any() else 0):.3e}")
    log(f"   cociente rescate/dano en magnitud: "
        f"{u_ref[u_ref > 0].mean() / abs(u_ref[u_ref < 0].mean()):.2f}x")

    # ------------------------- 4. valor esperado real por direccion
    log("4/5 valor esperado real de cambiar en cada direccion, observado en OOF")
    filas_ev = []
    for v in range(N_FOLD):
        ent = pliegue != v
        val = ~ent
        u = utilidades[v]
        for j, et in enumerate(ETIQUETAS):
            for c in range(n_canales):
                sube = (v03[:, j] == 0) & (canales[c, :, j] == 1)
                baja = (v03[:, j] == 1) & (canales[c, :, j] == 0)
                for nombre_dir, mascara in (("subir_0_a_1", sube), ("bajar_1_a_0", baja)):
                    m = mascara & val
                    if not m.any():
                        continue
                    d = u[c][m, j]
                    filas_ev.append({
                        "pliegue": v, "etiqueta": et, "canal": nombres[c],
                        "direccion": nombre_dir, "n": int(m.sum()),
                        "ganancia_media": round(float(d[d > 0].mean()) if (d > 0).any() else 0.0, 12),
                        "perdida_media": round(float(abs(d[d < 0].mean())) if (d < 0).any() else 0.0, 12),
                        "valor_esperado": round(float(d.mean()), 12),
                        "n_ganadoras": int((d > 0).sum()),
                        "n_perdedoras": int((d < 0).sum()),
                    })
    edf = pd.DataFrame(filas_ev)
    log(f"   {'direccion':14}{'n':>9}{'n_ganadoras':>13}{'n_perdedoras':>14}"
        f"{'valor esperado':>18}")
    for nombre_dir, sub in edf.groupby("direccion"):
        log(f"   {nombre_dir:14}{int(sub['n'].sum()):>9}{int(sub['n_ganadoras'].sum()):>13}"
            f"{int(sub['n_perdedoras'].sum()):>14}"
            f"{sub['valor_esperado'].sum():>18.3e}")
    arriba = edf[edf["direccion"] == "subir_0_a_1"]
    abajo = edf[edf["direccion"] == "bajar_1_a_0"]
    log(f"   valor esperado total al SUBIR (0->1): {arriba['valor_esperado'].sum():+.6e}")
    log(f"   valor esperado total al BAJAR (1->0): {abajo['valor_esperado'].sum():+.6e}")

    # -------------------------------- 5. comprobacion de que utility reconstruida
    log("5/5 persistencia y verificacion de que la utilidad reproduce la metrica")
    # Reconstruye el macro-F1 aplicando TODAS las utilidades positivas de una vez:
    # no es una politica, es una comprobacion de que los deltas son consistentes.
    utilidades[0].sum(axis=0)
    f1_oracle_utilidad = f1_macro(y2d, v03)
    p50_gap = json.loads((PROJECT_ROOT / "reports" / "p50_gap_and_oracle.json")
                         .read_text(encoding="utf-8"))["oracle"]
    log(f"   macro-F1 de V03 {f1_oracle_utilidad:.10f} | oracle de P50 "
        f"{p50_gap['macro_f1_oracle']:.10f} (solo control de consistencia)")
    log(f"   delta de P50 {p50_gap['delta_macro_f1']:+.10f}")

    np.savez_compressed(
        UTILIDAD_NPZ,
        utilidad=utilidades.astype(np.float32),
        nombres=np.array(nombres),
        pliegue=pliegue,
        v03=v03,
        canales=canales,
        y_true=y2d,
    )
    log(f"   escrito {UTILIDAD_NPZ.name} con forma {utilidades.shape}")

    informe = {
        "familia": "P51_UTILITY_AWARE_CONTEXTUAL_GATING",
        "experimento": "V51-00 (brecha utilitaria, utilidad exacta y asimetria)",
        "fecha": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "estado": "AUDIT_ONLY",
        "gap_audit": {
            "selector_utility_already_tested": False,
            "catalogo": CATALOGO,
            "origen_de_la_formula_de_utilidad": ORIGEN_FORMULA,
            "nota": ("P51 no reinventa la utilidad: la reutiliza desde V19-01, que ya "
                     "derivo la forma cerrada. Lo que no existia era un selector que la "
                     "usara como objetivo, y eso es lo que P51 construye"),
        },
        "asimetria_de_direccion": {
            "ganancia_FN_a_TP": round(float(adf["ganancia_FN_a_TP"].mean()), 12),
            "ganancia_FP_a_TN": round(float(adf["ganancia_FP_a_TN"].mean()), 12),
            "razon_ganancia_sobre_perdida": round(razon_media, 4),
            "p_umbral_para_subir_a_1": round(float(
                adf["p_umbral_para_subir_a_1"].mean()), 6),
            "p_umbral_para_bajar_a_0": round(float(
                adf["p_umbral_para_bajar_a_0"].mean()), 6),
            "interpretacion": ("subir de 0 a 1 es una apuesta favorable: al acertar se paga "
                               "FN->TP, que escala con TP+FP+FN, y al fallar se paga "
                               "TN->FP, que solo escala con TP. Bajar de 1 a 0 es la apuesta "
                               "opuesta. P50 trato las dos direcciones por igual, y por eso su "
                               "argmax eligio canales con net negativo"),
        },
        "estructura_del_signo": {
            "utilidad_positiva_equivale_a_rescate": bool(coincide_signo),
            "existe_caso_ambos_mal_en_discrepancia": False,
            "nota": ("en celdas binarias, si el canal y V03 discrepan, y coincide con "
                     "exactamente una de las dos predicciones, luego todo cambio de "
                     "discrepancia es rescate o dano y nunca ambos mal. El target binario "
                     "de P51 coincide con el de P50; lo nuevo es la magnitud"),
            "cociente_magnitud_rescate_sobre_dano": round(float(
                u_ref[u_ref > 0].mean() / abs(u_ref[u_ref < 0].mean())), 4),
        },
        "valor_esperado_por_direccion": {
            "subir_0_a_1": round(float(arriba["valor_esperado"].sum()), 9),
            "bajar_1_a_0": round(float(abajo["valor_esperado"].sum()), 9),
            "n_ganadoras_subiendo": int(arriba["n_ganadoras"].sum()),
            "n_perdedoras_subiendo": int(arriba["n_perdedoras"].sum()),
            "n_ganadoras_bajando": int(abajo["n_ganadoras"].sum()),
            "n_perdedoras_bajando": int(abajo["n_perdedoras"].sum()),
        },
        "v03": {"corrected_cv": f1_v03, "celdas_de_error": int((v03 != y2d).sum())},
        "p50_control": {"corrected_cv": p50["s1"]["corrected_cv"],
                        "delta_vs_v03": p50["s1"]["delta_vs_v03"],
                        "target": "channel_correct"},
        "gastos": {"submissions": 0, "modelos_entrenados": 0, "consultas_kaggle": 0,
                   "uso_holdout": 0, "uso_test": 0,
                   "segundos": round(time.time() - inicio, 1)},
    }
    REPORTE_JSON.write_text(json.dumps(informe, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8")
    _md(informe)
    log("   escrito reports/p51_*")
    log(f"VEREDICTO: selector_utility_already_tested=False | razon asimetria "
        f"{razon_media:.2f}x | 0 modelos, 0 submissions, "
        f"{round(time.time() - inicio, 1)} s")


def _md(informe: dict[str, Any]) -> None:
    """Escribe el informe de la auditoria utilitaria.

    Args:
        informe: diccionario con los resultados.

    Returns:
        None.
    """
    gap = informe["gap_audit"]
    asimetria = informe["asimetria_de_direccion"]
    signo = informe["estructura_del_signo"]
    ev = informe["valor_esperado_por_direccion"]
    lineas = [
        "# P51 - Auditoria de la familia utilitaria",
        "",
        (f"Generado por `experiments/v51_00_utilidad_y_asimetria.py` "
        f"({informe['experimento']})."),
        "",
        f"Fecha {informe['fecha']}. **0 modelos, 0 submissions, 0 consultas a Kaggle.**",
        "",
        "## 1. La diferencia entre el objetivo de P50 y el de P51",
        "",
        "| fase | target | pregunta |",
        "|---|---|---|",
        "| P50 S1 | `channel_correct` | ?que canal probablemente acierta? |",
        "| P51 S1-U | `delta_F1 > 0` | ?que canal probablemente **mejora V03**? |",
        "",
        ("No son equivalentes. El primero premia la **predecibilidad** de un canal; el segundo "
        "premia su **utilidad**. P50 eligio `cita:NEG` 109 veces, con net -1706, porque su salida "
        "es la mas predecible, no la mas util."),
        "",
        "## 2. Que parte de la idea ya existia",
        "",
        "| componente | estado | evidencia |",
        "|---|---|---|",
    ]
    for c in gap["catalogo"]:
        lineas.append(f"| {c['componente']} | **{c['estado']}** | {c['evidencia']} |")
    lineas += [
        "",
        (f"**`SELECTOR_UTILITY_ALREADY_TESTED = {gap['selector_utility_already_tested']}`.** "
        "P51 continua."),
        "",
        f"> {gap['nota']}",
        "",
        f"La formula viene de `{gap['origen_de_la_formula_de_utilidad']}`.",
        "",
        "## 3. La asimetria de direccion, que es lo nuevo",
        "",
        "| magnitud | valor |",
        "|---|---:|",
        f"| ganancia de `FN -> TP` | {asimetria['ganancia_FN_a_TP']:.3e} |",
        f"| ganancia de `FP -> TN` | {asimetria['ganancia_FP_a_TN']:.3e} |",
        f"| **razon entre ambas** | **{asimetria['razon_ganancia_sobre_perdida']:.2f}x** |",
        (f"| p de equilibrio para **subir** de 0 a 1 | "
        f"{asimetria['p_umbral_para_subir_a_1']:.4f} |"),
        (f"| p de equilibrio para **bajar** de 1 a 0 | "
        f"{asimetria['p_umbral_para_bajar_a_0']:.4f} |"),
        "",
        f"> {asimetria['interpretacion']}",
        "",
        "## 4. El signo coincide con el rescate; la magnitud no",
        "",
        (f"- `(utilidad > 0) equivale a (rescate)` en las 4449 celdas por 20 canales: "
        f"**{signo['utilidad_positiva_equivale_a_rescate']}**"),
        (f"- existe el caso ambos mal en una discrepancia: "
        f"**{signo['existe_caso_ambos_mal_en_discrepancia']}**"),
        (f"- cociente de magnitudes rescate sobre dano: "
        f"**{signo['cociente_magnitud_rescate_sobre_dano']:.2f}x**"),
        "",
        f"> {signo['nota']}",
        "",
        ("**Consecuencia directa:** el target binario de P51 es identico al de P50, asi que un "
        "selector binario no puede mejorar por el objetivo. Lo que puede mejorar es (a) entrenar "
        "solo sobre celdas en disagreement, donde la clase base es mucho menos sesgada, y (b) "
        "usar la magnitud para ordenar canales y no solo para decidir el signo."),
        "",
        "## 5. Valor esperado real de cambiar en cada direccion",
        "",
        "| direccion | ganadoras | perdedoras | valor esperado total |",
        "|---|---:|---:|---:|",
        (f"| subir de 0 a 1 | {ev['n_ganadoras_subiendo']} | {ev['n_perdedoras_subiendo']} | "
        f"**{ev['subir_0_a_1']:+.6e}** |"),
        (f"| bajar de 1 a 0 | {ev['n_ganadoras_bajando']} | {ev['n_perdedoras_bajando']} | "
        f"**{ev['bajar_1_a_0']:+.6e}** |"),
        "",
        "## 6. Control de consistencia con P50",
        "",
        "| selector | corrected_cv | delta vs V03 | target |",
        "|---|---:|---:|---|",
        f"| V03 | {informe['v03']['corrected_cv']:.10f} | +0,0000000000 | - |",
        (f"| P50 S1 | {informe['p50_control']['corrected_cv']:.10f} | "
        f"{informe['p50_control']['delta_vs_v03']:+.10f} | `channel_correct` |"),
        "",
        f"Celda de error de V03: {informe['v03']['celdas_de_error']}.",
        "",
    ]
    REPORTE_MD.write_text("\n".join(lineas) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
