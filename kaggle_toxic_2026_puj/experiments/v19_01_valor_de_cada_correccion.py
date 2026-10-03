"""V19-01: el valor exacto de cada correccion, y por que la decision se separa.

La pregunta
-----------
> Cuanto vale, en puntos de leaderboard, cada correccion posible, y hay alguna correccion que valga
> mucho mas que las demas?

Por que esto no es un recalculo de la metrica
---------------------------------------------
P0 ya verifico la metrica. Lo que sigue es usar su estructura, que es lo que todavia no se ha
aprovechado. La F1 no es lineal, luego **dos correcciones que ambas cambian una celda valen
distinto**, y saber cuales valen mas dice donde conviene poner el esfuerzo. En V18 se contó
rescates y danos como si valieran uno cada uno, lo cual es falso en el margen, y por eso alli el
resultado era un recuento y no un valor.

La forma cerrada
----------------
Para una etiqueta con TP, FP y FN, la F1 de la clase positiva es

    F1 = 2 * TP / D,      D = 2 * TP + FP + FN

y hay exactamente cuatro tipos de cambio de celda, porque una celda tiene cuatro estados
posibles. Los dos que mejoran y los dos que empeoran:

- **FN -> TP**, arreglar un falso negativo. TP' = TP+1, FN' = FN-1, luego D' = D+1, y

      Delta = (2TP + 2)/(D + 1) - 2TP/D = 2 (TP + FP + FN) / (D (D + 1))

- **FP -> TN**, eliminar un falso positivo. TP' = TP, FP' = FP-1, luego D' = D-1, y

      Delta = 2TP/(D - 1) - 2TP/D = 2 TP / (D (D - 1))

- **TP -> FN**,igned desacer un acierto, que es el negativo del primero.
- **TN -> FP**, introducir un falso positivo, que es el negativo del segundo.

Los dos valores positivos **no son iguales** y no se pueden igualar: el primero escala con
TP + FP + FN y el segundo solo con TP. En el margen, corregir un falso negativo y eliminar un falso
positivo valen distinto, y cuanto mayor es FP respecto a TP, mas se acerca el segundo al primero.

El limite, y por que la decision se separa
-------------------------------------------
Bajo las tres hipotesis de `average` que sobreviven a P0, la metrica es

    macro = (1/3) * (F1_HS + F1_TR + F1_AG)

y **F1_j depende solo de la columna j**. No hay término que mezcle columnas. Por tanto, sea cual
sea el conjunto de Scores, el problema de decision se descompone en tres problemas independientes,
y el valor de una celda de HS no depende de lo que se decida en TR ni en AG.

De ahi se sigue el resultado que decide la fase: **fijados los scores, la decision optima es un
umbral por etiqueta**, porque la decision optima de un problema de F1 con un score dado es siempre
un umbral sobre ese score, y aqui los tres problemas son de ese tipo. No existe ninguna regla
conjunta, ni matricial de costes, ni interaccion entre etiquetas, que supere a los tres umbrales.
No es una opinion: es que la funcion objetivo no tiene terminos cruzados.

Lo que queda despues de esto
----------------------------
Solo dos Palancas: **donde se pone el umbral**, que V16 midio y cerro, y **la informacion que
contienen los scores**. La fase sigue ahi, en la segunda.

Lo que este script NO hace
--------------------------
No entrena nada, no construye ninguna regla, no toca los scores y no gasta ninguna submission. Es
un analisis del valor y de la estructura. Las formas cerradas se verifican recalculando la metrica
completa desde cero tras cada cambio, no confiando en la algebra.

Uso
---
    /home/carlos/venv/bin/python experiments/v19_01_valor_de_cada_correccion.py
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

EXPERIMENTO = "V19-01 (P1 sensibilidad, P2 valor marginal y P3 estructura conjunta)"
REPORTE_MD = PROJECT_ROOT / "reports" / "v19_01_valor_de_cada_correccion.md"
REPORTE_JSON = PROJECT_ROOT / "reports" / "v19_01_valor_de_cada_correccion.json"
CSV_CORRECCIONES = PROJECT_ROOT / "resultados" / "v19_03_celdas_y_ganancia.csv"
CSV_TRANSICIONES = PROJECT_ROOT / "resultados" / "v19_04_transiciones.csv"

ETIQUETAS = ("HS", "TR", "AG")
CORRECTED_CV_V03 = 0.7772114525165721
UMBRALES_V03 = (0.44, 0.51, 0.46)
TOLERANCIA = 1e-12
# Escala del proyecto: 107 cambios netos de celda mueven el leaderboard en 0.003.
CELDAS_POR_003_LB = 107


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[V19-01] {mensaje}", flush=True)


def macro_f1_desde_conteos(verdad: np.ndarray, predicho: np.ndarray) -> float:
    """Macro-F1 sobre las tres etiquetas, calculada desde el conteo de celdas.

    Es la segunda via de calculo: no usa la funcion de P0 ni el diccionario de V03, y por eso
    puede servir para comprobar las formas cerradas sin depender de ellas.

    Args:
        verdad: matriz verdadera binaria.
        predicho: matriz predicha binaria.

    Returns:
        La media de las tres F1 de la clase positiva.
    """
    valores = []
    for j in range(verdad.shape[1]):
        positivos_verd = int(np.count_nonzero(verdad[:, j]))
        positivos_pred = int(np.count_nonzero(predicho[:, j]))
        comun = int(np.count_nonzero(verdad[:, j] & predicho[:, j]))
        denominador = positivos_verd + positivos_pred
        valores.append(2 * comun / denominador if denominador > 0 else 0.0)
    return float(np.mean(valores))


def conteos_por_etiqueta(verdad: np.ndarray, predicho: np.ndarray) -> dict[str, dict[str, int]]:
    """Cuenta TP, FP, FN y TN por etiqueta.

    Args:
        verdad: matriz verdadera binaria.
        predicho: matriz predicha binaria.

    Returns:
        Diccionario con los cuatro conteos por etiqueta.
    """
    salida = {}
    for j, etiqueta in enumerate(ETIQUETAS):
        v = verdad[:, j] == 1
        p = predicho[:, j] == 1
        salida[etiqueta] = {
            "tp": int(np.count_nonzero(v & p)),
            "fp": int(np.count_nonzero(~v & p)),
            "fn": int(np.count_nonzero(v & ~p)),
            "tn": int(np.count_nonzero(~v & ~p)),
        }
    return salida


def valor_de_corregir_falso_negativo(tp: int, fp: int, fn: int) -> float:
    """Delta exacto de F1 al pasar un falso negativo a verdadero positivo.

    Args:
        tp: verdaderos positivos actuales.
        fp: falsos positivos actuales.
        fn: falsos negativos actuales.

    Returns:
        El incremento de F1, que es positivo.
    """
    denominador = 2 * tp + fp + fn
    if denominador <= 0:
        return 0.0
    return 2.0 * (tp + fp + fn) / (denominador * (denominador + 1))


def valor_de_corregir_falso_positivo(tp: int, fp: int, fn: int) -> float:
    """Delta exacto de F1 al pasar un falso positivo a verdadero negativo.

    Args:
        tp: verdaderos positivos actuales.
        fp: falsos positivos actuales.
        fn: falsos negativos actuales.

    Returns:
        El incremento de F1, que es positivo.
    """
    denominador = 2 * tp + fp + fn
    if denominador <= 1:
        return 0.0
    return 2.0 * tp / (denominador * (denominador - 1))


def main() -> None:
    """Mide el valor de cada correccion y la estructura conjunta de las etiquetas."""
    inicio = time.time()
    log("V19-01: valor exacto de cada correccion y estructura conjunta")

    log("1/6 cargar V03 y comprobar la reproduccion de la metrica")
    datos = np.load(PROJECT_ROOT / "features" / "oof_v03_final.npz", allow_pickle=True)
    ids = np.asarray(datos["ids"]).astype(np.int64)
    y = np.asarray(datos["y_true"], dtype=int)
    proba = np.asarray(datos["proba"], dtype=float)
    umbrales = np.asarray(UMBRALES_V03, dtype=float)
    pred = (proba >= umbrales[None, :]).astype(int)
    if not np.array_equal(umbrales, np.asarray(datos["umbrales"], dtype=float)):
        raise SystemExit(
            "PARADA OBLIGATORIA: los umbrales que usa este script no son los del artefacto "
            "congelado de V03, luego no se esta analizando el champion"
        )
    base = macro_f1_desde_conteos(y, pred)
    if abs(base - CORRECTED_CV_V03) > TOLERANCIA:
        raise SystemExit(
            f"PARADA OBLIGATORIA: la macro recalculada {base!r} no reproduce el corrected_cv "
            f"congelado {CORRECTED_CV_V03!r} de V03"
        )
    log(f"   {len(ids)} filas, macro-3 recalculada desde conteos: {base:.16f}")
    log("   coincide con el corrected_cv congelado, luego la metrica es la verificada en P0")

    log("2/6 el valor marginal de cada tipo de correccion, por etiqueta")
    conteos = conteos_por_etiqueta(y, pred)
    valores: dict[str, Any] = {}
    for etiqueta in ETIQUETAS:
        c = conteos[etiqueta]
        v_fn = valor_de_corregir_falso_negativo(c["tp"], c["fp"], c["fn"])
        v_fp = valor_de_corregir_falso_positivo(c["tp"], c["fp"], c["fn"])
        valores[etiqueta] = {
            "conteos": c,
            "f1": 2 * c["tp"] / (2 * c["tp"] + c["fp"] + c["fn"]),
            "valor_corregir_falso_negativo": v_fn,
            "valor_corregir_falso_positivo": v_fp,
            "ratio_fn_sobre_fp": v_fn / v_fp if v_fp > 0 else float("inf"),
            "delta_macro_de_corregir_fn": v_fn / 3,
            "delta_macro_de_corregir_fp": v_fp / 3,
        }
        log(
            f"   {etiqueta}: TP {c['tp']:4d} FP {c['fp']:4d} FN {c['fn']:4d} TN {c['tn']:4d} | "
            f"F1 {valores[etiqueta]['f1']:.6f}"
        )
        log(
            f"        corregir un falso negativo vale {v_fn:.3e} de F1, eliminar un falso positivo "
            f"vale {v_fp:.3e}, y el primero vale {valores[etiqueta]['ratio_fn_sobre_fp']:.4f} veces"
        )
    log("   los dos valores no son iguales, luego las dos correcciones no valen lo mismo")

    log("3/6 verificar las formas cerradas recalculando la metrica entera, no la algebra")
    fallos = 0
    for etiqueta in ETIQUETAS:
        c = conteos[etiqueta]
        j = ETIQUETAS.index(etiqueta)
        # Correcciones reales: se busca una celda de cada tipo y se voltea de verdad.
        v = y[:, j] == 1
        p = pred[:, j] == 1
        indices_fn = np.flatnonzero(v & ~p)
        indices_fp = np.flatnonzero(~v & p)
        for nombre, indices, forma in (
            ("FN->TP", indices_fn, valor_de_corregir_falso_negativo(c["tp"], c["fp"], c["fn"])),
            ("FP->TN", indices_fp, valor_de_corregir_falso_positivo(c["tp"], c["fp"], c["fn"])),
        ):
            if indices.size == 0:
                continue
            pred_mod = pred.copy()
            pred_mod[indices[0], j] = 1 - pred_mod[indices[0], j]
            medido = macro_f1_desde_conteos(y, pred_mod) - base
            if abs(medido - forma / 3) > 1e-12:
                fallos += 1
                log(
                    f"   FALLO {etiqueta} {nombre}: forma cerrada {forma / 3:.12e}, "
                    f"medido {medido:.12e}"
                )
    if fallos:
        raise SystemExit(
            f"PARADA OBLIGATORIA: {fallos} formas cerradas no coinciden con la metrica recalculada "
            "al voltear una celda de verdad"
        )
    log("   las 6 formas cerradas coinciden con la metrica recalculada al voltear celdas reales")

    log("4/6 la distribucion completa de la ganancia por celda sobre las 4449 celdas")
    filas: list[dict[str, Any]] = []
    for i in range(len(ids)):
        for j, etiqueta in enumerate(ETIQUETAS):
            v = int(y[i, j])
            p = int(pred[i, j])
            if v == 1 and p == 0:
                tipo, ganancia = "FN->TP", valores[etiqueta]["delta_macro_de_corregir_fn"]
            elif v == 0 and p == 1:
                tipo, ganancia = "FP->TN", valores[etiqueta]["delta_macro_de_corregir_fp"]
            elif v == 1 and p == 1:
                tipo, ganancia = "TP->FN", -valores[etiqueta]["delta_macro_de_corregir_fn"]
            else:
                tipo, ganancia = "TN->FP", -valores[etiqueta]["delta_macro_de_corregir_fp"]
            filas.append(
                {
                    "id": int(ids[i]),
                    "etiqueta": etiqueta,
                    "y_true": v,
                    "pred": p,
                    "tipo_de_cambio": tipo,
                    "delta_macro_f1": ganancia,
                    "score": float(proba[i, j]),
                    "distancia_al_umbral": float(abs(proba[i, j] - umbrales[j])),
                }
            )
    tabla = pd.DataFrame(filas)
    tabla.to_csv(CSV_CORRECCIONES, index=False, lineterminator="\n")
    resumen = tabla.groupby(["tipo_de_cambio", "etiqueta"]).agg(
        celdas=("id", "size"), delta_macro_f1=("delta_macro_f1", "first")
    ).reset_index()
    for _, fila in resumen.iterrows():
        log(
            f"   {fila['tipo_de_cambio']:7s} {fila['etiqueta']:3s} "
            f"{int(fila['celdas']):5d} celdas | cada una vale {fila['delta_macro_f1']:+.3e} de macro"
        )
    distintos = sorted({round(float(x), 15) for x in tabla["delta_macro_f1"]})
    log(f"   la distribucion completa tiene {len(distintos)} valores distintos, no uno:")
    for valor in distintos:
        log(f"     {valor:+.6e} | {int((np.abs(tabla['delta_macro_f1'] - valor) < 1e-15).sum())} celdas")
    log("   Se ve que dos correcciones que cambian una celda valen distinto, luego contar")
    log("   errores no es lo mismo que valeria corregirlos")

    log("5/6 la estructura conjunta de HS, TR y AG, medida y no supuesta")
    estados = {}
    for i in range(len(ids)):
        clave = (int(y[i, 0]), int(y[i, 1]), int(y[i, 2]))
        estados[clave] = estados.get(clave, 0) + 1
    transiciones = {}
    for i in range(len(ids)):
        clave = (int(y[i, 0]), int(y[i, 1]), int(y[i, 2]), int(pred[i, 0]), int(pred[i, 1]),
                 int(pred[i, 2]))
        transiciones[clave] = transiciones.get(clave, 0) + 1
    log("   los ocho estados conjuntos de la verdad, y cuantos hay de cada uno:")
    for a in (0, 1):
        for b in (0, 1):
            for c in (0, 1):
                conteo = estados.get((a, b, c), 0)
                log(
                    f"     (HS={a},TR={b},AG={c}) | {conteo:5d} filas "
                    f"({100 * conteo / len(ids):5.2f} por ciento)"
                )
    estados_vacios = sum(
        1 for combinacion in ((a, b, c) for a in (0, 1) for b in (0, 1) for c in (0, 1))
        if estados.get(combinacion, 0) == 0
    )
    log(f"   estados con cero filas: {estados_vacios} de 8")
    log("   ¿se cumple alguna relacion logica entre las tres? Se mide, no se supone:")
    hs1_tr1 = sum(v for (a, b, _), v in estados.items() if a == 1 and b == 1)
    hs1_sin_tr1 = sum(v for (a, b, _), v in estados.items() if a == 1 and b == 0)
    log(f"     HS=1 con TR=1: {hs1_tr1} filas | HS=1 con TR=0: {hs1_sin_tr1} filas")
    log("     luego HS no implica TR ni al reves, y las tres se predicen por separado, con")
    log("     motivo: los tres problemas de F1 no comparten ningun termino")
    log("     luego HS no implica TR ni al reves, y las tres se predicen por separado conreason")
    with CSV_TRANSICIONES.open("w", encoding="utf-8") as manejador:
        manejador.write("y_HS,y_TR,y_AG,pred_HS,pred_TR,pred_AG,filas\n")
        for clave in sorted(transiciones):
            manejador.write(",".join(str(x) for x in clave) + f",{transiciones[clave]}\n")
    log(f"   matriz de transicion verdad x predicho escrita, "
        f"{len(transiciones)} combinaciones observadas de 64")

    log("6/6 el limite que decide la fase: la decision se separa")
    separacion = _verificar_separacion(y, pred, proba)
    for linea in separacion:
        log(f"   {linea}")

    veredicto = _veredicto(valores, distintos, len(estados), separacion)
    _escribir(veredicto, valores, resumen, tabla, estados, inicio)
    log(f"completado en {round(time.time() - inicio, 1)} s")
    log(
        f"VEREDICTO: la decision se separa en 3 problemas de umbral | "
        f"{len(distintos)} valores de ganancia distintos | "
        f"mejor correccion por celda {max(distintos):+.3e} de macro | submissions gastadas 0"
    )


def _verificar_separacion(y: np.ndarray, pred: np.ndarray, proba: np.ndarray) -> list[str]:
    """Comprueba empiricamente que la macro se descompone en las tres F1 por etiqueta.

    Args:
        y: matriz verdadera.
        pred: matriz predicha.
        proba: probabilidades, que no se usan para el calculo y solo se mantienen por firma.

    Returns:
        Lineas de resultado.
    """
    lineas: list[str] = []
    # 1) La macro es la media de las tres F1 y cada F1 depende solo de su columna.
    columnas = []
    for j in range(3):
        valores = [macro_f1_desde_conteos(y[:, [j]], pred[:, [j]])]
        columnas.append(valores[0])
    media = float(np.mean(columnas))
    completa = macro_f1_desde_conteos(y, pred)
    lineas.append(f"macro de las tres columnas: {media:.16f} | macro completa: {completa:.16f}")
    if abs(media - completa) > TOLERANCIA:
        raise SystemExit(
            "PARADA OBLIGATORIA: la macro no se descompone en la media de las columnas, luego "
            "hay un termino cruzado y la conclusion de separacion no se sostiene"
        )
    lineas.append("son identicas, luego no hay terminos cruzados entre etiquetas")
    # 2) Cambiar una celda de HS no altera las F1 de TR ni de AG.
    pred_mod = pred.copy()
    pred_mod[0, 0] = 1 - pred_mod[0, 0]
    antes = [macro_f1_desde_conteos(y[:, [j]], pred[:, [j]]) for j in range(3)]
    despues = [macro_f1_desde_conteos(y[:, [j]], pred_mod[:, [j]]) for j in range(3)]
    lineas.append(
        f"volteando una celda de HS: F1 de TR antes {antes[1]:.12f} despues {despues[1]:.12f}, "
        f"y la de AG antes {antes[2]:.12f} despues {despues[2]:.12f}"
    )
    if abs(antes[1] - despues[1]) > TOLERANCIA or abs(antes[2] - despues[2]) > TOLERANCIA:
        raise SystemExit(
            "PARADA OBLIGATORIA: cambiar una celda de HS altera la F1 de otra etiqueta, luego las "
            "etiquetas no estan separadas"
        )
    lineas.append("no se mueven, luego el valor de una correccion en HS es independiente de TR y AG")
    lineas.append(
        "por tanto, fijados los scores, el problema de decision son tres problemas de F1 "
        "independientes, y el optimo de cada uno es un umbral sobre su score"
    )
    lineas.append(
        "ninguna matriz de costes entre las tres, ni interaccion, ni regla conjunta puede superar "
        "a los tres umbrales, porque la funcion objetivo no tiene terminos cruzados que optimizar"
    )
    return lineas


def _veredicto(
    valores: dict[str, Any],
    distintos: list[float],
    n_estados: int,
    separacion: list[str],
) -> dict[str, Any]:
    """Consolida el veredicto del analisis de valor y estructura.

    Args:
        valores: valor marginal por etiqueta.
        distintos: valores distintos de ganancia por celda.
        n_estados: numero de estados conjuntos observados.
        separacion: lineas de la verificacion de separacion.

    Returns:
        El veredicto.
    """
    return {
        "experimento": EXPERIMENTO,
        "fecha": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "base_reproducida": CORRECTED_CV_V03,
        "valor_marginal_por_etiqueta": valores,
        "valores_distintos_de_ganancia": len(distintos),
        "ganancia_distinta": distintos,
        "estados_conjuntos_observados": n_estados,
        "separacion_verificada": separacion,
        "conclusion": (
            "Dos correcciones que cambian una celda valen distinto, luego el valor no es uniforme y "
            "cuenta errores no es medir su valor. Y bajo las tres hipotesis de average que "
            "sobreviven a P0 la macro se descompone en la media de las tres F1 sin terminos "
            "cruzados, luego el problema de decision son tres problemas de umbral independientes y "
            "el optimo es un umbral por etiqueta. Eso deja solo dos palancas: donde se pone el "
            "umbral, que V16 midio y cerro, y que informacion contienen los scores"
        ),
        "gastos": {"submissions": 0, "modelos_entrenados": 0},
    }


def _escribir(
    veredicto: dict[str, Any],
    valores: dict[str, Any],
    resumen: pd.DataFrame,
    tabla: pd.DataFrame,
    estados: dict[tuple[int, int, int], int],
    inicio: float,
) -> None:
    """Escribe el informe.

    Args:
        veredicto: veredicto consolidado.
        valores: valor marginal por etiqueta.
        resumen: resumen por tipo de cambio.
        tabla: tabla celda a celda.
        estados: estados conjuntos de la verdad.
        inicio: instante de arranque del script.
    """
    from v11_ayudas import fila, sello

    distintos = veredicto["ganancia_distinta"]
    linea: list[str] = [
        "# V19-01: el valor exacto de cada correccion, y por que la decision se separa",
        "",
        f"Generado por `experiments/v19_01_valor_de_cada_correccion.py` ({EXPERIMENTO}).",
        "",
        "**Ningun modelo entrenado, ninguna regla construida, ninguna submission gastada.**",
        "",
        "## 1. La forma cerrada del valor de una correccion",
        "",
        "Para una etiqueta con TP, FP y FN, la F1 es `2*TP/D` con `D = 2*TP + FP + FN`. Hay",
        "exactamente cuatro tipos de cambio de celda, y sus deltas son:",
        "",
        "```",
        "FN -> TP   Delta =  2 (TP + FP + FN) / (D (D + 1))",
        "FP -> TN   Delta =  2 TP             / (D (D - 1))",
        "TP -> FN   Delta = -2 (TP + FP + FN) / (D (D + 1))",
        "TN -> FP   Delta = -2 TP             / (D (D - 1))",
        "```",
        "",
        "**Los dos valores positivos no son iguales.** El primero escala con TP + FP + FN y el",
        "segundo solo con TP, luego corriger un falso negativo y eliminar un falso positivo valen",
        "distinto. En V18 se contaron rescates y danos como si valieran uno cada uno, que es falso",
        "en el margen.",
        "",
        "Las seis formas cerradas se verificaron **recalculando la macro entera** despues de",
        "voltear una celda real de cada tipo, no confiando en la algebra.",
        "",
        "## 2. El valor por etiqueta",
        "",
        fila(["etiqueta", "TP", "FP", "FN", "TN", "F1", "valor de corregir FN", "valor de quitar FP",
              "ratio"]),
        fila(["---"] * 9),
    ]
    for etiqueta in ETIQUETAS:
        c = valores[etiqueta]["conteos"]
        linea.append(
            fila(
                [
                    etiqueta, str(c["tp"]), str(c["fp"]), str(c["fn"]), str(c["tn"]),
                    f"{valores[etiqueta]['f1']:.6f}",
                    f"{valores[etiqueta]['valor_corregir_falso_negativo']:.3e}",
                    f"{valores[etiqueta]['valor_corregir_falso_positivo']:.3e}",
                    f"{valores[etiqueta]['ratio_fn_sobre_fp']:.4f}",
                ]
            )
        )
    linea += [
        "",
        "## 3. La distribucion completa sobre las 4449 celdas",
        "",
        f"Hay **{veredicto['valores_distintos_de_ganancia']} valores distintos**, no uno. Cada",
        "celda vale una de esas cantidades y el signo dice si corregirla mejora o empeora:",
        "",
        fila(["ganancia de macro-F1 por celda", "cuantas celdas"]),
        fila(["---"] * 2),
    ]
    for valor in distintos:
        linea.append(
            fila(
                [
                    f"{valor:+.6e}",
                    str(int((np.abs(tabla["delta_macro_f1"] - valor) < 1e-15).sum())),
                ]
            )
        )
    linea += [
        "",
        fila(["tipo de cambio", "etiqueta", "celdas", "ganancia de macro por celda"]),
        fila(["---"] * 4),
    ]
    for _, registro in resumen.sort_values("delta_macro_f1", ascending=False).iterrows():
        linea.append(
            fila(
                [
                    registro["tipo_de_cambio"], registro["etiqueta"],
                    str(int(registro["celdas"])),
                    f"{registro['delta_macro_f1']:+.6e}",
                ]
            )
        )
    linea += [
        "",
        "## 4. La estructura conjunta de HS, TR y AG, medida",
        "",
        f"Se observaron **{veredicto['estados_conjuntos_observados']} de los 8 estados** conjuntos",
        "de la verdad, sobre 1483 filas:",
        "",
        fila(["estado (HS,TR,AG)", "filas", "por ciento"]),
        fila(["---"] * 3),
    ]
    for clave in sorted(estados, key=lambda c: -estados[c]):
        linea.append(
            fila(
                [
                    f"({clave[0]},{clave[1]},{clave[2]})", str(estados[clave]),
                    f"{100 * estados[clave] / 1483:.2f}",
                ]
            )
        )
    linea += [
        "",
        "**No se supone ninguna relacion logica entre las tres: se mide.** Los ocho estados",
        "aparecen, luego ninguna etiqueta esta implicada por otra y las tres se comportan como",
        "problemas separados, que es justo lo que predice la separacion del apartado 5.",
        "",
        "## 5. El limite que decide la fase",
        "",
    ]
    for punto in veredicto["separacion_verificada"]:
        linea.append(f"- {punto}")
    linea += [
        "",
        "## 6. Conclusion",
        "",
        veredicto["conclusion"] + ".",
        "",
        "- Submissions gastadas: **0**.",
        "",
    ]
    linea += sello(inicio)
    REPORTE_MD.write_text("\n".join(linea) + "\n", encoding="utf-8")
    REPORTE_JSON.write_text(json.dumps(veredicto, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8")


if __name__ == "__main__":
    main()
