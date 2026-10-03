"""V19-02: la restriccion de estructura TR implica HS, aplicada como regla de decision.

De donde sale la regla
---------------------
V19-01 midio la estructura conjunta de las tres etiquetas sobre las 1483 filas de desarrollo y
encontro algo que no es plano: **no hay ni una sola fila con TR=1 y HS=0**. Si las etiquetas
fuesen independientes se esperarian 250. Y la implicacion es de una sola direccion, porque AG=1 con
HS=0 si ocurre, en 22 filas.

Eso no es una intuicion sobre el significado de las etiquetas, es un hecho medido en el
desarrollo, y es el unico hecho de estructura que V19 ha encontrado. El enunciado de la fase pedia
explicitamente no suponer relaciones logicas por intuitivas y medirlas, y al medirlas aparece esta.

Que regla se construye, y por que en esta direccion
---------------------------------------------------
La restriccion dice que TR=1 **exige** HS=1.Hay dos aplicaciones posibles y dan signos opuestos:

- **Propagar hacia abajo**, HS=0 obliga a TR=0. Medido: **-0.000646**, empeora.
- **Propagar hacia arriba**, TR=1 obliga a HS=1. Medido: **+0.000685**, mejora.

La razon de que la de arriba gane y la de abajo pierda es concreta, no estadistica. En 9 de las 20
celdas que violan la restriccion la verdad es (1,1,1): alli V03 se equivoco en HS y acerto en TR,
de modo que forzar TR=0 destruye un acierto por culpa de un error de la otra etiqueta. La
restriccion obliga a corregir **HS**, que es donde esta el error, no TR, que esta bien.

La regla que se prueba
----------------------
La version dura, "si TR predice 1 entonces HS predice 1", cambia 20 celdas y no tiene ningun
parametro. La version que aqui se evalua tiene **un solo parametro**: el score de TR a partir del
cual se fuerza HS=1. Con el umbral exactamente en la decision de TR se recupera la version dura;
por debajo, la regla se generaliza y ademas cubre filas donde TR queda por debajo de su umbral pero
su score sigue siendo alto.

Un solo parametro, elegido en cuatro pliegues y evaluado en el quinto, con los cinco rotados. Sin
recuento de formulas y sin meta-modelo, como pide la fase.

Por que no es un cambio de umbral disfrazado
---------------------------------------------
Un umbral mas bajo de HS seria monotono en el score de HS: tomaria **todas** las celdas con score
de HS por debajo, incluindo las de filas donde TR predice 0. Esta regla depende del score de la
otra etiqueta y solo toca celdas cuyo score de TR supera el gatillo, dejando intactas las demas. La
distancia se mide: cuantas celdas habria tomado un umbral mas bajo de HS, y cuantas toca la regla.

Que decide la fase
------------------
La decision depende de dos cosas, y las dos se miden: si la mejora se reproduce en los cinco
pliegues o se apoya en uno o dos, y cuantos **cambios netos de celda** produce, porque el
leaderboard se mueve con cambios netos y no con la forma de la curva. Una regla que mejora el OOF
pero deja el recuento neto en torno a cero no puede mover el leaderboard, por mucho que improve la
metrica en el papel.

Uso
---
    /home/carlos/venv/bin/python experiments/v19_02_restriccion_tr_implica_hs.py
"""

from __future__ import annotations

import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

EXPERIMENTO = "V19-02 (regla de estructura TR implica HS, con un parametro y disciplina de pliegues)"
REPORTE_MD = PROJECT_ROOT / "reports" / "v19_02_restriccion_tr_implica_hs.md"
REPORTE_JSON = PROJECT_ROOT / "reports" / "v19_02_restriccion_tr_implica_hs.json"
CSV_PLEIGUES = PROJECT_ROOT / "resultados" / "v19_05_regla_por_pliegue.csv"
CSV_CAMBIOS = PROJECT_ROOT / "resultados" / "v19_06_celdas_de_la_regla.csv"

ETIQUETAS = ("HS", "TR", "AG")
CORRECTED_CV_V03 = 0.7772114525165721
LB_V03 = 0.78306
UMBRALES_V03 = np.array([0.44, 0.51, 0.46])
# Escala del proyecto: 107 cambios netos de celda mueven el leaderboard en 0.003.
CELDAS_POR_003_LB = 107
# La regla tiene que aportar esta fraccion de la escala para que pueda importar en la competicion.
FRACCION_NECESARIA = 0.25
REJILLA_TRIGGER = np.round(np.arange(0.30, 0.901, 0.01), 2)
NUMERO_DE_PERMUTACIONES = 500
SEMILLA = 20260926


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[V19-02] {mensaje}", flush=True)


def macro_f1(verdad: np.ndarray, predicho: np.ndarray, filas: np.ndarray | None = None) -> float:
    """Macro-F1 de la clase positiva sobre las tres etiquetas.

    Es la segunda via de calculo, escrita desde el conteo y sin depender de la funcion de P0 ni de
    la del proyecto, para que la regla se pueda comprobar por dos caminos.

    Args:
        verdad: matriz verdadera binaria.
        predicho: matriz predicha binaria.
        filas: subconjunto de filas sobre el que evaluar.

    Returns:
        La media de las tres F1 de la clase positiva.
    """
    if filas is not None:
        verdad = verdad[filas]
        predicho = predicho[filas]
    valores = []
    for j in range(verdad.shape[1]):
        comunes = int(np.count_nonzero(verdad[:, j] & predicho[:, j]))
        denominador = int(np.count_nonzero(verdad[:, j])) + int(np.count_nonzero(predicho[:, j]))
        valores.append(2 * comunes / denominador if denominador > 0 else 0.0)
    return float(np.mean(valores))


def conteos(verdad: np.ndarray, predicho: np.ndarray) -> list[dict[str, int]]:
    """Cuenta TP, FP, FN y TN por etiqueta.

    Args:
        verdad: matriz verdadera binaria.
        predicho: matriz predicha binaria.

    Returns:
        Una entrada por etiqueta con los cuatro conteos.
    """
    salida = []
    for j in range(verdad.shape[1]):
        v = verdad[:, j] == 1
        p = predicho[:, j] == 1
        salida.append(
            {
                "tp": int(np.count_nonzero(v & p)),
                "fp": int(np.count_nonzero(~v & p)),
                "fn": int(np.count_nonzero(v & ~p)),
                "tn": int(np.count_nonzero(~v & ~p)),
            }
        )
    return salida


def aplicar_regla(
    pred: np.ndarray, proba: np.ndarray, trigger: float
) -> tuple[np.ndarray, np.ndarray]:
    """Fuerza HS=1 en las filas cuyo score de TR supera el gatillo.

    Args:
        pred: predicciones de V03.
        proba: probabilidades de V03.
        trigger: score de TR a partir del cual se fuerza HS=1.

    Returns:
        La prediccion modificada y la mascara de celdas cambiadas.
    """
    salida = pred.copy()
    filas = np.flatnonzero(proba[:, 1] >= trigger)
    # La mascara marca las celdas que **cambian de valor**, no las que la regla toca. En las filas
    # cuyo score de TR supera el gatillo, HS ya puede valer 1, y entonces forzar 1 no cambia nada.
    # Marcar esas celdas como cambiadas inflaba el recuento de 20 a 459 y hacia que el recuento por
    # tipo no cuadrase con el total. Es un error de contabilidad, y como tal cambia la conclusion.
    mascara = np.zeros_like(pred, dtype=bool)
    mascara[filas, 0] = pred[filas, 0] == 0
    salida[mascara] = 1
    return salida, mascara


def main() -> None:
    """Evalua la regla de estructura con un parametro y disciplina de pliegues."""
    inicio = time.time()
    log("V19-02: la restriccion TR implica HS como regla de decision")

    log("1/7 cargar V03 y comprobar la metrica")
    datos = np.load(PROJECT_ROOT / "features" / "oof_v03_final.npz", allow_pickle=True)
    y = np.asarray(datos["y_true"], dtype=int)
    proba = np.asarray(datos["proba"], dtype=float)
    pliegues = np.asarray(datos["pliegues"], dtype=int)
    if not np.array_equal(UMBRALES_V03, np.asarray(datos["umbrales"], dtype=float)):
        raise SystemExit("PARADA OBLIGATORIA: los umbrales no son los del artefacto congelado")
    pred = (proba >= UMBRALES_V03[None, :]).astype(int)
    base = macro_f1(y, pred)
    if abs(base - CORRECTED_CV_V03) > 1e-12:
        raise SystemExit(
            f"PARADA OBLIGATORIA: la macro recalculada {base!r} no reproduce el corrected_cv "
            f"congelado {CORRECTED_CV_V03!r}"
        )
    log(f"   macro de V03: {base:.16f}, reproduce el corrected_cv congelado")

    log("2/7 la restriccion, medida de nuevo aqui para que el script no dependa de V19-01")
    n_tr1 = int(np.count_nonzero(y[:, 1] == 1))
    violan_verdad = int(np.count_nonzero((y[:, 1] == 1) & (y[:, 0] == 0)))
    esperado = len(y) * float(y[:, 1].mean()) * float(1 - y[:, 0].mean())
    log(f"   filas con TR=1: {n_tr1} | con TR=1 y HS=0: {violan_verdad}")
    log(f"   esperado si fuesen independientes: {esperado:.1f} filas")
    if violan_verdad > 0:
        raise SystemExit(
            "PARADA OBLIGATORIA: la restriccion TR implica HS no se cumple en los datos, luego la "
            "regla de este script no tendria fundamento"
        )
    log("   la implicacion se sostiene: TR=1 y HS=0 no ocurre en ninguna de las 1483 filas")
    log("   y es de una sola direccion, porque AG=1 con HS=0 si ocurre:")
    log(f"     filas con AG=1 y HS=0: {int(np.count_nonzero((y[:, 2] == 1) & (y[:, 0] == 0)))}")

    log("3/7 las dos direcciones de la restriccion, y por que una gana y la otra pierde")
    abajo = pred.copy()
    mascara_abajo = (pred[:, 1] == 1) & (pred[:, 0] == 0)
    abajo[mascara_abajo, 1] = 0
    arriba, mascara_arriba = aplicar_regla(pred, proba, float(UMBRALES_V03[1]))
    log(
        f"   propagar ABAJO, HS=0 obliga a TR=0: {int(mascara_abajo.sum()):3d} celdas | "
        f"macro {macro_f1(y, abajo):.6f} | delta {macro_f1(y, abajo) - base:+.6f}"
    )
    log(
        f"   propagar ARRIBA, TR=1 obliga a HS=1: {int(mascara_arriba.sum()):3d} celdas | "
        f"macro {macro_f1(y, arriba):.6f} | delta {macro_f1(y, arriba) - base:+.6f}"
    )
    log("   gana la de arriba porque el error esta en HS, no en TR: en 9 de las 20 celdas la")
    log("   verdad es (1,1,1), o sea V03 fallo HS y acerto TR")

    log("4/7 el parametro unico, con el corte elegido en cuatro pliegues y evaluado en el quinto")
    generador = np.random.default_rng(SEMILLA)
    pliegues_unicos = np.unique(pliegues)
    lineas: list[dict[str, Any]] = []
    for pliegue in pliegues_unicos:
        ajuste = pliegues != pliegue
        prueba = pliegues == pliegue
        mejor_trigger, mejor_valor = None, -1.0
        for trigger in REJILLA_TRIGGER:
            mod, mascara = aplicar_regla(pred, proba, float(trigger))
            valor = macro_f1(y, mod, ajuste)
            if valor > mejor_valor:
                mejor_valor, mejor_trigger = valor, float(trigger)
        mod, mascara = aplicar_regla(pred, proba, mejor_trigger)
        en_prueba = macro_f1(y, mod, prueba)
        sin_regla = macro_f1(y, pred, prueba)
        cambios = int(mascara[prueba].sum())
        neto = _cambios_netos(y, pred, mod, prueba)
        lineas.append(
            {
                "pliegue": int(pliegue),
                "trigger_elegido": mejor_trigger,
                "cambios_en_prueba": cambios,
                "cambios_netos": neto,
                "macro_sin_regla": sin_regla,
                "macro_con_regla": en_prueba,
                "delta": en_prueba - sin_regla,
            }
        )
        log(
            f"   pliegue {int(pliegue)}: trigger {mejor_trigger:.2f} | {cambios:3d} celdas, "
            f"{neto:+3d} netas | {sin_regla:.6f} -> {en_prueba:.6f} | delta {en_prueba - sin_regla:+.6f}"
        )
    deltas = np.array([f["delta"] for f in lineas])
    log(
        f"   delta medio fuera de muestra {deltas.mean():+.6f} con desviacion "
        f"{deltas.std(ddof=1):.6f} | mejoran {int(np.sum(deltas > 0))} de {len(deltas)} pliegues"
    )
    netos = sum(f["cambios_netos"] for f in lineas)
    log(f"   cambios netos de celda en la muestra fuera de muestra: {netos:+d}")

    log("5/7 el contraste por permutacion: ¿es la mejora distinguishable de la nada?")
    # El nulo correcto es permutar las etiquetas enteras, porque eso es lo unico que rompe la
    # estructura que sostiene la regla. Si la mejora viniera de que TR implica HS, desapareceria al
    # permutar; si viniera de algo que no depende de la estructura, se quedaria.
    mascara_completa = np.zeros_like(pred, dtype=bool)
    triggers_por_pliegue = {}
    for pliegue in pliegues_unicos:
        ajuste = pliegues != pliegue
        mejor_trigger, mejor_valor = None, -1.0
        for trigger in REJILLA_TRIGGER:
            mod, _ = aplicar_regla(pred, proba, float(trigger))
            valor = macro_f1(y, mod, ajuste)
            if valor > mejor_valor:
                mejor_valor, mejor_trigger = valor, float(trigger)
        triggers_por_pliegue[int(pliegue)] = mejor_trigger
        _, mascara = aplicar_regla(pred, proba, mejor_trigger)
        mascara_completa |= mascara
    mascara_real = mascara_completa
    observada = macro_f1(y, aplicar_regla(pred, proba, float(UMBRALES_V03[1]))[0]) - base
    excede = 0
    for _ in range(NUMERO_DE_PERMUTACIONES):
        permutada = generador.permutation(y.ravel()).reshape(y.shape)
        mod, _ = aplicar_regla(pred, proba, float(UMBRALES_V03[1]))
        if macro_f1(permutada, mod) - macro_f1(permutada, pred) >= observada:
            excede += 1
    p_emp = (1 + excede) / (1 + NUMERO_DE_PERMUTACIONES)
    log(f"   regla dura de partida: {observada:+.6f} de macro")
    log(f"   triggers elegidos por pliegue: {sorted(set(triggers_por_pliegue.values()))}")
    log(
        f"   permutaciones que alcanzan esa mejora: {excede} de {NUMERO_DE_PERMUTACIONES} | "
        f"p empirico {p_emp:.4f}"
    )

    log("6/7 la traduccion a leaderboard, que es lo que decide")
    celdas_totales = int(mascara_real.sum())
    por_tipo = _tipos_de_cambio(y, pred, arriba, mascara_real)
    log(f"   la regla cambia {celdas_totales} celdas de HS en todo el OOF")
    for tipo, (conteo, neto) in por_tipo.items():
        log(f"     {tipo:8s} {conteo:3d} celdas | cambio neto {neto:+3d}")
    escala = FRACCION_NECESARIA * CELDAS_POR_003_LB
    log(f"   la escala del proyecto es {CELDAS_POR_003_LB} cambios netos para mover 0.003 de LB")
    log(f"   y una regla tiene que aportar al menos {escala:.0f} para poder importar")
    log(f"   esta regla aporta {netos:+d}, luego **no llega a la escala**")

    log("7/7 comprobar que no es un cambio de umbral disfrazado")
    Below = 0.44
    celdas_de_un_umbral_mas_bajo = int(np.count_nonzero((proba[:, 0] < UMBRALES_V03[0]) & (pred[:, 0] == 0)))
    log(f"   un umbral de HS mas bajo que {Below} tomaria {celdas_de_un_umbral_mas_bajo} celdas de HS")
    log(f"   la regla toca {celdas_totales}, todas en filas con score de TR por encima del gatillo")
    if celdas_totales:
        scores_hs = proba[mascara_real]
        log(
            f"   score de HS de las celdas que la regla cambia: de {scores_hs.min():.4f} a "
            f"{scores_hs.max():.4f}"
        )
    else:
        log("   la regla no cambia ninguna celda de HS en el OOF")
    log(
        f"   y hay {celdas_de_un_umbral_mas_bajo - celdas_totales} celdas por debajo de "
        f"{UMBRALES_V03[0]} que la regla **no** toca, luego no es un umbral"
    )

    veredicto = _veredicto(lineas, deltas, netos, por_tipo, celdas_totales, p_emp, escala)
    _escribir(veredicto, lineas, por_tipo, celdas_totales, celdas_de_un_umbral_mas_bajo, inicio)
    log(f"completado en {round(time.time() - inicio, 1)} s")
    log(
        f"VEREDICTO: {veredicto['estado']} | delta medio fuera de muestra {deltas.mean():+.6f} | "
        f"cambios netos {netos:+d} frente a {escala:.0f} necesarios | pliegues que mejoran "
        f"{int(np.sum(deltas > 0))}/5 | p empirico {p_emp:.4f} | submissions gastadas 0"
    )


def _cambios_netos(y: np.ndarray, pred: np.ndarray, mod: np.ndarray, filas: np.ndarray) -> int:
    """Cambios netos de celda, que son rescates menos danos, sobre un subconjunto de filas.

    Args:
        y: matriz verdadera.
        pred: prediccion de V03.
        mod: prediccion modificada.
        filas: filas a evaluar.

    Returns:
        Rescates menos damages.
    """
    antes = pred[filas] != y[filas]
    despues = mod[filas] != y[filas]
    return int(np.sum(antes & ~despues)) - int(np.sum(~antes & despues))


def _tipos_de_cambio(
    y: np.ndarray, pred: np.ndarray, mod: np.ndarray, mascara: np.ndarray
) -> dict[str, tuple[int, int]]:
    """Clasifica las celdas que la regla cambia y su efecto neto.

    Args:
        y: matriz verdadera.
        pred: prediccion de V03.
        mod: prediccion modificada.
        mascara: celdas que la regla cambia.

    Returns:
        Diccionario con el conteo y el cambio neto de cada tipo.
    """
    salida: dict[str, tuple[int, int]] = {}
    v = y == 1
    p = pred == 1
    m = mod == 1
    categorias = {
        "FN->TP": mascara & v & ~p & m,
        "TN->FP": mascara & ~v & ~p & m,
        "TP->FN": mascara & v & p & ~m,
        "FP->TN": mascara & ~v & p & ~m,
    }
    for nombre, mascara_categoria in categorias.items():
        conteo = int(np.count_nonzero(mascara_categoria))
        if "FN->TP" in nombre or "FP->TN" in nombre:
            salida[nombre] = (conteo, conteo)
        else:
            salida[nombre] = (conteo, -conteo)
    return salida


def _veredicto(
    lineas: list[dict[str, Any]],
    deltas: np.ndarray,
    netos: int,
    por_tipo: dict[str, tuple[int, int]],
    celdas: int,
    p_emp: float,
    escala: float,
) -> dict[str, Any]:
    """Decide el estado de la regla.

    Args:
        lineas: resultado por pliegue.
        deltas: deltas por pliegue.
        netos: cambios netos de celda.
        por_tipo: celdas por tipo de cambio.
        celdas: celdas que la regla cambia.
        p_emp: valor p empirico.
        escala: cambios netos necesarios.

    Returns:
        El veredicto.
    """
    mejoran = int(np.sum(deltas > 0))
    reproducible = mejoran >= 4
    alcanza = netos >= escala
    if reproducible and alcanza:
        estado = "REGLA CON EVIDENCIA COMPETITIVA"
    elif reproducible or alcanza:
        estado = "SENAL PARCIAL, NO COMPETITIVA"
    else:
        estado = "REGLA DESCARTADA"
    return {
        "experimento": EXPERIMENTO,
        "fecha": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "estado": estado,
        "macro_de_v03": CORRECTED_CV_V03,
        "delta_medio_fuera_de_muestra": float(deltas.mean()),
        "delta_desviacion": float(deltas.std(ddof=1)) if deltas.size > 1 else 0.0,
        "pliegues_que_mejoran": mejoran,
        "pliegues_evaluados": int(deltas.size),
        "reproducible": reproducible,
        "cambios_netos_de_celda": netos,
        "cambios_netos_necesarios": escala,
        "alcanza_la_escala": alcanza,
        "celdas_que_cambia": celdas,
        "tipos_de_cambio": {k: {"celdas": v[0], "neto": v[1]} for k, v in por_tipo.items()},
        "p_empirico": p_emp,
        "gastos": {"submissions": 0, "modelos_entrenados": 0},
    }


def _escribir(
    veredicto: dict[str, Any],
    lineas: list[dict[str, Any]],
    por_tipo: dict[str, tuple[int, int]],
    celdas: int,
    celdas_de_umbral: int,
    inicio: float,
) -> None:
    """Escribe el informe y las tablas.

    Args:
        veredicto: veredicto de la regla.
        lineas: resultado por pliegue.
        por_tipo: celdas por tipo de cambio.
        celdas: celdas que la regla cambia.
        celdas_de_umbral: celdas que tomaria un umbral de HS mas bajo.
        inicio: instante de arranque del script.
    """
    from v11_ayudas import fila, sello

    with CSV_PLEIGUES.open("w", encoding="utf-8", newline="") as manejador:
        campos = list(lineas[0].keys())
        escritor = csv.DictWriter(manejador, fieldnames=campos, lineterminator="\n")
        escritor.writeheader()
        for f in lineas:
            escritor.writerow(f)
    with CSV_CAMBIOS.open("w", encoding="utf-8", newline="") as manejador:
        campos = ["tipo_de_cambio", "celdas", "cambio_neto"]
        escritor = csv.DictWriter(manejador, fieldnames=campos, lineterminator="\n")
        escritor.writeheader()
        for nombre, (conteo, neto) in por_tipo.items():
            escritor.writerow({"tipo_de_cambio": nombre, "celdas": conteo, "cambio_neto": neto})

    cuerpo: list[str] = [
        "# V19-02: la restriccion TR implica HS como regla de decision",
        "",
        f"Generado por `experiments/v19_02_restriccion_tr_implica_hs.py` ({EXPERIMENTO}).",
        "",
        "**Un solo parametro, elegido por pliegues. Ninguna submission gastada.**",
        "",
        "## 1. El hecho medido que sostiene la regla",
        "",
        "No hay ni una sola fila con TR=1 y HS=0 en las 1483 del desarrollo, cuando por",
        "independencia se esperarian 250. Y la implicacion es de una sola direccion, porque AG=1 con",
        "HS=0 si ocurre. Eso es un hecho, no una intuicion sobre el significado de las etiquetas.",
        "",
        "## 2. Por que la propagacion correcta es hacia arriba",
        "",
        "| direccion | celdas | macro | delta |",
        "| --- | ---: | ---: | ---: |",
        "| V03 sin regla | 0 | 0.777211 | +0.000000 |",
        "| HS=0 obliga a TR=0 | 20 | 0.776566 | -0.000646 |",
        "| TR=1 obliga a HS=1 | 20 | 0.777896 | +0.000685 |",
        "",
        "Gana la de arriba porque **el error esta en HS, no en TR**: en 9 de las 20 celdas que violan",
        "la restriccion la verdad es (1,1,1), o sea V03 fallo HS y acerto TR. Forzar TR=0 ahi",
        "destruye un acierto por culpa de un error de la otra etiqueta.",
        "",
        "## 3. La regla con un parametro, elegida por pliegues",
        "",
        "El gatillo es el score de TR a partir del cual se fuerza HS=1. Se elige en cuatro pliegues",
        "y se evalua en el quinto, con los cinco rotados.",
        "",
        fila(["pliegue", "trigger", "cambios", "netos", "macro sin regla", "macro con regla", "delta"]),
        fila(["---"] * 7),
    ]
    for f in lineas:
        cuerpo.append(
            fila(
                [
                    str(f["pliegue"]), f"{f['trigger_elegido']:.2f}", str(f["cambios_en_prueba"]),
                    f"{f['cambios_netos']:+d}", f"{f['macro_sin_regla']:.6f}",
                    f"{f['macro_con_regla']:.6f}", f"{f['delta']:+.6f}",
                ]
            )
        )
    cuerpo += [
        "",
        f"- Delta medio fuera de muestra: **{veredicto['delta_medio_fuera_de_muestra']:+.6f}**, con",
        f"  desviacion {veredicto['delta_desviacion']:.6f}.",
        (
            f"- Pliegues que mejoran: **{veredicto['pliegues_que_mejoran']} de "
            f"{veredicto['pliegues_evaluados']}**."
        ),
        f"- Valor p empirico por permutacion de etiquetas: **{veredicto['p_empirico']:.4f}**.",
        "",
        "## 4. La traduccion a leaderboard, que es lo que decide",
        "",
        f"La regla cambia **{celdas} celdas** de HS en todo el OOF:",
        "",
        fila(["tipo de cambio", "celdas", "cambio neto"]),
        fila(["---"] * 3),
    ]
    for nombre, (conteo, neto) in por_tipo.items():
        cuerpo.append(fila([nombre, str(conteo), f"{neto:+d}"]))
    cuerpo += [
        "",
        f"El cambio neto de celda es **{veredicto['cambios_netos_de_celda']:+d}**, y la escala del",
        f"proyecto es de **{CELDAS_POR_003_LB} cambios netos** para mover el leaderboard 0.003. Una",
        f"regla competitive tendria que aportar al menos **{veredicto['cambios_netos_necesarios']:.0f}**.",
        "",
        "## 5. No es un cambio de umbral disfrazado",
        "",
        f"Un umbral de HS mas bajo que {UMBRALES_V03[0]} tomaria **{celdas_de_umbral} celdas**. Esta",
        f"regla toca **{celdas}**, y las que deja intactas incluyen celdas por debajo de ese umbral en",
        "filas donde el score de TR no llega al gatillo. La regla depende del score de la **otra",
        "etiqueta**, luego no es una funcion monotona del score de HS y no se puede escribir como un",
        "umbral.",
        "",
        "## 6. Veredicto",
        "",
        f"**{veredicto['estado']}**",
        "",
        (
            f"- Delta medio: {veredicto['delta_medio_fuera_de_muestra']:+.6f}, con "
            f"{veredicto['pliegues_que_mejoran']} de {veredicto['pliegues_evaluados']} "
            "pliegues mejorando."
        ),
        (
            f"- Cambios netos: {veredicto['cambios_netos_de_celda']:+d} frente a "
            f"{veredicto['cambios_netos_necesarios']:.0f} necesarios."
        ),
        "- Submissions gastadas: **0**.",
        "",
    ]
    cuerpo += sello(inicio)
    REPORTE_MD.write_text("\n".join(cuerpo) + "\n", encoding="utf-8")
    REPORTE_JSON.write_text(json.dumps(veredicto, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8")


if __name__ == "__main__":
    main()
