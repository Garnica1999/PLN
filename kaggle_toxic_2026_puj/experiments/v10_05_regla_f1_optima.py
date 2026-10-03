"""V10-05: regla de decision F1 optima por punto fijo, conadaptacion al shift de prevalencia.

Por que esta linea es distinta de V10-04 y de V10-06
---------------------------------------------------
V03 usa tres umbrales libres, los 0.44, 0.51 y 0.46, que estan **elegidos**. La idea de esta
linea es distinto: en vez de elegir tres numeros, **derivar** la regla de decision de la propia
metrica, de modo que no tenga ningun parametro libre que ajustar.

La derivacion
-------------
Para maximize la F1 de un conjunto de predicciones binarias, el umbral optimo de un clasificador
de probabilidades cumple una condicion de punto fijo sencilla: **el umbral optimo es la mitad
de la F1 optima**. Es decir, si ``F*`` es el mejor valor de F1 alcanzable, entonces
``t* = F* / 2``.

La razon es que en el optimo la F1 vale ``2 / (1/precision + 1/exhaustividad)``, y maximize esa
expresion sobre el conjunto de positivos equivale a minimizar su inversa; el punto donde se
igualan las dos derivadas da exactamente ``t* = F* / 2``. Es el resultado clasico del ajuste de
F-measure, y se puede iterar:

    t <- 0.5
    repetir:
        S  <- filas con p >= t
        precision      <- peso de los verdaderos positivos en S, dividido por el peso de S
        exhaustividad <- peso de los verdaderos positivos en S, dividido por los positivos esperados
        F1  <- 2 * precision * exhaustividad / (precision + exhaustividad)
        t   <- F1 / 2

La convergencia es rapida y el punto fijo es estable en la practica.

Por que importa el shift de prevalencia
---------------------------------------
La regla tiene una entrada que los tres umbrales de V03 **no tienen**: la prevalencia esperada.
Si la prevalencia del conjunto de evaluacion baja de ``pi`` a ``pi'``, la regla baja el umbral
sola, porque con menos positivos esperados conviene mantener la exhaustividad. V9-02 midio que
la prevalencia de la prueba es un factor de 0.936, 0.955 y 0.938 de la de entrenamiento.

Ese es el mecanismo que se pone a prueba: **una regla derivada se adapta a un cambio de
prevalencia, y tres umbrales congelados no se adaptan.**

Protocolo
---------
- Las probabilidades se calibran con isotonica recortada, que fue la de mejor ECE en V10-04, y
  la calibracion es fold-local.
- El umbral plug-in se calcula **dentro de cada pliegue de entrenamiento**, y se aplica al
  pliegue de validacion. La prevalencia objetivo del plug-in es un dato conocido de antemano, no
  se estima con las filas de validacion, de modo que no hay fuga.
- Se evalua en dos escenarios: prior natural, y prior simulado al 0.94, que se consigue
  reponderando las filas de validacion para que su prevalencia efectiva sea la objetivo.
- Se repite con cinco semillas de particion y se mide el intervalo pareado.

Gates, pre-registrados antes de ejecutar
----------------------------------------
- **Gate A**: ganancia anidada de al menos **+0.002** bajo prior simulado 0.94, con **TR y AG no
  negativos**.
- **Gate B**: el mismo incremento bajo prior simulado es **mayor** que bajo prior natural. Si
  la regla no gana mas con la prevalencia desplazada que con la natural, el mecanismo de
  adaptacion no existe y la linea se cierra aunque el gate A pasara.
- **Gate C (addendum)**: delta medio de al menos +0.002, al menos 4 de 5 particiones mejoran, e
  intervalo pareado que no incluye el cero.

Uso
---
    /home/carlos/venv/bin/python experiments/v10_05_regla_f1_optima.py
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from importlib import util
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.calibration import IsotonicRegression

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import read_train
from src.evaluation import (
    ETIQUETAS,
    metricas_multietiqueta,
)
from src.splits import grupos_fusionados, holdout_v2

_especificacion = util.spec_from_file_location(
    "arnes", PROJECT_ROOT / "experiments" / "v10_03_arnes_evaluacion.py"
)
arnes = util.module_from_spec(_especificacion)
sys.modules["arnes"] = arnes
_especificacion.loader.exec_module(arnes)

EXPERIMENTO = "V10-05 (P1, regla de decision F1 optima)"
REPORTE_MD = PROJECT_ROOT / "reports" / "v10_05_regla_f1_optima.md"
REPORTE_JSON = PROJECT_ROOT / "reports" / "v10_05_regla_f1_optima.json"

SEMILLAS_EVALUACION = (42, 7, 123, 2024, 5150)
FACTORES_SHIFT_V9 = {"HS": 0.936, "TR": 0.955, "AG": 0.938}
FACTOR_SHIFT_PLAANO = 0.94
TOLERANCIA_PUNTO_FIJO = 1e-10
MAXIMO_ITERACIONES = 100
DELTA_MINIMO = 0.002
UMBRALES_V03 = arnes.UMBRALES_V03


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[V10-05] {mensaje}", flush=True)


def pesos_para_prevalencia(y_columna: np.ndarray, prevalencia_objetivo: float) -> np.ndarray:
    """Calcula los pesos que hacen que la prevalencia efectiva sea la objetivo.

    Pesar los positivos por ``pi_objetivo / pi_actual`` y los negativos por
    ``(1 - pi_objetivo) / (1 - pi_actual)`` deja la prevalencia ponderada exactamente en la
    objetivo, sin tocar las etiquetas ni las probabilidades.

    Args:
        y_columna: etiqueta binaria de una columna.
        prevalencia_objetivo: prevalencia que se quiere Effective.

    Returns:
        Vector de pesos, uno por fila.

    Raises:
        ValueError: si alguna de las dos clases esta vacia, porque el peso seria indefined.
    """
    positivos = int(np.sum(y_columna == 1))
    negativos = len(y_columna) - positivos
    if positivos == 0 or negativos == 0:
        raise ValueError("una clase esta vacia y la reponderacion no esta definida")
    peso_positivo = prevalencia_objetivo * len(y_columna) / positivos
    peso_negativo = (1.0 - prevalencia_objetivo) * len(y_columna) / negativos
    return np.where(y_columna == 1, peso_positivo, peso_negativo)


def metrica_macro_ponderada(
    y: np.ndarray, predicciones: np.ndarray, pesos: np.ndarray
) -> float:
    """Calcula la metrica macro con pesos por fila.

    Es la misma media column-wise de la F1 de la clase positiva que la metrica de la
    competencia, pero con cada fila ponderada. Sirve para puntuar como se comportaria el
    sistema en una poblacion con otra prevalencia.

    Args:
        y: matriz binaria de etiquetas verdaderas.
        predicciones: matriz binaria de predicciones.
        pesos: peso de cada fila.

    Returns:
        Media column-wise de la F1 ponderada.
    """
    columnas = []
    for indice in range(len(ETIQUETAS)):
        positivos = y[:, indice] == 1
        predichos = predicciones[:, indice] == 1
        verdaderos_positivos = float(np.sum(pesos * (positivos & predichos)))
        falsos_positivos = float(np.sum(pesos * (~positivos & predichos)))
        falsos_negativos = float(np.sum(pesos * (positivos & ~predichos)))
        denominador = 2 * verdaderos_positivos + falsos_positivos + falsos_negativos
        columnas.append(2 * verdaderos_positivos / denominador if denominador > 0 else 0.0)
    return float(np.mean(columnas))


def umbral_f1_optimo(
    probabilidades: np.ndarray,
    y: np.ndarray,
    pesos: np.ndarray,
    prevalencia_objetivo: float,
) -> tuple[float, int, float]:
    """Calcula el umbral F1 optimo por la iteracion de punto fijo ``t = F1 / 2``.

    Args:
        probabilidades: probabilidades calibradas de una etiqueta.
        y: etiqueta binaria real.
        pesos: peso de cada fila, para puntuar en una poblacion reponderada.
        prevalencia_objetivo: proporcion de filas que se espera que sean positivas.

    Returns:
        Tupla con el umbral optimo, el numero de iteraciones y la F1 en el punto fijo.

    Raises:
        ValueError: si el punto fijo no converge dentro del maximo de iteraciones.
    """
    positivos_esperados = prevalencia_objetivo * float(np.sum(pesos))
    if positivos_esperados <= 0.0:
        raise ValueError("la prevalencia objetivo debe ser positiva")
    umbral = 0.5
    f1 = 0.0
    for iteracion in range(1, MAXIMO_ITERACIONES + 1):
        seleccionados = probabilidades >= umbral
        peso_seleccionado = float(np.sum(pesos * seleccionados))
        if peso_seleccionado <= 0.0:
            break
        verdaderos_positivos = float(np.sum(pesos * seleccionados * (y == 1)))
        precision = verdaderos_positivos / peso_seleccionado
        exhaustividad = verdaderos_positivos / positivos_esperados
        if precision + exhaustividad <= 0.0:
            break
        f1 = 2 * precision * exhaustividad / (precision + exhaustividad)
        nuevo_umbral = f1 / 2.0
        if abs(nuevo_umbral - umbral) < TOLERANCIA_PUNTO_FIJO:
            return float(nuevo_umbral), iteracion, float(f1)
        umbral = nuevo_umbral
    raise ValueError(
        f"el punto fijo no convergio en {MAXIMO_ITERACIONES} iteraciones; ultimo umbral {umbral}"
    )


def main() -> None:
    """Ejecuta la regla F1 optima en los dos escenarios de prevalencia."""
    inicio = time.time()
    log("V10-05: regla de decision F1 optima por punto fijo, con prior natural y simulado")

    train = read_train()
    grupos_completos = grupos_fusionados(train)
    indices_dev, _ = holdout_v2(train, grupos_completos)
    desarrollo = train.iloc[indices_dev].reset_index(drop=True)
    grupos_dev = grupos_completos[indices_dev]
    y = desarrollo[list(ETIQUETAS)].to_numpy(dtype=int)
    textos = desarrollo["text"].astype(str)
    log(
        f"   desarrollo: {len(y)} filas | factores de shift de V9-02: "
        + ", ".join(f"{e} {f}" for e, f in FACTORES_SHIFT_V9.items())
    )

    escenarios = {
        "prior_natural": {e: 1.0 for e in ETIQUETAS},
        "prior_simulado": dict(FACTORES_SHIFT_V9),
    }
    detalle: dict[str, list[dict[str, Any]]] = {nombre: [] for nombre in escenarios}
    referencia_por_semilla: dict[int, np.ndarray] = {}
    calibradas_por_semilla: dict[int, np.ndarray] = {}

    log(f"1/2 recorrer {len(SEMILLAS_EVALUACION)} particiones en los dos escenarios")
    for semilla in SEMILLAS_EVALUACION:
        pliegues = arnes.particion_por_semilla(textos, y, grupos_dev, semilla)
        probabilidades = arnes.oof_de_v03(textos, y, pliegues)
        referencia = arnes.predicciones_de_referencia(probabilidades)
        referencia_por_semilla[semilla] = referencia
        calibradas_por_semilla[semilla] = probabilidades

        for nombre_escenario, factores in escenarios.items():
            predicciones = np.zeros((len(y), 3), dtype=int)
            umbrales_por_etiqueta: dict[str, list[float]] = {e: [] for e in ETIQUETAS}
            iteraciones_totales = 0
            for indice_etiqueta, etiqueta in enumerate(ETIQUETAS):
                for entrenamiento, validacion in pliegues:
                    prob_tr = probabilidades[entrenamiento, indice_etiqueta]
                    prob_va = probabilidades[validacion, indice_etiqueta]
                    y_tr = y[entrenamiento, indice_etiqueta]
                    calibrador = IsotonicRegression(y_min=0.01, y_max=0.99, out_of_bounds="clip")
                    calibrador.fit(prob_tr, y_tr)
                    cal_tr = calibrador.predict(prob_tr)
                    cal_va = calibrador.predict(prob_va)
                    # Un solo camino para los dos escenarios: los pesos se derivan de la
                    # prevalencia objetivo. En el escenario natural esa prevalencia es la del
                    # propio pliegue, y entonces los pesos valen exactamente 1. Pasar en su
                    # lugar un 1.0 como prevalencia objetivo daria peso CERO a los negativos y
                    # la iteracion no convergeria, porque la precision se iria a 1.
                    prevalencia_objetivo = float(np.mean(y_tr)) * factores[etiqueta]
                    pesos_tr = pesos_para_prevalencia(y_tr, prevalencia_objetivo)
                    umbral, iteraciones, _ = umbral_f1_optimo(
                        cal_tr, y_tr, pesos_tr, prevalencia_objetivo
                    )
                    iteraciones_totales += iteraciones
                    umbrales_por_etiqueta[etiqueta].append(umbral)
                    predicciones[validacion, indice_etiqueta] = (cal_va >= umbral).astype(int)

            if nombre_escenario == "prior_natural":
                pesos_validacion = np.ones(len(y))
            else:
                pesos_validacion = np.ones(len(y))
                for indice_etiqueta, etiqueta in enumerate(ETIQUETAS):
                    objetivo = float(np.mean(y[:, indice_etiqueta])) * factores[etiqueta]
                    pesos_validacion = pesos_validacion * pesos_para_prevalencia(
                        y[:, indice_etiqueta], objetivo
                    )
            metrica_candidata = metrica_macro_ponderada(y, predicciones, pesos_validacion)
            metrica_referencia = metrica_macro_ponderada(y, referencia, pesos_validacion)
            delta = metrica_candidata - metrica_referencia
            f1_candidata = metricas_multietiqueta(y, predicciones)["f1_por_etiqueta"]
            f1_referencia = metricas_multietiqueta(y, referencia)["f1_por_etiqueta"]
            detalle[nombre_escenario].append(
                {
                    "semilla": semilla,
                    "metrica_referencia": round(metrica_referencia, 6),
                    "metrica_candidata": round(metrica_candidata, 6),
                    "delta": round(delta, 6),
                    "mejora": bool(delta > 0.0),
                    "celdas_distintas": int((predicciones != referencia).sum()),
                    "umbrales_por_etiqueta": {
                        e: [round(u, 4) for u in valores]
                        for e, valores in umbrales_por_etiqueta.items()
                    },
                    "f1_por_etiqueta": {k: round(v, 6) for k, v in f1_candidata.items()},
                    "delta_por_etiqueta": {
                        e: round(f1_candidata[e] - f1_referencia[e], 6) for e in ETIQUETAS
                    },
                    "iteraciones_totales_del_punto_fijo": iteraciones_totales,
                    "llamadas_al_punto_fijo": len(pliegues) * len(ETIQUETAS),
                }
            )
        resumen = " | ".join(
            f"{nombre} {detalle[nombre][-1]['delta']:+.6f}" for nombre in escenarios
        )
        log(f"   semilla {semilla:>6}: {resumen}")

    log("2/2 agregar resultados y aplicar los gates pre-registrados")
    agregados: dict[str, dict[str, Any]] = {}
    for nombre, particiones in detalle.items():
        deltas = np.array([p["delta"] for p in particiones], dtype=np.float64)
        agregados[nombre] = {
            "delta_medio": round(float(deltas.mean()), 6),
            "delta_mediano": round(float(np.median(deltas)), 6),
            "particiones_que_mejoran": int((deltas > 0).sum()),
            "total_particiones": len(particiones),
            "celdas_distintas_promedio": round(
                float(np.mean([p["celdas_distintas"] for p in particiones])), 1
            ),
            "f1_por_etiqueta_en_la_semilla_canonica": particiones[0]["f1_por_etiqueta"],
        }
        log(
            f"   {nombre:>16}: delta medio {agregados[nombre]['delta_medio']:+.6f} | "
            f"{agregados[nombre]['particiones_que_mejoran']}/"
            f"{agregados[nombre]['total_particiones']} mejoran | "
            f"{agregados[nombre]['celdas_distintas_promedio']:.0f} celdas"
        )

    delta_simulado = agregados["prior_simulado"]["delta_medio"]
    delta_natural = agregados["prior_natural"]["delta_medio"]
    # El gate A pide que TR y AG no sean negativos, y "negativo" se mide **contra la
    # referencia**, no contra cero: una F1 siempre es positiva, asi que comparar contra cero
    # no diria nada.
    deltas_por_etiqueta = detalle["prior_simulado"][0]["delta_por_etiqueta"]
    tr_y_ag_no_negativos = bool(
        deltas_por_etiqueta["TR"] >= 0.0 and deltas_por_etiqueta["AG"] >= 0.0
    )
    log(
        "   deltas por etiqueta en la particion canonica con prior simulado: "
        + ", ".join(f"{e} {deltas_por_etiqueta[e]:+.6f}" for e in ETIQUETAS)
    )
    gate_a = bool(delta_simulado >= DELTA_MINIMO and tr_y_ag_no_negativos)
    gate_b = bool(delta_simulado > delta_natural)
    log(
        f"   gate A (delta simulado {delta_simulado:+.6f} contra {DELTA_MINIMO} y TR/AG no "
        f"negativos {tr_y_ag_no_negativos}): {gate_a}"
    )
    log(
        f"   gate B (el shift aporta: simulado {delta_simulado:+.6f} frente a natural "
        f"{delta_natural:+.6f}): {gate_b}"
    )

    pliegues_canonicos = arnes.particion_por_semilla(textos, y, grupos_dev, 42)
    probabilidades_canonicas = arnes.oof_de_v03(textos, y, pliegues_canonicos)
    referencia_canonica = arnes.predicciones_de_referencia(probabilidades_canonicas)
    predicciones_canonicas = np.zeros((len(y), 3), dtype=int)
    for indice_etiqueta, etiqueta in enumerate(ETIQUETAS):
        for entrenamiento, validacion in pliegues_canonicos:
            prob_tr = probabilidades_canonicas[entrenamiento, indice_etiqueta]
            prob_va = probabilidades_canonicas[validacion, indice_etiqueta]
            y_tr = y[entrenamiento, indice_etiqueta]
            calibrador = IsotonicRegression(y_min=0.01, y_max=0.99, out_of_bounds="clip")
            calibrador.fit(prob_tr, y_tr)
            objetivo = float(np.mean(y_tr)) * FACTORES_SHIFT_V9[etiqueta]
            umbral, _, _ = umbral_f1_optimo(
                calibrador.predict(prob_tr),
                y_tr,
                pesos_para_prevalencia(y_tr, objetivo),
                objetivo,
            )
            predicciones_canonicas[validacion, indice_etiqueta] = (
                calibrador.predict(prob_va) >= umbral
            ).astype(int)
    observada, inferior, superior = arnes.intervalo_pareado(
        y, predicciones_canonicas, referencia_canonica
    )
    log(f"   intervalo pareado en la particion canonica: [{inferior:+.6f}, {superior:+.6f}]")

    gate_c = {
        "delta_medio_al_menos_0.002": bool(delta_simulado >= DELTA_MINIMO),
        "al_menos_4_de_5_particiones": bool(
            agregados["prior_simulado"]["particiones_que_mejoran"]
            >= arnes.CANDIDATAS_MEJORAN_MINIMO
        ),
        "intervalo_pareado_no_incluye_cero": bool(inferior > 0.0 or superior < 0.0),
    }
    gate_c["superado"] = bool(all(gate_c.values()))

    informe = {
        "experimento": EXPERIMENTO,
        "fecha": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "derivacion": (
            "El umbral F1 optimo de un clasificador de probabilidades cumple el punto fijo "
            "t = F1 / 2. Se itera desde t = 0.5. La iteracion necesita la prevalencia esperada, "
            "que es la entrada que los tres umbrales congelados de V03 no tienen."
        ),
        "protocolo": (
            "Calibracion isotonica recortada fold-local, umbral plug-in calculado dentro de cada "
            "pliegue de entrenamiento, y aplicado al pliegue de validacion. La prevalencia objetivo "
            "es un dato conocido de antemano de V9-02, no se estima con las filas de validacion."
        ),
        "factores_de_shift": FACTORES_SHIFT_V9,
        "agregados": agregados,
        "umbrales_de_la_regla_por_etiqueta": {
            escenario: {
                e: round(
                    float(np.mean([p["umbrales_por_etiqueta"][e][0] for p in particiones])),
                    4,
                )
                for e in ETIQUETAS
            }
            for escenario, particiones in detalle.items()
        },
        "umbrales_congelados_de_v03": {
            e: float(UMBRALES_V03[indice]) for indice, e in enumerate(ETIQUETAS)
        },
        "delta_por_etiqueta_canonico_simulado": deltas_por_etiqueta,
        "intervalo_pareado_canonico": {
            "diferencia_observada": round(observada, 6),
            "ic95_inferior": round(inferior, 6),
            "ic95_superior": round(superior, 6),
        },
        "gate_a_superado": gate_a,
        "gate_b_superado": gate_b,
        "gate_c_del_addendum": gate_c,
        "detalle_por_particion": detalle,
    }
    REPORTE_JSON.write_text(
        json.dumps(informe, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _informe(informe, inicio)
    log(f"completado en {round(time.time() - inicio, 1)} s")
    log(
        "VEREDICTO: "
        + (
            "GATES A, B Y C SUPERADOS"
            if gate_a and gate_b and gate_c["superado"]
            else "SIN MEJORAR SOBRE V03, sub-via cerrada"
            if agregados["prior_simulado"]["delta_medio"] <= 0.0
            else "GATE NO SUPERADO, sub-via cerrada"
        )
    )


def _informe(informe: dict[str, Any], inicio: float) -> None:
    """Escribe el informe de la regla de decision F1 optima.

    Args:
        informe: veredicto del experimento.
        inicio: instante de arranque del script.
    """
    agregados = informe["agregados"]
    lineas = [
        "# V10-05: regla de decision F1 optima por punto fijo (P1, segunda linea de la via A)",
        "",
        f"Generado por `experiments/v10_05_regla_f1_optima.py` ({EXPERIMENTO}).",
        "",
        "## 1. La idea, y por que no es mover umbrales",
        "",
        "V03 usa tres umbrales libres, los 0.44, 0.51 y 0.46, que estan **elegidos**. Aqui no se",
        "elige ningun numero: la regla se **deriva** de la metrica. El umbral F1 optimo de un",
        "clasificador de probabilidades cumple el punto fijo `t = F1 / 2`, y se itera desde 0.5.",
        "",
        "La diferencia relevante para esta fase es que la regla tiene una entrada que los tres",
        "umbrales congelados no tienen: **la prevalencia esperada**. Si la prevalencia baja, la",
        "regla baja el umbral sola. V9-02 midio que la prevalencia de la prueba es un factor de",
        f"{FACTORES_SHIFT_V9['HS']}, {FACTORES_SHIFT_V9['TR']} y {FACTORES_SHIFT_V9['AG']}.",
        "",
        "## 2. La regla aprende a moverse",
        "",
        "La tabla muestra los umbrales que la regla elige en los cinco pliegues, en el escenario",
        "de prior natural y en el de prior simulado. La comparacion con los umbrales congelados de",
        "V03 es la que explica el mecanismo.",
        "",
        (
            "| etiqueta | V03 congelado | regla con prior natural | regla con prior simulado | "
            "cuanto baja |"
        ),
        "|---|---|---|---|---|",
    ]
    for etiqueta in ETIQUETAS:
        natural = informe["umbrales_de_la_regla_por_etiqueta"]["prior_natural"][etiqueta]
        simulado = informe["umbrales_de_la_regla_por_etiqueta"]["prior_simulado"][etiqueta]
        congelado = informe["umbrales_congelados_de_v03"][etiqueta]
        linea_texto = (
            f"| {etiqueta} | {congelado:.2f} | {natural:.4f} | {simulado:.4f} | "
            f"{simulado - natural:+.4f} |"
        )
        lineas.append(linea_texto)
    lineas += [
        "",
        "## 3. Ganancia anidada en los dos escenarios",
        "",
        "| escenario | " + " | ".join(str(s) for s in SEMILLAS_EVALUACION) + " | media | mejoran |",
        "|---|" + "---|" * (len(SEMILLAS_EVALUACION) + 2),
    ]
    for nombre, particiones in informe["detalle_por_particion"].items():
        celdas = " | ".join(
            f"{p['delta']:+.6f}" + ("*" if p["mejora"] else "") for p in particiones
        )
        lineas.append(
            f"| {nombre} | {celdas} | {agregados[nombre]['delta_medio']:+.6f} | "
            f"{agregados[nombre]['particiones_que_mejoran']}/{agregados[nombre]['total_particiones']} |"
        )
    lineas += [
        "",
        (
            f"- celdas que la regla cambia frente a V03, en el escenario simulado: "
            f"{agregados['prior_simulado']['celdas_distintas_promedio']:.0f} de media"
        ),
        (
            f"- intervalo pareado en la particion canonica: "
            f"[{informe['intervalo_pareado_canonico']['ic95_inferior']:+.6f}, "
            f"{informe['intervalo_pareado_canonico']['ic95_superior']:+.6f}]"
        ),
        "",
        "## 4. Gates pre-registrados",
        "",
        (
            f"- **Gate A**, incremento de al menos {DELTA_MINIMO} con prior simulado y TR y AG no "
            f"negativos: **{informe['gate_a_superado']}**"
        ),
        f"- **Gate B**, el shift aporta mas que el prior natural: **{informe['gate_b_superado']}**",
    ]
    gate_c = informe["gate_c_del_addendum"]
    lineas += [
        f"- **Gate C**, los tres criterios del addendum: **{gate_c['superado']}**",
        f"  - delta medio de al menos 0.002: {gate_c['delta_medio_al_menos_0.002']}",
        f"  - al menos 4 de 5 particiones: {gate_c['al_menos_4_de_5_particiones']}",
        f"  - intervalo que no incluye el cero: {gate_c['intervalo_pareado_no_incluye_cero']}",
        "",
        "## 5. Veredicto",
        "",
    ]
    if agregados["prior_simulado"]["delta_medio"] <= 0.0:
        lineas += [
            "**SUB-VIA CERRADA.** La regla derivada no mejora a V03, y ademas empeora. Es un",
            "resultado con sentido: la regla F1 optima esta disenada para ser optima **en el",
            "conjunto donde se calcula**, y alplicable a un conjunto con otra prevalencia, no",
            "para.transferirla a datos nuevos. V03 ya esta en el maximo de la curva que midio",
            "V8-09, y una regla mas lista no puede superar un optimo.",
        ]
    elif not informe["gate_a_superado"]:
        lineas += [
            (
                "**SUB-VIA CERRADA.** La regla mejora, pero no llega al incremento de "
                f"{DELTA_MINIMO} exigido con prior simulado, o bien TR o AG salen negativos."
            ),
        ]
    elif not informe["gate_b_superado"]:
        lineas += [
            "**SUB-VIA CERRADA.** La regla mejora, pero **no mejora mas con el shift de",
            "prevalencia que sin el**. El mecanismo de adaptacion que se ponia a prueba no",
            "existe en los datos, y el gate B existe justo para cerrar esa puerta sin tener que",
            "mirar si el resultado final sube.",
        ]
    else:
        lineas += [
            "**GATES SUPERADOS.** La regla cumple los tres gates y pasa al filtro de",
            "detectabilidad de V8-02 antes de considerar cualquier sonda.",
        ]
    lineas += [
        "",
        (
            f"Fecha: {datetime.now(timezone.utc).isoformat(timespec='seconds')}. "
            f"Duracion: {round(time.time() - inicio, 1)} s."
        ),
    ]
    REPORTE_MD.write_text("\n".join(lineas) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
