"""V11-06: validacion metodologica de los estimadores de prevalencia bajo shift SIMULADO.

La pregunta que decide el resto de la fase
-------------------------------------------
La auditoria de legalidad, V11-02, cerro el uso de la prueba para generar un submission, de modo
que la v11 solo puede estudiar la **metodologia**: si el mecanismo recupera un desplazamiento de
prevalencia que se conoce de antemano.

Antes de medir si la correccion de prior o la calibracion ayudan bajo shift, hay que responder
una pregunta mas basica y mas dura: **los estimadores de prevalencia funcionan?** Es decir, si se
implanta artificialmente un desplazamiento conocido en el OOF, lo recupera el estimador?

Si la respuesta es no, todo lo que se construya encima estara construyendose sobre arena, y es
mucho mas barato saberlo ahora que al final.

Los dos estimadores, que son conceptualmente distintos
-----------------------------------------------------
- **EM de Saerens**, con iteracion de punto fijo. Es el que uso V9-02. Necesita las
  probabilidades del modelo y un prior de partida, que se toma del entrenamiento.
- **Clasificar-y-contar ajustado**, de Forman y Crawford. No usa probabilidades sino **conteos**:
  si se predicen ``m`` positivos en un conjunto de ``n`` filas, y se conocen la sensibilidad y la
  especificidad en la distribucion de origen, la prevalencia destino se despeja de

      m/n = Se * pi_destino + (1 - Sp) * (1 - pi_destino)
      pi_destino = (m/n - (1 - Sp)) / (Se - (1 - Sp))

  Es una idea completamente distinta: no ajusta una likelihood, despeja una identidad
  algebraica. Por eso conviene tener los dos: si los dos fallan, el problema no es el estimador
  sino la idea; si solo falla uno, es el estimador.

Como se simula el desplazamiento
--------------------------------
Un cambio de prevalencia es un cambio de ``P(y)`` manteniendo ``P(x|y)``. Eso significa que las
poblaciones de origen y destino tienen **las mismas probabilidades por fila** y distinto peso por
clase. Reponderar los positivos y los negativos por factores que dejan la prevalencia ponderada
en el objetivo reproduce exactamente esa situacion, y es lo que se hace aqui.

La simulacion es ponderada y no por submuestreo, porque el submuestreo anade introduce ruido de
muestreo en la estimacion y aqui lo que se quiere medir es el sesgo del estimador, no ese ruido.

Protocolo
---------
Todo el OOF del desarrollo, con las probabilidades **calibradas de forma anidada**, porque un
estimador de prevalencia sobre probabilidades mal calibradas esta midiendo dos errores a la vez.
La calibracion se ajusta dentro de cada pliegue de entrenamiento y se aplica al de validacion.

Cada estimador se evalua sobre una rejilla de prevalencias objetivo, y se reportan el error de
recuperacion **absoluto** y el **relativo**, que es el que importa: un factor de 0,94 recuperado
como 0,90 es un error del 4 por ciento relativo, que es justo la magnitud del efecto que se
quiere medir.

Criterio, pre-registrado antes de ejecutar
------------------------------------------
- **Gate A, recuperacion**: el error relativo medio de cada estimador, sobre la rejilla, es
  menor que **0,05**, o sea la quinta parte del propio efecto que se quiere medir. Un estimador
  que falla mas que eso no sirve para nada downstream.
- **Gate B, incorrupcion**: el estimador **no mejora cuando el shift es nulo**. Si declarando que
  no hay desplazamiento el estimador inventa una correccion, es peor que no hacer nada, porque
  genera confianza falsa.
- **Cierre**: si algun estimador no pasa, se documenta y se elige el que pase, o se cierra la via
  de estimacion y solo se trabaja con la prevalencia conocida por construccion, que es la
  simulacion misma.

Uso
---
    /home/carlos/venv/bin/python experiments/v11_06_validacion_estimadores.py
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
sys.path.insert(0, str(Path(__file__).resolve().parent))

from v11_ayudas import fila, sello

from src import data as data_module
from src.evaluation import ETIQUETAS
from src.splits import RANDOM_STATE, grupos_fusionados, holdout_v2

_especificacion_arnes = util.spec_from_file_location(
    "arnes", PROJECT_ROOT / "experiments" / "v10_03_arnes_evaluacion.py"
)
arnes = util.module_from_spec(_especificacion_arnes)
sys.modules["arnes"] = arnes
_especificacion_arnes.loader.exec_module(arnes)

EXPERIMENTO = "V11-06 (P2.3, validacion de estimadores bajo shift simulado)"
REPORTE_MD = PROJECT_ROOT / "reports" / "v11_06_validacion_estimadores.md"
REPORTE_JSON = PROJECT_ROOT / "reports" / "v11_06_validacion_estimadores.json"
CSV_ANALISIS = PROJECT_ROOT / "reports" / "v11_shift_analysis.csv"

ETIQUETAS_LISTA = list(ETIQUETAS)
REFERENCIA_V03 = 0.7772114525165721
REJILLA_FACTORES = (0.70, 0.80, 0.90, 1.00, 1.10)
TOLERANCIA_CONVERGENCIA_EM = 1e-8
MAXIMO_ITERACIONES_EM = 500
ERROR_RELATIVO_MAXIMO = 0.05
CAMPOS_CSV = [
    "Label",
    "Train_Prevalence",
    "Estimated_Test_Prevalence",
    "Shift_Factor",
    "Estimator",
    "Uncertainty",
    "Simulation_Error",
    "Decision_Impact",
]


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[V11-06] {mensaje}", flush=True)


def em_ponderado(
    probabilidades: np.ndarray, pesos: np.ndarray, prior_inicial: float
) -> tuple[float, int]:
    """Ejecuta el EM de Saerens sobre probabilidades con pesos por fila.

    El EM de Saerens actualiza el prior iterativamente. Ponderar por fila es lo que permite
    evaluar el estimador sobre una poblacion cuya composicion se ha simulado, sin alterar las
    probabilidades, que es justamente lo que hace un cambio de prevalencia real.

    Args:
        probabilidades: probabilidades calibradas de una etiqueta.
        pesos: peso de cada fila en la poblacion destino.
        prior_inicial: prior de partida, que se toma de la distribucion de origen.

    Returns:
        Tupla con el prior estimado y el numero de iteraciones.

    Raises:
        ValueError: si el prior de partida no esta en el intervalo unitario.
    """
    if not 0.0 < prior_inicial < 1.0:
        raise ValueError(f"prior inicial fuera de intervalo: {prior_inicial}")
    prior = prior_inicial
    total_peso = float(np.sum(pesos))
    iteraciones = 0
    for _ in range(MAXIMO_ITERACIONES_EM):
        iteraciones += 1
        peso_positivo = prior / prior_inicial
        peso_negativo = (1.0 - prior) / (1.0 - prior_inicial)
        numerador = peso_positivo * probabilidades
        denominador = numerador + peso_negativo * (1.0 - probabilidades)
        posteriores = numerador / np.maximum(denominador, 1e-12)
        nuevo = float(np.sum(pesos * posteriores) / total_peso)
        if abs(nuevo - prior) < TOLERANCIA_CONVERGENCIA_EM:
            return nuevo, iteraciones
        prior = nuevo
    return prior, iteraciones


def sensibilidad_y_especificidad(
    probabilidades: np.ndarray, y: np.ndarray, umbral: float
) -> tuple[float, float]:
    """Calcula sensibilidad y especificidad de un clasificador en la distribucion de origen.

    Son los dos numeros que necesita el estimador de clasificar-y-contar, y se calculan sobre el
    OOF, que es la unica fuente permitted a para estadisticas supervisadas.

    Args:
        probabilidades: probabilidades calibradas de una etiqueta.
        y: etiqueta binaria real.
        umbral: punto de corte.

    Returns:
        Tupla con la sensibilidad y la especificidad.
    """
    predichas = probabilidades >= umbral
    positivos = y == 1
    negativos = ~positivos
    sensibilidad = float(np.sum(predichas & positivos) / max(1, np.sum(positivos)))
    especificidad = float(np.sum(~predichas & negativos) / max(1, np.sum(negativos)))
    return sensibilidad, especificidad


def clasificar_y_contar_ajustado(
    probabilidades: np.ndarray,
    pesos: np.ndarray,
    sensibilidad: float,
    especificidad: float,
    umbral: float,
) -> float:
    """Estima la prevalencia destino con el metodo de clasificar-y-contar ajustado.

        m/n = Se * pi_destino + (1 - Sp) * (1 - pi_destino)

    de donde se despeja la prevalencia. El estimador es puramente algebraico y no usa las
    probabilidades mas alla del recuento de positivos predichos.

    Args:
        probabilidades: probabilidades calibradas de una etiqueta.
        pesos: peso de cada fila en la poblacion destino.
        sensibilidad: sensibilidad medida en la distribucion de origen.
        especificidad: especificidad medida en la distribucion de origen.
        umbral: punto de corte del clasificador.

    Returns:
        Prevalencia destino estimada.

    Raises:
        ValueError: si el clasificador no separa las clases en la distribucion de origen, porque
            entonces el denominador se anula y la estimate no esta definida.
    """
    total_peso = float(np.sum(pesos))
    positivos_predichos = float(np.sum(pesos * (probabilidades >= umbral)))
    fraccion = positivos_predichos / total_peso
    denominador = sensibilidad - (1.0 - especificidad)
    if abs(denominador) < 1e-6:
        raise ValueError("el clasificador no separa las clases: estimate indefinida")
    estimate = (fraccion - (1.0 - especificidad)) / denominador
    return float(np.clip(estimate, 0.0, 1.0))


def pesos_para_prevalencia(
    y_columna: np.ndarray, prevalencia_objetivo: float
) -> np.ndarray:
    """Calcula los pesos que llevan la prevalencia ponderada a la objetivo.

    Pesar los positivos por ``pi_objetivo / pi_actual`` y los negativos por
    ``(1 - pi_objetivo) / (1 - pi_actual)`` deja la prevalencia ponderada exactamente en la
    objetivo, sin tocar una sola probabilidad.

    Args:
        y_columna: etiqueta binaria de una columna.
        prevalencia_objetivo: prevalencia que se quiere Effective.

    Returns:
        Vector de pesos, uno por fila.

    Raises:
        ValueError: si alguna de las dos clases esta vacia, porque el peso no estaria definido.
    """
    positivos = int(np.sum(y_columna == 1))
    negativos = len(y_columna) - positivos
    if positivos == 0 or negativos == 0:
        raise ValueError("una clase esta vacia y la reponderacion no esta definida")
    peso_positivo = prevalencia_objetivo * len(y_columna) / positivos
    peso_negativo = (1.0 - prevalencia_objetivo) * len(y_columna) / negativos
    return np.where(y_columna == 1, peso_positivo, peso_negativo)


def main() -> None:
    """Ejecuta la validacion de los dos estimadores bajo shift simulado."""
    inicio = time.time()
    log("V11-06: validacion de estimadores de prevalencia bajo shift simulado")

    train = data_module.read_train()
    grupos_completos = grupos_fusionados(train)
    indices_dev, _ = holdout_v2(train, grupos_completos)
    desarrollo = train.iloc[indices_dev].reset_index(drop=True)
    grupos_dev = grupos_completos[indices_dev]
    y = desarrollo[ETIQUETAS_LISTA].to_numpy(dtype=int)
    textos = desarrollo["text"].astype(str)
    pliegues = arnes.particion_por_semilla(textos, y, grupos_dev, RANDOM_STATE)
    log(f"   desarrollo: {len(y)} filas | pliegues: {len(pliegues)}")

    log("1/4 probabilidades calibradas de forma anidada, sobre el OOF canonico")
    oof = np.load(PROJECT_ROOT / "features" / "oof_v03_final.npz", allow_pickle=True)
    probabilidades_crudas = oof["proba"]
    calibradas = np.zeros_like(probabilidades_crudas)
    for indice_etiqueta in range(3):
        for entrenamiento, validacion in pliegues:
            modelo = IsotonicRegression(y_min=0.01, y_max=0.99, out_of_bounds="clip")
            modelo.fit(
                probabilidades_crudas[entrenamiento, indice_etiqueta],
                y[entrenamiento, indice_etiqueta],
            )
            calibradas[validacion, indice_etiqueta] = modelo.predict(
                probabilidades_crudas[validacion, indice_etiqueta]
            )
    log("   calibracion isotonica anidada aplicada a los cinco pliegues")

    log("2/4 sensibilidad y especificidad de la distribucion de origen, sobre el OOF")
    # El punto de corte se elige con la regla de decision congelada de V03, porque es el punto de
    # partida de la fase y no se va a optimizar nada aqui.
    operacion = {"HS": 0.44, "TR": 0.51, "AG": 0.46}
    metricas_origen: dict[str, dict[str, float]] = {}
    for indice_etiqueta, etiqueta in enumerate(ETIQUETAS_LISTA):
        sensibilidad, especificidad = sensibilidad_y_especificidad(
            calibradas[:, indice_etiqueta], y[:, indice_etiqueta], operacion[etiqueta]
        )
        metricas_origen[etiqueta] = {
            "sensibilidad": round(sensibilidad, 6),
            "especificidad": round(especificidad, 6),
            "prevalencia_origen": round(float(np.mean(y[:, indice_etiqueta])), 6),
        }
        log(
            f"   {etiqueta}: sensibilidad {sensibilidad:.4f} | especificidad "
            f"{especificidad:.4f} | prevalencia origen "
            f"{metricas_origen[etiqueta]['prevalencia_origen']:.4f}"
        )

    log(f"3/4 estimadores sobre {len(REJILLA_FACTORES)} prevalencias objetivo simuladas")
    filas_csv: list[dict[str, Any]] = []
    detalle: list[dict[str, Any]] = []
    for indice_etiqueta, etiqueta in enumerate(ETIQUETAS_LISTA):
        origen = metricas_origen[etiqueta]
        for factor in REJILLA_FACTORES:
            objetivo = float(np.clip(origen["prevalencia_origen"] * factor, 0.01, 0.99))
            pesos = pesos_para_prevalencia(y[:, indice_etiqueta], objetivo)
            prevalencia_efectiva = float(
                np.sum(pesos * y[:, indice_etiqueta]) / np.sum(pesos)
            )
            estimacion_em, iteraciones = em_ponderado(
                calibradas[:, indice_etiqueta], pesos, origen["prevalencia_origen"]
            )
            estimacion_cac = clasificar_y_contar_ajustado(
                calibradas[:, indice_etiqueta],
                pesos,
                origen["sensibilidad"],
                origen["especificidad"],
                operacion[etiqueta],
            )
            error_em = estimacion_em - prevalencia_efectiva
            error_cac = estimacion_cac - prevalencia_efectiva
            rel_em = error_em / prevalencia_efectiva
            rel_cac = error_cac / prevalencia_efectiva
            detalle.append(
                {
                    "etiqueta": etiqueta,
                    "factor_simulado": factor,
                    "prevalencia_objetivo": round(objetivo, 6),
                    "prevalencia_efectiva_ponderada": round(prevalencia_efectiva, 6),
                    "factor_efectivo": round(prevalencia_efectiva / origen["prevalencia_origen"], 6),
                    "em_estimada": round(estimacion_em, 6),
                    "em_error_absoluto": round(error_em, 6),
                    "em_error_relativo": round(rel_em, 6),
                    "em_iteraciones": iteraciones,
                    "cac_estimada": round(estimacion_cac, 6),
                    "cac_error_absoluto": round(error_cac, 6),
                    "cac_error_relativo": round(rel_cac, 6),
                }
            )
            filas_csv.append(
                {
                    "Label": etiqueta,
                    "Train_Prevalence": round(origen["prevalencia_origen"], 6),
                    "Estimated_Test_Prevalence": round(estimacion_em, 6),
                    "Shift_Factor": round(estimacion_em / origen["prevalencia_origen"], 6),
                    "Estimator": "EM de Saerens ponderado, sobre isotonica anidada",
                    "Uncertainty": "sin cuantificar en este experimento",
                    "Simulation_Error": round(rel_em, 6),
                    "Decision_Impact": "PENDIENTE_V11_07",
                }
            )
            log(
                f"   {etiqueta} factor {factor:.2f}: efectiva "
                f"{prevalencia_efectiva:.4f} | EM {estimacion_em:.4f} "
                f"(rel {rel_em:+.4f}) | clasificar-y-contar {estimacion_cac:.4f} "
                f"(rel {rel_cac:+.4f})"
            )

    log("4/4 gates pre-registrados")
    errores_em = [abs(d["em_error_relativo"]) for d in detalle]
    errores_cac = [abs(d["cac_error_relativo"]) for d in detalle]
    sin_desplazamiento = [d for d in detalle if d["factor_simulado"] == 1.00]
    em_en_cero = float(np.mean([abs(d["em_error_relativo"]) for d in sin_desplazamiento]))
    cac_en_cero = float(np.mean([abs(d["cac_error_relativo"]) for d in sin_desplazamiento]))

    gate_a = {
        "error_relativo_medio_em": round(float(np.mean(errores_em)), 6),
        "error_relativo_medio_cac": round(float(np.mean(errores_cac)), 6),
        "tolerancia": ERROR_RELATIVO_MAXIMO,
        "em_supera": bool(float(np.mean(errores_em)) < ERROR_RELATIVO_MAXIMO),
        "cac_supera": bool(float(np.mean(errores_cac)) < ERROR_RELATIVO_MAXIMO),
    }
    gate_a["superado"] = bool(gate_a["em_supera"] or gate_a["cac_supera"])
    gate_b = {
        "error_relativo_em_con_shift_nulo": round(em_en_cero, 6),
        "error_relativo_cac_con_shift_nulo": round(cac_en_cero, 6),
        "em_no_incorrompe": bool(em_en_cero < ERROR_RELATIVO_MAXIMO),
        "cac_no_incorrompe": bool(cac_en_cero < ERROR_RELATIVO_MAXIMO),
    }
    gate_b["superado"] = bool(gate_b["em_no_incorrompe"] or gate_b["cac_no_incorrompe"])

    with CSV_ANALISIS.open("w", encoding="utf-8", newline="") as manejador:
        escritor = csv.DictWriter(manejador, fieldnames=CAMPOS_CSV, lineterminator="\n")
        escritor.writeheader()
        escritor.writerows(filas_csv)
    log(f"   {len(filas_csv)} filas escritas en {CSV_ANALISIS.relative_to(PROJECT_ROOT)}")

    informe = {
        "experimento": EXPERIMENTO,
        "fecha": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "referencia_v03": REFERENCIA_V03,
        "pregunta": (
            "Si se implanta artificialmente un desplazamiento de prevalencia conocido en el OOF, "
            "los estimadores lo recuperan? Es la condicion necesaria para que todo lo demas de la "
            "via de shift tenga sentido."
        ),
        "protocolo": (
            "Simulacion ponderada y no por submuestreo, porque lo que se mide es el sesgo del "
            "estimador y no el ruido de muestreo. Probabilidades calibradas con isotonica de forma "
            "anidada, porque estimar la prevalencia sobre probabilidades mal calibradas mide dos "
            "errores a la vez. Sensibilidad y especificidad medidas sobre el OOF, que es la unica "
            "fuente permitida para estadisticas supervisadas, y con el umbral congelado de V03."
        ),
        "rejilla_de_factores": list(REJILLA_FACTORES),
        "metricas_de_la_distribucion_de_origen": metricas_origen,
        "detalle": detalle,
        "gate_a_recuperacion": gate_a,
        "gate_b_incorrupcion": gate_b,
        "estimador_recomendado": (
            "EM de Saerens"
            if gate_a["em_supera"] and (not gate_a["cac_supera"] or gate_b["em_no_incorrompe"])
            else "clasificar-y-contar"
            if gate_a["cac_supera"]
            else "ninguno"
        ),
        "puede_continuar_la_fase": bool(gate_a["superado"] and gate_b["superado"]),
    }
    REPORTE_JSON.write_text(
        json.dumps(informe, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _informe(informe, inicio)
    log(f"completado en {round(time.time() - inicio, 1)} s")
    log(
        f"VEREDICTO: gate A {gate_a['superado']} (EM {gate_a['error_relativo_medio_em']:.4f}, "
        f"contar {gate_a['error_relativo_medio_cac']:.4f}) | "
        f"gate B {gate_b['superado']} | estimador: {informe['estimador_recomendado']}"
    )


import csv


def _informe(informe: dict[str, Any], inicio: float) -> None:
    """Escribe el informe de la validacion de estimadores.

    Args:
        informe: veredicto del experimento.
        inicio: instante de arranque del script.
    """
    gate_a = informe["gate_a_recuperacion"]
    gate_b = informe["gate_b_incorrupcion"]
    lineas: list[str] = [
        "# V11-06: validacion de los estimadores de prevalencia bajo shift simulado (P2.3)",
        "",
        f"Generado por `experiments/v11_06_validacion_estimadores.py` ({EXPERIMENTO}).",
        "",
        "## 1. La pregunta que decide el resto de la fase",
        "",
        "La auditoria de legalidad cerro el uso de la prueba para generar un submission, de modo",
        "que la v11 solo puede estudiar la metodologia: si el mecanismo recupera un desplazamiento",
        "de prevalencia que se conoce de antemano.",
        "",
        "Antes de medir si la correccion de prior o la calibracion ayudan bajo shift, hay que",
        "responder una pregunta mas basica y mas dura: **los estimadores funcionan?** Si se",
        "implanta artificialmente un desplazamiento conocido y el estimador no lo recupera, todo",
        "lo que se construya encima estara construido sobre arena, y es mucho mas barato saberlo",
        "ahora que al final.",
        "",
        "## 2. Los dos estimadores, y por que conviene tenerlos",
        "",
        "- **EM de Saerens**, iteracion de punto fijo. Es el que uso V9-02. Necesita las",
        "  probabilidades y un prior de partida.",
        "- **Clasificar-y-contar ajustado**, de Forman y Crawford. No usa probabilidades sino",
        "  **conteos**: si se predicen `m` positivos en `n` filas y se conocen la sensibilidad y la",
        "  especificidad en origen, la prevalencia se despeja de una identidad algebraica.",
        "",
        "Son ideas completamente distintas: uno ajusta una verosimilitud, el otro despeja una",
        "ecuacion. Si los dos fallan, el problema no es el estimador sino la idea; si solo falla",
        "uno, es el estimador. Por eso se miden los dos.",
        "",
        "## 3. Como se simula el desplazamiento",
        "",
        "Un cambio de prevalencia cambia `P(y)` manteniendo `P(x|y)`: las mismas probabilidades",
        "por fila y distinto peso por clase. Reponderar positivos y negativos para que la",
        "prevalencia ponderada sea la objetivo reproduce exactamente eso.",
        "",
        "La simulacion es ponderada y **no** por submuestreo, porque lo que se mide es el sesgo",
        "del estimador y no el ruido de muestreo que el submuestreo introduciria.",
        "",
        "## 4. Metricas de la distribucion de origen, medidas sobre el OOF",
        "",
        fila(["etiqueta", "prevalencia origen", "sensibilidad", "especificidad"]),
        fila(["---", "---", "---", "---"]),
    ]
    for etiqueta, valores in informe["metricas_de_la_distribucion_de_origen"].items():
        lineas.append(
            fila(
                [
                    etiqueta,
                    f"{valores['prevalencia_origen']:.6f}",
                    f"{valores['sensibilidad']:.4f}",
                    f"{valores['especificidad']:.4f}",
                ]
            )
        )
    lineas += [
        "",
        "## 5. Recuperacion del desplazamiento, etiqueta por etiqueta",
        "",
        fila(
            [
                "etiqueta",
                "factor pedido",
                "factor efectivo",
                "EM estimada",
                "error rel. EM",
                "contar ajustada",
                "error rel. contar",
            ]
        ),
        fila(["---"] * 7),
    ]
    for detalle in informe["detalle"]:
        linea_texto = fila(
            [
                detalle["etiqueta"],
                f"{detalle['factor_simulado']:.2f}",
                f"{detalle['factor_efectivo']:.4f}",
                f"{detalle['em_estimada']:.4f}",
                f"{detalle['em_error_relativo']:+.4f}",
                f"{detalle['cac_estimada']:.4f}",
                f"{detalle['cac_error_relativo']:+.4f}",
            ]
        )
        lineas.append(linea_texto)
    lineas += [
        "",
        "## 6. Gates pre-registrados",
        "",
        (
            f"- **Gate A, recuperacion**: error relativo medio menor que "
            f"{gate_a['tolerancia']}."
        ),
        f"  - EM: **{gate_a['error_relativo_medio_em']:.6f}**, supera: {gate_a['em_supera']}",
        (
            f"  - clasificar y contar: **{gate_a['error_relativo_medio_cac']:.6f}**, supera: "
            f"{gate_a['cac_supera']}"
        ),
        "- **Gate B, no incorromper**: con shift nulo, el error debe seguir siendo pequeno. Un",
        "  estimador que sin desplazamiento inventa una correccion es peor que no hacer nada,",
        "  porque genera confianza falsa.",
        (
            f"  - EM: **{gate_b['error_relativo_em_con_shift_nulo']:.6f}**, no corrompe: "
            f"{gate_b['em_no_incorrompe']}"
        ),
        (
            f"  - clasificar y contar: **{gate_b['error_relativo_cac_con_shift_nulo']:.6f}**, "
            f"no corrompe: {gate_b['cac_no_incorrompe']}"
        ),
        "",
        f"**La fase puede continuar: {informe['puede_continuar_la_fase']}.**",
        f"**Estimador recomendado: {informe['estimador_recomendado']}.**",
        "",
        "## 7. Veredicto",
        "",
    ]
    if not informe["puede_continuar_la_fase"]:
        lineas += [
            "**NINGUN ESTIMADOR PASA.** La via de estimacion de prevalencia se cierra, y con ella",
            "la correccion de prior, que no tendria sobre que aplicarse. La unica via que queda",
            "es trabajar con la prevalencia conocida por construccion, que es la simulacion misma,",
            "y medir ahi si la correccion mejora la metrica.",
        ]
    elif informe["estimador_recomendado"] == "EM de Saerens":
        lineas += [
            "**LOS ESTIMADORES FUNCIONAN.** El error relativo medio esta por debajo de la",
            "tolerancia y el estimador no incorrumpe con shift nulo, de modo que la metodologia de",
            "correccion de prevalencia tiene sobre que trabajar.",
            "",
            "Conviene decir lo que esto **no** demuestra: que el estimador recupera un",
            "desplazamiento que uno mismo ha implantado, sobre probabilidades que uno mismo ha",
            "calibrado. No demuestra nada sobre la prevalencia real de la prueba, que es",
            "precisamente lo que la auditoria de legalidad prohibits usar. Lo que demuestra es",
            "que el mecanismo es correcto **en el caso en el que se puede comprobar**, que es lo",
            "maximo que se puede pedir sin usar la prueba.",
        ]
    else:
        lineas += [
            "**LOS ESTIMADORES FUNCIONAN, y el que mejor recupera es clasificar-y-contar.** El EM",
            "de Saerens queda por detras en error relativo, lo que es informativo porque son",
            "mecanismos muy distintos: uno ajusta una verosimilitud y el otro despeja una",
            "identidad algebraica. Cuando el mas simple gana, suele ser porque el mas complejo",
            "estabiese ajustando algo que no hace falta ajustar.",
        ]
    lineas += sello(inicio)
    REPORTE_MD.write_text("\n".join(lineas) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
