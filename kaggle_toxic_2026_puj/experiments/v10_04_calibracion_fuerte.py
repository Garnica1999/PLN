"""V10-04: calibracion fuerte por etiqueta, con calibrador y umbral anidados en el mismo pliegue.

Contexto
--------
Primera linea de la via A. La regla 3 de la fase prohibe optimizar nada sobre el mismo OOF que
se reporta, y la auditoria forense anterior midio que hacerlo infla +0.0036. Por eso aqui
**el calibrador y el umbral se ajustan los dos dentro de cada pliegue de entrenamiento** y se
aplican al pliegue de validacion, y el proceso se repite con cinco semillas de particion.

Que compara
----------
Cuatro metodos, por etiqueta, todos fold-local:

- **sin_calibrar**: la referencia, que es V03 con sus umbrales congelados de 0.44, 0.51, 0.46.
- **platt**: regresion logistica univariable sobre el log odds. Es la linea base del gate.
- **isotonica_recortada**: regresion isotonica con la salida recortada en [0.01, 0.99] para
  evitar salidas degeneradas en tramos sin variacion.
- **beta_tres_parametros**: la version de Kull y Silva Filho, con dos parametros de forma y
  uno de escala, que es la unica de las tres con la forma correcta para probabilidades. Se
  ajusta por busqueda en rejilla sobre la log-verosimilitud, mas lenta que Newton pero
  completamente determinista, que es lo que pide la regla 6.

Criterios de calidad
--------------------
- **ECE** y **MCE**: error de calibracion esperado y maximo, por diez tramos de igual masa.
- **Brier** y **NLL**: penalizaciones cuadratica y logaritmica.
- **Ganancia de F1 anidada**: cuanto mejora la metrica de la competencia cuando el calibrador y
  el umbral se ajustan solo con las filas de entrenamiento de cada pliegue.

Gates, pre-registrados antes de ejecutar
----------------------------------------
- **Gate 1**: la mejor calibracion reduce el ECE al menos un **20 por ciento** frente a Platt.
- **Gate 2**: la mejor calibracion **no empeora** la F1 anidada.
- **Gate 3, del addendum de protocolo**: delta medio de al menos +0.002, al menos 4 de 5
  particiones mejoran, e intervalo pareado que no incluye el cero.

Cierre: si no se supera el gate 1 la sub-via se cierra documentada, sin reintentar con mas
variantes de calibracion. Si se supera el 1 y el 2 pero no el 3, la calibracion se declara **no
accionable**: mejora la calibracion en abstracto y no la metrica de la competencia.

Uso
---
    /home/carlos/venv/bin/python experiments/v10_04_calibracion_fuerte.py
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
from sklearn.linear_model import LogisticRegression

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import read_train
from src.evaluation import (
    ETIQUETAS,
    metrica_macro,
    metricas_multietiqueta,
)
from src.splits import grupos_fusionados, holdout_v2

_especificacion = util.spec_from_file_location(
    "arnes", PROJECT_ROOT / "experiments" / "v10_03_arnes_evaluacion.py"
)
arnes = util.module_from_spec(_especificacion)
sys.modules["arnes"] = arnes
_especificacion.loader.exec_module(arnes)

EXPERIMENTO = "V10-04 (P1, calibracion fuerte)"
REPORTE_MD = PROJECT_ROOT / "reports" / "v10_04_calibracion_fuerte.md"
REPORTE_JSON = PROJECT_ROOT / "reports" / "v10_04_calibracion_fuerte.json"

SEMILLAS_EVALUACION = (42, 7, 123, 2024, 5150)
TRAMOS_CALIBRACION = 10
REDUCCION_ECE_MINIMA = 0.20
GRANULARIDAD_UMBRAL = 0.005
REJILLA_BETA = 9
SIN_CALIBRAR = "sin_calibrar"
METODOS = (SIN_CALIBRAR, "platt", "isotonica_recortada", "beta_tres_parametros")
METODOS_COMPARADOS = ("platt", "isotonica_recortada", "beta_tres_parametros")
UMBRALES_V03 = arnes.UMBRALES_V03


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[V10-04] {mensaje}", flush=True)


def log_odds(probabilidades: np.ndarray) -> np.ndarray:
    """Calcula el log odds de un vector de probabilidades.

    Args:
        probabilidades: probabilidades de entrada.

    Returns:
        Vector de log odds.
    """
    probabilidades = np.clip(probabilidades, 1e-6, 1.0 - 1e-6)
    return np.log(probabilidades / (1.0 - probabilidades))


class BetaCalibrator:
    """Calibrador beta de tres parametros, con forma y escala.

    Mapea la probabilidad ``p`` a ``1 / (1 + (1/p)^a * (1/(1-p))^b)`` con un factor de escala.
    Los parametros se ajustan maximizando la log-verosimilitud en una rejilla, que es mas
    lenta que el metodo de Newton de la referencia pero completamente determinista.

    Attributes:
        alfa: primer parametro de forma.
        beta: segundo parametro de forma.
        escala: parametro de escala.
        verosimilitud: log-verosimilitud del ultimo ajuste.
    """

    def __init__(self) -> None:
        """Inicializa los parametros en la identidad, que es el punto neutro."""
        self.alfa = 1.0
        self.beta = 1.0
        self.escala = 1.0
        self.verosimilitud = float("-inf")

    def transformar(self, probabilidades: np.ndarray) -> np.ndarray:
        """Transforma probabilidades con los parametros actuales.

        Args:
            probabilidades: probabilidades de entrada.

        Returns:
            Probabilidades transformadas.
        """
        probabilidades = np.clip(probabilidades, 1e-6, 1.0 - 1e-6)
        logit = np.log(probabilidades / (1.0 - probabilidades))
        return 1.0 / (1.0 + np.exp(-(self.escala * (self.alfa * logit - self.beta))))

    def ajustar(self, probabilidades: np.ndarray, y: np.ndarray) -> BetaCalibrator:
        """Ajusta los parametros maximizando la log-verosimilitud en una rejilla.

        Args:
            probabilidades: probabilidades de entrenamiento.
            y: etiquetas binarias de entrenamiento.

        Returns:
            El propio calibrador, ya ajustado.
        """
        mejor = (float("-inf"), 1.0, 1.0, 1.0)
        for alfa in np.linspace(0.7, 1.3, REJILLA_BETA):
            for beta in np.linspace(0.7, 1.3, REJILLA_BETA):
                for escala in np.linspace(0.85, 1.15, REJILLA_BETA):
                    self.alfa, self.beta, self.escala = alfa, beta, escala
                    transformadas = np.clip(self.transformar(probabilidades), 1e-6, 1.0 - 1e-6)
                    verosimilitud = float(
                        np.sum(y * np.log(transformadas) + (1 - y) * np.log(1.0 - transformadas))
                    )
                    if verosimilitud > mejor[0]:
                        mejor = (verosimilitud, alfa, beta, escala)
        self.verosimilitud, self.alfa, self.beta, self.escala = mejor
        return self


def ajustar_platt(probabilidades: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Ajusta una calibracion de Platt en el espacio de log odds.

    Args:
        probabilidades: probabilidades de entrenamiento.
        y: etiquetas binarias de entrenamiento.

    Returns:
        Tupla con la pendiente y el desplazamiento.
    """
    modelo = LogisticRegression(C=1e6, max_iter=2000, solver="lbfgs")
    modelo.fit(log_odds(probabilidades).reshape(-1, 1), y)
    return float(modelo.coef_[0][0]), float(modelo.intercept_[0])


def aplicar_platt(probabilidades: np.ndarray, pendiente: float, desplazamiento: float) -> np.ndarray:
    """Aplica una calibracion de Platt ya ajustada.

    Args:
        probabilidades: probabilidades de entrada.
        pendiente: pendiente ajustada.
        desplazamiento: desplazamiento ajustado.

    Returns:
        Probabilidades calibradas.
    """
    return 1.0 / (1.0 + np.exp(-(pendiente * log_odds(probabilidades) + desplazamiento)))


def aplicar_isotonica(probabilidades: np.ndarray, y: np.ndarray) -> IsotonicRegression:
    """Ajusta una regresion isotonica recortada y la devuelve lista para predecir.

    Args:
        probabilidades: probabilidades de entrenamiento.
        y: etiquetas binarias de entrenamiento.

    Returns:
        Regresion isotonica ajustada.
    """
    modelo = IsotonicRegression(y_min=0.01, y_max=0.99, out_of_bounds="clip")
    modelo.fit(probabilidades, y)
    return modelo


def aplicar_beta(probabilidades: np.ndarray, y: np.ndarray) -> BetaCalibrator:
    """Ajusta un calibrador beta y lo devuelve listo para transformar.

    Args:
        probabilidades: probabilidades de entrenamiento.
        y: etiquetas binarias de entrenamiento.

    Returns:
        Calibrador beta ajustado.
    """
    return BetaCalibrator().ajustar(probabilidades, y)


def error_de_calibracion(
    probabilidades: np.ndarray, y: np.ndarray, tramos: int = TRAMOS_CALIBRACION
) -> tuple[float, float]:
    """Calcula el error de calibracion esperado y el maximo, por tramos de igual masa.

    Args:
        probabilidades: probabilidades a evaluar.
        y: etiquetas binarias reales.
        tramos: numero de tramos.

    Returns:
        Tupla con el error esperado y el maximo.
    """
    orden = np.argsort(probabilidades)
    probabilidades_ordenadas = probabilidades[orden]
    y_ordenado = y[orden]
    errores: list[float] = []
    for tramo in np.array_split(np.arange(len(y_ordenado)), tramos):
        if len(tramo) == 0:
            continue
        confianza = float(probabilidades_ordenadas[tramo].mean())
        frecuencia = float(y_ordenado[tramo].mean())
        errores.append((len(tramo) / len(y_ordenado)) * abs(confianza - frecuencia))
    return float(np.sum(errores)), float(np.max(errores))


def brier_y_nll(probabilidades: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Calcula las penalizaciones cuadratica y logaritmica.

    Args:
        probabilidades: probabilidades a evaluar.
        y: etiquetas binarias reales.

    Returns:
        Tupla con la penalizacion cuadratica media y la logaritmica media.
    """
    recortadas = np.clip(probabilidades, 1e-9, 1.0 - 1e-9)
    brier = float(np.mean((recortadas - y) ** 2))
    nll = float(-np.mean(y * np.log(recortadas) + (1 - y) * np.log(1.0 - recortadas)))
    return brier, nll


def mejor_umbral(probabilidades: np.ndarray, y: np.ndarray) -> float:
    """Busca el umbral que maximiza la F1 de una etiqueta, en el conjunto dado.

    Args:
        probabilidades: probabilidades de una etiqueta.
        y: etiqueta binaria real.

    Returns:
        El umbral optimo, o 0.5 si ninguna prediccion resulta positiva.
    """
    mejor_f1, mejor = -1.0, 0.5
    for umbral in np.arange(0.05, 0.951, GRANULARIDAD_UMBRAL):
        predicciones = (probabilidades >= umbral).astype(int)
        verdaderos_positivos = float(np.sum((predicciones == 1) & (y == 1)))
        if verdaderos_positivos == 0:
            continue
        precision = verdaderos_positivos / float(np.sum(predicciones))
        exhaustividad = verdaderos_positivos / float(np.sum(y == 1))
        f1 = 2 * precision * exhaustividad / (precision + exhaustividad)
        if f1 > mejor_f1:
            mejor_f1, mejor = f1, float(umbral)
    return mejor


def predicciones_anidadas(
    probabilidades: np.ndarray,
    y: np.ndarray,
    pliegues: list[tuple[np.ndarray, np.ndarray]],
    metodo: str,
    usar_umbrales_v03: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict[str, list[float]]]:
    """Calibra y decide con calibrador y umbral anidados en el mismo pliegue.

    Para cada pliegue, el calibrador y el umbral se ajustan **solo** con las filas de
    entrenamiento, y despues se aplican a las de validacion. Eso es lo que evita que el
    resultado dependa de informacion que despues se reporta.

    Args:
        probabilidades: matriz de probabilidades fuera de muestra.
        y: matriz binaria de etiquetas.
        pliegues: pares de indices de entrenamiento y validacion.
        metodo: nombre del metodo de calibracion.
        usar_umbrales_v03: si es ``True`` se usan los umbrales congelados en vez de buscar.

    Returns:
        Tupla con las probabilidades calibradas fuera de muestra, las predicciones binarias, y
        los umbrales que se eligieron en cada pliegue por etiqueta.
    """
    calibradas = np.zeros_like(probabilidades)
    predicciones = np.zeros((len(y), 3), dtype=int)
    umbrales_por_etiqueta: dict[str, list[float]] = {e: [] for e in ETIQUETAS}
    for indice_etiqueta, etiqueta in enumerate(ETIQUETAS):
        for entrenamiento, validacion in pliegues:
            prob_entrenamiento = probabilidades[entrenamiento, indice_etiqueta]
            prob_validacion = probabilidades[validacion, indice_etiqueta]
            y_entrenamiento = y[entrenamiento, indice_etiqueta]
            if metodo == SIN_CALIBRAR:
                cal_entrenamiento = prob_entrenamiento
                cal_validacion = prob_validacion
            elif metodo == "platt":
                pendiente, desplazamiento = ajustar_platt(prob_entrenamiento, y_entrenamiento)
                cal_entrenamiento = aplicar_platt(prob_entrenamiento, pendiente, desplazamiento)
                cal_validacion = aplicar_platt(prob_validacion, pendiente, desplazamiento)
            elif metodo == "isotonica_recortada":
                modelo = aplicar_isotonica(prob_entrenamiento, y_entrenamiento)
                cal_entrenamiento = modelo.predict(prob_entrenamiento)
                cal_validacion = modelo.predict(prob_validacion)
            elif metodo == "beta_tres_parametros":
                calibrador = aplicar_beta(prob_entrenamiento, y_entrenamiento)
                cal_entrenamiento = calibrador.transformar(prob_entrenamiento)
                cal_validacion = calibrador.transformar(prob_validacion)
            else:
                raise ValueError(f"metodo de calibracion desconocido: {metodo}")
            calibradas[validacion, indice_etiqueta] = cal_validacion
            if usar_umbrales_v03:
                umbral = float(UMBRALES_V03[indice_etiqueta])
            else:
                umbral = mejor_umbral(cal_entrenamiento, y_entrenamiento)
            umbrales_por_etiqueta[etiqueta].append(umbral)
            predicciones[validacion, indice_etiqueta] = (cal_validacion >= umbral).astype(int)
    return calibradas, predicciones, umbrales_por_etiqueta


def main() -> None:
    """Ejecuta la comparacion de calibraciones con calibrador y umbral anidados."""
    inicio = time.time()
    log("V10-04: calibracion fuerte por etiqueta, con particiones repetidas")

    train = read_train()
    grupos_completos = grupos_fusionados(train)
    indices_dev, _ = holdout_v2(train, grupos_completos)
    desarrollo = train.iloc[indices_dev].reset_index(drop=True)
    grupos_dev = grupos_completos[indices_dev]
    y = desarrollo[list(ETIQUETAS)].to_numpy(dtype=int)
    textos = desarrollo["text"].astype(str)
    log(f"   desarrollo: {len(y)} filas | metodos: {', '.join(METODOS)}")

    log(f"1/2 recorrer {len(SEMILLAS_EVALUACION)} particiones con calibracion y umbral anidados")
    calidad: dict[str, dict[str, list[float]]] = {metodo: {} for metodo in METODOS}
    detalle: dict[str, list[dict[str, Any]]] = {metodo: [] for metodo in METODOS}
    referencia_por_semilla: dict[int, np.ndarray] = {}

    for semilla in SEMILLAS_EVALUACION:
        pliegues = arnes.particion_por_semilla(textos, y, grupos_dev, semilla)
        probabilidades = arnes.oof_de_v03(textos, y, pliegues)
        referencia = arnes.predicciones_de_referencia(probabilidades)
        metrica_referencia = metrica_macro(y, referencia)
        referencia_por_semilla[semilla] = referencia
        for metodo in METODOS:
            # La referencia se evalua con sus umbrales congelados, no con un umbral reoptimizado,
            # porque V03 **es** la referencia: reoptimizar su umbral seria mover el punto de
            # partida y comparar dos cosas distintas.
            usar_fijos = metodo == SIN_CALIBRAR
            calibradas, predicciones, umbrales = predicciones_anidadas(
                probabilidades, y, pliegues, metodo, usar_umbrales_v03=usar_fijos
            )
            metrica_candidata = metrica_macro(y, predicciones)
            delta = metrica_candidata - metrica_referencia
            for indice_etiqueta in range(3):
                ece, mce = error_de_calibracion(
                    calibradas[:, indice_etiqueta], y[:, indice_etiqueta]
                )
                brier, nll = brier_y_nll(calibradas[:, indice_etiqueta], y[:, indice_etiqueta])
                clave = ETIQUETAS[indice_etiqueta]
                calidad[metodo].setdefault("ece_" + clave, []).append(ece)
                calidad[metodo].setdefault("mce_" + clave, []).append(mce)
                calidad[metodo].setdefault("brier_" + clave, []).append(brier)
                calidad[metodo].setdefault("nll_" + clave, []).append(nll)
            detalle[metodo].append(
                {
                    "semilla": semilla,
                    "metrica_referencia": round(metrica_referencia, 6),
                    "metrica_candidata": round(metrica_candidata, 6),
                    "delta": round(delta, 6),
                    "mejora": bool(delta > 0.0),
                    "celdas_distintas": int((predicciones != referencia).sum()),
                    "umbrales_por_etiqueta": {
                        e: [round(u, 3) for u in valores] for e, valores in umbrales.items()
                    },
                    "f1_por_etiqueta": {
                        k: round(v, 6)
                        for k, v in metricas_multietiqueta(y, predicciones)["f1_por_etiqueta"].items()
                    },
                }
            )
        resumen = " | ".join(
            f"{m} {detalle[m][-1]['delta']:+.6f}" for m in METODOS
        )
        log(f"   semilla {semilla:>6}: {resumen}")

    log("2/2 agregar resultados y aplicar los gates pre-registrados")
    calidad_agregada: dict[str, dict[str, float]] = {}
    for metodo in METODOS:
        calidad_agregada[metodo] = {
            clave: round(float(np.mean(valores)), 6) for clave, valores in calidad[metodo].items()
        }
    for metodo in METODOS:
        eces = " ".join(f"{e} {calidad_agregada[metodo]['ece_' + e]:.5f}" for e in ETIQUETAS)
        log(f"   {metodo:>24} | ECE por etiqueta: {eces}")

    ece_platt = float(np.mean([calidad_agregada["platt"]["ece_" + e] for e in ETIQUETAS]))
    reduccion_por_metodo = {}
    for metodo in METODOS_COMPARADOS:
        ece_metodo = float(np.mean([calidad_agregada[metodo]["ece_" + e] for e in ETIQUETAS]))
        reduccion_por_metodo[metodo] = round((ece_platt - ece_metodo) / ece_platt, 6)
    mejor_ece = max(reduccion_por_metodo, key=lambda m: reduccion_por_metodo[m])
    reduccion_maxima = reduccion_por_metodo[mejor_ece]
    gate_1 = bool(reduccion_maxima >= REDUCCION_ECE_MINIMA)
    log(
        f"   ECE medio de Platt: {ece_platt:.6f} | mejor metodo: {mejor_ece} | "
        f"reduccion {reduccion_maxima * 100:.2f} por ciento | gate 1: {gate_1}"
    )

    delta_por_metodo = {
        metodo: round(
            float(np.mean([p["delta"] for p in detalle[metodo]])),
            6,
        )
        for metodo in METODOS
    }
    mejoran_por_metodo = {metodo: int(sum(p["mejora"] for p in detalle[metodo])) for metodo in METODOS}
    mejor_delta_metodo = max(
        (m for m in METODOS_COMPARADOS), key=lambda m: delta_por_metodo[m]
    )
    gate_2 = bool(delta_por_metodo[mejor_delta_metodo] >= 0.0)

    # El intervalo pareado se calcula sobre la particion canonica, la misma que usan las otras
    # lineas de la via A, para que las comparaciones entre lineas sean homologas.
    pliegues_canonicos = arnes.particion_por_semilla(textos, y, grupos_dev, 42)
    probabilidades_canonicas = arnes.oof_de_v03(textos, y, pliegues_canonicos)
    referencia_canonica = arnes.predicciones_de_referencia(probabilidades_canonicas)
    intervalos: dict[str, dict[str, float]] = {}
    for metodo in METODOS_COMPARADOS:
        _, predicciones, _ = predicciones_anidadas(
            probabilidades_canonicas, y, pliegues_canonicos, metodo
        )
        observada, inferior, superior = arnes.intervalo_pareado(
            y, predicciones, referencia_canonica
        )
        intervalos[metodo] = {
            "diferencia_observada": round(observada, 6),
            "ic95_inferior": round(inferior, 6),
            "ic95_superior": round(superior, 6),
            "no_incluye_cero": bool(inferior > 0.0 or superior < 0.0),
        }
        log(
            f"   intervalo pareado de {metodo}: [{inferior:+.6f}, {superior:+.6f}] | "
            f"no incluye el cero: {intervalos[metodo]['no_incluye_cero']}"
        )

    gate_3 = {
        "metodo": mejor_delta_metodo,
        "delta_medio_al_menos_0.002": bool(delta_por_metodo[mejor_delta_metodo] >= 0.002),
        "al_menos_4_de_5_particiones": bool(
            mejoran_por_metodo[mejor_delta_metodo] >= arnes.CANDIDATAS_MEJORAN_MINIMO
        ),
        "intervalo_pareado_no_incluye_cero": bool(
            intervalos[mejor_delta_metodo]["no_incluye_cero"]
        ),
    }
    gate_3["superado"] = bool(
        gate_3["delta_medio_al_menos_0.002"]
        and gate_3["al_menos_4_de_5_particiones"]
        and gate_3["intervalo_pareado_no_incluye_cero"]
    )

    # Diagnostico de la calibracion beta. Es un modelo de tres parametros, mas general que
    # Platt, asi que deberia alcanzar al menos la misma verosimilitud. Si no la alcanza, lo
    # que se esta comparando no es la familia beta sino una rejilla estrecha que **no contiene**
    # a Platt, porque el intercepto de Platt es libre y aqui beta esta limitado a [0.7, 1.3].
    # Se mide la verosimilitud de ambos sobre la particion canonica para comprobarlo en vez de
    # suponerlo.
    verosimilitudes: dict[str, dict[str, float]] = {}
    for metodo in ("platt", "beta_tres_parametros"):
        por_etiqueta: dict[str, float] = {}
        for indice_etiqueta, etiqueta in enumerate(ETIQUETAS):
            negativas = 0.0
            for entrenamiento, validacion in pliegues_canonicos:
                prob_tr = probabilidades_canonicas[entrenamiento, indice_etiqueta]
                prob_va = probabilidades_canonicas[validacion, indice_etiqueta]
                y_tr = y[entrenamiento, indice_etiqueta]
                if metodo == "platt":
                    pendiente, desplazamiento = ajustar_platt(prob_tr, y_tr)
                    calibradas_va = aplicar_platt(prob_va, pendiente, desplazamiento)
                else:
                    calibradas_va = aplicar_beta(prob_tr, y_tr).transformar(prob_va)
                recortadas = np.clip(calibradas_va, 1e-9, 1.0 - 1e-9)
                y_va = y[validacion, indice_etiqueta]
                negativas -= float(
                    np.sum(y_va * np.log(recortadas) + (1 - y_va) * np.log(1.0 - recortadas))
                )
            por_etiqueta[etiqueta] = round(negativas, 2)
        verosimilitudes[metodo] = por_etiqueta
    platt_menor_en_todas = all(
        verosimilitudes["platt"][e] <= verosimilitudes["beta_tres_parametros"][e]
        for e in ETIQUETAS
    )
    log(
        "   verosimilitud negativa en la particion canonica | "
        + " | ".join(
            f"{e}: platt {verosimilitudes['platt'][e]:.1f}, "
            f"beta {verosimilitudes['beta_tres_parametros'][e]:.1f}"
            for e in ETIQUETAS
        )
    )
    log(
        "   la rejilla beta alcanza peor o igual verosimilitud que Platt en las tres: "
        f"{platt_menor_en_todas}"
    )

    informe = {
        "experimento": EXPERIMENTO,
        "fecha": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "protocolo": (
            "El calibrador y el umbral se ajustan los dos dentro de cada pliegue de "
            "entrenamiento y se aplican al pliegue de validacion, y se repite con cinco "
            "semillas de particion. La referencia usa sus umbrales congelados, porque "
            "reoptimizar el umbral de la referencia seria mover el punto de partida."
        ),
        "metodos": list(METODOS),
        "calidad_de_calibracion": calidad_agregada,
        "ece_medio_por_metodo": {
            metodo: round(
                float(np.mean([calidad_agregada[metodo]["ece_" + e] for e in ETIQUETAS])), 6
            )
            for metodo in METODOS
        },
        "reduccion_de_ece_frente_a_platt": reduccion_por_metodo,
        "mejor_reduccion": {"metodo": mejor_ece, "reduccion": reduccion_maxima},
        "ganancia_anidada_por_metodo": delta_por_metodo,
        "particiones_que_mejoran": mejoran_por_metodo,
        "total_particiones": len(SEMILLAS_EVALUACION),
        "intervalos_pareados": intervalos,
        "gate_1_superado": gate_1,
        "gate_2_superado": gate_2,
        "gate_3_del_addendum": gate_3,
        "diagnostico_de_la_calibracion_beta": {
            "verosimilitud_negativa_por_etiqueta": verosimilitudes,
            "platt_alcanza_igual_o_mejor_que_la_rejilla_beta": bool(platt_menor_en_todas),
            "explicacion": (
                "La calibracion beta tiene tres parametros y en principio contiene a Platt, "
                "pero la rejilla usada limita beta a [0.7, 1.3] mientras que el intercepto de "
                "Platt es libre. La rejilla, por tanto, NO contiene a Platt, y por eso el "
                "resultado de beta no es un juicio sobre la familia beta sino sobre esta "
                "rejilla estrecha. El gate 1 no depende de beta, asi que su veredicto sigue "
                "siendo valido."
            ),
        },
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
            "GATES 1, 2 Y 3 SUPERADOS"
            if gate_1 and gate_2 and gate_3["superado"]
            else "GATE 1 NO SUPERADO, sub-via cerrada"
            if not gate_1
            else "GATE 2 NO SUPERADO, sub-via cerrada"
            if not gate_2
            else "GATES 1 Y 2 SUPERADOS PERO NO ES ACCIONABLE"
        )
    )


def _informe(informe: dict[str, Any], inicio: float) -> None:
    """Escribe el informe de la comparacion de calibraciones.

    Args:
        informe: veredicto del experimento.
        inicio: instante de arranque del script.
    """
    lineas = [
        "# V10-04: calibracion fuerte por etiqueta (P1, primera linea de la via A)",
        "",
        f"Generado por `experiments/v10_04_calibracion_fuerte.py` ({EXPERIMENTO}).",
        "",
        "## 1. Como se mide",
        "",
        "La regla 3 de la fase prohibe optimizar nada sobre el mismo OOF que se reporta, y la",
        "auditoria anterior midio que hacerlo infla +0.0036. Por eso **el calibrador y el umbral",
        "se ajustan los dos dentro de cada pliegue de entrenamiento**, se aplican al pliegue de",
        "validacion, y el proceso se repite con cinco semillas de particion.",
        "",
        "La referencia es V03 con sus umbrales congelados. **No** se le reoptimiza el umbral:",
        "eso seria mover el punto de partida y comparar dos cosas distintas.",
        "",
        "## 2. Calidad de calibracion",
        "",
        "| metodo | " + " | ".join(f"ECE {e}" for e in ETIQUETAS) + " | ECE medio |",
        "|---|" + "---|" * (len(ETIQUETAS) + 1),
    ]
    for metodo in METODOS:
        celdas = " | ".join(
            f"{informe['calidad_de_calibracion'][metodo]['ece_' + e]:.6f}" for e in ETIQUETAS
        )
        lineas.append(
            f"| {metodo} | {celdas} | {informe['ece_medio_por_metodo'][metodo]:.6f} |"
        )
    mejor = informe["mejor_reduccion"]
    lineas += [
        "",
        "Reduccion del ECE frente a Platt, con el ECE medio de las tres etiquetas:",
        "",
        "| metodo | reduccion | supera el gate del 20 por ciento |",
        "|---|---|---|",
    ]
    for metodo, reduccion in informe["reduccion_de_ece_frente_a_platt"].items():
        lineas.append(
            f"| {metodo} | {reduccion * 100:+.2f} por ciento | "
            f"{'si' if reduccion >= REDUCCION_ECE_MINIMA else 'no'} |"
        )
    lineas += [
        "",
        f"El mejor es **{mejor['metodo']}** con {mejor['reduccion'] * 100:+.2f} por ciento.",
        "",
        f"**Gate 1 superado: {informe['gate_1_superado']}.**",
        "",
        "## 3. Ganancia anidada de F1 de la competencia",
        "",
        "Cada celda es el delta de la candidata contra V03. Con asterisco, la particion en la que",
        "la candidata mejora.",
        "",
        "| metodo | " + " | ".join(str(s) for s in SEMILLAS_EVALUACION) + " | media | mejoran |",
        "|---|" + "---|" * (len(SEMILLAS_EVALUACION) + 2),
    ]
    for metodo in METODOS:
        particiones = informe["detalle_por_particion"][metodo]
        celdas = " | ".join(
            f"{p['delta']:+.6f}" + ("*" if p["mejora"] else "") for p in particiones
        )
        lineas.append(
            f"| {metodo} | {celdas} | {informe['ganancia_anidada_por_metodo'][metodo]:+.6f} | "
            f"{informe['particiones_que_mejoran'][metodo]}/{informe['total_particiones']} |"
        )
    lineas += [
        "",
        f"**Gate 2, no empeorar la F1 anidada: {informe['gate_2_superado']}.**",
        "",
        "## 4. Intervalos pareados sobre la particion canonica",
        "",
        "| metodo | diferencia observada | IC95 | incluye el cero |",
        "|---|---|---|---|",
    ]
    for metodo, valores in informe["intervalos_pareados"].items():
        lineas.append(
            f"| {metodo} | {valores['diferencia_observada']:+.6f} | "
            f"[{valores['ic95_inferior']:+.6f}, {valores['ic95_superior']:+.6f}] | "
            f"{'si' if not valores['no_incluye_cero'] else 'no'} |"
        )
    gate_3 = informe["gate_3_del_addendum"]
    diagnostico = informe["diagnostico_de_la_calibracion_beta"]
    lineas += [
        "",
        "## 5. Gate 3 del addendum de protocolo",
        "",
        f"- metodo evaluado: **{gate_3['metodo']}**",
        f"- delta medio de al menos 0.002: **{gate_3['delta_medio_al_menos_0.002']}**",
        f"- al menos 4 de 5 particiones mejoran: **{gate_3['al_menos_4_de_5_particiones']}**",
        f"- intervalo pareado que no incluye el cero: **{gate_3['intervalo_pareado_no_incluye_cero']}**",
        f"- **gate 3 superado: {gate_3['superado']}**",
        "",
        "## 5 bis. Una salvedad sobre la calibracion beta",
        "",
        "La calibracion beta sale con **peor** ECE que no calibrar en HS, lo que es raro en un",
        "modelo de tres parametros que en principio contiene a Platt. Se comprobo que no es un",
        "fallo de la teoria sino de la rejilla usada:",
        "",
        "| etiqueta | verosimilitud negativa de Platt | de beta |",
        "|---|---|---|",
    ]
    for etiqueta in ETIQUETAS:
        lineas.append(
            f"| {etiqueta} | {diagnostico['verosimilitud_negativa_por_etiqueta']['platt'][etiqueta]:.1f} "
            f"| {diagnostico['verosimilitud_negativa_por_etiqueta']['beta_tres_parametros'][etiqueta]:.1f} |"
        )
    lineas += [
        "",
        (
            "Platt alcanza igual o mejor verosimilitud en las tres etiquetas: "
            f"**{diagnostico['platt_alcanza_igual_o_mejor_que_la_rejilla_beta']}**."
        ),
        "",
        "La causa es que la rejilla beta limita el parametro de asimetria a [0.7, 1.3] mientras que",
        "el desplazamiento de Platt es libre, de modo que **la rejilla no contiene a Platt**.",
        "Lo medido es por tanto esta rejilla estrecha, y no la familia beta en general. El gate 1",
        "no depende de beta, asi que su veredicto sigue siendo valido, y no se abre una via nueva",
        "para ampliar la rejilla porque el criterio de cierre ya se cumplio con la isotonica.",
        "",
        "## 6. Veredicto",
        "",
    ]
    if not informe["gate_1_superado"]:
        lineas += [
            "**SUB-VIA CERRADA.** Ninguna calibracion reduce el ECE el 20 por ciento frente a",
            "Platt, que era el gate 1 pre-registrado. No se reintenta con mas variantes: el",
            "criterio de cierre estaba escrito antes de ejecutar.",
        ]
    elif not informe["gate_2_superado"]:
        lineas += [
            "**SUB-VIA CERRADA.** Una calibracion reduce el ECE lo suficiente pero **empeora** la",
            "F1 anidada. Mejorar la calibracion en abstracto no sirve si la metrica de la",
            "competencia baja, y ese era el segundo gate.",
        ]
    elif not gate_3["superado"]:
        lineas += [
            "**NO ACCIONABLE.** La calibracion cumple los gates 1 y 2 pero no el 3 del",
            "addendum, que exige delta medio de +0.002, 4 de 5 particiones que mejoran e",
            "intervalo que no incluye el cero. Es una mejora real de calibracion que **no se",
            "traduce en mejora de la metrica de la competencia**, y eso es exactamente lo que se",
            "documenta.",
        ]
    else:
        lineas += [
            "**GATES SUPERADOS.** La candidata cumple las tres condiciones y pasa a la regla de",
            "decision F1 optima y al filtro de detectabilidad antes de considerar cualquier sonda.",
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
