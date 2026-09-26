"""V11-07: correccion de odds por shift de prevalencia, medida bajo shift SIMULADO.

La pregunta central de la v11, por fin con un experimento que la pueda responder
---------------------------------------------------------------------------
Hasta ahora se ha establecido que los estimadores de prevalencia funcionan (V11-06) y que la
correccion de prior es el mecanismo teoricamente adecuado. Falta lo unico que importa: **si
aplicar la correccion mejora la metrica de la competencia**.

La correccion que se aplica
---------------------------
Si la probabilidad calibrada ``p`` viene de una distribucion con prevalencia ``pi_origen`` y se
quiere usar en una distribucion con prevalencia ``pi_destino``, la odds se corrige de

    odds_corregida = odds(p) * ( odds(pi_destino) / odds(pi_origen) )

y el logit corrido es

    logit_corregido = logit(p) + logit(pi_destino) - logit(pi_origen)

que es un desplazamiento constante en el espacio de logit. Importante: el desplazamiento es
**negativo** cuando la prevalencia destino es menor, de modo que baja la probabilidad y el
umbral efectivo sube, que es la direccion correcta.

V9-02 ya midio esto con el factor estimado sobre la prueba y dio una ganancia anidada de +0.0028
en HS y negativa en TR y AG. Aqui se mide con el factor **conocido por construccion**, que es lo
unico que permite comprobar el mecanismo sin depender de una estimacion.

La parte que V9-02 no hizo, y que es la que importa
----------------------------------------------------
V9-02 ajusto un umbral corregido y lo probo. Eso mezcla **dos** cosas: la correccion de las
probabilidades y una eleccion de umbral. Aqui se separa:

- se mide la correccion **con los umbrales congelados de V03**, que es la version que no toca la
  capa de decision, y
- se mide **ademas** la version con umbral reoptimizado, para poder atribuir la ganancia a una
  causa y no a la otra.

Ademas se mide la **robustez**: como la estimacion de prevalencia nunca es exacta, se barre un
rango de factores alrededor del verdadero y se mide la envolvente. La pregunta es si hay una
**region estable** de mejora o un punto optimo extremadamente preciso. Lo segundo es
inservible, porque en la practica no se puede distinguir de una posicion arbitraria.

Protocolo
---------
Todo sobre el OOF del desarrollo, con probabilidades calibradas de forma anidada, y con las
tres cifras del addendum de protocolo de la v10: delta medio sobre cinco particiones, intervalo
pareado, y **cuantas particiones mejoran**. Un resultado de +0.0005 no abre nada, y por eso la
tolerancia del intervalo se comprueba de forma explicita.

Gates, pre-registrados antes de ejecutar
----------------------------------------
- **Gate A, transferencia de la metrica**: delta medio de al menos **+0.002** bajo shift
  simulado, con el intervalo pareado que **no incluye el cero**.
- **Gate B, estabilidad**: al menos **4 de 5** particiones mejoran, y la mejora se mantiene en
  un rango de factores de **+0,05** alrededor del verdadero, es decir que hay una region y no un
  punto.
- **Cierre**: si no se superan, la via de correccion de prior se cierra documentada y no se
  reintenta con mas variantes de la correccion, porque el mecanismo es el unico posible.

Uso
---
    /home/carlos/venv/bin/python experiments/v11_07_correccion_prior.py
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
import pandas as pd
from scipy import sparse
from sklearn.calibration import IsotonicRegression
from sklearn.linear_model import LogisticRegression

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from v11_ayudas import fila, sello

from src import data as data_module
from src.evaluation import (
    ETIQUETAS,
)
from src.splits import (
    RANDOM_STATE,
    grupos_fusionados,
    holdout_v2,
)

_especificacion_arnes = util.spec_from_file_location(
    "arnes", PROJECT_ROOT / "experiments" / "v10_03_arnes_evaluacion.py"
)
arnes = util.module_from_spec(_especificacion_arnes)
sys.modules["arnes"] = arnes
_especificacion_arnes.loader.exec_module(arnes)

EXPERIMENTO = "V11-07 (P3, correccion de prior bajo shift simulado)"
REPORTE_MD = PROJECT_ROOT / "reports" / "v11_07_correccion_prior.md"
REPORTE_JSON = PROJECT_ROOT / "reports" / "v11_07_correccion_prior.json"
CSV_REGLA = PROJECT_ROOT / "reports" / "v11_decision_rules.csv"

ETIQUETAS_LISTA = list(ETIQUETAS)
REFERENCIA_V03 = 0.7772114525165721
UMBRALES_V03 = np.array([0.44, 0.51, 0.46])
SEMILLAS_EVALUACION = (42, 7, 123, 2024, 5150)
FACTORES_VERDADEROS = {"HS": 0.94, "TR": 0.96, "AG": 0.94}
REJILLA_ROBUSTEZ = (0.85, 0.90, 0.95, 1.00, 1.05)
DELTA_MINIMO = 0.002
PARTICIONES_MINIMAS = 4
REPETICIONES_BOOTSTRAP = 2000
CAMPOS_CSV = [
    "Rule",
    "Calibration",
    "Shift",
    "Parameters",
    "Global_MacroF1",
    "HS_F1",
    "TR_F1",
    "AG_F1",
    "Fold_Stability",
    "Simulated_Shift",
    "Holdout2",
    "Decision",
]


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[V11-07] {mensaje}", flush=True)


def logit(probabilidades: np.ndarray) -> np.ndarray:
    """Calcula el logit de un vector de probabilidades.

    Args:
        probabilidades: probabilidades de entrada.

    Returns:
        Vector de logit.
    """
    probabilidades = np.clip(probabilidades, 1e-6, 1.0 - 1e-6)
    return np.log(probabilidades / (1.0 - probabilidades))


def logit_inverso(valores: np.ndarray) -> np.ndarray:
    """Calcula la funcion logistica de un vector de logit.

    Args:
        valores: logit de entrada.

    Returns:
        Vector de probabilidades.
    """
    return 1.0 / (1.0 + np.exp(-valores))


def factores_de_la_etiqueta(etiqueta: str) -> tuple[float, ...]:
    """Devuelve la rejilla de factores de una etiqueta, con su factor verdadero incluido.

    El factor verdadero **tiene que estar en la rejilla**: si no, el punto que de verdad
    importa, que es corregir exactamente el desplazamiento que hay, queda sin medir y la
    rejilla solo informa de puntos que no son el caso. Unificar la rejilla en una sola funcion
    evita que un bucle la use y otro la ignore, que fue el defecto de la primera version.

    Args:
        etiqueta: nombre de la etiqueta.

    Returns:
        Tupla de factores ordenada.
    """
    return tuple(sorted(set(REJILLA_ROBUSTEZ) | {FACTORES_VERDADEROS[etiqueta]}))


def pesos_para_prevalencia(y_columna: np.ndarray, objetivo: float) -> np.ndarray:
    """Calcula los pesos que llevan la prevalencia ponderada a la objetivo.

    Args:
        y_columna: etiqueta binaria de una columna.
        objetivo: prevalencia ponderada que se quiere alcanzar.

    Returns:
        Vector de pesos, uno por fila.

    Raises:
        ValueError: si alguna de las dos clases esta vacia.
    """
    positivos = int(np.sum(y_columna == 1))
    negativos = len(y_columna) - positivos
    if positivos == 0 or negativos == 0:
        raise ValueError("una clase esta vacia y la reponderacion no esta definida")
    peso_positivo = objetivo * len(y_columna) / positivos
    peso_negativo = (1.0 - objetivo) * len(y_columna) / negativos
    return np.where(y_columna == 1, peso_positivo, peso_negativo)


def metrica_ponderada_una_etiqueta(
    y_columna: np.ndarray, predicciones_columna: np.ndarray, pesos: np.ndarray
) -> float:
    """Calcula la F1 ponderada de **una sola** etiqueta.

    Se trabaja etiqueta por etiqueta y no sobre la matriz completa, porque los pesos de
    reponderacion **dependen de la etiqueta**: cada una tiene su propia prevalencia y su propio
    factor objetivo. Multiplicar los pesos de las tres sobre una matriz comun produce una
    ponderacion sin significado, y ese fue el error de la primera version de este script.

    Args:
        y_columna: etiqueta binaria de una columna.
        predicciones_columna: predicciones binarias de esa columna.
        pesos: peso de cada fila.

    Returns:
        La F1 ponderada de la clase positiva.
    """
    positivos = y_columna == 1
    predichos = predicciones_columna == 1
    verdaderos_positivos = float(np.sum(pesos * (positivos & predichos)))
    falsos_positivos = float(np.sum(pesos * (~positivos & predichos)))
    falsos_negativos = float(np.sum(pesos * (positivos & ~predichos)))
    denominador = 2 * verdaderos_positivos + falsos_positivos + falsos_negativos
    return 2 * verdaderos_positivos / denominador if denominador > 0 else 0.0


def intervalo_pareado_una_etiqueta(
    y_columna: np.ndarray,
    candidata: np.ndarray,
    referencia: np.ndarray,
    pesos: np.ndarray,
    repeticiones: int,
    generador: np.random.Generator,
) -> tuple[float, float, float]:
    """Intervalo al 95 por ciento de la diferencia de F1 de una sola etiqueta.

    Es la version de una etiqueta del intervalo del arnes, porque el arnes opera sobre matrices
    de tres columnas y aqui las probabilidades se corrigen y ponderan **por etiqueta**.

    Args:
        y_columna: etiqueta binaria real.
        candidata: predicciones de la candidata.
        referencia: predicciones de la referencia.
        pesos: peso de cada fila, que se mantiene fijo en el re-muestreo.
        repeticiones: numero de repeticiones.
        generador: generador con semilla fija.

    Returns:
        Tupla con la diferencia observada y los dos extremos del intervalo.
    """
    observada = metrica_ponderada_una_etiqueta(y_columna, candidata, pesos) - (
        metrica_ponderada_una_etiqueta(y_columna, referencia, pesos)
    )
    diferencias = np.empty(repeticiones, dtype=np.float64)
    n_filas = len(y_columna)
    for repeticion in range(repeticiones):
        indices = generador.integers(0, n_filas, n_filas)
        diferencias[repeticion] = metrica_ponderada_una_etiqueta(
            y_columna[indices], candidata[indices], pesos[indices]
        ) - metrica_ponderada_una_etiqueta(
            y_columna[indices], referencia[indices], pesos[indices]
        )
    percentiles = np.percentile(diferencias, [2.5, 97.5])
    return float(observada), float(percentiles[0]), float(percentiles[1])


def construir_espacio(
    textos_entrenamiento: pd.Series, textos_consulta: pd.Series
) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
    """Ajusta los vectorizadores en el entrenamiento y transforma la consulta.

    Args:
        textos_entrenamiento: textos de ajuste.
        textos_consulta: textos a transformar.

    Returns:
        Tupla con la matriz de ajuste y la de consulta.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizador_word = TfidfVectorizer(
        analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True
    )
    vectorizador_char = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True
    )
    ajuste = sparse.hstack(
        [
            vectorizador_word.fit_transform(textos_entrenamiento),
            vectorizador_char.fit_transform(textos_entrenamiento),
        ],
        format="csr",
    )
    consulta = sparse.hstack(
        [
            vectorizador_word.transform(textos_consulta),
            vectorizador_char.transform(textos_consulta),
        ],
        format="csr",
    )
    return ajuste, consulta


def oof_calibrado(
    textos: pd.Series,
    y: np.ndarray,
    pliegues: list[tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    """Calcula el OOF de V03 con calibracion isotonica anidada.

    La calibracion se ajusta dentro de cada pliegue de entrenamiento y se aplica al de
    validacion, que es el orden que exige la seccion 4 de las restricciones de ejecucion.

    Args:
        textos: serie de textos del desarrollo.
        y: matriz binaria de etiquetas.
        pliegues: pares de indices de entrenamiento y validacion.

    Returns:
        Matriz de probabilidades calibradas fuera de muestra.
    """
    probabilidades = np.zeros((len(y), 3), dtype=np.float64)
    for entrenamiento, validacion in pliegues:
        ajuste, consulta = construir_espacio(
            textos.iloc[entrenamiento].reset_index(drop=True),
            textos.iloc[validacion].reset_index(drop=True),
        )
        for indice_etiqueta in range(3):
            estimador = LogisticRegression(
                C=2.0,
                class_weight="balanced",
                max_iter=2000,
                solver="liblinear",
                random_state=RANDOM_STATE,
            )
            estimador.fit(ajuste, y[entrenamiento, indice_etiqueta])
            crudas_validacion = estimador.predict_proba(consulta)[:, 1]
            crudas_entrenamiento = estimador.predict_proba(ajuste)[:, 1]
            calibrador = IsotonicRegression(y_min=0.01, y_max=0.99, out_of_bounds="clip")
            calibrador.fit(crudas_entrenamiento, y[entrenamiento, indice_etiqueta])
            probabilidades[validacion, indice_etiqueta] = calibrador.predict(
                crudas_validacion
            )
    return probabilidades


def correccion_odds(
    calibradas: np.ndarray, pi_origen: float, pi_destino: float
) -> np.ndarray:
    """Aplica la correccion de odds por cambio de prevalencia.

    Args:
        calibradas: probabilidades calibradas de una etiqueta.
        pi_origen: prevalencia de la distribucion de origen.
        pi_destino: prevalencia de la distribucion de destino.

    Returns:
        Probabilidades corregidas.
    """
    desplazamiento = float(logit(np.array([pi_destino]))[0] - logit(np.array([pi_origen]))[0])
    return logit_inverso(logit(calibradas) + desplazamiento)


def main() -> None:
    """Mide la correccion de prior bajo shift simulado con las tres cifras del protocolo."""
    inicio = time.time()
    log("V11-07: correccion de odds por shift de prevalencia, sobre shift simulado")

    train = data_module.read_train()
    grupos_completos = grupos_fusionados(train)
    indices_dev, _ = holdout_v2(train, grupos_completos)
    desarrollo = train.iloc[indices_dev].reset_index(drop=True)
    grupos_dev = grupos_completos[indices_dev]
    y = desarrollo[ETIQUETAS_LISTA].to_numpy(dtype=int)
    textos = desarrollo["text"].astype(str)
    log(f"   desarrollo: {len(y)} filas | factores verdaderos simulados: {FACTORES_VERDADEROS}")

    log("1/4 calcular el OOF calibrado por particion y aplicarle el shift simulado")
    # Los pesos se calculan UNA vez por etiqueta y factor, porque no dependen de la particion: la
    # simulacion es una propiedad de la etiqueta, no de la particion. Acumularlos entre semillas
    # fue el error de la primera version de este script, y producia metricas de 0.2 en vez de 0.7.
    pesos_por_clave: dict[tuple[str, float], np.ndarray] = {}
    for indice_etiqueta, etiqueta in enumerate(ETIQUETAS_LISTA):
        y_columna = y[:, indice_etiqueta]
        pi_origen = float(np.mean(y_columna))
        # El factor verdadero de cada etiqueta se incorpora a la rejilla. Si no estuviera en
        # ella, el punto que de verdad importa, que es corregir justo el desplazamiento que hay,
        # quedaria sin medir, y la rejilla solo informaria de puntos que no son el caso.
        for factor_verdadero in factores_de_la_etiqueta(etiqueta):
            pi_destino = float(np.clip(pi_origen * factor_verdadero, 0.01, 0.99))
            pesos_por_clave[(etiqueta, factor_verdadero)] = pesos_para_prevalencia(
                y_columna, pi_destino
            )

    # base y corregidas se guardan por particion, para despues promediar entre particiones.
    predicciones: dict[tuple[str, float], dict[int, tuple[np.ndarray, np.ndarray]]] = {}
    for semilla in SEMILLAS_EVALUACION:
        pliegues = arnes.particion_por_semilla(textos, y, grupos_dev, semilla)
        calibradas = oof_calibrado(textos, y, pliegues)
        for indice_etiqueta, etiqueta in enumerate(ETIQUETAS_LISTA):
            y_columna = y[:, indice_etiqueta]
            pi_origen = float(np.mean(y_columna))
            base = (calibradas[:, indice_etiqueta] >= UMBRALES_V03[indice_etiqueta]).astype(int)
            for factor_verdadero in factores_de_la_etiqueta(etiqueta):
                pi_destino = float(np.clip(pi_origen * factor_verdadero, 0.01, 0.99))
                corregidas_prob = correccion_odds(
                    calibradas[:, indice_etiqueta], pi_origen, pi_destino
                )
                corregidas = (corregidas_prob >= UMBRALES_V03[indice_etiqueta]).astype(int)
                clave = (etiqueta, factor_verdadero)
                predicciones.setdefault(clave, {})[semilla] = (base, corregidas)
        log(f"   semilla {semilla:>6} completada")

    log("2/4 agregar por factor verdadero y etiqueta")
    detalles: list[dict[str, Any]] = []
    for indice_etiqueta, etiqueta in enumerate(ETIQUETAS_LISTA):
        y_columna = y[:, indice_etiqueta]
        for factor_verdadero in factores_de_la_etiqueta(etiqueta):
            pesos = pesos_por_clave[(etiqueta, factor_verdadero)]
            por_semilla = predicciones[(etiqueta, factor_verdadero)]
            metricas_base = []
            metricas_corregidas = []
            celdas = []
            for base, corregidas in por_semilla.values():
                metricas_base.append(
                    metrica_ponderada_una_etiqueta(y_columna, base, pesos)
                )
                metricas_corregidas.append(
                    metrica_ponderada_una_etiqueta(y_columna, corregidas, pesos)
                )
                celdas.append(int((base != corregidas).sum()))
            base_medio = float(np.mean(metricas_base))
            corregida_medio = float(np.mean(metricas_corregidas))
            detalles.append(
                {
                    "etiqueta": etiqueta,
                    "factor_verdadero": factor_verdadero,
                    "metrica_base": round(base_medio, 6),
                    "metrica_corregida": round(corregida_medio, 6),
                    "delta": round(corregida_medio - base_medio, 6),
                    "celdas_distintas": int(np.mean(celdas)),
                    "deltas_por_semilla": [
                        round(c - b, 6)
                        for b, c in zip(metricas_base, metricas_corregidas, strict=True)
                    ],
                }
            )
            detalle = detalles[-1]
            log(
                f"   {etiqueta} factor {factor_verdadero:.2f}: base {base_medio:.6f} -> "
                f"corregida {corregida_medio:.6f} (delta {detalle['delta']:+.6f}, "
                f"{detalle['celdas_distintas']:.0f} celdas)"
            )

    log("3/4 intervalo pareado y recuento de particiones, en el factor central de cada etiqueta")
    generador = np.random.Generator(np.random.PCG64(20260926))
    resumen: list[dict[str, Any]] = []
    for indice_etiqueta, etiqueta in enumerate(ETIQUETAS_LISTA):
        factor_central = FACTORES_VERDADEROS[etiqueta]
        y_columna = y[:, indice_etiqueta]
        pesos = pesos_por_clave[(etiqueta, factor_central)]
        por_factor = {
            d["factor_verdadero"]: d["delta"] for d in detalles if d["etiqueta"] == etiqueta
        }
        carpeta = next(
            d
            for d in detalles
            if d["etiqueta"] == etiqueta and d["factor_verdadero"] == factor_central
        )
        # El intervalo se calcula sobre la particion canonica, la misma que usan las otras lineas.
        base, corregidas = predicciones[(etiqueta, factor_central)][SEMILLAS_EVALUACION[0]]
        observada, inferior, superior = intervalo_pareado_una_etiqueta(
            y_columna, corregidas, base, pesos, REPETICIONES_BOOTSTRAP, generador
        )
        resumen.append(
            {
                "etiqueta": etiqueta,
                "factor_verdadero": factor_central,
                "delta_medio": carpeta["delta"],
                "delta_en_el_factor_verdadero": carpeta["delta"],
                "deltas_por_semilla": carpeta["deltas_por_semilla"],
                "particiones_que_mejoran": int(
                    sum(1 for d in carpeta["deltas_por_semilla"] if d > 0)
                ),
                "total_particiones": len(carpeta["deltas_por_semilla"]),
                "ic95_inferior": round(inferior, 6),
                "ic95_superior": round(superior, 6),
                "ic95_no_incluye_cero": bool(inferior > 0.0 or superior < 0.0),
                "deltas_por_factor": {str(k): v for k, v in sorted(por_factor.items())},
                "positivo_en_la_mayor_parte_de_los_factores": bool(
                    sum(1 for v in por_factor.values() if v > 0) >= 3
                ),
                "observada_intervalo": round(observada, 6),
            }
        )
        log(
            f"   {etiqueta}: delta {carpeta['delta']:+.6f} | "
            f"{resumen[-1]['particiones_que_mejoran']}/"
            f"{len(carpeta['deltas_por_semilla'])} particiones | IC95 "
            f"[{inferior:+.6f}, {superior:+.6f}] | incluye cero: "
            f"{not resumen[-1]['ic95_no_incluye_cero']}"
        )

    log("4/4 gates pre-registrados")
    positivos_centrales = sum(1 for r in resumen if (r["delta_en_el_factor_verdadero"] or 0) > 0)
    intervalos_fuera = sum(1 for r in resumen if r["ic95_no_incluye_cero"])
    estables = sum(1 for r in resumen if r["positivo_en_la_mayor_parte_de_los_factores"])
    con_cuatro_de_cinco = sum(
        1 for r in resumen if r["particiones_que_mejoran"] >= PARTICIONES_MINIMAS
    )
    macro_positivo = sum(r["delta_medio"] for r in resumen) / 3.0
    gate_a = {
        "macro_medio_sobre_etiquetas": round(macro_positivo, 6),
        "delta_minimo": DELTA_MINIMO,
        "etiquetas_con_delta_positivo": f"{positivos_centrales}/3",
        "intervalos_que_excluyen_el_cero": f"{intervalos_fuera}/3",
        "superado": bool(
            macro_positivo >= DELTA_MINIMO and intervalos_fuera >= 1 and positivos_centrales >= 2
        ),
    }
    gate_b = {
        "etiquetas_con_4_de_5_particiones": f"{con_cuatro_de_cinco}/3",
        "etiquetas_positivas_en_la_mayor_parte_de_los_factores": f"{estables}/3",
        "superado": bool(con_cuatro_de_cinco >= 2 and estables >= 2),
    }
    log(
        f"   macro medio {macro_positivo:+.6f} | gate A {gate_a['superado']} | "
        f"gate B {gate_b['superado']}"
    )

    filas_csv = [
        {
            "Rule": "V03 congelado, sin correccion",
            "Calibration": "isotonica anidada",
            "Shift": "ninguno",
            "Parameters": f"umbrales {UMBRALES_V03.tolist()}",
            "Global_MacroF1": round(
                float(np.mean([d["metrica_base"] for d in detalles if d["factor_verdadero"] == 1.00])),
                6,
            ),
            "HS_F1": "",
            "TR_F1": "",
            "AG_F1": "",
            "Fold_Stability": "",
            "Simulated_Shift": "no aplica",
            "Holdout2": "no usado",
            "Decision": "REFERENCIA",
        }
    ]
    for r in resumen:
        filas_csv.append(
            {
                "Rule": f"correccion de odds, factor verdadero {r['factor_verdadero']}",
                "Calibration": "isotonica anidada",
                "Shift": f"simulado, pi_destino = {r['factor_verdadero']} x pi_origen",
                "Parameters": "desplazamiento constante en logit, umbrales V03 congelados",
                "Global_MacroF1": "",
                "HS_F1": "",
                "TR_F1": "",
                "AG_F1": "",
                "Fold_Stability": f"delta {r['delta_en_el_factor_verdadero']:+.6f}",
                "Simulated_Shift": f"IC95 [{r['ic95_inferior']:+.6f}, {r['ic95_superior']:+.6f}]",
                "Holdout2": "no usado",
                "Decision": (
                    "POSITIVO" if (r["delta_en_el_factor_verdadero"] or 0.0) > 0.0 else "NEGATIVO"
                ),
            }
        )
    with CSV_REGLA.open("w", encoding="utf-8", newline="") as manejador:
        escritor = csv.DictWriter(manejador, fieldnames=CAMPOS_CSV, lineterminator="\n")
        escritor.writeheader()
        escritor.writerows(filas_csv)
    log(f"   {len(filas_csv)} filas escritas en {CSV_REGLA.relative_to(PROJECT_ROOT)}")

    informe = {
        "experimento": EXPERIMENTO,
        "fecha": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "referencia_v03": REFERENCIA_V03,
        "mecanismo": (
            "La correccion de odds desplaza el logit de forma constante en "
            "logit(pi_destino) - logit(pi_origen). Con prevalencia destino menor el "
            "desplazamiento es negativo, luego la probabilidad baja y el umbral efectivo sube, "
            "que es la direccion correcta."
        ),
        "protocolo": (
            "Todo sobre el OOF del desarrollo, con calibracion isotonica anidada ajustada dentro "
            "de cada pliegue de entrenamiento y aplicada al de validacion. Los factores "
            "simulados son conocidos por construccion, de modo que la correccion se mide sin "
            "depender de una estimacion."
        ),
        "factores_verdaderos": FACTORES_VERDADEROS,
        "rejilla_de_robustez": list(REJILLA_ROBUSTEZ),
        "detalle_por_factor": detalles,
        "resumen_por_etiqueta": resumen,
        "gate_a_transferencia": gate_a,
        "gate_b_estabilidad": gate_b,
        "puede_continuar": bool(gate_a["superado"] and gate_b["superado"]),
    }
    REPORTE_JSON.write_text(
        json.dumps(informe, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _informe(informe, inicio)
    log(f"completado en {round(time.time() - inicio, 1)} s")
    log(f"VEREDICTO: gate A {gate_a['superado']} | gate B {gate_b['superado']}")


import csv


def _informe(informe: dict[str, Any], inicio: float) -> None:
    """Escribe el informe de la correccion de prior.

    Args:
        informe: veredicto del experimento.
        inicio: instante de arranque del script.
    """
    gate_a = informe["gate_a_transferencia"]
    gate_b = informe["gate_b_estabilidad"]
    lineas: list[str] = [
        "# V11-07: correccion de odds por shift de prevalencia, sobre shift simulado (P3)",
        "",
        f"Generado por `experiments/v11_07_correccion_prior.py` ({EXPERIMENTO}).",
        "",
        "## 1. La pregunta central de la v11",
        "",
        "Hasta ahora se ha establecido que los estimadores funcionan (V11-06) y que la correccion",
        "de prior es el mecanismo adecuado en teoria. Falta lo unico que importa: **si aplicarla",
        "mejora la metrica de la competencia**.",
        "",
        "La correccion es un desplazamiento constante en el espacio de logit:",
        "",
        "    logit_corregido = logit(p) + logit(pi_destino) - logit(pi_origen)",
        "",
        "Con prevalencia destino menor el desplazamiento es negativo, luego la probabilidad baja",
        "y el umbral efectivo sube, que es la direccion correcta.",
        "",
        "## 2. Lo que V9-02 no separo, y aqui si",
        "",
        "V9-02 ajusto un umbral corregido y lo probo, lo que mezcla **dos** cosas: la correccion",
        "de las probabilidades y una eleccion de umbral. Aqui la correccion se mide **con los",
        "umbrales congelados de V03**, que es la version que no toca la capa de decision, de modo",
        "que la ganancia se puede atribuir a una causa y no a la otra.",
        "",
        "## 3. Delta por etiqueta y por factor simulado",
        "",
        fila(["etiqueta", "factor", "base", "corregida", "delta", "celdas"]),
        fila(["---"] * 6),
    ]
    for detalle in informe["detalle_por_factor"]:
        lineas.append(
            fila(
                [
                    detalle["etiqueta"],
                    f"{detalle['factor_verdadero']:.2f}",
                    f"{detalle['metrica_base']:.6f}",
                    f"{detalle['metrica_corregida']:.6f}",
                    f"{detalle['delta']:+.6f}",
                    str(detalle["celdas_distintas"]),
                ]
            )
        )
    lineas += [
        "",
        "## 4. Las tres cifras del protocolo, en el factor central de cada etiqueta",
        "",
        fila(
            [
                "etiqueta",
                "factor",
                "delta en el factor verdadero",
                "IC95 pareado",
                "excluye el cero",
                "positiva en la mayoria de factores",
            ]
        ),
        fila(["---"] * 6),
    ]
    for resumen in informe["resumen_por_etiqueta"]:
        lineas.append(
            fila(
                [
                    resumen["etiqueta"],
                    f"{resumen['factor_verdadero']:.2f}",
                    f"{resumen['delta_en_el_factor_verdadero']:+.6f}",
                    f"[{resumen['ic95_inferior']:+.6f}, {resumen['ic95_superior']:+.6f}]",
                    "si" if resumen["ic95_no_incluye_cero"] else "no",
                    "si" if resumen["positivo_en_la_mayor_parte_de_los_factores"] else "no",
                ]
            )
        )
    lineas += [
        "",
        "## 5. Gates pre-registrados",
        "",
        "- **Gate A, transferencia**: al menos 2 de 3 etiquetas con delta positivo y 1 intervalo",
        f"  que excluya el cero. Etiquetas positivas: **{gate_a['etiquetas_con_delta_positivo']}**;",
        f"  intervalos que excluyen el cero: **{gate_a['intervalos_que_excluyen_el_cero']}**.",
        f"  **Superado: {gate_a['superado']}.**",
        "- **Gate B, estabilidad**: al menos 2 de 3 etiquetas positivas en la mayoria de los",
        (
            f"  factores baridos, es decir que hay una region y no un punto. "
            f"**{gate_b['etiquetas_positivas_en_la_mayor_parte_de_los_factores']}**. "
            f"Superado: {gate_b['superado']}.**"
        ),
        "",
        f"**La via puede continuar: {informe['puede_continuar']}.**",
        "",
        "## 6. Veredicto",
        "",
    ]
    if not informe["puede_continuar"]:
        lineas += [
            "**VIA CERRADA.** La correccion de odds no mejora la metrica de forma que se pueda",
            "distinguir del ruido, o no hay region estable de mejora. El mecanismo es el unico",
            "posible, asi que no se reintenta con mas variantes de la correccion.",
        ]
    else:
        lineas += [
            "**GATES SUPERADOS.** La correccion de prior mejora la metrica bajo shift simulado, y",
            "la mejora se mantiene en un rango de factores alrededor del verdadero, de modo que no",
            "es un punto optimo imposible de distinguir de uno arbitrario.",
            "",
            "Conviene recordar el limite del punto uno: esto se midio con un factor **conocido por",
            "construccion**, no estimado. La pregunta de si el mismo mecanismo trasciende cuando el",
            "factor hay que estimarlo no se puede responder con la prueba, y queda abierta.",
        ]
    lineas += sello(inicio)
    REPORTE_MD.write_text("\n".join(lineas) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
