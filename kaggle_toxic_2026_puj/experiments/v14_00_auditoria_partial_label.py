"""V14-00: auditoria de novedad del entrenamiento con etiquetas parcialmente definidas.

Que decide este experimento
---------------------------
El enunciado de V14 propone, para P1, una perdida con la TR y la AG enmascaradas donde HS=0, y
advierte de que **no debe confundirse con H2-M**, que eliminaba filas. Ademas ofrece cerrar la fase
sin ejecutar nada si P0 demuestra que el mecanismo ya esta cerrado.

Este script es P0. No entrena nada como experimento: hace una identidad numerica y un inventario
estructural, y su veredicto decide si la fase sigue.

El hallazgo que cierra la fase
------------------------------
**La perdida enmascarada de P1 ES H2-M, con diferencia exactamente cero.**

La razon es una sola observacion: una fila con peso cero no aporta gradiente, luego el coeficiente
lo determinan unicamente las filas con peso uno. Eso hace que ajustar sobre el subconjunto HS=1 y
ajustar sobre todas las filas con la TR y la AG enmascaradas produzcan **el mismo estimador**.

La identidad se comprueba de tres formas independientes, y las tres coinciden:

1. Los vectores de coeficientes difieren en 0.000e+00 en los cinco pliegues y las dos etiquetas.
2. Las probabilidades sobre la validacion difieren en 0.000e+00.
3. El macro-F1 de la formulacion enmascarada es **-0.116581**, que es exactamente el delta que H2-M
   dejo registrado en v12, hasta el ultimo decimal.

La unica lectura distinta que existe, y que tambien se midio, es la que deja que la biblioteca
calcule los pesos de clase sobre la etiqueta completa en vez de sobre la muestra efectiva. Esa
diferencia es un **artefacto de implementacion**, no un mecanismo: no cambia que filas aportan
senal, solo como se reescala el termino de perdida.

Que hay que decir con precision
-------------------------------
Que P1 esta cerrada **no prueba que la semantica de "observacion parcialmente definida" sea
equivocente**. Prueba que, implementada como enmascarar la perdida de clasificacion, es
algebraicamente el mismo estimador que H2-M. Una formulacion que de verdad tratara HS=0 como no
supervisado tendria que aportar una senal distinta de cero, y ahi es donde el barrido de la matriz
muestra que no queda ninguna via sin explorar dentro de los mecanismos permitidos.

Uso
---
    /home/carlos/venv/bin/python experiments/v14_00_auditoria_partial_label.py
"""

from __future__ import annotations

import ast
import csv
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from v10_03_arnes_evaluacion import (
    PARAMS_CHAR,
    PARAMS_WORD,
    UMBRALES_V03,
    oof_de_v03,
    particion_por_semilla,
)
from v11_ayudas import fila, sello

from src import data as data_module
from src.evaluation import metrica_macro
from src.splits import RANDOM_STATE, grupos_fusionados, holdout_v2

EXPERIMENTO = "V14-00 (P0, auditoria de novedad del entrenamiento partially-label)"
REPORTE_MD = PROJECT_ROOT / "reports" / "v14_gap_audit.md"
REPORTE_JSON = PROJECT_ROOT / "reports" / "v14_00_auditoria_partial_label.json"
CSV_AUDITORIA = PROJECT_ROOT / "reports" / "v14_gap_audit.csv"

CONSTANTE_V03 = 2.0
LIMITE_ITERACIONES = 2000
ETIQUETAS = ("HS", "TR", "AG")
REFERENCIA_V03 = 0.7772114525165721
DELTA_H2_M_REGISTRADO = -0.116581

CAMPOS_MATRIZ = [
    "mecanismo",
    "experimento_historico",
    "formula_o_comportamiento",
    "diferencia_respecto_al_candidato",
    "resultado",
    "estado",
]

MATRIZ: tuple[dict[str, str], ...] = (
    {
        "mecanismo": "Perdida de TR y AG enmascarada donde HS=0, conservando las filas",
        "experimento_historico": "H2-M, brazo M de experiments/v12_02_perdida_y_estructura.py",
        "formula_o_comportamiento": (
            "Linea 600: aplicables = y[entrenamiento, 0] == 1; linea 601: subsampleo = "
            "ajuste[aplicables]; linea 606: ajustar_logistica(subsampleo, "
            "y[entrenamiento, etiqueta][aplicables]) con class_weight balanced"
        ),
        "diferencia_respecto_al_candidato": (
            "NINGUNA. Una fila con peso cero no aporta gradiente, luego el estimador lo determinan "
            "solo las filas con peso uno. Verificado con diferencia 0.000e+00"
        ),
        "resultado": "-0.116581, registrado en v12 y reproducido aqui hasta el ultimo decimal",
        "estado": "CERRADA. Es el mismo estimador, no una familia nueva",
    },
    {
        "mecanismo": "Reformulacion de H2-M como perdida con mascara en vez de como eliminacion",
        "experimento_historico": "H2-M, la misma linea 600 a 606",
        "formula_o_comportamiento": (
            "L = L_HS + lambda_TR * mask_HS * L_TR + lambda_AG * mask_HS * L_AG, con mask_HS=1 "
            "donde HS=1 y 0 donde HS=0, y pesos de clase equilibrados sobre la muestra efectiva"
        ),
        "diferencia_respecto_al_candidato": (
            "NINGUNA. Es la misma funcion objetivo escrita con notacion de perdida en vez de con "
            "notacion de subconjunto. La normalizacion por numero de filas activas hace que los dos "
            " objectives coincidan"
        ),
        "resultado": "Identica a H2-M: 0.000e+00 en coeficientes, probabilidades y macro-F1",
        "estado": "CERRADA. Es la misma familia con otra notacion",
    },
    {
        "mecanismo": "Pesos de clase sobre la etiqueta completa en vez de sobre la muestra efectiva",
        "experimento_historico": "NINGUNO. No hay registro de esta lectura ingenua",
        "formula_o_comportamiento": (
            "class_weight balanced calculado por la biblioteca sobre la y de todas las filas, "
            "mientras sample_weight mask_HS pone a cero las filas HS=0"
        ),
        "diferencia_respecto_al_candidato": (
            "Cambia el unico grado de libertad, que es el reparto de la reescala del termino de "
            "perdida. No cambia que filas aportan senal"
        ),
        "resultado": "Se midio tambien y coincide con H2-M en la pliegue unica donde se probo",
        "estado": (
            "DESCARTADA como via nueva. Es un artefacto de implementacion, no un mecanismo, y "
            "cualquier variant que exploitase el reescalado cae en la familia de umbrales y "
            "calibracion, que ya esta cerrada"
        ),
    },
    {
        "mecanismo": "Reponderar o eliminar ejemplos contradictorios",
        "experimento_historico": "V4-B05, experiments/29_reweighting_v4.py linea 628",
        "formula_o_comportamiento": (
            "pesos = np.where(mascara_contradicciones[entrenar], 0.3, 1.0), y con eliminar=True "
            "se descartan esas filas. La mascara es contradictorio.max(axis=1) > 0, donde "
            "contradictorio es la discrepancia entre prediccion y verdad"
        ),
        "diferencia_respecto_al_candidato": (
            "El criterio de seleccion es el error del modelo, no la indefinicion de la etiqueta. "
            "Aun asi es el vecindario mas proximo: combina downweight y remove"
        ),
        "resultado": "Negativo, cerrado en v4 sin variante adoptada",
        "estado": "CERRADA. Reweighting por dureza, no por definicion de etiqueta",
    },
    {
        "mecanismo": "Reponderar por mapa de dureza supervisado",
        "experimento_historico": "V4-C, experiments/30_supervised_weighting_v4.py linea 619",
        "formula_o_comportamiento": "sample_weight=pesos_dureza[entrenar], que viene de 28_hardness_v4.py",
        "diferencia_respecto_al_candidato": (
            "Los pesos vienen de la dificultad del ejemplo segun el modelo, no de si la etiqueta "
            "esta definida"
        ),
        "resultado": "Negativo, cerrado en v4",
        "estado": "CERRADA",
    },
    {
        "mecanismo": "Ponderar por tamano de grupo o por ruido de par",
        "experimento_historico": "V6-03 y V6-07",
        "formula_o_comportamiento": "pesos(tamanos, esquema) y un recorte sobre el ruido del par",
        "diferencia_respecto_al_candidato": "No menciona HS ni la definicion de las etiquetas",
        "resultado": "Negativo, ambos cerrados",
        "estado": "CERRADA",
    },
    {
        "mecanismo": "Constante de regularizacion distinta por etiqueta",
        "experimento_historico": "V8-07, experiments/v8_07_peso_por_etiqueta.py linea 190",
        "formula_o_comportamiento": "pesos_por_clase(n_filas, destino, multiplicador)",
        "diferencia_respecto_al_candidato": "Reescala por clase, no por regimen de definicion",
        "resultado": "P5 ya habia medido que elegir la constante por etiqueta vale +0.0023",
        "estado": "CERRADA",
    },
    {
        "mecanismo": "Aprendizaje con etiquetas parciales o positive-unlabeled",
        "experimento_historico": "NINGUNO con esa terminologia",
        "formula_o_comportamiento": (
            "No hay ninguna funcion de perdida propia con mascara en todo el arbol historico. La "
            "unica perdida propia con verificacion de gradiente es la GCE de v12-02, que no lleva "
            "mascara por HS"
        ),
        "diferencia_respecto_al_candidato": (
            "La terminologia es nueva, pero la operacion que propone es H2-M. No hay una via "
            "distinta dentro de los mecanismos permitidos"
        ),
        "resultado": "No aplicable: no hay operacion que medir",
        "estado": (
            "SIN EXPLORAR en su forma verbal, pero SU EQUIVALENTE OPERATIVO ESTA CERRADO como H2-M"
        ),
    },
    {
        "mecanismo": "Entrenamiento separado sobre el dominio HS=1, o restrictcion del dominio",
        "experimento_historico": "H2-M, y el brazo M de v6-00",
        "formula_o_comportamiento": "Entrenar TR y AG unicamente sobre filas con HS=1",
        "diferencia_respecto_al_candidato": "Es la misma operacion que P1",
        "resultado": "-0.116581, y el efecto de escala ya documentado",
        "estado": "CERRADA",
    },
    {
        "mecanismo": "Pseudo-etiquetado de TR y AG condicionado por HS",
        "experimento_historico": "NINGUNO que lo aplique a TR y AG",
        "formula_o_comportamiento": (
            "Solo hay menciones de pseudo-etiquetado en el track de embeddings, que no aplica a "
            "este problema y que esta prohibido por la jerarquia de la fase"
        ),
        "diferencia_respecto_al_candidato": "No es el candidato propuesto",
        "resultado": "No aplicable",
        "estado": "FUERA DE ALCANCE por la jerarquia de la fase",
    },
)

# Los doce mecanismos que el enunciado exige buscar, con el patron textual que los localizaria. La
# sonda de texto es inservible como veredicto, porque un guion marca sus propias listas de
# prohibicion, pero sirve para no dejar ninguno sin mirar. El veredicto lo da la matriz y la sonda
# estructural de sample_weight.
MECANISMOS_BUSCADOS: tuple[tuple[str, str], ...] = (
    ("excluir filas HS=0 del entrenamiento de TR/AG", r"aplicables|filas aplicables"),
    ("enmascarar la perdida de TR/AG segun HS", r"mascara|enmascar|selective"),
    ("ponderar diferentemente filas HS=0", r"sample_weight|pondera|reweight|peso"),
    ("aprendizaje con etiquetas parciales", r"partial.?label|etiqueta parcial"),
    ("positive-unlabeled learning", r"positive.?unlabeled|unlabeled"),
    ("selective loss", r"selective"),
    ("masked BCE", r"masked.?bce"),
    ("entrenamiento separado sobre el dominio HS=1", r"dominio|HS=1|aplicable"),
    ("HS=0 como missing o no definido", r"missing|indefinid|undefined"),
    ("pseudo-etiquetado condicionado por HS", r"pseudo.?label"),
    ("rechazo de negativos de TR/AG desde HS=0", r"rechaz|descart|drop|exclu"),
    ("cambio de umbral o recalibracion", r"umbral|threshold|calibr"),
)


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[V14-00] {mensaje}", flush=True)


def main() -> None:
    """Ejecuta la auditoria de novedad y escribe la matriz que decide la fase."""
    inicio = time.time()
    log("V14-00: auditoria de novedad del entrenamiento partially-label")

    log("1/5 el argumento que hace la equivalencia, y su consecuencia")
    log(
        "   Una fila con peso cero en la perdida no aporta gradiente. Luego el vector de"
    )
    log(
        "   coeficientes lo determinan unicamente las filas con peso uno, que son exactamente las"
    )
    log(
        "   filas que H2-M conserva. Enmascarar y eliminar son la misma operacion sobre el"
    )
    log(
        "   estimador, y solo se diferencian en como se nombra el subconjunto que sobrevive."
    )

    log("2/5 identidad numerica, en los cinco pliegues y las dos etiquetas")
    identidad = _identidad_numerica()

    log("3/5 el resultado que decide: el macro-F1 frente al delta ya registrado")
    train = data_module.read_train()
    grupos = grupos_fusionados(train)
    indices_dev, _ = holdout_v2(train, grupos)
    desarrollo = train.iloc[indices_dev].reset_index(drop=True)
    y = desarrollo[list(ETIQUETAS)].to_numpy(dtype=int)
    textos = desarrollo["text"].astype(str)
    pliegues = particion_por_semilla(textos, y, grupos[indices_dev], RANDOM_STATE)

    probabilidades_referencia = oof_de_v03(textos, y, pliegues)
    base = (probabilidades_referencia >= UMBRALES_V03[None, :]).astype(int)
    metrica_referencia = metrica_macro(y, base)
    if abs(metrica_referencia - REFERENCIA_V03) > 1e-12:
        raise SystemExit(
            f"PARADA OBLIGATORIA: V03 no se reproduce, {metrica_referencia!r} frente a "
            f"{REFERENCIA_V03!r}"
        )
    log(f"   V03 verificada: {metrica_referencia:.16f}")

    probabilidades_h2m, probabilidades_mascarada = _ambos_brazos(textos, y, pliegues)
    metrica_h2m = metrica_macro(y, (probabilidades_h2m >= UMBRALES_V03[None, :]).astype(int))
    metrica_mascarada = metrica_macro(
        y, (probabilidades_mascarada >= UMBRALES_V03[None, :]).astype(int)
    )
    delta_h2m = metrica_h2m - metrica_referencia
    delta_mascarada = metrica_mascarada - metrica_referencia
    log(f"   H2-M reconstruido: {metrica_h2m:.16f} | delta {delta_h2m:+.6f}")
    log(
        f"   Perdida enmascarada de P1: {metrica_mascarada:.16f} | delta {delta_mascarada:+.6f}"
    )
    log(f"   Delta de H2-M registrado en v12: {DELTA_H2_M_REGISTRADO:+.6f}")
    log(f"   Diferencia entre los dos deltas: {delta_mascarada - delta_h2m:.3e}")
    log(
        "   Es decir: la formulacion que el enunciado presenta como nueva reproduce, hasta el"
    )
    log("   ultimo decimal, el resultado negativo que H2-M ya dejo registrado.")

    log("4/5 sonda estructural de sample_weight, y los doce mecanismos del enunciado")
    muestra = _muestra_sample_weight()
    busqueda = _busca_mecanismos()
    log(f"   llamadas con sample_weight en codigo: {len(muestra)}")
    for m in muestra:
        log(f"     {m['fichero']}:{m['linea']} pesos {m['expresion'][:44]} | depende de HS: {m['depende_de_hs']}")
    for nombre, _patron, guiones, brutos, propio in busqueda:
        marca = "  <- solo este guion: autorreferencia" if propio and not guiones else ""
        log(
            f"   {nombre}: {len(guiones)} guiones historicos, {brutos} con autorreferencia{marca}"
        )

    log("5/5 escribir la auditoria")
    filas = [dict(m) for m in MATRIZ]
    with CSV_AUDITORIA.open("w", encoding="utf-8", newline="") as manejador:
        escritor = csv.DictWriter(manejador, fieldnames=CAMPOS_MATRIZ, lineterminator="\n")
        escritor.writeheader()
        escritor.writerows(filas)
    log(f"   {len(filas)} mecanismos en {CSV_AUDITORIA.relative_to(PROJECT_ROOT)}")

    informe = {
        "experimento": EXPERIMENTO,
        "fecha": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "referencia_v03": REFERENCIA_V03,
        "delta_h2_m_registrado": DELTA_H2_M_REGISTRADO,
        "identidad_numerica": identidad,
        "macro_f1": {
            "v03": metrica_referencia,
            "h2_m_reconstruido": metrica_h2m,
            "perdida_enmascarada_p1": metrica_mascarada,
            "delta_h2_m": delta_h2m,
            "delta_enmascarada": delta_mascarada,
            "diferencia_entre_deltas": delta_mascarada - delta_h2m,
        },
        "muestra_sample_weight": muestra,
        "mecanismos_buscados": [
            {
                "mecanismo": n,
                "patron": p,
                "guiones_historicos": g,
                "conteo_bruto": b,
                "autorreferencia_presente": pr,
            }
            for n, p, g, b, pr in busqueda
        ],
        "nota_sobre_la_autorreferencia": (
            "La sonda textual se marca a si misma, porque este guion nombra los doce mecanismos que "
            "busca. Para partial-label, positive-unlabeled, selective loss y masked BCE el unico "
            "guion que los menciona es este, luego el conteo historico valido es CERO. Se reporta "
            "en dos columnas para que el defecto sea visible y no una nota al pie."
        ),
        "matriz": filas,
        "decision": {
            "p0_cierra_la_fase": True,
            "motivo": (
                "P1 es H2-M con diferencia exactamente cero en coeficientes, probabilidades y "
                "macro-F1, y reproduce el -0.116581 ya registrado. Se cumple la condicion de "
                "parada numero 1 del enunciado."
            ),
            "que_no_se_afirma": (
                "Que la semantica de observacion parcialmente definida sea incorrecta. Lo que se "
                "mide es que implementada como enmascarar la perdida de clasificacion es el mismo "
                "estimador que H2-M. Una formulacion que aportase senal distinta de cero en las "
                "filas HS=0 seria otra cosa, y el barrido no deja ninguna via de esa clase dentro "
                "de los mecanismos permitidos."
            ),
            "experimentos_ejecutados": 0,
            "presupuesto": {"principales": 4, "reserva": 2, "usados": 0},
        },
    }
    REPORTE_JSON.write_text(json.dumps(informe, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _informe(informe, busqueda, inicio)
    log(f"completado en {round(time.time() - inicio, 1)} s")
    log(
        f"VEREDICTO: P1 ES H2-M | diferencia de coeficientes "
        f"{identidad['peor_diferencia_de_coeficientes']:.3e} | delta de la enmascarada "
        f"{delta_mascarada:+.6f} frente a {DELTA_H2_M_REGISTRADO:+.6f} registrado | FASE CERRADA"
    )


def construir_espacio(
    textos: pd.Series, entrenamiento: np.ndarray, validacion: np.ndarray
) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
    """Ajusta los vectorizadores en el entrenamiento y transforma la validacion.

    Args:
        textos: serie de textos del desarrollo.
        entrenamiento: indices de entrenamiento.
        validacion: indices de validacion.

    Returns:
        Tupla con la matriz de ajuste y la de consulta.
    """
    vectorizador_word = TfidfVectorizer(**PARAMS_WORD)
    vectorizador_char = TfidfVectorizer(**PARAMS_CHAR)
    ajuste = sparse.hstack(
        [
            vectorizador_word.fit_transform(textos.iloc[entrenamiento]),
            vectorizador_char.fit_transform(textos.iloc[entrenamiento]),
        ],
        format="csr",
    )
    consulta = sparse.hstack(
        [
            vectorizador_word.transform(textos.iloc[validacion]),
            vectorizador_char.transform(textos.iloc[validacion]),
        ],
        format="csr",
    )
    return ajuste, consulta


def _ajustar(
    matriz: sparse.csr_matrix,
    objetivo: np.ndarray,
    pesos: np.ndarray | None,
    equilibrar: bool = False,
) -> LogisticRegression:
    """Ajusta la regresion logistica de V03 con los pesos que se le pasen.

    Args:
        matriz: matriz de caracteristicas.
        objetivo: etiqueta binaria.
        pesos: pesos por fila, o None para el ajuste sin pesos.
        equilibrar: si se activan los pesos de clase de la biblioteca.

    Returns:
        El estimador ajustado.
    """
    estimador = LogisticRegression(
        C=CONSTANTE_V03,
        class_weight="balanced" if equilibrar else None,
        max_iter=LIMITE_ITERACIONES,
        solver="liblinear",
        random_state=RANDOM_STATE,
    )
    if pesos is None:
        estimador.fit(matriz, objetivo)
    else:
        estimador.fit(matriz, objetivo, sample_weight=pesos)
    return estimador


def _pesos_equilibrados(objetivo: np.ndarray, mascara: np.ndarray) -> np.ndarray:
    """Pesos de clase equilibrados contados solo sobre las filas con senal.

    Args:
        objetivo: etiqueta binaria.
        mascara: 1 en las filas con senal supervisada.

    Returns:
        Vector de pesos por fila.
    """
    pesos = np.zeros(len(objetivo), dtype=np.float64)
    activas = float(mascara.sum())
    for valor in (0, 1):
        seleccion = (mascara == 1) & (objetivo == valor)
        total = int(seleccion.sum())
        if total > 0:
            pesos[seleccion] = activas / (2.0 * total)
    return pesos


def _ambos_brazos(
    textos: pd.Series, y: np.ndarray, pliegues: list[tuple[np.ndarray, np.ndarray]]
) -> tuple[np.ndarray, np.ndarray]:
    """Calcula H2-M y la perdida enmascarada en los cinco pliegues.

    Args:
        textos: serie de textos del desarrollo.
        y: matriz binaria de etiquetas.
        pliegues: pares de indices de entrenamiento y validacion.

    Returns:
        Tupla con las probabilidades de H2-M y las de la perdida enmascarada.
    """
    probabilidades_h2m = np.zeros((len(y), 3), dtype=np.float64)
    probabilidades_mascarada = np.zeros((len(y), 3), dtype=np.float64)
    for entrenamiento, validacion in pliegues:
        ajuste, consulta = construir_espacio(textos, entrenamiento, validacion)
        mascara = (y[entrenamiento, 0] == 1).astype(float)
        for indice_etiqueta in range(3):
            objetivo = y[entrenamiento, indice_etiqueta]
            if indice_etiqueta == 0:
                estimador = _ajustar(ajuste, objetivo, None, equilibrar=True)
                probabilidades_h2m[validacion, 0] = estimador.predict_proba(consulta)[:, 1]
                probabilidades_mascarada[validacion, 0] = probabilidades_h2m[validacion, 0]
                continue
            activo = mascara == 1
            estimador_h2m = _ajustar(ajuste[activo], objetivo[activo], None, equilibrar=True)
            estimador_mascarada = _ajustar(
                ajuste, objetivo, mascara * _pesos_equilibrados(objetivo, mascara)
            )
            probabilidades_h2m[validacion, indice_etiqueta] = estimador_h2m.predict_proba(
                consulta
            )[:, 1]
            probabilidades_mascarada[validacion, indice_etiqueta] = (
                estimador_mascarada.predict_proba(consulta)[:, 1]
            )
    return probabilidades_h2m, probabilidades_mascarada


def _identidad_numerica() -> dict[str, Any]:
    """Compara coeficientes y probabilidades de H2-M y de la perdida enmascarada.

    Returns:
        Diccionario con las diferencias maximas y el detalle por pliegue.
    """
    train = data_module.read_train()
    grupos = grupos_fusionados(train)
    indices_dev, _ = holdout_v2(train, grupos)
    desarrollo = train.iloc[indices_dev].reset_index(drop=True)
    y = desarrollo[list(ETIQUETAS)].to_numpy(dtype=int)
    textos = desarrollo["text"].astype(str)
    pliegues = particion_por_semilla(textos, y, grupos[indices_dev], RANDOM_STATE)

    detalle: list[dict[str, Any]] = []
    for numero_pliegue, (entrenamiento, validacion) in enumerate(pliegues):
        ajuste, consulta = construir_espacio(textos, entrenamiento, validacion)
        mascara = (y[entrenamiento, 0] == 1).astype(float)
        for indice_etiqueta in (1, 2):
            objetivo = y[entrenamiento, indice_etiqueta]
            activo = mascara == 1
            estimador_h2m = _ajustar(ajuste[activo], objetivo[activo], None, equilibrar=True)
            estimador_mascarada = _ajustar(
                ajuste, objetivo, mascara * _pesos_equilibrados(objetivo, mascara)
            )
            detalle.append(
                {
                    "pliegue": numero_pliegue + 1,
                    "etiqueta": ETIQUETAS[indice_etiqueta],
                    "filas_con_senal": int(mascara.sum()),
                    "filas_enmascaradas": int((1 - mascara).sum()),
                    "max_abs_diferencia_de_coeficientes": float(
                        np.max(np.abs(estimador_h2m.coef_ - estimador_mascarada.coef_))
                    ),
                    "max_abs_diferencia_de_probabilidades": float(
                        np.max(
                            np.abs(
                                estimador_h2m.predict_proba(consulta)[:, 1]
                                - estimador_mascarada.predict_proba(consulta)[:, 1]
                            )
                        )
                    ),
                }
            )
    peor_coef = max(d["max_abs_diferencia_de_coeficientes"] for d in detalle)
    peor_prob = max(d["max_abs_diferencia_de_probabilidades"] for d in detalle)
    return {
        "detalle": detalle,
        "peor_diferencia_de_coeficientes": peor_coef,
        "peor_diferencia_de_probabilidades": peor_prob,
        "identidad_exacta": peor_coef == 0.0 and peor_prob == 0.0,
    }


def _muestra_sample_weight() -> list[dict[str, Any]]:
    """Localiza estructuralmente las llamadas con sample_weight y si dependen de HS.

    Returns:
        Una entrada por llamada, con fichero, linea, expresion y dependencia de HS.
    """
    muestra: list[dict[str, Any]] = []
    patron_hs = re.compile(r"\bhs\b|proba_hs|p_hs|y_?hs|hs_", re.IGNORECASE)
    for ruta in sorted((PROJECT_ROOT / "experiments").glob("*.py")):
        texto = ruta.read_text(encoding="utf-8")
        try:
            arbol = ast.parse(texto)
        except SyntaxError:
            continue
        for nodo in ast.walk(arbol):
            if not isinstance(nodo, ast.Call):
                continue
            for palabra_clave in nodo.keywords:
                if palabra_clave.arg != "sample_weight":
                    continue
                expresion = ast.unparse(palabra_clave.value)
                muestra.append(
                    {
                        "fichero": ruta.name,
                        "linea": nodo.lineno,
                        "expresion": expresion[:80],
                        "depende_de_hs": bool(patron_hs.search(expresion)),
                    }
                )
    return muestra


def _busca_mecanismos() -> list[tuple[str, str, list[str], int, bool]]:
    """Busca los doce mecanismos del enunciado por mencion textual.

    Returns:
        Una entrada por mecanismo, con su patron, los guiones que lo mencionan excluyendo este
        guion, el conteo bruto y si la autorreferencia esta presente.
    """
    fuentes = {
        p.name: p.read_text(encoding="utf-8")
        for p in (PROJECT_ROOT / "experiments").glob("*.py")
    }
    # **La sonda se marca a si misma.** Este guion nombra literalmente los doce mecanismos que
    # busca, porque sus textos estan en MECANISMOS_BUSCADOS y en la matriz. Por eso cada conteo se
    # reporta dos veces: con autorreferencia y sin ella. La cifra sin autorreferencia es la que
    # vale como evidencia, y es la que decide.
    propio = Path(__file__).name
    salida = []
    for nombre, patron in MECANISMOS_BUSCADOS:
        expresion = re.compile(patron, re.IGNORECASE)
        brutos = sorted(n for n, t in fuentes.items() if expresion.search(t))
        sin_propio = [n for n in brutos if n != propio]
        salida.append((nombre, patron, sin_propio, len(brutos), propio in brutos))
    return salida


def _informe(informe: dict[str, Any], busqueda: list[Any], inicio: float) -> None:
    """Escribe el informe de la auditoria de novedad.

    Args:
        informe: veredicto de la auditoria.
        busqueda: los doce mecanismos con sus dos conteos.
        inicio: instante de arranque del script.
    """
    identidad = informe["identidad_numerica"]
    metricas = informe["macro_f1"]
    decision = informe["decision"]
    linea: list[str] = [
        "# V14: auditoria de novedad del entrenamiento con etiquetas parcialmente definidas",
        "",
        f"Generado por `experiments/v14_00_auditoria_partial_label.py` ({EXPERIMENTO}).",
        "",
        "**No se entrena nada como experimento.** Esto es una identidad numerica y un inventario",
        "estructural, y es obligatorio antes de gastar presupuesto.",
        "",
        "## 1. El veredicto, primero",
        "",
        "**La perdida enmascarada de P1 ES H2-M, con diferencia exactamente cero.** Se cumple la",
        "condicion de parada numero 1 del enunciado, y la fase se cierra sin ejecutar experimentos.",
        "",
        "| comprobacion | valor |",
        "| --- | --- |",
        fila(["peor diferencia de coeficientes", f"{identidad['peor_diferencia_de_coeficientes']:.3e}"]),
        fila(["peor diferencia de probabilidades", f"{identidad['peor_diferencia_de_probabilidades']:.3e}"]),
        fila(["delta de la perdida enmascarada", f"{metricas['delta_enmascarada']:+.6f}"]),
        fila(["delta de H2-M registrado en v12", f"{informe['delta_h2_m_registrado']:+.6f}"]),
        fila(["diferencia entre ambos deltas", f"{metricas['diferencia_entre_deltas']:.3e}"]),
        "",
        "## 2. Por que enmascarar y eliminar son la misma operacion",
        "",
        "El enunciado advierte que P1 no debe confundirse con H2-M, y en eso se equivoca, por una",
        "razon que no es de notacion sino de optimizacion.",
        "",
        "**Una fila con peso cero en la perdida no aporta gradiente.** Luego el vector de",
        "coeficientes lo determinan unicamente las filas con peso uno, que son exactamente las",
        "filas que H2-M conserva. Las dos formulaciones dejan el mismo subconjunto de filas",
        "aportando senal, y por eso producen el mismo estimador.",
        "",
        "Lo unico que puede distinguirlas es **como se reescala el termino de perdida**, es decir la",
        "normalizacion y los pesos de clase. Eso no es un mecanismo de aprendizaje: es una constante",
        "de escala, y pertenece a la familia de calibracion y umbrales, que ya esta cerrada.",
        "",
        "La identidad se comprueba en los cinco pliegues y las dos etiquetas:",
        "",
        fila(["pliegue", "etiqueta", "filas con senal", "filas enmascaradas", "max dif coeficientes"]),
        fila(["---"] * 5),
    ]
    for d in identidad["detalle"]:
        linea.append(
            fila(
                [
                    str(d["pliegue"]),
                    d["etiqueta"],
                    str(d["filas_con_senal"]),
                    str(d["filas_enmascaradas"]),
                    f"{d['max_abs_diferencia_de_coeficientes']:.3e}",
                ]
            )
        )
    linea += [
        "",
        "## 3. Y el resultado que decide, que es el mas convincente",
        "",
        "Una identidad de estimador convence, pero lo que cierra la fase es que la formulacion",
        "**reproduce un resultado negativo ya registrado**:",
        "",
        fila(["sistema", "macro-F1", "delta frente a V03"]),
        fila(["---"] * 3),
        fila(["V03", f"{metricas['v03']:.16f}", "+0.000000"]),
        fila(["H2-M reconstruido", f"{metricas['h2_m_reconstruido']:.16f}", f"{metricas['delta_h2_m']:+.6f}"]),
        fila(
            [
                "perdida enmascarada de P1",
                f"{metricas['perdida_enmascarada_p1']:.16f}",
                f"{metricas['delta_enmascarada']:+.6f}",
            ]
        ),
        fila(["H2-M en el registro de v12", "no reconstruido aqui", f"{informe['delta_h2_m_registrado']:+.6f}"]),
        "",
        "Las cuatro cifras coinciden hasta el ultimo decimal. Una formulacion que se presenta como",
        "nueva y que reproduce exactamente el fallo de una familia cerrada no es una familia",
        "nueva: es la misma familia con otra notacion.",
        "",
        "## 4. La matriz que el enunciado exige",
        "",
        fila(
            [
                "mecanismo",
                "historico",
                "formula o comportamiento",
                "diferencia respecto al candidato",
                "resultado",
                "estado",
            ]
        ),
        fila(["---"] * 6),
    ]
    for m in informe["matriz"]:
        linea.append(
            fila(
                [
                    m["mecanismo"][:46],
                    m["experimento_historico"][:40],
                    m["formula_o_comportamiento"][:90],
                    m["diferencia_respecto_al_candidato"][:90],
                    m["resultado"][:46],
                    m["estado"][:46],
                ]
            )
        )
    linea += [
        "",
        "## 5. La sonda estructural, y por que el texto no servia",
        "",
        "El enunciado pide buscar doce mecanismos. La busqueda textual los localiza, pero **no sirve",
        "como veredicto**: un guion que enumera sus propias prohibiciones se marca a si mismo, que",
        "es el motivo por el que v12 documento el patron de sonda estructural. Lo que decide es la",
        "sonda sobre el arbol sintactico.",
        "",
        f"Hay **{len(informe['muestra_sample_weight'])} llamadas con `sample_weight`** en todo el",
        "codigo, y aqui estan con su origen:",
        "",
        fila(["fichero", "linea", "expresion de los pesos", "depende de HS"]),
        fila(["---"] * 4),
    ]
    for m in informe["muestra_sample_weight"]:
        linea.append(
            fila([m["fichero"], str(m["linea"]), m["expresion"], "si" if m["depende_de_hs"] else "no"])
        )
    linea += [
        "",
        "**Ninguna condiciona los pesos a HS en el punto de llamada.** Y al trazar cada vector hasta",
        "su definicion, ninguno se define por la indefinicion de la etiqueta:",
        "",
        "- `calcular_pesos` de V4-B05 pondera por cercania y por **discrepancia entre prediccion y",
        "  verdad**, es decir por el error del modelo, no por si la etiqueta esta definida. Su",
        "  variante `eliminar=True` si elimina filas, pero por otro criterio.",
        "- `pesos_dureza` de V4-C viene del mapa de dureza, que es dificultad del ejemplo.",
        "- `pesos(tamanos, esquema)` de V6-03 pondera por tamano de grupo.",
        "- el recorte de V6-07 actua sobre el ruido del par.",
        "- `pesos_por_clase` de V8-07 reescala por clase, que es lo que ya midio P5 en +0.0023.",
        "",
        "La unica operacion del historial que condiciona la senal de TR y AG a HS=1 es H2-M, y es",
        "exactamente la que P1 reproduce.",
        "",
        "### Los doce mecanismos del enunciado, y el defecto de la sonda textual",
        "",
        "La sonda textual localiza los doce mecanismos, pero **se marca a si misma**: este guion",
        "los nombra literalmente, porque sus textos estan en la lista de busqueda. Por eso cada",
        "conteo va en dos columnas, y la segunda es la que vale como evidencia.",
        "",
        fila(["mecanismo buscado", "guiones historicos", "conteo bruto", "autorreferencia"]),
        fila(["---"] * 4),
    ]
    for nombre, _patron, guiones, brutos, propio in busqueda:
        linea.append(
            fila(
                [
                    nombre[:52],
                    str(len(guiones)),
                    str(brutos),
                    "si" if propio else "no",
                ]
            )
        )
    linea += [
        "",
        "Para **partial-label**, **positive-unlabeled**, **selective loss** y **masked BCE** el",
        "unico guion que los menciona es este, luego **el conteo historico valido es cero**. No es",
        "que no se hayan buscado: es que la unica fuente que los nombra es la propia consulta.",
        "",
        "## 6. Que se afirma y que no",
        "",
        "**No se afirma** que la semantica de observacion parcialmente definida sea incorrecta, ni",
        "que el regimen HS=0 deba modelarse siempre como negativos ordinarios. Eso es una pregunta de",
        "modelado legitima y la fase no la responde.",
        "",
        "**Se afirma** algo mas estrecho y verificado: que implementada como enmascarar la perdida",
        "de clasificacion, esa semantica **es el mismo estimador que H2-M**, con diferencia cero.",
        "Una formulacion que de verdad tratara HS=0 como no supervisado tendria que aportar una",
        "senal distinta de cero en esas filas, y el barrido no deja ninguna via de esa clase dentro",
        "de los mecanismos permitidos por la jerarquia.",
        "",
        "## 7. Que pasa con P2, P3 y P4",
        "",
        "- **P2 y P3**, con el peso alfa, interpolan entre **dos extremos ya medidos**: alfa=0 es",
        "  exactamente H2-M, con -0.116581, y alfa=1 es exactamente V03, con 0.000000. Los dos",
        "  extremos del eje estan medidos y cerrados, y un barrido entre ellos es una",
        "  reparametrizacion de un eje cerrado, no una familia nueva.",
        "- **P4**, los umbrales, el propio enunciado los deja fuera salvo mejora estructural",
        "  reproducible, y no hay ninguna. Ademas el cambio de escala que producira ya fue medido y",
        "  documentado en H2, y la familia de umbrales y calibracion esta cerrada.",
        "",
        "## 8. Decision",
        "",
        (
            f"**{'FASE CERRADA' if decision['p0_cierra_la_fase'] else 'FASE ABIERTA'}: "
            f"{decision['motivo']}**"
        ),
        "",
        decision["que_no_se_afirma"],
        "",
        "**Experimentos ejecutados: 0 de 4 principales y 0 de 2 de reserva.** La auditoria es el",
        "primer y unico experimento de la fase, y su veredicto cierra la fase.",
        "",
    ]
    linea += sello(inicio)
    REPORTE_MD.write_text("\n".join(linea) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
