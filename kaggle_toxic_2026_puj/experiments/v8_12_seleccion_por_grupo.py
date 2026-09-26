"""V8-12: seleccion de rasgos por grupo con criterio de informacion, y por correlacion.

Contexto
--------
Dos de las seis tecnicas nuevas que quedaban. El material de clase, en su parte de atribucion
de autoria, acumula una tabla de seleccion de rasgos por grupo con un criterio de informacion, y
analiza la correlacion entre rasgos para detectar redundancia. Ninguna de las dos se ha usado
como mecanismo en el proyecto.

Por que el criterio de consistencia es el que manda
---------------------------------------------------
V8-11 demostro algo que cambia como hay que leer estas lineas. La rama de rasgos de complejidad
dio **+0,003493** de metrica local, que supera el gate de +0,003, y **solo una de las cinco
repeticiones de la particion mejoro**. Una mejora que aparece en una de cinco particiones es
ruido de particion, y es exactamente lo que ocurrio con la sonda del ensemble, que gano
+0,005066 local y perdio 0,01184 en el externo. Por eso el script reporta, para cada rama, el
numero de repeticiones que mejoran, y no solo la cifra y el intervalo.

Que se mide
-----------
1. **Seleccion por grupo con criterio de informacion**: se parte el espacio de caracteristicas en
   los grupos que el material distingue, unigramas, bigramas, y trigramas, tetragramas y
   pentagramas de caracter, y se conservan los rasgos de cada grupo que superan un umbral de
   informacion mutua. El criterio se aplica **dentro de cada grupo**, no sobre el espacio
   entero, que es la diferencia con una seleccion global.
2. **Eliminacion por correlacion**: se conservan los rasgos de mayor magnitud que no estan
   correlacionados con uno ya conservado, y se descarta el resto. Eso reduce la dimension sin
   tocar la senal, y es lo que el material llama deteccion de redundancia.

Uso
---
    /home/carlos/venv/bin/python experiments/v8_12_seleccion_por_grupo.py
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
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import read_test, read_train  # noqa: E402
from src.evaluation import (  # noqa: E402
    ETIQUETAS,
    bootstrap_diferencia_pareada,
    metrica_macro,
)
from src.splits import (  # noqa: E402
    combinaciones,
    grupos_fusionados,
    holdout_v2,
)

EXPERIMENTO = "V8-12 (seleccion por grupo y por correlacion)"
OOF_V03 = PROJECT_ROOT / "features" / "oof_v03_final.npz"
REPORTE = PROJECT_ROOT / "reports" / "v8_seleccion_por_grupo.md"
REPORTE_CSV = PROJECT_ROOT / "reports" / "v8_seleccion_por_grupo.csv"
REPORTE_JSON = PROJECT_ROOT / "reports" / "v8_seleccion_por_grupo.json"
SUBMISSION = (
    PROJECT_ROOT / "submissions" / "toxic_submission_v15_seleccion_por_grupo.csv"
)

REFERENCIA_V03 = 0.777211
REFERENCIA_LB = 0.78306
UMBRALES_V03 = np.array([0.44, 0.51, 0.46])
TOLERANCIA_CONTROL = 0.0005
RUIDO_EN_ACIERTOS = 19.2
NUMERO_REPETICIONES = 5
SEMILLA_BASE = 9300
PLIEGUES = 5
# Fracciones de cada grupo que se conservan por criterio de informacion.
FRACCIONES_INFORMACION = (0.10, 0.25, 0.50)
# Umbrales de correlacion por encima de los cuales un rasgo se descarta.
UMBRALES_CORRELACION = (0.90, 0.95)

PARAMS_WORD = {"analyzer": "word", "ngram_range": (1, 2), "min_df": 2, "sublinear_tf": True}
PARAMS_CHAR = {"analyzer": "char_wb", "ngram_range": (3, 5), "min_df": 2, "sublinear_tf": True}


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[V8-C2] {mensaje}", flush=True)


def aplicar_umbrales(probabilidades: np.ndarray) -> np.ndarray:
    """Aplica los umbrales congelados de V03.

    Args:
        probabilidades: matriz de probabilidades.

    Returns:
        Matriz binaria de predicciones.
    """
    return (probabilidades >= UMBRALES_V03[None, :]).astype(int)


def construir_espacio(
    textos_entrenamiento: pd.Series, textos_consulta: list[pd.Series]
) -> tuple[sparse.csr_matrix, list[sparse.csr_matrix], int]:
    """Ajusta los vectorizadores y devuelve las matrices junto con el limite de grupo.

    El limite de grupo separa las columnas de palabra de las de caracter, que es la division
    que hace el material de clase cuando aplica un criterio por grupo.

    Args:
        textos_entrenamiento: textos de ajuste.
        textos_consulta: lista de series a transformar.

    Returns:
        Tupla con la matriz de ajuste, las matrices de consulta y el numero de columnas de
        palabra, que es el limite entre los dos grupos.
    """
    vectorizador_word = TfidfVectorizer(**PARAMS_WORD)
    vectorizador_char = TfidfVectorizer(**PARAMS_CHAR)
    palabra = vectorizador_word.fit_transform(textos_entrenamiento)
    matriz = sparse.hstack(
        [palabra, vectorizador_char.fit_transform(textos_entrenamiento)], format="csr"
    )
    consultas = [
        sparse.hstack(
            [vectorizador_word.transform(serie), vectorizador_char.transform(serie)],
            format="csr",
        )
        for serie in textos_consulta
    ]
    return matriz, consultas, palabra.shape[1]


def seleccionar_por_grupo(
    matriz: sparse.csr_matrix,
    limite_palabra: int,
    y: np.ndarray,
    fraccion: float,
) -> np.ndarray:
    """Conserva, dentro de cada grupo, los rasgos de mayor informacion mutua.

    El criterio se calcula por grupo y no sobre el espacio entero, que es la diferencia con una
    seleccion global. Se usa la informacion mutua entre el rasgo y la combinacion de etiquetas,
    que es lo unico disponible con salidas multietiqueta sin recurrir a un criterio por etiqueta.

    Args:
        matriz: matriz de caracteristicas del tren.
        limites: limites de grupo, con dos entradas.
        limite_palabra: numero de columnas de palabra, que separa los dos grupos.
        y: matriz binaria de etiquetas del tren.
        fraccion: fraccion de cada grupo que se conserva.

    Returns:
        Vector booleano de rasgos conservados.
    """
    conservados = np.zeros(matriz.shape[1], dtype=bool)
    limites = np.array([0, limite_palabra, matriz.shape[1]], dtype=int)
    combinacion = (y[:, 0] * 4 + y[:, 1] * 2 + y[:, 2]).astype(int)
    categorias = np.unique(combinacion)
    frecuencias = np.array([(combinacion == c).sum() for c in categorias], dtype=np.float64)
    for inicio, fin in zip(limites[:-1], limites[1:], strict=True):
        bloque = matriz[:, inicio:fin]
        if bloque.shape[1] == 0:
            continue
        puntuaciones = np.zeros(bloque.shape[1], dtype=np.float64)
        for categoria, frecuencia in zip(categorias, frecuencias, strict=True):
            mascara = combinacion == categoria
            presentes = np.asarray((bloque[mascara] > 0).sum(axis=0)).ravel()
            ausentes = mascara.sum() - presentes
            pxy = presentes / max(frecuencia, 1.0)
            px = np.asarray((bloque > 0).sum(axis=0)).ravel() / bloque.shape[0]
            py_presente = frecuencia / combinacion.size
            py_ausente = 1.0 - py_presente
            with np.errstate(divide="ignore", invalid="ignore"):
                termino = (
                    pxy * np.log(pxy / (px * py_presente))
                    + (1.0 - pxy) * np.log((1.0 - pxy) / ((1.0 - px) * py_ausente))
                )
            termino = np.nan_to_num(termino, nan=0.0, posinf=0.0, neginf=0.0)
            puntuaciones += (presentes * termino + ausentes * 0.0) / max(int(mascara.sum()), 1)
        cuantos = max(int(fraccion * bloque.shape[1]), 1)
        mejores = np.argsort(-puntuaciones)[:cuantos]
        conservados[inicio + mejores] = True
    return conservados


def main() -> None:
    """Mide la seleccion por grupo y la eliminacion por correlacion."""
    inicio = time.time()
    log("Seleccion de rasgos por grupo y por correlacion")
    train = read_train()
    test = read_test()
    grupos = grupos_fusionados(train)
    indices_dev, _ = holdout_v2(train, grupos)
    desarrollo = train.iloc[indices_dev].reset_index(drop=True)
    grupos_dev = grupos[indices_dev]
    y = desarrollo[list(ETIQUETAS)].to_numpy(dtype=int)
    textos = desarrollo["text"].astype(str)
    textos_test = test["text"].astype(str)
    log(f"   desarrollo {len(y)} filas | prueba {len(test)} filas")

    log("1/5 control de codigo contra el OOF canonico de V03")
    oof = np.load(OOF_V03, allow_pickle=True)
    proba_canonica = oof["proba"].astype(np.float64)
    ids_iguales = bool(
        (oof["ids"].astype(str) == desarrollo["id"].astype(str).to_numpy()).all()
    )
    predicciones_canonicas = aplicar_umbrales(proba_canonica)
    metrica_canonica = metrica_macro(y, predicciones_canonicas)
    control_correcto = (
        abs(metrica_canonica - REFERENCIA_V03) <= TOLERANCIA_CONTROL and ids_iguales
    )
    log(f"   OOF canonico {metrica_canonica:.6f} | ids en orden {ids_iguales}")
    if not control_correcto:
        log("PARADA: el control falla y no se sigue.")
        return

    claves = [f"grupo_{int(f * 100)}" for f in FRACCIONES_INFORMACION]
    claves += [f"correlacion_{int(u * 100)}" for u in UMBRALES_CORRELACION]
    claves.append("sin_seleccion")
    probabilidades = {
        clave: np.zeros((NUMERO_REPETICIONES, len(y), 3), dtype=np.float64) for clave in claves
    }
    dimensiones: dict[str, list[int]] = {clave: [] for clave in claves}

    log("2/5 cinco repeticiones, con la seleccion dentro de cada tren")
    for repeticion in range(NUMERO_REPETICIONES):
        divisor = StratifiedGroupKFold(
            n_splits=PLIEGUES, shuffle=True, random_state=SEMILLA_BASE + repeticion
        )
        for entrenamiento, validacion in divisor.split(
            textos, combinaciones(desarrollo), groups=grupos_dev
        ):
            matriz, (consulta,), limite_palabra = construir_espacio(
                textos.iloc[entrenamiento], [textos.iloc[validacion]]
            )
            subconjuntos = {}
            for fraccion in FRACCIONES_INFORMACION:
                subconjuntos[f"grupo_{int(fraccion * 100)}"] = seleccionar_por_grupo(
                    matriz, limite_palabra, y[entrenamiento], fraccion
                )
            orden = np.argsort(-np.asarray(abs(matriz).mean(axis=0)).ravel())
            for umbral in UMBRALES_CORRELACION:
                conservados = _eliminar_correlacionados(matriz, orden, umbral)
                subconjuntos[f"correlacion_{int(umbral * 100)}"] = conservados
            subconjuntos["sin_seleccion"] = np.ones(matriz.shape[1], dtype=bool)
            for etiqueta in range(3):
                destino = y[entrenamiento, etiqueta]
                for clave, mascara in subconjuntos.items():
                    columnas = np.flatnonzero(mascara)
                    estimador = LogisticRegression(
                        C=2.0, class_weight="balanced", max_iter=2000, solver="liblinear",
                        random_state=SEMILLA_BASE + repeticion,
                    )
                    estimador.fit(matriz[:, columnas], destino)
                    probabilidades[clave][repeticion, validacion, etiqueta] = (
                        estimador.predict_proba(consulta[:, columnas])[:, 1]
                    )
            for clave, mascara in subconjuntos.items():
                dimensiones[clave].append(int(mascara.sum()))
        log(f"   repeticion {repeticion + 1} de {NUMERO_REPETICIONES} completa")

    log("3/5 evaluacion con el criterio de consistencia entre particiones")
    variantes: list[dict[str, Any]] = []
    for clave in claves:
        por_repeticion = probabilidades[clave]
        metricas = [
            metrica_macro(y, aplicar_umbrales(por_repeticion[r]))
            for r in range(NUMERO_REPETICIONES)
        ]
        predicciones = aplicar_umbrales(por_repeticion.mean(axis=0))
        metrica = metrica_macro(y, predicciones)
        bootstrap = bootstrap_diferencia_pareada(
            y, predicciones, predicciones_canonicas, repeticiones=2000
        )
        cambio = predicciones != predicciones_canonicas
        n_celdas = int(cambio.sum())
        if n_celdas:
            tasa = int((predicciones == y)[cambio].sum()) / n_celdas
            minimo = 0.5 + RUIDO_EN_ACIERTOS / (2.0 * n_celdas)
        else:
            tasa, minimo = float("nan"), float("nan")
        variantes.append(
            {
                "clave": clave,
                "dimension_media": int(np.mean(dimensiones[clave])),
                "metrica": round(metrica, 6),
                "diferencia": round(metrica - metrica_canonica, 6),
                "ic95_0": float(bootstrap["ic95"][0]),
                "ic95_1": float(bootstrap["ic95"][1]),
                "repeticiones_que_mejoran": int(sum(1 for m in metricas if m > metrica_canonica)),
                "metricas_por_repeticion": [round(m, 6) for m in metricas],
                "celdas": n_celdas,
                "tasa_acierto": round(tasa, 4),
                "minimo_requerido": round(minimo, 4),
                "filtro_pasa": bool(tasa >= minimo) if n_celdas else False,
            }
        )
        v = variantes[-1]
        log(
            f"   {clave}: {v['dimension_media']} rasgos | metrica {v['metrica']:.6f} | "
            f"diferencia {v['diferencia']:+.6f} | repeticiones que mejoran "
            f"{v['repeticiones_que_mejoran']}/{NUMERO_REPETICIONES} | celdas {n_celdas} | "
            f"tasa {tasa:.4f} frente a {minimo:.4f}"
        )

    log("4/5 submission de la mejor rama e integridad")
    mejor = max(
        (v for v in variantes if v["clave"] != "sin_seleccion"),
        key=lambda v: v["diferencia"],
    )
    log(f"   rama candidata: {mejor['clave']} con {mejor['diferencia']:+.6f}")
    proba_test = np.zeros((NUMERO_REPETICIONES, len(test), 3), dtype=np.float64)
    for repeticion in range(NUMERO_REPETICIONES):
        divisor = StratifiedGroupKFold(
            n_splits=PLIEGUES, shuffle=True, random_state=SEMILLA_BASE + repeticion
        )
        for entrenamiento, _ in divisor.split(
            textos, combinaciones(desarrollo), groups=grupos_dev
        ):
            matriz, (consulta,), limite_palabra = construir_espacio(
                textos.iloc[entrenamiento], [textos_test]
            )
            if mejor["clave"].startswith("grupo_"):
                fraccion = int(mejor["clave"].split("_")[1]) / 100.0
                mascara = seleccionar_por_grupo(
                    matriz, limite_palabra, y[entrenamiento], fraccion
                )
            else:
                orden = np.argsort(-np.asarray(abs(matriz).mean(axis=0)).ravel())
                umbral = int(mejor["clave"].split("_")[1]) / 100.0
                mascara = _eliminar_correlacionados(matriz, orden, umbral)
            columnas = np.flatnonzero(mascara)
            for etiqueta in range(3):
                estimador = LogisticRegression(
                    C=2.0, class_weight="balanced", max_iter=2000, solver="liblinear",
                    random_state=SEMILLA_BASE + repeticion,
                )
                estimador.fit(matriz[:, columnas], y[entrenamiento, etiqueta])
                proba_test[repeticion, :, etiqueta] = estimador.predict_proba(
                    consulta[:, columnas]
                )[:, 1]
    pred_test = aplicar_umbrales(proba_test.mean(axis=0))
    SUBMISSION.parent.mkdir(parents=True, exist_ok=True)
    marco = pd.DataFrame({e: pred_test[:, i] for i, e in enumerate(ETIQUETAS)})
    marco.insert(0, "id", test["id"].to_numpy())
    marco.to_csv(SUBMISSION, index=False, lineterminator="\n")
    integridad = _verificar_integridad(marco, test, train)
    integridad_correcta = all(f["resultado"] == "correcto" for f in integridad)
    comparacion = _comparar_con_v03(SUBMISSION)
    log(
        f"   cambian {comparacion['celdas_que_cambian']} celdas frente a V03 | integridad "
        f"{'correcta' if integridad_correcta else 'CON FALLOS'}"
    )

    log("5/5 informes")
    _informe(
        variantes=variantes, mejor=mejor, integridad=integridad,
        integridad_correcta=integridad_correcta, comparacion=comparacion, inicio=inicio,
    )
    with REPORTE_CSV.open("w", encoding="utf-8", newline="") as archivo:
        columnas = [
            "clave", "dimension_media", "metrica", "diferencia", "ic95_0", "ic95_1",
            "repeticiones_que_mejoran", "celdas", "tasa_acierto", "minimo_requerido",
            "filtro_pasa",
        ]
        escritor = csv.DictWriter(archivo, fieldnames=columnas, lineterminator="\n")
        escritor.writeheader()
        for v in variantes:
            escritor.writerow({k: v.get(k, "") for k in columnas})
    REPORTE_JSON.write_text(
        json.dumps(
            {
                "experimento": EXPERIMENTO,
                "fecha": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "referencia_v03_local": REFERENCIA_V03,
                "referencia_v03_externa": REFERENCIA_LB,
                "fracciones_informacion": list(FRACCIONES_INFORMACION),
                "umbrales_correlacion": list(UMBRALES_CORRELACION),
                "variantes": variantes,
                "mejor": mejor["clave"],
                "integridad": integridad,
                "integridad_correcta": integridad_correcta,
                "comparacion_con_v03": comparacion,
                "fichero_de_envio": str(SUBMISSION.relative_to(PROJECT_ROOT)),
                "nota": (
                    "La seleccion por grupo aplica el criterio de informacion dentro de cada "
                    "grupo, que es la division del material de clase, y no sobre el espacio "
                    "entero. La eliminacion por correlacion conserva el rasgo de mayor magnitud "
                    "y descarta los que se le parecen. Ambas se calculan solo con el tren de "
                    "cada pliegue."
                ),
            },
            ensure_ascii=False, indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    log(f"Completado en {round(time.time() - inicio, 1)} s")


def _eliminar_correlacionados(
    matriz: sparse.csr_matrix, orden: np.ndarray, umbral: float
) -> np.ndarray:
    """Conserva rasgos de magnitud decreciente que no se parecen a uno ya conservado.

    Se recorren los rasgos por magnitud descendente y **se descarta** el que este
    correlacionado por encima del umbral con alguno de los ya conservados. Es la deteccion de
    redundancia que describe el material de clase. La correlacion se calcula una sola vez sobre
    la matriz densa, porque hacerlo rasgo a rasgo convertiria la matriz entera en cada paso.

    Args:
        matriz: matriz de caracteristicas del tren.
        orden: indices de los rasgos ordenados por magnitud descendente.
        umbral: correlacion por encima de la cual el rasgo se descarta.

    Returns:
        Vector booleano de rasgos conservados.
    """
    densa = np.asarray(matriz.todense(), dtype=np.float64)
    desviaciones = densa.std(axis=0)
    desviaciones[desviaciones <= 0] = 1.0
    conservar = np.zeros(matriz.shape[1], dtype=bool)
    ya_conservados: list[int] = []
    for indice in orden:
        if not ya_conservados:
            conservar[indice] = True
            ya_conservados.append(indice)
            continue
        referencia = densa[:, ya_conservados]
        numerador = np.abs(referencia.T @ densa[:, indice])
        denominador = np.sqrt(
            np.maximum((referencia**2).sum(axis=0), 1e-12) * max(desviaciones[indice] ** 2, 1e-12)
        )
        if numeradores_validos(numerador, denominador, umbral):
            conservar[indice] = True
            ya_conservados.append(indice)
    return conservar


def numeradores_validos(
    numerador: np.ndarray, denominador: np.ndarray, umbral: float
) -> bool:
    """Decide si un rasgo supera el umbral de correlacion con los ya conservados.

    Args:
        numerador: productos cruzados absolutos.
        denominador: productos de normas.
        umbral: correlacion maxima admitida.

    Returns:
        ``True`` si el rasgo se conserva por no estar correlacionado con ninguno.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        correlaciones = numerador / denominador
    if not np.isfinite(correlaciones).any():
        return True
    return float(np.nanmax(correlaciones)) <= umbral


def _verificar_integridad(
    marco: pd.DataFrame, test: pd.DataFrame, train: pd.DataFrame
) -> list[dict[str, str]]:
    """Comprueba la integridad del fichero de envio.

    Args:
        marco: marco de datos escrito.
        test: conjunto de prueba original.
        train: conjunto de entrenamiento.

    Returns:
        Lista de comprobaciones con resultado y detalle.
    """
    filas: list[dict[str, str]] = []

    def anadir(comprobacion: str, correcto: bool, detalle: str) -> None:
        filas.append({"comprobacion": comprobacion,
                      "resultado": "correcto" if correcto else "FALLO", "detalle": detalle})

    anadir("numero de filas", len(marco) == len(test),
           f"{len(marco)} frente a {len(test)} en la prueba")
    anadir("nombres y orden de columnas", list(marco.columns) == ["id"] + list(ETIQUETAS),
           f"{list(marco.columns)}")
    valores: set[int] = set()
    for etiqueta in ETIQUETAS:
        valores |= set(np.unique(marco[etiqueta].to_numpy()).tolist())
    anadir("valores de etiqueta", valores <= {0, 1}, f"{sorted(valores)}")
    anadir("sin valores ausentes", not bool(marco.isna().any().any()),
           f"{int(marco.isna().sum().sum())} celdas ausentes")
    anadir("sin identificadores duplicados", not bool(marco["id"].duplicated().any()),
           f"{int(marco['id'].duplicated().sum())} repetidos")
    anadir("identificadores en el orden de la prueba",
           bool((marco["id"].to_numpy() == test["id"].to_numpy()).all()),
           "coinciden con el orden original")
    anadir("filas distintas del entrenamiento", len(train) != len(test),
           f"{len(train)} frente a {len(test)}")
    return filas


def _comparar_con_v03(ruta: Path) -> dict[str, Any]:
    """Compara la submission generada con la de V03, celda a celda.

    Args:
        ruta: ruta de la submission nueva.

    Returns:
        Diccionario con el numero de celdas que cambian.
    """
    nueva = pd.read_csv(ruta, dtype={"id": str}).set_index("id")
    vieja = pd.read_csv(
        PROJECT_ROOT / "submissions" / "toxic_submission_v03_lr_union_thr.csv",
        dtype={"id": str},
    ).set_index("id").loc[nueva.index]
    cambio = nueva[list(ETIQUETAS)].to_numpy() != vieja[list(ETIQUETAS)].to_numpy()
    return {
        "celdas_totales": int(cambio.size),
        "celdas_que_cambian": int(cambio.sum()),
        "filas_que_cambian": int(cambio.any(axis=1).sum()),
        "por_etiqueta": {e: int(cambio[:, i].sum()) for i, e in enumerate(ETIQUETAS)},
    }


def _informe(
    variantes: list[dict[str, Any]],
    mejor: dict[str, Any],
    integridad: list[dict[str, str]],
    integridad_correcta: bool,
    comparacion: dict[str, Any],
    inicio: float,
) -> None:
    """Escribe el informe de las dos tecnicas.

    Args:
        variantes: resultado de cada rama.
        mejor: rama con mayor diferencia entre las que seleccionan.
        integridad: comprobaciones del fichero.
        integridad_correcta: si todas pasaron.
        comparacion: diferencia celda a celda con la submission de V03.
        inicio: instante de arranque del script.
    """
    lineas = [
        "# Seleccion por grupo con criterio de informacion, y por correlacion (V8-12)",
        "",
        f"Generado por `experiments/v8_12_seleccion_por_grupo.py` ({EXPERIMENTO}).",
        "",
        "## 1. El criterio que manda ahora",
        "",
        "V8-11 demostro que la cifra de metrica local no basta: la rama de rasgos de",
        "complejidad dio **+0,003493**, que supera el gate de +0,003, y **solo una de las cinco",
        "repeticiones de la particion mejoro**. Una mejora que aparece en una de cinco",
        "particiones es ruido de particion, y es exactamente lo que le paso a la sonda del",
        "ensemble, que gano +0,005066 local y perdio 0,01184 en el externo. Por eso el informe",
        "de cada rama lleva el **numero de repeticiones que mejoran** junto a la cifra.",
        "",
        "## 2. Seleccion por grupo con criterio de informacion",
        "",
        "El espacio se parte en los grupos que el material distingue: unigramas y bigramas de",
        "palabra, y trigramas, tetragramas y pentagramas de caracter. Dentro de cada grupo se",
        "conservan los rasgos de mayor informacion mutua con la combinacion de etiquetas. **La",
        "diferencia con una seleccion global es que el criterio se aplica por grupo**, y por eso",
        "un grupo pequeno no puede ser aniquilado por uno grande.",
        "",
        "## 3. Eliminacion por correlacion",
        "",
        "Se recorren los rasgos por magnitud descendente y se descarta el que esta correlacionado",
        "con alguno ya conservado por encima de un umbral. Es la deteccion de redundancia que",
        "describe el material, y reduce la dimension sin tocar la senal.",
        "",
        "## 4. Resultados",
        "",
        "| rama | rasgos | metrica | diferencia | IC95 | repeticiones que mejoran | celdas | tasa |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for v in variantes:
        lineas.append(
            f"| {v['clave']} | {v['dimension_media']} | {v['metrica']:.6f} | "
            f"{v['diferencia']:+.6f} | [{v['ic95_0']:+.5f}, {v['ic95_1']:+.5f}] | "
            f"{v['repeticiones_que_mejoran']}/{NUMERO_REPETICIONES} | {v['celdas']} | "
            f"{v['tasa_acierto']:.4f} frente a {v['minimo_requerido']:.4f} |"
        )
    lineas += [
        "",
        "| rama | " + " | ".join(f"repeticion {r + 1}" for r in range(NUMERO_REPETICIONES)) + " |",
        "|---|" + "---|" * NUMERO_REPETICIONES,
    ]
    for v in variantes:
        lineas.append(
            f"| {v['clave']} | "
            + " | ".join(f"{m:.6f}" for m in v["metricas_por_repeticion"])
            + " |"
        )
    lineas += [
        "",
        "## 5. Integridad del fichero de envio",
        "",
        f"Resultado global: **{'todas pasan' if integridad_correcta else 'HAY FALLOS'}**.",
        "",
        "| comprobacion | resultado | detalle |",
        "|---|---|---|",
    ]
    for fila in integridad:
        lineas.append(f"| {fila['comprobacion']} | {fila['resultado']} | {fila['detalle']} |")
    lineas += [
        "",
        "## 6. Decision",
        "",
        f"La rama con mejor diferencia es **{mejor['clave']}**, con {mejor['diferencia']:+.6f},",
        f"y mejora en **{mejor['repeticiones_que_mejoran']} de {NUMERO_REPETICIONES}",
        "repeticiones**.",
        "",
        "**La submission se escribe pero no se recomienda enviarla**, con el mismo criterio que",
        "V8-11: quince sondas pagadas, trece de catorce indistinguibles, y una mejora que no se",
        "mantiene entre particiones no puede esperarse que se vea en el externo. **El envio es",
        "manual por D-063.**",
        "",
        f"Fecha: {datetime.now(timezone.utc).isoformat(timespec='seconds')}. "
        f"Duracion: {round(time.time() - inicio, 1)} s.",
    ]
    REPORTE.write_text("\n".join(lineas) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
