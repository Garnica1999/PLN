"""V18-01: conjunto de cambio de cada challenger, y si algun feature acepta unos cambios y no otros.

La pregunta de la fase
----------------------
> Entre las filas que el challenger quiere cambiar, algun feature permite distinguir de forma
> consistente los rescates de los danos?

Que se mide, y por que el objetivo no es la etiqueta
---------------------------------------------------
La distincion es lo que hace nueva a esta fase. El objetivo **no** es la etiqueta: es si el cambio
que propone el challenger es correcto o no frente a V03. Una feature puede predecir la etiqueta
muy bien y aun asi no decir nada sobre si conviene aceptar el cambio, porque dentro del conjunto
de cambio las dos clases son precisamente los desacuerdos entre dos modelos, y lo que decide es
cual de los dos se equivoco, no cual era la etiqueta.

Por eso el objetivo de cada feature es la etiqueta binaria

    1 = el cambio es un rescate, V03 falla y el challenger acierta
    0 = el cambio es un dano, V03 acierta y el challenger falla

y no ``y_true``. Es un objetivo distinto del que midio V15, y por eso este script no repite
aquel trabajo: mide otra cosa sobre las mismas celdas.

Un analisis por challenger, no uno agregado
-------------------------------------------
Cada challenger propone un conjunto de cambio distinto, con celdas distintas, y su veredicto tiene
que sostenerse en su propio conjunto. Agregarlos en una rejilla comun exigiria que coincidieran las
celdas, que es justo lo que no pasa, y una rejilla comun solo artificial restringiria el analisis al
nucleo donde todos coinciden, que es donde la decision es mas facil y menos informativa. Por eso
cada challenger se mide por separado y el test se repite cinco veces.

Como se controla el multiplicidad
---------------------------------
Se prueban mas de cien features y cada una en varios regimenes, luego con alpha 0.05 aparecerian
decenas de falsos positivos. Por eso el estadistico es el **maximo** sobre todos los regimenes
probados, y su nula se obtiene permutando las etiquetas de rescate y dano dentro del conjunto de
cambio. Ese test es exacto bajo la nula global: si el maximo observado no se alcanza en ninguna
permutacion, ningun regimeno de ninguna feature separa los rescates de los danos, y no hace falta
corregir a mano por el numero de pruebas.

El contraste que decide
-----------------------
Un feature no es interesante por mover el AUC. Es interesante por lo que improve la **decision
operativa**: la regla "acepta el cambio si el feature supera el corte" tiene que mejorar el
rescate neto frente a aceptar todos los cambios. Por eso se mide el neto de la regla, y el neto de
aceptar todo se reporta como la referencia contra la que compite.

Y el corte se elige con disciplina de pliegues desde el principio, no como comprobacion posterior:
la regla se elige en cuatro pliegues y se evalua en el quinto, con los cinco rotados. Sin eso, el
numero que sale seria el maximo sobre el mismo conjunto que despues se evalua, y una regla elegida
a mano sobre doscientas celdas siempre parece buena.

Que NO hace este script
----------------------
No entrena un meta-modelo, no construye la regla final, no gasta ninguna submission y no toca el
champion. Si el test global sale negativo en los cinco challengers, la fase se cierra sin haber
gastado nada.

Uso
---
    /home/carlos/venv/bin/python experiments/v18_01_cambio_y_senal.py
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

EXPERIMENTO = "V18-01 (P1 conjunto de cambio y P2 senal de aceptacion)"
REPORTE_MD = PROJECT_ROOT / "reports" / "v18_01_cambio_y_senal.md"
REPORTE_JSON = PROJECT_ROOT / "reports" / "v18_01_cambio_y_senal.json"
CSV_CAMBIOS = PROJECT_ROOT / "resultados" / "v18_01_tabla_de_cambios.csv"
CSV_SENAL = PROJECT_ROOT / "resultados" / "v18_02_senal_por_feature.csv"

ETIQUETAS = ("HS", "TR", "AG")
REFERENCIA_V03 = 0.7772114525165721
LB_V03 = 0.78306
CELDAS_POR_003_LB = 107
# Una regla tiene que aportar esta fraccion de la escala de 107 celdas para que la mejora que
# produce pueda llegar a mover el leaderboard. Es un liston bajo a proposito: si la regla no mueve
# ni una cuarta parte de lo que hace falta para 0.003, no es una candidata competitiva.
FRACCION_DE_ESCALA_NECESARIA = 0.25
PERMUTACIONES = 1000
SEMILLA = 20260926
CORTES_CUANTILES = (0.25, 0.50, 0.75)
# Un regimen debe tener al menos este numero de celdas para que su proporcion tenga precision
# suficiente. Con menos de 30, el error tipico de una proporcion es de mas de 0.08 y cualquier
# lift pequeno seria indistinguible del ruido.
MINIMO_CELDAS_POR_REGIMEN = 30

CANDIDATOS: tuple[tuple[str, str, tuple[float, float, float]], ...] = (
    ("oof_candidates_v4", "C_meta36_downweight", (0.47, 0.44, 0.50)),
    ("oof_ens04_v04", "proba_ens", (0.47, 0.49, 0.52)),
    ("oof_meta_shift_v4", "meta_control", (0.46, 0.49, 0.48)),
    ("oof_mod03b", "proba", (0.47, 0.50, 0.50)),
    ("oof_candidates_v4", "B_meta36", (0.47, 0.44, 0.50)),
)


def log(mensaje: str) -> None:
    """Imprime una linea de progreso.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[V18-01] {mensaje}", flush=True)


def main() -> None:
    """Construye los conjuntos de cambio y mide la senal de aceptacion de cada uno."""
    inicio = time.time()
    log("V18-01: conjunto de cambio y senal de aceptacion")

    log("1/6 cargar V03, las features auxiliares y los challengers")
    datos_v03 = np.load(PROJECT_ROOT / "features" / "oof_v03_final.npz", allow_pickle=True)
    ids = np.asarray(datos_v03["ids"]).astype(np.int64)
    y = np.asarray(datos_v03["y_true"], dtype=int)
    proba_v03 = np.asarray(datos_v03["proba"], dtype=float)
    margen_v03 = np.asarray(datos_v03["margen"], dtype=float)
    confianza_v03 = np.asarray(datos_v03["confianza"], dtype=float)
    entropia_v03 = np.asarray(datos_v03["entropia"], dtype=float)
    pliegues = np.asarray(datos_v03["pliegues"], dtype=int)
    umbrales_v03 = np.asarray(datos_v03["umbrales"], dtype=float)
    pred_v03 = (proba_v03 >= umbrales_v03[None, :]).astype(int)
    error_v03 = pred_v03 != y

    # **Los identificadores tienen que estar en el mismo tipo antes de comparar.** La tabla
    # estructurada los guarda como cadenas y el OOF de V03 los guarda como enteros, de modo que sin
    # esta normalizacion no coincide ninguno, la cobertura es de cero filas, y la fase entera se
    # ejecuta sin ninguna feature auxiliar mientras reporta un veredicto sobre si las features
    # auxiliares tienen senal. Es una degradacion silenciosa: no falla, solo mide otra cosa.
    estructura = pd.read_parquet(
        PROJECT_ROOT / "features" / "v2_structured_features.parquet"
    ).drop_duplicates("id")
    estructura["id"] = estructura["id"].astype(np.int64)
    estructura = estructura.set_index("id")
    cubiertos = int(np.isin(ids, estructura.index.to_numpy()).sum())
    log(
        f"   {len(estructura.columns) - 1} features auxiliares disponibles, "
        f"{cubiertos} de {len(ids)} filas cubiertas"
    )
    if cubiertos < len(ids):
        raise SystemExit(
            f"PARADA OBLIGATORIA: las featuresauxiliares cubren {cubiertos} de {len(ids)} filas. "
            "Medir la senal de aceptacion sin ellas no responde a la pregunta de la fase, que es "
            "justamente si la informacionauxiliar discrimina el cambio correcto del incorrecto"
        )
    log(f"   errores de V03: {int(error_v03.sum())} de {y.size} celdas")

    log("2/6 construir el conjunto de cambio de cada challenger (P1)")
    challengers: list[dict[str, Any]] = []
    for artefacto, variante, umbrales in CANDIDATOS:
        proba = _proba_del_challenger(artefacto, variante, ids, y)
        if proba is None:
            log(f"   {artefacto}/{variante}: no se pudo alinear, se omite")
            continue
        pred = (proba >= np.asarray(umbrales)[None, :]).astype(int)
        cambia = pred != pred_v03
        error_challenger = pred != y
        rescate = cambia & error_v03 & ~error_challenger
        dano = cambia & ~error_v03 & error_challenger
        n_cambio = int(cambia.sum())
        n_rescate = int(rescate.sum())
        n_dano = int(dano.sum())
        if n_cambio < 2 * MINIMO_CELDAS_POR_REGIMEN:
            log(f"   {artefacto}/{variante}: solo {n_cambio} celdas, demasiado pocas")
            continue
        filas, cols = np.nonzero(cambia)
        challengers.append(
            {
                "artefacto": artefacto,
                "variante": variante,
                "umbrales": umbrales,
                "proba": proba,
                "pred": pred,
                "filas": filas,
                "cols": cols,
                "objetivo": np.where(rescate[filas, cols], 1, 0).astype(np.int8),
                "n_cambio": n_cambio,
                "n_rescate": n_rescate,
                "n_dano": n_dano,
                "neto_aceptar_todo": n_rescate - n_dano,
                "proporcion_rescate": n_rescate / n_cambio,
                "n_filas_distintas": len(np.unique(filas)),
            }
        )
        log(
            f"   {artefacto}/{variante[:24]:24s} cambia {n_cambio:4d} celdas en "
            f"{len(np.unique(filas)):4d} filas | rescata {n_rescate:4d} | dana {n_dano:4d} | "
            f"neto {n_rescate - n_dano:+4d} | P(rescate) {n_rescate / n_cambio:.3f}"
        )
    if not challengers:
        raise SystemExit("PARADA OBLIGATORIA: ningun challenger propone un conjunto de cambio util")
    CSV_CAMBIOS.parent.mkdir(exist_ok=True)
    _escribir_cambios(challengers, ids, y, pred_v03, proba_v03, margen_v03, confianza_v03,
                      entropia_v03, pliegues, estructura)
    log(f"   tabla de cambios escrita: {CSV_CAMBIOS.relative_to(PROJECT_ROOT)}")

    log("3/6 la senal de aceptacion por challenger (P2)")
    generador = np.random.default_rng(SEMILLA)
    for c in challengers:
        vectores, nombres = _caracteristicas(c, proba_v03, margen_v03, confianza_v03,
                                            entropia_v03, estructura, ids)
        reglas = _reglas(vectores, nombres)
        base = float(c["objetivo"].mean())
        observado, por_regimen = _maximo_desviacion(c["objetivo"], reglas, base)
        excede = 0
        for _ in range(PERMUTACIONES):
            maximo, _ = _maximo_desviacion(generador.permutation(c["objetivo"]), reglas, base)
            if maximo >= observado:
                excede += 1
        p_global = (1 + excede) / (1 + PERMUTACIONES)
        c["n_caracteristicas"] = len(nombres)
        c["n_regimenes"] = len(por_regimen)
        c["maximo_desvio"] = observado
        c["p_global"] = p_global
        c["mejor_regimen"] = por_regimen[0] if por_regimen else None
        c["por_regimen"] = por_regimen
        mejor = por_regimen[0] if por_regimen else {"nombre": "ninguno", "auc": 0.5}
        log(
            f"   {c['artefacto']}/{c['variante'][:22]:22s} {len(por_regimen):4d} regimenes | "
            f"maximo |AUC-0.5| {observado:.4f} | p {p_global:.4f} | mejor {mejor['nombre'][:34]}"
        )

    log("4/6 lo que la regla daria fuera de muestra, con el corte elegido por pliegues")
    for c in challengers:
        vectores, nombres = _caracteristicas(c, proba_v03, margen_v03, confianza_v03,
                                            entropia_v03, estructura, ids)
        c["operativo"] = _rendimiento_operativo(c, vectores, nombres, pliegues)
    for c in challengers:
        for fila in c["operativo"][:3]:
            log(
                f"   {c['variante'][:20]:20s} {fila['caracteristica'][:26]:26s} "
                f"mejora fuera de muestra {fila['mejora_media']:+6.2f} celdas | "
                f"{fila['pliegues_mejoran']}/{fila['pliegues_evaluados']} pliegues mejoran"
            )

    log("5/6 el veredicto")
    veredicto = _veredicto(challengers)
    log(f"   estado: {veredicto['estado']}")
    log(
        f"   senal nominal: {veredicto['challengers_con_senal_nominal']} de "
        f"{len(challengers)} challengers | tras Bonferroni: "
        f"{veredicto['challengers_con_senal_tras_bonferroni']}, con umbral "
        f"{veredicto['umbral_bonferroni']:.4f}"
    )
    log(
        f"   mejor mejora fuera de muestra: {veredicto['mejor_mejora_fuera_de_muestra']:+.2f} "
        f"celdas, y hacen falta {veredicto['mejora_necesaria_para_la_escala']:.0f} para que la "
        "regla pueda importar en el leaderboard"
    )

    log("6/6 escribir")
    _escribir(challengers, veredicto, inicio)
    log(f"completado en {round(time.time() - inicio, 1)} s")
    log(
        f"VEREDICTO: {veredicto['estado']} | mejor p global "
        f"{veredicto['mejor_p_global']:.4f} | mejor mejora "
        f"{veredicto['mejor_mejora_fuera_de_muestra']:+.2f} celdas | submissions gastadas 0"
    )


def _proba_del_challenger(
    artefacto: str, variante: str, ids: np.ndarray, y_referencia: np.ndarray
) -> np.ndarray | None:
    """Extrae la matriz de probabilidades de un challenger, alineada al orden de V03.

    Args:
        artefacto: nombre del artefacto.
        variante: nombre de la variante dentro del artefacto.
        ids: identificadores de V03, que fijan el orden de referencia.
        y_referencia: etiquetas verdaderas de V03, para comprobar que la alineacion es correcta.

    Returns:
        La matriz de probabilidades en el orden de V03, o None si no se puede alinear.
    """
    ruta = PROJECT_ROOT / "features" / f"{artefacto}.npz"
    if not ruta.exists():
        return None
    datos = np.load(ruta, allow_pickle=True)
    if "#" in variante:
        clave, numero = variante.split("#")
        matriz = np.moveaxis(np.asarray(datos[clave], dtype=float), 1, 0)[:, int(numero), :]
    else:
        matriz = np.asarray(datos[variante], dtype=float)
    if matriz.ndim != 2 or matriz.shape[1] != 3:
        return None
    # **El mapa de posiciones es el del artefacto, no el de V03.** Los artefactos de 1731 filas no
    # tienen el mismo orden que las 1483 de V03, y usar el mapa equivocado alinea cada fila con
    # otra: asi, mod03b parec ia cambiar 2057 celdas cuando son 271. El orden de las filas no se
    # puede suponer, y el fallo es silencioso porque produce numeros plausibles, no un error.
    propios = np.asarray(datos["ids"]).astype(np.int64)
    if matriz.shape[0] != len(propios):
        return None
    mapa = {v: i for i, v in enumerate(propios)}
    if any(v not in mapa for v in ids):
        return None
    # Comprobacion de integridad: las etiquetas del artefacto, en las filas alineadas, tienen que
    # coincidir con las de V03. Si no coinciden la alineacion esta mal, y se detiene el script en
    # vez de devolver un recuento de cambios que no significa nada. Esta comprobacion es la que
    # habria atrapado el fallo en la primera ejecucion, y se anadio al verla ocurrir.
    y_artefacto = (
        np.asarray(datos["y_true"], dtype=int) if "y_true" in datos.files else None
    )
    if (
        y_artefacto is not None
        and y_artefacto.ndim == 2
        and y_artefacto.shape[1] == 3
        and not np.array_equal(y_artefacto[[mapa[v] for v in ids]], y_referencia)
    ):
        raise SystemExit(
            f"PARADA OBLIGATORIA: las etiquetas de {artefacto} no coinciden con las de V03 "
            "en las filas alineadas, luego la alineacion por identificador esta mal"
        )
    return matriz[[mapa[v] for v in ids]]


def _escribir_cambios(
    challengers: list[dict[str, Any]],
    ids: np.ndarray,
    y: np.ndarray,
    pred_v03: np.ndarray,
    proba_v03: np.ndarray,
    margen_v03: np.ndarray,
    confianza_v03: np.ndarray,
    entropia_v03: np.ndarray,
    pliegues: np.ndarray,
    estructura: pd.DataFrame,
) -> None:
    """Escribe la tabla celda a celda de los conjuntos de cambio, una fila por celda y challenger.

    Args:
        challengers: challengers con su conjunto de cambio ya construido.
        ids: identificadores de V03.
        y: etiquetas verdaderas.
        pred_v03: predicciones de V03.
        proba_v03: probabilidades de V03.
        margen_v03: margenes de V03.
        confianza_v03: confianza de V03.
        entropia_v03: entropia de V03.
        pliegues: pliegue de cada fila.
        estructura: tabla de features auxiliares indexada por identificador.
    """
    principales = [
        "n_mencion_inicial", "n_mencion_interna", "n_hashtags", "n_urls", "n_tokens",
        "n_oraciones", "n_negaciones", "n_imperativos", "n_propn", "n_puntuacion_interna",
    ]
    disponibles = [c for c in principales if c in estructura.columns]
    bloques = []
    for c in challengers:
        filas, cols = c["filas"], c["cols"]
        marco = pd.DataFrame(
            {
                "challenger": f"{c['artefacto']}/{c['variante']}",
                "id": ids[filas],
                "etiqueta": [ETIQUETAS[k] for k in cols],
                "y_true": y[filas, cols],
                "pred_v03": pred_v03[filas, cols],
                "cambio": c["pred"][filas, cols],
                "es_rescate": c["objetivo"],
                "margen_v03": margen_v03[filas, cols],
                "confianza_v03": confianza_v03[filas, cols],
                "entropia_v03": entropia_v03[filas, cols],
                "pliegue": pliegues[filas],
                "score_v03": proba_v03[filas, cols],
                "score_challenger": c["proba"][filas, cols],
                "diferencia_de_scores": c["proba"][filas, cols] - proba_v03[filas, cols],
            }
        )
        if disponibles:
            # Asignacion posicional, no join por columna: un join con el indice del marco derecho
            # chocando con el del izquierdo duplica filas, y la tabla terminaba con 266 filas para
            # un conjunto de cambio de 202 celdas.
            for columna in disponibles:
                marco[columna] = (
                    estructura[columna].reindex(marco["id"].to_numpy()).to_numpy()
                )
        bloques.append(marco)
    pd.concat(bloques, ignore_index=True).to_csv(CSV_CAMBIOS, index=False, lineterminator="\n")


def _caracteristicas(
    challenger: dict[str, Any],
    proba_v03: np.ndarray,
    margen_v03: np.ndarray,
    confianza_v03: np.ndarray,
    entropia_v03: np.ndarray,
    estructura: pd.DataFrame,
    ids: np.ndarray,
) -> tuple[list[np.ndarray], list[str]]:
    """Construye las caracteristicas candidatas sobre el conjunto de cambio de un challenger.

    Las tres primeras son las que el encargo de la fase admite explicitamente: margen de V03,
    diferencia de score frente a V03, y confianza. Las demas son todas las features de la tabla
    estructurada, y se prueban todas precisamente para no elegir a mano las que mas gusten, que es
    como se encuentra una coincidencia en vez de una senal.

    Args:
        challenger: challenger con su conjunto de cambio.
        proba_v03: probabilidades de V03.
        margen_v03: margenes de V03.
        confianza_v03: confianza de V03.
        entropia_v03: entropia de V03.
        estructura: tabla de features auxiliares indexada por identificador.
        ids: identificadores de V03.

    Returns:
        La lista de vectores y la lista de nombres, en el mismo orden.
    """
    filas, cols = challenger["filas"], challenger["cols"]
    vectores = [
        np.abs(margen_v03[filas, cols]),
        np.abs(challenger["proba"][filas, cols] - proba_v03[filas, cols]),
        confianza_v03[filas, cols],
        entropia_v03[filas, cols],
    ]
    nombres = ["margen_abs_v03", "diferencia_abs_de_scores", "confianza_v03", "entropia_v03"]
    identificadores = ids[filas]
    for columna in estructura.columns:
        if columna == "particion" or not pd.api.types.is_numeric_dtype(estructura[columna]):
            continue
        valores = estructura[columna].reindex(identificadores).to_numpy(dtype=float)
        if not np.isfinite(valores).all() or float(np.std(valores)) == 0.0:
            continue
        vectores.append(valores)
        nombres.append(f"aux_{columna}")
    return vectores, nombres


def _reglas(vectores: list[np.ndarray], nombres: list[str]) -> list[tuple[str, np.ndarray, bool]]:
    """Convierte cada caracteristica en reglas de aceptacion candidatas.

    Una regla es una mascara booleana sobre el conjunto de cambio: dice que celdas aceptan el
    cambio del challenger. Para una feature continua se prueban cortes en los cuantiles, en los dos
    sentidos, porque no se sabe de antemano si el extremo alto rescata o dana. Para una booleana se
    prueban las dos condiciones.

    Args:
        vectores: vectores de caracteristicas.
        nombres: nombres de las caracteristicas.

    Returns:
        La lista de reglas, con su nombre, su mascara y si su variable es continua.
    """
    reglas: list[tuple[str, np.ndarray, bool]] = []
    for vector, nombre in zip(vectores, nombres, strict=True):
        valores_unicos = np.unique(vector)
        if valores_unicos.size <= 2 and set(valores_unicos.tolist()) <= {0.0, 1.0}:
            reglas.append((f"{nombre}>=1", vector >= 1.0, False))
            reglas.append((f"{nombre}==0", vector == 0.0, False))
            continue
        for corte in np.quantile(vector, CORTES_CUANTILES):
            reglas.append((f"{nombre}>={corte:.4f}", vector >= corte, True))
            reglas.append((f"{nombre}<={corte:.4f}", vector <= corte, True))
    return reglas


def _auc(objetivo: np.ndarray, mascara: np.ndarray) -> float:
    """Cuanto separa un regimen los rescates de los danos, por la definicion de pares.

    El score es **la pertenencia al regimen**: dentro vale 1 y fuera vale 0. Ordenar por la propia
    etiqueta no seria un AUC, seria un circulo que siempre da 1.

    Se cuentan los pares (rescate, dano) y se weighan segun en que lado del regimen caiga cada uno:

    - rescate dentro y dano fuera: el rescate va por delante, pesa 1.
    - los dos dentro, o los dos fuera: empate, pesa 0.5.
    - rescate fuera y dano dentro: el rescate va por detras, pesa 0.

    El denominador es el total de pares, que es el mismo para todas las permutaciones de las
    etiquetas, de modo que la distribucion nula de este estadistico es la del propio recuento y no
    hay que estimar nada mas.

    Args:
        objetivo: 1 si la celda es rescate, 0 si es dano.
        mascara: celdas del regimen.

    Returns:
        El AUC del regimen, o 0.5 si el regimen no contiene ambas clases.
    """
    rescates = objetivo == 1
    danos = objetivo == 0
    rescates_dentro = int(np.sum(rescates & mascara))
    danos_dentro = int(np.sum(danos & mascara))
    rescates_fuera = int(np.sum(rescates & ~mascara))
    danos_fuera = int(np.sum(danos & ~mascara))
    if (rescates_dentro + rescates_fuera) == 0 or (danos_dentro + danos_fuera) == 0:
        return 0.5
    if (rescates_dentro + danos_dentro) == 0 or (rescates_fuera + danos_fuera) == 0:
        return 0.5
    por_delante = rescates_dentro * danos_fuera
    empates = rescates_dentro * danos_dentro + rescates_fuera * danos_fuera
    total = (rescates_dentro + rescates_fuera) * (danos_dentro + danos_fuera)
    return float((por_delante + 0.5 * empates) / total)


def _maximo_desviacion(
    objetivo: np.ndarray, reglas: list[tuple[str, np.ndarray, bool]], base: float
) -> tuple[float, list[dict[str, Any]]]:
    """Calcula el maximo de |AUC - 0.5| sobre todas las reglas, y las ordena.

    Args:
        objetivo: 1 si la celda es rescate, 0 si es dano.
        reglas: reglas de aceptacion candidatas.
        base: proporcion de rescates de referencia.

    Returns:
        El maximo observado y la lista de reglas medidas, ordenada.
    """
    filas = []
    for nombre, mascara, _ in reglas:
        n = int(mascara.sum())
        if n < MINIMO_CELDAS_POR_REGIMEN or n > objetivo.size - MINIMO_CELDAS_POR_REGIMEN:
            continue
        dentro = objetivo[mascara]
        auc = _auc(objetivo, mascara)
        filas.append(
            {
                "nombre": nombre,
                "n": n,
                "auc": auc,
                "desvio": abs(auc - 0.5),
                "p_rescate": float(dentro.mean()),
                "p_base": base,
                "lift": float(dentro.mean()) - base,
            }
        )
    filas.sort(key=lambda f: -f["desvio"])
    if not filas:
        return 0.0, []
    return filas[0]["desvio"], filas


def _neto(objetivo: np.ndarray, acepta: np.ndarray, evaluados: np.ndarray) -> int:
    """Rescate neto de una regla sobre un subconjunto de celdas.

    Args:
        objetivo: 1 si la celda es rescate, 0 si es dano.
        acepta: celdas que la regla acepta.
        evaluados: celdas sobre las que se evalua.

    Returns:
        Rescates aceptados menos danos aceptados.
    """
    dentro = evaluados & acepta
    rescates = int(objetivo[dentro].sum())
    return rescates - (int(dentro.sum()) - rescates)


def _rendimiento_operativo(
    challenger: dict[str, Any],
    vectores: list[np.ndarray],
    nombres: list[str],
    pliegues: np.ndarray,
) -> list[dict[str, Any]]:
    """Mide cuanto mejora la regla el rescate neto, eligiendo la regla fuera de muestra.

    La regla se elige en cuatro pliegues, con la regla fija "aceptar todos los cambios" como
    referencia, y se evalua en el quinto. Se rotan los cinco pliegues. Asi el numero que sale es el
    de una regla desplegada, y no el maximo sobre los mismos datos que se evaluan.

    Args:
        challenger: challenger con su conjunto de cambio.
        vectores: vectores de caracteristicas.
        nombres: nombres de las caracteristicas.
        pliegues: pliegue de cada fila.

    Returns:
        Una fila por caracteristica, ordenada por mejora media.
    """
    objetivo = challenger["objetivo"]
    plegue_celda = pliegues[challenger["filas"]]
    neto_aceptar_todo = int(objetivo.sum()) - int(objetivo.size - objetivo.sum())
    reglas = _reglas(vectores, nombres)
    # Las reglas se indexan por caracteristica para poder reportar la mejor de cada una, que es lo
    # que.comparece una regla real usaria.
    por_caracteristica: dict[str, list[tuple[str, np.ndarray, bool]]] = {}
    for nombre, mascara, continua in reglas:
        por_caracteristica.setdefault(nombre.split(">=")[0].split("<=")[0], []).append(
            (nombre, mascara, continua)
        )
    # **La referencia es la mejor de las dos operaciones triviales.** Aceptar todos los cambios y
    # no aceptar ninguno son las dos cosas que siempre se pueden hacer, y la segunda es V03, que es
    # el champion con el mejor score conocido. Medir la mejora solo contra aceptar todo hace que un
    # challenger perdedor obtenga una mejora enorme con la regla degenerada de no aceptar nada,
    # que no es una regla sino el champion disfrazado. Por eso la referencia es el maximo de los
    # dos, y una regla tiene que superarlo para significar algo.
    referencia = max(0, neto_aceptar_todo)
    salida = []
    todos = np.ones(objetivo.size, dtype=bool)
    for caracteristica, candidatas in por_caracteristica.items():
        diferencias = []
        for pliegue in np.unique(plegue_celda):
            ajuste = plegue_celda != pliegue
            prueba = plegue_celda == pliegue
            if ajuste.sum() < MINIMO_CELDAS_POR_REGIMEN or prueba.sum() < 5:
                continue
            elegido = max(candidatas, key=lambda r: _neto(objetivo, r[1], ajuste))
            base_prueba = max(0, _neto(objetivo, todos, prueba))
            diferencias.append(_neto(objetivo, elegido[1], prueba) - base_prueba)
        if not diferencias:
            continue
        valores = np.array(diferencias, dtype=float)
        salida.append(
            {
                "caracteristica": caracteristica,
                "neto_aceptar_todo": neto_aceptar_todo,
                "referencia": referencia,
                "mejora_media": float(valores.mean()),
                "mejora_desviacion": float(valores.std(ddof=1)) if valores.size > 1 else 0.0,
                "pliegues_mejoran": int(np.sum(valores > 0)),
                "pliegues_evaluados": int(valores.size),
            }
        )
    salida.sort(key=lambda f: -f["mejora_media"])
    return salida


def _veredicto(challengers: list[dict[str, Any]]) -> dict[str, Any]:
    """Decide el estado de la fase a partir de la senal medida en cada challenger.

    Args:
        challengers: challengers ya medidos.

    Returns:
        El veredicto de la fase.
    """
    # Cada challenger aporta un test global propio, asi que tambien hay que corregir entre ellos.
    # Con cinco challengers el umbral de Bonferroni es 0.01, y no basta con que uno solo pase
    # 0.05. Ademas, un p nominal sin efecto operativo no es una senal competitiva: sirve para
    # entender el comportamiento del modelo, no para gastar una submission.
    con_senal_nominal = [c for c in challengers if c["p_global"] < 0.05]
    umbral_corregido = 0.05 / len(challengers)
    con_senal_corregida = [c for c in challengers if c["p_global"] < umbral_corregido]
    mejor_mejora = max(
        (f["mejora_media"] for c in challengers for f in c["operativo"]), default=0.0
    )
    escala = FRACCION_DE_ESCALA_NECESARIA * CELDAS_POR_003_LB
    hay_mejora_competitiva = mejor_mejora >= escala
    if con_senal_corregida and hay_mejora_competitiva:
        estado = "NEW COMPETITIVE SIGNAL FOUND"
    else:
        estado = "NO SELECTIVE CORRECTION SIGNAL"
    return {
        "estado": estado,
        "challengers_medidos": len(challengers),
        "challengers_con_senal_nominal": len(con_senal_nominal),
        "challengers_con_senal_tras_bonferroni": len(con_senal_corregida),
        "umbral_bonferroni": umbral_corregido,
        "mejor_p_global": min(c["p_global"] for c in challengers),
        "mejor_mejora_fuera_de_muestra": mejor_mejora,
        "mejora_necesaria_para_la_escala": escala,
        "hay_mejora_competitiva": hay_mejora_competitiva,
        "detalle_por_challenger": [
            {
                "challenger": f"{c['artefacto']}/{c['variante']}",
                "n_celdas_de_cambio": c["n_cambio"],
                "n_rescate": c["n_rescate"],
                "n_dano": c["n_dano"],
                "neto_aceptar_todo": c["neto_aceptar_todo"],
                "n_regimenes": c["n_regimenes"],
                "maximo_desvio": c["maximo_desvio"],
                "p_global": c["p_global"],
                "mejor_regimen": c["mejor_regimen"]["nombre"] if c["mejor_regimen"] else None,
                "mejor_auc": c["mejor_regimen"]["auc"] if c["mejor_regimen"] else 0.5,
                "mejor_mejora_fuera_de_muestra": (
                    c["operativo"][0]["mejora_media"] if c["operativo"] else 0.0
                ),
                "mejor_caracteristica": (
                    c["operativo"][0]["caracteristica"] if c["operativo"] else None
                ),
            }
            for c in challengers
        ],
    }


def _escribir(
    challengers: list[dict[str, Any]], veredicto: dict[str, Any], inicio: float
) -> None:
    """Escribe la tabla de senal y el informe de la fase.

    Args:
        challengers: challengers ya medidos.
        veredicto: veredicto de la fase.
        inicio: instante de arranque del script.
    """
    with CSV_SENAL.open("w", encoding="utf-8") as manejador:
        manejador.write("challenger,nombre,n,auc,desvio,p_rescate,p_base,lift\n")
        for c in challengers:
            for f in c["por_regimen"]:
                manejador.write(
                    f"{c['artefacto']}/{c['variante']},{f['nombre']},{f['n']},"
                    f"{f['auc']:.6f},{f['desvio']:.6f},{f['p_rescate']:.6f},"
                    f"{f['p_base']:.6f},{f['lift']:+.6f}\n"
                )
    from v11_ayudas import fila, sello

    linea: list[str] = [
        "# V18-01: conjunto de cambio y senal de aceptacion",
        "",
        f"Generado por `experiments/v18_01_cambio_y_senal.py` ({EXPERIMENTO}).",
        "",
        "**No se entrena nada, no se construye ninguna regla y no se gasta ninguna submission.**",
        "",
        "## 1. Los conjuntos de cambio, uno por challenger (P1)",
        "",
        "Cada challenger propone un conjunto distinto, con celdas distintas, y su veredicto tiene que",
        "sostenerse en su propio conjunto. Agregarlos exigiria que coincidieran las celdas, que es",
        "justo lo que no pasa.",
        "",
        fila(["challenger", "cambia", "filas", "rescata", "dana", "neto", "P(rescate)"]),
        fila(["---"] * 7),
    ]
    for c in challengers:
        linea.append(
            fila(
                [
                    f"{c['artefacto']}/{c['variante'][:22]}", str(c["n_cambio"]),
                    str(c["n_filas_distintas"]), str(c["n_rescate"]), str(c["n_dano"]),
                    f"{c['neto_aceptar_todo']:+d}", f"{c['proporcion_rescate']:.3f}",
                ]
            )
        )
    linea += [
        "",
        "## 2. Lo que se midio, y por que el objetivo no es la etiqueta",
        "",
        "El objetivo de cada feature es si el **cambio propuesto** es correcto frente a V03: 1 es",
        "rescate y 0 es dano. Predecir la etiqueta es otra pregunta, y es la que ya se midio en V15",
        "sin que eso produjera nada. Aqui las dos clases son los desacuerdos entre dos modelos, y",
        "lo que decide es cual de los dos se equivoco.",
        "",
        "## 3. El test global de cada challenger, que es el que decide",
        "",
        "El estadistico es el maximo de |AUC - 0.5| sobre todas las reglas, y su nula se obtiene",
        f"permutando las etiquetas {PERMUTACIONES} veces. Es un test exacto bajo la nula global, asi",
        "que no hace falta corregir a mano por el numero de pruebas. Un regimen necesita al menos",
        f"{MINIMO_CELDAS_POR_REGIMEN} celdas para que su proporcion tenga precision suficiente.",
        "",
        fila(["challenger", "regimenes", "maximo desvio", "p global", "mejor regimen",
              "mejor AUC"]),
        fila(["---"] * 6),
    ]
    for c in challengers:
        linea.append(
            fila(
                [
                    f"{c['artefacto']}/{c['variante'][:20]}", str(c["n_regimenes"]),
                    f"{c['maximo_desvio']:.4f}", f"{c['p_global']:.4f}",
                    (c["mejor_regimen"]["nombre"][:32] if c["mejor_regimen"] else "ninguno"),
                    f"{c['mejor_regimen']['auc']:.4f}" if c["mejor_regimen"] else "0.5000",
                ]
            )
        )
    linea += [
        "",
        "## 4. Lo que la regla daria fuera de muestra",
        "",
        "La regla se elige en cuatro pliegues y se evalua en el quinto, rotando los cinco. La",
        "mejora es sobre el neto de aceptar todos los cambios, que es la referencia real: una regla",
        "que no mejora esa cifra no aporta nada, por buena que parezca la separacion.",
        "",
        fila(["challenger", "caracteristica", "mejora media", "desviacion",
              "pliegues que mejoran"]),
        fila(["---"] * 5),
    ]
    for c in challengers:
        for f in c["operativo"][:3]:
            linea.append(
                fila(
                    [
                        c["variante"][:20], f["caracteristica"][:26], f"{f['mejora_media']:+.2f}",
                        f"{f['mejora_desviacion']:.2f}",
                        f"{f['pliegues_mejoran']}/{f['pliegues_evaluados']}",
                    ]
                )
            )
    linea += [
        "",
        "## 5. Veredicto",
        "",
        f"**{veredicto['estado']}**",
        "",
        (
            "- Challengers con p nominal por debajo de 0.05: "
            f"**{veredicto['challengers_con_senal_nominal']} de "
            f"{veredicto['challengers_medidos']}**."
        ),
        (
            "- Challengers que sobreviven a la correccion de Bonferroni entre challengers, "
            f"con umbral {veredicto['umbral_bonferroni']:.4f}: "
            f"**{veredicto['challengers_con_senal_tras_bonferroni']}**."
        ),
        f"- Mejor valor p global: **{veredicto['mejor_p_global']:.4f}**.",
        (
            "- Mejor mejora de neto fuera de muestra: "
            f"**{veredicto['mejor_mejora_fuera_de_muestra']:+.2f} celdas**, frente a las "
            f"**{veredicto['mejora_necesaria_para_la_escala']:.0f}** que hacen falta para que una "
            "regla de este tamano pueda mover el leaderboard."
        ),
        "- Submissions gastadas: **0**.",
        "",
    ]
    linea += sello(inicio)
    REPORTE_MD.write_text("\n".join(linea) + "\n", encoding="utf-8")
    REPORTE_JSON.write_text(
        json.dumps(
            {
                "experimento": EXPERIMENTO,
                "fecha": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "referencia_v03": REFERENCIA_V03,
                "lb_v03": LB_V03,
                "permuta": PERMUTACIONES,
                "minimo_celdas_por_regimen": MINIMO_CELDAS_POR_REGIMEN,
                "veredicto": veredicto,
                "gastos": {"submissions": 0, "modelos_entrenados": 0, "artefactos_regenerados": 0},
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
