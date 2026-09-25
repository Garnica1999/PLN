"""FASE FINAL - Candidatos finales: OOF de V03 y screening de metas de canales.

Pregunta que responde
---------------------
La fase v2 cerro con V03 (LB 0.78306) y con el meta de dependencia
``META-BASES-RIDGE`` (corrected_cv 0.7874, holdout 0.8465) adoptado en P6 pero
NO transferido al leaderboard (v08 0.78163, D-020). A la vez, la fase v2 dejo
cuatro canales OOF nuevos (emocion EMO-ALL, sintaxis SYN-3, targeting
CH-TGT+DIST y cita/negacion CITA) que nunca se probaron contra el meta
adoptado: P6 solo probo los canales de targeting. Esta fase cierra la pregunta
con un screening restringido:

1. Congelar la referencia V03 en un artefacto propio
   (``features/oof_v03_final.npz``) con probabilidades, prediccion con los
   umbrales congelados 0.44/0.51/0.46, margen, entropia y confianza.
2. Evaluar metas pequenos anidados (Ridge alpha {0.3, 1, 3}; LR C
   {0.1, 0.3, 1}) que usan como unico bloque las probabilidades OOF de V03 mas
   las probabilidades OOF de los canales, sin concatenar features densas.
3. Clasificar los candidatos como PROMISING (mejora corrected CV y >= 3/5
   pliegues) o COMPLEMENTARY (nivel <= referencia pero fusion consistente con
   IC bootstrap > 0) y gastar como maximo 2 evaluaciones NUEVAS del holdout
   congelado (quedan 3 libres; P7 uso 2/5) en el mejor y el segundo mejor
   candidato nuevo.

Metodo
------
- Desarrollo n=1483, pliegues corregidos v2 (D-016) y verificacion de que el
  OOF congelado de M2 con los umbrales de V03 reproduce 0.7772.
- Los OOF de canales se cargan de los stores v2 con verificacion de ids,
  etiquetas y pliegues; nada se ajusta con validacion.
- El meta se ajusta SOLO con el entrenamiento de cada pliegue (escalador y 3
  modelos binarios) y predice su validacion, reutilizando las funciones de
  ``experiments/17``. El sesgo del OOF congelado dentro del entrenamiento del
  meta ya se midio en P1 (<= 0.0003).
- Regla de seleccion de canal: por familia se elige la variante PURA (sin la
  representacion de V03; excluye prefijos ``TXT+``, ``NB-`` y ``ORT-``) con
  mayor F1 standalone a 0.5; el ranking completo queda como evidencia.
- Candidatos: C-EMO, C-SYN, C-EMO-SYN, C-EMO-SYN-TGT-CYT (probabilidades de
  V03 + canales), C-REF-44 (control que debe reproducir el meta adoptado
  0.7874) y C-REF-44-CH (el meta de 44 columnas + probabilidades de los cuatro
  canales puros). Para el mejor candidato pequeno se prueba ademas la variante
  +6 (margen y entropia de las probabilidades de V03).
- Holdout (maximo 2 evaluaciones nuevas): se reajustan sobre el desarrollo la
  referencia V03 (M2) y los canales seleccionados reutilizando las recetas de
  ``experiments/14``, ``16``, ``19`` y ``20`` con un pseudo-pliegue
  desarrollo -> holdout; el meta se ajusta con el OOF del desarrollo y predice
  el holdout a 0.5. La medicion de V03 en el holdout no se registra como
  evaluacion nueva: reproduce la ya registrada P7-01 y solo sirve de columna de
  comparacion.

Reglas de honestidad
--------------------
- El test no se lee en ningun punto del script.
- El holdout solo se usa para medir los finalistas elegidos en el desarrollo;
  ningun hiperparametro se ajusta con el.
- Si ningun candidato supera las referencias con evidencia, se reporta con los
  numeros y sin adoptar nada.

Salidas
-------
``features/oof_v03_final.npz``, ``reports/canal_ranking_final.csv``,
``reports/final_candidates.csv``, filas ``FIN-01..`` en ``experiments.csv`` y
las evaluaciones nuevas en ``reports/holdout_evaluaciones_v2.csv``.

Ejecucion
---------
    /home/carlos/venv/bin/python experiments/23_candidatos_finales.py
"""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src import data as data_module
from src.evaluation import metricas_multietiqueta
from src.splits import ETIQUETAS

RUTA_P17 = PROJECT_ROOT / "experiments" / "17_stacking_v2.py"
RUTA_P14 = PROJECT_ROOT / "experiments" / "14_oof_corregido_v2.py"
RUTA_P16 = PROJECT_ROOT / "experiments" / "16_targeting_v2.py"
RUTA_P19 = PROJECT_ROOT / "experiments" / "19_sintaxis_cita_negacion_v2.py"
RUTA_P20 = PROJECT_ROOT / "experiments" / "20_emocion_nbsvm_ortografia_v2.py"

PARTICION_PATH = PROJECT_ROOT / "data" / "processed" / "holdout_v2_ids.csv"
HOLDOUT_CONFIG_PATH = PROJECT_ROOT / "config" / "holdout_v2.json"
STORES_CANALES: Final[dict[str, Path]] = {
    "emocion": PROJECT_ROOT / "features" / "v2_oof_emocion_nbsvm_ortografia.npz",
    "sintaxis": PROJECT_ROOT / "features" / "v2_oof_syntax.npz",
    "targeting": PROJECT_ROOT / "features" / "v2_oof_targeting.npz",
    "cita_negacion": PROJECT_ROOT / "features" / "v2_oof_cita_negacion.npz",
}
OOF_V03_FINAL_PATH = PROJECT_ROOT / "features" / "oof_v03_final.npz"
RANKING_CANALES_PATH = PROJECT_ROOT / "reports" / "canal_ranking_final.csv"
FINAL_CANDIDATES_PATH = PROJECT_ROOT / "reports" / "final_candidates.csv"
EXPERIMENTS_CSV = PROJECT_ROOT / "experiments.csv"
HOLDOUT_EVALUACIONES_PATH = PROJECT_ROOT / "reports" / "holdout_evaluaciones_v2.csv"

CONFIG: Final[dict[str, Any]] = {
    "fecha": "2026-09-25",
    "referencia_v03": 0.7772,
    "referencia_meta_v08": 0.7874,
    "referencia_m2_umbral_05": 0.7718,
    "umbrales_v03": {"HS": 0.44, "TR": 0.51, "AG": 0.46},
    "umbral_meta": 0.5,
    "ridge_alpha": (0.3, 1.0, 3.0),
    "lr_C": (0.1, 0.3, 1.0),
    "mejora_minima": 0.005,
    "ensemble_gain_minimo": 0.0015,
    "pliegues_minimos_mejorados": 3,
    "max_evaluaciones_holdout": 5,
    "evaluaciones_nuevas_holdout": 2,
    "tolerancia_replicacion_v03": 5e-4,
    "tolerancia_holdout_v03": 1e-3,
    "n_holdout_esperado": 248,
    "holdout_v03_registrado": 0.8285,
    "holdout_meta_v08_registrado": 0.8465,
}

# Candidatos pequenos: clave -> familias de canal incluidas (orden de concatenacion).
CANDIDATOS_PEQUENOS: Final[dict[str, tuple[str, ...]]] = {
    "C-EMO": ("emocion",),
    "C-SYN": ("sintaxis",),
    "C-EMO-SYN": ("emocion", "sintaxis"),
    "C-EMO-SYN-TGT-CYT": ("emocion", "sintaxis", "targeting", "cita_negacion"),
}
CANDIDATOS_REFERENCIA: Final[tuple[str, ...]] = ("C-REF-44", "C-REF-44-CH")
PREFIJOS_NO_PUROS: Final[tuple[str, ...]] = ("TXT+", "NB-", "ORT-")
CLAVE_V03: Final[str] = "v03"


def log(mensaje: str) -> None:
    """Imprime un mensaje de progreso con prefijo de fase.

    Args:
        mensaje: texto a imprimir.
    """
    print(f"[FIN] {mensaje}", flush=True)


def cargar_modulo(nombre: str, ruta: Path) -> Any:
    """Importa un script de experimento como modulo reutilizable.

    Args:
        nombre: nombre interno del modulo.
        ruta: ruta del script a importar.

    Returns:
        Modulo cargado.

    Raises:
        ImportError: si el script no se puede cargar.
    """
    especificacion = importlib.util.spec_from_file_location(nombre, ruta)
    if especificacion is None or especificacion.loader is None:
        raise ImportError(f"No se pudo cargar {ruta}")
    modulo = importlib.util.module_from_spec(especificacion)
    sys.modules[especificacion.name] = modulo
    especificacion.loader.exec_module(modulo)
    return modulo


# ===== SECCION 1: carga del holdout y de los stores OOF =====


def cargar_holdout() -> pd.DataFrame:
    """Carga el holdout congelado por su particion registrada.

    Returns:
        DataFrame del holdout.

    Raises:
        RuntimeError: si el tamano no coincide con ``config/holdout_v2.json``.
    """
    train = data_module.read_train()
    particion = pd.read_csv(PARTICION_PATH)
    ids_holdout = particion.loc[particion["particion"] == "holdout", "id"].astype(int).tolist()
    holdout = train[train["id"].astype(int).isin(ids_holdout)].reset_index(drop=True)
    if len(holdout) != CONFIG["n_holdout_esperado"]:
        raise RuntimeError(
            f"El holdout tiene {len(holdout)} filas, se esperaban {CONFIG['n_holdout_esperado']}"
        )
    return holdout


def cargar_store_canales(
    ruta: Path,
    desarrollo: pd.DataFrame,
    pliegue_de_fila: NDArray[np.int_],
    y_verdadera: NDArray[np.int_],
) -> tuple[NDArray[np.float64], tuple[str, ...]]:
    """Carga un store OOF de canales y lo alinea con el desarrollo.

    Args:
        ruta: ruta del ``.npz`` del store.
        desarrollo: DataFrame del desarrollo.
        pliegue_de_fila: pliegue corregido de cada fila del desarrollo.
        y_verdadera: etiquetas verdaderas ``(n, 3)``.

    Returns:
        Tupla ``(proba, nombres)`` con ``proba`` de forma ``(n, k, 3)``.

    Raises:
        RuntimeError: si los ids, etiquetas, pliegues o nombres no coinciden.
    """
    datos = np.load(ruta, allow_pickle=True)
    ids = datos["ids"].astype(int)
    mapa_fila = {
        int(identificador): posicion
        for posicion, identificador in enumerate(desarrollo["id"].astype(int))
    }
    orden = np.array([mapa_fila[int(identificador)] for identificador in ids])
    if not np.array_equal(datos["y_true"], y_verdadera[orden]):
        raise RuntimeError(f"Las etiquetas de {ruta.name} no coinciden con el desarrollo")
    clave_pliegue = "pliegues" if "pliegues" in datos.files else "pliegue"
    if not np.array_equal(datos[clave_pliegue].astype(int), pliegue_de_fila[orden]):
        raise RuntimeError(f"Los pliegues de {ruta.name} no coinciden con los pliegues corregidos")
    etiquetas_store = tuple(str(valor) for valor in datos["etiquetas"])
    if etiquetas_store != tuple(ETIQUETAS):
        raise RuntimeError(f"Orden de etiquetas inesperado en {ruta.name}: {etiquetas_store}")
    clave_nombres = "canales" if "canales" in datos.files else "variantes"
    nombres = tuple(str(valor) for valor in datos[clave_nombres])
    proba = np.asarray(datos["proba"], dtype=float)
    if proba.shape[0] == len(nombres) and proba.shape[1] == len(ids):
        proba = np.transpose(proba, (1, 0, 2))
    if proba.shape != (len(ids), len(nombres), len(ETIQUETAS)):
        raise RuntimeError(f"Forma inesperada del OOF de {ruta.name}: {proba.shape}")
    alineada = np.zeros((len(desarrollo), len(nombres), len(ETIQUETAS)))
    alineada[orden] = proba
    return alineada, nombres


def clave_emocion_evaluada(nombre: str) -> str:
    """Convierte el nombre de un canal de emocion del store en su clave evaluada.

    Los canales del store de emocion incluyen el valor de ``C`` en el nombre
    (``EMO-ALL@C=1.0``); las claves que produce
    ``experiments/20.evaluar_seccion_emocion`` tienen esa misma forma.

    Args:
        nombre: nombre del canal en el store.

    Returns:
        Clave con la forma ``VARIANTE@C=valor``.
    """
    if "@C=" in nombre:
        variante, valor_c = nombre.split("@C=")
        return f"{variante}@C={valor_c}"
    return f"{nombre}@C=1.0"


def es_canal_puro(nombre: str) -> bool:
    """Indica si una variante de canal es pura (no contiene la representacion de V03).

    Args:
        nombre: nombre de la variante.

    Returns:
        ``True`` si la variante puede entrar al screening de canales.
    """
    return not any(nombre.startswith(prefijo) for prefijo in PREFIJOS_NO_PUROS)


def ranking_canales(
    probas: dict[str, NDArray[np.float64]],
    nombres: dict[str, tuple[str, ...]],
    y_verdadera: NDArray[np.int_],
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Mide el F1 standalone de cada canal y elige el representante puro por familia.

    Args:
        probas: mapa familia -> probabilidades ``(n, k, 3)``.
        nombres: mapa familia -> nombres de las variantes.
        y_verdadera: etiquetas verdaderas ``(n, 3)``.

    Returns:
        Tupla ``(tabla, seleccion)`` con el ranking completo y el nombre elegido
        por familia.
    """
    filas = []
    seleccion: dict[str, str] = {}
    for familia, matriz in probas.items():
        candidatos: list[tuple[float, str]] = []
        for posicion, nombre in enumerate(nombres[familia]):
            prediccion = (matriz[:, posicion, :] >= 0.5).astype(int)
            metrica = metricas_multietiqueta(y_verdadera, prediccion)
            filas.append(
                {
                    "familia": familia,
                    "variante": nombre,
                    "puro": es_canal_puro(nombre),
                    "f1_standalone_05": metrica["f1_media_columnas"],
                    "f1_HS": metrica["f1_por_etiqueta"]["HS"],
                    "f1_TR": metrica["f1_por_etiqueta"]["TR"],
                    "f1_AG": metrica["f1_por_etiqueta"]["AG"],
                }
            )
            if es_canal_puro(nombre):
                candidatos.append((metrica["f1_media_columnas"], nombre))
        if not candidatos:
            raise RuntimeError(f"La familia {familia} no tiene variantes puras")
        seleccion[familia] = max(candidatos)[1]
        log(f"canal puro de {familia}: {seleccion[familia]} (F1 standalone {max(candidatos)[0]:.4f})")
    tabla = pd.DataFrame(filas).sort_values(["familia", "f1_standalone_05"], ascending=[True, False])
    return tabla, seleccion


# ===== SECCION 2: OOF de la referencia V03 congelada =====


def guardar_oof_v03_final(
    proba_v03: NDArray[np.float64],
    y_verdadera: NDArray[np.int_],
    pliegue_de_fila: NDArray[np.int_],
    ids: NDArray[np.int_],
    referencias: dict[str, Any],
) -> dict[str, Any]:
    """Guarda el OOF de V03 con prediccion, margen, entropia y confianza.

    Args:
        proba_v03: probabilidades OOF de M2 ``(n, 3)``.
        y_verdadera: etiquetas verdaderas ``(n, 3)``.
        pliegue_de_fila: pliegue corregido de cada fila.
        ids: identificadores de las filas.
        referencias: referencias pre-registradas (umbrales de V03).

    Returns:
        Diccionario con las verificaciones de replicacion.

    Raises:
        RuntimeError: si la F1 con umbrales congelados no reproduce 0.7772.
    """
    umbrales = np.array([referencias["umbrales_v03"][etiqueta] for etiqueta in ETIQUETAS])
    prediccion = (proba_v03 >= umbrales).astype(int)
    metrica = metricas_multietiqueta(y_verdadera, prediccion)
    diferencia = abs(metrica["f1_media_columnas"] - CONFIG["referencia_v03"])
    if diferencia > CONFIG["tolerancia_replicacion_v03"]:
        raise RuntimeError(
            f"V03 no reproduce: {metrica['f1_media_columnas']:.6f} vs {CONFIG['referencia_v03']}"
        )
    margen = np.abs(proba_v03 - 0.5)
    recortada = np.clip(proba_v03, 1e-6, 1 - 1e-6)
    entropia = -(recortada * np.log(recortada) + (1 - recortada) * np.log(1 - recortada))
    confianza = np.maximum(proba_v03, 1 - proba_v03)
    f1_pliegues = [
        float(
            metricas_multietiqueta(
                y_verdadera[pliegue_de_fila == numero], prediccion[pliegue_de_fila == numero]
            )["f1_media_columnas"]
        )
        for numero in sorted(np.unique(pliegue_de_fila).tolist())
    ]
    np.savez_compressed(
        OOF_V03_FINAL_PATH,
        ids=ids,
        y_true=y_verdadera,
        pliegues=pliegue_de_fila,
        proba=proba_v03,
        pred=prediccion,
        margen=margen,
        entropia=entropia,
        confianza=confianza,
        umbrales=umbrales,
        etiquetas=np.array(ETIQUETAS),
        corrected_cv=np.array([metrica["f1_media_columnas"]]),
        f1_por_etiqueta=np.array([metrica["f1_por_etiqueta"][etiqueta] for etiqueta in ETIQUETAS]),
        f1_pliegues=np.array(f1_pliegues),
        fuente=np.array(
            [
                (
                    "features/v2_oof_base_corrected.npz, modelo 'M2_union_lr_c2' (fila 2 de 4; "
                    "mismos ids, grupos y pliegues corregidos v2)"
                )
            ]
        ),
        nota_mapeo=np.array(
            [
                (
                    "proba = v2_oof_base_corrected.npz['proba'][:, posicion_M2, :] reordenada por "
                    "su clave 'ids'; pred = proba >= [0.44, 0.51, 0.46] en el orden HS, TR, AG"
                )
            ]
        ),
    )
    log(
        f"OOF V03 guardado en {OOF_V03_FINAL_PATH.name}: corrected_cv "
        f"{metrica['f1_media_columnas']:.6f} (referencia {CONFIG['referencia_v03']:.4f}); "
        f"pliegues {[round(valor, 4) for valor in f1_pliegues]}"
    )
    return {
        "corrected_cv": metrica["f1_media_columnas"],
        "f1_por_etiqueta": dict(metrica["f1_por_etiqueta"]),
        "f1_pliegues": f1_pliegues,
        "diferencia_vs_referencia": diferencia,
    }


# ===== SECCION 3: matrices de los candidatos =====


def bloque_probs_simples(
    probas: Sequence[NDArray[np.float64]], nombres: Sequence[str]
) -> tuple[NDArray[np.float64], list[str]]:
    """Concatena probabilidades de varios canales como features del meta.

    Args:
        probas: secuencia de matrices ``(n, 3)`` o ``(n, k, 3)``.
        nombres: nombre de cada bloque.

    Returns:
        Tupla ``(matriz, nombres_columnas)`` con las probabilidades.

    Raises:
        ValueError: si el numero de bloques y de nombres no coincide.
    """
    if len(probas) != len(nombres):
        raise ValueError("Cada bloque necesita un nombre")
    bloques = [np.asarray(proba, dtype=float).reshape(proba.shape[0], -1) for proba in probas]
    columnas = [
        f"proba_{nombre}_{etiqueta}"
        for nombre, bloque in zip(nombres, bloques)
        for etiqueta in ETIQUETAS[: bloque.shape[1]]
    ]
    return np.hstack(bloques), columnas


def construir_matriz_candidato(
    candidato: str,
    familias: tuple[str, ...],
    fuentes: dict[str, Any],
    textos: Sequence[str],
    con_margen_v03: bool = False,
) -> tuple[NDArray[np.float64], list[str]]:
    """Construye la matriz de features de un candidato del screening.

    Args:
        candidato: nombre del candidato.
        familias: familias de canal incluidas (candidatos pequenos).
        fuentes: mapa con ``modulo17``, ``bases`` (``(n, 4, 3)``), ``v03``
            (``(n, 3)``), ``canales`` (familia -> ``(n, 3)``) y ``seleccion``.
        textos: textos alineados con las filas.
        con_margen_v03: si se anaden margen y entropia de las probs de V03.

    Returns:
        Tupla ``(matriz, nombres_columnas)``.

    Raises:
        ValueError: si el candidato no esta definido.
    """
    modulo17 = fuentes["modulo17"]
    if candidato == "C-REF-44":
        return modulo17.construir_bloque_meta(fuentes["bases"], None, (), textos, "bases")
    if candidato == "C-REF-44-CH":
        familias_referencia = CANDIDATOS_PEQUENOS["C-EMO-SYN-TGT-CYT"]
        canales = np.concatenate(
            [fuentes["canales"][familia] for familia in familias_referencia], axis=1
        )
        nombres = [fuentes["seleccion"][familia] for familia in familias_referencia]
        return modulo17.construir_bloque_meta(
            fuentes["bases"], canales, nombres, textos, "bases_canales"
        )
    if candidato not in CANDIDATOS_PEQUENOS:
        clave_base = candidato.replace("-margenes", "")
        if clave_base not in CANDIDATOS_PEQUENOS:
            raise ValueError(f"Candidato desconocido: {candidato}")
        candidato = clave_base
    bloques = [fuentes["v03"]] + [fuentes["canales"][familia] for familia in familias]
    nombres = [CLAVE_V03] + [fuentes["seleccion"][familia] for familia in familias]
    matriz, columnas = bloque_probs_simples(bloques, nombres)
    if con_margen_v03:
        v03 = fuentes["v03"]
        margen = np.abs(v03 - 0.5)
        recortada = np.clip(v03, 1e-6, 1 - 1e-6)
        entropia = -(recortada * np.log(recortada) + (1 - recortada) * np.log(1 - recortada))
        matriz = np.hstack([matriz, margen, entropia])
        columnas = columnas + [f"margen_v03_{etiqueta}" for etiqueta in ETIQUETAS] + [
            f"entropia_v03_{etiqueta}" for etiqueta in ETIQUETAS
        ]
    return matriz, columnas


# ===== SECCION 4: evaluacion de los candidatos en el desarrollo =====


def evaluar_candidatos(
    desarrollo: pd.DataFrame,
    pliegues: list[tuple[NDArray[np.int_], NDArray[np.int_]]],
    y_verdadera: NDArray[np.int_],
    fuentes: dict[str, Any],
    candidatos: dict[str, tuple[str, ...]],
    con_margen_v03: bool = False,
) -> tuple[dict[str, NDArray[np.float64]], dict[str, NDArray[np.float64]], list[Any]]:
    """Evalua candidatos fold a fold con el meta anidado de ``experiments/17``.

    Args:
        desarrollo: DataFrame del desarrollo.
        pliegues: pliegues corregidos.
        y_verdadera: etiquetas verdaderas.
        fuentes: fuentes de features.
        candidatos: mapa candidato -> familias.
        con_margen_v03: si la matriz de cada candidato incluye los +6 features.

    Returns:
        Tupla ``(probas, matrices, especificaciones)``.
    """
    modulo17 = fuentes["modulo17"]
    matrices: dict[str, NDArray[np.float64]] = {}
    for clave, familias in candidatos.items():
        matriz, _ = construir_matriz_candidato(
            clave, familias, fuentes, desarrollo["text"], con_margen_v03
        )
        matrices[clave] = matriz
        log(f"candidato {clave}: {matriz.shape[1]} features")
    especificaciones: list[Any] = []
    for clave in candidatos:
        especificaciones.extend(
            modulo17.expandir_especificaciones(
                [
                    (clave, f"cand:{clave}", "ridge", CONFIG["ridge_alpha"]),
                    (clave, f"cand:{clave}", "lr", CONFIG["lr_C"]),
                ]
            )
        )
    probas: dict[str, NDArray[np.float64]] = {}
    for clave, matriz in matrices.items():
        por_pliegue = {
            numero: (matriz[idx_entrenamiento], matriz[idx_validacion])
            for numero, (idx_entrenamiento, idx_validacion) in enumerate(pliegues, start=1)
        }
        especificaciones_candidato = [espec for espec in especificaciones if espec.variante == clave]
        probas.update(
            modulo17.evaluar_meta_por_pliegue(
                por_pliegue, pliegues, y_verdadera, especificaciones_candidato
            )
        )
    return probas, matrices, especificaciones


def complementariedad_v03(
    modulo17: Any,
    proba: NDArray[np.float64],
    proba_v03: NDArray[np.float64],
    y_verdadera: NDArray[np.int_],
) -> tuple[float, dict[str, Any]]:
    """Mide la complementariedad de un candidato frente a V03 (convencion P6).

    Args:
        modulo17: modulo de ``experiments/17``.
        proba: probabilidades del candidato.
        proba_v03: probabilidades de V03 (M2).
        y_verdadera: etiquetas verdaderas.

    Returns:
        Tupla ``(ganancia, bootstrap)`` con la ganancia de la fusion a 0.5 frente
        a M2 a 0.5 y el IC bootstrap pareado del delta.
    """
    sin_umbral = {etiqueta: 0.5 for etiqueta in ETIQUETAS}
    fusion = (proba + proba_v03) / 2.0
    f1_fusion = modulo17.f1_media_rapida(y_verdadera, (fusion >= 0.5).astype(int))
    ganancia = f1_fusion - CONFIG["referencia_m2_umbral_05"]
    bootstrap = modulo17.bootstrap_delta(proba_v03, fusion, y_verdadera, sin_umbral, sin_umbral)
    return ganancia, bootstrap


def resumir_candidato(
    clave: str,
    probas: dict[str, NDArray[np.float64]],
    especificaciones: Sequence[Any],
    y_verdadera: NDArray[np.int_],
    pliegue_de_fila: NDArray[np.int_],
    proba_v03: NDArray[np.float64],
    modulo17: Any,
) -> dict[str, Any]:
    """Resume un candidato: mejor hiperparametro, F1, pliegues y complementariedad.

    Args:
        clave: nombre del candidato.
        probas: probabilidades OOF por especificacion.
        especificaciones: especificaciones del candidato.
        y_verdadera: etiquetas verdaderas.
        pliegue_de_fila: pliegue corregido de cada fila.
        proba_v03: probabilidades OOF de V03.
        modulo17: modulo de ``experiments/17``.

    Returns:
        Diccionario con el resumen completo del candidato.
    """
    sin_umbral = {etiqueta: 0.5 for etiqueta in ETIQUETAS}
    umbrales_v03 = CONFIG["umbrales_v03"]
    prediccion_v03 = (proba_v03 >= modulo17.vector_umbrales(umbrales_v03)).astype(int)
    f1_v03_pliegues = modulo17.f1_por_pliegue(proba_v03, y_verdadera, pliegue_de_fila, umbrales_v03)
    por_parametro = {
        f"{espec.modelo}:{espec.parametro:g}": modulo17.f1_media_rapida(
            y_verdadera, (probas[espec.key] >= 0.5).astype(int)
        )
        for espec in especificaciones
        if espec.variante == clave
    }
    mejor = max(por_parametro, key=lambda nombre: por_parametro[nombre])
    mejor_espec = next(
        espec
        for espec in especificaciones
        if espec.variante == clave and f"{espec.modelo}:{espec.parametro:g}" == mejor
    )
    proba = probas[mejor_espec.key]
    prediccion = (proba >= CONFIG["umbral_meta"]).astype(int)
    metrica = metricas_multietiqueta(y_verdadera, prediccion)
    f1_pliegues = modulo17.f1_por_pliegue(proba, y_verdadera, pliegue_de_fila, sin_umbral)
    desacuerdo = float(np.mean(np.any(prediccion != prediccion_v03, axis=1)))
    desacuerdo_por_etiqueta = {
        etiqueta: float(np.mean(prediccion[:, posicion] != prediccion_v03[:, posicion]))
        for posicion, etiqueta in enumerate(ETIQUETAS)
    }
    ganancia, bootstrap_ensemble = complementariedad_v03(modulo17, proba, proba_v03, y_verdadera)
    bootstrap_v03 = modulo17.bootstrap_delta(proba_v03, proba, y_verdadera, umbrales_v03, sin_umbral)
    delta_v03 = metrica["f1_media_columnas"] - CONFIG["referencia_v03"]
    delta_meta = metrica["f1_media_columnas"] - CONFIG["referencia_meta_v08"]
    pliegues_mejorados = int(np.sum(np.array(f1_pliegues) > np.array(f1_v03_pliegues)))
    cumple_nivel = bool(delta_v03 >= CONFIG["mejora_minima"])
    cumple_pliegues = bool(pliegues_mejorados >= CONFIG["pliegues_minimos_mejorados"])
    cumple_complementariedad = bool(
        ganancia >= CONFIG["ensemble_gain_minimo"] and bootstrap_ensemble["ic_inferior"] > 0
    )
    if cumple_nivel and cumple_pliegues:
        decision = "PROMISING"
    elif cumple_complementariedad:
        decision = "COMPLEMENTARY"
    else:
        decision = "no adoptado"
    return {
        "candidato": clave,
        "mejor_configuracion": mejor,
        "modelo": mejor_espec.modelo,
        "parametro": mejor_espec.parametro,
        "por_parametro": por_parametro,
        "corrected_cv": metrica["f1_media_columnas"],
        "f1_por_etiqueta": dict(metrica["f1_por_etiqueta"]),
        "f1_pliegues": f1_pliegues,
        "std_pliegues": float(np.std(f1_pliegues)),
        "f1_v03_pliegues": f1_v03_pliegues,
        "pliegues_mejorados_vs_v03": pliegues_mejorados,
        "delta_vs_v03": delta_v03,
        "delta_vs_meta_v08": delta_meta,
        "desacuerdo_vs_v03": desacuerdo,
        "desacuerdo_por_etiqueta": desacuerdo_por_etiqueta,
        "ensemble_gain": ganancia,
        "ensemble_corrected_cv": modulo17.f1_media_rapida(
            y_verdadera, (((proba + proba_v03) / 2.0) >= 0.5).astype(int)
        ),
        "ensemble_ic_inferior": bootstrap_ensemble["ic_inferior"],
        "ensemble_ic_superior": bootstrap_ensemble["ic_superior"],
        "bootstrap_vs_v03": bootstrap_v03,
        "criterio_nivel": {"objetivo": CONFIG["referencia_v03"] + CONFIG["mejora_minima"], "cumple": cumple_nivel},
        "criterio_pliegues": {"objetivo": CONFIG["pliegues_minimos_mejorados"], "cumple": cumple_pliegues},
        "criterio_complementariedad": {"cumple": cumple_complementariedad},
        "decision": decision,
    }


# ===== SECCION 5: evaluacion del holdout congelado =====


def anotar_holdout(holdout: pd.DataFrame, anotador: Any) -> list[Any]:
    """Anota los textos del holdout sin escribir en las caches de la fase v2.

    Args:
        holdout: DataFrame del holdout.
        anotador: funcion de anotacion (``anotar_textos`` o ``anotar_con_spacy``).

    Returns:
        Lista de documentos del holdout.

    Raises:
        RuntimeError: si el numero de documentos no coincide con el holdout.
    """
    resultado = anotador(holdout["text"].tolist())
    documentos = resultado[0] if isinstance(resultado, tuple) else resultado
    if len(documentos) != len(holdout):
        raise RuntimeError("La anotacion del holdout no coincide en tamano")
    log(f"anotacion del holdout completada: {len(documentos)} documentos (caches intactas)")
    return list(documentos)


def refit_bases_holdout(
    desarrollo: pd.DataFrame,
    holdout: pd.DataFrame,
    y_desarrollo: NDArray[np.int_],
    modulo14: Any,
    modulo17: Any,
) -> tuple[NDArray[np.float64], dict[str, Any]]:
    """Reajusta los 4 bases con el desarrollo y predice el holdout (patron P7).

    Args:
        desarrollo: DataFrame del desarrollo.
        holdout: DataFrame del holdout.
        y_desarrollo: etiquetas del desarrollo.
        modulo14: modulo de ``experiments/14`` con la receta de los bases.
        modulo17: modulo de ``experiments/17`` con ``ajustar_y_predecir``.

    Returns:
        Tupla ``(bases, info)`` con las probabilidades ``(n_hold, 4, 3)`` de los
        4 bases (mismo orden que ``modulo14.MODELOS``).
    """
    nombres = tuple(modulo14.MODELOS)
    if nombres != ("M1_word_lr", "M2_union_lr_c2", "M3_union_svc", "M4_char_lr"):
        raise RuntimeError(f"Los modelos base de experiments/14 no son los esperados: {nombres}")
    matrices = {
        representacion: modulo14.construir_matrices(
            desarrollo["text"], holdout["text"], representacion
        )
        for representacion in ("word", "char", "union")
    }
    bases = np.zeros((len(holdout), len(nombres), len(ETIQUETAS)))
    for posicion, (_, especificacion) in enumerate(modulo14.MODELOS.items()):
        matriz_entrenamiento, matriz_holdout = matrices[especificacion["representacion"]]
        for indice_etiqueta, _ in enumerate(ETIQUETAS):
            bases[:, posicion, indice_etiqueta] = modulo17.ajustar_y_predecir(
                especificacion,
                matriz_entrenamiento,
                y_desarrollo[:, indice_etiqueta],
                matriz_holdout,
            )
    info = {
        "dimensiones": {
            representacion: int(matriz[0].shape[1]) for representacion, matriz in matrices.items()
        },
        "modelos": list(nombres),
    }
    return bases, info


def refit_canales_holdout(
    desarrollo: pd.DataFrame,
    holdout: pd.DataFrame,
    y_desarrollo: NDArray[np.int_],
    y_holdout: NDArray[np.int_],
    seleccion: dict[str, str],
    familias_necesarias: set[str],
    constantes: dict[str, Any],
    modulo14: Any,
    modulo16: Any,
    modulo19: Any,
    modulo20: Any,
) -> tuple[dict[str, NDArray[np.float64]], dict[str, Any]]:
    """Reajusta los canales necesarios con el desarrollo y predice el holdout.

    Reutiliza las recetas congeladas de los experimentos 16, 19 y 20 con un
    pseudo-pliegue desarrollo -> holdout: lexicos, vectorizadores y modelos se
    ajustan solo con el desarrollo. Cada canal se recalcula unicamente si algun
    finalista lo necesita.

    Args:
        desarrollo: DataFrame del desarrollo.
        holdout: DataFrame del holdout.
        y_desarrollo: etiquetas del desarrollo.
        y_holdout: etiquetas del holdout.
        seleccion: canal puro seleccionado por familia.
        familias_necesarias: familias exigidas por los finalistas.
        constantes: constantes estructurales del script 06.
        modulo14: modulo de ``experiments/14``.
        modulo16: modulo de ``experiments/16``.
        modulo19: modulo de ``experiments/19``.
        modulo20: modulo de ``experiments/20``.

    Returns:
        Tupla ``(canales, info)`` con las probabilidades de cada canal en el
        holdout y el diagnostico de las recetas usadas.

    Raises:
        ValueError: si se pide una familia desconocida.
    """
    desconocidas = familias_necesarias - set(STORES_CANALES)
    if desconocidas:
        raise ValueError(f"Familias desconocidas: {sorted(desconocidas)}")
    n_desarrollo = len(desarrollo)
    combinado = pd.concat([desarrollo, holdout], ignore_index=True)
    y_combinado = np.vstack([y_desarrollo, y_holdout])
    idx_desarrollo = np.arange(n_desarrollo)
    idx_holdout = np.arange(n_desarrollo, len(combinado))
    seudopliegue = [(idx_desarrollo, idx_holdout)]
    canales: dict[str, NDArray[np.float64]] = {}

    if familias_necesarias & {"emocion"}:
        docs16 = list(modulo16.cargar_o_crear_anotaciones(desarrollo)[0]) + anotar_holdout(
            holdout, modulo16.anotar_con_spacy
        )
        if "targeting" in familias_necesarias:
            resultado16 = modulo16.evaluar_pliegues(
                combinado, docs16, y_combinado, seudopliegue, constantes, modulo14
            )
            canales["targeting"] = resultado16.oof[(seleccion["targeting"], 2.0)][idx_holdout]
            log(f"canal {seleccion['targeting']} reajustado con el desarrollo y predicho en el holdout")
        resultado20 = modulo20.evaluar_seccion_emocion(
            combinado,
            docs16,
            y_combinado,
            seudopliegue,
            constantes,
            modulo20.cargar_p2(),
            modulo20.cargar_receta_v03(),
        )[0]
        clave_emocion = clave_emocion_evaluada(seleccion["emocion"])
        canales["emocion"] = resultado20[clave_emocion][idx_holdout]
        log(f"canal {clave_emocion} reajustado con el desarrollo y predicho en el holdout")

    if familias_necesarias & {"sintaxis", "cita_negacion"}:
        docs19 = list(modulo19.cargar_o_crear_anotaciones(desarrollo)[0]) + anotar_holdout(
            holdout, modulo19.anotar_textos
        )
        resultado19 = modulo19.evaluar_pliegues(
            combinado, docs19, y_combinado, seudopliegue, constantes, modulo14
        )
        if "sintaxis" in familias_necesarias:
            canales["sintaxis"] = resultado19.oof[(seleccion["sintaxis"], 1.0)][idx_holdout]
        if "cita_negacion" in familias_necesarias:
            canales["cita_negacion"] = resultado19.oof[(seleccion["cita_negacion"], 2.0)][idx_holdout]
        log(
            "canales "
            + ", ".join(sorted(familias_necesarias & {"sintaxis", "cita_negacion"}))
            + " reajustados con el desarrollo y predichos en el holdout"
        )
    return canales, {
        "canales": {familia: seleccion[familia] for familia in sorted(canales)},
        "familias_necesarias": sorted(familias_necesarias),
        "docs_holdout_anotados": len(holdout),
    }


def evaluar_candidato_holdout(
    candidato: str,
    familias: tuple[str, ...],
    datos_holdout: dict[str, Any],
    y_holdout: NDArray[np.int_],
    especificacion: Any,
    fuentes_dev: dict[str, Any],
    desarrollo: pd.DataFrame,
    y_desarrollo: NDArray[np.int_],
    con_margen_v03: bool = False,
) -> dict[str, Any]:
    """Ajusta el meta del candidato con el OOF del desarrollo y mide el holdout.

    Args:
        candidato: nombre del candidato (sin sufijo de variante).
        familias: familias de canal incluidas.
        datos_holdout: features del holdout reajustadas sobre el desarrollo.
        y_holdout: etiquetas del holdout.
        especificacion: especificacion congelada en el desarrollo.
        fuentes_dev: fuentes de features del desarrollo.
        desarrollo: DataFrame del desarrollo.
        y_desarrollo: etiquetas del desarrollo.
        con_margen_v03: si la matriz incluye margen y entropia de V03.

    Returns:
        Diccionario con las metricas del holdout y el bootstrap pareado vs V03.
    """
    modulo17 = fuentes_dev["modulo17"]
    matriz_holdout, _ = construir_matriz_candidato(
        candidato, familias, datos_holdout, datos_holdout["textos"], con_margen_v03
    )
    matriz_dev, _ = construir_matriz_candidato(
        candidato, familias, fuentes_dev, desarrollo["text"], con_margen_v03
    )
    escalador = StandardScaler().fit(matriz_dev)
    x_dev = escalador.transform(matriz_dev)
    x_holdout = escalador.transform(matriz_holdout)
    proba_holdout = np.zeros((len(y_holdout), len(ETIQUETAS)))
    for posicion, _ in enumerate(ETIQUETAS):
        proba_holdout[:, posicion] = modulo17.ajustar_modelo_meta(
            especificacion, x_dev, y_desarrollo[:, posicion], x_holdout
        )
    prediccion = (proba_holdout >= CONFIG["umbral_meta"]).astype(int)
    metrica = metricas_multietiqueta(y_holdout, prediccion)
    bootstrap = modulo17.bootstrap_delta(
        datos_holdout["v03"],
        proba_holdout,
        y_holdout,
        CONFIG["umbrales_v03"],
        {etiqueta: CONFIG["umbral_meta"] for etiqueta in ETIQUETAS},
    )
    return {
        "candidato": candidato,
        "modelo": especificacion.modelo,
        "parametro": especificacion.parametro,
        "holdout_f1_media_columnas": metrica["f1_media_columnas"],
        "holdout_f1_por_etiqueta": dict(metrica["f1_por_etiqueta"]),
        "holdout_delta_vs_v03_pareado": bootstrap["media"],
        "holdout_bootstrap_ic_inferior": bootstrap["ic_inferior"],
        "holdout_bootstrap_ic_superior": bootstrap["ic_superior"],
        "holdout_fraccion_positiva": bootstrap["fraccion_positiva"],
    }


# ===== SECCION 6: registros y salidas =====


def registrar_holdout(filas: list[dict[str, str]]) -> int:
    """Registra evaluaciones nuevas del holdout verificando el maximo permitido.

    Args:
        filas: filas ``{fecha, experimento, motivo, resultado}`` a registrar.

    Returns:
        Numero de evaluaciones acumuladas tras el registro.

    Raises:
        RuntimeError: si se supera el maximo de evaluaciones permitido.
    """
    existente = pd.read_csv(HOLDOUT_EVALUACIONES_PATH)
    nuevos = pd.DataFrame(filas)
    nuevos = nuevos[~nuevos["experimento"].isin(existente["experimento"].astype(str))]
    combinado = pd.concat([existente, nuevos], ignore_index=True)
    if len(combinado) > CONFIG["max_evaluaciones_holdout"]:
        raise RuntimeError(
            f"El holdout superaria {CONFIG['max_evaluaciones_holdout']} evaluaciones: {len(combinado)}"
        )
    combinado.to_csv(HOLDOUT_EVALUACIONES_PATH, index=False)
    return len(combinado)


def escribir_final_candidates(
    resumenes: list[dict[str, Any]],
    evaluaciones_holdout: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    """Escribe ``reports/final_candidates.csv`` con los campos de la spec 42.

    Args:
        resumenes: resumenes de los candidatos evaluados en el desarrollo.
        evaluaciones_holdout: metricas del holdout por candidato.

    Returns:
        DataFrame escrito.
    """
    filas = []
    for resumen in resumenes:
        holdout = evaluaciones_holdout.get(resumen["candidato"])
        filas.append(
            {
                "experiment_id": resumen["experiment_id"],
                "hypothesis": resumen["hipotesis"],
                "channel": resumen["canal"],
                "representation": resumen["representacion"],
                "model": f"{resumen['modelo']} por etiqueta ({resumen['mejor_configuracion']})",
                "parameters": json.dumps(
                    {"mejor": resumen["mejor_configuracion"], "por_parametro": resumen["por_parametro"]},
                    ensure_ascii=False,
                ),
                "historical_cv": resumen["historical_cv"],
                "corrected_cv": round(resumen["corrected_cv"], 6),
                "fold_scores": json.dumps([round(valor, 6) for valor in resumen["f1_pliegues"]]),
                "holdout": (
                    f"{holdout['holdout_f1_media_columnas']:.4f} "
                    f"(por etiqueta {json.dumps({clave: round(valor, 4) for clave, valor in holdout['holdout_f1_por_etiqueta'].items()})})"
                    if holdout
                    else "no evaluado"
                ),
                "delta_vs_v03": round(resumen["delta_vs_v03"], 6),
                "complementarity": (
                    f"ensemble_gain {resumen['ensemble_gain']:+.4f} "
                    f"(IC [{resumen['ensemble_ic_inferior']:+.4f},{resumen['ensemble_ic_superior']:+.4f}]) "
                    f"vs M2 a 0.5 -> fusion {resumen['ensemble_corrected_cv']:.4f} "
                    f"(meta v08 {CONFIG['referencia_meta_v08']}); desacuerdo {resumen['desacuerdo_vs_v03']:.3f}"
                ),
                "decision": resumen["decision"],
                "submission_candidate": (
                    "si"
                    if resumen["decision"] == "PROMISING"
                    else (
                        "no (fusion por debajo del meta v08)"
                        if resumen["ensemble_corrected_cv"] < CONFIG["referencia_meta_v08"]
                        else "si (componente de fusion)"
                    )
                ),
            }
        )
    filas.append(
        {
            "experiment_id": "V03",
            "hypothesis": "Referencia congelada (modelo final actual, LB 0.78306)",
            "channel": "union word+char (M2)",
            "representation": "TF-IDF word(1,2)+char_wb(3,5) min_df=2 sublinear + LR balanced C=2",
            "model": "logistic regression por etiqueta con umbrales 0.44/0.51/0.46",
            "parameters": json.dumps({"C": 2.0, "umbrales": CONFIG["umbrales_v03"]}),
            "historical_cv": 0.7989,
            "corrected_cv": CONFIG["referencia_v03"],
            "fold_scores": "ver features/oof_v03_final.npz",
            "holdout": "0.8285 (HS 0.8571 | TR 0.8493 | AG 0.7789) [P7-01]",
            "delta_vs_v03": 0.0,
            "complementarity": "referencia",
            "decision": "referencia",
            "submission_candidate": "si (modelo final actual)",
        }
    )
    filas.append(
        {
            "experiment_id": "META-BASES-RIDGE",
            "hypothesis": "Meta de dependencia sobre los 4 bases (P6, v08): mejor candidato conocido",
            "channel": "4 bases OOF + margenes + entropias + 8 rasgos estructurales",
            "representation": "44 features densas estandarizadas en el pliegue",
            "model": "ridge por etiqueta (alpha=3)",
            "parameters": json.dumps({"alpha": 3.0}),
            "historical_cv": "no medido (protocolo v2)",
            "corrected_cv": CONFIG["referencia_meta_v08"],
            "fold_scores": json.dumps([0.803466, 0.793890, 0.785788, 0.767131, 0.787035]),
            "holdout": "0.8465 (HS 0.8624 | TR 0.8571 | AG 0.8200) [P7-02]",
            "delta_vs_v03": round(CONFIG["referencia_meta_v08"] - CONFIG["referencia_v03"], 6),
            "complementarity": "ensemble_gain +0.0089 (IC [+0.0019,+0.0160]); LB v08 0.78163 (no transferido, D-020)",
            "decision": "adoptado en P6; el LB no supera V03",
            "submission_candidate": "no (sonda v08 ya enviada)",
        }
    )
    tabla = pd.DataFrame(filas)
    tabla.to_csv(FINAL_CANDIDATES_PATH, index=False)
    return tabla


def registrar_experimentos(
    resumenes: list[dict[str, Any]], evaluaciones_holdout: dict[str, dict[str, Any]]
) -> None:
    """Anade las filas ``FIN-*`` al registro de experimentos.

    Args:
        resumenes: resumenes de los candidatos.
        evaluaciones_holdout: metricas del holdout por candidato.
    """
    existente = pd.read_csv(EXPERIMENTS_CSV)
    previas = existente[~existente["experiment_id"].astype(str).str.startswith("FIN-")]
    filas = []
    for resumen in resumenes:
        identificador = resumen["experiment_id"]
        holdout = evaluaciones_holdout.get(resumen["candidato"])
        resultado = {
            "corrected_cv": resumen["corrected_cv"],
            "f1_por_etiqueta": resumen["f1_por_etiqueta"],
            "pliegues": resumen["f1_pliegues"],
            "delta_vs_v03": resumen["delta_vs_v03"],
            "delta_vs_meta_v08": resumen["delta_vs_meta_v08"],
            "pliegues_mejorados_vs_v03": resumen["pliegues_mejorados_vs_v03"],
            "ensemble_gain": resumen["ensemble_gain"],
            "desacuerdo_vs_v03": resumen["desacuerdo_vs_v03"],
        }
        if holdout:
            resultado["holdout_f1_media_columnas"] = holdout["holdout_f1_media_columnas"]
            resultado["holdout_por_etiqueta"] = holdout["holdout_f1_por_etiqueta"]
            resultado["holdout_delta_pareado"] = holdout["holdout_delta_vs_v03_pareado"]
        filas.append(
            {
                "experiment_id": identificador,
                "competition": "KAGGLE_TOXIC_2026_PUJ",
                "question": (
                    f"Supera {resumen['candidato']} a V03 (0.7772) y al meta v08 (0.7874) "
                    "en el protocolo corregido v2?"
                ),
                "hypothesis": resumen["hipotesis"],
                "preprocessing": (
                    "sin texto: probabilidades OOF de V03 y de los canales, meta ajustado fold a fold; "
                    "sin concatenar features densas al TF-IDF"
                ),
                "feature_groups": resumen["feature_groups"],
                "representation": resumen["representacion"],
                "model": f"{resumen['modelo']} por etiqueta ({resumen['mejor_configuracion']})",
                "hyperparameters": json.dumps(
                    {"mejor": resumen["mejor_configuracion"], "por_parametro": resumen["por_parametro"]},
                    ensure_ascii=False,
                ),
                "validation": (
                    "5 pliegues corregidos v2 con el meta ajustado fold a fold; holdout congelado "
                    "solo para el top-2 nuevo"
                ),
                "f1": round(resumen["corrected_cv"], 6),
                "training_time": resumen["tiempo_seg"],
                "dimension": resumen["n_features"],
                "result": json.dumps(resultado, ensure_ascii=False),
                "interpretation": resumen["interpretacion"],
                "decision": resumen["decision"],
                "date": CONFIG["fecha"],
                "script": "experiments/23_candidatos_finales.py",
                "notes": (
                    "historical_cv no aplica para metas (protocolo v2); "
                    f"holdout congelado: {'evaluado' if holdout else 'no evaluado'}; "
                    f"referencias V03 {CONFIG['referencia_v03']} y v08 {CONFIG['referencia_meta_v08']}"
                ),
            }
        )
    if not filas:
        log("filas FIN-* ya presentes: no se reescribe experiments.csv")
        return
    combinado = pd.concat([previas, pd.DataFrame(filas)], ignore_index=True)
    if combinado["experiment_id"].duplicated().any():
        raise RuntimeError("Hay experiment_id duplicados en experiments.csv")
    combinado.to_csv(EXPERIMENTS_CSV, index=False)
    log(f"experiments.csv actualizado con {len(filas)} filas FIN-* ({len(combinado)} totales)")


def preparar_resumenes(resumenes: list[dict[str, Any]], inicio: float) -> list[dict[str, Any]]:
    """Completa los metadatos editoriales de cada resumen y asigna los ids FIN.

    Args:
        resumenes: resumenes crudos del desarrollo.
        inicio: marca de tiempo del inicio de la ejecucion.

    Returns:
        Lista de resumenes ordenada por corrected_cv con ``experiment_id``.
    """
    inventario = {
        "C-REF-44": {
            "canal": "4 bases (M1-M4) + margenes + entropias + 8 rasgos estructurales",
            "representacion": "44 features densas del meta de P6 (probs + margenes + entropias + 8 estructurales)",
            "feature_groups": "META_SCORES+META_UNCERTAINTY+STRUCTURAL",
            "historical_cv": "no medido (protocolo v2)",
            "hipotesis": "Control: el meta adoptado en P6 (44 columnas) se reproduce con este pipeline",
            "interpretacion": "Control de comparabilidad: si no reproduce 0.7874, los demas candidatos no son comparables",
        },
        "C-REF-44-CH": {
            "canal": "4 bases + emocion + sintaxis + targeting + cita_negacion",
            "representacion": (
                "44 features del meta de P6 + probabilidades, margen y entropia de los 4 canales puros "
                "(80 columnas)"
            ),
            "feature_groups": "META_SCORES+META_UNCERTAINTY+STRUCTURAL+CHANNEL_PROBS",
            "historical_cv": "no medido (protocolo v2)",
            "hipotesis": (
                "Los canales v2 aportan al meta adoptado de 44 columnas "
                "(P6 solo habia probado los canales de targeting)"
            ),
            "interpretacion": "Test directo de si los canales nuevos aportan al mejor meta conocido",
        },
    }
    for resumen in resumenes:
        clave = resumen["candidato"]
        if clave in inventario:
            resumen.update(inventario[clave])
        else:
            familias = CANDIDATOS_PEQUENOS.get(clave.replace("-margenes", ""), ())
            resumen["representacion"] = (
                "probabilidades OOF de V03 + " + ", ".join(familias) + " (sin features densas)"
                if familias
                else "probabilidades OOF de V03"
            ) + (" + margen y entropia de V03" if clave.endswith("-margenes") else "")
            resumen["feature_groups"] = "META_PROBS_V03+CHANNEL_PROBS" + (
                "+V03_UNCERTAINTY" if clave.endswith("-margenes") else ""
            )
            resumen["historical_cv"] = "no medido (protocolo v2)"
            resumen["canal"] = ", ".join(familias) if familias else "v03 (M2)"
            resumen["hipotesis"] = (
                "La fusion tardia de las probabilidades de V03 con "
                f"{', '.join(familias) if familias else 'ningun canal'} mejora la referencia V03 (0.7772) "
                "con >= 3/5 pliegues o demuestra complementariedad con IC bootstrap > 0"
            )
            resumen["interpretacion"] = (
                f"Meta pequeno sobre probabilidades; corrected_cv {resumen['corrected_cv']:.4f} frente a "
                f"V03 {CONFIG['referencia_v03']} y al meta v08 {CONFIG['referencia_meta_v08']}; "
                f"{resumen['pliegues_mejorados_vs_v03']}/5 pliegues mejorados que V03"
            )
        resumen["tiempo_seg"] = round(time.perf_counter() - inicio, 2)
    resumenes.sort(key=lambda fila: fila["corrected_cv"], reverse=True)
    for posicion, resumen in enumerate(resumenes, start=1):
        resumen["experiment_id"] = f"FIN-{posicion:02d}-{resumen['candidato']}"
    return resumenes


def main() -> None:
    """Ejecuta el screening de candidatos finales y guarda los artefactos.

    El flag ``--sin-holdout`` evalua solo el desarrollo (util para revisar antes
    de gastar evaluaciones del holdout); el registro del holdout es acumulativo
    y las filas ``FIN-*`` se regeneran en cada ejecucion.
    """
    if "--sin-holdout" in sys.argv:
        CONFIG["evaluaciones_nuevas_holdout"] = 0
        log("modo --sin-holdout: no se evalua el holdout congelado")
    inicio = time.perf_counter()
    modulo17 = cargar_modulo("experimento_17_fin", RUTA_P17)
    modulo14 = cargar_modulo("experimento_14_fin", RUTA_P14)
    modulo16 = cargar_modulo("experimento_16_fin", RUTA_P16)
    modulo19 = cargar_modulo("experimento_19_fin", RUTA_P19)
    modulo20 = cargar_modulo("experimento_20_fin", RUTA_P20)

    desarrollo, grupos, pliegues = modulo17.cargar_desarrollo()
    y_verdadera = desarrollo[list(ETIQUETAS)].to_numpy()
    pliegue_de_fila = np.zeros(len(desarrollo), dtype=int)
    for numero, (_, idx_validacion) in enumerate(pliegues, start=1):
        pliegue_de_fila[idx_validacion] = numero
    modulo17.verificar_datos(desarrollo, grupos, pliegue_de_fila)
    referencias = modulo17.cargar_referencias()
    proba_bases, nombres_bases, _, _, _ = modulo17.cargar_oof(
        desarrollo, grupos, y_verdadera, pliegue_de_fila
    )
    proba_v03 = proba_bases[:, nombres_bases.index("M2_union_lr_c2"), :]
    verificacion_v03 = guardar_oof_v03_final(
        proba_v03, y_verdadera, pliegue_de_fila, desarrollo["id"].astype(int).to_numpy(), referencias
    )

    probas_canales: dict[str, NDArray[np.float64]] = {}
    nombres_canales: dict[str, tuple[str, ...]] = {}
    for familia, ruta in STORES_CANALES.items():
        probas_canales[familia], nombres_canales[familia] = cargar_store_canales(
            ruta, desarrollo, pliegue_de_fila, y_verdadera
        )
        log(f"store {familia}: {len(nombres_canales[familia])} variantes cargadas y verificadas")
    tabla_ranking, seleccion = ranking_canales(probas_canales, nombres_canales, y_verdadera)
    tabla_ranking.to_csv(RANKING_CANALES_PATH, index=False)
    log(f"ranking de canales guardado en {RANKING_CANALES_PATH.name} ({len(tabla_ranking)} filas)")

    fuentes_dev: dict[str, Any] = {
        "modulo17": modulo17,
        "bases": proba_bases,
        "v03": proba_v03,
        "canales": {
            familia: probas_canales[familia][:, nombres_canales[familia].index(nombre), :]
            for familia, nombre in seleccion.items()
        },
        "seleccion": seleccion,
    }

    candidatos: dict[str, tuple[str, ...]] = dict(CANDIDATOS_PEQUENOS)
    for clave in CANDIDATOS_REFERENCIA:
        candidatos[clave] = ()
    probas, matrices, especificaciones = evaluar_candidatos(
        desarrollo, pliegues, y_verdadera, fuentes_dev, candidatos
    )
    resumenes = [
        resumir_candidato(
            clave,
            probas,
            [espec for espec in especificaciones if espec.variante == clave],
            y_verdadera,
            pliegue_de_fila,
            proba_v03,
            modulo17,
        )
        for clave in candidatos
    ]
    for resumen in resumenes:
        resumen["n_features"] = matrices[resumen["candidato"]].shape[1]
        log(
            f"{resumen['candidato']}: corrected_cv {resumen['corrected_cv']:.4f} "
            f"(mejor {resumen['mejor_configuracion']}; delta vs V03 {resumen['delta_vs_v03']:+.4f}; "
            f"pliegues {resumen['pliegues_mejorados_vs_v03']}/5; {resumen['decision']})"
        )

    mejor_pequeno = max(
        (resumen for resumen in resumenes if resumen["candidato"] in CANDIDATOS_PEQUENOS),
        key=lambda fila: fila["corrected_cv"],
    )
    clave_margenes = f"{mejor_pequeno['candidato']}-margenes"
    probas_extra, matrices_extra, especificaciones_extra = evaluar_candidatos(
        desarrollo,
        pliegues,
        y_verdadera,
        fuentes_dev,
        {clave_margenes: CANDIDATOS_PEQUENOS[mejor_pequeno["candidato"]]},
        con_margen_v03=True,
    )
    resumen_extra = resumir_candidato(
        clave_margenes,
        probas_extra,
        especificaciones_extra,
        y_verdadera,
        pliegue_de_fila,
        proba_v03,
        modulo17,
    )
    resumen_extra["n_features"] = matrices_extra[clave_margenes].shape[1]
    resumenes.append(resumen_extra)
    log(
        f"{clave_margenes}: corrected_cv {resumen_extra['corrected_cv']:.4f} "
        f"(delta vs V03 {resumen_extra['delta_vs_v03']:+.4f}; "
        f"pliegues {resumen_extra['pliegues_mejorados_vs_v03']}/5; {resumen_extra['decision']})"
    )

    resumenes = preparar_resumenes(resumenes, inicio)
    todas_especificaciones = list(especificaciones) + list(especificaciones_extra)

    def es_reproduccion_p7(resumen: dict[str, Any]) -> bool:
        """Indica si el candidato coincide con la configuracion ya medida en P7-02.

        Args:
            resumen: resumen del candidato.

        Returns:
            ``True`` si no debe consumir una evaluacion nueva del holdout.
        """
        return bool(
            resumen["candidato"] == "C-REF-44"
            and resumen["modelo"] == "ridge"
            and abs(resumen["parametro"] - 3.0) < 1e-9
        )

    finalistas: list[dict[str, Any]] = []
    canales_vistos: set[str] = set()
    for resumen in resumenes:
        if es_reproduccion_p7(resumen):
            continue
        if resumen["canal"] in canales_vistos:
            continue
        canales_vistos.add(resumen["canal"])
        finalistas.append(resumen)
        if len(finalistas) == CONFIG["evaluaciones_nuevas_holdout"]:
            break
    log(
        "finalistas para el holdout: "
        + ", ".join(f"{resumen['candidato']} ({resumen['corrected_cv']:.4f})" for resumen in finalistas)
    )

    evaluaciones_holdout: dict[str, dict[str, Any]] = {}
    filas_holdout: list[dict[str, str]] = []
    if finalistas:
        holdout = cargar_holdout()
        y_holdout = holdout[list(ETIQUETAS)].to_numpy()
        bases_holdout, info_bases = refit_bases_holdout(
            desarrollo, holdout, y_verdadera, modulo14, modulo17
        )
        log(f"bases reajustados con el desarrollo: {info_bases}")
        prediccion_v03 = (
            bases_holdout[:, 1, :] >= modulo17.vector_umbrales(CONFIG["umbrales_v03"])
        ).astype(int)
        metrica_v03 = metricas_multietiqueta(y_holdout, prediccion_v03)
        if abs(metrica_v03["f1_media_columnas"] - CONFIG["holdout_v03_registrado"]) > CONFIG["tolerancia_holdout_v03"]:
            log(
                f"AVISO: la verificacion de V03 en el holdout da {metrica_v03['f1_media_columnas']:.4f} "
                f"frente a {CONFIG['holdout_v03_registrado']} registrado (P7-01)"
            )
        else:
            log(
                f"verificacion del protocolo: V03 en el holdout {metrica_v03['f1_media_columnas']:.4f} "
                f"(P7-01 registrado {CONFIG['holdout_v03_registrado']}); no se registra como evaluacion nueva"
            )
        constantes, origen_constantes = modulo16.cargar_constantes_error_analysis()
        familias_necesarias = {
            familia
            for resumen in finalistas
            for familia in CANDIDATOS_PEQUENOS.get(resumen["candidato"].replace("-margenes", ""), ())
        }
        canales_holdout, info_canales = refit_canales_holdout(
            desarrollo,
            holdout,
            y_verdadera,
            y_holdout,
            seleccion,
            familias_necesarias,
            constantes,
            modulo14,
            modulo16,
            modulo19,
            modulo20,
        )
        log(f"canales reajustados: {info_canales}; constantes: {origen_constantes}")
        datos_holdout: dict[str, Any] = {
            "modulo17": modulo17,
            "bases": bases_holdout,
            "v03": bases_holdout[:, 1, :],
            "canales": canales_holdout,
            "seleccion": seleccion,
            "textos": holdout["text"],
        }
        for resumen in finalistas:
            clave_base = resumen["candidato"].replace("-margenes", "")
            especificacion = next(
                espec
                for espec in todas_especificaciones
                if espec.variante == resumen["candidato"]
                and espec.modelo == resumen["modelo"]
                and abs(espec.parametro - resumen["parametro"]) < 1e-9
            )
            evaluacion = evaluar_candidato_holdout(
                clave_base,
                CANDIDATOS_PEQUENOS.get(clave_base, ()),
                datos_holdout,
                y_holdout,
                especificacion,
                fuentes_dev,
                desarrollo,
                y_verdadera,
                resumen["candidato"].endswith("-margenes"),
            )
            evaluacion["candidato"] = resumen["candidato"]
            evaluacion["corrected_cv_dev"] = resumen["corrected_cv"]
            evaluaciones_holdout[resumen["candidato"]] = evaluacion
            log(
                f"holdout {resumen['candidato']}: {evaluacion['holdout_f1_media_columnas']:.4f} "
                f"(delta pareado vs V03 {evaluacion['holdout_delta_vs_v03_pareado']:+.4f}, IC "
                f"[{evaluacion['holdout_bootstrap_ic_inferior']:+.4f},"
                f"{evaluacion['holdout_bootstrap_ic_superior']:+.4f}])"
            )
            filas_holdout.append(
                {
                    "fecha": CONFIG["fecha"],
                    "experimento": f"{resumen['experiment_id']} (corrected_cv {resumen['corrected_cv']:.4f})",
                    "motivo": (
                        "Candidato nuevo de la fase final: meta sobre probabilidades OOF de V03 + canales "
                        f"({', '.join(CANDIDATOS_PEQUENOS.get(clave_base, ())) or 'referencia'}), "
                        f"{resumen['modelo']} {resumen['parametro']:g}, elegido en el desarrollo"
                    ),
                    "resultado": json.dumps(
                        {
                            "holdout_f1_media_columnas": evaluacion["holdout_f1_media_columnas"],
                            "holdout_f1_por_etiqueta": evaluacion["holdout_f1_por_etiqueta"],
                            "delta_pareado_vs_v03": evaluacion["holdout_delta_vs_v03_pareado"],
                            "ic95": [
                                evaluacion["holdout_bootstrap_ic_inferior"],
                                evaluacion["holdout_bootstrap_ic_superior"],
                            ],
                        },
                        ensure_ascii=False,
                    ),
                }
            )
        total = registrar_holdout(filas_holdout)
        log(f"evaluaciones del holdout registradas; total acumulado {total}/{CONFIG['max_evaluaciones_holdout']}")

    escribir_final_candidates(resumenes, evaluaciones_holdout)
    registrar_experimentos(resumenes, evaluaciones_holdout)
    log(f"final_candidates.csv escrito con {len(resumenes) + 2} filas")
    log(f"duracion total {time.perf_counter() - inicio:.1f} s")
    log(
        f"verificacion V03: corrected_cv {verificacion_v03['corrected_cv']:.6f} "
        f"(dif {verificacion_v03['diferencia_vs_referencia']:.2e})"
    )


if __name__ == "__main__":
    main()
