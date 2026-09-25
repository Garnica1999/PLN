"""P9 - Artefactos de auditoria v2 exigidos por la spec (FASES 43/44/47/50/51/55/88/89/90).

Pregunta que responde
---------------------
La spec ``docs/advanced_research_v2_spec.md`` exige un conjunto de artefactos de
auditoria que el proyecto no habia materializado: la interseccion de errores
(FASES 43/44), el error por combinacion de etiquetas (FASE 47), la
complementariedad entre canales OOF (FASES 50/51/91), el catalogo de features
(FASES 55/88), el dataset consolidado de features estructuradas (FASES 55/89) y
el registro de las tres auditorias en el log de experimentos (FASE 90). Este
script los genera TODOS desde la evidencia OOF ya guardada, sin entrenar ningun
modelo, sin generar submissions y sin tocar el holdout congelado ni el test.

Metodo
------
1. Se reproduce el protocolo de ``experiments/17_stacking_v2.py``: desarrollo
   (n=1483) de la particion congelada ``data/processed/holdout_v2_ids.csv``,
   pliegues corregidos v2 (D-016) y alineacion por ``id`` de los stores OOF
   (los stores no comparten el orden de filas del desarrollo).
2. Referencias auditadas: ``v03_corrected`` (M2_union_lr_c2 con los umbrales
   congelados 0.44/0.51/0.46, F1 0.7772) y ``meta_v2`` (META-BASES-RIDGE a 0.5,
   F1 0.7874, adoptado por complementariedad en P6). Ambas se replican en el
   script con tolerancia 1e-3 antes de usarse.
3. Senales estructurales: regex fijas (longitud, URL, mencion, hashtag,
   comillas, colectivo, segunda persona, risas, mayusculas, elongacion,
   repeticion, puntuacion interna, leetspeak) y anotacion spaCy simbolica
   cacheada (POS, dependencias, morfologia, cita, negacion). Ninguna usa
   etiquetas ni escribe texto crudo en las salidas. La unica senal lexica es el
   ``n_insultos_foldlocal``: el lexico de insulto se induce con los 4 pliegues
   de entrenamiento de cada pliegue y se aplica al pliegue evaluado (fold-safe,
   sin ver la validacion). El catalogo de features de FASE 55/88 se lee del
   codigo de ``experiments/14`` a ``experiments/21`` (nombres, formulas y
   dependencia de pliegue); no se inventa ninguna feature.
4. Honestidad: si una cifra no cuadra con lo documentado se registra la
   discrepancia en el log y en el manifest, nunca se "arregla". La referencia
   historica de FASE 43 (101 tweets, 5.8 % sobre train completo n=1731 con el
   protocolo historico) se contrasta explicitamente; el desarrollo corregido
   (n=1483) se reporta por separado y no se compara como si fuera el mismo
   protocolo.
5. Sin GPU (todo es CPU), sin entrenamiento, sin umbrales nuevos: los umbrales
   de cada canal son los ya documentados (V03 para la referencia; 0.5 para el
   resto de canales OOF segun sus reportes). El ``ensemble_gain`` de un par solo
   se copia si ya estaba medido en un reporte; no se recalcula ningun ensemble.

Salidas
-------
``reports/error_intersection_v2.csv``, ``reports/all_three_errors.csv``,
``reports/error_by_label_combination_v2.csv``,
``reports/model_complementarity_v2.csv``, ``reports/feature_catalog_v2.csv``,
``features/v2_structured_features.parquet``,
``reports/v2_structured_features_manifest.md``, tres filas ``AUD-01`` a
``AUD-03`` en ``experiments.csv`` (solo append) y la seccion
``## Artefactos de auditoria v2 (2026-09-25)`` (maximo 15 lineas) en
``reports/contexto.md``. La cache de anotacion de las filas fuera del desarrollo
se guarda en ``features/v2_audit_annotations_extra.json.gz`` (determinista).

Ejecucion
---------
    /home/carlos/venv/bin/python experiments/22_artefactos_auditoria_v2.py
"""

from __future__ import annotations

import csv as csv_module
import gzip
import importlib.util
import json
import re
import sys
import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd
from numpy.typing import NDArray

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src import data as data_module
from src.evaluation import ETIQUETAS, metricas_multietiqueta
from src.splits import RANDOM_STATE, grupos_fusionados, pliegues_corregidos

# ===== SECCION 1: rutas, configuracion y carga de modulos del proyecto =====

IDS_PATH: Final[Path] = PROJECT_ROOT / "data" / "processed" / "holdout_v2_ids.csv"
OOF_BASES_PATH: Final[Path] = PROJECT_ROOT / "features" / "v2_oof_base_corrected.npz"
OOF_META_PATH: Final[Path] = PROJECT_ROOT / "features" / "v2_meta_oof.npz"
OOF_TARGETING_PATH: Final[Path] = PROJECT_ROOT / "features" / "v2_oof_targeting.npz"
OOF_SYNTAX_PATH: Final[Path] = PROJECT_ROOT / "features" / "v2_oof_syntax.npz"
OOF_CITA_PATH: Final[Path] = PROJECT_ROOT / "features" / "v2_oof_cita_negacion.npz"
OOF_EMO_PATH: Final[Path] = PROJECT_ROOT / "features" / "v2_oof_emocion_nbsvm_ortografia.npz"
CACHE_SINTAXIS: Final[Path] = PROJECT_ROOT / "features" / "v2_syntax_annotations.json.gz"
CACHE_AUDITORIA_EXTRA: Final[Path] = PROJECT_ROOT / "features" / "v2_audit_annotations_extra.json.gz"

REPORTE_INTERSECCION: Final[Path] = PROJECT_ROOT / "reports" / "error_intersection_v2.csv"
REPORTE_TRES_MAL: Final[Path] = PROJECT_ROOT / "reports" / "all_three_errors.csv"
REPORTE_COMBINACION: Final[Path] = PROJECT_ROOT / "reports" / "error_by_label_combination_v2.csv"
REPORTE_COMPLEMENTARIEDAD: Final[Path] = PROJECT_ROOT / "reports" / "model_complementarity_v2.csv"
REPORTE_CATALOGO: Final[Path] = PROJECT_ROOT / "reports" / "feature_catalog_v2.csv"
PARQUET_ESTRUCTURADO: Final[Path] = PROJECT_ROOT / "features" / "v2_structured_features.parquet"
MANIFEST_ESTRUCTURADO: Final[Path] = PROJECT_ROOT / "reports" / "v2_structured_features_manifest.md"

REPORTE_BASES_JSON: Final[Path] = PROJECT_ROOT / "reports" / "baseline_v03_dev_metrics.json"
REPORTE_STACKING_JSON: Final[Path] = PROJECT_ROOT / "reports" / "stacking_v2.json"
REPORTE_TARGETING_JSON: Final[Path] = PROJECT_ROOT / "reports" / "targeting_v2.json"
REPORTE_SINTAXIS_JSON: Final[Path] = PROJECT_ROOT / "reports" / "sintaxis_cita_negacion_v2.json"
REPORTE_EMO_JSON: Final[Path] = PROJECT_ROOT / "reports" / "emocion_nbsvm_ortografia_v2.json"

EXPERIMENTS_CSV: Final[Path] = PROJECT_ROOT / "experiments.csv"
CONTEXTO_MD: Final[Path] = PROJECT_ROOT / "reports" / "contexto.md"

CONFIG: Final[dict[str, Any]] = {
    "fecha": "2026-09-25",
    "n_bootstrap": 2000,
    "semilla_bootstrap": RANDOM_STATE,
    "umbral_defecto": 0.5,
    "umbrales_v03": {"HS": 0.44, "TR": 0.51, "AG": 0.46},
    "ventana_frontera": 0.05,
    "tolerancia_f1": 1e-3,
    "f1_v03_corrected": 0.7772114525165721,
    "f1_m2_umbral_05": 0.771778476256122,
    "f1_meta_v2": 0.7873574608549214,
    "historico_n_las_3_mal": 101,
    "historico_base": 1731,
    "historico_fraccion": 101 / 1731,
    "historico_origen": (
        "reports/error_analysis.md (lineas 8 y 115): V03 historico con umbrales "
        "sobre train completo n=1731 (protocolo historico, sin correccion D-016)"
    ),
    "combinaciones_orden": ("000", "100", "101", "110", "111", "001"),
    "combinacion_todas": "TODAS",
}

# Patrones de error por subconjunto de etiquetas equivocadas. El indice es la
# mascara binaria HS*1 + TR*2 + AG*4 de las etiquetas MAL predichas.
PATRONES_ERROR: Final[tuple[tuple[int, str], ...]] = (
    (0, "las_3_correctas"),
    (1, "solo_HS_mal"),
    (2, "solo_TR_mal"),
    (4, "solo_AG_mal"),
    (3, "HS_TR_mal"),
    (5, "HS_AG_mal"),
    (6, "TR_AG_mal"),
    (7, "las_3_mal"),
)

COLUMNAS_CATALOGO: Final[tuple[str, ...]] = (
    "name",
    "family",
    "formula",
    "dtype",
    "source",
    "hypothesis",
    "leakage_risk",
    "fold_dependency",
    "experiment_id",
)

# Proxy deterministico de leetspeak: digito o simbolo de la tabla leet de P5
# dentro de una palabra (no es el transformador del canal D, solo su bandera).
PATRON_LEET_PROXY: Final[re.Pattern[str]] = re.compile(r"[a-z]+[430$1]|[430$1]+[a-z]+")

# Grupos de columnas del dataset consolidado, usados tambien por el manifest.
BLOQUES_ESTRUCTURADOS: Final[dict[str, tuple[str, ...]]] = {
    "longitudes_y_conteos": (
        "n_chars",
        "n_tokens_split",
        "n_tokens",
        "n_oraciones",
        "n_menciones",
        "n_mencion_inicial",
        "n_mencion_interna",
        "n_hashtags",
        "n_urls",
        "n_propn",
        "n_targets",
        "tiene_mencion",
        "tiene_mencion_inicial",
        "tiene_hashtag",
        "tiene_url",
    ),
    "targeting_estructural": (
        "n_colectivos",
        "tiene_colectivo",
        "col_densidad",
        "n_cuant_universal",
        "n_plurales",
        "plural_densidad",
        "plural_generico",
        "n_segunda",
        "tiene_segunda_persona",
        "n_imperativos",
        "col_segunda_persona",
        "tiene_colectivo_regex",
        "tiene_segunda_persona_regex",
    ),
    "cita_negacion": (
        "n_tramos_cita",
        "tiene_cita",
        "n_tokens_en_cita",
        "frac_tokens_cita",
        "comillas_dobles",
        "comillas_simples",
        "comillas_angulares",
        "rt_presente",
        "n_marcadores_atribucion",
        "atribucion_presente",
        "n_negaciones",
        "tiene_negacion",
        "n_neg_polaridad",
    ),
    "ortografia": (
        "tiene_repeticion",
        "n_repeticiones",
        "tiene_puntuacion_interna",
        "n_puntuacion_interna",
        "tiene_leet",
        "n_caracteres_leet",
        "tiene_elongacion",
        "n_elongaciones",
        "tiene_risas",
        "n_risas",
        "mayusculas_sostenidas",
        "tiene_mayusculas_sostenidas",
    ),
    "morfologia": ("n_morph_plural", "n_morph_femenino", "n_morph_pasado"),
}

FUENTES_BLOQUES: Final[dict[str, str]] = {
    "longitudes_y_conteos": (
        "len(text), text.split() y regex de P6.PATRON_MENCION/PATRON_HASHTAG/"
        "PATRON_URL; n_tokens/n_oraciones/n_propn/n_targets de la anotacion spaCy"
    ),
    "targeting_estructural": (
        "regla estructural de P2.es_colectivo_estructural y P2.es_segunda_persona/"
        "P2.es_imperativo replicadas sobre la anotacion serializada; inventarios "
        "fijos de P2.NEGACION/CUANTIFICADORES_UNIVERSALES y constantes del script 06"
    ),
    "cita_negacion": (
        "P3.tramos_comillas, P3.PATRON_RT, P3.MARCADORES_ATRIBUCION y "
        "P2.NEGACION sobre la anotacion serializada"
    ),
    "ortografia": (
        "P5.PATRON_REPETICIONES, P5.PATRON_PUNTUACION_INTERNA, PATRON_LEET_PROXY "
        "(proxy de la tabla leet de P5), P6.PATRON_ELONGACION, "
        "P6.PATRON_RISAS y P6.PATRON_MAYUSCULAS_SOSTENIDAS"
    ),
    "morfologia": "rasgos Number/Gender/Tense de la anotacion spaCy serializada",
    "sintaxis": "anotacion spaCy simbolica (es_core_news_md, sin vectores ni NER)",
}


def log(mensaje: str) -> None:
    """Imprime un mensaje de progreso con prefijo de fase.

    Args:
        mensaje: texto a imprimir.
    """
    print(f"[P9] {mensaje}", flush=True)


def cargar_modulo_proyecto(nombre: str, ruta_relativa: str) -> Any:
    """Importa un script de ``experiments/`` sin ejecutar su ``main``.

    Args:
        nombre: nombre interno del modulo (unico en ``sys.modules``).
        ruta_relativa: ruta del script respecto de la raiz del proyecto.

    Returns:
        Modulo importado.

    Raises:
        ImportError: si el script no se puede cargar.
    """
    ruta = PROJECT_ROOT / ruta_relativa
    especificacion = importlib.util.spec_from_file_location(nombre, ruta)
    if especificacion is None or especificacion.loader is None:
        raise ImportError(f"No se pudo cargar {ruta}")
    modulo = importlib.util.module_from_spec(especificacion)
    sys.modules[especificacion.name] = modulo
    especificacion.loader.exec_module(modulo)
    return modulo


# Los modulos se cargan una sola vez; solo se leen sus constantes y funciones
# ya revisadas (no se modifican ni se reejecutan sus flujos de experimento).
P2 = cargar_modulo_proyecto("experimento_16_p2_auditoria", "experiments/16_targeting_v2.py")
P3 = cargar_modulo_proyecto("experimento_19_p3_auditoria", "experiments/19_sintaxis_cita_negacion_v2.py")
P5 = cargar_modulo_proyecto("experimento_20_p5_auditoria", "experiments/20_emocion_nbsvm_ortografia_v2.py")
P6 = cargar_modulo_proyecto("experimento_17_p6_auditoria", "experiments/17_stacking_v2.py")
P8 = cargar_modulo_proyecto("experimento_21_p8_auditoria", "experiments/21_backlog_restante_v2.py")
CONSTANTES_ESTRUCTURALES, ORIGEN_CONSTANTES = P2.cargar_constantes_error_analysis()


# ===== SECCION 2: carga del desarrollo, canales OOF y anotaciones =====


@dataclass(frozen=True)
class CanalOOF:
    """Canal OOF auditado con sus umbrales documentados.

    Attributes:
        nombre: identificador del canal en el store o en el reporte.
        familia: familia del canal (base, meta, targeting, sintaxis,
            cita_negacion, emocion, nbsvm, ortografia).
        proba: probabilidades OOF ``(n_dev, 3)`` alineadas con el desarrollo.
        umbrales: umbral por etiqueta usado para binarizar.
        origen_umbrales: donde esta documentado ese umbral.
        representante: si es el representante de su familia para la lectura
            principal del reporte de complementariedad.
        nota: anotacion libre (variante, parametro, etc.).
    """

    nombre: str
    familia: str
    proba: NDArray[np.float64]
    umbrales: dict[str, float]
    origen_umbrales: str
    representante: bool
    nota: str


def cargar_desarrollo() -> pd.DataFrame:
    """Carga el conjunto de desarrollo de la particion congelada.

    Returns:
        DataFrame del desarrollo (n=1483) en el orden del archivo de train.

    Raises:
        RuntimeError: si la particion congelada no coincide con el train.
    """
    train = data_module.read_train()
    particion = pd.read_csv(IDS_PATH)
    ids_dev = particion.loc[particion["particion"] == "dev", "id"].astype(int).tolist()
    desarrollo = train[train["id"].astype(int).isin(ids_dev)].reset_index(drop=True)
    if len(desarrollo) != len(ids_dev) or desarrollo["id"].nunique() != len(ids_dev):
        raise RuntimeError("La particion congelada no coincide con el train")
    log(f"desarrollo cargado: n={len(desarrollo)} (holdout congelado intacto)")
    return desarrollo


def _alinear_por_id(ids_store: NDArray[np.int_], mapa_fila: dict[int, int]) -> NDArray[np.int_]:
    """Devuelve la posicion en el desarrollo de cada fila del store OOF.

    Args:
        ids_store: identificadores del store.
        mapa_fila: ``id -> posicion`` en el desarrollo.

    Returns:
        Arreglo de posiciones ``(len(ids_store),)``.

    Raises:
        RuntimeError: si algun id del store no esta en el desarrollo.
    """
    faltantes = [int(valor) for valor in ids_store if int(valor) not in mapa_fila]
    if faltantes:
        raise RuntimeError(f"El store OOF trae ids fuera del desarrollo: {faltantes[:5]}")
    return np.array([mapa_fila[int(valor)] for valor in ids_store])


def cargar_canales(
    desarrollo: pd.DataFrame, y_verdadera: NDArray[np.int_]
) -> dict[str, CanalOOF]:
    """Carga todos los canales OOF disponibles alineados con el desarrollo.

    Los umbrales de cada canal son los documentados en los reportes: V03
    (0.44/0.51/0.46) para la referencia M2 y 0.5 para el resto de variantes.

    Args:
        desarrollo: DataFrame del desarrollo.
        y_verdadera: etiquetas verdaderas ``(n, 3)``.

    Returns:
        Diccionario ``nombre -> CanalOOF``.

    Raises:
        RuntimeError: si las etiquetas de un store no coinciden con el desarrollo.
    """
    mapa_fila = {int(i): p for p, i in enumerate(desarrollo["id"].astype(int))}
    n_filas = len(desarrollo)
    canales: dict[str, CanalOOF] = {}
    umbral_05 = {etiqueta: CONFIG["umbral_defecto"] for etiqueta in ETIQUETAS}
    origen_05 = "umbral 0.5 documentado en el reporte del canal"

    datos = np.load(OOF_BASES_PATH)
    orden = _alinear_por_id(datos["ids"], mapa_fila)
    if not np.array_equal(datos["y_true"], y_verdadera[orden]):
        raise RuntimeError("Las etiquetas del OOF de bases no coinciden con el desarrollo")
    for posicion, modelo in enumerate(str(valor) for valor in datos["modelos"]):
        proba = np.zeros((n_filas, len(ETIQUETAS)))
        proba[orden] = datos["proba"][:, posicion, :]
        es_v03 = modelo == "M2_union_lr_c2"
        nombre = f"v03_base_{modelo}" if es_v03 else f"base_{modelo}"
        canales[nombre] = CanalOOF(
            nombre=nombre,
            familia="base",
            proba=proba,
            umbrales=dict(CONFIG["umbrales_v03"]) if es_v03 else dict(umbral_05),
            origen_umbrales=(
                "umbrales congelados de V03 (0.44/0.51/0.46), reports/baseline_v03_dev_metrics.json"
                if es_v03
                else origen_05
            ),
            representante=es_v03,
            nota="referencia V03 con umbrales congelados" if es_v03 else "base del OOF corregido v2",
        )
        if es_v03:
            # Variante adicional del mismo M2 a 0.5: comparador exacto con el que
            # los reportes midieron el ensemble_gain (media de probabilidades).
            canales["base_M2_union_lr_c2_umbral_05"] = CanalOOF(
                nombre="base_M2_union_lr_c2_umbral_05",
                familia="base",
                proba=proba.copy(),
                umbrales=dict(umbral_05),
                origen_umbrales="umbral 0.5 (comparador de ensemble_gain en P1-P6, 0.7718)",
                representante=False,
                nota="comparador documentado de ensemble_gain",
            )

    datos_meta = np.load(OOF_META_PATH)
    orden_meta = _alinear_por_id(datos_meta["ids"], mapa_fila)
    if not np.array_equal(datos_meta["y_true"], y_verdadera[orden_meta]):
        raise RuntimeError("Las etiquetas del OOF del meta no coinciden con el desarrollo")
    proba_meta = np.zeros((n_filas, len(ETIQUETAS)))
    proba_meta[orden_meta] = datos_meta["proba"]
    variante_meta = str(datos_meta["variante"][0])
    canales[f"meta_{variante_meta}"] = CanalOOF(
        nombre=f"meta_{variante_meta}",
        familia="meta",
        proba=proba_meta,
        umbrales=dict(umbral_05),
        origen_umbrales="umbral 0.5 del meta (reports/stacking_v2.json)",
        representante=True,
        nota=(
            f"meta adoptado en P6 (modelo {datos_meta['modelo'][0]!s}, "
            f"parametro {float(datos_meta['parametro'][0]):g})"
        ),
    )

    for ruta, familia in (
        (OOF_TARGETING_PATH, "targeting"),
        (OOF_SYNTAX_PATH, "sintaxis"),
        (OOF_CITA_PATH, "cita_negacion"),
    ):
        datos_canal = np.load(ruta)
        orden_canal = _alinear_por_id(datos_canal["ids"], mapa_fila)
        if not np.array_equal(datos_canal["y_true"], y_verdadera[orden_canal]):
            raise RuntimeError(f"Las etiquetas de {ruta.name} no coinciden con el desarrollo")
        for posicion, variante in enumerate(str(valor) for valor in datos_canal["variantes"]):
            proba = np.zeros((n_filas, len(ETIQUETAS)))
            proba[orden_canal] = datos_canal["proba"][:, posicion, :]
            canales[f"{familia}_{variante}"] = CanalOOF(
                nombre=f"{familia}_{variante}",
                familia=familia,
                proba=proba,
                umbrales=dict(umbral_05),
                origen_umbrales=origen_05,
                representante=False,
                nota=f"variante {variante} con C={float(datos_canal['C'][posicion]):g}",
            )

    datos_emo = np.load(OOF_EMO_PATH)
    orden_emo = _alinear_por_id(datos_emo["ids"], mapa_fila)
    if not np.array_equal(datos_emo["y_true"], y_verdadera[orden_emo]):
        raise RuntimeError("Las etiquetas del store de emocion/NB-SVM no coinciden con el desarrollo")
    for posicion, canal in enumerate(str(valor) for valor in datos_emo["canales"]):
        proba = np.zeros((n_filas, len(ETIQUETAS)))
        proba[orden_emo] = datos_emo["proba"][posicion]
        prefijo = canal.split("@")[0]
        if prefijo.startswith("NB-"):
            familia = "nbsvm"
        elif prefijo.startswith("ORT-"):
            familia = "ortografia"
        else:
            familia = "emocion"
        canales[f"{familia}_{canal}"] = CanalOOF(
            nombre=f"{familia}_{canal}",
            familia=familia,
            proba=proba,
            umbrales=dict(umbral_05),
            origen_umbrales="umbral 0.5 documentado en reports/emocion_nbsvm_ortografia_v2.json",
            representante=False,
            nota=f"canal {canal} del store de P5",
        )

    marcar_representantes(canales)
    log(f"canales OOF cargados: {len(canales)}")
    return canales


def marcar_representantes(canales: dict[str, CanalOOF]) -> None:
    """Marca el representante de cada familia segun el F1 documentado.

    El representante es la variante de mayor F1 documentado dentro de cada
    familia; se declara en el reporte para que la lectura principal no dependa
    de una eleccion posterior.

    Args:
        canales: diccionario de canales cargados.

    Raises:
        RuntimeError: si el representante documentado no esta entre los canales.
    """
    representantes_documentados = {
        "base": "v03_base_M2_union_lr_c2",
        "meta": "meta_META-BASES-RIDGE",
        "targeting": "targeting_TXT+CH",
        "sintaxis": "sintaxis_TXT+POS",
        "cita_negacion": "cita_negacion_TXT+NEG",
        "emocion": "emocion_EMO-ALL@C=1.0",
        "nbsvm": "nbsvm_NB-UNION-LR@C=1.0",
        "ortografia": "ortografia_ORT-ABCD@C=2.0",
    }
    for familia, nombre in representantes_documentados.items():
        if nombre not in canales:
            raise RuntimeError(f"No existe el representante documentado de {familia}: {nombre}")
        actual = canales[nombre]
        canales[nombre] = CanalOOF(
            nombre=actual.nombre,
            familia=actual.familia,
            proba=actual.proba,
            umbrales=actual.umbrales,
            origen_umbrales=actual.origen_umbrales,
            representante=True,
            nota=actual.nota,
        )


def cargar_referencias_canales() -> dict[str, dict[str, Any]]:
    """Lee F1 y ensemble_gain documentados de cada canal en los reportes.

    Returns:
        Diccionario ``nombre_de_canal -> metadatos`` con ``f1_documentado``,
        ``origen_f1``, ``ensemble_gain``, ``ic_inferior``, ``ic_superior``,
        ``origen_gain`` y ``referencia_gain`` (siempre la media con M2 a 0.5).

    Raises:
        RuntimeError: si un reporte esperado no tiene la estructura conocida.
    """
    referencias: dict[str, dict[str, Any]] = {}
    base_json = json.loads(REPORTE_BASES_JSON.read_text(encoding="utf-8"))
    for modelo, bloque in base_json["modelos"].items():
        referencias[f"base_{modelo}"] = {
            "f1_documentado": float(bloque["resumen"]["f1_media_columnas"]),
            "origen_f1": "reports/baseline_v03_dev_metrics.json (umbral 0.5)",
            "ensemble_gain": None,
            "ic_inferior": None,
            "ic_superior": None,
            "origen_gain": None,
            "referencia_gain": None,
        }
    referencias["v03_base_M2_union_lr_c2"] = {
        "f1_documentado": float(base_json["v03_con_umbrales_corrected"]["f1_media_columnas"]),
        "origen_f1": "reports/baseline_v03_dev_metrics.json (umbrales V03)",
        "ensemble_gain": None,
        "ic_inferior": None,
        "ic_superior": None,
        "origen_gain": None,
        "referencia_gain": None,
    }
    referencias["base_M2_union_lr_c2_umbral_05"] = dict(referencias["base_M2_union_lr_c2"])

    stacking = json.loads(REPORTE_STACKING_JSON.read_text(encoding="utf-8"))
    for variante, bloque in stacking["variantes"].items():
        bootstrap = bloque.get("bootstrap_ensemble") or {}
        referencias[f"meta_{variante}"] = {
            "f1_documentado": float(bloque["corrected_cv"]),
            "origen_f1": "reports/stacking_v2.json (umbral 0.5)",
            "ensemble_gain": float(bloque["ensemble_gain"]),
            "ic_inferior": bootstrap.get("ic_inferior"),
            "ic_superior": bootstrap.get("ic_superior"),
            "origen_gain": "reports/stacking_v2.json (P6)",
            "referencia_gain": "media de probabilidades con M2 a 0.5 (0.7718)",
        }

    for ruta, familia_fija in (
        (REPORTE_TARGETING_JSON, "targeting"),
        (REPORTE_SINTAXIS_JSON, None),
    ):
        reporte = json.loads(ruta.read_text(encoding="utf-8"))
        for variante, bloque in reporte["variantes"].items():
            resumen = bloque["resumen"]
            if familia_fija is not None:
                familia = familia_fija
                origen = "reports/targeting_v2.json (P2)"
            elif resumen.get("seccion") == "P3":
                familia = "sintaxis"
                origen = "reports/sintaxis_cita_negacion_v2.json (P3)"
            elif resumen.get("seccion") == "P4":
                familia = "cita_negacion"
                origen = "reports/sintaxis_cita_negacion_v2.json (P4)"
            else:
                continue
            referencias[f"{familia}_{variante}"] = {
                "f1_documentado": float(resumen["corrected_cv"]),
                "origen_f1": f"{origen} (umbral 0.5)",
                "ensemble_gain": float(resumen["ensemble_gain"]),
                "ic_inferior": None,
                "ic_superior": None,
                "origen_gain": origen,
                "referencia_gain": "media de probabilidades con M2 a 0.5 (0.7718)",
            }

    emocion = json.loads(REPORTE_EMO_JSON.read_text(encoding="utf-8"))
    for clave, resumen in emocion["todas_las_claves"].items():
        if clave.startswith("BASE@"):
            continue
        prefijo = clave.split("@")[0]
        if prefijo.startswith("NB-"):
            familia = "nbsvm"
        elif prefijo.startswith("ORT-"):
            familia = "ortografia"
        else:
            familia = "emocion"
        bootstrap = resumen.get("bootstrap") or {}
        referencias[f"{familia}_{clave}"] = {
            "f1_documentado": float(resumen["corrected_cv"]),
            "origen_f1": "reports/emocion_nbsvm_ortografia_v2.json (umbral 0.5)",
            "ensemble_gain": float(resumen["ensemble_gain"]),
            "ic_inferior": bootstrap.get("ic_inferior"),
            "ic_superior": bootstrap.get("ic_superior"),
            "origen_gain": "reports/emocion_nbsvm_ortografia_v2.json (P5)",
            "referencia_gain": "media de probabilidades con M2 a 0.5 (0.7718)",
        }
    return referencias


def f1_de_canal(canal: CanalOOF, y_verdadera: NDArray[np.int_]) -> float:
    """Calcula la F1 media por columna del canal con sus umbrales.

    Args:
        canal: canal OOF.
        y_verdadera: etiquetas verdaderas ``(n, 3)``.

    Returns:
        F1 media por columna.
    """
    umbrales = np.array([canal.umbrales[etiqueta] for etiqueta in ETIQUETAS])
    prediccion = (canal.proba >= umbrales).astype(int)
    return float(metricas_multietiqueta(y_verdadera, prediccion)["f1_media_columnas"])


def verificar_canales(
    canales: dict[str, CanalOOF],
    referencias: dict[str, dict[str, Any]],
    y_verdadera: NDArray[np.int_],
) -> list[str]:
    """Comprueba que cada canal reproduce su F1 documentado.

    Args:
        canales: canales OOF cargados.
        referencias: metadatos documentados por canal.
        y_verdadera: etiquetas verdaderas.

    Returns:
        Lista de discrepancias (vacia si todo cuadra dentro de la tolerancia).
    """
    discrepancias: list[str] = []
    for nombre, canal in sorted(canales.items()):
        metadatos = referencias.get(nombre)
        if metadatos is None:
            discrepancias.append(f"{nombre}: sin F1 documentado localizable")
            continue
        f1 = f1_de_canal(canal, y_verdadera)
        diferencia = abs(f1 - metadatos["f1_documentado"])
        if diferencia > CONFIG["tolerancia_f1"]:
            discrepancias.append(
                f"{nombre}: F1 recalculada {f1:.6f} vs documentada "
                f"{metadatos['f1_documentado']:.6f} (dif {diferencia:.2e})"
            )
    for nombre, esperado in (
        ("v03_base_M2_union_lr_c2", CONFIG["f1_v03_corrected"]),
        ("meta_META-BASES-RIDGE", CONFIG["f1_meta_v2"]),
        ("base_M2_union_lr_c2_umbral_05", CONFIG["f1_m2_umbral_05"]),
    ):
        f1 = f1_de_canal(canales[nombre], y_verdadera)
        if abs(f1 - float(esperado)) > CONFIG["tolerancia_f1"]:
            discrepancias.append(f"{nombre}: no reproduce la referencia {esperado:.6f} ({f1:.6f})")
    for mensaje in discrepancias:
        log(f"DISCREPANCIA: {mensaje}")
    return discrepancias


# ===== SECCION 3: rasgos estructurales deterministas (sin etiquetas) =====


@dataclass(frozen=True)
class TokenSimple:
    """Token minimo para reutilizar los detectores estructurales de P2.

    Attributes:
        texto: forma original del token.
        lema: lema normalizado en ASCII y minusculas.
        pos: etiqueta POS universal de spaCy.
        numero: rasgo morfologico Number (``""`` si no aplica).
        persona: rasgo morfologico Person (``""`` si no aplica).
        modo: rasgo morfologico Mood (``""`` si no aplica).
    """

    texto: str
    lema: str
    pos: str
    numero: str
    persona: str
    modo: str


@dataclass(frozen=True)
class DocumentoSimple:
    """Documento minimo (lista de tokens) para los detectores de P2.

    Attributes:
        tokens: tokens del documento.
    """

    tokens: tuple[TokenSimple, ...]


def cargar_anotaciones(desarrollo: pd.DataFrame) -> list[list[dict[str, Any]]]:
    """Carga la anotacion spaCy cacheada del desarrollo.

    Args:
        desarrollo: DataFrame del desarrollo.

    Returns:
        Lista de documentos; cada documento es una lista de diccionarios con
        ``t, l, p, n, pe, m, g, te, po, o, i, c, d, h, hp, hd``.

    Raises:
        RuntimeError: si la cache no corresponde al desarrollo.
    """
    with gzip.open(CACHE_SINTAXIS, "rt", encoding="utf-8") as archivo:
        carga = json.load(archivo)
    if carga.get("ids") != desarrollo["id"].astype(int).tolist() or carga.get("modo") != "spacy":
        raise RuntimeError("La cache de anotaciones no coincide con el desarrollo")
    log(f"anotacion spaCy reutilizada de la cache ({len(carga['documentos'])} documentos)")
    return carga["documentos"]


def cargar_o_crear_anotaciones_extra(
    ids: Sequence[int], textos: Sequence[str]
) -> list[list[dict[str, Any]]]:
    """Anota con spaCy las filas fuera del desarrollo (holdout y test).

    La anotacion es determinista (mismo modelo y mismos parametros que P3) y no
    usa etiquetas: solo agrega rasgos de superficie de filas que no estaban
    cacheadas. No interviene en ninguna decision de modelado.

    Args:
        ids: identificadores de las filas a anotar.
        textos: textos alineados con ``ids``.

    Returns:
        Lista de documentos serializados como en la cache de P3.

    Raises:
        RuntimeError: si spaCy no esta disponible.
    """
    if CACHE_AUDITORIA_EXTRA.exists():
        with gzip.open(CACHE_AUDITORIA_EXTRA, "rt", encoding="utf-8") as archivo:
            carga = json.load(archivo)
        if carga.get("ids") == list(ids) and carga.get("modo") == "spacy":
            log(f"anotacion extra reutilizada de la cache ({len(carga['documentos'])} documentos)")
            return carga["documentos"]
        log("la cache extra no coincide con las filas esperadas: se reanota")
    try:
        documentos = P3.anotar_con_spacy(textos)
    except Exception as error:  # sin spaCy no hay rasgos sintacticos
        raise RuntimeError(f"spaCy no disponible para anotar las filas extra: {error}") from error
    # Se reutiliza la serializacion de P3 para que la cache sea identica a la suya.
    serializados = P3._serializar(documentos)
    CACHE_AUDITORIA_EXTRA.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(CACHE_AUDITORIA_EXTRA, "wt", encoding="utf-8") as archivo:
        json.dump({"ids": list(ids), "modo": "spacy", "documentos": serializados}, archivo)
    log(f"anotacion extra completada con spaCy ({len(serializados)} documentos)")
    return serializados


def _tokens_simples(documento: Sequence[dict[str, Any]]) -> tuple[TokenSimple, ...]:
    """Convierte un documento serializado en tokens simples.

    Args:
        documento: documento serializado (lista de diccionarios).

    Returns:
        Tupla de :class:`TokenSimple`.
    """
    return tuple(
        TokenSimple(
            texto=str(registro["t"]),
            lema=str(registro["l"]),
            pos=str(registro["p"]),
            numero=str(registro["n"]),
            persona=str(registro["pe"]),
            modo=str(registro["m"]),
        )
        for registro in documento
    )


def _es_mencion(registro: dict[str, Any]) -> bool:
    """Indica si el token es una mencion social.

    Args:
        registro: token serializado.

    Returns:
        ``True`` si la forma empieza por ``@``.
    """
    return str(registro["t"]).startswith("@")


def _es_colectivo_estructural(
    tokens: Sequence[dict[str, Any]], indice: int, constantes: dict[str, Any]
) -> bool:
    """Replica la regla estructural de referencia colectiva de P2.

    Args:
        tokens: tokens serializados del documento.
        indice: posicion del token.
        constantes: constantes estructurales del script 06.

    Returns:
        ``True`` si el token es una referencia colectiva estructural.
    """
    token = tokens[indice]
    if str(token["p"]) in ("NOUN", "PROPN") and str(token["l"]) in constantes["colectivos"]:
        return True
    return bool(
        str(token["p"]) in P2.POS_COLECTIVO
        and str(token["n"]) == "Plur"
        and indice > 0
        and str(tokens[indice - 1]["p"]) == "DET"
        and str(tokens[indice - 1]["n"]) == "Plur"
    )


def _es_segunda_persona(token: dict[str, Any], constantes: dict[str, Any]) -> bool:
    """Replica el detector de segunda persona de P2.

    Args:
        token: token serializado.
        constantes: constantes estructurales del script 06.

    Returns:
        ``True`` si el token marca segunda persona.
    """
    if str(token["l"]) in constantes["segunda_persona"] or str(token["pe"]) == "2":
        return True
    return str(token["m"]) == "Imp" or str(token["l"]) in constantes["imperativos"]


def rasgos_estructurados(
    textos: Sequence[str],
    textos_minuscula: Sequence[str],
    documentos: Sequence[Sequence[dict[str, Any]]],
) -> pd.DataFrame:
    """Construye los rasgos estructurales deterministas de cada tweet.

    Combina tres fuentes sin etiquetas: regex fijas del proyecto (P3/P5/P6),
    anotacion spaCy simbolica (POS, dependencias, morfologia, cita, negacion) y
    detectores estructurales de P2 (mencion, colectivo, segunda persona,
    imperativo) replicados sobre la anotacion serializada.

    Args:
        textos: textos originales.
        textos_minuscula: textos en minusculas (para las regex de P5).
        documentos: documentos serializados alineados con ``textos``.

    Returns:
        DataFrame con una fila por tweet y las columnas documentadas en
        ``reports/v2_structured_features_manifest.md``.
    """
    pos_catalogo = sorted(
        {str(token["p"]) for documento in documentos for token in documento if str(token["p"])}
    )
    dep_catalogo = sorted(
        {str(token["d"]) for documento in documentos for token in documento if str(token["d"])}
    )
    filas: list[dict[str, Any]] = []
    for texto, minuscula, documento in zip(textos, textos_minuscula, documentos, strict=True):
        n_tokens = len(documento)
        denominador = max(n_tokens, 1)
        posiciones_colectivo = [
            indice
            for indice in range(n_tokens)
            if _es_colectivo_estructural(documento, indice, CONSTANTES_ESTRUCTURALES)
        ]
        plurales = [
            token
            for token in documento
            if str(token["p"]) in P2.POS_COLECTIVO
            and str(token["n"]) == "Plur"
            and str(token["p"]) != "PROPN"
        ]
        menciones = [indice for indice, token in enumerate(documento) if _es_mencion(token)]
        primera = 1 if documento and str(documento[0]["t"]).lower() == "rt" and n_tokens > 1 else 0
        propn = [
            indice
            for indice, token in enumerate(documento)
            if str(token["p"]) == "PROPN" and not _es_mencion(token)
        ]
        segunda = [
            indice
            for indice in range(n_tokens)
            if _es_segunda_persona(documento[indice], CONSTANTES_ESTRUCTURALES)
        ]
        tramos = P3.tramos_comillas(texto)
        en_cita = [bool(token["c"]) for token in documento]
        fila: dict[str, Any] = {
            "n_chars": len(texto),
            "n_tokens_split": len(texto.split()),
            "n_tokens": n_tokens,
            "n_oraciones": len({int(token["o"]) for token in documento}) if documento else 0,
            "n_menciones": len(menciones),
            "n_mencion_inicial": sum(1 for indice in menciones if indice == primera),
            "n_mencion_interna": sum(1 for indice in menciones if indice != primera),
            "n_hashtags": len(P6.PATRON_HASHTAG.findall(texto)),
            "n_urls": len(P6.PATRON_URL.findall(texto)),
            "tiene_mencion": int(bool(menciones)),
            "tiene_mencion_inicial": int(any(indice == primera for indice in menciones)),
            "tiene_hashtag": int(bool(P6.PATRON_HASHTAG.search(texto))),
            "tiene_url": int(bool(P6.PATRON_URL.search(texto))),
            "n_propn": len(propn),
            "n_targets": len(menciones) + len(propn),
            "n_colectivos": len(posiciones_colectivo),
            "tiene_colectivo": int(bool(posiciones_colectivo)),
            "col_densidad": len(posiciones_colectivo) / denominador,
            "n_cuant_universal": sum(
                1 for token in documento if str(token["l"]) in P2.CUANTIFICADORES_UNIVERSALES
            ),
            "n_plurales": len(plurales),
            "plural_densidad": len(plurales) / denominador,
            "plural_generico": int(len(plurales) >= 2 and not segunda and not bool(menciones)),
            "n_segunda": len(segunda),
            "tiene_segunda_persona": int(bool(segunda)),
            "n_imperativos": sum(
                1
                for token in documento
                if str(token["m"]) == "Imp"
                or str(token["l"]) in CONSTANTES_ESTRUCTURALES["imperativos"]
            ),
            "col_segunda_persona": int(bool(posiciones_colectivo) and bool(segunda)),
            "n_tramos_cita": len(tramos),
            "tiene_cita": int(bool(tramos)),
            "n_tokens_en_cita": int(sum(en_cita)),
            "frac_tokens_cita": float(sum(en_cita)) / denominador,
            "comillas_dobles": texto.count('"'),
            "comillas_simples": texto.count("'"),
            "comillas_angulares": texto.count("\u00ab") + texto.count("\u2039"),
            "rt_presente": int(bool(P3.PATRON_RT.search(texto))),
            "n_marcadores_atribucion": sum(
                1 for token in documento if str(token["l"]) in P3.MARCADORES_ATRIBUCION
            ),
            "atribucion_presente": int(
                any(str(token["l"]) in P3.MARCADORES_ATRIBUCION for token in documento)
            ),
            "n_negaciones": sum(1 for token in documento if str(token["l"]) in P2.NEGACION),
            "tiene_negacion": int(any(str(token["l"]) in P2.NEGACION for token in documento)),
            "n_neg_polaridad": sum(1 for token in documento if str(token["po"]) == "Neg"),
            "n_morph_plural": sum(1 for token in documento if str(token["n"]) == "Plur"),
            "n_morph_femenino": sum(1 for token in documento if str(token["g"]) == "Fem"),
            "n_morph_pasado": sum(1 for token in documento if str(token["te"]) == "Past"),
            "tiene_repeticion": int(bool(P5.PATRON_REPETICIONES.search(texto))),
            "n_repeticiones": len(P5.PATRON_REPETICIONES.findall(texto)),
            "tiene_puntuacion_interna": int(bool(P5.PATRON_PUNTUACION_INTERNA.search(texto))),
            "n_puntuacion_interna": len(P5.PATRON_PUNTUACION_INTERNA.findall(texto)),
            "tiene_leet": int(bool(PATRON_LEET_PROXY.search(minuscula))),
            "n_caracteres_leet": len(PATRON_LEET_PROXY.findall(minuscula)),
            "tiene_elongacion": int(bool(P6.PATRON_ELONGACION.search(texto))),
            "n_elongaciones": len(P6.PATRON_ELONGACION.findall(texto)),
            "tiene_risas": int(bool(P6.PATRON_RISAS.search(texto))),
            "n_risas": len(P6.PATRON_RISAS.findall(texto)),
            "mayusculas_sostenidas": max(
                [len(tramo) for tramo in P6.PATRON_MAYUSCULAS_SOSTENIDAS.findall(texto)] or [0]
            ),
            "tiene_mayusculas_sostenidas": int(
                bool(P6.PATRON_MAYUSCULAS_SOSTENIDAS.search(texto))
            ),
            "tiene_colectivo_regex": int(bool(P6.PATRON_COLECTIVO.search(texto))),
            "tiene_segunda_persona_regex": int(bool(P6.PATRON_SEGUNDA_PERSONA.search(texto))),
        }
        conteo_pos = Counter(str(token["p"]) for token in documento)
        for etiqueta_pos in pos_catalogo:
            fila[f"pos_{etiqueta_pos}_n"] = conteo_pos.get(etiqueta_pos, 0)
            fila[f"pos_{etiqueta_pos}_frac"] = conteo_pos.get(etiqueta_pos, 0) / denominador
        conteo_dep = Counter(str(token["d"]) for token in documento)
        for etiqueta_dep in dep_catalogo:
            fila[f"dep_{etiqueta_dep}_n"] = conteo_dep.get(etiqueta_dep, 0)
        filas.append(fila)
    return pd.DataFrame(filas)


def verificar_detectores(
    rasgos: pd.DataFrame, documentos: Sequence[Sequence[dict[str, Any]]]
) -> list[str]:
    """Comprueba la equivalencia de los detectores replicados con los de P2.

    Args:
        rasgos: rasgos estructurales del desarrollo.
        documentos: anotaciones serializadas del desarrollo.

    Returns:
        Lista de discrepancias detectadas (vacia si todo coincide).
    """
    discrepancias: list[str] = []
    documentos_simples = [
        DocumentoSimple(tokens=_tokens_simples(documento)) for documento in documentos
    ]
    mascara_colectivo = P2.rebanada_colectiva_referencia(
        documentos_simples, CONSTANTES_ESTRUCTURALES
    )
    diferencias_colectivo = int(
        np.count_nonzero(mascara_colectivo.astype(int) != rasgos["tiene_colectivo"].to_numpy())
    )
    if diferencias_colectivo:
        discrepancias.append(
            f"regla de colectivo replicada: {diferencias_colectivo} filas difieren de "
            "P2.rebanada_colectiva_referencia"
        )
    else:
        log(
            "verificacion de colectivo: la regla replicada coincide con P2 en las "
            f"{len(rasgos)} filas (n colectiva={int(mascara_colectivo.sum())})"
        )
    segunda_p2 = np.array(
        [
            sum(
                1
                for token in documento.tokens
                if P2.es_segunda_persona(token, CONSTANTES_ESTRUCTURALES)
            )
            for documento in documentos_simples
        ]
    )
    diferencias_segunda = int(
        np.count_nonzero(segunda_p2 != rasgos["n_segunda"].to_numpy())
    )
    if diferencias_segunda:
        discrepancias.append(
            f"detector de segunda persona replicado: {diferencias_segunda} filas difieren de P2"
        )
    else:
        log("verificacion de segunda persona: coincide con P2 en las 1483 filas")
    imperativos_p2 = np.array(
        [
            sum(1 for token in documento.tokens if P2.es_imperativo(token, CONSTANTES_ESTRUCTURALES))
            for documento in documentos_simples
        ]
    )
    diferencias_imperativo = int(
        np.count_nonzero(imperativos_p2 != rasgos["n_imperativos"].to_numpy())
    )
    if diferencias_imperativo:
        discrepancias.append(
            f"detector de imperativo replicado: {diferencias_imperativo} filas difieren de P2"
        )
    else:
        log("verificacion de imperativo: coincide con P2 en las 1483 filas")
    for mensaje in discrepancias:
        log(f"DISCREPANCIA: {mensaje}")
    return discrepancias


def insultos_fold_local(
    desarrollo: pd.DataFrame,
    documentos: Sequence[Sequence[dict[str, Any]]],
    pliegues: list[tuple[NDArray[np.int_], NDArray[np.int_]]],
    y_verdadera: NDArray[np.int_],
) -> tuple[NDArray[np.int_], list[dict[str, Any]]]:
    """Cuenta insultos con el lexico inducido sin ver el pliegue evaluado.

    Args:
        desarrollo: DataFrame del desarrollo.
        documentos: anotaciones serializadas del desarrollo.
        pliegues: pliegues corregidos ``(entrenamiento, validacion)``.
        y_verdadera: etiquetas verdaderas ``(n, 3)``.

    Returns:
        Tupla ``(conteos, diagnostico)``; ``conteos`` es el numero de tokens de
        cada fila cuyo lema pertenece al lexico de insulto inducido con los
        pliegues de entrenamiento de su propio pliegue.
    """
    tokens_simples = [_tokens_simples(documento) for documento in documentos]
    conteos = np.zeros(len(desarrollo), dtype=int)
    diagnostico: list[dict[str, Any]] = []
    for numero, (idx_entrenamiento, idx_validacion) in enumerate(pliegues, start=1):
        documentos_entrenamiento = [
            DocumentoSimple(tokens=tokens_simples[posicion]) for posicion in idx_entrenamiento
        ]
        lexicos = P2.inducir_lexicos(documentos_entrenamiento, y_verdadera[idx_entrenamiento])
        for posicion in idx_validacion:
            conteos[posicion] = sum(
                1 for token in tokens_simples[posicion] if token.lema in lexicos.insulto
            )
        diagnostico.append(
            {
                "pliegue": numero,
                "n_entrenamiento": len(idx_entrenamiento),
                "n_validacion": len(idx_validacion),
                "insulto": int(lexicos.tamanos["insulto"]),
                "adjetivos_negativos": int(lexicos.tamanos["adjetivos_negativos"]),
                "colectivos": int(lexicos.tamanos["colectivos"]),
            }
        )
    return conteos, diagnostico


# ===== SECCION 4: FASE 43/44 - interseccion de errores =====


def patron_de_error(errores: NDArray[np.bool_]) -> NDArray[np.int_]:
    """Codifica el patron de error de cada tweet como mascara de 3 bits.

    Args:
        errores: matriz booleana ``(n, 3)`` con ``True`` en la etiqueta mal
            predicha, en el orden de :data:`ETIQUETAS`.

    Returns:
        Arreglo ``(n,)`` con la mascara ``HS + 2*TR + 4*AG``.
    """
    return errores.astype(int) @ np.array([1, 2, 4], dtype=int)


def bootstrap_fracciones(
    matriz_una_caliente: NDArray[np.float64], n_bootstrap: int, semilla: int
) -> dict[str, tuple[float, float]]:
    """Calcula el IC bootstrap 95 % de la fraccion de cada patron.

    El remuestreo es a nivel de tweet y no pareado: cada iteracion remuestrea
    ``n`` tweets con reemplazo y recalcula la fraccion de cada patron.

    Args:
        matriz_una_caliente: matriz ``(n, k)`` con 1 en el patron de la fila.
        n_bootstrap: numero de remuestreos.
        semilla: semilla del generador.

    Returns:
        Diccionario ``patron -> (ic_inferior, ic_superior)``.
    """
    generador = np.random.default_rng(semilla)
    n_filas = matriz_una_caliente.shape[0]
    acumulado = np.empty((n_bootstrap, matriz_una_caliente.shape[1]), dtype=float)
    for iteracion in range(n_bootstrap):
        indices = generador.integers(0, n_filas, size=n_filas)
        acumulado[iteracion] = matriz_una_caliente[indices].mean(axis=0)
    inferior = np.quantile(acumulado, 0.025, axis=0)
    superior = np.quantile(acumulado, 0.975, axis=0)
    return {
        nombre: (float(inferior[posicion]), float(superior[posicion]))
        for posicion, (_, nombre) in enumerate(PATRONES_ERROR)
    }


def tabla_interseccion(
    nombre_referencia: str,
    prediccion: NDArray[np.int_],
    y_verdadera: NDArray[np.int_],
) -> pd.DataFrame:
    """Construye la tabla de patrones de error de una referencia.

    Args:
        nombre_referencia: identificador de la referencia.
        prediccion: prediccion binaria ``(n, 3)``.
        y_verdadera: etiquetas verdaderas ``(n, 3)``.

    Returns:
        DataFrame con n, fraccion e IC bootstrap 95 % por patron.
    """
    errores = prediccion.astype(int) != y_verdadera
    mascaras = patron_de_error(errores)
    n_filas = len(y_verdadera)
    matriz = np.zeros((n_filas, len(PATRONES_ERROR)), dtype=float)
    for posicion, (mascara, _) in enumerate(PATRONES_ERROR):
        matriz[:, posicion] = (mascaras == mascara).astype(float)
    intervalos = bootstrap_fracciones(matriz, CONFIG["n_bootstrap"], CONFIG["semilla_bootstrap"])
    filas: list[dict[str, Any]] = []
    for posicion, (_, nombre) in enumerate(PATRONES_ERROR):
        conteo = int(matriz[:, posicion].sum())
        fraccion = conteo / n_filas
        inferior, superior = intervalos[nombre]
        es_las_3 = nombre == "las_3_mal"
        filas.append(
            {
                "referencia": nombre_referencia,
                "base_desarrollo": f"desarrollo corregido v2, n={n_filas}",
                "patron": nombre,
                "n": conteo,
                "fraccion": round(fraccion, 6),
                "ic_inferior_95": round(inferior, 6),
                "ic_superior_95": round(superior, 6),
                "n_bootstrap": int(CONFIG["n_bootstrap"]),
                "n_historico_las_3_mal": CONFIG["historico_n_las_3_mal"] if es_las_3 else None,
                "fraccion_historico_las_3_mal": (
                    round(float(CONFIG["historico_fraccion"]), 6) if es_las_3 else None
                ),
                "base_historica": CONFIG["historico_base"] if es_las_3 else None,
                "origen_historico": CONFIG["historico_origen"] if es_las_3 else None,
                "diferencia_vs_historico_puntos_pct": (
                    round((fraccion - float(CONFIG["historico_fraccion"])) * 100, 3)
                    if es_las_3
                    else None
                ),
            }
        )
    return pd.DataFrame(filas)


def fila_historica_interseccion() -> pd.DataFrame:
    """Construye la fila de contraste con la referencia historica de la spec.

    Returns:
        DataFrame de una fila con la cifra historica 101 (5.8 %) documentada
        sobre el train completo n=1731 y el protocolo historico.
    """
    return pd.DataFrame(
        [
            {
                "referencia": "historico_spec",
                "base_desarrollo": (
                    f"train completo n={CONFIG['historico_base']} (protocolo historico)"
                ),
                "patron": "las_3_mal",
                "n": CONFIG["historico_n_las_3_mal"],
                "fraccion": round(float(CONFIG["historico_fraccion"]), 6),
                "ic_inferior_95": None,
                "ic_superior_95": None,
                "n_bootstrap": None,
                "n_historico_las_3_mal": CONFIG["historico_n_las_3_mal"],
                "fraccion_historico_las_3_mal": round(float(CONFIG["historico_fraccion"]), 6),
                "base_historica": CONFIG["historico_base"],
                "origen_historico": CONFIG["historico_origen"],
                "diferencia_vs_historico_puntos_pct": 0.0,
            }
        ]
    )


def escribir_interseccion(
    canales: dict[str, CanalOOF],
    referencias_usadas: Sequence[str],
    y_verdadera: NDArray[np.int_],
) -> pd.DataFrame:
    """Genera ``reports/error_intersection_v2.csv`` (FASE 43/44).

    Args:
        canales: canales cargados.
        referencias_usadas: nombres de las dos referencias auditadas.
        y_verdadera: etiquetas verdaderas.

    Returns:
        Tabla escrita.
    """
    tablas = []
    for nombre in referencias_usadas:
        canal = canales[nombre]
        umbrales = np.array([canal.umbrales[etiqueta] for etiqueta in ETIQUETAS])
        prediccion = (canal.proba >= umbrales).astype(int)
        tabla = tabla_interseccion(nombre, prediccion, y_verdadera)
        tabla.insert(2, "f1_referencia", round(f1_de_canal(canal, y_verdadera), 6))
        tablas.append(tabla)
    piezas = [fila_historica_interseccion(), *tablas]
    columnas = list(dict.fromkeys(columna for pieza in piezas for columna in pieza.columns))
    tabla_final = pd.DataFrame(
        [registro for pieza in piezas for registro in pieza.to_dict("records")],
        columns=columnas,
    )
    if "n_bootstrap" in tabla_final.columns:
        tabla_final["n_bootstrap"] = tabla_final["n_bootstrap"].astype("Int64")
    tabla_final.to_csv(REPORTE_INTERSECCION, index=False, encoding="utf-8")
    log(f"escrito {REPORTE_INTERSECCION.name}: {len(tabla_final)} filas")
    return tabla_final


# ===== SECCION 5: FASE 43 - tweets con las tres etiquetas mal =====


COLUMNAS_TRES_MAL: Final[tuple[str, ...]] = (
    "n_chars",
    "n_tokens",
    "n_oraciones",
    "n_menciones",
    "n_hashtags",
    "n_urls",
    "n_propn",
    "n_targets",
    "n_segunda",
    "n_colectivos",
    "n_negaciones",
    "n_tramos_cita",
    "frac_tokens_cita",
    "n_repeticiones",
    "n_puntuacion_interna",
    "n_caracteres_leet",
    "n_elongaciones",
    "n_risas",
    "mayusculas_sostenidas",
    "tiene_mencion_inicial",
    "tiene_colectivo",
    "tiene_segunda_persona",
    "tiene_cita",
    "tiene_negacion",
    "tiene_repeticion",
    "tiene_puntuacion_interna",
    "tiene_leet",
    "rt_presente",
)


def tabla_tres_mal(
    nombre_referencia: str,
    prediccion: NDArray[np.int_],
    desarrollo: pd.DataFrame,
    y_verdadera: NDArray[np.int_],
    pliegue_de_fila: NDArray[np.int_],
    rasgos: pd.DataFrame,
    insultos: NDArray[np.int_],
) -> pd.DataFrame:
    """Construye las filas de tweets con las tres etiquetas mal en una referencia.

    Args:
        nombre_referencia: identificador de la referencia.
        prediccion: prediccion binaria ``(n, 3)``.
        desarrollo: DataFrame del desarrollo.
        y_verdadera: etiquetas verdaderas ``(n, 3)``.
        pliegue_de_fila: pliegue corregido de cada fila.
        rasgos: rasgos estructurados deterministas del desarrollo.
        insultos: conteo de insultos fold-local del desarrollo.

    Returns:
        DataFrame con una fila por tweet que falla las tres etiquetas.
    """
    errores = prediccion.astype(int) != y_verdadera
    mascara = errores.all(axis=1)
    tabla = pd.DataFrame(
        {
            "tweet_id": desarrollo.loc[mascara, "id"].astype(int).to_numpy(),
            "referencia": nombre_referencia,
            "pliegue": pliegue_de_fila[mascara],
            "HS_true": y_verdadera[mascara, 0],
            "TR_true": y_verdadera[mascara, 1],
            "AG_true": y_verdadera[mascara, 2],
        }
    )
    for columna in COLUMNAS_TRES_MAL:
        tabla[columna] = rasgos.loc[mascara, columna].to_numpy()
    tabla["n_insultos_foldlocal"] = insultos[mascara]
    tabla["tiene_insulto_foldlocal"] = (insultos[mascara] > 0).astype(int)
    return tabla.sort_values(["referencia", "tweet_id"]).reset_index(drop=True)


def escribir_tres_mal(
    canales: dict[str, CanalOOF],
    referencias_usadas: Sequence[str],
    desarrollo: pd.DataFrame,
    y_verdadera: NDArray[np.int_],
    pliegue_de_fila: NDArray[np.int_],
    rasgos: pd.DataFrame,
    insultos: NDArray[np.int_],
) -> pd.DataFrame:
    """Genera ``reports/all_three_errors.csv`` (FASE 43).

    Args:
        canales: canales cargados.
        referencias_usadas: nombres de las dos referencias auditadas.
        desarrollo: DataFrame del desarrollo.
        y_verdadera: etiquetas verdaderas.
        pliegue_de_fila: pliegue corregido de cada fila.
        rasgos: rasgos estructurados del desarrollo.
        insultos: conteo de insultos fold-local.

    Returns:
        Tabla escrita.
    """
    tablas = []
    for nombre in referencias_usadas:
        canal = canales[nombre]
        umbrales = np.array([canal.umbrales[etiqueta] for etiqueta in ETIQUETAS])
        prediccion = (canal.proba >= umbrales).astype(int)
        tablas.append(
            tabla_tres_mal(
                nombre, prediccion, desarrollo, y_verdadera, pliegue_de_fila, rasgos, insultos
            )
        )
    tabla = pd.concat(tablas, ignore_index=True)
    tabla.to_csv(REPORTE_TRES_MAL, index=False, encoding="utf-8")
    log(
        f"escrito {REPORTE_TRES_MAL.name}: {len(tabla)} filas "
        f"({int((tabla['referencia'] == referencias_usadas[0]).sum())} "
        f"{referencias_usadas[0]}, "
        f"{int((tabla['referencia'] == referencias_usadas[1]).sum())} {referencias_usadas[1]})"
    )
    return tabla


# ===== SECCION 6: FASE 47 - error por combinacion de etiquetas =====


def metricas_etiquetas(
    verdad: NDArray[np.int_], prediccion: NDArray[np.int_]
) -> dict[str, float]:
    """Calcula precision, recall y F1 por etiqueta dentro de un subconjunto.

    Se implementa de forma explicita para no depender de advertencias de
    sklearn cuando una etiqueta es constante en el subconjunto.

    Args:
        verdad: etiquetas verdaderas ``(n, 3)`` del subconjunto.
        prediccion: prediccion binaria ``(n, 3)`` del subconjunto.

    Returns:
        Diccionario plano ``precision_<etiqueta>``, ``recall_<etiqueta>`` y
        ``f1_<etiqueta>``.
    """
    resultado: dict[str, float] = {}
    for posicion, etiqueta in enumerate(ETIQUETAS):
        positivo_real = verdad[:, posicion].astype(bool)
        positivo_predicho = prediccion[:, posicion].astype(bool)
        verdaderos_positivos = int(np.count_nonzero(positivo_real & positivo_predicho))
        falsos_positivos = int(np.count_nonzero(~positivo_real & positivo_predicho))
        falsos_negativos = int(np.count_nonzero(positivo_real & ~positivo_predicho))
        precision = (
            verdaderos_positivos / (verdaderos_positivos + falsos_positivos)
            if verdaderos_positivos + falsos_positivos
            else 0.0
        )
        recall = (
            verdaderos_positivos / (verdaderos_positivos + falsos_negativos)
            if verdaderos_positivos + falsos_negativos
            else 0.0
        )
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        resultado[f"precision_{etiqueta}"] = round(precision, 6)
        resultado[f"recall_{etiqueta}"] = round(recall, 6)
        resultado[f"f1_{etiqueta}"] = round(f1, 6)
    return resultado


def _medias_estructurales(
    rasgos: pd.DataFrame, insultos: NDArray[np.int_], mascara: NDArray[np.bool_]
) -> dict[str, Any]:
    """Calcula las medias de senales estructurales de un subconjunto.

    Args:
        rasgos: rasgos estructurales del desarrollo.
        insultos: conteo de insultos fold-local del desarrollo.
        mascara: mascara booleana del subconjunto.

    Returns:
        Diccionario de medias (``None`` si el subconjunto esta vacio).
    """
    if not np.any(mascara):
        return {
            clave: None
            for clave in (
                "media_n_chars",
                "media_n_tokens",
                "media_n_oraciones",
                "frac_mencion",
                "frac_mencion_inicial",
                "frac_hashtag",
                "frac_url",
                "frac_colectivo",
                "frac_segunda_persona",
                "frac_cita",
                "frac_negacion",
                "frac_insulto_foldlocal",
                "media_n_insultos_foldlocal",
                "media_frac_tokens_cita",
                "media_n_repeticiones",
                "frac_rt",
            )
        }
    return {
        "media_n_chars": round(float(rasgos.loc[mascara, "n_chars"].mean()), 3),
        "media_n_tokens": round(float(rasgos.loc[mascara, "n_tokens"].mean()), 3),
        "media_n_oraciones": round(float(rasgos.loc[mascara, "n_oraciones"].mean()), 3),
        "frac_mencion": round(float(rasgos.loc[mascara, "tiene_mencion"].mean()), 4),
        "frac_mencion_inicial": round(
            float(rasgos.loc[mascara, "tiene_mencion_inicial"].mean()), 4
        ),
        "frac_hashtag": round(float(rasgos.loc[mascara, "tiene_hashtag"].mean()), 4),
        "frac_url": round(float(rasgos.loc[mascara, "tiene_url"].mean()), 4),
        "frac_colectivo": round(float(rasgos.loc[mascara, "tiene_colectivo"].mean()), 4),
        "frac_segunda_persona": round(
            float(rasgos.loc[mascara, "tiene_segunda_persona"].mean()), 4
        ),
        "frac_cita": round(float(rasgos.loc[mascara, "tiene_cita"].mean()), 4),
        "frac_negacion": round(float(rasgos.loc[mascara, "tiene_negacion"].mean()), 4),
        "frac_insulto_foldlocal": round(float((insultos[mascara] > 0).mean()), 4),
        "media_n_insultos_foldlocal": round(float(insultos[mascara].mean()), 3),
        "media_frac_tokens_cita": round(float(rasgos.loc[mascara, "frac_tokens_cita"].mean()), 4),
        "media_n_repeticiones": round(float(rasgos.loc[mascara, "n_repeticiones"].mean()), 3),
        "frac_rt": round(float(rasgos.loc[mascara, "rt_presente"].mean()), 4),
    }


def escribir_combinaciones(
    canales: dict[str, CanalOOF],
    referencias_usadas: Sequence[str],
    desarrollo: pd.DataFrame,
    y_verdadera: NDArray[np.int_],
    rasgos: pd.DataFrame,
    insultos: NDArray[np.int_],
) -> pd.DataFrame:
    """Genera ``reports/error_by_label_combination_v2.csv`` (FASE 47).

    Las metricas por etiqueta se calculan DENTRO del subconjunto de cada
    combinacion verdadera: cuando una etiqueta es constante en el subconjunto,
    su precision coincide con su recall por construccion (no es un error de
    calculo, se documenta en la columna ``nota``).

    Args:
        canales: canales cargados.
        referencias_usadas: nombres de las dos referencias auditadas.
        desarrollo: DataFrame del desarrollo.
        y_verdadera: etiquetas verdaderas.
        rasgos: rasgos estructurados del desarrollo.
        insultos: conteo de insultos fold-local.

    Returns:
        Tabla escrita.
    """
    combinaciones = (
        desarrollo["HS"].astype(str) + desarrollo["TR"].astype(str) + desarrollo["AG"].astype(str)
    ).to_numpy()
    filas: list[dict[str, Any]] = []
    for nombre in referencias_usadas:
        canal = canales[nombre]
        umbrales = np.array([canal.umbrales[etiqueta] for etiqueta in ETIQUETAS])
        prediccion = (canal.proba >= umbrales).astype(int)
        for combinacion in (*CONFIG["combinaciones_orden"], CONFIG["combinacion_todas"]):
            if combinacion == CONFIG["combinacion_todas"]:
                mascara = np.ones(len(desarrollo), dtype=bool)
            else:
                mascara = combinaciones == combinacion
            n = int(mascara.sum())
            fila: dict[str, Any] = {
                "referencia": nombre,
                "combinacion": combinacion,
                "n": n,
                "f1_referencia_global": round(f1_de_canal(canal, y_verdadera), 6),
            }
            if n:
                fila.update(metricas_etiquetas(y_verdadera[mascara], prediccion[mascara]))
            else:
                fila.update(
                    {
                        f"{metrica}_{etiqueta}": None
                        for metrica in ("precision", "recall", "f1")
                        for etiqueta in ETIQUETAS
                    }
                )
            fila.update(_medias_estructurales(rasgos, insultos, mascara))
            fila["nota"] = (
                "metricas por etiqueta dentro del subconjunto; si la etiqueta es constante "
                "en el subconjunto, precision = recall por construccion; insulto = lexico "
                "inducido fold-local (sin ver la validacion)"
            )
            filas.append(fila)
    tabla = pd.DataFrame(filas)
    tabla.to_csv(REPORTE_COMBINACION, index=False, encoding="utf-8")
    log(f"escrito {REPORTE_COMBINACION.name}: {len(tabla)} filas")
    return tabla


# ===== SECCION 7: FASE 50/51/91 - complementariedad entre canales =====


def matriz_decision(canal: CanalOOF) -> NDArray[np.bool_]:
    """Binariza las probabilidades del canal con sus umbrales.

    Args:
        canal: canal OOF.

    Returns:
        Matriz booleana ``(n, 3)``.
    """
    umbrales = np.array([canal.umbrales[etiqueta] for etiqueta in ETIQUETAS])
    return canal.proba >= umbrales


def metricas_par(
    canal_a: CanalOOF,
    canal_b: CanalOOF,
    y_verdadera: NDArray[np.int_],
) -> dict[str, Any]:
    """Calcula diversidad, correlacion de errores y correccion mutua de un par.

    Las metricas se calculan sobre las ``n * 3`` celdas (fila x etiqueta) para
    conservar la informacion por etiqueta, igual que
    ``reports/ensembles_diversidad.csv``. La correlacion de errores es la de
    Pearson entre los vectores de error y la ``q_statistic`` la definida en la
    Fase 2 del proyecto. ``desacuerdo_frontera`` es la tasa de desacuerdo entre
    las celdas en las que al menos un canal esta a menos de
    ``CONFIG['ventana_frontera']`` de su umbral.

    Args:
        canal_a: primer canal.
        canal_b: segundo canal.
        y_verdadera: etiquetas verdaderas ``(n, 3)``.

    Returns:
        Diccionario con las metricas del par.
    """
    decision_a = matriz_decision(canal_a)
    decision_b = matriz_decision(canal_b)
    verdad = y_verdadera.astype(bool)
    acierto_a = (decision_a == verdad).ravel()
    acierto_b = (decision_b == verdad).ravel()
    prediccion_a = decision_a.ravel()
    prediccion_b = decision_b.ravel()
    error_a = ~acierto_a
    error_b = ~acierto_b
    n11 = int(np.count_nonzero(acierto_a & acierto_b))
    n00 = int(np.count_nonzero(error_a & error_b))
    n01 = int(np.count_nonzero(acierto_a & ~acierto_b))
    n10 = int(np.count_nonzero(~acierto_a & acierto_b))
    denominador_q = n11 * n00 + n01 * n10
    desviacion_a = float(np.std(error_a.astype(float)))
    desviacion_b = float(np.std(error_b.astype(float)))
    correlacion = (
        float(np.corrcoef(error_a.astype(float), error_b.astype(float))[0, 1])
        if desviacion_a > 0 and desviacion_b > 0
        else 0.0
    )
    umbrales_a = np.array([canal_a.umbrales[etiqueta] for etiqueta in ETIQUETAS])
    umbrales_b = np.array([canal_b.umbrales[etiqueta] for etiqueta in ETIQUETAS])
    n_filas = len(y_verdadera)
    frontera_a = (
        np.abs(canal_a.proba.ravel() - np.tile(umbrales_a, n_filas))
        <= CONFIG["ventana_frontera"]
    )
    frontera_b = (
        np.abs(canal_b.proba.ravel() - np.tile(umbrales_b, n_filas))
        <= CONFIG["ventana_frontera"]
    )
    frontera = frontera_a | frontera_b
    desacuerdo = prediccion_a != prediccion_b
    return {
        "n_celdas": int(verdad.size),
        "acuerdo_decision": round(float(np.mean(~desacuerdo)), 6),
        "tasa_desacuerdo": round(float(np.mean(desacuerdo)), 6),
        "correlacion_errores": round(correlacion, 6),
        "q_statistic": (
            round(float((n11 * n00 - n01 * n10) / denominador_q), 6) if denominador_q else 0.0
        ),
        "n_celdas_frontera": int(np.count_nonzero(frontera)),
        "desacuerdo_frontera": (
            round(float(np.mean(desacuerdo[frontera])), 6) if np.any(frontera) else None
        ),
        "errores_a": int(np.count_nonzero(error_a)),
        "errores_b": int(np.count_nonzero(error_b)),
        "solo_a_correcto": n01,
        "solo_b_correcto": n10,
        "ambos_correctos": n11,
        "ambos_incorrectos": n00,
        "tasa_correccion_mutua_a_sobre_b": (
            round(float(np.mean(acierto_b[error_a])), 6) if np.any(error_a) else None
        ),
        "tasa_correccion_mutua_b_sobre_a": (
            round(float(np.mean(acierto_a[error_b])), 6) if np.any(error_b) else None
        ),
    }


def _ganancia_documentada(
    canal_a: CanalOOF,
    canal_b: CanalOOF,
    metadatos_a: dict[str, Any],
    metadatos_b: dict[str, Any],
) -> tuple[float | None, str | None, str | None, str | None]:
    """Recupera el ensemble_gain de un par solo si ya estaba medido.

    Los reportes midieron el ``ensemble_gain`` como la F1 de la media de
    probabilidades del canal con M2 a 0.5 menos 0.7718. Por eso solo se copia
    cuando uno de los dos canales es exactamente el comparador documentado
    ``base_M2_union_lr_c2_umbral_05``.

    Args:
        canal_a: primer canal.
        canal_b: segundo canal.
        metadatos_a: metadatos documentados del primer canal.
        metadatos_b: metadatos documentados del segundo canal.

    Returns:
        Tupla ``(ganancia, canal_medido, origen, referencia)``; los tres
        ultimos son ``None`` si no hay ganancia documentada.
    """
    comparador = "base_M2_union_lr_c2_umbral_05"
    for canal, metadatos in ((canal_a, metadatos_a), (canal_b, metadatos_b)):
        if canal.nombre == comparador:
            continue
        otro = canal_b if canal is canal_a else canal_a
        if otro.nombre != comparador:
            continue
        if metadatos.get("ensemble_gain") is None:
            continue
        return (
            round(float(metadatos["ensemble_gain"]), 6),
            canal.nombre,
            metadatos.get("origen_gain"),
            metadatos.get("referencia_gain"),
        )
    return None, None, None, None


def escribir_complementariedad(
    canales: dict[str, CanalOOF],
    referencias: dict[str, dict[str, Any]],
    y_verdadera: NDArray[np.int_],
) -> pd.DataFrame:
    """Genera ``reports/model_complementarity_v2.csv`` (FASE 50/51/91).

    Args:
        canales: canales cargados.
        referencias: metadatos documentados por canal.
        y_verdadera: etiquetas verdaderas.

    Returns:
        Tabla escrita con todas las parejas de canales no ordenadas.
    """
    nombres = sorted(canales)
    f1_por_canal = {nombre: f1_de_canal(canales[nombre], y_verdadera) for nombre in nombres}
    filas: list[dict[str, Any]] = []
    for posicion_a in range(len(nombres)):
        for posicion_b in range(posicion_a + 1, len(nombres)):
            canal_a = canales[nombres[posicion_a]]
            canal_b = canales[nombres[posicion_b]]
            metricas = metricas_par(canal_a, canal_b, y_verdadera)
            metadatos_a = referencias.get(canal_a.nombre, {})
            metadatos_b = referencias.get(canal_b.nombre, {})
            ganancia, ganancia_canal, origen_gain, referencia_gain = _ganancia_documentada(
                canal_a, canal_b, metadatos_a, metadatos_b
            )
            metadatos_ganancia = metadatos_a if ganancia_canal == canal_a.nombre else metadatos_b
            filas.append(
                {
                    "canal_a": canal_a.nombre,
                    "canal_b": canal_b.nombre,
                    "familia_a": canal_a.familia,
                    "familia_b": canal_b.familia,
                    "representante_a": int(canal_a.representante),
                    "representante_b": int(canal_b.representante),
                    "f1_standalone_a": round(f1_por_canal[canal_a.nombre], 6),
                    "f1_standalone_b": round(f1_por_canal[canal_b.nombre], 6),
                    "f1_documentado_a": (
                        round(float(metadatos_a["f1_documentado"]), 6) if metadatos_a else None
                    ),
                    "f1_documentado_b": (
                        round(float(metadatos_b["f1_documentado"]), 6) if metadatos_b else None
                    ),
                    "origen_f1_a": metadatos_a.get("origen_f1"),
                    "origen_f1_b": metadatos_b.get("origen_f1"),
                    "umbrales_a": json.dumps(canal_a.umbrales, ensure_ascii=False, sort_keys=True),
                    "umbrales_b": json.dumps(canal_b.umbrales, ensure_ascii=False, sort_keys=True),
                    "origen_umbrales_a": canal_a.origen_umbrales,
                    "origen_umbrales_b": canal_b.origen_umbrales,
                    **metricas,
                    "ensemble_gain": ganancia,
                    "ensemble_gain_ic_inferior": (
                        metadatos_ganancia.get("ic_inferior") if ganancia_canal else None
                    ),
                    "ensemble_gain_ic_superior": (
                        metadatos_ganancia.get("ic_superior") if ganancia_canal else None
                    ),
                    "ensemble_gain_canal": ganancia_canal,
                    "ensemble_gain_origen": origen_gain,
                    "ensemble_gain_referencia": referencia_gain,
                    "nota": (
                        "metricas sobre las n*3 celdas; correlacion_errores = Pearson de los "
                        "indicadores de error; tasa_correccion_mutua_X_sobre_Y = P(Y correcto | "
                        "X error); desacuerdo_frontera = desacuerdo entre celdas con "
                        f"|p-umbral| <= {CONFIG['ventana_frontera']}; ensemble_gain solo si "
                        "estaba medido en un reporte"
                    ),
                }
            )
    tabla = pd.DataFrame(filas)
    tabla.to_csv(REPORTE_COMPLEMENTARIEDAD, index=False, encoding="utf-8")
    log(
        f"escrito {REPORTE_COMPLEMENTARIEDAD.name}: {len(tabla)} parejas de "
        f"{len(nombres)} canales"
    )
    return tabla


def mejor_par(tabla: pd.DataFrame) -> dict[str, Any]:
    """Identifica la pareja con mayor correccion mutua simetrica.

    Args:
        tabla: tabla de complementariedad.

    Returns:
        Diccionario con la pareja y sus tasas, o vacio si no hay datos.
    """
    valida = tabla.dropna(
        subset=["tasa_correccion_mutua_a_sobre_b", "tasa_correccion_mutua_b_sobre_a"]
    )
    if valida.empty:
        return {}
    simetrica = valida.assign(
        min_correccion=valida[
            ["tasa_correccion_mutua_a_sobre_b", "tasa_correccion_mutua_b_sobre_a"]
        ].min(axis=1)
    ).sort_values("min_correccion", ascending=False)
    fila = simetrica.iloc[0]
    return {
        "canal_a": str(fila["canal_a"]),
        "canal_b": str(fila["canal_b"]),
        "correccion_a_sobre_b": float(fila["tasa_correccion_mutua_a_sobre_b"]),
        "correccion_b_sobre_a": float(fila["tasa_correccion_mutua_b_sobre_a"]),
        "f1_standalone_a": float(fila["f1_standalone_a"]),
        "f1_standalone_b": float(fila["f1_standalone_b"]),
    }


# ===== SECCION 8: FASE 55/88 - catalogo de features implementadas =====


def _fila_catalogo(
    name: str,
    family: str,
    formula: str,
    dtype: str,
    source: str,
    hypothesis: str,
    leakage_risk: str,
    fold_dependency: str,
    experiment_id: str,
) -> dict[str, str]:
    """Construye una fila del catalogo con las nueve columnas exigidas.

    Args:
        name: nombre de la feature.
        family: familia de la feature.
        formula: formula o regla de calculo.
        dtype: tipo de dato.
        source: archivo y funcion de origen.
        hypothesis: hipotesis que motiva la feature.
        leakage_risk: riesgo de fuga y su justificacion.
        fold_dependency: dependencia del pliegue.
        experiment_id: identificador(es) del log de experimentos.

    Returns:
        Diccionario con las columnas de :data:`COLUMNAS_CATALOGO`.
    """
    return {
        "name": name,
        "family": family,
        "formula": formula,
        "dtype": dtype,
        "source": source,
        "hypothesis": hypothesis,
        "leakage_risk": leakage_risk,
        "fold_dependency": fold_dependency,
        "experiment_id": experiment_id,
    }


def _catalogo_texto_base() -> list[dict[str, str]]:
    """Catalogo de la representacion de texto del baseline V03.

    Returns:
        Filas del catalogo.
    """
    return [
        _fila_catalogo(
            "tfidf_word_1_2",
            "texto_base",
            "TF-IDF word n-gramas (1,2), min_df=2, sublinear_tf=True",
            "sparse_csr (float64)",
            "experiments/14_oof_corregido_v2.py PARAMS_WORD",
            "La union word+char es la representacion ganadora del baseline (E1)",
            "bajo (vectorizador ajustado solo con el pliegue de entrenamiento)",
            "fold-local (vocabulario y IDF del pliegue)",
            "V2-BASE-M1_word_lr, V2-BASE-M2_union_lr_c2",
        ),
        _fila_catalogo(
            "tfidf_char_wb_3_5",
            "texto_base",
            "TF-IDF char_wb n-gramas (3,5), min_df=2, sublinear_tf=True",
            "sparse_csr (float64)",
            "experiments/14_oof_corregido_v2.py PARAMS_CHAR",
            "char_wb captura morfologia y ofuscacion que word pierde",
            "bajo (vectorizador ajustado solo con el pliegue de entrenamiento)",
            "fold-local (vocabulario y IDF del pliegue)",
            "V2-BASE-M2_union_lr_c2, V2-BASE-M4_char_lr",
        ),
    ]


FORMULAS_E2_05: Final[dict[str, str]] = {
    "ti_tiene_par": "1 si hay al menos un target y al menos un insulto",
    "ti_n_pares": "n_targets * n_insultos",
    "ti_n_targets": "numero de menciones y PROPN (posiciones unicas)",
    "ti_n_insultos": "tokens cuyo lema pertenece al lexico de insulto inducido fold-local",
    "ti_n_mencion_inicial": "menciones en la primera posicion (tras RT)",
    "ti_n_mencion_interna": "menciones que no son la inicial",
    "ti_n_propn": "tokens PROPN que no son mencion",
    "ti_densidad": "n_pares / max(n_targets, 1)",
    "ti_dist_tok_min": "distancia minima en tokens entre pares target-insulto (-1 si no hay par)",
    "ti_dist_tok_media": "distancia media en tokens entre pares (-1 si no hay par)",
    "ti_dist_tok_mediana": "mediana de la distancia en tokens entre pares (-1 si no hay par)",
    "ti_dist_tok_max": "distancia maxima en tokens entre pares (-1 si no hay par)",
    "ti_dist_car_min": "distancia minima en caracteres entre pares (-1 si no hay par)",
    "ti_dist_car_media": "distancia media en caracteres entre pares (-1 si no hay par)",
    "ti_dist_car_mediana": "mediana de la distancia en caracteres entre pares (-1 si no hay par)",
    "ti_dist_car_max": "distancia maxima en caracteres entre pares (-1 si no hay par)",
    "ti_orden_media": "media del signo de (posicion_insulto - posicion_target)",
    "ti_frac_antes": "fraccion de pares con el insulto antes del target",
    "ti_frac_despues": "fraccion de pares con el insulto despues del target",
    "ti_par_misma_oracion": "1 si algun par cae en la misma oracion",
    "ti_par_le3": "1 si algun par esta a 3 tokens o menos",
    "ti_par_le5": "1 si algun par esta a 5 tokens o menos",
    "ti_par_adyacente": "1 si algun par esta a 1 token o menos",
}

FORMULAS_E2_06: Final[dict[str, str]] = {
    "col_presente": "1 si hay al menos un termino colectivo (regla estructural o inventario inducido)",
    "col_conteo": "numero de terminos colectivos",
    "col_intensidad": "log1p(col_conteo)",
    "col_densidad": "col_conteo / max(n_tokens, 1)",
    "col_insulto": "1 si existe algun par colectivo-insulto",
    "col_insulto_conteo": "numero de pares colectivo-insulto",
    "col_insulto_le3": "1 si algun par colectivo-insulto esta a 3 tokens o menos",
    "col_adj_neg": "1 si existe algun par colectivo-adjetivo negativo inducido",
    "col_adj_neg_conteo": "numero de pares colectivo-adjetivo negativo",
    "cuant_universal": "1 si hay un cuantificador universal (inventario fijo del script 06)",
    "plural_densidad": "plurales NOUN/ADJ/PROPN (sin PROPN) / max(n_tokens, 1)",
    "plural_generico": "1 si hay >= 2 plurales y no hay segunda persona ni menciones",
    "col_segunda_persona": "1 si hay termino colectivo y marca de segunda persona",
}


def _catalogo_targeting() -> list[dict[str, str]]:
    """Catalogo de la familia de targeting (E2-05/E2-06).

    Returns:
        Filas del catalogo.
    """
    filas = []
    for nombre in P2.NOMBRES_E2_05:
        filas.append(
            _fila_catalogo(
                nombre,
                "targeting",
                FORMULAS_E2_05[nombre],
                "float64",
                "experiments/16_targeting_v2.py features_e2_05 / _agregados_distancias",
                "La distancia target-insulto separa agresion dirigida de insulto generico",
                "bajo (lexico de insulto inducido solo con el pliegue de entrenamiento)",
                "fold-local (lexico de insulto y targets del pliegue)",
                "E2-05",
            )
        )
    for nombre in P2.NOMBRES_E2_06:
        filas.append(
            _fila_catalogo(
                nombre,
                "targeting",
                FORMULAS_E2_06[nombre],
                "float64",
                "experiments/16_targeting_v2.py _agregados_colectivo",
                "La referencia colectiva proxima a la agresion es la firma del discurso de odio",
                "bajo (inventario colectivo minado solo con el pliegue de entrenamiento)",
                "fold-local (inventario colectivo y lexicos del pliegue)",
                "E2-06",
            )
        )
    filas.append(
        _fila_catalogo(
            "targeting_secuencia_simbolica",
            "targeting",
            "secuencia de simbolos por token (TARGET, INSULT, COLLECTIVE, SECOND_PERSON, NEG, QUOTE, SEP, O) -> TF-IDF word (1,2)",
            "sparse_csr (float64)",
            "experiments/16_targeting_v2.py simbolizar + PARAMS_SIMBOLICO",
            "El orden relativo de target, insulto y colectivo codifica la agresion dirigida",
            "bajo (simbolos derivados de lexicos fold-locales y reglas estructurales)",
            "fold-local (lexicos y vocabulario del pliegue)",
            "E2-05, E2-06 (canales CH-TGT+DIST y TXT+CH)",
        )
    )
    return filas


def _catalogo_sintaxis() -> list[dict[str, str]]:
    """Catalogo de la familia de sintaxis sparse (E2-03/E2-04).

    Returns:
        Filas del catalogo.
    """
    return [
        _fila_catalogo(
            "syn_pos_tfidf",
            "sintaxis",
            "secuencia de etiquetas POS universales -> TF-IDF (1,4) simbolico (min_df=1, lowercase=False)",
            "sparse_csr (float64)",
            "experiments/19_sintaxis_cita_negacion_v2.py PARAMS_POS",
            "Los patrones POS capturan estructura que word/char no representan",
            "ninguno (solo etiquetas POS, sin etiquetas de clase)",
            "fold-local (vocabulario e IDF del pliegue)",
            "E2-03 (canales SYN-POS, TXT+POS)",
        ),
        _fila_catalogo(
            "syn_deps_tfidf",
            "sintaxis",
            "por arco (hijo, cabeza) se emiten POS_HIJO_POS_CABEZA, DEP_CABEZA_POS y DEP_HEAD -> TF-IDF (1,3)",
            "sparse_csr (float64)",
            "experiments/19_sintaxis_cita_negacion_v2.py PARAMS_DEPS",
            "Las relaciones de dependencia (sujeto, modificador, negacion) discriminan agresion",
            "ninguno (etiquetas de dependencia, sin etiquetas de clase)",
            "fold-local (vocabulario e IDF del pliegue)",
            "E2-04 (canales SYN-DEPS, TXT+DEPS)",
        ),
        _fila_catalogo(
            "syn_morph_tfidf",
            "sintaxis",
            "por token, un termino Rasgo=Valor (Number/Gender/Mood/Tense/Person/Polarity) -> TF-IDF (1,2)",
            "sparse_csr (float64)",
            "experiments/19_sintaxis_cita_negacion_v2.py PARAMS_MORPH, FEATURES_MORFOLOGICAS",
            "La morfologia (persona, numero, tiempo) marca segunda persona y colectivos",
            "ninguno (rasgos morfologicos, sin etiquetas de clase)",
            "fold-local (vocabulario e IDF del pliegue)",
            "E2-04 (canales SYN-MORPH, TXT+MORPH)",
        ),
    ]


FORMULAS_CITA_NEG: Final[dict[str, str]] = {
    "cita_presente": "1 si el texto tiene al menos un tramo entre comillas cerrado",
    "n_tramos_cita": "numero de tramos entre comillas (escaner de pila sobre los pares soportados)",
    "comillas_dobles": "numero de comillas dobles rectas",
    "comillas_simples": "numero de comillas simples rectas",
    "comillas_angulares": "numero de comillas angulares o tipograficas",
    "rt_presente": "1 si el texto empieza por RT @usuario",
    "atribucion_presente": "1 si algun lema esta en los marcadores de atribucion (decir, segun, afirmar, declarar, comentar, publicar, acusar, llamar)",
    "n_marcadores_atribucion": "numero de marcadores de atribucion presentes",
    "frac_chars_en_cita": "caracteres dentro de comillas / caracteres totales",
    "insulto_dentro_de_cita": "insultos inducidos dentro de los tramos de cita",
    "insulto_fuera_de_cita": "insultos inducidos fuera de los tramos de cita",
    "target_dentro_de_cita": "targets (mencion/PROPN) dentro de los tramos de cita",
    "target_fuera_de_cita": "targets (mencion/PROPN) fuera de los tramos de cita",
    "negacion_presente": "1 si hay al menos un negador (inventario fijo o polaridad negativa)",
    "negacion_count": "numero de negadores del documento",
    "neg_dep_tag": "numero de tokens con dep_ == neg (0 con es_core_news_md, documentado en P3)",
    "insulto_bajo_negacion": "1 si hay un insulto en el alcance de un negador",
    "target_bajo_negacion": "1 si hay un target en el alcance de un negador",
    "neg_verbo": "1 si hay un verbo negado (NEG + VERB)",
}


def _catalogo_cita_negacion() -> list[dict[str, str]]:
    """Catalogo de la familia de cita/atribucion y negacion (E2-07/E2-08).

    Returns:
        Filas del catalogo.
    """
    filas = []
    for nombre in P3.NOMBRES_CITA:
        filas.append(
            _fila_catalogo(
                nombre,
                "cita_negacion",
                FORMULAS_CITA_NEG[nombre],
                "float64",
                "experiments/19_sintaxis_cita_negacion_v2.py NOMBRES_CITA",
                "La cita y la atribucion a terceros producen falsos positivos del modelo",
                "bajo (lexico de insulto inducido del pliegue de entrenamiento)",
                "fold-local (lexico de insulto del pliegue)",
                "E2-07",
            )
        )
    for nombre in P3.NOMBRES_NEG:
        filas.append(
            _fila_catalogo(
                nombre,
                "cita_negacion",
                FORMULAS_CITA_NEG[nombre],
                "float64",
                "experiments/19_sintaxis_cita_negacion_v2.py NOMBRES_NEG",
                "La negacion sintactica cambia el sentido del insulto o del target",
                "bajo (inventario fijo de negadores + polaridad de spaCy)",
                "fold-local (lexico de insulto del pliegue)",
                "E2-08",
            )
        )
    filas.append(
        _fila_catalogo(
            "cita_texto_dentro_y_fuera",
            "cita_negacion",
            "dos representaciones word (1,2) separadas (texto citado y texto propio) concatenadas",
            "sparse_csr (float64)",
            "experiments/19_sintaxis_cita_negacion_v2.py PARAMS_CITA",
            "Separar la voz citada de la voz del autor aisla el insulto reproducido",
            "ninguno (solo separacion determinista por comillas)",
            "fold-local (vocabulario e IDF del pliegue)",
            "E2-07 (canales CITA, TXT+CITA)",
        )
    )
    filas.append(
        _fila_catalogo(
            "neg_patrones_tfidf",
            "cita_negacion",
            "secuencia de patrones (NEG_INSULT, NEG_VERB, NEG_TARGET) -> TF-IDF (1,2) simbolico",
            "sparse_csr (float64)",
            "experiments/19_sintaxis_cita_negacion_v2.py PARAMS_NEG",
            "El patron de negacion sobre insulto/target distingue reproche de elogio",
            "bajo (inventario fijo de negadores + lexico de insulto del pliegue)",
            "fold-local (lexico de insulto del pliegue)",
            "E2-08 (canales NEG, TXT+NEG, TXT+CITA+NEG)",
        )
    )
    return filas


FORMULAS_EMO_ESTATICAS: Final[dict[str, str]] = {
    "est_risa_n": "conteo de marcadores de risa (ja/je/jo repetidos, lol, xd)",
    "est_risa_densidad": "est_risa_n / max(n_tokens, 1)",
    "est_afecto_n": "conteo de marcadores de afecto del lexico estatico",
    "est_afecto_densidad": "est_afecto_n / max(n_tokens, 1)",
    "est_diminutivo_n": "conteo de diminutivos -ito/-ita/-itos/-itas",
    "est_diminutivo_densidad": "est_diminutivo_n / max(n_tokens, 1)",
    "est_vocativo_n": "menciones + palabras capitalizadas + palabras tras coma",
    "est_vocativo_densidad": "est_vocativo_n / max(n_tokens, 1)",
    "est_mencion_n": "numero de menciones",
    "est_mencion_inicial": "1 si la mencion esta en la posicion inicial (tras RT)",
    "est_emoticono_n": "conteo de emoticones de texto",
    "est_emoticono_densidad": "est_emoticono_n / max(n_tokens, 1)",
    "est_intensificador_n": "conteo de intensificadores",
    "est_intensificador_densidad": "est_intensificador_n / max(n_tokens, 1)",
    "est_intensificador_grafia_n": "conteo de grafias elongadas (PATRON_GRAFIA_ELONGADA)",
    "est_amenaza_n": "conteo de marcadores de amenaza",
    "est_amenaza_densidad": "est_amenaza_n / max(n_tokens, 1)",
    "est_deseo_negativo_n": "conteo de marcadores de deseo negativo",
    "est_deseo_negativo_densidad": "est_deseo_negativo_n / max(n_tokens, 1)",
    "est_int_risa_insulto": "est_risa_n * n_insultos inducidos del pliegue",
    "est_int_afecto_insulto": "est_afecto_n * n_insultos inducidos del pliegue",
    "est_int_vocativo_insulto": "est_vocativo_n * n_insultos inducidos del pliegue",
    "est_int_risa_sin_colectivo": "est_risa_n * (0 si hay colectivo, 1 si no)",
    "est_n_tokens": "numero de tokens de la anotacion spaCy",
}

FORMULAS_EMO_LEXICO: Final[dict[str, str]] = {
    "lex_suma_HS": "suma de los log-odds inducidos (HS) de los lemas del documento",
    "lex_suma_TR": "suma de los log-odds inducidos (TR) de los lemas del documento",
    "lex_suma_AG": "suma de los log-odds inducidos (AG) de los lemas del documento",
    "lex_media_HS": "lex_suma_HS / max(n_tokens, 1)",
    "lex_media_TR": "lex_suma_TR / max(n_tokens, 1)",
    "lex_media_AG": "lex_suma_AG / max(n_tokens, 1)",
    "lex_max_HS": "maximo log-odds inducido (HS) del documento",
    "lex_max_TR": "maximo log-odds inducido (TR) del documento",
    "lex_max_AG": "maximo log-odds inducido (AG) del documento",
    "lex_margen_HS_AG": "lex_suma_HS - lex_suma_AG",
    "lex_margen_HS_TR": "lex_suma_HS - lex_suma_TR",
    "lex_margen_TR_AG": "lex_suma_TR - lex_suma_AG",
    "lex_tokens_neg_HS": "tokens con log-odds inducido (HS) menor que -1",
    "lex_tokens_neg_TR": "tokens con log-odds inducido (TR) menor que -1",
    "lex_tokens_neg_AG": "tokens con log-odds inducido (AG) menor que -1",
}


def _formula_ventana(nombre: str) -> str:
    """Construye la formula de una feature de ventana a partir de su nombre.

    Args:
        nombre: nombre de la feature (``ven_<ancla>_...``).

    Returns:
        Formula legible derivada del nombre.
    """
    partes = nombre.split("_")
    ancla = f"{partes[1]}s" if partes[1] == "mencion" else "colectivos"
    etiqueta = partes[-1]
    if "_k" in nombre:
        tamano = nombre.split("_k")[1].split("_")[0]
        return (
            f"media de la polaridad inducida ({etiqueta}) en ventanas de +-{tamano} tokens "
            f"alrededor de {ancla}"
        )
    if nombre.endswith((f"_max_{etiqueta}",)):
        return (
            f"maximo entre ventanas +-3/5/10 de la media de polaridad ({etiqueta}) "
            f"alrededor de {ancla}"
        )
    if nombre.endswith((f"_min_{etiqueta}",)):
        return (
            f"minimo entre ventanas +-3/5/10 de la media de polaridad ({etiqueta}) "
            f"alrededor de {ancla}"
        )
    if nombre == "ven_n_menciones":
        return "numero de menciones usadas como ancla"
    return "numero de referencias colectivas usadas como ancla"


def _catalogo_emocion() -> list[dict[str, str]]:
    """Catalogo de la familia de emocion/afecto (E2-09).

    Returns:
        Filas del catalogo.
    """
    filas = []
    for nombre in P5.NOMBRES_ESTATICAS:
        filas.append(
            _fila_catalogo(
                nombre,
                "emocion",
                FORMULAS_EMO_ESTATICAS[nombre],
                "float64",
                "experiments/20_emocion_nbsvm_ortografia_v2.py features_estaticas",
                "La risa, el afecto y el vocativo desactivan el falso positivo por broma o afiliacion",
                "bajo (regex y listas fijas; las interacciones usan el lexico de insulto del pliegue)",
                "fold-local (solo en las interacciones con n_insultos)",
                "E2-09 (canales EMO-EST, EMO-ALL)",
            )
        )
    for nombre in P5.NOMBRES_LEXICO:
        filas.append(
            _fila_catalogo(
                nombre,
                "emocion",
                FORMULAS_EMO_LEXICO[nombre],
                "float64",
                "experiments/20_emocion_nbsvm_ortografia_v2.py features_lexico",
                "El lexico de afecto inducido mide el tono del autor, no solo el insulto",
                "bajo (log-odds inducidos solo con el pliegue de entrenamiento)",
                "fold-local (lexico de afecto del pliegue)",
                "E2-09 (canales EMO-LEX, EMO-ALL)",
            )
        )
    for nombre in P5.NOMBRES_VENTANAS:
        filas.append(
            _fila_catalogo(
                nombre,
                "emocion",
                _formula_ventana(nombre),
                "float64",
                "experiments/20_emocion_nbsvm_ortografia_v2.py polaridad_ventanas",
                "La polaridad local alrededor del target captura el afecto dirigido",
                "bajo (log-odds inducidos solo con el pliegue de entrenamiento)",
                "fold-local (lexico de afecto y colectivos del pliegue)",
                "E2-09 (canales EMO-LEX, EMO-ALL)",
            )
        )
    return filas


def _catalogo_ortografia() -> list[dict[str, str]]:
    """Catalogo de la familia ortografica (E2-10).

    Returns:
        Filas del catalogo.
    """
    return [
        _fila_catalogo(
            "ort_canal_a_raw",
            "ortografia",
            "texto crudo (canal A del multicanal ortografico)",
            "str -> TF-IDF word+char",
            "experiments/20_emocion_nbsvm_ortografia_v2.py VARIANTES_ORT ('A')",
            "El canal crudo conserva el estilo y es la referencia del multicanal",
            "ninguno (no transforma)",
            "fold-local (vectorizador del pliegue)",
            "E2-10 (canal ORT-A)",
        ),
        _fila_catalogo(
            "ort_canal_b_colapso_repeticiones",
            "ortografia",
            "colapso de repeticiones de 3 o mas caracteres a una sola ocurrencia",
            "str -> TF-IDF word+char",
            "experiments/20_emocion_nbsvm_ortografia_v2.py VARIANTES_ORT ('B'), PATRON_REPETICIONES",
            "El colapso de elongaciones recupera el lexema ofuscado (holaaa -> hola)",
            "ninguno (transformacion determinista del texto)",
            "fold-local (vectorizador del pliegue)",
            "E2-10 (canales ORT-B, ORT-AB, ORT-ABCD)",
        ),
        _fila_catalogo(
            "ort_canal_c_sin_puntuacion_interna",
            "ortografia",
            "eliminacion de separadores internos de palabra (p.u.t.a -> puta)",
            "str -> TF-IDF word+char",
            "experiments/20_emocion_nbsvm_ortografia_v2.py VARIANTES_ORT ('C'), PATRON_PUNTUACION_INTERNA",
            "La puntuacion interna se usa para evadir filtros de lexico",
            "ninguno (transformacion determinista del texto)",
            "fold-local (vectorizador del pliegue)",
            "E2-10 (canales ORT-C, ORT-ABCD)",
        ),
        _fila_catalogo(
            "ort_canal_d_leetspeak",
            "ortografia",
            "normalizacion leetspeak con tabla fija (4->a, 3->e, 0->o, 1->i, $->s) y 1 inicial -> l",
            "str -> TF-IDF word+char",
            "experiments/20_emocion_nbsvm_ortografia_v2.py VARIANTES_ORT ('D'), TABLA_LEET",
            "La sustitucion de grafias tipo leet oculta insultos y agresion",
            "ninguno (transformacion determinista del texto)",
            "fold-local (vectorizador del pliegue)",
            "E2-10 (canales ORT-D, ORT-ABCD)",
        ),
        _fila_catalogo(
            "ort_flags_superficie",
            "ortografia",
            "flags de repeticion, puntuacion interna, leet, elongacion, mayusculas sostenidas y risas",
            "int",
            "experiments/22_artefactos_auditoria_v2.py rasgos_estructurados (PATRON_LEET_PROXY)",
            "Los rasgos de superficie son deterministas y auditables para el analisis de error",
            "ninguno (regex fijas, sin etiquetas)",
            "ninguna (determinista)",
            "AUD-01, AUD-02, AUD-03",
        ),
    ]


def _catalogo_nbsvm() -> list[dict[str, str]]:
    """Catalogo de la familia NB-SVM (E2-15).

    Returns:
        Filas del catalogo.
    """
    filas = []
    for nombre, representacion, canales in (
        ("word", "word (1,2)", "NB-WORD-LR, NB-WORD-SVC"),
        ("char", "char_wb (3,5)", "NB-CHAR-LR, NB-CHAR-SVC"),
        ("union", "union word+char", "NB-UNION-LR, NB-UNION-SVC"),
    ):
        filas.append(
            _fila_catalogo(
                f"nbsvm_log_count_ratio_{nombre}",
                "nbsvm",
                "r = log((p + alpha) / (q + alpha)) con p/q las frecuencias normalizadas de la clase positiva/negativa -> features x*r para LR o LinearSVC",
                "sparse_csr (float64)",
                f"experiments/20_emocion_nbsvm_ortografia_v2.py (NB-SVM {representacion})",
                "El peso log-count ratio de Wang y Manning (2012) da un modelo barato y diverso",
                "bajo (pesos ajustados solo con el pliegue de entrenamiento)",
                "fold-local (pesos y modelo del pliegue)",
                f"E2-15 (canales {canales})",
            )
        )
    return filas


def _catalogo_entidades() -> list[dict[str, str]]:
    """Catalogo de la familia de entidades por target encoding (E2-13).

    Returns:
        Filas del catalogo.
    """
    filas = []
    for tipo in P8.TIPOS_ENTIDAD:
        for etiqueta in ETIQUETAS:
            filas.append(
                _fila_catalogo(
                    f"ent_{tipo}_{etiqueta}",
                    "entidades",
                    f"target encoding OOF de {tipo} ({etiqueta}): media de (positivos + m*prior) / (frecuencia + m), m=10, frecuencia minima 5, bucket desconocido = prior",
                    "float64",
                    "experiments/21_backlog_restante_v2.py features_entidades",
                    "Las entidades (mencion, hashtag, dominio) perfilan el objetivo del mensaje",
                    "medio (target encoding; se ajusta solo con el pliegue y se evalua fuera de el)",
                    "fold-local (tablas y prior del pliegue de ajuste)",
                    "E2-13 (meta con entidades)",
                )
            )
        filas.append(
            _fila_catalogo(
                f"ent_{tipo}_n",
                "entidades",
                f"numero de entidades distintas de tipo {tipo} en el tweet",
                "float64",
                "experiments/21_backlog_restante_v2.py codificar_entidades",
                "La cobertura de entidades modula la confianza del encoding",
                "ninguno (conteo determinista)",
                "ninguna (determinista)",
                "E2-13 (meta con entidades)",
            )
        )
    return filas


FORMULAS_ESTRUCTURALES_META: Final[dict[str, str]] = {
    "n_chars": "numero de caracteres del texto",
    "n_tokens": "numero de tokens separados por espacio",
    "tiene_mencion": "1 si hay mencion (@usuario)",
    "tiene_hashtag": "1 si hay hashtag (#tema)",
    "tiene_url": "1 si hay URL o t.co",
    "tiene_colectivo": "1 si la regex de cuantificadores/colectivos encuentra un termino",
    "tiene_segunda_persona": "1 si la regex de segunda persona encuentra un termino",
    "tiene_comillas": "1 si hay comillas rectas o tipograficas",
    "tiene_risas": "1 si hay marcadores de risa (ja/je/jo repetidos, lol, xd)",
    "mayusculas_sostenidas": "longitud del tramo de mayusculas sostenidas mas largo",
    "tiene_elongacion": "1 si hay un caracter repetido 3 o mas veces",
}


def _catalogo_dependencia_meta() -> list[dict[str, str]]:
    """Catalogo de la familia de dependencia entre etiquetas y meta-modelo.

    Returns:
        Filas del catalogo derivadas de los conjuntos documentados en
        ``reports/stacking_v2.json`` (no se inventan nombres).
    """
    stacking = json.loads(REPORTE_STACKING_JSON.read_text(encoding="utf-8"))
    nombres_bases = [str(valor) for valor in stacking["conjuntos"]["bases"]["nombres"]]
    nombres_canales = [
        str(valor) for valor in stacking["conjuntos"]["bases_canales"]["nombres"]
    ]
    nombres_ampliado = [
        str(valor) for valor in stacking["conjuntos"]["bases_canales_ampliado"]["nombres"]
    ]
    canales_exclusivos = [
        nombre for nombre in nombres_canales if nombre not in set(nombres_bases)
    ]
    extra_estructural = [
        nombre for nombre in nombres_ampliado if nombre not in set(nombres_canales)
    ]
    filas: list[dict[str, str]] = []
    for nombre in nombres_bases:
        if nombre.startswith("proba_"):
            modelo, etiqueta = nombre[len("proba_") :].rsplit("_", 1)
            formula = (
                f"probabilidad OOF del base {modelo} para {etiqueta} "
                "(sigmoide del decision_function en SVC)"
            )
            hypothesis = "Los scores de los 4 bases contienen informacion complementaria por etiqueta"
            fold_dep = "OOF (el base correspondiente no vio el pliegue evaluado)"
            experiment = "E2-02, E2-18"
        elif nombre.startswith("margen_"):
            modelo, etiqueta = nombre[len("margen_") :].rsplit("_", 1)
            formula = f"|proba_{modelo}_{etiqueta} - 0.5|"
            hypothesis = "El margen mide la confianza del base"
            fold_dep = "OOF (derivado de un score OOF)"
            experiment = "E2-02, E2-18"
        elif nombre.startswith("entropia_"):
            modelo, etiqueta = nombre[len("entropia_") :].rsplit("_", 1)
            formula = f"entropia binaria de proba_{modelo}_{etiqueta}"
            hypothesis = "La entropia separa decisiones ambiguas de decisiones firmes"
            fold_dep = "OOF (derivado de un score OOF)"
            experiment = "E2-18"
        else:
            formula = FORMULAS_ESTRUCTURALES_META[nombre]
            hypothesis = "Los rasgos estructurales fijos modulan la dependencia entre etiquetas"
            fold_dep = "ninguna (determinista)"
            experiment = "E2-02, E2-18"
        filas.append(
            _fila_catalogo(
                nombre,
                "dependencia_etiquetas",
                formula,
                "float64",
                "reports/stacking_v2.json conjuntos.bases; experiments/17_stacking_v2.py bloque_scores, rasgos_estructurales",
                hypothesis,
                "medio (los scores OOF del entrenamiento del meta provienen de modelos que vieron el pliegue; sesgo medido <= 0.0003 en P1)",
                fold_dep,
                experiment,
            )
        )
    for nombre in canales_exclusivos:
        bloque = nombre.split("_", 1)[0]
        cuerpo = nombre.split("_", 1)[1]
        canal, etiqueta = cuerpo.rsplit("_", 1)
        if bloque == "proba":
            formula = f"probabilidad OOF del canal {canal} para {etiqueta}"
        elif bloque == "margen":
            formula = f"|proba_{canal}_{etiqueta} - 0.5|"
        else:
            formula = f"entropia binaria de proba_{canal}_{etiqueta}"
        filas.append(
            _fila_catalogo(
                nombre,
                "dependencia_etiquetas",
                formula,
                "float64",
                "reports/stacking_v2.json conjuntos.bases_canales; experiments/17_stacking_v2.py bloque_scores",
                "Los canales de targeting/sintaxis/cita/emocion pueden aportar al meta aunque no ganen solos",
                "medio (canales OOF con lexicos fold-locales; el meta se ajusta sin el pliegue evaluado)",
                "OOF (derivado del OOF congelado del canal)",
                "E2-18 (META+CH, META+CH+ESTR)",
            )
        )
    for nombre in extra_estructural:
        filas.append(
            _fila_catalogo(
                nombre,
                "dependencia_etiquetas",
                FORMULAS_ESTRUCTURALES_META[nombre],
                "float64",
                "reports/stacking_v2.json conjuntos.bases_canales_ampliado; experiments/17_stacking_v2.py rasgos_estructurados",
                "Risas, mayusculas sostenidas y elongacion amplian los rasgos fijos del meta",
                "ninguno (rasgos deterministas)",
                "ninguna (determinista)",
                "E2-18 (META+CH+ESTR)",
            )
        )
    for etiqueta in ETIQUETAS:
        filas.append(
            _fila_catalogo(
                f"cadena_score_previo_{etiqueta}",
                "dependencia_etiquetas",
                f"score de la etiqueta previa a {etiqueta} en el orden de la cadena (LR del pliegue)",
                "float64",
                "experiments/15_cadenas_etiquetas_v2.py bloque_denso",
                "La etiqueta previa (dependencia HS->TR->AG) aporta al clasificador de la siguiente",
                "medio (el score previo se produce dentro del pliegue; el ultimo paso no ve su validacion)",
                "fold-local (cadena ajustada por pliegue)",
                "E2-01",
            )
        )
        filas.append(
            _fila_catalogo(
                f"cadena_margen_previo_{etiqueta}",
                "dependencia_etiquetas",
                f"|score_previo_{etiqueta} - 0.5| en el orden de la cadena",
                "float64",
                "experiments/15_cadenas_etiquetas_v2.py bloque_denso",
                "El margen de la etiqueta previa mide la confianza de la dependencia",
                "medio (derivado de un score fold-local)",
                "fold-local (cadena ajustada por pliegue)",
                "E2-01",
            )
        )
    return filas


def escribir_catalogo() -> pd.DataFrame:
    """Genera ``reports/feature_catalog_v2.csv`` (FASE 55/88).

    Returns:
        Tabla escrita.

    Raises:
        RuntimeError: si hay nombres duplicados en el catalogo.
    """
    filas = [
        *_catalogo_texto_base(),
        *_catalogo_targeting(),
        *_catalogo_sintaxis(),
        *_catalogo_cita_negacion(),
        *_catalogo_emocion(),
        *_catalogo_ortografia(),
        *_catalogo_nbsvm(),
        *_catalogo_entidades(),
        *_catalogo_dependencia_meta(),
    ]
    tabla = pd.DataFrame(filas, columns=list(COLUMNAS_CATALOGO))
    duplicados = tabla["name"][tabla["name"].duplicated()].tolist()
    if duplicados:
        raise RuntimeError(f"Nombres duplicados en el catalogo: {sorted(set(duplicados))}")
    tabla.to_csv(REPORTE_CATALOGO, index=False, encoding="utf-8")
    log(
        f"escrito {REPORTE_CATALOGO.name}: {len(tabla)} features en "
        f"{tabla['family'].nunique()} familias"
    )
    return tabla


# ===== SECCION 9: FASE 55/89 - dataset consolidado de features estructuradas =====


def construir_features_estructuradas() -> pd.DataFrame:
    """Consolida las features estructuradas deterministas de train y test.

    Train incluye dev y holdout con la columna ``particion`` para que el
    protocolo D-016 siga siendo explicito; no se incluye ninguna etiqueta. Las
    filas fuera del desarrollo se anotan con spaCy en una cache aparte
    (determinista, sin etiquetas) para que el bloque sintactico este completo.

    Returns:
        DataFrame con ``id``, ``particion`` y las features estructuradas.

    Raises:
        RuntimeError: si la particion no cubre el train o hay ids repetidos.
    """
    train = data_module.read_train()
    test = data_module.read_test()
    particion = pd.read_csv(IDS_PATH)
    mapa_particion = {
        int(identificador): str(valor)
        for identificador, valor in zip(
            particion["id"].astype(int), particion["particion"].astype(str), strict=True
        )
    }
    train = train.assign(particion=train["id"].astype(int).map(mapa_particion))
    if train["particion"].isna().any():
        raise RuntimeError("Hay ids de train fuera de la particion congelada")
    test = test.assign(particion="test")
    if set(train["id"].astype(int)) & set(test["id"].astype(int)):
        raise RuntimeError("Train y test comparten identificadores")

    desarrollo = train[train["particion"] == "dev"].reset_index(drop=True)
    ids_extra = (
        train.loc[train["particion"] != "dev", "id"].astype(int).tolist()
        + test["id"].astype(int).tolist()
    )
    textos_extra = (
        train.loc[train["particion"] != "dev", "text"].tolist() + test["text"].tolist()
    )
    documentos_dev = cargar_anotaciones(desarrollo)
    documentos_extra = cargar_o_crear_anotaciones_extra(ids_extra, textos_extra)
    documentos_por_id: dict[int, list[dict[str, Any]]] = {}
    for identificador, documento in zip(
        desarrollo["id"].astype(int), documentos_dev, strict=True
    ):
        documentos_por_id[int(identificador)] = documento
    for identificador, documento in zip(ids_extra, documentos_extra, strict=True):
        documentos_por_id[int(identificador)] = documento

    bloques = []
    for marco in (train, test):
        textos = marco["text"].tolist()
        documentos = [
            documentos_por_id[int(identificador)] for identificador in marco["id"].astype(int)
        ]
        rasgos = rasgos_estructurados(
            textos, [texto.lower() for texto in textos], documentos
        )
        bloques.append(
            pd.concat(
                [marco[["id", "particion"]].reset_index(drop=True), rasgos.reset_index(drop=True)],
                axis=1,
            )
        )
    tabla = pd.concat(bloques, ignore_index=True)
    if tabla["id"].duplicated().any():
        raise RuntimeError("Hay ids duplicados en el dataset consolidado")
    if len(tabla) != len(train) + len(test):
        raise RuntimeError("El dataset consolidado no cubre train + test")
    log(
        f"features estructuradas: {len(tabla)} filas x {tabla.shape[1] - 2} columnas "
        f"(dev={int((tabla['particion'] == 'dev').sum())}, "
        f"holdout={int((tabla['particion'] == 'holdout').sum())}, "
        f"test={int((tabla['particion'] == 'test').sum())})"
    )
    return tabla


def escribir_manifest(
    tabla: pd.DataFrame,
    catalogo: pd.DataFrame,
    diagnostico_insultos: Sequence[dict[str, Any]],
    discrepancias: Sequence[str],
) -> None:
    """Escribe ``reports/v2_structured_features_manifest.md``.

    Args:
        tabla: dataset consolidado escrito.
        catalogo: catalogo de features.
        diagnostico_insultos: tamanos de los lexicos inducidos por pliegue.
        discrepancias: discrepancias detectadas en las verificaciones.
    """
    columnas = [columna for columna in tabla.columns if columna not in ("id", "particion")]
    asignadas = {
        columna for grupo in BLOQUES_ESTRUCTURADOS.values() for columna in grupo
    }
    pos_deps = [
        columna
        for columna in columnas
        if columna.startswith(("pos_", "dep_"))
    ]
    sin_grupo = [
        columna
        for columna in columnas
        if columna not in asignadas and columna not in set(pos_deps)
    ]
    if sin_grupo:
        raise RuntimeError(f"Columnas sin documentar en el manifest: {sin_grupo}")
    familias = catalogo["family"].value_counts().to_dict()
    lineas = [
        "# Manifest - features estructuradas v2 (FASE 55/89)",
        "",
        f"Fecha: {CONFIG['fecha']}. Artefacto: `features/v2_structured_features.parquet`.",
        f"Contenido: {len(tabla)} filas x {len(columnas)} features, mas `id` y `particion`",
        "(dev=1483, holdout=248, test=1154). Solo features deterministas y sin etiquetas.",
        "",
        "## Bloques y fuentes",
        "",
        "| bloque | columnas | fuente | fold-safety |",
        "| --- | --- | --- | --- |",
    ]
    for grupo, nombres in BLOQUES_ESTRUCTURADOS.items():
        lineas.append(
            f"| {grupo} | {len(nombres)} | {FUENTES_BLOQUES[grupo]} | determinista, sin etiquetas |"
        )
    lineas.append(
        f"| sintaxis (POS y dependencias) | {len(pos_deps)} | {FUENTES_BLOQUES['sintaxis']} | "
        "determinista, sin etiquetas |"
    )
    lineas.extend(
        [
            "",
            "## Columnas",
            "",
            "| columna | dtype | fuente | fold-safety |",
            "| --- | --- | --- | --- |",
        ]
    )
    for columna in columnas:
        if columna in asignadas:
            grupo = next(g for g, nombres in BLOQUES_ESTRUCTURADOS.items() if columna in nombres)
            fuente = FUENTES_BLOQUES[grupo]
        else:
            fuente = FUENTES_BLOQUES["sintaxis"]
        lineas.append(
            f"| {columna} | {tabla[columna].dtype} | {fuente} | determinista, sin etiquetas |"
        )
    lineas.extend(
        [
            "",
            "## Verificaciones ejecutadas",
            "",
            "- El dataset solo tiene `id` y `particion` como metadatos: no hay ninguna etiqueta HS/TR/AG.",
            "- Las filas fuera del desarrollo (holdout y test) se anotan con spaCy con los mismos parametros",
            "  de P3 (`es_core_news_md`, batch 64, sin NER ni vectores); la anotacion se cachea en",
            "  `features/v2_audit_annotations_extra.json.gz`. Es determinista y no usa etiquetas.",
            "- La regla estructural de colectivo, el detector de segunda persona y el de imperativo se",
            "  replicaron sobre la anotacion serializada y se compararon con las funciones de P2 en los 1483",
            "  documentos del desarrollo (resultado en la seccion de discrepancias).",
            "- Los tamanos de los lexicos de insulto inducidos por pliegue se contrastan con",
            "  `reports/targeting_v2.json` (P2): ver tabla siguiente.",
            "",
            "## Excluido y por que (honestidad > cobertura)",
            "",
            "- **Lexicos inducidos** (insulto, adjetivos negativos, afecto) y **target encoding**: dependen de",
            "  las etiquetas del pliegue de entrenamiento (fold-local), por lo que NO son deterministas ni",
            "  exportables a un parquet unico. Se usan solo en las auditorias sobre el desarrollo y estan en",
            "  el catalogo.",
            "- **Distancia target-insulto y pares colectivo-insulto**: requieren el lexico de insulto inducido;",
            "  existen solo como features fold-locales de E2-05/E2-06, no en este dataset.",
            "- **Canales sparse (TF-IDF word/char/POS/dep/morph/NB-SVM)**: vocabulario e IDF se ajustan por",
            "  pliegue (fold-local); no se consolidan como parquet.",
            "- **Emocion/afecto** (E2-09) y **espacios por etiqueta** (E2-21): fold-locales por lexicos",
            "  inducidos; se catalogan, no se consolidan.",
            "- **Umbrales y scores de modelos**: no son features; viven en los stores OOF.",
            "",
            "## Catalogo de features (FASE 55/88)",
            "",
            f"`reports/feature_catalog_v2.csv`: {len(catalogo)} features. Cobertura por familia: "
            + ", ".join(f"{familia}={conteo}" for familia, conteo in sorted(familias.items()))
            + ".",
            "",
            "## Lexicos de insulto fold-local usados en las auditorias",
            "",
            "| pliegue | n_train | n_validacion | insulto | adjetivos negativos | colectivos |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for fila in diagnostico_insultos:
        lineas.append(
            f"| {fila['pliegue']} | {fila['n_entrenamiento']} | {fila['n_validacion']} | "
            f"{fila['insulto']} | {fila['adjetivos_negativos']} | {fila['colectivos']} |"
        )
    lineas.extend(
        [
            "",
            "Contraste con `reports/targeting_v2.json` (diagnostico_por_pliegue): los tamanos deben coincidir;",
            "cualquier diferencia se registra abajo como discrepancia.",
            "",
            "## Discrepancias detectadas",
            "",
        ]
    )
    if discrepancias:
        lineas.extend(f"- {mensaje}" for mensaje in discrepancias)
    else:
        lineas.append(
            "- Ninguna: todos los canales OOF reproducen su F1 documentado con tolerancia "
            f"{CONFIG['tolerancia_f1']:g} y los detectores replicados coinciden con P2."
        )
    MANIFEST_ESTRUCTURADO.write_text("\n".join(lineas) + "\n", encoding="utf-8")
    log(f"escrito {MANIFEST_ESTRUCTURADO.name}: {len(lineas)} lineas")


# ===== SECCION 10: FASE 90 - registro en experiments.csv y contexto.md =====


def _fila_experimento(
    identificador: str,
    question: str,
    hypothesis: str,
    representation: str,
    model: str,
    hyperparameters: dict[str, Any],
    validation: str,
    f1: float,
    dimension: int,
    result: dict[str, Any],
    interpretation: str,
    duracion: float,
) -> dict[str, str]:
    """Construye una fila del log de experimentos para una auditoria.

    Args:
        identificador: id de la fila (``AUD-*``).
        question: pregunta que responde la auditoria.
        hypothesis: hipotesis evaluada.
        representation: representacion usada (analisis de OOF y rasgos).
        model: modelo auditado (no se entrena ninguno nuevo).
        hyperparameters: umbrales y parametros de la auditoria.
        validation: esquema de validacion del material auditado.
        f1: F1 de la referencia auditada (no de un modelo nuevo).
        dimension: numero de filas del artefacto generado.
        result: resultados principales.
        interpretation: lectura de los resultados.
        duracion: segundos del script completo.

    Returns:
        Fila serializada a texto.
    """
    return {
        "experiment_id": identificador,
        "competition": "KAGGLE_TOXIC_2026_PUJ",
        "question": question,
        "hypothesis": hypothesis,
        "preprocessing": "sin texto crudo en las salidas; rasgos estructurales deterministas y anotacion spaCy simbolica cacheada",
        "feature_groups": "AUDITORIA_V2",
        "representation": representation,
        "model": model,
        "hyperparameters": json.dumps(hyperparameters, ensure_ascii=False),
        "validation": validation,
        "f1": f"{f1:.4f}",
        "macro_f1": "",
        "weighted_f1": "",
        "precision": "",
        "recall": "",
        "training_time": f"{duracion:.1f}",
        "dimension": str(dimension),
        "result": json.dumps(result, ensure_ascii=False),
        "interpretation": interpretation,
        "decision": "artefacto de auditoria; sin modelo nuevo",
        "date": CONFIG["fecha"],
        "script": "experiments/22_artefactos_auditoria_v2.py",
        "notes": "el campo f1 repite la referencia auditada; no se entreno ningun modelo y el holdout congelado no se uso",
    }


def _serializar_fila_csv(fila: dict[str, str], columnas: Sequence[str]) -> str:
    """Serializa una fila a una linea CSV con quoting estandar.

    Args:
        fila: valores por columna.
        columnas: orden de columnas del encabezado.

    Returns:
        Linea CSV sin salto final.
    """
    buffer = StringIO()
    escritor = csv_module.writer(buffer, lineterminator="")
    escritor.writerow([fila[columna] for columna in columnas])
    return buffer.getvalue()


def anexar_experimentos(filas_nuevas: Sequence[dict[str, str]]) -> dict[str, Any]:
    """Anexa filas ``AUD-*`` a ``experiments.csv`` sin tocar las previas.

    El archivo se trata linea a linea: la cabecera y las filas previas quedan
    byte a byte iguales; solo se eliminan las filas ``AUD-*`` de una ejecucion
    anterior (idempotencia) y se anexan las nuevas.

    Args:
        filas_nuevas: filas a anexar.

    Returns:
        Diccionario con el resumen de la validacion del archivo.

    Raises:
        RuntimeError: si la validacion final falla.
    """
    contenido = EXPERIMENTS_CSV.read_text(encoding="utf-8")
    terminaba_en_salto = contenido.endswith("\n")
    lineas = contenido.splitlines()
    cabecera = lineas[0]
    columnas = next(csv_module.reader([cabecera]))
    previas = [linea for linea in lineas[1:] if not linea.startswith("AUD-")]
    ids_previos = {next(csv_module.reader([linea]))[0] for linea in previas if linea}
    nuevas = [_serializar_fila_csv(fila, columnas) for fila in filas_nuevas]
    salida = "\n".join([cabecera, *previas, *nuevas])
    if terminaba_en_salto:
        salida += "\n"
    EXPERIMENTS_CSV.write_text(salida, encoding="utf-8")

    verificacion = EXPERIMENTS_CSV.read_text(encoding="utf-8").splitlines()
    ids_totales = [next(csv_module.reader([linea]))[0] for linea in verificacion[1:] if linea]
    ids_aud = [identificador for identificador in ids_totales if identificador.startswith("AUD-")]
    resumen = {
        "filas_totales": len(ids_totales),
        "filas_previas": len(previas),
        "filas_nuevas": len(nuevas),
        "ids_unicos": len(set(ids_totales)) == len(ids_totales),
        "ids_aud": ids_aud,
        "cabecera_intacta": verificacion[0] == cabecera,
        "ids_previos_conservados": ids_previos.issubset(set(ids_totales)),
    }
    if resumen["filas_nuevas"] != 3 or len(ids_aud) != 3 or not resumen["ids_unicos"]:
        raise RuntimeError(f"La anexion a experiments.csv fallo: {resumen}")
    if not resumen["cabecera_intacta"] or not resumen["ids_previos_conservados"]:
        raise RuntimeError(f"Se altero contenido previo de experiments.csv: {resumen}")
    log(
        f"experiments.csv: {resumen['filas_previas']} filas previas + "
        f"{resumen['filas_nuevas']} nuevas = {resumen['filas_totales']} "
        f"(ids unicos: {resumen['ids_unicos']})"
    )
    return resumen


def escribir_contexto(
    tabla_interseccion: pd.DataFrame,
    tabla_tres_mal: pd.DataFrame,
    tabla_combinacion: pd.DataFrame,
    tabla_complementariedad: pd.DataFrame,
    tabla_catalogo: pd.DataFrame,
    features: pd.DataFrame,
    duracion: float,
) -> None:
    """Anade la seccion de artefactos de auditoria a ``reports/contexto.md``.

    Args:
        tabla_interseccion: tabla de interseccion de errores.
        tabla_tres_mal: tabla de tweets con las tres etiquetas mal.
        tabla_combinacion: tabla por combinacion de etiquetas.
        tabla_complementariedad: tabla de complementariedad.
        tabla_catalogo: catalogo de features.
        features: dataset consolidado.
        duracion: segundos de ejecucion.

    Raises:
        RuntimeError: si la seccion excede el maximo de 15 lineas.
    """
    encabezado = f"## Artefactos de auditoria v2 ({CONFIG['fecha']})"
    referencia = "v03_base_M2_union_lr_c2"
    meta = "meta_META-BASES-RIDGE"
    las3 = tabla_interseccion[tabla_interseccion["patron"] == "las_3_mal"].set_index("referencia")
    cobertura = tabla_catalogo["family"].value_counts().to_dict()
    canales = len({*tabla_complementariedad["canal_a"], *tabla_complementariedad["canal_b"]})
    similitud = mejor_par(tabla_complementariedad)
    lineas = [
        encabezado,
        (
            "Artefactos de las FASES 43/44/47/50/51/55/88/89/90 generados desde los OOF congelados del "
            "desarrollo (n=1483); sin entrenar modelos, sin submissions y sin usar el holdout ni el test."
        ),
        (
            f"Interseccion (FASE 43/44): {referencia} falla las 3 etiquetas en "
            f"{int(las3.loc[referencia, 'n'])} tweets ({las3.loc[referencia, 'fraccion']:.2%}, IC 95 % "
            f"[{las3.loc[referencia, 'ic_inferior_95']:.2%}, {las3.loc[referencia, 'ic_superior_95']:.2%}]); "
            f"{meta} en {int(las3.loc[meta, 'n'])} ({las3.loc[meta, 'fraccion']:.2%}, IC 95 % "
            f"[{las3.loc[meta, 'ic_inferior_95']:.2%}, {las3.loc[meta, 'ic_superior_95']:.2%}])."
        ),
        (
            "Contraste historico de FASE 43: 101 tweets (5.83 %) sobre train completo n=1731 con el "
            "protocolo historico (reports/error_analysis.md); el desarrollo corregido no es comparable 1:1."
        ),
        (
            f"Tres etiquetas mal (FASE 43): {len(tabla_tres_mal)} filas tweet x referencia "
            f"({int((tabla_tres_mal['referencia'] == referencia).sum())} + "
            f"{int((tabla_tres_mal['referencia'] == meta).sum())}); solo id y rasgos, sin texto crudo."
        ),
        (
            f"Error por combinacion (FASE 47): {len(tabla_combinacion)} filas (2 referencias x 7 "
            "combinaciones) con precision/recall/F1 por etiqueta y medias estructurales; en las etiquetas "
            "constantes del subconjunto precision = recall por construccion."
        ),
        (
            f"Complementariedad (FASE 50/51/91): {len(tabla_complementariedad)} parejas de {canales} "
            "canales OOF; el ensemble_gain solo se copia si ya estaba medido (media con M2 a 0.5) y no se "
            "recalculo ninguno."
        ),
        (
            "Mejor pareja por correccion mutua simetrica: "
            + (
                f"{similitud['canal_a']} / {similitud['canal_b']} "
                f"({similitud['correccion_a_sobre_b']:.1%} y {similitud['correccion_b_sobre_a']:.1%} de "
                "los errores del otro)."
                if similitud
                else "sin datos."
            )
        ),
        (
            "Catalogo (FASE 55/88): "
            + ", ".join(f"{familia}={conteo}" for familia, conteo in sorted(cobertura.items()))
            + f" ({len(tabla_catalogo)} features leidas del codigo de experiments/14 a 21)."
        ),
        (
            "Dataset estructurado (FASE 55/89): features/v2_structured_features.parquet con "
            f"{len(features)} filas (dev+holdout+test) x {features.shape[1] - 2} features deterministas sin "
            "etiquetas; manifest en reports/v2_structured_features_manifest.md con las exclusiones "
            "documentadas."
        ),
        (
            "Registro (FASE 90): filas AUD-01, AUD-02 y AUD-03 en experiments.csv (solo append) con "
            "decision 'artefacto de auditoria; sin modelo nuevo'."
        ),
        (
            "Honestidad: ninguna cifra se ajusto para cuadrar con lo documentado; las discrepancias se "
            "registran en el manifest. El holdout congelado sigue intacto y no se genero ninguna submission."
        ),
        f"Duracion: {duracion:.1f} s. Script: experiments/22_artefactos_auditoria_v2.py.",
    ]
    if len(lineas) > 15:
        raise RuntimeError(f"La seccion de contexto excede 15 lineas: {len(lineas)}")
    contenido = CONTEXTO_MD.read_text(encoding="utf-8")
    if encabezado in contenido:
        prefijo = contenido.split(encabezado)[0].rstrip() + "\n\n"
    else:
        prefijo = contenido.rstrip() + "\n\n"
    CONTEXTO_MD.write_text(prefijo + "\n".join(lineas) + "\n", encoding="utf-8")
    log(f"contexto.md actualizado: seccion de {len(lineas)} lineas")


# ===== SECCION 11: orquestacion =====


def calcular_pliegues(desarrollo: pd.DataFrame) -> NDArray[np.int_]:
    """Calcula el pliegue corregido (1..5) de cada fila del desarrollo.

    Args:
        desarrollo: DataFrame del desarrollo.

    Returns:
        Arreglo ``(n,)`` con el numero de pliegue de cada fila.
    """
    train = data_module.read_train()
    grupos_completos = grupos_fusionados(train)
    mapa_grupo = {
        int(identificador): int(grupo)
        for identificador, grupo in zip(train["id"].astype(int), grupos_completos, strict=True)
    }
    grupos = np.array(
        [mapa_grupo[int(identificador)] for identificador in desarrollo["id"].astype(int)]
    )
    pliegues = pliegues_corregidos(desarrollo, grupos)
    pliegue_de_fila = np.zeros(len(desarrollo), dtype=int)
    for numero, (_, idx_validacion) in enumerate(pliegues, start=1):
        pliegue_de_fila[idx_validacion] = numero
    log(f"pliegues corregidos v2: {len(pliegues)} (grupos={len(np.unique(grupos))})")
    return pliegue_de_fila


def main() -> None:
    """Ejecuta las auditorias y escribe todos los artefactos."""
    inicio = time.time()
    log(f"inicio P9 - artefactos de auditoria v2 ({CONFIG['fecha']})")
    log(f"constantes estructurales importadas de {ORIGEN_CONSTANTES}")
    desarrollo = cargar_desarrollo()
    y_verdadera = desarrollo[list(ETIQUETAS)].to_numpy()
    pliegue_de_fila = calcular_pliegues(desarrollo)

    canales = cargar_canales(desarrollo, y_verdadera)
    referencias = cargar_referencias_canales()
    discrepancias = verificar_canales(canales, referencias, y_verdadera)
    referencias_usadas = ("v03_base_M2_union_lr_c2", "meta_META-BASES-RIDGE")

    documentos_dev = cargar_anotaciones(desarrollo)
    textos_dev = desarrollo["text"].tolist()
    rasgos = rasgos_estructurados(
        textos_dev, [texto.lower() for texto in textos_dev], documentos_dev
    )
    log(f"rasgos estructurales del desarrollo: {rasgos.shape[0]} x {rasgos.shape[1]}")
    discrepancias.extend(verificar_detectores(rasgos, documentos_dev))

    pliegues = [
        (
            np.where(pliegue_de_fila != numero)[0],
            np.where(pliegue_de_fila == numero)[0],
        )
        for numero in sorted(np.unique(pliegue_de_fila).tolist())
    ]
    insultos, diagnostico_insultos = insultos_fold_local(
        desarrollo, documentos_dev, pliegues, y_verdadera
    )
    log(
        "lexicos de insulto fold-local: "
        + "; ".join(
            f"pliegue {fila['pliegue']} insulto={fila['insulto']} "
            f"adj_neg={fila['adjetivos_negativos']} colectivos={fila['colectivos']}"
            for fila in diagnostico_insultos
        )
    )

    tabla_interseccion = escribir_interseccion(canales, referencias_usadas, y_verdadera)
    tabla_tres_mal = escribir_tres_mal(
        canales, referencias_usadas, desarrollo, y_verdadera, pliegue_de_fila, rasgos, insultos
    )
    tabla_combinacion = escribir_combinaciones(
        canales, referencias_usadas, desarrollo, y_verdadera, rasgos, insultos
    )
    tabla_complementariedad = escribir_complementariedad(canales, referencias, y_verdadera)
    tabla_catalogo = escribir_catalogo()

    features = construir_features_estructuradas()
    PARQUET_ESTRUCTURADO.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(PARQUET_ESTRUCTURADO, index=False)
    log(f"escrito {PARQUET_ESTRUCTURADO.name}: {features.shape[0]} x {features.shape[1]}")
    escribir_manifest(features, tabla_catalogo, diagnostico_insultos, discrepancias)

    duracion = time.time() - inicio
    las3 = tabla_interseccion[tabla_interseccion["patron"] == "las_3_mal"].set_index("referencia")
    mejor = mejor_par(tabla_complementariedad)
    filas_experimentos = [
        _fila_experimento(
            "AUD-01",
            "Cuantos tweets fallan exactamente las tres etiquetas y que patrones de error coexisten en el desarrollo corregido?",
            "Los errores de las tres etiquetas son un fenomeno prioritario y de baja frecuencia (FASE 43/44)",
            "OOF congelado de V03 (M2 con umbrales 0.44/0.51/0.46) y del meta v2 (Ridge a 0.5); matriz de aciertos por tweet",
            "sin modelo nuevo (auditoria de OOF)",
            {
                "umbrales_v03": CONFIG["umbrales_v03"],
                "umbral_meta": CONFIG["umbral_defecto"],
                "n_bootstrap": CONFIG["n_bootstrap"],
            },
            "5 pliegues corregidos v2 (OOF congelado); bootstrap 2000 a nivel de tweet",
            CONFIG["f1_v03_corrected"],
            len(tabla_interseccion),
            {
                "v03_corrected_las_3_mal": int(las3.loc[referencias_usadas[0], "n"]),
                "v03_corrected_fraccion": float(las3.loc[referencias_usadas[0], "fraccion"]),
                "v03_corrected_ic95": [
                    float(las3.loc[referencias_usadas[0], "ic_inferior_95"]),
                    float(las3.loc[referencias_usadas[0], "ic_superior_95"]),
                ],
                "meta_v2_las_3_mal": int(las3.loc[referencias_usadas[1], "n"]),
                "meta_v2_fraccion": float(las3.loc[referencias_usadas[1], "fraccion"]),
                "meta_v2_ic95": [
                    float(las3.loc[referencias_usadas[1], "ic_inferior_95"]),
                    float(las3.loc[referencias_usadas[1], "ic_superior_95"]),
                ],
                "historico_101_sobre_1731": round(float(CONFIG["historico_fraccion"]), 6),
                "discrepancias": list(discrepancias),
            },
            "El nucleo duro de error se concentra en las tres etiquetas mal; la comparacion con el 5.83 % "
            "historico no es directa (base n=1731 vs desarrollo n=1483 y protocolo distinto), se reporta "
            "por separado con su propio IC bootstrap.",
            duracion,
        ),
        _fila_experimento(
            "AUD-02",
            "Que fenomeno diferencia las combinaciones verdaderas de etiquetas (000/100/101/110/111/001) en el error del modelo?",
            "Cada combinacion verdadera tiene un perfil estructural propio que explica el error por etiqueta (FASE 47)",
            "mismo OOF auditado + rasgos estructurales deterministas (longitud, url, mencion, hashtag, colectivo, cita, negacion, insulto fold-local)",
            "sin modelo nuevo (auditoria de OOF)",
            {"umbrales_v03": CONFIG["umbrales_v03"], "umbral_meta": CONFIG["umbral_defecto"]},
            "5 pliegues corregidos v2 (OOF congelado); insulto con lexico inducido sin ver el pliegue evaluado",
            CONFIG["f1_v03_corrected"],
            len(tabla_combinacion),
            {
                "combinaciones": list(CONFIG["combinaciones_orden"]),
                "referencias": list(referencias_usadas),
                "n_por_combinacion": {
                    combinacion: int(
                        (tabla_combinacion["combinacion"] == combinacion).sum()
                    )
                    for combinacion in CONFIG["combinaciones_orden"]
                },
            },
            "El perfil estructural por combinacion muestra donde se concentra el error (cita/atribucion, "
            "colectivo, longitud) sin usar etiquetas como feature.",
            duracion,
        ),
        _fila_experimento(
            "AUD-03",
            "Que canales OOF son complementarios entre si y que gana cada pareja al corregir errores del otro?",
            "Hay canales con F1 menor pero complementarios: acuerdo, correlacion de errores y correccion mutua lo cuantifican (FASE 50/51/91)",
            "OOF congelado de los canales disponibles (4 bases + M2 a 0.5, meta, targeting, sintaxis, cita/negacion, emocion, NB-SVM, ortografia)",
            "sin modelo nuevo (auditoria de OOF; ensemble_gain solo si ya estaba medido)",
            {
                "umbrales_v03": CONFIG["umbrales_v03"],
                "umbral_resto": CONFIG["umbral_defecto"],
                "ventana_frontera": CONFIG["ventana_frontera"],
            },
            "5 pliegues corregidos v2 (OOF congelado); comparaciones sobre las n*3 celdas",
            CONFIG["f1_meta_v2"],
            len(tabla_complementariedad),
            {
                "canales": len(canales),
                "parejas": len(tabla_complementariedad),
                "f1_meta_v2": CONFIG["f1_meta_v2"],
                "mejor_par_correccion_mutua": mejor,
            },
            "La diversidad de canales se mide con acuerdo, correlacion de errores, desacuerdo de frontera y "
            "correccion mutua; el ensemble_gain se reporta solo donde ya estaba medido para no inventar "
            "ganancias.",
            duracion,
        ),
    ]
    anexar_experimentos(filas_experimentos)
    escribir_contexto(
        tabla_interseccion,
        tabla_tres_mal,
        tabla_combinacion,
        tabla_complementariedad,
        tabla_catalogo,
        features,
        duracion,
    )
    log(f"fin P9 en {duracion:.1f} s")
    log(
        "resumen: las_3_mal "
        + "; ".join(
            f"{nombre}={int(las3.loc[nombre, 'n'])} ({las3.loc[nombre, 'fraccion']:.4f})"
            for nombre in referencias_usadas
        )
    )


if __name__ == "__main__":
    try:
        main()
    except Exception:  # noqa: BLE001 - reporte explicito y salida con codigo 1
        import traceback

        traceback.print_exc()
        sys.exit(1)
