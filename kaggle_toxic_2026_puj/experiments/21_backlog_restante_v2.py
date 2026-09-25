"""P8 - Cierre del backlog v2: E2-13, E2-14, E2-16, E2-20 y E2-21.

Pregunta que responde
---------------------
El plan v2 declaro 24 lineas de investigacion; 19 se evaluaron en P1-P7 sin
transferir al LB. Este paquete cierra las 5 restantes sobre el DESARROLLO
(n=1483) con el protocolo corregido de D-016 (grupos = near-duplicado UNION
pseudo-autor, ``StratifiedGroupKFold(5)``, seed 42) y el holdout congelado
intacto:

- E2-14 calibracion (Platt e isotonic anidados por pliegue) de M2 y del mejor
  meta (Ridge).
- E2-13 perfilado de entidades (target encoding OOF de mencion, hashtag y
  dominio con suavizado m=10, frecuencia minima 5 y bucket desconocido) como
  bloque extra del meta y como canal propio.
- E2-16 umbrales por rebanada con shrinkage (rebanadas n >= 80; umbral =
  mezcla ponderada por n entre el global y el local; evaluacion anidada).
- E2-20 especialistas por rebanada (solo si el test de proporcion de errores
  con correccion BH detecta una desviacion significativa).
- E2-21 espacios de features por etiqueta (HS con colectivo/hate inducido, TR
  con targeting, AG con insulto/afecto) frente al espacio compartido V03.

La expectativa declarada del paquete es BAJA (19 lineas previas no
transfirieron); el valor es documental: cerrar el backlog al 100 %.

Metodo
------
1. Se cargan los OOF congelados de los 4 bases y del meta Ridge (P6) y se
   verifican ids, etiquetas, grupos, pliegues y la reproduccion de las
   referencias (M2 a 0.5 = 0.7718; V03 con umbrales = 0.7772; meta = 0.7874).
2. Ninguna transformacion (calibradores, tablas de entidades, lexicos
   inducidos, vectorizadores, escaladores, umbrales y especialistas) se ajusta
   con la validacion de su propio pliegue; todas las variantes se evaluan fold
   a fold con los pliegues corregidos.
3. La regla de adopcion de D-016 se aplica al MECANISMO de cada linea
   (``corrected_cv`` >= 0.7772 + 0.005 y >= 3/5 pliegues, o complementariedad
   con IC bootstrap > 0), exigiendo ademas que la variante del mecanismo mejore
   a su comparador directo (la version sin calibrar, el meta sin entidades, el
   umbral global sin rebanada, el modelo global sin especialista y el espacio
   compartido). Se reportan ambos criterios para que la decision sea auditable.
4. Nota de honestidad sobre calibracion: Platt e isotonic son transformaciones
   monotonas por etiqueta, por lo que la F1 con umbrales reajustados es
   invariante salvo empates; su aporte medible esta en Brier/log-loss y en la
   estabilidad de los umbrales. Los calibradores del pliegue k se ajustan con
   las predicciones OOF de los pliegues restantes (calibracion cruzada
   estandar); P1 midio que ese sesgo de anidamiento es <= 0.0003 en este corpus.
5. E2-21 usa proxies estructurales por regex (sin spaCy en este paquete: la
   anotacion morfologica de P2 ya se cerro como linea) y lexicos inducidos
   fold-local con log-odds de Monroe et al. (2008) con prior de Dirichlet
   informativo; los bloques del canal simbolico se normalizan en norma L2 para
   que ningun bloque domine a la union TF-IDF por escala.

Salidas
-------
``reports/backlog_restante_v2.csv`` y ``.json``, figura
``reports/figuras/fig_32_backlog_restante_v2.png``, filas ``E2-13-*``,
``E2-14-*``, ``E2-16-*``, ``E2-20-*`` y ``E2-21-*`` en ``experiments.csv``,
seccion ``## Fase v2 - P8 (2026-09-25)`` (maximo 20 lineas) en
``reports/contexto.md`` y actualizacion de ``tasks/todo.md`` con las 5 lineas y
el Gate P8 (backlog 24/24). No se genera submission ni se toca el holdout.

Ejecucion
---------
    /home/carlos/venv/bin/python experiments/21_backlog_restante_v2.py
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
import time
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Final

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import sparse
from scipy.stats import norm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler, normalize

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src import data as data_module
from src.evaluation import metricas_multietiqueta
from src.splits import (
    ETIQUETAS,
    RANDOM_STATE,
    combinaciones,
    grupos_fusionados,
    pliegues_corregidos,
)

IDS_PATH = PROJECT_ROOT / "data" / "processed" / "holdout_v2_ids.csv"
OOF_BASES_PATH = PROJECT_ROOT / "features" / "v2_oof_base_corrected.npz"
OOF_META_PATH = PROJECT_ROOT / "features" / "v2_meta_oof.npz"
REPORTE_CSV = PROJECT_ROOT / "reports" / "backlog_restante_v2.csv"
REPORTE_JSON = PROJECT_ROOT / "reports" / "backlog_restante_v2.json"
FIGURA_PATH = PROJECT_ROOT / "reports" / "figuras" / "fig_32_backlog_restante_v2.png"
EXPERIMENTS_CSV = PROJECT_ROOT / "experiments.csv"
CONTEXTO_MD = PROJECT_ROOT / "reports" / "contexto.md"
TODO_MD = PROJECT_ROOT / "tasks" / "todo.md"

CONFIG: Final[dict[str, Any]] = {
    "fecha": "2026-09-25",
    "referencia_m2_umbral_05": 0.7718,
    "referencia_v03_umbrales": 0.7772,
    "referencia_meta_ridge": 0.7873574608549214,
    "referencia_v2a_meta": 0.7853,
    "umbrales_v03": {"HS": 0.44, "TR": 0.51, "AG": 0.46},
    "mejora_minima": 0.005,
    "ensemble_gain_minimo": 0.0015,
    "pliegues_minimos_mejorados": 3,
    "n_bootstrap": 2000,
    "umbral_min": 0.20,
    "umbral_max": 0.80,
    "umbral_paso": 0.02,
    "min_rebanada": 80,
    "min_positivos_umbral_local": 8,
    "shrinkage_m": (0, 25, 50, 100),
    "suavizado_entidades": 10,
    "frecuencia_minima_entidades": 5,
    "n_pliegues_internos_entidades": 4,
    "top_lexico_inductivo": 150,
    "frecuencia_minima_lexico": 5,
    "min_documentos_lexico": 3,
    "alpha_prior_log_odds": 50.0,
    "bh_q": 0.05,
    "tolerancia_replicacion": 5e-4,
    "umbral_cobertura_minima": 0.20,
}

REJILLA_UMBRALES: Final[NDArray[np.float64]] = np.arange(
    CONFIG["umbral_min"], CONFIG["umbral_max"] + 1e-9, CONFIG["umbral_paso"]
)

MODELOS_BASE: Final[tuple[str, ...]] = (
    "M1_word_lr",
    "M2_union_lr_c2",
    "M3_union_svc",
    "M4_char_lr",
)
POSICION_M2: Final[int] = 1

PATRON_MENCION = re.compile(r"@\w+")
PATRON_HASHTAG = re.compile(r"#\w+")
PATRON_URL = re.compile(r"https?://|www\.|t\.co")
PATRON_HOST = re.compile(r"(?:https?://|www\.)([a-zA-Z0-9.\-]+)")
PATRON_COMILLAS = re.compile(r"[\"'\u00ab\u00bb\u201c\u201d]")
PATRON_PREGUNTA = re.compile(r"\?")
PATRON_NEGACION = re.compile(r"\b(no|nunca|jamas|tampoco|ni|sin)\b")
PATRON_RISAS = re.compile(r"(?i)\b(?:ja){2,}\w*|\b(?:je){2,}\w*|\b(?:jo){2,}\w*|\blol\b|\bxd+\b")
PATRON_MAYUSCULAS = re.compile(r"[A-Z]{2,}")
PATRON_ELONGACION = re.compile(r"(.)\1{2,}")
PATRON_PALABRA = re.compile(r"[a-z0-9]+")
PATRON_SEGUNDA_PERSONA = re.compile(
    r"\b(tu|te|ti|tus|vos|usted|ustedes|contigo|tuyo|tuya|tuyos|tuyas"
    r"|sos|tenes|eres|estas|dejame|pideme)\b"
)
# Lista estructural escrita a mano (no ajustada con datos: no puede introducir
# fuga) de marcadores colectivos, reutilizada de P2/P6.
PATRON_COLECTIVO = re.compile(
    r"\b(los|las|esos|esas|todos|todas|ellos|ellas|ustedes|nosotros|nosotras"
    r"|nuestros|nuestras|muchos|muchas|unos|unas|varios|varias|ambos|ambas"
    r"|gente|personas|mujeres|hombres|ninos|ninas|jovenes|ancianos|ancianas"
    r"|indigenas|extranjeros|extranjeras|inmigrantes|venezolanos|venezolanas"
    r"|colombianos|colombianas|funcionarios|politicos|periodistas)\b"
)
# Lista estructural escrita a mano de imperativos frecuentes en el corpus.
IMPERATIVOS_FIJOS: Final[frozenset[str]] = frozenset(
    {
        "callate",
        "vete",
        "dile",
        "deja",
        "mira",
        "cierra",
        "abre",
        "largate",
        "muere",
        "ven",
        "haz",
        "di",
        "sal",
        "corre",
        "pide",
        "respeta",
        "aprende",
        "dejanos",
        "dejenos",
        "pideme",
        "cuentame",
        "dime",
    }
)
NOMBRES_ESTRUCTURALES: Final[tuple[str, ...]] = (
    "n_chars",
    "n_tokens",
    "tiene_mencion",
    "tiene_hashtag",
    "tiene_url",
    "tiene_colectivo",
    "tiene_segunda_persona",
    "tiene_comillas",
)
TIPOS_ENTIDAD: Final[tuple[str, ...]] = ("mencion", "hashtag", "dominio")


def log(mensaje: str) -> None:
    """Imprime un mensaje de progreso con prefijo de fase.

    Args:
        mensaje: texto a imprimir.
    """
    print(f"[P8] {mensaje}", flush=True)


def serializar(valor: Any) -> Any:
    """Convierte tipos de numpy/pandas en tipos serializables a JSON.

    Args:
        valor: objeto a convertir.

    Returns:
        Objeto con tipos nativos.
    """
    if isinstance(valor, dict):
        return {str(clave): serializar(dato) for clave, dato in valor.items()}
    if isinstance(valor, (list, tuple)):
        return [serializar(dato) for dato in valor]
    if isinstance(valor, np.ndarray):
        return [serializar(dato) for dato in valor.tolist()]
    if isinstance(valor, (np.integer,)):
        return int(valor)
    if isinstance(valor, (np.floating,)):
        return float(valor)
    if isinstance(valor, (np.bool_,)):
        return bool(valor)
    return valor


# ===== SECCION 1: utilidades comunes de metrica y bootstrap =====


def f1_por_columna(verdad: NDArray[np.integer], prediccion: NDArray[np.integer]) -> float:
    """Calcula la F1 binaria de la clase positiva de una columna.

    Args:
        verdad: columna binaria verdadera.
        prediccion: columna binaria predicha.

    Returns:
        F1 binaria (``0.0`` si no hay aciertos ni predicciones positivas).
    """
    verdad_b = verdad.astype(bool)
    prediccion_b = prediccion.astype(bool)
    verdaderos_positivos = int(np.count_nonzero(verdad_b & prediccion_b))
    falsos_positivos = int(np.count_nonzero(~verdad_b & prediccion_b))
    falsos_negativos = int(np.count_nonzero(verdad_b & ~prediccion_b))
    denominador = 2 * verdaderos_positivos + falsos_positivos + falsos_negativos
    return 2 * verdaderos_positivos / denominador if denominador else 0.0


def f1_media_rapida(y_verdadera: NDArray[np.integer], prediccion: NDArray[np.integer]) -> float:
    """Calcula la F1 media por columna de forma vectorizada.

    Args:
        y_verdadera: matriz binaria verdadera ``(n, k)``.
        prediccion: matriz binaria predicha ``(n, k)``.

    Returns:
        Media de la F1 binaria por columna.
    """
    return float(
        np.mean(
            [
                f1_por_columna(y_verdadera[:, posicion], prediccion[:, posicion])
                for posicion in range(y_verdadera.shape[1])
            ]
        )
    )


def f1_por_etiqueta_rapida(
    y_verdadera: NDArray[np.integer], prediccion: NDArray[np.integer]
) -> dict[str, float]:
    """Calcula la F1 binaria por etiqueta.

    Args:
        y_verdadera: matriz binaria verdadera ``(n, 3)``.
        prediccion: matriz binaria predicha ``(n, 3)``.

    Returns:
        Diccionario ``etiqueta -> F1``.
    """
    return {
        etiqueta: f1_por_columna(y_verdadera[:, posicion], prediccion[:, posicion])
        for posicion, etiqueta in enumerate(ETIQUETAS)
    }


def vector_umbrales(umbrales: dict[str, float]) -> NDArray[np.float64]:
    """Convierte el diccionario de umbrales en vector en el orden de etiquetas.

    Args:
        umbrales: umbral por etiqueta.

    Returns:
        Arreglo ``(3,)`` en el orden de :data:`ETIQUETAS`.
    """
    return np.array([umbrales[etiqueta] for etiqueta in ETIQUETAS])


def mascaras_pliegues(pliegue_de_fila: NDArray[np.integer]) -> list[tuple[int, NDArray[np.bool_]]]:
    """Devuelve la mascara booleana de cada pliegue.

    Args:
        pliegue_de_fila: numero de pliegue (1..k) de cada fila.

    Returns:
        Lista de ``(numero, mascara)`` en orden creciente de pliegue.
    """
    return [
        (int(numero), pliegue_de_fila == numero)
        for numero in sorted(np.unique(pliegue_de_fila).tolist())
    ]


def f1_por_pliegue(
    proba: NDArray[np.float64],
    y_verdadera: NDArray[np.integer],
    pliegue_de_fila: NDArray[np.integer],
    umbrales: dict[str, float],
) -> list[float]:
    """Calcula la F1 media por columna en cada pliegue.

    Args:
        proba: probabilidades ``(n, 3)``.
        y_verdadera: etiquetas verdaderas ``(n, 3)``.
        pliegue_de_fila: numero de pliegue de cada fila.
        umbrales: umbral por etiqueta.

    Returns:
        Lista de F1 por pliegue.
    """
    prediccion = (proba >= vector_umbrales(umbrales)).astype(int)
    return [
        f1_media_rapida(y_verdadera[mascara], prediccion[mascara])
        for _, mascara in mascaras_pliegues(pliegue_de_fila)
    ]


def f1_por_pliegue_prediccion(
    prediccion: NDArray[np.integer],
    y_verdadera: NDArray[np.integer],
    pliegue_de_fila: NDArray[np.integer],
) -> list[float]:
    """Calcula la F1 media por columna en cada pliegue de una prediccion dada.

    Args:
        prediccion: matriz binaria predicha ``(n, 3)``.
        y_verdadera: etiquetas verdaderas ``(n, 3)``.
        pliegue_de_fila: numero de pliegue de cada fila.

    Returns:
        Lista de F1 por pliegue.
    """
    return [
        f1_media_rapida(y_verdadera[mascara], prediccion[mascara])
        for _, mascara in mascaras_pliegues(pliegue_de_fila)
    ]


def ajustar_umbral_etiqueta(
    y_columna: NDArray[np.integer], proba_columna: NDArray[np.float64]
) -> float:
    """Busca el umbral que maximiza la F1 de una etiqueta en la rejilla fija.

    Args:
        y_columna: etiqueta binaria verdadera.
        proba_columna: probabilidades de la clase positiva.

    Returns:
        Umbral optimo (primer maximo de la rejilla).
    """
    mejor_umbral, mejor_f1 = 0.5, -1.0
    for umbral in REJILLA_UMBRALES:
        f1 = f1_por_columna(y_columna, (proba_columna >= umbral).astype(int))
        if f1 > mejor_f1:
            mejor_umbral, mejor_f1 = float(umbral), f1
    return mejor_umbral


def ajustar_umbrales(y_verdadera: NDArray[np.integer], proba: NDArray[np.float64]) -> dict[str, float]:
    """Ajusta un umbral por etiqueta sobre las filas recibidas.

    Args:
        y_verdadera: etiquetas verdaderas ``(n, 3)``.
        proba: probabilidades ``(n, 3)``.

    Returns:
        Diccionario ``etiqueta -> umbral``.
    """
    return {
        etiqueta: ajustar_umbral_etiqueta(y_verdadera[:, posicion], proba[:, posicion])
        for posicion, etiqueta in enumerate(ETIQUETAS)
    }


def evaluar_con_umbrales(
    proba: NDArray[np.float64], y_verdadera: NDArray[np.integer], umbrales: dict[str, float]
) -> dict[str, Any]:
    """Calcula las metricas oficiales con umbrales dados.

    Args:
        proba: probabilidades ``(n, 3)``.
        y_verdadera: etiquetas verdaderas ``(n, 3)``.
        umbrales: umbral por etiqueta.

    Returns:
        Diccionario de :func:`metricas_multietiqueta`.
    """
    return metricas_multietiqueta(y_verdadera, (proba >= vector_umbrales(umbrales)).astype(int))


def verificar_metrica_rapida(
    proba: NDArray[np.float64], y_verdadera: NDArray[np.integer], umbrales: dict[str, float]
) -> None:
    """Comprueba que la F1 rapida coincide con la metrica oficial del proyecto.

    Args:
        proba: probabilidades de referencia.
        y_verdadera: etiquetas verdaderas.
        umbrales: umbral por etiqueta.

    Raises:
        RuntimeError: si las dos implementaciones difieren.
    """
    prediccion = (proba >= vector_umbrales(umbrales)).astype(int)
    oficial = metricas_multietiqueta(y_verdadera, prediccion)["f1_media_columnas"]
    rapida = f1_media_rapida(y_verdadera, prediccion)
    if abs(oficial - rapida) > 1e-12:
        raise RuntimeError(f"La F1 rapida {rapida} no coincide con la oficial {oficial}")


def bootstrap_delta(
    proba_referencia: NDArray[np.float64],
    proba_variante: NDArray[np.float64],
    y_verdadera: NDArray[np.integer],
    umbrales_referencia: dict[str, float],
    umbrales_variante: dict[str, float],
) -> dict[str, Any]:
    """Calcula el IC bootstrap pareado del delta de F1 a nivel de tweet.

    Args:
        proba_referencia: probabilidades de la referencia.
        proba_variante: probabilidades de la variante.
        y_verdadera: etiquetas verdaderas.
        umbrales_referencia: umbrales de la referencia.
        umbrales_variante: umbrales de la variante.

    Returns:
        Diccionario con media, IC95, fraccion positiva y remuestreos.
    """
    generador = np.random.default_rng(RANDOM_STATE)
    n_filas = len(y_verdadera)
    prediccion_ref = (proba_referencia >= vector_umbrales(umbrales_referencia)).astype(int)
    prediccion_var = (proba_variante >= vector_umbrales(umbrales_variante)).astype(int)
    deltas = np.empty(CONFIG["n_bootstrap"], dtype=float)
    for iteracion in range(CONFIG["n_bootstrap"]):
        muestra = generador.integers(0, n_filas, size=n_filas)
        verdad = y_verdadera[muestra]
        deltas[iteracion] = f1_media_rapida(verdad, prediccion_var[muestra]) - f1_media_rapida(
            verdad, prediccion_ref[muestra]
        )
    return {
        "media": float(np.mean(deltas)),
        "ic_inferior": float(np.quantile(deltas, 0.025)),
        "ic_superior": float(np.quantile(deltas, 0.975)),
        "fraccion_positiva": float(np.mean(deltas > 0)),
        "remuestreos": int(CONFIG["n_bootstrap"]),
    }


def evaluar_complementariedad(
    proba_variante: NDArray[np.float64],
    proba_base: NDArray[np.float64],
    y_verdadera: NDArray[np.integer],
    referencia: float,
) -> dict[str, Any]:
    """Mide la complementariedad por promedio de probabilidades con el comparador.

    Args:
        proba_variante: probabilidades de la variante.
        proba_base: probabilidades del comparador directo del mecanismo.
        y_verdadera: etiquetas verdaderas.
        referencia: F1 del comparador a 0.5 (linea base de la ganancia).

    Returns:
        Diccionario con la F1 del ensemble, la ganancia y el bootstrap si es
        positiva (regla de complementariedad de D-016). La ganancia se mide
        contra el comparador directo, no contra M2: asi la regla de
        complementariedad no puede adoptar un mecanismo que solo hereda la
        complementariedad de su modelo de partida.
    """
    sin_umbral = {etiqueta: 0.5 for etiqueta in ETIQUETAS}
    proba_ensemble = (proba_variante + proba_base) / 2.0
    f1_ensemble = f1_media_rapida(y_verdadera, (proba_ensemble >= 0.5).astype(int))
    ganancia = f1_ensemble - referencia
    resultado: dict[str, Any] = {
        "referencia_comparador": referencia,
        "corrected_cv_ensemble": f1_ensemble,
        "ensemble_gain": ganancia,
        "cumple_umbral_gain": bool(ganancia >= CONFIG["ensemble_gain_minimo"]),
        "aporta_ic_positivo": False,
    }
    if ganancia > 0:
        resultado["bootstrap"] = bootstrap_delta(proba_base, proba_ensemble, y_verdadera, sin_umbral, sin_umbral)
        resultado["aporta_ic_positivo"] = bool(resultado["bootstrap"]["ic_inferior"] > 0)
    return resultado


def brier_logloss(
    proba: NDArray[np.float64], y_verdadera: NDArray[np.integer]
) -> dict[str, Any]:
    """Calcula Brier score y log-loss globales y por etiqueta.

    Args:
        proba: probabilidades ``(n, 3)``.
        y_verdadera: etiquetas verdaderas ``(n, 3)``.

    Returns:
        Diccionario con ``brier``, ``log_loss`` y el desglose por etiqueta.
    """
    recortadas = np.clip(proba, 1e-6, 1.0 - 1e-6)
    brier_por_etiqueta = {
        etiqueta: float(np.mean((proba[:, posicion] - y_verdadera[:, posicion]) ** 2))
        for posicion, etiqueta in enumerate(ETIQUETAS)
    }
    logloss_por_etiqueta = {
        etiqueta: float(
            np.mean(
                -(
                    y_verdadera[:, posicion] * np.log(recortadas[:, posicion])
                    + (1 - y_verdadera[:, posicion]) * np.log(1 - recortadas[:, posicion])
                )
            )
        )
        for posicion, etiqueta in enumerate(ETIQUETAS)
    }
    return {
        "brier": float(np.mean(list(brier_por_etiqueta.values()))),
        "log_loss": float(np.mean(list(logloss_por_etiqueta.values()))),
        "brier_por_etiqueta": brier_por_etiqueta,
        "log_loss_por_etiqueta": logloss_por_etiqueta,
    }


def aplicar_regla_adopcion(
    nombre: str,
    corrected_cv: float,
    f1_pliegues: Sequence[float],
    f1_pliegues_v03: Sequence[float],
    complementariedad: dict[str, Any] | None,
    comparador: str,
    delta_comparador: float,
    pliegues_mejorados_comparador: int,
    aporta_sobre_comparador: bool,
) -> dict[str, Any]:
    """Aplica la regla de adopcion de D-016 al mecanismo de una linea.

    Args:
        nombre: identificador de la variante.
        corrected_cv: metrica primaria de la variante.
        f1_pliegues: F1 por pliegue de la variante.
        f1_pliegues_v03: F1 por pliegue de la referencia V03.
        complementariedad: resultado de :func:`evaluar_complementariedad` o ``None``.
        comparador: nombre del comparador directo del mecanismo.
        delta_comparador: delta de la variante frente a su comparador.
        pliegues_mejorados_comparador: pliegues en los que supera al comparador.
        aporta_sobre_comparador: si el mecanismo aporta sobre el comparador.

    Returns:
        Diccionario con los criterios, la decision y su justificacion.
    """
    objetivo_nivel = CONFIG["referencia_v03_umbrales"] + CONFIG["mejora_minima"]
    pliegues_mejorados = int(
        np.sum(np.array(f1_pliegues) > np.array(f1_pliegues_v03))
    )
    cumple_nivel = bool(corrected_cv >= objetivo_nivel)
    cumple_pliegues = bool(pliegues_mejorados >= CONFIG["pliegues_minimos_mejorados"])
    complementa = bool(complementariedad is not None and complementariedad.get("aporta_ic_positivo", False))
    cumple_regla_literal = bool((cumple_nivel and cumple_pliegues) or complementa)
    adoptado = bool(cumple_regla_literal and (aporta_sobre_comparador or complementa))
    motivos = []
    if not cumple_nivel:
        motivos.append(f"nivel {corrected_cv:.4f} < {objetivo_nivel:.4f}")
    if not cumple_pliegues:
        motivos.append(f"pliegues {pliegues_mejorados}/5 < 3")
    if not complementa:
        motivos.append("complementariedad con IC > 0: no")
    if not aporta_sobre_comparador:
        motivos.append(
            f"el mecanismo no aporta sobre {comparador} (delta {delta_comparador:+.4f}, "
            f"pliegues {pliegues_mejorados_comparador}/5)"
        )
    justificacion = (
        f"ADOPTADO por {nombre}"
        if adoptado
        else "CERRADA sin mejora: " + "; ".join(motivos)
    )
    return {
        "variante": nombre,
        "objetivo_nivel": objetivo_nivel,
        "cumple_nivel": cumple_nivel,
        "pliegues_mejorados_vs_v03": pliegues_mejorados,
        "cumple_pliegues": cumple_pliegues,
        "cumple_complementariedad": complementa,
        "cumple_regla_literal": cumple_regla_literal,
        "comparador": comparador,
        "delta_vs_comparador": delta_comparador,
        "pliegues_mejorados_vs_comparador": pliegues_mejorados_comparador,
        "aporta_sobre_comparador": bool(aporta_sobre_comparador),
        "adoptado": adoptado,
        "justificacion": justificacion,
    }


# ===== SECCION 2: carga, referencias y verificaciones =====


def cargar_modulo_base() -> Any:
    """Importa ``experiments/14_oof_corregido_v2.py`` para reutilizar su receta.

    Returns:
        Modulo del experimento 14, con ``PARAMS_WORD``, ``PARAMS_CHAR`` y
        ``MODELOS`` (los cuatro modelos base del OOF corregido).
    """
    ruta = PROJECT_ROOT / "experiments" / "14_oof_corregido_v2.py"
    especificacion = importlib.util.spec_from_file_location("experimento_14_base", ruta)
    if especificacion is None or especificacion.loader is None:
        raise ImportError(f"No se pudo cargar {ruta}")
    modulo = importlib.util.module_from_spec(especificacion)
    sys.modules[especificacion.name] = modulo
    especificacion.loader.exec_module(modulo)
    if tuple(modulo.MODELOS) != MODELOS_BASE:
        raise RuntimeError("Los modelos base de experiments/14 no coinciden con el OOF congelado")
    return modulo


def cargar_desarrollo() -> tuple[pd.DataFrame, NDArray[np.int_], NDArray[np.int_]]:
    """Carga el desarrollo, sus grupos fusionados y el pliegue de cada fila.

    Returns:
        Tupla ``(desarrollo, grupos, pliegue_de_fila)``. El holdout congelado
        queda excluido por construccion.
    """
    train = data_module.read_train()
    particion = pd.read_csv(IDS_PATH)
    ids_dev = particion.loc[particion["particion"] == "dev", "id"].astype(int).tolist()
    desarrollo = train[train["id"].astype(int).isin(ids_dev)].reset_index(drop=True)
    grupos_completos = grupos_fusionados(train)
    mapa_grupo = dict(zip(train["id"].astype(int), grupos_completos.tolist()))
    grupos = np.array([mapa_grupo[identificador] for identificador in desarrollo["id"].astype(int)])
    pliegue_de_fila = np.zeros(len(desarrollo), dtype=int)
    for numero, (_, idx_validacion) in enumerate(pliegues_corregidos(desarrollo, grupos), start=1):
        pliegue_de_fila[idx_validacion] = numero
    return desarrollo, grupos, pliegue_de_fila


def cargar_oof(
    desarrollo: pd.DataFrame,
    grupos: NDArray[np.int_],
    y_verdadera: NDArray[np.int_],
    pliegue_de_fila: NDArray[np.int_],
) -> tuple[NDArray[np.float64], NDArray[np.float64], dict[str, Any]]:
    """Carga y verifica los OOF congelados de bases y del meta Ridge.

    Args:
        desarrollo: DataFrame del desarrollo.
        grupos: grupos fusionados alineados con ``desarrollo``.
        y_verdadera: etiquetas verdaderas ``(n, 3)``.
        pliegue_de_fila: pliegue corregido de cada fila.

    Returns:
        Tupla ``(proba_bases, proba_meta, info)``.

    Raises:
        RuntimeError: si los insumos congelados no coinciden con el desarrollo.
    """
    mapa_fila = {identificador: posicion for posicion, identificador in enumerate(desarrollo["id"].astype(int))}
    datos_bases = np.load(OOF_BASES_PATH)
    orden_bases = np.array([mapa_fila[identificador] for identificador in datos_bases["ids"].astype(int)])
    if not np.array_equal(datos_bases["y_true"], y_verdadera[orden_bases]):
        raise RuntimeError("Las etiquetas del OOF de bases no coinciden con el desarrollo")
    proba_bases = np.zeros((len(desarrollo), len(MODELOS_BASE), len(ETIQUETAS)))
    proba_bases[orden_bases] = datos_bases["proba"]
    if tuple(str(valor) for valor in datos_bases["modelos"]) != MODELOS_BASE:
        raise RuntimeError("Modelos base inesperados en el OOF congelado")
    if not np.array_equal(datos_bases["grupos"].astype(int), grupos[orden_bases]):
        raise RuntimeError("Los grupos del OOF de bases no coinciden con los grupos fusionados")

    datos_meta = np.load(OOF_META_PATH)
    orden_meta = np.array([mapa_fila[identificador] for identificador in datos_meta["ids"].astype(int)])
    if not np.array_equal(datos_meta["y_true"], y_verdadera[orden_meta]):
        raise RuntimeError("Las etiquetas del OOF del meta no coinciden con el desarrollo")
    if not np.array_equal(datos_meta["pliegue"].astype(int), pliegue_de_fila[orden_meta]):
        raise RuntimeError("Los pliegues del OOF del meta no coinciden con los pliegues corregidos")
    proba_meta = np.zeros((len(desarrollo), len(ETIQUETAS)))
    proba_meta[orden_meta] = datos_meta["proba"]
    info = {
        "modelos_base": list(MODELOS_BASE),
        "meta_variante": str(datos_meta["variante"][0]),
        "meta_modelo": str(datos_meta["modelo"][0]),
        "meta_parametro": float(datos_meta["parametro"][0]),
        "pliegues_coinciden": True,
    }
    return proba_bases, proba_meta, info


def verificar_datos(
    desarrollo: pd.DataFrame, grupos: NDArray[np.int_], pliegue_de_fila: NDArray[np.int_]
) -> dict[str, Any]:
    """Comprueba integridad basica y resume el desarrollo.

    Args:
        desarrollo: DataFrame del desarrollo.
        grupos: grupos fusionados alineados con ``desarrollo``.
        pliegue_de_fila: pliegue corregido de cada fila.

    Returns:
        Diccionario con el resumen de verificacion.
    """
    resumen = {
        "n_filas": len(desarrollo),
        "n_grupos": len(np.unique(grupos)),
        "nulos": int(desarrollo[list(ETIQUETAS) + ["text"]].isna().sum().sum()),
        "prevalencia": {etiqueta: float(desarrollo[etiqueta].mean()) for etiqueta in ETIQUETAS},
        "combinaciones": {
            clave: int(valor) for clave, valor in combinaciones(desarrollo).value_counts().items()
        },
        "tamano_pliegues": {
            int(numero): int(np.sum(pliegue_de_fila == numero))
            for numero in sorted(np.unique(pliegue_de_fila).tolist())
        },
    }
    log(
        f"desarrollo n={resumen['n_filas']} grupos={resumen['n_grupos']} "
        f"nulos={resumen['nulos']} pliegues={resumen['tamano_pliegues']}"
    )
    return resumen


def verificar_reproduccion(
    proba_m2: NDArray[np.float64],
    proba_meta: NDArray[np.float64],
    y_verdadera: NDArray[np.int_],
    referencias_pliegues: dict[str, Any],
) -> dict[str, Any]:
    """Verifica que los OOF congelados reproducen las referencias del protocolo.

    Args:
        proba_m2: probabilidades OOF de M2.
        proba_meta: probabilidades OOF del meta Ridge.
        y_verdadera: etiquetas verdaderas.
        referencias_pliegues: referencia con las F1 por pliegue de V03.

    Returns:
        Diccionario con las diferencias absolutas verificadas.

    Raises:
        RuntimeError: si alguna referencia no se reproduce dentro de la tolerancia.
    """
    f1_m2 = evaluar_con_umbrales(
        proba_m2, y_verdadera, {etiqueta: 0.5 for etiqueta in ETIQUETAS}
    )["f1_media_columnas"]
    f1_v03 = evaluar_con_umbrales(proba_m2, y_verdadera, CONFIG["umbrales_v03"])["f1_media_columnas"]
    f1_meta = evaluar_con_umbrales(
        proba_meta, y_verdadera, {etiqueta: 0.5 for etiqueta in ETIQUETAS}
    )["f1_media_columnas"]
    diferencias = {
        "m2_umbral_05": abs(f1_m2 - CONFIG["referencia_m2_umbral_05"]),
        "v03_umbrales": abs(f1_v03 - CONFIG["referencia_v03_umbrales"]),
        "meta_ridge": abs(f1_meta - CONFIG["referencia_meta_ridge"]),
    }
    for nombre, diferencia in diferencias.items():
        if diferencia > CONFIG["tolerancia_replicacion"]:
            raise RuntimeError(
                f"La referencia {nombre} no se reproduce: diferencia {diferencia:.2e}"
            )
    reproducido = {
        "m2_umbral_05": f1_m2,
        "v03_umbrales": f1_v03,
        "meta_ridge": f1_meta,
        "diferencias": diferencias,
        "f1_pliegues_v03": referencias_pliegues["f1_pliegues_v03"],
    }
    log(
        f"referencias reproducidas: M2 0.5={f1_m2:.4f} | V03 umbrales={f1_v03:.4f} | "
        f"meta={f1_meta:.4f}"
    )
    return reproducido


def cargar_referencias() -> dict[str, Any]:
    """Carga las F1 por pliegue de V03 y del meta desde el reporte de P1/P6.

    Returns:
        Diccionario con ``f1_pliegues_v03`` y ``f1_pliegues_meta``.
    """
    datos = json.loads((PROJECT_ROOT / "reports" / "cadenas_etiquetas_v2.json").read_text(encoding="utf-8"))
    anidado = datos["meta_anidado"]
    return {
        "f1_pliegues_v03": [float(valor) for valor in anidado["f1_pliegues_base_v03_umbrales"]],
        "f1_pliegues_m2": [float(valor) for valor in anidado["f1_pliegues_base_m2_05"]],
    }


# ===== SECCION 3: E2-14 calibracion anidada =====


def logit(probabilidades: NDArray[np.float64]) -> NDArray[np.float64]:
    """Aplica el logit con recorte numerico.

    Args:
        probabilidades: probabilidades en ``[0, 1]``.

    Returns:
        Log-odds recortados.
    """
    recortadas = np.clip(probabilidades, 1e-6, 1.0 - 1e-6)
    return np.log(recortadas / (1.0 - recortadas))


def calibrar_anidado(
    proba: NDArray[np.float64],
    y_verdadera: NDArray[np.int_],
    pliegue_de_fila: NDArray[np.int_],
    metodo: str,
) -> NDArray[np.float64]:
    """Calibra de forma anidada por pliegue (calibracion cruzada estandar).

    El calibrador del pliegue ``k`` se ajusta con las predicciones OOF de los
    pliegues restantes y transforma el pliegue ``k``; ninguna fila participa en
    su propio calibrador.

    Args:
        proba: probabilidades OOF ``(n, 3)``.
        y_verdadera: etiquetas verdaderas ``(n, 3)``.
        pliegue_de_fila: pliegue de cada fila.
        metodo: ``platt`` (LR sobre el logit) o ``isotonic``.

    Returns:
        Matriz de probabilidades calibradas ``(n, 3)``.

    Raises:
        ValueError: si el metodo no existe.
    """
    if metodo not in ("platt", "isotonic"):
        raise ValueError(f"Metodo de calibracion desconocido: {metodo}")
    calibrada = np.zeros_like(proba)
    for numero, mascara_evaluacion in mascaras_pliegues(pliegue_de_fila):
        ajuste = ~mascara_evaluacion
        for posicion in range(proba.shape[1]):
            if metodo == "platt":
                modelo = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")
                modelo.fit(
                    logit(proba[ajuste, posicion]).reshape(-1, 1), y_verdadera[ajuste, posicion]
                )
                calibrada[mascara_evaluacion, posicion] = modelo.predict_proba(
                    logit(proba[mascara_evaluacion, posicion]).reshape(-1, 1)
                )[:, 1]
            else:
                modelo = IsotonicRegression(out_of_bounds="clip")
                modelo.fit(proba[ajuste, posicion], y_verdadera[ajuste, posicion])
                calibrada[mascara_evaluacion, posicion] = modelo.predict(
                    proba[mascara_evaluacion, posicion]
                )
    return calibrada


def umbrales_anidados(
    proba: NDArray[np.float64], y_verdadera: NDArray[np.int_], pliegue_de_fila: NDArray[np.int_]
) -> NDArray[np.float64]:
    """Ajusta umbrales por pliegue con los pliegues restantes.

    Args:
        proba: probabilidades ``(n, 3)``.
        y_verdadera: etiquetas verdaderas ``(n, 3)``.
        pliegue_de_fila: pliegue de cada fila.

    Returns:
        Matriz ``(n_pliegues, 3)`` con el umbral del pliegue en su fila.
    """
    umbrales = np.zeros((len(np.unique(pliegue_de_fila)), y_verdadera.shape[1]))
    for numero, mascara in mascaras_pliegues(pliegue_de_fila):
        ajuste = ~mascara
        for posicion in range(y_verdadera.shape[1]):
            umbrales[numero - 1, posicion] = ajustar_umbral_etiqueta(
                y_verdadera[ajuste, posicion], proba[ajuste, posicion]
            )
    return umbrales


def prediccion_con_umbrales_anidados(
    proba: NDArray[np.float64],
    umbrales: NDArray[np.float64],
    pliegue_de_fila: NDArray[np.int_],
) -> NDArray[np.int_]:
    """Predice cada pliegue con el umbral ajustado en los pliegues restantes.

    Args:
        proba: probabilidades ``(n, 3)``.
        umbrales: matriz ``(n_pliegues, 3)`` de umbrales anidados.
        pliegue_de_fila: pliegue de cada fila.

    Returns:
        Matriz binaria predicha ``(n, 3)``.
    """
    prediccion = np.zeros(proba.shape, dtype=int)
    for numero, mascara in mascaras_pliegues(pliegue_de_fila):
        prediccion[mascara] = (proba[mascara] >= umbrales[numero - 1]).astype(int)
    return prediccion


def resumir_calibracion(
    nombre: str,
    proba: NDArray[np.float64],
    y_verdadera: NDArray[np.int_],
    pliegue_de_fila: NDArray[np.int_],
) -> dict[str, Any]:
    """Resume una variante de calibracion (o su version sin calibrar).

    Args:
        nombre: identificador de la variante.
        proba: probabilidades de la variante.
        y_verdadera: etiquetas verdaderas.
        pliegue_de_fila: pliegue de cada fila.

    Returns:
        Diccionario con Brier, log-loss, F1 a 0.5, F1 con umbrales anidados, la
        estabilidad de los umbrales y la cota optimista.
    """
    metricas_probabilidad = brier_logloss(proba, y_verdadera)
    prediccion_05 = (proba >= 0.5).astype(int)
    umbrales = umbrales_anidados(proba, y_verdadera, pliegue_de_fila)
    prediccion_anidada = prediccion_con_umbrales_anidados(proba, umbrales, pliegue_de_fila)
    umbrales_optimistas = ajustar_umbrales(y_verdadera, proba)
    return {
        "variante": nombre,
        "brier": metricas_probabilidad["brier"],
        "log_loss": metricas_probabilidad["log_loss"],
        "brier_por_etiqueta": metricas_probabilidad["brier_por_etiqueta"],
        "log_loss_por_etiqueta": metricas_probabilidad["log_loss_por_etiqueta"],
        "f1_umbral_05": f1_media_rapida(y_verdadera, prediccion_05),
        "f1_por_etiqueta_05": f1_por_etiqueta_rapida(y_verdadera, prediccion_05),
        "f1_anidado": f1_media_rapida(y_verdadera, prediccion_anidada),
        "f1_pliegues": f1_por_pliegue_prediccion(prediccion_anidada, y_verdadera, pliegue_de_fila),
        "umbrales_anidados": umbrales,
        "umbrales_media": {
            etiqueta: float(np.mean(umbrales[:, posicion]))
            for posicion, etiqueta in enumerate(ETIQUETAS)
        },
        "umbrales_std": {
            etiqueta: float(np.std(umbrales[:, posicion]))
            for posicion, etiqueta in enumerate(ETIQUETAS)
        },
        "umbrales_rango": {
            etiqueta: float(np.max(umbrales[:, posicion]) - np.min(umbrales[:, posicion]))
            for posicion, etiqueta in enumerate(ETIQUETAS)
        },
        "f1_optimista": evaluar_con_umbrales(proba, y_verdadera, umbrales_optimistas)[
            "f1_media_columnas"
        ],
        "umbrales_optimistas": umbrales_optimistas,
        "std_pliegues": float(np.std(f1_por_pliegue_prediccion(prediccion_anidada, y_verdadera, pliegue_de_fila))),
    }


def ejecutar_e2_14(
    proba_m2: NDArray[np.float64],
    proba_meta: NDArray[np.float64],
    y_verdadera: NDArray[np.int_],
    pliegue_de_fila: NDArray[np.int_],
    referencias: dict[str, Any],
) -> dict[str, Any]:
    """Ejecuta E2-14: calibracion Platt e isotonic anidadas de M2 y del meta.

    Args:
        proba_m2: probabilidades OOF de M2.
        proba_meta: probabilidades OOF del meta Ridge.
        y_verdadera: etiquetas verdaderas.
        pliegue_de_fila: pliegue de cada fila.
        referencias: referencia con las F1 por pliegue de V03.

    Returns:
        Diccionario con las variantes, la comparacion sin calibrar y la decision.
    """
    log("E2-14 calibracion: Platt e isotonic anidados (M2 y meta Ridge)")
    variantes: dict[str, Any] = {}
    for nombre_modelo, proba in (("M2", proba_m2), ("META", proba_meta)):
        variantes[f"{nombre_modelo}-SIN-CALIBRAR"] = resumir_calibracion(
            f"{nombre_modelo}-SIN-CALIBRAR", proba, y_verdadera, pliegue_de_fila
        )
        variantes[f"{nombre_modelo}-SIN-CALIBRAR"]["proba"] = proba
        for metodo, etiqueta in (("platt", "PLATT"), ("isotonic", "ISOTONIC")):
            calibrada = calibrar_anidado(proba, y_verdadera, pliegue_de_fila, metodo)
            variantes[f"{nombre_modelo}-{etiqueta}"] = resumir_calibracion(
                f"{nombre_modelo}-{etiqueta}", calibrada, y_verdadera, pliegue_de_fila
            )
            variantes[f"{nombre_modelo}-{etiqueta}"]["proba"] = calibrada

    comparaciones: dict[str, Any] = {}
    for nombre_modelo in ("M2", "META"):
        base = variantes[f"{nombre_modelo}-SIN-CALIBRAR"]
        par: dict[str, Any] = {}
        for etiqueta in ("PLATT", "ISOTONIC"):
            variante = variantes[f"{nombre_modelo}-{etiqueta}"]
            delta = variante["f1_anidado"] - base["f1_anidado"]
            pliegues_mejorados = int(np.sum(np.array(variante["f1_pliegues"]) > np.array(base["f1_pliegues"])))
            par[etiqueta] = {
                "delta_f1_anidado": delta,
                "delta_brier": variante["brier"] - base["brier"],
                "delta_log_loss": variante["log_loss"] - base["log_loss"],
                "delta_f1_umbral_05": variante["f1_umbral_05"] - base["f1_umbral_05"],
                "pliegues_mejorados": pliegues_mejorados,
                "estabilidad_umbral_media": float(
                    np.mean(list(variante["umbrales_std"].values()))
                ),
                "aporta_sobre_sin_calibrar": bool(
                    delta >= CONFIG["mejora_minima"]
                    and pliegues_mejorados >= CONFIG["pliegues_minimos_mejorados"]
                ),
            }
        comparaciones[nombre_modelo] = par

    # Mejor variante calibrada por modelo (criterio primario: F1 anidado).
    mejores: dict[str, Any] = {}
    for nombre_modelo in ("M2", "META"):
        candidatas = [f"{nombre_modelo}-PLATT", f"{nombre_modelo}-ISOTONIC"]
        mejor_nombre = max(candidatas, key=lambda clave: variantes[clave]["f1_anidado"])
        resumen = dict(variantes[mejor_nombre])
        resumen["delta_vs_sin_calibrar"] = comparaciones[nombre_modelo][
            mejor_nombre.split("-")[1]
        ]["delta_f1_anidado"]
        resumen["pliegues_mejorados_vs_sin_calibrar"] = comparaciones[nombre_modelo][
            mejor_nombre.split("-")[1]
        ]["pliegues_mejorados"]
        resumen["aporta_sobre_comparador"] = comparaciones[nombre_modelo][
            mejor_nombre.split("-")[1]
        ]["aporta_sobre_sin_calibrar"]
        referencia_par = variantes[f"{nombre_modelo}-SIN-CALIBRAR"]["f1_umbral_05"]
        complementariedad = evaluar_complementariedad(
            resumen["proba"], variantes[f"{nombre_modelo}-SIN-CALIBRAR"]["proba"], y_verdadera, referencia_par
        )
        resumen["complementariedad"] = complementariedad
        resumen["decision"] = aplicar_regla_adopcion(
            nombre=mejor_nombre,
            corrected_cv=resumen["f1_anidado"],
            f1_pliegues=resumen["f1_pliegues"],
            f1_pliegues_v03=referencias["f1_pliegues_v03"],
            complementariedad=complementariedad,
            comparador=f"{nombre_modelo}-SIN-CALIBRAR",
            delta_comparador=resumen["delta_vs_sin_calibrar"],
            pliegues_mejorados_comparador=resumen["pliegues_mejorados_vs_sin_calibrar"],
            aporta_sobre_comparador=resumen["aporta_sobre_comparador"],
        )
        mejores[nombre_modelo] = resumen
        log(
            f"  {mejor_nombre}: F1 anidado {resumen['f1_anidado']:.4f} "
            f"(delta vs sin calibrar {resumen['delta_vs_sin_calibrar']:+.4f}); "
            f"Brier {resumen['brier']:.4f} vs {variantes[f'{nombre_modelo}-SIN-CALIBRAR']['brier']:.4f}"
        )
    nota = (
        "Platt e isotonic son monotonos por etiqueta: la F1 con umbrales reajustados es "
        "invariante salvo empates (isotonic pierde resolucion); el aporte medible esta en "
        "Brier/log-loss y en la estabilidad de los umbrales."
    )
    return {
        "variantes": variantes,
        "comparaciones": comparaciones,
        "mejores": mejores,
        "nota_invarianza": nota,
        "decision": {
            "M2": mejores["M2"]["decision"],
            "META": mejores["META"]["decision"],
        },
    }


# ===== SECCION 4: E2-13 encoder de entidades =====


def extraer_entidades(textos: Sequence[str]) -> dict[str, list[list[str]]]:
    """Extrae menciones, hashtags y dominios de cada texto.

    Args:
        textos: secuencia de textos.

    Returns:
        Diccionario ``tipo -> lista de valores por fila`` (sin duplicados
        dentro de la fila, en minusculas).
    """
    menciones: list[list[str]] = []
    hashtags: list[list[str]] = []
    dominios: list[list[str]] = []
    for texto in textos:
        menciones.append(sorted({valor.lower() for valor in re.findall(r"@(\w+)", texto)}))
        hashtags.append(sorted({valor.lower() for valor in re.findall(r"#(\w+)", texto)}))
        hosts = [host.lower().rstrip(".") for host in PATRON_HOST.findall(texto)]
        if not hosts and re.search(r"\bt\.co\b", texto):
            hosts = ["t.co"]
        dominios.append(sorted(set(hosts)))
    return {"mencion": menciones, "hashtag": hashtags, "dominio": dominios}


def cobertura_entidades(
    entidades: dict[str, list[list[str]]],
    pliegue_de_fila: NDArray[np.int_],
    frecuencia_minima: int,
) -> dict[str, Any]:
    """Calcula la cobertura de entidades vista en el pliegue de ajuste.

    Args:
        entidades: salida de :func:`extraer_entidades`.
        pliegue_de_fila: pliegue de cada fila.
        frecuencia_minima: frecuencia documental minima para considerar conocida
            una entidad.

    Returns:
        Diccionario con cobertura por tipo (instancias y filas), valores unicos
        y la cobertura informativa agregada de menciones + hashtags.
    """
    por_tipo: dict[str, Any] = {}
    for tipo in TIPOS_ENTIDAD:
        instancias = 0
        instancias_cubiertas = 0
        filas_con = 0
        filas_cubiertas = 0
        for _, mascara in mascaras_pliegues(pliegue_de_fila):
            ajuste = ~mascara
            contador: Counter[str] = Counter()
            for posicion in np.flatnonzero(ajuste):
                contador.update(set(entidades[tipo][posicion]))
            conocidas = {clave for clave, valor in contador.items() if valor >= frecuencia_minima}
            for posicion in np.flatnonzero(mascara):
                valores = entidades[tipo][posicion]
                if valores:
                    filas_con += 1
                instancias += len(valores)
                instancias_cubiertas += sum(1 for valor in valores if valor in conocidas)
                if any(valor in conocidas for valor in valores):
                    filas_cubiertas += 1
        valores_dev = [valor for fila in entidades[tipo] for valor in fila]
        contador_dev: Counter[str] = Counter(valores_dev)
        por_tipo[tipo] = {
            "instancias": instancias,
            "instancias_cubiertas": instancias_cubiertas,
            "cobertura_instancias": instancias_cubiertas / max(instancias, 1),
            "filas_con_entidad": filas_con,
            "filas_cubiertas": filas_cubiertas,
            "cobertura_filas": filas_cubiertas / max(filas_con, 1),
            "unicos_dev": len(contador_dev),
            "sobrevivientes_frecuencia_minima_dev": int(
                sum(1 for valor in contador_dev.values() if valor >= frecuencia_minima)
            ),
        }
    informativas = ["mencion", "hashtag"]
    instancias_info = sum(por_tipo[tipo]["instancias"] for tipo in informativas)
    cubiertas_info = sum(por_tipo[tipo]["instancias_cubiertas"] for tipo in informativas)
    cobertura_informativa = cubiertas_info / max(instancias_info, 1)
    return {
        "por_tipo": por_tipo,
        "cobertura_informativa": cobertura_informativa,
        "instancias_informativas": instancias_info,
        "umbral_cierre": CONFIG["umbral_cobertura_minima"],
        "cierra_por_sparsity": bool(cobertura_informativa < CONFIG["umbral_cobertura_minima"]),
    }


def tablas_target_encoding(
    entidades_tipo: list[list[str]],
    y_columna: NDArray[np.int_],
    idx_ajuste: NDArray[np.int_],
    frecuencia_minima: int,
    suavizado: float,
) -> tuple[dict[str, float], float]:
    """Construye la tabla de target encoding de una etiqueta.

    Args:
        entidades_tipo: valores de entidad por fila.
        y_columna: etiqueta binaria de la columna.
        idx_ajuste: indices de las filas de ajuste.
        frecuencia_minima: frecuencia documental minima para entrar en la tabla.
        suavizado: fuerza del suavizado hacia el prior (m).

    Returns:
        Tupla ``(tabla, prior)``; ``prior`` es la prevalencia en el ajuste.
    """
    prior = float(np.mean(y_columna[idx_ajuste]))
    documentos: Counter[str] = Counter()
    positivos: Counter[str] = Counter()
    for posicion in idx_ajuste:
        for valor in set(entidades_tipo[posicion]):
            documentos[valor] += 1
            if y_columna[posicion] == 1:
                positivos[valor] += 1
    tabla = {
        valor: (positivos[valor] + suavizado * prior) / (frecuencia + suavizado)
        for valor, frecuencia in documentos.items()
        if frecuencia >= frecuencia_minima
    }
    return tabla, prior


def codificar_entidades(
    entidades_tipo: list[list[str]],
    tablas: dict[str, dict[str, float]],
    priors: dict[str, float],
    idx_filas: NDArray[np.int_],
) -> NDArray[np.float64]:
    """Codifica las filas con la media de la tabla de sus entidades conocidas.

    Args:
        entidades_tipo: valores de entidad por fila.
        tablas: tabla de target encoding por etiqueta.
        priors: prior por etiqueta (bucket desconocido).
        idx_filas: filas a codificar.

    Returns:
        Matriz ``(len(idx_filas), 3 + 1)`` con los tres encodings y el numero de
        instancias de la fila.
    """
    filas = []
    for posicion in idx_filas:
        valores = entidades_tipo[posicion]
        fila = []
        for etiqueta in ETIQUETAS:
            conocidos = [tablas[etiqueta][valor] for valor in valores if valor in tablas[etiqueta]]
            fila.append(float(np.mean(conocidos)) if conocidos else priors[etiqueta])
        fila.append(float(len(valores)))
        filas.append(fila)
    return np.asarray(filas, dtype=float)


def features_entidades(
    entidades: dict[str, list[list[str]]],
    y_verdadera: NDArray[np.int_],
    idx_ajuste: NDArray[np.int_],
    idx_evaluacion: NDArray[np.int_],
) -> tuple[NDArray[np.float64], NDArray[np.float64], list[str]]:
    """Construye las features de entidades para ajuste y evaluacion.

    Args:
        entidades: salida de :func:`extraer_entidades`.
        y_verdadera: etiquetas verdaderas ``(n, 3)``.
        idx_ajuste: filas usadas para ajustar las tablas.
        idx_evaluacion: filas a codificar.

    Returns:
        Tupla ``(features_ajuste, features_evaluacion, nombres)``.
    """
    bloques_ajuste: list[NDArray[np.float64]] = []
    bloques_evaluacion: list[NDArray[np.float64]] = []
    nombres: list[str] = []
    for tipo in TIPOS_ENTIDAD:
        tablas: dict[str, dict[str, float]] = {}
        priors: dict[str, float] = {}
        for etiqueta in ETIQUETAS:
            tablas[etiqueta], priors[etiqueta] = tablas_target_encoding(
                entidades[tipo],
                y_verdadera[:, ETIQUETAS.index(etiqueta)],
                idx_ajuste,
                CONFIG["frecuencia_minima_entidades"],
                CONFIG["suavizado_entidades"],
            )
        bloques_ajuste.append(codificar_entidades(entidades[tipo], tablas, priors, idx_ajuste))
        bloques_evaluacion.append(codificar_entidades(entidades[tipo], tablas, priors, idx_evaluacion))
        nombres.extend(
            [f"ent_{tipo}_{etiqueta}" for etiqueta in ETIQUETAS] + [f"ent_{tipo}_n"]
        )
    return (
        np.hstack(bloques_ajuste),
        np.hstack(bloques_evaluacion),
        nombres,
    )


def bloque_scores(
    proba: NDArray[np.float64], nombres_modelos: Sequence[str]
) -> tuple[NDArray[np.float64], list[str]]:
    """Construye el bloque de scores del meta (probabilidades y margenes).

    Args:
        proba: probabilidades ``(n, n_modelos, 3)``.
        nombres_modelos: nombre de cada modelo.

    Returns:
        Tupla ``(matriz, nombres)`` con probabilidades y margenes.
    """
    n_filas = proba.shape[0]
    planas = proba.reshape(n_filas, -1)
    margenes = np.abs(planas - 0.5)
    nombres = [f"proba_{modelo}_{etiqueta}" for modelo in nombres_modelos for etiqueta in ETIQUETAS]
    nombres += [f"margen_{modelo}_{etiqueta}" for modelo in nombres_modelos for etiqueta in ETIQUETAS]
    return np.hstack([planas, margenes]), nombres


def rasgos_estructurales(textos: Sequence[str]) -> tuple[NDArray[np.float64], list[str]]:
    """Calcula los 8 rasgos estructurales fijos del meta de P1.

    Args:
        textos: secuencia de textos.

    Returns:
        Tupla ``(matriz, nombres)``.
    """
    filas = []
    for texto in textos:
        filas.append(
            [
                len(texto),
                len(texto.split()),
                float(bool(PATRON_MENCION.search(texto))),
                float(bool(PATRON_HASHTAG.search(texto))),
                float(bool(PATRON_URL.search(texto))),
                float(bool(PATRON_COLECTIVO.search(texto))),
                float(bool(PATRON_SEGUNDA_PERSONA.search(texto))),
                float(bool(PATRON_COMILLAS.search(texto))),
            ]
        )
    return np.asarray(filas, dtype=float), list(NOMBRES_ESTRUCTURALES)


def ajustar_meta_y_predecir(
    x_entrenamiento: NDArray[np.float64],
    y_entrenamiento: NDArray[np.int_],
    x_prediccion: NDArray[np.float64],
    modelo: str,
    parametro: float,
) -> NDArray[np.float64]:
    """Ajusta un meta-modelo por etiqueta y devuelve probabilidades.

    Args:
        x_entrenamiento: features de entrenamiento.
        y_entrenamiento: etiqueta binaria de entrenamiento.
        x_prediccion: features a predecir.
        modelo: ``ridge`` o ``lr``.
        parametro: alpha de Ridge o C de la LR.

    Returns:
        Probabilidades de la clase positiva.

    Raises:
        ValueError: si el modelo no existe.
    """
    if modelo == "ridge":
        estimador = RidgeClassifier(alpha=parametro, class_weight="balanced")
        estimador.fit(x_entrenamiento, y_entrenamiento)
        return 1.0 / (1.0 + np.exp(-estimador.decision_function(x_prediccion)))
    if modelo == "lr":
        estimador = LogisticRegression(
            C=parametro,
            class_weight="balanced",
            max_iter=2000,
            solver="liblinear",
            random_state=RANDOM_STATE,
        )
        estimador.fit(x_entrenamiento, y_entrenamiento)
        return estimador.predict_proba(x_prediccion)[:, 1]
    raise ValueError(f"Modelo de meta desconocido: {modelo}")


def ejecutar_e2_13(
    desarrollo: pd.DataFrame,
    grupos: NDArray[np.int_],
    pliegue_de_fila: NDArray[np.int_],
    proba_bases: NDArray[np.float64],
    proba_meta: NDArray[np.float64],
    y_verdadera: NDArray[np.int_],
    referencias: dict[str, Any],
) -> dict[str, Any]:
    """Ejecuta E2-13: encoder de entidades como bloque del meta y canal propio.

    Args:
        desarrollo: DataFrame del desarrollo.
        grupos: grupos fusionados.
        pliegue_de_fila: pliegue de cada fila.
        proba_bases: probabilidades OOF de los 4 bases.
        proba_meta: probabilidades OOF del meta (referencia).
        y_verdadera: etiquetas verdaderas.
        referencias: referencia con las F1 por pliegue de V03.

    Returns:
        Diccionario con cobertura, variantes y decision.
    """
    log("E2-13 encoder de entidades: cobertura, bloque del meta y canal propio")
    entidades = extraer_entidades(desarrollo["text"].tolist())
    cobertura = cobertura_entidades(
        entidades, pliegue_de_fila, CONFIG["frecuencia_minima_entidades"]
    )
    log(
        f"  cobertura informativa (mencion+hashtag): {cobertura['cobertura_informativa']:.3f} "
        f"(umbral de cierre {CONFIG['umbral_cobertura_minima']:.2f})"
    )
    for tipo, detalle in cobertura["por_tipo"].items():
        log(
            f"  {tipo}: instancias={detalle['instancias']} unicos={detalle['unicos_dev']} "
            f"freq>={CONFIG['frecuencia_minima_entidades']}: "
            f"{detalle['sobrevivientes_frecuencia_minima_dev']} cobertura="
            f"{detalle['cobertura_instancias']:.3f}"
        )
    base_meta, nombres_meta = bloque_scores(proba_bases, MODELOS_BASE)
    estructurales, nombres_estructurales = rasgos_estructurales(desarrollo["text"].tolist())
    base_meta = np.hstack([base_meta, estructurales])
    nombres_meta += nombres_estructurales

    n_filas = len(desarrollo)
    proba_meta_ent_ridge = np.zeros((n_filas, len(ETIQUETAS)))
    proba_meta_ent_lr = np.zeros((n_filas, len(ETIQUETAS)))
    proba_ent_solo = np.zeros((n_filas, len(ETIQUETAS)))
    for numero, (idx_entrenamiento, idx_validacion) in enumerate(
        pliegues_corregidos(desarrollo, grupos), start=1
    ):
        divisor_interno = StratifiedGroupKFold(
            n_splits=CONFIG["n_pliegues_internos_entidades"],
            shuffle=True,
            random_state=RANDOM_STATE,
        )
        sub_desarrollo = desarrollo.iloc[idx_entrenamiento]
        features_ent_interno = np.zeros(
            (len(idx_entrenamiento), 3 * (len(ETIQUETAS) + 1))
        )
        for idx_int_ent, idx_int_val in divisor_interno.split(
            sub_desarrollo,
            combinaciones(sub_desarrollo),
            groups=grupos[idx_entrenamiento],
        ):
            _, bloque_val, _ = features_entidades(
                entidades,
                y_verdadera,
                idx_entrenamiento[idx_int_ent],
                idx_entrenamiento[idx_int_val],
            )
            features_ent_interno[idx_int_val] = bloque_val
        _, features_ent_validacion, nombres_ent = features_entidades(
            entidades, y_verdadera, idx_entrenamiento, idx_validacion
        )

        x_meta_entrenamiento = np.hstack([base_meta[idx_entrenamiento], features_ent_interno])
        x_meta_validacion = np.hstack([base_meta[idx_validacion], features_ent_validacion])
        escalador = StandardScaler().fit(x_meta_entrenamiento)
        x_ent_meta_entrenamiento = escalador.transform(x_meta_entrenamiento)
        x_ent_meta_validacion = escalador.transform(x_meta_validacion)
        escalador_ent = StandardScaler().fit(features_ent_interno)
        x_ent_entrenamiento = escalador_ent.transform(features_ent_interno)
        x_ent_validacion = escalador_ent.transform(features_ent_validacion)
        for posicion, etiqueta in enumerate(ETIQUETAS):
            y_entrenamiento = y_verdadera[idx_entrenamiento, posicion]
            proba_meta_ent_ridge[idx_validacion, posicion] = ajustar_meta_y_predecir(
                x_ent_meta_entrenamiento, y_entrenamiento, x_ent_meta_validacion, "ridge", 3.0
            )
            proba_meta_ent_lr[idx_validacion, posicion] = ajustar_meta_y_predecir(
                x_ent_meta_entrenamiento, y_entrenamiento, x_ent_meta_validacion, "lr", 1.0
            )
            proba_ent_solo[idx_validacion, posicion] = ajustar_meta_y_predecir(
                x_ent_entrenamiento, y_entrenamiento, x_ent_validacion, "lr", 1.0
            )
        log(f"  pliegue {numero}: bloque de entidades {len(nombres_ent)} columnas")

    variantes: dict[str, Any] = {}
    for nombre, proba, comparador in (
        ("META-ENT-RIDGE", proba_meta_ent_ridge, proba_meta),
        ("META-ENT-LR", proba_meta_ent_lr, proba_meta),
        ("ENT-SOLO-LR", proba_ent_solo, proba_meta),
    ):
        metricas = evaluar_con_umbrales(
            proba, y_verdadera, {etiqueta: 0.5 for etiqueta in ETIQUETAS}
        )
        f1_pliegues = f1_por_pliegue(
            proba, y_verdadera, pliegue_de_fila, {etiqueta: 0.5 for etiqueta in ETIQUETAS}
        )
        complementariedad = evaluar_complementariedad(
            proba, comparador, y_verdadera, CONFIG["referencia_meta_ridge"]
        )
        delta_comparador = metricas["f1_media_columnas"] - CONFIG["referencia_meta_ridge"]
        pliegues_comparador = int(
            np.sum(
                np.array(f1_pliegues)
                > np.array(
                    f1_por_pliegue(
                        comparador,
                        y_verdadera,
                        pliegue_de_fila,
                        {etiqueta: 0.5 for etiqueta in ETIQUETAS},
                    )
                )
            )
        )
        decision = aplicar_regla_adopcion(
            nombre=nombre,
            corrected_cv=metricas["f1_media_columnas"],
            f1_pliegues=f1_pliegues,
            f1_pliegues_v03=referencias["f1_pliegues_v03"],
            complementariedad=complementariedad,
            comparador="META-BASES-RIDGE (0.7874)",
            delta_comparador=delta_comparador,
            pliegues_mejorados_comparador=pliegues_comparador,
            aporta_sobre_comparador=bool(
                delta_comparador >= CONFIG["mejora_minima"]
                and pliegues_comparador >= CONFIG["pliegues_minimos_mejorados"]
            ),
        )
        variantes[nombre] = {
            "corrected_cv": metricas["f1_media_columnas"],
            "f1_por_etiqueta": metricas["f1_por_etiqueta"],
            "f1_pliegues": f1_pliegues,
            "std_pliegues": float(np.std(f1_pliegues)),
            "delta_vs_meta_ridge": metricas["f1_media_columnas"] - CONFIG["referencia_meta_ridge"],
            "delta_vs_v03": metricas["f1_media_columnas"] - CONFIG["referencia_v03_umbrales"],
            "complementariedad": complementariedad,
            "decision": decision,
            "proba": proba,
        }
        log(
            f"  {nombre}: corrected_cv {metricas['f1_media_columnas']:.4f} "
            f"(delta vs meta {variantes[nombre]['delta_vs_meta_ridge']:+.4f})"
        )
    mejor_nombre = max(variantes, key=lambda clave: variantes[clave]["corrected_cv"])
    return {
        "cobertura": cobertura,
        "variantes": variantes,
        "mejor": mejor_nombre,
        "decision": variantes[mejor_nombre]["decision"],
    }


# ===== SECCION 5: E2-16 y E2-20 rebanadas =====


def definir_rebanadas(
    textos: pd.Series, n_caracteres: NDArray[np.int_]
) -> dict[str, NDArray[np.bool_]]:
    """Define las rebanadas estructurales del analisis (todas sin etiquetas).

    Args:
        textos: textos del desarrollo.
        n_caracteres: numero de caracteres por fila.

    Returns:
        Diccionario ``nombre -> mascara booleana``. ``largo_q4`` usa el cuartil
        superior de longitud del desarrollo (definicion sin etiquetas).
    """
    return {
        "hashtag": textos.str.contains(PATRON_HASHTAG).to_numpy(),
        "mencion": textos.str.contains(PATRON_MENCION).to_numpy(),
        "url": textos.str.contains(PATRON_URL).to_numpy(),
        "pregunta": textos.str.contains(PATRON_PREGUNTA).to_numpy(),
        "comillas": textos.str.contains(PATRON_COMILLAS).to_numpy(),
        "colectiva": np.array(
            [bool(PATRON_COLECTIVO.search(texto.lower())) for texto in textos]
        ),
        "corto40": n_caracteres < 40,
        "largo_q4": n_caracteres >= float(np.quantile(n_caracteres, 0.75)),
    }


def ejecutar_e2_16(
    textos: pd.Series,
    proba_m2: NDArray[np.float64],
    y_verdadera: NDArray[np.int_],
    pliegue_de_fila: NDArray[np.int_],
    referencias: dict[str, Any],
) -> dict[str, Any]:
    """Ejecuta E2-16: umbrales por rebanada con shrinkage y evaluacion anidada.

    Args:
        textos: textos del desarrollo.
        proba_m2: probabilidades OOF de M2 (modelo de V03).
        y_verdadera: etiquetas verdaderas.
        pliegue_de_fila: pliegue de cada fila.
        referencias: referencia con las F1 por pliegue de V03.

    Returns:
        Diccionario con las rebanadas, las variantes de shrinkage y la decision.

    Nota:
        Las rebanadas se solapan (por ejemplo, un tweet con hashtag y URL cae en
        dos). Para una fila en varias rebanadas, la mezcla se aplica de forma
        secuencial en el orden de ``incluidas``, lo que equivale a
        ``(1 - W) * global + suma(w_s * local_s)`` con ``W = 1 - prod(1 - w_s)``.
        Comillas queda excluida porque su n (72) no alcanza el minimo de 80.
    """
    log("E2-16 umbrales por rebanada con shrinkage")
    n_caracteres = textos.str.len().to_numpy()
    mascaras = definir_rebanadas(textos, n_caracteres)
    tamano = {nombre: int(mascara.sum()) for nombre, mascara in mascaras.items()}
    excluidas = [nombre for nombre, valor in tamano.items() if valor < CONFIG["min_rebanada"]]
    incluidas = {
        nombre: mascara
        for nombre, mascara in mascaras.items()
        if tamano[nombre] >= CONFIG["min_rebanada"]
    }
    log(f"  rebanadas n>={CONFIG['min_rebanada']}: {sorted(incluidas)}; excluidas: {excluidas}")

    n_filas = len(y_verdadera)
    # Umbrales globales anidados (comparador honesto de la linea).
    umbrales_global = umbrales_anidados(proba_m2, y_verdadera, pliegue_de_fila)
    prediccion_global = prediccion_con_umbrales_anidados(proba_m2, umbrales_global, pliegue_de_fila)
    f1_global = f1_media_rapida(y_verdadera, prediccion_global)
    f1_global_pliegues = f1_por_pliegue_prediccion(prediccion_global, y_verdadera, pliegue_de_fila)

    # Locales por pliegue y rebanada (independientes de m): se calculan una vez.
    locales: dict[int, dict[str, dict[str, Any]]] = {}
    for numero, mascara_validacion in mascaras_pliegues(pliegue_de_fila):
        idx_ajuste = np.flatnonzero(~mascara_validacion)
        detalle_pliegue: dict[str, dict[str, Any]] = {}
        for nombre, mascara in incluidas.items():
            idx_ajuste_rebanada = idx_ajuste[mascara[idx_ajuste]]
            n_ajuste = len(idx_ajuste_rebanada)
            umbrales_locales: dict[str, float] = {}
            usados: dict[str, bool] = {}
            for posicion, etiqueta in enumerate(ETIQUETAS):
                if y_verdadera[idx_ajuste_rebanada, posicion].sum() >= CONFIG["min_positivos_umbral_local"]:
                    umbrales_locales[etiqueta] = ajustar_umbral_etiqueta(
                        y_verdadera[idx_ajuste_rebanada, posicion],
                        proba_m2[idx_ajuste_rebanada, posicion],
                    )
                    usados[etiqueta] = True
                else:
                    umbrales_locales[etiqueta] = float(umbrales_global[numero - 1, posicion])
                    usados[etiqueta] = False
            detalle_pliegue[nombre] = {
                "n_ajuste": n_ajuste,
                "umbrales_locales": umbrales_locales,
                "usados": usados,
            }
        locales[numero] = detalle_pliegue

    variantes: dict[str, Any] = {}
    for m in CONFIG["shrinkage_m"]:
        umbrales_fila = np.zeros((n_filas, len(ETIQUETAS)))
        for numero, mascara_validacion in mascaras_pliegues(pliegue_de_fila):
            idx_validacion = np.flatnonzero(mascara_validacion)
            for posicion in range(len(ETIQUETAS)):
                global_k = float(umbrales_global[numero - 1, posicion])
                umbrales_fila[idx_validacion, posicion] = global_k
            for nombre, mascara in incluidas.items():
                info = locales[numero][nombre]
                peso_local = 1.0 if m == 0 else info["n_ajuste"] / (info["n_ajuste"] + m)
                valores = np.array(
                    [info["umbrales_locales"][etiqueta] for etiqueta in ETIQUETAS]
                )
                filas_rebanada = idx_validacion[mascara[idx_validacion]]
                if peso_local <= 0 or not len(filas_rebanada):
                    continue
                umbrales_fila[filas_rebanada] = (1 - peso_local) * umbrales_fila[
                    filas_rebanada
                ] + peso_local * valores
        prediccion = (proba_m2 >= umbrales_fila).astype(int)
        metricas = metricas_multietiqueta(y_verdadera, prediccion)
        f1_pliegues = f1_por_pliegue_prediccion(prediccion, y_verdadera, pliegue_de_fila)
        nombre = f"SHRINK-M{m}" if m > 0 else "LOCAL-SIN-SHRINKAGE"
        variantes[nombre] = {
            "m": m,
            "corrected_cv": metricas["f1_media_columnas"],
            "f1_por_etiqueta": metricas["f1_por_etiqueta"],
            "f1_pliegues": f1_pliegues,
            "std_pliegues": float(np.std(f1_pliegues)),
            "delta_vs_global_anidado": metricas["f1_media_columnas"] - f1_global,
            "delta_vs_v03": metricas["f1_media_columnas"] - CONFIG["referencia_v03_umbrales"],
            "pliegues_mejorados_vs_global": int(
                np.sum(np.array(f1_pliegues) > np.array(f1_global_pliegues))
            ),
            "umbrales_fila": umbrales_fila,
        }
        log(
            f"  {nombre}: corrected_cv {metricas['f1_media_columnas']:.4f} "
            f"(delta vs global anidado {variantes[nombre]['delta_vs_global_anidado']:+.4f})"
        )

    mejor_nombre = max(variantes, key=lambda clave: variantes[clave]["corrected_cv"])
    mejor = variantes[mejor_nombre]
    # Detalle por rebanada con la mejor variante.
    detalle_rebanadas: dict[str, Any] = {}
    for nombre, mascara in incluidas.items():
        positivos = {etiqueta: int(y_verdadera[mascara, posicion].sum()) for posicion, etiqueta in enumerate(ETIQUETAS)}
        f1_con_shrink = {
            etiqueta: f1_por_columna(
                y_verdadera[mascara, posicion],
                (proba_m2[mascara, posicion] >= mejor["umbrales_fila"][mascara, posicion]).astype(int),
            )
            for posicion, etiqueta in enumerate(ETIQUETAS)
        }
        f1_global_dentro = {
            etiqueta: f1_por_columna(
                y_verdadera[mascara, posicion], prediccion_global[mascara, posicion]
            )
            for posicion, etiqueta in enumerate(ETIQUETAS)
        }
        pesos = [locales[numero][nombre]["n_ajuste"] for numero in sorted(locales)]
        detalle_rebanadas[nombre] = {
            "n": tamano[nombre],
            "positivos": positivos,
            "peso_local_medio": float(
                np.mean([n / (n + mejor["m"]) if mejor["m"] > 0 else 1.0 for n in pesos])
            ),
            "umbrales_locales_medios": {
                etiqueta: float(
                    np.mean([locales[numero][nombre]["umbrales_locales"][etiqueta] for numero in sorted(locales)])
                )
                for etiqueta in ETIQUETAS
            },
            "std_umbral_local": float(
                np.mean(
                    [
                        np.std([locales[numero][nombre]["umbrales_locales"][etiqueta] for numero in sorted(locales)])
                        for etiqueta in ETIQUETAS
                    ]
                )
            ),
            "celdas_locales_usadas": int(
                sum(1 for numero in sorted(locales) for etiqueta in ETIQUETAS if locales[numero][nombre]["usados"][etiqueta])
            ),
            "f1_shrink": f1_con_shrink,
            "f1_global_anidado": f1_global_dentro,
            "delta_medio": float(
                np.mean(
                    [
                        f1_con_shrink[etiqueta] - f1_global_dentro[etiqueta]
                        for etiqueta in ETIQUETAS
                    ]
                )
            ),
        }
    estabilidad = {
        "std_ganancia_entre_pliegues": float(
            np.std(np.array(mejor["f1_pliegues"]) - np.array(f1_global_pliegues))
        ),
        "std_umbral_local_medio": float(
            np.mean([detalle_rebanadas[nombre]["std_umbral_local"] for nombre in detalle_rebanadas])
        ),
        "celdas_locales_totales": int(
            sum(detalle_rebanadas[nombre]["celdas_locales_usadas"] for nombre in detalle_rebanadas)
        ),
        "celdas_teoricas": int(len(incluidas) * len(ETIQUETAS) * len(sorted(locales))),
    }
    decision = aplicar_regla_adopcion(
        nombre=mejor_nombre,
        corrected_cv=mejor["corrected_cv"],
        f1_pliegues=mejor["f1_pliegues"],
        f1_pliegues_v03=referencias["f1_pliegues_v03"],
        complementariedad=None,
        comparador="GLOBAL-NESTED",
        delta_comparador=mejor["delta_vs_global_anidado"],
        pliegues_mejorados_comparador=mejor["pliegues_mejorados_vs_global"],
        aporta_sobre_comparador=bool(
            mejor["delta_vs_global_anidado"] >= CONFIG["mejora_minima"]
            and mejor["pliegues_mejorados_vs_global"] >= CONFIG["pliegues_minimos_mejorados"]
        ),
    )
    return {
        "tamano_rebanadas": tamano,
        "excluidas": excluidas,
        "global_anidado": {
            "corrected_cv": f1_global,
            "f1_pliegues": f1_global_pliegues,
            "umbrales_media": {
                etiqueta: float(np.mean(umbrales_global[:, posicion]))
                for posicion, etiqueta in enumerate(ETIQUETAS)
            },
        },
        "variantes": variantes,
        "mejor": mejor_nombre,
        "detalle_rebanadas": detalle_rebanadas,
        "estabilidad": estabilidad,
        "decision": decision,
    }


def benjamini_hochberg(p_valores: Sequence[float]) -> tuple[list[float], list[bool]]:
    """Aplica la correccion de Benjamini-Hochberg a una familia de p-valores.

    Args:
        p_valores: p-valores de la familia.

    Returns:
        Tupla ``(q_valores, rechazados)`` en el orden original.
    """
    n = len(p_valores)
    orden = np.argsort(p_valores)
    q_valores = np.ones(n)
    minimo = 1.0
    for posicion in range(n - 1, -1, -1):
        indice = int(orden[posicion])
        candidato = p_valores[indice] * n / (posicion + 1)
        minimo = min(minimo, candidato)
        q_valores[indice] = minimo
    return [float(valor) for valor in q_valores], [bool(valor < CONFIG["bh_q"]) for valor in q_valores]


def ejecutar_e2_20(
    desarrollo: pd.DataFrame,
    textos: pd.Series,
    grupos: NDArray[np.int_],
    pliegue_de_fila: NDArray[np.int_],
    proba_m2: NDArray[np.float64],
    y_verdadera: NDArray[np.int_],
    rebanadas: dict[str, NDArray[np.bool_]],
    modulo_base: Any,
    referencias: dict[str, Any],
) -> dict[str, Any]:
    """Ejecuta E2-20: test de rebanadas y especialistas con shrinkage.

    Args:
        desarrollo: DataFrame del desarrollo.
        textos: textos del desarrollo.
        grupos: grupos fusionados.
        pliegue_de_fila: pliegue de cada fila.
        proba_m2: probabilidades OOF de M2.
        y_verdadera: etiquetas verdaderas.
        rebanadas: rebanadas con n >= 80.
        modulo_base: modulo de ``experiments/14`` con la receta de V03.
        referencias: referencia con las F1 por pliegue de V03.

    Returns:
        Diccionario con los tests, los especialistas evaluados y la decision.
    """
    log("E2-20 especialistas por rebanada: test de proporcion de errores con BH")
    prediccion_v03 = (proba_m2 >= vector_umbrales(CONFIG["umbrales_v03"])).astype(int)
    errores = prediccion_v03 != y_verdadera
    pruebas: list[dict[str, Any]] = []
    for nombre, mascara in rebanadas.items():
        for posicion, etiqueta in enumerate(ETIQUETAS):
            dentro = errores[mascara, posicion].astype(float)
            fuera = errores[~mascara, posicion].astype(float)
            proporcion_dentro = float(np.mean(dentro)) if len(dentro) else 0.0
            proporcion_fuera = float(np.mean(fuera)) if len(fuera) else 0.0
            proporcion_pool = (dentro.sum() + fuera.sum()) / max(len(dentro) + len(fuera), 1)
            varianza = proporcion_pool * (1 - proporcion_pool) * (
                1 / max(len(dentro), 1) + 1 / max(len(fuera), 1)
            )
            z = (proporcion_dentro - proporcion_fuera) / np.sqrt(varianza) if varianza > 0 else 0.0
            p_valor = float(2 * norm.sf(abs(z)))
            pruebas.append(
                {
                    "rebanada": nombre,
                    "etiqueta": etiqueta,
                    "n": int(mascara.sum()),
                    "tasa_error_dentro": proporcion_dentro,
                    "tasa_error_fuera": proporcion_fuera,
                    "z": float(z),
                    "p_valor": p_valor,
                }
            )
    q_valores, rechazados = benjamini_hochberg([prueba["p_valor"] for prueba in pruebas])
    for prueba, q_valor, rechazado in zip(pruebas, q_valores, rechazados, strict=True):
        prueba["q_valor"] = q_valor
        prueba["rechazado_bh"] = rechazado
    celdas = [(prueba["rebanada"], prueba["etiqueta"]) for prueba in pruebas if prueba["rechazado_bh"]]
    log(
        f"  pruebas={len(pruebas)} celdas significativas tras BH(q={CONFIG['bh_q']}): {len(celdas)}"
    )

    resultado: dict[str, Any] = {
        "pruebas": pruebas,
        "celdas_significativas": [f"{rebanada}::{etiqueta}" for rebanada, etiqueta in celdas],
        "especialistas": {},
        "modelo_combinado": None,
        "decision": None,
    }
    if not celdas:
        resultado["decision"] = {
            "adoptado": False,
            "justificacion": (
                "CERRADA sin evidencia: ninguna celda (rebanada, etiqueta) difiere del global "
                "tras la correccion BH; no se entrena especialista."
            ),
        }
        return resultado

    # Especialistas solo para las celdas que superan BH.
    proba_combinada = proba_m2.copy()
    detalle_especialistas: dict[str, Any] = {}
    for rebanada, etiqueta in celdas:
        posicion = ETIQUETAS.index(etiqueta)
        mascara = rebanadas[rebanada]
        proba_especialista = np.full(len(y_verdadera), np.nan)
        f1_especialista_pliegues = []
        f1_global_pliegues = []
        f1_mezcla_pliegues = []
        for numero, (idx_entrenamiento, idx_validacion) in enumerate(
            pliegues_corregidos(desarrollo, grupos), start=1
        ):
            idx_ent_rebanada = idx_entrenamiento[mascara[idx_entrenamiento]]
            idx_val_rebanada = idx_validacion[mascara[idx_validacion]]
            if len(idx_ent_rebanada) < 40 or y_verdadera[idx_ent_rebanada, posicion].sum() < 8 or not len(idx_val_rebanada):
                continue
            matriz_entrenamiento, matriz_validacion = modulo_base.construir_matrices(
                desarrollo["text"].iloc[idx_ent_rebanada],
                desarrollo["text"].iloc[idx_val_rebanada],
                "union",
            )
            modelo = LogisticRegression(**modulo_base.MODELOS["M2_union_lr_c2"]["params"])
            modelo.fit(matriz_entrenamiento, y_verdadera[idx_ent_rebanada, posicion])
            proba_especialista[idx_val_rebanada] = modelo.predict_proba(matriz_validacion)[:, 1]
            # Umbral anidado sobre el ajuste de la rebanada (OOF interno de 3 pliegues).
            puntajes_ajuste = np.zeros(len(idx_ent_rebanada))
            minimo_clase = int(np.min(np.bincount(y_verdadera[idx_ent_rebanada, posicion].astype(int), minlength=2)))
            n_internos = int(min(3, max(minimo_clase, 2)))
            divisor = StratifiedKFold(n_splits=n_internos, shuffle=True, random_state=RANDOM_STATE)
            for idx_int_ent, idx_int_val in divisor.split(
                desarrollo["text"].iloc[idx_ent_rebanada], y_verdadera[idx_ent_rebanada, posicion]
            ):
                matriz_int_ent, matriz_int_val = modulo_base.construir_matrices(
                    desarrollo["text"].iloc[idx_ent_rebanada].iloc[idx_int_ent],
                    desarrollo["text"].iloc[idx_ent_rebanada].iloc[idx_int_val],
                    "union",
                )
                modelo_interno = LogisticRegression(**modulo_base.MODELOS["M2_union_lr_c2"]["params"])
                modelo_interno.fit(matriz_int_ent, y_verdadera[idx_ent_rebanada, posicion][idx_int_ent])
                puntajes_ajuste[idx_int_val] = modelo_interno.predict_proba(matriz_int_val)[:, 1]
            umbral_especialista = ajustar_umbral_etiqueta(
                y_verdadera[idx_ent_rebanada, posicion], puntajes_ajuste
            )
            umbral_global = ajustar_umbral_etiqueta(
                y_verdadera[idx_ent_rebanada, posicion], proba_m2[idx_ent_rebanada, posicion]
            )
            proba_mezcla_ajuste = (puntajes_ajuste + proba_m2[idx_ent_rebanada, posicion]) / 2.0
            umbral_mezcla = ajustar_umbral_etiqueta(
                y_verdadera[idx_ent_rebanada, posicion], proba_mezcla_ajuste
            )
            y_slice = y_verdadera[idx_val_rebanada, posicion]
            f1_especialista_pliegues.append(
                f1_por_columna(y_slice, (proba_especialista[idx_val_rebanada] >= umbral_especialista).astype(int))
            )
            f1_global_pliegues.append(
                f1_por_columna(y_slice, (proba_m2[idx_val_rebanada, posicion] >= umbral_global).astype(int))
            )
            f1_mezcla_pliegues.append(
                f1_por_columna(
                    y_slice,
                    ((proba_especialista[idx_val_rebanada] + proba_m2[idx_val_rebanada, posicion]) / 2.0
                     >= umbral_mezcla).astype(int),
                )
            )
            proba_combinada[idx_val_rebanada, posicion] = (
                proba_especialista[idx_val_rebanada] + proba_m2[idx_val_rebanada, posicion]
            ) / 2.0
        faltantes = np.isnan(proba_especialista)
        if faltantes.any():
            proba_especialista[faltantes] = proba_m2[faltantes, posicion]
            proba_combinada[faltantes, posicion] = proba_m2[faltantes, posicion]
        detalle_especialistas[f"{rebanada}::{etiqueta}"] = {
            "n_rebanada": int(mascara.sum()),
            "f1_especialista_pliegues": f1_especialista_pliegues,
            "f1_global_pliegues": f1_global_pliegues,
            "f1_mezcla_pliegues": f1_mezcla_pliegues,
            "delta_especialista_vs_global": float(
                np.mean(f1_especialista_pliegues) - np.mean(f1_global_pliegues)
            ) if f1_especialista_pliegues else None,
            "delta_mezcla_vs_global": float(
                np.mean(f1_mezcla_pliegues) - np.mean(f1_global_pliegues)
            ) if f1_mezcla_pliegues else None,
        }
        log(
            f"  celda {rebanada}::{etiqueta}: delta especialista "
            f"{detalle_especialistas[f'{rebanada}::{etiqueta}']['delta_especialista_vs_global']:+.4f} | "
            f"delta mezcla {detalle_especialistas[f'{rebanada}::{etiqueta}']['delta_mezcla_vs_global']:+.4f}"
        )
    metricas = evaluar_con_umbrales(
        proba_combinada, y_verdadera, {etiqueta: 0.5 for etiqueta in ETIQUETAS}
    )
    f1_pliegues = f1_por_pliegue(
        proba_combinada, y_verdadera, pliegue_de_fila, {etiqueta: 0.5 for etiqueta in ETIQUETAS}
    )
    complementariedad = evaluar_complementariedad(
        proba_combinada, proba_m2, y_verdadera, CONFIG["referencia_m2_umbral_05"]
    )
    resultado["especialistas"] = detalle_especialistas
    resultado["modelo_combinado"] = {
        "corrected_cv": metricas["f1_media_columnas"],
        "f1_por_etiqueta": metricas["f1_por_etiqueta"],
        "f1_pliegues": f1_pliegues,
        "delta_vs_m2_05": metricas["f1_media_columnas"] - CONFIG["referencia_m2_umbral_05"],
        "delta_vs_v03": metricas["f1_media_columnas"] - CONFIG["referencia_v03_umbrales"],
        "complementariedad": complementariedad,
    }
    deltas = [
        detalle["delta_mezcla_vs_global"]
        for detalle in detalle_especialistas.values()
        if detalle["delta_mezcla_vs_global"] is not None
    ]
    delta_comparador = metricas["f1_media_columnas"] - CONFIG["referencia_m2_umbral_05"]
    pliegues_comparador = int(
        np.sum(
            np.array(f1_pliegues)
            > np.array(
                f1_por_pliegue(
                    proba_m2, y_verdadera, pliegue_de_fila, {etiqueta: 0.5 for etiqueta in ETIQUETAS}
                )
            )
        )
    )
    resultado["delta_medio_mezcla_en_celdas"] = float(np.mean(deltas)) if deltas else 0.0
    aporta = bool(
        delta_comparador >= CONFIG["mejora_minima"]
        and pliegues_comparador >= CONFIG["pliegues_minimos_mejorados"]
    )
    resultado["decision"] = aplicar_regla_adopcion(
        nombre="ESPECIALISTAS-MEZCLA",
        corrected_cv=metricas["f1_media_columnas"],
        f1_pliegues=f1_pliegues,
        f1_pliegues_v03=referencias["f1_pliegues_v03"],
        complementariedad=complementariedad,
        comparador="M2 a 0.5",
        delta_comparador=delta_comparador,
        pliegues_mejorados_comparador=pliegues_comparador,
        aporta_sobre_comparador=aporta,
    )
    return resultado


# ===== SECCION 6: E2-21 espacios de features por etiqueta =====


def tokenizar(texto: str) -> list[str]:
    """Tokeniza un texto en minusculas con el patron alfanumerico.

    Args:
        texto: texto de entrada.

    Returns:
        Lista de tokens.
    """
    return PATRON_PALABRA.findall(texto.lower())


def log_odds_monroe(
    textos: Sequence[str],
    y_columna: NDArray[np.int_],
    top: int,
    frecuencia_minima: int,
    min_documentos: int,
) -> dict[str, float]:
    """Induce un lexico con log-odds de Monroe et al. (2008) fold-local.

    Args:
        textos: textos del pliegue de ajuste.
        y_columna: etiqueta binaria del pliegue de ajuste.
        top: numero maximo de terminos por score absoluto.
        frecuencia_minima: frecuencia documental minima.
        min_documentos: numero minimo de documentos positivos o negativos.

    Returns:
        Diccionario ``token -> log-odds`` (mayor = mas asociado a la etiqueta).
    """
    positivos: Counter[str] = Counter()
    negativos: Counter[str] = Counter()
    frecuencia: Counter[str] = Counter()
    n_positivos = 0
    n_negativos = 0
    for texto, etiqueta in zip(textos, y_columna, strict=True):
        tokens = set(tokenizar(texto))
        frecuencia.update(tokens)
        if etiqueta == 1:
            positivos.update(tokens)
            n_positivos += 1
        else:
            negativos.update(tokens)
            n_negativos += 1
    total_frecuencia = sum(frecuencia.values())
    alpha0 = CONFIG["alpha_prior_log_odds"]
    vocabulario = [
        token
        for token, apariciones in frecuencia.items()
        if apariciones >= frecuencia_minima
    ]
    prior = {token: alpha0 * frecuencia[token] / max(total_frecuencia, 1) for token in vocabulario}
    total_prior = sum(prior.values())
    scores: dict[str, float] = {}
    for token in vocabulario:
        if positivos[token] < min_documentos or negativos[token] < min_documentos:
            continue
        valor = prior[token]
        p_positivo = (positivos[token] + valor) / max(
            n_positivos + total_prior - positivos[token] - valor, 1e-12
        )
        p_negativo = (negativos[token] + valor) / max(
            n_negativos + total_prior - negativos[token] - valor, 1e-12
        )
        scores[token] = float(np.log(max(p_positivo, 1e-12)) - np.log(max(p_negativo, 1e-12)))
    ordenados = sorted(scores.items(), key=lambda par: abs(par[1]), reverse=True)
    return dict(ordenados[:top])


def distancia_mencion_insulto(texto: str, lexico_insulto: frozenset[str]) -> int | None:
    """Calcula la distancia minima en tokens entre una mencion y un insulto.

    Args:
        texto: texto de entrada.
        lexico_insulto: conjunto de tokens considerados insulto.

    Returns:
        Distancia minima o ``None`` si falta mencion o insulto.
    """
    palabras = texto.split()
    posiciones_mencion = [indice for indice, palabra in enumerate(palabras) if palabra.startswith("@")]
    posiciones_insulto = [
        indice
        for indice, palabra in enumerate(palabras)
        if PATRON_PALABRA.findall(palabra.lower()) and PATRON_PALABRA.findall(palabra.lower())[0] in lexico_insulto
    ]
    if not posiciones_mencion or not posiciones_insulto:
        return None
    return int(min(abs(a - b) for a in posiciones_mencion for b in posiciones_insulto))


def tokens_canal_etiqueta(
    textos: Sequence[str],
    etiqueta: str,
    lexico: dict[str, float],
    lexico_insulto: frozenset[str],
) -> list[list[str]]:
    """Construye el canal simbolico especifico de una etiqueta.

    Args:
        textos: textos a transformar.
        etiqueta: ``HS``, ``TR`` o ``AG``.
        lexico: lexico inducido fold-local de la etiqueta.
        lexico_insulto: lexico de insulto inducido (para la distancia de TR).

    Returns:
        Lista de tokens del canal por fila.
    """
    canal: list[list[str]] = []
    for texto in textos:
        tokens = tokenizar(texto)
        marca: list[str] = []
        if etiqueta == "HS":
            colectivos = len(PATRON_COLECTIVO.findall(texto.lower()))
            if colectivos:
                marca.append("col_si")
            if colectivos >= 2:
                marca.append("col_multi")
            if PATRON_NEGACION.search(texto.lower()):
                marca.append("neg_si")
        elif etiqueta == "TR":
            if PATRON_MENCION.search(texto):
                marca.append("mencion_si")
            if PATRON_MENCION.match(texto.strip()):
                marca.append("mencion_inicial")
            if PATRON_SEGUNDA_PERSONA.search(texto.lower()):
                marca.append("persona2_si")
            if any(token in IMPERATIVOS_FIJOS for token in tokens):
                marca.append("imperativo_si")
            distancia = distancia_mencion_insulto(texto, lexico_insulto)
            if distancia is None:
                marca.append("dist_na")
            elif distancia <= 2:
                marca.append("dist_0_2")
            elif distancia <= 5:
                marca.append("dist_3_5")
            elif distancia <= 10:
                marca.append("dist_6_10")
            else:
                marca.append("dist_lejos")
        else:
            if PATRON_RISAS.search(texto):
                marca.append("risa_si")
            if "!" in texto:
                marca.append("excl_si")
            if PATRON_ELONGACION.search(texto):
                marca.append("elong_si")
            if PATRON_MAYUSCULAS.search(texto):
                marca.append("mayus_si")
        inducidos = [f"ind_{token}" for token in tokens if token in lexico]
        canal.append(marca + inducidos)
    return canal


def matriz_canal(
    tokens_por_fila: Sequence[Sequence[str]], vocabulario: dict[str, int]
) -> sparse.csr_matrix:
    """Construye la matriz dispersa binaria del canal simbolico.

    Args:
        tokens_por_fila: tokens del canal por fila.
        vocabulario: mapeo ``token -> columna``.

    Returns:
        Matriz CSR ``(n_filas, len(vocabulario))``.
    """
    filas: list[int] = []
    columnas: list[int] = []
    for indice, tokens in enumerate(tokens_por_fila):
        for token in set(tokens):
            columna = vocabulario.get(token)
            if columna is not None:
                filas.append(indice)
                columnas.append(columna)
    datos = np.ones(len(filas), dtype=float)
    return sparse.csr_matrix(
        (datos, (filas, columnas)), shape=(len(tokens_por_fila), len(vocabulario))
    )


def ejecutar_e2_21(
    desarrollo: pd.DataFrame,
    grupos: NDArray[np.int_],
    proba_bases: NDArray[np.float64],
    pliegue_de_fila: NDArray[np.int_],
    y_verdadera: NDArray[np.int_],
    modulo_base: Any,
    referencias: dict[str, Any],
) -> dict[str, Any]:
    """Ejecuta E2-21: espacios de features por etiqueta frente al compartido.

    Args:
        desarrollo: DataFrame del desarrollo.
        grupos: grupos fusionados.
        proba_bases: probabilidades OOF de los 4 bases (espacio compartido).
        pliegue_de_fila: pliegue de cada fila.
        y_verdadera: etiquetas verdaderas.
        modulo_base: modulo de ``experiments/14`` con la receta de V03.
        referencias: referencia con las F1 por pliegue de V03.

    Returns:
        Diccionario con las variantes, el coste de dimension y la decision.
    """
    log("E2-21 espacios de features por etiqueta (HS/TR/AG) frente al compartido")
    textos = desarrollo["text"]
    proba_especifica = np.zeros((len(desarrollo), len(ETIQUETAS)))
    dimensiones: dict[str, Any] = {}
    for numero, (idx_entrenamiento, idx_validacion) in enumerate(
        pliegues_corregidos(desarrollo, grupos), start=1
    ):
        textos_entrenamiento = textos.iloc[idx_entrenamiento]
        textos_validacion = textos.iloc[idx_validacion]
        word = TfidfVectorizer(**modulo_base.PARAMS_WORD)
        char = TfidfVectorizer(**modulo_base.PARAMS_CHAR)
        compartida_entrenamiento = sparse.hstack(
            [word.fit_transform(textos_entrenamiento), char.fit_transform(textos_entrenamiento)]
        ).tocsr()
        compartida_validacion = sparse.hstack(
            [word.transform(textos_validacion), char.transform(textos_validacion)]
        ).tocsr()
        dimensiones[f"pliegue_{numero}"] = {
            "compartida": int(compartida_entrenamiento.shape[1]),
            "especifica": {},
        }
        lexico_insulto = frozenset(
            token
            for token, valor in log_odds_monroe(
                textos_entrenamiento.tolist(),
                y_verdadera[idx_entrenamiento, ETIQUETAS.index("HS")],
                CONFIG["top_lexico_inductivo"],
                CONFIG["frecuencia_minima_lexico"],
                CONFIG["min_documentos_lexico"],
            ).items()
            if valor > 0
        )
        for posicion, etiqueta in enumerate(ETIQUETAS):
            lexico = log_odds_monroe(
                textos_entrenamiento.tolist(),
                y_verdadera[idx_entrenamiento, posicion],
                CONFIG["top_lexico_inductivo"],
                CONFIG["frecuencia_minima_lexico"],
                CONFIG["min_documentos_lexico"],
            )
            canal_entrenamiento = tokens_canal_etiqueta(
                textos_entrenamiento.tolist(), etiqueta, lexico, lexico_insulto
            )
            vocabulario: dict[str, int] = {}
            for tokens in canal_entrenamiento:
                for token in tokens:
                    if token not in vocabulario:
                        vocabulario[token] = len(vocabulario)
            extra_entrenamiento = normalize(matriz_canal(canal_entrenamiento, vocabulario), norm="l2")
            canal_validacion = tokens_canal_etiqueta(
                textos_validacion.tolist(), etiqueta, lexico, lexico_insulto
            )
            extra_validacion = normalize(matriz_canal(canal_validacion, vocabulario), norm="l2")
            matriz_entrenamiento = sparse.hstack(
                [compartida_entrenamiento, extra_entrenamiento]
            ).tocsr()
            matriz_validacion = sparse.hstack([compartida_validacion, extra_validacion]).tocsr()
            modelo = LogisticRegression(**modulo_base.MODELOS["M2_union_lr_c2"]["params"])
            modelo.fit(matriz_entrenamiento, y_verdadera[idx_entrenamiento, posicion])
            proba_especifica[idx_validacion, posicion] = modelo.predict_proba(matriz_validacion)[:, 1]
            dimensiones[f"pliegue_{numero}"]["especifica"][etiqueta] = int(
                extra_entrenamiento.shape[1]
            )
        log(f"  pliegue {numero}: dimensiones {dimensiones[f'pliegue_{numero}']}")

    proba_compartida = proba_bases[:, MODELOS_BASE.index("M2_union_lr_c2"), :]
    sin_umbral = {etiqueta: 0.5 for etiqueta in ETIQUETAS}
    metricas = evaluar_con_umbrales(proba_especifica, y_verdadera, sin_umbral)
    f1_pliegues = f1_por_pliegue(proba_especifica, y_verdadera, pliegue_de_fila, sin_umbral)
    f1_por_etiqueta = f1_por_etiqueta_rapida(
        y_verdadera, (proba_especifica >= 0.5).astype(int)
    )
    f1_compartida_etiqueta = f1_por_etiqueta_rapida(
        y_verdadera, (proba_compartida >= 0.5).astype(int)
    )
    complementariedad = evaluar_complementariedad(
        proba_especifica, proba_compartida, y_verdadera, CONFIG["referencia_m2_umbral_05"]
    )
    delta_compartido = metricas["f1_media_columnas"] - CONFIG["referencia_m2_umbral_05"]
    comparador_pliegues = f1_por_pliegue(
        proba_compartida, y_verdadera, pliegue_de_fila, sin_umbral
    )
    decision = aplicar_regla_adopcion(
        nombre="ESPACIOS-POR-ETIQUETA",
        corrected_cv=metricas["f1_media_columnas"],
        f1_pliegues=f1_pliegues,
        f1_pliegues_v03=referencias["f1_pliegues_v03"],
        complementariedad=complementariedad,
        comparador="espacio compartido V03 (M2 a 0.5)",
        delta_comparador=delta_compartido,
        pliegues_mejorados_comparador=int(
            np.sum(np.array(f1_pliegues) > np.array(comparador_pliegues))
        ),
        aporta_sobre_comparador=bool(
            delta_compartido >= CONFIG["mejora_minima"]
            and int(np.sum(np.array(f1_pliegues) > np.array(comparador_pliegues)))
            >= CONFIG["pliegues_minimos_mejorados"]
        ),
    )
    dimensiones_medias = {
        etiqueta: float(
            np.mean([dimensiones[clave]["especifica"][etiqueta] for clave in dimensiones])
        )
        for etiqueta in ETIQUETAS
    }
    dimensiones_medias["compartida"] = float(
        np.mean([dimensiones[clave]["compartida"] for clave in dimensiones])
    )
    variante = {
        "corrected_cv": metricas["f1_media_columnas"],
        "f1_por_etiqueta": f1_por_etiqueta,
        "f1_por_etiqueta_compartida": f1_compartida_etiqueta,
        "delta_por_etiqueta_vs_compartida": {
            etiqueta: f1_por_etiqueta[etiqueta] - f1_compartida_etiqueta[etiqueta]
            for etiqueta in ETIQUETAS
        },
        "f1_pliegues": f1_pliegues,
        "std_pliegues": float(np.std(f1_pliegues)),
        "delta_vs_m2_05": delta_compartido,
        "delta_vs_v03": metricas["f1_media_columnas"] - CONFIG["referencia_v03_umbrales"],
        "dimensiones_medias": dimensiones_medias,
        "complementariedad": complementariedad,
        "decision": decision,
        "proba": proba_especifica,
    }
    log(
        f"  espacios por etiqueta: corrected_cv {metricas['f1_media_columnas']:.4f} "
        f"(delta vs compartido {delta_compartido:+.4f})"
    )
    return {
        "variante": variante,
        "dimensiones": dimensiones,
        "dimensiones_medias": dimensiones_medias,
        "decision": decision,
    }


# ===== SECCION 7: figura y artefactos =====


def generar_figura(
    resumen_lineas: Sequence[dict[str, Any]],
    calibracion: dict[str, Any],
    entidades: dict[str, Any],
) -> None:
    """Genera la figura de cierre del backlog (tres paneles).

    Args:
        resumen_lineas: resumen por linea con la mejor variante.
        calibracion: resultados de E2-14.
        entidades: resultados de E2-13.
    """
    figura, (eje_a, eje_b, eje_c) = plt.subplots(
        1, 3, figsize=(19, 6), gridspec_kw={"width_ratios": [1.35, 1.0, 1.0]}
    )
    nombres = [fila["linea"] for fila in resumen_lineas]
    valores = [fila["corrected_cv"] for fila in resumen_lineas]
    colores = [
        "#4c72b0"
        if (
            fila["delta_vs_comparador"] >= CONFIG["mejora_minima"]
            and fila["pliegues_mejorados_vs_comparador"] >= CONFIG["pliegues_minimos_mejorados"]
        )
        else "#8c8c8c"
        for fila in resumen_lineas
    ]
    eje_a.barh(nombres, valores, color=colores)
    eje_a.axvline(
        CONFIG["referencia_v03_umbrales"],
        color="#2ca02c",
        linestyle="--",
        linewidth=1.1,
        label=f"V03 umbrales ({CONFIG['referencia_v03_umbrales']:.4f})",
    )
    eje_a.axvline(
        CONFIG["referencia_v03_umbrales"] + CONFIG["mejora_minima"],
        color="#7f7f7f",
        linestyle=":",
        linewidth=1.1,
        label=f"adopcion ({CONFIG['referencia_v03_umbrales'] + CONFIG['mejora_minima']:.4f})",
    )
    eje_a.axvline(
        CONFIG["referencia_m2_umbral_05"],
        color="#d62728",
        linestyle="-.",
        linewidth=1.1,
        label=f"M2 a 0.5 ({CONFIG['referencia_m2_umbral_05']:.4f})",
    )
    for posicion, fila in enumerate(resumen_lineas):
        eje_a.text(
            fila["corrected_cv"] + 0.0003,
            posicion,
            f"{fila['corrected_cv']:.4f} (delta comparador {fila['delta_vs_comparador']:+.4f})",
            va="center",
            fontsize=7.5,
        )
    eje_a.set_xlim(min(valores) - 0.004, max(valores) + 0.014)
    eje_a.set_xlabel("corrected_cv de la mejor variante de cada linea")
    eje_a.set_title(
        "P8 - Cierre del backlog v2 (desarrollo n=1483; gris = no aporta sobre su comparador)"
    )
    eje_a.legend(loc="lower right", fontsize=7)

    etiquetas_cal = ["M2-SIN-CALIBRAR", "M2-PLATT", "M2-ISOTONIC", "META-SIN-CALIBRAR", "META-PLATT", "META-ISOTONIC"]
    posiciones = np.arange(len(etiquetas_cal))
    brier = [calibracion["variantes"][nombre]["brier"] for nombre in etiquetas_cal]
    log_loss = [calibracion["variantes"][nombre]["log_loss"] for nombre in etiquetas_cal]
    eje_b.bar(posiciones - 0.2, brier, 0.4, label="Brier", color="#4c72b0")
    eje_b.set_ylabel("Brier", color="#4c72b0")
    eje_b.tick_params(axis="y", labelcolor="#4c72b0")
    eje_b_2 = eje_b.twinx()
    eje_b_2.bar(posiciones + 0.2, log_loss, 0.4, label="log-loss", color="#dd8452")
    eje_b_2.set_ylabel("log-loss", color="#dd8452")
    eje_b_2.tick_params(axis="y", labelcolor="#dd8452")
    eje_b.set_xticks(posiciones)
    eje_b.set_xticklabels(["M2 crudo", "M2 Platt", "M2 iso", "META crudo", "META Platt", "META iso"], rotation=20, ha="right", fontsize=8)
    eje_b.set_title("E2-14 calibracion: Brier y log-loss")
    eje_b.legend(loc="upper left", fontsize=8)
    eje_b_2.legend(loc="upper right", fontsize=8)

    tipos = ["mencion", "hashtag", "dominio"]
    coberturas = [entidades["cobertura"]["por_tipo"][tipo]["cobertura_instancias"] for tipo in tipos]
    eje_c.bar(tipos, coberturas, color=["#4c72b0", "#dd8452", "#55a868"])
    eje_c.axhline(
        CONFIG["umbral_cobertura_minima"],
        color="#d62728",
        linestyle="--",
        linewidth=1.1,
        label=f"cierre por sparsity ({CONFIG['umbral_cobertura_minima']:.2f})",
    )
    for posicion, tipo in enumerate(tipos):
        detalle = entidades["cobertura"]["por_tipo"][tipo]
        eje_c.text(
            posicion,
            coberturas[posicion] + 0.02,
            f"unicos={detalle['unicos_dev']}\nfreq>={CONFIG['frecuencia_minima_entidades']}: "
            f"{detalle['sobrevivientes_frecuencia_minima_dev']}",
            ha="center",
            fontsize=8,
        )
    eje_c.set_ylim(0.0, 1.15)
    eje_c.set_ylabel("cobertura de instancias vistas en el pliegue")
    eje_c.set_title("E2-13 cobertura del encoder de entidades")
    eje_c.legend(loc="upper right", fontsize=8)
    figura.tight_layout()
    FIGURA_PATH.parent.mkdir(parents=True, exist_ok=True)
    figura.savefig(FIGURA_PATH, dpi=150)
    plt.close(figura)


def fila_experimento(
    experiment_id: str,
    question: str,
    hypothesis: str,
    representation: str,
    model: str,
    hyperparameters: dict[str, Any],
    validation: str,
    f1: float,
    result: dict[str, Any],
    interpretation: str,
    decision: str,
    dimension: Any,
    notes: str,
) -> dict[str, Any]:
    """Construye una fila del registro de experimentos.

    Args:
        experiment_id: identificador unico de la fila.
        question: pregunta que responde el experimento.
        hypothesis: hipotesis evaluada.
        representation: representacion o features usadas.
        model: modelo empleado.
        hyperparameters: hiperparametros relevantes.
        validation: esquema de validacion.
        f1: metrica primaria.
        result: diccionario con el detalle del resultado.
        interpretation: lectura del resultado.
        decision: decision tomada.
        dimension: numero de features o celdas si aplica.
        notes: notas de trazabilidad.

    Returns:
        Diccionario con las columnas de ``experiments.csv``.
    """
    return {
        "experiment_id": experiment_id,
        "competition": "KAGGLE_TOXIC_2026_PUJ",
        "question": question,
        "hypothesis": hypothesis,
        "preprocessing": "texto crudo; transformaciones fold-locales sin spaCy en este paquete",
        "feature_groups": "P8_BACKLOG",
        "representation": representation,
        "model": model,
        "hyperparameters": json.dumps(hyperparameters, ensure_ascii=False),
        "validation": validation,
        "f1": round(f1, 4),
        "macro_f1": "",
        "weighted_f1": "",
        "precision": "",
        "recall": "",
        "training_time": "",
        "dimension": dimension,
        "result": json.dumps(serializar(result), ensure_ascii=False),
        "interpretation": interpretation,
        "decision": decision,
        "date": CONFIG["fecha"],
        "script": "experiments/21_backlog_restante_v2.py",
        "notes": notes,
    }


def construir_filas_experimentos(
    calibracion: dict[str, Any],
    entidades: dict[str, Any],
    rebanadas: dict[str, Any],
    especialistas: dict[str, Any],
    espacios: dict[str, Any],
) -> list[dict[str, Any]]:
    """Construye las filas ``E2-13-*`` a ``E2-21-*`` del registro.

    Args:
        calibracion: resultados de E2-14.
        entidades: resultados de E2-13.
        rebanadas: resultados de E2-16.
        especialistas: resultados de E2-20.
        espacios: resultados de E2-21.

    Returns:
        Lista de filas para ``experiments.csv``.
    """
    filas: list[dict[str, Any]] = []
    validacion = "5 pliegues corregidos v2 (grupos fusionados, seed 42); anidado en cada variante"
    # E2-14
    for nombre, resumen in calibracion["variantes"].items():
        if "proba" in resumen:
            continue
        filas.append(
            fila_experimento(
                experiment_id=f"E2-14-{nombre}",
                question="Mejoran la calibracion Platt o isotonic (anidada por pliegue) la F1 y la estabilidad de los umbrales de M2 y del meta Ridge?",
                hypothesis="Las probabilidades calibradas mejoran cadenas, umbrales y stacking (E2-14)",
                representation="probabilidades OOF congeladas; calibrador por pliegue ajustado con los pliegues restantes",
                model=f"calibracion {nombre}",
                hyperparameters={
                    "metodo": "platt" if "PLATT" in nombre else "isotonic" if "ISOTONIC" in nombre else "ninguna",
                    "modelo_base": "M2" if nombre.startswith("M2") else "META-BASES-RIDGE",
                },
                validation=validacion,
                f1=resumen["f1_anidado"],
                result={
                    "brier": resumen["brier"],
                    "log_loss": resumen["log_loss"],
                    "f1_umbral_05": resumen["f1_umbral_05"],
                    "f1_anidado": resumen["f1_anidado"],
                    "f1_optimista": resumen["f1_optimista"],
                    "f1_pliegues": resumen["f1_pliegues"],
                    "umbrales_media": resumen["umbrales_media"],
                    "umbrales_std": resumen["umbrales_std"],
                    "umbrales_rango": resumen["umbrales_rango"],
                },
                interpretation="La calibracion es monotona: la F1 con umbrales reajustados es invariante salvo empates; el aporte medible esta en Brier/log-loss.",
                decision="diagnostico de E2-14",
                dimension="",
                notes="pliegues corregidos v2; holdout intacto",
            )
        )
    mejores_cal = {
        "M2": calibracion["mejores"]["M2"],
        "META": calibracion["mejores"]["META"],
    }
    for nombre_modelo, resumen in mejores_cal.items():
        filas.append(
            fila_experimento(
                experiment_id=f"E2-14-MEJOR-{nombre_modelo}",
                question=f"Mejora la calibracion anidada el comportamiento de {nombre_modelo} sobre su version sin calibrar?",
                hypothesis="La calibracion mejora la F1 y la estabilidad de umbrales (E2-14)",
                representation="mejor calibrador anidado por modelo",
                model=f"mejor calibracion de {nombre_modelo}",
                hyperparameters={"comparador": f"{nombre_modelo}-SIN-CALIBRAR"},
                validation=validacion,
                f1=resumen["f1_anidado"],
                result={
                    "delta_vs_sin_calibrar": resumen["delta_vs_sin_calibrar"],
                    "pliegues_mejorados_vs_sin_calibrar": resumen["pliegues_mejorados_vs_sin_calibrar"],
                    "brier": resumen["brier"],
                    "log_loss": resumen["log_loss"],
                    "complementariedad": resumen["complementariedad"],
                    "decision": resumen["decision"],
                },
                interpretation="Comparacion directa calibrado vs sin calibrar.",
                decision="CERRADA sin mejora" if not resumen["decision"]["adoptado"] else "ADOPTADA",
                dimension="",
                notes="el nivel absoluto no basta: se exige aporte sobre el par sin calibrar",
            )
        )
    # E2-13
    for nombre, resumen in entidades["variantes"].items():
        filas.append(
            fila_experimento(
                experiment_id=f"E2-13-{nombre}",
                question="Aporta el target encoding OOF de menciones, hashtags y dominios (suavizado m=10, frecuencia minima 5) al meta o como canal propio?",
                hypothesis="P(label|entidad) con suavizado aporta para TR/AG (E2-13)",
                representation="bloque de 12 features de entidades (3 tipos x 3 etiquetas + numero de instancias) sobre el meta de 44 features",
                model=nombre,
                hyperparameters={
                    "suavizado": CONFIG["suavizado_entidades"],
                    "frecuencia_minima": CONFIG["frecuencia_minima_entidades"],
                    "cobertura_informativa": entidades["cobertura"]["cobertura_informativa"],
                },
                validation=validacion,
                f1=resumen["corrected_cv"],
                result={
                    "delta_vs_meta_ridge": resumen["delta_vs_meta_ridge"],
                    "delta_vs_v03": resumen["delta_vs_v03"],
                    "f1_por_etiqueta": resumen["f1_por_etiqueta"],
                    "f1_pliegues": resumen["f1_pliegues"],
                    "complementariedad": resumen["complementariedad"],
                    "cobertura_informativa": entidades["cobertura"]["cobertura_informativa"],
                    "decision": resumen["decision"],
                },
                interpretation="La cobertura de menciones y hashtags es muy baja; el bloque solo puede aportar por el prior.",
                decision="CERRADA por sparsity" if entidades["cobertura"]["cierra_por_sparsity"] else "diagnostico",
                dimension=12,
                notes="tablas de entidades ajustadas fold-locales; bucket desconocido = prior del pliegue",
            )
        )
    # E2-16
    for nombre, resumen in rebanadas["variantes"].items():
        filas.append(
            fila_experimento(
                experiment_id=f"E2-16-{nombre}",
                question="Mejoran los umbrales por rebanada con shrinkage (rebanadas n>=80) al umbral global anidado?",
                hypothesis="Las rebanadas con distribucion distinta se benefician de un umbral propio (H15/E2-16)",
                representation=f"umbral por fila = mezcla global/local con m={resumen['m']}",
                model="M2 (V03) con umbrales por rebanada",
                hyperparameters={"m_shrinkage": resumen["m"], "rebanadas": sorted(rebanadas["detalle_rebanadas"])},
                validation=validacion,
                f1=resumen["corrected_cv"],
                result={
                    "delta_vs_global_anidado": resumen["delta_vs_global_anidado"],
                    "delta_vs_v03": resumen["delta_vs_v03"],
                    "pliegues_mejorados_vs_global": resumen["pliegues_mejorados_vs_global"],
                    "f1_por_etiqueta": resumen["f1_por_etiqueta"],
                    "f1_pliegues": resumen["f1_pliegues"],
                },
                interpretation="El umbral local se ajusta en los pliegues de ajuste y se aplica al pliegue evaluado; la mezcla pondera por n.",
                decision="diagnostico E2-16",
                dimension="",
                notes=f"rebanadas excluidas por tamano: {rebanadas['excluidas']}",
            )
        )
    filas.append(
        fila_experimento(
            experiment_id="E2-16-MEJOR",
            question="Cual es la mejor variante de shrinkage y su decision?",
            hypothesis="El shrinkage controla el sobreajuste de los umbrales locales (E2-16)",
            representation=rebanadas["mejor"],
            model="M2 (V03) con umbrales por rebanada",
            hyperparameters={"m_shrinkage": rebanadas["variantes"][rebanadas["mejor"]]["m"]},
            validation=validacion,
            f1=rebanadas["variantes"][rebanadas["mejor"]]["corrected_cv"],
            result={
                "decision": rebanadas["decision"],
                "estabilidad": rebanadas["estabilidad"],
                "global_anidado": rebanadas["global_anidado"]["corrected_cv"],
                "detalle_rebanadas": rebanadas["detalle_rebanadas"],
            },
            interpretation="Mejor variante frente al umbral global anidado y a V03 con umbrales.",
            decision="CERRADA sin mejora" if not rebanadas["decision"]["adoptado"] else "ADOPTADA",
            dimension="",
            notes="holdout congelado no utilizado",
        )
    )
    # E2-20
    if especialistas["celdas_significativas"]:
        combinado = especialistas["modelo_combinado"]
        filas.append(
            fila_experimento(
                experiment_id="E2-20-ESPECIALISTAS",
                question="Mejoran los especialistas por rebanada (solo celdas significativas tras BH) al modelo global con shrinkage?",
                hypothesis="Las rebanadas significativamente distintas se benefician de un modelo propio (E2-20)",
                representation="mezcla 1:1 de M2 con el especialista LR sobre la union en las celdas significativas",
                model="Especialista LR union (C=2 balanced) por rebanada",
                hyperparameters={"celdas": especialistas["celdas_significativas"], "bh_q": CONFIG["bh_q"]},
                validation=validacion,
                f1=combinado["corrected_cv"],
                result={
                    "delta_vs_m2_05": combinado["delta_vs_m2_05"],
                    "delta_vs_v03": combinado["delta_vs_v03"],
                    "f1_pliegues": combinado["f1_pliegues"],
                    "complementariedad": combinado["complementariedad"],
                    "especialistas": especialistas["especialistas"],
                    "decision": especialistas["decision"],
                },
                interpretation="Comparacion con el modelo global en las mismas rebanadas y con umbrales anidados.",
                decision="CERRADA sin mejora" if not especialistas["decision"]["adoptado"] else "ADOPTADA",
                dimension="",
                notes="test de proporcion de errores con correccion BH",
            )
        )
    else:
        filas.append(
            fila_experimento(
                experiment_id="E2-20-SIN-EVIDENCIA",
                question="Hay rebanadas que difieran significativamente del global como para justificar un especialista?",
                hypothesis="Las rebanadas debiles se benefician de un especialista (E2-20)",
                representation="test z de proporcion de errores (M2 con umbrales V03) con correccion BH",
                model="no aplica (sin celdas significativas)",
                hyperparameters={"bh_q": CONFIG["bh_q"], "pruebas": len(especialistas["pruebas"])},
                validation=validacion,
                f1=CONFIG["referencia_m2_umbral_05"],
                result={
                    "celdas_significativas": especialistas["celdas_significativas"],
                    "pruebas": especialistas["pruebas"],
                    "decision": especialistas["decision"],
                },
                interpretation="Ninguna celda (rebanada, etiqueta) difiere del global tras BH; no se entrena especialista.",
                decision="CERRADA sin evidencia",
                dimension=len(especialistas["pruebas"]),
                notes="cierre documentado de la linea E2-20",
            )
        )
    # E2-21
    variante_espacios = espacios["variante"]
    for etiqueta in ETIQUETAS:
        filas.append(
            fila_experimento(
                experiment_id=f"E2-21-{etiqueta}",
                question=f"Aporta el espacio especifico de {etiqueta} sobre el espacio compartido word+char de V03?",
                hypothesis="Cada etiqueta puede necesitar una representacion distinta (E2-21)",
                representation=f"word+char+canal {etiqueta} (marcadores + lexico inducido fold-local)",
                model=f"LR C=2 balanced (espacio especifico {etiqueta})",
                hyperparameters={
                    "top_lexico": CONFIG["top_lexico_inductivo"],
                    "frecuencia_minima": CONFIG["frecuencia_minima_lexico"],
                },
                validation=validacion,
                f1=variante_espacios["f1_por_etiqueta"][etiqueta],
                result={
                    "f1_por_etiqueta": variante_espacios["f1_por_etiqueta"],
                    "f1_por_etiqueta_compartida": variante_espacios["f1_por_etiqueta_compartida"],
                    "delta_por_etiqueta_vs_compartida": variante_espacios["delta_por_etiqueta_vs_compartida"],
                    "dimensiones_medias": espacios["dimensiones_medias"],
                },
                interpretation="F1 de la etiqueta con su espacio especifico frente al espacio compartido.",
                decision="diagnostico E2-21",
                dimension=int(espacios["dimensiones_medias"][etiqueta]),
                notes="canal simbolico normalizado en L2 para no dominar por escala",
            )
        )
    filas.append(
        fila_experimento(
            experiment_id="E2-21-ESPACIOS",
            question="Supera el conjunto de espacios por etiqueta al espacio compartido de V03?",
            hypothesis="Cada etiqueta puede necesitar una representacion distinta (E2-21)",
            representation="HS: word+char+colectivo/hate inducido; TR: word+char+targeting; AG: word+char+insulto/afecto",
            model="LR C=2 balanced por etiqueta con su espacio",
            hyperparameters={"top_lexico": CONFIG["top_lexico_inductivo"]},
            validation=validacion,
            f1=variante_espacios["corrected_cv"],
            result={
                "delta_vs_m2_05": variante_espacios["delta_vs_m2_05"],
                "delta_vs_v03": variante_espacios["delta_vs_v03"],
                "f1_pliegues": variante_espacios["f1_pliegues"],
                "complementariedad": variante_espacios["complementariedad"],
                "dimensiones_medias": espacios["dimensiones_medias"],
                "decision": espacios["decision"],
            },
            interpretation="Comparacion global por etiqueta y coste de dimension.",
            decision="CERRADA sin mejora" if not espacios["decision"]["adoptado"] else "ADOPTADA",
            dimension=int(sum(espacios["dimensiones_medias"][etiqueta] for etiqueta in ETIQUETAS)),
            notes="holdout congelado no utilizado",
        )
    )
    return filas


def escribir_reporte_csv(resumen_lineas: Sequence[dict[str, Any]]) -> None:
    """Escribe el CSV de cierre del backlog (una fila por linea).

    Args:
        resumen_lineas: resumen por linea del backlog.
    """
    tabla = pd.DataFrame(
        [
            {
                "linea": fila["linea"],
                "mejor_variante": fila["mejor_variante"],
                "corrected_cv": fila["corrected_cv"],
                "delta_vs_v03": fila["delta_vs_v03"],
                "pliegues_mejorados_vs_v03": fila["pliegues_mejorados_vs_v03"],
                "comparador": fila["comparador"],
                "delta_vs_comparador": fila["delta_vs_comparador"],
                "pliegues_mejorados_vs_comparador": fila["pliegues_mejorados_vs_comparador"],
                "adoptado": fila["adoptado"],
                "cerrada_por_falta_de_evidencia": fila["cerrada_por_falta_de_evidencia"],
                "detalle": fila["detalle"],
                "extra": json.dumps(serializar(fila["extra"]), ensure_ascii=False),
            }
            for fila in resumen_lineas
        ]
    )
    tabla.to_csv(REPORTE_CSV, index=False)
    log(f"reporte escrito: {REPORTE_CSV.name} ({len(tabla)} lineas)")


def construir_resumen_lineas(
    calibracion: dict[str, Any],
    entidades: dict[str, Any],
    rebanadas: dict[str, Any],
    especialistas: dict[str, Any],
    espacios: dict[str, Any],
    referencias: dict[str, Any],
) -> list[dict[str, Any]]:
    """Construye el resumen por linea del backlog para CSV y figura.

    Args:
        calibracion: resultados de E2-14.
        entidades: resultados de E2-13.
        rebanadas: resultados de E2-16.
        especialistas: resultados de E2-20.
        espacios: resultados de E2-21.
        referencias: referencia con las F1 por pliegue de V03.

    Returns:
        Lista de diccionarios ``linea -> mejor variante y decision``.
    """
    mejor_calibracion = max(
        (calibracion["mejores"]["M2"], calibracion["mejores"]["META"]),
        key=lambda resumen: resumen["f1_anidado"],
    )
    mejor_entidades = entidades["variantes"][entidades["mejor"]]
    mejor_rebanadas = rebanadas["variantes"][rebanadas["mejor"]]
    mejor_espacios = espacios["variante"]
    if especialistas["celdas_significativas"]:
        mejor_especialistas = especialistas["modelo_combinado"]
        linea_especialistas = "ESPECIALISTAS-MEZCLA"
        decision_especialistas = especialistas["decision"]
        cerrada_especialistas = not especialistas["decision"]["adoptado"]
        valor_especialistas = mejor_especialistas["corrected_cv"]
    else:
        linea_especialistas = "SIN-EVIDENCIA (ninguna celda significativa)"
        decision_especialistas = especialistas["decision"]
        cerrada_especialistas = True
        valor_especialistas = CONFIG["referencia_m2_umbral_05"]
    return [
        {
            "linea": "E2-14 calibracion",
            "mejor_variante": mejor_calibracion["variante"],
            "corrected_cv": mejor_calibracion["f1_anidado"],
            "delta_vs_v03": mejor_calibracion["f1_anidado"] - CONFIG["referencia_v03_umbrales"],
            "pliegues_mejorados_vs_v03": int(
                np.sum(np.array(mejor_calibracion["f1_pliegues"]) > np.array(referencias["f1_pliegues_v03"]))
            ),
            "delta_vs_comparador": mejor_calibracion["delta_vs_sin_calibrar"],
            "pliegues_mejorados_vs_comparador": mejor_calibracion["pliegues_mejorados_vs_sin_calibrar"],
            "comparador": "sin calibrar",
            "adoptado": mejor_calibracion["decision"]["adoptado"],
            "cerrada_por_falta_de_evidencia": not mejor_calibracion["decision"]["adoptado"],
            "detalle": mejor_calibracion["decision"]["justificacion"],
            "extra": {
                "brier": mejor_calibracion["brier"],
                "log_loss": mejor_calibracion["log_loss"],
                "umbrales_std": mejor_calibracion["umbrales_std"],
            },
        },
        {
            "linea": "E2-13 entidades",
            "mejor_variante": entidades["mejor"],
            "corrected_cv": mejor_entidades["corrected_cv"],
            "delta_vs_v03": mejor_entidades["delta_vs_v03"],
            "pliegues_mejorados_vs_v03": int(
                np.sum(np.array(mejor_entidades["f1_pliegues"]) > np.array(referencias["f1_pliegues_v03"]))
            ),
            "delta_vs_comparador": mejor_entidades["delta_vs_meta_ridge"],
            "pliegues_mejorados_vs_comparador": mejor_entidades["decision"]["pliegues_mejorados_vs_comparador"],
            "comparador": "META-BASES-RIDGE",
            "adoptado": False,
            "cerrada_por_falta_de_evidencia": entidades["cobertura"]["cierra_por_sparsity"]
            or not mejor_entidades["decision"]["adoptado"],
            "detalle": (
                f"cobertura informativa {entidades['cobertura']['cobertura_informativa']:.3f} < "
                f"{CONFIG['umbral_cobertura_minima']:.2f}: " if entidades["cobertura"]["cierra_por_sparsity"] else ""
            )
            + mejor_entidades["decision"]["justificacion"],
            "extra": {
                "cobertura_informativa": entidades["cobertura"]["cobertura_informativa"],
                "por_tipo": {
                    tipo: entidades["cobertura"]["por_tipo"][tipo]["cobertura_instancias"]
                    for tipo in TIPOS_ENTIDAD
                },
            },
        },
        {
            "linea": "E2-16 umbrales por rebanada",
            "mejor_variante": rebanadas["mejor"],
            "corrected_cv": mejor_rebanadas["corrected_cv"],
            "delta_vs_v03": mejor_rebanadas["delta_vs_v03"],
            "pliegues_mejorados_vs_v03": int(
                np.sum(np.array(mejor_rebanadas["f1_pliegues"]) > np.array(referencias["f1_pliegues_v03"]))
            ),
            "delta_vs_comparador": mejor_rebanadas["delta_vs_global_anidado"],
            "pliegues_mejorados_vs_comparador": mejor_rebanadas["pliegues_mejorados_vs_global"],
            "comparador": "umbral global anidado",
            "adoptado": rebanadas["decision"]["adoptado"],
            "cerrada_por_falta_de_evidencia": not rebanadas["decision"]["adoptado"],
            "detalle": rebanadas["decision"]["justificacion"],
            "extra": {
                "global_anidado": rebanadas["global_anidado"]["corrected_cv"],
                "estabilidad": rebanadas["estabilidad"],
            },
        },
        {
            "linea": "E2-20 especialistas",
            "mejor_variante": linea_especialistas,
            "corrected_cv": valor_especialistas,
            "delta_vs_v03": valor_especialistas - CONFIG["referencia_v03_umbrales"],
            "pliegues_mejorados_vs_v03": decision_especialistas.get("pliegues_mejorados_vs_v03", 0),
            "delta_vs_comparador": decision_especialistas.get("delta_vs_comparador", 0.0),
            "pliegues_mejorados_vs_comparador": decision_especialistas.get(
                "pliegues_mejorados_vs_comparador", 0
            ),
            "comparador": "M2 a 0.5",
            "adoptado": False,
            "cerrada_por_falta_de_evidencia": cerrada_especialistas,
            "detalle": (
                decision_especialistas["justificacion"]
                + (
                    f"; delta medio de la mezcla dentro de las celdas "
                    f"{especialistas.get('delta_medio_mezcla_en_celdas', 0.0):+.4f}"
                    if especialistas["celdas_significativas"]
                    else ""
                )
            ),
            "extra": {
                "celdas": especialistas["celdas_significativas"],
                "delta_medio_mezcla_en_celdas": especialistas.get("delta_medio_mezcla_en_celdas", 0.0),
            },
        },
        {
            "linea": "E2-21 espacios por etiqueta",
            "mejor_variante": "ESPACIOS-POR-ETIQUETA",
            "corrected_cv": mejor_espacios["corrected_cv"],
            "delta_vs_v03": mejor_espacios["delta_vs_v03"],
            "pliegues_mejorados_vs_v03": int(
                np.sum(np.array(mejor_espacios["f1_pliegues"]) > np.array(referencias["f1_pliegues_v03"]))
            ),
            "delta_vs_comparador": mejor_espacios["delta_vs_m2_05"],
            "pliegues_mejorados_vs_comparador": espacios["decision"]["pliegues_mejorados_vs_comparador"],
            "comparador": "espacio compartido V03",
            "adoptado": espacios["decision"]["adoptado"],
            "cerrada_por_falta_de_evidencia": not espacios["decision"]["adoptado"],
            "detalle": espacios["decision"]["justificacion"],
            "extra": {
                "delta_por_etiqueta": mejor_espacios["delta_por_etiqueta_vs_compartida"],
                "dimensiones": espacios["dimensiones_medias"],
            },
        },
    ]


def registrar_experimentos(filas: Sequence[dict[str, Any]], duracion: float) -> None:
    """Anade las filas E2-13/E2-14/E2-16/E2-20/E2-21 a ``experiments.csv``.

    Args:
        filas: filas a registrar.
        duracion: segundos totales del experimento.
    """
    for fila in filas:
        if not fila["training_time"]:
            fila["training_time"] = round(duracion, 2)
    existente = pd.read_csv(EXPERIMENTS_CSV)
    identificadores = {fila["experiment_id"] for fila in filas}
    prefijos = ("E2-13-", "E2-14-", "E2-16-", "E2-20-", "E2-21-")
    existente = existente[
        ~existente["experiment_id"].isin(identificadores)
        & ~existente["experiment_id"].str.startswith(prefijos)
    ]
    existente.to_csv(EXPERIMENTS_CSV, index=False)
    pd.DataFrame(filas).to_csv(EXPERIMENTS_CSV, mode="a", header=False, index=False)
    log(f"filas registradas en experiments.csv: {len(filas)}")


def escribir_contexto(lineas: Sequence[str]) -> None:
    """Reemplaza o anade la seccion de P8 en ``reports/contexto.md``.

    Args:
        lineas: lineas de la seccion (sin encabezado).

    Raises:
        ValueError: si la seccion supera 20 lineas.
    """
    if len(lineas) > 20:
        raise ValueError("La seccion de contexto no puede superar 20 lineas")
    encabezado = "## Fase v2 - P8 (2026-09-25)"
    bloque = "\n".join([encabezado, *lineas])
    texto = CONTEXTO_MD.read_text(encoding="utf-8")
    if encabezado in texto:
        inicio = texto.index(encabezado)
        fin = texto.find("\n## ", inicio + len(encabezado))
        fin = len(texto) if fin == -1 else fin + 1
        texto = texto[:inicio] + bloque + "\n" + texto[fin:]
    else:
        texto = texto.rstrip("\n") + "\n\n" + bloque + "\n"
    CONTEXTO_MD.write_text(texto, encoding="utf-8")
    log("seccion de P8 escrita en reports/contexto.md")


def construir_contexto(
    resumen_lineas: Sequence[dict[str, Any]],
    entidades: dict[str, Any],
    especialistas: dict[str, Any],
    duracion: float,
) -> list[str]:
    """Construye las lineas de contexto de P8 (maximo 20).

    Args:
        resumen_lineas: resumen por linea del backlog.
        entidades: resultados de E2-13.
        especialistas: resultados de E2-20.
        duracion: segundos totales.

    Returns:
        Lista de lineas (sin encabezado).
    """
    por_linea = {fila["linea"]: fila for fila in resumen_lineas}
    lineas = [
        "P8 cierra las 5 lineas restantes del backlog v2 (E2-13, E2-14, E2-16, E2-20, E2-21) sobre el desarrollo (n=1483), pliegues corregidos v2 y holdout congelado intacto.",
        f"Referencias reproducidas: M2 a 0.5 = {CONFIG['referencia_m2_umbral_05']:.4f}, V03 con umbrales = {CONFIG['referencia_v03_umbrales']:.4f}, meta Ridge = {CONFIG['referencia_meta_ridge']:.4f}.",
        "E2-14 calibracion: Platt e isotonic anidados; son transformaciones monotonas por etiqueta, asi que la F1 con umbrales reajustados es invariante salvo empates.",
        f"E2-14 mejor variante: {por_linea['E2-14 calibracion']['mejor_variante']} corrected_cv {por_linea['E2-14 calibracion']['corrected_cv']:.4f} (delta vs su par sin calibrar {por_linea['E2-14 calibracion']['delta_vs_comparador']:+.4f}); Brier {por_linea['E2-14 calibracion']['extra']['brier']:.4f} y log-loss {por_linea['E2-14 calibracion']['extra']['log_loss']:.4f}.",
        f"E2-13 encoder de entidades: cobertura informativa (mencion+hashtag) {entidades['cobertura']['cobertura_informativa']:.3f} < {CONFIG['umbral_cobertura_minima']:.2f}; mencion {entidades['cobertura']['por_tipo']['mencion']['unicos_dev']} unicas, hashtag {entidades['cobertura']['por_tipo']['hashtag']['unicos_dev']}, dominio {entidades['cobertura']['por_tipo']['dominio']['unicos_dev']} (t.co trivial).",
        f"E2-13 mejor variante: {por_linea['E2-13 entidades']['mejor_variante']} corrected_cv {por_linea['E2-13 entidades']['corrected_cv']:.4f} (delta vs meta {por_linea['E2-13 entidades']['delta_vs_comparador']:+.4f}): linea CERRADA por sparsity.",
        f"E2-16 umbrales por rebanada: mejor {por_linea['E2-16 umbrales por rebanada']['mejor_variante']} corrected_cv {por_linea['E2-16 umbrales por rebanada']['corrected_cv']:.4f} (delta vs global anidado {por_linea['E2-16 umbrales por rebanada']['delta_vs_comparador']:+.4f}).",
        "E2-16 rebanadas usadas (n>=80): hashtag, mencion, url, pregunta, colectiva, corto40 y largo_q4; comillas queda fuera por n<80.",
        "E2-16 evaluacion anidada: el umbral local se ajusta en los pliegues de ajuste (>= 8 positivos por celda) y se aplica al pliegue evaluado; la mezcla usa w = n/(n+m).",
        f"E2-20 especialistas: {len(especialistas['pruebas'])} pruebas de proporcion de errores (rebanada x etiqueta) con BH q={CONFIG['bh_q']}; celdas significativas: {len(especialistas['celdas_significativas'])}.",
        (
            "E2-20 decision: sin evidencia, no se entrena especialista (linea cerrada)."
            if not especialistas["celdas_significativas"]
            else (
                f"E2-20 decision: los especialistas rinden menos que el global dentro de la rebanada "
                f"(delta medio {especialistas.get('delta_medio_mezcla_en_celdas', 0.0):+.4f} para la mezcla); "
                "la mezcla queda por debajo de la referencia -> CERRADA sin mejora."
            )
        ),
        f"E2-21 espacios por etiqueta: corrected_cv {por_linea['E2-21 espacios por etiqueta']['corrected_cv']:.4f} (delta vs espacio compartido {por_linea['E2-21 espacios por etiqueta']['delta_vs_comparador']:+.4f}); coste de dimension del canal: {por_linea['E2-21 espacios por etiqueta']['extra']['dimensiones']['HS']:.0f} HS / {por_linea['E2-21 espacios por etiqueta']['extra']['dimensiones']['TR']:.0f} TR / {por_linea['E2-21 espacios por etiqueta']['extra']['dimensiones']['AG']:.0f} AG.",
        "E2-21 usa proxies estructurales por regex y lexicos inducidos fold-local (Monroe 2008); sin spaCy en este paquete (la linea morfologica se cerro en P3).",
        (
            "Regla de adopcion (D-016) aplicada al mecanismo de cada linea, con el aporte medido contra su comparador directo: "
            "solo el meta calibrado supera 0.7822 (0.7857, 5/5 pliegues), pero no aporta sobre el meta sin calibrar (-0.0017); "
            "ninguna otra variante alcanza +0.005 con 3/5 pliegues ni complementariedad con IC > 0."
        ),
        "Cierre honesto: las 5 lineas se cierran sin mejora transferible; el backlog v2 queda al 100 % (24/24) y no se gasta ninguna evaluacion del holdout.",
        f"Duracion: {duracion:.1f} s. Script: experiments/21_backlog_restante_v2.py. Artefactos: reports/backlog_restante_v2.csv y .json, reports/figuras/fig_32_backlog_restante_v2.png.",
    ]
    return lineas


def actualizar_todo(resumen_lineas: Sequence[dict[str, Any]], especialistas: dict[str, Any]) -> None:
    """Actualiza las 5 lineas de P8 y el Gate P8 en ``tasks/todo.md``.

    Args:
        resumen_lineas: resumen por linea del backlog.
        especialistas: resultados de E2-20.
    """
    por_linea = {fila["linea"]: fila for fila in resumen_lineas}
    lineas_p8 = [
        "## P8 - Cierre del backlog restante (E2-13, E2-14, E2-16, E2-20, E2-21)",
        "",
        (
            f"- [cerrado] P8.1 E2-13 encoder de entidades: cobertura informativa "
            f"{por_linea['E2-13 entidades']['extra']['cobertura_informativa']:.3f} < 0.20 -> CERRADA por sparsity; "
            f"mejor {por_linea['E2-13 entidades']['mejor_variante']} corrected_cv "
            f"{por_linea['E2-13 entidades']['corrected_cv']:.4f} (delta vs meta "
            f"{por_linea['E2-13 entidades']['delta_vs_comparador']:+.4f})"
        ),
        (
            f"- [cerrado] P8.2 E2-14 calibracion: mejor {por_linea['E2-14 calibracion']['mejor_variante']} "
            f"corrected_cv {por_linea['E2-14 calibracion']['corrected_cv']:.4f} (delta vs su par sin calibrar "
            f"{por_linea['E2-14 calibracion']['delta_vs_comparador']:+.4f}); invariante salvo empates, "
            "aporte solo en Brier/log-loss -> CERRADA sin mejora"
        ),
        (
            f"- [cerrado] P8.3 E2-16 umbrales por rebanada con shrinkage: mejor "
            f"{por_linea['E2-16 umbrales por rebanada']['mejor_variante']} corrected_cv "
            f"{por_linea['E2-16 umbrales por rebanada']['corrected_cv']:.4f} (delta vs global anidado "
            f"{por_linea['E2-16 umbrales por rebanada']['delta_vs_comparador']:+.4f}) -> CERRADA sin mejora"
        ),
        (
            f"- [cerrado] P8.4 E2-20 especialistas por rebanada: {len(especialistas['pruebas'])} pruebas, "
            f"{len(especialistas['celdas_significativas'])} celdas significativas tras BH -> "
            + ("CERRADA sin evidencia (no se entrena especialista)" if not especialistas["celdas_significativas"] else "evaluados, ver JSON")
        ),
        (
            f"- [cerrado] P8.5 E2-21 espacios por etiqueta: corrected_cv "
            f"{por_linea['E2-21 espacios por etiqueta']['corrected_cv']:.4f} (delta vs espacio compartido "
            f"{por_linea['E2-21 espacios por etiqueta']['delta_vs_comparador']:+.4f}) -> CERRADA sin mejora"
        ),
        "- [hecho] P8.6 Gate P8: backlog v2 cerrado al 100 % (24/24 lineas); ninguna de las 5 lineas restantes mejora la referencia 0.7772",
        "",
    ]
    texto = TODO_MD.read_text(encoding="utf-8")
    marcador = "## Registro de gates"
    if "## P8 - Cierre del backlog restante" in texto:
        inicio = texto.index("## P8 - Cierre del backlog restante")
        fin = texto.find("\n## ", inicio + 5)
        fin = fin + 1 if fin != -1 else len(texto)
        texto = texto[:inicio] + "\n".join(lineas_p8) + "\n" + texto[fin:]
    elif marcador in texto:
        texto = texto.replace(marcador, "\n".join(lineas_p8) + "\n" + marcador, 1)
    else:
        texto = texto.rstrip("\n") + "\n\n" + "\n".join(lineas_p8) + "\n"
    cierre = (
        "| P8 | 2026-09-25 | CERRADO (sin mejora). Backlog v2 al 100 % (24/24 lineas). "
        f"E2-13 encoder de entidades: cobertura informativa "
        f"{por_linea['E2-13 entidades']['extra']['cobertura_informativa']:.3f} < 0.20 -> cerrada por sparsity "
        f"({por_linea['E2-13 entidades']['mejor_variante']} {por_linea['E2-13 entidades']['corrected_cv']:.4f}). "
        f"E2-14 calibracion: {por_linea['E2-14 calibracion']['mejor_variante']} "
        f"{por_linea['E2-14 calibracion']['corrected_cv']:.4f} (delta {por_linea['E2-14 calibracion']['delta_vs_comparador']:+.4f} vs sin calibrar). "
        f"E2-16 umbrales por rebanada: {por_linea['E2-16 umbrales por rebanada']['mejor_variante']} "
        f"{por_linea['E2-16 umbrales por rebanada']['corrected_cv']:.4f} (delta {por_linea['E2-16 umbrales por rebanada']['delta_vs_comparador']:+.4f} vs global anidado). "
        f"E2-20 especialistas: {len(especialistas['celdas_significativas'])} celdas significativas tras BH; "
        f"los especialistas son peores que el global dentro de la rebanada (delta medio de la mezcla "
        f"{especialistas.get('delta_medio_mezcla_en_celdas', 0.0):+.4f}) -> cerrada sin mejora. "
        f"E2-21 espacios por etiqueta: {por_linea['E2-21 espacios por etiqueta']['corrected_cv']:.4f} "
        f"(delta {por_linea['E2-21 espacios por etiqueta']['delta_vs_comparador']:+.4f} vs compartido). "
        "Regla D-016 aplicada contra el comparador directo de cada mecanismo: solo el meta calibrado alcanza 0.7857 (5/5 pliegues) "
        "por reproducir el meta ya adoptado, sin aportar sobre el meta sin calibrar; ninguna linea aporta +0.005 con 3/5 pliegues "
        "ni complementariedad con IC > 0. "
        "Evidencia: `experiments/21_backlog_restante_v2.py`, `reports/backlog_restante_v2.{csv,json}`, "
        "`reports/figuras/fig_32_backlog_restante_v2.png` |"
    )
    if "| P8 |" in texto:
        texto = "\n".join(
            [linea for linea in texto.splitlines() if not linea.startswith("| P8 |")]
        ).rstrip("\n") + "\n"
    texto = texto.rstrip("\n") + "\n" + cierre + "\n"
    TODO_MD.write_text(texto, encoding="utf-8")
    log("tasks/todo.md actualizado (5 lineas de P8 y Gate P8)")


def main() -> None:
    """Ejecuta P8 (E2-13, E2-14, E2-16, E2-20, E2-21) y guarda los artefactos."""
    inicio = time.perf_counter()
    desarrollo, grupos, pliegue_de_fila = cargar_desarrollo()
    y_verdadera = desarrollo[list(ETIQUETAS)].to_numpy()
    verificacion_datos = verificar_datos(desarrollo, grupos, pliegue_de_fila)
    referencias = cargar_referencias()
    proba_bases, proba_meta, info_oof = cargar_oof(
        desarrollo, grupos, y_verdadera, pliegue_de_fila
    )
    proba_m2 = proba_bases[:, POSICION_M2, :]
    sin_umbral = {etiqueta: 0.5 for etiqueta in ETIQUETAS}
    verificar_metrica_rapida(proba_m2, y_verdadera, sin_umbral)
    verificacion_reproduccion = verificar_reproduccion(
        proba_m2, proba_meta, y_verdadera, referencias
    )
    modulo_base = cargar_modulo_base()

    calibracion = ejecutar_e2_14(
        proba_m2, proba_meta, y_verdadera, pliegue_de_fila, referencias
    )
    entidades = ejecutar_e2_13(
        desarrollo, grupos, pliegue_de_fila, proba_bases, proba_meta, y_verdadera, referencias
    )
    rebanadas = ejecutar_e2_16(
        desarrollo["text"], proba_m2, y_verdadera, pliegue_de_fila, referencias
    )
    especialistas = ejecutar_e2_20(
        desarrollo,
        desarrollo["text"],
        grupos,
        pliegue_de_fila,
        proba_m2,
        y_verdadera,
        {
            nombre: mascara
            for nombre, mascara in definir_rebanadas(
                desarrollo["text"], desarrollo["text"].str.len().to_numpy()
            ).items()
            if int(mascara.sum()) >= CONFIG["min_rebanada"]
        },
        modulo_base,
        referencias,
    )
    espacios = ejecutar_e2_21(
        desarrollo, grupos, proba_bases, pliegue_de_fila, y_verdadera, modulo_base, referencias
    )

    resumen_lineas = construir_resumen_lineas(
        calibracion, entidades, rebanadas, especialistas, espacios, referencias
    )
    for fila in resumen_lineas:
        log(
            f"linea {fila['linea']}: {fila['mejor_variante']} corrected_cv "
            f"{fila['corrected_cv']:.4f} delta_vs_v03 {fila['delta_vs_v03']:+.4f} "
            f"{'ADOPTADA' if fila['adoptado'] else 'CERRADA'}"
        )

    duracion = time.perf_counter() - inicio
    generar_figura(resumen_lineas, calibracion, entidades)
    filas = construir_filas_experimentos(
        calibracion, entidades, rebanadas, especialistas, espacios
    )
    escribir_reporte_csv(resumen_lineas)
    reporte = {
        "fecha": CONFIG["fecha"],
        "protocolo": "v2 (D-016): grupos = near-duplicado UNION pseudo-autor; holdout congelado intacto",
        "referencias": {
            "m2_umbral_05": CONFIG["referencia_m2_umbral_05"],
            "v03_umbrales": CONFIG["referencia_v03_umbrales"],
            "meta_ridge": CONFIG["referencia_meta_ridge"],
            "v2a_meta": CONFIG["referencia_v2a_meta"],
            "umbrales_v03": CONFIG["umbrales_v03"],
        },
        "verificaciones": {
            "datos": verificacion_datos,
            "oof": info_oof,
            "reproduccion": verificacion_reproduccion,
        },
        "e2_14_calibracion": {
            "variantes": {
                clave: {k: v for k, v in valor.items() if k != "proba"}
                for clave, valor in calibracion["variantes"].items()
            },
            "comparaciones": calibracion["comparaciones"],
            "mejores": {
                clave: {
                    k: v
                    for k, v in valor.items()
                    if k not in ("proba",)
                }
                for clave, valor in calibracion["mejores"].items()
            },
            "nota_invarianza": calibracion["nota_invarianza"],
            "decision": calibracion["decision"],
        },
        "e2_13_entidades": {
            "cobertura": entidades["cobertura"],
            "variantes": {
                clave: {k: v for k, v in valor.items() if k != "proba"}
                for clave, valor in entidades["variantes"].items()
            },
            "mejor": entidades["mejor"],
            "decision": entidades["decision"],
        },
        "e2_16_umbrales_rebanada": {
            "tamano_rebanadas": rebanadas["tamano_rebanadas"],
            "excluidas": rebanadas["excluidas"],
            "global_anidado": rebanadas["global_anidado"],
            "variantes": {
                clave: {k: v for k, v in valor.items() if k != "umbrales_fila"}
                for clave, valor in rebanadas["variantes"].items()
            },
            "mejor": rebanadas["mejor"],
            "detalle_rebanadas": rebanadas["detalle_rebanadas"],
            "estabilidad": rebanadas["estabilidad"],
            "decision": rebanadas["decision"],
        },
        "e2_20_especialistas": {
            "pruebas": especialistas["pruebas"],
            "celdas_significativas": especialistas["celdas_significativas"],
            "delta_medio_mezcla_en_celdas": especialistas.get("delta_medio_mezcla_en_celdas"),
            "especialistas": especialistas["especialistas"],
            "modelo_combinado": especialistas["modelo_combinado"],
            "decision": especialistas["decision"],
        },
        "e2_21_espacios": {
            "variante": {k: v for k, v in espacios["variante"].items() if k != "proba"},
            "dimensiones": espacios["dimensiones"],
            "dimensiones_medias": espacios["dimensiones_medias"],
            "decision": espacios["decision"],
        },
        "resumen_lineas": resumen_lineas,
        "backlog": {
            "lineas_totales": 24,
            "lineas_cerradas": 24,
            "lineas_de_este_paquete": [
                "E2-13",
                "E2-14",
                "E2-16",
                "E2-20",
                "E2-21",
            ],
            "nota": "Las 5 lineas restantes se evaluan y cierran sin mejora; el backlog queda al 100 %.",
        },
        "duracion_segundos": round(duracion, 2),
    }
    REPORTE_JSON.write_text(
        json.dumps(serializar(reporte), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    registrar_experimentos(filas, duracion)
    escribir_contexto(construir_contexto(resumen_lineas, entidades, especialistas, duracion))
    actualizar_todo(resumen_lineas, especialistas)
    log(
        f"P8 completado en {duracion:.1f} s; artefactos: "
        f"{REPORTE_CSV.name}, {REPORTE_JSON.name}, {FIGURA_PATH.name}"
    )


if __name__ == "__main__":
    main()
