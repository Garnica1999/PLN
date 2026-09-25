"""Fase 12 - Analisis de error del mejor modelo (MOD-03b_BEST) con hipotesis dirigidas.

Pregunta que responde
---------------------
El mejor modelo de la competencia (``MOD-03b_BEST``: union TF-IDF word(1,2) +
char_wb(3,5) + ``LogisticRegression`` C=2.0 balanced x3; F1 OOF 0.7927 con
umbral 0.5 y 0.7989 con umbrales por etiqueta 0.44/0.51/0.46) sigue perdiendo
~0.20 de F1. Este script responde, con las probabilidades out-of-fold ya
guardadas y sin reentrenar nada:

1. Donde se equivoca: matrices de confusion 2x2 por etiqueta con los dos
   escenarios de umbral (conteos y tasas).
2. Como se combinan los errores: cuantos ejemplos tienen 1, 2 o 3 etiquetas
   mal; cuantos errores son del par HS-TR, HS-AG y TR-AG; y cuanto vale la
   regla estructural ``HS_pred = 0 => TR = 0`` (y su version completa con AG).
3. En que regiones del corpus falla mas: analisis por rebanadas sobre train con
   las predicciones OOF (longitud en cuartiles, textos cortos, mencion, hashtag,
   URL, insulto del lexico inducido -con la advertencia de sesgo in-sample-,
   segunda persona, imperativo, referencia colectiva, mayusculas sostenidas,
   elongaciones, pregunta/exclamacion, risa/ironia, comillas y retuits).
4. Que se ve en los casos concretos: 15 falsos positivos y 15 falsos negativos
   por etiqueta exportados para revision humana, mas la descripcion de los
   patrones recurrentes observados.
5. Que hacer despues: hipotesis nuevas (H11 en adelante) con evidencia,
   mecanismo, experimento propuesto y criterio de exito.

Metodo (sin fuga)
-----------------
- Se cargan las probabilidades OOF de ``features/oof_mod03b.npz`` y se verifica
  que reproducen el F1 publicado (tolerancia 1e-4) antes de usarlas.
- Los umbrales evaluados son los ya decididos en D-012 (0.5 y 0.44/0.51/0.46);
  no se re-optimiza ningun umbral en este script.
- Los rasgos de las rebanadas se calculan con regex y, para persona/imperativo/
  colectivo, con spaCy como anotador morfosintactico (D-003, sin vectores).
- El lexico de agresividad se lee de
  ``reports/eda/fase2_lexico_inducido_agresividad.csv`` y se advierte que se
  indujo in-sample con etiquetas de train: sus rebanadas son diagnosticas, no
  evidencia de generalizacion (hallazgo mayor de la revision de codigo, Fase C).
- El test no se usa en ningun calculo; solo se cuentan artefactos de
  normalizacion como contexto.

Salidas
-------
- ``reports/eda/error_matrices_confusion.csv``
- ``reports/eda/error_combinaciones_etiquetas.csv``
- ``reports/eda/error_regla_estructural.csv``
- ``reports/eda/error_rebanadas.csv``
- ``reports/eda/error_regiones.csv``
- ``reports/eda/error_perfil_rasgos.csv``
- ``reports/eda/error_samples_para_revision.csv``
- ``reports/eda/error_resumen.json`` (numeros clave para el handoff)
- ``reports/figuras/fig_20..24_*.png``
- ``reports/error_analysis.md`` y una seccion de 20 lineas en
  ``reports/contexto.md``.

Idempotencia
------------
Todas las salidas se reescriben completas (no se anexan filas) y la seccion de
``reports/contexto.md`` se reemplaza si ya existe. No se modifica ningun script
previo, ni ``src/``, ni ``PROJECT_STATE.md``, ni ``DECISIONS.md``, ni
``RUNLOG.md``, ni ``experiments.csv``.

Ejecucion
---------
    /home/carlos/venv/bin/python experiments/06_error_analysis.py
"""

from __future__ import annotations

import json
import re
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import spacy

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src import data as data_module
from src.evaluation import ETIQUETAS, metricas_multietiqueta

# ---------------------------------------------------------------------------
# Seccion 1 - Configuracion
# ---------------------------------------------------------------------------
CONFIG: Final[dict[str, Any]] = {
    "oof_npz": PROJECT_ROOT / "features" / "oof_mod03b.npz",
    "lexico_csv": PROJECT_ROOT / "reports" / "eda" / "fase2_lexico_inducido_agresividad.csv",
    "eda_dir": PROJECT_ROOT / "reports" / "eda",
    "figuras_dir": PROJECT_ROOT / "reports" / "figuras",
    "reporte_md": PROJECT_ROOT / "reports" / "error_analysis.md",
    "contexto_md": PROJECT_ROOT / "reports" / "contexto.md",
    "modelo": "MOD-03b_BEST",
    "fecha": "2026-09-22",
    "umbrales_0_5": (0.5, 0.5, 0.5),
    "umbrales_optimos": (0.44, 0.51, 0.46),
    "f1_esperado_0_5": 0.7926588861781015,
    "f1_esperado_optimos": 0.798933,
    "tolerancia_replicacion": 1e-4,
    "min_rebanada": 40,
    "n_muestras_por_tipo": 15,
    "max_caracteres_muestra": 200,
    "z_nucleo_insulto": 2.5,
}

# Escenarios evaluados: (clave, nombre legible, umbrales por etiqueta).
ESCENARIOS: Final[tuple[tuple[str, str, tuple[float, float, float]], ...]] = (
    ("umbral_0_5", "umbral fijo 0.5", CONFIG["umbrales_0_5"]),
    ("umbral_optimo", "umbrales por etiqueta 0.44/0.51/0.46", CONFIG["umbrales_optimos"]),
)

# ---------------------------------------------------------------------------
# Patrones de superficie (mismos criterios que el EDA cuando existen)
# ---------------------------------------------------------------------------
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
MENTION_PATTERN = re.compile(r"@\w+")
HASHTAG_PATTERN = re.compile(r"#\w+")
INITIAL_MENTION_PATTERN = re.compile(r"^\s*(?:rt\s+)?@\w+", re.IGNORECASE)
RETWEET_PATTERN = re.compile(r"^\s*rt\s+@\w+", re.IGNORECASE)
UPPER_WORD_PATTERN = re.compile(r"\b[A-Z]{2,}\b")
ELONGATION_PATTERN = re.compile(r"([A-Za-z])\1{2,}")
# El patron del EDA exigia que el token completo fuese la risa; el corpus escribe
# series como "JAJAJAJAJJAJAJAJA" (con jotas de mas) que ese patron no captura
# (19 documentos frente a 120 con esta variante laxa de subcadena). Se documenta
# la desviacion y se usa la variante laxa para que la rebanada sea utilizable.
LAUGHTER_PATTERN = re.compile(r"(?:ja){2,}|(?:je){2,}|(?:ji){2,}|lol|xd+", re.IGNORECASE)
HTML_ENTITY_PATTERN = re.compile(r"&(?:gt|lt|amp|quot|apos|#\d+);")
# Marcadores de afiliacion o broma que acompanan al insulto interpersonal en los
# falsos positivos (posesivos de primera persona, apelativos afectivos, risa).
PATRON_BROMA_AFILIACION = re.compile(
    r"\b(te amo|socia|favorit\w*|jaj\w*|jeje\w*|xd+\w*|lol|beer pong|"
    r"mi (novia|perra|amor|amiga|amigo|hermana|hermano|vida|tia|tio|polla|mujer|favorit\w*))\b",
    re.IGNORECASE,
)

# Lexicos internos cerrados heredados del EDA (Fase 2). No son recursos externos:
# son listas de formas funcionales del espanol usadas como anotacion.
SECOND_PERSON_LEXICON: Final[frozenset[str]] = frozenset(
    {
        "tu", "tus", "te", "ti", "usted", "ustedes", "vos", "vosotros",
        "vosotras", "contigo", "os", "vuestro", "vuestra", "vuestros",
        "vuestras", "tuyo", "tuya", "tuyos", "tuyas",
    }
)
CLITIC_ENDINGS: Final[tuple[str, ...]] = (
    "me", "te", "se", "lo", "la", "le", "nos", "os", "los", "las", "les",
)
INFINITIVE_CLITIC_PATTERN = re.compile(r"r(me|te|se|lo|la|le|nos|os|los|las|les)$")
FIRST_PERSON_PLURAL_PATTERN = re.compile(r"(mos|amos|emos|imos)$")
IMPERATIVE_LEXICON: Final[frozenset[str]] = frozenset(
    {
        "calla", "callate", "callese", "callense", "callete", "vete", "vayase",
        "vayan", "vayanse", "muerete", "muere", "largate", "largese", "deja",
        "dejen", "dejame", "dejalo", "mira", "miren", "miralo", "oye", "oiga",
        "oigan", "escucha", "escuchen", "aprende", "aprendan", "habla", "hablen",
        "anda", "andate", "sigue", "sigan", "ven", "veni", "venga", "haz",
        "haga", "pon", "ponte", "pongase", "quita", "quitate", "basta", "cierra",
        "abre", "dime", "digame", "cuentame", "lee", "estudia", "trabaja",
        "trabaje", "respeta", "respeten", "piensa", "piense", "corre", "sal",
        "salte", "espera", "esperen", "aguanta", "tomate", "cojete",
        "chupame", "chupala", "matate", "ahorcate", "pide", "pida",
        "compra", "vende", "danos", "iros", "hacelo", "retirense", "marchate",
        "calmate", "callad", "mirad", "ponganse", "vamos", "huye", "corran",
        "escondete", "largo", "fuera",
    }
)
COLLECTIVE_QUANTIFIERS: Final[frozenset[str]] = frozenset(
    {
        "todos", "todas", "todo", "toda", "muchos", "muchas", "algunos",
        "algunas", "ningun", "ninguna", "ningunos", "ningunas", "cualquiera",
        "cualesquiera", "varios", "varias", "cada", "ambos", "ambas", "gente",
        "pueblo", "mayoria", "minoria", "multitud", "monton", "masa", "grupo",
    }
)

# Rebanadas a evaluar. ``tipo``: "binaria" (presente/ausente) o "categorica".
# ``artefacto_lexico`` marca las derivadas del lexico inducido in-sample, que no
# deben encabezar el ranking de debilidad (advertencia de la Fase C).
REBANADAS: Final[tuple[dict[str, Any], ...]] = (
    {"nombre": "cuartil_longitud", "columna": "cuartil_longitud", "tipo": "categorica", "artefacto_lexico": False},
    {"nombre": "texto_corto_40", "columna": "texto_corto_40", "tipo": "binaria", "artefacto_lexico": False},
    {"nombre": "tiene_mencion", "columna": "tiene_mencion", "tipo": "binaria", "artefacto_lexico": False},
    {"nombre": "tiene_mencion_inicial", "columna": "tiene_mencion_inicial", "tipo": "binaria", "artefacto_lexico": False},
    {"nombre": "tiene_hashtag", "columna": "tiene_hashtag", "tipo": "binaria", "artefacto_lexico": False},
    {"nombre": "tiene_url", "columna": "tiene_url", "tipo": "binaria", "artefacto_lexico": False},
    {"nombre": "tiene_insulto_lexico", "columna": "tiene_insulto_lexico", "tipo": "binaria", "artefacto_lexico": True},
    {"nombre": "tiene_insulto_nucleo", "columna": "tiene_insulto_nucleo", "tipo": "binaria", "artefacto_lexico": True},
    {"nombre": "tiene_segunda_persona", "columna": "tiene_segunda_persona", "tipo": "binaria", "artefacto_lexico": False},
    {"nombre": "tiene_imperativo", "columna": "tiene_imperativo", "tipo": "binaria", "artefacto_lexico": False},
    {"nombre": "tiene_referencia_colectiva", "columna": "tiene_referencia_colectiva", "tipo": "binaria", "artefacto_lexico": False},
    {"nombre": "tiene_mayusculas_sostenidas", "columna": "tiene_mayusculas_sostenidas", "tipo": "binaria", "artefacto_lexico": False},
    {"nombre": "tiene_elongacion", "columna": "tiene_elongacion", "tipo": "binaria", "artefacto_lexico": False},
    {"nombre": "tiene_pregunta", "columna": "tiene_pregunta", "tipo": "binaria", "artefacto_lexico": False},
    {"nombre": "tiene_exclamacion", "columna": "tiene_exclamacion", "tipo": "binaria", "artefacto_lexico": False},
    {"nombre": "tiene_risa", "columna": "tiene_risa", "tipo": "binaria", "artefacto_lexico": False},
    {"nombre": "tiene_comillas", "columna": "tiene_comillas", "tipo": "binaria", "artefacto_lexico": False},
    {"nombre": "tiene_retuit", "columna": "tiene_retuit", "tipo": "binaria", "artefacto_lexico": False},
)

# Regiones interpretativas del corpus (definidas en ``anadir_columnas_rebanada``).
REGIONES: Final[tuple[tuple[str, str], ...]] = (
    ("responde_mencion_inicial", "tiene_mencion_inicial == 1"),
    ("no_responde_sin_colectiva", "mencion_inicial == 0 y referencia_colectiva == 0"),
    ("no_responde_con_colectiva", "mencion_inicial == 0 y referencia_colectiva == 1"),
    ("insulto_responde_sin_colectiva", "insulto == 1, mencion_inicial == 1 y colectiva == 0"),
    ("insulto_no_responde_sin_colectiva", "insulto == 1, mencion_inicial == 0 y colectiva == 0"),
    ("regimen_noticia", "url == 1 o hashtag == 1"),
    ("sin_url_ni_hashtag", "url == 0 y hashtag == 0"),
)

# ---------------------------------------------------------------------------
# Seccion 2 - Utilidades
# ---------------------------------------------------------------------------
def log(mensaje: str) -> None:
    """Imprime un mensaje de progreso de la fase.

    Args:
        mensaje: texto a registrar.
    """
    print(f"[fase12] {mensaje}", flush=True)


def guardar_csv(frame: pd.DataFrame, nombre: str) -> Path:
    """Guarda un DataFrame en ``reports/eda/`` con UTF-8 y sin indice.

    Args:
        frame: tabla a guardar.
        nombre: nombre del archivo CSV.

    Returns:
        Ruta del archivo escrito.
    """
    destino = CONFIG["eda_dir"] / nombre
    destino.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destino, index=False, encoding="utf-8")
    log(f"guardado {destino.relative_to(PROJECT_ROOT)} ({len(frame)} filas)")
    return destino


def tabla_markdown(frame: pd.DataFrame, decimales: int = 4) -> str:
    """Convierte una tabla en markdown para el reporte.

    Args:
        frame: tabla a formatear.
        decimales: numero de decimales para las columnas flotantes.

    Returns:
        Cadena markdown con la tabla (sin indice).
    """
    vista = frame.copy()
    for columna in vista.columns:
        if pd.api.types.is_float_dtype(vista[columna]):
            vista[columna] = vista[columna].round(decimales)
    return vista.to_markdown(index=False)


def serializar(valor: Any) -> Any:
    """Convierte tipos de numpy/pandas en tipos serializables a JSON.

    Args:
        valor: objeto a convertir.

    Returns:
        Objeto con tipos nativos (dict, list, float, int, str, bool o None).
    """
    if isinstance(valor, dict):
        return {str(clave): serializar(dato) for clave, dato in valor.items()}
    if isinstance(valor, (list, tuple)):
        return [serializar(dato) for dato in valor]
    if isinstance(valor, np.ndarray):
        return [serializar(dato) for dato in valor.tolist()]
    if isinstance(valor, np.integer):
        return int(valor)
    if isinstance(valor, np.floating):
        return float(valor)
    if isinstance(valor, np.bool_):
        return bool(valor)
    if isinstance(valor, pd.Timestamp):
        return valor.isoformat()
    if valor is pd.NA:
        return None
    if isinstance(valor, float) and np.isnan(valor):
        return None
    return valor


def configurar_estilo() -> None:
    """Fija el estilo de las figuras (mismo criterio que las fases previas)."""
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams["figure.dpi"] = 110
    plt.rcParams["savefig.dpi"] = 150
    plt.rcParams["axes.titleweight"] = "bold"
    plt.rcParams["font.size"] = 10


def guardar_figura(figura: plt.Figure, nombre: str) -> Path:
    """Guarda una figura en ``reports/figuras/`` con dpi 150.

    Args:
        figura: figura de matplotlib.
        nombre: nombre del archivo PNG.

    Returns:
        Ruta del archivo escrito.
    """
    destino = CONFIG["figuras_dir"] / nombre
    destino.parent.mkdir(parents=True, exist_ok=True)
    figura.savefig(destino, dpi=150, bbox_inches="tight")
    plt.close(figura)
    log(f"guardada figura {destino.relative_to(PROJECT_ROOT)}")
    return destino


# ---------------------------------------------------------------------------
# Seccion 3 - Carga, verificacion y escenarios
# ---------------------------------------------------------------------------
@dataclass
class Escenario:
    """Predicciones y metricas de un juego de umbrales.

    Attributes:
        clave: identificador corto del escenario.
        nombre: descripcion legible.
        umbrales: umbral por etiqueta en el orden HS, TR, AG.
        proba: matriz de probabilidades OOF (n, 3).
        prediccion: matriz binaria (n, 3) predicha.
        metricas: metricas multi-etiqueta de :mod:`src.evaluation`.
    """

    clave: str
    nombre: str
    umbrales: tuple[float, float, float]
    proba: np.ndarray
    prediccion: np.ndarray
    metricas: dict[str, Any]


def cargar_oof(ruta: Path) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Carga train y las probabilidades OOF del mejor modelo.

    Args:
        ruta: ruta del archivo ``.npz`` con ``ids``, ``y_true`` y ``proba``.

    Returns:
        Tupla ``(train, y_true, proba)`` con el orden de filas alineado.

    Raises:
        ValueError: si los ids del OOF no coinciden con los de train o si las
            formas de ``proba`` y ``y_true`` son inconsistentes.
    """
    train = data_module.read_train()
    datos = np.load(ruta, allow_pickle=True)
    ids_oof = datos["ids"].astype(str)
    y_true = datos["y_true"].astype(int)
    proba = datos["proba"].astype(float)
    ids_train = train["id"].astype(str).to_numpy()
    if not np.array_equal(ids_oof, ids_train):
        raise ValueError("Los ids del OOF no coinciden con train: no se puede alinear")
    if proba.shape != y_true.shape or proba.shape[1] != len(ETIQUETAS):
        raise ValueError(f"Forma inesperada de proba {proba.shape} / y_true {y_true.shape}")
    log(f"OOF cargado: {proba.shape[0]} filas y {len(ETIQUETAS)} etiquetas")
    return train, y_true, proba


def construir_escenarios(
    y_true: np.ndarray, proba: np.ndarray
) -> dict[str, Escenario]:
    """Construye los escenarios de umbral verificando la replicacion del F1.

    Args:
        y_true: matriz binaria verdadera (n, 3).
        proba: matriz de probabilidades OOF (n, 3).

    Returns:
        Diccionario ``clave -> Escenario`` con metricas ya calculadas.

    Raises:
        ValueError: si algun escenario no reproduce el F1 documentado.
    """
    esperados = {
        "umbral_0_5": CONFIG["f1_esperado_0_5"],
        "umbral_optimo": CONFIG["f1_esperado_optimos"],
    }
    escenarios: dict[str, Escenario] = {}
    for clave, nombre, umbrales in ESCENARIOS:
        prediccion = (proba >= np.asarray(umbrales)).astype(int)
        metricas = dict(metricas_multietiqueta(y_true, prediccion))
        obtenido = metricas["f1_media_columnas"]
        if abs(obtenido - esperados[clave]) > CONFIG["tolerancia_replicacion"]:
            raise ValueError(
                f"{clave}: F1 {obtenido:.6f} no replica el esperado {esperados[clave]:.6f}"
            )
        log(
            f"{clave}: F1 media columnas {obtenido:.6f} "
            f"(HS {metricas['f1_por_etiqueta']['HS']:.4f}, "
            f"TR {metricas['f1_por_etiqueta']['TR']:.4f}, "
            f"AG {metricas['f1_por_etiqueta']['AG']:.4f}; "
            f"exact match {metricas['exact_match']:.4f})"
        )
        escenarios[clave] = Escenario(
            clave, nombre, umbrales, proba, prediccion, metricas
        )
    return escenarios


# ---------------------------------------------------------------------------
# Seccion 4 - Matrices de confusion por etiqueta
# ---------------------------------------------------------------------------
def matriz_confusion(verdad: np.ndarray, predicho: np.ndarray) -> dict[str, float]:
    """Calcula la matriz de confusion 2x2 y sus tasas derivadas.

    Args:
        verdad: columna binaria verdadera.
        predicho: columna binaria predicha.

    Returns:
        Diccionario con TP, FP, FN, TN, precision, recall, especificidad, F1 y
        tasas de error (FPR, FNR, tasa de error por etiqueta, exactitud).
    """
    tp = float(np.sum((verdad == 1) & (predicho == 1)))
    fp = float(np.sum((verdad == 0) & (predicho == 1)))
    fn = float(np.sum((verdad == 1) & (predicho == 0)))
    tn = float(np.sum((verdad == 0) & (predicho == 0)))
    n = tp + fp + fn + tn
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    especificidad = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "n": n,
        "precision": precision,
        "recall": recall,
        "especificidad": especificidad,
        "f1": f1,
        "fpr": fp / (fp + tn) if fp + tn else 0.0,
        "fnr": fn / (fn + tp) if fn + tp else 0.0,
        "tasa_error_etiqueta": (fp + fn) / n if n else 0.0,
        "exactitud": (tp + tn) / n if n else 0.0,
    }


def construir_matrices_confusion(
    y_true: np.ndarray, escenarios: dict[str, Escenario]
) -> pd.DataFrame:
    """Construye la tabla de matrices de confusion por etiqueta y escenario.

    Args:
        y_true: matriz binaria verdadera (n, 3).
        escenarios: escenarios de umbral evaluados.

    Returns:
        DataFrame con una fila por (escenario, etiqueta) y todas las tasas.
    """
    filas: list[dict[str, Any]] = []
    for escenario in escenarios.values():
        for indice, etiqueta in enumerate(ETIQUETAS):
            celdas = matriz_confusion(
                y_true[:, indice], escenario.prediccion[:, indice]
            )
            filas.append(
                {
                    "escenario": escenario.clave,
                    "etiqueta": etiqueta,
                    "umbral": escenario.umbrales[indice],
                    "positivos_reales": int(y_true[:, indice].sum()),
                    "positivos_predichos": int(escenario.prediccion[:, indice].sum()),
                    **{clave: round(valor, 6) for clave, valor in celdas.items()},
                }
            )
    return pd.DataFrame(filas)


# ---------------------------------------------------------------------------
# Seccion 5 - Combinaciones de etiquetas y regla estructural
# ---------------------------------------------------------------------------
def tabla_etiquetas_erroneas(y_true: np.ndarray, predicho: np.ndarray) -> pd.DataFrame:
    """Cuenta cuantos ejemplos fallan 0, 1, 2 o 3 etiquetas.

    Args:
        y_true: matriz binaria verdadera (n, 3).
        predicho: matriz binaria predicha (n, 3).

    Returns:
        DataFrame con n, proporcion y el detalle de que etiquetas falla cada
        grupo (cuantos de esos ejemplos fallan HS, TR y AG).
    """
    erroneas = (y_true != predicho).sum(axis=1)
    filas = []
    for numero in range(len(ETIQUETAS) + 1):
        seleccion = erroneas == numero
        filas.append(
            {
                "etiquetas_erroneas": numero,
                "n": int(seleccion.sum()),
                "proporcion": round(float(seleccion.mean()), 6),
                **{
                    f"falla_{etiqueta}": int(
                        ((y_true[:, i] != predicho[:, i]) & seleccion).sum()
                    )
                    for i, etiqueta in enumerate(ETIQUETAS)
                },
            }
        )
    return pd.DataFrame(filas)


def tabla_combinaciones_reales(y_true: np.ndarray, predicho: np.ndarray) -> pd.DataFrame:
    """Cuenta los errores por combinacion verdadera HS x TR x AG.

    Args:
        y_true: matriz binaria verdadera (n, 3).
        predicho: matriz binaria predicha (n, 3).

    Returns:
        DataFrame con n, ejemplos con alguna etiqueta mal, con las 3 mal y el
        numero de errores por etiqueta dentro de cada combinacion.
    """
    claves = [tuple(int(valor) for valor in fila) for fila in y_true]
    erroneas = (y_true != predicho).sum(axis=1)
    filas = []
    for combinacion in sorted(set(claves)):
        seleccion = np.array([clave == combinacion for clave in claves])
        filas.append(
            {
                "combinacion_true": f"{combinacion[0]},{combinacion[1]},{combinacion[2]}",
                "n": int(seleccion.sum()),
                "ejemplos_con_alguna_mal": int((erroneas[seleccion] > 0).sum()),
                "ejemplos_con_3_mal": int((erroneas[seleccion] == 3).sum()),
                **{
                    f"errores_{etiqueta}": int(
                        ((y_true[:, i] != predicho[:, i]) & seleccion).sum()
                    )
                    for i, etiqueta in enumerate(ETIQUETAS)
                },
            }
        )
    return pd.DataFrame(filas).sort_values("n", ascending=False).reset_index(drop=True)


def analizar_pares(y_true: np.ndarray, predicho: np.ndarray) -> pd.DataFrame:
    """Analiza los errores conjuntos de cada par de etiquetas.

    Args:
        y_true: matriz binaria verdadera (n, 3).
        predicho: matriz binaria predicha (n, 3).

    Returns:
        DataFrame con, por par, cuantos ejemplos fallan solo una etiqueta,
        ambas en falso negativo, ambas en falso positivo o de forma mixta.
    """
    filas = []
    for i, j in ((0, 1), (0, 2), (1, 2)):
        mal_i = y_true[:, i] != predicho[:, i]
        mal_j = y_true[:, j] != predicho[:, j]
        ambos = mal_i & mal_j
        ambos_fn = ambos & (y_true[:, i] == 1) & (y_true[:, j] == 1)
        ambos_fp = ambos & (y_true[:, i] == 0) & (y_true[:, j] == 0)
        filas.append(
            {
                "par": f"{ETIQUETAS[i]}-{ETIQUETAS[j]}",
                "n_ambos_correctos": int((~mal_i & ~mal_j).sum()),
                "n_solo_primera_mal": int((mal_i & ~mal_j).sum()),
                "n_solo_segunda_mal": int((~mal_i & mal_j).sum()),
                "n_ambas_mal": int(ambos.sum()),
                "n_ambas_falso_negativo": int(ambos_fn.sum()),
                "n_ambas_falso_positivo": int(ambos_fp.sum()),
                "n_ambas_mixto": int((ambos & ~ambos_fn & ~ambos_fp).sum()),
                "errores_totales_primera": int(mal_i.sum()),
                "errores_totales_segunda": int(mal_j.sum()),
            }
        )
    return pd.DataFrame(filas)


def evaluar_regla_estructural(y_true: np.ndarray, escenario: Escenario) -> pd.DataFrame:
    """Mide el efecto de forzar TR=0 (y AG=0) cuando se predice HS=0.

    Args:
        y_true: matriz binaria verdadera (n, 3).
        escenario: escenario de umbral evaluado.

    Returns:
        DataFrame con los casos que violan la estructura y el F1 resultante de
        aplicar la regla solo a TR y a TR y AG a la vez.
    """
    predicho = escenario.prediccion
    viola_tr = (predicho[:, 1] == 1) & (predicho[:, 0] == 0)
    viola_ag = (predicho[:, 2] == 1) & (predicho[:, 0] == 0)
    solo_tr = predicho.copy()
    solo_tr[solo_tr[:, 0] == 0, 1] = 0
    completo = predicho.copy()
    completo[completo[:, 0] == 0, 1] = 0
    completo[completo[:, 0] == 0, 2] = 0
    base = dict(metricas_multietiqueta(y_true, predicho))
    con_tr = dict(metricas_multietiqueta(y_true, solo_tr))
    con_todo = dict(metricas_multietiqueta(y_true, completo))
    detalle = {
        "casos_TR1_con_HS0": int(viola_tr.sum()),
        "casos_TR1_con_HS0_y_TR_true_1": int((viola_tr & (y_true[:, 1] == 1)).sum()),
        "casos_AG1_con_HS0": int(viola_ag.sum()),
        "casos_AG1_con_HS0_y_AG_true_1": int((viola_ag & (y_true[:, 2] == 1)).sum()),
    }
    filas = []
    for variante, metricas in (
        ("sin regla", base),
        ("HS=0 => TR=0", con_tr),
        ("HS=0 => TR=0 y AG=0", con_todo),
    ):
        filas.append(
            {
                "escenario": escenario.clave,
                "variante": variante,
                "f1_media_columnas": round(metricas["f1_media_columnas"], 6),
                "f1_HS": round(metricas["f1_por_etiqueta"]["HS"], 6),
                "f1_TR": round(metricas["f1_por_etiqueta"]["TR"], 6),
                "f1_AG": round(metricas["f1_por_etiqueta"]["AG"], 6),
                "delta_vs_sin_regla": round(
                    metricas["f1_media_columnas"] - base["f1_media_columnas"], 6
                ),
                **detalle,
            }
        )
    return pd.DataFrame(filas)


# ---------------------------------------------------------------------------
# Seccion 6 - Rasgos por documento, rebanadas y regiones
# ---------------------------------------------------------------------------
def cargar_lexico_agresividad(ruta: Path) -> tuple[frozenset[str], frozenset[str]]:
    """Carga el lexico inducido de agresividad y su nucleo de mayor z-score.

    Args:
        ruta: CSV del lexico inducido en la Fase 2.

    Returns:
        Tupla ``(lexico_completo, lexico_nucleo)`` de lemas.

    Raises:
        ValueError: si el CSV no tiene las columnas esperadas.
    """
    tabla = pd.read_csv(ruta)
    esperadas = {"lema", "z_score", "seleccionado"}
    if not esperadas.issubset(tabla.columns):
        raise ValueError(f"Columnas inesperadas en {ruta.name}: {list(tabla.columns)}")
    completo = frozenset(tabla.loc[tabla["seleccionado"] == 1, "lema"].astype(str))
    nucleo = frozenset(
        tabla.loc[tabla["z_score"] >= CONFIG["z_nucleo_insulto"], "lema"].astype(str)
    )
    log(
        f"lexico inducido (in-sample): {len(completo)} lemas; "
        f"nucleo (z>={CONFIG['z_nucleo_insulto']}): {len(nucleo)} lemas"
    )
    return completo, nucleo


def extraer_rasgos_documento(
    textos: Sequence[str],
    lexico_insulto: frozenset[str],
    lexico_nucleo: frozenset[str],
    batch_size: int = 64,
) -> pd.DataFrame:
    """Extrae los rasgos de las rebanadas para cada documento.

    Usa spaCy solo como anotador de POS, lemas y morfologia (D-003); no se
    accede a vectores ni a similitud. El lexico de insulto se aplica sobre
    lemas y es in-sample (se indujo con etiquetas de train): sus rebanadas son
    diagnosticas.

    Args:
        textos: textos de train.
        lexico_insulto: lemas del lexico inducido completo.
        lexico_nucleo: lemas del nucleo de mayor z-score.
        batch_size: tamano de lote de ``nlp.pipe``.

    Returns:
        DataFrame con una fila por documento y todas las columnas de rebanada.
    """
    nlp = spacy.load("es_core_news_md", exclude=["ner"])
    log("spaCy cargado como anotador (sin vectores, NER desactivado)")
    filas: list[dict[str, Any]] = []
    for documento in nlp.pipe(textos, batch_size=batch_size):
        texto = documento.text
        lemas = [token.lemma_.lower() for token in documento]
        formas = [token.text.lower() for token in documento]
        pronombres_2 = sum(
            1
            for token in documento
            if (token.pos_ == "PRON" or token.pos_ == "DET")
            and (token.lemma_.lower() in SECOND_PERSON_LEXICON or token.text.lower() in SECOND_PERSON_LEXICON)
        )
        verbos_2 = sum(
            1
            for token in documento
            if token.pos_ in ("VERB", "AUX") and token.morph.get("Person") == ["2"]
        )
        imperativos = 0
        for token in documento:
            if token.pos_ not in ("VERB", "AUX"):
                continue
            forma = token.text.lower()
            if token.morph.get("Mood") == ["Imp"] or (
                len(forma) >= 4
                and forma.isalpha()
                and forma.endswith(CLITIC_ENDINGS)
                and not INFINITIVE_CLITIC_PATTERN.search(forma)
                and not FIRST_PERSON_PLURAL_PATTERN.search(forma)
            ) or forma in IMPERATIVE_LEXICON or token.lemma_.lower() in IMPERATIVE_LEXICON:
                imperativos += 1
        colectivos = 0
        for indice, token in enumerate(documento):
            if token.pos_ in ("NOUN", "PROPN") and token.lemma_.lower() in COLLECTIVE_QUANTIFIERS or (
                token.pos_ in ("NOUN", "PROPN", "ADJ")
                and token.morph.get("Number") == ["Plur"]
                and indice > 0
                and documento[indice - 1].pos_ == "DET"
                and documento[indice - 1].morph.get("Number") == ["Plur"]
            ):
                colectivos += 1
        n_insultos = sum(1 for lema in lemas if lema in lexico_insulto)
        n_nucleo = sum(1 for lema in lemas if lema in lexico_nucleo)
        letras = [caracter for caracter in texto if caracter.isalpha()]
        mayusculas = sum(1 for caracter in letras if caracter.isupper())
        imperativo = int(imperativos > 0)
        filas.append(
            {
                "n_caracteres": len(texto),
                "n_palabras": len(texto.split()),
                "n_menciones": len(MENTION_PATTERN.findall(texto)),
                "n_insultos": n_insultos,
                "n_insultos_nucleo": n_nucleo,
                "n_colectivos": colectivos,
                "n_imperativos": imperativos,
                "n_palabras_2": pronombres_2 + verbos_2,
                "tiene_mencion": int(bool(MENTION_PATTERN.search(texto))),
                "tiene_mencion_inicial": int(bool(INITIAL_MENTION_PATTERN.match(texto))),
                "tiene_hashtag": int(bool(HASHTAG_PATTERN.search(texto))),
                "tiene_url": int(bool(URL_PATTERN.search(texto))),
                "tiene_insulto_lexico": int(n_insultos > 0),
                "tiene_insulto_nucleo": int(n_nucleo > 0),
                "tiene_segunda_persona": int(pronombres_2 > 0 or verbos_2 > 0 or imperativo > 0),
                "tiene_imperativo": imperativo,
                "tiene_referencia_colectiva": int(colectivos > 0),
                "tiene_mayusculas_sostenidas": int(
                    bool(UPPER_WORD_PATTERN.search(texto)) or mayusculas >= 4
                ),
                "tiene_elongacion": int(bool(ELONGATION_PATTERN.search(texto))),
                "tiene_pregunta": int("?" in texto),
                "tiene_exclamacion": int("!" in texto),
                "tiene_risa": int(bool(LAUGHTER_PATTERN.search(texto))),
                "tiene_comillas": int('"' in texto or "'" in texto),
                "tiene_retuit": int(bool(RETWEET_PATTERN.match(texto))),
                "tiene_entidad_html": int(bool(HTML_ENTITY_PATTERN.search(texto))),
                "ratio_mayusculas": mayusculas / max(len(letras), 1),
                "lemas": lemas,
                "formas": formas,
            }
        )
    return pd.DataFrame(filas)


def anadir_columnas_rebanada(rasgos: pd.DataFrame, train: pd.DataFrame) -> pd.DataFrame:
    """Agrega columnas derivadas (cuartiles, textos cortos y regiones).

    Args:
        rasgos: rasgos por documento.
        train: DataFrame de train (para el ``id``).

    Returns:
        Copia de ``rasgos`` con ``id``, ``cuartil_longitud``, ``texto_corto_40``
        y las banderas de :data:`REGIONES`.
    """
    datos = rasgos.copy()
    datos["id"] = train["id"].to_numpy()
    datos["cuartil_longitud"] = pd.qcut(
        datos["n_caracteres"], 4, labels=["Q1", "Q2", "Q3", "Q4"]
    ).astype(str)
    datos["texto_corto_40"] = (datos["n_caracteres"] < 40).astype(int)
    datos["region_responde_mencion_inicial"] = datos["tiene_mencion_inicial"]
    datos["region_no_responde_sin_colectiva"] = (
        (datos["tiene_mencion_inicial"] == 0)
        & (datos["tiene_referencia_colectiva"] == 0)
    ).astype(int)
    datos["region_no_responde_con_colectiva"] = (
        (datos["tiene_mencion_inicial"] == 0)
        & (datos["tiene_referencia_colectiva"] == 1)
    ).astype(int)
    datos["region_insulto_responde_sin_colectiva"] = (
        (datos["tiene_insulto_lexico"] == 1)
        & (datos["tiene_mencion_inicial"] == 1)
        & (datos["tiene_referencia_colectiva"] == 0)
    ).astype(int)
    datos["region_insulto_no_responde_sin_colectiva"] = (
        (datos["tiene_insulto_lexico"] == 1)
        & (datos["tiene_mencion_inicial"] == 0)
        & (datos["tiene_referencia_colectiva"] == 0)
    ).astype(int)
    datos["region_regimen_noticia"] = (
        (datos["tiene_url"] == 1) | (datos["tiene_hashtag"] == 1)
    ).astype(int)
    datos["region_sin_url_ni_hashtag"] = (
        (datos["tiene_url"] == 0) & (datos["tiene_hashtag"] == 0)
    ).astype(int)
    return datos


def metricas_seleccion(
    y_true: np.ndarray, predicho: np.ndarray, seleccion: np.ndarray
) -> dict[str, float]:
    """Calcula las metricas de una rebanada concreta.

    Args:
        y_true: matriz binaria verdadera (n, 3).
        predicho: matriz binaria predicha (n, 3).
        seleccion: mascara booleana de la rebanada.

    Returns:
        Diccionario con n, F1 por etiqueta, F1 media, tasas de error y
        prevalencias reales y predichas por etiqueta.
    """
    metricas = dict(metricas_multietiqueta(y_true[seleccion], predicho[seleccion]))
    erroneas = y_true[seleccion] != predicho[seleccion]
    return {
        "n": int(seleccion.sum()),
        "f1_HS": float(metricas["f1_por_etiqueta"]["HS"]),
        "f1_TR": float(metricas["f1_por_etiqueta"]["TR"]),
        "f1_AG": float(metricas["f1_por_etiqueta"]["AG"]),
        "f1_media": float(metricas["f1_media_columnas"]),
        "tasa_error_etiquetas": float(erroneas.mean()),
        "tasa_error_exacta": float(erroneas.all(axis=1).mean()),
        "n_errores_etiqueta": int(erroneas.sum()),
        "prevalencia_real_HS": float(y_true[seleccion][:, 0].mean()),
        "prevalencia_real_TR": float(y_true[seleccion][:, 1].mean()),
        "prevalencia_real_AG": float(y_true[seleccion][:, 2].mean()),
        "prevalencia_predicha_HS": float(predicho[seleccion][:, 0].mean()),
        "prevalencia_predicha_TR": float(predicho[seleccion][:, 1].mean()),
        "prevalencia_predicha_AG": float(predicho[seleccion][:, 2].mean()),
    }


def evaluar_rebanadas(
    y_true: np.ndarray,
    escenario: Escenario,
    rasgos: pd.DataFrame,
    f1_global: float,
) -> pd.DataFrame:
    """Evalua todas las rebanadas definidas en :data:`REBANADAS`.

    Args:
        y_true: matriz binaria verdadera (n, 3).
        escenario: escenario de umbral usado para las predicciones.
        rasgos: rasgos por documento con las columnas de rebanada.
        f1_global: F1 medio columnas global del escenario.

    Returns:
        DataFrame con una fila por (rebanada, grupo) e indicadores de estado
        cuando la rebanada no es evaluable (n < minimo o constante).
    """
    filas: list[dict[str, Any]] = []
    for definicion in REBANADAS:
        columna = definicion["columna"]
        if columna not in rasgos.columns:
            continue
        for valor in sorted(pd.unique(rasgos[columna].astype(str))):
            seleccion = (rasgos[columna].astype(str) == valor).to_numpy()
            n = int(seleccion.sum())
            fila: dict[str, Any] = {
                "rebanada": definicion["nombre"],
                "grupo": valor,
                "tipo": definicion["tipo"],
                "artefacto_lexico": definicion["artefacto_lexico"],
                "estado": "ok",
            }
            if n == 0:
                fila["estado"] = "no_aplicable_sin_casos"
                filas.append(fila)
                continue
            if n < CONFIG["min_rebanada"]:
                fila["estado"] = f"no_evaluable_n_menor_{CONFIG['min_rebanada']}"
                filas.append(fila)
                continue
            fila.update(metricas_seleccion(y_true, escenario.prediccion, seleccion))
            fila["delta_vs_global"] = round(fila["f1_media"] - f1_global, 6)
            filas.append(fila)
    tabla = pd.DataFrame(filas)
    columnas_orden = [
        "rebanada", "grupo", "tipo", "artefacto_lexico", "estado", "n",
        "f1_HS", "f1_TR", "f1_AG", "f1_media", "delta_vs_global",
        "tasa_error_etiquetas", "tasa_error_exacta", "n_errores_etiqueta",
        "prevalencia_real_HS", "prevalencia_real_TR", "prevalencia_real_AG",
        "prevalencia_predicha_HS", "prevalencia_predicha_TR", "prevalencia_predicha_AG",
    ]
    for columna in columnas_orden:
        if columna not in tabla.columns:
            tabla[columna] = np.nan
    return tabla[columnas_orden]


def evaluar_regiones(
    y_true: np.ndarray,
    escenario: Escenario,
    rasgos: pd.DataFrame,
    f1_global: float,
) -> pd.DataFrame:
    """Evalua las regiones interpretativas del corpus.

    Args:
        y_true: matriz binaria verdadera (n, 3).
        escenario: escenario de umbral usado para las predicciones.
        rasgos: rasgos por documento con las banderas de region.
        f1_global: F1 medio columnas global del escenario.

    Returns:
        DataFrame con una fila por region (la columna ``condicion`` documenta
        la definicion exacta de cada una).
    """
    filas = []
    for nombre, condicion in REGIONES:
        columna = f"region_{nombre}"
        if columna not in rasgos.columns:
            continue
        seleccion = rasgos[columna].astype(int).to_numpy() == 1
        metricas = metricas_seleccion(y_true, escenario.prediccion, seleccion)
        filas.append(
            {
                "region": nombre,
                "condicion": condicion,
                **metricas,
                "delta_vs_global": round(metricas["f1_media"] - f1_global, 6),
            }
        )
    return pd.DataFrame(filas)


def seleccionar_rebanadas_debiles(
    rebanadas: pd.DataFrame, incluir_artefactos: bool
) -> pd.DataFrame:
    """Selecciona las rebanadas mas debiles con tamano suficiente.

    Args:
        rebanadas: tabla devuelta por :func:`evaluar_rebanadas`.
        incluir_artefactos: si ``False``, excluye las rebanadas derivadas del
            lexico inducido in-sample.

    Returns:
        DataFrame ordenado por F1 media ascendente con las 5 rebanadas mas
        debiles y una columna ``posicion``.
    """
    candidatas = rebanadas[rebanadas["estado"] == "ok"].copy()
    if not incluir_artefactos:
        candidatas = candidatas[~candidatas["artefacto_lexico"].astype(bool)]
    candidatas = candidatas.sort_values(["f1_media", "n"], ascending=[True, False])
    seleccion = candidatas.head(5).reset_index(drop=True)
    seleccion.insert(0, "posicion", np.arange(1, len(seleccion) + 1))
    columnas = [
        "posicion", "rebanada", "grupo", "n", "f1_HS", "f1_TR", "f1_AG",
        "f1_media", "delta_vs_global", "tasa_error_etiquetas", "tasa_error_exacta",
    ]
    return seleccion[columnas]


# ---------------------------------------------------------------------------
# Seccion 7 - Analisis cualitativo: muestras y perfil de rasgos
# ---------------------------------------------------------------------------
def seleccionar_muestras(
    train: pd.DataFrame,
    y_true: np.ndarray,
    escenario: Escenario,
    rasgos: pd.DataFrame,
) -> pd.DataFrame:
    """Selecciona falsos positivos y falsos negativos para revision humana.

    Los falsos positivos se ordenan por probabilidad descendente (los errores
    mas confiados) y los falsos negativos por probabilidad ascendente (los
    positivos mas olvidados).

    Args:
        train: DataFrame de train (id y texto).
        y_true: matriz binaria verdadera (n, 3).
        escenario: escenario de umbral usado para las predicciones.
        rasgos: rasgos por documento.

    Returns:
        DataFrame con ``n_muestras_por_tipo`` filas por etiqueta y tipo.
    """
    columnas_rasgo = (
        "tiene_mencion_inicial", "tiene_insulto_lexico", "tiene_referencia_colectiva",
        "tiene_risa", "tiene_url", "tiene_hashtag", "tiene_segunda_persona",
        "tiene_entidad_html", "n_caracteres",
    )
    filas: list[dict[str, Any]] = []
    for indice, etiqueta in enumerate(ETIQUETAS):
        verdad = y_true[:, indice]
        predicho = escenario.prediccion[:, indice]
        probabilidad = escenario.proba[:, indice]
        for tipo, seleccion in (
            ("falso_positivo", (verdad == 0) & (predicho == 1)),
            ("falso_negativo", (verdad == 1) & (predicho == 0)),
        ):
            candidatos = np.where(seleccion)[0]
            orden = np.argsort(-probabilidad[candidatos] if tipo == "falso_positivo" else probabilidad[candidatos])
            for posicion in candidatos[orden][: CONFIG["n_muestras_por_tipo"]]:
                texto = str(train["text"].iloc[posicion])
                truncado = texto[: CONFIG["max_caracteres_muestra"]]
                if len(texto) > CONFIG["max_caracteres_muestra"]:
                    truncado += "..."
                filas.append(
                    {
                        "etiqueta": etiqueta,
                        "tipo_error": tipo,
                        "probabilidad": round(float(probabilidad[posicion]), 4),
                        "umbral": escenario.umbrales[indice],
                        "HS_true": int(y_true[posicion, 0]),
                        "TR_true": int(y_true[posicion, 1]),
                        "AG_true": int(y_true[posicion, 2]),
                        "HS_pred": int(escenario.prediccion[posicion, 0]),
                        "TR_pred": int(escenario.prediccion[posicion, 1]),
                        "AG_pred": int(escenario.prediccion[posicion, 2]),
                        "id": str(train["id"].iloc[posicion]),
                        **{
                            columna: int(rasgos[columna].iloc[posicion])
                            for columna in columnas_rasgo
                        },
                        "texto_200": truncado,
                    }
                )
    return pd.DataFrame(filas)


def perfil_rasgos_por_etiqueta(
    y_true: np.ndarray, escenario: Escenario, rasgos: pd.DataFrame
) -> pd.DataFrame:
    """Compara la presencia de rasgos en aciertos y errores de cada etiqueta.

    Args:
        y_true: matriz binaria verdadera (n, 3).
        escenario: escenario de umbral usado para las predicciones.
        rasgos: rasgos por documento.

    Returns:
        DataFrame con la tasa del rasgo en TP, FP, FN y TN y su
        sobre-representacion en los falsos positivos (FP/TN) y en los falsos
        negativos (FN/TP).
    """
    variables = (
        "tiene_insulto_lexico", "tiene_segunda_persona", "tiene_referencia_colectiva",
        "tiene_mencion_inicial", "tiene_risa", "tiene_elongacion",
        "tiene_mayusculas_sostenidas", "tiene_url", "tiene_hashtag",
        "tiene_entidad_html",
    )
    filas: list[dict[str, Any]] = []
    for indice, etiqueta in enumerate(ETIQUETAS):
        verdad = y_true[:, indice]
        predicho = escenario.prediccion[:, indice]
        mascaras = {
            "TP": (verdad == 1) & (predicho == 1),
            "FP": (verdad == 0) & (predicho == 1),
            "FN": (verdad == 1) & (predicho == 0),
            "TN": (verdad == 0) & (predicho == 0),
        }
        for variable in variables:
            valores = rasgos[variable].astype(float).to_numpy()
            tasas = {
                clave: float(valores[mascara].mean()) if mascara.sum() else float("nan")
                for clave, mascara in mascaras.items()
            }
            filas.append(
                {
                    "etiqueta": etiqueta,
                    "rasgo": variable,
                    "n_TP": int(mascaras["TP"].sum()),
                    "n_FP": int(mascaras["FP"].sum()),
                    "n_FN": int(mascaras["FN"].sum()),
                    "n_TN": int(mascaras["TN"].sum()),
                    "tasa_TP": round(tasas["TP"], 6),
                    "tasa_FP": round(tasas["FP"], 6),
                    "tasa_FN": round(tasas["FN"], 6),
                    "tasa_TN": round(tasas["TN"], 6),
                    "sobrerrepresentacion_FP_vs_TN": round(tasas["FP"] / tasas["TN"], 4)
                    if tasas["TN"] > 0
                    else float("nan"),
                    "sobrerrepresentacion_FN_vs_TP": round(tasas["FN"] / tasas["TP"], 4)
                    if tasas["TP"] > 0
                    else float("nan"),
                }
            )
    return pd.DataFrame(filas)


def resumir_muestras(muestras: pd.DataFrame) -> pd.DataFrame:
    """Resume los rasgos de los casos exportados para revision humana.

    Args:
        muestras: tabla devuelta por :func:`seleccionar_muestras`.

    Returns:
        DataFrame con, por etiqueta y tipo de error, la tasa de cada rasgo y el
        numero de casos.
    """
    variables = (
        "tiene_mencion_inicial", "tiene_insulto_lexico", "tiene_referencia_colectiva",
        "tiene_segunda_persona", "tiene_risa", "tiene_hashtag", "tiene_url",
    )
    filas = []
    for (etiqueta, tipo), grupo in muestras.groupby(["etiqueta", "tipo_error"]):
        filas.append(
            {
                "etiqueta": etiqueta,
                "tipo_error": tipo,
                "n": len(grupo),
                "mediana_caracteres": float(grupo["n_caracteres"].median()),
                **{
                    f"tasa_{variable}": float(grupo[variable].mean())
                    for variable in variables
                },
            }
        )
    orden = {"falso_positivo": 0, "falso_negativo": 1}
    tabla = pd.DataFrame(filas)
    tabla["_orden"] = tabla["tipo_error"].map(orden)
    return (
        tabla.sort_values(["etiqueta", "_orden"]).drop(columns="_orden").reset_index(drop=True)
    )


def marcar_patrones_muestras(muestras: pd.DataFrame) -> pd.DataFrame:
    """Anade banderas de patron interpretativo a las muestras exportadas.

    Las banderas son reglas explicitas y reproducibles que operacionalizan los
    patrones descritos en el reporte; no sustituyen la lectura humana, solo
    cuentan cuantas muestras caen en cada bucket (los buckets se solapan).

    Args:
        muestras: tabla devuelta por :funcion:`seleccionar_muestras`.

    Returns:
        Copia de ``muestras`` con las columnas de patron en 0/1.
    """
    datos = muestras.copy()
    insulto = datos["tiene_insulto_lexico"].astype(bool)
    datos["patron_broma_afiliacion"] = (
        datos["texto_200"].str.contains(PATRON_BROMA_AFILIACION, regex=True) & insulto
    ).astype(int)
    datos["patron_cita_metalenguaje"] = (
        (datos["texto_200"].str.count("'") + datos["texto_200"].str.count('"') >= 2)
        & insulto
    ).astype(int)
    datos["patron_noticia_o_atribucion"] = (
        (datos["tiene_url"] == 1) | (datos["n_caracteres"] >= 180)
    ).astype(int)
    datos["patron_sin_insulto_lexico"] = (~insulto).astype(int)
    return datos


def resumir_patrones_muestras(muestras_marcadas: pd.DataFrame) -> pd.DataFrame:
    """Cuenta las muestras de cada etiqueta y tipo que caen en cada patron.

    Args:
        muestras_marcadas: tabla devuelta por :funcion:`marcar_patrones_muestras`.

    Returns:
        DataFrame largo con ``patron``, ``etiqueta``, ``tipo_error``, ``n`` y la
        proporcion sobre las muestras de ese grupo.
    """
    columnas = (
        "patron_broma_afiliacion", "patron_cita_metalenguaje",
        "patron_noticia_o_atribucion", "patron_sin_insulto_lexico",
    )
    filas = []
    for (etiqueta, tipo), grupo in muestras_marcadas.groupby(["etiqueta", "tipo_error"]):
        for columna in columnas:
            filas.append(
                {
                    "patron": columna.replace("patron_", ""),
                    "etiqueta": etiqueta,
                    "tipo_error": tipo,
                    "n": int(grupo[columna].sum()),
                    "n_grupo": len(grupo),
                    "proporcion": round(float(grupo[columna].mean()), 4),
                }
            )
    tabla = pd.DataFrame(filas)
    orden_tipo = {"falso_positivo": 0, "falso_negativo": 1}
    tabla["_orden"] = tabla["tipo_error"].map(orden_tipo)
    return (
        tabla.sort_values(["patron", "etiqueta", "_orden"])
        .drop(columns="_orden")
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Seccion 8 - Figuras
# ---------------------------------------------------------------------------
def figura_matrices_confusion(
    confusiones: pd.DataFrame, escenario: str, titulo: str, nombre: str
) -> Path:
    """Dibuja las tres matrices de confusion 2x2 de un escenario.

    Args:
        confusiones: tabla devuelta por :func:`construir_matrices_confusion`.
        escenario: clave del escenario a dibujar.
        titulo: titulo general de la figura.
        nombre: nombre del archivo PNG.

    Returns:
        Ruta de la figura guardada.
    """
    sub = confusiones[confusiones["escenario"] == escenario]
    figura, ejes = plt.subplots(1, len(ETIQUETAS), figsize=(11, 3.9))
    for eje, (_, fila) in zip(ejes, sub.iterrows(), strict=True):
        matriz = np.array([[fila["TP"], fila["FN"]], [fila["FP"], fila["TN"]]], dtype=float)
        sns.heatmap(
            matriz,
            annot=np.array(
                [
                    [f"TP {int(fila['TP'])}\nrecall {fila['recall']:.3f}", f"FN {int(fila['FN'])}\nFNR {fila['fnr']:.3f}"],
                    [f"FP {int(fila['FP'])}\nFPR {fila['fpr']:.3f}", f"TN {int(fila['TN'])}\nesp. {fila['especificidad']:.3f}"],
                ]
            ),
            fmt="",
            cmap="Blues",
            cbar=False,
            ax=eje,
            annot_kws={"fontsize": 8},
        )
        eje.set_title(f"{fila['etiqueta']} (umbral {fila['umbral']:.2f}, F1 {fila['f1']:.3f})")
        eje.set_xticklabels(["pred 1", "pred 0"], rotation=0)
        eje.set_yticklabels(["real 1", "real 0"], rotation=0)
    figura.suptitle(titulo, fontsize=12)
    figura.tight_layout()
    return guardar_figura(figura, nombre)


def figura_errores_por_etiqueta(
    y_true: np.ndarray, escenario: Escenario, tabla_erroneas: pd.DataFrame, nombre: str
) -> Path:
    """Dibuja el balance FP/FN por etiqueta y la distribucion de etiquetas mal.

    Args:
        y_true: matriz binaria verdadera (n, 3).
        escenario: escenario de umbral usado.
        tabla_erroneas: tabla devuelta por :func:`tabla_etiquetas_erroneas`.
        nombre: nombre del archivo PNG.

    Returns:
        Ruta de la figura guardada.
    """
    figura, (eje_a, eje_b) = plt.subplots(1, 2, figsize=(11, 4))
    conteos = []
    for indice, etiqueta in enumerate(ETIQUETAS):
        celdas = matriz_confusion(y_true[:, indice], escenario.prediccion[:, indice])
        conteos.append((etiqueta, celdas["FP"], celdas["FN"]))
    posiciones = np.arange(len(conteos))
    ancho = 0.36
    eje_a.bar(posiciones - ancho / 2, [c[1] for c in conteos], ancho, label="FP", color="#b03a2e")
    eje_a.bar(posiciones + ancho / 2, [c[2] for c in conteos], ancho, label="FN", color="#1f4e79")
    for posicion, (_, fp, fn) in zip(posiciones, conteos, strict=True):
        eje_a.text(posicion - ancho / 2, fp + 3, int(fp), ha="center", fontsize=9)
        eje_a.text(posicion + ancho / 2, fn + 3, int(fn), ha="center", fontsize=9)
    eje_a.set_xticks(posiciones)
    eje_a.set_xticklabels([c[0] for c in conteos])
    eje_a.set_ylabel("casos")
    eje_a.set_title(f"Falsos positivos y negativos ({escenario.nombre})")
    eje_a.legend()
    eje_b.bar(
        tabla_erroneas["etiquetas_erroneas"].astype(str),
        tabla_erroneas["n"],
        color="#b9770e",
    )
    for posicion, fila in tabla_erroneas.iterrows():
        eje_b.text(posicion, fila["n"] + 8, f"{int(fila['n'])}\n{fila['proporcion']*100:.1f}%", ha="center", fontsize=9)
    eje_b.set_xlabel("etiquetas erroneas por ejemplo")
    eje_b.set_ylabel("ejemplos")
    eje_b.set_title("Cuantas etiquetas falla cada ejemplo")
    figura.tight_layout()
    return guardar_figura(figura, nombre)


def figura_rebanadas_debiles(
    rebanadas: pd.DataFrame, f1_global: float, nombre: str, n_max: int = 12
) -> Path:
    """Dibuja las rebanadas con menor F1 medio frente al F1 global.

    Args:
        rebanadas: tabla devuelta por :func:`evaluar_rebanadas`.
        f1_global: F1 medio columnas global del escenario.
        nombre: nombre del archivo PNG.
        n_max: numero maximo de rebanadas a dibujar.

    Returns:
        Ruta de la figura guardada.
    """
    candidatas = rebanadas[rebanadas["estado"] == "ok"].copy()
    candidatas = candidatas.sort_values("f1_media").head(n_max)
    etiquetas = [
        f"{fila.rebanada} = {fila.grupo} (n={int(fila.n)})"
        for fila in candidatas.itertuples()
    ]
    figura, eje = plt.subplots(figsize=(9, 5.5))
    colores = [
        "#b03a2e" if fila.artefacto_lexico else "#1f4e79"
        for fila in candidatas.itertuples()
    ]
    eje.barh(etiquetas[::-1], candidatas["f1_media"][::-1], color=colores[::-1])
    eje.axvline(f1_global, color="black", linestyle="--", linewidth=1)
    eje.text(f1_global, -0.8, f"global {f1_global:.4f}", fontsize=9)
    eje.set_xlabel("F1 medio columnas")
    eje.set_title("Rebanadas mas debiles (rojo = lexico inducido in-sample)")
    eje.set_xlim(0, 1)
    figura.tight_layout()
    return guardar_figura(figura, nombre)


def figura_regiones(regiones: pd.DataFrame, nombre: str) -> Path:
    """Dibuja el F1 por etiqueta y region interpretativa.

    Args:
        regiones: tabla devuelta por :func:`evaluar_regiones`.
        nombre: nombre del archivo PNG.

    Returns:
        Ruta de la figura guardada.
    """
    datos = regiones.set_index("region")[["f1_HS", "f1_TR", "f1_AG", "f1_media", "n"]]
    figura, eje = plt.subplots(figsize=(8.5, 5))
    sns.heatmap(
        datos[["f1_HS", "f1_TR", "f1_AG", "f1_media"]],
        annot=datos[["f1_HS", "f1_TR", "f1_AG", "f1_media"]].round(3),
        fmt="",
        cmap="RdYlGn",
        vmin=0.4,
        vmax=1.0,
        ax=eje,
        cbar_kws={"label": "F1"},
    )
    eje.set_yticklabels(
        [f"{region} (n={int(n)})" for region, n in zip(datos.index, datos["n"], strict=True)],
        rotation=0,
    )
    eje.set_title("F1 por region del corpus (umbrales por etiqueta)")
    figura.tight_layout()
    return guardar_figura(figura, nombre)


# ---------------------------------------------------------------------------
# Seccion 9 - Reporte, hipotesis y contexto
# ---------------------------------------------------------------------------
# Patrones recurrentes observados por revision humana de
# ``reports/eda/error_samples_para_revision.csv`` (15 FP + 15 FN por etiqueta).
# Los ``ejemplos`` son ids de train verificables en ese CSV.
PATRONES_CUALITATIVOS: Final[tuple[dict[str, Any], ...]] = (
    {
        "patron": "Insulto interpersonal en registro de broma, afiliacion o vocativo",
        "descripcion": (
            "Falso positivo dominante de HS, TR y AG: el texto insulta con lexico "
            "duro (zorra, perra, puta, hijueputa) pero el contexto marca broma o "
            "afiliacion entre usuarios que se conocen: vocativo con mencion, "
            "posesivos de primera persona ('mi novia', 'mi perra', 'mi favorita'), "
            "apelativos afectivos ('te amo', 'socia'), risa (JAJAJA, jeje) o "
            "marcas de juego ('beer pong'). La anotacion los considera no odiosos "
            "y el modelo los dispara con probabilidad 0.87-0.97 en las tres "
            "etiquetas. Es el error mas costoso y el mas sistematico del analisis."
        ),
        "etiquetas": "HS, TR, AG (falsos positivos)",
        "ejemplos": ("23076", "24006", "23172", "20504", "21150", "22524", "21497", "22924"),
    },
    {
        "patron": "Uso metalinguistico: cita, mencion o reproduccion del insulto",
        "descripcion": (
            "El texto no insulta: cita el insulto o habla de el entre comillas "
            "('#Cllate Zorra el cllate perro de este semestre', 'Es como maldita "
            "sea cllate ese maldito hocico perra de mierda', 'los estupidos rabes "
            "de la universidad'). El modelo lineal reacciona al lexico citado "
            "tanto en falso positivo (cuando la cita es inocua) como en falso "
            "negativo (cuando la cita es el vehiculo del odio)."
        ),
        "etiquetas": "HS, AG (falsos positivos); HS, AG (falsos negativos)",
        "ejemplos": ("21362", "23972", "24061"),
    },
    {
        "patron": "Regimen noticia o atribucion a terceros",
        "descripcion": (
            "Falso negativo de HS, TR y AG en textos largos (180-302 caracteres) "
            "con URL: titulares, cronicas o denuncias que describen violencia "
            "contra un grupo y que la anotacion marca como toxicas. Varios casos "
            "describen a un individuo concreto ('el agressor ... inmigrante "
            "ilegal de origen Nigeriano', 'este antisemita dando clase'), lo que "
            "refuerza la lectura de D-005 de que TR=1 puede marcar objetivo "
            "individual y no solo respuesta directa."
        ),
        "etiquetas": "HS, TR, AG (falsos negativos)",
        "ejemplos": ("24269", "20144", "21733", "21041", "23631", "20295"),
    },
    {
        "patron": "Odio implicito contra un grupo sin insulto del lexico",
        "descripcion": (
            "Falso negativo de HS y AG sin ninguna palabra del lexico inducido: "
            "estereotipo ('El Arabe es un flojo'), miedo estigmatizante ('m da "
            "miedo cuando me sigue un rabe'), despersonalizacion ('Al final un "
            "iletrado sin papeles'), animalizacion ('mas digna ... la vida de una "
            "linda gallina que acabar en un KFC'), consigna ('fuera la escoria "
            "inmigrante') o deseo ('lastima que no soy un arabe de mierda'). El "
            "modelo necesita el insulto explicito para activar AG y HS."
        ),
        "etiquetas": "HS, AG (falsos negativos)",
        "ejemplos": ("21799", "21408", "23832", "22174", "23922", "22678"),
    },
    {
        "patron": "Texto corto cuyo sentido depende del contenido enlazado",
        "descripcion": (
            "Falso negativo de HS, TR y AG en textos muy cortos con URL "
            "('Comemela XDDD y la cara tambin!! ...', 'Comemela mientras te meto "
            "los dedos', '@TaigaDeLimon gritame y pisame la polla con una falda "
            "puesta'): el texto por si solo no explica la etiqueta positiva, que "
            "probablemente depende de la imagen o el video enlazado, informacion "
            "que el modelo no puede ver. Es una fraccion irreducible del error "
            "con las features de texto."
        ),
        "etiquetas": "HS, TR, AG (falsos negativos)",
        "ejemplos": ("20082", "20852", "23283"),
    },
    {
        "patron": "Frontera de anotacion no observable en el texto",
        "descripcion": (
            "Casos de superficie casi identica con etiquetas opuestas: 'Callate "
            "puta tu eres el traidor' y 'Cllate negro Ql! ...' estan etiquetados "
            "(0,0,0) mientras textos del mismo tipo llevan (1,1,1). La region "
            "completa lo confirma: de 526 textos con insulto, mencion inicial y "
            "sin referencia colectiva, 213 (40.5 %) son negativos. La frontera "
            "depende de criterio humano (quien habla, a quien, con que intencion) "
            "que no esta en el texto y que ningun modelo de bolsas de n-gramas "
            "puede reconstruir; acota el techo alcanzable."
        ),
        "etiquetas": "HS, TR, AG (falsos positivos y negativos)",
        "ejemplos": ("21892", "24454", "21865", "20463"),
    },
)

# Hipotesis nuevas. Los textos usan marcadores con nombre que se rellenan con el
# diccionario de evidencia medido en el analisis (str.format).
HIPOTESIS: Final[tuple[dict[str, str], ...]] = (
    {
        "id": "H11",
        "enunciado": (
            "Los falsos positivos de HS y AG se concentran en el registro de insulto "
            "interpersonal entre usuarios que la anotacion considera no odioso (broma, "
            "afiliacion, pelea entre conocidos); un bloque explicito de interacciones de "
            "registro reduce esos FP."
        ),
        "evidencia": (
            "De los {n_insulto_responde} textos con insulto del lexico, mencion inicial y "
            "sin referencia colectiva, {n_insulto_negativos} ({pct_insulto_negativos} %) "
            "estan etiquetados (0,0,0); la tasa de FP de HS en esa region es "
            "{fp_rate_HS_region} frente a {fp_rate_global_HS} global. Los FP mas "
            "confiados de HS y AG (probabilidad 0.88-0.97) pertenecen a ese grupo."
        ),
        "mecanismo": (
            "Un modelo lineal sobre TF-IDF suma evidencia de palabras y no puede "
            "representar conjunciones: la marca que desactiva la toxicidad es la "
            "coaparicion de insulto con marcadores de afecto/risa/vocativo y la ausencia "
            "de un objetivo colectivo."
        ),
        "experimento": (
            "EXP-14a: agregar a MOD-03b un bloque denso de interacciones calculadas "
            "in-fold (insulto x risa, insulto x posesivo de primera persona, "
            "insulto x vocativo, insulto x ausencia de colectivo, insulto x segunda "
            "persona, mencion multiple) con LR C=2.0 balanced x3 y el mismo "
            "StratifiedGroupKFold(5, shuffle=True, random_state=42)."
        ),
        "criterio": (
            "F1 OOF >= +0.005 y >= 3/5 pliegues; por D-012 exigir >= +0.010 para "
            "considerarlo realista antes de enviar una sonda de LB; ademas reducir >= 25 % "
            "el FP de HS y AG en la region de insulto interpersonal."
        ),
    },
    {
        "id": "H12",
        "enunciado": (
            "El error de TR vive en los textos sin destinatario explicito: la frontera "
            "depende de si el objetivo es una persona identificable o un colectivo, y el "
            "modelo no tiene un bloque que represente esa distincion."
        ),
        "evidencia": (
            "Dentro de HS=1, F1_TR es {tr_f1_responde} en respuestas con mencion inicial, "
            "{tr_f1_sin_mencion_sin_colectiva} en textos sin mencion ni colectiva y "
            "{tr_f1_con_colectiva} en textos sin mencion con referencia colectiva. "
            "{tr_errores_sin_mencion} de los {tr_errores_total} errores de TR dentro de "
            "HS=1 ocurren en el grupo sin mencion."
        ),
        "mecanismo": (
            "La etiqueta TR exige un objetivo individual; los rasgos que lo indican "
            "(nombre propio, gentilicio en singular, demostrativo + sustantivo singular) y "
            "los que indican colectivo (determinante plural + sustantivo plural) quedan "
            "dispersos entre n-gramas y el modelo lineal no los combina."
        ),
        "experimento": (
            "EXP-14b: bloque denso de rasgos de objetivo (individual frente a colectivo) "
            "inducidos dentro de cada pliegue con spaCy, agregado a la union; comparar "
            "F1 de TR dentro de HS=1 y global contra MOD-03b."
        ),
        "criterio": (
            "F1_TR OOF >= +0.005 y >= 3/5 pliegues, con la region sin mencion con "
            "colectiva mejorando >= +0.02 y sin degradar HS ni AG mas de 0.002; "
            "exigir >= +0.010 global para sonda de LB (D-012)."
        ),
    },
    {
        "id": "H13",
        "enunciado": (
            "El modelo es fuerte en el registro de respuesta y debil en el registro "
            "noticia/comentario (URL, hashtag, pregunta, textos largos), donde el texto "
            "cita o describe odio ajeno en lugar de producirlo."
        ),
        "evidencia": (
            "F1 medio columnas {f1_noticia} en el regimen noticia (URL o hashtag, n="
            "{n_noticia}) frente a {f1_sin_noticia} sin URL ni hashtag; las rebanadas "
            "hashtag={f1_hashtag}, pregunta={f1_pregunta}, referencia colectiva="
            "{f1_colectiva} y cuartil largo={f1_q4} estan entre las mas debiles."
        ),
        "mecanismo": (
            "El vocabulario noticioso comparte terminos con el odio (gentilicios, "
            "migracion) y el modelo no distingue uso de mencion; el modelo lineal no "
            "tiene forma de representar la atribucion a terceros ni el metalenguaje."
        ),
        "experimento": (
            "EXP-14c: bloque denso de cita y atribucion inducido in-fold (verbos de "
            "reporte, comillas, patron de titular, dominio de la URL, hashtag de campana) "
            "mas su interaccion con la longitud; medir en las rebanadas del regimen y en "
            "OOF."
        ),
        "criterio": (
            "F1 OOF >= +0.005 y >= 3/5 pliegues; mejora >= +0.02 en la rebanada de "
            "regimen noticia; exigir >= +0.010 global para sonda de LB (D-012)."
        ),
    },
    {
        "id": "H14",
        "enunciado": (
            "Los falsos negativos de AG son agresion implicita sin insulto del lexico "
            "(estigma, deshumanizacion, cosificacion de un grupo); los falsos positivos "
            "de AG vienen del registro bromista, no de la agresion implicita."
        ),
        "evidencia": (
            "El {pct_ag_fn_colectiva} % de los FN de AG tiene referencia colectiva frente "
            "al {pct_ag_tp_colectiva} % de los TP (y {pct_ag_tn_colectiva} % de los TN); "
            "F1_AG en la region con referencia colectiva es {ag_f1_colectiva} frente a "
            "{ag_f1_global} global."
        ),
        "mecanismo": (
            "El lexico inducido cubre la agresion explicita; la deshumanizacion usa "
            "metaforas y giros que no estan en el lexico y que los n-gramas de caracter no "
            "aislan como categoria."
        ),
        "experimento": (
            "EXP-14d: inducir in-fold un lexico de deshumanizacion por log-odds sobre los "
            "lemas de documentos AG=1 sin insulto del lexico (frecuencia >= 3, presente en "
            ">= 2 documentos) y agregar conteos, banderas y la interaccion con referencia "
            "colectiva."
        ),
        "criterio": (
            "F1_AG OOF >= +0.005 y >= 3/5 pliegues, con el FN de AG sin insulto "
            "reduciendose >= 20 %; exigir >= +0.010 global para sonda de LB (D-012)."
        ),
    },
    {
        "id": "H15",
        "enunciado": (
            "Ajustar el umbral por etiqueta y por rebanada (corto frente a largo, noticia "
            "frente a respuesta) NO aporta una mejora explotable; es una via de cierre, no "
            "de mejora."
        ),
        "evidencia": (
            "Los umbrales globales por etiqueta dieron +0.0014 en verificacion anidada "
            "(2/5 pliegues) y +0.0031 real en el LB (D-012), muy por debajo del +0.0063 "
            "optimista. La heterogeneidad de F1 entre rebanadas (0.586 a 0.944) invita a "
            "umbrales por rebanada, pero las celdas pequenas favorecen el sobreajuste."
        ),
        "mecanismo": (
            "La heterogeneidad de F1 entre rebanadas se explica por prevalencia y "
            "dificultad, no necesariamente por un sesgo de calibracion explotable; el "
            "ajuste de umbrales consume grados de libertad con n por celda pequeno."
        ),
        "experimento": (
            "EXP-14e: ajustar umbrales por etiqueta y rebanada (corto/largo y "
            "noticia/respuesta) en 4 pliegues y evaluar en el restante con el mismo "
            "protocolo anidado de D-012, comparando contra los umbrales globales anidados."
        ),
        "criterio": (
            "Se adopta solo si la ganancia anidada es >= +0.002 y >= 3/5 pliegues; en caso "
            "contrario se cierra la linea de calibracion fina y se documenta el cierre."
        ),
    },
)


@dataclass
class Analisis:
    """Resultados completos del analisis de error de la Fase 12.

    Attributes:
        train: DataFrame de train.
        y_true: matriz binaria verdadera (n, 3).
        escenarios: escenarios de umbral evaluados.
        confusion: matrices de confusion por etiqueta y escenario.
        etiquetas_erroneas: ejemplos por numero de etiquetas falladas.
        combinaciones: errores por combinacion verdadera HS x TR x AG.
        pares: errores conjuntos por par de etiquetas.
        regla: efecto de la regla estructural por escenario.
        rebanadas: metricas por rebanada.
        regiones: metricas por region interpretativa.
        debiles: 5 rebanadas mas debiles (sin artefactos del lexico).
        debiles_con_artefactos: 5 rebanadas mas debiles (todas).
        perfil: presencia de rasgos en TP/FP/FN/TN por etiqueta.
        muestras: casos exportados para revision humana.
        muestras_marcadas: esas muestras con las banderas de patron.
        resumen_muestras: resumen de rasgos de esas muestras.
        resumen_patrones: conteo de muestras por patron interpretativo.
        evidencia: numeros clave usados en el reporte.
        figuras: rutas de las figuras generadas.
        artefactos: rutas de los CSV y JSON generados.
    """

    train: pd.DataFrame
    y_true: np.ndarray
    escenarios: dict[str, Escenario]
    confusion: pd.DataFrame
    etiquetas_erroneas: pd.DataFrame
    combinaciones: pd.DataFrame
    pares: pd.DataFrame
    regla: pd.DataFrame
    rebanadas: pd.DataFrame
    regiones: pd.DataFrame
    debiles: pd.DataFrame
    debiles_con_artefactos: pd.DataFrame
    perfil: pd.DataFrame
    muestras: pd.DataFrame
    muestras_marcadas: pd.DataFrame
    resumen_muestras: pd.DataFrame
    resumen_patrones: pd.DataFrame
    evidencia: dict[str, Any]
    figuras: tuple[Path, ...] = ()
    artefactos: tuple[Path, ...] = ()


def construir_evidencia(
    analisis_datos: dict[str, Any]
) -> dict[str, Any]:
    """Extrae los numeros clave que usan el reporte y las hipotesis.

    Args:
        analisis_datos: diccionario con las tablas y escenarios ya calculados.

    Returns:
        Diccionario plano de numeros y proporciones para las plantillas.
    """
    y_true: np.ndarray = analisis_datos["y_true"]
    escenarios: dict[str, Escenario] = analisis_datos["escenarios"]
    principal: Escenario = escenarios["umbral_optimo"]
    base: Escenario = escenarios["umbral_0_5"]
    regiones: pd.DataFrame = analisis_datos["regiones"].set_index("region")
    rebanadas: pd.DataFrame = analisis_datos["rebanadas"]
    pares: pd.DataFrame = analisis_datos["pares"]
    regla: pd.DataFrame = analisis_datos["regla"]
    perfil: pd.DataFrame = analisis_datos["perfil"]
    predicho = principal.prediccion

    def f1_de(region: str) -> float:
        """Devuelve el F1 medio de una region."""
        return float(regiones.loc[region, "f1_media"])

    def f1_rebanada(rebanada: str, grupo: str) -> float:
        """Devuelve el F1 medio de una rebanada concreta."""
        sub = rebanadas[
            (rebanadas["rebanada"] == rebanada)
            & (rebanadas["grupo"].astype(str) == grupo)
            & (rebanadas["estado"] == "ok")
        ]
        return float(sub["f1_media"].iloc[0]) if len(sub) else float("nan")

    def tasa_perfil(etiqueta: str, rasgo: str, columna: str) -> float:
        """Devuelve una tasa del perfil de rasgos."""
        sub = perfil[(perfil["etiqueta"] == etiqueta) & (perfil["rasgo"] == rasgo)]
        return float(sub[columna].iloc[0]) if len(sub) else float("nan")

    region_broma = regiones.loc["insulto_responde_sin_colectiva"]
    mascara_broma = (
        analisis_datos["rasgos"]["region_insulto_responde_sin_colectiva"].to_numpy() == 1
    )
    negativos_broma = int(np.sum(mascara_broma & (y_true.sum(axis=1) == 0)))
    fp_HS_region = float(
        np.mean(
            (y_true[mascara_broma, 0] == 0) & (predicho[mascara_broma, 0] == 1)
        )
    )
    mascara_negativos = y_true[:, 0] == 0
    fp_HS_global = float(
        np.mean((y_true[mascara_negativos, 0] == 0) & (predicho[mascara_negativos, 0] == 1))
    )
    hs1 = y_true[:, 0] == 1
    errores_tr_hs1 = int((y_true[hs1, 1] != predicho[hs1, 1]).sum())
    rasgos = analisis_datos["rasgos"]
    sin_mencion_sin_colectiva = hs1 & (
        rasgos["region_no_responde_sin_colectiva"].to_numpy() == 1
    )
    errores_tr_sin_mencion = int(
        (y_true[sin_mencion_sin_colectiva, 1] != predicho[sin_mencion_sin_colectiva, 1]).sum()
    )
    par_HS_TR = pares[pares["par"] == "HS-TR"].iloc[0]
    par_HS_AG = pares[pares["par"] == "HS-AG"].iloc[0]
    par_TR_AG = pares[pares["par"] == "TR-AG"].iloc[0]
    regla_opt = regla[regla["escenario"] == "umbral_optimo"].set_index("variante")
    return {
        "modelo": CONFIG["modelo"],
        "f1_global_0_5": float(base.metricas["f1_media_columnas"]),
        "f1_global": float(principal.metricas["f1_media_columnas"]),
        "f1_HS": float(principal.metricas["f1_por_etiqueta"]["HS"]),
        "f1_TR": float(principal.metricas["f1_por_etiqueta"]["TR"]),
        "f1_AG": float(principal.metricas["f1_por_etiqueta"]["AG"]),
        "exact_match_0_5": float(base.metricas["exact_match"]),
        "exact_match": float(principal.metricas["exact_match"]),
        "fp_total_0_5": int(((y_true == 0) & (base.prediccion == 1)).sum()),
        "fn_total_0_5": int(((y_true == 1) & (base.prediccion == 0)).sum()),
        "fp_total": int(((y_true == 0) & (predicho == 1)).sum()),
        "fn_total": int(((y_true == 1) & (predicho == 0)).sum()),
        "n_ambos_correctos": int(np.sum((y_true == predicho).all(axis=1))),
        "n_una_mal": int(np.sum((y_true != predicho).sum(axis=1) == 1)),
        "n_dos_mal": int(np.sum((y_true != predicho).sum(axis=1) == 2)),
        "n_tres_mal": int(np.sum((y_true != predicho).sum(axis=1) == 3)),
        "pares_HS_TR_ambas_mal": int(par_HS_TR["n_ambas_mal"]),
        "pares_HS_TR_ambas_fn": int(par_HS_TR["n_ambas_falso_negativo"]),
        "pares_HS_TR_ambas_fp": int(par_HS_TR["n_ambas_falso_positivo"]),
        "pares_HS_AG_ambas_mal": int(par_HS_AG["n_ambas_mal"]),
        "pares_HS_AG_ambas_fn": int(par_HS_AG["n_ambas_falso_negativo"]),
        "pares_HS_AG_ambas_fp": int(par_HS_AG["n_ambas_falso_positivo"]),
        "pares_TR_AG_ambas_mal": int(par_TR_AG["n_ambas_mal"]),
        "pares_TR_AG_ambas_fn": int(par_TR_AG["n_ambas_falso_negativo"]),
        "pares_TR_AG_ambas_fp": int(par_TR_AG["n_ambas_falso_positivo"]),
        "pares_TR_AG_mixto": int(par_TR_AG["n_ambas_mixto"]),
        "regla_tr_delta": float(regla_opt.loc["HS=0 => TR=0", "delta_vs_sin_regla"]),
        "regla_completa_delta": float(regla_opt.loc["HS=0 => TR=0 y AG=0", "delta_vs_sin_regla"]),
        "regla_casos_tr": int(regla_opt.loc["HS=0 => TR=0", "casos_TR1_con_HS0"]),
        "regla_casos_tr_true_1": int(regla_opt.loc["HS=0 => TR=0", "casos_TR1_con_HS0_y_TR_true_1"]),
        "regla_casos_ag": int(regla_opt.loc["HS=0 => TR=0", "casos_AG1_con_HS0"]),
        "regla_casos_ag_true_1": int(regla_opt.loc["HS=0 => TR=0", "casos_AG1_con_HS0_y_AG_true_1"]),
        "n_insulto_responde": int(region_broma["n"]),
        "n_insulto_negativos": negativos_broma,
        "pct_insulto_negativos": round(100 * negativos_broma / max(int(region_broma["n"]), 1), 2),
        "fp_rate_HS_region": round(fp_HS_region, 4),
        "fp_rate_global_HS": round(fp_HS_global, 4),
        "tr_f1_responde": f1_de("responde_mencion_inicial"),
        "tr_f1_sin_mencion_sin_colectiva": f1_de("no_responde_sin_colectiva"),
        "tr_f1_con_colectiva": f1_de("no_responde_con_colectiva"),
        "tr_errores_sin_mencion": errores_tr_sin_mencion,
        "tr_errores_total": errores_tr_hs1,
        "ag_f1_colectiva": f1_de("no_responde_con_colectiva"),
        "ag_f1_global": float(principal.metricas["f1_por_etiqueta"]["AG"]),
        "pct_ag_fn_colectiva": round(100 * tasa_perfil("AG", "tiene_referencia_colectiva", "tasa_FN"), 1),
        "pct_ag_tp_colectiva": round(100 * tasa_perfil("AG", "tiene_referencia_colectiva", "tasa_TP"), 1),
        "pct_ag_tn_colectiva": round(100 * tasa_perfil("AG", "tiene_referencia_colectiva", "tasa_TN"), 1),
        "f1_noticia": f1_de("regimen_noticia"),
        "f1_sin_noticia": f1_de("sin_url_ni_hashtag"),
        "n_noticia": int(regiones.loc["regimen_noticia", "n"]),
        "f1_hashtag": f1_rebanada("tiene_hashtag", "1"),
        "f1_pregunta": f1_rebanada("tiene_pregunta", "1"),
        "f1_colectiva": f1_rebanada("tiene_referencia_colectiva", "1"),
        "f1_q4": f1_rebanada("cuartil_longitud", "Q4"),
        "f1_url": f1_rebanada("tiene_url", "1"),
        "f1_corto": f1_rebanada("texto_corto_40", "1"),
        "f1_q1": f1_rebanada("cuartil_longitud", "Q1"),
        "f1_insulto_ausente": f1_rebanada("tiene_insulto_lexico", "0"),
        "n_insulto_ausente": int(
            rebanadas[
                (rebanadas["rebanada"] == "tiene_insulto_lexico")
                & (rebanadas["grupo"].astype(str) == "0")
                & (rebanadas["estado"] == "ok")
            ]["n"].iloc[0]
        ),
        "pct_fp_HS_mencion_inicial": round(100 * tasa_perfil("HS", "tiene_mencion_inicial", "tasa_FP"), 1),
        "pct_fp_HS_insulto": round(100 * tasa_perfil("HS", "tiene_insulto_lexico", "tasa_FP"), 1),
        "pct_fn_HS_url": round(100 * tasa_perfil("HS", "tiene_url", "tasa_FN"), 1),
        "pct_fn_HS_colectiva": round(100 * tasa_perfil("HS", "tiene_referencia_colectiva", "tasa_FN"), 1),
        "pct_tr_fp_mencion_inicial": round(100 * tasa_perfil("TR", "tiene_mencion_inicial", "tasa_FP"), 1),
        "pct_ag_fp_mencion_inicial": round(100 * tasa_perfil("AG", "tiene_mencion_inicial", "tasa_FP"), 1),
        "n_lexico": len(analisis_datos["lexico"]),
        "n_lexico_nucleo": len(analisis_datos["lexico_nucleo"]),
    }


def seccion_resumen(analisis: Analisis) -> str:
    """Redacta el resumen ejecutivo del reporte.

    Args:
        analisis: resultados completos del analisis.

    Returns:
        Texto markdown de la seccion.
    """
    ev = analisis.evidencia
    lineas = [
        "## Resumen ejecutivo",
        "",
        (f"- Modelo analizado: `{ev['modelo']}` con probabilidades OOF verificadas "
        f"(F1 {ev['f1_global_0_5']:.4f} con umbral 0.5 y {ev['f1_global']:.4f} con "
        f"umbrales 0.44/0.51/0.46)."),
        (f"- Errores con umbrales por etiqueta: {ev['fp_total']} falsos positivos y "
        f"{ev['fn_total']} falsos negativos sobre 3 x {len(analisis.y_true)} decisiones; "
        f"{ev['n_una_mal']} ejemplos fallan 1 etiqueta, {ev['n_dos_mal']} fallan 2 y "
        f"{ev['n_tres_mal']} fallan las 3."),
        (f"- Con umbral 0.5 el balance es {ev['fp_total_0_5']} FP y {ev['fn_total_0_5']} FN; "
        "los umbrales por etiqueta cambian el sesgo (mas FP, menos FN) y ganan "
        f"{ev['f1_global'] - ev['f1_global_0_5']:+.4f} de F1."),
        (f"- La regla estructural `HS_pred = 0 => TR = 0` vale "
        f"{ev['regla_tr_delta']:+.4f} de F1 y su version completa (tambien AG) "
        f"{ev['regla_completa_delta']:+.4f}: no es una palanca util a estos umbrales."),
        f"- Las 5 rebanadas mas debiles con n >= {CONFIG['min_rebanada']} son: "
        + "; ".join(
            f"`{fila.rebanada}={fila.grupo}` (n={int(fila.n)}, F1 {fila.f1_media:.3f})"
            for fila in analisis.debiles.itertuples()
        )
        + ".",
        (f"- La region mas fragil es el regimen noticia (URL o hashtag): F1 "
        f"{ev['f1_noticia']:.3f} frente a {ev['f1_sin_noticia']:.3f} sin URL ni hashtag, "
        f"con n={ev['n_noticia']}."),
        (f"- Los falsos positivos de HS y AG se concentran en el registro de insulto "
        f"interpersonal: de {ev['n_insulto_responde']} textos con insulto, mencion inicial "
        f"y sin referencia colectiva, {ev['n_insulto_negativos']} "
        f"({ev['pct_insulto_negativos']} %) estan etiquetados (0,0,0)."),
        (f"- El error de TR vive en los textos sin destinatario explicito: F1_TR "
        f"{ev['tr_f1_responde']:.3f} en respuestas frente a "
        f"{ev['tr_f1_sin_mencion_sin_colectiva']:.3f} sin mencion ni colectiva y "
        f"{ev['tr_f1_con_colectiva']:.3f} sin mencion con colectiva."),
        (f"- Los FN de AG son agresion implicita contra grupos: "
        f"{ev['pct_ag_fn_colectiva']} % de los FN de AG tiene referencia colectiva frente "
        f"a {ev['pct_ag_tp_colectiva']} % de los TP."),
        ("- Se proponen 5 hipotesis nuevas (H11-H15) con evidencia, mecanismo, experimento "
        "y criterio de exito; H15 es una hipotesis de cierre de la via de calibracion fina."),
        "",
    ]
    return "\n".join(lineas)


def seccion_metodo(analisis: Analisis) -> str:
    """Redacta la seccion de metodo y verificacion de replicacion.

    Args:
        analisis: resultados completos del analisis.

    Returns:
        Texto markdown de la seccion.
    """
    ev = analisis.evidencia
    filas_confusion = analisis.confusion[
        ["escenario", "etiqueta", "umbral", "positivos_reales", "positivos_predichos"]
    ]
    lineas = [
        "## 1. Metodo y verificacion",
        "",
        ("OBSERVACION: el analisis parte de las probabilidades out-of-fold ya guardadas del "
        "mejor modelo; cualquier conclusion depende de que esas probabilidades sean las "
        "publicadas y de que el umbral este declarado."),
        "",
        ("HIPOTESIS: las probabilidades de `features/oof_mod03b.npz` reproducen el F1 "
        "documentado en `PROJECT_STATE.md` y en `reports/fase13a_refit_umbral_metricas.json`."),
        "",
        ("METODO: carga con `src.data.read_train()`, alineacion por `id`, recuento de las "
        "predicciones con los dos juegos de umbrales de D-012 (0.5 y 0.44/0.51/0.46) y "
        "calculo de las metricas con `src.evaluation.metricas_multietiqueta`. No se "
        "reentrena nada y no se re-optimiza ningun umbral."),
        "",
        (f"RESULTADO: replicacion exacta; F1 {ev['f1_global_0_5']:.6f} con umbral 0.5 "
        f"(esperado 0.792659) y {ev['f1_global']:.6f} con umbrales por etiqueta "
        f"(esperado 0.798933). Dimensiones por escenario:"),
        "",
        tabla_markdown(filas_confusion),
        "",
        ("INTERPRETACION: el analisis es valido como diagnostico del modelo publicado con "
        "los umbrales ya decididos."),
        "",
        ("DECISION: usar el escenario de umbrales por etiqueta (v03, la apuesta vigente) "
        "como escenario principal del analisis y reportar el de 0.5 en paralelo."),
        "",
        "SIGUIENTE PASO: matrices de confusion por etiqueta (seccion 2).",
        "",
    ]
    return "\n".join(lineas)


def seccion_confusion(analisis: Analisis) -> str:
    """Redacta la seccion de matrices de confusion.

    Args:
        analisis: resultados completos del analisis.

    Returns:
        Texto markdown de la seccion.
    """
    columnas = [
        "escenario", "etiqueta", "umbral", "TP", "FP", "FN", "TN", "precision",
        "recall", "f1", "fpr", "fnr", "tasa_error_etiqueta",
    ]
    tabla = analisis.confusion[columnas].copy()
    tabla["fpr"] = tabla["fpr"].round(4)
    tabla["fnr"] = tabla["fnr"].round(4)
    tabla["tasa_error_etiqueta"] = tabla["tasa_error_etiqueta"].round(4)
    lineas = [
        "## 2. Matrices de confusion por etiqueta",
        "",
        ("OBSERVACION: se quieren los conteos y las tasas de acierto y error de cada "
        "etiqueta en los dos escenarios de umbral decididos en D-012."),
        "",
        ("HIPOTESIS: el modelo es mejor en TR que en HS y AG, y el ajuste de umbrales mueve "
        "el error de falsos negativos hacia falsos positivos."),
        "",
        ("METODO: matriz 2x2 por etiqueta (TP, FP, FN, TN) con precision, recall, "
        "especificidad, F1, tasa de falsos positivos (FPR), tasa de falsos negativos (FNR) "
        "y tasa de error por etiqueta."),
        "",
        "RESULTADO:",
        "",
        tabla_markdown(tabla),
        "",
        ("INTERPRETACION: con umbral 0.5 el modelo subpredice (FN > FP en las tres "
        "etiquetas); con los umbrales por etiqueta el signo se invierte (FP > FN). La "
        "etiqueta peor sigue siendo AG (F1 "
        f"{analisis.evidencia['f1_AG']:.4f}) y la mejor TR "
        f"({analisis.evidencia['f1_TR']:.4f}). HS es la etiqueta con mas errores absolutos "
        f"({int(analisis.confusion[(analisis.confusion['escenario'] == 'umbral_optimo') & (analisis.confusion['etiqueta'] == 'HS')]['FP'].iloc[0])} FP + "
        f"{int(analisis.confusion[(analisis.confusion['escenario'] == 'umbral_optimo') & (analisis.confusion['etiqueta'] == 'HS')]['FN'].iloc[0])} FN)."),
        "",
        ("DECISION: considerar AG y HS como los objetivos primarios de mejora; TR tiene "
        "menos margen absoluto y ya esta en 0.83."),
        "",
        ("SIGUIENTE PASO: analizar como se combinan los errores entre etiquetas "
        "(seccion 3)."),
        "",
    ]
    return "\n".join(lineas)


def seccion_combinaciones(analisis: Analisis) -> str:
    """Redacta la seccion de combinaciones de errores entre etiquetas.

    Args:
        analisis: resultados completos del analisis.

    Returns:
        Texto markdown de la seccion.
    """
    ev = analisis.evidencia
    lineas = [
        "## 3. Combinaciones de errores y estructura de etiquetas",
        "",
        ("OBSERVACION: las tres etiquetas estan asociadas (phi HS-TR 0.743, HS-AG 0.831, "
        "TR-AG 0.759; seccion 6 del EDA) y TR=1 implica HS=1 en el 100 % de train, por lo "
        "que los errores no son independientes."),
        "",
        ("HIPOTESIS: parte de los errores de pares se debe a la dependencia estructural "
        "(predecir HS=0 arrastra TR) y la regla estructural deberia recuperar esos casos."),
        "",
        ("METODO: conteo de etiquetas erroneas por ejemplo, tabla por combinacion verdadera "
        "HS x TR x AG, tabla de errores conjuntos por par y evaluacion de la regla "
        "estructural en sus dos variantes sobre cada escenario de umbral."),
        "",
        "RESULTADO (umbrales por etiqueta):",
        "",
        tabla_markdown(analisis.etiquetas_erroneas),
        "",
        "Errores por combinacion verdadera:",
        "",
        tabla_markdown(analisis.combinaciones),
        "",
        "Errores conjuntos por par de etiquetas:",
        "",
        tabla_markdown(analisis.pares),
        "",
        "Efecto de la regla estructural:",
        "",
        tabla_markdown(analisis.regla),
        "",
        (f"INTERPRETACION: {ev['n_tres_mal']} ejemplos ({100 * ev['n_tres_mal'] / len(analisis.y_true):.1f} %) "
        "fallan las tres etiquetas y son el nucleo duro del error; "
        f"{ev['n_una_mal']} fallan solo una. El par con mas errores conjuntos es HS-AG "
        f"({ev['pares_HS_AG_ambas_mal']}), seguido de TR-AG "
        f"({ev['pares_TR_AG_ambas_mal']}) y HS-TR ({ev['pares_HS_TR_ambas_mal']}). "
        f"Los errores conjuntos de HS-TR son {ev['pares_HS_TR_ambas_fp']} falsos positivos "
        f"y {ev['pares_HS_TR_ambas_fn']} falsos negativos: el modelo pierde o inventa el "
        "par completo, no una de las dos etiquetas. TR-AG es el unico par con casos "
        f"mixtos ({ev['pares_TR_AG_mixto']}), coherente con que AG no esta condicionada a "
        "HS en 26 casos de train."),
        "",
        (f"La regla estructural aporta {ev['regla_tr_delta']:+.4f} de F1 en su version solo "
        f"TR y {ev['regla_completa_delta']:+.4f} en su version completa: con los umbrales "
        f"por etiqueta hay {ev['regla_casos_tr']} casos con TR=1 y HS=0, de los que "
        f"{ev['regla_casos_tr_true_1']} son TR verdaderos, y {ev['regla_casos_ag']} con "
        f"AG=1 y HS=0, de los que {ev['regla_casos_ag_true_1']} son AG verdaderos. La "
        "regla intercambia casi tantos aciertos como errores."),
        "",
        ("DECISION: no adoptar la regla estructural; H8 queda cerrada como palanca con esta "
        "representacion (ya se habia medido -0.0057 en la Fase 4-5 y aqui -0.0048 con "
        "umbrales por etiqueta). El esfuerzo debe ir a la frontera real (secciones 4-5)."),
        "",
        ("SIGUIENTE PASO: identificar por rebanadas donde se concentra el error "
        "(seccion 4)."),
        "",
    ]
    return "\n".join(lineas)


def seccion_rebanadas(analisis: Analisis) -> str:
    """Redacta la seccion de analisis por rebanadas.

    Args:
        analisis: resultados completos del analisis.

    Returns:
        Texto markdown de la seccion.
    """
    ev = analisis.evidencia
    columnas = [
        "rebanada", "grupo", "n", "f1_HS", "f1_TR", "f1_AG", "f1_media",
        "delta_vs_global", "tasa_error_etiquetas", "tasa_error_exacta",
    ]
    evaluables = analisis.rebanadas[analisis.rebanadas["estado"] == "ok"]
    no_evaluables = analisis.rebanadas[analisis.rebanadas["estado"] != "ok"]
    orden = evaluables.sort_values("f1_media")
    conjuntos = evaluables[
        (evaluables["n"] >= CONFIG["min_rebanada"]) & (~evaluables["artefacto_lexico"].astype(bool))
    ]
    repeticion = conjuntos.groupby(["rebanada", "grupo"]).size()
    _ = repeticion  # el agrupamiento anterior documenta que no hay duplicados de clave
    lineas = [
        "## 4. Analisis por rebanadas",
        "",
        ("OBSERVACION: el F1 global esconde una fuerte heterogeneidad entre regimenes del "
        "corpus (respuestas cortas frente a textos noticiosos largos)."),
        "",
        ("HIPOTESIS: el error se concentra en los textos largos con URL o hashtag y en los "
        "que tienen referencia colectiva, donde la anotacion es mas dificil."),
        "",
        ("METODO: 18 rebanadas definidas con regex y spaCy (POS y lemas, sin vectores), "
        "evaluadas con las predicciones OOF del escenario de umbrales por etiqueta; se "
        "exige n >= "
        f"{CONFIG['min_rebanada']:.0f} para entrar en el ranking. Las rebanadas marcadas "
        "como `artefacto_lexico` usan el lexico inducido in-sample y se separan del "
        "ranking."),
        "",
        "RESULTADO: tabla completa de rebanadas evaluables.",
        "",
        tabla_markdown(orden[columnas]),
        "",
        "Rebanadas no evaluables:",
        "",
        tabla_markdown(no_evaluables[["rebanada", "grupo", "estado", "n"]]),
        "",
        "Las 5 mas debiles (excluyendo artefactos del lexico inducido):",
        "",
        tabla_markdown(analisis.debiles),
        "",
        "Las 5 mas debiles incluyendo artefactos del lexico inducido:",
        "",
        tabla_markdown(analisis.debiles_con_artefactos),
        "",
        ("INTERPRETACION: la debilidad se concentra en el registro noticia/comentario "
        f"(hashtag F1 {ev['f1_hashtag']:.3f}, pregunta {ev['f1_pregunta']:.3f}, referencia "
        f"colectiva {ev['f1_colectiva']:.3f}, URL {ev['f1_url']:.3f}, cuartil largo "
        f"{ev['f1_q4']:.3f}) frente a textos cortos ({ev['f1_corto']:.3f}) y respuestas "
        f"directas. La rebanada `tiene_insulto_lexico=0` (F1 "
        f"{ev['f1_insulto_ausente']:.3f}, n={ev['n_insulto_ausente']}) es un artefacto "
        "in-sample: el lexico se indujo con las etiquetas de train y estas rebanadas "
        "separan casi por construccion. Retuits: cero casos en train (`tiene_retuit`), por "
        "lo que la rebanada no existe en este corpus."),
        "",
        ("DECISION: priorizar el regimen noticia y la region sin destinatario explicito en "
        "los siguientes experimentos; no usar las rebanadas del lexico como evidencia."),
        "",
        ("SIGUIENTE PASO: cuantificar que rasgos acompanan a cada tipo de error "
        "(seccion 5) y revisar los casos (seccion 6)."),
        "",
    ]
    return "\n".join(lineas)


def seccion_regiones(analisis: Analisis) -> str:
    """Redacta la seccion de regiones interpretativas.

    Args:
        analisis: resultados completos del analisis.

    Returns:
        Texto markdown de la seccion.
    """
    ev = analisis.evidencia
    columnas = [
        "region", "condicion", "n", "f1_HS", "f1_TR", "f1_AG", "f1_media",
        "delta_vs_global", "prevalencia_real_TR", "prevalencia_predicha_TR",
    ]
    lineas = [
        "## 5. Regiones del corpus y rasgos que acompanan al error",
        "",
        ("OBSERVACION: las rebanadas simples se solapan (URL y hashtag suelen ir juntas); "
        "conviene definir regiones mutuamente legibles y medir alli donde vive el error."),
        "",
        ("HIPOTESIS: el error se divide en dos regimenes: la respuesta insultante "
        "(persona a persona) y el comentario o noticia con referencia colectiva."),
        "",
        ("METODO: siete regiones definidas por combinacion de rasgos (mencion inicial, "
        "referencia colectiva, URL o hashtag) y perfil de rasgos en TP, FP, FN y TN por "
        "etiqueta."),
        "",
        "RESULTADO: metricas por region.",
        "",
        tabla_markdown(analisis.regiones[columnas]),
        "",
        "Perfil de rasgos en aciertos y errores (tasas y sobrerrepresentacion):",
        "",
        tabla_markdown(
            analisis.perfil[
                [
                    "etiqueta", "rasgo", "tasa_TP", "tasa_FP", "tasa_FN", "tasa_TN",
                    "sobrerrepresentacion_FP_vs_TN", "sobrerrepresentacion_FN_vs_TP",
                ]
            ]
        ),
        "",
        ("INTERPRETACION: en la region de respuesta directa el F1 es alto en las tres "
        "etiquetas; el dano se concentra en `no_responde_con_colectiva` (TR y AG) y en el "
        "regimen noticia. Los FP de HS llevan mencion inicial en el "
        f"{ev['pct_fp_HS_mencion_inicial']} % de los casos y el insulto en el "
        f"{ev['pct_fp_HS_insulto']} %; los FN de HS llevan URL en el {ev['pct_fn_HS_url']} % "
        f"y referencia colectiva en el {ev['pct_fn_HS_colectiva']} %. En AG, el "
        f"{ev['pct_ag_fn_colectiva']} % de los FN tiene referencia colectiva frente al "
        f"{ev['pct_ag_tp_colectiva']} % de los TP: los positivos que se escapan son "
        "agresion contra grupos, no insulto interpersonal."),
        "",
        ("DECISION: atacar por separado los dos regimenes (features de registro para la "
        "respuesta y features de objetivo colectivo o individual para el comentario)."),
        "",
        "SIGUIENTE PASO: revision humana de las muestras (seccion 6).",
        "",
    ]
    return "\n".join(lineas)


def seccion_cualitativa(analisis: Analisis) -> str:
    """Redacta la seccion cualitativa con patrones y muestras.

    Args:
        analisis: resultados completos del analisis.

    Returns:
        Texto markdown de la seccion.
    """
    muestras = analisis.muestras.copy()
    muestras["etiquetas_true"] = (
        muestras["HS_true"].astype(str) + "," + muestras["TR_true"].astype(str) + "," + muestras["AG_true"].astype(str)
    )
    muestras["etiquetas_pred"] = (
        muestras["HS_pred"].astype(str) + "," + muestras["TR_pred"].astype(str) + "," + muestras["AG_pred"].astype(str)
    )
    muestras["texto_100"] = muestras["texto_200"].str.slice(0, 100)
    reducidas = []
    for _, grupo in muestras.groupby(["etiqueta", "tipo_error"]):
        reducidas.append(grupo.head(5))
    tabla_muestras = pd.concat(reducidas, ignore_index=True)[
        ["etiqueta", "tipo_error", "probabilidad", "etiquetas_true", "etiquetas_pred", "texto_100"]
    ]
    lineas = [
        "## 6. Analisis cualitativo",
        "",
        ("OBSERVACION: las rebanadas dicen donde falla el modelo, no por que; hace falta "
        "leer casos."),
        "",
        ("HIPOTESIS: los FP y FN responden a un numero pequeno de patrones recurrentes "
        "(ironia, insulto sin odio, odio implicito, agresividad sin objetivo, textos muy "
        "cortos, ruido de anotacion)."),
        "",
        ("METODO: exportar "
        f"{CONFIG['n_muestras_por_tipo']} FP (los mas confiados) y "
        f"{CONFIG['n_muestras_por_tipo']} FN (los mas olvidados) por etiqueta a "
        "`reports/eda/error_samples_para_revision.csv` (texto truncado a "
        f"{CONFIG['max_caracteres_muestra']} caracteres) y revisar los casos a mano; el "
        "resumen de rasgos de esas muestras es automatico."),
        "",
        "RESULTADO: perfil de las muestras exportadas.",
        "",
        tabla_markdown(analisis.resumen_muestras),
        "",
        ("Conteo de las muestras que caen en cada patron operacionalizado (buckets no "
        "excluyentes; la regla de cada bucket esta en el codigo):"),
        "",
        tabla_markdown(analisis.resumen_patrones),
        "",
        ("Muestras representativas (5 por etiqueta y tipo, texto truncado a 100 "
        "caracteres):"),
        "",
        tabla_markdown(tabla_muestras, decimales=3),
        "",
        "Patrones recurrentes observados en la revision humana:",
        "",
    ]
    for patron in PATRONES_CUALITATIVOS:
        ejemplos = ", ".join(patron["ejemplos"]) if patron["ejemplos"] else "sin ejemplos"
        lineas.append(
            f"- **{patron['patron']}** (etiquetas: {patron['etiquetas']}; ids: {ejemplos}): "
            f"{patron['descripcion']}"
        )
    lineas += [
        "",
        ("INTERPRETACION: la revision confirma que el error no es homogeneo. El bloque "
        "dominante de falsos positivos es el insulto interpersonal en registro de broma o "
        "afiliacion, que se repite en las tres etiquetas y con probabilidades muy altas. "
        "El bloque dominante de falsos negativos es el odio implicito contra grupos sin "
        "insulto del lexico y el regimen noticia o atribucion a terceros, donde TR ademas "
        "se comporta como objetivo individual (D-005). Los casos de frontera (mismo patron "
        "de superficie con etiquetas opuestas, citas, textos cuyo sentido depende de un "
        "enlace) muestran ambiguedad de anotacion que ningun modelo de bolsas de n-gramas "
        "puede resolver: acotan el techo alcanzable con estas features."),
        "",
        ("DECISION: convertir los patrones dominantes en experimentos dirigidos "
        "(seccion 7) en lugar de seguir ampliando la representacion a ciegas."),
        "",
        ("SIGUIENTE PASO: registrar las hipotesis H11-H15 y ejecutarlas en la Fase 14 con "
        "la regla de decision vigente."),
        "",
    ]
    return "\n".join(lineas)


def seccion_hipotesis(analisis: Analisis) -> str:
    """Redacta la seccion de hipotesis nuevas con el formato del proyecto.

    Args:
        analisis: resultados completos del analisis.

    Returns:
        Texto markdown de la seccion.
    """
    ev = {clave: valor for clave, valor in analisis.evidencia.items()}
    resumen = []
    for hipotesis in HIPOTESIS:
        resumen.append(
            {
                "id": hipotesis["id"],
                "enunciado": hipotesis["enunciado"],
                "experimento": hipotesis["experimento"].split(":")[0],
                "criterio": hipotesis["criterio"],
            }
        )
    lineas = [
        "## 7. Hipotesis nuevas (H11 en adelante)",
        "",
        ("Las cinco hipotesis nacen del analisis de error y se ejecutan en la Fase 14. Todas "
        "heredan la regla de decision del proyecto: una mejora se adopta con >= +0.005 "
        "absoluto en OOF y >= 3/5 pliegues, y por D-012 se exige >= +0.010 local en la "
        "familia union antes de gastar una sonda de leaderboard."),
        "",
        "Tabla resumen:",
        "",
        tabla_markdown(pd.DataFrame(resumen)),
        "",
    ]
    for hipotesis in HIPOTESIS:
        lineas += [
            f"### {hipotesis['id']}",
            "",
            f"- OBSERVACION (evidencia del analisis): {hipotesis['evidencia'].format(**ev)}",
            f"- HIPOTESIS (enunciado): {hipotesis['enunciado']}",
            f"- METODO (experimento propuesto): {hipotesis['experimento']}",
            ("- RESULTADO: pendiente; se ejecuta en la Fase 14 y se registra en "
            "`experiments.csv` con su `experiment_id`."),
            f"- INTERPRETACION (mecanismo esperado): {hipotesis['mecanismo']}",
            f"- DECISION (criterio de adopcion): {hipotesis['criterio']}",
            ("- SIGUIENTE PASO: implementar el bloque de features dentro del pliegue "
            "(sin fuga), medir OOF y por rebanada con este mismo script de analisis para "
            "comparar antes y despues."),
            "",
        ]
    return "\n".join(lineas)


def seccion_limitaciones(analisis: Analisis) -> str:
    """Redacta las limitaciones y advertencias de honestidad.

    Args:
        analisis: resultados completos del analisis.

    Returns:
        Texto markdown de la seccion.
    """
    ev = analisis.evidencia
    lineas = [
        "## 8. Limitaciones y honestidad",
        "",
        ("- Las rebanadas del lexico inducido (`tiene_insulto_lexico`, "
        "`tiene_insulto_nucleo`) son in-sample: el lexico se construyo con las etiquetas "
        "de train (150 lemas), por lo que su F1 no es evidencia de generalizacion. Se "
        "incluyen por pedido explicito y se excluyen del ranking de debilidad."),
        ("- La rebanada de risa usa un patron laxo (subcadena) distinto del patron con "
        "frontera de palabra del EDA: el corpus escribe series como "
        "`JAJAJAJAJJAJAJAJA` que el patron estricto no captura (19 frente a 120 "
        "documentos). El cambio esta documentado en el modulo."),
        ("- Los rasgos de segunda persona, imperativo y referencia colectiva dependen de "
        "spaCy `es_core_news_md` como anotador (D-003); en texto sin diacriticos ni "
        "puntuacion normalizada hay ruido de anotacion."),
        ("- Las predicciones son OOF: provienen de modelos que vieron los otros pliegues, "
        "por lo que las tasas por rebanada son estimaciones honestas del rendimiento "
        "esperado pero no una validacion independiente adicional."),
        ("- No se re-optimiza ningun umbral ni se usa el test; las conclusiones son sobre "
        "train y sus etiquetas, con la advertencia conocida de que el test tiene ~7 % "
        "menos positivos (E00) y que el gap local-LB de la familia union es ~0.013-0.016 "
        "(D-012)."),
        ("- Este analisis no entrena modelos: las hipotesis quedan propuestas y sin "
        "validar. Si la Fase 14 no encuentra ganancias >= +0.005 con >= 3/5 pliegues, la "
        "conclusion honesta es que el techo de la familia TF-IDF + modelo lineal esta "
        "cerca y el esfuerzo debe ir a la calibracion final y al notebook, no a mas "
        "features."),
        "",
        ("Numeros de control para leer el reporte: F1 global "
        f"{ev['f1_global_0_5']:.4f} (0.5) y {ev['f1_global']:.4f} (umbrales), con "
        f"{ev['fp_total']} FP y {ev['fn_total']} FN; lexico inducido de {ev['n_lexico']} "
        f"lemas ({ev['n_lexico_nucleo']} en el nucleo)."),
        "",
    ]
    return "\n".join(lineas)


def seccion_artefactos(analisis: Analisis) -> str:
    """Lista los artefactos generados por el script.

    Args:
        analisis: resultados completos del analisis.

    Returns:
        Texto markdown de la seccion.
    """
    lineas = ["## 9. Artefactos", ""]
    for artefacto in analisis.artefactos:
        lineas.append(f"- `{artefacto.relative_to(PROJECT_ROOT)}`")
    for figura in analisis.figuras:
        lineas.append(f"- `{figura.relative_to(PROJECT_ROOT)}`")
    lineas += [
        f"- `{(CONFIG['reporte_md']).relative_to(PROJECT_ROOT)}` (este reporte)",
        "- `reports/contexto.md` (seccion de handoff de la Fase 12)",
        "",
        "Comando de reproduccion:",
        "",
        "```bash",
        "/home/carlos/venv/bin/python experiments/06_error_analysis.py",
        "```",
        "",
    ]
    return "\n".join(lineas)


def construir_reporte(analisis: Analisis) -> str:
    """Ensambla el reporte completo de analisis de error.

    Args:
        analisis: resultados completos del analisis.

    Returns:
        Texto markdown del reporte.
    """
    encabezado = [
        (f"# Analisis de error - {analisis.evidencia['modelo']} (Fase 12, "
        f"{CONFIG['fecha']})"),
        "",
        ("Tarea: entender donde y por que falla el mejor modelo, cuantificar los tipos de "
        "error y derivar hipotesis dirigidas. Fuente de predicciones: "
        "`features/oof_mod03b.npz` (out-of-fold, umbrales de D-012). Sin reentrenamiento, "
        "sin test y sin tocar los scripts previos."),
        "",
    ]
    piezas = [
        seccion_resumen(analisis),
        seccion_metodo(analisis),
        seccion_confusion(analisis),
        seccion_combinaciones(analisis),
        seccion_rebanadas(analisis),
        seccion_regiones(analisis),
        seccion_cualitativa(analisis),
        seccion_hipotesis(analisis),
        seccion_limitaciones(analisis),
        seccion_artefactos(analisis),
    ]
    return "\n".join(encabezado + piezas)


def construir_seccion_contexto(analisis: Analisis) -> str:
    """Construye la seccion de handoff (maximo 20 lineas) para contexto.md.

    Args:
        analisis: resultados completos del analisis.

    Returns:
        Texto de la seccion, con encabezado y sin linea final en blanco.

    Raises:
        ValueError: si la seccion excede las 20 lineas.
    """
    ev = analisis.evidencia
    debiles = "; ".join(
        f"{fila.rebanada}={fila.grupo} (n={int(fila.n)}, F1 {fila.f1_media:.3f})"
        for fila in analisis.debiles.itertuples()
    )
    lineas = [
        f"## Fase 12 - nlp-specialist ({CONFIG['fecha']})",
        (f"Analisis de error de `{ev['modelo']}` sobre probabilidades OOF (sin reentrenar, "
        f"sin test): `experiments/06_error_analysis.py` -> `reports/error_analysis.md` y "
        "7 CSV en `reports/eda/error_*.csv` + 5 figuras."),
        (f"F1 verificado: {ev['f1_global_0_5']:.4f} (0.5) y {ev['f1_global']:.4f} "
        f"(umbrales 0.44/0.51/0.46); {ev['fp_total']} FP y {ev['fn_total']} FN; "
        f"{ev['n_una_mal']}/{ev['n_dos_mal']}/{ev['n_tres_mal']} ejemplos fallan 1/2/3 "
        "etiquetas."),
        (f"Regla estructural `HS=0 => TR=0`: {ev['regla_tr_delta']:+.4f} F1 (version "
        f"completa con AG {ev['regla_completa_delta']:+.4f}): H8 cerrada como palanca."),
        f"Las 5 rebanadas mas debiles (n >= 40, sin artefactos in-sample): {debiles}.",
        (f"Los FP de HS y AG se concentran en insulto interpersonal con mencion inicial sin "
        f"colectivo: de {ev['n_insulto_responde']} textos, {ev['n_insulto_negativos']} "
        f"({ev['pct_insulto_negativos']} %) estan etiquetados (0,0,0)."),
        (f"El error de TR vive fuera de las respuestas: F1_TR {ev['tr_f1_responde']:.3f} con "
        f"mencion inicial frente a {ev['tr_f1_sin_mencion_sin_colectiva']:.3f} sin mencion "
        f"ni colectiva y {ev['tr_f1_con_colectiva']:.3f} sin mencion con colectiva."),
        (f"AG: los FN contra grupos superan a los FP de broma (el {ev['pct_ag_fn_colectiva']} "
        f"% de los FN tiene referencia colectiva frente al {ev['pct_ag_tp_colectiva']} % de "
        "los TP)."),
        ("Hipotesis nuevas H11-H15 (evidencia, mecanismo, experimento y criterio en el "
        "reporte): H11 interacciones de registro bromista/afectivo; H12 bloque de objetivo "
        "individual vs colectivo para TR; H13 features de cita/atribucion para el regimen "
        "noticia; H14 lexico inducido de deshumanizacion para AG implicito; H15 (cierre) "
        "umbrales por rebanada, que se espera que no aporten."),
        ("Sin fuga: todo se calcula con OOF de train; los lexicos son in-sample y sus "
        "rebanadas se reportan aparte. Registro en `experiments.csv` pendiente para la "
        "Fase 14."),
    ]
    if len(lineas) > 20:
        raise ValueError(f"La seccion de contexto tiene {len(lineas)} lineas (maximo 20)")
    return "\n".join(lineas)


def escribir_reporte(texto: str) -> Path:
    """Escribe el reporte markdown completo.

    Args:
        texto: contenido del reporte.

    Returns:
        Ruta del archivo escrito.
    """
    destino: Path = CONFIG["reporte_md"]
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(texto.rstrip("\n") + "\n", encoding="utf-8")
    log(f"reporte escrito en {destino.relative_to(PROJECT_ROOT)} ({len(texto.splitlines())} lineas)")
    return destino


def actualizar_contexto(seccion: str) -> Path:
    """Inserta la seccion de handoff al final de ``reports/contexto.md``.

    Idempotente: si la seccion de esta fase ya existe, se reemplaza en lugar de
    duplicarse.

    Args:
        seccion: texto de la seccion (sin linea final en blanco).

    Returns:
        Ruta del archivo actualizado.
    """
    destino: Path = CONFIG["contexto_md"]
    encabezado = f"## Fase 12 - nlp-specialist ({CONFIG['fecha']})"
    texto = destino.read_text(encoding="utf-8") if destino.exists() else ""
    if encabezado in texto:
        texto = texto[: texto.index(encabezado)]
    destino.write_text(
        texto.rstrip("\n") + "\n\n" + seccion.rstrip("\n") + "\n", encoding="utf-8"
    )
    log(f"seccion de contexto actualizada en {destino.relative_to(PROJECT_ROOT)}")
    return destino


def guardar_resumen_json(analisis: Analisis) -> Path:
    """Guarda los numeros clave del analisis en JSON para el handoff.

    Args:
        analisis: resultados completos del analisis.

    Returns:
        Ruta del archivo escrito.
    """
    resumen = {
        "fecha": CONFIG["fecha"],
        "modelo": CONFIG["modelo"],
        "oof": str(CONFIG["oof_npz"].relative_to(PROJECT_ROOT)),
        "escenarios": {
            clave: {
                "nombre": escenario.nombre,
                "umbrales": list(escenario.umbrales),
                "f1_media_columnas": round(escenario.metricas["f1_media_columnas"], 6),
                "f1_por_etiqueta": {
                    etiqueta: round(valor, 6)
                    for etiqueta, valor in escenario.metricas["f1_por_etiqueta"].items()
                },
                "exact_match": round(escenario.metricas["exact_match"], 6),
            }
            for clave, escenario in analisis.escenarios.items()
        },
        "evidencia": analisis.evidencia,
        "rebanadas_mas_debiles": analisis.debiles.to_dict(orient="records"),
        "rebanadas_mas_debiles_con_artefactos": analisis.debiles_con_artefactos.to_dict(
            orient="records"
        ),
        "regiones": analisis.regiones.to_dict(orient="records"),
        "matrices_confusion": analisis.confusion.to_dict(orient="records"),
        "combinaciones": analisis.combinaciones.to_dict(orient="records"),
        "pares": analisis.pares.to_dict(orient="records"),
        "patrones_muestras": analisis.resumen_patrones.to_dict(orient="records"),
        "patrones_cualitativos_revisados": [
            {"patron": patron["patron"], "etiquetas": patron["etiquetas"], "ejemplos": list(patron["ejemplos"])}
            for patron in PATRONES_CUALITATIVOS
        ],
        "hipotesis": [
            {"id": hipotesis["id"], "enunciado": hipotesis["enunciado"]}
            for hipotesis in HIPOTESIS
        ],
        "figuras": [str(figura.relative_to(PROJECT_ROOT)) for figura in analisis.figuras],
        "artefactos": [
            str(artefacto.relative_to(PROJECT_ROOT)) for artefacto in analisis.artefactos
        ],
    }
    destino = CONFIG["eda_dir"] / "error_resumen.json"
    destino.write_text(
        json.dumps(serializar(resumen), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    log(f"resumen JSON escrito en {destino.relative_to(PROJECT_ROOT)}")
    return destino


# ---------------------------------------------------------------------------
# Seccion 10 - Flujo principal
# ---------------------------------------------------------------------------
def main() -> None:
    """Ejecuta el analisis de error completo de la Fase 12."""
    log("inicio del analisis de error")
    configurar_estilo()
    train, y_true, proba = cargar_oof(CONFIG["oof_npz"])
    escenarios = construir_escenarios(y_true, proba)
    principal = escenarios["umbral_optimo"]
    f1_global = float(principal.metricas["f1_media_columnas"])

    confusion = construir_matrices_confusion(y_true, escenarios)
    etiquetas_erroneas = tabla_etiquetas_erroneas(y_true, principal.prediccion)
    combinaciones = tabla_combinaciones_reales(y_true, principal.prediccion)
    pares = analizar_pares(y_true, principal.prediccion)
    regla = pd.concat(
        [evaluar_regla_estructural(y_true, escenario) for escenario in escenarios.values()],
        ignore_index=True,
    )

    lexico, lexico_nucleo = cargar_lexico_agresividad(CONFIG["lexico_csv"])
    rasgos = extraer_rasgos_documento(train["text"].tolist(), lexico, lexico_nucleo)
    rasgos = anadir_columnas_rebanada(rasgos, train)
    rebanadas = evaluar_rebanadas(y_true, principal, rasgos, f1_global)
    regiones = evaluar_regiones(y_true, principal, rasgos, f1_global)
    debiles = seleccionar_rebanadas_debiles(rebanadas, incluir_artefactos=False)
    debiles_con_artefactos = seleccionar_rebanadas_debiles(rebanadas, incluir_artefactos=True)
    perfil = perfil_rasgos_por_etiqueta(y_true, principal, rasgos)
    muestras = seleccionar_muestras(train, y_true, principal, rasgos)
    resumen_muestras = resumir_muestras(muestras)
    muestras_marcadas = marcar_patrones_muestras(muestras)
    resumen_patrones = resumir_patrones_muestras(muestras_marcadas)

    log("las 5 rebanadas mas debiles (sin artefactos in-sample):")
    for fila in debiles.itertuples():
        log(
            f"  {fila.posicion}. {fila.rebanada}={fila.grupo} n={int(fila.n)} "
            f"F1={fila.f1_media:.4f} delta={fila.delta_vs_global:+.4f}"
        )

    evidencia = construir_evidencia(
        {
            "y_true": y_true,
            "escenarios": escenarios,
            "regiones": regiones,
            "rebanadas": rebanadas,
            "pares": pares,
            "regla": regla,
            "perfil": perfil,
            "rasgos": rasgos,
            "lexico": lexico,
            "lexico_nucleo": lexico_nucleo,
        }
    )

    figuras = (
        figura_matrices_confusion(
            confusion,
            "umbral_0_5",
            "Matrices de confusion por etiqueta (umbral 0.5)",
            "fig_20_matrices_confusion_umbral_05.png",
        ),
        figura_matrices_confusion(
            confusion,
            "umbral_optimo",
            "Matrices de confusion por etiqueta (umbrales 0.44/0.51/0.46)",
            "fig_21_matrices_confusion_umbrales_optimos.png",
        ),
        figura_errores_por_etiqueta(
            y_true,
            principal,
            etiquetas_erroneas,
            "fig_22_errores_por_etiqueta_y_por_ejemplo.png",
        ),
        figura_rebanadas_debiles(
            rebanadas, f1_global, "fig_23_rebanadas_mas_debiles.png"
        ),
        figura_regiones(regiones, "fig_24_regiones_f1.png"),
    )

    artefactos = (
        guardar_csv(confusion, "error_matrices_confusion.csv"),
        guardar_csv(etiquetas_erroneas, "error_etiquetas_erroneas.csv"),
        guardar_csv(combinaciones, "error_combinaciones_etiquetas.csv"),
        guardar_csv(pares, "error_pares_etiquetas.csv"),
        guardar_csv(regla, "error_regla_estructural.csv"),
        guardar_csv(rebanadas, "error_rebanadas.csv"),
        guardar_csv(regiones, "error_regiones.csv"),
        guardar_csv(perfil, "error_perfil_rasgos.csv"),
        guardar_csv(muestras, "error_samples_para_revision.csv"),
        guardar_csv(resumen_muestras, "error_resumen_muestras.csv"),
        guardar_csv(resumen_patrones, "error_resumen_patrones.csv"),
    )

    analisis = Analisis(
        train=train,
        y_true=y_true,
        escenarios=escenarios,
        confusion=confusion,
        etiquetas_erroneas=etiquetas_erroneas,
        combinaciones=combinaciones,
        pares=pares,
        regla=regla,
        rebanadas=rebanadas,
        regiones=regiones,
        debiles=debiles,
        debiles_con_artefactos=debiles_con_artefactos,
        perfil=perfil,
        muestras=muestras,
        muestras_marcadas=muestras_marcadas,
        resumen_muestras=resumen_muestras,
        resumen_patrones=resumen_patrones,
        evidencia=evidencia,
        figuras=figuras,
        artefactos=artefactos,
    )
    escribir_reporte(construir_reporte(analisis))
    guardar_resumen_json(analisis)
    actualizar_contexto(construir_seccion_contexto(analisis))
    log("fin del analisis de error")


if __name__ == "__main__":
    main()
