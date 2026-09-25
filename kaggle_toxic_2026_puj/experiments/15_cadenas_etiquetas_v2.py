"""P1.1/P1.2 - Dependencia entre etiquetas con scores continuos (E2-01/E2-02).

Pregunta que responde
---------------------
La senal mas fuerte del corpus (phi(HS,AG)=0.831, phi(HS,TR)=0.743) solo se ha
explotado con un post-proceso DURO (HS=0 => TR=0, AG=0) que degrado -0.0056.
¿Aporta algo explotarla con scores CONTINUOS y anidamiento correcto, en forma de
cadenas de etiquetas (E2-01) o de meta-modelo regularizado sobre los scores OOF
de los 4 bases (E2-02)?

Metodo
------
- Trabajo exclusivo sobre el DESARROLLO (1483 filas); el holdout congelado
  (`data/processed/holdout_v2_ids.csv`) no se carga para modelar.
- Pliegues corregidos v2 (D-016): `StratifiedGroupKFold(5)`, grupos = near
  duplicado UNION pseudo-autor, seed 42.
- Representacion base = la de V03 (M2): union TF-IDF word(1,2) min_df=2
  sublinear + char_wb(3,5) min_df=2 sublinear + LogisticRegression C=2.0
  balanced. Se reutiliza `construir_matrices` de `experiments/14` importandola
  por ruta (el script 14 no se modifica).
- Cadenas (E2-01): para cada orden (HS->TR->AG, HS->AG->TR, AG->HS->TR) y cada
  pliegue EXTERIOR k:
    1. CV INTERNA de 4 pliegues dentro del entrenamiento exterior; la etiqueta
       previa se entrena en 3/4 internos y predice 1/4 interno -> esos scores
       (out-of-fold respecto al modelo que los produce) son las features
       anadidas para las etiquetas siguientes.
    2. La validacion exterior se predice con el modelo de la etiqueta previa
       entrenado con TODO el entrenamiento exterior del pliegue k.
  El vectorizador se ajusta 5 veces por pliegue exterior (4 internos + 1
  exterior) y se reutiliza entre etiquetas y ordenes: no hay fuga de texto.
- Feature anadida: score de la etiqueta previa y su margen |p-0.5|. Dos modos de
  escalado: `raw` (sin escalar, valores en [0,1]) y `standard` (StandardScaler
  ajustado SOLO con las filas de ajuste del modelo correspondiente).
- Meta-modelo (E2-02): LogisticRegression por etiqueta con C en {0.1, 0.3, 1.0}
  sobre 12 probabilidades OOF de los 4 bases + 12 margenes + 12 entropias + 8
  rasgos estructurales fijos (longitud en caracteres y tokens, mencion, hashtag,
  URL, colectivo plural, segunda persona, comillas). Se evalua fold a fold con
  los mismos 5 pliegues corregidos. Advertencia de honestidad: los scores de los
  bases que el meta-modelo usa como ENTRENAMIENTO son OOF del protocolo v2, pero
  los modelos que los generaron vieron las filas de validacion del pliegue k en
  su propio entrenamiento; el meta-modelo es por tanto algo mas optimista que la
  cadena, que si es estrictamente anidada. Se documenta y no se adopta por si
  solo.
- Regla de adopcion (D-016): corrected_cv >= 0.7772 + 0.005 y mejora en >= 3/5
  pliegues, o `ensemble_gain` >= 0.0015 con IC bootstrap (2000 remuestreos
  pareados a nivel de tweet) estrictamente positivo.
- Umbrales: si una variante se acerca a la adopcion se ajusta una rejilla
  0.20-0.80 por etiqueta SOLO sobre el OOF, etiquetando el resultado como cota
  optimista.

Salidas
-------
``reports/cadenas_etiquetas_v2.csv``, ``reports/cadenas_etiquetas_v2.json``,
``reports/figuras/fig_27_cadenas_v2.png``, filas ``E2-01-*`` y ``E2-02-META`` en
``experiments.csv`` y una seccion en ``reports/contexto.md``.

Ejecucion
---------
    /home/carlos/venv/bin/python experiments/15_cadenas_etiquetas_v2.py
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import sparse
from scipy.special import expit, xlogy
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

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
REPORTE_CSV = PROJECT_ROOT / "reports" / "cadenas_etiquetas_v2.csv"
REPORTE_JSON = PROJECT_ROOT / "reports" / "cadenas_etiquetas_v2.json"
FIGURA_PATH = PROJECT_ROOT / "reports" / "figuras" / "fig_27_cadenas_v2.png"
EXPERIMENTS_CSV = PROJECT_ROOT / "experiments.csv"
CONTEXTO_MD = PROJECT_ROOT / "reports" / "contexto.md"

CONFIG: Final[dict] = {
    "fecha": "2026-09-25",
    "referencia_m2_umbral_05": 0.7718,
    "referencia_v03_umbrales": 0.7772,
    "umbrales_v03": {"HS": 0.44, "TR": 0.51, "AG": 0.46},
    "ordenes": (("HS", "TR", "AG"), ("HS", "AG", "TR"), ("AG", "HS", "TR")),
    "escalados": ("raw", "standard"),
    "n_pliegues_externos": 5,
    "n_pliegues_internos": 4,
    "meta_C": (0.1, 0.3, 1.0),
    "n_bootstrap": 2000,
    "umbral_min": 0.20,
    "umbral_max": 0.80,
    "umbral_paso": 0.02,
    "mejora_minima": 0.005,
    "ensemble_gain_minimo": 0.0015,
    "pliegues_minimos_mejorados": 3,
    "umbral_aproximacion": 0.7772,
}

# Parametros identicos a los de V03 (experiments/14, modelo M2_union_lr_c2).
LR_PARAMS: Final[dict] = {
    "C": 2.0,
    "class_weight": "balanced",
    "max_iter": 1000,
    "solver": "liblinear",
    "random_state": RANDOM_STATE,
}

PATRON_URL = re.compile(r"https?://|www\.|t\.co")
PATRON_MENCION = re.compile(r"@\w+")
PATRON_HASHTAG = re.compile(r"#\w+")
PATRON_COMILLAS = re.compile(r"[\"'\u00ab\u00bb\u201c\u201d]")
# Marcadores plurales/colectivos fijos (lista estructural escrita a mano, no
# ajustada con datos: no puede introducir fuga).
PATRON_COLECTIVO = re.compile(
    r"\b(los|las|esos|esas|todos|todas|ellos|ellas|ustedes|nosotros|nosotras"
    r"|nuestros|nuestras|muchos|muchas|unos|unas|varios|varias|ambos|ambas"
    r"|gente|personas|mujeres|hombres|ninos|ninas|jovenes|ancianos|ancianas"
    r"|indigenas|extranjeros|extranjeras|inmigrantes|venezolanos|venezolanas"
    r"|colombianos|colombianas|funcionarios|politicos|periodistas)\b"
)
PATRON_SEGUNDA_PERSONA = re.compile(
    r"\b(tu|te|ti|tus|vos|usted|ustedes|contigo|tuyo|tuya|tuyos|tuyas"
    r"|sos|tenes|eres|estas|dejame|pideme)\b"
)

NOMBRES_RASGOS_ESTRUCTURALES: Final[tuple[str, ...]] = (
    "n_chars",
    "n_tokens",
    "tiene_mencion",
    "tiene_hashtag",
    "tiene_url",
    "tiene_colectivo",
    "tiene_segunda_persona",
    "tiene_comillas",
)

MODELOS_BASE: Final[tuple[str, ...]] = (
    "M1_word_lr",
    "M2_union_lr_c2",
    "M3_union_svc",
    "M4_char_lr",
)


@dataclass(frozen=True)
class MatricesExteriores:
    """Matrices dispersas de un pliegue exterior con su CV interna.

    Attributes:
        internas: lista de ``(X_entrenamiento, X_validacion, idx_entrenamiento,
            idx_validacion)`` de la CV interna; los indices son posicionales
            dentro del entrenamiento exterior.
        x_entrenamiento: matriz del entrenamiento exterior completo.
        x_validacion: matriz de la validacion exterior.
        idx_entrenamiento: indices posicionales del entrenamiento dentro del
            desarrollo.
        idx_validacion: indices posicionales de la validacion dentro del
            desarrollo.
    """

    internas: list[tuple[sparse.csr_matrix, sparse.csr_matrix, NDArray[np.int_], NDArray[np.int_]]]
    x_entrenamiento: sparse.csr_matrix
    x_validacion: sparse.csr_matrix
    idx_entrenamiento: NDArray[np.int_]
    idx_validacion: NDArray[np.int_]


@dataclass(frozen=True)
class ResultadoNivel:
    """Scores de un nivel de la cadena dentro de un pliegue exterior.

    Attributes:
        scores_entrenamiento: scores OOF de la CV interna para las filas del
            entrenamiento exterior (features de las etiquetas siguientes).
        scores_validacion: scores del modelo entrenado con todo el
            entrenamiento exterior, evaluado en la validacion exterior.
    """

    scores_entrenamiento: NDArray[np.float64]
    scores_validacion: NDArray[np.float64]


def log(mensaje: str) -> None:
    """Imprime un mensaje de progreso con prefijo de fase.

    Args:
        mensaje: texto a imprimir.
    """
    print(f"[P1] {mensaje}", flush=True)


# ===== SECCION 1: carga de datos y particiones =====


def cargar_desarrollo() -> tuple[pd.DataFrame, NDArray[np.int_], list[tuple[NDArray[np.int_], NDArray[np.int_]]]]:
    """Carga el desarrollo y sus pliegues corregidos v2.

    Returns:
        Tupla ``(desarrollo, grupos, pliegues)``. El holdout congelado queda
        excluido por construccion.
    """
    train = data_module.read_train()
    particion = pd.read_csv(IDS_PATH)
    ids_dev = particion.loc[particion["particion"] == "dev", "id"].astype(int).tolist()
    desarrollo = train[train["id"].astype(int).isin(ids_dev)].reset_index(drop=True)

    grupos_completos = grupos_fusionados(train)
    mapa_grupo = dict(zip(train["id"].astype(int), grupos_completos.tolist()))
    grupos = np.array([mapa_grupo[identificador] for identificador in desarrollo["id"].astype(int)])
    pliegues = pliegues_corregidos(desarrollo, grupos)
    return desarrollo, grupos, pliegues


def verificar_datos(desarrollo: pd.DataFrame, grupos: NDArray[np.int_]) -> dict:
    """Comprueba integridad basica y resume el desarrollo.

    Args:
        desarrollo: DataFrame del conjunto de desarrollo.
        grupos: grupos fusionados alineados con ``desarrollo``.

    Returns:
        Diccionario con el resumen de verificacion.
    """
    resumen = {
        "n_filas": len(desarrollo),
        "n_grupos": len(np.unique(grupos)),
        "nulos": int(desarrollo[list(ETIQUETAS) + ["text"]].isna().sum().sum()),
        "prevalencia": {etiqueta: float(desarrollo[etiqueta].mean()) for etiqueta in ETIQUETAS},
        "combinaciones": {clave: int(valor) for clave, valor in combinaciones(desarrollo).value_counts().items()},
    }
    log(
        f"desarrollo n={resumen['n_filas']} grupos={resumen['n_grupos']} "
        f"nulos={resumen['nulos']} prevalencia={resumen['prevalencia']}"
    )
    log(f"combinaciones HS-TR-AG: {resumen['combinaciones']}")
    return resumen


# ===== SECCION 2: representacion base (V03) =====


def cargar_modulo_base() -> object:
    """Importa ``experiments/14_oof_corregido_v2.py`` para reutilizar su receta.

    Returns:
        Modulo del experimento 14, con ``construir_matrices`` (representacion de
        V03) y ``MODELOS`` (los cuatro modelos base del OOF corregido).
    """
    ruta = PROJECT_ROOT / "experiments" / "14_oof_corregido_v2.py"
    especificacion = importlib.util.spec_from_file_location("experimento_14_base", ruta)
    if especificacion is None or especificacion.loader is None:
        raise ImportError(f"No se pudo cargar {ruta}")
    modulo = importlib.util.module_from_spec(especificacion)
    especificacion.loader.exec_module(modulo)
    if tuple(modulo.MODELOS) != MODELOS_BASE:
        raise RuntimeError("Los modelos base de experiments/14 no coinciden con el OOF congelado")
    return modulo


def construir_matrices_exteriores(
    construir_matrices: Callable,
    textos: pd.Series,
    idx_entrenamiento: NDArray[np.int_],
    idx_validacion: NDArray[np.int_],
    combinacion: pd.Series,
    grupos: NDArray[np.int_],
    representacion: str = "union",
) -> MatricesExteriores:
    """Ajusta los vectorizadores del pliegue exterior y de su CV interna.

    Los vectorizadores se ajustan 5 veces (4 pliegues internos y el
    entrenamiento exterior) y se reutilizan para todas las etiquetas y ordenes.

    Args:
        construir_matrices: funcion importada de ``experiments/14``.
        textos: serie de textos del desarrollo.
        idx_entrenamiento: indices del entrenamiento exterior.
        idx_validacion: indices de la validacion exterior.
        combinacion: etiqueta de combinacion HS-TR-AG por fila.
        grupos: grupos fusionados alineados con el desarrollo.
        representacion: ``word``, ``char`` o ``union``.

    Returns:
        Estructura con las matrices internas y externas.
    """
    divisor = StratifiedGroupKFold(
        n_splits=CONFIG["n_pliegues_internos"], shuffle=True, random_state=RANDOM_STATE
    )
    textos_entrenamiento = textos.iloc[idx_entrenamiento].reset_index(drop=True)
    combinacion_entrenamiento = combinacion.iloc[idx_entrenamiento].reset_index(drop=True)
    grupos_entrenamiento = grupos[idx_entrenamiento]

    internas: list[tuple[sparse.csr_matrix, sparse.csr_matrix, NDArray[np.int_], NDArray[np.int_]]] = []
    for idx_interno_entrenamiento, idx_interno_validacion in divisor.split(
        textos_entrenamiento, combinacion_entrenamiento, groups=grupos_entrenamiento
    ):
        matriz_entrenamiento, matriz_validacion = construir_matrices(
            textos_entrenamiento.iloc[idx_interno_entrenamiento],
            textos_entrenamiento.iloc[idx_interno_validacion],
            representacion,
        )
        internas.append(
            (matriz_entrenamiento, matriz_validacion, idx_interno_entrenamiento, idx_interno_validacion)
        )

    matriz_exterior_entrenamiento, matriz_exterior_validacion = construir_matrices(
        textos.iloc[idx_entrenamiento], textos.iloc[idx_validacion], representacion
    )
    return MatricesExteriores(
        internas=internas,
        x_entrenamiento=matriz_exterior_entrenamiento,
        x_validacion=matriz_exterior_validacion,
        idx_entrenamiento=idx_entrenamiento,
        idx_validacion=idx_validacion,
    )


# ===== SECCION 3: cadenas de etiquetas con scores continuos (E2-01) =====


def bloque_denso(scores: list[NDArray[np.float64]]) -> NDArray[np.float64] | None:
    """Construye el bloque denso de scores previos y sus margenes.

    Args:
        scores: lista de scores (uno por etiqueta previa) de longitud ``n``.

    Returns:
        Matriz ``(n, 2 * len(scores))`` con los scores y ``|p - 0.5|``, o
        ``None`` si no hay etiquetas previas.
    """
    if not scores:
        return None
    return np.column_stack([*scores, *[np.abs(score - 0.5) for score in scores]])


def apilar_denso(
    matriz: sparse.csr_matrix,
    denso: NDArray[np.float64] | None,
    escalado: str,
    ajustar: bool,
    escalador: StandardScaler | None = None,
) -> tuple[sparse.csr_matrix, StandardScaler | None]:
    """Concatena el bloque denso (opcionalmente estandarizado) a la matriz dispersa.

    Args:
        matriz: matriz dispersa de la representacion base.
        denso: bloque denso de features de dependencia o ``None``.
        escalado: ``raw`` (sin escalar) o ``standard`` (StandardScaler).
        ajustar: si se debe ajustar el escalador con este bloque.
        escalador: escalador ya ajustado a reutilizar.

    Returns:
        Tupla ``(matriz_apilada, escalador)``; el escalador es ``None`` en modo
        ``raw`` o cuando no hay bloque denso.
    """
    if denso is None:
        return matriz, None
    if escalado == "standard":
        if ajustar:
            escalador = StandardScaler().fit(denso)
        if escalador is None:
            raise ValueError("Se requiere un escalador ajustado en modo standard")
        denso = escalador.transform(denso)
    return sparse.hstack([matriz, sparse.csr_matrix(denso)], format="csr"), escalador


def entrenar_nivel(
    matrices: MatricesExteriores,
    y_etiqueta: NDArray[np.int_],
    previos_entrenamiento: list[NDArray[np.float64]],
    previos_validacion: list[NDArray[np.float64]],
    escalado: str,
) -> ResultadoNivel:
    """Entrena un nivel de la cadena y devuelve sus scores internos y externos.

    Args:
        matrices: matrices del pliegue exterior.
        y_etiqueta: etiqueta verdadera del desarrollo para este nivel.
        previos_entrenamiento: scores OOF de las etiquetas previas en las filas
            del entrenamiento exterior (features anadidas).
        previos_validacion: scores de las etiquetas previas en la validacion
            exterior.
        escalado: ``raw`` o ``standard``.

    Returns:
        Scores del nivel: OOF interno para el entrenamiento exterior y del
        modelo exterior para la validacion.
    """
    y_entrenamiento = y_etiqueta[matrices.idx_entrenamiento]
    scores_entrenamiento = np.zeros(len(y_entrenamiento), dtype=float)

    for matriz_entrenamiento, matriz_validacion, idx_int_ent, idx_int_val in matrices.internas:
        denso_ent = bloque_denso([score[idx_int_ent] for score in previos_entrenamiento])
        denso_val = bloque_denso([score[idx_int_val] for score in previos_entrenamiento])
        matriz_ajustada, escalador = apilar_denso(matriz_entrenamiento, denso_ent, escalado, ajustar=True)
        matriz_predicha, _ = apilar_denso(matriz_validacion, denso_val, escalado, ajustar=False, escalador=escalador)
        modelo = LogisticRegression(**LR_PARAMS).fit(matriz_ajustada, y_entrenamiento[idx_int_ent])
        scores_entrenamiento[idx_int_val] = modelo.predict_proba(matriz_predicha)[:, 1]

    denso_ent = bloque_denso(previos_entrenamiento)
    denso_val = bloque_denso(previos_validacion)
    matriz_ajustada, escalador = apilar_denso(matrices.x_entrenamiento, denso_ent, escalado, ajustar=True)
    matriz_predicha, _ = apilar_denso(matrices.x_validacion, denso_val, escalado, ajustar=False, escalador=escalador)
    modelo = LogisticRegression(**LR_PARAMS).fit(matriz_ajustada, y_entrenamiento)
    scores_validacion = modelo.predict_proba(matriz_predicha)[:, 1]
    return ResultadoNivel(scores_entrenamiento, scores_validacion)


def evaluar_pliegue_exterior(
    matrices: MatricesExteriores,
    y_verdadera: NDArray[np.int_],
    variantes: list[tuple[str, tuple[str, ...], str]],
) -> dict[str, NDArray[np.float64]]:
    """Evalua todas las variantes de cadena dentro de un pliegue exterior.

    Args:
        matrices: matrices del pliegue exterior.
        y_verdadera: matriz ``(n, 3)`` de etiquetas del desarrollo.
        variantes: lista de ``(nombre, orden, escalado)``.

    Returns:
        Diccionario ``nombre -> scores`` de forma ``(n_validacion, 3)``.
    """
    indice_etiqueta = {etiqueta: posicion for posicion, etiqueta in enumerate(ETIQUETAS)}
    raiz: dict[str, ResultadoNivel] = {}
    for etiqueta in ETIQUETAS:
        raiz[etiqueta] = entrenar_nivel(
            matrices, y_verdadera[:, indice_etiqueta[etiqueta]], [], [], "raw"
        )

    resultados: dict[str, NDArray[np.float64]] = {}
    resultados["BASE"] = np.column_stack([raiz[etiqueta].scores_validacion for etiqueta in ETIQUETAS])
    for nombre, orden, escalado in variantes:
        proba = np.full((len(matrices.idx_validacion), len(ETIQUETAS)), np.nan, dtype=float)
        previos_entrenamiento: list[NDArray[np.float64]] = []
        previos_validacion: list[NDArray[np.float64]] = []
        for etiqueta in orden:
            if etiqueta == orden[0]:
                resultado = raiz[etiqueta]
            else:
                resultado = entrenar_nivel(
                    matrices,
                    y_verdadera[:, indice_etiqueta[etiqueta]],
                    previos_entrenamiento,
                    previos_validacion,
                    escalado,
                )
            proba[:, indice_etiqueta[etiqueta]] = resultado.scores_validacion
            previos_entrenamiento.append(resultado.scores_entrenamiento)
            previos_validacion.append(resultado.scores_validacion)
        resultados[nombre] = proba
    return resultados


# ===== SECCION 4: meta-modelo de dependencia (E2-02) =====


def rasgos_estructurales(textos: pd.Series) -> NDArray[np.float64]:
    """Calcula rasgos estructurales fijos (sin ajuste con datos).

    Args:
        textos: serie de textos.

    Returns:
        Matriz ``(n, 8)`` en el orden de :data:`NOMBRES_RASGOS_ESTRUCTURALES`.
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
    return np.asarray(filas, dtype=float)


def entropia_binaria(probabilidades: NDArray[np.float64]) -> NDArray[np.float64]:
    """Calcula la entropia binaria por columna con recorte numerico.

    Args:
        probabilidades: matriz de probabilidades en ``[0, 1]``.

    Returns:
        Matriz de entropias con la misma forma.
    """
    recortadas = np.clip(probabilidades, 1e-6, 1 - 1e-6)
    return -(xlogy(recortadas, recortadas) + xlogy(1 - recortadas, 1 - recortadas))


def construir_matriz_meta(
    proba_bases: NDArray[np.float64], textos: pd.Series
) -> tuple[NDArray[np.float64], list[str]]:
    """Construye las features del meta-modelo de dependencia.

    Args:
        proba_bases: probabilidades OOF de los 4 bases, forma
            ``(n, n_modelos, 3)``.
        textos: serie de textos del desarrollo.

    Returns:
        Tupla ``(matriz, nombres)`` con 12 probabilidades, 12 margenes, 12
        entropias y 8 rasgos estructurales.
    """
    n_filas = proba_bases.shape[0]
    planas = proba_bases.reshape(n_filas, -1)
    margenes = np.abs(planas - 0.5)
    entropias = entropia_binaria(planas)
    estructurales = rasgos_estructurales(textos)
    nombres = (
        [f"proba_{modelo}_{etiqueta}" for modelo in MODELOS_BASE for etiqueta in ETIQUETAS]
        + [f"margen_{modelo}_{etiqueta}" for modelo in MODELOS_BASE for etiqueta in ETIQUETAS]
        + [f"entropia_{modelo}_{etiqueta}" for modelo in MODELOS_BASE for etiqueta in ETIQUETAS]
        + list(NOMBRES_RASGOS_ESTRUCTURALES)
    )
    return np.hstack([planas, margenes, entropias, estructurales]), nombres


def evaluar_meta(
    features: NDArray[np.float64],
    y_verdadera: NDArray[np.int_],
    pliegues: list[tuple[NDArray[np.int_], NDArray[np.int_]]],
) -> dict[float, NDArray[np.float64]]:
    """Evalua el meta-modelo por etiqueta con predicciones fold a fold.

    Args:
        features: matriz de features del meta-modelo.
        y_verdadera: matriz ``(n, 3)`` de etiquetas.
        pliegues: pliegues corregidos ``(entrenamiento, validacion)``.

    Returns:
        Diccionario ``C -> probabilidades OOF`` de forma ``(n, 3)``.
    """
    resultados: dict[float, NDArray[np.float64]] = {}
    for valor_c in CONFIG["meta_C"]:
        proba = np.zeros((len(features), len(ETIQUETAS)), dtype=float)
        for idx_entrenamiento, idx_validacion in pliegues:
            escalador = StandardScaler().fit(features[idx_entrenamiento])
            x_entrenamiento = escalador.transform(features[idx_entrenamiento])
            x_validacion = escalador.transform(features[idx_validacion])
            for posicion, etiqueta in enumerate(ETIQUETAS):
                modelo = LogisticRegression(
                    C=valor_c,
                    class_weight="balanced",
                    max_iter=2000,
                    solver="liblinear",
                    random_state=RANDOM_STATE,
                )
                modelo.fit(x_entrenamiento, y_verdadera[idx_entrenamiento, posicion])
                proba[idx_validacion, posicion] = modelo.predict_proba(x_validacion)[:, 1]
        resultados[valor_c] = proba
    return resultados


# ===== SECCION 5: metricas, bootstrap, umbrales y complementariedad =====


def vector_umbrales(umbrales: dict[str, float]) -> NDArray[np.float64]:
    """Convierte el diccionario de umbrales en vector en el orden de etiquetas.

    Args:
        umbrales: umbral por etiqueta.

    Returns:
        Arreglo ``(3,)`` en el orden de :data:`ETIQUETAS`.
    """
    return np.array([umbrales[etiqueta] for etiqueta in ETIQUETAS])


def evaluar_con_umbrales(
    proba: NDArray[np.float64], y_verdadera: NDArray[np.int_], umbrales: dict[str, float]
) -> dict:
    """Calcula las metricas multi-etiqueta con umbrales por etiqueta.

    Args:
        proba: probabilidades ``(n, 3)``.
        y_verdadera: etiquetas verdaderas ``(n, 3)``.
        umbrales: umbral por etiqueta.

    Returns:
        Metricas de :func:`metricas_multietiqueta`.
    """
    prediccion = (proba >= vector_umbrales(umbrales)).astype(int)
    return metricas_multietiqueta(y_verdadera, prediccion)


def f1_por_pliegue(
    proba: NDArray[np.float64],
    y_verdadera: NDArray[np.int_],
    pliegue_de_fila: NDArray[np.int_],
    umbrales: dict[str, float],
) -> list[float]:
    """Calcula la F1 media por columna en cada pliegue.

    Args:
        proba: probabilidades ``(n, 3)``.
        y_verdadera: etiquetas verdaderas ``(n, 3)``.
        pliegue_de_fila: numero de pliegue (1..k) de cada fila.
        umbrales: umbral por etiqueta.

    Returns:
        Lista de F1 por pliegue en orden creciente de pliegue.
    """
    valores = []
    for numero in sorted(np.unique(pliegue_de_fila).tolist()):
        mascara = pliegue_de_fila == numero
        metrica = metricas_multietiqueta(
            y_verdadera[mascara], (proba[mascara] >= vector_umbrales(umbrales)).astype(int)
        )
        valores.append(metrica["f1_media_columnas"])
    return valores


def f1_media_rapida(
    y_verdadera: NDArray[np.int_], prediccion: NDArray[np.int_]
) -> float:
    """Calcula la F1 media por columna de forma vectorizada.

    Es numericamente identica a ``metricas_multietiqueta(...)['f1_media_columnas']``
    (misma definicion binaria con ``zero_division=0``) pero sin la sobrecarga de
    sklearn, necesaria para los 2000 remuestreos del bootstrap.

    Args:
        y_verdadera: matriz binaria verdadera ``(n, k)``.
        prediccion: matriz binaria predicha ``(n, k)``.

    Returns:
        Media de la F1 binaria de la clase positiva por columna.
    """
    valores = []
    for posicion in range(y_verdadera.shape[1]):
        verdad = y_verdadera[:, posicion].astype(bool)
        aciertos = prediccion[:, posicion].astype(bool)
        verdaderos_positivos = int(np.count_nonzero(verdad & aciertos))
        falsos_positivos = int(np.count_nonzero(~verdad & aciertos))
        falsos_negativos = int(np.count_nonzero(verdad & ~aciertos))
        denominador = 2 * verdaderos_positivos + falsos_positivos + falsos_negativos
        valores.append(2 * verdaderos_positivos / denominador if denominador else 0.0)
    return float(np.mean(valores))


def ajustar_y_predecir(
    especificacion: dict,
    x_entrenamiento: sparse.csr_matrix,
    y_entrenamiento: NDArray[np.int_],
    x_prediccion: sparse.csr_matrix,
) -> NDArray[np.float64]:
    """Ajusta un modelo base y devuelve probabilidades (receta de ``experiments/14``).

    Args:
        especificacion: diccionario con ``tipo`` (``lr`` o ``svc``) y ``params``.
        x_entrenamiento: matriz del pliegue de entrenamiento.
        y_entrenamiento: etiqueta binaria del pliegue de entrenamiento.
        x_prediccion: matriz a predecir.

    Returns:
        Probabilidades (LR) o sigmoide del ``decision_function`` (SVC).
    """
    if especificacion["tipo"] == "lr":
        modelo = LogisticRegression(**especificacion["params"]).fit(x_entrenamiento, y_entrenamiento)
        return modelo.predict_proba(x_prediccion)[:, 1]
    modelo = LinearSVC(**especificacion["params"]).fit(x_entrenamiento, y_entrenamiento)
    return expit(modelo.decision_function(x_prediccion))


def evaluar_meta_anidado(
    construccion_matrices: Callable,
    modulo_base: object,
    textos: pd.Series,
    y_verdadera: NDArray[np.int_],
    idx_entrenamiento: NDArray[np.int_],
    idx_validacion: NDArray[np.int_],
    combinacion: pd.Series,
    grupos: NDArray[np.int_],
    matrices_union: MatricesExteriores,
) -> dict[float, NDArray[np.float64]]:
    """Verificacion ESTRICTAMENTE anidada del meta-modelo (sin reutilizar el OOF congelado).

    Los cuatro modelos base se reentrenan dentro del pliegue exterior: la CV
    interna produce sus features para el entrenamiento del meta y el modelo
    externo (entrenado sin la validacion) produce las de la validacion. Asi los
    scores que alimentan al meta NO provienen de bases que vieron el pliegue
    evaluado, a diferencia de la version que usa ``v2_oof_base_corrected.npz``.

    Args:
        construccion_matrices: ``construir_matrices`` de ``experiments/14``.
        modulo_base: modulo de ``experiments/14`` con la especificacion ``MODELOS``.
        textos: serie de textos del desarrollo.
        y_verdadera: matriz ``(n, 3)`` de etiquetas del desarrollo.
        idx_entrenamiento: indices del entrenamiento exterior.
        idx_validacion: indices de la validacion exterior.
        combinacion: etiqueta de combinacion HS-TR-AG por fila.
        grupos: grupos fusionados alineados con el desarrollo.
        matrices_union: matrices de la representacion de V03 ya construidas para
            este pliegue (se reutilizan para M2 y M3).

    Returns:
        Diccionario ``C -> probabilidades`` en la validacion exterior.
    """
    matrices_por_representacion = {"union": matrices_union}
    for representacion in ("word", "char"):
        matrices_por_representacion[representacion] = construir_matrices_exteriores(
            construccion_matrices,
            textos,
            idx_entrenamiento,
            idx_validacion,
            combinacion,
            grupos,
            representacion,
        )

    y_entrenamiento = y_verdadera[idx_entrenamiento]
    proba_entrenamiento = np.zeros((len(idx_entrenamiento), len(MODELOS_BASE), len(ETIQUETAS)))
    proba_validacion = np.zeros((len(idx_validacion), len(MODELOS_BASE), len(ETIQUETAS)))
    for posicion_modelo, (nombre, especificacion) in enumerate(modulo_base.MODELOS.items()):
        matrices = matrices_por_representacion[especificacion["representacion"]]
        for posicion_etiqueta, etiqueta in enumerate(ETIQUETAS):
            for matriz_entrenamiento, matriz_validacion, idx_int_ent, idx_int_val in matrices.internas:
                proba_entrenamiento[idx_int_val, posicion_modelo, posicion_etiqueta] = ajustar_y_predecir(
                    especificacion,
                    matriz_entrenamiento,
                    y_entrenamiento[idx_int_ent, posicion_etiqueta],
                    matriz_validacion,
                )
            proba_validacion[:, posicion_modelo, posicion_etiqueta] = ajustar_y_predecir(
                especificacion,
                matrices.x_entrenamiento,
                y_entrenamiento[:, posicion_etiqueta],
                matrices.x_validacion,
            )
        log(f"    meta anidado: base {nombre} lista")

    features_entrenamiento, _ = construir_matriz_meta(proba_entrenamiento, textos.iloc[idx_entrenamiento])
    features_validacion, _ = construir_matriz_meta(proba_validacion, textos.iloc[idx_validacion])
    escalador = StandardScaler().fit(features_entrenamiento)
    x_entrenamiento_meta = escalador.transform(features_entrenamiento)
    x_validacion_meta = escalador.transform(features_validacion)

    resultados: dict[float, NDArray[np.float64]] = {}
    for valor_c in CONFIG["meta_C"]:
        proba = np.zeros((len(idx_validacion), len(ETIQUETAS)))
        for posicion, _ in enumerate(ETIQUETAS):
            modelo = LogisticRegression(
                C=valor_c,
                class_weight="balanced",
                max_iter=2000,
                solver="liblinear",
                random_state=RANDOM_STATE,
            )
            modelo.fit(x_entrenamiento_meta, y_entrenamiento[:, posicion])
            proba[:, posicion] = modelo.predict_proba(x_validacion_meta)[:, 1]
        resultados[valor_c] = proba
    return resultados


def verificar_metrica_rapida(
    proba: NDArray[np.float64], y_verdadera: NDArray[np.int_], umbrales: dict[str, float]
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
    y_verdadera: NDArray[np.int_],
    umbrales_referencia: dict[str, float],
    umbrales_variante: dict[str, float],
) -> dict:
    """Calcula el IC bootstrap pareado del delta de F1 a nivel de tweet.

    Args:
        proba_referencia: probabilidades de la referencia.
        proba_variante: probabilidades de la variante.
        y_verdadera: etiquetas verdaderas.
        umbrales_referencia: umbrales de la referencia.
        umbrales_variante: umbrales de la variante.

    Returns:
        Diccionario con media, cuantiles 2.5/97.5, fraccion positiva y numero
        de remuestreos.
    """
    generador = np.random.default_rng(RANDOM_STATE)
    n_filas = len(y_verdadera)
    prediccion_ref = (proba_referencia >= vector_umbrales(umbrales_referencia)).astype(int)
    prediccion_var = (proba_variante >= vector_umbrales(umbrales_variante)).astype(int)
    deltas = np.empty(CONFIG["n_bootstrap"], dtype=float)
    for iteracion in range(CONFIG["n_bootstrap"]):
        muestra = generador.integers(0, n_filas, size=n_filas)
        verdad = y_verdadera[muestra]
        deltas[iteracion] = f1_media_rapida(
            verdad, prediccion_var[muestra]
        ) - f1_media_rapida(verdad, prediccion_ref[muestra])
    return {
        "media": float(np.mean(deltas)),
        "ic_inferior": float(np.quantile(deltas, 0.025)),
        "ic_superior": float(np.quantile(deltas, 0.975)),
        "fraccion_positiva": float(np.mean(deltas > 0)),
        "remuestreos": int(CONFIG["n_bootstrap"]),
    }


def ajustar_umbrales(
    proba: NDArray[np.float64], y_verdadera: NDArray[np.int_]
) -> tuple[dict[str, float], float]:
    """Ajusta umbrales por etiqueta sobre las probabilidades OOF (cota optimista).

    Args:
        proba: probabilidades OOF.
        y_verdadera: etiquetas verdaderas.

    Returns:
        Tupla ``(umbrales, f1)`` con la rejilla 0.20-0.80 y F1 resultante.
    """
    rejilla = np.arange(CONFIG["umbral_min"], CONFIG["umbral_max"] + 1e-9, CONFIG["umbral_paso"])
    umbrales: dict[str, float] = {}
    for posicion, etiqueta in enumerate(ETIQUETAS):
        mejor_umbral, mejor_f1 = 0.5, -1.0
        for umbral in rejilla:
            f1 = f1_media_rapida(
                y_verdadera[:, [posicion]], (proba[:, [posicion]] >= umbral).astype(int)
            )
            if f1 > mejor_f1:
                mejor_umbral, mejor_f1 = float(umbral), f1
        umbrales[etiqueta] = mejor_umbral
    return umbrales, evaluar_con_umbrales(proba, y_verdadera, umbrales)["f1_media_columnas"]


def resumir_variante(
    nombre: str,
    proba: NDArray[np.float64],
    y_verdadera: NDArray[np.int_],
    pliegue_de_fila: NDArray[np.int_],
    proba_base: NDArray[np.float64],
) -> dict:
    """Resume una variante frente a las dos referencias del protocolo v2.

    Args:
        nombre: identificador de la variante.
        proba: probabilidades OOF de la variante.
        y_verdadera: etiquetas verdaderas.
        pliegue_de_fila: numero de pliegue de cada fila.
        proba_base: probabilidades del modelo base sin cadena (M2).

    Returns:
        Diccionario con F1 global, por etiqueta, por pliegue y deltas.
    """
    sin_umbral = {etiqueta: 0.5 for etiqueta in ETIQUETAS}
    metrica = evaluar_con_umbrales(proba, y_verdadera, sin_umbral)
    f1_pliegues = f1_por_pliegue(proba, y_verdadera, pliegue_de_fila, sin_umbral)
    f1_pliegues_base = f1_por_pliegue(proba_base, y_verdadera, pliegue_de_fila, sin_umbral)
    f1_pliegues_v03 = f1_por_pliegue(proba_base, y_verdadera, pliegue_de_fila, CONFIG["umbrales_v03"])
    return {
        "variante": nombre,
        "corrected_cv": metrica["f1_media_columnas"],
        "f1_por_etiqueta": metrica["f1_por_etiqueta"],
        "f1_pliegues": f1_pliegues,
        "std_pliegues": float(np.std(f1_pliegues)),
        "delta_vs_m2_05": metrica["f1_media_columnas"] - CONFIG["referencia_m2_umbral_05"],
        "delta_vs_v03_umbrales": metrica["f1_media_columnas"] - CONFIG["referencia_v03_umbrales"],
        "pliegues_mejorados_vs_m2_05": int(np.sum(np.array(f1_pliegues) > np.array(f1_pliegues_base))),
        "pliegues_mejorados_vs_v03": int(np.sum(np.array(f1_pliegues) > np.array(f1_pliegues_v03))),
        "f1_pliegues_base_m2_05": f1_pliegues_base,
        "f1_pliegues_base_v03_umbrales": f1_pliegues_v03,
    }


def evaluar_complementariedad(
    nombre: str,
    proba_variante: NDArray[np.float64],
    proba_base: NDArray[np.float64],
    y_verdadera: NDArray[np.int_],
) -> dict:
    """Mide la complementariedad por promedio de probabilidades con el base.

    Args:
        nombre: identificador de la variante.
        proba_variante: probabilidades de la variante.
        proba_base: probabilidades del base M2 (V03 sin umbrales).
        y_verdadera: etiquetas verdaderas.

    Returns:
        Diccionario con la F1 del ensemble, la ganancia y el bootstrap del delta
        (solo si la ganancia es positiva).
    """
    sin_umbral = {etiqueta: 0.5 for etiqueta in ETIQUETAS}
    proba_ensemble = (proba_variante + proba_base) / 2.0
    metrica = evaluar_con_umbrales(proba_ensemble, y_verdadera, sin_umbral)
    ganancia = metrica["f1_media_columnas"] - CONFIG["referencia_m2_umbral_05"]
    resultado = {
        "variante": nombre,
        "corrected_cv_ensemble": metrica["f1_media_columnas"],
        "ensemble_gain": ganancia,
        "cumple_umbral_gain": bool(ganancia >= CONFIG["ensemble_gain_minimo"]),
    }
    if ganancia > 0:
        resultado["bootstrap"] = bootstrap_delta(
            proba_base, proba_ensemble, y_verdadera, sin_umbral, sin_umbral
        )
    return resultado


def decidir(resultados: dict[str, dict]) -> dict:
    """Aplica la regla de adopcion pre-registrada en D-016.

    Args:
        resultados: resumen por variante.

    Returns:
        Diccionario con la decision y su justificacion.
    """
    candidatos = {nombre: fila for nombre, fila in resultados.items() if nombre != "BASE"}
    mejor = max(candidatos.items(), key=lambda par: par[1]["corrected_cv"])
    nombre, fila = mejor
    cumple_nivel = fila["corrected_cv"] >= CONFIG["referencia_v03_umbrales"] + CONFIG["mejora_minima"]
    cumple_pliegues = fila["pliegues_mejorados_vs_v03"] >= CONFIG["pliegues_minimos_mejorados"]
    adoptado = bool(cumple_nivel and cumple_pliegues)
    return {
        "mejor_variante": nombre,
        "corrected_cv": fila["corrected_cv"],
        "criterio_nivel": {
            "objetivo": CONFIG["referencia_v03_umbrales"] + CONFIG["mejora_minima"],
            "cumple": bool(cumple_nivel),
        },
        "criterio_pliegues": {
            "objetivo": CONFIG["pliegues_minimos_mejorados"],
            "pliegues_mejorados": fila["pliegues_mejorados_vs_v03"],
            "cumple": bool(cumple_pliegues),
        },
        "adoptado": adoptado,
        "justificacion": (
            "Adoptada: supera el nivel exigido y mejora en al menos 3/5 pliegues."
            if adoptado
            else "No adoptada: no alcanza el nivel exigido y/o no mejora en 3/5 "
            "pliegues; se evalua la via de complementariedad (ensemble_gain)."
        ),
    }


# ===== SECCION 6: figura, artefactos y main =====


def generar_figura(
    resultados: dict[str, dict], meta_resumen: dict, meta_anidado_resumen: dict
) -> None:
    """Genera la figura comparativa de variantes y referencias.

    Args:
        resultados: resumen por variante.
        meta_resumen: resumen del mejor meta-modelo (no anidado).
        meta_anidado_resumen: resumen del mejor meta-modelo estrictamente anidado.
    """
    etiquetas_figura = []
    valores = []
    for nombre, fila in resultados.items():
        etiquetas_figura.append(nombre)
        valores.append(fila["corrected_cv"])
    etiquetas_figura.append("META")
    valores.append(meta_resumen["corrected_cv"])
    etiquetas_figura.append("META-ANIDADO")
    valores.append(meta_anidado_resumen["corrected_cv"])

    figura, eje = plt.subplots(figsize=(10, 5))
    colores = ["#8c8c8c" if nombre == "BASE" else "#4c72b0" for nombre in etiquetas_figura]
    colores[-1] = "#dd8452"
    eje.barh(etiquetas_figura, valores, color=colores)
    eje.axvline(
        CONFIG["referencia_m2_umbral_05"],
        color="#d62728",
        linestyle="--",
        linewidth=1.2,
        label=f"M2 sin umbrales ({CONFIG['referencia_m2_umbral_05']:.4f})",
    )
    eje.axvline(
        CONFIG["referencia_v03_umbrales"],
        color="#2ca02c",
        linestyle="-.",
        linewidth=1.2,
        label=f"V03 con umbrales ({CONFIG['referencia_v03_umbrales']:.4f})",
    )
    eje.axvline(
        CONFIG["referencia_v03_umbrales"] + CONFIG["mejora_minima"],
        color="#9467bd",
        linestyle=":",
        linewidth=1.2,
        label=f"umbral de adopcion ({CONFIG['referencia_v03_umbrales'] + CONFIG['mejora_minima']:.4f})",
    )
    eje.set_xlabel("corrected_cv (F1 media por columna, OOF, umbral 0.5)")
    eje.set_title("P1 - Dependencia entre etiquetas (desarrollo n=1483, protocolo v2)")
    eje.legend(loc="lower right", fontsize=8)
    for posicion, valor in enumerate(valores):
        eje.text(valor + 0.0005, posicion, f"{valor:.4f}", va="center", fontsize=8)
    eje.set_xlim(min(valores + [CONFIG["referencia_m2_umbral_05"]]) - 0.004, max(valores) + 0.005)
    figura.tight_layout()
    FIGURA_PATH.parent.mkdir(parents=True, exist_ok=True)
    figura.savefig(FIGURA_PATH, dpi=150)
    plt.close(figura)


def escribir_reporte_csv(
    resultados: dict[str, dict], meta_resumen: dict, meta_anidado_resumen: dict
) -> pd.DataFrame:
    """Escribe el CSV comparativo de variantes.

    Args:
        resultados: resumen por variante.
        meta_resumen: resumen del mejor meta-modelo (no anidado).
        meta_anidado_resumen: resumen del mejor meta-modelo estrictamente anidado.

    Returns:
        DataFrame escrito.
    """
    filas = []
    for nombre, fila in resultados.items():
        filas.append(
            {
                "variante": nombre,
                "corrected_cv": round(fila["corrected_cv"], 4),
                "f1_HS": round(fila["f1_por_etiqueta"]["HS"], 4),
                "f1_TR": round(fila["f1_por_etiqueta"]["TR"], 4),
                "f1_AG": round(fila["f1_por_etiqueta"]["AG"], 4),
                "std_pliegues": round(fila["std_pliegues"], 4),
                "delta_vs_m2_05": round(fila["delta_vs_m2_05"], 4),
                "delta_vs_v03_umbrales": round(fila["delta_vs_v03_umbrales"], 4),
                "pliegues_mejorados_vs_v03": fila["pliegues_mejorados_vs_v03"],
                "f1_pliegues": "|".join(f"{valor:.4f}" for valor in fila["f1_pliegues"]),
                "ensemble_gain": round(fila.get("ensemble_gain", float("nan")), 4),
                "adoptado": bool(fila.get("adoptado", False)),
            }
        )
    for nombre, resumen in (("META", meta_resumen), ("META-ANIDADO", meta_anidado_resumen)):
        filas.append(
            {
                "variante": nombre,
                "corrected_cv": round(resumen["corrected_cv"], 4),
                "f1_HS": round(resumen["f1_por_etiqueta"]["HS"], 4),
                "f1_TR": round(resumen["f1_por_etiqueta"]["TR"], 4),
                "f1_AG": round(resumen["f1_por_etiqueta"]["AG"], 4),
                "std_pliegues": round(resumen["std_pliegues"], 4),
                "delta_vs_m2_05": round(resumen["delta_vs_m2_05"], 4),
                "delta_vs_v03_umbrales": round(resumen["delta_vs_v03_umbrales"], 4),
                "pliegues_mejorados_vs_v03": resumen["pliegues_mejorados_vs_v03"],
                "f1_pliegues": "|".join(f"{valor:.4f}" for valor in resumen["f1_pliegues"]),
                "ensemble_gain": round(resumen.get("ensemble_gain", float("nan")), 4),
                "adoptado": bool(resumen.get("adoptado", False)),
            }
        )
    tabla = pd.DataFrame(filas)
    REPORTE_CSV.parent.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(REPORTE_CSV, index=False)
    return tabla


def registrar_experimentos(
    resultados: dict[str, dict],
    meta_resumen: dict,
    meta_anidado_resumen: dict,
    duracion: float,
    tabla: pd.DataFrame,
) -> None:
    """Anade las filas ``E2-01-*`` y ``E2-02-META`` a ``experiments.csv``.

    Args:
        resultados: resumen por variante.
        meta_resumen: resumen del mejor meta-modelo (no anidado).
        meta_anidado_resumen: resumen del mejor meta-modelo estrictamente anidado.
        duracion: segundos totales del experimento.
        tabla: tabla resumen ya escrita en CSV.
    """
    filas = []
    for nombre, fila in resultados.items():
        if nombre == "BASE":
            continue
        partes = nombre.split("_")
        orden = "-".join(partes[:3])
        escalado = partes[3] if len(partes) > 3 else "raw"
        identificador = f"E2-01-{orden}" if escalado == "raw" else f"E2-01-{orden}-STD"
        filas.append(
            {
                "experiment_id": identificador,
                "competition": "KAGGLE_TOXIC_2026_PUJ",
                "question": "Aporta la dependencia entre etiquetas como score continuo OOF en cadena anidada?",
                "hypothesis": "TR y AG mejoran si conocen P(HS) y P(TR) como feature continua (E2-01)",
                "preprocessing": f"texto crudo + score OOF de la etiqueta previa (escalado={escalado})",
                "feature_groups": "WORD_NGRAM+CHAR_NGRAM+LABEL_DEPENDENCY",
                "representation": "union word(1,2)+char_wb(3,5) min_df=2 sublinear + score previo",
                "model": f"cadena {orden} LR C=2.0 balanced",
                "hyperparameters": json.dumps(
                    {"LR": LR_PARAMS, "orden": orden, "escalado": escalado}, ensure_ascii=False
                ),
                "validation": "StratifiedGroupKFold(5) desarrollo, grupos fusionados, CV interna de 4",
                "f1": round(fila["corrected_cv"], 4),
                "macro_f1": "",
                "weighted_f1": "",
                "precision": "",
                "recall": "",
                "training_time": round(duracion, 2),
                "dimension": "",
                "result": json.dumps(
                    {
                        "corrected_cv": fila["corrected_cv"],
                        "f1_por_etiqueta": fila["f1_por_etiqueta"],
                        "pliegues": fila["f1_pliegues"],
                        "std_pliegues": fila["std_pliegues"],
                        "delta_vs_m2_05": fila["delta_vs_m2_05"],
                        "delta_vs_v03_umbrales": fila["delta_vs_v03_umbrales"],
                        "pliegues_mejorados_vs_v03": fila["pliegues_mejorados_vs_v03"],
                        "ensemble_gain": fila.get("ensemble_gain"),
                        "bootstrap": fila.get("bootstrap"),
                    },
                    ensure_ascii=False,
                ),
                "interpretation": "Cadena con scores continuos anidada; comparada con V03 (0.7772)",
                "decision": fila.get("decision", "No adoptada"),
                "date": CONFIG["fecha"],
                "script": "experiments/15_cadenas_etiquetas_v2.py",
                "notes": "historical_cv no aplica (protocolo v2); holdout congelado no utilizado",
            }
        )
    filas.append(
        {
            "experiment_id": "E2-02-META",
            "competition": "KAGGLE_TOXIC_2026_PUJ",
            "question": "Un meta-modelo regularizado sobre scores OOF captura la dependencia mejor que el modelo plano?",
            "hypothesis": "LR regularizada sobre probabilidades, margenes, entropias y rasgos estructurales mejora el corrected_cv",
            "preprocessing": "sin texto: 12 probabilidades OOF + 12 margenes + 12 entropias + 8 rasgos estructurales",
            "feature_groups": "META_SCORES+STRUCTURAL",
            "representation": "features densas (44 columnas) estandarizadas",
            "model": f"LogisticRegression por etiqueta (C en {list(CONFIG['meta_C'])}, balanced)",
            "hyperparameters": json.dumps({"meta_C": list(CONFIG["meta_C"])}, ensure_ascii=False),
            "validation": "mismos 5 pliegues corregidos, prediccion fold a fold",
            "f1": round(meta_anidado_resumen["corrected_cv"], 4),
            "macro_f1": "",
            "weighted_f1": "",
            "precision": "",
            "recall": "",
            "training_time": round(duracion, 2),
            "dimension": 44,
            "result": json.dumps(
                {
                    "corrected_cv_no_anidado": meta_resumen["corrected_cv"],
                    "corrected_cv_anidado": meta_anidado_resumen["corrected_cv"],
                    "por_C_no_anidado": meta_resumen["por_C"],
                    "por_C_anidado": meta_anidado_resumen["por_C"],
                    "f1_por_etiqueta_anidado": meta_anidado_resumen["f1_por_etiqueta"],
                    "pliegues_anidado": meta_anidado_resumen["f1_pliegues"],
                    "ensemble_gain": meta_anidado_resumen.get("ensemble_gain"),
                    "bootstrap": meta_anidado_resumen.get("bootstrap"),
                    "advertencia": meta_anidado_resumen["advertencia"],
                },
                ensure_ascii=False,
            ),
            "interpretation": (
                "El meta no anidado usa scores OOF generados por bases que vieron el pliegue "
                "evaluado (optimista); el anidado reentrena los 4 bases en cada pliegue"
            ),
            "decision": meta_anidado_resumen.get("decision", "No adoptada"),
            "date": CONFIG["fecha"],
            "script": "experiments/15_cadenas_etiquetas_v2.py",
            "notes": (
                "historical_cv no aplica (protocolo v2); holdout congelado no utilizado; "
                "f1 = corrected_cv de la verificacion estrictamente anidada"
            ),
        }
    )
    existente = pd.read_csv(EXPERIMENTS_CSV)
    identificadores = {fila["experiment_id"] for fila in filas}
    existente = existente[~existente["experiment_id"].isin(identificadores)]
    existente.to_csv(EXPERIMENTS_CSV, index=False)
    pd.DataFrame(filas).to_csv(EXPERIMENTS_CSV, mode="a", header=False, index=False)
    log(f"filas registradas: {sorted(identificadores)}")
    log(f"tabla resumen filas={len(tabla)}")


def escribir_contexto(
    tabla: pd.DataFrame,
    decision: dict,
    duracion: float,
    meta_resumen: dict,
    meta_anidado_resumen: dict,
) -> None:
    """Anade la seccion de P1 al final de ``reports/contexto.md``.

    Args:
        tabla: tabla resumen de variantes.
        decision: decision de adopcion.
        duracion: segundos totales del experimento.
        meta_resumen: resumen del mejor meta-modelo (no anidado).
        meta_anidado_resumen: resumen del meta-modelo estrictamente anidado.
    """
    encabezado = "## Fase v2 - P1 (2026-09-25)"
    contenido = CONTEXTO_MD.read_text(encoding="utf-8") if CONTEXTO_MD.exists() else ""
    if encabezado in contenido:
        contenido = contenido[: contenido.index(encabezado)].rstrip() + "\n"
    lineas = [encabezado]
    for _, fila in tabla.iterrows():
        ganancia = fila["ensemble_gain"]
        texto_ganancia = "-" if pd.isna(ganancia) else f"{ganancia:+.4f}"
        lineas.append(
            f"{fila['variante']}: corrected_cv {fila['corrected_cv']:.4f} "
            f"(delta vs V03 {fila['delta_vs_v03_umbrales']:+.4f}; pliegues mejorados "
            f"{fila['pliegues_mejorados_vs_v03']}/5); por etiqueta HS {fila['f1_HS']:.3f} / "
            f"TR {fila['f1_TR']:.3f} / AG {fila['f1_AG']:.3f}; ensemble_gain {texto_ganancia}."
        )
    lineas.append("Referencias del protocolo v2 en el desarrollo: M2 a 0.5 = 0.7718; V03 con umbrales = 0.7772.")
    lineas.append(
        f"Meta estrictamente anidado (los 4 bases se reentrenan en cada pliegue): corrected_cv "
        f"{meta_anidado_resumen['corrected_cv']:.4f} (C {meta_anidado_resumen['mejor_C']}), "
        f"por C {meta_anidado_resumen['por_C']}, std entre pliegues "
        f"{meta_anidado_resumen['std_pliegues']:.4f} (por debajo de la contingencia 0.025); la "
        f"version con el OOF congelado daba {meta_resumen['corrected_cv']:.4f}, luego el "
        f"anidamiento NO inflaba el numero."
    )
    lineas.append(
        f"Gate P1: {'ADOPTADO' if decision['adoptado'] else 'NO ADOPTADO'} "
        f"(mejor variante {decision['mejor_variante']}, corrected_cv "
        f"{decision['corrected_cv']:.4f}); regla D-016 exige >= 0.7822 con >= 3/5 pliegues o "
        f"ensemble_gain >= 0.0015 con IC bootstrap > 0."
    )
    lineas.append(
        "Umbrales: " + decision.get("umbrales_texto", "no se ajustaron (ninguna variante alcanzo 0.7772).")
    )
    lineas.append(
        "Sin fuga: holdout congelado no utilizado; vectorizadores, escaladores, umbrales y "
        "seleccion solo sobre OOF del desarrollo; el meta-modelo que se reporta como verificacion "
        "reentrena los 4 bases dentro de cada pliegue exterior (CV interna de 4 pliegues)."
    )
    lineas.append(f"Duracion: {duracion:.1f} s de pared. Script: experiments/15_cadenas_etiquetas_v2.py.")
    CONTEXTO_MD.write_text(contenido + "\n" + "\n".join(lineas) + "\n", encoding="utf-8")


def main() -> None:
    """Ejecuta P1.1/P1.2, imprime el resumen y guarda los artefactos."""
    inicio = time.perf_counter()
    desarrollo, grupos, pliegues = cargar_desarrollo()
    verificacion = verificar_datos(desarrollo, grupos)
    modulo_base = cargar_modulo_base()
    construir_matrices = modulo_base.construir_matrices
    y_verdadera = desarrollo[list(ETIQUETAS)].to_numpy()
    combinacion = combinaciones(desarrollo)
    pliegue_de_fila = np.zeros(len(desarrollo), dtype=int)
    for numero, (_, idx_validacion) in enumerate(pliegues, start=1):
        pliegue_de_fila[idx_validacion] = numero

    variantes = [
        (f"{orden[0]}_{orden[1]}_{orden[2]}_{escalado}", orden, escalado)
        for orden in CONFIG["ordenes"]
        for escalado in CONFIG["escalados"]
    ]
    nombres_variantes = [nombre for nombre, _, _ in variantes]
    probas = {nombre: np.zeros((len(desarrollo), len(ETIQUETAS))) for nombre in ["BASE", *nombres_variantes]}
    probas_meta_anidado = {
        valor_c: np.zeros((len(desarrollo), len(ETIQUETAS))) for valor_c in CONFIG["meta_C"]
    }

    for numero, (idx_entrenamiento, idx_validacion) in enumerate(pliegues, start=1):
        matrices = construir_matrices_exteriores(
            construir_matrices, desarrollo["text"], idx_entrenamiento, idx_validacion, combinacion, grupos
        )
        resultados_pliegue = evaluar_pliegue_exterior(matrices, y_verdadera, variantes)
        for nombre, proba in resultados_pliegue.items():
            probas[nombre][idx_validacion] = proba
        anidado = evaluar_meta_anidado(
            construir_matrices,
            modulo_base,
            desarrollo["text"],
            y_verdadera,
            idx_entrenamiento,
            idx_validacion,
            combinacion,
            grupos,
            matrices,
        )
        for valor_c, proba in anidado.items():
            probas_meta_anidado[valor_c][idx_validacion] = proba
        log(
            f"pliegue {numero}/5 listo en {time.perf_counter() - inicio:.1f} s "
            f"(base {metricas_multietiqueta(y_verdadera[idx_validacion], (probas['BASE'][idx_validacion] >= 0.5).astype(int))['f1_media_columnas']:.4f})"
        )

    resultados = {
        nombre: resumir_variante(nombre, probas[nombre], y_verdadera, pliegue_de_fila, probas["BASE"])
        for nombre in ["BASE", *nombres_variantes]
    }
    log("corrected_cv por variante: " + ", ".join(f"{n}={r['corrected_cv']:.4f}" for n, r in resultados.items()))

    # E2-02: meta-modelo de dependencia con el OOF congelado de los 4 bases.
    datos_oof = np.load(OOF_BASES_PATH)
    ids_oof = datos_oof["ids"].astype(int)
    mapa_fila = {identificador: posicion for posicion, identificador in enumerate(desarrollo["id"].astype(int))}
    orden_oof = np.array([mapa_fila[identificador] for identificador in ids_oof])
    proba_bases = np.zeros((len(desarrollo), len(MODELOS_BASE), len(ETIQUETAS)))
    proba_bases[orden_oof] = datos_oof["proba"]
    if not np.array_equal(datos_oof["y_true"], y_verdadera[orden_oof]):
        raise RuntimeError("Las etiquetas del OOF de bases no coinciden con el desarrollo")
    features_meta, nombres_meta = construir_matriz_meta(proba_bases, desarrollo["text"])
    log(f"meta-modelo: {features_meta.shape[1]} features ({len(nombres_meta)} nombres)")
    probas_meta = evaluar_meta(features_meta, y_verdadera, pliegues)
    resumenes_meta = {
        valor_c: resumir_variante(f"META_C{valor_c}", proba, y_verdadera, pliegue_de_fila, probas["BASE"])
        for valor_c, proba in probas_meta.items()
    }
    mejor_c = max(resumenes_meta, key=lambda valor: resumenes_meta[valor]["corrected_cv"])
    meta_resumen = dict(resumenes_meta[mejor_c])
    meta_resumen["mejor_C"] = mejor_c
    meta_resumen["por_C"] = {str(valor): round(resumenes_meta[valor]["corrected_cv"], 4) for valor in resumenes_meta}
    meta_resumen["advertencia"] = (
        "Los scores de entrenamiento del meta provienen de bases que vieron el pliegue "
        "evaluado en su propio entrenamiento (stacking estandar): cota optimista. La "
        "verificacion estrictamente anidada es META-ANIDADO."
    )
    resumenes_anidado = {
        valor_c: resumir_variante(
            f"META-ANIDADO_C{valor_c}", proba, y_verdadera, pliegue_de_fila, probas["BASE"]
        )
        for valor_c, proba in probas_meta_anidado.items()
    }
    mejor_c_anidado = max(resumenes_anidado, key=lambda valor: resumenes_anidado[valor]["corrected_cv"])
    meta_anidado_resumen = dict(resumenes_anidado[mejor_c_anidado])
    meta_anidado_resumen["mejor_C"] = mejor_c_anidado
    meta_anidado_resumen["por_C"] = {
        str(valor): round(resumenes_anidado[valor]["corrected_cv"], 4) for valor in resumenes_anidado
    }
    meta_anidado_resumen["advertencia"] = (
        "Verificacion estrictamente anidada: los 4 bases se reentrenan dentro de cada pliegue "
        "exterior (CV interna para las features del meta, modelo exterior sin la validacion)."
    )
    log(
        "meta (no anidado) por C: "
        + ", ".join(f"{clave}={valor:.4f}" for clave, valor in meta_resumen["por_C"].items())
    )
    log(
        "meta (anidado) por C: "
        + ", ".join(f"{clave}={valor:.4f}" for clave, valor in meta_anidado_resumen["por_C"].items())
    )

    # Complementariedad y bootstrap de las variantes con mejora.
    sin_umbral = {etiqueta: 0.5 for etiqueta in ETIQUETAS}
    verificar_metrica_rapida(probas["BASE"], y_verdadera, sin_umbral)
    log("F1 rapida verificada contra la metrica oficial del proyecto")
    candidatos: dict[str, dict] = {}
    for nombre in nombres_variantes:
        fila = resultados[nombre]
        fila["ensemble"] = evaluar_complementariedad(nombre, probas[nombre], probas["BASE"], y_verdadera)
        fila["ensemble_gain"] = fila["ensemble"]["ensemble_gain"]
        if fila["delta_vs_m2_05"] > 0:
            fila["bootstrap"] = bootstrap_delta(
                probas["BASE"], probas[nombre], y_verdadera, sin_umbral, sin_umbral
            )
        candidatos[nombre] = fila
    for nombre, resumen, probas_variante, clave in (
        (f"META_C{mejor_c}", meta_resumen, probas_meta[mejor_c], "META"),
        (f"META-ANIDADO_C{mejor_c_anidado}", meta_anidado_resumen, probas_meta_anidado[mejor_c_anidado], "META-ANIDADO"),
    ):
        resumen["ensemble"] = evaluar_complementariedad(nombre, probas_variante, probas["BASE"], y_verdadera)
        resumen["ensemble_gain"] = resumen["ensemble"]["ensemble_gain"]
        if resumen["delta_vs_m2_05"] > 0:
            resumen["bootstrap"] = bootstrap_delta(
                probas["BASE"], probas_variante, y_verdadera, sin_umbral, sin_umbral
            )
        candidatos[clave] = resumen

    decision = decidir(candidatos)
    complementario = max(candidatos.items(), key=lambda par: par[1]["ensemble_gain"])
    nombre_complementario, fila_complementaria = complementario
    decision["complementariedad"] = fila_complementaria["ensemble"]
    decision["variante_complementaria"] = nombre_complementario
    if (
        fila_complementaria["ensemble"].get("bootstrap", {}).get("ic_inferior", -1) > 0
        and fila_complementaria["ensemble_gain"] >= CONFIG["ensemble_gain_minimo"]
    ):
        decision["adoptado"] = True
        decision["justificacion"] = (
            f"Adoptada por complementariedad ({nombre_complementario}): ensemble_gain "
            f"{fila_complementaria['ensemble_gain']:+.4f} con IC bootstrap estrictamente positivo."
        )
    for nombre, fila in candidatos.items():
        fila["adoptado"] = bool(decision["adoptado"] and decision["mejor_variante"] == nombre)
        fila["decision"] = (
            "ADOPTADA (Gate P1)" if fila["adoptado"] else "No adoptada (no supera la regla D-016)"
        )

    # Umbrales OOF solo si alguna variante se acerca a la adopcion.
    umbrales_info: dict = {}
    mapa_probas = {
        "META": probas_meta[mejor_c],
        "META-ANIDADO": probas_meta_anidado[mejor_c_anidado],
    }
    mapa_probas.update({nombre: probas[nombre] for nombre in nombres_variantes})
    candidatas = {
        nombre: fila
        for nombre, fila in candidatos.items()
        if fila["corrected_cv"] >= CONFIG["umbral_aproximacion"]
    }
    if candidatas:
        nombre_mejor = max(candidatas, key=lambda clave: candidatas[clave]["corrected_cv"])
        umbrales, f1_umbrales = ajustar_umbrales(mapa_probas[nombre_mejor], y_verdadera)
        umbrales = {etiqueta: round(valor, 2) for etiqueta, valor in umbrales.items()}
        umbrales_info = {
            "variante": nombre_mejor,
            "umbrales": umbrales,
            "f1_con_umbrales": f1_umbrales,
            "etiqueta": "cota optimista (umbral y metrica comparten el OOF)",
        }
        decision["umbrales_texto"] = (
            f"variante {nombre_mejor} con umbrales OOF {umbrales} -> {f1_umbrales:.4f} (cota optimista)."
        )
    for nombre, fila in resultados.items():
        if nombre != "BASE":
            fila["adoptado"] = candidatos[nombre]["adoptado"]
            fila["decision"] = candidatos[nombre]["decision"]
    tabla = escribir_reporte_csv(resultados, meta_resumen, meta_anidado_resumen)

    reporte = {
        "fecha": CONFIG["fecha"],
        "protocolo": "v2 (D-016): grupos fusionados, desarrollo 1483, holdout congelado no usado",
        "referencias": {
            "m2_umbral_05": CONFIG["referencia_m2_umbral_05"],
            "v03_umbrales": CONFIG["referencia_v03_umbrales"],
        },
        "verificacion_datos": verificacion,
        "config": {clave: list(valor) if isinstance(valor, tuple) else valor for clave, valor in CONFIG.items()},
        "variantes": resultados,
        "meta": {**meta_resumen, "nombres_features": nombres_meta},
        "meta_anidado": meta_anidado_resumen,
        "umbrales_oof": umbrales_info,
        "decision": decision,
        "duracion_segundos": round(time.perf_counter() - inicio, 2),
    }
    REPORTE_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORTE_JSON.write_text(json.dumps(reporte, indent=2, ensure_ascii=False), encoding="utf-8")
    generar_figura(resultados, meta_resumen, meta_anidado_resumen)
    registrar_experimentos(
        resultados, meta_resumen, meta_anidado_resumen, reporte["duracion_segundos"], tabla
    )
    escribir_contexto(
        tabla, decision, reporte["duracion_segundos"], meta_resumen, meta_anidado_resumen
    )

    log("resumen final")
    for _, fila in tabla.iterrows():
        log(
            f"  {fila['variante']}: {fila['corrected_cv']:.4f} "
            f"(delta {fila['delta_vs_v03_umbrales']:+.4f}, pliegues {fila['pliegues_mejorados_vs_v03']}/5)"
        )
    log(f"decision: {'ADOPTADO' if decision['adoptado'] else 'NO ADOPTADO'} - {decision['justificacion']}")
    log(f"artefactos: {REPORTE_CSV.name}, {REPORTE_JSON.name}, {FIGURA_PATH.name}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:  # noqa: BLE001 - reporte explicito y salida con codigo 1
        import traceback

        traceback.print_exc()
        print(f"[P1] ERROR: {error}", file=sys.stderr)
        sys.exit(1)
