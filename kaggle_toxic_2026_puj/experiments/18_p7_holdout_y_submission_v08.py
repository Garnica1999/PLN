"""P7 - Validacion del candidato v2 en el holdout congelado y submission v08.

Pregunta que responde
---------------------
El Gate P6 adopto el meta META-BASES-RIDGE (Ridge alpha=3 sobre las 44
meta-features OOF de los 4 bases) por complementariedad frente al candidato
v2-A (D-019). Antes de gastar una sonda de leaderboard (maximo 3, D-016; D-014
la reserva a finalistas), P7 exige validar los finalistas en el holdout
congelado (248 tweets; maximo 5 evaluaciones en todo el proyecto). ¿Confirma el
holdout la mejora del meta sobre V03 y justifica generar la submission v08?

Metodo
------
1. Contador de evaluaciones: `reports/holdout_evaluaciones_v2.csv` registra
   fecha, experimento, motivo y resultado de cada evaluacion al holdout; esta
   tarea consume 2 (P7-01 V03 de referencia y P7-02 meta candidato).
2. Paso 1 (referencia): se reajusta M2 (union TF-IDF word(1,2) + char_wb(3,5),
   min_df=2, sublinear_tf; LR balanced C=2.0; receta de `experiments/14`) con
   TODO el desarrollo, se predice el holdout y se aplican los umbrales
   congelados 0.44/0.51/0.46.
3. Paso 2 (candidato): el meta Ridge (alpha=3, balanced) se ajusta sobre las
   meta-features OOF del desarrollo (44 columnas: probabilidades, margenes y
   entropias de los 4 bases + 8 rasgos estructurales, exactamente como en
   `experiments/17`); los 4 bases se reajustan con TODO el desarrollo, con
   ellos se construyen las meta-features del holdout y se predice a 0.5. El
   ajuste se verifica contra `models/v2_meta_candidato.joblib`.
4. Decision pre-registrada: se genera la submission solo si la F1 del meta en
   el holdout es >= F1 de V03 en el holdout - 0.005.
5. Paso 3 (submission): si la decision es GO, los 4 bases se reajustan con TODO
   el train (desarrollo + holdout), se construyen las meta-features del test y
   se aplica el meta (umbral 0.5). El test SOLO se usa para predecir.

Reglas de honestidad
--------------------
- El holdout nunca se usa para ajustar: solo para medir los 2 finalistas; los
  umbrales y el alpha del meta estan congelados de fases anteriores.
- El meta se ajusta exclusivamente con el OOF del desarrollo; las features del
  holdout usan bases ajustados con el desarrollo y las del test, bases
  ajustados con el train completo (stacking estandar).
- Si el holdout no confirma el criterio, NO se genera submission y se reporta
  el resultado con los numeros.

Salidas
-------
`reports/p7_holdout_resultados.json`, `reports/holdout_evaluaciones_v2.csv`,
`models/v2_meta_final.joblib` y `submissions/toxic_submission_v08_meta_v2.csv`
(solo si GO), seccion "## Fase v2 - P7 (2026-09-25)" en `reports/contexto.md`,
actualizacion de P7.1-P7.3 y Gate P7 en `tasks/todo.md`, fila `S08` en
`experiments.csv` y entrada `v08_meta_v2` en `submissions/leaderboard_log.csv`
con `kaggle_f1` vacio: la sonda la envia Carlos.

Idempotencia
------------
Particiones, semillas y recetas son deterministas; el contador de holdout, la
submission y los registros se reescriben por identificador, por lo que dos
ejecuciones dejan el mismo contenido salvo los tiempos.

Ejecucion
---------
    /home/carlos/venv/bin/python experiments/18_p7_holdout_y_submission_v08.py
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import joblib
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import sparse
from scipy.special import expit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src import data as data_module
from src.evaluation import metricas_multietiqueta
from src.splits import ETIQUETAS

PARTICION_PATH = PROJECT_ROOT / "data" / "processed" / "holdout_v2_ids.csv"
HOLDOUT_CONFIG_PATH = PROJECT_ROOT / "config" / "holdout_v2.json"
OOF_BASES_PATH = PROJECT_ROOT / "features" / "v2_oof_base_corrected.npz"
META_CANDIDATO_PATH = PROJECT_ROOT / "models" / "v2_meta_candidato.joblib"
REPORTE_JSON = PROJECT_ROOT / "reports" / "p7_holdout_resultados.json"
EVALUACIONES_PATH = PROJECT_ROOT / "reports" / "holdout_evaluaciones_v2.csv"
MODELO_FINAL_PATH = PROJECT_ROOT / "models" / "v2_meta_final.joblib"
SUBMISSION_PATH = PROJECT_ROOT / "submissions" / "toxic_submission_v08_meta_v2.csv"
EXPERIMENTS_CSV = PROJECT_ROOT / "experiments.csv"
LEADERBOARD_CSV = PROJECT_ROOT / "submissions" / "leaderboard_log.csv"
CONTEXTO_MD = PROJECT_ROOT / "reports" / "contexto.md"
TODO_MD = PROJECT_ROOT / "tasks" / "todo.md"

CONFIG: Final[dict[str, Any]] = {
    "fecha": "2026-09-25",
    "n_dev": 1483,
    "n_holdout": 248,
    "max_evaluaciones_holdout": 5,
    "evaluaciones_de_esta_tarea": 2,
    "tolerancia_decision": 0.005,
    "alpha_meta": 3.0,
    "umbral_meta": 0.5,
    "n_filas_test": 1154,
    "columnas_submission": ("id", "HS", "TR", "AG"),
    # Estimacion E00: el test tiene ~7 % menos positivos que train (0.93x).
    "prevalencia_test_estimada": {"HS": 0.406, "TR": 0.279, "AG": 0.356},
    "referencias_desarrollo": {"v03_umbrales": 0.7772, "meta_ridge_alpha3": 0.7874},
    "tolerancia_reproduccion_candidato": 1e-9,
}

COLUMNAS_EXPERIMENTS: Final[tuple[str, ...]] = (
    "experiment_id",
    "competition",
    "question",
    "hypothesis",
    "preprocessing",
    "feature_groups",
    "representation",
    "model",
    "hyperparameters",
    "validation",
    "f1",
    "macro_f1",
    "weighted_f1",
    "precision",
    "recall",
    "training_time",
    "dimension",
    "result",
    "interpretation",
    "decision",
    "date",
    "script",
    "notes",
)
COLUMNAS_LEADERBOARD: Final[tuple[str, ...]] = (
    "submission_id",
    "fecha",
    "archivo",
    "local_f1",
    "kaggle_f1",
    "diferencia",
    "features",
    "modelo",
    "hiperparametros",
    "interpretacion",
)
COLUMNAS_EVALUACIONES: Final[tuple[str, ...]] = (
    "fecha",
    "experimento",
    "motivo",
    "resultado",
)

ID_EVALUACION_V03: Final[str] = "P7-01-V03"
ID_EVALUACION_META: Final[str] = "P7-02-META"
ID_EXPERIMENTO_SUBMISSION: Final[str] = "S08"
ID_LEADERBOARD: Final[str] = "v08_meta_v2"


@dataclass(frozen=True)
class BaseAjustado:
    """Modelo base reajustado con todo un conjunto de textos.

    Attributes:
        nombre: nombre del base en la receta de ``experiments/14``.
        tipo: ``lr`` (LogisticRegression, probabilidades) o ``svc`` (LinearSVC,
            sigmoide de la decision).
        representacion: ``word``, ``char`` o ``union``.
        vectorizadores: vectorizadores ajustados por bloque (``word``/``char``).
        modelos: modelo por etiqueta en el orden de :data:`ETIQUETAS`.
        dimension: numero de columnas de la matriz resultante.
    """

    nombre: str
    tipo: str
    representacion: str
    vectorizadores: dict[str, TfidfVectorizer]
    modelos: dict[str, Any]
    dimension: int


def log(mensaje: str) -> None:
    """Imprime un mensaje de progreso con prefijo de fase.

    Args:
        mensaje: texto a imprimir.
    """
    print(f"[P7] {mensaje}", flush=True)


# ===== SECCION 1: modulos base, particion y carga de datos =====


def cargar_modulo(nombre: str, ruta: Path) -> Any:
    """Importa un script de experimento sin ejecutar su ``main``.

    Args:
        nombre: nombre unico de modulo con el que registrarlo.
        ruta: ruta del script a cargar.

    Returns:
        Modulo cargado.

    Raises:
        ImportError: si la ruta no se puede cargar.
    """
    especificacion = importlib.util.spec_from_file_location(nombre, ruta)
    if especificacion is None or especificacion.loader is None:
        raise ImportError(f"No se pudo cargar {ruta}")
    modulo = importlib.util.module_from_spec(especificacion)
    # Registrar en sys.modules es necesario para que dataclasses/typing
    # resuelvan el modulo de origen durante exec_module.
    sys.modules[especificacion.name] = modulo
    especificacion.loader.exec_module(modulo)
    return modulo


def sha256_de(ruta: Path) -> str:
    """Calcula el sha256 de un archivo.

    Args:
        ruta: ruta del archivo.

    Returns:
        Hash hexadecimal.
    """
    return hashlib.sha256(ruta.read_bytes()).hexdigest()


def verificar_particion(particion: pd.DataFrame, train: pd.DataFrame) -> dict[str, Any]:
    """Verifica el hash, los conteos y las etiquetas de la particion congelada.

    Args:
        particion: CSV de la particion congelada (``id``, ``particion``, ...).
        train: train completo con ``id`` y etiquetas.

    Returns:
        Diccionario con el resumen de verificacion (hash, conteos y enfoque).

    Raises:
        RuntimeError: si el hash o los conteos no coinciden con lo congelado.
        ValueError: si las etiquetas de la particion no coinciden con el train.
    """
    config = json.loads(HOLDOUT_CONFIG_PATH.read_text(encoding="utf-8"))
    hash_actual = sha256_de(PARTICION_PATH)
    if hash_actual != config["hash_ids_csv"]:
        raise RuntimeError(
            f"El hash de {PARTICION_PATH.name} no coincide con config/holdout_v2.json: "
            f"{hash_actual} vs {config['hash_ids_csv']}"
        )
    conteos = particion["particion"].value_counts().to_dict()
    if conteos.get("dev") != CONFIG["n_dev"] or conteos.get("holdout") != CONFIG["n_holdout"]:
        raise RuntimeError(f"Conteos de la particion inesperados: {conteos}")
    if not particion["id"].is_unique:
        raise RuntimeError("La particion congelada tiene ids duplicados")
    if set(particion["id"].astype(int)) != set(train["id"].astype(int)):
        raise RuntimeError("La particion congelada no cubre exactamente los ids del train")
    mapa_etiquetas = {
        int(identificador): (int(hs), int(tr), int(ag))
        for identificador, hs, tr, ag in zip(train["id"], train["HS"], train["TR"], train["AG"])
    }
    for fila in particion.itertuples():
        etiquetas_particion = (int(fila.HS), int(fila.TR), int(fila.AG))
        if mapa_etiquetas[int(fila.id)] != etiquetas_particion:
            raise ValueError(f"Etiquetas inconsistentes en la particion para el id {fila.id}")
    return {
        "hash_ids_csv": hash_actual,
        "hash_coincide": True,
        "n_dev": int(conteos["dev"]),
        "n_holdout": int(conteos["holdout"]),
        "n_grupos_total": int(config.get("n_grupos_total", 0)),
    }


def cargar_datos() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Carga train, desarrollo, holdout, test y la plantilla oficial.

    Returns:
        Tupla ``(train, desarrollo, holdout, test, plantilla)``. El desarrollo y
        el holdout se reconstruyen desde la particion congelada (mismo orden de
        filas que ``train``).

    Raises:
        RuntimeError: si los tamanos o el orden de ids del test no cuadran.
    """
    train = data_module.read_train()
    test = data_module.read_test()
    plantilla = data_module.read_submission()
    particion = pd.read_csv(PARTICION_PATH)
    verificacion = verificar_particion(particion, train)
    ids_dev = set(particion.loc[particion["particion"] == "dev", "id"].astype(int))
    ids_holdout = set(particion.loc[particion["particion"] == "holdout", "id"].astype(int))
    desarrollo = train[train["id"].astype(int).isin(ids_dev)].reset_index(drop=True)
    holdout = train[train["id"].astype(int).isin(ids_holdout)].reset_index(drop=True)
    if len(desarrollo) != CONFIG["n_dev"] or len(holdout) != CONFIG["n_holdout"]:
        raise RuntimeError(
            f"Tamano inesperado: desarrollo {len(desarrollo)}, holdout {len(holdout)}"
        )
    if list(plantilla["id"]) != list(test["id"]):
        raise RuntimeError("El orden de ids del test no coincide con la plantilla oficial")
    if len(test) != CONFIG["n_filas_test"]:
        raise RuntimeError(f"Test inesperado: {len(test)} filas")
    log(
        f"datos: train {train.shape} | desarrollo {len(desarrollo)} | holdout {len(holdout)} | "
        f"test {len(test)} | particion verificada ({verificacion['hash_ids_csv'][:12]}...)"
    )
    return train, desarrollo, holdout, test, plantilla


# ===== SECCION 2: bases (receta congelada de experiments/14) =====


def ajustar_vectorizadores(
    textos: pd.Series, representacion: str, modulo_14: Any
) -> dict[str, TfidfVectorizer]:
    """Ajusta los vectorizadores de una representacion con los textos dados.

    Args:
        textos: textos de ajuste (todo el desarrollo o todo el train).
        representacion: ``word``, ``char`` o ``union``.
        modulo_14: modulo de ``experiments/14`` con ``PARAMS_WORD`` y
            ``PARAMS_CHAR``.

    Returns:
        Diccionario ``bloque -> vectorizador ajustado``.
    """
    if representacion == "word":
        return {"word": TfidfVectorizer(**modulo_14.PARAMS_WORD).fit(textos)}
    if representacion == "char":
        return {"char": TfidfVectorizer(**modulo_14.PARAMS_CHAR).fit(textos)}
    return {
        "word": TfidfVectorizer(**modulo_14.PARAMS_WORD).fit(textos),
        "char": TfidfVectorizer(**modulo_14.PARAMS_CHAR).fit(textos),
    }


def transformar_textos(
    vectorizadores: dict[str, TfidfVectorizer], textos: pd.Series, representacion: str
):
    """Transforma textos con los vectorizadores ajustados.

    Args:
        vectorizadores: diccionario devuelto por :func:`ajustar_vectorizadores`.
        textos: textos a transformar.
        representacion: ``word``, ``char`` o ``union``.

    Returns:
        Matriz dispersa (una o la concatenacion word+char de los bloques).
    """
    if representacion == "word":
        return vectorizadores["word"].transform(textos)
    if representacion == "char":
        return vectorizadores["char"].transform(textos)
    return sparse.hstack(
        [vectorizadores["word"].transform(textos), vectorizadores["char"].transform(textos)]
    ).tocsr()


def ajustar_base(
    nombre: str, textos: pd.Series, y: NDArray[np.int_], modulo_14: Any
) -> BaseAjustado:
    """Reajusta un modelo base con la receta congelada de ``experiments/14``.

    Args:
        nombre: clave de ``modulo_14.MODELOS``.
        textos: textos de ajuste.
        y: etiquetas ``(n, 3)`` en el orden de :data:`ETIQUETAS`.
        modulo_14: modulo de ``experiments/14``.

    Returns:
        Modelo base reajustado con sus vectorizadores y modelos.
    """
    especificacion = modulo_14.MODELOS[nombre]
    representacion = str(especificacion["representacion"])
    vectorizadores = ajustar_vectorizadores(textos, representacion, modulo_14)
    matriz = transformar_textos(vectorizadores, textos, representacion)
    modelos: dict[str, Any] = {}
    for posicion, etiqueta in enumerate(ETIQUETAS):
        parametros = dict(especificacion["params"])
        if especificacion["tipo"] == "lr":
            modelo: Any = LogisticRegression(**parametros)
        else:
            modelo = LinearSVC(**parametros)
        modelo.fit(matriz, y[:, posicion])
        modelos[etiqueta] = modelo
    return BaseAjustado(
        nombre=nombre,
        tipo=str(especificacion["tipo"]),
        representacion=representacion,
        vectorizadores=vectorizadores,
        modelos=modelos,
        dimension=int(matriz.shape[1]),
    )


def predecir_base(base: BaseAjustado, textos: pd.Series) -> NDArray[np.float64]:
    """Predice las 3 etiquetas de un conjunto con un base reajustado.

    Args:
        base: modelo base reajustado.
        textos: textos a predecir.

    Returns:
        Matriz ``(n, 3)`` de probabilidades (LR) o sigmoide de la decision (SVC).
    """
    matriz = transformar_textos(base.vectorizadores, textos, base.representacion)
    columnas = []
    for etiqueta in ETIQUETAS:
        modelo = base.modelos[etiqueta]
        if base.tipo == "lr":
            columnas.append(modelo.predict_proba(matriz)[:, 1])
        else:
            columnas.append(expit(modelo.decision_function(matriz)))
    return np.column_stack(columnas)


def predecir_bases(
    bases: dict[str, BaseAjustado], textos: pd.Series
) -> tuple[NDArray[np.float64], tuple[str, ...]]:
    """Predice con todos los bases en el orden de la receta congelada.

    Args:
        bases: diccionario ``nombre -> base reajustado``.
        textos: textos a predecir.

    Returns:
        Tupla ``(proba, nombres)`` con forma ``(n, 4, 3)`` y el orden usado.
    """
    nombres = tuple(bases)
    proba = np.zeros((len(textos), len(nombres), len(ETIQUETAS)))
    for posicion, nombre in enumerate(nombres):
        proba[:, posicion, :] = predecir_base(bases[nombre], textos)
    return proba, nombres


def cargar_oof_dev(
    desarrollo: pd.DataFrame, y_dev: NDArray[np.int_], modulo_14: Any
) -> tuple[NDArray[np.float64], tuple[str, ...]]:
    """Carga y alinea el OOF congelado de los 4 bases sobre el desarrollo.

    Args:
        desarrollo: DataFrame del desarrollo.
        y_dev: etiquetas ``(n, 3)`` del desarrollo.
        modulo_14: modulo de ``experiments/14``.

    Returns:
        Tupla ``(proba, modelos)`` con forma ``(1483, 4, 3)``.

    Raises:
        RuntimeError: si el OOF no coincide con el desarrollo o con la receta.
    """
    datos = np.load(OOF_BASES_PATH)
    modelos = tuple(str(valor) for valor in datos["modelos"])
    if modelos != tuple(modulo_14.MODELOS):
        raise RuntimeError(f"Modelos del OOF inesperados: {modelos}")
    if tuple(str(valor) for valor in datos["etiquetas"]) != tuple(ETIQUETAS):
        raise RuntimeError(f"Etiquetas del OOF inesperadas: {datos['etiquetas']}")
    mapa_fila = {
        identificador: posicion
        for posicion, identificador in enumerate(desarrollo["id"].astype(int))
    }
    ids = datos["ids"].astype(int)
    orden = np.array([mapa_fila[identificador] for identificador in ids])
    if not np.array_equal(datos["y_true"], y_dev[orden]):
        raise RuntimeError("Las etiquetas del OOF de bases no coinciden con el desarrollo")
    proba = np.zeros((len(desarrollo), len(modelos), len(ETIQUETAS)))
    proba[orden] = datos["proba"]
    return proba, modelos


# ===== SECCION 3: meta-modelo (construccion exacta de experiments/17) =====


def ajustar_meta(
    matriz_dev: NDArray[np.float64], y_dev: NDArray[np.int_]
) -> tuple[StandardScaler, dict[str, RidgeClassifier]]:
    """Ajusta el meta Ridge (alpha=3) sobre las meta-features OOF del desarrollo.

    Replica el ajuste de ``guardar_artefactos_candidato`` de ``experiments/17``:
    escalador estandar ajustado con todo el desarrollo y un RidgeClassifier
    balanceado por etiqueta.

    Args:
        matriz_dev: meta-features ``(n, 44)`` del desarrollo.
        y_dev: etiquetas ``(n, 3)`` del desarrollo.

    Returns:
        Tupla ``(escalador, modelos)`` con un modelo por etiqueta.
    """
    escalador = StandardScaler().fit(matriz_dev)
    matriz_escalada = escalador.transform(matriz_dev)
    modelos: dict[str, RidgeClassifier] = {}
    for posicion, etiqueta in enumerate(ETIQUETAS):
        modelo = RidgeClassifier(alpha=CONFIG["alpha_meta"], class_weight="balanced")
        modelo.fit(matriz_escalada, y_dev[:, posicion])
        modelos[etiqueta] = modelo
    return escalador, modelos


def predecir_meta(
    escalador: StandardScaler,
    modelos: dict[str, RidgeClassifier],
    matriz: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Predice con el meta aplicando el escalador y la sigmoide de la decision.

    Args:
        escalador: escalador ajustado con el desarrollo.
        modelos: modelo por etiqueta.
        matriz: meta-features a predecir.

    Returns:
        Matriz ``(n, 3)`` de probabilidades.
    """
    matriz_escalada = escalador.transform(matriz)
    return np.column_stack(
        [expit(modelos[etiqueta].decision_function(matriz_escalada)) for etiqueta in ETIQUETAS]
    )


def verificar_reproduccion_candidato(
    escalador: StandardScaler,
    modelos: dict[str, RidgeClassifier],
    matriz_dev: NDArray[np.float64],
    nombres: list[str],
) -> dict[str, Any]:
    """Comprueba que el meta reajustado reproduce ``v2_meta_candidato.joblib``.

    Args:
        escalador: escalador reajustado.
        modelos: modelos reajustados por etiqueta.
        matriz_dev: meta-features del desarrollo.
        nombres: nombres de las 44 features del meta.

    Returns:
        Diccionario con la verificacion y la diferencia maxima en el desarrollo.

    Raises:
        RuntimeError: si los nombres o las probabilidades no coinciden con el
            artefacto del Gate P6.
    """
    candidato = joblib.load(META_CANDIDATO_PATH)
    if list(candidato["nombres_features"]) != list(nombres):
        raise RuntimeError("Los nombres de las meta-features no coinciden con el candidato de P6")
    if candidato["modelo"] != "ridge" or float(candidato["mejor_parametro"]) != CONFIG["alpha_meta"]:
        raise RuntimeError(
            f"El candidato de P6 no es el esperado: {candidato['modelo']} "
            f"{candidato['mejor_parametro']}"
        )
    matriz_escalada = candidato["escalador"].transform(matriz_dev)
    proba_candidato = np.column_stack(
        [
            expit(candidato["modelos"][etiqueta].decision_function(matriz_escalada))
            for etiqueta in ETIQUETAS
        ]
    )
    proba_nueva = predecir_meta(escalador, modelos, matriz_dev)
    diferencia = float(np.max(np.abs(proba_candidato - proba_nueva)))
    if diferencia > CONFIG["tolerancia_reproduccion_candidato"]:
        raise RuntimeError(
            f"El meta reajustado no reproduce el candidato de P6 (dif max {diferencia:.3e})"
        )
    log(
        f"meta verificado contra models/v2_meta_candidato.joblib "
        f"(dif max en desarrollo {diferencia:.3e})"
    )
    return {
        "variante": str(candidato["variante"]),
        "alpha": float(candidato["mejor_parametro"]),
        "n_features": len(nombres),
        "diferencia_maxima_dev": diferencia,
        "reproduce": True,
    }


# ===== SECCION 4: metricas, decision y contador del holdout =====


def vector_umbrales(umbrales: dict[str, float]) -> NDArray[np.float64]:
    """Convierte un diccionario de umbrales en vector en el orden de etiquetas.

    Args:
        umbrales: umbral por etiqueta.

    Returns:
        Arreglo ``(3,)``.
    """
    return np.array([umbrales[etiqueta] for etiqueta in ETIQUETAS])


def resumir_proba(
    proba: NDArray[np.float64], y_verdadera: NDArray[np.int_], umbrales: NDArray[np.float64]
) -> dict[str, Any]:
    """Calcula el resumen de metricas de una matriz de probabilidades.

    Args:
        proba: probabilidades ``(n, 3)``.
        y_verdadera: etiquetas verdaderas ``(n, 3)``.
        umbrales: vector de umbrales por etiqueta.

    Returns:
        Diccionario con F1 global, por etiqueta, secundarias y conteos.
    """
    prediccion = (proba >= umbrales[None, :]).astype(int)
    metrica = metricas_multietiqueta(y_verdadera, prediccion)
    return {
        "f1_media_columnas": float(metrica["f1_media_columnas"]),
        "f1_por_etiqueta": {etiqueta: float(valor) for etiqueta, valor in metrica["f1_por_etiqueta"].items()},
        "f1_macro6": float(metrica["f1_macro6"]),
        "f1_micro": float(metrica["f1_micro"]),
        "f1_samples": float(metrica["f1_samples"]),
        "exact_match": float(metrica["exact_match"]),
        "positivos_predichos": {
            etiqueta: int(prediccion[:, posicion].sum()) for posicion, etiqueta in enumerate(ETIQUETAS)
        },
        "positivos_reales": {
            etiqueta: int(y_verdadera[:, posicion].sum()) for posicion, etiqueta in enumerate(ETIQUETAS)
        },
    }


def cargar_contador_holdout() -> pd.DataFrame:
    """Carga (o crea vacio) el contador de evaluaciones al holdout.

    Returns:
        DataFrame con las columnas :data:`COLUMNAS_EVALUACIONES`.

    Raises:
        ValueError: si el esquema del archivo existente no coincide.
    """
    if EVALUACIONES_PATH.exists():
        contador = pd.read_csv(EVALUACIONES_PATH, dtype=str, keep_default_na=False)
        if list(contador.columns) != list(COLUMNAS_EVALUACIONES):
            raise ValueError(
                f"El esquema de {EVALUACIONES_PATH.name} cambio inesperadamente: "
                f"{list(contador.columns)}"
            )
        return contador
    return pd.DataFrame(columns=list(COLUMNAS_EVALUACIONES))


def registrar_evaluaciones_holdout(filas: list[dict[str, str]]) -> tuple[int, int]:
    """Registra las evaluaciones de esta tarea y verifica el presupuesto global.

    Args:
        filas: filas nuevas (fecha, experimento, motivo, resultado).

    Returns:
        Tupla ``(evaluaciones_antes, evaluaciones_despues)``.

    Raises:
        RuntimeError: si el contador supera el maximo de 5 evaluaciones.
    """
    contador = cargar_contador_holdout()
    identificadores = {fila["experimento"] for fila in filas}
    ajenas = contador[~contador["experimento"].isin(identificadores)]
    antes = len(ajenas)
    despues = antes + len(filas)
    if despues > CONFIG["max_evaluaciones_holdout"]:
        raise RuntimeError(
            f"El contador de holdout quedaria en {despues} > "
            f"{CONFIG['max_evaluaciones_holdout']} evaluaciones; abortado sin tocar el holdout"
        )
    combinado = pd.concat([ajenas, pd.DataFrame(filas)], ignore_index=True)
    EVALUACIONES_PATH.parent.mkdir(parents=True, exist_ok=True)
    combinado.to_csv(EVALUACIONES_PATH, index=False, encoding="utf-8")
    log(
        f"contador de holdout: {antes} evaluaciones previas + {len(filas)} de esta tarea = "
        f"{despues}/{CONFIG['max_evaluaciones_holdout']}"
    )
    return antes, despues


# ===== SECCION 5: submission =====


def validar_submission(submission: pd.DataFrame, plantilla: pd.DataFrame, ruta: Path) -> None:
    """Valida una submission contra la plantilla oficial.

    Args:
        submission: DataFrame candidato ``id, HS, TR, AG``.
        plantilla: plantilla oficial ``data/raw/submission.csv``.
        ruta: ruta destino (solo para mensajes de error).

    Raises:
        ValueError: si el formato, los ids, los valores o los nulos no cumplen
            el contrato de la competencia.
    """
    if tuple(submission.columns) != CONFIG["columnas_submission"]:
        raise ValueError(f"{ruta.name}: columnas {tuple(submission.columns)} distintas de id,HS,TR,AG")
    if len(submission) != CONFIG["n_filas_test"]:
        raise ValueError(
            f"{ruta.name}: {len(submission)} filas frente a {CONFIG['n_filas_test']} esperadas"
        )
    if list(submission["id"]) != list(plantilla["id"]):
        raise ValueError(f"{ruta.name}: los ids no coinciden con la plantilla en contenido u orden")
    if submission.isna().any().any():
        raise ValueError(f"{ruta.name}: hay valores nulos")
    if not submission["id"].is_unique:
        raise ValueError(f"{ruta.name}: hay ids duplicados")
    etiquetas = submission[list(ETIQUETAS)].to_numpy()
    if etiquetas.min() < 0 or etiquetas.max() > 1 or not np.isin(etiquetas, (0, 1)).all():
        raise ValueError(f"{ruta.name}: las etiquetas deben ser enteras 0/1")


def escribir_submission(plantilla: pd.DataFrame, prediccion: NDArray[np.int_]) -> Path:
    """Construye, valida y escribe la submission v08.

    Args:
        plantilla: plantilla oficial (mismo orden de ids que el test).
        prediccion: matriz binaria ``(n, 3)`` en el orden de :data:`ETIQUETAS`.

    Returns:
        Ruta del archivo escrito.

    Raises:
        ValueError: si la submission no cumple el formato oficial.
    """
    salida = plantilla.copy()
    for posicion, etiqueta in enumerate(ETIQUETAS):
        salida[etiqueta] = prediccion[:, posicion]
    validar_submission(salida, plantilla, SUBMISSION_PATH)
    SUBMISSION_PATH.parent.mkdir(parents=True, exist_ok=True)
    salida.to_csv(SUBMISSION_PATH, index=False, encoding="utf-8")
    releida = pd.read_csv(SUBMISSION_PATH, dtype={"id": str})
    validar_submission(releida, plantilla, SUBMISSION_PATH)
    log(f"submission escrita y validada: {SUBMISSION_PATH.name} ({len(releida)} filas)")
    return SUBMISSION_PATH


def resumir_prediccion_test(
    train: pd.DataFrame, prediccion: NDArray[np.int_]
) -> dict[str, Any]:
    """Resume los positivos predichos en el test frente a las prevalencias.

    Args:
        train: train completo (aportando la prevalencia real).
        prediccion: matriz binaria del test.

    Returns:
        Diccionario con positivos, tasas, prevalencias y diferencias.
    """
    tasas_predichas = {
        etiqueta: float(prediccion[:, posicion].mean()) for posicion, etiqueta in enumerate(ETIQUETAS)
    }
    tasas_train = {etiqueta: float(train[etiqueta].mean()) for etiqueta in ETIQUETAS}
    return {
        "positivos_test": {
            etiqueta: int(prediccion[:, posicion].sum()) for posicion, etiqueta in enumerate(ETIQUETAS)
        },
        "tasa_positivos_test": tasas_predichas,
        "tasa_positivos_train": tasas_train,
        "tasa_test_estimada_e00": dict(CONFIG["prevalencia_test_estimada"]),
        "diferencia_vs_train": {
            etiqueta: tasas_predichas[etiqueta] - tasas_train[etiqueta] for etiqueta in ETIQUETAS
        },
        "diferencia_vs_test_estimado": {
            etiqueta: tasas_predichas[etiqueta] - CONFIG["prevalencia_test_estimada"][etiqueta]
            for etiqueta in ETIQUETAS
        },
    }


# ===== SECCION 6: registro (experiments, leaderboard, contexto, todo) =====


def registrar_en_experiments(fila: dict[str, Any]) -> None:
    """Anade (o refresca) la fila ``S08`` en ``experiments.csv``.

    Args:
        fila: fila completa con las columnas de ``experiments.csv``.

    Raises:
        ValueError: si el esquema del archivo no coincide con el esperado.
    """
    existente = pd.read_csv(EXPERIMENTS_CSV, dtype=str, keep_default_na=False)
    if list(existente.columns) != list(COLUMNAS_EXPERIMENTS):
        raise ValueError("El esquema de experiments.csv cambio inesperadamente")
    conservadas = existente[existente["experiment_id"] != fila["experiment_id"]]
    nueva = pd.DataFrame([{clave: fila[clave] for clave in COLUMNAS_EXPERIMENTS}])
    pd.concat([conservadas, nueva], ignore_index=True).to_csv(
        EXPERIMENTS_CSV, index=False, encoding="utf-8"
    )
    log(f"fila {fila['experiment_id']} registrada en experiments.csv")


def registrar_en_leaderboard(fila: dict[str, Any]) -> None:
    """Anade (o refresca) la entrada ``v08_meta_v2`` en ``leaderboard_log.csv``.

    Args:
        fila: fila con las columnas de ``submissions/leaderboard_log.csv``.

    Raises:
        ValueError: si el esquema del archivo no coincide con el esperado.
    """
    existente = pd.read_csv(LEADERBOARD_CSV, dtype=str, keep_default_na=False)
    if list(existente.columns) != list(COLUMNAS_LEADERBOARD):
        raise ValueError("El esquema de leaderboard_log.csv cambio inesperadamente")
    conservadas = existente[existente["submission_id"] != fila["submission_id"]]
    nueva = pd.DataFrame([{clave: fila[clave] for clave in COLUMNAS_LEADERBOARD}])
    pd.concat([conservadas, nueva], ignore_index=True).to_csv(
        LEADERBOARD_CSV, index=False, encoding="utf-8"
    )
    log(f"entrada {fila['submission_id']} registrada en leaderboard_log.csv (kaggle_f1 vacio)")


def escribir_contexto(lineas: list[str]) -> None:
    """Reemplaza o anade la seccion de P7 en ``reports/contexto.md``.

    Args:
        lineas: lineas de la seccion (sin encabezado).

    Raises:
        RuntimeError: si la seccion excede 20 lineas.
    """
    encabezado = "## Fase v2 - P7 (2026-09-25)"
    bloque = "\n".join([encabezado, *lineas])
    if len(bloque.splitlines()) > 20:
        raise RuntimeError(f"La seccion de P7 excede 20 lineas ({len(bloque.splitlines())})")
    texto = CONTEXTO_MD.read_text(encoding="utf-8")
    if encabezado in texto:
        inicio = texto.index(encabezado)
        fin = texto.find("\n## ", inicio + len(encabezado))
        fin = len(texto) if fin == -1 else fin + 1
        texto = texto[:inicio] + bloque + "\n" + texto[fin:]
    else:
        texto = texto.rstrip("\n") + "\n\n" + bloque + "\n"
    CONTEXTO_MD.write_text(texto, encoding="utf-8")
    log("seccion de P7 escrita en reports/contexto.md")


def construir_fila_evaluacion(
    identificador: str, motivo: str, resumen: dict[str, Any]
) -> dict[str, str]:
    """Construye una fila del contador de evaluaciones al holdout.

    Args:
        identificador: id de la evaluacion (``P7-01-V03`` o ``P7-02-META``).
        motivo: motivo de la evaluacion.
        resumen: resumen de metricas obtenido.

    Returns:
        Fila con las columnas de :data:`COLUMNAS_EVALUACIONES`.
    """
    resultado = {
        "holdout_f1_media_columnas": round(resumen["f1_media_columnas"], 4),
        "holdout_f1_por_etiqueta": {
            etiqueta: round(valor, 4) for etiqueta, valor in resumen["f1_por_etiqueta"].items()
        },
    }
    return {
        "fecha": CONFIG["fecha"],
        "experimento": identificador,
        "motivo": motivo,
        "resultado": json.dumps(resultado, ensure_ascii=False),
    }


def actualizar_todo(resultado: dict[str, Any]) -> None:
    """Actualiza P7.1-P7.3 y el Gate P7 en ``tasks/todo.md``.

    Args:
        resultado: resultado completo del experimento (para las cifras).
    """
    v03 = resultado["paso_1_v03_holdout"]["f1_media_columnas"]
    meta = resultado["paso_2_meta_holdout"]["f1_media_columnas"]
    delta = resultado["decision"]["delta_meta_vs_v03"]
    genera = bool(resultado["decision"]["submission_generada"])
    reemplazos = [
        (
            "- [pendiente] P7.1 Finalistas evaluados en el holdout congelado (max 3)",
            (
                "- [hecho] P7.1 Finalistas evaluados en el holdout congelado (2/5 evaluaciones, "
                f"P7-01 V03 y P7-02 meta): V03 {v03:.4f} vs meta {meta:.4f} (delta {delta:+.4f}; regla "
                f"pre-registrada meta >= V03 - {CONFIG['tolerancia_decision']:.3f}). Evidencia: "
                "`experiments/18_p7_holdout_y_submission_v08.py`, `reports/p7_holdout_resultados.json`, "
                "`reports/holdout_evaluaciones_v2.csv`"
            ),
        ),
        (
            "- [pendiente] P7.3 2-3 sondas de LB con hipotesis",
            (
                "- [en curso] P7.3 2-3 sondas de LB con hipotesis: submission "
                "`submissions/toxic_submission_v08_meta_v2.csv` "
                + (
                    "generada y validada (sonda v2 del meta Ridge alpha=3; la envia Carlos; "
                    "hipotesis: superar el LB de v03 0.78306)"
                    if genera
                    else "NO generada (el holdout no confirmo el criterio)"
                )
            ),
        ),
    ]
    texto = TODO_MD.read_text(encoding="utf-8")
    for viejo, nuevo in reemplazos:
        if viejo in texto:
            texto = texto.replace(viejo, nuevo)
    if genera:
        submission = resultado["paso_3_submission"]
        fila_gate = (
            f"| P7 | {CONFIG['fecha']} | GO (parcial). Holdout congelado (2/5 evaluaciones): V03 "
            f"{v03:.4f} vs meta Ridge alpha=3 {meta:.4f} (delta {delta:+.4f}); submission v08 generada "
            f"y validada (positivos HS {submission['positivos_test']['HS']}, TR "
            f"{submission['positivos_test']['TR']}, AG {submission['positivos_test']['AG']} de "
            f"{CONFIG['n_filas_test']}). Pendientes: auditoria P7.2 y resultado de la sonda de LB "
            "(la envia Carlos). Evidencia: `experiments/18_p7_holdout_y_submission_v08.py`, "
            "`reports/p7_holdout_resultados.json` |"
        )
    else:
        fila_gate = (
            f"| P7 | {CONFIG['fecha']} | NO-GO. Holdout congelado (2/5 evaluaciones): V03 {v03:.4f} vs "
            f"meta Ridge alpha=3 {meta:.4f} (delta {delta:+.4f} < -{CONFIG['tolerancia_decision']:.3f}); "
            "no se genera submission (ver `reports/p7_holdout_resultados.json`). Evidencia: "
            "`experiments/18_p7_holdout_y_submission_v08.py` |"
        )
    if "| P7 |" in texto:
        lineas = [linea for linea in texto.splitlines() if not linea.startswith("| P7 |")]
        texto = "\n".join(lineas).rstrip("\n") + "\n"
    texto = texto.rstrip("\n") + "\n" + fila_gate + "\n"
    TODO_MD.write_text(texto, encoding="utf-8")
    log("tasks/todo.md actualizado (P7.1, P7.3 y Gate P7)")


# ===== SECCION 7: main =====


def main() -> None:
    """Ejecuta P7: holdout de finalistas, decision y submission v08 si GO."""
    inicio = time.perf_counter()
    modulo_14 = cargar_modulo("experimento_14_p7", PROJECT_ROOT / "experiments" / "14_oof_corregido_v2.py")
    modulo_17 = cargar_modulo("experimento_17_p7", PROJECT_ROOT / "experiments" / "17_stacking_v2.py")
    if modulo_17.MODELOS_BASE != tuple(modulo_14.MODELOS):
        raise RuntimeError("El orden de modelos de experiments/17 no coincide con experiments/14")
    train, desarrollo, holdout, test, plantilla = cargar_datos()
    y_dev = desarrollo[list(ETIQUETAS)].to_numpy()
    y_holdout = holdout[list(ETIQUETAS)].to_numpy()
    log(
        "referencias de desarrollo: V03 con umbrales "
        f"{CONFIG['referencias_desarrollo']['v03_umbrales']:.4f} | meta Ridge alpha=3 "
        f"{CONFIG['referencias_desarrollo']['meta_ridge_alpha3']:.4f}"
    )

    # Paso 1 + 2a: los 4 bases se reajustan con TODO el desarrollo.
    bases_dev: dict[str, BaseAjustado] = {}
    for nombre in modulo_14.MODELOS:
        inicio_base = time.perf_counter()
        bases_dev[nombre] = ajustar_base(nombre, desarrollo["text"], y_dev, modulo_14)
        log(
            f"base {nombre} reajustado con el desarrollo (dim {bases_dev[nombre].dimension}) en "
            f"{time.perf_counter() - inicio_base:.1f} s"
        )
    proba_bases_holdout, nombres_bases = predecir_bases(bases_dev, holdout["text"])

    # Paso 1: V03 de referencia en el holdout (M2 con los umbrales congelados).
    umbrales_v03 = modulo_14.UMBRALES_V03
    posicion_m2 = nombres_bases.index("M2_union_lr_c2")
    proba_v03_holdout = proba_bases_holdout[:, posicion_m2, :]
    resumen_v03 = resumir_proba(proba_v03_holdout, y_holdout, vector_umbrales(umbrales_v03))
    resumen_v03["f1_a_0_5"] = resumir_proba(
        proba_v03_holdout, y_holdout, np.full(len(ETIQUETAS), 0.5)
    )["f1_media_columnas"]
    log(
        f"Paso 1 - V03 en el holdout: F1 {resumen_v03['f1_media_columnas']:.4f} "
        f"(HS {resumen_v03['f1_por_etiqueta']['HS']:.3f} / TR {resumen_v03['f1_por_etiqueta']['TR']:.3f} / "
        f"AG {resumen_v03['f1_por_etiqueta']['AG']:.3f}; a 0.5 {resumen_v03['f1_a_0_5']:.4f})"
    )

    # Paso 2: meta ajustado con el OOF del desarrollo y aplicado al holdout.
    proba_bases_dev, nombres_oof = cargar_oof_dev(desarrollo, y_dev, modulo_14)
    if nombres_oof != nombres_bases:
        raise RuntimeError(f"Orden de bases inconsistente: {nombres_oof} vs {nombres_bases}")
    matriz_dev, nombres_meta = modulo_17.construir_bloque_meta(
        proba_bases_dev, None, (), desarrollo["text"], "bases"
    )
    if len(nombres_meta) != 44 or list(nombres_meta[-8:]) != list(modulo_17.NOMBRES_ESTRUCTURALES_BASE):
        raise RuntimeError(f"Meta-features inesperadas: {len(nombres_meta)} columnas")
    escalador, modelos_meta = ajustar_meta(matriz_dev, y_dev)
    reproduccion = verificar_reproduccion_candidato(
        escalador, modelos_meta, matriz_dev, nombres_meta
    )
    matriz_holdout, nombres_holdout = modulo_17.construir_bloque_meta(
        proba_bases_holdout, None, (), holdout["text"], "bases"
    )
    if nombres_holdout != nombres_meta:
        raise RuntimeError("Los nombres de las meta-features del holdout no coinciden")
    proba_meta_holdout = predecir_meta(escalador, modelos_meta, matriz_holdout)
    resumen_meta = resumir_proba(
        proba_meta_holdout, y_holdout, np.full(len(ETIQUETAS), CONFIG["umbral_meta"])
    )
    log(
        f"Paso 2 - meta en el holdout: F1 {resumen_meta['f1_media_columnas']:.4f} "
        f"(HS {resumen_meta['f1_por_etiqueta']['HS']:.3f} / TR {resumen_meta['f1_por_etiqueta']['TR']:.3f} / "
        f"AG {resumen_meta['f1_por_etiqueta']['AG']:.3f})"
    )

    # Decision pre-registrada.
    f1_v03 = float(resumen_v03["f1_media_columnas"])
    f1_meta = float(resumen_meta["f1_media_columnas"])
    delta = f1_meta - f1_v03
    objetivo = f1_v03 - CONFIG["tolerancia_decision"]
    genera_submission = bool(f1_meta >= objetivo)
    if genera_submission:
        justificacion = (
            f"El meta ({f1_meta:.4f}) alcanza el criterio pre-registrado (V03 - "
            f"{CONFIG['tolerancia_decision']:.3f} = {objetivo:.4f}); delta {delta:+.4f}. Se genera la "
            "submission v08 (sonda v2 del meta)."
        )
    else:
        justificacion = (
            f"El meta ({f1_meta:.4f}) NO alcanza el criterio pre-registrado (V03 - "
            f"{CONFIG['tolerancia_decision']:.3f} = {objetivo:.4f}); delta {delta:+.4f}. NO se genera "
            "submission."
        )
    log(f"decision: {'GO' if genera_submission else 'NO-GO'} - {justificacion}")

    # Contador del holdout: esta tarea consume 2 evaluaciones.
    filas_evaluaciones = [
        construir_fila_evaluacion(
            ID_EVALUACION_V03,
            "Referencia V03 (M2 reajustado con todo el desarrollo + umbrales congelados 0.44/0.51/0.46)",
            resumen_v03,
        ),
        construir_fila_evaluacion(
            ID_EVALUACION_META,
            "Candidato meta Ridge alpha=3 (44 meta-features OOF del desarrollo; bases reajustados con el desarrollo)",
            resumen_meta,
        ),
    ]
    evaluaciones_antes, evaluaciones_despues = registrar_evaluaciones_holdout(filas_evaluaciones)

    # Paso 3: refit final y submission (solo si GO).
    artefactos: dict[str, Any] = {}
    resultado_submission: dict[str, Any] | None = None
    if genera_submission:
        bases_finales: dict[str, BaseAjustado] = {}
        for nombre in modulo_14.MODELOS:
            inicio_base = time.perf_counter()
            bases_finales[nombre] = ajustar_base(nombre, train["text"], train[list(ETIQUETAS)].to_numpy(), modulo_14)
            log(
                f"base {nombre} reajustado con el train completo (dim "
                f"{bases_finales[nombre].dimension}) en {time.perf_counter() - inicio_base:.1f} s"
            )
        proba_bases_test, nombres_test = predecir_bases(bases_finales, test["text"])
        if nombres_test != nombres_bases:
            raise RuntimeError("El orden de bases del refit final es inconsistente")
        matriz_test, _ = modulo_17.construir_bloque_meta(
            proba_bases_test, None, (), test["text"], "bases"
        )
        proba_meta_test = predecir_meta(escalador, modelos_meta, matriz_test)
        prediccion_test = (proba_meta_test >= CONFIG["umbral_meta"]).astype(int)
        ruta_submission = escribir_submission(plantilla, prediccion_test)
        resumen_test = resumir_prediccion_test(train, prediccion_test)
        resultado_submission = {
            "archivo": str(ruta_submission.relative_to(PROJECT_ROOT)),
            "n_filas": len(plantilla),
            "umbral": CONFIG["umbral_meta"],
            "validacion": (
                "OK contra la plantilla oficial: columnas id,HS,TR,AG; 1154 filas; ids identicos "
                "y en el mismo orden; valores en {0,1}; sin nulos ni duplicados"
            ),
            **resumen_test,
            "dimension_meta": int(matriz_test.shape[1]),
            "dimensiones_bases": {nombre: base.dimension for nombre, base in bases_finales.items()},
        }
        log(
            "Paso 3 - submission v08: positivos "
            f"HS {resumen_test['tasa_positivos_test']['HS']:.4f} / "
            f"TR {resumen_test['tasa_positivos_test']['TR']:.4f} / "
            f"AG {resumen_test['tasa_positivos_test']['AG']:.4f} "
            f"(train {resumen_test['tasa_positivos_train']['HS']:.4f} / "
            f"{resumen_test['tasa_positivos_train']['TR']:.4f} / "
            f"{resumen_test['tasa_positivos_train']['AG']:.4f}; test estimado "
            f"{CONFIG['prevalencia_test_estimada']['HS']:.4f} / "
            f"{CONFIG['prevalencia_test_estimada']['TR']:.4f} / "
            f"{CONFIG['prevalencia_test_estimada']['AG']:.4f})"
        )
        joblib.dump(
            {
                "fecha": CONFIG["fecha"],
                "script": "experiments/18_p7_holdout_y_submission_v08.py",
                "variante": reproduccion["variante"],
                "conjunto": "bases",
                "modelo": "ridge",
                "alpha": CONFIG["alpha_meta"],
                "umbral": CONFIG["umbral_meta"],
                "nombres_features": nombres_meta,
                "meta": {"escalador": escalador, "modelos": modelos_meta},
                "bases": {
                    nombre: {
                        "representacion": base.representacion,
                        "tipo": base.tipo,
                        "vectorizadores": base.vectorizadores,
                        "modelos": base.modelos,
                        "dimension": base.dimension,
                    }
                    for nombre, base in bases_finales.items()
                },
                "entrenado_con": (
                    "meta: OOF del desarrollo (1483 filas); bases: train completo (1731 filas)"
                ),
                "holdout": {
                    "v03_f1": f1_v03,
                    "meta_f1": f1_meta,
                    "delta": delta,
                    "f1_por_etiqueta_meta": resumen_meta["f1_por_etiqueta"],
                    "decision": "GO",
                },
                "advertencia": (
                    "El meta se ajusto con el OOF del desarrollo y no vio el holdout ni el test; los "
                    "bases del test se reajustaron con el train completo. El gap local-LB de la familia "
                    "union word+char es -0.013 a -0.016 (D-012): el LB esperado puede ser menor."
                ),
            },
            MODELO_FINAL_PATH,
        )
        artefactos["modelo_final"] = str(MODELO_FINAL_PATH.relative_to(PROJECT_ROOT))
        log(f"modelo final guardado: {MODELO_FINAL_PATH.name}")
    else:
        log("Paso 3 omitido: sin GO no se toca el test ni se generan artefactos de submission")

    duracion = time.perf_counter() - inicio
    reporte: dict[str, Any] = {
        "fecha": CONFIG["fecha"],
        "fase": "P7 - validacion en el holdout congelado y submission v08 (sonda v2)",
        "script": "experiments/18_p7_holdout_y_submission_v08.py",
        "protocolo": {
            "holdout": "config/holdout_v2.json (particion congelada, sha256 verificado)",
            "particion_csv": str(PARTICION_PATH.relative_to(PROJECT_ROOT)),
            "n_dev": CONFIG["n_dev"],
            "n_holdout": CONFIG["n_holdout"],
            "evaluaciones_antes": evaluaciones_antes,
            "evaluaciones_consumidas": CONFIG["evaluaciones_de_esta_tarea"],
            "evaluaciones_despues": evaluaciones_despues,
            "evaluaciones_restantes": CONFIG["max_evaluaciones_holdout"] - evaluaciones_despues,
        },
        "referencias_desarrollo": CONFIG["referencias_desarrollo"],
        "paso_1_v03_holdout": {
            "modelo": "M2_union_lr_c2 reajustado con todo el desarrollo (1483 filas)",
            "umbrales_congelados": {etiqueta: float(umbrales_v03[etiqueta]) for etiqueta in ETIQUETAS},
            "dimension_base": bases_dev["M2_union_lr_c2"].dimension,
            **resumen_v03,
        },
        "paso_2_meta_holdout": {
            "descripcion": (
                "meta Ridge alpha=3 (balanced) sobre 44 meta-features OOF del desarrollo; bases "
                "reajustados con todo el desarrollo; umbral 0.5"
            ),
            "reproduccion_candidato_p6": reproduccion,
            **resumen_meta,
        },
        "decision": {
            "regla": (
                f"generar submission si meta_holdout >= v03_holdout - {CONFIG['tolerancia_decision']}"
            ),
            "v03_holdout_f1": f1_v03,
            "meta_holdout_f1": f1_meta,
            "delta_meta_vs_v03": delta,
            "objetivo": objetivo,
            "cumple": genera_submission,
            "submission_generada": genera_submission,
            "justificacion": justificacion,
        },
        "paso_3_submission": resultado_submission,
        "advertencia": (
            "El holdout tiene 248 filas (1 evaluacion puntual; error estandar aproximado del orden de "
            "0.03); la decision sigue la regla pre-registrada y NO se re-optimiza ningun umbral con el "
            "holdout. El gap local-LB de la familia union word+char es -0.013 a -0.016 (D-012)."
        ),
        "evaluaciones_holdout": filas_evaluaciones,
        "artefactos": artefactos,
        "duracion_segundos": round(duracion, 2),
    }
    REPORTE_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORTE_JSON.write_text(json.dumps(reporte, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"reporte escrito: {REPORTE_JSON.name}")

    # Registros (experiments y leaderboard solo si hay submission).
    if genera_submission and resultado_submission is not None:
        registrar_en_experiments(
            {
                "experiment_id": ID_EXPERIMENTO_SUBMISSION,
                "competition": "KAGGLE_TOXIC_2026_PUJ",
                "question": (
                    "¿Confirma el holdout congelado la mejora del meta v2 (Ridge alpha=3) sobre V03 y "
                    "justifica una submission para el leaderboard?"
                ),
                "hypothesis": (
                    "El meta v2 mantiene en el holdout la mejora medida en el desarrollo y supera el "
                    "LB de v03 (0.78306)"
                ),
                "preprocessing": (
                    "sin texto en el meta: probabilidades, margenes y entropias OOF de los 4 bases + 8 "
                    "rasgos estructurales; bases = TF-IDF word(1,2) y char_wb(3,5), min_df=2, sublinear_tf"
                ),
                "feature_groups": "META_SCORES+META_UNCERTAINTY+STRUCTURAL",
                "representation": (
                    "features densas (44 columnas) estandarizadas con el desarrollo; bases del test "
                    "reajustados con todo el train"
                ),
                "model": (
                    "RidgeClassifier(alpha=3, class_weight=balanced) x3 (HS, TR, AG) sobre el meta del "
                    "stacking v2"
                ),
                "hyperparameters": json.dumps(
                    {
                        "alpha": CONFIG["alpha_meta"],
                        "umbral": CONFIG["umbral_meta"],
                        "bases": {nombre: bias.dimension for nombre, bias in bases_finales.items()},
                        "refit": "todo el train (1731 filas)",
                    },
                    ensure_ascii=False,
                ),
                "validation": (
                    "holdout congelado (248 filas; evaluaciones 2/5: P7-01 V03 y P7-02 meta); decision "
                    "pre-registrada meta >= V03 - 0.005; submission validada contra la plantilla oficial"
                ),
                "f1": round(f1_meta, 4),
                "macro_f1": round(resumen_meta["f1_macro6"], 4),
                "weighted_f1": "",
                "precision": "",
                "recall": "",
                "training_time": round(duracion, 2),
                "dimension": int(matriz_dev.shape[1]),
                "result": json.dumps(
                    {
                        "holdout_v03": f1_v03,
                        "holdout_meta": f1_meta,
                        "delta": delta,
                        "holdout_meta_por_etiqueta": resumen_meta["f1_por_etiqueta"],
                        "positivos_test": resultado_submission["positivos_test"],
                        "tasa_positivos_test": resultado_submission["tasa_positivos_test"],
                        "tasa_positivos_train": resultado_submission["tasa_positivos_train"],
                        "evaluaciones_holdout": evaluaciones_despues,
                    },
                    ensure_ascii=False,
                ),
                "interpretation": (
                    f"El holdout confirma el criterio (delta {delta:+.4f}); la submission v08 es la sonda "
                    "v2 del meta en el leaderboard; el LB esperado lleva el gap de la familia (-0.013 a "
                    "-0.016) frente a la referencia del holdout"
                ),
                "decision": "submission generada (sonda v2); pendiente de envio por Carlos a Kaggle",
                "date": CONFIG["fecha"],
                "script": "experiments/18_p7_holdout_y_submission_v08.py",
                "notes": (
                    "holdout usado 2/5 evaluaciones; meta ajustado con el OOF del desarrollo; bases del "
                    "test reajustados con todo el train; kaggle_f1 pendiente"
                ),
            }
        )
        registrar_en_leaderboard(
            {
                "submission_id": ID_LEADERBOARD,
                "fecha": CONFIG["fecha"],
                "archivo": resultado_submission["archivo"],
                "local_f1": round(f1_meta, 5),
                "kaggle_f1": "",
                "diferencia": "",
                "features": (
                    "meta de stacking v2: 44 columnas (scores, margenes, entropias de 4 bases TF-IDF "
                    "word+char + 8 estructurales)"
                ),
                "modelo": (
                    "RidgeClassifier(alpha=3, balanced) x3 sobre el meta; bases reajustados con todo el "
                    "train; umbral 0.5"
                ),
                "hiperparametros": json.dumps(
                    {
                        "alpha": CONFIG["alpha_meta"],
                        "umbral": CONFIG["umbral_meta"],
                        "bases": list(modulo_14.MODELOS),
                    },
                    ensure_ascii=False,
                ),
                "interpretacion": (
                    f"Sonda v2 (la envia Carlos). Referencia local: holdout congelado 248 filas, meta "
                    f"{f1_meta:.4f} vs V03 {f1_v03:.4f} (delta {delta:+.4f}); positivos predichos "
                    f"HS {resultado_submission['tasa_positivos_test']['HS']:.4f} / "
                    f"TR {resultado_submission['tasa_positivos_test']['TR']:.4f} / "
                    f"AG {resultado_submission['tasa_positivos_test']['AG']:.4f} frente a train 0.4367 / "
                    "0.2998 / 0.3824. Advertencia: gap local-LB de la familia -0.013 a -0.016 (D-012); "
                    "el LB de v03 es 0.78306 y el error estandar del LB es ~0.012."
                ),
            }
        )

    lineas_contexto = [
        (
            "P7 (D-016): validacion del candidato en el holdout congelado (248 filas; particion "
            f"verificada por sha256) y decision de submission v08. Evaluaciones al holdout: "
            f"{evaluaciones_antes} previas + 2 de esta tarea = {evaluaciones_despues}/5 "
            "(`reports/holdout_evaluaciones_v2.csv`)."
        ),
        (
            f"Paso 1 - V03 de referencia en el holdout (M2 reajustado con todo el desarrollo; umbrales "
            f"congelados 0.44/0.51/0.46): F1 {f1_v03:.4f} (HS {resumen_v03['f1_por_etiqueta']['HS']:.3f} / "
            f"TR {resumen_v03['f1_por_etiqueta']['TR']:.3f} / AG {resumen_v03['f1_por_etiqueta']['AG']:.3f}; "
            f"a 0.5 {resumen_v03['f1_a_0_5']:.4f})."
        ),
        (
            f"Paso 2 - meta Ridge alpha=3 (44 meta-features OOF del desarrollo; bases reajustados con el "
            f"desarrollo; umbral 0.5) en el holdout: F1 {f1_meta:.4f} "
            f"(HS {resumen_meta['f1_por_etiqueta']['HS']:.3f} / TR {resumen_meta['f1_por_etiqueta']['TR']:.3f} / "
            f"AG {resumen_meta['f1_por_etiqueta']['AG']:.3f}); reproduce el candidato de P6 "
            f"(dif max en desarrollo {reproduccion['diferencia_maxima_dev']:.1e})."
        ),
        (
            f"Decision pre-registrada (meta >= V03 - {CONFIG['tolerancia_decision']:.3f}): delta "
            f"{delta:+.4f} -> {'GO' if genera_submission else 'NO-GO'}. {justificacion}"
        ),
    ]
    if genera_submission and resultado_submission is not None:
        lineas_contexto.extend(
            [
                (
                    "Paso 3 - submission v08: bases reajustados con el train completo (1731), meta-features "
                    "del test y umbral 0.5; validada contra la plantilla oficial (1154 filas, ids en orden, "
                    "valores en {0,1}, sin nulos ni duplicados)."
                ),
                (
                    f"Positivos predichos en test: HS {resultado_submission['positivos_test']['HS']} "
                    f"({resultado_submission['tasa_positivos_test']['HS'] * 100:.1f} %) / TR "
                    f"{resultado_submission['positivos_test']['TR']} "
                    f"({resultado_submission['tasa_positivos_test']['TR'] * 100:.1f} %) / AG "
                    f"{resultado_submission['positivos_test']['AG']} "
                    f"({resultado_submission['tasa_positivos_test']['AG'] * 100:.1f} %); referencia train "
                    "43.7/30.0/38.2 % y test estimado 40.6/27.9/35.6 % (E00)."
                ),
                (
                    "Registro: fila S08 en `experiments.csv` y `v08_meta_v2` en "
                    "`submissions/leaderboard_log.csv` (kaggle_f1 vacio; la sonda la envia Carlos). "
                    "Hipotesis de la sonda: el meta supera el LB de v03 (0.78306)."
                ),
            ]
        )
    lineas_contexto.extend(
        [
            (
                "Sin fuga: el holdout solo se uso para medir los 2 finalistas y el test solo para "
                "predecir; el meta, su escalador y los umbrales se ajustaron antes con el OOF del "
                "desarrollo. Advertencia: 248 filas y 1 evaluacion puntual (error estandar ~0.03); el gap "
                "local-LB de la familia union es -0.013 a -0.016 (D-012)."
            ),
            (
                f"Duracion: {duracion:.1f} s. Script: experiments/18_p7_holdout_y_submission_v08.py. "
                "Artefactos: reports/p7_holdout_resultados.json, reports/holdout_evaluaciones_v2.csv"
                + (
                    ", models/v2_meta_final.joblib, submissions/toxic_submission_v08_meta_v2.csv."
                    if genera_submission
                    else " (sin submission: NO-GO)."
                )
            ),
        ]
    )
    escribir_contexto(lineas_contexto)
    actualizar_todo(reporte)

    log("resumen final")
    log(
        f"Paso 1 - V03 holdout: {f1_v03:.4f} | Paso 2 - meta holdout: {f1_meta:.4f} | delta {delta:+.4f}"
    )
    log(f"decision: {'GO' if genera_submission else 'NO-GO'} - {justificacion}")
    log(
        f"evaluaciones al holdout: {evaluaciones_antes} previas + 2 = {evaluaciones_despues}/"
        f"{CONFIG['max_evaluaciones_holdout']}"
    )
    log(f"duracion total: {duracion:.1f} s")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:  # noqa: BLE001 - reporte explicito y salida con codigo 1
        import traceback

        traceback.print_exc()
        print(f"[P7] ERROR: {error}", file=sys.stderr)
        sys.exit(1)
