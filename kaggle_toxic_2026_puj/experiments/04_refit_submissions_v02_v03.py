"""04_refit_submissions_v02_v03.py - Fase 13a: refit del ganador, umbrales y submissions.

Pregunta que responde
---------------------
1. ¿Reproduce el codigo, bit a bit, la metrica out-of-fold del ganador de la
   Fase 4-5 (``MOD-03b_BEST``, ``f1_media_columnas`` 0.7927)?
2. ¿Mejora la metrica oficial si se ajusta un umbral de decision por etiqueta
   sobre las probabilidades out-of-fold?
3. ¿Que submissions se generan con el modelo reajustado con todo el train?

Metodo
------
- Configuracion replicada SIN cambios desde ``03_variantes_representaciones.py``:
  representacion ``REP-05`` (union TF-IDF word (1,2) + char_wb (3,5), min_df=2,
  max_df=1.0, sublinear_tf) y modelo ``MOD-03b_BEST``
  (``LogisticRegression(class_weight="balanced", C=2.0, max_iter=1000,
  solver="liblinear", random_state=42)`` por etiqueta).
- Validacion: ``StratifiedGroupKFold(5, shuffle=True, random_state=42)`` con
  grupos = componentes de near-duplicado (D-004) y estratificacion por la
  combinacion HS x TR x AG. Una sola particion, identica a la de la Fase 4-5.
- Umbrales: rejilla 0.20-0.80 paso 0.01 maximizando la F1 binaria de cada
  etiqueta SOBRE LAS PROBABILIDADES OUT-OF-FOLD. El test nunca interviene en la
  seleccion de umbrales.
- Regla de decision pre-registrada (``PROJECT_STATE.md``): los umbrales optimos
  se adoptan solo si la mejora es >= +0.005 absoluto en ``f1_media_columnas`` y
  la mejora se mantiene en >= 3 de los 5 pliegues. En caso contrario se deja
  constancia y se usa 0.5.
- Post-proceso jerarquico: NO se usa (la Fase 4-5 lo midio en -0.0056).

Artefactos
----------
- ``features/oof_mod03b.npz``: ids, y_true, proba ``(n, 3)`` y nombres de
  etiquetas out-of-fold, para ensembles futuros.
- ``models/mod03b_lr_union.joblib``: vectorizador + 3 modelos + umbrales +
  etiquetas del refit final (todo el train).
- ``submissions/toxic_submission_v02_lr_union.csv``: umbral 0.5 por etiqueta.
- ``submissions/toxic_submission_v03_lr_union_thr.csv``: solo si los umbrales
  optimos superan la regla pre-registrada.
- ``reports/fase13a_refit_umbral_metricas.json``: detalle completo del refit,
  del ajuste de umbrales y de la validacion de las submissions.

Idempotencia
------------
Los pliegues, las metricas, los umbrales y las submissions son deterministas.
``experiments.csv`` y ``submissions/leaderboard_log.csv`` se actualizan
eliminando primero las filas de los ids de esta fase y reescribiendolas, por lo
que dos ejecuciones seguidas dejan el mismo contenido. El unico campo no
determinista es el tiempo de ejecucion (ruido de maquina).

Ejecucion
---------
    /home/carlos/venv/bin/python experiments/04_refit_submissions_v02_v03.py
"""

from __future__ import annotations

import json
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Final

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import FeatureUnion

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src import data as data_module
from src.evaluation import ETIQUETAS, metricas_multietiqueta

CONFIG: Final[dict] = {
    "grupos_path": PROJECT_ROOT / "data" / "processed" / "grupos_near_duplicados.csv",
    "referencia_json": PROJECT_ROOT / "reports" / "variantes_representaciones.json",
    "oof_npz": PROJECT_ROOT / "features" / "oof_mod03b.npz",
    "modelo_joblib": PROJECT_ROOT / "models" / "mod03b_lr_union.joblib",
    "metricas_json": PROJECT_ROOT / "reports" / "fase13a_refit_umbral_metricas.json",
    "experiments_csv": PROJECT_ROOT / "experiments.csv",
    "leaderboard_csv": PROJECT_ROOT / "submissions" / "leaderboard_log.csv",
    "contexto_md": PROJECT_ROOT / "reports" / "contexto.md",
    "submissions_dir": PROJECT_ROOT / "submissions",
    "submission_v02": "toxic_submission_v02_lr_union.csv",
    "submission_v03": "toxic_submission_v03_lr_union_thr.csv",
    "random_state": 42,
    "n_splits": 5,
    "umbral_base": 0.5,
    "rejilla_umbral": (0.20, 0.80, 0.01),
    "umbral_mejora": 0.005,
    "pliegues_minimos_mejora": 3,
    # Valor exacto publicado en reports/variantes_representaciones.json para
    # MOD-03b_BEST (la tabla CSV lo redondea a 0.7927). La replicacion debe
    # coincidir con el valor completo dentro de 1e-6.
    "f1_oof_referencia": 0.7926588861781015,
    "f1_oof_referencia_redondeada": 0.7927,
    "tolerancia": 1e-6,
    "id_umbrales": "TH-01",
    "id_submission_base": "S02",
    "id_submission_umbral": "S03",
    "id_leaderboard_base": "v02_lr_union",
    "id_leaderboard_umbral": "v03_lr_union_thr",
    "competencia": "KAGGLE_TOXIC_2026_PUJ",
    "fecha": "2026-09-22",
    "momento_contexto": "2026-09-22",
    "script": "experiments/04_refit_submissions_v02_v03.py",
    "n_filas_test": 1154,
    "columnas_submission": ("id", "HS", "TR", "AG"),
}

# Representacion REP-05 de la Fase 4-5 (copiada literalmente).
VECTORIZADOR_PARAMS: Final[tuple[dict, ...]] = (
    {
        "min_df": 2,
        "max_df": 1.0,
        "sublinear_tf": True,
        "lowercase": True,
        "strip_accents": None,
        "analyzer": "word",
        "ngram_range": (1, 2),
    },
    {
        "min_df": 2,
        "max_df": 1.0,
        "sublinear_tf": True,
        "lowercase": True,
        "strip_accents": None,
        "analyzer": "char_wb",
        "ngram_range": (3, 5),
    },
)

# Modelo MOD-03b de la Fase 4-5 (copiado literalmente).
MODELO_PARAMS: Final[dict] = {
    "C": 2.0,
    "class_weight": "balanced",
    "max_iter": 1000,
    "solver": "liblinear",
    "random_state": 42,
}

NOMBRE_REPRESENTACION: Final[str] = (
    "union TF-IDF word (1,2) + char_wb (3,5), min_df=2, sublinear_tf"
)
NOMBRE_MODELO: Final[str] = (
    "LogisticRegression(class_weight=balanced, C=2.0, liblinear) x3"
)
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


def log(message: str) -> None:
    """Imprime un mensaje de progreso con prefijo de fase.

    Args:
        message: texto a imprimir.
    """
    print(f"[04] {message}", flush=True)


# ===== SECCION 1: carga de datos, pliegues y replicacion del ganador =====


def cargar_datos() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Carga train, test y la asignacion de grupos de near-duplicado.

    Returns:
        Tupla ``(train, test, grupos)``; ``grupos`` es una serie indexada por
        ``id`` con el numero de grupo de near-duplicado.

    Raises:
        FileNotFoundError: si falta la particion generada en la Fase 0.
    """
    train = data_module.read_train()
    test = data_module.read_test()
    grupos_path: Path = CONFIG["grupos_path"]
    if not grupos_path.exists():
        raise FileNotFoundError(
            f"Falta {grupos_path}. Ejecuta experiments/00_auditoria_datos.py primero."
        )
    grupos = pd.read_csv(grupos_path).set_index("id")["grupo"]
    return train, test, grupos


def construir_pliegues(
    train: pd.DataFrame, grupos: pd.Series
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Construye la particion 5-fold agrupada y estratificada (D-004).

    Replica exactamente la particion de la Fase 4-5 para que las metricas sean
    comparables con el ranking publicado.

    Args:
        train: DataFrame de entrenamiento con ``id``, ``text`` y etiquetas.
        grupos: serie de grupos de near-duplicado indexada por ``id``.

    Returns:
        Lista de 5 tuplas ``(indices_entrenamiento, indices_validacion)`` con
        indices posicionales.

    Raises:
        ValueError: si algun ``id`` de train no tiene grupo asignado.
    """
    combinacion = (
        train["HS"].astype(str) + train["TR"].astype(str) + train["AG"].astype(str)
    )
    asignacion = train["id"].astype(int).map(grupos)
    if asignacion.isna().any():
        faltantes = int(asignacion.isna().sum())
        raise ValueError(
            f"{faltantes} ids de train sin grupo de near-duplicado; "
            "regenera data/processed/grupos_near_duplicados.csv"
        )
    divisor = StratifiedGroupKFold(
        n_splits=CONFIG["n_splits"],
        shuffle=True,
        random_state=CONFIG["random_state"],
    )
    pliegues = [
        (entrenamiento, validacion)
        for entrenamiento, validacion in divisor.split(
            train, combinacion, groups=asignacion.to_numpy()
        )
    ]
    for numero, (entrenamiento, validacion) in enumerate(pliegues, start=1):
        log(
            f"pliegue {numero}: {len(entrenamiento)} train / {len(validacion)} validacion"
        )
    return pliegues


def construir_vectorizador() -> FeatureUnion:
    """Crea la union de vectorizadores de la representacion ganadora (REP-05).

    Returns:
        ``FeatureUnion`` con el bloque de palabras (1,2) y el de caracteres
        ``char_wb`` (3,5), en el mismo orden que en la Fase 4-5.
    """
    return FeatureUnion(
        [
            (f"bloque_{indice}", TfidfVectorizer(**parametros))
            for indice, parametros in enumerate(VECTORIZADOR_PARAMS)
        ]
    )


def construir_modelo() -> LogisticRegression:
    """Instancia un clasificador binario nuevo con los hiperparametros de MOD-03b.

    Returns:
        ``LogisticRegression`` sin ajustar.
    """
    return LogisticRegression(**MODELO_PARAMS)


def probabilidades_oof(
    train: pd.DataFrame, pliegues: Sequence[tuple[np.ndarray, np.ndarray]]
) -> np.ndarray:
    """Calcula las probabilidades out-of-fold del ganador de la Fase 4-5.

    El vectorizador se ajusta SIEMPRE con el pliegue de entrenamiento y se
    transforma el de validacion; nunca se ve el pliegue de validacion ni el test.

    Args:
        train: DataFrame de entrenamiento con ``text`` y etiquetas.
        pliegues: particion agrupada compartida.

    Returns:
        Matriz ``(n, 3)`` con la probabilidad de la clase positiva por etiqueta.
    """
    probabilidad = np.zeros((len(train), len(ETIQUETAS)), dtype=float)
    for entrenamiento, validacion in pliegues:
        vectorizador = construir_vectorizador()
        matriz_entrenamiento = vectorizador.fit_transform(train["text"].iloc[entrenamiento])
        matriz_validacion = vectorizador.transform(train["text"].iloc[validacion])
        for indice, etiqueta in enumerate(ETIQUETAS):
            modelo = construir_modelo()
            modelo.fit(matriz_entrenamiento, train[etiqueta].iloc[entrenamiento])
            probabilidad[validacion, indice] = modelo.predict_proba(matriz_validacion)[:, 1]
    return probabilidad


def predecir_con_umbrales(
    probabilidad: np.ndarray, umbrales: Sequence[float]
) -> np.ndarray:
    """Umbraliza una matriz de probabilidades columna a columna.

    Args:
        probabilidad: matriz ``(n, 3)`` con probabilidades de la clase positiva.
        umbrales: umbral por etiqueta, en el orden de :data:`ETIQUETAS`.

    Returns:
        Matriz ``(n, 3)`` con valores en ``{0, 1}``.
    """
    return (probabilidad >= np.asarray(umbrales, dtype=float)[None, :]).astype(int)


def verificar_replicacion(probabilidad: np.ndarray, y_true: np.ndarray) -> dict:
    """Comprueba que la F1 out-of-fold reproduce el valor de la Fase 4-5.

    Args:
        probabilidad: matriz ``(n, 3)`` de probabilidades out-of-fold.
        y_true: matriz ``(n, 3)`` con las etiquetas verdaderas.

    Returns:
        Diccionario con las metricas reproducidas y la diferencia frente a la
        referencia.

    Raises:
        AssertionError: si la F1 reproducida no coincide con la referencia
            dentro de la tolerancia declarada.
    """
    prediccion = predecir_con_umbrales(probabilidad, [CONFIG["umbral_base"]] * 3)
    metricas = metricas_multietiqueta(y_true, prediccion)
    f1_reproducida = float(metricas["f1_media_columnas"])
    referencia = float(CONFIG["f1_oof_referencia"])
    diferencia = f1_reproducida - referencia
    log(
        f"f1_media_columnas OOF reproducida: {f1_reproducida:.6f} "
        f"(referencia {referencia:.6f}; diferencia {diferencia:+.9f})"
    )
    assert abs(diferencia) <= CONFIG["tolerancia"], (
        "La replicacion de MOD-03b_BEST no coincide con la Fase 4-5: "
        f"{f1_reproducida!r} vs {referencia!r}"
    )
    assert round(f1_reproducida, 4) == CONFIG["f1_oof_referencia_redondeada"], (
        "La F1 reproducida no redondea al valor publicado 0.7927: "
        f"{round(f1_reproducida, 4)!r}"
    )
    return {
        "f1_media_columnas": f1_reproducida,
        "f1_media_columnas_redondeada": round(f1_reproducida, 4),
        "f1_referencia_fase_4_5": referencia,
        "diferencia": diferencia,
        "tolerancia": CONFIG["tolerancia"],
        "metricas_oof": metricas,
    }


def guardar_probabilidades_oof(
    train: pd.DataFrame, probabilidad: np.ndarray
) -> Path:
    """Guarda las probabilidades out-of-fold en un ``.npz`` para ensembles.

    Los identificadores se guardan como array Unicode (no como array de objetos)
    para que el archivo se pueda releer con ``allow_pickle=False``; la relectura
    se verifica antes de devolver la ruta.

    Args:
        train: DataFrame de entrenamiento (aporta ``id`` y etiquetas).
        probabilidad: matriz ``(n, 3)`` de probabilidades out-of-fold.

    Returns:
        Ruta del archivo escrito.

    Raises:
        ValueError: si el archivo guardado no se puede releer con las formas
            esperadas y sin pickle.
    """
    destino: Path = CONFIG["oof_npz"]
    destino.parent.mkdir(parents=True, exist_ok=True)
    identificadores = np.asarray([str(valor) for valor in train["id"]], dtype=str)
    np.savez(
        destino,
        ids=identificadores,
        y_true=train[list(ETIQUETAS)].to_numpy(),
        proba=probabilidad,
        etiquetas=np.asarray(ETIQUETAS),
    )
    with np.load(destino, allow_pickle=False) as releido:
        formas = {clave: releido[clave].shape for clave in releido.files}
    esperadas = {
        "ids": (len(train),),
        "y_true": (len(train), len(ETIQUETAS)),
        "proba": (len(train), len(ETIQUETAS)),
        "etiquetas": (len(ETIQUETAS),),
    }
    if formas != esperadas:
        raise ValueError(f"El .npz guardado tiene formas inesperadas: {formas}")
    return destino


# ===== SECCION 2: ajuste de umbral por etiqueta =====


def rejilla_umbrales() -> np.ndarray:
    """Construye la rejilla de umbrales 0.20-0.80 con paso 0.01.

    Returns:
        Vector de umbrales redondeados a dos decimales.
    """
    inicio, fin, paso = CONFIG["rejilla_umbral"]
    return np.round(np.arange(inicio, fin + paso / 2, paso), 2)


def optimizar_umbral(
    y_true: np.ndarray, probabilidad: np.ndarray, rejilla: np.ndarray
) -> tuple[float, float]:
    """Busca el umbral que maximiza la F1 binaria de una etiqueta.

    La busqueda se hace exclusivamente sobre probabilidades out-of-fold. Si dos
    umbrales empatan, se conserva el menor (primer maximo de la rejilla), lo que
    deja la decision determinista.

    Args:
        y_true: vector binario verdadero de una etiqueta.
        probabilidad: vector de probabilidades out-of-fold de esa etiqueta.
        rejilla: umbrales candidatos.

    Returns:
        Tupla ``(umbral_optimo, f1_optima)``.
    """
    mejor_umbral = float(rejilla[0])
    mejor_f1 = -1.0
    for umbral in rejilla:
        prediccion = (probabilidad >= umbral).astype(int)
        valor = float(f1_score(y_true, prediccion, zero_division=0))
        if valor > mejor_f1:
            mejor_umbral = float(umbral)
            mejor_f1 = valor
    return mejor_umbral, mejor_f1


def evaluar_umbrales(
    y_true: np.ndarray,
    probabilidad: np.ndarray,
    pliegues: Sequence[tuple[np.ndarray, np.ndarray]],
) -> dict:
    """Ajusta y evalua los umbrales por etiqueta sobre las probabilidades OOF.

    Args:
        y_true: matriz ``(n, 3)`` con las etiquetas verdaderas.
        probabilidad: matriz ``(n, 3)`` con las probabilidades out-of-fold.
        pliegues: particion usada para las probabilidades (cuenta la mejora por
            pliegue).

    Returns:
        Diccionario con los umbrales, las metricas con 0.5 y con los umbrales
        optimos, el recuento de pliegues mejorados y la decision.
    """
    rejilla = rejilla_umbrales()
    umbrales_optimos: dict[str, float] = {}
    por_etiqueta: dict[str, dict] = {}
    for indice, etiqueta in enumerate(ETIQUETAS):
        verdad = y_true[:, indice]
        columna = probabilidad[:, indice]
        umbral, f1_optima = optimizar_umbral(verdad, columna, rejilla)
        f1_base = float(
            f1_score(verdad, (columna >= CONFIG["umbral_base"]).astype(int), zero_division=0)
        )
        umbrales_optimos[etiqueta] = umbral
        por_etiqueta[etiqueta] = {
            "umbral_optimo": umbral,
            "f1_umbral_0.5": f1_base,
            "f1_umbral_optimo": f1_optima,
            "ganancia": f1_optima - f1_base,
        }

    umbrales = [umbrales_optimos[etiqueta] for etiqueta in ETIQUETAS]
    metricas_base = metricas_multietiqueta(
        y_true, predecir_con_umbrales(probabilidad, [CONFIG["umbral_base"]] * 3)
    )
    metricas_optimas = metricas_multietiqueta(
        y_true, predecir_con_umbrales(probabilidad, umbrales)
    )
    ganancia = float(metricas_optimas["f1_media_columnas"]) - float(
        metricas_base["f1_media_columnas"]
    )

    por_pliegue: list[dict] = []
    for numero, (_, validacion) in enumerate(pliegues, start=1):
        verdad = y_true[validacion]
        f1_base_pliegue = float(
            metricas_multietiqueta(
                verdad,
                predecir_con_umbrales(
                    probabilidad[validacion], [CONFIG["umbral_base"]] * 3
                ),
            )["f1_media_columnas"]
        )
        f1_optimo_pliegue = float(
            metricas_multietiqueta(
                verdad, predecir_con_umbrales(probabilidad[validacion], umbrales)
            )["f1_media_columnas"]
        )
        por_pliegue.append(
            {
                "pliegue": numero,
                "n_validacion": len(validacion),
                "f1_umbral_0.5": f1_base_pliegue,
                "f1_umbrales_optimos": f1_optimo_pliegue,
                "delta": f1_optimo_pliegue - f1_base_pliegue,
            }
        )
    pliegues_mejorados = int(sum(fila["delta"] > 0 for fila in por_pliegue))
    pliegues_mejorados_umbral = int(
        sum(fila["delta"] >= CONFIG["umbral_mejora"] for fila in por_pliegue)
    )
    adoptado = bool(
        ganancia >= CONFIG["umbral_mejora"]
        and pliegues_mejorados >= CONFIG["pliegues_minimos_mejora"]
    )
    log(
        f"umbrales optimos {umbrales_optimos} | f1 con 0.5 "
        f"{metricas_base['f1_media_columnas']:.4f} -> con optimos "
        f"{metricas_optimas['f1_media_columnas']:.4f} ({ganancia:+.4f}); "
        f"pliegues mejorados {pliegues_mejorados}/5 -> "
        f"{'adoptados' if adoptado else 'descartados'}"
    )
    return {
        "rejilla": {
            "inicio": CONFIG["rejilla_umbral"][0],
            "fin": CONFIG["rejilla_umbral"][1],
            "paso": CONFIG["rejilla_umbral"][2],
        },
        "por_etiqueta": por_etiqueta,
        "umbrales_optimos": umbrales_optimos,
        "f1_media_columnas_umbral_0.5": float(metricas_base["f1_media_columnas"]),
        "f1_media_columnas_umbrales_optimos": float(metricas_optimas["f1_media_columnas"]),
        "macro6_umbral_0.5": float(metricas_base["f1_macro6"]),
        "macro6_umbrales_optimos": float(metricas_optimas["f1_macro6"]),
        "ganancia": ganancia,
        "por_pliegue": por_pliegue,
        "pliegues_mejorados": pliegues_mejorados,
        "pliegues_con_mejora_mayor_umbral": pliegues_mejorados_umbral,
        "regla": (
            f"adoptar si ganancia >= {CONFIG['umbral_mejora']} absoluta y mejora en "
            f">= {CONFIG['pliegues_minimos_mejora']}/5 pliegues"
        ),
        "adoptado": adoptado,
        "metricas_optimas": metricas_optimas,
        "advertencia_sesgo_optimista": (
            "La ganancia de umbrales medida en el mismo OOF con el que se eligieron es una "
            "cota optimista. Verificacion anidada del revisor (umbral ajustado en los "
            "pliegues no evaluados): delta medio +0.0014, rango -0.0069..+0.0094, 2/5 "
            "pliegues positivos. Conclusion: v03 es una apuesta, v02 es el control."
        ),
    }


# ===== SECCION 3: refit con todo el train y submissions =====


def ajustar_modelo_final(
    train: pd.DataFrame,
) -> tuple[FeatureUnion, list[LogisticRegression], int]:
    """Ajusta el vectorizador y los tres modelos con TODO el train.

    Args:
        train: DataFrame de entrenamiento con etiquetas.

    Returns:
        Tupla ``(vectorizador, modelos, dimension)`` ajustados.
    """
    vectorizador = construir_vectorizador()
    matriz = vectorizador.fit_transform(train["text"])
    modelos = []
    for etiqueta in ETIQUETAS:
        modelo = construir_modelo()
        modelo.fit(matriz, train[etiqueta])
        modelos.append(modelo)
    return vectorizador, modelos, int(matriz.shape[1])


def predecir_test(
    test: pd.DataFrame,
    vectorizador: FeatureUnion,
    modelos: Sequence[LogisticRegression],
) -> np.ndarray:
    """Calcula las probabilidades de la clase positiva en el test.

    Args:
        test: DataFrame de test con ``text``.
        vectorizador: vectorizador ajustado con todo el train.
        modelos: modelos binarios ajustados con todo el train.

    Returns:
        Matriz ``(n, 3)`` con la probabilidad de la clase positiva por etiqueta.
    """
    matriz = vectorizador.transform(test["text"])
    return np.column_stack([modelo.predict_proba(matriz)[:, 1] for modelo in modelos])


def validar_submission(submission: pd.DataFrame, plantilla: pd.DataFrame, ruta: Path) -> None:
    """Valida una submission contra la plantilla oficial antes de escribirla.

    Args:
        submission: DataFrame candidato con ``id, HS, TR, AG``.
        plantilla: plantilla oficial ``data/raw/submission.csv``.
        ruta: ruta destino (solo para mensajes de error).

    Raises:
        ValueError: si el formato, los ids, los valores o los nulos no cumplen
            el contrato de la competencia.
    """
    columnas = tuple(submission.columns)
    if columnas != CONFIG["columnas_submission"]:
        raise ValueError(f"{ruta.name}: columnas {columnas} distintas de id,HS,TR,AG")
    if len(submission) != CONFIG["n_filas_test"]:
        raise ValueError(
            f"{ruta.name}: {len(submission)} filas frente a "
            f"{CONFIG['n_filas_test']} esperadas"
        )
    if list(submission["id"]) != list(plantilla["id"]):
        raise ValueError(
            f"{ruta.name}: los ids no coinciden con la plantilla oficial en "
            "contenido o en orden"
        )
    if submission.isna().any().any():
        raise ValueError(f"{ruta.name}: hay valores nulos")
    if not submission["id"].is_unique:
        raise ValueError(f"{ruta.name}: hay ids duplicados")
    etiquetas = submission[list(ETIQUETAS)].to_numpy()
    if etiquetas.min() < 0 or etiquetas.max() > 1:
        raise ValueError(f"{ruta.name}: hay valores fuera de {{0, 1}}")
    if not np.isin(etiquetas, (0, 1)).all():
        raise ValueError(f"{ruta.name}: las etiquetas deben ser enteras 0/1")


def escribir_submission(
    test: pd.DataFrame,
    plantilla: pd.DataFrame,
    prediccion: np.ndarray,
    nombre: str,
) -> Path:
    """Construye, valida y escribe una submission.

    La validacion ocurre en memoria: si algo falla se lanza la excepcion y NO se
    escribe ningun archivo invalido.

    Args:
        test: DataFrame de test (mismo orden que la plantilla oficial).
        plantilla: plantilla oficial ``data/raw/submission.csv``.
        prediccion: matriz ``(n, 3)`` binaria en el orden de :data:`ETIQUETAS`.
        nombre: nombre del archivo dentro de ``submissions/``.

    Returns:
        Ruta del archivo de submission escrito.

    Raises:
        ValueError: si la submission no cumple el formato oficial.
    """
    salida = plantilla.copy()
    for indice, etiqueta in enumerate(ETIQUETAS):
        salida[etiqueta] = prediccion[:, indice]
    destino = CONFIG["submissions_dir"] / nombre
    validar_submission(salida, plantilla, destino)
    CONFIG["submissions_dir"].mkdir(parents=True, exist_ok=True)
    salida.to_csv(destino, index=False, encoding="utf-8")
    releida = pd.read_csv(destino, dtype={"id": str})
    validar_submission(releida, plantilla, destino)
    log(f"submission escrita y validada: {destino.name} ({len(releida)} filas)")
    return destino


def resumir_prediccion(train: pd.DataFrame, prediccion: np.ndarray) -> dict:
    """Resume los positivos predichos en el test frente a la prevalencia de train.

    Args:
        train: DataFrame de entrenamiento con etiquetas.
        prediccion: matriz ``(n, 3)`` binaria del test.

    Returns:
        Diccionario con el conteo y la tasa de positivos por etiqueta en test y
        la tasa de positivos de train como referencia.
    """
    return {
        "positivos_test": {
            etiqueta: int(prediccion[:, indice].sum())
            for indice, etiqueta in enumerate(ETIQUETAS)
        },
        "tasa_positivos_test": {
            etiqueta: round(float(prediccion[:, indice].mean()), 4)
            for indice, etiqueta in enumerate(ETIQUETAS)
        },
        "tasa_positivos_train": {
            etiqueta: round(float(train[etiqueta].mean()), 4) for etiqueta in ETIQUETAS
        },
    }


def guardar_modelo(
    vectorizador: FeatureUnion,
    modelos: Sequence[LogisticRegression],
    umbrales: dict[str, float],
    usar_umbrales: bool,
    dimension: int,
) -> Path:
    """Guarda el modelo final con vectorizador, modelos, umbrales y etiquetas.

    Args:
        vectorizador: vectorizador ajustado con todo el train.
        modelos: modelos binarios ajustados con todo el train.
        umbrales: umbral por etiqueta.
        usar_umbrales: True si la submission v03 se genero con esos umbrales.
        dimension: numero de features del vectorizador final.

    Returns:
        Ruta del archivo ``.joblib`` escrito.
    """
    destino: Path = CONFIG["modelo_joblib"]
    destino.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "vectorizador": vectorizador,
            "modelos": list(modelos),
            "umbrales": umbrales,
            "umbral_base": CONFIG["umbral_base"],
            "usar_umbrales_optimizados": usar_umbrales,
            "etiquetas": list(ETIQUETAS),
            "dimension": dimension,
            "representacion": NOMBRE_REPRESENTACION,
            "modelo": NOMBRE_MODELO,
            "f1_oof": CONFIG["f1_oof_referencia"],
            "entrenado_con": "todo el train (1731 filas)",
            "fecha": CONFIG["fecha"],
            "script": CONFIG["script"],
        },
        destino,
    )
    return destino


# ===== SECCION 4: registro en experiments.csv, leaderboard y contexto =====


def registrar_en_experiments(filas: Sequence[dict]) -> int:
    """Anade (o refresca) las filas de la fase en ``experiments.csv``.

    La operacion es idempotente: elimina las filas cuyo ``experiment_id``
    pertenece a esta fase y despues las reescribe con el esquema existente.

    Args:
        filas: filas completas con las columnas de ``COLUMNAS_EXPERIMENTS``.

    Returns:
        Numero de filas escritas.

    Raises:
        ValueError: si el esquema del archivo no coincide con el esperado.
    """
    destino: Path = CONFIG["experiments_csv"]
    existente = pd.read_csv(destino, dtype=str, keep_default_na=False)
    if list(existente.columns) != list(COLUMNAS_EXPERIMENTS):
        raise ValueError("El esquema de experiments.csv cambio inesperadamente")
    identificadores = {fila["experiment_id"] for fila in filas}
    conservadas = existente[~existente["experiment_id"].isin(identificadores)]
    nuevas = pd.DataFrame(
        [{clave: fila[clave] for clave in COLUMNAS_EXPERIMENTS} for fila in filas],
        columns=list(COLUMNAS_EXPERIMENTS),
    )
    combinado = pd.concat([conservadas, nuevas], ignore_index=True)
    combinado.to_csv(destino, index=False, encoding="utf-8")
    return len(nuevas)


def registrar_en_leaderboard(filas: Sequence[dict]) -> int:
    """Anade (o refresca) las filas de submissions en ``leaderboard_log.csv``.

    El ``kaggle_f1`` se deja vacio: lo completa Carlos cuando envie la
    submission. La operacion es idempotente por ``submission_id``.

    Args:
        filas: filas con las columnas de ``COLUMNAS_LEADERBOARD``.

    Returns:
        Numero de filas escritas.

    Raises:
        ValueError: si el esquema del archivo no coincide con el esperado.
    """
    destino: Path = CONFIG["leaderboard_csv"]
    existente = pd.read_csv(destino, dtype=str, keep_default_na=False)
    if list(existente.columns) != list(COLUMNAS_LEADERBOARD):
        raise ValueError("El esquema de leaderboard_log.csv cambio inesperadamente")
    identificadores = {fila["submission_id"] for fila in filas}
    conservadas = existente[~existente["submission_id"].isin(identificadores)]
    nuevas = pd.DataFrame(
        [{clave: fila[clave] for clave in COLUMNAS_LEADERBOARD} for fila in filas],
        columns=list(COLUMNAS_LEADERBOARD),
    )
    combinado = pd.concat([conservadas, nuevas], ignore_index=True)
    combinado.to_csv(destino, index=False, encoding="utf-8")
    return len(nuevas)


def construir_fila_umbrales(
    resultado: dict, segundos: float, dimension: int
) -> dict:
    """Construye la fila ``TH-01`` del registro de experimentos.

    Args:
        resultado: resultado de :func:`evaluar_umbrales`.
        segundos: tiempo del ajuste de umbrales.
        dimension: numero de features del vectorizador final.

    Returns:
        Diccionario con las columnas de ``COLUMNAS_EXPERIMENTS``.
    """
    etiquetas = {
        etiqueta: {
            "umbral_optimo": resultado["por_etiqueta"][etiqueta]["umbral_optimo"],
            "f1_umbral_0.5": round(
                resultado["por_etiqueta"][etiqueta]["f1_umbral_0.5"], 6
            ),
            "f1_umbral_optimo": round(
                resultado["por_etiqueta"][etiqueta]["f1_umbral_optimo"], 6
            ),
        }
        for etiqueta in ETIQUETAS
    }
    adoptado = bool(resultado["adoptado"])
    decision = (
        "adoptada: los umbrales optimos se aplican en la submission v03"
        if adoptado
        else (
            "descartada: no cumple la regla pre-registrada (+0.005 absoluto y "
            ">= 3/5 pliegues); se mantiene el umbral 0.5"
        )
    )
    return {
        "experiment_id": CONFIG["id_umbrales"],
        "competition": CONFIG["competencia"],
        "question": (
            "¿Un umbral de decision por etiqueta, ajustado sobre las probabilidades "
            "out-of-fold, mejora la metrica oficial del ganador MOD-03b_BEST?"
        ),
        "hypothesis": (
            "Con class_weight=balanced el umbral 0.5 no es optimo y un umbral por "
            "etiqueta sube f1_media_columnas al menos +0.005"
        ),
        "preprocessing": (
            "ninguno (post-proceso sobre las probabilidades out-of-fold del refit; "
            "sin post-proceso jerarquico)"
        ),
        "feature_groups": "WORD_NGRAM+CHAR_NGRAM",
        "representation": NOMBRE_REPRESENTACION,
        "model": (
            "LogisticRegression(class_weight=balanced, C=2.0, liblinear) x3 con "
            "umbral de decision por etiqueta"
        ),
        "hyperparameters": json.dumps(
            {
                "umbrales_optimos": resultado["umbrales_optimos"],
                "rejilla": resultado["rejilla"],
                "umbral_base": CONFIG["umbral_base"],
            },
            ensure_ascii=False,
        ),
        "validation": (
            "Umbrales ajustados SOLO con las probabilidades out-of-fold de "
            "StratifiedGroupKFold(5, shuffle=True, random_state=42) agrupado por "
            "near-duplicado y estratificado por HSxTRxAG; el test nunca se usa"
        ),
        "f1": round(resultado["f1_media_columnas_umbrales_optimos"], 4),
        "macro_f1": round(resultado["macro6_umbrales_optimos"], 4),
        "weighted_f1": "",
        "precision": "",
        "recall": "",
        "training_time": round(segundos, 2),
        "dimension": dimension,
        "result": json.dumps(
            {
                "f1_media_columnas_umbral_0.5": resultado[
                    "f1_media_columnas_umbral_0.5"
                ],
                "f1_media_columnas_umbrales_optimos": resultado[
                    "f1_media_columnas_umbrales_optimos"
                ],
                "ganancia": resultado["ganancia"],
                "macro6_umbral_0.5": resultado["macro6_umbral_0.5"],
                "macro6_umbrales_optimos": resultado["macro6_umbrales_optimos"],
                "por_etiqueta": etiquetas,
                "por_pliegue": resultado["por_pliegue"],
                "pliegues_mejorados": resultado["pliegues_mejorados"],
                "pliegues_con_mejora_mayor_umbral": resultado[
                    "pliegues_con_mejora_mayor_umbral"
                ],
                "regla": resultado["regla"],
                "adoptado": adoptado,
            },
            ensure_ascii=False,
        ),
        "interpretation": (
            f"F1 con umbral 0.5 = {resultado['f1_media_columnas_umbral_0.5']:.4f}; "
            f"con umbrales optimos = {resultado['f1_media_columnas_umbrales_optimos']:.4f} "
            f"({resultado['ganancia']:+.4f} absoluto); la mejora se mantiene en "
            f"{resultado['pliegues_mejorados']}/5 pliegues"
        ),
        "decision": decision,
        "date": CONFIG["fecha"],
        "script": CONFIG["script"],
        "notes": (
            "Los umbrales se eligen sobre OOF y se aplican al refit con todo el "
            "train; la ganancia OOF es una cota optimista porque el umbral se "
            "optimiza con los mismos datos con que se mide"
        ),
    }


def construir_fila_submission(
    experiment_id: str,
    question: str,
    hypothesis: str,
    modelo: str,
    umbrales: dict[str, float] | None,
    ruta: Path,
    resumen_prediccion: dict,
    segundos: float,
    dimension: int,
    notas: str,
    f1_valor: float | None = None,
    macro_f1_valor: float | None = None,
) -> dict:
    """Construye una fila ``S02``/``S03`` del registro de experimentos.

    Args:
        experiment_id: ``S02`` o ``S03``.
        question: pregunta del experimento.
        hypothesis: hipotesis asociada.
        modelo: descripcion del modelo y los umbrales aplicados.
        umbrales: umbrales usados, o ``None`` si se uso 0.5 por etiqueta.
        ruta: ruta de la submission generada.
        resumen_prediccion: conteo de positivos por etiqueta en test y en train.
        segundos: segundos del refit y la prediccion.
        dimension: numero de features del vectorizador final.
        notas: nota metodologica.

    Returns:
        Diccionario con las columnas de ``COLUMNAS_EXPERIMENTS``.
    """
    return {
        "experiment_id": experiment_id,
        "competition": CONFIG["competencia"],
        "question": question,
        "hypothesis": hypothesis,
        "preprocessing": "texto crudo, lowercase, sin stopwords",
        "feature_groups": "WORD_NGRAM+CHAR_NGRAM",
        "representation": NOMBRE_REPRESENTACION,
        "model": modelo,
        "hyperparameters": json.dumps(
            {
                "modelo": MODELO_PARAMS,
                "umbral_base": CONFIG["umbral_base"],
                "umbrales_por_etiqueta": umbrales
                if umbrales is not None
                else {etiqueta: CONFIG["umbral_base"] for etiqueta in ETIQUETAS},
                "postproceso": "ninguno",
            },
            ensure_ascii=False,
        ),
        "validation": (
            "F1 out-of-fold reproducido de MOD-03b_BEST con "
            "StratifiedGroupKFold(5, shuffle=True, random_state=42); submission "
            "validada contra la plantilla oficial (columnas, 1154 filas, orden de "
            "ids, valores en {0,1}, sin nulos ni duplicados)"
        ),
        "f1": round(
            f1_valor if f1_valor is not None else CONFIG["f1_oof_referencia"], 4
        ),
        "macro_f1": round(
            macro_f1_valor if macro_f1_valor is not None else 0.8349, 4
        ),
        "weighted_f1": "",
        "precision": "",
        "recall": "",
        "training_time": round(segundos, 2),
        "dimension": dimension,
        "result": json.dumps(
            {
                "submission": str(ruta.relative_to(PROJECT_ROOT)),
                "f1_oof_umbral_0.5": round(CONFIG["f1_oof_referencia"], 6),
                "positivos_test": resumen_prediccion["positivos_test"],
                "tasa_positivos_test": resumen_prediccion["tasa_positivos_test"],
                "tasa_positivos_train": resumen_prediccion["tasa_positivos_train"],
            },
            ensure_ascii=False,
        ),
        "interpretation": (
            "Refit del ganador MOD-03b_BEST con todo el train (1731 filas) y "
            "prediccion del test; el modelo final queda en models/mod03b_lr_union.joblib"
        ),
        "decision": "submission generada; pendiente de envio por Carlos a Kaggle",
        "date": CONFIG["fecha"],
        "script": CONFIG["script"],
        "notes": notas,
    }


def construir_seccion_contexto(
    replica: dict,
    umbrales: dict,
    rutas: dict[str, Path],
    filas_experiments: Sequence[dict],
) -> str:
    """Redacta la seccion de handoff (maximo 20 lineas) para ``reports/contexto.md``.

    Args:
        replica: resultado de :func:`verificar_replicacion`.
        umbrales: resultado de :func:`evaluar_umbrales`.
        rutas: rutas de las submissions generadas por clave ``v02``/``v03``.
        filas_experiments: filas registradas en ``experiments.csv``.

    Returns:
        Texto de la seccion, con encabezado y sin linea final en blanco.

    Raises:
        RuntimeError: si la seccion generada excede las 20 lineas permitidas.
    """
    detalle_etiquetas = ", ".join(
        f"{etiqueta} {umbrales['por_etiqueta'][etiqueta]['umbral_optimo']:.2f} "
        f"(F1 {umbrales['por_etiqueta'][etiqueta]['f1_umbral_0.5']:.4f} -> "
        f"{umbrales['por_etiqueta'][etiqueta]['f1_umbral_optimo']:.4f})"
        for etiqueta in ETIQUETAS
    )
    if umbrales["adoptado"]:
        linea_decision = (
            f"Umbrales optimos ADOPTADOS (regla +{CONFIG['umbral_mejora']:.3f} y >= "
            f"{CONFIG['pliegues_minimos_mejora']}/5 pliegues): {umbrales['ganancia']:+.4f} "
            f"absoluto y {umbrales['pliegues_mejorados']}/5 pliegues mejorados; se genera v03."
        )
    else:
        linea_decision = (
            f"Umbrales optimos DESCARTADOS: ganancia {umbrales['ganancia']:+.4f} y "
            f"{umbrales['pliegues_mejorados']}/5 pliegues mejorados no cumplen la regla "
            f"(+{CONFIG['umbral_mejora']:.3f} y >= {CONFIG['pliegues_minimos_mejora']}/5); "
            "se mantiene 0.5 y NO se genera v03."
        )
    rutas_texto = "; ".join(
        f"`{ruta.relative_to(PROJECT_ROOT)}`" for ruta in rutas.values()
    )
    lineas = [
        f"## Fase 13a - ml-trainer ({CONFIG['momento_contexto']})",
        "",
        (
            "Refit de la configuracion ganadora `MOD-03b_BEST` (union TF-IDF word (1,2) + "
            "char_wb (3,5), min_df=2, sublinear_tf; LR balanced C=2.0 liblinear x3, "
            "umbral 0.5, sin post-proceso jerarquico) con "
            "`StratifiedGroupKFold(5, shuffle=True, random_state=42)` agrupado por "
            "near-duplicado y estratificado por HSxTRxAG."
        ),
        (
            f"F1 OOF reproducida: {replica['f1_media_columnas']:.6f} frente a "
            f"{replica['f1_referencia_fase_4_5']:.6f} de la Fase 4-5 "
            f"(diferencia {replica['diferencia']:+.3e}, tolerancia "
            f"{replica['tolerancia']:.0e}); redondeada 0.7927: replicacion exacta."
        ),
        (
            "Probabilidades OOF guardadas en `features/oof_mod03b.npz` (ids, y_true, "
            "proba (n,3), etiquetas) para ensembles futuros."
        ),
        (
            f"Umbrales optimos por etiqueta (rejilla 0.20-0.80 paso 0.01 sobre OOF, "
            f"nunca sobre test): {detalle_etiquetas}."
        ),
        (
            f"f1_media_columnas OOF con 0.5 = {umbrales['f1_media_columnas_umbral_0.5']:.4f} "
            f"-> con umbrales optimos = {umbrales['f1_media_columnas_umbrales_optimos']:.4f} "
            f"({umbrales['ganancia']:+.4f}); mejora por pliegue "
            f"{[round(fila['delta'], 4) for fila in umbrales['por_pliegue']]}."
        ),
        linea_decision,
        (
            "Modelo final reajustado con todo el train en `models/mod03b_lr_union.joblib` "
            "(vectorizador + 3 modelos + umbrales + etiquetas)."
        ),
        f"Submissions generadas y validadas contra la plantilla oficial: {rutas_texto}.",
        (
            "Registro: `experiments.csv` "
            f"({', '.join(fila['experiment_id'] for fila in filas_experiments)}) y "
            "`submissions/leaderboard_log.csv` con `kaggle_f1` vacio, pendiente de que "
            "Carlos envie las submissions."
        ),
        (
            "Sin fuga: el test solo se uso para predecir; los umbrales y la seleccion de "
            "modelo se decidieron con probabilidades out-of-fold. Detalle completo en "
            "`reports/fase13a_refit_umbral_metricas.json`."
        ),
        (
            "ADVERTENCIA de honestidad: la ganancia de umbrales (+0.0063) es una cota "
            "OPTIMISTA (umbral y metrica comparten el mismo OOF); la verificacion anidada "
            "del revisor da +0.0014 con 2/5 pliegues positivos -> v03 es una apuesta y v02 "
            "es el control."
        ),
    ]
    while lineas and lineas[-1] == "":
        lineas.pop()
    if len(lineas) > 20:
        raise RuntimeError(f"La seccion de contexto excede 20 lineas ({len(lineas)})")
    return "\n".join(lineas)


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
    encabezado = f"## Fase 13a - ml-trainer ({CONFIG['momento_contexto']})"
    texto = destino.read_text(encoding="utf-8") if destino.exists() else ""
    if encabezado in texto:
        texto = texto[: texto.index(encabezado)]
    destino.write_text(
        texto.rstrip("\n") + "\n\n" + seccion.rstrip("\n") + "\n", encoding="utf-8"
    )
    return destino


# ===== SECCION 5: flujo principal =====


def main() -> None:
    """Ejecuta la Fase 13a completa: refit, umbrales, submissions y registro."""
    inicio = time.perf_counter()
    train, test, grupos = cargar_datos()
    plantilla = data_module.read_submission()
    if list(plantilla["id"]) != list(test["id"]):
        raise ValueError(
            "El orden de ids del test no coincide con la plantilla oficial; "
            "abortado antes de generar submissions"
        )
    log(
        f"train {train.shape} | test {test.shape} | plantilla {plantilla.shape} | "
        f"grupos {grupos.nunique()}"
    )

    pliegues = construir_pliegues(train, grupos)
    y_true = train[list(ETIQUETAS)].to_numpy()

    # 1) Replicacion exacta del ganador y probabilidades out-of-fold.
    inicio_oof = time.perf_counter()
    probabilidad_oof = probabilidades_oof(train, pliegues)
    replica = verificar_replicacion(probabilidad_oof, y_true)
    segundos_oof = time.perf_counter() - inicio_oof
    ruta_npz = guardar_probabilidades_oof(train, probabilidad_oof)
    log(f"probabilidades OOF guardadas: {ruta_npz.name} en {segundos_oof:.2f} s")

    # 2) Ajuste de umbral por etiqueta sobre las probabilidades OOF.
    inicio_umbral = time.perf_counter()
    umbrales = evaluar_umbrales(y_true, probabilidad_oof, pliegues)
    segundos_umbral = time.perf_counter() - inicio_umbral

    # 3) Refit con todo el train y prediccion del test.
    inicio_refit = time.perf_counter()
    vectorizador, modelos, dimension = ajustar_modelo_final(train)
    probabilidad_test = predecir_test(test, vectorizador, modelos)
    umbrales_v02 = {etiqueta: CONFIG["umbral_base"] for etiqueta in ETIQUETAS}
    prediccion_v02 = predecir_con_umbrales(
        probabilidad_test, [umbrales_v02[etiqueta] for etiqueta in ETIQUETAS]
    )
    prediccion_v03 = predecir_con_umbrales(
        probabilidad_test,
        [umbrales["umbrales_optimos"][etiqueta] for etiqueta in ETIQUETAS],
    )
    segundos_refit = time.perf_counter() - inicio_refit

    rutas: dict[str, Path] = {}
    rutas["v02"] = escribir_submission(
        test, plantilla, prediccion_v02, CONFIG["submission_v02"]
    )
    if umbrales["adoptado"]:
        rutas["v03"] = escribir_submission(
            test, plantilla, prediccion_v03, CONFIG["submission_v03"]
        )
    else:
        ruta_descartada = CONFIG["submissions_dir"] / CONFIG["submission_v03"]
        if ruta_descartada.exists():
            ruta_descartada.unlink()
            log(f"v03 descartada: se elimino la submission obsoleta {ruta_descartada.name}")
        log("v03 no generada: los umbrales optimos no superan la regla pre-registrada")

    ruta_modelo = guardar_modelo(
        vectorizador,
        modelos,
        umbrales["umbrales_optimos"],
        bool(umbrales["adoptado"]),
        dimension,
    )
    log(f"modelo final guardado: {ruta_modelo.name} (dimension {dimension})")

    resumen_v02 = resumir_prediccion(train, prediccion_v02)
    resumen_v03 = resumir_prediccion(train, prediccion_v03)
    log(f"test v02: {json.dumps(resumen_v02, ensure_ascii=False)}")
    if umbrales["adoptado"]:
        log(f"test v03: {json.dumps(resumen_v03, ensure_ascii=False)}")

    # 4) Registro en experiments.csv.
    filas_experiments = [
        construir_fila_umbrales(umbrales, segundos_umbral, dimension),
        construir_fila_submission(
            CONFIG["id_submission_base"],
            (
                "¿Que submission genera la mejor configuracion (MOD-03b_BEST) "
                "reajustada con todo el train y umbral 0.5 por etiqueta?"
            ),
            (
                "El refit con todo el train reproduce la F1 local agrupada y mejora "
                "el LB frente a la submission v01"
            ),
            (
                "LogisticRegression(class_weight=balanced, C=2.0, liblinear) x3 con "
                "umbral 0.5 por etiqueta"
            ),
            None,
            rutas["v02"],
            resumen_v02,
            segundos_refit,
            dimension,
            (
                "Primera submission de la fase 13; sin post-proceso jerarquico (la "
                "Fase 4-5 lo midio en -0.0056); pendiente de envio por Carlos"
            ),
        ),
    ]
    if umbrales["adoptado"]:
        filas_experiments.append(
            construir_fila_submission(
                CONFIG["id_submission_umbral"],
                (
                    "¿Que submission genera el refit con los umbrales optimos por "
                    "etiqueta adoptados sobre las probabilidades out-of-fold?"
                ),
                (
                    "Los umbrales optimos por etiqueta superan la regla pre-registrada "
                    "(+0.005 absoluto y >= 3/5 pliegues)"
                ),
                (
                    "LogisticRegression(class_weight=balanced, C=2.0, liblinear) x3 con "
                    "umbral optimo por etiqueta"
                ),
                umbrales["umbrales_optimos"],
                rutas["v03"],
                resumen_v03,
                segundos_refit,
                dimension,
                (
                    "Umbrales elegidos solo con OOF y aplicados al refit; la ganancia "
                    "OOF es una cota optimista (umbral y metrica usan los mismos datos); "
                    "verificacion anidada del revisor: delta medio +0.0014, 2/5 pliegues "
                    "positivos -> v03 es una apuesta, v02 es el control"
                ),
                f1_valor=float(umbrales["f1_media_columnas_umbrales_optimos"]),
                macro_f1_valor=float(umbrales["macro6_umbrales_optimos"]),
            )
        )
    escritas = registrar_en_experiments(filas_experiments)
    log(f"filas escritas en experiments.csv: {escritas}")

    # 5) Registro en el leaderboard log (sin kaggle_f1 todavia).
    filas_leaderboard = [
        {
            "submission_id": CONFIG["id_leaderboard_base"],
            "fecha": CONFIG["fecha"],
            "archivo": str(rutas["v02"].relative_to(PROJECT_ROOT)),
            "local_f1": round(replica["f1_media_columnas"], 5),
            "kaggle_f1": "",
            "diferencia": "",
            "features": NOMBRE_REPRESENTACION,
            "modelo": (
                "LogisticRegression balanced C=2.0 x3 (HS, TR, AG), umbral 0.5, "
                "sin post-proceso"
            ),
            "hiperparametros": json.dumps(
                {
                    "C": 2.0,
                    "class_weight": "balanced",
                    "max_iter": 1000,
                    "solver": "liblinear",
                    "umbral": 0.5,
                },
                ensure_ascii=False,
            ),
            "interpretacion": (
                "Refit del ganador MOD-03b_BEST (F1 OOF 0.7927 con validacion "
                "agrupada). Pendiente de envio por Carlos; el LB esperado es ~0.79 "
                "segun el acuerdo local-LB observado en v01 (diferencia ~ -0.001)."
            ),
        }
    ]
    if umbrales["adoptado"]:
        filas_leaderboard.append(
            {
                "submission_id": CONFIG["id_leaderboard_umbral"],
                "fecha": CONFIG["fecha"],
                "archivo": str(rutas["v03"].relative_to(PROJECT_ROOT)),
                "local_f1": round(umbrales["f1_media_columnas_umbrales_optimos"], 5),
                "kaggle_f1": "",
                "diferencia": "",
                "features": NOMBRE_REPRESENTACION,
                "modelo": (
                    "LogisticRegression balanced C=2.0 x3 (HS, TR, AG) con umbral "
                    "optimo por etiqueta, sin post-proceso"
                ),
                "hiperparametros": json.dumps(
                    {
                        "C": 2.0,
                        "class_weight": "balanced",
                        "solver": "liblinear",
                        "umbrales": umbrales["umbrales_optimos"],
                    },
                    ensure_ascii=False,
                ),
                "interpretacion": (
                    "Variante con umbrales optimos por etiqueta (elegidos sobre OOF): "
                    f"F1 OOF {umbrales['f1_media_columnas_umbrales_optimos']:.4f} frente a "
                    f"{umbrales['f1_media_columnas_umbral_0.5']:.4f} con 0.5 "
                    f"({umbrales['ganancia']:+.4f}). La ganancia local es optimista "
                    "porque el umbral se ajusto con los mismos datos de medida."
                ),
            }
        )
    escritas_lb = registrar_en_leaderboard(filas_leaderboard)
    log(f"filas escritas en leaderboard_log.csv: {escritas_lb}")

    # 6) Handoff en reports/contexto.md.
    seccion = construir_seccion_contexto(replica, umbrales, rutas, filas_experiments)
    ruta_contexto = actualizar_contexto(seccion)
    log(f"contexto actualizado: {ruta_contexto.relative_to(PROJECT_ROOT)}")

    # 7) Detalle completo en JSON.
    detalle = {
        "metadata": {
            "fecha": CONFIG["fecha"],
            "script": CONFIG["script"],
            "competencia": CONFIG["competencia"],
            "fase": "Fase 13a (refit del ganador, umbrales y submissions v02/v03)",
            "configuracion": NOMBRE_MODELO,
            "representacion": NOMBRE_REPRESENTACION,
            "validacion": (
                f"StratifiedGroupKFold({CONFIG['n_splits']}, shuffle=True, "
                f"random_state={CONFIG['random_state']}) agrupado por near-duplicado y "
                "estratificado por HSxTRxAG"
            ),
            "postproceso_jerarquico": "no usado (empeora en -0.0056, Fase 4-5)",
            "dimension_final": dimension,
            "segundos_oof": round(segundos_oof, 2),
            "segundos_umbrales": round(segundos_umbral, 2),
            "segundos_refit": round(segundos_refit, 2),
        },
        "replicacion": replica,
        "umbrales": {
            clave: valor for clave, valor in umbrales.items() if clave != "metricas_optimas"
        },
        "submissions": {
            "v02": {
                "ruta": str(rutas["v02"].relative_to(PROJECT_ROOT)),
                "umbrales": umbrales_v02,
                "positivos": int(prediccion_v02.sum()),
                "prediccion_test": resumen_v02,
            },
            "v03_generada": bool(umbrales["adoptado"]),
            "v03": (
                {
                    "ruta": str(rutas["v03"].relative_to(PROJECT_ROOT)),
                    "umbrales": umbrales["umbrales_optimos"],
                    "positivos": int(prediccion_v03.sum()),
                    "prediccion_test": resumen_v03,
                }
                if umbrales["adoptado"]
                else None
            ),
        },
        "artefactos": {
            "oof_npz": str(ruta_npz.relative_to(PROJECT_ROOT)),
            "modelo_joblib": str(ruta_modelo.relative_to(PROJECT_ROOT)),
        },
    }
    CONFIG["metricas_json"].parent.mkdir(parents=True, exist_ok=True)
    CONFIG["metricas_json"].write_text(
        json.dumps(detalle, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )

    log(f"detalle: {CONFIG['metricas_json'].relative_to(PROJECT_ROOT)}")
    log(f"duracion total: {time.perf_counter() - inicio:.2f} s")


if __name__ == "__main__":
    try:
        main()
    except Exception:  # noqa: BLE001 - el punto de entrada CLI reporta todo
        import traceback

        traceback.print_exc()
        sys.exit(1)
