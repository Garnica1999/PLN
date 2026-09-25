"""experiments/25_submissions_candidatos.py
Fecha: 2026-09-25
Fase: FINAL CANDIDATE - materializacion de las submissions experimentales de
C-EMO, C-SYN y C-EMO-SYN.

Pregunta que responde
---------------------
El screening de candidatos finales (``experiments/23``) dejo tres metas
pequenos evaluados en el desarrollo (corrected_cv 0.7809, 0.7793 y 0.7797,
``reports/final_candidates.csv``) y sin submission materializada. Esta fase
genera los tres archivos experimentales para que Carlos decida, con su propio
criterio, si gasta una sonda de leaderboard. NO se envia nada a Kaggle, NO se
re-evalua ningun candidato y NO se toca V03 ni la sonda v08.

Definiciones congeladas (no se cambian en esta fase)
----------------------------------------------------
- C-EMO: proba de V03 (M2) + canal de emocion puro ``EMO-ALL@C=1.0``
  (6 features), meta Ridge alpha=0.3 por etiqueta (corrected_cv 0.780941).
- C-SYN: proba de V03 (M2) + canal de sintaxis puro ``SYN-3`` (6 features),
  meta Ridge alpha=1.0 por etiqueta (corrected_cv 0.779313).
- C-EMO-SYN: proba de V03 (M2) + emocion + sintaxis (9 features), meta Ridge
  alpha=1.0 por etiqueta (corrected_cv 0.779661).
Los canales puros y sus probabilidades OOF salen de
``reports/canal_ranking_final.csv`` y de los stores ``features/v2_oof_*.npz``;
los hiperparametros y el umbral del meta (0.5) son los de ``experiments/23``.

Criterio de datos (verificado en ``experiments/18``, patron de la submission v08)
---------------------------------------------------------------------------------
La submission v08 (``experiments/18_p7_holdout_y_submission_v08.py``) uso dos
conjuntos distintos para dos propositos distintos:

1. el meta se ajusta SOLO con las probabilidades OOF del DESARROLLO (1483
   filas; el holdout congelado y el test nunca participan del ajuste);
2. las features del test se construyen reajustando los modelos que las generan
   con el TRAIN COMPLETO (1731 = desarrollo + holdout; ``main`` de
   ``experiments/18``, paso 3: ``ajustar_base(nombre, train["text"], ...)``).

Este script reproduce ese mismo criterio: el meta (escalador + Ridge por
etiqueta) se ajusta con el OOF del desarrollo y las probabilidades del test se
obtienen reajustando con el train completo la base M2 de V03 y los dos canales
con sus recetas congeladas (``experiments/14``, ``16``, ``19`` y ``20``). El
test solo se usa para predecir.

Umbrales de decision
--------------------
El screening de ``experiments/23`` midio los candidatos con la regla congelada
del meta: probabilidad del meta >= 0.5 en las tres etiquetas. Por lo tanto los
umbrales por etiqueta de las tres submissions son HS 0.50 / TR 0.50 / AG 0.50
(no hay umbrales optimizados por etiqueta en estos candidatos).

Verificaciones de honestidad
----------------------------
- Antes de generar nada, se recomputa la corrected_cv de los tres candidatos en
  el desarrollo (memoria) y se compara con la documentada en
  ``reports/final_candidates.csv``; si no reproduce, el script aborta.
- Se verifica el sha256 de la particion congelada, de V03 y de la sonda v08 (no
  se sobrescriben: solo se leen).
- El script es idempotente: reescribe las submissions, el manifiesto y los
  reportes por identificador y conserva las filas previas de ``experiments.csv``.
  Los sha256 se guardan en ``reports/submission_hashes_v2.json`` para comprobar
  la reproducibilidad bit a bit en una segunda ejecucion.

Salidas
-------
``submissions/toxic_submission_v09_candidate_a.csv`` (C-EMO),
``submissions/toxic_submission_v10_candidate_b.csv`` (C-SYN),
``submissions/toxic_submission_v11_candidate_c.csv`` (C-EMO-SYN),
``submissions/submission_manifest.csv``, ``reports/submission_validation_v2.md``,
``reports/submission_hashes_v2.json``, la columna "Submission file" en
``reports/final_candidate_comparison.md`` y las filas ``FIN-08``/``FIN-09``/
``FIN-10`` en ``experiments.csv``.

Ejecucion
---------
    /home/carlos/venv/bin/python experiments/25_submissions_candidatos.py
"""

from __future__ import annotations

import hashlib
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
from src.evaluation import ETIQUETAS, metricas_multietiqueta

RUTA_P14 = PROJECT_ROOT / "experiments" / "14_oof_corregido_v2.py"
RUTA_P16 = PROJECT_ROOT / "experiments" / "16_targeting_v2.py"
RUTA_P17 = PROJECT_ROOT / "experiments" / "17_stacking_v2.py"
RUTA_P18 = PROJECT_ROOT / "experiments" / "18_p7_holdout_y_submission_v08.py"
RUTA_P19 = PROJECT_ROOT / "experiments" / "19_sintaxis_cita_negacion_v2.py"
RUTA_P20 = PROJECT_ROOT / "experiments" / "20_emocion_nbsvm_ortografia_v2.py"
RUTA_P23 = PROJECT_ROOT / "experiments" / "23_candidatos_finales.py"

PARTICION_PATH = PROJECT_ROOT / "data" / "processed" / "holdout_v2_ids.csv"
HOLDOUT_CONFIG_PATH = PROJECT_ROOT / "config" / "holdout_v2.json"
OOF_EMOCION_PATH = PROJECT_ROOT / "features" / "v2_oof_emocion_nbsvm_ortografia.npz"
OOF_SYNTAX_PATH = PROJECT_ROOT / "features" / "v2_oof_syntax.npz"
RANKING_CANALES_PATH = PROJECT_ROOT / "reports" / "canal_ranking_final.csv"
FINAL_CANDIDATES_PATH = PROJECT_ROOT / "reports" / "final_candidates.csv"
P7_RESULTADOS_PATH = PROJECT_ROOT / "reports" / "p7_holdout_resultados.json"
LEADERBOARD_PATH = PROJECT_ROOT / "submissions" / "leaderboard_log.csv"
PLANTILLA_PATH = PROJECT_ROOT / "submissions" / "toxic_submission_v00_plantilla.csv"
V03_PATH = PROJECT_ROOT / "submissions" / "toxic_submission_v03_lr_union_thr.csv"
V08_PATH = PROJECT_ROOT / "submissions" / "toxic_submission_v08_meta_v2.csv"
HASHES_V03_PATH = PROJECT_ROOT / "experiments" / "baseline_v03_frozen" / "hashes.txt"
MANIFEST_PATH = PROJECT_ROOT / "submissions" / "submission_manifest.csv"
VALIDACION_PATH = PROJECT_ROOT / "reports" / "submission_validation_v2.md"
HASHES_JSON_PATH = PROJECT_ROOT / "reports" / "submission_hashes_v2.json"
COMPARISON_PATH = PROJECT_ROOT / "reports" / "final_candidate_comparison.md"
EXPERIMENTS_CSV = PROJECT_ROOT / "experiments.csv"

COLUMNAS_SUBMISSION: Final[tuple[str, ...]] = ("id", "HS", "TR", "AG")

# Definicion congelada de cada candidato: familias de canal (orden de
# concatenacion), alpha del meta, corrected_cv documentada, archivo de salida e
# identificador de la fila de experiments.csv.
CANDIDATOS: Final[dict[str, dict[str, Any]]] = {
    "C-EMO": {
        "familias": ("emocion",),
        "alpha": 0.3,
        "corrected_cv": 0.7809412227404883,
        "submission": "submissions/toxic_submission_v09_candidate_a.csv",
        "experiment_id": "FIN-08-C-EMO",
    },
    "C-SYN": {
        "familias": ("sintaxis",),
        "alpha": 1.0,
        "corrected_cv": 0.7793129845794055,
        "submission": "submissions/toxic_submission_v10_candidate_b.csv",
        "experiment_id": "FIN-09-C-SYN",
    },
    "C-EMO-SYN": {
        "familias": ("emocion", "sintaxis"),
        "alpha": 1.0,
        "corrected_cv": 0.7796609169850828,
        "submission": "submissions/toxic_submission_v11_candidate_c.csv",
        "experiment_id": "FIN-10-C-EMO-SYN",
    },
}
# Canal puro elegido por el screening de experiments/23 para cada familia.
CANAL_ESPERADO: Final[dict[str, str]] = {
    "emocion": "EMO-ALL@C=1.0",
    "sintaxis": "SYN-3",
}
# Valor de C del canal SYN-3 seleccionado (segundo elemento de la clave OOF de
# experiments/19 y columna 3 del store; el store guarda C=[1,1,1,1,2,2,2,2]).
C_SINTAXIS: Final[float] = 1.0

CONFIG: Final[dict[str, Any]] = {
    "fecha": "2026-09-25",
    "n_dev": 1483,
    "n_holdout": 248,
    "n_train": 1731,
    "n_filas_test": 1154,
    "umbral_meta": 0.5,
    # Tolerancia de reproduccion: final_candidates.csv guarda corrected_cv con 6
    # decimales (error de redondeo <= 5e-7) y el reajuste es determinista.
    "tolerancia_reproduccion_dev": 1e-6,
    "prevalencia_test_estimada": {"HS": 0.406, "TR": 0.279, "AG": 0.356},
    "columnas_manifest": (
        "submission_id",
        "version",
        "model",
        "features",
        "thresholds",
        "corrected_cv",
        "historical_cv",
        "holdout",
        "hypothesis",
        "expected_behavior",
        "status",
    ),
    "columnas_experiments": (
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
    ),
}


def log(mensaje: str) -> None:
    """Imprime un mensaje de progreso con prefijo de fase.

    Args:
        mensaje: texto a imprimir.
    """
    print(f"[FIN-SUB] {mensaje}", flush=True)


def cargar_modulo(nombre: str, ruta: Path) -> Any:
    """Importa un script de experimento como modulo reutilizable.

    Args:
        nombre: nombre interno del modulo en ``sys.modules``.
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


def sha256_de(ruta: Path) -> str:
    """Calcula el sha256 de un archivo.

    Args:
        ruta: ruta del archivo.

    Returns:
        Hash hexadecimal en minusculas.

    Raises:
        FileNotFoundError: si el archivo no existe.
    """
    if not ruta.exists():
        raise FileNotFoundError(f"No existe {ruta}")
    return hashlib.sha256(ruta.read_bytes()).hexdigest()


# ===== SECCION 1: datos y particion congelada =====


def cargar_datos() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Carga train, desarrollo, holdout, test y la plantilla oficial.

    Reconstruye el desarrollo y el holdout desde la particion congelada
    verificando su sha256 contra ``config/holdout_v2.json``.

    Returns:
        Tupla ``(train, desarrollo, holdout, test, plantilla)``. El desarrollo y
        el holdout conservan el orden de filas de ``train``.

    Raises:
        RuntimeError: si el hash, los tamanos o el orden de ids no coinciden.
    """
    config = json.loads(HOLDOUT_CONFIG_PATH.read_text(encoding="utf-8"))
    hash_actual = sha256_de(PARTICION_PATH)
    if hash_actual != config["hash_ids_csv"]:
        raise RuntimeError(
            f"El hash de {PARTICION_PATH.name} no coincide con config/holdout_v2.json: "
            f"{hash_actual} vs {config['hash_ids_csv']}"
        )
    train = data_module.read_train()
    test = data_module.read_test()
    plantilla_raw = data_module.read_submission()
    particion = pd.read_csv(PARTICION_PATH)
    if not particion["id"].is_unique:
        raise RuntimeError("La particion congelada tiene ids duplicados")
    if set(particion["id"].astype(int)) != set(train["id"].astype(int)):
        raise RuntimeError("La particion congelada no cubre exactamente los ids del train")
    ids_dev = set(particion.loc[particion["particion"] == "dev", "id"].astype(int))
    ids_holdout = set(particion.loc[particion["particion"] == "holdout", "id"].astype(int))
    desarrollo = train[train["id"].astype(int).isin(ids_dev)].reset_index(drop=True)
    holdout = train[train["id"].astype(int).isin(ids_holdout)].reset_index(drop=True)
    if len(train) != CONFIG["n_train"] or len(desarrollo) != CONFIG["n_dev"]:
        raise RuntimeError(f"Tamano inesperado del train/desarrollo: {len(train)}/{len(desarrollo)}")
    if len(holdout) != CONFIG["n_holdout"]:
        raise RuntimeError(f"Tamano inesperado del holdout: {len(holdout)}")
    if len(test) != CONFIG["n_filas_test"]:
        raise RuntimeError(f"Tamano inesperado del test: {len(test)}")
    plantilla_v00 = pd.read_csv(PLANTILLA_PATH)
    if list(plantilla_v00.columns) != list(COLUMNAS_SUBMISSION):
        raise RuntimeError(f"Columnas inesperadas en {PLANTILLA_PATH.name}")
    if len(plantilla_v00) != CONFIG["n_filas_test"]:
        raise RuntimeError(f"Tamano inesperado de la plantilla v00: {len(plantilla_v00)}")
    if not pd.api.types.is_integer_dtype(plantilla_v00["id"]):
        raise RuntimeError(f"El id de {PLANTILLA_PATH.name} no es entero")
    if not plantilla_v00["id"].is_unique:
        raise RuntimeError(f"{PLANTILLA_PATH.name} tiene ids duplicados")
    # read_submission devuelve el id como texto (read_csv_robusto); se compara
    # como entero para verificar que la plantilla v00, data/raw/submission.csv y
    # el test comparten ids y orden.
    ids_v00 = plantilla_v00["id"].to_numpy().tolist()
    if ids_v00 != plantilla_raw["id"].astype(int).to_numpy().tolist():
        raise RuntimeError("Los ids de la plantilla v00 no coinciden con data/raw/submission.csv")
    if ids_v00 != test["id"].astype(int).to_numpy().tolist():
        raise RuntimeError("El orden de ids del test no coincide con la plantilla oficial")
    log(
        f"datos: train {train.shape} | desarrollo {len(desarrollo)} | holdout {len(holdout)} | "
        f"test {len(test)} | particion verificada ({hash_actual[:12]}...)"
    )
    return train, desarrollo, holdout, test, plantilla_v00


def cargar_oof_desarrollo(
    modulo_17: Any, modulo_23: Any, desarrollo: pd.DataFrame
) -> tuple[NDArray[np.float64], dict[str, NDArray[np.float64]], dict[str, str], Any]:
    """Carga el OOF del desarrollo de V03 (M2) y de los canales seleccionados.

    Args:
        modulo_17: modulo de ``experiments/17`` (carga y verifica el OOF de bases).
        modulo_23: modulo de ``experiments/23`` (carga los stores de canales).
        desarrollo: DataFrame del desarrollo.

    Returns:
        Tupla ``(proba_v03, canales, seleccion, pliegues)`` con la probabilidad
        de M2 ``(n, 3)``, las probabilidades de cada canal puro ``(n, 3)``, el
        nombre del canal elegido por familia y los pliegues corregidos.
    """
    _, grupos, pliegues = modulo_17.cargar_desarrollo()
    y_verdadera = desarrollo[list(ETIQUETAS)].to_numpy()
    pliegue_de_fila = np.zeros(len(desarrollo), dtype=int)
    for numero, (_, idx_validacion) in enumerate(pliegues, start=1):
        pliegue_de_fila[idx_validacion] = numero
    modulo_17.verificar_datos(desarrollo, grupos, pliegue_de_fila)
    proba_bases, nombres_bases, _, _, _ = modulo_17.cargar_oof(
        desarrollo, grupos, y_verdadera, pliegue_de_fila
    )
    proba_v03 = proba_bases[:, nombres_bases.index("M2_union_lr_c2"), :]
    canales: dict[str, NDArray[np.float64]] = {}
    for familia, ruta in (
        ("emocion", OOF_EMOCION_PATH),
        ("sintaxis", OOF_SYNTAX_PATH),
    ):
        probas, nombres = modulo_23.cargar_store_canales(
            ruta, desarrollo, pliegue_de_fila, y_verdadera
        )
        nombre = CANAL_ESPERADO[familia]
        if nombre not in nombres:
            raise RuntimeError(f"El canal {nombre} no esta en {ruta.name}: {list(nombres)}")
        canales[familia] = probas[:, nombres.index(nombre), :]
        log(f"OOF de canal {familia} cargado y verificado: {nombre}")
    return proba_v03, canales, dict(CANAL_ESPERADO), pliegues


def seleccionar_canales() -> dict[str, str]:
    """Lee el canal puro ganador por familia del ranking del screening.

    Returns:
        Mapa ``familia -> variante`` limitado a las familias usadas por los
        candidatos.

    Raises:
        RuntimeError: si la seleccion leida no coincide con la esperada
            (``EMO-ALL@C=1.0`` y ``SYN-3``), o si falta una familia.
    """
    ranking = pd.read_csv(RANKING_CANALES_PATH)
    seleccion: dict[str, str] = {}
    for familia in CANAL_ESPERADO:
        filas = ranking[(ranking["familia"] == familia) & (ranking["puro"])]
        if filas.empty:
            raise RuntimeError(f"El ranking no tiene variantes puras de {familia}")
        ganadora = filas.sort_values("f1_standalone_05", ascending=False).iloc[0]
        seleccion[familia] = str(ganadora["variante"])
        if seleccion[familia] != CANAL_ESPERADO[familia]:
            raise RuntimeError(
                f"El canal puro de {familia} cambio: {seleccion[familia]} vs "
                f"{CANAL_ESPERADO[familia]}"
            )
    log(f"canales puros verificados contra el ranking: {seleccion}")
    return seleccion


# ===== SECCION 2: verificacion de la definicion de los candidatos =====


def verificar_candidatos_dev(
    modulo_17: Any,
    modulo_23: Any,
    desarrollo: pd.DataFrame,
    pliegues: Any,
    fuentes: dict[str, Any],
) -> dict[str, dict[str, float]]:
    """Reproduce la corrected_cv de los tres candidatos antes de generar nada.

    Recomputa el meta con los mismos pliegues corregidos y el mismo
    hiperparametro congelado y compara con la corrected_cv documentada en
    ``reports/final_candidates.csv``. Es una verificacion de definicion, no una
    nueva evaluacion: no se prueba ningun hiperparametro alternativo.

    Args:
        modulo_17: modulo de ``experiments/17``.
        modulo_23: modulo de ``experiments/23``.
        desarrollo: DataFrame del desarrollo.
        pliegues: pliegues corregidos.
        fuentes: fuentes de features del desarrollo.

    Returns:
        Mapa ``candidato -> {corrected_cv, documentada, diferencia}``.

    Raises:
        RuntimeError: si algun candidato no reproduce su corrected_cv.
    """
    documentadas = pd.read_csv(FINAL_CANDIDATES_PATH)
    y_verdadera = desarrollo[list(ETIQUETAS)].to_numpy()
    detalle: dict[str, dict[str, float]] = {}
    for candidato, definicion in CANDIDATOS.items():
        matriz, columnas = modulo_23.construir_matriz_candidato(
            candidato, definicion["familias"], fuentes, desarrollo["text"]
        )
        especificacion = modulo_17.EspecMeta(
            variante=candidato,
            conjunto=f"cand:{candidato}",
            modelo="ridge",
            parametro=definicion["alpha"],
        )
        por_pliegue = {
            numero: (matriz[idx_entrenamiento], matriz[idx_validacion])
            for numero, (idx_entrenamiento, idx_validacion) in enumerate(pliegues, start=1)
        }
        probas = modulo_17.evaluar_meta_por_pliegue(
            por_pliegue, pliegues, y_verdadera, [especificacion]
        )
        prediccion = (probas[especificacion.key] >= CONFIG["umbral_meta"]).astype(int)
        corrected_cv = float(metricas_multietiqueta(y_verdadera, prediccion)["f1_media_columnas"])
        fila = documentadas[documentadas["experiment_id"].str.endswith(candidato)]
        if len(fila) != 1:
            raise RuntimeError(f"No hay una fila unica de {candidato} en final_candidates.csv")
        documentada = float(fila["corrected_cv"].iloc[0])
        if abs(documentada - definicion["corrected_cv"]) > CONFIG["tolerancia_reproduccion_dev"]:
            raise RuntimeError(
                f"La corrected_cv documentada de {candidato} cambio: {documentada:.6f} vs "
                f"{definicion['corrected_cv']:.6f} congelada en este script"
            )
        diferencia = abs(corrected_cv - documentada)
        if diferencia > CONFIG["tolerancia_reproduccion_dev"]:
            raise RuntimeError(
                f"{candidato} no reproduce su corrected_cv: {corrected_cv:.6f} vs "
                f"{documentada:.6f} (dif {diferencia:.2e})"
            )
        detalle[candidato] = {
            "corrected_cv": corrected_cv,
            "documentada": documentada,
            "diferencia": diferencia,
            "n_features": float(matriz.shape[1]),
            "columnas": len(columnas),
        }
        log(
            f"{candidato}: corrected_cv reproducida {corrected_cv:.6f} "
            f"(documentada {documentada:.6f}; {matriz.shape[1]} features)"
        )
    return detalle


# ===== SECCION 3: refit sobre el train completo y prediccion del test =====


def anotar_resto(textos: Sequence[str], anotador: Any) -> list[Any]:
    """Anota textos nuevos con el anotador simbolico de un script de fase v2.

    Args:
        textos: textos a anotar (holdout + test).
        anotador: ``anotar_con_spacy`` (P2) o ``anotar_textos`` (sintaxis).

    Returns:
        Lista de documentos anotados.

    Raises:
        RuntimeError: si el numero de documentos no coincide con los textos o si
            la anotacion cae al modo heuristico.
    """
    resultado = anotador(list(textos))
    documentos, modo = resultado if isinstance(resultado, tuple) else (resultado, "spacy")
    if modo != "spacy":
        raise RuntimeError(f"La anotacion de los textos nuevos no fue con spaCy: {modo}")
    if len(documentos) != len(textos):
        raise RuntimeError("La anotacion no coincide en tamano con los textos")
    return list(documentos)


def refit_emocion_test(
    modulo_20: Any,
    combinado: pd.DataFrame,
    documentos: Sequence[Any],
    y_combinado: NDArray[np.int_],
    idx_entrenamiento: NDArray[np.int_],
    idx_test: NDArray[np.int_],
    constantes: dict[str, Any],
) -> NDArray[np.float64]:
    """Reajusta el canal de emocion con el train completo y predice el test.

    Reutiliza ``evaluar_seccion_emocion`` de ``experiments/20`` con un
    pseudo-pliegue ``train -> test``: lexicos, escaladores y LR se ajustan solo
    con el train completo.

    Args:
        modulo_20: modulo de ``experiments/20`` (receta de emocion).
        combinado: frame con desarrollo + holdout + test en ese orden.
        documentos: documentos anotados alineados con ``combinado``.
        y_combinado: etiquetas del train completo y ceros de relleno para el
            test (el relleno nunca participa de ningun ajuste).
        idx_entrenamiento: posiciones del train completo en ``combinado``.
        idx_test: posiciones del test en ``combinado``.
        constantes: constantes estructurales de ``experiments/06``.

    Returns:
        Probabilidades del canal puro de emocion en el test ``(n_test, 3)``.
    """
    clave = CANAL_ESPERADO["emocion"]
    oof, _ = modulo_20.evaluar_seccion_emocion(
        combinado,
        documentos,
        y_combinado,
        [(idx_entrenamiento, idx_test)],
        constantes,
        modulo_20.cargar_p2(),
        modulo_20.cargar_receta_v03(),
    )
    if clave not in oof:
        raise RuntimeError(f"El canal {clave} no esta entre los evaluados: {sorted(oof)}")
    log(f"canal de emocion reajustado con el train completo: {clave} (variante densa estandarizada)")
    return oof[clave][idx_test]


def refit_sintaxis_test(
    modulo_19: Any,
    modulo_14: Any,
    combinado: pd.DataFrame,
    documentos: Sequence[Any],
    y_combinado: NDArray[np.int_],
    idx_entrenamiento: NDArray[np.int_],
    idx_test: NDArray[np.int_],
    constantes: dict[str, Any],
) -> NDArray[np.float64]:
    """Reajusta el canal de sintaxis con el train completo y predice el test.

    Reutiliza ``evaluar_pliegues`` de ``experiments/19`` con un pseudo-pliegue
    ``train -> test`` y extrae la clave ``("SYN-3", C=1.0)`` congelada.

    Args:
        modulo_19: modulo de ``experiments/19`` (receta de sintaxis).
        modulo_14: modulo de ``experiments/14`` (representacion word/char).
        combinado: frame con desarrollo + holdout + test en ese orden.
        documentos: documentos anotados alineados con ``combinado``.
        y_combinado: etiquetas del train completo y ceros de relleno para el
            test (el relleno nunca participa de ningun ajuste).
        idx_entrenamiento: posiciones del train completo en ``combinado``.
        idx_test: posiciones del test en ``combinado``.
        constantes: constantes estructurales de ``experiments/06``.

    Returns:
        Probabilidades del canal puro de sintaxis en el test ``(n_test, 3)``.
    """
    clave = (CANAL_ESPERADO["sintaxis"], C_SINTAXIS)
    resultado = modulo_19.evaluar_pliegues(
        combinado,
        documentos,
        y_combinado,
        [(idx_entrenamiento, idx_test)],
        constantes,
        modulo_14,
    )
    if clave not in resultado.oof:
        raise RuntimeError(f"El canal {clave} no esta entre los evaluados: {sorted(resultado.oof)}")
    log(
        f"canal de sintaxis reajustado con el train completo: {clave[0]} con C={clave[1]:g} "
        "(bloques POS+DEPS+MORPH)"
    )
    return resultado.oof[clave][idx_test]


def predecir_v03_test(
    modulo_18: Any,
    modulo_14: Any,
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[NDArray[np.float64], int]:
    """Reajusta la base M2 (V03) con el train completo y predice el test.

    Args:
        modulo_18: modulo de ``experiments/18`` (receta de refit de la v08).
        modulo_14: modulo de ``experiments/14`` (modelos base).
        train: train completo.
        test: test oficial.

    Returns:
        Tupla ``(proba, dimension)`` con las probabilidades de M2 en el test y
        la dimension del TF-IDF union ajustado con el train completo.
    """
    base = modulo_18.ajustar_base(
        "M2_union_lr_c2", train["text"], train[list(ETIQUETAS)].to_numpy(), modulo_14
    )
    proba = modulo_18.predecir_base(base, test["text"])
    log(
        f"base M2 (V03) reajustada con el train completo: dim {base.dimension}, "
        f"test {proba.shape}"
    )
    return proba, int(base.dimension)


# ===== SECCION 4: meta y submissions =====


def predecir_meta(
    modulo_17: Any,
    matriz_dev: NDArray[np.float64],
    y_dev: NDArray[np.int_],
    matriz_test: NDArray[np.float64],
    alpha: float,
    candidato: str,
) -> NDArray[np.float64]:
    """Ajusta el meta del candidato con el OOF del desarrollo y predice el test.

    Reutiliza ``ajustar_modelo_meta`` de ``experiments/17`` (mismo camino de
    codigo que la evaluacion fold a fold de ``experiments/23``): escalador
    estandar ajustado con el desarrollo y RidgeClassifier balanceado por
    etiqueta, con la sigmoide de la decision como probabilidad.

    Args:
        modulo_17: modulo de ``experiments/17``.
        matriz_dev: features del desarrollo (proba de V03 + canales).
        y_dev: etiquetas del desarrollo ``(n, 3)``.
        matriz_test: features del test en el mismo orden de columnas.
        alpha: hiperparametro del Ridge congelado del candidato.
        candidato: nombre del candidato (para el identificador de la especificacion).

    Returns:
        Probabilidades del meta en el test ``(n_test, 3)``.
    """
    escalador = StandardScaler().fit(matriz_dev)
    x_dev = escalador.transform(matriz_dev)
    x_test = escalador.transform(matriz_test)
    especificacion = modulo_17.EspecMeta(
        variante=candidato, conjunto=f"cand:{candidato}", modelo="ridge", parametro=alpha
    )
    proba = np.zeros((matriz_test.shape[0], len(ETIQUETAS)))
    for posicion, _ in enumerate(ETIQUETAS):
        proba[:, posicion] = modulo_17.ajustar_modelo_meta(
            especificacion, x_dev, y_dev[:, posicion], x_test
        )
    return proba


def verificar_submission(
    submission: pd.DataFrame, plantilla: pd.DataFrame, ruta: Path, etapa: str
) -> dict[str, Any]:
    """Verifica el contrato completo de una submission contra la plantilla v00.

    Args:
        submission: DataFrame candidato.
        plantilla: plantilla ``submissions/toxic_submission_v00_plantilla.csv``.
        ruta: ruta destino (para los mensajes de error).
        etapa: descripcion de la verificacion (antes/despues de escribir).

    Returns:
        Diccionario con el resultado de cada verificacion.

    Raises:
        ValueError: si columnas, filas, ids, valores, nulos, duplicados o dtype
            no cumplen el contrato.
    """
    if list(submission.columns) != list(COLUMNAS_SUBMISSION):
        raise ValueError(f"{ruta.name} [{etapa}]: columnas {list(submission.columns)}")
    if len(submission) != CONFIG["n_filas_test"]:
        raise ValueError(
            f"{ruta.name} [{etapa}]: {len(submission)} filas frente a "
            f"{CONFIG['n_filas_test']} esperadas"
        )
    if submission["id"].to_numpy().tolist() != plantilla["id"].to_numpy().tolist():
        raise ValueError(f"{ruta.name} [{etapa}]: los ids difieren de la plantilla v00")
    if submission.isna().any().any():
        raise ValueError(f"{ruta.name} [{etapa}]: hay valores nulos")
    if not submission["id"].is_unique:
        raise ValueError(f"{ruta.name} [{etapa}]: hay ids duplicados")
    if not pd.api.types.is_integer_dtype(submission["id"]):
        raise ValueError(f"{ruta.name} [{etapa}]: el id no es entero")
    for etiqueta in ETIQUETAS:
        if not pd.api.types.is_integer_dtype(submission[etiqueta]):
            raise ValueError(f"{ruta.name} [{etapa}]: {etiqueta} no es entero")
        valores = submission[etiqueta].to_numpy()
        if valores.min() < 0 or valores.max() > 1 or not np.isin(valores, (0, 1)).all():
            raise ValueError(f"{ruta.name} [{etapa}]: {etiqueta} fuera de {{0, 1}}")
    return {
        "columnas": list(submission.columns),
        "filas": len(submission),
        "ids_identicos_plantilla_v00": True,
        "valores_en_0_1": True,
        "sin_nulos": True,
        "sin_ids_duplicados": True,
        "dtype_valido": True,
    }


def escribir_submission(
    plantilla: pd.DataFrame, prediccion: NDArray[np.int_], ruta: Path
) -> dict[str, Any]:
    """Escribe y valida una submission binaria en el formato oficial.

    Args:
        plantilla: plantilla oficial (mismo orden de ids que el test).
        prediccion: matriz binaria ``(n, 3)`` en el orden de :data:`ETIQUETAS`.
        ruta: ruta de salida.

    Returns:
        Diccionario con positivos, tasas y verificaciones de la submission.

    Raises:
        ValueError: si la submission no cumple el formato oficial.
    """
    salida = plantilla.copy()
    for posicion, etiqueta in enumerate(ETIQUETAS):
        salida[etiqueta] = prediccion[:, posicion].astype(int)
    verificacion = verificar_submission(salida, plantilla, ruta, "antes de escribir")
    ruta.parent.mkdir(parents=True, exist_ok=True)
    salida.to_csv(ruta, index=False, encoding="utf-8")
    releida = pd.read_csv(ruta)
    verificacion = verificar_submission(releida, plantilla, ruta, "tras releer")
    positivos = {
        etiqueta: int(releida[etiqueta].sum()) for etiqueta in ETIQUETAS
    }
    tasas = {etiqueta: positivos[etiqueta] / len(releida) for etiqueta in ETIQUETAS}
    log(
        f"submission escrita y validada: {ruta.name} ({len(releida)} filas; positivos "
        f"HS {positivos['HS']}, TR {positivos['TR']}, AG {positivos['AG']})"
    )
    return {
        "archivo": str(ruta.relative_to(PROJECT_ROOT)),
        "n_filas": len(releida),
        "positivos": positivos,
        "tasas_positivos": tasas,
        "verificacion": verificacion,
        "sha256": sha256_de(ruta),
    }


# ===== SECCION 5: registros y reportes =====


def verificar_congelados() -> dict[str, Any]:
    """Comprueba que V03 y la sonda v08 siguen intactas y devuelve sus hashes.

    Returns:
        Diccionario con los hashes de V03 (contra
        ``experiments/baseline_v03_frozen/hashes.txt``), v08 y la plantilla v00.

    Raises:
        RuntimeError: si el sha256 de V03 no coincide con el artefacto congelado.
    """
    lineas = HASHES_V03_PATH.read_text(encoding="utf-8").splitlines()
    encontrados = [
        linea.split()[0]
        for linea in lineas
        if linea.split() and linea.split()[-1].endswith("submissions/toxic_submission_v03_lr_union_thr.csv")
    ]
    if len(encontrados) != 1:
        raise RuntimeError(
            f"No hay un unico hash de V03 en {HASHES_V03_PATH.name}: {encontrados}"
        )
    hash_congelado = encontrados[0]
    hash_v03 = sha256_de(V03_PATH)
    if hash_v03 != hash_congelado:
        raise RuntimeError(f"V03 cambio: {hash_v03} vs {hash_congelado} congelado")
    resultado = {
        "v03": {"archivo": str(V03_PATH.relative_to(PROJECT_ROOT)), "sha256": hash_v03,
                "hash_congelado": hash_congelado, "intacta": True},
        "v08": {"archivo": str(V08_PATH.relative_to(PROJECT_ROOT)), "sha256": sha256_de(V08_PATH)},
    }
    log(f"V03 intacta (sha256 {hash_v03[:12]}...); sonda v08 solo leida")
    return resultado


def escribir_hashes_json(resultado: dict[str, Any]) -> dict[str, Any]:
    """Guarda los sha256 de las submissions y verifica la reproducibilidad.

    Si ``reports/submission_hashes_v2.json`` ya existe (ejecucion previa), se
    comparan los hashes actuales con los guardados para comprobar que la
    generacion es reproducible bit a bit.

    Args:
        resultado: resultado completo de la fase.

    Returns:
        Diccionario con el bloque de reproducibilidad.
    """
    actuales = {
        nombre: datos["sha256"] for nombre, datos in resultado["submissions"].items()
    }
    previos: dict[str, str] = {}
    if HASHES_JSON_PATH.exists():
        previos = json.loads(HASHES_JSON_PATH.read_text(encoding="utf-8")).get("hashes", {})
    comparacion = {
        nombre: {
            "sha256_actual": hash_actual,
            "sha256_previo": previos.get(nombre),
            "identico": bool(previos.get(nombre) == hash_actual),
        }
        for nombre, hash_actual in actuales.items()
    }
    reproducibilidad = {
        "ejecucion_previa_detectada": bool(previos),
        "hashes_identicos": (
            all(datos["identico"] for datos in comparacion.values()) if previos else None
        ),
        "detalle": comparacion,
    }
    contenido = {
        "fecha": CONFIG["fecha"],
        "script": "experiments/25_submissions_candidatos.py",
        "hashes": actuales,
        "hashes_congelados": {
            "v03": resultado["v03"]["sha256"],
            "v08": resultado["v08"]["sha256"],
            "plantilla_v00": sha256_de(PLANTILLA_PATH),
        },
        "tolerancia_meta": CONFIG["umbral_meta"],
        "reproducibilidad": reproducibilidad,
    }
    HASHES_JSON_PATH.write_text(
        json.dumps(contenido, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    if previos:
        log(
            "reproducibilidad: hashes identicos en las 3 submissions"
            if reproducibilidad["hashes_identicos"]
            else "AVISO: los hashes cambiaron respecto de la ejecucion previa"
        )
    return reproducibilidad


def escribir_validacion_md(resultado: dict[str, Any]) -> None:
    """Escribe ``reports/submission_validation_v2.md`` con las verificaciones.

    Args:
        resultado: resultado completo de la fase.
    """
    lineas: list[str] = [
        "# Validacion de submissions v2 (V03 y candidatos experimentales)",
        "",
        f"Fecha: {CONFIG['fecha']}. Script: `experiments/25_submissions_candidatos.py`.",
        "Los archivos se validan contra `submissions/toxic_submission_v00_plantilla.csv`",
        "(1154 filas; ids identicos y en el mismo orden). V03 y v08 solo se leen.",
        "",
        "## Contrato de formato",
        "",
        "| archivo | sha256 (12) | columnas | filas | ids v00 (mismo orden) | valores {0,1} | sin NaN | sin duplicados | dtype | resultado |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for nombre, datos in resultado["submissions"].items():
        archivo = datos["archivo"]
        lineas.append(
            f"| {archivo} | {datos['sha256'][:12]} | id,HS,TR,AG | {datos['n_filas']} | OK | OK | "
            "OK | OK | int64 | OK |"
        )
    lineas.append(
        f"| {resultado['v03']['archivo']} | {resultado['v03']['sha256'][:12]} | id,HS,TR,AG | "
        f"{CONFIG['n_filas_test']} | OK | OK | OK | OK | int64 | OK (hash congelado intacto) |"
    )
    lineas.extend(
        [
            "",
            "## Umbrales por etiqueta y positivos predichos",
            "",
            (
                f"El screening de `experiments/23` congelo un unico umbral del meta "
                f"({CONFIG['umbral_meta']:.2f}) para las tres etiquetas; no hay umbrales "
                "optimizados por etiqueta en estos candidatos."
            ),
            "",
            "| submission | HS | TR | AG | positivos HS/TR/AG | tasa HS/TR/AG |",
            "|---|---|---|---|---|---|",
        ]
    )
    for nombre, datos in resultado["submissions"].items():
        tasas = datos["tasas_positivos"]
        positivos = datos["positivos"]
        lineas.append(
            f"| {nombre} ({Path(datos['archivo']).name}) | {CONFIG['umbral_meta']:.2f} | "
            f"{CONFIG['umbral_meta']:.2f} | {CONFIG['umbral_meta']:.2f} | "
            f"{positivos['HS']}/{positivos['TR']}/{positivos['AG']} | "
            f"{tasas['HS']:.4f}/{tasas['TR']:.4f}/{tasas['AG']:.4f} |"
        )
    prevalencia = CONFIG["prevalencia_test_estimada"]
    lineas.extend(
        [
            "",
            (
                "Referencias de tasa de positivos: train 0.4367/0.2998/0.3824; test estimado E00 "
                f"{prevalencia['HS']:.3f}/{prevalencia['TR']:.3f}/{prevalencia['AG']:.3f}."
            ),
            "",
            "## Criterio de datos y reproducibilidad",
            "",
            (
                "- Criterio verificado en `experiments/18_p7_holdout_y_submission_v08.py`: el meta se "
                "ajusta con el OOF del desarrollo (1483 filas) y las features del test se construyen "
                "con el train completo (1731 = desarrollo + holdout). El test solo se predice."
            ),
            "- Verificacion previa de definiciones: " + ", ".join(
                f"{nombre} corrected_cv {datos['corrected_cv']:.6f} (documentada "
                f"{datos['documentada']:.6f}, dif {datos['diferencia']:.1e})"
                for nombre, datos in resultado["verificacion_dev"].items()
            )
            + ".",
            (
                "- Reproducibilidad: se guardan los sha256 en `reports/submission_hashes_v2.json`; "
                "una segunda ejecucion reescribe los mismos bytes. Ultima comprobacion: "
                f"{resultado['reproducibilidad']['hashes_identicos']}."
            ),
            (
                f"- V03 intacta (sha256 {resultado['v03']['sha256'][:12]}...); la sonda v08 no se "
                "toca. Ninguna submission se sube a Kaggle en esta fase."
            ),
            "",
        ]
    )
    VALIDACION_PATH.write_text("\n".join(lineas), encoding="utf-8")
    log(f"reporte de validacion escrito: {VALIDACION_PATH.name}")


def escribir_manifest(resultado: dict[str, Any]) -> None:
    """Escribe ``submissions/submission_manifest.csv`` con lo ya documentado.

    Los valores se leen de los artefactos existentes (``leaderboard_log.csv``,
    ``final_candidates.csv`` y ``p7_holdout_resultados.json``); no se inventa
    ninguna metrica.

    Args:
        resultado: resultado completo de la fase.
    """
    leaderboard = pd.read_csv(LEADERBOARD_PATH, dtype=str, keep_default_na=False)
    finales = pd.read_csv(FINAL_CANDIDATES_PATH)
    p7 = json.loads(P7_RESULTADOS_PATH.read_text(encoding="utf-8"))
    fila_v03 = leaderboard[leaderboard["submission_id"] == "v03_lr_union_thr"].iloc[0]
    fila_v08 = leaderboard[leaderboard["submission_id"] == "v08_meta_v2"].iloc[0]
    filas: list[dict[str, Any]] = [
        {
            "submission_id": "v03_lr_union_thr",
            "version": "v03",
            "model": "LogisticRegression balanced C=2.0 x3 (M2 union word+char), umbral por etiqueta",
            "features": fila_v03["features"],
            "thresholds": 'HS 0.44 | TR 0.51 | AG 0.46',
            "corrected_cv": 0.7772,
            "historical_cv": 0.7989,
            "holdout": (
                f"{p7['paso_1_v03_holdout']['f1_media_columnas']:.4f} "
                f"(HS {p7['paso_1_v03_holdout']['f1_por_etiqueta']['HS']:.4f} | "
                f"TR {p7['paso_1_v03_holdout']['f1_por_etiqueta']['TR']:.4f} | "
                f"AG {p7['paso_1_v03_holdout']['f1_por_etiqueta']['AG']:.4f}) [P7-01]"
            ),
            "hypothesis": "Referencia congelada: modelo final actual antes de la sonda v08",
            "expected_behavior": (
                f"LB observado {fila_v03['kaggle_f1']} (mejor de las 8 sondas de la familia); "
                "gap dev-LB -0.0158 documentado en leaderboard_log.csv"
            ),
            "status": "baseline congelada; modelo final actual (no se regenera)",
        },
        {
            "submission_id": "v08_meta_v2",
            "version": "v08_meta_v2",
            "model": fila_v08["modelo"],
            "features": fila_v08["features"],
            "thresholds": "HS 0.50 | TR 0.50 | AG 0.50 (umbral del meta)",
            "corrected_cv": 0.7874,
            "historical_cv": "no medido (protocolo v2)",
            "holdout": (
                f"{p7['paso_2_meta_holdout']['f1_media_columnas']:.4f} "
                f"(HS {p7['paso_2_meta_holdout']['f1_por_etiqueta']['HS']:.4f} | "
                f"TR {p7['paso_2_meta_holdout']['f1_por_etiqueta']['TR']:.4f} | "
                f"AG {p7['paso_2_meta_holdout']['f1_por_etiqueta']['AG']:.4f}) [P7-02]"
            ),
            "hypothesis": "El meta de dependencia sobre los 4 bases supera a V03 en el LB",
            "expected_behavior": (
                f"Sonda enviada: LB {fila_v08['kaggle_f1']} (no supera a V03; D-020). El gap "
                "dev-LB del meta es -0.0058"
            ),
            "status": "sonda ya enviada; no se regenera",
        },
    ]
    for nombre, datos in resultado["submissions"].items():
        fila = finales[finales["experiment_id"].str.endswith(nombre)].iloc[0]
        archivo = Path(datos["archivo"]).name
        version = archivo.replace("toxic_submission_", "").replace(".csv", "")
        filas.append(
            {
                "submission_id": version,
                "version": version,
                "model": f"RidgeClassifier(alpha={CANDIDATOS[nombre]['alpha']:g}, balanced) x3 sobre "
                "probabilidades OOF de V03 + canales",
                "features": str(fila["representation"]),
                "thresholds": (
                    f"HS {CONFIG['umbral_meta']:.2f} | TR {CONFIG['umbral_meta']:.2f} | "
                    f"AG {CONFIG['umbral_meta']:.2f} (umbral del meta)"
                ),
                "corrected_cv": float(fila["corrected_cv"]),
                "historical_cv": str(fila["historical_cv"]),
                "holdout": str(fila["holdout"]),
                "hypothesis": str(fila["hypothesis"]),
                "expected_behavior": (
                    "Sin evidencia para adoptar: corrected_cv por debajo del meta v08 (0.7874) y "
                    "no PROMISING en el screening; si se enviara, el nivel esperado cae en la banda "
                    "documentada de las 8 sondas del LB (0.770-0.783). Archivo: "
                    f"submissions/{archivo}"
                ),
                "status": (
                    "candidato experimental generado; sin evidencia suficiente para sonda LB "
                    f"(archivo submissions/{archivo})"
                ),
            }
        )
    tabla = pd.DataFrame(filas, columns=list(CONFIG["columnas_manifest"]))
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(MANIFEST_PATH, index=False, encoding="utf-8")
    log(f"manifiesto escrito: {MANIFEST_PATH.name} ({len(tabla)} filas)")


def actualizar_comparison(resultado: dict[str, Any]) -> None:
    """Anade la columna "Submission file" a la comparacion de candidatos.

    La columna se anade una sola vez: si el encabezado ya termina en
    ``| Submission file |`` no se toca la tabla. Las celdas de la tabla pueden
    contener caracteres ``|`` (p. ej. la F1 por etiqueta del holdout), por lo
    que las filas se identifican por el id ``FIN-*`` al inicio de la linea en
    lugar de partir la linea por el separador de columnas.

    Args:
        resultado: resultado completo de la fase.
    """
    texto = COMPARISON_PATH.read_text(encoding="utf-8")
    reemplazos = (
        (
            "# Comparacion de candidatos finales (borrador, sin submissions)",
            "# Comparacion de candidatos finales (con submissions experimentales)",
        ),
        (
            "2 de esta fase). No se genera ninguna submission en esta tarea.",
            (
                "2 de esta fase). Las submissions experimentales de los 3 candidatos se "
                "materializaron despues en `experiments/25_submissions_candidatos.py`."
            ),
        ),
    )
    for viejo, nuevo in reemplazos:
        if viejo in texto:
            texto = texto.replace(viejo, nuevo)
    documentadas = pd.read_csv(FINAL_CANDIDATES_PATH)
    archivo_por_fila: dict[str, str] = {}
    for nombre, definicion in CANDIDATOS.items():
        fila = documentadas[documentadas["experiment_id"].str.endswith(nombre)]
        if len(fila) != 1:
            raise RuntimeError(f"No hay una fila unica de {nombre} en final_candidates.csv")
        archivo_por_fila[str(fila["experiment_id"].iloc[0])] = Path(
            definicion["submission"]
        ).name
    lineas = texto.splitlines()
    if any(linea.rstrip().endswith("| Submission file |") for linea in lineas):
        log("la columna Submission file ya estaba presente: tabla intacta")
    else:
        nuevas: list[str] = []
        en_tabla = False
        for linea in lineas:
            es_fila = linea.startswith("|") and linea.rstrip().endswith("|")
            if es_fila and "corrected CV" in linea and "candidato" in linea:
                nuevas.append(linea.rstrip() + " Submission file |")
                en_tabla = True
                continue
            if en_tabla and es_fila and not linea.replace("|", "").replace("-", "").strip():
                nuevas.append(linea.rstrip() + "---|")
                continue
            if en_tabla and es_fila:
                identificador = next(
                    (clave for clave in archivo_por_fila if linea.startswith(f"| {clave} ")),
                    None,
                )
                celda = archivo_por_fila[identificador] if identificador else "no generada"
                nuevas.append(linea.rstrip() + f" {celda} |")
                continue
            en_tabla = False
            nuevas.append(linea)
        texto = "\n".join(nuevas)
    encabezado = "## Materializacion de submissions experimentales (2026-09-25)"
    if encabezado not in texto:
        detalle = "\n".join(
            f"- {nombre}: `submissions/{Path(datos['archivo']).name}` (sha256 "
            f"{datos['sha256'][:12]}...; positivos HS/TR/AG "
            f"{datos['positivos']['HS']}/{datos['positivos']['TR']}/{datos['positivos']['AG']})"
            for nombre, datos in resultado["submissions"].items()
        )
        texto = texto.rstrip("\n") + (
            f"\n\n{encabezado}\n\n{detalle}\n"
            "- Artefactos experimentales sin sonda de LB: la decision documentada no cambia "
            "(V03 sigue siendo el modelo final y el screening no encontro evidencia para "
            "adoptar los candidatos).\n"
            "- No se re-evaluo nada: las definiciones y los hiperparametros congelados salen de "
            "`reports/final_candidates.csv` y se verifico que reproducen su corrected_cv.\n"
        )
    COMPARISON_PATH.write_text(texto, encoding="utf-8")
    log(f"comparacion actualizada con la columna Submission file: {COMPARISON_PATH.name}")


def registrar_experimentos(resultado: dict[str, Any]) -> None:
    """Anade (o conserva) las filas ``FIN-08``/``FIN-09``/``FIN-10``.

    Las filas previas del archivo no se modifican; si una fila de esta fase ya
    existe se conserva tal cual (idempotencia).

    Args:
        resultado: resultado completo de la fase.

    Raises:
        ValueError: si el esquema de ``experiments.csv`` no coincide con el esperado.
        RuntimeError: si quedan identificadores duplicados.
    """
    experiment_id_nuevos = {
        nombre: definicion["experiment_id"] for nombre, definicion in CANDIDATOS.items()
    }
    existente = pd.read_csv(EXPERIMENTS_CSV, dtype=str, keep_default_na=False)
    if list(existente.columns) != list(CONFIG["columnas_experiments"]):
        raise ValueError("El esquema de experiments.csv cambio inesperadamente")
    previas_nuevas = existente[existente["experiment_id"].isin(experiment_id_nuevos.values())]
    if not previas_nuevas["experiment_id"].is_unique:
        raise RuntimeError("Hay filas FIN-08..10 duplicadas en experiments.csv")
    previas = existente[~existente["experiment_id"].isin(experiment_id_nuevos.values())]
    filas: list[dict[str, str]] = []
    for nombre, definicion in CANDIDATOS.items():
        identificador = definicion["experiment_id"]
        if identificador in set(existente["experiment_id"]):
            continue
        datos = resultado["submissions"][nombre]
        detalle_dev = resultado["verificacion_dev"][nombre]
        filas.append(
            {
                "experiment_id": identificador,
                "competition": "KAGGLE_TOXIC_2026_PUJ",
                "question": (
                    f"Se materializa la submission experimental de {nombre} sin re-evaluar el "
                    "candidato y sin subirla a Kaggle?"
                ),
                "hypothesis": (
                    f"{nombre} generado por el screening de la fase final: la fusion tardia de las "
                    "probabilidades OOF de V03 con los canales no alcanzo el nivel del meta v08 "
                    "(0.7874) ni evidencia para adoptar el candidato"
                ),
                "preprocessing": (
                    "sin texto: probabilidades OOF del desarrollo (1483 filas) como features del "
                    f"meta; test predicho con base M2 y canales reajustados con el train completo "
                    f"({CONFIG['n_train']} filas)"
                ),
                "feature_groups": "META_PROBS_V03+CHANNEL_PROBS",
                "representation": (
                    "probabilidades del meta: V03 (M2) + "
                    + " + ".join(definicion["familias"])
                    + " (probabilidades OOF del desarrollo; test con refit del train completo)"
                ),
                "model": (
                    f"RidgeClassifier(alpha={definicion['alpha']:g}, balanced) por etiqueta; "
                    f"umbral del meta {CONFIG['umbral_meta']:.2f}"
                ),
                "hyperparameters": json.dumps(
                    {
                        "alpha": definicion["alpha"],
                        "umbral": CONFIG["umbral_meta"],
                        "canales": list(definicion["familias"]),
                        "corrected_cv_documentada": definicion["corrected_cv"],
                    },
                    ensure_ascii=False,
                ),
                "validation": (
                    "definicion reproducida en el desarrollo (misma corrected_cv documentada) y "
                    "submission validada contra la plantilla v00; sin evaluacion nueva"
                ),
                "f1": f"{detalle_dev['corrected_cv']:.6f}",
                "macro_f1": "",
                "weighted_f1": "",
                "precision": "",
                "recall": "",
                "training_time": f"{resultado['duracion_generacion_seg']:.2f}",
                "dimension": f"{int(detalle_dev['n_features'])}",
                "result": json.dumps(
                    {
                        "submission": datos["archivo"],
                        "sha256": datos["sha256"],
                        "n_filas": datos["n_filas"],
                        "positivos": datos["positivos"],
                        "tasas_positivos": datos["tasas_positivos"],
                        "umbral_meta": CONFIG["umbral_meta"],
                        "corrected_cv": detalle_dev["corrected_cv"],
                        "corrected_cv_documentada": detalle_dev["documentada"],
                        "diferencia_reproduccion": detalle_dev["diferencia"],
                    },
                    ensure_ascii=False,
                ),
                "interpretation": (
                    f"Submission experimental de {nombre} (corrected_cv {detalle_dev['corrected_cv']:.4f}); "
                    "no adoptado por el screening y sin sonda de LB: se genera solo como artefacto "
                    "de comparacion"
                ),
                "decision": "candidato experimental generado; sin evidencia suficiente para sonda LB",
                "date": CONFIG["fecha"],
                "script": "experiments/25_submissions_candidatos.py",
                "notes": (
                    "no se re-evaluo el candidato; test solo para predecir; V03 y v08 intactas; "
                    "sha256 en reports/submission_hashes_v2.json"
                ),
            }
        )
    if not filas:
        log("filas FIN-08..FIN-10 ya presentes: no se reescribe experiments.csv")
        return
    combinado = pd.concat([previas, pd.DataFrame(filas, columns=list(CONFIG["columnas_experiments"]))],
                          ignore_index=True)
    if combinado["experiment_id"].duplicated().any():
        raise RuntimeError("Hay experiment_id duplicados en experiments.csv")
    combinado.to_csv(EXPERIMENTS_CSV, index=False, encoding="utf-8", lineterminator="\n")
    log(
        f"experiments.csv actualizado con {len(filas)} filas FIN-08..10 "
        f"({len(combinado)} totales)"
    )


# ===== SECCION 6: main =====


def main() -> None:
    """Genera las tres submissions experimentales y sus registros."""
    inicio = time.perf_counter()
    modulo_14 = cargar_modulo("experimento_14_sub", RUTA_P14)
    modulo_16 = cargar_modulo("experimento_16_sub", RUTA_P16)
    modulo_17 = cargar_modulo("experimento_17_sub", RUTA_P17)
    modulo_18 = cargar_modulo("experimento_18_sub", RUTA_P18)
    modulo_19 = cargar_modulo("experimento_19_sub", RUTA_P19)
    modulo_20 = cargar_modulo("experimento_20_sub", RUTA_P20)
    modulo_23 = cargar_modulo("experimento_23_sub", RUTA_P23)

    train, desarrollo, holdout, test, plantilla = cargar_datos()
    y_dev = desarrollo[list(ETIQUETAS)].to_numpy()
    proba_v03, canales, seleccion, pliegues = cargar_oof_desarrollo(
        modulo_17, modulo_23, desarrollo
    )
    canales_ranking = seleccionar_canales()
    if canales_ranking != seleccion:
        raise RuntimeError(
            f"La seleccion de canales no coincide: {canales_ranking} vs {seleccion}"
        )
    fuentes_dev = {
        "modulo17": modulo_17,
        "v03": proba_v03,
        "canales": canales,
        "seleccion": seleccion,
    }
    verificacion_dev = verificar_candidatos_dev(
        modulo_17, modulo_23, desarrollo, pliegues, fuentes_dev
    )

    # Refit con el train completo (desarrollo + holdout) y prediccion del test.
    inicio_refit = time.perf_counter()
    combinado = pd.concat([desarrollo, holdout, test], ignore_index=True)
    y_combinado = np.vstack(
        [y_dev, holdout[list(ETIQUETAS)].to_numpy(), np.zeros((len(test), len(ETIQUETAS)), dtype=int)]
    )
    idx_entrenamiento = np.arange(CONFIG["n_train"], dtype=int)
    idx_test = np.arange(CONFIG["n_train"], len(combinado), dtype=int)
    if len(combinado) != CONFIG["n_train"] + CONFIG["n_filas_test"]:
        raise RuntimeError(f"Tamano inesperado del frame combinado: {len(combinado)}")
    constantes, origen_constantes = modulo_16.cargar_constantes_error_analysis()
    log(f"constantes estructurales: {origen_constantes}")
    docs16 = list(modulo_16.cargar_o_crear_anotaciones(desarrollo)[0]) + anotar_resto(
        combinado["text"].iloc[CONFIG["n_dev"]:].tolist(), modulo_16.anotar_con_spacy
    )
    docs19 = list(modulo_19.cargar_o_crear_anotaciones(desarrollo)[0]) + anotar_resto(
        combinado["text"].iloc[CONFIG["n_dev"]:].tolist(), modulo_19.anotar_textos
    )
    log(f"anotacion alineada con el frame combinado: {len(docs16)} documentos (dos flavours)")
    canales_test = {
        "emocion": refit_emocion_test(
            modulo_20, combinado, docs16, y_combinado,
            idx_entrenamiento, idx_test, constantes,
        ),
        "sintaxis": refit_sintaxis_test(
            modulo_19, modulo_14, combinado, docs19, y_combinado,
            idx_entrenamiento, idx_test, constantes,
        ),
    }
    proba_v03_test, dimension_v03 = predecir_v03_test(modulo_18, modulo_14, train, test)
    fuentes_test = {
        "modulo17": modulo_17,
        "v03": proba_v03_test,
        "canales": canales_test,
        "seleccion": seleccion,
    }
    log(f"refit y prediccion del test completados en {time.perf_counter() - inicio_refit:.1f} s")

    submissions: dict[str, dict[str, Any]] = {}
    predicciones: dict[str, tuple[NDArray[np.int_], list[str]]] = {}
    for nombre, definicion in CANDIDATOS.items():
        matriz_dev, columnas_dev = modulo_23.construir_matriz_candidato(
            nombre, definicion["familias"], fuentes_dev, desarrollo["text"]
        )
        matriz_test, columnas_test = modulo_23.construir_matriz_candidato(
            nombre, definicion["familias"], fuentes_test, test["text"]
        )
        if columnas_dev != columnas_test:
            raise RuntimeError(f"Las columnas del meta de {nombre} no coinciden dev/test")
        proba = predecir_meta(
            modulo_17, matriz_dev, y_dev, matriz_test, definicion["alpha"], nombre
        )
        predicciones[nombre] = ((proba >= CONFIG["umbral_meta"]).astype(int), columnas_test)
        log(
            f"{nombre}: meta Ridge alpha={definicion['alpha']:g} con {matriz_test.shape[1]} "
            f"features; test predicho"
        )
    for nombre, definicion in CANDIDATOS.items():
        prediccion, columnas_test = predicciones[nombre]
        submissions[nombre] = escribir_submission(
            plantilla, prediccion, PROJECT_ROOT / definicion["submission"]
        )
        submissions[nombre]["columnas_meta"] = columnas_test
        submissions[nombre]["dimension_v03_test"] = dimension_v03

    duracion = time.perf_counter() - inicio
    resultado: dict[str, Any] = {
        "fecha": CONFIG["fecha"],
        "script": "experiments/25_submissions_candidatos.py",
        "criterio_datos": (
            "meta con el OOF del desarrollo (1483 filas); base M2 y canales reajustados con el train "
            "completo (1731 = desarrollo + holdout) para predecir el test; test solo para predecir"
        ),
        "seleccion_canales": seleccion,
        "verificacion_dev": verificacion_dev,
        "submissions": submissions,
        "duracion_generacion_seg": round(duracion, 2),
    }
    resultado.update(verificar_congelados())
    resultado["reproducibilidad"] = escribir_hashes_json(resultado)
    escribir_validacion_md(resultado)
    escribir_manifest(resultado)
    actualizar_comparison(resultado)
    registrar_experimentos(resultado)
    log(f"duracion total: {duracion:.1f} s")


if __name__ == "__main__":
    main()
