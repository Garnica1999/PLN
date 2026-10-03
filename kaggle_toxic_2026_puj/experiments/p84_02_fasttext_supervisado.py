"""p84_02_fasttext_supervisado.py
Fecha: 2026-10-02
Descripcion: EXP-02 (P84-02): clasificador fastText supervisado desde cero (binario oficial
congelado) sobre las 1483 filas de desarrollo, fold-local y por etiqueta, con subwords 3-6
(brazo principal) y sin subwords (brazo control). La epoca E* se elige en particiones internas
agrupadas (partir_interna, 3 splits) y con E* se reentrena sobre outer_train para predecir
outer_val una sola vez por (brazo, pliegue, etiqueta). Se publican tres lecturas de umbral
(anidada, 0.5 y congelada de V03) y la comparacion pareada contra el OOF congelado de V03.
Estado del arte aplicado: fastText supervised de Joulin et al. (bolsa de n-gramas hasheados
con subwords), protocolo anidado P75, preregistro P84 (seccion 3 y convenciones comunes).

Especificacion: docs/p84_spec.md seccion 3 + seccion 1 (convenciones comunes).
Preregistro: docs/p84_preregistro.md (Gate 0, particiones internas y bootstrap).

Reglas duras de esta corrida:
- Gate 0: hashes congelados verificados; dev 1483 / holdout 0 / test 0; identidad V03 contra
  features/oof_v03_final.npz con tolerancia 1e-12 (fastText no usa el harness de V03).
- Nunca se abre data/raw/toxic_test.csv; nunca se evalua ni entrena con el holdout.
- El texto de outer_val nunca entra al archivo de entrenamiento (solo al predict final).
- Hiperparametros fijos del preregistro; unico cambio cosmetico: -verbose 0 en lugar del
  -verbose 2 citado en la spec (no altera el modelo; el runner exige logs limpios).
- Particiones internas con partir_interna y semillas 20261101 + 100*k + 7*j (j=0,1,2);
  E* = mediana de las 3 epocas elegidas; empate de F1 media: menor epoca.
- Umbral anidado por (brazo, pliegue, etiqueta) con probabilidades de inner_val de los 3
  splits, rejilla 0.20 a 0.80 paso 0.01; 0.5 y congelados 0.44/0.51/0.46 como controles.
- Determinismo: -thread 1 y -seed fijos; doble corrida identica sobre (brazo, etiqueta,
  pliegue) con tolerancia exactamente 0; el resultado se publica.
- Bootstrap pareado por filas, 10.000 remuestreos, semilla 20260926 (explicita).

Uso:
    /home/carlos/venv/bin/python experiments/p84_02_fasttext_supervisado.py
    /home/carlos/venv/bin/python experiments/p84_02_fasttext_supervisado.py --humo
    /home/carlos/venv/bin/python experiments/p84_02_fasttext_supervisado.py --pliegues 1,2 \\
        --brazos principal,control --etiquetas HS,TR,AG
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy import stats
from sklearn.metrics import precision_score, recall_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

from p73_02_textcnn_vs_random import partir_interna as partir_interna_p73

from src.data import read_train, sha256_of
from src.evaluation import (
    ETIQUETAS,
    bootstrap_diferencia_pareada,
    metrica_macro,
    metricas_multietiqueta,
)
from src.splits import grupos_fusionados, holdout_v2, pliegues_corregidos

# ===== SECCION 0: constantes congeladas, configuracion y cabeceras =====

HASH_TRAIN: Final[str] = "3d40e7345b588860cf4744914be88ddc094e4f03591516c8176ab40b7f76d42f"
HASH_OOF_V03: Final[str] = "559265816ce4f6312c87f008f90324f0eec70a74d1e6247364c162793338feab"
HASH_GRUPOS: Final[str] = "afabf9ae1c295ea8f203120bf28578cb4bdb4e8f16b89178d820baeaac62e622"
HASH_FASTTEXT: Final[str] = "59c51cc53802bcad9e7832145f6525a281d22d42e7c476bfbac545c0c392137f"

V03_CCV: Final[float] = 0.7772114525165721
V03_F1_AT_05: Final[float] = 0.771778476256122
UMBRALES_CONGELADOS: Final[np.ndarray] = np.array([0.44, 0.51, 0.46], dtype=np.float64)

TOL_IDENTIDAD: Final[float] = 1e-12
TOL_DETERMINISMO: Final[float] = 0.0
SEMILLA_BOOTSTRAP: Final[int] = 20260926
REPETICIONES_BOOTSTRAP: Final[int] = 10_000

SEMILLA_INTERNA_BASE: Final[int] = 20261101
N_INTERNOS: Final[int] = 3
FRACCION_INTERNA: Final[float] = 0.8
MAX_INTENTOS_INTERNA: Final[int] = 20
TOL_PREVALENCIA: Final[float] = 0.07

SEMILLA_FASTTEXT: Final[int] = 20260926
EPOCAS_CANDIDATAS: Final[tuple[int, ...]] = (5, 10, 20)
BRAZOS: Final[dict[str, tuple[int, int]]] = {"principal": (3, 6), "control": (0, 0)}

REJILLA_UMBRALES: Final[np.ndarray] = np.round(np.arange(20, 81, dtype=np.float64) / 100.0, 2)
UMBRAL_DEFECTO: Final[float] = 0.5

GATE_DELTA: Final[float] = 0.005
GATE_PLIEGUES: Final[int] = 4
GATE_ENSEMBLE_GAIN: Final[float] = 0.0015

FILAS_DEV: Final[int] = 1483
FILAS_HOLDOUT: Final[int] = 248
FILAS_TRAIN: Final[int] = 1731
TAMANOS_PLIEGUES: Final[list[int]] = [297, 297, 296, 296, 297]
INDICE_ETIQUETA: Final[dict[str, int]] = {etiqueta: i for i, etiqueta in enumerate(ETIQUETAS)}

HIPERPARAMETROS_FIJOS: Final[dict[str, Any]] = {
    "dim": 100,
    "lr": 0.1,
    "minCount": 2,
    "bucket": 2_000_000,
    "loss": "softmax",
    "wordNgrams": 2,
    "thread": 1,
    "seed": SEMILLA_FASTTEXT,
    "verbose": 0,
    "nota_verbose": "la spec citaba -verbose 2; se fija 0 (cosmetico, no altera el modelo)",
}

CONFIG: Final[dict[str, Any]] = {
    "bin_fasttext": PROJECT_ROOT / "tools" / "fastText" / "build" / "fasttext",
    "train_path": PROJECT_ROOT / "data" / "raw" / "toxic_train.csv",
    "grupos_path": PROJECT_ROOT / "data" / "processed" / "grupos_near_duplicados.csv",
    "oof_v03_path": PROJECT_ROOT / "features" / "oof_v03_final.npz",
    "experiments_csv": PROJECT_ROOT / "experiments.csv",
    "parciales_dir": PROJECT_ROOT / "reports" / "p84_02_parciales",
    "informe_json": PROJECT_ROOT / "reports" / "p84_02_fasttext.json",
    "informe_md": PROJECT_ROOT / "reports" / "p84_02_fasttext.md",
    "oof_npz": PROJECT_ROOT / "features" / "p84_02_oof_fasttext.npz",
    "tmp_dir": PROJECT_ROOT / "experimentos" / "p84_02_tmp",
    "fecha": "2026-10-02",
    "script": "experiments/p84_02_fasttext_supervisado.py",
    "competencia": "KAGGLE_TOXIC_2026_PUJ",
    "pliegues_por_defecto": (1, 2, 3, 4, 5),
}

# Nota: data/raw/toxic_test.csv no se referencia en ninguna ruta de este script.
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


def log(mensaje: str) -> None:
    """Imprime una linea de progreso con el prefijo de la fase.

    Args:
        mensaje: texto a mostrar.
    """
    print(f"[P84-02] {mensaje}", flush=True)


def verificar_ascii(texto: str, contexto: str) -> None:
    """Comprueba que un texto es ASCII puro.

    Args:
        texto: contenido a validar.
        contexto: descripcion usada en el mensaje de error.

    Raises:
        SystemExit: si hay algun caracter con codigo mayor que 127.
    """
    malos = sorted({caracter for caracter in texto if ord(caracter) > 127})
    if malos:
        raise SystemExit(f"STOP: {contexto} contiene caracteres no ASCII: {malos}")


def _json_default(objeto: Any) -> Any:
    """Convierte tipos de numpy y Path a tipos serializables en JSON.

    Args:
        objeto: valor a convertir.

    Returns:
        Valor serializable.

    Raises:
        TypeError: si el tipo no esta soportado.
    """
    if isinstance(objeto, np.integer):
        return int(objeto)
    if isinstance(objeto, np.floating):
        return float(objeto)
    if isinstance(objeto, np.bool_):
        return bool(objeto)
    if isinstance(objeto, np.ndarray):
        return objeto.tolist()
    if isinstance(objeto, Path):
        return str(objeto)
    raise TypeError(f"tipo no serializable en JSON: {type(objeto)}")


def escribir_json(ruta: Path, carga: dict[str, Any]) -> None:
    """Escribe un JSON ASCII con indentacion.

    Args:
        ruta: destino.
        carga: contenido serializable.
    """
    texto = json.dumps(carga, ensure_ascii=True, indent=2, default=_json_default)
    verificar_ascii(texto, f"JSON {ruta.name}")
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(texto + "\n", encoding="ascii")


def escribir_texto(ruta: Path, texto: str) -> None:
    """Escribe un informe de texto ASCII.

    Args:
        ruta: destino.
        texto: contenido ASCII.
    """
    verificar_ascii(texto, f"informe {ruta.name}")
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(texto, encoding="ascii")


def leer_ruta(datos: Any, ruta: str) -> Any:
    """Lee una ruta anidada del informe y falla con un mensaje claro si falta.

    Args:
        datos: estructura raiz (informe JSON de la corrida).
        ruta: claves separadas por punto, p. ej. "lecturas.principal.anidada".

    Returns:
        Valor almacenado en la ruta.

    Raises:
        SystemExit: si algun tramo de la ruta no existe o no es un objeto.
    """
    actual: Any = datos
    for tramo in ruta.split("."):
        if not isinstance(actual, dict) or tramo not in actual:
            raise SystemExit(f"STOP: falta la ruta '{ruta}' en el informe (tramo '{tramo}')")
        actual = actual[tramo]
    return actual


def verificar_sha256(ruta: Path, esperado: str, nombre: str) -> dict[str, Any]:
    """Verifica el sha256 de un artefacto congelado.

    Args:
        ruta: archivo a comprobar.
        esperado: sha256 congelado en el preregistro.
        nombre: etiqueta legible.

    Returns:
        Evidencia de la verificacion.

    Raises:
        SystemExit: si falta el archivo o el hash no coincide.
    """
    if not ruta.exists():
        raise SystemExit(f"STOP: falta el artefacto congelado {ruta}")
    real = sha256_of(ruta)
    if real != esperado:
        raise SystemExit(f"STOP: sha256 de {nombre} no coincide: {real} != {esperado}")
    return {
        "nombre": nombre,
        "ruta": str(ruta.relative_to(PROJECT_ROOT)),
        "sha256": real,
        "esperado": esperado,
        "coincide": True,
    }


def versiones_entorno() -> dict[str, str]:
    """Recoge las versiones relevantes del entorno.

    Returns:
        Diccionario con python, numpy, pandas, sklearn, scipy y device de computo.
    """
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "sklearn": sklearn.__version__,
        "scipy": scipy.__version__,
        "device": "cpu (fastText -thread 1; sin torch en esta corrida)",
    }


# ===== SECCION 1: Gate 0 (datos, particiones, contadores e identidad V03) =====


@dataclass
class DatosGate0:
    """Datos de desarrollo y particiones verificadas de Gate 0.

    Attributes:
        des: DataFrame de desarrollo (1483 filas).
        textos: textos alineados con des.
        y: etiquetas (1483, 3).
        ids: ids alineados con el OOF canonico.
        p_v03: probabilidades OOF de V03.
        pred_v03: predicciones de V03 con umbrales congelados.
        umbrales_v03: umbrales congelados [0.44, 0.51, 0.46].
        grupos_dev: grupos fusionados de desarrollo.
        pliegues: 5 pliegues (indices de entrenamiento, validacion).
        folde: numero de pliegue (1..5) por fila.
        evidencia: informe de Gate 0.
    """

    des: pd.DataFrame
    textos: np.ndarray
    y: np.ndarray
    ids: np.ndarray
    p_v03: np.ndarray
    pred_v03: np.ndarray
    umbrales_v03: np.ndarray
    grupos_dev: np.ndarray
    pliegues: list[tuple[np.ndarray, np.ndarray]]
    folde: np.ndarray
    evidencia: dict[str, Any]


def cargar_gate0() -> DatosGate0:
    """Ejecuta Gate 0: hashes, particiones, contadores e identidad del OOF congelado.

    Returns:
        Datos y evidencia de Gate 0.

    Raises:
        SystemExit: si cualquier verificacion congelada falla.
    """
    hashes = [
        verificar_sha256(CONFIG["train_path"], HASH_TRAIN, "toxic_train.csv"),
        verificar_sha256(CONFIG["oof_v03_path"], HASH_OOF_V03, "oof_v03_final.npz"),
        verificar_sha256(CONFIG["grupos_path"], HASH_GRUPOS, "grupos_near_duplicados.csv"),
        verificar_sha256(CONFIG["bin_fasttext"], HASH_FASTTEXT, "fasttext (binario oficial)"),
    ]

    train = read_train()
    if len(train) != FILAS_TRAIN:
        raise SystemExit(f"STOP: toxic_train.csv tiene {len(train)} filas, se esperaban {FILAS_TRAIN}")
    grupos = grupos_fusionados(train)
    indice_dev, indice_holdout = holdout_v2(train, grupos)
    if len(indice_dev) != FILAS_DEV or len(indice_holdout) != FILAS_HOLDOUT:
        raise SystemExit(
            f"STOP: particion inesperada dev={len(indice_dev)} holdout={len(indice_holdout)}"
        )
    des = train.iloc[indice_dev].reset_index(drop=True)
    grupos_dev = grupos[indice_dev]

    oof = np.load(CONFIG["oof_v03_path"], allow_pickle=True)
    y = oof["y_true"].astype(np.int64)
    p_v03 = oof["proba"].astype(np.float64)
    ids_v03 = oof["ids"].astype(np.int64)
    umbrales_v03 = oof["umbrales"].astype(np.float64)
    if not np.array_equal(des["id"].to_numpy(dtype=np.int64), ids_v03):
        raise SystemExit("STOP: los ids de desarrollo no coinciden con el OOF canonico")
    y_dev = des[list(ETIQUETAS)].to_numpy(dtype=np.int64)
    if not np.array_equal(y_dev, y):
        raise SystemExit("STOP: las etiquetas de desarrollo no coinciden con el OOF canonico")
    if not np.allclose(umbrales_v03, UMBRALES_CONGELADOS, atol=0.0, rtol=0.0):
        raise SystemExit(f"STOP: umbrales congelados inesperados: {umbrales_v03.tolist()}")
    pred_v03 = (p_v03 >= umbrales_v03[None, :]).astype(np.int64)
    if not np.array_equal(pred_v03, oof["pred"].astype(np.int64)):
        raise SystemExit("STOP: pred del npz no reproduce proba >= umbrales congelados")

    pliegues = pliegues_corregidos(des, grupos_dev)
    tamanos = [len(np.asarray(b)) for _, b in pliegues]
    if tamanos != TAMANOS_PLIEGUES:
        raise SystemExit(f"STOP: tamanos de pliegue inesperados: {tamanos}")
    folde = np.zeros(len(des), dtype=np.int64)
    for numero, (_, validacion) in enumerate(pliegues, start=1):
        folde[np.asarray(validacion)] = numero
    if not np.array_equal(folde, oof["pliegues"].astype(np.int64)):
        raise SystemExit("STOP: el numero de pliegue no coincide con el npz canonico")

    contadores = {
        "dev_filas_leidas": len(des),
        "holdout_filas_usadas": 0,
        "test_filas_usadas": 0,
        "toxic_test_abierto": False,
        "nota": "el holdout (248) se aparta por indice y no se usa; toxic_test.csv nunca se abre",
    }
    evidencia = {
        "hashes": hashes,
        "particiones": {
            "dev": len(des),
            "holdout": len(indice_holdout),
            "pliegues_tamanos": tamanos,
            "pliegues_sin_grupos_compartidos": True,
            "filas_cubiertas_una_vez": bool(
                (np.bincount(folde, minlength=6)[1:] == np.array(tamanos)).all()
            ),
        },
        "contadores": contadores,
        "oof_v03": {
            "fuente": str(oof["fuente"][0]) if "fuente" in oof else "",
            "corrected_cv_npz": float(oof["corrected_cv"][0]),
            "umbrales": umbrales_v03.tolist(),
            "f1_pliegues": [float(valor) for valor in np.asarray(oof["f1_pliegues"])],
            "f1_por_etiqueta": [float(valor) for valor in np.asarray(oof["f1_por_etiqueta"])],
        },
    }
    return DatosGate0(
        des=des,
        textos=des["text"].to_numpy(dtype=object),
        y=y,
        ids=ids_v03,
        p_v03=p_v03,
        pred_v03=pred_v03,
        umbrales_v03=umbrales_v03,
        grupos_dev=grupos_dev,
        pliegues=pliegues,
        folde=folde,
        evidencia=evidencia,
    )


def control_identidad_v03(datos: DatosGate0) -> dict[str, Any]:
    """Recalcula las referencias congeladas de V03 desde el npz.

    La comparacion de EXP-02 es contra el OOF congelado (fastText no usa el harness
    de V03), por lo que la identidad consiste en verificar hash, ids, etiquetas,
    pliegues, umbrales, pred y las dos cifras de referencia con tolerancia 1e-12.

    Args:
        datos: datos de Gate 0.

    Returns:
        Evidencia de identidad con tolerancia 1e-12.

    Raises:
        SystemExit: si alguna referencia no se reproduce.
    """
    oof = np.load(CONFIG["oof_v03_path"], allow_pickle=True)
    ccv_npz = float(oof["corrected_cv"][0])
    f1_congelado = float(metrica_macro(datos.y, datos.pred_v03))
    delta_ccv = abs(f1_congelado - ccv_npz)
    if delta_ccv > TOL_IDENTIDAD:
        raise SystemExit(
            f"STOP: corrected_cv no reproducible: {f1_congelado!r} vs {ccv_npz!r}"
        )
    if abs(ccv_npz - V03_CCV) > TOL_IDENTIDAD:
        raise SystemExit(f"STOP: corrected_cv del npz inesperado: {ccv_npz!r}")
    f1_at_05 = float(metrica_macro(datos.y, (datos.p_v03 >= 0.5).astype(np.int64)))
    delta_05 = abs(f1_at_05 - V03_F1_AT_05)
    if delta_05 > TOL_IDENTIDAD:
        raise SystemExit(f"STOP: V03@0.5 no reproducible: {f1_at_05!r} vs {V03_F1_AT_05!r}")
    f1_pliegues_recalculado = [
        float(metrica_macro(datos.y[np.asarray(val)], datos.pred_v03[np.asarray(val)]))
        for _, val in datos.pliegues
    ]
    f1_pliegues_npz = [float(valor) for valor in np.asarray(oof["f1_pliegues"])]
    if max(abs(a - b) for a, b in zip(f1_pliegues_recalculado, f1_pliegues_npz)) > TOL_IDENTIDAD:
        raise SystemExit("STOP: f1_pliegues de V03 no se reproduce desde pred")
    return {
        "corrected_cv_npz": ccv_npz,
        "corrected_cv_reconstruido": round(f1_congelado, 12),
        "delta_corrected_cv": delta_ccv,
        "f1_at_05_recalculado": round(f1_at_05, 12),
        "delta_f1_at_05": delta_05,
        "tolerancia": TOL_IDENTIDAD,
        "f1_pliegues_reconstruidos": [round(valor, 12) for valor in f1_pliegues_recalculado],
        "identidad_ok": True,
    }


# ===== SECCION 2: envoltorio del binario fastText =====


def sanitizar_texto(texto: str) -> str:
    """Normaliza el texto para el formato de una linea por ejemplo de fastText.

    Solo se aplica ``lower()`` (regla del preregistro) mas el reemplazo de saltos
    de linea, imprescindible para no romper el formato linea a linea.

    Args:
        texto: texto crudo.

    Returns:
        Texto en minusculas y sin saltos de linea.
    """
    return str(texto).lower().replace("\r", " ").replace("\n", " ")


def escribir_archivo_entrenamiento(
    ruta: Path, textos: np.ndarray, etiquetas_binarias: np.ndarray
) -> dict[str, Any]:
    """Escribe el archivo supervisado (una linea por fila del conjunto recibido).

    Args:
        ruta: destino del archivo.
        textos: textos del subconjunto de entrenamiento.
        etiquetas_binarias: etiqueta binaria por fila (1 = POS).

    Returns:
        Resumen con recuentos de clases y ruta.
    """
    lineas = []
    for texto, etiqueta in zip(textos, etiquetas_binarias, strict=True):
        marca = "__label__POS" if int(etiqueta) == 1 else "__label__NEG"
        lineas.append(f"{marca} {sanitizar_texto(texto)}")
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text("\n".join(lineas) + "\n", encoding="utf-8")
    n_positivos = int(np.sum(etiquetas_binarias))
    return {
        "ruta": str(ruta),
        "n_lineas": len(lineas),
        "n_positivos": n_positivos,
        "n_negativos": len(lineas) - n_positivos,
    }


def comando_entrenamiento(
    ruta_train: Path, base_modelo: Path, epocas: int, brazo: str
) -> list[str]:
    """Construye el comando ``fasttext supervised`` con los hiperparametros fijos.

    Args:
        ruta_train: archivo de entrenamiento.
        base_modelo: ruta base del modelo (fastText anade ``.bin``).
        epocas: numero de epocas E*.
        brazo: nombre del brazo (define minn y maxn).

    Returns:
        Lista de argumentos del comando.
    """
    minn, maxn = BRAZOS[brazo]
    comando = [
        str(CONFIG["bin_fasttext"]),
        "supervised",
        "-input",
        str(ruta_train),
        "-output",
        str(base_modelo),
        "-dim",
        str(HIPERPARAMETROS_FIJOS["dim"]),
        "-lr",
        str(HIPERPARAMETROS_FIJOS["lr"]),
        "-minCount",
        str(HIPERPARAMETROS_FIJOS["minCount"]),
        "-bucket",
        str(HIPERPARAMETROS_FIJOS["bucket"]),
        "-loss",
        str(HIPERPARAMETROS_FIJOS["loss"]),
        "-wordNgrams",
        str(HIPERPARAMETROS_FIJOS["wordNgrams"]),
        "-thread",
        str(HIPERPARAMETROS_FIJOS["thread"]),
        "-seed",
        str(HIPERPARAMETROS_FIJOS["seed"]),
        "-verbose",
        str(HIPERPARAMETROS_FIJOS["verbose"]),
    ]
    comando += ["-minn", str(minn), "-maxn", str(maxn), "-epoch", str(epocas)]
    return comando


def ejecutar_comando(
    comando: list[str], entrada: str | None = None
) -> subprocess.CompletedProcess[str]:
    """Ejecuta el binario fastText y valida el codigo de salida.

    Args:
        comando: argumentos del proceso.
        entrada: contenido enviado a stdin (None = sin stdin).

    Returns:
        Resultado del proceso.

    Raises:
        SystemExit: si fastText termina con codigo distinto de cero.
    """
    resultado = subprocess.run(
        comando,
        input=entrada,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )
    if resultado.returncode != 0:
        raise SystemExit(
            f"STOP: fastText fallo (codigo {resultado.returncode}): {' '.join(comando)}\n"
            f"{resultado.stderr[-2000:]}"
        )
    return resultado


def ruta_modelo(base: Path) -> Path:
    """Devuelve la ruta del ``.bin`` que produce fastText para una base dada.

    Args:
        base: ruta base usada en ``-output``.

    Returns:
        Ruta del archivo ``.bin``.
    """
    return base.with_name(base.name + ".bin")


def entrenar_modelo(
    ruta_train: Path, base_modelo: Path, epocas: int, brazo: str
) -> float:
    """Entrena un modelo supervisado y devuelve el tiempo empleado.

    Args:
        ruta_train: archivo de entrenamiento.
        base_modelo: ruta base del modelo.
        epocas: numero de epocas.
        brazo: nombre del brazo.

    Returns:
        Segundos de entrenamiento.

    Raises:
        SystemExit: si el modelo no se genera.
    """
    inicio = time.perf_counter()
    ejecutar_comando(comando_entrenamiento(ruta_train, base_modelo, epocas, brazo))
    segundos = time.perf_counter() - inicio
    if not ruta_modelo(base_modelo).exists():
        raise SystemExit(f"STOP: fastText no genero el modelo {ruta_modelo(base_modelo)}")
    return segundos


def eliminar_modelo(base_modelo: Path) -> None:
    """Elimina el modelo temporal (los modelos no son artefactos de la corrida).

    Args:
        base_modelo: ruta base usada en ``-output``.
    """
    for sufijo in (".bin", ".vec"):
        ruta = base_modelo.with_name(base_modelo.name + sufijo)
        if ruta.exists():
            ruta.unlink()


def limpiar_temporales_txt(directorio: Path) -> list[str]:
    """Elimina los archivos .txt temporales al cerrar un par (pliegue, brazo).

    Los modelos .bin/.vec ya se eliminan con :func:`eliminar_modelo`; este
    barrido retira los archivos de entrenamiento .txt del directorio temporal.
    Los JSON de parciales viven en reports/p84_02_parciales y no se tocan.

    Args:
        directorio: directorio temporal de la corrida.

    Returns:
        Nombres de los archivos eliminados, en orden alfabetico.
    """
    if not directorio.exists():
        return []
    eliminados: list[str] = []
    for ruta in sorted(directorio.glob("*.txt")):
        ruta.unlink()
        eliminados.append(ruta.name)
    return eliminados


def parsear_linea_prob_positivo(linea: str) -> float:
    """Extrae la probabilidad de ``__label__POS`` por nombre, nunca por posicion.

    Args:
        linea: linea de salida de ``predict-prob`` con pares etiqueta/probabilidad.

    Returns:
        Probabilidad de ``__label__POS``.

    Raises:
        RuntimeError: si la linea no contiene el token ``__label__POS``.
    """
    tokens = linea.split()
    for indice in range(0, len(tokens) - 1, 2):
        if tokens[indice] == "__label__POS":
            return float(tokens[indice + 1])
    raise RuntimeError(f"predict-prob no devolvio __label__POS en la linea: {linea!r}")


def calcular_proba_positivo(base_modelo: Path, textos: np.ndarray) -> np.ndarray:
    """Ejecuta ``predict-prob`` con k=2 sobre los textos dados.

    Args:
        base_modelo: ruta base del modelo entrenado.
        textos: textos a predecir.

    Returns:
        Vector de probabilidades de ``__label__POS``.

    Raises:
        SystemExit: si el numero de lineas de salida no coincide con las entradas.
    """
    modelo = ruta_modelo(base_modelo)
    entrada = "\n".join(sanitizar_texto(texto) for texto in textos) + "\n"
    resultado = ejecutar_comando(
        [str(CONFIG["bin_fasttext"]), "predict-prob", str(modelo), "-", "2"], entrada
    )
    lineas = [linea for linea in resultado.stdout.splitlines() if linea.strip()]
    if len(lineas) != len(textos):
        raise SystemExit(
            f"STOP: predict-prob devolvio {len(lineas)} lineas para {len(textos)} textos"
        )
    return np.array([parsear_linea_prob_positivo(linea) for linea in lineas], dtype=np.float64)


def entrenar_si_posible(
    resumen_train: dict[str, Any], base_modelo: Path, epocas: int, brazo: str
) -> tuple[float, str | None]:
    """Entrena el modelo binario salvo que no haya positivos en el conjunto.

    Args:
        resumen_train: resumen de :func:`escribir_archivo_entrenamiento`.
        base_modelo: ruta base del modelo.
        epocas: numero de epocas.
        brazo: nombre del brazo.

    Returns:
        Tupla (segundos, incidencia o None).
    """
    if resumen_train["n_positivos"] == 0:
        return 0.0, (
            "sin positivos en el conjunto de entrenamiento "
            f"({Path(resumen_train['ruta']).name}): probabilidad POS fijada a 0"
        )
    return entrenar_modelo(Path(resumen_train["ruta"]), base_modelo, epocas, brazo), None


def predecir_o_cero(
    resumen_train: dict[str, Any], base_modelo: Path, textos: np.ndarray
) -> np.ndarray:
    """Predice con el modelo entrenado o devuelve ceros si no hay positivos.

    Args:
        resumen_train: resumen de :func:`escribir_archivo_entrenamiento`.
        base_modelo: ruta base del modelo.
        textos: textos a predecir.

    Returns:
        Vector de probabilidades de ``__label__POS``.
    """
    if resumen_train["n_positivos"] == 0:
        return np.zeros(len(textos), dtype=np.float64)
    return calcular_proba_positivo(base_modelo, textos)


def verificar_determinismo(
    pliegue: int,
    brazo: str,
    etiqueta: str,
    ruta_train: Path,
    textos_val: np.ndarray,
    epocas: int,
) -> dict[str, Any]:
    """Entrena dos veces la misma configuracion y compara las probabilidades.

    Args:
        pliegue: pliegue externo de la verificacion.
        brazo: brazo de la verificacion.
        etiqueta: etiqueta de la verificacion.
        ruta_train: archivo de entrenamiento comun.
        textos_val: textos de validacion externa a predecir.
        epocas: epocas de la doble corrida.

    Returns:
        Evidencia con ``max_delta_probabilidades`` e ``identicas`` (tolerancia 0).
    """
    inicio = time.perf_counter()
    bases = [
        CONFIG["tmp_dir"] / f"det_a_f{pliegue}_{brazo}_{etiqueta}",
        CONFIG["tmp_dir"] / f"det_b_f{pliegue}_{brazo}_{etiqueta}",
    ]
    probas: list[np.ndarray] = []
    for base in bases:
        entrenar_modelo(ruta_train, base, epocas, brazo)
        probas.append(calcular_proba_positivo(base, textos_val))
        eliminar_modelo(base)
    delta = float(np.max(np.abs(probas[0] - probas[1]))) if len(textos_val) else 0.0
    return {
        "brazo": brazo,
        "etiqueta": etiqueta,
        "pliegue": pliegue,
        "epocas": epocas,
        "max_delta_probabilidades": delta,
        "identicas": bool(delta <= TOL_DETERMINISMO),
        "tolerancia": TOL_DETERMINISMO,
        "segundos_extra": round(time.perf_counter() - inicio, 2),
    }


# ===== SECCION 3: particiones internas, epocas y umbrales =====


def _partir_interna_local(
    grupos: np.ndarray, y: np.ndarray, semilla: int
) -> tuple[np.ndarray, int]:
    """Reimplementacion local de partir_interna para verificar equivalencia.

    Args:
        grupos: etiqueta de cluster por fila.
        y: etiquetas.
        semilla: semilla base.

    Returns:
        Mascara booleana de inner_train y el numero de intentos usados.
    """
    unicos = np.unique(grupos)
    prevalencia = y.mean(axis=0)
    mascara = np.zeros(len(grupos), dtype=bool)
    intentos = 0
    for intento in range(MAX_INTENTOS_INTERNA):
        intentos = intento + 1
        rng = np.random.default_rng(semilla + intento)
        permutacion = rng.permutation(len(unicos))
        n_sel = round(FRACCION_INTERNA * len(unicos))
        elegidos = set(unicos[permutacion[:n_sel]].tolist())
        mascara = np.array([grupo in elegidos for grupo in grupos])
        if not mascara.any() or mascara.all():
            continue
        prevalencia_val = y[~mascara].mean(axis=0)
        if float(np.abs(prevalencia - prevalencia_val).max()) <= TOL_PREVALENCIA:
            break
    return mascara, intentos


def partir_interna_equivalente(
    grupos: np.ndarray, y: np.ndarray, semilla: int
) -> tuple[np.ndarray, int, dict[str, Any]]:
    """Ejecuta partir_interna de P73 y verifica equivalencia con la copia local.

    Args:
        grupos: etiqueta de cluster por fila.
        y: etiquetas.
        semilla: semilla base.

    Returns:
        Tupla (mascara inner_train, intentos, evidencia de equivalencia).

    Raises:
        SystemExit: si la copia local no reproduce la mascara canonica.
    """
    mascara_p73, intentos_p73 = partir_interna_p73(grupos, y, semilla)
    mascara_local, intentos_local = _partir_interna_local(grupos, y, semilla)
    if not (
        np.array_equal(mascara_p73, mascara_local) and intentos_p73 == intentos_local
    ):
        raise SystemExit(
            f"STOP: partir_interna no es equivalente para la semilla {semilla} "
            f"(intentos p73={intentos_p73}, local={intentos_local})"
        )
    return mascara_p73, intentos_p73, {
        "semilla": int(semilla),
        "intentos_p73": int(intentos_p73),
        "intentos_local": int(intentos_local),
        "equivalentes": True,
    }


def evaluar_prevalencia_interna(y: np.ndarray, mascara: np.ndarray) -> dict[str, Any]:
    """Comprueba la tolerancia de prevalencia marginal del holdout interno.

    Args:
        y: etiquetas del pliegue externo.
        mascara: mascara booleana de inner_train.

    Returns:
        Evidencia con el maximo delta de prevalencia por etiqueta.
    """
    prevalencia_global = y.mean(axis=0)
    prevalencia_val = y[~mascara].mean(axis=0)
    max_delta = float(np.abs(prevalencia_global - prevalencia_val).max())
    return {
        "max_delta_prevalencia": round(max_delta, 6),
        "tolerancia": TOL_PREVALENCIA,
        "cumple": bool(max_delta <= TOL_PREVALENCIA),
    }


def f1_binaria(verdad: np.ndarray, pred: np.ndarray) -> float:
    """F1 binaria de la clase positiva con zero_division=0.

    Args:
        verdad: etiquetas binarias.
        pred: predicciones binarias.

    Returns:
        F1 de la clase positiva.
    """
    verdaderos = verdad == 1
    predichos = pred == 1
    tp = float(np.sum(verdaderos & predichos))
    fp = float(np.sum(~verdaderos & predichos))
    fn = float(np.sum(verdaderos & ~predichos))
    denominador = 2 * tp + fp + fn
    return 2 * tp / denominador if denominador > 0 else 0.0


def elegir_epoca_por_split(
    f1_por_epoca: dict[int, dict[str, float]], etiquetas: Sequence[str]
) -> tuple[int, dict[int, float]]:
    """Elige la epoca de F1@0.5 media maxima en inner_val (empate: menor epoca).

    Args:
        f1_por_epoca: F1@0.5 por epoca y etiqueta.
        etiquetas: etiquetas procesadas.

    Returns:
        Tupla (epoca elegida, F1 media por epoca).
    """
    medias = {
        epoca: float(np.mean([f1_por_epoca[epoca][etiqueta] for etiqueta in etiquetas]))
        for epoca in EPOCAS_CANDIDATAS
    }
    mejor = min(EPOCAS_CANDIDATAS, key=lambda epoca: (-medias[epoca], epoca))
    return int(mejor), medias


def elegir_umbrales_anidados(
    proba: np.ndarray, y: np.ndarray, etiquetas: Sequence[str]
) -> dict[str, float]:
    """Elige un umbral por etiqueta con las probabilidades internas agregadas.

    La rejilla declarada es 0.20 a 0.80 paso 0.01 y el criterio es la F1 maxima
    por etiqueta; en caso de empate se toma el primer maximo (menor umbral).

    Args:
        proba: probabilidades internas agregadas (n, 3).
        y: etiquetas internas (n, 3).
        etiquetas: etiquetas procesadas.

    Returns:
        Diccionario etiqueta -> umbral.
    """
    umbrales: dict[str, float] = {}
    for etiqueta in etiquetas:
        columna = INDICE_ETIQUETA[etiqueta]
        valores = [
            f1_binaria(y[:, columna], (proba[:, columna] >= umbral).astype(np.int64))
            for umbral in REJILLA_UMBRALES
        ]
        indice = int(np.argmax(valores))
        umbrales[etiqueta] = float(REJILLA_UMBRALES[indice])
    return umbrales


# ===== SECCION 4: procesamiento de un (pliegue, brazo) =====


@dataclass
class ResultadoBrazoPliegue:
    """Resultado completo de un par (pliegue externo, brazo).

    Attributes:
        pliegue: pliegue externo 1..5.
        brazo: nombre del brazo.
        etiquetas: etiquetas procesadas.
        e_estrella: mediana de las epocas elegidas en los 3 splits internos.
        epocas_por_split: epoca elegida por split j.
        intentos_por_split: intentos de partir_interna por split.
        equivalencias: evidencia de equivalencia por split.
        prevalencias: evidencia de prevalencia por split.
        f1_por_epoca: F1@0.5 por split, epoca y etiqueta.
        umbrales_anidados: umbral por etiqueta elegido en inner_val.
        proba_outer: probabilidad POS en outer_val (n_val, 3).
        y_outer: etiquetas de outer_val (n_val, 3).
        proba_outer_train: probabilidad POS en outer_train (diagnostico de brecha).
        pred_anidada: lectura primaria (umbral anidado).
        pred_umbral_0_5: control de arquitectura.
        pred_congelada: control de transferencia (0.44/0.51/0.46).
        f1_val_05: F1@0.5 por etiqueta en outer_val.
        f1_train_05: F1@0.5 por etiqueta en outer_train.
        rutas_train_outer: archivo de entrenamiento externo por etiqueta.
        resumenes_train_outer: resumen de clases del archivo externo por etiqueta.
        segundos: segundos de entrenamiento del par.
        incidencias: avisos registrados.
    """

    pliegue: int
    brazo: str
    etiquetas: tuple[str, ...]
    e_estrella: int
    epocas_por_split: list[int]
    intentos_por_split: list[int]
    equivalencias: list[dict[str, Any]]
    prevalencias: list[dict[str, Any]]
    f1_por_epoca: dict[int, dict[int, dict[str, float]]]
    umbrales_anidados: dict[str, float]
    proba_outer: np.ndarray
    y_outer: np.ndarray
    proba_outer_train: np.ndarray
    pred_anidada: np.ndarray
    pred_umbral_0_5: np.ndarray
    pred_congelada: np.ndarray
    f1_val_05: dict[str, float]
    f1_train_05: dict[str, float]
    rutas_train_outer: dict[str, Path]
    resumenes_train_outer: dict[str, dict[str, Any]]
    segundos: float
    incidencias: list[str] = field(default_factory=list)

    def a_dict(self) -> dict[str, Any]:
        """Convierte el resultado a un diccionario serializable para el parcial.

        Returns:
            Diccionario sin matrices (solo evidencia resumida).
        """
        return {
            "pliegue": self.pliegue,
            "brazo": self.brazo,
            "etiquetas": list(self.etiquetas),
            "E_estrella": self.e_estrella,
            "epocas_por_split": self.epocas_por_split,
            "intentos_por_split": self.intentos_por_split,
            "equivalencias": self.equivalencias,
            "prevalencias": self.prevalencias,
            "f1_por_epoca": {
                str(j): {
                    str(epoca): {et: round(valor, 8) for et, valor in por_etiqueta.items()}
                    for epoca, por_etiqueta in por_epoca.items()
                }
                for j, por_epoca in self.f1_por_epoca.items()
            },
            "umbrales_anidados": self.umbrales_anidados,
            "f1_val_05": {et: round(valor, 8) for et, valor in self.f1_val_05.items()},
            "f1_train_05": {et: round(valor, 8) for et, valor in self.f1_train_05.items()},
            "rutas_train_outer": {et: str(ruta) for et, ruta in self.rutas_train_outer.items()},
            "resumenes_train_outer": self.resumenes_train_outer,
            "segundos": round(self.segundos, 2),
            "incidencias": self.incidencias,
        }


def procesar_split_interno(
    pliegue: int,
    brazo: str,
    j: int,
    textos_train: np.ndarray,
    y_train: np.ndarray,
    grupos_train: np.ndarray,
    etiquetas: Sequence[str],
) -> dict[str, Any]:
    """Procesa un split interno: partir_interna, epocas {5,10,20} y F1@0.5.

    Args:
        pliegue: pliegue externo.
        brazo: nombre del brazo.
        j: indice del split interno (0, 1, 2).
        textos_train: textos de outer_train.
        y_train: etiquetas de outer_train.
        grupos_train: grupos fusionados de outer_train.
        etiquetas: etiquetas procesadas.

    Returns:
        Diccionario con mascara, epoca elegida, F1 y probabilidades internas.
    """
    semilla = SEMILLA_INTERNA_BASE + 100 * pliegue + 7 * j
    mascara, intentos, equivalencia = partir_interna_equivalente(grupos_train, y_train, semilla)
    prevalencia = evaluar_prevalencia_interna(y_train, mascara)
    textos_in = textos_train[mascara]
    y_in = y_train[mascara]
    textos_va = textos_train[~mascara]
    y_va = y_train[~mascara]

    f1_por_epoca: dict[int, dict[str, float]] = {}
    proba_por_epoca: dict[int, dict[str, np.ndarray]] = {}
    incidencias: list[str] = []
    segundos = 0.0
    for etiqueta in etiquetas:
        columna = INDICE_ETIQUETA[etiqueta]
        ruta_train = CONFIG["tmp_dir"] / f"train_int_f{pliegue}_{brazo}_j{j}_{etiqueta}.txt"
        resumen = escribir_archivo_entrenamiento(ruta_train, textos_in, y_in[:, columna])
        if resumen["n_positivos"] == 0 or resumen["n_negativos"] == 0:
            incidencias.append(
                f"pliegue {pliegue} {brazo} split {j} {etiqueta}: "
                f"clases degeneradas (POS={resumen['n_positivos']}, NEG={resumen['n_negativos']})"
            )
        for epoca in EPOCAS_CANDIDATAS:
            base = CONFIG["tmp_dir"] / f"mod_int_f{pliegue}_{brazo}_j{j}_{etiqueta}_e{epoca}"
            segundos_modelo, incidencia = entrenar_si_posible(resumen, base, epoca, brazo)
            if incidencia:
                incidencias.append(
                    f"pliegue {pliegue} {brazo} split {j} {etiqueta} epoca {epoca}: {incidencia}"
                )
            proba = predecir_o_cero(resumen, base, textos_va)
            eliminar_modelo(base)
            segundos += segundos_modelo
            f1_por_epoca.setdefault(epoca, {})[etiqueta] = round(
                f1_binaria(y_va[:, columna], (proba >= 0.5).astype(np.int64)), 8
            )
            proba_por_epoca.setdefault(epoca, {})[etiqueta] = proba

    epoca_elegida, medias = elegir_epoca_por_split(f1_por_epoca, etiquetas)
    proba_elegida = {
        etiqueta: proba_por_epoca[epoca_elegida][etiqueta] for etiqueta in etiquetas
    }
    log(
        f"   pliegue {pliegue} {brazo} split {j}: semilla {semilla} intentos {intentos} "
        f"| F1 media por epoca "
        f"{ {e: round(v, 4) for e, v in medias.items()} } -> E_j={epoca_elegida}"
    )
    return {
        "j": j,
        "semilla": semilla,
        "intentos": intentos,
        "equivalencia": equivalencia,
        "prevalencia": prevalencia,
        "epoca_elegida": epoca_elegida,
        "medias_por_epoca": medias,
        "f1_por_epoca": f1_por_epoca,
        "y_inner_val": y_va,
        "proba_inner_val": proba_elegida,
        "segundos": segundos,
        "incidencias": incidencias,
    }


def procesar_pliegue_brazo(
    pliegue: int,
    brazo: str,
    etiquetas: Sequence[str],
    datos: DatosGate0,
) -> ResultadoBrazoPliegue:
    """Ejecuta un (pliegue externo, brazo): splits internos, E*, refit y lecturas.

    Args:
        pliegue: pliegue externo 1..5.
        brazo: nombre del brazo.
        etiquetas: etiquetas procesadas.
        datos: datos de Gate 0.

    Returns:
        Resultado completo del par (pliegue, brazo).
    """
    indices_train = np.asarray(datos.pliegues[pliegue - 1][0])
    indices_val = np.asarray(datos.pliegues[pliegue - 1][1])
    textos_train = datos.textos[indices_train]
    y_train = datos.y[indices_train]
    grupos_train = datos.grupos_dev[indices_train]
    textos_val = datos.textos[indices_val]
    y_val = datos.y[indices_val]

    splits = [
        procesar_split_interno(pliegue, brazo, j, textos_train, y_train, grupos_train, etiquetas)
        for j in range(N_INTERNOS)
    ]
    epocas_por_split = [int(split["epoca_elegida"]) for split in splits]
    e_estrella = int(np.median(epocas_por_split))

    # Probabilidades internas agregadas de los 3 splits (modelos con E_j).
    n_interna = sum(len(split["y_inner_val"]) for split in splits)
    proba_inner = np.zeros((n_interna, len(ETIQUETAS)), dtype=np.float64)
    y_inner = np.zeros((n_interna, len(ETIQUETAS)), dtype=np.int64)
    inicio = 0
    for split in splits:
        n_split = len(split["y_inner_val"])
        for etiqueta in etiquetas:
            columna = INDICE_ETIQUETA[etiqueta]
            proba_inner[inicio : inicio + n_split, columna] = split["proba_inner_val"][etiqueta]
            y_inner[inicio : inicio + n_split, columna] = split["y_inner_val"][:, columna]
        inicio += n_split
    umbrales_anidados = elegir_umbrales_anidados(proba_inner, y_inner, etiquetas)
    umbrales_matriz = np.full(len(ETIQUETAS), UMBRAL_DEFECTO, dtype=np.float64)
    for etiqueta, umbral in umbrales_anidados.items():
        umbrales_matriz[INDICE_ETIQUETA[etiqueta]] = umbral

    # Refit sobre outer_train con E* y prediccion unica de outer_val por etiqueta.
    proba_outer = np.zeros((len(indices_val), len(ETIQUETAS)), dtype=np.float64)
    proba_outer_train = np.zeros((len(indices_train), len(ETIQUETAS)), dtype=np.float64)
    f1_val_05: dict[str, float] = {}
    f1_train_05: dict[str, float] = {}
    rutas_train_outer: dict[str, Path] = {}
    resumenes_train_outer: dict[str, dict[str, Any]] = {}
    incidencias: list[str] = []
    for split in splits:
        incidencias.extend(split["incidencias"])
    segundos = float(sum(split["segundos"] for split in splits))
    for etiqueta in etiquetas:
        columna = INDICE_ETIQUETA[etiqueta]
        ruta_train = CONFIG["tmp_dir"] / f"train_ext_f{pliegue}_{brazo}_{etiqueta}.txt"
        resumen = escribir_archivo_entrenamiento(ruta_train, textos_train, y_train[:, columna])
        rutas_train_outer[etiqueta] = ruta_train
        resumenes_train_outer[etiqueta] = resumen
        if resumen["n_positivos"] == 0 or resumen["n_negativos"] == 0:
            incidencias.append(
                f"pliegue {pliegue} {brazo} {etiqueta} (outer_train): clases degeneradas "
                f"(POS={resumen['n_positivos']}, NEG={resumen['n_negativos']})"
            )
        base = CONFIG["tmp_dir"] / f"mod_ext_f{pliegue}_{brazo}_{etiqueta}_e{e_estrella}"
        segundos_modelo, incidencia = entrenar_si_posible(resumen, base, e_estrella, brazo)
        if incidencia:
            incidencias.append(f"pliegue {pliegue} {brazo} {etiqueta} (refit): {incidencia}")
        proba_val_etiqueta = predecir_o_cero(resumen, base, textos_val)
        proba_train_etiqueta = predecir_o_cero(resumen, base, textos_train)
        eliminar_modelo(base)
        segundos += segundos_modelo
        proba_outer[:, columna] = proba_val_etiqueta
        proba_outer_train[:, columna] = proba_train_etiqueta
        f1_val_05[etiqueta] = round(
            f1_binaria(y_val[:, columna], (proba_val_etiqueta >= 0.5).astype(np.int64)), 8
        )
        f1_train_05[etiqueta] = round(
            f1_binaria(y_train[:, columna], (proba_train_etiqueta >= 0.5).astype(np.int64)), 8
        )

    predicciones = {
        "anidada": (proba_outer >= umbrales_matriz[None, :]).astype(np.int64),
        "umbral_0_5": (proba_outer >= UMBRAL_DEFECTO).astype(np.int64),
        "congelada": (proba_outer >= UMBRALES_CONGELADOS[None, :]).astype(np.int64),
    }
    log(
        f"   pliegue {pliegue} {brazo}: E_j {epocas_por_split} -> E*={e_estrella} | "
        f"umbrales {umbrales_anidados} | F1 val@0.5 "
        f"{round(float(np.mean(list(f1_val_05.values()))), 4)} en {segundos:.1f}s"
    )
    return ResultadoBrazoPliegue(
        pliegue=pliegue,
        brazo=brazo,
        etiquetas=tuple(etiquetas),
        e_estrella=e_estrella,
        epocas_por_split=epocas_por_split,
        intentos_por_split=[int(split["intentos"]) for split in splits],
        equivalencias=[split["equivalencia"] for split in splits],
        prevalencias=[split["prevalencia"] for split in splits],
        f1_por_epoca={int(split["j"]): split["f1_por_epoca"] for split in splits},
        umbrales_anidados=umbrales_anidados,
        proba_outer=proba_outer,
        y_outer=y_val,
        proba_outer_train=proba_outer_train,
        pred_anidada=predicciones["anidada"],
        pred_umbral_0_5=predicciones["umbral_0_5"],
        pred_congelada=predicciones["congelada"],
        f1_val_05=f1_val_05,
        f1_train_05=f1_train_05,
        rutas_train_outer=rutas_train_outer,
        resumenes_train_outer=resumenes_train_outer,
        segundos=segundos,
        incidencias=incidencias,
    )


def resumir_brazo(
    resultados: dict[int, ResultadoBrazoPliegue],
    brazo: str,
    datos: DatosGate0,
    etiquetas: Sequence[str],
) -> dict[str, Any]:
    """Agrega los resultados por pliegue de un brazo en la matriz OOF de 1483 filas.

    Args:
        resultados: resultados por pliegue del brazo.
        brazo: nombre del brazo.
        datos: datos de Gate 0.
        etiquetas: etiquetas procesadas.

    Returns:
        Resumen del brazo con matrices OOF y evidencia por pliegue.
    """
    proba = np.zeros((FILAS_DEV, len(ETIQUETAS)), dtype=np.float64)
    proba_train = np.zeros((FILAS_DEV, len(ETIQUETAS)), dtype=np.float64)
    pred_anidada = np.zeros((FILAS_DEV, len(ETIQUETAS)), dtype=np.int64)
    pred_umbral_0_5 = np.zeros((FILAS_DEV, len(ETIQUETAS)), dtype=np.int64)
    pred_congelada = np.zeros((FILAS_DEV, len(ETIQUETAS)), dtype=np.int64)
    e_estrella: dict[str, int] = {}
    epocas_por_split: dict[str, list[int]] = {}
    umbrales_anidados: dict[str, dict[str, float]] = {}
    umbrales_matriz = np.full((len(datos.pliegues), len(ETIQUETAS)), UMBRAL_DEFECTO)
    f1_val_05: dict[str, dict[str, float]] = {}
    f1_train_05: dict[str, dict[str, float]] = {}
    for pliegue in sorted(resultados):
        resultado = resultados[pliegue]
        indices_val = np.asarray(datos.pliegues[pliegue - 1][1])
        indices_train = np.asarray(datos.pliegues[pliegue - 1][0])
        proba[indices_val] = resultado.proba_outer
        proba_train[indices_train] = resultado.proba_outer_train
        pred_anidada[indices_val] = resultado.pred_anidada
        pred_umbral_0_5[indices_val] = resultado.pred_umbral_0_5
        pred_congelada[indices_val] = resultado.pred_congelada
        e_estrella[str(pliegue)] = resultado.e_estrella
        epocas_por_split[str(pliegue)] = resultado.epocas_por_split
        umbrales_anidados[str(pliegue)] = resultado.umbrales_anidados
        for etiqueta, umbral in resultado.umbrales_anidados.items():
            umbrales_matriz[pliegue - 1, INDICE_ETIQUETA[etiqueta]] = umbral
        f1_val_05[str(pliegue)] = resultado.f1_val_05
        f1_train_05[str(pliegue)] = resultado.f1_train_05
    brechas = [
        float(np.mean([resultados[p].f1_train_05[et] - resultados[p].f1_val_05[et] for et in etiquetas]))
        for p in sorted(resultados)
    ]
    return {
        "brazo": brazo,
        "etiquetas": list(etiquetas),
        "proba": proba,
        "proba_train": proba_train,
        "pred_anidada": pred_anidada,
        "pred_umbral_0_5": pred_umbral_0_5,
        "pred_congelada": pred_congelada,
        "E_estrella_por_pliegue": e_estrella,
        "epocas_por_split": epocas_por_split,
        "umbrales_anidados_por_pliegue": umbrales_anidados,
        "umbrales_anidados_matriz": umbrales_matriz,
        "f1_val_05_por_pliegue": f1_val_05,
        "f1_train_05_por_pliegue": f1_train_05,
        "brecha_05_media": round(float(np.mean(brechas)), 8),
        "segundos_entrenamiento": round(
            float(sum(resultados[p].segundos for p in sorted(resultados))), 2
        ),
    }


# ===== SECCION 5: lecturas, comparaciones y gate =====


def metricas_de_lectura(y: np.ndarray, pred: np.ndarray) -> dict[str, Any]:
    """Calcula las metricas publicadas de una lectura de umbral.

    Args:
        y: etiquetas verdaderas (n, 3).
        pred: predicciones binarias (n, 3).

    Returns:
        Diccionario con metricas primarias, de diagnostico y por etiqueta.
    """
    metricas = dict(metricas_multietiqueta(y, pred))
    metricas["precision_micro"] = float(precision_score(y, pred, average="micro", zero_division=0))
    metricas["recall_micro"] = float(recall_score(y, pred, average="micro", zero_division=0))
    metricas["precision_por_etiqueta"] = {
        etiqueta: float(precision_score(y[:, j], pred[:, j], zero_division=0))
        for j, etiqueta in enumerate(ETIQUETAS)
    }
    metricas["recall_por_etiqueta"] = {
        etiqueta: float(recall_score(y[:, j], pred[:, j], zero_division=0))
        for j, etiqueta in enumerate(ETIQUETAS)
    }
    metricas["tasa_positivos"] = [float(valor) for valor in pred.mean(axis=0)]
    return metricas


def comparar_predicciones(
    datos: DatosGate0, pred: np.ndarray, pred_ref: np.ndarray
) -> dict[str, Any]:
    """Compara pareadamente una lectura contra una referencia (V03 congelado).

    Args:
        datos: datos de Gate 0.
        pred: predicciones del candidato.
        pred_ref: predicciones de la referencia.

    Returns:
        Diccionario con delta global, deltas por pliegue, pliegues mejorados,
        rescate/dano, tasas de positivos y F1 por pliegue.
    """
    f1_global = float(metrica_macro(datos.y, pred))
    f1_ref = float(metrica_macro(datos.y, pred_ref))
    deltas: list[float] = []
    f1_pliegues: list[float] = []
    f1_pliegues_ref: list[float] = []
    for _, indice_val in datos.pliegues:
        b = np.asarray(indice_val)
        valor = float(metrica_macro(datos.y[b], pred[b]))
        valor_ref = float(metrica_macro(datos.y[b], pred_ref[b]))
        f1_pliegues.append(round(valor, 8))
        f1_pliegues_ref.append(round(valor_ref, 8))
        deltas.append(round(valor - valor_ref, 8))
    ok_ref = pred_ref == datos.y
    ok_cand = pred == datos.y
    rescate = int(np.sum(~ok_ref & ok_cand))
    dano = int(np.sum(ok_ref & ~ok_cand))
    return {
        "f1_global": round(f1_global, 10),
        "f1_referencia_global": round(f1_ref, 10),
        "delta_global": round(f1_global - f1_ref, 10),
        "delta_medio_pliegues": round(float(np.mean(deltas)), 10),
        "delta_por_pliegue": deltas,
        "pliegues_mejorados": int(np.sum(np.array(deltas) > 0)),
        "f1_por_pliegue": f1_pliegues,
        "f1_por_pliegue_referencia": f1_pliegues_ref,
        "rescate_celdas": rescate,
        "dano_celdas": dano,
        "neto_celdas": rescate - dano,
        "tasa_positivos": [float(valor) for valor in pred.mean(axis=0)],
        "tasa_positivos_referencia": [float(valor) for valor in pred_ref.mean(axis=0)],
    }


def bootstrap_ensemble_gain(
    y: np.ndarray,
    proba_candidato: np.ndarray,
    proba_v03: np.ndarray,
    repeticiones: int,
    semilla: int,
) -> dict[str, Any]:
    """Calcula ensemble_gain con IC bootstrap pareado por filas.

    Convencion P6/P26: F1 de la media de probabilidades a 0.5 menos la F1 de
    V03 a 0.5 recalculada desde el npz; se verifica que la referencia coincide
    con 0.771778476256122 antes de publicar.

    Args:
        y: etiquetas (n, 3).
        proba_candidato: probabilidades del candidato.
        proba_v03: probabilidades de V03.
        repeticiones: numero de remuestreos.
        semilla: semilla del generador.

    Returns:
        Diccionario con la referencia recalculada, el gain y su IC95.

    Raises:
        SystemExit: si la F1 de V03 a 0.5 no reproduce la constante congelada.
    """
    pred_media = ((proba_candidato + proba_v03) / 2.0 >= 0.5).astype(np.int64)
    pred_v03 = (proba_v03 >= 0.5).astype(np.int64)
    f1_media = float(metrica_macro(y, pred_media))
    f1_v03 = float(metrica_macro(y, pred_v03))
    if abs(f1_v03 - V03_F1_AT_05) > TOL_IDENTIDAD:
        raise SystemExit(f"STOP: V03@0.5 no reproduce en ensemble_gain: {f1_v03!r}")
    base = bootstrap_diferencia_pareada(
        y, pred_media, pred_v03, repeticiones=repeticiones, semilla=semilla
    )
    return {
        "f1_media_candidato_v03_05": round(f1_media, 10),
        "f1_v03_05_recalculado": round(f1_v03, 12),
        "ensemble_gain": round(f1_media - f1_v03, 10),
        "ic95": base["ic95"],
        "mediana_bootstrap": base["mediana_bootstrap"],
        "repeticiones": repeticiones,
        "semilla": semilla,
        "celdas_mejoradas": base["celdas_mejoradas"],
        "celdas_igualadas": base["celdas_igualadas"],
        "celdas_empeoradas": base["celdas_empeoradas"],
    }


def correlaciones_con_v03(proba_a: np.ndarray, proba_b: np.ndarray) -> dict[str, Any]:
    """Calcula Pearson y Spearman por etiqueta y global entre dos matrices.

    Args:
        proba_a: probabilidades del candidato (n, 3).
        proba_b: probabilidades de referencia (n, 3).

    Returns:
        Diccionario con correlaciones por etiqueta y globales.
    """

    def _correlaciones(vector_a: np.ndarray, vector_b: np.ndarray) -> dict[str, float | None]:
        """Calcula el par de correlaciones de dos vectores.

        Args:
            vector_a: primer vector.
            vector_b: segundo vector.

        Returns:
            Diccionario con pearson y spearman (None si no es finito).
        """
        with np.errstate(invalid="ignore", divide="ignore"):
            pearson = float(stats.pearsonr(vector_a, vector_b)[0])
            spearman = float(stats.spearmanr(vector_a, vector_b)[0])
        return {
            "pearson": None if not np.isfinite(pearson) else round(pearson, 6),
            "spearman": None if not np.isfinite(spearman) else round(spearman, 6),
        }

    por_etiqueta = {
        etiqueta: _correlaciones(proba_a[:, j], proba_b[:, j])
        for j, etiqueta in enumerate(ETIQUETAS)
    }
    globales = _correlaciones(proba_a.ravel(), proba_b.ravel())
    return {"por_etiqueta": por_etiqueta, "global": globales}


def matriz_errores(y: np.ndarray, pred: np.ndarray, pred_ref: np.ndarray) -> dict[str, Any]:
    """Construye la matriz de solapamiento de errores frente a una referencia.

    Args:
        y: etiquetas verdaderas (n, 3).
        pred: predicciones del candidato.
        pred_ref: predicciones de la referencia (V03 congelado).

    Returns:
        Recuentos por etiqueta y globales mas el Jaccard de conjuntos de error.
    """
    por_etiqueta: dict[str, Any] = {}
    for j, etiqueta in enumerate(ETIQUETAS):
        ok_cand = pred[:, j] == y[:, j]
        ok_ref = pred_ref[:, j] == y[:, j]
        error_cand = ~ok_cand
        error_ref = ~ok_ref
        interseccion = int(np.sum(error_cand & error_ref))
        union = int(np.sum(error_cand | error_ref))
        por_etiqueta[etiqueta] = {
            "ambos_correctos": int(np.sum(ok_cand & ok_ref)),
            "solo_candidato_error": int(np.sum(error_cand & ok_ref)),
            "solo_v03_error": int(np.sum(ok_cand & error_ref)),
            "ambos_error": interseccion,
            "jaccard_error": round(interseccion / union, 6) if union > 0 else None,
        }
    ok_cand_global = (pred == y).ravel()
    ok_ref_global = (pred_ref == y).ravel()
    error_cand_global = ~ok_cand_global
    error_ref_global = ~ok_ref_global
    interseccion = int(np.sum(error_cand_global & error_ref_global))
    union = int(np.sum(error_cand_global | error_ref_global))
    globales = {
        "celdas_totales": int(y.size),
        "ambos_correctos": int(np.sum(ok_cand_global & ok_ref_global)),
        "solo_candidato_error": int(np.sum(error_cand_global & ok_ref_global)),
        "solo_v03_error": int(np.sum(ok_cand_global & error_ref_global)),
        "ambos_error": interseccion,
        "jaccard_error": round(interseccion / union, 6) if union > 0 else None,
    }
    return {"por_etiqueta": por_etiqueta, "global": globales}


def evaluar_gate(
    lecturas: dict[str, Any],
    ensemble_gain: dict[str, Any],
    determinismo: dict[str, Any] | None,
) -> dict[str, Any]:
    """Evalua Gate 1 standalone y el gate de complementariedad de EXP-02.

    El Gate 1 standalone usa el delta global pareado del OOF (coherente con el
    IC95 pareado por filas y con p84_01); delta_medio_pliegues se publica como
    diagnostico, no decide el gate.

    Args:
        lecturas: lecturas por brazo y umbral con sus comparaciones y bootstrap.
        ensemble_gain: ensemble_gain del brazo principal frente a V03.
        determinismo: evidencia de determinismo (None si no se verifico).

    Returns:
        Diccionario con los dos gates, la decision y los motivos.
    """
    comparacion = lecturas["principal"]["anidada"]["comparacion"]
    bootstrap = lecturas["principal"]["anidada"]["bootstrap"]
    delta_global = float(comparacion["delta_global"])
    delta_pliegues = float(comparacion["delta_medio_pliegues"])
    pliegues = int(comparacion["pliegues_mejorados"])
    ic95 = bootstrap["ic95"]
    standalone = {
        "delta_global": delta_global,
        "delta_medio_pliegues": delta_pliegues,
        "umbral_delta": GATE_DELTA,
        "pliegues_mejorados": pliegues,
        "pliegues_minimos": GATE_PLIEGUES,
        "ic95": ic95,
        "pasa": bool(
            delta_global >= GATE_DELTA and pliegues >= GATE_PLIEGUES and ic95[0] > 0
        ),
    }
    gain = float(ensemble_gain["ensemble_gain"])
    ic_gain = ensemble_gain["ic95"]
    complementariedad = {
        "ensemble_gain": gain,
        "umbral_gain": GATE_ENSEMBLE_GAIN,
        "ic95": ic_gain,
        "pasa": bool(gain >= GATE_ENSEMBLE_GAIN and ic_gain[0] > 0),
    }
    determinismo_ok = bool(determinismo is not None and determinismo["identicas"])
    elegible = bool((standalone["pasa"] or complementariedad["pasa"]) and determinismo_ok)
    motivos: list[str] = []
    if not standalone["pasa"]:
        motivos.append(
            f"Gate 1 standalone NO: delta global {delta_global:+.6f} (min {GATE_DELTA}), "
            f"delta medio/pliegue {delta_pliegues:+.6f} (diagnostico), "
            f"pliegues {pliegues}/5 (min {GATE_PLIEGUES}), IC95 "
            f"[{ic95[0]:+.6f}, {ic95[1]:+.6f}]"
        )
    if not complementariedad["pasa"]:
        motivos.append(
            f"complementariedad NO: ensemble_gain {gain:+.6f} (min {GATE_ENSEMBLE_GAIN}), "
            f"IC95 [{ic_gain[0]:+.6f}, {ic_gain[1]:+.6f}]"
        )
    if not determinismo_ok:
        motivos.append("determinismo NO verificado con tolerancia 0: no se promueve")
    decision = (
        "ELEGIBLE PARA EXP-06 (pendiente de Gate 2; sin submission)"
        if elegible
        else "CERRAR fastText supervisado: sin rejilla nueva (resultado verificado)"
    )
    return {
        "standalone": standalone,
        "complementariedad": complementariedad,
        "determinismo_ok": determinismo_ok,
        "elegible_exp06": elegible,
        "decision": decision,
        "motivos": motivos,
    }


# ===== SECCION 6: salidas (parciales, npz, informes y CSV) =====


def escribir_parcial(resultado: ResultadoBrazoPliegue, humo: bool) -> Path:
    """Escribe el parcial JSON de un par (pliegue, brazo).

    Args:
        resultado: resultado del par.
        humo: si True el parcial lleva el sufijo _humo.

    Returns:
        Ruta escrita.
    """
    directorio: Path = CONFIG["parciales_dir"]
    directorio.mkdir(parents=True, exist_ok=True)
    sufijo = "_humo" if humo else ""
    ruta = directorio / f"p84_02_p{resultado.pliegue}_{resultado.brazo}{sufijo}.json"
    carga = {"experiment_id": "P84-02", "modo_humo": humo}
    carga.update(resultado.a_dict())
    escribir_json(ruta, carga)
    return ruta


def guardar_oof_npz(
    resumenes: dict[str, dict[str, Any]], datos: DatosGate0
) -> Path:
    """Guarda el npz OOF con las probabilidades principal y control.

    Args:
        resumenes: resumen por brazo.
        datos: datos de Gate 0.

    Returns:
        Ruta escrita.
    """
    ruta: Path = CONFIG["oof_npz"]
    ruta.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        ruta,
        proba_principal=resumenes["principal"]["proba"],
        proba_control=resumenes["control"]["proba"],
        pliegues=datos.folde.astype(np.int64),
        y_true=datos.y.astype(np.int64),
        ids=datos.ids.astype(np.int64),
        umbrales_anidados_principal=resumenes["principal"]["umbrales_anidados_matriz"],
        umbrales_anidados_control=resumenes["control"]["umbrales_anidados_matriz"],
        E_estrella_principal=np.array(
            [resumenes["principal"]["E_estrella_por_pliegue"][str(p)] for p in range(1, 6)],
            dtype=np.int64,
        ),
        pred_principal_anidada=resumenes["principal"]["pred_anidada"].astype(np.int64),
        pred_control_anidada=resumenes["control"]["pred_anidada"].astype(np.int64),
        etiquetas=np.array(ETIQUETAS),
    )
    return ruta


def upsert_experiments_csv(fila: dict[str, str]) -> dict[str, Any]:
    """Inserta o reemplaza la fila P84-02 en experiments.csv (23 columnas).

    Valida la cabecera, el ancho de todas las filas, la ausencia de ids
    duplicados y que la fila defina exactamente las 23 columnas antes de
    escribir. Escribe utf-8-sig y no duplica ids.

    Args:
        fila: valores por columna de la fila P84-02.

    Returns:
        Evidencia del upsert.

    Raises:
        SystemExit: si la estructura del CSV no es la esperada.
    """
    ruta: Path = CONFIG["experiments_csv"]
    if not ruta.exists():
        raise SystemExit(f"STOP: no existe {ruta}")
    with ruta.open(encoding="utf-8-sig", newline="") as manejador:
        filas = list(csv.reader(manejador))
    if not filas:
        raise SystemExit("STOP: experiments.csv esta vacio")
    cabecera = [columna.strip() for columna in filas[0]]
    if tuple(cabecera) != COLUMNAS_EXPERIMENTS:
        raise SystemExit(f"STOP: cabecera inesperada en experiments.csv: {cabecera}")
    cuerpo = filas[1:]
    ids = [registro[0] for registro in cuerpo if registro]
    if len(ids) != len(set(ids)):
        raise SystemExit("STOP: experiments.csv tiene ids duplicados")
    for numero, registro in enumerate(cuerpo, start=2):
        if len(registro) != len(COLUMNAS_EXPERIMENTS):
            raise SystemExit(
                f"STOP: la fila {numero} de experiments.csv tiene {len(registro)} columnas"
            )
    faltantes = [columna for columna in COLUMNAS_EXPERIMENTS if columna not in fila]
    if faltantes:
        raise SystemExit(f"STOP: faltan columnas en la fila P84-02: {faltantes}")
    sobrantes = sorted(set(fila) - set(COLUMNAS_EXPERIMENTS))
    if sobrantes:
        raise SystemExit(f"STOP: columnas desconocidas en la fila P84-02: {sobrantes}")
    nueva = [fila.get(columna, "") for columna in COLUMNAS_EXPERIMENTS]
    identificador = nueva[0]
    if identificador in ids:
        cuerpo[ids.index(identificador)] = nueva
        accion = "reemplazada"
    else:
        cuerpo.append(nueva)
        accion = "anadida"
    temporal = ruta.with_name(ruta.name + ".tmp")
    with temporal.open("w", encoding="utf-8-sig", newline="") as manejador:
        escritor = csv.writer(manejador, lineterminator="\n")
        escritor.writerow(COLUMNAS_EXPERIMENTS)
        escritor.writerows(cuerpo)
    os.replace(temporal, ruta)
    return {"accion": accion, "filas": len(cuerpo), "columnas": len(COLUMNAS_EXPERIMENTS)}


def construir_fila_experiments(
    lecturas: dict[str, Any],
    principal_vs_control: dict[str, Any],
    ensemble_gain: dict[str, Any],
    gate: dict[str, Any],
    determinismo: dict[str, Any] | None,
    segundos_totales: float,
) -> dict[str, str]:
    """Construye la fila P84-02 para experiments.csv.

    Args:
        lecturas: lecturas por brazo y umbral.
        principal_vs_control: comparacion del brazo principal frente al control.
        ensemble_gain: ensemble_gain del brazo principal.
        gate: resultado del gate.
        determinismo: evidencia de determinismo.
        segundos_totales: tiempo total de la corrida.

    Returns:
        Diccionario columna -> valor.
    """
    lectura = lecturas["principal"]["anidada"]
    metricas = lectura["metricas"]
    comparacion = lectura["comparacion"]
    resultado = {
        "lectura_primaria_anidada": comparacion["f1_global"],
        "lectura_umbral_0_5": lecturas["principal"]["umbral_0_5"]["comparacion"]["f1_global"],
        "lectura_congelada_v03": lecturas["principal"]["congelada"]["comparacion"]["f1_global"],
        "delta_global_anidada": comparacion["delta_global"],
        "delta_medio_pliegues": comparacion["delta_medio_pliegues"],
        "delta_por_pliegue": comparacion["delta_por_pliegue"],
        "pliegues_mejorados": comparacion["pliegues_mejorados"],
        "ic95_bootstrap": lectura["bootstrap"]["ic95"],
        "semilla_bootstrap": SEMILLA_BOOTSTRAP,
        "ensemble_gain": ensemble_gain,
        "rescate_celdas": comparacion["rescate_celdas"],
        "dano_celdas": comparacion["dano_celdas"],
        "neto_celdas": comparacion["neto_celdas"],
        "tasa_positivos": metricas["tasa_positivos"],
        "f1_por_etiqueta": metricas["f1_por_etiqueta"],
        "precision_micro": metricas["precision_micro"],
        "recall_micro": metricas["recall_micro"],
        "principal_vs_control": principal_vs_control,
        "determinismo": determinismo,
        "gate": gate,
    }
    return {
        "experiment_id": "P84-02",
        "competition": str(CONFIG["competencia"]),
        "question": (
            "Un clasificador fastText supervisado desde cero (subwords 3-6, fold-local, "
            "epoca elegida en particion interna) aporta errores complementarios a V03 "
            "(corrected_cv 0.7772114525165721) sobre las 1483 filas de desarrollo, o supera "
            "a V03 con delta >= +0.005 y al menos 4/5 pliegues?"
        ),
        "hypothesis": (
            "Expectativa pre-registrada: alta correlacion con los char n-gramas de V03 y "
            "ganancia standalone improbable (familia lineal/hashing cerrada); la via realista "
            "es la complementariedad (ensemble_gain >= 0.0015 con IC95 > 0)."
        ),
        "preprocessing": (
            "texto lower() sin otra normalizacion (el texto de outer_val nunca entra al "
            "archivo de entrenamiento); una linea por fila con __label__POS/__label__NEG; "
            "vocabulario y subwords internos de fastText; sin vectores preentrenados, sin "
            "corpus externo, sin holdout ni test"
        ),
        "feature_groups": "texto crudo (bolsa de n-gramas de palabra y subwords hasheados)",
        "representation": (
            "fastText supervised dim 100, wordNgrams 2, bucket 2000000; principal subwords "
            "3-6; control -minn 0 -maxn 0"
        ),
        "model": "fastText supervised (binario oficial congelado 59c51cc5...)",
        "hyperparameters": json.dumps(
            {
                **{clave: valor for clave, valor in HIPERPARAMETROS_FIJOS.items()},
                "brazos": {brazo: {"minn": minn, "maxn": maxn} for brazo, (minn, maxn) in BRAZOS.items()},
                "epocas_candidatas": list(EPOCAS_CANDIDATAS),
                "E_estrella": "mediana de las 3 epocas elegidas en splits internos",
                "particiones_internas": "partir_interna 80/20 por grupos, semillas 20261101+100k+7j",
            },
            ensure_ascii=True,
        ),
        "validation": (
            "pliegues_corregidos de 5 pliegues sobre desarrollo (1483, grupos fusionados); "
            "E* en splits internos (partir_interna, 3 splits, tolerancia de prevalencia 0.07); "
            "umbral anidado 0.20-0.80 paso 0.01 sobre inner_val; lecturas anidada (primaria), "
            "0.5 y congelada 0.44/0.51/0.46; una sola prediccion de outer_val por etiqueta"
        ),
        "f1": f"{comparacion['f1_global']:.6f}",
        "macro_f1": f"{metricas['f1_macro6']:.6f}",
        "weighted_f1": "",
        "precision": f"{metricas['precision_micro']:.6f}",
        "recall": f"{metricas['recall_micro']:.6f}",
        "training_time": f"{segundos_totales:.2f}",
        "dimension": (
            "dim 100; bucket 2000000; subwords 3-6 (principal) / 0 (control); "
            "vocabulario interno por pliegue y etiqueta"
        ),
        "result": json.dumps(resultado, ensure_ascii=True, default=_json_default),
        "interpretation": (
            f"F1 anidada {comparacion['f1_global']:.6f} vs V03 congelado "
            f"{comparacion['f1_referencia_global']:.6f}; delta global (gate) "
            f"{comparacion['delta_global']:+.6f}; delta medio por pliegue (diagnostico) "
            f"{comparacion['delta_medio_pliegues']:+.6f}; pliegues mejorados "
            f"{comparacion['pliegues_mejorados']}/5; ensemble_gain "
            f"{ensemble_gain['ensemble_gain']:+.6f} con IC95 "
            f"[{ensemble_gain['ic95'][0]:+.6f}, {ensemble_gain['ic95'][1]:+.6f}]; "
            f"aporte sublexico (principal vs control) delta "
            f"{principal_vs_control['delta_medio_pliegues']:+.6f}"
        ),
        "decision": str(gate["decision"]),
        "date": str(CONFIG["fecha"]),
        "script": str(CONFIG["script"]),
        "notes": (
            "EXP-02 del preregistro P84. Determinismo verificado con tolerancia 0; contadores "
            "dev 1483 / holdout 0 / test 0; bootstrap 10000 semilla 20260926. RUNLOG.md y "
            "DECISIONS.md los actualiza la corrida del trainer, no el script."
        ),
    }


def construir_informe_md(carga: dict[str, Any]) -> str:
    """Construye el informe Markdown ASCII de la corrida.

    Args:
        carga: informe JSON completo.

    Returns:
        Texto Markdown.
    """
    lineas: list[str] = []
    lineas.append("# P84-02: fastText supervisado desde cero (EXP-02)")
    lineas.append("")
    lineas.append(f"- Fecha: {carga['fecha']}")
    lineas.append(f"- Script: {carga['script']}")
    lineas.append(
        "- Especificacion: docs/p84_spec.md seccion 3; preregistro docs/p84_preregistro.md"
    )
    lineas.append(
        f"- Referencia V03: corrected_cv {V03_CCV}; V03@0.5 {V03_F1_AT_05}"
    )
    lineas.append("- V03 remains the best verified leaderboard result.")
    lineas.append(
        f"- Modo: {'HUMO (no interpretable)' if carga['modo_humo'] else ('corrida completa' if carga['corrida_completa'] else 'corrida parcial (no interpretable)')}"
    )
    lineas.append("")
    lineas.append("## Gate 0")
    lineas.append("")
    evidencia = carga["gate0"]["evidencia"]
    for hash_ in evidencia["hashes"]:
        lineas.append(f"- sha256 {hash_['nombre']}: {hash_['sha256'][:16]}... OK")
    contadores = evidencia["contadores"]
    lineas.append(
        f"- Contadores: dev {contadores['dev_filas_leidas']} / holdout "
        f"{contadores['holdout_filas_usadas']} / test {contadores['test_filas_usadas']}"
    )
    particiones = evidencia["particiones"]
    lineas.append(
        f"- Particiones: dev {particiones['dev']} + holdout {particiones['holdout']}; "
        f"pliegues {particiones['pliegues_tamanos']} sin grupos compartidos"
    )
    identidad = carga["gate0"]["identidad_v03"]
    lineas.append(
        f"- Identidad V03: corrected_cv reconstruido {identidad['corrected_cv_reconstruido']} "
        f"(delta {identidad['delta_corrected_cv']:.3e}); V03@0.5 "
        f"{identidad['f1_at_05_recalculado']} (delta {identidad['delta_f1_at_05']:.3e}); "
        f"tolerancia {identidad['tolerancia']:g}"
    )
    lineas.append("")
    lineas.append("## Determinismo (tolerancia 0)")
    lineas.append("")
    determinismo = carga.get("determinismo")
    if determinismo is None:
        lineas.append("- No verificado en esta corrida.")
    else:
        lineas.append(
            f"- ({determinismo['brazo']}, {determinismo['etiqueta']}, pliegue "
            f"{determinismo['pliegue']}) con E*={determinismo['epocas']}: "
            f"max|delta proba| = {determinismo['max_delta_probabilidades']}; "
            f"identicas = {determinismo['identicas']}"
        )
    lineas.append("")
    if carga["corrida_completa"]:
        lineas.append("## Epocas y umbrales anidados")
        lineas.append("")
        lineas.append("| pliegue | brazo | E_j (j=0,1,2) | E* | umbrales anidados |")
        lineas.append("|---|---|---|---|---|")
        for brazo in ("principal", "control"):
            resumen = carga["brazos"][brazo]["resumen"]
            for pliegue in range(1, 6):
                clave = str(pliegue)
                if clave not in resumen["E_estrella_por_pliegue"]:
                    continue
                lineas.append(
                    f"| {pliegue} | {brazo} | {resumen['epocas_por_split'][clave]} | "
                    f"{resumen['E_estrella_por_pliegue'][clave]} | "
                    f"{resumen['umbrales_anidados_por_pliegue'][clave]} |"
                )
        lineas.append("")
        lineas.append("## Lecturas y comparacion con V03 (OOF congelado)")
        lineas.append("")
        lineas.append(
            "| brazo | lectura | F1 | delta global | delta medio/pliegue | pliegues | "
            "IC95 | rescate | dano |"
        )
        lineas.append("|---|---|---|---|---|---|---|---|---|")
        for brazo in ("principal", "control"):
            for clave in ("anidada", "umbral_0_5", "congelada"):
                lectura = carga["lecturas"][brazo][clave]
                comparacion = lectura["comparacion"]
                ic = lectura["bootstrap"]["ic95"]
                lineas.append(
                    f"| {brazo} | {clave} | {comparacion['f1_global']:.6f} | "
                    f"{comparacion['delta_global']:+.6f} | "
                    f"{comparacion['delta_medio_pliegues']:+.6f} | "
                    f"{comparacion['pliegues_mejorados']}/5 | "
                    f"[{ic[0]:+.6f}, {ic[1]:+.6f}] | "
                    f"{comparacion['rescate_celdas']} | {comparacion['dano_celdas']} |"
                )
        lineas.append("")
        lineas.append("## Principal vs control (aporte sublexico)")
        lineas.append("")
        pvc = carga["principal_vs_control"]
        lineas.append(
            f"- Delta medio por pliegue (anidada): {pvc['delta_medio_pliegues']:+.6f}; "
            f"pliegues mejorados {pvc['pliegues_mejorados']}/5; IC95 "
            f"[{pvc['bootstrap']['ic95'][0]:+.6f}, {pvc['bootstrap']['ic95'][1]:+.6f}]"
        )
        lineas.append("")
        lineas.append("## ensemble_gain (media de probabilidades a 0.5)")
        lineas.append("")
        for brazo in ("principal", "control"):
            gain = carga["ensemble_gain"][brazo]
            lineas.append(
                f"- {brazo}: gain {gain['ensemble_gain']:+.6f} con IC95 "
                f"[{gain['ic95'][0]:+.6f}, {gain['ic95'][1]:+.6f}]; F1 media "
                f"{gain['f1_media_candidato_v03_05']:.6f}; V03@0.5 "
                f"{gain['f1_v03_05_recalculado']}"
            )
        lineas.append("")
        lineas.append("## Correlaciones con V03")
        lineas.append("")
        for brazo in ("principal", "control"):
            corr = carga["correlaciones"][brazo]
            por_etiqueta = "; ".join(
                f"{et}: r={valor['pearson']}, rho={valor['spearman']}"
                for et, valor in corr["por_etiqueta"].items()
            )
            lineas.append(
                f"- {brazo} global (r, rho) = "
                f"({corr['global']['pearson']}, {corr['global']['spearman']}); {por_etiqueta}"
            )
        lineas.append("")
        lineas.append("## Matriz de solapamiento de errores (vs V03 congelado, lectura anidada)")
        lineas.append("")
        errores = carga["matriz_errores"]["principal"]
        lineas.append(
            "| etiqueta | ambos correctos | solo candidato error | solo V03 error | ambos error | "
            "jaccard |"
        )
        lineas.append("|---|---|---|---|---|---|")
        for etiqueta, valores in errores["por_etiqueta"].items():
            lineas.append(
                f"| {etiqueta} | {valores['ambos_correctos']} | "
                f"{valores['solo_candidato_error']} | {valores['solo_v03_error']} | "
                f"{valores['ambos_error']} | {valores['jaccard_error']} |"
            )
        globales = errores["global"]
        lineas.append(
            f"- Global: ambos correctos {globales['ambos_correctos']}, solo candidato error "
            f"{globales['solo_candidato_error']}, solo V03 error {globales['solo_v03_error']}, "
            f"ambos error {globales['ambos_error']}, jaccard {globales['jaccard_error']}"
        )
        lineas.append("")
        lineas.append("## Gate y decision")
        lineas.append("")
        gate = carga["gate"]
        for clave in ("standalone", "complementariedad"):
            bloque = gate[clave]
            lineas.append(f"- {clave}: {bloque}")
        lineas.append(f"- Determinismo OK: {gate['determinismo_ok']}")
        lineas.append(f"- Elegible para EXP-06: {gate['elegible_exp06']}")
        lineas.append(f"- Decision: {gate['decision']}")
        for motivo in gate["motivos"]:
            lineas.append(f"- Motivo: {motivo}")
        lineas.append("")
    incidencias = carga.get("incidencias") or []
    lineas.append("## Incidencias")
    lineas.append("")
    if incidencias:
        for incidencia in incidencias:
            lineas.append(f"- {incidencia}")
    else:
        lineas.append("- Ninguna.")
    lineas.append("")
    lineas.append("## Trazabilidad")
    lineas.append("")
    lineas.append(
        f"- Tiempo total {carga['tiempo_total_segundos']} s; versiones "
        f"{json.dumps(carga['versiones'], ensure_ascii=True)}"
    )
    lineas.append(f"- Hiperparametros fijos: {json.dumps(HIPERPARAMETROS_FIJOS, ensure_ascii=True)}")
    lineas.append("- Bootstrap pareado por filas, 10000 remuestreos, semilla 20260926.")
    lineas.append("")
    return "\n".join(lineas)


# ===== SECCION 7: CLI y orquestacion =====


def parsear_lista_enteros(texto: str, nombre: str, validos: Sequence[int]) -> tuple[int, ...]:
    """Parsea una lista de enteros separados por comas y valida el dominio.

    Args:
        texto: cadena de entrada.
        nombre: nombre del argumento para mensajes.
        validos: valores admitidos.

    Returns:
        Tupla de enteros sin duplicados.

    Raises:
        SystemExit: si el formato es invalido, hay duplicados o valores fuera del dominio.
    """
    try:
        valores = tuple(int(parte) for parte in texto.split(",") if parte.strip())
    except ValueError as error:
        raise SystemExit(f"STOP: --{nombre} invalido: {texto!r}") from error
    if not valores:
        raise SystemExit(f"STOP: --{nombre} no puede estar vacio")
    if len(set(valores)) != len(valores):
        raise SystemExit(f"STOP: --{nombre} tiene valores duplicados: {valores}")
    invalidos = [valor for valor in valores if valor not in validos]
    if invalidos:
        raise SystemExit(f"STOP: --{nombre} fuera de {list(validos)}: {invalidos}")
    return valores


def parsear_lista_nombres(texto: str, nombre: str, validos: Sequence[str]) -> tuple[str, ...]:
    """Parsea una lista de nombres separados por comas y valida el dominio.

    Args:
        texto: cadena de entrada.
        nombre: nombre del argumento para mensajes.
        validos: valores admitidos.

    Returns:
        Tupla de nombres sin duplicados.

    Raises:
        SystemExit: si el formato es invalido, hay duplicados o valores fuera del dominio.
    """
    valores = tuple(parte.strip() for parte in texto.split(",") if parte.strip())
    if not valores:
        raise SystemExit(f"STOP: --{nombre} no puede estar vacio")
    if len(set(valores)) != len(valores):
        raise SystemExit(f"STOP: --{nombre} tiene valores duplicados: {valores}")
    invalidos = [valor for valor in valores if valor not in validos]
    if invalidos:
        raise SystemExit(f"STOP: --{nombre} fuera de {list(validos)}: {invalidos}")
    return valores


def parsear_argumentos(argv: Sequence[str] | None) -> argparse.Namespace:
    """Define y parsea los argumentos de la linea de comandos.

    Args:
        argv: argumentos (None usa sys.argv[1:]).

    Returns:
        Namespace con humo, pliegues, brazos y etiquetas.
    """
    analizador = argparse.ArgumentParser(
        description="P84-02: EXP-02 (fastText supervisado fold-local) frente a V03."
    )
    analizador.add_argument(
        "--humo", action="store_true", help="modo humo: pliegue 1, brazo principal, etiqueta HS"
    )
    analizador.add_argument(
        "--pliegues", default="1,2,3,4,5", help="pliegues separados por comas (defecto 1,2,3,4,5)"
    )
    analizador.add_argument(
        "--brazos", default="principal,control", help="brazos: principal,control"
    )
    analizador.add_argument(
        "--etiquetas", default="HS,TR,AG", help="etiquetas separadas por comas (defecto HS,TR,AG)"
    )
    return analizador.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Ejecuta P84-02 completo.

    Args:
        argv: argumentos de linea de comandos.

    Returns:
        Codigo de salida 0 si todo termino bien.
    """
    inicio = time.perf_counter()
    argumentos = parsear_argumentos(argv)
    humo = bool(argumentos.humo)
    pliegues = parsear_lista_enteros(argumentos.pliegues, "pliegues", CONFIG["pliegues_por_defecto"])
    brazos = parsear_lista_nombres(argumentos.brazos, "brazos", tuple(BRAZOS))
    etiquetas = parsear_lista_nombres(argumentos.etiquetas, "etiquetas", ETIQUETAS)
    if humo:
        pliegues, brazos, etiquetas = (1,), ("principal",), ("HS",)
        log("MODO HUMO: pliegue 1, brazo principal, etiqueta HS; sin CSV, npz ni informes finales")
    corrida_completa = bool(
        set(pliegues) == set(CONFIG["pliegues_por_defecto"])
        and set(brazos) == set(BRAZOS)
        and set(etiquetas) == set(ETIQUETAS)
    )
    log(
        f"pliegues {pliegues} | brazos {brazos} | etiquetas {etiquetas} | "
        f"humo {humo} | completa {corrida_completa}"
    )

    # ---------- Gate 0 ----------
    log("Gate 0: hashes congelados, particiones, contadores e identidad V03")
    datos = cargar_gate0()
    identidad = control_identidad_v03(datos)
    log(
        f"   dev {len(datos.des)} | holdout {datos.evidencia['particiones']['holdout']} usado 0 | "
        f"test usado 0 | pliegues {datos.evidencia['particiones']['pliegues_tamanos']}"
    )
    log(
        f"   identidad V03: corrected_cv {identidad['corrected_cv_reconstruido']:.10f} | "
        f"V03@0.5 {identidad['f1_at_05_recalculado']:.12f} | tolerancia "
        f"{identidad['tolerancia']:g} OK"
    )

    CONFIG["tmp_dir"].mkdir(parents=True, exist_ok=True)

    # ---------- bucle principal ----------
    resultados: dict[str, dict[int, ResultadoBrazoPliegue]] = {brazo: {} for brazo in brazos}
    rutas_parciales: list[str] = []
    determinismo: dict[str, Any] | None = None
    for pliegue in sorted(pliegues):
        log(f"pliegue {pliegue}: inicio")
        for brazo in brazos:
            resultado = procesar_pliegue_brazo(pliegue, brazo, etiquetas, datos)
            resultados[brazo][pliegue] = resultado
            rutas_parciales.append(str(escribir_parcial(resultado, humo)))
            if pliegue == 1 and brazo == "principal":
                etiqueta_det = "HS" if "HS" in etiquetas else etiquetas[0]
                log(f"determinismo: doble corrida ({brazo}, {etiqueta_det}, pliegue 1)")
                determinismo = verificar_determinismo(
                    pliegue=1,
                    brazo=brazo,
                    etiqueta=etiqueta_det,
                    ruta_train=resultado.rutas_train_outer[etiqueta_det],
                    textos_val=datos.textos[np.asarray(datos.pliegues[0][1])],
                    epocas=resultado.e_estrella,
                )
                log(
                    f"   max|delta proba| {determinismo['max_delta_probabilidades']} | "
                    f"identicas {determinismo['identicas']} (tolerancia "
                    f"{determinismo['tolerancia']})"
                )
            eliminados_txt = limpiar_temporales_txt(CONFIG["tmp_dir"])
            if eliminados_txt:
                log(f"   temporales .txt eliminados: {len(eliminados_txt)}")

    # ---------- lecturas, comparaciones y gate ----------
    resumenes: dict[str, dict[str, Any]] = {}
    lecturas: dict[str, dict[str, Any]] = {}
    principal_vs_control: dict[str, Any] = {}
    gain_por_brazo: dict[str, Any] = {}
    correlaciones: dict[str, Any] = {}
    errores: dict[str, Any] = {}
    gate: dict[str, Any] = {
        "standalone": {},
        "complementariedad": {},
        "determinismo_ok": bool(determinismo is not None and determinismo["identicas"]),
        "elegible_exp06": False,
        "decision": "NO EVALUABLE",
        "motivos": ["corrida parcial o humo: sin OOF completo"],
    }
    incidencias: list[str] = []
    for brazo in brazos:
        for resultado in resultados[brazo].values():
            incidencias.extend(resultado.incidencias)
        resumenes[brazo] = resumir_brazo(resultados[brazo], brazo, datos, etiquetas)
    if corrida_completa:
        for brazo in brazos:
            resumen = resumenes[brazo]
            lecturas[brazo] = {}
            for clave, pred in (
                ("anidada", resumen["pred_anidada"]),
                ("umbral_0_5", resumen["pred_umbral_0_5"]),
                ("congelada", resumen["pred_congelada"]),
            ):
                lecturas[brazo][clave] = {
                    "metricas": metricas_de_lectura(datos.y, pred),
                    "comparacion": comparar_predicciones(datos, pred, datos.pred_v03),
                    "bootstrap": bootstrap_diferencia_pareada(
                        datos.y,
                        pred,
                        datos.pred_v03,
                        repeticiones=REPETICIONES_BOOTSTRAP,
                        semilla=SEMILLA_BOOTSTRAP,
                    ),
                }
        pred_principal = resumenes["principal"]["pred_anidada"]
        pred_control = resumenes["control"]["pred_anidada"]
        comparacion_pvc = comparar_predicciones(datos, pred_principal, pred_control)
        comparacion_pvc["bootstrap"] = bootstrap_diferencia_pareada(
            datos.y,
            pred_principal,
            pred_control,
            repeticiones=REPETICIONES_BOOTSTRAP,
            semilla=SEMILLA_BOOTSTRAP,
        )
        principal_vs_control = comparacion_pvc
        for brazo in brazos:
            gain_por_brazo[brazo] = bootstrap_ensemble_gain(
                datos.y,
                resumenes[brazo]["proba"],
                datos.p_v03,
                repeticiones=REPETICIONES_BOOTSTRAP,
                semilla=SEMILLA_BOOTSTRAP,
            )
            correlaciones[brazo] = correlaciones_con_v03(resumenes[brazo]["proba"], datos.p_v03)
            errores[brazo] = matriz_errores(datos.y, resumenes[brazo]["pred_anidada"], datos.pred_v03)
        gate = evaluar_gate(lecturas, gain_por_brazo["principal"], determinismo)
        log(f"GATE: {gate['decision']}")
        for motivo in gate["motivos"]:
            log(f"   motivo: {motivo}")

    # ---------- salidas finales ----------
    tiempo_total = round(time.perf_counter() - inicio, 2)
    carga: dict[str, Any] = {
        "experiment_id": "P84-02",
        "script": str(CONFIG["script"]),
        "fecha": str(CONFIG["fecha"]),
        "modo_humo": humo,
        "corrida_completa": corrida_completa,
        "seleccion": {
            "pliegues": list(pliegues),
            "brazos": list(brazos),
            "etiquetas": list(etiquetas),
        },
        "preregistro": {
            "referencia": "docs/p84_spec.md seccion 3 (EXP-02) + docs/p84_preregistro.md",
            "criterio_cierre": (
                "si no pasa Gate 1 standalone (+0.005, 4/5, IC95 > 0) ni el gate de "
                "complementariedad (ensemble_gain >= 0.0015, IC95 > 0), cerrar fastText "
                "supervisado sin abrir otra rejilla"
            ),
            "gate": {
                "delta_minimo": GATE_DELTA,
                "pliegues_minimos": GATE_PLIEGUES,
                "ensemble_gain_minimo": GATE_ENSEMBLE_GAIN,
                "ic95_excluye_cero": True,
                "lectura_primaria": "anidada (umbral elegido en inner_val)",
                "determinismo_exacto": True,
            },
        },
        "config": {
            "bin_fasttext": str(CONFIG["bin_fasttext"]),
            "hash_fasttext": HASH_FASTTEXT,
            "hiperparametros_fijos": HIPERPARAMETROS_FIJOS,
            "brazos": {brazo: {"minn": minn, "maxn": maxn} for brazo, (minn, maxn) in BRAZOS.items()},
            "epocas_candidatas": list(EPOCAS_CANDIDATAS),
            "rejilla_umbrales": [float(REJILLA_UMBRALES[0]), float(REJILLA_UMBRALES[-1]), 0.01],
            "bootstrap": {"repeticiones": REPETICIONES_BOOTSTRAP, "semilla": SEMILLA_BOOTSTRAP},
            "semillas_internas": [
                SEMILLA_INTERNA_BASE + 100 * pliegue + 7 * j
                for pliegue in range(1, 6)
                for j in range(N_INTERNOS)
            ],
        },
        "gate0": {"evidencia": datos.evidencia, "identidad_v03": identidad},
        "versiones": versiones_entorno(),
        "determinismo": determinismo,
        "brazos": {
            brazo: {"resumen": {clave: valor for clave, valor in resumenes[brazo].items() if not isinstance(valor, np.ndarray)}}
            for brazo in brazos
        },
        "lecturas": lecturas,
        "principal_vs_control": principal_vs_control,
        "ensemble_gain": gain_por_brazo,
        "correlaciones": correlaciones,
        "matriz_errores": errores,
        "gate": gate,
        "parciales": rutas_parciales,
        "incidencias": incidencias,
        "tiempo_total_segundos": tiempo_total,
    }

    if humo:
        log("modo humo: no se escriben CSV, npz ni informes finales")
    elif corrida_completa:
        ruta_npz = guardar_oof_npz(resumenes, datos)
        carga["oof_npz"] = str(ruta_npz)
        fila = construir_fila_experiments(
            lecturas, principal_vs_control, gain_por_brazo["principal"], gate, determinismo, tiempo_total
        )
        carga["experiments_csv"] = upsert_experiments_csv(fila)
        escribir_json(CONFIG["informe_json"], carga)
        escribir_texto(CONFIG["informe_md"], construir_informe_md(carga))
        log(f"informes: {CONFIG['informe_json'].name} y {CONFIG['informe_md'].name}")
    else:
        ruta_json_parcial = CONFIG["parciales_dir"] / "p84_02_informe_parcial.json"
        ruta_md_parcial = CONFIG["parciales_dir"] / "p84_02_informe_parcial.md"
        escribir_json(ruta_json_parcial, carga)
        escribir_texto(ruta_md_parcial, construir_informe_md(carga))
        log(
            "AVISO: corrida parcial: sin npz OOF ni experiments.csv; informe etiquetado en "
            f"{ruta_json_parcial.name} y {ruta_md_parcial.name}"
        )

    # ---------- resumen final ----------
    log("=" * 78)
    log(f"P84-02 {'HUMO' if humo else 'OK'} | device cpu | {tiempo_total:.1f}s")
    if determinismo is not None:
        log(
            f"   determinismo ({determinismo['brazo']}, {determinismo['etiqueta']}, "
            f"pliegue {determinismo['pliegue']}): max|delta| "
            f"{determinismo['max_delta_probabilidades']} | identicas {determinismo['identicas']}"
        )
    if corrida_completa:
        lectura = lecturas["principal"]["anidada"]
        comparacion = lectura["comparacion"]
        ic = lectura["bootstrap"]["ic95"]
        log(
            f"   principal anidada: F1 {comparacion['f1_global']:.6f} | delta global "
            f"{comparacion['delta_global']:+.6f} | pliegues {comparacion['pliegues_mejorados']}/5 "
            f"| IC95 [{ic[0]:+.6f}, {ic[1]:+.6f}]"
        )
        gain = gain_por_brazo["principal"]
        log(
            f"   ensemble_gain {gain['ensemble_gain']:+.6f} | IC95 "
            f"[{gain['ic95'][0]:+.6f}, {gain['ic95'][1]:+.6f}] | "
            f"principal vs control {principal_vs_control['delta_medio_pliegues']:+.6f}"
        )
        log(f"   decision: {gate['decision']}")
    else:
        log("   modo humo/parcial: resultados de pliegue en reports/p84_02_parciales/")
    log("=" * 78)
    return 0


if __name__ == "__main__":
    try:
        codigo_salida = main()
    except SystemExit:
        raise
    except Exception:  # noqa: BLE001 - red de seguridad del punto de entrada
        traceback.print_exc()
        sys.exit(1)
    sys.exit(codigo_salida)
