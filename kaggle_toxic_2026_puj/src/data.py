"""Carga, descarga y validación de los datos de KAGGLE_TOXIC_2026_PUJ.

Competencia: https://www.kaggle.com/competitions/tweet-toxic-2026-puj
Organización: JarvisLab_PUJ (antes NLP_Course_PUJ)
Métrica oficial: F1 Score. Deadline: 2026-10-19.

Formato verificado empíricamente el 2026-09-22
----------------------------------------------
- ``toxic_train.csv`` (205.417 bytes): UTF-8 puro, separador ``;``, quoting
  estándar de CSV. Las 1.731 filas de datos producen exactamente 5 campos
  (``id;text;HS;TR;AG``). El texto está normalizado a ASCII por los
  organizadores (sin tildes ni eñe), por lo que las variantes de
  preprocesamiento propias deben reconstruir esa información si se desea.
- ``toxic_test.csv`` (127.006 bytes): **cp1252**, no UTF-8 (6 bytes ``0xBF``
  correspondientes a "¿"). Separador ``;``, quoting estándar; 1.154 filas con
  2 campos (``id;text``).
- ``submission.csv`` (13.860 bytes): UTF-8, separador ``,``, 4 columnas
  (``id,HS,TR,AG``) con la plantilla todo-uno.

Regla de oro: el conjunto de test nunca se usa para entrenar, ajustar
vocabularios, vectorizadores, umbrales ni selección de features.
"""

from __future__ import annotations

import hashlib
import os
import unicodedata
from pathlib import Path
from typing import Final

import pandas as pd
import requests

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
RAW_DIR: Final[Path] = PROJECT_ROOT / "data" / "raw"

TRAIN_PATH: Final[Path] = RAW_DIR / "toxic_train.csv"
TEST_PATH: Final[Path] = RAW_DIR / "toxic_test.csv"
SUBMISSION_PATH: Final[Path] = RAW_DIR / "submission.csv"

COMPETITION_SLUG: Final[str] = "tweet-toxic-2026-puj"
KAGGLE_DOWNLOAD_URL: Final[str] = (
    "https://www.kaggle.com/api/v1/competitions/data/download/{slug}/{filename}"
)

TRAIN_COLUMNS: Final[tuple[str, ...]] = ("id", "text", "HS", "TR", "AG")
TEST_COLUMNS: Final[tuple[str, ...]] = ("id", "text")
SUBMISSION_COLUMNS: Final[tuple[str, ...]] = ("id", "HS", "TR", "AG")
LABEL_COLUMNS: Final[tuple[str, ...]] = ("HS", "TR", "AG")

# Tamaños oficiales reportados por la API de Kaggle (bytes).
EXPECTED_BYTES: Final[dict[str, int]] = {
    "toxic_train.csv": 205_417,
    "toxic_test.csv": 127_006,
    "submission.csv": 13_860,
}
EXPECTED_ROWS: Final[dict[str, int]] = {
    "toxic_train.csv": 1_731,
    "toxic_test.csv": 1_154,
    "submission.csv": 1_154,
}


def sha256_of(path: Path) -> str:
    """Devuelve el hash sha256 de un archivo.

    Args:
        path: ruta del archivo.

    Returns:
        Cadena hexadecimal con el hash sha256.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def decode_bytes(raw: bytes) -> tuple[str, str]:
    """Decodifica bytes CSV probando UTF-8 y cayendo a cp1252.

    Args:
        raw: contenido binario del archivo.

    Returns:
        Tupla ``(texto, encoding_usado)``.
    """
    try:
        return raw.decode("utf-8"), "utf-8"
    except UnicodeDecodeError:
        return raw.decode("cp1252", errors="replace"), "cp1252"


def read_csv_robusto(path: Path, sep: str, columns: tuple[str, ...]) -> pd.DataFrame:
    """Lee un CSV oficial respetando el encoding y el quoting verificados.

    El quoting estándar de CSV es el único que produce el número exacto de
    campos en ``toxic_train.csv`` (ver docstring del módulo); por eso no se
    usa ``quoting=QUOTE_NONE``.

    Args:
        path: ruta del CSV.
        sep: separador de campos (``;`` o ``,``).
        columns: nombres de columna esperados, en orden.

    Returns:
        DataFrame con ``columns`` y todas las columnas en tipo ``str``.

    Raises:
        ValueError: si el número de columnas leídas no coincide con el esperado.
    """
    raw = path.read_bytes()
    _, encoding = decode_bytes(raw)
    frame = pd.read_csv(
        path,
        sep=sep,
        dtype=str,
        keep_default_na=False,
        encoding="utf-8" if encoding == "utf-8" else "cp1252",
        encoding_errors="replace",
    )
    frame.columns = [column.strip() for column in frame.columns]
    if tuple(frame.columns) != columns:
        raise ValueError(f"Columnas inesperadas en {path.name}: {list(frame.columns)}")
    frame["id"] = frame["id"].str.strip().str.strip('"')
    return frame


def read_train(path: Path | None = None) -> pd.DataFrame:
    """Carga ``toxic_train.csv`` con etiquetas HS, TR y AG como enteros.

    Args:
        path: ruta alternativa del archivo; por defecto ``data/raw/toxic_train.csv``.

    Returns:
        DataFrame con columnas ``id, text, HS, TR, AG``.
    """
    frame = read_csv_robusto(path or TRAIN_PATH, sep=";", columns=TRAIN_COLUMNS)
    for column in LABEL_COLUMNS:
        frame[column] = frame[column].astype(int)
    return frame


def read_test(path: Path | None = None) -> pd.DataFrame:
    """Carga ``toxic_test.csv`` (sin etiquetas).

    Args:
        path: ruta alternativa del archivo; por defecto ``data/raw/toxic_test.csv``.

    Returns:
        DataFrame con columnas ``id, text``.
    """
    return read_csv_robusto(path or TEST_PATH, sep=";", columns=TEST_COLUMNS)


def read_submission(path: Path | None = None) -> pd.DataFrame:
    """Carga la plantilla ``submission.csv`` con etiquetas enteras.

    Args:
        path: ruta alternativa del archivo; por defecto ``data/raw/submission.csv``.

    Returns:
        DataFrame con columnas ``id, HS, TR, AG`` como enteros.
    """
    frame = read_csv_robusto(path or SUBMISSION_PATH, sep=",", columns=SUBMISSION_COLUMNS)
    for column in LABEL_COLUMNS:
        frame[column] = frame[column].astype(int)
    return frame


def validate_frame(frame: pd.DataFrame, columns: tuple[str, ...], n_rows: int) -> dict:
    """Valida integridad básica de un DataFrame oficial.

    Args:
        frame: DataFrame a validar.
        columns: columnas esperadas.
        n_rows: número de filas esperado.

    Returns:
        Diccionario con el informe de validación (todo booleano salvo conteos).
    """
    report = {
        "n_rows": len(frame),
        "n_rows_esperado": n_rows,
        "columnas": tuple(frame.columns),
        "sin_nulos": not frame.isna().any().any(),
        "ids_numericos": bool(frame["id"].str.fullmatch(r"\d+").all()),
        "ids_unicos": bool(frame["id"].is_unique),
        "texto_no_vacio": bool((frame["text"].str.strip() != "").all()),
    }
    report["ok"] = bool(
        report["n_rows"] == report["n_rows_esperado"]
        and report["sin_nulos"]
        and report["ids_numericos"]
        and report["ids_unicos"]
        and report["texto_no_vacio"]
        and report["columnas"] == columns
    )
    return report


def strip_accents(text: str) -> str:
    """Elimina diacríticos de un texto (NFKD + descarte de marcas).

    Args:
        text: texto de entrada.

    Returns:
        Texto sin marcas diacríticas.
    """
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(char for char in decomposed if not unicodedata.combining(char))


def download_competition(
    dest_dir: Path | None = None,
    token: str | None = None,
    competition: str = COMPETITION_SLUG,
    filenames: tuple[str, ...] = tuple(EXPECTED_BYTES),
    overwrite: bool = False,
) -> dict[str, str]:
    """Descarga los CSV oficiales desde la API pública de Kaggle.

    Alternativa reproducible al CLI ``kaggle``: usa el token de la API
    (formato ``KGAT_...``) mediante el encabezado ``Authorization: Bearer``.
    El token se lee de ``token`` o de la variable de entorno
    ``KAGGLE_API_TOKEN`` y nunca se escribe en disco.

    Args:
        dest_dir: carpeta destino; por defecto ``data/raw``.
        token: token de la API; si es ``None`` se lee del entorno.
        competition: slug de la competencia.
        filenames: nombres de los archivos a descargar.
        overwrite: si ``False``, omite archivos ya existentes.

    Returns:
        Diccionario ``nombre_archivo -> sha256``.

    Raises:
        RuntimeError: si falta el token o una descarga falla.
    """
    token = token or os.environ.get("KAGGLE_API_TOKEN", "")
    if not token:
        raise RuntimeError("Falta el token: define KAGGLE_API_TOKEN o pasa token=...")
    dest = Path(dest_dir) if dest_dir else RAW_DIR
    dest.mkdir(parents=True, exist_ok=True)
    headers = {"Authorization": f"Bearer {token}"}
    hashes: dict[str, str] = {}
    for filename in filenames:
        target = dest / filename
        if target.exists() and not overwrite:
            hashes[filename] = sha256_of(target)
            continue
        url = KAGGLE_DOWNLOAD_URL.format(slug=competition, filename=filename)
        response = requests.get(url, headers=headers, timeout=120)
        if response.status_code != 200:
            raise RuntimeError(
                f"Descarga fallida de {filename}: HTTP {response.status_code}"
            )
        target.write_bytes(response.content)
        hashes[filename] = sha256_of(target)
    return hashes


if __name__ == "__main__":
    train_frame = read_train()
    test_frame = read_test()
    submission_frame = read_submission()
    print("train :", train_frame.shape, validate_frame(train_frame, TRAIN_COLUMNS, EXPECTED_ROWS["toxic_train.csv"]))
    print("test  :", test_frame.shape, validate_frame(test_frame, TEST_COLUMNS, EXPECTED_ROWS["toxic_test.csv"]))
    print(
        "subm  :",
        submission_frame.shape,
        validate_frame(submission_frame, SUBMISSION_COLUMNS, EXPECTED_ROWS["submission.csv"]),
    )
