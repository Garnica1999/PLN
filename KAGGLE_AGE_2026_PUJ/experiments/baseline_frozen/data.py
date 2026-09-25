"""Descarga, verificación de integridad y carga de los datos de la competencia.

Uso típico::

    python -m src.data --download --verify

El token de Kaggle se resuelve en este orden:
1. Variable de entorno ``KAGGLE_API_TOKEN`` (formato ``KGAT_...``).
2. ``~/.kaggle/kaggle.json`` con la clave ``token`` (tokens nuevos).
3. ``~/.kaggle/kaggle.json`` con ``username`` + ``key`` (autenticación básica clásica).

Reglas del proyecto: el token NUNCA se escribe dentro del repositorio y el test no se usa
para entrenar ni para construir vocabulario (ver DECISIONS.md, D-001).
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import os
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

from src import config as cfg

API_BASE = "https://www.kaggle.com/api/v1"
DOWNLOAD_ALL = f"{API_BASE}/competitions/data/download-all/{cfg.COMPETITION}"
CHUNK = 1 << 20  # 1 MiB


class DataError(RuntimeError):
    """Error de descarga, integridad o esquema de los datos."""


# --------------------------------------------------------------------------------------
# Credenciales
# --------------------------------------------------------------------------------------
def get_credentials() -> dict[str, str]:
    """Devuelve la credencial de Kaggle disponible en el sistema.

    Returns
    -------
    dict
        Diccionario con ``mode`` (``bearer`` o ``basic``) y las claves necesarias.

    Raises
    ------
    DataError
        Si no se encuentra ninguna credencial utilizable.
    """
    env_token = os.environ.get("KAGGLE_API_TOKEN")
    if env_token:
        return {"mode": "bearer", "token": env_token.strip()}

    cred_path = Path.home() / ".kaggle" / "kaggle.json"
    if not cred_path.exists():
        raise DataError(
            "No hay credenciales de Kaggle: exporta KAGGLE_API_TOKEN o crea "
            f"{cred_path} con la clave 'token' o 'username'+'key'."
        )
    cred = json.loads(cred_path.read_text(encoding="utf-8"))
    if cred.get("token"):
        return {"mode": "bearer", "token": cred["token"].strip()}
    if cred.get("username") and cred.get("key"):
        return {"mode": "basic", "username": cred["username"], "key": cred["key"]}
    raise DataError(f"Credencial ilegible en {cred_path}: faltan 'token' o 'username'/'key'.")


def _auth_headers(cred: dict[str, str]) -> dict[str, str]:
    """Construye las cabeceras HTTP de autenticación."""
    if cred["mode"] == "bearer":
        return {"Authorization": f"Bearer {cred['token']}"}
    raw = f"{cred['username']}:{cred['key']}".encode()
    return {"Authorization": "Basic " + base64.b64encode(raw).decode()}


# --------------------------------------------------------------------------------------
# Descarga
# --------------------------------------------------------------------------------------
class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Evita que urllib siga redirecciones para controlar las cabeceras por salto."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: D102
        return None


def _open(url: str, headers: dict[str, str] | None = None, timeout: int = 120):
    """Abre una URL sin seguir redirecciones automáticamente."""
    request = urllib.request.Request(url, headers=headers or {})
    opener = urllib.request.build_opener(_NoRedirect)
    try:
        return opener.open(request, timeout=timeout)
    except urllib.error.HTTPError as exc:  # 3xx/4xx/5xx llegan aquí con NoRedirect
        return exc


def _stream_to_file(response, destination: Path) -> tuple[int, str]:
    """Escribe la respuesta HTTP en disco y calcula su SHA-256.

    Returns
    -------
    tuple
        ``(bytes_escritos, sha256_hex)``.
    """
    digest = hashlib.sha256()
    total = 0
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as handle:
        while True:
            chunk = response.read(CHUNK)
            if not chunk:
                break
            handle.write(chunk)
            digest.update(chunk)
            total += len(chunk)
    return total, digest.hexdigest()


def download_raw(force: bool = False) -> dict[str, str]:
    """Descarga el archivo ZIP de la competencia a ``data/raw`` y extrae los CSV.

    Parameters
    ----------
    force : bool
        Si es ``True`` vuelve a descargar aunque el ZIP exista.

    Returns
    -------
    dict
        ``{"zip": sha256, "size": bytes, "files": nombres extraídos}``.

    Raises
    ------
    DataError
        Si la descarga no devuelve un ZIP válido.
    """
    if cfg.RAW_ZIP.exists() and not force:
        with zipfile.ZipFile(cfg.RAW_ZIP) as zf:
            names = zf.namelist()
        return {"zip": _sha256_file(cfg.RAW_ZIP), "size": str(cfg.RAW_ZIP.stat().st_size),
                "files": ",".join(names), "cached": "true"}

    cred = get_credentials()
    headers = _auth_headers(cred)
    response = _open(DOWNLOAD_ALL, headers)
    status = getattr(response, "status", None) or getattr(response, "code", None)
    if status in (301, 302, 303, 307, 308):
        location = response.headers["Location"]
        # La URL firmada de GCS ya trae su propia autorización: no se reenvía el token.
        response = _open(location, {})
        status = getattr(response, "status", None) or getattr(response, "code", None)
    if status != 200:
        detail = response.read(300).decode("utf-8", "replace") if hasattr(response, "read") else ""
        raise DataError(f"Descarga fallida (HTTP {status}): {detail}")

    size, digest = _stream_to_file(response, cfg.RAW_ZIP)
    if not zipfile.is_zipfile(cfg.RAW_ZIP):
        cfg.RAW_ZIP.unlink(missing_ok=True)
        raise DataError("El archivo descargado no es un ZIP válido.")

    with zipfile.ZipFile(cfg.RAW_ZIP) as zf:
        names = zf.namelist()
        for name in names:
            target = cfg.RAW / Path(name).name
            with zf.open(name) as source, target.open("wb") as destination:
                destination.write(source.read())

    _write_sha256(cfg.RAW_ZIP.name, digest)
    return {"zip": digest, "size": str(size), "files": ",".join(names), "cached": "false"}


def _sha256_file(path: Path) -> str:
    """Calcula el SHA-256 de un archivo en disco."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_sha256(name: str, digest: str) -> None:
    """Registra el hash de un archivo en ``data/raw/SHA256SUMS.txt``."""
    lines: list[str] = []
    if cfg.SHA256_FILE.exists():
        lines = [ln for ln in cfg.SHA256_FILE.read_text(encoding="utf-8").splitlines()
                 if not ln.endswith(f"  {name}")]
    lines.append(f"{digest}  {name}")
    cfg.SHA256_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------------------
# Integridad estructural
# --------------------------------------------------------------------------------------
def check_field_integrity(path: Path, sep: str, expected_fields: int, max_report: int = 5) -> dict:
    """Verifica con ``csv.reader`` que cada registro tenga el número de campos esperado.

    Detecta separadores embebidos no escapados, filas truncadas y saltos de línea dentro
    de los campos (los maneja correctamente al ser un parser de registros).

    Returns
    -------
    dict
        ``{"records": int, "bad_records": int, "examples": [str, ...]}``.
    """
    records = 0
    bad = 0
    examples: list[str] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for line_no, row in enumerate(csv.reader(handle, delimiter=sep), start=1):
            records += 1
            if len(row) != expected_fields:
                bad += 1
                if len(examples) < max_report:
                    examples.append(f"linea {line_no}: {len(row)} campos -> {row[:4]}")
    return {"records": records, "bad_records": bad, "examples": examples}


# --------------------------------------------------------------------------------------
# Carga
# --------------------------------------------------------------------------------------
def clean_text(series: pd.Series) -> pd.Series:
    """Elimina el padding de espacios del ancho fijo y normaliza cadenas vacías.

    El corpus almacena el texto en un campo de 140 caracteres rellenado con espacios
    (ver D-005): ese relleno es un artefacto de exportación, no señal.

    Parameters
    ----------
    series : pandas.Series
        Columna de texto cruda.

    Returns
    -------
    pandas.Series
        Texto sin espacios finales/iniciales; ``""`` donde el valor era nulo.
    """
    return series.fillna("").astype(str).str.strip()


def load_train() -> pd.DataFrame:
    """Carga ``age_train.csv`` con validación de esquema.

    Returns
    -------
    pandas.DataFrame
        Columnas ``id`` (int64), ``text`` (str limpio) y ``age_range`` (categoría).
    """
    _require(cfg.TRAIN_FILE)
    frame = pd.read_csv(cfg.TRAIN_FILE, sep=cfg.TRAIN_SEP, encoding="utf-8-sig")
    _expect_columns(frame, [cfg.ID_COL, cfg.TEXT_COL, cfg.TARGET], cfg.TRAIN_FILE.name)
    frame[cfg.TEXT_COL] = clean_text(frame[cfg.TEXT_COL])
    frame[cfg.TARGET] = pd.Categorical(frame[cfg.TARGET], categories=cfg.CLASSES, ordered=True)
    return frame


def load_test() -> pd.DataFrame:
    """Carga ``age_test.csv`` con validación de esquema."""
    _require(cfg.TEST_FILE)
    frame = pd.read_csv(cfg.TEST_FILE, sep=cfg.TEST_SEP, encoding="utf-8-sig")
    _expect_columns(frame, [cfg.ID_COL, cfg.TEXT_COL], cfg.TEST_FILE.name)
    frame[cfg.TEXT_COL] = clean_text(frame[cfg.TEXT_COL])
    return frame


def load_sample_submission() -> pd.DataFrame:
    """Carga ``age_submission.csv`` (plantilla oficial de envío)."""
    _require(cfg.SUBMISSION_FILE)
    frame = pd.read_csv(cfg.SUBMISSION_FILE, sep=cfg.SUBMISSION_SEP, encoding="utf-8-sig")
    _expect_columns(frame, [cfg.ID_COL, cfg.TARGET], cfg.SUBMISSION_FILE.name)
    return frame


def _require(path: Path) -> None:
    """Verifica que un archivo exista antes de leerlo."""
    if not path.exists():
        raise DataError(f"Falta {path}. Ejecuta primero: python -m src.data --download")


def _expect_columns(frame: pd.DataFrame, columns: list[str], name: str) -> None:
    """Valida que el DataFrame tenga exactamente las columnas esperadas."""
    if list(frame.columns) != columns:
        raise DataError(f"{name}: columnas {list(frame.columns)} != esperadas {columns}")


# --------------------------------------------------------------------------------------
# Verificación integral
# --------------------------------------------------------------------------------------
def verify() -> dict:
    """Ejecuta la verificación completa de descarga, integridad y esquema.

    Returns
    -------
    dict
        Resumen con hashes, conteos, integridad de campos y validación de la submission.
    """
    train = load_train()
    test = load_test()
    sample = load_sample_submission()

    integrity_train = check_field_integrity(cfg.TRAIN_FILE, cfg.TRAIN_SEP, 3)
    integrity_test = check_field_integrity(cfg.TEST_FILE, cfg.TEST_SEP, 2)
    integrity_sample = check_field_integrity(cfg.SUBMISSION_FILE, cfg.SUBMISSION_SEP, 2)

    checks = {
        "train_rows": len(train),
        "test_rows": len(test),
        "sample_rows": len(sample),
        "train_id_unique": bool(train[cfg.ID_COL].is_unique),
        "test_id_unique": bool(test[cfg.ID_COL].is_unique),
        "test_ids_equal_sample_ids": bool(test[cfg.ID_COL].equals(sample[cfg.ID_COL])),
        "train_classes": train[cfg.TARGET].value_counts().to_dict(),
        "train_empty_text": int((train[cfg.TEXT_COL] == "").sum()),
        "test_empty_text": int((test[cfg.TEXT_COL] == "").sum()),
        "train_max_text_len": int(train[cfg.TEXT_COL].str.len().max()),
        "test_max_text_len": int(test[cfg.TEXT_COL].str.len().max()),
        "integrity_train": integrity_train,
        "integrity_test": integrity_test,
        "integrity_sample": integrity_sample,
    }
    return checks


def save_interim() -> dict[str, str]:
    """Guarda los datos limpios en Parquet dentro de ``data/interim``."""
    paths: dict[str, str] = {}
    for name, frame in (("train", load_train()), ("test", load_test()),
                        ("sample_submission", load_sample_submission())):
        destination = cfg.INTERIM / f"{name}.parquet"
        frame.to_parquet(destination, index=False)
        paths[name] = str(destination)
    return paths


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    """Punto de entrada de línea de comandos del módulo."""
    parser = argparse.ArgumentParser(description="Datos de TweetAge_2026_PUJ")
    parser.add_argument("--download", action="store_true", help="descarga el ZIP de la competencia")
    parser.add_argument("--force", action="store_true", help="fuerza nueva descarga")
    parser.add_argument("--verify", action="store_true", help="valida integridad y esquema")
    parser.add_argument("--save-interim", action="store_true", help="guarda Parquet intermedio")
    args = parser.parse_args(argv)

    if not any([args.download, args.verify, args.save_interim]):
        parser.print_help()
        return 1

    if args.download:
        info = download_raw(force=args.force)
        print(f"[download] sha256={info['zip']} bytes={info['size']} cached={info.get('cached')}")
        print(f"[download] archivos={info['files']}")

    if args.save_interim:
        print(f"[interim] {save_interim()}")

    if args.verify:
        checks = verify()
        print(json.dumps(checks, indent=2, ensure_ascii=False))

    return 0


if __name__ == "__main__":
    sys.exit(main())
