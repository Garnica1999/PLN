"""Configuración central del proyecto KAGGLE_AGE_2026_PUJ.

Define rutas, constantes de la competencia, clases objetivo y utilidades de semillas.
Todos los módulos del proyecto importan las rutas desde aquí para evitar rutas
absolutas dispersas.
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------------------
# Rutas
# --------------------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"
INTERIM = DATA / "interim"
PROCESSED = DATA / "processed"
EXTERNAL = DATA / "external"
NOTEBOOKS = ROOT / "notebooks"
SCRIPTS = ROOT / "scripts"
SRC = ROOT / "src"
FEATURES = ROOT / "features"
MODELS = ROOT / "models"
EXPERIMENTS = ROOT / "experiments"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
TABLES = REPORTS / "tables"
SUBMISSIONS = ROOT / "submissions"

for _path in (DATA, RAW, INTERIM, PROCESSED, EXTERNAL, NOTEBOOKS, SCRIPTS, FEATURES,
              MODELS, EXPERIMENTS, REPORTS, FIGURES, TABLES, SUBMISSIONS):
    _path.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------------------
# Competencia
# --------------------------------------------------------------------------------------
COMPETITION = "tweet-age-2026-puj"
COMPETITION_ID = 167385
RAW_ZIP = RAW / f"{COMPETITION}.zip"
SHA256_FILE = RAW / "SHA256SUMS.txt"

TRAIN_FILE = RAW / "age_train.csv"
TEST_FILE = RAW / "age_test.csv"
SUBMISSION_FILE = RAW / "age_submission.csv"

#: Separadores reales de cada archivo (verificado en la auditoría de datos).
TRAIN_SEP = ";"
TEST_SEP = ";"
SUBMISSION_SEP = ","

TARGET = "age_range"
ID_COL = "id"
TEXT_COL = "text"

#: Clases en el orden canónico de la competencia.
CLASSES = ["13-17", "18-24", "25-34", "35-49", "50-64", "65-xx"]

#: Ancho fijo con el que el corpus almacena el texto (límite histórico de Twitter).
PADDED_WIDTH = 140

# --------------------------------------------------------------------------------------
# Reproducibilidad
# --------------------------------------------------------------------------------------
SEED = 42
N_SPLITS = 5
N_REPEATS = 3


def set_seed(seed: int = SEED) -> None:
    """Fija las semillas de ``random`` y ``numpy`` para reproducibilidad.

    Parameters
    ----------
    seed : int
        Semilla a fijar.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
