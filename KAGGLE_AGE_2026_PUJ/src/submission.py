"""Construcción y validación estricta del archivo de submission.

La plantilla oficial (``age_submission.csv``) usa **coma** como separador y las columnas
``id,age_range`` con las 6 etiquetas canónicas. Este módulo valida cualquier candidato
antes de dejarlo en ``submissions/`` y no envía nada a Kaggle (los envíos son manuales).
"""

from __future__ import annotations

import pandas as pd

from src import config as cfg


class SubmissionError(ValueError):
    """La submission no cumple el formato exigido por la competencia."""


def validate_submission(candidate: pd.DataFrame, test_ids: pd.Series,
                        sample: pd.DataFrame | None = None,
                        strict_order: bool = True) -> dict:
    """Valida una submission contra el test y la plantilla oficial.

    Parameters
    ----------
    candidate : pandas.DataFrame
        Predicciones con columnas ``id`` y ``age_range``.
    test_ids : pandas.Series
        Identificadores del archivo de test (referencia de filas y orden).
    sample : pandas.DataFrame, optional
        Plantilla oficial para comparar columnas.
    strict_order : bool
        Si es ``True`` exige que el orden de ``id`` coincida con el de test.

    Returns
    -------
    dict
        Resumen de comprobaciones (todas deben ser ``True``).

    Raises
    ------
    SubmissionError
        Si alguna comprobación falla; el mensaje detalla cuáles.
    """
    checks: dict[str, bool] = {}
    checks["columns_ok"] = list(candidate.columns) == [cfg.ID_COL, cfg.TARGET]
    if sample is not None:
        checks["columns_match_sample"] = list(candidate.columns) == list(sample.columns)
    checks["rows_ok"] = len(candidate) == len(test_ids)
    checks["ids_equal_test"] = bool(set(candidate[cfg.ID_COL]) == set(test_ids))
    checks["ids_unique"] = bool(candidate[cfg.ID_COL].is_unique)
    if strict_order:
        checks["ids_same_order"] = bool(candidate[cfg.ID_COL].reset_index(drop=True)
                                        .equals(test_ids.reset_index(drop=True)))
    checks["no_nan"] = bool(candidate[cfg.TARGET].notna().all())
    checks["labels_valid"] = bool(set(candidate[cfg.TARGET].unique()) <= set(cfg.CLASSES))

    failed = [name for name, value in checks.items() if not value]
    if failed:
        detail = {
            "rows": (len(candidate), len(test_ids)),
            "labels_encontradas": sorted(map(str, candidate[cfg.TARGET].unique()))[:10],
            "ids_faltantes": int(len(set(test_ids) - set(candidate[cfg.ID_COL]))),
            "ids_sobrantes": int(len(set(candidate[cfg.ID_COL]) - set(test_ids))),
            "nan": int(candidate[cfg.TARGET].isna().sum()),
        }
        raise SubmissionError(f"Submission inválida: fallan {failed}. Detalle: {detail}")
    return checks


def write_submission(candidate: pd.DataFrame, version: int, test_ids: pd.Series,
                     sample: pd.DataFrame | None = None) -> str:
    """Valida y guarda la submission como ``submissions/age_submission_vNN.csv``.

    Returns
    -------
    str
        Ruta del archivo escrito.
    """
    validate_submission(candidate, test_ids, sample)
    destination = cfg.SUBMISSIONS / f"age_submission_v{version:02d}.csv"
    candidate.to_csv(destination, index=False)
    return str(destination)
