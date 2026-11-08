"""Pruebas del cálculo de kappa y del contraste de H1."""
from __future__ import annotations

from src.scripts.annotation_kappa import agreement, kappa, paired_bootstrap


def test_kappa_acuerdo_perfecto() -> None:
    y = [0, 1, 2, 3] * 5
    assert kappa(y, y) == 1.0
    assert agreement(y, y) == 1.0


def test_kappa_sin_varianza() -> None:
    assert kappa([3] * 10, [3] * 10) is None


def test_bootstrap_diferencia_constante() -> None:
    a = {f"q{i}": 3.0 for i in range(10)}
    b = {f"q{i}": 1.0 for i in range(10)}
    diff, lo, hi = paired_bootstrap(a, b)
    assert diff == 2.0
    assert lo == 2.0
    assert hi == 2.0


def test_load_votes_xlsx(tmp_path) -> None:
    from openpyxl import Workbook

    from src.scripts.annotation_kappa import load_votes

    book = Workbook()
    sheet = book.active
    sheet.title = "Anotacion"
    sheet.append(["item_id", "correccion", "fidelidad", "citas", "abstencion"])
    sheet.append(["abc", 3, 2, 3, 1])
    path = tmp_path / "Julian_Carrillo.xlsx"
    book.save(path)
    assert load_votes(path) == {"abc": {"correccion": 3, "fidelidad": 2, "citas": 3, "abstencion": 1}}
