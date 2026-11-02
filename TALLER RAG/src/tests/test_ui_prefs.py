"""Pruebas de las preferencias persistidas de la interfaz."""
from __future__ import annotations

from pathlib import Path

from src.rag.ui_prefs import DEFAULTS, read_prefs, write_prefs


def test_archivo_ausente_usa_defaults(tmp_path: Path) -> None:
    assert read_prefs(tmp_path / "no_existe.json") == DEFAULTS


def test_archivo_corrupto_usa_defaults(tmp_path: Path) -> None:
    path = tmp_path / "prefs.json"
    path.write_text("{roto", encoding="utf-8")
    assert read_prefs(path) == DEFAULTS


def test_ida_y_vuelta(tmp_path: Path) -> None:
    path = tmp_path / "prefs.json"
    write_prefs(path, {"think": False, "strict": False, "modo_noche": True})
    assert read_prefs(path) == {"think": False, "strict": False, "modo_noche": True}


def test_claves_desconocidas_se_ignoran(tmp_path: Path) -> None:
    path = tmp_path / "prefs.json"
    path.write_text('{"think": false, "extra": true}', encoding="utf-8")
    assert read_prefs(path) == {"think": False, "strict": True, "modo_noche": False}


def test_valores_no_booleanos_se_normalizan(tmp_path: Path) -> None:
    path = tmp_path / "prefs.json"
    path.write_text('{"think": 1, "strict": 0, "modo_noche": null}', encoding="utf-8")
    assert read_prefs(path) == {"think": True, "strict": False, "modo_noche": False}
