"""Preferencias de la interfaz guardadas en disco."""
from __future__ import annotations

import json
from pathlib import Path

DEFAULTS = {"think": True, "strict": True, "modo_noche": False}


def read_prefs(path: Path) -> dict:
    """Lee las preferencias; si el archivo falta o no es valido, usa los valores por defecto."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return dict(DEFAULTS)
    return {key: bool(data.get(key, default)) for key, default in DEFAULTS.items()}


def write_prefs(path: Path, values: dict) -> None:
    """Escribe las preferencias; si el disco falla, la interfaz sigue funcionando."""
    data = {key: bool(values.get(key, default)) for key, default in DEFAULTS.items()}
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    except OSError:
        pass
