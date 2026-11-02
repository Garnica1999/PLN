"""Comprueba el contraste WCAG de la paleta de la interfaz.

Uso:
    python src/scripts/check_contrast.py
"""
from __future__ import annotations

import sys

# (esquema, uso, primer plano, fondo, minimo)
PAIRS = [
    ("claro", "texto sobre superficie", "#1b1c19", "#fbf8f2", 4.5),
    ("claro", "texto sobre tarjeta", "#1b1c19", "#f6f3ec", 4.5),
    ("claro", "caption sobre tarjeta", "#434944", "#f6f3ec", 4.5),
    ("claro", "etiqueta sobre panel", "#434944", "#f0ece5", 4.5),
    ("claro", "chip de intencion", "#00201a", "#a6f2d9", 4.5),
    ("claro", "mensaje del usuario", "#0a1f18", "#cfe8dd", 4.5),
    ("claro", "aviso informativo", "#3a0b00", "#ffdbd0", 4.5),
    ("claro", "aviso de abstencion", "#410002", "#ffdad6", 4.5),
    ("claro", "enlaces", "#1f6b57", "#fbf8f2", 4.5),
    ("claro", "borde de campo", "#737971", "#fbf8f2", 3.0),
    ("claro", "icono del boton enviar", "#ffffff", "#1f6b57", 3.0),
    ("oscuro", "texto sobre superficie", "#e1e3de", "#101412", 4.5),
    ("oscuro", "texto sobre tarjeta", "#e1e3de", "#181c19", 4.5),
    ("oscuro", "caption sobre tarjeta", "#bfc9c1", "#181c19", 4.5),
    ("oscuro", "etiqueta sobre panel", "#bfc9c1", "#1c201d", 4.5),
    ("oscuro", "chip de intencion", "#a6f2d9", "#005140", 4.5),
    ("oscuro", "mensaje del usuario", "#cfe8dd", "#354a42", 4.5),
    ("oscuro", "aviso informativo", "#ffdbd0", "#6f2f18", 4.5),
    ("oscuro", "aviso de abstencion", "#ffdad6", "#93000a", 4.5),
    ("oscuro", "enlaces", "#7fd8bb", "#101412", 4.5),
    ("oscuro", "borde de campo", "#89938b", "#101412", 3.0),
    ("oscuro", "icono del boton enviar", "#003828", "#7fd8bb", 3.0),
]


def luminance(color: str) -> float:
    """Luminancia relativa segun WCAG 2.1."""
    raw = color.lstrip("#")
    channels = [int(raw[index:index + 2], 16) / 255 for index in (0, 2, 4)]
    linear = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def ratio(foreground: str, background: str) -> float:
    """Razon de contraste entre dos colores."""
    first, second = luminance(foreground), luminance(background)
    high, low = max(first, second), min(first, second)
    return (high + 0.05) / (low + 0.05)


def main() -> None:
    failures = 0
    for scheme, usage, foreground, background, minimum in PAIRS:
        value = ratio(foreground, background)
        passed = value >= minimum
        failures += 0 if passed else 1
        print(
            f"{scheme:7} {usage:26} {foreground} sobre {background}  "
            f"{value:5.2f}:1  minimo {minimum}  {'ok' if passed else 'FALLA'}"
        )
    print(f"fallos: {failures}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
