"""Primera pasada de estilo sobre los archivos LaTeX del informe.

No reescribe nada: solo señala rayas, frases gastadas y marcas pendientes.
La revisión final se hace leyendo el texto completo.
"""

import re
from pathlib import Path

BASE = Path(__file__).resolve().parent
ARCHIVOS = [BASE / "main.tex", *sorted((BASE / "secciones").glob("*.tex"))]

PATRONES = {
    "raya larga": r"---|—",
    "doble guion": r"(?<!-)--(?!-)",
    "cabe destacar": r"cabe destacar",
    "es importante mencionar": r"es importante mencionar",
    "no solo": r"no solo",
    "sino tambien": r"sino también",
    "en el corazon de": r"en el corazón de",
    "sinergia": r"sinergia",
    "apalancar": r"apalancar",
    "potenciar": r"potenciar",
    "robusto": r"robusto",
    "en la era de": r"en la era de",
    "revolucionario": r"revolucionario",
    "sin lugar a dudas": r"sin lugar a dudas",
    "TODO": r"%\s*TODO",
}

for archivo in ARCHIVOS:
    texto = archivo.read_text(encoding="utf-8")
    hallazgos = []
    for nombre, patron in PATRONES.items():
        for coincidencia in re.finditer(patron, texto, flags=re.IGNORECASE):
            linea = texto[: coincidencia.start()].count("\n") + 1
            hallazgos.append((linea, nombre, coincidencia.group(0)))
    if hallazgos:
        print(f"\n{archivo.relative_to(BASE)}")
        for linea, nombre, fragmento in sorted(hallazgos):
            print(f"  línea {linea}: {nombre} -> {fragmento!r}")
