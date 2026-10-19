"""Verificacion de caracteres no latinos en los ficheros propios del proyecto.

Por que existe
--------------
En este proyecto se escribe en espanol y sin emojis, y varias veces se ha colado un
caracter de otro alfabeto al redactar deprisa: un hebreo en D-045, un par de ideogramas
chinese en D-035, dos kanji en el docstring de `embedding_16_negative_v5_1.py`. Son
errores invisibles al leer y evidentes al buscar. Este script los detecta de forma
repetible, con la diferencia importante de que **distingue el error del dato legitimo**:

- **Simbolos matematicos griegos** (`Delta`, `alfa`, `beta`, `sigma`, `epsil`, ...) en
  tablas e informes: son notacion estandar y se permiten.
- **Alfabetos no latinos como dato** (hebreo, griego, japones) en los informes de
  cobertura de vocabulario: son tokens que el tokenizador debe recortar bien. Se
  permiten solo en esos informes concretos, y en ningun otro sitio.
- **Todo lo demas** en codigo, decisiones, informes o scripts: es un error.

Uso
---
    /home/carlos/venv/bin/python tools/verificar_caracteres.py

Devuelve codigo de salida 1 si encuentra algun error, para poder usarse como puerta en
la auditoria final sin tener que mirar la salida a ojo.
"""

from __future__ import annotations

import re
import sys
import unicodedata
from pathlib import Path
from typing import NamedTuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]

EXTENSIONES = {".md", ".py", ".csv", ".json", ".txt", ".ipynb"}
DIRECTORIOS_EXCLUIDOS = {
    ".git",
    "__pycache__",
    "data",
    "node_modules",
    "venv",
    "tools/fastText",  # codigo de terceros: no se corrige ni se audita aqui
    "features",
    "modelos",
}

# Simbolos matematicos griegos admitidos como notacion. Se construyen con chr() para que
# este fichero no contenga ninguno de ellos en su fuente y no se detecte a si mismo.
SIMBOLOS_PERMITIDOS = frozenset(
    chr(codigo)
    for codigo in (
        0x0394,  # Delta
        0x03A3,  # Sigma mayuscula (sumatoria)
        0x03B1,  # alfa
        0x03B2,  # beta
        0x03B5,  # epsil
        0x03B9,  # iota
        0x03BA,  # kappa
        0x03BB,  # lambda
        0x03BC,  # mu
        0x03BD,  # nu
        0x03C0,  # pi
        0x03C3,  # sigma minuscula
        0x03C4,  # tau
        0x03C6,  # phi (correlacion de Pearson / Phi)
        0x2212,  # signo menos
        0x2264,  # menor o igual
        0x2265,  # mayor o igual
        0x00B1,  # mas menos
        0x00D7,  # por
    )
)

# Ficheros donde un alfabeto no latino es el CONTIENIDO y no un error: son informes que
# listan tokens de otros alfabetos para comprobar que el tokenizador los recorta bien.
EXCEPCIONES_POR_RUTA = {
    "reports/embedding_vocabulary_coverage.md",
    "reports/embedding_vocabulary_coverage.csv",
}

ALFABETOS: dict[str, str] = {
    "hebreo": r"[\u0590-\u05FF]",
    "arabe": r"[\u0600-\u06FF]",
    "devanagari": r"[\u0900-\u097F]",
    "tamil y telugu": r"[\u0B80-\u0BFF\u0C00-\u0C7F]",
    "cirilico": r"[\u0400-\u04FF]",
    "griego y copto": r"[\u0370-\u03FF]",
    "hangul": r"[\uAC00-\uD7AF\u1100-\u11FF]",
    "kanji y kana": r"[\u3000-\u9FFF\uFF00-\uFFEF]",
    "emoji y simbolos": r"[\U0001F300-\U0001FAFF\u2600-\u27BF]",
}

COMPIADOS = {nombre: re.compile(patron) for nombre, patron in ALFABETOS.items()}


class Hallazgo(NamedTuple):
    """Un caracter no permitido encontrado en un fichero propio.

    Attributes:
        ruta: ruta relativa al proyecto.
        nombre: nombre del alfabeto detectado.
        caracter: el caracter encontrado.
        contexto: fragmento de texto que lo rodea.
        es_error: si es un error o una excepcion permitida.
    """

    ruta: str
    nombre: str
    caracter: str
    contexto: str
    es_error: bool


def _es_excepcion(ruta_relativa: str) -> bool:
    """Indica si un fichero es una de las excepciones de contenido.

    Args:
        ruta_relativa: ruta del fichero relativa al proyecto.

    Returns:
        ``True`` si en ese fichero un alfabeto no latino es dato y no error.
    """
    return ruta_relativa in EXCEPCIONES_POR_RUTA


def _es_simbolo(caracter: str) -> bool:
    """Indica si un caracter es un simbolo matematico admitido.

    Args:
        caracter: caracter a clasificar.

    Returns:
        ``True`` si es notacion matematica Greeks permitida.
    """
    return caracter in SIMBOLOS_PERMITIDOS


def escanear() -> list[Hallazgo]:
    """Recorre los ficheros propios y recoge los caracteres no permitidos.

    Returns:
        Lista de hallazgos, con los errores primero.
    """
    hallazgos: list[Hallazgo] = []
    for ruta in sorted(PROJECT_ROOT.rglob("*")):
        if not ruta.is_file() or ruta.suffix not in EXTENSIONES:
            continue
        relativa = ruta.relative_to(PROJECT_ROOT).as_posix()
        if any(
            relativa == excluido or relativa.startswith(excluido + "/")
            for excluido in DIRECTORIOS_EXCLUIDOS
        ):
            continue
        try:
            texto = ruta.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for nombre, patron in COMPIADOS.items():
            for coincidencia in patron.finditer(texto):
                caracter = coincidencia.group()
                if _es_simbolo(caracter):
                    continue
                inicio = max(0, coincidencia.start() - 35)
                contexto = texto[inicio : coincidencia.end() + 35].replace("\n", " ")
                hallazgos.append(
                    Hallazgo(
                        ruta=relativa,
                        nombre=nombre,
                        caracter=caracter,
                        contexto=contexto,
                        es_error=not _es_excepcion(relativa),
                    )
                )
    return sorted(hallazgos, key=lambda h: (not h.es_error, h.ruta, h.nombre))


def main() -> int:
    """Ejecuta la verificacion y devuelve el codigo de salida.

    Returns:
        ``0`` si no hay errores, ``1`` si hay al menos uno.
    """
    hallazgos = escanear()
    errores = [h for h in hallazgos if h.es_error]
    excepciones = [h for h in hallazgos if not h.es_error]

    if excepciones:
        print(f"Permitidos como dato o notacion ({len(excepciones)}):")
        vistas: set[tuple[str, str]] = set()
        for hallazgo in excepciones:
            clave = (hallazgo.ruta, hallazgo.nombre)
            if clave in vistas:
                continue
            vistas.add(clave)
            print(f"  {hallazgo.ruta} | {hallazgo.nombre}")
        print()

    if errores:
        print(f"ERRORES: {len(errores)} caracteres no permitidos en ficheros propios")
        for hallazgo in errores:
            nombre_unicode = unicodedata.name(hallazgo.caracter, "desconocido")
            print(
                f"  {hallazgo.ruta} | {hallazgo.nombre} | "
                f"U+{ord(hallazgo.caracter):04X} ({nombre_unicode})"
            )
            print(f"     ...{hallazgo.contexto}...")
        return 1

    print("OK: 0 caracteres no permitidos en los ficheros propios del proyecto.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
