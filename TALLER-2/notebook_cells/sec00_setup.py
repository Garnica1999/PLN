# %% [markdown]
# # Sec00 — Setup del entorno, imports y constantes del proyecto
#
# Este bloque concentra, al inicio del notebook:
# 1. Instalación de dependencias y descarga de corpus NLTK (solo primera vez).
# 2. **TODOS los imports** de librerías usadas en el notebook (una sola vez).
# 3. **TODAS las constantes** del proyecto (rutas, colores, umbrales, nombres).
#
# Las constantes de dominio de los algoritmos (SLUG_RAIZ, EJE_POR_GRUPO,
# umbrales IDF, STOPWORDS, TOKENS_GENERICOS) viven en la sección 00b junto a
# las clases que las consumen (un solo lugar de definición por concepto).
#
# **Cómo ejecutarlo**: localmente desde `TALLER-2/notebooks/` (rutas relativas);
# en Google Colab, subiendo la carpeta `TALLER-2` completa.

# %%
"""Instalación de dependencias (solo necesario la primera vez en Colab)."""
import importlib.util
import subprocess
import sys


def instalar_si_falta(paquetes: list[str]) -> None:
    """Instala paquetes pip que no estén disponibles, en silencio.

    Args:
        paquetes: Lista de nombres de paquetes a verificar/instalar.
    """
    faltantes = [p for p in paquetes if importlib.util.find_spec(p) is None]
    if not faltantes:
        print("Dependencias ya instaladas:", ", ".join(paquetes))
        return
    print("Instalando:", ", ".join(faltantes))
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *faltantes])


instalar_si_falta(
    ["owlready2", "rdflib", "nltk", "pandas", "matplotlib", "seaborn", "networkx"]
)

# %%
"""Descarga de corpus NLTK (WordNet + Open Multilingual WordNet 1.4)."""
import nltk

for corpus in ("wordnet", "omw-1.4"):
    try:
        nltk.data.find(f"corpora/{corpus}")
        print(f"[OK] corpus '{corpus}' presente")
    except LookupError:
        print(f"Descargando corpus '{corpus}' ...")
        nltk.download(corpus)

# %%
"""TODOS los imports del proyecto (una sola vez; las secciones los reutilizan).

Se importa además el núcleo POO embebido a continuación (sec00_clases): sus
clases usan estas mismas librerías, ya cargadas en este punto.
"""
import json  # noqa: F401
import os
import re  # noqa: F401
import time  # noqa: F401
import types  # noqa: F401
import unicodedata  # noqa: F401
from abc import ABC, abstractmethod  # noqa: F401
from collections import Counter, defaultdict  # noqa: F401
from dataclasses import dataclass, field  # noqa: F401
from itertools import combinations  # noqa: F401
from pathlib import Path
from typing import Any, ClassVar  # noqa: F401

import matplotlib  # noqa: F401
import matplotlib.pyplot as plt
import networkx as nx  # noqa: F401
import numpy as np  # noqa: F401
import pandas as pd  # noqa: F401
import rdflib  # noqa: F401
import seaborn as sns
from IPython.display import display  # noqa: F401
from nltk.corpus import wordnet as wn  # noqa: F401
from owlready2 import (
    AnnotationProperty,  # noqa: F401
    ObjectProperty,  # noqa: F401
    Restriction,  # noqa: F401
    SymmetricProperty,  # noqa: F401
    Thing,  # noqa: F401
    get_ontology,  # noqa: F401
)

# Backend inline: las figuras (plt.show) quedan embebidas en el notebook.
try:
    get_ipython().run_line_magic("matplotlib", "inline")
except NameError:
    pass  # ejecución como script: backend por defecto

# %%
"""TODAS las constantes del proyecto.

Rutas (relativas al notebook: la estructura del repositorio es TALLER-2 con
data/, imagenes/, ontologia/, notebooks/, notebook_cells/), colores por grupo,
nombres de los 5 grupos, tipos de relación WordNet e IRI de la ontología.
"""


def _ruta_base() -> Path:
    """Directorio raíz del proyecto TALLER-2 (portátil: cualquier máquina).

    Estrategia de resolución en orden:
    1. Modo script (notebook_cells/): el archivo sabe su ubicación.
    2. Modo notebook: el kernel corre en algún lugar dentro del proyecto;
       se busca la raíz probando candidatos: el directorio actual, su padre
       y el padre del padre, eligiendo el primero que contenga `data/`
       con el Excel del tesauro. Esto permite abrir el notebook desde
       `notebooks/`, desde la raíz o desde cualquier subcarpeta, tanto en
       Colab (carpeta subida) como en una máquina local con el ZIP
       descomprimido en cualquier ubicación.
    """
    try:
        return Path(__file__).resolve().parent.parent
    except NameError:
        cwd = Path(os.getcwd()).resolve()
        candidatos = [cwd, cwd.parent, cwd.parent.parent, cwd / "notebooks"]
        for candidato in candidatos:
            if (candidato / "data" / "tesauro_orginal_cev.xlsx").exists():
                return candidato
        return cwd.parent  # último recurso (comportamiento previo)


RUTA_BASE = _ruta_base()
RUTA_DATOS = RUTA_BASE / "data"
RUTA_IMAGENES = RUTA_BASE / "imagenes"
RUTA_ONTOLOGIA_DIR = RUTA_BASE / "ontologia"
RUTA_NOTEBOOKS = RUTA_BASE / "notebooks"
RUTA_CELDAS = RUTA_BASE / "notebook_cells"

# Archivos de datos y resultados
RUTA_EXCEL = RUTA_DATOS / "tesauro_orginal_cev.xlsx"
RUTA_JSON_GRUPOS = RUTA_DATOS / "grupos_seleccionados.json"
RUTA_CSV_COBERTURA = RUTA_DATOS / "cobertura_wordnet_grupos.csv"
RUTA_CSV_WORDNET = RUTA_DATOS / "wordnet_relaciones.csv"
RUTA_CSV_GRAFO = RUTA_DATOS / "grafo_relaciones_edges.csv"

# Ontología
IRI_BASE = "http://www.semanticweb.org/cev/tesauro#"
RUTA_ONTOLOGIA = RUTA_ONTOLOGIA_DIR / "tesauro_cev.owl"
RUTA_TRAZABILIDAD = RUTA_ONTOLOGIA_DIR / "relaciones_afines.txt"

# Figuras (imagenes/)
FIGURAS = {
    "top_grupos": RUTA_IMAGENES / "barras_top_grupos.png",
    "profundidad": RUTA_IMAGENES / "profundidad_grupos.png",
    "tipos_termino": RUTA_IMAGENES / "distribucion_tipos_termino.png",
    "heatmap_nivel": RUTA_IMAGENES / "heatmap_terminos_grupos_nivel.png",
    "arbol_grupo": RUTA_IMAGENES / "arbol_jerarquico_armas.png",
    "cobertura_grupos": RUTA_IMAGENES / "cobertura_wordnet_grupos.png",
    "arbol_ontologia": RUTA_IMAGENES / "arbol_ontologia.png",
    "grafo_completo": RUTA_IMAGENES / "grafo_ontologia_completo.png",
    "grafo_grupos": RUTA_IMAGENES / "grafo_ontologia_grupos.png",
    "cobertura_conceptos": RUTA_IMAGENES / "cobertura_wordnet_conceptos.png",
    "heatmap_relaciones": RUTA_IMAGENES / "heatmap_relaciones_grupos.png",
    "top_conceptos": RUTA_IMAGENES / "top_relaciones_conceptos.png",
    "grafo_relaciones": RUTA_IMAGENES / "grafo_relaciones_conceptos.png",
    "profundidad_cobertura": RUTA_IMAGENES / "profundidad_versus_cobertura.png",
}

# Los 5 grupos seleccionados (sección 02) y sus ejes conceptuales
GRUPOS_SELECCIONADOS: tuple[str, ...] = (
    "Actores armados y otros participantes en las dinámicas de la guerra",
    "Impactos",
    "Armas",
    "Poblaciones",
    "Hechos victimizantes",
    "Afrontamientos y resistencias no violentas - cotidianas",
)
EJES_POR_GRUPO: dict[str, str] = {
    "Actores armados y otros participantes en las dinámicas de la guerra": "agentes",
    "Impactos": "estados_efectos",
    "Armas": "instrumentos",
    "Poblaciones": "poblaciones",
    "Hechos victimizantes": "eventos",
    "Afrontamientos y resistencias no violentas - cotidianas": "afrontamientos_resistencias",
}

# Colores por grupo (figuras)
COLORES_GRUPO: dict[str, str] = {
    "Actores armados y otros participantes en las dinámicas de la guerra": "#d62728",
    "Impactos": "#1f77b4",
    "Armas": "#2ca02c",
    "Poblaciones": "#ff7f0e",
    "Hechos victimizantes": "#9467bd",
    "Afrontamientos y resistencias no violentas - cotidianas": "#8c564b",
}
COLOR_SUPERCLASE = "#999999"

# Tipos de relación WordNet (las 9 mínimas exigidas por el taller)
TIPOS_RELACION: tuple[str, ...] = (
    "sinonimia",
    "antonimia",
    "hiponimia",
    "hiperonimia",
    "meronimia",
    "holonimia",
    "troponimia",
    "coordinados",
    "derivacion",
)

# Umbrales de EDA y selección (secciones 01-02)
UMBRAL_COBERTURA = 80          # grupos con >= 80 términos: candidatos a cobertura
UMBRAL_MIN_COBERTURA = 1.0     # cobertura WordNet mínima (%) para seleccionar
RANGO_TAMANO_SELECCION = (50, 450)  # tamaño aceptable para la ontología
TOP_GRUPOS_FIGURA = 15         # top de grupos en las figuras de barras
NUM_GRUPOS_SELECCION = 5       # grupos del criterio greedy (seccion 02)
GRUPOS_ADICIONALES: tuple[str, ...] = (
    "Afrontamientos y resistencias no violentas - cotidianas",
)  # union con el taller resuelto ajeno (ciclo del conflicto)

# Configuración gráfica global
plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 9,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
})
sns.set_theme(style="whitegrid")

print(f"RUTA_BASE: {RUTA_BASE}")
print("Constantes del proyecto cargadas:", len(FIGURAS), "rutas de figuras,", len(GRUPOS_SELECCIONADOS), "grupos")
