# %% [markdown]
"""# Sec 00 — Entorno, rutas y funciones de carga

Este bloque concentra al inicio del notebook:

1. Verificacion de dependencias (instalacion silenciosa solo si falta algo,
   pensado para la primera ejecucion en Google Colab).
2. **Todos los imports** de librerias usados por las secciones siguientes.
3. Semilla global (42) y estilo de las figuras matplotlib.
4. `buscar_raiz()`: localiza la raiz del proyecto TALLER-4 subiendo desde el
   directorio actual hasta encontrar la carpeta que contiene
   `data/corpus/segmentos_libros.jsonl`. Funciona local (kernel abierto desde
   `notebooks/`, desde la raiz o desde cualquier subcarpeta) y en Colab con la
   carpeta `TALLER-4` completa subida.
5. Funciones de carga reutilizables por las demas secciones:
   `cargar_lemas(corpus)`, `cargar_metadatos(corpus)` y `aplicar_parada(texto)`
   (parada EN CONSUMO, mismo criterio del experimento IR de la Fase 6).

**Como ejecutarlo**: localmente desde `TALLER-4/notebooks/` (o desde la raiz
del proyecto); en Google Colab subiendo la carpeta `TALLER-4` completa. Todas
las rutas se resuelven de forma relativa a la raiz detectada, nunca absolutas.
"""

# %%
"""Instalacion de dependencias (solo necesaria la primera vez en Colab)."""
import importlib.util
import subprocess
import sys


def instalar_si_falta(paquetes: list[str]) -> None:
    """Instala paquetes pip que no esten disponibles, en silencio.

    Args:
        paquetes: Lista de paquetes a verificar e instalar.
    """
    faltantes = [p for p in paquetes if importlib.util.find_spec(p) is None]
    if not faltantes:
        print("Dependencias ya instaladas:", ", ".join(paquetes))
        return
    print("Instalando:", ", ".join(faltantes))
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *faltantes])


instalar_si_falta(
    ["pandas", "numpy", "scikit-learn", "matplotlib", "spacy"]
)

# %%
"""Imports globales del notebook.

Se importan aqui una sola vez; las secciones posteriores los reutilizan sin
volver a importar. Varios se usan en secciones posteriores del notebook y no
en este archivo; por eso llevan `# noqa: F401` (ruff), igual que en el
TALLER-2.
"""
import importlib
import json
import os
import random
from collections import Counter  # noqa: F401  (seccion 02)
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # noqa: F401  (secciones 01-03)
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: F401

# Backend inline: las figuras (plt.show) quedan embebidas en el notebook.
try:
    get_ipython().run_line_magic("matplotlib", "inline")
except NameError:
    pass  # ejecucion como script: backend por defecto

# %%
"""Semilla global y estilo de figuras del notebook."""

SEMILLA = 42
random.seed(SEMILLA)
np.random.seed(SEMILLA)

plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 9,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
})
print(f"Semilla fija: {SEMILLA}")

# %%
"""Localizacion de la raiz del proyecto y rutas relativas a ella."""


def buscar_raiz() -> Path:
    """Directorio raiz del proyecto TALLER-4 (portatil: cualquier maquina).

    Estrategia de resolucion:
    1. Se parte de `os.getcwd()` y se sube por los padres hasta encontrar un
       directorio que contenga `data/corpus/segmentos_libros.jsonl`.
    2. Respaldo Colab: si no se encontro (kernel en /content), se prueban las
       ubicaciones tipicas de la carpeta TALLER-4 subida.

    Returns:
        Ruta absoluta a la raiz del proyecto (contiene data/, config/, ...).
    """
    candidato = Path(os.getcwd()).resolve()
    while candidato != candidato.parent:
        if (candidato / "data" / "corpus" / "segmentos_libros.jsonl").exists():
            return candidato
        candidato = candidato.parent
    for respaldo in (Path("/content/TALLER-4"), Path("/content/drive/MyDrive/TALLER-4")):
        if (respaldo / "data" / "corpus" / "segmentos_libros.jsonl").exists():
            return respaldo
    raise FileNotFoundError(
        "No se encontro la raiz del proyecto: falta data/corpus/segmentos_libros.jsonl "
        "en el arbol de directorios (sube la carpeta TALLER-4 completa en Colab)."
    )


RUTA_RAIZ = buscar_raiz()
RUTA_CORPUS = RUTA_RAIZ / "data" / "corpus"
RUTA_PREPROCESADO = RUTA_CORPUS / "preprocesado"
RUTA_CONFIG = RUTA_RAIZ / "config"
RUTA_SCRIPTS = RUTA_RAIZ / "scripts"
RUTA_EXPERIMENTOS = RUTA_RAIZ / "experimentos"
RUTA_RESULTADOS = RUTA_RAIZ / "resultados"
RUTA_EDA_RESULTADOS = RUTA_RESULTADOS / "eda"
RUTA_IMAGENES = RUTA_RAIZ / "imagenes"

print(f"RUTA_RAIZ: {RUTA_RAIZ}")

# %%
"""Carga de lemas, metadatos y parada en consumo (criterio de la Fase 6).

La parada y las correcciones aplican el MISMO criterio que
`experimentos/experimento_1_tfidf_base.py` (parada en consumo sobre lemas:
spacy.lang.es.STOP_WORDS + config/stopwords_habla.txt + ruido tipografico de
config/tfidf_config.json, y correcciones de config/lemas_correcciones.json).
Para no duplicar esa logica, se importa el modulo del experimento con
importlib (sys.path hacia experimentos/) y se delega en sus funciones
`construir_parada()`, `cargar_correcciones()` y `limpiar_linea()`. La primera
importacion tarda ~2-3 s (spacy.lang.es); las siguientes usan la cache de
sys.modules.
"""
def cargar_lemas(corpus: str) -> list[str]:
    """Lee el archivo de lemas de un corpus (1 documento por linea).

    Args:
        corpus: "libros" (coleccion de segmentos) o "entrevistas" (queries).

    Returns:
        Lista de lineas de lemas, en el orden alineado con *_docs.json.
    """
    ruta = RUTA_PREPROCESADO / f"{corpus}_lemas.txt"
    return [linea.rstrip("\n") for linea in ruta.open(encoding="utf-8")]


def cargar_metadatos(corpus: str) -> list[dict]:
    """Lee los metadatos de un corpus desde su *_docs.json.

    Args:
        corpus: "libros" o "entrevistas".

    Returns:
        Lista de dicts de metadatos (mismo orden que el archivo de lemas).
    """
    ruta = RUTA_PREPROCESADO / f"{corpus}_docs.json"
    return json.loads(ruta.read_text(encoding="utf-8"))["docs"]


def cargar_modulo_ir():
    """Importa experimentos/experimento_1_tfidf_base.py como modulo.

    La primera llamada registra el modulo en sys.modules; las siguientes son
    O(1). El modulo no ejecuta nada al importarse (solo constantes y
    definiciones), de modo que las secciones 00-03 comparten una unica fuente
    de verdad para parada, correcciones y pipeline del experimento.

    Returns:
        Modulo `experimento_1_tfidf_base` (con sus funciones publicas).
    """
    if "experimento_1_tfidf_base" in sys.modules:
        return sys.modules["experimento_1_tfidf_base"]
    if str(RUTA_EXPERIMENTOS) not in sys.path:
        sys.path.insert(0, str(RUTA_EXPERIMENTOS))
    return importlib.import_module("experimento_1_tfidf_base")


def aplicar_parada(linea_lemas: str) -> str:
    """Aplica correcciones y parada en consumo a una linea de lemas.

    Delega en `limpiar_linea()` del experimento (ver su docstring): primero la
    correccion de lemas anomalos (62 mapas) y despues la parada efectiva
    (611 tokens = STOP_WORDS de spacy + 51 muletillas + 42 ruido tipografico).
    Con esto el notebook usa exactamente el mismo criterio que el modelo IR.

    Args:
        linea_lemas: linea de un archivo *_lemas.txt (lemas separados por
            espacio).

    Returns:
        Linea limpia con los lemas de contenido separados por espacio.
    """
    exp = cargar_modulo_ir()
    parada = exp.construir_parada(incluir_muletillas=True)
    correcciones = exp.cargar_correcciones()
    return exp.limpiar_linea(linea_lemas, parada, correcciones)

# %%
"""Verificacion rapida de carga: lineas por corpus y ejemplo de parada."""

for corpus in ("libros", "entrevistas"):
    lemas = cargar_lemas(corpus)
    metas = cargar_metadatos(corpus)
    print(
        f"{corpus:12s} | {len(lemas)} documentos | "
        f"{sum(len(linea.split()) for linea in lemas):,} lemas fase 4 | "
        f"metadatos alineados: {len(metas) == len(lemas)}"
    )

demo = "atreveir el él pues cion y violencia sexual"
print("ejemplo parada en consumo:")
print("  antes :", demo)
print("  despues:", aplicar_parada(demo))
