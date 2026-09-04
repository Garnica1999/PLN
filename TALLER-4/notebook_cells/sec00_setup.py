# %% [markdown]
"""# Sec 00 — Entorno, rutas y funciones de carga

Este bloque concentra al inicio del notebook:

1. Verificacion de dependencias (instalacion silenciosa solo si falta algo,
   pensado para la primera ejecucion en Google Colab).
2. **Todos los imports** de librerias usados por las secciones siguientes
   (incluye `wordcloud` para las nubes de la Sec 02 e `IPython` para el
   helper `mostrar_png`).
3. Semilla global (42) y estilo de las figuras matplotlib.
4. `buscar_raiz()`: localiza la raiz del proyecto TALLER-4 subiendo desde el
   directorio actual hasta encontrar la carpeta que contiene
   `data/corpus/segmentos_libros.jsonl`. Funciona local (kernel abierto desde
   `notebooks/`, desde la raiz o desde cualquier subcarpeta) y en Colab con la
   carpeta `TALLER-4` completa subida.
5. Funciones de carga y de parada reutilizables por las demas secciones:
   `cargar_lemas(corpus)`, `cargar_metadatos(corpus)`, `cargar_correcciones()`
   (config/lemas_correcciones.json), `cargar_muletillas()`
   (config/stopwords_habla.txt), `cargar_ruido()`
   (config/tfidf_config.json, ruido tipografico de libros) y
   `aplicar_parada(linea)` (parada EN CONSUMO sobre lemas).

**Independencia del notebook**: todo el codigo de las secciones 00-03 es
autocontenido (no usa `importlib` ni `sys.path` hacia `scripts/` ni
`experimentos/`): la parada, las correcciones y el pipeline IR se replican
aqui leyendo solo los archivos versionados de `config/` y `data/`, con la
misma semantica que `scripts/03_eda.py` y
`experimentos/experimento_1_tfidf_base.py` (paridad verificada en la
ejecucion del notebook).

**Como ejecutarlo**: localmente desde `TALLER-4/notebooks/` (o desde la raiz
del proyecto); en Google Colab subiendo la carpeta `TALLER-4` completa. Todas
las rutas se resuelven de forma relativa a la raiz detectada, nunca absolutas.
"""


# %%
"""Instalacion de dependencias (solo necesaria la primera vez en Colab)."""
import subprocess
import sys


def instalar_si_falta(paquetes: list[str]) -> None:
    """Instala paquetes pip que no esten disponibles, en silencio.

    La verificacion se hace por import (try/except ImportError) con los
    nombres de modulo: la lista recibe modulos (p. ej. "sklearn", no
    "scikit-learn").

    Args:
        paquetes: Lista de modulos a verificar e instalar (nombres pip).
    """
    faltantes = []
    for paquete in paquetes:
        try:
            __import__(paquete)
        except ImportError:
            faltantes.append(paquete)
    if not faltantes:
        print("Dependencias ya instaladas:", ", ".join(paquetes))
        return
    print("Instalando:", ", ".join(faltantes))
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *faltantes])


instalar_si_falta(
    ["pandas", "numpy", "sklearn", "matplotlib", "spacy", "wordcloud"]
)


# %%
"""Imports globales del notebook.

Se importan aqui una sola vez; las secciones posteriores los reutilizan sin
volver a importar. Varios se usan en secciones posteriores del notebook y no
en este archivo; por eso llevan `# noqa: F401` (ruff), igual que en el
TALLER-2. IPython solo se usa dentro de `mostrar_png()` (celda de helpers),
con import local protegido para que el archivo tambien corra como script.
"""
import json
import os
import random
import re  # noqa: F401  (secciones 01-02)
import time  # noqa: F401  (secciones 02-03)
from collections import Counter  # noqa: F401  (seccion 02)
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # noqa: F401  (secciones 01-03)
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: F401  (seccion 03)
from spacy.lang.es import STOP_WORDS
from wordcloud import WordCloud  # noqa: F401  (seccion 02)

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
RUTA_RESULTADOS = RUTA_RAIZ / "resultados"
RUTA_EDA_RESULTADOS = RUTA_RESULTADOS / "eda"
RUTA_IMAGENES = RUTA_RAIZ / "imagenes"

print(f"RUTA_RAIZ: {RUTA_RAIZ}")


# %%
"""Carga de lemas, metadatos y parada en consumo (criterio de la Fase 6).

Las funciones replican localmente (sin importar modulos de `scripts/` ni
`experimentos/`) el criterio congelado en config/:

- Parada EN CONSUMO sobre lemas = `spacy.lang.es.STOP_WORDS` +
  `config/stopwords_habla.txt` (51 muletillas) +, solo para el pipeline IR
  (Sec 03), el ruido tipografico de libros de `config/tfidf_config.json`
  (42 tokens; lista verificada identica a RUIDO_LIBROS de 03_eda.py).
- Correcciones de lemas anomalos del lematizador en consumo:
  `config/lemas_correcciones.json` (62 mapas, p. ej. atreveir -> atrever).
- Anos 1900-2025 y categorias canonicas: no se tocan (la Fase 4 ya los
  dejo como lemas; la correccion no los mapea y la parada no los contiene).

La parada del EDA (Sec 02) NO incluye el ruido tipografico, igual que
scripts/03_eda.py (paridad numerica con resultados/eda/); el ruido solo se
excluye del analisis comparativo (solapamiento y log-odds), como en la
etapa 6 del EDA de produccion.
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


def cargar_correcciones() -> dict[str, str]:
    """Carga el mapa de correcciones de lemas de config/lemas_correcciones.json.

    Returns:
        Mapa lema anomalo -> lema correcto (claves con la forma del corpus).
    """
    ruta = RUTA_CONFIG / "lemas_correcciones.json"
    datos = json.loads(ruta.read_text(encoding="utf-8"))
    correcciones = datos.get("correcciones", {})
    return {str(k): str(v) for k, v in correcciones.items()}


def cargar_muletillas() -> set[str]:
    """Carga los tokens de config/stopwords_habla.txt (ignora '#' y vacias).

    Returns:
        Conjunto de muletillas congeladas en el EDA (51 tokens).
    """
    ruta = RUTA_CONFIG / "stopwords_habla.txt"
    return {
        linea.strip()
        for linea in ruta.read_text(encoding="utf-8").splitlines()
        if linea.strip() and not linea.startswith("#")
    }


def cargar_ruido() -> set[str]:
    """Carga el ruido tipografico de libros de config/tfidf_config.json.

    Los 42 tokens (fragmentos de palabras partidas por el guion del PDF y
    ruido bibliografico como 'ibid.') se leen de la lista congelada
    `ruido_tipografico_libros_excluido_en_eda` (verificada identica a
    RUIDO_LIBROS de scripts/03_eda.py). El experimento IR los trata como
    parada; el EDA solo los excluye del analisis comparativo.

    Returns:
        Conjunto de tokens de ruido a excluir del vocabulario.
    """
    ruta = RUTA_CONFIG / "tfidf_config.json"
    datos = json.loads(ruta.read_text(encoding="utf-8"))
    ruido = datos.get("ruido_tipografico_libros_excluido_en_eda", [])
    return {str(token) for token in ruido}


def construir_parada(incluir_muletillas: bool = True, incluir_ruido: bool = False) -> set[str]:
    """Construye la parada en consumo sobre lemas para un consumo dado.

    Parada base = spacy.lang.es.STOP_WORDS; opcionalmente se anaden las
    muletillas de config/stopwords_habla.txt y el ruido tipografico de
    config/tfidf_config.json (este ultimo solo en el pipeline IR de la
    Sec 03, mismo criterio del experimento base de la Fase 6).

    Args:
        incluir_muletillas: False para la variante B del IR (parada solo
            spaCy + ruido).
        incluir_ruido: True para el pipeline IR (criterio del experimento);
            False para el EDA (paridad con scripts/03_eda.py).

    Returns:
        Conjunto de tokens que se eliminan del texto lematizado.
    """
    parada = set(STOP_WORDS)
    if incluir_muletillas:
        parada |= cargar_muletillas()
    if incluir_ruido:
        parada |= cargar_ruido()
    return parada


# Paradas efectivas del notebook, calculadas una sola vez (deterministas).
CORRECCIONES = cargar_correcciones()
MULETILLAS = cargar_muletillas()
RUIDO_LIBROS = cargar_ruido()
PARADA_EDA = construir_parada(incluir_muletillas=True, incluir_ruido=False)
PARADA_IR = construir_parada(incluir_muletillas=True, incluir_ruido=True)


def aplicar_parada(linea_lemas: str, parada: set[str] | None = None) -> str:
    """Aplica correcciones y parada en consumo a una linea de lemas.

    Por token: primero la correccion (lema anomalo -> lema correcto,
    config/lemas_correcciones.json) y, sobre el resultado, la parada. Anos
    1900-2025 y categorias canonicas no se ven afectados. Con esto el
    notebook usa exactamente el mismo criterio que el EDA de la Fase 5
    (parada sin ruido tipografico) y que el modelo IR cuando se le pasa
    PARADA_IR.

    Args:
        linea_lemas: linea de un archivo *_lemas.txt (lemas separados por
            espacio).
        parada: conjunto de tokens a eliminar; si es None se usa PARADA_EDA.

    Returns:
        Linea limpia con los lemas de contenido separados por espacio.
    """
    if parada is None:
        parada = PARADA_EDA
    salida: list[str] = []
    for token in linea_lemas.split():
        if CORRECCIONES:
            token = CORRECCIONES.get(token, token)
        if token not in parada:
            salida.append(token)
    return " ".join(salida)


def mostrar_png(nombre: str, ancho: int = 900) -> None:
    """Muestra un PNG de imagenes/ embebido en el notebook (ruta absoluta).

    Las figuras pesadas del EDA de produccion (chars y paginas de
    entrevistas, near-duplicates) requieren archivos no versionados (>100 MB)
    o el pipeline completo de scripts/03_eda.py; el notebook las muestra con
    este helper desde `imagenes/` (versionado). Si el archivo no existe se
    imprime un aviso claro en lugar de fallar.

    Args:
        nombre: nombre del archivo dentro de imagenes/ (p. ej.
            "eda_entrevistas_hist_chars.png").
        ancho: ancho en pixeles del PNG embebido.
    """
    ruta = RUTA_IMAGENES / nombre
    if not ruta.exists():
        print(f"[aviso] No existe la figura oficial {ruta.name} en imagenes/;")
        print("        regenere el EDA de produccion con scripts/03_eda.py.")
        return
    try:
        from IPython.display import Image, display
    except ImportError:
        # Respaldo sin IPython (ejecucion como script): mostrar con matplotlib.
        imagen = plt.imread(ruta)
        plt.figure(figsize=(10, imagen.shape[0] / imagen.shape[1] * 10))
        plt.imshow(imagen)
        plt.axis("off")
        plt.show()
        return
    display(Image(filename=str(ruta), width=ancho))


print(
    f"Configuracion cargada: {len(CORRECCIONES)} correcciones | "
    f"{len(MULETILLAS)} muletillas | {len(RUIDO_LIBROS)} ruido | "
    f"parada EDA {len(PARADA_EDA)} tokens | parada IR {len(PARADA_IR)} tokens"
)


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
print("ejemplo parada en consumo (correcciones -> parada, criterio EDA):")
print("  antes :", demo)
print("  despues:", aplicar_parada(demo))
print("  con parada IR (incluye ruido tipografico):", aplicar_parada(demo, PARADA_IR))

