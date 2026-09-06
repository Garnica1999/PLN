# %% [markdown]
"""# Sec 00 — Entorno, rutas y funciones de carga

Este bloque concentra al inicio del notebook:

1. Verificacion de dependencias (instalacion silenciosa solo si falta algo,
   pensado para la primera ejecucion en Google Colab).
2. **Todos los imports** de librerias usados por las secciones siguientes
   (incluye `wordcloud` para las nubes de la Sec 02 e `IPython` para el
   helper `mostrar_png`).
3. Semilla global (42) y estilo de las figuras matplotlib.
4. Configuracion del entorno: `AUTO_DESCARGAR_REPO` (False por defecto;
   activala en Colab si el repositorio aun no esta en /content).
5. `buscar_raiz()`: localiza la raiz del proyecto TALLER-4 (el directorio que
   contiene `data/corpus/segmentos_libros.jsonl`) con una estrategia robusta:
   (1) variable de entorno `TALLER4_ROOT`, (2) ancestros desde el directorio
   actual, (3) subcarpetas de hasta 3 niveles bajo el cwd y bajo cada ancestro
   (cubre `/content/PLN-main/TALLER-4` cuando se descomprime el zip del
   repositorio en /content), (4) rutas tipicas de Colab
   (`/content/TALLER-4`, `/content/drive/MyDrive/TALLER-4`,
   `/content/*/TALLER-4`) y, si `AUTO_DESCARGAR_REPO=True`, (5) descarga el
   zip del repositorio y reintenta. Si nada funciona, lanza un error claro con
   las opciones (NUNCA cae a la raiz del filesystem).
6. **Diagnostico de prerrequisitos**: checklist con marca por archivo
   (existencia y tamano en MB) de los archivos que usan las secciones 01-03.
7. Funciones de carga y de parada reutilizables por las demas secciones:
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
ejecucion del notebook). La seccion de regeneracion (Sec 0b) es la unica que
trae codigo de `scripts/` (expandido como celdas de codigo normales por el
ensamblador, fuente unica en build) y esa seccion solo se ejecuta cuando se
activan sus flags; por defecto esta inerte (sus celdas solo definen).

**Como ejecutarlo en Colab**: (A) sube la carpeta `TALLER-4` completa a
/content, o descomprime en /content el zip del repositorio (queda en
`/content/PLN-main/TALLER-4` y tambien se detecta); (B) abre el notebook con
"Open in Colab" y pon `AUTO_DESCARGAR_REPO = True` en la celda de
configuracion (descarga el repositorio automaticamente); (C) ten la carpeta
en Google Drive montado (`/content/drive/MyDrive/TALLER-4`) o define la
variable de entorno `TALLER4_ROOT`. Localmente basta abrir el kernel desde
`TALLER-4/notebooks/`, desde la raiz o desde cualquier subcarpeta. Todas las
rutas se resuelven de forma relativa a la raiz detectada, nunca absolutas.
"""


# %%
"""Instalacion de dependencias (solo necesaria la primera vez en Colab)."""
import subprocess
import sys


def instalar_si_falta(paquetes: list[str]) -> None:
    """Instala paquetes pip que no esten disponibles, en silencio.

    La verificacion se hace por import (try/except ImportError) con los
    nombres de modulo: la lista recibe modulos (p. ej. "sklearn", no
    "scikit-learn"; "pymupdf" verifica e instala el paquete PyMuPDF, que
    provee el modulo `fitz`; "gdown" descarga archivos del Drive del taller
    en la seccion de regeneracion).

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
    ["pandas", "numpy", "sklearn", "matplotlib", "spacy", "wordcloud",
     "gdown", "pymupdf"]
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
"""Configuracion del entorno (flags editables por el usuario).

`AUTO_DESCARGAR_REPO` controla el paso 5 de `buscar_raiz()`: si la raiz no se
encuentra en el arbol, se descarga el zip del repositorio
(https://codeload.github.com/Garnica1999/PLN/zip/refs/heads/main) con urllib,
se descomprime bajo /content (o el directorio actual) y se reintenta la
busqueda. En False (default) no se descarga nada y la celda de resolucion
lanza un error claro con las opciones para subir los datos.
"""
AUTO_DESCARGAR_REPO = False
print(f"AUTO_DESCARGAR_REPO = {AUTO_DESCARGAR_REPO}")


# %%
"""Funciones de localizacion de la raiz del proyecto (no se ejecutan aqui).

`buscar_raiz()` busca el marcador `data/corpus/segmentos_libros.jsonl` con
este orden: (1) variable de entorno `TALLER4_ROOT`; (2) ancestros desde
`os.getcwd()`; (3) subcarpetas de hasta 3 niveles bajo el cwd y bajo cada
ancestro (os.walk podado: omite directorios ocultos y los de entorno/git);
(4) rutas tipicas de Colab; (5) si `AUTO_DESCARGAR_REPO` esta activo,
descarga y descomprime el repositorio y reintenta (2)-(4). Nunca devuelve la
raiz del filesystem: si no encuentra, `raise FileNotFoundError` con el
marcador buscado, los lugares inspeccionados (resumidos) y 3 opciones.
"""

MARCADOR_RAIZ = Path("data") / "corpus" / "segmentos_libros.jsonl"
_MAX_PROFUNDIDAD_WALK = 3
_URL_REPO_ZIP = "https://codeload.github.com/Garnica1999/PLN/zip/refs/heads/main"

# Directorios que nunca se exploran en los walks (ocultos + entornos).
_DIRS_EXCLUIDOS_WALK = frozenset({
    ".git", "drive", "sample_data", "venv", "node_modules",
})
# Podas adicionales solo cuando el walk parte de la raiz del filesystem.
_DIRS_SISTEMA_RAIZ = frozenset({
    "bin", "boot", "dev", "etc", "home", "lib", "lib64", "lost+found",
    "media", "mnt", "opt", "proc", "root", "run", "sbin", "snap", "srv",
    "sys", "tmp", "usr", "var",
})


def _cumple_marcador(directorio: Path) -> bool:
    """True si el directorio contiene el marcador de la raiz del proyecto.

    Args:
        directorio: Candidato a raiz.

    Returns:
        True si existe directorio/data/corpus/segmentos_libros.jsonl.
    """
    return (directorio / MARCADOR_RAIZ).exists()


def _candidatos_por_walk(base: Path, max_profundidad: int):
    """Genera los directorios de base y sus subcarpetas hasta la profundidad.

    Args:
        base: Directorio donde comienza el recorrido (incluido).
        max_profundidad: Niveles maximos de subcarpetas visitadas.

    Yields:
        Path de cada directorio visitado (orden por niveles).
    """
    base_resuelta = Path(base).resolve()
    profundidad_base = len(base_resuelta.parts)
    es_raiz_fs = base_resuelta == Path(base_resuelta.anchor)
    for dir_actual, subdirs, _ in os.walk(base_resuelta):
        nivel = len(Path(dir_actual).parts) - profundidad_base
        if nivel >= max_profundidad:
            subdirs[:] = []
        else:
            subdirs[:] = sorted(
                d
                for d in subdirs
                if not d.startswith(".")
                and d not in _DIRS_EXCLUIDOS_WALK
                and not (es_raiz_fs and d in _DIRS_SISTEMA_RAIZ)
            )
        yield Path(dir_actual)


def _descargar_repo_zip(destino: Path) -> Path:
    """Descarga y descomprime el zip del repositorio (rama main) en destino.

    Args:
        destino: Carpeta donde se escribe PLN-main.zip y se extrae el arbol.

    Returns:
        Path de destino (contiene la carpeta PLN-main/ del repositorio).

    Raises:
        FileNotFoundError: Si la descarga o la descompresion fallan.
    """
    import shutil
    import urllib.request
    import zipfile

    destino.mkdir(parents=True, exist_ok=True)
    ruta_zip = destino / "PLN-main.zip"
    print(f"[auto-descarga] Descargando {_URL_REPO_ZIP} ...")
    try:
        with (
            urllib.request.urlopen(_URL_REPO_ZIP, timeout=180) as respuesta,
            ruta_zip.open("wb") as salida,
        ):
            shutil.copyfileobj(respuesta, salida)
    except Exception as exc:
        raise FileNotFoundError(
            "No se pudo descargar el repositorio (AUTO_DESCARGAR_REPO=True): "
            f"{exc}. Descargalo manualmente desde GitHub (Code -> Download "
            "ZIP) y descomprimelo en /content."
        ) from exc
    with zipfile.ZipFile(ruta_zip) as archivo_zip:
        archivo_zip.extractall(destino)
    print(f"[auto-descarga] Repositorio extraido en {destino}")
    return destino


def buscar_raiz(auto_descarga: bool | None = None) -> Path:
    """Directorio raiz del proyecto TALLER-4 (portatil: cualquier maquina).

    Estrategia de resolucion (ver docstring de la celda): TALLER4_ROOT,
    ancestros desde el cwd, subcarpetas de hasta 3 niveles bajo el cwd y bajo
    cada ancestro, rutas tipicas de Colab y, opcionalmente, auto-descarga del
    repositorio con reintento. El marcador es
    `data/corpus/segmentos_libros.jsonl`.

    Args:
        auto_descarga: Override del flag global AUTO_DESCARGAR_REPO (celda de
            configuracion). None = usar el flag global (default False).

    Returns:
        Ruta absoluta a la raiz del proyecto (contiene data/, config/, ...).

    Raises:
        FileNotFoundError: Si el marcador no aparece en ningun lugar, con el
            detalle de lo buscado y las opciones para resolverlo.
    """
    if auto_descarga is None:
        auto_descarga = bool(globals().get("AUTO_DESCARGAR_REPO", False))
    cwd = Path(os.getcwd()).resolve()
    lugares: list[str] = []

    def _ancestros_y_walks() -> Path | None:
        """Pasos 2-3: ancestros y subcarpetas (<= 3 niveles) de cada uno."""
        ancestros = [cwd, *cwd.parents]
        for ancestro in ancestros:
            lugares.append(str(ancestro))
            if _cumple_marcador(ancestro):
                return ancestro
        for ancestro in ancestros:
            lugares.append(f"subcarpetas (<=3 niveles) bajo {ancestro}")
            for candidato in _candidatos_por_walk(ancestro, _MAX_PROFUNDIDAD_WALK):
                if _cumple_marcador(candidato):
                    return candidato
        return None

    def _rutas_tipicas() -> Path | None:
        """Paso 4: rutas tipicas de Colab (fijas y globs acotados)."""
        import glob as _glob

        for candidata in (Path("/content/TALLER-4"), Path("/content/drive/MyDrive/TALLER-4")):
            lugares.append(f"ruta tipica {candidata}")
            if _cumple_marcador(candidata):
                return candidata
        for patron in ("/content/*/TALLER-4", "/content/*/*/TALLER-4"):
            for coincidencia in sorted(_glob.glob(patron)):
                lugares.append(f"ruta tipica {coincidencia}")
                candidata = Path(coincidencia)
                if _cumple_marcador(candidata):
                    return candidata
        return None

    env_root = os.environ.get("TALLER4_ROOT", "").strip()
    if env_root:
        lugares.append(f"TALLER4_ROOT={env_root}")
        candidato = Path(env_root).resolve()
        if _cumple_marcador(candidato):
            print(f"[raiz] Resuelta por TALLER4_ROOT: {candidato}")
            return candidato

    encontrada = _ancestros_y_walks() or _rutas_tipicas()
    if encontrada is not None:
        return encontrada

    if auto_descarga:
        base = Path("/content") if Path("/content").exists() else cwd
        destino = _descargar_repo_zip(base)
        lugares.append(f"auto-descarga extraida en {destino}")
        encontrada = _ancestros_y_walks() or _rutas_tipicas()
        if encontrada is not None:
            print(f"[raiz] Resuelta tras la auto-descarga: {encontrada}")
            return encontrada

    resumen = "; ".join(dict.fromkeys(lugares))
    if len(resumen) > 600:
        resumen = resumen[:600] + " ..."
    raise FileNotFoundError(
        "No se encontro la raiz del proyecto TALLER-4: falta el marcador "
        "data/corpus/segmentos_libros.jsonl en el arbol de directorios.\n"
        f"Lugares inspeccionados: {resumen}.\n"
        "Opciones:\n"
        "  A) Sube la carpeta TALLER-4 completa a /content (o descomprime en "
        "/content el zip del repositorio: queda en /content/PLN-main/TALLER-4 "
        "y se detecta igual).\n"
        "  B) Activa la auto-descarga: pon AUTO_DESCARGAR_REPO = True en la "
        "celda de configuracion y vuelve a ejecutar esta celda.\n"
        "  C) Ten la carpeta en Google Drive montado en "
        "/content/drive/MyDrive/TALLER-4, o define la variable de entorno "
        "TALLER4_ROOT con la ruta exacta de la carpeta TALLER-4.\n"
        "Depuracion: ejecuta '!ls /content' para ver el contenido de /content "
        "y confirmar donde esta la carpeta."
    )


# %%
"""Resolucion de la raiz del proyecto y rutas relativas a ella."""

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
"""Diagnostico de prerrequisitos: checklist de archivos criticos (marca y
tamano en MB) y estado de los archivos opcionales no versionados.

Los criticos estan versionados en git y viajan en el zip del repositorio; si
falta alguno, el mensaje final indica como obtenerlos. Los opcionales
(entrevistas_normalizadas.jsonl, entrevistas_tokens.txt y el JSON crudo de
entrevistas) pesan mas de 100 MB y NO se versionan: las secciones 01-03 usan
respaldo estatico cuando no estan, o se regeneran con la seccion de
regeneracion de este notebook (Sec 0b) descargando las fuentes del Drive del
taller.
"""

_MARCA_OK = "\u2713"      # checklist: archivo presente
_MARCA_FALTA = "\u2717"   # checklist: archivo faltante


def _tamano_mb(ruta: Path) -> float:
    """Tamano de un archivo en MB (0.0 si no existe).

    Args:
        ruta: Ruta del archivo.

    Returns:
        Tamano en MB redondeado a 1 decimal.
    """
    if not ruta.exists():
        return 0.0
    return ruta.stat().st_size / (1024 * 1024)


def diagnostico_prerrequisitos() -> dict[str, list[tuple[str, bool]]]:
    """Recorre el checklist de archivos que requieren las secciones 01-03.

    Returns:
        Dict con "criticos" y "opcionales": listas de (nombre, presente).
        Ademas imprime el estado detallado con marcas y tamanos.
    """
    criticos = [
        ("segmentos_libros.jsonl", RUTA_CORPUS / "segmentos_libros.jsonl"),
        ("libros_lemas.txt", RUTA_PREPROCESADO / "libros_lemas.txt"),
        ("entrevistas_lemas.txt", RUTA_PREPROCESADO / "entrevistas_lemas.txt"),
        ("libros_docs.json", RUTA_PREPROCESADO / "libros_docs.json"),
        ("entrevistas_docs.json", RUTA_PREPROCESADO / "entrevistas_docs.json"),
        ("stopwords_habla.txt", RUTA_CONFIG / "stopwords_habla.txt"),
        ("lemas_correcciones.json", RUTA_CONFIG / "lemas_correcciones.json"),
        ("tfidf_config.json", RUTA_CONFIG / "tfidf_config.json"),
        ("ranking_top5.csv", RUTA_RESULTADOS / "ranking_top5.csv"),
        ("ranking_top5_A.csv", RUTA_RESULTADOS / "ranking_top5_A.csv"),
        ("ranking_top5_B.csv", RUTA_RESULTADOS / "ranking_top5_B.csv"),
        ("ranking_top5_C.csv", RUTA_RESULTADOS / "ranking_top5_C.csv"),
        ("extraccion_libros_report.json", RUTA_RESULTADOS / "extraccion_libros_report.json"),
        (
            "inventario_patrones_entrevistas.json",
            RUTA_RESULTADOS / "preprocesado" / "inventario_patrones_entrevistas.json",
        ),
        ("evaluacion_muestra_etiquetada.csv", RUTA_RESULTADOS / "evaluacion_muestra_etiquetada.csv"),
        ("evaluacion_metricas.json", RUTA_RESULTADOS / "evaluacion_metricas.json"),
        ("distintivos_logodds.csv", RUTA_EDA_RESULTADOS / "distintivos_logodds.csv"),
        ("eda_entrevistas_hist_chars.png", RUTA_IMAGENES / "eda_entrevistas_hist_chars.png"),
        ("eda_entrevistas_hist_paginas.png", RUTA_IMAGENES / "eda_entrevistas_hist_paginas.png"),
        ("eda_near_duplicates_entrevistas.png", RUTA_IMAGENES / "eda_near_duplicates_entrevistas.png"),
    ]
    opcionales = [
        (
            "entrevistas_normalizadas.jsonl",
            RUTA_CORPUS / "entrevistas_normalizadas.jsonl",
            "~194 MB",
        ),
        (
            "entrevistas_tokens.txt",
            RUTA_PREPROCESADO / "entrevistas_tokens.txt",
            "~191 MB",
        ),
        (
            "entrevistas_all_2023-03-21_14-24_05.json (crudo)",
            RUTA_RAIZ / "data" / "raw" / "entrevistas_all_2023-03-21_14-24_05.json",
            "~231 MB",
        ),
    ]

    print(f"RUTA_RAIZ: {RUTA_RAIZ}")
    print("\nChecklist de prerrequisitos (archivos versionados del repo):")
    resultado_criticos: list[tuple[str, bool]] = []
    for nombre, ruta in criticos:
        presente = ruta.exists()
        resultado_criticos.append((nombre, presente))
        marca = _MARCA_OK if presente else _MARCA_FALTA
        print(f"  {marca} {nombre:<42} {_tamano_mb(ruta):>9.1f} MB")

    n_presentes = sum(1 for _, presente in resultado_criticos if presente)
    n_total = len(resultado_criticos)
    print(f"\nChecklist critico: {n_presentes}/{n_total} presente(s)")
    if n_presentes < n_total:
        print(
            "Faltan archivos criticos: el notebook necesita el contenido "
            "versionado del repositorio. Descarga el zip (GitHub -> Code -> "
            "Download ZIP) y descomprimelo en /content (deja la carpeta en "
            "/content/PLN-main/TALLER-4), o sube la carpeta TALLER-4 completa."
        )

    print(
        "Archivos opcionales (no versionados, con respaldo estatico o "
        "regenerables):"
    )
    resultado_opcionales: list[tuple[str, bool]] = []
    for nombre, ruta, esperado in opcionales:
        presente = ruta.exists()
        resultado_opcionales.append((nombre, presente))
        marca = _MARCA_OK if presente else _MARCA_FALTA
        estado = f"presente ({_tamano_mb(ruta):.0f} MB)" if presente else "no presente"
        print(f"  {marca} {nombre:<48} {estado} (esperado {esperado})")
    print(
        "  Si faltan los opcionales, las secciones 01-03 usan su respaldo "
        "estatico o se regeneran con la seccion de regeneracion (Sec 0b, "
        "flags en False por defecto) descargando las fuentes del Drive del "
        "taller."
    )
    return {"criticos": resultado_criticos, "opcionales": resultado_opcionales}


_resultado_diagnostico = diagnostico_prerrequisitos()


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
