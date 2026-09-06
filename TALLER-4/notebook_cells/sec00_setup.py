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
   contiene el cache `data/corpus/segmentos_libros.jsonl` o CUALQUIERA de los
   3 insumos en `data/raw/`: dataset de entrevistas, libros CEV o tesauro)
   con una estrategia robusta:
   (1) variable de entorno `TALLER4_ROOT`, (2) ancestros desde el directorio
   actual, (3) subcarpetas de hasta 3 niveles bajo el cwd y bajo cada ancestro
   (cubre `/content/PLN-main/TALLER-4` cuando se descomprime el zip del
   repositorio en /content), (4) rutas tipicas de Colab
   (`/content/TALLER-4`, `/content/drive/MyDrive/TALLER-4`,
   `/content/*/TALLER-4`) y, si `AUTO_DESCARGAR_REPO=True`, (5) descarga el
   zip del repositorio y reintenta. Si nada funciona, lanza un error claro con
   las opciones (NUNCA cae a la raiz del filesystem).
6. **Configuracion curada EMBEBIDA** (independencia total): la config del
   pipeline (muletillas del habla, correcciones de lemas, ruido tipografico
   y parametros TF-IDF) viaja como constantes Python literales
   (`STOPWORDS_HABLA_EMBEDIDAS`, `CORRECCIONES_EMBEDIDAS`, `RUIDO_EMBEDIDO`
   y `TFIDF_PARAMS_EMBEDIDOS`), copias congeladas generadas desde los
   archivos de `config/`. Los archivos de `config/` siguen siendo la fuente
   para los scripts de produccion (v4) y, si existen, la funcion
   `verificar_consistencia_config()` los compara opcionalmente con las
   copias embebidas (no bloqueante).
7. **Diagnostico de prerrequisitos**: checklist de los caches derivados
   REQUERIDOS por las secciones 01-03 (se generan automaticamente en la
   seccion Regeneracion a partir de los 3 insumos permitidos: dataset de
   entrevistas, libros y tesauro) y de los OPCIONALES (config/ embebida,
   resultados/ e imagenes/: nunca requeridos).
8. Funciones de carga y de parada reutilizables por las demas secciones:
   `cargar_lemas(corpus)`, `cargar_metadatos(corpus)`, `cargar_correcciones()`
   (usa CORRECCIONES_EMBEDIDAS), `cargar_muletillas()` (usa
   STOPWORDS_HABLA_EMBEDIDAS), `cargar_ruido()` (usa RUIDO_EMBEDIDO),
   `construir_parada()` y `aplicar_parada(linea)` (parada EN CONSUMO sobre
   lemas). `mostrar_png()` muestra las figuras de imagenes/ con fallback
   suave (si no existen, aviso claro y sigue).

**Independencia del notebook**: todo el codigo de las secciones 00-03 es
autocontenido (no usa `importlib` ni `sys.path` hacia `scripts/` ni
`experimentos/`) y NO depende de `config/`, `resultados/` ni `imagenes/`:
la parada, las correcciones, el ruido y la config TF-IDF se replican con las
copias congeladas embebidas (misma semantica que `scripts/03_eda.py` y
`experimentos/experimento_1_tfidf_base.py`, paridad verificada en la
ejecucion del notebook), y los caches derivados de `data/corpus/`
(segmentos, lemas, docs, normalizadas) se GENERAN automaticamente en la
seccion Regeneracion (Sec 0b) desde los 3 insumos permitidos si faltan: el
dataset de entrevistas (descarga manual del Drive del taller), los libros
(descarga manual) y el tesauro. La seccion de regeneracion es la unica que
trae codigo de `scripts/` (expandido como celdas de codigo normales por el
ensamblador, fuente unica en build) y corre bajo demanda o con su flag
automatico cuando faltan caches derivados.

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

`buscar_raiz()` reconoce la raiz por el cache derivado
`data/corpus/segmentos_libros.jsonl` o por CUALQUIERA de los 3 insumos
permitidos en `data/raw/` (dataset de entrevistas, libros CEV o tesauro),
de modo que la raiz se resuelve ANTES de generar los caches derivados. La
busqueda sigue este orden: (1) variable de entorno `TALLER4_ROOT`; (2)
ancestros desde
`os.getcwd()`; (3) subcarpetas de hasta 3 niveles bajo el cwd y bajo cada
ancestro (os.walk podado: omite directorios ocultos y los de entorno/git);
(4) rutas tipicas de Colab; (5) si `AUTO_DESCARGAR_REPO` esta activo,
descarga y descomprime el repositorio y reintenta (2)-(4). Nunca devuelve la
raiz del filesystem: si no encuentra, `raise FileNotFoundError` con lo
buscado (cache o insumos), los lugares inspeccionados (resumidos) y 3
opciones.
"""

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


def _es_raiz(candidato) -> bool:
    """True si el candidato es la raiz del proyecto TALLER-4.

    Un directorio es la raiz si contiene el cache derivado
    data/corpus/segmentos_libros.jsonl o CUALQUIERA de los 3 insumos
    permitidos en data/raw/ (dataset de entrevistas, libros CEV o
    tesauro): asi la raiz se resuelve ANTES de generar los caches
    derivados.

    Args:
        candidato: Directorio candidato a raiz del proyecto.

    Returns:
        True si contiene el cache derivado o algun insumo de
        data/raw/.
    """
    return (
        (candidato / "data" / "corpus" / "segmentos_libros.jsonl").exists()
        or (candidato / "data" / "raw" / "entrevistas_all_2023-03-21_14-24_05.json").exists()
        or (candidato / "data" / "raw" / "Resource-tesauro.rdf").exists()
        or bool(list((candidato / "data" / "raw").glob("books-*.zip")))
        or bool(list((candidato / "data" / "raw" / "books").glob("*.pdf")))
    )


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
    repositorio con reintento. La raiz se reconoce por el cache derivado
    `data/corpus/segmentos_libros.jsonl` o por CUALQUIERA de los 3 insumos
    en `data/raw/` (dataset de entrevistas, libros CEV o tesauro), de modo
    que se resuelve antes de generar los caches.

    Args:
        auto_descarga: Override del flag global AUTO_DESCARGAR_REPO (celda de
            configuracion). None = usar el flag global (default False).

    Returns:
        Ruta absoluta a la raiz del proyecto (contiene data/, config/, ...).

    Raises:
        FileNotFoundError: Si no aparece el cache derivado ni ningun insumo,
            con el detalle de lo buscado y las opciones para resolverlo.
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
            if _es_raiz(ancestro):
                return ancestro
        for ancestro in ancestros:
            lugares.append(f"subcarpetas (<=3 niveles) bajo {ancestro}")
            for candidato in _candidatos_por_walk(ancestro, _MAX_PROFUNDIDAD_WALK):
                if _es_raiz(candidato):
                    return candidato
        return None

    def _rutas_tipicas() -> Path | None:
        """Paso 4: rutas tipicas de Colab (fijas y globs acotados)."""
        import glob as _glob

        for candidata in (Path("/content/TALLER-4"), Path("/content/drive/MyDrive/TALLER-4")):
            lugares.append(f"ruta tipica {candidata}")
            if _es_raiz(candidata):
                return candidata
        for patron in ("/content/*/TALLER-4", "/content/*/*/TALLER-4"):
            for coincidencia in sorted(_glob.glob(patron)):
                lugares.append(f"ruta tipica {coincidencia}")
                candidata = Path(coincidencia)
                if _es_raiz(candidata):
                    return candidata
        return None

    env_root = os.environ.get("TALLER4_ROOT", "").strip()
    if env_root:
        lugares.append(f"TALLER4_ROOT={env_root}")
        candidato = Path(env_root).resolve()
        if _es_raiz(candidato):
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
        "No se encontro la raiz del proyecto TALLER-4: se busca la carpeta "
        "que contenga data/corpus/segmentos_libros.jsonl o alguno de los 3 "
        "insumos en data/raw/ (dataset de entrevistas, libros CEV o "
        "tesauro) en el arbol de directorios.\n"
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
"""Configuracion curada del pipeline EMBEBIDA como constantes literales.

Copias congeladas embebidas para independencia total: el notebook NO lee
`config/` (ni `resultados/` ni `imagenes/`) para funcionar. Los archivos de
`config/` son la fuente para los scripts de produccion (v4) y, si existen en
la raiz detectada, `verificar_consistencia_config()` (celda de carga) los
compara opcionalmente con estas copias y avisa si alguna difiere.
"""

STOPWORDS_HABLA_EMBEDIDAS: frozenset[str] = frozenset({
    "eh", "ehh", "digamos", "listo",
    "aja", "aj\u00e1", "ah", "ahh",
    "mmm", "mm", "umm", "vea",
    "ve", "jum", "uy", "ujum",
    "mire", "mira", "mir\u00e1", "osea",
    "\u00f3seo", "pa", "all\u00e1", "ac\u00e1",
    "ok", "okey", "okay", "uhm",
    "mmjm", "mjmm", "eee", "hmm",
    "ehhh", "ehm", "bla", "al\u00f3",
    "mjm", "ujumm", "etc", "inf",
    "interrup", "jumm", "hum", "mmmm",
    "\u00f3sea", "se\u00f1o", "unuma", "em",
    "ser", "estar", "haber",
})

CORRECCIONES_EMBEDIDAS: dict[str, str] = {
    "atreve\u00edr": "atrever",
    "\u00edr": "ir",
    "de\u00edr": "dar",
    "dije\u00edr": "decir",
    "traje\u00edr": "traer",
    "sigar": "seguir",
    "tenia": "tener",
    "teniar": "tener",
    "entonce": "entonces",
    "ento": "entonces",
    "ent\u00f3n": "entonces",
    "creser": "creer",
    "estabar": "estar",
    "estuviste": "estar",
    "has": "haber",
    "habiar": "haber",
    "indigena": "ind\u00edgena",
    "vivistir": "vivir",
    "viv\u00edas": "vivir",
    "podriar": "poder",
    "recuerdas": "recordar",
    "naciste": "nacer",
    "saliste": "salir",
    "cu\u00e9ntanos": "contar",
    "toes": "entonces",
    "uste": "usted",
    "jui": "huir",
    "crees": "creer",
    "contaste": "contar",
    "ejercito": "ej\u00e9rcito",
    "policia": "polic\u00eda",
    "lider": "l\u00edder",
    "entoes": "entonces",
    "conociste": "conocer",
    "empezaste": "empezar",
    "mencionastir": "mencionar",
    "cu\u00e9ntamar": "contar",
    "rededor": "alrededor",
    "hablastir": "hablar",
    "dijistar": "decir",
    "llegaste": "llegar",
    "contabas": "contar",
    "preguntarte": "preguntar",
    "tuviste": "tener",
    "acuerdas": "acordar",
    "mencionabas": "mencionar",
    "recuerda": "recordar",
    "cu\u00e9ntenos": "contar",
    "entoz": "entonces",
    "t\u00e1": "estar",
    "fuiste": "ir",
    "jueron": "ir",
    "mag\u00ednese": "imaginar",
    "entoe": "entonces",
    "sentistar": "sentir",
    "organizac\u00f3n": "organizaci\u00f3n",
    "organizacion": "organizaci\u00f3n",
    "acu\u00e9rdesir": "acordar",
    "alla": "all\u00e1",
    "publica": "p\u00fablica",
    "futbol": "f\u00fatbol",
    "quedaste": "quedar",
}

RUIDO_EMBEDIDO: frozenset[str] = frozenset({
    "ban", "cio", "ci\u00f3n", "cont\u00ednuum", "dad",
    "documentation", "et", "exintegrant", "flicto", "ib\u00edd",
    "iii", "iv", "lencia", "lidad", "litar",
    "maci\u00f3n", "map", "mento", "mos", "nado",
    "nal", "nar", "nas", "nes", "nidad",
    "nizaci\u00f3n", "nsa", "of", "project", "p\u00e1rr",
    "si\u00f3n", "sona", "taci\u00f3n", "tado", "tamiento",
    "tes", "tivo", "torio", "tro", "vs",
    "zaci\u00f3n", "zado",
})

TFIDF_PARAMS_EMBEDIDOS: dict = {
    "analyzer": "word",
    "token_pattern": r"(?u)\b\w+\b",
    "lowercase": True,
    "ngram_range": [1, 2],
    "sublinear_tf": True,
    "norm": "l2",
    "use_idf": True,
    "smooth_idf": True,
    "min_df": 2,
    "max_df": 0.85,
}

print(
    "Config embebida (independencia total): "
    f"{len(STOPWORDS_HABLA_EMBEDIDAS)} muletillas | "
    f"{len(CORRECCIONES_EMBEDIDAS)} correcciones | "
    f"{len(RUIDO_EMBEDIDO)} ruido | "
    f"vectorizador TF-IDF: min_df={TFIDF_PARAMS_EMBEDIDOS['min_df']}, "
    f"max_df={TFIDF_PARAMS_EMBEDIDOS['max_df']}, "
    f"ngram_range={tuple(TFIDF_PARAMS_EMBEDIDOS['ngram_range'])}"
)


# %%
"""Diagnostico de prerrequisitos: checklist de caches REQUERIDOS y opcionales.

REQUERIDOS = caches derivados que usan las secciones 01-03 (segmentos,
lemas y docs de ambos corpus): el notebook los GENERA automaticamente en la
seccion Regeneracion (Sec 0b) a partir de los 3 insumos permitidos (dataset
de entrevistas, libros y tesauro en data/raw/). Cuando el insumo dataset
esta presente tambien se chequean las normalizadas y los tokens de
entrevistas (derivados del mismo insumo). OPCIONALES = config/ (embebida en
las constantes de esta seccion), resultados/ e imagenes/ (nunca requeridos:
el notebook calcula inline o muestra un aviso suave).
"""

_MARCA_OK = "\u2713"      # checklist: archivo presente
_MARCA_FALTA = "\u2717"   # checklist: archivo faltante

_MENSAJE_REQUERIDO_FALTANTE = (
    "se generara automaticamente en la seccion Regeneracion si los insumos "
    "(dataset/libros/tesauro) estan en data/raw; descargalos manualmente del "
    "Drive del taller si faltan."
)


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


def _imprimir_checklist(
    filas: list[tuple[str, Path, str | None]],
) -> list[tuple[str, bool]]:
    """Imprime una lista de (nombre, ruta, nota) con marca y tamano en MB.

    Args:
        filas: Nombre descriptivo, ruta y nota opcional de cada archivo.

    Returns:
        Lista de (nombre, presente) para el resumen numerico.
    """
    resultado: list[tuple[str, bool]] = []
    for nombre, ruta, nota in filas:
        presente = ruta.exists()
        resultado.append((nombre, presente))
        marca = _MARCA_OK if presente else _MARCA_FALTA
        linea = f"  {marca} {nombre:<42} {_tamano_mb(ruta):>9.1f} MB"
        if not presente and nota:
            linea += f"\n      -> {nota}"
        print(linea)
    return resultado


def diagnostico_prerrequisitos() -> dict[str, list[tuple[str, bool]]]:
    """Recorre el checklist de caches derivados y archivos opcionales.

    Returns:
        Dict con "requeridos" y "opcionales": listas de (nombre, presente).
        Ademas imprime el estado detallado con marcas, tamanos y, para los
        requeridos faltantes, el aviso de regeneracion automatica.
    """
    ruta_dataset = (
        RUTA_RAIZ / "data" / "raw" / "entrevistas_all_2023-03-21_14-24_05.json"
    )
    requeridos: list[tuple[str, Path, str | None]] = [
        ("segmentos_libros.jsonl", RUTA_CORPUS / "segmentos_libros.jsonl", None),
        ("libros_lemas.txt", RUTA_PREPROCESADO / "libros_lemas.txt", None),
        ("libros_docs.json", RUTA_PREPROCESADO / "libros_docs.json", None),
        ("entrevistas_lemas.txt", RUTA_PREPROCESADO / "entrevistas_lemas.txt", None),
        ("entrevistas_docs.json", RUTA_PREPROCESADO / "entrevistas_docs.json", None),
    ]
    if ruta_dataset.exists():
        requeridos += [
            (
                "entrevistas_normalizadas.jsonl",
                RUTA_CORPUS / "entrevistas_normalizadas.jsonl",
                None,
            ),
            (
                "entrevistas_tokens.txt",
                RUTA_PREPROCESADO / "entrevistas_tokens.txt",
                None,
            ),
        ]
    requeridos = [
        (nombre, ruta, nota if nota else _MENSAJE_REQUERIDO_FALTANTE)
        for nombre, ruta, nota in requeridos
    ]

    print(f"RUTA_RAIZ: {RUTA_RAIZ}")
    print("\nChecklist de caches REQUERIDOS (se generan desde los 3 insumos:")
    print("dataset de entrevistas + libros + tesauro en data/raw/):")
    resultado_requeridos = _imprimir_checklist(requeridos)
    n_presentes = sum(1 for _, presente in resultado_requeridos if presente)
    n_total = len(resultado_requeridos)
    print(f"\nChecklist requerido: {n_presentes}/{n_total} presente(s)")
    if n_presentes < n_total:
        faltan = n_total - n_presentes
        print(f"Faltan {faltan} cache(s) derivado(s): {_MENSAJE_REQUERIDO_FALTANTE}")
    if not ruta_dataset.exists():
        print(
            "  [info] El dataset crudo de entrevistas no esta en data/raw: las "
            "normalizadas/tokens de entrevistas no se chequean (el dataset es "
            "descarga manual del Drive del taller)."
        )

    opcionales: list[tuple[str, Path, str | None]] = [
        ("config/stopwords_habla.txt", RUTA_CONFIG / "stopwords_habla.txt",
         "config embebida en sec00 (verificar_consistencia_config)"),
        ("config/lemas_correcciones.json", RUTA_CONFIG / "lemas_correcciones.json",
         "config embebida en sec00 (verificar_consistencia_config)"),
        ("config/tfidf_config.json", RUTA_CONFIG / "tfidf_config.json",
         "config embebida en sec00 (verificar_consistencia_config)"),
        ("resultados/ranking_top5.csv", RUTA_RESULTADOS / "ranking_top5.csv",
         "el ranking se calcula inline en la Sec 03 (verificacion opcional)"),
        ("resultados/evaluacion_muestra_etiquetada.csv",
         RUTA_RESULTADOS / "evaluacion_muestra_etiquetada.csv",
         "etiquetas embebidas en la Sec 03 (verificacion opcional)"),
        ("imagenes/eda_entrevistas_hist_chars.png",
         RUTA_IMAGENES / "eda_entrevistas_hist_chars.png",
         "mostrar_png avisa si falta; la figura se regenera en la etapa E3"),
        ("imagenes/eda_near_duplicates_entrevistas.png",
         RUTA_IMAGENES / "eda_near_duplicates_entrevistas.png",
         "mostrar_png avisa si falta; la figura se regenera en la etapa E3"),
    ]
    print("\nOPCIONALES (nunca requeridos; config embebida, resultados se")
    print("recalculan inline e imagenes usa mostrar_png con fallback):")
    resultado_opcionales = _imprimir_checklist(opcionales)
    n_opc = sum(1 for _, presente in resultado_opcionales if presente)
    print(
        f"\nChecklist opcional: {n_opc}/{len(resultado_opcionales)} presente(s); "
        "el notebook corre completo sin ninguno."
    )
    return {"requeridos": resultado_requeridos, "opcionales": resultado_opcionales}


_resultado_diagnostico = diagnostico_prerrequisitos()


# %%
"""Carga de lemas, metadatos y config embebida; parada en consumo (Fase 6).

Las funciones replican localmente (sin importar modulos de `scripts/` ni
`experimentos/`) el criterio congelado en config/ usando las COPIAS
EMBEBIDAS de la celda de constantes (independencia total):

- Parada EN CONSUMO sobre lemas = `spacy.lang.es.STOP_WORDS` +
  STOPWORDS_HABLA_EMBEDIDAS (51 muletillas) +, solo para el pipeline IR
  (Sec 03), RUIDO_EMBEDIDO (42 tokens tipograficos de libros; lista
  verificada identica a RUIDO_LIBROS de 03_eda.py).
- Correcciones de lemas anomalos del lematizador en consumo:
  CORRECCIONES_EMBEDIDAS (62 mapas, p. ej. atreveir -> atrever).
- Anos 1900-2025 y categorias canonicas: no se tocan (la Fase 4 ya los
  dejo como lemas; la correccion no los mapea y la parada no los contiene).

La parada del EDA (Sec 02) NO incluye el ruido tipografico, igual que
scripts/03_eda.py (paridad numerica con resultados/eda/); el ruido solo se
excluye del analisis comparativo (solapamiento y log-odds), como en la
etapa 6 del EDA de produccion. `verificar_consistencia_config()` compara
las copias embebidas contra los archivos de config/ cuando existen.
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


def cargar_correcciones(usar_archivo: bool = False) -> dict[str, str]:
    """Devuelve el mapa de correcciones de lemas (embebido por defecto).

    Por defecto usa CORRECCIONES_EMBEDIDAS (copia congelada, independencia
    total). Con `usar_archivo=True` lee config/lemas_correcciones.json (lo
    usa verificar_consistencia_config) y degrada a la copia embebida si el
    archivo no existe.

    Args:
        usar_archivo: True para leer el archivo de config/ (verificacion).

    Returns:
        Mapa lema anomalo -> lema correcto (claves con la forma del corpus).
    """
    if usar_archivo:
        ruta = RUTA_CONFIG / "lemas_correcciones.json"
        if ruta.exists():
            datos = json.loads(ruta.read_text(encoding="utf-8"))
            correcciones = datos.get("correcciones", {})
            return {str(k): str(v) for k, v in correcciones.items()}
    return dict(CORRECCIONES_EMBEDIDAS)


def cargar_muletillas(usar_archivo: bool = False) -> set[str]:
    """Devuelve las muletillas del habla (embebidas por defecto).

    Por defecto usa STOPWORDS_HABLA_EMBEDIDAS (51 tokens congelados). Con
    `usar_archivo=True` lee config/stopwords_habla.txt (verificacion) y
    degrada a la copia embebida si el archivo no existe.

    Args:
        usar_archivo: True para leer el archivo de config/ (verificacion).

    Returns:
        Conjunto de muletillas congeladas en el EDA (51 tokens).
    """
    if usar_archivo:
        ruta = RUTA_CONFIG / "stopwords_habla.txt"
        if ruta.exists():
            return {
                linea.strip()
                for linea in ruta.read_text(encoding="utf-8").splitlines()
                if linea.strip() and not linea.startswith("#")
            }
    return set(STOPWORDS_HABLA_EMBEDIDAS)


def cargar_ruido(usar_archivo: bool = False) -> set[str]:
    """Devuelve el ruido tipografico de libros (embebido por defecto).

    Por defecto usa RUIDO_EMBEDIDO (42 tokens: fragmentos de palabras
    partidas por el guion del PDF y ruido bibliografico como 'ibid.',
    verificados identicos a RUIDO_LIBROS de scripts/03_eda.py). Con
    `usar_archivo=True` lee config/tfidf_config.json (verificacion) y
    degrada a la copia embebida si el archivo no existe.

    Args:
        usar_archivo: True para leer el archivo de config/ (verificacion).

    Returns:
        Conjunto de tokens de ruido a excluir del vocabulario.
    """
    if usar_archivo:
        ruta = RUTA_CONFIG / "tfidf_config.json"
        if ruta.exists():
            datos = json.loads(ruta.read_text(encoding="utf-8"))
            ruido = datos.get("ruido_tipografico_libros_excluido_en_eda", [])
            return {str(token) for token in ruido}
    return set(RUIDO_EMBEDIDO)


def construir_parada(incluir_muletillas: bool = True, incluir_ruido: bool = False) -> set[str]:
    """Construye la parada en consumo sobre lemas para un consumo dado.

    Parada base = spacy.lang.es.STOP_WORDS; opcionalmente se anaden las
    muletillas embebidas y el ruido tipografico embebido (este ultimo solo
    en el pipeline IR de la Sec 03, mismo criterio del experimento base de
    la Fase 6).

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


def verificar_consistencia_config() -> None:
    """Compara las copias embebidas con los archivos de config/ (opcional).

    Si los archivos config/ existen, imprime OK cuando las copias
    embebidas coinciden y un warning con las diferencias si no; si no
    existen, imprime una nota informativa. Nunca bloquea: el notebook usa
    siempre las constantes embebidas.

    Returns:
        None (solo imprime el resultado de la comparacion).
    """
    ruta_muletillas = RUTA_CONFIG / "stopwords_habla.txt"
    ruta_correcciones = RUTA_CONFIG / "lemas_correcciones.json"
    ruta_tfidf = RUTA_CONFIG / "tfidf_config.json"
    presentes = [r for r in (ruta_muletillas, ruta_correcciones, ruta_tfidf) if r.exists()]
    if not presentes:
        print(
            "[info] config/ no presente; se usan las constantes embebidas "
            "(independencia total)."
        )
        return
    desajustes: list[str] = []
    if cargar_muletillas(usar_archivo=True) != set(STOPWORDS_HABLA_EMBEDIDAS):
        desajustes.append("stopwords_habla.txt")
    if cargar_correcciones(usar_archivo=True) != dict(CORRECCIONES_EMBEDIDAS):
        desajustes.append("lemas_correcciones.json")
    if cargar_ruido(usar_archivo=True) != set(RUIDO_EMBEDIDO):
        desajustes.append("tfidf_config.json (ruido)")
    if desajustes:
        print(
            "[warning] config embebida != config/ en: " + ", ".join(desajustes)
            + ". El notebook usa las copias embebidas; regenera las "
            "constantes si config/ cambio (celda de constantes)."
        )
    else:
        print("config embebida == config/ ... OK")


# Paradas efectivas del notebook, calculadas una sola vez (deterministas).
CORRECCIONES = cargar_correcciones()
MULETILLAS = cargar_muletillas()
RUIDO_LIBROS = cargar_ruido()
PARADA_EDA = construir_parada(incluir_muletillas=True, incluir_ruido=False)
PARADA_IR = construir_parada(incluir_muletillas=True, incluir_ruido=True)


def aplicar_parada(linea_lemas: str, parada: set[str] | None = None) -> str:
    """Aplica correcciones y parada en consumo a una linea de lemas.

    Por token: primero la correccion (lema anomalo -> lema correcto,
    CORRECCIONES_EMBEDIDAS) y, sobre el resultado, la parada. Anos
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


def aviso_archivo_faltante(archivo_nombre: str, ruta: Path | None = None) -> None:
    """Imprime el aviso canonico de cache derivado faltante (guard suave).

    Args:
        archivo_nombre: Nombre del archivo que falta (para el mensaje).
        ruta: Ruta completa del archivo faltante (opcional, informativa).
    """
    detalle = f" {ruta}" if ruta is not None else ""
    print(
        f"[aviso] Falta {archivo_nombre}{detalle}; ejecuta la seccion "
        "Regeneracion con los insumos descargados (dataset/libros/tesauro) "
        "y vuelve a ejecutar Run all."
    )


def mostrar_png(nombre: str, ancho: int = 900) -> None:
    """Muestra un PNG de imagenes/ embebido en el notebook (fallback suave).

    Las figuras pesadas del EDA de produccion (chars y paginas de
    entrevistas, near-duplicates) requieren datos >100 MB o el pipeline de
    scripts/03_eda.py; el notebook las muestra con este helper desde
    `imagenes/`. Si el archivo no existe NO falla: imprime una nota clara
    (la figura se genera automaticamente en la seccion Regeneracion, etapa
    E3, cuando el insumo dataset este en data/raw) y sigue con la ejecucion.

    Args:
        nombre: nombre del archivo dentro de imagenes/ (p. ej.
            "eda_entrevistas_hist_chars.png").
        ancho: ancho en pixeles del PNG embebido.
    """
    ruta = RUTA_IMAGENES / nombre
    if not ruta.exists():
        print(
            f"[aviso] Figura {nombre} no disponible (requiere datos >100 MB); "
            "se genera automaticamente en la seccion Regeneracion (etapa E3) "
            "cuando el insumo dataset este en data/raw."
        )
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


verificar_consistencia_config()
print(
    f"Configuracion cargada (embebida): {len(CORRECCIONES)} correcciones | "
    f"{len(MULETILLAS)} muletillas | {len(RUIDO_LIBROS)} ruido | "
    f"parada EDA {len(PARADA_EDA)} tokens | parada IR {len(PARADA_IR)} tokens"
)


# %%
"""Verificacion rapida de carga: lineas por corpus y ejemplo de parada."""

# Guard suave: los caches derivados los genera la seccion Regeneracion desde
# los 3 insumos (dataset/libros/tesauro); si aun asi faltan, aviso claro.
for corpus in ("libros", "entrevistas"):
    ruta_lemas = RUTA_PREPROCESADO / f"{corpus}_lemas.txt"
    ruta_docs = RUTA_PREPROCESADO / f"{corpus}_docs.json"
    if not (ruta_lemas.exists() and ruta_docs.exists()):
        aviso_archivo_faltante(f"{corpus}_lemas.txt/_docs.json")
        continue
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
