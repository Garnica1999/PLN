# %% [markdown]
"""# Sec 0b — Regeneracion del corpus dentro del notebook (opcional)

Los archivos derivados del corpus (segmentos, normalizadas, lemas/tokens y
los CSVs del EDA) los producen los scripts de produccion
(`scripts/01_extraer_libros.py`, `scripts/02_preprocesar.py` y
`scripts/03_eda.py`). Aqui su codigo canonico esta **embebido
automaticamente por el ensamblador** (`scripts/ensamblar_notebook_t4.py`),
que inserta el contenido literal de cada script en las celdas de esta
seccion: fuente unica en el build, si cambian los scripts se re-ensambla el
notebook y las celdas quedan al dia.

Las etapas de regeneracion **solo se ejecutan si faltan archivos** en disco y
si ademas activas los flags de la celda de configuracion (ver abajo). Por
defecto TODO esta apagado: la seccion no hace nada pesado, solo imprime el
estado (en el repositorio todo lo necesario ya esta presente).

Flags de la celda de configuracion:

- `REGEN_AUTOMATICO = False`: si lo pones en True, la ultima celda de la
  seccion ejecuta `regen_etapa()` sobre cada etapa con archivos faltantes
  (o todas si `FUERZAR_REGENERACION = True`).
- `FUERZAR_REGENERACION = False`: re-ejecuta la etapa aunque sus archivos ya
  existan (reescribe los derivados; `config/` NUNCA se modifica: la etapa
  "congelar" del EDA no se ejecuta desde el notebook).
- `DESCARGAR_FUENTES = False`: permite descargar las fuentes crudas del
  Drive del taller (JSON de entrevistas y zip/carpeta de libros) con gdown
  cuando una etapa las necesita y no estan en `data/raw/`.
- `RUTA_FUENTES = None`: ruta local alternativa con las fuentes ya
  descargadas (carpeta con `entrevistas_all_2023-03-21_14-24_05.json` y el
  zip `books-*.zip` o la carpeta `books/` con los PDFs).

Las 4 etapas y su codigo canónico:

| Etapa | Script embebido | Produce |
|---|---|---|
| `extraccion` | 01_extraer_libros.py | `data/corpus/segmentos_libros.jsonl`, `segmentos/<TOMO>.jsonl`, reporte |
| `normalizar` | 02_preprocesar.py `normalizar` | `data/corpus/entrevistas_normalizadas.jsonl` |
| `preprocesado_spacy` | 02_preprocesar.py `spacy` | `data/corpus/preprocesado/*_lemas.txt`, `*_tokens.txt`, `*_docs.json` |
| `eda` | 03_eda.py (sin "congelar") | CSVs de `resultados/eda/` y figuras de `imagenes/` |

La primera regeneracion completa (extraccion + normalizar + spaCy + EDA) es
lenta: 15-60 minutos en Colab (la normalizacion y spaCy dominan). El
repositorio ya trae todo lo derivado versionado (salvo 2 archivos > 100 MB
opcionales con respaldo estatico en las secciones 01-03), asi que en el uso
normal esta seccion queda inerte.
"""


# %%
"""Rutas de la seccion (usa las de sec00 si existen; replica solo standalone).

En el notebook sec00 siempre corre antes y define RUTA_RAIZ, RUTA_CORPUS,
RUTA_CONFIG y el resto de RUTA_*; este bloque las usa si existen y replica la
resolucion robusta de la raiz solo si la seccion se ejecuta como script
standalone (misma convencion de autocontencion que el TALLER-2).
"""
from pathlib import Path


def _raiz_standalone() -> Path:
    """Localiza la raiz del proyecto TALLER-4 (respaldo standalone robusto).

    Copia autocontenida del algoritmo de sec00 (sin TALLER4_ROOT ni
    auto-descarga, para no sorprender en ejecuciones sueltas): ancestros
    desde el cwd, subcarpetas de hasta 3 niveles bajo el cwd y bajo cada
    ancestro, y rutas tipicas de Colab. Si no encuentra, lanza un error
    claro; nunca devuelve la raiz del filesystem.

    Returns:
        Path absoluto de la raiz (contiene data/corpus/segmentos_libros.jsonl).

    Raises:
        FileNotFoundError: Con instrucciones de como subir la carpeta.
    """
    import os as _os

    marcador = Path("data") / "corpus" / "segmentos_libros.jsonl"
    excluidos = {".git", "drive", "sample_data", "venv", "node_modules"}
    cwd = Path(_os.getcwd()).resolve()

    def cumple(directorio: Path) -> bool:
        """True si el directorio contiene el marcador de la raiz."""
        return (directorio / marcador).exists()

    def caminar(base: Path, max_profundidad: int):
        """Itera base y sus subcarpetas hasta la profundidad dada."""
        nivel_base = len(Path(base).parts)
        for dir_actual, subdirs, _ in _os.walk(base):
            nivel = len(Path(dir_actual).parts) - nivel_base
            if nivel >= max_profundidad:
                subdirs[:] = []
            else:
                subdirs[:] = sorted(
                    d for d in subdirs if not d.startswith(".") and d not in excluidos
                )
            yield Path(dir_actual)

    ancestros = [cwd, *cwd.parents]
    for ancestro in ancestros:
        if cumple(ancestro):
            return ancestro
    for ancestro in ancestros:
        for candidato in caminar(ancestro, 3):
            if cumple(candidato):
                return candidato
    for candidata in (Path("/content/TALLER-4"), Path("/content/drive/MyDrive/TALLER-4")):
        if cumple(candidata):
            return candidata
    raise FileNotFoundError(
        "No se encontro la raiz del proyecto TALLER-4: falta "
        "data/corpus/segmentos_libros.jsonl en el arbol de directorios "
        "(sube la carpeta TALLER-4 completa a /content, o descomprime ahi el "
        "zip del repositorio para que quede en /content/PLN-main/TALLER-4)."
    )


if "RUTA_RAIZ" not in globals():
    RUTA_RAIZ = _raiz_standalone()
    RUTA_CORPUS = RUTA_RAIZ / "data" / "corpus"
    RUTA_CONFIG = RUTA_RAIZ / "config"

_RUTA_RAIZ = RUTA_RAIZ
_RUTA_CORPUS = RUTA_RAIZ / "data" / "corpus"
_RUTA_PREPROCESADO = _RUTA_CORPUS / "preprocesado"
_RUTA_RAW = RUTA_RAIZ / "data" / "raw"
_RUTA_CONFIG = RUTA_RAIZ / "config"
_RUTA_SCRIPTS = RUTA_RAIZ / "scripts"
_RUTA_RESULTADOS = RUTA_RAIZ / "resultados"
_RUTA_EDA = _RUTA_RESULTADOS / "eda"
_RUTA_IMAGENES = RUTA_RAIZ / "imagenes"

print(f"[regen] Raiz usada: {_RUTA_RAIZ}")


# %%
"""Configuracion de la seccion de regeneracion (flags, todo apagado).

`_archivos_etapa()` declara, por etapa, los archivos que produce y los
insumos que necesita (rutas absolutas sobre la raiz detectada); la usa el
dispatcher para decidir si falta algo. `CENTINELAS_EDA` ordena las subetapas
del EDA cuyo archivo representativo se comprueba (orden canonico de
scripts/03_eda.py, sin "nubes" - solo produce figuras - y sin "congelar",
que reescribiria config/).
"""
import shutil
import subprocess
import sys
from pathlib import Path

REGEN_AUTOMATICO = False
DESCARGAR_FUENTES = False
FUERZAR_REGENERACION = False
RUTA_FUENTES = None  # ruta local opcional con las fuentes crudas (override)

N_PROCESOS_SPACY = 4  # workers para el preprocesado spaCy en Colab

ORDEN_TOMOS = (
    "HALLAZGOS", "CONVOCATORIA", "NARRATIVA", "SUFRIR", "TESTIMONIAL",
    "CUERPO", "NNA", "VIOLACIONES", "ETNICO", "FUERA",
)

# Subetapas del EDA en orden canonico (03_eda.py) con su archivo centinela.
CENTINELAS_EDA = (
    ("estadisticas", _RUTA_EDA / "estadisticas_corpus.csv"),
    ("frecuencias", _RUTA_EDA / "frecuencias_libros_unigramas_top30.csv"),
    ("diversidad", _RUTA_EDA / "diversidad_lexica.csv"),
    ("near_dups", _RUTA_EDA / "near_duplicates_entrevistas.csv"),
    ("comparacion", _RUTA_EDA / "solapamiento_vocabulario.csv"),
    ("tesauro", _RUTA_EDA / "tesauro_cache_etiquetas.json"),
)


def _archivos_etapa() -> dict[str, dict[str, list[Path]]]:
    """Declara los archivos que produce y los insumos de cada etapa.

    Returns:
        Dict {etapa: {"produce": [Path...], "inputs": [Path...]}} con los
        archivos representativos de cada etapa (los que usan las secciones
        01-03 del notebook).
    """
    return {
        "extraccion": {
            "produce": [
                _RUTA_CORPUS / "segmentos_libros.jsonl",
                _RUTA_RESULTADOS / "extraccion_libros_report.json",
                *[_RUTA_CORPUS / "segmentos" / f"{tomo}.jsonl" for tomo in ORDEN_TOMOS],
            ],
            "inputs": [_RUTA_RAW / "books-20260904T025322Z-1-001.zip"],
        },
        "normalizar": {
            "produce": [_RUTA_CORPUS / "entrevistas_normalizadas.jsonl"],
            "inputs": [_RUTA_RAW / "entrevistas_all_2023-03-21_14-24_05.json"],
        },
        "preprocesado_spacy": {
            "produce": [
                _RUTA_PREPROCESADO / f"{corpus}_{tipo}.txt"
                for corpus in ("libros", "entrevistas")
                for tipo in ("lemas", "tokens")
            ]
            + [
                _RUTA_PREPROCESADO / f"{corpus}_docs.json"
                for corpus in ("libros", "entrevistas")
            ],
            "inputs": [
                _RUTA_CORPUS / "segmentos_libros.jsonl",
                _RUTA_CORPUS / "entrevistas_normalizadas.jsonl",
                _RUTA_CONFIG / "stopwords_habla.txt",
            ],
        },
        "eda": {
            "produce": [ruta for _, ruta in CENTINELAS_EDA],
            "inputs": [
                _RUTA_PREPROCESADO / "libros_lemas.txt",
                _RUTA_PREPROCESADO / "entrevistas_lemas.txt",
                _RUTA_CONFIG / "lemas_correcciones.json",
                _RUTA_CONFIG / "stopwords_habla.txt",
                _RUTA_CONFIG / "tfidf_config.json",
            ],
        },
    }


print(
    f"[regen] REGEN_AUTOMATICO={REGEN_AUTOMATICO} | "
    f"FUERZAR_REGENERACION={FUERZAR_REGENERACION} | "
    f"DESCARGAR_FUENTES={DESCARGAR_FUENTES} | RUTA_FUENTES={RUTA_FUENTES}"
)
print(
    "[regen] Marcador de la raiz presente: "
    f"{(_RUTA_CORPUS / 'segmentos_libros.jsonl').exists()}"
)


# %%
"""Dispatcher de la regeneracion: estado y ejecucion por etapa.

`estado_faltantes()` comprueba contra el disco (sobre la raiz detectada) los
archivos que produce cada etapa y devuelve los faltantes; `regen_etapa()`
ejecuta el codigo embebido correspondiente (celdas marcadas con el codigo
canonico de scripts/01-03) solo si faltan archivos (o si `forzar=True`) y
descarga las fuentes crudas antes cuando hacen falta y estan habilitadas.
"""


def _inputs_disponibles(etapa: str) -> bool:
    """True si existen todos los insumos declarados de la etapa.

    Para "extraccion" tambien se aceptan los PDFs sueltos de data/raw/books
    (el wrapper sabe crear el zip canonico desde ellos).

    Args:
        etapa: Nombre de la etapa ("extraccion", "normalizar", ...).

    Returns:
        True si la etapa puede ejecutarse con lo que hay en disco.
    """
    declarados = _archivos_etapa()[etapa]["inputs"]
    if etapa == "extraccion":
        if all(ruta.exists() for ruta in declarados):
            return True
        return any((_RUTA_RAW / "books").glob("*.pdf"))
    return all(ruta.exists() for ruta in declarados)


def estado_faltantes() -> dict[str, list[Path]]:
    """Comprueba en disco los archivos producidos por las 4 etapas.

    Returns:
        Dict {etapa: [archivos faltantes]} (lista vacia si la etapa esta
        completa).
    """
    faltantes: dict[str, list[Path]] = {}
    for etapa, info in _archivos_etapa().items():
        faltan = [ruta for ruta in info["produce"] if not ruta.exists()]
        faltantes[etapa] = faltan
        estado = "OK" if not faltan else f"faltan {len(faltan)} archivo(s)"
        print(f"  [estado] {etapa:<20} {estado}")
    return faltantes


def regen_etapa(nombre_etapa: str, forzar: bool = False) -> bool:
    """Ejecuta la regeneracion de una etapa si faltan archivos (o forzar).

    Args:
        nombre_etapa: "extraccion", "normalizar", "preprocesado_spacy" o
            "eda".
        forzar: True re-ejecuta aunque los archivos existan.

    Returns:
        True si la etapa quedo completa (o no requeria accion); False si no
        se pudo ejecutar (faltan insumos o prerrequisitos del sistema).
    """
    declaradas = _archivos_etapa()
    if nombre_etapa not in declaradas:
        print(f"[regen] Etapa desconocida: {nombre_etapa}")
        return False
    produce = declaradas[nombre_etapa]["produce"]
    faltan = [ruta for ruta in produce if not ruta.exists()]
    if not faltan and not forzar:
        print(
            f"[regen] Etapa {nombre_etapa}: OK "
            f"({len(produce)} archivos presentes), sin accion."
        )
        return True
    if nombre_etapa == "extraccion":
        return regen_extraccion_libros(forzar=forzar)
    if nombre_etapa == "normalizar":
        return regen_normalizar_entrevistas(forzar=forzar)
    if nombre_etapa == "preprocesado_spacy":
        return regen_preprocesado_spacy(forzar=forzar)
    if nombre_etapa == "eda":
        return regen_eda(forzar=forzar)
    return False


# %%
"""Descarga de fuentes crudas (Drive del taller) con gdown.

La carpeta de libros del enunciado es
https://drive.google.com/drive/folders/1W9UOdkqnAcztnFZ8xresW4ZeZ2pqXe4v y
el archivo de entrevistas es
https://drive.google.com/file/d/1XeFS_mQFmTLQ_VUiODTBO5eONWv-Ege5 (el
tesauro, 1XgZzMtFQp0RzwqdY7Em5q5ProsWLMMX4, viaja versionado en data/raw).
`descargar_fuentes()` copia desde `RUTA_FUENTES` (override local) o descarga
con gdown si `DESCARGAR_FUENTES=True`; si el Drive no es accesible desde el
entorno, imprime un mensaje claro con los enlaces y como dejar los archivos
a mano en data/raw. NO se llama si la etapa no la necesita.
"""

# IDs de Google Drive del enunciado del taller (docs/).
ID_ENTREVISTAS_DRIVE = "1XeFS_mQFmTLQ_VUiODTBO5eONWv-Ege5"
ID_CARPETA_LIBROS_DRIVE = "1W9UOdkqnAcztnFZ8xresW4ZeZ2pqXe4v"
NOMBRE_JSON_ENTREVISTAS = "entrevistas_all_2023-03-21_14-24_05.json"


def _copiar_fuentes_locales(destino_data_raw: Path) -> list[str]:
    """Copia las fuentes desde RUTA_FUENTES (si esta definida) a data/raw.

    RUTA_FUENTES debe contener el JSON de entrevistas y el zip de libros
    (books-*.zip) o la carpeta books/ con los PDFs.

    Args:
        destino_data_raw: Carpeta data/raw del proyecto.

    Returns:
        Lista de descripciones copiadas (vacia si no habia nada que copiar).
    """
    if not RUTA_FUENTES:
        return []
    origen = Path(RUTA_FUENTES)
    if not origen.exists():
        print(f"[fuentes] RUTA_FUENTES no existe: {origen}")
        return []
    copiadas: list[str] = []
    destino_data_raw.mkdir(parents=True, exist_ok=True)
    json_origen = origen / NOMBRE_JSON_ENTREVISTAS
    if json_origen.exists() and not (destino_data_raw / NOMBRE_JSON_ENTREVISTAS).exists():
        shutil.copyfile(json_origen, destino_data_raw / NOMBRE_JSON_ENTREVISTAS)
        copiadas.append("JSON de entrevistas")
    zips_origen = sorted(origen.glob("books-*.zip"))
    if zips_origen and not (destino_data_raw / "books").exists():
        for zip_origen in zips_origen:
            shutil.copyfile(zip_origen, destino_data_raw / zip_origen.name)
        copiadas.append(f"{len(zips_origen)} zip(s) de libros")
    libros_origen = origen / "books"
    if libros_origen.is_dir() and not (destino_data_raw / "books").exists():
        shutil.copytree(libros_origen, destino_data_raw / "books")
        copiadas.append("carpeta books con PDFs")
    return copiadas


def descargar_fuentes(destino_data_raw: Path) -> dict[str, bool]:
    """Deja en data/raw las fuentes crudas (JSON + zip/carpeta de libros).

    Args:
        destino_data_raw: Carpeta data/raw del proyecto (se crea si falta).

    Returns:
        Dict {"entrevistas": bool, "libros": bool} con lo disponible al
        terminar (True si el archivo/carpeta quedo en disco).

    Raises:
        FileNotFoundError: Si no hay RUTA_FUENTES ni DESCARGAR_FUENTES y
            falta alguna fuente pedida explicitamente.
    """
    destino_data_raw.mkdir(parents=True, exist_ok=True)
    ruta_json = destino_data_raw / NOMBRE_JSON_ENTREVISTAS
    zip_disponible = sorted(destino_data_raw.glob("books-*.zip"))
    libros_dir = destino_data_raw / "books"
    libros_ok = bool(zip_disponible) or libros_dir.is_dir()

    copiadas = _copiar_fuentes_locales(destino_data_raw)
    if copiadas:
        print(f"[fuentes] Copiado desde RUTA_FUENTES: {', '.join(copiadas)}")

    if not ruta_json.exists() and DESCARGAR_FUENTES:
        print("[fuentes] Descargando entrevistas (gdown, ~231 MB) ...")
        try:
            import gdown

            gdown.download(
                id=ID_ENTREVISTAS_DRIVE, output=str(ruta_json), quiet=True
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[fuentes] Fallo la descarga de entrevistas: {exc}")
    if not libros_ok and DESCARGAR_FUENTES:
        print("[fuentes] Descargando carpeta de libros (gdown, ~300 MB) ...")
        try:
            import gdown

            salida_libros = destino_data_raw / "libros_gdown"
            gdown.download_folder(
                id=ID_CARPETA_LIBROS_DRIVE,
                output=str(salida_libros),
                quiet=True,
                use_cookies=False,
            )
            libros_ok = bool(sorted(salida_libros.glob("books-*.zip"))) or (
                salida_libros / "books"
            ).is_dir()
        except Exception as exc:  # noqa: BLE001
            print(f"[fuentes] Fallo la descarga de libros: {exc}")

    disponibles = {
        "entrevistas": ruta_json.exists(),
        "libros": libros_ok or bool(sorted(destino_data_raw.glob("books-*.zip"))),
    }
    if not disponibles["entrevistas"]:
        print(
            "[fuentes] Falta el JSON de entrevistas. Descargalo del Drive del "
            f"taller ({ID_ENTREVISTAS_DRIVE}) y dejalo como data/raw/"
            f"{NOMBRE_JSON_ENTREVISTAS}, o define RUTA_FUENTES con la carpeta "
            "que lo contiene."
        )
    if not disponibles["libros"]:
        print(
            "[fuentes] Faltan los libros. Descarga la carpeta del Drive del "
            f"taller ({ID_CARPETA_LIBROS_DRIVE}) y deja el zip books-*.zip (o "
            "la carpeta books/ con los PDFs) en data/raw/, o define "
            "RUTA_FUENTES."
        )
    return disponibles


# %%
"""Etapa E1: extraccion y segmentacion de los libros (codigo canonico de
scripts/01_extraer_libros.py, embebido por el ensamblador como fuente unica).

El ensamblador reemplaza la linea siguiente por el contenido completo del
script; esta celda sin ensamblar no ejecuta nada (aviso). Prerrequisitos:
binario `pdftotext` (poppler-utils; si falta se intenta instalar con apt-get
en Colab) y el zip de libros en data/raw (o los PDFs en data/raw/books, con
los que el wrapper construye el zip canonico). El script usa sys.argv en su
main(), asi que el wrapper parchea sys.argv temporalmente y lo restaura.
"""
CODIGO_EMBED_01 = None  # [embed: scripts/01_extraer_libros.py]


def _instalar_poppler_utils() -> bool:
    """Intenta instalar poppler-utils con apt-get si pdftotext no existe.

    Returns:
        True si pdftotext quedo disponible al terminar.
    """
    if shutil.which("pdftotext"):
        return True
    if shutil.which("apt-get") is None:
        return False
    try:
        print("[E1] Instalando poppler-utils (pdftotext) ...")
        subprocess.check_call(["apt-get", "update", "-qq"], timeout=300)
        subprocess.check_call(
            ["apt-get", "install", "-y", "-qq", "poppler-utils"], timeout=900
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[E1] Fallo la instalacion de poppler-utils: {exc}")
        return False
    return shutil.which("pdftotext") is not None


def _preparar_zip_libros() -> str | None:
    """Asegura el zip canonico que espera scripts/01_extraer_libros.py.

    Si no existe data/raw/books-20260904T025322Z-1-001.zip: se copia
    cualquier otro books-*.zip con ese nombre, o se crea un zip desde los
    PDFs sueltos de data/raw/books (miembros con prefijo books/, igual que el
    zip original del Drive).

    Returns:
        None si quedo listo; mensaje de error si no hay zip ni PDFs.
    """
    ruta_esperada = _RUTA_RAW / "books-20260904T025322Z-1-001.zip"
    if ruta_esperada.exists():
        return None
    candidatos = sorted(_RUTA_RAW.glob("books-*.zip"))
    if candidatos:
        shutil.copyfile(candidatos[0], ruta_esperada)
        print(f"[E1] Zip de libros renombrado a {ruta_esperada.name}")
        return None
    pdfs = sorted((_RUTA_RAW / "books").glob("*.pdf")) if (_RUTA_RAW / "books").is_dir() else []
    if not pdfs:
        return (
            "no hay zip de libros ni PDFs en data/raw: activa "
            "DESCARGAR_FUENTES (o define RUTA_FUENTES) y vuelve a intentar."
        )
    import zipfile

    with zipfile.ZipFile(ruta_esperada, "w", zipfile.ZIP_STORED) as archivo:
        for pdf in pdfs:
            archivo.write(pdf, f"books/{pdf.name}")
    print(f"[E1] Zip creado desde {len(pdfs)} PDFs de data/raw/books")
    return None


def regen_extraccion_libros(forzar: bool = False) -> bool:
    """Regenera la extraccion de segmentos de los libros (script 01 embebido).

    Args:
        forzar: True re-extrae los tomos aunque sus JSONL existan (--force).

    Returns:
        True si la etapa quedo completa (o no habia nada que hacer); False
        si faltan prerrequisitos (pdftotext, zip/PDFs) y no se pudieron
        resolver.
    """
    if CODIGO_EMBED_01 is None:
        print(
            "[E1] Celda sin ensamblar: ejecuta "
            "scripts/ensamblar_notebook_t4.py y vuelve a abrir el notebook."
        )
        return False
    if shutil.which("pdftotext") is None and not _instalar_poppler_utils():
        print(
            "[E1] No se encontro pdftotext (poppler-utils). En Colab se "
            "intento apt-get y fallo; instalalo a mano con: "
            "!apt-get update && apt-get install -y poppler-utils"
        )
        return False
    if not _inputs_disponibles("extraccion"):
        descargar_fuentes(_RUTA_RAW)
    error_zip = _preparar_zip_libros()
    if error_zip:
        print(f"[E1] {error_zip}")
        return False

    namespace = {
        "__name__": "__embebido_01__",
        "__file__": str(_RUTA_SCRIPTS / "01_extraer_libros.py"),
    }
    exec(compile(CODIGO_EMBED_01, "01_extraer_libros.py (embebido)", "exec"), namespace)  # noqa: S102 (uso intencional del codigo canonico embebido)
    argv = ["regen_01"] + (["--force"] if forzar else [])
    argv_previo = list(sys.argv)
    try:
        sys.argv = argv
        namespace["main"]()
    except SystemExit:
        pass  # el script usa sys.exit para errores de argumentos
    finally:
        sys.argv = argv_previo
    # Si no habia tomos que procesar pero falta el archivo global, se
    # reconstruye desde los JSONL por tomo (funcion del propio script).
    if not (_RUTA_CORPUS / "segmentos_libros.jsonl").exists():
        namespace["reconstruir_global"]()
    print("[E1] Etapa de extraccion terminada.")
    return True


# %%
"""Etapa E2: normalizacion estructural y preprocesado spaCy (codigo canonico
de scripts/02_preprocesar.py, embebido por el ensamblador como fuente unica).

Dos subetapas: `normalizar` (JSON crudo de entrevistas ->
entrevistas_normalizadas.jsonl, solo si el crudo existe en data/raw) y
`spacy --corpus ambos` (segmentos_libros.jsonl + normalizadas -> lemas,
tokens y docs de libros y entrevistas). El script fija mp fork
internamente; los workers se limitan a N_PROCESOS_SPACY (4) para Colab. El
modelo es_core_news_md debe estar descargado; si falta se intenta con
`python -m spacy download`.
"""
CODIGO_EMBED_02 = None  # [embed: scripts/02_preprocesar.py]


def _verificar_modelo_spacy() -> bool:
    """Comprueba es_core_news_md y lo descarga si hace falta.

    Returns:
        True si el modelo queda disponible.
    """
    try:
        import spacy

        spacy.load("es_core_news_md")
        return True
    except ImportError:
        print("[E2] spaCy no esta instalado; instala la celda de sec00.")
        return False
    except OSError:
        pass
    try:
        print("[E2] Descargando el modelo es_core_news_md de spaCy ...")
        subprocess.check_call(
            [sys.executable, "-m", "spacy", "download", "es_core_news_md"],
            timeout=1200,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[E2] Fallo la descarga del modelo spaCy: {exc}")
        return False
    import spacy

    try:
        spacy.load("es_core_news_md")
        return True
    except OSError:
        return False


def _ejecutar_embebido_02(argv: list[str]) -> bool:
    """Ejecuta el codigo embebido de 02_preprocesar.py con argv dado.

    Args:
        argv: Argumentos para main(["normalizar", ...]) de 02_preprocesar.

    Returns:
        True si main devolvio 0 (sin errores).
    """
    if CODIGO_EMBED_02 is None:
        print(
            "[E2] Celda sin ensamblar: ejecuta "
            "scripts/ensamblar_notebook_t4.py y vuelve a abrir el notebook."
        )
        return False
    namespace = {
        "__name__": "__embebido_02__",
        "__file__": str(_RUTA_SCRIPTS / "02_preprocesar.py"),
    }
    exec(compile(CODIGO_EMBED_02, "02_preprocesar.py (embebido)", "exec"), namespace)  # noqa: S102 (uso intencional del codigo canonico embebido)
    return namespace["main"](["regen_02", *argv]) == 0


def regen_normalizar_entrevistas(forzar: bool = False) -> bool:
    """Regenera data/corpus/entrevistas_normalizadas.jsonl desde el crudo.

    Args:
        forzar: True re-normaliza aunque el archivo exista.

    Returns:
        True si la etapa quedo completa o no aplicaba; False si no existe el
        JSON crudo en data/raw y no se pudo descargar.
    """
    ruta_normalizadas = _RUTA_CORPUS / "entrevistas_normalizadas.jsonl"
    ruta_crudo = _RUTA_RAW / "entrevistas_all_2023-03-21_14-24_05.json"
    if ruta_normalizadas.exists() and not forzar:
        return True
    if not ruta_crudo.exists():
        descargar_fuentes(_RUTA_RAW)
    if not ruta_crudo.exists():
        print(
            "[E2] No existe el JSON crudo de entrevistas en data/raw: la "
            "etapa normalizar se omite (las secciones 01-03 usan su respaldo "
            "estatico)."
        )
        return False
    ok = _ejecutar_embebido_02(["normalizar"] + (["--force"] if forzar else []))
    print("[E2] Subetapa normalizar terminada." if ok else "[E2] Error en normalizar.")
    return ok


def regen_preprocesado_spacy(forzar: bool = False, procesos: int = N_PROCESOS_SPACY) -> bool:
    """Regenera lemas/tokens/docs de libros y entrevistas (spaCy, ambos).

    Args:
        forzar: True ignora los checkpoints y reprocesa desde cero.
        procesos: Workers de spaCy (default 4, pensado para Colab).

    Returns:
        True si la etapa quedo completa; False si faltan insumos o el modelo.
    """
    ruta_crudo_ent = _RUTA_RAW / "entrevistas_all_2023-03-21_14-24_05.json"
    ruta_normalizadas = _RUTA_CORPUS / "entrevistas_normalizadas.jsonl"
    if not (_RUTA_CORPUS / "segmentos_libros.jsonl").exists():
        print("[E2] Falta data/corpus/segmentos_libros.jsonl: ejecuta antes la etapa extraccion.")
        return False
    if not ruta_normalizadas.exists() and ruta_crudo_ent.exists():
        print("[E2] Falta entrevistas_normalizadas.jsonl; normalizando primero ...")
        regen_normalizar_entrevistas(forzar=False)
    if not ruta_normalizadas.exists():
        print(
            "[E2] Falta entrevistas_normalizadas.jsonl (no esta el JSON crudo "
            "en data/raw): la subetapa de entrevistas se omite."
        )
    if not _verificar_modelo_spacy():
        return False
    ok = _ejecutar_embebido_02(
        ["spacy", "--corpus", "ambos", "--procesos", str(procesos)]
        + (["--force"] if forzar else [])
    )
    print("[E2] Subetapa spacy terminada." if ok else "[E2] Error en spacy.")
    return ok


# %%
"""Etapa E3: EDA (codigo canonico de scripts/03_eda.py, embebido por el
ensamblador como fuente unica), ejecutado SIN la subetapa "congelar":
config/ nunca se reescribe desde el notebook.

El script fija matplotlib con backend "Agg" al importar; como la celda corre
en el kernel del notebook, al terminar se restaura el backend inline. Las
subetapas se eligen como la cadena canonica desde la primera con archivo
faltante (estadisticas -> frecuencias -> ... -> tesauro), porque cada etapa
requiere en contexto las anteriores. "tesauro" solo se incluye si su cache
o los CSVs posteriores faltan (sin data/raw/Resource-tesauro.rdf la propia
etapa avisa y omite el cruce).
"""
CODIGO_EMBED_03 = None  # [embed: scripts/03_eda.py]

_ORDEN_SUBETAPAS_EDA = tuple(nombre for nombre, _ in CENTINELAS_EDA)


def _restaurar_backend_inline() -> None:
    """Restaura el backend de matplotlib del notebook tras ejecutar 03_eda.

    03_eda.py fija Agg a nivel de modulo; en el notebook las figuras deben
    volver a quedar inline. Silencioso si el backend inline no esta
    disponible (p. ej. ejecucion como script).
    """
    try:
        import matplotlib

        matplotlib.use("module://matplotlib_inline.backend_inline")
    except Exception as exc:  # noqa: BLE001
        print(f"[E3] Aviso: no se pudo restaurar el backend inline: {exc}")


def regen_eda(forzar: bool = False) -> bool:
    """Regenera los CSVs y figuras del EDA (script 03 embebido, sin congelar).

    Args:
        forzar: True re-ejecuta toda la cadena aunque no falte nada.

    Returns:
        True si la etapa quedo completa o no requeria accion; False si fallo.
    """
    if CODIGO_EMBED_03 is None:
        print(
            "[E3] Celda sin ensamblar: ejecuta "
            "scripts/ensamblar_notebook_t4.py y vuelve a abrir el notebook."
        )
        return False
    faltantes = [
        (i, nombre)
        for i, (nombre, ruta) in enumerate(CENTINELAS_EDA)
        if forzar or not ruta.exists()
    ]
    if not faltantes:
        print(
            f"[E3] EDA completo ({len(CENTINELAS_EDA)} centinelas presentes), "
            "sin accion."
        )
        return True
    indice_minimo = min(i for i, _ in faltantes)
    etapas = list(_ORDEN_SUBETAPAS_EDA[indice_minimo:])
    print(f"[E3] Regenerando subetapas del EDA: {' '.join(etapas)}")

    namespace = {
        "__name__": "__embebido_03__",
        "__file__": str(_RUTA_SCRIPTS / "03_eda.py"),
    }
    exec(compile(CODIGO_EMBED_03, "03_eda.py (embebido)", "exec"), namespace)  # noqa: S102 (uso intencional del codigo canonico embebido)
    try:
        codigo = namespace["main"](["regen_03", "--solo", *etapas])
    finally:
        _restaurar_backend_inline()
    print("[E3] Etapa EDA terminada." if codigo == 0 else "[E3] Error en el EDA.")
    return codigo == 0


# %%
"""Ejecucion de la seccion: con REGEN_AUTOMATICO en False (default) no se
regenera nada; con True se recorre cada etapa con faltantes (o todas si
FUERZAR_REGENERACION) y se imprime el resumen final de estado_faltantes().
"""
if REGEN_AUTOMATICO:
    for etapa in _archivos_etapa():
        faltan = estado_faltantes().get(etapa, [])
        if faltan or FUERZAR_REGENERACION:
            print(f"[regen] Regenerando etapa {etapa} ...")
            regen_etapa(etapa, forzar=FUERZAR_REGENERACION)
    print("\n[regen] Resumen final de archivos faltantes:")
    for etapa, faltan in estado_faltantes().items():
        if faltan:
            print(f"  {etapa}: {len(faltan)} archivo(s) faltante(s)")
        else:
            print(f"  {etapa}: completo")
else:
    print(
        "REGEN_AUTOMATICO=False: sin regeneracion (todo lo necesario ya esta "
        "en el repo; los 2 archivos opcionales >100 MB usan respaldo "
        "estatico)."
    )
