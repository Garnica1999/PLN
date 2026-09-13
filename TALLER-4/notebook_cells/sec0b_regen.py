# %% [markdown]
"""# Sec 0b — Regeneracion automatica del corpus dentro del notebook

El notebook es **independiente**: sus unicas dependencias externas son los 3
insumos permitidos, descargados manualmente del Drive del taller a
`data/raw/` (ver README):

1. **Dataset de entrevistas**: `data/raw/entrevistas_all_2023-03-21_14-24_05.json`
   (~231 MB).
2. **Libros**: `data/raw/books-*.zip` (zip del Drive) o la carpeta
   `data/raw/books/` con los PDFs (~300 MB).
3. **Tesauro**: `data/raw/Resource-tesauro.rdf` (solo lo usa el cruce de
   vocabulario del EDA de produccion).

Todo lo demas (segmentos, normalizadas, lemas/tokens/docs, CSVs y figuras
del EDA) son **caches derivados** que esta seccion GENERA automaticamente a
partir de esos insumos con el codigo canonico de los scripts de produccion
(`scripts/01_extraer_libros.py`, `scripts/02_preprocesar.py` y
`scripts/03_eda.py`). Ese codigo viaja **expandido por el ensamblador**
(`scripts/ensamblar_notebook_t4.py`) como **celdas de codigo normales** de
esta seccion: fuente unica en el build, si cambian los scripts se
re-ensambla el notebook y las celdas quedan al dia. El notebook no incrusta
codigo-como-string ni lo ejecuta dinamicamente.

En Run all las celdas expandidas solo **definen** funciones y constantes (la
expansion elimina el bloque `if __name__ == "__main__"` de cada script): no
ejecutan nada por si solas. La regeneracion real la dispara el dispatcher de
la ultima celda de esta seccion: con `REGEN_AUTOMATICO = True` (default)
recorre las etapas con caches faltantes y, si los insumos estan en
`data/raw/`, ejecuta la etapa (primera ejecucion: ~15-60 min: libros ~5-15
min y spaCy ~10-40 min). Si un insumo falta, indica como descargarlo
manualmente y el notebook sigue (las secciones posteriores muestran avisos).
Pon `REGEN_AUTOMATICO = False` para generar etapa por etapa con los
wrappers, o `FUERZAR_REGENERACION = True` para re-ejecutar todo aunque los
caches existan.

Flags de la celda de configuracion:

- `REGEN_AUTOMATICO = True`: al final de la seccion, ejecuta cada etapa con
  archivos faltantes (o todas si `FUERZAR_REGENERACION = True`) solo cuando
  sus insumos estan presentes.
- `FUERZAR_REGENERACION = False`: re-ejecuta la etapa aunque sus archivos ya
  existan (reescribe los derivados; `config/` NUNCA se modifica desde el
  notebook: los scripts de produccion la usan como fuente, pero la config
  curada del notebook esta embebida en sec00).
- `DESCARGAR_FUENTES = False`: permite descargar las fuentes crudas del
  Drive del taller (JSON de entrevistas y zip/carpeta de libros) con gdown
  cuando una etapa las necesita y no estan en `data/raw/`.
- `RUTA_FUENTES = None`: ruta local alternativa con las fuentes ya
  descargadas (carpeta con `entrevistas_all_2023-03-21_14-24_05.json` y el
  zip `books-*.zip` o la carpeta `books/` con los PDFs).

Las 4 etapas, sus insumos de los 3 permitidos y lo que producen:

| Etapa | Codigo expandido | Insumos (de los 3) | Produce |
|---|---|---|---|
| `extraccion` | celdas de 01_extraer_libros.py | libros | `data/corpus/segmentos_libros.jsonl`, `segmentos/<TOMO>.jsonl`, reporte |
| `normalizar` | celdas de 02_preprocesar.py (`normalizar`) | dataset | `data/corpus/entrevistas_normalizadas.jsonl` |
| `preprocesado_spacy` | celdas de 02_preprocesar.py (`spacy`) | segmentos (E1) y normalizadas (E2, desde el dataset) | `data/corpus/preprocesado/*_lemas.txt`, `*_tokens.txt`, `*_docs.json` |
| `eda` | celdas de 03_eda.py (sin "congelar") | lemas (E2); config/ del paquete (solo el EDA oficial) | CSVs de `resultados/eda/` y figuras de `imagenes/` |

Con caches previos (o tras una primera ejecucion completa) todos los caches
derivados ya existen y esta seccion termina en segundos sin regenerar nada:
las secciones 01-03 leen esos caches y calculan inline todo lo demas. La
config de `config/`, `resultados/` e `imagenes/` son OPCIONALES para el
notebook (embebida o recalculada inline); los scripts de produccion (v4)
siguen usandolas como fuente.
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

    Copia autocontenida del algoritmo de sec00 (sin TALLER4_ROOT, para no
    sorprender en ejecuciones sueltas): ancestros
    desde el cwd, subcarpetas de hasta 3 niveles bajo el cwd y bajo cada
    ancestro, y rutas tipicas de Colab. La raiz se reconoce por el cache
    derivado data/corpus/segmentos_libros.jsonl o por cualquiera de los
    3 insumos en data/raw/ (dataset de entrevistas, libros CEV o tesauro),
    de modo que se resuelve antes de generar los caches. Si no encuentra,
    lanza un error claro; nunca devuelve la raiz del filesystem.

    Returns:
        Path absoluto de la raiz del proyecto.

    Raises:
        FileNotFoundError: si no hay data/corpus/segmentos_libros.jsonl ni
            ninguno de los 3 insumos en data/raw/, con instrucciones de
            como subir la carpeta.
    """
    import os as _os

    excluidos = {".git", "drive", "sample_data", "venv", "node_modules"}
    cwd = Path(_os.getcwd()).resolve()

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
        if _es_raiz(ancestro):
            return ancestro
    for ancestro in ancestros:
        for candidato in caminar(ancestro, 3):
            if _es_raiz(candidato):
                return candidato
    for candidata in (Path("/content/TALLER-4"), Path("/content/drive/MyDrive/TALLER-4")):
        if _es_raiz(candidata):
            return candidata
    raise FileNotFoundError(
        "No se encontro la raiz del proyecto TALLER-4: se busca la "
        "carpeta con data/corpus/segmentos_libros.jsonl o alguno de los "
        "3 insumos en data/raw/ (dataset de entrevistas, libros CEV o "
        "tesauro) en el arbol de directorios (sube la carpeta TALLER-4 "
        "completa a /content o descomprime ahi el ZIP del proyecto "
        "(carpeta del taller); se detecta aunque quede en una subcarpeta)."
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
_RUTA_RESULTADOS = RUTA_RAIZ / "resultados"
_RUTA_EDA = _RUTA_RESULTADOS / "eda"
_RUTA_IMAGENES = RUTA_RAIZ / "imagenes"

print(f"[regen] Raiz usada: {_RUTA_RAIZ}")


# %%
"""Configuracion de la seccion de regeneracion (flags).

`REGEN_AUTOMATICO` (True por defecto) regenera en Run all las etapas con
caches derivados faltantes cuando sus insumos de los 3 permitidos estan en
data/raw; `_archivos_etapa()` declara, por etapa, los archivos que produce y
los insumos que necesita (rutas absolutas sobre la raiz detectada).
`insumos_faltantes_etapa()` describe los insumos crudos o caches previos que
faltan (dataset, libros, tesauro o salidas de etapas anteriores) para que el
dispatcher decida si una etapa puede ejecutarse o debe avisar de la descarga
manual. `CENTINELAS_EDA` ordena las subetapas del EDA cuyo archivo
representativo se comprueba (orden canonico de scripts/03_eda.py, sin
"nubes" - solo produce figuras - y sin "congelar", que reescribiria
config/).
"""
import shutil
import subprocess
import sys
from pathlib import Path

REGEN_AUTOMATICO = True
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

NOMBRE_JSON_ENTREVISTAS = "entrevistas_all_2023-03-21_14-24_05.json"
_ZIP_LIBROS_CANONICO = "books-20260904T025322Z-1-001.zip"

# Insumos crudos permitidos (descarga manual del Drive del taller a data/raw).
RUTA_DATASET_CRUDO = _RUTA_RAW / NOMBRE_JSON_ENTREVISTAS
RUTA_TESAURO_CRUDO = _RUTA_RAW / "Resource-tesauro.rdf"


def _libros_crudos_disponibles() -> bool:
    """True si hay zip canonico, cualquier books-*.zip o PDFs en data/raw/books.

    Returns:
        True si la etapa de extraccion puede ejecutarse con lo que hay.
    """
    if (_RUTA_RAW / _ZIP_LIBROS_CANONICO).exists():
        return True
    if sorted(_RUTA_RAW.glob("books-*.zip")):
        return True
    return (_RUTA_RAW / "books").is_dir() and bool(
        sorted((_RUTA_RAW / "books").glob("*.pdf"))
    )


def insumos_faltantes_etapa(nombre_etapa: str) -> list[str]:
    """Describe los insumos faltantes (de los 3 permitidos) de una etapa.

    Args:
        nombre_etapa: "extraccion", "normalizar", "preprocesado_spacy" o
            "eda".

    Returns:
        Lista de mensajes de insumos faltantes; vacia si la etapa puede
        ejecutarse con lo que hay en disco.
    """
    if nombre_etapa == "extraccion":
        if not _libros_crudos_disponibles():
            return [
                ("libros: zip books-*.zip (o la carpeta books/ con los PDFs); "
                "descarga manual del Drive del taller a data/raw/")
            ]
        return []
    if nombre_etapa == "normalizar":
        if not RUTA_DATASET_CRUDO.exists():
            return [
                (f"dataset: {NOMBRE_JSON_ENTREVISTAS}; descarga manual del "
                "Drive del taller a data/raw/")
            ]
        return []
    if nombre_etapa == "preprocesado_spacy":
        mensajes: list[str] = []
        if not (_RUTA_CORPUS / "segmentos_libros.jsonl").exists():
            mensajes.append(
                "segmentos de libros (los genera la etapa extraccion desde "
                "el insumo libros)"
            )
        normalizadas = _RUTA_CORPUS / "entrevistas_normalizadas.jsonl"
        if not normalizadas.exists() and not RUTA_DATASET_CRUDO.exists():
            mensajes.append(
                "dataset para generar entrevistas_normalizadas.jsonl "
                "(descarga manual del Drive del taller a data/raw/)"
            )
        return mensajes
    if nombre_etapa == "eda":
        mensajes = []
        for corpus in ("libros", "entrevistas"):
            if not (_RUTA_PREPROCESADO / f"{corpus}_lemas.txt").exists():
                mensajes.append(
                    f"lemas de {corpus} (los genera la etapa preprocesado_spacy)"
                )
        # El EDA oficial (scripts/03_eda.py) congelo su config en config/; el
        # notebook embebe esa config para sus propios calculos, pero el
        # script de produccion la lee de disco (config/ viaja en el paquete).
        for nombre in ("stopwords_habla.txt", "lemas_correcciones.json", "tfidf_config.json"):
            if not (_RUTA_CONFIG / nombre).exists():
                mensajes.append(
                    f"config/{nombre} (config congelada del EDA de produccion, "
                    "incluida en el paquete)"
                )
        return mensajes
    return ["etapa desconocida"]


def _archivos_etapa() -> dict[str, dict[str, list[Path]]]:
    """Declara los archivos que produce y los insumos de cada etapa.

    Returns:
        Dict {etapa: {"produce": [Path...], "inputs": [Path...]}} con los
        archivos representativos de cada etapa (los que usan las secciones
        01-03 del notebook). Los "inputs" son los insumos crudos o caches
        derivados de otras etapas que el dispatcher comprueba antes de
        ejecutar la etapa.
    """
    return {
        "extraccion": {
            "produce": [
                _RUTA_CORPUS / "segmentos_libros.jsonl",
                _RUTA_RESULTADOS / "extraccion_libros_report.json",
                *[_RUTA_CORPUS / "segmentos" / f"{tomo}.jsonl" for tomo in ORDEN_TOMOS],
            ],
            "inputs": [_RUTA_RAW / _ZIP_LIBROS_CANONICO],
        },
        "normalizar": {
            "produce": [_RUTA_CORPUS / "entrevistas_normalizadas.jsonl"],
            "inputs": [RUTA_DATASET_CRUDO],
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
print(
    "[regen] Insumos de los 3 permitidos en data/raw -> dataset: "
    f"{RUTA_DATASET_CRUDO.exists()} | libros: {_libros_crudos_disponibles()} | "
    f"tesauro: {RUTA_TESAURO_CRUDO.exists()}"
)


# %%
"""Dispatcher de la regeneracion: estado y ejecucion por etapa.

`estado_faltantes()` comprueba contra el disco (sobre la raiz detectada) los
archivos que produce cada etapa y devuelve los faltantes; `regen_etapa()`
ejecuta la etapa correspondiente (los mains expandidos de scripts/01-03 que
definen las celdas de codigo de esta seccion, ver la tabla de arriba) solo
si faltan archivos (o si `forzar=True`) y descarga las fuentes crudas antes
cuando hacen falta y estan habilitadas. Si las celdas de codigo de la
seccion no se han ejecutado, los wrappers avisan con un mensaje claro.
"""

MENSAJE_SIN_CELDAS_EJECUTADAS = (
    "Ejecuta primero las celdas de codigo de esta seccion (o Run all) antes "
    "de activar la regeneracion."
)


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
# NOMBRE_JSON_ENTREVISTAS ya se define en la celda de configuracion (flags).


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
scripts/01_extraer_libros.py, expandido por el ensamblador).

La celda marcada abajo se convierte en el ensamblado en las celdas de codigo
del script (partidas por sus banners de seccion, sin el bloque de '__main__'
y con '__file__' sintetico) mas la celda de alias `main_extraccion`. En Run
all esas celdas solo definen; la regeneracion real la dispara el wrapper de
la celda siguiente cuando faltan archivos (o con `forzar`).
"""

# Nombres que definen las celdas expandidas por el ensamblador (v4). Aqui se
# declaran como None para el analisis estatico y como valor inicial seguro:
# las celdas expandidas (ejecutadas despues de esta celda en Run all)
# sobrescriben el alias con la funcion real antes de cualquier uso.
main_extraccion = None  # [auto] lo define la expansion de scripts/01 (alias)
reconstruir_global = None  # [auto] funcion del propio scripts/01


# %%
# %% [expand: scripts/01_extraer_libros.py]


# %%
"""Wrappers de la etapa E1 (ver el docstring de E1 unas celdas arriba).

Prerrequisitos antes de ejecutar `main_extraccion()`: binario `pdftotext`
(poppler-utils; si falta se intenta instalar con apt-get en Colab) y el zip
de libros en data/raw (o los PDFs en data/raw/books, con los que el wrapper
construye el zip canonico). El script usa argparse sobre sys.argv en su
main(), asi que el wrapper parchea sys.argv temporalmente (try/finally) con
los tomos faltantes y restaura el argv original al terminar.
"""


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
    """Regenera la extraccion de segmentos de los libros (script 01 expandido).

    Args:
        forzar: True re-extrae los tomos aunque sus JSONL existan (--force).

    Returns:
        True si la etapa quedo completa (o no habia nada que hacer); False
        si faltan prerrequisitos (pdftotext, zip/PDFs) o las celdas de
        codigo de la seccion no se han ejecutado.
    """
    if not callable(globals().get("main_extraccion")) or not callable(
        globals().get("reconstruir_global")
    ):
        print(f"[E1] {MENSAJE_SIN_CELDAS_EJECUTADAS}")
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

    # argv del script: prog + (tomos faltantes) + (--force si forzar).
    faltantes = [
        tomo
        for tomo in ORDEN_TOMOS
        if not (_RUTA_CORPUS / "segmentos" / f"{tomo}.jsonl").exists()
    ]
    argv = ["regen_01"] + (["--libros", *faltantes] if faltantes else [])
    if forzar:
        argv.append("--force")
    argv_previo = list(sys.argv)
    try:
        sys.argv = argv
        main_extraccion()
    except SystemExit:
        pass  # el script usa sys.exit para errores de argumentos
    finally:
        sys.argv = argv_previo
    # Si no habia tomos que procesar pero falta el archivo global, se
    # reconstruye desde los JSONL por tomo (funcion del propio script).
    if not (_RUTA_CORPUS / "segmentos_libros.jsonl").exists():
        reconstruir_global()
    print("[E1] Etapa de extraccion terminada.")
    return True


# %%
"""Etapa E2: normalizacion estructural y preprocesado spaCy (codigo canonico
de scripts/02_preprocesar.py, expandido por el ensamblador).

Dos subetapas: `normalizar` (JSON crudo de entrevistas ->
entrevistas_normalizadas.jsonl, solo si el crudo existe en data/raw) y
`spacy --corpus ambos` (segmentos_libros.jsonl + normalizadas -> lemas,
tokens y docs de libros y entrevistas). El script fija mp fork
internamente; los workers se limitan a N_PROCESOS_SPACY (4) para Colab. El
modelo es_core_news_md debe estar descargado; si falta se intenta con
`python -m spacy download`.
"""

# Alias [auto] que define la expansion de scripts/02 (ver nota en E1).
main_preprocesar = None  # [auto] lo define la expansion de scripts/02 (alias)


# %%
# %% [expand: scripts/02_preprocesar.py]


# %%
"""Wrappers de la etapa E2 (ver el docstring de E2 unas celdas arriba).

`main_preprocesar(argv)` es el alias [auto] que deja la expansion de
scripts/02_preprocesar.py al final de su bloque; los wrappers lo invocan con
la lista de argumentos del subcomando. Si las celdas de codigo de la seccion
no se ejecutaron, avisan con un mensaje claro.
"""


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


def _main_preprocesar_definido() -> bool:
    """True si las celdas expandidas de scripts/02 ya definieron el alias.

    Returns:
        True si `main_preprocesar` es llamable en el kernel; si no, imprime
        el mensaje de celdas no ejecutadas y devuelve False.
    """
    if not callable(globals().get("main_preprocesar")):
        print(f"[E2] {MENSAJE_SIN_CELDAS_EJECUTADAS}")
        return False
    return True


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
    if not _main_preprocesar_definido():
        return False
    codigo = main_preprocesar(["normalizar"] + (["--force"] if forzar else []))
    ok = codigo == 0
    print("[E2] Subetapa normalizar terminada." if ok else "[E2] Error en normalizar.")
    return ok


def regen_preprocesado_spacy(
    forzar: bool = False,
    procesos: int = N_PROCESOS_SPACY,
    corpus: str = "ambos",
) -> bool:
    """Regenera lemas/tokens/docs con spaCy (subconjunto por corpus).

    El wrapper decide el subconjunto real a procesar: "libros" requiere
    data/corpus/segmentos_libros.jsonl (salida de la etapa extraccion);
    "entrevistas" requiere entrevistas_normalizadas.jsonl (que se intenta
    normalizar antes si el dataset crudo esta en data/raw). Si el insumo de
    un subconjunto pedido falta, se omite con aviso (sin excepcion) y el
    resto se procesa igual.

    Args:
        forzar: True ignora los checkpoints y reprocesa desde cero.
        procesos: Workers de spaCy (default 4, pensado para Colab).
        corpus: Subconjunto pedido: "ambos" (default), "libros" o
            "entrevistas".

    Returns:
        True si la etapa quedo completa o no requeria accion; False si no
        quedo ningun subconjunto ejecutable o fallo el modelo.
    """
    ruta_crudo_ent = _RUTA_RAW / NOMBRE_JSON_ENTREVISTAS
    ruta_normalizadas = _RUTA_CORPUS / "entrevistas_normalizadas.jsonl"
    candidatos = (
        ["libros", "entrevistas"]
        if corpus == "ambos"
        else [corpus]
        if corpus in ("libros", "entrevistas")
        else []
    )
    ejecutar: list[str] = []
    for nombre_corpus in candidatos:
        if nombre_corpus == "libros":
            if not (_RUTA_CORPUS / "segmentos_libros.jsonl").exists():
                print(
                    "[E2] Falta data/corpus/segmentos_libros.jsonl: ejecuta "
                    "antes la etapa extraccion (insumo libros en data/raw)."
                )
                continue
            ejecutar.append(nombre_corpus)
        else:
            if not ruta_normalizadas.exists() and ruta_crudo_ent.exists():
                print("[E2] Falta entrevistas_normalizadas.jsonl; normalizando primero ...")
                regen_normalizar_entrevistas(forzar=False)
            if not ruta_normalizadas.exists():
                print(
                    "[E2] Falta entrevistas_normalizadas.jsonl (no esta el "
                    "JSON crudo en data/raw): la subetapa de entrevistas se "
                    "omite."
                )
                continue
            ejecutar.append(nombre_corpus)
    if not ejecutar:
        return False
    if not _verificar_modelo_spacy():
        return False
    if not _main_preprocesar_definido():
        return False
    corpus_efectivo = "ambos" if len(ejecutar) == 2 else ejecutar[0]
    codigo = main_preprocesar(
        ["spacy", "--corpus", corpus_efectivo, "--procesos", str(procesos)]
        + (["--force"] if forzar else [])
    )
    ok = codigo == 0
    print("[E2] Subetapa spacy terminada." if ok else "[E2] Error en spacy.")
    return ok


# %%
"""Etapa E3: EDA (codigo canonico de scripts/03_eda.py, expandido por el
ensamblador), ejecutado SIN la subetapa "congelar": config/ nunca se
reescribe desde el notebook.

El script fija matplotlib con backend "Agg" a nivel de modulo; la celda de
alias [auto] que cierra su bloque (y el wrapper de abajo al terminar)
restauran el backend inline del notebook. Las subetapas se eligen como la
cadena canonica desde la primera con archivo faltante (estadisticas ->
frecuencias -> ... -> tesauro), porque cada etapa requiere en contexto las
anteriores. "tesauro" solo se incluye si su cache o los CSVs posteriores
faltan (sin data/raw/Resource-tesauro.rdf la propia etapa avisa y omite el
cruce).
"""

# Alias [auto] que define la expansion de scripts/03 (ver nota en E1).
main_eda = None  # [auto] lo define la expansion de scripts/03 (alias)


# %%
# %% [expand: scripts/03_eda.py]


# %%
"""Wrappers de la etapa E3 (ver el docstring de E3 unas celdas arriba).

`main_eda(argv)` es el alias [auto] de la expansion de scripts/03_eda.py;
recibe la lista de argumentos reales (sin nombre de programa) y devuelve el
codigo de salida.
"""

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
    """Regenera los CSVs y figuras del EDA (script 03 expandido, sin congelar).

    Args:
        forzar: True re-ejecuta toda la cadena aunque no falte nada.

    Returns:
        True si la etapa quedo completa o no requeria accion; False si fallo
        o las celdas de codigo de la seccion no se han ejecutado.
    """
    if not callable(globals().get("main_eda")):
        print(f"[E3] {MENSAJE_SIN_CELDAS_EJECUTADAS}")
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

    try:
        codigo = main_eda(["--solo", *etapas])
    finally:
        _restaurar_backend_inline()
    print("[E3] Etapa EDA terminada." if codigo == 0 else "[E3] Error en el EDA.")
    return codigo == 0


# %%
"""Ejecucion de la seccion: regeneracion automatica al final de Run all.

Con `REGEN_AUTOMATICO = True` (default) se recorren las etapas con salidas
faltantes (o todas si `FUERZAR_REGENERACION = True`):

- si los insumos estan presentes (`insumos_faltantes_etapa()` vacio) la
  etapa se ejecuta, con mensaje de tiempo estimado por consola;
- si falta un insumo de los 3 permitidos se imprime un mensaje claro de
  descarga manual (Drive del taller a data/raw/) y la etapa queda
  pendiente: el notebook NO lanza excepcion y las secciones 01-03 muestran
  avisos suaves cuando un archivo requerido no existe.

Para `preprocesado_spacy` el disparo es por subconjunto de corpus: se
regeneran solo los corpus (libros/entrevistas) con caches spaCy faltantes
cuyos insumos existan (segmentos para libros; normalizadas/dataset para
entrevistas).
"""
TIEMPOS_ESTIMADOS_ETAPAS = {
    "extraccion": "~5-15 min (10 tomos PDF con pdftotext + PyMuPDF)",
    "normalizar": "~2-5 min (dataset de entrevistas ~231 MB)",
    "preprocesado_spacy": "~10-40 min (spaCy es_core_news_md, 4 procesos)",
    "eda": "~2-5 min (CSVs del EDA y figuras)",
}


def _salidas_spacy_faltantes_por_corpus() -> list[str]:
    """Corpus (libros/entrevistas) con caches spaCy faltantes.

    Returns:
        Lista de corpus con al menos una salida de la etapa spacy
        (lemas/tokens/docs) faltante en disco.
    """
    pendientes: list[str] = []
    for nombre_corpus in ("libros", "entrevistas"):
        salidas = [
            _RUTA_PREPROCESADO / f"{nombre_corpus}_{tipo}.txt"
            for tipo in ("lemas", "tokens")
        ] + [_RUTA_PREPROCESADO / f"{nombre_corpus}_docs.json"]
        if any(not ruta.exists() for ruta in salidas):
            pendientes.append(nombre_corpus)
    return pendientes


def _bloqueantes_spacy_corpus(nombre_corpus: str) -> list[str]:
    """Insumos faltantes para el subconjunto spacy de un corpus.

    Args:
        nombre_corpus: "libros" o "entrevistas".

    Returns:
        Mensajes de insumo faltante (lista vacia = ejecutable).
    """
    if nombre_corpus == "libros":
        if not (_RUTA_CORPUS / "segmentos_libros.jsonl").exists():
            return [
                ("segmentos de libros (los genera la etapa extraccion desde "
                "el insumo libros; descarga manual del Drive del taller a "
                "data/raw/)")
            ]
        return []
    normalizadas = _RUTA_CORPUS / "entrevistas_normalizadas.jsonl"
    if not normalizadas.exists() and not RUTA_DATASET_CRUDO.exists():
        return [
            (f"dataset {NOMBRE_JSON_ENTREVISTAS} para generar "
            "entrevistas_normalizadas.jsonl (descarga manual del Drive del "
            "taller a data/raw/)")
        ]
    return []


if REGEN_AUTOMATICO:
    print("\n[regen] REGEN_AUTOMATICO=True: revisando caches derivados faltantes ...")
    faltantes_global = estado_faltantes()
    n_etapas_ejecutadas = 0
    n_etapas_pendientes = 0
    for etapa, faltan in faltantes_global.items():
        if not faltan and not FUERZAR_REGENERACION:
            continue
        if etapa == "preprocesado_spacy":
            for nombre_corpus in _salidas_spacy_faltantes_por_corpus():
                bloqueantes = _bloqueantes_spacy_corpus(nombre_corpus)
                if bloqueantes:
                    n_etapas_pendientes += 1
                    print(
                        f"\n[regen] Etapa {etapa} (corpus {nombre_corpus}) "
                        "NO ejecutada: para generar "
                        f"{nombre_corpus}_lemas/tokens.txt y "
                        f"{nombre_corpus}_docs.json se necesita: "
                        f"{'; '.join(bloqueantes)}"
                    )
                    print(
                        "        (descarga manual desde el Drive del taller; "
                        "el notebook sigue y las secciones posteriores "
                        "muestran avisos)"
                    )
                    continue
                print(
                    f"\n[regen] Ejecutando etapa {etapa} (corpus "
                    f"{nombre_corpus}) ... tiempo estimado: "
                    f"{TIEMPOS_ESTIMADOS_ETAPAS[etapa]}"
                )
                ok = regen_preprocesado_spacy(
                    forzar=FUERZAR_REGENERACION, corpus=nombre_corpus
                )
                if ok:
                    n_etapas_ejecutadas += 1
            continue
        bloqueantes = insumos_faltantes_etapa(etapa)
        if bloqueantes:
            n_etapas_pendientes += 1
            salidas = ", ".join(sorted({p.name for p in faltan})) or "(todas)"
            print(
                f"\n[regen] Etapa {etapa} NO ejecutada: para generar "
                f"{salidas} se necesita: {'; '.join(bloqueantes)}"
            )
            print(
                "        (descarga manual desde el Drive del taller a "
                "data/raw/ y vuelve a ejecutar Run all; el notebook sigue y "
                "las secciones posteriores muestran avisos)"
            )
            continue
        print(
            f"\n[regen] Ejecutando etapa {etapa} ... tiempo estimado: "
            f"{TIEMPOS_ESTIMADOS_ETAPAS[etapa]}"
        )
        ok = regen_etapa(etapa, forzar=FUERZAR_REGENERACION)
        if ok:
            n_etapas_ejecutadas += 1

    print("\n[regen] Resumen final de archivos faltantes:")
    for etapa, faltan in estado_faltantes().items():
        if faltan:
            print(f"  {etapa}: {len(faltan)} archivo(s) faltante(s)")
        else:
            print(f"  {etapa}: completo")
    print(
        f"[regen] Etapas ejecutadas: {n_etapas_ejecutadas} | etapas "
        f"pendientes por insumos (descarga manual): {n_etapas_pendientes}"
    )
else:
    print(
        "REGEN_AUTOMATICO=False: sin regeneracion automatica (los caches "
        "derivados presentes se usan tal cual; ejecuta los wrappers regen_* "
        "a mano para regenerar etapa por etapa)."
    )
