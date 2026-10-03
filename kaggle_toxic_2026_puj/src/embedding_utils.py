"""Utilidades de la fase v5: normalizacion, tokenizacion y acceso a fastText.

Alcance
-------
Este modulo concentra lo que comparten la construccion del corpus (P1) y el
entrenamiento y la evaluacion del embedding (P2 en adelante):

- la normalizacion **EMB-NORMAL** y el tokenizador identicos a los del corpus,
- el plegado a ASCII y el esqueleto vocalico usados en el diagnostico de
  cobertura,
- el envoltorio del binario de fastText C++ (entrenamiento y obtencion de
  vectores, incluidos los compuestos por subpalabras para tipos fuera de
  vocabulario).

Nota de trazabilidad
--------------------
``experiments/embedding_01_corpus_wikipedia_v5.py`` (P1) conserva su propia copia
de estas funciones porque es evidencia historica de la construccion del corpus y
**no se modifica**. ``experiments/embedding_02_fasttext_baseline_v5.py``
comprueba de forma explicita que ambas implementaciones coinciden sobre los 2.885
textos de TOXIC antes de usarlas, de modo que la duplicacion queda verificada y
no simplemente asumida. Si alguna vez divergieran, ese chequeo falla.
"""

from __future__ import annotations

import re
import subprocess
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np

VOCALES = frozenset("aeiouáéíóúüAEIOUÁÉÍÓÚÜ")

RE_COMENTARIO = re.compile(r"<!--.*?-->", re.DOTALL)
RE_REF = re.compile(r"<ref[^>]*?/>|<ref[^>]*?>.*?</ref>", re.DOTALL | re.IGNORECASE)
RE_ETIQUETA_HTML = re.compile(r"<[^>]{1,200}>")
RE_PLANTILLA = re.compile(r"\{\{.*?\}\}", re.DOTALL)
RE_TABLA = re.compile(r"\{\|.*?\|\}", re.DOTALL)
RE_ENLACE_ARCHIVO = re.compile(
    r"\[\[\s*(?:Archivo|File|Imagen|Image|Categor[ií]a|Category|Wikipedia|Plantilla|Template)\s*:[^\]]*\]\]",
    re.IGNORECASE,
)
RE_ENLACE_INTERNO = re.compile(r"\[\[\s*([^\[\]|]*)\s*\|\s*([^\[\]|]*?)\s*\]\]")
RE_ENLACE_SIMPLE = re.compile(r"\[\[\s*([^\[\]|]*?)\s*\]\]")
RE_ENLACE_EXTERNO = re.compile(r"\[(?:https?|ftp)://[^\s\]]+\s*([^\]]*)\]")
RE_URL_SIMPLE = re.compile(r"(?:https?|ftp)://\S+")
RE_MARCAS = re.compile(r"'{2,5}")
RE_ESPACIOS = re.compile(r"\s+")
RE_CARACTERES_PERMITIDOS = re.compile(r"[^\w\s']", re.UNICODE)
RE_ENTIDADES = re.compile(r"&[a-zA-Z#0-9]{2,8};")
RE_URL = re.compile(r"https?://\S+|www\.\S+")
RE_MENCION = re.compile(r"@\w+")
RE_HASHTAG = re.compile(r"#\w+")
RE_NUMERO = re.compile(r"\d+")
# El apostrofo solo puede ser interno: evita tipos basura como `'` o `''es`.
RE_TOKEN = re.compile(r"\w+(?:['’]\w+)*", re.UNICODE)


def limpiar_marcado(marcado: str) -> str:
    """Elimina el marcado MediaWiki y conserva el texto plano.

    Args:
        marcado: texto original del campo ``text`` de una revision.

    Returns:
        Texto plano con una linea por parrafo.
    """
    texto = RE_COMENTARIO.sub(" ", marcado)
    texto = RE_REF.sub(" ", texto)
    texto = RE_ETIQUETA_HTML.sub(" ", texto)
    texto = RE_TABLA.sub(" ", texto)
    for _ in range(8):
        nuevo = RE_PLANTILLA.sub(" ", texto)
        if nuevo == texto:
            break
        texto = nuevo
    texto = RE_ENLACE_ARCHIVO.sub(" ", texto)
    texto = RE_ENLACE_EXTERNO.sub(r" \1 ", texto)
    texto = RE_ENLACE_INTERNO.sub(r" \2 ", texto)
    texto = RE_ENLACE_SIMPLE.sub(r" \1 ", texto)
    texto = RE_URL_SIMPLE.sub(" url ", texto)
    texto = RE_MARCAS.sub("", texto)
    texto = RE_ENTIDADES.sub(" ", texto)
    lineas = []
    for linea in texto.splitlines():
        limpia = linea.strip()
        if not limpia or limpia.startswith("="):
            continue
        limpia = limpia.lstrip("*#:; ")
        limpia = RE_CARACTERES_PERMITIDOS.sub(" ", limpia)
        limpia = RE_ESPACIOS.sub(" ", limpia).strip()
        if limpia:
            lineas.append(limpia)
    return "\n".join(lineas)


def plegar_ascii(texto: str) -> str:
    """Elimina los diacriticos sin alterar el resto de caracteres.

    Define la variante **EMB-ASCII** de la seccion 16 de la spec.

    Args:
        texto: texto de entrada.

    Returns:
        Texto sin marcas diacriticas.
    """
    descompuesto = unicodedata.normalize("NFD", texto)
    return "".join(caracter for caracter in descompuesto if unicodedata.category(caracter) != "Mn")


def esqueleto_vocalico(texto: str) -> str:
    """Elimina las vocales para comparar formas ortograficamente deformadas.

    Args:
        texto: palabra de entrada.

    Returns:
        Palabra sin vocales, en minusculas y sin diacriticos.
    """
    plegada = plegar_ascii(texto).lower()
    return "".join(caracter for caracter in plegada if caracter not in VOCALES)


def normalizar_emb_normal(texto: str) -> str:
    """Aplica la normalizacion EMB-NORMAL.

    Minusculas conservando tildes, URLs, menciones, hashtags y numeros reducidos
    a marcadores estables. **Sin** corrector ortografico ni reconstruccion de
    tildes.

    Args:
        texto: texto de entrada.

    Returns:
        Texto normalizado.
    """
    texto = texto.lower()
    texto = RE_URL.sub(" url ", texto)
    texto = RE_HASHTAG.sub(" hashtag ", texto)
    texto = RE_MENCION.sub(" usuario ", texto)
    texto = RE_NUMERO.sub(" numero ", texto)
    return RE_ESPACIOS.sub(" ", texto).strip()


def normalizar_emb_ascii(texto: str) -> str:
    """Aplica la normalizacion EMB-ASCII (variante de la seccion 16).

    Args:
        texto: texto de entrada.

    Returns:
        Texto en minusculas y sin diacriticos.
    """
    return normalizar_emb_normal(plegar_ascii(texto))


def tokenizar(texto: str) -> list[str]:
    """Tokeniza con el criterio usado para construir el corpus.

    Args:
        texto: texto normalizado.

    Returns:
        Lista de tokens.
    """
    return RE_TOKEN.findall(texto)


def vocabulario_textos(textos: list[str]) -> Counter[str]:
    """Cuenta los tipos de una lista de textos ya normalizados.

    Args:
        textos: textos normalizados.

    Returns:
        ``Counter`` tipo -> frecuencia.
    """
    vocabulario: Counter[str] = Counter()
    for texto in textos:
        vocabulario.update(tokenizar(texto))
    return vocabulario


# ---------------------------------------------------------------------------
# Envoltorio de fastText C++
# ---------------------------------------------------------------------------
def sha256_de(ruta: Path) -> str:
    """Calcula el sha256 de un archivo.

    Es la huella que se registra en el manifiesto de cada modelo de la fase
    (seccion 36 de la spec): identifica el `.vec` de forma reproducible.

    Args:
        ruta: ruta del archivo.

    Returns:
        Hash hexadecimal.
    """
    import hashlib

    digest = hashlib.sha256()
    with ruta.open("rb") as archivo:
        for bloque in iter(lambda: archivo.read(1 << 20), b""):
            digest.update(bloque)
    return digest.hexdigest()


def entrenar_fasttext(
    binario: Path,
    corpus: Path,
    prefijo: Path,
    *,
    dim: int,
    ventana: int = 5,
    min_count: int = 3,
    negativos: int = 10,
    epocas: int = 5,
    hilos: int = 16,
    semilla: int = 42,
    minn: int = 3,
    maxn: int = 6,
    bucket: int = 200_000,
) -> dict[str, object]:
    """Entrena un modelo de fastText C++ desde cero y devuelve sus parametros.

    Args:
        binario: ruta del ejecutable ``fasttext``.
        corpus: archivo de entrenamiento (un documento por linea).
        prefijo: prefijo de salida (genera ``.bin`` y ``.vec``).
        dim: dimension de los vectores.
        ventana: tamano de la ventana de contexto.
        min_count: frecuencia minima para entrar en el vocabulario.
        negativos: numero de muestras negativas.
        epocas: numero de épocas.
        hilos: hilos de entrenamiento.
        semilla: semilla de inicializacion.
        minn: longitud minima de subpalabra (0 desactiva los subpalabras).
        maxn: longitud maxima de subpalabra (0 desactiva los subpalabras).
        bucket: numero de cubos de subpalabras.

    Returns:
        Diccionario con la linea de comando usada y los ficheros generados.
    """
    comando = [
        str(binario),
        "skipgram",
        "-input", str(corpus),
        "-output", str(prefijo),
        "-dim", str(dim),
        "-ws", str(ventana),
        "-minCount", str(min_count),
        "-neg", str(negativos),
        "-epoch", str(epocas),
        "-thread", str(hilos),
        "-seed", str(semilla),
        "-minn", str(minn),
        "-maxn", str(maxn),
        "-bucket", str(bucket),
        "-verbose", "0",
    ]
    subprocess.run(comando, check=True, capture_output=True)
    return {
        "comando": " ".join(comando[1:]),
        "bin": f"{prefijo}.bin",
        "vec": f"{prefijo}.vec",
        "subpalabras": minn > 0 and maxn > 0,
    }


def vectores_de_tipos(binario: Path, modelo: Path, tipos: list[str], salida: Path) -> dict[str, np.ndarray]:
    """Pide a fastText los vectores de una lista de tipos.

    Para un tipo **fuera del vocabulario**, fastText compone el vector con sus
    subpalabras; si el modelo se entreno con ``minn=0, maxn=0`` el vector sale
    nulo, que es precisamente la diferencia que se quiere medir entre FastText y
    el control Word2Vec.

    Args:
        binario: ruta del ejecutable ``fasttext``.
        modelo: ruta del modelo ``.bin``.
        tipos: tipos a consultar.
        salida: archivo donde se escriben los vectores.

    Returns:
        Diccionario tipo -> vector (``float32``), sin vectores nulos.
    """
    salida.parent.mkdir(parents=True, exist_ok=True)
    entrada = salida.with_suffix(".entrada.txt")
    entrada.write_text("\n".join(tipos) + "\n", encoding="utf-8")
    with entrada.open("rb") as origen, salida.open("wb") as destino:
        subprocess.run(
            [str(binario), "print-word-vectors", str(modelo)],
            stdin=origen,
            stdout=destino,
            check=True,
            stderr=subprocess.DEVNULL,
        )
    vectores: dict[str, np.ndarray] = {}
    with salida.open(encoding="utf-8", errors="replace") as archivo:
        for linea in archivo:
            partes = linea.split()
            if len(partes) < 2:
                continue
            valores = np.asarray(partes[1:], dtype=np.float32)
            if not np.isfinite(valores).all() or float(np.linalg.norm(valores)) == 0.0:
                continue
            vectores[partes[0]] = valores
    return vectores


def cargar_matriz_vec(ruta_vec: Path) -> tuple[dict[str, int], np.ndarray]:
    """Carga un archivo ``.vec`` de fastText en una matriz normalizada.

    Args:
        ruta_vec: archivo ``.vec``.

    Returns:
        Tupla ``(indice, matriz)`` con filas normalizadas.
    """
    palabras: list[str] = []
    filas: list[np.ndarray] = []
    with ruta_vec.open(encoding="utf-8", errors="replace") as archivo:
        archivo.readline()
        for linea in archivo:
            partes = linea.split()
            if len(partes) < 3:
                continue
            valores = np.asarray(partes[1:], dtype=np.float32)
            if not np.isfinite(valores).all():
                continue
            norma = float(np.linalg.norm(valores))
            if norma == 0.0:
                continue
            palabras.append(partes[0])
            filas.append(valores / norma)
    indice = {palabra: posicion for posicion, palabra in enumerate(palabras)}
    matriz = np.vstack(filas) if filas else np.empty((0, 1), dtype=np.float32)
    return indice, matriz


def centrar(matriz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Resta la media de la matriz y renormaliza.

    Los vectores de fastText comparten un componente comun grande; sin centrar,
    el coseno entre cualesquiera dos vectores sale alto y el vecino mas cercano
    no significa nada.

    Args:
        matriz: matriz de vectores ya normalizados.

    Returns:
        Tupla ``(centro, matriz_centrada)``.
    """
    centro = matriz.mean(axis=0)
    centrada = matriz - centro
    norms = np.linalg.norm(centrada, axis=1, keepdims=True)
    return centro, centrada / np.maximum(norms, 1e-9)
