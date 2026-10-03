"""Features estructurales del texto, sin contenido externo y sin semantica aprendida.

Las tres fuentes de senal que identifico V30-01, implementadas como bloques de features
independientes:

- **URL / dominio**: solo la estructura textual presente en el tweet. No se consulta el
  contenido del dominio ni se usa como tabla de lookup sin control.
- **Mencion / hashtag**: posicion y contexto estructural. No se usa el nombre del usuario ni el
  contenido semantico del hashtag.
- **Enfasis ortografico**: intensidad y repeticion. No se construye un lexicon por palabra.

Reglas comunes
--------------
- Todo se deriva exclusivamente del campo `text`.
- Ninguna feature usa etiquetas, identificadores ni metadatos externos.
- Las posiciones son **relativas** al texto, nunca absolutas.
- Las features son de baja cardinalidad por construccion, para que un modelo lineal pueda
  usarlas sin convertirlas en tabla de lookup.

Uso
---
    from src.structural_features import features_de_url, features_de_mencion, features_de_enfasis
"""

from __future__ import annotations

import re

RE_URL = re.compile(r"(?:https?://|www\.)[^\s]+")
RE_QUERY = re.compile(r"[?&][^\s]+")
RE_FRAGMENT = re.compile(r"#\S+$")
RE_PUERTO = re.compile(r":\d{2,5}(?:/|$)")
RE_PATH = re.compile(r"^https?://[^/?#]+([^?#]*)")
RE_TLD = re.compile(r"\.([a-z]{2,6})(?:[/?#]|$)", re.IGNORECASE)
RE_MENCION = re.compile(r"@\w+")
RE_HASHTAG = re.compile(r"#\w+")
RE_REPETICION = re.compile(r"(.)\1{2,}")
RE_ELONGACION = re.compile(r"(\w)\1{2,}\w*")
RE_EXCLAMACION = re.compile(r"!")
RE_INTERROGACION = re.compile(r"\?")
RE_SIGNO_MIXTO = re.compile(r"[!?]{2,}")
RE_MAYUS_RUN = re.compile(r"[A-Z\u00c0-\u00dc]{2,}")


def _dominio_de(url: str) -> str:
    """Parte del dominio de una URL, sin el esquema.

    Args:
        url: URL completa.

    Returns:
        Dominio en minusculas, o cadena vacia si no se puede extraer.
    """
    sin_esquema = re.sub(r"^(?:https?://|www\.)", "", url)
    return sin_esquema.split("/")[0].split("?")[0].split("#")[0].lower()


def features_de_url(texto: str) -> dict[str, float]:
    """Features estructurales de las URLs presentes en el texto.

    Solo se usa la estructura textual. El dominio completo **no** se devuelve como categoria:
    un dominio con pocas observaciones es una tabla de lookup, y la directiva lo prohibe. Se
    devuelven en cambio la longitud del dominio, el numero de subdominios y el TLD como
    categoria de cardinalidad controlada.

    Args:
        texto: contenido del documento.

    Returns:
        Diccionario con un valor por feature.
    """
    urls = RE_URL.findall(texto)
    n = len(urls)
    if n == 0:
        return {
            "url_n": 0.0,
            "url_longitud_total": 0.0,
            "url_longitud_media": 0.0,
            "url_longitud_max": 0.0,
            "url_con_query": 0.0,
            "url_con_fragment": 0.0,
            "url_profundidad_path": 0.0,
            "url_con_puerto": 0.0,
            "url_longitud_dominio": 0.0,
            "url_subdominios": 0.0,
            "url_proporcion_dominio": 0.0,
            "url_es_corta": 0.0,
            "url_n_parametros": 0.0,
        }
    longitudes = [float(len(u)) for u in urls]
    dominios = [_dominio_de(u) for u in urls]
    caminos = []
    for u in urls:
        encontrado = RE_PATH.search(u)
        caminos.append(encontrado.group(1) if encontrado else "")
    subdominios = [float(max(d.count("."), 0)) for d in dominios]
    tlds = [RE_TLD.search(u).group(1).lower() if RE_TLD.search(u) else "sin_tld" for u in urls]
    n_parametros = [float(len([p for p in re.split(r"[?&]", u)[1:] if p])) for u in urls]
    return {
        "url_n": float(n),
        "url_longitud_total": float(sum(longitudes)),
        "url_longitud_media": float(np_mean(longitudes)),
        "url_longitud_max": float(max(longitudes)),
        "url_con_query": float(any(RE_QUERY.search(u) for u in urls)),
        "url_con_fragment": float(any(RE_FRAGMENT.search(u) for u in urls)),
        "url_profundidad_path": float(np_mean([c.count("/") for c in caminos])),
        "url_con_puerto": float(any(RE_PUERTO.search(u) for u in urls)),
        "url_longitud_dominio": float(np_mean([len(d) for d in dominios])),
        "url_subdominios": float(np_mean(subdominios)),
        "url_proporcion_dominio": float(
            np_mean([len(d) / max(len(u), 1) for d, u in zip(dominios, urls, strict=True)])
        ),
        "url_es_corta": float(np_mean([1.0 if l <= 20 else 0.0 for l in longitudes])),
        "url_n_parametros": float(np_mean(n_parametros)),
        # El TLD como categoria de cardinalidad controlada, nunca el dominio completo.
        "__tld_mas_frecuente": tlds[0] if tlds else "sin_tld",
    }


def features_de_mencion(texto: str) -> dict[str, float]:
    """Features estructurales de menciones y hashtags.

    No se usa el nombre del usuario ni el contenido semantico del hashtag. Solo la posicion
    relativa al texto y el contexto estructural.

    Args:
        texto: contenido del documento.

    Returns:
        Diccionario con un valor por feature.
    """
    n = max(len(texto), 1)
    tokens = texto.split()
    n_tokens = max(len(tokens), 1)
    menciones = list(RE_MENCION.finditer(texto))
    hashtags = list(RE_HASHTAG.finditer(texto))
    pos_mencion = [m.start() / n for m in menciones]
    pos_hashtag = [m.start() / n for m in hashtags]
    return {
        "mencion_existe": float(bool(menciones)),
        "mencion_pos_primera_rel": float(pos_mencion[0]) if pos_mencion else -1.0,
        "mencion_pos_ultima_rel": float(pos_mencion[-1]) if pos_mencion else -1.0,
        "mencion_al_inicio": float(bool(pos_mencion) and pos_mencion[0] < 0.10),
        "mencion_al_final": float(bool(pos_mencion) and pos_mencion[-1] > 0.90),
        "mencion_n": float(len(menciones)),
        "mencion_densidad": len(menciones) / n_tokens,
        "hashtag_existe": float(bool(hashtags)),
        "hashtag_pos_primero_rel": float(pos_hashtag[0]) if pos_hashtag else -1.0,
        "hashtag_pos_ultimo_rel": float(pos_hashtag[-1]) if pos_hashtag else -1.0,
        "hashtag_inicial": float(bool(pos_hashtag) and pos_hashtag[0] < 0.10),
        "hashtag_final": float(bool(pos_hashtag) and pos_hashtag[-1] > 0.90),
        "hashtag_n": float(len(hashtags)),
    }


def features_de_enfasis(texto: str) -> dict[str, float]:
    """Features de enfasis ortografico y repeticion.

    No se construye un lexicon por palabra. Se representan la intensidad y la repeticion como
    patrones de caracteres, que es lo que un modelo lineal puede usar sin memorizar.

    Args:
        texto: contenido del documento.

    Returns:
        Diccionario con un valor por feature.
    """
    max(len(texto), 1)
    tokens = texto.split()
    n_tokens = max(len(tokens), 1)
    repeticiones = RE_REPETICION.findall(texto)
    elongadas = RE_ELONGACION.findall(texto)
    mayus_runs = RE_MAYUS_RUN.findall(texto)
    return {
        "enfasis_n_repeticiones": float(len(repeticiones)),
        "enfasis_long_max_repeticion": float(max((len(r) for r in repeticiones), default=0)),
        "enfasis_palabras_elongadas": float(len(elongadas)),
        "enfasis_ratio_elongadas": len(elongadas) / n_tokens,
        "enfasis_n_exclamaciones": float(len(RE_EXCLAMACION.findall(texto))),
        "enfasis_n_interrogaciones": float(len(RE_INTERROGACION.findall(texto))),
        "enfasis_n_signos_mixtos": float(len(RE_SIGNO_MIXTO.findall(texto))),
        "enfasis_n_mayus_consecutivas": float(len(mayus_runs)),
        "enfasis_ratio_mayus": len(RE_MAYUS_RUN.findall(texto)) / n_tokens,
        "enfasis_long_max_mayus": float(max((len(r) for r in mayus_runs), default=0)),
    }


def np_mean(valores: list[float]) -> float:
    """Media de una lista, con 0.0 en vez de NaN para una lista vacia.

    Args:
        valores: lista de numeros.

    Returns:
        La media, o 0.0 si la lista esta vacia.
    """
    if not valores:
        return 0.0
    return float(sum(valores) / len(valores))


def columnas(bloque: str) -> list[str]:
    """Nombres de las columnas de un bloque, en orden estable.

    Args:
        bloque: uno de ``url``, ``mencion`` o ``enfasis``.

    Returns:
        Lista de nombres de columna.

    Raises:
        ValueError: si el bloque no existe.
    """
    if bloque == "url":
        return [
            "url_n", "url_longitud_total", "url_longitud_media", "url_longitud_max",
            "url_con_query", "url_con_fragment", "url_profundidad_path", "url_con_puerto",
            "url_longitud_dominio", "url_subdominios", "url_proporcion_dominio",
            "url_es_corta", "url_n_parametros",
        ]
    if bloque == "mencion":
        return [
            "mencion_existe", "mencion_pos_primera_rel", "mencion_pos_ultima_rel",
            "mencion_al_inicio", "mencion_al_final", "mencion_n", "mencion_densidad",
            "hashtag_existe", "hashtag_pos_primero_rel", "hashtag_pos_ultimo_rel",
            "hashtag_inicial", "hashtag_final", "hashtag_n",
        ]
    if bloque == "enfasis":
        return [
            "enfasis_n_repeticiones", "enfasis_long_max_repeticion",
            "enfasis_palabras_elongadas", "enfasis_ratio_elongadas",
            "enfasis_n_exclamaciones", "enfasis_n_interrogaciones",
            "enfasis_n_signos_mixtos", "enfasis_n_mayus_consecutivas",
            "enfasis_ratio_mayus", "enfasis_long_max_mayus",
        ]
    raise ValueError(f"bloque desconocido: {bloque}")


def extraer(bloque: str, textos: list[str]) -> list[dict[str, float]]:
    """Extrae las features de un bloque para una lista de textos.

    Args:
        bloque: uno de ``url``, ``mencion`` o ``enfasis``.
        textos: textos a procesar.

    Returns:
        Lista de diccionarios, uno por texto.
    """
    funcion = {"url": features_de_url, "mencion": features_de_mencion,
               "enfasis": features_de_enfasis}[bloque]
    return [funcion(t) for t in textos]


def interacciones(bloque: str, fila: dict[str, float], n_tokens: float) -> dict[str, float]:
    """Interacciones explicitas entre un bloque y la longitud del texto.

    Se codifican como **cruces discretos**, no como productos de variables continuas de alta
    cardinalidad. La longitud se agrupa en tres tramos y la posicion en cuatro, de modo que cada
    interaccion sea una flag binaria con volumen controlado.

    Args:
        bloque: bloque de origen.
        fila: features ya extraidas del bloque.
        n_tokens: numero de tokens del texto, para los tramos de longitud.

    Returns:
        Diccionario con las flags de interaccion.
    """
    tramo = "corto" if n_tokens <= 12 else ("medio" if n_tokens <= 22 else "largo")
    salida: dict[str, float] = {}
    if bloque == "url":
        tiene = fila["url_n"] > 0
        salida["url_x_corto"] = float(tiene and tramo == "corto")
        salida["url_x_medio"] = float(tiene and tramo == "medio")
        salida["url_x_largo"] = float(tiene and tramo == "largo")
        salida["url_query_x_largo"] = float(fila["url_con_query"] > 0 and tramo == "largo")
        salida["url_corta_x_corto"] = float(fila["url_es_corta"] > 0 and tramo == "corto")
    elif bloque == "mencion":
        tiene = fila["mencion_existe"] > 0
        salida["mencion_x_corto"] = float(tiene and tramo == "corto")
        salida["mencion_x_medio"] = float(tiene and tramo == "medio")
        salida["mencion_x_largo"] = float(tiene and tramo == "largo")
        inicio = fila["mencion_al_inicio"] > 0
        final = fila["mencion_al_final"] > 0
        salida["mencion_inicio_x_corto"] = float(inicio and tramo == "corto")
        salida["mencion_final_x_largo"] = float(final and tramo == "largo")
        hashtag = fila["hashtag_existe"] > 0
        salida["hashtag_x_corto"] = float(hashtag and tramo == "corto")
        salida["hashtag_x_medio"] = float(hashtag and tramo == "medio")
        salida["hashtag_x_largo"] = float(hashtag and tramo == "largo")
        salida["hashtag_inicial_x_medio"] = float(
            fila["hashtag_inicial"] > 0 and tramo == "medio"
        )
    elif bloque == "enfasis":
        enfasis = fila["enfasis_n_repeticiones"] + fila["enfasis_n_exclamaciones"]
        salida["enfasis_x_corto"] = float(enfasis > 0 and tramo == "corto")
        salida["enfasis_x_medio"] = float(enfasis > 0 and tramo == "medio")
        salida["enfasis_x_largo"] = float(enfasis > 0 and tramo == "largo")
        salida["elongada_x_largo"] = float(
            fila["enfasis_palabras_elongadas"] > 0 and tramo == "largo"
        )
        salida["mayus_x_corto"] = float(
            fila["enfasis_n_mayus_consecutivas"] > 0 and tramo == "corto"
        )
    else:
        raise ValueError(f"bloque desconocido: {bloque}")
    return {f"_int_{k}": v for k, v in salida.items()}


def columnas_interaccion(bloque: str) -> list[str]:
    """Nombres de las columnas de interaccion de un bloque.

    Args:
        bloque: bloque de origen.

    Returns:
        Lista de nombres de columna.
    """
    return sorted(interacciones(bloque, {c: 0.0 for c in columnas(bloque)}, 15.0).keys())
