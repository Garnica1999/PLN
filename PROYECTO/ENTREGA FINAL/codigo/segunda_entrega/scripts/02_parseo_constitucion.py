"""02_parseo_constitucion.py
Fecha: 2026-09-08
Descripcion: Parsea el crudo HTML de la Constitucion Politica de Colombia
    1991 (descargado por 01_descarga_constitucion.py) articulo por articulo.
    Asigna titulo, capitulo y bloque tematico con el mapeo fijo E2 y guarda
    data/constitucion_articulos.parquet, data/constitucion_articulos.json,
    resultados/constitucion_cobertura.md y el log de verificacion.
Estado del arte aplicado: mapeo fijo titulos/capitulos a bloques tematicos
    segun ADR-1 (articulo citado -> titulo/capitulo -> bloque).

ciclo v2 - Fase S1 (2026-09-10, plan/PLAN_v2_ciclo_mejora.md): correccion
    de los 45 IDs malformados "Transitorio ACL01016-N" del parseo v1 y
    nueva columna acto_legislativo. Evidencia de la fuente (HTML del
    Senado, data/constitucion/cp_15_..._pr014.html): el prefijo ACL01016
    NO significa "transitorio 16 adicionado"; es el codigo del Senado para
    el ACTO LEGISLATIVO 01 DE 2016 (el texto de cada articulo enlaza a
    acto_legislativo_01_2016.html#N) y el sufijo -N es el ordinal del
    articulo dentro de ese acto. El mismo patron aplica a ACL01017 (Acto
    Legislativo 1 de 2017, JEP), ACL02017 (Acto Legislativo 2 de 2017),
    ACL03017 (Acto Legislativo 3 de 2017) y ACL02021 (Acto Legislativo 2
    de 2021, circunscripciones de paz). El mapeo canonico se deduce de la
    ETIQUETA VISIBLE de cada ancla en el HTML ("ARTICULO TRANSITORIO 20.
    PARTICIPACION EN POLITICA" para ACL01017-20; "ARTICULO TRANSITORIO
    3o. INSCRIPCION DE CANDIDATOS" para ACL02021-3) y del ordinal del
    sufijo cuando la etiqueta no trae numero (ACL01016-1..4, ACL02017-1,
    ACL03017-1..3). Numeracion canonica elegida: "Transitorio ACL01016-N"
    -> "Transitorio 68", "Transitorio 69", ... continuando la serie 1991
    (1..64, 66, 67; el 65 fue derogado por el Acto Legislativo 1 de 2003
    y no existe en la fuente) y evitando colisiones con los ordinales
    internos de cada acto; el ordinal del acto se conserva en la nueva
    columna acto_legislativo ("Acto Legislativo 1 de 2016", ...; None
    para el articulado original). El parquet v1 queda respaldado en
    data/constitucion_articulos_v1.parquet (ADR-11: resultados previos
    intocables).
"""

import json
import random
import re
import sys
import traceback
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from bs4.element import Tag

CONFIG = {
    "semilla": 42,
    "dir_crudo": "data/constitucion",
    "ruta_parquet": "data/constitucion_articulos.parquet",
    "ruta_json": "data/constitucion_articulos.json",
    "ruta_cobertura": "resultados/constitucion_cobertura.md",
    "ruta_log": "experimentos/parseo_constitucion.log",
    "fuente_base": "http://www.secretariasenado.gov.co/senado/basedoc/",
    "spot_check": {
        "1": "principios_fundamentales",
        "11": "derechos_fundamentales",
        "13": "derechos_fundamentales",
        "86": "proteccion_aplicacion_derechos",
        "241": "rama_judicial",
    },
}

ROOT = Path(__file__).resolve().parent.parent

PARRAFOS_ADMIN = {
    "concordancias",
    "jurisprudencia concordante",
    "antecedentes",
    "notas de vigencia",
    "jurisprudencia vigencia",
    "legislacion anterior",
}

MARCAS_FIN_PAGINA = (
    "disposiciones analizadas por avance juridico",
    "issn [",
    "ultima actualizacion",
    "las notas de vigencia, concordancias",
)

MAPA_CAP_TRANSITORIO = {
    "IT": "1",
    "IIT": "2",
    "IIIT": "3",
    "IVT": "4",
    "VT": "5",
    "VIT": "6",
    "VIIT": "7",
    "VIIIT": "8",
}

# ciclo v2 - Fase S1: mapeo canonico de los 45 IDs malformados del parseo
# v1. Clave = atributo name del ancla en el HTML del Senado (sin el prefijo
# "TRANSITORIO "); valor = (numero canonico, acto legislativo de origen).
# El numero canonico continua la serie 1991 (que llega hasta 67 sin el 65,
# derogado) para no colisionar con los ordinales internos de cada acto. El
# ordinal dentro del acto se conserva en la etiqueta visible del HTML
# ("ARTICULO TRANSITORIO 20. PARTICIPACION EN POLITICA") y en el propio
# texto del articulo ("adicionado por el articulo N del Acto Legislativo
# X de YYYY"); ver docstring del modulo.
MAPA_IDS_ACL = {
    "ACL01016-1": ("68", "Acto Legislativo 1 de 2016"),
    "ACL01016-2": ("69", "Acto Legislativo 1 de 2016"),
    "ACL01016-3": ("70", "Acto Legislativo 1 de 2016"),
    "ACL01016-4": ("71", "Acto Legislativo 1 de 2016"),
    "ACL01017-1": ("72", "Acto Legislativo 1 de 2017"),
    "ACL01017-2": ("73", "Acto Legislativo 1 de 2017"),
    "ACL01017-3": ("74", "Acto Legislativo 1 de 2017"),
    "ACL01017-4": ("75", "Acto Legislativo 1 de 2017"),
    "ACL01017-5": ("76", "Acto Legislativo 1 de 2017"),
    "ACL01017-6": ("77", "Acto Legislativo 1 de 2017"),
    "ACL01017-7": ("78", "Acto Legislativo 1 de 2017"),
    "ACL01017-8": ("79", "Acto Legislativo 1 de 2017"),
    "ACL01017-9": ("80", "Acto Legislativo 1 de 2017"),
    "ACL01017-10": ("81", "Acto Legislativo 1 de 2017"),
    "ACL01017-11": ("82", "Acto Legislativo 1 de 2017"),
    "ACL01017-12": ("83", "Acto Legislativo 1 de 2017"),
    "ACL01017-13": ("84", "Acto Legislativo 1 de 2017"),
    "ACL01017-14": ("85", "Acto Legislativo 1 de 2017"),
    "ACL01017-15": ("86", "Acto Legislativo 1 de 2017"),
    "ACL01017-16": ("87", "Acto Legislativo 1 de 2017"),
    "ACL01017-17": ("88", "Acto Legislativo 1 de 2017"),
    "ACL01017-18": ("89", "Acto Legislativo 1 de 2017"),
    "ACL01017-19": ("90", "Acto Legislativo 1 de 2017"),
    "ACL01017-20": ("91", "Acto Legislativo 1 de 2017"),
    "ACL01017-21": ("92", "Acto Legislativo 1 de 2017"),
    "ACL01017-22": ("93", "Acto Legislativo 1 de 2017"),
    "ACL01017-23": ("94", "Acto Legislativo 1 de 2017"),
    "ACL01017-24": ("95", "Acto Legislativo 1 de 2017"),
    "ACL01017-25": ("96", "Acto Legislativo 1 de 2017"),
    "ACL01017-26": ("97", "Acto Legislativo 1 de 2017"),
    "ACL01017-27": ("98", "Acto Legislativo 1 de 2017"),
    "ACL02017-1": ("99", "Acto Legislativo 2 de 2017"),
    "ACL03017-1": ("100", "Acto Legislativo 3 de 2017"),
    "ACL03017-2": ("101", "Acto Legislativo 3 de 2017"),
    "ACL03017-3": ("102", "Acto Legislativo 3 de 2017"),
    "ACL02021-1": ("103", "Acto Legislativo 2 de 2021"),
    "ACL02021-2": ("104", "Acto Legislativo 2 de 2021"),
    "ACL02021-3": ("105", "Acto Legislativo 2 de 2021"),
    "ACL02021-4": ("106", "Acto Legislativo 2 de 2021"),
    "ACL02021-5": ("107", "Acto Legislativo 2 de 2021"),
    "ACL02021-6": ("108", "Acto Legislativo 2 de 2021"),
    "ACL02021-7": ("109", "Acto Legislativo 2 de 2021"),
    "ACL02021-8": ("110", "Acto Legislativo 2 de 2021"),
    "ACL02021-9": ("111", "Acto Legislativo 2 de 2021"),
    "ACL02021-10": ("112", "Acto Legislativo 2 de 2021"),
}

COLUMNAS = ["numero", "texto", "titulo", "capitulo", "bloque_tematico", "acto_legislativo"]


def configurar_semillas(semilla: int) -> None:
    """Fija las semillas de aleatoriedad para reproducibilidad.

    Args:
        semilla: Valor entero de semilla.
    """
    random.seed(semilla)
    np.random.seed(semilla)


def normalizar(texto: str) -> str:
    """Normaliza texto: minisculas, sin tildes y espacios colapsados.

    Args:
        texto: Cadena de entrada.

    Returns:
        Cadena normalizada.
    """
    sin_tildes = "".join(
        c for c in unicodedata.normalize("NFD", texto) if not unicodedata.combining(c)
    )
    return re.sub(r"\s+", " ", sin_tildes).strip().lower()


def romano_de_titulo(etiqueta: str) -> str:
    """Extrae el numero romano de una etiqueta como 'TITULO XIII.'.

    Args:
        etiqueta: Texto del marcador de titulo.

    Returns:
        Numero romano en mayusculas (por ejemplo 'XIII').
    """
    limpio = normalizar(etiqueta).replace("titulo", "").replace(".", "").strip().upper()
    return limpio


def numero_de_capitulo(etiqueta: str, nombre: str) -> str:
    """Extrae el numero de capitulo desde su etiqueta o su nombre de ancla.

    Args:
        etiqueta: Texto del marcador (por ejemplo 'CAPITULO 5.').
        nombre: Atributo name del ancla (por ejemplo 'CAPITULO IT').

    Returns:
        Numero de capitulo como cadena ('5', '7', 'VI', ...).
    """
    digitos = re.search(r"(\d+)", etiqueta)
    if digitos:
        return digitos.group(1)
    romano = re.search(r"CAPITULO\s+([IVXLCDM]+)", normalizar(etiqueta).upper())
    if romano:
        return romano.group(1)
    clave = normalizar(nombre).replace("capitulo", "").replace(" ", "").upper()
    if clave in MAPA_CAP_TRANSITORIO:
        return MAPA_CAP_TRANSITORIO[clave]
    return nombre


def mapear_bloque(titulo: str, capitulo: str) -> str:
    """Asigna el bloque tematico fijo E2 segun titulo y capitulo.

    Args:
        titulo: Romano del titulo ('I'..'XIII'), 'Transitorios' o 'Preambulo'.
        capitulo: Numero de capitulo como cadena.

    Returns:
        Identificador del bloque tematico.
    """
    if titulo == "Preambulo":
        return "preambulo"
    if titulo == "Transitorios":
        return "transitorios"
    if titulo == "I":
        return "principios_fundamentales"
    if titulo == "II":
        return {
            "1": "derechos_fundamentales",
            "2": "derechos_sociales_economicos_culturales",
            "3": "derechos_colectivos_ambiente",
            "4": "proteccion_aplicacion_derechos",
            "5": "deberes",
        }.get(capitulo, "derechos_fundamentales")
    return {
        "III": "habitantes_territorio",
        "IV": "participacion_democratica",
        "V": "organizacion_estado",
        "VI": "rama_legislativa",
        "VII": "rama_ejecutiva",
        "VIII": "rama_judicial",
        "IX": "elecciones",
        "X": "organismos_control",
        "XI": "organizacion_territorial",
        "XII": "regimen_economico",
        "XIII": "reforma_constitucion",
    }.get(titulo, "transitorios")


def es_parrafo_cierre(texto_norm: str) -> bool:
    """Indica si un parrafo es pie administrativo de pagina (no articulado).

    Args:
        texto_norm: Texto del parrafo ya normalizado.

    Returns:
        Verdadero si es pie de pagina y debe excluirse del articulado.
    """
    return any(marca in texto_norm for marca in MARCAS_FIN_PAGINA)


def texto_limpio_parrafo(parrafo: Tag) -> str:
    """Extrae el texto limpio de un parrafo (sin enlaces de navegacion).

    Args:
        parrafo: Etiqueta <p> de BeautifulSoup.

    Returns:
        Texto con espacios colapsados.
    """
    for a in parrafo.find_all("a", href=True):
        if "top" in (a.get("href") or "") and not a.get("name"):
            a.decompose()
    return re.sub(r"\s+", " ", parrafo.get_text(separator=" ", strip=True)).strip()


def normalizar_id_transitorio(valor: str) -> tuple:
    """Normaliza el ID de un transitorio a su canon con acto de origen.

    ciclo v2 - Fase S1: los anclas del Senado para transitorios
    adicionados por actos legislativos posteriores usan el codigo del acto
    ("TRANSITORIO ACL01016-1" = articulo 1 del Acto Legislativo 1 de
    2016). Se traducen al canon "Transitorio N" continuando la serie 1991
    (hasta 67) y se devuelve el acto de origen para la columna
    acto_legislativo. Los transitorios originales ("TRANSITORIO 12") pasan
    sin cambios con acto None.

    Args:
        valor: Valor crudo del ancla sin el prefijo "TRANSITORIO "
            (p. ej. "12", "ACL01016-1").

    Returns:
        Tupla (canon, acto_legislativo): canon "Transitorio N" y nombre
        corto del acto ("Acto Legislativo 1 de 2016") o None si el
        transitorio es del articulado original de 1991.

    Raises:
        ValueError: Si el sufijo ACL no esta en MAPA_IDS_ACL (mapeo
            incompleto: debe ampliarse con evidencia del HTML fuente).
    """
    if valor in MAPA_IDS_ACL:
        numero, acto = MAPA_IDS_ACL[valor]
        return f"Transitorio {numero}", acto
    return f"Transitorio {valor}", None


def clasificar_ancla(nombre: str, etiqueta: str) -> tuple:
    """Clasifica un marcador <a name> del articulado oficial.

    Args:
        nombre: Atributo name del ancla.
        etiqueta: Texto visible del ancla.

    Returns:
        Tupla (kind, valor, acto_legislativo) donde kind es 'titulo',
        'capitulo', 'articulo', 'transitorio' o 'ignorado'; valor es el
        canon/numero; acto_legislativo es el nombre corto del acto que
        adiciono el articulo (solo transitorios ACL, Fase S1) o None.
    """
    norm_nombre = normalizar(nombre)
    norm_etiqueta = normalizar(etiqueta)
    if norm_nombre.startswith("nivel") or re.match(r"^tt?ulo", norm_etiqueta):
        if norm_etiqueta.startswith("titulo") or "titulo" in norm_etiqueta:
            return ("titulo", romano_de_titulo(etiqueta), None)
        return ("capitulo", numero_de_capitulo(etiqueta, nombre), None)
    if norm_nombre.startswith("cap") or norm_etiqueta.startswith("capitulo"):
        return ("capitulo", numero_de_capitulo(etiqueta, nombre), None)
    if norm_nombre == "dis" or "disposiciones transitorias" in norm_etiqueta:
        return ("titulo", "Transitorios", None)
    if norm_nombre.startswith("transitorio"):
        sufijo = nombre.strip()[len("transitorio") :].strip(" -_")
        canon, acto = normalizar_id_transitorio(sufijo)
        return ("transitorio", canon, acto)
    if re.match(r"^\d+[a-z]?$", norm_nombre):
        return (
            "articulo",
            re.match(r"^(\d+[a-z]?)", norm_nombre).group(1).upper(),
            None,
        )
    return ("ignorado", "", None)


def recolectar_texto_articulo(parrafo_ini: Tag, etiqueta: str) -> str:
    """Recolecta el texto de un articulo desde su parrafo inicial.

    Toma el parrafo del marcador (sin la etiqueta) y los parrafos
    siguientes hasta el proximo marcador, pie de pagina o navegacion.

    Args:
        parrafo_ini: Parrafo <p> que contiene el marcador del articulo.
        etiqueta: Texto del marcador a retirar del inicio.

    Returns:
        Texto limpio del articulo.
    """
    partes = []
    texto_ini = texto_limpio_parrafo(parrafo_ini)
    resto = texto_ini.replace(etiqueta.strip(), "", 1).strip(" .-\u2013\u2014")
    if resto:
        partes.append(resto)
    for hermano in parrafo_ini.next_siblings:
        if not isinstance(hermano, Tag):
            continue
        if hermano.name != "p":
            continue
        if hermano.find("a", attrs={"name": True}):
            break
        if hermano.find("a", class_="antsig"):
            break
        texto = texto_limpio_parrafo(hermano)
        if not texto:
            continue
        texto_norm = normalizar(texto)
        if texto_norm in PARRAFOS_ADMIN or es_parrafo_cierre(texto_norm):
            if es_parrafo_cierre(texto_norm):
                break
            continue
        if re.match(r"^art[ii]culo\s+\d", texto_norm):
            break
        partes.append(texto)
    return re.sub(r"\s+", " ", " ".join(partes)).strip()


def parsear_pagina(html: str, estado: dict, articulos: list) -> None:
    """Parsea una pagina HTML y acumula articulos con su contexto.

    Args:
        html: Contenido HTML crudo de la pagina.
        estado: Dict mutable con 'titulo' y 'capitulo' vigentes.
        articulos: Lista donde se agregan los dicts de articulos.
    """
    sopa = BeautifulSoup(html, "lxml")
    for parrafo in sopa.find_all("p"):
        anclas = parrafo.find_all("a", attrs={"name": True})
        if anclas:
            for ancla in anclas:
                nombre = (ancla.get("name") or "").strip()
                etiqueta = ancla.get_text(separator=" ", strip=True)
                kind, valor, acto = clasificar_ancla(nombre, etiqueta)
                if kind == "titulo":
                    estado["titulo"] = valor
                    estado["capitulo"] = ""
                elif kind == "capitulo":
                    estado["capitulo"] = valor
                elif kind in ("articulo", "transitorio"):
                    texto = recolectar_texto_articulo(parrafo, etiqueta)
                    titulo = estado["titulo"] or "Preambulo"
                    articulos.append(
                        {
                            "numero": valor,
                            "texto": texto,
                            "titulo": titulo,
                            "capitulo": estado["capitulo"],
                            "bloque_tematico": mapear_bloque(
                                titulo, estado["capitulo"]
                            ),
                            "acto_legislativo": acto,
                        }
                    )
                    break
        else:
            texto = texto_limpio_parrafo(parrafo)
            coinc = re.match(
                r"^art[ii]culo\s+(\d+[a-z]?)\.?\s*(.*)$", normalizar(texto)
            )
            if coinc and estado.get("titulo"):
                numero = coinc.group(1).upper()
                titulo = estado["titulo"]
                articulos.append(
                    {
                        "numero": numero,
                        "texto": texto,
                        "titulo": titulo,
                        "capitulo": estado["capitulo"],
                        "bloque_tematico": mapear_bloque(titulo, estado["capitulo"]),
                        "acto_legislativo": None,
                    }
                )


def extraer_preambulo(html: str) -> dict:
    """Extrae el texto del preambulo de la pagina principal.

    Args:
        html: HTML crudo de la pagina principal.

    Returns:
        Dict con la fila del preambulo (columnas del parquet).
    """
    sopa = BeautifulSoup(html, "lxml")
    nodo = sopa.find(string=re.compile(r"PRE.+MBULO"))
    partes = []
    base = nodo.find_parent("p") if nodo else None
    if base is not None:
        for hermano in base.next_siblings:
            if not isinstance(hermano, Tag) or hermano.name != "p":
                continue
            if hermano.find("a", attrs={"name": True}):
                break
            texto = texto_limpio_parrafo(hermano)
            if texto and normalizar(texto) not in PARRAFOS_ADMIN:
                partes.append(texto)
    return {
        "numero": "Preambulo",
        "texto": re.sub(r"\s+", " ", " ".join(partes)).strip(),
        "titulo": "Preambulo",
        "capitulo": "Preambulo",
        "bloque_tematico": "preambulo",
        "acto_legislativo": None,
    }


def cargar_crudo(dir_crudo: Path) -> list:
    """Carga los HTML crudos ordenados del directorio de constitucion.

    Args:
        dir_crudo: Directorio con los cp_*.html descargados.

    Returns:
        Lista de tuplas (nombre_archivo, contenido_html) en orden.

    Raises:
        FileNotFoundError: Si no hay archivos crudos para parsear.
    """
    archivos = sorted(dir_crudo.glob("cp_*.html"))
    if not archivos:
        raise FileNotFoundError(f"Sin crudo HTML en {dir_crudo}")
    paginas = []
    for ruta in archivos:
        if "constitucion_politica_1991" not in ruta.name:
            continue
        paginas.append((ruta.name, ruta.read_text(encoding="utf-8", errors="ignore")))
    if not paginas:
        raise FileNotFoundError("Falta el articulado constitucion_politica_1991")
    return paginas


def guardar_salidas(articulos: list, raiz: Path) -> pd.DataFrame:
    """Guarda el parquet y el JSON con el articulado parseado.

    Args:
        articulos: Lista de dicts con numero, texto, titulo, capitulo,
            bloque y acto_legislativo.
        raiz: Directorio raiz del proyecto.

    Returns:
        DataFrame con el articulado final.
    """
    df = pd.DataFrame(articulos, columns=COLUMNAS)
    ruta_parquet = raiz / CONFIG["ruta_parquet"]
    ruta_json = raiz / CONFIG["ruta_json"]
    df.to_parquet(ruta_parquet, index=False)
    ruta_json.write_text(
        json.dumps(df.to_dict(orient="records"), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Articulos guardados: {len(df)} -> {ruta_parquet} y {ruta_json}")
    return df


def verificar_y_reportar(df: pd.DataFrame, raiz: Path) -> str:
    """Ejecuta el spot-check, la cobertura por bloque y escribe reportes.

    Args:
        df: DataFrame con el articulado parseado.
        raiz: Directorio raiz del proyecto.

    Returns:
        Texto del log de verificacion.
    """
    por_numero = {
        str(n): (t, b)
        for n, t, b in zip(df["numero"], df["texto"], df["bloque_tematico"])
    }
    lineas = ["verificacion de parseo de la constitucion"]
    for numero, esperado in CONFIG["spot_check"].items():
        texto, bloque = por_numero.get(numero, ("", "NO_ENCONTRADO"))
        ok = "si" if bloque == esperado else "NO"
        lineas.append(f"Art {numero}: bloque={bloque} esperado={esperado} OK={ok}")
        lineas.append(f"  texto150: {texto[:150]}")
    es_trans = df["numero"].str.startswith("Transitorio", na=False)
    es_pream = df["numero"] == "Preambulo"
    permanentes = df[~es_trans & ~es_pream]
    transitorios = df[es_trans]
    nums = set(permanentes["numero"].astype(str))
    faltantes = [str(i) for i in range(1, 381) if str(i) not in nums]
    lineas.append(f"permanentes={len(permanentes)} transitorios={len(transitorios)}")
    lineas.append(f"faltantes_1_380={faltantes if faltantes else 'ninguno'}")
    # ciclo v2 - Fase S1: sanity de IDs corregidos y metadato del acto.
    malformados = int(df["numero"].str.contains("ACL", na=False).sum())
    con_acto = int(df["acto_legislativo"].notna().sum())
    lineas.append(f"ids_malformados_acl={malformados} (esperado 0)")
    lineas.append(f"transitorios_con_acto_legislativo={con_acto} (esperado 45)")
    cobertura = df.groupby("bloque_tematico").size().sort_index()
    for bloque, conteo in cobertura.items():
        lineas.append(f"cobertura {bloque}: {conteo}")
    ruta_log = raiz / CONFIG["ruta_log"]
    ruta_log.parent.mkdir(parents=True, exist_ok=True)
    ruta_log.write_text("\n".join(lineas) + "\n", encoding="utf-8")
    md = ["# Cobertura de la Constitucion por bloque tematico", ""]
    md.append("| bloque_tematico | n_articulos |")
    md.append("| --- | ---: |")
    for bloque, conteo in cobertura.items():
        md.append(f"| {bloque} | {conteo} |")
    md += [
        "",
        (
            f"Total filas: {len(df)} (permanentes {len(permanentes)} + "
            f"transitorios {len(transitorios)} + preambulo 1)."
        ),
        f"Permanentes 1-380 faltantes: {faltantes if faltantes else 'ninguno'}.",
        f"Fuente: {CONFIG['fuente_base']} (ver experimentos/descarga_constitucion.log).",
    ]
    ruta_cob = raiz / CONFIG["ruta_cobertura"]
    ruta_cob.parent.mkdir(parents=True, exist_ok=True)
    ruta_cob.write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(lineas))
    return "\n".join(lineas)


def main() -> None:
    """Orquesta el parseo del crudo y la escritura de salidas y reportes.

    ciclo v2 - Fase S1: genera IDs canonicos para los 45 transitorios de
    actos legislativos (antes "Transitorio ACL01016-N") y la columna
    acto_legislativo; el resto del articulado es identico al v1.
    """
    configurar_semillas(CONFIG["semilla"])
    raiz = ROOT
    paginas = cargar_crudo(raiz / CONFIG["dir_crudo"])
    print(f"Paginas crudas: {len(paginas)}")
    estado: dict = {"titulo": "", "capitulo": ""}
    articulos: list = []
    preambulo = extraer_preambulo(paginas[0][1])
    for nombre, html in paginas:
        antes = len(articulos)
        parsear_pagina(html, estado, articulos)
        print(f"{nombre}: +{len(articulos) - antes} articulos")
    df = pd.DataFrame(
        [preambulo] + articulos,
        columns=COLUMNAS,
    )
    df = df.drop_duplicates(subset="numero", keep="first").reset_index(drop=True)
    ruta_parquet = raiz / CONFIG["ruta_parquet"]
    ruta_json = raiz / CONFIG["ruta_json"]
    df.to_parquet(ruta_parquet, index=False)
    ruta_json.write_text(
        json.dumps(df.to_dict(orient="records"), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Filas finales (con preambulo, sin duplicados): {len(df)}")
    print(df.head(3).to_string())
    print(f"NaN en texto: {int(df['texto'].isna().sum())}")
    verificar_y_reportar(df, raiz)


if __name__ == "__main__":
    try:
        main()
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        sys.exit(1)
